"""
Download the raw data for the project from the BV-BRC public API
(Bacterial and Viral Bioinformatics Resource Center, https://www.bv-brc.org).

Files written to data/raw/:
  amr_phenotypes.csv   lab-measured ciprofloxacin AST results for E. coli (genome_amr)
  genomes.csv          assembly / isolate metadata for those genomes (genome)
  amr_genes.csv        NDARO-curated AMR genes detected in each genome (sp_gene)
  qrdr_features.csv    gyrA / parC / parE protein features per genome (genome_feature)
  qrdr_sequences.csv   unique protein sequences for those features (feature_sequence)
  reference_qrdr.csv   gyrA / parC / parE from E. coli K-12 MG1655 (genome 511145.12)

Every step is cached, so the script can be re-run after a network failure and
continues where it stopped.  Usage:  python scripts/download_bvbrc.py
"""
import os
import sys
import time
import json
import urllib.parse

import pandas as pd
import requests

API = "https://www.bv-brc.org/api"
HEADERS = {"Accept": "application/json", "Content-Type": "application/rqlquery+x-www-form-urlencoded"}
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")

QRDR_PRODUCTS = {
    "DNA gyrase subunit A (EC 5.99.1.3)": "gyrA",
    "DNA topoisomerase IV subunit A (EC 5.99.1.3)": "parC",
    "DNA topoisomerase IV subunit B (EC 5.99.1.3)": "parE",
}
REFERENCE_GENOME = "511145.12"  # E. coli str. K-12 substr. MG1655


def query(endpoint, rql, retries=8, timeout=180):
    """POST an RQL query (POST avoids URL-length limits) and return a list of dicts."""
    url = f"{API}/{endpoint}/"
    for attempt in range(retries):
        try:
            r = requests.post(url, data=rql, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:  # network errors are common; back off and retry
            wait = min(60, 5 * 2 ** attempt)
            print(f"  [{endpoint}] attempt {attempt + 1} failed ({e}); retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"query to {endpoint} failed after {retries} attempts")


def quote(v):
    return urllib.parse.quote(str(v), safe="")


def qstr(v):
    """Quoted string literal for RQL values that contain spaces."""
    return "%22" + quote(v) + "%22"


def in_list(field, values):
    return f"in({field},({','.join(quote(v) for v in values)}))"


def batched(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def cached(name):
    path = os.path.join(OUT, name)
    return path if os.path.exists(path) else None


def fetch_phenotypes():
    path = os.path.join(OUT, "amr_phenotypes.csv")
    if cached("amr_phenotypes.csv"):
        return pd.read_csv(path, dtype={"genome_id": str})
    fields = ["genome_id", "genome_name", "antibiotic", "resistant_phenotype", "measurement",
              "measurement_sign", "measurement_value", "measurement_unit", "laboratory_typing_method",
              "testing_standard", "testing_standard_year", "evidence", "pmid"]
    rows, start, page = [], 0, 2500
    while True:
        rql = (f"and(eq(taxon_id,562),eq(antibiotic,ciprofloxacin),eq(evidence,{qstr('Laboratory Method')}))"
               f"&select({','.join(fields)})&sort(+genome_id)&limit({page},{start})")
        batch = query("genome_amr", rql)
        rows += batch
        print(f"phenotypes: {len(rows)} records", flush=True)
        if len(batch) < page:
            break
        start += page
    df = pd.DataFrame(rows)
    df["pmid"] = df["pmid"].apply(lambda x: ";".join(map(str, x)) if isinstance(x, list) else x)
    df.to_csv(path, index=False)
    return df


def fetch_by_genome(endpoint, filename, genome_ids, extra_filter, fields, batch_size, per_genome_limit):
    """Query an endpoint for a batch of genomes at a time, caching partial progress."""
    path = os.path.join(OUT, filename)
    part = path + ".partial.jsonl"
    done = set()
    if os.path.exists(part):
        with open(part) as fh:
            for line in fh:
                done.add(json.loads(line)["_batch"])
    ids = sorted(genome_ids)
    batches = list(batched(ids, batch_size))
    with open(part, "a") as fh:
        for i, b in enumerate(batches):
            key = b[0]
            if key in done:
                continue
            filt = in_list("genome_id", b)
            if extra_filter:
                filt = f"and({filt},{extra_filter})"
            rql = f"{filt}&select({','.join(fields)})&limit({batch_size * per_genome_limit})"
            res = query(endpoint, rql)
            for row in res:
                row["_batch"] = key
                fh.write(json.dumps(row) + "\n")
            if not res:  # mark empty batches as done too
                fh.write(json.dumps({"_batch": key}) + "\n")
            fh.flush()
            print(f"{endpoint}: batch {i + 1}/{len(batches)} ({len(res)} rows)", flush=True)
    rows = [json.loads(l) for l in open(part)]
    df = pd.DataFrame(rows).drop(columns="_batch")
    df = df.dropna(subset=["genome_id"]) if "genome_id" in df else df
    df.to_csv(path, index=False)
    os.remove(part)
    return df


def main():
    os.makedirs(OUT, exist_ok=True)

    pheno = fetch_phenotypes()
    gids = set(pheno["genome_id"].astype(str))
    print(f"{len(pheno)} phenotype records, {len(gids)} genomes")

    if not cached("genomes.csv"):
        fetch_by_genome(
            "genome", "genomes.csv", gids, None,
            ["genome_id", "genome_name", "genome_status", "genome_quality", "contigs", "genome_length",
             "gc_content", "checkm_completeness", "checkm_contamination", "mlst", "isolation_country",
             "isolation_source", "host_common_name", "collection_year", "bioproject_accession",
             "biosample_accession", "assembly_accession", "public"],
            batch_size=200, per_genome_limit=1)

    if not cached("amr_genes.csv"):
        fetch_by_genome(
            "sp_gene", "amr_genes.csv", gids,
            f"and(eq(property,{qstr('Antibiotic Resistance')}),eq(source,NDARO))",
            ["genome_id", "patric_id", "gene", "product", "source", "source_id", "identity",
             "query_coverage", "subject_coverage", "evidence", "classification"],
            batch_size=50, per_genome_limit=60)

    # Exact-match filters on product names containing parentheses are rejected by the API,
    # so fetch gyrase/topoisomerase features by keyword and keep exact products client-side.
    kw_filter = "or(keyword(gyrase),keyword(topoisomerase))"
    if not cached("qrdr_features.csv"):
        fetch_by_genome(
            "genome_feature", "qrdr_features.csv", gids,
            f"and(eq(annotation,PATRIC),eq(feature_type,CDS),{kw_filter})",
            ["genome_id", "patric_id", "product", "aa_length", "aa_sequence_md5", "accession", "start", "end"],
            batch_size=100, per_genome_limit=15)
        qf = pd.read_csv(os.path.join(OUT, "qrdr_features.csv"), dtype={"genome_id": str})
        qf[qf["product"].isin(QRDR_PRODUCTS)].to_csv(os.path.join(OUT, "qrdr_features.csv"), index=False)

    if not cached("reference_qrdr.csv"):
        ref = query("genome_feature",
                    f"and(eq(genome_id,{REFERENCE_GENOME}),eq(annotation,PATRIC),{kw_filter})"
                    "&select(genome_id,patric_id,product,aa_length,aa_sequence_md5)&limit(50)")
        ref = pd.DataFrame(ref)
        ref = ref[ref["product"].isin(QRDR_PRODUCTS)]
        seqs = query("feature_sequence",
                     in_list("md5", ref["aa_sequence_md5"].unique()) + "&select(md5,sequence)&limit(50)")
        ref = ref.merge(pd.DataFrame(seqs), left_on="aa_sequence_md5", right_on="md5").drop(columns="md5")
        ref["gene"] = ref["product"].map(QRDR_PRODUCTS)
        ref.to_csv(os.path.join(OUT, "reference_qrdr.csv"), index=False)

    if not cached("qrdr_sequences.csv"):
        feats = pd.read_csv(os.path.join(OUT, "qrdr_features.csv"), dtype={"genome_id": str})
        md5s = sorted(feats["aa_sequence_md5"].dropna().unique())
        rows = []
        for i, b in enumerate(batched(md5s, 100)):
            rows += query("feature_sequence", in_list("md5", b) + "&select(md5,sequence)&limit(200)")
            print(f"sequences: {len(rows)}/{len(md5s)}", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "qrdr_sequences.csv"), index=False)

    print("All downloads complete:", sorted(os.listdir(OUT)))


if __name__ == "__main__":
    sys.exit(main())
