# Predicting ciprofloxacin resistance in *E. coli* from resistance genes and mutations

Computing Bioinformatics, Group 1 project: Kipkemboi Joy Chepchumba, Kristein Gichuhi Mwaura, Leriari Peter Tobiko, Oringe Emmanuel Magunga, Gichuru Victor Kimotho.

**Research question.** Can known resistance genes and mutations in *E. coli* genomes predict ciprofloxacin resistance, and how much better is the prediction than a naive baseline?

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/eoringe/Bioinformatics/blob/main/Ecoli_Ciprofloxacin_Resistance.ipynb)

## Main results

* **Data:** 11,220 quality-controlled *E. coli* genomes from BV-BRC with lab-measured ciprofloxacin phenotypes (4,528 resistant, 6,692 susceptible).
* **Associations:** the strongest are the classic quinolone-resistance-determining region (QRDR) mutations. *parC* S80I has an odds ratio of about 950, *gyrA* D87N about 540, *parC* E84V about 390 and *gyrA* S83L about 200. Of isolates with ≥ 3 QRDR mutations, 97 to 99% are resistant.
* **Prediction (cross-validation that holds out whole sequence types):**

| Model | Balanced accuracy | MCC | ROC-AUC |
|---|---|---|---|
| Majority-class baseline | 0.50 | 0.00 | 0.50 |
| Catalogue rule (≥ 1 known QRDR mutation or *qnr*/*aac(6')-Ib-cr*) | 0.90 | 0.79 | 0.90 |
| Random forest, genes only | 0.84 | 0.67 | 0.89 |
| Logistic regression, genes + mutations | **0.96** | **0.92** | **0.99** |

* **Interpretation:** resistance is driven by target-site mutations in DNA gyrase and topoisomerase IV. Many acquired genes that look predictive (*bla*CTX-M, *sul2*, *tet(A)*) are co-resistance passengers on multidrug-resistance plasmids and in the ST131 clone.

## Repository layout

| Path | Content |
|---|---|
| `Ecoli_Ciprofloxacin_Resistance.ipynb` | The analysis notebook (all 8 stages), executed with outputs saved |
| `data/raw/*.csv` | Cached BV-BRC downloads (phenotypes, genome metadata, AMR genes, GyrA/ParC/ParE sequences, K-12 reference) |
| `scripts/download_bvbrc.py` | Re-downloads all data from the BV-BRC API; it resumes after network failures |
| `scripts/notebook_source.py` | Plain-Python source of the notebook (percent-format cells) |
| `scripts/build_notebook.py` | Converts the source into the `.ipynb` |

## How to run

**Google Colab:** click the badge above and choose *Runtime → Run all*. The first cell installs Biopython and clones this repository to get the data. The full run takes a few minutes.

**Locally:**

```bash
pip install pandas numpy scipy statsmodels scikit-learn matplotlib seaborn networkx biopython jupyter
jupyter nbconvert --to notebook --execute --inplace Ecoli_Ciprofloxacin_Resistance.ipynb
```

To rebuild the data from scratch (takes about an hour), run `python scripts/download_bvbrc.py` after deleting `data/raw/`, or set `REDOWNLOAD = True` in the notebook.

## Data source

BV-BRC (https://www.bv-brc.org). Olson RD et al. *Introducing the Bacterial and Viral Bioinformatics Resource Center (BV-BRC): a resource combining PATRIC, IRD and ViPR.* Nucleic Acids Research 51:D678–D689 (2023). Data downloaded on 18 November 2026.
