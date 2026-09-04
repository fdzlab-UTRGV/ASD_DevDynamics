# Developmental Dynamics of Phenotypic Architecture in ASD

**Fernandez, A. — Cell Reports Medicine**

This Dash application reproduces all analyses reported in the paper using
SPARK phenotypic data (pairwise analytic Ns ranged 1,842–105,261).

> **Note:** This repository contains the analysis application only. The
> figure-generation script (`figures_cellpress`) is maintained separately and
> is available from the lead contact on request. The figure references below
> map each tab to the analyses that underlie the corresponding paper figures.

---

## Tabs and Analysis Mapping

| Tab | Analyses (paper figure) |
|-----|-------------------|
| **Correlations** | Figure 1 — √ΔR² effect-size matrix; CBCL vs. ADOS distributions |
| **√ΔR² Analysis** | Figure 2 — hubness index + PCA; Figure 3A-B — split-half reproducibility + bootstrap; Figure 5 — anxiety-adjustment sensitivity |
| **Age-Stratified Hubs** | Figure 4 — developmental hub reorganization across lifespan |
| **Dev. Coupling** | Figure 6 — domain × age-band psychopathology coupling (ANOVA, Tukey HSD, GLM, mixed models, logistic) |
| **Ridge Regression** | Figure 3C — out-of-sample generalization (SSC holdout) |
| **Replication Cohort** | No dedicated figure — runs the full √ΔR² analytic suite (hubness, PCA, suppressor, age-stratified hubs, domain composites) on an independently uploaded cohort (e.g. SSC) |

---

## Setup

### Requirements

- Python 3.10+
- Dependencies in `requirements.txt`

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:8050 in your browser.

### Using the setup script

Double-click `setup.sh` or run:

```bash
bash setup.sh
```

The script finds the app zip in `~/Downloads`, extracts it, creates a venv,
installs dependencies, and opens the app in Chrome.

---

## Data Upload

### Main SPARK instruments (sidebar)

Upload each instrument's CSV or XLSX file from SFARI Base:

| Slot | File | Contents |
|------|------|----------|
| DCDQ | `dcdq_*.csv` | Developmental Coordination Disorder Questionnaire |
| RBS-R | `rbs_*.csv` | Repetitive Behavior Scale – Revised |
| SCQ | `scq_*.csv` | Social Communication Questionnaire |
| ADOS | `ados_*.csv` | ADOS Calibrated Severity Scores |
| CBCL | `cbcl_*.csv` | Child Behavior Checklist |
| Covariates | `core_descriptive_variables.csv` | Age, sex, nonverbal IQ |

Multiple files per instrument are supported; they are merged automatically.

### SSC holdout (Ridge Regression tab)

Upload the SSC holdout cohort on the **Ridge Regression** tab.
This file is kept separate from the main SPARK data. It must contain the
same behavioral predictor columns used in the main analysis plus CBCL outcomes.

### Independent replication cohort (Replication Cohort tab)

Upload an independent cohort's raw instrument files (DCDQ, RBS-R, SCQ, CBCL)
plus one descriptive/covariate file containing sex, age, and nonverbal IQ on
the **Replication Cohort** tab. Column names are auto-detected; any unmatched
covariates can be hand-mapped via the UI before running. Because √ΔR² is
scale-free, the cohort does not need to share a measurement scale with
discovery.

---

## Statistical Methods

All methods follow the paper exactly:

- **√ΔR²**: signed semi-partial correlation controlling for age, sex, and nonverbal IQ
- **Hubness Index**: Σ|√ΔR²| across FDR-significant outcomes (Benjamini-Hochberg)
- **PCA**: applied to the column-centered, column-scaled 10×9 √ΔR² matrix
- **Split-half**: stratified by sex × age quartile; concordance r and hub rank ρ
- **Bootstrap**: 50 subsamples at p ∈ {0.10, …, 0.70} dropout; Spearman ρ vs. full sample
- **Ridge regression**: α by 5-fold CV in discovery; permutation p-values (N=1,000) in holdout
- **Developmental coupling**: √ΔR² per domain × age-band; weighted ANOVA, Tukey HSD,
  Bayesian cell model, population GLM, mixed-effects models, logistic regression

---

## Data Availability

SPARK data are available through SFARI Base (sfari.org/resource/spark) subject
to a data use agreement. Analysis code is deposited at
https://github.com/fdzlab-UTRGV/ASD_DevDynamics.

Lead contact: Alejandra Fernandez (alejandra.fernandez@utrgv.edu)
