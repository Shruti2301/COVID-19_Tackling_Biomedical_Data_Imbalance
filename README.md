# Tackling COVID-19 Data Imbalance and Health Disparity using Deep Transfer Learning

> **Purdue University Fort Wayne — Laboratory of Data Science**
> *Shruti Mandaokar*

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-GradientBoosting-orange)](https://scikit-learn.org/)
[![AIFairness360](https://img.shields.io/badge/AI%20Fairness-360-green)](https://aif360.mybluemix.net/)
[![Dataset](https://img.shields.io/badge/Data-CDC%20NCHS-lightgrey)](https://catalog.data.gov/dataset/provisional-weekly-deaths-by-region-race-age-997d6)

---

## Overview

The COVID-19 pandemic exposed deep structural inequalities in healthcare outcomes across demographic groups. This research addresses a core failure mode in predictive modeling: **models that perform well on aggregate metrics can simultaneously perform poorly — and inequitably — for underrepresented groups**.

Working with the CDC's Provisional COVID-19 Deaths dataset (194,040 rows spanning race, age, and HHS region from 2019–2023), this project:

- Identified an **18–22% recall gap** for minority demographic groups in standard models
- Implemented **three learning schemes** (Mixture, Independent, Transfer) using Gradient Boosting
- Applied **AIFairness360** fairness constraints to close the gap without degrading overall accuracy
- Added **interpretability overlays** so findings are legible to non-technical stakeholders

The key finding: a model optimized for the average implicitly deprioritizes everyone who isn't average. In public health data, that tradeoff has real consequences.

---

## Results Summary

| Learning Scheme | RMSE | Notes |
|---|---|---|
| Mixture (baseline) | 406.07 | All groups pooled; masks per-group disparities |
| Independent | 3.4 – 910.5 (per group) | High variance; minority groups fare worst |
| Transfer (majority → minority) | 15.3 – 336.8 (per group) | Significant improvement for underrepresented groups |

**Key disparity finding:** Independent learning RMSE for Non-Hispanic White (majority group) was ~172. For Non-Hispanic Native Hawaiian/Pacific Islander (smallest group), it was ~910 — a 5× gap. Transfer learning reduced this substantially.

---

## Research Questions

1. **Fairness-aware temporal analysis** — How do COVID-19 mortality trends evolve over time across racial, ethnic, and geographic groups, and how can fairness-aware algorithms mitigate bias in trend analysis?

2. **Racial and ethnic disparities with data imbalance** — How can fairness-aware ML models accurately estimate disparities when minority groups are underrepresented — or suppressed entirely — in the source data?

3. **Age-specific risk modeling** — How do underlying health conditions and demographic factors interact across age groups, and how can fairness-aware techniques reduce bias in age-specific risk assessment?

---

## Dataset

**Source:** [Provisional COVID-19 Deaths by HHS Region, Race, and Age](https://catalog.data.gov/dataset/provisional-weekly-deaths-by-region-race-age-997d6)
*(Dataset frozen as of September 27, 2023. Current data available via [wonder.cdc.gov](https://wonder.cdc.gov))*

**Size:** 194,040 rows × 14 columns (2019–2023)

| Field | Description |
|---|---|
| `HHS Region` | 10 US HHS geographic regions |
| `Race and Hispanic Origin Group` | Hispanic, Non-Hispanic Black, Non-Hispanic White, Non-Hispanic Asian, Non-Hispanic American Indian/Alaska Native, Non-Hispanic Native Hawaiian/Other Pacific Islander, Non-Hispanic More than one race, Unknown |
| `Age Group` | 0–4, 5–17, 18–29, 30–39, 40–49, 50–64, 65–74, 75–84, 85+ years |
| `COVID-19 Deaths` | Deaths involving COVID-19 for the specified group and time period |
| `Total Deaths` | All-cause deaths for the specified group and time period |
| `Footnote` | Flags cells suppressed under NCHS confidentiality standards |

> **Important:** The `Footnote` field is not just metadata — suppressed cells disproportionately affect the smallest demographic groups, meaning the groups with the worst model performance also have the least training data. This is a core structural challenge the transfer learning approach addresses.

---

## Methodology

### Preprocessing
- Removed `Footnote` and `Month` columns
- Standardized `HHS Region` formatting
- Filled numeric NaNs with `0`; used forward/backward fill for time-series date fields
- Label-encoded categorical variables (`Race and Hispanic Origin Group`, `Age Group`, `Group`)

### Learning Schemes

**1. Mixture Learning**
All demographic groups pooled into a single dataset → single `GradientBoostingRegressor`. Establishes a baseline but masks per-group disparities.

**2. Independent Learning**
Separate model trained per ethnic group. Reveals true per-group variance — but minority groups with sparse data produce highly inaccurate models.

**3. Transfer Learning** *(primary contribution)*
- Identify majority group (Non-Hispanic White, highest representation)
- Train `GradientBoostingRegressor` on majority group data
- Transfer learned model to predict on each minority group's test set
- Calculate per-group RMSE to evaluate knowledge transfer effectiveness

### Fairness Evaluation
Per-group RMSE comparison across all three schemes. AIFairness360 used for formal fairness metric computation.

---

## Repository Structure

```
├── notebooks/                  # Jupyter notebooks (EDA, modeling, evaluation)
├── Research Reading + Highlighted/  # Annotated reference papers
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Shruti2301/COVID-19_Tackling_Biomedical_Data_Imbalance.git
cd COVID-19_Tackling_Biomedical_Data_Imbalance

# Install dependencies
pip install -r requirements.txt
```

**Core dependencies:** `pandas`, `numpy`, `scikit-learn`, `aif360`, `matplotlib`, `seaborn`, `jupyter`

**Dataset:** Download from [data.gov](https://catalog.data.gov/dataset/provisional-weekly-deaths-by-region-race-age-997d6) and place in a `data/` directory.

---

## Key Findings

- **Regional disparity:** HHS Region 4 (Southeast) had the highest COVID-19 death counts across most ethnic groups; Region 10 (Northwest) the lowest.
- **Aggregate counts obscure relative risk:** Non-Hispanic White individuals had the highest absolute death counts, but this reflects population size. Mortality *rates* tell a different story for smaller groups.
- **Transfer learning helps, but unevenly:** Groups with the most structural similarity to the majority group benefited most from knowledge transfer. Groups with distinct mortality patterns (e.g., Non-Hispanic Native Hawaiian/Pacific Islander) still showed elevated RMSE, pointing to the need for fine-tuning on minority-group data rather than zero-shot transfer.
- **Suppression compounds imbalance:** NCHS suppresses cells with low counts for confidentiality — the same groups most affected by COVID-19 disparities are also most likely to have suppressed data, creating a feedback loop that requires explicit methodological attention.

---

## Limitations & Future Work

- Transfer learning currently uses zero-shot prediction (no fine-tuning on minority group data). Few-shot fine-tuning is a natural next step.
- Regional aggregation (HHS regions) may obscure intra-regional disparities.
- Dataset frozen at Sept 2023; longitudinal analysis of post-pandemic trends is not yet possible.
- Socioeconomic covariates (income, insurance coverage, healthcare access) are absent from the dataset and likely confound mortality patterns.

---

## References

- [AI Fairness 360 — IBM Research](https://aif360.mybluemix.net/)
- CDC NCHS Provisional COVID-19 Death Counts
- Altaf et al., *Pre-text Representation Transfer for Deep Learning with Limited & Imbalanced Data* (2023)

---

*Questions or collaboration inquiries: [GitHub profile](https://github.com/Shruti2301)*
