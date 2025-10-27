<!-- BADGES_START -->

[🇨🇳 中文版](README.zh-CN.md) ｜ [⭐ Star](https://github.com/jia-ning-tech/lymphoma-ewarn/stargazers) ｜ [�� Issues](https://github.com/jia-ning-tech/lymphoma-ewarn/issues)

![GitHub Repo stars](https://img.shields.io/github/stars/jia-ning-tech/lymphoma-ewarn?style=flat)
![GitHub issues](https://img.shields.io/github/issues/jia-ning-tech/lymphoma-ewarn?style=flat)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Release](https://img.shields.io/github/v/release/jia-ning-tech/lymphoma-ewarn?display_name=tag&sort=semver)

<!-- BADGES_END -->

<!-- TOC_START -->

## Table of Contents

- [Table of Contents](#table-of-contents)
- [TL;DR — Highlights](#tldr-highlights)
- [1. Abstract](#1-abstract)
- [2. Methods](#2-methods)
  - [2.1 Cohort & Feature Windows](#21-cohort-feature-windows)
  - [2.2 Models & Training](#22-models-training)
  - [2.3 Evaluation Metrics](#23-evaluation-metrics)
- [3. Results](#3-results)
  - [3.1 Discrimination (Window-level, Test)](#31-discrimination-window-level-test)
  - [3.2 Calibration](#32-calibration)
  - [Decision Thresholds (auto-generated)](#decision-thresholds-auto-generated)
  - [3.3 Lead Time (Test)](#33-lead-time-test)
  - [3.4 Interpretability (SHAP)](#34-interpretability-shap)
  - [3.5 Error Analysis (Test Example)](#35-error-analysis-test-example)
  - [3.6 Ablation (24h, 5-fold, 600 trees)](#36-ablation-24h-5-fold-600-trees)
  - [Keep-only (use only one group)](#keep-only-use-only-one-group)
  - [Drop-one (remove one group from the full set)](#drop-one-remove-one-group-from-the-full-set)
- [Ablation Studies](#ablation-studies)
  - [Ablation (h=24)](#ablation-h24)
- [4. Repository Layout & Project Structure](#4-repository-layout-project-structure)
- [Project Structure](#project-structure)
  - [Key Directories (auto-generated)](#key-directories-auto-generated)
  - [Full Tree (auto-generated)](#full-tree-auto-generated)
- [5. How to Reproduce](#5-how-to-reproduce)
  - [5.1 Environment](#51-environment)
  - [5.2 End-to-End Steps](#52-end-to-end-steps)
- [6. Repository Guide](#6-repository-guide)
- [7. Roadmap](#7-roadmap)
- [8. Acknowledgements & Disclaimer](#8-acknowledgements-disclaimer)
- [9. Limitations](#9-limitations)
- [10. Ethics & Data Access](#10-ethics-data-access)
- [11. License & Citation](#11-license-citation)
- [Decision Curve Analysis (DCA)](#decision-curve-analysis-dca)
  - [Decision Curve Analysis — h=24h, split=val](#decision-curve-analysis-h24h-splitval)
  - [Decision Curve Analysis — h=24h, split=test](#decision-curve-analysis-h24h-splittest)
  - [Decision Curve Analysis — h=48h, split=val](#decision-curve-analysis-h48h-splitval)
  - [Decision Curve Analysis — h=48h, split=test](#decision-curve-analysis-h48h-splittest)

<!-- TOC_END -->
<!-- Badges / Shields -->

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" />
  <img alt="scikit-learn" src="https://img.shields.io/badge/scikit--learn-1.x-ff9900?logo=scikitlearn&logoColor=white" />
  <img alt="SHAP" src="https://img.shields.io/badge/Explainability-SHAP-8A2BE2" />
  <img alt="Calibration" src="https://img.shields.io/badge/Calibration-Isotonic%20%7C%20Sigmoid-2aa198" />
  <img alt="CI" src="https://img.shields.io/badge/Status-Research%20Prototype-lightgrey" />
  <img alt="PRs" src="https://img.shields.io/badge/PRs-welcome-brightgreen" />
  <img alt="Love" src="https://img.shields.io/badge/Made%20with-%E2%9D%A4%EF%B8%8F-red" />
  <img alt="License" src="https://img.shields.io/badge/License-MIT-informational" />
</p>

<h1 align="center">Lymphoma-EWARN: Early Warning for Deterioration in ICU Lymphoma Patients</h1>

<p align="center">
  <em>Window-level risk prediction with lead time, stay-level aggregation, post-hoc calibration, SHAP interpretability, error analysis, and ablation studies.</em><br/>
  <a href="README.zh-CN.md">中文说明（Chinese version）</a>
</p>

---

## TL;DR — Highlights

* **Two horizons:** 24h & 48h early-warning windows
* **Discrimination (test):** AUROC ≈ **0.79** (24h) / **0.81** (48h); AP ≈ **0.14** / **0.29**
* **Lead time (test):** median ≈ **39.83h (24h model)** / **34.35h (48h model)**
* **Post-hoc calibration:** Isotonic / Sigmoid; ECE tracked at window & stay levels
* **Interpretability:** SHAP global & individual explanations
* **Error analysis:** FP/FN cohorts + top-K windows for review
* **Ablation:** feature-group contributions (vitals / labs / vent / others)

---

## 1. Abstract

Timely detection of clinical deterioration among ICU lymphoma patients is challenging due to sparse, noisy, and heterogeneous signals. We develop **Lymphoma-EWARN**, a practical early-warning system that ingests engineered window-level features and outputs next-24h/48h risk scores. We evaluate both **window** and **stay** levels, quantify **lead time**, apply **post-hoc calibration**, provide **SHAP** explanations, perform **error analysis**, and conduct **ablation** on feature groups. Results indicate reliable discrimination and clinically meaningful lead time, paving the way for shadow deployment and standardized monitoring.

---

## 2. Methods

### 2.1 Cohort & Feature Windows

* Windows: **60,593** rows (`data_interim/trainset_hXX.parquet`)
* ICU stays: **721**
* Window positive rate (test): **2.72% (24h)** / **3.87% (48h)**
* Stay positive rate: **15.53%**
* Median windows per stay: **52** (p10–p90: 19–163)
* Time span per stay (median): **51h**

<details>
<summary>How these numbers were produced</summary>

```bash
python -m src.cli.cohort_stats --horizon 24
python -m src.cli.cohort_stats --horizon 48
# Outputs:
# outputs/reports/cohort_stats_h24_all.json
# outputs/reports/cohort_stats_h48_all.json
# outputs/reports/cohort_missingness_hXX_all.csv
# outputs/reports/cohort_numeric_summary_hXX_all.csv
```

</details>

### 2.2 Models & Training

* **Estimator**: `RandomForestClassifier` (e.g., `n_estimators=600`, `class_weight="balanced_subsample"`)
* **Preprocessing**: `SimpleImputer` (mean) within a sklearn `Pipeline`
* **Calibration**: Isotonic or Sigmoid (Platt), applied post-hoc
* **Evaluation splits**: `val`, `test` (window-level); aggregated to stay-level via **max probability** across windows per stay

### 2.3 Evaluation Metrics

* **Window-level**: AUROC, Average Precision (AP), Brier score, Expected Calibration Error (ECE)
* **Stay-level**: Precision / Recall / F1 @ chosen threshold (fixed or alert-rate target)
* **Lead time**: hours from **first alert** to **first event** per stay

---

## 3. Results

### 3.1 Discrimination (Window-level, Test)

| Horizon |      AUROC |         AP | Curves                                                                            |
| ------- | ---------: | ---------: | --------------------------------------------------------------------------------- |
| 24h     | **0.7903** | **0.1372** | ROC → `outputs/figures/roc_h24_test.png` · PR → `outputs/figures/pr_h24_test.png` |
| 48h     | **0.8121** | **0.2881** | ROC → `outputs/figures/roc_h48_test.png` · PR → `outputs/figures/pr_h48_test.png` |

```bash
python -m src.cli.plot_curves --horizon 24 --split test
python -m src.cli.plot_curves --horizon 48 --split test
```




<div align="center">
  <img src="docs/figures/roc_h24_test.png" alt="ROC 24h" width="45%"/>
  <img src="docs/figures/pr_h24_test.png"  alt="PR 24h"  width="45%"/><br/><br/>
  <img src="docs/figures/roc_h48_test.png" alt="ROC 48h" width="45%"/>
  <img src="docs/figures/pr_h48_test.png"  alt="PR 48h"  width="45%"/>
</div>


### 3.2 Calibration

**Window-level (test):**

| Horizon |      Brier |        ECE | Figures                                                                                                   |
| ------- | ---------: | ---------: | --------------------------------------------------------------------------------------------------------- |
| 24h     | **0.0497** | **0.0284** | `outputs/figures/calibration_h24_test_window.png`, `outputs/figures/calibration_hist_h24_test_window.png` |
| 48h     | **0.0747** | **0.0478** | `outputs/figures/calibration_h48_test_window.png`, `outputs/figures/calibration_hist_h48_test_window.png` |

**Stay-level (test):**

| Horizon |      Brier |        ECE | Notes             |
| ------- | ---------: | ---------: | ----------------- |
| 24h     | **0.1768** | **0.1506** | Max-prob per stay |
| 48h     | **0.1595** | **0.1350** | Max-prob per stay |

```bash
# Window-level
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform
# Stay-level
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform --stay_level
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform --stay_level
```

<!-- THRESHOLDS_START -->

### Decision Thresholds (auto-generated)

| Horizon | Method   | Chosen Threshold | Source JSON |
|:-------:|:--------:|:----------------:|:------------|
| 24h | isotonic | **0.0577** | `outputs/reports/posthoc_calibration_h24_isotonic.json` |
| 24h | sigmoid | **0.0369** | `outputs/reports/posthoc_calibration_h24_sigmoid.json` |
| 48h | isotonic | **0.0544** | `outputs/reports/posthoc_calibration_h48_isotonic.json` |
| 48h | sigmoid | **0.0355** | `outputs/reports/posthoc_calibration_h48_sigmoid.json` |

<!-- THRESHOLDS_END -->




### 3.3 Lead Time (Test)

| Horizon |  n | Mean (h) | Median (h) |  P10 |  P25 |   P75 |   P90 |   Max |
| ------- | -: | -------: | ---------: | ---: | ---: | ----: | ----: | ----: |
| 48h     | 13 |    30.91 |      34.35 | 0.84 | 6.98 | 52.25 | 57.50 | 72.25 |
| 24h     |  9 |    34.81 |      39.83 | 0.51 | 2.00 | 54.50 | 61.05 | 72.25 |

```bash
python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366
python -m src.cli.leadtime_plot --horizon 24 --split test --threshold 0.1205866239132141
# Figures:
# outputs/figures/leadtime_hist_hXX_test_thr*.png
# outputs/figures/leadtime_box_hXX_test_thr*.png
```

### 3.4 Interpretability (SHAP)

* Global: `outputs/figures/shap_global_beeswarm_h48_test.png`, `outputs/figures/shap_global_bar_h48_test.png`, CSV: `outputs/reports/shap_values_top_h48_test.csv`
* Individual top-K explanations are also exported (see CLI)

```bash
python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5
python -m src.cli.shap_explain --horizon 24 --split test --top_n 200 --top_k_individual 5
```

<!-- FI_START -->
## Feature importance

> We report permutation importance (mean ΔAUC over repeats). SHAP is currently unavailable due to model wrapper; will be added later.

### 预测提前窗 h=24 — validation

![Feature importance — h24 val](outputs/figures/fi_forest_h24_val.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.07928 | 0.00493 |
| `resp_rate__24h__count` | 0.00000 | 0.02508 | 0.00587 |
| `dbp__24h__count` | 0.00000 | 0.02164 | 0.00495 |
| `sbp__24h__count` | 0.00000 | 0.02109 | 0.00668 |
| `heart_rate__24h__count` | 0.00000 | 0.01949 | 0.00670 |
| `peep__6h__count` | 0.00000 | 0.01846 | 0.00923 |
| `hco3__24h__max` | 0.00000 | 0.01639 | 0.00088 |
| `hco3__24h__min` | 0.00000 | 0.01475 | 0.00270 |
| `spo2__24h__count` | 0.00000 | 0.01462 | 0.00264 |
| `hco3__24h__mean` | 0.00000 | 0.01456 | 0.00127 |
| `mbp__24h__count` | 0.00000 | 0.01282 | 0.00410 |
| `na__24h__mean` | 0.00000 | 0.01192 | 0.00538 |
| `na__24h__min` | 0.00000 | 0.01178 | 0.00284 |
| `temperature__24h__count` | 0.00000 | 0.01168 | 0.00628 |
| `cl__24h__mean` | 0.00000 | 0.01126 | 0.00388 |
| `cl__24h__max` | 0.00000 | 0.01015 | 0.00245 |
| `hco3__24h__std` | 0.00000 | 0.00990 | 0.00099 |
| `hco3__24h__last` | 0.00000 | 0.00983 | 0.00148 |
| `dbp__24h__max` | 0.00000 | 0.00959 | 0.00215 |
| `spo2__24h__mean` | 0.00000 | 0.00891 | 0.00449 |

### 预测提前窗 h=24 — test

![Feature importance — h24 test](outputs/figures/fi_forest_h24_test.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.04246 | 0.00154 |
| `cl__24h__mean` | 0.00000 | 0.00261 | 0.00034 |
| `peep__24h__std` | 0.00000 | 0.00246 | 0.00048 |
| `dbp__24h__mean` | 0.00000 | 0.00241 | 0.00116 |
| `cl__24h__min` | 0.00000 | 0.00207 | 0.00097 |
| `temperature__24h__mean` | 0.00000 | 0.00191 | 0.00035 |
| `heart_rate__6h__mean` | 0.00000 | 0.00173 | 0.00036 |
| `peep__24h__mean` | 0.00000 | 0.00168 | 0.00059 |
| `pt__24h__min` | 0.00000 | 0.00165 | 0.00040 |
| `lactate__24h__last` | 0.00000 | 0.00160 | 0.00070 |
| `spo2__24h__std` | 0.00000 | 0.00153 | 0.00046 |
| `na__6h__max` | 0.00000 | 0.00131 | 0.00012 |
| `peep__24h__max` | 0.00000 | 0.00130 | 0.00077 |
| `na__6h__mean` | 0.00000 | 0.00129 | 0.00017 |
| `heart_rate__24h__mean` | 0.00000 | 0.00120 | 0.00052 |
| `temperature__24h__max` | 0.00000 | 0.00113 | 0.00073 |
| `resp_rate__24h__std` | 0.00000 | 0.00112 | 0.00019 |
| `sbp__6h__min` | 0.00000 | 0.00111 | 0.00021 |
| `platelet__24h__mean` | 0.00000 | 0.00106 | 0.00090 |
| `pao2__24h__min` | 0.00000 | 0.00106 | 0.00018 |

### 预测提前窗 h=48 — validation

![Feature importance — h48 val](outputs/figures/fi_forest_h48_val.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.08343 | 0.00585 |
| `heart_rate__24h__count` | 0.00000 | 0.03088 | 0.00677 |
| `resp_rate__24h__count` | 0.00000 | 0.02665 | 0.00518 |
| `peep__6h__count` | 0.00000 | 0.02411 | 0.00649 |
| `sbp__24h__count` | 0.00000 | 0.02174 | 0.00532 |
| `ph__24h__min` | 0.00000 | 0.01863 | 0.00264 |
| `na__24h__mean` | 0.00000 | 0.01810 | 0.00643 |
| `spo2__24h__count` | 0.00000 | 0.01789 | 0.00408 |
| `dbp__24h__count` | 0.00000 | 0.01697 | 0.00680 |
| `mbp__24h__count` | 0.00000 | 0.01483 | 0.00472 |
| `ca__24h__max` | 0.00000 | 0.01478 | 0.00277 |
| `na__24h__min` | 0.00000 | 0.01420 | 0.00390 |
| `temperature__24h__count` | 0.00000 | 0.01020 | 0.00447 |
| `peep__24h__std` | 0.00000 | 0.00858 | 0.00394 |
| `hco3__24h__min` | 0.00000 | 0.00826 | 0.00059 |
| `spo2__24h__mean` | 0.00000 | 0.00803 | 0.00145 |
| `cl__24h__mean` | 0.00000 | 0.00791 | 0.00300 |
| `peep__24h__mean` | 0.00000 | 0.00710 | 0.00493 |
| `peep__24h__max` | 0.00000 | 0.00670 | 0.00362 |
| `cl__24h__min` | 0.00000 | 0.00649 | 0.00320 |

### 预测提前窗 h=48 — test

![Feature importance — h48 test](outputs/figures/fi_forest_h48_test.png)

| Feature | SHAP mean(|v|) | Perm mean | Perm std |
|---|---:|---:|---:|
| `peep__24h__count` | 0.00000 | 0.10095 | 0.00459 |
| `resp_rate__24h__max` | 0.00000 | 0.00641 | 0.00119 |
| `hgb__24h__min` | 0.00000 | 0.00563 | 0.00111 |
| `temperature__24h__count` | 0.00000 | 0.00520 | 0.00181 |
| `spo2__24h__mean` | 0.00000 | 0.00505 | 0.00183 |
| `platelet__24h__min` | 0.00000 | 0.00505 | 0.00126 |
| `bun__24h__max` | 0.00000 | 0.00483 | 0.00093 |
| `platelet__24h__max` | 0.00000 | 0.00421 | 0.00072 |
| `temperature__24h__mean` | 0.00000 | 0.00368 | 0.00065 |
| `peep__24h__std` | 0.00000 | 0.00358 | 0.00224 |
| `resp_rate__24h__mean` | 0.00000 | 0.00342 | 0.00223 |
| `ca__24h__mean` | 0.00000 | 0.00340 | 0.00242 |
| `lactate__24h__mean` | 0.00000 | 0.00302 | 0.00043 |
| `lactate__24h__last` | 0.00000 | 0.00299 | 0.00076 |
| `platelet__24h__mean` | 0.00000 | 0.00298 | 0.00103 |
| `creatinine__24h__min` | 0.00000 | 0.00294 | 0.00028 |
| `temperature__24h__min` | 0.00000 | 0.00283 | 0.00063 |
| `spo2__24h__count` | 0.00000 | 0.00272 | 0.00141 |
| `heart_rate__24h__count` | 0.00000 | 0.00269 | 0.00281 |
| `platelet__24h__last` | 0.00000 | 0.00243 | 0.00136 |
<!-- FI_END -->


### 3.5 Error Analysis (Test Example)

* Threshold by 10% alert rate (48h, raw): chosen thr ≈ **0.0675**
* Stay-level @ thr=0.0675: **Precision=0.379**, **Recall=0.223**, **F1=0.281**
* Exports:

  * `outputs/reports/errors_fp_stay_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fn_stay_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fp_windows_top3_h48_test_thr0.0675.parquet`
  * `outputs/reports/errors_fn_windows_top3_h48_test_thr0.0675.parquet`

```bash
python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10
# or fix threshold
python -m src.cli.error_analysis --horizon 48 --split test --threshold 0.3346
# or use calibrated probs after:
python -m src.cli.posthoc_calibrate --horizon 48 --method isotonic --refit_threshold_rate 0.10
python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10 --calibrated isotonic
```

### 3.6 Ablation (24h, 5-fold, 600 trees)

* Baseline_all: AUROC ≈ **0.797 ± 0.023**; AP ≈ **0.130 ± 0.040**
* Drop vitals: AUROC ≈ **0.771 ± 0.028**; AP ≈ **0.070 ± 0.020**
* Ongoing: drop labs / vent / others …

```bash
python -m src.cli.ablation_study --horizon 24 --folds 5 --n_estimators 600 --mode both
```

We quantify the contribution of feature groups using both **drop-one** and **keep-only** settings (5-fold CV, 600 trees, horizon=24h). Full CSV: `outputs/reports/ablation_h24.csv`.

### Keep-only (use only one group)
| Group   | AUROC (mean ± sd) | AP (mean ± sd) |
|--------|--------------------:|---------------:|
| **Vitals (n=98)** | **0.6604 ± 0.0494** | **0.0602 ± 0.0197** |
| **Labs (n=112)**  | **0.6662 ± 0.0444** | **0.0488 ± 0.0091** |
| **Vent (n=14)**   | **0.7719 ± 0.0236** | **0.0686 ± 0.0156** |
| **Others (n=148)**| **0.4985 ± 0.0503** | **0.0296 ± 0.0076** |

**Observation.** Using ventilator-related features alone surprisingly yields a relatively strong signal (AUROC ~0.77), while vitals/labs alone are moderate and “others” alone are weak.

### Drop-one (remove one group from the full set)
Baseline (all features): **AUROC 0.7974 ± 0.0228**, **AP 0.1303 ± 0.0398** (n_features=372)

| Removed Group | n_features (kept) | AUROC (mean ± sd) | AP (mean ± sd) |
|--------------|-------------------:|-------------------:|---------------:|
| **Vitals**   | 274 | **0.7711 ± 0.0276** | **0.0696 ± 0.0197** |
| **Labs**     | 260 | **0.7646 ± 0.0274** | **0.1260 ± 0.0447** |
| **Others**   | 224 | **0.8021 ± 0.0262** | **0.1249 ± 0.0360** |

**Interpretation.** Dropping **vitals** or **labs** degrades AUROC the most overall, suggesting both groups carry substantial signal when combined with others. Removing **others** barely hurts AUROC and may slightly fluctuate around baseline, implying their marginal utility is limited in this cohort and setup.

> Reproduce:
> ```bash
> python -m src.cli.ablation_study --horizon 24 --folds 5 --n_estimators 600 --mode both
> # Summary CSV → outputs/reports/ablation_h24.csv
> ```

---

<!-- ABLATION:START -->
## Ablation Studies

This section unifies **keep-only** and **drop-one** ablation; all points are 5-fold CV **mean ± std**.

### Ablation (h=24)

| Setting | Group | #Features | AUROC(mean±std) | AP(mean±std) |
|---|---:|---:|---:|---:|
| baseline_all | - | 372 | 0.7974 ± 0.0228 | 0.1303 ± 0.0398 |
| drop-one | labs | 260 | 0.7646 ± 0.0274 | 0.1260 ± 0.0447 |
| drop-one | others | 224 | 0.8021 ± 0.0262 | 0.1249 ± 0.0360 |
| drop-one | vent | 358 | 0.7225 ± 0.0465 | 0.0773 ± 0.0148 |
| drop-one | vitals | 274 | 0.7711 ± 0.0276 | 0.0696 ± 0.0197 |
| keep-only | labs | 112 | 0.6662 ± 0.0444 | 0.0488 ± 0.0091 |
| keep-only | others | 148 | 0.4985 ± 0.0503 | 0.0296 ± 0.0076 |
| keep-only | vent | 14 | 0.7719 ± 0.0236 | 0.0686 ± 0.0156 |
| keep-only | vitals | 98 | 0.6604 ± 0.0494 | 0.0602 ± 0.0197 |

<!-- ABLATION:END -->

---

## 4. Repository Layout & Project Structure

> The section below is **auto-generated** by the Makefile and can be refreshed with `make structure && make inject-structure`.
> The block is injected between anchors — please do not edit it by hand.

<!-- PROJECT_STRUCTURE:START -->
## Project Structure

### Key Directories (auto-generated)

| Path | Description | Exists | #Files |
|---|---|:---:|---:|
| `data_interim` | Intermediate/engineered feature tables for modeling. | yes | 13 |
| `data_raw` | Original/raw data (not tracked). | yes | 1 |
| `notebooks` | Exploratory notebooks (optional). | yes | 2 |
| `outputs` | Auto-generated artifacts. | yes | 158 |
| `outputs/figures` | Figures (ROC/PR, calibration, SHAP, lead-time, etc.). | yes | 31 |
| `outputs/models` | Trained model bundles (.joblib). | yes | 5 |
| `outputs/preds` | Inference & validation predictions (.parquet). | yes | 18 |
| `outputs/release` | Model release package for monitoring. | yes | 3 |
| `outputs/reports` | Metrics/tables used in paper & README (CSV/JSON/Parquet). | yes | 65 |
| `scripts` | Utility scripts (optional). | yes | 2 |
| `src` | Source code. | yes | 107 |
| `src/cli` | Command-line tools for training, evaluation, plots, release. | yes | 58 |

### Full Tree (auto-generated)

```text
├── conf/
│   ├── config.yaml
│   ├── dictionaries.yaml
│   └── schema_features.yaml
├── data_features/
│   └── .gitkeep
├── data_interim/
│   ├── .gitkeep
│   ├── cohort.parquet
│   ├── events_first.csv
│   ├── features_24h.parquet
│   ├── features_6h.parquet
│   ├── features_all.parquet
│   ├── labels_all.parquet
│   ├── labels_h24.parquet
│   ├── labels_h48.parquet
│   ├── trainset_h24.parquet
│   ├── trainset_h48.parquet
│   ├── ts_labs.parquet
│   └── ts_vitals.parquet
├── data_raw/
│   ├── hosp/
│   │   ├── admissions.csv
│   │   ├── d_hcpcs.csv
│   │   ├── d_icd_diagnoses.csv
│   │   ├── d_icd_procedures.csv
│   │   ├── d_labitems.csv
│   │   ├── diagnoses_icd.csv
│   │   ├── drgcodes.csv
│   │   ├── emar.csv
│   │   ├── emar_detail.csv
│   │   ├── hcpcsevents.csv
│   │   ├── labevents.csv
│   │   ├── microbiologyevents.csv
│   │   ├── omr.csv
│   │   ├── patients.csv
│   │   ├── pharmacy.csv
│   │   ├── poe.csv
│   │   ├── poe_detail.csv
│   │   ├── prescriptions.csv
│   │   ├── procedures_icd.csv
│   │   ├── provider.csv
│   │   ├── services.csv
│   │   └── transfers.csv
│   ├── icu/
│   │   ├── caregiver.csv
│   │   ├── chartevents.csv
│   │   ├── d_items.csv
│   │   ├── datetimeevents.csv
│   │   ├── icustays.csv
│   │   ├── ingredientevents.csv
│   │   ├── inputevents.csv
│   │   ├── outputevents.csv
│   │   └── procedureevents.csv
│   └── README.md
├── env/
│   ├── conda.yml
│   └── requirements.txt
├── notebooks/
│   ├── 01_qc_glance.ipynb
│   └── 02_figures_for_paper.ipynb
├── outputs/
│   ├── alerts/
│   │   ├── smoke_h48_stay.csv
│   │   ├── smoke_h48_stay.parquet
│   │   └── smoke_h48_window.parquet
│   ├── artifacts/
│   │   ├── .gitkeep
│   │   ├── debug_chartevents_head.csv
│   │   ├── debug_chartevents_itemid.csv
│   │   ├── debug_chartevents_stay.csv
│   │   ├── debug_chartevents_stay_itemid_window.csv
│   │   ├── debug_labevents_hadm.csv
│   │   ├── debug_labevents_hadm_itemid_window.csv
│   │   ├── debug_labevents_head.csv
│   │   ├── debug_labevents_itemid.csv
│   │   └── event_itemids.json
│   ├── docs/
│   │   └── model_card.md
│   ├── figures/
│   │   ├── .gitkeep
│   │   ├── calibration_h24_test_stay.png
│   │   ├── calibration_h24_test_window.png
│   │   ├── calibration_h48_test_stay.png
│   │   ├── calibration_h48_test_window.png
│   │   ├── calibration_hist_h24_test_stay.png
│   │   ├── calibration_hist_h24_test_window.png
│   │   ├── calibration_hist_h48_test_stay.png
│   │   ├── calibration_hist_h48_test_window.png
│   │   ├── leadtime_box_h24_test_thr0.1206.png
│   │   ├── leadtime_box_h48_test_thr0.2414.png
│   │   ├── leadtime_hist_h24_test_thr0.1206.png
│   │   ├── leadtime_hist_h48_test_thr0.2414.png
│   │   ├── pr_h24_test.png
│   │   ├── pr_h48_test.png
│   │   ├── roc_h24_test.png
│   │   ├── roc_h48_test.png
│   │   ├── shap_global_bar_h24_test.png
│   │   ├── shap_global_bar_h48_test.png
│   │   ├── shap_global_beeswarm_h24_test.png
│   │   ├── shap_global_beeswarm_h48_test.png
│   │   ├── shap_waterfall_h24_test_rank1.png
│   │   ├── shap_waterfall_h24_test_rank2.png
│   │   ├── shap_waterfall_h24_test_rank3.png
│   │   ├── shap_waterfall_h24_test_rank4.png
│   │   ├── shap_waterfall_h24_test_rank5.png
│   │   ├── shap_waterfall_h48_test_rank1.png
│   │   ├── shap_waterfall_h48_test_rank2.png
│   │   ├── shap_waterfall_h48_test_rank3.png
│   │   ├── shap_waterfall_h48_test_rank4.png
│   │   └── shap_waterfall_h48_test_rank5.png
│   ├── logs/
│   │   ├── cli.build_cohort.log
│   │   ├── cli.events_labels.log
│   │   ├── cohort.log
│   │   ├── debug.extract.log
│   │   ├── demo.log
│   │   ├── events.detect.log
│   │   ├── events.dicts.log
│   │   ├── features.extract.log
│   │   ├── labeling.log
│   │   ├── lymphoma-ewarn.log
│   │   ├── quicktest.log
│   │   ├── readers.selftest.log
│   │   ├── timeidx.selftest.log
│   │   ├── to_parquet.log
│   │   └── to_parquet.selftest.log
│   ├── metrics/
│   │   └── .gitkeep
│   ├── models/
│   │   ├── .gitkeep
│   │   ├── baseline_h24.joblib
│   │   ├── baseline_h48.joblib
│   │   ├── grouped_tvt_h24.joblib
│   │   └── grouped_tvt_h48.joblib
│   ├── preds/
│   │   ├── baseline_h24_test_preds.csv
│   │   ├── baseline_h48_test_preds.csv
│   │   ├── infer_h24.parquet
│   │   ├── infer_h48.parquet
│   │   ├── preds_h24_test.parquet
│   │   ├── preds_h24_test_cal_isotonic.parquet
│   │   ├── preds_h24_test_cal_sigmoid.parquet
│   │   ├── preds_h24_train.parquet
│   │   ├── preds_h24_val.parquet
│   │   ├── preds_h24_val_cal_isotonic.parquet
│   │   ├── preds_h24_val_cal_sigmoid.parquet
│   │   ├── preds_h48_test.parquet
│   │   ├── preds_h48_test_cal_isotonic.parquet
│   │   ├── preds_h48_test_cal_sigmoid.parquet
│   │   ├── preds_h48_train.parquet
│   │   ├── preds_h48_val.parquet
│   │   ├── preds_h48_val_cal_isotonic.parquet
│   │   └── preds_h48_val_cal_sigmoid.parquet
│   ├── release/
│   │   └── h48_v0.1/
│   │       ├── config.json
│   │       ├── model.joblib
│   │       └── release.json
│   ├── reports/
│   │   ├── ablation_h24.csv
│   │   ├── alert_table_h24.csv
│   │   ├── alert_table_h48.csv
│   │   ├── baseline_h24.json
│   │   ├── baseline_h48.json
│   │   ├── calibration_points_h24_test_stay.csv
│   │   ├── calibration_points_h24_test_window.csv
│   │   ├── calibration_points_h48_test_stay.csv
│   │   ├── calibration_points_h48_test_window.csv
│   │   ├── calibration_summary_h24_test_stay.json
│   │   ├── calibration_summary_h24_test_window.json
│   │   ├── calibration_summary_h48_test_stay.json
│   │   ├── calibration_summary_h48_test_window.json
│   │   ├── cohort_missingness_h24_all.csv
│   │   ├── cohort_missingness_h48_all.csv
│   │   ├── cohort_numeric_summary_h24_all.csv
│   │   ├── cohort_numeric_summary_h48_all.csv
│   │   ├── cohort_stats_h24_all.json
│   │   ├── cohort_stats_h48_all.json
│   │   ├── error_summary_h48_test_thr0.0675.json
│   │   ├── error_summary_h48_test_thr0.3346.json
│   │   ├── errors_fn_stay_h48_test_thr0.0675.parquet
│   │   ├── errors_fn_stay_h48_test_thr0.3346.parquet
│   │   ├── errors_fn_windows_top3_h48_test_thr0.0675.parquet
│   │   ├── errors_fn_windows_top3_h48_test_thr0.3346.parquet
│   │   ├── errors_fp_stay_h48_test_thr0.0675.parquet
│   │   ├── errors_fp_stay_h48_test_thr0.3346.parquet
│   │   ├── errors_fp_windows_top3_h48_test_thr0.0675.parquet
│   │   ├── errors_fp_windows_top3_h48_test_thr0.3346.parquet
│   │   ├── errors_tn_stay_h48_test_thr0.0675.parquet
│   │   ├── errors_tn_stay_h48_test_thr0.3346.parquet
│   │   ├── errors_tp_stay_h48_test_thr0.0675.parquet
│   │   ├── errors_tp_stay_h48_test_thr0.3346.parquet
│   │   ├── feature_importance_h24.csv
│   │   ├── feature_importance_h48.csv
│   │   ├── infer_h24_summary.json
│   │   ├── infer_h48_summary.json
│   │   ├── leadtime_details_h24_test_thr0.1206.parquet
│   │   ├── leadtime_details_h48_test_thr0.2414.parquet
│   │   ├── leadtime_h24_test_thr0.1206.csv
│   │   ├── leadtime_h48_test_thr0.2414.csv
│   │   ├── posthoc_calibration_h24_isotonic.json
│   │   ├── posthoc_calibration_h24_sigmoid.json
│   │   ├── posthoc_calibration_h48_isotonic.json
│   │   ├── posthoc_calibration_h48_sigmoid.json
│   │   ├── pr_points_h24_test.csv
│   │   ├── pr_points_h48_test.csv
│   │   ├── report_h24_test.json
│   │   ├── report_h24_train.json
│   │   ├── report_h24_val.json
│   │   ├── report_h48_test.json
│   │   ├── report_h48_train.json
│   │   ├── report_h48_val.json
│   │   ├── roc_points_h24_test.csv
│   │   ├── roc_points_h48_test.csv
│   │   ├── shap_values_top_h24_test.csv
│   │   ├── shap_values_top_h48_test.csv
│   │   ├── stay_level_h24_thr0.0067.parquet
│   │   ├── stay_level_h24_thr0.5000.parquet
│   │   ├── stay_level_h48_thr0.0100.parquet
│   │   ├── stay_level_h48_thr0.5000.parquet
│   │   ├── summary_h24.json
│   │   ├── summary_h48.json
│   │   ├── thr_sweep_h24_test.csv
│   │   └── thr_sweep_h48_test.csv
│   ├── tables/
│   │   └── .gitkeep
│   └── weekly/
│       ├── wk_h48_leadtime.parquet
│       ├── wk_h48_leadtime_stats.json
│       ├── wk_h48_stay.parquet
│       ├── wk_h48_stay_metrics.json
│       └── wk_h48_window_metrics.json
├── reports/
│   └── summary_report.md
├── scripts/
│   ├── gen_structure.py
│   └── inject_structure.py
├── src/
│   ├── cli/
│   │   ├── __init__.py
│   │   ├── ablation_study.py
│   │   ├── aggregate_by_stay.py
│   │   ├── build_cohort.py
│   │   ├── build_events_labels.py
│   │   ├── build_features.py
│   │   ├── calibration_plot.py
│   │   ├── cohort_stats.py
│   │   ├── cv_grouped.py
│   │   ├── debug_extract_sample.py
│   │   ├── deploy_batch.py
│   │   ├── error_analysis.py
│   │   ├── evaluate_all.py
│   │   ├── feature_importance.py
│   │   ├── finalize_release.py
│   │   ├── hparam_sweep.py
│   │   ├── infer_and_eval.py
│   │   ├── lead_time.py
│   │   ├── leadtime_plot.py
│   │   ├── make_splits.py
│   │   ├── make_threshold_profiles.py
│   │   ├── mk_readme.py
│   │   ├── plot_curves.py
│   │   ├── posthoc_calibrate.py
│   │   ├── shap_explain.py
│   │   ├── threshold_sweep.py
│   │   ├── train_24h.py
│   │   ├── train_48h.py
│   │   ├── train_baseline.py
│   │   ├── train_val_test.py
│   │   └── weekly_monitor.py
│   ├── cohort/
│   │   └── build_cohort.py
│   ├── datasets/
│   │   └── assemble_training.py
│   ├── eval/
│   │   ├── calibration.py
│   │   ├── dca.py
│   │   ├── metrics.py
│   │   └── report.py
│   ├── events/
│   │   ├── detect_events.py
│   │   └── dictionaries.py
│   ├── features/
│   │   ├── aggregate_windows.py
│   │   ├── build_feature_table.py
│   │   ├── derive_interventions.py
│   │   ├── extract_labs.py
│   │   ├── extract_outputs.py
│   │   ├── extract_vitals.py
│   │   ├── extract_vitals_labs.py
│   │   ├── extract_vitals_labs.py.bak
│   │   ├── extract_vitals_labs.py.bak2
│   │   ├── extract_vitals_labs.py.bak3
│   │   ├── extract_vitals_labs.py.bak_pandas
│   │   └── windows.py
│   ├── io/
│   │   ├── readers.py
│   │   ├── schema_check.py
│   │   └── to_parquet.py
│   ├── labeling/
│   │   └── make_labels.py
│   ├── modeling/
│   │   ├── calibrate.py
│   │   ├── models_gbm.py
│   │   ├── shap_tools.py
│   │   └── train.py
│   ├── models/
│   │   └── baseline.py
│   ├── split/
│   │   └── make_splits.py
│   ├── utils/
│   │   ├── log.py
│   │   ├── parallel.py
│   │   └── timeidx.py
│   ├── __init__.py
│   └── config.py
├── tests/
│   ├── test_event_rules.py
│   ├── test_time_leak.py
│   └── test_windows.py
├── Makefile
├── Makefile.bak.1761379016
└── README.md
```
<!-- PROJECT_STRUCTURE:END -->

---

## 5. How to Reproduce

### 5.1 Environment

```bash
# Example
conda create -n ewarn python=3.10 -y
conda activate ewarn
pip install -r requirements.txt
```

### 5.2 End-to-End Steps

```bash
# Curves
python -m src.cli.plot_curves --horizon 24 --split test
python -m src.cli.plot_curves --horizon 48 --split test

# Calibration (window & stay)
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform
python -m src.cli.calibration_plot --horizon 24 --split test --bins 20 --strategy uniform --stay_level
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform --stay_level

# Lead-time
python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366
python -m src.cli.leadtime_plot --horizon 24 --split test --threshold 0.1205866239132141

# Interpretability
python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5
python -m src.cli.shap_explain --horizon 24 --split test --top_n 200 --top_k_individual 5

# Post-hoc calibration & error analysis
python -m src.cli.posthoc_calibrate --horizon 24 --method isotonic --refit_threshold_rate 0.10
python -m src.cli.error_analysis --horizon 24 --split test --alert_rate 0.10 --calibrated isotonic

# Cohort stats
python -m src.cli.cohort_stats --horizon 24
python -m src.cli.cohort_stats --horizon 48
```

---

## 6. Repository Guide

* `src/` — source code
* `src/cli/` — command-line interfaces for training, evaluation, plots, release bundle
* `data_raw/` — original data (not tracked)
* `data_interim/` — engineered features and intermediate artifacts
* `outputs/` — auto-generated results

  * `outputs/models/` — trained model bundles (`.joblib`)
  * `outputs/preds/` — predictions (`.parquet`)
  * `outputs/reports/` — metrics/tables for paper & README
  * `outputs/figures/` — ROC/PR, calibration, SHAP, lead-time, etc.
  * `outputs/release/` — packaged artifacts for deployment/monitoring
* `notebooks/` — exploratory analyses (optional)
* `scripts/` — utilities (optional)

> For a fresh snapshot of the full tree, use the Makefile helpers to generate and inject the structure section.

---

## 7. Roadmap

* ✅ Curves, calibration, lead-time, SHAP, error tables
* ✅ Ablation with progress bars & ETA
* ⏳ Shadow run & standardized monitoring
* ⏳ Paper polishing and auto-syncing figures/tables
* ⏳ Multi-center generalization & fairness diagnostics

---

## 8. Acknowledgements & Disclaimer

We are grateful to our clinical collaborators and the open-source community (scikit-learn, SHAP). This repository is intended for **research use** only. Any clinical deployment requires rigorous validation, governance, and oversight.

> *With humility:* we see this as a starting point—not an endpoint—towards transparent, reliable early-warning tools that can one day assist clinicians and benefit patients.

> We know there is still much to improve. If you notice issues or have ideas, we sincerely welcome discussions and pull requests. Thank you for your patience and feedback. 🙏

---
## 9. Limitations

* Single-center cohort with lymphoma patients; generalizability to other centers/populations remains to be validated.
* Retrospective labeling and windowing may introduce time-alignment biases.
* Class imbalance is substantial; alert-rate–based thresholding can still yield non-trivial false alarms at stay level.
* We only explored RandomForest in depth; stronger gradient-boosting or temporal models (e.g., XGBoost, LightGBM, RNN/Transformer) may further improve performance.

## 10. Ethics & Data Access

This work uses MIMIC-style ICU data under the corresponding data use agreement. Access requires credentialed researchers to complete the dataset’s required training and approval process. **We do not redistribute raw patient data.** All experiments in this repo operate on derived, de-identified tables generated locally. For any clinical deployment, institutional review, safety assessments, and governance are mandatory.

---

## 11. License & Citation

**License.** Released under the MIT License (see `LICENSE`).

**Citation.** If you use this repository, please cite:

```bibtex
@misc{lymphoma_ewarn_2025,
  title        = {Lymphoma-EWARN: Early Warning for Clinical Deterioration in ICU Lymphoma Patients},
  author       = {Jia-Ning Tech and collaborators},
  year         = {2025},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/jia-ning-tech/lymphoma-ewarn}}
}

```
---

**Language:** [English](README.md) | [中文](README.zh-CN.md)

---




<!-- DCA_START -->

### Decision Curve Analysis (h=24h, split=test)

![DCA (raw probability)](outputs/figures/dca_h24_test.png)

![DCA (calibrated: isotonic)](outputs/figures/dca_h24_test_cal_isotonic.png)

![DCA (calibrated: sigmoid)](outputs/figures/dca_h24_test_cal_sigmoid.png)


### Decision Curve Analysis (h=48h, split=test)

![DCA (raw probability)](outputs/figures/dca_h48_test.png)

![DCA (calibrated: isotonic)](outputs/figures/dca_h48_test_cal_isotonic.png)

![DCA (calibrated: sigmoid)](outputs/figures/dca_h48_test_cal_sigmoid.png)

<!-- DCA_END -->
