# Lymphoma-EWARN – Model Card / README

_Auto-generated on 2025-10-25 07:03 UTC_

## 1. Overview
This repository contains early-warning models for lymphoma-related adverse events with rolling window predictions (24h / 48h horizons), built on MIMIC-derived cohorts. 
We report window-level and stay-level metrics, calibration, lead-time distributions, error analysis, feature importance (SHAP), and ablations.

## 2. Key Results (Test set)
### Horizon = 24h
- **Window-level**: AUROC=0.7903, AP=0.1372
- **Stay-level**:  threshold=0.1206, P=0.429, R=0.360, F1=0.391
  - Confusion: TP=9, FP=12, TN=72, FN=16
  - ROC: `outputs/figures/roc_h24_test.png`
  - PR : `outputs/figures/pr_h24_test.png`
  - Calibration (window): `outputs/figures/calibration_h24_test_window.png`
  - Calibration (stay)  : `outputs/figures/calibration_h24_test_stay.png`
  - SHAP beeswarm: `outputs/figures/shap_global_beeswarm_h24_test.png`
  - SHAP bar: `outputs/figures/shap_global_bar_h24_test.png`

### Horizon = 48h
- **Window-level**: AUROC=0.8121, AP=0.2881
- **Stay-level**:  threshold=0.2414, P=0.650, R=0.520, F1=0.578
  - Confusion: TP=13, FP=7, TN=77, FN=12
  - ROC: `outputs/figures/roc_h48_test.png`
  - PR : `outputs/figures/pr_h48_test.png`
  - Calibration (window): `outputs/figures/calibration_h48_test_window.png`
  - Calibration (stay)  : `outputs/figures/calibration_h48_test_stay.png`
  - Lead-time (hist): `outputs/figures/leadtime_hist_h48_test_thr0.2414.png`
  - Lead-time (box): `outputs/figures/leadtime_box_h48_test_thr0.2414.png`
  - SHAP beeswarm: `outputs/figures/shap_global_beeswarm_h48_test.png`
  - SHAP bar: `outputs/figures/shap_global_bar_h48_test.png`

## 3. Cohort
### Horizon = 24h
- Rows=60593, Stays=721, Window-positive rate=0.0272
- Stay-positive rate=0.1553, median windows/stay=52.0
- Index time range: 2110-05-10 02:00:00 → 2206-12-19 10:00:00
- Missingness CSV: `outputs/reports/cohort_missingness_h24_all.csv`
- Numeric summary CSV: `outputs/reports/cohort_numeric_summary_h24_all.csv`

### Horizon = 48h
- Rows=60593, Stays=721, Window-positive rate=0.0387
- Stay-positive rate=0.1553, median windows/stay=52.0
- Index time range: 2110-05-10 02:00:00 → 2206-12-19 10:00:00
- Missingness CSV: `outputs/reports/cohort_missingness_h48_all.csv`
- Numeric summary CSV: `outputs/reports/cohort_numeric_summary_h48_all.csv`

## 4. Calibration & Post-hoc Mapping
- Isotonic and Platt(sigmoid) calibration supported; calibrated preds saved under `outputs/preds/*_cal_*.parquet`.

## 5. Ablation & Feature Importance
- SHAP global plots are saved under `outputs/figures/shap_*`. Individual-case plots can be added similarly.

## 6. Reproducible CLI
Selected commands that produced the artifacts (see `src/cli/*` for full list):
```bash
# Curves
python -m src.cli.plot_curves --horizon 48 --split test
# Lead-time (example thresholds)
python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366
# SHAP
python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5
# Calibration plots
python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform
# Error analysis
python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10
```

## 7. Limitations & Notes
- Class imbalance; AP is a key metric. Stay-level aggregation uses max-window prob.
- Prospective validation and clinical integration are out-of-scope in this repo but planned.
