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
