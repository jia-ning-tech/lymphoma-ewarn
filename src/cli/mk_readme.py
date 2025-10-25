from __future__ import annotations
import argparse, json, textwrap
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
REPS = ROOT / "outputs" / "reports"
FIGS = ROOT / "outputs" / "figures"
DOCS = ROOT / "outputs" / "docs"

def _read_json(p:Path):
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

def safe(p:Path)->str:
    return str(p) if p.exists() else ""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", type=str, default="24,48")
    ap.add_argument("--split", type=str, default="test", choices=["val","test"])
    ap.add_argument("--out", type=str, default="README.md")
    args = ap.parse_args()

    DOCS.mkdir(parents=True, exist_ok=True)
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    # 尝试读取我们之前产出的关键 JSON
    reports = {}
    for h in horizons:
        rep = _read_json(REPS / f"report_h{h}_{args.split}.json")
        if rep is not None:
            reports[h] = rep

    # 常用图表/表格路径（若存在则会在 README 中展示）
    paths = {}
    for h in horizons:
        paths[h] = {
            "roc_csv":  REPS / f"roc_points_h{h}_{args.split}.csv",
            "roc_png":  FIGS / f"roc_h{h}_{args.split}.png",
            "pr_csv":   REPS / f"pr_points_h{h}_{args.split}.csv",
            "pr_png":   FIGS / f"pr_h{h}_{args.split}.png",
            "cal_win_csv": REPS / f"calibration_points_h{h}_{args.split}_window.csv",
            "cal_win_png": FIGS / f"calibration_h{h}_{args.split}_window.png",
            "cal_stay_csv": REPS / f"calibration_points_h{h}_{args.split}_stay.csv",
            "cal_stay_png": FIGS / f"calibration_h{h}_{args.split}_stay.png",
            "lead_hist": FIGS / f"leadtime_hist_h{h}_{args.split}_thr0.2414.png",  # 可能不存在，不强制
            "lead_box":  FIGS / f"leadtime_box_h{h}_{args.split}_thr0.2414.png",   # 可能不存在，不强制
            "shap_bee":  FIGS / f"shap_global_beeswarm_h{h}_{args.split}.png",
            "shap_bar":  FIGS / f"shap_global_bar_h{h}_{args.split}.png",
            "ablation_csv": REPS / f"ablation_h{h}.csv",
            "cohort_json": REPS / f"cohort_stats_h{h}_all.json",
            "missing_csv": REPS / f"cohort_missingness_h{h}_all.csv",
            "numeric_csv": REPS / f"cohort_numeric_summary_h{h}_all.csv",
        }

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append(f"# Lymphoma-EWARN – Model Card / README")
    lines.append("")
    lines.append(f"_Auto-generated on {now}_")
    lines.append("")

    lines.append("## 1. Overview")
    lines.append(textwrap.dedent("""
    This repository contains early-warning models for lymphoma-related adverse events with rolling window predictions (24h / 48h horizons), built on MIMIC-derived cohorts. 
    We report window-level and stay-level metrics, calibration, lead-time distributions, error analysis, feature importance (SHAP), and ablations.
    """).strip())
    lines.append("")

    lines.append("## 2. Key Results (Test set)")
    for h in horizons:
        rep = reports.get(h)
        lines.append(f"### Horizon = {h}h")
        if rep:
            wl = rep.get("window_level", {})
            sl = rep.get("stay_level", {})
            lines.append(f"- **Window-level**: AUROC={wl.get('roc_auc'):.4f}, AP={wl.get('average_precision'):.4f}")
            lines.append(f"- **Stay-level**:  threshold={sl.get('threshold'):.4f}, P={sl.get('precision'):.3f}, R={sl.get('recall'):.3f}, F1={sl.get('f1'):.3f}")
            lines.append(f"  - Confusion: TP={sl.get('TP')}, FP={sl.get('FP')}, TN={sl.get('TN')}, FN={sl.get('FN')}")
        else:
            lines.append(f"- (No consolidated JSON report found at {REPS / f'report_h{h}_{args.split}.json'})")
        # 曲线图
        p = paths[h]
        if p["roc_png"].exists(): lines.append(f"  - ROC: `outputs/figures/{p['roc_png'].name}`")
        if p["pr_png"].exists():  lines.append(f"  - PR : `outputs/figures/{p['pr_png'].name}`")
        if p["cal_win_png"].exists(): lines.append(f"  - Calibration (window): `outputs/figures/{p['cal_win_png'].name}`")
        if p["cal_stay_png"].exists(): lines.append(f"  - Calibration (stay)  : `outputs/figures/{p['cal_stay_png'].name}`")
        if p["lead_hist"].exists(): lines.append(f"  - Lead-time (hist): `outputs/figures/{p['lead_hist'].name}`")
        if p["lead_box"].exists():  lines.append(f"  - Lead-time (box): `outputs/figures/{p['lead_box'].name}`")
        if p["shap_bee"].exists():  lines.append(f"  - SHAP beeswarm: `outputs/figures/{p['shap_bee'].name}`")
        if p["shap_bar"].exists():  lines.append(f"  - SHAP bar: `outputs/figures/{p['shap_bar'].name}`")
        lines.append("")

    lines.append("## 3. Cohort")
    for h in horizons:
        p = paths[h]
        lines.append(f"### Horizon = {h}h")
        cj = _read_json(p["cohort_json"])
        if cj:
            bc = cj.get("basic_counts", {})
            sl = cj.get("stay_level", {})
            lines.append(f"- Rows={bc.get('n_rows')}, Stays={sl.get('n_stay')}, Window-positive rate={bc.get('positive_rate_window'):.4f}")
            lines.append(f"- Stay-positive rate={sl.get('positive_rate_stay'):.4f}, median windows/stay={sl.get('median_windows_per_stay')}")
            lines.append(f"- Index time range: {bc.get('index_time_min')} → {bc.get('index_time_max')}")
        if p["missing_csv"].exists():
            lines.append(f"- Missingness CSV: `outputs/reports/{p['missing_csv'].name}`")
        if p["numeric_csv"].exists():
            lines.append(f"- Numeric summary CSV: `outputs/reports/{p['numeric_csv'].name}`")
        lines.append("")

    lines.append("## 4. Calibration & Post-hoc Mapping")
    lines.append("- Isotonic and Platt(sigmoid) calibration supported; calibrated preds saved under `outputs/preds/*_cal_*.parquet`.")
    lines.append("")

    lines.append("## 5. Ablation & Feature Importance")
    for h in horizons:
        p = paths[h]
        if p["ablation_csv"].exists():
            lines.append(f"- Ablation CSV (h={h}): `outputs/reports/{p['ablation_csv'].name}`")
    lines.append("- SHAP global plots are saved under `outputs/figures/shap_*`. Individual-case plots can be added similarly.")
    lines.append("")

    lines.append("## 6. Reproducible CLI")
    lines.append("Selected commands that produced the artifacts (see `src/cli/*` for full list):")
    lines.append("```bash")
    lines.append("# Curves")
    lines.append("python -m src.cli.plot_curves --horizon 48 --split test")
    lines.append("# Lead-time (example thresholds)")
    lines.append("python -m src.cli.leadtime_plot --horizon 48 --split test --threshold 0.24142504229084366")
    lines.append("# SHAP")
    lines.append("python -m src.cli.shap_explain --horizon 48 --split test --top_n 200 --top_k_individual 5")
    lines.append("# Calibration plots")
    lines.append("python -m src.cli.calibration_plot --horizon 48 --split test --bins 20 --strategy uniform")
    lines.append("# Error analysis")
    lines.append("python -m src.cli.error_analysis --horizon 48 --split test --alert_rate 0.10")
    lines.append("```")
    lines.append("")

    lines.append("## 7. Limitations & Notes")
    lines.append("- Class imbalance; AP is a key metric. Stay-level aggregation uses max-window prob.")
    lines.append("- Prospective validation and clinical integration are out-of-scope in this repo but planned.")
    lines.append("")

    out = ROOT / args.out
    out.write_text("\n".join(lines), encoding="utf-8")
    print(str(out))

if __name__ == "__main__":
    main()
