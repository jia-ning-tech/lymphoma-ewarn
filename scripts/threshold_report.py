# scripts/threshold_report.py
# -*- coding: utf-8 -*-
"""
阈值选择 + 混淆矩阵 + 指标 + 单阈值净获益（per 100 patients）
- 不依赖你现有训练/评估脚本，独立读取 parquet/json
- 自动从 posthoc_calibration_h{h}_{method}.json 读取已选阈值（若存在）
- 若无 JSON，则可选：固定阈值列表（--extra-thr），或按默认列表演示
- 生成：
  - outputs/reports/threshold_metrics_h{h}_{split}_{method}.csv
  - outputs/docs/threshold_h{h}_{split}_{method}.md
"""

from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT   = Path(__file__).resolve().parents[1].parent  # 项目根
DATAI  = ROOT / "data_interim"
PREDS  = ROOT / "outputs" / "preds"
REPS   = ROOT / "outputs" / "reports"
DOCS   = ROOT / "outputs" / "docs"
REPS.mkdir(parents=True, exist_ok=True)
DOCS.mkdir(parents=True, exist_ok=True)

CAND_LABELS = ["label","y","target","outcome","event","label_window","label_stay"]

def _coerce_label(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAND_LABELS:
        if c in df.columns:
            if c != "label":
                df = df.rename(columns={c: "label"})
            df["label"] = df["label"].astype(int)
            return df
    raise KeyError(f"could not find outcome column among {CAND_LABELS}; got {df.columns.tolist()}")

def _load_raw(h: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found raw: {p}"
    df = pd.read_parquet(p)
    if "index_time" in df.columns:
        df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df = _coerce_label(df)
    keep = [c for c in ["stay_id","index_time","label"] if c in df.columns]
    return df[keep]

def _load_preds(h: int, split: str, method: str) -> pd.DataFrame:
    if method == "raw":
        p = PREDS / f"preds_h{h}_{split}.parquet"
        assert p.exists(), f"missing preds: {p}"
        df = pd.read_parquet(p)
        assert "prob" in df.columns, f"'prob' not in {p}"
    else:
        p = PREDS / f"preds_h{h}_{split}_cal_{method}.parquet"
        assert p.exists(), f"missing calibrated preds: {p}"
        df = pd.read_parquet(p)
        if "prob_cal" not in df.columns:
            raise KeyError(f"'prob_cal' not in {p} (calibrated probs expected)")
        if "prob" in df.columns:
            df = df.drop(columns=["prob"])
        df = df.rename(columns={"prob_cal": "prob"})
    if "index_time" in df.columns:
        df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    keep = [c for c in ["stay_id","index_time","prob"] if c in df.columns]
    return df[keep]

def _pick_threshold_from_json(h: int, method: str) -> float | None:
    # 兼容不同键名
    p = REPS / f"posthoc_calibration_h{h}_{method}.json"
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        for key in ["chosen_threshold_val", "chosen_threshold", "best_threshold", "threshold"]:
            if key in obj and isinstance(obj[key], (int,float)):
                return float(obj[key])
        # 可能嵌套
        nested = obj.get("stay_level") or obj.get("meta") or {}
        for key in ["threshold", "chosen_threshold_val", "best_threshold"]:
            if key in nested and isinstance(nested[key], (int,float)):
                return float(nested[key])
    except Exception:
        return None
    return None

def _metrics_at_threshold(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    TP = int(((pred==1)&(y==1)).sum())
    FP = int(((pred==1)&(y==0)).sum())
    TN = int(((pred==0)&(y==0)).sum())
    FN = int(((pred==0)&(y==1)).sum())
    N  = TP+FP+TN+FN
    eps = 1e-12
    sens = TP / max(TP+FN, eps)   # recall
    spec = TN / max(TN+FP, eps)
    ppv  = TP / max(TP+FP, eps)
    npv  = TN / max(TN+FN, eps)
    acc  = (TP+TN) / max(N, eps)
    f1   = (2*TP) / max(2*TP+FP+FN, eps)
    prev = y.mean()  # prevalence
    # DCA 单点净获益（per 100）
    pt = min(max(thr, 1e-6), 1-1e-6)
    nb_model = (TP/N) - (FP/N) * (pt/(1-pt))
    nb_all   = (prev) - (1-prev) * (pt/(1-pt))
    return dict(
        threshold=thr,
        TP=TP, FP=FP, TN=TN, FN=FN, N=N,
        prevalence=prev,
        sensitivity=sens, specificity=spec, PPV=ppv, NPV=npv,
        accuracy=acc, F1=f1,
        net_benefit_per100=nb_model*100.0,
        treat_all_per100=nb_all*100.0
    )

def run_one(h:int, split:str, method:str, extra_thr:list[float]) -> tuple[pd.DataFrame, Path, Path]:
    df_raw   = _load_raw(h)
    df_pred  = _load_preds(h, split, method)
    df = df_raw.merge(df_pred, on=[c for c in ["stay_id","index_time"] if c in df_raw.columns and c in df_pred.columns], how="left")
    df = df.dropna(subset=["prob"])
    y = df["label"].astype(int).to_numpy()
    p = df["prob"].astype(float).to_numpy()

    thr_list = []
    t_json = _pick_threshold_from_json(h, method)
    if t_json is not None:
        thr_list.append(("json", t_json))
    for t in extra_thr:
        thr_list.append(("manual", float(t)))
    if not thr_list:
        thr_list = [("demo", t) for t in (0.05, 0.10, 0.20, 0.30, 0.40)]

    rows = []
    for kind, thr in thr_list:
        m = _metrics_at_threshold(y, p, thr)
        m["source"] = kind
        rows.append(m)
    dfm = pd.DataFrame(rows)
    # 保存 CSV
    csv_out = REPS / f"threshold_metrics_h{h}_{split}_{method}.csv"
    dfm.to_csv(csv_out, index=False)

    # 保存 MD（相对路径、GitHub 可渲染）
    md_out = DOCS / f"threshold_h{h}_{split}_{method}.md"
    lines = []
    lines.append(f"### h={h}, split={split}, method={method}")
    lines.append("")
    lines.append("| source | threshold | N | prevalence | sensitivity | specificity | PPV | NPV | accuracy | F1 | net benefit/100 | treat-all/100 | TP | FP | TN | FN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in dfm.iterrows():
        lines.append(
            f"| {r['source']} | {r['threshold']:.4f} | {int(r['N'])} | {r['prevalence']:.3f} | "
            f"{r['sensitivity']:.3f} | {r['specificity']:.3f} | {r['PPV']:.3f} | {r['NPV']:.3f} | "
            f"{r['accuracy']:.3f} | {r['F1']:.3f} | {r['net_benefit_per100']:.2f} | {r['treat_all_per100']:.2f} | "
            f"{int(r['TP'])} | {int(r['FP'])} | {int(r['TN'])} | {int(r['FN'])} |"
        )
    md_out.write_text("\n".join(lines), encoding="utf-8")
    return dfm, csv_out, md_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizons", default="24,48", help="e.g. 24,48")
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--methods", default="raw,isotonic,sigmoid", help="subset of {raw,isotonic,sigmoid}")
    ap.add_argument("--extra-thr", default="", help="comma separated thresholds; used when JSON missing. e.g. 0.1,0.2,0.3")
    args = ap.parse_args()

    horizons = [int(x) for x in str(args.horizons).split(",") if x.strip()]
    methods  = [m.strip() for m in args.methods.split(",") if m.strip()]
    extra_thr = [float(x) for x in args.extra_thr.split(",") if x.strip()]

    for h in horizons:
        for m in methods:
            try:
                dfm, csv_out, md_out = run_one(h, args.split, m, extra_thr)
                print(f"[threshold] saved CSV -> {csv_out}")
                print(f"[threshold] saved MD  -> {md_out}")
            except AssertionError as e:
                print(f"[skip] h={h} {m}: {e}")
            except FileNotFoundError as e:
                print(f"[skip] h={h} {m}: {e}")

if __name__ == "__main__":
    main()
