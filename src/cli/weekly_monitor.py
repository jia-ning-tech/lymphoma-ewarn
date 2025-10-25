from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "weekly"

def stay_aggregate(df: pd.DataFrame, thr: float):
    df = df.copy()
    df["hit"] = (df["prob"] >= thr).astype(int)
    # 首次报警时间
    first_hit = (df.loc[df["hit"]==1]
                   .sort_values(["stay_id","index_time"])
                   .groupby("stay_id", as_index=False).first()
                   [["stay_id","index_time"]]
                   .rename(columns={"index_time":"first_alert_time"}))
    stay = df.groupby("stay_id", as_index=False).agg(
        stay_label=("label","max"),
        max_prob=("prob","max"),
    ).merge(first_hit, on="stay_id", how="left")
    stay["stay_pred"] = (stay["max_prob"] >= thr).astype(int)
    return stay

def leadtime(stay_df: pd.DataFrame, ref: pd.DataFrame):
    # ref 需包含 stay_id, first_event_time
    d = stay_df.merge(ref[["stay_id","first_event_time"]], on="stay_id", how="left")
    m = d.loc[(d["stay_pred"]==1) & (d["first_alert_time"].notna()) & (d["first_event_time"].notna())].copy()
    m["lead_hours"] = (pd.to_datetime(m["first_event_time"]) - pd.to_datetime(m["first_alert_time"])).dt.total_seconds()/3600.0
    m = m.loc[m["lead_hours"]>=0]
    if m.empty:
        return {}, m
    s = m["lead_hours"].describe(percentiles=[.1,.25,.5,.75,.9]).to_dict()
    return s, m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--input_parquet", required=True, help="一周窗口级数据（含 stay_id,index_time,label,first_event_time 其中 label/first_event_time 可选）")
    ap.add_argument("--threshold_profile", default="", help="可选：在 release.json 的 threshold_profiles 中选（如 10p）")
    args = ap.parse_args()

    rdir = Path(args.release)
    cfg = json.loads((rdir/"config.json").read_text())
    meta = json.loads((rdir/"release.json").read_text())
    thr = cfg["threshold_window"]
    if args.threshold_profile and "threshold_profiles" in meta and args.threshold_profile in meta["threshold_profiles"]:
        thr = float(meta["threshold_profiles"][args.threshold_profile]["threshold_window"])

    df = pd.read_parquet(args.input_parquet)
    for c in ("index_time","first_event_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # 若没有 prob，则按模型跑一次
    if "prob" not in df.columns:
        bundle = joblib.load(cfg["model_path"])
        feats = bundle["features"]
        pipe  = bundle["pipeline"]
        X = df.reindex(columns=feats)
        df["prob"] = pipe.predict_proba(X)[:,1]

    # 窗口级指标（如标签存在）
    window_metrics = None
    if "label" in df.columns:
      y = df["label"].astype(int).values
      p = df["prob"].values
      try:
          window_metrics = dict(
              roc_auc=float(roc_auc_score(y, p)),
              average_precision=float(average_precision_score(y, p))
          )
      except Exception:
          window_metrics = None

    # 住院级 + 提前量
    stay = stay_aggregate(df, thr)
    stay_metrics = {}
    if "label" in df.columns:
        yhat = stay["stay_pred"].astype(int).values
        y = stay["stay_label"].astype(int).values
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        stay_metrics = dict(precision=float(pr), recall=float(rc), f1=float(f1), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
    stay_metrics["alert_rate"] = float(stay["stay_pred"].mean())

    # 提前量（需要 first_event_time）
    lead_stats, lead_df = ({}, pd.DataFrame())
    if "first_event_time" in df.columns:
        ref = df.groupby("stay_id", as_index=False).agg(first_event_time=("first_event_time","max"))
        lead_stats, lead_df = leadtime(stay, ref)

    OUT.mkdir(parents=True, exist_ok=True)
    prefix = f"wk_h{cfg['horizon_hours']}"
    (OUT / f"{prefix}_window_metrics.json").write_text(json.dumps(window_metrics, indent=2, ensure_ascii=False))
    (OUT / f"{prefix}_stay_metrics.json").write_text(json.dumps(stay_metrics, indent=2, ensure_ascii=False))
    stay.to_parquet(OUT / f"{prefix}_stay.parquet", index=False)
    if not lead_df.empty:
        lead_df.to_parquet(OUT / f"{prefix}_leadtime.parquet", index=False)
        (OUT / f"{prefix}_leadtime_stats.json").write_text(json.dumps(lead_stats, indent=2, ensure_ascii=False))

    print(json.dumps({"threshold_used": thr, "window_metrics": window_metrics, "stay_metrics": stay_metrics, "leadtime_stats": lead_stats}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
