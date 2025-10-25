from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "outputs" / "release"
OUT = ROOT / "outputs" / "alerts"

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

def load_config(release_dir: Path)->dict:
    cfg = json.loads((release_dir/"config.json").read_text())
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True, help="如 outputs/release/h48_v0.1")
    ap.add_argument("--input_parquet", required=True, help="待推理数据（含 stay_id,index_time,[label,first_event_time 可无]）")
    ap.add_argument("--out_prefix", default="batch")
    args = ap.parse_args()

    rdir = Path(args.release)
    cfg = load_config(rdir)
    model = joblib.load(cfg["model_path"])
    feats = model["features"]
    pipe  = model["pipeline"]

    df = pd.read_parquet(args.input_parquet)
    # 时间列规整
    if "index_time" in df.columns:
        df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")

    # 只取数值特征
    X = df.reindex(columns=feats)
    prob = pipe.predict_proba(X)[:,1]

    pred = df[["stay_id","index_time"]].copy()
    if "label" in df.columns: pred["label"] = df["label"].astype(int)
    pred["prob"] = prob
    OUT.mkdir(parents=True, exist_ok=True)
    p_pred = OUT / f"{args.out_prefix}_window.parquet"
    pred.to_parquet(p_pred, index=False)

    # 住院级聚合 + 首次报警时间
    thr = cfg["threshold_window"]
    pred["hit"] = (pred["prob"] >= thr).astype(int)
    first_hit = (pred.loc[pred["hit"]==1]
                 .sort_values(["stay_id","index_time"])
                 .groupby("stay_id", as_index=False).first()
                 [["stay_id","index_time"]]
                 .rename(columns={"index_time":"first_alert_time"}))

    stay = pred.groupby("stay_id", as_index=False).agg(
        max_prob=("prob","max"),
    ).merge(first_hit, on="stay_id", how="left")

    # 抑制重复报警（同一住院内：首次后 cfg["refractory_hours"] 小时不再报）
    # 批处理按“首次触发即报”导出一条记录即可；若需要窗口级明细抑制，可拓展此处。

    stay["stay_pred"] = (stay["max_prob"]>=thr).astype(int)

    # 每日报警上限（按 first_alert_time 的日期截断）
    cap = int(cfg.get("daily_alert_cap", 999999))
    if cap < 999999 and "first_alert_time" in stay.columns:
        stay["alert_date"] = pd.to_datetime(stay["first_alert_time"]).dt.date
        stay = (stay
                .sort_values(["alert_date","first_alert_time"])  # 早触发优先
                .groupby("alert_date")
                .head(cap)
                .drop(columns=["alert_date"]))

    p_stay = OUT / f"{args.out_prefix}_stay.parquet"
    stay.to_parquet(p_stay, index=False)

    # 同时导出 csv 给临床查看
    p_csv = OUT / f"{args.out_prefix}_stay.csv"
    out_csv = stay[["stay_id","first_alert_time","max_prob","stay_pred"]].sort_values("first_alert_time")
    out_csv.to_csv(p_csv, index=False)

    print(str(p_pred))
    print(str(p_stay))
    print(str(p_csv))

if __name__=="__main__":
    main()
