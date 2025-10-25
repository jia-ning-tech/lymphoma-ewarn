from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
PREDS = ROOT / "outputs" / "preds"
REPS  = ROOT / "outputs" / "reports"

def load(h:int, split:str):
    # 预测文件只包含 stay_id / index_time / label / prob
    dfp = pd.read_parquet(PREDS / f"preds_h{h}_{split}.parquet")
    dfr = pd.read_parquet(DATAI / f"trainset_h{h}.parquet")[["stay_id","index_time","first_event_time","label"]]
    # 只在 dfr 上转 first_event_time；dfp 无该列
    dfr["index_time"] = pd.to_datetime(dfr["index_time"], errors="coerce")
    dfr["first_event_time"] = pd.to_datetime(dfr["first_event_time"], errors="coerce")
    dfp["index_time"] = pd.to_datetime(dfp["index_time"], errors="coerce")
    # 合并以拿到 prob
    df = dfr.merge(dfp[["stay_id","index_time","prob"]], on=["stay_id","index_time"], how="left")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    df = load(args.horizon, args.split)
    df["hit"] = (df["prob"] >= args.threshold).astype(int)

    # 计算首次报警时间（按窗口时间排序后取第一条命中）
    first_hit = (
        df.loc[df["hit"]==1]
        .sort_values(["stay_id","index_time"])
        .groupby("stay_id", as_index=False)
        .first()[["stay_id","index_time"]]
        .rename(columns={"index_time":"first_alert_time"})
    )

    # 住院级聚合：真实事件时间、是否阳性
    stay = df.groupby("stay_id", as_index=False).agg(
        stay_label=("label","max"),
        first_event_time=("first_event_time","max")
    ).merge(first_hit, on="stay_id", how="left")

    # 仅在阳性住院中计算提前量（可能有NaN：未触发报警）
    stay_pos = stay.loc[stay["stay_label"]==1].copy()
    stay_pos["lead_hours"] = (
        (stay_pos["first_event_time"] - stay_pos["first_alert_time"])
        .dt.total_seconds() / 3600.0
    )

    # 导出描述统计
    desc = stay_pos["lead_hours"].dropna().describe(percentiles=[.1,.25,.5,.75,.9]).to_frame(name="hours")
    REPS.mkdir(parents=True, exist_ok=True)
    out_csv = REPS / f"leadtime_h{args.horizon}_{args.split}_thr{args.threshold:.4f}.csv"
    desc.to_csv(out_csv)

    # 导出明细（便于核对个案）：仅阳性住院
    details = stay_pos[["stay_id","first_event_time","first_alert_time","lead_hours"]]
    out_parquet = REPS / f"leadtime_details_h{args.horizon}_{args.split}_thr{args.threshold:.4f}.parquet"
    details.to_parquet(out_parquet, index=False)

    print(str(out_csv))
    print(desc.to_string())
    print(str(out_parquet))

if __name__=="__main__":
    main()
