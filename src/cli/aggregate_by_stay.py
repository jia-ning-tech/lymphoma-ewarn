from __future__ import annotations
import argparse, logging
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
PREDS = ROOT / "outputs" / "preds"

lg = logging.getLogger("cli.agg_stay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load(h:int):
    df_pred = pd.read_parquet(PREDS / f"infer_h{h}.parquet")
    df_raw  = pd.read_parquet(DATAI / f"trainset_h{h}.parquet")[["stay_id","index_time","label"]]
    df_pred["index_time"]=pd.to_datetime(df_pred["index_time"], errors="coerce")
    df_raw["index_time"]=pd.to_datetime(df_raw["index_time"], errors="coerce")
    df = df_raw.merge(df_pred[["stay_id","index_time","prob"]], on=["stay_id","index_time"], how="left")
    return df

def agg_by_stay(h:int, threshold:float):
    df = load(h)
    # 窗口级 → 住院级
    g = df.groupby("stay_id", as_index=False)
    stay = g.agg(
        stay_label = ("label", "max"),
        max_prob = ("prob", "max")
    )
    # 首次报警时间
    df["hit"] = (df["prob"]>=threshold).astype(int)
    first_hit = df.loc[df["hit"]==1].sort_values(["stay_id","index_time"]).groupby("stay_id", as_index=False).first()[["stay_id","index_time"]]
    first_hit = first_hit.rename(columns={"index_time":"first_alert_time"})
    stay = stay.merge(first_hit, on="stay_id", how="left")

    stay["stay_pred"] = (stay["max_prob"]>=threshold).astype(int)

    y = stay["stay_label"].values.astype(int)
    yhat = stay["stay_pred"].values.astype(int)

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    lg.info(f"[h={h}] stay-level @thr={threshold:.4f} | precision={pr:.3f} | recall={rc:.3f} | f1={f1:.3f} | TP={tp} FP={fp} TN={tn} FN={fn}")

    # 保存明细
    out = ROOT / "outputs" / "reports" / f"stay_level_h{h}_thr{threshold:.4f}.parquet"
    stay.to_parquet(out, index=False)
    print(str(out))

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()
    agg_by_stay(h=args.horizon, threshold=args.threshold)

if __name__=="__main__":
    main()
