from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "outputs" / "preds"
REPS  = ROOT / "outputs" / "reports"

def load_split(h:int, split:str)->pd.DataFrame:
    p = PREDS / f"preds_h{h}_{split}.parquet"
    df = pd.read_parquet(p)
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    return df

def stay_table(df:pd.DataFrame):
    g = df.groupby("stay_id", as_index=False).agg(stay_prob=("prob","max"), stay_label=("label","max"))
    g["stay_label"]=g["stay_label"].astype(int)
    return g

def metrics_at_thr(stay_prob, stay_label, thr):
    yhat = (stay_prob >= thr).astype(int)
    y = stay_label.astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    alert_rate = yhat.mean()
    return dict(threshold=float(thr), alert_rate=float(alert_rate), precision=float(pr), recall=float(rc), f1=float(f1),
                TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--min_rate", type=float, default=0.02)
    ap.add_argument("--max_rate", type=float, default=0.20)
    ap.add_argument("--step", type=float, default=0.01)
    args = ap.parse_args()

    df = load_split(args.horizon, args.split)
    # 窗口级曲线
    roc = roc_auc_score(df["label"], df["prob"])
    ap_score = average_precision_score(df["label"], df["prob"])
    # 住院级扫描（按stay max pooling）
    st = stay_table(df)
    rates = np.arange(args.min_rate, args.max_rate + 1e-9, args.step)
    # 用分位数反解阈值：alert_rate ≈ 目标比例
    thrs = [np.quantile(st["stay_prob"], 1-r) for r in rates]
    rows = [metrics_at_thr(st["stay_prob"].values, st["stay_label"].values, t) for t in thrs]
    out_csv = REPS / f"thr_sweep_h{args.horizon}_{args.split}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)
    # 同时打印窗口级整体能力，便于对照
    print(json.dumps({"horizon":args.horizon,"split":args.split,"window_level":{"roc_auc":roc,"average_precision":ap_score}}, indent=2))
if __name__ == "__main__":
    main()
