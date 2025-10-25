from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_PREDS = ROOT / "outputs" / "preds"
OUT_REPORTS = ROOT / "outputs" / "reports"

lg = logging.getLogger("cli.infer_eval")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_data(h: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found: {p}"
    df = pd.read_parquet(p)
    # 保险：时间列为 datetime
    for c in ("index_time", "first_event_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def infer_and_eval(h: int, threshold: float | None, target_alert_rate: float | None, table_only: bool, seed: int = 42):
    model_path = ROOT / "outputs" / "models" / f"baseline_h{h}.joblib"
    assert model_path.exists(), f"not found model: {model_path}"
    bundle = joblib.load(model_path)
    pipe = bundle["pipeline"]
    feat_cols = bundle["features"]

    df = load_data(h)
    y = df["label"].astype(int).values
    X = df.reindex(columns=feat_cols)

    # 推理
    prob = pipe.predict_proba(X)[:, 1]
    pred_df = df[["stay_id","index_time","label"]].copy()
    pred_df["prob"] = prob

    OUT_PREDS.mkdir(parents=True, exist_ok=True)
    out_pred_path = OUT_PREDS / f"infer_h{h}.parquet"
    pred_df.to_parquet(out_pred_path, index=False)
    lg.info(f"saved predictions: {out_pred_path} | rows={pred_df.shape[0]}")

    # 评估（整集合）——注意这不是严格的外部验证，只做描述
    roc_auc = roc_auc_score(y, prob)
    ap = average_precision_score(y, prob)
    lg.info(f"[h={h}] overall ROC-AUC={roc_auc:.4f} | AP={ap:.4f} | pos_rate={pred_df['label'].mean():.4f}")

    # 阈值表（0.05 ~ 0.95 步长0.05）
    rows = []
    for t in np.round(np.arange(0.05, 0.951, 0.05), 3):
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        alert_rate = pred.mean()
        rows.append({
            "threshold": float(t),
            "alert_rate": float(alert_rate),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        })
    table = pd.DataFrame(rows).sort_values("threshold")

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    table_path = OUT_REPORTS / f"alert_table_h{h}.csv"
    table.to_csv(table_path, index=False)
    lg.info(f"saved alert table: {table_path}")

    # 反推阈值：基于整体样本的分位数（近似控制总体报警率）
    chosen_t = None
    if target_alert_rate is not None:
        assert 0 < target_alert_rate < 1, "--target_alert_rate must be (0,1)"
        q = 1 - target_alert_rate
        chosen_t = float(np.quantile(prob, q))
        lg.info(f"target_alert_rate={target_alert_rate:.3f} -> chosen threshold≈{chosen_t:.4f}")

    # 如果指定了 threshold，则输出基于该阈值的混淆矩阵
    if threshold is not None or chosen_t is not None:
        t = threshold if threshold is not None else chosen_t
        pred = (prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        alert_rate = pred.mean()

        summary = {
            "horizon_hours": h,
            "threshold": float(t),
            "alert_rate": float(alert_rate),
            "precision": float(pr),
            "recall": float(rc),
            "f1": float(f1),
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "roc_auc_overall": float(roc_auc),
            "average_precision_overall": float(ap),
        }
        summ_path = OUT_REPORTS / f"infer_h{h}_summary.json"
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        lg.info(f"saved summary: {summ_path}")
        if not table_only:
            print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        # 仅打印全局指标
        summary = {
            "horizon_hours": h,
            "roc_auc_overall": float(roc_auc),
            "average_precision_overall": float(ap),
        }
        summ_path = OUT_REPORTS / f"infer_h{h}_summary.json"
        with open(summ_path, "w") as f:
            json.dump(summary, f, indent=2)
        lg.info(f"saved summary: {summ_path}")
        if not table_only:
            print(json.dumps(summary, ensure_ascii=False, indent=2))

def main():
    ap = argparse.ArgumentParser(description="Run inference and produce alert-rate table.")
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--threshold", type=float, default=None, help="fixed threshold to apply")
    ap.add_argument("--target_alert_rate", type=float, default=None, help="desired overall alert rate, e.g., 0.1 for 10%%")
    ap.add_argument("--table_only", action="store_true", help="only save tables/reports without printing json summary")
    args = ap.parse_args()
    infer_and_eval(h=args.horizon, threshold=args.threshold, target_alert_rate=args.target_alert_rate, table_only=args.table_only)

if __name__ == "__main__":
    main()
