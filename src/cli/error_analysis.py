from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
PREDS = ROOT / "outputs" / "preds"
REPS  = ROOT / "outputs" / "reports"
REPS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.error_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

REQ_COLS = {"stay_id","index_time","label","prob"}

def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 20) -> float:
    """Simple ECE (uniform bins on predicted prob)."""
    y = y.astype(int)
    p = p.astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins-1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        gap = abs(y[mask].mean() - p[mask].mean())
        ece += gap * (mask.mean())
    return float(ece)

def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y)**2))

def load_raw(h: int, split: str) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found raw: {p}"
    df = pd.read_parquet(p)[["stay_id","index_time","label"]]
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df

def load_preds(h:int, split:str, calibrated_suffix: Optional[str]) -> pd.DataFrame:
    if calibrated_suffix:
        p = PREDS / f"preds_h{h}_{split}_cal_{calibrated_suffix}.parquet"
        assert p.exists(), f"not found calibrated preds: {p}"
        df = pd.read_parquet(p)
        # 只保留我们需要的列，优先 prob_cal → prob，丢弃可能存在的原始 prob
        if "prob" in df.columns:
            df = df.drop(columns=["prob"])
        assert "prob_cal" in df.columns, "calibrated file must contain 'prob_cal'"
        df = df.rename(columns={"prob_cal":"prob"})
    else:
        p = PREDS / f"preds_h{h}_{split}.parquet"
        assert p.exists(), f"not found preds: {p}"
        df = pd.read_parquet(p)
        # 只保留需要列（若有多余列或重复列，统一去重）
        keep = [c for c in ["stay_id","index_time","label","prob"] if c in df.columns]
        df = df[keep]
    # 去重&规范类型
    df = df.loc[:, ~df.columns.duplicated()]
    assert REQ_COLS.issubset(set(df.columns)), f"preds must contain {REQ_COLS} (got {df.columns.tolist()})"
    df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df

def merge_raw_pred(df_raw: pd.DataFrame, df_pred: pd.DataFrame) -> pd.DataFrame:
    # 左连接，确保缺失预测也在（后面窗口级会 dropna，住院级把 NaN 当 0）
    df = df_raw.merge(df_pred[["stay_id","index_time","prob"]], on=["stay_id","index_time"], how="left", suffixes=("",""))
    return df

def choose_threshold_by_alert_rate(prob: np.ndarray, alert_rate: float) -> float:
    assert 0.0 < alert_rate < 1.0, "alert_rate must be in (0,1)"
    # 以概率分位数反推阈值：使 yhat.mean() ≈ alert_rate
    thr = np.quantile(prob[~np.isnan(prob)], 1.0 - alert_rate)
    return float(thr)

def eval_window(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    return dict(
        roc_auc=float(roc_auc_score(y, p)),
        average_precision=float(average_precision_score(y, p)),
        brier=brier_score(y, p),
        ece=expected_calibration_error(y, p, n_bins=20),
        pos_rate=float(y.mean()),
        n=int(len(y)),
    )

def stay_aggregate(df: pd.DataFrame, thr: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # 缺失 prob 视为 0，保证住院级不漏
    df2 = df.copy()
    df2["prob_filled"] = df2["prob"].fillna(0.0).astype(float)
    stay = df2.groupby("stay_id", as_index=False).agg(
        stay_prob=("prob_filled","max"),
        stay_label=("label","max")
    )
    stay["stay_pred"] = (stay["stay_prob"] >= thr).astype(int)
    y = stay["stay_label"].astype(int).values
    yhat = stay["stay_pred"].astype(int).values
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    metrics = dict(
        precision=float(pr), recall=float(rc), f1=float(f1),
        TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn),
        roc_auc=float(roc_auc_score(y, stay["stay_prob"].values)),
        average_precision=float(average_precision_score(y, stay["stay_prob"].values)),
        brier=brier_score(y, stay["stay_prob"].values)
    )
    return stay, metrics

def dump_error_tables(df: pd.DataFrame, stay: pd.DataFrame, thr: float, h: int, split: str, topk: int = 3) -> Dict[str,str]:
    tag = f"h{h}_{split}_thr{thr:.4f}"
    # 标注每行是否命中
    df_w = df.copy()
    df_w["hit"] = (df_w["prob"].fillna(0.0) >= thr).astype(int)
    # 住院级四类
    s = stay.copy()
    s["type"] = np.where((s["stay_label"]==1) & (s["stay_pred"]==1), "TP",
                  np.where((s["stay_label"]==0) & (s["stay_pred"]==1), "FP",
                  np.where((s["stay_label"]==0) & (s["stay_pred"]==0), "TN", "FN")))
    paths = {}
    for t in ["TP","FP","TN","FN"]:
        p = REPS / f"errors_{t.lower()}_stay_{tag}.parquet"
        s.loc[s["type"]==t].to_parquet(p, index=False)
        paths[f"{t.lower()}_stay"] = str(p)

    # 为 FP/FN 导出对应住院的 topk 窗口（按 prob 降序）
    for t, lab in [("FP",0), ("FN",1)]:
        stays = s.loc[s["type"]==t, "stay_id"].unique()
        sub = df_w[df_w["stay_id"].isin(stays)].copy()
        sub["prob_filled"] = sub["prob"].fillna(0.0)
        sub = sub.sort_values(["stay_id","prob_filled","index_time"], ascending=[True,False,True])
        # 组内取前 k
        sub["rank"] = sub.groupby("stay_id")["prob_filled"].rank(method="first", ascending=False)
        sub_topk = sub[sub["rank"]<=topk].drop(columns=["rank"])
        p = REPS / f"errors_{t.lower()}_windows_top{topk}_{tag}.parquet"
        sub_topk.to_parquet(p, index=False)
        paths[f"{t.lower()}_windows_topk"] = str(p)
    return paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test","train"], default="test")
    # 二选一：alert_rate 或 threshold
    ap.add_argument("--alert_rate", type=float, default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--calibrated", choices=["isotonic","sigmoid"], default=None,
                    help="use calibrated predictions: preds_h{h}_{split}_cal_{method}.parquet (use prob_cal)")
    ap.add_argument("--topk", type=int, default=3, help="for FP/FN windows export per-stay top-k windows")
    args = ap.parse_args()

    assert (args.alert_rate is not None) ^ (args.threshold is not None), \
        "Specify exactly one of --alert_rate or --threshold"

    df_raw  = load_raw(args.horizon, args.split)
    df_pred = load_preds(args.horizon, args.split, args.calibrated)
    df = merge_raw_pred(df_raw, df_pred)

    # 窗口级评估：去除 prob 的 NaN 行
    mask_valid = df["prob"].notna()
    n_nan = int((~mask_valid).sum())
    if n_nan > 0:
        lg.warning(f"found {n_nan}/{len(df)} rows with NaN prob after merge; "
                   f"window-level metrics & threshold selection will drop NaN rows; "
                   f"stay-level aggregation will treat missing prob as 0.0.")
    y_w = df.loc[mask_valid, "label"].values.astype(int)
    p_w = df.loc[mask_valid, "prob"].values.astype(float)

    # 选阈值
    if args.threshold is not None:
        thr = float(args.threshold)
    else:
        thr = choose_threshold_by_alert_rate(p_w, args.alert_rate)
        lg.info(f"chosen threshold by alert_rate={args.alert_rate:.3f} -> thr≈{thr:.4f}")

    # 窗口级指标
    win_metrics = eval_window(y_w, p_w)

    # 住院级聚合 + 指标
    stay, stay_metrics = stay_aggregate(df, thr)
    alert_rate = float((stay["stay_pred"]==1).mean())

    # 导出误报/漏报明细
    paths = dump_error_tables(df, stay, thr, args.horizon, args.split, topk=args.topk)

    summary = {
        "horizon": args.horizon,
        "split": args.split,
        "prob_source": "calibrated:"+args.calibrated if args.calibrated else "raw",
        "threshold": thr,
        "alert_rate": alert_rate,
        "nan_prob_rows_dropped_for_window_metrics": n_nan,
        "window_level": win_metrics,
        "stay_level": stay_metrics,
        "files": paths
    }
    lg.info(json.dumps(summary, ensure_ascii=False))
