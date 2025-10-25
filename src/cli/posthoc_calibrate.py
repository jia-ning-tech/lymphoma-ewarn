from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix,
    brier_score_loss
)

ROOT = Path(__file__).resolve().parents[2]
PREDS = ROOT / "outputs" / "preds"
REPS  = ROOT / "outputs" / "reports"
PREDS.mkdir(parents=True, exist_ok=True)
REPS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.posthoc_calibrate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def load_split(h:int, split:str)->pd.DataFrame:
    p = PREDS / f"preds_h{h}_{split}.parquet"
    assert p.exists(), f"not found: {p}（请先运行 train_val_test 生成 preds_h{h}_{split}.parquet）"
    df = pd.read_parquet(p)
    for c in ("index_time",):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    assert {"stay_id","label","prob"}.issubset(df.columns), "preds 需要包含：stay_id,label,prob"
    df["label"] = df["label"].astype(int)
    return df

def to_stay(df:pd.DataFrame, prob_col:str="prob")->pd.DataFrame:
    g = df.groupby("stay_id", as_index=False).agg(
        stay_label=("label","max"),
        stay_prob=(prob_col,"max")
    )
    return g

def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins:int=20, strategy:str="uniform")->float:
    if strategy == "quantile":
        qs = np.linspace(0,1,n_bins+1)
        bins = np.unique(np.quantile(y_prob, qs))
        if len(bins) - 1 < n_bins:
            bins = np.linspace(0,1,n_bins+1)
    else:
        bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(y_prob, bins) - 1
    idx = np.clip(idx, 0, len(bins)-2)
    ece = 0.0
    for b in range(len(bins)-1):
        m = (idx==b)
        if not np.any(m): 
            continue
        acc = y_true[m].mean()
        conf = y_prob[m].mean()
        ece += m.mean() * abs(acc-conf)
    return float(ece)

def select_threshold_for_alert_rate(probs: np.ndarray, target_rate: float)->float:
    target_rate = float(target_rate)
    target_rate = min(max(target_rate, 0.0), 1.0)
    if target_rate <= 0: 
        return 1.0
    if target_rate >= 1: 
        return 0.0
    thr = np.quantile(probs, 1.0 - target_rate)
    return float(thr)

def sigmoid_fit(p: np.ndarray, y: np.ndarray) -> dict:
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    z = np.log(p/(1-p))  # logit
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(z.reshape(-1,1), y)
    a = float(lr.coef_.ravel()[0])
    b = float(lr.intercept_.ravel()[0])
    return {"a": a, "b": b}

def sigmoid_apply(p: np.ndarray, params: dict) -> np.ndarray:
    eps = 1e-6
    p = np.clip(p, eps, 1-eps)
    z = np.log(p/(1-p))
    a, b = params["a"], params["b"]
    s = 1.0/(1.0 + np.exp(-(a*z + b)))
    return s.astype(float)

def isotonic_fit(p: np.ndarray, y: np.ndarray) -> dict:
    ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    ir.fit(p, y)
    return {"x_thresholds": ir.X_thresholds_.tolist(), "y_thresholds": ir.y_thresholds_.tolist()}

def isotonic_apply(p: np.ndarray, params: dict) -> np.ndarray:
    # 手工插值以避免依赖对象序列化
    x = np.asarray(params["x_thresholds"])
    y = np.asarray(params["y_thresholds"])
    return np.interp(p, x, y).astype(float)

def evaluate_window(y: np.ndarray, p: np.ndarray)->dict:
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(ece_score(y, p, n_bins=20, strategy="uniform")),
        "pos_rate": float(np.mean(y)),
        "n": int(len(y))
    }

def evaluate_stay(df: pd.DataFrame, prob_col:str, threshold: float | None = None)->dict:
    sdf = to_stay(df, prob_col=prob_col)
    y = sdf["stay_label"].astype(int).values
    p = sdf["stay_prob"].astype(float).values
    out = {"n": int(len(y)), "pos_rate": float(np.mean(y))}
    if threshold is not None:
        yhat = (p >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
        pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
        out.update({"threshold": float(threshold), "precision": float(pr), "recall": float(rc),
                    "f1": float(f1), "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn)})
    out.update({"roc_auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
                "brier": float(brier_score_loss(y, p)), "ece": float(ece_score(y, p, n_bins=20, strategy="uniform"))})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--method", choices=["isotonic","sigmoid"], required=True)
    ap.add_argument("--refit_threshold_rate", type=float, default=None,
                    help="（可选）在验证集上按校准后概率重选阈值，使报警率≈该比例；随后在测试集上评估住院级。")
    ap.add_argument("--save_suffix", default=None, help="输出文件名后缀（默认=方法名）")
    args = ap.parse_args()

    # 读取验证/测试
    df_val  = load_split(args.horizon, "val")
    df_test = load_split(args.horizon, "test")

    y_val, p_val = df_val["label"].values.astype(int), df_val["prob"].values.astype(float)
    y_tst, p_tst = df_test["label"].values.astype(int), df_test["prob"].values.astype(float)

    # 拟合映射
    if args.method == "isotonic":
        params = isotonic_fit(p_val, y_val)
        p_val_cal = isotonic_apply(p_val, params)
        p_tst_cal = isotonic_apply(p_tst, params)
    else:
        params = sigmoid_fit(p_val, y_val)
        p_val_cal = sigmoid_apply(p_val, params)
        p_tst_cal = sigmoid_apply(p_tst, params)

    # 评估（窗口级）
    win_before_val = evaluate_window(y_val, p_val)
    win_after_val  = evaluate_window(y_val, p_val_cal)
    win_before_tst = evaluate_window(y_tst, p_tst)
    win_after_tst  = evaluate_window(y_tst, p_tst_cal)

    # 保存校准后预测（窗口级）
    suffix = args.save_suffix or args.method
    out_val = df_val.copy();  out_val["prob_cal"] = p_val_cal
    out_tst = df_test.copy(); out_tst["prob_cal"] = p_tst_cal
    p1 = PREDS / f"preds_h{args.horizon}_val_cal_{suffix}.parquet"
    p2 = PREDS / f"preds_h{args.horizon}_test_cal_{suffix}.parquet"
    out_val.to_parquet(p1, index=False); out_tst.to_parquet(p2, index=False)

    # 可选：按验证集重选阈值并在测试集做住院级评估
    stay_eval = {}
    chosen_thr = None
    if args.refit_threshold_rate is not None:
        chosen_thr = select_threshold_for_alert_rate(p_val_cal, args.refit_threshold_rate)
        stay_eval["val"] = evaluate_stay(out_val, "prob_cal", threshold=chosen_thr)
        stay_eval["test"] = evaluate_stay(out_tst, "prob_cal", threshold=chosen_thr)
    else:
        # 仅给出不带阈值的住院级 AUC/AP/Brier/ECE
        stay_eval["val"]  = evaluate_stay(out_val, "prob_cal", threshold=None)
        stay_eval["test"] = evaluate_stay(out_tst, "prob_cal", threshold=None)

    # 汇总并保存
    summary = {
        "horizon": args.horizon,
        "method": args.method,
        "mapping_params": params,
        "files": {"val_parquet": str(p1), "test_parquet": str(p2)},
        "window_level": {"val": {"before": win_before_val, "after": win_after_val},
                         "test":{"before": win_before_tst, "after": win_after_tst}},
        "stay_level": stay_eval,
        "chosen_threshold_val": float(chosen_thr) if chosen_thr is not None else None,
        "refit_threshold_rate": args.refit_threshold_rate
    }
    out_json = REPS / f"posthoc_calibration_h{args.horizon}_{suffix}.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    lg.info(json.dumps(summary, ensure_ascii=False))

if __name__ == "__main__":
    main()
