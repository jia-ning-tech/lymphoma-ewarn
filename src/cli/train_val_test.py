from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support,
    confusion_matrix, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_PREDS = ROOT / "outputs" / "preds"
OUT_REPORTS = ROOT / "outputs" / "reports"

lg = logging.getLogger("cli.train_val_test")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

def load(h:int)->pd.DataFrame:
    p=DATAI/f"trainset_h{h}.parquet"
    assert p.exists(), f"not found: {p}"
    df=pd.read_parquet(p)
    for c in ("index_time","first_event_time"):
        if c in df.columns:
            df[c]=pd.to_datetime(df[c], errors="coerce")
    return df

def select_num_cols(df:pd.DataFrame, nan_threshold:float=0.95)->List[str]:
    drop = ID_TIME_LABEL & set(df.columns)
    cols=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    keep=[]
    for c in cols:
        s=df[c]
        if s.isna().mean()>=nan_threshold: 
            continue
        v=s.dropna().values
        if v.size==0: 
            continue
        if np.nanstd(v)==0:
            continue
        keep.append(c)
    return keep

def split_by_stay(df:pd.DataFrame, val_size:float, test_size:float, seed:int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
    # 先划出 test，再在余下里划出 val；保证 stay 不泄漏
    rng = np.random.RandomState(seed)
    stays = df["stay_id"].unique()
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    trval_idx, te_idx = next(gss1.split(df, groups=df["stay_id"]))
    df_trval = df.iloc[trval_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size/(1-test_size), random_state=seed)
    tr_idx, va_idx = next(gss2.split(df_trval, groups=df_trval["stay_id"]))
    # 索引映射回原 df
    tr_index = df_trval.iloc[tr_idx].index.values
    va_index = df_trval.iloc[va_idx].index.values
    te_index = df.iloc[te_idx].index.values
    return tr_index, va_index, te_index

def fit_model(X:pd.DataFrame, y:np.ndarray, seed:int, calibrate:str|None):
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", RandomForestClassifier(
            n_estimators=600,
            max_depth=None,
            n_jobs=-1,
            random_state=seed,
            class_weight="balanced_subsample"
        ))
    ])
    if calibrate in ("isotonic","sigmoid"):
        lg.info(f"enable calibration: {calibrate}")
        # CalibratedClassifierCV 会在内部做 CV，不用 Val 数据以免泄漏
        model = CalibratedClassifierCV(base, method=("sigmoid" if calibrate=="sigmoid" else "isotonic"), cv=5)
        model.fit(X, y)
        return model
    else:
        base.fit(X, y)
        return base

def predict_proba(model, X:pd.DataFrame)->np.ndarray:
    return model.predict_proba(X)[:,1]

def choose_threshold_by_alert_rate(probs:np.ndarray, groups:np.ndarray, labels:np.ndarray, target_alert_rate:float)->float:
    # 在 Val 集按 stay 聚合：阈值使得 触发的 stay 占比 ≈ target_alert_rate
    df = pd.DataFrame({"stay_id":groups, "prob":probs, "label":labels})
    g = df.groupby("stay_id", as_index=False).agg(stay_prob=("prob","max"), stay_label=("label","max"))
    thr = np.quantile(g["stay_prob"].values, 1-target_alert_rate)
    return float(thr)

def stay_metrics_by_threshold(stay_prob:np.ndarray, stay_label:np.ndarray, thr:float)->Dict:
    y = (stay_label>0).astype(int)
    yhat = (stay_prob>=thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0,1]).ravel()
    pr, rc, f1, _ = precision_recall_fscore_support(y, yhat, average="binary", zero_division=0)
    return dict(precision=float(pr), recall=float(rc), f1=float(f1), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))

def eval_and_dump(df:pd.DataFrame, probs:np.ndarray, split_tag:str, h:int, thr:float):
    # 窗口级
    y = df["label"].astype(int).values
    roc = roc_auc_score(y, probs)
    ap  = average_precision_score(y, probs)
    OUT_PREDS.mkdir(parents=True, exist_ok=True)
    pred_path = OUT_PREDS / f"preds_h{h}_{split_tag}.parquet"
    pd.DataFrame({"stay_id":df["stay_id"],"index_time":df["index_time"],"label":y,"prob":probs}).to_parquet(pred_path, index=False)

    # 住院级（max pooling）
    stay = pd.DataFrame({"stay_id":df["stay_id"].values, "prob":probs, "label":y})
    g = stay.groupby("stay_id", as_index=False).agg(stay_prob=("prob","max"), stay_label=("label","max"))
    sm = stay_metrics_by_threshold(g["stay_prob"].values, g["stay_label"].values, thr)

    # 保存概要
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    rep = {
        "horizon_hours": h,
        "split": split_tag,
        "window_level": {"roc_auc": float(roc), "average_precision": float(ap)},
        "stay_level": {"threshold": float(thr), **sm}
    }
    out_path = OUT_REPORTS / f"report_h{h}_{split_tag}.json"
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    lg.info(f"[h={h}] {split_tag} | window AUROC={roc:.4f} AP={ap:.4f} | stay@thr={thr:.4f} P={sm['precision']:.3f} R={sm['recall']:.3f} F1={sm['f1']:.3f}")
    return rep

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--val_size", type=float, default=0.15)
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nan_threshold", type=float, default=0.95)
    ap.add_argument("--target_alert_rate", type=float, default=0.10, help="按住院级的目标报警比例在Val上选阈值")
    ap.add_argument("--calibrate", choices=["none","sigmoid","isotonic"], default="none")
    args = ap.parse_args()

    df_all = load(args.horizon)
    tr_idx, va_idx, te_idx = split_by_stay(df_all, args.val_size, args.test_size, args.seed)
    dtr, dva, dte = df_all.loc[tr_idx].copy(), df_all.loc[va_idx].copy(), df_all.loc[te_idx].copy()

    # 特征仅以 Train 决策与拟合
    feat_cols = select_num_cols(dtr, nan_threshold=args.nan_threshold)
    Xtr, ytr = dtr[feat_cols], dtr["label"].astype(int).values
    Xva, yva = dva.reindex(columns=feat_cols), dva["label"].astype(int).values
    Xte, yte = dte.reindex(columns=feat_cols), dte["label"].astype(int).values

    model = fit_model(Xtr, ytr, seed=args.seed, calibrate=(None if args.calibrate=="none" else args.calibrate))

    # 在 Val 上选阈值（住院级 alert rate）
    p_va = predict_proba(model, Xva)
    thr = choose_threshold_by_alert_rate(
        probs=p_va, groups=dva["stay_id"].values, labels=yva, target_alert_rate=args.target_alert_rate
    )
    lg.info(f"Val 选阈值：target_alert_rate={args.target_alert_rate:.3f} -> threshold≈{thr:.4f}")

    # 在 Train/Val/Test 分别评估并落盘
    rep_tr = eval_and_dump(dtr, predict_proba(model, Xtr), "train", args.horizon, thr)
    rep_va = eval_and_dump(dva, p_va, "val", args.horizon, thr)
    rep_te = eval_and_dump(dte, predict_proba(model, Xte), "test", args.horizon, thr)

    # 存模型 & 元信息
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    bundle = {"pipeline": model, "features": feat_cols, "horizon": args.horizon, "threshold": float(thr)}
    out_model = OUT_MODELS / f"grouped_tvt_h{args.horizon}.joblib"
    joblib.dump(bundle, out_model)
    meta = {
        "horizon": args.horizon,
        "feat_count": len(feat_cols),
        "val_selected_threshold": float(thr),
        "target_alert_rate": args.target_alert_rate,
        "reports": {
            "train": f"{OUT_REPORTS}/report_h{args.horizon}_train.json",
            "val":   f"{OUT_REPORTS}/report_h{args.horizon}_val.json",
            "test":  f"{OUT_REPORTS}/report_h{args.horizon}_test.json",
        },
        "preds": {
            "train": f"{OUT_PREDS}/preds_h{args.horizon}_train.parquet",
            "val":   f"{OUT_PREDS}/preds_h{args.horizon}_val.parquet",
            "test":  f"{OUT_PREDS}/preds_h{args.horizon}_test.parquet",
        }
    }
    with open(OUT_REPORTS/f"summary_h{args.horizon}.json","w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(str(out_model))
    print(json.dumps({"horizon":args.horizon, "threshold":thr}, ensure_ascii=False))

if __name__=="__main__":
    main()
