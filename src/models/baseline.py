from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_REPORTS = ROOT / "outputs" / "reports"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_PREDS = ROOT / "outputs" / "preds"

lg = logging.getLogger("models.baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id", "hadm_id", "index_time", "first_event_time", "label"}

def _load_trainset(horizon: int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{horizon}.parquet"
    assert p.exists(), f"not found: {p}"
    df = pd.read_parquet(p)
    # 强制时间列为 datetime
    for c in ("index_time", "first_event_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df

def _numeric_feature_cols(df: pd.DataFrame, nan_threshold: float = 0.95) -> List[str]:
    """
    仅保留数值列；丢弃:
      1) ID/时间/标签列
      2) 缺失率 > nan_threshold 的列
      3) 常量列（std == 0）
    """
    # 只保留数值类型
    num_df = df.select_dtypes(include=[np.number])
    # 去掉标签列与可能的标识列
    for c in ID_TIME_LABEL:
        if c in num_df.columns:
            num_df = num_df.drop(columns=[c])

    keep = []
    for c in num_df.columns:
        s = num_df[c]
        na_frac = s.isna().mean()
        if na_frac > nan_threshold:
            continue
        # 常量列（忽略 NaN）
        v = s.dropna().values
        if v.size == 0:
            continue
        if np.nanstd(v) == 0:
            continue
        keep.append(c)

    lg.info(f"feature select -> keep={len(keep)} / total_num={num_df.shape[1]}")
    return keep

def _build_pipeline(n_estimators: int = 300, random_state: int = 42) -> Pipeline:
    # RF 对尺度不敏感，StandardScaler可省略；保留以便未来切到线性模型也不改接口
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None,
        )),
    ])

def train(horizon: int, seed: int = 42, max_features: int | None = None) -> Tuple[Path, Path]:
    df = _load_trainset(horizon)

    # 目标/特征
    y = df["label"].astype(int).values
    feat_cols = _numeric_feature_cols(df, nan_threshold=0.95)

    if not feat_cols:
        raise RuntimeError("No usable numeric features after filtering. Please inspect features_all.parquet.")

    if max_features is not None and max_features > 0:
        feat_cols = feat_cols[:max_features]
        lg.info(f"truncate features to first {len(feat_cols)} cols for training")

    X = df[feat_cols]

    # 切分
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # 训练
    pipe = _build_pipeline(random_state=seed)
    pipe.fit(X_tr, y_tr)

    # 评估
    prob_te = pipe.predict_proba(X_te)[:, 1]
    roc_auc = roc_auc_score(y_te, prob_te)
    ap = average_precision_score(y_te, prob_te)

    # 曲线点
    fpr, tpr, _ = roc_curve(y_te, prob_te)
    prec, rec, _ = precision_recall_curve(y_te, prob_te)

    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    OUT_PREDs = OUT_PREDS
    OUT_PREDs.mkdir(parents=True, exist_ok=True)

    model_path = OUT_MODELS / f"baseline_h{horizon}.joblib"
    joblib.dump({"pipeline": pipe, "features": feat_cols, "horizon": horizon}, model_path)

    # 保存测试集预测
    te_df = pd.DataFrame({
        "label": y_te,
        "prob": prob_te,
    })
    preds_csv = OUT_PREDs / f"baseline_h{horizon}_test_preds.csv"
    te_df.to_csv(preds_csv, index=False)

    # 报告
    report = {
        "horizon_hours": horizon,
        "n_samples": int(df.shape[0]),
        "n_features_used": int(len(feat_cols)),
        "roc_auc": float(roc_auc),
        "average_precision": float(ap),
        "positive_rate_overall": float(df["label"].mean()),
        "test_size": int(X_te.shape[0]),
    }
    report_path = OUT_REPORTS / f"baseline_h{horizon}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    lg.info(f"[baseline h={horizon}] ROC-AUC={roc_auc:.4f} | AP(auPRC)={ap:.4f} | feats={len(feat_cols)}")
    lg.info(f"model => {model_path}")
    lg.info(f"report => {report_path}")
    lg.info(f"test preds => {preds_csv}")

    return model_path, report_path
