#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, logging, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from joblib import load as joblib_load
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score

# === NEW: load trained estimator (pipeline) and training feature order ===
from pathlib import Path
import joblib

MODELDIR = Path("outputs/models")

def _load_trained_estimator(h: int):
    """
    返回:
      est: 可用于 sklearn.permutation_importance 的估计器对象
      feat_train: 训练时使用并保存下来的特征名次序(list)；若不存在则为 None
    """
    p = MODELDIR / f"grouped_tvt_h{h}.joblib"
    assert p.exists(), f"missing trained model: {p}"
    blob = joblib.load(p)

    # 常见两种保存方式：直接存 estimator；或存 dict
    if hasattr(blob, "fit") and hasattr(blob, "predict"):
        est = blob
        feat_train = None
    elif isinstance(blob, dict):
        # 你的模型包是这种：{'pipeline': CalibratedClassifierCV(...), 'features': [...], ...}
        if "pipeline" not in blob:
            raise ValueError(f"model bundle {p} has no 'pipeline' key; keys={list(blob.keys())}")
        est = blob["pipeline"]
        feat_train = blob.get("features", None)
    else:
        raise TypeError(f"unsupported model bundle type: {type(blob)}")

    # 双保险：校准器/管道都可以被 permutation_importance 调用（使用 scoring）
    if not hasattr(est, "fit"):
        raise TypeError("loaded estimator does not implement .fit()")

    return est, feat_train


lg = logging.getLogger("featimp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def find_project_root(start: Path) -> Path:
    """
    从脚本所在目录向上寻找包含以下特征之一的目录，并视为项目根：
    - 有 'src' 目录且其中含 __init__.py
    - 有 'outputs' 目录
    - 有 '.git' 目录
    找不到就退回 start 的上一层。
    """
    cur = start.resolve()
    for _ in range(6):  # 最多向上 6 层
        if (cur / "src").is_dir() and (cur / "src/__init__.py").exists():
            return cur
        if (cur / "outputs").is_dir():
            return cur
        if (cur / ".git").exists():
            return cur
        cur = cur.parent
    return start.resolve().parent  # 兜底

# 以当前脚本文件为起点寻根
ROOT   = find_project_root(Path(__file__).resolve().parent)
DATAI  = ROOT / "data_interim"
PREDS  = ROOT / "outputs" / "preds"
MODELS = ROOT / "outputs" / "models"
REPS   = ROOT / "outputs" / "reports"
FIGS   = ROOT / "outputs" / "figures"
REPS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

CAND_LABELS = ["label","y","target","outcome","event","label_window"]
CAND_SPLIT  = ["split","set","fold_tag"]
CAND_IDTS   = ["stay_id","index_time"]

def _coerce_label(df: pd.DataFrame) -> pd.DataFrame:
    for c in CAND_LABELS:
        if c in df.columns:
            if c != "label":
                df = df.rename(columns={c: "label"})
            df["label"] = df["label"].astype(int)
            return df
    raise KeyError(f"no outcome column among {CAND_LABELS}; got {df.columns.tolist()}")

def _get_split_col(df: pd.DataFrame) -> str|None:
    for c in CAND_SPLIT:
        if c in df.columns:
            return c
    return None

def _align_rows(h:int, split:str, df: pd.DataFrame) -> pd.DataFrame:
    col = _get_split_col(df)
    if col is not None:
        m = (df[col].astype(str).str.lower()==split.lower())
        lg.info("found split column '%s' -> keep %s rows for split=%s", col, m.sum(), split)
        return df.loc[m].copy()

    p = PREDS / f"preds_h{h}_{split}.parquet"
    assert p.exists(), f"fallback align needs {p}"
    pp = pd.read_parquet(p)
    keys = [c for c in CAND_IDTS if c in df.columns and c in pp.columns]
    if len(keys)<1:
        raise KeyError(f"cannot align without split; need one of {CAND_IDTS}")
    df2 = df.copy()
    for k in keys:
        if "time" in k:
            df2[k] = pd.to_datetime(df2[k], errors="coerce")
            pp[k] = pd.to_datetime(pp[k], errors="coerce")
    mdf = df2.merge(pp[keys], on=keys, how="inner")
    lg.info("aligned by keys=%s -> %s rows", keys, len(mdf))
    return mdf

def _load_features(h:int, split:str) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    载入特征表 -> 过滤出数值/布尔类型的特征列 -> 与目标 split 对齐 -> 返回 X, y, feat_cols
    - 去掉 ID/时间/标签等非特征列
    - 仅保留数值/布尔类型，避免 DatetimeArray 导致 astype/模型出错
    - 做一次稳健缺失值填补（列中位数/众数），以防模型不是含 Imputer 的 Pipeline
    """
    path = DATAI / f"trainset_h{h}.parquet"
    assert path.exists(), f"missing {path}"
    df = pd.read_parquet(path)
    df = _coerce_label(df)            # 标准化 label
    df = _align_rows(h, split, df)    # 与 split 对齐

    # 明确排除的非特征列
    drop_like = {
        "label","stay_id","subject_id","hadm_id","index_time",
        "admittime","dischtime","split","set","fold_tag"
    }
    # 仅保留数值/布尔类型（自动排除 datetime / object）
    df_num = df.drop(columns=[c for c in drop_like if c in df.columns], errors="ignore")
    df_num = df_num.select_dtypes(include=["number", "bool"]).copy()

    # 记录特征列名
    feat_cols = df_num.columns.tolist()

    # 稳健填补缺失（列中位数；布尔列缺失填 False）
    if df_num.isna().any().any():
        num_cols = df_num.select_dtypes(include=["number"]).columns
        bool_cols = df_num.select_dtypes(include=["bool"]).columns
        if len(num_cols) > 0:
            med = df_num[num_cols].median(numeric_only=True)
            df_num[num_cols] = df_num[num_cols].fillna(med)
        if len(bool_cols) > 0:
            df_num[bool_cols] = df_num[bool_cols].fillna(False)

    X = df_num                    # 不再强制 astype(float)，DataFrame 让 permutation_importance 能按列名置乱
    y = df["label"].astype(int)   # 目标
    return X, y, feat_cols


def _load_model(h:int) -> tuple[object, str]:
    cands = sorted(MODELS.glob(f"*h{h}*.*"))
    if not cands:
        raise FileNotFoundError(f"no model file under {MODELS} matching *h{h}*")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    model_path = cands[0]
    model = joblib_load(model_path)
    lg.info("loaded model: %s", model_path.relative_to(ROOT))
    return model, model_path.name

def _plot_topk(df_imp: pd.DataFrame, value_col:str, title:str, out_png:Path, k:int=20):
    d = df_imp.sort_values(value_col, ascending=True).tail(k)
    plt.figure(figsize=(8, max(4, 0.35*len(d))))
    plt.barh(d["feature"], d[value_col])
    plt.xlabel(value_col.replace("_", " "))
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=150)
    plt.close()
    lg.info("saved figure -> %s", out_png.relative_to(ROOT))



def run_perm(h: int, split: str, n_repeats: int = 5, scoring: str = "average_precision", random_state: int = 42):
    # 1) 载入对齐后的特征矩阵（你已有的函数）
    X, y, feat_cols = _load_features(h, split)  # X: DataFrame, y: Series, feat_cols: list

    # 2) 加载训练好的估计器 & 训练时的特征名次序
    est, feat_train = _load_trained_estimator(h)

    # 3) 若模型包里保存了训练特征列表，则按该列表进行**严格对齐**（顺序一致；缺失列补 0，多余列丢弃）
    if feat_train is not None:
        missing = [c for c in feat_train if c not in X.columns]
        extra   = [c for c in X.columns if c not in feat_train]
        if missing:
            # 训练时有、现在没有的列，用 0 填充（或你也可以选择丢弃；这里取稳妥策略）
            for c in missing:
                X[c] = 0.0
        # 丢弃训练时未使用的特征，避免顺序不一致
        if extra:
            X = X.drop(columns=extra)
        # 最终按训练时顺序排列
        X = X[feat_train]

    # 4) permutation importance（使用 AP 作 scoring；也可改成 roc_auc）
    # 注意：CalibratedClassifierCV/ Pipeline 都 OK，因为我们用 scoring 计算分数
    r = permutation_importance(
        est, X, y,
        scoring=scoring,            # "average_precision" / "roc_auc"
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    # 5) 写出 CSV
    import numpy as np, pandas as pd
    imp = pd.DataFrame({
        "feature": X.columns,
        "importance_mean": r.importances_mean,
        "importance_std":  r.importances_std,
        "n_repeats": n_repeats,
        "scoring": scoring,
        "horizon": h,
        "split": split,
    }).sort_values("importance_mean", ascending=False)
    out_csv = Path("outputs/reports") / f"fi_perm_h{h}_{split}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(out_csv, index=False)
    print(f"[featimp] saved permutation FI -> {out_csv}")


def run_shap(h:int, split:str, nsample:int=5000, approx:bool=True):
    try:
        import shap
    except Exception as e:
        lg.warning("SHAP not available (%s); skip.", e)
        return
    X, y, feat_cols = _load_features(h, split)
    model, model_name = _load_model(h)
    if nsample and len(X) > nsample:
        Xs = X.sample(nsample, random_state=42)
        lg.info("sampled %d/%d rows for SHAP", len(Xs), len(X))
    else:
        Xs = X
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xs)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    except Exception as e:
        lg.warning("TreeExplainer failed (%s); try KernelExplainer (slow).", e)
        try:
            bg = shap.kmeans(Xs, 50) if approx and len(Xs)>2000 else Xs.iloc[:200]
            explainer = shap.KernelExplainer(model.predict_proba if hasattr(model, "predict_proba") else model.predict, bg)
            shap_vals = explainer.shap_values(Xs, nsamples=100 if approx else "auto")
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1] if len(shap_vals)>1 else shap_vals[0]
        except Exception as e2:
            lg.error("KernelExplainer failed: %s; skip SHAP.", e2)
            return
    abs_mean = np.abs(shap_vals).mean(axis=0)
    df = pd.DataFrame({"feature": feat_cols, "shap_abs_mean": abs_mean}).sort_values("shap_abs_mean", ascending=False)
    out_csv = REPS / f"fi_shap_h{h}_{split}.csv"
    df.to_csv(out_csv, index=False)
    _plot_topk(df, "shap_abs_mean", f"SHAP | mean(|value|) (h={h}, {split})", FIGS / f"fi_shap_h{h}_{split}.png")
    lg.info("SHAP importance saved -> %s", out_csv.relative_to(ROOT))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", default="test", choices=["val","test"])
    ap.add_argument("--no-perm", action="store_true")
    ap.add_argument("--no-shap", action="store_true")
    ap.add_argument("--n-repeats", type=int, default=5, help="Permutation repeats")
    ap.add_argument("--nsample", type=int, default=5000, help="SHAP sample rows (0 for all)")
    args = ap.parse_args()
    if not args.no_perm:
        run_perm(args.horizon, args.split, n_repeats=args.n_repeats)
    if not args.no_shap:
        run_shap(args.horizon, args.split, nsample=args.nsample)

if __name__ == "__main__":
    main()
