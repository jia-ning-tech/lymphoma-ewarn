from __future__ import annotations
import argparse, logging, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
from sklearn.exceptions import NotFittedError

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_REPS = ROOT / "outputs" / "reports"
OUT_FIGS.mkdir(parents=True, exist_ok=True)
OUT_REPS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.shap_explain")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

def _load_bundle(h:int):
    p1 = ROOT/"outputs"/"models"/f"grouped_tvt_h{h}.joblib"
    p2 = ROOT/"outputs"/"models"/f"baseline_h{h}.joblib"
    p = p1 if p1.exists() else p2
    assert p.exists(), f"no model found: {p1} nor {p2}"
    bundle = joblib.load(p)
    pipe = bundle.get("pipeline", None)
    assert pipe is not None, "bundle must contain 'pipeline'"
    feats = bundle.get("features", None)
    assert feats is not None, "bundle must contain 'features'"
    return bundle, pipe, feats, p

def _last_estimator(obj):
    if isinstance(obj, Pipeline):
        return obj.steps[-1][1]
    return obj

def _unwrap_calibrated(model):
    est = model
    seen = set()
    while True:
        key = (type(est), id(est))
        if key in seen: break
        seen.add(key)
        est = _last_estimator(est)
        if isinstance(est, CalibratedClassifierCV):
            try:
                cc = est.calibrated_classifiers_[0]
                for name in ("base_estimator","estimator","classifier"):
                    if hasattr(cc, name):
                        est = getattr(cc, name)
                        break
                else:
                    lg.warning("CalibratedClassifier found but base estimator not located; keep as-is.")
                    break
            except Exception as e:
                lg.warning(f"failed to unwrap CalibratedClassifierCV: {e}")
                break
        else:
            break
    return est

def _extract_preproc_and_estimator(pipe):
    if isinstance(pipe, Pipeline):
        steps = pipe.steps
        model = steps[-1][1]
        preproc = Pipeline(steps[:-1]) if len(steps)>1 else None
    else:
        model, preproc = pipe, None
    est = _unwrap_calibrated(model)

    if preproc is None:
        transform_fn = lambda X: X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=X.columns if hasattr(X,'columns') else None)
    else:
        def _tf(X):
            Xdf = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=X.columns if hasattr(X,'columns') else None)
            try:
                return preproc.transform(Xdf)
            except NotFittedError:
                lg.warning("preproc pipeline not fitted, return raw matrix as DataFrame")
                return Xdf
        transform_fn = _tf
    return preproc, est, transform_fn

def _load_data_matrix(h:int, feats:list[str]):
    df = pd.read_parquet(DATAI / f"trainset_h{h}.parquet")
    for c in ("index_time","first_event_time"):
        if c in df.columns: df[c] = pd.to_datetime(df[c], errors="coerce")
    y = df["label"].astype(int).values
    X = df.reindex(columns=feats)
    return df, X, y

def _build_background(Xt_np: np.ndarray):
    try:
        bg_size = min(100, len(Xt_np))
        bg = shap.kmeans(Xt_np, k=min(50, bg_size)) if len(Xt_np) > 50 else Xt_np[:bg_size]
    except Exception:
        bg = Xt_np[:min(100, len(Xt_np))]
    return bg

def _to_array(x):
    if hasattr(x, "values"):
        return np.asarray(x.values)
    return np.asarray(x)

def _coerce_shap_matrix(raw, Xt_np, n_features:int) -> np.ndarray:
    """
    将各种返回形状规范为 (n_samples, n_features)，并固定取正类（class 1）。
    支持的可能形状举例：
      - list[ (n_samples, n_features), (n_samples, n_features) ]
      - (n_samples, n_features, 2)
      - (2, n_samples, n_features)
      - (n_samples, 2, n_features)
      - (n_features, n_samples) / (n_features, 2) / (372, 2)
      - (n_samples, n_features)
    """
    n_samples = Xt_np.shape[0]

    # list/tuple => 取正类（最后一个）
    if isinstance(raw, (list, tuple)):
        try:
            arr = np.asarray(raw[-1])
        except Exception:
            arr = np.asarray(raw)
    else:
        arr = _to_array(raw)

    # 3D 情况
    if arr.ndim == 3:
        s = arr.shape
        # (n_samples, n_features, 2)
        if s[0] == n_samples and s[1] == n_features and s[2] == 2:
            return arr[:, :, -1]
        # (2, n_samples, n_features)
        if s[0] == 2 and s[1] == n_samples and s[2] == n_features:
            return arr[-1, :, :]
        # (n_samples, 2, n_features)
        if s[0] == n_samples and s[1] == 2 and s[2] == n_features:
            return arr[:, -1, :]
        # (n_features, n_samples, 2)
        if s[0] == n_features and s[1] == n_samples and s[2] == 2:
            return arr[:, :, -1].T
        # 无法识别则压到 2D 再处理
        arr = arr.reshape(-1, s[-1]) if s[-1] == n_features else arr.mean(axis=0)

    # 2D 情况
    if arr.ndim == 2:
        r, c = arr.shape
        # (n_samples, n_features)
        if r == n_samples and c == n_features:
            return arr
        # (n_features, n_samples) -> 转置
        if r == n_features and c == n_samples:
            return arr.T
        # (n_features, 2) -> 只有特征×类别，没有样本；无法用于样本级解释，退化为重复成 n_samples 行
        if r == n_features and c == 2:
            lg.warning("Got (n_features, 2) SHAP matrix; repeating positive-class column for each sample.")
            return np.tile(arr[:, -1], (n_samples, 1))
        # (n_samples, 2) -> 只有样本×类别；无法用于特征级解释，退化为全零
        if r == n_samples and c == 2:
            lg.warning("Got (n_samples, 2) SHAP matrix; fallback to zeros.")
            return np.zeros((n_samples, n_features), dtype=float)

    # 1D 情况（可能是单样本特征重要度）
    if arr.ndim == 1 and arr.size == n_features:
        return arr.reshape(1, -1)

    raise ValueError(f"Cannot coerce SHAP matrix with shape {arr.shape} to (n_samples, n_features)")

def _force_tree_explainer(est, Xt_np, n_features:int):
    explainer = shap.TreeExplainer(est, model_output="raw")
    raw = explainer.shap_values(Xt_np, check_additivity=False)
    sm = _coerce_shap_matrix(raw, Xt_np, n_features)
    lg.info("SHAP via TreeExplainer(model_output='raw') succeeded.")
    return sm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["val","test"], default="test")
    ap.add_argument("--top_n", type=int, default=200)
    ap.add_argument("--top_k_individual", type=int, default=5)
    args = ap.parse_args()

    bundle, pipe_or_est, feats, model_path = _load_bundle(args.horizon)
    preproc, est, transform_fn = _extract_preproc_and_estimator(pipe_or_est)
    lg.info(f"Resolved estimator for SHAP: est={est.__class__.__name__}")

    # 准备数据
    df_all, X_df, y = _load_data_matrix(args.horizon, feats)
    try:
        prob_all = bundle["pipeline"].predict_proba(X_df)[:, 1]
    except Exception:
        Xt_all = transform_fn(X_df)
        Xt_all_np = Xt_all.values if isinstance(Xt_all, pd.DataFrame) else np.asarray(Xt_all)
        prob_all = est.predict_proba(Xt_all_np)[:, 1]

    idx = np.argsort(-prob_all)[:min(args.top_n, len(prob_all))]
    X_top_df = X_df.iloc[idx].copy()
    Xt = transform_fn(X_top_df)
    Xt_np = Xt.values if isinstance(Xt, pd.DataFrame) else np.asarray(Xt)
    feature_names = list(feats)
    n_features = len(feature_names)

    # 先尝试通用 Explainer
    shap_mat = None
    try:
        bg_np = _build_background(Xt_np)
        explainer = shap.Explainer(est, bg_np, algorithm="auto")
        shap_values = explainer(Xt_np, check_additivity=False)
        cand = _coerce_shap_matrix(shap_values, Xt_np, n_features)
        lg.info("SHAP via shap.Explainer(est, masker=bg_np) succeeded.")
        shap_mat = cand
    except Exception as e1:
        lg.info(f"shap.Explainer failed: {e1}; fallback to TreeExplainer(raw).")
        shap_mat = None

    if shap_mat is None:
        shap_mat = _force_tree_explainer(est, Xt_np, n_features)

    lg.info(f"SHAP matrix shape: {shap_mat.shape}, Xt_np shape: {Xt_np.shape}, n_features={n_features}")
    assert shap_mat.ndim == 2, f"expect 2D shap matrix, got {shap_mat.shape}"
    assert shap_mat.shape[0] == Xt_np.shape[0], "Feature and SHAP matrices must have the same number of rows!"
    assert shap_mat.shape[1] == n_features, "SHAP values shape mismatch with feature names"

    # ---- 全局输出 ----
    vals = np.abs(shap_mat)
    mean_abs = np.ravel(vals.mean(axis=0))
    df_imp = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    out_csv = OUT_REPS / f"shap_values_top_h{args.horizon}_{args.split}.csv"
    df_imp.to_csv(out_csv, index=False)

    # Beeswarm
    plt.figure(figsize=(10, 7))
    try:
        exp = shap.Explanation(values=shap_mat, base_values=None, data=Xt_np, feature_names=feature_names)
        shap.plots.beeswarm(exp, show=False, max_display=30)
    except Exception:
        shap.summary_plot(shap_mat, Xt_np, feature_names=feature_names, show=False, max_display=30)
    out_bee = OUT_FIGS / f"shap_global_beeswarm_h{args.horizon}_{args.split}.png"
    plt.tight_layout(); plt.savefig(out_bee, dpi=200); plt.close()

    # Bar
    plt.figure(figsize=(9, 7))
    topn = df_imp.head(30)
    plt.barh(topn["feature"][::-1], topn["mean_abs_shap"][::-1])
    plt.xlabel("mean(|SHAP|)"); plt.title(f"Global importance (top 30) h={args.horizon} split={args.split}")
    out_bar = OUT_FIGS / f"shap_global_bar_h{args.horizon}_{args.split}.png"
    plt.tight_layout(); plt.savefig(out_bar, dpi=200); plt.close()

    # ---- 个体输出 ----
    k = min(args.top_k_individual, shap_mat.shape[0])
    for rank in range(k):
        try:
            plt.figure(figsize=(8, 6))
            exp_i = shap.Explanation(values=shap_mat[rank], base_values=0.0, data=Xt_np[rank], feature_names=feature_names)
            shap.plots.waterfall(exp_i, show=False, max_display=20)
        except Exception:
            order = np.argsort(-np.abs(shap_mat[rank]))[:20]
            vals_i = shap_mat[rank, order]; names = [feature_names[j] for j in order]
            plt.barh(names[::-1], vals_i[::-1]); plt.title(f"Sample rank#{rank+1}")
        out_ind = OUT_FIGS / f"shap_waterfall_h{args.horizon}_{args.split}_rank{rank+1}.png"
        plt.tight_layout(); plt.savefig(out_ind, dpi=200); plt.close()

    meta = {
        "model_path": str(model_path),
        "estimator": type(est).__name__,
        "n_explained": int(shap_mat.shape[0]),
        "top_k_individual": int(k),
        "global_csv": str(out_csv),
        "beeswarm_png": str(out_bee),
        "bar_png": str(out_bar)
    }
    lg.info(json.dumps(meta, ensure_ascii=False))

if __name__=="__main__":
    main()
