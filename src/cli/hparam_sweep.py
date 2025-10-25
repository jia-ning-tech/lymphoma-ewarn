from __future__ import annotations
import argparse, itertools, json, logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier
import joblib

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_REPS = ROOT / "outputs" / "reports"
OUT_MODELS = ROOT / "outputs" / "models"
OUT_REPS.mkdir(parents=True, exist_ok=True)
OUT_MODELS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.hparam_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

def load(h:int)->pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found: {p}"
    df = pd.read_parquet(p)
    for c in ("index_time","first_event_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["label"] = df["label"].astype(int)
    return df

def select_num_cols(df: pd.DataFrame, nan_threshold: float=0.95) -> List[str]:
    drop = ID_TIME_LABEL & set(df.columns)
    num = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    keep=[]
    for c in num:
        s = df[c]
        if s.isna().mean() >= nan_threshold: 
            continue
        v = s.dropna().values
        if v.size == 0: 
            continue
        if np.nanstd(v)==0:
            continue
        keep.append(c)
    lg.info(f"feature select -> keep={len(keep)} / total_num={len(num)}")
    return keep

def parse_grid(grid_str: str) -> Dict[str, List]:
    """
    形如：
      "n_estimators=300,600;max_depth=None,12,20;max_features=sqrt,0.5;min_samples_leaf=1,2;class_weight=balanced,balanced_subsample"
    """
    grid: Dict[str, List] = {}
    for part in grid_str.split(";"):
        part = part.strip()
        if not part:
            continue
        k, vs = part.split("=", 1)
        k = k.strip()
        vals=[]
        for v in vs.split(","):
            v=v.strip()
            if v.lower()=="none":
                vals.append(None)
            else:
                # 尝试转成 int/float，否则保留字符串
                try:
                    if "." in v:
                        vals.append(float(v))
                    else:
                        vals.append(int(v))
                except ValueError:
                    vals.append(v)
        grid[k]=vals
    return grid

def product_dict(d: Dict[str,List]) -> List[Dict]:
    keys=list(d.keys())
    combos=[]
    for vals in itertools.product(*[d[k] for k in keys]):
        combos.append({k:v for k,v in zip(keys, vals)})
    return combos

def eval_one_cfg(X: np.ndarray, y: np.ndarray, groups, cfg: Dict, n_splits: int, seed: int):
    gkf = GroupKFold(n_splits=n_splits)
    aurocs, aps = [], []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups), 1):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", RandomForestClassifier(
                n_jobs=-1,
                random_state=seed,
                **cfg
            ))
        ])
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[va])[:,1]
        auc = roc_auc_score(y[va], p)
        ap  = average_precision_score(y[va], p)
        aurocs.append(auc); aps.append(ap)
        lg.info(f"fold={fold} | AUROC={auc:.4f} | AP={ap:.4f} | cfg={cfg}")
    res = {
        "AUROC_mean": float(np.mean(aurocs)),
        "AUROC_std":  float(np.std(aurocs)),
        "AP_mean":    float(np.mean(aps)),
        "AP_std":     float(np.std(aps)),
        "scores_per_fold": {"auroc": aurocs, "ap": aps}
    }
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--nan_threshold", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grid", type=str, default=(
        "n_estimators=300,600;"
        "max_depth=None,12,20;"
        "max_features=sqrt,0.5;"
        "min_samples_leaf=1,2;"
        "class_weight=balanced,balanced_subsample"
    ))
    ap.add_argument("--score_formula", type=str, default="0.7*AP_mean + 0.3*AUROC_mean",
                    help='用于选最优配置的表达式，变量可用：AP_mean, AUROC_mean')
    ap.add_argument("--save_topk", type=int, default=1, help="保存得分最高的前K个配置（依次训练全集并保存模型）")
    args = ap.parse_args()

    df = load(args.horizon)
    feat_cols = select_num_cols(df, nan_threshold=args.nan_threshold)
    X = df.reindex(columns=feat_cols).values
    y = df["label"].values
    groups = df["stay_id"].values if "stay_id" in df.columns else np.arange(len(df))

    grid = parse_grid(args.grid)
    combos = product_dict(grid)
    lg.info(f"total grid size = {len(combos)}")

    rows=[]
    for i, cfg in enumerate(combos, 1):
        res = eval_one_cfg(X, y, groups, cfg, n_splits=args.folds, seed=args.seed)
        row = {**cfg, **res}
        # 计算综合得分
        safe_locals = dict(AP_mean=row["AP_mean"], AUROC_mean=row["AUROC_mean"])
        try:
            score = eval(args.score_formula, {"__builtins__":{}}, safe_locals)
        except Exception:
            score = 0.7*row["AP_mean"] + 0.3*row["AUROC_mean"]
        row["score"] = float(score)
        rows.append(row)

    df_res = pd.DataFrame(rows).sort_values("score", ascending=False)
    out_csv = OUT_REPS / f"hparam_sweep_h{args.horizon}.csv"
    df_res.to_csv(out_csv, index=False)

    # 训练并保存 topk
    saved=[]
    for rank, (_, r) in enumerate(df_res.head(args.save_topk).iterrows(), 1):
        cfg = {k: r[k] for k in grid.keys()}
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", RandomForestClassifier(
                n_jobs=-1, random_state=args.seed, **cfg
            ))
        ])
        pipe.fit(X, y)
        bundle = {
            "pipeline": pipe,
            "features": feat_cols,
            "config": cfg,
            "seed": args.seed,
            "score": float(r["score"]),
            "AP_mean": float(r["AP_mean"]),
            "AUROC_mean": float(r["AUROC_mean"]),
            "cv_folds": args.folds,
        }
        out_p = OUT_MODELS / f"rf_hparam_best{rank}_h{args.horizon}.joblib"
        joblib.dump(bundle, out_p)
        saved.append(str(out_p))

    summary = {
        "horizon": args.horizon,
        "folds": args.folds,
        "grid_size": len(combos),
        "score_formula": args.score_formula,
        "results_csv": str(out_csv),
        "saved_models": saved,
        "top1": df_res.head(1).to_dict(orient="records")[0] if len(df_res)>0 else None
    }
    lg.info(json.dumps(summary, ensure_ascii=False))
    print(str(out_csv))
    for p in saved: print(p)

if __name__ == "__main__":
    main()
