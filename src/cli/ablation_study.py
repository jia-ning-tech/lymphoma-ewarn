from __future__ import annotations
import argparse, logging, json, re, time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

# ---- tqdm（可选）----
try:
    from tqdm import tqdm
    _HAS_TQDM = True
except Exception:
    _HAS_TQDM = False
    class _DummyTQDM:
        def __init__(self, total=None, desc=None, unit=None, leave=False, disable=False):
            self.total = total
        def update(self, n=1): pass
        def set_postfix(self, **kw): pass
        def close(self): pass
        def __enter__(self): return self
        def __exit__(self, exc_type, exc, tb): pass
    def tqdm(*args, **kwargs):  # type: ignore
        return _DummyTQDM()

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT_REPS = ROOT / "outputs" / "reports"
OUT_REPS.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.ablation")
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

def select_num_cols(df:pd.DataFrame, nan_threshold:float=0.95)->List[str]:
    drop = ID_TIME_LABEL & set(df.columns)
    num = [c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    keep=[]
    for c in num:
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

def default_groups() -> Dict[str, List[str]]:
    return {
        "vitals":   [r"^(heart_rate|resp_rate|sbp|dbp|mbp|spo2|temperature)"],
        "labs":     [r"^(bun|creatinine|hgb|platelet|ca|cl|ph|pt)"],
        "vent":     [r"^(peep)"],
    }

def split_by_groups(cols: List[str], regex_groups: Dict[str,List[str]]) -> Dict[str, List[str]]:
    assigned = {k: [] for k in regex_groups}
    others = []
    for c in cols:
        hit=False
        for g, patterns in regex_groups.items():
            for pat in patterns:
                if re.search(pat, c):
                    assigned[g].append(c)
                    hit=True
                    break
            if hit: break
        if not hit:
            others.append(c)
    if others:
        assigned["others"] = others
    return assigned

def cv_score(X: np.ndarray, y: np.ndarray, groups, n_splits:int, seed:int, n_estimators:int,
             progress_tag:str="", use_tqdm:bool=True, log_each_fold:bool=True)->Dict[str,float]:
    gkf = GroupKFold(n_splits=n_splits)
    aucs, aps = [], []
    t0 = time.time()
    bar_desc = progress_tag or "cv"
    # 折内进度条
    with tqdm(total=n_splits, desc=bar_desc, unit="fold", leave=False, disable=not use_tqdm) as bar:
        for i, (tr, va) in enumerate(gkf.split(X, y, groups), start=1):
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler(with_mean=True, with_std=True)),
                ("clf", RandomForestClassifier(
                    n_estimators=n_estimators, n_jobs=-1, random_state=seed,
                    class_weight="balanced_subsample",
                ))
            ])
            fold_start = time.time()
            pipe.fit(X[tr], y[tr])
            p = pipe.predict_proba(X[va])[:,1]
            auc = roc_auc_score(y[va], p)
            ap  = average_precision_score(y[va], p)
            aucs.append(auc); aps.append(ap)

            # 心跳 + 进度条
            elapsed = time.time() - t0
            per_fold = elapsed / i
            eta = per_fold * (n_splits - i)
            if log_each_fold:
                lg.info(f"[{bar_desc}] fold {i}/{n_splits} done | AUROC={auc:.4f} AP={ap:.4f} | elapsed={elapsed/60:.1f}m ETA={eta/60:.1f}m")
            bar.set_postfix(AUROC=f"{auc:.3f}", AP=f"{ap:.3f}")
            bar.update(1)

    return {
        "AUROC_mean": float(np.mean(aucs)),
        "AUROC_std":  float(np.std(aucs)),
        "AP_mean":    float(np.mean(aps)),
        "AP_std":     float(np.std(aps)),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nan_threshold", type=float, default=0.95)
    ap.add_argument("--n_estimators", type=int, default=600)
    ap.add_argument("--mode", choices=["drop-one","keep-only","both"], default="both")
    ap.add_argument("--subsample", type=int, default=0, help="随机下采样到 N 行以加速（0=不采样）")
    ap.add_argument("--groups", type=str, default="")
    ap.add_argument("--quiet", action="store_true", help="不打印逐折心跳（仍会显示 tqdm）")
    ap.add_argument("--no_tqdm", action="store_true", help="禁用 tqdm 进度条（仅心跳日志）")
    args = ap.parse_args()

    df = load(args.horizon)
    if args.subsample and args.subsample < len(df):
        df = df.sample(args.subsample, random_state=args.seed).reset_index(drop=True)
        lg.info(f"subsample to n={len(df)}")

    feat_cols = select_num_cols(df, args.nan_threshold)
    groups = df["stay_id"].values if "stay_id" in df.columns else np.arange(len(df))
    y = df["label"].values

    # 分组规则
    if args.groups.strip():
        regex_groups: Dict[str,List[str]] = {}
        for block in args.groups.split(";"):
            if not block.strip(): continue
            name, pats = block.split(":",1)
            regex_groups[name.strip()] = [p.strip() for p in pats.split("|") if p.strip()]
    else:
        regex_groups = default_groups()

    group_cols = split_by_groups(feat_cols, regex_groups)
    lg.info("feature groups (counts): " + ", ".join([f"{k}={len(v)}" for k,v in group_cols.items()]))

    results = []
    use_tqdm = (not args.no_tqdm) and _HAS_TQDM

    # 基线：全部特征
    X_full = df.reindex(columns=feat_cols).values
    lg.info(f"[baseline_all] start | folds={args.folds} n_features={len(feat_cols)}")
    base = cv_score(X_full, y, groups, args.folds, args.seed, args.n_estimators,
                    progress_tag="baseline_all", use_tqdm=use_tqdm, log_each_fold=(not args.quiet))
    results.append({"setting":"baseline_all", "group":"-", **base, "n_features": len(feat_cols)})
    lg.info(f"[baseline_all] AUROC={base['AUROC_mean']:.4f}±{base['AUROC_std']:.4f} | AP={base['AP_mean']:.4f}±{base['AP_std']:.4f}")

    # 外层组别进度条
    items = list(group_cols.items())

    # drop-one
    if args.mode in ("drop-one","both"):
        lg.info(f"[drop-one] start | {len(items)} groups")
        with tqdm(total=len(items), desc="drop-one", unit="group", leave=False, disable=not use_tqdm) as outer:
            for g, cols in items:
                lg.info(f"[drop {g}] start | remove={len(cols)} keep={len(feat_cols)-len(cols)}")
                keep_cols = [c for c in feat_cols if c not in cols]
                if not keep_cols:
                    lg.info(f"[drop {g}] skip (no features left)")
                    outer.update(1); outer.set_postfix(group=g)
                    continue
                X = df.reindex(columns=keep_cols).values
                s = cv_score(X, y, groups, args.folds, args.seed, args.n_estimators,
                             progress_tag=f"drop-{g}", use_tqdm=use_tqdm, log_each_fold=(not args.quiet))
                results.append({"setting":"drop-one", "group":g, **s, "n_features": len(keep_cols)})
                lg.info(f"[drop {g}] AUROC={s['AUROC_mean']:.4f}±{s['AUROC_std']:.4f} | AP={s['AP_mean']:.4f}±{s['AP_std']:.4f}")
                outer.update(1); outer.set_postfix(group=g)

    # keep-only
    if args.mode in ("keep-only","both"):
        lg.info(f"[keep-only] start | {len(items)} groups")
        with tqdm(total=len(items), desc="keep-only", unit="group", leave=False, disable=not use_tqdm) as outer:
            for g, cols in items:
                lg.info(f"[keep {g}] start | keep={len(cols)}")
                if not cols:
                    lg.info(f"[keep {g}] skip (0 feature)")
                    outer.update(1); outer.set_postfix(group=g)
                    continue
                X = df.reindex(columns=cols).values
                s = cv_score(X, y, groups, args.folds, args.seed, args.n_estimators,
                             progress_tag=f"keep-{g}", use_tqdm=use_tqdm, log_each_fold=(not args.quiet))
                results.append({"setting":"keep-only", "group":g, **s, "n_features": len(cols)})
                lg.info(f"[keep {g}] AUROC={s['AUROC_mean']:.4f}±{s['AUROC_std']:.4f} | AP={s['AP_mean']:.4f}±{s['AP_std']:.4f}")
                outer.update(1); outer.set_postfix(group=g)

    df_res = pd.DataFrame(results)
    out_csv = OUT_REPS / f"ablation_h{args.horizon}.csv"
    df_res.to_csv(out_csv, index=False)

    summary = {
        "horizon": args.horizon,
        "folds": args.folds,
        "n_estimators": args.n_estimators,
        "subsample": args.subsample,
        "groups": {k: len(v) for k,v in group_cols.items()},
        "results_csv": str(out_csv),
        "baseline": df_res.loc[df_res["setting"]=="baseline_all"].to_dict(orient="records")[0],
        "top_by_AP": df_res.sort_values("AP_mean", ascending=False).head(3).to_dict(orient="records"),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
