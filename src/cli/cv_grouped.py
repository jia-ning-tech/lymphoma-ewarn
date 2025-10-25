from __future__ import annotations
import argparse, logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"

lg = logging.getLogger("cli.cv_grouped")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

def load(h:int)->pd.DataFrame:
    p=DATAI/f"trainset_h{h}.parquet"
    assert p.exists(), f"not found: {p}"
    df=pd.read_parquet(p)
    for c in ("index_time","first_event_time"):
        if c in df.columns:
            df[c]=pd.to_datetime(df[c],errors="coerce")
    return df

def select_num_cols(df:pd.DataFrame, nan_threshold:float=0.95)->list[str]:
    # 仅数值列，排除ID/时间/标签
    drop=ID_TIME_LABEL & set(df.columns)
    num_cols=[c for c in df.columns if c not in drop and pd.api.types.is_numeric_dtype(df[c])]
    # 缺失率过滤
    keep=[]
    for c in num_cols:
        s=df[c]
        if s.isna().mean()>=nan_threshold: 
            continue
        # 零方差过滤（忽略缺失）
        v=s.dropna().values
        if v.size==0: 
            continue
        if np.nanstd(v)==0:
            continue
        keep.append(c)
    return keep

def run_cv(h:int, n_splits:int, random_state:int):
    df=load(h)
    y=df["label"].astype(int).values
    groups=df["stay_id"].values
    gkf=GroupKFold(n_splits=n_splits)

    aurocs, aps=[],[]
    fold=0
    for tr_idx, va_idx in gkf.split(df, y, groups):
        fold+=1
        dtr, dva=df.iloc[tr_idx].copy(), df.iloc[va_idx].copy()

        feat_cols=select_num_cols(dtr, nan_threshold=0.95)
        Xtr=dtr[feat_cols]
        Xva=dva.reindex(columns=feat_cols)
        ytr=dtr["label"].astype(int).values
        yva=dva["label"].astype(int).values

        pipe=Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
            ("clf", RandomForestClassifier(
                n_estimators=400,
                max_depth=None,
                n_jobs=-1,
                random_state=random_state,
                class_weight="balanced_subsample"
            )),
        ])
        pipe.fit(Xtr, ytr)
        pva=pipe.predict_proba(Xva)[:,1]
        auroc=roc_auc_score(yva, pva)
        ap=average_precision_score(yva, pva)
        aurocs.append(auroc); aps.append(ap)
        lg.info(f"[h={h}] fold={fold} | AUROC={auroc:.4f} | AP={ap:.4f} | feats={len(feat_cols)} | n_tr={len(tr_idx)} | n_va={len(va_idx)}")

    lg.info(f"[h={h}] GroupKFold={n_splits} | AUROC mean={np.mean(aurocs):.4f}±{np.std(aurocs):.4f} | AP mean={np.mean(aps):.4f}±{np.std(aps):.4f}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args=ap.parse_args()
    run_cv(h=args.horizon, n_splits=args.folds, random_state=args.seed)

if __name__=="__main__":
    main()
