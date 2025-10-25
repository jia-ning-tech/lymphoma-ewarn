from __future__ import annotations
import argparse, json, logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"
OUT = ROOT / "outputs" / "reports"
OUT.mkdir(parents=True, exist_ok=True)

lg = logging.getLogger("cli.cohort_stats")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

ID_TIME_LABEL = {"stay_id","hadm_id","index_time","first_event_time","label"}

# 可选的人口学/就诊列名（存在才统计）
DEMOGRAPHIC_CANDIDATES = {
    "age": ["age","anchor_age","age_years","age_at_admission"],
    "sex": ["sex","gender","GENDER","Sex"],
    "icu": ["icu_type","first_careunit","careunit","ICUType"],
}

def _find_col(df: pd.DataFrame, options: List[str]) -> str | None:
    for c in options:
        if c in df.columns:
            return c
    return None

def _load(h:int) -> pd.DataFrame:
    p = DATAI / f"trainset_h{h}.parquet"
    assert p.exists(), f"not found: {p}"
    df = pd.read_parquet(p)
    for c in ("index_time","first_event_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    if "label" in df.columns:
        df["label"] = df["label"].astype(int)
    return df

def _basic_counts(df: pd.DataFrame) -> Dict:
    n_rows = len(df)
    n_stay = df["stay_id"].nunique() if "stay_id" in df.columns else None
    n_hadm = df["hadm_id"].nunique() if "hadm_id" in df.columns else None
    pos_rate = float(df["label"].mean()) if "label" in df.columns else None
    time_min = pd.to_datetime(df["index_time"]).min() if "index_time" in df.columns else None
    time_max = pd.to_datetime(df["index_time"]).max() if "index_time" in df.columns else None
    return {
        "n_rows": int(n_rows),
        "n_unique_stay": int(n_stay) if n_stay is not None else None,
        "n_unique_hadm": int(n_hadm) if n_hadm is not None else None,
        "positive_rate_window": pos_rate,
        "index_time_min": str(time_min) if time_min is not None else None,
        "index_time_max": str(time_max) if time_max is not None else None,
    }

def _stay_level(df: pd.DataFrame) -> Dict:
    if not {"stay_id","label"}.issubset(df.columns):
        return {}
    g = df.groupby("stay_id", as_index=False).agg(
        stay_label=("label","max"),
        n_windows=("label","size"),
        first_index_time=("index_time","min"),
        last_index_time=("index_time","max"),
    )
    return {
        "n_stay": int(len(g)),
        "positive_rate_stay": float(g["stay_label"].mean()),
        "median_windows_per_stay": float(g["n_windows"].median()),
        "p10_p90_windows_per_stay": [float(g["n_windows"].quantile(0.1)), float(g["n_windows"].quantile(0.9))],
        "stay_time_span_hours_median": float(((g["last_index_time"] - g["first_index_time"]).dt.total_seconds()/3600.0).median()) if "first_index_time" in g and "last_index_time" in g else None,
    }

def _demographics(df: pd.DataFrame) -> Dict:
    out = {}
    age_c = _find_col(df, DEMOGRAPHIC_CANDIDATES["age"])
    sex_c = _find_col(df, DEMOGRAPHIC_CANDIDATES["sex"])
    icu_c = _find_col(df, DEMOGRAPHIC_CANDIDATES["icu"])
    if age_c:
        age = pd.to_numeric(df[age_c], errors="coerce")
        out["age"] = {
            "n": int(age.notna().sum()),
            "mean": float(age.mean()),
            "std": float(age.std(ddof=0)),
            "p25": float(age.quantile(0.25)),
            "p50": float(age.median()),
            "p75": float(age.quantile(0.75)),
        }
    if sex_c:
        vc = df[sex_c].astype(str).str.upper().value_counts(dropna=False)
        out["sex_counts"] = {str(k): int(v) for k,v in vc.items()}
    if icu_c:
        vc = df[icu_c].astype(str).value_counts(dropna=False).head(20)
        out["icu_top20_counts"] = {str(k): int(v) for k,v in vc.items()}
    return out

def _missingness(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ID_TIME_LABEL]
    miss = df[cols].isna().mean().rename("missing_rate").to_frame()
    return miss.sort_values("missing_rate", ascending=False)

def _numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c not in ID_TIME_LABEL and pd.api.types.is_numeric_dtype(df[c])]
    desc = df[cols].describe(percentiles=[.1,.25,.5,.75,.9]).T
    # 附加缺失率
    desc["missing_rate"] = df[cols].isna().mean()
    return desc.sort_values("missing_rate", ascending=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, required=True, choices=[24,48])
    ap.add_argument("--split", choices=["train","val","test","all"], default="all",
                    help="统计哪个切分（默认 all：直接读 trainset_h{h}.parquet 全集）")
    args = ap.parse_args()

    df = _load(args.horizon)

    # 如果用户希望按 split 统计，但当前文件是全集级（我们未在此脚本再切分；由上游形成）
    # 这里将 split 字段仅作为 tag 写入输出，数值按全集计算。
    basic = _basic_counts(df)
    stay = _stay_level(df)
    demo = _demographics(df)

    miss = _missingness(df)
    numsum = _numeric_summary(df)

    tag = f"h{args.horizon}_{args.split}"
    miss_csv = OUT / f"cohort_missingness_{tag}.csv"
    numsum_csv = OUT / f"cohort_numeric_summary_{tag}.csv"
    miss.to_csv(miss_csv)
    numsum.to_csv(numsum_csv)

    report = {
        "horizon": args.horizon,
        "split_tag": args.split,
        "basic_counts": basic,
        "stay_level": stay,
        "demographics": demo,
        "files": {
            "missingness_csv": str(miss_csv),
            "numeric_summary_csv": str(numsum_csv),
        },
    }
    js = OUT / f"cohort_stats_{tag}.json"
    js.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    lg.info(json.dumps(report, ensure_ascii=False))
    print(str(js))
    print(str(miss_csv))
    print(str(numsum_csv))

if __name__ == "__main__":
    main()
