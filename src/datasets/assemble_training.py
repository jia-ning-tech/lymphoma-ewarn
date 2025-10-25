from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"

lg = logging.getLogger("datasets.assemble_training")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")

def _ensure_types(df: pd.DataFrame) -> pd.DataFrame:
    if "stay_id" in df.columns:
        df["stay_id"] = pd.to_numeric(df["stay_id"], errors="coerce").astype("Int64")
    if "index_time" in df.columns:
        df["index_time"] = pd.to_datetime(df["index_time"], errors="coerce")
    return df

def assemble_training(horizon: int, head: int = 10) -> Path:
    labels_p   = DATAI / "labels_all.parquet"
    features_p = DATAI / "features_all.parquet"
    assert labels_p.exists(), f"missing labels: {labels_p}"
    assert features_p.exists(), f"missing features: {features_p}"

    lg.info(f"loading labels: {labels_p}")
    lab = pd.read_parquet(labels_p)
    lab = _ensure_types(lab)
    # 只保留该 horizon 的标签
    lab = lab.loc[lab["horizon_hours"] == horizon, ["stay_id", "index_time", "label", "first_event_time"]]

    lg.info(f"loading features: {features_p}")
    feat = pd.read_parquet(features_p)
    feat = _ensure_types(feat)

    # 主键对齐：stay_id + index_time（两侧都已经在聚合阶段按小时对齐）
    lg.info("merging labels ⊕ features on ['stay_id','index_time'] ...")
    df = lab.merge(feat, on=["stay_id", "index_time"], how="left")

    # 基础清洗
    before = len(df)
    df = df.dropna(subset=["stay_id", "index_time", "label"])
    after = len(df)

    # 简要统计
    pos_rate = float(df["label"].mean()) if after > 0 else np.nan
    lg.info(f"assembled rows={after} (dropped {before-after}) | positives={df['label'].sum()} | rate={pos_rate:.4f}")

    # 预览（可选）
    if head:
        lg.info(f"[head h={horizon}]")
        with pd.option_context("display.max_columns", 12, "display.width", 160):
            lg.info("\n" + df.head(head).to_string(index=False))

    out_p = DATAI / f"trainset_h{horizon}.parquet"
    df.to_parquet(out_p, index=False)
    lg.info(f"trainset saved: {out_p} | rows={len(df)} | cols={df.shape[1]}")
    return out_p

def main():
    ap = argparse.ArgumentParser(description="Assemble training dataset by merging labels and features.")
    ap.add_argument("--horizon", type=int, required=True, help="horizon in hours, e.g., 24 / 48")
    ap.add_argument("--head", type=int, default=10, help="preview head rows")
    args = ap.parse_args()
    p = assemble_training(horizon=args.horizon, head=args.head)
    print(p)

if __name__ == "__main__":
    main()
