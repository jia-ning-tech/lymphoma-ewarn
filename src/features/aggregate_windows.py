from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

ROOT = Path(__file__).resolve().parents[2]
DATAI = ROOT / "data_interim"

lg = logging.getLogger("features.aggregate")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# ---------------------------
# helpers
# ---------------------------

def _ensure_dt(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if not np.issubdtype(df[col].dtype, np.datetime64):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

def _ensure_int(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            if str(df[c].dtype) not in ("int64", "Int64"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    return df

def _floor_hour(ts: pd.Series) -> pd.Series:
    # 统一到小时，确保滚动窗口按整点计算
    return pd.to_datetime(ts).dt.floor("h")

def _hourly_matrix(ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：一个 stay 的时序 ts_df，列 = ['charttime','variable','value']
    输出：按小时聚合后的 wide 矩阵（index=小时，columns=变量，values=小时内平均）
    """
    if ts_df.empty:
        return pd.DataFrame()
    g = ts_df.groupby(["charttime", "variable"])["value"].mean().unstack("variable")
    g = g.sort_index()
    return g

def _window_stats(frame: pd.DataFrame, end_time: pd.Timestamp, window_h: int) -> Dict[str, float]:
    """
    在小时级矩阵 frame（index=小时），以 end_time 为“当前时点”，取过去 window_h 小时（含 end_time，不看未来）。
    计算每个变量的统计量，并返回扁平字典。
    """
    start_time = end_time - pd.Timedelta(hours=window_h) + pd.Timedelta(hours=1)
    win = frame.loc[start_time:end_time]
    out: Dict[str, float] = {}
    if win.empty:
        return out

    last_vals = win.ffill().iloc[-1]
    mean_vals = win.mean(numeric_only=True)
    min_vals  = win.min(numeric_only=True)
    max_vals  = win.max(numeric_only=True)
    std_vals  = win.std(numeric_only=True, ddof=0)
    cnt_vals  = win.count()

    for var in win.columns:
        prefix = f"{var}__{window_h}h"
        v_last = last_vals.get(var, np.nan)
        v_mean = mean_vals.get(var, np.nan)
        out[f"{prefix}__last"]  = v_last
        out[f"{prefix}__mean"]  = v_mean
        out[f"{prefix}__min"]   = min_vals.get(var, np.nan)
        out[f"{prefix}__max"]   = max_vals.get(var, np.nan)
        out[f"{prefix}__std"]   = std_vals.get(var, np.nan)
        out[f"{prefix}__count"] = float(cnt_vals.get(var, 0))
        out[f"{prefix}__trend_last_mean"] = (v_last - v_mean) if pd.notna(v_last) and pd.notna(v_mean) else np.nan

    return out

# ---------------------------
# main builder
# ---------------------------

def build_features(windows: List[int] = [6, 24],
                   labels_path: Path = DATAI / "labels_all.parquet",
                   vitals_path: Path = DATAI / "ts_vitals.parquet",
                   labs_path: Path = DATAI / "ts_labs.parquet",
                   head: int | None = None,
                   stats: bool = False) -> List[Path]:
    """
    在 labels_all 的 (stay_id, index_time) 点上，按照过去 windows 小时滚动，计算
    生命体征与化验的统计特征；返回各窗口与总表路径。
    """
    assert Path(labels_path).exists(), f"labels not found: {labels_path}"
    assert Path(vitals_path).exists(), f"vitals not found: {vitals_path}"
    assert Path(labs_path).exists(), f"labs not found: {labs_path}"

    labels = pd.read_parquet(labels_path, columns=["stay_id", "index_time"])
    labels = _ensure_int(labels, ["stay_id"])
    labels = _ensure_dt(labels, "index_time")
    labels = labels.dropna(subset=["stay_id", "index_time"]).drop_duplicates().reset_index(drop=True)

    # 读取时序 & 规范化
    vit = pd.read_parquet(vitals_path)
    lab = pd.read_parquet(labs_path)
    vit = _ensure_int(vit, ["stay_id", "hadm_id"])
    lab = _ensure_int(lab, ["stay_id", "hadm_id"])
    _ensure_dt(vit, "charttime")
    _ensure_dt(lab, "charttime")

    vit = vit[["stay_id", "charttime", "variable", "value"]].copy()
    lab = lab[["stay_id", "charttime", "variable", "value"]].copy()

    ts = pd.concat([vit, lab], ignore_index=True)
    ts = ts.dropna(subset=["stay_id", "charttime", "variable", "value"]).reset_index(drop=True)
    ts["charttime"] = _floor_hour(ts["charttime"])

    stay_ids = labels["stay_id"].dropna().astype("Int64").unique().tolist()
    out_by_win: Dict[int, List[pd.DataFrame]] = {w: [] for w in windows}

    lg.info(f"Building features on {len(stay_ids)} stays | windows={windows} ...")

    for sid in tqdm(stay_ids, desc="features per stay"):
        sid = int(sid)
        idx_times = (
            labels.loc[labels["stay_id"] == sid, "index_time"]
            .sort_values()
            .unique()
        )

        ts_s = ts.loc[ts["stay_id"] == sid, ["charttime", "variable", "value"]].copy()
        if ts_s.empty:
            for w in windows:
                out_by_win[w].append(pd.DataFrame({"stay_id": sid, "index_time": idx_times}))
            continue

        mat = _hourly_matrix(ts_s)
        if mat.empty:
            for w in windows:
                out_by_win[w].append(pd.DataFrame({"stay_id": sid, "index_time": idx_times}))
            continue

        mat = mat.sort_index()
        mat.index = pd.to_datetime(mat.index)

        for w in windows:
            rows = []
            for t in idx_times:
                t = pd.Timestamp(t).floor("h")
                stats_dict = _window_stats(mat, t, w)
                stats_dict["stay_id"] = sid
                stats_dict["index_time"] = t
                rows.append(stats_dict)
            feat_w = pd.DataFrame(rows)
            out_by_win[w].append(feat_w)

    saved_paths = []
    for w in windows:
        dfw = pd.concat(out_by_win[w], ignore_index=True)
        dfw = dfw.sort_values(["stay_id", "index_time"]).reset_index(drop=True)
        out_w = DATAI / f"features_{w}h.parquet"
        dfw.to_parquet(out_w, index=False)
        lg.info(f"features[{w}h] saved: {out_w} | rows={len(dfw)} | cols={len(dfw.columns)}")
        if head:
            lg.info(f"[head {w}h]\n{dfw.head(head).to_string(index=False)[:1000]}")
        saved_paths.append(out_w)

    # 合并所有窗口为宽表（按 key merge）
    base = None
    for w, p in zip(windows, saved_paths):
        dfw = pd.read_parquet(p)
        if base is None:
            base = dfw
        else:
            base = base.merge(dfw, on=["stay_id", "index_time"], how="outer")
    out_all = DATAI / "features_all.parquet"
    base = base.sort_values(["stay_id", "index_time"]).reset_index(drop=True)
    base.to_parquet(out_all, index=False)
    lg.info(f"features (ALL windows) saved: {out_all} | rows={len(base)} | cols={len(base.columns)}")
    saved_paths.append(out_all)

    if stats and base is not None:
        sample_cols = [c for c in base.columns if c.endswith("__mean")]
        if sample_cols:
            nn = base[sample_cols].notna().mean().sort_values(ascending=False).head(10).to_dict()
            lg.info(f"[stats] top non-null rates (mean features): {nn}")

    return saved_paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[6, 24],
                    help="rolling windows in hours, e.g., --windows 6 24 48")
    ap.add_argument("--head", type=int, default=None, help="show head rows after saving")
    ap.add_argument("--stats", action="store_true", help="print simple stats")
    args = ap.parse_args()

    paths = build_features(windows=args.windows, head=args.head, stats=args.stats)
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
