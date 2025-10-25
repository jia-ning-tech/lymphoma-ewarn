from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import pandas as pd

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat

# 与 extract_vitals_labs 中一致的关键词（保持口径）
VITAL_KEYWORDS: Dict[str, List[str]] = {
    "heart_rate": ["Heart Rate"],
    "resp_rate": ["Respiratory Rate"],
    "spo2": ["SpO2", "O2 saturation", "SaO2"],
    "temperature": ["Temperature Fahrenheit", "Temperature Celsius", "Temperature"],
    "sbp": ["Non Invasive Blood Pressure systolic", "Invasive BP Systolic", "Arterial Blood Pressure systolic"],
    "dbp": ["Non Invasive Blood Pressure diastolic", "Invasive BP Diastolic", "Arterial Blood Pressure diastolic"],
    "mbp": ["Non Invasive Blood Pressure mean", "Invasive BP Mean", "Arterial Blood Pressure mean"],
    "fio2": ["FiO2"],
    "peep": ["PEEP"],
}
LAB_KEYWORDS: Dict[str, List[str]] = {
    "lactate": ["Lactate"],
    "ph": ["pH"],
    "paco2": ["pCO2"],
    "pao2": ["pO2"],
    "hco3": ["Bicarbonate"],
    "na": ["Sodium"],
    "k": ["Potassium"],
    "cl": ["Chloride"],
    "ca": ["Calcium"],
    "bun": ["Urea Nitrogen", "BUN"],
    "creatinine": ["Creatinine"],
    "ast": ["AST"],
    "alt": ["ALT"],
    "tbil": ["Bilirubin, Total", "Total Bilirubin"],
    "alb": ["Albumin"],
    "inr": ["INR"],
    "pt": ["PT"],
    "aptt": ["aPTT", "APTT"],
    "wbc": ["WBC"],
    "hgb": ["Hemoglobin"],
    "platelet": ["Platelet", "Platelets"],
}


def _ensure_dirs():
    cfg = get_cfg()
    os.makedirs(cfg.paths.outputs, exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.outputs, "artifacts"), exist_ok=True)


def _discover_itemids_d_items(cfg, patterns: Dict[str, List[str]]) -> Dict[str, List[int]]:
    lg = get_logger("debug.extract")
    p = os.path.join(cfg.paths.raw_icu, "d_items.csv")
    di = pd.read_csv(p)
    di["label"] = di["label"].astype(str).str.strip().str.lower()
    out: Dict[str, List[int]] = {}
    for var, kws in patterns.items():
        mask = pd.Series(False, index=di.index)
        for kw in kws:
            mask |= di["label"].str.contains(str(kw).lower(), na=False)
        ids = di.loc[mask, "itemid"].dropna().astype(int).tolist()
        out[var] = sorted(set(ids))
        lg.info(f"[discover] chartevents {var}: {len(ids)} itemids")
    return out


def _discover_itemids_d_labitems(cfg, patterns: Dict[str, List[str]]) -> Dict[str, List[int]]:
    lg = get_logger("debug.extract")
    p = os.path.join(cfg.paths.raw_hosp, "d_labitems.csv")
    dl = pd.read_csv(p)
    dl["label"] = dl["label"].astype(str).str.strip().str.lower()
    out: Dict[str, List[int]] = {}
    for var, kws in patterns.items():
        mask = pd.Series(False, index=dl.index)
        for kw in kws:
            mask |= dl["label"].str.contains(str(kw).lower(), na=False)
        ids = dl.loc[mask, "itemid"].dropna().astype(int).tolist()
        out[var] = sorted(set(ids))
        lg.info(f"[discover] labevents {var}: {len(ids)} itemids")
    return out


def _pick_sample_from_cohort(stay_id: Optional[int], hadm_id: Optional[int]) -> Tuple[int, int, pd.Timestamp, pd.Timestamp]:
    cfg = get_cfg()
    cohort_p = os.path.join(cfg.paths.interim, "cohort.parquet")
    df = pd.read_parquet(cohort_p)
    df["icu_in"] = pd.to_datetime(df["icu_in"], errors="coerce")
    df["icu_out"] = pd.to_datetime(df["icu_out"], errors="coerce")
    row = None
    if stay_id is not None:
        row = df.loc[df["stay_id"] == stay_id].head(1)
    elif hadm_id is not None:
        row = df.loc[df["hadm_id"] == hadm_id].head(1)
    else:
        row = df.head(1)
    if row is None or len(row) == 0:
        raise RuntimeError("Cannot locate a sample row from cohort.")
    r = row.iloc[0]
    return int(r.stay_id), int(r.hadm_id), pd.Timestamp(r.icu_in), pd.Timestamp(r.icu_out)


def _conn() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(database=":memory:")


def _save_preview(df: pd.DataFrame, name: str, n: int = 10) -> str:
    cfg = get_cfg()
    out = os.path.join(cfg.paths.outputs, "artifacts", f"debug_{name}.csv")
    (df.head(n) if len(df) > 0 else df).to_csv(out, index=False)
    return out


def _read_head_any(conn, path: str, limit: int) -> pd.DataFrame:
    conn.execute(f"SELECT * FROM read_csv_auto('{path}', ALL_VARCHAR=FALSE) LIMIT {limit}")
    return conn.fetch_df()


def _read_selective(conn, path: str, where: str, limit: int) -> pd.DataFrame:
    conn.execute(f"SELECT * FROM read_csv_auto('{path}', ALL_VARCHAR=FALSE) {where} LIMIT {limit}")
    return conn.fetch_df()


def _normalize_columns(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    统一列名：返回只保留以下列的副本，并将值列命名为 'value'。
    kind: 'chartevents' or 'labevents'
    """
    cols = {c.lower(): c for c in df.columns}
    # 时间列：优先 charttime，其次 storetime
    tcol = None
    for cand in ["charttime", "chart_time", "chart_time_utc", "storetime", "store_time"]:
        if cand in cols:
            tcol = cols[cand]; break
    # 值列：优先 valuenum，其次 value
    vcol = None
    for cand in ["valuenum", "value", "valuetext", "value_num"]:
        if cand in cols:
            vcol = cols[cand]; break
    # id 列
    itemid_col = cols.get("itemid", None)
    stay_col = cols.get("stay_id", None)
    hadm_col = cols.get("hadm_id", None)

    keep = []
    if stay_col: keep.append(stay_col)
    if hadm_col: keep.append(hadm_col)
    if tcol: keep.append(tcol)
    if itemid_col: keep.append(itemid_col)
    if vcol: keep.append(vcol)
    if not keep:
        return pd.DataFrame()

    out = df[keep].copy()
    if tcol:
        out.rename(columns={tcol: "charttime"}, inplace=True)
    if vcol:
        out.rename(columns={vcol: "value"}, inplace=True)
    if itemid_col:
        out.rename(columns={itemid_col: "itemid"}, inplace=True)
    return out


def main():
    parser = argparse.ArgumentParser(description="Debug extractor for vitals/labs")
    parser.add_argument("--stay-id", type=int, default=None)
    parser.add_argument("--hadm-id", type=int, default=None)
    parser.add_argument("--n-rows", type=int, default=20, help="preview rows per probe")
    parser.add_argument("--pad-hours", type=int, default=6, help="window pad around ICU in/out")
    args = parser.parse_args()

    cfg = get_cfg()
    lg = get_logger("debug.extract")
    _ensure_dirs()

    stay_id, hadm_id, icu_in, icu_out = _pick_sample_from_cohort(args.stay_id, args.hadm_id)
    lg.info(f"Using sample: stay_id={stay_id} hadm_id={hadm_id} | icu_in={icu_in} icu_out={icu_out}")

    vit_map = _discover_itemids_d_items(cfg, VITAL_KEYWORDS)
    lab_map = _discover_itemids_d_labitems(cfg, LAB_KEYWORDS)
    vit_itemids = sorted({iid for ids in vit_map.values() for iid in ids})
    lab_itemids = sorted({iid for ids in lab_map.values() for iid in ids})
    lg.info(f"Total vitals itemids={len(vit_itemids)} | labs itemids={len(lab_itemids)}")

    conn = _conn()
    chartevents_p = os.path.join(cfg.paths.raw_icu, "chartevents.csv")
    labevents_p = os.path.join(cfg.paths.raw_hosp, "labevents.csv")

    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="debug-extract"):
        # --------- 1) HEAD (no filter) ----------
        try:
            c0 = _read_head_any(conn, chartevents_p, args.n_rows)
            c0n = _normalize_columns(c0, "chartevents")
            lg.info(f"[chartevents head] rows={len(c0)} | normalized_cols={list(c0n.columns)}")
            lg.info(f"saved -> {_save_preview(c0n, 'chartevents_head', args.n_rows)}")
        except Exception as e:
            lg.warning(f"chartevents head error: {e}")
            c0n = pd.DataFrame()

        try:
            l0 = _read_head_any(conn, labevents_p, args.n_rows)
            l0n = _normalize_columns(l0, "labevents")
            lg.info(f"[labevents head] rows={len(l0)} | normalized_cols={list(l0n.columns)}")
            lg.info(f"saved -> {_save_preview(l0n, 'labevents_head', args.n_rows)}")
        except Exception as e:
            lg.warning(f"labevents head error: {e}")
            l0n = pd.DataFrame()

        # --------- 2) only itemid filter ----------
        if vit_itemids:
            try:
                c1 = _read_selective(conn, chartevents_p, f"WHERE itemid IN ({','.join(map(str, vit_itemids[:50]))})", args.n_rows)
                c1n = _normalize_columns(c1, "chartevents")
                lg.info(f"[chartevents itemid] rows={len(c1)} | normalized_cols={list(c1n.columns)}")
                lg.info(f"saved -> {_save_preview(c1n, 'chartevents_itemid', args.n_rows)}")
            except Exception as e:
                lg.warning(f"chartevents itemid error: {e}")

        if lab_itemids:
            try:
                l1 = _read_selective(conn, labevents_p, f"WHERE itemid IN ({','.join(map(str, lab_itemids[:50]))})", args.n_rows)
                l1n = _normalize_columns(l1, "labevents")
                lg.info(f"[labevents itemid] rows={len(l1)} | normalized_cols={list(l1n.columns)}")
                lg.info(f"saved -> {_save_preview(l1n, 'labevents_itemid', args.n_rows)}")
            except Exception as e:
                lg.warning(f"labevents itemid error: {e}")

        # --------- 3) only stay/hadm filter ----------
        try:
            c2 = _read_selective(conn, chartevents_p, f"WHERE stay_id = {stay_id}", args.n_rows)
            c2n = _normalize_columns(c2, "chartevents")
            lg.info(f"[chartevents stay_id] rows={len(c2)} | normalized_cols={list(c2n.columns)}")
            lg.info(f"saved -> {_save_preview(c2n, 'chartevents_stay', args.n_rows)}")
        except Exception as e:
            lg.warning(f"chartevents stay filter error: {e}")

        try:
            l2 = _read_selective(conn, labevents_p, f"WHERE hadm_id = {hadm_id}", args.n_rows)
            l2n = _normalize_columns(l2, "labevents")
            lg.info(f"[labevents hadm_id] rows={len(l2)} | normalized_cols={list(l2n.columns)}")
            lg.info(f"saved -> {_save_preview(l2n, 'labevents_hadm', args.n_rows)}")
        except Exception as e:
            lg.warning(f"labevents hadm filter error: {e}")

        # --------- 4) stay/itemid + time window ----------
        win_start = icu_in - pd.Timedelta(hours=args.pad_hours)
        win_end   = icu_out + pd.Timedelta(hours=args.pad_hours)

        # 对 time 列做宽松兼容：在 SQL 里先不加时间，读出来后在 pandas 里根据 charttime/storetime 过滤
        try:
            if vit_itemids:
                c3 = _read_selective(conn, chartevents_p, f"WHERE stay_id = {stay_id} AND itemid IN ({','.join(map(str, vit_itemids[:200]))})", 5000)
            else:
                c3 = _read_selective(conn, chartevents_p, f"WHERE stay_id = {stay_id}", 5000)
            c3n = _normalize_columns(c3, "chartevents")
            if not c3n.empty and "charttime" in c3n.columns:
                c3n["charttime"] = pd.to_datetime(c3n["charttime"], errors="coerce")
                c3n = c3n.dropna(subset=["charttime"])
                c3n = c3n[(c3n["charttime"] >= win_start) & (c3n["charttime"] <= win_end)]
            lg.info(f"[chartevents stay+itemid | window={args.pad_hours}h pad] rows={len(c3n)} | cols={list(c3n.columns)}")
            lg.info(f"saved -> {_save_preview(c3n, 'chartevents_stay_itemid_window', args.n_rows)}")
        except Exception as e:
            lg.warning(f"chartevents stay+itemid window error: {e}")

        try:
            if lab_itemids:
                l3 = _read_selective(conn, labevents_p, f"WHERE hadm_id = {hadm_id} AND itemid IN ({','.join(map(str, lab_itemids[:200]))})", 5000)
            else:
                l3 = _read_selective(conn, labevents_p, f"WHERE hadm_id = {hadm_id}", 5000)
            l3n = _normalize_columns(l3, "labevents")
            if not l3n.empty and "charttime" in l3n.columns:
                l3n["charttime"] = pd.to_datetime(l3n["charttime"], errors="coerce")
                l3n = l3n.dropna(subset=["charttime"])
                l3n = l3n[(l3n["charttime"] >= win_start) & (l3n["charttime"] <= win_end)]
            lg.info(f"[labevents hadm+itemid | window={args.pad_hours}h pad] rows={len(l3n)} | cols={list(l3n.columns)}")
            lg.info(f"saved -> {_save_preview(l3n, 'labevents_hadm_itemid_window', args.n_rows)}")
        except Exception as e:
            lg.warning(f"labevents hadm+itemid window error: {e}")

    print("Done.")
    print("Artifacts saved under:", os.path.join(cfg.paths.outputs, "artifacts"))

if __name__ == "__main__":
    main()
