from __future__ import annotations

import os
import pandas as pd
from typing import List

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat, progress_bar


def _load_csv(path: str, usecols: List[str] | None = None, parse_dates: List[str] | None = None) -> pd.DataFrame:
    """
    Robust CSV reader (no chunks), explicit sep/encoding; safe for MIMIC tables.
    """
    kw = dict(sep=",", encoding="utf-8", low_memory=False)
    if usecols is not None:
        kw["usecols"] = usecols
    if parse_dates is not None:
        kw["parse_dates"] = parse_dates
    return pd.read_csv(path, **kw)


def build_and_save_cohort() -> str:
    cfg = get_cfg()
    lg = get_logger("cohort")

    paths = cfg.paths
    out_path = os.path.join(paths.interim, "cohort.parquet")
    os.makedirs(paths.interim, exist_ok=True)

    lg.info("Building lymphoma ICU cohort (adult, first ICU, ICU LOS>=24h, ICD lymphoma)...")
    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="cohort build"):

        # ---------- Load base tables ----------
        lg.info("Loading base tables...")

        p_cols = ["subject_id", "anchor_age", "gender"]  # patients
        a_cols = ["subject_id", "hadm_id", "admittime", "dischtime"]  # admissions
        i_cols = ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"]  # icustays
        d_cols = ["subject_id", "hadm_id", "icd_code", "icd_version"]  # diagnoses_icd
        di_cols = ["icd_code", "long_title"]  # d_icd_diagnoses (optional)

        patients = _load_csv(os.path.join(paths.raw_hosp, "patients.csv"), usecols=p_cols)
        admissions = _load_csv(os.path.join(paths.raw_hosp, "admissions.csv"), usecols=a_cols, parse_dates=["admittime", "dischtime"])
        icu = _load_csv(os.path.join(paths.raw_icu, "icustays.csv"), usecols=i_cols, parse_dates=["intime", "outtime"])
        diag = _load_csv(os.path.join(paths.raw_hosp, "diagnoses_icd.csv"), usecols=d_cols)
        dmap = _load_csv(os.path.join(paths.raw_hosp, "d_icd_diagnoses.csv"), usecols=di_cols)

        lg.info(f"Loaded: patients={len(patients)}, admissions={len(admissions)}, icustays={len(icu)}, diagnoses_icd={len(diag)}")

        # ---------- Adult filter ----------
        adults = patients[patients["anchor_age"] >= 18].copy()
        lg.info(f"Adults (anchor_age>=18): {len(adults)}")

        # ---------- Lymphoma ICD filter ----------
        icd9_prefix = tuple(cfg.cohort.lymphoma_icd.icd9_prefix)
        icd10_prefix = tuple(cfg.cohort.lymphoma_icd.icd10_prefix)

        # 某些版本 icd_version 读入为字符串，这里统一到数值比较
        iv = pd.to_numeric(diag["icd_version"], errors="coerce")
        icd_code = diag["icd_code"].astype(str)

        is_lymphoma = ((iv == 9) & icd_code.str.startswith(icd9_prefix)) | ((iv == 10) & icd_code.str.startswith(icd10_prefix))
        diag_lymph = diag.loc[is_lymphoma, ["subject_id", "hadm_id", "icd_code"]].drop_duplicates()
        lg.info(f"Lymphoma-positive admissions: {diag_lymph['hadm_id'].nunique()} hadm_ids")

        # （可选）连接长标题
        try:
            diag_lymph = diag_lymph.merge(dmap, on="icd_code", how="left")
        except Exception:
            pass

        # ---------- Join admissions with ICU stays ----------
        adm_lymph = admissions.merge(diag_lymph[["subject_id", "hadm_id"]].drop_duplicates(),
                                     on=["subject_id", "hadm_id"], how="inner")
        adm_lymph = adm_lymph.merge(adults[["subject_id", "anchor_age", "gender"]], on="subject_id", how="inner")

        stay = icu.merge(adm_lymph, on=["subject_id", "hadm_id"], how="inner")
        if stay.empty:
            raise RuntimeError("No ICU stays matched lymphoma admissions and adult filter.")

        # ---------- First ICU stay per subject (robust) ----------
        # 先按 subject_id + intime 排序，再取每个 subject 的第一条
        stay_sorted = stay.sort_values(["subject_id", "intime"])
        first_stay = stay_sorted.groupby("subject_id", as_index=False).head(1).copy()

        # ---------- ICU LOS >= 24h ----------
        los_hours = first_stay["los"] * 24.0
        if "outtime" in first_stay and "intime" in first_stay:
            los_hours = los_hours.fillna(((first_stay["outtime"] - first_stay["intime"]).dt.total_seconds() / 3600.0))
        first_stay["los_hours"] = los_hours

        cohort = first_stay[first_stay["los_hours"] >= float(cfg.cohort.min_icu_hours)].copy()
        lg.info(f"After LOS>= {cfg.cohort.min_icu_hours}h: {len(cohort)} stays")

        if cohort.empty:
            raise RuntimeError("After ICU LOS >= 24h filter, cohort is empty.")

        # ---------- Final columns ----------
        cohort = cohort.rename(columns={
            "intime": "icu_in",
            "outtime": "icu_out",
            "anchor_age": "age"
        })
        keep_cols = [
            "subject_id", "hadm_id", "stay_id",
            "icu_in", "icu_out", "los_hours",
            "age", "gender",
            "first_careunit", "last_careunit",
            "admittime", "dischtime",
        ]
        cohort = cohort[keep_cols].sort_values(["subject_id", "icu_in"]).reset_index(drop=True)

        # ---------- Save ----------
        cohort.to_parquet(out_path, index=False)
        lg.info(f"Cohort saved: {out_path} | n={len(cohort)}")

    return out_path


if __name__ == "__main__":
    path = build_and_save_cohort()
    print(path)
