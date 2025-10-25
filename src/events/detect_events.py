from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import pandas as pd

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat, progress_bar


def _read_event_itemids(path: str) -> Dict[str, Dict[str, list]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flatten_ids(m: Dict[str, list]) -> List[int]:
    out = []
    for _, ids in m.items():
        out.extend(int(x) for x in ids)
    # 去重排序
    return sorted({int(x) for x in out})


def _ensure_duckdb():
    try:
        import duckdb  # noqa
        return True
    except Exception:
        return False


def _detect_from_inputevents_duckdb(input_csv: str, stay_ids: List[int], itemids: List[int]) -> pd.DataFrame:
    """
    在 icu/inputevents.csv 里，检出 (stay_id, first_time)：
      条件：stay_id in cohort & itemid in 目标升压药 & (rate>0 OR amount>0)
      时间：starttime 最早值
    返回列：['stay_id', 'vaso_time']
    """
    import duckdb
    q = r"""
    WITH src AS (
      SELECT
        stay_id,
        itemid,
        CAST(starttime AS TIMESTAMP) AS starttime,
        -- 以下列可能不存在；用 COALESCE 兜底到 0
        COALESCE(CAST(rate AS DOUBLE), 0.0)   AS rate,
        COALESCE(CAST(amount AS DOUBLE), 0.0) AS amount
      FROM read_csv_auto(?, HEADER=TRUE)
      WHERE stay_id IN (SELECT * FROM __stay_ids)
        AND itemid  IN (SELECT * FROM __item_ids)
    )
    SELECT stay_id, MIN(starttime) AS vaso_time
    FROM src
    WHERE (rate > 0 OR amount > 0)
      AND starttime IS NOT NULL
    GROUP BY stay_id
    """
    con = duckdb.connect(database=":memory:")
    con.register("__stay_ids", pd.DataFrame({"stay_id": stay_ids}))
    con.register("__item_ids", pd.DataFrame({"itemid": itemids}))
    df = con.execute(q, [input_csv]).fetch_df()
    con.close()
    return df


def _detect_from_procedureevents_duckdb(proc_csv: str, stay_ids: List[int], itemids: List[int],
                                        out_col: str) -> pd.DataFrame:
    """
    在 icu/procedureevents.csv 里，检出 (stay_id, first_time)：
      条件：stay_id in cohort & itemid in 目标集合
      时间：starttime 最早值
    out_col: 命名为 'imv_time' 或 'crrt_time'
    """
    import duckdb
    q = r"""
    SELECT
      stay_id,
      MIN(CAST(starttime AS TIMESTAMP)) AS first_time
    FROM read_csv_auto(?, HEADER=TRUE)
    WHERE stay_id IN (SELECT * FROM __stay_ids)
      AND itemid  IN (SELECT * FROM __item_ids)
    GROUP BY stay_id
    """
    con = duckdb.connect(database=":memory:")
    con.register("__stay_ids", pd.DataFrame({"stay_id": stay_ids}))
    con.register("__item_ids", pd.DataFrame({"itemid": itemids}))
    df = con.execute(q, [proc_csv]).fetch_df()
    con.close()
    if not df.empty:
        df = df.rename(columns={"first_time": out_col})
    else:
        df = pd.DataFrame(columns=["stay_id", out_col])
    return df


def detect_events_for_cohort(save_csv: bool = True) -> str:
    """
    主入口：读取 cohort + 事件 ITEMID 字典，用 DuckDB 在 CSV 上检出
    每个 stay_id 的首次事件时间：vaso_time, imv_time, crrt_time, first_event_time。
    落盘：data_interim/events_first.csv
    """
    cfg = get_cfg()
    lg = get_logger("events.detect")

    if not _ensure_duckdb():
        raise RuntimeError("duckdb 未安装。请先: pip install --user duckdb")

    # 路径
    cohort_path = os.path.join(cfg.paths.interim, "cohort.parquet")
    dict_json = os.path.join(cfg.paths.outputs, "artifacts", "event_itemids.json")
    input_csv = os.path.join(cfg.paths.raw_icu, "inputevents.csv")
    proc_csv  = os.path.join(cfg.paths.raw_icu, "procedureevents.csv")
    out_csv   = os.path.join(cfg.paths.interim, "events_first.csv")

    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Cohort not found: {cohort_path} (先运行 build_cohort)")
    if not os.path.exists(dict_json):
        raise FileNotFoundError(f"Dictionary not found: {dict_json} (先运行 src.events.dictionaries)")
    if not os.path.exists(input_csv):
        lg.warning(f"Missing ICU inputevents.csv: {input_csv} (将跳过升压药检测)")
    if not os.path.exists(proc_csv):
        lg.warning(f"Missing ICU procedureevents.csv: {proc_csv} (将跳过 IMV/CRRT 检测)")

    # 载入 cohort 与字典
    cohort = pd.read_parquet(cohort_path, columns=["stay_id"])
    stay_ids = cohort["stay_id"].astype("int64").tolist()
    dct = _read_event_itemids(dict_json)
    vaso_ids = _flatten_ids(dct.get("vasopressors", {}))
    imv_ids  = _flatten_ids(dct.get("imv", {}))
    crrt_ids = _flatten_ids(dct.get("crrt", {}))

    lg.info(f"cohort stays={len(stay_ids)} | vaso_itemids={len(vaso_ids)} | imv_itemids={len(imv_ids)} | crrt_itemids={len(crrt_ids)}")

    # 检测
    dfs = [cohort.copy()]
    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="detect events"):
        # 升压药（inputevents）
        if os.path.exists(input_csv) and vaso_ids:
            try:
                df_vaso = _detect_from_inputevents_duckdb(input_csv, stay_ids, vaso_ids)
                dfs.append(df_vaso)
                lg.info(f"vaso detected: {len(df_vaso)} stays")
            except Exception as e:
                lg.warning(f"vaso detection failed: {e}")
        else:
            lg.info("Skip vaso detection (no file or no ids)")

        # IMV（procedureevents）
        if os.path.exists(proc_csv) and imv_ids:
            try:
                df_imv = _detect_from_procedureevents_duckdb(proc_csv, stay_ids, imv_ids, out_col="imv_time")
                dfs.append(df_imv)
                lg.info(f"imv detected: {len(df_imv)} stays")
            except Exception as e:
                lg.warning(f"imv detection failed: {e}")
        else:
            lg.info("Skip imv detection (no file or no ids)")

        # CRRT（procedureevents）
        if os.path.exists(proc_csv) and crrt_ids:
            try:
                df_crrt = _detect_from_procedureevents_duckdb(proc_csv, stay_ids, crrt_ids, out_col="crrt_time")
                dfs.append(df_crrt)
                lg.info(f"crrt detected: {len(df_crrt)} stays")
            except Exception as e:
                lg.warning(f"crrt detection failed: {e}")
        else:
            lg.info("Skip crrt detection (no file or no ids)")

    # 合并与 first_event_time
    from functools import reduce
    def _merge(left, right):
        return left.merge(right, on="stay_id", how="left")

    merged = reduce(_merge, dfs)
    # 计算最早事件时间
    for col in ["vaso_time", "imv_time", "crrt_time"]:
        if col in merged.columns:
            merged[col] = pd.to_datetime(merged[col], errors="coerce")

    merged["first_event_time"] = merged[["vaso_time", "imv_time", "crrt_time"]].min(axis=1, skipna=True)

    # 落盘
    if save_csv:
        os.makedirs(cfg.paths.interim, exist_ok=True)
        merged.to_csv(out_csv, index=False)
        lg.info(f"events_first saved: {out_csv} | rows={len(merged)}")

    return out_csv


if __name__ == "__main__":
    path = detect_events_for_cohort(save_csv=True)
    print(path)
