from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import numpy as np

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat  # 去掉 progress_bar 以避免迭代器干扰


def _ensure_paths():
    cfg = get_cfg()
    os.makedirs(cfg.paths.interim, exist_ok=True)
    os.makedirs(cfg.paths.outputs, exist_ok=True)


def _load_cohort_events() -> pd.DataFrame:
    cfg = get_cfg()
    cohort_path = os.path.join(cfg.paths.interim, "cohort.parquet")
    events_path = os.path.join(cfg.paths.interim, "events_first.csv")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Missing cohort: {cohort_path}")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing events: {events_path}")

    cohort = pd.read_parquet(cohort_path, columns=["stay_id", "icu_in", "icu_out"])
    for c in ["icu_in", "icu_out"]:
        cohort[c] = pd.to_datetime(cohort[c], errors="coerce")

    events = pd.read_csv(events_path, usecols=["stay_id", "first_event_time"])
    events["first_event_time"] = pd.to_datetime(events["first_event_time"], errors="coerce")

    df = cohort.merge(events, on="stay_id", how="left")
    return df


def _ceil_to_next_hour(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    if ts.minute == 0 and ts.second == 0 and ts.microsecond == 0 and ts.nanosecond == 0:
        return ts
    return (ts.floor("1h") + pd.Timedelta(hours=1)).replace(minute=0, second=0, microsecond=0, nanosecond=0)


def _build_grid_by_hours(icu_in: pd.Timestamp,
                         icu_out: pd.Timestamp,
                         fev: Optional[pd.Timestamp],
                         start_after_hours: int,
                         exclude_after_event: bool):
    """返回 (grid, debug_dict)；grid 为 DatetimeIndex。"""
    start_dt = pd.Timestamp(icu_in) + pd.Timedelta(hours=start_after_hours)
    end_cap = pd.Timestamp(icu_out)
    fev_ts = pd.Timestamp(fev) if pd.notna(fev) else pd.NaT

    if exclude_after_event and pd.notna(fev_ts):
        if fev_ts >= start_dt and fev_ts < end_cap:
            end_cap = fev_ts

    if end_cap <= start_dt:
        return pd.DatetimeIndex([], dtype="datetime64[ns]"), {
            "start_dt": start_dt, "end_cap": end_cap, "note": "end_cap<=start_dt"
        }

    start = _ceil_to_next_hour(start_dt)
    hours = int(np.floor((end_cap - start).total_seconds() / 3600.0))

    if hours < 0:
        return pd.DatetimeIndex([], dtype="datetime64[ns]"), {
            "start_dt": start_dt, "start": start, "end_cap": end_cap, "hours": hours, "note": "hours<0"
        }

    grid = pd.date_range(start=start, periods=hours + 1, freq="1H")

    dbg = {
        "start_dt": start_dt,
        "start": start,
        "end_cap": end_cap,
        "fev": fev_ts if pd.notna(fev_ts) else None,
        "hours": hours,
        "grid_head": list(map(str, grid[:3])),
        "grid_len": len(grid),
    }
    return grid, dbg


def build_labels(horizons: Optional[List[int]] = None,
                 save_all: bool = True) -> List[str]:
    cfg = get_cfg()
    lg = get_logger("labeling")
    _ensure_paths()

    if horizons is None or len(horizons) == 0:
        horizons = list(map(int, cfg.prediction.get("horizons_hours", [24, 48])))

    start_after_hours = int(cfg.prediction.get("start_after_hours", 6))
    exclude_after_event = bool(cfg.get("labeling", {}).get("exclude_after_event", True))

    df = _load_cohort_events()
    lg.info(f"Labeling on cohort n={len(df)} | horizons={horizons} | start_after={start_after_hours}h | exclude_after_event={exclude_after_event}")

    out_paths: List[str] = []
    parts = []

    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="make_labels"):
        for h in horizons:
            rows = []
            n = len(df)
            non_empty = 0

            for i in range(n):
                r = df.iloc[i]
                stay_id = int(r.stay_id)
                icu_in = r.icu_in
                icu_out = r.icu_out
                fev = r.first_event_time

                grid, dbg = _build_grid_by_hours(icu_in, icu_out, fev,
                                                 start_after_hours=start_after_hours,
                                                 exclude_after_event=exclude_after_event)
                if i < 5:
                    lg.info(f"[debug h={h}] stay={stay_id} start_dt={dbg.get('start_dt')} "
                            f"start={dbg.get('start', None)} end_cap={dbg.get('end_cap')} "
                            f"fev={dbg.get('fev')} hours={dbg.get('hours', None)} "
                            f"grid_len={dbg.get('grid_len', 0)} grid_head={dbg.get('grid_head', [])} "
                            f"note={dbg.get('note', '')}")

                if len(grid) == 0:
                    continue
                non_empty += 1

                if pd.isna(fev):
                    rows.append(pd.DataFrame({
                        "stay_id": stay_id,
                        "index_time": grid,
                        "horizon_hours": h,
                        "label": 0,
                        "first_event_time": pd.NaT,
                    }))
                else:
                    fev_ts = pd.Timestamp(fev)
                    right = grid + pd.Timedelta(hours=h)
                    y = (fev_ts > grid) & (fev_ts <= right)
                    rows.append(pd.DataFrame({
                        "stay_id": stay_id,
                        "index_time": grid,
                        "horizon_hours": h,
                        "label": y.astype(int),
                        "first_event_time": fev_ts,
                    }))

            lab = (pd.concat(rows, ignore_index=True)
                   if rows else
                   pd.DataFrame(columns=["stay_id", "index_time", "horizon_hours", "label", "first_event_time"]))

            out_h = os.path.join(cfg.paths.interim, f"labels_h{h}.parquet")
            lab.to_parquet(out_h, index=False)
            lg.info(f"labels saved: {out_h} | rows={len(lab)} | positives={int(lab['label'].sum())} | non_empty_grids={non_empty}/{n}")
            out_paths.append(out_h)
            parts.append(lab)

    if save_all and len(parts) > 0:
        all_df = pd.concat(parts, ignore_index=True)
        out_all = os.path.join(cfg.paths.interim, "labels_all.parquet")
        all_df.to_parquet(out_all, index=False)
        lg.info(f"labels (ALL) saved: {out_all} | rows={len(all_df)}")
        out_paths.append(out_all)

    return out_paths


if __name__ == "__main__":
    paths = build_labels(horizons=None, save_all=True)
    print("\n".join(paths))
