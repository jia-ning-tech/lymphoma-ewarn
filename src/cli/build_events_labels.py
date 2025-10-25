from __future__ import annotations

import os
import argparse
from typing import List, Optional

import pandas as pd
import numpy as np

from ..config import get_cfg
from ..utils.log import get_logger, heartbeat, progress_bar
from ..utils.timeidx import build_time_grid_for_stay
from ..events.detect_events import detect_events_for_cohort


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="python -m src.cli.build_events_labels",
        description="Build ICU deterioration events (vaso/IMV/CRRT) and rolling labels for horizons.",
    )
    p.add_argument("--stage", choices=["events", "labels", "all"], default="all",
                   help="Run only event detection, only label building, or both (default: all).")
    p.add_argument("--horizons", nargs="*", type=int, default=None,
                   help="Horizons (hours) for labels, e.g. --horizons 24 48. Default: use config.")
    p.add_argument("--head", type=int, default=0,
                   help="Show first N rows of the final labels (0=skip).")
    p.add_argument("--stats", action="store_true",
                   help="Print basic stats of labels (per horizon positives).")
    return p.parse_args(argv)


def _ensure_paths(cfg):
    need = [
        (cfg.paths.interim, True),
        (cfg.paths.outputs, True),
    ]
    for p, is_dir in need:
        if is_dir:
            os.makedirs(p, exist_ok=True)


def _load_cohort_and_events(cfg):
    cohort_path = os.path.join(cfg.paths.interim, "cohort.parquet")
    events_path = os.path.join(cfg.paths.interim, "events_first.csv")
    if not os.path.exists(cohort_path):
        raise FileNotFoundError(f"Missing cohort: {cohort_path} (run src.cli.build_cohort first)")
    if not os.path.exists(events_path):
        raise FileNotFoundError(f"Missing events: {events_path} (run with --stage events or --stage all)")

    cohort = pd.read_parquet(cohort_path, columns=["stay_id", "icu_in", "icu_out"])
    events = pd.read_csv(events_path)
    # unify dtypes
    for c in ["icu_in", "icu_out"]:
        cohort[c] = pd.to_datetime(cohort[c], errors="coerce")
    for c in ["vaso_time", "imv_time", "crrt_time", "first_event_time"]:
        if c in events.columns:
            events[c] = pd.to_datetime(events[c], errors="coerce")

    df = cohort.merge(events[["stay_id", "first_event_time"]], on="stay_id", how="left")
    return df


def _build_labels_for_horizon(df: pd.DataFrame, horizon: int, cfg) -> pd.DataFrame:
    """
    For a given horizon (hours), build rolling labels.
    Output cols: stay_id, index_time, horizon_hours, label, first_event_time
    """
    lg = get_logger("cli.events_labels")
    rows = []
    it = progress_bar(df.itertuples(index=False), total=len(df), desc=f"labels h{horizon}")

    start_after_hours = int(cfg.prediction.get("start_after_hours", 6))
    index_freq = str(cfg.prediction.get("index_freq", "1H"))

    for r in it:
        stay_id = int(r.stay_id)
        icu_in = r.icu_in
        icu_out = r.icu_out
        fev = r.first_event_time if hasattr(r, "first_event_time") else None

        grid = build_time_grid_for_stay(icu_in, icu_out, fev,
                                        start_after_hours=start_after_hours,
                                        index_freq=index_freq)
        if len(grid) == 0:
            continue

        if pd.isna(fev):
            rows.append(pd.DataFrame({
                "stay_id": stay_id,
                "index_time": grid,
                "horizon_hours": horizon,
                "label": 0,
                "first_event_time": pd.NaT,
            }))
        else:
            fev_ts = pd.Timestamp(fev)
            right = grid + pd.Timedelta(hours=horizon)
            y = (fev_ts > grid) & (fev_ts <= right)
            rows.append(pd.DataFrame({
                "stay_id": stay_id,
                "index_time": grid,
                "horizon_hours": horizon,
                "label": y.astype(int).values,
                "first_event_time": fev_ts,
            }))

    if not rows:
        return pd.DataFrame(columns=["stay_id", "index_time", "horizon_hours", "label", "first_event_time"])
    out = pd.concat(rows, ignore_index=True)
    return out


def build_labels(save_all: bool = True, horizons: Optional[List[int]] = None) -> List[str]:
    cfg = get_cfg()
    lg = get_logger("cli.events_labels")
    _ensure_paths(cfg)

    if horizons is None or len(horizons) == 0:
        horizons = list(map(int, cfg.prediction.get("horizons_hours", [24, 48])))

    df = _load_cohort_and_events(cfg)
    saved_paths: List[str] = []
    parts = []

    with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="build labels"):
        for h in horizons:
            lab = _build_labels_for_horizon(df, h, cfg)
            out_path = os.path.join(cfg.paths.interim, f"labels_h{h}.parquet")
            lab.to_parquet(out_path, index=False)
            lg.info(f"labels saved: {out_path} | rows={len(lab)} | positives={int(lab['label'].sum())}")
            saved_paths.append(out_path)
            parts.append(lab)

    if save_all and parts:
        all_df = pd.concat(parts, ignore_index=True)
        out_all = os.path.join(cfg.paths.interim, "labels_all.parquet")
        all_df.to_parquet(out_all, index=False)
        lg.info(f"labels (ALL) saved: {out_all} | rows={len(all_df)}")
        saved_paths.append(out_all)

    return saved_paths


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = get_cfg()
    lg = get_logger("cli.events_labels")
    _ensure_paths(cfg)

    horizons: Optional[List[int]] = args.horizons

    if args.stage in ("events", "all"):
        with heartbeat(lg, secs=int(cfg.logging.heartbeat_secs), note="detect events"):
            path = detect_events_for_cohort(save_csv=True)
            lg.info(f"events done: {path}")

    if args.stage in ("labels", "all"):
        saved = build_labels(save_all=True, horizons=horizons)
        all_path = [p for p in saved if p.endswith("labels_all.parquet")]
        view_path = all_path[0] if all_path else saved[-1]

        if args.head and args.head > 0:
            lg.info(f"Showing first {args.head} rows from {view_path}")
            df = pd.read_parquet(view_path)
            print(df.head(args.head).to_string(index=False))

        if args.stats:
            df = pd.read_parquet(view_path, columns=["horizon_hours", "label"])
            stat = df.groupby("horizon_hours")["label"].agg(["count", "sum"]).rename(columns={"sum": "positives"}).reset_index()
            stat["rate_pct"] = (stat["positives"] / stat["count"] * 100).round(2)
            lg.info("Label stats per horizon (count / positives / rate%):")
            for row in stat.itertuples(index=False):
                lg.info(f"h{int(row.horizon_hours)}: {int(row.count)} / {int(row.positives)} / {row.rate_pct}%")

    lg.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
