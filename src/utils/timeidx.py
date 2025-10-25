from __future__ import annotations

import math
from typing import Iterable, Iterator, Optional, Tuple, Union

import numpy as np
import pandas as pd

# 仅用于心跳与日志（可选）
from .log import get_logger, heartbeat
from ..config import get_cfg


# ------------------------- Basic time helpers ------------------------- #

def to_ts(x: Union[str, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    """
    Convert input to naive pandas Timestamp (no timezone).
    """
    if isinstance(x, pd.Timestamp):
        ts = x
    else:
        ts = pd.to_datetime(x, utc=False)
    # 保持 naive（MIMIC 时间戳本身为本地/naive）
    if ts.tz is not None:
        ts = ts.tz_convert(None) if hasattr(ts, "tz_convert") else ts.tz_localize(None)  # type: ignore
    return ts


def floor_hour(t: Union[str, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    return to_ts(t).floor("H")


def ceil_hour(t: Union[str, pd.Timestamp, np.datetime64]) -> pd.Timestamp:
    return to_ts(t).ceil("H")


def ensure_start_before_end(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Timestamp, pd.Timestamp]:
    s, e = to_ts(start), to_ts(end)
    if e <= s:
        raise ValueError(f"end ({e}) must be after start ({s})")
    return s, e


# ------------------------- Hourly index builders ---------------------- #

def make_hourly_index(
    icu_in: Union[str, pd.Timestamp],
    icu_out: Union[str, pd.Timestamp],
    start_after_hours: int = 6,
    freq: str = "1H",
) -> pd.DatetimeIndex:
    """
    Build an hourly time grid for an ICU stay.
    - Start at ceil(icu_in + start_after_hours), end at floor(icu_out) - 1 step
    - Empty index is possible when icu length < start_after_hours + 1*freq
    """
    s, e = ensure_start_before_end(icu_in, icu_out)
    start = (s + pd.Timedelta(hours=start_after_hours)).ceil(freq)
    end = e.floor(freq)
    if end <= start:
        return pd.DatetimeIndex([], dtype="datetime64[ns]")
    # 到 end - step（右开区间）
    grid = pd.date_range(start=start, end=end - pd.Timedelta(freq), freq=freq)
    return grid


def iter_label_windows(grid: pd.DatetimeIndex, horizon_hours: int) -> Iterator[Tuple[pd.Timestamp, pd.Timestamp]]:
    """
    For each index time t in grid, yield (t, t + horizon].
    """
    delta = pd.Timedelta(hours=horizon_hours)
    for t in grid:
        yield t, t + delta


# ------------------------- Event alignment / pruning ----------------- #

def align_to_grid(
    ts: Union[str, pd.Timestamp],
    grid: pd.DatetimeIndex,
    tolerance_minutes: int = 5,
) -> Optional[pd.Timestamp]:
    """
    Align an arbitrary timestamp to the nearest point in `grid` within tolerance.
    Returns the aligned grid timestamp or None if no point within tolerance.
    """
    if len(grid) == 0:
        return None
    t = to_ts(ts)
    # 找到最邻近索引
    # 将 grid 视为升序数组，使用 searchsorted
    pos = grid.searchsorted(t)
    candidates = []
    if 0 <= pos < len(grid):
        candidates.append(grid[pos])
    if pos - 1 >= 0:
        candidates.append(grid[pos - 1])
    if pos + 1 < len(grid):
        candidates.append(grid[pos + 1])
    # 计算差并选最近
    if not candidates:
        return None
    diffs = [abs((c - t).total_seconds()) for c in candidates]
    best = candidates[int(np.argmin(diffs))]
    if min(diffs) <= tolerance_minutes * 60:
        return best
    return None


def prune_grid_after_event(
    grid: pd.DatetimeIndex,
    first_event_time: Optional[Union[str, pd.Timestamp]],
    exclude_at_event: bool = True,
) -> pd.DatetimeIndex:
    """
    Remove grid times at/after the first_event_time, because after the first deterioration
    we stop generating rolling labels.
    """
    if first_event_time is None or pd.isna(first_event_time):
        return grid
    event_ts = to_ts(first_event_time)
    if exclude_at_event:
        mask = grid < event_ts
    else:
        mask = grid <= event_ts
    return grid[mask]


# ------------------------- Leakage checks ---------------------------- #

def check_no_leakage_features_vs_label(
    features: pd.DataFrame,
    time_col: str,
    cutoff_col: str,
) -> Tuple[bool, int]:
    """
    Check 'no future lookahead': all feature timestamps <= cutoff (t).
    Returns (ok, n_violations).
    `features[time_col]` and `features[cutoff_col]` should be datetime-like.
    """
    if features.empty:
        return True, 0
    t = pd.to_datetime(features[time_col], utc=False)
    c = pd.to_datetime(features[cutoff_col], utc=False)
    viol = (t > c).sum()
    return (viol == 0), int(viol)


# ------------------------- High-level helper ------------------------- #

def build_time_grid_for_stay(
    icu_in: Union[str, pd.Timestamp],
    icu_out: Union[str, pd.Timestamp],
    first_event_time: Optional[Union[str, pd.Timestamp]],
    start_after_hours: Optional[int] = None,
    index_freq: Optional[str] = None,
    exclude_after_event: Optional[bool] = None,
) -> pd.DatetimeIndex:
    """
    Convenience wrapper that reads defaults from config and returns
    the final time grid for one stay (after pruning by first event).
    """
    cfg = get_cfg()
    start_after_hours = int(cfg.prediction.get("start_after_hours", 6)) if start_after_hours is None else start_after_hours
    index_freq = str(cfg.prediction.get("index_freq", "1H")) if index_freq is None else index_freq
    if exclude_after_event is None:
        exclude_after_event = bool(cfg.labeling.get("exclude_after_event", True))

    grid = make_hourly_index(icu_in, icu_out, start_after_hours=start_after_hours, freq=index_freq)
    if exclude_after_event:
        grid = prune_grid_after_event(grid, first_event_time, exclude_at_event=True)
    return grid


# ------------------------- Self test --------------------------------- #

if __name__ == "__main__":
    lg = get_logger("timeidx.selftest")
    cfg = get_cfg()
    lg.info("Running timeidx self-test...")

    # 假设一个 ICU 住院
    icu_in = pd.Timestamp("2020-01-01 08:23:00")
    icu_out = pd.Timestamp("2020-01-03 05:40:00")
    first_event = pd.Timestamp("2020-01-02 01:15:00")  # 事件发生在第2天凌晨

    with heartbeat(lg, secs=int(cfg.logging.get("heartbeat_secs", 30)), note="timeidx self-test"):
        grid = build_time_grid_for_stay(icu_in, icu_out, first_event)
        lg.info(f"Grid length (after prune): {len(grid)} | head={grid[:3].tolist()} ... tail={grid[-3:].tolist() if len(grid)>2 else grid.tolist()}")

        # 标签窗口示例（24h）
        windows = list(iter_label_windows(grid[:5], horizon_hours=24))
        lg.info(f"First 3 windows (24h): {[(str(a), str(b)) for a,b in windows[:3]]}")

        # 对齐事件到网格（5分钟容差）
        aligned = align_to_grid(first_event, grid, tolerance_minutes=int(cfg.labeling.get("tolerance_minutes", 5)))
        lg.info(f"Aligned event to grid: {aligned}")

        # 泄漏检查（伪造 features 时间 > t 的样例）
        df = pd.DataFrame({
            "feat_time": [grid[0], grid[1], grid[2] + pd.Timedelta(minutes=10)],
            "cutoff_t":  [grid[0], grid[1], grid[2]],
        })
        ok, n = check_no_leakage_features_vs_label(df, "feat_time", "cutoff_t")
        lg.info(f"Leakage check ok={ok}, violations={n}")

    lg.info("timeidx self-test finished.")
