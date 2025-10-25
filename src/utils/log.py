from __future__ import annotations

# --- make both "python -m src.utils.log" and "python src/utils/log.py" work ---
import os, sys, pathlib
if __package__ is None or __package__ == "":
    # running as script: add project root (two levels up from this file) to sys.path
    _here = pathlib.Path(__file__).resolve()
    _root = _here.parents[2]  # .../lymphoma-ewarn
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    # now absolute import works
    from src.config import get_cfg
else:
    # running as module: relative import is fine
    from ..config import get_cfg

import logging
import time
import threading
import itertools
from contextlib import contextmanager
from typing import Iterable, Optional, Any, Callable

# 尝试导入 tqdm，若不可用则优雅降级
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

# ----------------------------- Logger -------------------------------- #

_LOGGERS = {}

def get_logger(name: str = "lymphoma-ewarn") -> logging.Logger:
    """
    Create or get a configured logger that logs to console and file.
    Config is read from conf/config.yaml (logging.*).
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    cfg = get_cfg()
    log_cfg = cfg.logging
    level = getattr(logging, str(log_cfg.get("level", "INFO")).upper(), logging.INFO)
    log_dir = log_cfg.get("log_dir")
    to_file = bool(log_cfg.get("to_file", True))
    timefmt = log_cfg.get("timefmt", "%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # 避免重复

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt=timefmt,
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if to_file and log_dir:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


# ----------------------------- Heartbeat ------------------------------ #

class Heartbeat:
    """
    Periodically logs a heartbeat message in a background thread.
    Usage:
        hb = Heartbeat(logger, secs=30, note="build_features"); hb.start(); ...; hb.stop()
    Or:
        with heartbeat(logger, secs=30, note="training"): ...
    """
    def __init__(self, logger: logging.Logger, secs: Optional[int] = None, note: str = "working"):
        cfg = get_cfg()
        self.logger = logger
        self.secs = secs if secs is not None else int(cfg.logging.get("heartbeat_secs", 30))
        self.note = note
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _run(self):
        while not self._stop.wait(self.secs):
            try:
                self.logger.info(f"[heartbeat] still alive: {self.note}")
            except Exception:
                pass

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="Heartbeat", daemon=True)
            self._thread.start()
            self.logger.debug(f"Heartbeat started ({self.secs}s) for: {self.note}")

    def stop(self):
        if self._thread is not None:
            self._stop.set()
            self._thread.join(timeout=self.secs + 1)
            self.logger.debug("Heartbeat stopped.")


@contextmanager
def heartbeat(logger: logging.Logger, secs: Optional[int] = None, note: str = "working"):
    hb = Heartbeat(logger, secs=secs, note=note)
    hb.start()
    try:
        yield
    finally:
        hb.stop()


# ----------------------------- Progress ------------------------------- #

def progress_bar(iterable: Iterable[Any],
                 total: Optional[int] = None,
                 desc: str = "",
                 leave: bool = True):
    """
    Wrap an iterable with a progress bar if tqdm is available/enabled.
    Falls back to identity if tqdm is not installed or disabled in config.
    """
    cfg = get_cfg()
    enable = bool(cfg.progress.get("enable_tqdm", True))
    refresh = float(cfg.progress.get("refresh_secs", 0.5))
    if enable and tqdm is not None:
        return tqdm(iterable, total=total, desc=desc, leave=leave,
                    dynamic_ncols=True, mininterval=refresh,
                    disable=False, unit="it", smoothing=0.2)
    else:
        logger = get_logger()
        logger.info(f"[progress] {desc} started (no tqdm).")
        for x in iterable:
            yield x
        logger.info(f"[progress] {desc} finished.")


def spinner(duration_secs: Optional[int] = None,
           desc: str = "working",
           interval: Optional[float] = None):
    """
    A lightweight terminal spinner. If duration_secs is None, spin until caller breaks.
    Usage:
        for _ in spinner(5, "loading model"): time.sleep(0.1)
    """
    cfg = get_cfg()
    interval = interval or float(cfg.progress.get("refresh_secs", 0.1))
    frames = {
        "dots": "⠁⠂⠄⡀⢀⠠⠐⠈",
        "line": "-\\|/",
        "arc": "◜◠◝◞◡◟",
    }
    style = str(cfg.progress.get("spinner", "dots"))
    seq = frames.get(style, frames["dots"])
    cyc = itertools.cycle(seq)

    start = time.time()
    try:
        while True:
            ch = next(cyc)
            elapsed = time.time() - start
            if duration_secs is not None:
                remaining = max(0, duration_secs - elapsed)
                sys.stdout.write(f"\r{ch} {desc} | ETA: {remaining:5.1f}s ")
            else:
                sys.stdout.write(f"\r{ch} {desc} ")
            sys.stdout.flush()
            time.sleep(interval)
            if duration_secs is not None and elapsed >= duration_secs:
                break
            yield
    finally:
        sys.stdout.write("\r")
        sys.stdout.flush()


def countdown(total_secs: int, desc: str = "waiting", tick_secs: Optional[float] = None):
    """
    Print a second-level countdown with ETA.
    """
    tick = tick_secs or 1.0
    logger = get_logger()
    end = time.time() + total_secs
    while True:
        now = time.time()
        left = int(max(0, end - now))
        sys.stdout.write(f"\r⏳ {desc}: {left:4d}s remaining ")
        sys.stdout.flush()
        if left <= 0:
            break
        time.sleep(tick)
    sys.stdout.write("\n")
    logger.debug(f"Countdown finished: {desc}")


# ----------------------------- Decorators ----------------------------- #

def timed(logger: Optional[logging.Logger] = None, note: str = "task"):
    """
    Decorator to time a function and log duration.
    """
    lg = logger or get_logger()
    def deco(fn: Callable):
        def wrap(*args, **kwargs):
            t0 = time.time()
            lg.info(f"[start] {note}")
            try:
                return fn(*args, **kwargs)
            finally:
                dt = time.time() - t0
                lg.info(f"[done]  {note} | took {dt:.2f}s")
        return wrap
    return deco


# ----------------------------- Self-test ------------------------------ #

if __name__ == "__main__":
    logger = get_logger("demo")
    logger.info("Logger demo started.")
    from time import sleep
    with heartbeat(logger, secs=2, note="self-test"):
        rng = range(20)
        bar = progress_bar(rng, total=len(rng), desc="progress demo")
        for _ in bar:
            sleep(0.05)
        countdown(3, "short countdown")
    logger.info("Logger demo finished.")
