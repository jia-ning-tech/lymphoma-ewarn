# src/eval/dca.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt


@dataclass
class DCAMetrics:
    thresholds: np.ndarray          # shape [T]
    nb_model: np.ndarray            # shape [T]   per-patient net benefit
    nb_treat_all: np.ndarray        # shape [T]
    nb_treat_none: np.ndarray       # zeros [T]
    prevalence: float               # y.mean() （若 per_100=True，则已乘以 100）
    n: int                          # number of samples actually used


def _ensure_1d(x: Iterable) -> np.ndarray:
    x = np.asarray(x).reshape(-1)
    return x


def _remove_nan(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    m = np.isfinite(y) & np.isfinite(p)
    if w is not None:
        m = m & np.isfinite(w)
        w = w[m]
    return y[m], p[m], w


def _tp_fp_at_threshold(y: np.ndarray, p: np.ndarray, thr: float, w: Optional[np.ndarray]) -> Tuple[float, float]:
    """Return TP 和 FP 的计数（或加权计数）at threshold."""
    pred = (p >= thr).astype(int)
    if w is None:
        tp = float(((pred == 1) & (y == 1)).sum())
        fp = float(((pred == 1) & (y == 0)).sum())
    else:
        tp = float(w[(pred == 1) & (y == 1)].sum())
        fp = float(w[(pred == 1) & (y == 0)].sum())
    return tp, fp


def decision_curve(
    y: Iterable,
    p: Iterable,
    thresholds: Optional[Iterable[float]] = None,
    sample_weight: Optional[Iterable[float]] = None,
    per_100: bool = True,
) -> DCAMetrics:
    """
    标准 DCA：
      NB_model(pt)     = TP/N - FP/N * pt/(1-pt)
      NB_treat_all(pt) = prevalence - (1-prevalence) * pt/(1-pt)
      NB_treat_none(pt)= 0
    其中 N 是有效样本数（去除 NaN 后；若有 sample_weight 则为权重和）。
    per_100=True 时，最终把三条 NB 曲线统一乘以 100（每 100 名患者净获益）。
    """
    y = _ensure_1d(y).astype(int)
    p = _ensure_1d(p).astype(float)
    w = None if sample_weight is None else _ensure_1d(sample_weight).astype(float)

    # 去 NaN
    y, p, w = _remove_nan(y, p, w)
    if len(y) == 0:
        raise ValueError("No valid samples after removing NaNs.")

    # 阈值网格
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)  # 避免 0 和 1
    thresholds = _ensure_1d(thresholds).astype(float)
    thresholds = thresholds[(thresholds > 0) & (thresholds < 1)]
    thresholds = np.unique(np.clip(thresholds, 1e-6, 1 - 1e-6))

    # 样本量（或加权样本量）
    if w is None:
        N = float(len(y))
        prevalence = float(y.mean())
    else:
        N = float(w.sum())
        prevalence = float((w[y == 1].sum()) / N)

    nb_model = []
    nb_treat_all = []
    for pt in thresholds:
        tp, fp = _tp_fp_at_threshold(y, p, pt, w)
        nb_m = (tp / N) - (fp / N) * (pt / (1 - pt))
        nb_all = prevalence - (1 - prevalence) * (pt / (1 - pt))
        nb_model.append(nb_m)
        nb_treat_all.append(nb_all)

    nb_model = np.asarray(nb_model)
    nb_treat_all = np.asarray(nb_treat_all)
    nb_treat_none = np.zeros_like(nb_model)

    if per_100:
        nb_model *= 100.0
        nb_treat_all *= 100.0
        nb_treat_none *= 100.0
        prevalence_out = prevalence * 100.0
    else:
        prevalence_out = prevalence

    return DCAMetrics(
        thresholds=thresholds,
        nb_model=nb_model,
        nb_treat_all=nb_treat_all,
        nb_treat_none=nb_treat_none,
        prevalence=prevalence_out,
        n=int(N if w is None else round(N)),
    )


def plot_dca(
    dca: DCAMetrics,
    title: str = "",
    outfile: Optional[Path] = None,
    ylim: Optional[Tuple[float, float]] = None,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.plot(dca.thresholds, dca.nb_model, label="Model")
    plt.plot(dca.thresholds, dca.nb_treat_all, "--", label="Treat-all")
    plt.plot(dca.thresholds, dca.nb_treat_none, ":", label="Treat-none")
    plt.axhline(0, color="k", linewidth=0.7)
    plt.xlabel("Threshold probability (pt)")
    plt.ylabel("Net benefit (per 100 patients)")
    if title:
        plt.title(f"DCA: {title}  |  prevalence={dca.prevalence:.3f}")
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(loc="lower left")
    plt.tight_layout()
    if outfile is not None:
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=150)
        plt.close()
    else:
        plt.show()
