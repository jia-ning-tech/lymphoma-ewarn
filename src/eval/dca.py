from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class DCAMetrics:
    thresholds: np.ndarray
    nb_model: np.ndarray
    nb_all: np.ndarray
    nb_none: np.ndarray
    prevalence: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame({
            "threshold": self.thresholds,
            "nb_model": self.nb_model,
            "nb_all": self.nb_all,
            "nb_none": self.nb_none,
            "prevalence": self.prevalence,
        })


def _net_benefit(y_true: np.ndarray, y_pred_bin: np.ndarray, pt: float) -> float:
    """
    Net benefit = TP/N - FP/N * (pt / (1-pt))
    """
    assert 0.0 < pt < 1.0, "pt must be in (0,1)"
    y = y_true.astype(int)
    yhat = y_pred_bin.astype(int)
    N = len(y)
    tp = (yhat & (y == 1)).sum()
    fp = (yhat & (y == 0)).sum()
    return (tp / N) - (fp / N) * (pt / (1.0 - pt))


def _net_benefit_treat_all(y_true: np.ndarray, pt: float) -> float:
    """
    Treat-all strategy: sensitivity=1, specificity=0  => NB_all = prevalence - (1-prevalence) * (pt/(1-pt))
    """
    y = y_true.astype(int)
    prev = y.mean()
    return prev - (1.0 - prev) * (pt / (1.0 - pt))


def decision_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
    per_100: bool = False,
) -> DCAMetrics:
    """
    Compute decision curve analysis metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Binary outcomes (0/1).
    y_score : array-like of shape (n_samples,)
        Predicted probabilities.
    thresholds : iterable of floats in (0,1)
        Thresholds at which to compute net benefit.
    per_100 : bool
        If True, multiply NB by 100 (per 100 patients).

    Returns
    -------
    DCAMetrics
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(y_score, dtype=float)
    mask = ~np.isnan(p)
    y = y[mask]
    p = p[mask]
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
    thr = np.asarray(list(thresholds), dtype=float)
    nb_model = np.zeros_like(thr, dtype=float)
    nb_all = np.zeros_like(thr, dtype=float)
    nb_none = np.zeros_like(thr, dtype=float)

    for i, t in enumerate(thr):
        yhat = (p >= t).astype(int)
        nb_model[i] = _net_benefit(y, yhat, t)
        nb_all[i] = _net_benefit_treat_all(y, t)
        nb_none[i] = 0.0

    if per_100:
        nb_model *= 100.0
        nb_all *= 100.0
        nb_none *= 100.0

    return DCAMetrics(
        thresholds=thr,
        nb_model=nb_model,
        nb_all=nb_all,
        nb_none=nb_none,
        prevalence=float(y.mean()),
    )
