# src/eval/dca.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Dict
import numpy as np

@dataclass
class DCAMetrics:
    thresholds: np.ndarray          # shape (T,)
    nb_model: np.ndarray            # net benefit of model (per 'per' patients)
    nb_all: np.ndarray              # treat-all
    nb_none: np.ndarray             # zeros
    prevalence: float               # y.mean()
    per: int = 100                  # scaling factor for readability

def _check_inputs(y: np.ndarray, p: np.ndarray):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    assert set(np.unique(y)).issubset({0, 1}), "y must be binary in {0,1}"
    assert p.ndim == 1 and y.shape[0] == p.shape[0], "y and p length mismatch"
    return y, p

def decision_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Optional[Iterable[float]] = None,
    per: int = 100,
) -> DCAMetrics:
    """
    Compute decision curve analysis arrays.
    Net benefit formula (Vickers): NB = TPR - FPR * (pt/(1-pt)),
    where TPR = TP/N, FPR = FP/N. 'per' is a final display scaling (e.g., 100).
    """
    y, p = _check_inputs(y_true, y_prob)
    n = float(len(y))
    prev = float(y.mean())

    if thresholds is None:
        thresholds = np.linspace(1e-4, 0.9999, 200)
    thr = np.asarray(thresholds, dtype=float)

    nb_model = np.zeros_like(thr)
    for i, t in enumerate(thr):
        pred = (p >= t).astype(int)
        tp = float((pred == 1).sum() and np.logical_and(pred == 1, y == 1).sum())
        fp = float((pred == 1).sum() and np.logical_and(pred == 1, y == 0).sum())
        # 转为率
        tpr = tp / n
        fpr = fp / n
        w = t / (1.0 - t)
        nb_model[i] = tpr - fpr * w

    w_all = thr / (1.0 - thr)
    nb_all = prev - (1.0 - prev) * w_all
    nb_none = np.zeros_like(thr)

    if per is not None and per != 1:
        nb_model = nb_model * per
        nb_all = nb_all * per
        nb_none = nb_none * per

    return DCAMetrics(
        thresholds=thr,
        nb_model=nb_model,
        nb_all=nb_all,
        nb_none=nb_none,
        prevalence=prev,
        per=per,
    )
