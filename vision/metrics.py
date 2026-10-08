"""Evaluation metrics for ordinal condition scores (1-5) and room types."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred))) if len(y_true) else float("nan")


def confusion(y_true, y_pred, labels) -> np.ndarray:
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred, strict=True):
        if t in idx and p in idx:
            m[idx[t], idx[p]] += 1
    return m


def quadratic_weighted_kappa(y_true, y_pred, labels=(1, 2, 3, 4, 5)) -> float:
    """1 = perfect agreement, 0 = chance, negative = worse than chance. NaN if undefined."""
    obs = confusion(y_true, y_pred, labels).astype(float)
    n = obs.sum()
    if n == 0:
        return float("nan")
    k = len(labels)
    w = np.array([[(i - j) ** 2 / (k - 1) ** 2 for j in range(k)] for i in range(k)])
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    denom = (w * exp).sum()
    if denom == 0:
        return float("nan")  # one of the raters used a single value: kappa is undefined
    return float(1.0 - (w * obs).sum() / denom)


def per_group_mae(y_true, y_pred, groups) -> dict[str, tuple[float, int]]:
    """MAE and count per group (e.g. per room type)."""
    out: dict[str, tuple[float, int]] = {}
    for g in sorted(set(groups)):
        idx = [i for i, x in enumerate(groups) if x == g]
        out[g] = (mae([y_true[i] for i in idx], [y_pred[i] for i in idx]), len(idx))
    return out
