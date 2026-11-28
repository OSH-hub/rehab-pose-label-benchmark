from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    from numpy import trapezoid as _trapezoid
except ImportError:  # pragma: no cover - older NumPy
    from numpy import trapz as _trapezoid

__all__ = [
    "area_under_learning_curve",
    "budget_to_target",
    "annotation_time_reduction",
    "correction_rate",
]


def _sorted_curve(budgets: Any, scores: Any) -> tuple[np.ndarray, np.ndarray]:
    b = np.asarray(budgets, dtype=np.float64).ravel()
    s = np.asarray(scores, dtype=np.float64).ravel()
    if b.shape != s.shape:
        raise ValueError(f"budgets/scores length mismatch: {b.shape} vs {s.shape}")
    if b.size == 0:
        return b, s
    order = np.argsort(b, kind="mergesort")
    return b[order], s[order]


def area_under_learning_curve(
    budgets: Any, scores: Any, *, normalize: bool = False
) -> float:
    b, s = _sorted_curve(budgets, scores)
    if b.size == 0:
        return float("nan")
    if b.size == 1:
        return float(s[0]) if normalize else float("nan")
    area = float(_trapezoid(s, b))
    if normalize:
        span = b[-1] - b[0]
        if span <= 0:
            return float(np.mean(s))
        return area / span
    return area


def budget_to_target(
    budgets: Any, scores: Any, target: float, *, interpolate: bool = True
) -> float:
    b, s = _sorted_curve(budgets, scores)
    if b.size == 0:
        return float("inf")

    meets = s >= target
    if not np.any(meets):
        return float("inf")

    first = int(np.argmax(meets))
    if first == 0:
        return float(b[0])
    if not interpolate:
        return float(b[first])

    b0, b1 = b[first - 1], b[first]
    s0, s1 = s[first - 1], s[first]
    if s1 == s0:
        return float(b1)
    frac = (target - s0) / (s1 - s0)
    return float(b0 + frac * (b1 - b0))


def annotation_time_reduction(
    al_budgets: Any,
    al_scores: Any,
    baseline_budgets: Any,
    baseline_scores: Any,
    target: float,
    *,
    interpolate: bool = True,
) -> float:
    base = budget_to_target(
        baseline_budgets, baseline_scores, target, interpolate=interpolate
    )
    al = budget_to_target(al_budgets, al_scores, target, interpolate=interpolate)
    if not np.isfinite(base) or base <= 0:
        return float("nan")
    if not np.isfinite(al):
        return float("-inf")
    return float((base - al) / base)


def correction_rate(corrected: Any, total: Any = None) -> float:
    if total is None:
        flags = np.asarray(corrected).astype(bool).ravel()
        if flags.size == 0:
            return float("nan")
        return float(np.mean(flags))
    n_total = float(total)
    if n_total <= 0:
        return float("nan")
    return float(float(corrected) / n_total)
