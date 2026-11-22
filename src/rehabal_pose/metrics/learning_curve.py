"""Active-learning / label-efficiency curve metrics (pure NumPy).

RehabAL-Pose evaluates how efficiently performance grows as annotation budget
is spent. These metrics summarize a learning curve (budget on x, score on y):

* :func:`area_under_learning_curve` -- (optionally normalized) trapezoidal AUC.
* :func:`budget_to_target` -- minimal budget to reach a target score, with
  linear interpolation of the crossing.
* :func:`annotation_time_reduction` -- relative budget saved vs a baseline.
* :func:`correction_rate` -- fraction of clinician-corrected predictions.

Budgets are assumed sortable ascending; inputs are sorted defensively.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:  # NumPy >= 2.0 renamed trapz -> trapezoid
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
    """Area under the learning curve via the trapezoidal rule.

    Parameters
    ----------
    budgets:
        x-axis values (annotation budget, e.g. number of labeled samples).
    scores:
        y-axis values (performance metric at each budget).
    normalize:
        If ``True``, divide by the budget span ``(b_max - b_min)`` to return the
        *mean* score over the curve (a.k.a. ALC), invariant to budget scale.
        If ``False`` (default) return the raw trapezoidal area.

    Returns
    -------
    float
        AUC (raw) or mean score (normalized). NaN if < 2 points. With a single
        point and ``normalize=True`` returns that score.
    """
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
    """Minimal budget at which the score first reaches ``target``.

    Parameters
    ----------
    budgets, scores:
        Learning-curve points.
    target:
        Score threshold to reach (``score >= target``).
    interpolate:
        If ``True`` (default) linearly interpolate the crossing budget between
        the last sub-target point and the first at-or-above point. If ``False``
        return the budget of the first point at/above target.

    Returns
    -------
    float
        Budget reaching the target, or ``inf`` if the target is never reached.
        If the very first point already meets the target, that budget is
        returned.
    """
    b, s = _sorted_curve(budgets, scores)
    if b.size == 0:
        return float("inf")

    meets = s >= target
    if not np.any(meets):
        return float("inf")

    first = int(np.argmax(meets))  # first True index
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
    """Relative annotation saved by active learning to reach ``target``.

    ``reduction = (baseline_budget - al_budget) / baseline_budget`` where each
    budget is the amount needed to reach ``target`` (via
    :func:`budget_to_target`). Positive => AL is cheaper.

    Parameters
    ----------
    al_budgets, al_scores:
        Active-learning curve.
    baseline_budgets, baseline_scores:
        Baseline (e.g. random sampling) curve.
    target:
        Score both strategies must reach.
    interpolate:
        Passed through to :func:`budget_to_target`.

    Returns
    -------
    float
        Fractional reduction in ``[<=1]``; ``nan`` if the baseline never reaches
        the target; ``-inf`` if AL never reaches it (infinite cost).
    """
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
    """Fraction of model predictions a clinician had to correct.

    Two calling conventions:

    * ``correction_rate(n_corrected, n_total)`` -- scalar counts.
    * ``correction_rate(flags)`` -- a boolean/0-1 array where ``True`` marks a
      corrected prediction (``total`` omitted).

    Parameters
    ----------
    corrected:
        Count of corrected predictions, or a per-prediction boolean array.
    total:
        Total predictions (required only for the scalar-count form).

    Returns
    -------
    float
        Correction rate in ``[0, 1]``; ``nan`` if ``total`` is 0.
    """
    if total is None:
        flags = np.asarray(corrected).astype(bool).ravel()
        if flags.size == 0:
            return float("nan")
        return float(np.mean(flags))
    n_total = float(total)
    if n_total <= 0:
        return float("nan")
    return float(float(corrected) / n_total)
