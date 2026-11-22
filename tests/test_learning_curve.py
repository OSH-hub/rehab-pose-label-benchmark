"""Tests for active-learning curve metrics on known points."""

from __future__ import annotations

import numpy as np
import pytest

from rehabal_pose.metrics.learning_curve import (
    annotation_time_reduction,
    area_under_learning_curve,
    budget_to_target,
    correction_rate,
)


# ---------------------------------------------------------------------------
# area_under_learning_curve
# ---------------------------------------------------------------------------


def test_auc_trapezoid_known_points():
    budgets = [0.0, 1.0, 2.0]
    scores = [0.0, 1.0, 1.0]
    # trapezoids: (0+1)/2*1 + (1+1)/2*1 = 0.5 + 1.0 = 1.5
    assert area_under_learning_curve(budgets, scores) == pytest.approx(1.5)


def test_auc_rectangle():
    budgets = [0.0, 10.0]
    scores = [0.5, 0.5]
    assert area_under_learning_curve(budgets, scores) == pytest.approx(5.0)


def test_auc_normalized_is_mean_for_constant():
    budgets = [0.0, 10.0]
    scores = [0.7, 0.7]
    assert area_under_learning_curve(budgets, scores, normalize=True) == pytest.approx(0.7)


def test_auc_unsorted_input():
    budgets = [2.0, 0.0, 1.0]
    scores = [1.0, 0.0, 1.0]
    assert area_under_learning_curve(budgets, scores) == pytest.approx(1.5)


def test_auc_monotonic_curve_sane():
    budgets = np.arange(0, 11, dtype=float)
    scores = np.linspace(0.0, 1.0, 11)
    auc = area_under_learning_curve(budgets, scores)
    # straight line from 0 to 1 over width 10 => area = 0.5 * 1 * 10 = 5.0
    assert auc == pytest.approx(5.0)
    norm = area_under_learning_curve(budgets, scores, normalize=True)
    assert norm == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# budget_to_target
# ---------------------------------------------------------------------------


def test_budget_to_target_interpolated_crossing():
    budgets = [0.0, 10.0, 20.0]
    scores = [0.0, 0.5, 1.0]
    # target 0.75 crosses between (10,0.5) and (20,1.0): frac=0.5 -> 15
    assert budget_to_target(budgets, scores, 0.75) == pytest.approx(15.0)


def test_budget_to_target_exact_point():
    budgets = [0.0, 10.0, 20.0]
    scores = [0.0, 0.5, 1.0]
    assert budget_to_target(budgets, scores, 0.5) == pytest.approx(10.0)


def test_budget_to_target_first_point_meets():
    assert budget_to_target([5.0, 10.0], [0.9, 0.95], 0.8) == pytest.approx(5.0)


def test_budget_to_target_never_reached():
    assert budget_to_target([0.0, 10.0], [0.0, 0.5], 0.9) == float("inf")


def test_budget_to_target_no_interpolate():
    budgets = [0.0, 10.0, 20.0]
    scores = [0.0, 0.5, 1.0]
    assert budget_to_target(budgets, scores, 0.75, interpolate=False) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# annotation_time_reduction
# ---------------------------------------------------------------------------


def test_annotation_time_reduction():
    # AL reaches target 0.8 at budget 10; baseline at 20 -> 50% reduction
    al_b = [0.0, 10.0]
    al_s = [0.0, 0.8]
    base_b = [0.0, 20.0]
    base_s = [0.0, 0.8]
    red = annotation_time_reduction(al_b, al_s, base_b, base_s, 0.8)
    assert red == pytest.approx(0.5)


def test_annotation_time_reduction_baseline_unreached():
    red = annotation_time_reduction([0.0, 10.0], [0.0, 0.9], [0.0, 10.0], [0.0, 0.5], 0.8)
    assert np.isnan(red)


# ---------------------------------------------------------------------------
# correction_rate
# ---------------------------------------------------------------------------


def test_correction_rate_counts():
    assert correction_rate(3, 10) == pytest.approx(0.3)


def test_correction_rate_flags():
    assert correction_rate([True, False, True, False]) == pytest.approx(0.5)


def test_correction_rate_zero_total():
    assert np.isnan(correction_rate(0, 0))
