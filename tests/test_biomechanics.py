"""Known-geometry tests for biomechanics and pose metrics."""

from __future__ import annotations

import numpy as np
import pytest

from rehabal_pose.metrics.biomechanics import (
    JOINT_TRIPLETS,
    compute_rom,
    jerk,
    joint_angle,
    joint_angle_mae,
    peak_angle_error,
    phase_error,
    rom_error,
    temporal_smoothness,
)
from rehabal_pose.metrics.pose import mean_per_joint_error, oks, pck


# ---------------------------------------------------------------------------
# joint_angle
# ---------------------------------------------------------------------------


def test_right_angle_2d():
    a = [1.0, 0.0]
    b = [0.0, 0.0]
    c = [0.0, 1.0]
    assert joint_angle(a, b, c) == pytest.approx(90.0)


def test_right_angle_3d():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 0.0, 0.0]
    c = [0.0, 0.0, 1.0]
    assert joint_angle(a, b, c) == pytest.approx(90.0)


def test_straight_angle_2d():
    a = [-1.0, 0.0]
    b = [0.0, 0.0]
    c = [1.0, 0.0]
    assert joint_angle(a, b, c) == pytest.approx(180.0)


def test_joint_angle_vectorized_over_time():
    # T=3 right angles stacked.
    a = np.array([[1.0, 0.0]] * 3)
    b = np.array([[0.0, 0.0]] * 3)
    c = np.array([[0.0, 1.0]] * 3)
    ang = joint_angle(a, b, c)
    assert ang.shape == (3,)
    assert np.allclose(ang, 90.0)


# ---------------------------------------------------------------------------
# compute_rom
# ---------------------------------------------------------------------------


def test_compute_rom_sine_sweep():
    t = np.linspace(0, np.pi, 200)  # sin peaks at pi/2
    series = 50.0 * np.sin(t)  # 0 .. 50 .. ~0
    rom = compute_rom(series)
    assert rom["min"] == pytest.approx(0.0, abs=1e-6)
    assert rom["max"] == pytest.approx(50.0, abs=1e-2)
    assert rom["rom"] == pytest.approx(50.0, abs=1e-2)
    # peak near the middle of the sweep
    assert rom["peak_idx"] == int(np.argmax(series))


def test_compute_rom_empty():
    rom = compute_rom([])
    assert np.isnan(rom["rom"])
    assert rom["peak_idx"] == -1


# ---------------------------------------------------------------------------
# joint_angle_mae / rom_error / peak / phase
# ---------------------------------------------------------------------------


def _make_kps_sequence(T=10, K=16):
    rng = np.random.default_rng(0)
    return rng.normal(size=(T, K, 3))


def test_joint_angle_mae_zero_when_equal():
    kps = _make_kps_sequence()
    res = joint_angle_mae(kps, kps, JOINT_TRIPLETS)
    for name in JOINT_TRIPLETS:
        assert res[name] == pytest.approx(0.0, abs=1e-9)
    assert res["mean"] == pytest.approx(0.0, abs=1e-9)


def test_joint_angle_mae_ignores_visibility_channel():
    """The trailing visibility/confidence channel must NOT affect joint angles.

    Regression guard for the visibility-contamination bug: the SAME 2D pose,
    presented once as ``(T, K, 2)`` and once as ``(T, K, 3)`` whose 3rd column
    is a per-joint visibility that differs between pred (YOLO conf) and GT
    (label flag in {0,1,2}), must yield identical angle MAE (zero here because
    the spatial coordinates are equal). The third column is metadata and must
    never enter the angle calculation as a spatial coordinate.
    """
    rng = np.random.default_rng(7)
    xy = rng.normal(size=(8, 16, 2))
    # Same spatial pose; visibility VARIES per joint and differs pred vs GT.
    pred_vis = rng.uniform(0.3, 0.99, size=(8, 16, 1))               # YOLO conf
    gt_vis = rng.integers(0, 3, size=(8, 16, 1)).astype(np.float64)  # GT flags
    pred3 = np.concatenate([xy, pred_vis], axis=-1)
    gt3 = np.concatenate([xy, gt_vis], axis=-1)

    mae_3col = joint_angle_mae(pred3, gt3, JOINT_TRIPLETS)
    mae_2col = joint_angle_mae(xy, xy, JOINT_TRIPLETS)  # reference: equal xy -> 0
    for name in JOINT_TRIPLETS:
        assert mae_2col[name] == pytest.approx(0.0, abs=1e-9)
        assert mae_3col[name] == pytest.approx(mae_2col[name], abs=1e-9)
    assert mae_3col["mean"] == pytest.approx(0.0, abs=1e-9)


def test_rom_and_peak_error_zero_when_equal():
    series = 30.0 * np.sin(np.linspace(0, np.pi, 50))
    assert rom_error(series, series) == pytest.approx(0.0, abs=1e-9)
    assert peak_angle_error(series, series) == pytest.approx(0.0, abs=1e-9)
    assert phase_error(series, series) == pytest.approx(0.0, abs=1e-9)


def test_rom_error_known():
    pred = np.array([0.0, 10.0, 20.0])  # rom 20
    gt = np.array([0.0, 5.0, 10.0])  # rom 10
    assert rom_error(pred, gt) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# jerk / smoothness
# ---------------------------------------------------------------------------


def test_jerk_zero_for_constant_velocity():
    # linear motion -> zero jerk
    traj = np.stack([np.linspace(0, 9, 10), np.zeros(10)], axis=1)
    j = jerk(traj)
    assert j.shape[0] == 7
    assert np.allclose(j, 0.0)
    assert temporal_smoothness(traj) == pytest.approx(0.0)


def test_jerk_short_trajectory_empty():
    traj = np.zeros((2, 2))
    assert jerk(traj).size == 0
    assert temporal_smoothness(traj) == 0.0


# ---------------------------------------------------------------------------
# pose metrics
# ---------------------------------------------------------------------------


def test_pck_one_when_equal():
    rng = np.random.default_rng(1)
    kps = rng.normal(size=(5, 16, 2))
    res = pck(kps, kps, thr=0.05, norm=np.ones(5))
    assert res["mean"] == pytest.approx(1.0)
    assert np.allclose(res["per_joint"], 1.0)


def test_pck_threshold_behavior():
    gt = np.zeros((1, 1, 2))
    pred = np.zeros((1, 1, 2))
    pred[0, 0, 0] = 0.1  # distance 0.1
    # norm=1: 0.1 < 0.05 ? no
    assert pck(pred, gt, thr=0.05, norm=1.0)["mean"] == pytest.approx(0.0)
    # threshold 0.2 -> correct
    assert pck(pred, gt, thr=0.2, norm=1.0)["mean"] == pytest.approx(1.0)


def test_mpjpe_zero_when_equal():
    rng = np.random.default_rng(2)
    kps = rng.normal(size=(4, 16, 3))
    res = mean_per_joint_error(kps, kps)
    assert res["mean"] == pytest.approx(0.0, abs=1e-12)


def test_mpjpe_known_value():
    gt = np.zeros((1, 2, 2))
    pred = np.array([[[3.0, 4.0], [0.0, 0.0]]])  # joint0 dist 5, joint1 dist 0
    res = mean_per_joint_error(pred, gt)
    assert res["per_joint"][0] == pytest.approx(5.0)
    assert res["per_joint"][1] == pytest.approx(0.0)
    assert res["mean"] == pytest.approx(2.5)


def test_oks_one_when_equal():
    rng = np.random.default_rng(3)
    kps = rng.normal(size=(3, 16, 3))  # 2D + visibility=normal>... ensure visible
    kps[..., 2] = 1.0  # all visible
    areas = np.full(3, 100.0)
    out = oks(kps, kps, areas)
    assert out.shape == (3,)
    assert np.allclose(out, 1.0)


def test_oks_decreases_with_error():
    gt = np.zeros((1, 16, 3))
    gt[..., 2] = 1.0
    pred = gt.copy()
    pred[0, 0, 0] = 5.0  # shift one keypoint
    areas = np.array([100.0])
    perfect = oks(gt, gt, areas)[0]
    shifted = oks(pred, gt, areas)[0]
    assert perfect == pytest.approx(1.0)
    assert shifted < perfect
