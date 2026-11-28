from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from rehabal_pose.pipelines.evaluate_rehab import (
    REHAB26_JOINT_TRIPLETS,
    evaluate_rehab,
    evaluate_rehab_files,
    main,
)


def test_no_heavy_imports_on_module_load():
    code = (
        "import rehabal_pose.pipelines.evaluate_rehab, sys; "
        "bad = [m for m in ('torch', 'ultralytics', 'cv2', 'mmpose') "
        "       if m in sys.modules]; "
        "assert not bad, f'heavy deps leaked: {bad}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Heavy-import guard failed:\n{result.stdout}{result.stderr}"
    )


def _make_straight_kps(T: int = 20, offset: float = 0.0) -> np.ndarray:
    kps = np.zeros((T, 26, 2), dtype=np.float64)
    kps[:, 7, :] = [1.0, 0.0]
    kps[:, 8, :] = [0.0, 0.0]
    kps[:, 9, :] = [0.0, 1.0]
    kps[:, 12, :] = [2.0, 0.0]
    kps[:, 13, :] = [1.0, 0.0]
    kps[:, 14, :] = [1.0, 1.0]
    kps[:, 16, :] = [3.0, 0.0]
    kps[:, 17, :] = [2.0, 0.0]
    kps[:, 18, :] = [2.0, 1.0]

    if offset != 0.0:
        rad = np.deg2rad(offset)
        cos_o, sin_o = np.cos(rad), np.sin(rad)
        dx, dy = 0.0, 1.0
        new_dx = cos_o * dx - sin_o * dy
        new_dy = sin_o * dx + cos_o * dy
        kps[:, 9, :] = [new_dx, new_dy]

    return kps


def _make_sine_kps(T: int = 40) -> np.ndarray:
    kps = np.zeros((T, 26, 2), dtype=np.float64)
    t = np.linspace(0, np.pi, T)
    kps[:, 7, :] = np.c_[np.ones(T), np.zeros(T)]
    kps[:, 8, :] = np.zeros((T, 2))
    kps[:, 9, 0] = 0.0
    kps[:, 9, 1] = 0.5 + np.sin(t)
    return kps


def test_evaluate_rehab_returns_expected_keys():
    kps = _make_straight_kps()
    res = evaluate_rehab(kps, kps)
    assert "n_frames" in res
    assert "num_joints" in res
    assert "joint_angle_mae" in res
    assert "rom_error" in res
    assert "peak_angle_error" in res
    assert "smoothness_error" in res
    assert "assessment" in res


def test_evaluate_rehab_shape_metadata():
    T, K = 15, 26
    kps = _make_straight_kps(T=T)
    res = evaluate_rehab(kps, kps)
    assert res["n_frames"] == T
    assert res["num_joints"] == K


def test_joint_angle_mae_zero_when_equal():
    kps = _make_straight_kps()
    res = evaluate_rehab(kps, kps)
    mae = res["joint_angle_mae"]
    for name in ("left_elbow", "right_elbow", "left_knee"):
        assert mae[name] == pytest.approx(0.0, abs=1e-8), f"MAE nonzero for {name}"
    assert mae["mean"] == pytest.approx(0.0, abs=1e-8)


def test_joint_angle_mae_known_offset():
    offset = 10.0
    gt_kps = _make_straight_kps(offset=0.0)
    pred_kps = _make_straight_kps(offset=offset)
    res = evaluate_rehab(pred_kps, gt_kps)
    mae = res["joint_angle_mae"]
    assert mae["left_elbow"] == pytest.approx(offset, abs=0.5)


def test_rom_error_zero_when_equal():
    kps = _make_sine_kps()
    res = evaluate_rehab(kps, kps)
    rom = res["rom_error"]
    assert rom["left_elbow"] == pytest.approx(0.0, abs=1e-8)
    assert rom["mean"] == pytest.approx(0.0, abs=1e-8)


def test_rom_and_peak_ignore_visibility_channel():
    rng = np.random.default_rng(11)
    xy = _make_sine_kps()
    T, K, _ = xy.shape
    pred_vis = rng.uniform(0.3, 0.99, size=(T, K, 1))
    gt_vis = rng.integers(0, 3, size=(T, K, 1)).astype(np.float64)
    pred3 = np.concatenate([xy, pred_vis], axis=-1)
    gt3 = np.concatenate([xy, gt_vis], axis=-1)

    res2 = evaluate_rehab(xy, xy)
    res3 = evaluate_rehab(pred3, gt3)

    for metric in ("rom_error", "peak_angle_error"):
        for name, val2 in res2[metric].items():
            assert res3[metric][name] == pytest.approx(val2, abs=1e-9, nan_ok=True)
        assert res3[metric]["mean"] == pytest.approx(0.0, abs=1e-9)


def test_peak_angle_error_zero_when_equal():
    kps = _make_sine_kps()
    res = evaluate_rehab(kps, kps)
    peak = res["peak_angle_error"]
    assert peak["left_elbow"] == pytest.approx(0.0, abs=1e-8)


def test_smoothness_error_zero_when_equal():
    kps = _make_sine_kps()
    res = evaluate_rehab(kps, kps)
    assert res["smoothness_error"] == pytest.approx(0.0, abs=1e-9)


def test_smoothness_error_increases_with_jitter():
    kps = _make_sine_kps(T=40)
    rng = np.random.default_rng(42)
    noisy = kps.copy()
    noisy += rng.normal(0, 0.5, size=kps.shape)
    res = evaluate_rehab(noisy, kps)
    assert res["smoothness_error"] > 0.0


def test_proxy_path_used_when_no_labels():
    kps = _make_straight_kps()
    res = evaluate_rehab(kps, kps)
    assess = res["assessment"]
    assert assess["proxy"] is True
    assert "proxy_deviation" in assess
    assert assess["proxy_deviation"] == pytest.approx(0.0, abs=1e-8)


def test_proxy_deviation_increases_with_noise():
    gt = _make_straight_kps()
    pred = gt.copy()
    pred += 1.0
    res = evaluate_rehab(pred, gt)
    assert res["assessment"]["proxy_deviation"] > 0.0


def test_labeled_quality_path():
    kps = _make_straight_kps()
    labels = {
        "pred_quality": [3.0, 2.5, 4.0],
        "gt_quality":   [3.0, 3.0, 4.0],
    }
    res = evaluate_rehab(kps, kps, assessment_labels=labels)
    assess = res["assessment"]
    assert assess.get("proxy") is False
    assert "quality_score_mae" in assess
    assert "quality_score_corr_pearson" in assess
    assert "quality_score_corr_spearman" in assess
    assert assess["quality_score_mae"] == pytest.approx(0.5 / 3, abs=1e-8)


def test_labeled_quality_perfect_corr():
    kps = _make_straight_kps()
    scores = [1.0, 2.0, 3.0, 4.0, 5.0]
    labels = {"pred_quality": scores, "gt_quality": scores}
    res = evaluate_rehab(kps, kps, assessment_labels=labels)
    assert res["assessment"]["quality_score_corr_pearson"] == pytest.approx(1.0)
    assert res["assessment"]["quality_score_corr_spearman"] == pytest.approx(1.0)


def test_labeled_correctness_f1_perfect():
    kps = _make_straight_kps()
    labels = {
        "pred_correctness": [1, 0, 1, 1],
        "gt_correctness":   [1, 0, 1, 1],
    }
    res = evaluate_rehab(kps, kps, assessment_labels=labels)
    assess = res["assessment"]
    assert "correctness_f1" in assess
    assert assess["correctness_f1"] == pytest.approx(1.0)
    assert assess["correctness_precision"] == pytest.approx(1.0)
    assert assess["correctness_recall"] == pytest.approx(1.0)


def test_labeled_correctness_f1_known():
    kps = _make_straight_kps()
    labels = {
        "pred_correctness": [1, 1, 0],
        "gt_correctness":   [1, 0, 1],
    }
    res = evaluate_rehab(kps, kps, assessment_labels=labels)
    assess = res["assessment"]
    assert assess["correctness_f1"] == pytest.approx(0.5)


def test_custom_joint_triplets():
    kps = _make_straight_kps()
    custom = {"test_joint": (7, 8, 9)}
    res = evaluate_rehab(kps, kps, joint_triplets=custom)
    assert "test_joint" in res["joint_angle_mae"]
    assert list(k for k in res["joint_angle_mae"] if k != "mean") == ["test_joint"]


def test_single_frame_promoted():
    kps = _make_straight_kps(T=1)[0]
    res = evaluate_rehab(kps, kps)
    assert res["n_frames"] == 1
    assert res["joint_angle_mae"]["left_elbow"] == pytest.approx(0.0, abs=1e-8)


def test_shape_mismatch_raises():
    pred = np.zeros((10, 26, 2))
    gt = np.zeros((10, 17, 2))
    with pytest.raises(ValueError):
        evaluate_rehab(pred, gt)


def test_result_json_serialisable():
    kps = _make_straight_kps()
    res = evaluate_rehab(kps, kps)
    text = json.dumps(res)
    assert isinstance(text, str)


def test_evaluate_rehab_files_proxy(tmp_path):
    kps = _make_straight_kps()
    pred_p = tmp_path / "pred.npy"
    gt_p = tmp_path / "gt.npy"
    out_p = tmp_path / "metrics.json"
    np.save(pred_p, kps)
    np.save(gt_p, kps)

    res = evaluate_rehab_files(pred_p, gt_p, out_p)
    assert out_p.exists()
    on_disk = json.loads(out_p.read_text(encoding="utf-8"))
    assert on_disk["n_frames"] == kps.shape[0]
    assert on_disk["assessment"]["proxy"] is True
    assert res["joint_angle_mae"]["mean"] == pytest.approx(0.0, abs=1e-8)


def test_evaluate_rehab_files_with_labels(tmp_path):
    kps = _make_straight_kps()
    pred_p = tmp_path / "pred.npy"
    gt_p = tmp_path / "gt.npy"
    np.save(pred_p, kps)
    np.save(gt_p, kps)

    labels = {
        "pred_quality": [2.0, 4.0],
        "gt_quality":   [2.0, 4.0],
        "pred_correctness": [1, 0],
        "gt_correctness":   [1, 0],
    }
    labels_p = tmp_path / "labels.json"
    labels_p.write_text(json.dumps(labels), encoding="utf-8")

    res = evaluate_rehab_files(pred_p, gt_p, assessment_labels_path=labels_p)
    assert res["assessment"]["proxy"] is False
    assert res["assessment"]["quality_score_mae"] == pytest.approx(0.0)
    assert res["assessment"]["correctness_f1"] == pytest.approx(1.0)


def test_main_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0


def test_main_cli_writes_json(tmp_path):
    kps = _make_straight_kps()
    pred_p = tmp_path / "pred.npy"
    gt_p = tmp_path / "gt.npy"
    out_p = tmp_path / "metrics.json"
    np.save(pred_p, kps)
    np.save(gt_p, kps)

    rc = main(
        ["--pred", str(pred_p), "--gt", str(gt_p), "--out", str(out_p)]
    )
    assert rc == 0
    assert out_p.exists()
    data = json.loads(out_p.read_text(encoding="utf-8"))
    assert "joint_angle_mae" in data
    assert "assessment" in data
