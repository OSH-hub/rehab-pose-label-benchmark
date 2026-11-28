from __future__ import annotations

import sys

import numpy as np
import pytest

from rehabal_pose.pipelines import evaluate_pose as ep


def test_no_heavy_imports():
    for mod in ("torch", "ultralytics", "cv2", "mmpose"):
        assert mod not in sys.modules, f"{mod} leaked into sys.modules"


def _make_gt(n=6, k=5, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(10.0, 200.0, size=(n, k, 2))
    vis = np.full((n, k, 1), 2.0)
    return np.concatenate([xy, vis], axis=2)


def test_pck_perfect_when_equal():
    gt = _make_gt()
    res = ep.evaluate_pose(gt.copy(), gt, thresholds=(0.05, 0.1, 0.2), norm=None)
    for key in ("0.05", "0.1", "0.2"):
        assert res["pck"][key] == pytest.approx(1.0)
    for vals in res["pck_per_joint"].values():
        assert all(v == pytest.approx(1.0) for v in vals)


def test_pck_decreases_with_noise():
    gt = _make_gt()
    diag = float(np.hypot(200.0, 200.0))
    norm = np.full(gt.shape[0], diag)
    rng = np.random.default_rng(1)

    pck_clean = ep.evaluate_pose(gt.copy(), gt, thresholds=(0.05,), norm=norm)["pck"]["0.05"]

    small = gt.copy()
    small[..., :2] += rng.normal(0.0, 3.0, size=small[..., :2].shape)
    pck_small = ep.evaluate_pose(small, gt, thresholds=(0.05,), norm=norm)["pck"]["0.05"]

    big = gt.copy()
    big[..., :2] += rng.normal(0.0, 30.0, size=big[..., :2].shape)
    pck_big = ep.evaluate_pose(big, gt, thresholds=(0.05,), norm=norm)["pck"]["0.05"]

    assert pck_clean == pytest.approx(1.0)
    assert pck_clean >= pck_small >= pck_big
    assert pck_big < pck_clean


def test_oks_perfect_when_equal():
    gt = _make_gt()
    res = ep.evaluate_pose(gt.copy(), gt)
    assert res["oks"]["mean"] == pytest.approx(1.0)
    assert res["oks"]["mAP"] == pytest.approx(1.0)
    for v in res["oks"]["AP"].values():
        assert v == pytest.approx(1.0)


def test_oks_decreases_with_noise():
    gt = _make_gt()
    rng = np.random.default_rng(2)
    noisy = gt.copy()
    noisy[..., :2] += rng.normal(0.0, 25.0, size=noisy[..., :2].shape)
    res = ep.evaluate_pose(noisy, gt)
    assert res["oks"]["mean"] < 1.0
    assert res["oks"]["mAP"] <= 1.0


def test_mpjpe_zero_when_equal():
    gt = _make_gt()
    res = ep.evaluate_pose(gt.copy(), gt)
    assert res["mpjpe"]["mean"] == pytest.approx(0.0)
    assert all(v == pytest.approx(0.0) for v in res["mpjpe"]["per_joint"])


def test_mpjpe_matches_known_offset():
    gt = _make_gt()
    pred = gt.copy()
    pred[..., 0] += 3.0
    pred[..., 1] += 4.0
    res = ep.evaluate_pose(pred, gt)
    assert res["mpjpe"]["mean"] == pytest.approx(5.0)


def test_single_sample_promoted():
    gt = _make_gt(n=1)[0]
    res = ep.evaluate_pose(gt.copy(), gt)
    assert res["n_samples"] == 1
    assert res["mpjpe"]["mean"] == pytest.approx(0.0)


def test_shape_mismatch_raises():
    gt = _make_gt(n=4, k=5)
    bad = _make_gt(n=4, k=6)
    with pytest.raises(ValueError):
        ep.evaluate_pose(bad, gt)


def test_supplied_areas_used():
    gt = _make_gt()
    res = ep.evaluate_pose(gt.copy(), gt, areas=1000.0)
    assert res["oks"]["derived_areas"] is False
    assert res["oks"]["mean"] == pytest.approx(1.0)


def test_evaluate_from_manifests_aligns_and_groups():
    rng = np.random.default_rng(3)
    k = 5
    kp = {sid: rng.uniform(10, 200, size=(k, 3)) for sid in ["a", "b", "c", "d"]}
    for v in kp.values():
        v[:, 2] = 2.0

    gt_manifest = [
        {"sample_id": "a", "exercise": "Ex1"},
        {"sample_id": "b", "exercise": "Ex1"},
        {"sample_id": "c", "exercise": "Ex2"},
        {"sample_id": "z", "exercise": "Ex9"},
    ]
    pred_manifest = [
        {"sample_id": "a"},
        {"sample_id": "b"},
        {"sample_id": "c"},
        {"sample_id": "q"},
    ]

    def loader(sid):
        return kp[sid]

    res = ep.evaluate_from_manifests(
        pred_manifest, gt_manifest, loader, loader, thresholds=(0.1,)
    )
    assert res["n_aligned"] == 3
    assert set(res["sample_ids"]) == {"a", "b", "c"}
    assert res["overall"]["pck"]["0.1"] == pytest.approx(1.0)
    assert res["overall"]["mpjpe"]["mean"] == pytest.approx(0.0)
    assert set(res["per_exercise"]) == {"Ex1", "Ex2"}
    assert res["per_exercise"]["Ex1"]["n_samples"] == 2
    assert res["per_exercise"]["Ex2"]["n_samples"] == 1


def test_evaluate_from_manifests_detects_noise_per_exercise():
    k = 4
    rng = np.random.default_rng(4)
    gt_kp = {sid: rng.uniform(10, 200, size=(k, 3)) for sid in ["a", "b"]}
    for v in gt_kp.values():
        v[:, 2] = 2.0
    pred_kp = {sid: v.copy() for sid, v in gt_kp.items()}
    pred_kp["b"][:, :2] += 50.0

    gt_manifest = [
        {"sample_id": "a", "exercise": "Ex1"},
        {"sample_id": "b", "exercise": "Ex2"},
    ]
    pred_manifest = list(gt_manifest)

    res = ep.evaluate_from_manifests(
        pred_manifest,
        gt_manifest,
        lambda s: pred_kp[s],
        lambda s: gt_kp[s],
        thresholds=(0.1,),
    )
    assert res["per_exercise"]["Ex1"]["mpjpe"]["mean"] == pytest.approx(0.0)
    assert res["per_exercise"]["Ex2"]["mpjpe"]["mean"] > 1.0


def test_evaluate_from_manifests_no_overlap():
    res = ep.evaluate_from_manifests(
        [{"sample_id": "x"}],
        [{"sample_id": "y"}],
        lambda s: np.zeros((3, 3)),
        lambda s: np.zeros((3, 3)),
    )
    assert res["n_aligned"] == 0
    assert res["overall"] is None
    assert res["per_exercise"] == {}


def test_evaluate_from_manifests_accepts_dataframe():
    pd = pytest.importorskip("pandas")
    k = 3
    rng = np.random.default_rng(5)
    kp = {sid: np.concatenate(
        [rng.uniform(10, 200, size=(k, 2)), np.full((k, 1), 2.0)], axis=1
    ) for sid in ["a", "b"]}
    df = pd.DataFrame(
        [{"sample_id": "a", "exercise": "Ex1"}, {"sample_id": "b", "exercise": "Ex1"}]
    )
    res = ep.evaluate_from_manifests(df, df, lambda s: kp[s], lambda s: kp[s])
    assert res["n_aligned"] == 2
    assert res["per_exercise"]["Ex1"]["n_samples"] == 2


def test_main_writes_metrics_json(tmp_path):
    gt = _make_gt()
    pred = gt.copy()
    pred[..., 0] += 3.0
    pred[..., 1] += 4.0
    pred_p = tmp_path / "pred.npy"
    gt_p = tmp_path / "gt.npy"
    out_p = tmp_path / "metrics.json"
    np.save(pred_p, pred)
    np.save(gt_p, gt)

    rc = ep.main(["--pred", str(pred_p), "--gt", str(gt_p), "--out", str(out_p)])
    assert rc == 0
    assert out_p.exists()

    import json

    metrics = json.loads(out_p.read_text(encoding="utf-8"))
    assert metrics["n_samples"] == gt.shape[0]
    assert metrics["mpjpe"]["mean"] == pytest.approx(5.0)
    assert "pck" in metrics and "oks" in metrics
