from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest


_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def test_no_heavy_imports_on_module_load():
    code = (
        "import sys; sys.path.insert(0, r'" + str(_SCRIPTS_DIR) + "'); "
        "import run_downstream_eval; "
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


def test_load_gt_keypoints_parses_yolo_format(tmp_path):
    from run_downstream_eval import _load_gt_keypoints

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    K = 26
    kp_flat = np.arange(K * 3, dtype=np.float64) * 0.01
    row = [0, 0.5, 0.5, 0.3, 0.4] + kp_flat.tolist()
    (labels_dir / "frame_001.txt").write_text(
        " ".join(str(v) for v in row) + "\n", encoding="utf-8"
    )

    gt = _load_gt_keypoints(labels_dir, ["frame_001"])
    assert "frame_001" in gt
    arr = gt["frame_001"]
    assert arr.shape == (K, 3)
    np.testing.assert_allclose(arr[0, :3], kp_flat[:3], atol=1e-10)


def test_load_gt_keypoints_missing_file(tmp_path):
    from run_downstream_eval import _load_gt_keypoints

    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()

    gt = _load_gt_keypoints(labels_dir, ["missing_stem"])
    assert "missing_stem" in gt
    assert gt["missing_stem"].shape == (26, 3)
    assert np.all(gt["missing_stem"] == 0.0)


def test_group_by_clip():
    from run_downstream_eval import _group_by_clip

    stems = [
        "PM_045_rep00_f100",
        "PM_045_rep00_f20",
        "PM_045_rep00_f0",
        "PM_045_rep01_f001",
        "PM_099_rep00_f001",
        "standalone",
    ]
    groups = _group_by_clip(stems)
    assert "PM_045_rep00" in groups
    assert "PM_045_rep01" in groups
    assert "PM_099_rep00" in groups
    assert len(groups["PM_045_rep00"]) == 3
    assert groups["PM_045_rep00"] == [
        "PM_045_rep00_f0",
        "PM_045_rep00_f20",
        "PM_045_rep00_f100",
    ]


def test_stems_from_dir(tmp_path):
    from run_downstream_eval import _stems_from_dir

    img_dir = tmp_path / "images"
    img_dir.mkdir()
    (img_dir / "frame_001.jpg").write_bytes(b"\xff\xd8")
    (img_dir / "frame_002.jpg").write_bytes(b"\xff\xd8")
    (img_dir / "other.txt").write_text("ignored")

    stems = _stems_from_dir(img_dir)
    assert "frame_001" in stems
    assert "frame_002" in stems
    assert "other" not in stems


def _make_fake_gt_labels(labels_dir: Path, stems: list[str]) -> None:
    K = 26
    for stem in stems:
        kp_flat = [0.0] * (K * 3)
        row = [0, 0.5, 0.5, 0.3, 0.4] + kp_flat
        (labels_dir / f"{stem}.txt").write_text(
            " ".join(str(v) for v in row) + "\n", encoding="utf-8"
        )


def _fake_infer(weights: str, val_img_dir: Path, batch: int = 16) -> dict:
    stems = [p.stem for p in val_img_dir.glob("*.jpg")]
    rng = np.random.default_rng(42)
    return {s: (rng.random((26, 3)) * 0.8).astype(np.float64) for s in stems}


def test_eval_budget_produces_aggregate(tmp_path):
    from run_downstream_eval import _eval_budget

    val_dir = tmp_path / "images" / "val"
    val_dir.mkdir(parents=True)
    labels_dir = tmp_path / "labels" / "val"
    labels_dir.mkdir(parents=True)

    stems = [f"PM_000_rep00_f{i:03d}" for i in range(4)]
    for s in stems:
        (val_dir / f"{s}.jpg").write_bytes(b"\xff\xd8")
    _make_fake_gt_labels(labels_dir, stems)

    fake_weights = str(tmp_path / "best.pt")
    Path(fake_weights).write_bytes(b"")

    sizes = {stem: (1920.0, 1080.0) for stem in stems}
    with (
        patch("run_downstream_eval._infer_val_keypoints", side_effect=_fake_infer),
        patch("run_downstream_eval._image_sizes", return_value=sizes),
    ):
        result = _eval_budget(fake_weights, val_dir, labels_dir)

    assert "aggregate" in result
    assert "per_clip" in result
    agg = result["aggregate"]
    assert agg["n_clips"] >= 1
    assert "joint_angle_mae_mean" in agg
    assert "rom_error_mean" in agg
    assert "smoothness_error_mean" in agg
    assert "mean" in agg["joint_angle_mae_mean"]


def test_main_missing_val_dir(tmp_path):
    from run_downstream_eval import main

    rc = main([
        "--weights-list", str(tmp_path / "best.pt"),
        "--val-dir", str(tmp_path / "nonexistent_val"),
        "--labels-dir", str(tmp_path),
        "--out", str(tmp_path / "out.json"),
    ])
    assert rc == 1


def test_main_missing_labels_dir(tmp_path):
    from run_downstream_eval import main

    val_dir = tmp_path / "val"
    val_dir.mkdir()
    rc = main([
        "--weights-list", str(tmp_path / "best.pt"),
        "--val-dir", str(val_dir),
        "--labels-dir", str(tmp_path / "nonexistent_labels"),
        "--out", str(tmp_path / "out.json"),
    ])
    assert rc == 1


def test_main_budgets_length_mismatch(tmp_path):
    from run_downstream_eval import main

    val_dir = tmp_path / "val"; val_dir.mkdir()
    labels_dir = tmp_path / "labels"; labels_dir.mkdir()
    rc = main([
        "--weights-list", str(tmp_path / "a.pt"), str(tmp_path / "b.pt"),
        "--budgets", "0.05",
        "--val-dir", str(val_dir),
        "--labels-dir", str(labels_dir),
        "--out", str(tmp_path / "out.json"),
    ])
    assert rc == 1


def test_main_returns_failure_when_a_budget_fails(tmp_path):
    from run_downstream_eval import main

    val_dir = tmp_path / "val"; val_dir.mkdir()
    labels_dir = tmp_path / "labels"; labels_dir.mkdir()
    out_path = tmp_path / "out.json"
    rc = main([
        "--weights-list", str(tmp_path / "missing.pt"),
        "--budgets", "0.05",
        "--val-dir", str(val_dir),
        "--labels-dir", str(labels_dir),
        "--out", str(out_path),
    ])
    assert rc == 1
    assert "error" in json.loads(out_path.read_text(encoding="utf-8"))["budgets"]["budget_0.05"]


def test_main_writes_json_with_expected_schema(tmp_path):
    from run_downstream_eval import main

    val_dir = tmp_path / "images" / "val"; val_dir.mkdir(parents=True)
    labels_dir = tmp_path / "labels" / "val"; labels_dir.mkdir(parents=True)

    stems = [f"PM_000_rep00_f{i:03d}" for i in range(6)]
    for s in stems:
        (val_dir / f"{s}.jpg").write_bytes(b"\xff\xd8")
    _make_fake_gt_labels(labels_dir, stems)

    wts = []
    for b in ["005", "010"]:
        wp = tmp_path / f"best_{b}.pt"
        wp.write_bytes(b"")
        wts.append(str(wp))

    out_path = tmp_path / "downstream.json"

    sizes = {stem: (1920.0, 1080.0) for stem in stems}
    with (
        patch("run_downstream_eval._infer_val_keypoints", side_effect=_fake_infer),
        patch("run_downstream_eval._image_sizes", return_value=sizes),
    ):
        rc = main([
            "--weights-list", *wts,
            "--budgets", "0.05", "0.10",
            "--val-dir", str(val_dir),
            "--labels-dir", str(labels_dir),
            "--out", str(out_path),
        ])

    assert rc == 0
    assert out_path.is_file()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "budgets" in data
    assert "n_budgets" in data
    assert data["n_budgets"] == 2
    for bkey in ["budget_0.05", "budget_0.1"]:
        assert bkey in data["budgets"], f"missing key {bkey}: {list(data['budgets'])}"
        entry = data["budgets"][bkey]
        assert "aggregate" in entry, f"budget {bkey} missing 'aggregate'"
        agg = entry["aggregate"]
        assert agg["n_clips"] >= 1


def test_pixel_and_yolo_keypoints_share_isotropic_frame():
    from run_downstream_eval import (
        _pixel_keypoints_to_isotropic,
        _yolo_keypoints_to_isotropic,
    )

    rng = np.random.default_rng(0)
    W, H = 1920.0, 1080.0
    K = 26
    gt_norm = rng.random((K, 3))
    pred_px = gt_norm.copy()
    pred_px[:, 0] *= W
    pred_px[:, 1] *= H

    pred_iso = _pixel_keypoints_to_isotropic(pred_px, W, H)
    gt_iso = _yolo_keypoints_to_isotropic(gt_norm, W, H)
    np.testing.assert_allclose(pred_iso[:, :2], gt_iso[:, :2], atol=1e-9)
    np.testing.assert_allclose(pred_iso[:, 2], pred_px[:, 2], atol=1e-12)


def test_isotropic_keypoints_reject_bad_size():
    from run_downstream_eval import _pixel_keypoints_to_isotropic

    kp = np.arange(26 * 3, dtype=np.float64).reshape(26, 3)
    with pytest.raises(ValueError):
        _pixel_keypoints_to_isotropic(kp, 0.0, 0.0)
    with pytest.raises(ValueError):
        _pixel_keypoints_to_isotropic(kp, float("nan"), 5.0)


def test_isotropic_transform_preserves_pixel_geometry():
    from run_downstream_eval import (
        _pixel_keypoints_to_isotropic,
        _yolo_keypoints_to_isotropic,
    )
    from rehabal_pose.pipelines.evaluate_rehab import evaluate_rehab

    rng = np.random.default_rng(1)
    T, K = 6, 26
    W, H = 1280.0, 720.0
    gt_norm = rng.random((T, K, 2))
    pred_px = gt_norm.copy()
    pred_px[..., 0] *= W
    pred_px[..., 1] *= H

    pred_iso = np.stack(
        [_pixel_keypoints_to_isotropic(pred_px[t], W, H) for t in range(T)], axis=0
    )
    gt_iso = np.stack(
        [_yolo_keypoints_to_isotropic(gt_norm[t], W, H) for t in range(T)], axis=0
    )
    m_fixed = evaluate_rehab(pred_iso, gt_iso)
    assert abs(m_fixed["joint_angle_mae"]["mean"]) < 1e-6
    assert abs(m_fixed["rom_error"]["mean"]) < 1e-6

    m_mismatched = evaluate_rehab(pred_px, gt_norm)
    assert m_mismatched["joint_angle_mae"]["mean"] > 1.0
