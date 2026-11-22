"""Tests for converters, subject/disease splitting, and quality checks.

All tests run on tiny in-memory manifests and injected dummy keypoint arrays;
nothing touches real data or heavy ML dependencies.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from rehabal_pose.data import MANIFEST_COLUMNS, ManifestBuilder
from rehabal_pose.data.converters import to_coco_keypoints, to_yolo_pose
from rehabal_pose.data.quality_checks import run_quality_checks
from rehabal_pose.data.split import (
    cross_disease_split,
    subject_independent_split,
)

K = 16
IMG_W, IMG_H = 640, 480


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------


def _dummy_kpts(seed: int) -> np.ndarray:
    """Deterministic (K, 3) keypoints inside the image frame, all visible."""
    rng = np.random.default_rng(seed)
    xy = np.column_stack([
        rng.uniform(0, IMG_W, size=K),
        rng.uniform(0, IMG_H, size=K),
    ])
    vis = np.full((K, 1), 2.0)
    return np.concatenate([xy, vis], axis=1)


def _make_manifest(n: int = 4) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Build a small manifest DataFrame plus a sample_id -> kpts dict."""
    b = ManifestBuilder()
    kpts: dict[str, np.ndarray] = {}
    for i in range(n):
        sid = f"s{i:03d}"
        kpts[sid] = _dummy_kpts(i)
        b.add_row(
            sample_id=sid,
            dataset="aihub219",
            subject_id=f"subj{i % 2}",
            image_path=f"img_{sid}.jpg",
            width=IMG_W,
            height=IMG_H,
            disease="stroke",
            exercise="ex1",
            has_bbox=True,
            has_keypoints2d=True,
            bbox=[50.0, 40.0, 100.0, 120.0],
        )
    df = b.to_dataframe()
    assert list(df.columns) == MANIFEST_COLUMNS
    return df, kpts


def _kp_loader(kpts: dict[str, np.ndarray]):
    return lambda sid: kpts[sid]


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def test_to_coco_keypoints_valid(tmp_path):
    df, kpts = _make_manifest(4)
    out = tmp_path / "coco.json"
    res = to_coco_keypoints(df, _kp_loader(kpts), out, num_keypoints=K)
    assert res == out and out.is_file()

    coco = json.loads(out.read_text(encoding="utf-8"))
    assert set(coco) >= {"images", "annotations", "categories"}
    assert len(coco["images"]) == 4
    assert len(coco["annotations"]) == 4

    cat = coco["categories"][0]
    assert len(cat["keypoints"]) == K

    for ann in coco["annotations"]:
        assert len(ann["keypoints"]) == 3 * K
        assert len(ann["bbox"]) == 4
        assert ann["num_keypoints"] == K  # all visible
        # image_id must reference a real image
        assert any(im["id"] == ann["image_id"] for im in coco["images"])


def test_to_coco_skips_rows_without_keypoints(tmp_path):
    df, kpts = _make_manifest(3)
    df.loc[0, "has_keypoints2d"] = False
    out = tmp_path / "coco.json"
    to_coco_keypoints(df, _kp_loader(kpts), out, num_keypoints=K)
    coco = json.loads(out.read_text(encoding="utf-8"))
    assert len(coco["annotations"]) == 2


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def test_to_yolo_pose_valid(tmp_path):
    df, kpts = _make_manifest(4)
    out_dir = tmp_path / "yolo"
    size_lookup = lambda sid: (IMG_W, IMG_H)
    res = to_yolo_pose(df, _kp_loader(kpts), out_dir, size_lookup, num_keypoints=K)
    assert res == out_dir

    assert (out_dir / "data.yaml").is_file()
    label_files = sorted((out_dir / "labels").glob("*.txt"))
    assert len(label_files) == 4

    expected_fields = 5 + 3 * K
    for lf in label_files:
        line = lf.read_text(encoding="utf-8").strip()
        parts = line.split()
        assert len(parts) == expected_fields
        assert int(parts[0]) == 0  # class id
        # bbox cx, cy, w, h normalized within [0, 1]
        for v in parts[1:5]:
            assert 0.0 <= float(v) <= 1.0
        # keypoints: kx, ky normalized <= 1; kv integer flag
        kp_vals = parts[5:]
        for j in range(0, len(kp_vals), 3):
            kx, ky, kv = float(kp_vals[j]), float(kp_vals[j + 1]), kp_vals[j + 2]
            assert 0.0 <= kx <= 1.0
            assert 0.0 <= ky <= 1.0
            assert kv in {"0", "1", "2"}


def test_to_yolo_falls_back_to_manifest_size(tmp_path):
    df, kpts = _make_manifest(2)
    out_dir = tmp_path / "yolo2"
    # No size lookup -> must use width/height columns.
    to_yolo_pose(df, _kp_loader(kpts), out_dir, None, num_keypoints=K)
    assert len(list((out_dir / "labels").glob("*.txt"))) == 2


# ---------------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------------


def _multi_subject_manifest(n_subjects: int = 20, rows_per: int = 5) -> pd.DataFrame:
    b = ManifestBuilder()
    diseases = ["stroke", "frozen_shoulder", "acl"]
    for s in range(n_subjects):
        for r in range(rows_per):
            b.add_row(
                sample_id=f"s{s:03d}_{r}",
                dataset="aihub219",
                subject_id=f"subj{s:03d}",
                disease=diseases[s % len(diseases)],
                camera_id=f"cam{s % 4}",
                has_keypoints2d=True,
            )
    return b.to_dataframe()


def test_subject_independent_split_no_leakage_and_ratios():
    df = _multi_subject_manifest(20, 5)
    out = subject_independent_split(df, ratios=(0.7, 0.15, 0.15), seed=0)

    assert out["split"].notna().all()
    assert set(out["split"].unique()) <= {"train", "val", "test"}

    # No subject appears in more than one split.
    leak = out.groupby("subject_id")["split"].nunique()
    assert (leak == 1).all()

    # Group-level ratios should be roughly honored (20 subjects).
    subj_split = out.drop_duplicates("subject_id")["split"].value_counts()
    assert subj_split.get("train", 0) == 14  # round(0.7*20)
    assert subj_split.get("val", 0) == 3
    assert subj_split.get("test", 0) == 3


def test_subject_independent_split_is_deterministic():
    df = _multi_subject_manifest(12, 3)
    a = subject_independent_split(df, seed=42)
    b = subject_independent_split(df, seed=42)
    assert (a["split"].values == b["split"].values).all()


def test_cross_disease_split_leaves_one_out():
    df = _multi_subject_manifest(18, 4)
    out = cross_disease_split(df, holdout_disease="acl", seed=0)

    # Every 'acl' row is test; no other disease is test.
    test_rows = out[out["split"] == "test"]
    assert (test_rows["disease"] == "acl").all()
    assert set(out.loc[out["disease"] == "acl", "split"]) == {"test"}

    # 'acl' must not appear in train/val.
    tv = out[out["split"].isin(["train", "val"])]
    assert "acl" not in set(tv["disease"])
    assert set(tv["disease"]) <= {"stroke", "frozen_shoulder"}


# ---------------------------------------------------------------------------
# quality checks
# ---------------------------------------------------------------------------


def test_quality_checks_clean_manifest():
    df, kpts = _make_manifest(4)
    report, summary = run_quality_checks(
        df, kp_loader=_kp_loader(kpts), num_keypoints=K
    )
    assert report.ok
    assert summary["n_rows"] == 4
    assert summary["n_subjects"] == 2
    assert summary["per_disease"] == {"stroke": 4}
    assert summary["modality_counts"]["has_keypoints2d"] == 4


def test_quality_checks_flags_missing_and_bad_bbox():
    b = ManifestBuilder()
    b.add_row(sample_id=None, dataset="aihub219")  # missing required
    b.add_row(sample_id="ok", dataset="aihub219", width=640, height=480,
              bbox=[10.0, 10.0, -5.0, 20.0])  # bad bbox (negative w)
    report, summary = run_quality_checks(b)
    codes = {i.code for i in report.issues}
    assert "MISSING_REQUIRED" in codes
    assert "BAD_BBOX" in codes
    assert not report.ok


def test_quality_checks_empty_manifest():
    report, summary = run_quality_checks(ManifestBuilder())
    assert summary["n_rows"] == 0
    assert any(i.code == "EMPTY_MANIFEST" for i in report.issues)
