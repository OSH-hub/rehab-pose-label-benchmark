"""Unit tests for frame extraction + the YOLO-Pose dataset builder.

Mock-first: NO heavy deps (cv2 / torch / ultralytics), no GPU, no real video
codecs. The default ``frame_provider`` is replaced by a tiny synthetic one that
writes placeholder JPGs, so ``VideoFrameExtractor``'s cv2 path is never
exercised here. The image-sequence fallback (which needs no cv2) and the
cv2-import guard are tested directly.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rehabal_pose.data.frames import (
    VideoFrameExtractor,
    add_representative_frames,
    build_yolo_pose_dataset,
)


def _fake_provider(video_path, frame_indices, out_dir, stem):
    """Write a tiny placeholder JPG per requested frame (no cv2)."""
    written = []
    for idx in frame_indices:
        p = Path(out_dir) / f"{stem}_f{idx}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")
        written.append(p)
    return written


def test_add_representative_frames_sets_frame_path(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "sample_id": ["c1", "c2", "c3"],
            "video_id": ["v1", "v2", "v3"],
            "video_path": [None, None, None],
        }
    )
    kp = {
        "c1": np.zeros((10, 26, 3)),
        "c2": np.zeros((5, 26, 3)),
        "c3": np.zeros((0, 26, 3)),  # empty -> null frame_path
    }
    out = add_representative_frames(
        df,
        lambda s: kp[s],
        tmp_path / "frames",
        video_root=tmp_path / "vids",   # makes the default resolver yield a path
        frame_provider=_fake_provider,
    )
    assert "frame_path" in out.columns
    assert out.loc[out.sample_id == "c1", "frame_path"].notna().all()
    assert out.loc[out.sample_id == "c2", "frame_path"].notna().all()
    assert out.loc[out.sample_id == "c3", "frame_path"].isna().all()
    # representative frame = middle of the clip (10//2 = 5, 5//2 = 2).
    assert out.loc[out.sample_id == "c1", "frame_path"].iloc[0].endswith("_f5.jpg")
    assert out.loc[out.sample_id == "c2", "frame_path"].iloc[0].endswith("_f2.jpg")
from rehabal_pose.data.schemas import MANIFEST_COLUMNS

HEAVY = ("torch", "ultralytics", "cv2")
NUM_KP = 26
W, H = 640.0, 480.0


# ---------------------------------------------------------------------------
# helpers / fixtures
# ---------------------------------------------------------------------------


def _make_manifest(specs):
    """Build a canonical-column manifest from (sample_id, subject_id, video_id)."""
    rows = []
    for sid, subj, vid in specs:
        row = {c: None for c in MANIFEST_COLUMNS}
        row["sample_id"] = sid
        row["dataset"] = "rehab24_6"
        row["subject_id"] = subj
        row["video_id"] = vid
        row["video_path"] = f"{vid}.mp4"
        row["has_keypoints2d"] = True
        rows.append(row)
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def _kp_array(t: int, k: int = NUM_KP, seed: int = 0) -> np.ndarray:
    """A (T, k, 3) [x, y, vis] keypoint array in pixel coords within W x H."""
    rng = np.random.default_rng(seed)
    arr = np.empty((t, k, 3), dtype=np.float64)
    arr[..., 0] = rng.uniform(0, W, size=(t, k))
    arr[..., 1] = rng.uniform(0, H, size=(t, k))
    arr[..., 2] = 2.0
    return arr


class _SyntheticProvider:
    """A frame_provider that writes tiny placeholder JPGs (no codec needed)."""

    def __init__(self):
        self.calls = []

    def __call__(self, video_path, frame_indices, out_dir, stem):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.calls.append((video_path, list(frame_indices), str(out), stem))
        written = []
        for idx in frame_indices:
            p = out / f"{stem}_f{idx}.jpg"
            # 1x1 placeholder bytes; builder never reads pixels (size via lookup).
            p.write_bytes(b"\xff\xd8\xff\xd9")
            written.append(p)
        return written


# ---------------------------------------------------------------------------
# no-heavy-import guarantee
# ---------------------------------------------------------------------------


def test_importing_frames_does_not_import_heavy_deps():
    for name in list(sys.modules):
        if name in HEAVY or name.startswith(tuple(h + "." for h in HEAVY)):
            del sys.modules[name]
    importlib.import_module("rehabal_pose.data.frames")
    for h in HEAVY:
        assert h not in sys.modules, f"{h} imported at module load time"


def test_cv2_path_is_import_guarded():
    """VideoFrameExtractor construction + image-seq path must not import cv2."""
    for name in list(sys.modules):
        if name == "cv2" or name.startswith("cv2."):
            del sys.modules[name]
    ex = VideoFrameExtractor()
    assert ex.ext == ".jpg"
    assert "cv2" not in sys.modules


# ---------------------------------------------------------------------------
# VideoFrameExtractor image-sequence fallback (no cv2)
# ---------------------------------------------------------------------------


def test_image_sequence_fallback_copies_selected_frames(tmp_path):
    seq = tmp_path / "frames"
    seq.mkdir()
    for i in range(5):
        (seq / f"{i:06d}.jpg").write_bytes(bytes([i]))
    out = tmp_path / "out"
    ex = VideoFrameExtractor()
    written = ex.extract(seq, [0, 2, 4], out, "clipA")
    names = sorted(p.name for p in written)
    assert names == ["clipA_f0.jpg", "clipA_f2.jpg", "clipA_f4.jpg"]
    assert all(p.exists() for p in written)
    assert "cv2" not in sys.modules


def test_image_sequence_skips_out_of_range(tmp_path):
    seq = tmp_path / "frames"
    seq.mkdir()
    for i in range(3):
        (seq / f"{i:06d}.png").write_bytes(b"x")
    out = tmp_path / "out"
    written = VideoFrameExtractor().extract(seq, [1, 99], out, "c")
    assert [p.name for p in written] == ["c_f1.jpg"]


# ---------------------------------------------------------------------------
# build_yolo_pose_dataset
# ---------------------------------------------------------------------------


def _build(tmp_path, **kwargs):
    manifest = _make_manifest(
        [
            ("S01_rep00", "1", "S01"),
            ("S01_rep01", "1", "S01"),
            ("S02_rep00", "2", "S02"),
            ("S03_rep00", "3", "S03"),
        ]
    )
    lengths = {"S01_rep00": 4, "S01_rep01": 3, "S02_rep00": 5, "S03_rep00": 2}

    def kp_loader(sid):
        return _kp_array(lengths[sid], seed=hash(sid) % 1000)

    def size_lookup(sid):
        return (W, H)

    out = build_yolo_pose_dataset(
        manifest,
        kp_loader,
        tmp_path / "ds",
        image_size_lookup=size_lookup,
        frame_provider=_SyntheticProvider(),
        seed=0,
        **kwargs,
    )
    return out, lengths


def test_build_creates_layout_and_matching_counts(tmp_path):
    out, lengths = _build(tmp_path)
    for split in ("train", "val"):
        assert (out / "images" / split).is_dir()
        assert (out / "labels" / split).is_dir()
    # Every image has a matching label (same stem) and vice versa.
    for split in ("train", "val"):
        imgs = {p.stem for p in (out / "images" / split).glob("*.jpg")}
        lbls = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        assert imgs == lbls, f"image/label stem mismatch in {split}"
    # Total labels == total sampled frames (stride 1 -> every frame).
    total = sum(
        len(list((out / "labels" / s).glob("*.txt"))) for s in ("train", "val")
    )
    assert total == sum(lengths.values())


def test_build_labels_have_83_normalized_tokens(tmp_path):
    out, _ = _build(tmp_path)
    txts = list((out / "labels" / "train").glob("*.txt")) + list(
        (out / "labels" / "val").glob("*.txt")
    )
    assert txts
    for txt in txts:
        line = txt.read_text(encoding="utf-8").strip()
        tokens = line.split()
        assert len(tokens) == 5 + NUM_KP * 3 == 83
        assert tokens[0] == "0"
        # bbox + keypoint xy normalized to [0, 1]; visibility ints in {0,1,2}.
        for i in range(1, 5):
            assert 0.0 <= float(tokens[i]) <= 1.0
        for k in range(NUM_KP):
            base = 5 + 3 * k
            assert 0.0 <= float(tokens[base]) <= 1.0
            assert 0.0 <= float(tokens[base + 1]) <= 1.0
            assert tokens[base + 2] in {"0", "1", "2"}


def test_build_data_yaml_has_kpt_shape(tmp_path):
    out, _ = _build(tmp_path)
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [26, 3]
    assert data["names"] == {0: "person"}
    assert data["train"] == "images/train"
    assert data["val"] == "images/val"


def test_build_split_has_no_subject_overlap(tmp_path):
    out, _ = _build(tmp_path)
    # Recover the subject prefix (S01/S02/S03) from each frame stem.
    def subjects(split):
        return {
            p.stem.split("_rep")[0]
            for p in (out / "images" / split).glob("*.jpg")
        }

    train_subj = subjects("train")
    val_subj = subjects("val")
    assert train_subj and val_subj  # both non-empty (3 subjects, frac 0.2 -> 1 val)
    assert train_subj.isdisjoint(val_subj), "subject leaked across train/val"


def test_build_explicit_val_subjects(tmp_path):
    out, _ = _build(tmp_path, val_subjects=["2"])
    val_stems = [p.stem for p in (out / "images" / "val").glob("*.jpg")]
    train_stems = [p.stem for p in (out / "images" / "train").glob("*.jpg")]
    assert all(s.startswith("S02") for s in val_stems)
    assert all(not s.startswith("S02") for s in train_stems)
    assert val_stems  # non-empty


def test_build_frame_stride_subsamples(tmp_path):
    out, lengths = _build(tmp_path, frame_stride=2)
    total = sum(
        len(list((out / "labels" / s).glob("*.txt"))) for s in ("train", "val")
    )
    expected = sum(len(range(0, t, 2)) for t in lengths.values())
    assert total == expected


def test_build_skips_bad_keypoint_shape(tmp_path):
    manifest = _make_manifest([("S01_rep00", "1", "S01"), ("S02_rep00", "2", "S02")])

    def kp_loader(sid):
        if sid == "S01_rep00":
            return _kp_array(3, k=17)  # wrong joint count -> skipped
        return _kp_array(3)

    out = build_yolo_pose_dataset(
        manifest,
        kp_loader,
        tmp_path / "ds",
        image_size_lookup=lambda s: (W, H),
        frame_provider=_SyntheticProvider(),
        val_subjects=["2"],
    )
    all_lbls = list((out / "labels" / "train").glob("*.txt")) + list(
        (out / "labels" / "val").glob("*.txt")
    )
    stems = {p.stem.split("_f")[0] for p in all_lbls}
    assert "S01_rep00" not in stems
    assert "S02_rep00" in stems


def test_build_invalid_frame_stride_raises(tmp_path):
    manifest = _make_manifest([("S01_rep00", "1", "S01")])
    with pytest.raises(ValueError):
        build_yolo_pose_dataset(
            manifest,
            lambda s: _kp_array(3),
            tmp_path / "ds",
            image_size_lookup=lambda s: (W, H),
            frame_provider=_SyntheticProvider(),
            frame_stride=0,
        )
