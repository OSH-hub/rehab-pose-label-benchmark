"""Unit tests for the YOLO-Pose dataset contract + model wrapper IMPORT safety.

Mock-first: NO heavy deps (torch / ultralytics / cv2), no GPU, no weights, no
real videos. These tests only exercise the pure-python dataset helpers and
assert that importing the model/dataset modules does not drag in the heavy ML
stack.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

from rehabal_pose.data.yolo_dataset import (
    REHAB26_NUM_KEYPOINTS,
    YOLO_POSE_LAYOUT,
    rehab26_flip_idx,
    write_pose_data_yaml,
    yolo_label_line,
)

HEAVY = ("torch", "ultralytics", "cv2")


# ---------------------------------------------------------------------------
# no-heavy-import guarantee
# ---------------------------------------------------------------------------


def test_importing_modules_does_not_import_heavy_deps():
    """Importing the model + dataset modules must not import torch/ultralytics/cv2."""
    # Drop any heavy modules a prior test may have imported, then (re)import ours
    # and confirm none of them got pulled in transitively.
    for name in list(sys.modules):
        if name in HEAVY or name.startswith(tuple(h + "." for h in HEAVY)):
            del sys.modules[name]

    importlib.import_module("rehabal_pose.data.yolo_dataset")
    importlib.import_module("rehabal_pose.models.yolo_pose")
    importlib.import_module("rehabal_pose.models")  # package re-export

    for h in HEAVY:
        assert h not in sys.modules, f"{h} was imported at module load time"


def test_is_available_is_pure_check():
    """is_available() must not import torch/ultralytics into sys.modules."""
    for name in list(sys.modules):
        if name in HEAVY or name.startswith(tuple(h + "." for h in HEAVY)):
            del sys.modules[name]
    from rehabal_pose.models.yolo_pose import is_available

    result = is_available()  # may be True (env has the deps) or False
    assert isinstance(result, bool)
    # find_spec must not execute the modules.
    assert "torch" not in sys.modules
    assert "ultralytics" not in sys.modules


# ---------------------------------------------------------------------------
# yolo_label_line
# ---------------------------------------------------------------------------


def _kpts(k: int = REHAB26_NUM_KEYPOINTS) -> np.ndarray:
    rng = np.random.default_rng(0)
    xy = rng.uniform(0.0, 1.0, size=(k, 2))
    vis = np.full((k, 1), 2.0)
    return np.concatenate([xy, vis], axis=1)


def test_label_line_token_count_is_83():
    line = yolo_label_line([0.5, 0.5, 0.2, 0.4], _kpts())
    tokens = line.split()
    assert len(tokens) == 5 + REHAB26_NUM_KEYPOINTS * 3 == 83
    assert tokens[0] == "0"
    assert not line.endswith("\n")


def test_label_line_clamps_out_of_range_values():
    kp = _kpts()
    kp[0] = [-0.5, 1.7, 2.0]  # out of [0,1]
    line = yolo_label_line([1.9, -0.3, 0.5, 0.5], kp)
    tokens = line.split()
    # bbox cx clamped to 1, cy clamped to 0
    assert float(tokens[1]) == pytest.approx(1.0)
    assert float(tokens[2]) == pytest.approx(0.0)
    # first keypoint px clamped to 0, py clamped to 1
    assert float(tokens[5]) == pytest.approx(0.0)
    assert float(tokens[6]) == pytest.approx(1.0)


def test_label_line_visibility_coerced_to_int_in_0_1_2():
    kp = _kpts()
    kp[0, 2] = 0.0
    kp[1, 2] = 1.0
    kp[2, 2] = 5.0  # clamp -> 2
    kp[3, 2] = -3.0  # clamp -> 0
    tokens = yolo_label_line([0.5, 0.5, 0.2, 0.2], kp).split()
    # visibility is the 3rd token of each keypoint triple (index 5 + 3*i + 2)
    assert tokens[5 + 0 * 3 + 2] == "0"
    assert tokens[5 + 1 * 3 + 2] == "1"
    assert tokens[5 + 2 * 3 + 2] == "2"
    assert tokens[5 + 3 * 3 + 2] == "0"


def test_label_line_rejects_wrong_keypoint_count():
    with pytest.raises(ValueError):
        yolo_label_line([0.5, 0.5, 0.2, 0.2], _kpts(17))


def test_label_line_rejects_bad_bbox():
    with pytest.raises(ValueError):
        yolo_label_line([0.5, 0.5, 0.2], _kpts())


# ---------------------------------------------------------------------------
# rehab26_flip_idx
# ---------------------------------------------------------------------------


def test_flip_idx_is_permutation_and_involutive():
    flip = rehab26_flip_idx()
    assert len(flip) == REHAB26_NUM_KEYPOINTS
    assert sorted(flip) == list(range(REHAB26_NUM_KEYPOINTS))  # permutation
    # Applying twice returns identity (left<->right is its own inverse).
    twice = [flip[flip[i]] for i in range(len(flip))]
    assert twice == list(range(len(flip)))


def test_flip_idx_swaps_left_right_and_fixes_midline():
    from rehabal_pose.data.skeletons import get_skeleton

    skel = get_skeleton("rehab26")
    names = list(skel.joint_names)
    flip = rehab26_flip_idx()
    # Midline joints map to themselves.
    for mid in ("Hips", "Spine", "Spine1", "Neck", "Head", "Head_end"):
        i = names.index(mid)
        assert flip[i] == i
    # LeftArm <-> RightArm.
    li, ri = names.index("LeftArm"), names.index("RightArm")
    assert flip[li] == ri and flip[ri] == li


# ---------------------------------------------------------------------------
# write_pose_data_yaml
# ---------------------------------------------------------------------------


def test_write_pose_data_yaml_has_kpt_shape_and_person(tmp_path):
    out = tmp_path / "ds"
    (out / "images" / "train").mkdir(parents=True)
    (out / "images" / "val").mkdir(parents=True)
    yaml_path = write_pose_data_yaml(
        out,
        out / "images" / "train",
        out / "images" / "val",
    )
    assert yaml_path.exists()
    text = yaml_path.read_text(encoding="utf-8")
    assert "kpt_shape: [26, 3]" in text
    assert "0: person" in text
    assert "flip_idx:" in text
    assert "train: images/train" in text
    assert "val: images/val" in text


def test_write_pose_data_yaml_parses_with_yaml(tmp_path):
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "ds"
    out.mkdir()
    yaml_path = write_pose_data_yaml(out, out / "images/train", out / "images/val")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [26, 3]
    assert data["names"] == {0: "person"}
    assert len(data["flip_idx"]) == 26
    assert sorted(data["flip_idx"]) == list(range(26))


def test_write_pose_data_yaml_custom_num_keypoints(tmp_path):
    yaml = pytest.importorskip("yaml")
    out = tmp_path / "ds17"
    out.mkdir()
    flip = list(range(17))
    yaml_path = write_pose_data_yaml(
        out,
        out / "images/train",
        out / "images/val",
        num_keypoints=17,
        flip_idx=flip,
    )
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    assert data["kpt_shape"] == [17, 3]
    assert data["flip_idx"] == flip


def test_write_pose_data_yaml_flip_idx_length_mismatch_raises(tmp_path):
    out = tmp_path / "bad"
    out.mkdir()
    with pytest.raises(ValueError):
        write_pose_data_yaml(
            out,
            out / "images/train",
            out / "images/val",
            flip_idx=[0, 1, 2],  # wrong length
        )


def test_layout_constant_documents_token_count():
    assert "83" in YOLO_POSE_LAYOUT
    assert "labels" in YOLO_POSE_LAYOUT
    assert "images" in YOLO_POSE_LAYOUT
