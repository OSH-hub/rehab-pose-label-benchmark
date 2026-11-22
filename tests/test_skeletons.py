"""Unit tests for the skeleton/joint-layout registry."""

from __future__ import annotations

import numpy as np
import pytest

from rehabal_pose.data.skeletons import (
    SKELETONS,
    Skeleton,
    get_skeleton,
    list_skeletons,
    make_skeleton,
    register_skeleton,
)

EXPECTED_LAYOUTS = {"coco17", "mpii16", "kinect25", "uiprmd22", "rehab26"}
EXPECTED_COUNTS = {
    "coco17": 17,
    "mpii16": 16,
    "kinect25": 25,
    "uiprmd22": 22,
    "rehab26": 26,
}


def test_expected_layouts_registered():
    assert EXPECTED_LAYOUTS.issubset(set(SKELETONS))
    assert EXPECTED_LAYOUTS.issubset(set(list_skeletons()))


@pytest.mark.parametrize("name", sorted(EXPECTED_LAYOUTS))
def test_num_joints_matches_names(name):
    skel = get_skeleton(name)
    assert skel.num_joints == len(skel.joint_names) == EXPECTED_COUNTS[name]


@pytest.mark.parametrize("name", sorted(SKELETONS))
def test_all_edge_indices_in_range(name):
    skel = get_skeleton(name)
    k = skel.num_joints
    for a, b in skel.edges:
        assert 0 <= a < k
        assert 0 <= b < k
    # edge_array helper shape
    ea = skel.edge_array()
    assert ea.ndim == 2 and ea.shape[1] == 2
    if len(skel.edges):
        assert ea.max() < k


@pytest.mark.parametrize("name", sorted(SKELETONS))
def test_parents_consistency(name):
    skel = get_skeleton(name)
    if skel.parents is None:
        return
    assert len(skel.parents) == skel.num_joints
    for p in skel.parents:
        assert p == -1 or 0 <= p < skel.num_joints


@pytest.mark.parametrize("name", sorted(SKELETONS))
def test_joint_names_unique(name):
    skel = get_skeleton(name)
    assert len(set(skel.joint_names)) == skel.num_joints


def test_index_roundtrip():
    skel = get_skeleton("coco17")
    assert skel.index("nose") == 0
    assert skel.index("right_ankle") == skel.num_joints - 1
    assert skel.joint_names[skel.index("left_hip")] == "left_hip"
    assert skel.has_joint("left_hip")
    assert not skel.has_joint("tail")


def test_index_unknown_raises():
    skel = get_skeleton("mpii16")
    with pytest.raises(KeyError):
        skel.index("nonexistent_joint")


def test_get_skeleton_case_insensitive():
    assert get_skeleton("COCO17") is get_skeleton("coco17")


def test_get_skeleton_unknown_raises():
    with pytest.raises(KeyError):
        get_skeleton("not_a_skeleton")


def test_kinect25_known_joints():
    skel = get_skeleton("kinect25")
    assert skel.index("spine_base") == 0
    assert skel.index("head") == 3
    assert skel.parents[0] == -1  # root


def test_make_and_register_custom():
    sk = make_skeleton(
        "tiny3",
        ["a", "b", "c"],
        edges=[(0, 1), (1, 2)],
        parents=[-1, 0, 1],
    )
    assert isinstance(sk, Skeleton)
    assert sk.num_joints == 3
    register_skeleton(sk, overwrite=True)
    assert get_skeleton("tiny3").num_joints == 3


def test_make_skeleton_rejects_bad_edge():
    with pytest.raises(ValueError):
        make_skeleton("bad", ["a", "b"], edges=[(0, 5)])


def test_register_duplicate_without_overwrite_raises():
    with pytest.raises(ValueError):
        register_skeleton(
            make_skeleton("coco17", ["x"]),
        )


def test_register_requires_args():
    with pytest.raises(ValueError):
        register_skeleton()


def test_edge_array_dtype():
    ea = get_skeleton("coco17").edge_array()
    assert ea.dtype == np.int64
