"""Regression tests for RehabAL's clinical-term skeleton wiring.

These guard the CRITICAL bug where RehabAL's joint triplets / limbs / L-R pairs
were hard-coded to the AIHub-16 layout while the data is rehab26 (26 joints),
so the angle / ROM / compensation terms silently computed on the wrong joints.
"""

from __future__ import annotations

import numpy as np

from rehabal_pose.al.base import Pool
from rehabal_pose.al.rehabal import (
    JOINT_TRIPLETS,
    LIMBS,
    LR_JOINT_PAIRS,
    RehabALStrategy,
)
from rehabal_pose.data.skeletons import get_skeleton

REHAB26 = get_skeleton("rehab26")


def test_triplets_map_to_named_rehab26_joints() -> None:
    """Each default triplet's vertex must be an anatomically named joint.

    If someone reverts to the AIHub-16 indices, the named joints no longer line
    up (e.g. (2,3,4) would be Spine1/Neck/Head, not an elbow) and this fails.
    """
    names = REHAB26.joint_names
    # vertex (middle index) -> expected substring of the rehab26 joint name
    expected_vertex = {
        (7, 8, 9): "ForeArm",   # left elbow vertex = LeftForeArm
        (12, 13, 14): "ForeArm",  # right elbow
        (16, 17, 18): "Leg",     # left knee vertex = LeftLeg
        (21, 22, 23): "Leg",     # right knee
    }
    for trip, sub in expected_vertex.items():
        assert trip in JOINT_TRIPLETS, f"{trip} missing from JOINT_TRIPLETS"
        vertex_name = names[trip[1]]
        assert sub in vertex_name, f"vertex of {trip} is {vertex_name!r}, expected ~{sub}"


def test_all_indices_within_rehab26() -> None:
    k = REHAB26.num_joints  # 26
    for a, b, c in JOINT_TRIPLETS:
        assert max(a, b, c) < k
    for i, j in LIMBS:
        assert max(i, j) < k
    for left, right in LR_JOINT_PAIRS:
        assert max(*left, *right) < k
    # Limbs should be real rehab26 bone edges.
    edges = {tuple(sorted(e)) for e in REHAB26.edges}
    for i, j in LIMBS:
        assert tuple(sorted((i, j))) in edges, f"limb {(i, j)} is not a rehab26 edge"


def test_rom_term_responds_to_the_correct_joint() -> None:
    """Bending the LEFT ELBOW joint (LeftForeArm=8) must change the ROM term;
    bending an unrelated joint should not (much)."""
    N, K = 6, 26
    P = np.zeros((N, K, 2), dtype=np.float64)
    # straight left arm: LeftArm(7)-LeftForeArm(8)-LeftHand(9) collinear
    P[:, 7] = [0.0, 0.0]
    P[:, 8] = [1.0, 0.0]
    P[:, 9] = [2.0, 0.0]
    P[3, 9] = [1.0, 1.0]   # sample 3: bend the elbow to ~90deg
    preds = np.concatenate([P, np.ones((N, K, 1))], axis=-1)
    pool = Pool(
        sample_ids=[str(i) for i in range(N)], predictions=preds,
        clip_ids=["c"] * N, frame_ids=list(range(N)),
        metadata={"disease": [None] * N, "exercise": ["x"] * N, "camera_id": ["c"] * N},
    )
    strat = RehabALStrategy()
    rom = strat.term_rom_criticality(pool)
    # |cos|: straight (180deg)=1 -> max criticality; bent (~90deg)=0 -> min.
    assert rom[3] < rom[0], "ROM term did not respond to the left-elbow joint"
    assert np.isfinite(strat.score(pool)).all()


def test_weight_keys_match_six_term_implementation() -> None:
    """Ablation relies on these exact keys; guards config<->code drift."""
    s = RehabALStrategy()
    assert set(s.weights) == {
        "uncertainty", "temporal", "angle", "rom", "compensation", "rarity",
    }
    # zeroing one term is the ablation mechanism
    s2 = RehabALStrategy(weights={"angle": 0.0})
    assert s2.weights["angle"] == 0.0
