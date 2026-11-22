"""Deterministically regenerate the tiny FAKE REHAB24-6 mock dataset.

Schema-faithful, self-contained fixture for the REHAB24-6 parser tests. It
contains NO real data; everything is generated from a fixed seed so repeated
runs are byte-stable.

Layout produced under this directory (mirrors the real Zenodo layout)::

    marker_names.txt                              # 41 mocap marker names
    joints_names.txt                              # 26 skeleton joint names
    Segmentation.csv                              # ';'-delimited, real columns
    videos/<video_id>.mp4                         # 0-byte stub (NO real video)
    2d_joints/Ex<k>/<video_id>-c<cam>-30fps.npy   # (T, 26, 2) pixel coords
    3d_joints/Ex<k>/<video_id>-c<cam>-30fps.npy   # (T, 26, 3) cm coords

Run directly to regenerate::

    python tests/fixtures/rehab24_6_mock/make_fixtures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

NUM_JOINTS = 26
NUM_MARKERS = 41
SEED = 20240607

#: 26 REHAB24-6 skeleton joint names — CONFIRMED from the real Zenodo
#: joints_names.txt (record 13305826); mirrors the "rehab26" skeleton.
JOINT_NAMES = [
    "Hips", "Spine", "Spine1", "Neck", "Head", "Head_end",
    "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand", "LeftHand_end",
    "RightShoulder", "RightArm", "RightForeArm", "RightHand", "RightHand_end",
    "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToeBase_end",
    "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToeBase_end",
]
assert len(JOINT_NAMES) == NUM_JOINTS

#: 41 REHAB24-6 mocap marker names — CONFIRMED from the real marker_names.txt.
MARKER_NAMES = [
    "BackLeft", "BackRight", "BackTop", "Chest", "HeadFront", "HeadSide",
    "HeadTop", "LAnkleOut", "LElbowOut", "LHandOut", "LHeel", "LKneeOut",
    "LShin", "LShoulderBack", "LShoulderTop", "LThigh", "LToeIn", "LToeOut",
    "LToeTip", "LUArmHigh", "LWristIn", "LWristOut", "RAnkleOut", "RElbowOut",
    "RHandOut", "RHeel", "RKneeOut", "RShin", "RShoulderBack", "RShoulderTop",
    "RThigh", "RToeIn", "RToeOut", "RToeTip", "RUArmHigh", "RWristIn",
    "RWristOut", "WaistLBack", "WaistLFront", "WaistRBack", "WaistRFront",
]
assert len(MARKER_NAMES) == NUM_MARKERS

#: Recording specs. ``recording`` doubles as the real ``video_id`` and is kept
#: in the ``S<NN>_...`` form so the parser's ``subject_id_regex`` test resolves
#: a subject; ``person_id`` mirrors the real numeric subject field.
#: Each rep = (first_frame, last_frame, correctness, exercise_subtype,
#: cam17_orientation, lights_on). exercise_subtype is intentionally blank for
#: squats (as in the real data) to exercise the empty-field fallback.
RECORDINGS = [
    {
        "recording": "S01_arm_abduction_C1",
        "person_id": 1,
        "exercise_id": 1,            # arm_abduction
        "exercise": "arm_abduction",
        "n_frames": 40,
        "reps": [
            (0, 18, 1, "right arm", "front", 0),
            (19, 38, 0, "right arm", "front", 0),
        ],
    },
    {
        "recording": "S02_squats_C1",
        "person_id": 2,
        "exercise_id": 6,            # squats (no subtype)
        "exercise": "squats",
        "n_frames": 36,
        "reps": [
            (0, 16, 1, "", "half-profile", 1),
            (17, 34, 1, "", "half-profile", 1),
        ],
    },
    {
        "recording": "S03_leg_lunges_C1",
        "person_id": 3,
        "exercise_id": 5,            # leg_lunges
        "exercise": "leg_lunges",
        "n_frames": 30,
        "reps": [
            (0, 28, 0, "left leg", "profile", 0),
        ],
    },
]


def _write_names() -> None:
    (HERE / "marker_names.txt").write_text(
        "\n".join(MARKER_NAMES) + "\n", encoding="utf-8"
    )
    (HERE / "joints_names.txt").write_text(
        "\n".join(JOINT_NAMES) + "\n", encoding="utf-8"
    )


#: Camera views to materialize (real data also has 120fps; mock keeps 30fps).
CAMERAS = ("c17", "c18")
FPS = 30


def generate() -> dict[str, list[Path]]:
    """(Re)generate all fixture files. Returns a map of created paths.

    Mirrors the REAL REHAB24-6 on-disk layout (confirmed from Zenodo record
    13305826): pose arrays are ``.npy`` of shape ``(T, 26, 2|3)`` under
    ``2d_joints/Ex<k>/`` and ``3d_joints/Ex<k>/``, named
    ``<video_id>-c<cam>-<fps>fps.npy`` (two cameras; 30 fps kept here).
    """
    rng = np.random.default_rng(SEED)

    vid_dir = HERE / "videos"
    p2d_root = HERE / "2d_joints"
    p3d_root = HERE / "3d_joints"
    vid_dir.mkdir(parents=True, exist_ok=True)

    created: dict[str, list[Path]] = {
        "videos": [], "poses_2d": [], "poses_3d": [], "meta": []
    }

    _write_names()
    created["meta"].extend(
        [HERE / "marker_names.txt", HERE / "joints_names.txt"]
    )

    seg_rows: list[dict] = []
    for spec in RECORDINGS:
        rec = spec["recording"]
        t = spec["n_frames"]
        ex_dir = f"Ex{spec['exercise_id']}"

        # 0-byte video stub (NEVER a real video).
        vid = vid_dir / f"{rec}.mp4"
        vid.write_bytes(b"")
        created["videos"].append(vid)

        # 2D pixel coords: one (T, 26, 2) array PER camera view (~1080p frame).
        for cam in CAMERAS:
            xy = np.empty((t, NUM_JOINTS, 2), dtype=np.float64)
            xy[..., 0] = rng.uniform(0, 1920, size=(t, NUM_JOINTS))
            xy[..., 1] = rng.uniform(0, 1080, size=(t, NUM_JOINTS))
            p2d_dir = p2d_root / ex_dir
            p2d_dir.mkdir(parents=True, exist_ok=True)
            p2d = p2d_dir / f"{rec}-{cam}-{FPS}fps.npy"
            np.save(p2d, xy)
            created["poses_2d"].append(p2d)

        # 3D OptiTrack coords are camera-independent: ONE (T, 26, 4) array
        # [x, y, z, validity] in metres (no camera token in the filename).
        xyzv = np.empty((t, NUM_JOINTS, 4), dtype=np.float64)
        xyzv[..., :3] = rng.uniform(-1.0, 2.2, size=(t, NUM_JOINTS, 3))
        xyzv[..., 3] = 1.0  # validity flag
        p3d_dir = p3d_root / ex_dir
        p3d_dir.mkdir(parents=True, exist_ok=True)
        p3d = p3d_dir / f"{rec}-{FPS}fps.npy"
        np.save(p3d, xyzv)
        created["poses_3d"].append(p3d)

        for rep_num, (start, end, correct, subtype, orient, lights) in enumerate(
            spec["reps"], start=1
        ):
            seg_rows.append(
                {
                    "video_id": rec,
                    "repetition_number": rep_num,
                    "exercise_id": spec["exercise_id"],
                    "person_id": spec["person_id"],
                    "first_frame": start,
                    "last_frame": end,
                    "cam17_orientation": orient,
                    "mocap_erroneous": 0,
                    "exercise_subtype": subtype,
                    "lights_on": lights,
                    "extra_person_in_cam17": 0,
                    "extra_person_in_cam18": 0,
                    "correctness": correct,
                }
            )

    # Real REHAB24-6 Segmentation.csv is ';'-delimited with these columns.
    seg_path = HERE / "Segmentation.csv"
    fieldnames = [
        "video_id", "repetition_number", "exercise_id", "person_id",
        "first_frame", "last_frame", "cam17_orientation", "mocap_erroneous",
        "exercise_subtype", "lights_on", "extra_person_in_cam17",
        "extra_person_in_cam18", "correctness",
    ]
    with seg_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        writer.writerows(seg_rows)
    created["meta"].append(seg_path)

    return created


if __name__ == "__main__":
    out = generate()
    for group, paths in out.items():
        for p in paths:
            print(group, p.relative_to(HERE))
