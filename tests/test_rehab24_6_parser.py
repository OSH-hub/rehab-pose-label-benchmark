from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rehabal_pose.data.assessment_labels import ASSESSMENT_COLUMNS
from rehabal_pose.data.rehab24_6 import (
    REHAB24_6_EXERCISES,
    REHAB24_6_NUM_JOINTS,
    Rehab24_6Parser,
)
from rehabal_pose.data.schemas import MANIFEST_COLUMNS

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "rehab24_6_mock"
SEG_CSV = FIXTURE_ROOT / "Segmentation.csv"

N_REPS = 5


@pytest.fixture(scope="session", autouse=True)
def _ensure_fixtures() -> None:
    if not SEG_CSV.is_file():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "_mk_rehab_fixtures", FIXTURE_ROOT / "make_fixtures.py"
        )
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate()


def _make_parser(tmp_path: Path) -> Rehab24_6Parser:
    return Rehab24_6Parser(
        {
            "dataset": "rehab24_6",
            "roots": [str(FIXTURE_ROOT)],
            "output_dir": str(tmp_path / "keypoints"),
            "subject_id_regex": r"^(S\d+)_",
            "camera_id": "cam0",
            "view": "frontal",
            "split": "train",
            "disease": "rehab",
        }
    )


def test_scan_finds_files(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    found = parser.scan()
    assert len(found["videos"]) == 3
    assert len(found["pose2d"]) == 6
    assert len(found["pose3d"]) == 3
    assert len(found["segmentation"]) == 1
    assert len(found["marker_names"]) == 1
    assert len(found["joints_names"]) == 1


def test_marker_and_joint_name_counts(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    found = parser.scan()
    markers = parser.load_joint_names(found["marker_names"][0])
    joints = parser.load_joint_names(found["joints_names"][0])
    assert len(markers) == 41
    assert len(joints) == REHAB24_6_NUM_JOINTS == 26


def test_parse_builds_manifest(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    df = parser.parse().to_dataframe()

    assert len(df) == N_REPS
    assert list(df.columns) == MANIFEST_COLUMNS

    assert (df["dataset"] == "rehab24_6").all()
    assert df["has_keypoints2d"].all()
    assert df["has_keypoints3d"].all()
    assert df["sample_id"].is_unique

    assert df["keypoints2d_path"].notna().all()
    assert df["keypoints3d_path"].notna().all()
    assert df["video_path"].str.endswith(".mp4").all()


def test_parse_exercise_and_standardtype(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    df = parser.parse().to_dataframe()

    assert set(df["exercise"]).issubset(set(REHAB24_6_EXERCISES))
    assert {"arm_abduction", "squats", "leg_lunges"} <= set(df["exercise"])

    assert set(df["standardtype"]) == {"correct", "incorrect"}


def test_parse_subject_id_from_regex(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    df = parser.parse().to_dataframe()
    assert set(df["subject_id"]) == {"S01", "S02", "S03"}


def test_parse_keypoint_arrays_shape(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    df = parser.parse().to_dataframe()

    for path2d, path3d in zip(df["keypoints2d_path"], df["keypoints3d_path"]):
        kp2d = np.load(path2d)
        kp3d = np.load(path3d)
        assert kp2d.ndim == 3 and kp2d.shape[1:] == (REHAB24_6_NUM_JOINTS, 3)
        assert kp3d.ndim == 3 and kp3d.shape[1] == REHAB24_6_NUM_JOINTS
        assert kp3d.shape[2] in (3, 4)
        assert kp2d.dtype == np.float64 and kp3d.dtype == np.float64
        assert kp2d.shape[0] == kp3d.shape[0]
        assert np.allclose(kp2d[..., 2], 1.0)


def test_clip_ids_present(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    df = parser.parse().to_dataframe()
    assert df["clip_id"].notna().all()
    assert df["clip_id"].is_unique
    assert df["video_id"].nunique() == 3
    assert df.groupby("video_id").size().sum() == N_REPS


def test_parse_segmentation_labels(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse_segmentation(SEG_CSV)
    adf = builder.to_dataframe()

    assert list(adf.columns) == ASSESSMENT_COLUMNS
    correctness = adf[adf["label_type"] == "correctness"]
    repetition = adf[adf["label_type"] == "repetition"]
    assert len(correctness) == N_REPS
    assert len(repetition) == N_REPS

    assert correctness["value_bool"].notna().all()
    assert set(correctness["value_bool"]) == {True, False}

    assert (repetition["end_frame"] >= repetition["start_frame"]).all()
    assert repetition["label_name"].notna().all()

    assert adf["sample_id"].notna().all()
    assert (adf["dataset"] == "rehab24_6").all()


def test_segmentation_validate_passes(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse_segmentation(SEG_CSV)
    report = builder.validate()
    assert report.ok, [i.to_dict() for i in report.errors]


def test_validate_ok(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse()
    report = parser.validate(builder, segmentation_csv=SEG_CSV)
    assert report.ok, [i.to_dict() for i in report.errors]
    assert len(report.errors) == 0


def test_validate_detects_duplicate(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse()

    first = builder.to_dataframe().iloc[0].to_dict()
    builder.add_row(**first)

    report = parser.validate(builder)
    assert not report.ok
    codes = {i.code for i in report.errors}
    assert "DUPLICATE_SAMPLE_ID" in codes


def test_validate_detects_bad_joint_count(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse()

    bad = tmp_path / "bad_kp.npy"
    np.save(bad, np.zeros((5, 10, 3), dtype=np.float64))
    builder.add_row(
        sample_id="badkp_1",
        dataset="rehab24_6",
        has_keypoints2d=True,
        keypoints2d_path=str(bad),
    )
    report = parser.validate(builder)
    codes = {i.code for i in report.errors}
    assert "BAD_KEYPOINT_COUNT" in codes


def test_validate_detects_frame_count_mismatch(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse()

    p2d = tmp_path / "m2d.npy"
    p3d = tmp_path / "m3d.npy"
    np.save(p2d, np.zeros((10, 26, 3), dtype=np.float64))
    np.save(p3d, np.zeros((7, 26, 3), dtype=np.float64))
    builder.add_row(
        sample_id="mismatch_1",
        dataset="rehab24_6",
        has_keypoints2d=True,
        has_keypoints3d=True,
        keypoints2d_path=str(p2d),
        keypoints3d_path=str(p3d),
    )
    report = parser.validate(builder)
    codes = {i.code for i in report.errors}
    assert "FRAME_COUNT_MISMATCH" in codes


def test_validate_detects_bad_segmentation(tmp_path: Path) -> None:
    parser = _make_parser(tmp_path)
    builder = parser.parse()

    bad_csv = tmp_path / "BadSeg.csv"
    bad_csv.write_text(
        "recording,subject,exercise,start_frame,end_frame,correctness,direction,lighting\n"
        "S09_squats_C1,S09,squats,40,10,2,up,normal\n",
        encoding="utf-8",
    )
    report = parser.validate(builder, segmentation_csv=bad_csv)
    codes = {i.code for i in report.errors}
    assert "BAD_SEGMENT_FRAMES" in codes
    assert "BAD_CORRECTNESS" in codes
