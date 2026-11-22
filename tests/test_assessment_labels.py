"""Unit tests for the long-form assessment-label table."""

from __future__ import annotations

import pandas as pd
import pytest

from rehabal_pose.data.assessment_labels import (
    ASSESSMENT_COLUMNS,
    AssessmentLabel,
    AssessmentLabelBuilder,
    attach_labels,
    load_assessment_table,
)


def _populated_builder() -> AssessmentLabelBuilder:
    b = AssessmentLabelBuilder()
    b.add_clinical_score("kimore_s1_ex1", "cTS", 42.0, rater="clin", dataset="kimore")
    b.add_clinical_score("kimore_s1_ex1", "cPO", 12.0, dataset="kimore")
    b.add_correctness("uiprmd_s1_e1_r1", True, dataset="uiprmd")
    b.add_gesture("irds_s1_g1", "elbow_flexion", dataset="irds")
    b.add_repetition("rehab_s1_squats", 30, 90, is_correct=True, dataset="rehab246")
    return b


def test_columns_order():
    b = AssessmentLabelBuilder()
    b.add_clinical_score("s1", "cTS", 1.0)
    df = b.to_dataframe()
    assert list(df.columns) == ASSESSMENT_COLUMNS


def test_empty_builder_dataframe_has_columns():
    df = AssessmentLabelBuilder().to_dataframe()
    assert list(df.columns) == ASSESSMENT_COLUMNS
    assert len(df) == 0


def test_add_label_unknown_column_raises():
    b = AssessmentLabelBuilder()
    with pytest.raises(ValueError):
        b.add_label(sample_id="s1", bogus="x")


def test_add_assessment_label_instance():
    b = AssessmentLabelBuilder()
    lab = AssessmentLabel(
        sample_id="s1", label_type="clinical_score", label_name="cTS", value_num=3.0
    )
    b.add(lab)
    assert len(b) == 1
    assert b.to_dataframe().iloc[0]["value_num"] == 3.0


def test_convenience_helpers_populate_right_fields():
    b = _populated_builder()
    df = b.to_dataframe()

    cts = df[(df.label_type == "clinical_score") & (df.label_name == "cTS")].iloc[0]
    assert cts["value_num"] == 42.0
    assert pd.isna(cts["value_str"]) or cts["value_str"] is None

    corr = df[df.label_type == "correctness"].iloc[0]
    assert bool(corr["value_bool"]) is True

    gest = df[df.label_type == "gesture"].iloc[0]
    assert gest["value_str"] == "elbow_flexion"

    rep = df[df.label_type == "repetition"].iloc[0]
    assert rep["start_frame"] == 30
    assert rep["end_frame"] == 90
    assert bool(rep["value_bool"]) is True


def test_len():
    assert len(_populated_builder()) == 5


@pytest.mark.parametrize("ext", [".parquet", ".csv"])
def test_roundtrip(tmp_path, ext):
    b = _populated_builder()
    path = tmp_path / f"assess{ext}"
    b.save(path)
    df = load_assessment_table(path)
    assert list(df.columns) == ASSESSMENT_COLUMNS
    assert len(df) == 5
    # builder classmethod load matches module loader
    df2 = AssessmentLabelBuilder.load(path)
    assert list(df2.columns) == ASSESSMENT_COLUMNS
    assert len(df2) == 5


def test_save_bad_extension(tmp_path):
    with pytest.raises(ValueError):
        _populated_builder().save(tmp_path / "x.txt")


def test_validate_ok():
    report = _populated_builder().validate()
    assert report.ok
    assert len(report.errors) == 0


def test_validate_catches_missing_value():
    b = AssessmentLabelBuilder()
    # clinical_score but no value_num
    b.add_label(sample_id="s1", label_type="clinical_score", label_name="cTS")
    report = b.validate()
    assert not report.ok
    assert any(i.code == "MISSING_VALUE" for i in report.errors)


def test_validate_catches_missing_sample_id():
    b = AssessmentLabelBuilder()
    b.add_label(label_type="correctness", value_bool=True)
    report = b.validate()
    assert any(i.code == "MISSING_SAMPLE_ID" for i in report.errors)


def test_validate_catches_missing_label_type():
    b = AssessmentLabelBuilder()
    b.add_label(sample_id="s1", value_num=1.0)
    report = b.validate()
    assert any(i.code == "MISSING_LABEL_TYPE" for i in report.errors)


def test_validate_bad_repetition_span():
    b = AssessmentLabelBuilder()
    b.add_label(
        sample_id="s1", label_type="repetition", start_frame=90, end_frame=30
    )
    report = b.validate()
    assert any(i.code == "BAD_REPETITION_SPAN" for i in report.errors)


def test_validate_repetition_missing_frames():
    b = AssessmentLabelBuilder()
    b.add_label(sample_id="s1", label_type="repetition")
    report = b.validate()
    assert any(i.code == "BAD_REPETITION_SPAN" for i in report.errors)


def test_validate_unknown_label_type_warns():
    b = AssessmentLabelBuilder()
    b.add_label(sample_id="s1", label_type="weird_type", value_str="x")
    report = b.validate()
    assert report.ok  # warning only
    assert any(i.code == "UNKNOWN_LABEL_TYPE" for i in report.warnings)


def test_attach_labels_left_join():
    manifest = pd.DataFrame(
        {"sample_id": ["kimore_s1_ex1", "other"], "dataset": ["kimore", "x"]}
    )
    assess = _populated_builder().to_dataframe()
    joined = attach_labels(manifest, assess, label_type="clinical_score")
    # kimore_s1_ex1 has two clinical scores -> fan out; "other" stays (1 null row)
    kim = joined[joined.sample_id == "kimore_s1_ex1"]
    assert len(kim) == 2
    assert set(kim["label_name"]) == {"cTS", "cPO"}
    other = joined[joined.sample_id == "other"]
    assert len(other) == 1
    assert pd.isna(other.iloc[0]["label_name"])


def test_attach_labels_no_filter():
    manifest = pd.DataFrame({"sample_id": ["irds_s1_g1"], "dataset": ["irds"]})
    assess = _populated_builder().to_dataframe()
    joined = attach_labels(manifest, assess)
    assert (joined["label_type"] == "gesture").any()
