"""Tests for aggregate-only labeled-set overlap analysis."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_labeled_set_overlap as overlap  # noqa: E402


def _state(
    root: Path,
    session: str,
    labeled: list[str],
    *,
    history: list[dict] | None = None,
) -> None:
    path = root / session / "state.json"
    path.parent.mkdir(parents=True)
    payload: dict = {"labeled": labeled}
    if history is not None:
        payload["history"] = history
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_overlap_outputs_are_aggregate_only(tmp_path: Path) -> None:
    root = tmp_path / "full_local_run"
    _state(root, "A_s0", ["frame-a", "frame-b", "frame-c"])
    _state(root, "B_s0_sig0", ["frame-b", "frame-c", "frame-d"])
    _state(root, "ref_s0", ["frame-c", "frame-d", "frame-e"])

    records = overlap.calculate_records(root, [0], 0.0)
    assert records[0] == {
        "seed": 0,
        "pair": "A-B",
        "left_size": 3,
        "right_size": 3,
        "intersection": 2,
        "union": 4,
        "jaccard": 0.5,
    }
    assert overlap.summarize(records)["A-B"]["mean_jaccard"] == 0.5

    out_csv = tmp_path / "aggregate.csv"
    out_json = tmp_path / "aggregate.json"
    overlap.write_csv(out_csv, records)
    overlap.write_json(out_json, records, 0.0)
    exported = out_csv.read_text(encoding="utf-8") + out_json.read_text(encoding="utf-8")
    assert "frame-a" not in exported
    assert "frame-e" not in exported


def test_duplicate_identifiers_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(json.dumps({"labeled": ["frame-a", "frame-a"]}), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate identifiers"):
        overlap._load_labeled(path)


def test_proposal_accounting_excludes_identifiers_and_initial_round(
    tmp_path: Path,
) -> None:
    root = tmp_path / "full_local_run"
    history = [
        {
            "round": 0,
            "n_labeled": 10,
            "n_proposed": 10,
            "n_accepted": 10,
            "n_rejected": 0,
            "n_corrected": 10,
        },
        {
            "round": 1,
            "n_labeled": 20,
            "n_proposed": 25,
            "n_accepted": 10,
            "n_rejected": 15,
            "n_corrected": 10,
            "candidate_ids": ["private-frame-a"],
        },
    ]
    _state(root, "B_s0_sig0", ["private-frame-a"], history=history)

    records = overlap.calculate_proposal_records(root, [0], 0.0)
    assert records == [
        {
            "seed": 0,
            "round": 1,
            "budget": 0.1,
            "n_labeled": 20,
            "n_proposed": 25,
            "n_accepted": 10,
            "n_rejected": 15,
            "n_corrected": 10,
            "reject_fraction": 0.6,
        }
    ]
    assert overlap.summarize_proposals(records)["1"]["mean_n_proposed"] == 25

    out_csv = tmp_path / "proposal_accounting.csv"
    out_json = tmp_path / "proposal_accounting.json"
    overlap.write_proposal_csv(out_csv, records)
    overlap.write_proposal_json(out_json, records, 0.0)
    exported = out_csv.read_text(encoding="utf-8") + out_json.read_text(
        encoding="utf-8"
    )
    assert "private-frame-a" not in exported
