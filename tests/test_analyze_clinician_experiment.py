from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import analyze_clinician_experiment as analysis  # noqa: E402


def _proposal_row(seed: int, budget: float, proposed: int, rejected: int) -> dict:
    return {
        "seed": seed,
        "budget": budget,
        "n_labeled": 20,
        "n_proposed": proposed,
        "n_accepted": 10,
        "n_rejected": rejected,
        "n_corrected": 10,
        "reject_fraction": rejected / proposed,
    }


def test_sanitized_proposal_accounting_is_used_when_raw_state_is_absent(
    tmp_path: Path,
) -> None:
    records = [
        _proposal_row(seed, budget, proposed, rejected)
        for budget, proposed, rejected in ((0.1, 20, 10), (0.2, 30, 20))
        for seed in (0, 1)
    ]
    payload = {
        "condition_b_sigma_px": 0.0,
        "records": records,
    }
    (tmp_path / "proposal_accounting.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    rows = analysis._state_process_rows(
        tmp_path, [0, 1], [0.05, 0.1, 0.2], 0.0
    )
    assert rows[0.05] == []
    assert [row["n_proposed"] for row in rows[0.1]] == [20, 20]
    assert [row["n_rejected"] for row in rows[0.2]] == [20, 20]


def test_complete_raw_state_remains_primary(tmp_path: Path) -> None:
    budgets = [0.05, 0.1, 0.2]
    for seed in (0, 1):
        path = tmp_path / f"B_s{seed}_sig0" / "state.json"
        path.parent.mkdir(parents=True)
        history = [
            {"n_proposed": 0, "n_rejected": 0},
            {"n_proposed": 21, "n_rejected": 11},
            {"n_proposed": 31, "n_rejected": 21},
        ]
        path.write_text(json.dumps({"history": history}), encoding="utf-8")

    (tmp_path / "proposal_accounting.json").write_text(
        json.dumps(
            {
                "condition_b_sigma_px": 0.0,
                "records": [
                    _proposal_row(seed, budget, 999, 998)
                    for budget in budgets[1:]
                    for seed in (0, 1)
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = analysis._state_process_rows(tmp_path, [0, 1], budgets, 0.0)
    assert [row["n_proposed"] for row in rows[0.1]] == [21, 21]
    assert [row["n_rejected"] for row in rows[0.2]] == [21, 21]
