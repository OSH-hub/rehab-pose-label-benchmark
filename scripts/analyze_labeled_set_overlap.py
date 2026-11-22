"""Aggregate state-derived transparency metrics without exporting identifiers.

This analysis is intended for a full local experiment directory that contains
``state.json`` files. It reads the cumulative ``labeled`` arrays for conditions
A, B (the selected correction-noise slice), and ref, then writes set sizes,
intersection/union counts, and Jaccard indices. It also aggregates condition B's
post-initialization proposal, commit, rejection, and correction counts. Raw state
files and sample identifiers must not be copied into a review or release package.

Example::

    python scripts/analyze_labeled_set_overlap.py \
        --root outputs/experiments/clinician_i4_uniform30 \
        --out-csv verification/labeled_set_overlap.csv \
        --out-json verification/labeled_set_overlap.json \
        --out-proposals-csv verification/proposal_accounting.csv \
        --out-proposals-json verification/proposal_accounting.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean


PAIR_ORDER = (("A", "B"), ("A", "ref"), ("B", "ref"))
REPORTED_BUDGET_BY_ROUND = {0: 0.05, 1: 0.10, 2: 0.20, 3: 0.40}


def _session_name(condition: str, seed: int, sigma: float) -> str:
    if condition == "B":
        return f"B_s{seed}_sig{sigma:g}"
    return f"{condition}_s{seed}"


def _load_labeled(path: Path) -> set[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    labeled = data.get("labeled")
    if not isinstance(labeled, list) or not all(isinstance(x, str) for x in labeled):
        raise ValueError(f"{path}: expected a string array at 'labeled'")
    unique = set(labeled)
    if len(unique) != len(labeled):
        raise ValueError(f"{path}: duplicate identifiers in cumulative labeled set")
    return unique


def calculate_records(root: Path, seeds: list[int], sigma: float) -> list[dict]:
    """Return aggregate-only pairwise overlap records for the requested seeds."""
    records: list[dict] = []
    for seed in seeds:
        sets = {
            condition: _load_labeled(
                root / _session_name(condition, seed, sigma) / "state.json"
            )
            for condition in ("A", "B", "ref")
        }
        for left, right in PAIR_ORDER:
            intersection = len(sets[left] & sets[right])
            union = len(sets[left] | sets[right])
            records.append(
                {
                    "seed": seed,
                    "pair": f"{left}-{right}",
                    "left_size": len(sets[left]),
                    "right_size": len(sets[right]),
                    "intersection": intersection,
                    "union": union,
                    "jaccard": intersection / union if union else 1.0,
                }
            )
    return records


def summarize(records: list[dict]) -> dict[str, dict]:
    """Summarize Jaccard values and intersections by condition pair."""
    result: dict[str, dict] = {}
    for left, right in PAIR_ORDER:
        pair = f"{left}-{right}"
        subset = [row for row in records if row["pair"] == pair]
        if not subset:
            continue
        result[pair] = {
            "seed_count": len(subset),
            "mean_jaccard": mean(row["jaccard"] for row in subset),
            "min_jaccard": min(row["jaccard"] for row in subset),
            "max_jaccard": max(row["jaccard"] for row in subset),
            "min_intersection": min(row["intersection"] for row in subset),
            "max_intersection": max(row["intersection"] for row in subset),
        }
    return result


def calculate_proposal_records(
    root: Path, seeds: list[int], sigma: float
) -> list[dict]:
    """Return condition-B post-initialization accounting without sample IDs."""
    records: list[dict] = []
    for seed in seeds:
        path = root / _session_name("B", seed, sigma) / "state.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        history = data.get("history")
        if not isinstance(history, list):
            raise ValueError(f"{path}: expected an array at 'history'")
        for item in history:
            if not isinstance(item, dict):
                raise ValueError(f"{path}: history entries must be objects")
            round_index = int(item.get("round", -1))
            if round_index == 0:
                continue
            if round_index not in REPORTED_BUDGET_BY_ROUND:
                raise ValueError(f"{path}: unexpected round index {round_index}")
            required = (
                "n_labeled",
                "n_proposed",
                "n_accepted",
                "n_rejected",
                "n_corrected",
            )
            missing = [field for field in required if field not in item]
            if missing:
                raise ValueError(f"{path}: round {round_index} missing {missing}")
            proposed = int(item["n_proposed"])
            if proposed <= 0:
                raise ValueError(
                    f"{path}: round {round_index} has non-positive n_proposed"
                )
            rejected = int(item["n_rejected"])
            records.append(
                {
                    "seed": seed,
                    "round": round_index,
                    "budget": REPORTED_BUDGET_BY_ROUND[round_index],
                    "n_labeled": int(item["n_labeled"]),
                    "n_proposed": proposed,
                    "n_accepted": int(item["n_accepted"]),
                    "n_rejected": rejected,
                    "n_corrected": int(item["n_corrected"]),
                    "reject_fraction": rejected / proposed,
                }
            )
    return sorted(records, key=lambda row: (row["round"], row["seed"]))


def summarize_proposals(records: list[dict]) -> dict[str, dict]:
    """Summarize proposal-accounting fields by post-initialization round."""
    result: dict[str, dict] = {}
    for round_index in sorted({row["round"] for row in records}):
        subset = [row for row in records if row["round"] == round_index]
        result[str(round_index)] = {
            "budget": subset[0]["budget"],
            "seed_count": len(subset),
            "mean_n_labeled": mean(row["n_labeled"] for row in subset),
            "mean_n_proposed": mean(row["n_proposed"] for row in subset),
            "mean_n_accepted": mean(row["n_accepted"] for row in subset),
            "mean_n_rejected": mean(row["n_rejected"] for row in subset),
            "mean_n_corrected": mean(row["n_corrected"] for row in subset),
            "mean_reject_fraction": mean(row["reject_fraction"] for row in subset),
        }
    return result


def write_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "pair",
        "left_size",
        "right_size",
        "intersection",
        "union",
        "jaccard",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, records: list[dict], sigma: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "review_package_note": (
            "Aggregate-only derivative of full local state.json files; sample "
            "identifiers and raw state files are not included."
        ),
        "source": "outputs/experiments/clinician_i4_uniform30/*/state.json",
        "condition_b_sigma_px": sigma,
        "records": records,
        "summary": summarize(records),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_proposal_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "seed",
        "round",
        "budget",
        "n_labeled",
        "n_proposed",
        "n_accepted",
        "n_rejected",
        "n_corrected",
        "reject_fraction",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def write_proposal_json(path: Path, records: list[dict], sigma: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "review_package_note": (
            "Aggregate-only condition-B accounting derived from full local "
            "state.json history; sample identifiers and raw state are not included."
        ),
        "source_pattern": (
            "outputs/experiments/clinician_i4_uniform30/"
            f"B_s*_sig{sigma:g}/state.json"
        ),
        "condition": "B",
        "condition_b_sigma_px": sigma,
        "records": records,
        "summary_by_round": summarize_proposals(records),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/experiments/clinician_i4_uniform30"),
        help="full local experiment directory containing per-condition state.json files",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--sigma", type=float, default=0.0)
    parser.add_argument(
        "--out-csv", type=Path, default=Path("verification/labeled_set_overlap.csv")
    )
    parser.add_argument(
        "--out-json", type=Path, default=Path("verification/labeled_set_overlap.json")
    )
    parser.add_argument(
        "--out-proposals-csv",
        type=Path,
        default=Path("verification/proposal_accounting.csv"),
    )
    parser.add_argument(
        "--out-proposals-json",
        type=Path,
        default=Path("verification/proposal_accounting.json"),
    )
    args = parser.parse_args(argv)

    records = calculate_records(args.root, args.seeds, args.sigma)
    write_csv(args.out_csv, records)
    write_json(args.out_json, records, args.sigma)
    proposal_records = calculate_proposal_records(args.root, args.seeds, args.sigma)
    write_proposal_csv(args.out_proposals_csv, proposal_records)
    write_proposal_json(args.out_proposals_json, proposal_records, args.sigma)
    print(f"Wrote aggregate-only overlap records: {args.out_csv}")
    print(f"Wrote aggregate-only overlap summary: {args.out_json}")
    print(f"Wrote aggregate-only proposal records: {args.out_proposals_csv}")
    print(f"Wrote aggregate-only proposal summary: {args.out_proposals_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
