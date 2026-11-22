"""Descriptive audit of the two new policies; never infers missing original-A diagnostics."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys

METHODS = ("A_jointmask", "A_flipcons")
BUDGETS = (.05, .1, .2, .4)
TARGETS = (418, 837, 1673, 3347)


def lp(path):
    s = str(Path(path).resolve())
    return s if os.name != "nt" or s.startswith("\\\\?\\") else "\\\\?\\" + s


def integer(value, name):
    x = float(value)
    if not math.isfinite(x) or x != int(x) or x < 0:
        raise ValueError(f"Invalid nonnegative integer {name}: {value}")
    return int(x)


def sha(path):
    h = hashlib.sha256()
    with open(lp(path), "rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def extract_row(r, initialization):
    method, seed = r["condition"], integer(r["seed"], "seed")
    index = integer(r["round"], "round")
    if method not in METHODS or seed not in range(6) or index not in range(4):
        raise ValueError("Unexpected policy/seed/round")
    budget = float(r["budget"])
    if r.get("status") != "complete" or budget != BUDGETS[index]:
        raise ValueError("Only completed fixed-protocol records are allowed")
    fields = ("n_labeled", "accepted_frames", "proposed_frames", "rejected_frames", "gt_initial_frames",
              "gt_additional_frames", "gt_query_calls", "visible_pseudo_joints", "automatic_masked_joints", "consensus_joints")
    n = {f: integer(r[f], f) for f in fields}
    increment = TARGETS[index] - (TARGETS[index-1] if index else 0)
    if n["n_labeled"] != TARGETS[index] or n["accepted_frames"] != increment:
        raise ValueError("Wrong accepted-frame budget or increment")
    if index and n["proposed_frames"] < n["accepted_frames"] + n["rejected_frames"]:
        raise ValueError("Impossible proposed/accepted/rejected frame counts")
    if n["gt_additional_frames"] != 0 or n["gt_query_calls"] != (418 if index == 0 else 0):
        raise ValueError("Unexpected additional GT queries")
    if n["gt_initial_frames"] != (418 if index == 0 else 0):
        raise ValueError("Unexpected initial GT-label count")
    reused = initialization.get("reuse") is True and initialization.get("kind") in {
        "legacy_reference_verified_initialization", "same_process_shared_initial"}
    if not reused:
        raise ValueError("Original shared initialization was not reused")
    tau, visible_fraction = None, None
    if index:
        tau = float(r["confidence_threshold"])
        if not math.isfinite(tau) or not 0 < tau <= 1:
            raise ValueError("Missing or invalid observed confidence threshold")
        total = 26 * n["accepted_frames"]
        if n["visible_pseudo_joints"] + n["automatic_masked_joints"] != total:
            raise ValueError("Joint accounting differs from 26 times accepted pseudo-labeled frames")
        if not 6*n["accepted_frames"] <= n["visible_pseudo_joints"] <= total:
            raise ValueError("Accepted-joint count violates six-joint usability policy")
        if n["consensus_joints"] != (n["visible_pseudo_joints"] if method == "A_flipcons" else 0):
            raise ValueError("Unexpected consensus-joint accounting")
        visible_fraction = n["visible_pseudo_joints"] / total
    elif any(n[f] != 0 for f in ("visible_pseudo_joints", "automatic_masked_joints", "consensus_joints")):
        raise ValueError("Initial GT round must not be described as pseudo-label masking")
    return {"condition": method, "seed": seed, "budget": budget, "round": index, **n,
            "confidence_threshold": tau, "threshold_below_0_5": tau < .5 if tau is not None else None,
            "visible_joint_fraction_among_accepted_pseudo_labels": visible_fraction,
            "rejected_frame_fraction_among_proposed": n["rejected_frames"] / n["proposed_frames"] if index and n["proposed_frames"] else None,
            "proposed_minus_accepted": n["proposed_frames"] - n["accepted_frames"] if index else None,
            "other_unaccepted_proposals": n["proposed_frames"] - n["accepted_frames"] - n["rejected_frames"] if index else None,
            "topup_call_count": None, "topup_count_status": "not_recorded_do_not_infer",
            "initialization_reused": reused, "initialization_kind": initialization["kind"],
            "initial_checkpoint_sha256": initialization.get("checkpoint_sha256"),
            "post_initialization_gt_queries_zero": n["gt_query_calls"] == 0 if index else None,
            "checkpoint_sha256": r["checkpoint_sha256"]}


def collect(base):
    rows, sources = [], []
    for method in METHODS:
        for seed in range(6):
            source = base / f"{method}_s{seed}" / "audit_rounds.jsonl"
            with open(lp(source), encoding="utf-8-sig") as f:
                records = [json.loads(line) for line in f if line.strip()]
            records = [r for r in records if r.get("status") == "complete"]
            if len(records) != 4 or {integer(r["round"], "round") for r in records} != {0,1,2,3}:
                raise ValueError(f"Four unique complete rounds required: {source}")
            initial = next(r for r in records if int(r["round"]) == 0).get("initialization") or {}
            for record in sorted(records, key=lambda r: int(r["round"])):
                if record["condition"] != method or int(record["seed"]) != seed:
                    raise ValueError("Mixed session identity in audit")
                rows.append(extract_row(record, initial))
            sources.append({"path": str(source), "sha256": sha(source)})
    return rows, sources


def summarize(rows):
    out = []
    for method in METHODS:
        for index, budget in enumerate(BUDGETS):
            selected = [r for r in rows if r["condition"] == method and r["round"] == index]
            if len(selected) != 6:
                raise ValueError("Descriptive summary needs all six seeds")
            result = {"condition": method, "budget": budget, "round": index, "training_seeds": 6,
                      "all_initializations_reused": all(r["initialization_reused"] for r in selected),
                      "post_initialization_gt_queries_total": sum(r["gt_query_calls"] for r in selected) if index else None,
                      "topup_call_count": None, "topup_count_status": "not_recorded_do_not_infer"}
            for field in ("accepted_frames", "proposed_frames", "rejected_frames", "visible_pseudo_joints", "automatic_masked_joints", "consensus_joints"):
                result[field + "_sum_across_seeds"] = sum(r[field] for r in selected)
            for field in ("confidence_threshold", "visible_joint_fraction_among_accepted_pseudo_labels", "rejected_frame_fraction_among_proposed"):
                values = [r[field] for r in selected if r[field] is not None]
                result[field + "_mean_across_seeds"] = statistics.mean(values) if values else None
                result[field + "_min"] = min(values) if values else None
                result[field + "_max"] = max(values) if values else None
            out.append(result)
    return out


def write_csv(path, rows):
    with open(lp(path), "x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        r = {"condition":"A_jointmask", "seed":0, "round":1, "budget":.1, "status":"complete",
             "n_labeled":837, "accepted_frames":419, "proposed_frames":838, "rejected_frames":20,
             "gt_initial_frames":0, "gt_additional_frames":0, "gt_query_calls":0,
             "visible_pseudo_joints":419*20, "automatic_masked_joints":419*6, "consensus_joints":0,
             "confidence_threshold":.4, "checkpoint_sha256":"a"*64}
        initial={"reuse":True,"kind":"legacy_reference_verified_initialization","checkpoint_sha256":"b"*64}
        row=extract_row(r,initial)
        assert row["threshold_below_0_5"] and row["topup_call_count"] is None
        assert math.isclose(row["visible_joint_fraction_among_accepted_pseudo_labels"],20/26)
        initial_record={**r,"round":0,"budget":.05,"n_labeled":418,"accepted_frames":418,"proposed_frames":0,"rejected_frames":0,"gt_initial_frames":418,"gt_query_calls":418,"visible_pseudo_joints":0,"automatic_masked_joints":0,"confidence_threshold":None}
        initial_row=extract_row(initial_record,initial)
        assert initial_row["proposed_minus_accepted"] is None and initial_row["rejected_frame_fraction_among_proposed"] is None
        for edit in ({"gt_query_calls":1},{"automatic_masked_joints":0},{"confidence_threshold":float("nan")},{"consensus_joints":1}):
            try: extract_row({**r,**edit},initial)
            except ValueError: continue
            raise AssertionError("Invalid policy diagnostics accepted")
        print("POLICY_DIAGNOSTICS_SYNTHETIC_PASS: 6 cases; no measurements read")
        return 0
    if not args.training_dir or not args.output:
        parser.error("Both --training-dir and --output required; this utility reads completed audit records")
    if Path(lp(args.output)).exists():
        raise ValueError("Diagnostic output already exists; preserve it")
    rows, sources = collect(args.training_dir)
    summary = summarize(rows)
    Path(lp(args.output)).mkdir(parents=True, exist_ok=False)
    write_csv(args.output/"policy_diagnostics_rounds.csv",rows)
    write_csv(args.output/"policy_diagnostics_summary.csv",summary)
    metadata={"status":"complete_descriptive_policy_audit","records":len(rows),"new_sessions":12,
              "source_files":sources,"code_sha256":sha(__file__),"original_A_diagnostics_available":False,
              "topup_call_count_available":False,"no_original_A_values_imputed":True,
              "interpretation":"Observed policy operations; not accuracy, a component-effect ablation, or human annotation costs"}
    with open(lp(args.output/"policy_diagnostics_metadata.json"),"x",encoding="utf-8") as f:
        json.dump(metadata,f,ensure_ascii=False,indent=2,allow_nan=False)
    notes=("새 두 정책의 라운드별 실제 감사기록을 기술한 표입니다. 정확도/AUC 검정은 바꾸지 않습니다.\n"
           "초기 GT round0는 직접 주입되어 proposed=0일 수 있으므로 threshold/visible fraction/거절비율/미채택 proposal 수를 비워 두었습니다. r1–r3 visible fraction은 채택한 의사라벨의 visible joints/(26×accepted frames)입니다.\n"
           "원 A0의 관절/거절 감사값은 제공되지 않아 추정하지 않았습니다. tau<0.5만으로 A0보다 실제 retained joints가 많았다고 결론내릴 수 없습니다.\n"
           "proposed-minus-accepted는 rejected와 초과 proposal을 함께 포함합니다. other-unaccepted는 rejected를 뺀 값입니다. 둘 다 top-up 호출 횟수가 아닙니다.\n"
           "현재 audit에는 정확한 top-up 호출 횟수가 없어 null로 보존합니다. 거절 후 목표 수를 채우는 top-up 규칙은 코드에 있고 실제 정확한 횟수는 이 표에서 추론하지 않습니다.\n"
           "6 seed의 합계는 독립 환자 수가 아니며 seed평균/min/max는 기술통계입니다. 인력시간이나 임상효율을 뜻하지 않습니다.\n")
    with open(lp(args.output/"정책진단_읽는법.txt"),"x",encoding="utf-8") as f:f.write(notes)
    print("새 정책 감사표 완료: 48 rounds, 12 sessions")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
