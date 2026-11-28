"""Publish the fixed 5-arm/6-seed analysis only after complete validation."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone
import uuid

DEFAULT_WORK = Path(r"rehabal_exp")
LOCKED_ANALYSIS_PROTOCOL_SHA256 = "93f45b374a6c8ccae1d8b1eb1588d29b4ee4a98afcd65f54ded8d80f455ee879"
LOCKED_PENN_PROTOCOL_SHA256 = "3fae3a70ad923399e57c7ffbbf1e98c3672eabd29806cfc011060610a2224749"
LOCKED_PENN_PREPARE_SHA256 = "afdbc1f05f41d43040cfbfddb303830e1beb9f3d6bae154c54ac108f32ab4f91"
NEW_METHODS = {"A_jointmask", "A_flipcons"}
NONPRIMARY_ISSUES = {"outside_planned_external_method_or_backbone", "outside_planned_external_matrix"}
TABLES = ["training_completed_raw", "source_budget_curve_summary", "source_budget_auc_per_seed",
          "source_budget_auc_summary", "source_budget_auc_paired_seed_differences",
          "penn_action_per_seed", "penn_action_summary", "penn_action_paired_clip_differences",
          "penn_action_by_action_per_seed"]
FIGURES = ["source_budget_curves", "source_budget_auc_per_seed",
           "penn_action_macro_pck_budget_0.40", "penn_action_paired_differences_budget_0.40"]


def lp(path):
    s = str(Path(path).resolve())
    return s if os.name != "nt" or s.startswith("\\\\?\\") else "\\\\?\\" + s


def read_json(path):
    with open(lp(path), encoding="utf-8-sig") as stream:
        return json.load(stream)


def sha(path):
    h = hashlib.sha256()
    with open(lp(path), "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    with open(lp(path), "x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def require_complete(training, external, p, comparable, issues):
    import itertools
    expected_train = set(itertools.product(p["methods"], p["seeds"], p["budgets"]))
    expected_external = set(itertools.product(p["methods"], p["seeds"], p["penn_budgets"]))
    if len(expected_train) != 120 or len(expected_external) != 30 or p["penn_budgets"] != [.4]:
        raise ValueError("This launcher is fixed to 120 source rows and 30 final-budget models")
    if set(training) != expected_train or set(external) != expected_external:
        raise ValueError(f"Incomplete matrix: training {len(training)}/120, external {len(external)}/30")
    if not comparable:
        raise ValueError("Qualified protocol comparability is not verified")
    fatal = [r for r in issues if r.get("type") not in NONPRIMARY_ISSUES]
    if fatal:
        raise ValueError("Input validation issues: " + json.dumps(fatal, ensure_ascii=False))
    for k, model in external.items():
        if training[k]["checkpoint_sha256"] != model["checkpoint_sha256"]:
            raise ValueError(f"Training/external checkpoint mismatch: {k}")


def validate_new_sessions(base, p, training, analysis):
    sources = []
    if Path(lp(base / "clinician_curve.csv")).exists():
        raise ValueError("Flat collector sessions required; an extra master CSV could hide per-session measurements")
    expected_dirs = {f"{m}_s{s}" for m in NEW_METHODS for s in p["seeds"]}
    found = {q.parent.name for q in base.glob("*/experiment_curve.csv") if not q.parent.name.startswith(".incoming_")}
    if found != expected_dirs:
        raise ValueError("New collector must provide exactly 12 flat completed sessions")
    for method in sorted(NEW_METHODS):
        for seed in p["seeds"]:
            session = base / f"{method}_s{seed}"
            for name in ("experiment_curve.csv", "audit_rounds.jsonl", "completion.json", "collection_manifest.json", "collect_receipt.json"):
                item = session / name
                if not Path(lp(item)).is_file():
                    raise ValueError(f"Missing collector-verified session input: {item}")
                sources.append(item)
            manifest = read_json(session / "collection_manifest.json")
            transfer = read_json(session / "collect_receipt.json")
            validation = manifest.get("validation", {})
            canonical = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
            final_sha = training[(method, seed, .4)]["checkpoint_sha256"]
            if (manifest.get("status") != "verified" or validation.get("status") != "verified"
                or validation.get("condition") != method or validation.get("seed") != seed
                or validation.get("budgets") != p["budgets"] or validation.get("targets") != [418,837,1673,3347]
                or validation.get("completed_rounds") != 4 or validation.get("gt_query_calls_after_initialization") != 0
                or validation.get("gt_additional_frames") != 0 or validation.get("initialization_reuse") is not True
                or validation.get("checkpoint_sha256") != final_sha
                or transfer.get("status") != "complete" or transfer.get("manifest_digest") != canonical
                or transfer.get("session_id") != session.name or transfer.get("completed_rounds") != 4
                or transfer.get("checkpoint_sha256") != final_sha
                or transfer.get("exact_budget_verified") is not True
                or transfer.get("gt_query_calls_after_initialization") != 0
                or transfer.get("initialization_reuse_verified") is not True
                or transfer.get("transfer_and_file_checksums_verified") is not True):
                raise ValueError(f"Collector provenance not verified: {session}")
            files = manifest.get("files", [])
            if not files:
                raise ValueError(f"Collector manifest has no file evidence: {session}")
            seen_files = set()
            for item in files:
                relative = Path(item["archive_path"])
                local = (session / relative).resolve()
                if relative.is_absolute() or ".." in relative.parts or not local.is_relative_to(session.resolve()):
                    raise ValueError("Invalid collector archive path")
                if str(local) in seen_files:
                    raise ValueError("Duplicate collector archive path")
                seen_files.add(str(local))
                if not Path(lp(local)).is_file() or Path(lp(local)).stat().st_size != item["size"] or sha(local) != item["sha256"]:
                    raise ValueError(f"Collected bytes differ from manifest: {local}")
                sources.append(local)
            required_hashed = {str((session / name).resolve()) for name in ("experiment_curve.csv", "audit_rounds.jsonl", "completion.json")}
            if not required_hashed.issubset(seen_files):
                raise ValueError(f"Collector omitted required file checksums: {session}")
            with open(lp(session / "audit_rounds.jsonl"), encoding="utf-8-sig") as stream:
                audit = [json.loads(line) for line in stream if line.strip()]
            records = [r for r in audit if r.get("status") == "complete"]
            wanted = {(method, seed, b) for b in p["budgets"]}
            if len(records) != 4 or {analysis.key(r) for r in records} != wanted:
                raise ValueError(f"New session needs four unique completed budgets: {session}")
            for r in records:
                k = analysis.key(r)
                index = p["budgets"].index(k[2])
                if int(r["round"]) != index or int(r["n_labeled"]) != round(p["source_pool_size_expected"] * k[2]):
                    raise ValueError(f"Wrong new-session round or accepted-frame target: {k}")
                if int(r["gt_additional_frames"]) != 0:
                    raise ValueError(f"Additional GT labels in automatic policy: {k}")
                if index:
                    if int(r["gt_initial_frames"]) != 0 or int(r["gt_query_calls"]) != 0:
                        raise ValueError(f"Post-initialization GT queries in automatic policy: {k}")
                else:
                    initial = r.get("initialization") or {}
                    if initial.get("reuse") is not True or initial.get("kind") not in {
                        "legacy_reference_verified_initialization", "same_process_shared_initial"
                    }:
                        raise ValueError(f"Shared original initialization was not reused: {k}")
                    original_sha = training[("A", seed, .05)]["checkpoint_sha256"]
                    if r["checkpoint_sha256"] != original_sha or initial.get("checkpoint_sha256") != original_sha:
                        raise ValueError(f"Initial checkpoint differs from original A seed {seed}")
                    target = round(p["source_pool_size_expected"] * .05)
                    if int(r["gt_initial_frames"]) != target or int(r["gt_query_calls"]) != target:
                        raise ValueError(f"Unexpected initial GT count: {k}")
    return sources


def load_analysis(path):
    spec = importlib.util.spec_from_file_location("rehabal_final_analysis", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def self_test():
    import itertools
    p = {"methods": ["A", "B", "ref", "A_jointmask", "A_flipcons"], "seeds": list(range(6)),
         "budgets": [.05, .1, .2, .4], "penn_budgets": [.4]}
    t = {k: {"checkpoint_sha256": "a"*64} for k in itertools.product(p["methods"], p["seeds"], p["budgets"])}
    e = {k: {"checkpoint_sha256": "a"*64} for k in itertools.product(p["methods"], p["seeds"], [.4])}
    require_complete(t, e, p, True, [])
    require_complete(t, e, p, True, [{"type": "outside_planned_external_matrix"}])
    failures = [(dict(list(t.items())[:-1]), e, True, []), (t, dict(list(e.items())[:-1]), True, []),
                (t, e, False, []), (t, e, True, [{"type": "training_csv_row_not_complete"}])]
    bad = {k: dict(v) for k, v in e.items()}
    bad[next(iter(bad))]["checkpoint_sha256"] = "b"*64
    failures.append((t, bad, True, []))
    for a, b, c, d in failures:
        try:
            require_complete(a, b, p, c, d)
        except ValueError:
            continue
        raise AssertionError("Unsafe completion was accepted")
    import tempfile
    from types import SimpleNamespace
    with tempfile.TemporaryDirectory(prefix="rehabal_final_gate_SYNTHETIC_") as temp:
        base = Path(temp)
        p["source_pool_size_expected"] = 8367
        for method in sorted(NEW_METHODS):
            for seed in p["seeds"]:
                session = base / f"{method}_s{seed}"
                session.mkdir()
                (session / "experiment_curve.csv").write_text("SYNTHETIC_SCHEMA_ONLY\n", encoding="utf-8")
                (session / "completion.json").write_text("{}", encoding="utf-8")
                records = []
                for index, budget in enumerate(p["budgets"]):
                    increment = [418,419,836,1674][index]
                    records.append({"condition":method,"seed":seed,"budget":budget,"round":index,"status":"complete",
                                    "n_labeled":[418,837,1673,3347][index],"accepted_frames":increment,
                                    "proposed_frames":2*increment if index else 0,"rejected_frames":0,
                                    "gt_initial_frames":418 if index==0 else 0,"gt_additional_frames":0,
                                    "gt_query_calls":418 if index==0 else 0,"checkpoint_sha256":"a"*64,
                                    "visible_pseudo_joints":20*increment if index else 0,
                                    "automatic_masked_joints":6*increment if index else 0,
                                    "consensus_joints":20*increment if index and method=="A_flipcons" else 0,
                                    "confidence_threshold":.4 if index else None,
                                    "initialization":{"reuse":True,"kind":"legacy_reference_verified_initialization",
                                                      "checkpoint_sha256":"a"*64} if index==0 else None})
                (session / "audit_rounds.jsonl").write_text("\n".join(json.dumps(r) for r in records)+"\n",encoding="utf-8")
                files=[{"archive_path":name,"source_path":"SYNTHETIC","size":(session/name).stat().st_size,
                        "sha256":sha(session/name)} for name in ("experiment_curve.csv","audit_rounds.jsonl","completion.json")]
                manifest={"status":"verified","validation":{"status":"verified","condition":method,"seed":seed,
                          "budgets":p["budgets"],"targets":[418,837,1673,3347],"completed_rounds":4,
                          "gt_query_calls_after_initialization":0,"gt_additional_frames":0,"initialization_reuse":True,
                          "checkpoint_sha256":"a"*64},"files":files}
                digest=hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
                receipt={"status":"complete","session_id":session.name,"manifest_digest":digest,"completed_rounds":4,
                         "checkpoint_sha256":"a"*64,"exact_budget_verified":True,"gt_query_calls_after_initialization":0,
                         "initialization_reuse_verified":True,"transfer_and_file_checksums_verified":True}
                write_json(session/"collection_manifest.json",manifest)
                write_json(session/"collect_receipt.json",receipt)
        helper=SimpleNamespace(key=lambda r:(str(r["condition"]),int(r["seed"]),round(float(r["budget"]),8)))
        incoming=base/".incoming_A_jointmask_s0_SYNTHETIC"
        incoming.mkdir()
        (incoming/"experiment_curve.csv").write_text("INCOMPLETE_SYNTHETIC_TRANSFER",encoding="utf-8")
        validate_new_sessions(base,p,t,helper)
        diagnostic_module=load_analysis(Path(__file__).with_name("summarize_policy_diagnostics.py"))
        rows,_=diagnostic_module.collect(base)
        assert len(rows)==48 and len(diagnostic_module.summarize(rows))==8
        bad_session=base/"A_jointmask_s0"
        bad_manifest=read_json(bad_session/"collection_manifest.json")
        bad_manifest["files"]=[r for r in bad_manifest["files"] if r["archive_path"]!="audit_rounds.jsonl"]
        bad_receipt=read_json(bad_session/"collect_receipt.json")
        bad_receipt["manifest_digest"]=hashlib.sha256(json.dumps(bad_manifest,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
        (bad_session/"collection_manifest.json").write_text(json.dumps(bad_manifest),encoding="utf-8")
        (bad_session/"collect_receipt.json").write_text(json.dumps(bad_receipt),encoding="utf-8")
        try:
            validate_new_sessions(base,p,t,helper)
        except ValueError as error:
            assert "omitted required file checksums" in str(error)
        else:
            raise AssertionError("Collector manifest omission was accepted")
    print("FINAL_GATE_SYNTHETIC_PASS: 9 cases including 12 synthetic collector sessions, ignored incoming transfer, and 48 diagnostic rows; no real measurements read")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-root", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--execute", action="store_true", help="Read completed measurements and publish the final analysis")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if args.self_test:
        return self_test()
    work = args.work_root.resolve()
    results = work / "results"
    new = results / "new_baselines"
    source_csv = work / "server_metadata/outputs/experiments/clinician_i4_uniform30/clinician_curve.csv"
    original_audit = results / "provenance/original_nano_training_audit.jsonl"
    receipt = results / "provenance/comparability_receipt.json"
    protocol = results / "analysis_protocol.json"
    penn_protocol = results / "protocol_penn.txt"
    penn = results / "penn_action"
    analyzer = results / "code/summarize_experiments.py"
    diagnostics = results / "code/summarize_policy_diagnostics.py"
    destination = results / "final_analysis"
    new_csvs = [new / f"{method}_s{seed}" / "experiment_curve.csv" for method in sorted(NEW_METHODS) for seed in range(6)]
    new_audits = [q.with_name("audit_rounds.jsonl") for q in new_csvs]
    command = [sys.executable, str(analyzer), "--protocol", str(protocol),
               "--comparability-receipt", str(receipt), "--penn-dir", str(penn)]
    for path in [source_csv, *new_csvs]:
        command.extend(["--training-csv", str(path)])
    for path in [original_audit, *new_audits]:
        command.extend(["--training-audit-jsonl", str(path)])
    if not args.execute:
        print(json.dumps({"mode": "plan_only_no_measurement_files_read", "command": command + ["--output", "<new attempt>/artifacts"],
                          "destination_after_validation_only": str(destination), "training_rows_required": 120,
                          "external_models_required": 30, "policy_diagnostics": str(diagnostics), "new_session_layout": str(new / "<condition>_s<seed>"),
                          "next_step": "Rerun with --execute only after collection and all Penn models finish"}, ensure_ascii=False, indent=2))
        return 0
    if Path(lp(destination)).exists():
        raise ValueError("Final output already exists; preserve it and explicitly review before any replacement")
    lock = results / ".final_analysis.lock"
    with open(lp(lock), "x", encoding="utf-8") as stream:
        stream.write(f"pid={os.getpid()}\n")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    attempt = results / "analysis_attempts" / stamp
    try:
        Path(lp(attempt)).mkdir(parents=True, exist_ok=False)
        write_json(attempt / "run_started.json", {"status": "validating", "created_at_utc": datetime.now(timezone.utc).isoformat(), "final_not_published": True})
        analysis = load_analysis(analyzer)
        if sha(protocol) != LOCKED_ANALYSIS_PROTOCOL_SHA256 or sha(penn_protocol) != LOCKED_PENN_PROTOCOL_SHA256:
            raise ValueError("Frozen analysis or Penn protocol changed; review explicitly before any new analysis")
        p = analysis.read_json(protocol)
        analysis.check_protocol(p)
        training, train_sources, train_issues = analysis.load_training(None, p, [source_csv, *new_csvs], [original_audit, *new_audits])
        external, ext_sources, ext_issues = analysis.load_penn(penn, p)
        for model in external.values():
            if model["metadata"].get("protocol_sha256") != LOCKED_PENN_PROTOCOL_SHA256 or model["metadata"].get("prepare_sha256") != LOCKED_PENN_PREPARE_SHA256:
                raise ValueError("External measurements differ from the prospectively frozen Penn protocol/prepare")
        comparable, comp_issues = analysis.check_comparability(receipt)
        require_complete(training, external, p, comparable, train_issues + ext_issues + comp_issues)
        collected_sources = validate_new_sessions(new, p, training, analysis)
        input_files = sorted(set(map(str, [protocol, penn_protocol, receipt, analyzer, diagnostics, Path(__file__), *train_sources, *ext_sources, *collected_sources])))
        hashes_before = {path: sha(path) for path in input_files}
        out = attempt / "artifacts"
        command += ["--output", str(out)]
        write_json(attempt / "execution_plan.json", {"command": command, "input_sha256": hashes_before})
        with open(lp(attempt / "analysis.log"), "xb") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=work, check=False)
        if process.returncode:
            raise RuntimeError(f"Analyzer exited {process.returncode}; see analysis_attempts log")
        diagnostic_command = [sys.executable, str(diagnostics), "--training-dir", str(new), "--output", str(out / "policy_diagnostics")]
        with open(lp(attempt / "policy_diagnostics.log"), "xb") as log:
            diagnostic_process = subprocess.run(diagnostic_command, stdout=log, stderr=subprocess.STDOUT, cwd=work, check=False)
        if diagnostic_process.returncode:
            raise RuntimeError("Policy audit failed; final results remain unpublished")
        diagnostic_status = read_json(out / "policy_diagnostics/policy_diagnostics_metadata.json")
        if diagnostic_status.get("status") != "complete_descriptive_policy_audit" or diagnostic_status.get("records") != 48:
            raise RuntimeError("Incomplete policy diagnostics")
        status = read_json(out / "analysis_status.json")
        if (status.get("status") != "complete_planned_suite" or status.get("training_completed") != 120
            or status.get("external_completed") != 30 or status.get("missing_training") or status.get("missing_external")
            or not status.get("comparability_verified") or not status.get("manuscript_insert_generated")
            or any(r.get("type") not in NONPRIMARY_ISSUES for r in status.get("issues", []))):
            raise RuntimeError("Analyzer did not validate a fully completed planned suite")
        if hashes_before != {path: sha(path) for path in input_files}:
            raise RuntimeError("Input or analysis code changed during final analysis")
        expected = [name + ".csv" for name in TABLES] + [name + ext for name in FIGURES for ext in (".png", ".pdf")]
        expected += ["analysis_status.json", "결과_읽는법.txt", "manuscript_insert_observed_results.txt"]
        expected += ["policy_diagnostics/" + name for name in ("policy_diagnostics_rounds.csv", "policy_diagnostics_summary.csv", "policy_diagnostics_metadata.json", "정책진단_읽는법.txt")]
        for name in expected:
            if not Path(lp(out / name)).is_file() or Path(lp(out / name)).stat().st_size == 0:
                raise RuntimeError(f"Missing final artifact: {name}")
        receipt_record = {"status": "complete_validated", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                          "final_path": str(destination), "attempt_path": str(attempt), "command": command,
                          "training_rows": 120, "external_models": 30, "new_sessions": 12, "policy_diagnostics_command": diagnostic_command,
                          "input_sha256": hashes_before, "historical_comparability_qualifications_preserved": True,
                          "artifact_sha256": {name: sha(out / name) for name in expected}}
        write_json(out / "final_publication_receipt.json", receipt_record)
        if Path(lp(destination)).exists():
            raise RuntimeError("Final destination appeared concurrently; refusing to overwrite")
        os.rename(lp(out), lp(destination))
        write_json(attempt / "run_completed.json", {"status": "complete_validated", "final_path": str(destination)})
        print("최종 분석 완료: 120개 source 관측, 30개 Penn 모델. " + str(destination))
        return 0
    except Exception as error:
        if Path(lp(attempt)).is_dir():
            write_json(attempt / "run_failed.json", {"status": "failed_not_published", "error_type": type(error).__name__,
                                                      "error": str(error), "final_not_published": not Path(lp(destination)).exists()})
        print("최종 분석을 완료로 발행하지 못했습니다: " + str(error), file=sys.stderr)
        return 2
    finally:
        Path(lp(lock)).unlink()


if __name__ == "__main__":
    raise SystemExit(main())
