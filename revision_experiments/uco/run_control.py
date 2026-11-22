"""Post-hoc matched-resolution control. Never writes to the original UCO tree."""
from pathlib import Path
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import traceback

HERE = Path(__file__).resolve().parent
WORK = HERE.parent
OLD = WORK / "uco"
CODE = HERE / "code"
REGISTRY = WORK / "penn" / "registry_server.json"
METRICS = ["detection_rate", "angle_mae_det", "angle_acc10_all", "angle_acc15_all", "pck02_all", "pck02_det", "px_err_det"]
ORDER = [f"{m}_n_s{s}_b040" for m in ("A", "B", "ref") for s in range(6)]
ORDER += [f"A_n_s{s}_b005" for s in range(6)]
ORDER += [f"{m}_n_s{s}_b040" for m in ("A_jointmask", "A_flipcons") for s in range(6)]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def write_new(p, value):
    if p.exists():
        existing = p.read_text(encoding="utf-8-sig")
        if existing != value:
            raise RuntimeError("Refusing to change existing artifact: " + str(p))
        return
    with p.open("x", encoding="utf-8", newline="\n") as f:
        f.write(value)


def json_new(p, value):
    if p.exists():
        existing = read_json(p)
        if "created_at_utc" in existing and "created_at_utc" in value:
            value["created_at_utc"] = existing["created_at_utc"]
    write_new(p, json.dumps(value, ensure_ascii=False, indent=2))


def state(status, **extra):
    p = HERE / "status.json"
    old = read_json(p) if p.exists() else {}
    value = {**old, "status": status, "updated_at_utc": now(), "pid": os.getpid(), **extra}
    temp = p.with_suffix(".tmp")
    write_new(temp, json.dumps(value, ensure_ascii=False, indent=2))
    os.replace(temp, p)
    print(json.dumps(value), flush=True)


def run(args, logname):
    logpath = HERE / "logs" / logname
    if logpath.exists():
        with logpath.open(encoding="utf-8", errors="replace") as previous:
            previous.read()
    with logpath.open("a", encoding="utf-8") as log:
        log.write("\nRUN START " + now() + "\n")
        log.flush()
        subprocess.run(args, cwd=HERE, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)


def csv_new(p, rows):
    if p.exists():
        with p.open(encoding="utf-8-sig", newline="") as f:
            previous = list(csv.DictReader(f))
        expected = [{k: str(v) for k, v in r.items()} for r in rows]
        if previous != expected:
            raise RuntimeError("Refusing to change existing table: " + str(p))
        return
    with p.open("x", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(preflight):
    import numpy as np
    pair_rows = []
    group_rows = []
    checks = []
    for mid in ORDER:
        old = read_json(OLD / "evaluations" / mid / "metrics.json")
        new = read_json(HERE / "evaluations" / mid / "metrics.json")
        equal_keys = ["checkpoint_sha256", "actual_kpt_shape", "joint_map", "frames", "clips", "stride", "offset",
                      "primary_box_conf_threshold", "manifest_sha256", "gt_npz_sha256", "batch"]
        for k in equal_keys:
            assert old[k] == new[k], (mid, k, old[k], new[k])
        oldcfg, newcfg = dict(old["inference_config"]), dict(new["inference_config"])
        assert oldcfg.pop("imgsz") == 640 and newcfg.pop("imgsz") == 416
        assert oldcfg == newcfg, mid
        oldv, newv = dict(old["package_versions"]), dict(new["package_versions"])
        oldv.pop("cuda_visible_devices", None)
        newv.pop("cuda_visible_devices", None)
        assert oldv == newv, (mid, "package_versions")
        assert new["evaluator_sha256"] == preflight["control_evaluator_sha256"]
        for metric in METRICS:
            a = old["metrics"]["subject_macro"]["overall"][metric]
            b = new["metrics"]["subject_macro"]["overall"][metric]
            pair_rows.append({"model_id": mid, "method": old["method"], "budget": old["budget"], "seed": old["seed"],
                              "metric": metric, "value_640": a, "value_416": b, "delta_416_minus_640": b - a})
        checks.append({"model_id": mid, "invariants_match": True, "checkpoint_sha256": new["checkpoint_sha256"],
                       "physical_gpu_mask": new["package_versions"].get("cuda_visible_devices"),
                       "old_metrics_sha256": sha(OLD / "evaluations" / mid / "metrics.json"),
                       "new_metrics_sha256": sha(HERE / "evaluations" / mid / "metrics.json")})
    groups = [("A", .4), ("B", .4), ("ref", .4), ("A", .05), ("A_jointmask", .4), ("A_flipcons", .4)]
    for method, budget in groups:
        for metric in METRICS:
            rows = [r for r in pair_rows if (r["method"], r["budget"], r["metric"]) == (method, budget, metric)]
            assert sorted(r["seed"] for r in rows) == list(range(6))
            row = {"method": method, "budget": budget, "n_seeds": 6, "metric": metric}
            for key in ("value_640", "value_416", "delta_416_minus_640"):
                values = np.asarray([r[key] for r in rows], dtype=float)
                row[key + "_mean"] = float(values.mean())
                row[key + "_sd"] = float(values.std(ddof=1))
            group_rows.append(row)
    csv_new(HERE / "comparison" / "paired_per_model_416_vs_640.csv", pair_rows)
    csv_new(HERE / "comparison" / "paired_per_method_416_vs_640.csv", group_rows)
    summary = {"status": "complete", "created_at_utc": now(), "models": 36, "groups": 6, "seeds_per_group": 6,
               "frames_per_model": 60824, "clips_per_model": 1968, "subjects": 27, "only_inference_change": {"imgsz": [640, 416]},
               "matched_keys": equal_keys, "all_inference_fields_except_imgsz_match": True,
               "same_package_versions": True, "gpu_note": "Physical GPU masks are recorded per model; the original CUDA wrapper exposes logical device 0.",
               "original_results_preserved": True, "preflight": preflight, "model_checks": checks,
               "interpretation": ["Post-hoc matched-resolution sensitivity analysis, not a new training run.",
                                  "Rates are fractions. Multiply rate differences by 100 for percentage points.",
                                  "Per-method SD is sample SD across six matched training seeds, not a confidence interval.",
                                  "Angle MAE is conditional on detection; detection/all-frame metrics must be interpreted alongside it.",
                                  "The original paired subject bootstrap is retained separately in aggregate/contrasts_vs_A.csv."]}
    json_new(HERE / "comparison" / "comparison_summary.json", summary)
    lines = ["UCO 416 vs 640 matched-resolution control", "", "36 unchanged checkpoints; 6 groups x seeds 0-5.",
             "Same 60,824 cached full frames from 1,968 clips and 27 participants; stride 25, offset +5.",
             "Only inference imgsz changed from 640 to 416. All remaining scoring/inference settings and package versions match.",
             "Rates below are percentages; differences are percentage points. Angle MAE is degrees and conditional on detection.", ""]
    for method, budget in groups:
        lines.append(f"{method}, budget={budget}")
        for r in group_rows:
            if (r["method"], r["budget"]) == (method, budget):
                scale = 1. if r["metric"] in ("angle_mae_det", "px_err_det") else 100.
                lines.append(f"  {r['metric']}: 640={r['value_640_mean']*scale:.4f} +/- {r['value_640_sd']*scale:.4f}; "
                             f"416={r['value_416_mean']*scale:.4f} +/- {r['value_416_sd']*scale:.4f}; "
                             f"paired delta={r['delta_416_minus_640_mean']*scale:+.4f} +/- {r['delta_416_minus_640_sd']*scale:.4f}")
    write_new(HERE / "comparison" / "comparison_summary.txt", "\n".join(lines) + "\n")


def main():
    if (HERE / "DONE.txt").exists():
        print((HERE / "DONE.txt").read_text(encoding="utf-8"), flush=True)
        return
    state("preflight", started_at_utc=now(), expected_models=36, physical_gpu=1)
    for name in ("code", "logs", "evaluations", "aggregate", "comparison"):
        (HERE / name).mkdir(exist_ok=True)
    original = (OLD / "code" / "uco_evaluate.py").read_text(encoding="utf-8")
    modified, count = re.subn(r'("imgsz"\s*:\s*)640\b', r'\g<1>416', original)
    assert count == 1, count
    write_new(CODE / "uco_evaluate.py", modified)
    for name in ("uco_aggregate.py", "cuda_wrapper.py"):
        target = CODE / name
        if target.exists():
            assert target.read_bytes() == (OLD / "code" / name).read_bytes(), target
        else:
            shutil.copy2(OLD / "code" / name, target)
    original_protocol = OLD / "protocol_uco.txt"
    addendum = ("\n\nPOST-HOC MATCHED-RESOLUTION CONTROL ADDENDUM, 2026-10-26\n"
                "This addendum overrides only the inference imgsz=640 entry above: use imgsz=416.\n"
                "All checkpoints, 60,824 cached sample frames, GT, +5 offset, stride 25, mapping, masks,\n"
                "thresholds, NMS, batch, precision, scoring, and aggregation remain identical.\n"
                "This is a post-hoc resolution sensitivity analysis, not a pre-registered confirmatory analysis.\n"
                "Original 640 outputs are read-only. Control outputs use uco_416_control_20261026.\n")
    write_new(HERE / "protocol_416_control.txt", original_protocol.read_text(encoding="utf-8-sig") + addendum)
    registry = read_json(REGISTRY)
    entries = {e["model_id"]: e for e in registry["models"]}
    assert len(entries) == 36 and set(entries) == set(ORDER)
    src_hash = sha(OLD / "code" / "uco_evaluate.py")
    meta = read_json(OLD / "prepared" / "prepare_meta.json")
    assert meta["status"] == "complete" and meta["frames_scored"] == 60824 and meta["clips_scored"] == 1968
    assert meta["stride"] == 25 and meta["offset"] == 5
    assert sha(OLD / "prepared" / "manifest.jsonl") == meta["manifest_sha256"]
    assert sha(OLD / "prepared" / "gt.npz") == meta["gt_npz_sha256"]
    weight_checks = []
    for mid in ORDER:
        e = entries[mid]
        old = read_json(OLD / "evaluations" / mid / "metrics.json")
        assert old["evaluator_sha256"] == src_hash, (mid, "original_evaluator_hash")
        assert old["inference_config"]["imgsz"] == 640
        assert old["frames"] == 60824 and old["clips"] == 1968
        ck = Path(e["checkpoint"])
        cksha = sha(ck)
        assert cksha == e["checkpoint_sha256"] == old["checkpoint_sha256"], mid
        weight_checks.append({"model_id": mid, "checkpoint_sha256": cksha, "original_metrics_sha256": sha(OLD / "evaluations" / mid / "metrics.json")})
    json_new(HERE / "registry_snapshot.json", registry)
    preflight = {"created_at_utc": now(), "models_verified": 36, "weight_checks": weight_checks,
                 "original_evaluator_sha256": src_hash, "control_evaluator_sha256": sha(CODE / "uco_evaluate.py"),
                 "original_protocol_sha256": sha(original_protocol), "control_protocol_sha256": sha(HERE / "protocol_416_control.txt"),
                 "registry_sha256": sha(REGISTRY), "manifest_sha256": meta["manifest_sha256"], "gt_npz_sha256": meta["gt_npz_sha256"],
                 "evaluator_change": "Exactly one imgsz literal 640 -> 416; all other source text unchanged.",
                 "prepare_meta": meta}
    json_new(HERE / "preflight.json", preflight)
    os.environ.update(KMP_DUPLICATE_LIB_OK="TRUE", PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="1")
    args = [sys.executable, "-u", str(CODE / "cuda_wrapper.py"), str(CODE / "uco_evaluate.py"),
            "--prepared", str(OLD / "prepared"), "--registry", str(REGISTRY), "--protocol", str(HERE / "protocol_416_control.txt"),
            "--out", str(HERE / "evaluations"), "--device", "0", "--batch", "32", "--chunk-size", "512",
            "--loader-threads", "8", "--models", ",".join(ORDER)]
    json_new(HERE / "execution_command.json", {"argv": args, "cwd": str(HERE), "physical_gpu": 1})
    state("evaluating", models_verified=36)
    run(args, "evaluation.log")
    state("aggregating")
    if (HERE / "aggregate" / "summary.json").exists():
        aggregate = read_json(HERE / "aggregate" / "summary.json")
        assert set(aggregate["complete_models"]) == set(ORDER) and not aggregate["missing_models"]
    else:
        for existing in (HERE / "aggregate").glob("*"):
            if existing.is_file():
                existing.read_bytes()
        run([sys.executable, str(CODE / "uco_aggregate.py"), "--evaluations", str(HERE / "evaluations"),
             "--registry", str(REGISTRY), "--out", str(HERE / "aggregate"), "--require-all"], "aggregate.log")
    state("comparing")
    compare(preflight)
    state("complete", completed_models=36, completed_at_utc=now())
    write_new(HERE / "DONE.txt", now() + "\n36/36 models, aggregation and matched comparison complete.\n")


if __name__ == "__main__":
    lock = HERE / "runner.lock"
    with lock.open("x", encoding="utf-8") as f:
        f.write(json.dumps({"pid": os.getpid(), "started_at_utc": now()}))
    try:
        main()
    except BaseException as exc:
        state("failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        lock.read_text(encoding="utf-8")
        lock.unlink()
