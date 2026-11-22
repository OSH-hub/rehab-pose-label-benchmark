"""Read-only original checkpoint initialization preflight. NEVER calls training."""
import json
import os
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

WORK = Path(r"rehabal_work\experiment_20261012")
REPO = Path(r"RehabAL-pose")
OUT = WORK / "preflight" / "baseline_references"
if OUT.exists():
    raise SystemExit("Preflight output already exists; preserve it and inspect before repeating")
OUT.mkdir(parents=True)
os.chdir(REPO)
sys.path[:0] = [str(WORK / "code"), str(REPO / "scripts"), str(REPO / "src")]
import numpy as np
import torch
import cv2
import run_strengthened_baselines as extension
import run_clinician_experiment as original
from rehabal_pose.models.yolo_pose import YoloPoseModel
from rehabal_pose.review.clinician_loop import write_labels_dir

expected_path = WORK / "preflight" / "baseline_expected_packages.json"
reference_hashes = WORK / "preflight" / "baseline_reference_hashes.json"
expected = json.loads(expected_path.read_text(encoding="utf-8-sig"))
packages = {p: version(p) for p in expected}
assert packages == expected, (packages, expected)
assert cv2.__version__ == "4.12.0", cv2.__version__
training_calls = []
model_calls = []
def prohibit_train(*args, **kwargs):
    training_calls.append("forbidden_training_attempt")
    raise RuntimeError("TRAINING IS FORBIDDEN IN REFERENCE PREFLIGHT")
def prohibit_model(*args, **kwargs):
    model_calls.append("forbidden_model_execution_attempt")
    raise RuntimeError("MODEL EXECUTION IS FORBIDDEN IN REFERENCE PREFLIGHT")
YoloPoseModel.train = prohibit_train
YoloPoseModel._ensure_model = prohibit_model
YoloPoseModel.load = prohibit_model
YoloPoseModel.predict = prohibit_model
YoloPoseModel.evaluate = prohibit_model
core_paths = [REPO / "scripts/run_clinician_experiment.py", REPO / "scripts/al_budget_curve.py",
              REPO / "src/rehabal_pose/review/clinician_loop.py",
              REPO / "src/rehabal_pose/models/yolo_pose.py"]
core_before = {str(p): extension.sha256(p) for p in core_paths}
images = [str(p) for p in sorted(original.IMAGES_TRAIN.glob("*.jpg"))]
stems = [Path(p).stem for p in images]
targets = original._budget_targets([.05, .10, .20, .40], stems, len(stems))
assert len(stems) == 8367 and targets == [418, 837, 1673, 3347], (len(stems), targets)
# Build exactly the same original GT lookup; no scorer, inferencer, or trainer is invoked.
_, _, _, gt_lookup = original._make_callables(images, {}, "temporal", 30, 416, 16, 30, "yolo11n-pose.pt")
args = SimpleNamespace(reference_root=REPO / "outputs/experiments/clinician_i4_uniform30",
                       reference_hashes=reference_hashes, epochs_round0=30, imgsz=416, batch=16)
config = {"weights": "yolo11n-pose.pt"}
records = []
for seed in range(6):
    indices = np.random.default_rng(1000 + seed).choice(len(stems), size=targets[0], replace=False)
    ids = [stems[i] for i in indices]
    labels = {}
    sizes = {}
    for sid in ids:
        labels[sid] = np.asarray(gt_lookup(sid), dtype=np.float64).tolist()
        sizes[sid] = original._get_image_size(original.IMAGES_TRAIN / f"{sid}.jpg")
    new_round = OUT / f"seed_{seed}" / "round_0"
    write_labels_dir(labels, new_round, image_sizes=sizes)
    result, provenance = extension.try_reference_initialization(args, original, seed, ids, labels,
                                                                new_round, config)
    records.append({"seed": seed, "initial_frames": len(ids), "gt_lookup_calls": len(ids),
                    "initial_ids_sha256": extension.json_hash(ids), "provenance": provenance,
                    "result": result})
    print("REFERENCE_PREFLIGHT", seed, "REUSE", provenance.get("reuse"),
          provenance.get("reason", ""), flush=True)
core_after = {str(p): extension.sha256(p) for p in core_paths}
assert core_before == core_after
receipt = {"created_at_utc": datetime.now(timezone.utc).isoformat(),
           "status": "passed" if all(r["provenance"].get("reuse") for r in records) else "failed",
           "training_performed": False, "training_call_count": len(training_calls),
           "model_execution_call_count": len(model_calls), "model_execution_disabled_by_tripwire": True,
           "torch_cuda_initialized": torch.cuda.is_initialized(),
           "python_executable": sys.executable, "packages": packages, "cv2_version": cv2.__version__,
           "source_repository": str(REPO), "original_core_sources_unchanged": core_before == core_after,
           "original_core_sha256": core_before, "pool_frames": len(stems), "budget_targets": targets,
           "reference_reuse_count": sum(r["provenance"].get("reuse", False) for r in records),
           "total_seeds": len(records), "gt_lookup_calls_total": sum(r["gt_lookup_calls"] for r in records),
           "wrapper_sha256": extension.sha256(WORK / "code/run_strengthened_baselines.py"),
           "preflight_script_sha256": extension.sha256(__file__),
           "reference_hashes_sha256": extension.sha256(reference_hashes),
           "records": records,
           "limitation": "Legacy initialization artifacts verified; historical whole-dataset immutability is not proven."}
assert not training_calls and not model_calls and not torch.cuda.is_initialized()
receipt_path = OUT / "reference_preflight_receipt.json"
receipt_path.write_text(json.dumps(receipt, indent=2, allow_nan=False), encoding="utf-8")
print("RECEIPT", receipt_path, "STATUS", receipt["status"], flush=True)
raise SystemExit(0 if receipt["status"] == "passed" else 1)
