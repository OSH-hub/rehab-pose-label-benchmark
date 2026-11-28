"""Additional GT-free REHAB pseudo-label baselines; original files remain unchanged."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ARMS = ("A_jointmask", "A_flipcons")
FLIP = np.array([0, 1, 2, 3, 4, 5, 11, 12, 13, 14, 15, 6, 7, 8, 9, 10,
                 21, 22, 23, 24, 25, 16, 17, 18, 19, 20])
POLICY_VERSION = "joint-confidence-quantile-flip-consensus-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.read_bytes()
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temp, path)


def append_event(path, event):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, allow_nan=False) + "\n")


def confidence_quantile(predictions, q):
    values = np.asarray(predictions, dtype=float)[..., 2]
    values = values[np.isfinite(values) & (values > 0) & (values <= 1)]
    if not values.size:
        raise ValueError("No positive finite pool confidence; no label-budget fallback")
    if not 0 <= q <= 1:
        raise ValueError("Quantile must be in [0,1]")
    return float(np.quantile(values, q))


def valid_joints(kp, width, height, tau):
    kp = np.asarray(kp, dtype=float)
    if kp.shape != (26, 3):
        raise ValueError(f"Expected (26,3) keypoints, got {kp.shape}")
    return (np.isfinite(kp).all(axis=1) & (kp[:, 2] >= tau) & (kp[:, 2] > 0)
            & (kp[:, 2] <= 1) & (kp[:, 0] >= 0) & (kp[:, 0] < width)
            & (kp[:, 1] >= 0) & (kp[:, 1] < height))


def unflip_keypoints(kp, width):
    out = np.asarray(kp, dtype=float)[FLIP].copy()
    if out.shape != (26, 3):
        raise ValueError("Flip prediction must have 26 joints")
    out[:, 0] = width - 1 - out[:, 0]
    return out


def make_pseudo_label(kp, width, height, tau, flipped=None, tolerance=.05):
    kp = np.asarray(kp, dtype=float)
    valid = valid_joints(kp, width, height, tau)
    xy = kp[:, :2].copy()
    confidence_pass = int(valid.sum())
    if flipped is not None:
        flipped = np.asarray(flipped, dtype=float)
        valid &= valid_joints(flipped, width, height, tau)
        error = np.linalg.norm(kp[:, :2] - flipped[:, :2], axis=1)
        valid &= error <= tolerance * np.hypot(width, height)
        xy = (kp[:, :2] + flipped[:, :2]) / 2
    label = np.zeros((26, 3), dtype=float)
    label[valid, :2] = xy[valid]
    label[valid, 2] = 1
    return label, {"visible_joints": int(valid.sum()), "masked_joints": int((~valid).sum()),
                   "confidence_pass_joints": confidence_pass,
                   "consensus_joints": int(valid.sum()) if flipped is not None else 0}


def label_is_usable(label, minimum):
    visible = np.asarray(label)[np.asarray(label)[:, 2] > 0, :2]
    return (len(visible) >= minimum and np.ptp(visible[:, 0]) > 0
            and np.ptp(visible[:, 1]) > 0)


def guard_gt(round_index, lookup, stem):
    if round_index != 0:
        raise RuntimeError("GT-free policy attempted training GT access after initialization")
    return lookup(stem)


def validate_initial_state(state, ids, labels):
    if list(state.get("labeled", [])[:len(ids)]) != list(ids):
        raise ValueError("Original initialization order/IDs differ")
    for sid in ids:
        ref = np.asarray(state.get("labels", {}).get(sid), dtype=float)
        if ref.shape != (26, 3) or not np.array_equal(ref, np.asarray(labels[sid], dtype=float)):
            raise ValueError(f"Original initial GT coordinates differ: {sid}")


def build_pool_without_embeddings(original_builder, images, model, metadata):
    class PredictionOnly:
        def predict(self, paths):
            return model.predict(paths)

        def extract_embeddings(self, paths):
            return np.zeros((len(paths), 1), dtype=np.float64)

    pool = original_builder(images, PredictionOnly(), metadata)
    pool.embeddings = None
    return pool


def try_reference_initialization(args, runner, seed, ids, labels, out_dir, config):
    import pandas as pd
    import yaml
    if args.reference_root is None:
        return None, {"reuse": False, "reason": "No reference root specified"}
    session = Path(args.reference_root) / f"A_s{seed}"
    try:
        state_path = session / "state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        validate_initial_state(state, ids, labels)
        rd = session / "round_0"
        args_path = rd / "run" / "train" / "args.yaml"
        previous = yaml.safe_load(args_path.read_text(encoding="utf-8"))
        expected = {"seed": seed, "epochs": args.epochs_round0, "imgsz": args.imgsz,
                    "batch": args.batch, "workers": 0, "deterministic": True,
                    "resume": False, "pretrained": True, "optimizer": "auto",
                    "fliplr": .5, "flipud": 0., "mosaic": 1., "mixup": 0.,
                    "pose": 12., "kobj": 1., "lr0": .01, "nbs": 64,
                    "hsv_h": .015, "hsv_s": .7, "hsv_v": .4,
                    "translate": .1, "scale": .5, "close_mosaic": 10}
        for key, expected_value in expected.items():
            if previous.get(key) != expected_value:
                raise ValueError(f"Original train setting differs: {key}")
        if Path(str(previous.get("model"))).name != Path(config["weights"]).name:
            raise ValueError("Original pretrained model name differs")
        old_labels = rd / "labels" / "train"
        new_labels = Path(out_dir) / "labels" / "train"
        if {p.stem for p in old_labels.glob("*.txt")} != set(ids):
            raise ValueError("Original round0 label file set differs")
        label_ids = []
        for sid in sorted(ids):
            old_file, new_file = old_labels / f"{sid}.txt", new_labels / f"{sid}.txt"
            if old_file.read_bytes() != new_file.read_bytes():
                raise ValueError(f"Original YOLO initialization label bytes differ: {sid}")
            label_ids.append([sid, sha256(old_file)])
        curve = pd.read_csv(session / "curve.csv")
        row = curve.loc[curve["round"] == 0]
        if len(row) != 1 or int(row.iloc[0]["n_labeled"]) != len(ids):
            raise ValueError("Original round0 curve count differs")
        row = row.iloc[0]
        history = [h for h in state.get("history", []) if h.get("round") == 0]
        if len(history) != 1:
            raise ValueError("Missing/duplicate original round0 history")
        for metric in ("map50", "map50_95"):
            if not np.isclose(float(history[0]["pose_" + metric]), float(row[metric]),
                              rtol=0, atol=1e-12):
                raise ValueError("Original curve/state metrics disagree")
        checkpoint = rd / "run" / "train" / "weights" / "best.pt"
        weight_sha = sha256(checkpoint)
        if args.reference_hashes:
            expected_hashes = json.loads(Path(args.reference_hashes).read_text(encoding="utf-8"))
            key = f"A_s{seed}/round_0/run/train/weights/best.pt"
            if expected_hashes.get(key) != weight_sha:
                raise ValueError("Reference checkpoint differs from inventoried SHA256")
        result = {"weights": str(checkpoint.resolve()), "map50": float(row["map50"]),
                  "map50_95": float(row["map50_95"])}
        provenance = {"reuse": True, "kind": "legacy_reference_verified_initialization",
                      "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": weight_sha,
                      "state_sha256": sha256(state_path), "args_sha256": sha256(args_path),
                      "initial_ids_sha256": json_hash(list(ids)),
                      "initial_label_files_sha256": json_hash(label_ids),
                      "historical_dataset_immutability_verified": False}
        return result, provenance
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return None, {"reuse": False, "reason": f"{type(exc).__name__}: {exc}"}


def install_runtime(args, runner, abc, base_loop, context):
    from rehabal_pose.models.yolo_pose import YoloPoseModel, _best_instance
    from rehabal_pose.review.clinician_loop import Decision
    original_builder = runner._build_pool
    runner._build_pool = lambda images, model, metadata: build_pool_without_embeddings(
        original_builder, images, model, metadata)
    original_source = runner._source_identity
    runner._source_identity = lambda: json_hash({"original": original_source(),
                                                 "wrapper": sha256(__file__)})
    runner.CONDITIONS = list(runner.CONDITIONS) + list(ARMS)
    original_decision = runner._decision_fn_for

    def decision_factory(condition, oracle, gt_lookup, **kwargs):
        if condition not in ARMS:
            return original_decision(condition, oracle, gt_lookup, **kwargs)
        def decide(candidate):
            kp = candidate.pred_keypoints
            if label_is_usable(kp, args.min_joints):
                return Decision(sample_id=candidate.sample_id, action="accept", keypoints=kp)
            return Decision(sample_id=candidate.sample_id, action="reject")
        return decide
    runner._decision_fn_for = decision_factory

    class StrengthenedLoop(base_loop):
        def __init__(self, state, **kwargs):
            kwargs["infer_fn"] = None
            kwargs["seg_table"] = None
            super().__init__(state, **kwargs)
            context.loop = self
            context.tau_weights = None
            context.tau = None
            context.diagnostics = {}
            context.gt_queries = 0
            context.train_provenance = {}
            context.inference_images = 0
            context.proposal_seconds = 0.

        def propose(self, n):
            start = time.perf_counter()
            pool, _ = self.score_fn(self._current_weights)
            if context.tau_weights != self._current_weights:
                labeled_set = set(self.state.labeled)
                remaining = np.array([s not in labeled_set for s in pool.sample_ids])
                context.tau = confidence_quantile(pool.predictions[remaining], args.confidence_quantile)
                context.tau_weights = self._current_weights
            candidates = super().propose(n)
            flipped_predictions = {}
            if self.state.condition == "A_flipcons" and candidates:
                import cv2
                model = YoloPoseModel().load(self._current_weights)
                try:
                    net = model._ensure_model()
                    for offset in range(0, len(candidates), args.inference_batch):
                        batch = candidates[offset:offset + args.inference_batch]
                        sources = []
                        for candidate in batch:
                            image = cv2.imread(candidate.image_path)
                            if image is None:
                                raise OSError(f"Unreadable image: {candidate.image_path}")
                            sources.append(np.ascontiguousarray(image[:, ::-1]))
                        results = net.predict(source=sources, conf=.25, imgsz=640,
                                              device=model._resolve_device(model.config["device"]),
                                              verbose=False, workers=0)
                        if len(results) != len(batch):
                            raise RuntimeError("Flip inference result alignment mismatch")
                        context.inference_images += len(batch)
                        for candidate, result, source in zip(batch, results, sources):
                            kp, _, _ = _best_instance(result, 26)
                            if kp is not None:
                                flipped_predictions[candidate.sample_id] = unflip_keypoints(kp, source.shape[1])
                finally:
                    del model
                    runner._free_cuda()
            for candidate in candidates:
                width, height = runner._get_image_size(Path(candidate.image_path))
                flipped = None
                if self.state.condition == "A_flipcons":
                    flipped = flipped_predictions.get(candidate.sample_id, np.zeros((26, 3)))
                label, stats = make_pseudo_label(candidate.pred_keypoints, width, height,
                                                 context.tau, flipped, args.flip_tolerance)
                candidate.pred_keypoints = label.tolist()
                self._last_pred_kp[candidate.sample_id] = candidate.pred_keypoints
                context.diagnostics[candidate.sample_id] = stats
            context.proposal_seconds += time.perf_counter() - start
            return candidates

        def apply_decisions(self, decisions):
            for decision in decisions:
                if decision.action == "accept" and decision.keypoints is not None:
                    self._last_pred_kp[decision.sample_id] = decision.keypoints
            return super().apply_decisions(decisions)

        def _append_curve_csv(self, record):
            record.clinician = "none"
            path = self._session_dir / "curve.csv"
            prior = path.read_text(encoding="utf-8") if path.exists() else ""
            row = {"round": record.round, "n_labeled": record.n_labeled,
                   "map50": record.pose_map50, "map50_95": record.pose_map50_95,
                   "n_proposed": record.n_proposed, "n_accepted": record.n_accepted,
                   "n_rejected": record.n_rejected, "n_corrected": record.n_corrected,
                   "clinician": record.clinician}
            tmp = path.with_suffix(".csv.tmp")
            with tmp.open("w", newline="", encoding="utf-8") as f:
                f.write(prior)
                writer = csv.DictWriter(f, fieldnames=list(row))
                if not prior:
                    writer.writeheader()
                writer.writerow(row)
            os.replace(tmp, path)

        def commit_and_train(self):
            r = self.state.round
            accepted = list(self._pending_accepted)[:self.state.budget_per_round]
            if len(accepted) < self.state.budget_per_round:
                self.state.status = "failed"
                self.state.failure = {"round": r, "error": "Exact label budget cannot be met",
                                      "accepted": len(accepted), "required": self.state.budget_per_round}
                self._save_state()
                append_event(self._session_dir / "audit_rounds.jsonl", {
                    "condition": self.state.condition, "seed": self.state.seed, "round": r,
                    "budget": args.budgets[r], "status": "failed", **self.state.failure})
                raise RuntimeError("Policy exhausted available candidates; no budget fallback")
            start = time.perf_counter()
            record = super().commit_and_train()
            record.clinician = "none"
            self._save_state()
            stats = [context.diagnostics[s] for s in accepted if s in context.diagnostics]
            audit = {"condition": self.state.condition, "seed": self.state.seed,
                     "budget": args.budgets[r], "round": r, "status": "complete",
                     "accepted_frames": record.n_accepted, "n_labeled": record.n_labeled,
                     "proposed_frames": record.n_proposed, "rejected_frames": record.n_rejected,
                     "gt_initial_frames": len(accepted) if r == 0 else 0,
                     "gt_additional_frames": 0, "gt_query_calls": context.gt_queries,
                     "human_minutes_measured": None, "annotation_time_claim": False,
                     "automatic_masked_joints": sum(s["masked_joints"] for s in stats),
                     "consensus_joints": sum(s["consensus_joints"] for s in stats),
                     "visible_pseudo_joints": sum(s["visible_joints"] for s in stats),
                     "confidence_threshold": context.tau if r else None,
                     "extra_flip_inference_images": context.inference_images,
                     "proposal_seconds": context.proposal_seconds,
                     "training_seconds": context.train_seconds,
                     "commit_seconds": time.perf_counter() - start,
                     "checkpoint": str(Path(self.state.current_weights).resolve()),
                     "checkpoint_sha256": sha256(self.state.current_weights),
                     "initialization": context.train_provenance if r == 0 else None,
                     "labels_sha256": json_hash(self.state.labels)}
            append_event(self._session_dir / "audit_rounds.jsonl", audit)
            context.gt_queries = 0
            context.inference_images = 0
            context.proposal_seconds = 0.
            context.diagnostics = {}
            return record
    runner.ClinicianLoop = StrengthenedLoop
    return StrengthenedLoop


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, default=Path("outputs/rehab24_6_yolo"))
    parser.add_argument("--manifest", type=Path, default=Path("outputs/rehab24_6_real/manifest.parquet"))
    parser.add_argument("--segmentation", type=Path, default=Path("data/rehab24_6_raw/Segmentation.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/training/strengthened_suite"))
    parser.add_argument("--reference-root", type=Path)
    parser.add_argument("--reference-hashes", type=Path)
    parser.add_argument("--expected-packages", type=Path, help="JSON package->exact version; mismatch aborts")
    parser.add_argument("--conditions", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--budgets", type=float, nargs="+", default=[.05, .10, .20, .40])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--epochs-round0", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--weights", default="yolo11n-pose.pt")
    parser.add_argument("--confidence-quantile", type=float, default=.25)
    parser.add_argument("--min-joints", type=int, default=6)
    parser.add_argument("--flip-tolerance", type=float, default=.05)
    parser.add_argument("--inference-batch", type=int, default=16)
    parser.add_argument("--restart-incomplete", action="store_true")
    args = parser.parse_args(argv)
    if (not 0 <= args.confidence_quantile <= 1 or not 1 <= args.min_joints <= 26
            or not 0 < args.flip_tolerance <= 1 or args.inference_batch < 1):
        parser.error("Invalid policy parameter")
    for field in ("conditions", "seeds"):
        if len(set(getattr(args, field))) != len(getattr(args, field)):
            parser.error(f"Duplicate {field}")
    sys.path.insert(0, str(args.source_root.resolve() / "src"))
    sys.path.insert(0, str(args.source_root.resolve() / "scripts"))
    import pandas as pd
    import al_budget_curve as abc
    import run_clinician_experiment as runner
    from importlib.metadata import version
    from rehabal_pose.data.yolo_dataset import rehab26_flip_idx
    from rehabal_pose.review.clinician_loop import ClinicianLoop
    if list(FLIP) != rehab26_flip_idx():
        raise RuntimeError("Skeleton flip definition differs")
    packages = {p: version(p) for p in ("numpy", "pandas", "scipy", "torch", "ultralytics", "opencv-python")}
    if args.expected_packages:
        expected = json.loads(args.expected_packages.read_text(encoding="utf-8"))
        mismatches = {p: (v, version(p)) for p, v in expected.items() if version(p) != v}
        if mismatches:
            raise RuntimeError(f"Environment version mismatch: {mismatches}")
    runner.DATASET = abc.DATASET = args.dataset.resolve()
    runner.MANIFEST = abc.MANIFEST = args.manifest.resolve()
    runner.IMAGES_TRAIN = runner.DATASET / "images" / "train"
    runner.SEG_CSV = args.segmentation.resolve()
    pool_images = [str(p) for p in sorted(runner.IMAGES_TRAIN.glob("*.jpg"))]
    stems = [Path(p).stem for p in pool_images]
    targets = runner._budget_targets(args.budgets, stems, len(stems))
    metadata = runner._recording_meta(runner.ManifestBuilder.load(runner.MANIFEST))
    input_identity = runner._input_identity(stems, None)
    context = SimpleNamespace(loop=None, train_seconds=0.)
    install_runtime(args, runner, abc, ClinicianLoop, context)
    original_callables = runner._make_callables(pool_images, metadata, "temporal", args.epochs,
                                               args.imgsz, args.batch, args.epochs_round0, args.weights)
    _, score_fn, original_train, original_gt = original_callables
    config = dict(original_train.experiment_config)
    config.update(policy_version=POLICY_VERSION, confidence_quantile=args.confidence_quantile,
                  min_joints=args.min_joints, flip_tolerance=args.flip_tolerance,
                  inference_imgsz=640, inference_box_conf=.25, inference_batch=args.inference_batch,
                  package_audit=packages, gt_free_after_initialization=True,
                  reference_root=str(args.reference_root.resolve()) if args.reference_root else None,
                  reference_hashes_sha256=sha256(args.reference_hashes) if args.reference_hashes else None)
    shared_initial = {}

    def train(ids, labels, out_dir, seed):
        start = time.perf_counter()
        r = context.loop.state.round
        if r == 0:
            result, provenance = try_reference_initialization(args, runner, seed, ids, labels, out_dir, config)
            cache_key = json_hash({"seed": seed, "ids": ids, "labels": labels, "config": config,
                                   "inputs": input_identity})
            if result is None and cache_key in shared_initial:
                cached, cached_provenance = shared_initial[cache_key]
                if sha256(cached["weights"]) != cached_provenance["checkpoint_sha256"]:
                    raise RuntimeError("Fresh shared initial checkpoint changed")
                result = dict(cached)
                provenance = {**cached_provenance, "reuse": True, "kind": "same_process_shared_initial"}
            if result is None:
                result = original_train(ids, labels, out_dir, seed)
                provenance = {"reuse": False, "kind": "fresh_initial_training",
                              "reference_reuse_rejection": provenance,
                              "checkpoint_sha256": sha256(result["weights"])}
                shared_initial[cache_key] = (dict(result), dict(provenance))
            context.train_provenance = provenance
        else:
            result = original_train(ids, labels, out_dir, seed)
        context.train_seconds = time.perf_counter() - start
        return result
    train.experiment_config = config

    def gt(stem):
        if context.loop is None:
            raise RuntimeError("GT lookup outside active initialization")
        value = guard_gt(context.loop.state.round, original_gt, stem)
        context.gt_queries += 1
        return value

    args.out.mkdir(parents=True, exist_ok=True)
    failures = []
    for seed in args.seeds:
        for condition in args.conditions:
            session = args.out / f"{condition}_s{seed}"
            try:
                rows = runner._run_session(condition, seed, 0., args.budgets, stems, len(stems),
                                           (None, score_fn, train, gt), None, args.out, "temporal",
                                           run_config=config, input_identity=input_identity,
                                           restart_incomplete=args.restart_incomplete)
                for row in rows:
                    row["status"] = "complete"
                csv_path = session / "experiment_curve.csv"
                if csv_path.exists():
                    csv_path.read_bytes()
                temp = csv_path.with_suffix(".csv.tmp")
                pd.DataFrame(rows).to_csv(temp, index=False)
                os.replace(temp, csv_path)
                print("STRENGTHENED_SESSION_COMPLETE", condition, seed, flush=True)
            except Exception as exc:
                failures.append(f"{condition}_s{seed}")
                append_event(args.out / "failures.jsonl", {"session": failures[-1],
                              "error": repr(exc), "traceback": traceback.format_exc(), "status": "failed"})
                print("STRENGTHENED_SESSION_FAILED", failures[-1], repr(exc), flush=True)
                runner._free_cuda()
    print("STRENGTHENED_DONE", "failed=" + ",".join(failures), flush=True)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
