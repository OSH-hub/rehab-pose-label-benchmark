"""Headless simulated-correction AL experiment: A vs B vs ref across seeds.

The experiment compares three feedback policies under the same acquisition rule
and cumulative label budget. Because each arm retrains its own model, later pool
scores and selected samples may differ between conditions.

Conditions (contract docs/CLINICIAN_LOOP_CONTRACT.md §0):
  A   — AL + pseudo-label : selected frames labeled with the current model's
        (noisy) predictions. No human.
  B   — AL + correction   : SimulatedClinician rejects frames outside labeled
        repetitions or carrying the camera-specific extra-person flag and
        replaces every accepted pseudo-label with GT-derived keypoints
        (optionally perturbed by sigma noise).
  ref — AL + full GT       : every selected frame labeled with exact GT
        (empirical comparator, not a mathematical upper bound).

Protocol (mirrors scripts/run_paper_experiments.py for comparability):
  Round 0 = shared random init set labeled with GT, IDENTICAL across all
            conditions for a given seed (fair start; also yields the first real
            26-kp model so round 1+ scoring never loads the default 17-kp model).
  Round r>=1 = the AL strategy (default ``temporal``) scores the pool with the
            condition's round model and acquires the next
            budget increment; the condition's decision policy then labels them.

Budgets are cumulative label fractions of the train pool; per round we set
``budget_per_round`` so the labeled set lands exactly on the next budget target.
AUC / significance are computed by the lead's analysis step from the master CSV.

Completed sessions are reusable only with the same input manifest and valid
metrics/counts. Partial sessions require --restart-incomplete (archive then
restart); different configurations and legacy outputs require a new --out.

Usage::
    python scripts/run_clinician_experiment.py --smoke
    python scripts/run_clinician_experiment.py            # full A/B/ref x 3 seeds
    python scripts/run_clinician_experiment.py --sigmas 0 5 15   # clinician-quality ablation
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

# scripts/ + src/ on path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "src"))

from al_budget_curve import (  # noqa: E402
    DATASET,
    MANIFEST,
    _build_pool,
    _free_cuda,
    _recording_meta,
)
from rehabal_pose.al import get_strategy  # noqa: E402
from rehabal_pose.data import ManifestBuilder  # noqa: E402
from rehabal_pose.review.clinician_loop import (  # noqa: E402
    ClinicianLoop,
    Decision,
    SessionState,
    SimulatedClinician,
    _get_image_size,
    load_seg_table,
    read_gt_label,
)
from rehabal_pose.utils.io import ensure_dir  # noqa: E402
from rehabal_pose.utils.logging import get_logger  # noqa: E402

_LOG = get_logger("clinician_exp")

IMAGES_TRAIN = DATASET / "images" / "train"
SEG_CSV = Path("data/rehab24_6_raw/Segmentation.csv")

DEFAULT_BUDGETS = [0.05, 0.10, 0.20, 0.40]
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_CONDITIONS = ["A", "B", "ref"]
CONDITIONS = DEFAULT_CONDITIONS + ["A_filter", "B_reject_only", "ref_filter",
                                   "B_correct_only", "A_conf"]


class SessionReuseError(RuntimeError):
    """An existing output cannot safely be reused or continued."""


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _file_identity(path):
    """Hash bytes rather than timestamps; missing inputs are explicit."""
    path = Path(path)
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        sha256 = digest.hexdigest()
    else:
        sha256 = None
    return {"path": str(path.resolve()), "sha256": sha256}


def _input_identity(pool_stems, seg_table):
    """Once per CLI invocation: train/val bytes, GT, metadata and segmentation."""
    paths = {MANIFEST, SEG_CSV}
    paths.update(IMAGES_TRAIN / f"{stem}.jpg" for stem in pool_stems)
    paths.update(DATASET / "labels" / "train" / f"{stem}.txt" for stem in pool_stems)
    for folder in (DATASET / "images" / "val", DATASET / "labels" / "val"):
        paths.update(p for p in folder.rglob("*") if p.is_file())
    files = [_file_identity(p) for p in sorted(paths, key=lambda p: str(p.resolve()))]
    return {
        "files_sha256": _json_hash(files), "file_count": len(files),
        "seg_table_sha256": hashlib.sha256(
            ("null" if seg_table is None else seg_table.to_json(orient="split"))
            .encode("utf-8")).hexdigest(),
    }


def _source_identity():
    # Algorithm changes invalidate reuse. Tests and documentation do not.
    paths = [_HERE / "run_clinician_experiment.py", _HERE / "al_budget_curve.py"]
    paths.extend(sorted((_HERE.parent / "src" / "rehabal_pose").rglob("*.py")))
    return _json_hash([{"name": p.relative_to(_HERE.parent).as_posix(),
                        "sha256": _file_identity(p)["sha256"]} for p in paths])


def _atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _budget_targets(budgets, pool_stems, N):
    if N != len(pool_stems) or N < 1 or len(set(pool_stems)) != N:
        raise ValueError("Pool size must equal the number of unique nonempty pool stems")
    if not budgets or any(not np.isfinite(b) or not 0 < b <= 1 for b in budgets):
        raise ValueError("Budgets must be finite fractions in (0, 1]")
    targets = [max(1, int(round(b * N))) for b in budgets]
    if any(b <= a for a, b in zip(targets, targets[1:])):
        raise ValueError("Budgets must produce strictly increasing label counts")
    return targets


def _validate_curve(df, targets):
    required = {"round", "n_labeled", "map50", "map50_95", "n_accepted",
                "n_rejected", "n_corrected", "clinician"}
    if len(df) != len(targets) or not required.issubset(df.columns):
        raise ValueError("Curve must contain exactly one complete row per budget")
    for column in required - {"clinician"}:
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Curve contains nonfinite or missing {column}")
        if column in ("map50", "map50_95"):
            if ((values < 0) | (values > 1)).any():
                raise ValueError(f"Curve {column} must lie in [0, 1]")
        elif (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError(f"Curve {column} must be nonnegative integers")
    if list(df["round"]) != list(range(len(targets))):
        raise ValueError("Curve round indices are duplicated, missing or out of order")
    if list(df["n_labeled"]) != targets:
        raise ValueError("Curve label counts do not match exact budget targets")
    increments = np.diff([0] + targets)
    if not np.array_equal(df["n_accepted"].to_numpy(), increments):
        raise ValueError("Curve accepted counts do not match budget increments")
    if (df["n_corrected"] > df["n_accepted"]).any():
        raise ValueError("Curve corrected count exceeds accepted count")


def _validate_completed(sess_dir, fingerprint, targets, pool_stems):
    done = pd.read_csv(sess_dir / "curve.csv")
    _validate_curve(done, targets)
    state = ClinicianLoop.load_state(sess_dir)
    if (not state.current_weights or not Path(state.current_weights).is_file()
            or Path(state.current_weights).stat().st_size == 0):
        raise ValueError("Completed session has missing or empty weights")
    completion = json.loads((sess_dir / "completion.json").read_text(encoding="utf-8"))
    if not isinstance(completion, dict):
        raise ValueError("Completion marker must be a dictionary")
    checkpoints = completion.get("round_checkpoints", [])
    if (not isinstance(checkpoints, list) or len(checkpoints) != len(targets)
            or not all(isinstance(item, dict) for item in checkpoints)):
        raise ValueError("Completion must retain a checkpoint for every budget")
    for index, checkpoint in enumerate(checkpoints):
        path = Path(checkpoint["path"])
        if (checkpoint.get("round") != index or not path.is_file()
                or path.stat().st_size == 0
                or checkpoint != {"round": index, **_file_identity(path)}):
            raise ValueError(f"Round {index} checkpoint is missing or modified")
    if checkpoints[-1]["path"] != str(Path(state.current_weights).resolve()):
        raise ValueError("Final checkpoint disagrees with the saved state")
    if completion != {"fingerprint": fingerprint,
                      "curve_sha256": _file_identity(sess_dir / "curve.csv")["sha256"],
                      "state_sha256": _file_identity(sess_dir / "state.json")["sha256"],
                      "weights_sha256": _file_identity(state.current_weights)["sha256"],
                      "round_checkpoints": checkpoints}:
        raise ValueError("Completion fingerprint or result integrity mismatch")
    if (state.status != "done" or state.round != len(targets)
            or len(state.history) != len(targets) or not state.current_weights
            or state.failure is not None or len(state.labeled) != targets[-1]
            or len(set(state.labeled)) != targets[-1]
            or set(state.labels) != set(state.labeled)
            or not set(state.labeled).issubset(pool_stems)):
        raise ValueError("Persisted state is not a completed session at the requested budget")
    for rec, row in zip(state.history, done.to_dict("records")):
        if (rec.round != row["round"] or rec.n_labeled != row["n_labeled"]
                or rec.n_accepted != row["n_accepted"]
                or not np.isclose(rec.pose_map50, row["map50"], rtol=1e-12, atol=1e-15)
                or not np.isclose(rec.pose_map50_95, row["map50_95"], rtol=1e-12, atol=1e-15)):
            raise ValueError("Persisted history disagrees with the curve")
    return done


def _prepare_session(sess_dir, manifest, targets, pool_stems, restart_incomplete):
    if sess_dir.exists() and any(sess_dir.iterdir()):
        try:
            previous = json.loads((sess_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SessionReuseError(
                f"{sess_dir}: legacy or unreadable manifest; preserved. Use a new --out."
            ) from exc
        if previous != manifest:
            raise SessionReuseError(
                f"{sess_dir}: configuration/data/model fingerprint differs; preserved. "
                "Use a new --out for different settings (restart cannot override this)."
            )
        try:
            return _validate_completed(sess_dir, manifest["fingerprint"], targets, pool_stems)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if not restart_incomplete:
                raise SessionReuseError(
                    f"{sess_dir}: incomplete/invalid session ({exc}); preserved. "
                    "Use --restart-incomplete to archive it and start from round 0, "
                    "or choose a new --out."
                ) from exc
            archive = sess_dir.parent / "_archive"
            archive.mkdir(exist_ok=True)
            destination = archive / f"{sess_dir.name}_{uuid.uuid4().hex}"
            root = sess_dir.parent.resolve()
            if (not sess_dir.resolve().is_relative_to(root)
                    or not destination.resolve().is_relative_to(root)
                    or sess_dir.resolve() == root):
                raise SessionReuseError("Archive paths must stay inside the output root")
            sess_dir.rename(destination)
            _LOG.warning("Archived incomplete session to %s", destination)
    sess_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(sess_dir / "manifest.json", manifest)
    return None


# --------------------------------------------------------------------------- #
# Real callables (no fastapi). score_fn HONORS the chosen AL strategy.         #
# --------------------------------------------------------------------------- #

def _resolve_checkpoint(weights, download_fn=None):
    """Resolve/download once before fingerprinting, without loading a model."""
    path = Path(weights)
    if not path.is_file():
        if download_fn is None:
            # Same resolver used by Ultralytics; lazy for mock-only imports.
            from ultralytics.utils.downloads import attempt_download_asset
            download_fn = attempt_download_asset
        _LOG.info("Resolving base checkpoint before fingerprinting: %s", weights)
        path = Path(download_fn(str(weights)))
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Base checkpoint could not be resolved: {weights}")
    return str(path.resolve())


def _make_callables(pool_images: List[str], rec_meta: dict, strategy_name: str,
                    epochs: int, imgsz: int, batch: int, epochs_round0: int,
                    weights: Optional[str] = None):
    """Build (infer_fn, score_fn, train_eval_fn, gt_lookup). All weights passed
    to these are REAL 26-kp weights (round >= 1); round 0 never calls them."""

    from rehabal_pose.models.yolo_pose import _DEFAULT_CONFIG
    # Pin starting bytes before writing any manifest: a first-run alias download
    # and a repeat run have the same identity, but changed local bytes do not.
    weights = _resolve_checkpoint(weights or _DEFAULT_CONFIG["weights"])

    # Cache the full-pool scoring by weights. The model weights are FIXED within
    # an AL round, so run_auto_round's top-up calls propose() repeatedly with the
    # same weights — re-scoring all 8367 images each time is both wasteful and
    # the cause of the GPU OOM (each call reloaded a model + predict + embeddings
    # before the previous allocation was fully reclaimed). Memoizing by weights
    # collapses a round's repeated scoring to a single pass.
    _score_cache = {"weights": "\x00__never__", "result": None}

    def score_fn(weights):
        if not weights:
            # Round-0 training failed (no best.pt) -> we have no real 26-kp model
            # to score the pool with. Refuse loudly so the session aborts cleanly
            # instead of YOLO(None) crashing the whole experiment.
            raise RuntimeError(
                "score_fn called with empty weights — round-0 training likely "
                "failed (CUDA OOM?); aborting this session.")
        if weights == _score_cache["weights"] and _score_cache["result"] is not None:
            return _score_cache["result"]
        from rehabal_pose.models.yolo_pose import YoloPoseModel
        model = YoloPoseModel().load(weights)
        pool = _build_pool(pool_images, model, rec_meta)
        del model
        _free_cuda()
        strat = get_strategy(strategy_name)
        scores = np.asarray(strat.score(pool), dtype=np.float64)
        # propose() sums the term arrays -> composite == the strategy score.
        result = (pool, {"score": scores})
        _score_cache["weights"] = weights
        _score_cache["result"] = result
        return result

    def infer_fn(weights, stems):
        from rehabal_pose.models.yolo_pose import YoloPoseModel
        # OOM hardening: this fires inside run_auto_round's top-up loop right
        # after a train/score pass, on up to ~budget*2 stems at once (round 1 can
        # be ~800). Reclaim residual VRAM BEFORE allocating, predict in chunks so
        # peak memory is bounded, and retry once with a smaller chunk on OOM.
        _free_cuda()
        paths = [str(IMAGES_TRAIN / f"{s}.jpg") for s in stems]

        def _predict_chunked(chunk):
            model = YoloPoseModel().load(weights)
            collected = []
            try:
                for s in range(0, len(paths), chunk):
                    collected.extend(model.predict(paths[s:s + chunk]))
            finally:
                del model
                _free_cuda()
            return collected

        try:
            preds = _predict_chunked(256)
        except RuntimeError as exc:  # noqa: BLE001
            if "out of memory" not in str(exc).lower():
                raise
            _LOG.warning("infer_fn OOM on %d stems; retry with chunk=64", len(paths))
            _free_cuda()
            preds = _predict_chunked(64)

        out = {}
        for p in preds:
            if p.keypoints is not None and np.asarray(p.keypoints).shape == (26, 3):
                out[p.sample_id] = np.asarray(p.keypoints, dtype=np.float64).tolist()
        return out

    def train_eval_fn(labeled_stems, labels, out_dir, seed):
        import os
        import shutil
        from al_budget_curve import _write_round_yaml
        from rehabal_pose.models.yolo_pose import YoloPoseModel
        from rehabal_pose.review.clinician_loop import write_labels_dir
        out_path = Path(out_dir)
        # Option A (docs/ROUND0_INVESTIGATION.md): round 0 cold-starts the
        # 17->26-kp head on the smallest data slice (~418 frames); at 10 ep
        # (~260 steps) this is below the convergence threshold and lands at
        # map=0.0 for some seeds. Give round 0 more epochs (round dir is
        # "round_0", set in clinician_loop.commit_and_train).
        actual_epochs = epochs_round0 if out_path.name == "round_0" else epochs
        img_dir = out_path / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        image_sizes = {
            s: _get_image_size(IMAGES_TRAIN / f"{s}.jpg")
            for s in labeled_stems if (IMAGES_TRAIN / f"{s}.jpg").exists()
        }
        # CRITICAL: write OUR (pseudo/corrected/GT) labels to out_path/labels/train
        # AND materialize the images next to them, so YOLO's images->labels path
        # resolution reads THESE labels — not the prebuilt GT under the canonical
        # dataset dir. Pointing train.txt at the canonical images would silently
        # train every condition on GT (A=B=ref collapse).
        write_labels_dir({k: labels[k] for k in labeled_stems if k in labels},
                         out_path, image_sizes=image_sizes)
        linked = []
        for s in labeled_stems:
            src = IMAGES_TRAIN / f"{s}.jpg"
            if not src.exists():
                continue
            dst = img_dir / f"{s}.jpg"
            if not dst.exists():
                try:
                    os.link(src, dst)  # hardlink (same drive, instant)
                except OSError:
                    shutil.copyfile(src, dst)
            linked.append(dst)
        train_list = out_path / "train.txt"
        train_list.write_text(
            "\n".join(str(p.resolve()) for p in linked) + "\n", encoding="utf-8",
        )
        data_yaml = _write_round_yaml(out_path, train_list)
        train_cfg = {"epochs": actual_epochs, "imgsz": imgsz, "batch": batch,
                     "seed": seed, "project": str(out_path), "name": "train"}
        # Backbone passthrough: when --weights is given, every round cold-starts
        # from THAT checkpoint (e.g. yolo11s/m-pose.pt) instead of the default
        # yolo11n-pose.pt. Pure config passthrough — ultralytics auto-downloads
        # a known alias on first use. None keeps the default backbone.
        if weights:
            train_cfg["weights"] = weights
        model = YoloPoseModel(train_cfg)
        res = model.train(str(data_yaml), output_dir=str(out_path / "run"))
        del model
        _free_cuda()
        return res

    def gt_lookup(stem):
        img = IMAGES_TRAIN / f"{stem}.jpg"
        w, h = _get_image_size(img)
        return read_gt_label(stem, DATASET, image_w=w, image_h=h)

    effective = dict(_DEFAULT_CONFIG)
    effective.update(epochs=epochs, imgsz=imgsz, batch=batch, epochs_round0=epochs_round0)
    effective["weights"] = weights or effective["weights"]
    effective["checkpoint_identity"] = _file_identity(effective["weights"])
    from importlib.metadata import PackageNotFoundError, version
    effective["packages"] = {}
    for package in ("ultralytics", "torch", "numpy", "pandas"):
        try:
            effective["packages"][package] = version(package)
        except PackageNotFoundError:
            effective["packages"][package] = None
    train_eval_fn.experiment_config = effective
    return infer_fn, score_fn, train_eval_fn, gt_lookup


def _pred_confidence(cand) -> float:
    """Frame-level pseudo-label confidence for I3 (A_conf).

    The candidate's ``pred_keypoints`` are ``(26,3)`` ``[x, y, score]`` from the
    round's YOLO model (see ``infer_fn``). We use the mean per-keypoint score as
    the model-reported confidence for FixMatch-style thresholding. Robust
    to the ``[[0,0,0]]*26`` fallback (→ 0.0) so a missing pseudo-label is never
    accepted above a positive threshold.
    """
    kp = np.asarray(cand.pred_keypoints, dtype=np.float64)
    if kp.ndim != 2 or kp.shape[0] == 0 or kp.shape[1] < 3:
        return 0.0
    return float(np.mean(kp[:, 2]))


def _decision_fn_for(condition: str, oracle, gt_lookup,
                     *, conf_threshold: Optional[float] = None) -> Callable:
    if condition == "A":
        return lambda c: Decision(sample_id=c.sample_id, action="accept")
    if condition == "ref":
        def ref_decide(c):
            gt = gt_lookup(c.sample_id)
            if gt is None:
                return Decision(sample_id=c.sample_id, action="accept")
            return Decision(sample_id=c.sample_id, action="correct",
                            keypoints=np.asarray(gt, dtype=np.float64).tolist())
        return ref_decide

    # ------------------------------------------------------------------ #
    # Additive ablation conditions. These do NOT alter A/B/ref.           #
    #                                                                     #
    # I2 (filter × correct 2×2): decompose B's gain into "label
    #     correction" vs "frame filtering". The seg-based rejection is
    #     byte-identical to SimulatedClinician.decide (reject when
    #     NOT in_labeled_rep OR occluded); the clinical_flags are already
    #     computed for EVERY condition inside ClinicianLoop.propose() via
    #     _seg_flags, so the same signal is injected into the A/ref family.
    # ------------------------------------------------------------------ #
    def _to_gt(c):
        gt = gt_lookup(c.sample_id)
        if gt is None:
            return Decision(sample_id=c.sample_id, action="accept")
        return Decision(sample_id=c.sample_id, action="correct",
                        keypoints=np.asarray(gt, dtype=np.float64).tolist())

    def _seg_reject(c) -> bool:
        # Mirrors SimulatedClinician.decide (reject_occluded=True default).
        flags = c.clinical_flags or {}
        in_rep = flags.get("in_labeled_rep", True)
        occluded = flags.get("occluded", False)
        return (not in_rep) or occluded

    # A_filter / B_reject_only: keep pseudo-label on accepted frames, apply
    # B's identical seg-based rejection. (Two names for the same cell of the
    # 2×2 — "A + filter" and "B minus correction" — kept distinct so both the
    # A-anchored and B-anchored decompositions read cleanly in the results.)
    if condition in ("A_filter", "B_reject_only"):
        def decide_reject_pseudo(c):
            if _seg_reject(c):
                return Decision(sample_id=c.sample_id, action="reject")
            return Decision(sample_id=c.sample_id, action="accept")
        return decide_reject_pseudo

    # ref_filter: full GT on accepted frames + B's identical seg rejection.
    if condition == "ref_filter":
        def decide_reject_gt(c):
            if _seg_reject(c):
                return Decision(sample_id=c.sample_id, action="reject")
            return _to_gt(c)
        return decide_reject_gt

    # B_correct_only: no rejection; correct every accepted frame to GT
    # (isolates the label-correction effect; == ref's labelling, no filter).
    if condition == "B_correct_only":
        return _to_gt

    # I3 confidence-gated pseudo-label (FixMatch-style baseline):
    #     accept the pseudo-label only when the model's own confidence >= τ,
    #     otherwise reject the frame. τ comes from --conf-threshold; when it is
    #     None (default) τ=0 so every frame passes → A_conf reduces to plain A
    #     (keeps the default-arg behaviour inert for the existing grid).
    if condition == "A_conf":
        tau = 0.0 if conf_threshold is None else float(conf_threshold)

        def decide_conf(c):
            if _pred_confidence(c) >= tau:
                return Decision(sample_id=c.sample_id, action="accept")
            return Decision(sample_id=c.sample_id, action="reject")
        return decide_conf

    # condition == "B"
    return oracle.decide


# --------------------------------------------------------------------------- #
# One (condition, seed, sigma) session                                         #
# --------------------------------------------------------------------------- #

def _run_session(condition, seed, sigma, budgets, pool_stems, N,
                 callables, seg_table, out_root, strategy_name,
                 conf_threshold=None, verify_oks_threshold=None, *,
                 run_config=None, input_identity=None, restart_incomplete=False) -> List[dict]:
    infer_fn, score_fn, train_eval_fn, gt_lookup = callables
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown experiment condition: {condition!r}")
    targets = _budget_targets(budgets, pool_stems, N)
    tag = f"{condition}_s{seed}" + (f"_sig{sigma:g}" if condition == "B" else "")
    out_root = Path(out_root)
    sess_dir = out_root / tag
    sess_curve = sess_dir / "curve.csv"
    config = dict(run_config if run_config is not None else
                  getattr(train_eval_fn, "experiment_config", {}))
    identity = {
        "schema_version": 1, "condition": condition, "seed": int(seed),
        "sigma": float(sigma) if condition == "B" else None,
        "budgets": list(budgets), "targets": targets, "strategy": strategy_name,
        "conf_threshold": conf_threshold, "verify_oks_threshold": verify_oks_threshold,
        "pool_stems_sha256": _json_hash(list(pool_stems)), "pool_size": N,
        "training": config,
        "inputs": input_identity if input_identity is not None else
                  _input_identity(pool_stems, seg_table),
        "source_sha256": _source_identity(),
    }
    manifest = {"fingerprint": _json_hash(identity), "identity": identity}
    done = _prepare_session(sess_dir, manifest, targets, pool_stems, restart_incomplete)
    if done is not None:
        _LOG.info("[%s] SKIP (manifest and %d completed rounds verified)", tag, len(done))
        return _rows_from_session(done, condition, seed, sigma, budgets)

    state = SessionState(session_id=tag, condition=condition, seed=seed,
                         strategy=strategy_name, budget_per_round=targets[0],
                         round=0, labeled=[], labels={}, history=[], status="active")

    # image size for the simulated clinician's keypoint clipping
    sample_img = IMAGES_TRAIN / f"{pool_stems[0]}.jpg"
    iw, ih = _get_image_size(sample_img)
    oracle = None
    if condition == "B":
        # I1 verify wiring: forward --verify-oks-threshold to the clinician ONLY
        # when explicitly set. When None (default) the kwarg is not passed at all,
        # so condition B's construction is byte-identical to the pre-change call
        # (and stays compatible with a clinician_loop that predates the param).
        oracle_kwargs = {}
        if verify_oks_threshold is not None:
            oracle_kwargs["verify_oks_threshold"] = verify_oks_threshold
        oracle = SimulatedClinician(gt_lookup, seg_table=seg_table,
                                    correct_sigma=float(sigma), seed=seed,
                                    image_w=iw, image_h=ih, **oracle_kwargs)

    loop = ClinicianLoop(
        state, infer_fn=infer_fn, score_fn=score_fn, train_eval_fn=train_eval_fn,
        gt_lookup=gt_lookup, oracle=oracle, run_mode="auto",
        output_root=str(out_root), image_dir=str(IMAGES_TRAIN),
        seg_table=seg_table, train_image_stems=set(pool_stems),
    )

    # --- Round 0: shared random GT init (identical across conditions/seed) ----
    rng = np.random.default_rng(1000 + seed)
    init_idx = rng.choice(N, size=targets[0], replace=False)
    init_stems = [pool_stems[i] for i in init_idx]
    decisions = []
    for sid in init_stems:
        gt = gt_lookup(sid)
        if gt is not None:
            decisions.append(Decision(sample_id=sid, action="correct",
                                      keypoints=np.asarray(gt, dtype=np.float64).tolist()))
    loop.state.budget_per_round = targets[0]
    loop.apply_decisions(decisions)
    if len(decisions) != targets[0]:
        raise RuntimeError("Round 0 has missing GT; cannot meet the exact initial budget")
    _LOG.info("[%s] round 0: random GT init on %d frames", tag, len(decisions))
    loop.commit_and_train()
    checkpoints = [{"round": 0, **_file_identity(loop.state.current_weights)}]

    # --- Round r >= 1: AL acquisition + condition policy ----------------------
    decision_fn = _decision_fn_for(condition, oracle, gt_lookup,
                                   conf_threshold=conf_threshold)
    for r in range(1, len(budgets)):
        loop.state.budget_per_round = targets[r] - len(loop.state.labeled)
        if loop.state.budget_per_round <= 0:
            continue
        _LOG.info("[%s] round %d: acquire %d -> budget %.2f",
                  tag, r, loop.state.budget_per_round, budgets[r])
        loop.run_auto_round(decision_fn)
        checkpoints.append({"round": r, **_file_identity(loop.state.current_weights)})
        if len(loop.state.labeled) != targets[r]:
            raise RuntimeError(f"Round {r} did not reach exact label budget {targets[r]}")

    done = pd.read_csv(sess_curve)
    _validate_curve(done, targets)
    loop.state.status = "done"
    loop._save_state()
    _atomic_json(sess_dir / "completion.json", {
        "fingerprint": manifest["fingerprint"],
        "curve_sha256": _file_identity(sess_curve)["sha256"],
        "state_sha256": _file_identity(sess_dir / "state.json")["sha256"],
        "weights_sha256": _file_identity(loop.state.current_weights)["sha256"],
        "round_checkpoints": checkpoints,
    })
    _validate_completed(sess_dir, manifest["fingerprint"], targets, pool_stems)
    return _rows_from_session(done, condition, seed, sigma, budgets)


def _rows_from_session(df, condition, seed, sigma, budgets) -> List[dict]:
    if len(df) != len(budgets):
        raise ValueError("Cannot assign budgets to an incomplete or overlong curve")
    rows = []
    for i, rec in df.reset_index(drop=True).iterrows():
        b = budgets[i]
        rows.append({
            "condition": condition, "seed": seed,
            "sigma": sigma if condition == "B" else "",
            "budget": b, "n_labeled": int(rec["n_labeled"]),
            "map50": rec.get("map50"), "map50_95": rec.get("map50_95"),
            "n_proposed": int(rec.get("n_proposed", 0)) if "n_proposed" in rec else None,
            "n_accepted": int(rec["n_accepted"]), "n_rejected": int(rec["n_rejected"]),
            "n_corrected": int(rec["n_corrected"]), "clinician": rec["clinician"],
        })
    return rows


# --------------------------------------------------------------------------- #
# main                                                                          #
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=DEFAULT_CONDITIONS)
    ap.add_argument("--budgets", nargs="+", type=float, default=DEFAULT_BUDGETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--sigmas", nargs="+", type=float, default=[0.0],
                    help="clinician correction noise sigma (px) for condition B")
    ap.add_argument("--strategy", default="temporal")
    ap.add_argument("--epochs", type=int, default=30,
                    help="epochs for rounds 1..N (default 30 = uniform u30 "
                         "manuscript schedule; older Option-A used 10)")
    ap.add_argument("--epochs-round0", type=int, default=30,
                    help="epochs for round 0 only; default matches --epochs "
                         "(uniform 30ep). The older Option-A schedule used "
                         "round0=30 / later=10 (see docs/ROUND0_INVESTIGATION.md)")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--weights", default=None,
                    help="backbone checkpoint every round cold-starts from "
                         "(e.g. yolo11s-pose.pt / yolo11m-pose.pt). Default "
                         "None = yolo11n-pose.pt (yolo_pose._DEFAULT_CONFIG).")
    ap.add_argument("--conf-threshold", type=float, default=None,
                    help="I3: pseudo-label confidence threshold τ for condition "
                         "A_conf (mean per-keypoint pred score >= τ is accepted, "
                         "else rejected — FixMatch-style self-training). Default "
                         "None leaves every existing condition unchanged (and "
                         "makes A_conf reduce to A).")
    ap.add_argument("--verify-oks-threshold", type=float, default=None,
                    help="I1: forwarded verbatim to SimulatedClinician as "
                         "verify_oks_threshold= (condition B only). Default None "
                         "is NOT forwarded, preserving the clinician's existing "
                         "always-correct behaviour byte-for-byte.")
    ap.add_argument("--out", default="outputs/experiments/clinician")
    ap.add_argument("--restart-incomplete", action="store_true",
                    help="archive a same-configuration incomplete/invalid session, "
                         "then restart at round 0; legacy/different settings require a new --out")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny: A/B/ref, budgets 0.05/0.10, 1 seed, 1 epoch, imgsz 320")
    args = ap.parse_args(argv)

    if args.smoke:
        args.conditions = ["A", "B", "ref"]
        args.budgets = [0.05, 0.10]
        args.seeds = [0]
        args.sigmas = [0.0]
        args.epochs = 1
        args.epochs_round0 = 1
        args.imgsz = 320
        _LOG.info("SMOKE mode")

    for name in ("conditions", "seeds", "sigmas"):
        values = getattr(args, name)
        if len(values) != len(set(values)):
            ap.error(f"--{name} must not contain duplicate values")
    if any(not np.isfinite(value) or value < 0 for value in args.sigmas):
        ap.error("--sigmas must contain finite nonnegative values")

    pool_images = sorted(glob.glob(str(IMAGES_TRAIN / "*.jpg")))
    if not pool_images:
        _LOG.error("No train images under %s; build the dataset first.", IMAGES_TRAIN)
        return 1
    pool_stems = [Path(p).stem for p in pool_images]
    N = len(pool_stems)
    try:
        _budget_targets(args.budgets, pool_stems, N)
    except ValueError as exc:
        ap.error(str(exc))
    _LOG.info("Pool %d train imgs; budgets %s; conditions %s; seeds %s; sigmas %s",
              N, args.budgets, args.conditions, args.seeds, args.sigmas)

    manifest_df = ManifestBuilder.load(MANIFEST)
    rec_meta = _recording_meta(manifest_df)
    seg_table = load_seg_table(SEG_CSV if SEG_CSV.exists() else None)
    if seg_table is None or len(seg_table) == 0:
        _LOG.warning("Segmentation table empty (%s) — condition B will not reject; "
                     "B collapses toward ref. Check the path.", SEG_CSV)

    callables = _make_callables(pool_images, rec_meta, args.strategy,
                                args.epochs, args.imgsz, args.batch,
                                args.epochs_round0, args.weights)
    out_root = ensure_dir(Path(args.out))
    master_csv = out_root / "clinician_curve.csv"
    _LOG.info("Fingerprinting train/validation images, labels and metadata once...")
    input_identity = _input_identity(pool_stems, seg_table)
    run_config = dict(getattr(callables[2], "experiment_config", {}))
    # Keep CLI conditions explicit even for injected callables in mock runs.
    run_config.update(epochs=args.epochs, epochs_round0=args.epochs_round0,
                      imgsz=args.imgsz, batch=args.batch, weights=args.weights or "yolo11n-pose.pt")
    attempt_dir = ensure_dir(out_root / "attempts" / uuid.uuid4().hex)
    attempt_csv = attempt_dir / "clinician_curve.csv"

    all_rows: List[dict] = []
    failed: List[str] = []
    for seed in args.seeds:
        for condition in args.conditions:
            sigmas = args.sigmas if condition == "B" else [0.0]
            for sigma in sigmas:
                tag = f"{condition}_s{seed}" + (f"_sig{sigma:g}" if condition == "B" else "")
                try:
                    rows = _run_session(condition, seed, sigma, args.budgets,
                                        pool_stems, N, callables, seg_table,
                                        out_root, args.strategy,
                                        conf_threshold=args.conf_threshold,
                                        verify_oks_threshold=args.verify_oks_threshold,
                                        run_config=run_config, input_identity=input_identity,
                                        restart_incomplete=args.restart_incomplete)
                except Exception as exc:  # noqa: BLE001
                    _LOG.error("Session %s FAILED: %s — existing outputs preserved.", tag, exc)
                    with (attempt_dir / "failed.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps({"session": tag, "error": str(exc),
                                                 "error_type": type(exc).__name__,
                                                 "time": datetime.now(timezone.utc).isoformat()}) + "\n")
                    failed.append(tag)
                    _free_cuda()
                    continue
                all_rows.extend(rows)
                pd.DataFrame(all_rows).to_csv(attempt_csv, index=False)
                _LOG.info("[%s s%d sig%g] done; master rows=%d",
                          condition, seed, sigma, len(all_rows))

    pd.DataFrame(all_rows).to_csv(attempt_csv, index=False)
    if failed:
        _LOG.warning("%d session(s) FAILED: %s. Previous master CSV preserved. "
                     "See attempts/ failure log; use --restart-incomplete for a "
                     "same-configuration restart, or a new --out for changed settings.",
                     len(failed), ", ".join(failed))
    else:
        # Publish only a fully successful requested grid. Keep any previous
        # master (including a broader legacy grid) before atomic replacement.
        if master_csv.exists():
            shutil.copy2(master_csv, attempt_dir / "previous_clinician_curve.csv")
        temp_master = master_csv.with_suffix(".csv.tmp")
        shutil.copyfile(attempt_csv, temp_master)
        os.replace(temp_master, master_csv)
        _LOG.info("Saved %s (%d rows)", master_csv, len(all_rows))
    _atomic_json(attempt_dir / "status.json", {"status": "failed" if failed else "done",
                                               "failed": failed, "rows": len(all_rows)})
    print("CLINICIAN_EXP_FAILED" if failed else "CLINICIAN_EXP_DONE", out_root,
          "failed=" + ",".join(failed) if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
