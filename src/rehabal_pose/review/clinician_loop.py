from __future__ import annotations

import csv
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "Candidate",
    "Decision",
    "RoundRecord",
    "SessionState",
    "CommitResult",
    "RoundTrainingError",
    "ClinicianLoop",
    "SimulatedClinician",
    "read_gt_label",
    "write_labels_dir",
    "load_seg_table",
    "_seg_flags",
    "_get_image_size",
    "_oks_similarity",
]

_LOG = logging.getLogger(__name__)


class RoundTrainingError(RuntimeError):
    pass


@dataclass
class Candidate:
    sample_id: str
    image_path: str
    pred_keypoints: list
    term_scores: dict
    clinical_flags: dict


@dataclass
class Decision:
    sample_id: str
    action: str
    keypoints: Optional[list] = None


@dataclass
class RoundRecord:
    round: int
    n_labeled: int
    pose_map50: Optional[float]
    pose_map50_95: Optional[float]
    n_proposed: int
    n_accepted: int
    n_rejected: int
    n_corrected: int
    clinician: str


@dataclass
class SessionState:
    session_id: str
    condition: str
    seed: int
    strategy: str
    budget_per_round: int
    round: int
    labeled: List[str]
    labels: Dict[str, List[List[float]]]
    history: List[RoundRecord]
    status: str
    current_weights: Optional[str] = None
    failure: Optional[dict] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["history"] = [asdict(r) for r in self.history]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SessionState":
        history = [RoundRecord(**r) for r in d.get("history", [])]
        return cls(
            session_id=d["session_id"],
            condition=d["condition"],
            seed=d["seed"],
            strategy=d.get("strategy", "temporal"),
            budget_per_round=d["budget_per_round"],
            round=d["round"],
            labeled=d.get("labeled", []),
            labels=d.get("labels", {}),
            history=history,
            status=d.get("status", "active"),
            current_weights=d.get("current_weights"),
            failure=d.get("failure"),
        )


@dataclass
class CommitResult:
    accepted: List[str]
    rejected: List[str]
    corrected: List[str]
    need_more: bool


def read_gt_label(
    stem: str,
    labels_dir: "str | Path",
    image_w: int = 640,
    image_h: int = 640,
    *,
    split: str = "train",
) -> Optional[np.ndarray]:
    if split != "train":
        raise ValueError(
            f"read_gt_label: split must be 'train' (got {split!r}). "
            "Val labels must never be used as GT for training."
        )

    labels_dir = Path(labels_dir)
    candidates = [
        labels_dir / "labels" / "train" / f"{stem}.txt",
        labels_dir / "train" / f"{stem}.txt",
        labels_dir / f"{stem}.txt",
    ]
    txt_path: Optional[Path] = None
    for p in candidates:
        if p.exists():
            txt_path = p
            break
    if txt_path is None:
        return None

    with txt_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 5:
                continue
            n_kp = (len(tokens) - 5) // 3
            if n_kp == 0:
                continue
            kp = np.zeros((n_kp, 3), dtype=np.float64)
            for k in range(n_kp):
                px = float(tokens[5 + k * 3]) * image_w
                py = float(tokens[5 + k * 3 + 1]) * image_h
                pv = float(tokens[5 + k * 3 + 2])
                kp[k] = [px, py, pv]
            return kp
    return None


def _get_image_size(image_path: "str | Path") -> Tuple[int, int]:
    p = Path(image_path)
    if not p.exists():
        return 640, 640
    try:
        data = p.read_bytes()
        if data[:2] == b'\xff\xd8':
            i = 2
            while i < len(data) - 8:
                marker = int.from_bytes(data[i:i + 2], "big")
                seg_len = int.from_bytes(data[i + 2:i + 4], "big")
                if marker in (0xFFC0, 0xFFC1, 0xFFC2, 0xFFC3,
                              0xFFC9, 0xFFCA, 0xFFCB):
                    h = int.from_bytes(data[i + 5:i + 7], "big")
                    w = int.from_bytes(data[i + 7:i + 9], "big")
                    return w, h
                i += 2 + seg_len
        if data[:8] == b'\x89PNG\r\n\x1a\n':
            w = int.from_bytes(data[16:20], "big")
            h = int.from_bytes(data[20:24], "big")
            return w, h
    except Exception:
        pass
    return 640, 640


def write_labels_dir(
    labels: Dict[str, List[List[float]]],
    out_dir: "str | Path",
    *,
    image_sizes: Optional[Dict[str, Tuple[int, int]]] = None,
) -> None:
    from ..data.yolo_dataset import yolo_label_line  # noqa: PLC0415

    out = Path(out_dir)
    label_root = out / "labels" / "train"
    label_root.mkdir(parents=True, exist_ok=True)
    sizes = image_sizes or {}

    for stem, kp_raw in labels.items():
        kp = np.asarray(kp_raw, dtype=np.float64)
        if kp.ndim != 2 or kp.shape[1] != 3:
            _LOG.warning("Skipping label for %s: unexpected shape %s", stem, kp.shape)
            continue
        K = kp.shape[0]
        w, h = sizes.get(stem, (640, 640))
        kp_norm = kp.copy()
        kp_norm[:, 0] = np.clip(kp_norm[:, 0] / max(w, 1), 0.0, 1.0)
        kp_norm[:, 1] = np.clip(kp_norm[:, 1] / max(h, 1), 0.0, 1.0)
        valid = kp[:, 2] > 0
        if valid.any():
            xs = kp[valid, 0]
            ys = kp[valid, 1]
            x_min, x_max = float(xs.min()), float(xs.max())
            y_min, y_max = float(ys.min()), float(ys.max())
        else:
            x_min, x_max = 0.0, float(w)
            y_min, y_max = 0.0, float(h)
        cx = ((x_min + x_max) / 2.0) / max(w, 1)
        cy = ((y_min + y_max) / 2.0) / max(h, 1)
        bw = (x_max - x_min) / max(w, 1)
        bh = (y_max - y_min) / max(h, 1)
        line = yolo_label_line([cx, cy, bw, bh], kp_norm, num_keypoints=K)
        (label_root / f"{stem}.txt").write_text(line + "\n", encoding="utf-8")


_STEM_RE = re.compile(r"^(?P<rec>.+?)_rep(?P<rep>\d+)_f(?P<fidx>\d+)$")


def load_seg_table(seg_csv: "str | Path | None") -> Optional[pd.DataFrame]:
    if seg_csv is None:
        _LOG.warning("load_seg_table: no seg_csv provided; flags disabled")
        return None
    p = Path(seg_csv)
    if not p.exists():
        _LOG.warning("load_seg_table: %s not found; flags disabled", p)
        return None
    try:
        df = pd.read_csv(p, sep=None, engine="python", dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]

        for alias in ("video_id", "videoid", "recording"):
            if alias in df.columns and "video_id" not in df.columns:
                df = df.rename(columns={alias: "video_id"})
                break
        if "repetition_number" in df.columns:
            df["rep_index"] = pd.to_numeric(df["repetition_number"], errors="coerce") - 1
        elif "rep" in df.columns:
            df["rep_index"] = pd.to_numeric(df["rep"], errors="coerce")
        elif "repetition" in df.columns:
            df["rep_index"] = pd.to_numeric(df["repetition"], errors="coerce")
        else:
            df["rep_index"] = np.nan
        for alias, canon in [
            ("first_frame", "first_frame"), ("start_frame", "first_frame"),
            ("last_frame", "last_frame"), ("end_frame", "last_frame"),
        ]:
            if alias in df.columns and canon not in df.columns:
                df = df.rename(columns={alias: canon})
        for col in ("first_frame", "last_frame"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        for col in ("extra_person_in_cam17", "extra_person_in_cam18"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if "extra_person" in df.columns and \
                "extra_person_in_cam17" not in df.columns and \
                "extra_person_in_cam18" not in df.columns:
            val = pd.to_numeric(df["extra_person"], errors="coerce").fillna(0)
            df["extra_person_in_cam17"] = val
            df["extra_person_in_cam18"] = val

        _LOG.info("load_seg_table: loaded %d rows from %s", len(df), p)
        return df
    except Exception as exc:
        _LOG.warning("load_seg_table: failed to parse %s: %s", p, exc)
        return None


def _camera_from_stem(stem: str) -> Optional[str]:
    m = re.search(r"-c(\d+)", stem)
    if m:
        return m.group(1)
    return None


def _video_id_from_rec(rec: str) -> str:
    cleaned = re.sub(r"-c\d+-\d+fps$", "", rec)
    return cleaned


def _seg_flags(
    stem: str,
    seg_table: Optional[pd.DataFrame],
    *,
    camera_id: Optional[str] = None,
) -> Tuple[bool, bool]:
    if seg_table is None:
        return True, False

    m = _STEM_RE.match(stem)
    if not m:
        return True, False

    rec = m.group("rec")
    fidx = int(m.group("fidx"))

    cam = camera_id or _camera_from_stem(rec)
    occlusion_col: Optional[str]
    if cam == "17" and "extra_person_in_cam17" in seg_table.columns:
        occlusion_col = "extra_person_in_cam17"
    elif cam == "18" and "extra_person_in_cam18" in seg_table.columns:
        occlusion_col = "extra_person_in_cam18"
    elif "extra_person_in_cam17" in seg_table.columns:
        occlusion_col = "extra_person_in_cam17"
    elif "extra_person" in seg_table.columns:
        occlusion_col = "extra_person"
    else:
        occlusion_col = None

    if "video_id" not in seg_table.columns:
        return True, False

    video_id = _video_id_from_rec(rec)
    rows = seg_table[seg_table["video_id"] == video_id]
    if rows.empty:
        return True, False

    in_rep = False
    occluded = False
    for row in rows.itertuples():
        first = getattr(row, "first_frame", None)
        last = getattr(row, "last_frame", None)
        first_ok = first is not None and not (first != first)
        last_ok = last is not None and not (last != last)
        if first_ok and last_ok and int(first) <= fidx <= int(last):
            in_rep = True
            if occlusion_col:
                count = getattr(row, occlusion_col, 0)
                try:
                    if float(count) > 0:
                        occluded = True
                except (TypeError, ValueError):
                    pass

    return in_rep, occluded


def _oks_similarity(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    image_w: int = 640,
    image_h: int = 640,
) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.shape != gt.shape or pred.ndim != 2 or pred.shape[1] < 3:
        raise ValueError(
            f"_oks_similarity expects matching (K,3) arrays, got "
            f"{pred.shape} vs {gt.shape}"
        )

    vis = gt[:, 2] > 0
    if not vis.any():
        return 0.0

    xs = gt[vis, 0]
    ys = gt[vis, 1]
    area = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
    if area <= 0.0:
        area = float(max(image_w, 1) * max(image_h, 1))

    try:
        from ..metrics.pose import oks as _oks  # noqa: PLC0415 (pure numpy)

        val = _oks(pred, gt, area)
        return float(np.asarray(val).reshape(-1)[0])
    except Exception:  # pragma: no cover - defensive numpy fallback
        d2 = np.sum((pred[:, :2] - gt[:, :2]) ** 2, axis=-1)
        k = 0.05
        e = d2 / (2.0 * area * (k ** 2) + 1e-12)
        sim = np.exp(-e)
        return float(sim[vis].mean())


class SimulatedClinician:
    def __init__(
        self,
        gt_lookup: Callable[[str], Optional[np.ndarray]],
        *,
        seg_table: Optional[pd.DataFrame] = None,
        correct_sigma: float = 0.0,
        reject_occluded: bool = True,
        seed: int = 0,
        image_w: int = 640,
        image_h: int = 640,
        verify_oks_threshold: Optional[float] = None,
    ) -> None:
        self.gt_lookup = gt_lookup
        self.seg_table = seg_table
        self.correct_sigma = float(correct_sigma)
        self.reject_occluded = reject_occluded
        self._rng = np.random.default_rng(seed)
        self._image_w = image_w
        self._image_h = image_h
        self.verify_oks_threshold = (
            float(verify_oks_threshold) if verify_oks_threshold is not None else None
        )

    def decide(self, cand: Candidate) -> Decision:
        stem = cand.sample_id
        flags = cand.clinical_flags or {}
        in_rep = flags.get("in_labeled_rep", True)
        occluded = flags.get("occluded", False)

        if not in_rep:
            return Decision(sample_id=stem, action="reject")
        if self.reject_occluded and occluded:
            return Decision(sample_id=stem, action="reject")

        gt_kp = self.gt_lookup(stem)
        if gt_kp is None:
            return Decision(sample_id=stem, action="accept")

        gt = np.asarray(gt_kp, dtype=np.float64)

        if self.verify_oks_threshold is not None:
            pred_raw = getattr(cand, "pred_keypoints", None)
            if pred_raw is not None:
                pred = np.asarray(pred_raw, dtype=np.float64)
                if pred.shape == gt.shape:
                    sim = _oks_similarity(
                        pred, gt,
                        image_w=self._image_w, image_h=self._image_h,
                    )
                    if sim >= self.verify_oks_threshold:
                        return Decision(
                            sample_id=stem,
                            action="accept",
                            keypoints=pred.tolist(),
                        )

        corrected = gt.copy()
        if self.correct_sigma > 0.0:
            noise = self._rng.normal(0.0, self.correct_sigma, size=gt[:, :2].shape)
            corrected[:, :2] = gt[:, :2] + noise
        corrected[:, 0] = np.clip(corrected[:, 0], 0.0, float(self._image_w))
        corrected[:, 1] = np.clip(corrected[:, 1], 0.0, float(self._image_h))

        return Decision(
            sample_id=stem,
            action="correct",
            keypoints=corrected.tolist(),
        )


class ClinicianLoop:
    def __init__(
        self,
        state: SessionState,
        *,
        infer_fn: Callable,
        score_fn: Callable,
        train_eval_fn: Callable,
        gt_lookup: Callable,
        oracle: Optional[Any] = None,
        run_mode: str = "human",
        output_root: "str | Path" = "outputs/clinician_loop",
        image_dir: "str | Path" = "outputs/rehab24_6_yolo/images/train",
        seg_table: Optional[pd.DataFrame] = None,
        train_image_stems: Optional[Set[str]] = None,
    ) -> None:
        self.state = state
        self.infer_fn = infer_fn
        self.score_fn = score_fn
        self.train_eval_fn = train_eval_fn
        self.gt_lookup = gt_lookup
        self.oracle = oracle
        self.run_mode = run_mode
        self.seg_table = seg_table
        self.image_dir = Path(image_dir)
        self._train_image_stems: Optional[Set[str]] = train_image_stems

        self._session_dir = Path(output_root) / state.session_id
        self._session_dir.mkdir(parents=True, exist_ok=True)

        self._pending_accepted: Dict[str, List[List[float]]] = {}
        self._pending_rejected: List[str] = []
        self._pending_corrected: List[str] = []
        self._round_seen: Set[str] = set()
        self._last_pred_kp: Dict[str, List[List[float]]] = {}
        self._current_weights: Optional[str] = state.current_weights


    def propose(self, n: int) -> List[Candidate]:
        seen_this_round = (
            set(self.state.labeled)
            | set(self._pending_accepted.keys())
            | set(self._pending_rejected)
            | self._round_seen
        )

        pool, term_scores = self.score_fn(self._current_weights)
        all_ids: List[str] = pool.sample_ids
        N = len(all_ids)

        if isinstance(term_scores, dict) and N > 0:
            first_val = next(iter(term_scores.values()), None)
            if isinstance(first_val, np.ndarray):
                composite = np.zeros(N, dtype=np.float64)
                for arr in term_scores.values():
                    composite += np.asarray(arr, dtype=np.float64)
            elif isinstance(first_val, dict):
                composite = np.array(
                    [sum(term_scores.get(sid, {}).values()) for sid in all_ids],
                    dtype=np.float64,
                )
            else:
                composite = np.array(
                    [float(term_scores.get(sid, 0.0)) for sid in all_ids],
                    dtype=np.float64,
                )
        else:
            composite = np.zeros(N, dtype=np.float64)

        available_mask = np.array([sid not in seen_this_round for sid in all_ids])
        if not available_mask.any():
            return []

        scores_avail = np.where(available_mask, composite, -np.inf)
        ranked_idx = np.argsort(-scores_avail)
        top_idx = [i for i in ranked_idx if available_mask[i]][:n]
        top_stems = [all_ids[i] for i in top_idx]

        if self._train_image_stems is not None:
            bad = [s for s in top_stems if s not in self._train_image_stems]
            if bad:
                _LOG.error(
                    "Proposed stems not in train image set: %s", bad[:5]
                )
                top_stems = [s for s in top_stems if s in self._train_image_stems]
                top_idx = [i for i in top_idx if all_ids[i] in set(top_stems)]

        pseudo_labels: Dict[str, Any] = {}
        if top_stems and self.infer_fn is not None:
            try:
                pseudo_labels = self.infer_fn(self._current_weights, top_stems) or {}
            except Exception as exc:
                _LOG.warning("infer_fn failed: %s", exc)

        candidates: List[Candidate] = []
        for i in top_idx:
            sid = all_ids[i]
            img_path = str(self.image_dir / f"{sid}.jpg")
            pred_kp = pseudo_labels.get(sid)
            if pred_kp is None:
                if hasattr(pool, "predictions") and pool.predictions is not None:
                    pred_kp = pool.predictions[i].tolist()
                else:
                    pred_kp = [[0.0, 0.0, 0.0]] * 26
            elif hasattr(pred_kp, "tolist"):
                pred_kp = pred_kp.tolist()

            if isinstance(term_scores, dict):
                fv = next(iter(term_scores.values()), None)
                if isinstance(fv, np.ndarray):
                    sample_terms = {k: float(v[i]) for k, v in term_scores.items()}
                elif isinstance(fv, dict):
                    sample_terms = dict(term_scores.get(sid, {}))
                else:
                    sample_terms = {"score": float(term_scores.get(sid, 0.0))}
            else:
                sample_terms = {}

            cam = _camera_from_stem(sid)
            in_rep, occ = _seg_flags(sid, self.seg_table, camera_id=cam)
            clinical_flags = {
                "critical_joints": [],
                "in_labeled_rep": in_rep,
                "occluded": occ,
            }

            candidates.append(Candidate(
                sample_id=sid,
                image_path=img_path,
                pred_keypoints=pred_kp,
                term_scores=sample_terms,
                clinical_flags=clinical_flags,
            ))
            self._last_pred_kp[sid] = pred_kp
            self._round_seen.add(sid)

        return candidates


    def apply_decisions(self, decisions: List[Decision]) -> CommitResult:
        for dec in decisions:
            sid = dec.sample_id
            if dec.action == "reject":
                if sid not in self._pending_rejected:
                    self._pending_rejected.append(sid)
            elif dec.action == "correct":
                if dec.keypoints is None:
                    _LOG.warning(
                        "Decision 'correct' for %s has no keypoints; using pseudo-label",
                        sid,
                    )
                    kp = self._get_pseudo_kp(sid)
                else:
                    kp = dec.keypoints
                    if sid not in self._pending_corrected:
                        self._pending_corrected.append(sid)
                self._pending_accepted[sid] = kp
            else:
                if sid not in self._pending_accepted:
                    self._pending_accepted[sid] = self._get_pseudo_kp(sid)

        K = self.state.budget_per_round
        need_more = len(self._pending_accepted) < K

        return CommitResult(
            accepted=list(self._pending_accepted.keys()),
            rejected=list(self._pending_rejected),
            corrected=list(self._pending_corrected),
            need_more=need_more,
        )

    def _get_pseudo_kp(self, sid: str) -> List[List[float]]:
        kp = self._last_pred_kp.get(sid)
        if kp is not None:
            return kp
        return [[0.0, 0.0, 0.0]] * 26


    def run_auto_round(self, decision_fn: Callable[[Candidate], Decision]) -> RoundRecord:
        K = self.state.budget_per_round
        max_iters = 20

        for _ in range(max_iters):
            needed = K - len(self._pending_accepted)
            if needed <= 0:
                break
            batch = self.propose(n=needed * 2)
            if not batch:
                _LOG.warning(
                    "Session %s round %d: pool exhausted after %d accepted (need %d)",
                    self.state.session_id, self.state.round,
                    len(self._pending_accepted), K,
                )
                break
            decisions = [decision_fn(c) for c in batch]
            self.apply_decisions(decisions)

        return self.commit_and_train()


    def commit_and_train(self) -> RoundRecord:
        if self.state.status in ("failed", "training", "done"):
            raise RoundTrainingError(
                f"Session is {self.state.status}; do not grant another label budget. "
                "Restart the experiment in a new output directory, or use the "
                "runner's --restart-incomplete option to archive a failed session."
            )
        K = self.state.budget_per_round
        round_idx = self.state.round

        accepted_stems = list(self._pending_accepted.keys())[:K]
        n_accepted = len(accepted_stems)
        n_rejected = len(self._pending_rejected)
        n_corrected = len([s for s in self._pending_corrected if s in accepted_stems])
        n_proposed = len(self._round_seen)

        if n_accepted < K:
            _LOG.warning(
                "Round %d: committed %d/%d (pool may be exhausted)",
                round_idx, n_accepted, K,
            )

        if self.run_mode == "auto":
            if self.state.condition == "A":
                clinician = "none"
            else:
                clinician = "simulated"
        else:
            if self.state.condition in ("A", "ref"):
                clinician = "none"
            else:
                clinician = "human"

        for sid in accepted_stems:
            if sid not in self.state.labeled:
                self.state.labeled.append(sid)
            self.state.labels[sid] = self._pending_accepted[sid]

        round_dir = self._session_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        image_sizes: Dict[str, Tuple[int, int]] = {}
        for sid in self.state.labeled:
            image_sizes[sid] = _get_image_size(self.image_dir / f"{sid}.jpg")

        previous_weights = self._current_weights
        previous_state_weights = self.state.current_weights
        previous_history_size = len(self.state.history)
        self.state.status = "training"
        try:
            self._save_state()
            write_labels_dir(self.state.labels, round_dir, image_sizes=image_sizes)
            train_result = self.train_eval_fn(
                list(self.state.labeled),
                self.state.labels,
                str(round_dir),
                self.state.seed,
            )
            if not isinstance(train_result, dict):
                raise ValueError("train_eval_fn must return a result dictionary")
            weights = train_result.get("weights")
            if not isinstance(weights, (str, Path)) or not str(weights).strip():
                raise ValueError("train_eval_fn returned no weights")
            if not Path(weights).is_file() or Path(weights).stat().st_size == 0:
                raise ValueError(f"train_eval_fn returned missing/empty weights: {weights}")
            for key in ("map50", "map50_95"):
                value = train_result.get(key)
                if (isinstance(value, (bool, str)) or not np.isscalar(value)
                        or not np.isfinite(float(value))
                        or not 0.0 <= float(value) <= 1.0):
                    raise ValueError(f"train_eval_fn returned invalid {key}: {value!r}")
            rec = RoundRecord(
                round=round_idx,
                n_labeled=len(self.state.labeled),
                pose_map50=float(train_result["map50"]),
                pose_map50_95=float(train_result["map50_95"]),
                n_proposed=n_proposed,
                n_accepted=n_accepted,
                n_rejected=n_rejected,
                n_corrected=n_corrected,
                clinician=clinician,
            )
            self._current_weights = str(weights)
            self.state.current_weights = str(weights)
            self.state.failure = None
            self.state.history.append(rec)
            self.state.round += 1
            self._save_state()
            self._append_curve_csv(rec)
            self.state.status = "active"
            self._save_state()
        except Exception as exc:
            _LOG.error("Round %d training/result persistence failed: %s", round_idx, exc)
            self._current_weights = previous_weights
            self.state.current_weights = previous_state_weights
            del self.state.history[previous_history_size:]
            self.state.round = round_idx
            self.state.status = "failed"
            self.state.failure = {
                "round": round_idx, "error_type": type(exc).__name__,
                "error": str(exc), "n_labeled": len(self.state.labeled),
                "accepted": accepted_stems, "n_accepted": n_accepted,
                "n_rejected": n_rejected, "n_corrected": n_corrected,
                "previous_weights": previous_weights,
            }
            try:
                self._save_state()
                failure_path = round_dir / "failure.json"
                failure_path.write_text(json.dumps(self.state.failure, indent=2),
                                        encoding="utf-8")
            except OSError:
                _LOG.exception("Could not persist failure details for round %d", round_idx)
            raise RoundTrainingError(
                f"Round {round_idx} training/evaluation or result persistence failed: {exc}"
            ) from exc

        self._pending_accepted.clear()
        self._pending_rejected.clear()
        self._pending_corrected.clear()
        self._round_seen.clear()
        self._last_pred_kp.clear()

        return rec


    def _save_state(self) -> None:
        state_path = self._session_dir / "state.json"
        tmp_path = state_path.with_suffix(".json.tmp")
        try:
            text = json.dumps(self.state.to_dict(), ensure_ascii=False, indent=2,
                              default=str)
            tmp_path.write_text(text, encoding="utf-8")
            os.replace(str(tmp_path), str(state_path))
        except Exception as exc:
            _LOG.error("Failed to persist state.json: %s", exc)
            raise

    def _append_curve_csv(self, rec: RoundRecord) -> None:
        csv_path = self._session_dir / "curve.csv"
        row = {
            "round": rec.round,
            "n_labeled": rec.n_labeled,
            "map50": rec.pose_map50,
            "map50_95": rec.pose_map50_95,
            "n_accepted": rec.n_accepted,
            "n_rejected": rec.n_rejected,
            "n_corrected": rec.n_corrected,
            "clinician": rec.clinician,
        }
        write_header = not csv_path.exists()
        tmp_path = csv_path.with_suffix(".csv.tmp")
        previous = csv_path.read_text(encoding="utf-8") if not write_header else ""
        with tmp_path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(previous)
            writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        os.replace(tmp_path, csv_path)


    def curve(self) -> List[dict]:
        return [
            {
                "round": rec.round,
                "n_labeled": rec.n_labeled,
                "map50": rec.pose_map50,
                "map50_95": rec.pose_map50_95,
            }
            for rec in self.state.history
        ]

    @classmethod
    def load_state(cls, session_dir: "str | Path") -> SessionState:
        p = Path(session_dir) / "state.json"
        with p.open("r", encoding="utf-8") as fh:
            d = json.load(fh)
        return SessionState.from_dict(d)
