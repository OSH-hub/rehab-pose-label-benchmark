"""Simulated-correction active-learning engine.

Engine module for the paper experiment comparing:
  - Cond A:   AL + pseudo-labels (no human)
  - Cond B:   AL + simulated correction and segmentation-metadata rejection
  - Cond ref: Full-GT AL comparator (every selected frame receives exact GT,
              with no rejections)

Import-safety requirement: numpy/pandas only at module load. No torch/cv2/ultralytics/fastapi
imports here. Heavy model work is done via INJECTED callables only.

Round cycle: propose -> (clinician decisions) -> commit -> train+eval -> record

Proposal bookkeeping excludes both committed and pending-seen stems so that
repeated calls within a round always surface NEW frames. commit_and_train()
owns the propose→decide→top-up loop internally and guarantees exactly K labels
are committed per round unless the pool is genuinely exhausted.
"""

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
    # private helper exposed for tests
    "_seg_flags",
    "_get_image_size",
    "_oks_similarity",
]

_LOG = logging.getLogger(__name__)


class RoundTrainingError(RuntimeError):
    """Accepted labels were saved, but this round has no valid trained result."""

# ---------------------------------------------------------------------------
# Dataclasses (match contract §2 exactly)
# ---------------------------------------------------------------------------

@dataclass
class Candidate:
    """A frame proposed for clinician review."""
    sample_id: str            # image stem, e.g. "PM_045_rep00_f100"
    image_path: str
    pred_keypoints: list      # (26,3) [x,y,score] in PIXEL coords
    term_scores: dict         # {"uncertainty":.., "temporal":.., ...}
    clinical_flags: dict      # {"critical_joints":[...], "in_labeled_rep":bool, "occluded":bool}


@dataclass
class Decision:
    """Clinician decision on a single candidate."""
    sample_id: str
    action: str               # "accept" | "reject" | "correct"
    keypoints: Optional[list] = None  # (26,3), required iff action=="correct"


@dataclass
class RoundRecord:
    """Metrics and counts for one committed AL round."""
    round: int
    n_labeled: int
    pose_map50: Optional[float]
    pose_map50_95: Optional[float]
    n_proposed: int           # total unique stems surfaced during the round
    n_accepted: int           # stems actually committed (== K unless pool exhausted)
    n_rejected: int
    n_corrected: int
    clinician: str            # "simulated" | "human" | "none"


@dataclass
class SessionState:
    """Full persisted session state (written as state.json after each commit)."""
    session_id: str
    condition: str            # "A" | "B" | "ref"
    seed: int
    strategy: str             # default "temporal"
    budget_per_round: int     # K frames added per round
    round: int                # 0-based; incremented after each commit
    labeled: List[str]        # sample_ids committed to training
    labels: Dict[str, List[List[float]]]  # stem -> (26,3) [x,y,score]
    history: List[RoundRecord]
    status: str               # "active" | "training" | "done" | "failed"
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
    """Result of apply_decisions: how many were accepted/rejected/corrected."""
    accepted: List[str]
    rejected: List[str]
    corrected: List[str]
    need_more: bool           # True when accepted < budget_per_round


# ---------------------------------------------------------------------------
# GT label IO helpers (no heavy imports)
# ---------------------------------------------------------------------------

def read_gt_label(
    stem: str,
    labels_dir: "str | Path",
    image_w: int = 640,
    image_h: int = 640,
    *,
    split: str = "train",
) -> Optional[np.ndarray]:
    """Read a YOLO-pose label txt and return denormalized (26,3) [x,y,vis].

    Only reads from the ``train`` split to prevent validation leakage. If ``split``
    is not ``"train"`` this function raises ``ValueError``.

    The file is resolved at ``labels_dir/labels/train/{stem}.txt`` (primary)
    or ``labels_dir/train/{stem}.txt`` (fallback).  Returns ``None`` when
    the file is missing.

    Coordinates are denormalized to pixel space using ``image_w`` / ``image_h``.
    Visibility stays as-is (integer 0/1/2 from the label line).
    """
    # Refuse any non-training split to prevent validation-set leakage.
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
            # YOLO-pose line: class cx cy w h  (px py pv)*K
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
    """Return (width, height) from image file WITHOUT importing cv2.

    Uses minimal JPEG/PNG header parsing. Falls back to (640, 640).
    """
    p = Path(image_path)
    if not p.exists():
        return 640, 640
    try:
        data = p.read_bytes()
        # JPEG SOF markers
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
        # PNG IHDR
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
    """Write per-frame YOLO-pose label txts from a ``labels`` dict.

    ``labels[stem]`` is ``(K, 3)`` in pixel coordinates ``[x, y, vis_or_score]``.
    The label lines are normalized back to ``[0,1]`` using ``image_sizes``
    (or falling back to 640×640) and written as YOLO-pose format. Prediction
    scores are converted to integer visibility by the effective 0.5 split in
    :func:`yolo_label_line`. Each frame's box is derived from the extent of
    keypoints whose input third-column value is positive; if none are positive,
    the full frame is used.

    Writes to ``out_dir/labels/train/<stem>.txt`` to match the YOLO layout.
    """
    # yolo_label_line has no heavy deps (numpy + stdlib only); safe lazy import.
    from ..data.yolo_dataset import yolo_label_line  # noqa: PLC0415

    out = Path(out_dir)
    label_root = out / "labels" / "train"
    label_root.mkdir(parents=True, exist_ok=True)
    sizes = image_sizes or {}

    for stem, kp_raw in labels.items():
        kp = np.asarray(kp_raw, dtype=np.float64)  # (K, 3) pixel
        if kp.ndim != 2 or kp.shape[1] != 3:
            _LOG.warning("Skipping label for %s: unexpected shape %s", stem, kp.shape)
            continue
        K = kp.shape[0]
        w, h = sizes.get(stem, (640, 640))
        # Normalize to [0,1]
        kp_norm = kp.copy()
        kp_norm[:, 0] = np.clip(kp_norm[:, 0] / max(w, 1), 0.0, 1.0)
        kp_norm[:, 1] = np.clip(kp_norm[:, 1] / max(h, 1), 0.0, 1.0)
        # Derive tight bbox from visible keypoints
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


# ---------------------------------------------------------------------------
# Segmentation table loader (REHAB24-6 Segmentation.csv)
# ---------------------------------------------------------------------------

_STEM_RE = re.compile(r"^(?P<rec>.+?)_rep(?P<rep>\d+)_f(?P<fidx>\d+)$")


def load_seg_table(seg_csv: "str | Path | None") -> Optional[pd.DataFrame]:
    """Load REHAB24-6 ``Segmentation.csv`` (``;``-delimited).

    Handles the REAL column schema:
      ``video_id ; repetition_number ; exercise_id ; person_id ;
        first_frame ; last_frame ; cam17_orientation ; mocap_erroneous ;
        exercise_subtype ; lights_on ; extra_person_in_cam17 ;
        extra_person_in_cam18 ; correctness``

    ``repetition_number`` is 1-based in the CSV; it is converted to
    0-based (rep_index) to match stem encoding ``rep00``→0, ``rep01``→1, …

    ``extra_person_in_cam17`` and ``extra_person_in_cam18`` are retained as
    separate numeric columns. ``_seg_flags`` receives the frame's camera
    id (from the stem via ``_camera_from_stem``) to pick the right column.

    Returns ``None`` (with a warning) when the file is missing/unparseable.
    """
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

        # --- Column aliasing (real names → canonical names used internally) ---
        # video id
        for alias in ("video_id", "videoid", "recording"):
            if alias in df.columns and "video_id" not in df.columns:
                df = df.rename(columns={alias: "video_id"})
                break
        # rep: real column is repetition_number (1-based) → convert to rep_index (0-based)
        # Decrement so _seg_flags compares against the 0-based stem repetition.
        if "repetition_number" in df.columns:
            df["rep_index"] = pd.to_numeric(df["repetition_number"], errors="coerce") - 1
        elif "rep" in df.columns:
            df["rep_index"] = pd.to_numeric(df["rep"], errors="coerce")
        elif "repetition" in df.columns:
            df["rep_index"] = pd.to_numeric(df["repetition"], errors="coerce")
        else:
            df["rep_index"] = np.nan  # no rep column: match any row for video_id
        # frame ranges
        for alias, canon in [
            ("first_frame", "first_frame"), ("start_frame", "first_frame"),
            ("last_frame", "last_frame"), ("end_frame", "last_frame"),
        ]:
            if alias in df.columns and canon not in df.columns:
                df = df.rename(columns={alias: canon})
        for col in ("first_frame", "last_frame"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Keep cam17/cam18 extra-person counts as separate numeric columns.
        for col in ("extra_person_in_cam17", "extra_person_in_cam18"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        # Also accept legacy alias "extra_person" (boolean-ish)
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
    """Extract camera id from a stem like ``PM_000-c17-25fps_rep00_f0010``.

    Returns ``"17"`` or ``"18"`` if the stem contains ``-c17`` / ``-c18``,
    else ``None`` (no camera info → occlusion falls back to cam17 column).
    """
    m = re.search(r"-c(\d+)", stem)
    if m:
        return m.group(1)
    return None


def _video_id_from_rec(rec: str) -> str:
    """Strip the camera/fps suffix from the rec group to get the base video_id.

    Real REHAB24-6 stem format: ``<video_id>-c<cam>-<fps>fps``
    e.g. ``PM_000-c17-25fps`` → video_id = ``PM_000``.
    If no such suffix, returns ``rec`` unchanged.
    """
    # Strip trailing -c<N>-<fps>fps
    cleaned = re.sub(r"-c\d+-\d+fps$", "", rec)
    return cleaned


def _seg_flags(
    stem: str,
    seg_table: Optional[pd.DataFrame],
    *,
    camera_id: Optional[str] = None,
) -> Tuple[bool, bool]:
    """Return ``(in_labeled_rep, occluded)`` for a frame stem.

    Production stem format: ``<video_id>_rep00_f<global_frame_idx>``
    The ``rep`` token is ALWAYS ``00`` — a fixed placeholder, not a true rep
    index. All 8367 production stems use it. The GLOBAL frame index ``f<idx>``
    is what matters and aligns with Segmentation.csv ``first_frame/last_frame``.

    Matching logic (rep token is IGNORED):
      1. Parse ``video_id`` (everything before ``_rep``) and ``fidx`` (global
         frame index after ``_f``).
      2. ``in_labeled_rep = True`` iff ``fidx`` falls within
         ``[first_frame, last_frame]`` of ANY rep row for that ``video_id``.
         Frames in between-rep gaps and before/after the entire rep span are
         out-of-rep and are legitimately rejectable.
      3. ``occluded`` = ``extra_person_in_cam17 > 0`` on the rep row whose
         range contains ``fidx`` (or any row if no containing row found and the
         video has any extra-person rows). Production stems carry no camera
         token; defaults to cam17 (most common in dataset). If a ``camera_id``
         is passed explicitly (tests / cam-encoded stems) it is honoured.

    Defaults when seg_table is None or video_id not found:
      ``in_labeled_rep=True, occluded=False`` (conservative: don't reject).
    """
    if seg_table is None:
        return True, False

    m = _STEM_RE.match(stem)
    if not m:
        return True, False

    rec = m.group("rec")
    # rep token (m.group("rep")) is IGNORED — it is always "00" in production.
    fidx = int(m.group("fidx"))

    # Determine camera column from explicit override, stem encoding, or default.
    cam = camera_id or _camera_from_stem(rec)
    occlusion_col: Optional[str]
    if cam == "17" and "extra_person_in_cam17" in seg_table.columns:
        occlusion_col = "extra_person_in_cam17"
    elif cam == "18" and "extra_person_in_cam18" in seg_table.columns:
        occlusion_col = "extra_person_in_cam18"
    elif "extra_person_in_cam17" in seg_table.columns:
        # No camera info in stem — default to cam17 (production convention).
        occlusion_col = "extra_person_in_cam17"
    elif "extra_person" in seg_table.columns:
        occlusion_col = "extra_person"
    else:
        occlusion_col = None

    if "video_id" not in seg_table.columns:
        return True, False

    # Strip camera/fps suffix from rec to get the base video_id for lookup.
    video_id = _video_id_from_rec(rec)
    rows = seg_table[seg_table["video_id"] == video_id]
    if rows.empty:
        return True, False

    in_rep = False
    occluded = False
    for row in rows.itertuples():
        # Check whether the GLOBAL frame index falls within this rep's range.
        first = getattr(row, "first_frame", None)
        last = getattr(row, "last_frame", None)
        first_ok = first is not None and not (first != first)  # NaN guard
        last_ok = last is not None and not (last != last)
        if first_ok and last_ok and int(first) <= fidx <= int(last):
            in_rep = True
            # Occlusion comes from the rep row that CONTAINS this frame.
            if occlusion_col:
                count = getattr(row, occlusion_col, 0)
                try:
                    if float(count) > 0:
                        occluded = True
                except (TypeError, ValueError):
                    pass

    return in_rep, occluded


# ---------------------------------------------------------------------------
# OKS similarity helper (pure numpy; reuses metrics.pose.oks when available)
# ---------------------------------------------------------------------------

def _oks_similarity(
    pred: np.ndarray,
    gt: np.ndarray,
    *,
    image_w: int = 640,
    image_h: int = 640,
) -> float:
    """Return COCO-style OKS between a prediction and GT for ONE frame.

    ``pred`` / ``gt`` are ``(K, 3)`` arrays. For ``gt`` the third column is
    visibility (``vis > 0`` marks a keypoint as valid); for ``pred`` it is the
    detector score and is ignored for the distance. Only keypoints visible in
    GT contribute to the similarity.

    The object area ``s^2`` is the bbox area of the visible GT keypoints
    (falling back to the full frame when GT carries no visible keypoints).
    Uniform per-keypoint constants (``k_i = 0.05``) are used — the same default
    as :func:`rehabal_pose.metrics.pose.oks`, which this delegates to when
    importable. Returns ``0.0`` when there is no visible keypoint to score.

    Implementation constraint: numpy-only. ``metrics.pose`` is pure numpy (no torch/cv2), so the
    lazy import is safe; a numpy fallback is provided for isolation.
    """
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
        # Degenerate bbox (single visible point / collinear): fall back to frame.
        area = float(max(image_w, 1) * max(image_h, 1))

    try:
        from ..metrics.pose import oks as _oks  # noqa: PLC0415 (pure numpy)

        val = _oks(pred, gt, area)  # (1,) since inputs are single-sample
        return float(np.asarray(val).reshape(-1)[0])
    except Exception:  # pragma: no cover - defensive numpy fallback
        d2 = np.sum((pred[:, :2] - gt[:, :2]) ** 2, axis=-1)  # (K,)
        k = 0.05
        e = d2 / (2.0 * area * (k ** 2) + 1e-12)
        sim = np.exp(-e)
        return float(sim[vis].mean())


# ---------------------------------------------------------------------------
# SimulatedClinician (oracle, GT-based)
# ---------------------------------------------------------------------------

class SimulatedClinician:
    """Oracle clinician using ground-truth labels.

    Decision logic (contract §2):
      - reject if NOT in_labeled_rep or occluded.
      - else, for an accepted (in-rep, non-occluded) frame:
          * if ``verify_oks_threshold`` is None (default) → correct toward GT
            with optional Gaussian noise (``correct_sigma``); ``correct_sigma==0``
            makes the correction exactly GT.
          * if ``verify_oks_threshold`` is set → compute the OKS similarity of
            the model prediction (``Candidate.pred_keypoints``) against GT. When
            OKS ``>=`` the threshold the clinician accepts the prediction as-is
            (``action="accept"``, GT is not applied);
            otherwise it follows the GT correction path above. This thresholded
            mode is an OKS-gated selective-correction arm, not a passive
            verification-only arm.

    Corrected keypoints are clipped to ``[0, w] × [0, h]`` so noise cannot push
    joints outside the image.

    Parameters
    ----------
    gt_lookup:
        Callable(stem) -> (26,3) ndarray in pixel coords, or None.
    seg_table:
        REHAB24-6 Segmentation.csv DataFrame (from ``load_seg_table``).
    correct_sigma:
        Std-dev of Gaussian noise added to GT keypoints (pixels). 0 = exact GT.
    reject_occluded:
        If True (default), reject occluded frames.
    seed:
        Random seed for reproducibility.
    image_w, image_h:
        Frame dimensions for keypoint clipping. Defaults to 640×640.
    verify_oks_threshold:
        OKS threshold in ``[0, 1]`` for selective correction. ``None`` (default)
        uses the always-correct-to-GT behavior. When set, an
        accepted frame whose prediction has OKS ``>=`` this value is verified and
        adopted unchanged (no GT overwrite); below it, GT correction applies.
    """

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
        """Return a Decision for a single Candidate."""
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

        # OKS-gated selective correction is active only when a threshold is set.
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
                        # Prediction meets the threshold: adopt it without a GT overwrite.
                        return Decision(
                            sample_id=stem,
                            action="accept",
                            keypoints=pred.tolist(),
                        )
                    # else: fall through to GT correction below.

        corrected = gt.copy()
        if self.correct_sigma > 0.0:
            noise = self._rng.normal(0.0, self.correct_sigma, size=gt[:, :2].shape)
            corrected[:, :2] = gt[:, :2] + noise
        # Clip x to [0, w] and y to [0, h].
        corrected[:, 0] = np.clip(corrected[:, 0], 0.0, float(self._image_w))
        corrected[:, 1] = np.clip(corrected[:, 1], 0.0, float(self._image_h))

        return Decision(
            sample_id=stem,
            action="correct",
            keypoints=corrected.tolist(),
        )


# ---------------------------------------------------------------------------
# ClinicianLoop — the main engine
# ---------------------------------------------------------------------------

class ClinicianLoop:
    """Active-learning loop with simulated or externally supplied decisions.

    ``propose(n)`` excludes both committed stems and stems
    already seen in the current round (pending-accepted + pending-rejected),
    so repeated calls within a round reliably yield NEW frames.

    ``commit_and_train()`` enforces that exactly ``K`` labels are committed
    per round, by internally calling ``propose → decide`` in a top-up loop.
    External callers (the server's ``/auto`` endpoint) drive this by calling
    ``run_auto_round()``, which wraps the full propose→decide→top-up→commit
    cycle inside the engine.

    ``oracle`` and the clinician tag are determined by the explicit
    ``run_mode`` field (``"auto"`` → SimulatedClinician / ``"human"`` → no
    oracle) rather than condition alone.

    Parameters
    ----------
    state : SessionState
    infer_fn : (weights, stems) -> {stem: (26,3)} pixel keypoints
    score_fn : (weights) -> (Pool, {term_name: (N,)})
    train_eval_fn : (labeled_stems, labels, out_dir, seed) -> {map50, map50_95, weights}
    gt_lookup : (stem) -> (26,3) ndarray | None
    oracle : SimulatedClinician or any object with ``decide(Candidate)->Decision``
        Set by the server when running headless auto rounds for cond B/ref.
    run_mode : "auto" | "human"
        "auto"  → oracle-driven / pseudo-label (headless experiment).
        "human" → no oracle; decisions come from the web UI.
    output_root, image_dir, seg_table : see class body
    train_image_stems : optional set of valid training-image stems
    """

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

        # Round-scoped staging (cleared on commit)
        self._pending_accepted: Dict[str, List[List[float]]] = {}
        self._pending_rejected: List[str] = []
        self._pending_corrected: List[str] = []
        # Track all stems surfaced this round for accurate n_proposed counts.
        self._round_seen: Set[str] = set()
        # Quick-lookup for pseudo-labels of last proposed batch
        self._last_pred_kp: Dict[str, List[List[float]]] = {}
        # Current weights (updated after each commit)
        self._current_weights: Optional[str] = state.current_weights

    # ------------------------------------------------------------------ propose

    def propose(self, n: int) -> List[Candidate]:
        """Score the pool and return up to ``n`` candidates not yet seen this round.

        The exclusion set is ``committed ∪ pending_accepted ∪
        pending_rejected``, so re-proposals within the same round always
        yield fresh frames.
        """
        # Build the exclusion set for this round
        seen_this_round = (
            set(self.state.labeled)
            | set(self._pending_accepted.keys())
            | set(self._pending_rejected)
            | self._round_seen
        )

        pool, term_scores = self.score_fn(self._current_weights)
        all_ids: List[str] = pool.sample_ids
        N = len(all_ids)

        # Composite score
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

        # Keep proposals within the training set when an allow-list is provided.
        if self._train_image_stems is not None:
            bad = [s for s in top_stems if s not in self._train_image_stems]
            if bad:
                _LOG.error(
                    "Proposed stems not in train image set: %s", bad[:5]
                )
                top_stems = [s for s in top_stems if s in self._train_image_stems]
                top_idx = [i for i in top_idx if all_ids[i] in set(top_stems)]

        # Infer pseudo-labels for proposed stems
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

            # Per-sample term_scores dict for the frontend
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

            # Clinical flags from seg_table
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
            # Cache pseudo-label for _get_pseudo_kp
            self._last_pred_kp[sid] = pred_kp
            # Accumulate all stems surfaced during the round.
            self._round_seen.add(sid)

        return candidates

    # --------------------------------------------------------- apply_decisions

    def apply_decisions(self, decisions: List[Decision]) -> CommitResult:
        """Stage per-candidate decisions. Accumulates across multiple calls.

        Returns a ``CommitResult`` indicating whether K accepted has been reached.
        """
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
            else:  # "accept"
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
        """Return cached pseudo-label keypoints for a stem."""
        kp = self._last_pred_kp.get(sid)
        if kp is not None:
            return kp
        return [[0.0, 0.0, 0.0]] * 26

    # ---------------------------------------------------------- run_auto_round

    def run_auto_round(self, decision_fn: Callable[[Candidate], Decision]) -> RoundRecord:
        """Run a full propose→decide→top-up→commit cycle inside the engine.

        This method owns the loop so top-up proposals exclude already-
        seen stems of this round. The cycle continues until K frames are accepted
        or the pool is exhausted.

        ``decision_fn`` maps a ``Candidate`` to a ``Decision``:
          - Cond A:   always returns ``Decision(sid, "accept")``
          - Cond B:   SimulatedClinician.decide
          - Cond ref: accept-all-to-GT lambda (no rejections)
        """
        K = self.state.budget_per_round
        max_iters = 20  # safety cap against infinite loops on tiny test pools

        for _ in range(max_iters):
            needed = K - len(self._pending_accepted)
            if needed <= 0:
                break
            # Over-propose to absorb expected rejections
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

    # --------------------------------------------------------- commit_and_train

    def commit_and_train(self) -> RoundRecord:
        """Commit pending accepted frames (up to K), write labels, train+eval.

        Takes exactly K accepted stems, or fewer if the pool is exhausted.
        Caller should use ``run_auto_round`` for the full propose→commit cycle;
        or use ``propose``/``apply_decisions``/``commit_and_train`` for the
        interactive human-UI flow.
        """
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
        # n_proposed is the number of unique stems surfaced this round.
        n_proposed = len(self._round_seen)

        if n_accepted < K:
            _LOG.warning(
                "Round %d: committed %d/%d (pool may be exhausted)",
                round_idx, n_accepted, K,
            )

        # Derive the clinician tag from run_mode rather than oracle presence.
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

        # Commit to state
        for sid in accepted_stems:
            if sid not in self.state.labeled:
                self.state.labeled.append(sid)
            self.state.labels[sid] = self._pending_accepted[sid]

        # Write per-frame label txts from state.labels (not prebuilt files)
        round_dir = self._session_dir / f"round_{round_idx}"
        round_dir.mkdir(parents=True, exist_ok=True)

        image_sizes: Dict[str, Tuple[int, int]] = {}
        for sid in self.state.labeled:
            image_sizes[sid] = _get_image_size(self.image_dir / f"{sid}.jpg")

        # Train + eval
        previous_weights = self._current_weights
        previous_state_weights = self.state.current_weights
        previous_history_size = len(self.state.history)
        self.state.status = "training"
        try:
            # Acceptance is a separate durable event from training success. A
            # killed process leaves status=training, never a successful round.
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
            # Keep the durable session locked until both result files exist.
            # A storage failure must never permit a UI retry of this budget.
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
            # Do not append a failed trial to the successful metric history or
            # increment round. Previously accepted labels remain auditable.
            # A full/read-only disk can also reject the failure record. The
            # in-memory lock still holds; the last durable state is training.
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

        # Clear round-scoped staging
        self._pending_accepted.clear()
        self._pending_rejected.clear()
        self._pending_corrected.clear()
        self._round_seen.clear()
        self._last_pred_kp.clear()

        return rec

    # ---------------------------------------------------------------- helpers

    def _save_state(self) -> None:
        """Crash-safe atomic write of state.json."""
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

    # ------------------------------------------------------------------ curve

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
