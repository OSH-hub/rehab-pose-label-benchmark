"""Unit tests for src/rehabal_pose/review/clinician_loop.py.

Test-isolation requirement: no real GPU, no torch/cv2/ultralytics/fastapi. All heavy model
work is replaced with deterministic FAKE callables.

Coverage includes proposal exclusion and top-up behavior, segmentation-schema
handling, the reference condition, run-mode tags, training-split enforcement,
keypoint clipping, and round-level proposal counts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from rehabal_pose.review.clinician_loop import (
    Candidate,
    ClinicianLoop,
    CommitResult,
    Decision,
    RoundRecord,
    SessionState,
    SimulatedClinician,
    load_seg_table,
    read_gt_label,
    write_labels_dir,
    _seg_flags,
    _camera_from_stem,
    _oks_similarity,
)

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

NUM_KP = 26
K_BUDGET = 4
POOL_SIZE = 20


def _make_kp(val: float = 0.5, w: int = 640, h: int = 480) -> np.ndarray:
    """Return (26,3) pixel keypoints."""
    kp = np.full((NUM_KP, 3), val * w, dtype=np.float64)
    kp[:, 1] = val * h
    kp[:, 2] = 2.0
    return kp


def _make_stems(n: int = POOL_SIZE, prefix: str = "PM_000_rep00_f") -> List[str]:
    return [f"{prefix}{i:04d}" for i in range(n)]


# ---------------------------------------------------------------------------
# FAKE injected callables
# ---------------------------------------------------------------------------

class FakePool:
    def __init__(self, stems: List[str]) -> None:
        N = len(stems)
        self.sample_ids = stems
        self.size = N
        self.predictions = np.zeros((N, NUM_KP, 3), dtype=np.float64)
        for i in range(N):
            self.predictions[i] = _make_kp(0.3 + 0.01 * i)
        self.embeddings = None
        self.clip_ids = None
        self.frame_ids = list(range(N))
        self.metadata: dict = {}

    def keypoints_xy(self) -> np.ndarray:
        return self.predictions[:, :, :2]

    def meta(self, key: str):
        return self.metadata.get(key)


def fake_score_fn(stems: List[str]):
    def _fn(weights):
        pool = FakePool(stems)
        N = len(stems)
        return pool, {
            "uncertainty": np.linspace(0.0, 1.0, N),
            "temporal": np.linspace(0.1, 0.9, N),
        }
    return _fn


def fake_infer_fn(stems: List[str]):
    def _fn(weights, query_stems):
        return {s: _make_kp(0.4).tolist() for s in query_stems}
    return _fn


_train_call_count = [0]


def _fake_checkpoint(out_dir):
    checkpoint = Path(out_dir) / "mock-best.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"synthetic checkpoint; no model loading")
    return str(checkpoint)


def fake_train_eval_fn():
    def _fn(labeled_stems, labels, out_dir, seed):
        _train_call_count[0] += 1
        return {
            "map50": 0.1 * _train_call_count[0],
            "map50_95": 0.05 * _train_call_count[0],
            "weights": _fake_checkpoint(out_dir),
        }
    return _fn


def fake_gt_lookup(stem: str) -> Optional[np.ndarray]:
    return _make_kp(0.6)


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------

def make_session(
    tmp_path: Path,
    condition: str = "A",
    run_mode: str = "human",
    pool_size: int = POOL_SIZE,
    budget: int = K_BUDGET,
) -> Tuple[ClinicianLoop, SessionState]:
    stems = _make_stems(n=pool_size)
    _train_call_count[0] = 0
    state = SessionState(
        session_id="test_session_001",
        condition=condition,
        seed=42,
        strategy="temporal",
        budget_per_round=budget,
        round=0,
        labeled=[],
        labels={},
        history=[],
        status="active",
    )
    loop = ClinicianLoop(
        state,
        infer_fn=fake_infer_fn(stems),
        score_fn=fake_score_fn(stems),
        train_eval_fn=fake_train_eval_fn(),
        gt_lookup=fake_gt_lookup,
        oracle=None,
        run_mode=run_mode,
        output_root=str(tmp_path / "cl_output"),
        image_dir=str(tmp_path / "images"),
        seg_table=None,
    )
    return loop, state


# ===========================================================================
# Proposal exclusion and exact-budget top-up behavior
# ===========================================================================

class TestProposalExclusionAndTopUp:
    def test_second_propose_yields_different_stems(self, tmp_path):
        """Re-proposing in a round must not return already-seen stems."""
        loop, state = make_session(tmp_path)
        batch1 = loop.propose(n=K_BUDGET)
        stems1 = {c.sample_id for c in batch1}

        batch2 = loop.propose(n=K_BUDGET)
        stems2 = {c.sample_id for c in batch2}

        assert stems1.isdisjoint(stems2), (
            f"Second propose returned already-seen stems: "
            f"{stems1 & stems2}"
        )

    def test_propose_excludes_pending_rejected(self, tmp_path):
        """Rejected stems in the current round must not reappear."""
        loop, state = make_session(tmp_path)
        batch1 = loop.propose(n=2)
        # Reject all from batch1
        decisions = [Decision(sample_id=c.sample_id, action="reject") for c in batch1]
        loop.apply_decisions(decisions)

        batch2 = loop.propose(n=K_BUDGET)
        seen_rejected = {c.sample_id for c in batch1}
        for c in batch2:
            assert c.sample_id not in seen_rejected, (
                f"Rejected stem {c.sample_id} reappeared in the next proposal"
            )

    def test_run_auto_round_commits_exactly_k(self, tmp_path):
        """run_auto_round commits exactly K labels when the pool is large enough."""
        loop, state = make_session(tmp_path, pool_size=POOL_SIZE * 3)
        # All stems are in-rep, no occlusion → oracle always accepts
        def accept_all(c: Candidate) -> Decision:
            return Decision(sample_id=c.sample_id, action="accept")

        rr = loop.run_auto_round(accept_all)
        assert rr.n_accepted == K_BUDGET, (
            f"Expected n_accepted=={K_BUDGET}, got {rr.n_accepted}"
        )
        assert len(state.labeled) == K_BUDGET

    def test_run_auto_round_commits_k_despite_rejections(self, tmp_path):
        """The top-up loop reaches K even when half the batch is rejected."""
        # Use a large pool so there are always fresh frames
        stems = _make_stems(n=60)
        _train_call_count[0] = 0
        state = SessionState(
            session_id="topup_test",
            condition="B",
            seed=0,
            strategy="temporal",
            budget_per_round=K_BUDGET,
            round=0,
            labeled=[],
            labels={},
            history=[],
            status="active",
        )
        # Reject first 2 of every batch, accept the rest
        reject_count = [0]
        def alternating_decision(c: Candidate) -> Decision:
            if reject_count[0] < 2:
                reject_count[0] += 1
                return Decision(sample_id=c.sample_id, action="reject")
            reject_count[0] = 0
            return Decision(sample_id=c.sample_id, action="accept")

        loop = ClinicianLoop(
            state,
            infer_fn=fake_infer_fn(stems),
            score_fn=fake_score_fn(stems),
            train_eval_fn=fake_train_eval_fn(),
            gt_lookup=fake_gt_lookup,
            oracle=None,
            run_mode="auto",
            output_root=str(tmp_path / "topup"),
            image_dir=str(tmp_path / "images"),
        )
        rr = loop.run_auto_round(alternating_decision)
        assert rr.n_accepted == K_BUDGET, (
            f"Need K={K_BUDGET} after top-up, got {rr.n_accepted}"
        )


# ===========================================================================
# Seg-flags: real production stem semantics
#
# Every production train stem is <video_id>_rep00_f<global_frame_idx>.
# The ``rep`` token is ALWAYS ``00`` — a fixed placeholder (8367/8367 stems).
# Matching is entirely on the GLOBAL frame index vs. Segmentation.csv ranges.
# ===========================================================================

# Fixture that mirrors the real PM_109 rows (20 reps, some with occlusion,
# gaps between reps).  Values verified against data/rehab24_6_raw/Segmentation.csv.
_PM109_SEG = (
    "video_id;repetition_number;first_frame;last_frame;"
    "extra_person_in_cam17;extra_person_in_cam18;correctness\n"
    "PM_109;1;185;500;0;0;1\n"
    "PM_109;2;501;770;0;0;1\n"
    "PM_109;3;771;995;0;0;1\n"
    "PM_109;4;996;1185;0;0;1\n"
    "PM_109;5;1186;1370;0;0;1\n"
    # gap 1371-1819 (transition / rest)
    "PM_109;6;1820;2020;2;0;0\n"   # extra_person_in_cam17=2
    "PM_109;7;2021;2235;0;0;1\n"
    "PM_109;8;2236;2408;0;0;1\n"
    "PM_109;9;2409;2606;0;0;1\n"
    "PM_109;10;2607;2762;2;0;0\n"  # extra_person_in_cam17=2
)


class TestRealStemSemantics:
    """Tests using real-style stems (rep00 placeholder) and the real CSV schema.

    Production stems are always <video_id>_rep00_f<global_frame>.
    The rep token is ignored; only the global frame index is matched against
    Segmentation.csv first_frame/last_frame ranges.
    """

    def _df(self, tmp_path: Path, content: str = _PM109_SEG) -> "pd.DataFrame":
        import pandas as pd  # pd already imported at module level in engine
        p = tmp_path / "Segmentation.csv"
        p.write_text(content, encoding="utf-8")
        return load_seg_table(p)

    # --- rep token is ignored: same stem with different rep tokens behave
    #     identically as long as the frame index is the same ---

    def test_rep_token_ignored_rep00_and_rep01_same_frame(self, tmp_path):
        """rep token is a placeholder: rep00 and rep05 on same frame → same result."""
        df = self._df(tmp_path)
        r00 = _seg_flags("PM_109_rep00_f2440", df)
        r05 = _seg_flags("PM_109_rep05_f2440", df)
        assert r00 == r05, (
            f"rep token must be ignored: rep00={r00} != rep05={r05} for same frame"
        )

    # --- PM_109 frame-range regression cases ---

    def test_pm109_f2440_in_rep(self, tmp_path):
        """PM_109_rep00_f2440 → in_rep TRUE (2440 ∈ [2409,2606], rep9)."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_109_rep00_f2440", df)
        assert in_rep is True, f"f2440 must be in-rep (∈ [2409,2606]), got in_rep={in_rep}"

    def test_pm109_f1500_gap_not_in_rep(self, tmp_path):
        """PM_109_rep00_f1500 → in_rep FALSE (1500 is in gap 1371-1819)."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_109_rep00_f1500", df)
        assert in_rep is False, f"f1500 is in the gap [1371,1819], must be out-of-rep, got {in_rep}"

    def test_pm109_f0_before_first_rep_not_in_rep(self, tmp_path):
        """PM_109_rep00_f0 → in_rep FALSE (before first rep at 185)."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_109_rep00_f0000", df)
        assert in_rep is False, f"f0 is before first rep (185), must be out-of-rep, got {in_rep}"

    # --- Boundary: last frame of a rep ---

    def test_pm109_last_frame_of_rep_in_rep(self, tmp_path):
        """Boundary: f1370 is the last frame of rep5 → in_rep TRUE."""
        df = self._df(tmp_path)
        in_rep, _ = _seg_flags("PM_109_rep00_f1370", df)
        assert in_rep is True

    def test_pm109_first_frame_after_gap_in_rep(self, tmp_path):
        """Boundary: f1820 is the first frame of rep6 → in_rep TRUE."""
        df = self._df(tmp_path)
        in_rep, _ = _seg_flags("PM_109_rep00_f1820", df)
        assert in_rep is True

    # --- Occlusion via extra_person_in_cam17 ---

    def test_pm109_occluded_frame_in_rep6(self, tmp_path):
        """f1900 is in rep6 ([1820,2020]) which has extra_person_in_cam17=2 → occluded."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_109_rep00_f1900", df)
        assert in_rep is True, f"f1900 must be in-rep (∈ rep6 [1820,2020])"
        assert occ is True, f"rep6 has extra_person_in_cam17=2, must be occluded"

    def test_pm109_no_occlusion_rep9(self, tmp_path):
        """f2440 is in rep9 which has extra_person_in_cam17=0 → not occluded."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_109_rep00_f2440", df)
        assert occ is False, f"rep9 has extra_person=0, must not be occluded, got occ={occ}"

    # --- load_seg_table: real column name preserved ---

    def test_load_seg_real_columns_present(self, tmp_path):
        """load_seg_table must preserve extra_person_in_cam17/cam18 as numeric."""
        df = self._df(tmp_path)
        assert df is not None
        assert "extra_person_in_cam17" in df.columns
        assert "extra_person_in_cam18" in df.columns
        # Values must be numeric (sum-able)
        assert df["extra_person_in_cam17"].sum() == 4, (
            "PM_109 fixture has two rows with 2 each = 4 total"
        )

    def test_unknown_video_returns_safe_defaults(self, tmp_path):
        """Unknown video_id → in_rep=True, occluded=False (conservative)."""
        df = self._df(tmp_path)
        in_rep, occ = _seg_flags("PM_999_rep00_f0100", df)
        assert in_rep is True
        assert occ is False


# ===========================================================================
# Camera-specific occlusion columns
# ===========================================================================

class TestCameraOcclusionColumns:
    def _seg(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "Segmentation.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_extra_person_in_cam17_fires_for_c17_stem(self, tmp_path):
        """extra_person_in_cam17=1 marks a cam17 stem as occluded."""
        p = self._seg(tmp_path,
            "video_id;repetition_number;first_frame;last_frame;"
            "extra_person_in_cam17;extra_person_in_cam18;correctness\n"
            "PM_000;1;0;200;1;0;1\n"
        )
        df = load_seg_table(p)
        # Stem encodes camera as -c17
        stem = "PM_000-c17-25fps_rep00_f0050"
        in_rep, occ = _seg_flags(stem, df, camera_id="17")
        assert occ is True, "cam17 stem with extra_person_in_cam17=1 should be occluded"

    def test_extra_person_in_cam18_fires_for_c18_stem(self, tmp_path):
        """extra_person_in_cam18=1 marks a cam18 stem as occluded."""
        p = self._seg(tmp_path,
            "video_id;repetition_number;first_frame;last_frame;"
            "extra_person_in_cam17;extra_person_in_cam18;correctness\n"
            "PM_000;1;0;200;0;2;1\n"
        )
        df = load_seg_table(p)
        stem = "PM_000-c18-25fps_rep00_f0050"
        in_rep, occ = _seg_flags(stem, df, camera_id="18")
        assert occ is True, "cam18 stem with extra_person_in_cam18=2 should be occluded"

    def test_zero_extra_person_not_occluded(self, tmp_path):
        """A zero count in both camera columns is not occluded."""
        p = self._seg(tmp_path,
            "video_id;repetition_number;first_frame;last_frame;"
            "extra_person_in_cam17;extra_person_in_cam18;correctness\n"
            "PM_000;1;0;200;0;0;1\n"
        )
        df = load_seg_table(p)
        in_rep, occ = _seg_flags("PM_000_rep00_f0050", df)
        assert occ is False

    def test_camera_from_stem_parses_correctly(self):
        """_camera_from_stem extracts the camera id from dataset-format stems."""
        assert _camera_from_stem("PM_000-c17-25fps_rep00_f0050") == "17"
        assert _camera_from_stem("PM_000-c18-25fps_rep01_f0010") == "18"
        assert _camera_from_stem("PM_000_rep00_f0050") is None

    def test_real_full_schema_csv(self, tmp_path):
        """Load a CSV matching the documented REHAB24-6 schema."""
        full_schema = (
            "video_id;repetition_number;exercise_id;person_id;first_frame;"
            "last_frame;cam17_orientation;mocap_erroneous;exercise_subtype;"
            "lights_on;extra_person_in_cam17;extra_person_in_cam18;correctness\n"
            "PM_000;1;1;PM_000;0;199;frontal;0;full_body;1;0;0;1\n"
            "PM_000;2;1;PM_000;210;399;frontal;0;full_body;1;1;0;0\n"
        )
        p = tmp_path / "Segmentation.csv"
        p.write_text(full_schema, encoding="utf-8")
        df = load_seg_table(p)
        assert df is not None
        assert "rep_index" in df.columns
        assert "extra_person_in_cam17" in df.columns
        assert "extra_person_in_cam18" in df.columns

        # f0100 ∈ [0,199] (rep1 range), extra_person_in_cam17=0 → in_rep, not occluded
        in_rep, occ = _seg_flags("PM_000_rep00_f0100", df, camera_id="17")
        assert in_rep is True
        assert occ is False

        # f0300 ∈ [210,399] (rep2 range), extra_person_in_cam17=1 → in_rep, occluded
        # rep token is ignored (always "rep00" in production)
        in_rep2, occ2 = _seg_flags("PM_000_rep00_f0300", df, camera_id="17")
        assert in_rep2 is True
        assert occ2 is True


# ===========================================================================
# Reference condition: accept-all-GT with no rejections
# ===========================================================================

class TestReferenceCondition:
    def test_ref_auto_round_uses_gt_labels(self, tmp_path):
        """The reference condition corrects every frame to GT."""
        stems = _make_stems(n=POOL_SIZE)
        state = SessionState(
            session_id="ref_test",
            condition="ref",
            seed=0,
            strategy="temporal",
            budget_per_round=K_BUDGET,
            round=0,
            labeled=[],
            labels={},
            history=[],
            status="active",
        )
        # GT-correction oracle: no rejections, corrects to GT
        gt_val = 0.6

        def ref_oracle_decide(c: Candidate) -> Decision:
            return Decision(
                sample_id=c.sample_id,
                action="correct",
                keypoints=fake_gt_lookup(c.sample_id).tolist(),
            )

        loop = ClinicianLoop(
            state,
            infer_fn=fake_infer_fn(stems),
            score_fn=fake_score_fn(stems),
            train_eval_fn=fake_train_eval_fn(),
            gt_lookup=fake_gt_lookup,
            oracle=None,
            run_mode="auto",
            output_root=str(tmp_path / "ref_out"),
            image_dir=str(tmp_path / "images"),
        )
        rr = loop.run_auto_round(ref_oracle_decide)
        assert rr.n_accepted == K_BUDGET
        # All committed labels must equal GT (0.6 * 640)
        gt_kp = _make_kp(gt_val)
        for sid in state.labeled:
            stored = np.asarray(state.labels[sid])
            np.testing.assert_allclose(stored, gt_kp, atol=1e-9,
                                       err_msg=f"ref label for {sid} not GT")

    def test_ref_no_rejections(self, tmp_path):
        """The reference oracle accepts every frame without rejection."""
        loop, state = make_session(tmp_path, condition="ref")
        cands = loop.propose(n=K_BUDGET * 2)
        # Simulate ref oracle (accept all → correct to GT)
        decisions = [
            Decision(
                sample_id=c.sample_id,
                action="correct",
                keypoints=fake_gt_lookup(c.sample_id).tolist(),
            )
            for c in cands
        ]
        result = loop.apply_decisions(decisions)
        assert len(result.rejected) == 0
        assert result.need_more is False


# ===========================================================================
# Clinician tag from run_mode rather than oracle presence
# ===========================================================================

class TestClinicianTag:
    def test_human_mode_cond_b_tagged_human(self, tmp_path):
        """Condition B in human run_mode has the 'human' clinician tag."""
        loop, state = make_session(tmp_path, condition="B", run_mode="human")
        cands = loop.propose(n=K_BUDGET)
        decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
        loop.apply_decisions(decisions)
        rr = loop.commit_and_train()
        assert rr.clinician == "human", (
            f"Human-mode B session should tag 'human', got {rr.clinician!r}"
        )

    def test_auto_mode_cond_b_tagged_simulated(self, tmp_path):
        """Condition B in auto run_mode has the 'simulated' clinician tag."""
        loop, state = make_session(tmp_path, condition="B", run_mode="auto")
        cands = loop.propose(n=K_BUDGET)
        decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
        loop.apply_decisions(decisions)
        rr = loop.commit_and_train()
        assert rr.clinician == "simulated", (
            f"Auto-mode B session should tag 'simulated', got {rr.clinician!r}"
        )

    def test_auto_mode_cond_a_tagged_none(self, tmp_path):
        """Condition A has the 'none' clinician tag in either run mode."""
        for mode in ("human", "auto"):
            loop, state = make_session(tmp_path, condition="A", run_mode=mode)
            cands = loop.propose(n=K_BUDGET)
            decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
            loop.apply_decisions(decisions)
            rr = loop.commit_and_train()
            assert rr.clinician == "none", (
                f"Condition A / mode {mode!r}: expected 'none', got {rr.clinician!r}"
            )
            _train_call_count[0] = 0  # reset for next iteration


# ===========================================================================
# read_gt_label refuses non-training splits
# ===========================================================================

class TestNoValidationLeakage:
    def test_raises_on_val_split(self, tmp_path):
        """read_gt_label raises ValueError for split='val'."""
        with pytest.raises(ValueError, match="split must be 'train'"):
            read_gt_label("any_stem", tmp_path, split="val")

    def test_raises_on_test_split(self, tmp_path):
        with pytest.raises(ValueError, match="split must be 'train'"):
            read_gt_label("any_stem", tmp_path, split="test")

    def test_train_split_still_works(self, tmp_path):
        """The training split is allowed."""
        label_dir = tmp_path / "labels" / "train"
        label_dir.mkdir(parents=True)
        kp_fields = " ".join("0.500000 0.500000 2" for _ in range(26))
        (label_dir / "stem.txt").write_text(f"0 0.5 0.5 0.4 0.4 {kp_fields}\n",
                                             encoding="utf-8")
        kp = read_gt_label("stem", tmp_path, image_w=640, image_h=480, split="train")
        assert kp is not None


# ===========================================================================
# Correction noise is clipped to the image bounds
# ===========================================================================

class TestCorrectionNoiseClipping:
    def test_noise_clipped_within_frame(self):
        """Corrected keypoints stay within [0,w]×[0,h]."""
        W, H = 640, 480
        # GT near the corner → large sigma could push outside
        gt_corner = np.zeros((NUM_KP, 3), dtype=np.float64)
        gt_corner[:, 0] = 5.0    # x near left edge
        gt_corner[:, 1] = 5.0    # y near top edge
        gt_corner[:, 2] = 2.0

        def corner_gt(stem: str) -> np.ndarray:
            return gt_corner.copy()

        clinician = SimulatedClinician(
            corner_gt,
            correct_sigma=100.0,   # very large sigma — would push off-frame without clipping
            seed=0,
            image_w=W,
            image_h=H,
        )
        cand = Candidate(
            sample_id="corner_stem",
            image_path="",
            pred_keypoints=_make_kp(0.0).tolist(),
            term_scores={},
            clinical_flags={"in_labeled_rep": True, "occluded": False, "critical_joints": []},
        )
        dec = clinician.decide(cand)
        assert dec.action == "correct"
        kp = np.asarray(dec.keypoints)
        assert kp[:, 0].min() >= 0.0, "x < 0 after noise"
        assert kp[:, 0].max() <= W,   "x > W after noise"
        assert kp[:, 1].min() >= 0.0, "y < 0 after noise"
        assert kp[:, 1].max() <= H,   "y > H after noise"

    def test_correct_sigma_bounded_6sigma(self):
        """With a moderate sigma, maximum deviation remains below 6*sigma."""
        sigma = 5.0
        clinician = SimulatedClinician(
            fake_gt_lookup,
            correct_sigma=sigma,
            seed=123,
            image_w=640,
            image_h=480,
        )
        gt = _make_kp(0.6)
        cand = Candidate(
            sample_id="noisy_stem",
            image_path="",
            pred_keypoints=_make_kp(0.0).tolist(),
            term_scores={},
            clinical_flags={"in_labeled_rep": True, "occluded": False, "critical_joints": []},
        )
        dec = clinician.decide(cand)
        kp = np.asarray(dec.keypoints)
        diff = np.abs(kp[:, :2] - gt[:, :2])
        assert diff.max() < 6 * sigma + 1e-6


# ===========================================================================
# n_proposed counts unique stems surfaced per round
# ===========================================================================

class TestProposedCount:
    def test_n_proposed_counts_all_unique_stems(self, tmp_path):
        """n_proposed counts every unique stem seen across proposal calls."""
        loop, state = make_session(tmp_path, pool_size=30)
        # First propose: K stems
        batch1 = loop.propose(n=K_BUDGET)
        # Reject all, propose again: K more stems
        decisions = [Decision(sample_id=c.sample_id, action="reject") for c in batch1]
        loop.apply_decisions(decisions)
        batch2 = loop.propose(n=K_BUDGET)
        decisions2 = [Decision(sample_id=c.sample_id, action="accept") for c in batch2]
        loop.apply_decisions(decisions2)

        rr = loop.commit_and_train()
        # Two propose calls of K_BUDGET each → 2*K_BUDGET unique stems surfaced
        assert rr.n_proposed == K_BUDGET * 2, (
            f"Expected n_proposed=={K_BUDGET * 2}, got {rr.n_proposed}"
        )


# ===========================================================================
# Cond A — auto pseudo-label path (original tests, kept and extended)
# ===========================================================================

class TestCondA:
    def test_propose_returns_candidates(self, tmp_path):
        loop, state = make_session(tmp_path, "A")
        cands = loop.propose(n=K_BUDGET * 2)
        assert len(cands) == K_BUDGET * 2
        for c in cands:
            assert isinstance(c, Candidate)
            assert len(c.pred_keypoints) == NUM_KP

    def test_accept_all_pseudo_labels(self, tmp_path):
        loop, state = make_session(tmp_path, "A")
        cands = loop.propose(n=K_BUDGET * 2)
        decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
        result = loop.apply_decisions(decisions)
        assert isinstance(result, CommitResult)
        assert len(result.accepted) == K_BUDGET * 2
        assert len(result.rejected) == 0
        assert result.need_more is False

    def test_commit_grows_labeled_and_history(self, tmp_path):
        loop, state = make_session(tmp_path, "A")
        cands = loop.propose(n=K_BUDGET)
        decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
        loop.apply_decisions(decisions)
        rr = loop.commit_and_train()
        assert rr.round == 0
        assert rr.n_labeled == K_BUDGET
        assert rr.clinician == "none"
        assert len(state.labeled) == K_BUDGET
        assert state.round == 1

    def test_commit_persists_state_json(self, tmp_path):
        loop, state = make_session(tmp_path, "A")
        cands = loop.propose(n=K_BUDGET)
        decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
        loop.apply_decisions(decisions)
        loop.commit_and_train()
        state_json = tmp_path / "cl_output" / "test_session_001" / "state.json"
        assert state_json.exists()
        loaded = json.loads(state_json.read_text())
        assert loaded["session_id"] == "test_session_001"
        assert len(loaded["labeled"]) == K_BUDGET

    def test_curve_grows_after_each_commit(self, tmp_path):
        loop, state = make_session(tmp_path, "A", pool_size=60)
        for _ in range(3):
            cands = loop.propose(n=K_BUDGET)
            decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
            loop.apply_decisions(decisions)
            loop.commit_and_train()
        curve = loop.curve()
        assert len(curve) == 3
        assert curve[0]["n_labeled"] < curve[1]["n_labeled"] < curve[2]["n_labeled"]
        assert curve[0]["map50"] < curve[2]["map50"]

    def test_state_json_valid_after_multiple_rounds(self, tmp_path):
        loop, state = make_session(tmp_path, "A", pool_size=60)
        for _ in range(2):
            cands = loop.propose(n=K_BUDGET)
            decisions = [Decision(sample_id=c.sample_id, action="accept") for c in cands]
            loop.apply_decisions(decisions)
            loop.commit_and_train()
        loaded_state = ClinicianLoop.load_state(
            tmp_path / "cl_output" / "test_session_001"
        )
        assert loaded_state.round == 2
        assert len(loaded_state.history) == 2


# ===========================================================================
# Cond B — SimulatedClinician (original tests, kept + n_accepted==K assertion)
# ===========================================================================

class TestCondBSimulatedClinician:
    def _make_clinician(self, correct_sigma: float = 0.0) -> SimulatedClinician:
        return SimulatedClinician(
            fake_gt_lookup,
            seg_table=None,
            correct_sigma=correct_sigma,
            reject_occluded=True,
            seed=0,
        )

    def test_accept_in_rep_no_occlusion(self):
        clinician = self._make_clinician()
        cand = Candidate(
            sample_id="PM_000_rep00_f0010",
            image_path="",
            pred_keypoints=_make_kp(0.4).tolist(),
            term_scores={},
            clinical_flags={"in_labeled_rep": True, "occluded": False,
                            "critical_joints": []},
        )
        dec = clinician.decide(cand)
        assert dec.action == "correct"
        kp = np.asarray(dec.keypoints)
        np.testing.assert_allclose(kp, _make_kp(0.6), atol=1e-9)

    def test_reject_out_of_rep(self):
        clinician = self._make_clinician()
        cand = Candidate(
            sample_id="stem_oor",
            image_path="",
            pred_keypoints=_make_kp(0.4).tolist(),
            term_scores={},
            clinical_flags={"in_labeled_rep": False, "occluded": False,
                            "critical_joints": []},
        )
        assert clinician.decide(cand).action == "reject"

    def test_reject_occluded(self):
        clinician = self._make_clinician()
        cand = Candidate(
            sample_id="occ_stem",
            image_path="",
            pred_keypoints=_make_kp(0.4).tolist(),
            term_scores={},
            clinical_flags={"in_labeled_rep": True, "occluded": True,
                            "critical_joints": []},
        )
        assert clinician.decide(cand).action == "reject"

    def test_need_more_when_rejections_drop_below_k(self, tmp_path):
        loop, state = make_session(tmp_path, "A")
        cands = loop.propose(n=K_BUDGET)
        decisions = [Decision(sample_id=c.sample_id, action="reject")
                     for c in cands[:-1]]
        decisions.append(Decision(sample_id=cands[-1].sample_id, action="accept"))
        result = loop.apply_decisions(decisions)
        assert result.need_more is True
        assert len(result.accepted) == 1
        assert len(result.rejected) == K_BUDGET - 1

    def test_cond_b_n_accepted_equals_k(self, tmp_path):
        """Condition B run_auto_round commits exactly K labels."""
        stems = _make_stems(n=40)
        state = SessionState(
            session_id="b_k_test",
            condition="B",
            seed=0,
            strategy="temporal",
            budget_per_round=K_BUDGET,
            round=0,
            labeled=[],
            labels={},
            history=[],
            status="active",
        )
        oracle = SimulatedClinician(fake_gt_lookup, seg_table=None, seed=0)
        loop = ClinicianLoop(
            state,
            infer_fn=fake_infer_fn(stems),
            score_fn=fake_score_fn(stems),
            train_eval_fn=fake_train_eval_fn(),
            gt_lookup=fake_gt_lookup,
            oracle=oracle,
            run_mode="auto",
            output_root=str(tmp_path / "b_k"),
            image_dir=str(tmp_path / "images"),
            seg_table=None,
        )
        rr = loop.run_auto_round(oracle.decide)
        assert rr.n_accepted == K_BUDGET, (
            f"Expected n_accepted=={K_BUDGET}, got {rr.n_accepted}"
        )
        assert len(state.labeled) == K_BUDGET

    def test_cond_b_committed_labels_are_gt(self, tmp_path):
        """Labels stored in state for cond B must equal GT (sigma=0)."""
        stems = _make_stems(n=40)
        state = SessionState(
            session_id="b_gt_test",
            condition="B",
            seed=0,
            strategy="temporal",
            budget_per_round=K_BUDGET,
            round=0,
            labeled=[],
            labels={},
            history=[],
            status="active",
        )
        oracle = SimulatedClinician(fake_gt_lookup, seg_table=None, seed=0)
        loop = ClinicianLoop(
            state,
            infer_fn=fake_infer_fn(stems),
            score_fn=fake_score_fn(stems),
            train_eval_fn=fake_train_eval_fn(),
            gt_lookup=fake_gt_lookup,
            oracle=oracle,
            run_mode="auto",
            output_root=str(tmp_path / "b_gt"),
            image_dir=str(tmp_path / "images"),
            seg_table=None,
        )
        loop.run_auto_round(oracle.decide)
        gt = _make_kp(0.6)
        for sid in state.labeled:
            np.testing.assert_allclose(
                np.asarray(state.labels[sid]), gt, atol=1e-9,
                err_msg=f"label for {sid} should be GT"
            )


# ===========================================================================
# I1 — OKS-gated selective-correction branch (B differs from always-correct B)
# ===========================================================================

def _spread_kp(offset: float = 0.0) -> np.ndarray:
    """(26,3) keypoints spread over a real ~200x200 bbox (non-degenerate area).

    A meaningful bbox area is required so that OKS actually discriminates near
    vs. far predictions (a degenerate zero-area bbox falls back to the whole
    frame and keeps OKS artificially high).
    """
    kp = np.zeros((NUM_KP, 3), dtype=np.float64)
    kp[:, 0] = np.linspace(100.0, 300.0, NUM_KP) + offset
    kp[:, 1] = np.linspace(100.0, 300.0, NUM_KP) + offset
    kp[:, 2] = 2.0
    return kp


def _verify_cand(pred: np.ndarray, sid: str = "PM_000_rep00_f0010") -> Candidate:
    return Candidate(
        sample_id=sid,
        image_path="",
        pred_keypoints=pred.tolist(),
        term_scores={},
        clinical_flags={"in_labeled_rep": True, "occluded": False,
                        "critical_joints": []},
    )


class TestI1VerifyBranch:
    def test_oks_similarity_identical_is_high(self):
        """_oks_similarity of identical keypoints ≈ 1.0."""
        gt = _spread_kp()
        assert _oks_similarity(gt.copy(), gt) > 0.99

    def test_oks_similarity_far_is_low(self):
        """_oks_similarity of a large-offset prediction is near 0."""
        gt = _spread_kp()
        pred = _spread_kp(offset=100.0)
        assert _oks_similarity(pred, gt) < 0.1

    def test_none_threshold_preserves_legacy_correct(self):
        """verify_oks_threshold=None → always correct to GT (legacy)."""
        gt = _spread_kp()
        # Prediction is far from GT; without verify it must still be corrected.
        clinician = SimulatedClinician(
            lambda s: gt.copy(), correct_sigma=0.0, seed=0,
            verify_oks_threshold=None,
        )
        dec = clinician.decide(_verify_cand(_spread_kp(offset=100.0)))
        assert dec.action == "correct"
        np.testing.assert_allclose(np.asarray(dec.keypoints), gt, atol=1e-9)

    def test_high_oks_verified_adopts_prediction(self):
        """OKS ≥ threshold → action 'accept', prediction adopted UNCHANGED.

        correct_sigma is large: if the frame had gone down the correct path the
        stored keypoints would be GT + noise, not the (near-GT) prediction.
        """
        gt = _spread_kp()
        pred = _spread_kp(offset=2.0)  # tiny offset → OKS high
        clinician = SimulatedClinician(
            lambda s: gt.copy(), correct_sigma=50.0, seed=0,
            verify_oks_threshold=0.5,
        )
        dec = clinician.decide(_verify_cand(pred))
        assert dec.action == "accept"
        # Prediction adopted verbatim (NOT overwritten by GT, NOT noised).
        np.testing.assert_allclose(np.asarray(dec.keypoints), pred, atol=1e-9)

    def test_low_oks_falls_back_to_gt_correction(self):
        """OKS < threshold → action 'correct' toward GT (sigma=0 → exact GT)."""
        gt = _spread_kp()
        pred = _spread_kp(offset=100.0)  # large offset → OKS low
        clinician = SimulatedClinician(
            lambda s: gt.copy(), correct_sigma=0.0, seed=0,
            verify_oks_threshold=0.5,
        )
        dec = clinician.decide(_verify_cand(pred))
        assert dec.action == "correct"
        np.testing.assert_allclose(np.asarray(dec.keypoints), gt, atol=1e-9)

    def test_verify_still_rejects_out_of_rep_and_occluded(self):
        """Selective-correction mode must not bypass rejection rules."""
        gt = _spread_kp()
        clinician = SimulatedClinician(
            lambda s: gt.copy(), seed=0, verify_oks_threshold=0.5,
        )
        oor = _verify_cand(gt.copy())
        oor.clinical_flags["in_labeled_rep"] = False
        assert clinician.decide(oor).action == "reject"

        occ = _verify_cand(gt.copy())
        occ.clinical_flags["occluded"] = True
        assert clinician.decide(occ).action == "reject"

    def test_verify_default_signature_unchanged(self):
        """Default construction (no verify kwarg) keeps threshold None."""
        clinician = SimulatedClinician(fake_gt_lookup, seed=0)
        assert clinician.verify_oks_threshold is None


# ===========================================================================
# GT label IO helpers (kept + extended)
# ===========================================================================

class TestReadGtLabel:
    def test_reads_normalized_label_denormalized(self, tmp_path):
        label_dir = tmp_path / "labels" / "train"
        label_dir.mkdir(parents=True)
        kp_fields = " ".join("0.500000 0.500000 2" for _ in range(26))
        (label_dir / "test_frame.txt").write_text(
            f"0 0.5 0.5 0.4 0.4 {kp_fields}\n", encoding="utf-8"
        )
        kp = read_gt_label("test_frame", tmp_path, image_w=800, image_h=600)
        assert kp is not None
        assert kp.shape == (26, 3)
        np.testing.assert_allclose(kp[:, 0], 400.0, atol=1e-3)
        np.testing.assert_allclose(kp[:, 1], 300.0, atol=1e-3)

    def test_returns_none_when_file_missing(self, tmp_path):
        assert read_gt_label("nonexistent", tmp_path) is None

    def test_ignores_empty_file(self, tmp_path):
        d = tmp_path / "labels" / "train"
        d.mkdir(parents=True)
        (d / "empty.txt").write_text("", encoding="utf-8")
        assert read_gt_label("empty", tmp_path) is None


class TestWriteLabelsDir:
    def test_writes_label_files(self, tmp_path):
        stems = [f"frame_{i:04d}" for i in range(5)]
        labels = {s: _make_kp(0.5, w=640, h=480).tolist() for s in stems}
        write_labels_dir(labels, tmp_path, image_sizes={s: (640, 480) for s in stems})
        label_dir = tmp_path / "labels" / "train"
        for s in stems:
            content = (label_dir / f"{s}.txt").read_text()
            tokens = content.strip().split()
            assert len(tokens) == 83
            assert tokens[0] == "0"

    def test_skips_malformed_keypoints(self, tmp_path):
        write_labels_dir({"bad_stem": [[0.5, 0.5]]}, tmp_path)
        assert not (tmp_path / "labels" / "train" / "bad_stem.txt").exists()


# ===========================================================================
# load_seg_table (basic)
# ===========================================================================

class TestLoadSegTable:
    def test_returns_none_when_path_is_none(self):
        assert load_seg_table(None) is None

    def test_returns_none_when_file_missing(self, tmp_path):
        assert load_seg_table(tmp_path / "no_such_file.csv") is None

    def test_seg_flags_default_when_stem_not_found(self, tmp_path):
        p = tmp_path / "Segmentation.csv"
        p.write_text(
            "video_id;repetition_number;first_frame;last_frame;correctness\n"
            "PM_999;1;0;100;1\n",
            encoding="utf-8",
        )
        df = load_seg_table(p)
        in_rep, occ = _seg_flags("PM_000_rep00_f0050", df)
        assert in_rep is True
        assert occ is False


# ===========================================================================
# Import-leak guard
# ===========================================================================

def test_no_heavy_imports_at_module_load():
    import sys
    heavy = {"torch", "cv2", "ultralytics", "fastapi"}
    leaked = heavy & set(sys.modules)
    assert not leaked, f"Heavy imports leaked: {leaked}"


# ===========================================================================
# SessionState serialization roundtrip
# ===========================================================================

# ===========================================================================
# YOLO label-pairing regression (GPU-free)
#
# The critical YOLO invariant: for every image path written into train.txt,
# YOLO derives the label path by replacing the FIRST occurrence of /images/
# (or \images\) with /labels/ and changing the extension to .txt.
# If train.txt lists canonical IMAGES_TRAIN paths the substitution resolves
# to the prebuilt GT, silently ignoring corrected labels.
# This test instantiates a spy train_eval_fn that performs the same path
# substitution YOLO would do and asserts the resulting label file contains
# the corrected keypoints, not the prebuilt GT.
# ===========================================================================

class TestYoloLabelPairing:
    """Regression: train.txt must list LOCAL image copies so YOLO's
    /images/→/labels/ substitution resolves to the round's written labels,
    not the prebuilt GT under the canonical dataset root.
    """

    def _make_spy_train_eval_fn(self, corrected_labels: Dict[str, List]):
        """Return a train_eval_fn that verifies images↔labels pairing."""
        findings: Dict[str, Any] = {"errors": []}

        def train_eval_fn(labeled_stems: List[str], labels: Dict[str, List],
                          out_dir: str, seed: int) -> dict:
            import re as _re
            out = Path(out_dir)
            # Locate train.txt written by the engine/server
            train_txt = out / "train.txt"
            if not train_txt.exists():
                findings["errors"].append("train.txt not written")
                return {"map50": 0.0, "map50_95": 0.0, "weights": _fake_checkpoint(out_dir)}

            image_lines = [
                l.strip() for l in train_txt.read_text(encoding="utf-8").splitlines()
                if l.strip()
            ]
            for img_path_str in image_lines:
                img_path = Path(img_path_str)
                stem = img_path.stem

                # Reproduce YOLO label resolution: swap /images/ → /labels/,
                # change extension .jpg → .txt
                # Use the same substitution on the path string (first occurrence,
                # case-insensitive on Windows).
                label_path_str = _re.sub(
                    r"(?i)[/\\]images[/\\]",
                    lambda m: m.group(0).replace(
                        m.group(0).lstrip("/\\").rstrip("/\\"), "labels"
                    ),
                    img_path_str,
                    count=1,
                )
                # Replace separator chars properly
                label_path_str = img_path_str
                parts = list(img_path.parts)
                try:
                    img_idx = next(
                        i for i, p in enumerate(parts)
                        if p.lower() == "images"
                    )
                    parts[img_idx] = "labels"
                    label_path = Path(*parts).with_suffix(".txt")
                except StopIteration:
                    findings["errors"].append(
                        f"'images' not found in path {img_path_str!r}"
                    )
                    continue

                if not label_path.exists():
                    findings["errors"].append(
                        f"label file does not exist: {label_path}"
                    )
                    continue

                # Read the label file and parse its keypoints
                raw = label_path.read_text(encoding="utf-8").strip()
                if not raw:
                    findings["errors"].append(f"empty label file: {label_path}")
                    continue

                tokens = raw.split()
                # YOLO-pose format: cls cx cy bw bh kp0x kp0y kp0v ...
                # normalized [0,1]; 5 bbox tokens + 3*K kp tokens
                if len(tokens) < 5 + 3:
                    findings["errors"].append(
                        f"too few tokens in {label_path}: {len(tokens)}"
                    )
                    continue

                # Verify keypoints in the label match what commit_and_train stored
                # in state.labels (which comes from corrected_labels fixture).
                if stem in corrected_labels:
                    expected_kp = np.asarray(corrected_labels[stem], dtype=np.float64)
                    # tokens[5:] are normalized x y v triplets
                    kp_tokens = list(map(float, tokens[5:]))
                    n_kp = len(kp_tokens) // 3
                    actual_norm = np.array(kp_tokens, dtype=np.float64).reshape(n_kp, 3)
                    # The label file is normalized; expected_kp is in pixels (640x640).
                    # Reconstruct pixel coords with the default 640x640 fallback.
                    actual_px_x = actual_norm[:, 0] * 640.0
                    actual_px_y = actual_norm[:, 1] * 640.0
                    exp_x = expected_kp[:, 0]
                    exp_y = expected_kp[:, 1]
                    # They must match within 1 pixel (rounding from normalization).
                    max_err_x = float(np.abs(actual_px_x - exp_x).max())
                    max_err_y = float(np.abs(actual_px_y - exp_y).max())
                    if max_err_x > 1.5 or max_err_y > 1.5:
                        findings["errors"].append(
                            f"label mismatch for {stem}: "
                            f"max_err_x={max_err_x:.3f}, max_err_y={max_err_y:.3f}"
                        )

            return {"map50": 0.0, "map50_95": 0.0, "weights": _fake_checkpoint(out_dir)}

        return train_eval_fn, findings

    def _make_fake_image(self, path: Path) -> None:
        """Write a minimal 1x1 white JPEG without cv2/PIL."""
        # Minimal valid JPEG bytes for a 1x1 white pixel.
        jpeg_bytes = bytes([
            0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00,
            0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xDB,
            0x00, 0x43, 0x00, 0x08, 0x06, 0x06, 0x07, 0x06, 0x05, 0x08, 0x07,
            0x07, 0x07, 0x09, 0x09, 0x08, 0x0A, 0x0C, 0x14, 0x0D, 0x0C, 0x0B,
            0x0B, 0x0C, 0x19, 0x12, 0x13, 0x0F, 0x14, 0x1D, 0x1A, 0x1F, 0x1E,
            0x1D, 0x1A, 0x1C, 0x1C, 0x20, 0x24, 0x2E, 0x27, 0x20, 0x22, 0x2C,
            0x23, 0x1C, 0x1C, 0x28, 0x37, 0x29, 0x2C, 0x30, 0x31, 0x34, 0x34,
            0x34, 0x1F, 0x27, 0x39, 0x3D, 0x38, 0x32, 0x3C, 0x2E, 0x33, 0x34,
            0x32, 0xFF, 0xC0, 0x00, 0x0B, 0x08, 0x00, 0x01, 0x00, 0x01, 0x01,
            0x01, 0x11, 0x00, 0xFF, 0xC4, 0x00, 0x1F, 0x00, 0x00, 0x01, 0x05,
            0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
            0x09, 0x0A, 0x0B, 0xFF, 0xC4, 0x00, 0xB5, 0x10, 0x00, 0x02, 0x01,
            0x03, 0x03, 0x02, 0x04, 0x03, 0x05, 0x05, 0x04, 0x04, 0x00, 0x00,
            0x01, 0x7D, 0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21,
            0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07, 0x22, 0x71, 0x14, 0x32,
            0x81, 0x91, 0xA1, 0x08, 0x23, 0x42, 0xB1, 0xC1, 0x15, 0x52, 0xD1,
            0xF0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0A, 0x16, 0x17, 0x18,
            0x19, 0x1A, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2A, 0x34, 0x35, 0x36,
            0x37, 0x38, 0x39, 0x3A, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
            0x4A, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5A, 0x63, 0x64,
            0x65, 0x66, 0x67, 0x68, 0x69, 0x6A, 0x73, 0x74, 0x75, 0x76, 0x77,
            0x78, 0x79, 0x7A, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8A,
            0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99, 0x9A, 0xA2, 0xA3,
            0xA4, 0xA5, 0xA6, 0xA7, 0xA8, 0xA9, 0xAA, 0xB2, 0xB3, 0xB4, 0xB5,
            0xB6, 0xB7, 0xB8, 0xB9, 0xBA, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
            0xC8, 0xC9, 0xCA, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8, 0xD9,
            0xDA, 0xE1, 0xE2, 0xE3, 0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA,
            0xF1, 0xF2, 0xF3, 0xF4, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFF,
            0xDA, 0x00, 0x08, 0x01, 0x01, 0x00, 0x00, 0x3F, 0x00, 0xFB, 0xDB,
            0xFF, 0xD9,
        ])
        path.write_bytes(jpeg_bytes)

    def test_label_file_paired_with_image_path_in_train_txt(self, tmp_path):
        """Core regression: YOLO's /images/→/labels/ substitution on the paths
        written in train.txt must resolve to the corrected labels file,
        not to any pre-existing GT in a different directory tree.
        """
        import os as _os

        # Build a fake session with 2 stems and corrected keypoints (GT-like).
        stems = ["PM_000_rep00_f0000", "PM_000_rep00_f0001"]
        corrected_kp: Dict[str, List] = {
            s: _make_kp(0.7).tolist() for s in stems  # distinct from pseudo (0.4)
        }

        # Create fake image files under tmp_path/images/train/ (no cv2 needed).
        img_dir = tmp_path / "images" / "train"
        img_dir.mkdir(parents=True, exist_ok=True)
        for sid in stems:
            self._make_fake_image(img_dir / f"{sid}.jpg")

        # Build a spy train_eval_fn that verifies the pairing.
        spy_fn, findings = self._make_spy_train_eval_fn(corrected_kp)

        # The engine writes labels and then calls train_eval_fn.
        # We replicate what commit_and_train does here with the spy:
        #   1. write_labels_dir(corrected_kp, out_path)
        #   2. materialize images into out_path/images/train/ (hardlink/copy)
        #   3. write train.txt listing those LOCAL image paths
        #   4. call spy_fn
        import os
        import shutil

        out_path = tmp_path / "round_00"
        out_path.mkdir(parents=True, exist_ok=True)

        image_sizes = {sid: (640, 640) for sid in stems}
        write_labels_dir(corrected_kp, out_path, image_sizes=image_sizes)

        local_img_dir = out_path / "images" / "train"
        local_img_dir.mkdir(parents=True, exist_ok=True)
        local_paths = []
        for sid in stems:
            src = img_dir / f"{sid}.jpg"
            dst = local_img_dir / f"{sid}.jpg"
            if not dst.exists():
                try:
                    os.link(str(src), str(dst))
                except OSError:
                    shutil.copyfile(str(src), str(dst))
            local_paths.append(str(dst))

        train_txt = out_path / "train.txt"
        train_txt.write_text("\n".join(local_paths) + "\n", encoding="utf-8")

        spy_fn(stems, corrected_kp, str(out_path), seed=0)

        assert not findings["errors"], (
            "YOLO label-pairing errors:\n" + "\n".join(findings["errors"])
        )

    def test_canonical_path_would_miss_corrected_labels(self, tmp_path):
        """Anti-regression: if train.txt lists the CANONICAL images/ path,
        the /images/→/labels/ substitution resolves to the WRONG directory
        (not the round's labels/). This test confirms the old bug would fail.
        """
        stems = ["PM_000_rep00_f0000"]
        corrected_kp = {stems[0]: _make_kp(0.7).tolist()}

        # Canonical root (separate from the round output dir).
        canonical_root = tmp_path / "canonical"
        canonical_img_dir = canonical_root / "images" / "train"
        canonical_img_dir.mkdir(parents=True, exist_ok=True)
        self._make_fake_image(canonical_img_dir / f"{stems[0]}.jpg")

        # Round output dir has labels written to it.
        out_path = tmp_path / "round_00"
        out_path.mkdir(parents=True, exist_ok=True)
        write_labels_dir(corrected_kp, out_path, image_sizes={stems[0]: (640, 640)})

        # Buggy behavior: train.txt lists the CANONICAL image path.
        # YOLO substitution → canonical_root/labels/train/<stem>.txt
        # That file does NOT exist (we only wrote round_00/labels/train/).
        canonical_img_path = str((canonical_img_dir / f"{stems[0]}.jpg").resolve())
        train_txt = out_path / "train.txt"
        train_txt.write_text(canonical_img_path + "\n", encoding="utf-8")

        # Spy checks: the label file YOLO would find from canonical path is absent.
        spy_fn, findings = self._make_spy_train_eval_fn(corrected_kp)
        spy_fn(stems, corrected_kp, str(out_path), seed=0)

        # We expect an error: label file missing under canonical_root/labels/train/
        assert any("label file does not exist" in e or "not written" in e
                   for e in findings["errors"]), (
            "Anti-regression check failed: canonical path should miss the label file, "
            f"but no error was reported. findings={findings}"
        )


class TestSessionStateSerialization:
    def test_to_dict_from_dict_roundtrip(self):
        rr = RoundRecord(
            round=0, n_labeled=5, pose_map50=0.3, pose_map50_95=0.15,
            n_proposed=10, n_accepted=5, n_rejected=3, n_corrected=2,
            clinician="simulated",
        )
        state = SessionState(
            session_id="test_rt", condition="B", seed=1, strategy="rehabal",
            budget_per_round=10, round=1,
            labeled=["stem_0", "stem_1"],
            labels={"stem_0": _make_kp(0.5).tolist()},
            history=[rr], status="active",
        )
        restored = SessionState.from_dict(state.to_dict())
        assert restored.session_id == state.session_id
        assert len(restored.history) == 1
        assert restored.history[0].clinician == "simulated"
        assert len(restored.labeled) == 2
