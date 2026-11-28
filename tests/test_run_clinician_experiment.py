from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rehabal_pose.review.clinician_loop import Candidate, Decision  # noqa: E402

import run_clinician_experiment as rce  # noqa: E402

NUM_KP = 26


def _kp(xy_frac: float = 0.6, score: float = 2.0, w: int = 640, h: int = 480) -> np.ndarray:
    kp = np.zeros((NUM_KP, 3), dtype=np.float64)
    kp[:, 0] = xy_frac * w
    kp[:, 1] = xy_frac * h
    kp[:, 2] = score
    return kp


_GT = _kp(0.6)


def _gt_lookup(sid: str) -> Optional[np.ndarray]:
    return _GT.copy()


def _gt_lookup_none(sid: str) -> Optional[np.ndarray]:
    return None


def _cand(
    sid: str = "PM_000_rep00_f0010",
    *,
    in_rep: bool = True,
    occluded: bool = False,
    pred_score: float = 0.9,
    pred_kp: Optional[list] = None,
) -> Candidate:
    if pred_kp is None:
        pred_kp = _kp(0.4, score=pred_score).tolist()
    return Candidate(
        sample_id=sid,
        image_path="",
        pred_keypoints=pred_kp,
        term_scores={},
        clinical_flags={"in_labeled_rep": in_rep, "occluded": occluded,
                        "critical_joints": []},
    )


def _make_stems(n: int) -> List[str]:
    return [f"PM_000_rep00_f{i:04d}" for i in range(n)]


class TestExistingConditionsUnchanged:
    def test_A_accepts_pseudo_ignores_flags(self):
        fn = rce._decision_fn_for("A", None, _gt_lookup)
        d = fn(_cand())
        assert d.action == "accept" and d.keypoints is None
        d2 = fn(_cand(in_rep=False, occluded=True))
        assert d2.action == "accept" and d2.keypoints is None

    def test_ref_corrects_to_gt_no_rejection(self):
        fn = rce._decision_fn_for("ref", None, _gt_lookup)
        d = fn(_cand(in_rep=False, occluded=True))
        assert d.action == "correct"
        np.testing.assert_allclose(np.asarray(d.keypoints), _GT, atol=1e-9)

    def test_ref_gt_none_falls_back_to_accept(self):
        fn = rce._decision_fn_for("ref", None, _gt_lookup_none)
        d = fn(_cand())
        assert d.action == "accept" and d.keypoints is None

    def test_B_returns_oracle_decide(self):
        sentinel = object()

        class _Oracle:
            def decide(self, c):
                return sentinel

        oracle = _Oracle()
        fn = rce._decision_fn_for("B", oracle, _gt_lookup)
        assert getattr(fn, "__self__", None) is oracle
        assert fn(_cand()) is sentinel


class TestI2Decomposition:
    def test_A_filter_accepts_in_rep_pseudo(self):
        fn = rce._decision_fn_for("A_filter", None, _gt_lookup)
        d = fn(_cand(in_rep=True, occluded=False))
        assert d.action == "accept"
        assert d.keypoints is None

    def test_A_filter_rejects_out_of_rep(self):
        fn = rce._decision_fn_for("A_filter", None, _gt_lookup)
        assert fn(_cand(in_rep=False)).action == "reject"

    def test_A_filter_rejects_occluded(self):
        fn = rce._decision_fn_for("A_filter", None, _gt_lookup)
        assert fn(_cand(in_rep=True, occluded=True)).action == "reject"

    def test_B_reject_only_matches_A_filter(self):
        fa = rce._decision_fn_for("A_filter", None, _gt_lookup)
        fb = rce._decision_fn_for("B_reject_only", None, _gt_lookup)
        for c in [
            _cand(in_rep=True, occluded=False),
            _cand(in_rep=False, occluded=False),
            _cand(in_rep=True, occluded=True),
            _cand(in_rep=False, occluded=True),
        ]:
            da, db = fa(c), fb(c)
            assert da.action == db.action
            assert da.keypoints == db.keypoints

    def test_ref_filter_corrects_in_rep_to_gt(self):
        fn = rce._decision_fn_for("ref_filter", None, _gt_lookup)
        d = fn(_cand(in_rep=True, occluded=False))
        assert d.action == "correct"
        np.testing.assert_allclose(np.asarray(d.keypoints), _GT, atol=1e-9)

    def test_ref_filter_rejects_out_of_rep_and_occluded(self):
        fn = rce._decision_fn_for("ref_filter", None, _gt_lookup)
        assert fn(_cand(in_rep=False)).action == "reject"
        assert fn(_cand(in_rep=True, occluded=True)).action == "reject"

    def test_ref_filter_gt_none_falls_back_to_accept(self):
        fn = rce._decision_fn_for("ref_filter", None, _gt_lookup_none)
        assert fn(_cand(in_rep=True)).action == "accept"

    def test_B_correct_only_corrects_even_out_of_rep(self):
        fn = rce._decision_fn_for("B_correct_only", None, _gt_lookup)
        d = fn(_cand(in_rep=False, occluded=True))
        assert d.action == "correct"
        np.testing.assert_allclose(np.asarray(d.keypoints), _GT, atol=1e-9)

    def test_B_correct_only_matches_ref_behaviour(self):
        f_ref = rce._decision_fn_for("ref", None, _gt_lookup)
        f_bco = rce._decision_fn_for("B_correct_only", None, _gt_lookup)
        for c in [_cand(in_rep=True), _cand(in_rep=False, occluded=True)]:
            dr, db = f_ref(c), f_bco(c)
            assert dr.action == db.action == "correct"
            assert dr.keypoints == db.keypoints

    def test_2x2_cells_out_of_rep_frame(self):
        c = _cand(in_rep=False, occluded=False)
        assert rce._decision_fn_for("A", None, _gt_lookup)(c).action == "accept"
        assert rce._decision_fn_for("A_filter", None, _gt_lookup)(c).action == "reject"
        assert rce._decision_fn_for("B_correct_only", None, _gt_lookup)(c).action == "correct"
        assert rce._decision_fn_for("ref_filter", None, _gt_lookup)(c).action == "reject"


class TestI3ConfidenceGated:
    def test_pred_confidence_is_mean_score(self):
        c = _cand(pred_score=0.7)
        assert rce._pred_confidence(c) == pytest.approx(0.7)

    def test_pred_confidence_degenerate_shapes_zero(self):
        assert rce._pred_confidence(_cand(pred_kp=[])) == 0.0
        assert rce._pred_confidence(_cand(pred_kp=[[0.1, 0.2]])) == 0.0

    def test_default_threshold_accepts_all_like_A(self):
        fn = rce._decision_fn_for("A_conf", None, _gt_lookup, conf_threshold=None)
        assert fn(_cand(pred_score=0.0)).action == "accept"
        assert fn(_cand(pred_score=0.9)).action == "accept"

    def test_threshold_gates_low_confidence(self):
        fn = rce._decision_fn_for("A_conf", None, _gt_lookup, conf_threshold=0.5)
        assert fn(_cand(pred_score=0.9)).action == "accept"
        assert fn(_cand(pred_score=0.2)).action == "reject"

    def test_threshold_boundary_is_inclusive(self):
        fn = rce._decision_fn_for("A_conf", None, _gt_lookup, conf_threshold=0.5)
        assert fn(_cand(pred_score=0.5)).action == "accept"

    def test_accepted_frames_keep_pseudo(self):
        fn = rce._decision_fn_for("A_conf", None, _gt_lookup, conf_threshold=0.5)
        d = fn(_cand(pred_score=0.9))
        assert d.action == "accept" and d.keypoints is None

    def test_fallback_pseudo_rejected_under_positive_threshold(self):
        fn = rce._decision_fn_for("A_conf", None, _gt_lookup, conf_threshold=0.5)
        d = fn(_cand(pred_kp=[[0.0, 0.0, 0.0]] * NUM_KP))
        assert d.action == "reject"


class _Sentinel(Exception):
    pass


class _SpyClinician:
    last_kwargs: Optional[dict] = None

    def __init__(self, gt_lookup, **kwargs):
        type(self).last_kwargs = dict(kwargs)
        self.gt_lookup = gt_lookup

    def decide(self, c):  # pragma: no cover - never reached (loop raises first)
        return Decision(sample_id=c.sample_id, action="accept")


class _BoomLoop:
    def __init__(self, *a, **k):
        raise _Sentinel()


def _dummy_callables():
    return (lambda *a, **k: {}, lambda *a, **k: (None, {}),
            lambda *a, **k: {}, _gt_lookup)


class TestI1VerifyWiring:
    def test_verify_forwarded_when_set(self, monkeypatch, tmp_path):
        _SpyClinician.last_kwargs = None
        monkeypatch.setattr(rce, "SimulatedClinician", _SpyClinician)
        monkeypatch.setattr(rce, "ClinicianLoop", _BoomLoop)
        with pytest.raises(_Sentinel):
            rce._run_session("B", 0, 0.0, [0.05, 0.10],
                             _make_stems(100), 100, _dummy_callables(), None,
                             tmp_path, "temporal",
                             conf_threshold=None, verify_oks_threshold=0.6)
        assert _SpyClinician.last_kwargs is not None
        assert _SpyClinician.last_kwargs.get("verify_oks_threshold") == 0.6

    def test_verify_not_forwarded_when_none(self, monkeypatch, tmp_path):
        _SpyClinician.last_kwargs = None
        monkeypatch.setattr(rce, "SimulatedClinician", _SpyClinician)
        monkeypatch.setattr(rce, "ClinicianLoop", _BoomLoop)
        with pytest.raises(_Sentinel):
            rce._run_session("B", 0, 0.0, [0.05, 0.10],
                             _make_stems(100), 100, _dummy_callables(), None,
                             tmp_path, "temporal",
                             conf_threshold=None, verify_oks_threshold=None)
        assert _SpyClinician.last_kwargs is not None
        assert "verify_oks_threshold" not in _SpyClinician.last_kwargs


class TestArgparseAcceptsNewFlags:
    def test_new_flags_parse_and_grid_guard_returns_early(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rce, "IMAGES_TRAIN", tmp_path / "no_images")
        rc = rce.main(["--conditions", "A_conf", "--conf-threshold", "0.7",
                       "--verify-oks-threshold", "0.5"])
        assert rc == 1

    def test_baseline_invocation_still_parses(self, monkeypatch, tmp_path):
        monkeypatch.setattr(rce, "IMAGES_TRAIN", tmp_path / "no_images")
        assert rce.main(["--conditions", "A", "B", "ref"]) == 1
