"""RehabAL composite acquisition strategy.

Combines six clinically-motivated, numerically-defined informativeness terms
into a single weighted score, then applies an MMR / k-center diversity filter
in :meth:`select`. Every term is a documented function over the :class:`Pool`
that is min-max normalized to ``[0, 1]``; the weighted sum (weights normalized
to sum to 1) is therefore itself in ``[0, 1]``.

Terms
-----
U  uncertainty        — ``1 - mean keypoint confidence`` (least-confidence).
T  temporal           — within-clip motion (velocity/accel/jerk) anomaly.
A  angle sensitivity  — geometric sensitivity of joint angles to keypoint
                        noise, ``~ sum_j (1/len(limb)) ``; short/foreshortened
                        limbs make angle estimates fragile.
R  ROM criticality    — proximity of joint angles to range-of-motion extremes
                        (full flexion/extension), ``|cos(angle)|``; extremes are
                        the clinically scored part of a rehab exercise.
C  compensation risk  — left/right joint-angle asymmetry, a proxy for
                        compensatory movement patterns.
D  domain rarity      — inverse frequency of the (disease, exercise, camera)
                        metadata cell; rare domains are under-represented.

The default skeleton indexing targets the AIHub-219 16-keypoint layout but the
math is robust to any ``K`` (joints referencing out-of-range indices are
skipped).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool, minmax_normalize
from .temporal import TemporalStrategy
from .uncertainty import UncertaintyStrategy

__all__ = ["RehabALStrategy", "DEFAULT_WEIGHTS"]


# --- Default skeleton topology (REHAB24-6 "rehab26" 26-keypoint order) -------
# CONFIRMED against the real Zenodo joints_names.txt (record 13305826):
# 0 Hips, 1 Spine, 2 Spine1, 3 Neck, 4 Head, 5 Head_end,
# 6 LeftShoulder, 7 LeftArm, 8 LeftForeArm, 9 LeftHand, 10 LeftHand_end,
# 11 RightShoulder, 12 RightArm, 13 RightForeArm, 14 RightHand, 15 RightHand_end,
# 16 LeftUpLeg, 17 LeftLeg, 18 LeftFoot, 19 LeftToeBase, 20 LeftToeBase_end,
# 21 RightUpLeg, 22 RightLeg, 23 RightFoot, 24 RightToeBase, 25 RightToeBase_end.
# (Previously these were the AIHub-16 layout — a critical bug: the clinical
# angle/ROM/compensation terms were computed on the wrong joints. The strategy
# now also accepts joint_triplets/limbs/lr_pairs overrides for other skeletons.)

#: Joint triplets ``(a, vertex, c)`` whose angle at the clinical ``vertex`` joint
#: is measured (elbows, knees, shoulders, hips — the joints REHAB24-6 exercises
#: load: arm abduction/VW, push-ups, leg abduction/lunges, squats).
JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (7, 8, 9),      # left elbow   (LeftArm-LeftForeArm-LeftHand)
    (12, 13, 14),   # right elbow  (RightArm-RightForeArm-RightHand)
    (16, 17, 18),   # left knee    (LeftUpLeg-LeftLeg-LeftFoot)
    (21, 22, 23),   # right knee   (RightUpLeg-RightLeg-RightFoot)
    (2, 6, 7),      # left shoulder  (Spine1-LeftShoulder-LeftArm)
    (2, 11, 12),    # right shoulder (Spine1-RightShoulder-RightArm)
    (1, 16, 17),    # left hip     (Spine-LeftUpLeg-LeftLeg)
    (1, 21, 22),    # right hip    (Spine-RightUpLeg-RightLeg)
]

#: Limb segments ``(i, j)`` whose lengths drive angle-sensitivity — the rehab26
#: bone edges.
LIMBS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (2, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]

#: Left/right joint-triplet pairs whose angle difference flags compensation
#: (asymmetry) — elbows, knees, shoulders, hips.
LR_JOINT_PAIRS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((7, 8, 9), (12, 13, 14)),      # elbows
    ((16, 17, 18), (21, 22, 23)),   # knees
    ((2, 6, 7), (2, 11, 12)),       # shoulders
    ((1, 16, 17), (1, 21, 22)),     # hips
]

#: Default term weights (need not sum to 1; normalized at runtime).
DEFAULT_WEIGHTS: dict[str, float] = {
    "uncertainty": 0.30,
    "temporal": 0.15,
    "angle": 0.15,
    "rom": 0.20,
    "compensation": 0.10,
    "rarity": 0.10,
}

_EPS = 1e-8


def _angle_at(P: np.ndarray, a: int, b: int, c: int) -> np.ndarray:
    """Angle (radians) at vertex *b* of triplet ``a-b-c`` for every sample.

    Parameters
    ----------
    P:
        ``(N, K, 2)`` keypoint coordinates.
    a, b, c:
        Keypoint indices; ``b`` is the vertex.

    Returns
    -------
    numpy.ndarray
        ``(N,)`` angles in ``[0, pi]``.
    """
    v1 = P[:, a, :] - P[:, b, :]
    v2 = P[:, c, :] - P[:, b, :]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2 + _EPS
    cos = np.einsum("nd,nd->n", v1, v2) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


class RehabALStrategy(AcquisitionStrategy):
    """Composite acquisition strategy for RehabAL-Pose active learning.

    Parameters
    ----------
    weights:
        Optional mapping overriding :data:`DEFAULT_WEIGHTS`. Keys:
        ``uncertainty, temporal, angle, rom, compensation, rarity``. Values are
        normalized to sum to 1 (all-zero falls back to the defaults).
    rarity_keys:
        Metadata columns defining a "domain" cell for the rarity term
        (default ``("disease", "exercise", "camera_id")``).
    diversify:
        Diversity filter for :meth:`select` (defaults to ``"mmr"``).
    """

    name = "rehabal"

    def __init__(
        self,
        *,
        weights: Optional[dict[str, float]] = None,
        rarity_keys: Sequence[str] = ("disease", "exercise", "camera_id"),
        diversify: Optional[str] = "mmr",
        joint_triplets: Optional[Sequence[tuple[int, int, int]]] = None,
        limbs: Optional[Sequence[tuple[int, int]]] = None,
        lr_pairs: Optional[Sequence] = None,
        **kwargs,
    ) -> None:
        super().__init__(diversify=diversify, **kwargs)
        # Skeleton topology (defaults = rehab26). Override for other skeletons.
        self._triplets = list(joint_triplets) if joint_triplets is not None else JOINT_TRIPLETS
        self._limbs = list(limbs) if limbs is not None else LIMBS
        self._lr_pairs = list(lr_pairs) if lr_pairs is not None else LR_JOINT_PAIRS
        w = dict(DEFAULT_WEIGHTS)
        if weights:
            for key in weights:
                if key not in DEFAULT_WEIGHTS:
                    raise ValueError(
                        f"Unknown RehabAL weight {key!r}; valid keys: "
                        f"{sorted(DEFAULT_WEIGHTS)}"
                    )
            w.update({k: float(v) for k, v in weights.items()})
        total = sum(w.values())
        if total <= 0:
            w = dict(DEFAULT_WEIGHTS)
            total = sum(w.values())
        self.weights = {k: v / total for k, v in w.items()}
        self.rarity_keys = tuple(rarity_keys)

        self._uncertainty = UncertaintyStrategy(variance_weight=0.0)
        self._temporal = TemporalStrategy()

    # -- individual terms (each returns (N,) normalized to [0, 1]) ---------

    def term_uncertainty(self, pool: Pool) -> np.ndarray:
        """U: least-confidence uncertainty, normalized to ``[0, 1]``."""
        return minmax_normalize(self._uncertainty.score(pool))

    def term_temporal(self, pool: Pool) -> np.ndarray:
        """T: within-clip motion anomaly (already ``[0, 1]``)."""
        if pool.clip_ids is None:
            return np.zeros(pool.size)
        return self._temporal.score(pool)

    def term_angle_sensitivity(self, pool: Pool) -> np.ndarray:
        """A: angle-estimation fragility ``~ sum_j 1/limb_length``.

        Short or foreshortened limbs make joint-angle estimates highly
        sensitive to keypoint noise, so such samples are valuable to label.
        """
        P = pool.keypoints_xy()
        n, k = P.shape[0], P.shape[1]
        acc = np.zeros(n, dtype=np.float64)
        count = 0
        for i, j in self._limbs:
            if i >= k or j >= k:
                continue
            length = np.linalg.norm(P[:, i, :] - P[:, j, :], axis=1)
            acc += 1.0 / (length + _EPS)
            count += 1
        if count == 0:
            return np.zeros(n)
        return minmax_normalize(acc / count)

    def term_rom_criticality(self, pool: Pool) -> np.ndarray:
        """R: proximity of joint angles to ROM extremes via ``|cos(angle)|``.

        ``|cos(angle)|`` is 1 at full flexion (0 deg) / full extension
        (180 deg) and 0 at mid-range (90 deg); extremes carry the clinical
        scoring signal. Aggregated as the per-sample mean over valid joints and
        scaled by an optional disease/exercise criticality weight (1.0 here).
        """
        P = pool.keypoints_xy()
        n, k = P.shape[0], P.shape[1]
        vals = []
        for a, b, c in self._triplets:
            if max(a, b, c) >= k:
                continue
            ang = _angle_at(P, a, b, c)
            vals.append(np.abs(np.cos(ang)))
        if not vals:
            return np.zeros(n)
        crit = np.mean(np.stack(vals, axis=1), axis=1)
        return minmax_normalize(crit)

    def term_compensation_risk(self, pool: Pool) -> np.ndarray:
        """C: left/right joint-angle asymmetry (compensation proxy)."""
        P = pool.keypoints_xy()
        n, k = P.shape[0], P.shape[1]
        diffs = []
        for (ar, br, cr), (al, bl, cl) in self._lr_pairs:
            if max(ar, br, cr, al, bl, cl) >= k:
                continue
            ang_r = _angle_at(P, ar, br, cr)
            ang_l = _angle_at(P, al, bl, cl)
            diffs.append(np.abs(ang_r - ang_l) / np.pi)
        if not diffs:
            return np.zeros(n)
        asym = np.mean(np.stack(diffs, axis=1), axis=1)
        return minmax_normalize(asym)

    def term_domain_rarity(self, pool: Pool) -> np.ndarray:
        """D: inverse frequency of the (disease, exercise, camera) cell."""
        n = pool.size
        cols = [pool.meta(key) for key in self.rarity_keys]
        cols = [c for c in cols if c is not None]
        if not cols:
            return np.zeros(n)
        keys = [tuple(str(col[i]) for col in cols) for i in range(n)]
        counts: dict[tuple, int] = {}
        for key in keys:
            counts[key] = counts.get(key, 0) + 1
        rarity = np.array([1.0 / counts[key] for key in keys], dtype=np.float64)
        return minmax_normalize(rarity)

    # -- combined score ----------------------------------------------------

    def term_scores(self, pool: Pool) -> dict[str, np.ndarray]:
        """Return all six normalized term arrays keyed by name."""
        return {
            "uncertainty": self.term_uncertainty(pool),
            "temporal": self.term_temporal(pool),
            "angle": self.term_angle_sensitivity(pool),
            "rom": self.term_rom_criticality(pool),
            "compensation": self.term_compensation_risk(pool),
            "rarity": self.term_domain_rarity(pool),
        }

    def score(self, pool: Pool) -> np.ndarray:
        """Weighted composite score in ``[0, 1]`` (finite for every sample)."""
        terms = self.term_scores(pool)
        score = np.zeros(pool.size, dtype=np.float64)
        for name, w in self.weights.items():
            score += w * terms[name]
        return np.nan_to_num(np.clip(score, 0.0, 1.0), nan=0.0)
