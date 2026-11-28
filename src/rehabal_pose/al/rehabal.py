from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from .base import AcquisitionStrategy, Pool, minmax_normalize
from .temporal import TemporalStrategy
from .uncertainty import UncertaintyStrategy

__all__ = ["RehabALStrategy", "DEFAULT_WEIGHTS"]


JOINT_TRIPLETS: list[tuple[int, int, int]] = [
    (7, 8, 9),
    (12, 13, 14),
    (16, 17, 18),
    (21, 22, 23),
    (2, 6, 7),
    (2, 11, 12),
    (1, 16, 17),
    (1, 21, 22),
]

LIMBS: list[tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    (2, 6), (6, 7), (7, 8), (8, 9), (9, 10),
    (2, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 16), (16, 17), (17, 18), (18, 19), (19, 20),
    (0, 21), (21, 22), (22, 23), (23, 24), (24, 25),
]

LR_JOINT_PAIRS: list[tuple[tuple[int, int, int], tuple[int, int, int]]] = [
    ((7, 8, 9), (12, 13, 14)),
    ((16, 17, 18), (21, 22, 23)),
    ((2, 6, 7), (2, 11, 12)),
    ((1, 16, 17), (1, 21, 22)),
]

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
    v1 = P[:, a, :] - P[:, b, :]
    v2 = P[:, c, :] - P[:, b, :]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2 + _EPS
    cos = np.einsum("nd,nd->n", v1, v2) / denom
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


class RehabALStrategy(AcquisitionStrategy):
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


    def term_uncertainty(self, pool: Pool) -> np.ndarray:
        return minmax_normalize(self._uncertainty.score(pool))

    def term_temporal(self, pool: Pool) -> np.ndarray:
        if pool.clip_ids is None:
            return np.zeros(pool.size)
        return self._temporal.score(pool)

    def term_angle_sensitivity(self, pool: Pool) -> np.ndarray:
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


    def term_scores(self, pool: Pool) -> dict[str, np.ndarray]:
        return {
            "uncertainty": self.term_uncertainty(pool),
            "temporal": self.term_temporal(pool),
            "angle": self.term_angle_sensitivity(pool),
            "rom": self.term_rom_criticality(pool),
            "compensation": self.term_compensation_risk(pool),
            "rarity": self.term_domain_rarity(pool),
        }

    def score(self, pool: Pool) -> np.ndarray:
        terms = self.term_scores(pool)
        score = np.zeros(pool.size, dtype=np.float64)
        for name, w in self.weights.items():
            score += w * terms[name]
        return np.nan_to_num(np.clip(score, 0.0, 1.0), nan=0.0)
