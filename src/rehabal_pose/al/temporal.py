from __future__ import annotations

import numpy as np

from .base import AcquisitionStrategy, Pool, minmax_normalize

__all__ = ["TemporalStrategy"]


def _frame_motion(positions: np.ndarray) -> np.ndarray:
    P = np.asarray(positions, dtype=np.float64)
    T = P.shape[0]
    vel = np.zeros_like(P)
    acc = np.zeros_like(P)
    jerk = np.zeros_like(P)

    if T >= 2:
        vel[1:] = P[1:] - P[:-1]
    if T >= 3:
        acc[1:-1] = P[2:] - 2.0 * P[1:-1] + P[:-2]
    if T >= 4:
        jerk[1 : T - 2] = P[3:T] - 3.0 * P[2 : T - 1] + 3.0 * P[1 : T - 2] - P[0 : T - 3]

    vmag = np.linalg.norm(vel, axis=2).mean(axis=1)
    amag = np.linalg.norm(acc, axis=2).mean(axis=1)
    jmag = np.linalg.norm(jerk, axis=2).mean(axis=1)
    return np.stack([vmag, amag, jmag], axis=1)


class TemporalStrategy(AcquisitionStrategy):
    name = "temporal"

    def __init__(
        self,
        *,
        w_velocity: float = 0.2,
        w_accel: float = 0.3,
        w_jerk: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.w_velocity = float(w_velocity)
        self.w_accel = float(w_accel)
        self.w_jerk = float(w_jerk)

    def score(self, pool: Pool) -> np.ndarray:
        if pool.predictions is None:
            raise ValueError("TemporalStrategy requires pool.predictions")

        xy = pool.keypoints_xy()
        raw = np.zeros(pool.size, dtype=np.float64)

        for _clip, idxs in pool.clip_groups().items():
            if len(idxs) == 1:
                continue
            motion = _frame_motion(xy[idxs])
            combined = (
                self.w_velocity * motion[:, 0]
                + self.w_accel * motion[:, 1]
                + self.w_jerk * motion[:, 2]
            )
            for local_t, global_i in enumerate(idxs):
                raw[global_i] = combined[local_t]

        return minmax_normalize(raw)
