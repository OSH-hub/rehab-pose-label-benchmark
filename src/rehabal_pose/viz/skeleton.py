from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

try:
    import matplotlib.pyplot as _plt  # type: ignore

    HAS_MPL = True
except Exception:  # pragma: no cover - exercised only without matplotlib
    _plt = None  # type: ignore
    HAS_MPL = False


def _require_mpl() -> None:
    if not HAS_MPL:
        raise RuntimeError(
            "matplotlib is required for skeleton plotting; install it with "
            "`pip install matplotlib`."
        )


def plot_skeleton_2d(
    keypoints: np.ndarray,
    *,
    edges: Optional[Sequence[Sequence[int]]] = None,
    image: Optional[np.ndarray] = None,
    ax: Optional[Any] = None,
    vis_threshold: float = 0.0,
) -> Any:
    _require_mpl()
    kp = np.asarray(keypoints, dtype=float)
    if ax is None:
        _, ax = _plt.subplots()
    if image is not None:
        ax.imshow(image)
    vis = kp[:, 2] > vis_threshold if kp.shape[1] >= 3 else np.ones(len(kp), bool)
    ax.scatter(kp[vis, 0], kp[vis, 1], s=20)
    for a, b in edges or []:
        if a < len(kp) and b < len(kp) and vis[a] and vis[b]:
            ax.plot([kp[a, 0], kp[b, 0]], [kp[a, 1], kp[b, 1]])
    ax.set_aspect("equal")
    return ax


def plot_skeleton_3d(
    keypoints: np.ndarray,
    *,
    edges: Optional[Sequence[Sequence[int]]] = None,
    ax: Optional[Any] = None,
) -> Any:
    _require_mpl()
    kp = np.asarray(keypoints, dtype=float)
    if ax is None:
        fig = _plt.figure()
        ax = fig.add_subplot(111, projection="3d")
    ax.scatter(kp[:, 0], kp[:, 1], kp[:, 2], s=20)
    for a, b in edges or []:
        if a < len(kp) and b < len(kp):
            ax.plot(
                [kp[a, 0], kp[b, 0]],
                [kp[a, 1], kp[b, 1]],
                [kp[a, 2], kp[b, 2]],
            )
    return ax
