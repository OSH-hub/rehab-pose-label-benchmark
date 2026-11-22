"""Skeleton overlay plotting (2D and 3D), matplotlib-guarded.

Importing this module never requires matplotlib. The plotting functions raise a
clear :class:`RuntimeError` if matplotlib is not installed.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

try:  # matplotlib is optional — guard at import time.
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
    """Plot a single 2D skeleton ``(K, 3)`` ``[x, y, visibility]``.

    Args:
        keypoints: ``(K, 3)`` array.
        edges: Index pairs to draw as bones.
        image: Optional HxW(x3) background to ``imshow``.
        ax: Existing matplotlib axes; created if ``None``.
        vis_threshold: Only draw joints with visibility above this.

    Returns:
        The matplotlib axes used.

    Raises:
        RuntimeError: If matplotlib is unavailable.
    """
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
    """Plot a single 3D skeleton ``(K, 3)`` ``[x, y, z]`` (or ``(K, 4)``).

    Args:
        keypoints: ``(K, 3)`` or ``(K, 4)`` array.
        edges: Index pairs to draw as bones.
        ax: Existing 3D axes; created if ``None``.

    Returns:
        The matplotlib 3D axes used.

    Raises:
        RuntimeError: If matplotlib is unavailable.
    """
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
