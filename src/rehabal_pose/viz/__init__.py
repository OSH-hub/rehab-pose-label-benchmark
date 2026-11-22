"""Visualization helpers (skeletons, budget curves, dashboards).

All matplotlib usage is guarded behind import-time ``try/except`` inside the
submodules, so importing :mod:`rehabal_pose.viz` never fails when matplotlib is
absent. Functions that actually plot raise a clear ``RuntimeError`` if the
backend is unavailable.
"""

from __future__ import annotations

from .skeleton import HAS_MPL, plot_skeleton_2d, plot_skeleton_3d
from .curves import plot_budget_curve
from .dashboard import build_dashboard

__all__ = [
    "HAS_MPL",
    "plot_skeleton_2d",
    "plot_skeleton_3d",
    "plot_budget_curve",
    "build_dashboard",
]
