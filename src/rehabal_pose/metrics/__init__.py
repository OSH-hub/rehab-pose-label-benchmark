"""Evaluation metrics for RehabAL-Pose.

Re-exports pose-accuracy, biomechanical, clinical-assessment, and
active-learning curve metrics. All implementations are pure NumPy with no
deep-learning dependencies, so this package imports cleanly in the test suite.
"""

from __future__ import annotations

from .assessment import (
    compensation_f1,
    proxy_deviation_score,
    quality_score_corr,
    quality_score_mae,
)
from .biomechanics import (
    JOINT_TRIPLETS,
    compute_rom,
    jerk,
    joint_angle,
    joint_angle_mae,
    peak_angle_error,
    phase_error,
    rom_error,
    temporal_smoothness,
)
from .learning_curve import (
    annotation_time_reduction,
    area_under_learning_curve,
    budget_to_target,
    correction_rate,
)
from .pose import mean_per_joint_error, oks, pck
from .significance import (
    bootstrap_ci,
    compare_curves,
    paired_bootstrap_ci,
    paired_difference_test,
    summarize_win_region,
)

__all__ = [
    # pose
    "pck",
    "oks",
    "mean_per_joint_error",
    # biomechanics
    "JOINT_TRIPLETS",
    "joint_angle",
    "compute_rom",
    "joint_angle_mae",
    "rom_error",
    "peak_angle_error",
    "phase_error",
    "jerk",
    "temporal_smoothness",
    # assessment
    "quality_score_mae",
    "quality_score_corr",
    "compensation_f1",
    "proxy_deviation_score",
    # learning curve
    "area_under_learning_curve",
    "budget_to_target",
    "annotation_time_reduction",
    "correction_rate",
    # significance
    "paired_difference_test",
    "bootstrap_ci",
    "paired_bootstrap_ci",
    "compare_curves",
    "summarize_win_region",
]
