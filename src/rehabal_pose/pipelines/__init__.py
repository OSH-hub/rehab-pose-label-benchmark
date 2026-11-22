"""Command-line pipelines for the RehabAL-Pose workflow.

Each module exposes an argparse ``main()`` and is runnable via
``python -m rehabal_pose.pipelines.<name>``. Dataset preparation and acquisition
commands support the included synthetic fixtures. Training and inference
commands load optional pose-estimation dependencies only when invoked.
"""

from __future__ import annotations

__all__ = [
    "build_manifest",
    "convert_dataset",
    "train_pose",
    "infer_pool",
    "acquire_batch",
    "run_al_loop",
    "evaluate_pose",
    "evaluate_rehab",
    "build_report",
]
