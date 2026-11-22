"""Model layer: abstract pose-model interface, prediction container, and the
real Ultralytics YOLO-Pose backend.

The abstract :class:`PoseModel` / :class:`Prediction` contracts live in
``pose_base``; :class:`YoloPoseModel` is the concrete YOLO-Pose backend.

Importing this package stays heavy-import-free: ``yolo_pose`` only imports
stdlib + numpy at module load (``torch`` / ``ultralytics`` / ``cv2`` are
imported lazily inside its methods), so ``from .yolo_pose import YoloPoseModel``
does NOT pull in the heavy ML stack.
"""

from __future__ import annotations

from .pose_base import PoseModel, Prediction
from .yolo_pose import YoloPoseModel, is_available

__all__ = ["PoseModel", "Prediction", "YoloPoseModel", "is_available"]
