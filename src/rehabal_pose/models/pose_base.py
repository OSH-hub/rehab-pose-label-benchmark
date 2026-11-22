"""Abstract pose-model interface and prediction container.

Concrete backends such as Ultralytics YOLO-Pose implement :class:`PoseModel`.
The abstract methods define the common training, inference, evaluation, and
feature-extraction interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["PoseModel", "Prediction"]


@dataclass
class Prediction:
    """A single-sample pose prediction.

    Attributes
    ----------
    sample_id:
        Identifier matching a manifest ``sample_id``.
    keypoints:
        ``float`` ndarray of shape ``(K, 3)`` = ``[x, y, score]``.
    bbox:
        Optional ``[x, y, w, h]`` detection box, or ``None``.
    score:
        Optional overall detection/instance confidence in ``[0, 1]``.
    """

    sample_id: str
    keypoints: np.ndarray
    bbox: Optional[list[float]] = None
    score: Optional[float] = None

    def __post_init__(self) -> None:
        kp = np.asarray(self.keypoints, dtype=np.float64)
        if kp.ndim != 2 or kp.shape[1] != 3:
            raise ValueError(
                f"Prediction.keypoints must have shape (K, 3) [x, y, score]; "
                f"got shape {kp.shape}"
            )
        self.keypoints = kp


class PoseModel(ABC):
    """Abstract base class for 2D/3D pose estimation backends.

    Subclasses implement :meth:`train`, :meth:`predict`, :meth:`evaluate`, and
    :meth:`extract_embeddings`.
    """

    #: Human-readable backend name; subclasses should override.
    name: str = "abstract-pose-model"

    @abstractmethod
    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Train the model."""
        raise NotImplementedError(
            f"{type(self).__name__}.train must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Sequence["Prediction"]:
        """Run inference and return a sequence of :class:`Prediction`."""
        raise NotImplementedError(
            f"{type(self).__name__}.predict must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        """Evaluate the model and return a metrics dict."""
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate must be implemented by a concrete "
            "pose-model backend."
        )

    @abstractmethod
    def extract_embeddings(self, *args: Any, **kwargs: Any) -> np.ndarray:
        """Return per-sample feature embeddings for active-learning queries."""
        raise NotImplementedError(
            f"{type(self).__name__}.extract_embeddings must be implemented by a "
            "concrete pose-model backend."
        )
