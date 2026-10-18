"""The calculator's camera lens: the calibrated intrinsics, or ones synthesized from the image size.

Owns the (optional) calibrated camera matrix and the per-instance warn-once synthetic-intrinsics latch.
Constructed once in :meth:`GraspCalculator.__init__`, so the "no camera_matrix -> synthesizing intrinsics"
warning fires once per calculator rather than once per call; a second calculator warns again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.robot.grasping.geometry import CameraIntrinsics


class _SharedGeometryUtil:
    """The calculator's lens: the calibrated intrinsics, else synthesized ones, said once."""

    def __init__(self, camera_matrix: np.ndarray | None, logger: Any) -> None:
        # ``camera_matrix`` is the already-validated (3,3) array (or None) from GraspCalculator.__init__.
        self._K = camera_matrix
        # Loud-once latch: warn at most once per instance that intrinsics are synthesized.
        self._warned_synthetic_K = False
        self._logger = logger

    def intrinsics(self, depth_shape: Sequence[int]) -> CameraIntrinsics:
        if self._K is not None:
            return CameraIntrinsics.from_matrix(self._K)
        self.warn_synthetic_intrinsics()
        height, width = depth_shape[:2]
        focal = max(float(width) / 2.0, 1.0)
        return CameraIntrinsics(
            fx=focal,
            fy=focal,
            cx=float(width) / 2.0,
            cy=float(height) / 2.0,
        )

    def warn_synthetic_intrinsics(self) -> None:
        """Loud-once warning that intrinsics are synthesized (no calibrated K): metric 3D is unreliable."""
        if not self._warned_synthetic_K:
            self._warned_synthetic_K = True
            self._logger.warning(
                "No camera_matrix supplied -> synthesizing intrinsics (focal=width/2). Metric BASE-frame "
                "grasps are unreliable; pass a calibrated camera_matrix for real-hardware/metric picks."
            )
