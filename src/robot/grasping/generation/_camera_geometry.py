"""The shared camera-geometry helper (intrinsics / pinhole back-projection / depth-patch sampling).

Owns the (optional) calibrated camera matrix and the per-instance warn-once synthetic-intrinsics latch.
Constructed once in :meth:`GraspCalculator.__init__` and shared with the candidate generator, so the
"no camera_matrix -> synthesizing intrinsics" warning fires once per calculator rather than once per
call; a second calculator warns again.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.grasping.geometry import CameraIntrinsics

if TYPE_CHECKING:
    from src.robot.grasping.generation.mask_analyzer import MaskAnalysis


class _SharedGeometryUtil:
    """Shared camera-geometry helper (intrinsics + back-projection + depth sampling)."""

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

    def center_point_mm(
        self,
        analysis: MaskAnalysis,
        depth_map: np.ndarray,
        median_depth_mm: float,
        point_3d_cam: np.ndarray | None,
        scale_to_mm: float,
    ) -> np.ndarray:
        if point_3d_cam is not None:
            point = np.asarray(point_3d_cam, dtype=np.float64) * scale_to_mm
            if point.shape != (3,):
                raise ValueError(f"point_3d_cam must be shape (3,), got {point.shape}")
            return point
        return self.pixel_depth_to_3d(
            analysis.centroid_xy[0],
            analysis.centroid_xy[1],
            median_depth_mm,
            depth_map,
        )

    @staticmethod
    def unit(vector: np.ndarray) -> np.ndarray:
        arr = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(arr))
        if norm < 1e-12:
            raise ValueError("cannot normalise zero vector")
        return arr / norm

    @staticmethod
    def principal_to_3d_axis(axis_2d: np.ndarray) -> np.ndarray:
        """Lift an image-space direction into the camera frame by zeroing its z.

        The zero puts the axis in the image plane, which is the plane a grasp closes in only when the
        camera looks straight down the support normal. A tilted camera, this rig included, does not:
        the result must be passed through :meth:`level_axis_to_support_plane`.
        """
        vector = np.array([axis_2d[0], axis_2d[1], 0.0], dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-12:
            return np.array([1.0, 0.0, 0.0], dtype=np.float64)
        return vector / norm

    @staticmethod
    def level_axis_to_support_plane(axis_cam: np.ndarray, up_cam: np.ndarray) -> np.ndarray:
        """Project a camera-frame closing axis into the support plane.

        A parallel jaw closes across an object that is standing on something, so almost every real
        grasp closes in the plane the object rests on: 98.9 % of the 28 332 labels in the datagen
        reference have an exactly horizontal closing axis. :meth:`principal_to_3d_axis` builds the axis
        by zeroing z in the camera frame, which lands it in the image plane instead. The two coincide
        only for a camera that looks straight down the support normal, and this project's rig is tilted
        36.87 deg (the obliques; the wrist camera 25.99 deg), so the generator's axes leave horizontal
        by a median 8.78 deg, up to 43.9 deg, on 94.4 % of the candidates that get executed.

        A tilted closing axis on a box lands on a face and an edge rather than on two opposing faces,
        which is why ``not_antipodal`` is 42 % of all wrong candidates even on objects that stand
        completely free with nothing occluding them. Removing the component along the support normal is
        the whole correction: top-1 22.0 -> 32.1 %, ten percentage points, over all 270 reference
        scenes.

        ``up_cam`` is the support-plane normal expressed in the camera frame, not necessarily BASE +Z.
        Passing the normal rather than assuming vertical lets a tilted tray or a bin floor be handled
        by the caller without touching this function.

        Degenerate input returns the axis untouched rather than guessing: an axis parallel to the
        support normal (a grasp closing straight down onto the table) has no projection, and inventing
        a direction there would be worse than leaving it.
        """
        axis = np.asarray(axis_cam, dtype=np.float64).reshape(3)
        up = np.asarray(up_cam, dtype=np.float64).reshape(3)
        up_norm = float(np.linalg.norm(up))
        if up_norm < 1e-12:
            return axis
        up = up / up_norm
        levelled = axis - float(axis @ up) * up
        norm = float(np.linalg.norm(levelled))
        if norm < 1e-6:
            return axis
        return levelled / norm

    def pixel_depth_to_3d(
        self,
        px: float,
        py: float,
        depth_mm: float,
        depth_map: np.ndarray,
    ) -> np.ndarray:
        intrinsics = self.intrinsics(depth_map.shape)
        x3d = (px - intrinsics.cx) * depth_mm / intrinsics.fx
        y3d = (py - intrinsics.cy) * depth_mm / intrinsics.fy
        return np.array([x3d, y3d, depth_mm], dtype=np.float64)

    @staticmethod
    def sample_depth_mm(
        depth_map: np.ndarray,
        px: float,
        py: float,
        scale_to_mm: float,
        patch_radius: int = 3,
    ) -> float | None:
        height, width = depth_map.shape[:2]
        ix, iy = int(round(px)), int(round(py))
        x0 = max(0, ix - patch_radius)
        x1 = min(width, ix + patch_radius + 1)
        y0 = max(0, iy - patch_radius)
        y1 = min(height, iy + patch_radius + 1)
        if x0 >= x1 or y0 >= y1:
            return None
        patch = depth_map[y0:y1, x0:x1].astype(np.float64)
        valid = patch[np.isfinite(patch) & (patch > 0.0)]
        if valid.size == 0:
            return None
        return float(np.median(valid)) * scale_to_mm

    def warn_synthetic_intrinsics(self) -> None:
        """Loud-once warning that intrinsics are synthesized (no calibrated K): metric 3D is unreliable."""
        if not self._warned_synthetic_K:
            self._warned_synthetic_K = True
            self._logger.warning(
                "No camera_matrix supplied -> synthesizing intrinsics (focal=width/2). Metric BASE-frame "
                "grasps are unreliable; pass a calibrated camera_matrix for real-hardware/metric picks."
            )

    def estimate_pixel_to_mm(self, median_depth_mm: float, depth_map: np.ndarray) -> float:
        if self._K is not None:
            focal_px = 0.5 * (float(self._K[0, 0]) + float(self._K[1, 1]))
        else:
            self.warn_synthetic_intrinsics()
            focal_px = max(float(depth_map.shape[1]) / 2.0, 1.0)
        return max(float(median_depth_mm) / focal_px, 0.01)
