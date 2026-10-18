from __future__ import annotations

import logging
from typing import Optional, Sequence, TYPE_CHECKING

import numpy as np

from .config import StereoRigConfig
from .factory import StereoRigFactory

from .sub_modules.aruco_esti import ArucoPoseEstimator

from src.calibration.exceptions import StereoCalibrationError

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.camera import CalibrationConfig

__all__ = ["StereoCam3D"]

class StereoCam3D:
    """Facade over one or more calibrated stereo rigs, addressed by index.

    Builds a `StereoRigRunTime` per configuration through `StereoRigFactory` and
    forwards each call to the rig named by `rig`, which defaults to the first one.
    All rigs share the frame size and board geometry of the one `CalibrationConfig`;
    each matches at the measured settings over the search its own calibration gives
    (`sub_modules.disparity`). `marker_length_mm` overrides it for marker pose
    estimation only; `aruco_dict_name` overrides the dictionary for pose estimation
    and for the ChArUco calibration alike.
    """

    def __init__(
        self,
        rigs: Sequence[StereoRigConfig],
        calibration: CalibrationConfig,
        marker_length_mm: float | None = None,
        aruco_dict_name: str | None = None,
    ):
        self.logger = logging.getLogger("src.calibration.stereo.StereoCam3D")
        self.frame_size = tuple(calibration.frame_size)
        self.marker_length_mm = (
            float(marker_length_mm)
            if marker_length_mm is not None
            else float(calibration.marker_length_mm)
        )
        dict_name = aruco_dict_name or calibration.aruco_dict_name

        factory = StereoRigFactory(
            squares_x=calibration.charuco_squares_x,
            squares_y=calibration.charuco_squares_y,
            square_length_mm=calibration.charuco_square_length_mm,
            marker_length_mm=calibration.charuco_marker_length_mm,
            aruco_dict_name=dict_name,
            rectify_alpha=calibration.rectify_alpha,
            logger=self.logger,
        )

        self.rigs = [factory.create(cfg, self.frame_size) for cfg in rigs]
        if not self.rigs:
            raise StereoCalibrationError("at least one stereo rig configuration is required")

        self.aruco_esti = ArucoPoseEstimator(
            marker_length_mm=self.marker_length_mm,
            dict_name=dict_name,
        )

    def _rig(self, rig: int):
        """Returns the runtime at index `rig`. A bool is not an index here."""
        if not isinstance(rig, int) or isinstance(rig, bool):
            raise IndexError(f"rig index must be an integer, got {type(rig).__name__}")
        if rig < 0 or rig >= len(self.rigs):
            raise IndexError(f"Invalid rig index {rig}. Available: 0..{len(self.rigs)-1}")
        return self.rigs[rig]

    def rectify(self, left_bgr, right_bgr, rig=0):
        return self._rig(rig).rectify(left_bgr, right_bgr)

    def compute_disparity(self, rect_left_bgr, rect_right_bgr, rig=0):
        return self._rig(rig).compute_disparity(rect_left_bgr, rect_right_bgr)

    def reproject_to_3d(self, disparity_px: np.ndarray, rig=0):
        return self._rig(rig).reproject_to_3d(disparity_px)

    def compute_depth_map(self, rect_left_bgr, rect_right_bgr, unit: str = "mm", rig=0):
        return self._rig(rig).compute_depth_map(rect_left_bgr, rect_right_bgr, unit)

    def compute_3D_point(self, rect_left_bgr, rect_right_bgr, mask=None, reducer: str = "median", unit: str = "mm", rig=0):
        return self._rig(rig).compute_3d_point(rect_left_bgr, rect_right_bgr, mask, reducer, unit)

    def compute_3D_point_xy(self, rect_left_bgr, rect_right_bgr, x, y, unit="mm", rig=0):
        return self._rig(rig).compute_3d_point_xy(rect_left_bgr, rect_right_bgr, x, y, unit)

    def estimate_marker_pose_left(self, rect_left_bgr, target_id: Optional[int] = None, rig=0):
        """Estimates T_cam_to_marker in the rig's rectified left frame.

        The frame must already be rectified: the rig's rectified intrinsics are used
        with zero distortion. With `target_id`, one 4x4 matrix or None; without it, a
        dict of every detected marker id.
        """
        runtime = self._rig(rig)
        K_rect = runtime.calib_result.K_rect
        dist_rect = np.zeros(5)  # rectified images have zero distortion
        return self.aruco_esti.estimate(rect_left_bgr, K_rect, dist_rect, target_id)
