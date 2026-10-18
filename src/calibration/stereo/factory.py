from __future__ import annotations

import time
from pathlib import Path

from .config import CalibrationResult, StereoRigConfig

from .repository import StereoRigRepository
from .runtime import StereoRigRunTime

from .sub_modules.calibrator import StereoCalibrator
from .sub_modules.depth import rectified_focal_and_baseline, working_depth_range_mm
from .sub_modules.disparity import DisparityComputer, disparity_search
from .sub_modules.pointcloud import PointCloudReconstructor
from .sub_modules.rectifier import StereoRectifier

from src.calibration.constants import CALIBRATION_LOG_DIR, STEREO_FACTORY_LOG_FILE
from src.calibration.exceptions import StereoCalibrationError
from src.utility.log_cfg import create_logger

__all__ = ["StereoRigFactory"]

# The injected `self.logger` belongs to the owning rig (StereoCam3D) and keeps its existing
# lines. This module logger is the calibration package's own trail: the calibration timings,
# artifact sizes and provenance a rig with the wrong geometry is diagnosed from.
logger = create_logger("StereoRigFactory", STEREO_FACTORY_LOG_FILE, log_dir=CALIBRATION_LOG_DIR)

class StereoRigFactory:
    """Builds one `StereoRigRunTime` per rig configuration.

    The ChArUco board geometry is fixed per factory and shared by every rig it builds; only the
    stereomap path and the image globs come from the individual `StereoRigConfig`. Each rig's
    matcher searches the depths a rig searches where no robot section is at hand
    (`working_depth_range_mm`), in disparities its own calibration gives (`disparity_search`).
    """

    def __init__(
        self,
        squares_x,
        squares_y,
        square_length_mm,
        marker_length_mm,
        aruco_dict_name,
        rectify_alpha,
        logger,
    ):
        self.logger = logger
        self.calibrator = StereoCalibrator(
            squares_x, squares_y, square_length_mm, marker_length_mm, aruco_dict_name, rectify_alpha
        )
        self.repository = StereoRigRepository()

    def create(self, config: StereoRigConfig, frame_size) -> StereoRigRunTime:
        """Assembles rectifier, disparity and reconstruction for one rig."""
        calib_result = self._load_or_calibrate(config, frame_size)
        focal_px, baseline_mm = rectified_focal_and_baseline(calib_result)
        search = disparity_search(focal_px, baseline_mm, *working_depth_range_mm())

        rectifier = StereoRectifier(calib_result)
        disparity = DisparityComputer(search)
        pointcloud = PointCloudReconstructor(calib_result.Q)

        logger.info(
            "Stereo rig runtime built: frame_size=%s, stereomap=%s, disparities %d to %d.",
            frame_size,
            config.stereomap_file,
            search.min_disparity,
            search.max_disparity - 1,
        )
        return StereoRigRunTime(
            config=config,
            calib_result=calib_result,
            rectifier=rectifier,
            disparity=disparity,
            pointcloud=pointcloud,
        )

    def _load_or_calibrate(self, config: StereoRigConfig, frame_size) -> CalibrationResult:
        """Loads the stereomap when it exists, calibrating from the image globs otherwise.

        A calibration computed here is written back to the stereomap path. A stereomap that
        exists but cannot be read is an error, never a recalibration.
        """
        stereomap_file = Path(config.stereomap_file)

        if stereomap_file.exists():
            try:
                calib_result = self.repository.load(stereomap_file)
            except (OSError, RuntimeError, ValueError, StereoCalibrationError) as exc:
                raise StereoCalibrationError(
                    f"failed to load stereo map {stereomap_file!s}"
                ) from exc
            self.logger.info("[Rig] Loaded stereomap: %s", stereomap_file)
            logger.info(
                "Stereomap loaded: %s (%d bytes, frame_size=%s).",
                stereomap_file,
                stereomap_file.stat().st_size,
                calib_result.frame_size,
            )
            return calib_result

        self.logger.info(
            "[Rig] Start stereo calibration using images: %s | %s",
            config.left_glob,
            config.right_glob,
        )

        started = time.perf_counter()
        calib_result = self.calibrator.calibrate(
            frame_size,
            config.left_glob,
            config.right_glob
        )
        logger.info(
            "Stereo calibration finished in %.1f s (frame_size=%s).",
            time.perf_counter() - started,
            calib_result.frame_size,
        )

        self.repository.save(stereomap_file, calib_result)
        self.logger.info("[Rig] Calibration saved -> %s", stereomap_file)
        logger.info(
            "Stereomap written: %s (%d bytes); its pose in the cell is the rig's own extrinsics, from a hand-eye "
            "calibration.",
            stereomap_file,
            stereomap_file.stat().st_size,
        )
        return calib_result
