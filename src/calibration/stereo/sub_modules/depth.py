"""A calibrated stereo pair's depth, as an RGB-D camera hands it out, and the depths its matcher searches.

:class:`StereoDepth` turns a pair into what the pick path takes from an RGB-D camera (the owner, 2026-10-17): the
rectified left image is the colour image, the depth lies on its pixels in ``uint16`` millimetres with 0 where nothing
was measured, and the camera matrix is the rectified left one with no distortion. Perception, the hand-eye calibration,
the guard's perceived world and the planner then read a stereo rig as they read a RealSense.

The depth is ``focal * baseline / disparity``, both read off the calibration. Its matcher searches the disparities
:func:`working_depth_range_mm` gives, and nothing else is set per rig (``disparity.MEASURED_SETTINGS``):

* The far end is infinity, for every rig. A camera sees past the robot's workspace, the floor beside the bench and the
  walls of the room, and the matcher gives a surface beyond its search a wrong depth inside it rather than none:
  measured on the owner's wrist camera, a search ending at the workspace's diagonal read twice as many pixels too near
  as one that runs to infinity. A search to infinity starts at disparity 0, and costs a few disparities more.
* The near end is where the work comes nearest the camera. A camera fixed in the cell, whose calibration places it
  in BASE, searches from the nearest corner of the workspace box (``robot.motion.workspace``) as it sees it, and never
  from nearer than :data:`NEAR_LIMIT_MM`. A camera on the wrist is carried to the parts it looks at, and one not yet
  placed, or opened by a tool without the robot's section, has no box to read: these search from
  :data:`NEAR_LIMIT_MM`. A surface nearer than the near end, the arm between a fixed camera and its box or a part held
  in the jaws before a wrist camera, is not read right either: it is a hole, or a depth behind where it stands.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Final

import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.config import CalibrationResult

from .disparity import MEASURED_SETTINGS, DisparityComputer, DisparitySearch, MatcherSettings, disparity_search
from .rectifier import StereoRectifier

__all__ = ["NEAR_LIMIT_MM", "StereoDepth", "rectified_focal_and_baseline", "working_depth_range_mm"]

#: The nearest a stereo rig is asked to read, in millimetres, wherever no workspace box says the work stays further
#: away: on the wrist, before a rig is placed, and in a tool that holds no robot section. At the grasp the owner's
#: wrist camera stands 150 mm over the bench (the review's ray cast of the owner's pick, 2026-09-23, in
#: ``robot/safety/planning/live_world.py``), and a wrist camera that reads the bench there is the one the grasp's last
#: millimetres are checked against. Nearer than this, the parts between the jaws, is not read.
NEAR_LIMIT_MM: Final = 150.0

#: The largest depth a frame holds, millimetres: ``uint16``. A disparity under a pixel's worth of it, 49 m at the D415's
#: focal length and baseline, is read as no depth rather than clipped onto the farthest one.
_DEPTH_MAX_MM: Final = float(np.iinfo(np.uint16).max)


def working_depth_range_mm(*, workspace: Any = None, camera_to_base: Any = None) -> tuple[float, float]:
    """The depths a stereo rig's matcher searches, ``(near_mm, far_mm)`` in its camera frame (the module docstring).

    Args:
        workspace (Any): The robot's workspace box in BASE millimetres, anything with ``x_min`` to ``z_max`` such as
            ``robot.motion.workspace``; ``None`` where no robot section is at hand (default: None).
        camera_to_base (Any): A fixed camera's CAMERA to BASE, a 4 x 4 in millimetres; ``None`` for a camera on the
            wrist and for one not yet placed (default: None).

    Returns:
        tuple[float, float]: ``near_mm``, the nearest corner of the box the camera sees and at least
            :data:`NEAR_LIMIT_MM`, or :data:`NEAR_LIMIT_MM` without a box or a placement; and ``far_mm``, infinity.

    Raises:
        StereoCalibrationError: A box with an edge not above 0, or a placement that puts the whole box behind the
            camera, which is a CAMERA to BASE that is wrong.
    """
    if workspace is None or camera_to_base is None:
        return NEAR_LIMIT_MM, math.inf
    low = np.array([float(workspace.x_min), float(workspace.y_min), float(workspace.z_min)])
    high = np.array([float(workspace.x_max), float(workspace.y_max), float(workspace.z_max)])
    if not np.all(np.isfinite(low) & np.isfinite(high)) or np.any(high <= low):
        raise StereoCalibrationError(f"the workspace box must have finite edges above 0, got {low} to {high}")
    transform = np.asarray(camera_to_base, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise StereoCalibrationError("camera_to_base must be a finite 4 x 4")
    corners = np.array([[x, y, z] for x in (low[0], high[0]) for y in (low[1], high[1]) for z in (low[2], high[2])])
    base_to_camera = np.linalg.inv(transform)
    depths = corners @ base_to_camera[:3, :3].T[:, 2] + base_to_camera[2, 3]
    if float(depths.max()) <= 0.0:
        raise StereoCalibrationError(
            "the camera's CAMERA to BASE puts the whole workspace box behind it, so the camera sees none of the work: "
            "its calibration is wrong; calibrate it again")
    return max(NEAR_LIMIT_MM, float(depths.min())), math.inf


class StereoDepth:
    """A calibrated pair's depth in its rectified left image, as an RGB-D camera hands it out.

    Args:
        calibration (CalibrationResult): The pair's stereo calibration, rectified side by side, the right camera to the
            right of the left one, the two principal points one.
        depth_range_mm (Sequence[float]): ``(near_mm, far_mm)``, the depths to search (:func:`working_depth_range_mm`).
        settings (MatcherSettings): How the pair is matched; the measured ones unless a measurement compares others
            (default: MEASURED_SETTINGS).

    Raises:
        StereoCalibrationError: A calibration whose rectified cameras are not side by side, the right one to the right,
            with one principal point, or a range ``disparity_search`` refuses.
    """

    def __init__(self, calibration: CalibrationResult, depth_range_mm: Sequence[float], *,
                 settings: MatcherSettings = MEASURED_SETTINGS) -> None:
        near_mm, far_mm = (float(value) for value in depth_range_mm)
        self._calibration = calibration
        self._focal_px, self._baseline_mm = rectified_focal_and_baseline(calibration)
        self._range = (near_mm, far_mm)
        self._disparity = DisparityComputer(disparity_search(self._focal_px, self._baseline_mm, near_mm, far_mm),
                                            settings)
        self._rectifier = StereoRectifier(calibration)

    @property
    def calibration(self) -> CalibrationResult:
        """The pair's stereo calibration."""
        return self._calibration

    @property
    def focal_px(self) -> float:
        """The rectified pair's focal length, pixels."""
        return self._focal_px

    @property
    def baseline_mm(self) -> float:
        """The distance between the two cameras' centres, millimetres."""
        return self._baseline_mm

    @property
    def depth_range_mm(self) -> tuple[float, float]:
        """The depths searched, ``(near_mm, far_mm)``."""
        return self._range

    @property
    def search(self) -> DisparitySearch:
        """The disparities searched."""
        return self._disparity.search

    def intrinsics(self) -> np.ndarray:
        """The camera matrix of the rectified left image, the one the depth lies on: a copy, pixels."""
        return np.array(self._calibration.K_rect, dtype=np.float64, copy=True)

    @staticmethod
    def distortion() -> np.ndarray:
        """The rectified left image's distortion, none: five zeros."""
        return np.zeros(5, dtype=np.float64)

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Both images of a raw pair, rectified.

        Args:
            left (np.ndarray): The left camera's image, of the calibration's frame size.
            right (np.ndarray): The right camera's image, of the same size.

        Returns:
            tuple[np.ndarray, np.ndarray]: The rectified left and right images, as their own type.

        Raises:
            StereoCalibrationError: An image of another size than the calibration's.
        """
        return self._rectifier.rectify(left, right)

    def depth_mm(self, rect_left: np.ndarray, rect_right: np.ndarray) -> np.ndarray:
        """The depth of a rectified pair on the left image's pixels.

        Args:
            rect_left (np.ndarray): The rectified left image, grey or BGR.
            rect_right (np.ndarray): The rectified right image, of the same size.

        Returns:
            np.ndarray: ``uint16`` millimetres along the left camera's axis, 0 where nothing was measured.

        Raises:
            StereoCalibrationError: Images of different sizes.
        """
        disparity, valid = self._disparity.compute(rect_left, rect_right)
        depth = np.zeros(disparity.shape, dtype=np.float64)
        depth[valid] = (self._focal_px * self._baseline_mm) / disparity[valid].astype(np.float64)
        depth[depth > _DEPTH_MAX_MM] = 0.0
        return np.rint(depth).astype(np.uint16)


def rectified_focal_and_baseline(calibration: CalibrationResult) -> tuple[float, float]:
    """The rectified pair's focal length and baseline, the two numbers its depth and its search are read from.

    Args:
        calibration (CalibrationResult): The pair's stereo calibration.

    Returns:
        tuple[float, float]: The focal length, pixels, and the baseline, millimetres, both above 0.

    Raises:
        StereoCalibrationError: Rectified cameras one above the other, the right one to the left of the left one, or
            two principal points: a pair the matcher cannot read along its rows by ``f * B / d``.
    """
    left, right = np.asarray(calibration.projL), np.asarray(calibration.projR)
    focal = float(left[0, 0])
    if not (math.isfinite(focal) and focal > 0.0):
        raise StereoCalibrationError(f"the rectified focal length must be above 0, got {focal!r}")
    if abs(float(right[1, 3])) > 1e-6 * abs(float(right[0, 3])) + 1e-9:
        raise StereoCalibrationError(
            "the calibration rectifies the pair one above the other, and the matcher reads disparity along rows: "
            "mount the cameras side by side and calibrate again")
    baseline = -float(right[0, 3]) / float(right[0, 0])
    if not baseline > 0.0:
        raise StereoCalibrationError(
            "the calibration puts the right camera to the left of the left one: left and right are swapped. Swap the "
            "rig's two cameras (cam_left_id and cam_right_id, or the halves of a single device) and calibrate again")
    if abs(float(left[0, 2]) - float(right[0, 2])) > 1e-6:
        raise StereoCalibrationError(
            "the calibration's rectified cameras have two principal points, so a disparity is no depth by f * B / d: "
            "calibrate with this repository's calibrator, which rectifies them to one")
    return focal, baseline
