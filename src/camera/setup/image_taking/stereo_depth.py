"""A stereo pair as an RGB-D camera: two eyes in, colour and depth out (the owner, 2026-10-17).

:class:`StereoDepthStreamer` stands where a RealSense stands. It wraps the rig's own pair streamer, a webcam pair
(``source: webcam_pair``) or one side by side device (``source: single_device``), loads the pair's stereo calibration
from ``calibration_paths.stereo_map_file`` when it opens, and hands out every grab as an :class:`RGBDFrame`: the colour
is the left image rectified, BGR ``uint8``, and the depth lies on its pixels in ``uint16`` millimetres, 0 where nothing
was measured (``calibration.stereo.StereoDepth``). Its camera matrix is the rectified left one and its distortion none,
so the pick perception, the hand-eye calibration, the guard's perceived world and the planner read it as they read a
RealSense, and a camera window reads the left image rectified through :meth:`StereoDepthStreamer.peek`.

The matcher keeps no history, so there is nothing for :meth:`StereoDepthStreamer.camera_moved` to drop, and every grab
is matched afresh: a frame of a carried camera holds depth seen from where it stands. What it searches is said once as
it opens: the depths, from a fixed camera's workspace box or from the near limit on the wrist
(``working_depth_range_mm``), and the disparities they give.

The stereo calibration tools keep the pair itself (``create_device_streamer``): they record and rectify raw eyes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import cv2 as cv
import numpy as np

from src.calibration.exceptions import StereoCalibrationError
from src.calibration.stereo.repository import StereoRigRepository
from src.calibration.stereo.sub_modules.depth import StereoDepth, working_depth_range_mm
from src.camera.setup.capture_backend import refuse_a_backend_this_platform_lacks
from src.camera.setup.image_taking.frames import RGBDFrame
from src.camera.setup.image_taking.rgbd import CAMERA_LOG_DIR
from src.config.schema.camera import SingleDeviceRigConfig, WebcamPairRigConfig
from src.utility.log_cfg import create_logger

__all__ = ["StereoDepthStreamer", "StereoNotCalibrated", "eye_size", "stereo_calibration_refusal"]

create_logger(__name__, "stereo_depth.log", log_dir=CAMERA_LOG_DIR)

StereoRig = WebcamPairRigConfig | SingleDeviceRigConfig

#: Where a person calibrates a pair, as every refusal of an uncalibrated one names it.
_HOW_TO_CALIBRATE = (
    "Calibrate the pair against the ChArUco board camera.cameras.stereo_calibration describes, which writes that "
    "file: examples/real_robot/19_calibrate_a_stereo_pair.py, StereoCalibration.from_tree(tree, rig_id=...) and its "
    "record() and calibrate()"
)


class StereoNotCalibrated(RuntimeError):
    """A stereo rig was opened as a depth camera, and the stereo calibration its depth is read through is not there."""


def stereo_calibration_refusal(rig: Any) -> str | None:
    """Why ``rig``, a stereo rig, gives no depth yet, or ``None`` once its stereo calibration is on disk.

    Asked where a camera is chosen (``select_rig``) and where it opens, so both say one sentence. An RGB-D rig is
    never refused here: its depth is its own.

    Args:
        rig (Any): One entry of ``camera.cameras.rigs``.

    Returns:
        str | None: The refusal, naming the file and how to write it; ``None`` where the rig gives depth.
    """
    source = getattr(rig, "source", None)
    if source not in ("webcam_pair", "single_device"):
        return None
    rig_id = getattr(rig, "rig_id", "?")
    written = getattr(getattr(rig, "calibration_paths", None), "stereo_map_file", None)
    if written and Path(str(written)).is_file():
        return None
    where = f", {Path(str(written))}" if written else ", which names no file"
    return (f"rig {rig_id!r} is a {source!r} rig, which gives depth through its stereo calibration, and there is "
            f"none at camera.cameras.rigs[{rig_id!r}].calibration_paths.stereo_map_file{where}. {_HOW_TO_CALIBRATE}.")


class StereoDepthStreamer:
    """A webcam pair or a side by side stereo device as an RGB-D camera (the module docstring).

    Building one touches no device: the pair is opened by :meth:`open`. The camera owner builds the pair's own streamer
    and hands it in (``camera.create_streamer``), as it is the one module that builds a device's streamer.

    Args:
        rig (StereoRig): The rig, one ``webcam_pair`` or ``single_device`` entry of ``camera.cameras.rigs``.
        pair (Any): The device behind it, answering ``open()``, ``grab()`` with a ``StereoFrame``, ``release()`` and
            ``is_opened()``: the rig's own pair streamer (``camera.create_device_streamer``).
        workspace (Any): ``robot.motion.workspace``, the box a fixed camera's search starts at; ``None`` where no robot
            section is at hand, which searches from the near limit (default: None).
    """

    def __init__(self, rig: StereoRig, *, pair: Any, workspace: Any = None) -> None:
        refuse_a_backend_this_platform_lacks(rig)
        self.cfg = rig
        self.logger = logging.getLogger(f"{__name__}.{rig.rig_id}")
        self._workspace = workspace
        self._pair = pair
        self._depth: StereoDepth | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self) -> None:
        """Load the pair's stereo calibration, derive its search, and open the pair.

        Raises:
            StereoNotCalibrated: The rig's stereo calibration is not on disk.
            StereoCalibrationError: It does not load, it was solved at another size than the rig's eyes, or it
                describes a pair the matcher cannot read along its rows.
            OSError: The pair does not open.
        """
        refusal = stereo_calibration_refusal(self.cfg)
        if refusal is not None:
            raise StereoNotCalibrated(refusal)
        path = Path(str(self.cfg.calibration_paths.stereo_map_file))
        calibration = StereoRigRepository().load(path)
        eye = eye_size(self.cfg)
        if tuple(calibration.frame_size) != eye:
            raise StereoCalibrationError(
                f"rig {self.cfg.rig_id!r}'s stereo calibration {path} was solved at {calibration.frame_size[0]} x "
                f"{calibration.frame_size[1]} and its eyes deliver {eye[0]} x {eye[1]}: calibrate the pair again at "
                "this size, or set the rig's size back")
        near_mm, far_mm = working_depth_range_mm(workspace=self._workspace, camera_to_base=self._camera_to_base())
        depth = StereoDepth(calibration, (near_mm, far_mm))
        self._pair.open()
        self._depth = depth
        search = depth.search
        self.logger.info(
            "stereo rig %s opened as a depth camera: focal %.1f px, baseline %.1f mm; depth searched from %.0f mm to %s, "
            "disparities %d to %d; calibration %s",
            self.cfg.rig_id, depth.focal_px, depth.baseline_mm, near_mm,
            "infinity" if np.isinf(far_mm) else f"{far_mm:.0f} mm", search.min_disparity, search.max_disparity - 1, path)

    def release(self) -> None:
        """Give the pair back. Idempotent."""
        self._pair.release()
        self._depth = None

    def is_opened(self) -> bool:
        """Whether the pair is open and its calibration loaded."""
        return self._depth is not None and bool(self._pair.is_opened())

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def grab(self) -> RGBDFrame:
        """One pair, rectified and matched: the left image as the colour and the depth on its pixels.

        Returns:
            RGBDFrame: ``color`` BGR ``uint8``, the left image rectified; ``depth`` ``uint16`` millimetres, 0 where
                nothing was measured.

        Raises:
            RuntimeError: The streamer is not open, or the pair's grab failed.
        """
        depth = self._require_open()
        frame = self._pair.grab()
        rect_left, rect_right = depth.rectify(frame.left, frame.right)
        return RGBDFrame(color=_bgr(rect_left), depth=depth.depth_mm(rect_left, rect_right))

    def peek(self) -> np.ndarray:
        """The left image rectified, for a person to look at: one pair grabbed and nothing matched.

        The matcher keeps no history, so a frame looked at changes nothing a later grab measures.

        Returns:
            np.ndarray: BGR ``uint8``, a new array.

        Raises:
            RuntimeError: The streamer is not open.
        """
        depth = self._require_open()
        frame = self._pair.grab()
        rect_left, _ = depth.rectify(frame.left, frame.right)
        return _bgr(rect_left)

    def camera_moved(self) -> None:
        """Nothing to drop: every grab is matched afresh, and no frame keeps a trace of the one before it."""

    # ------------------------------------------------------------------
    # Lens
    # ------------------------------------------------------------------

    def get_intrinsics(self) -> np.ndarray | None:
        """The rectified left camera's matrix, the one the depth lies on, once open; ``None`` before."""
        return None if self._depth is None else self._depth.intrinsics()

    def get_distortion(self) -> np.ndarray | None:
        """No distortion, five zeros: the images handed out are rectified. ``None`` before the streamer opens."""
        return None if self._depth is None else self._depth.distortion()

    @property
    def stereo_depth(self) -> StereoDepth | None:
        """The pair's depth as it opened, its search and its lens; ``None`` before."""
        return self._depth

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> StereoDepthStreamer:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> Literal[False]:
        self.release()
        return False

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_open(self) -> StereoDepth:
        if self._depth is None or not self._pair.is_opened():
            raise RuntimeError("Stereo rig is not open. Call open() first.")
        return self._depth

    def _camera_to_base(self) -> np.ndarray | None:
        """A fixed rig's CAMERA to BASE, from its calibration; ``None`` on the wrist, before it is placed, and where
        the calibration does not load, which searches from the near limit and says why."""
        extrinsics = getattr(self.cfg, "extrinsics", None)
        if extrinsics is None or getattr(extrinsics, "mounting_mode", None) != "eye_to_hand":
            return None
        from src.calibration.rig_calibration import RigCalibration, RigCalibrationError  # noqa: PLC0415

        try:
            return np.asarray(RigCalibration.from_config(self.cfg.rig_id, extrinsics).camera_to_base().to_matrix(),
                              dtype=np.float64)
        except RigCalibrationError as exc:
            self.logger.warning("stereo rig %s: its calibration does not load (%s), so its depth is searched from the "
                                "near limit, as a rig not yet placed is", self.cfg.rig_id, exc)
            return None


def eye_size(rig: StereoRig) -> tuple[int, int]:
    """The size of one eye a stereo rig delivers, ``(width, height)`` pixels: its stereo calibration is solved at it.

    Args:
        rig (StereoRig): A ``webcam_pair`` or ``single_device`` rig.

    Returns:
        tuple[int, int]: A webcam pair's ``frame_size``, a single device's ``per_eye_frame_size``.
    """
    if isinstance(rig, SingleDeviceRigConfig):
        return int(rig.per_eye_frame_size[0]), int(rig.per_eye_frame_size[1])
    return int(rig.frame_size[0]), int(rig.frame_size[1])


def _bgr(image: np.ndarray) -> np.ndarray:
    """A BGR ``uint8`` image: a grey eye made three channels, a colour one as it is."""
    array = np.asarray(image)
    if array.ndim == 2:
        return cv.cvtColor(array, cv.COLOR_GRAY2BGR)
    return np.ascontiguousarray(array)
