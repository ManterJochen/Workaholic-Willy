"""Build the detectors from config: the readers of `models.handdetect` and `models.gesturedetect`.

Each function consumes its block field by field, and these are the only readers of those two
blocks, so a key dropped here is a key nobody reads. The detector keys reach MediaPipe;
`palm_patch_radius_px` and `min_depth_samples` reach `HandFinder`, which samples depth around the
palm.

The 3-D builder does not source its transforms from config. A CAMERA->BASE transform is a
calibration artefact with an existing loader (`grasping.fusion` extrinsics, or an eye-in-hand
resolver composed from the live TCP), and a second spelling of it in `models.handdetect` would be a
second source of truth for the most safety-relevant number here. The caller passes the transforms it
already has; over one open wrist camera, every frame is placed by the composition the `Locator`
places its frames by (`robot.core.shutter_motion`), read here and never written again.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.config.schema.models import GestureDetectConfig, HandDetectConfig
from src.calibration.rig_calibration import RigCalibrationError, flange_to_tcp_refusal
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration.frame_provider import FrameProvider
from src.geometry import Pose
from src.models.handdetection.gestures import ThumbGestureRecognizer
from src.models.handdetection.hand_finder import (
    FrameTransform,
    HandFinder,
    HandObserver,
    OneCamera,
    OneWristCamera,
)
from src.models.handdetection.palm_detector import PalmDetector

__all__ = [
    "build_gesture_recognizer",
    "build_hand_finder",
    "build_hand_finder_on_camera",
    "build_palm_detector",
]


def build_palm_detector(config: HandDetectConfig) -> PalmDetector:
    """Where hands are in a colour frame, in pixels, with no gesture.

    Args:
        config (HandDetectConfig): ``models.handdetect``: the ``.task`` bundle and the thresholds.

    Returns:
        PalmDetector: The detector; ``observe(frame_bgr)`` answers.

    Raises:
        ImportError: MediaPipe is not installed.
        FileNotFoundError: The bundle ``models.handdetect.model_path`` names is not there.
    """
    return PalmDetector(
        config.model_path,
        max_hands=config.max_hands,
        threshold=config.threshold,
        tracking_threshold=config.tracking_threshold,
        presence_threshold=config.presence_threshold,
        config_key="models.handdetect.model_path",
    )


def build_gesture_recognizer(config: GestureDetectConfig) -> ThumbGestureRecognizer:
    """A thumbs-up or thumbs-down recogniser that also reports the palm centre.

    Args:
        config (GestureDetectConfig): ``models.gesturedetect``: the ``.task`` bundle and the thresholds.

    Returns:
        ThumbGestureRecognizer: The recogniser; ``observe(frame_bgr)`` answers with a ``HandGesture``.

    Raises:
        ImportError: MediaPipe is not installed.
        FileNotFoundError: The bundle is not there.
    """
    return ThumbGestureRecognizer(
        config.model_path,
        max_hands=config.max_hands,
        threshold=config.threshold,
        tracking_threshold=config.tracking_threshold,
        presence_threshold=config.presence_threshold,
        min_gesture_confidence=config.min_gesture_confidence,
        config_key="models.gesturedetect.model_path",
    )


def build_hand_finder(
    config: HandDetectConfig,
    *,
    provider: FrameProvider,
    transforms: Mapping[str, np.ndarray],
    observer: Optional[HandObserver] = None,
    stereo: Optional[StereoCam3D] = None,
    camera_matrices: Optional[Mapping[str, np.ndarray]] = None,
    rig_ids: Optional[Sequence[str]] = None,
    frame_transforms: Optional[Mapping[str, FrameTransform]] = None,
) -> HandFinder:
    """`models.handdetect` + the caller's calibration -> a 3-D hand search.

    `observer` defaults to a landmark-only detector built from the same block. Pass a
    `ThumbGestureRecognizer` from `build_gesture_recognizer` when the located hand should also carry
    a gesture: that model returns landmarks too, so nothing is detected twice. `frame_transforms`
    places a rig that moves frame by frame (see `HandFinder`).
    """
    return HandFinder(
        observer if observer is not None else build_palm_detector(config),
        provider=provider,
        transforms=transforms,
        frame_transforms=frame_transforms,
        stereo=stereo,
        camera_matrices=camera_matrices,
        rig_ids=rig_ids,
        palm_patch_radius_px=config.palm_patch_radius_px,
        min_depth_samples=config.min_depth_samples,
    )


def build_hand_finder_on_camera(
    config: HandDetectConfig,
    camera: Any,
    *,
    observer: Optional[HandObserver] = None,
    tool_pose: Optional[Callable[[], Pose]] = None,
    tool_frame: Any = None,
    attempts: Optional[int] = None,
) -> HandFinder:
    """Where a hand is, in millimetres in the robot's base frame, over one camera you already hold open, read through
    that camera's own calibration.

    Args:
        config (HandDetectConfig): ``models.handdetect``.
        camera (Any): An open camera owner (:class:`Camera`).
        observer (HandObserver | None): The detector to use; ``None`` builds one from ``config`` (default: None).
        tool_pose (Callable[[], Pose] | None): For a camera on the wrist: the arm's ``get_tcp_pose``, read at each
            shutter; required there (default: None).
        tool_frame (Any): For a camera on the wrist: the cell's ``robot.gripper.tool_frame`` (default: None).
        attempts (int | None): How many more frames a wrist camera takes while the tool moved (default: None).

    Returns:
        HandFinder: The finder; ``find_hand()`` answers in BASE millimetres.

    Raises:
        RigCalibrationError: A camera on the wrist without ``tool_pose`` or ``tool_frame``, or whose calibration was
            solved against another tool frame; ``RigNotCalibrated``, one of these, where it declares no calibration.
    """
    matrix = camera.get_intrinsics()
    rig_id = str(camera.rig_id)
    frames = OneCamera(camera)
    if frames.is_rgbd(rig_id) and matrix is None:
        raise ValueError(
            f"rig {rig_id!r} is RGB-D and its device reports no intrinsics, so a palm cannot be "
            "back-projected into millimetres. Nothing is invented here: give the rig a camera "
            "matrix, or use build_hand_finder with camera_matrices of your own.")
    calibration = camera.calibration()
    transforms: dict[str, np.ndarray] = {}
    frame_transforms: Optional[dict[str, FrameTransform]] = None
    if str(getattr(calibration, "mounting_mode", "")) == "eye_in_hand":
        wrist = _wrist_frames(camera, calibration, tool_pose=tool_pose, tool_frame=tool_frame,
                              attempts=attempts)
        frames, frame_transforms = wrist, {rig_id: wrist.camera_to_base_of}
    else:
        transforms[rig_id] = np.asarray(calibration.camera_to_base().to_matrix(), dtype=np.float64)
    return build_hand_finder(
        config,
        provider=frames,  # type: ignore[arg-type]  # RigFrames, the surface HandFinder reads
        transforms=transforms,
        frame_transforms=frame_transforms,
        observer=observer,
        camera_matrices={rig_id: np.asarray(matrix, dtype=np.float64)} if matrix is not None else None,
        rig_ids=[rig_id],
    )


def _wrist_frames(camera: Any, calibration: Any, *, tool_pose: Optional[Callable[[], Pose]],
                  tool_frame: Any, attempts: Optional[int]) -> OneWristCamera:
    """The frames of a camera on the wrist, each stamped at its shutter and placed by it.

    Refused as `Locator.from_parts` refuses a wrist rig, in its order and before anything is
    grabbed or read: no reader of the arm's TCP, no tool frame, a calibration not solved against it.
    """
    rig_id = str(camera.rig_id)
    if tool_pose is None or not callable(tool_pose):
        raise RigCalibrationError(
            f"rig {rig_id!r} is on the wrist (eye_in_hand): its calibration is CAMERA to TOOL, so "
            "where it stood at each shutter is where the tool stood, and no reader of the arm's TCP "
            "was handed in. Pass tool_pose=robot.arm.get_tcp_pose, and "
            "tool_frame=tree.robot.gripper.tool_frame")
    if tool_frame is None:
        raise RigCalibrationError(
            f"rig {rig_id!r} is on the wrist and no tool frame was handed in, so its calibration "
            "cannot be held to the flange to TCP this cell applies: pass "
            "tool_frame=tree.robot.gripper.tool_frame")
    stale = flange_to_tcp_refusal(calibration, tool_frame)
    if stale is not None:
        raise RigCalibrationError(stale)
    # The pick frame's stamp and the Locator's composition, from where they are written. Loaded
    # here, for a wrist rig, so a fixed camera's search loads nothing of the robot.
    from src.robot.core.shutter_motion import (  # noqa: PLC0415
        PICK_FRAME_WARMUP_GRABS,
        ShutterStamp,
        camera_to_base_at_shutter,
    )

    if attempts is None:
        from src.config.schema.robot.safety_schema import PerceivedWorldConfig  # noqa: PLC0415

        attempts = int(PerceivedWorldConfig.model_fields["fresh_frame_attempts"].default)
    stamp = ShutterStamp.of(
        tool_pose,
        motion_tolerance=(calibration.shutter_motion_tolerance_mm,
                          calibration.shutter_motion_tolerance_deg),
        attempts=attempts,
    )
    return OneWristCamera(camera, stamp=stamp, place=functools.partial(camera_to_base_at_shutter, calibration),
                          warmups=PICK_FRAME_WARMUP_GRABS)
