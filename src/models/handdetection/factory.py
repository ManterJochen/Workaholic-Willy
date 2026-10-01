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
    """`models.handdetect` -> a landmark detector. Raises if mediapipe or the bundle is missing."""
    return PalmDetector(
        config.model_path,
        max_hands=config.max_hands,
        threshold=config.threshold,
        tracking_threshold=config.tracking_threshold,
        presence_threshold=config.presence_threshold,
        config_key="models.handdetect.model_path",
    )


def build_gesture_recognizer(config: GestureDetectConfig) -> ThumbGestureRecognizer:
    """`models.gesturedetect` -> a thumbs-up/down recogniser that also reports palm centres."""
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
    """A hand search over one `Camera` that is already open, reading that camera's own calibration.

    `build_hand_finder` takes a rig catalogue and a transform per rig, which is what a search over
    a whole cell needs. A caller that already holds one open camera has both facts on that object:
    `camera.calibration()` is the rig's declared calibration and `camera.get_intrinsics()` its
    lens. Opening a second `FrameProvider` to rediscover them would claim every other configured
    streamer, which is how asking where a hand is takes the cell's cameras away from it.

    A fixed camera's CAMERA to BASE is `RigCalibration.camera_to_base()`, one method with one
    answer, and `tool_pose`, `tool_frame` and `attempts` are ignored for it. A camera on the wrist
    has none: its artifact is CAMERA to TOOL, and where it stood is where the tool stood when the
    shutter opened. It takes what `Locator.from_parts` takes for one: `tool_pose`, the arm's
    `get_tcp_pose`, and `tool_frame`, the cell's `robot.gripper.tool_frame`, which its calibration
    must have been solved against. Each frame is then taken as the pick frame is (its warm-ups,
    then the tool pose read before and after the grab, the grab taken again while the tool moved
    beyond the rig's shutter tolerance, up to `attempts` more times; unset, the schema's
    `perceived.fresh_frame_attempts` default, as `Locator.from_parts` takes it, while
    `Locator.from_tree` reads the profile's, so pass that to match it), and placed by the
    composition the `Locator` places its frames by, `shutter_motion.camera_to_base_at_shutter`:
    called here, never written a second time. A frame the tool moved across on every attempt gives
    no hand, so the arm holds still while it looks.

    Raises
    ------
    RigNotCalibrated
        The rig declares no calibration; nothing can place what it sees.
    RigCalibrationError
        The rig's artifact does not load; or the rig is on the wrist and was handed no `tool_pose`
        or no `tool_frame`, or its calibration was solved against another flange to TCP. Each says
        its fix. Refused before anything is grabbed or read.
    ValueError
        The rig is RGB-D and its device reports no intrinsics, so a palm cannot be back-projected.
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
