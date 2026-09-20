"""Build the detectors from config: the readers of `models.handdetect` and `models.gesturedetect`.

Each function consumes its block field by field, and these are the only readers of those two
blocks, so a key dropped here is a key nobody reads. The detector keys reach MediaPipe;
`palm_patch_radius_px` and `min_depth_samples` reach `HandFinder`, which samples depth around the
palm.

The 3-D builder does not source its transforms from config. A CAMERA->BASE transform is a
calibration artefact with an existing loader (`grasping.fusion` extrinsics, or an eye-in-hand
resolver composed from the live TCP), and a second spelling of it in `models.handdetect` would be a
second source of truth for the most safety-relevant number here. The caller passes the transforms it
already has.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from src.config.schema.models import GestureDetectConfig, HandDetectConfig
from src.calibration.stereo.manager import StereoCam3D
from src.camera.orchestration.frame_provider import FrameProvider
from src.models.handdetection.gestures import ThumbGestureRecognizer
from src.models.handdetection.hand_finder import HandFinder, HandObserver, OneCamera
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
) -> HandFinder:
    """`models.handdetect` + the caller's calibration -> a 3-D hand search.

    `observer` defaults to a landmark-only detector built from the same block. Pass a
    `ThumbGestureRecognizer` from `build_gesture_recognizer` when the located hand should also carry
    a gesture: that model returns landmarks too, so nothing is detected twice.
    """
    return HandFinder(
        observer if observer is not None else build_palm_detector(config),
        provider=provider,
        transforms=transforms,
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
) -> HandFinder:
    """A hand search over one `Camera` that is already open, reading that camera's own calibration.

    `build_hand_finder` takes a rig catalogue and a transform per rig, which is what a search over
    a whole cell needs. A caller that already holds one open camera has both facts on that object:
    `camera.calibration()` is the rig's declared calibration and `camera.get_intrinsics()` its
    lens. Opening a second `FrameProvider` to rediscover them would claim every other configured
    streamer, which is how asking where a hand is takes the cell's cameras away from it.

    No transform is composed here, and that is the point of the refusals below. A fixed camera's
    CAMERA to BASE is `RigCalibration.camera_to_base()`, one method with one answer. A wrist
    camera has none: its artifact is CAMERA to TOOL, and turning that into a base position needs
    the pose the arm stood at when the shutter opened, which is a composition `Locator` already
    owns. Writing it a second time here would make this the second source of truth for the most
    safety-relevant number in the package, so a wrist rig is refused by name instead.

    Raises
    ------
    RigNotCalibrated
        The rig declares no calibration; nothing can place what it sees.
    RigCalibrationError
        The rig is `eye_in_hand`, or its artifact does not load.
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
    return build_hand_finder(
        config,
        provider=frames,  # type: ignore[arg-type]  # RigFrames, the surface HandFinder reads
        transforms={rig_id: np.asarray(camera.calibration().camera_to_base().to_matrix(),
                                       dtype=np.float64)},
        observer=observer,
        camera_matrices={rig_id: np.asarray(matrix, dtype=np.float64)} if matrix is not None else None,
        rig_ids=[rig_id],
    )
