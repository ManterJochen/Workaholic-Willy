"""Where the hand is in the robot's frame, over a stereo rig or an RGB-D (RealSense) rig.

The 2-D detectors give a palm centre in pixels. Turning that into millimetres in the base frame
takes depth and two transforms, and the two supported camera kinds get there differently:

* RGB-D (`RGBDFrame`): read the depth map at the palm and back-project through the colour
  intrinsics. Needs a `K` per rig.
* Stereo (`StereoFrame`): rectify the pair and ask `StereoCam3D` for a robust 3-D point inside a
  small mask around the palm. Needs the rig's stereo calibration, which `StereoCam3D` holds.

Intrinsics and transforms are per-rig facts and `find_hand` iterates over several rigs, so neither
is held as a single value. `transforms` is keyed by rig id, and a rig missing from it is skipped
with a warning: one rig's `T_cam_to_base` applied to another rig's detection yields confident,
wrong base coordinates. An RGB-D rig without intrinsics refuses rather than substituting
`fx = fy = width / 2`, which completes the arithmetic and returns a position no measurement
supports.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Protocol, Sequence

import numpy as np

from src.calibration.stereo.manager import StereoCam3D
from src.camera.setup.image_taking.frames import AnyFrame, RGBDFrame, StereoFrame
from src.config.schema.camera import RGBDDeviceRigConfig
from src.models.handdetection.constants import HAND_FINDER_LOG_FILE, MODELS_LOG_DIR
from src.models.handdetection.landmarks import draw_hand_landmarks
from src.models.handdetection.types import (
    GestureReading,
    HandGesture,
    HandObservation,
    HandPosition3D,
    LocatedHand,
)
from src.utility.log_cfg import create_logger

__all__ = ["HandFinder", "HandObserver", "OneCamera", "RigFrames"]


class HandObserver(Protocol):
    """What `HandFinder` needs from a 2-D model: hands in a frame, each with a gesture.

    Structural, so the 3-D search never branches on which model the operator configured.
    `PalmDetector` satisfies it with `gesture=NONE` on every observation, `ThumbGestureRecognizer`
    with a real reading.
    """

    def observe(self, frame_bgr: np.ndarray) -> list[HandObservation]: ...


class RigFrames(Protocol):
    """Where `HandFinder` reads frames from: three methods, and `FrameProvider` is one of them.

    Structural rather than the concrete catalogue, because of who owns the devices. A cell-wide
    search over several rigs is what `FrameProvider` is for. A program that has already opened one
    `Camera` -- a pick loop, or anything holding the camera that builds the arm's live world --
    cannot build a second catalogue to ask where a hand is: `FrameProvider.open()` claims every
    configured streamer, so the question would take the cell's other cameras away from it, and one
    device opened twice is what the camera owner exists to prevent. `OneCamera` below serves this
    surface over a camera that is already open, and claims nothing.
    """

    @property
    def rig_ids(self) -> list[str]: ...

    def is_rgbd(self, rig_id: str) -> bool: ...

    def grab(self, rig_id: str) -> AnyFrame: ...

    def get_stereo_rig_index(self, rig_id: str) -> int: ...


class OneCamera:
    """One already-open `Camera`, shaped as the `RigFrames` surface above.

    It owns nothing: whoever opened the camera closes it, and a grab here goes through that owner's
    lock like every other grab. `rig_ids` is that one rig, so a search over it can reach no other.
    """

    __slots__ = ("camera",)

    def __init__(self, camera: Any) -> None:
        self.camera = camera

    @property
    def rig_ids(self) -> list[str]:
        return [str(self.camera.rig_id)]

    def is_rgbd(self, rig_id: str) -> bool:
        """The rig's own declaration, the same test `FrameProvider.is_rgbd` makes."""
        self._require(rig_id)
        return isinstance(self.camera.rig, RGBDDeviceRigConfig)

    def grab(self, rig_id: str) -> AnyFrame:
        self._require(rig_id)
        frame: AnyFrame = self.camera.grab()
        return frame

    def get_stereo_rig_index(self, rig_id: str) -> int:
        """Refused: the index is a position in a stereo calibration a single camera does not hold.

        Unreachable on the stereo path in practice, because that path refuses first when no
        `StereoCam3D` was supplied and this door supplies none. It refuses by name rather than
        returning 0, which would triangulate against whatever calibration sat first in the file.
        """
        self._require(rig_id)
        raise ValueError(
            f"rig {rig_id!r} is stereo, and a search over one open camera carries no stereo "
            "calibration index. Build the search with a FrameProvider and a StereoCam3D.")

    def _require(self, rig_id: str) -> None:
        if rig_id != str(self.camera.rig_id):
            raise KeyError(
                f"this hand search holds camera {self.camera.rig_id!r} and was asked for "
                f"{rig_id!r}. One camera reaches one rig; search the cell with a FrameProvider.")

    def __repr__(self) -> str:  # pragma: no cover (debugging aid)
        return f"OneCamera({self.camera.rig_id!r})"


class HandFinder:
    """Search one or more camera rigs for exactly one hand and locate it in the base frame.

    Parameters
    ----------
    observer:
        The 2-D model. See `HandObserver`.
    provider:
        Where the frames come from: an already-opened `FrameProvider`, or `OneCamera` over a single
        open `Camera`. Neither opened nor released here: whoever opened the cameras closes them.
    transforms:
        `{rig_id: 4x4 CAMERA->BASE}`. A rig missing from this map cannot be expressed in base
        coordinates and is skipped.
    stereo:
        The stereo engine, or `None` on an all-RGB-D cell.
    camera_matrices:
        `{rig_id: 3x3 K}` for the RGB-D rigs. Required for every RGB-D rig that is searched.
    rig_ids:
        Search order. Defaults to every rig the provider knows, primary first.
    palm_patch_radius_px:
        Radius of the disc around the palm centre whose median depth is taken.
    min_depth_samples:
        How many valid depth pixels that disc must hold before its median is trusted. Below this
        the median is noise rather than a measurement.
    """

    def __init__(
        self,
        observer: HandObserver,
        *,
        provider: RigFrames,
        transforms: Mapping[str, np.ndarray],
        stereo: Optional[StereoCam3D] = None,
        camera_matrices: Optional[Mapping[str, np.ndarray]] = None,
        rig_ids: Optional[Sequence[str]] = None,
        palm_patch_radius_px: int = 15,
        min_depth_samples: int = 20,
    ) -> None:
        self.logger = create_logger("HandFinder", HAND_FINDER_LOG_FILE, log_dir=MODELS_LOG_DIR)
        self.observer = observer
        self.provider = provider
        self.stereo = stereo
        self.rig_ids = list(rig_ids) if rig_ids is not None else list(provider.rig_ids)
        self.palm_patch_radius_px = int(palm_patch_radius_px)
        self.min_depth_samples = int(min_depth_samples)

        self.transforms = {
            rig: np.asarray(matrix, dtype=np.float64) for rig, matrix in transforms.items()
        }
        for rig, matrix in self.transforms.items():
            if matrix.shape != (4, 4):
                raise ValueError(
                    f"transforms[{rig!r}] must be a 4x4 CAMERA->BASE matrix, got {matrix.shape}"
                )
        self.camera_matrices = {
            rig: np.asarray(matrix, dtype=np.float64)
            for rig, matrix in (camera_matrices or {}).items()
        }
        for rig, matrix in self.camera_matrices.items():
            if matrix.shape != (3, 3):
                raise ValueError(
                    f"camera_matrices[{rig!r}] must be a 3x3 intrinsics matrix, got {matrix.shape}"
                )

    # --- Public ----------------------------------------------------------------------------------

    def find_hand(self) -> tuple[Optional[LocatedHand], Optional[np.ndarray]]:
        """Search every rig in order for exactly one hand with usable depth.

        Exactly one, not the best of several: the answer decides where a robot may move, and two
        hands in the workspace is a reason to stop rather than to choose. Returns the located hand
        and an annotated BGR image, or `(None, None)`.
        """
        for rig_id in self.rig_ids:
            frame = self.provider.grab(rig_id)
            work = self._work_image(frame)
            observations = self.observer.observe(work)

            if len(observations) != 1:
                self.logger.debug(
                    "rig %s: %d hands seen, need exactly one; skipping", rig_id, len(observations)
                )
                continue

            observation = observations[0]
            position = self.locate(observation, frame, rig_id)
            if position is None:
                continue

            annotated = draw_hand_landmarks(
                work,
                observation.palm.landmarks,
                observation.palm.palm_center_xy,
                label=self._annotation(observation.gesture),
            )
            self.logger.info(
                "rig %s: hand at base (%.1f, %.1f, %.1f) mm, depth %.1f mm, gesture %s",
                rig_id, *position.position_base, position.depth_mm, observation.gesture.gesture,
            )
            return (
                LocatedHand(
                    position=position, gesture=observation.gesture, palm=observation.palm
                ),
                annotated,
            )

        return None, None

    def locate(
        self, observation: HandObservation, frame: AnyFrame, rig_id: str
    ) -> Optional[HandPosition3D]:
        """Base-frame position for one observed hand.

        `None` when the rig has no CAMERA->BASE transform or its depth at the palm is unusable.
        """
        transform = self.transforms.get(rig_id)
        if transform is None:
            self.logger.warning(
                "rig %s has no CAMERA->BASE transform, so its detections cannot be expressed in "
                "the base frame; skipping. Add it to `transforms` or drop the rig from rig_ids.",
                rig_id,
            )
            return None

        if self.provider.is_rgbd(rig_id):
            return self._locate_rgbd(observation, frame, rig_id, transform)
        return self._locate_stereo(observation, frame, rig_id, transform)

    # --- RGB-D (RealSense and any other depth camera) --------------------------------------------

    def _locate_rgbd(
        self,
        observation: HandObservation,
        frame: AnyFrame,
        rig_id: str,
        transform: np.ndarray,
    ) -> Optional[HandPosition3D]:
        if not isinstance(frame, RGBDFrame):
            raise TypeError(
                f"rig {rig_id!r} reports RGB-D but produced {type(frame).__name__}"
            )

        matrix = self.camera_matrices.get(rig_id)
        if matrix is None:
            raise ValueError(
                f"rig {rig_id!r} is RGB-D but no intrinsics were supplied for it. Back-projecting "
                f"a palm without a real K would return a confident, unmeasured position; supply "
                f"camera_matrices[{rig_id!r}] (from the rig's intrinsics file, or from the "
                f"RealSense streamer's get_intrinsics())."
            )

        depth = frame.depth
        if depth is None or depth.size == 0:
            self.logger.debug("rig %s: RGB-D frame carries no depth; skipping", rig_id)
            return None

        centre = observation.palm.palm_center_xy
        depth_mm = self._patch_depth_mm(depth, centre, rig_id)
        if depth_mm is None:
            return None

        fx, fy = float(matrix[0, 0]), float(matrix[1, 1])
        cx_i, cy_i = float(matrix[0, 2]), float(matrix[1, 2])
        cx, cy = centre
        position_cam = np.array(
            [(cx - cx_i) * depth_mm / fx, (cy - cy_i) * depth_mm / fy, depth_mm],
            dtype=np.float64,
        )
        return self._to_base(position_cam, centre, depth_mm, rig_id, transform, observation)

    def _patch_depth_mm(
        self, depth: np.ndarray, centre: tuple[float, float], rig_id: str
    ) -> Optional[float]:
        """Median depth over a disc at the palm centre, or `None` if too few pixels in it are valid.

        A median rather than the single centre pixel, because depth maps drop out on skin and at
        edges. A floor on the sample count, because a median over a handful of survivors is noise
        that would be handed onward as a hand position.
        """
        import cv2 as cv

        height, width = depth.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.circle(
            mask,
            (int(round(centre[0])), int(round(centre[1]))),
            self.palm_patch_radius_px,
            1,
            -1,
        )
        samples = depth[mask > 0].astype(np.float64)
        samples = samples[samples > 0.0]
        if samples.size < self.min_depth_samples:
            self.logger.debug(
                "rig %s: only %d valid depth pixels at the palm (need %d); skipping",
                rig_id, int(samples.size), self.min_depth_samples,
            )
            return None
        return float(np.median(samples))

    # --- Stereo ----------------------------------------------------------------------------------

    def _locate_stereo(
        self,
        observation: HandObservation,
        frame: AnyFrame,
        rig_id: str,
        transform: np.ndarray,
    ) -> Optional[HandPosition3D]:
        if self.stereo is None:
            raise ValueError(
                f"rig {rig_id!r} is a stereo rig but HandFinder was built without a StereoCam3D, "
                f"so its frames cannot be triangulated."
            )
        if not isinstance(frame, StereoFrame):
            raise TypeError(f"rig {rig_id!r} reports stereo but produced {type(frame).__name__}")

        import cv2 as cv

        centre = observation.palm.palm_center_xy
        height, width = frame.left.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        cv.circle(
            mask,
            (int(round(centre[0])), int(round(centre[1]))),
            self.palm_patch_radius_px,
            1,
            -1,
        )

        rig_index = self.provider.get_stereo_rig_index(rig_id)
        rect_left, rect_right = self.stereo.rectify(frame.left, frame.right, rig=rig_index)
        position_cam = self.stereo.compute_3D_point(
            rect_left, rect_right, mask=mask, reducer="median", unit="mm", rig=rig_index,
        )
        if position_cam is None:
            self.logger.debug("rig %s: stereo returned no 3-D point at the palm", rig_id)
            return None

        position_cam = np.asarray(position_cam, dtype=np.float64).reshape(-1)
        depth_mm = float(position_cam[2])
        if depth_mm <= 0.0:
            self.logger.debug(
                "rig %s: stereo depth %.1f mm is behind the camera; discarding", rig_id, depth_mm
            )
            return None
        return self._to_base(position_cam, centre, depth_mm, rig_id, transform, observation)

    # --- Shared ----------------------------------------------------------------------------------

    def _to_base(
        self,
        position_cam: np.ndarray,
        centre: tuple[float, float],
        depth_mm: float,
        rig_id: str,
        transform: np.ndarray,
        observation: HandObservation,
    ) -> HandPosition3D:
        rotation = transform[:3, :3]
        translation = transform[:3, 3]
        return HandPosition3D(
            position_base=rotation @ position_cam + translation,
            position_cam=position_cam,
            palm_center_xy=centre,
            depth_mm=depth_mm,
            rig_id=rig_id,
            handedness=observation.palm.handedness,
        )

    @staticmethod
    def _work_image(frame: AnyFrame) -> np.ndarray:
        """The BGR image a 2-D model should look at: colour for RGB-D, the left eye for stereo.

        `isinstance` rather than a check for a `color` attribute, which any object can carry and
        which lets a wrong frame type reach inference and fail several layers later.
        """
        if isinstance(frame, RGBDFrame):
            return frame.color
        if isinstance(frame, StereoFrame):
            return frame.left
        raise TypeError(f"unsupported frame type for hand detection: {type(frame).__name__}")

    @staticmethod
    def _annotation(gesture: GestureReading) -> Optional[str]:
        """The overlay label, or `None` when the reading is `HandGesture.NONE`."""
        if gesture.gesture is HandGesture.NONE:
            return None
        return f"{gesture.gesture.value} {gesture.confidence:.2f}"
