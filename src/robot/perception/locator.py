"""Where the prompted objects are, in BASE, from one open camera: the locator.

A pick loop grounds a prompt, grasps and moves in one run. User code that wants to place a part, check a fixture or
reach into a bin needs the first half alone: which objects the camera sees, where they are in the cell, and what a
planner world has to leave out to reach one. ``Locator.locate(prompt)`` answers that from the frame the pick loop
would have taken, through the same RealSense source, so the warm-ups, the intrinsics, the labels, the mask
completion, the shutter stamp and the wrist check are the pick frame's.

The locator never opens or releases a camera; the caller owns the device. It refuses what it cannot place: a rig
with no depth of its own, a rig that declares no calibration, a wrist rig with no reader of the arm's TCP, and a wrist
rig whose calibration was not solved against the cell's flange to TCP. An empty result is an answer and says it
cannot be told apart from a detector that failed.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Pose
from src.robot.core.keep_out import SegmentationOffer
from src.robot.grasping.multiview.scene_geometry import to_base_mm
from src.robot.perception.realsense_source import RealSenseVisionPerceptionSource

__all__ = ["Located", "LocatedObject", "LocatedOrientation", "Locator", "LocatorRefused"]


class LocatorRefused(ValueError):
    """A locator that cannot place what its camera sees, refused before it grabs, or a frame it cannot place."""


@dataclass(frozen=True, slots=True)
class LocatedOrientation:
    """How far an object's orientation is known. No estimator runs, so every orientation says it is unmeasured."""

    quality: str = "unmeasured"
    measured: bool = False


@dataclass(frozen=True, slots=True, eq=False)
class LocatedObject:
    """One object the camera grounded: its label and score, its mask, and its surface in BASE."""

    label: str
    score: float | None
    #: The detection box in pixels, ``(x0, y0, x1, y1)``, or ``None`` when the backend gave none.
    box_px: "tuple[float, float, float, float] | None"
    mask: np.ndarray
    #: The measured surface under the mask, ``(N, 3)`` BASE millimetres; empty when the mask selects no valid depth.
    points_base_mm: np.ndarray
    #: The per-axis median of the surface, or ``None`` when there is no surface.
    centre_mm: "tuple[float, float, float] | None"
    orientation: LocatedOrientation | None = LocatedOrientation()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        where = ("no surface under its mask" if self.centre_mm is None
                 else "centre ({:.1f}, {:.1f}, {:.1f}) mm".format(*self.centre_mm))
        score = "" if self.score is None else f" score {self.score:.2f}"
        return f"{self.label}{score}: {where}, {self.points_base_mm.shape[0]} point(s)"

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: no mask and no points."""
        return {
            "label": self.label,
            "score": self.score,
            "box_px": None if self.box_px is None else list(self.box_px),
            "centre_mm": None if self.centre_mm is None else list(self.centre_mm),
            "points": int(self.points_base_mm.shape[0]),
            "orientation": None if self.orientation is None else {
                "quality": self.orientation.quality, "measured": self.orientation.measured},
        }


@dataclass(frozen=True, slots=True, eq=False)
class Located:
    """What one frame of one camera located, stamped at its shutter."""

    camera: str
    captured_at_s: float
    #: ``eye_to_hand`` or ``eye_in_hand``.
    mounting: str
    #: Where the tool stood at the shutter, 4x4 millimetres, for a camera on the wrist; ``None`` for a fixed camera.
    tool_to_base_mm: "tuple[tuple[float, ...], ...] | None"
    objects: tuple[LocatedObject, ...]

    def keep_out(self, target: int) -> SegmentationOffer:
        """What a planner world has to leave out to reach object ``target``: its box, and this frame's masks.

        Hand it to ``robot.core.keep_out.keeping_out`` around the motions that reach for the object.
        """
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        chosen_object = self.objects[int(target)]
        points = chosen_object.points_base_mm
        return SegmentationOffer(
            captured_at_s=self.captured_at_s,
            camera=self.camera,
            labelled_masks=tuple((obj.label, obj.mask) for obj in self.objects),
            exclude_masks=(chosen_object.mask,),
            target_points_base_mm=points if points.shape[0] else None,
            target_label=chosen_object.label,
        )

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if not self.objects:
            return (f"camera {self.camera!r} located nothing at {self.captured_at_s:.3f} s: an empty scene and a "
                    "detector that failed read the same here")
        rows = [f"  {index}: {obj.render()}" for index, obj in enumerate(self.objects)]
        return "\n".join([f"camera {self.camera!r} ({self.mounting}) located {len(self.objects)} object(s) at "
                          f"{self.captured_at_s:.3f} s:", *rows])

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "camera": self.camera,
            "captured_at_s": self.captured_at_s,
            "mounting": self.mounting,
            "tool_to_base_mm": None if self.tool_to_base_mm is None else [list(row) for row in self.tool_to_base_mm],
            "objects": [obj.to_dict() for obj in self.objects],
        }


class _DepthOnly:
    """The camera's handle, refusing a frame that carries a stereo pair and no depth before the source reads it."""

    def __init__(self, handle: Any, camera: str) -> None:
        self._handle = handle
        self._camera = camera
        self.rig_id = camera

    def grab(self) -> Any:
        frame = self._handle.grab()
        if getattr(frame, "depth", None) is None and hasattr(frame, "left") and hasattr(frame, "right"):
            raise LocatorRefused(
                f"camera {self._camera!r} answered with a stereo pair, which carries no depth of its own, so nothing "
                "it sees can be placed in BASE: locate with an RGB-D rig")
        return frame

    def get_intrinsics(self) -> Any:
        return self._handle.get_intrinsics()


class Locator:
    """One open camera, a perception backend, and what it takes to place its frames in BASE."""

    def __init__(self, *, camera: Any, backend: Any, calibration: Any, tool_pose: "Callable[[], Pose] | None",
                 attempts: int) -> None:
        self._camera = camera
        self._backend = backend
        self._calibration = calibration
        self._tool_pose = tool_pose
        self._attempts = int(attempts)

    @classmethod
    def from_parts(
        cls,
        *,
        camera: Any,
        backend: Any,
        tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
        attempts: Maybe[int] = UNSET,
        tool_frame: Maybe[Any] = UNSET,
    ) -> "Locator":
        """A locator over ``camera``, an open owner answering ``rig_id``, ``source``, ``handle()`` and ``calibration()``.

        ``tool_pose`` is the arm's ``get_tcp_pose`` and ``tool_frame`` the cell's ``robot.gripper.tool_frame``, both
        required for a camera on the wrist and ignored for a fixed one. ``attempts`` is how many more frames a wrist
        camera takes while the tool moved; unset, the schema's ``perceived.fresh_frame_attempts`` default. Refuses a
        rig with no depth, a rig that declares no calibration (``RigNotCalibrated`` names its key), a wrist rig with
        no TCP reader, and a wrist rig whose calibration was not solved against ``tool_frame``.
        """
        rig_id = str(getattr(camera, "rig_id", "camera"))
        source = getattr(camera, "source", "rgbd")
        if source != "rgbd":
            raise LocatorRefused(
                f"camera {rig_id!r} is a {source!r} rig, which carries no depth of its own, so nothing it sees can be "
                "placed in BASE: locate with an RGB-D rig")
        calibration = camera.calibration()
        if not chosen(attempts):
            from src.config.schema.robot.safety_schema import PerceivedWorldConfig

            attempts = int(PerceivedWorldConfig.model_fields["fresh_frame_attempts"].default)
        reader = None
        if str(calibration.mounting_mode) == "eye_in_hand":
            if not chosen(tool_pose) or not callable(tool_pose):
                raise LocatorRefused(
                    f"camera {rig_id!r} is on the wrist and no reader of the arm's TCP was handed in, so where it "
                    "stood at each shutter would be unknown: pass tool_pose=arm.get_tcp_pose")
            if not chosen(tool_frame):
                raise LocatorRefused(
                    f"camera {rig_id!r} is on the wrist and no tool frame was handed in, so its calibration cannot be "
                    "held to the flange to TCP this cell applies: pass tool_frame=robot_config.gripper.tool_frame")
            from src.calibration.rig_calibration import flange_to_tcp_refusal

            stale = flange_to_tcp_refusal(calibration, tool_frame)
            if stale is not None:
                raise LocatorRefused(stale)
            reader = tool_pose
        return cls(camera=camera, backend=backend, calibration=calibration, tool_pose=reader, attempts=int(attempts))

    @classmethod
    def from_config(cls, robot_cfg: Any, models_cfg: Any, *, camera: Any,
                    tool_pose: "Maybe[Callable[[], Pose]]" = UNSET) -> "Locator":
        """A locator for the cell ``robot_cfg`` describes, with the backend ``models_cfg`` builds, as the cell builds it."""
        from src.models.perception_spec import PerceptionSpec

        backend = PerceptionSpec.from_config(models_cfg).build()
        return cls.from_parts(
            camera=camera, backend=backend, tool_pose=tool_pose,
            attempts=int(robot_cfg.safety.planning_world.perceived.fresh_frame_attempts),
            tool_frame=robot_cfg.gripper.tool_frame,
        )

    def locate(self, prompt: str) -> Located:
        """Ground ``prompt`` in one frame and place every object in BASE. Raises ``PerceptionFrameMoved`` on the wrist."""
        rig_id = str(getattr(self._camera, "rig_id", "camera"))
        source = RealSenseVisionPerceptionSource(
            streamer=_DepthOnly(self._camera.handle(), rig_id), backend=self._backend, prompt=prompt)
        wrist = self._tool_pose is not None
        if wrist:
            source.stamp_tool_pose_with(
                self._tool_pose,  # type: ignore[arg-type]
                motion_tolerance=(float(self._calibration.shutter_motion_tolerance_mm),
                                  float(self._calibration.shutter_motion_tolerance_deg)),
                attempts=self._attempts,
            )
        frame = source.acquire()
        if wrist:
            if frame.tool_pose is None:
                raise LocatorRefused(f"camera {rig_id!r} is on the wrist and its frame carries no tool pose")
            tool_to_base = np.asarray(frame.tool_pose.to_matrix(), dtype=np.float64)
            camera_to_base = tool_to_base @ np.asarray(self._calibration.camera_to_tool().to_matrix(), dtype=np.float64)
        else:
            tool_to_base = None
            camera_to_base = np.asarray(self._calibration.camera_to_base().to_matrix(), dtype=np.float64)
        depth = frame.surface_depth_map if frame.surface_depth_map is not None else frame.depth_map
        objects: list[LocatedObject] = []
        for seg in frame.segmentations:
            mask = np.asarray(getattr(seg, "mask")).astype(bool)
            points = to_base_mm(mask, depth, frame.intrinsics, camera_to_base)
            centre = None if points.shape[0] == 0 else tuple(float(v) for v in np.median(points, axis=0))
            score = getattr(seg, "score", None)
            box = getattr(seg, "bbox_xyxy", None)
            objects.append(LocatedObject(
                label=str(getattr(seg, "label", "") or f"object_{len(objects)}"),
                score=None if score is None or not math.isfinite(float(score)) else float(score),
                box_px=None if box is None else tuple(float(v) for v in box),  # type: ignore[arg-type]
                mask=mask, points_base_mm=points, centre_mm=centre,  # type: ignore[arg-type]
            ))
        return Located(
            camera=rig_id,
            captured_at_s=float(frame.timestamp if frame.timestamp is not None else float("nan")),
            mounting="eye_in_hand" if wrist else "eye_to_hand",
            tool_to_base_mm=None if tool_to_base is None else tuple(tuple(float(v) for v in row) for row in tool_to_base),
            objects=tuple(objects),
        )
