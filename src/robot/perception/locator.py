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

``Located.scene(i, robot_config)`` is the step from seeing a part to picking it: object i's surface as the grasp
``Scene``, the other objects as its obstacles, and a candidate's ``pose()`` and ``grip_width_mm`` as what
``Robot.pick`` takes. ``Located.set_down(i, grasp=, part_bottom_mm=)`` is the step from holding that part to putting
it on object i of another frame: a :class:`SetDown`, whose ``pose`` ``Robot.place`` takes with ``keep_out(i)``, measured
from the target's top, the part's hang below the grasp and the air the owner leaves (example 13, 2026-09-24). The
hang is measured from a LOWER bound on where the part stood, the grasp scene's ``declared_support_height_mm``, so an
error goes to more air and never presses the part into the target.

``Locator.for_cameras(tree, cameras)`` is one locator per camera of a cell over ONE perception backend: the detector
and the segmenter load their weights once for the cell, not once per camera.

A locator handed a view (``view=``, a ``LiveView`` of ``src/camera/live_view.py``) gives its camera a window there and
shows every ``Located`` it produces on it, the masks, labels and centres pinned, with the prompt and the colour image
the locate segmented: a wrist camera's window holds its masks on that image while the arm moves on. Handing the image
over is no read of the camera: the locate grabs as it would without a view. It says there why a locate failed, too.
Display only, read by its methods: a view that raises changes nothing a locate returns or raises.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core.keep_out import SegmentationOffer
from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
from src.robot.grasping.multiview.scene_geometry import to_base_mm
from src.robot.grasping.scene import Scene
from src.robot.perception.realsense_source import RealSenseVisionPerceptionSource

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig

__all__ = ["SET_DOWN_AIR_MM", "Located", "LocatedObject", "LocatedOrientation", "Locator", "LocatorRefused", "SetDown"]

logger = logging.getLogger(__name__)

#: The gap left between a set-down part and what it lands on when the hand opens, millimetres: the owner's choice for
#: example 13 (2026-09-24). The part comes down to it and drops that far, so a camera that reads the target's top
#: low by less than it still leaves air. The part's hang is measured from a lower bound on its bottom, so an error there
#: only adds to it (see :class:`SetDown`).
SET_DOWN_AIR_MM = 5.0

#: Which percentile of a target's surface heights is its top. A high percentile and not the maximum: the maximum of a
#: real cloud is a flying pixel at an edge or a stray reading above the surface, and one of them would lift the release
#: by as much. Up to 5 % of the surface may read high without moving the top, and a noisy flat top reads high by about
#: 1.6 of its noise's deviations, never low. A choice, not a measurement.
_TOP_PERCENTILE = 95.0

#: How many surface points a target's top is read from at least. Fewer say where a few pixels are, not where a surface
#: is. A choice, not a measurement: an object the locator places carries hundreds to thousands.
_MIN_TOP_POINTS = 20

#: How far below its top a target's surface still counts as the top a part is set down on, millimetres, for where the
#: middle of that top is. Wide enough for a flat top's depth noise; narrow enough that of the sides a tilted camera sees
#: only a strip along the top's own edges comes in, which stands on those edges and so widens nothing. A choice, not a
#: measurement.
_TOP_BAND_MM = 10.0


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
    #: Mask pixels that measure the background behind a depth step at the object's edge are not in it (see
    #: ``grasping.generation.depth_steps``): on a tilted view they are the table 40 mm behind the part.
    points_base_mm: np.ndarray
    #: The per-axis median of the surface, or ``None`` when there is no surface.
    centre_mm: "tuple[float, float, float] | None"
    orientation: LocatedOrientation | None = LocatedOrientation()

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

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


def _ascii(text: str) -> str:
    return text.encode("ascii", "backslashreplace").decode("ascii")


def _top_centre_xy(surface: np.ndarray, top_mm: float) -> tuple[float, float]:
    """The middle of a target's top, BASE X and Y: halfway across the extent of the surface within the top band.

    Not the median of the whole surface (``LocatedObject.centre_mm``): a camera tilted 45 degrees sees a box's top and the
    face toward it, and that face pulls the median toward the camera, 20 mm on a 40 mm cube, half of it, so a part set
    down there hangs over the top's near edge. Not the median of the band either: pixels crowd on the near half of a top
    seen at a slant. The extent runs between the same percentiles the top is read at, so a stray point is not an edge.
    """
    band = surface[surface[:, 2] >= top_mm - _TOP_BAND_MM][:, :2]
    low = np.percentile(band, 100.0 - _TOP_PERCENTILE, axis=0)
    high = np.percentile(band, _TOP_PERCENTILE, axis=0)
    return float((low[0] + high[0]) / 2.0), float((low[1] + high[1]) / 2.0)


def _finite_number(value: object, what: str) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        number = math.nan
    if not math.isfinite(number):
        raise ValueError(f"a set-down takes {what} as a finite number of millimetres, not {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class SetDown:
    """Where the tool sets a held part down on a located object, and what that was measured from.

    Nothing in it is written in a program. The target's top comes from the camera, the part's hang from the grasp and
    the part's bottom, the air from the owner. ``pose`` is the TCP pose ``Robot.place`` takes: the grasp's own turn, so
    the part hangs below the tool as it did when the hand closed, over the middle of the target's top (the surface
    within 10 mm of the top, halfway across its extent; not ``centre_mm``, which the sides a tilted camera sees pull
    toward it), at ``top_mm + hang_mm + air_mm``. The part's bottom then stands at least ``air_mm`` over the target's top
    when the hand opens, and ``Robot.place`` opens it only once its line in arrived.

    The owner's rule (2026-09-24) is 5 mm of air, released only then, and the part NEVER pressed into the target. A
    hang too short brings the part down into the target by as much, a hang too long only adds air, so the hang is
    measured from a LOWER bound on where the part's bottom was: the support the cell declares (see :meth:`onto`), never
    the part's lowest seen point. Every error in it goes toward more air; the cost is a larger drop for a part that
    stood higher than the declared surface.

    ``pose`` is ``None`` where the target's top cannot be read (no surface under its mask, or too few points) or the
    grasp did not stand above the part's bottom, and ``reason`` says which: nothing is set down blind.

    What it cannot know. That the part stayed where the fingers closed on it: a hand with no sensor measures nothing,
    and a part that slipped hangs lower than the grasp planned. That the camera reads the target where it stands: its
    top is read by the same hand-eye as the grasp, so a camera whose calibration reads the scene low by some
    millimetres leaves that much less air, which is what the air is for. And a target whose rim stands above its middle
    is topped at its rim, so a part set down over a lower middle drops the difference.
    """

    #: The located object the part goes on, by its label.
    target: str
    #: The TCP pose of the set-down in BASE, or ``None``; ``reason`` then says why.
    pose: Pose | None
    #: The target's top in BASE Z, mm: the 95th percentile of its surface heights. ``None`` where it could not be read.
    top_mm: float | None
    #: How far the part hangs below the grasp, mm: the grasp's Z less the part's bottom it was given. At least the
    #: real hang when that bottom is a lower bound on where the part stood, as the declared support is.
    hang_mm: float
    #: The gap between the part's bottom and the target's top when the hand opens, mm.
    air_mm: float
    #: How many surface points the top was read from.
    points: int
    #: Why there is no pose; empty when there is one.
    reason: str = ""

    @classmethod
    def onto(cls, target: LocatedObject, *, grasp: Pose, part_bottom_mm: float,
             air_mm: float = SET_DOWN_AIR_MM) -> "SetDown":
        """Set the part the tool grasped at ``grasp`` down on ``target``, ``air_mm`` over its top when the hand opens.

        ``part_bottom_mm`` is where the part's bottom was when it was grasped, in BASE Z, and it must be a LOWER bound
        on it: the grasp's Z less it is the hang, and a bottom read too high shortens the hang and brings the part
        down into the target by as much. Pass the ``declared_support_height_mm`` of the scene its grasp came from
        (``located.scene(i, robot_config)``): the table, or the container floor, the cell declares, which no part
        stands below. Every error then goes toward more air. The cost, plainly: a part taken off a raised block, or
        off another part, hangs further below the grasp than that, and it drops the block's height on top of the air
        (a 30 mm block: 35 mm). A caller that KNOWS where the part stood (a fixture of known height, a part it put
        there itself) passes that height, and gets the air exactly.

        Not the scene's ``support_height_mm``. That is the declared support raised to the part's lowest SEEN point,
        the height the grasp's clearance is planned from, and an UPPER bound on the part's base: a camera never sees
        below the base, but an occluder in front, a mask that stops short, or a tilted view that misses the foot of the
        near face leave the lowest seen point above it by as much as they hid. Measured from there (the review of
        2026-09-24, the real ``Scene`` and this method), 8 mm of the near face unseen set the part down 3 mm INTO the
        target and 12 mm set it 7 mm in; on the owner's 45 degree wrist view 6 mm hidden shortened the hang by 4.7 to
        6.2 mm.

        Raises ``ValueError`` for a programmer's error: a grasp not in BASE, a bottom or an air that is not a finite
        number, a negative air.
        """
        if grasp.frame is not Frame.BASE:
            raise ValueError(f"the grasp is in {grasp.frame.value!r}, and a set-down is measured in BASE")
        bottom = _finite_number(part_bottom_mm, "what the part stood on")
        air = _finite_number(air_mm, "the air")
        if air < 0.0:
            raise ValueError(f"the air under a set-down part is {air} mm; below zero it presses the part into the target")
        label = str(target.label)
        grasp_z = float(grasp.position_mm[2])
        hang = grasp_z - bottom
        points = np.asarray(target.points_base_mm, dtype=np.float64).reshape(-1, 3)
        surface = points[np.isfinite(points).all(axis=1)]
        count = int(surface.shape[0])

        def refused(reason: str, top: "float | None" = None) -> "SetDown":
            return cls(target=label, pose=None, top_mm=top, hang_mm=hang, air_mm=air, points=count, reason=reason)

        if count == 0:
            return refused(f"the camera measured no surface under {label!r}, so its top is unknown and nothing is set "
                           "down blind")
        if count < _MIN_TOP_POINTS:
            return refused(f"the camera measured {count} point(s) of {label!r}, fewer than the {_MIN_TOP_POINTS} a top "
                           "is read from, so its top is unknown and nothing is set down blind")
        top = float(np.percentile(surface[:, 2], _TOP_PERCENTILE))
        if hang <= 0.0:
            return refused(f"the grasp at Z {grasp_z:.1f} mm does not stand above the part's bottom "
                           f"({bottom:.1f} mm), so how far the part hangs below the tool is unknown", top)
        centre = _top_centre_xy(surface, top)
        pose = Pose(position_mm=np.array([centre[0], centre[1], top + hang + air]),
                    quaternion_xyzw=np.asarray(grasp.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                    label=f"set down on {label}")
        return cls(target=label, pose=pose, top_mm=top, hang_mm=hang, air_mm=air, points=count)

    @property
    def ok(self) -> bool:
        """Whether there is a pose to set the part down at."""
        return self.pose is not None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if self.pose is None:
            return _ascii(f"set down on {self.target!r}: NO POSE, {self.reason}")
        x, y, z = (float(v) for v in self.pose.position_mm)
        return _ascii(
            f"set down on {self.target!r}: its top at {self.top_mm:.1f} mm (the {_TOP_PERCENTILE:.0f}th percentile of "
            f"{self.points} point(s)), the part hanging {self.hang_mm:.1f} mm below the grasp, {self.air_mm:.1f} mm of "
            f"air: the tool to ({x:.1f}, {y:.1f}, {z:.1f}) mm, turned as it grasped")

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "target": self.target,
            "ok": self.ok,
            "pose_mm": None if self.pose is None else [float(v) for v in self.pose.position_mm],
            "quaternion_xyzw": None if self.pose is None else [float(v) for v in self.pose.quaternion_xyzw],
            "top_mm": self.top_mm,
            "top_percentile": _TOP_PERCENTILE,
            "hang_mm": self.hang_mm,
            "air_mm": self.air_mm,
            "points": self.points,
            "reason": self.reason,
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

    def scene(self, target: int, robot_config: "RobotConfig") -> Scene:
        """Object ``target`` as the :class:`Scene` its grasps are planned on, every other object an obstacle.

        The support height and the jaw come from ``robot_config`` through ``Scene.from_robot_config``, the
        target cloud is the object's measured surface, and every other object this frame located with a surface
        goes in as observed obstacle points. A candidate gives the BASE pose and the width ``Robot.pick`` takes,
        and :meth:`keep_out` holds the same object out of the planner world while the arm reaches for it:

            best = located.scene(0, app.robot).grasps().best
            if best is not None:
                robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))

        The camera sees one side of a part, so the scene extrudes the seen footprint down to the support. The
        support is the declared one, raised to the object's own lowest point when its surface reaches down to what
        it stands on, and the jaw is the hand ``grasping.gripper_geometry`` describes, both as the cell's pick path
        takes them. The generator is the geometric one whatever ``grasping.calculator`` says, and
        ``SceneGrasps.generator`` says so on every result. An object with no surface under its mask gives a scene
        with no grasps.
        """
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        index = int(target)
        others = [obj.points_base_mm for position, obj in enumerate(self.objects)
                  if position != index and obj.points_base_mm.shape[0]]
        return Scene.from_robot_config(
            robot_config, self.objects[index].points_base_mm,
            obstacle_points_base_mm=np.concatenate(others, axis=0) if others else None,
        )

    def set_down(self, target: int, *, grasp: Pose, part_bottom_mm: float,
                 air_mm: float = SET_DOWN_AIR_MM) -> SetDown:
        """Where the tool sets the part it grasped at ``grasp`` down on object ``target``: :meth:`SetDown.onto`.

        The step after :meth:`scene` and ``Robot.pick``. The part's bottom is the support the scene its grasp came from
        declares, a lower bound on where the part stood, and :meth:`keep_out` holds the target out of the planner world
        while the part comes down onto it:

            scene = seen.scene(0, app.robot)
            best = scene.grasps().best
            ...                                  # robot.pick(best.pose(), ...), then locate the target
            set_down = onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.declared_support_height_mm)
            if set_down.pose is not None:
                robot.place(set_down.pose, keep_out=onto.keep_out(0))

        The tool keeps the grasp's turn over the middle of the target's top, at that top (the 95th percentile of the
        target's surface heights) plus the part's hang plus ``air_mm``, the owner's 5 mm unless chosen. From the
        declared support the part lands with at least that air, and a part taken off a raised block drops the block's
        height on top of it; a caller that knows where the part stood passes that as ``part_bottom_mm`` instead. Not
        ``scene.support_height_mm``: it is raised to the part's lowest SEEN point and presses the part into the target
        by whatever the camera missed of its foot (see :meth:`SetDown.onto`).
        """
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        return SetDown.onto(self.objects[int(target)], grasp=grasp, part_bottom_mm=part_bottom_mm, air_mm=air_mm)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

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


def _on_view(view: Any, method: str, *args: Any, **keywords: Any) -> None:
    """``view.<method>(*args, **keywords)``, for display only: a view that lacks it, or raises, changes nothing; it is
    logged."""
    call = getattr(view, method, None)
    if not callable(call):
        return
    try:
        call(*args, **keywords)
    except Exception as exc:  # noqa: BLE001 (a window is never a reason to fail a locate)
        logger.debug("locator: the view's %s raised %s: %s", method, type(exc).__name__, exc)


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
    """One open camera, a perception backend, and what it takes to place its frames in BASE.

    ``view``, where given, is the cameras' windows (a ``LiveView``): the camera gets a window there, and every
    ``Located`` is shown on it (see the module docstring).
    """

    def __init__(self, *, camera: Any, backend: Any, calibration: Any, tool_pose: "Callable[[], Pose] | None",
                 attempts: int, view: Any = None) -> None:
        self._camera = camera
        self._backend = backend
        self._calibration = calibration
        self._tool_pose = tool_pose
        self._attempts = int(attempts)
        self._view = view
        if view is not None:
            _on_view(view, "watch", camera)

    @classmethod
    def from_parts(
        cls,
        *,
        camera: Any,
        backend: Any,
        tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
        attempts: Maybe[int] = UNSET,
        tool_frame: Maybe[Any] = UNSET,
        view: Any = None,
    ) -> "Locator":
        """A locator over ``camera``, an open owner answering ``rig_id``, ``source``, ``handle()`` and ``calibration()``.

        ``tool_pose`` is the arm's ``get_tcp_pose`` and ``tool_frame`` the cell's ``robot.gripper.tool_frame``, both
        required for a camera on the wrist and ignored for a fixed one. ``attempts`` is how many more frames a wrist
        camera takes while the tool moved; unset, the schema's ``perceived.fresh_frame_attempts`` default. Refuses a
        rig with no depth, a rig that declares no calibration (``RigNotCalibrated`` names its key), a wrist rig with
        no TCP reader, and a wrist rig whose calibration was not solved against ``tool_frame``. ``view`` (a
        ``LiveView``) shows each ``Located`` on the camera's window; it is handed the camera once every refusal passed.
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
        return cls(camera=camera, backend=backend, calibration=calibration, tool_pose=reader, attempts=int(attempts),
                   view=view)

    @classmethod
    def from_config(cls, robot_cfg: Any, models_cfg: Any, *, camera: Any,
                    tool_pose: "Maybe[Callable[[], Pose]]" = UNSET, view: Any = None) -> "Locator":
        """A locator for the cell ``robot_cfg`` describes, with the backend ``models_cfg`` builds, as the cell builds it.

        Every refusal of :meth:`from_parts` runs before the backend is built, so a rig with no depth, no declared
        calibration or no TCP reader is refused before the detector and the segmenter load. ``view`` is handed to the
        locator built, as :meth:`from_parts` hands it. For several cameras, :meth:`for_cameras` builds one backend for
        all of them.
        """
        (locator,) = cls._over_one_backend(robot_cfg, models_cfg, [camera], tool_pose=tool_pose, view=view)
        return locator

    @classmethod
    def from_tree(cls, tree: Any, *, camera: Any, tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
                  view: Any = None) -> "Locator":
        """A locator for the cell a loaded tree describes, with the perception stack its models section
        builds (see :meth:`from_config`). A tree that did not load is refused with its own refusal, as
        ``ConfigError``. ``view`` (a ``LiveView``) shows each ``Located`` on the camera's window."""
        return cls.from_config(tree.robot, tree.app_config.models, camera=camera, tool_pose=tool_pose, view=view)

    @classmethod
    def for_cameras(cls, tree: Any, cameras: "Sequence[Any]", *, tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
                    view: Any = None) -> "list[Locator]":
        """One locator per open camera of the cell a loaded tree describes, in the order given, all over ONE backend.

        Building a backend loads the detector's and the segmenter's weights (``PerceptionSpec.build``, with no cache),
        so a locator per camera from :meth:`from_tree` loads every model once per camera: N sets of weights in memory,
        and N loads' time, for what one set answers. The backend is stateless per call, and the cell's own pick path
        shares one across its cameras for that reason (``autonomous_grasp.cells``); so does this.

        Every camera is checked as :meth:`from_parts` checks one, all of them before anything loads: one rig it cannot
        place (no depth, no declared calibration, a wrist rig with no TCP reader or solved against another tool frame)
        refuses the lot, with no model loaded and no window opened. ``tool_pose`` goes to every camera on the wrist and
        is ignored by a fixed one; ``view`` (a ``LiveView``) gives each camera its window. No camera at all is a
        ``LocatorRefused``, since there would be nothing to locate with. A tree that did not load is refused with its
        own refusal, as ``ConfigError``.
        """
        return cls._over_one_backend(tree.robot, tree.app_config.models, list(cameras), tool_pose=tool_pose, view=view)

    @classmethod
    def _over_one_backend(cls, robot_cfg: Any, models_cfg: Any, cameras: "list[Any]", *,
                          tool_pose: "Maybe[Callable[[], Pose]]", view: Any) -> "list[Locator]":
        """Check every camera, then build the backend ``models_cfg`` describes once and hand it to a locator each."""
        if not cameras:
            raise LocatorRefused("no camera was handed in, so there is nothing to locate with and no model is loaded")
        attempts = int(robot_cfg.safety.planning_world.perceived.fresh_frame_attempts)
        checked = [cls.from_parts(camera=camera, backend=None, tool_pose=tool_pose, attempts=attempts,
                                  tool_frame=robot_cfg.gripper.tool_frame) for camera in cameras]
        from src.models.perception_spec import PerceptionSpec

        backend = PerceptionSpec.from_config(models_cfg).build()
        return [cls(camera=camera, backend=backend, calibration=each._calibration, tool_pose=each._tool_pose,
                    attempts=each._attempts, view=view) for camera, each in zip(cameras, checked, strict=True)]

    @property
    def on_the_wrist(self) -> bool:
        """Whether this locator's camera rides on the wrist, so it sees only what the arm points it at.

        A program moves the arm to a look before such a camera locates (``robot.move_joints(look)``), and every frame is
        placed by the tool pose at its shutter. A fixed camera locates as the arm stands. Read off the calibration the
        locator was built with, as :attr:`Located.mounting` is.
        """
        return self._tool_pose is not None

    def locate(self, prompt: str) -> Located:
        """Ground ``prompt`` in one frame and place every object in BASE. Raises ``PerceptionFrameMoved`` on the wrist.

        With a view, what was located is pinned on the camera's window, handed over with ``prompt`` and the colour
        image (BGR) of the frame it was located in, and a locate that raised says why there before it raises as it
        would have.
        """
        try:
            located, image = self._locate(prompt)
        except Exception as exc:
            if self._view is not None:
                _on_view(self._view, "note", f"locate failed: {type(exc).__name__}: {exc}",
                         str(getattr(self._camera, "rig_id", "camera")))
            raise
        if self._view is not None:
            _on_view(self._view, "show_located", located, image=image, prompt=prompt)
        return located

    def _locate(self, prompt: str) -> "tuple[Located, np.ndarray | None]":
        """What one frame located, and that frame's colour image as the backend segmented it (BGR), where the source
        kept it."""
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
            # The surface is the mask less the pixels behind a depth step; the mask itself stays as segmented,
            # because ``keep_out`` holds the whole of it out of the planner world.
            surface = mask & ~pixels_behind_depth_steps(mask, depth)
            points = to_base_mm(surface, depth, frame.intrinsics, camera_to_base)
            centre = None if points.shape[0] == 0 else tuple(float(v) for v in np.median(points, axis=0))
            score = getattr(seg, "score", None)
            box = getattr(seg, "bbox_xyxy", None)
            objects.append(LocatedObject(
                label=str(getattr(seg, "label", "") or f"object_{len(objects)}"),
                score=None if score is None or not math.isfinite(float(score)) else float(score),
                box_px=None if box is None else tuple(float(v) for v in box),  # type: ignore[arg-type]
                mask=mask, points_base_mm=points, centre_mm=centre,  # type: ignore[arg-type]
            ))
        located = Located(
            camera=rig_id,
            captured_at_s=float(frame.timestamp if frame.timestamp is not None else float("nan")),
            mounting="eye_in_hand" if wrist else "eye_to_hand",
            tool_to_base_mm=None if tool_to_base is None else tuple(tuple(float(v) for v in row) for row in tool_to_base),
            objects=tuple(objects),
        )
        # The source publishes the frame's colour as RGB; the backend was handed it, and a view shows it, as BGR.
        return located, None if frame.rgb is None else np.asarray(frame.rgb)[..., ::-1]
