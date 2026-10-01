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
hang is measured from a LOWER bound on where the part stood, the grasp scene's ``part_bottom_mm``, so an error goes to
more air and never presses the part into the target.

``Locator.look_around(prompt, looks, robot=robot)`` is how a camera on the wrist finds a part (the owner, 2026-09-28 and
2026-09-29): the arm goes to each look in turn through the robot's own verbs, the camera locates there, and each look is
fused with the looks before it by association at the wrist's tolerances, so object 0 of the first look that located
anything is the part and stays it across the looks. The part's grasp is computed again after every look, on everything
the looks saw of it, and the looking stops at the first look whose grasp is valid and which the looks agree on;
``both_faces=True`` asks, too, that both jaw contact faces of that grasp were seen, and refuses the part where no look
showed them. The goal is a safe grasp point, never a full scan, so it is how a part to grip is found: what a held part
is set down on has no grasp to judge, and is located look by look until one sees it (example 13). The frames of the
looks are held in the arm's live planner world from there until ``Robot.pick`` ends, so every motion of the pick is
planned against all of them. A fixed camera locates once, where it stands.

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
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core.keep_out import SegmentationOffer
from src.robot.core.shutter_motion import camera_to_base_at_shutter
from src.robot.grasping.generation.depth_steps import pixels_behind_depth_steps
from src.robot.grasping.geometry.closing_axis import (
    CLOSING_AXIS_TOLERANCE_DEG,
    ClosingAxis,
    ClosingAxisLike,
    closing_axis_of,
)
from src.robot.grasping.multiview.scene_geometry import to_base_mm
from src.robot.grasping.scene import Scene
from src.robot.perception.realsense_source import RealSenseVisionPerceptionSource

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig
    from src.robot.execution.looks import Look

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

#: What :meth:`Locator.look_around` calls the pose the arm stands at when it is handed no looks: it looks from there, as
#: a pick handed no looks perceives there (the pick loop's name for it).
_HERE = "here"

#: How many surface points object 0 of a look needs before :meth:`Locator.look_around` keeps it as the part, the count a
#: target's top is read from. A mask with fewer measured points under it (a dark or shiny face at a bad angle, which the
#: detector masks and the depth does not measure) says where the part is in the image, not where its surface is in the
#: cell: nothing a later look sees would associate with it, and every look after it would be left out as one that did
#: not see the part. A choice, not a measurement.
_MIN_PART_POINTS = _MIN_TOP_POINTS


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
    measured from a LOWER bound on where the part's bottom was: the support the cell declares, lowered to where two or
    more looks of a wrist camera measured the part's foot only where that lies below it (``Scene.part_bottom_mm``, see
    :meth:`onto`), and never raised to it. Every error in it goes toward more air; the cost is a larger drop for a part
    that stood higher than the declared surface.

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
        down into the target by as much. Pass the ``part_bottom_mm`` of the scene its grasp came from
        (``located.scene(i, robot_config)``): the table, or the container floor, the cell declares, which no part
        stands below, lowered only where two or more looks of a wrist camera measured the part standing below it (see
        ``Scene.part_bottom_mm``, which never raises it, and keeps the declared value for one view). Its
        ``declared_support_height_mm`` is the same bound without the measurement. Every error then goes toward more
        air. The cost, plainly: a part taken off a raised block, or
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
    """What one frame of one camera located, stamped at its shutter; or, from :meth:`Locator.look_around` on a camera on
    the wrist, what all its looks located, fused.

    A looked-around ``Located`` is the frame of the last look that saw the part, with every object's surface as all the
    looks saw it: object 0 is the part the looks kept, its cloud fused over :attr:`looks_fused`, and an object only an
    earlier look saw is kept too, with no pixels in this frame, so it stays an obstacle to the part's grasps.
    :attr:`refused` set says nothing may be planned on it: :meth:`scene` and :meth:`set_down` raise ``LocatorRefused``.
    """

    camera: str
    #: The frame's shutter, host seconds; for a look around that reached none of its looks, when it ended: no frame was
    #: taken.
    captured_at_s: float
    #: ``eye_to_hand`` or ``eye_in_hand``.
    mounting: str
    #: Where the tool stood at the shutter, 4x4 millimetres, for a camera on the wrist; ``None`` for a fixed camera.
    tool_to_base_mm: "tuple[tuple[float, ...], ...] | None"
    objects: tuple[LocatedObject, ...]
    #: The looks :meth:`Locator.look_around` located from, in the order the arm reached them (``here`` for the pose it
    #: stood at when handed none), a look refused before anything was sent left out. Empty for one locate.
    looks: tuple[str, ...] = ()
    #: The looks whose surfaces make up object 0's cloud, this frame's look first. Empty for one locate.
    looks_fused: tuple[str, ...] = ()
    #: Whether each jaw contact face of object 0's best grasp was seen, ``(jaw 1, jaw 2)``; ``None`` where it was not
    #: judged: one locate, no grasp, or a grasp the check cannot read.
    jaw_faces_seen: "tuple[bool, bool] | None" = None
    #: Why nothing may be planned on what was located, or ``""``: a look the arm did not reach and what its motion
    #: report said, a controller that stopped, or, with ``both_faces``, the contact face of the grasp no look saw.
    refused: str = ""
    #: The hand-eye check of the looks: how far apart, in mm, the looks measured the surface of the part they shared
    #: (median); ``None`` where fewer than two looks shared one. Above ``association.HAND_EYE_DRIFT_WARN_MM`` it is said
    #: that the calibration may have drifted, and nothing else is done about it.
    hand_eye_gap_mm: float | None = None
    #: How far the one view the looks generated, once the declared looks were used up and the grasp was still not safe
    #: enough, turned about the part, degrees (``src/robot/execution/generated_view.py``); its look is among
    #: :attr:`looks`. ``None`` where no view was generated.
    generated_view_deg: float | None = None
    #: Where the frames of the looks were kept, for training (``look_around(..., record_views=True)``,
    #: ``src/robot/execution/record_views.py``); ``""`` where they were not.
    views_file: str = ""
    #: The closing axis :meth:`Locator.look_around` judged object 0's grasp along (``closing_axis``), ``None`` for the best
    #: grasp of any axis, and for one locate. :meth:`scene` hands it on, so the grasp gripped is the grasp judged.
    closing_axis: "ClosingAxis | None" = None

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
        with no grasps. A ``Located`` that was :attr:`refused` raises ``LocatorRefused`` with the reason. Where the
        cell names how its hand and camera naturally stand (``robot.natural_closing_axis``), every grasp is turned the
        way round nearer it, as the pick loop turns its own, and a look around judges the grasp so turned.

        The scene's ``part_bottom_mm``, the bottom a set-down hangs the part from, is the declared support, lowered
        where the looks measured the part standing below it only for object 0 of a look around whose part two or more
        looks fused (:attr:`looks_fused`, addendum 7.6); one view, a fixed camera's above all, keeps the declared
        support as before.

        Object 0's scene of a look around that judged its grasp (from :attr:`looks`, on its jaw faces,
        :attr:`jaw_faces_seen`, or along a closing axis, :attr:`closing_axis`) chooses along the axis the looks judged it
        along, and among grasps of any axis where they named none: ``grasps()`` does so when asked for no axis, and
        refuses any other (``ValueError``), so the grasp gripped is the grasp the looks judged.
        """
        self._refuse_if_refused()
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        index = int(target)
        others = [obj.points_base_mm for position, obj in enumerate(self.objects)
                  if position != index and obj.points_base_mm.shape[0]]
        scene = Scene.from_robot_config(
            robot_config, self.objects[index].points_base_mm,
            obstacle_points_base_mm=np.concatenate(others, axis=0) if others else None,
        )
        if index == 0 and len(self.looks_fused) >= 2:
            scene = replace(scene, _bottom_from_looks=True)
        if index == 0 and (self.looks or self.jaw_faces_seen is not None or self.closing_axis is not None):
            scene = replace(scene, _judged_along=self.closing_axis)
        return scene

    def set_down(self, target: int, *, grasp: Pose, part_bottom_mm: float,
                 air_mm: float = SET_DOWN_AIR_MM) -> SetDown:
        """Where the tool sets the part it grasped at ``grasp`` down on object ``target``: :meth:`SetDown.onto`.

        The step after :meth:`scene` and ``Robot.pick``. The part's bottom is the support the scene its grasp came from
        declares, lowered only where two or more looks of a wrist camera measured the part standing below it
        (``scene.part_bottom_mm``; one view keeps the declared support as before), a lower bound on where the part
        stood, and :meth:`keep_out` holds the target out of the planner world while the part comes down onto it:

            scene = seen.scene(0, app.robot)
            best = scene.grasps().best
            ...                                  # robot.pick(best.pose(), ...), then locate the target
            set_down = onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.part_bottom_mm)
            if set_down.pose is not None:
                robot.place(set_down.pose, keep_out=onto.keep_out(0))

        The tool keeps the grasp's turn over the middle of the target's top, at that top (the 95th percentile of the
        target's surface heights) plus the part's hang plus ``air_mm``, the owner's 5 mm unless chosen. From that
        bottom the part lands with at least that air, and a part taken off a raised block drops the block's height on
        top of it; a caller that knows where the part stood passes that as ``part_bottom_mm`` instead. Not
        ``scene.support_height_mm``: it is raised to the part's lowest SEEN point and presses the part into the target
        by whatever the camera missed of its foot (see :meth:`SetDown.onto`). A ``Located`` that was :attr:`refused`
        raises ``LocatorRefused`` with the reason.
        """
        self._refuse_if_refused()
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        return SetDown.onto(self.objects[int(target)], grasp=grasp, part_bottom_mm=part_bottom_mm, air_mm=air_mm)

    def _refuse_if_refused(self) -> None:
        """Fail closed: nothing is planned on what a look around refused (:attr:`refused`)."""
        if self.refused:
            raise LocatorRefused(f"nothing is planned on what camera {self.camera!r} located: {self.refused}")

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if not self.objects and self.refused and not self.looks:
            lines = [f"camera {self.camera!r} located nothing: the arm reached none of the looks, so no frame was taken"]
        elif not self.objects:
            lines = [f"camera {self.camera!r} located nothing at {self.captured_at_s:.3f} s: an empty scene and a "
                     "detector that failed read the same here"]
        else:
            rows = [f"  {index}: {obj.render()}" for index, obj in enumerate(self.objects)]
            lines = [f"camera {self.camera!r} ({self.mounting}) located {len(self.objects)} object(s) at "
                     f"{self.captured_at_s:.3f} s:", *rows]
        if self.looks:
            fused = f"; object 0 fuses {', '.join(self.looks_fused)}" if self.looks_fused else ""
            lines.append(f"  looked from {len(self.looks)} look(s): {', '.join(self.looks)}{fused}")
        if self.jaw_faces_seen is not None:
            faces = ", ".join(f"jaw {number} {'seen' if seen else 'NOT seen'}"
                              for number, seen in enumerate(self.jaw_faces_seen, start=1))
            lines.append(f"  the contact faces of object 0's best grasp: {faces}")
        if self.closing_axis is not None:
            lines.append(f"  object 0's grasp judged closing along {self.closing_axis}, within "
                         f"{CLOSING_AXIS_TOLERANCE_DEG:g} deg either way round: its scene takes that axis")
        if self.hand_eye_gap_mm is not None:
            lines.append(f"  hand-eye check: the looks measure the part's shared surface {self.hand_eye_gap_mm:.1f} mm "
                         "apart (median)")
        if self.generated_view_deg is not None:
            lines.append(f"  generated one view, the looks' last resort: turned {self.generated_view_deg:+.0f} deg about "
                         "the part")
        if self.views_file:
            lines.append(_ascii(f"  the looks were kept in {self.views_file}"))
        if self.refused:
            lines.append(f"  REFUSED: {self.refused}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            "camera": self.camera,
            "captured_at_s": self.captured_at_s,
            "mounting": self.mounting,
            "tool_to_base_mm": None if self.tool_to_base_mm is None else [list(row) for row in self.tool_to_base_mm],
            "objects": [obj.to_dict() for obj in self.objects],
            "looks": list(self.looks),
            "looks_fused": list(self.looks_fused),
            "jaw_faces_seen": None if self.jaw_faces_seen is None else list(self.jaw_faces_seen),
            "refused": self.refused,
            "hand_eye_gap_mm": self.hand_eye_gap_mm,
            "generated_view_deg": self.generated_view_deg,
            "views_file": self.views_file,
            **({"closing_axis": {"name": self.closing_axis.name, "heading_deg": self.closing_axis.heading_deg}}
               if self.closing_axis is not None else {}),
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

        Such a camera finds a part from the looks a program declares (:meth:`look_around`, which moves the arm to each
        one), and every frame is placed by the tool pose at its shutter. A fixed camera locates as the arm stands. Read
        off the calibration the locator was built with, as :attr:`Located.mounting` is.
        """
        return self._tool_pose is not None

    def locate(self, prompt: str) -> Located:
        """Ground ``prompt`` in one frame and place every object in BASE. Raises ``PerceptionFrameMoved`` on the wrist.

        With a view, what was located is pinned on the camera's window, handed over with ``prompt`` and the colour
        image (BGR) of the frame it was located in, and a locate that raised says why there before it raises as it
        would have.
        """
        return self._placed(prompt).located

    def look_around(self, prompt: str, looks: "Look | None" = None, *, robot: Any, both_faces: bool = False,
                    record_views: bool = False, closing_axis: "Maybe[ClosingAxisLike]" = UNSET) -> Located:
        """Locate the part ``prompt`` names from each of ``looks`` in turn, every look fused with the ones before, until
        its grasp is safe.

        It is how a part to grip is found: every look is judged by the part's grasp. What a held part is set down on has
        no grasp to judge, so a look around for it would visit every look while the hand carries the part: example 13
        locates its target look by look instead, the first look that sees it answering, and holds nothing for the place.

        A camera on the wrist sees only what the arm points it at, so the arm goes to each look in the order given
        through ``robot``'s own verbs (``robot.move_joints``, ``robot.home`` for ``"home"``), today's rules for a taught
        look, and the camera locates there, each frame placed by the tool pose stamped at its shutter. ``looks`` is one
        look or several (``JointPositions``, ``"home"``); none means the pose the arm stands at is the one look, and a
        list that names none, or an entry that is not a look, raises before anything moves.

        Each look's objects are fused with what the looks before it located, by association at the wrist's tolerances
        (``association.WRIST_VIEW_*``: two views of one hand-eye disagree by more the more the wrist turned), each
        look's surface thinned of the points it sees at a grazing incidence first (``scene_geometry.grazing_pixels``).
        Object 0 of the first look that measured its surface is the part, and it stays the part: a later look finds it
        by association, and a look that does not see it is left out and said. A look whose object 0 has no surface under
        its mask (a dark or shiny face at a bad angle) makes no part, and is said: nothing would associate with it.
        After each look the part's grasp is computed again on all the looks saw of it (``Located.scene(0,
        robot.robot_config).grasps().best``, its support read from all of them too), and the looking stops at the first
        look whose grasp is valid and which the looks agree on: looks whose detector calls the part by different labels
        make its grasp uncertain, and the looking goes on (a name the locator made up for an unlabelled object is no
        label). A safe grasp point is the goal, never a full scan: with ``both_faces`` (off by default) the looking goes
        on until both jaw contact faces of the chosen grasp were seen (``unseen_side.jaw_faces_seen``), and a part no
        look showed both of is refused (:attr:`Located.refused`), so nothing is gripped on it.

        Once every declared look was visited and the grasp is still not safe enough, one view is generated, the last
        resort, as a pick's looks generate it (``src/robot/execution/generated_view.py``, the one implementation both
        call): the look the part was last seen from turned about the part, toward the jaw contact face of the chosen
        grasp that no look showed, or toward the side no look faced where there is no grasp; every turn screened by the
        arm before it moves (``nearest_configuration``), the smallest first, and driven on the straight joint line alone,
        never planned around (no cuRobo, no retract, no detour); it is located and judged as a look is, and
        ``both_faces`` still refuses a part the generated view did not show both faces of either. An arm that names no
        configuration (the simulator's, a desk arm) or drives no straight joint line alone, or whose world does not hold
        the frames of the looks, generates none, and says why. Then, where the part was last seen from a look other than
        the one the arm stands at, the arm goes back there on the straight joint line; where that line is not clear, or
        would turn a joint past the generated view's travel cap, the pick approaches from where the arm stands, planned
        and judged as every approach is. A look around handed no looks generates nothing and moves back nowhere. Once
        that is done the looking ends with the views fused so far.

        The motions, as a pick's looks: nothing is said to the hand. A look the planner or a guard refused before
        anything was sent is skipped and said, and the looking goes on (a refused look is used up as a reached one is,
        so the generated view may still follow it); a look around that reaches none of its looks
        ends with nothing located and the first refusal in :attr:`Located.refused`. A look motion that failed once it
        may have been commanded, one refused before its command for a reason every look shares (a link that is not
        open, a camera world or a route the arm refuses, an arm that did not come to rest), a controller that cannot
        move and a camera that could not vouch for the cell on the way end the looking there with nothing else
        commanded, the reason in :attr:`Located.refused`.

        The frames the looks were taken in are held in the arm's live planner world from the first look the arm stands
        at (``LivePlannerWorld.hold_pick_views``), each placed where it was taken, so the pick that follows plans every
        motion against all of what the looks saw rather than only the frame where the arm stands; the frame of the pose
        the look around started from is not held. ``Robot.pick`` lets them go when it ends, however it ends, as do the
        next look around, before it moves, and the disconnect (``Robot.connected()``
        or ``Cell.connected()``); a place does not need them. A look around that raises lets them go itself. A program
        that tries another candidate after a pick that failed looks around again first, so that the frames are held for
        that pick too.

        What comes back is a :class:`Located`: object 0 the part with its fused cloud, every other object as the looks
        saw it, and :attr:`Located.looks`, :attr:`Located.looks_fused`, :attr:`Located.jaw_faces_seen` and the hand-eye
        check of the looks (:attr:`Located.hand_eye_gap_mm`, said above ``HAND_EYE_DRIFT_WARN_MM``). ``robot`` must keep
        the robot section its looks are judged on (``Robot.from_tree``), and its hand must be a parallel jaw, the one
        ``Located.scene`` plans grasps for, or ``LocatorRefused`` is raised before anything moves.

        ``record_views`` (off by default) keeps the frames of the looks for training, one file for the look around
        (``src/robot/execution/record_views.py``, the layout ``PickRun(record_views=True)`` writes): each look's colour
        image, the depth it was placed by, its lens, the tool pose stamped at its shutter, where its camera stood, and
        the part's fused cloud. Where the file went is :attr:`Located.views_file`; one that cannot be written is said,
        and the look around answers all the same.

        A fixed camera locates once, where it stands, and moves nothing: :meth:`locate`, with ``both_faces`` judging
        that one frame's grasp the same way. It has no looks to keep.

        ``closing_axis`` (unset by default) judges only the grasps that close along the axis named, as
        ``Located.scene(0, ...).grasps(closing_axis=...)`` takes them (the owner's "choose, don't twist", 2026-09-30):
        the looking stops at a valid grasp along it, and ``both_faces`` asks for its faces. The ``Located`` keeps it
        (:attr:`Located.closing_axis`), and object 0's scene chooses along it: ``grasps()`` takes it when asked for
        none and refuses another, so the grasp a program grips is the grasp the looks judged. A look around that judged
        the part's grasp along any axis refuses a scene asked for one the same way. A value that names no axis raises
        before anything moves.
        """
        wanted = closing_axis_of(closing_axis) if chosen(closing_axis) else None
        if not self.on_the_wrist:
            judged_on = _judged_on(robot) if both_faces else None
            located = replace(self.locate(prompt), closing_axis=wanted)
            if judged_on is None or not located.objects:
                return located
            verdict = _verdict(located, judged_on, disagree=(), both_faces=True, closing_axis=wanted)
            refused = _faces_refusal(verdict)
            if refused:
                logger.warning("look_around: %s; nothing is to be gripped on the part", refused)
            return replace(located, jaw_faces_seen=verdict.faces_seen, refused=refused)
        from src.robot.execution.looks import look_label, looks_of  # noqa: PLC0415

        robot_config = _judged_on(robot)
        plan: list[tuple[str, Any]] = (
            [(look_label(pose), pose) for pose in looks_of(looks)] if looks is not None else [(_HERE, None)])
        arm = getattr(robot, "arm", None)
        # What an earlier look around held goes before the arm moves; this one holds from its first look on
        # (:meth:`_look_from_each`), so the frame of the pose it starts from is never held (owner, 2026-09-30).
        _let_go_of_earlier_looks(arm)
        looking = _Looking(camera=str(getattr(self._camera, "rig_id", "camera")), robot_config=robot_config,
                           both_faces=both_faces, record_views=record_views,
                           name=f"look_around-{prompt}", closing_axis=wanted)
        try:
            return self._look_from_each(prompt, looks, plan, looking, robot, arm)
        except BaseException:
            # A look around that raised answers nothing, and the frames it began holding must not outlive it in the
            # planner world: a program that catches the error and moves on would have every later motion judged
            # against a pick that never came, and its goal test read frames that belong to none. As the pick loop's
            # look around lets go of its own.
            from src.robot.execution.lifecycle import let_go_of_held_views  # noqa: PLC0415

            let_go_of_held_views(arm)
            raise

    def _look_from_each(self, prompt: str, looks: "Look | None", plan: "list[tuple[str, Any]]", looking: "_Looking",
                        robot: Any, arm: Any) -> Located:
        """The looking of :meth:`look_around` on a camera on the wrist, from each look of ``plan`` in turn, then the
        generated view and the move back; what it came to."""
        first_refusal = ""
        for label, pose in plan:
            if pose is not None:
                moved = robot.home() if isinstance(pose, str) else robot.move_joints(pose)
                if not moved.ok:
                    stop = _why_the_looking_stops(moved, arm)
                    if stop:
                        logger.error("look %s was not reached, and the looking ends there with nothing else "
                                     "commanded: %s", label, stop)
                        return looking.answer(refused=f"the arm did not reach look {label}, and nothing else was "
                                                      f"commanded after it: {stop}")
                    why = f"{moved.status.value}: {moved.message}"
                    logger.warning(
                        "look %s was refused before anything was sent (%s): it is skipped, and the looking goes on "
                        "%sto the next look or, once none is left and the grasp is still not safe enough, to the one "
                        "generated view", label, why, "with the views fused so far, " if looking.found_a_grasp else "")
                    first_refusal = first_refusal or (
                        f"the arm reached none of the looks; the first, {label}, was refused before anything was sent "
                        f"({why}):\n{moved.render()}")
                    continue
            if not looking.hold_started:
                # The arm stands at its first look: every frame of the pick is held from here on.
                looking.holding, looking.hold_started = _hold_the_looks(arm), True
            # Where the arm stands for this look, read off it (a declared look may have been twinned into the cable
            # window): what the move back returns to.
            joints = _joints_of(arm) if looks is not None else None
            judged = looking.take(label, pose, self._placed(prompt), joints=joints)
            if judged is not None and judged.good:
                break
        # The last resort once every declared look was reached or refused and the part's grasp is still not safe enough:
        # the one generated view, then the move back to the look the part was last seen from. Both run on the straight
        # joint line ONLY, through the pick loop's own two functions (`execution.generated_view`), never through
        # `robot.move_joints` or `robot.home`, which plan around a line that is not clear (a declared look's rule,
        # forbidden for these two). A place's target never comes here (see the docstring).
        if looks is not None and looking.parts and looking.verdict is not None and not looking.verdict.good:
            stop = self._look_from_a_generated_view(prompt, looking, arm)
            if stop:
                logger.error("look_around: %s", stop.split("\n", 1)[0])
                return looking.answer(refused=stop)
        if (looks is not None and looking.found_a_grasp and not looking.faces_refusal() and looking.visited
                and looking.latest_look != looking.visited[-1]):
            stop = _move_back(looking, arm)
            if stop:
                logger.error("look_around: %s", stop.split("\n", 1)[0])
                return looking.answer(refused=stop)
        if not looking.visited and first_refusal:
            logger.error("look_around: %s", first_refusal.split("\n", 1)[0])
            return looking.answer(refused=first_refusal)
        refused = looking.faces_refusal()
        if refused:
            logger.warning("look_around: %s; nothing is to be gripped on the part", refused)
        return looking.answer(refused=refused)

    def _look_from_a_generated_view(self, prompt: str, looking: "_Looking", arm: Any) -> str:
        """The one view a look around generates, located and judged as a look; why the looking ends there, or ``""``.

        Aimed by :func:`_aim` and turned from the frame of the look the part was last seen from (its camera, the tool
        pose stamped at its shutter, its lens and image), from where the arm stands, by the pick loop's own
        ``generated_view.go_to_generated_view``: every turn screened by the arm before it moves, the smallest first, the
        straight joint line alone, and only where the arm's world holds the frames of the looks. A view that is not
        generated (why is logged) leaves the looking where it was. A motion that failed once it may have been
        commanded, a stopped controller and a camera that could not vouch on the way end the looking, as a declared
        look's do.
        """
        from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
        from src.robot.execution.generated_view import go_to_generated_view  # noqa: PLC0415

        aim, why = _aim(looking)
        placed = looking.latest
        tool = None if placed is None else placed.located.tool_to_base_mm
        if aim is not None and (placed is None or tool is None or placed.camera_to_base is None
                                or placed.intrinsics is None or placed.depth is None):
            why = f"the frame of look {looking.latest_look} carries no tool pose, camera placement or lens to turn"
        here = _joints_of(arm) if aim is not None and not why else None
        if aim is not None and not why and here is None:
            why = "the arm could not say where it stands"
        if why:
            logger.info("no view is generated: %s", why)
            return ""
        assert aim is not None and placed is not None and here is not None  # narrowed above
        height, width = np.asarray(placed.depth).shape[:2]
        try:
            generated = go_to_generated_view(
                arm, aim=aim, camera_to_base_mm=placed.camera_to_base,
                tool_to_base_mm=np.asarray(tool, dtype=np.float64), intrinsics=placed.intrinsics,
                image_size=(int(width), int(height)), here=here, home=getattr(arm, "home_joint_positions", None),
                holding=looking.holding, controller_stopped=lambda: _controller_cannot_move(arm) or None,
            )
        except CameraWorldUnavailable as exc:
            return (f"the arm did not reach the view generated for the part, and nothing else was commanded after it: "
                    f"a camera could not vouch for the cell on the way ({exc})")
        if generated.stopped is not None:
            why = (generated.controller or _why_the_looking_stops(generated.stopped, arm)
                   or f"its motion failed ({generated.stopped.status.value})")
            return (f"the arm did not reach the generated view {generated.label}, and nothing else was commanded after "
                    f"it: {why}")
        if not generated.moved:
            return ""
        assert generated.joints is not None  # moved
        looking.generated_view_deg = generated.azimuth_deg
        looking.take(generated.label, generated.joints, self._placed(prompt), joints=generated.joints)
        return ""

    def _placed(self, prompt: str) -> "_Placed":
        """:meth:`locate`, with what a fusion of looks reads from the frame kept beside what it located."""
        try:
            placed = self._locate(prompt)
        except Exception as exc:
            if self._view is not None:
                _on_view(self._view, "note", f"locate failed: {type(exc).__name__}: {exc}",
                         str(getattr(self._camera, "rig_id", "camera")))
            raise
        if self._view is not None:
            _on_view(self._view, "show_located", placed.located, image=placed.image, prompt=prompt)
        return placed

    def _locate(self, prompt: str) -> "_Placed":
        """What one frame located, that frame's colour image as the backend segmented it (BGR) where the source kept
        it, and the depth, lens, placement and surfaces a fusion of looks reads."""
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
        else:
            tool_to_base = None
        # The composition the hand finder over a wrist camera shares, so the two place one frame alike.
        camera_to_base = camera_to_base_at_shutter(self._calibration, frame.tool_pose if wrist else None)
        depth = frame.surface_depth_map if frame.surface_depth_map is not None else frame.depth_map
        objects: list[LocatedObject] = []
        surfaces: list[np.ndarray] = []
        labels: list[str] = []
        for seg in frame.segmentations:
            mask = np.asarray(getattr(seg, "mask")).astype(bool)
            # The surface is the mask less the pixels behind a depth step; the mask itself stays as segmented,
            # because ``keep_out`` holds the whole of it out of the planner world.
            surface = mask & ~pixels_behind_depth_steps(mask, depth)
            surfaces.append(surface)
            points = to_base_mm(surface, depth, frame.intrinsics, camera_to_base)
            centre = None if points.shape[0] == 0 else tuple(float(v) for v in np.median(points, axis=0))
            score = getattr(seg, "score", None)
            box = getattr(seg, "bbox_xyxy", None)
            labels.append(str(getattr(seg, "label", "") or ""))
            objects.append(LocatedObject(
                label=labels[-1] or f"object_{len(objects)}",
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
        return _Placed(
            located=located, image=None if frame.rgb is None else np.asarray(frame.rgb)[..., ::-1],
            depth=np.asarray(depth, dtype=np.float64), intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
            camera_to_base=camera_to_base, surfaces=tuple(surfaces), labels=tuple(labels),
        )


# ---------------------------------------------------------------------------------------------------------------------
# The looks of a camera on the wrist (Locator.look_around)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, eq=False)
class _Placed:
    """One frame as the locator placed it: what it located, and what a fusion of looks reads from the frame.

    ``surfaces`` are the objects' masks less the pixels behind a depth step, as their points were placed, on ``depth``
    (the measured surface) through ``intrinsics`` and ``camera_to_base``. ``labels`` are what the detector called each
    object, ``""`` where it gave no label: the located object then carries a name the locator made up from its place in
    the frame (``object_1``), which says nothing of what it is. A record that carries only ``located`` (a double's)
    fuses the located points as they are, under the located labels.
    """

    located: Located
    image: "np.ndarray | None" = None
    depth: "np.ndarray | None" = None
    intrinsics: "np.ndarray | None" = None
    camera_to_base: "np.ndarray | None" = None
    surfaces: "tuple[np.ndarray, ...]" = ()
    labels: "tuple[str, ...]" = ()

    def called(self) -> "tuple[str, ...]":
        """What the detector called each object, as the looks compare labels: trimmed, lower case, ``""`` where it gave
        none (the pick loop's reading of a segmentation's label)."""
        objects = self.located.objects
        given = self.labels if len(self.labels) == len(objects) else tuple(obj.label for obj in objects)
        return tuple(str(label or "").strip().lower() for label in given)

    @property
    def seen_from_mm(self) -> "np.ndarray | None":
        """Where the camera stood, BASE mm, or ``None`` where the record does not say."""
        if self.camera_to_base is None:
            return None
        return np.asarray(self.camera_to_base, dtype=np.float64)[:3, 3].copy()

    def fusion_clouds(self) -> "tuple[np.ndarray, ...]":
        """Each object's surface as a fusion of looks takes it, BASE mm: less the pixels it sees at a grazing incidence.

        The owner's rule (2026-09-29, addendum 7.4) through the pick loop's own (``scene_geometry.grazing_pixels``,
        ``GRAZING_INCIDENCE_DEG``): a point seen that obliquely says more about the edge it was seen past than about
        where the surface is, and a look that faces the surface squarer measures it better.
        """
        objects = self.located.objects
        if (self.depth is None or self.intrinsics is None or self.camera_to_base is None
                or len(self.surfaces) != len(objects)):
            return tuple(np.asarray(obj.points_base_mm, dtype=np.float64).reshape(-1, 3) for obj in objects)
        from src.robot.grasping.multiview.scene_geometry import grazing_pixels  # noqa: PLC0415

        return tuple(
            to_base_mm(surface & ~grazing_pixels(surface, self.depth, self.intrinsics), self.depth, self.intrinsics,
                       self.camera_to_base)
            for surface in self.surfaces
        )


@dataclass(frozen=True, slots=True, eq=False)
class _Sight:
    """One look's surface of one part: which look, the surface as the fusion takes it, what the look called the part,
    and where its camera stood (BASE mm, ``None`` where unknown)."""

    look: str
    cloud: np.ndarray
    label: str
    seen_from_mm: "np.ndarray | None"


@dataclass(frozen=True, slots=True, eq=False)
class _Part:
    """One part as the looks so far saw it: the object of the latest look that saw it, every look's surface of it, and
    its cloud: as that one look located it where one look saw it, the looks' surfaces fused where several did."""

    shown: LocatedObject
    sights: "tuple[_Sight, ...]"
    points: np.ndarray

    def fusion_cloud(self) -> np.ndarray:
        """Every look's surface of this part together, as the next look is associated against it."""
        clouds = [sight.cloud for sight in self.sights if sight.cloud.shape[0]]
        return np.vstack(clouds) if clouds else np.zeros((0, 3), dtype=np.float64)

    def as_object(self) -> LocatedObject:
        if self.points is self.shown.points_base_mm:
            return self.shown
        centre = None if self.points.shape[0] == 0 else tuple(float(v) for v in np.median(self.points, axis=0))
        return replace(self.shown, points_base_mm=self.points, centre_mm=centre)  # type: ignore[arg-type]


@dataclass(frozen=True, slots=True, eq=False)
class _Verdict:
    """How the part's grasp stood after one look: the best candidate (``None`` for none), whether each jaw's contact
    face of it was seen (``None`` where not judged), the earlier looks that called the part otherwise, and whether that
    is safe enough to stop looking at."""

    best: Any
    faces: Any
    disagree: "tuple[tuple[str, str], ...]"
    good: bool
    #: What the closing axis the look around was handed left out of the part's grasps, said; ``""`` for none.
    withheld: str = ""

    @property
    def faces_seen(self) -> "tuple[bool, bool] | None":
        return None if self.faces is None else tuple(self.faces.as_pair())  # type: ignore[return-value]


def _jaw_number(face: Any) -> int:
    """Jaw 1 closes on the face against the closing axis, jaw 2 on the face along it (``JawFacesSeen.as_pair``)."""
    return 1 if int(face.side) < 0 else 2


def _verdict(located: Located, robot_config: Any, *, disagree: "tuple[tuple[str, str], ...]",
             both_faces: bool, closing_axis: "ClosingAxis | None" = None) -> _Verdict:
    """Object 0's best grasp on ``located`` as the cell plans it, whether its jaw contact faces were seen, and whether
    that is safe enough: a valid grasp the looks agree on, and with ``both_faces`` both contact faces seen. With
    ``closing_axis`` the best of the grasps that close along it (``Scene.grasps(closing_axis=...)``)."""
    from src.robot.grasping.multiview.unseen_side import jaw_faces_seen  # noqa: PLC0415

    scene = located.scene(0, robot_config)
    grasps = scene.grasps(closing_axis=closing_axis if closing_axis is not None else UNSET)
    best = grasps.best
    faces = None
    if best is not None and scene.jaw is not None:
        try:
            faces = jaw_faces_seen(
                scene.target_points_base_mm, centre_mm=best.position_mm, approach=best.approach,
                closing_axis=best.closing_axis, grip_width_mm=float(best.grip_width_mm),
                pad_ahead_mm=float(scene.jaw.pad_ahead_mm), pad_behind_mm=float(scene.jaw.pad_behind_mm),
                finger_width_mm=float(scene.jaw.finger_width_mm), support_height_mm=float(scene.support_height_mm),
            )
        except ValueError as exc:
            logger.info("the jaw contact faces of the chosen grasp could not be judged: %s", exc)
    good = best is not None and not disagree and (not both_faces or (faces is not None and faces.both))
    return _Verdict(best=best, faces=faces, disagree=disagree, good=good, withheld=grasps.withheld)


def _faces_refusal(verdict: "_Verdict | None") -> str:
    """Why ``both_faces`` refuses the part, or ``""``: the contact face(s) of its grasp no look showed."""
    if verdict is None or verdict.best is None:
        return ""
    if verdict.faces is None:
        return ("the jaw contact faces of the chosen grasp could not be judged, and both_faces asks that both were "
                "seen before gripping")
    if verdict.faces.both:
        return ""
    missing = [f"jaw {_jaw_number(face)}" for face in verdict.faces.missing]
    faces = "face" if len(missing) == 1 else "faces"
    was = "was" if len(missing) == 1 else "were"
    return (f"the contact {faces} at {' and '.join(missing)} of the chosen grasp {was} not seen from any look, and "
            "both_faces asks that both were seen before gripping")


def _judged_on(robot: Any) -> Any:
    """The robot section a look's grasp is judged on (the hand, the support, the floor, as the cell plans them), refused
    before anything moves where no grasp could be judged on it: no section, or a hand that is no parallel jaw, since
    ``Located.scene`` plans jaw grasps only (``SupportFootprintJaw.from_robot_config``, the same refusal)."""
    config = getattr(robot, "robot_config", None)
    if config is None:
        raise LocatorRefused(
            "a look around judges each look's grasp on the hand and the support the cell describes, and this robot "
            "keeps no robot section: build it with Robot.from_tree(tree, ...), or pass robot_config= to "
            "Robot.from_parts")
    from src.robot.grasping.generation.support_footprint import SupportFootprintJaw  # noqa: PLC0415

    try:
        SupportFootprintJaw.from_robot_config(config)
    except ValueError as exc:
        raise LocatorRefused(
            f"a look around judges each look's grasp as Located.scene plans it, for a parallel jaw, and this cell's "
            f"hand is none ({exc}); nothing was moved: locate() finds the part without judging a grasp") from exc
    return config


def _hold_the_looks(arm: Any) -> bool:
    """Start holding the frames of the looks in ``arm``'s live planner world, as a pick's looks are held
    (``LivePlannerWorld.hold_pick_views``; ``keep_out.holding_views`` is the same hold as a ``with`` block). Starting
    again drops what an earlier look around held. ``Robot.pick`` lets them go when it ends. Whether the world has a
    wrist frame to hold; ``False`` for an arm with no live world."""
    world = getattr(arm, "live_planner_world", None)
    hold = getattr(world, "hold_pick_views", None)
    return bool(hold()) if callable(hold) else False


def _let_go_of_earlier_looks(arm: Any) -> None:
    """Let go of the frames ``arm``'s live planner world holds for an earlier look around, where it says it holds any
    (``LivePlannerWorld.holds_pick_views``): the motion to this look around's first look is judged without them, and
    the frame it is planned on, where the arm stands now, is not held."""
    world = getattr(arm, "live_planner_world", None)
    if bool(getattr(world, "holds_pick_views", False)):
        from src.robot.execution.lifecycle import let_go_of_held_views  # noqa: PLC0415

        let_go_of_held_views(arm)


def _sent_nothing(moved: Any) -> bool:
    """Whether a look's failed motion was refused before anything was sent: the one rule the pick loop's looks, its
    generated view and its move back skip on too (``generated_view.refused_before_sending``), so the two paths skip the
    same refusals and nothing else."""
    from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415

    return bool(refused_before_sending(moved))


def _controller_cannot_move(arm: Any) -> str:
    """Why ``arm``'s controller cannot move, or ``""``, in the words of a look it did not reach.

    The robot verbs' criterion (``handling._controller_refusal``): only an arm that says (``SupportsRobotStatus``) is
    asked, ``not is_operational`` cannot move, and a controller whose state cannot be read cannot be said to move. Not
    that verb's sentence, which says nothing was commanded: here the motion to the look may have been.
    """
    from src.robot.core.arm_capabilities import SupportsRobotStatus  # noqa: PLC0415

    if not isinstance(arm, SupportsRobotStatus):
        return ""
    try:
        status = arm.get_robot_status()
    except Exception as exc:  # noqa: BLE001 (a controller that cannot be asked cannot be said to move)
        return f"the controller's state could not be read ({type(exc).__name__}: {exc})"
    if status.is_operational:
        return ""
    detail = f": {status.message}" if status.message else ""
    return (f"the controller cannot move (robot_mode={status.robot_mode.value}, "
            f"safety_mode={status.safety_mode.value}, protective_stop={status.protective_stopped}, "
            f"emergency_stop={status.emergency_stopped}{detail}); clear the stop where the arm is visible, then run again")


def _hand_eye_limits() -> "tuple[float, int]":
    """The hand-eye check of the looks, held to the pick loop's limits (``association.HAND_EYE_DRIFT_WARN_MM`` and
    ``HAND_EYE_MIN_POINTS``, beside the distances they read): the gap above which the calibration may have drifted, mm,
    and the fewest shared-surface distances it speaks on."""
    from src.robot.grasping.multiview.association import (  # noqa: PLC0415
        HAND_EYE_DRIFT_WARN_MM,
        HAND_EYE_MIN_POINTS,
    )

    return float(HAND_EYE_DRIFT_WARN_MM), int(HAND_EYE_MIN_POINTS)


def _joints_of(arm: Any) -> Any:
    """The configuration ``arm`` stands at, read off it (``JointPositions``); ``None``, and said, where it cannot say."""
    from src.robot.core import JointPositions  # noqa: PLC0415
    from src.robot.core.errors import RobotError  # noqa: PLC0415

    read = getattr(arm, "get_joint_positions", None)
    if not callable(read):
        return None
    try:
        joints = read()
    except (RobotError, RuntimeError, OSError) as exc:
        logger.warning("the arm could not say where it stands (%s: %s)", type(exc).__name__, exc)
        return None
    return joints if isinstance(joints, JointPositions) else None


def _aim(looking: "_Looking") -> "tuple[Any, str]":
    """What the generated view is to show (``generated_view.ViewAim``), or ``None`` and why there is nothing to show.

    The owner's rule (2026-09-29), as the pick loop aims: with a grasp, the jaw contact faces of the chosen grasp no
    look showed, turning about its centre; a grasp both of whose faces were seen (its looks disagreed on the label), or
    whose faces could not be judged, leaves nothing to show. With no grasp at all, the side of the part no look faced,
    about the middle of its cloud.
    """
    from src.robot.execution.generated_view import aim_at_faces, aim_at_unseen_side  # noqa: PLC0415

    verdict = looking.verdict
    assert verdict is not None and looking.parts  # the caller's condition
    if verdict.best is not None:
        if verdict.faces is None:
            return None, "the jaw contact faces of the chosen grasp could not be judged, so there is no face to show"
        if not verdict.faces.missing:
            return None, "both jaw contact faces of the chosen grasp were seen, so there is no face to show"
        return aim_at_faces(verdict.faces.missing, grasp_centre_mm=verdict.best.position_mm), ""
    try:
        return aim_at_unseen_side(looking.parts[0].points, camera_centres_mm=looking.seen_from), ""
    except ValueError as exc:
        return None, f"no side of the part can be told unseen ({exc})"


def _move_back(looking: "_Looking", arm: Any) -> str:
    """The move back to the look the part was last seen from, on the straight joint line only; why the looking ends, or
    ``""``.

    The pick loop's own ``generated_view.move_back_to_look``: the configuration the arm stood at for that look, read
    off it there, driven to on the straight joint line and nothing planned around it, within the generated view's
    travel cap from where the arm stands, and only where the arm's world holds the frames of the looks. Refused, the
    pick approaches from where the arm stands, said as a WARNING. A motion that failed once it may have been commanded,
    a stopped controller and a camera that could not vouch on the way end the looking, as a declared look's do.
    """
    from src.robot.core.errors import CameraWorldUnavailable  # noqa: PLC0415
    from src.robot.execution.generated_view import move_back_to_look  # noqa: PLC0415

    look = looking.latest_look
    try:
        back = move_back_to_look(arm, looking.latest_joints, look=look, here=_joints_of(arm), holding=looking.holding,
                                 controller_stopped=lambda: _controller_cannot_move(arm) or None)
    except CameraWorldUnavailable as exc:
        return (f"the arm did not reach look {look} again, and nothing else was commanded after it: a camera could not "
                f"vouch for the cell on the way ({exc})")
    if back.stopped is not None:
        why = (back.controller or _why_the_looking_stops(back.stopped, arm)
               or f"its motion failed ({back.stopped.status.value})")
        return f"the arm did not reach look {look} again, and nothing else was commanded after it: {why}"
    if not back.done:
        logger.warning(
            "the move back to look %s, which saw the part, was refused (%s): the pick approaches from where the arm "
            "stands, at look %s, planned and judged as every approach is", look, back.refused, looking.visited[-1])
    return ""


def _why_the_looking_stops(moved: Any, arm: Any) -> str:
    """Why a look motion that failed ends the looking, or ``""`` for a refusal the looking skips.

    A camera that could not vouch for the cell on the way, and a controller that cannot move, end it as they end every
    motion of a pick. So does a motion that failed once it may have been commanded: the arm may have moved part of the
    way, or may still be moving. And so does a motion the robot verb refused before its command for a reason every look
    shares (``MotionOutcome.REFUSED``: a link that is not open, a camera world or a route the arm refuses, an arm that
    did not come to rest), as the pick loop's looks end on it; the arm stands where it stood. Only a guard's or the
    planner's refusal of this one look before anything was sent is skipped.
    """
    from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

    if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE:
        return f"a camera could not vouch for the cell on the way:\n{moved.render()}"
    controller = _controller_cannot_move(arm)
    if controller:
        return controller
    if moved.outcome is MotionOutcome.REFUSED:
        return (f"its motion was refused before any command ({moved.status.value}), so the arm stands where it stood, "
                f"for a reason every look shares:\n{moved.render()}")
    if not _sent_nothing(moved):
        return (f"its motion failed ({moved.status.value}) once it may have been commanded, so the arm may have moved "
                f"part of the way:\n{moved.render()}")
    return ""


class _Looking:
    """The looks of one :meth:`Locator.look_around`: every part they located, fused, object 0 the part they keep, and
    how its grasp stood after each look."""

    def __init__(self, *, camera: str, robot_config: Any, both_faces: bool, holding: bool = False,
                 record_views: bool = False, name: str = "", closing_axis: "ClosingAxis | None" = None) -> None:
        self.camera = camera
        self.robot_config = robot_config
        self.both_faces = both_faces
        #: The closing axis the part's grasp is judged along (``look_around(closing_axis=...)``), or ``None`` for any.
        self.closing_axis = closing_axis
        #: Whether the frames of the looks are kept for training when the looking answers, and the name their file
        #: goes by (``record_views``).
        self.record_views = record_views
        self.name = name
        #: Whether the arm's live planner world holds the frames of the looks (``LivePlannerWorld.hold_pick_views``):
        #: a generated view and a move back refuse to run without it.
        self.holding = holding
        #: Whether the hold was asked for yet: once, when the arm first stands at a look.
        self.hold_started = False
        #: The looks located from, in order.
        self.visited: list[str] = []
        #: Every look's frame as the locator placed it, in order: its colour, its depth, its lens, where its camera
        #: stood and what it located, kept whole for whoever records the looks.
        self.frames: list[tuple[str, _Placed]] = []
        #: Where the camera stood at each of them, BASE mm.
        self.seen_from: list[np.ndarray] = []
        #: The parts, object 0 first: empty until a look measured the surface of its object 0.
        self.parts: list[_Part] = []
        #: The frame of the last look the part was seen from, its label and its pose; before any was, the frame of the
        #: latest look that located the part with no surface under its mask, else of the latest that located nothing
        #: (``empty``).
        self.latest: "_Placed | None" = None
        self.latest_look = ""
        self.latest_pose: Any = None
        #: The configuration the arm stood at for that look, read off it: what the move back returns to; ``None`` where
        #: it was not read (a look around handed no looks) or the arm could not say.
        self.latest_joints: Any = None
        self.empty: "_Placed | None" = None
        #: How the part's grasp stood after the last look it was seen from.
        self.verdict: "_Verdict | None" = None
        #: How far the one generated view turned about the part, degrees; ``None`` while none was generated.
        self.generated_view_deg: float | None = None

    @property
    def found_a_grasp(self) -> bool:
        return self.verdict is not None and self.verdict.best is not None

    def take(self, look: str, pose: Any, placed: _Placed, *, joints: Any = None) -> "_Verdict | None":
        """Add one look: fuse what it located with the looks before, and judge the part's grasp on the result.

        ``joints`` is the configuration the arm stood at for it, read off the arm. ``None`` where the look adds nothing
        to the part: it located nothing, not the part the looks before found, or, before any look measured the part, a
        part with no surface under its mask.
        """
        self.visited.append(look)
        self.frames.append((look, placed))
        seen_from = placed.seen_from_mm
        if seen_from is not None:
            self.seen_from.append(seen_from)
        objects = placed.located.objects
        if not objects:
            if self.parts:
                logger.warning("look %s located nothing: it is left out, and the part stands as the looks before it "
                               "saw it", look)
            else:
                if self.empty is None or not self.empty.located.objects:
                    self.empty = placed
                logger.info("look %s located nothing", look)
            return None
        clouds = placed.fusion_clouds()
        called = placed.called()
        if not self.parts:
            measured = int(clouds[0].shape[0])
            if measured < _MIN_PART_POINTS:
                self.empty = placed
                logger.warning(
                    "look %s located the part (%r) with no surface under its mask to fuse (%d point(s), fewer than the "
                    "%d a part is kept on: a dark or shiny face at a bad angle?): it makes no part, and the looking "
                    "goes on", look, objects[0].label, measured, _MIN_PART_POINTS)
                return None
            parts = [_Part(obj, (_Sight(look, cloud, label, seen_from),), obj.points_base_mm)
                     for obj, cloud, label in zip(objects, clouds, called, strict=True)]
        else:
            fused = self._fused_with(look, objects, clouds, called, seen_from)
            if fused is None:
                return None
            parts = fused
        self.parts = parts
        self.latest, self.latest_look, self.latest_pose, self.latest_joints = placed, look, pose, joints
        self.verdict = self._judge(look)
        return self.verdict

    def _fused_with(self, look: str, objects: "tuple[LocatedObject, ...]", clouds: "tuple[np.ndarray, ...]",
                    called: "tuple[str, ...]", seen_from: "np.ndarray | None") -> "list[_Part] | None":
        """This look's objects fused with the parts the looks before it located, the part first; ``None`` where this
        look did not see the part.

        One association of this look against every part so far (``fuse_scene_clouds`` at the wrist tolerances), one to
        one, so no two of this look's objects join one part. A part this look did not see stays, with no pixels in its
        frame: an obstacle an earlier look saw is still there.
        """
        from src.robot.grasping.multiview.association import (  # noqa: PLC0415
            WRIST_VIEW_MIN_SCORE,
            WRIST_VIEW_NEIGHBOUR_MM,
            WRIST_VIEW_SCORE_VOXEL_MM,
            AssociationMetric,
            ViewCandidates,
            fuse_scene_clouds,
        )

        fused, associations = fuse_scene_clouds(
            list(clouds), [ViewCandidates("the looks before", tuple(part.fusion_cloud() for part in self.parts))],
            metric=AssociationMetric.OVERLAP, min_score=WRIST_VIEW_MIN_SCORE, neighbour_mm=WRIST_VIEW_NEIGHBOUR_MM,
            score_voxel_mm=WRIST_VIEW_SCORE_VOXEL_MM,
        )
        assignment = associations[0].assignment
        target = next((index for index, part in enumerate(assignment) if part == 0), None)
        if target is None:
            logger.warning("look %s did not see the part the looks before it located (%r): it is left out, and the "
                           "part stands as they saw it", look, self.parts[0].shown.label)
            return None
        parts: list[_Part] = []
        for index in (target, *(other for other in range(len(objects)) if other != target)):
            sight = _Sight(look, clouds[index], called[index], seen_from)
            before = assignment[index]
            if before is None:
                parts.append(_Part(objects[index], (sight,), objects[index].points_base_mm))
            else:
                parts.append(_Part(objects[index], (*self.parts[before].sights, sight), fused[index]))
        matched = {before for before in assignment if before is not None}
        blank = np.zeros(np.shape(objects[0].mask), dtype=bool)
        parts.extend(_Part(replace(part.shown, mask=blank, box_px=None), part.sights, part.points)
                     for number, part in enumerate(self.parts) if number not in matched)
        return parts

    def _judge(self, look: str) -> _Verdict:
        """The part's grasp on everything the looks saw of it, and whether it is safe enough to stop looking at.

        Looks whose detector calls the part by different labels make the grasp uncertain (the owner, 2026-09-29,
        addendum 7.3): one of them saw something else there, and the looking goes on. A look whose detector gave no
        label says nothing either way, as in the pick loop. The account of the part says how many looks agreed (``label
        agreed in N of M looks``, the owner's decision 4, 2026-09-30), and that changes nothing.
        """
        from src.robot.grasping.multiview.association import label_agreement, label_agreement_said  # noqa: PLC0415

        sights = self.parts[0].sights
        called = sights[-1].label
        others = [(sight.look, sight.label) for sight in sights[:-1]]
        _, disagree = label_agreement(called, others)
        if disagree:
            logger.warning("look %s calls the part %r, and %s: the grasp is uncertain", look, called, ", ".join(
                f"look {earlier} calls it {label!r}" for earlier, label in disagree))
        verdict = _verdict(self._located(), self.robot_config, disagree=disagree, both_faces=self.both_faces,
                           closing_axis=self.closing_axis)
        if verdict.best is None:
            said = f"no grasp on the part ({verdict.withheld})" if verdict.withheld else "no grasp on the part"
        elif verdict.good:
            said = "a grasp safe enough to stop at"
        elif disagree:
            said = "a grasp, uncertain: the looks disagree on what the part is"
        else:
            missing = verdict.faces.missing if verdict.faces is not None else ()
            said = ("a grasp, " + " and ".join(f"the contact face at jaw {_jaw_number(face)} of the chosen grasp not "
                                               "seen" for face in missing)
                    if missing else "a grasp whose contact faces could not be judged")
        logger.info("look %s: %s, on the looks %s; %s", look, said, ", ".join(sight.look for sight in sights),
                    label_agreement_said(called, others))
        return verdict

    def _located(self, *, refused: str = "", hand_eye_gap_mm: "float | None" = None, views_file: str = "") -> Located:
        """What the looks located so far, as one :class:`Located`: the frame of the last look the part was seen from.

        Where no look was reached no frame was taken, and its stamp is when the looking ended.
        """
        placed = self.latest if self.latest is not None else self.empty
        target = self.parts[0] if self.parts else None
        fused = () if target is None else (target.sights[-1].look, *(sight.look for sight in target.sights[:-1]))
        if placed is None:
            base = Located(camera=self.camera, captured_at_s=time.time(), mounting="eye_in_hand",
                           tool_to_base_mm=None, objects=())
        else:
            base = placed.located
        objects = tuple(part.as_object() for part in self.parts) if self.latest is not None else base.objects
        return replace(base, objects=objects, looks=tuple(self.visited), looks_fused=fused,
                       jaw_faces_seen=None if self.verdict is None else self.verdict.faces_seen, refused=refused,
                       hand_eye_gap_mm=hand_eye_gap_mm, generated_view_deg=self.generated_view_deg,
                       views_file=views_file, closing_axis=self.closing_axis)

    def faces_refusal(self) -> str:
        """With ``both_faces``, why the part is refused: the contact face(s) of its grasp no look showed."""
        return _faces_refusal(self.verdict) if self.both_faces else ""

    def answer(self, *, refused: str = "") -> Located:
        """What the looks came to: the parts fused, the looks, the hand-eye check said, ``refused``, and, where they
        were to be kept, the file the frames of the looks went to."""
        gap = self._hand_eye_gap_mm()
        if gap is not None:
            warn_mm, _ = _hand_eye_limits()
            if gap > warn_mm:
                logger.warning(
                    "the looks measure the part's shared surface %.1f mm apart (median), more than %.1f mm: the "
                    "hand-eye calibration may have drifted (a camera loosened on the wrist?). Nothing is changed for "
                    "it; check the calibration", gap, warn_mm)
            else:
                logger.info("hand-eye check: the looks measure the part's shared surface %.1f mm apart (median)", gap)
        return self._located(refused=refused, hand_eye_gap_mm=gap, views_file=self._kept())

    def _kept(self) -> str:
        """Keep the frames of the looks for training (``record_views``), where asked and a look was taken; the file, or
        ``""``. Through the one writer ``PickRun(record_views=True)`` uses (``execution.record_views``), each look as a
        pick's look is kept: its colour (RGB), the depth it was placed by, its lens, the tool pose stamped at its
        shutter, where its camera stood, and the part's fused cloud. A file that cannot be written is said, and the
        looking answers all the same: the frames are for training."""
        if not self.record_views or not self.frames:
            return ""
        from types import SimpleNamespace  # noqa: PLC0415

        from src.robot.execution.record_views import record_views  # noqa: PLC0415

        views = [
            SimpleNamespace(
                label=look, name=f"{self.camera}@{look}", camera_to_base=placed.camera_to_base,
                frame=SimpleNamespace(
                    rgb=None if placed.image is None else np.ascontiguousarray(placed.image[..., ::-1]),
                    depth_map=placed.depth, surface_depth_map=None, intrinsics=placed.intrinsics,
                    tool_pose=(None if placed.located.tool_to_base_mm is None
                               else np.asarray(placed.located.tool_to_base_mm, dtype=np.float64)),
                ),
            )
            for look, placed in self.frames
        ]
        try:
            written = record_views(views, target_cloud_base_mm=self.parts[0].points if self.parts else None,
                                   name=self.name)
        except Exception as exc:  # noqa: BLE001 (a lost training file never costs the answer)
            logger.warning("look_around: the frames of the looks were not kept: %s: %s", type(exc).__name__, exc)
            return ""
        if written is None:
            return ""
        logger.info("look_around: the frames of %d look(s) were kept in %s", len(views), written)
        return str(written)

    def _hand_eye_gap_mm(self) -> "float | None":
        """The median distance between the last look's surface of the part and every earlier look's, over the surface
        they both saw (``association.nearest_surface_distances_mm``, the pick loop's check of its looks); ``None``
        where fewer distances than that check speaks on were measured."""
        if not self.parts:
            return None
        sights = self.parts[0].sights
        last = sights[-1]
        if len(sights) < 2 or last.seen_from_mm is None or not last.cloud.shape[0]:
            return None
        from src.robot.grasping.multiview.association import nearest_surface_distances_mm  # noqa: PLC0415

        found = [
            nearest_surface_distances_mm(last.cloud, sight.cloud, seen_from_mm=last.seen_from_mm,
                                         other_seen_from_mm=sight.seen_from_mm)
            for sight in sights[:-1] if sight.seen_from_mm is not None and sight.cloud.shape[0]
        ]
        pooled = np.concatenate(found) if found else np.zeros(0)
        _, fewest = _hand_eye_limits()
        return float(np.median(pooled)) if pooled.size >= fewest else None
