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

``Locator.measure(points)`` is the quick look at something located before (the owner, 2026-10-08: speed first): one
frame, taken as a locate takes it but with no detector, and per point of BASE the depth it should read there, the depth
the frame measured at its pixel and that pixel's colour (:class:`Measured`). It says what the frame reads where the
points should stand, never what stands there: a task reads a kept bin's rim with it before it asks the detector again.
``Located.measure(points, image)`` reads the same off the frame a locate placed.

``Locator.locate("yellow bin | blue bin")`` grounds a class list (``src.models.vlm.qwen.class_list_prompt``) in one call:
each object carries the description its box named, an object no one description claims is ``ambiguous``, and
``Located.split(prompt)`` says which objects each description located, so one locate finds every bin a task still
looks for (the sorting package, 2026-10-09).

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
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.constants import create_robot_logger
from src.robot.core.keep_out import KeepOutBox, SegmentationOffer
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
from src.robot.perception.colour_check import ColourCheck
from src.robot.perception.realsense_source import RealSenseVisionPerceptionSource

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.config.schema.robot import RobotConfig
    from src.robot.execution.looks import Look

__all__ = ["SET_DOWN_AIR_MM", "Located", "LocatedObject", "LocatedOrientation", "Locator", "LocatorRefused", "Measured",
           "SetDown"]

#: The locator's own lines, to the robot log and ``locator.log`` (RC6 of the cell-fix plan: they reached no file).
logger: logging.Logger = create_robot_logger(__name__, "locator.log")

#: How far past a target's measured surface its keep-out box reaches when a scene asks what it stands on, millimetres:
#: the camera world's margin, so the part's own top is never taken for its support.
_TARGET_BOX_MARGIN_MM = 15.0

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

    Nothing in it is written in a program: the target's top comes from the camera, the part's hang from the grasp and
    the part's bottom, the air from the owner (5 mm, released only then, the part never pressed into the target). Build
    it with :meth:`onto`, or :meth:`Located.set_down`.

    Attributes:
        target (str): The located object the part goes on, by its label.
        pose (Pose | None): The TCP pose ``Robot.place`` takes, in BASE: the grasp's own turn, over the middle of the
            target's top, at ``top_mm + hang_mm + air_mm``; ``None`` where it cannot be measured, ``reason`` saying why.
            Nothing is set down blind.
        top_mm (float | None): The target's top in BASE Z, millimetres: the 95th percentile of its surface heights;
            ``None`` where it could not be read.
        hang_mm (float): How far the part hangs below the grasp, millimetres: the grasp's Z less the part's bottom.
        air_mm (float): The gap between the part's bottom and the target's top when the hand opens, millimetres.
        points (int): How many surface points the top was read from.
        reason (str): Why there is no pose; empty when there is one (default: "").

    What it cannot know: that the part stayed where the fingers closed (a hand with no sensor measures nothing, and a
    part that slipped hangs lower), and that the camera reads the target where it stands (a calibration reading the
    scene low by some millimetres leaves that much less air, which is what the air is for). A target whose rim stands
    above its middle is topped at its rim.
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

        Args:
            target (LocatedObject): What the part goes on: ``located.objects[i]``.
            grasp (Pose): Where the tool closed on the part, in ``Frame.BASE``.
            part_bottom_mm (float): Where the part's bottom was when it was grasped, BASE Z in millimetres, and a LOWER
                bound on it: pass the ``part_bottom_mm`` of the scene the grasp came from. A bottom read too high brings
                the part down into the target by as much, one too low only adds air. A caller that knows where the part
                stood (a fixture of known height) passes that, and gets the air exactly.
            air_mm (float): The gap between the part's bottom and the target's top when the hand opens, millimetres
                (default: 5.0, the owner's rule).

        Returns:
            SetDown: The set-down; ``pose`` is ``None``, with a ``reason``, where the target's top cannot be read or the
                grasp did not stand above the part's bottom.

        Raises:
            ValueError: A grasp not in BASE, a bottom or an air that is not a finite number, a negative air.

        Not the scene's ``support_height_mm``: that is raised to the part's lowest SEEN point, an upper bound on its
        base, and a near face the camera missed (8 mm in the review of 2026-09-24) set the part down 3 mm INTO the
        target.
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
        """The set-down as a person reads it.

        Returns:
            str: ASCII, no trailing newline.
        """
        if self.pose is None:
            return _ascii(f"set down on {self.target!r}: NO POSE, {self.reason}")
        x, y, z = (float(v) for v in self.pose.position_mm)
        return _ascii(
            f"set down on {self.target!r}: its top at {self.top_mm:.1f} mm (the {_TOP_PERCENTILE:.0f}th percentile of "
            f"{self.points} point(s)), the part hanging {self.hang_mm:.1f} mm below the grasp, {self.air_mm:.1f} mm of "
            f"air: the tool to ({x:.1f}, {y:.1f}, {z:.1f}) mm, turned as it grasped")

    def to_dict(self) -> dict[str, Any]:
        """The set-down as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe; ``pose`` as ``position_mm`` and ``quaternion_xyzw``, or None.
        """
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
class _Frame:
    """The frame a ``Located`` was placed from, as :meth:`Located.scene` reads the parts beside a target in it: the
    measured depth (CAMERA millimetres), the lens and where the camera stood at the shutter."""

    depth_mm: np.ndarray
    intrinsics: np.ndarray
    camera_to_base: np.ndarray


@dataclass(frozen=True, slots=True, eq=False)
class Measured:
    """What one frame of one camera read where given points of BASE should stand: :meth:`Locator.measure` on a frame it
    takes, :meth:`Located.measure` on the frame a locate placed. No detector ran: it says what the frame reads there,
    never what stands there.

    Per point, in the order given: ``in_frame``, whether it lands on a pixel of the frame in front of the camera;
    ``expected_mm``, the depth the camera reads where the point stands (its CAMERA z, NaN behind the camera);
    ``measured_mm``, the depth the frame measured at its pixel (NaN outside the frame, and where the sensor measured
    nothing); ``lab``, the CIE L*a*b* colour of that pixel (D65, L* from 0 to 100; NaN outside the frame, and where no
    colour image was kept).
    """

    camera: str
    captured_at_s: float
    in_frame: np.ndarray
    expected_mm: np.ndarray
    measured_mm: np.ndarray
    lab: np.ndarray

    @property
    def points(self) -> int:
        """How many points were read."""
        return int(self.expected_mm.shape[0])

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        measured = int(np.count_nonzero(np.isfinite(self.measured_mm)))
        return _ascii(f"camera {self.camera!r} read {self.points} point(s) at {self.captured_at_s:.3f} s: "
                      f"{int(np.count_nonzero(self.in_frame))} in the frame, {measured} with a depth measured")


def _lab_of_rgb(rgb: Any) -> np.ndarray:
    """``(N, 3)`` sRGB colours (0 to 255) as CIE L*a*b* under D65, the standard conversion. No camera profile: it
    compares two colours one camera saw, and says nothing absolute about either."""
    values = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    to_xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                       [0.2126729, 0.7151522, 0.0721750],
                       [0.0193339, 0.1191920, 0.9503041]])
    xyz = (linear @ to_xyz.T) / np.array([0.95047, 1.0, 1.08883])
    epsilon, kappa = 216.0 / 24389.0, 24389.0 / 27.0
    f = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    return np.column_stack([116.0 * f[:, 1] - 16.0, 500.0 * (f[:, 0] - f[:, 1]), 200.0 * (f[:, 1] - f[:, 2])])


def _measured(points_base_mm: Any, *, camera: str, captured_at_s: float, depth_mm: Any, intrinsics: Any,
              camera_to_base: Any, rgb: "np.ndarray | None") -> Measured:
    """What a frame (``depth_mm``, CAMERA mm, 0 or NaN where nothing was measured; ``rgb`` its colour, or ``None``) read
    where each of ``points_base_mm`` should stand, the camera at ``camera_to_base`` through ``intrinsics``: the inverse
    of how a locate places a pixel (``masked_points``, a pixel's centre at its whole index), so a point placed from a
    frame lands on the pixel it came from."""
    points = np.asarray(points_base_mm, dtype=np.float64).reshape(-1, 3)
    count = points.shape[0]
    matrix = np.asarray(camera_to_base, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"a frame is read where the camera stood, a finite 4x4 CAMERA to BASE, not {matrix.shape}")
    lens = np.asarray(intrinsics, dtype=np.float64)
    depth = np.asarray(depth_mm, dtype=np.float64)
    # Row by row, R^T (p - t): the point in the camera's own frame.
    in_camera = (points - matrix[:3, 3]) @ matrix[:3, :3]
    z = in_camera[:, 2]
    ahead = np.isfinite(z) & (z > 0.0)
    safe_z = np.where(ahead, z, 1.0)
    cols = np.rint(lens[0, 0] * in_camera[:, 0] / safe_z + lens[0, 2])
    rows = np.rint(lens[1, 1] * in_camera[:, 1] / safe_z + lens[1, 2])
    height, width = depth.shape[:2]
    in_frame = ahead & np.isfinite(cols) & np.isfinite(rows) & (cols >= 0) & (cols < width) & (rows >= 0) & \
        (rows < height)
    row, col = rows[in_frame].astype(np.int64), cols[in_frame].astype(np.int64)
    read = depth[row, col]
    measured = np.full(count, np.nan)
    measured[in_frame] = np.where(np.isfinite(read) & (read > 0.0), read, np.nan)
    lab = np.full((count, 3), np.nan)
    colour = None if rgb is None else np.asarray(rgb)
    if colour is not None and colour.ndim == 3 and colour.shape[:2] == depth.shape[:2] and colour.shape[2] >= 3:
        lab[in_frame] = _lab_of_rgb(colour[row, col, :3])
    return Measured(camera=str(camera), captured_at_s=float(captured_at_s), in_frame=in_frame,
                    expected_mm=np.where(ahead, z, np.nan), measured_mm=measured, lab=lab)


@dataclass(frozen=True, slots=True, eq=False)
class Located:
    """What one frame of one camera located, stamped at its shutter; or, from :meth:`Locator.look_around` on a camera on
    the wrist, what all its looks located, fused.

    A looked-around ``Located`` is the frame of the last look that saw the part: object 0 is the part the looks kept,
    its cloud fused over :attr:`looks_fused`, and an object only an earlier look saw is kept too, with no pixels in this
    frame, so it stays an obstacle to the part's grasps. With :attr:`refused` set nothing may be planned on it:
    :meth:`scene` and :meth:`set_down` raise ``LocatorRefused``.

    Attributes:
        camera (str): The rig id of the camera that took the frame.
        captured_at_s (float): The frame's shutter, host seconds; for a look around that reached none of its looks, when
            it ended.
        mounting (str): ``"eye_to_hand"`` (a fixed camera) or ``"eye_in_hand"`` (on the wrist).
        tool_to_base_mm (tuple[tuple[float, ...], ...] | None): Where the tool stood at the shutter, a 4 x 4 matrix in
            millimetres, for a camera on the wrist; ``None`` for a fixed camera.
        objects (tuple[LocatedObject, ...]): Every object placed in BASE, each with its label, score, box, mask, points
            (``points_base_mm``, N x 3 millimetres) and centre; object 0 is a look around's part.
        looks (tuple[str, ...]): The looks a look around located from, in the order the arm reached them; empty for one
            locate (default: ()).
        looks_fused (tuple[str, ...]): The looks whose surfaces make up object 0's cloud, this frame's look first; empty
            for one locate (default: ()).
        jaw_faces_seen (tuple[bool, bool] | None): Whether each jaw contact face of object 0's best grasp was seen, (jaw
            1, jaw 2); ``None`` where it was not judged (default: None).
        refused (str): Why nothing may be planned on it, or ``""``: a look the arm did not reach, a controller that
            stopped, or, with ``both_faces``, the face no look saw (default: "").
        hand_eye_gap_mm (float | None): How far apart, in millimetres, the looks measured the part's shared surface
            (median); above ``HAND_EYE_DRIFT_WARN_MM`` the calibration may have drifted, and it is said. ``None`` where
            fewer than two looks shared one (default: None).
        generated_view_deg (float | None): How far the one generated view turned about the part, degrees; ``None`` where
            none was generated (default: None).
        views_file (str): Where the looks' frames were kept with ``record_views``; ``""`` otherwise (default: "").
        closing_axis (ClosingAxis | None): The axis the look around judged object 0's grasp along; ``None`` for any
            axis, and for one locate (default: None).
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
    #: The frame this was placed from (a look around's: the last look that saw the part), which :meth:`scene` reads the
    #: parts the prompt did not name in; ``None`` where none was kept (a double, or no look reached).
    _frame: "_Frame | None" = field(default=None, repr=False)

    def split(self, prompt: str) -> "dict[str, tuple[int, ...]]":
        """Which objects each description of a class-list prompt located.

        Args:
            prompt (str): The prompt this was located for. For a class list (``class_list_prompt(["yellow bin", "blue
                bin"])``), each description; for one description, every object.

        Returns:
            dict[str, tuple[int, ...]]: Every description, in the prompt's order, mapped to the indices into
                :attr:`objects` of the objects whose label is that description; ``()`` for one no object carries. An
                object two descriptions claimed is ``ambiguous``: it is neither's, and stays an obstacle.
        """
        # Here, not at the module's top: the locator imports no model package until it is asked.
        from src.models.vlm.qwen import classes_of  # noqa: PLC0415

        descriptions = classes_of(prompt)
        if not descriptions:
            return {str(prompt): tuple(range(len(self.objects)))}
        return {description: tuple(index for index, obj in enumerate(self.objects) if obj.label == description)
                for description in descriptions}

    def keep_out(self, target: int) -> SegmentationOffer:
        """What a planner world has to leave out so the arm may reach object ``target``: its box and this frame's masks.

        Args:
            target (int): The index into :attr:`objects`.

        Returns:
            SegmentationOffer: Hand it to ``Robot.pick(..., keep_out=)`` or ``Robot.place(..., keep_out=)``.
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

        ```python
        best = located.scene(0, tree.robot).grasps().best
        if best is not None:
            robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))
        ```

        Args:
            target (int): The index into :attr:`objects`; 0 is a look around's part.
            robot_config (RobotConfig): The cell's robot section, ``tree.robot``: the support height and the jaw come
                from it.

        Returns:
            Scene: The object's measured surface, extruded down to the support, the jaw the cell's
                ``grasping.gripper_geometry`` describes, every other located object as observed obstacle points.
                ``scene.grasps()`` ranks the grasps (always the geometric generator, which ``SceneGrasps.generator``
                says); ``scene.part_bottom_mm`` is what a set-down hangs the part from. An object with no surface under
                its mask gives a scene with no grasps.

        Raises:
            LocatorRefused: This ``Located`` was :attr:`refused`; the message is the reason.
            IndexError: ``target`` is out of range.

        Where the cell names how its hand and camera naturally stand (``robot.natural_closing_axis``) every grasp is
        turned the way round nearer it. Object 0's scene of a look around chooses along the axis the looks judged its
        grasp along, and ``grasps()`` refuses any other (``ValueError``), so the grasp gripped is the grasp judged.
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
        scene = self._seen_beside(scene, index, robot_config, others)
        if index == 0 and len(self.looks_fused) >= 2:
            scene = replace(scene, _bottom_from_looks=True)
        if index == 0 and (self.looks or self.jaw_faces_seen is not None or self.closing_axis is not None):
            scene = replace(scene, _judged_along=self.closing_axis)
        return scene

    def _seen_beside(self, scene: Scene, index: int, robot_config: Any, others: "list[np.ndarray]") -> Scene:
        """``scene`` with what this frame shows beside object ``index`` that nobody segmented, by the planner world's
        rules (``robot.grasping.scene_obstacles``, the cell fixes' Track A), the support it stands on as the camera
        world finds it, and the frame's own corridor test for side approaches. ``scene`` as it is where the tree turns
        the rules off, no frame was kept, or the object has no pixels in it (an object only an earlier look saw)."""
        from src.robot.grasping.generation.scene_obstacles import (  # noqa: PLC0415
            SceneObstacleRules,
            corridor_seen_in,
            scene_obstacle_points,
        )

        frame = self._frame
        mask = np.asarray(self.objects[index].mask).astype(bool)
        if frame is None or mask.shape != np.shape(frame.depth_mm) or not mask.any():
            return scene
        if scene._side_approaches:  # noqa: SLF001 (the scene's own switch, set by from_robot_config)
            scene = replace(scene, _corridor_seen=corridor_seen_in(frame.depth_mm, frame.intrinsics,
                                                                   frame.camera_to_base))
        if not bool(getattr(robot_config.grasping, "scene_obstacles", False)):
            return scene
        rules = SceneObstacleRules.from_robot_config(robot_config)
        points = self.objects[index].points_base_mm
        model = _support_model(frame, robot_config, points)
        support = float(scene.support_height_mm)
        if model is not None and points.shape[0]:
            under = model.height_under(np.asarray(points, dtype=np.float64)[:, :2])
            if under is not None and float(under) > support:
                support = float(under)
        seen = scene_obstacle_points(frame.depth_mm, mask, frame.intrinsics, frame.camera_to_base, rules=rules,
                                     support_height_mm=support, support_model=model)
        logger.info("scene of object %d (%s): %s", index, self.objects[index].label, seen.render())
        obstacles = [*others, seen.points_base_mm] if seen.points_base_mm.shape[0] else others
        return replace(
            scene, support_height_mm=support,
            obstacle_points_base_mm=np.concatenate(obstacles, axis=0) if obstacles else None,
            _rules=rules, _held=tuple(model.solids) if model is not None else (),
            _seen_points=int(seen.points_base_mm.shape[0]), _seen_clusters=int(seen.clusters),
        )

    def set_down(self, target: int, *, grasp: Pose, part_bottom_mm: float,
                 air_mm: float = SET_DOWN_AIR_MM) -> SetDown:
        """Where the tool sets the part it grasped down on object ``target``: :meth:`SetDown.onto`.

        ```python
        set_down = onto.set_down(0, grasp=best.pose(), part_bottom_mm=scene.part_bottom_mm)
        if set_down.pose is not None:
            robot.place(set_down.pose, keep_out=onto.keep_out(0))
        ```

        Args:
            target (int): The index into :attr:`objects` of what the part goes on.
            grasp (Pose): Where the tool closed on the part, in BASE.
            part_bottom_mm (float): A lower bound on where the part's bottom stood, BASE Z millimetres: the
                ``part_bottom_mm`` of the scene its grasp came from.
            air_mm (float): The gap over the target's top when the hand opens, millimetres (default: 5.0).

        Returns:
            SetDown: The set-down; ``pose`` is ``None`` with a ``reason`` where it cannot be measured.

        Raises:
            LocatorRefused: This ``Located`` was :attr:`refused`.
            ValueError: A grasp not in BASE, or a bottom or an air that is not a finite number.
        """
        self._refuse_if_refused()
        if not 0 <= int(target) < len(self.objects):
            raise IndexError(f"object {target} of {len(self.objects)} located by camera {self.camera!r}")
        return SetDown.onto(self.objects[int(target)], grasp=grasp, part_bottom_mm=part_bottom_mm, air_mm=air_mm)

    def _refuse_if_refused(self) -> None:
        """Fail closed: nothing is planned on what a look around refused (:attr:`refused`)."""
        if self.refused:
            raise LocatorRefused(f"nothing is planned on what camera {self.camera!r} located: {self.refused}")

    def measure(self, points_base_mm: Any, image_bgr: "np.ndarray | None" = None) -> "Measured | None":
        """What this frame read where given points should stand, as :meth:`Locator.measure` reads a new frame.

        Args:
            points_base_mm (Any): The points, an N x 3 array in BASE millimetres.
            image_bgr (np.ndarray | None): The colour image this frame was located in (BGR), for each point's colour;
                ``None`` gives NaN colours (default: None).

        Returns:
            Measured | None: Per point, in order: ``in_frame``, ``expected_mm`` (the depth the camera reads where it
                stands), ``measured_mm`` (what the frame measured at its pixel) and ``lab`` (the pixel's CIE L*a*b*
                colour). ``None`` where no frame was kept (a double's, a look around that reached no look).
        """
        frame = self._frame
        if frame is None:
            return None
        rgb = None if image_bgr is None else np.asarray(image_bgr)[..., ::-1]
        return _measured(points_base_mm, camera=self.camera, captured_at_s=self.captured_at_s, depth_mm=frame.depth_mm,
                         intrinsics=frame.intrinsics, camera_to_base=frame.camera_to_base, rgb=rgb)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """What was located, as a person reads it: the camera, the looks, each object with its centre.

        Returns:
            str: ASCII, no trailing newline. ``print(located)`` shows the same.
        """
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
        """What was located, as plain data.

        Returns:
            dict[str, Any]: ``json.dumps`` safe; each object without its mask and points.
        """
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


def _support_model(frame: _Frame, robot_config: Any, target_points: np.ndarray) -> Any:
    """The surfaces ``frame`` shows the parts standing on, as the cell's camera world finds them on a build of the same
    frame (``support_surfaces.find_supports``), the target's own box held out; ``None`` where the tree turns support
    surfaces off or declares no bench."""
    from src.robot.safety.planning.perceived import DepthView, WorldBuildTuning  # noqa: PLC0415
    from src.robot.safety.planning.reservation import planner_world_limits  # noqa: PLC0415
    from src.robot.safety.planning.self_envelope import ROBOT_BASES  # noqa: PLC0415
    from src.robot.safety.planning.support_surfaces import find_supports  # noqa: PLC0415

    world = getattr(robot_config.safety, "planning_world", None)
    perceived = getattr(world, "perceived", None)
    if world is None or not bool(getattr(world, "enabled", False)) or perceived is None:
        return None
    if not bool(getattr(perceived, "support_surfaces", False)):
        return None
    limits = planner_world_limits(robot_config)
    if limits is None:
        return None
    tuning = WorldBuildTuning(
        pixel_stride=int(perceived.pixel_stride), voxel_size_mm=float(perceived.voxel_size_mm),
        cluster_voxel_mm=float(perceived.cluster_voxel_mm), min_points=int(perceived.min_points),
        margin_mm=float(perceived.margin_mm), max_boxes=int(perceived.max_boxes),
        floor_to_plane=bool(perceived.floor_to_plane), support_surfaces=True,
        support_allowance_mm=float(perceived.support_allowance_mm))
    model_name = str(getattr(getattr(robot_config, "ur", None), "model", "") or "").lower()
    keep_out: tuple[KeepOutBox, ...] = ()
    points = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    points = points[np.isfinite(points).all(axis=1)]
    if points.shape[0]:
        low, high = points.min(axis=0), points.max(axis=0)
        place = np.eye(4)
        place[:3, 3] = (low + high) / 2.0
        keep_out = (KeepOutBox.from_matrix("target", place, (high - low) / 2.0 + _TARGET_BOX_MARGIN_MM),)
    return find_supports([DepthView(surface_depth_mm=frame.depth_mm, intrinsics=frame.intrinsics,
                                    camera_to_base=frame.camera_to_base, name="locator")],
                         limits=limits, tuning=tuning, base=ROBOT_BASES.get(model_name), keep_out=keep_out)


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


class _DetectsNothing:
    """A perception backend that grounds nothing: what :meth:`Locator.measure` takes its frame through, so a measure
    costs a frame and never a detector's call."""

    def perceive(self, _image_bgr: Any, _prompt: str) -> tuple[()]:
        return ()


#: What the source of a measure is handed as its phrase: the backend above never reads it.
_MEASURE_PROMPT = "nothing: a measure grounds no phrase"


class Locator:
    """One open camera, a perception backend, and what it takes to place its frames in BASE.

    Build it with :meth:`from_tree` (one camera) or :meth:`for_cameras` (every camera of a cell, over one backend); then
    :meth:`locate` from one frame, or :meth:`look_around` for a part to grip.

    Args:
        camera (Any): An open camera owner (:class:`Camera`).
        backend (Any): A perception backend with ``perceive(image_bgr, prompt)``.
        calibration (Any): The camera's calibration, which places its frames in BASE.
        tool_pose (Callable[[], Pose] | None): The arm's ``get_tcp_pose``, for a camera on the wrist; ``None`` for a
            fixed camera.
        attempts (int): How many more frames a wrist camera takes while the tool moved.
        view (Any): A ``LiveView``, or ``None`` (default: None).
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
        """A locator over one open camera and a backend you built.

        Args:
            camera (Any): An open owner answering ``rig_id``, ``source``, ``handle()`` and ``calibration()``, as
                :class:`Camera` does.
            backend (Any): A perception backend with ``perceive(image_bgr, prompt)``.
            tool_pose (Maybe[Callable[[], Pose]]): The arm's ``get_tcp_pose``, read at each shutter to place a wrist
                camera's frame; required for a camera on the wrist, ignored for a fixed one (default: UNSET).
            attempts (Maybe[int]): How many more frames a wrist camera takes while the tool moved; unset is the schema's
                ``perceived.fresh_frame_attempts`` (default: UNSET).
            tool_frame (Maybe[Any]): The cell's ``robot.gripper.tool_frame``; a wrist camera's calibration must have
                been solved against it. Required on the wrist, ignored for a fixed camera (default: UNSET).
            view (Any): A ``LiveView``: the camera gets a window there, and every ``Located`` is pinned on it
                (default: None).

        Returns:
            Locator: The locator; the view gets the camera once every refusal passed.

        Raises:
            LocatorRefused: A rig with no depth, a wrist rig with no TCP reader, or a wrist rig whose calibration was
                not solved against ``tool_frame``.
            RigNotCalibrated: The rig declares no calibration; it names its key.
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
        """A locator for the cell a robot section describes, with the backend a models section builds, as the cell
        builds it.

        Args:
            robot_cfg (Any): The cell's robot section, ``tree.robot``.
            models_cfg (Any): Its models section, ``tree.app_config.models``.
            camera (Any): An open camera owner (:class:`Camera`).
            tool_pose (Maybe[Callable[[], Pose]]): The arm's ``get_tcp_pose``, read at each shutter to place a wrist
                camera's frame; required for a camera on the wrist, ignored for a fixed one (default: UNSET).
            view (Any): A ``LiveView``: the camera gets a window there, and every ``Located`` is pinned on it
                (default: None).

        Returns:
            Locator: The locator; the detector and the segmenter loaded.

        Raises:
            LocatorRefused: As :meth:`from_parts` refuses, before any model loads.
            RigNotCalibrated: The rig declares no calibration.
        """
        (locator,) = cls._over_one_backend(robot_cfg, models_cfg, [camera], tool_pose=tool_pose, view=view)
        return locator

    @classmethod
    def from_tree(cls, tree: Any, *, camera: Any, tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
                  view: Any = None) -> "Locator":
        """A locator for one camera of the cell a loaded tree describes, with the perception stack its models section
        builds.

        ```python
        with Camera.from_tree(tree) as camera:
            locator = Locator.from_tree(tree, camera=camera, tool_pose=robot.arm.get_tcp_pose)
        ```

        Args:
            tree (Any): A loaded tree, ``load_tree()``.
            camera (Any): An open camera owner (:class:`Camera`).
            tool_pose (Maybe[Callable[[], Pose]]): The arm's ``get_tcp_pose``, read at each shutter to place a wrist
                camera's frame; required for a camera on the wrist, ignored for a fixed one (default: UNSET).
            view (Any): A ``LiveView``: the camera gets a window there, and every ``Located`` is pinned on it
                (default: None).

        Returns:
            Locator: The locator; the detector and the segmenter loaded.

        Raises:
            ConfigError: The tree did not load.
            LocatorRefused: As :meth:`from_parts` refuses, before any model loads.
            RigNotCalibrated: The rig declares no calibration.
        """
        return cls.from_config(tree.robot, tree.app_config.models, camera=camera, tool_pose=tool_pose, view=view)

    @classmethod
    def for_cameras(cls, tree: Any, cameras: "Sequence[Any]", *, tool_pose: "Maybe[Callable[[], Pose]]" = UNSET,
                    view: Any = None) -> "list[Locator]":
        """One locator per open camera of a cell, in the order given, all over ONE backend.

        A locator per camera from :meth:`from_tree` would load every model once per camera; the backend is stateless per
        call, so the cell's own pick path shares one, and so does this.

        Args:
            tree (Any): A loaded tree, ``load_tree()``.
            cameras (Sequence[Any]): The open camera owners, the primary first.
            tool_pose (Maybe[Callable[[], Pose]]): The arm's ``get_tcp_pose``, for every camera on the wrist; ignored by
                a fixed one (default: UNSET).
            view (Any): A ``LiveView``: each camera gets its window (default: None).

        Returns:
            list[Locator]: One per camera, in the order given.

        Raises:
            LocatorRefused: No camera at all, or one rig it cannot place (no depth, a wrist rig with no TCP reader or
                solved against another tool frame): the lot is refused, before any model loads or window opens.
            RigNotCalibrated: A rig declares no calibration.
            ConfigError: The tree did not load.
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
        """Ground a prompt in one frame and place every object in BASE.

        Args:
            prompt (str): What to find: a phrase (``"a red cube"``), or a class list (``class_list_prompt(["yellow bin",
                "blue bin"])``) grounded in one call, every object under the description its box named
                (:meth:`Located.split` says which are whose).

        Returns:
            Located: Every object grounded, placed in BASE; with a view, pinned on the camera's window.

        Raises:
            PerceptionFrameMoved: A camera on the wrist and the tool would not hold still for the frame.
            LocatorRefused: The frame cannot be placed.
        """
        return self._placed(prompt).located

    def measure(self, points_base_mm: Any) -> Measured:
        """What one new frame reads where given points should stand.

        The frame is taken as :meth:`locate` takes one, through the same source, but the backend grounds nothing, so a
        measure costs a frame, never a detector's call.

        Args:
            points_base_mm (Any): The points, an N x 3 array in BASE millimetres.

        Returns:
            Measured: Per point, in order: ``in_frame``, ``expected_mm``, ``measured_mm`` and ``lab``, as
                :meth:`Located.measure` gives them.

        Raises:
            PerceptionFrameMoved: A camera on the wrist and the tool would not hold still.
            LocatorRefused: The frame cannot be taken or placed.
        """
        try:
            frame, camera_to_base, depth = self._take(_DetectsNothing(), _MEASURE_PROMPT)
        except Exception as exc:
            if self._view is not None:
                _on_view(self._view, "note", f"measure failed: {type(exc).__name__}: {exc}",
                         str(getattr(self._camera, "rig_id", "camera")))
            raise
        return _measured(points_base_mm, camera=str(getattr(self._camera, "rig_id", "camera")),
                         captured_at_s=float(frame.timestamp if frame.timestamp is not None else float("nan")),
                         depth_mm=depth, intrinsics=frame.intrinsics, camera_to_base=camera_to_base, rgb=frame.rgb)

    def look_around(self, prompt: str, looks: "Look | None" = None, *, robot: Any, both_faces: bool = False,
                    record_views: bool = False, closing_axis: "Maybe[ClosingAxisLike]" = UNSET) -> Located:
        """Find the part a prompt names from each look in turn, every look fused with the ones before, until its grasp
        is safe; a camera on the wrist moves there through ``robot``'s own verbs.

        Args:
            prompt (str): What to find, as :meth:`locate` takes it.
            looks (Look | None): One look or several: ``JointPositions``, or ``"home"``. ``None`` takes the pose the arm
                stands at as the one look (default: None).
            robot (Any): The connected :class:`Robot` that moves the camera; it must keep its robot section
                (``Robot.from_tree``) and its hand must be a parallel jaw.
            both_faces (bool): Look on until both jaw contact faces of the chosen grasp were seen, and refuse a part no
                look showed both of: the switch for safety-critical processes (default: False).
            record_views (bool): Keep the looks' frames for training, one file for the look around: each look's colour
                image, depth, lens, tool pose and camera pose, and the part's fused cloud (default: False).
            closing_axis (Maybe[ClosingAxisLike]): Judge only the grasps closing along this axis (``"-y"``, ...), as
                ``scene.grasps(closing_axis=...)`` takes it; the ``Located`` keeps it and object 0's scene holds to it
                (default: UNSET, any axis).

        Returns:
            Located: Object 0 the part with its fused cloud, every other object as the looks saw it, and
                :attr:`Located.looks`, :attr:`Located.looks_fused`, :attr:`Located.jaw_faces_seen` and
                :attr:`Located.hand_eye_gap_mm`. A look around that reached no look, or stopped, says why in
                :attr:`Located.refused`, and nothing may be planned on it.

        Raises:
            LocatorRefused: ``robot`` keeps no robot section, or its hand is not a parallel jaw; raised before anything
                moves.
            ValueError: ``looks`` names no look, an entry is not a look, or ``closing_axis`` names no axis; raised
                before anything moves.
            TypeError: A look entry or a ``closing_axis`` of another type, raised before anything moves.

        How the looks run. Each look's objects are fused with what the looks before located, by association at the
        wrist's tolerances; object 0 of the first look that measured its surface is the part, and it stays the part.
        After each look the part's grasp is computed again on everything the looks saw of it, and the looking stops at
        the first look whose grasp is valid and which the looks agree on. A safe grasp point is the goal, never a full
        scan. Once every declared look was visited and the grasp is still not safe enough, one view is generated: the
        last look turned about the part toward the face no look showed, every turn screened by the arm before it moves
        and driven on the straight joint line alone (no planner, no detour); an arm that cannot generates none and says
        why. A look the planner or a guard refused before anything was sent is skipped and said; a motion that failed
        once commanded, a controller that cannot move, or a camera that could not vouch for the cell end the looking
        with nothing else commanded. Nothing is said to the hand.

        The frames of the looks are held in the arm's live planner world, each where it was taken, so the pick that
        follows plans against everything the looks saw; ``Robot.pick`` lets them go when it ends, however it ends, as do
        the next look around and the disconnect. A program that tries another candidate after a failed pick looks around
        again first. A fixed camera locates once, where it stands, and moves nothing.
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

    def _take(self, backend: Any, prompt: str, *,
              object_labels: "tuple[str, ...]" = ()) -> "tuple[Any, np.ndarray, np.ndarray]":
        """One frame through the pick frame's source over ``backend``: the frame, where the camera stood at its shutter
        (CAMERA to BASE, 4x4 mm) and the depth it measured. A locate and a measure take their frames here alike.

        ``object_labels`` are the descriptions of a class list, which the source maps each box's label onto (and onto
        none where it names another thing or two of them); ``()`` passes the labels through, as every locate of one
        description always did. No colour is judged here: a bin's mask holds whatever lies in it, and the colour
        check's numbers were measured on parts on the mat, so a locate keeps what the detector called each object."""
        rig_id = str(getattr(self._camera, "rig_id", "camera"))
        source = RealSenseVisionPerceptionSource(
            streamer=_DepthOnly(self._camera.handle(), rig_id), backend=backend, prompt=prompt,
            object_labels=tuple(object_labels), colour_check=ColourCheck.OFF)
        wrist = self._tool_pose is not None
        if wrist:
            source.stamp_tool_pose_with(
                self._tool_pose,  # type: ignore[arg-type]
                motion_tolerance=(float(self._calibration.shutter_motion_tolerance_mm),
                                  float(self._calibration.shutter_motion_tolerance_deg)),
                attempts=self._attempts,
            )
        frame = source.acquire()
        if wrist and frame.tool_pose is None:
            raise LocatorRefused(f"camera {rig_id!r} is on the wrist and its frame carries no tool pose")
        # The composition the hand finder over a wrist camera shares, so the two place one frame alike.
        camera_to_base = camera_to_base_at_shutter(self._calibration, frame.tool_pose if wrist else None)
        depth = frame.surface_depth_map if frame.surface_depth_map is not None else frame.depth_map
        return frame, camera_to_base, depth

    def _locate(self, prompt: str) -> "_Placed":
        """What one frame located, that frame's colour image as the backend segmented it (BGR) where the source kept
        it, and the depth, lens, placement and surfaces a fusion of looks reads.

        A class list (``src.models.vlm.qwen.class_list_prompt``) is grounded in this one call, and each object's label
        is the description its box named (:meth:`Located.split`)."""
        # Here, not at the module's top: the locator imports no model package until it locates.
        from src.models.vlm.qwen import classes_of  # noqa: PLC0415

        rig_id = str(getattr(self._camera, "rig_id", "camera"))
        frame, camera_to_base, depth = self._take(self._backend, prompt, object_labels=classes_of(prompt))
        wrist = self._tool_pose is not None
        tool_to_base = np.asarray(frame.tool_pose.to_matrix(), dtype=np.float64) if wrist else None
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
            _frame=_Frame(depth_mm=np.asarray(depth, dtype=np.float64),
                          intrinsics=np.asarray(frame.intrinsics, dtype=np.float64),
                          camera_to_base=np.asarray(camera_to_base, dtype=np.float64)),
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

    The robot verbs' criterion (``handling._controller_refusal``): the halt latch first, of any arm that carries one
    (the dummy included), then a status that says so in words, said in the halt's own words (a person confirms the cell
    is clear, then Restart; a halt has no stop to clear); otherwise only an arm that says (``SupportsRobotStatus``) is
    asked, ``not is_operational`` cannot move, and a controller whose state cannot be read cannot be said to move. Not
    that verb's sentence, which says nothing was commanded: here the motion to the look may have been.
    """
    from src.robot.core.arm_capabilities import (  # noqa: PLC0415
        SupportsRobotStatus,
        halt_state_of,
        halted_refusal,
    )

    halted = halt_state_of(arm)
    if halted is not None:
        return halted_refusal(halted.reason, "it stands where the halt left it")
    if not isinstance(arm, SupportsRobotStatus):
        return ""
    try:
        status = arm.get_robot_status()
    except Exception as exc:  # noqa: BLE001 (a controller that cannot be asked cannot be said to move)
        return f"the controller's state could not be read ({type(exc).__name__}: {exc})"
    said = getattr(status, "halted", "")
    if isinstance(said, str) and said:
        return halted_refusal(said, "it stands where the halt left it")
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
