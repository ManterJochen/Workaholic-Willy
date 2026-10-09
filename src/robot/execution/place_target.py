"""Where a task lets its part go: the bin its camera found, kept and looked at again, or a pose a person taught.

**The bin** (the owner's OD 19, with Q3, Q5 and Q13 A, 2026-09-30). A task that places "into the blue bin" finds the bin
with the cell's own detector on the cell's own cameras (:func:`locators_for_service`: the pick service lends its
perception backend, so no second copy of a model loads), before its first pick (:func:`survey`): the arm goes to each
of the task's looks in turn, the camera locates the bin's phrase (a program may have it count the part's phrase too; a
task does not, 2026-10-08: the count was one grounding for a chat number), and the first look whose bin has at least
:data:`MIN_TARGET_POINTS` points of surface is the look it is kept from; of several, the most confident. A bin an
earlier task kept (``known``) is looked at again from its look first, as before a drop; followed there, it is kept and
no other look runs. What a task keeps of it (:class:`KeptTarget`) is measured, never written:

* its rim, the 95th percentile of the heights seen (``SetDown``'s top, so a stray pixel above it lifts nothing);
* its footprint and its turn, the smallest rectangle about it (``height_map.turn_of``), between the 1st and 99th
  percentiles of its rim band (the surface within 10 mm of the rim, its walls' tops) along its own axes;
* its opening, the gap between the inner edges of its walls' tops along each of those axes where the inside reads lower
  than the rim (:data:`OPENING_MIN_MM` or more), read from the rim band alone across the middle half of the band's own
  extent, so the inner faces and the floor a slanted camera sees fill nothing; a box with a flat top has none, and its
  part is set down on that top.

Before every drop the camera looks at the bin again from that look (:func:`recheck`). First its rim band's depth and
colour, in one frame with no detector (the owner, 2026-10-08: speed first): where the walls' tops read where they read
when the bin was last seen, and in their colour, the bin stands where it stood and is followed there; wherever that
check is unsure, a bin hidden, moved by a few millimetres, swapped or gone, the detector looks, as it always did: the
bin seen nearest the kept one is followed only within min(:data:`FOLLOW_WITHIN_MM`, half the kept footprint's diagonal),
a footprint within :data:`FOOTPRINT_TOLERANCE` of the kept one, side for side, and a rim within :data:`RIM_TOLERANCE_MM`
of the kept one's; anything else is the target lost, so a bin removed or swapped is never mistaken for it. The kept one
is the bin the survey found, never a later sighting, so a bin that creeps is not followed further and further. The drop
(:func:`drop_plan`) is ``SetDown.onto`` the bin, then four rules in order: the air over the rim is the task's rim air
(Q3: :func:`rim_clearance_mm`, 20 mm as shipped, the operator's choice of 10 to 50); a grasp within
:data:`VERTICAL_WITHIN_DEG` of vertical is turned about the vertical until it closes along the cell's natural closing
axis (``robot.natural_closing_axis``), any other keeps its own turn; the arm is asked where it would stand
(``nearest_configuration``) and the exact guard and the planner screen that (``screen_configuration``), and no
configuration or an ERROR is unreachable; the part, turned as it will hang, must fit the opening (or the top).

**Every place of a sort** (the owner, 2026-10-09: each kind of part into its own bin, by colour and kind). Every bin is
found before the first pick (:func:`survey_places`): from each look in turn, one locate of the phrases of every place
still missing, written as one class list (:data:`CLASS_LIST_SEPARATOR`), each located object the place its label names
exactly, and the survey ends once every place is kept; a place no look saw is named (:attr:`PlacesSurvey.missing`). A
bin the check before a drop lost (not seen, moved too far, another size there) is found again (:func:`relocate`): the
looks once more, for it alone, while the task moves the arm, and only a bin of the size the survey found and the colour
the task followed, standing in no other place, is taken; the drop is planned anew over it. Found nowhere, a person
decides. A check of one place replaces only its own keep-out region (``ExclusionZones.forget_region``). Where the
detector looks at one place of several, at a check or a search, it grounds every place in one locate (``together``)
and takes only what it labels that place's phrase, so a second bin in view is never taken for it.

**A taught pose** (the build plan's item 7). The person teaches a place pose with empty jaws, the fingertips where the
part's BOTTOM is to be let go; each part hangs below the tool by its own amount, so the tool goes to the taught pose
raised in BASE Z by the part's hang (:func:`pose_drop`): the grasp's Z less the part's declared support, an upper bound,
so every error goes toward more air, or the declared ``payload.length_mm`` where the pick reports no grasp. The raised
drop is screened the same way.

**The tilt is the grasp's** (both drops). The hang bounds how far the part reaches below the tool only while the part
hangs as it was gripped: a turn about a level axis swings a part held off its middle further down by as much as it
reaches out. So a drop is only ever turned about the vertical, which leaves every height under the tool as it was: a
taught drop to the taught heading, a camera drop to the natural axis, each keeping the tilt the tool gripped with (the
taught turn whole only where the pick reports no grasp, with the declared length as the hang).

**Set down, not dropped, and never piled** (the owner, 2026-10-08 night; ``robot.place``, each off as shipped):

* *Below a box's rim* (``drop_plan(..., below_rim_mm=)``, :func:`_below_the_rim`): the part's bottom ``below_rim_mm``
  under the rim, never lower than :data:`INSIDE_AIR_MM` over the highest thing the check before the drop read inside
  the box (:class:`InsideRead`: its floor, or the parts already in it, read upward place by place on the frame the rim
  was read in, so every error goes up). The part as it hangs and the open hand (:class:`OpenHand`), wherever its
  fingertips go below the rim, keep a margin inside the opening, judged here because nothing models the part and the
  guard judges the hand closed on it; the drop over the rim rides along (``DropPlan.instead``) for a line in the guard
  refuses. Anything else lets the part go over the rim, and the plan says why.
* *The hang from what the part stood on* (:func:`part_bottom`): the lowest reading of a surface the pick's looks found
  under the part's cloud, or the cloud's lowest point where that is lower, never the declared support where the looks
  read one.
* *Side by side on a flat place* (:class:`Spot`): the spots of a grid about a taught pose (:func:`grid_spots`) or over a
  target's top (:func:`top_spots`), the camera's reading first (:func:`views_spot_reader` from the pick's looks,
  :func:`top_spot_reader` from the check), then the spots the task has not filled (:func:`choose_spot`).
* *The carry over the rim*, what the task reads for it off the pick's own looks: the bin checked on the frame of the
  look it was found from (:func:`the_bins_look`, :func:`recheck_on_look`: no detector, no new frame), and the highest
  thing on the way to it (:func:`highest_on_the_way`).

Nothing here commands the arm but the surveys' looks (:func:`survey`, :func:`survey_places`), each through the arm's own
judged verb (``move_to_look``); a place looked for again (:func:`relocate`) moves nothing, the task drives its looks. The
drop is placed by the task through ``Robot.place``. Every refusal is an answer, never a raise; a programmer's error
raises.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Iterable, Iterator, Mapping, Sequence

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.constants import create_robot_logger
from src.robot.core.errors import CameraWorldUnavailable, RobotConnectionError, RobotError
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry.closing_axis import ClosingAxis
    from src.robot.core import JointPositions
    from src.robot.execution.looks import LookPose
    from src.robot.execution.motion import MotionReport
    from src.robot.perception.locator import Located, LocatedObject, Locator

__all__ = [
    "BELOW_RIM_LEAST_MM",
    "CLASS_LIST_SEPARATOR",
    "COLOUR_CHECK_DELTA_E",
    "DEPTH_CHECK_AGREE",
    "DEPTH_CHECK_FARTHER",
    "DEPTH_CHECK_SEEN",
    "DEPTH_CHECK_TOLERANCE_MM",
    "FOLLOW_WITHIN_MM",
    "FOOTPRINT_TOLERANCE",
    "INSIDE_AIR_MM",
    "KEEP_OUT_MARGIN_MM",
    "MIN_TARGET_POINTS",
    "OPENING_MARGIN_MM",
    "OPENING_MIN_MM",
    "RELOCATE_COLOUR_DELTA_E",
    "RIM_AIR_DEFAULT_MM",
    "RIM_AIR_MAX_MM",
    "RIM_AIR_MIN_MM",
    "RIM_TOLERANCE_MM",
    "SPOT_BAND_MM",
    "SURFACE_TOLERANCE_MM",
    "VERTICAL_WITHIN_DEG",
    "WAY_SEEN_SHARE",
    "DropPlan",
    "InsideRead",
    "KeptTarget",
    "OpenHand",
    "PartBottom",
    "PlaceFound",
    "PlacesSurvey",
    "Relocated",
    "Relocation",
    "Screen",
    "Spot",
    "Survey",
    "TargetCheck",
    "choose_spot",
    "drop_plan",
    "grid_spots",
    "highest_on_the_way",
    "located_image",
    "locators_for_service",
    "look_points",
    "nominal_drop",
    "part_bottom",
    "part_reach_mm",
    "pose_drop",
    "recheck",
    "recheck_on_look",
    "relocate",
    "rim_clearance_mm",
    "screen_joints",
    "screen_pose",
    "survey",
    "survey_places",
    "the_bins_look",
    "top_spot_reader",
    "top_spots",
    "views_spot_reader",
]

logger = create_robot_logger(__name__, "place_target.log")

#: How many points of surface a target needs before a task keeps it: the count ``SetDown`` reads a top from. Fewer say
#: where a few pixels are, not where a surface is.
MIN_TARGET_POINTS = 20
#: Which percentile of a target's heights is its rim: ``SetDown``'s top (``locator._TOP_PERCENTILE``).
_RIM_PERCENTILE = 95.0
#: How far below its rim a target's surface still counts as its rim's top band, mm (``locator._TOP_BAND_MM``).
_RIM_BAND_MM = 10.0
#: Which percentiles a footprint is read between, so one stray point is no edge. A choice, not a measurement.
_FOOTPRINT_PERCENTILES = (1.0, 99.0)
#: The narrowest gap between a bin's walls that counts as its opening, mm: anything narrower is the spacing of a flat
#: top's points, or no opening a part goes through. A choice, not a measurement.
OPENING_MIN_MM = 20.0
#: A grasp whose approach is within this many degrees of straight down is dropped along the cell's natural closing axis
#: (the build plan's item 6); any other keeps its own turn, so the part hangs as it was gripped.
VERTICAL_WITHIN_DEG = 5.0
#: The air over a bin's rim a task leaves when the operator chose none (Q3): ``SET_DOWN_AIR_MM`` and the perceived
#: world's margin as shipped (5 + 15 mm), the clearance the planner keeps about what the camera saw of the rim.
RIM_AIR_DEFAULT_MM = 20.0
#: The owner's bounds on the air over a bin's rim (Q3: 10 to 50 mm): the operator's choice, and the cell's own.
RIM_AIR_MIN_MM = 10.0
RIM_AIR_MAX_MM = 50.0
#: How far a kept bin may have moved, at most, and still be followed (Q13 (b)); half its footprint's diagonal where that
#: is less.
FOLLOW_WITHIN_MM = 100.0
#: How far a kept bin's footprint may differ, side for side, and still be the same bin (Q13 (b)).
FOOTPRINT_TOLERANCE = 0.2
#: How far a kept bin's rim may stand from where it was, up or down, mm, and still be the same bin: a bin of the same
#: footprint and another height in its place is another bin, and the drop would follow its rim down. A choice, not a
#: measurement: a rim read twice from the same look agrees within millimetres.
RIM_TOLERANCE_MM = 20.0
#: How far a kept bin's footprint is grown on every side when a task keeps it out of its picks, mm: a part against the
#: inside of a wall stands within this of the footprint read between its percentiles. A choice, not a measurement.
KEEP_OUT_MARGIN_MM = 10.0

# The quick check of a kept bin (:func:`recheck`, 2026-10-08): its rim band read again in one frame, before the detector
# is asked. Every bound is a choice, and every check logs what it read against them; where one is not met, the detector
# decides as it always did, so a bound too tight only costs the detector's time. Read on the cell's recorded wrist frames
# of 2026-10-07 and 10-08 (each bin kept by its colour from one frame, its rim band read in the others of that look): a
# bin left alone read 98 to 100 % seen, 96.5 to 100 % within 10 mm, 0 to 3.5 % farther and dE 0.3 to 7.5 apart; a bin
# moved by about 2 mm read 5.2 to 6.1 % farther (the detector looks); a bin moved between the two days, 0 to 2 % within;
# the yellow and the blue bin's rims stand dE 120 and 128 apart.
#: How much of the rim band must land in the frame and read no nearer than it should: the hand or the carried part in
#: front of it hides it.
DEPTH_CHECK_SEEN = 0.80
#: How far a rim point may read from where it should, either way, mm.
DEPTH_CHECK_TOLERANCE_MM = 10.0
#: How much of the rim band seen must read within :data:`DEPTH_CHECK_TOLERANCE_MM`.
DEPTH_CHECK_AGREE = 0.90
#: How much of the rim band seen may read farther than :data:`DEPTH_CHECK_TOLERANCE_MM`: a wall that is gone.
DEPTH_CHECK_FARTHER = 0.05
#: How far the rim band's median colour may stand from the one kept, CIE76 delta E.
COLOUR_CHECK_DELTA_E = 25.0

#: The screen verdicts that mean no move goes there (``PoseScreen.is_error``).
_ERRORS = frozenset({"guard_refused", "planner_refused"})


# ---------------------------------------------------------------------------------------------------------------------
# What a task keeps of a bin
# ---------------------------------------------------------------------------------------------------------------------


def _finite_points(points: Any) -> np.ndarray:
    cloud = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return cloud[np.isfinite(cloud).all(axis=1)]


def _turned(xy: np.ndarray, yaw_rad: float) -> tuple[np.ndarray, np.ndarray]:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return c * xy[:, 0] + s * xy[:, 1], -s * xy[:, 0] + c * xy[:, 1]


def _largest_gap(values: np.ndarray) -> "tuple[float, float, float]":
    """The widest gap between neighbouring ``values``, and the values on either side of it; ``(0, nan, nan)`` for
    fewer than two."""
    if values.shape[0] < 2:
        return 0.0, math.nan, math.nan
    ordered = np.sort(values)
    steps = np.diff(ordered)
    at = int(np.argmax(steps))
    return float(steps[at]), float(ordered[at]), float(ordered[at + 1])


def _opening_gaps(along: np.ndarray, across: np.ndarray, z: np.ndarray,
                  rim_mm: float) -> "list[tuple[float, float, float]] | None":
    """The gap between the inner edges of a target's wall tops along each of its own axes, each with the edges on
    either side of it, ``[(gap, low, high) along, (gap, low, high) across]``, or ``None`` for no opening.

    Along each axis, the rim band's points (the walls' tops) across the middle half of the band's own extent along the
    other axis: an open bin's two end walls stand there, and the widest gap between them is its inside; a flat top fills
    it. The band alone, and its own extent: a camera at a slant sees most of a bin as its far wall's inner face and its
    floor, and a strip read off every point's spread lands on a wall top that spans it, which no gap crosses. A part
    standing above the band inside the bin narrows the gap, never widens it, so the error goes toward a smaller
    opening. ``None`` where the inside does not read lower than the rim (no point below the band), or a gap is under
    :data:`OPENING_MIN_MM`.
    """
    if not bool(np.any(z < rim_mm - _RIM_BAND_MM)):
        return None
    band = z >= rim_mm - _RIM_BAND_MM
    low_pct, high_pct = _FOOTPRINT_PERCENTILES
    gaps: list[tuple[float, float, float]] = []
    for axis, other in ((along, across), (across, along)):
        low, high = float(np.percentile(other[band], low_pct)), float(np.percentile(other[band], high_pct))
        middle, quarter = (low + high) / 2.0, (high - low) / 4.0
        strip = band & (other >= middle - quarter) & (other <= middle + quarter)
        gaps.append(_largest_gap(axis[strip]))
    if min(gap for gap, _low, _high in gaps) < OPENING_MIN_MM:
        return None
    return gaps


def _opening(along: np.ndarray, across: np.ndarray, z: np.ndarray, rim_mm: float) -> "tuple[float, float] | None":
    """The gap between the inner edges of a target's wall tops along each of its own axes (:func:`_opening_gaps`), or
    ``None`` for no opening."""
    gaps = _opening_gaps(along, across, z, rim_mm)
    if gaps is None:
        return None
    return gaps[0][0], gaps[1][0]


def _rim_middle(points: np.ndarray, rim_mm: float) -> tuple[float, float]:
    """The middle of a target's rim band in BASE X and Y, as ``SetDown`` places a part over it."""
    band = points[points[:, 2] >= rim_mm - _RIM_BAND_MM][:, :2]
    low = np.percentile(band, 100.0 - _RIM_PERCENTILE, axis=0)
    high = np.percentile(band, _RIM_PERCENTILE, axis=0)
    return float((low[0] + high[0]) / 2.0), float((low[1] + high[1]) / 2.0)


@dataclass(frozen=True, eq=False)
class KeptTarget:
    """A target a task found and keeps (a bin), measured from what one camera saw of it, BASE millimetres.

    ``rim_mm`` is the 95th percentile of its heights; ``centre_xy_mm`` the middle of its footprint, which is
    ``footprint_mm`` long along its own axes, turned ``yaw_rad`` from BASE X; ``opening_mm`` the gap between its walls'
    inner edges along those axes, ``None`` where it shows no inside (a flat top). ``camera`` is the rig that saw it,
    ``look`` the look it was seen from (``None`` for a fixed camera) and ``look_label`` how that look reads, ``seen_at``
    the frame's shutter, ``phrase`` what it was located for, ``image_png`` the sighting drawn over (display only), and
    ``colour_lab`` the median CIE L*a*b* colour of its rim band in the frame it was seen in, ``None`` where that frame
    kept no colour: what :func:`recheck` compares a quick look at the rim against. Build one with :meth:`of`.
    """

    label: str
    score: float | None
    points_base_mm: np.ndarray = field(repr=False)
    rim_mm: float
    centre_xy_mm: tuple[float, float]
    footprint_mm: tuple[float, float]
    opening_mm: "tuple[float, float] | None"
    yaw_rad: float
    camera: str
    look: "LookPose | None"
    look_label: str | None
    seen_at: float
    phrase: str = ""
    image_png: "bytes | None" = field(default=None, repr=False)
    colour_lab: "tuple[float, float, float] | None" = None

    @classmethod
    def of(cls, obj: "LocatedObject", *, camera: str, look: "LookPose | None", seen_at: float, phrase: str = "",
           image_png: "bytes | None" = None) -> "KeptTarget | None":
        """What a task keeps of ``obj``, a located object, or ``None`` where fewer than :data:`MIN_TARGET_POINTS`
        points of its surface were measured."""
        from src.robot.execution.looks import look_label  # noqa: PLC0415
        from src.robot.safety.planning.height_map import turn_of  # noqa: PLC0415

        points = _finite_points(obj.points_base_mm)
        if points.shape[0] < MIN_TARGET_POINTS:
            return None
        rim = float(np.percentile(points[:, 2], _RIM_PERCENTILE))
        yaw = float(turn_of(points[:, :2]))
        along, across = _turned(points[:, :2], yaw)
        # The outline is the rim band's: a bin's walls stand straight or flare, so its tops reach its outer edge, and
        # the inner faces and the floor a camera sees would pull percentiles of every point inward.
        band = points[:, 2] >= rim - _RIM_BAND_MM
        low_pct, high_pct = _FOOTPRINT_PERCENTILES
        a_low, a_high = (float(np.percentile(along[band], q)) for q in (low_pct, high_pct))
        c_low, c_high = (float(np.percentile(across[band], q)) for q in (low_pct, high_pct))
        middle_along, middle_across = (a_low + a_high) / 2.0, (c_low + c_high) / 2.0
        c, s = math.cos(yaw), math.sin(yaw)
        centre = (c * middle_along - s * middle_across, s * middle_along + c * middle_across)
        score = None if obj.score is None or not math.isfinite(float(obj.score)) else float(obj.score)
        return cls(
            label=str(obj.label), score=score, points_base_mm=points, rim_mm=rim,
            centre_xy_mm=(float(centre[0]), float(centre[1])), footprint_mm=(a_high - a_low, c_high - c_low),
            opening_mm=_opening(along, across, points[:, 2], rim), yaw_rad=yaw, camera=str(camera), look=look,
            look_label=None if look is None else look_label(look), seen_at=float(seen_at), phrase=str(phrase),
            image_png=image_png,
        )

    @property
    def points(self) -> int:
        """How many points of its surface it was measured from."""
        return int(self.points_base_mm.shape[0])

    @property
    def on_the_wrist(self) -> bool:
        """Whether a camera on the wrist saw it, from a look the arm goes back to before each drop."""
        return self.look is not None

    @property
    def diagonal_mm(self) -> float:
        """The diagonal of its footprint."""
        return float(math.hypot(*self.footprint_mm))

    @property
    def rim_band_mm(self) -> np.ndarray:
        """The points of its rim band, BASE mm: its surface within 10 mm of its rim, the walls' tops. What a quick look
        at it reads again (:func:`recheck`): a part placed inside changes its floor, never them."""
        return self.points_base_mm[self.points_base_mm[:, 2] >= self.rim_mm - _RIM_BAND_MM]

    def inner_opening(self) -> "tuple[tuple[float, float], tuple[float, float]] | None":
        """Where its opening lies: the inner edges of its walls' tops along its own axes (BASE X and Y turned by
        ``yaw_rad``, about BASE 0, 0), ``((low, high) along, (low, high) across)`` mm; ``None`` where it shows no
        inside. The gaps between them are :attr:`opening_mm`."""
        along, across = _turned(self.points_base_mm[:, :2], self.yaw_rad)
        gaps = _opening_gaps(along, across, self.points_base_mm[:, 2], self.rim_mm)
        if gaps is None:
            return None
        return (gaps[0][1], gaps[0][2]), (gaps[1][1], gaps[1][2])

    def as_object(self) -> "LocatedObject":
        """This target as the located object ``SetDown.onto`` measures a drop over."""
        from src.robot.perception.locator import LocatedObject  # noqa: PLC0415

        centre = (self.centre_xy_mm[0], self.centre_xy_mm[1], self.rim_mm)
        return LocatedObject(label=self.label, score=self.score, box_px=None, mask=np.zeros((1, 1), dtype=bool),
                             points_base_mm=self.points_base_mm, centre_mm=centre)

    def keep_out_region(self, margin_mm: float = KEEP_OUT_MARGIN_MM) -> ExclusionRegion:
        """Its footprint, grown by ``margin_mm`` on every side, kept out of every pick for every label (Q4): no pick
        takes the bin, or a part already inside it."""
        return ExclusionRegion.rectangle(
            self.centre_xy_mm, (self.footprint_mm[0] + 2.0 * margin_mm, self.footprint_mm[1] + 2.0 * margin_mm),
            yaw_rad=self.yaw_rad, reason=f"the {self.label} it places into")

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        score = "" if self.score is None else f" ({self.score:.2f})"
        opening = ("no opening seen" if self.opening_mm is None
                   else f"opening {self.opening_mm[0]:.0f} x {self.opening_mm[1]:.0f} mm")
        where = "a fixed camera" if self.look_label is None else f"look {self.look_label}"
        line = (f"{self.label}{score}: rim at {self.rim_mm:.1f} mm, middle ({self.centre_xy_mm[0]:.0f}, "
                f"{self.centre_xy_mm[1]:.0f}) mm, footprint {self.footprint_mm[0]:.0f} x {self.footprint_mm[1]:.0f} mm "
                f"turned {math.degrees(self.yaw_rad):.0f} deg, {opening}; seen by {self.camera} from {where}, "
                f"{self.points} point(s)")
        if self.colour_lab is not None:
            line += ", its rim's colour L*a*b* ({:.0f}, {:.0f}, {:.0f})".format(*self.colour_lab)
        return line.encode("ascii", "backslashreplace").decode("ascii")

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe, in the shape the console shows a target (``TargetOut``, less its overlay):
        a score the detector gave none of reads 0.0."""
        return {
            "label": self.label,
            "score": 0.0 if self.score is None else float(self.score),
            "centre_mm": [self.centre_xy_mm[0], self.centre_xy_mm[1], self.rim_mm],
            "rim_mm": self.rim_mm,
            "footprint_mm": [self.footprint_mm[0], self.footprint_mm[1]],
            "opening_mm": None if self.opening_mm is None else [self.opening_mm[0], self.opening_mm[1]],
            "look": self.look_label,
            "seen_at": self.seen_at,
            "camera": self.camera,
            "points": self.points,
        }


def _most_confident(targets: "list[KeptTarget]") -> "KeptTarget | None":
    """Of several targets seen, the most confident (Q5): the detector's score, one with none last; then more points."""
    if not targets:
        return None
    return max(targets, key=lambda target: (target.score is not None, target.score or 0.0, target.points))


# ---------------------------------------------------------------------------------------------------------------------
# The service's own cameras and detector
# ---------------------------------------------------------------------------------------------------------------------


class _Sightings:
    """What a task's locator is handed as its view: per prompt, the colour image (BGR) of the last frame located."""

    def __init__(self, rig_id: str, *, keep: bool) -> None:
        self.rig_id = rig_id
        self.images: "dict[str, np.ndarray] | None" = {} if keep else None

    def watch(self, *_cameras: Any) -> None:
        return None

    def show_located(self, located: Any, image: Any = None, *, prompt: str = "") -> None:
        if self.images is not None and image is not None:
            self.images[str(prompt)] = np.array(image, copy=True)

    def note(self, _text: str, _rig_id: "str | None" = None) -> None:
        return None


#: The locators :func:`locators_for_service` built, and what each keeps: weakly, so a locator that is let go is gone.
_SIGHTINGS: "weakref.WeakKeyDictionary[Any, _Sightings]" = weakref.WeakKeyDictionary()


def locators_for_service(service: Any, *, keep_image: bool = True, tool_frame: Maybe[Any] = UNSET,
                         attempts: Maybe[int] = UNSET) -> "list[Locator]":
    """One single-phrase ``Locator`` per camera owner the pick ``service`` holds, the primary first, every one over the
    perception backend the service's camera source was built with.

    So a task locates its target with the cell's own detector on the cell's own cameras: no second copy of a model
    loads (``PerceptionSpec.build`` is never called) and no camera is opened again. The owners are the ones
    ``lifecycle.service_cameras`` reads; the backend is the primary source's (``RealSenseVisionPerceptionSource
    .backend``), which the cell shares across its cameras. A wrist camera's locator reads the arm's TCP at each shutter
    (``arm.get_tcp_pose``) and is held to the cell's tool frame: ``tool_frame`` where given, else the one the arm's tree
    names (``robot.gripper.tool_frame``). ``attempts`` is how many more frames a wrist camera takes while the tool moved:
    the tree's ``fresh_frame_attempts`` where the arm keeps one. ``keep_image`` keeps the colour image of each prompt's
    last sighting (:func:`located_image`), for the target's overlay.

    Refused with ``LocatorRefused`` before anything is located: a cell with no camera owner (the rehearsal scene, a
    service built around doubles), a camera source that carries no backend to lend, and every refusal of
    ``Locator.from_parts`` (a rig with no depth or no calibration, a wrist rig with no TCP reader or no tool frame).
    """
    from src.robot.execution.lifecycle import service_cameras  # noqa: PLC0415
    from src.robot.perception.locator import Locator, LocatorRefused  # noqa: PLC0415

    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    cameras = service_cameras(service)
    if not cameras:
        raise LocatorRefused(
            "this cell's pick service holds no camera owner (the rehearsal scene, or a service built around doubles), "
            "so there is no camera of the cell to find a target with")
    backend = getattr(getattr(orchestrator, "perception", None), "backend", None)
    if backend is None:
        raise LocatorRefused(
            "this cell's camera source carries no perception backend to lend, so a target would need a second copy "
            "of the detector; nothing was located")
    arm = getattr(orchestrator, "arm", None)
    reader = getattr(arm, "get_tcp_pose", None)
    tree = getattr(arm, "config", None)
    if not chosen(tool_frame):
        named = getattr(getattr(tree, "gripper", None), "tool_frame", None)
        tool_frame = named if named is not None else UNSET
    if not chosen(attempts):
        fresh = getattr(getattr(getattr(getattr(tree, "safety", None), "planning_world", None), "perceived", None),
                        "fresh_frame_attempts", None)
        attempts = int(fresh) if isinstance(fresh, int) and not isinstance(fresh, bool) else UNSET
    locators: list[Locator] = []
    for camera in cameras:
        sightings = _Sightings(str(camera.rig_id), keep=keep_image)
        keywords: dict[str, Any] = {"camera": camera, "backend": backend, "view": sightings}
        if callable(reader):
            keywords["tool_pose"] = reader
        if chosen(tool_frame):
            keywords["tool_frame"] = tool_frame
        if chosen(attempts):
            keywords["attempts"] = attempts
        locator = Locator.from_parts(**keywords)
        _SIGHTINGS[locator] = sightings
        locators.append(locator)
    return locators


def _sightings_for(locator: Any) -> "_Sightings | None":
    """What :func:`locators_for_service` keeps for ``locator``; ``None`` for a locator it did not build, one that
    cannot be weakly referenced included."""
    try:
        return _SIGHTINGS.get(locator)
    except TypeError:
        return None


def located_image(locator: Any, prompt: str) -> "np.ndarray | None":
    """The colour image (BGR) of the last frame ``locator`` located ``prompt`` in, where it keeps one: a locator
    :func:`locators_for_service` built with ``keep_image``; ``None`` otherwise."""
    sightings = _sightings_for(locator)
    if sightings is None or sightings.images is None:
        return None
    return sightings.images.get(str(prompt))


def _rig_of(locator: Any) -> str:
    """Which rig a locator locates with: as :func:`locators_for_service` built it, else as it says."""
    sightings = _sightings_for(locator)
    if sightings is not None:
        return sightings.rig_id
    said = getattr(locator, "rig_id", None)
    if isinstance(said, str):
        return said
    return str(getattr(getattr(locator, "_camera", None), "rig_id", ""))


def _target_png(image: "np.ndarray | None", located: "Located", obj: "LocatedObject") -> "bytes | None":
    """The sighting drawn over (``draw_located``, the kept target alone), as PNG; ``None`` with no image or where the
    drawing fails: display only, never a reason to stop."""
    if image is None:
        return None
    try:
        import cv2  # noqa: PLC0415

        from src.camera.live_view import draw_located  # noqa: PLC0415

        drawn = draw_located(image, replace(located, objects=(obj,)))
        ok, encoded = cv2.imencode(".png", drawn)
        return bytes(encoded.tobytes()) if ok else None
    except Exception as exc:  # noqa: BLE001 (an overlay is never a reason to fail a task)
        logger.debug("place target: the target's overlay was not drawn: %s: %s", type(exc).__name__, exc)
        return None


def _gap_mm(read: Any) -> np.ndarray:
    """How much farther than it should each point of a ``Measured`` read, mm, negative where nearer; NaN outside the
    frame and where nothing was measured at its pixel."""
    expected = np.asarray(read.expected_mm, dtype=np.float64)
    measured = np.asarray(read.measured_mm, dtype=np.float64)
    return np.where(np.asarray(read.in_frame, dtype=bool), measured - expected, np.nan)


def _rim_colour(located: "Located", kept: KeptTarget, image: "np.ndarray | None") -> "tuple[float, float, float] | None":
    """The median CIE L*a*b* colour of ``kept``'s rim band in the frame it was located in (``image``, BGR), over the
    points that read there where they were placed; ``None`` without the frame or its image, or with fewer than
    :data:`MIN_TARGET_POINTS` such points. A colour nobody could read leaves the quick look at the rim unsure, so the
    detector looks: never a reason to stop."""
    if image is None:
        return None
    measure = getattr(located, "measure", None)
    if not callable(measure):
        return None
    try:
        read = measure(kept.rim_band_mm, image)
        if read is None:
            return None
        gap = _gap_mm(read)
        lab = np.asarray(read.lab, dtype=np.float64)
        on_the_rim = (np.abs(gap) <= DEPTH_CHECK_TOLERANCE_MM) & np.isfinite(lab).all(axis=1)
        if int(np.count_nonzero(on_the_rim)) < MIN_TARGET_POINTS:
            return None
        median = np.median(lab[on_the_rim], axis=0)
    except Exception as exc:  # noqa: BLE001 (a colour is a shortcut past the detector, never a reason to fail a task)
        logger.warning("place target: the colour of the %s's rim was not read (%s: %s); the detector checks it",
                       kept.label, type(exc).__name__, exc)
        return None
    return float(median[0]), float(median[1]), float(median[2])


def _sightings_of(locator: Any, phrase: str, *, look: "LookPose | None",
                  together: "Sequence[str]" = ()) -> "list[KeptTarget]":
    """Every target ``locator`` locates for ``phrase`` now, kept as seen from ``look``, each with its rim's colour in
    that frame where the locator kept its image; ``together`` the other places grounded in the same locate
    (:func:`_located_sightings`)."""
    return _located_sightings(locator, phrase, look=look, together=together)[1]


def _located_sightings(locator: Any, phrase: str, *, look: "LookPose | None",
                       together: "Sequence[str]" = ()) -> "tuple[Located, list[KeptTarget]]":
    """:func:`_sightings_of`, and the ``Located`` they were placed from, whose frame a part's place is read on too.

    ``together`` are the phrases of a sort's other places (the owner, 2026-10-09): one locate grounds them all, one
    class list (``class_list_prompt``), and only the objects labelled ``phrase`` are its sightings (``Located.split``),
    so a second bin in view is never taken for this one, as a locate of one phrase would read every box as that phrase.
    None is the locate of ``phrase`` alone, as it always was."""
    prompt = phrase
    if together:
        # Here, not at the module's top: a place of one phrase loads no model package.
        from src.models.vlm.qwen import class_list_prompt  # noqa: PLC0415

        prompt = class_list_prompt([phrase, *together])
    located = locator.locate(prompt)
    image = located_image(locator, prompt)
    objects = tuple(located.objects)
    if prompt != phrase:
        own = next((indices for description, indices in located.split(prompt).items()
                    if _said_as(description) == _said_as(phrase)), ())
        objects = tuple(located.objects[index] for index in own)
    found: list[KeptTarget] = []
    for obj in objects:
        kept = KeptTarget.of(obj, camera=located.camera, look=look, seen_at=located.captured_at_s, phrase=phrase)
        if kept is None:
            continue
        if image is not None:
            kept = replace(kept, image_png=_target_png(image, located, obj), colour_lab=_rim_colour(located, kept, image))
        found.append(kept)
    return located, found


# ---------------------------------------------------------------------------------------------------------------------
# Before the first pick: the survey
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Survey:
    """What a task's survey came to: the target kept, or none, and where it looked.

    ``kept`` is the target (``None`` where no look saw one with enough points); ``looks_tried`` the looks it located from,
    in order (the known target's look first, where it was looked at again); ``parts_seen`` how many parts of the
    object's phrase the look that kept the target saw outside it (``None`` for an empty phrase); ``refused`` the looks
    the planner refused before anything was sent, skipped, each with why. ``stopped`` is the look motion that ended the
    survey with nothing else commanded (``stopped_look`` its look), and ``halted`` why the task may not move any more,
    where that ended it before a motion. ``known`` is the check of the target an earlier task kept, where one was handed
    in and looked at again (``known.followed``: it was kept, and no other look ran), ``None`` otherwise.
    """

    kept: "KeptTarget | None"
    looks_tried: tuple[str, ...] = ()
    parts_seen: "int | None" = None
    refused: tuple[tuple[str, str], ...] = ()
    stopped: "MotionReport | None" = None
    stopped_look: str = ""
    halted: str = ""
    known: "TargetCheck | None" = None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        tried = ", ".join(self.looks_tried) or "no look"
        if self.kept is not None:
            head = f"kept {self.kept.render()}"
        elif self.halted:
            head = f"stopped before a motion: {self.halted}"
        elif self.stopped is not None:
            head = f"look {self.stopped_look} not reached: {self.stopped.status.value}: {self.stopped.message}"
        else:
            head = "no target seen"
        lines = [head, f"  located from {tried}"]
        if self.known is not None:
            lines.append(f"  the target kept before, looked at again: {self.known.render()}")
        lines.extend(f"  skipped look {look}: {why}" for look, why in self.refused)
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)


def _parts_seen(locators: "Sequence[Any]", object_phrase: str, kept: "KeptTarget | None") -> "int | None":
    """How many parts of ``object_phrase`` the locators see, outside ``kept``'s footprint where a target was kept there;
    ``None`` for no phrase."""
    if not str(object_phrase).strip():
        return None
    region = kept.keep_out_region() if kept is not None else None
    count = 0
    for locator in locators:
        for obj in locator.locate(object_phrase).objects:
            centre = getattr(obj, "centre_mm", None)
            if centre is not None and (region is None or not region.contains(centre)):
                count += 1
    return count


def _known_at(known: "KeptTarget | None", phrase: str, locators: "Sequence[Any]", wrist: "Sequence[Any]",
              looks: "Sequence[LookPose]") -> "tuple[str, LookPose | None] | None":
    """Where the target an earlier task kept is looked at again before a survey: its look as the task names it (the
    look's label and the task's own look), ``("", None)`` for a fixed camera's, which looks where it stands; ``None``
    where it is not used, said in the log: none was handed in, it was located for another phrase, by a camera the task
    does not hold, or from a look the task does not look from."""
    from src.robot.execution.looks import look_label  # noqa: PLC0415

    if known is None:
        return None
    if not isinstance(known, KeptTarget):
        raise TypeError(f"a known target is the KeptTarget an earlier task kept, not {type(known).__name__}")
    said = (known.phrase or known.label).strip()
    if said != str(phrase).strip():
        logger.info("survey: the target kept before was located for %r, not %r; it is not used", said, phrase)
        return None
    if known.camera not in {_rig_of(locator) for locator in (wrist or locators)}:
        logger.info("survey: the %r kept before was seen by camera %r, which this task does not look with; it is not "
                    "used", phrase, known.camera)
        return None
    if not wrist:
        if known.look is not None:
            logger.info("survey: the %r kept before was seen from a look, and this task's camera stands fixed; it is "
                        "not used", phrase)
            return None
        return "", None
    if known.look is None:
        logger.info("survey: the %r kept before was seen by a fixed camera, and this task's rides on the wrist; it is "
                    "not used", phrase)
        return None
    label = look_label(known.look)
    for pose in looks:
        if look_label(pose) == label:
            return label, pose
    logger.info("survey: the %r kept before was seen from look %s, which this task does not look from; it is not used",
                phrase, label)
    return None


def survey(arm: Any, locators: "Sequence[Any]", phrase: str, *, object_phrase: str = "",
           looks: "Sequence[LookPose]" = (), may_move: "Callable[[], str] | None" = None,
           known: "KeptTarget | None" = None) -> Survey:
    """Find the target ``phrase`` names before a task's first pick, from the task's ``looks``.

    A camera on the wrist (the locators that say ``on_the_wrist``) sees what the arm points it at: the arm goes to each
    look in turn through its own judged verb (``move_to_look``), the camera locates ``phrase`` there, then
    ``object_phrase`` where one is given (which only counts the parts; a task gives none), and the survey ends at the
    first look whose target has at least :data:`MIN_TARGET_POINTS` points, keeping the most confident of several (Q5);
    with no look it locates where the arm stands. A look the planner refused before anything was sent is skipped and
    said; a look motion that may have moved the arm, or a controller that stopped, ends the survey there with nothing
    else commanded (``stopped``), and so does ``may_move`` answering why the task may not move any more, asked before
    every motion (``halted``). A fixed camera locates once where it stands and moves nothing. A camera that could not
    vouch on the way raises, as a pick's does; so does a locate that raises.

    ``known`` is the target an earlier task kept (its ``TaskReport.kept_target``). Located for ``phrase`` by a camera of
    ``locators``, from one of ``looks`` (a fixed camera's: where it stands), it is looked at again first, from that look,
    as before a drop (:func:`recheck`, against it): followed there, it is kept and no other look runs, so a bin that stood
    still costs one quick look at its rim. Anywhere else, or not followed, the survey runs as it does without it, from
    its first look. ``Survey.known`` is that check.
    """
    from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
    from src.robot.execution.looks import look_label, move_to_look  # noqa: PLC0415
    from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

    wrist = [locator for locator in locators if bool(getattr(locator, "on_the_wrist", False))]
    usable = _known_at(known, phrase, locators, wrist, looks)
    if not wrist:
        check = recheck(locators, known) if usable is not None and known is not None else None
        if check is not None and check.kept is not None:
            logger.info("survey: the %r kept before stands where it stood (%s): %s", phrase, check.by,
                        check.kept.render())
            return Survey(kept=check.kept, parts_seen=_parts_seen(locators, object_phrase, check.kept), known=check)
        found = [target for locator in locators for target in _sightings_of(locator, phrase, look=None)]
        kept = _most_confident(found)
        parts = _parts_seen(locators, object_phrase, kept) if kept is not None else None
        return Survey(kept=kept, parts_seen=parts, known=check)
    tried: list[str] = []
    refused: list[tuple[str, str]] = []
    check = None
    if usable is not None and known is not None:
        label, pose = usable
        why = may_move() if may_move is not None else ""
        if why:
            return Survey(kept=None, halted=why)
        assert pose is not None  # a wrist camera's known target was seen from a look of the task's
        moved = move_to_look(arm, pose)
        if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                moved.result.exception, CameraWorldUnavailable):
            raise moved.result.exception
        if moved.ok:
            tried.append(label)
            check = recheck(wrist, known)
            if check.kept is not None:
                kept = replace(check.kept, look=pose, look_label=label)
                logger.info("survey: the %r kept before stands where it stood at look %s (%s): %s", phrase, label,
                            check.by, kept.render())
                return Survey(kept=kept, looks_tried=tuple(tried), parts_seen=_parts_seen(wrist, object_phrase, kept),
                              known=check)
            logger.info("survey: the %r kept before was not followed at look %s (%s); every look is surveyed", phrase,
                        label, check.render())
        elif refused_before_sending(moved):
            logger.warning("survey: look %s, where the %r kept before was seen, was refused before anything was sent "
                           "(%s: %s); every look is surveyed", label, phrase, moved.status.value, moved.message)
        else:
            return Survey(kept=None, stopped=moved, stopped_look=label)
    plan: list[tuple[str, Any]] = [(look_label(pose), pose) for pose in looks] if looks else [("here", None)]
    for label, pose in plan:
        if pose is not None:
            why = may_move() if may_move is not None else ""
            if why:
                return Survey(kept=None, looks_tried=tuple(tried), refused=tuple(refused), halted=why, known=check)
            moved = move_to_look(arm, pose)
            if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                    moved.result.exception, CameraWorldUnavailable):
                raise moved.result.exception
            if not moved.ok:
                if refused_before_sending(moved):
                    refused.append((label, f"{moved.status.value}: {moved.message}"))
                    logger.warning("survey: look %s was refused before anything was sent (%s: %s); it is skipped",
                                   label, moved.status.value, moved.message)
                    continue
                return Survey(kept=None, looks_tried=tuple(tried), refused=tuple(refused), stopped=moved,
                              stopped_look=label, known=check)
        tried.append(label)
        found = [target for locator in wrist for target in _sightings_of(locator, phrase, look=pose)]
        kept = _most_confident(found)
        if kept is not None and pose is None:
            kept = replace(kept, look_label=label)
        # The part's phrase at every look after the target's, where a program gave one: it counts what this look sees.
        parts = _parts_seen(wrist, object_phrase, kept)
        if kept is not None:
            logger.info("survey: %s", kept.render())
            return Survey(kept=kept, looks_tried=tuple(tried), parts_seen=parts, refused=tuple(refused), known=check)
        if parts is not None:
            logger.info("survey: look %s sees %d part(s) of %r and no %r", label, parts, object_phrase, phrase)
    logger.info("survey: no %r was seen from %s", phrase, ", ".join(tried) or "any look")
    return Survey(kept=None, looks_tried=tuple(tried), refused=tuple(refused), known=check)


# ---------------------------------------------------------------------------------------------------------------------
# Before the first pick of a sort: every place
# ---------------------------------------------------------------------------------------------------------------------

#: What separates the phrases of one locate's class list: the detector's own class-list prompt (the sorting map's S1,
#: ``class_list_prompt`` writes "yellow bin | blue bin", ``classes_of`` reads it back into the labels the locator maps
#: each box onto). One phrase is the phrase itself, located as it always was.
CLASS_LIST_SEPARATOR = " | "


def _said_as(text: Any) -> str:
    """How a place's phrase and a located object's label are compared: trimmed, each run of blanks one space, case
    folded, and nothing else. A label of other words names no place: a box two phrases claimed (``ambiguous``), a part
    whose pixels said another colour ("blue object, not yellow bin"), a word of the detector's own."""
    return " ".join(str(text or "").split()).casefold()


def _places_of(phrases: "str | Iterable[str]") -> tuple[str, ...]:
    """The places ``phrases`` name, in order, each trimmed, a phrase said twice one place. None at all, an empty phrase
    and one that holds the class list's separator (which would split it in two) are a programmer's error: raised."""
    bar = CLASS_LIST_SEPARATOR.strip()
    places: list[str] = []
    for phrase in [phrases] if isinstance(phrases, str) else list(phrases):
        text = str(phrase).strip()
        if not text:
            raise ValueError("a place names what to find, and an empty phrase names nothing")
        if bar in text:
            raise ValueError(f"a place's phrase may not hold {bar!r}, which separates the phrases of one locate: "
                             f"{text!r}")
        if all(_said_as(text) != _said_as(place) for place in places):
            places.append(text)
    if not places:
        raise ValueError("a survey of places is handed at least one place to find")
    return tuple(places)


def _known_of(known: Any) -> "list[KeptTarget]":
    """The bins earlier tasks kept, as handed in: one, a mapping's values or the entries, ``None`` left out. An entry
    that is no ``KeptTarget`` is a programmer's error: raised."""
    if known is None:
        return []
    if isinstance(known, KeptTarget):
        return [known]
    entries = list(known.values()) if isinstance(known, Mapping) else list(known)
    for entry in entries:
        if entry is not None and not isinstance(entry, KeptTarget):
            raise TypeError(f"a known place is the KeptTarget an earlier task kept, not {type(entry).__name__}")
    return [entry for entry in entries if entry is not None]


@dataclass(frozen=True)
class PlaceFound:
    """One place of a sort as its survey left it (:func:`survey_places`).

    ``phrase`` is the place as it was handed in (trimmed); ``kept`` the bin kept for it, ``None`` where no look saw one
    with enough points (the task names it and picks nothing); ``looks_tried`` the looks it was looked for from, in order,
    up to the one that kept it; ``known`` the check of the bin an earlier task kept for it, where one was handed in and
    looked at again (``known.followed``: kept with no locate), ``None`` otherwise.
    """

    phrase: str
    kept: "KeptTarget | None"
    looks_tried: tuple[str, ...] = ()
    known: "TargetCheck | None" = None

    @property
    def found(self) -> bool:
        return self.kept is not None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        if self.kept is not None:
            line = f"{self.phrase}: kept {self.kept.render()}"
            if self.known is not None and self.known.followed:
                line += f"; where the last task left it ({self.known.by})"
        else:
            line = (f"{self.phrase}: no {self.phrase} was seen from "
                    f"{', '.join(self.looks_tried) or 'where the camera stands'}")
        return line.encode("ascii", "backslashreplace").decode("ascii")


@dataclass(frozen=True)
class PlacesSurvey:
    """What the survey of a sort's places came to (:func:`survey_places`): every place, kept or missing, and where it
    looked.

    ``places`` is one :class:`PlaceFound` per place, in the order handed in; ``looks_tried`` the looks it located from,
    in order (the look of each bin an earlier task kept first, where one was looked at again); ``refused`` the looks the
    planner refused before anything was sent, skipped, each with why. ``stopped`` is the look motion that ended the
    survey with nothing else commanded (``stopped_look`` its look), and ``halted`` why the task may not move any more,
    where that ended it before a motion: either way what was kept before stands, and every other place is missing.
    """

    places: tuple[PlaceFound, ...]
    looks_tried: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    stopped: "MotionReport | None" = None
    stopped_look: str = ""
    halted: str = ""

    def place(self, phrase: str) -> PlaceFound:
        """The place ``phrase`` names, as it was handed in (case and blanks aside); ``KeyError`` for one this survey
        was not handed."""
        for place in self.places:
            if _said_as(place.phrase) == _said_as(phrase):
                return place
        raise KeyError(f"this survey looked for no place {phrase!r}, only for "
                       f"{', '.join(repr(place.phrase) for place in self.places)}")

    @property
    def missing(self) -> tuple[str, ...]:
        """The places no look saw, in order: the task names each and picks nothing."""
        return tuple(place.phrase for place in self.places if place.kept is None)

    @property
    def complete(self) -> bool:
        """Whether every place was kept."""
        return not self.missing

    def survey_of(self, phrase: str) -> Survey:
        """The place ``phrase`` names as a one-place :class:`Survey`, the shape a task says a bin found or missing in
        (``task.target_found``, ``task.target_missing``): its bin, the looks it was looked for from, the check of the bin
        an earlier task kept for it, and how the survey ended; ``parts_seen`` ``None``, as no part is counted."""
        place = self.place(phrase)
        return Survey(kept=place.kept, looks_tried=place.looks_tried, refused=self.refused, stopped=self.stopped,
                      stopped_look=self.stopped_look, halted=self.halted, known=place.known)

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        if self.halted:
            head = f"stopped before a motion: {self.halted}"
        elif self.stopped is not None:
            head = f"look {self.stopped_look} not reached: {self.stopped.status.value}: {self.stopped.message}"
        elif self.missing:
            head = (f"{len(self.places) - len(self.missing)} of {len(self.places)} place(s) found, missing: "
                    f"{', '.join(self.missing)}")
        else:
            head = f"every place found ({len(self.places)})"
        lines = [head, f"  located from {', '.join(self.looks_tried) or 'no look'}"]
        lines.extend(f"  {place.render()}" for place in self.places)
        lines.extend(f"  skipped look {look}: {why}" for look, why in self.refused)
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)


def _seen_places(locators: "Sequence[Any]", missing: "Sequence[str]", *, look: "LookPose | None",
                 kept: "Mapping[str, KeptTarget]", every: "Sequence[str]" = ()) -> "dict[str, KeptTarget]":
    """What ONE locate per locator sees of the ``missing`` places from where the arm stands, kept as seen from
    ``look``: per place, the most confident sighting with at least :data:`MIN_TARGET_POINTS` points (Q5) that stands in
    the footprint of no place kept already (``kept``, or one kept before it from this frame), with its rim's colour in
    that frame where the locator kept its image.

    The locate grounds the class list of ``every`` place of the survey where it holds two or more, the places kept
    already among them, so a second bin in view is never taken for a missing one, however few are still missing (the
    review of 2026-10-09); else the class list of ``missing``. A located object is a sighting of the place its label is
    (:func:`_said_as`): the locator maps the detector's words onto the class list's phrases (``object_labels``), and
    nothing here maps a label of other words onto a place; one labelled as a place kept already is no sighting of a
    missing one. One phrase is located as :func:`survey` locates it, every object a sighting of it."""
    grounded = list(every) if len(every) > 1 else list(missing)
    prompt = CLASS_LIST_SEPARATOR.join(grounded)
    named = {_said_as(phrase): phrase for phrase in missing}
    others = {_said_as(phrase) for phrase in grounded} - set(named)
    sightings: dict[str, list[KeptTarget]] = {phrase: [] for phrase in missing}
    unnamed: list[str] = []
    for locator in locators:
        located = locator.locate(prompt)
        image = located_image(locator, prompt)
        for obj in located.objects:
            phrase = missing[0] if len(grounded) == 1 else named.get(_said_as(obj.label))
            if phrase is None:
                if _said_as(obj.label) not in others:
                    unnamed.append(str(obj.label))
                continue
            target = KeptTarget.of(obj, camera=located.camera, look=look, seen_at=located.captured_at_s, phrase=phrase)
            if target is None:
                continue
            if image is not None:
                target = replace(target, image_png=_target_png(image, located, obj),
                                 colour_lab=_rim_colour(located, target, image))
            sightings[phrase].append(target)
    if unnamed:
        logger.info("survey: %d object(s) located for %r name no place: %s", len(unnamed), prompt,
                    ", ".join(sorted(set(unnamed))))
    chosen: dict[str, KeptTarget] = {}
    for phrase in missing:
        taken = [*kept.values(), *chosen.values()]
        free: list[KeptTarget] = []
        for target in sightings[phrase]:
            holder = next((other for other in taken if other.keep_out_region().contains(target.centre_xy_mm)), None)
            if holder is not None:
                logger.info("survey: a %r stands at (%.0f, %.0f) mm, in the %s kept already; it is not taken for it",
                            phrase, target.centre_xy_mm[0], target.centre_xy_mm[1], holder.phrase or holder.label)
                continue
            free.append(target)
        best = _most_confident(free)
        if best is not None:
            chosen[phrase] = best
    return chosen


def _besides(places: "Sequence[str]", phrase: str) -> "tuple[str, ...]":
    """The places a survey grounds beside ``phrase`` in one locate: every other one, none for a survey of one place."""
    return tuple(other for other in places if other != phrase)


def survey_places(arm: Any, locators: "Sequence[Any]", phrases: "str | Iterable[str]", *,
                  looks: "Sequence[LookPose]" = (), may_move: "Callable[[], str] | None" = None,
                  known: "KeptTarget | Iterable[KeptTarget | None] | Mapping[Any, KeptTarget | None] | None" = None,
                  ) -> PlacesSurvey:
    """Find every place ``phrases`` names before a sort's first pick, from the task's ``looks`` (the owner, 2026-10-09:
    every rule's bin is found before the first pick, else the task stops naming the bin it is missing).

    Look by look, as :func:`survey` finds one: the arm goes to each look in turn through its own judged verb
    (``move_to_look``), ``may_move`` asked before every motion (``halted``); a look the planner refused before anything
    was sent is skipped and said; a look motion that may have moved the arm, or a controller that stopped, ends the
    survey there with nothing else commanded (``stopped``); a camera that could not vouch on the way raises, and so does
    a locate that raises. A fixed camera locates once where it stands and moves nothing; with no look, the camera on the
    wrist locates where the arm stands.

    At each look, each camera on the wrist locates every place still missing ONCE, their phrases one class list
    ("yellow bin | blue bin", :data:`CLASS_LIST_SEPARATOR`): the locator maps the detector's words onto the list's
    phrases (``object_labels``, the camera source's own mapping), and a located object is a sighting of the place its
    label names, exactly (trimmed, case aside); any other label names none. Every locate of a survey of two or more
    places grounds every place, the ones kept already too, so a second bin in view is never taken for a missing one;
    one place is located as :func:`survey` locates it, every object a sighting of it. Of a place's sightings with at least
    :data:`MIN_TARGET_POINTS` points the most confident is kept (Q5), unless it stands in the footprint of a place kept
    already (one bin is never two places); the first look that sees a place keeps it, and the survey ends once every
    place is kept. A place no look saw is missing, and named (:attr:`PlacesSurvey.missing`).

    ``known`` are the bins earlier tasks kept (``TaskReport.kept_target``, the console's per phrase): each is looked at
    again first, from its own look, as :func:`survey` looks at one (:func:`recheck`, against it, every other place
    grounded beside it where the detector looks), and one followed there
    is kept with no locate; one of no place, of a camera the task does not hold or from a look it does not look from is
    not used. One place is surveyed exactly as :func:`survey` surveys it (with no ``object_phrase``).
    """
    from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
    from src.robot.execution.looks import look_label, move_to_look  # noqa: PLC0415
    from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

    places = _places_of(phrases)
    before = _known_of(known)
    wrist = [locator for locator in locators if bool(getattr(locator, "on_the_wrist", False))]
    usable: dict[str, tuple[KeptTarget, tuple[str, LookPose | None]]] = {}
    for phrase in places:
        candidate = next((kept for kept in before if _said_as(kept.phrase or kept.label) == _said_as(phrase)), None)
        at = _known_at(candidate, phrase, locators, wrist, looks)
        if candidate is not None and at is not None:
            usable[phrase] = (candidate, at)
    for kept in before:
        if all(_said_as(kept.phrase or kept.label) != _said_as(phrase) for phrase in places):
            logger.info("survey: the %r kept before is no place of this task; it is not used", kept.phrase or kept.label)
    kept_for: dict[str, KeptTarget] = {}
    checks: dict[str, TargetCheck] = {}
    tried: list[str] = []
    tried_for: dict[str, list[str]] = {phrase: [] for phrase in places}
    refused: list[tuple[str, str]] = []

    def answer(**ended: Any) -> PlacesSurvey:
        found = tuple(PlaceFound(phrase=phrase, kept=kept_for.get(phrase), looks_tried=tuple(tried_for[phrase]),
                                 known=checks.get(phrase)) for phrase in places)
        for place in found:
            if place.kept is None and not ended:
                logger.info("survey: no %r was seen from %s", place.phrase, ", ".join(place.looks_tried) or "any look")
        return PlacesSurvey(places=found, looks_tried=tuple(tried), refused=tuple(refused), **ended)

    def missing() -> list[str]:
        return [phrase for phrase in places if phrase not in kept_for]

    if not wrist:
        for phrase, (candidate, _at) in usable.items():
            check = checks[phrase] = recheck(locators, candidate, together=_besides(places, phrase))
            if check.kept is not None:
                logger.info("survey: the %r kept before stands where it stood (%s): %s", phrase, check.by,
                            check.kept.render())
                kept_for[phrase] = check.kept
        if missing():
            kept_for.update(_seen_places(locators, missing(), look=None, kept=kept_for, every=places))
        return answer()
    for phrase, (candidate, (label, pose)) in usable.items():
        why = may_move() if may_move is not None else ""
        if why:
            return answer(halted=why)
        assert pose is not None  # a wrist camera's known target was seen from a look of the task's
        moved = move_to_look(arm, pose)
        if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                moved.result.exception, CameraWorldUnavailable):
            raise moved.result.exception
        if moved.ok:
            tried.append(label)
            tried_for[phrase].append(label)
            check = checks[phrase] = recheck(wrist, candidate, together=_besides(places, phrase))
            if check.kept is not None:
                kept_for[phrase] = replace(check.kept, look=pose, look_label=label)
                logger.info("survey: the %r kept before stands where it stood at look %s (%s): %s", phrase, label,
                            check.by, kept_for[phrase].render())
                continue
            logger.info("survey: the %r kept before was not followed at look %s (%s); it is surveyed from every look",
                        phrase, label, check.render())
        elif refused_before_sending(moved):
            logger.warning("survey: look %s, where the %r kept before was seen, was refused before anything was sent "
                           "(%s: %s); it is surveyed from every look", label, phrase, moved.status.value, moved.message)
        else:
            return answer(stopped=moved, stopped_look=label)
    plan: list[tuple[str, Any]] = [(look_label(pose), pose) for pose in looks] if looks else [("here", None)]
    for label, pose in plan:
        if not missing():
            break
        if pose is not None:
            why = may_move() if may_move is not None else ""
            if why:
                return answer(halted=why)
            moved = move_to_look(arm, pose)
            if moved.outcome is MotionOutcome.CAMERA_WORLD_UNAVAILABLE and isinstance(
                    moved.result.exception, CameraWorldUnavailable):
                raise moved.result.exception
            if not moved.ok:
                if refused_before_sending(moved):
                    refused.append((label, f"{moved.status.value}: {moved.message}"))
                    logger.warning("survey: look %s was refused before anything was sent (%s: %s); it is skipped",
                                   label, moved.status.value, moved.message)
                    continue
                return answer(stopped=moved, stopped_look=label)
        tried.append(label)
        looked_for = missing()
        for phrase in looked_for:
            tried_for[phrase].append(label)
        for phrase, target in _seen_places(wrist, looked_for, look=pose, kept=kept_for, every=places).items():
            kept_for[phrase] = replace(target, look_label=label) if pose is None else target
            logger.info("survey: %s", kept_for[phrase].render())
    return answer()


# ---------------------------------------------------------------------------------------------------------------------
# What a frame reads where the part goes
# ---------------------------------------------------------------------------------------------------------------------

#: How far a point of a target's surface may read from where it should and still be that surface, mm, in the camera's
#: own depth: a point that reads nearer by more stands under something new, one that reads farther by more has nothing
#: at it. A choice, not a measurement.
SURFACE_TOLERANCE_MM = 5.0
#: How far apart the heights a box's inside is read at stand, mm, and how far within its walls' inner edges it is read.
#: Choices, not measurements.
_INSIDE_STEP_MM = 5.0
_INSIDE_INSET_MM = 5.0
#: A box's inside is read at the highest point of what stood in each cell of this size when it was last seen, mm, at
#: most :data:`_INSIDE_COLUMNS` of them; with fewer than :data:`MIN_TARGET_POINTS` its inside is not read.
_INSIDE_CELL_MM = 10.0
_INSIDE_COLUMNS = 400
#: How far apart, mm, the points of one such cell may stand and still be one surface rather than a face.
_INSIDE_FACE_MM = 10.0
#: How many points a cell holds at least, and at least what share of the cells' median, before it is a place of the
#: inside rather than a few stray readings.
_INSIDE_CELL_POINTS = 5
_INSIDE_SPARSE_SHARE = 0.2


@dataclass(frozen=True, eq=False)
class InsideRead:
    """What one frame read where a part may be set down on a target, BASE millimetres.

    ``kind`` is ``box`` for a target with an inside: ``top_mm`` is the highest the frame reads anything in it, its floor
    or the parts already in it, read at ``points_mm``, one place for each 10 mm cell of what the target showed of its
    inside when it was last seen (the highest point of that cell). Each place is read upward from where it stood then,
    every :data:`_INSIDE_STEP_MM`: the lowest height from which the frame twice running reads nothing there (its ray goes
    on past by more than :data:`SURFACE_TOLERANCE_MM`) is the top of what stands there, and a place the frame never reads
    clear up to the rim (hidden, or no depth at all) is full to the rim. So every error goes up: a part with no depth, one
    behind another, a reading short of the truth each raise the top, and none lowers it.

    ``kind`` is ``top`` for a target with a flat top: ``points_mm`` are its top's points (its rim band), ``seen`` whether
    the frame read each and ``covered`` whether it reads nearer than it should (something stands on it now). ``said``
    is what was read, for the log.
    """

    kind: str
    points_mm: np.ndarray = field(repr=False)
    top_mm: "float | None" = None
    covered: "np.ndarray | None" = field(default=None, repr=False)
    seen: "np.ndarray | None" = field(default=None, repr=False)
    said: str = ""


@dataclass(frozen=True, eq=False)
class _Probes:
    """The points a frame is asked at where a part may go on a target (:func:`_probes_of`): for a box, ``columns``, the
    places of its inside, each at ``heights`` steps of :data:`_INSIDE_STEP_MM` up from where it stood, one column after
    the other in ``points``; for a flat top, its points, each once."""

    kind: str
    points: np.ndarray
    columns: np.ndarray
    heights: int
    rim_mm: float


def _inside_columns(target: KeptTarget) -> np.ndarray:
    """The places of a box's inside ``target`` showed when it was seen, ``(C, 3)`` BASE mm: the highest of what stood
    in each :data:`_INSIDE_CELL_MM` cell within its walls' inner edges by :data:`_INSIDE_INSET_MM` and below its rim
    band. A cell whose points stand more than :data:`_INSIDE_FACE_MM` apart holds a face, a wall's inside or a part's
    side, which would read as something standing to the rim, and a cell of a few stray readings (a pixel flying off a
    wall's edge, behind it) would read the wall in front of it all the way up: both are left out. Empty for a target
    with no inside."""
    edges = target.inner_opening()
    if edges is None:
        return np.zeros((0, 3), dtype=np.float64)
    points = target.points_base_mm
    along, across = _turned(points[:, :2], target.yaw_rad)
    (a_low, a_high), (c_low, c_high) = edges
    inset = _INSIDE_INSET_MM
    inside = ((points[:, 2] < target.rim_mm - _RIM_BAND_MM) & (along >= a_low + inset) & (along <= a_high - inset)
              & (across >= c_low + inset) & (across <= c_high - inset))
    candidates = points[inside]
    if candidates.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float64)
    cells = np.floor(candidates[:, :2] / _INSIDE_CELL_MM).astype(np.int64)
    order = np.lexsort((-candidates[:, 2], cells[:, 1], cells[:, 0]))
    cells, candidates = cells[order], candidates[order]
    first = np.ones(cells.shape[0], dtype=bool)
    first[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    last = np.roll(first, -1)
    spread = candidates[first, 2] - candidates[last, 2]
    counts = np.diff(np.append(np.flatnonzero(first), cells.shape[0]))
    least = max(float(_INSIDE_CELL_POINTS), _INSIDE_SPARSE_SHARE * float(np.median(counts)))
    return candidates[first][(spread <= _INSIDE_FACE_MM) & (counts >= least)]


def _moved_with(columns: np.ndarray, then: KeptTarget, now: KeptTarget) -> np.ndarray:
    """``columns`` of a box as it stood when ``then`` was seen, carried along with it to where it stands as ``now``:
    whatever stood in it moved with it. Its turn is read within a quarter turn, as a rectangle's is."""
    if columns.shape[0] == 0:
        return columns
    turn = math.remainder(float(now.yaw_rad) - float(then.yaw_rad), math.pi / 2.0)
    c, s = math.cos(turn), math.sin(turn)
    xy = columns[:, :2] - np.asarray(then.centre_xy_mm, dtype=np.float64)
    moved = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    return np.column_stack([moved + np.asarray(now.centre_xy_mm, dtype=np.float64), columns[:, 2]])


def _probes_of(target: KeptTarget, *, also: "Sequence[np.ndarray]" = ()) -> "_Probes | None":
    """Where a frame reads what stands where a part may go on ``target``: a flat top's points (its rim band), or a box's
    inside at the places it and earlier sightings of it showed (:func:`_inside_columns`; ``also``, each carried to where
    it stands now), the highest of them in each cell, at most :data:`_INSIDE_COLUMNS`, each read from where it stood up
    to the rim; ``None`` where there is too little of either. An earlier sighting reads where a later one shows none: a
    part the detector's mask of the box left out leaves a hole there, which is where a part stands."""
    if target.opening_mm is None:
        band = target.rim_band_mm
        if band.shape[0] < MIN_TARGET_POINTS:
            return None
        return _Probes("top", band, band, 1, target.rim_mm)
    every = np.vstack([_inside_columns(target), *[np.asarray(more, dtype=np.float64).reshape(-1, 3) for more in also]])
    if every.shape[0] < MIN_TARGET_POINTS:
        return None
    cells = np.floor(every[:, :2] / _INSIDE_CELL_MM).astype(np.int64)
    order = np.lexsort((-every[:, 2], cells[:, 1], cells[:, 0]))
    cells, every = cells[order], every[order]
    first = np.ones(cells.shape[0], dtype=bool)
    first[1:] = np.any(cells[1:] != cells[:-1], axis=1)
    columns = every[first]
    if columns.shape[0] < MIN_TARGET_POINTS:
        return None
    if columns.shape[0] > _INSIDE_COLUMNS:
        columns = columns[np.linspace(0, columns.shape[0] - 1, _INSIDE_COLUMNS).round().astype(np.int64)]
    lowest = float(np.min(columns[:, 2]))
    heights = int(math.ceil(max(0.0, target.rim_mm - lowest) / _INSIDE_STEP_MM)) + 1
    probes = np.repeat(columns, heights, axis=0)
    probes[:, 2] += np.tile(np.arange(heights, dtype=np.float64) * _INSIDE_STEP_MM, columns.shape[0])
    return _Probes("box", probes, columns, heights, float(target.rim_mm))


def _earlier(now: KeptTarget, *before: "KeptTarget | None") -> "list[np.ndarray]":
    """The places of the inside the sightings ``before`` showed of the box ``now`` is, carried to where it stands now."""
    return [_moved_with(_inside_columns(then), then, now) for then in before
            if then is not None and then is not now and then.opening_mm is not None]


def _inside_of(probes: "_Probes | None", gap: "np.ndarray | None") -> "InsideRead | None":
    """What one frame read at ``probes``, ``gap`` how much farther than it should each read, mm (NaN where nothing was
    read: :func:`_gap_mm`), says stands where a part may go (:class:`InsideRead`); ``None`` with no probes or no
    reading of them."""
    if probes is None or gap is None or np.shape(gap) != (probes.points.shape[0],):
        return None
    gap = np.asarray(gap, dtype=np.float64)
    if probes.kind == "top":
        seen = np.isfinite(gap)
        covered = seen & (gap < -SURFACE_TOLERANCE_MM)
        said = (f"{int(np.count_nonzero(seen))} of its top's {gap.shape[0]} point(s) read, "
                f"{int(np.count_nonzero(covered))} under something that was not there")
        return InsideRead("top", probes.points, covered=covered, seen=seen, said=said)
    clear = (np.isfinite(gap) & (gap > SURFACE_TOLERANCE_MM)).reshape(probes.columns.shape[0], probes.heights)
    twice = clear.copy()
    twice[:, :-1] &= clear[:, 1:]
    found = twice.any(axis=1)
    first = np.argmax(twice, axis=1)
    tops = np.where(found, probes.columns[:, 2] + first * _INSIDE_STEP_MM, probes.rim_mm)
    top = float(np.max(tops))
    higher = int(np.count_nonzero(~found | (first > 2)))
    said = (f"its inside read at {probes.columns.shape[0]} place(s): the highest {top:.1f} mm, "
            f"{probes.rim_mm - top:.1f} mm under its rim; {higher} place(s) hold more than when it was last seen")
    return InsideRead("box", probes.columns, top_mm=top, said=said)


def _inside_on(located: Any, target: KeptTarget, *, also: "Sequence[np.ndarray]" = ()) -> "InsideRead | None":
    """What the frame ``located`` was placed from reads where a part may go on ``target`` (``Located.measure``, no new
    frame), at the places earlier sightings showed too (``also``); ``None`` where it keeps no frame or there is too
    little to read."""
    probes = _probes_of(target, also=also)
    measure = getattr(located, "measure", None)
    if probes is None or not callable(measure):
        return None
    read = measure(probes.points)
    return None if read is None else _inside_of(probes, _gap_mm(read))


# ---------------------------------------------------------------------------------------------------------------------
# What the pick's own looks saw
# ---------------------------------------------------------------------------------------------------------------------

#: Every how many pixels of a look's frame are read along each axis where a task reads what the pick's looks saw of a
#: place or of the way to it: the camera world's detection stride (``support_surfaces.DETECTION_STRIDE``).
_VIEW_STRIDE = 4


def _view_depth(view: Any) -> "np.ndarray | None":
    """The depth a pick's look measured, CAMERA mm: its measured surface where it kept one, else its frame's depth."""
    depth = getattr(view, "depth", None)
    if depth is None:
        depth = getattr(getattr(view, "frame", None), "depth_map", None)
    return None if depth is None else np.asarray(depth, dtype=np.float64)


def _view_placed(view: Any) -> bool:
    """Whether a pick's look kept what reading it again needs: its frame, its depth and where its camera stood."""
    return (getattr(view, "frame", None) is not None and getattr(view, "camera_to_base", None) is not None
            and _view_depth(view) is not None)


class _LookReader:
    """One of a pick's looks (``LookedAround.views``) as a camera that reads its frame again where points should stand,
    as ``Locator.measure`` reads a new frame: no detector, no new frame, nothing moved."""

    def __init__(self, view: Any) -> None:
        self.view = view
        self.rig_id = str(getattr(view, "name", "") or "").split("@")[0]

    def measure(self, points_base_mm: Any) -> Any:
        from src.robot.perception.locator import _measured  # noqa: PLC0415 (the locator's own reading of a frame)

        if not _view_placed(self.view):
            return None
        frame = self.view.frame
        stamp = getattr(frame, "timestamp", None)
        return _measured(points_base_mm, camera=self.rig_id or "camera",
                         captured_at_s=math.nan if stamp is None else float(stamp), depth_mm=_view_depth(self.view),
                         intrinsics=frame.intrinsics, camera_to_base=self.view.camera_to_base,
                         rgb=getattr(frame, "rgb", None))


def the_bins_look(looked: Any, kept: KeptTarget) -> Any:
    """The last of a pick's looks (``LookedAround.views``) taken from the look ``kept`` was found from, by that look's
    label and its camera, that kept its frame and where its camera stood; ``None`` where the pick took none."""
    if kept.look_label is None:
        return None
    for view in reversed(tuple(getattr(looked, "views", ()) or ())):
        if getattr(view, "label", None) != kept.look_label or not _view_placed(view):
            continue
        rig = str(getattr(view, "name", "") or "").split("@")[0]
        if rig and rig != kept.camera:
            continue
        return view
    return None


def recheck_on_look(view: Any, kept: KeptTarget, *, last: "KeptTarget | None" = None, inside: bool = False,
                    max_shift_mm: float = FOLLOW_WITHIN_MM, footprint_tolerance: float = FOOTPRINT_TOLERANCE,
                    rim_tolerance_mm: float = RIM_TOLERANCE_MM) -> "tuple[TargetCheck | None, str]":
    """``kept`` checked on the frame of one of a pick's looks (``view``), as :func:`recheck` reads it first: where
    ``last``, the sighting the task follows, reads there in depth and colour as it did when it was seen, within the same
    bounds of ``kept``, the check (followed, ``by`` ``depth``), with what that frame reads where the part goes where
    ``inside`` asks (:class:`InsideRead`); otherwise ``None`` and why it is unsure there. No detector and no new frame:
    a carry over the rim goes via the bin's look wherever this is unsure, and the bin is checked there as always."""
    bound = min(float(max_shift_mm), 0.5 * kept.diagonal_mm)
    last = kept if last is None else last
    probes = _probes_of(last, also=_earlier(last, kept)) if inside else None
    unsure, read = _rim_read(_LookReader(view), last, probes)
    if unsure:
        return None, unsure
    moved = _moved_mm(kept, last)
    if moved > bound or not _same_bin(kept, last, footprint_tolerance, rim_tolerance_mm):
        return None, f"the sighting it stands as is {moved:.0f} mm from the one kept, or of another size"
    return TargetCheck(seen=replace(last, image_png=None), kept=last, moved_mm=moved, why="", bound_mm=bound,
                       by="depth", inside=_inside_of(probes, read)), ""


def look_points(views: Any, *, stride: int = _VIEW_STRIDE) -> np.ndarray:
    """What a pick's looks (``LookedAround.views``) measured, BASE mm, ``(N, 3)``: every ``stride``-th pixel of each
    look's depth, placed where its camera stood; a look that kept no frame or no placement adds nothing."""
    from src.robot.grasping.multiview.scene_geometry import to_base_mm  # noqa: PLC0415

    clouds: list[np.ndarray] = []
    for view in views or ():
        if not _view_placed(view):
            continue
        depth = _view_depth(view)
        assert depth is not None  # placed
        mask = np.zeros(depth.shape[:2], dtype=bool)
        mask[::max(1, int(stride)), ::max(1, int(stride))] = True
        try:
            clouds.append(to_base_mm(mask, depth, np.asarray(view.frame.intrinsics, dtype=np.float64),
                                     np.asarray(view.camera_to_base, dtype=np.float64)))
        except (TypeError, ValueError) as exc:
            logger.info("a look of the pick could not be read again (%s: %s); it adds nothing", type(exc).__name__, exc)
    return np.vstack(clouds) if clouds else np.zeros((0, 3), dtype=np.float64)


#: The cells the way of a carry is read in, mm, and the share of them the pick's looks must have read before what stands
#: on the way counts as seen.
_WAY_CELL_MM = 20.0
WAY_SEEN_SHARE = 0.8


def highest_on_the_way(points: Any, start_xy: Any, end_xy: Any, *, reach_mm: float,
                       leave_out_xy: Any = None, leave_out_mm: float = 0.0) -> "tuple[float | None, float]":
    """The highest of ``points`` (BASE mm) within ``reach_mm`` of the straight way from ``start_xy`` to ``end_xy`` in
    BASE XY, those within ``leave_out_mm`` of any of ``leave_out_xy`` left out (``(M, 2)`` BASE XY: the part's own
    cloud, the place the jaws took it from), and the share of the way's :data:`_WAY_CELL_MM` cells, the left out ones
    aside, that hold a point: ``(None, share)`` where nothing stands there."""
    cloud = _finite_points(points)
    start, end = np.asarray(start_xy, dtype=np.float64)[:2], np.asarray(end_xy, dtype=np.float64)[:2]
    reach = max(0.0, float(reach_mm))
    away = None if leave_out_xy is None else np.asarray(leave_out_xy, dtype=np.float64).reshape(-1, 2)
    away = None if away is None or away.shape[0] == 0 or leave_out_mm <= 0.0 else away[np.isfinite(away).all(axis=1)]

    def off_the_way(xy: np.ndarray) -> np.ndarray:
        way = end - start
        length = float(np.dot(way, way))
        along = np.clip(((xy - start) @ way) / length, 0.0, 1.0) if length > 1e-9 else np.zeros(xy.shape[0])
        return np.linalg.norm(xy - (start + along[:, None] * way), axis=1)

    def left_out(xy: np.ndarray) -> np.ndarray:
        if away is None or away.shape[0] == 0 or xy.shape[0] == 0:
            return np.zeros(xy.shape[0], dtype=bool)
        from scipy.spatial import cKDTree  # noqa: PLC0415 (only a carry over the rim pays for it)

        distance, _index = cKDTree(away).query(xy, k=1)
        return np.asarray(distance, dtype=np.float64) <= float(leave_out_mm)

    low, high = np.minimum(start, end) - reach, np.maximum(start, end) + reach
    xs = np.arange(low[0] + _WAY_CELL_MM / 2.0, high[0], _WAY_CELL_MM)
    ys = np.arange(low[1] + _WAY_CELL_MM / 2.0, high[1], _WAY_CELL_MM)
    gx, gy = np.meshgrid(xs, ys)
    centres = np.column_stack([gx.ravel(), gy.ravel()])
    centres = centres[(off_the_way(centres) <= reach) & ~left_out(centres)] if centres.size else centres
    on = cloud[(off_the_way(cloud[:, :2]) <= reach) & ~left_out(cloud[:, :2])] if cloud.size else cloud
    if centres.shape[0] == 0:
        return (float(np.max(on[:, 2])) if on.shape[0] else None), 1.0
    held = {(int(i), int(j)) for i, j in np.floor((on[:, :2] - low) / _WAY_CELL_MM).astype(np.int64)}
    cells = np.floor((centres - low) / _WAY_CELL_MM).astype(np.int64)
    share = sum((int(i), int(j)) in held for i, j in cells) / centres.shape[0]
    return (float(np.max(on[:, 2])) if on.shape[0] else None), float(share)


# ---------------------------------------------------------------------------------------------------------------------
# Before every drop: the check
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetCheck:
    """What a look at a kept target before a drop came to (Q13 (b)).

    ``seen`` is the sighting nearest the kept target, ``None`` where none was seen; ``kept`` the sighting the task
    follows (``seen`` itself), ``None`` where it does not; ``moved_mm`` how far ``seen`` stands from the kept target in
    BASE XY; ``why`` is ``""`` where it is followed, else ``not_seen``, ``moved_too_far`` or ``footprint_changed``
    (another size: its footprint, or its rim's height); ``bound_mm`` how far it could have moved and been followed.
    ``by`` says what decided: ``depth`` where its rim read in depth and colour where it was last seen (``seen`` is that
    sighting, drawn over nothing new: no picture), ``detector`` where the detector looked. ``inside`` is what the same
    frame read where the part may go on the target followed (:class:`InsideRead`), where the check was asked to read it
    and could; ``None`` otherwise.
    """

    seen: "KeptTarget | None"
    kept: "KeptTarget | None"
    moved_mm: "float | None"
    why: str
    bound_mm: float
    by: str = ""
    inside: "InsideRead | None" = None

    @property
    def followed(self) -> bool:
        return self.kept is not None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        if self.seen is None:
            return "the target was not seen where it was kept"
        moved = 0.0 if self.moved_mm is None else self.moved_mm
        if self.followed and self.by == "depth":
            return (f"the target's rim reads in depth and colour where it was last seen, {moved:.0f} mm from where it "
                    f"was kept, within {self.bound_mm:.0f} mm: followed")
        if self.followed:
            return f"the target moved {moved:.0f} mm, within {self.bound_mm:.0f} mm: followed"
        if self.why == "moved_too_far":
            return f"the target moved {moved:.0f} mm, more than the {self.bound_mm:.0f} mm it is followed within: lost"
        return (f"another target stands where it was kept (footprint {self.seen.footprint_mm[0]:.0f} x "
                f"{self.seen.footprint_mm[1]:.0f} mm, rim at {self.seen.rim_mm:.0f} mm, moved {moved:.0f} mm): lost")

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe, as ``task.target_checked`` carries it (and ``why``)."""
        return {
            "moved_mm": self.moved_mm,
            "followed": self.followed,
            "target": None if self.seen is None else self.seen.to_dict(),
            "why": self.why,
        }


def _same_size(kept: KeptTarget, seen: KeptTarget, tolerance: float) -> bool:
    """Whether ``seen``'s footprint is within ``tolerance`` of ``kept``'s, side for side, either way round."""
    for wanted, got in zip(sorted(kept.footprint_mm), sorted(seen.footprint_mm)):
        if wanted <= 0.0 or abs(got - wanted) > tolerance * wanted:
            return False
    return True


def _moved_mm(kept: KeptTarget, seen: KeptTarget) -> float:
    """How far ``seen`` stands from ``kept`` in BASE XY, middle to middle."""
    return float(math.hypot(seen.centre_xy_mm[0] - kept.centre_xy_mm[0], seen.centre_xy_mm[1] - kept.centre_xy_mm[1]))


def _same_bin(kept: KeptTarget, seen: KeptTarget, footprint_tolerance: float, rim_tolerance_mm: float) -> bool:
    """Whether ``seen`` is of ``kept``'s size: its footprint within ``footprint_tolerance``, its rim within
    ``rim_tolerance_mm``."""
    return _same_size(kept, seen, float(footprint_tolerance)) and \
        abs(seen.rim_mm - kept.rim_mm) <= float(rim_tolerance_mm)


def _stands_where_it_stood(locator: Any, last: KeptTarget) -> str:
    """``""`` where ``last``'s rim band reads in one new frame of ``locator`` as it did when ``last`` was seen, else why
    that is unsure, said for the log.

    The rim band alone, the walls' tops (:attr:`KeptTarget.rim_band_mm`): a part placed inside changes the floor, never
    them. Read in one frame with no detector (``Locator.measure``), it stands where it stood where at least
    :data:`DEPTH_CHECK_SEEN` of it lands in the frame and reads no nearer than it should by more than
    :data:`DEPTH_CHECK_TOLERANCE_MM` (nearer: the hand or the carried part in front of it), at least
    :data:`DEPTH_CHECK_AGREE` of that within the tolerance, at most :data:`DEPTH_CHECK_FARTHER` farther (a wall that is
    gone), and its median colour within :data:`COLOUR_CHECK_DELTA_E` of ``last``'s. Unsure, too: a locator that measures
    nothing, a sighting whose colour was never read, a rim band of too few points. Every check that read a frame logs
    what it read against those bounds. A frame that cannot be taken raises, as a locate does.
    """
    return _rim_read(locator, last)[0]


def _rim_read(locator: Any, last: KeptTarget, probes: "_Probes | None" = None) -> "tuple[str, np.ndarray | None]":
    """:func:`_stands_where_it_stood`, and what the same frame read at ``probes`` where they are given: how much farther
    than they should each read, mm (:func:`_gap_mm`), ``None`` where no frame was read. One frame either way."""
    measure = getattr(locator, "measure", None)
    if not callable(measure):
        return "its camera reads no depth without the detector", None
    if last.colour_lab is None:
        return "the colour of its rim was never read", None
    band = last.rim_band_mm
    total = int(band.shape[0])
    if total < MIN_TARGET_POINTS:
        return f"its rim band holds {total} point(s), fewer than {MIN_TARGET_POINTS}", None
    extra = 0 if probes is None else int(probes.points.shape[0])
    read = measure(np.vstack([band, probes.points]) if probes is not None and extra else band)
    if read is None or np.shape(read.expected_mm) != (total + extra,):
        return "its camera answered with no reading of the rim band", None
    whole = _gap_mm(read)
    gap, at_the_probes = whole[:total], (whole[total:] if extra else None)
    hidden = gap < -DEPTH_CHECK_TOLERANCE_MM
    seen = np.isfinite(gap) & ~hidden
    count = int(np.count_nonzero(seen))
    agree = seen & (np.abs(gap) <= DEPTH_CHECK_TOLERANCE_MM)
    farther = seen & (gap > DEPTH_CHECK_TOLERANCE_MM)
    seen_share = count / total
    agree_share = int(np.count_nonzero(agree)) / count if count else 0.0
    farther_share = int(np.count_nonzero(farther)) / count if count else 0.0
    lab = np.asarray(read.lab, dtype=np.float64)[:total]
    coloured = agree & np.isfinite(lab).all(axis=1)
    delta_e = math.inf
    if int(np.count_nonzero(coloured)) >= MIN_TARGET_POINTS:
        delta_e = float(np.linalg.norm(np.median(lab[coloured], axis=0) - np.asarray(last.colour_lab, dtype=np.float64)))
    if seen_share < DEPTH_CHECK_SEEN:
        why = (f"only {seen_share:.0%} of its rim band was seen, {int(np.count_nonzero(hidden))} point(s) hidden by "
               f"something nearer (at least {DEPTH_CHECK_SEEN:.0%})")
    elif agree_share < DEPTH_CHECK_AGREE:
        why = f"only {agree_share:.0%} of its rim band read where it stood (at least {DEPTH_CHECK_AGREE:.0%})"
    elif farther_share > DEPTH_CHECK_FARTHER:
        why = f"{farther_share:.0%} of its rim band read farther, where a wall stood (at most {DEPTH_CHECK_FARTHER:.0%})"
    elif not math.isfinite(delta_e):
        why = "the colour of its rim could not be read"
    elif delta_e > COLOUR_CHECK_DELTA_E:
        why = f"its rim's colour stands dE {delta_e:.1f} from the one kept (at most {COLOUR_CHECK_DELTA_E:g})"
    else:
        why = ""
    logger.info("recheck: the %s's rim band, %d point(s) read by camera %r: %.1f %% seen, %.1f %% within %g mm, %.1f %% "
                "farther, colour dE %.1f: %s", last.label, total, read.camera, 100.0 * seen_share, 100.0 * agree_share,
                DEPTH_CHECK_TOLERANCE_MM, 100.0 * farther_share, delta_e, why or "it stands where it stood")
    return why, at_the_probes


def recheck(locators: "Sequence[Any]", kept: KeptTarget, *, last: "KeptTarget | None" = None,
            max_shift_mm: float = FOLLOW_WITHIN_MM, footprint_tolerance: float = FOOTPRINT_TOLERANCE,
            rim_tolerance_mm: float = RIM_TOLERANCE_MM, inside: bool = False,
            together: "Sequence[str]" = ()) -> TargetCheck:
    """Look at ``kept`` again with the camera that saw it, from where the arm stands (its look, on a wrist), and say
    whether it is still the target.

    First its rim, in one frame and no detector (the owner, 2026-10-08): where ``last``, the sighting the task follows
    now (``kept`` where none is given), reads in depth and colour as it did when it was seen (the module's ``DEPTH_CHECK``
    and ``COLOUR_CHECK`` bounds), it stands where it stood and is followed there, ``by`` ``depth``, with no detector
    asked. Wherever that is unsure, the detector looks, as it always did, ``by`` ``detector``: the sighting nearest
    ``kept`` is followed only within min(``max_shift_mm``, half its footprint's diagonal), a footprint within
    ``footprint_tolerance`` of its own, side for side, and a rim within ``rim_tolerance_mm`` of its own, up or down.

    Every bound is measured against ``kept``: a task passes the target its survey kept, never a later sighting it
    followed, so a bin that creeps between the parts is followed only within the bound of where it was found; ``last``
    is followed by its rim only while it stands within those bounds too. One frame, and one locate where the rim is
    unsure, with the locator of the rig that saw it (the first one where none names that rig). A frame or a locate that
    raises raises; the task decides what that means.

    ``inside`` reads, on the same frame, where the part may go on the target followed (:class:`InsideRead`, the check's
    ``inside``): a box's inside, in the one measure its rim is read with, or on the frame the detector located in; a
    flat top's points. Off, the check reads what it always read.

    ``together`` are the phrases of a sort's other places: where the detector looks, it grounds them all in one locate
    and takes only what it labels ``kept``'s phrase (:func:`_located_sightings`), so the bin of another rule standing
    nearer is never followed as this one. None, the locate of ``kept``'s phrase alone, as before.
    """
    bound = min(float(max_shift_mm), 0.5 * kept.diagonal_mm)
    named = [locator for locator in locators if _rig_of(locator) == kept.camera]
    locator = named[0] if named else (locators[0] if locators else None)
    if locator is None:
        return TargetCheck(seen=None, kept=None, moved_mm=None, why="not_seen", bound_mm=bound, by="detector")

    def distance(target: KeptTarget) -> float:
        return _moved_mm(kept, target)

    def same_bin(target: KeptTarget) -> bool:
        return _same_bin(kept, target, footprint_tolerance, rim_tolerance_mm)

    last = kept if last is None else last
    probes = _probes_of(last, also=_earlier(last, kept)) if inside else None
    unsure, read = _rim_read(locator, last, probes)
    if not unsure:
        moved = distance(last)
        if moved <= bound and same_bin(last):
            return TargetCheck(seen=replace(last, image_png=None), kept=last, moved_mm=moved, why="", bound_mm=bound,
                               by="depth", inside=_inside_of(probes, read))
        unsure = f"the sighting it stands as is {moved:.0f} mm from the one kept, or of another size"
    logger.info("recheck: the %s: %s; the detector looks", kept.label, unsure)
    located, seen = _located_sightings(locator, kept.phrase or kept.label, look=kept.look, together=together)
    if not seen:
        return TargetCheck(seen=None, kept=None, moved_mm=None, why="not_seen", bound_mm=bound, by="detector")
    nearest = min(seen, key=distance)
    moved = distance(nearest)
    if moved > bound:
        return TargetCheck(seen=nearest, kept=None, moved_mm=moved, why="moved_too_far", bound_mm=bound, by="detector")
    if not same_bin(nearest):
        return TargetCheck(seen=nearest, kept=None, moved_mm=moved, why="footprint_changed", bound_mm=bound,
                           by="detector")
    return TargetCheck(seen=nearest, kept=nearest, moved_mm=moved, why="", bound_mm=bound, by="detector",
                       inside=_inside_on(located, nearest, also=_earlier(nearest, kept, last)) if inside else None)


# ---------------------------------------------------------------------------------------------------------------------
# A place that moved: found again
# ---------------------------------------------------------------------------------------------------------------------

#: How far in colour a bin found again may stand from the one the task followed, CIE76 delta E, where both rims' colours
#: were read: twice what a quick look at the rim allows (:data:`COLOUR_CHECK_DELTA_E`), as another look and another spot
#: on the table light a rim otherwise. A choice, not a measurement; the owner's yellow and blue bins of one model read
#: dE 120 and 128 apart (2026-10-08), so the bin of another rule is never taken for this one.
RELOCATE_COLOUR_DELTA_E = 2.0 * COLOUR_CHECK_DELTA_E


@dataclass(frozen=True)
class Relocated:
    """What looking for a place again came to (:func:`relocate`).

    ``kept`` is the bin found again, kept as seen from the look it was seen from: the sighting the drop is planned over
    and every later check measures against; ``None``: found nowhere, and a person decides. ``phrase`` is what it was
    looked for as; ``moved_mm`` how far it stands from where the survey found it (``None`` where it was not found);
    ``by`` how it was found: ``check`` (the check's own sighting, no new locate), ``detector`` (a locate of the search),
    ``""`` where it was not. ``looks_tried`` are the looks it was located from, in order; ``refused`` the looks the task
    could not reach, skipped, each with why; ``passed_over`` what was seen of its phrase and not taken for it, each with
    why.
    """

    phrase: str
    kept: "KeptTarget | None"
    moved_mm: "float | None" = None
    by: str = ""
    looks_tried: tuple[str, ...] = ()
    refused: tuple[tuple[str, str], ...] = ()
    passed_over: tuple[str, ...] = ()

    @property
    def found_nowhere(self) -> bool:
        """Whether no look saw the place: the task puts the part back and asks a person."""
        return self.kept is None

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as text, ASCII, no trailing newline."""
        tried = ", ".join(self.looks_tried) or "no look"
        if self.kept is not None:
            how = "the check's own sighting" if self.by == "check" else "the detector"
            head = (f"the {self.phrase} was found again by {how}, {self.moved_mm or 0.0:.0f} mm from where the survey "
                    f"found it: {self.kept.render()}")
        else:
            head = f"the {self.phrase} was found nowhere: no {self.phrase} of its size and colour was seen from {tried}"
        lines = [head]
        lines.extend(f"  passed over {why}" for why in self.passed_over)
        lines.extend(f"  skipped look {look}: {why}" for look, why in self.refused)
        return "\n".join(line.encode("ascii", "backslashreplace").decode("ascii") for line in lines)

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe: what an event of the place found again, or found nowhere, carries."""
        return {
            "phrase": self.phrase,
            "found": self.kept is not None,
            "moved_mm": self.moved_mm,
            "by": self.by,
            "look": None if self.kept is None else self.kept.look_label,
            "target": None if self.kept is None else self.kept.to_dict(),
            "looks_tried": list(self.looks_tried),
            "refused": [[look, why] for look, why in self.refused],
            "passed_over": list(self.passed_over),
        }


class Relocation:
    """Looking for one place again, look by look, while the task moves the arm (:func:`relocate` builds one).

    Iterate it for the looks still to try, in order (``None``: where a fixed camera stands, no motion); for each, the
    task moves the arm there by its own judged verb and calls :meth:`at` with it, which locates the place's phrase once
    from where the arm stands; a look the task could not reach it hands to :meth:`skip`. The iteration ends once the
    place is found or every look was handed out; :meth:`result` says what it came to, at any time. Nothing here moves
    the arm. One task drives one from one thread.
    """

    def __init__(self, locators: "Sequence[Any]", surveyed: KeptTarget, *, last: KeptTarget,
                 plan: "Sequence[LookPose | None]", avoid: "Sequence[ExclusionRegion]", footprint_tolerance: float,
                 rim_tolerance_mm: float, colour_delta_e: float, together: "Sequence[str]" = ()) -> None:
        self.phrase = (surveyed.phrase or surveyed.label).strip()
        self.surveyed = surveyed
        self.last = last
        self._locators = list(locators)
        self._wrist = [locator for locator in self._locators if bool(getattr(locator, "on_the_wrist", False))]
        self._plan: list[LookPose | None] = list(plan)
        self._avoid = tuple(avoid)
        # The other places of a sort, grounded in every locate of this one (:func:`_located_sightings`).
        self._together = tuple(str(phrase) for phrase in together)
        self._footprint_tolerance = float(footprint_tolerance)
        self._rim_tolerance_mm = float(rim_tolerance_mm)
        self._colour_delta_e = float(colour_delta_e)
        self._kept: KeptTarget | None = None
        self._by = ""
        self._tried: list[str] = []
        self._refused: list[tuple[str, str]] = []
        self._passed: list[str] = []

    @property
    def kept(self) -> "KeptTarget | None":
        """The bin found again so far, ``None`` while it is not."""
        return self._kept

    @property
    def done(self) -> bool:
        """Whether the place was found, or every look was handed out."""
        return self._kept is not None or not self._plan

    @property
    def looks(self) -> "tuple[LookPose | None, ...]":
        """The looks still to try, in order; none once the place is found."""
        return () if self._kept is not None else tuple(self._plan)

    def __iter__(self) -> "Iterator[LookPose | None]":
        return self

    def __next__(self) -> "LookPose | None":
        if self.done:
            raise StopIteration
        return self._plan.pop(0)

    def _label(self, pose: "LookPose | None") -> str:
        from src.robot.execution.looks import look_label  # noqa: PLC0415

        if pose is not None:
            return look_label(pose)
        return "here" if self._wrist else ""

    def _takes(self, target: KeptTarget, where: str) -> bool:
        """Whether ``target`` is taken for the place: standing in no other place (``avoid``), of the size the survey
        found it, of the colour the task followed where both were read; why not goes on ``passed_over``."""
        at = f"({target.centre_xy_mm[0]:.0f}, {target.centre_xy_mm[1]:.0f}) mm" + (
            f" from look {where}" if where else "")
        region = next((region for region in self._avoid if region.contains(target.centre_xy_mm)), None)
        why = ""
        if region is not None:
            why = f"one at {at}, in {region.reason or 'an area another place keeps out'}"
        elif not _same_bin(self.surveyed, target, self._footprint_tolerance, self._rim_tolerance_mm):
            why = (f"one of another size at {at}: footprint {target.footprint_mm[0]:.0f} x {target.footprint_mm[1]:.0f} "
                   f"mm, rim at {target.rim_mm:.0f} mm, where the survey found {self.surveyed.footprint_mm[0]:.0f} x "
                   f"{self.surveyed.footprint_mm[1]:.0f} mm and {self.surveyed.rim_mm:.0f} mm")
        else:
            followed = self.last.colour_lab if self.last.colour_lab is not None else self.surveyed.colour_lab
            if followed is not None and target.colour_lab is not None:
                delta_e = float(np.linalg.norm(np.asarray(target.colour_lab, dtype=np.float64)
                                               - np.asarray(followed, dtype=np.float64)))
                if delta_e > self._colour_delta_e:
                    why = (f"one of another colour at {at}: its rim stands dE {delta_e:.0f} from the one followed (at "
                           f"most {self._colour_delta_e:g})")
        if why:
            self._passed.append(why)
            logger.info("relocate: the %s: %s; it is not taken for it", self.phrase, why)
            return False
        return True

    def _from_the_check(self, seen: KeptTarget) -> None:
        """Take the check's own sighting where it qualifies: the bin seen from the look the check was made from, moved
        past the bound it is followed within. No locate."""
        where = seen.look_label or ""
        if self._takes(seen, where):
            self._kept, self._by = seen, "check"
            self._tried.append(where or "where the camera stands")
            logger.info("relocate: the %s is found again by the check's own sighting, %.0f mm from where the survey "
                        "found it: %s", self.phrase, _moved_mm(self.surveyed, seen), seen.render())

    def at(self, pose: "LookPose | None") -> "KeptTarget | None":
        """Locate the place's phrase once from where the arm stands, at ``pose`` (``None``: where a fixed camera stands),
        and keep what is taken for the place there (:func:`relocate`'s rules), the nearest to where it was followed of
        several; ``None`` where nothing is. Once the place is found, nothing more is located. A locate that raises
        raises: the task decides what that means, as at a check. A sort's other places are grounded in the same locate
        (``together``), and only what the detector labels this place's phrase is a sighting of it."""
        if self._kept is not None:
            return self._kept
        label = self._label(pose)
        self._tried.append(label or "where the camera stands")
        seen: list[KeptTarget] = []
        for locator in self._wrist or self._locators:
            for target in _sightings_of(locator, self.phrase, look=pose, together=self._together):
                seen.append(replace(target, look_label=label) if pose is None and self._wrist else target)
        taken = [target for target in seen if self._takes(target, label)]
        if not taken:
            logger.info("relocate: no %s of its size and colour %s (%d sighting(s) of the phrase)", self.phrase,
                        f"from look {label}" if label else "where the camera stands", len(seen))
            return None
        best = min(taken, key=lambda target: _moved_mm(self.last, target))
        if len(taken) > 1:
            logger.info("relocate: %d sightings could be the %s from look %s; the nearest to where it was followed is "
                        "kept", len(taken), self.phrase, label or "where the camera stands")
        self._kept, self._by = best, "detector"
        logger.info("relocate: the %s is found again from %s, %.0f mm from where the survey found it: %s", self.phrase,
                    f"look {label}" if label else "where the camera stands", _moved_mm(self.surveyed, best),
                    best.render())
        return best

    def skip(self, pose: "LookPose | None", why: str) -> None:
        """Say that the task could not reach ``pose`` (a look refused before anything was sent): it is not located from,
        and the search goes on."""
        label = self._label(pose) or "where the camera stands"
        self._refused.append((label, str(why)))
        logger.warning("relocate: look %s was not reached (%s); it is skipped", label, why)

    def result(self) -> Relocated:
        """What the search came to so far: the bin found again, or found nowhere."""
        kept = self._kept
        return Relocated(phrase=self.phrase, kept=kept, moved_mm=None if kept is None else _moved_mm(self.surveyed, kept),
                         by=self._by, looks_tried=tuple(self._tried), refused=tuple(self._refused),
                         passed_over=tuple(self._passed))


def relocate(locators: "Sequence[Any]", surveyed: KeptTarget, *, looks: "Sequence[LookPose]" = (),
             check: "TargetCheck | None" = None, last: "KeptTarget | None" = None,
             avoid: "Iterable[ExclusionRegion]" = (), footprint_tolerance: float = FOOTPRINT_TOLERANCE,
             rim_tolerance_mm: float = RIM_TOLERANCE_MM, colour_delta_e: float = RELOCATE_COLOUR_DELTA_E,
             together: "Sequence[str]" = ()) -> Relocation:
    """Look for the place ``surveyed`` is again (the bin its survey found, which every check measures against), where
    the check before a drop lost it (``check``: not seen, moved too far, another size where it stood), so the part goes
    there, its drop planned anew, rather than back where it was gripped: the owner's "selbst wenn diese verstellt wird,
    findet er die Ablage" (2026-10-09). Only a bin found nowhere asks a person.

    Nothing here moves the arm: the task drives the looks, the part in its jaws, as it carries one to a look, and hands
    each look it reached to :meth:`Relocation.at`::

        search = relocate(locators, place.surveyed, looks=looks, check=check, last=place.kept, avoid=others)
        for look in search:                  # the looks still to try; none once the bin is found
            moved = move_to_look(arm, look)  # the task's own carry, with its own gates and stops (None: no motion)
            if refused_before_sending(moved):
                search.skip(look, f"{moved.status.value}: {moved.message}")
                continue
            if not moved.ok:
                ...                          # the task stops as after any carry that failed, the part in its jaws
            search.at(look)                  # one locate of the place's phrase from where the arm stands
        found = search.result()              # found.kept: the bin found again; None: found nowhere

    The looks: the look the check was made from first, where the arm stands (unless the detector saw nothing of the
    phrase there just now), then every other of the task's ``looks`` in order; a fixed camera's one place (``None``, no
    motion), unless the check saw nothing there. The check's own sighting is taken at once where it qualifies, with no
    locate and no look (``Relocated.by`` ``check``): a bin moved past the bound it is followed within, still in view.

    A sighting is taken for the place only where it stands in none of ``avoid`` (every other place's keep-out region,
    the circles about taught drops among them), is of the size the survey found it (its footprint within
    ``footprint_tolerance``, side for side, its rim within ``rim_tolerance_mm``) and of the rim colour the task followed
    within ``colour_delta_e`` where both were read: the bin of another rule, or another bin of the phrase, is never
    taken for it. Of several taken at one look, the nearest to where it was last followed (``last``, ``surveyed`` where
    none is given). What was seen and not taken is said (``Relocated.passed_over``). ``together`` are the phrases of a
    sort's other places: every locate of the search grounds them too, and only what the detector labels this place's
    phrase is a sighting of it (:func:`_located_sightings`); none, the locate of its phrase alone.

    What the task does with the bin found again: keeps it as the place's surveyed bin and the one it follows (later
    checks measure against it, so it is not lost again from where it stood before), lays its keep-out region in place of
    the old one (``ExclusionZones.forget_region``), and plans the drop over it (:func:`drop_plan`), checked from its look
    as every drop is. A ``surveyed`` that is no ``KeptTarget``, and a check that followed the place, are a programmer's
    error: raised.
    """
    from src.robot.execution.looks import look_label  # noqa: PLC0415

    if not isinstance(surveyed, KeptTarget):
        raise TypeError(f"a place is looked for again from the KeptTarget its survey found, not "
                        f"{type(surveyed).__name__}")
    if check is not None and check.followed:
        raise ValueError("the check followed the place where it stands, so nothing is looked for again")
    last = surveyed if last is None else last
    wrist = any(bool(getattr(locator, "on_the_wrist", False)) for locator in locators)
    saw_none = check is not None and check.by == "detector" and check.seen is None
    plan: list[LookPose | None] = []
    if wrist:
        here = last.look if last.look is not None else surveyed.look
        left_out = {look_label(here)} if here is not None and saw_none else set()
        for pose in ([here] if here is not None else []) + list(looks):
            label = look_label(pose)
            if label not in left_out and all(label != look_label(other) for other in plan if other is not None):
                plan.append(pose)
        if not plan and here is None and not looks and not saw_none:
            plan.append(None)
    elif not saw_none:
        plan.append(None)
    search = Relocation(locators, surveyed, last=last, plan=plan, avoid=tuple(avoid),
                        footprint_tolerance=footprint_tolerance, rim_tolerance_mm=rim_tolerance_mm,
                        colour_delta_e=colour_delta_e, together=together)
    if check is not None and check.seen is not None:
        search._from_the_check(check.seen)  # noqa: SLF001 (the search's own first step)
    return search


# ---------------------------------------------------------------------------------------------------------------------
# The drop
# ---------------------------------------------------------------------------------------------------------------------


def rim_clearance_mm(robot_config: Any) -> float:
    """The air a task leaves over a bin's rim when the operator chose none (Q3): ``SET_DOWN_AIR_MM`` (5 mm) and the
    perceived world's margin (``safety.planning_world.perceived.margin_mm``, 15 mm as shipped), 20 mm in all, so the
    carried part's bottom clears what the planner keeps about the rim it saw; held to the owner's
    :data:`RIM_AIR_MIN_MM` to :data:`RIM_AIR_MAX_MM` whatever the margin (the schema allows 0 to 500 mm). A tree that says
    nothing gets the owner's 20 mm."""
    from src.robot.perception.locator import SET_DOWN_AIR_MM  # noqa: PLC0415

    margin = getattr(getattr(getattr(getattr(robot_config, "safety", None), "planning_world", None), "perceived", None),
                     "margin_mm", None)
    if not isinstance(margin, (int, float)) or isinstance(margin, bool) or not math.isfinite(float(margin)):
        return RIM_AIR_DEFAULT_MM
    return min(RIM_AIR_MAX_MM, max(RIM_AIR_MIN_MM, float(SET_DOWN_AIR_MM) + float(margin)))


#: How far over what the camera reads inside a box the part's bottom stays when the jaws open below its rim, mm: the
#: owner's "never lower than 10 mm above what the camera reads inside the box" (2026-10-08 night).
INSIDE_AIR_MM = 10.0
#: The least a part goes under a box's rim to be set down there rather than let go over it, mm: the owner's range runs
#: from 10 to 50 mm (``robot.place.below_the_rim_mm``).
BELOW_RIM_LEAST_MM = 10.0
#: How far inside a box's opening the part and the open hand keep on every side before the part goes below the rim, mm,
#: where the cell says nothing (``robot.place.opening_margin_mm``).
OPENING_MARGIN_MM = 10.0
#: Which percentile of the part's cloud's reach from the grasp a part laid at a spot reaches, and how far either way a
#: part going below a rim reaches along the box's sides: so one stray point is no edge (``_fit``'s rule).
_REACH_PERCENTILE = 98.0


@dataclass(frozen=True)
class OpenHand:
    """The fingers of the open hand as a box in the tool frame, mm: ``half_x`` from the TCP's axis along the closing
    direction (tool X: half the stroke and a finger's thickness), ``half_y`` along the fingers' width (tool Y), from
    ``ahead`` past the TCP along the approach (the fingertips) back to ``behind`` before it (where the fingers leave the
    housing). What a set-down below a box's rim keeps inside the box's opening where the fingertips go below the rim,
    and what a part laid beside another keeps clear of it. Build one with :meth:`of`."""

    half_x: float
    half_y: float
    ahead: float
    behind: float

    @classmethod
    def of(cls, robot_config: Any) -> "OpenHand | None":
        """The hand ``robot_config`` names (``robot.gripper.model``, the registry's jaw), opened, where the TCP stands
        (``hand_past_the_tcp_mm``); ``None`` where no hand resolves."""
        from src.config.schema.robot import RobotConfig  # noqa: PLC0415
        from src.robot.safety.planning.hand import hand_past_the_tcp_mm, planner_hand  # noqa: PLC0415

        if not isinstance(robot_config, RobotConfig):
            return None
        try:
            hand = planner_hand(robot_config)
            past = float(hand_past_the_tcp_mm(robot_config))
        except Exception:  # noqa: BLE001 (a hand nobody can resolve is no hand to keep inside a box; said where it is built)
            return None
        if not chosen(hand):
            return None
        jaw = hand.jaw
        return cls(half_x=float(jaw.aperture_mm) / 2.0 + float(jaw.finger_thickness_mm),
                   half_y=float(jaw.finger_width_mm) / 2.0, ahead=float(jaw.finger_ahead_mm) + past,
                   behind=max(0.0, float(jaw.finger_behind_mm) - past))

    @property
    def reach_mm(self) -> float:
        """How far its fingers reach from the TCP's axis in the tool's XY plane."""
        return float(math.hypot(self.half_x, self.half_y))

    def corners_at(self, pose: Pose) -> np.ndarray:
        """Its eight corners, BASE mm, ``(8, 3)``, the tool standing at ``pose``."""
        local = np.array([(sx * self.half_x, sy * self.half_y, z) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                          for z in (-self.behind, self.ahead)], dtype=np.float64)
        matrix = np.asarray(pose.to_matrix(), dtype=np.float64)
        return local @ matrix[:3, :3].T + matrix[:3, 3]


@dataclass(frozen=True)
class PartBottom:
    """Where a part's bottom stood when the tool gripped it, BASE Z mm (``mm``), what its hang below the tool is measured
    from; ``measured`` whether the pick's own looks said so, else the support the cell declares; ``said`` why, for
    the log."""

    mm: float
    measured: bool
    said: str


def part_bottom(cloud_mm: Any, support_model: Any, *, declared_mm: float) -> PartBottom:
    """Where the part a pick gripped stood, a LOWER bound on its bottom, as ``robot.place.part_bottom: measured`` takes
    it: the lowest reading of a surface the pick's looks found under its cloud (``SupportModel.local_plane_under``,
    what the part stood on), or the cloud's own lowest point where that is lower (a camera never sees below a part's
    bottom, so its lowest point is at or over it, and a reading of what it stands on is at or under it).

    So the hang is the grasp's Z less what the part stood on, not less the support the cell declares (``declared_mm``):
    on the owner's cell a part on the 55 mm mat hung 55 mm short of what the drop was raised by, and the drop stood 85
    to 92 mm over the rim. A part that stood on another part hangs from the surface under both, so every error still
    goes toward more air. The declared support stands where the pick kept no cloud of the part, or its looks found no
    surface under it (no support model: a fixed camera's pick, a world that reads no supports)."""
    declared = PartBottom(float(declared_mm), False, f"the declared support at {float(declared_mm):.1f} mm")
    cloud = _finite_points(cloud_mm) if cloud_mm is not None else np.zeros((0, 3))
    if cloud.shape[0] < 2:
        return replace(declared, said=f"{declared.said}: the pick kept no cloud of the part")
    under = getattr(support_model, "local_plane_under", None)
    planes = under(cloud[:, :2]) if callable(under) else None
    planes = None if planes is None else np.asarray(planes, dtype=np.float64).reshape(-1, 4)
    usable = (np.isfinite(planes).all(axis=1) & (planes[:, 2] > 1e-6)) if planes is not None else None
    if planes is None or usable is None or not bool(np.any(usable)):
        return replace(declared, said=f"{declared.said}: the pick's looks found no surface under the part")
    rows, xy = planes[usable], cloud[usable, :2]
    readings = (rows[:, 3] - rows[:, 0] * xy[:, 0] - rows[:, 1] * xy[:, 1]) / rows[:, 2]
    surface, lowest = float(np.min(readings)), float(np.min(cloud[:, 2]))
    bottom = min(surface, lowest)
    return PartBottom(bottom, True, f"the surface under the part reads {surface:.1f} mm at its lowest and its cloud "
                                    f"reaches down to {lowest:.1f} mm: its bottom stood at {bottom:.1f} mm")


def part_reach_mm(part_cloud_mm: Any, grasp: Pose) -> "float | None":
    """How far the part the tool gripped at ``grasp`` reaches from the tool in BASE XY, mm: the 98th percentile of its
    cloud's distances from the grasp, which a turn about the vertical leaves as it is; ``None`` without a cloud."""
    if part_cloud_mm is None:
        return None
    cloud = _finite_points(part_cloud_mm)
    if cloud.shape[0] < 2:
        return None
    reach = np.linalg.norm(cloud[:, :2] - np.asarray(grasp.position_mm[:2], dtype=np.float64), axis=1)
    return float(np.percentile(reach, _REACH_PERCENTILE))


# ---------------------------------------------------------------------------------------------------------------------
# Side by side: the spots of a flat place
# ---------------------------------------------------------------------------------------------------------------------

#: How far over the support a point stands before it is something on a spot, mm: a choice, about the band a support
#: surface's own reading is taken within, and a little.
SPOT_BAND_MM = 8.0
#: How many points over the band (or under it: the support's edge) make a spot taken: one stray reading does not.
_SPOT_TAKEN_POINTS = 5
#: The share of a spot's cells (:data:`_SPOT_CELL_MM`) that must hold a reading of its support before it reads free.
_SPOT_SEEN_SHARE = 0.6
_SPOT_CELL_MM = 10.0
#: How far a spot's support may read from the taught pose's, mm, and still be the same place: past it a spot stands on
#: another surface, beside the mat, on a part.
_SPOT_LEVEL_MM = 5.0


@dataclass(frozen=True)
class Spot:
    """A spot of a flat place a part is laid on: its middle, BASE X and Y mm, how it was chosen (``camera``: the camera
    reads it free; ``grid``: the next the task has not filled, where no camera reads it), and how much higher than under
    the taught pose the camera reads its support, mm (0 where it read none)."""

    xy: "tuple[float, float]"
    by: str
    rise_mm: float = 0.0


def grid_spots(centre_xy: "Sequence[float]", heading_rad: float, *, rows: int, columns: int,
               pitch_mm: float) -> "list[tuple[float, float]]":
    """``rows`` x ``columns`` spots ``pitch_mm`` apart about ``centre_xy``, the rows along ``heading_rad`` from BASE X,
    the nearest the centre first (rows, then columns, in order on a tie)."""
    c, s = math.cos(float(heading_rad)), math.sin(float(heading_rad))
    spots: list[tuple[float, int, float, float]] = []
    for row in range(int(rows)):
        for column in range(int(columns)):
            a = (row - (int(rows) - 1) / 2.0) * float(pitch_mm)
            b = (column - (int(columns) - 1) / 2.0) * float(pitch_mm)
            spots.append((math.hypot(a, b), len(spots), float(centre_xy[0]) + c * a - s * b,
                          float(centre_xy[1]) + s * a + c * b))
    return [(x, y) for _distance, _index, x, y in sorted(spots)]


def top_spots(kept: KeptTarget, *, pitch_mm: float, reach_mm: float) -> "list[tuple[float, float]]":
    """The spots of ``kept``'s flat top a part reaching ``reach_mm`` from the tool lies on whole: ``pitch_mm`` apart along
    the top's own axes about its middle, each at least ``reach_mm`` inside its edges, the nearest the middle first; its
    middle alone where no other fits."""
    pitch = max(1.0, float(pitch_mm))
    counts = [max(0, int(math.floor((side / 2.0 - float(reach_mm)) / pitch))) for side in kept.footprint_mm]
    spots = grid_spots(kept.centre_xy_mm, kept.yaw_rad, rows=2 * counts[0] + 1, columns=2 * counts[1] + 1,
                       pitch_mm=pitch)
    return spots


def views_spot_reader(points: Any, support_model: Any,
                      taught_xy: "Sequence[float]") -> "Callable[[tuple[float, float], float], tuple[str, float]] | None":
    """How the camera reads a spot of the place about a taught pose, from what the pick's looks measured (``points``,
    BASE mm) and the surfaces they found the parts stand on (``support_model``): a callable of a spot's middle and
    radius answering ``free``, ``taken`` or ``unread``, and how much higher than under the taught pose its support reads.

    A spot is ``taken`` where its support reads more than :data:`_SPOT_LEVEL_MM` from the taught pose's (another
    surface), or at least :data:`_SPOT_TAKEN_POINTS` points within its radius stand over its support's band
    (:data:`SPOT_BAND_MM`: something on it) or under it (its support ends there); ``free`` where at least
    :data:`_SPOT_SEEN_SHARE` of its cells hold a reading of its support; ``unread`` otherwise. ``None`` where the looks
    measured nothing or found no surface under the taught pose: no camera reads that place."""
    cloud = _finite_points(points) if points is not None else np.zeros((0, 3))
    under = getattr(support_model, "height_under", None)
    if cloud.shape[0] == 0 or not callable(under):
        return None
    taught = under(np.asarray(taught_xy, dtype=np.float64)[:2].reshape(1, 2))
    if taught is None:
        return None
    level_there = float(taught)

    def read(xy: "tuple[float, float]", radius_mm: float) -> "tuple[str, float]":
        level = under(np.asarray(xy, dtype=np.float64).reshape(1, 2))
        if level is None or abs(float(level) - level_there) > _SPOT_LEVEL_MM:
            return "taken", 0.0
        near = cloud[np.linalg.norm(cloud[:, :2] - np.asarray(xy, dtype=np.float64), axis=1) <= float(radius_mm)]
        off = np.abs(near[:, 2] - float(level)) > SPOT_BAND_MM
        if int(np.count_nonzero(off)) >= _SPOT_TAKEN_POINTS:
            return "taken", 0.0
        return _seen_enough(near[~off], xy, radius_mm), max(0.0, float(level) - level_there)

    return read


def top_spot_reader(read: "InsideRead | None") -> "Callable[[tuple[float, float], float], tuple[str, float]] | None":
    """How the camera reads a spot of a target's flat top from what a check read of it (:class:`InsideRead` of kind
    ``top``): ``taken`` where at least :data:`_SPOT_TAKEN_POINTS` of its top's points within the spot's radius read under
    something new, ``free`` where at least :data:`_SPOT_SEEN_SHARE` of its cells hold a point read as its top, ``unread``
    otherwise; ``None`` where no such read was made."""
    if read is None or read.kind != "top" or read.covered is None or read.seen is None:
        return None
    points, covered, seen = read.points_mm, read.covered, read.seen

    def verdict(xy: "tuple[float, float]", radius_mm: float) -> "tuple[str, float]":
        near = np.linalg.norm(points[:, :2] - np.asarray(xy, dtype=np.float64), axis=1) <= float(radius_mm)
        if int(np.count_nonzero(near & covered)) >= _SPOT_TAKEN_POINTS:
            return "taken", 0.0
        return _seen_enough(points[near & seen & ~covered], xy, radius_mm), 0.0

    return verdict


def _seen_enough(points: np.ndarray, xy: "tuple[float, float]", radius_mm: float) -> str:
    """``free`` where ``points`` (what reads as the support within a spot's radius) hold at least
    :data:`_SPOT_SEEN_SHARE` of the spot's cells, else ``unread``."""
    cells = {(int(i), int(j)) for i, j in np.floor(points[:, :2] / _SPOT_CELL_MM).astype(np.int64)} if points.size \
        else set()
    need = math.pi * float(radius_mm) ** 2 / (_SPOT_CELL_MM * _SPOT_CELL_MM)
    return "free" if need > 0.0 and len(cells) >= _SPOT_SEEN_SHARE * need else "unread"


def choose_spot(candidates: "Sequence[tuple[float, float]]", *, reach_mm: float, margin_mm: float,
                placed: "Sequence[tuple[tuple[float, float], float]]" = (),
                read: "Callable[[tuple[float, float], float], tuple[str, float]] | None" = None,
                ) -> "tuple[Spot | None, str]":
    """The first of ``candidates`` (the nearest the place's middle first) where a part reaching ``reach_mm`` lies
    ``margin_mm`` clear of every part the task laid there before (``placed``: each one's middle and reach) and the camera
    does not read anything else (``read``, a spot reader; ``None`` where no camera reads the place): ``by`` ``camera``
    where the camera reads it free, ``grid`` where it reads nothing there. ``None`` and why where no spot is."""
    filled = taken = 0
    for xy in candidates:
        if any(math.hypot(xy[0] - other[0], xy[1] - other[1]) < float(reach_mm) + float(other_reach) + float(margin_mm)
               for other, other_reach in placed):
            filled += 1
            continue
        verdict, rise = read(xy, float(reach_mm) + float(margin_mm)) if read is not None else ("unread", 0.0)
        if verdict == "taken":
            taken += 1
            continue
        return Spot((float(xy[0]), float(xy[1])), "camera" if verdict == "free" else "grid", float(rise)), ""
    return None, (f"none of the place's {len(candidates)} spot(s) is free: {filled} hold a part this task laid there, "
                  f"the camera reads {taken} more taken")


@dataclass(frozen=True)
class Screen:
    """What the arm said about a configuration before any move went there: the joints (``None`` where none reaches
    the pose), the verdict (a ``PoseVerdict`` value), why, and the nearest configuration both authorities clear, in
    degrees, where one was found."""

    joints: "JointPositions | None"
    verdict: str
    detail: str
    nearby_deg: "list[float] | None" = None

    @property
    def is_error(self) -> bool:
        """Whether no move goes there: the exact guard, the planner, or no configuration at all refuses it."""
        return self.verdict in _ERRORS


def screen_joints(arm: Any, joints: "JointPositions") -> Screen:
    """What the exact guard and the planner say about ``joints`` before any move goes there (``screen_configuration``);
    UNSCREENED, said, for an arm that screens nothing or a screen that could not be read."""
    from src.robot.safety.planning.band import PoseScreen  # noqa: PLC0415

    if not callable(getattr(type(arm), "screen_configuration", None)):
        return Screen(joints, "unscreened", f"{type(arm).__name__} screens no configuration, so the move judges it "
                                            "when it runs")
    try:
        screen = arm.screen_configuration(joints)
    except (RobotError, RuntimeError, OSError, ValueError) as exc:
        return Screen(joints, "unscreened", f"the screen could not be read ({type(exc).__name__}: {exc}), so the "
                                            "move judges it when it runs")
    if not isinstance(screen, PoseScreen):
        return Screen(joints, "unscreened", "the arm's screen gave no verdict, so the move judges it when it runs")
    nearby = None if screen.nearby is None else [math.degrees(float(v)) for v in screen.nearby]
    return Screen(joints, screen.verdict.value, screen.render(), nearby)


def screen_pose(arm: Any, pose: Pose) -> Screen:
    """Where the arm would stand for ``pose`` (``nearest_configuration``), screened (:func:`screen_joints`).

    No configuration is ``guard_refused``, in the arm's words; an arm that names none, or cannot say where it stands, is
    UNSCREENED and said, and the move judges the pose when it runs."""
    if not callable(getattr(type(arm), "nearest_configuration", None)):
        return Screen(None, "unscreened", f"{type(arm).__name__} names no configuration for a pose, so it screens "
                                          "none and the move judges it when it runs")
    try:
        joints = arm.nearest_configuration(pose)
    except RobotConnectionError as exc:
        return Screen(None, "unscreened", f"the arm could not say where it stands ({exc}), so the move judges it when "
                                          "it runs")
    except RobotError as exc:
        return Screen(None, "guard_refused", f"no configuration reaches it: {exc}")
    return screen_joints(arm, joints)


@dataclass(frozen=True)
class DropPlan:
    """Where a task lets its part go, and what that was measured and judged from.

    ``kind`` is ``pose`` (a taught pose) or ``camera`` (a bin the camera found). ``pose`` is the TCP pose
    ``Robot.place`` takes, ``None`` where there is none; ``standoff_mm`` the place's standoff; ``rim_mm`` the bin's rim
    (``None`` at a taught pose); ``hang_mm`` how far the part hangs below the tool; ``air_mm`` the air over the rim;
    ``verdict``, ``detail``, ``joints`` and ``nearby_deg`` what the arm's screen said; ``refusal`` is ``""``,
    ``unreachable`` or ``does_not_fit`` and ``reason`` says why; ``fit`` whether the part fits the opening (``fits``,
    ``does_not_fit``, ``not_judged``); ``turned`` whether the drop was turned along the cell's natural closing axis.

    ``release`` says where the jaws open: ``over_the_rim`` (a box's rim, or a flat top, the air over it), ``below_the_rim``
    (``below_rim_mm`` under a box's rim, ``air_mm`` then below zero) or ``at_the_pose``. A drop below the rim carries the
    drop over it as ``instead``, the part let go there where the line into the box is refused before anything was sent;
    ``over_the_rim_why`` says why a part that could have gone below a rim did not. ``spot_xy`` is the spot of a flat place
    the part is laid on (``robot.place.side_by_side``), ``spot_by`` how it was chosen (``camera`` or ``grid``) and
    ``spot_reach_mm`` how far the part and the open hand reach from it; ``None`` for the one spot.
    """

    kind: str
    pose: "Pose | None"
    standoff_mm: float
    rim_mm: "float | None" = None
    hang_mm: "float | None" = None
    air_mm: "float | None" = None
    verdict: str = "unscreened"
    detail: str = ""
    joints: "JointPositions | None" = None
    nearby_deg: "list[float] | None" = None
    refusal: str = ""
    reason: str = ""
    fit: str = "not_judged"
    turned: bool = False
    release: str = ""
    below_rim_mm: "float | None" = None
    instead: "Pose | None" = None
    over_the_rim_why: str = ""
    spot_xy: "tuple[float, float] | None" = None
    spot_by: str = ""
    spot_reach_mm: "float | None" = None

    @property
    def ok(self) -> bool:
        """Whether there is a drop to place at."""
        return self.pose is not None and not self.refusal

    def to_event(self) -> dict[str, Any]:
        """Plain data as ``task.drop_planned`` carries it: below a box's rim and at a spot, how far under the rim and
        which spot too."""
        said: dict[str, Any] = {
            "kind": self.kind,
            "pose_mm": None if self.pose is None else [float(v) for v in self.pose.position_mm],
            "rim_mm": self.rim_mm,
            "hang_mm": self.hang_mm,
            "air_mm": self.air_mm,
            "verdict": self.verdict,
        }
        if self.below_rim_mm is not None:
            said["release"] = self.release
            said["below_rim_mm"] = self.below_rim_mm
        if self.spot_xy is not None:
            said["spot_mm"] = [float(self.spot_xy[0]), float(self.spot_xy[1])]
            said["spot_by"] = self.spot_by
        return said

    def to_dict(self) -> dict[str, Any]:
        """Plain data, ``json.dumps`` safe."""
        return {
            **self.to_event(),
            "quaternion_xyzw": None if self.pose is None else [float(v) for v in self.pose.quaternion_xyzw],
            "standoff_mm": self.standoff_mm,
            "detail": self.detail,
            "nearby_deg": self.nearby_deg,
            "refusal": self.refusal,
            "reason": self.reason,
            "fit": self.fit,
            "turned": self.turned,
            "release": self.release,
            "below_rim_mm": self.below_rim_mm,
            "instead_mm": None if self.instead is None else [float(v) for v in self.instead.position_mm],
            "over_the_rim_why": self.over_the_rim_why,
            "spot_mm": None if self.spot_xy is None else [float(self.spot_xy[0]), float(self.spot_xy[1])],
            "spot_by": self.spot_by,
            "spot_reach_mm": self.spot_reach_mm,
        }

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """Describe this to a person, as one line of ASCII, no trailing newline."""
        where = "at a taught pose" if self.kind == "pose" else "over the bin"
        if self.pose is None:
            line = f"no drop {where}: {self.reason}"
        else:
            x, y, z = (float(v) for v in self.pose.position_mm)
            parts = [f"drop {where} at ({x:.1f}, {y:.1f}, {z:.1f}) mm"]
            if self.rim_mm is not None:
                parts.append(f"rim {self.rim_mm:.1f} mm")
            if self.hang_mm is not None:
                parts.append(f"hang {self.hang_mm:.1f} mm")
            if self.below_rim_mm is not None:
                parts.append(f"set down {self.below_rim_mm:.1f} mm under the rim, over it where the line in is refused")
            elif self.air_mm is not None:
                parts.append(f"air {self.air_mm:.1f} mm")
            if self.spot_xy is not None:
                parts.append(f"at the spot ({self.spot_xy[0]:.0f}, {self.spot_xy[1]:.0f}) mm the {self.spot_by} chose")
            parts.append(f"screened {self.verdict}")
            if self.turned:
                parts.append("turned along the cell's natural closing axis")
            line = ", ".join(parts)
            if self.refusal:
                line += f"; {self.refusal.replace('_', ' ')}: {self.reason}"
        return line.encode("ascii", "backslashreplace").decode("ascii")


def _heading_rad(pose: Pose) -> float:
    tool_x = np.asarray(pose.to_matrix(), dtype=np.float64)[:3, 0]
    return float(math.atan2(float(tool_x[1]), float(tool_x[0])))


def _off_vertical_deg(pose: Pose) -> float:
    tool_z = np.asarray(pose.to_matrix(), dtype=np.float64)[:3, 2]
    return float(math.degrees(math.acos(max(-1.0, min(1.0, -float(tool_z[2]))))))


def _heading_of(pose: Pose) -> "float | None":
    """Where ``pose``'s tool +X heads, laid onto the base XY plane, radians; ``None`` where it points within
    ``NO_HEADING_WITHIN_DEG_OF_VERTICAL`` of the vertical and so heads nowhere on the table."""
    from src.geometry.closing_axis import NO_HEADING_WITHIN_DEG_OF_VERTICAL  # noqa: PLC0415

    tool_x = np.asarray(pose.to_matrix(), dtype=np.float64)[:3, 0]
    if math.hypot(float(tool_x[0]), float(tool_x[1])) < math.sin(math.radians(NO_HEADING_WITHIN_DEG_OF_VERTICAL)):
        return None
    return math.atan2(float(tool_x[1]), float(tool_x[0]))


def _turned_to(pose: Pose, heading_rad: float) -> "Pose | None":
    """``pose`` turned about the base vertical, through its own position, until its tool +X heads ``heading_rad``: its
    tilt is kept, so every height under the tool stays as it was and a hang measured at the grasp still bounds the
    part. ``None`` where its tool +X heads nowhere on the table."""
    from src.geometry.quaternion import multiply  # noqa: PLC0415

    current = _heading_of(pose)
    if current is None:
        return None
    turn = math.remainder(float(heading_rad) - current, 2.0 * math.pi)
    quaternion = np.asarray(pose.quaternion_xyzw, dtype=np.float64)
    if abs(turn) > 1e-12:
        quaternion = multiply(np.array([0.0, 0.0, math.sin(turn / 2.0), math.cos(turn / 2.0)]), quaternion)
    return Pose(position_mm=np.asarray(pose.position_mm, dtype=np.float64), quaternion_xyzw=quaternion,
                frame=Frame.BASE, label=pose.label)


def _turned_along(pose: Pose, grasp: Pose, natural_axis: "ClosingAxis | None") -> tuple[Pose, bool]:
    """``pose`` turned about the vertical until it closes along the cell's natural closing axis where ``grasp`` stood
    within :data:`VERTICAL_WITHIN_DEG` of straight down, else as it is (the grasp's own turn). Its tilt is the grasp's
    either way: stood upright, a part held off its middle would swing down by as much as it reaches out, past the hang
    the drop is raised by."""
    if natural_axis is None or _off_vertical_deg(grasp) > VERTICAL_WITHIN_DEG:
        return pose, False
    x, y, _z = (float(v) for v in pose.position_mm)
    try:
        heading = float(natural_axis.heading_at(x, y))
    except ValueError:
        return pose, False
    turned = _turned_to(pose, math.radians(heading))
    return (pose, False) if turned is None else (turned, True)


def _fit(kept: KeptTarget, part_cloud_mm: Any, grasp: Pose, drop: Pose) -> tuple[str, str]:
    """Whether the part, turned as it will hang, fits the bin's opening (or its top, where it shows none)."""
    if part_cloud_mm is None:
        return "not_judged", "the part's cloud is not known, so its fit through the opening was not judged"
    part = _finite_points(part_cloud_mm)
    if part.shape[0] < 2:
        return "not_judged", "the part's cloud is not known, so its fit through the opening was not judged"
    area, what = (kept.opening_mm, "opening") if kept.opening_mm is not None else (kept.footprint_mm, "top")
    turn = _heading_rad(drop) - _heading_rad(grasp)
    xy = part[:, :2] - np.asarray(grasp.position_mm[:2], dtype=np.float64)
    c, s = math.cos(turn), math.sin(turn)
    hanging = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    along, across = _turned(hanging, kept.yaw_rad)
    sizes = (float(np.percentile(along, 98.0) - np.percentile(along, 2.0)),
             float(np.percentile(across, 98.0) - np.percentile(across, 2.0)))
    if sizes[0] <= area[0] and sizes[1] <= area[1]:
        return "fits", f"the part, {sizes[0]:.0f} x {sizes[1]:.0f} mm as it hangs, fits the {what} of " \
                       f"{area[0]:.0f} x {area[1]:.0f} mm"
    return "does_not_fit", (f"the part, {sizes[0]:.0f} x {sizes[1]:.0f} mm as it will hang, does not fit the {kept.label}'s "
                            f"{what} of {area[0]:.0f} x {area[1]:.0f} mm")


def drop_plan(arm: Any, kept: KeptTarget, grasp: Pose, *, part_bottom_mm: float, air_mm: float,
              standoff_mm: float = 80.0, natural_axis: "ClosingAxis | None" = None,
              part_cloud_mm: Any = None, below_rim_mm: "float | None" = None, inside: "InsideRead | None" = None,
              opening_margin_mm: float = OPENING_MARGIN_MM, hand: "OpenHand | None" = None,
              spot: "Spot | None" = None) -> DropPlan:
    """Where the part the tool gripped at ``grasp`` goes into ``kept``: over the middle of its rim, at its rim plus the
    part's hang plus ``air_mm``.

    ``SetDown.onto`` measures it (the hang from ``part_bottom_mm``, the declared support, a lower bound on where the
    part stood, so every error goes toward more air), turned as the grasp was; then a grasp within
    :data:`VERTICAL_WITHIN_DEG` of vertical is turned about the vertical to close along ``natural_axis``, its tilt kept
    (so the hang still bounds the part); then the arm is asked where it would stand and that is screened
    (:func:`screen_pose`), no configuration or an ERROR being ``unreachable``; then the part, from ``part_cloud_mm`` (its
    cloud where it stood, BASE) turned as it will hang, must fit the opening (or the top), else it ``does_not_fit``.
    Without the cloud the fit is not judged, and said.

    ``spot`` lays the part on a spot of a flat top (:func:`top_spots`, ``robot.place.side_by_side``) rather than over its
    middle, at the same height. ``below_rim_mm`` sets a part going into a box down below its rim (``robot.place
    .release_in_a_box: below_the_rim``, :func:`_below_the_rim`), what the check read of its inside (``inside``) the floor
    it keeps :data:`INSIDE_AIR_MM` over, the part from ``part_cloud_mm`` and the open hand (``hand``) kept
    ``opening_margin_mm`` inside its opening; where any of that cannot be, the drop over the rim stands, and says why.
    """
    from src.robot.perception.locator import SetDown  # noqa: PLC0415

    set_down = SetDown.onto(kept.as_object(), grasp=grasp, part_bottom_mm=part_bottom_mm, air_mm=air_mm)
    if set_down.pose is None:
        return DropPlan(kind="camera", pose=None, standoff_mm=float(standoff_mm), rim_mm=set_down.top_mm,
                        hang_mm=set_down.hang_mm, air_mm=float(air_mm), refusal="unreachable", reason=set_down.reason)
    over = set_down.pose
    if spot is not None and kept.opening_mm is None:
        over = Pose(position_mm=np.array([spot.xy[0], spot.xy[1], float(over.position_mm[2])]),
                    quaternion_xyzw=np.asarray(over.quaternion_xyzw, dtype=np.float64), frame=Frame.BASE,
                    label=over.label)
    pose, turned = _turned_along(over, grasp, natural_axis)
    screen = screen_pose(arm, pose)
    plan = DropPlan(kind="camera", pose=pose, standoff_mm=float(standoff_mm), rim_mm=set_down.top_mm,
                    hang_mm=set_down.hang_mm, air_mm=float(air_mm), verdict=screen.verdict, detail=screen.detail,
                    joints=screen.joints, nearby_deg=screen.nearby_deg, turned=turned, release="over_the_rim")
    if spot is not None and kept.opening_mm is None:
        plan = replace(plan, spot_xy=spot.xy, spot_by=spot.by)
    if screen.is_error:
        return replace(plan, refusal="unreachable",
                       reason=f"no move goes to the drop over the {kept.label}: {screen.detail}")
    fit, said = _fit(kept, part_cloud_mm, grasp, pose)
    if fit == "does_not_fit":
        return replace(plan, fit=fit, refusal="does_not_fit", reason=said)
    plan = replace(plan, fit=fit, reason=said)
    if below_rim_mm is None or kept.opening_mm is None:
        return plan
    return _below_the_rim(arm, kept, grasp, plan, part_bottom_mm=float(part_bottom_mm), below_rim_mm=float(below_rim_mm),
                          inside=inside, margin_mm=float(opening_margin_mm), hand=hand, part_cloud_mm=part_cloud_mm)


def _below_the_rim(arm: Any, kept: KeptTarget, grasp: Pose, over: DropPlan, *, part_bottom_mm: float,
                   below_rim_mm: float, inside: "InsideRead | None", margin_mm: float, hand: "OpenHand | None",
                   part_cloud_mm: Any) -> DropPlan:
    """The part set down below the rim of the box ``kept`` rather than let go over it (``over``, the drop over its rim,
    which stands, with why, wherever this cannot be): the owner's "1-5 cm unter dem Kistenrand" (2026-10-08 night).

    The part's bottom goes ``below_rim_mm`` under the rim, never lower than :data:`INSIDE_AIR_MM` over the highest the
    check read anything inside (``inside``: the floor, or the parts already in it), and at least
    :data:`BELOW_RIM_LEAST_MM` under the rim, or the box reads too full and the part goes over the rim. The tool stands
    over the same middle, turned as over the rim. Only a grasp within :data:`VERTICAL_WITHIN_DEG` of vertical goes
    below a rim, and only where, at the drop, the part as it hangs (its cloud, rigid in the jaws, turned only about the
    vertical) and, wherever its fingertips go below the rim, the open hand (``hand``) stand ``margin_mm`` inside the
    box's opening on every side: the guard judges the hand on the line in, and nothing judges the part, which is not
    modelled, or the hand as it opens. Then the drop below is screened as the drop over the rim was. The plan keeps the
    drop over the rim as ``instead``: the place lets the part go there where the guard refuses the line into the box
    before anything is sent. Its standoff stays where the drop over the rim has it (``standoff_mm`` longer by as much as
    the part goes deeper): the part comes to the box as it always did, and only the line in goes on below the rim."""

    def stays_over(why: str) -> DropPlan:
        logger.info("drop: the part is let go over the %s's rim, not below it: %s", kept.label, why)
        return replace(over, over_the_rim_why=why)

    assert over.pose is not None  # the drop over the rim is the plan this one starts from
    tilt = _off_vertical_deg(grasp)
    if tilt > VERTICAL_WITHIN_DEG:
        return stays_over(f"its grasp stands {tilt:.0f} deg off vertical, and only a part gripped from above is set down "
                          "below a rim")
    if inside is None or inside.kind != "box" or inside.top_mm is None:
        return stays_over("the camera did not read what stands inside the box")
    rim = float(kept.rim_mm)
    bottom = max(rim - float(below_rim_mm), float(inside.top_mm) + INSIDE_AIR_MM)
    depth = rim - bottom
    if depth < BELOW_RIM_LEAST_MM:
        return stays_over(f"the camera reads the box full to {float(inside.top_mm):.0f} mm, {rim - float(inside.top_mm):.0f} "
                          f"mm under its rim, and a part set down in it keeps {INSIDE_AIR_MM:g} mm over that")
    hang = float(grasp.position_mm[2]) - float(part_bottom_mm)
    x, y, _z = (float(v) for v in over.pose.position_mm)
    pose = Pose(position_mm=np.array([x, y, bottom + hang]), quaternion_xyzw=np.asarray(over.pose.quaternion_xyzw,
                                                                                       dtype=np.float64),
                frame=Frame.BASE, label=f"set down in {kept.label}")
    outside = _outside_the_opening(kept, pose, grasp, part_cloud_mm, hand, margin_mm)
    if outside:
        return stays_over(outside)
    screen = screen_pose(arm, pose)
    if screen.is_error:
        return stays_over(f"no move goes to the drop {depth:.0f} mm under the rim: {screen.detail}")
    logger.info("drop: the part is set down %.1f mm under the %s's rim (the inside read to %.1f mm), over it where the "
                "line in is refused", depth, kept.label, float(inside.top_mm))
    # The standoff stays where the drop over the rim has it: the part comes to the box as it always did, and only the
    # line in goes deeper.
    deeper = float(over.pose.position_mm[2]) - float(pose.position_mm[2])
    return replace(over, pose=pose, standoff_mm=over.standoff_mm + deeper, air_mm=bottom - rim, verdict=screen.verdict,
                   detail=screen.detail, joints=screen.joints, nearby_deg=screen.nearby_deg, release="below_the_rim",
                   below_rim_mm=depth, instead=over.pose)


def _part_xy_at(part: np.ndarray, grasp: Pose, drop: Pose) -> np.ndarray:
    """Where the points of a part gripped at ``grasp`` stand in BASE XY with the tool at ``drop``: rigid in the jaws,
    the drop turned from the grasp only about the vertical."""
    turn = _heading_rad(drop) - _heading_rad(grasp)
    xy = part[:, :2] - np.asarray(grasp.position_mm[:2], dtype=np.float64)
    c, s = math.cos(turn), math.sin(turn)
    hanging = np.column_stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]])
    return hanging + np.asarray(drop.position_mm[:2], dtype=np.float64)


def _outside_the_opening(kept: KeptTarget, drop: Pose, grasp: Pose, part_cloud_mm: Any, hand: "OpenHand | None",
                         margin_mm: float) -> str:
    """Why the part as it hangs, or the open hand where its fingertips go below the rim, would not stand ``margin_mm``
    inside ``kept``'s opening on every side with the tool at ``drop``; ``""`` where both do. Unknown is not inside: no
    cloud of the part, no hand, an opening nobody can place."""
    edges = kept.inner_opening()
    if edges is None:
        return "the box's opening could not be placed"
    (a_low, a_high), (c_low, c_high) = edges
    margin = max(0.0, float(margin_mm))

    def past(along: np.ndarray, across: np.ndarray, what: str) -> str:
        beyond = max(a_low + margin - float(np.min(along)), float(np.max(along)) - (a_high - margin),
                     c_low + margin - float(np.min(across)), float(np.max(across)) - (c_high - margin))
        if beyond > 0.0:
            return (f"{what} would reach {beyond:.0f} mm past the opening's {a_high - a_low:.0f} x {c_high - c_low:.0f} mm "
                    f"less {margin:g} mm on every side")
        return ""

    part = _finite_points(part_cloud_mm) if part_cloud_mm is not None else np.zeros((0, 3))
    if part.shape[0] < 2:
        return "the part's cloud is not known, so it cannot be kept inside the opening"
    along, across = _turned(_part_xy_at(part, grasp, drop), kept.yaw_rad)
    trim = (100.0 - _REACH_PERCENTILE, _REACH_PERCENTILE)
    said = past(np.percentile(along, trim), np.percentile(across, trim), "the part as it hangs")
    if said:
        return said
    if hand is None:
        return "the open hand's size is not known, so it cannot be kept inside the opening"
    corners = hand.corners_at(drop)
    if float(np.min(corners[:, 2])) < float(kept.rim_mm):
        return past(*_turned(corners[:, :2], kept.yaw_rad), "the open hand")
    return ""


#: How much higher than its lowest a drop over a bin is screened where the lowest is refused (:func:`nominal_drop`). The
#: real drop hangs the part from the declared support, which no part stood below, and stood 85 to 92 mm over the rim on
#: the owner's cell (2026-10-07: "er ist immer Minimum 5cm darüber"), while the lowest put the open fingers among the
#: boxes the camera grew about the bin's walls and ended nine tasks before their first pick.
NOMINAL_RAISE_MM = 50.0


def nominal_drop(arm: Any, kept: KeptTarget, *, hang_mm: float, air_mm: float, standoff_mm: float = 80.0,
                 natural_axis: "ClosingAxis | None" = None) -> DropPlan:
    """The drop over ``kept`` a part hanging ``hang_mm`` below the tool would take, straight down, turned along
    ``natural_axis`` where the cell names one: what a task screens once it has found its bin, before its first pick,
    with the worst hang the cell declares. The hang is below the tool, the TCP: the fingertips' reach past it and the
    declared length past them, never the length alone, which would put the fingers into the rim. No fit is judged.

    Refused there, the drop is screened :data:`NOMINAL_RAISE_MM` higher, where the real drop stands at the least: a bin
    reachable from there costs no pick, and the drop each place takes is screened again with the grasp it holds. Only a
    bin refused at both heights is unreachable."""
    x, y = _rim_middle(kept.points_base_mm, kept.rim_mm)
    z = kept.rim_mm + max(0.0, float(hang_mm)) + float(air_mm)
    heading = 0.0
    if natural_axis is not None:
        try:
            heading = float(natural_axis.heading_at(x, y))
        except ValueError:
            heading = 0.0
    pose = Pose.tool_down(x, y, z, yaw_deg=heading, label=f"drop over {kept.label}")
    screen = screen_pose(arm, pose)
    raised = 0.0
    lowest = screen
    if screen.is_error:
        higher = Pose.tool_down(x, y, z + NOMINAL_RAISE_MM, yaw_deg=heading, label=f"drop over {kept.label}")
        again = screen_pose(arm, higher)
        if not again.is_error:
            pose, screen, raised = higher, again, NOMINAL_RAISE_MM
    detail = screen.detail if not raised else (
        f"{screen.detail} (screened {raised:g} mm higher, where the real drop stands at the least: at the lowest, "
        f"{lowest.detail})")
    plan = DropPlan(kind="camera", pose=pose, standoff_mm=float(standoff_mm), rim_mm=kept.rim_mm,
                    hang_mm=float(hang_mm), air_mm=float(air_mm) + raised, verdict=screen.verdict, detail=detail,
                    joints=screen.joints, nearby_deg=screen.nearby_deg, turned=natural_axis is not None)
    if screen.is_error:
        return replace(plan, refusal="unreachable", reason=(
            f"no move goes to a drop over the {kept.label}, at its lowest or {NOMINAL_RAISE_MM:g} mm higher: "
            f"{screen.detail}"))
    return plan


def pose_drop(arm: Any, taught: "JointPositions", grasp: "Pose | None", *, part_bottom_mm: float,
              length_mm: "float | None", standoff_mm: float = 80.0, fingertips_mm: float = 0.0,
              air_mm: float = 0.0, spot: "Spot | None" = None) -> DropPlan:
    """Where the part the tool gripped at ``grasp`` is let go at the pose ``taught``, which says where the part's bottom
    is let go: the tool at ``taught`` (``arm.fk``) raised in BASE Z by the part's hang.

    The hang is the grasp's Z less ``part_bottom_mm`` (the declared support, which no part stood below, so the hang is
    an upper bound and every error goes toward more air); where the pick reported no grasp pose, ``length_mm``, the
    declared ``payload.length_mm``, past the fingertips, which reach ``fingertips_mm`` below the tool. With neither
    nothing is set down blind. The drop is turned about the vertical as
    taught and tilted as the grasp was: the hang bounds the part only while it hangs as it was gripped, and a taught
    tilt the grasp did not have would swing a part held off its middle below the taught point (``SetDown``'s rule, its
    heading the taught one). Where the pick reported no grasp pose the taught turn is taken whole. The raised drop is
    screened (:func:`screen_pose`): no configuration or an ERROR is ``unreachable``.

    ``air_mm`` raises the drop over the taught point by that much more (a hang measured from the pick, not the declared
    support, keeps the set-down air: ``robot.place.part_bottom: measured``). ``spot`` lays the part on a spot of the
    place about the taught pose (``robot.place.side_by_side``, :func:`choose_spot`): the tool over the spot's middle,
    raised by as much as its support reads higher than the taught pose's, turned and tilted as at the taught pose.
    """
    try:
        at = arm.fk(taught)
    except (RobotError, RuntimeError, OSError, ValueError) as exc:
        return DropPlan(kind="pose", pose=None, standoff_mm=float(standoff_mm), refusal="unreachable",
                        reason=f"the arm could not say where the taught pose puts the tool: {type(exc).__name__}: "
                               f"{exc}")
    if grasp is not None:
        hang = float(grasp.position_mm[2]) - float(part_bottom_mm)
        if hang <= 0.0:
            return DropPlan(kind="pose", pose=None, standoff_mm=float(standoff_mm), hang_mm=hang,
                            refusal="unreachable",
                            reason=f"the grasp at Z {float(grasp.position_mm[2]):.1f} mm does not stand above the "
                                   f"part's bottom ({float(part_bottom_mm):.1f} mm), so its hang is unknown")
    elif length_mm is not None and math.isfinite(float(length_mm)) and float(length_mm) > 0.0:
        hang = max(0.0, float(fingertips_mm)) + float(length_mm)
    else:
        return DropPlan(kind="pose", pose=None, standoff_mm=float(standoff_mm), refusal="unreachable",
                        reason="the pick reported no grasp pose and safety.planning_world.payload.length_mm is "
                               "undeclared, so how far the part hangs below the tool is unknown and nothing is set "
                               "down blind")
    air = max(0.0, float(air_mm)) + (max(0.0, float(spot.rise_mm)) if spot is not None else 0.0)
    position = np.asarray(at.position_mm, dtype=np.float64) + np.array([0.0, 0.0, hang + air])
    if spot is not None:
        position[:2] = np.asarray(spot.xy, dtype=np.float64)
    pose = Pose(position_mm=position, quaternion_xyzw=np.asarray(at.quaternion_xyzw, dtype=np.float64),
                frame=Frame.BASE, label="drop at the taught pose")
    if grasp is not None:
        # The grasp's tilt at the taught position, turned to the taught heading; the grasp's own turn where either
        # heads nowhere on the table (a tool +X near the vertical), which still keeps every height under the tool.
        as_gripped = Pose(position_mm=position, quaternion_xyzw=np.asarray(grasp.quaternion_xyzw, dtype=np.float64),
                          frame=Frame.BASE, label="drop at the taught pose")
        heading = _heading_of(at)
        turned = _turned_to(as_gripped, heading) if heading is not None else None
        pose = turned if turned is not None else as_gripped
    screen = screen_pose(arm, pose)
    plan = DropPlan(kind="pose", pose=pose, standoff_mm=float(standoff_mm), hang_mm=hang, verdict=screen.verdict,
                    detail=screen.detail, joints=screen.joints, nearby_deg=screen.nearby_deg, release="at_the_pose",
                    air_mm=air if air > 0.0 else None, spot_xy=None if spot is None else spot.xy,
                    spot_by="" if spot is None else spot.by)
    if screen.is_error:
        return replace(plan, refusal="unreachable", reason=f"no move goes to the raised drop: {screen.detail}")
    return plan
