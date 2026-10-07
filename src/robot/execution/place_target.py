"""Where a task lets its part go: the bin its camera found, kept and looked at again, or a pose a person taught.

**The bin** (the owner's OD 19, with Q3, Q5 and Q13 A, 2026-09-30). A task that places "into the blue bin" finds the bin
with the cell's own detector on the cell's own cameras (:func:`locators_for_service`: the pick service lends its
perception backend, so no second copy of a model loads), before its first pick (:func:`survey`): the arm goes to each
of the task's looks in turn, the camera locates the bin's phrase, then the part's (which only counts the parts for the
chat), and the first look whose bin has at least :data:`MIN_TARGET_POINTS` points of surface is the look it is kept
from; of several, the most confident. What a task keeps of it (:class:`KeptTarget`) is measured, never written:

* its rim, the 95th percentile of the heights seen (``SetDown``'s top, so a stray pixel above it lifts nothing);
* its footprint and its turn, the smallest rectangle about it (``height_map.turn_of``), between the 1st and 99th
  percentiles of its rim band (the surface within 10 mm of the rim, its walls' tops) along its own axes;
* its opening, the gap between the inner edges of its walls' tops along each of those axes where the inside reads lower
  than the rim (:data:`OPENING_MIN_MM` or more), read from the rim band alone across the middle half of the band's own
  extent, so the inner faces and the floor a slanted camera sees fill nothing; a box with a flat top has none, and its
  part is set down on that top.

Before every drop the camera looks at the bin again from that look (:func:`recheck`): the bin seen nearest the kept one
is followed only within min(:data:`FOLLOW_WITHIN_MM`, half the kept footprint's diagonal), a footprint within
:data:`FOOTPRINT_TOLERANCE` of the kept one, side for side, and a rim within :data:`RIM_TOLERANCE_MM` of the kept one's;
anything else is the target lost, so a bin removed or swapped is never mistaken for it. The kept one is the bin the
survey found, never a later sighting, so a bin that creeps is not followed further and further. The drop
(:func:`drop_plan`) is ``SetDown.onto`` the bin, then four rules in order: the air over the rim is the task's rim air
(Q3: :func:`rim_clearance_mm`, 20 mm as shipped, the operator's choice of 10 to 50); a grasp within
:data:`VERTICAL_WITHIN_DEG` of vertical is turned about the vertical until it closes along the cell's natural closing
axis (``robot.natural_closing_axis``), any other keeps its own turn; the arm is asked where it would stand
(``nearest_configuration``) and the exact guard and the planner screen that (``screen_configuration``), and no
configuration or an ERROR is unreachable; the part, turned as it will hang, must fit the opening (or the top).

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

Nothing here commands the arm but the survey's looks, each through the arm's own judged verb (``move_to_look``); the
drop is placed by the task through ``Robot.place``. Every refusal is an answer, never a raise; a programmer's error
raises.
"""

from __future__ import annotations

import logging
import math
import weakref
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from src.contracts import UNSET, Maybe, chosen
from src.geometry import Frame, Pose
from src.robot.core.errors import CameraWorldUnavailable, RobotConnectionError, RobotError
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.geometry.closing_axis import ClosingAxis
    from src.robot.core import JointPositions
    from src.robot.execution.looks import LookPose
    from src.robot.execution.motion import MotionReport
    from src.robot.perception.locator import Located, LocatedObject, Locator

__all__ = [
    "FOLLOW_WITHIN_MM",
    "FOOTPRINT_TOLERANCE",
    "KEEP_OUT_MARGIN_MM",
    "MIN_TARGET_POINTS",
    "OPENING_MIN_MM",
    "RIM_AIR_DEFAULT_MM",
    "RIM_AIR_MAX_MM",
    "RIM_AIR_MIN_MM",
    "RIM_TOLERANCE_MM",
    "VERTICAL_WITHIN_DEG",
    "DropPlan",
    "KeptTarget",
    "Screen",
    "Survey",
    "TargetCheck",
    "drop_plan",
    "located_image",
    "locators_for_service",
    "nominal_drop",
    "pose_drop",
    "recheck",
    "rim_clearance_mm",
    "screen_joints",
    "screen_pose",
    "survey",
]

logger = logging.getLogger(__name__)

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


def _largest_gap(values: np.ndarray) -> float:
    if values.shape[0] < 2:
        return 0.0
    ordered = np.sort(values)
    return float(np.max(np.diff(ordered)))


def _opening(along: np.ndarray, across: np.ndarray, z: np.ndarray, rim_mm: float) -> "tuple[float, float] | None":
    """The gap between the inner edges of a target's wall tops along each of its own axes, or ``None`` for no opening.

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
    gaps: list[float] = []
    for axis, other in ((along, across), (across, along)):
        low, high = float(np.percentile(other[band], low_pct)), float(np.percentile(other[band], high_pct))
        middle, quarter = (low + high) / 2.0, (high - low) / 4.0
        strip = band & (other >= middle - quarter) & (other <= middle + quarter)
        gaps.append(_largest_gap(axis[strip]))
    if min(gaps) < OPENING_MIN_MM:
        return None
    return gaps[0], gaps[1]


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
    the frame's shutter, ``phrase`` what it was located for, ``image_png`` the sighting drawn over (display only).
    Build one with :meth:`of`.
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


def _sightings_of(locator: Any, phrase: str, *, look: "LookPose | None") -> "list[KeptTarget]":
    """Every target ``locator`` locates for ``phrase`` now, kept as seen from ``look``."""
    located = locator.locate(phrase)
    image = located_image(locator, phrase)
    found: list[KeptTarget] = []
    for obj in located.objects:
        kept = KeptTarget.of(obj, camera=located.camera, look=look, seen_at=located.captured_at_s, phrase=phrase)
        if kept is not None:
            found.append(replace(kept, image_png=_target_png(image, located, obj)) if image is not None else kept)
    return found


# ---------------------------------------------------------------------------------------------------------------------
# Before the first pick: the survey
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Survey:
    """What a task's survey came to: the target kept, or none, and where it looked.

    ``kept`` is the target (``None`` where no look saw one with enough points); ``looks_tried`` the looks it located from,
    in order; ``parts_seen`` how many parts of the object's phrase the look that kept the target saw outside it
    (``None`` for an empty phrase); ``refused`` the looks the planner refused before anything was sent, skipped, each
    with why. ``stopped`` is the look motion that ended the survey with nothing else commanded (``stopped_look`` its
    look), and ``halted`` why the task may not move any more, where that ended it before a motion.
    """

    kept: "KeptTarget | None"
    looks_tried: tuple[str, ...] = ()
    parts_seen: "int | None" = None
    refused: tuple[tuple[str, str], ...] = ()
    stopped: "MotionReport | None" = None
    stopped_look: str = ""
    halted: str = ""

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


def survey(arm: Any, locators: "Sequence[Any]", phrase: str, *, object_phrase: str = "",
           looks: "Sequence[LookPose]" = (), may_move: "Callable[[], str] | None" = None) -> Survey:
    """Find the target ``phrase`` names before a task's first pick, from the task's ``looks``.

    A camera on the wrist (the locators that say ``on_the_wrist``) sees what the arm points it at: the arm goes to each
    look in turn through its own judged verb (``move_to_look``), the camera locates ``phrase`` there, then
    ``object_phrase`` (which only counts the parts), and the survey ends at the first look whose target has at least
    :data:`MIN_TARGET_POINTS` points, keeping the most confident of several (Q5); with no look it locates where the arm
    stands. A look the planner refused before anything was sent is skipped and said; a look motion that may have moved
    the arm, or a controller that stopped, ends the survey there with nothing else commanded (``stopped``), and so
    does ``may_move`` answering why the task may not move any more, asked before every motion (``halted``). A fixed
    camera locates once where it stands and moves nothing. A camera that could not vouch on the way raises, as a pick's
    does; so does a locate that raises.
    """
    from src.robot.execution.generated_view import refused_before_sending  # noqa: PLC0415
    from src.robot.execution.looks import look_label, move_to_look  # noqa: PLC0415
    from src.robot.execution.motion import MotionOutcome  # noqa: PLC0415

    wrist = [locator for locator in locators if bool(getattr(locator, "on_the_wrist", False))]
    if not wrist:
        found = [target for locator in locators for target in _sightings_of(locator, phrase, look=None)]
        kept = _most_confident(found)
        parts = _parts_seen(locators, object_phrase, kept) if kept is not None else None
        return Survey(kept=kept, parts_seen=parts)
    plan: list[tuple[str, Any]] = [(look_label(pose), pose) for pose in looks] if looks else [("here", None)]
    tried: list[str] = []
    refused: list[tuple[str, str]] = []
    for label, pose in plan:
        if pose is not None:
            why = may_move() if may_move is not None else ""
            if why:
                return Survey(kept=None, looks_tried=tuple(tried), refused=tuple(refused), halted=why)
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
                              stopped_look=label)
        tried.append(label)
        found = [target for locator in wrist for target in _sightings_of(locator, phrase, look=pose)]
        kept = _most_confident(found)
        if kept is not None and pose is None:
            kept = replace(kept, look_label=label)
        # The part's phrase at every look, after the target's: it counts the parts this look sees for the chat.
        parts = _parts_seen(wrist, object_phrase, kept)
        if kept is not None:
            logger.info("survey: %s", kept.render())
            return Survey(kept=kept, looks_tried=tuple(tried), parts_seen=parts, refused=tuple(refused))
        if parts is not None:
            logger.info("survey: look %s sees %d part(s) of %r and no %r", label, parts, object_phrase, phrase)
    logger.info("survey: no %r was seen from %s", phrase, ", ".join(tried) or "any look")
    return Survey(kept=None, looks_tried=tuple(tried), refused=tuple(refused))


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
    """

    seen: "KeptTarget | None"
    kept: "KeptTarget | None"
    moved_mm: "float | None"
    why: str
    bound_mm: float

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


def recheck(locators: "Sequence[Any]", kept: KeptTarget, *, max_shift_mm: float = FOLLOW_WITHIN_MM,
            footprint_tolerance: float = FOOTPRINT_TOLERANCE,
            rim_tolerance_mm: float = RIM_TOLERANCE_MM) -> TargetCheck:
    """Look at ``kept`` again with the camera that saw it, from where the arm stands (its look, on a wrist), and say
    whether it is still the target: the sighting nearest it is followed only within min(``max_shift_mm``, half its
    footprint's diagonal), a footprint within ``footprint_tolerance`` of its own, side for side, and a rim within
    ``rim_tolerance_mm`` of its own, up or down.

    Every bound is measured against ``kept``: a task passes the target its survey kept, never a later sighting it
    followed, so a bin that creeps between the parts is followed only within the bound of where it was found. One
    locate, with the locator of the rig that saw it (the first one where none names that rig). A locate that raises
    raises; the task decides what that means.
    """
    bound = min(float(max_shift_mm), 0.5 * kept.diagonal_mm)
    named = [locator for locator in locators if _rig_of(locator) == kept.camera]
    locator = named[0] if named else (locators[0] if locators else None)
    if locator is None:
        return TargetCheck(seen=None, kept=None, moved_mm=None, why="not_seen", bound_mm=bound)
    seen = _sightings_of(locator, kept.phrase or kept.label, look=kept.look)
    if not seen:
        return TargetCheck(seen=None, kept=None, moved_mm=None, why="not_seen", bound_mm=bound)

    def distance(target: KeptTarget) -> float:
        return float(math.hypot(target.centre_xy_mm[0] - kept.centre_xy_mm[0],
                                target.centre_xy_mm[1] - kept.centre_xy_mm[1]))

    nearest = min(seen, key=distance)
    moved = distance(nearest)
    if moved > bound:
        return TargetCheck(seen=nearest, kept=None, moved_mm=moved, why="moved_too_far", bound_mm=bound)
    if not _same_size(kept, nearest, float(footprint_tolerance)) or \
            abs(nearest.rim_mm - kept.rim_mm) > float(rim_tolerance_mm):
        return TargetCheck(seen=nearest, kept=None, moved_mm=moved, why="footprint_changed", bound_mm=bound)
    return TargetCheck(seen=nearest, kept=nearest, moved_mm=moved, why="", bound_mm=bound)


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

    @property
    def ok(self) -> bool:
        """Whether there is a drop to place at."""
        return self.pose is not None and not self.refusal

    def to_event(self) -> dict[str, Any]:
        """Plain data as ``task.drop_planned`` carries it."""
        return {
            "kind": self.kind,
            "pose_mm": None if self.pose is None else [float(v) for v in self.pose.position_mm],
            "rim_mm": self.rim_mm,
            "hang_mm": self.hang_mm,
            "air_mm": self.air_mm,
            "verdict": self.verdict,
        }

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
            if self.air_mm is not None:
                parts.append(f"air {self.air_mm:.1f} mm")
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
              part_cloud_mm: Any = None) -> DropPlan:
    """Where the part the tool gripped at ``grasp`` goes into ``kept``: over the middle of its rim, at its rim plus the
    part's hang plus ``air_mm``.

    ``SetDown.onto`` measures it (the hang from ``part_bottom_mm``, the declared support, a lower bound on where the
    part stood, so every error goes toward more air), turned as the grasp was; then a grasp within
    :data:`VERTICAL_WITHIN_DEG` of vertical is turned about the vertical to close along ``natural_axis``, its tilt kept
    (so the hang still bounds the part); then the arm is asked where it would stand and that is screened
    (:func:`screen_pose`), no configuration or an ERROR being ``unreachable``; then the part, from ``part_cloud_mm`` (its
    cloud where it stood, BASE) turned as it will hang, must fit the opening (or the top), else it ``does_not_fit``.
    Without the cloud the fit is not judged, and said.
    """
    from src.robot.perception.locator import SetDown  # noqa: PLC0415

    set_down = SetDown.onto(kept.as_object(), grasp=grasp, part_bottom_mm=part_bottom_mm, air_mm=air_mm)
    if set_down.pose is None:
        return DropPlan(kind="camera", pose=None, standoff_mm=float(standoff_mm), rim_mm=set_down.top_mm,
                        hang_mm=set_down.hang_mm, air_mm=float(air_mm), refusal="unreachable", reason=set_down.reason)
    pose, turned = _turned_along(set_down.pose, grasp, natural_axis)
    screen = screen_pose(arm, pose)
    plan = DropPlan(kind="camera", pose=pose, standoff_mm=float(standoff_mm), rim_mm=set_down.top_mm,
                    hang_mm=set_down.hang_mm, air_mm=float(air_mm), verdict=screen.verdict, detail=screen.detail,
                    joints=screen.joints, nearby_deg=screen.nearby_deg, turned=turned)
    if screen.is_error:
        return replace(plan, refusal="unreachable",
                       reason=f"no move goes to the drop over the {kept.label}: {screen.detail}")
    fit, said = _fit(kept, part_cloud_mm, grasp, pose)
    if fit == "does_not_fit":
        return replace(plan, fit=fit, refusal="does_not_fit", reason=said)
    return replace(plan, fit=fit, reason=said)


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
              length_mm: "float | None", standoff_mm: float = 80.0, fingertips_mm: float = 0.0) -> DropPlan:
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
    position = np.asarray(at.position_mm, dtype=np.float64) + np.array([0.0, 0.0, hang])
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
                    detail=screen.detail, joints=screen.joints, nearby_deg=screen.nearby_deg)
    if screen.is_error:
        return replace(plan, refusal="unreachable", reason=f"no move goes to the raised drop: {screen.detail}")
    return plan
