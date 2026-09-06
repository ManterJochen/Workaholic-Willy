"""Is this grasp real? One predicate, answered from the true geometry: no tolerance, no matching.

A candidate is not compared to a label, it is checked against the scene. Do the two finger pads land
on the object, are the surface normals inside the friction cone, does the jaw open wide enough, is
the path in clear. Each answer is closed-form from :mod:`datagen.grasps.shapes`, so "the calculator
proposed 12 grasps and 4 of them cannot physically close" is a statement about the world rather than
about a second heuristic.

Independence is the point, and it costs something. The gripper envelope is re-implemented here from
its measured dimensions rather than imported from ``src.robot.grasping.collision``: a reference that
shared code with the thing it grades would agree with it for the wrong reasons. The numbers
themselves come from ``ParallelJawGripperModel`` because they are facts about a Robotiq 2F-85, not
opinions of an algorithm:

    pad 24 x 24 x 41.5 mm, finger 62.3 mm, palm 35 x 70 mm, aperture 84.84 mm

Every one of those is measured off the asset (see :class:`JawModel`), not read off a datasheet.

What this predicate does not know: reachability (no IK, so a verdict never depends on which arm is
bolted down), the sensor (the geometry is the settled truth, not what a camera could see), and the
dynamics (a valid grasp here is one that closes, not one that provably survives a lift). The physics
screen is what converts the last of those from an argument into a measurement.

A stated limitation: contacts on an edge. The contact normal is taken at the pad's extreme point,
and on a box edge that point's normal is discontinuous. A 40 mm square section closed on at 15 deg
reports a 15 deg contact, at 20 deg reports 70 deg (the pad corner has crossed onto the neighbouring
face), and at 25 deg reports 25 deg again. Treating an edge contact as non-antipodal is defensible
on its own terms, but the jump is a property of the single-point model rather than of the grasp.
Averaging the normal over the contact region would smooth it and has not been done.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

import numpy as np

from datagen.grasps.shapes import TABLE_Z_MM, Solid

__all__ = [
    "PROCEDURAL_JAWS",
    "jaw_contact_patch",
    "JawGrasp", "JawModel", "Rejection", "SuctionCup", "SuctionGrasp", "Verdict",
    "check_jaw_grasp", "check_suction_grasp",
]

_EPS = 1e-9
#: How far above the support plane a fingertip must stay. Small on purpose: the fingertip may sit on
#: the table, it may only not go through it. See :func:`check_jaw_grasp`.
_TABLE_CLEARANCE_MM = 1.0
#: How close two pad positions' spans must be to count as reaching the same surface, so that a
#: tie is broken by position order rather than by round-off. A thousandth of a millimetre: far
#: above the ~1e-9 that rotation arithmetic produces and the ~1e-5 a float32 ray engine does,
#: far below any difference a real surface can have. See the note in `_pad_contacts`.
_SPAN_TIE_MM = 1e-3


class Rejection(StrEnum):
    """Why a grasp is not real. Typed, because the rejection histogram is half the result."""

    NONE = "none"
    AXIS_NOT_PERPENDICULAR = "axis_not_perpendicular"   # must be perpendicular to approach by contract
    LINE_MISSES_OBJECT = "line_misses_object"           # the closing line never enters the target
    ANCHOR_OUTSIDE = "anchor_outside"                   # the anchor is not between the two contacts
    TOO_WIDE = "too_wide"                               # the object spans more than the jaw opens
    TOO_THIN = "too_thin"                               # below the minimum commandable width
    NOT_ANTIPODAL = "not_antipodal"                     # contact normals outside the friction cone
    FINGER_COLLISION = "finger_collision"               # a finger is inside something else at contact
    APPROACH_BLOCKED = "approach_blocked"               # the path in is not clear
    BELOW_TABLE = "below_table"                         # a fingertip would be under the support plane
    APPROACH_UNDER_TABLE = "approach_under_table"        # the path in passes under the support plane
    # suction only
    CONTACT_OFF_SURFACE = "contact_off_surface"
    SEAL_NOT_FLAT = "seal_not_flat"                     # the cup rim cannot sit on the surface
    APPROACH_NOT_NORMAL = "approach_not_normal"
    CUP_COLLISION = "cup_collision"


@dataclass(frozen=True, slots=True)
class Verdict:
    """The answer, plus the numbers that produced it: a bare bool would hide the near-misses."""

    ok: bool
    reason: Rejection = Rejection.NONE
    object_span_mm: float = 0.0        # what the jaw actually has to bridge
    contact_angle_deg: float = 0.0     # worst deviation of a contact normal from the closing axis
    approach_tilt_deg: float = 0.0     # angle between the approach and straight down
    detail: str = ""

    def __bool__(self) -> bool:
        return self.ok


@dataclass(frozen=True, slots=True)
class JawGrasp:
    """A parallel-jaw grasp in BASE mm: the same four quantities a ``GraspPoint`` carries."""

    position_mm: np.ndarray
    approach: np.ndarray        # unit, points toward the object (the direction the gripper travels)
    closing_axis: np.ndarray    # unit, perpendicular to approach, the line between the two pads
    width_mm: float


@dataclass(frozen=True, slots=True)
class SuctionGrasp:
    """A suction contact in BASE mm: where the cup lands and which way it presses."""

    position_mm: np.ndarray
    approach: np.ndarray        # unit, points into the surface


@dataclass(frozen=True, slots=True)
class JawModel:
    """The 2F-85 envelope, re-implemented here on purpose (see the module docstring)."""

    #: Measured off Isaac's ``Robotiq_2F_85_edit.usd`` in the base_link frame, with the gripper
    #: driven fully open, not taken from a datasheet and not guessed. The contact face is the prim
    #: named ``...PAD_OPEN_fingertipsstep``; the numbers below are its bounds:
    #:
    #:     pad (contact face)   24.0 wide x 24.0 thick x 41.5 long,  z = 122.1 ... 163.7 mm
    #:     whole inner finger   30.1 x 38.5 x 62.3 mm,               z = 102.3 ... 164.6 mm
    #:     full opening         84.84 mm at finger_joint = 0.8 rad
    #:
    #: An undersized pad makes the verdict too lenient: with pad_length 20 mm and thickness 12 mm,
    #: under half the real pad in both, a 40 mm cube gripped at mid-height passes while a real
    #: fingertip sits 11 mm under the table. Every correction here makes the verdict stricter, which
    #: is the direction an honest error should move a reference.
    aperture_mm: float = 85.0
    min_width_mm: float = 5.0
    #: Two reaches from the contact rather than one length, because the finger is not symmetric
    #: about it. A single ``finger_length_mm`` with the tip placed half a pad ahead of the contact
    #: puts the lowest modelled point 20.75 mm below the grasp centre, while the 2F-85's own
    #: collision shapes reach 28.72 mm below it: about 8 mm too lenient toward the table, which
    #: passes grasps a real gripper would drive into the surface and under-counts ``below_table``.
    #:
    #: The recorded pair is the worst case over apertures 10..82 mm, since the finger swings forward
    #: as the jaw closes and reaches deepest at a narrow aperture, which is exactly a thin part
    #: gripped close to the table.
    finger_ahead_mm: float = 28.72
    finger_behind_mm: float = 33.37
    finger_thickness_mm: float = 31.35
    finger_width_mm: float = 27.0
    palm_depth_mm: float = 35.0
    palm_width_mm: float = 70.0
    #: Coulomb friction at the pad. 0.5 gives a 26.6 deg half-angle cone, the textbook
    #: rubber-on-plastic figure. Exposed because the antipodal verdict is the one result that moves
    #: with it.
    friction_coefficient: float = 0.5
    #: How far back along -approach the path is checked, and how much wider the fingers sit while
    #: travelling. A gripper does not arrive already closed to width.
    approach_clearance_mm: float = 80.0
    approach_opening_margin_mm: float = 5.0
    #: Sample spacing along a finger. 3 mm is a quarter of the finger's own thickness, so a missed
    #: overlap would have to be thinner than that to escape.
    sample_step_mm: float = 3.0
    #: The contact patch along the approach: 38.0 mm in total, and not centred on the grasp point,
    #: since it runs 14.39 mm behind and 23.61 mm in front. A pad is a rectangle, not a point, and
    #: modelling it as one is not cosmetic: a single line through the anchor rejects candidates as
    #: "anchor outside the object" whose miss is a fraction of a millimetre, which is an
    #: idealisation failing rather than a grasp failing.
    #:
    #: This is the most sensitive parameter in the reference, and any number that depends on it must
    #: be quoted with it. The pad takes the widest contact over its whole face, so a real pad needs
    #: a wider opening than an idealised line, and a large share of ``TOO_WIDE`` comes from
    #: modelling the pad as a rectangle at all. A line pad is the floor no jaw can get under; the
    #: rest is what a physical pad costs.
    pad_ahead_mm: float = 23.61
    pad_behind_mm: float = 14.39
    #: How far outside the contact interval the anchor may sit. The anchor is derived from a quantised
    #: depth image; a pad that reaches the object does not care where the label point was placed.
    anchor_tolerance_mm: float = 3.0

    @property
    def friction_half_angle_deg(self) -> float:
        return float(np.degrees(np.arctan(self.friction_coefficient)))


@dataclass(frozen=True, slots=True)
class SuctionCup:
    """A cup, as geometry. The seal question here is 'can the rim sit flat', nothing more."""

    name: str = "standard"
    diameter_mm: float = 30.0
    #: How far the surface may deviate from the rim plane before the seal is not credible. A cup's
    #: bellows take some of this; 1.5 mm over a 30 mm rim is a 5 % slope and deliberately generous.
    flatness_tolerance_mm: float = 1.5
    #: Sample count around the rim. Eight points catch an edge overhang at any orientation.
    rim_samples: int = 8
    max_normal_deviation_deg: float = 15.0
    approach_clearance_mm: float = 60.0
    #: Vacuum pressure the cup can hold, kPa. Only used for the separate payload flag, never folded
    #: into the geometric verdict.
    vacuum_kpa: float = 60.0

    def holding_force_n(self) -> float:
        area_m2 = np.pi * (self.diameter_mm / 2000.0) ** 2
        return float(self.vacuum_kpa * 1000.0 * area_m2)


#: The two cups the cell is specced around: geometry only, with no learned model near them.
STANDARD_CUP = SuctionCup(name="standard", diameter_mm=30.0)
SLIM_CUP = SuctionCup(name="slim", diameter_mm=15.0, flatness_tolerance_mm=1.0)


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < _EPS:
        raise ValueError("direction cannot be the zero vector")
    return v / n


def _tilt_from_down_deg(approach: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(float(approach @ np.array([0.0, 0.0, -1.0])), -1.0, 1.0))))


def _finger_samples(
    contact: np.ndarray, approach: np.ndarray, binormal: np.ndarray, model: JawModel,
    *, length_mm: float | None = None,
) -> np.ndarray:
    """Points filling one finger: from the fingertip back along -approach, spread across the pad width.

    The tip leads the contact by ``finger_ahead_mm``. Starting the samples at the contact instead
    lets a jaw whose tip is under the table pass the table check, because the part of the finger
    that is actually below it is never sampled.

    Thickness is not sampled; it is handed to the obstacle as an inflation margin instead, which is
    both cheaper and conservative in the direction that matters.
    """
    tip = contact + model.finger_ahead_mm * approach
    reach = (model.finger_ahead_mm + model.finger_behind_mm) if length_mm is None else length_mm
    lengths = np.arange(0.0, reach + model.sample_step_mm, model.sample_step_mm)
    widths = np.linspace(-model.finger_width_mm / 2.0, model.finger_width_mm / 2.0, 3)
    grid = (tip[None, None, :]
            - lengths[:, None, None] * approach[None, None, :]
            + widths[None, :, None] * binormal[None, None, :])
    return grid.reshape(-1, 3)


def _pad_contacts(
    target: Solid, anchor: np.ndarray, axis: np.ndarray, approach: np.ndarray,
    binormal: np.ndarray, model: JawModel,
):
    """Where the two pads first touch, treating each as a rectangle rather than a point.

    Returns ``(t_enter, t_exit, contact_minus, contact_plus)`` along ``axis`` from ``anchor``, or
    ``None`` if no part of either pad meets the object. A real pad closes until its nearest point
    touches, so the extremes across the whole patch decide both the span and the contact normals:
    the same reason a wide pad grips a tilted face a single line would slide off.
    """
    half_width = model.finger_width_mm / 2.0
    # The pad is not centred on the contact: it runs 14.39 mm behind the grasp centre and 23.61 mm
    # in front of it, so assuming symmetry puts a third of the patch where there is none.
    offsets = [b * binormal + a * approach
               for b in (-half_width, 0.0, half_width)
               for a in (-model.pad_behind_mm, 0.0, model.pad_ahead_mm)]
    best_enter = best_exit = None
    origin_enter = origin_exit = None
    for offset in offsets:
        span = target.line_span(anchor + offset, axis)
        if span is None:
            continue
        # A candidate must beat the incumbent by more than `_SPAN_TIE_MM` to take the contact from
        # it, so pad positions that reach the same surface stay tied and the first one keeps it.
        #
        # Without a tolerance the winner is decided by whichever floating-point result happens to be
        # smaller. Every flat face has all nine positions reaching one plane; on a rotated box they
        # differ by about 1e-9 from the rotation arithmetic, and the contact then jumps to a
        # different corner of the pad, up to 38 mm along the approach. The finger samples are built
        # from that contact, so round-off rather than geometry would decide a collision verdict.
        #
        # The mesh path also quantises its own ray results (see `_ray_mesh`), which makes a float32
        # engine's spans deterministic in themselves; this tolerance does not do that for it, so the
        # two are complementary rather than redundant.
        if best_enter is None or span[0] < best_enter - _SPAN_TIE_MM:
            best_enter, origin_enter = span[0], anchor + offset
        if best_exit is None or span[1] > best_exit + _SPAN_TIE_MM:
            best_exit, origin_exit = span[1], anchor + offset
    if best_enter is None or best_exit is None:
        return None
    assert origin_enter is not None and origin_exit is not None
    return (float(best_enter), float(best_exit),
            anchor + best_enter * axis, anchor + best_exit * axis)


def _hits_any(points: np.ndarray, obstacles: Sequence[Solid], margin_mm: float) -> Solid | None:
    """First obstacle any sample point is inside, or ``None``.

    Two-level broad phase, because this is the hot loop of the whole evaluation: one sphere test for
    the cloud against each obstacle (no per-point work at all for anything far away), then a per-point
    sphere test before the exact containment.
    """
    if points.size == 0:
        return None
    cloud_centre = points.mean(axis=0)
    cloud_radius = float(np.linalg.norm(points - cloud_centre, axis=1).max())
    for solid in obstacles:
        reach = solid.bounding_radius_mm() + margin_mm
        if float(np.linalg.norm(solid.centre_mm - cloud_centre)) > cloud_radius + reach:
            continue
        near = np.linalg.norm(points - solid.centre_mm, axis=1) <= reach
        if not near.any():
            continue
        if solid.contains_many(points[near], margin_mm=margin_mm).any():
            return solid
    return None


def jaw_contact_patch(
    grasp: JawGrasp, target: Solid, *, model: JawModel | None = None,
) -> np.ndarray:
    """Every point on ``target`` the two pads touch: the contact patch, not two points.

    A public seam over the geometry `_pad_contacts` already walks, because the cloud corpus needs
    the contacts and both cheaper answers are wrong.

    `position_mm + axis * width/2` is wrong because `position_mm` is the anchor: this file has a
    whole rejection reason, `ANCHOR_OUTSIDE`, for the anchor not lying between the contacts, and a
    contact derived that way can land off the surface.

    `_pad_contacts`'s own two return points are on the surface but sit at whichever of the nine pad
    sample positions reached furthest, a corner of the patch rather than its centre. On a 40 mm box
    grasped top-down that is 13.5 mm off in the binormal and 14.4 mm along the approach, which for a
    graspability field with a 10 mm radius means marking one corner of a 27 x 38 mm pad.

    So this returns every pad position's contact, on both sides. Empty when the line misses the
    object.
    """
    model = model or JawModel()
    approach = _unit(grasp.approach)
    axis = _unit(grasp.closing_axis)
    binormal = np.cross(approach, axis)
    binormal = binormal / max(float(np.linalg.norm(binormal)), _EPS)
    anchor = np.asarray(grasp.position_mm, dtype=np.float64)

    half_width = model.finger_width_mm / 2.0
    # The same nine offsets `_pad_contacts` uses, including its asymmetry: the pad runs
    # `pad_behind_mm` behind the grasp centre and `pad_ahead_mm` in front, not equal amounts.
    offsets = [b * binormal + a * approach
               for b in (-half_width, 0.0, half_width)
               for a in (-model.pad_behind_mm, 0.0, model.pad_ahead_mm)]
    points: list[np.ndarray] = []
    for offset in offsets:
        origin = anchor + offset
        span = target.line_span(origin, axis)
        if span is None:
            continue
        points.append(origin + span[0] * axis)
        points.append(origin + span[1] * axis)
    return np.asarray(points, dtype=np.float64) if points else np.zeros((0, 3), dtype=np.float64)



#: Jaws that are not the 2F-85, for the one job of making the gripper input vary.
#:
#: These are not grippers this project owns, and no result may be read as gripper generalisation.
#: The learned generator takes a gripper description as an input, and with one gripper that input is
#: the same fourteen numbers in every sample: constant, therefore carrying no gradient, therefore a
#: seam that looks wired while doing nothing. A small subset of scenes labelled with these produces
#: the variation a differential measurement needs, and nothing more.
#:
#: They are the 2F-85's own measured proportions scaled by aperture, except `slim_pad`, which holds
#: the aperture fixed and halves the contact patch. That one is deliberate: two of the fourteen
#: numbers describe the pad, `too_wide` is the largest single label rejection, and a variation set
#: where every jaw differed only in aperture could not tell the pad channels apart from the width
#: ones.
PROCEDURAL_JAWS: Final[dict[str, JawModel]] = {
    "narrow_55": JawModel(
        aperture_mm=55.0, finger_ahead_mm=18.58, finger_behind_mm=21.59,
        finger_thickness_mm=20.29, finger_width_mm=17.47,
        pad_ahead_mm=15.28, pad_behind_mm=9.31),
    "wide_140": JawModel(
        aperture_mm=140.0, finger_ahead_mm=47.30, finger_behind_mm=54.96,
        finger_thickness_mm=51.63, finger_width_mm=44.47,
        pad_ahead_mm=38.88, pad_behind_mm=23.70),
    "slim_pad": JawModel(pad_ahead_mm=11.81, pad_behind_mm=7.20),
}


def check_jaw_grasp(
    grasp: JawGrasp,
    target: Solid,
    obstacles: Sequence[Solid] = (),
    *,
    model: JawModel | None = None,
    table_z_mm: float = TABLE_Z_MM,
) -> Verdict:
    """Would this jaw grasp physically close on ``target``? Checked against the scene, not a label."""
    model = model or JawModel()
    approach = _unit(grasp.approach)
    axis = _unit(grasp.closing_axis)
    tilt = _tilt_from_down_deg(approach)

    perpendicularity = abs(float(axis @ approach))
    if perpendicularity > 0.02:          # ~1.1 deg; the GraspPoint contract already promises this
        return Verdict(False, Rejection.AXIS_NOT_PERPENDICULAR, approach_tilt_deg=tilt,
                       detail=f"axis.approach = {perpendicularity:.3f}")

    position = np.asarray(grasp.position_mm, dtype=np.float64)
    binormal = np.cross(approach, axis)
    binormal = binormal / max(float(np.linalg.norm(binormal)), _EPS)
    contacts = _pad_contacts(target, position, axis, approach, binormal, model)
    if contacts is None:
        return Verdict(False, Rejection.LINE_MISSES_OBJECT, approach_tilt_deg=tilt)
    t_enter, t_exit, contact_b, contact_a = contacts
    object_span = float(t_exit - t_enter)
    if not (t_enter - model.anchor_tolerance_mm <= 0.0 <= t_exit + model.anchor_tolerance_mm):
        return Verdict(False, Rejection.ANCHOR_OUTSIDE, object_span_mm=object_span,
                       approach_tilt_deg=tilt,
                       detail=f"anchor at t=0 outside [{t_enter:.1f}, {t_exit:.1f}] mm")
    if object_span > model.aperture_mm:
        return Verdict(False, Rejection.TOO_WIDE, object_span_mm=object_span, approach_tilt_deg=tilt,
                       detail=f"{object_span:.1f} mm > {model.aperture_mm:.0f} mm aperture")
    if object_span < model.min_width_mm:
        return Verdict(False, Rejection.TOO_THIN, object_span_mm=object_span, approach_tilt_deg=tilt)

    # Antipodal: both contact normals must lie inside the friction cone about the closing axis.
    normal_a = target.normal_at(contact_a)
    normal_b = target.normal_at(contact_b)
    angle_a = float(np.degrees(np.arccos(np.clip(float(normal_a @ axis), -1.0, 1.0))))
    angle_b = float(np.degrees(np.arccos(np.clip(float(normal_b @ -axis), -1.0, 1.0))))
    worst = max(angle_a, angle_b)
    if worst > model.friction_half_angle_deg:
        return Verdict(False, Rejection.NOT_ANTIPODAL, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail=f"{worst:.1f} deg outside the {model.friction_half_angle_deg:.1f} deg cone")

    half_thickness = model.finger_thickness_mm / 2.0

    # At contact: the fingers stand just outside the two contact points.
    closed = np.concatenate([
        _finger_samples(contact_a + half_thickness * axis, approach, binormal, model),
        _finger_samples(contact_b - half_thickness * axis, approach, binormal, model),
    ])
    # Clearance under the fingertip, and not half the finger thickness: the pad's thickness extends
    # along the closing axis, which for a top-down grasp is horizontal, so subtracting it from a
    # vertical clearance is wrong and puts every low object out of reach. A real gripper may set its
    # fingertip on the table; it may only not go through it.
    if float(closed[:, 2].min()) < table_z_mm + _TABLE_CLEARANCE_MM:
        return Verdict(False, Rejection.BELOW_TABLE, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail=f"lowest finger sample at z={closed[:, 2].min():.1f} mm")
    blocker = _hits_any(closed, obstacles, half_thickness)
    if blocker is not None:
        return Verdict(False, Rejection.FINGER_COLLISION, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail=f"finger inside {blocker.asset_id or blocker.instance_id}")
    # The target itself: the fingers may touch it only at the pads, which the offset above already
    # cleared. Anything deeper means the jaw is inside the object it is meant to grip.
    if target.contains_many(closed, margin_mm=-0.5).any():
        return Verdict(False, Rejection.FINGER_COLLISION, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail="finger inside the target itself")

    # Travelling in: the fingers sit wider and the whole envelope moves along +approach to arrive.
    # The sweep needs no stepping. A finger translating along its own length axis sweeps exactly one
    # longer finger, so the union of every intermediate pose is a single sample cloud of length
    # (finger + clearance): identical coverage at a fraction of the cost.
    open_half = min(model.aperture_mm, object_span + 2.0 * model.approach_opening_margin_mm) / 2.0
    reach = model.finger_ahead_mm + model.finger_behind_mm + model.approach_clearance_mm
    swept = np.concatenate([
        _finger_samples(position + open_half * axis, approach, binormal, model, length_mm=reach),
        _finger_samples(position - open_half * axis, approach, binormal, model, length_mm=reach),
    ])
    # The path in may not pass under the table. Discarding every below-table sample here and leaving
    # it to `BELOW_TABLE` does not cover this: that check tests the closed fingertips at the grasp
    # point, not the corridor the gripper travels down. A corridor that reaches under the table is
    # the gripper being teleported through it.
    #
    # A distinct reason rather than a wider `BELOW_TABLE`, so that a `below_table` count stays
    # comparable across runs. Behind the grasp point only: for a downward approach the part that
    # dips under the table is the fingertip ahead of the grasp, and a real gripper may set a
    # fingertip on the table, which `BELOW_TABLE` already governs with a tolerance a few lines
    # above. What is impossible is the gripper body having to stand under the support plane before
    # it moves in.
    behind = swept[(swept - position) @ approach < 0.0]
    if behind.size and float(behind[:, 2].min()) < table_z_mm:
        lowest = float(behind[:, 2].min())
        return Verdict(False, Rejection.APPROACH_UNDER_TABLE, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail=f"the gripper would start {table_z_mm - lowest:.1f} mm under the table")
    blocker = _hits_any(swept, obstacles, half_thickness)
    if blocker is not None:
        return Verdict(False, Rejection.APPROACH_BLOCKED, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail=f"path crosses {blocker.asset_id or blocker.instance_id}")
    # The target is checked too, and it is not redundant: the opening is sized from the span at the
    # contact line, so approaching a shape that is wider further along, the flat of a bowl-shaped
    # cuboid or a tilted plate, would plough the fingers through it on the way in.
    if target.contains_many(swept, margin_mm=-0.5).any():
        return Verdict(False, Rejection.APPROACH_BLOCKED, object_span_mm=object_span,
                       contact_angle_deg=worst, approach_tilt_deg=tilt,
                       detail="path crosses the target itself")

    return Verdict(True, Rejection.NONE, object_span_mm=object_span, contact_angle_deg=worst,
                   approach_tilt_deg=tilt)


def check_suction_grasp(
    grasp: SuctionGrasp,
    target: Solid,
    obstacles: Sequence[Solid] = (),
    *,
    cup: SuctionCup | None = None,
    table_z_mm: float = TABLE_Z_MM,
) -> Verdict:
    """Can the cup seal here? Geometry only: flatness, orientation, clearance.

    Deliberately not a seal model. ``AnalyticalSuctionScorer`` is the thing this reference grades,
    so a reference built on seal x wrench would be marking its own homework. Whether the cup could
    lift the mass is reported separately by :func:`suction_payload_ok`, from textbook pressure x
    area.
    """
    cup = cup or STANDARD_CUP
    approach = _unit(grasp.approach)
    position = np.asarray(grasp.position_mm, dtype=np.float64)
    tilt = _tilt_from_down_deg(approach)

    # The contact must be on the surface: a hair inside is fine, floating is not.
    if not target.contains(position, margin_mm=0.5):
        return Verdict(False, Rejection.CONTACT_OFF_SURFACE, approach_tilt_deg=tilt)
    normal = target.normal_at(position)
    deviation = float(np.degrees(np.arccos(np.clip(float(-approach @ normal), -1.0, 1.0))))
    if deviation > cup.max_normal_deviation_deg:
        return Verdict(False, Rejection.APPROACH_NOT_NORMAL, contact_angle_deg=deviation,
                       approach_tilt_deg=tilt,
                       detail=f"{deviation:.1f} deg off the surface normal")

    # The rim: every sample point must find the same surface, at the same height along the normal.
    tangent = np.cross(normal, np.array([0.0, 0.0, 1.0]))
    if float(np.linalg.norm(tangent)) < 1e-3:
        tangent = np.cross(normal, np.array([1.0, 0.0, 0.0]))
    tangent = tangent / float(np.linalg.norm(tangent))
    bitangent = np.cross(normal, tangent)
    radius = cup.diameter_mm / 2.0
    angles = np.linspace(0.0, 2.0 * np.pi, cup.rim_samples, endpoint=False)
    rim = (position[None, :]
           + radius * (np.cos(angles)[:, None] * tangent[None, :]
                       + np.sin(angles)[:, None] * bitangent[None, :]))
    # How far along the normal each rim point would have to move to reach the surface. Off the edge of
    # a face there is no surface to reach and the ray simply misses.
    for point in rim:
        start = point + normal * (cup.flatness_tolerance_mm + 1.0)
        hit = target.line_span(start, -normal)
        if hit is None:
            return Verdict(False, Rejection.SEAL_NOT_FLAT, contact_angle_deg=deviation,
                           approach_tilt_deg=tilt, detail="rim overhangs the surface")
        gap = float(hit[0]) - (cup.flatness_tolerance_mm + 1.0)
        if abs(gap) > cup.flatness_tolerance_mm:
            return Verdict(False, Rejection.SEAL_NOT_FLAT, contact_angle_deg=deviation,
                           approach_tilt_deg=tilt,
                           detail=f"rim {gap:+.2f} mm off the contact plane")

    steps = np.arange(0.0, cup.approach_clearance_mm + 4.0, 4.0)
    swept = np.concatenate([rim - step * approach for step in steps] + [rim])
    swept = swept[swept[:, 2] >= table_z_mm]
    blocker = _hits_any(swept, obstacles, 0.0)
    if blocker is not None:
        return Verdict(False, Rejection.CUP_COLLISION, contact_angle_deg=deviation,
                       approach_tilt_deg=tilt,
                       detail=f"cup path crosses {blocker.asset_id or blocker.instance_id}")
    return Verdict(True, Rejection.NONE, contact_angle_deg=deviation, approach_tilt_deg=tilt)


def suction_payload_ok(mass_kg: float, cup: SuctionCup, *, safety_factor: float = 2.0) -> bool:
    """Pressure x area against mass x g. Textbook physics, kept out of the geometric verdict."""
    return cup.holding_force_n() >= safety_factor * mass_kg * 9.81
