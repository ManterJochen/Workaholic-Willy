"""The robot's own body, as the self filter takes it back out of what the cameras see.

A fixed camera watching a cell sees the robot, and a robot registered as an obstacle cannot move.
Capsules along the DH origins with two fixed radii do not follow the arm: measured over 20 joint
vectors, 83.4 % of a ur5e's upper arm and 66.7 % of a ur3e's lie outside such a polyline, because the
upper arm is offset from its joint axes.

So the body is built from the files the guard and the planner check against:

* the arm, one capsule per link, fitted once to the committed bundle in the link's own DH frame so
  that it holds every vertex of that link, and carrying the link's own surface, points laid over its
  triangles: a capsule holds a link and a good deal more (the UR10 upper arm's reaches 118 mm about its
  shoulder end at the height of a bin's rim), so a point inside it is the link only within the padding
  of that surface (``perceived.SelfBody``);
* the hand, the spheres of its resolved sphere map, each grown just enough to hold the hand's
  committed mesh, placed on the flange frame the way the guard places the hand: one plate out along
  the model's approach where the hand starts at its mounting face, then turned by the rotation the
  declared tool frame derives. As fitted, the maps leave up to 9.55 mm (2F-85), 16.26 mm (Hand-E) and
  4.45 mm (EGU-50) of their hands outside, the Hand-E's past the padding as well. One capsule for the
  whole hand does not work: its end cap reaches 54 to 77 mm past the fingertips on the approach, where
  it would take real obstacles out;
* a part in the gripper, one capsule from the fingertips along the hand's approach while it is
  attached;
* the robot's base, a cylinder about the base axis from the mounting face up to its top
  (:data:`ROBOT_BASES`), carrying its own surface like the arm links: the bundles hold no base, and a
  camera that sees it registered it as a box the shoulder stood in (fix plan RC3, robot.log 425 and
  448). The guard does not judge the base (the owner, 2026-09-30); only the camera world takes it out;
* all of it placed with the model and the base yaw the self-collision guard uses, so the filter and
  the guard stand the arm in one place.

The padding is added where the world is built, from ``perceived.margin_mm``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import yaml

from src.contracts import chosen

from .._ur_kinematics import ur_link_transforms_mm
from .environment import collision_mesh_bundle, hand_mesh_bundle
from .perceived import DeclaredBody, LinkCapsule, SelfEnvelope

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from src.robot.core.keep_out import GoalKeepOut
    from src.robot.safety.preflight import SafetyPreflight

    from .hand import PlannerHand

__all__ = [
    "ROBOT_BASES", "BaseShape", "arm_capsules", "base_capsule", "carried_part_box", "goal_keep_out", "hand_spheres",
    "payload_capsule", "self_envelope", "yawed_link_transforms_mm",
]


@dataclass(frozen=True, slots=True)
class BaseShape:
    """A robot's base, as the camera world takes it for the robot: a cylinder about the base axis, from the mounting face
    (base z 0) up to its top, which encloses the base's own mesh."""

    #: The cylinder's radius about the base axis, millimetres: the farthest any vertex of the base lies from the axis.
    radius_mm: float
    #: Its top over the mounting face, millimetres: where the shoulder's own housing begins.
    top_mm: float
    #: Where the two numbers were measured.
    provenance: str

    def __post_init__(self) -> None:
        for name in ("radius_mm", "top_mm"):
            value = float(getattr(self, name))
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"a robot base's {name} is finite and above 0, got {value!r}")


#: The robot bases the camera world takes for the robot, by the guard's arm model. A model with no entry keeps its
#: base in the world as before: none is guessed.
ROBOT_BASES: Final[Mapping[str, BaseShape]] = MappingProxyType({
    "ur10": BaseShape(
        radius_mm=95.05, top_mm=38.0,
        provenance=("ur10_base.obj of cuRobo's ur_description (ext_deps/curobo/.../meshes/ur10), measured "
                    "2026-10-01: 3736 vertices, z -0.002 to 38.000 mm, at most 95.05 mm from the base axis "
                    "(four lugs; the round housing 79.2 mm)"),
    ),
})

#: The bundle's arm links and the DH frame each is baked in, as
#: scripts/isaac/bake_ur_collision_meshes.py writes them.
_ARM_LINKS: Final = (
    ("shoulder", 1), ("upper_arm", 2), ("forearm", 3), ("wrist_1", 4), ("wrist_2", 5), ("wrist_3", 6),
)
#: The flange frame, where the hand and a carried part hang.
_FLANGE_FRAME: Final = 6
#: Added to a fitted radius so a vertex exactly on the surface is not lost to rounding, millimetres.
_ROUNDING_MM: Final = 1e-6
#: How far any point of an arm link's surface lies at most from the nearest point laid over it,
#: millimetres. The filter takes a point within the padding of the nearest laid point, so the link's
#: own surface leaves the world wherever the padding (``perceived.margin_mm``, 15) is at least this, and
#: a point the camera placed up to the padding less this off the link leaves with it. A choice.
_SURFACE_SPACING_MM: Final = 4.0
#: The bundle arrays that are the hand, and the stamp that puts them one plate out, as the
#: self-collision guard reads them (src/robot/safety/_fcl_self_collision.py).
_HAND_PARTS: Final = ("gripper", "lfinger", "rfinger")
_ORIGIN_KEY: Final = "gripper__origin"
_MOUNTING_FACE: Final = "mounting_face"
#: The approach of every committed hand model in its own axes (model y), along which a coupling plate
#: stacks.
_MODEL_APPROACH: Final = (0.0, 1.0, 0.0)
_IDENTITY_ROWS: Final = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _point(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _fit_link(vertices: np.ndarray, frame: int) -> LinkCapsule:
    """The tightest capsule along a link's long axis that holds every one of its vertices.

    The radius is the largest distance from the axis. Each end is then pulled in as far as the
    vertices near it allow, so the round caps do not reach a link radius past the link and take a real
    obstacle beside a joint out.
    """
    centre = vertices.mean(axis=0)
    centred = vertices - centre
    _, vectors = np.linalg.eigh(centred.T @ centred)
    axis = vectors[:, -1]
    along = centred @ axis
    radial = np.linalg.norm(centred - np.outer(along, axis), axis=1)
    radius = float(radial.max())
    reach = np.sqrt(np.maximum(radius * radius - radial * radial, 0.0))
    upper = float(np.max(along - reach))
    lower = float(np.min(along + reach))
    if lower > upper:
        # A link shorter than it is wide: one point between the two limits holds every vertex.
        lower = upper = (lower + upper) / 2.0
    # The rounding margin goes on after the ends are pulled in. Added before, the pull uses it up, and
    # the vertex that set each end sits exactly on the cap, where rounding puts some of them outside.
    return LinkCapsule(
        frame=frame, start_mm=_point(centre + axis * lower), end_mm=_point(centre + axis * upper),
        radius_mm=radius + _ROUNDING_MM,
    )


def _link_surface(name: str, vertices: np.ndarray, faces: np.ndarray) -> DeclaredBody:
    """A link's surface in its own frame, as points no point of its triangles lies further than the spacing from.

    Each triangle is laid in rows parallel to its longest edge, the rows and the points along each row
    :data:`_SURFACE_SPACING_MM` over the square root of two apart. The angles at the longest edge are acute, so a
    point of the triangle stands over the row below it within that row's ends: at most one row from it and half a
    step along it from a laid point, 0.79 of the spacing in all. A long thin triangle, which the bundles' tubes are made of, costs points by
    its area and not by the square of its length, as a grid laid across it would (2026-09-30: 188,000 points
    over the UR10's six links, laid in 0.6 s).
    """
    step = _SURFACE_SPACING_MM / math.sqrt(2.0)
    laid: list[np.ndarray] = []
    for corners in np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]:
        edges = np.linalg.norm(np.roll(corners, -1, axis=0) - corners, axis=1)
        longest = int(np.argmax(edges))
        first, second, apex = corners[longest], corners[(longest + 1) % 3], corners[(longest + 2) % 3]
        if float(edges[longest]) <= 0.0:
            laid.append(first[None, :])
            continue
        height = float(np.linalg.norm(np.cross(second - first, apex - first))) / float(edges[longest])
        for share in np.linspace(0.0, 1.0, max(1, math.ceil(height / step)) + 1):
            start, end = first + share * (apex - first), second + share * (apex - second)
            count = max(1, math.ceil(float(np.linalg.norm(end - start)) / step))
            laid.append(start + np.linspace(0.0, 1.0, count + 1)[:, None] * (end - start))
    points = np.unique(np.round(np.concatenate(laid), 6), axis=0)
    return DeclaredBody(name=name, surface_mm=points, spacing_mm=_SURFACE_SPACING_MM)


def _cylinder_surface(name: str, radius_mm: float, top_mm: float) -> DeclaredBody:
    """A base's surface in the base frame, its top disc and its side down to the mounting face, as points no point of
    it lies further than :data:`_SURFACE_SPACING_MM` from, laid as the links' are, rows and points along them the spacing
    over the square root of two apart.

    The face it stands on is left out: under it is the plate, which the bench band takes, and no camera sees it.
    """
    step = _SURFACE_SPACING_MM / math.sqrt(2.0)
    laid: list[np.ndarray] = [np.array([[0.0, 0.0, top_mm]])]
    for ring in np.linspace(0.0, radius_mm, max(1, math.ceil(radius_mm / step)) + 1)[1:]:
        count = max(3, math.ceil(2.0 * math.pi * float(ring) / step))
        angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
        laid.append(np.column_stack((ring * np.cos(angles), ring * np.sin(angles), np.full(count, top_mm))))
    count = max(3, math.ceil(2.0 * math.pi * radius_mm / step))
    angles = np.linspace(0.0, 2.0 * math.pi, count, endpoint=False)
    for height in np.linspace(0.0, top_mm, max(1, math.ceil(top_mm / step)) + 1):
        laid.append(np.column_stack((radius_mm * np.cos(angles), radius_mm * np.sin(angles), np.full(count, height))))
    return DeclaredBody(name=name, surface_mm=np.concatenate(laid), spacing_mm=_SURFACE_SPACING_MM)


@lru_cache(maxsize=None)
def base_capsule(model: str) -> LinkCapsule | None:
    """``model``'s base on the base frame (frame 0), carrying its own surface, or ``None`` where :data:`ROBOT_BASES`
    holds no base for it.

    The capsule runs up the base axis from the mounting face to the base's top with the base's radius, so it holds the
    whole cylinder; a point inside it is the base only within the padding of the cylinder's surface
    (``perceived.SelfBody``), as a point beside an arm link is that link only within the padding of the link's.
    """
    shape = ROBOT_BASES.get(str(model).lower())
    if shape is None:
        return None
    return LinkCapsule(
        frame=0, start_mm=(0.0, 0.0, 0.0), end_mm=(0.0, 0.0, float(shape.top_mm)),
        radius_mm=float(shape.radius_mm) + _ROUNDING_MM,
        surface=_cylinder_surface("base", float(shape.radius_mm), float(shape.top_mm)),
    )


@lru_cache(maxsize=None)
def arm_capsules(model: str) -> tuple[LinkCapsule, ...] | None:
    """One capsule per arm link of ``model``, in each link's DH frame, or ``None`` without a committed bundle.

    Each carries its link's surface (:func:`_link_surface`) where the bundle holds the link's triangles.
    """
    bundle = collision_mesh_bundle(model)
    if not bundle.is_file():
        return None
    with np.load(bundle) as data:
        if any(f"{name}__v" not in data.files for name, _ in _ARM_LINKS):
            return None
        capsules: list[LinkCapsule] = []
        for name, frame in _ARM_LINKS:
            vertices = np.asarray(data[f"{name}__v"], dtype=np.float64)
            capsule = _fit_link(vertices, frame)
            if f"{name}__f" in data.files:
                capsule = replace(capsule, surface=_link_surface(name, vertices, np.asarray(data[f"{name}__f"])))
            capsules.append(capsule)
        return tuple(capsules)


@lru_cache(maxsize=16)
def _map_spheres(path: str) -> tuple[tuple[tuple[float, float, float], float], ...]:
    spheres = yaml.safe_load(Path(path).read_text(encoding="utf-8"))["collision_spheres"]["tool0"]
    return tuple(
        ((float(s["center"][0]) * 1000.0, float(s["center"][1]) * 1000.0, float(s["center"][2]) * 1000.0),
         float(s["radius"]) * 1000.0)
        for s in spheres
    )


@lru_cache(maxsize=16)
def _held_spheres(
    sphere_map: str, bundle: str, coupling_mm: float, rotation: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[tuple[float, float, float], float], ...] | None:
    """The map's spheres on the flange, placed as the guard places the hand, each grown to hold the hand's vertices.

    In the hand model's own axes first: one plate out along its approach where the bundle starts at
    the mounting face, and each sphere grown to hold the vertices nearest it. Then the placement's
    rotation turns the centres; distances do not change under a rotation, so the radii are the
    model's. The identity turns nothing.
    """
    path = Path(bundle)
    if not path.is_file():
        return None
    with np.load(path) as data:
        if any(f"{part}__v" not in data.files for part in _HAND_PARTS):
            return None
        vertices = np.vstack([np.asarray(data[f"{part}__v"], dtype=np.float64) for part in _HAND_PARTS])
        origin = str(np.asarray(data[_ORIGIN_KEY]).reshape(-1)[0]) if _ORIGIN_KEY in data.files else ""
    shift = np.asarray(_MODEL_APPROACH, dtype=np.float64) * coupling_mm
    if origin == _MOUNTING_FACE:
        vertices = vertices + shift
    spheres = _map_spheres(sphere_map)
    centres = np.asarray([centre for centre, _ in spheres], dtype=np.float64) + shift
    radii = np.asarray([radius for _, radius in spheres], dtype=np.float64)
    distance = np.linalg.norm(vertices[:, None, :] - centres[None, :, :], axis=2)
    # Each vertex belongs to the sphere it is least outside of, so a sphere grows only for the part of
    # the hand it already stands for, and one that already holds its vertices keeps the map's radius.
    nearest = np.argmin(distance - radii[None, :], axis=1)
    grown = radii.copy()
    for index in range(len(radii)):
        held = distance[nearest == index, index]
        if held.size and float(held.max()) > grown[index]:
            grown[index] = float(held.max()) + _ROUNDING_MM
    if rotation != _IDENTITY_ROWS:
        centres = centres @ np.asarray(rotation, dtype=np.float64).T
    return tuple((_point(centre), float(radius)) for centre, radius in zip(centres, grown, strict=True))


def _approach(hand: PlannerHand) -> tuple[float, float, float]:
    """The hand's approach in tool0 axes, as its placement derives it. A refused placement has none."""
    if not chosen(hand.placement):
        raise ValueError(f"{hand.model} has no placement on the flange: {hand.placement_refusal}")
    return hand.placement.approach_in_tool0


def hand_spheres(hand: PlannerHand, model: str) -> tuple[LinkCapsule, ...] | None:
    """The hand on the flange frame as its map's spheres, placed and grown to hold its mesh, or ``None``.

    ``None`` comes back without the hand's bundle, and where the declared tool frame places no hand: a
    body nobody can place is not filtered. ``model`` is the arm the hand hangs from, and the hand's own
    bundle is the same on every arm.
    """
    if not chosen(hand.placement):
        return None
    held = _held_spheres(
        str(hand.sphere_map), str(hand_mesh_bundle(hand.model)), float(hand.coupling_mm), hand.placement.rotation,
    )
    if held is None:
        return None
    return tuple(
        LinkCapsule(frame=_FLANGE_FRAME, start_mm=centre, end_mm=centre, radius_mm=radius) for centre, radius in held
    )


def _tip_mm(spheres: tuple[LinkCapsule, ...], approach: tuple[float, float, float]) -> float:
    """How far the hand reaches along ``approach`` from the flange, millimetres."""
    axis = np.asarray(approach, dtype=np.float64)
    return max(float(np.dot(np.asarray(s.start_mm), axis)) + s.radius_mm for s in spheres)


def payload_capsule(
    hand: PlannerHand, spheres: tuple[LinkCapsule, ...], *, length_mm: float, lateral_margin_mm: float,
) -> LinkCapsule:
    """A carried part: from the tip of ``spheres`` along the hand's approach, as wide as the jaws open plus the margin."""
    approach = np.asarray(_approach(hand), dtype=np.float64)
    tip = _tip_mm(spheres, _approach(hand))
    return LinkCapsule(
        frame=_FLANGE_FRAME,
        start_mm=_point(approach * tip),
        end_mm=_point(approach * (tip + float(length_mm))),
        radius_mm=(float(hand.jaw.aperture_mm) + float(lateral_margin_mm)) / 2.0,
    )


def carried_part_box(
    hand: PlannerHand, spheres: tuple[LinkCapsule, ...], *, grip_width_mm: float, length_mm: float,
    lateral_margin_mm: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """The box the planner hangs on tool0 for a carried part, as ``(dims_mm, centre_mm)`` in tool0's axes.

    It is the part :func:`payload_capsule` takes out of the camera's view: ``length_mm`` along the
    hand's approach from the tip of ``spheres``, and on the two other axes the width the jaws closed to
    plus the lateral margin, the one measurement of the part there is. The approach follows the hand's
    placement: a box hung ``length_mm`` along tool0 +Z from the flange would sit 90 degrees off the
    measured hand and inside the wrist.
    """
    approach = np.asarray(_approach(hand), dtype=np.float64)
    lateral = max(1.0, float(grip_width_mm) + float(lateral_margin_mm))
    dims = np.where(np.abs(approach) > 0.5, float(length_mm), lateral)
    return _point(dims), _point(approach * (_tip_mm(spheres, _approach(hand)) + float(length_mm) / 2.0))


def yawed_link_transforms_mm(model: str, joints: Any, yaw_deg: float) -> list[np.ndarray] | None:
    """Every frame of ``model``'s chain at ``joints``, turned about the base by ``yaw_deg`` as the guard turns it."""
    transforms = ur_link_transforms_mm(model, np.asarray(joints, dtype=np.float64))
    if transforms is None:
        return None
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    base = np.eye(4, dtype=np.float64)
    base[:2, :2] = [[c, -s], [s, c]]
    return [base @ transform for transform in transforms]


def self_envelope(
    preflight: SafetyPreflight, arm: Any, joints: Any, *, payload: tuple[float, float] | None = None,
) -> SelfEnvelope | None:
    """``arm``'s own body at ``joints``, or ``None`` where it cannot be described.

    ``payload`` is ``(length_mm, lateral_margin_mm)`` while a part is attached. ``None`` comes back
    when the guard resolves no model, the model has no committed bundle, no hand is named, or the
    declared tool frame places no hand (``hand_spheres``): the world then refuses rather than
    filtering a body it cannot place.
    """
    kinematics = preflight.self_kinematics(arm)
    if kinematics is None:
        return None
    model, yaw_deg = kinematics
    frames = yawed_link_transforms_mm(model, joints, yaw_deg)
    capsules = arm_capsules(model)
    hand = preflight.planner_hand(arm)
    if frames is None or capsules is None or not chosen(hand):
        return None
    spheres = hand_spheres(hand, model)
    if spheres is None:
        return None
    body = capsules + spheres
    # The robot's base, where its shape is known: a camera that sees it otherwise registers a box the shoulder stands in.
    base = base_capsule(model)
    if base is not None:
        body = body + (base,)
    # A wrist camera's housing and bracket, the fill the planner carries, so a fixed camera does not
    # register the housing as an obstacle beside the arm.
    body = body + tuple(
        LinkCapsule(frame=_FLANGE_FRAME, start_mm=centre, end_mm=centre, radius_mm=radius)
        for wrist in preflight.wrist_bodies(arm) for centre, radius in wrist.envelope_spheres_mm()  # type: ignore[attr-defined]
    )
    if payload is not None:
        body = body + (payload_capsule(hand, spheres, length_mm=payload[0], lateral_margin_mm=payload[1]),)
    # The base's shape also rides on its own, so the support detection leaves the disc round the base out.
    return SelfEnvelope(frames_mm=tuple(frames), capsules=body, base=ROBOT_BASES.get(str(model).lower()))


def goal_keep_out(preflight: Any, arm: Any, tcp_to_base_mm: Any) -> "GoalKeepOut":
    """The space between ``arm``'s jaws at a goal whose TCP stands at ``tcp_to_base_mm``, or why none.

    Laid out by the hand's registry jaw (``KeepOutBox.from_jaw``) on the declared TCP, the frame the
    verb, the planner and the grasp calculator all place the goal with. There is no region for a hand
    that is not a ``PlannerHand``, so a guard double that answers anything to every question builds
    none; for a hand whose declared tool frame places no hand on the flange; and for a goal whose TCP
    is not known. Each says why.
    """
    from src.robot.core.keep_out import GoalKeepOut, KeepOutBox

    from .hand import PlannerHand

    hand = None if preflight is None else preflight.planner_hand(arm)
    if not isinstance(hand, PlannerHand):
        return GoalKeepOut(region=None, reason=(
            "this arm's planner reads no hand, so no space between its jaws is left out at the goal"))
    if not chosen(hand.placement):
        return GoalKeepOut(region=None, reason=(
            f"the hand {hand.model} has no placement on the flange, so no goal region is placed: "
            f"{hand.placement_refusal}"))
    if tcp_to_base_mm is None:
        return GoalKeepOut(region=None, reason="the TCP at this goal is not known, so no goal region is placed")
    jaw = hand.jaw
    return GoalKeepOut(region=KeepOutBox.from_jaw(
        aperture_mm=jaw.aperture_mm, finger_width_mm=jaw.finger_width_mm, pad_ahead_mm=jaw.pad_ahead_mm,
        pad_behind_mm=jaw.pad_behind_mm, tcp_to_base_mm=tcp_to_base_mm,
    ))
