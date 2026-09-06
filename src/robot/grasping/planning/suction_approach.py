"""Approach geometry for suction grasps, which have a single contact.

The parallel-jaw helpers in :mod:`approach_planner` build pre-grasp, retreat and waypoint poses
from a full :class:`GraspPose`, with its closing axis, its roll and its two contacts. A suction
grasp has none of that. It is one sealable contact plus a press direction, and the cup is
axisymmetric, so the roll about that direction is free. This module is the one home for the
suction approach recipe: the free-roll tool orientation together with the standoff, contact and
retreat poses that the sim runners would otherwise each rebuild by hand.

Everything is pure NumPy over :mod:`src.geometry`, with no driver, no motion planner and no
Isaac. Distances are millimetres, rotations are XYZW quaternions, and every pose carries a
frame tag.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.geometry import Frame, Pose, from_rotation_matrix
from src.robot.grasping.geometry import as_vec3

if TYPE_CHECKING:  # typing only, which avoids a runtime import from planning into suction
    from src.robot.grasping.suction.synthesis import SuctionGrasp

__all__ = [
    "SuctionApproach",
    "suction_approach_waypoints",
    "suction_grasp_poses",
    "suction_pose",
    "suction_tool_orientation",
]

_EPS = 1e-9


def _unit_vec3(value: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    """Return a finite, unit-length ``(3,)`` copy of ``value`` or raise ``ValueError``."""
    vec = as_vec3(value, name)
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        raise ValueError(f"{name} cannot be the zero vector")
    return vec / norm


def _nonneg(value: float, name: str) -> float:
    scalar = float(value)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return scalar


def _bridge_frame(grasp_frame: object) -> Frame:
    """Map a suction ``GraspFrame``, either ``'camera'`` or ``'base'``, onto the geometry :class:`Frame`.

    Both are string enums that share the same member values, so the bridge is a value lookup;
    a plain string or an already-``Frame`` value is accepted too.
    """
    return Frame(str(grasp_frame))


def suction_tool_orientation(approach: np.ndarray | Sequence[float]) -> np.ndarray:
    """Return a unit XYZW quaternion for an axisymmetric cup pressing along ``approach``.

    The cup is symmetric about its own axis, so the roll around the approach direction is free:
    tool ``+Z`` is fixed to the unit approach and tool ``+X`` is any stable perpendicular. The
    rotation columns follow the grasp convention ``[closing, binormal, approach]``, the same one
    :class:`~src.robot.grasping.planning.GraspPose` uses, so a suction wrist pose is oriented
    consistently with a parallel-jaw one.
    """
    axis = _unit_vec3(approach, "approach")
    # A stable reference that is not parallel to the approach, projected perpendicular for +X.
    reference = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    closing = reference - axis * float(np.dot(reference, axis))
    closing_norm = float(np.linalg.norm(closing))
    if closing_norm < _EPS:  # unreachable for the reference rule above; guarded for safety
        raise ValueError("cannot build a tool frame: approach is parallel to the reference axis")
    closing /= closing_norm
    binormal = np.cross(axis, closing)
    rotation = np.column_stack((closing, binormal, axis))
    return np.asarray(from_rotation_matrix(rotation), dtype=np.float64)


def suction_pose(
    position_mm: np.ndarray | Sequence[float],
    approach: np.ndarray | Sequence[float],
    *,
    frame: Frame = Frame.BASE,
    standoff_mm: float = 0.0,
    label: str = "suction",
) -> Pose:
    """Return the cup pose ``standoff_mm`` back from the contact along ``-approach``.

    ``standoff_mm == 0`` is the contact pose itself; a positive value hovers the cup above the
    surface (a pre-grasp), and a large value is a post-seal retreat. The orientation is the
    free-roll tool frame from :func:`suction_tool_orientation`.
    """
    origin = as_vec3(position_mm, "position_mm")
    axis = _unit_vec3(approach, "approach")
    distance = _nonneg(standoff_mm, "standoff_mm")
    return Pose(
        position_mm=origin - axis * distance,
        quaternion_xyzw=suction_tool_orientation(axis),
        frame=frame,
        label=label,
    )


def suction_approach_waypoints(
    position_mm: np.ndarray | Sequence[float],
    approach: np.ndarray | Sequence[float],
    *,
    frame: Frame = Frame.BASE,
    standoff_mm: float = 60.0,
    end_offset_mm: float = 0.0,
    num_waypoints: int = 4,
    label_prefix: str = "suction_approach",
) -> list[Pose]:
    """Return ``num_waypoints`` poses linearly interpolating the descent along ``-approach``.

    Inclusive at both ends: the first pose sits ``standoff_mm`` back from the contact and the
    last sits ``end_offset_mm`` back (``0`` = the contact itself). Every pose shares the free-roll
    tool orientation so the controller keeps a fixed wrist through the descent.
    """
    origin = as_vec3(position_mm, "position_mm")
    axis = _unit_vec3(approach, "approach")
    start_distance = _nonneg(standoff_mm, "standoff_mm")
    end_distance = _nonneg(end_offset_mm, "end_offset_mm")
    if num_waypoints < 2:
        raise ValueError("num_waypoints must be >= 2")
    quaternion = suction_tool_orientation(axis)
    start = origin - axis * start_distance
    end = origin - axis * end_distance
    waypoints: list[Pose] = []
    for index, fraction in enumerate(np.linspace(0.0, 1.0, num=num_waypoints)):
        waypoints.append(
            Pose(
                position_mm=start + (end - start) * float(fraction),
                quaternion_xyzw=quaternion.copy(),
                frame=frame,
                label=f"{label_prefix}_{index:02d}",
            )
        )
    return waypoints


@dataclass(frozen=True, slots=True)
class SuctionApproach:
    """The full pose recipe for one suction pick, all in the grasp's frame.

    Order of execution: ``pre_grasp``, then ``waypoints``, then ``contact``, then the seal, then
    ``retreat``. Every pose shares the single free-roll tool orientation ``quaternion_xyzw``.
    """

    pre_grasp: Pose
    contact: Pose
    retreat: Pose
    waypoints: tuple[Pose, ...]
    quaternion_xyzw: np.ndarray


def suction_grasp_poses(
    grasp: SuctionGrasp,
    *,
    standoff_mm: float = 60.0,
    contact_gap_mm: float = 25.0,
    lift_mm: float = 100.0,
    num_waypoints: int = 4,
) -> SuctionApproach:
    """Build the pre-grasp, contact and retreat recipe for a :class:`SuctionGrasp`.

    Reads ``grasp.position_mm`` as the contact point, ``grasp.approach`` as the unit press
    direction, and ``grasp.frame`` as a suction ``GraspFrame`` bridged to the geometry
    :class:`Frame`. The cup hovers ``standoff_mm`` above the surface, descends to
    ``contact_gap_mm`` to seal, then lifts by ``lift_mm``. Feed the poses straight to
    :meth:`RobotArm.move`.
    """
    frame = _bridge_frame(grasp.frame)
    position = np.asarray(grasp.position_mm, dtype=np.float64)
    approach = np.asarray(grasp.approach, dtype=np.float64)
    return SuctionApproach(
        pre_grasp=suction_pose(
            position, approach, frame=frame, standoff_mm=standoff_mm, label="suction_pre_grasp"
        ),
        contact=suction_pose(
            position, approach, frame=frame, standoff_mm=contact_gap_mm, label="suction_contact"
        ),
        retreat=suction_pose(
            position,
            approach,
            frame=frame,
            standoff_mm=contact_gap_mm + _nonneg(lift_mm, "lift_mm"),
            label="suction_retreat",
        ),
        waypoints=tuple(
            suction_approach_waypoints(
                position,
                approach,
                frame=frame,
                standoff_mm=standoff_mm,
                end_offset_mm=contact_gap_mm,
                num_waypoints=num_waypoints,
            )
        ),
        quaternion_xyzw=suction_tool_orientation(approach),
    )
