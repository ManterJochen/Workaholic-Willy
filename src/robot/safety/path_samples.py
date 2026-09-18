"""Turn a motion into the configurations a checker will judge, with the step derived from the geometry.

An endpoint check judges where a move ends. A move that grazes a bin wall halfway and lands clear
passes every endpoint check there is, which is why a planned path has to be judged as a path. To judge
it, the middle has to exist as configurations somebody can hand to a checker, and that is all this
module does: it produces them, and it states the bound it kept.

The step rule is geometric rather than a sample count:

* a joint step keeps ``sum(|dq_j| * radius_j)`` at or below the caller's bound, where the radius of a
  joint is an upper bound on the distance from its axis to anything beyond it. Each term bounds the arc
  that joint sweeps, so the sum bounds the travel of any point of the arm. One radius for the whole arm
  is allowed and is what a caller with no table gets; it is the shoulder's radius applied to the wrist
  as well, which on a ur5e samples an ordinary move about 1.8 times more densely than it has to;
* a line step keeps the travel of the tool at or below the bound, and the rotation over the reach at or
  below the same bound, so a move that turns without travelling is still sampled by how far the far end
  of the arm swings.

Both are upper bounds, so the gap a checker cannot see is bounded by a number the caller chose. Nothing
here is ever thinned to fit: thinning is exactly the silent gap this module exists to remove, so a path
too long for the backstop is refused with a sentence instead. The backstop is about how long a caller
waits and not about the size of a checker's request, which is a separate number the client splits along.

Pure functions over numpy, no config, no logger, no planner: what a caller does with the samples is the
caller's business, and the same samples serve the cuRobo batch check and the exact mesh gate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.geometry.frame import Frame
from src.geometry.pose import Pose
from src.geometry.quaternion import angle_between, slerp

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "LINE_MAX_CHORD_OFF_MM",
    "LINE_MAX_JOINT_STEP_RAD",
    "MAX_PATH_SAMPLES",
    "LineSamples",
    "PathSamples",
    "joint_path_samples",
    "line_samples",
    "waypoint_path_samples",
]

#: The most configurations one checked path may hold, and it is a backstop rather than a policy.
#:
#: It is not the cuRobo client's request size. One ``check_js`` request carries 1000 configurations,
#: which says nothing about how long a move may be: the client splits a longer path across requests,
#: and the local exact mesh gate is a loop with no limit at all. A cap of 1000 here refused about one
#: real cuRobo plan in ten on the M2 Isaac gate, plans nothing prevented anybody from checking.
#:
#: What is left is a bound on how long a caller waits. At about 2.2 ms a sample this is roughly 44 s
#: of checking before the arm starts moving, which is already unreasonable, so a path that reaches it
#: is a path somebody should have planned in legs. It is not a number to tune: if a real move ever
#: reaches it, the thing to change is the move or the margin, not this.
MAX_PATH_SAMPLES = 20_000

#: How far one joint may turn between two neighbouring samples of a checked line before the line is
#: taken to have left the arm's branch. On the Isaac dense cell a 10 mm step turns 0.03 rad and the
#: jump into the other wrist family 3.14 rad; on URSim the controller's own IK turns the elbow and
#: wrist 1 by about 0.026 rad a step, against each other. Both drivers read this one number.
#:
#: It is not the path gate's step bound. The sum |dq_j| * r_j bounds how far any point of the arm can
#: move, and it adds two joints turning against each other as if both carried the arm the same way:
#: on URSim a continuous 50 mm lift sums to 27 to 31 mm a step while no link origin moves more than
#: 10 mm, so a line gate reading the sum refuses every step as a branch change. A step the sum cannot
#: vouch for is filled in joint space instead.
LINE_MAX_JOINT_STEP_RAD = 0.35

#: How far the straight joint-space line between two neighbouring solutions of a controller-run line
#: may put the flange off the line the controller runs, measured at its midpoint. A joint jump is not
#: the only branch change: near a stretched elbow the two elbow branches are 0.30 rad apart and reach
#: the same flange pose, and nothing turns more than ``LINE_MAX_JOINT_STEP_RAD``. On the ur5e chain a
#: continuous 10 mm step of the URSim line puts the midpoint 0.03 mm off, a 35 mm shoulder step
#: 0.22 mm, and that elbow flip 2.29 mm. It is the check that a judged fill describes what moveL
#: executes, so it belongs where a controller runs the line; where the driver walks the joint line
#: itself, the fill is what executes.
LINE_MAX_CHORD_OFF_MM = 1.0

#: Absorbs the float noise of an exact multiple: ``999.0`` computed as ``(999 / 850) * 850`` can land
#: a machine epsilon above, and a plain ceil would then ask for one step more than the geometry needs.
#: The cost is that a step may exceed the bound by a part in a trillion, which no checker can see.
_EXACT_MULTIPLE_SLACK = 1e-9


@dataclass(frozen=True, slots=True)
class PathSamples:
    """The joint configurations of one move, both ends included, and the step they kept.

    ``step_bound_mm`` is the largest distance any point of the arm can travel between two neighbouring
    configurations. It is the number a reader needs to know what the check bought: a refusal means the
    path is unsafe, and a pass means the path is safe to within this much unexamined travel.
    """

    configs: tuple[tuple[float, ...], ...]
    step_bound_mm: float


@dataclass(frozen=True, slots=True)
class LineSamples:
    """The poses of one straight line in the base frame, both ends included, and the step they kept.

    A line is judged in joint space like everything else, so these poses still have to go through the
    controller's inverse kinematics before a checker sees them. They are kept apart from
    :class:`PathSamples` for that reason: a pose is not a configuration, and a type that held either
    would let one reach a gate that only understands the other.
    """

    poses: tuple[Pose, ...]
    step_bound_mm: float


def _positive(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{name} has to be a positive finite number of millimetres, got {value!r}")
    return number


def _radii(reach_mm: "float | Sequence[float]", *, joints: int) -> tuple[float, ...]:
    """One radius per joint: the caller's per-joint bounds, or one number applied to every joint."""
    if isinstance(reach_mm, (int, float)):
        return (_positive(float(reach_mm), name="reach_mm"),) * joints
    radii = tuple(_positive(value, name=f"reach_mm[{index}]") for index, value in enumerate(reach_mm))
    if len(radii) != joints:
        raise ValueError(
            f"reach_mm holds {len(radii)} radii for a {joints} joint configuration: a bound per joint "
            f"has to name every joint"
        )
    return radii


def _steps_for(travel_mm: float, *, max_step_mm: float, what: str) -> int:
    """How many steps this much travel needs, or a refusal naming the cap."""
    if travel_mm <= 0.0:
        return 0
    steps = max(1, math.ceil(travel_mm / max_step_mm - _EXACT_MULTIPLE_SLACK))
    if steps + 1 > MAX_PATH_SAMPLES:
        raise ValueError(
            f"{what} needs {steps + 1} samples at {max_step_mm:.6g} mm a step, above the cap of "
            f"{MAX_PATH_SAMPLES}, so it is refused rather than thinned: a sparser path would leave "
            f"the gaps between the samples unexamined, which is what checking the path is for. "
            f"Move in shorter legs, or raise the step the caller allows."
        )
    return steps


def _finite_config(values: "Sequence[float]", *, name: str) -> tuple[float, ...]:
    config = tuple(float(v) for v in values)
    if not config:
        raise ValueError(f"{name} holds no joints")
    for index, value in enumerate(config):
        if not math.isfinite(value):
            raise ValueError(f"{name} joint {index} is {value!r}, which is not a finite number")
    return config


def joint_path_samples(
    start: "Sequence[float]",
    goal: "Sequence[float]",
    *,
    reach_mm: "float | Sequence[float]",
    max_step_mm: float,
) -> PathSamples:
    """Sample the straight line between two joint configurations, both ends included.

    Straight in joint space, which is what a controller executes for a joint move and what a planner's
    neighbouring waypoints are. The count comes from ``sum(|dq_j| * reach_mm[j])``, so a move of one
    wrist joint is sampled far less densely than the same angle on the shoulder, which is the
    difference that matters to everything the arm could hit.

    ``reach_mm`` is one radius per joint, or one number applied to every joint where the caller has
    nothing better. A radius is an upper bound on the distance from that joint's axis to anything
    beyond it, so the product bounds the arc.

    A move that goes nowhere is one configuration, not two: there is one thing to check, and the caller
    that asked has already been told the step bound is zero.
    """
    step = _positive(max_step_mm, name="max_step_mm")
    first = _finite_config(start, name="start")
    last = _finite_config(goal, name="goal")
    if len(first) != len(last):
        raise ValueError(
            f"start has {len(first)} joints and goal has {len(last)}: a path between two arms is not "
            f"a path"
        )
    radii = _radii(reach_mm, joints=len(first))

    travel_mm = sum(abs(b - a) * radius for a, b, radius in zip(first, last, radii))
    steps = _steps_for(travel_mm, max_step_mm=step, what="this joint move")
    if steps == 0:
        return PathSamples(configs=(first,), step_bound_mm=0.0)

    configs = [first]
    for index in range(1, steps):
        fraction = index / steps
        configs.append(tuple(a + (b - a) * fraction for a, b in zip(first, last)))
    configs.append(last)  # the goal itself, not an interpolation that lands near it
    return PathSamples(configs=tuple(configs), step_bound_mm=travel_mm / steps)


def line_samples(
    start_pose: Pose,
    goal_pose: Pose,
    *,
    reach_mm: float,
    max_step_mm: float,
) -> LineSamples:
    """Sample the straight line between two base frame poses, both ends included.

    Linear in position and spherical in orientation, so the tool travels a straight line at a constant
    rate and turns at a constant rate along it. Both halves are bounded: the travel by the caller's
    bound directly, the rotation by the same bound over the reach, because a rotation about the tool
    can swing the elbow much further than the tool itself moves.

    Base frame only. The bound is stated against a reach measured from the base, so a line expressed in
    any other frame would be held to a number that does not describe it.
    """
    reach = _positive(reach_mm, name="reach_mm")
    step = _positive(max_step_mm, name="max_step_mm")
    for name, pose in (("start_pose", start_pose), ("goal_pose", goal_pose)):
        if pose.frame is not Frame.BASE:
            raise ValueError(
                f"{name} is in the {pose.frame} frame; a sampled line is judged against a reach "
                f"measured from the base, so both ends have to be in the {Frame.BASE} frame"
            )

    from_mm = np.asarray(start_pose.position_mm, dtype=np.float64)
    to_mm = np.asarray(goal_pose.position_mm, dtype=np.float64)
    travel_mm = float(np.linalg.norm(to_mm - from_mm))
    turned_rad = angle_between(start_pose.quaternion_xyzw, goal_pose.quaternion_xyzw)
    worst_mm = max(travel_mm, turned_rad * reach)

    steps = _steps_for(worst_mm, max_step_mm=step, what="this line")
    if steps == 0:
        return LineSamples(poses=(start_pose,), step_bound_mm=0.0)

    poses = [start_pose]
    for index in range(1, steps):
        fraction = index / steps
        poses.append(
            Pose(
                position_mm=from_mm + (to_mm - from_mm) * fraction,
                quaternion_xyzw=slerp(
                    start_pose.quaternion_xyzw, goal_pose.quaternion_xyzw, fraction
                ),
                frame=Frame.BASE,
            )
        )
    poses.append(goal_pose)
    return LineSamples(poses=tuple(poses), step_bound_mm=worst_mm / steps)


def waypoint_path_samples(
    waypoints: "Sequence[Sequence[float]]",
    *,
    reach_mm: "float | Sequence[float]",
    max_step_mm: float,
) -> PathSamples:
    """Sample every leg of a planner's path, joined into one set of configurations.

    A planner hands over corners, not a path. Between two of them the controller interpolates in joint
    space, on a real UR by moveJ-ing to each in turn and in the sim by applying each to the
    articulation, so the middle of every leg is executed and nothing has ever looked at it. Judging the
    waypoints alone judges the corners.

    The cap counts the whole path rather than a leg, because it is the checker's cap and the checker
    sees one request. A configuration shared by two legs is kept once: it is one place the arm is in.
    """
    legs = [_finite_config(w, name=f"waypoint {index}") for index, w in enumerate(waypoints)]
    if not legs:
        return PathSamples(configs=(), step_bound_mm=0.0)
    if len(legs) == 1:
        return PathSamples(configs=(legs[0],), step_bound_mm=0.0)

    configs: list[tuple[float, ...]] = []
    bound_mm = 0.0
    for start, goal in zip(legs, legs[1:]):
        leg = joint_path_samples(start, goal, reach_mm=reach_mm, max_step_mm=max_step_mm)
        bound_mm = max(bound_mm, leg.step_bound_mm)
        configs.extend(leg.configs[1:] if configs else leg.configs)
        if len(configs) > MAX_PATH_SAMPLES:
            raise ValueError(
                f"this path of {len(legs)} waypoints needs more than {MAX_PATH_SAMPLES} samples at "
                f"{max_step_mm:.6g} mm a step, so it is refused rather than thinned: a sparser path "
                f"would leave the gaps between the samples unexamined, which is what checking the "
                f"path is for. Plan it in shorter legs, or raise the step the caller allows."
            )
    return PathSamples(configs=tuple(configs), step_bound_mm=bound_mm)
