"""Closed-form inverse kinematics of the UR flange, on the DH tables the guards already use.

Every UR in :data:`~src.robot.safety._ur_kinematics.UR_DH_TABLES_M` has the same structure: a shoulder
offset, two parallel arm links and three wrist axes that do not meet in a point. For that chain the flange
pose has up to eight solutions, one per choice of shoulder (left or right), wrist 2 (flipped or not) and
elbow (up or down), and each of them is found in closed form. This module finds all of them, so a caller can
choose which one to go to rather than taking whichever one a numerical solver happened to land on.

What it is for: a cuRobo move to a Cartesian goal lets the planner pick the goal configuration, and it does
not prefer the one nearest the arm (measured on a UR10: 8 of 22 legs of a wrist sweep went the long way, one of
them turning the base 8.57 rad where 2.37 rad would do). With every solution in hand, the UR driver takes the
one nearest the arm on the branch the arm already holds (:func:`ur_branch`), runs the straight joint line to it
where that line is clear, and plans to it in joint space where it is not; cuRobo never chooses a goal. This
module only answers the geometry; which solution is admissible is the planner's and the guards' question.

Accuracy. The closed form assumes the table's twist angles are exactly a quarter turn, and the tables carry
1.570796327, 2e-10 rad off. One Newton step on the table's own chain removes that, so a returned solution puts
the flange on the goal to well under a micrometre and a microradian on every bundled model (the tests measure
it on random configurations). A solution the step cannot bring that close, which happens only next to a
singularity, is still returned when it is within :data:`IK_SOLUTION_TOL_MM` and :data:`IK_SOLUTION_TOL_RAD`, and
dropped beyond that.

Singularities. Where wrist 2 is at 0 or pi, wrist 1 and wrist 3 turn about parallel axes and only their sum is
fixed; wrist 3 then takes the caller's ``q6_if_singular`` (the arm passes the joint it stands at) and wrist 1
the rest. Where the elbow is stretched or the wrist centre sits on the shoulder's cylinder, two branches
coincide and are returned once. A pose out of reach has no solution and returns an empty tuple.

Pure numpy, no config and no logger. Joints are radians in UR order, base to wrist 3, and each comes back in
``(-pi, pi]``; which of its full turns to command is the caller's choice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ._ur_kinematics import UR_DH_TABLES_M, URDhRow, ur_link_transforms_mm

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = [
    "BRANCH_POINT_TOL_RAD",
    "IK_SOLUTION_TOL_MM",
    "IK_SOLUTION_TOL_RAD",
    "NearestGoal",
    "URBranch",
    "nearest_goals",
    "nearest_turn",
    "ur_branch",
    "ur_flange_ik",
]

#: How far a returned solution may put the flange from the goal. A solution the Newton step brings to the goal is
#: orders of magnitude inside it; one next to a singularity, where the step is ill-conditioned, may not be, and is
#: kept as long as it is inside this. It is well inside the 5 mm the UR driver holds a plan's end to.
IK_SOLUTION_TOL_MM = 1e-3
#: The rotation half of the same bound.
IK_SOLUTION_TOL_RAD = 1e-6

#: Below this, wrist 2 counts as at 0 or pi and wrist 3 is the caller's to choose.
_WRIST_SINGULAR = 1e-10
#: How far a cosine may stray past +-1 and still be read as the branch point rather than out of reach. The goal comes
#: off the table's chain, whose twists are 2e-10 rad off a quarter turn, so a wrist centre exactly on the shoulder's
#: cylinder was measured reading 1.05e-9 past it on a ur5; what the slack lets through is then held to the residual.
_COSINE_SLACK = 1e-7
#: Two solutions this close on every joint are the same one, reached through two coinciding branches.
_SAME_SOLUTION_RAD = 1e-9
_TWO_PI = 2.0 * math.pi


def _wrapped(value: float) -> float:
    """``value`` turned into ``(-pi, pi]``."""
    turned = math.fmod(value + math.pi, _TWO_PI)
    if turned <= 0.0:
        turned += _TWO_PI
    return turned - math.pi


def _apart(a: "Sequence[float]", b: "Sequence[float]") -> float:
    """The largest angle between two configurations on any joint, counting a full turn as no turn."""
    return max(abs(_wrapped(x - y)) for x, y in zip(a, b))


def _dh(theta: float, row: URDhRow) -> np.ndarray:
    """The standard DH transform in metres, with the table's own twist."""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(row.alpha_rad), math.sin(row.alpha_rad)
    return np.array([
        [ct, -st * ca, st * sa, row.a_m * ct],
        [st, ct * ca, -ct * sa, row.a_m * st],
        [0.0, sa, ca, row.d_m],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def _flange_m(model: str, joints: "Sequence[float]") -> np.ndarray:
    frames = ur_link_transforms_mm(model, np.asarray(joints, dtype=np.float64))
    assert frames is not None  # the model was checked by the caller
    flange = np.asarray(frames[-1], dtype=np.float64).copy()
    flange[:3, 3] /= 1000.0
    return flange


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    """The axis times the angle of a rotation matrix, for the small residuals the Newton step sees."""
    cosine = max(-1.0, min(1.0, (float(np.trace(rotation)) - 1.0) / 2.0))
    angle = math.acos(cosine)
    skew = np.array([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ], dtype=np.float64)
    if angle < 1e-12:
        return skew / 2.0
    return skew * (angle / (2.0 * math.sin(angle)))


def _residual(model: str, joints: "Sequence[float]", goal_m: np.ndarray) -> tuple[np.ndarray, float, float]:
    """The six-vector from where ``joints`` put the flange to ``goal_m``, and its position (mm) and angle (rad)."""
    reached = _flange_m(model, joints)
    position = goal_m[:3, 3] - reached[:3, 3]
    rotation = _rotation_vector(goal_m[:3, :3] @ reached[:3, :3].T)
    return (
        np.concatenate([position, rotation]),
        float(np.linalg.norm(position)) * 1000.0,
        float(np.linalg.norm(rotation)),
    )


def _jacobian(model: str, joints: "Sequence[float]") -> np.ndarray:
    """The geometric Jacobian of the flange, metres and radians, from the table's own frames."""
    frames = ur_link_transforms_mm(model, np.asarray(joints, dtype=np.float64))
    assert frames is not None
    end = np.asarray(frames[-1][:3, 3], dtype=np.float64) / 1000.0
    columns = []
    for frame in frames[:-1]:
        axis = np.asarray(frame[:3, 2], dtype=np.float64)
        origin = np.asarray(frame[:3, 3], dtype=np.float64) / 1000.0
        columns.append(np.concatenate([np.cross(axis, end - origin), axis]))
    return np.stack(columns, axis=1)


def _polished(model: str, joints: list[float], goal_m: np.ndarray) -> tuple[list[float], float, float]:
    """``joints`` after up to two Newton steps on the table's chain, kept only where a step made it closer."""
    best = list(joints)
    residual, error_mm, error_rad = _residual(model, best, goal_m)
    for _ in range(2):
        if error_mm < 1e-9 and error_rad < 1e-12:
            break
        step, *_ = np.linalg.lstsq(_jacobian(model, best), residual, rcond=1e-10)
        candidate = [q + float(dq) for q, dq in zip(best, step)]
        new_residual, new_mm, new_rad = _residual(model, candidate, goal_m)
        if new_mm + new_rad * 1000.0 >= error_mm + error_rad * 1000.0:
            break
        best, residual, error_mm, error_rad = candidate, new_residual, new_mm, new_rad
    return best, error_mm, error_rad


def ur_flange_ik(
    model: str,
    flange_mm: "np.ndarray | Sequence[Sequence[float]]",
    *,
    q6_if_singular: float = 0.0,
) -> "tuple[tuple[float, ...], ...] | None":
    """Every joint configuration that puts this UR's flange at ``flange_mm``, or ``None`` for a model with no table.

    ``flange_mm`` is a 4x4 in the controller's base frame (the DH base) with its translation in millimetres, the
    convention of :func:`~src.robot.safety._ur_kinematics.ur_link_transforms_mm`, and it is the flange, tool0, not
    a TCP. The result holds up to eight configurations, each joint in ``(-pi, pi]``, in a fixed order (shoulder,
    then wrist 2, then elbow branch). It is empty where the pose is out of reach.

    ``q6_if_singular`` is where wrist 3 goes when wrist 2 is at 0 or pi and only the sum of wrist 1 and wrist 3 is
    fixed; the arm passes the wrist 3 it stands at, so the solution does not turn it for nothing.
    """
    table = UR_DH_TABLES_M.get(str(model).lower())
    if table is None:
        return None
    goal = np.asarray(flange_mm, dtype=np.float64)
    if goal.shape != (4, 4) or not np.all(np.isfinite(goal)):
        raise ValueError(f"the flange goal has to be a finite 4x4 transform, got shape {goal.shape}")
    goal_m = goal.copy()
    goal_m[:3, 3] /= 1000.0

    a2, a3, d4, d6 = table[1].a_m, table[2].a_m, table[3].d_m, table[5].d_m
    # The wrist-2 centre: the flange drawn back along its own axis by d6. Its bearing from the base axis, less the
    # angle the shoulder offset d4 takes, is the base joint; the two ways round are the two shoulders.
    wrist = goal_m @ np.array([0.0, 0.0, -d6, 1.0])
    radius = math.hypot(float(wrist[0]), float(wrist[1]))
    if radius < abs(d4) * (1.0 - _COSINE_SLACK):
        return ()
    bearing = math.atan2(float(wrist[1]), float(wrist[0]))
    offset = math.acos(max(-1.0, min(1.0, d4 / radius)))
    inverse = np.linalg.inv(goal_m)

    solutions: list[tuple[float, ...]] = []
    for shoulder in (1.0, -1.0):
        q1 = bearing + shoulder * offset + math.pi / 2.0
        s1, c1 = math.sin(q1), math.cos(q1)
        c5 = (float(goal_m[0, 3]) * s1 - float(goal_m[1, 3]) * c1 - d4) / d6
        if abs(c5) > 1.0 + _COSINE_SLACK:
            continue
        c5 = max(-1.0, min(1.0, c5))
        for wrist_flip in (1.0, -1.0):
            q5 = wrist_flip * math.acos(c5)
            s5 = math.sin(q5)
            if abs(s5) < _WRIST_SINGULAR:
                q6 = float(q6_if_singular)
            else:
                # Only the sign of s5 matters to atan2, so dividing by it cannot blow up near the singularity.
                sign = 1.0 if s5 > 0.0 else -1.0
                q6 = math.atan2(
                    sign * (-float(inverse[1, 0]) * s1 + float(inverse[1, 1]) * c1),
                    sign * (float(inverse[0, 0]) * s1 - float(inverse[0, 1]) * c1),
                )
            # What is left between frame 1 and frame 4 is a planar two-link arm plus a turn of wrist 1.
            t14 = np.linalg.inv(_dh(q1, table[0])) @ goal_m @ np.linalg.inv(_dh(q5, table[4]) @ _dh(q6, table[5]))
            px, py = float(t14[0, 3]), float(t14[1, 3])
            c3 = (px * px + py * py - a2 * a2 - a3 * a3) / (2.0 * a2 * a3)
            if abs(c3) > 1.0 + _COSINE_SLACK:
                continue
            c3 = max(-1.0, min(1.0, c3))
            for elbow in (1.0, -1.0):
                q3 = elbow * math.acos(c3)
                q2 = math.atan2(py, px) - math.atan2(a3 * math.sin(q3), a2 + a3 * math.cos(q3))
                t34 = np.linalg.inv(_dh(q2, table[1]) @ _dh(q3, table[2])) @ t14
                q4 = math.atan2(float(t34[1, 0]), float(t34[0, 0]))
                joints, error_mm, error_rad = _polished(
                    str(model).lower(), [q1, q2, q3, q4, q5, q6], goal_m,
                )
                if error_mm > IK_SOLUTION_TOL_MM or error_rad > IK_SOLUTION_TOL_RAD:
                    continue
                wrapped = tuple(_wrapped(q) for q in joints)
                if any(_apart(wrapped, seen) < _SAME_SOLUTION_RAD for seen in solutions):
                    continue
                solutions.append(wrapped)
    return tuple(solutions)


#: How near its branch point a joint reads as on both of the branches it separates, radians. The UR10 CB3's home is
#: candle-straight: the elbow stretched, wrist 2 at 0 and the wrist centre on the shoulder's cylinder, all three branch
#: points at once, and the encoders put each a few microradians to one side or the other. Read as bare signs, that noise
#: would choose how the first move out of home holds the arm. Two degrees is far above the noise and far below any bend a
#: move is chosen by: a joint that close to its branch point turns to either side for the same cost.
BRANCH_POINT_TOL_RAD = math.radians(2.0)


def _side(angle: float) -> int:
    """``+1`` or ``-1`` for the side of its branch points (0 and pi) ``angle`` is on, ``0`` within the tolerance of one."""
    sine = math.sin(angle)
    if abs(sine) < math.sin(BRANCH_POINT_TOL_RAD):
        return 0
    return 1 if sine > 0.0 else -1


@dataclass(frozen=True, slots=True)
class URBranch:
    """Which of the closed form's eight ways of holding the arm a configuration is in, as three sides.

    ``shoulder`` is the side of the base axis the wrist centre is reached from, the sign of the offset the shoulder
    takes against the wrist centre's bearing. ``wrist`` is the side of wrist 2 and ``elbow`` the side of the elbow, the
    sign of the joint turned into ``(-pi, pi]``, so both can be read off the pendant. Two configurations on one branch
    reach the poses between them without a joint swinging through a branch point; a change of branch is a different way
    to hold the arm, with the cable, the camera and the hand turned with it.

    Each side is ``+1`` or ``-1``, and ``0`` within :data:`BRANCH_POINT_TOL_RAD` of the branch point (a stretched or
    folded elbow, wrist 2 at 0 or pi, the wrist centre on the shoulder's cylinder), where the two branches it separates
    meet and the configuration is on both: :meth:`agrees_with` reads a ``0`` as agreeing with either side.
    """

    shoulder: int
    wrist: int
    elbow: int

    def agrees_with(self, other: "URBranch") -> bool:
        """Whether ``other`` holds the arm the same way: on every one of the three, the same side or a branch point."""
        pairs = ((self.shoulder, other.shoulder), (self.wrist, other.wrist), (self.elbow, other.elbow))
        return all(mine == theirs or mine == 0 or theirs == 0 for mine, theirs in pairs)

    def render(self) -> str:
        """The three sides as a person reads them, ``shoulder +, wrist 2 -, elbow +/-``; ``+/-`` is a branch point."""
        def sign(value: int) -> str:
            return "+" if value > 0 else "-" if value < 0 else "+/-"

        return f"shoulder {sign(self.shoulder)}, wrist 2 {sign(self.wrist)}, elbow {sign(self.elbow)}"


def ur_branch(model: str, joints: "Sequence[float]") -> "URBranch | None":
    """The branch ``joints`` hold the arm in, by the conventions :func:`ur_flange_ik` solves with; ``None`` for no table.

    The same reading for the configuration the arm stands in and for every solution of a goal, so "the same branch"
    means the same three choices of the closed form and nothing a caller has to reconstruct. A full turn of any joint
    changes nothing here, as it changes nothing about the arm.
    """
    table = UR_DH_TABLES_M.get(str(model).lower())
    if table is None:
        return None
    q = [float(v) for v in joints]
    if len(q) != len(table) or not all(math.isfinite(v) for v in q):
        raise ValueError(f"a {model} configuration is {len(table)} finite joint values, got {joints!r}")
    # The closed form sets the base to bearing + shoulder * offset + pi/2, the offset in [0, pi/2] because d4 > 0, so
    # the shoulder is the side of what is left of the base joint once the wrist centre's bearing is taken off; it is 0
    # where the wrist centre stands on the shoulder's cylinder and both ways round are one. Wrist 2 and the elbow are
    # the signs the closed form gave them, acos in [0, pi] times the branch, with their branch points at 0 and pi.
    wrist_centre = _flange_m(str(model).lower(), q) @ np.array([0.0, 0.0, -table[5].d_m, 1.0])
    bearing = math.atan2(float(wrist_centre[1]), float(wrist_centre[0]))
    return URBranch(
        shoulder=_side(_wrapped(q[0] - math.pi / 2.0 - bearing)),
        wrist=_side(q[4]),
        elbow=_side(q[2]),
    )


@dataclass(frozen=True, slots=True)
class NearestGoal:
    """One configuration a goal could be reached in, turned to be nearest the arm, and what reaching it costs.

    ``joints`` are the ones to command, each on the full turn nearest where the arm stands inside the window it was
    turned into. ``branch`` is its index in what :func:`ur_flange_ik` returned, for a log line. ``largest_rad`` is the
    largest turn any joint makes and ``total_rad`` the sum of every joint's turn, both from the joints the arm stands
    at; ``time_s`` is ``largest_rad`` over the speed, the least time a ``moveJ`` there can take.
    """

    joints: tuple[float, ...]
    branch: int
    largest_rad: float
    total_rad: float
    time_s: float


def nearest_turn(value: float, *, near: float, lower: float, upper: float) -> "float | None":
    """``value`` plus the whole number of full turns that lands nearest ``near`` inside ``[lower, upper]``, or ``None``.

    A full turn of a UR joint is the same arm pose, so this changes which of the identical numbers is commanded and
    nothing about where the arm ends. ``None`` where no turn of it lies inside the window at all, which is how an
    elbow at 179 degrees reads against a window of +-174.
    """
    first = math.ceil((lower - value) / _TWO_PI - 1e-12)
    last = math.floor((upper - value) / _TWO_PI + 1e-12)
    turns = [value + k * _TWO_PI for k in range(first, last + 1)]
    inside = [turned for turned in turns if lower <= turned <= upper]
    if not inside:
        return None
    return min(inside, key=lambda turned: (abs(turned - near), turned))


def nearest_goals(
    solutions: "Sequence[Sequence[float]]",
    *,
    current: "Sequence[float]",
    lower: "Sequence[float]",
    upper: "Sequence[float]",
    velocity: float,
) -> tuple[NearestGoal, ...]:
    """Every solution turned to be nearest ``current`` inside ``[lower, upper]``, quickest first.

    Each joint of each solution goes on its full turn nearest the joint the arm stands at, within the window; a
    solution with a joint that has no turn inside is left out, because nothing may command it. What remains is ordered
    by the least time a ``moveJ`` there can take, ``max_j |dq_j| / velocity``, then by the total turn, then by branch,
    so the same inputs give the same order. One speed for every joint, because the UR runs every joint of a ``moveJ``
    at the one speed it is commanded.

    Two solutions that turn onto the same configuration are one goal, kept at the lower branch.
    """
    speed = float(velocity)
    if not math.isfinite(speed) or speed <= 0.0:
        raise ValueError(f"velocity has to be a positive number of rad/s, got {velocity!r}")
    here = [float(v) for v in current]
    lo = [float(v) for v in lower]
    hi = [float(v) for v in upper]
    if not len(here) == len(lo) == len(hi):
        raise ValueError(
            f"current has {len(here)} joints and the window {len(lo)} and {len(hi)}: one bound per joint"
        )
    goals: list[NearestGoal] = []
    for branch, solution in enumerate(solutions):
        if len(solution) != len(here):
            raise ValueError(f"solution {branch} has {len(solution)} joints and the arm has {len(here)}")
        turned = [
            nearest_turn(float(value), near=near, lower=low, upper=high)
            for value, near, low, high in zip(solution, here, lo, hi)
        ]
        if any(value is None for value in turned):
            continue
        joints = tuple(float(value) for value in turned if value is not None)
        if any(max(abs(a - b) for a, b in zip(joints, goal.joints)) < _SAME_SOLUTION_RAD for goal in goals):
            continue
        moves = [abs(a - b) for a, b in zip(joints, here)]
        goals.append(NearestGoal(
            joints=joints, branch=branch, largest_rad=max(moves), total_rad=sum(moves),
            time_s=max(moves) / speed,
        ))
    return tuple(sorted(goals, key=lambda goal: (goal.time_s, goal.total_rad, goal.branch)))
