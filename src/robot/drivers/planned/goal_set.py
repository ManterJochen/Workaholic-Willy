"""Where a planned move's goal configuration comes from: the closed form the arm's description states.

The owner, 2026-09-25: every planned arm plans only this way (the Isaac sim arm is not one yet: it keeps its own
route, which still hands cuRobo a pose, until it becomes a planned arm). cuRobo, handed a pose, chose the goal
configuration itself from random seeds over the whole joint range and did not prefer the one nearest the arm (measured
on the UR10: 8 of 22 legs of a wrist sweep went the long way), and its graph planner routed plans through the retract;
the owner wants neither for any vendor, least of all for the large industrial arms. So the planned arm never hands
cuRobo a pose. It asks this module for every configuration of the flange goal and for the branch each holds, turns
them onto the nearest full turns, and runs the straight line to the best one or plans to it in joint space
(:meth:`~src.robot.drivers.planned.arm.PlannedArm._route_to_the_nearest_goal`). An arm whose description states no
closed form is refused before anything is sent, naming this module: there is no fallback to cuRobo's own choice.

Today one family has an entry: the UR chain, whose description's DH rows (``config/arms/<model>.yaml``, read through
:func:`~src.robot.safety._ur_kinematics.described_ur_model`) the closed form of :mod:`src.robot.safety._ur_ik`
solves, up to eight solutions in a fixed branch order. An arm of another family adds its entry here from its own
description: a spherical-wrist industrial arm has a closed form of its own, and a seven-joint arm has no finite set of
solutions, so its entry needs a redundancy rule first. An entry is not all a family needs: the planned arm reads where
a plan ends on the UR's DH chain too (:meth:`~src.robot.drivers.planned.arm.PlannedArm._plan_end_refusal`), so a
second family also brings its forward kinematics there, or every plan it takes is refused at its end.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy as np

__all__ = ["Branch", "GoalSet", "goal_set_for"]


class Branch(Protocol):
    """Which way of holding the arm a configuration is in, as a family's closed form tells them apart."""

    def agrees_with(self, other: Any) -> bool:
        """Whether ``other`` holds the arm the same way; a configuration at a branch point agrees with both sides."""
        ...

    def render(self) -> str:
        """The branch as a person reads it in a log line."""
        ...


@dataclass(frozen=True, slots=True)
class GoalSet:
    """What an arm's description answers before a planned move: every configuration of a flange goal, and its branch.

    ``family`` names the closed form in a log line. ``solve(flange_mm, last_joint)`` returns every configuration that
    puts the flange at the 4x4 goal (millimetres, in the controller's base frame), each joint in ``(-pi, pi]`` and in a
    fixed order, and an empty tuple where the goal is out of reach; ``last_joint`` is where the last joint goes where
    only a sum of two joints is fixed, which the arm passes as the joint it stands at. ``branch(joints)`` is the branch a
    configuration holds, ``None`` where the family cannot say, which agrees with every branch.
    """

    family: str
    solve: "Callable[[np.ndarray, float], tuple[tuple[float, ...], ...]]"
    branch: "Callable[[Sequence[float]], Branch | None]"


def goal_set_for(model: str) -> "GoalSet | None":
    """The closed form ``model``'s description states, or ``None`` where it states none, which the route refuses."""
    from src.robot.safety._ur_ik import ur_branch, ur_flange_ik
    from src.robot.safety._ur_kinematics import described_ur_model

    name = described_ur_model(model)
    if name is None:
        return None
    described: str = name

    def solve(flange_mm: "np.ndarray", last_joint: float) -> "tuple[tuple[float, ...], ...]":
        solutions = ur_flange_ik(described, flange_mm, q6_if_singular=last_joint)
        return () if solutions is None else tuple(tuple(float(v) for v in s) for s in solutions)

    def branch(joints: "Sequence[float]") -> "Branch | None":
        return ur_branch(described, joints)

    return GoalSet(family="the UR closed form", solve=solve, branch=branch)
