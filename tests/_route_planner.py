"""A stand-in for the UR driver's cuRobo glue that records every question the arm asks it and answers as told.

The UR driver chooses where a move goes and how: the configuration nearest the arm on the branch it holds, the straight
joint line there where both authorities pass it, a cuRobo joint plan only where they do not. What that decides is
which questions reach the planner, in what order and at what clearance, and which list reaches ``execute``. This
double answers each question from a rule a test sets, and keeps the calls in order so a test reads the decision:

* ``screen(config)``: the planner's verdict on one configuration, a goal it screens (default: clear);
* ``line_clear(start, end)``: its verdict on a straight line judged at a clearance (default: clear);
* ``answer(goal)``: the trajectory ``plan_joint`` returns, ``None`` for no plan (default: 21 waypoints on the line).

Every check is recorded as ``("check", configs, refresh, clearance_mm)``, every plan as ``("plan_joint", goal,
refresh)``, every run as ``("execute", waypoints, command, target_joints)``. It has no Cartesian ``plan``: the glue has
none, and a driver that asked for one would fail here as it would on the real glue.
"""

from __future__ import annotations

from typing import Any

from src.robot.core import MotionCommand, MotionResult
from src.robot.safety.planning import JointCheckVerdict, StateRefusal


def line(start: "list[float] | tuple[float, ...]", end: "list[float] | tuple[float, ...]",
         count: int) -> list[list[float]]:
    """``count`` waypoints evenly along the straight joint line, both ends exact: cuRobo's plan of a free move."""
    out = [list(start)]
    for index in range(1, count - 1):
        out.append([a + (b - a) * index / (count - 1) for a, b in zip(start, end)])
    out.append(list(end))
    return out


class RoutePlanner:
    """A planner that plans joint goals and judges joint paths, answering as the test sets it up."""

    def __init__(self, *, here: "list[float] | tuple[float, ...]") -> None:
        self.here = [float(v) for v in here]
        self.calls: list[tuple[Any, ...]] = []
        self.last_refusal: "StateRefusal | None" = None
        self.screen = lambda config: True
        self.line_clear = lambda start, end: True
        self.answer = lambda goal: line(self.here, goal, 21)
        self.refusal_on_no_plan: "StateRefusal | None" = None
        self.refusal_on_a_line: "StateRefusal | None" = None

    def refresh_world(self, *, near_point_mm: Any = None, goal_keep_out: Any = None) -> None:
        self.calls.append(("refresh", near_point_mm))

    def plan_joint(self, goal: Any, *, refresh: bool = True, **_: Any) -> "list[list[float]] | None":
        asked = [float(v) for v in goal]
        self.calls.append(("plan_joint", asked, refresh))
        trajectory = self.answer(asked)
        self.last_refusal = None if trajectory else self.refusal_on_no_plan
        return trajectory

    def check_joint_path(self, samples: Any, *, refresh: bool = True, clearance_mm: float = 0.0) -> JointCheckVerdict:
        configs = [tuple(float(v) for v in sample) for sample in samples]
        self.calls.append(("check", configs, refresh, float(clearance_mm)))
        if len(configs) == 1 and not self.screen(configs[0]):
            return JointCheckVerdict(valid=False, first_invalid=0, checked=1, reason="the double refuses this goal")
        if clearance_mm > 0.0 and not self.line_clear(configs[0], configs[-1]):
            return JointCheckVerdict(valid=False, first_invalid=len(configs) // 2, checked=len(configs),
                                     reason="the double's line comes too close", refusal=self.refusal_on_a_line)
        return JointCheckVerdict(valid=True, first_invalid=None, checked=len(configs), reason="the double accepts")

    def execute(self, traj: Any, pose: Any = None, *, command: MotionCommand = MotionCommand.MOVE_TO,
                target_joints: Any = None, vel: Any = None, acc: Any = None) -> MotionResult:
        self.calls.append(("execute", [tuple(float(v) for v in w) for w in traj], command, target_joints))
        return MotionResult.executed(command, target_pose=pose, target_joints=target_joints, message="curobo")

    def named(self, name: str) -> list[tuple[Any, ...]]:
        return [call for call in self.calls if call[0] == name]

    def lines(self) -> list[tuple[Any, ...]]:
        """Every check asked at a clearance: the straight lines the arm would have run instead of a plan."""
        return [call for call in self.named("check") if call[3] > 0.0]
