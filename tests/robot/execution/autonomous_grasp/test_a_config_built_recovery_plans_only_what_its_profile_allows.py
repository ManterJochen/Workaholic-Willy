"""What the recovery loop of a cell built from config can actually do, pinned at the service.

The mode's built-in behaviour profile (`_PROFILES` in `autonomous_grasp/config.py`) is the outer gate of
`RecoveryOrchestrator.next_step`. Since 2026-10-16 no key sets a cell's recovery (the owner's cleanup): a cell built
from config runs `CELL_RECOVERY`, the values the owner's cell ran, a rescan once and a push twice a pick, at most two
actions, in `auto` and `dense_clutter`. `next_target`, `container_agitate` and the fixture box left that day.

MEASURED through `AutonomousGraspService._run_with_recovery` with a scripted pick:

* a failed pick rescans once, in `auto` and in `dense_clutter`, and a second failure ends the loop for want of an
  action (`recovery_exhausted`);
* `nudge_target` is never planned by this loop: the push runs inside the pick attempt (the pick loop's push gate), so
  for `ALL_COLLIDED` the loop goes on to the rescan and the arm never moves from here;
* `easy` never recovers.

The pick itself is scripted (`_pick_inner` replaced on the instance) because what is pinned here is the loop around
it, not perception or motion; the loop, the orchestrator, the dispatcher and the profile are the real ones a cell
built from config runs.
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
)
from src.robot.execution.autonomous_grasp.config import CELL_RECOVERY
from src.robot.grasping.types.feedback import GraspFailureReason
from tests.robot.grasping.types.test_grasping_config_wiring import _calc_and_perception


@dataclasses.dataclass(frozen=True)
class _ScriptedReport:
    """The fields of an `AutonomousGraspReport` the recovery loop reads and rewrites."""

    outcome: Any
    pick_report: Any
    telemetry: dict = dataclasses.field(default_factory=dict)
    recovery_actions: tuple = ()


def _failed(reason: GraspFailureReason) -> _ScriptedReport:
    return _ScriptedReport(
        outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
        pick_report=SimpleNamespace(attempts=[SimpleNamespace(reasons=(reason,))]),
    )


def _succeeded() -> _ScriptedReport:
    return _ScriptedReport(outcome=AutonomousGraspOutcome.SUCCEEDED, pick_report=None)


class _CountingArm:
    """Counts every move; the recovery loop must leave this at zero."""

    def __init__(self) -> None:
        self.moves: list[Any] = []

    def get_tcp_pose(self) -> Pose:
        return Pose(
            position_mm=np.array([0.0, -700.0, 200.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )

    def move(self, pose: Any, *args: Any, **kwargs: Any) -> Any:
        self.moves.append(pose)
        return SimpleNamespace(status="EXECUTED")


def _service(mode: str) -> AutonomousGraspService:
    cfg = RobotConfig(vendor="dummy", gripper={"vendor": "none"}, grasping={"default_mode": mode})
    calculator, perception = _calc_and_perception()
    return AutonomousGraspService.from_robot_config(cfg, calculator=calculator, perception=perception)


def _run(
    service: AutonomousGraspService, script: list[_ScriptedReport], arm: _CountingArm
) -> tuple[Any, int]:
    """Drive the service's own recovery loop over a scripted sequence of picks."""
    picks: Iterator[_ScriptedReport] = iter(script)
    calls = {"n": 0}

    def pick_inner(*, mode: Any = None) -> _ScriptedReport:
        calls["n"] += 1
        return next(picks)

    service._pick_inner = pick_inner  # type: ignore[method-assign,assignment]
    service.runtime.orchestrator.arm = arm  # type: ignore[assignment]
    report = service._run_with_recovery(mode=None)
    return report, calls["n"]


class AConfigBuiltCellRunsTheOwnersRecovery(unittest.TestCase):
    def test_the_snapshot_carries_the_owners_numbers(self) -> None:
        recovery = _service("dense_clutter").effective_config.recovery_orchestrator
        self.assertEqual(CELL_RECOVERY, recovery)
        self.assertEqual((True, 2, ("rescan", "nudge_target"), (("rescan", 1), ("nudge_target", 2)), 3, True),
                         (recovery.enabled, recovery.max_actions, recovery.allowed_actions,
                          recovery.per_action_budget, recovery.blocker_grasp_tries, recovery.blocker_into_the_place))
        self.assertFalse(_service("easy").effective_config.recovery_orchestrator.enabled, "easy never recovers")

    def test_a_failed_pick_rescans_once(self) -> None:
        for mode in ("auto", "dense_clutter"):
            with self.subTest(mode=mode):
                report, picks = _run(
                    _service(mode), [_failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()], _CountingArm())
                self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
                self.assertEqual(picks, 2)
                self.assertEqual(report.telemetry["recovery_trail_actions"], ["rescan"])
                self.assertEqual(
                    [(step["action"], step["step_result"]) for step in report.recovery_actions],
                    [("rescan", "completed")],
                )

    def test_a_second_failure_ends_the_loop_for_want_of_an_action(self) -> None:
        report, picks = _run(
            _service("auto"),
            [_failed(GraspFailureReason.NO_VALID_GRASP), _failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()],
            _CountingArm())
        self.assertEqual(picks, 2, "the cell rescans once a pick")
        self.assertIs(report.outcome, AutonomousGraspOutcome.RECOVERY_EXHAUSTED)
        self.assertEqual(report.telemetry["recovery_trail_actions"], ["rescan"])

    def test_the_loop_never_plans_a_nudge_and_goes_on_to_the_rescan(self) -> None:
        # The push runs inside the pick attempt; this loop never plans it, and the arm never moves from here.
        arm = _CountingArm()
        report, picks = _run(
            _service("dense_clutter"), [_failed(GraspFailureReason.ALL_COLLIDED), _succeeded()], arm)
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(picks, 2)
        self.assertEqual(report.telemetry["recovery_trail_actions"], ["rescan"])
        self.assertEqual(report.telemetry["recovery_trail_terminal_reason"], "recovered_success")
        self.assertEqual(arm.moves, [])

    def test_easy_never_recovers(self) -> None:
        report, picks = _run(_service("easy"), [_failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()],
                             _CountingArm())
        self.assertEqual(picks, 1)
        self.assertIs(report.outcome, AutonomousGraspOutcome.NO_VALID_GRASP)


if __name__ == "__main__":
    unittest.main()
