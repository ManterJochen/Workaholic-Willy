"""What the recovery loop of a cell built from config can actually do, pinned at the service.

The mode's built-in behaviour profile (`_PROFILES` in `autonomous_grasp/config.py`) is the outer gate of
`RecoveryOrchestrator.next_step`, and `recovery.allowed_actions` cannot widen it. The owner's decisions of
2026-09-29 changed four things here:

* `next_target` joined the auto and dense_clutter profiles.
* A refusal before anything moved falls through to the next action instead of ending the loop.
* The nudge is offered for `ALL_COLLIDED` only.
* `container_agitate` without a declared container is refused when the config loads.

MEASURED through `AutonomousGraspService._run_with_recovery` with a scripted pick. The service still plans
from the dispatcher alone here and hands the loop no memory of failed parts; its wiring of both comes with the
push and the exclusion zones. So:

* `next_target` next to `rescan` is planned and refused before anything moves (no memory of failed parts),
  then the rescan runs.
* `next_target` alone is planned and refused, and the failed pick is final.
* `container_agitate` without a declared container is refused at load. With one declared it loads and is
  never planned, because no built-in profile lists it, and the arm never moves.
* `nudge_target` with a fixture is planned for `ALL_COLLIDED` and refused `refused_no_offset` before the arm
  moves, then the rescan runs. For `NO_VALID_GRASP` it is not offered at all.

The pick itself is scripted (`_pick_inner` replaced on the instance) because what is pinned here is
the loop around it, not perception or motion; the loop, the orchestrator, the dispatcher, the
profile and the executor are the real ones a cell built from config runs.
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace
from typing import Any, Iterator

import numpy as np

from pydantic import ValidationError

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, Pose
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from tests.test_grasping_config_wiring import _calc_and_perception

_FIXTURE = {
    "center_mm": [0.0, -700.0, 150.0],
    "half_extents_mm": [300.0, 300.0, 300.0],
    "max_nudge_mm": 30.0,
}


@dataclasses.dataclass(frozen=True)
class _ScriptedReport:
    """The fields of an `AutonomousGraspReport` the recovery loop reads and rewrites."""

    outcome: Any
    pick_report: Any
    telemetry: dict = dataclasses.field(default_factory=dict)
    recovery_actions: tuple = ()
    uncertainty: Any = None


def _failed(reason: GraspFailureReason) -> _ScriptedReport:
    return _ScriptedReport(
        outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
        pick_report=SimpleNamespace(attempts=[SimpleNamespace(reasons=(reason,))]),
    )


def _succeeded() -> _ScriptedReport:
    return _ScriptedReport(outcome=AutonomousGraspOutcome.SUCCEEDED, pick_report=None)


class _CountingArm:
    """Counts every move; a refused recovery must leave this at zero."""

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


_CONTAINER = {"interior_min_mm": [-200.0, -900.0, 20.0], "interior_max_mm": [200.0, -500.0, 180.0]}


def _service(
    mode: str, allowed: list[str], *, fixture: bool = False, container: bool = False
) -> AutonomousGraspService:
    recovery: dict[str, Any] = {"enabled": True, "allowed_actions": allowed}
    if fixture:
        recovery["fixture"] = dict(_FIXTURE)
    grasping: dict[str, Any] = {"default_mode": mode, "recovery": recovery}
    if container:
        grasping["support"] = {"container": dict(_CONTAINER)}
    cfg = RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping=grasping,
    )
    calculator, perception = _calc_and_perception()
    return AutonomousGraspService.from_robot_config(
        cfg, calculator=calculator, perception=perception
    )


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


class AConfigBuiltRecoveryPlansOnlyWhatItsProfileAllows(unittest.TestCase):
    def test_next_target_beside_rescan_is_refused_then_the_rescan_runs(self) -> None:
        # The dispatcher lists NEXT_TARGET first for NO_VALID_GRASP and the profile allows it now; with no
        # memory of failed parts it cannot skip one, so it is refused before anything moves and falls through.
        service = _service("dense_clutter", ["next_target", "rescan"])
        report, picks = _run(
            service, [_failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()], _CountingArm()
        )
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(picks, 2)
        self.assertEqual(report.telemetry["recovery_trail_actions"], ["next_target", "rescan"])
        self.assertEqual(
            [(step["action"], step["step_result"]) for step in report.recovery_actions],
            [("next_target", "refused_no_part_memory"), ("rescan", "completed")],
        )

    def test_next_target_alone_is_refused_and_the_pick_is_final(self) -> None:
        for mode in ("auto", "dense_clutter"):
            with self.subTest(mode=mode):
                service = _service(mode, ["next_target"])
                report, picks = _run(
                    service,
                    [_failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()],
                    _CountingArm(),
                )
                self.assertEqual(picks, 1)
                self.assertIs(report.outcome, AutonomousGraspOutcome.NO_VALID_GRASP)
                self.assertEqual(report.telemetry["recovery_trail_actions"], ["next_target"])
                self.assertEqual(
                    report.telemetry["recovery_trail_terminal_reason"], "escalated_no_recovery"
                )

    def test_container_agitate_without_a_container_is_refused_at_load(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _service("dense_clutter", ["container_agitate"], fixture=True)
        self.assertIn("declares no interior box", str(caught.exception))

    def test_container_agitate_with_a_container_is_never_planned_and_nothing_moves(self) -> None:
        # ALL_COLLIDED is the jam the dispatcher routes to an agitate; no built-in profile lists it.
        arm = _CountingArm()
        service = _service("dense_clutter", ["container_agitate"], fixture=True, container=True)
        report, picks = _run(
            service, [_failed(GraspFailureReason.ALL_COLLIDED), _succeeded()], arm
        )
        self.assertEqual(picks, 1)
        self.assertEqual(report.telemetry["recovery_trail_actions"], [])
        self.assertEqual(report.recovery_actions, ())
        self.assertEqual(arm.moves, [])

    def test_a_nudge_is_planned_refused_before_the_arm_moves_then_the_rescan_runs(self) -> None:
        arm = _CountingArm()
        service = _service("dense_clutter", ["nudge_target", "rescan"], fixture=True)
        report, picks = _run(
            service, [_failed(GraspFailureReason.ALL_COLLIDED), _succeeded()], arm
        )
        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(picks, 2)
        self.assertEqual(report.telemetry["recovery_trail_actions"], ["nudge_target", "rescan"])
        self.assertEqual(
            report.telemetry["recovery_trail_terminal_reason"], "recovered_success"
        )
        self.assertEqual(
            [(step["action"], step["step_result"]) for step in report.recovery_actions],
            [("nudge_target", "refused_no_offset"), ("rescan", "completed")],
        )
        self.assertEqual(arm.moves, [])

    def test_no_nudge_is_offered_without_a_collision(self) -> None:
        arm = _CountingArm()
        service = _service("dense_clutter", ["nudge_target", "rescan"], fixture=True)
        report, picks = _run(
            service, [_failed(GraspFailureReason.NO_VALID_GRASP), _succeeded()], arm
        )
        self.assertEqual(picks, 2)
        self.assertEqual(report.telemetry["recovery_trail_actions"], ["rescan"])
        self.assertEqual(arm.moves, [])


if __name__ == "__main__":
    unittest.main()
