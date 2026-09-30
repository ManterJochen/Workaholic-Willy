"""The recovery loop's non-motion rules, as the owner set them on 2026-09-29.

* **Per-action strategy mode.** The orchestrator no longer plans every action from the dispatcher alone.
  ``RESCAN`` plans directly. Every other action needs a strategy, and without one it is skipped for the next
  action in the row. A physical action can never be planned directly: it would carry no motion to check.
* **A refusal before anything moved falls through** (any ``refused_*`` outcome). The loop tries the next
  action in the row. The refusal stays in the trail and spends no action budget. ``aborted_motion_failed`` still
  ends the loop, because the arm did move.
* **The fall-through always ends.** A refusal blocks its action for every reason of the failure. A strategy's
  plan for another action than its own is dropped. And one failure can have no more refusals than its rows
  hold actions: past that the loop stops, whatever the orchestrator says. Each test that could spin has a
  watchdog, so a regression fails instead of hanging the suite.
* **The executor clamps.** An agitate wider than ``max_agitate_amplitude_mm`` is cut to the limit, and it checks
  every waypoint before it moves any. A nudge is never driven here: the push runs inside the pick attempt, and the
  executor refuses the loop's nudge before anything moves (``refused_push_runs_in_the_pick``), so the loop falls
  through. The scene-blind offset it used to drive left on 2026-09-29.
* **Dispatcher rows.** The nudge is offered only for ``ALL_COLLIDED``. ``NEXT_TARGET`` is allowed after
  ``MOTION_PLAN_REFUSED``, where the next motion is still judged in full.
* **NEXT_TARGET is "rescan, skipping the failed part".** It runs only with the caller's memory of failed parts.
  Without that memory it is refused before anything moves, and the loop falls through.
* **The loop's end maps to the two outcomes nobody set before:** ``recovery_exhausted`` and
  ``unsafe_recovery_refused``.
* **Profiles.** ``next_target`` is in ``auto`` and ``dense_clutter``. ``nudge_target`` is in ``dense_clutter``
  only.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    GraspBehaviorProfile,
    GraspMode,
    _profile_for,
)
from src.robot.grasping.recovery.orchestrator import (
    DEFAULT_DIRECT_ACTIONS,
    OUTCOME_RECOVERY_EXHAUSTED,
    OUTCOME_UNSAFE_RECOVERY_REFUSED,
    RecoveryDispatcher,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
    run_recovery_loop,
    terminal_outcome_for_trail,
)
from src.robot.grasping.recovery.policy import (
    REFUSED_PUSH_RUNS_IN_THE_PICK,
    FixtureEnvelope,
    SceneRecoveryAction as A,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
    execute_recovery_motion,
    push_permitted,
    refused_before_motion,
)
from src.robot.grasping.types.feedback import GraspFailureReason as R
from src.robot.grasping.types.modes import GraspSamplingMode

_FIXTURE = FixtureEnvelope(center_mm=(0.0, 0.0, 200.0), half_extents_mm=(100.0, 100.0, 100.0), max_nudge_mm=30.0,
                           max_agitate_amplitude_mm=20.0)


def _profile(*actions: str, mode: GraspMode = GraspMode.DENSE_CLUTTER) -> GraspBehaviorProfile:
    return GraspBehaviorProfile(mode=mode, sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
                                recovery_allowed_actions=tuple(actions))


def _policy(*actions: A, max_actions: int = 3, fixture: Optional[FixtureEnvelope] = _FIXTURE,
            **kwargs: Any) -> SceneRecoveryPolicy:
    return SceneRecoveryPolicy(enabled=True, allowed_actions=tuple(actions), max_recovery_actions=max_actions,
                               fixture=fixture, **kwargs)


@dataclass(frozen=True)
class _Report:
    outcome: Any
    failure_reasons: tuple = ()
    tag: str = ""


def _script(*reports: _Report) -> tuple[Any, list[int]]:
    calls: list[int] = []
    queue = list(reports)

    def pick() -> _Report:
        calls.append(1)
        return queue.pop(0) if queue else _Report(AutonomousGraspOutcome.SUCCEEDED)

    return pick, calls


def _failed(*reasons: R, tag: str = "") -> _Report:
    return _Report(AutonomousGraspOutcome.NO_VALID_GRASP, tuple(reasons), tag)


def _tcp(xyz: tuple[float, float, float] = (0.0, 0.0, 200.0)) -> Pose:
    return Pose(position_mm=np.asarray(xyz, dtype=np.float64), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame=Frame.BASE)


@dataclass
class _Arm:
    """Records every move. ``fail`` makes every move come back refused by the safety layer."""

    fail: bool = False
    moves: list[Pose] = field(default_factory=list)

    def get_tcp_pose(self) -> Pose:
        return _tcp()

    def move(self, pose: Pose, **_: Any) -> MotionResult:
        self.moves.append(pose)
        if self.fail:
            return MotionResult(status=MotionStatus.WORKSPACE_REJECTED, command=MotionCommand.MOVE_TO,
                                target_pose=pose, message="refused")
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _NudgeStrategy:
    """Plans a nudge of ``offset`` for any context: the executor, not the strategy, is under test."""

    def __init__(self, offset: tuple[float, float, float] = (10.0, 0.0, 0.0)) -> None:
        self.offset = offset

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(action=A.NUDGE_TARGET, reason="test_nudge", nudge_offset_mm=self.offset)


class _AgitateStrategy:
    """Plans an agitate of ``amplitude`` for any context: the one physical action the loop still drives."""

    def __init__(self, amplitude: float = 5.0) -> None:
        self.amplitude = amplitude

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(action=A.CONTAINER_AGITATE, reason="test_agitate",
                                 agitate_amplitude_mm=self.amplitude)


def _ctx(profile: GraspBehaviorProfile, policy: SceneRecoveryPolicy, *reasons: R) -> SceneRecoveryContext:
    return SceneRecoveryContext(profile=profile, policy=policy, last_outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
                                failure_reasons=tuple(reasons))


#: A loop that never ends must fail its test, not hang the suite: every fake below raises past this many plans.
_WATCHDOG_PLANS = 25


class _Watchdog:
    """Wraps an orchestrator and fails the test past :data:`_WATCHDOG_PLANS` plans."""

    def __init__(self, inner: RecoveryOrchestrator) -> None:
        self.inner = inner
        self.dispatcher = inner.dispatcher
        self.calls = 0

    def next_step(self, context: SceneRecoveryContext, *,
                  typed_history: tuple[Any, ...] = ()) -> Optional[SceneRecoveryPlan]:
        self.calls += 1
        if self.calls > _WATCHDOG_PLANS:
            raise AssertionError(f"the recovery loop asked for {self.calls} plans without ending")
        return self.inner.next_step(context, typed_history=typed_history)


class _PlansAnotherAction:
    """Registered for NEXT_TARGET, it plans a nudge. The strategy protocol is open, so nothing stops that."""

    def __init__(self) -> None:
        self.calls = 0

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        self.calls += 1
        if self.calls > _WATCHDOG_PLANS:
            raise AssertionError(f"the strategy was asked {self.calls} times without the loop ending")
        return SceneRecoveryPlan(action=A.NUDGE_TARGET, reason="another_action")


class _KeepsPlanningNextTarget:
    """An orchestrator that ignores what was refused: the loop's own cap has to end it."""

    dispatcher = RecoveryDispatcher()

    def __init__(self) -> None:
        self.calls = 0

    def next_step(self, context: SceneRecoveryContext, *,
                  typed_history: tuple[Any, ...] = ()) -> Optional[SceneRecoveryPlan]:
        self.calls += 1
        if self.calls > _WATCHDOG_PLANS:
            raise AssertionError(f"the recovery loop asked for {self.calls} plans without ending")
        return SceneRecoveryPlan(action=A.NEXT_TARGET, reason="stuck")


class PerActionStrategyMode(unittest.TestCase):
    def test_rescan_plans_without_a_strategy(self) -> None:
        self.assertEqual(DEFAULT_DIRECT_ACTIONS, frozenset({A.RESCAN}))
        plan = RecoveryOrchestrator(dispatcher=RecoveryDispatcher()).next_step(
            _ctx(_profile("rescan"), _policy(A.RESCAN), R.RESCAN_RECOMMENDED))
        assert plan is not None
        self.assertEqual((plan.action, plan.reason), (A.RESCAN, "dispatcher:rescan"))
        self.assertEqual(plan.telemetry["orchestrator"], "direct_action")

    def test_an_action_without_a_strategy_is_skipped_for_the_next_in_its_row(self) -> None:
        # ALL_COLLIDED lists the nudge before the rescan. Allowed, fixture declared, but no strategy.
        plan = RecoveryOrchestrator(dispatcher=RecoveryDispatcher()).next_step(
            _ctx(_profile("rescan", "nudge_target"), _policy(A.NUDGE_TARGET, A.RESCAN), R.ALL_COLLIDED))
        assert plan is not None
        self.assertIs(plan.action, A.RESCAN)

    def test_a_strategy_is_consulted_where_one_is_registered(self) -> None:
        orchestrator = RecoveryOrchestrator(dispatcher=RecoveryDispatcher(),
                                            strategies={A.NUDGE_TARGET: _NudgeStrategy()})
        plan = orchestrator.next_step(
            _ctx(_profile("rescan", "nudge_target"), _policy(A.NUDGE_TARGET, A.RESCAN), R.ALL_COLLIDED))
        assert plan is not None
        self.assertEqual((plan.action, plan.reason), (A.NUDGE_TARGET, "test_nudge"))

    def test_next_target_can_be_planned_directly_when_the_caller_says_so(self) -> None:
        orchestrator = RecoveryOrchestrator(dispatcher=RecoveryDispatcher(),
                                            direct_actions=frozenset({A.RESCAN, A.NEXT_TARGET}))
        plan = orchestrator.next_step(
            _ctx(_profile("rescan", "next_target"), _policy(A.NEXT_TARGET, A.RESCAN), R.IK_FAILED))
        assert plan is not None
        self.assertIs(plan.action, A.NEXT_TARGET)

    def test_a_physical_action_is_never_planned_directly(self) -> None:
        for action in (A.NUDGE_TARGET, A.CONTAINER_AGITATE, A.NONE):
            with self.subTest(action=action), self.assertRaises(ValueError):
                RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), direct_actions=frozenset({action}))

    def test_bypass_is_still_the_test_seam_that_plans_everything_by_name(self) -> None:
        plan = RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), bypass_strategies=True).next_step(
            _ctx(_profile("rescan", "nudge_target"), _policy(A.NUDGE_TARGET, A.RESCAN), R.ALL_COLLIDED))
        assert plan is not None
        self.assertEqual((plan.action, plan.nudge_offset_mm), (A.NUDGE_TARGET, None))


class ARefusalBeforeMotionFallsThrough(unittest.TestCase):
    def test_a_refused_nudge_falls_through_to_the_rescan_and_spends_no_budget(self) -> None:
        # The bypass plans the nudge; the push runs inside the pick attempt, so the loop's nudge is refused before
        # the arm moves. With one action allowed per pick, the rescan must still run: the refusal spent nothing.
        pick, calls = _script(_failed(R.ALL_COLLIDED))
        arm = _Arm()
        final, trail = run_recovery_loop(
            pick=pick, profile=_profile("rescan", "nudge_target"),
            policy=_policy(A.NUDGE_TARGET, A.RESCAN, max_actions=1),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), bypass_strategies=True),
            frame_acquirer=lambda: None, arm=arm)
        self.assertIs(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual([(e.plan_action, e.outcome, e.executed) for e in trail.entries],
                         [(A.NUDGE_TARGET, REFUSED_PUSH_RUNS_IN_THE_PICK, False),
                          (A.RESCAN, "completed", True)])
        self.assertEqual(trail.terminal_reason, "recovered_success")
        self.assertEqual(len(calls), 2)  # the refusal did not re-run the pick
        self.assertEqual(arm.moves, [])

    def test_a_planned_nudge_is_refused_with_or_without_an_arm_and_falls_through(self) -> None:
        # Even a strategy that plans an offset gets no motion from the loop: the push runs inside the pick attempt.
        for arm in (None, _Arm()):
            with self.subTest(arm=arm):
                pick, _ = _script(_failed(R.ALL_COLLIDED))
                _, trail = run_recovery_loop(
                    pick=pick, profile=_profile("rescan", "nudge_target"), policy=_policy(A.NUDGE_TARGET, A.RESCAN),
                    orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(),
                                                      strategies={A.NUDGE_TARGET: _NudgeStrategy()}),
                    frame_acquirer=lambda: None, arm=arm)  # type: ignore[arg-type]
                self.assertEqual([e.outcome for e in trail.entries], [REFUSED_PUSH_RUNS_IN_THE_PICK, "completed"])
                if arm is not None:
                    self.assertEqual(arm.moves, [])

    def test_a_refusal_with_nothing_after_it_ends_escalated_not_anti_loop(self) -> None:
        pick, calls = _script(_failed(R.ALL_COLLIDED))
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("nudge_target"), policy=_policy(A.NUDGE_TARGET),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), bypass_strategies=True),
            frame_acquirer=lambda: None, arm=_Arm())
        self.assertEqual([e.outcome for e in trail.entries], [REFUSED_PUSH_RUNS_IN_THE_PICK])
        self.assertEqual(trail.terminal_reason, "escalated_no_recovery")
        self.assertEqual(len(calls), 1)
        self.assertIsNone(terminal_outcome_for_trail(trail))

    def test_a_refused_action_is_not_planned_again_for_the_same_failure(self) -> None:
        # ALL_COLLIDED and NO_CANDIDATES_GENERATED both list NEXT_TARGET; refused once, it is tried once.
        pick, _ = _script(_failed(R.ALL_COLLIDED, R.NO_CANDIDATES_GENERATED))
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("rescan", "next_target"), policy=_policy(A.NEXT_TARGET, A.RESCAN),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), bypass_strategies=True),
            frame_acquirer=lambda: None)
        self.assertEqual([e.plan_action for e in trail.entries], [A.NEXT_TARGET, A.RESCAN])

    def test_two_reasons_whose_rows_both_list_the_refused_action_end_after_one_refusal(self) -> None:
        # Only NEXT_TARGET is allowed and nothing remembers failed parts, so it is refused. Both reasons' rows
        # list it. The refusal must block it for both, or the second reason would plan it again, forever.
        pick, calls = _script(_failed(R.ALL_COLLIDED, R.NO_CANDIDATES_GENERATED))
        watchdog = _Watchdog(RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), bypass_strategies=True))
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("next_target"), policy=_policy(A.NEXT_TARGET),
            orchestrator=watchdog,  # type: ignore[arg-type]
            frame_acquirer=lambda: None)
        self.assertEqual([(e.plan_action, e.outcome) for e in trail.entries],
                         [(A.NEXT_TARGET, "refused_no_part_memory")])
        self.assertEqual(trail.terminal_reason, "escalated_no_recovery")
        self.assertEqual(len(calls), 1)
        self.assertEqual(watchdog.calls, 2)

    def test_a_strategy_that_plans_another_action_than_its_own_is_not_followed(self) -> None:
        # Refused, that nudge would block (nudge, reason), while the row asks for NEXT_TARGET again and again.
        strategy = _PlansAnotherAction()
        orchestrator = RecoveryOrchestrator(dispatcher=RecoveryDispatcher(), strategies={A.NEXT_TARGET: strategy})
        plan = orchestrator.next_step(_ctx(_profile("rescan", "next_target", "nudge_target"),
                                           _policy(A.NEXT_TARGET, A.NUDGE_TARGET, A.RESCAN), R.ALL_COLLIDED))
        assert plan is not None
        self.assertIs(plan.action, A.RESCAN)
        pick, calls = _script(_failed(R.NO_VALID_GRASP))
        arm = _Arm()
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("next_target"), policy=_policy(A.NEXT_TARGET), orchestrator=orchestrator,
            frame_acquirer=lambda: None, arm=arm)
        self.assertEqual((trail.entries, trail.terminal_reason), ((), "escalated_no_recovery"))
        self.assertEqual((len(calls), arm.moves), (1, []))

    def test_the_loop_ends_even_if_an_orchestrator_keeps_planning_what_was_refused(self) -> None:
        # IK_FAILED's row holds two actions, so at most two refusals can be new for one failure. A third means
        # the orchestrator is not blocking what was refused, and the loop stops rather than spin in place.
        pick, calls = _script(_failed(R.IK_FAILED))
        stuck = _KeepsPlanningNextTarget()
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("rescan", "next_target"), policy=_policy(A.NEXT_TARGET, A.RESCAN),
            orchestrator=stuck,  # type: ignore[arg-type]
            frame_acquirer=lambda: None)
        self.assertEqual([e.outcome for e in trail.entries], ["refused_no_part_memory"] * 3)
        self.assertEqual(trail.terminal_reason, "escalated_no_recovery")
        self.assertEqual((len(calls), stuck.calls), (1, 3))

    def test_a_motion_that_failed_still_ends_the_loop_as_unsafe(self) -> None:
        # The agitate is the one physical action the loop still drives; its first waypoint is refused after the
        # agitate began, so the loop ends there.
        pick, calls = _script(_failed(R.ALL_COLLIDED))
        arm = _Arm(fail=True)
        final, trail = run_recovery_loop(
            pick=pick, profile=_profile("rescan", "container_agitate"),
            policy=_policy(A.CONTAINER_AGITATE, A.RESCAN),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(),
                                              strategies={A.CONTAINER_AGITATE: _AgitateStrategy()}),
            frame_acquirer=lambda: None, arm=arm)
        self.assertEqual([e.outcome for e in trail.entries], ["aborted_motion_failed"])
        self.assertEqual(trail.terminal_reason, "escalated_no_recovery")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(arm.moves), 1)
        self.assertEqual(terminal_outcome_for_trail(trail), OUTCOME_UNSAFE_RECOVERY_REFUSED)

    def test_which_outcomes_count_as_before_motion(self) -> None:
        for outcome in (REFUSED_PUSH_RUNS_IN_THE_PICK, "refused_no_arm", "refused_envelope_violation",
                        "refused_no_part_memory", "refused_agitate_disabled"):
            self.assertTrue(refused_before_motion(outcome), outcome)
        for outcome in ("aborted_motion_failed", "completed", "skipped_no_action", ""):
            self.assertFalse(refused_before_motion(outcome), outcome)


class NextTargetSkipsTheFailedPart(unittest.TestCase):
    def _run(self, reports: tuple[_Report, ...], skip: Any, profile: Optional[GraspBehaviorProfile] = None,
             policy: Optional[SceneRecoveryPolicy] = None) -> tuple[Any, RecoveryTrail, list[int], list[int]]:
        pick, calls = _script(*reports)
        looked: list[int] = []
        final, trail = run_recovery_loop(
            pick=pick, profile=profile or _profile("rescan", "next_target"),
            policy=policy or _policy(A.NEXT_TARGET, A.RESCAN),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher(),
                                              direct_actions=frozenset({A.RESCAN, A.NEXT_TARGET})),
            frame_acquirer=lambda: looked.append(1), skip_failed_part=skip)
        return final, trail, calls, looked

    def test_the_failed_report_is_handed_to_the_callers_memory_then_the_scene_is_seen_again(self) -> None:
        handed: list[str] = []

        def skip(report: _Report) -> bool:
            handed.append(report.tag)
            return True

        final, trail, calls, looked = self._run((_failed(R.IK_FAILED, tag="first"),), skip)
        self.assertIs(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(handed, ["first"])
        self.assertEqual(looked, [1])
        self.assertEqual([(e.plan_action, e.outcome) for e in trail.entries], [(A.NEXT_TARGET, "completed")])
        self.assertEqual(len(calls), 2)

    def test_without_the_memory_next_target_is_refused_and_the_rescan_runs(self) -> None:
        final, trail, _, looked = self._run((_failed(R.IK_FAILED),), None)
        self.assertIs(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual([(e.plan_action, e.outcome) for e in trail.entries],
                         [(A.NEXT_TARGET, "refused_no_part_memory"), (A.RESCAN, "completed")])
        self.assertEqual(looked, [1])

    def test_a_report_that_names_no_part_is_refused_and_the_rescan_runs(self) -> None:
        _, trail, _, _ = self._run((_failed(R.IK_FAILED),), lambda report: False)
        self.assertEqual([e.outcome for e in trail.entries], ["refused_no_failed_part", "completed"])

    def test_next_target_is_allowed_after_a_motion_plan_refusal(self) -> None:
        skipped: list[int] = []
        _, trail, _, _ = self._run((_failed(R.MOTION_PLAN_REFUSED),), lambda r: skipped.append(1) or True)
        self.assertEqual([e.plan_action for e in trail.entries], [A.NEXT_TARGET])
        self.assertEqual(skipped, [1])

    def test_the_budget_counts_next_target_once_it_ran(self) -> None:
        _, trail, _, _ = self._run((_failed(R.IK_FAILED), _failed(R.IK_FAILED), _failed(R.IK_FAILED)),
                                   lambda r: True, policy=_policy(A.NEXT_TARGET, A.RESCAN, max_actions=2))
        self.assertEqual([e.plan_action for e in trail.entries], [A.NEXT_TARGET, A.RESCAN])
        self.assertEqual(trail.terminal_reason, "exhausted_budget")
        self.assertEqual(terminal_outcome_for_trail(trail), OUTCOME_RECOVERY_EXHAUSTED)


class TheDispatcherRows(unittest.TestCase):
    def setUp(self) -> None:
        self.dispatcher = RecoveryDispatcher()

    def test_the_nudge_is_offered_for_all_collided_only(self) -> None:
        self.assertEqual(self.dispatcher.actions_for(R.ALL_COLLIDED),
                         (A.NEXT_TARGET, A.NUDGE_TARGET, A.CONTAINER_AGITATE, A.RESCAN))
        for reason in (R.NO_VALID_GRASP, R.NO_CANDIDATES_GENERATED):
            with self.subTest(reason=reason):
                self.assertEqual(self.dispatcher.actions_for(reason), (A.NEXT_TARGET, A.RESCAN))
        rows_with_nudge = {reason for reason in R if A.NUDGE_TARGET in self.dispatcher.actions_for(reason)}
        self.assertEqual(rows_with_nudge, {R.ALL_COLLIDED})

    def test_a_motion_plan_refusal_may_skip_the_part_but_never_moves_for_it(self) -> None:
        actions = self.dispatcher.actions_for(R.MOTION_PLAN_REFUSED)
        self.assertEqual(actions, (A.NEXT_TARGET, A.RESCAN))
        self.assertNotIn(A.NUDGE_TARGET, actions)
        self.assertNotIn(A.CONTAINER_AGITATE, actions)


class TheExecutorClamps(unittest.TestCase):
    def test_a_nudge_is_refused_whatever_it_carries_and_nothing_moves(self) -> None:
        # Too long, within the limit, with no fixture, NaN: the executor drives none of them any more.
        cases = {
            "longer than max_nudge_mm": ((40.0, 0.0, 0.0), _policy(A.NUDGE_TARGET)),
            "within the limit": ((0.0, 12.0, 0.0), _policy(A.NUDGE_TARGET)),
            "no fixture": ((5.0, 0.0, 0.0), SceneRecoveryPolicy(enabled=True)),
            "NaN": ((float("nan"), 0.0, 0.0), _policy(A.NUDGE_TARGET)),
        }
        for name, (offset, policy) in cases.items():
            with self.subTest(case=name):
                arm = _Arm()
                report = execute_recovery_motion(
                    arm=arm,  # type: ignore[arg-type]
                    plan=SceneRecoveryPlan(action=A.NUDGE_TARGET, nudge_offset_mm=offset),
                    policy=policy, current_tcp=_tcp())
                self.assertEqual((report.executed, report.outcome), (False, REFUSED_PUSH_RUNS_IN_THE_PICK))
                self.assertEqual(arm.moves, [])

    def test_an_agitate_wider_than_its_amplitude_limit_is_cut_to_it(self) -> None:
        arm = _Arm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=SceneRecoveryPlan(action=A.CONTAINER_AGITATE, agitate_amplitude_mm=50.0),
            policy=_policy(A.CONTAINER_AGITATE), current_tcp=_tcp())
        self.assertTrue(report.executed)
        self.assertEqual([round(float(p.position_mm[0]), 6) for p in arm.moves], [20.0, -20.0, 0.0])
        self.assertEqual(report.telemetry["amplitude_mm"], 20.0)
        self.assertTrue(report.telemetry["clamped"])

    def test_a_nan_amplitude_is_refused_before_anything_moves(self) -> None:
        # Every envelope comparison with NaN is False, so FixtureEnvelope.contains would let it through.
        arm = _Arm()
        agitate = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=SceneRecoveryPlan(action=A.CONTAINER_AGITATE, agitate_amplitude_mm=float("nan")),
            policy=_policy(A.CONTAINER_AGITATE), current_tcp=_tcp())
        self.assertEqual((agitate.executed, agitate.outcome), (False, "refused_agitate_disabled"))
        self.assertEqual(arm.moves, [])

    def test_an_agitate_checks_every_waypoint_before_it_moves_any(self) -> None:
        # Waypoint 0 (x = 10) is inside the 12 mm half-extent, waypoint 1 (x = 15) is not. Nothing may move:
        # a refusal is a promise that nothing moved.
        fixture = FixtureEnvelope(center_mm=(0.0, 0.0, 200.0), half_extents_mm=(12.0, 50.0, 50.0),
                                  max_agitate_amplitude_mm=5.0, agitate_contact_depth_mm=20.0,
                                  agitate_sweep_offset_mm=10.0)
        arm = _Arm()
        report = execute_recovery_motion(
            arm=arm,  # type: ignore[arg-type]
            plan=SceneRecoveryPlan(action=A.CONTAINER_AGITATE, agitate_amplitude_mm=5.0),
            policy=_policy(A.CONTAINER_AGITATE, fixture=fixture), current_tcp=_tcp())
        self.assertEqual((report.executed, report.outcome), (False, "refused_envelope_violation"))
        self.assertEqual(report.telemetry["waypoint_index"], 1)
        self.assertEqual(arm.moves, [])


class TheLoopsEndMapsToAnOutcome(unittest.TestCase):
    @staticmethod
    def _trail(terminal: str, *steps: tuple[A, str, bool]) -> RecoveryTrail:
        return RecoveryTrail(
            entries=tuple(RecoveryTrailEntry(attempt_index=i, failure_reason=R.ALL_COLLIDED, plan_action=a,
                                             plan_reason="t", executed=ran, outcome=outcome)
                          for i, (a, outcome, ran) in enumerate(steps)),
            terminal_reason=terminal)

    def test_the_strings_are_the_services_outcomes(self) -> None:
        self.assertEqual(OUTCOME_RECOVERY_EXHAUSTED, AutonomousGraspOutcome.RECOVERY_EXHAUSTED.value)
        self.assertEqual(OUTCOME_UNSAFE_RECOVERY_REFUSED, AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED.value)

    def test_each_end(self) -> None:
        cases = {
            "success keeps the pick's own": (self._trail("recovered_success", (A.RESCAN, "completed", True)), None),
            "nothing ran keeps the pick's own": (self._trail("recovery_disabled"), None),
            "budget spent after actions": (self._trail("exhausted_budget", (A.RESCAN, "completed", True)),
                                           OUTCOME_RECOVERY_EXHAUSTED),
            "every action tried": (self._trail("anti_loop_blocked", (A.RESCAN, "completed", True)),
                                   OUTCOME_RECOVERY_EXHAUSTED),
            "only refusals": (self._trail("escalated_no_recovery",
                                          (A.NUDGE_TARGET, REFUSED_PUSH_RUNS_IN_THE_PICK, False)), None),
            "a motion failed": (self._trail("escalated_no_recovery", (A.RESCAN, "completed", True),
                                            (A.CONTAINER_AGITATE, "aborted_motion_failed", False)),
                                OUTCOME_UNSAFE_RECOVERY_REFUSED),
            "escalated after a rescan keeps the pick's own": (
                self._trail("escalated_no_recovery", (A.RESCAN, "completed", True)), None),
        }
        for name, (trail, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(terminal_outcome_for_trail(trail), expected)

    def test_a_loop_allowed_no_action_keeps_the_picks_own_outcome(self) -> None:
        # max_recovery_actions = 0 ends the loop as exhausted_budget with nothing run: that is not
        # recovery_exhausted, since no recovery was tried.
        pick, calls = _script(_failed(R.ALL_COLLIDED))
        _, trail = run_recovery_loop(
            pick=pick, profile=_profile("rescan"), policy=_policy(A.RESCAN, max_actions=0),
            orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher()), frame_acquirer=lambda: None)
        self.assertEqual((trail.entries, trail.terminal_reason), ((), "exhausted_budget"))
        self.assertIsNone(terminal_outcome_for_trail(trail))
        self.assertEqual(len(calls), 1)


class WhereAPushMayRun(unittest.TestCase):
    def test_dense_clutter_with_the_nudge_allowed_and_a_fixture(self) -> None:
        dense = _profile_for(GraspMode.DENSE_CLUTTER)
        self.assertTrue(push_permitted(dense, _policy(A.NUDGE_TARGET, A.RESCAN)))

    def test_every_gate_closes_it(self) -> None:
        dense = _profile_for(GraspMode.DENSE_CLUTTER)
        closed = {
            "auto": (_profile_for(GraspMode.AUTO), _policy(A.NUDGE_TARGET, A.RESCAN)),
            "easy": (_profile_for(GraspMode.EASY), _policy(A.NUDGE_TARGET, A.RESCAN)),
            "disabled": (dense, SceneRecoveryPolicy(enabled=False)),
            "disabled with the nudge allowed": (dense, SceneRecoveryPolicy(
                enabled=False, allowed_actions=(A.NUDGE_TARGET, A.RESCAN), fixture=_FIXTURE)),
            "not allowed": (dense, _policy(A.RESCAN)),
            "mode not applied": (dense, _policy(A.NUDGE_TARGET, apply_modes=("auto",))),
            "budget zero": (dense, _policy(A.NUDGE_TARGET, per_action_budget={A.NUDGE_TARGET: 0})),
            "max actions zero": (dense, _policy(A.NUDGE_TARGET, max_actions=0)),
        }
        for name, (profile, policy) in closed.items():
            with self.subTest(case=name):
                self.assertFalse(push_permitted(profile, policy))

    def test_no_fixture_closes_it_even_where_a_policy_slipped_past_its_own_check(self) -> None:
        # SceneRecoveryPolicy refuses a nudge without a fixture when it is built; the gate checks again.
        policy = _policy(A.NUDGE_TARGET, A.RESCAN)
        object.__setattr__(policy, "fixture", None)
        self.assertFalse(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), policy))


class TheProfiles(unittest.TestCase):
    def test_next_target_in_auto_and_dense_clutter_the_push_in_dense_clutter_only(self) -> None:
        self.assertEqual(_profile_for(GraspMode.EASY).recovery_allowed_actions, ())
        self.assertEqual(_profile_for(GraspMode.AUTO).recovery_allowed_actions, ("rescan", "next_target"))
        self.assertEqual(_profile_for(GraspMode.DENSE_CLUTTER).recovery_allowed_actions,
                         ("rescan", "next_target", "nudge_target"))
        for mode in GraspMode:
            with self.subTest(mode=mode):
                self.assertNotIn("container_agitate", _profile_for(mode).recovery_allowed_actions)


if __name__ == "__main__":
    unittest.main()
