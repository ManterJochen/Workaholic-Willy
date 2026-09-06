"""Phase T5 — RED tests for the closed-loop retry driver.

The driver wraps an external ``pick()`` callable in a bounded loop
driven by :class:`RecoveryOrchestrator`. After each failure the
driver:

1. extracts the typed failure-reason set from the report,
2. asks the orchestrator for the next plan,
3. executes the plan via the typed executor (or perception
   re-acquisition for RESCAN / NEXT_VIEWPOINT),
4. retries ``pick()``.

The driver returns the final :class:`AutonomousGraspReport` paired
with a typed :class:`RecoveryTrail`.

The service-side wrapper (``AutonomousGraspService.pick_with_recovery``)
is a thin façade over this driver — it does not add new logic; the
loop is fully testable here in isolation.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass, field

from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    GraspBehaviorProfile,
    GraspMode,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.recovery.policy import (
    SceneRecoveryAction,
    SceneRecoveryPolicy,
)
from src.robot.grasping.recovery.orchestrator import (
    RecoveryDispatcher,
    RecoveryOrchestrator,
    run_recovery_loop,
)


def _profile_dense() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.DENSE_CLUTTER,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        recovery_allowed_actions=("rescan", "next_viewpoint", "next_target"),
    )


def _profile_easy() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        recovery_allowed_actions=(),
    )


def _policy_enabled() -> SceneRecoveryPolicy:
    return SceneRecoveryPolicy(
        enabled=True,
        allowed_actions=(
            SceneRecoveryAction.RESCAN,
            SceneRecoveryAction.NEXT_VIEWPOINT,
        ),
        max_recovery_actions=3,
        apply_modes=("auto", "dense_clutter", "dense_autonomous"),
    )


def _orch() -> RecoveryOrchestrator:
    """Default orchestrator with no-op strategies (driver only uses dispatcher)."""

    from src.robot.grasping.recovery.policy import NoRecoveryStrategy

    return RecoveryOrchestrator(
        dispatcher=RecoveryDispatcher(),
        strategies={
            SceneRecoveryAction.RESCAN: NoRecoveryStrategy(),
            SceneRecoveryAction.NEXT_VIEWPOINT: NoRecoveryStrategy(),
            SceneRecoveryAction.NEXT_TARGET: NoRecoveryStrategy(),
            SceneRecoveryAction.NUDGE_TARGET: NoRecoveryStrategy(),
        },
        # Force the orchestrator to plan directly from the dispatcher mapping
        # without running per-action strategies (driver-test focus).
        bypass_strategies=True,
    )


# ---------------------------------------------------------------------------
# A tiny report builder. We don't construct a real AutonomousGraspReport
# tree — the driver only consults a small public surface (outcome +
# failure_reasons). Define a structural fake.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _FakeReport:
    outcome: AutonomousGraspOutcome
    failure_reasons: tuple[GraspFailureReason, ...] = ()
    attempt_id: str = "fake"
    telemetry: dict = field(default_factory=dict)


def _picker_failing_then_success(
    failure_sequence: tuple[tuple[AutonomousGraspOutcome, tuple[GraspFailureReason, ...]], ...],
):
    """Return a callable that yields each report in order, then SUCCESS forever."""

    state = {"i": 0}

    def pick():
        i = state["i"]
        state["i"] += 1
        if i < len(failure_sequence):
            outcome, reasons = failure_sequence[i]
            return _FakeReport(outcome=outcome, failure_reasons=reasons)
        return _FakeReport(outcome=AutonomousGraspOutcome.SUCCEEDED)

    return pick, state


def _picker_always_failing(
    outcome: AutonomousGraspOutcome,
    reasons: tuple[GraspFailureReason, ...],
):
    def pick():
        return _FakeReport(outcome=outcome, failure_reasons=reasons)

    return pick


class RecoveryLoopHappyPathTests(unittest.TestCase):
    def test_first_attempt_success_yields_empty_trail(self) -> None:
        def pick():
            return _FakeReport(outcome=AutonomousGraspOutcome.SUCCEEDED)
        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_dense(),
            policy=_policy_enabled(),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
        )
        self.assertEqual(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(trail.entries, ())
        self.assertEqual(trail.terminal_reason, "recovered_success")

    def test_recover_after_one_rescan(self) -> None:
        pick, state = _picker_failing_then_success(
            (
                (
                    AutonomousGraspOutcome.NO_VALID_GRASP,
                    (GraspFailureReason.RESCAN_RECOMMENDED,),
                ),
            )
        )
        acq_calls = {"n": 0}

        def acquire():
            acq_calls["n"] += 1

        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_dense(),
            policy=_policy_enabled(),
            orchestrator=_orch(),
            frame_acquirer=acquire,
        )
        self.assertEqual(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(state["i"], 2)
        self.assertEqual(acq_calls["n"], 1)
        self.assertEqual(len(trail.entries), 1)
        self.assertEqual(trail.entries[0].plan_action, SceneRecoveryAction.RESCAN)
        self.assertTrue(trail.entries[0].executed)
        self.assertEqual(trail.terminal_reason, "recovered_success")


class RecoveryLoopBudgetExhaustionTests(unittest.TestCase):
    def test_exhausted_budget_terminates_with_typed_trail(self) -> None:
        pick = _picker_always_failing(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            (GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_dense(),
            policy=SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(
                    SceneRecoveryAction.RESCAN,
                    SceneRecoveryAction.NEXT_VIEWPOINT,
                ),
                max_recovery_actions=2,
                apply_modes=("dense_clutter",),
            ),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
        )
        self.assertNotEqual(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(len(trail.entries), 2)
        self.assertEqual(trail.terminal_reason, "exhausted_budget")


class RecoveryLoopAntiLoopTests(unittest.TestCase):
    def test_anti_loop_blocks_when_dispatch_repeats(self) -> None:
        """When dispatch always returns the same action for the same class
        and that action has been tried, the loop terminates with
        ``anti_loop_blocked``."""

        pick = _picker_always_failing(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            (GraspFailureReason.TRY_NEXT_CANDIDATE,),
        )
        final, trail = run_recovery_loop(
            pick=pick,
            profile=GraspBehaviorProfile(
                mode=GraspMode.DENSE_CLUTTER,
                sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
                recovery_allowed_actions=("next_target",),
            ),
            policy=SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(SceneRecoveryAction.NEXT_TARGET,),
                max_recovery_actions=5,
                apply_modes=("dense_clutter",),
            ),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
        )
        # TRY_NEXT_CANDIDATE maps to (NEXT_TARGET,); after one attempt the
        # anti-loop refuses the same (action, class) pair → terminate.
        self.assertEqual(len(trail.entries), 1)
        self.assertEqual(trail.terminal_reason, "anti_loop_blocked")


class RecoveryLoopEscalationTests(unittest.TestCase):
    def test_escalation_failure_class_yields_no_recovery(self) -> None:
        pick = _picker_always_failing(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            (GraspFailureReason.SEMANTIC_REJECTED,),
        )
        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_dense(),
            policy=_policy_enabled(),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
        )
        self.assertEqual(trail.entries, ())
        self.assertEqual(trail.terminal_reason, "escalated_no_recovery")


class RecoveryLoopEasyModeTests(unittest.TestCase):
    def test_easy_mode_short_circuits_without_recovery(self) -> None:
        pick = _picker_always_failing(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            (GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        acq_calls = {"n": 0}

        def acquire():
            acq_calls["n"] += 1

        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_easy(),
            policy=_policy_enabled(),
            orchestrator=_orch(),
            frame_acquirer=acquire,
        )
        # EASY mode hard-out: orchestrator returns None on first call.
        self.assertEqual(trail.entries, ())
        self.assertEqual(trail.terminal_reason, "recovery_disabled")
        self.assertEqual(acq_calls["n"], 0)


class RecoveryLoopDisabledPolicyTests(unittest.TestCase):
    def test_disabled_policy_skips_orchestrator(self) -> None:
        pick = _picker_always_failing(
            AutonomousGraspOutcome.NO_VALID_GRASP,
            (GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        final, trail = run_recovery_loop(
            pick=pick,
            profile=_profile_dense(),
            policy=SceneRecoveryPolicy(enabled=False),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
        )
        self.assertEqual(trail.entries, ())
        self.assertEqual(trail.terminal_reason, "recovery_disabled")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
