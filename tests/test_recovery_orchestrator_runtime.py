"""Bucket-A feature: the T5 RecoveryOrchestrator is now wired into the live
service ``pick()`` path (opt-in via ``recovery_orchestrator.enabled``), and the
previously-inert ``uncertainty_recovery_aggressive_threshold`` knob drives the
recovery action ordering through a per-attempt callable bias.

These tests cover the new pieces. The *disabled* (default) path being byte-identical
is already proven by the full existing suite (every other test runs recovery off).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
    AutonomousGraspService,
    EffectiveGraspingConfig,
    EffectiveRecoveryOrchestratorConfig,
    GraspBehaviorProfile,
    GraspMode,
    _profile_for,
    _RecoveryReportAdapter,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.recovery.policy import SceneRecoveryAction, SceneRecoveryPolicy
from src.robot.grasping.recovery.orchestrator import (
    RecoveryDispatcher,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
    recovery_actions_from_trail,
    run_recovery_loop,
)


def _eff_cfg(**overrides: object) -> EffectiveGraspingConfig:
    base: dict[str, object] = dict(
        default_mode=GraspMode.DENSE_CLUTTER,
        max_attempts=1,
        closed_loop_enabled=False,
        verification_enabled=False,
        dense_recovery_enabled=False,
        dense_recovery_allowed_actions=(),
    )
    # Group flat ``recovery_orchestrator_*`` overrides into the nested sub-config
    # (the field split that replaced the old flat API). Call sites keep using the
    # flat kwarg names for ergonomics.
    ro_prefix = "recovery_orchestrator_"
    ro_kwargs = {
        key[len(ro_prefix):]: overrides.pop(key)
        for key in list(overrides)
        if key.startswith(ro_prefix)
    }
    if ro_kwargs:
        base["recovery_orchestrator"] = EffectiveRecoveryOrchestratorConfig(**ro_kwargs)  # type: ignore[arg-type]
    base.update(overrides)
    return EffectiveGraspingConfig(**base)  # type: ignore[arg-type]


def _profile_dense() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.DENSE_CLUTTER,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        recovery_allowed_actions=("rescan", "next_viewpoint", "next_target"),
    )


def _orch() -> RecoveryOrchestrator:
    return RecoveryOrchestrator(
        dispatcher=RecoveryDispatcher(), strategies={}, bypass_strategies=True
    )


class RecoveryPolicyBuilderTests(unittest.TestCase):
    def test_disabled_config_yields_disabled_policy(self) -> None:
        fake = SimpleNamespace(effective_config=_eff_cfg())  # default: disabled
        policy = AutonomousGraspService._build_recovery_orchestrator_policy(fake)
        self.assertFalse(policy.enabled)

    def test_none_config_yields_disabled_policy(self) -> None:
        fake = SimpleNamespace(effective_config=None)
        policy = AutonomousGraspService._build_recovery_orchestrator_policy(fake)
        self.assertFalse(policy.enabled)

    def test_enabled_config_maps_actions_modes_budget(self) -> None:
        cfg = _eff_cfg(
            recovery_orchestrator_enabled=True,
            recovery_orchestrator_allowed_actions=("rescan", "next_target"),
            recovery_orchestrator_apply_modes=("dense_clutter",),
            recovery_orchestrator_max_actions=3,
            recovery_orchestrator_per_action_budget=(("rescan", 2),),
        )
        policy = AutonomousGraspService._build_recovery_orchestrator_policy(
            SimpleNamespace(effective_config=cfg)
        )
        self.assertTrue(policy.enabled)
        self.assertEqual(
            policy.allowed_actions,
            (SceneRecoveryAction.RESCAN, SceneRecoveryAction.NEXT_TARGET),
        )
        self.assertEqual(policy.max_recovery_actions, 3)
        self.assertEqual(policy.apply_modes, ("dense_clutter",))
        self.assertEqual(policy.per_action_budget[SceneRecoveryAction.RESCAN], 2)


class RecoveryReportAdapterTests(unittest.TestCase):
    def test_derives_failure_reasons_from_last_attempt(self) -> None:
        last = SimpleNamespace(reasons=(GraspFailureReason.RESCAN_RECOMMENDED,))
        report = SimpleNamespace(
            outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
            pick_report=SimpleNamespace(attempts=(SimpleNamespace(reasons=()), last)),
        )
        adapter = _RecoveryReportAdapter(report)  # type: ignore[arg-type]
        self.assertIs(adapter.outcome, AutonomousGraspOutcome.EXECUTION_FAILED)
        self.assertEqual(
            adapter.failure_reasons, (GraspFailureReason.RESCAN_RECOMMENDED,)
        )

    def test_no_pick_report_yields_no_reasons(self) -> None:
        report = SimpleNamespace(
            outcome=AutonomousGraspOutcome.SUCCEEDED, pick_report=None
        )
        adapter = _RecoveryReportAdapter(report)  # type: ignore[arg-type]
        self.assertEqual(adapter.failure_reasons, ())


class CallableAggressiveBiasTests(unittest.TestCase):
    """The threshold wire: a per-attempt callable bias reads the most recent
    attempt's fused uncertainty and drives the recovery aggressiveness."""

    def _run(self, *, fused: float, threshold: float):
        seen_bias: list[float] = []

        def bias(report: object) -> bool:
            u = float(getattr(report, "telemetry", {}).get("uncertainty_fused", 0.0))
            seen_bias.append(u)
            return u > threshold

        seq = iter(
            [
                SimpleNamespace(
                    outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
                    failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
                    telemetry={"uncertainty_fused": fused},
                )
            ]
        )

        def pick():
            try:
                return next(seq)
            except StopIteration:
                return SimpleNamespace(
                    outcome=AutonomousGraspOutcome.SUCCEEDED,
                    failure_reasons=(),
                    telemetry={},
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
                apply_modes=("dense_clutter",),
                max_recovery_actions=3,
            ),
            orchestrator=_orch(),
            frame_acquirer=lambda: None,
            aggressive_recovery_bias=bias,
        )
        return final, trail, seen_bias

    def test_bias_evaluated_per_attempt_and_recovers(self) -> None:
        final, trail, seen = self._run(fused=0.8, threshold=0.5)
        # The callable bias was evaluated against the failing attempt's telemetry.
        self.assertTrue(seen)
        self.assertEqual(seen[0], 0.8)
        # Recovery drove a retry that succeeded.
        self.assertIs(final.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertEqual(trail.terminal_reason, "recovered_success")

    def test_low_uncertainty_does_not_trip_bias(self) -> None:
        _final, _trail, seen = self._run(fused=0.1, threshold=0.5)
        self.assertTrue(seen)
        self.assertEqual(seen[0], 0.1)  # below threshold -> bias False (no error)


class AttachRecoveryTrailTests(unittest.TestCase):
    def test_trail_summary_folded_into_telemetry(self) -> None:
        report = AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.SUCCEEDED,
            mode=GraspMode.DENSE_CLUTTER,
            profile=_profile_for(GraspMode.DENSE_CLUTTER),
        )
        trail = RecoveryTrail(
            entries=(
                RecoveryTrailEntry(
                    attempt_index=0,
                    failure_reason=GraspFailureReason.RESCAN_RECOMMENDED,
                    plan_action=SceneRecoveryAction.RESCAN,
                    plan_reason="dispatcher:rescan",
                    executed=True,
                    outcome="completed",
                ),
            ),
            terminal_reason="recovered_success",
        )
        out = AutonomousGraspService._attach_recovery_trail(report, trail)
        self.assertEqual(
            out.telemetry["recovery_trail_terminal_reason"], "recovered_success"
        )
        self.assertEqual(out.telemetry["recovery_trail_actions"], ["rescan"])
        # The original frozen report is untouched (replace returns a new one).
        self.assertEqual(dict(report.telemetry), {})

    def test_recovery_actions_field_populated_for_record(self) -> None:
        # P4.2: _attach_recovery_trail also carries the V6-trainable per-step list on the typed report
        # field (the K1 serializer writes it into GraspAttemptRecord.recovery_actions).
        report = AutonomousGraspReport(
            outcome=AutonomousGraspOutcome.SUCCEEDED,
            mode=GraspMode.DENSE_CLUTTER,
            profile=_profile_for(GraspMode.DENSE_CLUTTER),
        )
        trail = RecoveryTrail(
            entries=(
                RecoveryTrailEntry(
                    attempt_index=0,
                    failure_reason=GraspFailureReason.RESCAN_RECOMMENDED,
                    plan_action=SceneRecoveryAction.RESCAN,
                    plan_reason="dispatcher:rescan",
                    executed=True,
                    outcome="completed",
                ),
            ),
            terminal_reason="recovered_success",
        )
        out = AutonomousGraspService._attach_recovery_trail(report, trail)
        self.assertEqual(len(out.recovery_actions), 1)
        self.assertEqual(out.recovery_actions[0]["action"], "rescan")
        self.assertEqual(out.recovery_actions[0]["outcome"], "recovered_success")
        self.assertEqual(report.recovery_actions, ())  # original untouched


class RecoveryActionsFromTrailTests(unittest.TestCase):
    """P4.2: trail -> V6-trainable per-step recovery_actions (last-step recovered_success credit)."""

    @staticmethod
    def _entry(
        action: SceneRecoveryAction, outcome: str, *, executed: bool = True
    ) -> RecoveryTrailEntry:
        return RecoveryTrailEntry(
            attempt_index=0,
            failure_reason=GraspFailureReason.RESCAN_RECOMMENDED,
            plan_action=action,
            plan_reason="t",
            executed=executed,
            outcome=outcome,
        )

    def test_empty_trail_is_empty(self) -> None:
        trail = RecoveryTrail(entries=(), terminal_reason="recovered_success")
        self.assertEqual(recovery_actions_from_trail(trail), ())

    def test_success_credits_only_last_step(self) -> None:
        trail = RecoveryTrail(
            entries=(
                self._entry(SceneRecoveryAction.NEXT_VIEWPOINT, "completed"),
                self._entry(SceneRecoveryAction.RESCAN, "completed"),
            ),
            terminal_reason="recovered_success",
        )
        rows = recovery_actions_from_trail(trail)
        self.assertEqual([r["action"] for r in rows], ["next_viewpoint", "rescan"])
        self.assertEqual(rows[0]["outcome"], "completed")          # intermediate: no credit
        self.assertEqual(rows[1]["outcome"], "recovered_success")  # last: the reward signal
        self.assertEqual(rows[1]["step_result"], "completed")      # raw result preserved
        self.assertTrue(rows[1]["executed"])

    def test_failed_recovery_keeps_raw_outcomes(self) -> None:
        trail = RecoveryTrail(
            entries=(self._entry(SceneRecoveryAction.RESCAN, "completed", executed=False),),
            terminal_reason="escalated_no_recovery",
        )
        rows = recovery_actions_from_trail(trail)
        self.assertEqual(rows[0]["outcome"], "completed")  # NO recovered_success credit on failure
        self.assertFalse(rows[0]["executed"])

    def test_new_sim_action_logged_verbatim(self) -> None:
        # P4.2/B: a new concrete sim action logs its token (V6's token map identity-maps it now).
        trail = RecoveryTrail(
            entries=(self._entry(SceneRecoveryAction.NUDGE_TARGET, "completed"),),
            terminal_reason="recovered_success",
        )
        rows = recovery_actions_from_trail(trail)
        self.assertEqual(rows[0]["action"], "nudge_target")


if __name__ == "__main__":
    unittest.main()
