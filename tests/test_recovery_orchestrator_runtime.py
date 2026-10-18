"""The recovery loop's pieces on the service: the policy built from the snapshot, the report the loop reads, and the
trail folded into the report and the record.

The per-attempt bias the fused uncertainty drove its ordering with left on 2026-10-16 with the uncertainty layer, and
the recovery block with it: a cell built from config runs the owner's fixed recovery (the owner's cleanup).
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
    GraspMode,
    _profile_for,
    _RecoveryReportAdapter,
)
from src.robot.execution.autonomous_grasp.service import _recovery_rows
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.recovery.policy import SceneRecoveryAction
from src.robot.grasping.recovery.orchestrator import (
    RecoveryTrail,
    RecoveryTrailEntry,
)


def _eff_cfg(**overrides: object) -> EffectiveGraspingConfig:
    base: dict[str, object] = dict(
        default_mode=GraspMode.DENSE_CLUTTER,
        max_attempts=1,
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
            recovery_orchestrator_allowed_actions=("rescan", "nudge_target"),
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
            (SceneRecoveryAction.RESCAN, SceneRecoveryAction.NUDGE_TARGET),
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


class TheTrailAsTheRecordCarriesItTests(unittest.TestCase):
    """The trail as the record's ``recovery_actions``: one row a step, the last one credited ``recovered_success`` where
    the recovery ended in a success, each step's own result kept in ``step_result``."""

    @staticmethod
    def _entry(reason: GraspFailureReason, *, executed: bool = True) -> RecoveryTrailEntry:
        return RecoveryTrailEntry(attempt_index=0, failure_reason=reason, plan_action=SceneRecoveryAction.RESCAN,
                                  plan_reason="dispatcher:rescan", executed=executed, outcome="completed")

    def test_an_empty_trail_is_no_row(self) -> None:
        self.assertEqual((), _recovery_rows(RecoveryTrail(entries=(), terminal_reason="recovered_success"), ()))

    def test_a_success_credits_only_the_last_step(self) -> None:
        trail = RecoveryTrail(entries=(self._entry(GraspFailureReason.RESCAN_RECOMMENDED),
                                       self._entry(GraspFailureReason.NO_VALID_GRASP)),
                              terminal_reason="recovered_success")
        first, last = _recovery_rows(trail, ())
        self.assertEqual(("rescan", "completed", "rescan_recommended"),
                         (first["action"], first["outcome"], first["failure_reason"]))
        self.assertEqual(("recovered_success", "completed", True),
                         (last["outcome"], last["step_result"], last["executed"]))

    def test_a_recovery_that_failed_keeps_each_steps_own_result(self) -> None:
        trail = RecoveryTrail(entries=(self._entry(GraspFailureReason.NO_VALID_GRASP, executed=False),),
                              terminal_reason="escalated_no_recovery")
        (row,) = _recovery_rows(trail, ())
        self.assertEqual(("completed", False), (row["outcome"], row["executed"]))


if __name__ == "__main__":
    unittest.main()
