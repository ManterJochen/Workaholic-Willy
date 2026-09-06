"""Phase T5 — RED tests for :class:`RecoveryOrchestrator`.

The orchestrator is a pure decision driver. Given a profile, the
policy, the dispatcher, a last-failure reason set, and the
accumulated typed history, it returns the *next* :class:`SceneRecoveryPlan`
to execute, or ``None`` when no further recovery is permissible.

The orchestrator never executes motion or touches perception; the
service is responsible for plumbing.
"""

from __future__ import annotations

import unittest
from typing import Optional

from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    GraspBehaviorProfile,
    GraspMode,
)
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.recovery.policy import (
    FixtureEnvelope,
    NextTargetRecoveryStrategy,
    NoRecoveryStrategy,
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
)
from src.robot.grasping.recovery.orchestrator import (
    RecoveryDispatcher,
    RecoveryHistoryEntry,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
)


def _profile(
    mode: GraspMode = GraspMode.DENSE_CLUTTER,
    allowed: tuple[str, ...] = ("rescan", "next_viewpoint", "next_target", "nudge_target"),
) -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=mode,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        recovery_allowed_actions=allowed,
    )


def _easy_profile() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        recovery_allowed_actions=(),
    )


def _policy(
    *,
    enabled: bool = True,
    allowed: tuple[SceneRecoveryAction, ...] = (
        SceneRecoveryAction.RESCAN,
        SceneRecoveryAction.NEXT_VIEWPOINT,
        SceneRecoveryAction.NEXT_TARGET,
    ),
    max_total: int = 3,
    per_action: Optional[dict] = None,
    apply_modes: tuple[str, ...] = ("auto", "dense_clutter", "dense_autonomous"),
    fixture: Optional[FixtureEnvelope] = None,
) -> SceneRecoveryPolicy:
    return SceneRecoveryPolicy(
        enabled=enabled,
        allowed_actions=allowed,
        max_recovery_actions=max_total,
        per_action_budget=per_action or {},
        apply_modes=apply_modes,
        fixture=fixture,
    )


def _ctx(
    *,
    profile: GraspBehaviorProfile,
    policy: SceneRecoveryPolicy,
    failure_reasons: tuple[GraspFailureReason, ...] = (),
    history: tuple[SceneRecoveryAction, ...] = (),
) -> SceneRecoveryContext:
    return SceneRecoveryContext(
        profile=profile,
        policy=policy,
        last_outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
        failure_reasons=failure_reasons,
        history=history,
    )


class _StubStrategy:
    """Always emits a plan for its bound action when permitted."""

    __slots__ = ("_action",)

    def __init__(self, action: SceneRecoveryAction) -> None:
        self._action = action

    def plan(self, context: SceneRecoveryContext) -> SceneRecoveryPlan:
        return SceneRecoveryPlan(
            action=self._action,
            reason=f"stub:{self._action.value}",
        )


def _make_orchestrator(
    *,
    dispatcher: Optional[RecoveryDispatcher] = None,
) -> RecoveryOrchestrator:
    return RecoveryOrchestrator(
        dispatcher=dispatcher or RecoveryDispatcher(),
        strategies={
            SceneRecoveryAction.RESCAN: _StubStrategy(SceneRecoveryAction.RESCAN),
            SceneRecoveryAction.NEXT_VIEWPOINT: _StubStrategy(
                SceneRecoveryAction.NEXT_VIEWPOINT
            ),
            SceneRecoveryAction.NEXT_TARGET: _StubStrategy(
                SceneRecoveryAction.NEXT_TARGET
            ),
            SceneRecoveryAction.NUDGE_TARGET: _StubStrategy(
                SceneRecoveryAction.NUDGE_TARGET
            ),
        },
    )


class RecoveryOrchestratorGatingTests(unittest.TestCase):
    def test_disabled_policy_returns_none(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(enabled=False),
            failure_reasons=(GraspFailureReason.IK_FAILED,),
        )
        self.assertIsNone(orch.next_step(ctx))

    def test_easy_mode_excluded_by_apply_modes(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_easy_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        self.assertIsNone(orch.next_step(ctx))

    def test_mode_not_in_apply_modes_returns_none(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(mode=GraspMode.CLOSED_LOOP),
            policy=_policy(apply_modes=("dense_clutter",)),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        self.assertIsNone(orch.next_step(ctx))


class RecoveryOrchestratorDispatchTests(unittest.TestCase):
    def test_dispatch_picks_first_permitted_action(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        plan = orch.next_step(ctx)
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan.action, SceneRecoveryAction.RESCAN)

    def test_dispatch_skips_disallowed_and_falls_through(self) -> None:
        # Policy excludes RESCAN; mapping is (RESCAN, NEXT_VIEWPOINT).
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(
                allowed=(SceneRecoveryAction.NEXT_VIEWPOINT,),
            ),
            failure_reasons=(GraspFailureReason.HEAVY_OCCLUSION,),
        )
        plan = orch.next_step(ctx)
        self.assertIsNotNone(plan)
        assert plan is not None
        # HEAVY_OCCLUSION maps to NEXT_VIEWPOINT, RESCAN — NEXT_VIEWPOINT is
        # listed first by the default dispatcher anyway, so this confirms
        # ordering is respected. Add a second case where the *first* listed
        # action is disallowed:
        ctx2 = _ctx(
            profile=_profile(),
            policy=_policy(
                allowed=(SceneRecoveryAction.RESCAN,),
            ),
            failure_reasons=(GraspFailureReason.HEAVY_OCCLUSION,),
        )
        plan2 = orch.next_step(ctx2)
        self.assertIsNotNone(plan2)
        assert plan2 is not None
        self.assertEqual(plan2.action, SceneRecoveryAction.RESCAN)

    def test_escalation_class_returns_none(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.SEMANTIC_REJECTED,),
        )
        self.assertIsNone(orch.next_step(ctx))


class RecoveryOrchestratorBudgetTests(unittest.TestCase):
    def test_total_budget_exhausted_returns_none(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(max_total=2),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
            history=(
                SceneRecoveryAction.RESCAN,
                SceneRecoveryAction.NEXT_VIEWPOINT,
            ),
        )
        self.assertIsNone(orch.next_step(ctx))

    def test_per_action_budget_blocks_repeat(self) -> None:
        orch = _make_orchestrator()
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(
                max_total=5,
                per_action={SceneRecoveryAction.RESCAN: 1},
            ),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
            history=(SceneRecoveryAction.RESCAN,),
        )
        plan = orch.next_step(ctx)
        self.assertIsNotNone(plan)
        assert plan is not None
        # RESCAN was used; budget=1 forces fall-through to NEXT_VIEWPOINT.
        self.assertEqual(plan.action, SceneRecoveryAction.NEXT_VIEWPOINT)


class RecoveryOrchestratorAntiLoopTests(unittest.TestCase):
    def test_same_action_same_failure_class_blocks(self) -> None:
        """Anti-loop: refuse the same (action, failure_class) pair twice."""

        orch = _make_orchestrator()
        # First step: RESCAN_RECOMMENDED -> RESCAN.
        ctx1 = _ctx(
            profile=_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        )
        plan1 = orch.next_step(ctx1)
        assert plan1 is not None
        self.assertEqual(plan1.action, SceneRecoveryAction.RESCAN)

        # Same class again with RESCAN already in typed history.
        history_entry = RecoveryHistoryEntry(
            action=SceneRecoveryAction.RESCAN,
            failure_class=GraspFailureReason.RESCAN_RECOMMENDED,
            outcome="completed",
        )
        ctx2 = orch.context_with_history_entry(ctx1, history_entry)
        # Second call should skip RESCAN and pick NEXT_VIEWPOINT.
        plan2 = orch.next_step(ctx2)
        assert plan2 is not None
        self.assertEqual(plan2.action, SceneRecoveryAction.NEXT_VIEWPOINT)

    def test_same_action_different_failure_class_allowed(self) -> None:
        """Anti-loop is per (action, class), not per action alone."""

        orch = _make_orchestrator()
        history_entry = RecoveryHistoryEntry(
            action=SceneRecoveryAction.RESCAN,
            failure_class=GraspFailureReason.RESCAN_RECOMMENDED,
            outcome="completed",
        )
        # New failure class with same fallback to RESCAN is allowed.
        ctx_seed = _ctx(
            profile=_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.TARGET_LOST_DURING_REFINE,),
        )
        ctx = RecoveryOrchestrator.attach_history_entry(ctx_seed, history_entry)
        plan = orch.next_step(ctx)
        assert plan is not None
        self.assertEqual(plan.action, SceneRecoveryAction.RESCAN)


class RecoveryTrailTests(unittest.TestCase):
    def test_trail_terminal_reasons(self) -> None:
        empty = RecoveryTrail(entries=(), terminal_reason="recovery_disabled")
        self.assertEqual(empty.terminal_reason, "recovery_disabled")
        self.assertEqual(empty.entries, ())

    def test_trail_to_dict_is_jsonable(self) -> None:
        entry = RecoveryTrailEntry(
            attempt_index=0,
            failure_reason=GraspFailureReason.IK_FAILED,
            plan_action=SceneRecoveryAction.NEXT_TARGET,
            plan_reason="dispatcher",
            executed=True,
            outcome="completed",
        )
        trail = RecoveryTrail(
            entries=(entry,), terminal_reason="recovered_success"
        )
        d = trail.to_dict()
        self.assertEqual(d["terminal_reason"], "recovered_success")
        self.assertEqual(len(d["entries"]), 1)
        e0 = d["entries"][0]
        self.assertEqual(e0["attempt_index"], 0)
        self.assertEqual(e0["failure_reason"], "ik_failed")
        self.assertEqual(e0["plan_action"], "next_target")
        self.assertEqual(e0["plan_reason"], "dispatcher")
        self.assertTrue(e0["executed"])
        self.assertEqual(e0["outcome"], "completed")

    def test_trail_terminal_reason_validation(self) -> None:
        valid = {
            "recovered_success",
            "exhausted_budget",
            "anti_loop_blocked",
            "escalated_no_recovery",
            "recovery_disabled",
        }
        for reason in valid:
            RecoveryTrail(entries=(), terminal_reason=reason)
        with self.assertRaises(ValueError):
            RecoveryTrail(entries=(), terminal_reason="bogus_terminal")


class RecoveryOrchestratorIntegrationWithStrategiesTests(unittest.TestCase):
    """Confirm orchestrator delegates to per-action strategies."""

    def test_next_target_strategy_emits_none_when_no_alternatives(self) -> None:
        orch = RecoveryOrchestrator(
            dispatcher=RecoveryDispatcher(),
            strategies={
                SceneRecoveryAction.RESCAN: _StubStrategy(SceneRecoveryAction.RESCAN),
                SceneRecoveryAction.NEXT_VIEWPOINT: _StubStrategy(
                    SceneRecoveryAction.NEXT_VIEWPOINT
                ),
                SceneRecoveryAction.NEXT_TARGET: NextTargetRecoveryStrategy(),
                SceneRecoveryAction.NUDGE_TARGET: NoRecoveryStrategy(),
            },
        )
        # NEXT_TARGET strategy refuses without ≥2 segmentations and no
        # frame; orchestrator must fall through to RESCAN.
        ctx = _ctx(
            profile=_profile(),
            policy=_policy(),
            failure_reasons=(GraspFailureReason.IK_FAILED,),
        )
        plan = orch.next_step(ctx)
        assert plan is not None
        self.assertEqual(plan.action, SceneRecoveryAction.RESCAN)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
