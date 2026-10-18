"""The recovery orchestrator: the pure decision driver of the loop that looks again after a failed pick.

Given the mode's profile, the policy, the last failure's classes and the history,
:meth:`RecoveryOrchestrator.next_step` returns the next :class:`SceneRecoveryPlan`, or ``None`` when no further
recovery is permissible. It moves nothing and perceives nothing; the service plumbs it. It plans ``RESCAN`` alone, the
one action this loop runs, where the profile and the policy both allow it, its budget is left and it was not yet tried
for that failure class: ``NEXT_TARGET`` and the strategies that planned an action left on 2026-10-16 (the owner's
cleanup), and the push runs inside the pick attempt.

The loop's end maps to the service's outcome (:func:`terminal_outcome_for_trail`): ``recovery_exhausted`` where an
action ran and none was left, the last pick's own otherwise. The loop around a scripted pick is pinned in
``tests/test_t5_service_retry_loop.py``.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from typing import Any, Optional

from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome, GraspBehaviorProfile, GraspMode
from src.robot.grasping.recovery.orchestrator import (
    OUTCOME_RECOVERY_EXHAUSTED,
    RecoveryDispatcher,
    RecoveryHistoryEntry,
    RecoveryOrchestrator,
    RecoveryTrail,
    RecoveryTrailEntry,
    run_recovery_loop,
    terminal_outcome_for_trail,
)
from src.robot.grasping.recovery.policy import SceneRecoveryAction as A
from src.robot.grasping.recovery.policy import SceneRecoveryContext, SceneRecoveryPlan, SceneRecoveryPolicy
from src.robot.grasping.types.feedback import GraspFailureReason as R
from src.robot.grasping.types.modes import GraspSamplingMode


def _profile(mode: GraspMode = GraspMode.DENSE_CLUTTER,
             allowed: tuple[str, ...] = ("rescan", "nudge_target")) -> GraspBehaviorProfile:
    return GraspBehaviorProfile(mode=mode, sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
                                recovery_allowed_actions=allowed)


def _policy(*, enabled: bool = True, allowed: tuple[A, ...] = (A.RESCAN, A.NUDGE_TARGET), max_actions: int = 3,
            per_action: Optional[dict[A, int]] = None,
            apply_modes: tuple[str, ...] = ("auto", "dense_clutter")) -> SceneRecoveryPolicy:
    return SceneRecoveryPolicy(enabled=enabled, allowed_actions=allowed, max_recovery_actions=max_actions,
                               per_action_budget=per_action or {}, apply_modes=apply_modes)


def _tried(action: A, reason: R) -> RecoveryHistoryEntry:
    return RecoveryHistoryEntry(action=action, failure_class=reason, outcome="completed")


def _next(*, profile: Optional[GraspBehaviorProfile] = None, policy: Optional[SceneRecoveryPolicy] = None,
          reasons: tuple[R, ...] = (R.NO_VALID_GRASP,), history: tuple[A, ...] = (),
          tried: tuple[RecoveryHistoryEntry, ...] = (),
          tried_on_the_context: tuple[RecoveryHistoryEntry, ...] = ()) -> Optional[SceneRecoveryPlan]:
    context = SceneRecoveryContext(profile=profile or _profile(), policy=policy or _policy(),
                                   last_outcome=AutonomousGraspOutcome.NO_VALID_GRASP, failure_reasons=reasons,
                                   history=history, typed_history=tried_on_the_context)
    return RecoveryOrchestrator(dispatcher=RecoveryDispatcher()).next_step(context, typed_history=tried)


def _entry(reason: R = R.IK_FAILED) -> RecoveryTrailEntry:
    return RecoveryTrailEntry(attempt_index=0, failure_reason=reason, plan_action=A.RESCAN,
                              plan_reason="dispatcher:rescan", executed=True, outcome="completed")


class TheGatesTests(unittest.TestCase):
    def test_a_disabled_policy_plans_nothing(self) -> None:
        self.assertIsNone(_next(policy=_policy(enabled=False)))

    def test_easy_plans_nothing(self) -> None:
        easy = GraspBehaviorProfile(mode=GraspMode.EASY, sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
                                    recovery_allowed_actions=())
        self.assertIsNone(_next(profile=easy))
        self.assertIsNone(_next(profile=easy, policy=_policy(apply_modes=("auto", "dense_clutter", "easy"))),
                          "the easy profile allows nothing, whatever the policy says")

    def test_a_mode_outside_the_apply_modes_plans_nothing(self) -> None:
        self.assertIsNone(_next(profile=_profile(mode=GraspMode.AUTO), policy=_policy(apply_modes=("dense_clutter",))))

    def test_the_profile_and_the_policy_must_both_allow_the_rescan(self) -> None:
        self.assertIsNone(_next(profile=_profile(allowed=())))
        self.assertIsNone(_next(policy=_policy(allowed=())))

    def test_a_failure_with_no_class_plans_nothing(self) -> None:
        self.assertIsNone(_next(reasons=()))


class WhatItPlansTests(unittest.TestCase):
    def test_a_failure_a_look_again_can_help_plans_the_rescan(self) -> None:
        for reason in (R.NO_VALID_GRASP, R.ALL_COLLIDED, R.MOTION_PLAN_REFUSED):
            with self.subTest(reason=reason.value):
                plan = _next(reasons=(reason,))
                assert plan is not None
                self.assertEqual((A.RESCAN, "dispatcher:rescan"), (plan.action, plan.reason))

    def test_the_push_is_never_planned_here(self) -> None:
        # The profile and the policy allow the push and nothing else: the loop plans nothing, the push runs in the pick.
        self.assertIsNone(_next(reasons=(R.ALL_COLLIDED,), profile=_profile(allowed=("nudge_target",)),
                                policy=_policy(allowed=(A.NUDGE_TARGET,))))

    def test_an_escalation_plans_nothing_and_a_class_after_it_still_rescans(self) -> None:
        self.assertIsNone(_next(reasons=(R.SEMANTIC_REJECTED,)))
        plan = _next(reasons=(R.SEMANTIC_REJECTED, R.IK_FAILED))
        assert plan is not None
        self.assertIs(A.RESCAN, plan.action)


class TheBudgetsTests(unittest.TestCase):
    def test_the_total_budget_ends_it(self) -> None:
        self.assertIsNone(_next(policy=_policy(max_actions=2), history=(A.RESCAN, A.RESCAN)))
        self.assertIsNotNone(_next(policy=_policy(max_actions=2), history=(A.RESCAN,)))

    def test_the_rescans_own_budget_ends_it(self) -> None:
        self.assertIsNone(_next(policy=_policy(per_action={A.RESCAN: 1}), history=(A.RESCAN,)))
        self.assertIsNotNone(_next(policy=_policy(per_action={A.RESCAN: 2}), history=(A.RESCAN,)))


class TheAntiLoopTests(unittest.TestCase):
    def test_the_rescan_is_not_planned_twice_for_one_failure_class(self) -> None:
        self.assertIsNone(_next(reasons=(R.RESCAN_RECOMMENDED,), tried=(_tried(A.RESCAN, R.RESCAN_RECOMMENDED),)))

    def test_another_failure_class_may_rescan_again(self) -> None:
        plan = _next(reasons=(R.TARGET_LOST_DURING_REFINE,), tried=(_tried(A.RESCAN, R.RESCAN_RECOMMENDED),))
        assert plan is not None
        self.assertIs(A.RESCAN, plan.action)

    def test_what_the_context_carries_counts_as_tried_too(self) -> None:
        self.assertIsNone(_next(reasons=(R.RESCAN_RECOMMENDED,),
                                tried_on_the_context=(_tried(A.RESCAN, R.RESCAN_RECOMMENDED),)))


class TheTrailTests(unittest.TestCase):
    def test_it_takes_only_the_terminal_reasons_it_knows(self) -> None:
        for reason in ("recovered_success", "exhausted_budget", "anti_loop_blocked", "escalated_no_recovery",
                       "recovery_disabled"):
            with self.subTest(reason=reason):
                self.assertEqual(reason, RecoveryTrail(entries=(), terminal_reason=reason).terminal_reason)
        with self.assertRaises(ValueError):
            RecoveryTrail(entries=(), terminal_reason="bogus_terminal")

    def test_it_is_plain_data(self) -> None:
        said = RecoveryTrail(entries=(_entry(),), terminal_reason="recovered_success").to_dict()
        self.assertEqual({"terminal_reason": "recovered_success", "entries": [{
            "attempt_index": 0, "failure_reason": "ik_failed", "plan_action": "rescan",
            "plan_reason": "dispatcher:rescan", "executed": True, "outcome": "completed"}]}, said)
        json.dumps(said)


class TheLoopsEndMapsToAnOutcomeTests(unittest.TestCase):
    def test_the_string_is_the_services_outcome(self) -> None:
        self.assertEqual(OUTCOME_RECOVERY_EXHAUSTED, AutonomousGraspOutcome.RECOVERY_EXHAUSTED.value)

    def test_each_end(self) -> None:
        ran = (_entry(),)
        cases = {
            "success keeps the pick's own": (RecoveryTrail(entries=ran, terminal_reason="recovered_success"), None),
            "nothing ran keeps the pick's own": (RecoveryTrail(entries=(), terminal_reason="recovery_disabled"), None),
            "budget spent after a rescan": (RecoveryTrail(entries=ran, terminal_reason="exhausted_budget"),
                                            OUTCOME_RECOVERY_EXHAUSTED),
            "the rescan tried for the class": (RecoveryTrail(entries=ran, terminal_reason="anti_loop_blocked"),
                                               OUTCOME_RECOVERY_EXHAUSTED),
            "escalated after a rescan keeps the pick's own": (
                RecoveryTrail(entries=ran, terminal_reason="escalated_no_recovery"), None),
        }
        for name, (trail, expected) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(expected, terminal_outcome_for_trail(trail))

    def test_a_loop_allowed_no_action_keeps_the_picks_own_outcome(self) -> None:
        # max_recovery_actions = 0 ends the loop as exhausted_budget with nothing run: no recovery was tried.
        picks: list[int] = []

        def pick() -> Any:
            picks.append(1)
            return SimpleNamespace(outcome=AutonomousGraspOutcome.NO_VALID_GRASP, failure_reasons=(R.ALL_COLLIDED,))

        _, trail = run_recovery_loop(pick=pick, profile=_profile(), policy=_policy(max_actions=0),
                                     orchestrator=RecoveryOrchestrator(dispatcher=RecoveryDispatcher()),
                                     frame_acquirer=lambda: None)
        self.assertEqual(((), "exhausted_budget"), (trail.entries, trail.terminal_reason))
        self.assertIsNone(terminal_outcome_for_trail(trail))
        self.assertEqual([1], picks)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
