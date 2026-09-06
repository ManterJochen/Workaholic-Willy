"""Phase T5 — RED tests for the pure failure-class dispatcher.

The dispatcher maps a :class:`GraspFailureReason` to an *ordered* tuple
of :class:`SceneRecoveryAction` candidates the orchestrator will try
in priority order, subject to policy + profile + budget gates.

These tests pin the locked Q6 default mapping. They do *not* exercise
gating, anti-loop, or service wiring — those live in the orchestrator
and service test files.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.recovery.policy import SceneRecoveryAction
from src.robot.grasping.recovery.orchestrator import (
    RecoveryDispatcher,
)


class RecoveryDispatcherDefaultsTests(unittest.TestCase):
    """The default dispatcher must encode the operator-locked map."""

    def setUp(self) -> None:
        self.dispatcher = RecoveryDispatcher()

    def test_ik_failed_prefers_next_target_then_rescan(self) -> None:
        seq = self.dispatcher.actions_for(GraspFailureReason.IK_FAILED)
        self.assertEqual(
            seq,
            (
                SceneRecoveryAction.NEXT_TARGET,
                SceneRecoveryAction.RESCAN,
            ),
        )

    def test_workspace_and_table_share_ik_failed_priority(self) -> None:
        for reason in (
            GraspFailureReason.ALL_OUT_OF_WORKSPACE,
            GraspFailureReason.ALL_TABLE_CONFLICT,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (
                    SceneRecoveryAction.NEXT_TARGET,
                    SceneRecoveryAction.RESCAN,
                ),
                f"unexpected sequence for {reason}",
            )

    def test_perception_signals_prefer_rescan_then_next_viewpoint(self) -> None:
        for reason in (
            GraspFailureReason.RESCAN_RECOMMENDED,
            GraspFailureReason.LOW_DEPTH_CONFIDENCE,
            GraspFailureReason.LOW_MASK_CONFIDENCE,
            GraspFailureReason.EMPTY_MASK,
            GraspFailureReason.MASK_TOO_SMALL,
            GraspFailureReason.NO_VALID_DEPTH,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (
                    SceneRecoveryAction.RESCAN,
                    SceneRecoveryAction.NEXT_VIEWPOINT,
                ),
                f"unexpected sequence for {reason}",
            )

    def test_occlusion_signals_prefer_next_viewpoint_then_rescan(self) -> None:
        for reason in (
            GraspFailureReason.ACTIVE_PERCEPTION_RECOMMENDED,
            GraspFailureReason.HEAVY_OCCLUSION,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (
                    SceneRecoveryAction.NEXT_VIEWPOINT,
                    SceneRecoveryAction.RESCAN,
                ),
                f"unexpected sequence for {reason}",
            )

    def test_no_candidates_path_includes_nudge_target(self) -> None:
        for reason in (
            GraspFailureReason.NO_CANDIDATES_GENERATED,
            GraspFailureReason.NO_VALID_GRASP,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (
                    SceneRecoveryAction.NEXT_TARGET,
                    SceneRecoveryAction.NUDGE_TARGET,
                    SceneRecoveryAction.NEXT_VIEWPOINT,
                ),
                f"unexpected sequence for {reason}",
            )

    def test_all_collided_path_offers_agitate_for_clutter_jams(self) -> None:
        # G6: a clutter jam (ALL_COLLIDED) additionally offers the envelope-clamped agitate between the
        # nudge and the viewpoint change. Stays double-gated (profile + policy + fixture + amplitude).
        self.assertEqual(
            self.dispatcher.actions_for(GraspFailureReason.ALL_COLLIDED),
            (
                SceneRecoveryAction.NEXT_TARGET,
                SceneRecoveryAction.NUDGE_TARGET,
                SceneRecoveryAction.CONTAINER_AGITATE,
                SceneRecoveryAction.NEXT_VIEWPOINT,
            ),
        )

    def test_refinement_failures_prefer_rescan_then_next_target(self) -> None:
        for reason in (
            GraspFailureReason.TARGET_LOST_DURING_REFINE,
            GraspFailureReason.REFINEMENT_DIVERGED,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (
                    SceneRecoveryAction.RESCAN,
                    SceneRecoveryAction.NEXT_TARGET,
                ),
                f"unexpected sequence for {reason}",
            )

    def test_try_next_candidate_maps_to_next_target_only(self) -> None:
        self.assertEqual(
            self.dispatcher.actions_for(GraspFailureReason.TRY_NEXT_CANDIDATE),
            (SceneRecoveryAction.NEXT_TARGET,),
        )

    def test_escalation_signals_yield_empty_sequence(self) -> None:
        for reason in (
            GraspFailureReason.TOPOLOGY_RISK_REJECTED,
            GraspFailureReason.SEMANTIC_REJECTED,
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
        ):
            self.assertEqual(
                self.dispatcher.actions_for(reason),
                (),
                f"expected empty escalation sequence for {reason}",
            )


class RecoveryDispatcherOverrideTests(unittest.TestCase):
    """The dispatcher accepts operator overrides without changing defaults."""

    def test_override_replaces_specific_reason_only(self) -> None:
        override = {
            GraspFailureReason.IK_FAILED: (SceneRecoveryAction.RESCAN,),
        }
        dispatcher = RecoveryDispatcher(overrides=override)
        self.assertEqual(
            dispatcher.actions_for(GraspFailureReason.IK_FAILED),
            (SceneRecoveryAction.RESCAN,),
        )
        # Non-overridden reason retains default mapping.
        self.assertEqual(
            dispatcher.actions_for(GraspFailureReason.HEAVY_OCCLUSION),
            (
                SceneRecoveryAction.NEXT_VIEWPOINT,
                SceneRecoveryAction.RESCAN,
            ),
        )

    def test_override_to_empty_disables_class(self) -> None:
        dispatcher = RecoveryDispatcher(
            overrides={GraspFailureReason.IK_FAILED: ()}
        )
        self.assertEqual(
            dispatcher.actions_for(GraspFailureReason.IK_FAILED),
            (),
        )

    def test_dispatcher_is_frozen(self) -> None:
        dispatcher = RecoveryDispatcher()
        with self.assertRaises((AttributeError, Exception)):
            dispatcher.overrides = {}  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
