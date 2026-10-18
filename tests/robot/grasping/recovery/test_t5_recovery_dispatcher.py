"""The failure-class dispatcher: what each failure class asks the recovery loop for.

Every class a look again can help asks for ``RESCAN``, which moves nothing itself: the next pick re-perceives the
scene. An escalation asks for nothing, and the operator hears of it. ``NEXT_VIEWPOINT`` left every row on 2026-09-29,
merged into ``RESCAN``; ``NEXT_TARGET`` and ``CONTAINER_AGITATE`` left them on 2026-10-16 with their code, and the
push with them, because it runs inside the pick attempt and never from this loop (the owner's cleanup). The gates, the
budgets and the anti-loop rule are the orchestrator's (``tests/robot/grasping/recovery/test_t5_recovery_orchestrator.py``).
"""

from __future__ import annotations

import unittest

from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
from src.robot.grasping.recovery.policy import SceneRecoveryAction
from src.robot.grasping.types.feedback import GraspFailureReason as R

#: Every class a look again can help: reach and table conflicts, perception complaints, occlusion, no grasp, the
#: refinement classes an older record may carry, and a planner's refusal, which nothing that moves may answer
#: (``tests/robot/grasping/recovery/test_motion_plan_refusal_recovery.py``).
RESCANS = (
    R.IK_FAILED, R.ALL_OUT_OF_WORKSPACE, R.ALL_TABLE_CONFLICT,
    R.RESCAN_RECOMMENDED, R.LOW_DEPTH_CONFIDENCE, R.LOW_MASK_CONFIDENCE, R.EMPTY_MASK, R.MASK_TOO_SMALL,
    R.NO_VALID_DEPTH, R.ACTIVE_PERCEPTION_RECOMMENDED, R.HEAVY_OCCLUSION,
    R.NO_CANDIDATES_GENERATED, R.ALL_COLLIDED, R.NO_VALID_GRASP,
    R.TARGET_LOST_DURING_REFINE, R.REFINEMENT_DIVERGED, R.MOTION_PLAN_REFUSED,
)
#: The escalations: a stopped controller, a label nobody saw, the rejections, and the semantic hint that the next
#: candidate is the pick loop's own business.
ESCALATIONS = (
    R.CONTROLLER_NOT_OPERATIONAL, R.TARGET_LABEL_NOT_FOUND, R.TOPOLOGY_RISK_REJECTED, R.SEMANTIC_REJECTED,
    R.TRY_NEXT_CANDIDATE, R.DEFORMABLE_ROUTING_REQUIRED,
)


class TheDispatcherRowsTests(unittest.TestCase):
    def test_every_class_a_look_again_can_help_rescans(self) -> None:
        for reason in RESCANS:
            with self.subTest(reason=reason.value):
                self.assertEqual((SceneRecoveryAction.RESCAN,), RecoveryDispatcher().actions_for(reason))

    def test_an_escalation_asks_for_nothing(self) -> None:
        for reason in ESCALATIONS:
            with self.subTest(reason=reason.value):
                self.assertEqual((), RecoveryDispatcher().actions_for(reason))

    def test_every_failure_class_has_its_row(self) -> None:
        self.assertEqual(set(R), set(RESCANS) | set(ESCALATIONS))
        self.assertFalse(set(RESCANS) & set(ESCALATIONS))

    def test_the_dispatcher_is_frozen(self) -> None:
        with self.assertRaises((AttributeError, TypeError)):
            RecoveryDispatcher().rows = {}  # type: ignore[attr-defined]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
