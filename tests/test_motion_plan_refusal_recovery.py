"""A motion-planner refusal is a recoverable failure, and only ``rescan`` recovers it.

Measured on-box 2026-08-17 (``--scene occlude``, five picks in five): cuRobo refused to route to a
grasp that had already been found, scored and accepted. ``PickAttempt.reasons`` describes why no
grasp was *found*, so it was empty; the recovery driver saw nothing to recover from and terminated
``escalated_no_recovery`` with zero actions. The pick path's own explanation existed the whole time
-- on ``PolicyReport`` -- and was dropped one layer above.

Two invariants live here, and the second is a safety rule rather than a behaviour:

1. The refusal now reaches the recovery driver as :attr:`GraspFailureReason.MOTION_PLAN_REFUSED`.
2. It maps to ``(RESCAN,)`` and nothing else. A refusal is fail-closed; re-perceiving costs nothing,
   while every other action in the table either commands motion or silently redirects the cell to an
   object nobody asked for. Widening that tuple must fail a test, not pass review.

The discrimination is the delicate part: ``MotionStatus.TIMEOUT`` covers BOTH "the planner found no
route, nothing was sent" and "the command was accepted and overran its budget", and on real hardware
the arm may still be moving in the second. Only the first may be recovered from without a stop.
"""

from __future__ import annotations

import unittest

from src.robot.core import NO_PLAN_FAIL_SAFE_MESSAGE, MotionStatus
from src.robot.execution.autonomous_grasp.service import (
    _RecoveryReportAdapter,
    _is_motion_plan_refusal,
)
from src.robot.grasping.loop.pick_loop import (
    PickAttempt,
    PickOutcome,
    PickReport,
)
from src.robot.grasping.recovery.orchestrator import RecoveryDispatcher
from src.robot.grasping.recovery.policy import SceneRecoveryAction
from src.robot.grasping.types.feedback import GraspFailureReason


def _attempt(
    *,
    reasons: tuple = (),
    motion_status: str | None = None,
    motion_message: str | None = None,
) -> PickAttempt:
    return PickAttempt(
        attempt_index=0,
        target_index=0,
        reasons=reasons,
        score=0.9,
        action="execution_failed",
        motion_status=motion_status,
        motion_message=motion_message,
    )


class _Report:
    """The two fields :class:`_RecoveryReportAdapter` reads off an autonomous report."""

    def __init__(self, attempt: PickAttempt) -> None:
        self.outcome = PickOutcome.EXECUTION_FAILED
        self.pick_report = PickReport(
            outcome=PickOutcome.EXECUTION_FAILED, attempts=(attempt,)
        )


class MotionPlanRefusalDiscriminationTests(unittest.TestCase):
    """``TIMEOUT`` means two different things; only one of them is a refusal."""

    def test_planner_refusal_is_recognised(self) -> None:
        self.assertTrue(
            _is_motion_plan_refusal(
                _attempt(
                    motion_status=str(MotionStatus.TIMEOUT),
                    motion_message=NO_PLAN_FAIL_SAFE_MESSAGE,
                )
            )
        )

    def test_execution_timeout_is_NOT_a_refusal(self) -> None:
        # Same status, different event: the command was accepted and overran. The arm may still be
        # moving, so this must never be handed to the recovery driver as "nothing moved".
        self.assertFalse(
            _is_motion_plan_refusal(
                _attempt(
                    motion_status=str(MotionStatus.TIMEOUT),
                    motion_message="move_to did not settle within 5.0 s",
                )
            )
        )

    def test_other_motion_faults_are_not_refusals(self) -> None:
        for status in (
            MotionStatus.CONTROLLER_REJECTED,
            MotionStatus.CONNECTION_ERROR,
            MotionStatus.WORKSPACE_REJECTED,
            MotionStatus.SELF_COLLISION_REJECTED,
        ):
            with self.subTest(status=status):
                self.assertFalse(
                    _is_motion_plan_refusal(
                        _attempt(
                            motion_status=str(status),
                            motion_message=NO_PLAN_FAIL_SAFE_MESSAGE,
                        )
                    )
                )

    def test_attempt_without_the_motion_fields_is_unchanged(self) -> None:
        # Every construction site that predates the fields, and every test double.
        self.assertFalse(_is_motion_plan_refusal(_attempt()))
        self.assertFalse(_is_motion_plan_refusal(object()))


class RecoveryAdapterTests(unittest.TestCase):
    """What the recovery driver is actually handed."""

    def test_refusal_surfaces_a_typed_reason_where_there_was_none(self) -> None:
        adapter = _RecoveryReportAdapter(
            _Report(
                _attempt(
                    motion_status=str(MotionStatus.TIMEOUT),
                    motion_message=NO_PLAN_FAIL_SAFE_MESSAGE,
                )
            )
        )
        self.assertEqual(
            adapter.failure_reasons,
            (GraspFailureReason.MOTION_PLAN_REFUSED,),
        )

    def test_the_measured_before_state_is_pinned(self) -> None:
        # Without the discriminator this is what the driver saw on every one of the five occlude
        # picks: an empty tuple -> escalated_no_recovery -> recovery_actions=0.
        adapter = _RecoveryReportAdapter(
            _Report(
                _attempt(
                    motion_status=str(MotionStatus.TIMEOUT),
                    motion_message="move_to did not settle within 5.0 s",
                )
            )
        )
        self.assertEqual(adapter.failure_reasons, ())

    def test_a_real_reason_is_never_displaced(self) -> None:
        # The synthesis only fires when the attempt has nothing of its own to say.
        adapter = _RecoveryReportAdapter(
            _Report(
                _attempt(
                    reasons=(GraspFailureReason.ALL_COLLIDED,),
                    motion_status=str(MotionStatus.TIMEOUT),
                    motion_message=NO_PLAN_FAIL_SAFE_MESSAGE,
                )
            )
        )
        self.assertEqual(
            adapter.failure_reasons, (GraspFailureReason.ALL_COLLIDED,)
        )


class RefusalRecoversByRescanOnlyTests(unittest.TestCase):
    """The safety rule, pinned so that widening it has to be deliberate."""

    def test_the_mapping_is_rescan_and_nothing_else(self) -> None:
        self.assertEqual(
            RecoveryDispatcher().actions_for(
                GraspFailureReason.MOTION_PLAN_REFUSED
            ),
            (SceneRecoveryAction.RESCAN,),
        )

    def test_no_action_that_moves_or_redirects_is_reachable(self) -> None:
        # Named individually rather than asserted as a set difference: each of these is a distinct
        # way a fail-closed refusal would turn back into a motion or into picking the wrong object.
        actions = RecoveryDispatcher().actions_for(
            GraspFailureReason.MOTION_PLAN_REFUSED
        )
        for forbidden in (
            SceneRecoveryAction.NEXT_VIEWPOINT,
            SceneRecoveryAction.NUDGE_TARGET,
            SceneRecoveryAction.CONTAINER_AGITATE,
            SceneRecoveryAction.NEXT_TARGET,
        ):
            with self.subTest(action=forbidden):
                self.assertNotIn(forbidden, actions)


class DriverMessageContractTests(unittest.TestCase):
    """Both drivers must emit the constant, or the discriminator silently stops discriminating."""

    def test_sim_and_ur_use_the_shared_constant(self) -> None:
        # A source check on purpose: importing either driver pulls vendor SDKs, and the failure
        # this guards against is precisely someone re-typing the string in one of them.
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in ("src/robot/drivers/sim/arm.py",
                    "src/robot/drivers/ur/arm.py"):
            with self.subTest(driver=rel):
                text = (root / rel).read_text(encoding="utf-8")
                self.assertIn("NO_PLAN_FAIL_SAFE_MESSAGE", text)
                self.assertNotIn(f'message="{NO_PLAN_FAIL_SAFE_MESSAGE}"', text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
