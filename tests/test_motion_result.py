"""Contract tests for the typed motion-result primitives (Phase N).

Covers every closed-set :class:`MotionStatus` value plus the
constructor helpers (``executed`` / ``failed`` / ``from_bool``) and the
truthiness predicates (``ok`` / ``__bool__``).
"""

from __future__ import annotations

import unittest

from src.geometry import Frame, Pose
from src.robot.core import JointPositions
from src.robot.core.motion_result import (
    MotionCommand,
    MotionResult,
    MotionStatus,
)


def _pose() -> Pose:
    return Pose.identity(frame=Frame.BASE)


def _joints() -> JointPositions:
    return JointPositions((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


class MotionStatusEnumTests(unittest.TestCase):
    def test_closed_set_of_fifteen_values(self) -> None:
        # The closed set is locked at exactly fifteen categories after
        # the Phase R safety hardening: the original Phase N ten plus
        # the five Phase R safety-rejection categories. Adding or
        # renaming a value is a contract change that must propagate
        # through driver classification, the SafetyReason -> MotionStatus
        # mapping in ``backend/src/robot/safety/decision.py``, and
        # downstream reporting at the same time.
        expected = {
            # Phase N closed set
            "executed",
            "workspace_rejected",
            "ik_failed",
            "controller_rejected",
            "timeout",
            "connection_error",
            "unsupported",
            "invalid_target",
            "cancelled",
            "unknown",
            # Phase R safety-rejection extensions
            "joint_limit_rejected",
            "ik_quality_rejected",
            "self_collision_rejected",
            "payload_rejected",
            "continuity_rejected",
        }
        self.assertEqual({s.value for s in MotionStatus}, expected)

    def test_status_values_are_stable_strings(self) -> None:
        # MotionStatus is declared as ``str, Enum`` so the value is safe
        # to serialise straight into structured event payloads.
        for status in MotionStatus:
            self.assertIsInstance(status.value, str)
            self.assertEqual(MotionStatus(status.value), status)


class MotionCommandEnumTests(unittest.TestCase):
    def test_command_values(self) -> None:
        expected = {"move_to", "move_home", "move_joints", "other"}
        self.assertEqual({c.value for c in MotionCommand}, expected)


class MotionResultExecutedTests(unittest.TestCase):
    def test_executed_minimal(self) -> None:
        result = MotionResult.executed(MotionCommand.MOVE_HOME)

        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertIs(result.command, MotionCommand.MOVE_HOME)
        self.assertTrue(result.ok)
        self.assertTrue(bool(result))
        self.assertIsNone(result.target_pose)
        self.assertIsNone(result.target_joints)
        self.assertEqual(result.message, "")
        self.assertIsNone(result.exception)

    def test_executed_with_pose_and_joints(self) -> None:
        pose = _pose()
        joints = _joints()
        result = MotionResult.executed(
            MotionCommand.MOVE_TO,
            target_pose=pose,
            target_joints=joints,
            message="ok",
        )

        self.assertTrue(result.ok)
        self.assertIs(result.target_pose, pose)
        self.assertIs(result.target_joints, joints)
        self.assertEqual(result.message, "ok")

    def test_frozen_dataclass_rejects_mutation(self) -> None:
        result = MotionResult.executed(MotionCommand.MOVE_TO)
        with self.assertRaises(Exception):  # noqa: BLE001 - dataclasses.FrozenInstanceError
            result.status = MotionStatus.UNKNOWN  # type: ignore[misc]


class MotionResultFailureTests(unittest.TestCase):
    """One test per failure status — keeps the closed set honest."""

    def _assert_failure(
        self,
        status: MotionStatus,
        *,
        message: str = "",
        exception: BaseException | None = None,
    ) -> MotionResult:
        result = MotionResult.failed(
            status,
            MotionCommand.MOVE_TO,
            target_pose=_pose(),
            message=message,
            exception=exception,
        )
        self.assertIs(result.status, status)
        self.assertIs(result.command, MotionCommand.MOVE_TO)
        self.assertFalse(result.ok)
        self.assertFalse(bool(result))
        return result

    def test_workspace_rejected(self) -> None:
        self._assert_failure(
            MotionStatus.WORKSPACE_REJECTED, message="outside x_max",
        )

    def test_ik_failed(self) -> None:
        self._assert_failure(MotionStatus.IK_FAILED)

    def test_controller_rejected(self) -> None:
        self._assert_failure(MotionStatus.CONTROLLER_REJECTED)

    def test_timeout(self) -> None:
        self._assert_failure(MotionStatus.TIMEOUT)

    def test_connection_error_carries_exception(self) -> None:
        exc = ConnectionError("rtde link down")
        result = self._assert_failure(
            MotionStatus.CONNECTION_ERROR,
            message="rtde link down",
            exception=exc,
        )
        # ``exception`` is excluded from compare/repr but must still be
        # readable for diagnostics.
        self.assertIs(result.exception, exc)

    def test_unsupported(self) -> None:
        self._assert_failure(MotionStatus.UNSUPPORTED)

    def test_invalid_target(self) -> None:
        self._assert_failure(
            MotionStatus.INVALID_TARGET, message="pose not in BASE frame",
        )

    def test_cancelled(self) -> None:
        self._assert_failure(MotionStatus.CANCELLED)

    def test_unknown(self) -> None:
        self._assert_failure(MotionStatus.UNKNOWN)

    def test_failed_rejects_executed_status(self) -> None:
        with self.assertRaises(ValueError):
            MotionResult.failed(MotionStatus.EXECUTED, MotionCommand.MOVE_TO)


class MotionResultFromBoolTests(unittest.TestCase):
    def test_from_bool_true_maps_to_executed(self) -> None:
        pose = _pose()
        result = MotionResult.from_bool(
            True, MotionCommand.MOVE_TO, target_pose=pose,
        )
        self.assertTrue(result.ok)
        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertIs(result.target_pose, pose)

    def test_from_bool_false_defaults_to_controller_rejected(self) -> None:
        result = MotionResult.from_bool(False, MotionCommand.MOVE_TO)
        self.assertFalse(result.ok)
        # Default for legacy bool drivers without finer classification.
        self.assertIs(result.status, MotionStatus.CONTROLLER_REJECTED)

    def test_from_bool_false_respects_explicit_failure_status(self) -> None:
        result = MotionResult.from_bool(
            False,
            MotionCommand.MOVE_TO,
            failure_status=MotionStatus.WORKSPACE_REJECTED,
            message="outside box",
        )
        self.assertIs(result.status, MotionStatus.WORKSPACE_REJECTED)
        self.assertEqual(result.message, "outside box")


class MotionResultEqualityTests(unittest.TestCase):
    def test_equality_ignores_exception_field(self) -> None:
        # ``exception`` is declared ``compare=False`` so two otherwise
        # identical results compare equal even when they carry different
        # exception payloads. This keeps diagnostics from polluting
        # caller-side equality assertions in tests.
        a = MotionResult.failed(
            MotionStatus.CONNECTION_ERROR,
            MotionCommand.MOVE_TO,
            message="link down",
            exception=ConnectionError("a"),
        )
        b = MotionResult.failed(
            MotionStatus.CONNECTION_ERROR,
            MotionCommand.MOVE_TO,
            message="link down",
            exception=ConnectionError("b"),
        )
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
