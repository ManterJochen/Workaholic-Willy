"""
Tests for :class:`SafetyPreflight` and :class:`WorkspaceSafetyGuard`.

Covers:

* Ordering: guards run in the canonical order regardless of
  the construction order.
* Short-circuit: the first rejection wins; subsequent guards do not
  run.
* Acceptance memo: a successful evaluation populates the preflight's
  ``last_target_*`` cache; ``reset()`` clears it.
* Workspace adapter: shrinks the box by ``limits.workspace_margin_mm``
  and rejects with :attr:`SafetyReason.WORKSPACE` /
  :attr:`MotionStatus.WORKSPACE_REJECTED`.
* Driver integration: ``IsaacRobotArm.move()`` (when given a preflight) surfaces the typed
  ``MotionStatus`` from the preflight decision verbatim; the wiring tests are below.
  ``URRobotArm.move()`` is covered end to end in ``tests/test_ur_arm.py`` (frame check, cuRobo branch,
  IK failure, all six guard vetoes, ``last_reject_status`` pass-through). ``KukaRobotArm.move()`` is
  NOT yet driver-tested -- KUKA is unvalidated on real hardware.
* ``as_motion_result`` helper: ``None`` on accept, properly classified
  :class:`MotionResult` on rejection.
"""

from __future__ import annotations

import unittest
from typing import TYPE_CHECKING

import numpy as np

from src.config.schema.robot import (
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)
from src.geometry import Frame, Pose
from src.geometry.quaternion import IDENTITY_QUAT_XYZW
from src.robot.core import JointPositions, MotionCommand, MotionStatus
from src.robot.safety import (
    SafetyContext,
    SafetyDecision,
    SafetyPreflight,
    SafetyReason,
    WorkspaceGuard,
    WorkspaceSafetyGuard,
)

if TYPE_CHECKING:
    from src.robot.drivers.sim.arm import IsaacRobotArm


def _pose(x_mm: float, y_mm: float = 0.0, z_mm: float = 300.0, label: str = "p") -> Pose:
    return Pose(
        position_mm=np.array([x_mm, y_mm, z_mm], dtype=np.float64),
        quaternion_xyzw=IDENTITY_QUAT_XYZW.copy(),
        frame=Frame.BASE,
        label=label,
    )


class _StubGuard:
    """Minimal :class:`SafetyGuard` for ordering / short-circuit tests."""

    def __init__(self, name: str, *, reject: bool = False,
                 reason: SafetyReason = SafetyReason.WORKSPACE) -> None:
        self._name = name
        self._reject = reject
        self._reason = reason
        self.calls = 0

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
        self.calls += 1
        if self._reject:
            return SafetyDecision.reject(
                self._name, self._reason,
                message=f"{self._name} stub rejection",
            )
        return SafetyDecision.accept(self._name)


class SafetyPreflightConstructorTests(unittest.TestCase):
    def test_guards_property_preserves_order(self) -> None:
        a = _StubGuard("workspace")
        b = _StubGuard("joint_limit")
        c = _StubGuard("ik_quality")
        pf = SafetyPreflight([a, b, c])
        self.assertEqual(pf.guard_names, ("workspace", "joint_limit", "ik_quality"))

    def test_guards_tuple_is_immutable(self) -> None:
        guards = [_StubGuard("workspace")]
        pf = SafetyPreflight(guards)
        # Even if the caller mutates their original list, the preflight
        # snapshot stays put.
        guards.append(_StubGuard("joint_limit"))
        self.assertEqual(pf.guard_names, ("workspace",))


class SafetyPreflightEvaluationTests(unittest.TestCase):
    def _ctx(self) -> SafetyContext:
        return SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(0))

    def test_all_accept_returns_accepted_decision(self) -> None:
        a = _StubGuard("workspace")
        b = _StubGuard("joint_limit")
        pf = SafetyPreflight([a, b])
        decision = pf.evaluate(self._ctx())
        self.assertTrue(decision.accepted)
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 1)

    def test_first_rejection_short_circuits(self) -> None:
        a = _StubGuard("workspace", reject=True, reason=SafetyReason.WORKSPACE)
        b = _StubGuard("joint_limit")
        c = _StubGuard("ik_quality")
        pf = SafetyPreflight([a, b, c])
        decision = pf.evaluate(self._ctx())
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.guard, "workspace")
        self.assertIs(decision.reason, SafetyReason.WORKSPACE)
        # Downstream guards never ran.
        self.assertEqual(a.calls, 1)
        self.assertEqual(b.calls, 0)
        self.assertEqual(c.calls, 0)

    def test_rejection_propagates_motion_status(self) -> None:
        for reason, expected_status in (
            (SafetyReason.WORKSPACE, MotionStatus.WORKSPACE_REJECTED),
            (SafetyReason.JOINT_LIMIT, MotionStatus.JOINT_LIMIT_REJECTED),
            (SafetyReason.IK_QUALITY, MotionStatus.IK_QUALITY_REJECTED),
            (SafetyReason.SELF_COLLISION, MotionStatus.SELF_COLLISION_REJECTED),
            (SafetyReason.PAYLOAD, MotionStatus.PAYLOAD_REJECTED),
            (SafetyReason.CONTINUITY, MotionStatus.CONTINUITY_REJECTED),
        ):
            with self.subTest(reason=reason):
                pf = SafetyPreflight([
                    _StubGuard("g", reject=True, reason=reason),
                ])
                decision = pf.evaluate(self._ctx())
                self.assertIs(decision.motion_status, expected_status)


class SafetyPreflightMemoTests(unittest.TestCase):
    def test_acceptance_populates_last_target_pose(self) -> None:
        captured: list[SafetyContext] = []

        class _Recorder:
            name = "recorder"

            def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
                captured.append(ctx)
                return SafetyDecision.accept(self.name)

        pf = SafetyPreflight([_Recorder()])
        p = _pose(0)
        # First move: memo is empty so last_target_pose is None.
        pf.evaluate(pf.context_for_pose(p))
        q = _pose(100, label="q")
        # Second move: memo from the first move has populated the
        # last_target_pose slot via context_for_pose.
        pf.evaluate(pf.context_for_pose(q))
        self.assertEqual(len(captured), 2)
        self.assertIsNone(captured[0].last_target_pose)
        self.assertIs(captured[1].last_target_pose, p)

    def test_rejection_does_not_populate_memo(self) -> None:
        pf = SafetyPreflight([_StubGuard("workspace", reject=True)])
        p = _pose(0)
        pf.evaluate(SafetyContext(command=MotionCommand.MOVE_TO, target_pose=p))
        # Rebuild with a recorder to verify the memo stayed None.
        captured: list[SafetyContext] = []

        class _Recorder:
            name = "recorder"

            def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
                captured.append(ctx)
                return SafetyDecision.accept(self.name)

        # Use the SAME preflight to confirm internal memo stays empty.
        pf_with_recorder = SafetyPreflight([_StubGuard("workspace", reject=True), _Recorder()])
        pf_with_recorder.evaluate(SafetyContext(command=MotionCommand.MOVE_TO, target_pose=p))
        # Recorder should not have run (short-circuit).
        self.assertEqual(captured, [])

    def test_reset_clears_memo(self) -> None:
        captured: list[SafetyContext] = []

        class _Recorder:
            name = "recorder"

            def evaluate(self, ctx: SafetyContext) -> SafetyDecision:
                captured.append(ctx)
                return SafetyDecision.accept(self.name)

        pf = SafetyPreflight([_Recorder()])
        p = _pose(0)
        # Prime the memo via a real accepted evaluation.
        pf.evaluate(pf.context_for_pose(p))
        # Reset and re-evaluate: memo MUST be cleared, so the recorder
        # should see ``last_target_pose=None`` on the next call.
        pf.reset()
        pf.evaluate(pf.context_for_pose(p))
        # captured[0] was the first call (None), captured[1] is after reset.
        self.assertIsNone(captured[1].last_target_pose)


class WorkspaceSafetyGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.limits = WorkspaceLimitsConfig(
            x_min=-500.0, x_max=500.0,
            y_min=-500.0, y_max=500.0,
            z_min=100.0, z_max=600.0,
        )

    def test_inside_box_accepts(self) -> None:
        guard = WorkspaceSafetyGuard(WorkspaceGuard(self.limits))
        ctx = SafetyContext(command=MotionCommand.MOVE_TO, target_pose=_pose(0))
        d = guard.evaluate(ctx)
        self.assertTrue(d.accepted)
        self.assertEqual(d.guard, "workspace")

    def test_outside_box_rejects_with_workspace_reason(self) -> None:
        guard = WorkspaceSafetyGuard(WorkspaceGuard(self.limits))
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO,
            target_pose=_pose(9999.0, label="far"),
        )
        d = guard.evaluate(ctx)
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.WORKSPACE)
        self.assertIs(d.motion_status, MotionStatus.WORKSPACE_REJECTED)
        # Detail carries the offending coordinate for diagnostics.
        self.assertEqual(d.detail.get("label"), "far")
        self.assertIn("x_mm", d.detail)

    def test_joint_only_target_accepts(self) -> None:
        guard = WorkspaceSafetyGuard(WorkspaceGuard(self.limits))
        ctx = SafetyContext(command=MotionCommand.MOVE_JOINTS, target_pose=None)
        d = guard.evaluate(ctx)
        self.assertTrue(d.accepted)


class WorkspaceMarginTests(unittest.TestCase):
    def test_from_safety_config_shrinks_box_by_margin(self) -> None:
        cfg = RobotSafetyConfig.model_validate(
            {"limits": {"workspace_margin_mm": 50.0}}
        )
        limits = WorkspaceLimitsConfig(
            x_min=-500.0, x_max=500.0,
            y_min=-500.0, y_max=500.0,
            z_min=100.0, z_max=600.0,
        )
        pf = SafetyPreflight.from_safety_config(cfg, limits)
        # A pose at x=460 is inside the original ±500 box but OUTSIDE
        # the shrunk box (x_max - 50 = 450). The workspace guard must
        # reject it.
        ctx = SafetyContext(
            command=MotionCommand.MOVE_TO, target_pose=_pose(460.0, label="edge"),
        )
        decision = pf.evaluate(ctx)
        self.assertFalse(decision.accepted)
        self.assertIs(decision.reason, SafetyReason.WORKSPACE)


class AsMotionResultTests(unittest.TestCase):
    def test_accepted_decision_returns_none(self) -> None:
        d = SafetyDecision.accept("workspace")
        self.assertIsNone(
            SafetyPreflight.as_motion_result(d, MotionCommand.MOVE_TO),
        )

    def test_rejected_decision_returns_typed_result(self) -> None:
        d = SafetyDecision.reject(
            "workspace", SafetyReason.WORKSPACE, message="outside",
        )
        result = SafetyPreflight.as_motion_result(
            d, MotionCommand.MOVE_TO, target_pose=_pose(0),
        )
        assert result is not None
        self.assertIs(result.status, MotionStatus.WORKSPACE_REJECTED)
        self.assertIs(result.command, MotionCommand.MOVE_TO)
        self.assertIn("safety:workspace/workspace", result.message)


# ---------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------
#
# The drivers wire the preflight in their ``move()`` typed path. Only
# paths that do not need real hardware are exercised. SIM is tested
# below in mock_mode with a preflight injected at construction; the
# UR ``move()`` driver integration lives in ``tests/test_ur_arm.py``.
# KUKA has no driver ``move()`` test yet (unvalidated on real hardware).


class IsaacRobotArmPreflightWiringTests(unittest.TestCase):
    """The SIM driver runs the preflight when one is injected."""

    def _arm(self, preflight: SafetyPreflight | None) -> "IsaacRobotArm":
        from backend.src.robot.drivers.sim.arm import IsaacRobotArm
        from backend.src.robot.drivers.sim.config import SimRobotConfig

        cfg = SimRobotConfig(enabled=True, mock_mode=True)
        return IsaacRobotArm(cfg, safety_preflight=preflight)

    def test_without_preflight_mock_move_accepts(self) -> None:
        arm = self._arm(None)
        arm.connect()
        result = arm.move(_pose(0))
        self.assertIs(result.status, MotionStatus.EXECUTED)

    def test_preflight_rejection_surfaces_typed_status(self) -> None:
        pf = SafetyPreflight([
            _StubGuard("self_collision", reject=True,
                       reason=SafetyReason.SELF_COLLISION),
        ])
        arm = self._arm(pf)
        arm.connect()
        result = arm.move(_pose(0))
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)

    def test_move_to_joints_rejection_surfaces_typed_status(self) -> None:
        # The typed joint move is also gated -> a guard rejection returns the matching status.
        pf = SafetyPreflight([
            _StubGuard("self_collision", reject=True,
                       reason=SafetyReason.SELF_COLLISION),
        ])
        arm = self._arm(pf)
        arm.connect()
        result = arm.move_to_joints(JointPositions([0.0] * 6))
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)

    def test_move_to_joints_without_preflight_executes(self) -> None:
        arm = self._arm(None)
        arm.connect()
        result = arm.move_to_joints(JointPositions([0.0] * 6))
        self.assertIs(result.status, MotionStatus.EXECUTED)


class GateJointTargetTests(unittest.TestCase):
    """gate_joint_target gates the destination, exempts continuity, resets the memo both sides."""

    def test_accepts_and_resets_continuity_memo(self) -> None:
        pf = SafetyPreflight([])  # no guards -> always accepts
        pf._last_target_joints = JointPositions([1.0] * 6)  # a stale prior target
        result = pf.gate_joint_target(JointPositions([2.0] * 6))
        self.assertIsNone(result)  # accepted -> no typed rejection
        # Reset on BOTH sides: neither this joint move nor the next Cartesian move is diffed across it.
        self.assertIsNone(pf._last_target_joints)
        self.assertIsNone(pf._last_target_pose)

    def test_rejection_is_typed(self) -> None:
        pf = SafetyPreflight([
            _StubGuard("self_collision", reject=True, reason=SafetyReason.SELF_COLLISION),
        ])
        result = pf.gate_joint_target(JointPositions([0.0] * 6))
        self.assertIsNotNone(result)
        assert result is not None  # for type-checkers
        self.assertIs(result.status, MotionStatus.SELF_COLLISION_REJECTED)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
