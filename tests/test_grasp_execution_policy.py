"""Tests for :class:`GraspExecutionPolicy` and orchestrator integration."""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import MotionStatus
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
    _grasp_point_to_quaternion,
    _quaternion_from_axes,
    _yaw_axes_to_base_x,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.core.motion_result import MotionCommand, MotionResult


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeArm:
    def __init__(self) -> None:
        self.move_to_calls: list[Pose] = []
        self.fail_after: int | None = None

    def get_tcp_pose(self) -> Pose:
        return Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if self.fail_after is not None and len(self.move_to_calls) >= self.fail_after:
            raise RuntimeError("simulated motion failure")
        self.move_to_calls.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _SteadyArm(_FakeArm):
    """L2.5: a _FakeArm that also exposes ``wait_until_steady`` (records calls; configurable result)."""

    def __init__(self, steady: bool = True) -> None:
        super().__init__()
        self._steady = steady
        self.steady_calls = 0

    def wait_until_steady(self, timeout_s: float = 5.0, poll_interval_s: float = 0.02) -> bool:
        self.steady_calls += 1
        return self._steady


class _BasicGripper:
    """A :class:`Gripper` implementer with no object-detection capability."""

    min_width_mm = 0.0
    max_width_mm = 85.0
    is_connected = True

    def __init__(self) -> None:
        self.width_calls: list[float] = []
        self._width = self.max_width_mm

    def connect(self) -> None:  # pragma: no cover - unused
        pass

    def disconnect(self) -> None:  # pragma: no cover - unused
        pass

    def activate(self) -> None:  # pragma: no cover - unused
        pass

    def set_width_mm(
        self, width_mm: float, *, speed: float | None = None, force: float | None = None
    ) -> None:
        self.width_calls.append(float(width_mm))
        self._width = float(width_mm)

    def get_width_mm(self) -> float:
        return self._width


class _DetectingGripper(_BasicGripper):
    """A gripper that advertises :class:`ObjectDetectingGripper`."""

    def __init__(self, *, detects: bool) -> None:
        super().__init__()
        self._detects = bool(detects)
        self.is_object_detected_calls = 0

    def is_object_detected(self) -> bool:
        self.is_object_detected_calls += 1
        return self._detects


def _grasp(score: float = 0.9) -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=score,
        frame=GraspFrame.BASE,
        label="test",
    )


# ---------------------------------------------------------------------------
# Policy tests
# ---------------------------------------------------------------------------


class PolicyConstructorTests(unittest.TestCase):
    def test_rejects_invalid_standoff(self) -> None:
        with self.assertRaises(ValueError):
            GraspExecutionPolicy(arm=_FakeArm(), standoff_mm=-1.0)

    def test_rejects_invalid_retreat(self) -> None:
        with self.assertRaises(ValueError):
            GraspExecutionPolicy(arm=_FakeArm(), retreat_mm=-1.0)

    def test_rejects_too_few_approach_steps(self) -> None:
        with self.assertRaises(ValueError):
            GraspExecutionPolicy(arm=_FakeArm(), approach_steps=1)

    def test_rejects_invalid_retreat_steps(self) -> None:
        with self.assertRaises(ValueError):
            GraspExecutionPolicy(arm=_FakeArm(), retreat_steps=0)


class RetreatStepsTests(unittest.TestCase):
    """H4.1 (B2): the chunked retreat (``retreat_steps``) — splits the post-close lift into N settled steps so
    the lift pendulum damps between them; default 1 = the single retreat waypoint = byte-identical."""

    def test_default_one_is_single_retreat_waypoint(self) -> None:
        arm = _FakeArm()
        report = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(arm.move_to_calls), 5)  # 4 approach + 1 retreat
        retreat = arm.move_to_calls[-1]
        self.assertEqual(retreat.label, "retreat")
        self.assertAlmostEqual(float(retreat.position_mm[2]), 500.0)  # grasp z 400 + retreat_mm 100

    def test_chunks_the_retreat_into_settled_steps(self) -> None:
        arm = _FakeArm()
        report = GraspExecutionPolicy(
            arm=arm, gripper=None, approach_steps=4, retreat_steps=4,
        ).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(len(arm.move_to_calls), 8)  # 4 approach + 4 retreat
        retreat = arm.move_to_calls[-4:]
        self.assertEqual(
            [p.label for p in retreat], ["retreat_01", "retreat_02", "retreat_03", "retreat_04"]
        )
        # Interpolated vertical lifts from the grasp (z=400) to grasp+retreat_mm (z=500), monotone increasing;
        # the final waypoint equals the retreat_steps=1 endpoint (same total lift).
        zs = [float(p.position_mm[2]) for p in retreat]
        self.assertEqual(zs, sorted(zs))
        self.assertAlmostEqual(zs[0], 425.0)
        self.assertAlmostEqual(zs[-1], 500.0)


class SteadyGateTests(unittest.TestCase):
    """The steady gate (``safety.dwell``) is the arm's own since P0e E9, inside every motion verb of the planned arm.

    The policy held a copy until then, turned on by the pick service from the arm's tree; the planned arm waiting in
    ``move`` made it a second wait before every move. What the policy owes the gate now is only to read the arm's
    TIMEOUT as a failed motion and command nothing after it.
    """

    def test_the_policy_never_asks_the_arm_to_wait(self) -> None:
        arm = _SteadyArm(steady=False)
        report = GraspExecutionPolicy(arm=arm, gripper=None).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertEqual(0, arm.steady_calls)
        self.assertFalse(hasattr(GraspExecutionPolicy(arm=arm, gripper=None), "require_steady_before_motion"))

    def test_a_timeout_the_arm_reports_fails_the_motion_and_nothing_follows(self) -> None:
        arm = _TimesOutArm()
        report = GraspExecutionPolicy(arm=arm, gripper=None).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertIs(report.motion_status, MotionStatus.TIMEOUT)
        self.assertEqual(1, len(arm.move_to_calls))


class _TimesOutArm(_FakeArm):
    """An arm whose own steady gate times out: its move answers TIMEOUT and moves nothing."""

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_to_calls.append(pose)
        return MotionResult.failed(MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                                   message="the arm did not come to rest within 5.00 s before the motion (safety.dwell)")


class PolicyNoGripperTests(unittest.TestCase):
    """Without a gripper the policy must still drive the approach+retreat."""

    def test_executes_waypoints_without_gripper(self) -> None:
        arm = _FakeArm()
        policy = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4)
        report = policy.execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        # 4 approach waypoints + 1 retreat.
        self.assertEqual(len(arm.move_to_calls), 5)
        self.assertEqual(arm.move_to_calls[0].label, "approach_00")
        self.assertEqual(arm.move_to_calls[-1].label, "retreat")
        self.assertIsNone(report.object_detected)

    def test_motion_failure_is_reported(self) -> None:
        arm = _FakeArm()
        arm.fail_after = 2  # raise on the 3rd move
        policy = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4)
        report = policy.execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertIsInstance(report.error, RuntimeError)


class PolicyBasicGripperTests(unittest.TestCase):
    """Gripper without detection capability is trusted post-close."""

    def test_basic_gripper_is_closed_and_trusted(self) -> None:
        arm = _FakeArm()
        gripper = _BasicGripper()
        policy = GraspExecutionPolicy(
            arm=arm, gripper=gripper, pre_open_width_mm=80.0
        )
        report = policy.execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        # Pre-open then close = 2 width commands.
        self.assertEqual(len(gripper.width_calls), 2)
        self.assertAlmostEqual(gripper.width_calls[0], 80.0)
        # Default close width: max(min_width, grip_width - 1) = 39.0
        self.assertAlmostEqual(gripper.width_calls[1], 39.0)
        # No detection capability => report.object_detected is None.
        self.assertIsNone(report.object_detected)

    def test_default_close_width_respects_gripper_min(self) -> None:
        arm = _FakeArm()
        gripper = _BasicGripper()
        gripper.min_width_mm = 50.0  # force the floor
        policy = GraspExecutionPolicy(arm=arm, gripper=gripper)
        policy.execute(_grasp())
        # Floor at min_width_mm (50.0), not grip_width - 1 (39.0).
        self.assertAlmostEqual(gripper.width_calls[-1], 50.0)


class PolicyDetectingGripperTests(unittest.TestCase):
    """Capability-aware verification (L2 decision)."""

    def test_detecting_gripper_success(self) -> None:
        arm = _FakeArm()
        gripper = _DetectingGripper(detects=True)
        policy = GraspExecutionPolicy(arm=arm, gripper=gripper)
        report = policy.execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertTrue(report.object_detected)
        # Asked at the close and again once the part is lifted (the owner, 2026-10-15).
        self.assertEqual(gripper.is_object_detected_calls, 2)
        # Retreat commanded.
        self.assertEqual(arm.move_to_calls[-1].label, "retreat")

    def test_a_part_that_slips_out_on_the_lift_is_no_part_carried(self) -> None:
        """The owner's cell, 2026-10-15: now and then a part held at the close fell out as the arm lifted it, and the cell
        carried it to the place as gripped. Asked again once the lift is done, the jaws closed on nothing end the pick
        as an empty grasp, the arm standing where the lift ended."""

        class _SlipsOnTheLift(_DetectingGripper):
            def is_object_detected(self) -> bool:
                self.is_object_detected_calls += 1
                return self.is_object_detected_calls == 1

        arm = _FakeArm()
        gripper = _SlipsOnTheLift(detects=True)
        report = GraspExecutionPolicy(arm=arm, gripper=gripper).execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.OBJECT_NOT_DETECTED)
        self.assertFalse(report.object_detected)
        self.assertEqual(2, gripper.is_object_detected_calls)
        self.assertEqual("retreat", arm.move_to_calls[-1].label, "the lift ran before it was asked again")
        self.assertIn("slipped out", report.motion_message)

    def test_detecting_gripper_failure_skips_retreat(self) -> None:
        arm = _FakeArm()
        gripper = _DetectingGripper(detects=False)
        policy = GraspExecutionPolicy(arm=arm, gripper=gripper)
        report = policy.execute(_grasp())
        self.assertIs(report.outcome, PolicyOutcome.OBJECT_NOT_DETECTED)
        self.assertFalse(report.object_detected)
        # No 'retreat' waypoint commanded after empty-jaw detection.
        labels = [p.label for p in arm.move_to_calls]
        self.assertNotIn("retreat", labels)


# ---------------------------------------------------------------------------
# Orchestrator integration
# ---------------------------------------------------------------------------


class OrchestratorPolicyIntegrationTests(unittest.TestCase):
    """The orchestrator must build a default policy from its own params."""

    def test_default_policy_is_built_from_orchestrator_params(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        # Reuse the orchestrator's fake test doubles via duck typing.
        class _DummyPerception:
            def acquire(self):  # pragma: no cover - not invoked
                raise AssertionError("not used in this test")

        class _DummyCalc:
            def compute_result(self, *a, **k):  # pragma: no cover - not invoked
                raise AssertionError("not used in this test")

        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=_DummyCalc(),  # type: ignore[arg-type]
            perception=_DummyPerception(),  # type: ignore[arg-type]
            standoff_mm=42.0,
            retreat_mm=137.0,
        )
        self.assertIsNotNone(orch.policy)
        self.assertEqual(orch.policy.standoff_mm, 42.0)  # type: ignore[union-attr]
        self.assertEqual(orch.policy.retreat_mm, 137.0)  # type: ignore[union-attr]
        self.assertIsNone(orch.policy.gripper)  # type: ignore[union-attr]

    def test_explicit_policy_is_honoured(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        arm = _FakeArm()
        custom_policy = GraspExecutionPolicy(arm=arm, standoff_mm=5.0)

        class _DummyPerception:
            def acquire(self):  # pragma: no cover
                raise AssertionError("not used")

        class _DummyCalc:
            def compute_result(self, *a, **k):  # pragma: no cover
                raise AssertionError("not used")

        orch = BinPickingOrchestrator(
            arm=arm,
            calculator=_DummyCalc(),  # type: ignore[arg-type]
            perception=_DummyPerception(),  # type: ignore[arg-type]
            policy=custom_policy,
            standoff_mm=99.0,  # should be ignored when policy is explicit
        )
        self.assertIs(orch.policy, custom_policy)


def _grasp_axes(closing: np.ndarray, approach: np.ndarray, y: float = -95.0) -> GraspPoint:
    return GraspPoint(
        position=np.array([450.0, y, 37.0]), approach=np.asarray(approach, dtype=np.float64),
        axis=np.asarray(closing, dtype=np.float64), grip_width_mm=30.0, score=0.8,
        frame=GraspFrame.BASE, label="dense",
    )


class YawClosingToBaseXTests(unittest.TestCase):
    """P2: the rigid base-Z yaw that aligns a top-down close to the reachable base-X SIGN (+X or -X)."""

    def test_yaws_to_the_requested_sign(self) -> None:
        down = np.array([0.0, 0.0, -1.0])
        # base-Y-ish (the bad axis) AND already-±X starts all land on the REQUESTED sign
        for start in ([0.0, 1.0, 0.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]):
            plus, _ = _yaw_axes_to_base_x(np.array(start), down, prefer_plus_x=True)
            minus, _ = _yaw_axes_to_base_x(np.array(start), down, prefer_plus_x=False)
            np.testing.assert_allclose(plus, [1.0, 0.0, 0.0], atol=1e-9)
            np.testing.assert_allclose(minus, [-1.0, 0.0, 0.0], atol=1e-9)

    def test_vertical_approach_preserved(self) -> None:
        _, approach = _yaw_axes_to_base_x(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), prefer_plus_x=True)
        np.testing.assert_allclose(approach, [0.0, 0.0, -1.0], atol=1e-9)

    def test_vertical_closing_is_degenerate_noop(self) -> None:
        closing, _ = _yaw_axes_to_base_x(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), prefer_plus_x=True)
        np.testing.assert_allclose(closing, [0.0, 0.0, 1.0], atol=1e-9)

    def test_yaw_is_rigid_orthonormal(self) -> None:
        # orthonormal input (a tilted but closing-perpendicular approach) -> rigid yaw keeps it orthonormal
        closing, approach = _yaw_axes_to_base_x(
            np.array([0.0, 1.0, 0.0]), np.array([0.6, 0.0, -0.8]), prefer_plus_x=True
        )
        self.assertAlmostEqual(float(np.linalg.norm(closing)), 1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.norm(approach)), 1.0, places=6)
        self.assertAlmostEqual(float(np.dot(closing, approach)), 0.0, places=6)  # orthogonality preserved
        np.testing.assert_allclose(closing, [1.0, 0.0, 0.0], atol=1e-9)


class PolicyAlignClosingTests(unittest.TestCase):
    """The policy waypoint orientation: default-off byte-identical; on -> closing yawed to the side's sign."""

    def test_default_off_byte_identical(self) -> None:
        grasp = _grasp_axes([0.0, 1.0, 0.0], [0.0, 0.0, -1.0])
        wp = GraspExecutionPolicy(arm=_FakeArm(), gripper=None)._build_waypoints(grasp)
        np.testing.assert_allclose(wp[0].quaternion_xyzw, _grasp_point_to_quaternion(grasp), atol=1e-9)

    def test_minus_y_side_uses_plus_x(self) -> None:
        grasp = _grasp_axes([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], y=-95.0)  # -Y side -> +X close
        wp = GraspExecutionPolicy(
            arm=_FakeArm(), gripper=None, align_closing_to_base_x=True
        )._build_waypoints(grasp)
        yc, ya = _yaw_axes_to_base_x(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), prefer_plus_x=True)
        np.testing.assert_allclose(wp[0].quaternion_xyzw, _quaternion_from_axes(yc, ya), atol=1e-9)

    def test_plus_y_side_uses_minus_x(self) -> None:
        grasp = _grasp_axes([0.0, 1.0, 0.0], [0.0, 0.0, -1.0], y=95.0)  # +Y side -> -X close
        wp = GraspExecutionPolicy(
            arm=_FakeArm(), gripper=None, align_closing_to_base_x=True
        )._build_waypoints(grasp)
        yc, ya = _yaw_axes_to_base_x(np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, -1.0]), prefer_plus_x=False)
        np.testing.assert_allclose(wp[0].quaternion_xyzw, _quaternion_from_axes(yc, ya), atol=1e-9)


class CloseWidthResolveTests(unittest.TestCase):
    """Adaptive close: when close_width_mm is None, close = max(min_width, grip_width - close_squeeze_mm)."""

    def test_explicit_close_width_overrides_squeeze(self) -> None:
        p = GraspExecutionPolicy(
            arm=_FakeArm(), gripper=_BasicGripper(), close_width_mm=25.0, close_squeeze_mm=11.0
        )
        self.assertEqual(p._resolve_close_width(_grasp()), 25.0)  # explicit wins; squeeze ignored

    def test_default_squeeze_is_one_mm_byte_identical(self) -> None:
        p = GraspExecutionPolicy(arm=_FakeArm(), gripper=_BasicGripper())  # close_width_mm None, squeeze=1
        self.assertEqual(p._resolve_close_width(_grasp()), 39.0)  # 40 - 1 (legacy hug)

    def test_adaptive_squeeze_clamps_below_grip_width(self) -> None:
        p = GraspExecutionPolicy(arm=_FakeArm(), gripper=_BasicGripper(), close_squeeze_mm=11.0)
        self.assertEqual(p._resolve_close_width(_grasp()), 29.0)  # 40 - 11 -> firm clamp (adapts to grip_w)

    def test_floored_at_gripper_min_width(self) -> None:
        g = _BasicGripper()
        g.min_width_mm = 35.0
        p = GraspExecutionPolicy(arm=_FakeArm(), gripper=g, close_squeeze_mm=11.0)
        self.assertEqual(p._resolve_close_width(_grasp()), 35.0)  # max(35, 40-11=29) = 35

    def test_a_hand_that_closes_onto_the_part_closes_as_far_as_it_goes(self) -> None:
        """The owner's Robotiq (2026-10-14): told the part's width less a millimetre it stopped beside a part a little
        narrower than it read and lifted nothing. A hand whose fingers stop on the part by their own force is closed as
        far as it goes; a width the caller set still wins, and a double that answers every attribute is not one."""

        class _ClosesOnto(_BasicGripper):
            closes_onto_the_part = True

        g = _ClosesOnto()
        g.min_width_mm = 2.0
        self.assertEqual(GraspExecutionPolicy(arm=_FakeArm(), gripper=g, close_squeeze_mm=11.0)
                         ._resolve_close_width(_grasp()), 2.0)
        # Its closed end where it says one, past the planner's floor (the owner's velcro band, 2026-10-15).
        g.closed_width_mm = 0.0
        self.assertEqual(GraspExecutionPolicy(arm=_FakeArm(), gripper=g)._resolve_close_width(_grasp()), 0.0)
        self.assertEqual(GraspExecutionPolicy(arm=_FakeArm(), gripper=g, close_width_mm=25.0)
                         ._resolve_close_width(_grasp()), 25.0)
        from unittest import mock

        anything = mock.MagicMock(min_width_mm=2.0)
        self.assertEqual(GraspExecutionPolicy(arm=_FakeArm(), gripper=anything)._resolve_close_width(_grasp()), 39.0)


class TheDriversKeepTheirLinesTests(unittest.TestCase):
    """A real dummy arm and a mock Isaac arm report TELEPORT, so a pick is a standoff, a line and a lift."""

    @staticmethod
    def _counting(arm: object) -> list[bool]:
        calls: list[bool] = []
        original = arm.move  # type: ignore[attr-defined]

        def move(pose: Pose, **kwargs: object):  # noqa: ANN202
            calls.append(bool(kwargs.get("linear", False)))
            return original(pose, **kwargs)

        arm.move = move  # type: ignore[attr-defined]
        return calls

    def _grasp_at(self, z_mm: float) -> GraspPoint:
        return GraspPoint(
            position=np.array([300.0, 0.0, z_mm]), approach=np.array([0.0, 0.0, -1.0]),
            axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE, label="test",
        )

    def test_a_dummy_arm_pick_is_standoff_line_retreat(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm()
        arm.connect()
        calls = self._counting(arm)
        policy = GraspExecutionPolicy(arm=arm, gripper=None)

        report = policy.execute(self._grasp_at(200.0))

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual([False, True, True], calls)

    def test_a_mock_sim_arm_pick_is_standoff_line_retreat(self) -> None:
        from src.robot.drivers.sim.arm import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        arm = IsaacRobotArm(SimRobotConfig(enabled=True, mock_mode=True))
        arm.connect()
        calls = self._counting(arm)
        policy = GraspExecutionPolicy(arm=arm, gripper=None)

        report = policy.execute(self._grasp_at(200.0))

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED, report.motion_message)
        self.assertEqual([False, True, True], calls)


if __name__ == "__main__":
    unittest.main()
