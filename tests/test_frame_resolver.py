"""Tests for the Phase S2 :class:`FrameResolver` Protocol and built-ins,
plus the end-to-end fail-closed frame contract.

Coverage:

* :class:`StaticCameraToBaseResolver` validates calibration frames and
  returns the same transform regardless of arm state.
* :class:`EyeInHandFrameResolver` composes live TCP with
  ``T_cam_to_tool`` and refuses unusable TCP frames.
* :class:`IdentityFrameResolver` returns the literal identity
  ``CAMERA -> BASE`` for synthetic test rigs.
* :class:`BinPickingOrchestrator` forwards the resolved transform to
  :meth:`GraspCalculator.compute_result` per frame.
* :class:`GraspExecutionPolicy` with ``require_base_frame_grasp=True``
  refuses a camera-frame grasp **without commanding any motion** and
  emits :attr:`PolicyOutcome.CAMERA_FRAME_REJECTED`.
* :class:`AutonomousGraspService` plumbs the resolver through and maps
  the resulting :attr:`PickOutcome.CAMERA_FRAME_REJECTED` to
  :attr:`AutonomousGraspOutcome.MISSING_CAMERA_FRAME`.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose, Transform
from src.robot.core import (
    MotionCommand,
    MotionResult,
    RobotCapabilities,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    AutonomousGraspService,
    GraspMode,
)
from src.robot.grasping import (
    EyeInHandFrameResolver,
    FrameResolver,
    IdentityFrameResolver,
    StaticCameraToBaseResolver,
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
)
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
    PickOutcome,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


_REAL_UR_CAPS = RobotCapabilities(
    vendor="ur",
    model="ur5e",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=True,
    has_native_ik=True,
    has_force_control=False,
    is_simulated=False,
)


class _StaticTcpArm:
    """Fake arm reporting a fixed TCP pose. No motion side effects."""

    def __init__(self, tcp: Pose) -> None:
        self._tcp = tcp
        self.move_calls: list[Pose] = []

    @property
    def capabilities(self) -> RobotCapabilities:
        return _REAL_UR_CAPS

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


def _identity_tcp() -> Pose:
    return Pose(
        position_mm=np.zeros(3, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="home",
    )


def _segmentation() -> SimpleNamespace:
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    return SimpleNamespace(mask=mask)


def _perception_frame() -> PerceptionFrame:
    depth = np.full((16, 16), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 8.0], [0.0, 400.0, 8.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(_segmentation(),),
    )


class _RecordingCalculator:
    """Calculator double that records every kwarg passed to compute_result."""

    def __init__(self, result_factory) -> None:
        self._factory = result_factory
        self.calls: list[dict] = []

    def compute_result(self, *args: object, **kwargs: object) -> GraspResult:
        self.calls.append({"args": args, "kwargs": dict(kwargs)})
        return self._factory()


def _camera_frame_success() -> GraspResult:
    """A successful result with a CAMERA-frame candidate.

    Used to exercise the execution-policy fail-closed guard: the
    orchestrator selects this candidate, hands it to the policy, and
    the policy must refuse before any motion.
    """

    return GraspResult(
        candidates=(
            GraspPoint(
                position=np.array([0.0, 0.0, 300.0]),
                approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]),
                grip_width_mm=40.0,
                score=0.85,
                frame=GraspFrame.CAMERA,
                label="cam_grasp",
            ),
        ),
        reasons=(),
        top_score=0.85,
    )


def _base_frame_success() -> GraspResult:
    return GraspResult(
        candidates=(
            GraspPoint(
                position=np.array([100.0, 50.0, 400.0]),
                approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]),
                grip_width_mm=40.0,
                score=0.9,
                frame=GraspFrame.BASE,
                label="base_grasp",
            ),
        ),
        reasons=(),
        top_score=0.9,
    )


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


# ---------------------------------------------------------------------------
# StaticCameraToBaseResolver
# ---------------------------------------------------------------------------


class StaticCameraToBaseResolverTests(unittest.TestCase):
    def _valid_transform(self) -> Transform:
        return Transform.from_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, -50.0],
                    [0.0, 0.0, 1.0, 800.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )

    def test_returns_same_transform_regardless_of_arm_or_frame(self) -> None:
        t = self._valid_transform()
        resolver = StaticCameraToBaseResolver(transform=t)
        arm = _StaticTcpArm(_identity_tcp())
        frame = _perception_frame()
        first = resolver.camera_to_base_for_frame(frame, arm=arm)
        second = resolver.camera_to_base_for_frame(frame, arm=arm)
        self.assertIs(first, t)
        self.assertIs(second, t)
        # Resolver must not move or otherwise touch the arm.
        self.assertEqual(arm.move_calls, [])

    def test_rejects_wrong_frame_direction(self) -> None:
        wrong = Transform.identity(from_frame=Frame.BASE, to_frame=Frame.CAMERA)
        with self.assertRaises(ValueError) as ctx:
            StaticCameraToBaseResolver(transform=wrong)
        self.assertIn("CAMERA -> BASE", str(ctx.exception))

    def test_implements_protocol(self) -> None:
        resolver = StaticCameraToBaseResolver(transform=self._valid_transform())
        self.assertIsInstance(resolver, FrameResolver)


# ---------------------------------------------------------------------------
# EyeInHandFrameResolver
# ---------------------------------------------------------------------------


class EyeInHandFrameResolverTests(unittest.TestCase):
    def _t_cam_to_tool(self) -> Transform:
        # Camera is 30 mm above the TCP looking down (identity rotation).
        return Transform.from_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, -30.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            from_frame=Frame.CAMERA,
            to_frame=Frame.TOOL,
        )

    def test_composes_live_tcp_with_calibration(self) -> None:
        resolver = EyeInHandFrameResolver(t_cam_to_tool=self._t_cam_to_tool())
        tcp = Pose(
            position_mm=np.array([200.0, 100.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="tcp",
        )
        arm = _StaticTcpArm(tcp)
        resolved = resolver.camera_to_base_for_frame(_perception_frame(), arm=arm)
        self.assertIs(resolved.from_frame, Frame.CAMERA)
        self.assertIs(resolved.to_frame, Frame.BASE)
        # Origin in camera frame -> (0,0,-30) in tool -> tool origin in
        # base + (0,0,-30). Identity rotation makes the math trivial.
        np.testing.assert_allclose(
            resolved.translation_mm,
            np.array([200.0, 100.0, 470.0], dtype=np.float64),
        )

    def test_reads_arm_pose_each_call(self) -> None:
        resolver = EyeInHandFrameResolver(t_cam_to_tool=self._t_cam_to_tool())
        first_tcp = Pose(
            position_mm=np.array([100.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )
        arm = _StaticTcpArm(first_tcp)
        first = resolver.camera_to_base_for_frame(_perception_frame(), arm=arm)
        # Move the arm; the resolver must pick up the new pose.
        arm._tcp = Pose(  # type: ignore[attr-defined]
            position_mm=np.array([300.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )
        second = resolver.camera_to_base_for_frame(_perception_frame(), arm=arm)
        self.assertFalse(
            np.array_equal(first.translation_mm, second.translation_mm)
        )

    def test_rejects_wrong_calibration_frames(self) -> None:
        bad = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        with self.assertRaises(ValueError) as ctx:
            EyeInHandFrameResolver(t_cam_to_tool=bad)
        self.assertIn("CAMERA -> TOOL", str(ctx.exception))

    def test_rejects_non_base_tcp_pose(self) -> None:
        resolver = EyeInHandFrameResolver(t_cam_to_tool=self._t_cam_to_tool())
        bad_tcp = Pose(
            position_mm=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.TOOL,
        )
        arm = _StaticTcpArm(bad_tcp)
        with self.assertRaises(ValueError) as ctx:
            resolver.camera_to_base_for_frame(_perception_frame(), arm=arm)
        self.assertIn("Frame.BASE", str(ctx.exception))


# ---------------------------------------------------------------------------
# IdentityFrameResolver
# ---------------------------------------------------------------------------


class IdentityFrameResolverTests(unittest.TestCase):
    def test_returns_identity_camera_to_base(self) -> None:
        resolver = IdentityFrameResolver()
        arm = _StaticTcpArm(_identity_tcp())
        resolved = resolver.camera_to_base_for_frame(_perception_frame(), arm=arm)
        self.assertIs(resolved.from_frame, Frame.CAMERA)
        self.assertIs(resolved.to_frame, Frame.BASE)
        np.testing.assert_allclose(resolved.translation_mm, np.zeros(3))


# ---------------------------------------------------------------------------
# Orchestrator forwarding
# ---------------------------------------------------------------------------


class OrchestratorFrameResolverForwardingTests(unittest.TestCase):
    """The orchestrator must forward the resolved transform into
    :meth:`GraspCalculator.compute_result`.
    """

    def test_no_resolver_forwards_no_transform(self) -> None:
        calculator = _RecordingCalculator(_base_frame_success)
        orch = BinPickingOrchestrator(
            arm=_StaticTcpArm(_identity_tcp()),  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
        )
        orch.run()
        self.assertEqual(len(calculator.calls), 1)
        kwargs = calculator.calls[0]["kwargs"]
        # Neither legacy nor typed transform kwargs should be present.
        self.assertNotIn("camera_to_base", kwargs)
        self.assertNotIn("T_cam_to_base", kwargs)

    def test_static_resolver_forwards_transform(self) -> None:
        t = Transform.from_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 100.0],
                    [0.0, 1.0, 0.0, -50.0],
                    [0.0, 0.0, 1.0, 800.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            ),
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )
        resolver = StaticCameraToBaseResolver(transform=t)
        calculator = _RecordingCalculator(_base_frame_success)
        orch = BinPickingOrchestrator(
            arm=_StaticTcpArm(_identity_tcp()),  # type: ignore[arg-type]
            calculator=calculator,  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            frame_resolver=resolver,
        )
        orch.run()
        self.assertEqual(len(calculator.calls), 1)
        forwarded = calculator.calls[0]["kwargs"].get("camera_to_base")
        self.assertIs(forwarded, t)


# ---------------------------------------------------------------------------
# Execution-policy fail-closed guard
# ---------------------------------------------------------------------------


class ExecutionPolicyFailClosedTests(unittest.TestCase):
    def test_default_policy_executes_camera_frame_grasp(self) -> None:
        # Backward-compat: without the guard the legacy behaviour is
        # preserved exactly --- a CAMERA-frame grasp is still executed.
        arm = _StaticTcpArm(_identity_tcp())
        policy = GraspExecutionPolicy(arm=arm)  # type: ignore[arg-type]
        cam_grasp = _camera_frame_success().best
        assert cam_grasp is not None
        report = policy.execute(cam_grasp)
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertGreater(len(arm.move_calls), 0)

    def test_guarded_policy_rejects_camera_frame_without_motion(self) -> None:
        arm = _StaticTcpArm(_identity_tcp())
        policy = GraspExecutionPolicy(
            arm=arm,  # type: ignore[arg-type]
            require_base_frame_grasp=True,
        )
        cam_grasp = _camera_frame_success().best
        assert cam_grasp is not None
        report = policy.execute(cam_grasp)
        self.assertIs(report.outcome, PolicyOutcome.CAMERA_FRAME_REJECTED)
        self.assertEqual(report.waypoints, ())
        # No motion was commanded --- this is the core safety property.
        self.assertEqual(arm.move_calls, [])

    def test_guarded_policy_still_executes_base_frame_grasp(self) -> None:
        arm = _StaticTcpArm(_identity_tcp())
        policy = GraspExecutionPolicy(
            arm=arm,  # type: ignore[arg-type]
            require_base_frame_grasp=True,
        )
        base_grasp = _base_frame_success().best
        assert base_grasp is not None
        report = policy.execute(base_grasp)
        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)


# ---------------------------------------------------------------------------
# AutonomousGraspService end-to-end
# ---------------------------------------------------------------------------


class AutonomousGraspServiceFrameResolverTests(unittest.TestCase):
    def test_resolver_wires_fail_closed_default_policy(self) -> None:
        # Wiring a resolver but no explicit policy must flip the
        # fail-closed guard on the auto-built policy.
        resolver = IdentityFrameResolver()
        service = AutonomousGraspService.from_components(
            arm=_StaticTcpArm(_identity_tcp()),  # type: ignore[arg-type]
            calculator=_RecordingCalculator(_camera_frame_success),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
            frame_resolver=resolver,
        )
        policy = service.runtime.orchestrator.policy
        self.assertIsNotNone(policy)
        assert policy is not None  # for type checkers
        self.assertTrue(policy.require_base_frame_grasp)

    def test_camera_frame_grasp_maps_to_missing_camera_frame(self) -> None:
        resolver = IdentityFrameResolver()
        arm = _StaticTcpArm(_identity_tcp())
        service = AutonomousGraspService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_RecordingCalculator(_camera_frame_success),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
            frame_resolver=resolver,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.MISSING_CAMERA_FRAME)
        self.assertIsNotNone(report.pick_report)
        assert report.pick_report is not None
        self.assertIs(
            report.pick_report.outcome, PickOutcome.CAMERA_FRAME_REJECTED
        )
        # End-to-end safety: no motion was commanded.
        self.assertEqual(arm.move_calls, [])

    def test_no_resolver_preserves_legacy_camera_frame_execution(self) -> None:
        # Without a resolver the legacy path runs unchanged: a
        # CAMERA-frame grasp is still executed today. This is the
        # status quo we are preserving for byte-equivalent EASY mode.
        arm = _StaticTcpArm(_identity_tcp())
        service = AutonomousGraspService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_RecordingCalculator(_camera_frame_success),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
        )

        report = service.pick()

        self.assertIs(report.outcome, AutonomousGraspOutcome.SUCCEEDED)
        self.assertGreater(len(arm.move_calls), 0)

    def test_explicit_policy_is_not_mutated(self) -> None:
        # Operators who pass an explicit policy own its configuration.
        # The service must NOT silently flip require_base_frame_grasp.
        arm = _StaticTcpArm(_identity_tcp())
        explicit_policy = GraspExecutionPolicy(
            arm=arm,  # type: ignore[arg-type]
            require_base_frame_grasp=False,
        )
        service = AutonomousGraspService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_RecordingCalculator(_camera_frame_success),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            mode=GraspMode.EASY,
            frame_resolver=IdentityFrameResolver(),
            policy=explicit_policy,
        )
        self.assertIs(service.runtime.orchestrator.policy, explicit_policy)
        self.assertFalse(explicit_policy.require_base_frame_grasp)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
