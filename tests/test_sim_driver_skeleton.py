"""Tests for Stage 2 (typed PolicyReport) and Stage 3 (Isaac sim skeleton).

Stage 2
-------
Confirms that :class:`GraspExecutionPolicy` now prefers the typed
:meth:`RobotArm.move` Phase N contract when the driver supplies it,
and that the resulting :class:`MotionStatus` is preserved on the
:class:`PolicyReport`.

Stage 3
-------
Confirms that the Isaac-backed sim driver skeleton:

* imports cleanly on hosts without the Isaac SDK (macOS dev box);
* is registered against :class:`RobotVendor.SIM` and constructible
  through the public factory;
* honours the typed motion contract with
  :attr:`MotionStatus.UNSUPPORTED` for motion methods until Phase O
  Stage 4 lands;
* uses honest :class:`RobotCapabilities` (``is_simulated=True`` and no
  overclaimed features).
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotArm,
    RobotVendor,
)
from src.robot.drivers import available_vendors, create_arm
from src.robot.drivers.sim import (
    ISAAC_CAPABILITIES,
    IsaacNotAvailableError,
    IsaacRobotArm,
    SimCameraConfig,
    SimRobotConfig,
    willy_pose_to_isaac,
    isaac_pose_to_willy,
    isaac_rotmat_to_wxyz,
    isaac_wxyz_to_xyzw,
    metres_to_millimetres,
    millimetres_to_metres,
    xyzw_to_isaac_wxyz,
)
from src.robot.grasping.motion.execution_policy import (
    GraspExecutionPolicy,
    PolicyOutcome,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


# ---------------------------------------------------------------------------
# Stage 2 fixtures: typed-move fake
# ---------------------------------------------------------------------------


class _TypedArm:
    """Test double exposing the Phase N typed :meth:`move` surface.

    Used in Stage 2 tests so we can prove the policy:
    1. prefers ``arm.move`` over legacy ``arm.move_to``,
    2. surfaces :class:`MotionStatus` on :class:`PolicyReport`,
    3. routes typed failures to :attr:`PolicyOutcome.MOTION_FAILED`
       without raising.
    """

    def __init__(self, fail_at: int | None = None,
                 fail_status: MotionStatus = MotionStatus.WORKSPACE_REJECTED) -> None:
        self.move_calls: list[Pose] = []
        self.move_to_calls: list[Pose] = []
        self.fail_at = fail_at
        self.fail_status = fail_status

    def get_tcp_pose(self) -> Pose:
        return Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )

    def move(self, pose: Pose, **_: object) -> MotionResult:
        if self.fail_at is not None and len(self.move_calls) >= self.fail_at:
            return MotionResult.failed(
                self.fail_status,
                MotionCommand.MOVE_TO,
                target_pose=pose,
                message=f"simulated {self.fail_status.value}",
            )
        self.move_calls.append(pose)
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def move_to(self, pose: Pose, **_: object) -> None:
        # Should be unreachable when ``move`` is implemented \u2014 asserted
        # in :class:`PolicyTypedMoveTests` below.
        self.move_to_calls.append(pose)


def _grasp() -> GraspPoint:
    return GraspPoint(
        position=np.array([100.0, 50.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=0.9,
        frame=GraspFrame.BASE,
        label="test",
    )


# ---------------------------------------------------------------------------
# Stage 2 tests: typed PolicyReport plumbing
# ---------------------------------------------------------------------------


class PolicyTypedMoveTests(unittest.TestCase):
    def test_policy_prefers_typed_move_over_legacy_move_to(self) -> None:
        arm = _TypedArm()
        policy = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4)

        report = policy.execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        # Legacy bool path MUST stay untouched when the typed surface
        # is present \u2014 this is the core Phase N migration assertion.
        self.assertEqual(arm.move_to_calls, [])
        # 4 approach + 1 retreat = 5 commanded moves.
        self.assertEqual(len(arm.move_calls), 5)

    def test_executed_status_is_surfaced_on_report(self) -> None:
        arm = _TypedArm()
        policy = GraspExecutionPolicy(arm=arm, gripper=None)

        report = policy.execute(_grasp())

        self.assertIs(report.motion_status, MotionStatus.EXECUTED)
        self.assertEqual(report.motion_message, "")
        self.assertIsNone(report.error)

    def test_workspace_rejection_is_preserved_on_report(self) -> None:
        # Fail before the first approach waypoint so we get a clean
        # WORKSPACE_REJECTED on attempt 0.
        arm = _TypedArm(
            fail_at=0, fail_status=MotionStatus.WORKSPACE_REJECTED,
        )
        policy = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4)

        report = policy.execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertIs(report.motion_status, MotionStatus.WORKSPACE_REJECTED)
        self.assertIn("workspace_rejected", report.motion_message)
        # No waypoints were committed before the rejection.
        self.assertEqual(report.waypoints, ())
        # No legacy raise needed \u2014 typed result drove the outcome.
        self.assertIsNone(report.error)

    def test_typed_failure_after_some_waypoints_keeps_history(self) -> None:
        arm = _TypedArm(
            fail_at=2, fail_status=MotionStatus.CONTROLLER_REJECTED,
        )
        policy = GraspExecutionPolicy(arm=arm, gripper=None, approach_steps=4)

        report = policy.execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.MOTION_FAILED)
        self.assertIs(report.motion_status, MotionStatus.CONTROLLER_REJECTED)
        # First two approach waypoints landed before the rejection.
        self.assertEqual(len(report.waypoints), 2)

    def test_legacy_move_to_fake_still_works(self) -> None:
        """Drivers without ``move()`` must continue to function.

        Stage 2 keeps Q1 (coexist): the policy falls back to the bool
        ``move_to`` path when the typed contract is not implemented on
        the supplied arm. ``motion_status`` is ``None`` in that case
        because no typed information was produced.
        """

        class _LegacyArm:
            def __init__(self) -> None:
                self.calls: list[Pose] = []

            def get_tcp_pose(self) -> Pose:
                return Pose(
                    position_mm=np.zeros(3),
                    quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                    frame=Frame.BASE,
                )

            def move_to(self, pose: Pose, **_: object) -> None:
                self.calls.append(pose)

        arm = _LegacyArm()
        policy = GraspExecutionPolicy(arm=arm, gripper=None)
        report = policy.execute(_grasp())

        self.assertIs(report.outcome, PolicyOutcome.EXECUTED)
        self.assertIsNone(report.motion_status)
        self.assertEqual(report.motion_message, "")
        self.assertGreater(len(arm.calls), 0)


# ---------------------------------------------------------------------------
# Stage 3 tests: sim driver skeleton
# ---------------------------------------------------------------------------


class SimDriverImportSafetyTests(unittest.TestCase):
    def test_package_import_is_macos_safe(self) -> None:
        # The very fact that the imports at the top of this module
        # succeeded already proves the contract; we re-import here to
        # make the intent explicit and provide a stable assertion point.
        import importlib

        module = importlib.import_module(
            "src.robot.drivers.sim",
        )
        self.assertTrue(hasattr(module, "IsaacRobotArm"))
        self.assertTrue(hasattr(module, "SimRobotConfig"))

    def test_sim_vendor_is_registered(self) -> None:
        self.assertIn(RobotVendor.SIM.value, available_vendors())


class SimRobotConfigTests(unittest.TestCase):
    def test_defaults_are_disabled_and_isaac_backend(self) -> None:
        cfg = SimRobotConfig()
        # Master switch defaults to disabled so unconfigured runtimes
        # cannot accidentally spin up Isaac.
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.backend, "isaac")
        self.assertEqual(cfg.cameras, {})

    def test_camera_config_accepts_both_mounting_modes(self) -> None:
        eth = SimCameraConfig(prim_path="/World/cam_eth", mounting_mode="eye_to_hand")
        eih = SimCameraConfig(prim_path="/World/cam_eih", mounting_mode="eye_in_hand")
        self.assertEqual(eth.mounting_mode, "eye_to_hand")
        self.assertEqual(eih.mounting_mode, "eye_in_hand")


class SimAdapterTests(unittest.TestCase):
    """Conversion helpers MUST work without Isaac installed."""

    def test_unit_conversion_round_trip(self) -> None:
        mm = np.array([100.0, -200.5, 50.25])
        np.testing.assert_allclose(
            metres_to_millimetres(millimetres_to_metres(mm)), mm,
        )

    def test_quaternion_convention_round_trip(self) -> None:
        xyzw = np.array([0.1, -0.2, 0.3, 0.9])
        np.testing.assert_allclose(
            isaac_wxyz_to_xyzw(xyzw_to_isaac_wxyz(xyzw)), xyzw,
        )

    def test_rotmat_to_wxyz_known_and_roundtrip(self) -> None:
        # Identity -> [1, 0, 0, 0]; unit norm always.
        np.testing.assert_allclose(
            isaac_rotmat_to_wxyz(np.eye(3)), [1.0, 0.0, 0.0, 0.0], atol=1e-9
        )
        # 90 deg about +Z -> [cos45, 0, 0, sin45] (sign-agnostic: q and -q are equal rotations).
        rz = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        s = np.sqrt(0.5)
        np.testing.assert_allclose(np.abs(isaac_rotmat_to_wxyz(rz)), [s, 0.0, 0.0, s], atol=1e-7)
        # 180 deg about +X -> [0, 1, 0, 0] (exercises a non-trace branch).
        rx = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
        np.testing.assert_allclose(np.abs(isaac_rotmat_to_wxyz(rx)), [0.0, 1.0, 0.0, 0.0], atol=1e-7)

        def wxyz_to_rotmat(q: np.ndarray) -> np.ndarray:
            w, x, y, z = q
            return np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ])

        # Round-trip several non-trivial rotations: R -> wxyz -> R must be exact.
        for q in (
            np.array([0.5, 0.5, 0.5, 0.5]),
            np.array([0.9239, 0.0, 0.3827, 0.0]),
            np.array([0.7071, -0.7071, 0.0, 0.0]),
        ):
            q = q / np.linalg.norm(q)
            rotmat = wxyz_to_rotmat(q)
            recovered = isaac_rotmat_to_wxyz(rotmat)
            np.testing.assert_allclose(wxyz_to_rotmat(recovered), rotmat, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(recovered), 1.0, atol=1e-9)

    def test_pose_round_trip_preserves_position_and_orientation(self) -> None:
        pose = Pose(
            position_mm=np.array([300.0, -150.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="sim-test",
        )
        position_m, orientation_wxyz = willy_pose_to_isaac(pose)
        # 300 mm \u2192 0.3 m round-trip.
        np.testing.assert_allclose(position_m, np.array([0.3, -0.15, 0.5]))
        # WXYZ ordering check: identity-XYZW becomes [1, 0, 0, 0] in WXYZ.
        np.testing.assert_allclose(orientation_wxyz, np.array([1.0, 0.0, 0.0, 0.0]))

        recovered = isaac_pose_to_willy(position_m, orientation_wxyz, label="back")
        np.testing.assert_allclose(recovered.position_mm, pose.position_mm)
        np.testing.assert_allclose(recovered.quaternion_xyzw, pose.quaternion_xyzw)
        self.assertIs(recovered.frame, Frame.BASE)

    def test_pose_conversion_rejects_non_base_frame(self) -> None:
        pose = Pose(
            position_mm=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.CAMERA,
        )
        with self.assertRaises(ValueError):
            willy_pose_to_isaac(pose)


class SimDriverSkeletonTests(unittest.TestCase):
    def _config(self) -> SimRobotConfig:
        return SimRobotConfig(
            enabled=True,
            scene="placeholder.usd",
            robot_prim_path="/World/Robot",
            home_joint_positions=(0.0, -1.0, 1.0, 0.0, 1.0, 0.0),
        )

    def test_factory_returns_isaac_robot_arm(self) -> None:
        arm = create_arm(RobotVendor.SIM, config=self._config())
        self.assertIsInstance(arm, IsaacRobotArm)
        self.assertIsInstance(arm, RobotArm)

    def test_capabilities_are_honest(self) -> None:
        arm = IsaacRobotArm(self._config())
        caps = arm.capabilities
        self.assertEqual(caps.vendor, "sim")
        self.assertTrue(caps.is_simulated)
        # The skeleton MUST NOT overclaim features it has not built.
        self.assertFalse(caps.has_force_control)
        self.assertFalse(caps.supports_async_move)
        # Capability surface is shared with the module-level constant
        # so consumers can introspect either.
        self.assertEqual(arm.capabilities, ISAAC_CAPABILITIES)

    def test_construction_does_not_require_connection(self) -> None:
        arm = IsaacRobotArm(self._config())
        self.assertFalse(arm.is_connected)

    def test_connect_raises_isaac_not_available_on_macos(self) -> None:
        # CI on macOS will exercise this branch. On a Linux/Isaac host
        # the test still passes if Isaac is installed but the skeleton
        # session refuses to boot (NotImplementedError); both branches
        # are honest skeleton behaviour.
        arm = IsaacRobotArm(self._config())
        with self.assertRaises((IsaacNotAvailableError, NotImplementedError)):
            arm.connect()

    def test_typed_move_when_uninitialized_returns_connection_error(self) -> None:
        # Non-mock arm with the connected flag forced but no real articulation (Isaac was
        # never brought up): move() must return a typed CONNECTION_ERROR, not leak the
        # kinematics RobotConnectionError. The real RMPflow+IK path is exercised on-box.
        arm = IsaacRobotArm(self._config())
        arm._connected = True  # type: ignore[attr-defined]
        result = arm.move(
            Pose(
                position_mm=np.array([400.0, 0.0, 300.0]),
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame=Frame.BASE,
            ),
        )
        self.assertIsInstance(result, MotionResult)
        self.assertIs(result.status, MotionStatus.CONNECTION_ERROR)
        self.assertFalse(result.ok)

    def test_typed_move_invalid_frame_returns_invalid_target(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm._connected = True  # type: ignore[attr-defined]
        result = arm.move(
            Pose(
                position_mm=np.zeros(3),
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame=Frame.CAMERA,
            ),
        )
        self.assertIs(result.status, MotionStatus.INVALID_TARGET)

    def test_typed_move_when_disconnected_returns_connection_error(self) -> None:
        arm = IsaacRobotArm(self._config())
        result = arm.move(
            Pose(
                position_mm=np.zeros(3),
                quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                frame=Frame.BASE,
            ),
        )
        self.assertIs(result.status, MotionStatus.CONNECTION_ERROR)

    def test_skeleton_state_uses_configured_home_joints(self) -> None:
        cfg = self._config()
        arm = IsaacRobotArm(cfg)
        arm._connected = True  # type: ignore[attr-defined]
        joints = arm.get_joint_positions()
        self.assertIsInstance(joints, JointPositions)
        self.assertEqual(tuple(joints.values.tolist()), cfg.home_joint_positions)


# ---------------------------------------------------------------------------
# Phase Q-B: mock_mode \u2014 pure-Python kinematic mock (no Isaac required)
# ---------------------------------------------------------------------------


class SimMockModeTests(unittest.TestCase):
    """``SimRobotConfig.mock_mode`` makes the SIM driver fully usable
    on hosts without Isaac. These tests pin the contract so downstream
    Willy projects can lean on the mock for CI."""

    def _config(self, **overrides) -> SimRobotConfig:
        defaults = dict(
            enabled=True,
            mock_mode=True,
            scene="mock.usd",
            robot_prim_path="/World/Robot",
            home_joint_positions=(0.0, -1.0, 1.0, 0.0, 1.0, 0.0),
        )
        defaults.update(overrides)
        return SimRobotConfig(**defaults)

    def _pose(self, x: float = 500.0) -> Pose:
        return Pose(
            position_mm=np.array([x, 0.0, 300.0], dtype=np.float64),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            frame=Frame.BASE,
        )

    def test_mock_mode_connect_succeeds_without_isaac(self) -> None:
        arm = IsaacRobotArm(self._config())
        # No IsaacNotAvailableError on macOS: the whole point of
        # mock_mode is that the SDK is never touched.
        arm.connect()
        self.assertTrue(arm.is_connected)
        self.assertTrue(arm.mock_mode)

    def test_mock_mode_move_returns_executed_and_updates_tcp(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm.connect()
        target = self._pose(x=650.0)
        result = arm.move(target)
        self.assertIsInstance(result, MotionResult)
        self.assertIs(result.status, MotionStatus.EXECUTED)
        self.assertTrue(result.ok)
        tcp = arm.get_tcp_pose()
        np.testing.assert_allclose(tcp.position_mm, target.position_mm)

    def test_mock_mode_move_requires_connection(self) -> None:
        arm = IsaacRobotArm(self._config())
        # Did not call connect() \u2014 still must reject with typed status.
        result = arm.move(self._pose())
        self.assertIs(result.status, MotionStatus.CONNECTION_ERROR)

    def test_mock_mode_move_rejects_non_base_frame(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm.connect()
        bad = Pose(
            position_mm=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.CAMERA,
        )
        result = arm.move(bad)
        self.assertIs(result.status, MotionStatus.INVALID_TARGET)

    def test_mock_mode_move_to_and_move_home_track_state(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm.connect()
        home_tcp = arm.get_tcp_pose()
        target = self._pose(x=720.0)
        self.assertTrue(arm.move_to(target))
        np.testing.assert_allclose(
            arm.get_tcp_pose().position_mm, target.position_mm
        )
        self.assertTrue(arm.move_home())
        np.testing.assert_allclose(
            arm.get_tcp_pose().position_mm, home_tcp.position_mm
        )

    def test_mock_stop_is_safe_noop_and_workspace_accepts_base(self) -> None:
        # L3.3: stop() is a best-effort no-op in mock_mode (never raises); is_inside_workspace returns
        # True (the sim driver owns no box — it is enforced by the preflight WorkspaceGuard per L2.1).
        arm = IsaacRobotArm(self._config())
        arm.connect()
        arm.stop()  # must not raise
        self.assertTrue(arm.is_inside_workspace(self._pose(x=400.0)))

    def test_mock_mode_move_joint_updates_joints_not_tcp(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm.connect()
        tcp_before = arm.get_tcp_pose().position_mm.copy()
        new_joints = JointPositions(
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
        )
        arm.move_joint(new_joints)  # must NOT raise
        np.testing.assert_allclose(arm.get_joint_positions().values, new_joints.values)
        # Identity-kinematics contract: joint move does not move TCP.
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, tcp_before)

    def test_mock_mode_move_linear_updates_tcp(self) -> None:
        arm = IsaacRobotArm(self._config())
        arm.connect()
        target = self._pose(x=815.0)
        arm.move_linear(target)  # must NOT raise
        np.testing.assert_allclose(arm.get_tcp_pose().position_mm, target.position_mm)

    def test_mock_mode_does_not_import_isaac(self) -> None:
        # Belt-and-braces: even if some omni stub is installed, the
        # mock-mode session must not pull it in. We assert on
        # ``sys.modules`` after a connect() round-trip.
        import sys

        arm = IsaacRobotArm(self._config())
        before = {k for k in sys.modules if k.startswith(("omni", "isaacsim"))}
        arm.connect()
        after = {k for k in sys.modules if k.startswith(("omni", "isaacsim"))}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
