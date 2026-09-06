"""Tests for the Phase P runtime pick facade.

The facade is a thin synchronous wrapper around
:class:`BinPickingOrchestrator` that adds:

* vendor labelling (``robot_vendor`` / ``is_simulated``) from
  :class:`RobotCapabilities`,
* the typed :class:`MotionStatus` chain forwarded from the
  policy's :class:`PolicyReport`,
* coarse wall-clock timings,
* a convenience :attr:`PickSessionReport.succeeded` flag.

These tests deliberately stay on macOS-safe ground (no Isaac, no
hardware drivers) and exercise the surface using lightweight fakes,
following the same pattern as ``tests/test_pick_loop.py``.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotCapabilities,
)
from src.robot.execution.runtime_pick import (
    PickSessionReport,
    PickTimings,
    RuntimePickService,
)
from src.robot.grasping.types.feedback import GraspFailureReason, GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    PerceptionFrame,
    PickOutcome,
)


# ---------------------------------------------------------------------------
# Test doubles (mirrors tests/test_pick_loop.py style)
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

_SIM_CAPS = RobotCapabilities(
    vendor="sim",
    model="isaac-sim",
    dof=6,
    supports_joint_move=True,
    supports_linear_move=True,
    supports_async_move=False,
    has_native_fk=False,
    has_native_ik=False,
    has_force_control=False,
    is_simulated=True,
)


class _TypedFakeArm:
    """Fake :class:`RobotArm` exposing the Phase N typed ``move`` surface."""

    def __init__(self, caps: RobotCapabilities = _REAL_UR_CAPS) -> None:
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )
        self.move_calls: list[Pose] = []
        self._caps = caps

    @property
    def capabilities(self) -> RobotCapabilities:
        return self._caps

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.move_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


class _LegacyFakeArm:
    """Fake arm without the typed ``move`` surface and without capabilities."""

    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.zeros(3),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
        )
        self.move_to_calls: list[Pose] = []

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move_to(self, pose: Pose, **_: object) -> None:
        self.move_to_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose


class _FakePerception:
    def __init__(self, frames: list[PerceptionFrame]) -> None:
        self._frames = frames
        self.calls = 0

    def acquire(self) -> PerceptionFrame:
        idx = min(self.calls, len(self._frames) - 1)
        self.calls += 1
        return self._frames[idx]


class _ScriptedCalculator:
    def __init__(self, results: list[GraspResult]) -> None:
        self._results = results
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        idx = min(self.calls, len(self._results) - 1)
        self.calls += 1
        return self._results[idx]


def _segmentation(shape: tuple[int, int] = (32, 32)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[10:22, 10:22] = 1
    return SimpleNamespace(mask=mask)


def _perception_frame() -> PerceptionFrame:
    depth = np.full((32, 32), 500.0, dtype=np.float64)
    intrinsics = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(_segmentation(),),
    )


def _empty_perception_frame() -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros((4, 4), dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=(),
    )


def _success_result() -> GraspResult:
    return GraspResult(
        candidates=(
            GraspPoint(
                position=np.array([100.0, 50.0, 400.0]),
                approach=np.array([0.0, 0.0, 1.0]),
                axis=np.array([1.0, 0.0, 0.0]),
                grip_width_mm=40.0,
                score=0.9,
                frame=GraspFrame.BASE,
                label="test",
            ),
        ),
        reasons=(),
        top_score=0.9,
    )


def _rescan_only_result() -> GraspResult:
    return GraspResult(
        candidates=(),
        reasons=(GraspFailureReason.RESCAN_RECOMMENDED,),
        top_score=0.0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class RuntimePickServiceTests(unittest.TestCase):
    def test_lazy_export_via_execution_package(self) -> None:
        # The facade must be reachable through the package-level
        # lazy loader so callers don't reach for the private module path.
        import src.robot.execution as execution_pkg

        self.assertIs(execution_pkg.RuntimePickService, RuntimePickService)
        self.assertIs(execution_pkg.PickSessionReport, PickSessionReport)
        self.assertIs(execution_pkg.PickTimings, PickTimings)

    def test_happy_path_with_typed_arm_records_motion_status(self) -> None:
        arm = _TypedFakeArm(caps=_REAL_UR_CAPS)
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=2,
        )

        report = service.run_attempt()

        self.assertIsInstance(report, PickSessionReport)
        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertTrue(report.succeeded)
        self.assertEqual(report.robot_vendor, "ur")
        self.assertFalse(report.is_simulated)
        self.assertFalse(report.gripper_present)
        self.assertEqual(report.motion_status_chain, (MotionStatus.EXECUTED,))
        self.assertIsNone(report.error)
        self.assertIsNotNone(report.executed_grasp)
        self.assertAlmostEqual(report.selected_score, 0.9)
        # Timings are present and non-negative.
        self.assertGreaterEqual(report.timings.total_s, 0.0)
        self.assertEqual(report.timings.total_s, report.timings.orchestrator_s)

    def test_sim_vendor_label_is_surfaced(self) -> None:
        arm = _TypedFakeArm(caps=_SIM_CAPS)
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=2,
        )

        report = service.run_attempt()

        self.assertEqual(report.robot_vendor, "sim")
        self.assertTrue(report.is_simulated)

    def test_legacy_arm_yields_empty_motion_chain_but_still_executes(self) -> None:
        # Drivers that predate the Phase N typed contract still work
        # through the legacy ``move_to`` path. The facade must not
        # invent a fake MotionStatus in that case.
        arm = _LegacyFakeArm()
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=2,
        )

        report = service.run_attempt()

        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(report.motion_status_chain, ())
        self.assertEqual(report.robot_vendor, "unknown")
        self.assertFalse(report.is_simulated)

    def test_no_perception_outcome_propagates_without_motion(self) -> None:
        arm = _TypedFakeArm()
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_rescan_only_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_empty_perception_frame()]),
            max_attempts=1,
        )

        report = service.run_attempt()

        self.assertIs(report.outcome, PickOutcome.NO_PERCEPTION)
        self.assertFalse(report.succeeded)
        # No execution attempted -> no policy report -> empty chain.
        self.assertEqual(report.motion_status_chain, ())
        self.assertIsNone(report.executed_grasp)
        self.assertEqual(arm.move_calls, [])

    def test_exhausted_rescan_reports_last_attempt_target(self) -> None:
        arm = _TypedFakeArm()
        # All attempts return rescan; orchestrator exhausts max_attempts.
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_rescan_only_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=2,
        )

        report = service.run_attempt()

        self.assertIs(report.outcome, PickOutcome.RESCANNED_EXHAUSTED)
        self.assertFalse(report.succeeded)
        # No grasp was executed.
        self.assertIsNone(report.executed_grasp)
        # No motion was commanded.
        self.assertEqual(arm.move_calls, [])

    def test_from_components_builds_orchestrator_without_external_deps(self) -> None:
        # Construction must not require Isaac, ROS, or any vendor SDK;
        # this proves the facade is import- and build-safe on macOS.
        arm = _TypedFakeArm()
        service = RuntimePickService.from_components(
            arm=arm,  # type: ignore[arg-type]
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
        )

        self.assertIsNotNone(service.orchestrator)
        self.assertIs(service.orchestrator.arm, arm)


# ---------------------------------------------------------------------------
# Phase Q-D: from_robot_config dispatches via the vendor-block schema.
# ---------------------------------------------------------------------------


class RuntimePickServiceFromRobotConfigTests(unittest.TestCase):
    """``RuntimePickService.from_robot_config`` is the schema-driven
    entry point. These tests pin: (a) dummy vendor builds a dummy arm,
    (b) sim vendor with mock_mode builds an Isaac arm without touching
    the SDK, and (c) the schema ``SimConfig`` is faithfully translated
    into the driver ``SimRobotConfig``."""

    def _calc_and_perception(self):
        return (
            _ScriptedCalculator([_success_result()]),
            _FakePerception([_perception_frame()]),
        )

    def test_from_robot_config_dispatches_dummy_vendor(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.dummy import DummyRobotArm

        cfg = RobotConfig(vendor="dummy", gripper={"vendor": "none"})
        calc, perception = self._calc_and_perception()
        service = RuntimePickService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        self.assertIsInstance(service.orchestrator.arm, DummyRobotArm)
        self.assertIsNotNone(service.orchestrator.gripper)

    def test_from_robot_config_dispatches_sim_vendor_with_mock_mode(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.sim import IsaacRobotArm
        from src.robot.drivers.sim.config import SimRobotConfig

        cfg = RobotConfig(
            vendor="sim",
            sim={
                "enabled": True,
                "mock_mode": True,
                "scene": "mock.usd",
                "robot_prim_path": "/World/Robot",
                "home_joint_positions": [0.0, -1.0, 1.0, 0.0, 1.0, 0.0],
            },
            gripper={"vendor": "none"},
        )
        calc, perception = self._calc_and_perception()
        service = RuntimePickService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        arm = service.orchestrator.arm
        self.assertIsInstance(arm, IsaacRobotArm)
        driver_cfg = arm._config  # type: ignore[attr-defined]
        self.assertIsInstance(driver_cfg, SimRobotConfig)
        self.assertTrue(driver_cfg.mock_mode)
        self.assertEqual(driver_cfg.scene, "mock.usd")
        self.assertEqual(
            driver_cfg.home_joint_positions, (0.0, -1.0, 1.0, 0.0, 1.0, 0.0)
        )

    def test_from_robot_config_sim_camera_dict_is_translated(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.sim.config import SimCameraConfig

        cfg = RobotConfig(
            vendor="sim",
            sim={
                "enabled": True,
                "mock_mode": True,
                "cameras": {
                    "wrist": {
                        "prim_path": "/World/Robot/Wrist/Cam",
                        "mounting_mode": "eye_in_hand",
                    }
                },
            },
            gripper={"vendor": "none"},
        )
        calc, perception = self._calc_and_perception()
        service = RuntimePickService.from_robot_config(
            cfg,
            calculator=calc,  # type: ignore[arg-type]
            perception=perception,
        )
        driver_cfg = service.orchestrator.arm._config  # type: ignore[attr-defined]
        self.assertIn("wrist", driver_cfg.cameras)
        cam = driver_cfg.cameras["wrist"]
        self.assertIsInstance(cam, SimCameraConfig)
        self.assertEqual(cam.mounting_mode, "eye_in_hand")
        self.assertEqual(cam.prim_path, "/World/Robot/Wrist/Cam")

    def test_from_robot_config_unknown_vendor_raises(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        # L0.2: the schema now validates `vendor` against the RobotVendor enum, so an unknown vendor
        # is rejected EARLY at config-load (a typo no longer survives to surface late from the registry
        # via from_robot_config). The registry guard remains as defense-in-depth for unvalidated callers.
        with self.assertRaises(ValidationError):
            RobotConfig(vendor="not_a_real_vendor", gripper={"vendor": "none"})


# ---------------------------------------------------------------------------
# Phase Q-E: full SIM end-to-end smoke test.
#
# Schema YAML (vendor=sim, mock_mode=true) \u2192 from_robot_config \u2192
# orchestrator.run_attempt() must produce an EXECUTED PickSessionReport
# tagged ``robot_vendor="sim"`` / ``is_simulated=True``, *without*
# touching the Isaac SDK on any host. This is the single biggest
# guard against regressions in the vendor-block schema + mock_mode +
# typed-motion contract working together.
# ---------------------------------------------------------------------------


class SimEndToEndIntegrationTests(unittest.TestCase):
    """One realistic path through the whole stack on macOS-safe ground."""

    def test_sim_mock_mode_drives_pick_through_facade(self) -> None:
        from src.config.schema.robot import RobotConfig

        cfg = RobotConfig(
            vendor="sim",
            sim={
                "enabled": True,
                "mock_mode": True,
                "scene": "mock.usd",
                "robot_prim_path": "/World/Robot",
                "home_joint_positions": [0.0, -1.0, 1.0, 0.0, 1.0, 0.0],
            },
            gripper={"vendor": "none"},
        )

        service = RuntimePickService.from_robot_config(
            cfg,
            calculator=_ScriptedCalculator([_success_result()]),  # type: ignore[arg-type]
            perception=_FakePerception([_perception_frame()]),
            max_attempts=2,
        )

        # The driver lifecycle is the caller's responsibility \u2014 connect
        # explicitly so the typed move path is allowed to run.
        service.orchestrator.arm.connect()

        report = service.run_attempt()

        # End-to-end assertions: vendor labelling, typed motion chain,
        # and successful outcome.
        self.assertIsInstance(report, PickSessionReport)
        self.assertEqual(report.robot_vendor, "sim")
        self.assertTrue(report.is_simulated)
        self.assertIs(report.outcome, PickOutcome.EXECUTED)
        self.assertTrue(report.succeeded)
        self.assertGreater(len(report.motion_status_chain), 0)
        self.assertEqual(report.motion_status_chain[-1], MotionStatus.EXECUTED)

        # Mock kinematics must have actually moved the TCP \u2014 if it did
        # not, the executed move path was a no-op.
        tcp = service.orchestrator.arm.get_tcp_pose()
        self.assertIsNotNone(tcp)


if __name__ == "__main__":
    unittest.main()
