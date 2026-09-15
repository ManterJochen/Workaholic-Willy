"""Calibration declines the camera world for itself, and its CLI connects the arm through ``Robot``.

A hand-eye sweep is what produces the transform a camera world is built from, so the sweep cannot plan
against one. The routine says so on every motion it commands: its moves run inside a decline bound to its
own arm, with one reason per mounting unless the caller states another, and the stamps come back on
``CalibrationResult``.

The real-cell calibration CLI builds the arm with ``Robot.from_config(robot_config, gripper=None)``. The
readiness gate runs, the cell lock is taken before the arm is commanded, the gripper is never built or
connected, and a refused connect exits with the configuration exit code.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

from src.config.schema.camera import HandEyeConfig
from src.config.schema.robot import RobotConfig, WorkspaceLimitsConfig
from src.geometry import Frame, Pose
from src.robot.core import (
    MotionCommand,
    MotionResult,
    MotionStatus,
    RobotConnectionError,
)
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp, CameraWorldUse
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.cell_lock import CellLock, lock_path_for, peek
from src.robot.execution.lifecycle import StepOutcome, TeardownReport
from src.robot.execution.real_cell import calibrate
from src.robot.safety import SafetyAttestation, SafetyPreflight
from tests.test_camera_boundaries import _rgbd_rig
from tests.test_robot_boundaries import (
    _inverse,
    _synthetic_eye_in_hand_data,
    _synthetic_eye_to_hand_data,
)
from tests.test_ungated_motion_surfaces import _AcceptingGuard
from tests.test_ur_arm import _FakePlanner
from tests.test_ur_arm import _arm as _ur_arm

_ETH_REASON = (
    "eye-to-hand calibration: this sweep produces the CAMERA to BASE transform a camera world needs"
)
_EIH_REASON = "eye-in-hand calibration: the CAMERA to TOOL transform is being solved"

_WIDE = WorkspaceLimitsConfig(
    x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0, z_max=2000.0,
)


class _PlannerThatArrives(_FakePlanner):
    """Plans one waypoint and arrives at the commanded pose, so the arm reads back where it went."""

    def __init__(self, plan_result: list[list[float]]) -> None:
        super().__init__(plan_result=plan_result)
        self.at: Pose | None = None

    def execute(self, traj: object, pose: Pose, *, vel: object = None, acc: object = None) -> MotionResult:
        self.at = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)


def _curobo_ur(*, plans: bool = True) -> tuple[URRobotArm, list[MotionResult]]:
    """A UR arm on cuRobo with no live camera world, its planner a double, and every move result kept."""
    arm = _ur_arm("curobo")
    arm._preflight = SafetyPreflight([_AcceptingGuard("workspace")])
    planner = _PlannerThatArrives([[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]] if plans else [])
    arm._curobo_ur = planner  # type: ignore[assignment]
    arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
    # The path gate too. This file is about what a calibration sweep says about the camera world,
    # and the preflight above holds one accepting stand-in rather than a self collision guard,
    # which a judged path refuses outright (exact meshes on every sample, or no motion).
    # tests/test_planned_paths_are_judged.py holds that half.
    arm._preflight.gate_planned_path = (  # type: ignore[method-assign]
        lambda waypoints, *, arm=None, command=None: None
    )
    arm.get_tcp_pose = lambda: planner.at  # type: ignore[method-assign, assignment, return-value]
    seen: list[MotionResult] = []
    move = arm.move

    def recording(pose: Pose, **keywords: Any) -> MotionResult:
        result = move(pose, **keywords)
        seen.append(result)
        return result

    arm.move = recording  # type: ignore[method-assign]
    return arm, seen


def _settings(mode: str) -> SimpleNamespace:
    return SimpleNamespace(mode=mode, min_samples=4, min_distance_mm=0.0, min_angle=0.0,
                           min_angle_deg=0.0)


def _eye_to_hand(arm: object, **keywords: Any) -> tuple[CalibrationRoutine, list[Pose]]:
    """A routine over ``arm`` whose marker source agrees with a solvable eye-to-hand dataset."""
    T_cam_to_base, T_tool_to_marker, tool_poses = _synthetic_eye_to_hand_data()
    poses = [Pose.from_matrix(T, frame=Frame.BASE, label=f"pose_{i}") for i, T in enumerate(tool_poses)]
    markers = iter(_inverse(T_cam_to_base) @ T @ T_tool_to_marker for T in tool_poses)
    routine = CalibrationRoutine(
        arm=arm, marker_source=lambda: next(markers), workspace_limits=_WIDE,  # type: ignore[arg-type]
        eth_settings=_settings("eye_to_hand"), calibration_mode="eye_to_hand", settle_time_s=0.0,
        **keywords,
    )
    return routine, poses


def _eye_in_hand(arm: object) -> tuple[CalibrationRoutine, list[Pose]]:
    T_cam_to_tool, T_base_to_marker, tool_poses = _synthetic_eye_in_hand_data()
    poses = [Pose.from_matrix(T, frame=Frame.BASE, label=f"pose_{i}") for i, T in enumerate(tool_poses)]
    markers = iter(_inverse(T_cam_to_tool) @ _inverse(T) @ T_base_to_marker for T in tool_poses)
    routine = CalibrationRoutine(
        arm=arm, marker_source=lambda: next(markers), workspace_limits=_WIDE,  # type: ignore[arg-type]
        eth_settings=_settings("eye_in_hand"), calibration_mode="eye_in_hand", settle_time_s=0.0,
    )
    return routine, poses


# ---------------------------------------------------------------------------------------------------
# The routine
# ---------------------------------------------------------------------------------------------------


class TheRoutineDeclinesTests(unittest.TestCase):

    def test_every_move_of_an_eye_to_hand_sweep_is_declined_for_the_mounting(self) -> None:
        arm, seen = _curobo_ur()
        routine, poses = _eye_to_hand(arm)
        result = routine.run_with_poses(poses)
        declined = CameraWorldStamp.declined(CameraWorldDecline(_ETH_REASON))
        self.assertEqual([r.status for r in seen], [MotionStatus.EXECUTED] * len(poses))
        self.assertEqual([r.camera_world for r in seen], [declined] * len(poses))
        self.assertEqual(result.camera_worlds, (declined,) * len(poses))
        self.assertEqual(result.num_samples, len(poses), "the declined sweep did not solve")
        # The decline belongs to the sweep: the same arm moved outside it plans as it would without it.
        self.assertIs(arm.move(poses[0]).camera_world.use, CameraWorldUse.MISSING)

    def test_an_eye_in_hand_sweep_declines_with_its_own_reason(self) -> None:
        arm, seen = _curobo_ur()
        routine, poses = _eye_in_hand(arm)
        result = routine.run_with_poses(poses)
        declined = CameraWorldStamp.declined(CameraWorldDecline(_EIH_REASON))
        self.assertEqual([r.camera_world for r in seen], [declined] * len(poses))
        self.assertEqual(result.camera_worlds, (declined,) * len(poses))

    def test_a_decline_the_caller_states_replaces_the_mounting_reason(self) -> None:
        arm, seen = _curobo_ur()
        bench = CameraWorldDecline("bench rehearsal of the sweep, the cameras are switched off")
        routine, poses = _eye_to_hand(arm, camera_world=bench)
        result = routine.run_with_poses(poses)
        self.assertEqual({r.camera_world for r in seen}, {CameraWorldStamp.declined(bench)})
        self.assertEqual(result.camera_worlds, (CameraWorldStamp.declined(bench),) * len(poses))

    def test_a_rejected_move_is_logged_with_its_stamp(self) -> None:
        arm, _ = _curobo_ur(plans=False)
        routine, poses = _eye_to_hand(arm)
        with self.assertLogs(routine.logger, level="WARNING") as logs:
            self.assertFalse(routine._move_to_pose(poses[0]))
        stamp = CameraWorldStamp.declined(CameraWorldDecline(_ETH_REASON)).render()
        self.assertTrue(any("rejected" in line and stamp in line for line in logs.output), logs.output)

    def test_a_decline_is_a_decline_object_not_a_reason_or_none(self) -> None:
        for value in ("bench", None):
            with self.subTest(value=value), self.assertRaisesRegex(TypeError, "CameraWorldDecline"):
                _eye_to_hand(DummyRobotArm(), camera_world=value)


# ---------------------------------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------------------------------

_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"
_CREATE_ARM = "src.robot.drivers.create_arm"
_CAMERA = "src.camera.orchestration.camera.Camera"


def _tree(ip: str) -> SimpleNamespace:
    """The two sections the CLI reads: a real UR robot section and one enabled RGB-D rig."""
    return SimpleNamespace(
        robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": ip}}),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]),
                               hand_eye=HandEyeConfig()),
    )


class TheCliConnectsTheArmThroughRobotTests(unittest.TestCase):
    IP = "10.253.253.7"
    KEY = f"ur@{IP}"

    def setUp(self) -> None:
        self._out = tempfile.TemporaryDirectory()
        #: What came down, in the order it came down, where a test records it.
        self.order: list[str] = []

    def tearDown(self) -> None:
        self._out.cleanup()
        lock_path_for(self.KEY).unlink(missing_ok=True)

    def _run(
        self, *argv: str, arm: DummyRobotArm, sweep: BaseException = AssertionError("the sweep ran"),
    ) -> tuple[int, str, MagicMock, MagicMock]:
        """``main`` over the tree, with the gate, the arm factory, the camera and the sweep patched.

        ``sweep`` is what ``run_auto`` raises. Returns the exit code, what was printed, the arm factory
        and the camera handle, whose release is recorded on ``self.order``.
        """
        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value.release.side_effect = (
            lambda: self.order.append("camera released"))
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=_tree(self.IP)), patch(_READY), \
                patch(_CREATE_ARM, return_value=arm) as create_arm, patch(_CAMERA, camera_cls), \
                patch.object(CalibrationRoutine, "run_auto", side_effect=sweep), \
                redirect_stdout(printed):
            code = calibrate.main(["--rig", "overhead", "--out", self._out.name, *argv])
        return code, printed.getvalue(), create_arm, camera_cls.from_config.return_value.handle.return_value

    def test_a_held_cell_exits_config_and_names_the_holder(self) -> None:
        arm = DummyRobotArm()
        arm.connect = MagicMock(side_effect=AssertionError("the arm was commanded"))  # type: ignore[method-assign]
        with CellLock(self.KEY, owner="operator console"):
            code, printed, _, camera = self._run(arm=arm)
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertIn("operator console", printed)
        arm.connect.assert_not_called()
        camera.release.assert_called_once_with()

    def test_a_refused_connect_exits_config_and_gives_the_lock_back(self) -> None:
        arm = DummyRobotArm()
        arm.connect = MagicMock(  # type: ignore[method-assign]
            side_effect=RobotConnectionError("the controller refused the control script"))
        code, printed, _, camera = self._run(arm=arm)
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertIn("RobotConnectionError: the controller refused the control script", printed)
        arm.connect.assert_called_once_with()
        self.assertIsNone(peek(self.KEY), "the refused connect kept the cell lock")
        camera.release.assert_called_once_with()

    def test_a_sweep_that_raises_exits_error_once_the_arm_is_down(self) -> None:
        """A sweep that fails is not a refused connect, so it keeps exit 3. It is reported after the
        teardown, which reads the arm alone, and the camera is released after the arm came down."""
        arm = DummyRobotArm()
        disconnect = arm.disconnect

        def recording_disconnect() -> None:
            self.order.append("arm down")
            disconnect()

        arm.disconnect = recording_disconnect  # type: ignore[method-assign]
        code, printed, _, _ = self._run(arm=arm, sweep=RuntimeError("the marker left the frame"))
        self.assertEqual(code, calibrate._EXIT_ERROR)
        teardown = TeardownReport(StepOutcome.ABSENT, StepOutcome.RELEASED, StepOutcome.ABSENT).render()
        self.assertIn(teardown, printed)
        self.assertLess(printed.index(teardown),
                        printed.index("[sweep] FAILED: RuntimeError: the marker left the frame"))
        self.assertEqual(self.order, ["arm down", "camera released"])
        self.assertIsNone(peek(self.KEY), "the sweep kept the cell lock")

    def test_a_dry_run_prints_the_attestation_and_the_lock_and_connects_nothing(self) -> None:
        arm = DummyRobotArm()
        arm.connect = MagicMock(side_effect=AssertionError("--dry-run connected the arm"))  # type: ignore[method-assign]
        code, printed, create_arm, camera = self._run("--dry-run", arm=arm)
        self.assertEqual(code, calibrate._EXIT_OK)
        create_arm.assert_called_once()
        self.assertIn(SafetyAttestation.of(arm).render(), printed)
        self.assertIn(f"  lock     {self.KEY}", printed)
        self.assertIn("  gripper  none", printed)
        arm.connect.assert_not_called()
        self.assertIsNone(peek(self.KEY), "a dry run took the cell lock")
        camera.release.assert_called_once_with()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
