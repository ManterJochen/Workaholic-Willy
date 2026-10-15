"""A calibration sweep hands a robot only the wrist bodies it lacks (43fe699).

A robot handed in through ``from_parts`` that already carries the bodies the sweep would hand over is not handed them
again: an arm whose exact mesh guard is built refuses any hand-over, the same bodies included. A body it does not hold
is handed beside the ones it holds, never in their place, and refused where the guard is built.

The gap, before this: an eye in hand sweep handed its camera's body to the arm with ``WristBodies.hand_to``, which
replaces what the arm holds. A robot ``Robot.from_tree`` built carries every body its tree declares, so the sweep of
one wrist camera dropped the other's from the planner and the guard, and a robot whose guard had judged a path was
refused the build for being handed the body it held already.

Everything below the arm is real: the UR driver object (never connected) holds the bodies its guard and its planner
read. The camera is a double, and nothing moves: a sweep is run with ``dry_run=True``.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np

from src.config import ConfigTree
from src.config.schema.camera import HandEyeConfig
from src.config.tree import LoadedTree
from src.calibration.serialization import FlangeToTcp, save_cam_to_tool
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.hand_eye import CalibrationOutcome, CalibrationRunReport, HandEyeCalibration, SweepOptions
from src.robot.execution.robot import Robot
from src.geometry import Pose
from tests._wrist_body import camera_to_tool, declared_frame, wrist_rig
from tests._wrist_body import wrist_cell
from tests.test_camera_boundaries import _rgbd_rig

_CAMERA = "src.camera.orchestration.camera.Camera"
_STATIONS = [Pose.tool_down(400.0, 0.0, 350.0, label="down_0"), Pose.tool_down(420.0, 60.0, 380.0, label="down_1")]
_GUARD_BUILT = "the exact mesh guard was already built, so a wrist camera handed in now would not be in it"


class _Handle:
    """The camera handle a build prints and the marker source reads."""

    def __repr__(self) -> str:
        return "RigHandle('overhead')"

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def release(self) -> None:
        pass


class _Sweeps(unittest.TestCase):
    """A wrist cell (cuRobo, a declared ``willy`` tool frame) and its camera section, swept on a real UR arm object."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _wrist(self, rig_id: str = "wrist", *, calibrated: bool = True, artifact: bool = True,
               **changes: Any) -> SimpleNamespace:
        """A wrist rig; calibrated, its artifact is written with the declared frame, unless ``artifact`` is False."""
        rig = wrist_rig(rig_id, calibrated=calibrated, **changes)
        if rig.extrinsics is not None:
            path = self.root / f"eih_{rig_id}.json"
            if artifact:
                save_cam_to_tool(path, camera_to_tool(), rig_id=rig_id,
                                 flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame()))
            rig.extrinsics.artifact_path = str(path)
        return rig

    def _app(self, *wrist: SimpleNamespace, robot: Any = None) -> SimpleNamespace:
        """The two sections a sweep reads: the fixed camera ``overhead`` and the wrist rigs handed in."""
        return SimpleNamespace(
            robot=robot if robot is not None else wrist_cell(),
            camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead"), *wrist],
                                                           primary_rig_id="overhead"),
                                   hand_eye=HandEyeConfig()),
        )

    @staticmethod
    def _held(robot: Robot) -> list[str]:
        """The wrist bodies the arm's guard and planner read."""
        assert isinstance(robot.arm, URRobotArm)
        return [body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)]  # noqa: SLF001


class ARobotHandedInThatCarriesTheBodiesTests(_Sweeps):
    """``from_parts`` over a robot ``Robot.from_tree`` built, which already hands its arm every declared body."""

    def _robot(self, app: SimpleNamespace) -> Robot:
        return Robot.from_tree(LoadedTree(tree=ConfigTree.from_directory(profile=None), config=app), gripper=None)

    @staticmethod
    def _build_the_guard(robot: Robot) -> None:
        """Mark the exact mesh guard built, as its first judged path leaves it, and prove the arm now refuses."""
        guard = robot.arm._preflight._path_authority(robot.arm)  # type: ignore[attr-defined]  # noqa: SLF001
        guard._fcl_backend_built = True  # noqa: SLF001
        held = robot.arm._preflight.wrist_bodies(robot.arm)  # type: ignore[attr-defined]  # noqa: SLF001
        try:
            robot.arm.set_wrist_bodies(held)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            assert _GUARD_BUILT in str(exc), exc
        else:  # pragma: no cover - the precondition
            raise AssertionError("the guard took a hand-over after it was built")

    def _run(self, robot: Robot, section: Any, **kwargs: Any) -> CalibrationRunReport:
        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value = _Handle()
        calibration = HandEyeCalibration.from_parts(
            robot=robot, camera_config=section, rig_id=kwargs.pop("rig_id", "wrist"),
            mode=kwargs.pop("mode", "eye_in_hand"), options=SweepOptions(fixed_poses=_STATIONS))
        with patch(_CAMERA, camera_cls):
            return calibration.run(dry_run=True)

    def test_an_eye_in_hand_sweep_of_the_camera_it_holds_is_not_handed_it_again(self) -> None:
        """Before, the sweep handed the arm the body it held, and the built guard refused the build."""
        app = self._app(self._wrist(), self._wrist("side"))
        robot = self._robot(app)
        self._build_the_guard(robot)
        report = self._run(robot, app.camera)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(sorted(self._held(robot)), ["wrist_camera_side", "wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist camera wrist_camera_wrist: realsense_d435", report.build.wrist)

    def test_the_swept_camera_is_handed_beside_the_bodies_the_arm_holds(self) -> None:
        """Before, the hand-over replaced what the arm held: the sweep of 'wrist' dropped 'side' from the planner
        and the guard of a robot that carried it."""
        robot = self._robot(self._app(self._wrist("side")))
        self.assertEqual(self._held(robot), ["wrist_camera_side"])
        with patch.object(URRobotArm, "set_wrist_bodies", autospec=True,
                          side_effect=URRobotArm.set_wrist_bodies) as handed:
            report = self._run(robot, self._app(self._wrist(), self._wrist("side")).camera)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(handed.call_count, 1)
        self.assertEqual(sorted(self._held(robot)), ["wrist_camera_side", "wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist_camera_side", report.build.wrist)
        self.assertIn("wrist_camera_wrist", report.build.wrist)

    def test_a_different_body_on_a_built_guard_still_fails_closed(self) -> None:
        robot = self._robot(self._app(self._wrist()))
        self._build_the_guard(robot)
        report = self._run(robot, self._app(self._wrist(margin_mm=9.0)).camera)
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED, report.render())
        assert report.build is not None
        self.assertIn(_GUARD_BUILT, report.build.refusal)
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])

    def test_a_missing_body_on_a_built_guard_fails_closed(self) -> None:
        robot = self._robot(self._app(self._wrist("side")))
        self._build_the_guard(robot)
        report = self._run(robot, self._app(self._wrist(), self._wrist("side")).camera)
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED, report.render())
        assert report.build is not None
        self.assertIn(_GUARD_BUILT, report.build.refusal)
        self.assertEqual(self._held(robot), ["wrist_camera_side"])

    def test_a_robot_swept_twice_is_not_handed_the_same_body_again(self) -> None:
        """Found on 2026-09-25 (af2caa2): the robot's record of its bodies is set when it is built, so the first
        sweep's hand-over was not in it, and a second sweep over a built guard was refused for handing the same body
        again. What the arm's guard holds counts too."""
        app = self._app(self._wrist())
        robot = Robot.from_config(app.robot, gripper=None)
        first = self._run(robot, app.camera)
        self.assertIs(first.outcome, CalibrationOutcome.DRY_RUN, first.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        self._build_the_guard(robot)
        second = self._run(robot, app.camera)
        self.assertIs(second.outcome, CalibrationOutcome.DRY_RUN, second.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])

    def test_the_arm_the_sweep_builds_itself_is_handed_its_camera_as_before(self) -> None:
        """The control: a robot that holds nothing is handed the swept camera's body once."""
        app = self._app(self._wrist())
        robot = Robot.from_config(app.robot, gripper=None)
        with patch.object(URRobotArm, "set_wrist_bodies", autospec=True,
                          side_effect=URRobotArm.set_wrist_bodies) as handed:
            report = self._run(robot, app.camera)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(handed.call_count, 1)
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
