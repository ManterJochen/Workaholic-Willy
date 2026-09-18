"""A robot built from a loaded tree, able to describe itself, check its tree at a desk, and be imported by its package.

A noun that spans sections takes the loaded tree once (``Robot.from_tree``). A robot and a motion result print
themselves (``render()``, ``to_dict()``), and a robot runs the real cell's desk checklist for its own tree
(``preflight()``). ``src.robot.execution`` exports ``Robot``, ``Cell``, ``PickRun``, ``PlannerStart`` and the
refusals.
"""

from __future__ import annotations

import json
import unittest
from importlib import import_module

import numpy as np

from src.config import ConfigTree
from src.config.loader import ConfigError
from src.config.schema.robot import RobotConfig
from src.config.tree import LoadedTree
from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionResult, MotionStatus
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight
from src.robot.execution.robot import Robot
from src.robot.grippers.dummy import DummyGripper
from tests.test_camera_world_on_every_driver import _ur_ik
from tests.test_robot_parts import _loaded_after


def _console_dummy() -> LoadedTree:
    loaded = ConfigTree.from_directory(profile="console_dummy").load()
    assert loaded.ok, loaded.error
    return loaded


def _pose() -> Pose:
    return Pose(position_mm=np.array([400.0, 0.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                frame=Frame.BASE)


# ---------------------------------------------------------------------------------------------------
# MotionResult: both halves, no field changed
# ---------------------------------------------------------------------------------------------------


class AMotionResultDescribesItselfTests(unittest.TestCase):

    def test_render_names_the_status_the_target_and_the_camera_world(self) -> None:
        result = MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO, target_pose=_pose(),
                                     message="lfinger|fixture:seen_00: mesh distance 0.4 µm")

        text = result.render()

        self.assertTrue(text.isascii())
        self.assertFalse(text.endswith("\n"))
        self.assertTrue(text.startswith("move_to  SELF_COLLISION_REJECTED: lfinger|fixture:seen_00"), text)
        self.assertIn("\\xb5m", text, "a character outside ASCII is written as its escape")
        self.assertIn("target pose  (400.0, 0.0, 300.0) mm in base", text)
        self.assertIn("camera world  UNSTATED", text)

    def test_to_dict_is_a_view_of_the_fields(self) -> None:
        declined = CameraWorldStamp.declined(CameraWorldDecline("bench"))
        joints = JointPositions([0.0, -1.5, 1.5, 0.0, 1.5, 0.0])
        result = MotionResult.executed(MotionCommand.MOVE_JOINTS, target_joints=joints, message="move_to_joints",
                                       camera_world=declined)

        data = json.loads(json.dumps(result.to_dict()))

        self.assertEqual({
            "status": "executed", "command": "move_joints", "ok": True, "target_pose": None,
            "target_joints": [0.0, -1.5, 1.5, 0.0, 1.5, 0.0], "message": "move_to_joints", "exception": None,
            "camera_world": declined.to_dict(),
        }, data)
        self.assertIn("target joints  [0.0000, -1.5000, 1.5000, 0.0000, 1.5000, 0.0000] rad", result.render())

    def test_an_exception_is_named_and_stays_on_the_object(self) -> None:
        exc = ConnectionError("rtde link down")
        result = MotionResult.failed(MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO, message="link down",
                                     exception=exc)

        self.assertEqual("ConnectionError: rtde link down", result.to_dict()["exception"])
        self.assertIn("fault  ConnectionError: rtde link down", result.render())
        self.assertIs(exc, result.exception)
        self.assertEqual(result, MotionResult.failed(MotionStatus.CONNECTION_ERROR, MotionCommand.MOVE_TO,
                                                     message="link down"), "equality still ignores the exception")


# ---------------------------------------------------------------------------------------------------
# Robot.from_tree
# ---------------------------------------------------------------------------------------------------


class ARobotFromALoadedTreeTests(unittest.TestCase):

    def test_a_loaded_tree_builds_its_robot_and_keeps_the_tree(self) -> None:
        loaded = _console_dummy()

        robot = Robot.from_tree(loaded)

        self.assertIsInstance(robot.arm, DummyRobotArm)
        self.assertIsInstance(robot.gripper, DummyGripper)
        self.assertIsNone(robot.lock_key)
        self.assertIs(loaded, robot.tree)
        self.assertIs(loaded.config.robot, robot.robot_config)

    def test_the_gripper_choice_passes_through(self) -> None:
        self.assertIsNone(Robot.from_tree(_console_dummy(), gripper=None).gripper)

    def test_a_tree_that_did_not_load_is_refused_with_its_refusal(self) -> None:
        failed = LoadedTree(tree=ConfigTree.from_directory(profile=None), error="robot.gripper.model is unset")

        with self.assertRaises(ConfigError) as caught:
            Robot.from_tree(failed)

        self.assertIn("robot.gripper.model is unset", str(caught.exception))

    def test_anything_but_a_loaded_tree_is_a_type_error(self) -> None:
        with self.assertRaisesRegex(TypeError, r"load\(\)"):
            Robot.from_tree(_console_dummy().config)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------------
# Robot.preflight
# ---------------------------------------------------------------------------------------------------


class ARobotChecksItsTreeAtADeskTests(unittest.TestCase):

    def test_the_checklist_reads_both_halves_of_the_tree_the_robot_came_from(self) -> None:
        loaded = _console_dummy()

        report = Robot.from_tree(loaded).preflight()

        self.assertEqual(run_config_preflight(loaded.config.robot, camera=loaded.config.camera).to_dict(),
                         report.to_dict())

    def test_a_robot_from_a_section_alone_reads_no_other_tree(self) -> None:
        section = RobotConfig(vendor="dummy", gripper={"vendor": "dummy"})

        report = Robot.from_config(section).preflight()

        self.assertEqual(run_config_preflight(section).to_dict(), report.to_dict())

    def test_a_robot_that_keeps_no_tree_gets_one_block_row(self) -> None:
        report = Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None).preflight()

        self.assertEqual(["config"], [check.name for check in report.checks])
        self.assertIs(CheckStatus.BLOCK, report.checks[0].status)
        self.assertEqual(1, report.exit_code)
        self.assertIn("Robot.from_tree", report.checks[0].fix)


# ---------------------------------------------------------------------------------------------------
# Robot.render and Robot.to_dict
# ---------------------------------------------------------------------------------------------------


class ARobotDescribesItselfTests(unittest.TestCase):

    def test_render_names_arm_gripper_lock_planner_camera_world_and_tree(self) -> None:
        tree = _console_dummy()
        text = Robot.from_tree(tree).render()

        self.assertTrue(text.isascii())
        self.assertFalse(text.endswith("\n"))
        lines = text.split("\n")
        self.assertEqual("robot  DummyRobotArm (dummy, dummy-6dof)", lines[0])
        self.assertEqual("gripper  DummyGripper", lines[1])
        self.assertEqual("lock  none: this robot takes no lock", lines[2])
        self.assertEqual("safety  UNGATED", lines[3])
        self.assertTrue(lines[4].startswith("planner  UNPLANNED: a desk arm"), lines[4])
        self.assertTrue(lines[4].endswith("(line TELEPORT)"), lines[4])
        self.assertEqual("camera world  not handed any camera", lines[5])
        self.assertEqual("wrist cameras  not handed any camera", lines[6])
        self.assertEqual(f"tree  console_dummy under {tree.root}", lines[7])

    def test_to_dict_carries_the_same_readings(self) -> None:
        tree = _console_dummy()
        robot = Robot.from_tree(tree)

        data = json.loads(json.dumps(robot.to_dict()))

        self.assertEqual("DummyRobotArm", data["arm"])
        self.assertEqual("DummyGripper", data["gripper"])
        self.assertIsNone(data["lock_key"])
        self.assertEqual("ungated", data["safety"])
        self.assertEqual("unplanned", data["route"]["route"])
        self.assertEqual("teleport", data["route"]["line"]["motion"])
        self.assertEqual({"needs_decline": False, "wiring": None}, data["camera_world"])
        self.assertIsNone(data["wrist_bodies"])
        self.assertEqual({"chain": "console_dummy", "root": str(tree.root)}, data["tree"])

    def test_a_ur_on_ik_says_its_route_is_refused(self) -> None:
        text = Robot.from_parts(arm=_ur_ik(), gripper=None, lock_key=None).render()

        self.assertIn("planner  REFUSED: nothing plans", text)
        self.assertIn("'ik'", text)
        self.assertIn("tree  none kept: built around handles", text)


# ---------------------------------------------------------------------------------------------------
# The package exports
# ---------------------------------------------------------------------------------------------------

_SURFACE = (
    ("Robot", "src.robot.execution.robot"),
    ("Cell", "src.robot.execution.cell"),
    ("PickRun", "src.robot.execution.pick_run"),
    ("PlannerStart", "src.robot.execution.planner_start"),
    ("PlannerStartReport", "src.robot.execution.planner_start"),
    ("CameraWorldRequired", "src.robot.execution.camera_world_wiring"),
    ("CellBusy", "src.robot.execution.cell_lock"),
    ("CellNotBuilt", "src.robot.execution.cell"),
    ("LockKeyRequired", "src.robot.execution.robot"),
    ("NoRealGripper", "src.robot.execution.lifecycle"),
    ("WristBodyRequired", "src.robot.execution.wrist_bodies"),
    ("MotionReport", "src.robot.execution.motion"),
    ("MotionOutcome", "src.robot.execution.motion"),
    ("MotionRoute", "src.robot.execution.motion"),
    ("MotionVerb", "src.robot.execution.motion"),
    ("RouteReading", "src.robot.execution.motion"),
    ("GraspMotion", "src.robot.grasping.motion.grasp_motion"),
    ("HandEyeCalibration", "src.robot.execution.hand_eye"),
    ("SweepOptions", "src.robot.execution.hand_eye"),
    ("CalibrationRunReport", "src.robot.execution.hand_eye"),
)


class ThePackageExportsTheSurfaceTests(unittest.TestCase):

    def test_every_name_is_exported_and_is_the_class_its_module_defines(self) -> None:
        import src.robot.execution as execution

        for name, module in _SURFACE:
            with self.subTest(name):
                self.assertIn(name, execution.__all__)
                self.assertIs(getattr(import_module(module), name), getattr(execution, name))

    def test_importing_the_package_loads_none_of_them(self) -> None:
        """The control: the exports stay lazy, so the package import pulls no cell or driver."""
        loaded = _loaded_after("import src.robot.execution", (
            "src.robot.execution.cell", "src.robot.execution.planner_start",
            "src.robot.execution.motion", "src.robot.drivers.ur",
            "src.robot.execution.hand_eye", "src.robot.grasping.motion.grasp_motion",
        ))
        self.assertEqual([], loaded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
