"""A robot without a pick service: the arm, the gripper on it, and the one way to connect them.

``Cell`` builds a whole pick service before it can connect anything, while calibration, a bench check or
a jog from Python needs only the arm and the gripper. ``Robot`` is that noun. It builds both handles
through the builder the service uses, takes the cross-process lock ``Cell`` takes, and connects through
the enter and exit ``ConnectedCell`` uses, so the order exists once for both.
"""

from __future__ import annotations

import socket
import unittest
from unittest.mock import patch

from src.config import load_robot_config
from src.config.schema.robot import RobotConfig
from src.robot.core import MotionCommand, MotionResult, RobotCapabilities
from src.robot.core.camera_world import CameraWorldDecline, CameraWorldStamp, CameraWorldUse
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.drivers.kuka.arm import KukaRobotArm
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.cell_lock import CellBusy, CellLock, lock_path_for
from src.robot.execution.lifecycle import (
    ConnectedCell,
    ConnectedRobot,
    NoRealGripper,
    StepOutcome,
)
from src.robot.execution.robot import LockKeyRequired, Robot
from src.robot.grippers import SubstitutionReason
from src.robot.grippers.dummy import DummyGripper
from src.robot.grippers.null import GripperSubstitution, NullGripper
from src.robot.grippers.robotiq import GripperController
from src.robot.safety import SafetyPosture
from tests.test_cell_lifecycle import _Recorder, _service
from tests.test_robot_parts import _BUILDER_MUST_NOT_LOAD, _loaded_after
from tests.test_ur_arm import _FakePlanner, _pose
from tests.test_ur_arm import _arm as _ur_arm

_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"


class _RecordingLock:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    def acquire(self) -> None:
        self.log.append("lock.acquire")

    def release(self) -> None:
        self.log.append("lock.release")


class _Reports:
    """An arm double that reports a vendor and carries no config tree, so no lock key derives."""

    def __init__(self, vendor: str) -> None:
        self.capabilities = RobotCapabilities(vendor=vendor, model="double")


#: The lock, the narration hook and both handles on one log, so the interleaving is the assertion.
_THE_ORDER = [
    "lock.acquire",
    "arm.connect",
    "<arm_connected>",
    "<gripper_moving>",
    "gripper.connect",
    "<gripper_connected>",
    "gripper.disconnect",
    "arm.disconnect",
    "lock.release",
]


def _shipped_tree_on_a_dummy_arm() -> RobotConfig:
    """The repository's base tree with the vendor moved to dummy, as a rehearsal moves it."""
    return load_robot_config(profile=None).model_copy(update={"vendor": "dummy"})


class OneEnterAndExitTests(unittest.TestCase):

    def test_a_connected_robot_takes_the_lock_then_the_arm_then_the_gripper(self) -> None:
        log: list[str] = []
        session = ConnectedRobot(_Recorder(log, "arm"), _Recorder(log, "gripper"),
                                 lock=_RecordingLock(log),
                                 announce=lambda stage: log.append(f"<{stage.value}>"))
        with session:
            pass
        self.assertEqual(log, _THE_ORDER)

    def test_a_connected_cell_runs_the_same_order(self) -> None:
        log: list[str] = []
        session = ConnectedCell(_service(_Recorder(log, "arm"), _Recorder(log, "gripper")),
                                lock=_RecordingLock(log),
                                announce=lambda stage: log.append(f"<{stage.value}>"))
        with session:
            pass
        self.assertEqual(log, _THE_ORDER)

    def test_a_substituted_gripper_is_refused_before_the_arm_moves(self) -> None:
        log: list[str] = []
        gripper = _Recorder(log, "gripper")
        gripper.substitution = GripperSubstitution(  # type: ignore[attr-defined]
            reason=SubstitutionReason.ROBOTIQ_NEEDS_UR, requested="robotiq",
            detail="gripper.vendor='robotiq' on an arm that is not a UR.",
            fix="Set robot.vendor: ur.",
        )
        session = ConnectedRobot(_Recorder(log, "arm"), gripper, lock=_RecordingLock(log))
        with self.assertRaises(NoRealGripper):
            session.__enter__()
        self.assertEqual(log, ["lock.acquire", "lock.release"])
        self.assertIsNone(session.teardown)

    def test_an_arm_only_robot_connects_the_arm_alone(self) -> None:
        log: list[str] = []
        robot = Robot.from_parts(arm=_Recorder(log, "arm"), gripper=None,  # type: ignore[arg-type]
                                 lock_key=None)
        with robot.connected() as live:
            self.assertIsNone(live.lock)
        self.assertEqual(log, ["arm.connect", "arm.disconnect"])
        assert live.teardown is not None
        self.assertIs(live.teardown.arm, StepOutcome.RELEASED)
        self.assertIs(live.teardown.gripper, StepOutcome.ABSENT)
        self.assertIs(live.teardown.perception, StepOutcome.ABSENT)


class FromConfigTests(unittest.TestCase):

    def test_a_dummy_tree_builds_a_dummy_arm_and_gripper_and_takes_no_lock(self) -> None:
        robot = Robot.from_config(RobotConfig(vendor="dummy", gripper={"vendor": "dummy"}))
        self.assertIsInstance(robot.arm, DummyRobotArm)
        self.assertIsInstance(robot.gripper, DummyGripper)
        self.assertIsNone(robot.lock_key)
        self.assertIs(robot.safety().posture, SafetyPosture.UNGATED)
        with robot.connected() as live:
            self.assertTrue(robot.arm.is_connected)
        assert live.teardown is not None
        self.assertTrue(live.teardown.clean)

    def test_gripper_none_builds_the_arm_alone(self) -> None:
        tree = RobotConfig(vendor="dummy", gripper={"vendor": "dummy"})
        self.assertIsNone(Robot.from_config(tree, gripper=None).gripper)
        self.assertIsInstance(Robot.from_config(tree).gripper, DummyGripper)

    def test_the_shipped_tree_on_a_dummy_arm_substitutes_and_refuses_to_connect(self) -> None:
        robot = Robot.from_config(_shipped_tree_on_a_dummy_arm())
        self.assertIsInstance(robot.gripper, NullGripper)
        substitution = getattr(robot.gripper, "substitution", None)
        assert substitution is not None, "the rehearsal shape recorded no substitution"
        self.assertIs(substitution.reason, SubstitutionReason.ROBOTIQ_NEEDS_UR)
        with self.assertRaises(NoRealGripper) as caught:
            robot.connected().__enter__()
        self.assertIn("close on nothing", str(caught.exception))

    def test_the_same_tree_without_its_gripper_connects(self) -> None:
        """A substituted gripper blocks a robot that holds it, and only that robot."""
        robot = Robot.from_config(_shipped_tree_on_a_dummy_arm(), gripper=None)
        with robot.connected() as live:
            self.assertTrue(robot.arm.is_connected)
        assert live.teardown is not None
        self.assertTrue(live.teardown.clean)

    def test_a_ur_tree_builds_the_driver_and_its_robotiq_and_opens_no_socket(self) -> None:
        tree = RobotConfig.model_validate(
            {"vendor": "ur", "ur": {"ip": "10.9.9.9"}, "gripper": {"vendor": "robotiq", "model": "robotiq_2f85"}})
        refuse = AssertionError("building a robot opened a socket")
        with patch(_READY), \
                patch.object(socket.socket, "connect", side_effect=refuse) as connect, \
                patch("socket.create_connection", side_effect=refuse) as create:
            robot = Robot.from_config(tree)
        self.assertIsInstance(robot.arm, URRobotArm)
        assert isinstance(robot.gripper, GripperController)
        self.assertEqual(robot.gripper.ip, "10.9.9.9")
        self.assertEqual(robot.lock_key, "ur@10.9.9.9")
        self.assertIs(robot.safety().posture, SafetyPosture.GATED)
        connect.assert_not_called()
        create.assert_not_called()


class RobotAndCellContendTests(unittest.TestCase):
    KEY = "ur@10.254.254.1"

    def tearDown(self) -> None:
        lock_path_for(self.KEY).unlink(missing_ok=True)

    def test_a_robot_is_refused_while_a_cell_holds_the_controller(self) -> None:
        tree = RobotConfig.model_validate(
            {"vendor": "ur", "ur": {"ip": "10.254.254.1"}, "gripper": {"vendor": "none", "model": "robotiq_2f85"}})
        with patch(_READY):
            robot = Robot.from_config(tree)
        with CellLock(self.KEY, owner="Cell"), \
                patch.object(URRobotArm, "connect",
                             side_effect=AssertionError("the arm was commanded")), \
                self.assertRaises(CellBusy) as caught:
            robot.connected().__enter__()
        assert caught.exception.holder is not None
        self.assertEqual(caught.exception.holder.owner, "Cell")
        self.assertIn("Cell", str(caught.exception))


class FromPartsTests(unittest.TestCase):

    def test_the_lock_key_comes_from_the_tree_the_arm_was_built_with(self) -> None:
        ur = URRobotArm(RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.9.9.9"}, "gripper": {"model": "robotiq_2f85"}}))
        kuka = KukaRobotArm(RobotConfig.model_validate(
            {"vendor": "kuka", "kuka": {"controller_ip": "10.8.8.8"}}))
        for arm, key in ((ur, "ur@10.9.9.9"), (kuka, "kuka@10.8.8.8"), (DummyRobotArm(), None),
                         (_Reports("sim"), None), (_Recorder([], "arm"), None)):
            with self.subTest(type(arm).__name__):
                self.assertEqual(
                    Robot.from_parts(arm=arm, gripper=None).lock_key,  # type: ignore[arg-type]
                    key,
                )

    def test_a_real_arm_with_no_derivable_key_is_refused(self) -> None:
        """A UR driver built from a tree naming another vendor, and an adapter that keeps no tree,
        both report ``ur`` and derive no key. Taking no lock on either would let a second process
        compete for the controller."""
        foreign_tree = URRobotArm(RobotConfig.model_validate({"vendor": "dummy", "gripper": {"model": "robotiq_2f85"}}))
        for arm in (foreign_tree, _Reports("ur")):
            with self.subTest(type(arm).__name__):
                with self.assertRaises(LockKeyRequired) as caught:
                    Robot.from_parts(arm=arm, gripper=None)  # type: ignore[arg-type]
                self.assertIn("lock_key", str(caught.exception))
                self.assertIn("'ur'", str(caught.exception))

    def test_a_stated_none_is_a_robot_that_takes_no_lock(self) -> None:
        robot = Robot.from_parts(arm=_Reports("ur"), gripper=None,  # type: ignore[arg-type]
                                 lock_key=None)
        self.assertIsNone(robot.lock_key)
        self.assertIsNone(robot.connected().lock)

    def test_a_stated_key_is_the_key(self) -> None:
        robot = Robot.from_parts(arm=_Reports("ur"), gripper=None,  # type: ignore[arg-type]
                                 lock_key="ur@10.7.7.7")
        lock = robot.connected().lock
        assert lock is not None
        self.assertEqual((lock.key, lock.owner), ("ur@10.7.7.7", "Robot"))


class WithoutCameraWorldTests(unittest.TestCase):
    """A decline block entered on a robot is bound to that robot's arm, not to the process."""

    @staticmethod
    def _robot() -> Robot:
        arm = _ur_arm("curobo")
        arm._curobo_ur = _FakePlanner(  # type: ignore[assignment]
            plan_result=[[0.0, -1.5, 1.5, 0.0, 1.5, 0.0]],
            execute_result=MotionResult.executed(MotionCommand.MOVE_TO, target_pose=_pose()),
        )
        arm._gate_planned_config = lambda pose, joints: None  # type: ignore[method-assign]
        return Robot.from_parts(arm=arm, gripper=None, lock_key=None)

    def test_a_block_on_one_robot_declines_its_arm_and_not_another(self) -> None:
        declined, other = self._robot(), self._robot()
        with declined.without_camera_world("bench check, no cameras mounted"):
            inside = declined.arm.move(_pose())
            beside = other.arm.move(_pose())
        after = declined.arm.move(_pose())
        self.assertEqual(
            inside.camera_world,
            CameraWorldStamp.declined(CameraWorldDecline("bench check, no cameras mounted")),
        )
        self.assertIs(beside.camera_world.use, CameraWorldUse.MISSING)
        self.assertIs(after.camera_world.use, CameraWorldUse.MISSING)


class TheNameTests(unittest.TestCase):

    def test_both_package_names_are_the_noun(self) -> None:
        import src.robot as robot_package
        import src.robot.execution as execution

        self.assertIs(robot_package.Robot, Robot)
        self.assertIs(execution.Robot, Robot)


_ROBOT_MUST_NOT_LOAD = _BUILDER_MUST_NOT_LOAD + (
    "src.robot.execution.autonomous_grasp",
    "src.robot.execution.calibration",
)

_BUILD_AND_CONNECT = "\n".join((
    "from src.config.schema.robot import RobotConfig",
    "from src.robot.execution.robot import Robot",
    "robot = Robot.from_config(RobotConfig(vendor='dummy', gripper={'vendor': 'dummy'}))",
    "with robot.connected() as live:",
    "    assert robot.arm.is_connected",
    "assert live.teardown is not None and live.teardown.clean",
))


class ARobotLoadsNoPickServiceTests(unittest.TestCase):

    def test_building_and_connecting_a_robot_loads_no_grasping_stack(self) -> None:
        self.assertEqual(_loaded_after(_BUILD_AND_CONNECT, _ROBOT_MUST_NOT_LOAD), [])

    def test_the_probe_sees_the_pick_service_where_it_is(self) -> None:
        """The self-failing control: ``runtime_pick`` loads the grasping stack at module top."""
        loaded = _loaded_after("import src.robot.execution.runtime_pick",
                               _ROBOT_MUST_NOT_LOAD)
        self.assertIn("src.robot.grasping", loaded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
