"""A ``PlannedArm`` is built by keyword from typed pieces, and every method a vendor may answer has a declared place.

Until P0g slice 4 the planning half had no constructor: ``URRobotArm`` wrote its state attribute by attribute, the home
gate and the box raised ``NotImplementedError`` in the base, and ``connect`` and ``is_inside_workspace`` were the
``RobotArm`` Protocol's stubs, so a bare ``PlannedArm.connect()`` returned ``None`` with nothing opened. The fake
transport of slice 5 builds a ``PlannedArm`` with no vendor subclass at all, so each of those has an answer now: the
home gate and the box are neutral, the workspace check asks the box, and ``connect`` refused until slice 15 gave it
its neutral half (``tests/test_a_planned_arm_connects_through_its_transport.py`` holds it since). ``_VENDOR_HOOKS`` is
the one list of what a vendor may override.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.geometry import Frame, FrameMismatchError, Pose
from src.robot.core import JointPositions, MotionCommand, MotionStatus, RobotCapabilities
from src.robot.drivers.planned.arm import PlannedArm
from src.robot.safety import SafetyPreflight
from src.robot.safety.planning.hand import planner_hand
from src.robot.safety.workspace import WorkspaceGuard

_HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]


def _config() -> RobotConfig:
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"ip": "127.0.0.1", "model": "ur5e"},
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": "controller", "offset_mm": [0.0, 0.0, 150.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
    })


def _pieces(transport: Any) -> dict[str, Any]:
    cfg = _config()
    return {
        "config": cfg,
        "logger": logging.getLogger("tests.planned_arm_by_keyword"),
        "capabilities": RobotCapabilities(vendor="ur", model="ur5e", dof=6, controller="127.0.0.1",
                                          owns_a_controller=True),
        "joint_names": ("j1", "j2", "j3", "j4", "j5", "j6"),
        "controller_yaw_deg": 0,
        "transport": transport,
        "guard": WorkspaceGuard(cfg.workspace_limits),
        "preflight": SafetyPreflight.from_safety_config(cfg.safety, cfg.workspace_limits, hand=planner_hand(cfg),
                                                        arm_model="ur5e"),
        "home_joints": _HOME,
    }


def _pose(x: float, y: float, z: float, frame: Frame = Frame.BASE) -> Pose:
    return Pose(position_mm=np.array([x, y, z]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), frame=frame)


def _arm(transport: Any | None = None) -> PlannedArm:
    if transport is None:
        transport = MagicMock()
        transport.is_connected = False
    return PlannedArm(**_pieces(transport))


class APlannedArmIsBuiltByKeywordTests(unittest.TestCase):
    def test_it_is_built_from_its_pieces_with_no_vendor_subclass(self) -> None:
        transport = MagicMock()
        pieces = _pieces(transport)
        arm = PlannedArm(**pieces)
        self.assertIs(pieces["config"], arm.config)
        self.assertIs(pieces["capabilities"], arm.capabilities)
        self.assertIs(pieces["preflight"], arm.safety_preflight)
        self.assertEqual(_HOME, arm._home_joints)
        self.assertEqual(pieces["config"].motion.planner, arm._motion_planner, "the route is the tree's")
        self.assertIsNone(arm._curobo)
        self.assertIsNone(arm._attached_payload)
        self.assertEqual([], transport.mock_calls, "building an arm sends nothing")

    def test_a_positional_piece_is_refused(self) -> None:
        pieces = _pieces(MagicMock())
        with self.assertRaises(TypeError):
            PlannedArm(pieces["config"])  # type: ignore[misc]

class TheNeutralHooksTests(unittest.TestCase):
    def test_the_box_judges_a_base_frame_pose_as_it_is_and_refuses_another_frame(self) -> None:
        arm = _arm()
        pose = _pose(100.0, 50.0, 200.0)
        self.assertIs(pose, arm._boxed(pose))
        self.assertTrue(arm.is_inside_workspace(pose))
        self.assertFalse(arm.is_inside_workspace(_pose(900.0, 0.0, 200.0)))
        with self.assertRaisesRegex(FrameMismatchError, "PlannedArm requires Frame.BASE"):
            arm.is_inside_workspace(_pose(100.0, 50.0, 200.0, Frame.CAMERA))

    def test_the_home_gate_asks_the_box_about_the_grasp_centre_the_controller_reports(self) -> None:
        transport = MagicMock()
        transport.is_connected = True
        transport.controller_fk.return_value = _pose(900.0, 0.0, 200.0)
        refused = _arm(transport)._home_refusal(JointPositions(np.asarray(_HOME)))
        assert refused is not None
        self.assertIs(MotionStatus.WORKSPACE_REJECTED, refused.status)
        self.assertIs(MotionCommand.MOVE_HOME, refused.command)
        self.assertIn("the home grasp centre (900.0, 0.0, 200.0) mm is outside workspace_limits", refused.message)
        transport.controller_fk.return_value = _pose(100.0, 0.0, 200.0)
        self.assertIsNone(_arm(transport)._home_refusal(JointPositions(np.asarray(_HOME))))

    def test_a_home_whose_forward_kinematics_cannot_be_read_is_refused(self) -> None:
        transport = MagicMock()
        transport.is_connected = True
        transport.controller_fk.side_effect = RuntimeError("RTDE receive is down")
        refused = _arm(transport)._home_refusal(JointPositions(np.asarray(_HOME)))
        assert refused is not None
        self.assertIs(MotionStatus.CONTROLLER_REJECTED, refused.status)
        self.assertIn("could not check the home pose", refused.message)

    def test_an_arm_that_is_not_connected_leaves_the_home_to_the_path_judge(self) -> None:
        self.assertIsNone(_arm()._home_refusal(JointPositions(np.asarray(_HOME))))

    def test_a_home_the_destination_guards_refuse_is_refused_as_a_home(self) -> None:
        refused = _arm()._home_refusal(JointPositions(np.asarray([0.0, -1.57, 1.57, -1.57, -1.57, 400.0])))
        assert refused is not None
        self.assertIs(MotionCommand.MOVE_HOME, refused.command)
        self.assertIn("reads as degrees", refused.message)


class TheVendorHooksAreOneListTests(unittest.TestCase):
    def test_the_ur_arm_overrides_only_declared_hooks(self) -> None:
        from src.robot.drivers.ur.arm import URRobotArm

        methods = (property, classmethod, staticmethod)
        overridden = {name for name, value in vars(URRobotArm).items()
                      if not name.startswith("__") and name in vars(PlannedArm)
                      and (callable(value) or isinstance(value, methods))}
        self.assertLessEqual(overridden, PlannedArm._VENDOR_HOOKS)
        self.assertGreaterEqual(len(overridden), 1, "the UR answers the hooks it has its own answer to")

    def test_every_declared_hook_has_an_answer_in_the_base(self) -> None:
        for name in PlannedArm._VENDOR_HOOKS:
            with self.subTest(hook=name):
                self.assertIn(name, vars(PlannedArm), "a hook the base does not declare is inherited from a stub")


if __name__ == "__main__":
    unittest.main()
