"""The kit builds, from a tree, the planned arm the UR driver builds from it, or refuses it the same (P0g slice 13).

The core tests stand on ``tests/_arm_kit.planned_arm``: a ``PlannedArm`` over the fake transport, built from its keyword
constructor with the pieces a vendor hands it. The kit copies the UR driver's recipe for those pieces (the capabilities,
the joint order and turn of the description, the preflight the tree's safety block builds for the named hand, the home),
so a test moved off the UR still holds for it only while the copy is the recipe. This holds it: for every kind of tree
the moved tests build, both are built and compared piece by piece, and a tree one refuses the other refuses with the
same sentence. The sabotage row is a kit arm built another way, which the comparison must tell apart.
"""

from __future__ import annotations

import unittest
from collections.abc import Callable
from typing import Any

from src.robot.safety.planning.mounted_tool import MountedTool
from src.config import ConfigError, load_robot_config
from src.config.schema.robot import RobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from tests._arm_kit import kit_tree, planned_arm
from tests._wrist_body import wrist_cell

#: A +Z hand on a real UR flange, and the Isaac cell's +Y frame (tests/test_the_declared_frame_agrees_with_the_hand.py).
_REAL_UR = {"source": "willy", "offset_mm": (0.0, 0.0, 132.0), "rotation_quat_xyzw": (0.0, 0.0, 0.0, 1.0)}
_ISAAC = {"source": "willy", "offset_mm": (0.0, 132.0, 0.0),
          "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)}
_INTO_THE_WRIST = {"source": "willy", "offset_mm": (0.0, 0.0, -132.0), "rotation_quat_xyzw": (1.0, 0.0, 0.0, 0.0)}


def _camera_world_cell() -> RobotConfig:
    """The cuRobo cell that asks for a live world from its cameras (tests/test_robot.py, ARobotTakesItsCamerasTests)."""
    from tests.test_camera_world_wiring import _cell

    cfg = _cell(world=True)
    return cfg.model_copy(update={"gripper": cfg.gripper.model_copy(update={"model": "robotiq_2f85"})})


#: The trees the moved tests build: each route, other models, both frames, the shipped profiles, both camera cells and a
#: planning world.
_BUILT: dict[str, Callable[[], RobotConfig]] = {
    "ik": lambda: kit_tree("ik"),
    "curobo": lambda: kit_tree("curobo", safety={"self_collision": {"planner_margin_mm": 4.0}}),
    "ur3e": lambda: kit_tree("curobo", model="ur3e"),
    "ur10 with a home": lambda: kit_tree("curobo", model="ur10", home_joint_positions=[0.2, -1.4, 1.3, -1.5, -1.57, 0.1]),
    "+Z frame": lambda: kit_tree("curobo", gripper={"tool_frame": _REAL_UR}),
    "+Y frame": lambda: kit_tree("curobo", gripper={"tool_frame": _ISAAC}),
    "hande profile": lambda: load_robot_config(profile="hande"),
    "ursim profile": lambda: load_robot_config(profile="ursim"),
    "wrist camera cell": lambda: wrist_cell(),
    "camera world cell": lambda: _camera_world_cell(),
    "planning world with a part": lambda: kit_tree("curobo", safety={"planning_world": {
        "enabled": True, "support_plane": {"height_mm": 0.0, "extent_mm": [1600.0, 1600.0], "thickness_mm": 50.0},
        "payload": {"enabled": True, "length_mm": 120.0, "lateral_margin_mm": 10.0}}}),
}

#: The trees both must refuse, and a word of the sentence.
_REFUSED: dict[str, tuple[Callable[[], RobotConfig], str]] = {
    "no hand": (lambda: RobotConfig.model_validate({"vendor": "ur"}), "robot.gripper.model"),
    "an unknown hand": (lambda: RobotConfig.model_validate({"vendor": "ur", "gripper": {"model": "no_such_hand"}}),
                        "no_such_hand"),
    "-Z frame": (lambda: kit_tree("curobo", gripper={"tool_frame": _INTO_THE_WRIST}), "places no hand model"),
}


def _pieces(arm: Any) -> dict[str, Any]:
    """What a vendor hands ``PlannedArm``, read back off the built arm."""
    preflight = arm.safety_preflight
    return {
        "capabilities": arm.capabilities,
        "joint names": arm._joint_names,
        "controller yaw": arm._controller_yaw_deg,
        "home": arm.home_joint_positions,
        "planner": arm._motion_planner,
        "workspace": arm._guard.limits if hasattr(arm._guard, "limits") else type(arm._guard).__name__,
        "guards": tuple((guard.name, type(guard).__name__) for guard in preflight.guards),
        "arm model": preflight.arm_model,
        "hand": preflight.planner_hand(arm),
        # The tool on the flange (P0i i5): the UR driver's is read from the tree once, the kit's from the tree and the
        # guard's hand; the two agree.
        "tool": arm.mounted_tool,
        "path step": preflight.path_step_mm,
        "radii": preflight.joint_radii_mm(arm),
    }


def _refusal(build: Callable[[], Any]) -> str:
    try:
        build()
    except ConfigError as refused:
        return str(refused)
    raise AssertionError("it built")


class TheKitBuildsTheUrDriversArmTests(unittest.TestCase):
    def test_every_tree_the_moved_tests_build_is_built_the_same(self) -> None:
        for name, tree in _BUILT.items():
            with self.subTest(tree=name):
                config = tree()
                ur = URRobotArm(config)
                self.assertEqual(_pieces(ur), _pieces(planned_arm(config)))
                self.assertEqual(ur.mounted_tool, MountedTool.from_config(config))

    def test_every_tree_the_ur_refuses_the_kit_refuses_with_the_same_sentence(self) -> None:
        for name, (tree, said) in _REFUSED.items():
            with self.subTest(tree=name):
                ur, kit = _refusal(lambda: URRobotArm(tree())), _refusal(lambda: planned_arm(tree()))
                self.assertIn(said, ur)
                self.assertEqual(ur, kit)

    def test_the_comparison_tells_another_build_apart(self) -> None:
        """The sabotage row: a kit arm handed another home, or another turn, is not the UR's arm."""
        config = _BUILT["curobo"]()
        ur = _pieces(URRobotArm(config))
        for label, kit in (("home", planned_arm(config, home_joints=[0.5] * 6)),
                           ("turn", planned_arm(config, controller_yaw_deg=0))):
            with self.subTest(changed=label):
                self.assertNotEqual(ur, _pieces(kit))


if __name__ == "__main__":
    unittest.main()
