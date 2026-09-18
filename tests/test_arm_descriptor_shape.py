"""A cuRobo descriptor is built per arm and carries no hand (UM lane S11).

``build_ur_config.py`` used to write ``{arm}_{hand}.yml``, the hand's spheres on ``tool0``, one file per pairing. It now
writes ``willy_{arm}.yml`` and refuses ``--gripper`` and ``--coupling-mm``: the sidecar adds the hand the cell names as a
body link when it starts. The shape of an arm descriptor is ``scripts/curobo/_arm_descriptor.py``, loaded here by path
because the builder runs where this repository cannot be imported. The builder itself is read as source.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_HELPER = _ROOT / "scripts" / "curobo" / "_arm_descriptor.py"
_BUILD = _ROOT / "scripts" / "curobo" / "build_ur_config.py"
_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link", "wrist_2_link", "wrist_3_link"]
_SIM = {"source": "willy", "offset_mm": [0.0, 132.0, 0.0], "rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678]}


def _module():
    spec = importlib.util.spec_from_file_location("_arm_descriptor_under_test", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _template() -> dict:
    """The tables of cuRobo's UR template as the build holds them before the hand used to go in.

    Copied from ext_deps/curobo/curobo/content/configs/robot/_ur_template.yml:13-20 and :105-142 (the content directory is
    not in the repository), with the ``forarm_link`` typo the build corrects already corrected and one sphere per arm
    link: the build puts Lula's spheres there, and no Isaac UR Lula description names tool0.
    """
    return {"robot_cfg": {"kinematics": {
        "urdf_path": "robot/ur_description/ur10e.urdf",
        "tool_frames": ["tool0"],
        "collision_link_names": [*_LINKS, "tool0"],
        "collision_sphere_buffer": 0.0,
        "collision_spheres": {link: [{"center": [0.0, 0.0, 0.0], "radius": 0.05}] for link in _LINKS},
        "lock_joints": None,
        "self_collision_buffer": {"forearm_link": 0, "shoulder_link": 0.07, "tool0": 0.0, "upper_arm_link": 0,
                                  "wrist_1_link": 0, "wrist_2_link": 0, "wrist_3_link": 0},
        "self_collision_ignore": {
            "camera_mount": ["tool0", "wrist_3_link"],
            "forearm_link": ["wrist_1_link"],
            "upper_arm_link": ["forearm_link", "shoulder_link"],
            "wrist_1_link": ["wrist_2_link", "wrist_3_link"],
            "wrist_2_link": ["wrist_3_link", "tool0"],
            "wrist_3_link": ["tool0"],
        },
    }}}


class AnArmDescriptorGuardsNoToolFrameTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mod = _module()

    def test_tool0_leaves_the_links_and_the_buffer_and_stays_in_the_ignore_lists(self) -> None:
        kin = self.mod.arm_only(_template())["robot_cfg"]["kinematics"]
        self.assertEqual(kin["collision_link_names"], _LINKS)
        self.assertNotIn("tool0", kin["self_collision_buffer"])
        self.assertEqual(kin["self_collision_ignore"]["wrist_2_link"], ["wrist_3_link", "tool0"])
        self.assertEqual(kin["self_collision_ignore"]["wrist_3_link"], ["tool0"])
        self.assertEqual(kin["self_collision_buffer"]["shoulder_link"], 0.07)

    def test_the_camera_mount_key_goes_and_every_arm_key_stays(self) -> None:
        ignore = self.mod.arm_only(_template())["robot_cfg"]["kinematics"]["self_collision_ignore"]
        self.assertNotIn("camera_mount", ignore)
        self.assertEqual(sorted(ignore), sorted(set(_template()["robot_cfg"]["kinematics"]["self_collision_ignore"])
                                                - {"camera_mount"}))

    def test_it_says_it_carries_no_hand_and_keeps_what_the_provenance_said(self) -> None:
        cfg = _template()
        cfg["_provenance"] = {"arm": "ur10e", "fit_seed": 0}
        self.assertEqual(self.mod.arm_only(cfg)["_provenance"], {"arm": "ur10e", "fit_seed": 0, "carries_hand": False})

    def test_the_template_is_never_mutated(self) -> None:
        cfg = _template()
        before = copy.deepcopy(cfg)
        self.mod.arm_only(cfg)
        self.assertEqual(cfg, before)

    def test_a_config_that_carries_a_hand_is_refused(self) -> None:
        """⭐ THE CONTROLS: tool0 spheres, or a provenance that names the hand it was built for."""
        spheres = _template()
        spheres["robot_cfg"]["kinematics"]["collision_spheres"]["tool0"] = [{"center": [0, 0.05, 0], "radius": 0.04}]
        keyed = _template()
        keyed["_provenance"] = {"arm": "ur5e", "gripper_key": "robotiq_2f85"}
        for label, cfg in (("tool0 spheres", spheres), ("gripper_key", keyed)):
            with self.subTest(label), self.assertRaises(self.mod.ArmDescriptorError):
                self.mod.arm_only(cfg)


class TheHandComposesOntoTheArmDescriptorTests(unittest.TestCase):
    """S10 and S11 meet here: the body link composes onto what the builder writes, and not onto the raw template."""

    def _hand_link(self) -> dict:
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.body_link import HandLink
        from src.robot.safety.planning.hand import planner_hand

        hand = planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": {
            "model": "robotiq_hande", "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}], "tool_frame": _SIM}}))
        return HandLink.from_hand(hand).to_dict()

    def test_the_arm_descriptor_takes_the_hand(self) -> None:
        from src.robot.safety.planning._curobo_body_links import compose_sidecar_config

        composed = compose_sidecar_config(_module().arm_only(_template()), bodies=[self._hand_link()], margin_mm=10.0)
        self.assertEqual(composed["robot_cfg"]["kinematics"]["collision_link_names"], [*_LINKS, "hand"])

    def test_the_raw_template_does_not(self) -> None:
        """The control: it guards tool0 with no spheres, which cuRobo raises on, so the composition refuses first."""
        from src.robot.safety.planning._curobo_body_links import BodyLinkError, compose_sidecar_config

        with self.assertRaises(BodyLinkError):
            compose_sidecar_config(_template(), bodies=[self._hand_link()])


class TheBuilderWritesOneDescriptorPerArmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _BUILD.read_text(encoding="utf-8")

    def test_it_is_named_by_the_arm_and_made_arm_only(self) -> None:
        self.assertIn('f"configs/robot/willy_{MODEL}.yml"', self.source)
        self.assertIn("arm_only(cfg)", self.source)

    def test_nothing_of_a_hand_is_read_or_written(self) -> None:
        for name in ("_gripper_placement", "_gripper_spheres.yml", '"gripper_key"', '"coupling_mm"', "GRIPPER"):
            with self.subTest(absent=name):
                self.assertNotIn(name, self.source)

    def test_a_hand_flag_is_refused_by_name(self) -> None:
        for flag in ('"--gripper"', '"--coupling-mm"'):
            with self.subTest(flag=flag):
                self.assertIn(flag, self.source)
        self.assertIn("raise SystemExit", self.source[self.source.find('"--gripper"'):][:600])


if __name__ == "__main__":
    unittest.main()
