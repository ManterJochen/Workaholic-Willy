"""The URDF written from Universal Robots' pinned description is a whole robot, for every one of the 14 models.

``_ur_description.render_urdf`` transcribes ``ur_macro.xacro`` without ROS. The forward kinematics proof lives in
``test_ur_urdf_renders_the_dh_chain.py``; this file holds what that proof takes for granted: one root, every link and
joint the macro emits, the limits UR declares (the elbow at half a turn, wrist_3 continuous exactly where UR gives it no
position limits), mesh references in the folder UR names, and the same bytes every time.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import unittest
import xml.etree.ElementTree as ET

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MODULE = _ROOT / "scripts" / "curobo" / "_ur_description.py"

_LINKS = {"base_link", "base_link_inertia", "shoulder_link", "upper_arm_link", "forearm_link", "wrist_1_link",
          "wrist_2_link", "wrist_3_link", "ft_frame", "base", "flange", "tool0"}
_ARM_JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
               "wrist_3_joint")


def _load():
    name = "_ur_description_for_render_tests"
    spec = importlib.util.spec_from_file_location(name, _MODULE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TheRenderedRobotIsWholeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ur = _load()
        cls.rendered = {model: cls.ur.render_urdf(model) for model in cls.ur.models()}

    def test_every_model_renders(self) -> None:
        self.assertEqual(len(self.rendered), 14)

    def test_one_root_and_every_link(self) -> None:
        for model, text in self.rendered.items():
            robot = ET.fromstring(text)
            links = {link.get("name") for link in robot.findall("link")}
            children = {joint.find("child").get("link") for joint in robot.findall("joint")}
            with self.subTest(model=model):
                self.assertEqual(robot.get("name"), model)
                self.assertEqual(links, _LINKS)
                self.assertEqual(links - children, {"base_link"}, "the builder needs exactly one root")

    def test_the_limits_are_ur_s(self) -> None:
        for model, text in self.rendered.items():
            robot = ET.fromstring(text)
            joints = {joint.get("name"): joint for joint in robot.findall("joint")}
            declared = self.ur.load(model).joint_limits
            for name in _ARM_JOINTS:
                joint = joints[name]
                limit = joint.find("limit")
                with self.subTest(model=model, joint=name):
                    self.assertEqual(joint.find("axis").get("xyz"), "0 0 1")
                    self.assertAlmostEqual(float(limit.get("velocity")), declared[name].velocity_rad_s, delta=1e-12)
                    if declared[name].has_position_limits:
                        self.assertEqual(joint.get("type"), "revolute")
                        self.assertAlmostEqual(float(limit.get("lower")), declared[name].lower_rad, delta=1e-12)
                        self.assertAlmostEqual(float(limit.get("upper")), declared[name].upper_rad, delta=1e-12)
                    else:
                        self.assertEqual(joint.get("type"), "continuous")
                        self.assertIsNone(limit.get("lower"))
            elbow = joints["elbow_joint"].find("limit")
            with self.subTest(model=model, joint="elbow_joint", check="half a turn"):
                self.assertAlmostEqual(float(elbow.get("upper")), math.pi, delta=1e-12)

    def test_a_continuous_wrist_is_rendered_where_ur_declares_one(self) -> None:
        """The control for the branch above: at least one pinned model has no wrist_3 position limits, so the
        continuous half of that test is exercised and not vacuous."""
        continuous = [model for model in self.rendered
                      if not self.ur.load(model).joint_limits["wrist_3_joint"].has_position_limits]
        self.assertTrue(continuous, "no model declares a continuous wrist_3; the test above cannot fail that half")

    def test_meshes_are_referenced_where_ur_names_them(self) -> None:
        for model, text in self.rendered.items():
            robot = ET.fromstring(text)
            for link in robot.findall("link"):
                for kind in ("visual", "collision"):
                    for element in link.findall(kind):
                        filename = element.find("geometry/mesh").get("filename")
                        with self.subTest(model=model, link=link.get("name"), kind=kind):
                            parts = pathlib.PurePosixPath(filename).parts
                            self.assertEqual(parts[0], "meshes")
                            self.assertEqual(parts[2], kind)
                            self.assertNotIn("://", filename)

    def test_every_inertial_carries_an_inertia(self) -> None:
        for model, text in self.rendered.items():
            robot = ET.fromstring(text)
            for link in robot.findall("link"):
                inertial = link.find("inertial")
                if inertial is None:
                    continue
                with self.subTest(model=model, link=link.get("name")):
                    self.assertIsNotNone(inertial.find("mass"))
                    self.assertIsNotNone(inertial.find("inertia"))

    def test_the_same_bytes_every_time(self) -> None:
        for model in ("ur3", "ur5e", "ur30"):
            with self.subTest(model=model):
                self.assertEqual(self.ur.render_urdf(model), self.rendered[model])
                self.assertNotIn("\r", self.rendered[model])
