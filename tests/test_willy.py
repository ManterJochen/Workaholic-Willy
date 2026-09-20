"""The library door: `from willy import ...` reaches the library's own objects, and the conveniences the examples use.

Owner, 2026-09-18 afternoon: the examples should read "so wie man sie bei einer richtigen Bibliothek aufrufen wuerde",
through one top-level package. The door adds no behaviour, so what is held here is that it adds none: every name is
its home module's object, importing it loads nothing, and the four conveniences (`load_tree`, `Pose.tool_down`,
`print(report)`, the `from_tree` doors of `Camera` and `Locator`) are the calls they replace.
"""

from __future__ import annotations

import inspect
import os
import subprocess
import sys
import unittest
from importlib import import_module
from unittest import mock

import numpy as np

import willy
from src.config import ConfigError, ConfigTree, load_tree
from src.geometry import Frame, Pose


class TheDoorTests(unittest.TestCase):

    def test_every_name_is_the_object_its_home_module_defines(self) -> None:
        for name in willy.__all__:
            with self.subTest(name):
                home = willy._HOME[name]  # noqa: SLF001
                self.assertIs(getattr(import_module(home), name), getattr(willy, name))

    def test_importing_the_door_loads_nothing(self) -> None:
        code = ("import sys, willy; print(sorted(m for m in sys.modules "
                "if m.startswith(('src', 'datagen', 'torch', 'numpy'))))")
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        self.assertEqual(0, proc.returncode, proc.stderr[-1500:])
        self.assertEqual("[]", proc.stdout.strip())

    def test_an_unknown_name_is_an_attribute_error(self) -> None:
        with self.assertRaisesRegex(AttributeError, "no attribute 'Robotic'"):
            getattr(willy, "Robotic")

    def test_every_exported_class_that_renders_prints_as_its_render(self) -> None:
        """A `render()` that takes no argument describes the object; `DatasetBuild.render(...)` renders images."""
        for name in willy.__all__:
            value = getattr(willy, name)
            render = getattr(value, "render", None) if isinstance(value, type) else None
            if callable(render) and list(inspect.signature(render).parameters) == ["self"]:
                with self.subTest(name):
                    self.assertIn("__str__", vars(value), f"{name} renders but print() shows its repr")


#: Every class in the modules the door exports from that describes itself: print() shows the same text.
_RENDERED = {
    "src.config.explain": ["KeyExplanation"],
    "src.config.tree": ["LoadedTree"],
    "src.camera.orchestration.camera": ["Camera"],
    "src.models.speech.confirm": ["Confirmation"],
    "src.models.speech.listener": ["Utterance"],
    "src.models.speech.push_to_talk": ["TalkRecording"],
    "src.models.speech.transcript": ["Proposal", "SpeechCheck", "Transcript"],
    "src.robot.core.motion_result": ["MotionResult"],
    "src.robot.execution.autonomous_grasp.report": ["AutonomousGraspReport"],
    "src.robot.execution.camera_world_wiring": ["CameraWorldPlan", "CameraWorldWiring"],
    "src.robot.execution.hand_eye": ["CalibrationBuild", "CalibrationCheck", "CalibrationRunReport"],
    "src.robot.execution.handling": ["HandReport", "HandlingReport"],
    "src.robot.execution.lifecycle": ["TeardownReport"],
    "src.robot.execution.motion": ["MotionReport", "RouteReading"],
    "src.robot.execution.pick_run": ["PassRule", "PickAttempt", "PickRunReport", "Recording"],
    "src.robot.execution.planner_start": ["PlannerStartReport"],
    "src.robot.execution.real_cell.preflight": ["PreflightReport"],
    "src.robot.execution.robot": ["Robot"],
    "src.robot.execution.wrist_bodies": ["WristBodies"],
    "src.robot.grasping.scene": ["SceneGrasps"],
    "src.robot.perception.locator": ["Located", "LocatedObject"],
    "src.robot.safety.attestation": ["SafetyAttestation"],
    "src.calibration.rig_calibration": ["RigCalibration"],
    "src.models.perception_spec": ["PerceptionResolution"],
    "src.robot.grasping.deep.foreign.service": ["ImportReport"],
    "src.robot.grasping.deep.train.api": ["CorpusProbe"],
    "src.robot.grasping.deep.train.report": ["TrainingRunReport"],
    "src.robot.safety.planning.stack": ["MotionStackReport"],
    "datagen.api": ["DatasetReport"],
    "datagen.cost": ["Estimate"],
    "datagen.grasps.service": ["PhysicsReport"],
    "datagen.heldout": ["HeldOutReport"],
    "datagen.verify": ["VerifyReport"],
}


class PrintShowsTheReportTests(unittest.TestCase):

    def test_every_class_that_renders_defines_str(self) -> None:
        for module, names in _RENDERED.items():
            for name in names:
                with self.subTest(f"{module}.{name}"):
                    cls = getattr(import_module(module), name)
                    self.assertIn("__str__", vars(cls))

    def test_a_robot_and_its_reports_print_as_they_render(self) -> None:
        from willy import Robot

        robot = Robot.from_tree(load_tree("console_dummy"))
        self.assertEqual(robot.render(), str(robot))
        with robot.connected():
            report = robot.home()
        self.assertEqual(report.render(), str(report))
        tree = load_tree("console_dummy")
        self.assertEqual(tree.render(), str(tree))


class LoadTreeTests(unittest.TestCase):

    def test_a_named_chain_is_the_tree_the_directory_loads(self) -> None:
        self.assertEqual(ConfigTree.from_directory(profile="console_dummy").load().robot,
                         load_tree("console_dummy").robot)

    def test_unset_follows_the_profile_the_operator_set_and_none_is_the_base_tree(self) -> None:
        with mock.patch.dict(os.environ, {"WILLY_PROFILE": "console_dummy"}):
            self.assertEqual("console_dummy", load_tree().profile)
            self.assertIsNone(load_tree(None).profile)

    def test_a_tree_that_does_not_load_is_a_verdict(self) -> None:
        tree = load_tree("no_such_layer")
        self.assertFalse(tree.ok)
        with self.assertRaises(ConfigError):
            _ = tree.robot


class ToolDownTests(unittest.TestCase):

    def test_straight_down_with_no_yaw_is_half_a_turn_about_x(self) -> None:
        pose = Pose.tool_down(450.0, 100.0, 300.0)
        np.testing.assert_allclose([1.0, 0.0, 0.0, 0.0], pose.quaternion_xyzw, atol=1e-12)
        np.testing.assert_allclose([450.0, 100.0, 300.0], pose.position_mm)
        self.assertIs(Frame.BASE, pose.frame)

    def test_the_yaw_turns_the_closing_axis_and_keeps_the_tool_down(self) -> None:
        for yaw in (-135.0, -30.0, 45.0, 90.0, 180.0):
            with self.subTest(yaw=yaw):
                matrix = Pose.tool_down(0.0, 0.0, 0.0, yaw_deg=yaw).to_matrix()[:3, :3]
                turn = np.radians(yaw)
                np.testing.assert_allclose([np.cos(turn), np.sin(turn), 0.0], matrix[:, 0], atol=1e-12)
                np.testing.assert_allclose([0.0, 0.0, -1.0], matrix[:, 2], atol=1e-12)
                self.assertAlmostEqual(1.0, float(np.linalg.det(matrix)), places=12)

    def test_the_frame_and_the_label_are_the_callers(self) -> None:
        pose = Pose.tool_down(1.0, 2.0, 3.0, frame=Frame.CAMERA, label="seen")
        self.assertIs(Frame.CAMERA, pose.frame)
        self.assertEqual("seen", pose.label)


class TheTreeDoorsTests(unittest.TestCase):

    def test_a_camera_from_a_tree_is_the_rig_its_camera_section_gives(self) -> None:
        from src.camera.orchestration.camera import Camera

        tree = load_tree(None)
        with mock.patch.object(Camera, "from_config", return_value="owner") as from_config:
            self.assertEqual("owner", Camera.from_tree(tree, rig_id="overhead"))
        from_config.assert_called_once()
        self.assertIs(tree.app_config.camera, from_config.call_args.args[0])
        self.assertEqual("overhead", from_config.call_args.kwargs["rig_id"])

    def test_a_locator_from_a_tree_reads_the_robot_and_models_sections_of_it(self) -> None:
        from src.robot.perception.locator import Locator

        tree = load_tree("console_dummy")
        with mock.patch.object(Locator, "from_config", return_value="locator") as from_config:
            self.assertEqual("locator", Locator.from_tree(tree, camera="camera"))
        args = from_config.call_args
        self.assertEqual(tree.robot, args.args[0])
        self.assertIs(tree.app_config.models, args.args[1])
        self.assertEqual("camera", args.kwargs["camera"])

    def test_a_tree_that_did_not_load_is_refused_with_its_refusal(self) -> None:
        from src.camera.orchestration.camera import Camera

        with self.assertRaises(ConfigError):
            Camera.from_tree(load_tree("no_such_layer"))


class ASafetyPreflightFromATreeTests(unittest.TestCase):
    """The pipeline a cell's driver builds, from the loaded tree, with the hand resolved against the tree's root."""

    def test_it_is_the_pipeline_the_config_door_builds_with_the_trees_hand(self) -> None:
        from src.robot.safety import SafetyPreflight
        from src.robot.safety.planning.hand import planner_hand

        tree = load_tree(None).with_values({"robot.gripper.model": "robotiq_2f85"})
        robot = tree.robot
        by_hand = SafetyPreflight.from_safety_config(
            robot.safety, robot.workspace_limits, hand=planner_hand(robot, data_dir=tree.root),
            arm_model=robot.ur.model)
        self.assertEqual(by_hand.guard_names, SafetyPreflight.from_tree(tree).guard_names)
        self.assertIn("self_collision", SafetyPreflight.from_tree(tree).guard_names)

    def test_an_exact_mesh_guard_on_a_tree_that_names_no_hand_is_refused_naming_the_key(self) -> None:
        from src.robot.safety import SafetyPreflight

        with self.assertRaisesRegex(ConfigError, "robot.gripper.model"):
            SafetyPreflight.from_tree(load_tree(None))

    def test_a_tree_that_did_not_load_is_refused_with_its_refusal(self) -> None:
        from src.robot.safety import SafetyPreflight

        with self.assertRaises(ConfigError):
            SafetyPreflight.from_tree(load_tree("no_such_layer"))


if __name__ == "__main__":
    unittest.main()
