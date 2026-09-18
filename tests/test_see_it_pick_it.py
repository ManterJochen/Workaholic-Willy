"""From "I see it" to "I pick it": a located object or a grasp candidate gives the pose and the width Robot.pick takes.

``pose_from_grasp_axes`` is the one public function that turns a grasp's two directions into the ``Pose`` a motion
verb takes, and ``SupportFootprintCandidate.pose`` (the candidate ``Scene`` returns) and ``GraspPoint.pose`` both call
it. ``Located.scene`` hands a located object to the planner. ``Camera`` is a ``with`` block, the camera package
exports its refusals, and the perception package exports the locator.

The convention is the one ``Robot.pick`` reads and the pick service drives by: the pose's +Z is the approach, the
direction the tool travels toward the part, and its +X is the closing axis the pads close across. These tests hold it
by the axes and by what ``Robot.pick`` commands on the dummy arm, not by agreement between two helpers alone.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.config.schema.robot import RobotConfig
from src.camera.setup.image_taking.frames import RGBDFrame
from src.geometry import Frame, Pose, to_rotation_matrix
from src.robot.core import MotionResult
from src.robot.core.gripper import HoldEvidence
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.robot import Robot
from src.robot.grasping import Scene
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.perception.locator import Located, Locator
from tests.test_camera_noun import _Device, _opencv
from tests.test_locator import _backend, _Handle, _Owner, _Segmentation
from tests.test_robot_hand_verbs import _Hand

_REPO = Path(__file__).resolve().parents[1]
#: The tool-down orientation the Isaac probe hands Robot.pick (scripts/isaac/probe_robot_pick_place.py).
_TOOL_DOWN_XYZW = (0.0, 1.0, 0.0, 0.0)
#: Robot.pick's own defaults, restated so a change to them is seen here.
_STANDOFF_MM = 80.0
_SQUEEZE_MM = 1.0


def _cloud(*, x: tuple[float, float], y: tuple[float, float], height_mm: float, points: int = 900,
           seed: int = 0) -> np.ndarray:
    """A block resting on z = 0, sampled through its volume, in BASE millimetres."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(*x, points), rng.uniform(*y, points), rng.uniform(0.0, height_mm, points)])


def _tall_block() -> np.ndarray:
    """40 x 40 x 80 mm: tall enough that the best grasp comes straight down."""
    return _cloud(x=(380.0, 420.0), y=(-20.0, 20.0), height_mm=80.0)


def _flat_block() -> np.ndarray:
    """60 x 60 x 40 mm: too low for a vertical jaw to clear the table, so its grasps come from the side."""
    return _cloud(x=(400.0, 460.0), y=(-30.0, 30.0), height_mm=40.0)


class _TracingArm(DummyRobotArm):
    """The dummy arm, noting each motion's target position and whether it ran as a line."""

    def __init__(self) -> None:
        super().__init__()
        self.motions: list[tuple[tuple[float, ...], bool]] = []

    def move(self, pose: Pose, **kwargs: Any) -> MotionResult:
        self.motions.append((tuple(round(float(v), 3) for v in pose.position_mm), bool(kwargs.get("linear", False))))
        return super().move(pose, **kwargs)


def _robot() -> tuple[Robot, _TracingArm, _Hand]:
    arm = _TracingArm()
    arm.connect()
    hand = _Hand(hold=HoldEvidence.HELD, width_mm=39.0)
    return Robot.from_parts(arm=arm, gripper=hand, lock_key=None), arm, hand


def _rounded(values: Any) -> tuple[float, ...]:
    return tuple(round(float(v), 3) for v in values)


class ASceneCandidateGivesItsPoseTests(unittest.TestCase):
    """Fails with AttributeError where ``SupportFootprintCandidate`` has no ``pose``."""

    def test_every_candidate_gives_a_base_pose_whose_z_is_its_approach_and_whose_x_its_closing_axis(self) -> None:
        for name, cloud in (("tall", _tall_block()), ("flat", _flat_block())):
            grasps = Scene.from_cloud(cloud, support_height_mm=0.0).grasps(max_candidates=6)
            self.assertTrue(grasps.candidates, f"{name}: the fixture must yield candidates\n{grasps.render()}")
            for index, candidate in enumerate(grasps.candidates):
                with self.subTest(block=name, candidate=index):
                    pose = candidate.pose()
                    self.assertIsInstance(pose, Pose)
                    self.assertIs(Frame.BASE, pose.frame)
                    matrix = pose.to_matrix()
                    np.testing.assert_allclose(matrix[:3, 3], candidate.position_mm, atol=1e-9)
                    np.testing.assert_allclose(matrix[:3, 2], candidate.approach, atol=1e-9)
                    np.testing.assert_allclose(matrix[:3, 0], candidate.closing_axis, atol=1e-9)
                    self.assertAlmostEqual(1.0, float(np.linalg.det(matrix[:3, :3])), places=9)

    def test_the_fixtures_cover_a_vertical_and_a_side_approach(self) -> None:
        """The control on the test above: both approach families are exercised, not one of them twice."""
        tall = Scene.from_cloud(_tall_block(), support_height_mm=0.0).grasps().best
        flat = Scene.from_cloud(_flat_block(), support_height_mm=0.0).grasps().best
        assert tall is not None and flat is not None
        np.testing.assert_allclose(tall.pose().to_matrix()[:3, 2], (0.0, 0.0, -1.0), atol=1e-9)
        self.assertLess(abs(float(flat.pose().to_matrix()[2, 2])), 0.5, "the flat block's best grasp comes from the side")


class RobotPickTakesTheCandidateAsItIsTests(unittest.TestCase):
    """Fails with AttributeError where ``SupportFootprintCandidate`` has no ``pose``."""

    def test_robot_pick_backs_off_along_the_approach_and_closes_to_the_candidates_width(self) -> None:
        best = Scene.from_cloud(_tall_block(), support_height_mm=0.0).grasps().best
        assert best is not None
        robot, arm, hand = _robot()

        report = robot.pick(best.pose(), best.grip_width_mm)

        self.assertTrue(report.ok, report.render())
        grasp_at = _rounded(best.position_mm)
        standoff = _rounded(np.asarray(best.position_mm) - _STANDOFF_MM * np.asarray(best.approach))
        self.assertEqual([(standoff, False), (grasp_at, True), (standoff, True)], arm.motions)
        self.assertGreater(standoff[2], grasp_at[2], "a straight-down grasp stands off above the part")
        closes = [width for verb, width in hand.commands if verb == "set_width_mm"]
        self.assertAlmostEqual(best.grip_width_mm - _SQUEEZE_MM, closes[-1])
        where = arm.get_tcp_pose().to_matrix()
        np.testing.assert_allclose(where[:3, 2], best.approach, atol=1e-9)
        np.testing.assert_allclose(where[:3, 0], best.closing_axis, atol=1e-9)

    def test_a_camera_frame_grasp_gives_a_camera_pose_that_robot_pick_refuses_with_nothing_commanded(self) -> None:
        from src.robot.execution.handling import HandlingOutcome

        in_camera = GraspPoint(position=np.array([0.0, 0.0, 600.0]), approach=np.array([0.0, 0.0, 1.0]),
                               axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.8, frame=GraspFrame.CAMERA)
        pose = in_camera.pose()
        self.assertIs(Frame.CAMERA, pose.frame, "the frame travels; it is never relabelled")
        robot, arm, hand = _robot()

        report = robot.pick(pose, in_camera.grip_width_mm)

        self.assertIs(HandlingOutcome.REFUSED, report.outcome, report.render())
        self.assertEqual([], arm.motions)
        self.assertEqual([], hand.commands)


class AGraspPointGivesItsPoseTests(unittest.TestCase):
    """Fails with AttributeError where ``GraspPoint`` has no ``pose``."""

    def test_a_base_grasp_point_gives_the_pose_its_axes_describe(self) -> None:
        closing = np.array([np.cos(0.3), np.sin(0.3), 0.0])
        grasp = GraspPoint(position=np.array([410.0, -12.0, 55.0]), approach=np.array([0.0, 0.0, -1.0]), axis=closing,
                           grip_width_mm=33.0, score=0.7, frame=GraspFrame.BASE)
        matrix = grasp.pose().to_matrix()
        self.assertIs(Frame.BASE, grasp.pose().frame)
        np.testing.assert_allclose(matrix[:3, 3], (410.0, -12.0, 55.0), atol=1e-9)
        np.testing.assert_allclose(matrix[:3, 2], (0.0, 0.0, -1.0), atol=1e-9)
        np.testing.assert_allclose(matrix[:3, 0], closing, atol=1e-9)

    def test_it_agrees_with_the_collision_checks_adapter(self) -> None:
        """The pose a pick takes is the pose the collision check swept. Held alongside the axis test, not instead."""
        from src.robot.grasping.collision.candidate_filter import grasp_point_to_pose

        grasp = GraspPoint(position=np.array([400.0, 5.0, 60.0]), approach=np.array([0.0, 0.6, -0.8]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=30.0, score=0.5, frame=GraspFrame.BASE)
        np.testing.assert_allclose(grasp.pose().to_matrix(), grasp_point_to_pose(grasp).as_pose().to_matrix(),
                                   atol=1e-9)


class TheAxesFunctionTests(unittest.TestCase):
    """Fails with ImportError where ``pose_from_grasp_axes`` does not exist."""

    def test_straight_down_closing_along_minus_x_is_the_probes_tool_down(self) -> None:
        from src.robot.grasping import pose_from_grasp_axes

        pose = pose_from_grasp_axes((400.0, 0.0, 100.0), approach=(0.0, 0.0, -1.0), closing_axis=(-1.0, 0.0, 0.0),
                                    frame=Frame.BASE)
        np.testing.assert_allclose(pose.to_matrix()[:3, :3], to_rotation_matrix(np.array(_TOOL_DOWN_XYZW)), atol=1e-12)

    def test_a_closing_axis_off_the_perpendicular_is_used_perpendicular(self) -> None:
        from src.robot.grasping import pose_from_grasp_axes

        pose = pose_from_grasp_axes((0.0, 0.0, 0.0), approach=(0.0, 0.0, -2.0), closing_axis=(1.0, 0.0, 0.2),
                                    frame=Frame.BASE)
        np.testing.assert_allclose(pose.to_matrix()[:3, 0], (1.0, 0.0, 0.0), atol=1e-12)
        np.testing.assert_allclose(pose.to_matrix()[:3, 2], (0.0, 0.0, -1.0), atol=1e-12)

    def test_axes_that_define_no_rotation_are_refused(self) -> None:
        from src.robot.grasping import pose_from_grasp_axes

        cases = {
            "parallel": ((0.0, 0.0, -1.0), (0.0, 0.0, 3.0)),
            "zero approach": ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            "zero closing": ((0.0, 0.0, -1.0), (0.0, 0.0, 0.0)),
            "not finite": ((0.0, 0.0, float("nan")), (1.0, 0.0, 0.0)),
        }
        for name, (approach, closing) in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                pose_from_grasp_axes((0.0, 0.0, 0.0), approach=approach, closing_axis=closing, frame=Frame.BASE)


def _grounded(label: str, mask: np.ndarray) -> SimpleNamespace:
    """One object as the perception backend double returns it: a detection and a segmentation over ``mask``."""
    rows, cols = np.nonzero(mask)
    box = (float(cols.min()), float(rows.min()), float(cols.max() + 1), float(rows.max() + 1))
    return SimpleNamespace(detection=SimpleNamespace(score=0.9, box=list(box)),
                           segmentation=_Segmentation(label=label, score=0.9, bbox_xyxy=box, mask=mask))


def _two_objects() -> Located:
    """One frame, two blocks on separate pixels: a cube under the left mask and a box under the right one."""
    shape = (40, 40)
    depth = np.full(shape, 1000, dtype=np.uint16)
    depth[4:18, 4:18] = 920
    depth[24:36, 24:36] = 940
    left = np.zeros(shape, dtype=bool)
    left[4:18, 4:18] = True
    right = np.zeros(shape, dtype=bool)
    right[24:36, 24:36] = True
    objects = (_grounded("cube", left), _grounded("box", right))
    backend = SimpleNamespace(perceive=lambda _bgr, _prompt: objects)
    frame = RGBDFrame(color=np.zeros((*shape, 3), dtype=np.uint8), depth=depth, captured_at_s=50.0)
    return Locator.from_parts(camera=_Owner(handle=_Handle([frame])), backend=backend).locate("cube ; box")


class ALocatedObjectReachesThePickTests(unittest.TestCase):
    """Fails with AttributeError where ``Located`` has no ``scene``."""

    def test_the_scene_is_the_objects_surface_with_the_other_objects_as_obstacles(self) -> None:
        located = _two_objects()
        self.assertEqual(["cube", "box"], [obj.label for obj in located.objects])
        config = RobotConfig()

        scene = located.scene(0, config)

        np.testing.assert_array_equal(scene.target_points_base_mm, located.objects[0].points_base_mm)
        assert scene.obstacle_points_base_mm is not None
        np.testing.assert_array_equal(scene.obstacle_points_base_mm, located.objects[1].points_base_mm)
        self.assertEqual(float(config.grasping.support.height_mm), scene.support_height_mm)
        np.testing.assert_array_equal(located.scene(1, config).obstacle_points_base_mm,
                                      located.objects[0].points_base_mm)

    def test_a_lone_object_has_no_obstacles_and_an_index_outside_the_frame_is_refused(self) -> None:
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")
        self.assertIsNone(located.scene(0, RobotConfig()).obstacle_points_base_mm)
        with self.assertRaises(IndexError):
            located.scene(1, RobotConfig())

    def test_a_located_object_is_picked_through_its_scene_with_its_keep_out(self) -> None:
        """Without a simulator: locate, the scene, the best candidate's pose, Robot.pick with keep_out."""
        located = Locator.from_parts(camera=_Owner(), backend=_backend("red cube")).locate("a red cube")
        grasps = located.scene(0, RobotConfig()).grasps()
        best = grasps.best
        assert best is not None, grasps.render()
        robot, arm, _ = _robot()

        report = robot.pick(best.pose(), best.grip_width_mm, keep_out=located.keep_out(0))

        self.assertTrue(report.ok, report.render())
        np.testing.assert_allclose(best.pose().to_matrix()[:3, 2], (0.0, 0.0, -1.0), atol=1e-9)
        self.assertEqual((_rounded(best.position_mm), True), arm.motions[1])


class TheCameraIsAWithBlockTests(unittest.TestCase):
    """Fails with TypeError (not a context manager) where ``Camera`` has no ``__enter__``/``__exit__``."""

    def test_a_with_block_opens_the_owner_and_gives_the_device_back(self) -> None:
        from src.camera import Camera

        device = _Device()
        owner = Camera.from_rig(_opencv("overhead", 3), streamer=device)
        with owner as camera:
            self.assertIs(owner, camera)
            self.assertTrue(camera.is_open)
            camera.grab()
        self.assertFalse(owner.is_open)
        self.assertEqual((1, 1), (device.opens, device.releases))
        after = Camera.from_rig(_opencv("bench", 3), streamer=_Device())
        after.open()
        after.release()

    def test_a_block_that_raises_still_gives_the_device_back_and_the_exception_propagates(self) -> None:
        from src.camera import Camera

        device = _Device()
        owner = Camera.from_rig(_opencv("overhead", 3), streamer=device)
        with self.assertRaises(ZeroDivisionError), owner:
            _ = 1 / 0
        self.assertFalse(owner.is_open)
        self.assertEqual(1, device.releases)

    def test_a_busy_device_refuses_the_block_before_its_body_runs(self) -> None:
        from src.camera import Camera, CameraBusy

        holder = Camera.from_rig(_opencv("overhead", 3), streamer=_Device())
        holder.open()
        self.addCleanup(holder.release)
        second = Camera.from_rig(_opencv("bench", 3), streamer=_Device())
        ran = False
        with self.assertRaises(CameraBusy):
            with second:
                ran = True
        self.assertFalse(ran)
        self.assertTrue(holder.is_open)
        self.assertFalse(second.is_open)


class TheExportsTests(unittest.TestCase):
    def test_the_perception_package_exports_the_locator_and_its_refusal(self) -> None:
        """Fails where the package's ``__all__`` names only the RealSense source."""
        import src.robot.perception as perception
        from src.robot.perception import locator as module

        for name in ("Located", "LocatedObject", "LocatedOrientation", "Locator", "LocatorRefused"):
            with self.subTest(name=name):
                self.assertIn(name, perception.__all__)
                self.assertIs(getattr(module, name), getattr(perception, name))

    def test_the_camera_package_exports_the_owner_and_its_refusals(self) -> None:
        """Fails with ImportError where ``CameraNotOpen`` and ``CameraRefused`` are not on the package."""
        from src.calibration import rig_calibration
        from src.camera import (
            Camera,
            CameraBusy,
            CameraNotOpen,
            CameraRefused,
            RigCalibrationError,
            RigNotCalibrated,
        )
        from src.camera.orchestration import camera as module

        for exported, owner in ((Camera, module.Camera), (CameraBusy, module.CameraBusy),
                                (CameraNotOpen, module.CameraNotOpen), (CameraRefused, module.CameraRefused),
                                (RigCalibrationError, rig_calibration.RigCalibrationError),
                                (RigNotCalibrated, rig_calibration.RigNotCalibrated)):
            with self.subTest(name=exported.__name__):
                self.assertIs(owner, exported)

    def test_the_grasping_package_exports_the_axes_function(self) -> None:
        """Fails with ImportError where the package does not export it."""
        import src.robot.grasping as grasping
        from src.robot.grasping.geometry.grasp_frame import pose_from_grasp_axes

        self.assertIn("pose_from_grasp_axes", grasping.__all__)
        self.assertIs(pose_from_grasp_axes, grasping.pose_from_grasp_axes)

    def test_the_locator_from_the_package_imports_no_driver_pick_loop_camera_models_or_torch(self) -> None:
        """The weight the locator module keeps (tests/test_locator.py) holds for the package import too. Fails
        where the probe's own import of ``Locator`` from the package fails."""
        out = _probe(
            "from src.robot.perception import Locator\n"
            "bad = sorted(m for m in sys.modules if m.startswith(("
            "'torch', 'src.robot.drivers', 'src.robot.grasping.loop', 'src.camera', "
            "'src.models', 'pyrealsense2')))\n"
            "print(','.join(bad))\n"
        )
        self.assertEqual("", out)

    def test_importing_the_package_names_the_locator_without_loading_it(self) -> None:
        """Fails where the package does not name the locator. The laziness is the other half: loading the locator
        brings the multi-view association and scipy (527 modules), which a console peek through
        ``perception.viewfinder`` pays for on the package import unless the names resolve on first use."""
        out = _probe(
            "import src.robot.perception as perception\n"
            "print('Locator' in perception.__all__, 'src.robot.perception.locator' in sys.modules)\n"
        )
        self.assertEqual("True False", out)


def _probe(body: str) -> str:
    """What a fresh interpreter prints after ``body``: this process has long imported the grasping stack."""
    out = subprocess.run([sys.executable, "-c", "import sys\n" + body], cwd=_REPO, capture_output=True, text=True,
                         timeout=300)
    if out.returncode != 0:
        raise AssertionError(f"the probe exited {out.returncode}:\n{out.stderr[-2000:]}")
    return out.stdout.strip()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
