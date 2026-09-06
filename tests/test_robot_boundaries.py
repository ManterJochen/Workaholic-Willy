from __future__ import annotations

import importlib
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot import WorkspaceLimitsConfig
from src.calibration.eye_hand import MountingMode
from src.geometry import Frame, Pose, Transform, rotation_vector_to_matrix
from src.robot.core import JointPositions, RobotArm, RobotConnectionError, RobotVendor
from src.robot.core.motion_result import MotionCommand, MotionResult
from src.robot.drivers import available_vendors, create_arm
from src.robot.execution.calibration import CalibrationMode, CalibrationRoutine
from src.robot.grippers import create_gripper
from src.robot.grasping import GraspCalculator, GraspFrame


def _transform_matrix(rvec: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_vector_to_matrix(rvec)
    matrix[:3, 3] = np.asarray(translation_mm, dtype=np.float64)
    return matrix


def _inverse(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _sample_tool_pose(index: int) -> np.ndarray:
    rvecs = [
        np.array([0.25, 0.07, -0.04], dtype=np.float64),
        np.array([-0.05, 0.32, 0.09], dtype=np.float64),
        np.array([0.11, -0.08, 0.29], dtype=np.float64),
    ]
    return _transform_matrix(
        rvecs[index % len(rvecs)] * (1.0 + 0.2 * index),
        np.array(
            [120.0 * index, (-1) ** index * 40.0 * index, 100.0 + 30.0 * index],
            dtype=np.float64,
        ),
    )


def _synthetic_eye_to_hand_data() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    T_cam_to_base = _transform_matrix(
        np.array([0.18, -0.21, 0.34], dtype=np.float64),
        np.array([320.0, -140.0, 760.0], dtype=np.float64),
    )
    T_tool_to_marker = _transform_matrix(
        np.array([-0.08, 0.13, 0.05], dtype=np.float64),
        np.array([35.0, 20.0, 90.0], dtype=np.float64),
    )
    tool_poses = [_sample_tool_pose(index) for index in range(7)]
    return T_cam_to_base, T_tool_to_marker, tool_poses


def _synthetic_eye_in_hand_data() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Synthesize a consistent eye-in-hand dataset.

    Mirrors :func:`_synthetic_eye_to_hand_data` but with the mounting
    geometry flipped:

    * The camera is bolted to the tool. The calibrator's returned
      ``T_cam_to_tool`` is the unknown we want to recover.
    * A calibration marker is fixed in the world (``T_base_to_marker``
      constant) and observed from each pose.

    The calibrator follows the same convention as
    :class:`EyeInHandCalibrator`'s direct unit tests in
    ``tests/test_eye_hand_workflows.py``; the camera observation from
    tool pose ``i`` is::

        T_cam_to_marker_i = inv(T_cam_to_tool) @ inv(T_base_to_tool_i) @ T_base_to_marker

    Returns ``(T_cam_to_tool_truth, T_base_to_marker, [T_base_to_tool_i])``.
    """

    # Truth value chosen directly in the calibrator's output convention.
    T_cam_to_tool = _transform_matrix(
        np.array([0.11, -0.07, 0.22], dtype=np.float64),
        np.array([25.0, -10.0, 80.0], dtype=np.float64),
    )
    T_base_to_marker = _transform_matrix(
        np.array([0.05, 0.18, -0.04], dtype=np.float64),
        np.array([400.0, -50.0, 250.0], dtype=np.float64),
    )
    tool_poses = [_sample_tool_pose(index) for index in range(7)]
    return T_cam_to_tool, T_base_to_marker, tool_poses


class FakeArm:
    def __init__(self) -> None:
        self.current_pose: Pose | None = None
        self.settle_calls: list[float] = []

    def move_to(self, pose: Pose, **_: object) -> bool:
        self.current_pose = pose
        return True

    def move(self, pose: Pose, **_: object) -> MotionResult:
        self.current_pose = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def get_tcp_pose(self) -> Pose:
        if self.current_pose is None:
            raise RuntimeError("FakeArm has not moved yet")
        return self.current_pose

    def wait_until_steady(self, timeout_s: float) -> None:
        self.settle_calls.append(float(timeout_s))


class FakeProvider:
    def get_stereo_rig_index(self, rig_id: str) -> int:
        if rig_id != "rig_0":
            raise KeyError(rig_id)
        return 0

    def grab_rectified(self, rig_id: str) -> SimpleNamespace:
        del rig_id
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        return SimpleNamespace(left=image, right=image.copy())


class RobotImportBoundaryTests(unittest.TestCase):
    def test_robot_import_surfaces_do_not_require_removed_packages(self) -> None:
        modules = [
            "src.robot",
            "src.robot.execution",
            "src.robot.execution.calibration",
            "src.robot.grasping",
            "src.robot.grasping.generation.calculator",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                importlib.import_module(module_name)

    def test_stale_robot_dependency_paths_are_not_reintroduced(self) -> None:
        root = Path(__file__).resolve().parents[1]
        checked = [
            root / "src" / "robot" / "execution" / "calibration.py",
            root / "src" / "robot" / "grasping" / "generation" / "calculator.py",
        ]
        forbidden = [
            "src.camera.calibration",
            "src.pipelines.camera.orchestration",
            "src.models.segmentation",
        ]

        for path in checked:
            text = path.read_text(encoding="utf-8")
            for phrase in forbidden:
                with self.subTest(path=path.name, phrase=phrase):
                    self.assertNotIn(phrase, text)

    def test_vendor_sdk_imports_stay_inside_vendor_packages(self) -> None:
        root = Path(__file__).resolve().parents[1]
        patterns = [
            re.compile(r"^\s*(?:import|from)\s+rtde_control\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+rtde_receive\b", re.MULTILINE),
            re.compile(r"^\s*from\s+robotiq_gripper\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+rclpy\b", re.MULTILINE),
            # Isaac SDK (isaacsim/omni) + Franka (franky) — the most import-safety-sensitive drivers
            # (CLAUDE.md priority 1: Isaac imports stay lazy, inside the sim/willy_sim boundary).
            re.compile(r"^\s*(?:import|from)\s+isaacsim\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+omni\b", re.MULTILINE),
            re.compile(r"^\s*(?:import|from)\s+franky\b", re.MULTILINE),
        ]
        allowed = {
            "src/robot/drivers/ur/connection.py",
            "src/robot/grippers/robotiq.py",
        }
        # Vendor SDKs that legitimately live across a whole package: Isaac inside the sim driver, the
        # sim-only grippers, and the willy_sim integration layer; rclpy inside ros2/; franky inside franka/.
        allowed_prefixes = (
            "src/robot/drivers/sim/",
            "src/robot/grippers/sim/",
            "src/willy_sim/",
            "src/robot/drivers/ros2/",
            "src/robot/drivers/franka/",
        )
        offenders: list[str] = []

        for path in (root / "backend").rglob("*.py"):
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8")
            if (any(pattern.search(text) for pattern in patterns)
                    and rel not in allowed and not rel.startswith(allowed_prefixes)):
                offenders.append(rel)

        self.assertEqual(offenders, [])


class RobotRegistryBoundaryTests(unittest.TestCase):
    def test_dummy_arm_factory_returns_robot_arm_without_config(self) -> None:
        arm = create_arm(RobotVendor.DUMMY, dof=7, config=object())

        self.assertIsInstance(arm, RobotArm)
        self.assertIn(RobotVendor.DUMMY.value, available_vendors())
        arm.connect()
        target = Pose.identity(Frame.BASE, label="target")
        arm.move_linear(target)
        self.assertEqual(arm.get_tcp_pose(), target)
        joints = arm.get_joint_positions()
        self.assertIsInstance(joints, JointPositions)
        self.assertEqual(len(joints), 7)
        arm.disconnect()

    def test_reserved_unregistered_vendor_fails_clearly(self) -> None:
        with self.assertRaises(RobotConnectionError):
            create_arm(RobotVendor.FRANKA)

    def test_dummy_and_null_grippers_are_pure_python_protocols(self) -> None:
        dummy = create_gripper("dummy", min_width_mm=1.0, max_width_mm=20.0)
        dummy.connect()
        dummy.activate()
        dummy.set_width_mm(12.5)
        self.assertAlmostEqual(dummy.get_width_mm(), 12.5)
        dummy.disconnect()

        null = create_gripper("none")
        null.connect()
        null.set_width_mm(99.0)
        self.assertEqual(null.get_width_mm(), 0.0)


class GraspCalculatorBoundaryTests(unittest.TestCase):
    def _segmentation_fixture(self) -> tuple[SimpleNamespace, np.ndarray]:
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[7:17, 6:18] = 1
        depth = np.full((24, 24), 1000.0, dtype=np.float64)
        return SimpleNamespace(mask=mask, label="box"), depth

    def _calculator(self) -> GraspCalculator:
        return GraspCalculator(
            min_grip_width_mm=1.0,
            max_grip_width_mm=200.0,
            max_candidates=3,
            camera_matrix=np.array(
                [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
        )

    def test_compute_accepts_segmentation_like_object_and_typed_transform(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        T_cam_to_base = np.eye(4, dtype=np.float64)
        T_cam_to_base[:3, 3] = np.array([100.0, 200.0, 300.0], dtype=np.float64)
        transform = Transform.from_matrix(
            T_cam_to_base,
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )

        grasps = self._calculator().compute(
            segmentation,
            depth,
            camera_to_base=transform,
            pixel_to_mm=5.0,
            unit="mm",
        )

        self.assertTrue(grasps)
        self.assertTrue(all(grasp.frame is GraspFrame.BASE for grasp in grasps))

    def test_compute_without_transform_returns_camera_frame_only(self) -> None:
        segmentation, depth = self._segmentation_fixture()

        grasps = self._calculator().compute(
            segmentation,
            depth,
            pixel_to_mm=5.0,
            unit="mm",
        )

        self.assertTrue(grasps)
        self.assertTrue(all(grasp.frame is GraspFrame.CAMERA for grasp in grasps))

    def test_compute_rejects_bad_transform_frame_or_matrix(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        bad_frame = Transform.identity(from_frame=Frame.BASE, to_frame=Frame.CAMERA)
        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                camera_to_base=bad_frame,
                pixel_to_mm=5.0,
            )

        bad_matrix = np.eye(4, dtype=np.float64)
        bad_matrix[3, 3] = 2.0
        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                T_cam_to_base=bad_matrix,
                pixel_to_mm=5.0,
            )

    def test_compute_rejects_ambiguous_transform_arguments(self) -> None:
        segmentation, depth = self._segmentation_fixture()
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)

        with self.assertRaises(ValueError):
            self._calculator().compute(
                segmentation,
                depth,
                T_cam_to_base=transform,
                camera_to_base=transform,
                pixel_to_mm=5.0,
            )


class CalibrationRoutineBoundaryTests(unittest.TestCase):
    def test_calibration_mode_alias_maps_to_mounting_mode(self) -> None:
        self.assertIs(CalibrationMode.EYE_TO_HAND, MountingMode.EYE_TO_HAND)
        self.assertIs(CalibrationMode.EYE_IN_HAND, MountingMode.EYE_IN_HAND)

    def test_routine_uses_robot_arm_and_modern_eye_to_hand_calibrator(self) -> None:
        T_cam_to_base, T_tool_to_marker, T_base_to_tool_list = _synthetic_eye_to_hand_data()
        poses = [
            Pose.from_matrix(T_base_to_tool, frame=Frame.BASE, label=f"pose_{index}")
            for index, T_base_to_tool in enumerate(T_base_to_tool_list)
        ]
        marker_poses = iter(
            _inverse(T_cam_to_base) @ T_base_to_tool @ T_tool_to_marker
            for T_base_to_tool in T_base_to_tool_list
        )
        settings = SimpleNamespace(
            mode="eye_to_hand",
            min_samples=4,
            min_distance_mm=0.0,
            min_angle=0.0,
            min_angle_deg=0.0,
        )
        routine = CalibrationRoutine(
            arm=FakeArm(),
            provider=FakeProvider(),
            stereo=SimpleNamespace(),
            workspace_limits=WorkspaceLimitsConfig(
                x_min=-2000.0,
                x_max=2000.0,
                y_min=-2000.0,
                y_max=2000.0,
                z_min=-2000.0,
                z_max=2000.0,
            ),
            eth_settings=settings,
            calibration_mode="eye_to_hand",
            settle_time_s=0.0,
        )
        routine._capture_marker_pose = lambda: next(marker_poses)  # type: ignore[method-assign]

        result = routine._execute(poses, T_tool_to_marker=None, dataset_save_path=None)

        self.assertEqual(result.mode, MountingMode.EYE_TO_HAND)
        self.assertIsNotNone(result.extrinsics)
        self.assertIsNone(result.T_cam_to_tool)
        self.assertIsNotNone(result.T_cam_to_base)
        self.assertTrue(np.allclose(result.T_cam_to_base, T_cam_to_base, atol=1e-7))
        self.assertEqual(result.num_samples, len(poses))

    def test_routine_uses_robot_arm_and_modern_eye_in_hand_calibrator(self) -> None:
        """Phase Q.A: Eye-in-Hand mode runs end-to-end through the routine.

        Complements the existing eye-to-hand test by proving the Phase N
        UR-decoupling holds for both mounting modes. The routine must
        accept a vendor-neutral :class:`FakeArm`, drive every pose via
        the typed ``move()`` surface, read TCP through ``get_tcp_pose``,
        and surface ``T_cam_to_tool`` on the result --- with no UR
        symbols imported anywhere along the path.
        """

        T_cam_to_tool_truth, T_base_to_marker, T_base_to_tool_list = (
            _synthetic_eye_in_hand_data()
        )
        poses = [
            Pose.from_matrix(T_base_to_tool, frame=Frame.BASE, label=f"pose_{index}")
            for index, T_base_to_tool in enumerate(T_base_to_tool_list)
        ]
        # The camera observes the world-fixed marker from each tool pose.
        # See :func:`_synthetic_eye_in_hand_data` for the geometry.
        marker_poses = iter(
            _inverse(T_cam_to_tool_truth) @ _inverse(T_base_to_tool) @ T_base_to_marker
            for T_base_to_tool in T_base_to_tool_list
        )
        settings = SimpleNamespace(
            mode="eye_in_hand",
            min_samples=4,
            min_distance_mm=0.0,
            min_angle=0.0,
            min_angle_deg=0.0,
        )
        routine = CalibrationRoutine(
            arm=FakeArm(),
            provider=FakeProvider(),
            stereo=SimpleNamespace(),
            workspace_limits=WorkspaceLimitsConfig(
                x_min=-2000.0,
                x_max=2000.0,
                y_min=-2000.0,
                y_max=2000.0,
                z_min=-2000.0,
                z_max=2000.0,
            ),
            eth_settings=settings,
            calibration_mode="eye_in_hand",
            settle_time_s=0.0,
        )
        routine._capture_marker_pose = lambda: next(marker_poses)  # type: ignore[method-assign]

        result = routine._execute(poses, T_tool_to_marker=None, dataset_save_path=None)

        self.assertEqual(result.mode, MountingMode.EYE_IN_HAND)
        # EIH solves for T_cam_to_tool, not T_cam_to_base.
        self.assertIsNotNone(result.T_cam_to_tool)
        self.assertIsNone(result.T_cam_to_base)
        # EIH does not produce camera extrinsics in the world frame ---
        # those only make sense for a fixed camera (ETH).
        self.assertIsNone(result.extrinsics)
        # Reconstruction must match the synthesised truth to high precision
        # (numerical only --- no measurement noise in the synthetic fixture).
        self.assertTrue(
            np.allclose(result.T_cam_to_tool, T_cam_to_tool_truth, atol=1e-6),
            msg=(
                "EIH solver did not recover the synthesised T_cam_to_tool.\n"
                f"Expected:\n{T_cam_to_tool_truth}\nGot:\n{result.T_cam_to_tool}"
            ),
        )
        self.assertEqual(result.num_samples, len(poses))


if __name__ == "__main__":
    unittest.main()