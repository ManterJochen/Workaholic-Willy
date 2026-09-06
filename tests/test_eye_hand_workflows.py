from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.calibration import CalibrationDataError
from src.calibration.eye_hand import (
    EYE_HAND_DATASET_SCHEMA,
    EyeHandCalibrationSettings,
    EyeHandDataset,
    EyeInHandCalibrator,
    EyeToHandCalibrator,
    MountingMode,
)
from src.calibration.solver import HandEyeAXXB
from src.geometry import Frame


def _rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def _transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = translation
    return matrix


def _inverse(matrix: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -matrix[:3, :3].T @ matrix[:3, 3]
    return inverse


def _sample_pose(index: int) -> np.ndarray:
    axes = [
        np.array([1.0, 0.3, 0.2]),
        np.array([0.2, 1.0, 0.4]),
        np.array([0.3, 0.2, 1.0]),
    ]
    angle = 0.25 + 0.17 * index
    translation = np.array(
        [80.0 * index, (-1) ** index * 35.0 * index, 25.0 * index],
        dtype=np.float64,
    )
    return _transform(_rotation(axes[index % len(axes)], angle), translation)


def _synthetic_data() -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    T_cam_to_target = _transform(
        _rotation(np.array([0.4, -0.2, 1.0]), 0.7),
        np.array([120.0, -80.0, 450.0], dtype=np.float64),
    )
    fixed_offset = _transform(
        _rotation(np.array([0.1, 1.0, 0.3]), -0.35),
        np.array([30.0, 40.0, 70.0], dtype=np.float64),
    )
    robot_poses = [_sample_pose(index) for index in range(7)]
    return T_cam_to_target, fixed_offset, robot_poses


class EyeHandWorkflowTests(unittest.TestCase):
    def test_axxb_solver_synthetic_known_transform(self) -> None:
        T_cam_to_base, fixed_offset, T_base_to_tool_list = _synthetic_data()
        T_cam_to_marker_list = [
            _inverse(T_cam_to_base) @ T_base_to_tool @ fixed_offset
            for T_base_to_tool in T_base_to_tool_list
        ]
        A_mats = HandEyeAXXB.relative_motions(T_base_to_tool_list)
        B_mats = HandEyeAXXB.relative_motions(T_cam_to_marker_list)

        solved, rmse = HandEyeAXXB().solve(A_mats, B_mats)

        self.assertLess(rmse, 1e-8)
        self.assertTrue(np.allclose(solved, T_cam_to_base, atol=1e-7))

    def test_eye_to_hand_returns_camera_to_base(self) -> None:
        T_cam_to_base, fixed_offset, T_base_to_tool_list = _synthetic_data()
        calibrator = EyeToHandCalibrator(
            settings=EyeHandCalibrationSettings(
                min_samples=4,
                min_distance_mm=0.0,
                min_angle_deg=0.0,
            )
        )
        for T_base_to_tool in T_base_to_tool_list:
            T_cam_to_marker = _inverse(T_cam_to_base) @ T_base_to_tool @ fixed_offset
            self.assertTrue(calibrator.add_sample(T_base_to_tool, T_cam_to_marker))

        result = calibrator.calibrate()

        self.assertEqual(result.mode, MountingMode.EYE_TO_HAND)
        self.assertEqual(result.transform.from_frame, Frame.CAMERA)
        self.assertEqual(result.transform.to_frame, Frame.BASE)
        self.assertTrue(np.allclose(result.transform.to_matrix(), T_cam_to_base, atol=1e-7))

    def test_eye_in_hand_returns_camera_to_tool(self) -> None:
        T_cam_to_tool, fixed_marker, T_base_to_tool_list = _synthetic_data()
        calibrator = EyeInHandCalibrator(
            settings=EyeHandCalibrationSettings(
                min_samples=4,
                min_distance_mm=0.0,
                min_angle_deg=0.0,
            )
        )
        for T_base_to_tool in T_base_to_tool_list:
            T_cam_to_marker = _inverse(T_cam_to_tool) @ _inverse(T_base_to_tool) @ fixed_marker
            self.assertTrue(calibrator.add_sample(T_base_to_tool, T_cam_to_marker))

        result = calibrator.calibrate()

        self.assertEqual(result.mode, MountingMode.EYE_IN_HAND)
        self.assertEqual(result.transform.from_frame, Frame.CAMERA)
        self.assertEqual(result.transform.to_frame, Frame.TOOL)
        self.assertTrue(np.allclose(result.transform.to_matrix(), T_cam_to_tool, atol=1e-7))

    def test_dataset_persistence_roundtrip_and_schema_rejection(self) -> None:
        T_cam_to_base, fixed_offset, T_base_to_tool_list = _synthetic_data()
        dataset = EyeHandDataset()
        dataset.add(
            T_base_to_tool_list[0],
            _inverse(T_cam_to_base) @ T_base_to_tool_list[0] @ fixed_offset,
            marker_id=12,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "dataset.json"
            dataset.save(path)
            loaded = EyeHandDataset.load(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["schema"] = "willy.calibration.eye_hand.dataset/0"
            bad_path = Path(tmp_dir) / "bad_dataset.json"
            bad_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(CalibrationDataError):
                EyeHandDataset.load(bad_path)

        self.assertEqual(EYE_HAND_DATASET_SCHEMA, "willy.calibration.eye_hand.dataset/1")
        self.assertEqual(len(loaded), 1)
        sample = next(loaded.iter_samples())
        self.assertEqual(sample.marker_id, 12)


if __name__ == "__main__":
    unittest.main()