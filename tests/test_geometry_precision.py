from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

import numpy as np

from src.geometry import (
    Frame,
    Pose,
    Transform,
    canonicalise,
    from_rotation_matrix,
    matrix_to_rotation_vector,
    rotation_vector_to_matrix,
    to_rotation_matrix,
    transform_from_dict,
    transform_to_dict,
)
from src.geometry.validation import validate_homogeneous_matrix
from src.utility.unit_scaling import unit_scaling


def _transform_matrix(rvec: np.ndarray, translation_mm: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_vector_to_matrix(rvec)
    matrix[:3, 3] = np.asarray(translation_mm, dtype=np.float64)
    return matrix


class GeometryPrecisionTests(unittest.TestCase):
    def test_quaternion_normalization_and_matrix_roundtrip(self) -> None:
        raw = np.array([0.2, -0.3, 0.4, -0.5], dtype=np.float64)
        quat = canonicalise(raw)
        self.assertAlmostEqual(float(np.linalg.norm(quat)), 1.0, places=15)
        self.assertGreaterEqual(float(quat[3]), 0.0)

        rotation = to_rotation_matrix(quat)
        roundtrip = from_rotation_matrix(rotation)
        self.assertTrue(np.allclose(to_rotation_matrix(roundtrip), rotation, atol=1e-12))

    def test_rotation_vector_matrix_roundtrip(self) -> None:
        rvec = np.array([0.37, -0.22, 0.58], dtype=np.float64)
        rotation = rotation_vector_to_matrix(rvec)
        roundtrip = matrix_to_rotation_vector(rotation)
        self.assertTrue(np.allclose(rotation_vector_to_matrix(roundtrip), rotation, atol=1e-12))

    def test_homogeneous_validation_rejects_shear(self) -> None:
        matrix = np.eye(4, dtype=np.float64)
        matrix[0, 1] = 0.01
        with self.assertRaises(Exception):
            validate_homogeneous_matrix(matrix)

    def test_transform_inverse_and_composition_identity(self) -> None:
        matrix = _transform_matrix(
            np.array([0.18, -0.41, 0.29], dtype=np.float64),
            np.array([123.456789, -42.25, 987.000001], dtype=np.float64),
        )
        transform = Transform.from_matrix(matrix, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        identity = transform.compose(transform.inverse())

        self.assertEqual(identity.from_frame, Frame.CAMERA)
        self.assertEqual(identity.to_frame, Frame.CAMERA)
        self.assertTrue(np.allclose(identity.to_matrix(), np.eye(4), atol=1e-10))

    def test_pose_matrix_roundtrip_preserves_translation(self) -> None:
        matrix = _transform_matrix(
            np.array([0.31, 0.04, -0.17], dtype=np.float64),
            np.array([0.123456789, 1000.000000001, -250.333333333], dtype=np.float64),
        )
        pose = Pose.from_matrix(matrix, frame=Frame.BASE, label="tcp")
        roundtrip = Pose.from_matrix(pose.to_matrix(), frame=Frame.BASE, label="tcp")

        self.assertTrue(np.allclose(roundtrip.position_mm, pose.position_mm, atol=1e-12))
        self.assertTrue(np.allclose(roundtrip.to_matrix(), matrix, atol=1e-10))

    def test_repeated_transform_roundtrip_drift_stays_below_tolerance(self) -> None:
        original = _transform_matrix(
            np.array([0.23, -0.19, 0.11], dtype=np.float64),
            np.array([540.123456789, -88.765432101, 22.000000009], dtype=np.float64),
        )
        current = original.copy()
        for _ in range(250):
            transform = Transform.from_matrix(current, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
            current = transform.to_matrix()

        self.assertLess(float(np.max(np.abs(current - original))), 1e-10)

    def test_realistic_transform_chain_inverts_to_identity(self) -> None:
        T_cam_to_base = Transform.from_matrix(
            _transform_matrix(np.array([0.12, -0.24, 0.09]), np.array([320.0, -120.0, 850.0])),
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )
        T_base_to_tool = Transform.from_matrix(
            _transform_matrix(np.array([-0.08, 0.31, 0.17]), np.array([640.5, 210.25, 115.75])),
            from_frame=Frame.BASE,
            to_frame=Frame.TOOL,
        )
        chain = T_cam_to_base.compose(T_base_to_tool)
        identity = chain.compose(chain.inverse())

        self.assertTrue(np.allclose(identity.to_matrix(), np.eye(4), atol=1e-10))

    def test_grasp_pipeline_point_stability(self) -> None:
        matrix = _transform_matrix(
            np.array([0.0, 0.0, np.pi / 2.0], dtype=np.float64),
            np.array([500.0, 100.0, 250.0], dtype=np.float64),
        )
        T_cam_to_base = Transform.from_matrix(matrix, from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        point_camera_mm = np.array([10.0, 20.0, 30.0], dtype=np.float64)

        point_base_mm = T_cam_to_base.apply_point(point_camera_mm)
        expected = matrix[:3, :3] @ point_camera_mm + matrix[:3, 3]
        self.assertTrue(np.allclose(point_base_mm, expected, atol=1e-12))

    def test_transform_serialization_preserves_precision(self) -> None:
        transform = Transform.from_matrix(
            _transform_matrix(
                np.array([0.03, 0.02, -0.04], dtype=np.float64),
                np.array([1.234567890123, -2.345678901234, 3.456789012345], dtype=np.float64),
            ),
            from_frame=Frame.CAMERA,
            to_frame=Frame.BASE,
        )
        payload = json.loads(json.dumps(transform_to_dict(transform)))
        loaded = transform_from_dict(payload)

        self.assertTrue(np.allclose(loaded.to_matrix(), transform.to_matrix(), atol=1e-12))

    def test_unit_scaling_roundtrips_do_not_round(self) -> None:
        value_mm = 1234.567890123
        value_m = value_mm * unit_scaling("m")
        value_cm = value_mm * unit_scaling("cm")

        self.assertAlmostEqual(value_m / unit_scaling("m"), value_mm, places=12)
        self.assertAlmostEqual(value_cm / unit_scaling("cm"), value_mm, places=12)

    def test_scipy_imports_are_not_scattered(self) -> None:
        root = Path(__file__).resolve().parents[1]
        pattern = re.compile(r"^\s*(?:from\s+scipy\b|import\s+scipy\b)")
        offenders: list[str] = []
        for path in (root / "src").rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if pattern.search(text):
                offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()