from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


from src.calibration import (
    EXTRINSICS_SCHEMA,
    CalibrationDataError,
    Extrinsics,
    ExtrinsicsError,
    QualityBandsMm,
    classify_rmse,
    extrinsics_from_dict,
    load_extrinsics,
    save_extrinsics,
    unit_scaling,
)
from src.geometry import Frame, Transform


class CalibrationCoreTests(unittest.TestCase):
    def test_unit_scaling(self) -> None:
        self.assertEqual(unit_scaling("mm"), 1.0)
        self.assertEqual(unit_scaling("cm"), 0.1)
        self.assertEqual(unit_scaling("m"), 0.001)
        with self.assertRaises(CalibrationDataError):
            unit_scaling("inch")

    def test_quality_bands(self) -> None:
        self.assertEqual(classify_rmse(None), "unknown")
        self.assertEqual(classify_rmse(float("nan")), "unknown")
        self.assertEqual(classify_rmse(-0.1), "unknown")
        self.assertEqual(classify_rmse(1.0), "excellent")
        self.assertEqual(classify_rmse(2.5), "good")
        self.assertEqual(classify_rmse(5.0), "marginal")
        self.assertEqual(classify_rmse(5.1), "poor")
        with self.assertRaises(ValueError):
            QualityBandsMm(excellent=2.0, good=1.0, marginal=5.0)

    def test_extrinsics_frame_validation_and_roundtrip(self) -> None:
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        extrinsics = Extrinsics.from_solver(
            transform=transform,
            rmse_mm=0.8,
            max_error_mm=1.0,
            num_samples=6,
            rig_id="rig-0",
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "extrinsics.json"
            save_extrinsics(path, extrinsics)
            loaded = load_extrinsics(path)

        self.assertEqual(loaded.schema_version, EXTRINSICS_SCHEMA)
        self.assertEqual(loaded.transform, transform)
        self.assertEqual(loaded.quality, "excellent")

    def test_extrinsics_rejects_invalid_frame_direction(self) -> None:
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
        with self.assertRaises(ExtrinsicsError):
            Extrinsics.from_solver(
                transform=transform,
                rmse_mm=1.0,
                max_error_mm=1.0,
                num_samples=4,
                rig_id="rig-0",
            )

    def test_extrinsics_schema_mismatch_rejected(self) -> None:
        transform = Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE)
        extrinsics = Extrinsics.from_solver(
            transform=transform,
            rmse_mm=1.0,
            max_error_mm=1.0,
            num_samples=4,
            rig_id="rig-0",
        )
        payload = json.loads(json.dumps({
            "schema": "willy.calibration.extrinsics/0",
            "transform": {
                "schema": "willy.geometry.transform/1",
                "from_frame": "camera",
                "to_frame": "base",
                "translation_mm": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "rmse_mm": extrinsics.rmse_mm,
            "max_error_mm": extrinsics.max_error_mm,
            "num_samples": extrinsics.num_samples,
            "captured_at": extrinsics.captured_at.isoformat(),
            "rig_id": extrinsics.rig_id,
            "quality": extrinsics.quality,
        }))
        with self.assertRaises(ExtrinsicsError):
            extrinsics_from_dict(payload)

    def test_extrinsics_nested_transform_schema_mismatch_rejected(self) -> None:
        payload = {
            "schema": EXTRINSICS_SCHEMA,
            "transform": {
                "schema": "willy.geometry.transform/0",
                "from_frame": "camera",
                "to_frame": "base",
                "translation_mm": [0, 0, 0],
                "quaternion_xyzw": [0, 0, 0, 1],
            },
            "rmse_mm": 1.0,
            "max_error_mm": 1.0,
            "num_samples": 4,
            "captured_at": "2026-05-11T00:00:00+00:00",
            "rig_id": "rig-0",
            "quality": "excellent",
        }
        with self.assertRaises(ExtrinsicsError):
            extrinsics_from_dict(payload)


if __name__ == "__main__":
    unittest.main()