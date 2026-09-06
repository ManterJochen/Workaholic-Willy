"""Multi-camera per-camera calibration: CAMERA->TOOL persistence, config map, resolver-map builder."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.config.schema.robot.grasping_schema import (
    CameraExtrinsicsConfig,
    RobotGraspingConfig,
)
from src.calibration import (
    Extrinsics,
    ExtrinsicsError,
    load_cam_to_tool,
    save_cam_to_tool,
    save_extrinsics,
)
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolvers
from src.robot.grasping.motion.frame_resolver import (
    EyeInHandFrameResolver,
    StaticCameraToBaseResolver,
)


def _eth_artifact(path: Path, rig_id: str) -> None:
    ext = Extrinsics.from_solver(
        transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        rmse_mm=1.0, max_error_mm=2.0, num_samples=12, rig_id=rig_id,
    )
    save_extrinsics(path, ext)


def _eih_artifact(path: Path, rig_id: str) -> None:
    save_cam_to_tool(path, Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL), rig_id=rig_id)


def _grasping_cfg(cameras: dict, *, fusion_enabled: bool = True) -> RobotGraspingConfig:
    return RobotGraspingConfig.model_validate({"fusion": {"enabled": fusion_enabled, "cameras": cameras}})


class CamToToolSerializationTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "wrist.json"
            _eih_artifact(p, "wrist")
            t = load_cam_to_tool(p)
        self.assertIs(t.from_frame, Frame.CAMERA)
        self.assertIs(t.to_frame, Frame.TOOL)

    def test_save_rejects_wrong_frames(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ExtrinsicsError):  # CAMERA->BASE is not a cam_to_tool
                save_cam_to_tool(Path(d) / "x.json",
                                 Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE), rig_id="w")

    def test_load_rejects_wrong_schema(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "eth.json"
            _eth_artifact(p, "overhead")  # an Extrinsics artifact, wrong schema for cam_to_tool
            with self.assertRaises(ExtrinsicsError):
                load_cam_to_tool(p)


class CameraExtrinsicsConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = CameraExtrinsicsConfig(extrinsics_artifact_path="x.json")
        self.assertTrue(c.enabled)
        self.assertEqual(c.mounting_mode, "eye_to_hand")


class ResolverMapTests(unittest.TestCase):
    def test_empty_paths_are_byte_identical(self) -> None:
        self.assertEqual(build_config_frame_resolvers(None), {})
        self.assertEqual(build_config_frame_resolvers(_grasping_cfg({}, fusion_enabled=False)), {})
        self.assertEqual(build_config_frame_resolvers(_grasping_cfg({})), {})  # fusion on but no cameras

    def test_builds_per_camera_resolvers_and_skips_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            eth, eih = Path(d) / "overhead.json", Path(d) / "wrist.json"
            _eth_artifact(eth, "overhead")
            _eih_artifact(eih, "wrist")
            cfg = _grasping_cfg({
                "overhead": {"mounting_mode": "eye_to_hand", "extrinsics_artifact_path": str(eth)},
                "wrist": {"mounting_mode": "eye_in_hand", "extrinsics_artifact_path": str(eih)},
                "off": {"enabled": False, "mounting_mode": "eye_to_hand", "extrinsics_artifact_path": str(eth)},
            })
            resolvers = build_config_frame_resolvers(cfg)
        self.assertEqual(set(resolvers), {"overhead", "wrist"})  # disabled 'off' skipped
        self.assertIsInstance(resolvers["overhead"], StaticCameraToBaseResolver)
        self.assertIsInstance(resolvers["wrist"], EyeInHandFrameResolver)

    def test_missing_artifact_fails_closed(self) -> None:
        cfg = _grasping_cfg({
            "overhead": {"mounting_mode": "eye_to_hand", "extrinsics_artifact_path": "nope/missing.json"},
        })
        with self.assertRaises(RuntimeError):
            build_config_frame_resolvers(cfg)


if __name__ == "__main__":
    unittest.main()
