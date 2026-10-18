"""Multi-camera per-camera calibration: CAMERA->TOOL persistence, the rig's mounting, and the resolver map of the
cameras a cell fuses (every enabled RGB-D rig that declares its calibration, where two or more do)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from src.config import load_config
from src.config.schema.camera.shared_schema import RigExtrinsicsConfig
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

_WRIST_TOLERANCES = {"shutter_motion_tolerance_mm": 2.0, "shutter_motion_tolerance_deg": 0.5}


def _eth_artifact(path: Path, rig_id: str) -> None:
    ext = Extrinsics.from_solver(
        transform=Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        rmse_mm=1.0, max_error_mm=2.0, num_samples=12, rig_id=rig_id,
    )
    save_extrinsics(path, ext)


def _eih_artifact(path: Path, rig_id: str) -> None:
    save_cam_to_tool(path, Transform.identity(from_frame=Frame.CAMERA, to_frame=Frame.TOOL), rig_id=rig_id)


def _camera_section(extrinsics: dict[str, dict], *, disabled: tuple[str, ...] = ()) -> Any:
    """The shipped camera section with one RGB-D rig per id, the first the primary, each declaring its calibration on
    the rig, ``camera.cameras.rigs[<id>].extrinsics``; the ids ``disabled`` names are switched off."""
    shipped = load_config().camera
    data = shipped.model_dump(mode="json")
    template = next(rig for rig in data["cameras"]["rigs"] if rig["source"] == "rgbd")
    for rig_id, block in extrinsics.items():
        data["cameras"]["rigs"].append({**template, "rig_id": rig_id, "enabled": rig_id not in disabled,
                                        "serial_number": f"serial_{rig_id}", "extrinsics": block})
    data["cameras"]["primary_rig_id"] = next(iter(extrinsics))
    return type(shipped).model_validate(data)


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


class TheRigsMountingHasNoDefaultTests(unittest.TestCase):
    def test_a_calibration_without_its_mounting_is_refused(self) -> None:
        """The mounting is declared on the rig, where it has no default, so a camera is never assumed to be fixed."""
        with self.assertRaises(ValidationError):
            RigExtrinsicsConfig.model_validate({"artifact_path": "x.json"})


class ResolverMapTests(unittest.TestCase):
    def test_a_cell_that_fuses_no_camera_gets_no_map(self) -> None:
        """No camera section, the shipped one (no calibrated RGB-D rig), and a calibrated camera alone: its artifact is
        never opened, as one camera has nothing to fuse with."""
        self.assertEqual(build_config_frame_resolvers(), {})
        self.assertEqual(build_config_frame_resolvers(load_config().camera), {})
        alone = _camera_section({"overhead": {"mounting_mode": "eye_to_hand", "artifact_path": "never_opened.json"}})
        self.assertEqual(build_config_frame_resolvers(alone), {})

    def test_builds_a_resolver_for_every_fused_camera_by_its_mounting_and_skips_a_disabled_one(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            eth, eih = Path(d) / "overhead.json", Path(d) / "wrist.json"
            _eth_artifact(eth, "overhead")
            _eih_artifact(eih, "wrist")
            camera = _camera_section({
                "overhead": {"mounting_mode": "eye_to_hand", "artifact_path": str(eth)},
                "wrist": {"mounting_mode": "eye_in_hand", "artifact_path": str(eih), **_WRIST_TOLERANCES},
                # Never written: a disabled camera is skipped before its artifact is opened.
                "off": {"mounting_mode": "eye_to_hand", "artifact_path": str(Path(d) / "never_written.json")},
            }, disabled=("off",))
            resolvers = build_config_frame_resolvers(camera)
        self.assertEqual(set(resolvers), {"overhead", "wrist"})  # disabled 'off' skipped
        self.assertIsInstance(resolvers["overhead"], StaticCameraToBaseResolver)
        self.assertIsInstance(resolvers["wrist"], EyeInHandFrameResolver)

    def test_missing_artifact_fails_closed(self) -> None:
        camera = _camera_section({"overhead": {"mounting_mode": "eye_to_hand", "artifact_path": "nope/missing.json"},
                                  "side": {"mounting_mode": "eye_to_hand", "artifact_path": "nope/missing_too.json"}})
        with self.assertRaises(RuntimeError):
            build_config_frame_resolvers(camera)


if __name__ == "__main__":
    unittest.main()
