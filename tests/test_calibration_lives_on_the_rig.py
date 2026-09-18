"""A camera's calibration lives on its rig, `camera.cameras.rigs[<id>].extrinsics`, and nothing else names it.

The fusion calibration keys, `grasping.fusion.extrinsics_artifact_path` and each `grasping.fusion.cameras.<id>`'s
`mounting_mode` and `extrinsics_artifact_path`, are not in the schema and are refused at load with a sentence
naming the rig key; `fusion.cameras.<id>` carries `enabled` only. The owner decisions behind this are recorded in
`.commits/robot/52-one-owner-per-camera.md`. The primary's resolver follows its rig whether or not
`fusion.enabled` is on, so a wrist primary gets the eye-in-hand resolver that a single artifact path cannot
describe. Every artifact is opened by one function, `RigCalibration.from_config`.
"""

from __future__ import annotations

import ast
import datetime as dt
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from pydantic import ValidationError

from src.config import ConfigError, load_config
from src.config.loader import load_robot_section
from src.config.schema.app import AppConfig
from src.config.schema.robot import RobotConfig
from src.calibration.extrinsics import Extrinsics
from src.calibration.serialization import save_cam_to_tool, save_extrinsics
from src.geometry import Frame, Transform
from src.robot.execution.autonomous_grasp import AutonomousGraspService
from src.robot.execution.autonomous_grasp.builders import build_config_frame_resolver
from src.robot.execution.real_cell.preflight import CheckStatus, run_config_preflight
from src.robot.grasping.motion.frame_resolver import EyeInHandFrameResolver, StaticCameraToBaseResolver
from tests.test_grasping_config_wiring import _calc_and_perception
from tests.test_section_loaders import _ScratchTree

_ROOT = Path(__file__).resolve().parents[1]
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"
_T = np.array([400.0, -25.0, 812.5])
_Q = np.array([1.0, 0.0, 0.0, 0.0])
_GRASPING = {"default_mode": "auto", "max_attempts": 5}


def _robot(vendor: str) -> RobotConfig:
    gripper = {"vendor": "none", "model": "robotiq_2f85"} if vendor == "ur" else {"vendor": "none"}
    return RobotConfig(vendor=vendor, gripper=gripper, grasping=dict(_GRASPING))  # type: ignore[arg-type]


def _saved_fixed(folder: Path, rig_id: str = "overhead") -> str:
    return str(save_extrinsics(folder / f"eth_{rig_id}.json", Extrinsics(
        transform=Transform(translation_mm=_T, quaternion_xyzw=_Q, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        rmse_mm=0.8, max_error_mm=1.5, num_samples=22,
        captured_at=dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc), rig_id=rig_id)))


def _saved_wrist(folder: Path, rig_id: str = "wrist") -> str:
    return str(save_cam_to_tool(folder / f"eih_{rig_id}.json", Transform(
        translation_mm=_T, quaternion_xyzw=_Q, from_frame=Frame.CAMERA, to_frame=Frame.TOOL), rig_id=rig_id))


def _camera_section(primary: str, extrinsics: dict | None) -> Any:
    """The shipped camera section with its RGB-D rig renamed to ``primary``, switched on, and given ``extrinsics``."""
    shipped = load_config().camera
    data = shipped.model_dump(mode="json")
    rig = next(r for r in data["cameras"]["rigs"] if r["source"] == "rgbd")
    rig.update({"rig_id": primary, "enabled": True, "extrinsics": extrinsics})
    data["cameras"]["primary_rig_id"] = primary
    return type(shipped).model_validate(data)


class TheOldKeysAreRefusedAtLoadTests(_ScratchTree):

    def _refusal(self, fusion: dict) -> str:
        path = self.root / "robot" / "robot.yaml"
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        data["robot"].setdefault("grasping", {}).setdefault("fusion", {}).update(fusion)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        with self.assertRaises(ConfigError) as caught:
            load_robot_section(self.root, profile=None)
        return str(caught.exception)

    def test_a_tree_still_writing_the_fusion_artifact_is_refused_naming_the_rig_key(self) -> None:
        message = self._refusal({"extrinsics_artifact_path": "calibration/eth_overhead.json"})
        self.assertIn("removed on purpose", message)
        self.assertIn("camera.cameras.rigs[<id>].extrinsics", message)

    def test_a_fused_camera_still_writing_its_artifact_is_refused_naming_the_rig_key(self) -> None:
        message = self._refusal({"cameras": {"overhead": {
            "mounting_mode": "eye_to_hand", "extrinsics_artifact_path": "calibration/eth_overhead.json"}}})
        self.assertIn("robot.grasping.fusion.cameras.overhead.mounting_mode", message)
        self.assertIn("robot.grasping.fusion.cameras.overhead.extrinsics_artifact_path", message)
        self.assertEqual(message.count("removed on purpose"), 2)
        self.assertIn("camera.cameras.rigs[<id>].extrinsics", message)


class ThePrimaryResolverFollowsItsRigTests(unittest.TestCase):

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def test_the_primary_resolver_reads_the_rig(self) -> None:
        camera = _camera_section("overhead", {"mounting_mode": "eye_to_hand", "artifact_path": _saved_fixed(self.root)})
        grasping = _robot("dummy").grasping
        self.assertFalse(grasping.fusion.enabled, "the resolver must not wait for fusion to be switched on")
        resolver = build_config_frame_resolver(grasping, camera=camera)
        assert isinstance(resolver, StaticCameraToBaseResolver)
        np.testing.assert_allclose(resolver.transform.translation_mm, _T)
        np.testing.assert_allclose(resolver.transform.quaternion_xyzw, _Q)

    def test_an_uncalibrated_primary_and_a_missing_section_give_no_resolver(self) -> None:
        grasping = _robot("dummy").grasping
        self.assertIsNone(build_config_frame_resolver(grasping, camera=_camera_section("overhead", None)))
        self.assertIsNone(build_config_frame_resolver(grasping))

    def test_a_real_cell_whose_primary_is_a_wrist_rig_builds(self) -> None:
        camera = _camera_section("wrist", {
            "mounting_mode": "eye_in_hand", "artifact_path": _saved_wrist(self.root),
            "shutter_motion_tolerance_mm": 2.0, "shutter_motion_tolerance_deg": 0.5})
        calc, perc = _calc_and_perception()
        with patch(_READY):
            service = AutonomousGraspService.from_robot_config(
                _robot("ur"), calculator=calc, perception=perc, camera=camera)  # type: ignore[arg-type]
        resolver = service.runtime.orchestrator.frame_resolver
        assert isinstance(resolver, EyeInHandFrameResolver)
        np.testing.assert_allclose(resolver.t_cam_to_tool.translation_mm, _T)

    def test_a_real_cell_without_a_rig_calibration_names_the_rig_key(self) -> None:
        calc, perc = _calc_and_perception()
        with patch(_READY), self.assertRaises(ValueError) as caught:
            AutonomousGraspService.from_robot_config(_robot("ur"), calculator=calc, perception=perc)  # type: ignore[arg-type]
        message = str(caught.exception)
        self.assertIn("camera.cameras.rigs[", message)
        self.assertIn("].extrinsics", message)
        self.assertNotIn("extrinsics_artifact_path", message)

    def test_a_dummy_cell_with_no_camera_section_still_builds(self) -> None:
        """The control: a desk cell passes its resolver in code or needs none, and builds without a camera section."""
        calc, perc = _calc_and_perception()
        self.assertIsNotNone(AutonomousGraspService.from_robot_config(
            _robot("dummy"), calculator=calc, perception=perc))  # type: ignore[arg-type]

    def test_the_resolver_and_the_camera_agree_on_one_artifact(self) -> None:
        from src.camera.orchestration.camera import Camera

        camera = _camera_section("overhead", {"mounting_mode": "eye_to_hand", "artifact_path": _saved_fixed(self.root)})
        rig = next(r for r in camera.cameras.rigs if r.rig_id == "overhead")
        resolver = build_config_frame_resolver(_robot("dummy").grasping, camera=camera)
        assert isinstance(resolver, StaticCameraToBaseResolver)
        owner = Camera.from_rig(rig, streamer=object()).calibration().camera_to_base()
        for transform in (resolver.transform, owner):
            np.testing.assert_allclose(transform.translation_mm, _T)
            np.testing.assert_allclose(transform.quaternion_xyzw, _Q)


class OneLoaderOpensEveryArtifactTests(unittest.TestCase):
    _LOADERS = {"load_extrinsics", "load_cam_to_tool", "load_cam_to_tool_artifact"}
    _ONLY = "src/calibration/rig_calibration.py"

    @classmethod
    def _calls(cls, source: str) -> list[int]:
        return [node.lineno for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.Call)
                and ((isinstance(node.func, ast.Name) and node.func.id in cls._LOADERS)
                     or (isinstance(node.func, ast.Attribute) and node.func.attr in cls._LOADERS))]

    def test_every_calibration_read_goes_through_the_rig_loader(self) -> None:
        offenders = []
        for path in sorted((_ROOT / "src").rglob("*.py")):
            relative = path.relative_to(_ROOT).as_posix()
            if relative != self._ONLY:
                offenders += [f"{relative}:{line}" for line in self._calls(path.read_text(encoding="utf-8"))]
        self.assertEqual(offenders, [], "an artifact is opened outside RigCalibration.from_config")
        self.assertTrue(self._calls((_ROOT / self._ONLY).read_text(encoding="utf-8")), "the loader opens nothing")

    def test_the_scan_sees_a_call(self) -> None:
        """The control: the detector fires on the construction it is written to catch."""
        self.assertEqual(self._calls("from x import load_extrinsics\nload_extrinsics('a.json')\n"), [2])


class AFusedCameraNamesARigTests(unittest.TestCase):

    def test_a_fused_camera_that_names_no_rig_is_refused_at_load(self) -> None:
        tree = load_config().model_dump(mode="json")
        tree["robot"]["grasping"]["fusion"]["cameras"] = {"ghost": {"enabled": True}}
        with self.assertRaises(ValidationError) as caught:
            AppConfig.model_validate(tree)
        self.assertIn("robot.grasping.fusion.cameras names 'ghost', which is not a rig in camera.cameras.rigs",
                      str(caught.exception))

    def test_a_fused_camera_that_names_a_rig_loads(self) -> None:
        """The control, and the whole of what `fusion.cameras.<id>` carries: `enabled`."""
        tree = load_config().model_dump(mode="json")
        rig_id = tree["camera"]["cameras"]["rigs"][0]["rig_id"]
        tree["robot"]["grasping"]["fusion"]["cameras"] = {rig_id: {"enabled": True}}
        AppConfig.model_validate(tree)


class ThePreflightReadsTheRigTests(unittest.TestCase):

    @staticmethod
    def _row(report: Any) -> Any:
        return next(check for check in report.checks if check.name == "camera -> base")

    def test_the_preflight_row_names_the_rig_key(self) -> None:
        camera = _camera_section("overhead", {"mounting_mode": "eye_to_hand", "artifact_path": "calibration/eth_overhead.json"})
        row = self._row(run_config_preflight(_robot("ur"), camera=camera, curobo_available=True))
        self.assertEqual(row.status, CheckStatus.OK)
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", row.detail)
        self.assertIn("calibration/eth_overhead.json", row.detail)

    def test_an_uncalibrated_primary_blocks_a_real_cell_naming_the_rig_key(self) -> None:
        row = self._row(run_config_preflight(_robot("ur"), camera=_camera_section("overhead", None),
                                             curobo_available=True))
        self.assertEqual(row.status, CheckStatus.BLOCK)
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", row.detail + row.fix)
        self.assertNotIn("extrinsics_artifact_path", row.detail + row.fix)

    def test_a_preflight_without_the_camera_section_says_so(self) -> None:
        row = self._row(run_config_preflight(_robot("ur"), curobo_available=True))
        self.assertEqual(row.status, CheckStatus.BLOCK)
        self.assertIn("was not handed the camera section", row.detail)
        desk = self._row(run_config_preflight(_robot("dummy"), curobo_available=True))
        self.assertEqual(desk.status, CheckStatus.WARN)


class TheMultiviewRunnerLoadsItsCalibrationThroughTheRigLoaderTests(unittest.TestCase):
    """`run_multiview_pick --calibrated`: the sim profile carries no fusion calibration map, so the runner reads
    each camera's `eth_<id>.json` in code, through the one loader, and refuses a camera its mode needs and cannot
    load."""

    def test_the_multiview_runner_builds_calibrated_resolvers_without_fusion_keys(self) -> None:
        from src.willy_sim.run_multiview_pick import calibrated_camera_resolvers

        with tempfile.TemporaryDirectory() as folder:
            _saved_fixed(Path(folder), "overhead")
            resolvers = calibrated_camera_resolvers(["overhead"], artifact_dir=folder)
        assert isinstance(resolvers["overhead"], StaticCameraToBaseResolver)
        np.testing.assert_allclose(resolvers["overhead"].transform.translation_mm, _T)

    def test_a_camera_without_its_artifact_is_refused_naming_the_file(self) -> None:
        from src.calibration.rig_calibration import RigCalibrationError
        from src.willy_sim.run_multiview_pick import calibrated_camera_resolvers

        with tempfile.TemporaryDirectory() as folder, self.assertRaises(RigCalibrationError) as caught:
            calibrated_camera_resolvers(["oblique_L"], artifact_dir=folder)
        self.assertIn("eth_oblique_L.json", str(caught.exception))

    def test_the_runner_reads_where_the_calibration_runner_writes(self) -> None:
        """`run_eth_calibrate` saves into `calibration_dir(sim.robot_model)`, so that is where the runner looks.

        The control: an artifact left only in the shared root is not read for a named robot model.
        """
        from src.calibration.rig_calibration import RigCalibrationError
        from src.willy_sim.calibration import paths
        from src.willy_sim.run_multiview_pick import calibrated_camera_resolvers

        with tempfile.TemporaryDirectory() as root, patch.object(paths, "CALIBRATION_ROOT", Path(root)):
            (Path(root) / "ur5e").mkdir()
            _saved_fixed(Path(root) / "ur5e", "overhead")
            resolvers = calibrated_camera_resolvers(["overhead"], robot_model="ur5e")
            np.testing.assert_allclose(resolvers["overhead"].transform.translation_mm, _T)

            _saved_fixed(Path(root), "oblique_L")
            with self.assertRaises(RigCalibrationError):
                calibrated_camera_resolvers(["oblique_L"], robot_model="ur5e")


if __name__ == "__main__":
    unittest.main()
