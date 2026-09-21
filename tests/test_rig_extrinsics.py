"""CAMERA to BASE is declared per rig in the camera block, and loaded through one function.

The key is `camera.cameras.rigs[<id>].extrinsics`, an owner decision recorded in
`.commits/robot/52-one-owner-per-camera.md`. It names the mounting and the artifact, and for a wrist rig also the
arm motion it tolerates while a frame is taken. None means the rig is not calibrated, a stated state and not a
default transform. Every reader loads it through `RigCalibration`, which lives in the calibration package, beside
nothing the robot may not load.
"""

from __future__ import annotations

import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pydantic import ValidationError

from src.config.schema.camera import RGBDDeviceRigConfig
from src.calibration.extrinsics import Extrinsics
from src.calibration.rig_calibration import RigArtifactMissing, RigCalibration, RigCalibrationError, RigNotCalibrated
from src.calibration.serialization import save_cam_to_tool, save_extrinsics
from src.geometry import Frame, Transform
from tests.test_camera_boundaries import _rgbd_rig, _single_rig, _webcam_rig

_FIXED = {"mounting_mode": "eye_to_hand", "artifact_path": "x.json"}
_WRIST = {"mounting_mode": "eye_in_hand", "artifact_path": "x.json",
          "shutter_motion_tolerance_mm": 2.0, "shutter_motion_tolerance_deg": 0.5}


def _rgbd(extrinsics: dict | None, rig_id: str = "overhead") -> RGBDDeviceRigConfig:
    return RGBDDeviceRigConfig.model_validate(
        {"rig_id": rig_id, "enabled": True, "source": "rgbd", "fps": 30, "extrinsics": extrinsics})


def _transform(to: Frame) -> Transform:
    return Transform(translation_mm=np.array([400.0, -25.0, 812.5]),
                     quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                     from_frame=Frame.CAMERA, to_frame=to)


class TheKeyTests(unittest.TestCase):

    def test_a_rig_may_declare_its_calibration(self) -> None:
        rig = _rgbd(_FIXED)
        assert rig.extrinsics is not None
        self.assertEqual((rig.extrinsics.mounting_mode, rig.extrinsics.artifact_path), ("eye_to_hand", "x.json"))

    def test_a_rig_that_declares_nothing_is_not_calibrated(self) -> None:
        self.assertIsNone(_rgbd_rig("overhead").extrinsics)

    def test_a_block_that_names_no_artifact_is_refused_saying_to_leave_it_out(self) -> None:
        """An empty path is not "not calibrated yet"; no block is, and the refusal says so."""
        for block in ({**_FIXED, "artifact_path": ""}, {**_FIXED, "artifact_path": "   "},
                      {**_FIXED, "artifact_path": None}):
            with self.subTest(block=block), self.assertRaises(ValidationError) as caught:
                _rgbd(block)
            self.assertIn("declares no extrinsics block at all: leave it out", str(caught.exception))
            self.assertIn("--mode eye_to_hand", str(caught.exception))

    def test_a_missing_path_key_stays_pydantics_own_refusal(self) -> None:
        """The refusal is on the field's VALUE: a key left out, or misspelled, is reported as what it is,
        and nobody is told to delete a block that holds a real calibration under a typo."""
        with self.assertRaises(ValidationError) as caught:
            _rgbd({"mounting_mode": "eye_to_hand"})
        self.assertIn("Field required", str(caught.exception))
        self.assertNotIn("declares no extrinsics block at all", str(caught.exception))

    def test_the_mounting_mode_has_no_default(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _rgbd({"artifact_path": "x.json"})
        self.assertIn("mounting_mode", str(caught.exception))

    def test_a_wrist_rig_without_its_shutter_tolerance_is_refused_at_load(self) -> None:
        for missing in ("shutter_motion_tolerance_mm", "shutter_motion_tolerance_deg"):
            with self.subTest(missing=missing), self.assertRaises(ValidationError) as caught:
                _rgbd({key: value for key, value in _WRIST.items() if key != missing})
            self.assertIn("an eye_in_hand rig needs shutter_motion_tolerance_mm and shutter_motion_tolerance_deg",
                          str(caught.exception))
        with self.assertRaises(ValidationError):
            _rgbd({**_WRIST, "shutter_motion_tolerance_mm": 0.0})
        wrist = _rgbd(_WRIST).extrinsics
        assert wrist is not None
        self.assertEqual((wrist.shutter_motion_tolerance_mm, wrist.shutter_motion_tolerance_deg), (2.0, 0.5))

    def test_a_fixed_rig_refuses_a_shutter_tolerance(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            _rgbd({**_FIXED, "shutter_motion_tolerance_mm": 2.0})
        self.assertIn("an eye_to_hand rig does not move with the arm", str(caught.exception))

    def test_a_stereo_rig_refuses_extrinsics_with_its_sentence(self) -> None:
        for rig in (_webcam_rig("pair"), _single_rig("stereo")):
            with self.subTest(source=rig.source), self.assertRaises(ValidationError) as caught:
                type(rig).model_validate({**rig.model_dump(), "extrinsics": _FIXED})
            self.assertIn(
                f"camera.cameras.rigs[{rig.rig_id!r}] is a {rig.source!r} rig; stereo rigs carry no extrinsics "
                "until their own step", str(caught.exception))


class TheLoaderTests(unittest.TestCase):

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _fixed(self) -> RigCalibration:
        path = save_extrinsics(self.root / "eth_overhead.json", Extrinsics(
            transform=_transform(Frame.BASE), rmse_mm=0.8, max_error_mm=1.5, num_samples=22,
            captured_at=dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc), rig_id="overhead"))
        rig = _rgbd({**_FIXED, "artifact_path": str(path)})
        return RigCalibration.from_config("overhead", rig.extrinsics)

    def _wrist(self) -> RigCalibration:
        path = save_cam_to_tool(self.root / "eih_wrist.json", _transform(Frame.TOOL), rig_id="wrist")
        rig = _rgbd({**_WRIST, "artifact_path": str(path)}, rig_id="wrist")
        return RigCalibration.from_config("wrist", rig.extrinsics)

    def test_a_calibrated_fixed_rig_answers_camera_to_base(self) -> None:
        base = self._fixed().camera_to_base()
        self.assertEqual((base.from_frame, base.to_frame), (Frame.CAMERA, Frame.BASE))
        np.testing.assert_allclose(base.translation_mm, [400.0, -25.0, 812.5])
        np.testing.assert_allclose(base.quaternion_xyzw, [1.0, 0.0, 0.0, 0.0])

    def test_a_calibrated_wrist_rig_answers_camera_to_tool_and_its_tolerances(self) -> None:
        wrist = self._wrist()
        tool = wrist.camera_to_tool()
        self.assertEqual((tool.from_frame, tool.to_frame), (Frame.CAMERA, Frame.TOOL))
        np.testing.assert_allclose(tool.translation_mm, [400.0, -25.0, 812.5])
        self.assertEqual((wrist.shutter_motion_tolerance_mm, wrist.shutter_motion_tolerance_deg), (2.0, 0.5))

    def test_a_wrist_rig_refuses_a_fixed_transform(self) -> None:
        with self.assertRaises(RigCalibrationError) as caught:
            self._wrist().camera_to_base()
        self.assertIn("a wrist camera has no fixed CAMERA to BASE; ask its frames for the tool pose at the shutter",
                      str(caught.exception))

    def test_a_fixed_rig_refuses_a_tool_transform(self) -> None:
        with self.assertRaises(RigCalibrationError) as caught:
            self._fixed().camera_to_tool()
        self.assertIn("a fixed camera has no CAMERA to TOOL", str(caught.exception))

    def test_an_artifact_that_does_not_load_names_the_key_and_the_path(self) -> None:
        missing = str(self.root / "never_written.json")
        with self.assertRaises(RigCalibrationError) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": missing}).extrinsics)
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", str(caught.exception))
        self.assertIn(missing, str(caught.exception))

    def test_an_uncalibrated_rig_refuses_naming_the_key(self) -> None:
        with self.assertRaises(RigNotCalibrated) as caught:
            RigCalibration.from_config("overhead", None)
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", str(caught.exception))
        # The command defaults to eye_to_hand, and a wrist camera swept that way writes a CAMERA to BASE
        # file for a camera that moves: the refusal names both modes.
        self.assertIn("--mode eye_to_hand for a fixed camera", str(caught.exception))
        self.assertIn("--mode eye_in_hand for one the arm carries", str(caught.exception))

    def test_a_block_written_before_its_sweep_says_to_run_it_or_leave_the_block_out(self) -> None:
        """⚠ MEASURED 2026-09-21 ON A REAL CELL: the block was written first, the natural order.

        A file that is not there and a file that does not parse need different fixes, and the refusal
        used to give both the same `FileNotFoundError` sentence, which says nothing about a sweep.
        """
        missing = str(self.root / "never_written.json")
        with self.assertRaises(RigArtifactMissing) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": missing}).extrinsics)
        text = str(caught.exception)
        # The observation, not a guess at its cause: a sweep that ran can still look missing from the
        # wrong directory, so the sentence says what was looked for, as written, and where.
        self.assertIn(f"names {missing} (eye_to_hand), which does not load: there is no file at that path", text)
        self.assertIn("If its sweep has not run yet", text)
        self.assertIn("real_cell.calibrate --rig overhead --mode eye_to_hand", text)
        self.assertIn("set artifact_path to the file it writes; until then, leave the block out", text)
        self.assertNotIn("writes it there", text, "the sweep writes its own artifact, not the one a block names")
        self.assertTrue(text.endswith("."), "a caller that adds a sentence of its own starts it after a full stop")

    def test_the_observation_stands_without_the_real_cells_remedy(self) -> None:
        """The wrist sweep is the remedy and the sim runner has no block to edit: each says the observation."""
        missing = str(self.root / "never_written.json")
        with self.assertRaises(RigArtifactMissing) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": missing}).extrinsics)
        self.assertIn("there is no file at that path", caught.exception.observation)
        self.assertNotIn("sweep", caught.exception.observation)
        self.assertEqual(str(caught.exception), f"{caught.exception.observation}. {caught.exception.remedy}")

    def test_a_relative_path_names_the_working_directory_once(self) -> None:
        with self.assertRaises(RigCalibrationError) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": "nowhere/eth.json"}).extrinsics)
        self.assertIn("names nowhere/eth.json (eye_to_hand)", str(caught.exception), "the path as written")
        self.assertEqual(str(caught.exception).count(str(Path.cwd())), 1)

    def test_an_absolute_path_names_no_working_directory(self) -> None:
        with self.assertRaises(RigCalibrationError) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": str(self.root / "x.json")}).extrinsics)
        self.assertNotIn("working directory", str(caught.exception))

    def test_a_working_directory_removed_under_the_program_is_still_the_calibrations_refusal(self) -> None:
        """⛔ Measured by review: on POSIX `Path.cwd()` raises once the directory is gone, which is exactly
        when a relative artifact goes missing, and the callers catch only `RigCalibrationError`."""
        with mock.patch.object(Path, "cwd", side_effect=FileNotFoundError("gone")), \
                self.assertRaises(RigArtifactMissing) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": "nowhere/eth.json"}).extrinsics)
        self.assertIn("a working directory that no longer exists", str(caught.exception))

    def test_a_wrist_block_keeps_its_measured_tolerances(self) -> None:
        """⛔ Measured by review: pasting the sweep's block over a wrist block, or deleting it, threw away
        tolerances measured on the cell, which the printed block carries only as comments."""
        wrist = {"mounting_mode": "eye_in_hand", "artifact_path": str(self.root / "never_written.json"),
                 "shutter_motion_tolerance_mm": 2.0, "shutter_motion_tolerance_deg": 1.0}
        with self.assertRaises(RigCalibrationError) as caught:
            RigCalibration.from_config("wrist", _rgbd(wrist).extrinsics)
        self.assertIn("--mode eye_in_hand, or examples/real_robot/09 or 10", str(caught.exception))
        self.assertIn("keeping the tolerances the block holds; until then, comment the block out", str(caught.exception))
        self.assertNotIn("leave the block out", str(caught.exception))
        self.assertNotIn("paste", str(caught.exception))

    def test_a_file_that_is_there_and_does_not_parse_keeps_its_own_sentence(self) -> None:
        """The control: the new sentence is for a file that is not there, not for every failed load."""
        broken = self.root / "broken.json"
        broken.write_text("{ not json", encoding="utf-8")
        with self.assertRaises(RigCalibrationError) as caught:
            RigCalibration.from_config("overhead", _rgbd({**_FIXED, "artifact_path": str(broken)}).extrinsics)
        self.assertIn("which does not load", str(caught.exception))
        self.assertNotIn("there is no file at", str(caught.exception))
        self.assertNotIsInstance(caught.exception, RigArtifactMissing)


class TheCellBuildSaysItInWholeSentencesTests(unittest.TestCase):
    """The pick cell's build embeds the refusal and adds a sentence of its own."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())

    def _refusal(self, path: str) -> str:
        from src.robot.execution.autonomous_grasp.builders import _resolver_for_rig

        with self.assertRaises(RuntimeError) as caught:
            _resolver_for_rig("overhead", _rgbd({**_FIXED, "artifact_path": path}).extrinsics)
        return str(caught.exception)

    def test_a_missing_artifact_is_one_sentence_after_another_and_its_fix_is_said_once(self) -> None:
        """⛔ Measured by review: "...the directory the program was started in The cell is refused ...",
        followed by a second copy of the leave-the-block-out advice."""
        text = self._refusal(str(self.root / "never_written.json"))
        self.assertIn("If it has run, compare this path with the one the sweep printed. The cell is refused", text)
        self.assertNotIn("fix the artifact", text)
        self.assertEqual(text.count("extrinsics block"), 0)
        self.assertEqual(text.count("leave the block out"), 1)

    def test_a_file_that_does_not_parse_keeps_the_builds_fix(self) -> None:
        broken = self.root / "broken.json"
        broken.write_text("{ not json", encoding="utf-8")
        text = self._refusal(str(broken))
        self.assertIn("does not load", text)
        self.assertIn(". The cell is refused at construction", text)
        self.assertTrue(text.endswith("remove the rig's extrinsics block until the camera is calibrated."))


class TheOwnerLoadsThroughTheLoaderTests(unittest.TestCase):

    def test_an_uncalibrated_owner_says_so_and_refuses_naming_the_key(self) -> None:
        from src.camera.orchestration.camera import Camera

        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=object())
        self.assertFalse(camera.calibrated)
        with self.assertRaises(RigNotCalibrated) as caught:
            camera.calibration()
        self.assertIn("camera.cameras.rigs['overhead'].extrinsics", str(caught.exception))

    def test_a_calibrated_owner_answers_what_the_loader_answers(self) -> None:
        from src.camera.orchestration.camera import Camera

        with tempfile.TemporaryDirectory() as folder:
            path = save_extrinsics(Path(folder) / "eth_overhead.json", Extrinsics(
                transform=_transform(Frame.BASE), rmse_mm=0.8, max_error_mm=1.5, num_samples=22,
                captured_at=dt.datetime(2026, 9, 15, tzinfo=dt.timezone.utc), rig_id="overhead"))
            rig = _rgbd({**_FIXED, "artifact_path": str(path)})
            camera = Camera.from_rig(rig, streamer=object())
            self.assertTrue(camera.calibrated)
            np.testing.assert_allclose(camera.calibration().camera_to_base().translation_mm,
                                       RigCalibration.from_config("overhead", rig.extrinsics).transform.translation_mm)


class TheLoaderPlacementTests(unittest.TestCase):

    def test_the_calibration_loader_loads_nothing_the_robot_may_not(self) -> None:
        from tests.test_robot import _ROBOT_MUST_NOT_LOAD
        from tests.test_robot_parts import _loaded_after

        self.assertEqual(_loaded_after("import src.calibration.rig_calibration", _ROBOT_MUST_NOT_LOAD), [])

    def test_the_probe_sees_the_camera_package_where_it_is_loaded(self) -> None:
        """The control: the same probe over the camera owner sees the camera package the robot may not load."""
        from tests.test_robot import _ROBOT_MUST_NOT_LOAD
        from tests.test_robot_parts import _loaded_after

        self.assertIn("src.camera",
                      _loaded_after("import src.camera.orchestration.camera", _ROBOT_MUST_NOT_LOAD))


if __name__ == "__main__":
    unittest.main()
