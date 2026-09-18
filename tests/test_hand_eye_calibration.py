"""HandEyeCalibration: the calibrate CLI's flow as a library noun, and the CLI as a caller of it.

Owner Q8 (2026-09-18): calibration works the way the examples do, as a noun in the calling convention. ``check()``
decides from the config alone and builds nothing; ``run(dry_run=True)`` builds the arm alone and one camera and moves
nothing; ``run()`` connects through ``Robot.connected()``, sweeps, and writes the artifact and the rig block. Every
refusal is an outcome on a frozen report, never an exception.

Why each class is red on the tree before the surface step:

* Every test here imports ``src.robot.execution.hand_eye``, which does not exist yet.
* ``TheCliIsACallerOfTheNounTests`` would stay red past the import too: the CLI builds ``CalibrationRoutine`` itself,
  prints ``--dict``'s literal ``DICT_5X5_100`` whatever the tree says, and reads ``camera.hand_eye.eye_to_hand`` for a
  wrist sweep.

The CLI's own transcripts are pinned in ``tests/test_calibrate_cli_transcript.py``, which is green before and after.
"""

from __future__ import annotations

import ast
import datetime as dt
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from src.config import ConfigError
from src.config.schema.camera import HandEyeConfig
from src.config.schema.camera.shared_schema import RigExtrinsicsConfig
from src.config.schema.robot import RobotConfig
from src.config.tree import ConfigTree, LoadedTree
from src.calibration import Extrinsics, MountingMode
from src.camera.orchestration.camera import Camera
from src.contracts import UNSET, Rendered, Structured
from src.geometry import Frame, Transform
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution import hand_eye
from src.robot.execution.calibration import CalibrationResult, CalibrationRoutine
from src.robot.execution.cell_lock import CellLock, lock_path_for, peek
from src.robot.execution.hand_eye import (
    CalibrationBuild,
    CalibrationCheck,
    CalibrationOutcome,
    CalibrationRunReport,
    HandEyeCalibration,
    SweepOptions,
)
from src.robot.execution.real_cell import calibrate
from src.robot.execution.robot import Robot
from src.robot.safety import SafetyAttestation
from tests._wrist_body import declared_frame, wrist_cell
from tests.test_camera_boundaries import _rgbd_rig

_IP = "10.253.253.31"
_KEY = f"ur@{_IP}"
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"
_CREATE_ARM = "src.robot.drivers.create_arm"
_CAMERA = "src.camera.orchestration.camera.Camera"


def _ur() -> RobotConfig:
    return RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": _IP}})


def _no_geometry() -> RobotConfig:
    """A UR cell that reads no wrist geometry (IK planner, capsule guard) and declares its tool frame as ``willy``."""
    return wrist_cell(planner="ik", backend="capsule")


def _app(*rigs: Any, hand_eye: Any = None, robot: Any = None) -> SimpleNamespace:
    """The two sections a sweep reads: a robot section and a camera section with its hand_eye block."""
    return SimpleNamespace(
        robot=robot if robot is not None else _ur(),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=list(rigs) or [_rgbd_rig("overhead")]),
                               hand_eye=hand_eye if hand_eye is not None else HandEyeConfig()),
    )


def _eth_result(*, carrier: bool = True) -> CalibrationResult:
    """An eye to hand solve, its extrinsics stamped to the routine's rig id rather than the camera's."""
    transform = Transform(translation_mm=np.array([400.0, -25.0, 812.5]), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
                          from_frame=Frame.CAMERA, to_frame=Frame.BASE)
    extrinsics = Extrinsics(transform=transform, rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                            captured_at=dt.datetime(2026, 9, 18, tzinfo=dt.timezone.utc), rig_id="rig_0")
    return CalibrationResult(T_cam_to_base=transform.to_matrix(), rmse_mm=0.8, max_error_mm=1.5, num_samples=20,
                             dataset_path="dataset.json", extrinsics=extrinsics if carrier else None,
                             transform=transform, mode=MountingMode.EYE_TO_HAND)


def _eih_result() -> CalibrationResult:
    transform = Transform(translation_mm=np.array([0.0, 60.0, 40.0]), quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
                          from_frame=Frame.CAMERA, to_frame=Frame.TOOL)
    return CalibrationResult(T_cam_to_base=None, T_cam_to_tool=transform.to_matrix(), rmse_mm=1.7, max_error_mm=3.0,
                             num_samples=18, dataset_path="dataset.json", transform=transform,
                             mode=MountingMode.EYE_IN_HAND)


class _Streamer:
    """The device behind a real ``Camera`` owner: counts its opens and releases and reports a camera matrix."""

    def __init__(self, *, refuse: bool = False) -> None:
        self.refuse = refuse
        self.opened = 0
        self.released = 0

    def open(self) -> None:
        if self.refuse:
            raise RuntimeError("no device on this port")
        self.opened += 1

    def release(self) -> None:
        self.released += 1

    def grab(self) -> Any:
        raise AssertionError("a frame was grabbed")

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def get_distortion(self) -> None:
        return None


class _Arm(DummyRobotArm):
    """A dummy arm that counts its connects and applies the declared tool frame, as a ``willy`` UR cell does."""

    def __init__(self) -> None:
        super().__init__()
        self.connects = 0
        self.active_tool_frame = declared_frame()

    def connect(self) -> None:
        self.connects += 1
        super().connect()


class _Doubles(unittest.TestCase):
    """A sweep over a dummy arm and a real ``Camera`` owner around a device double."""

    def setUp(self) -> None:
        self.addCleanup(lambda: lock_path_for(_KEY).unlink(missing_ok=True))

    def _noun(self, *, mode: str = "eye_to_hand", rig_id: str = "overhead", robot_config: Any = None,
              lock_key: "str | None" = None, options: Any = UNSET, refuse: bool = False,
              **hooks: Any) -> HandEyeCalibration:
        self.arm = _Arm()
        self.streamer = _Streamer(refuse=refuse)
        self.camera = Camera.from_rig(_rgbd_rig(rig_id), streamer=self.streamer)
        self.addCleanup(self.camera.release)
        robot = Robot.from_parts(arm=self.arm, gripper=None, lock_key=lock_key)
        return HandEyeCalibration.from_parts(robot=robot, camera=self.camera,
                                             robot_config=robot_config if robot_config is not None else _ur(),
                                             mode=mode, options=options, **hooks)

    def _out(self) -> str:
        return str(self.enterContext(tempfile.TemporaryDirectory()))


# ---------------------------------------------------------------------------------------------------
# check(): the config alone
# ---------------------------------------------------------------------------------------------------


class CheckTouchesNothingTests(unittest.TestCase):

    def _shipped(self, rig_id: str) -> CalibrationCheck:
        """A check on the repository's base tree, with every builder rigged to fail loudly if it is reached."""
        loaded = ConfigTree.from_directory(profile=None).load()
        self.assertTrue(loaded.ok, loaded.error)
        explode = AssertionError("check() reached a builder")
        with patch(_READY, side_effect=explode), patch(_CREATE_ARM, side_effect=explode), \
                patch(_CAMERA) as camera_cls:
            check = HandEyeCalibration.from_tree(loaded, rig_id=rig_id, mode="eye_to_hand").check()
        camera_cls.from_config.assert_not_called()
        camera_cls.from_rig.assert_not_called()
        return check

    def test_the_shipped_tree_refuses_its_rgbd_rig_by_name_and_builds_nothing(self) -> None:
        """The base tree ships realsense_d435 switched off: the check says so, as a report, before anything is built."""
        check = self._shipped("realsense_d435")
        self.assertFalse(check.ok)
        self.assertIn("`enabled: false`", check.refusal)
        self.assertIn("produces an artifact nothing consumes", check.refusal)
        self.assertTrue(check.render().startswith("[config] REFUSED: rig 'realsense_d435' has"), check.render())

    def test_a_webcam_rig_is_refused_before_it_can_fail_about_aruco(self) -> None:
        self.assertIn("not an RGB-D device", self._shipped("webcam_main").refusal)

    def test_an_unknown_rig_lists_the_configured_ones(self) -> None:
        refusal = self._shipped("front").refusal
        self.assertIn("'front'", refusal)
        self.assertIn("realsense_d435", refusal)

    def test_a_usable_rig_reports_what_the_sweep_will_use(self) -> None:
        check = HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand").check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.render(), "\n".join((
            "  rig        'overhead' (rgbd)",
            "  mode       eye_to_hand",
            "  arm        ur",
            "  marker     50.0 mm, id 0, DICT_5X5_100",
            "  poses      22",
            "  artifact   calibration/real/eth_overhead.json",
        )))

    def test_a_chosen_option_wins_over_the_tree_and_the_tree_over_the_code(self) -> None:
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"marker_length_mm": 48.0, "aruco_dict_name": "DICT_4X4_50"}})
        from_tree = HandEyeCalibration.from_config(_app(hand_eye=tree), rig_id="overhead", mode="eye_to_hand")
        self.assertEqual((from_tree.marker_length_mm, from_tree.dict_name, from_tree.poses, from_tree.out_dir),
                         (48.0, "DICT_4X4_50", 22, "calibration/real"))
        chosen = HandEyeCalibration.from_config(
            _app(hand_eye=tree), rig_id="overhead", mode="eye_to_hand",
            options=SweepOptions(poses=10, marker_length_mm=39.7, marker_id=3, dict_name="DICT_5X5_100", out_dir="bench"))
        self.assertEqual(chosen.check().render().splitlines()[3:], [
            "  marker     39.7 mm, id 3, DICT_5X5_100",
            "  poses      10",
            "  artifact   bench/eth_overhead.json",
        ])

    def test_an_eye_in_hand_sweep_reads_the_eye_in_hand_block(self) -> None:
        """The CLI read ``camera.hand_eye.eye_to_hand`` for both mountings; the noun reads the block of its mode."""
        tree = HandEyeConfig.model_validate({"eye_in_hand": {"marker_length_mm": 30.0, "aruco_dict_name": "DICT_4X4_50"}})
        calibration = HandEyeCalibration.from_config(_app(_rgbd_rig("wrist"), hand_eye=tree, robot=_no_geometry()),
                                                     rig_id="wrist", mode="eye_in_hand")
        self.assertIs(calibration.settings, tree.eye_in_hand)
        self.assertEqual((calibration.marker_length_mm, calibration.dict_name), (30.0, "DICT_4X4_50"))
        check = calibration.check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.artifact_path, "calibration/real/eih_wrist.json")

    def test_a_robot_that_holds_a_live_camera_world_is_refused(self) -> None:
        """A sweep declines the world for every move, and an arm whose world is wired refuses a declined move."""
        streamer = _Streamer()
        robot = Robot(arm=DummyRobotArm(), gripper=None, lock_key=None,
                      camera_world=SimpleNamespace(world=object(), reason="", cameras=("overhead",)))  # type: ignore[arg-type]
        calibration = HandEyeCalibration.from_parts(
            robot=robot, camera=Camera.from_rig(_rgbd_rig("overhead"), streamer=streamer), robot_config=_ur(),
            mode="eye_to_hand")
        self.assertIn("holds a live camera world", calibration.check().refusal)
        report = calibration.run()
        self.assertIs(report.outcome, CalibrationOutcome.REFUSED)
        self.assertEqual(streamer.opened, 0)

    def test_a_tree_that_did_not_load_is_refused_with_its_own_refusal(self) -> None:
        loaded = LoadedTree(tree=ConfigTree(root=Path("nowhere"), profile=None),
                            error="robot/robot.yaml: this tree does not parse")
        with self.assertRaises(ConfigError) as caught:
            HandEyeCalibration.from_tree(loaded, rig_id="overhead", mode="eye_to_hand")
        self.assertIn("this tree does not parse", str(caught.exception))

    def test_a_tree_that_loaded_without_a_robot_block_is_refused_with_the_loaders_sentence(self) -> None:
        loaded = LoadedTree(tree=ConfigTree(root=Path("nowhere"), profile=None),
                            config=SimpleNamespace(robot=None, camera=None))
        with self.assertRaises(ConfigError) as caught:
            HandEyeCalibration.from_tree(loaded, rig_id="overhead", mode="eye_to_hand")
        self.assertIn("no `robot` block", str(caught.exception))

    def test_the_mounting_has_no_default_and_a_wrong_one_is_refused(self) -> None:
        """A wrist camera swept eye to hand writes a CAMERA to BASE file for a camera that moves."""
        with self.assertRaises(TypeError):
            HandEyeCalibration.from_config(_app(), rig_id="overhead")  # type: ignore[call-arg]
        with self.assertRaises(ValueError):
            HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_on_hand")

    def test_a_call_without_a_robot_section_or_a_camera_raises(self) -> None:
        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=_Streamer())
        with self.assertRaisesRegex(ValueError, "robot_config"):
            HandEyeCalibration.from_parts(camera=camera, mode="eye_to_hand")
        with self.assertRaisesRegex(ValueError, "camera"):
            HandEyeCalibration.from_parts(robot_config=_ur(), mode="eye_to_hand")


# ---------------------------------------------------------------------------------------------------
# run(dry_run=True): built, attested, given back
# ---------------------------------------------------------------------------------------------------


class DryRunTests(_Doubles):

    def test_a_dry_run_against_doubles_returns_a_report_and_moves_nothing(self) -> None:
        built: list[CalibrationBuild] = []
        report = self._noun(on_built=built.append).run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN)
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(built, [report.build], "on_built did not receive the build before the run returned")
        build = report.build
        assert build is not None
        self.assertEqual((build.arm, build.lock_key, build.intrinsics), ("_Arm", None, True))
        self.assertTrue(build.gripper.startswith("none"))
        self.assertEqual(build.safety, SafetyAttestation.of(self.arm))
        self.assertEqual(self.arm.connects, 0)
        self.assertEqual((self.streamer.opened, self.streamer.released), (1, 1))
        self.assertFalse(self.camera.is_open)
        self.assertIn("--dry-run: built cleanly and the camera answered", report.render())

    def test_a_camera_that_does_not_open_is_a_build_refusal_and_nothing_stays_claimed(self) -> None:
        report = self._noun(refuse=True).run()
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED)
        self.assertEqual(report.exit_code, 1)
        assert report.build is not None
        self.assertEqual(report.build.render(), "[build] REFUSED: RuntimeError: no device on this port")
        self.assertEqual(report.summary(), "")
        self.assertEqual(self.arm.connects, 0)
        self.assertFalse(self.camera.is_open)

    def test_an_arm_that_cannot_be_built_is_refused_before_the_camera_is_built(self) -> None:
        with patch(_READY), patch(_CREATE_ARM, side_effect=RuntimeError("no controller here")), \
                patch(_CAMERA) as camera_cls:
            report = HandEyeCalibration.from_config(_app(), rig_id="overhead", mode="eye_to_hand").run()
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED)
        assert report.build is not None
        self.assertEqual(report.build.refusal, "RuntimeError: no controller here")
        camera_cls.from_config.assert_not_called()


# ---------------------------------------------------------------------------------------------------
# run(): the sweep, with the routine's run_auto standing in for the motion
# ---------------------------------------------------------------------------------------------------


class SweepTests(_Doubles):

    def test_an_eye_to_hand_sweep_writes_the_artifact_keyed_by_the_rig(self) -> None:
        out = self._out()
        run_auto = MagicMock(return_value=_eth_result())
        with patch.object(CalibrationRoutine, "run_auto", run_auto):
            report = self._noun(options=SweepOptions(out_dir=out)).run()
        self.assertIs(report.outcome, CalibrationOutcome.WRITTEN)
        self.assertEqual(report.exit_code, 0)
        written = Path(out) / "eth_overhead.json"
        self.assertEqual(Path(report.artifact_path), written)
        self.assertEqual(json.loads(written.read_text(encoding="utf-8"))["rig_id"], "overhead")
        self.assertEqual((report.accepted_samples, report.quality), (20, "excellent"))
        assert report.transform_mm is not None
        self.assertEqual(report.transform_mm[0][3], 400.0)
        block = yaml.safe_load(report.rig_block)["camera"]["cameras"]["rigs"][0]
        self.assertEqual(block["rig_id"], "overhead")
        self.assertEqual(RigExtrinsicsConfig(**block["extrinsics"]).mounting_mode, "eye_to_hand")
        self.assertEqual(run_auto.call_args.args, (22,))
        keywords = run_auto.call_args.kwargs
        self.assertEqual((keywords["orientation_spread_deg"], keywords["seed"]), (30.0, 0))
        self.assertTrue(keywords["dataset_save_path"].endswith("eye_to_hand_overhead_dataset.json"))
        self.assertEqual(self.arm.connects, 1)
        assert report.teardown is not None
        self.assertTrue(report.teardown.clean)
        self.assertEqual(self.streamer.released, 1)
        self.assertIsNone(report.flange_to_tcp, "an eye to hand artifact records no flange")

    def test_an_eye_in_hand_sweep_records_the_flange_to_tcp(self) -> None:
        """The wrist record (owner E13): read from the connected arm, written as /2, named on the report."""
        out = self._out()
        with patch.object(CalibrationRoutine, "run_auto", MagicMock(return_value=_eih_result())):
            report = self._noun(mode="eye_in_hand", rig_id="wrist", robot_config=_no_geometry(),
                                options=SweepOptions(out_dir=out)).run()
        self.assertIs(report.outcome, CalibrationOutcome.WRITTEN, report.render())
        assert report.flange_to_tcp is not None
        self.assertEqual(report.flange_to_tcp.source, "willy")
        body = json.loads(Path(report.artifact_path).read_text(encoding="utf-8"))
        self.assertEqual(body["schema"], "willy.calibration.cam_to_tool/2")
        self.assertIn("  flange to TCP     recorded on willy", report.render())
        self.assertIn("# shutter_motion_tolerance_mm:", report.rig_block)

    def test_a_solve_with_no_carrier_writes_nothing_and_exits_2(self) -> None:
        out = self._out()
        with patch.object(CalibrationRoutine, "run_auto", MagicMock(return_value=_eth_result(carrier=False))):
            report = self._noun(options=SweepOptions(out_dir=out)).run()
        self.assertIs(report.outcome, CalibrationOutcome.NO_ARTIFACT)
        self.assertEqual(report.exit_code, 2)
        self.assertEqual(list(Path(out).iterdir()), [])
        self.assertEqual((report.artifact_path, report.rig_block), ("", ""))
        self.assertIn("[result] no Extrinsics carrier; nothing was written.", report.summary())

    def test_a_sweep_that_raises_is_reported_once_the_arm_is_down(self) -> None:
        with patch.object(CalibrationRoutine, "run_auto", MagicMock(side_effect=RuntimeError("the marker left the frame"))):
            report = self._noun(options=SweepOptions(out_dir=self._out())).run()
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertEqual(report.exit_code, 3)
        self.assertEqual(report.failure, "RuntimeError: the marker left the frame")
        assert report.teardown is not None
        summary = report.summary()
        self.assertLess(summary.index(report.teardown.render()), summary.index("[sweep] FAILED"))
        self.assertFalse(self.camera.is_open)

    def test_a_held_cell_is_an_outcome_not_an_exception(self) -> None:
        calibration = self._noun(lock_key=_KEY, options=SweepOptions(out_dir=self._out()))
        with CellLock(_KEY, owner="operator console"), \
                patch.object(CalibrationRoutine, "run_auto", MagicMock(side_effect=AssertionError("the sweep ran"))):
            report = calibration.run()
        self.assertIs(report.outcome, CalibrationOutcome.CELL_BUSY)
        self.assertEqual(report.exit_code, 1)
        self.assertIn("operator console", report.failure)
        self.assertTrue(report.summary().startswith("[connect] REFUSED: "))
        self.assertEqual(self.arm.connects, 0)
        self.assertEqual(self.streamer.released, 1)
        self.assertIsNone(peek(_KEY))


# ---------------------------------------------------------------------------------------------------
# The report halves
# ---------------------------------------------------------------------------------------------------


class ReportContractTests(_Doubles):

    def test_every_report_renders_ascii_without_a_trailing_newline_and_its_dict_survives_json(self) -> None:
        with patch.object(CalibrationRoutine, "run_auto", MagicMock(return_value=_eih_result())):
            report = self._noun(mode="eye_in_hand", rig_id="wrist", robot_config=_no_geometry(),
                                options=SweepOptions(out_dir=self._out())).run()
        assert report.build is not None
        for part in (report, report.check, report.build):
            with self.subTest(type(part).__name__):
                self.assertIsInstance(part, Rendered)
                self.assertIsInstance(part, Structured)
                text = part.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                payload = part.to_dict()
                self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertTrue(report.rig_block.endswith("\n"), "the rig block is YAML text, printed as it is pasted")

    def test_the_exit_codes_are_the_clis(self) -> None:
        check = CalibrationCheck(rig_id="overhead", mode=MountingMode.EYE_TO_HAND)
        expected = {
            CalibrationOutcome.WRITTEN: calibrate._EXIT_OK,
            CalibrationOutcome.DRY_RUN: calibrate._EXIT_OK,
            CalibrationOutcome.REFUSED: calibrate._EXIT_CONFIG,
            CalibrationOutcome.BUILD_REFUSED: calibrate._EXIT_CONFIG,
            CalibrationOutcome.CELL_BUSY: calibrate._EXIT_CONFIG,
            CalibrationOutcome.CONNECT_FAILED: calibrate._EXIT_CONFIG,
            CalibrationOutcome.NO_ARTIFACT: calibrate._EXIT_NO_ARTIFACT,
            CalibrationOutcome.SWEEP_FAILED: calibrate._EXIT_ERROR,
        }
        self.assertEqual(set(expected), set(CalibrationOutcome))
        for outcome, code in expected.items():
            with self.subTest(outcome.value):
                self.assertEqual(CalibrationRunReport(check=check, outcome=outcome).exit_code, code)


# ---------------------------------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------------------------------


def _calls(source: str) -> set[str]:
    """``Name(...)`` and ``Owner.attr(...)`` calls in ``source``, as ``Name`` and ``Owner.attr``."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            found.add(func.id)
        elif isinstance(func, ast.Attribute):
            owner = func.value.id if isinstance(func.value, ast.Name) else ""
            found.add(f"{owner}.{func.attr}" if owner else func.attr)
    return found


class TheCliIsACallerOfTheNounTests(unittest.TestCase):

    def _check(self, app: Any, *argv: str) -> tuple[int, str]:
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=app), redirect_stdout(printed):
            code = calibrate.main([*argv, "--check"])
        return code, printed.getvalue()

    def test_the_cli_builds_no_part_of_the_sweep_itself(self) -> None:
        """Red before the step: the CLI built the routine, the marker source and the camera, and wrote the file."""
        cli = _calls(Path(calibrate.__file__).read_text(encoding="utf-8"))
        noun = _calls(Path(hand_eye.__file__).read_text(encoding="utf-8"))
        for name in ("CalibrationRoutine", "RGBDArucoMarkerSource", "Camera.from_config", "Robot.from_config",
                     "save_extrinsics", "save_cam_to_tool"):
            with self.subTest(name):
                self.assertIn(name, noun, "the scan cannot see the call it forbids")
                self.assertNotIn(name, cli)
        self.assertIn("HandEyeCalibration.from_config", cli)

    def test_the_cli_reads_the_dictionary_from_the_tree(self) -> None:
        """Red before the step: `--dict` defaulted to the literal DICT_5X5_100 whatever the tree said."""
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"aruco_dict_name": "DICT_4X4_50"}})
        code, printed = self._check(_app(hand_eye=tree), "--rig", "overhead")
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertIn("  marker     50.0 mm, id 0, DICT_4X4_50\n", printed)

    def test_a_typed_dictionary_still_wins(self) -> None:
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"aruco_dict_name": "DICT_4X4_50"}})
        _, printed = self._check(_app(hand_eye=tree), "--rig", "overhead", "--dict", "DICT_5X5_100")
        self.assertIn("  marker     50.0 mm, id 0, DICT_5X5_100\n", printed)

    def test_a_wrist_sweep_on_the_cli_reads_the_eye_in_hand_block(self) -> None:
        """Red before the step: the CLI read camera.hand_eye.eye_to_hand for both mountings."""
        tree = HandEyeConfig.model_validate({"eye_in_hand": {"marker_length_mm": 30.0}})
        code, printed = self._check(_app(_rgbd_rig("wrist"), hand_eye=tree, robot=_no_geometry()),
                                    "--rig", "wrist", "--mode", "eye_in_hand")
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertIn("  marker     30.0 mm, id 0, DICT_5X5_100\n", printed)
        self.assertIn("  artifact   calibration/real/eih_wrist.json\n", printed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
