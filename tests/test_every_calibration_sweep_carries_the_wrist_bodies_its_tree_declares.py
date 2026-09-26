"""Every calibration sweep carries the wrist camera bodies its tree declares, whichever camera it calibrates.

The gap of 2026-09-25: an eye to hand sweep built its arm with ``Robot.from_config(robot_config, gripper=None)`` and
handed it no wrist body, because only an eye in hand sweep read one, and only its own camera's. On the owner's cell (a
UR10 CB3 with a D415 on a bracket on the wrist, declared as an eye_in_hand rig with a body) a sweep of a fixed camera
over fixed stations, where the arm drives itself, planned without the D415 and its bracket. ``Robot.from_tree``
carries every body the tree declares (tests/test_a_robot_carries_every_wrist_body_its_tree_declares.py); the sweep
now resolves them the same way (``WristBodies.from_config``), with the same rules:

* a placeable body is carried, and the check and the build name it;
* a declared body nothing can place yet (the wrist camera is not calibrated) refuses the sweep at the check, unless
  ``--unmodelled-wrist-body "<reason>"`` says why the arm may move without it, and the sweep then says so; the
  refusal gives this sweep's flag first, labelled, before the separate command that calibrates the wrist camera;
* a tree that declares no wrist camera sweeps exactly as before: no body, no refusal, no new line;
* an eye in hand sweep keeps its handling of the camera it calibrates.

The owner's decision of 2026-09-25, for calibration only: a wrist rig (``eye_in_hand`` extrinsics) the tree declares
without a body, switched on or off, refuses no sweep, eye to hand or eye in hand of another wrist camera, over fixed
stations or guided by hand. Nothing is carried for it, and the check and the build print one warning line naming it,
which is logged. ``Robot.from_tree``, and so picking, keeps refusing an enabled one.

A robot handed in through ``from_parts`` that already carries the bodies the sweep would hand over is not handed them
again: an arm whose exact mesh guard is built refuses any hand-over, the same bodies included. A body it does not hold
is still handed over, and refused where the guard is built.

Everything below the arm is real: the UR driver object (never connected) holds the bodies its guard and its planner
read. The camera is a double, and nothing moves: a sweep is run with ``dry_run=True`` or refused before it builds.
"""

from __future__ import annotations

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

from src.calibration.serialization import FlangeToTcp, save_cam_to_tool
from src.config import ConfigTree
from src.config.schema.camera import HandEyeConfig
from src.config.tree import LoadedTree
from src.contracts import UNSET
from src.geometry import Pose
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.hand_eye import CalibrationOutcome, CalibrationRunReport, HandEyeCalibration, SweepOptions
from src.robot.execution.real_cell import calibrate
from src.robot.execution.robot import Robot
from src.robot.execution.wrist_bodies import WristBodies, WristBodyRequired
from tests._wrist_body import camera_section, camera_to_tool, declared_frame, wrist_cell, wrist_rig
from tests.test_camera_boundaries import _rgbd_rig

_CAMERA = "src.camera.orchestration.camera.Camera"
_STATIONS = [Pose.tool_down(400.0, 0.0, 350.0, label="down_0"), Pose.tool_down(420.0, 60.0, 380.0, label="down_1")]
_REASON = "the fixed camera first, the D415 on the wrist next"
#: The owner's wording for a wrist rig declared without a body, the one line the check and the build print.
_NO_BODY = ("wrist camera 'wrist' is declared on the arm without a body: the planner and the guard do not know it is "
            "there during this sweep")
_GUARD_BUILT = "the exact mesh guard was already built, so a wrist camera handed in now would not be in it"


class _Handle:
    """The camera handle a build prints and the marker source reads."""

    def __repr__(self) -> str:
        return "RigHandle('overhead')"

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def release(self) -> None:
        pass


class _Sweeps(unittest.TestCase):
    """A wrist cell (cuRobo, a declared ``willy`` tool frame) and its camera section, swept on a real UR arm object."""

    def setUp(self) -> None:
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)

    def _wrist(self, rig_id: str = "wrist", *, calibrated: bool = True, artifact: bool = True,
               **changes: Any) -> SimpleNamespace:
        """A wrist rig; calibrated, its artifact is written with the declared frame, unless ``artifact`` is False."""
        rig = wrist_rig(rig_id, calibrated=calibrated, **changes)
        if rig.extrinsics is not None:
            path = self.root / f"eih_{rig_id}.json"
            if artifact:
                save_cam_to_tool(path, camera_to_tool(), rig_id=rig_id,
                                 flange_to_tcp=FlangeToTcp.from_matrix("willy", declared_frame()))
            rig.extrinsics.artifact_path = str(path)
        return rig

    def _app(self, *wrist: SimpleNamespace, robot: Any = None) -> SimpleNamespace:
        """The two sections a sweep reads: the fixed camera ``overhead`` and the wrist rigs handed in."""
        return SimpleNamespace(
            robot=robot if robot is not None else wrist_cell(),
            camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead"), *wrist],
                                                           primary_rig_id="overhead"),
                                   hand_eye=HandEyeConfig()),
        )

    def _sweep(self, app: SimpleNamespace, *, rig_id: str = "overhead", mode: str = "eye_to_hand",
               reason: Any = UNSET, freedrive: bool = False) -> HandEyeCalibration:
        """Over the two fixed stations, or guided by hand throughout with ``freedrive``."""
        options = (SweepOptions(freedrive=True, unmodelled_wrist_body=reason) if freedrive else
                   SweepOptions(fixed_poses=_STATIONS, unmodelled_wrist_body=reason))
        return HandEyeCalibration.from_config(app, rig_id=rig_id, mode=mode, options=options)

    def _dry_run(self, calibration: HandEyeCalibration) -> "tuple[CalibrationRunReport, Robot]":
        """Built and given back, nothing moved; the robot the sweep built, whose arm holds what its guard reads."""
        built: list[Robot] = []
        real = Robot.from_config

        def spy(*args: Any, **kwargs: Any) -> Robot:
            robot = real(*args, **kwargs)
            built.append(robot)
            return robot

        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value = _Handle()
        with patch.object(Robot, "from_config", side_effect=spy), patch(_CAMERA, camera_cls):
            report = calibration.run(dry_run=True)
        self.assertEqual(len(built), 1, report.render())
        return report, built[0]

    @staticmethod
    def _held(robot: Robot) -> list[str]:
        """The wrist bodies the arm's guard and planner read."""
        assert isinstance(robot.arm, URRobotArm)
        return [body.link_name for body in robot.arm._preflight.wrist_bodies(robot.arm)]  # type: ignore[attr-defined]  # noqa: SLF001


class AFixedCameraSweepCarriesTheWristCameraTests(_Sweeps):

    def test_a_placeable_body_is_carried_by_the_arm_the_sweep_drives(self) -> None:
        """⛔ The gap: before, an eye to hand sweep's arm held no body, and nothing on the report said so."""
        report, robot = self._dry_run(self._sweep(self._app(self._wrist())))
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist cameras  wrist camera wrist_camera_wrist: realsense_d435", report.build.wrist)
        self.assertIn("\n  wrist cameras  wrist camera wrist_camera_wrist: realsense_d435", report.build.render())
        self.assertEqual(report.build.unmodelled, "")

    def test_the_check_names_the_body_before_anything_is_built(self) -> None:
        with patch.object(Robot, "from_config", side_effect=AssertionError("check() built the arm")):
            check = self._sweep(self._app(self._wrist())).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertIn("wrist cameras  wrist camera wrist_camera_wrist: realsense_d435", check.wrist)
        self.assertIn("\n  wrist cameras  wrist camera wrist_camera_wrist", check.render())
        self.assertEqual(check.to_dict()["wrist"], check.wrist)

    def test_the_body_is_the_one_robot_from_tree_resolves(self) -> None:
        rig = self._wrist()
        expected = WristBodies.from_config(wrist_cell(), camera_section(rig))
        _, robot = self._dry_run(self._sweep(self._app(rig)))
        assert isinstance(robot.arm, URRobotArm)
        self.assertEqual([body.to_dict() for body in robot.arm._preflight.wrist_bodies(robot.arm)],  # type: ignore[attr-defined]  # noqa: SLF001
                         [body.to_dict() for body in expected.bodies])

    def test_a_switched_off_wrist_camera_is_still_carried(self) -> None:
        _, robot = self._dry_run(self._sweep(self._app(self._wrist(enabled=False))))
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])


class ABodyNothingPlacesTests(_Sweeps):

    def test_an_uncalibrated_wrist_camera_refuses_the_check_and_names_the_reason_to_give(self) -> None:
        with patch.object(Robot, "from_config", side_effect=AssertionError("a refused sweep built the arm")):
            report = self._sweep(self._app(self._wrist(calibrated=False))).run()
        self.assertIs(report.outcome, CalibrationOutcome.REFUSED)
        said = report.check.refusal
        self.assertIn("camera.cameras.rigs['wrist'].body", said)
        self.assertIn("nothing places the body on the flange", said)
        self.assertTrue(said.startswith("this eye_to_hand sweep of 'overhead' moves the arm that carries wrist camera "
                                        "'wrist', whose declared body cannot be placed yet."), said)
        self.assertIn('For THIS sweep (--rig overhead), say why the arm may move without that body: '
                      '--unmodelled-wrist-body "<reason>".', said)

    def test_this_sweeps_own_hint_comes_first_and_is_labelled(self) -> None:
        """⛔ Review: the refusal carried two ``--unmodelled-wrist-body`` hints, the one inside the suggested eye in
        hand command first, and nothing said which of the two excuses the sweep being run."""
        said = self._sweep(self._app(self._wrist(calibrated=False))).check().refusal
        self.assertEqual(said.count("--unmodelled-wrist-body"), 2, said)
        this_sweep = said.index("For THIS sweep (--rig overhead)")
        first_flag = said.index("--unmodelled-wrist-body")
        other_command = said.index("python -m src.robot.execution.real_cell.calibrate --rig wrist --mode eye_in_hand")
        self.assertLess(this_sweep, first_flag, said)
        self.assertLess(first_flag, other_command, said)
        self.assertGreater(said.rindex("--unmodelled-wrist-body"), other_command, said)

    def test_an_eye_in_hand_sweep_names_the_other_wrist_camera_it_cannot_place(self) -> None:
        app = self._app(self._wrist(), self._wrist("side", calibrated=False))
        said = self._sweep(app, rig_id="wrist", mode="eye_in_hand").check().refusal
        self.assertTrue(said.startswith("this eye_in_hand sweep of 'wrist' moves the arm that carries wrist camera "
                                        "'side', whose declared body cannot be placed yet."), said)
        self.assertIn("For THIS sweep (--rig wrist)", said)

    def test_a_missing_artifact_refuses_the_check_too(self) -> None:
        check = self._sweep(self._app(self._wrist(artifact=False))).check()
        self.assertFalse(check.ok)
        self.assertIn("which does not load: there is no file at that path", check.refusal)
        self.assertIn('For THIS sweep (--rig overhead), say why the arm may move without that body: '
                      '--unmodelled-wrist-body "<reason>".', check.refusal)
        self.assertEqual(check.refusal.count("--unmodelled-wrist-body"), 1, check.refusal)

    def test_a_stated_reason_sweeps_without_the_body_and_says_so(self) -> None:
        calibration = self._sweep(self._app(self._wrist(calibrated=False)), reason=_REASON)
        check = calibration.check()
        self.assertTrue(check.ok, check.refusal)
        self.assertIn("NOT carried, its body cannot be placed yet: 'wrist'", check.wrist)
        self.assertIn(_REASON, check.wrist)
        with self.assertLogs("src.robot.execution.hand_eye", level="WARNING") as logged:
            report, robot = self._dry_run(calibration)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), [])
        assert report.build is not None
        self.assertIn("NOT carried, its body cannot be placed yet: 'wrist'", report.build.wrist)
        self.assertEqual(report.build.unmodelled,
                         f"wrist camera 'wrist' rides on the arm without its body in the planner and the guard: {_REASON}")
        self.assertIn(f"\n  !! {report.build.unmodelled}", report.build.render())
        self.assertIn(report.build.unmodelled, "\n".join(logged.output))

    def test_a_blank_reason_is_no_reason(self) -> None:
        self.assertFalse(self._sweep(self._app(self._wrist(calibrated=False)), reason="   ").check().ok)

    def test_a_reason_does_not_excuse_an_unknown_camera(self) -> None:
        """The rule ``Robot.from_tree`` applies: a reason stands in only for a calibration."""
        unknown = self._sweep(self._app(self._wrist(calibrated=False, model="acme_cam")), reason=_REASON).check()
        self.assertFalse(unknown.ok)
        self.assertIn("acme_cam", unknown.refusal)

    def test_a_reason_changes_nothing_for_a_wrist_rig_without_a_body(self) -> None:
        """A rig without a body needs no reason and takes none: the warning line is the same either way."""
        plain = self._sweep(self._app(self._wrist(body=False))).check()
        reasoned = self._sweep(self._app(self._wrist(body=False)), reason=_REASON).check()
        self.assertTrue(plain.ok, plain.refusal)
        self.assertEqual(reasoned.render(), plain.render())
        self.assertNotIn("--unmodelled-wrist-body", plain.render())

    def test_one_placed_body_is_carried_beside_one_that_is_not(self) -> None:
        app = self._app(self._wrist(), self._wrist("second", calibrated=False))
        report, robot = self._dry_run(self._sweep(app, reason=_REASON))
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("NOT carried, its body cannot be placed yet: 'second'", report.build.wrist)
        self.assertIn("'second'", report.build.unmodelled)

    def test_the_cli_refuses_without_the_reason_and_sweeps_with_it(self) -> None:
        stations = self.root / "stations.json"
        stations.write_text(json.dumps([{"x": 400.0, "y": 0.0, "z": 350.0, "rx": 0.0, "ry": 3.14159265, "rz": 0.0}]),
                            encoding="utf-8")
        app = self._app(self._wrist(calibrated=False))

        def main(*argv: str) -> "tuple[int, str]":
            printed = io.StringIO()
            with patch.object(calibrate, "_load", return_value=app), redirect_stdout(printed):
                code = calibrate.main(["--rig", "overhead", "--fixed-poses", str(stations), "--check", *argv])
            return code, printed.getvalue()

        code, printed = main()
        self.assertEqual(code, calibrate._EXIT_CONFIG, printed)
        self.assertIn("[config] REFUSED: this eye_to_hand sweep of 'overhead' moves the arm that carries wrist camera "
                      "'wrist'", printed)
        code, printed = main("--unmodelled-wrist-body", _REASON)
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertIn("  wrist cameras  NOT carried, its body cannot be placed yet: 'wrist'", printed)


class ATreeWithNoWristCameraSweepsAsBeforeTests(_Sweeps):
    """The control: nothing is read, nothing is handed over, and the reports hold no new line."""

    def test_no_body_no_refusal_no_line(self) -> None:
        with patch.object(URRobotArm, "set_wrist_bodies", autospec=True) as handed:
            report, robot = self._dry_run(self._sweep(self._app()))
        handed.assert_not_called()
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), [])
        assert report.build is not None
        self.assertEqual((report.build.wrist, report.build.unmodelled, report.check.wrist), ("", "", ""))
        self.assertNotIn("wrist", report.build.render())
        self.assertNotIn("wrist", report.check.render())

    def test_a_reason_given_to_such_a_tree_changes_nothing(self) -> None:
        plain = self._sweep(self._app()).check()
        reasoned = self._sweep(self._app(), reason=_REASON).check()
        self.assertEqual(reasoned.render(), plain.render())
        self.assertEqual(reasoned.to_dict(), plain.to_dict())

    def test_a_cell_that_reads_no_geometry_reads_no_wrist_body(self) -> None:
        """An uncalibrated wrist camera on an IK cell with the capsule guard: swept as before, with no refusal."""
        app = self._app(self._wrist(calibrated=False), robot=wrist_cell(planner="ik", backend="capsule"))
        check = self._sweep(app).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.wrist, "")


class AReasonExcusesOnlyABodyNothingCanPlaceTests(_Sweeps):
    """A reason stands in for a calibration, never for a body that is placed and does not cover its camera (E14).

    ⛔ Review of 2026-09-25: the eye in hand sweep's own body caught every ``WristBodyRequired`` behind its reason, so
    a camera whose sphere fill leaves a hole was swept with no body at all once a reason was given (examples 09 and
    10 always give one), and without a reason its refusal pointed at ``--unmodelled-wrist-body``, which cannot help.
    The same body on any other rig was refused either way.
    """

    _HOLE = "the sphere fill leaves the housing's corner 4.0 mm uncovered"

    def _checked(self, *, rig_id: str, mode: str, reason: Any) -> Any:
        from unittest.mock import patch

        from src.robot.safety.planning.body_link import WristBody

        with patch.object(WristBody, "cover_refusal", return_value=self._HOLE):
            return self._sweep(self._app(self._wrist()), rig_id=rig_id, mode=mode, reason=reason).check()

    def test_the_swept_cameras_body_with_a_hole_is_refused_whatever_the_reason(self) -> None:
        for reason in (None, _REASON):
            with self.subTest(reason=reason):
                check = self._checked(rig_id="wrist", mode="eye_in_hand", reason=reason)
                self.assertFalse(check.ok, check.render())
                self.assertIn(self._HOLE, check.refusal)
                self.assertNotIn("--unmodelled-wrist-body", check.refusal)

    def test_another_rigs_body_with_a_hole_is_refused_whatever_the_reason_as_before(self) -> None:
        for reason in (None, _REASON):
            with self.subTest(reason=reason):
                check = self._checked(rig_id="overhead", mode="eye_to_hand", reason=reason)
                self.assertFalse(check.ok, check.render())
                self.assertIn(self._HOLE, check.refusal)

    def test_a_body_nothing_places_yet_is_still_excused_by_a_reason(self) -> None:
        excused = self._sweep(self._app(self._wrist(calibrated=False)), rig_id="wrist", mode="eye_in_hand",
                              reason=_REASON).check()
        self.assertTrue(excused.ok, excused.refusal)
        refused = self._sweep(self._app(self._wrist(calibrated=False)), rig_id="wrist", mode="eye_in_hand").check()
        self.assertIn('--unmodelled-wrist-body "<reason>"', refused.refusal)


class AWristCameraSweepKeepsItsOwnHandlingTests(_Sweeps):

    def test_an_eye_in_hand_sweep_carries_its_own_placed_body_as_before(self) -> None:
        rig = self._wrist()
        report, robot = self._dry_run(self._sweep(self._app(rig), rig_id="wrist", mode="eye_in_hand"))
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        assert report.build is not None
        self.assertEqual(report.build.wrist, WristBodies.from_config(wrist_cell(), camera_section(rig)).line())
        self.assertEqual((report.build.unmodelled, report.check.wrist), ("", ""))

    def test_an_eye_in_hand_first_calibration_sweeps_without_its_body_as_before(self) -> None:
        calibration = self._sweep(self._app(self._wrist(calibrated=False)), rig_id="wrist", mode="eye_in_hand",
                                  reason="first calibration")
        self.assertEqual(calibration.check().wrist, "")
        report, robot = self._dry_run(calibration)
        self.assertEqual(self._held(robot), [])
        assert report.build is not None
        self.assertEqual(report.build.wrist, "")
        self.assertEqual(report.build.unmodelled,
                         "wrist camera 'wrist' swept without its body in the planner and the guard: first calibration")

    def test_an_eye_in_hand_first_calibration_without_a_reason_is_refused_as_before(self) -> None:
        check = self._sweep(self._app(self._wrist(calibrated=False)), rig_id="wrist", mode="eye_in_hand").check()
        self.assertFalse(check.ok)
        self.assertIn("camera.cameras.rigs['wrist'].body declares a camera the arm carries", check.refusal)
        self.assertTrue(check.refusal.endswith('To sweep this camera before its body can be placed, say why: '
                                               '--unmodelled-wrist-body "<reason>"'), check.refusal)

    def test_another_wrist_camera_the_tree_declares_is_carried_beside_the_one_calibrated(self) -> None:
        """The owner's rule for every sweep: a second camera on the arm hangs there while the first is calibrated."""
        app = self._app(self._wrist(), self._wrist("side"))
        report, robot = self._dry_run(self._sweep(app, rig_id="wrist", mode="eye_in_hand"))
        self.assertEqual(self._held(robot), ["wrist_camera_wrist", "wrist_camera_side"])
        assert report.build is not None
        self.assertIn("wrist camera wrist_camera_side", report.build.wrist)
        self.assertIn("wrist camera wrist_camera_side", report.check.wrist)

    def test_the_camera_an_eye_in_hand_sweep_calibrates_is_swept_without_its_body_and_named(self) -> None:
        """The owner's second decision of 2026-09-25: the camera a sweep calibrates eye in hand, declared with no body
        yet (the bracket not measured), is swept too, as the other wrist cameras are; nothing is carried for it, and
        the one warning line names it. It used to be refused ('is swept eye_in_hand and declares no body')."""
        calibration = self._sweep(self._app(self._wrist(body=False)), rig_id="wrist", mode="eye_in_hand")
        check = calibration.check()
        self.assertTrue(check.ok, check.render())
        self.assertEqual(check.without_body, _NO_BODY)
        report, robot = self._dry_run(calibration)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), [])
        assert report.build is not None
        self.assertEqual(report.build.without_body, _NO_BODY)

    def test_a_reason_given_for_a_camera_that_declares_no_body_adds_no_second_line(self) -> None:
        """Examples 09 and 10 always pass a reason; with no body declared, the one warning line names the camera,
        and no decline speaks of a body that is not there."""
        calibration = self._sweep(self._app(self._wrist(body=False)), rig_id="wrist", mode="eye_in_hand",
                                  reason="first calibration")
        report, _ = self._dry_run(calibration)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        assert report.build is not None
        self.assertEqual((report.build.without_body, report.build.unmodelled), (_NO_BODY, ""))
        self.assertEqual(report.build.render().count("!!"), 1, report.build.render())


class AWristRigWithoutABodyStopsNoCalibrationTests(_Sweeps):
    """The owner's decision of 2026-09-25: calibration goes on while the bracket is not measured yet."""

    def _warned(self, calibration: HandEyeCalibration) -> "tuple[CalibrationRunReport, Robot, list[str]]":
        """The check, then a dry run whose warnings are captured."""
        check = calibration.check()
        self.assertTrue(check.ok, check.refusal)
        with self.assertLogs("src.robot.execution.hand_eye", level="WARNING") as logged:
            report, robot = self._dry_run(calibration)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        return report, robot, logged.output

    def _one_line_each(self, report: CalibrationRunReport, logged: "list[str]", said: str = _NO_BODY) -> None:
        """The warning, once in the check, once in the build and once in the log."""
        assert report.build is not None
        self.assertEqual(report.check.render().count(said), 1, report.check.render())
        self.assertIn(f"\n  !! {said}", report.check.render())
        self.assertEqual(report.build.render().count(said), 1, report.build.render())
        self.assertIn(f"\n  !! {said}", report.build.render())
        self.assertEqual(sum(said in line for line in logged), 1, logged)

    def test_an_enabled_one_does_not_refuse_an_eye_to_hand_sweep_over_fixed_stations(self) -> None:
        """⛔ Before, the check refused: an enabled wrist rig without a body refused every sweep."""
        report, robot, logged = self._warned(self._sweep(self._app(self._wrist(body=False))))
        self.assertEqual(self._held(robot), [])
        self._one_line_each(report, logged)
        assert report.build is not None
        self.assertEqual((report.check.wrist, report.build.wrist, report.build.unmodelled), ("", "", ""))

    def test_an_enabled_one_does_not_refuse_an_eye_to_hand_sweep_guided_by_hand(self) -> None:
        report, robot, logged = self._warned(self._sweep(self._app(self._wrist(body=False)), freedrive=True))
        self.assertEqual(self._held(robot), [])
        self._one_line_each(report, logged)

    def test_a_switched_off_one_says_so_instead_of_none_declared(self) -> None:
        """⛔ Before, a switched off wrist rig without a body printed 'wrist cameras  none declared'."""
        report, robot, logged = self._warned(self._sweep(self._app(self._wrist(body=False, enabled=False))))
        self.assertEqual(self._held(robot), [])
        self._one_line_each(report, logged)
        assert report.build is not None
        self.assertNotIn("none declared", report.check.render())
        self.assertNotIn("none declared", report.build.render())

    def test_an_uncalibrated_one_without_a_body_warns_the_same(self) -> None:
        """No body and not calibrated yet: nothing to place and nothing to refuse, so the same one line."""
        rig = self._wrist(body=False)
        rig.extrinsics.artifact_path = str(self.root / "not_there.json")
        report, _, logged = self._warned(self._sweep(self._app(rig)))
        self._one_line_each(report, logged)

    def test_it_does_not_refuse_an_eye_in_hand_sweep_of_another_wrist_camera(self) -> None:
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                app = self._app(self._wrist(), self._wrist("side", body=False, enabled=enabled))
                for freedrive in (False, True):
                    report, robot, logged = self._warned(
                        self._sweep(app, rig_id="wrist", mode="eye_in_hand", freedrive=freedrive))
                    self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
                    self._one_line_each(report, logged, _NO_BODY.replace("'wrist'", "'side'"))

    def test_it_is_named_beside_a_body_that_is_carried(self) -> None:
        report, robot, logged = self._warned(self._sweep(self._app(self._wrist(), self._wrist("side", body=False))))
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist cameras  wrist camera wrist_camera_wrist", report.build.wrist)
        self._one_line_each(report, logged, _NO_BODY.replace("'wrist'", "'side'"))

    def test_two_are_named_on_one_line(self) -> None:
        app = self._app(self._wrist(body=False), self._wrist("side", body=False, enabled=False))
        report, _, logged = self._warned(self._sweep(app))
        self._one_line_each(report, logged,
                            "wrist cameras 'wrist', 'side' are declared on the arm without a body: the planner and "
                            "the guard do not know they are there during this sweep")

    def test_the_cli_checks_and_prints_the_line(self) -> None:
        stations = self.root / "stations.json"
        stations.write_text(json.dumps([{"x": 400.0, "y": 0.0, "z": 350.0, "rx": 0.0, "ry": 3.14159265, "rz": 0.0}]),
                            encoding="utf-8")
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=self._app(self._wrist(body=False))), \
                redirect_stdout(printed):
            code = calibrate.main(["--rig", "overhead", "--fixed-poses", str(stations), "--check"])
        self.assertEqual(code, calibrate._EXIT_OK, printed.getvalue())
        self.assertIn(f"\n  !! {_NO_BODY}\n", printed.getvalue())

    def test_picking_still_refuses_it_robot_from_tree_is_unchanged(self) -> None:
        """Calibration only: the door a pick builds its arm through still refuses an enabled one."""
        app = self._app(self._wrist(body=False))
        self.assertTrue(self._sweep(app).check().ok)
        tree = LoadedTree(tree=ConfigTree.from_directory(profile=None), config=app)
        with self.assertRaises(WristBodyRequired) as refused:
            Robot.from_tree(tree, gripper=None)
        self.assertIn("is an enabled eye_in_hand camera the arm carries and declares no body", str(refused.exception))
        with self.assertRaises(WristBodyRequired):
            WristBodies.from_config(app.robot, app.camera)

    def test_a_cell_that_reads_no_geometry_prints_no_line(self) -> None:
        app = self._app(self._wrist(body=False), robot=wrist_cell(planner="ik", backend="capsule"))
        check = self._sweep(app).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertNotIn("wrist", check.render().replace("'overhead'", ""))


class ARobotHandedInThatCarriesTheBodiesTests(_Sweeps):
    """``from_parts`` over a robot ``Robot.from_tree`` built, which already hands its arm every declared body."""

    def _robot(self, app: SimpleNamespace) -> Robot:
        return Robot.from_tree(LoadedTree(tree=ConfigTree.from_directory(profile=None), config=app), gripper=None)

    @staticmethod
    def _build_the_guard(robot: Robot) -> None:
        """Mark the exact mesh guard built, as its first judged path leaves it, and prove the arm now refuses."""
        guard = robot.arm._preflight._path_authority(robot.arm)  # type: ignore[attr-defined]  # noqa: SLF001
        guard._fcl_backend_built = True  # noqa: SLF001
        held = robot.arm._preflight.wrist_bodies(robot.arm)  # type: ignore[attr-defined]  # noqa: SLF001
        try:
            robot.arm.set_wrist_bodies(held)  # type: ignore[attr-defined]
        except RuntimeError as exc:
            assert _GUARD_BUILT in str(exc), exc
        else:  # pragma: no cover (the precondition)
            raise AssertionError("the guard took a hand-over after it was built")

    def _run(self, robot: Robot, section: Any, **kwargs: Any) -> CalibrationRunReport:
        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value = _Handle()
        calibration = HandEyeCalibration.from_parts(
            robot=robot, camera_config=section, rig_id=kwargs.pop("rig_id", "overhead"),
            mode=kwargs.pop("mode", "eye_to_hand"), options=SweepOptions(fixed_poses=_STATIONS))
        with patch(_CAMERA, camera_cls):
            return calibration.run(dry_run=True)

    def test_a_built_guard_that_holds_the_same_bodies_is_not_handed_them_again(self) -> None:
        """⛔ The regression: the sweep handed the arm the bodies it held, and the built guard refused the build."""
        app = self._app(self._wrist())
        robot = self._robot(app)
        self._build_the_guard(robot)
        report = self._run(robot, app.camera)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist cameras  wrist camera wrist_camera_wrist: realsense_d435", report.build.wrist)

    def test_an_eye_in_hand_sweep_of_the_camera_it_holds_is_not_handed_it_again(self) -> None:
        app = self._app(self._wrist(), self._wrist("side"))
        robot = self._robot(app)
        self._build_the_guard(robot)
        report = self._run(robot, app.camera, rig_id="wrist", mode="eye_in_hand")
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(sorted(self._held(robot)), ["wrist_camera_side", "wrist_camera_wrist"])

    def test_a_different_body_on_a_built_guard_still_fails_closed(self) -> None:
        robot = self._robot(self._app(self._wrist()))
        self._build_the_guard(robot)
        other = self._app(self._wrist(margin_mm=9.0))
        report = self._run(robot, other.camera)
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED, report.render())
        assert report.build is not None
        self.assertIn(_GUARD_BUILT, report.build.refusal)
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])

    def test_a_body_the_arm_misses_is_handed_beside_those_it_holds(self) -> None:
        robot = self._robot(self._app(self._wrist()))
        with patch.object(URRobotArm, "set_wrist_bodies", autospec=True,
                          side_effect=URRobotArm.set_wrist_bodies) as handed:
            report = self._run(robot, self._app(self._wrist(), self._wrist("side")).camera)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(handed.call_count, 1)
        self.assertEqual(sorted(self._held(robot)), ["wrist_camera_side", "wrist_camera_wrist"])
        assert report.build is not None
        self.assertIn("wrist_camera_side", report.build.wrist)
        self.assertIn("wrist_camera_wrist", report.build.wrist)

    def test_every_body_excused_hands_nothing_to_a_built_guard(self) -> None:
        """⛔ Verification of 2026-09-25: every declared body excused by a reason left nothing to hand, and the sweep
        handed the built guard that nothing all the same, which it refused."""
        reason = "the bracket is not measured yet"
        app = self._app(self._wrist(calibrated=False))
        robot = Robot.from_tree(LoadedTree(tree=ConfigTree.from_directory(profile=None), config=app), gripper=None,
                                unmodelled_wrist_body=reason)
        self._build_the_guard(robot)
        camera_cls = MagicMock(name="Camera")
        camera_cls.from_config.return_value.handle.return_value = _Handle()
        calibration = HandEyeCalibration.from_parts(
            robot=robot, camera_config=app.camera, rig_id="overhead", mode="eye_to_hand",
            options=SweepOptions(fixed_poses=_STATIONS, unmodelled_wrist_body=reason))
        with patch(_CAMERA, camera_cls):
            report = calibration.run(dry_run=True)
        self.assertIs(report.outcome, CalibrationOutcome.DRY_RUN, report.render())
        self.assertEqual(self._held(robot), [])

    def test_a_robot_swept_twice_is_not_handed_the_same_bodies_again(self) -> None:
        """⛔ Verification of 2026-09-25: the robot's record of its bodies is set when it is built, so the first sweep's
        hand-over was not in it, and a second sweep over a built guard was refused for handing the same bodies again."""
        app = self._app(self._wrist())
        robot = Robot.from_config(app.robot, gripper=None)
        first = self._run(robot, app.camera)
        self.assertIs(first.outcome, CalibrationOutcome.DRY_RUN, first.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])
        self._build_the_guard(robot)
        second = self._run(robot, app.camera)
        self.assertIs(second.outcome, CalibrationOutcome.DRY_RUN, second.render())
        self.assertEqual(self._held(robot), ["wrist_camera_wrist"])

    def test_a_missing_body_on_a_built_guard_fails_closed(self) -> None:
        robot = self._robot(self._app(self._wrist()))
        self._build_the_guard(robot)
        report = self._run(robot, self._app(self._wrist(), self._wrist("side")).camera)
        self.assertIs(report.outcome, CalibrationOutcome.BUILD_REFUSED, report.render())
        assert report.build is not None
        self.assertIn(_GUARD_BUILT, report.build.refusal)

    def test_a_body_the_guard_no_longer_holds_is_handed_again_though_the_build_record_names_it(self) -> None:
        """⛔ Review of 2026-09-25: the robot's record of its bodies is set once, when it is built, and counted as held
        beside what the guard holds. A sweep that replaced the body (a new calibration, another margin) left the
        record naming the old one, so a later sweep that carries the old body again handed nothing: the planner and
        the guard kept the replaced body while the build said the other."""
        app = self._app(self._wrist())
        robot = self._robot(app)
        self.assertEqual(self._margins(robot), [5.0])
        wider = self._app(self._wrist(margin_mm=9.0))
        first = self._run(robot, wider.camera)
        self.assertIs(first.outcome, CalibrationOutcome.DRY_RUN, first.render())
        self.assertEqual(self._margins(robot), [9.0], "the guard is not built, so the wider body was handed")
        second = self._run(robot, app.camera)
        self.assertIs(second.outcome, CalibrationOutcome.DRY_RUN, second.render())
        self.assertEqual(self._margins(robot), [5.0], "the body the tree declares now is the one the arm holds")

    @staticmethod
    def _margins(robot: Robot) -> list[float]:
        assert isinstance(robot.arm, URRobotArm)
        return [body.margin_mm for body in robot.arm._preflight.wrist_bodies(robot.arm)]  # type: ignore[attr-defined]  # noqa: SLF001


class ACameraHandedInWithoutItsSectionTests(_Sweeps):
    """The residual the ``from_parts`` docstring states: an owner holds its rig, not the tree's camera section."""

    def test_the_other_wrist_bodies_are_read_only_from_a_section_handed_in(self) -> None:
        app = self._app(self._wrist())
        owner = SimpleNamespace(rig_id="overhead", rig=_rgbd_rig("overhead"))
        alone = HandEyeCalibration.from_parts(camera=owner, robot_config=app.robot, mode="eye_to_hand",  # type: ignore[arg-type]
                                              options=SweepOptions(fixed_poses=_STATIONS)).check()
        self.assertTrue(alone.ok, alone.refusal)
        self.assertEqual(alone.wrist, "")
        with_section = HandEyeCalibration.from_parts(camera=owner, robot_config=app.robot, camera_config=app.camera,  # type: ignore[arg-type]
                                                     mode="eye_to_hand",
                                                     options=SweepOptions(fixed_poses=_STATIONS)).check()
        self.assertIn("wrist camera wrist_camera_wrist", with_section.wrist)
        doc = HandEyeCalibration.from_parts.__doc__ or ""
        self.assertIn("camera_config=", doc)
        self.assertIn("reads no other wrist camera", doc)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
