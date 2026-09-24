"""A sweep says per pose whether it counted and why not: the events, the printed line, the report and the check.

Issue 4 of the calibration chain (owner, 2026-09-23): an operator running examples 09 and 10 saw ``Pose i/N`` log
headers and could not tell whether the board was in view or why a pose did not count. Measured before this change
(the analysis's probes, and red here on the tree before it):

* ``marker_not_found`` wrote nothing on the console: the log jumped from one pose header to the next;
* the events dropped the move's status and message and the calibrator's diversity numbers;
* the CLI printed ``[robot_calibration_moving_to_pose] 1`` (the accepted count, not the pose), an empty line for a
  detection and a bare ``exception``;
* a sweep of poses from a JSON file reported ``accepted samples 5/-1``;
* ``check()`` passed a dictionary OpenCV does not know, a marker of -5 mm and an id the dictionary does not hold.

Nothing here changes which move is commanded or which sample counts: the routine is driven over an arm double whose
moves are the ones it was asked for, and the events are compared, not the motion.
"""

from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from src.calibration import MountingMode
from src.calibration.exceptions import CalibrationDataError
from src.calibration.targets import CharucoPoseEstimator, Observation
from src.config.schema.camera import HandEyeConfig
from src.config.schema.robot import RobotConfig, WorkspaceLimitsConfig
from src.camera.orchestration.camera import Camera
from src.geometry import Frame, Pose
from src.robot.core.motion_result import MotionCommand, MotionResult, MotionStatus
from src.robot.events import RobotCalibrationEvent
from src.robot.execution.calibration import CalibrationResult, CalibrationRoutine, PoseVerdict
from src.robot.execution.hand_eye import (
    CalibrationCheck,
    CalibrationOutcome,
    CalibrationRunReport,
    HandEyeCalibration,
    SweepOptions,
    print_sweep_progress,
    render_sweep_event,
)
from src.robot.execution.real_cell import calibrate
from src.robot.execution.robot import Robot
from src.robot.drivers.dummy.arm import DummyRobotArm
from tests.test_camera_boundaries import _rgbd_rig
from tests.test_robot_boundaries import _inverse, _synthetic_eye_to_hand_data

_WIDE = WorkspaceLimitsConfig(x_min=-2000.0, x_max=2000.0, y_min=-2000.0, y_max=2000.0, z_min=-2000.0, z_max=2000.0)
_BOARD = "charuco:7x5:30:22"
#: The stations a sweep of these tests names: nothing generates one, so a sweep that names none is refused.
_STATIONS = [Pose.tool_down(400.0, 0.0, 350.0, label="down_0"), Pose.tool_down(420.0, 60.0, 380.0, label="down_1")]


class _Arm:
    """Arrives where it is sent, except at the labels it refuses, with a status and a message."""

    def __init__(self, refuse: dict[str, tuple[MotionStatus, str]]) -> None:
        self.refuse = refuse
        self.at: Pose | None = None

    def move(self, pose: Pose, **_keywords: Any) -> MotionResult:
        if pose.label in self.refuse:
            status, message = self.refuse[pose.label]
            return MotionResult.failed(status, MotionCommand.MOVE_TO, target_pose=pose, message=message)
        self.at = pose
        return MotionResult.executed(MotionCommand.MOVE_TO, target_pose=pose)

    def get_tcp_pose(self) -> Pose:
        assert self.at is not None
        return self.at


class _Source:
    """A marker source that keeps ``last_observation`` as the RGB-D source does, with a verdict per call."""

    def __init__(self, answers: list[Any]) -> None:
        self.answers = iter(answers)
        self.last_observation: Observation | None = None

    def __call__(self) -> np.ndarray | None:
        answer = next(self.answers)
        if isinstance(answer, str):
            self.last_observation = Observation(kind="aruco", why_not=answer)
            return None
        points = np.zeros((4, 2))
        self.last_observation = Observation(kind="aruco", T_cam_to_target=answer, points_px=points,
                                            reprojection_px=0.25, marker_ids=(0,))
        return answer


def _sweep(*, min_samples: int = 4) -> tuple[CalibrationRoutine, list[Pose], list[tuple[str, dict]]]:
    """Seven poses: 1-2 count, 3 sees no marker, 4 repeats pose 1, 5 is refused by the arm, 6-7 count."""
    T_cam_to_base, T_tool_to_marker, tool = _synthetic_eye_to_hand_data()
    matrices = [tool[0], tool[1], tool[2], tool[0], tool[3], tool[4], tool[5]]
    poses = [Pose.from_matrix(T, frame=Frame.BASE, label=f"p{i + 1}") for i, T in enumerate(matrices)]
    seen = [_inverse(T_cam_to_base) @ T @ T_tool_to_marker for T in matrices]
    answers = [seen[0], seen[1], "no DICT_5X5_100 marker in view", seen[3], seen[5], seen[6]]
    events: list[tuple[str, dict]] = []
    routine = CalibrationRoutine(
        arm=_Arm({"p5": (MotionStatus.WORKSPACE_REJECTED, "planned path hits the table")}),  # type: ignore[arg-type]
        marker_source=_Source(answers), workspace_limits=_WIDE, calibration_mode="eye_to_hand", settle_time_s=0.0,
        eth_settings=SimpleNamespace(mode="eye_to_hand", min_samples=min_samples, min_distance_mm=40.0,
                                     min_angle=10.0, min_angle_deg=10.0),  # type: ignore[arg-type]
        on_event=lambda kind, data: events.append((kind, dict(data))),
    )
    return routine, poses, events


def _rejections(events: list[tuple[str, dict]]) -> dict[str, dict]:
    return {data["reason"]: data for kind, data in events if kind == RobotCalibrationEvent.POSE_REJECTED}


# ---------------------------------------------------------------------------------------------------------------------
# The routine's events carry their reasons
# ---------------------------------------------------------------------------------------------------------------------


class TheEventsSayWhyTests(unittest.TestCase):

    def setUp(self) -> None:
        self.routine, self.poses, self.events = _sweep()
        with self.assertLogs(self.routine.logger, level="INFO") as self.logs:
            self.result = self.routine.run_with_poses(self.poses)

    def test_a_refused_move_carries_its_status_and_message(self) -> None:
        refused = _rejections(self.events)["move_rejected"]
        self.assertEqual((refused["index"], refused["label"]), (5, "p5"))
        self.assertEqual((refused["status"], refused["message"]), ("workspace_rejected", "planned path hits the table"))
        self.assertEqual(refused["detail"], "workspace_rejected: planned path hits the table")

    def test_a_missing_marker_carries_why_and_is_one_warning_on_the_console(self) -> None:
        """Red before: the event carried the marker id only, and the log went straight to the next pose."""
        missing = _rejections(self.events)["marker_not_found"]
        self.assertEqual(missing["why_not"], "no DICT_5X5_100 marker in view")
        self.assertEqual(missing["detail"], "no DICT_5X5_100 marker in view")
        warnings = [line for line in self.logs.output if "WARNING" in line and "no marker pose" in line]
        self.assertEqual(warnings, ["WARNING:CalibrationRoutine:Pose 3/7 'p3': no marker pose: no DICT_5X5_100 "
                                    "marker in view"])

    def test_a_refused_sample_carries_the_nearest_stored_sample_and_the_thresholds(self) -> None:
        refused = _rejections(self.events)["sample_rejected"]
        self.assertEqual(refused["index"], 4)
        self.assertEqual(refused["rejection"], {
            "reason": "not_diverse", "stored": 2, "nearest_sample": 1, "nearest_mm": 0.0, "nearest_deg": 0.0,
            "min_distance_mm": 40.0, "min_angle_deg": 10.0})
        self.assertEqual(refused["detail"], "pose not diverse: stored sample 1 is 0.0 mm and 0.0 deg away, and a new "
                                            "pose needs > 40.0 mm or > 10.0 deg from every one of the 2 stored")

    def test_a_detection_carries_what_the_pose_rests_on(self) -> None:
        detected = [data for kind, data in self.events if kind == RobotCalibrationEvent.MARKER_DETECTED]
        self.assertEqual([data["index"] for data in detected], [1, 2, 4, 6, 7])
        first = detected[0]
        self.assertEqual((first["label"], first["n_points"], first["reprojection_px"], first["hint"]),
                         ("p1", 4, 0.25, ""))
        self.assertGreater(first["distance_mm"], 0.0)

    def test_every_event_names_its_pose_and_the_move_says_how_many_are_needed(self) -> None:
        self.assertTrue(all(data["label"] == f"p{data['index']}" for _, data in self.events))
        moving = [data for kind, data in self.events if kind == RobotCalibrationEvent.MOVING_TO_POSE]
        self.assertEqual({data["min_samples"] for data in moving}, {4})

    def test_the_pose_log_holds_every_verdict_in_order(self) -> None:
        self.assertEqual([(v.index, v.counted, v.reason) for v in self.result.pose_log], [
            (1, True, ""), (2, True, ""), (3, False, "marker_not_found"), (4, False, "sample_rejected"),
            (5, False, "move_rejected"), (6, True, ""), (7, True, "")])
        self.assertEqual(self.result.pose_log, self.routine.pose_log)
        self.assertEqual(self.result.num_samples, 4)
        self.assertRegex(self.result.pose_log[0].detail, r"^4 corners, 0\.25 px, \d+ mm away$")

    def test_the_moves_commanded_are_the_poses_given(self) -> None:
        """What changed is what is said, not what moves: every pose is commanded once, in order."""
        self.assertEqual(len(self.result.camera_worlds), 7)


class TheRgbdSourceFeedsTheEventsTests(unittest.TestCase):
    """The seam end to end: a rendered ChArUco board through ``RGBDArucoMarkerSource`` into the routine's events."""

    def test_a_board_pose_rests_on_its_corners_in_the_event_and_the_log(self) -> None:
        from src.calibration.rgbd_marker_source import RGBDArucoMarkerSource
        from tests.test_calibration_targets import _K, _board_view

        class _Streamer:
            def grab(self) -> Any:
                return SimpleNamespace(color=_board_view())

            def get_intrinsics(self) -> np.ndarray:
                return _K

            def get_distortion(self) -> None:
                return None

        judged: list[Observation] = []
        source = RGBDArucoMarkerSource(
            streamer=_Streamer(), warmup_grabs=0, on_observation=lambda _bgr, seen, _K, _dist: judged.append(seen),
            estimator=CharucoPoseEstimator(squares_x=7, squares_y=5, square_length_mm=30.0, marker_length_mm=22.0))
        events: list[tuple[str, dict]] = []
        _, _, tool = _synthetic_eye_to_hand_data()
        routine = CalibrationRoutine(
            arm=_Arm({}), marker_source=source, workspace_limits=_WIDE,  # type: ignore[arg-type]
            calibration_mode="eye_to_hand", settle_time_s=0.0, marker_id=-1,
            eth_settings=SimpleNamespace(mode="eye_to_hand", min_samples=4, min_distance_mm=40.0,
                                         min_angle=10.0, min_angle_deg=10.0),  # type: ignore[arg-type]
            on_event=lambda kind, data: events.append((kind, dict(data))))
        with self.assertRaises(CalibrationDataError):  # one pose cannot solve; the events are what is held
            routine.run_with_poses([Pose.from_matrix(tool[0], frame=Frame.BASE, label="look_0")])
        (detected,) = [data for kind, data in events if kind == RobotCalibrationEvent.MARKER_DETECTED]
        self.assertEqual((detected["n_points"], detected["marker_id"]), (24, -1))
        self.assertLess(detected["reprojection_px"], 1.0)
        self.assertEqual(len(judged), 1)
        self.assertEqual(routine.pose_log[0].detail, judged[0].summary())
        self.assertTrue(routine.pose_log[0].counted)


class ThePoseLogOutlivesARaisingSolveTests(unittest.TestCase):

    def test_a_solve_refused_for_too_few_samples_leaves_the_log_on_the_routine(self) -> None:
        routine, poses, _ = _sweep(min_samples=6)
        with self.assertRaises(CalibrationDataError):
            routine.run_with_poses(poses)
        self.assertEqual(sum(v.counted for v in routine.pose_log), 4)
        self.assertEqual(len(routine.pose_log), 7)

    def test_a_routine_built_without_init_has_the_defaults(self) -> None:
        bare = object.__new__(CalibrationRoutine)
        self.assertEqual((bare.pose_log, bare._last_move), ((), None))  # noqa: SLF001
        self.assertIsNone(bare._last_observation())  # noqa: SLF001


# ---------------------------------------------------------------------------------------------------------------------
# One line per event
# ---------------------------------------------------------------------------------------------------------------------


class RenderSweepEventTests(unittest.TestCase):

    def test_each_event_in_its_words(self) -> None:
        cases = [
            (RobotCalibrationEvent.MOVING_TO_POSE, {"index": 3, "total": 22, "label": "look_2", "accepted": 2,
                                                    "min_samples": 6},
             "pose  3/22 'look_2'  moving, 2 counted so far, need 6"),
            (RobotCalibrationEvent.MARKER_DETECTED, {"index": 3, "total": 22, "label": "look_2", "n_points": 24,
                                                     "reprojection_px": 0.3125, "distance_mm": 519.6, "hint": ""},
             "pose  3/22 'look_2'  target seen: 24 corners, 0.31 px, 520 mm away"),
            (RobotCalibrationEvent.POSE_ACCEPTED, {"index": 3, "total": 22, "label": "look_2", "accepted": 3,
                                                   "min_samples": 6},
             "pose  3/22 'look_2'  COUNTED 3 (need 6)"),
            (RobotCalibrationEvent.POSE_REJECTED, {"index": 4, "total": 22, "label": "look_3",
                                                   "reason": "marker_not_found",
                                                   "detail": "no DICT_5X5_100 marker in view"},
             "pose  4/22 'look_3'  REJECTED marker_not_found: no DICT_5X5_100 marker in view"),
            (RobotCalibrationEvent.POSE_REJECTED, {"index": 12, "total": 22, "reason": "exception",
                                                   "detail": "RealSense frame timeout"},
             "pose 12/22  REJECTED exception: RealSense frame timeout"),
        ]
        for kind, data, line in cases:
            with self.subTest(line):
                self.assertEqual(render_sweep_event(kind, data), line)

    def test_the_move_line_shows_the_pose_not_the_accepted_count(self) -> None:
        """Red before: the CLI printed `[robot_calibration_moving_to_pose] 1`, the accepted count."""
        line = render_sweep_event(RobotCalibrationEvent.MOVING_TO_POSE, {"index": 7, "total": 22, "accepted": 1})
        self.assertTrue(line.startswith("pose  7/22"), line)

    def test_a_hint_is_marked_and_a_payload_with_less_still_prints(self) -> None:
        hint = "17 DICT_5X5_100 markers visible: is this a ChArUco board?"
        self.assertEqual(render_sweep_event(RobotCalibrationEvent.MARKER_DETECTED,
                                            {"index": 1, "total": 2, "distance_mm": 1200.0, "hint": hint}),
                         f"pose  1/2  target seen: 1200 mm away; !! {hint}")
        self.assertEqual(render_sweep_event(RobotCalibrationEvent.POSE_REJECTED, {}), "pose  REJECTED rejected")
        self.assertEqual(render_sweep_event("something_new", {"index": 1, "total": 1}), "pose  1/1  something_new")

    def test_the_line_is_ascii(self) -> None:
        line = render_sweep_event(RobotCalibrationEvent.POSE_REJECTED,
                                  {"index": 1, "total": 1, "label": "ecke_ü", "reason": "exception",
                                   "detail": "Zeitüberschreitung"})
        self.assertTrue(line.isascii(), line)

    def test_the_printer_indents_one_line_per_event(self) -> None:
        printed = io.StringIO()
        with redirect_stdout(printed):
            print_sweep_progress(RobotCalibrationEvent.POSE_ACCEPTED, {"index": 1, "total": 2, "accepted": 1})
        self.assertEqual(printed.getvalue(), "  pose  1/2  COUNTED 1\n")

    def test_the_cli_and_both_sim_runners_print_through_it(self) -> None:
        root = Path(__file__).resolve().parents[1]
        for path in ("src/robot/execution/real_cell/calibrate.py", "src/willy_sim/run_eih_calibrate.py",
                     "src/willy_sim/run_eth_calibrate.py"):
            with self.subTest(path):
                source = (root / path).read_text(encoding="utf-8")
                self.assertIn("on_event=print_sweep_progress", source)
                self.assertNotIn("data.get('reason', data.get('accepted'", source)


# ---------------------------------------------------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------------------------------------------------


def _check(poses: int = 22) -> CalibrationCheck:
    return CalibrationCheck(rig_id="wrist", mode=MountingMode.EYE_IN_HAND, rig_source="rgbd", vendor="ur",
                            marker_length_mm=50.0, dict_name="DICT_5X5_100", poses=poses,
                            artifact_path="calibration/real/eih_wrist.json")


_LOG = (
    PoseVerdict(1, "look_0", True, detail="4 corners, 0.21 px, 402 mm away"),
    PoseVerdict(2, "look_1", False, reason="marker_not_found", detail="no DICT_5X5_100 marker in view"),
    PoseVerdict(3, "", False, reason="move_rejected", detail="workspace_rejected: planned path hits the table"),
)


class TheReportListsEveryPoseTests(unittest.TestCase):

    def test_the_count_is_against_the_poses_the_sweep_ran_else_the_stations_it_named(self) -> None:
        """Red before: `accepted samples  5/-1` for poses read from a JSON file. The check now counts every file."""
        report = CalibrationRunReport(check=_check(poses=6), outcome=CalibrationOutcome.NO_ARTIFACT,
                                      accepted_samples=1, pose_log=_LOG)
        self.assertIn("  accepted samples  1/3\n", report.summary())
        without_log = CalibrationRunReport(check=_check(poses=6), outcome=CalibrationOutcome.NO_ARTIFACT,
                                           accepted_samples=5)
        self.assertIn("  accepted samples  5/6\n", without_log.summary())

    def test_the_result_stage_lists_every_pose(self) -> None:
        report = CalibrationRunReport(check=_check(), outcome=CalibrationOutcome.NO_ARTIFACT, accepted_samples=1,
                                      pose_log=_LOG)
        summary = report.summary()
        self.assertIn("\n".join((
            "=== 4. RESULT ===",
            "  per pose          1 of 3 counted",
            "      1  'look_0'  COUNTED   4 corners, 0.21 px, 402 mm away",
            "      2  'look_1'  REJECTED  marker_not_found: no DICT_5X5_100 marker in view",
            "      3            REJECTED  move_rejected: workspace_rejected: planned path hits the table",
            "  accepted samples  1/3",
        )), summary)
        self.assertEqual(report.to_dict()["pose_log"][1], {"index": 2, "label": "look_1", "counted": False,
                                                           "reason": "marker_not_found",
                                                           "detail": "no DICT_5X5_100 marker in view"})

    def test_a_sweep_that_raised_lists_the_poses_it_visited(self) -> None:
        report = CalibrationRunReport(check=_check(), outcome=CalibrationOutcome.SWEEP_FAILED,
                                      failure="CalibrationDataError: too few samples: got 1, need >= 6",
                                      pose_log=_LOG)
        self.assertEqual(report.summary().splitlines()[-5:], [
            "[sweep] FAILED: CalibrationDataError: too few samples: got 1, need >= 6",
            "  per pose          1 of 3 counted",
            "      1  'look_0'  COUNTED   4 corners, 0.21 px, 402 mm away",
            "      2  'look_1'  REJECTED  marker_not_found: no DICT_5X5_100 marker in view",
            "      3            REJECTED  move_rejected: workspace_rejected: planned path hits the table",
        ])

    def test_no_log_prints_no_table(self) -> None:
        report = CalibrationRunReport(check=_check(), outcome=CalibrationOutcome.NO_ARTIFACT, accepted_samples=20)
        self.assertNotIn("per pose", report.summary())
        self.assertIn("  accepted samples  20/22\n", report.summary())


# ---------------------------------------------------------------------------------------------------------------------
# check(): the target is validated before anything is built
# ---------------------------------------------------------------------------------------------------------------------


def _app(hand_eye: Any = None) -> SimpleNamespace:
    return SimpleNamespace(
        robot=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41"}}),
        camera=SimpleNamespace(cameras=SimpleNamespace(rigs=[_rgbd_rig("overhead")]),
                               hand_eye=hand_eye if hand_eye is not None else HandEyeConfig()))


def _noun(hand_eye: Any = None, **options: Any) -> HandEyeCalibration:
    options.setdefault("fixed_poses", _STATIONS)
    return HandEyeCalibration.from_config(_app(hand_eye), rig_id="overhead", mode="eye_to_hand",
                                          options=SweepOptions(**options))


class TheCheckValidatesTheTargetTests(unittest.TestCase):

    def test_what_check_passed_before_is_refused(self) -> None:
        """Red before (VERIFIED by probe): check() was ok for each of these and printed them."""
        for options, said in (
            ({"dict_name": "BOGUS_DICT"}, "unknown ArUco dictionary 'BOGUS_DICT'"),
            ({"marker_length_mm": -5.0}, "marker_length_mm: Input should be greater than 0"),
            ({"marker_id": 9999}, "marker_id 9999 is not in DICT_5X5_100, whose ids run from 0 to 99"),
        ):
            with self.subTest(options):
                check = _noun(**options).check()
                self.assertFalse(check.ok)
                self.assertIn(said, check.refusal)
                self.assertTrue(check.render().startswith("[config] REFUSED: calibration target: "), check.render())

    def test_a_dictionary_is_read_as_the_estimator_reads_it(self) -> None:
        """`--dict 5x5_100` worked at the build before; the strict schema name must not refuse it now."""
        check = _noun(dict_name="5x5_100").check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.render().splitlines()[3], "  marker     50.0 mm, id 0, DICT_5X5_100")

    def test_a_marker_renders_as_it_did(self) -> None:
        check = _noun().check()
        self.assertEqual(check.render().splitlines()[3], "  marker     50.0 mm, id 0, DICT_5X5_100")
        self.assertEqual(check.target, "")

    def test_the_marker_id_comes_from_the_tree(self) -> None:
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"marker_id": 7}})
        self.assertEqual(_noun(tree).check().render().splitlines()[3], "  marker     50.0 mm, id 7, DICT_5X5_100")

    def test_a_marker_target_in_the_tree_is_the_marker_line_and_an_option_still_wins(self) -> None:
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"target": {"kind": "aruco", "marker_id": 3,
                                                                        "marker_length_mm": 40.0}}})
        self.assertEqual(_noun(tree).check().render().splitlines()[3], "  marker     40.0 mm, id 3, DICT_5X5_100")
        self.assertEqual(_noun(tree, marker_length_mm=39.7).check().render().splitlines()[3],
                         "  marker     39.7 mm, id 3, DICT_5X5_100")

    def test_a_board_renders_as_a_board(self) -> None:
        check = _noun(target=_BOARD).check()
        self.assertTrue(check.ok, check.refusal)
        self.assertEqual(check.render().splitlines()[3],
                         "  board      charuco 7x5, square 30.0 mm, marker 22.0 mm, DICT_5X5_100, at least 6 corners")
        self.assertEqual((check.marker_id, check.marker_length_mm), (-1, 22.0))

    def test_a_board_in_the_tree_is_used_and_refuses_the_single_marker_options(self) -> None:
        tree = HandEyeConfig.model_validate({"eye_to_hand": {"target": {
            "kind": "charuco", "squares_x": 10, "squares_y": 7, "square_length_mm": 26.0, "marker_length_mm": 20.0}}})
        self.assertIn("charuco 10x7, square 26.0 mm", _noun(tree).check().render())
        refused = _noun(tree, marker_id=0).check()
        self.assertIn("camera.hand_eye.eye_to_hand.target names the whole target", refused.refusal)
        self.assertIn("--marker-id", refused.refusal)

    def test_a_target_given_and_a_single_marker_option_are_refused_together(self) -> None:
        refusal = _noun(target=_BOARD, marker_id=3).check().refusal
        self.assertIn(f"calibration target {_BOARD!r} names the whole target", refusal)

    def test_a_bad_board_is_refused_at_check(self) -> None:
        self.assertIn("must be smaller than square_length_mm", _noun(target="charuco:7x5:22:30").check().refusal)
        self.assertIn("the squares are written XxY", _noun(target="charuco:7:30:22").check().refusal)

    def test_a_board_file_is_read_and_validated_like_a_spec(self) -> None:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        good = folder / "board.yaml"
        good.write_text("kind: charuco\nsquares_x: 7\nsquares_y: 5\nsquare_length_mm: 30\nmarker_length_mm: 22\n",
                        encoding="utf-8")
        self.assertEqual(_noun(target=good).check().target, _noun(target=_BOARD).check().target)
        bad = folder / "board.json"
        bad.write_text('{"kind": "charuco", "squares_x": 20, "squares_y": 12, "square_length_mm": 30, '
                       '"marker_length_mm": 22}', encoding="utf-8")
        self.assertIn(f"file {bad}: a 20x12 board carries 120 markers", _noun(target=bad).check().refusal)
        listed = folder / "list.yaml"
        listed.write_text("- 1\n- 2\n", encoding="utf-8")
        self.assertIn("holds no mapping", _noun(target=listed).check().refusal)


# ---------------------------------------------------------------------------------------------------------------------
# The build poses the target the check validated
# ---------------------------------------------------------------------------------------------------------------------


class _Streamer:
    def open(self) -> None:
        pass

    def release(self) -> None:
        pass

    def grab(self) -> Any:
        raise AssertionError("a frame was grabbed")

    def get_intrinsics(self) -> np.ndarray:
        return np.eye(3)

    def get_distortion(self) -> None:
        return None


class TheBuildPosesTheCheckedTargetTests(unittest.TestCase):

    def _parts(self, **options: Any) -> Any:
        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=_Streamer())
        self.addCleanup(camera.release)
        calibration = HandEyeCalibration.from_parts(
            robot=Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None), camera=camera,
            robot_config=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41"}}),
            mode="eye_to_hand", options=SweepOptions(fixed_poses=_STATIONS, **options))
        build, parts = calibration._build(calibration._stage())  # noqa: SLF001
        self.assertTrue(build.ok, build.refusal)
        self.addCleanup(parts.give_back)
        return parts

    def test_a_board_is_posed_by_a_board_estimator_and_no_marker_id_is_recorded(self) -> None:
        parts = self._parts(target=_BOARD)
        source = parts.routine._marker_source  # noqa: SLF001
        self.assertIsInstance(source.estimator, CharucoPoseEstimator)
        self.assertEqual(source.estimator.squares, (7, 5))
        self.assertEqual(parts.routine.marker_id, -1)

    def test_a_marker_is_posed_with_its_normalised_dictionary_and_id(self) -> None:
        parts = self._parts(dict_name="4x4_50", marker_id=7, marker_length_mm=40.0)
        estimator = parts.routine._marker_source.estimator  # noqa: SLF001
        self.assertEqual((estimator.dict_name, estimator.target_id, estimator.marker_length), ("DICT_4X4_50", 7, 40.0))
        self.assertEqual(parts.routine.marker_id, 7)


class TheSweepReportsItsPosesTests(unittest.TestCase):
    """The noun hands the routine's pose log to the report, on a solve and on a sweep that raised."""

    def _run(self, sweep: Any) -> CalibrationRunReport:
        camera = Camera.from_rig(_rgbd_rig("overhead"), streamer=_Streamer())
        self.addCleanup(camera.release)
        out = self.enterContext(tempfile.TemporaryDirectory())
        calibration = HandEyeCalibration.from_parts(
            robot=Robot.from_parts(arm=DummyRobotArm(), gripper=None, lock_key=None), camera=camera,
            robot_config=RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "10.253.253.41"}}),
            mode="eye_to_hand", options=SweepOptions(out_dir=out, fixed_poses=_STATIONS))
        with patch.object(CalibrationRoutine, "run_with_poses", autospec=True, side_effect=sweep):
            return calibration.run()

    def test_a_sweep_that_raised_still_lists_its_poses(self) -> None:
        def too_few(routine: CalibrationRoutine, *_args: Any, **_keywords: Any) -> None:
            routine.pose_log = _LOG
            raise CalibrationDataError("too few samples: got 1, need >= 6")

        report = self._run(too_few)
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertEqual(report.pose_log, _LOG)
        self.assertIn("  per pose          1 of 3 counted", report.render())

    def test_a_solve_lists_its_poses_in_the_result_stage(self) -> None:
        def solved(_routine: CalibrationRoutine, *_args: Any, **_keywords: Any) -> CalibrationResult:
            return CalibrationResult(T_cam_to_base=None, rmse_mm=9.0, max_error_mm=9.0, num_samples=1,
                                     mode=MountingMode.EYE_TO_HAND, pose_log=_LOG)

        report = self._run(solved)
        self.assertIs(report.outcome, CalibrationOutcome.NO_ARTIFACT)
        self.assertIn("  accepted samples  1/3", report.render())


# ---------------------------------------------------------------------------------------------------------------------
# The CLI and the sim runners
# ---------------------------------------------------------------------------------------------------------------------


class TheCliNamesTheTargetTests(unittest.TestCase):

    def _check(self, *argv: str, hand_eye: Any = None) -> tuple[Any, str]:
        printed = io.StringIO()
        with patch.object(calibrate, "_load", return_value=_app(hand_eye)), redirect_stdout(printed):
            try:
                code: Any = calibrate.main(["--rig", "overhead", "--check", "--freedrive", *argv])
            except SystemExit as exc:
                code = f"exit {exc.code}"
        return code, printed.getvalue()

    def test_a_board_is_checked_and_printed(self) -> None:
        code, printed = self._check("--board", _BOARD)
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertIn("  board      charuco 7x5, square 30.0 mm, marker 22.0 mm, DICT_5X5_100, at least 6 corners\n",
                      printed)

    def test_a_board_with_a_marker_id_is_refused_at_check(self) -> None:
        code, printed = self._check("--board", _BOARD, "--marker-id", "3")
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertIn("[config] REFUSED: calibration target", printed)

    def test_a_board_file_is_checked_like_a_board(self) -> None:
        board = Path(self.enterContext(tempfile.TemporaryDirectory())) / "board.yaml"
        board.write_text("{kind: charuco, squares_x: 7, squares_y: 5, square_length_mm: 30, marker_length_mm: 22}\n",
                         encoding="utf-8")
        code, printed = self._check("--board-file", str(board))
        self.assertEqual(code, calibrate._EXIT_OK, printed)
        self.assertEqual(printed, self._check("--board", _BOARD)[1])

    def test_board_and_board_file_are_one_or_the_other(self) -> None:
        stderr = io.StringIO()
        with patch("sys.stderr", stderr):
            code, _ = self._check("--board", _BOARD, "--board-file", "board.yaml")
        self.assertEqual(code, "exit 2")
        self.assertIn("not allowed with argument", stderr.getvalue())

    def test_a_bad_dictionary_is_refused_at_check(self) -> None:
        code, printed = self._check("--dict", "BOGUS_DICT")
        self.assertEqual(code, calibrate._EXIT_CONFIG)
        self.assertIn("unknown ArUco dictionary 'BOGUS_DICT'", printed)


class TheSimRunnersRefuseABoardTests(unittest.TestCase):

    def test_a_board_is_refused_in_one_sentence_and_a_marker_is_the_blocks(self) -> None:
        from src.willy_sim.run_eih_calibrate import sim_aruco_target

        board = HandEyeConfig.model_validate({"eye_in_hand": {"target": {
            "kind": "charuco", "squares_x": 7, "squares_y": 5, "square_length_mm": 30.0, "marker_length_mm": 22.0}}})
        with self.assertRaises(SystemExit) as caught:
            sim_aruco_target(board.eye_in_hand, "camera.hand_eye.eye_in_hand")
        said = str(caught.exception.code)
        self.assertTrue(said.startswith("camera.hand_eye.eye_in_hand.target is a charuco board"), said)
        self.assertEqual(1, len(said.splitlines()))
        marker = HandEyeConfig.model_validate({"eye_in_hand": {"marker_length_mm": 48.0,
                                                               "aruco_dict_name": "DICT_4X4_50"}})
        target = sim_aruco_target(marker.eye_in_hand, "camera.hand_eye.eye_in_hand")
        self.assertEqual((target.marker_length_mm, target.aruco_dict_name), (48.0, "DICT_4X4_50"))


class TheExamplesPrintTheSweepTests(unittest.TestCase):

    def test_examples_07_to_10_print_progress_and_hard_code_no_marker(self) -> None:
        root = Path(__file__).resolve().parents[1] / "examples" / "real_robot"
        for number in ("07", "08", "09", "10"):
            (path,) = sorted(root.glob(f"{number}_*.py"))
            with self.subTest(path.name):
                source = path.read_text(encoding="utf-8")
                self.assertIn("on_event=print_sweep_progress", source)
                self.assertNotIn("marker_length_mm=40.0", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
