"""An aimed sweep turns the camera about its own line of sight, and finds a heading that admits its views.

Review 4 (2026-09-23) found two things wrong with the owner's aimed sweep (a D415 tilted 45 degrees on a bracket beside
the Hand-E, a 150 mm marker at (-130, -700, 50) mm, a UR10 box less 20 mm):

* every one of the 17 views kept the tool's heading, so the camera never turned about its own axis and the AX=XB solve
  was 2 to 4 times worse than it had to be: at K2's noise a point 500 mm down the optical axis landed about 5 mm off.
  Each default view now also rolls the camera about its line of sight (-45, 0 or +45 degrees), the camera standing
  where it stood and looking where it looked;
* the heading was fixed in code, so a camera tilted toward the tool's -X stood every view outside the box (0 of 17)
  and the sweep ended "too few samples" with no word about the heading. A heading that admits fewer views than the
  solve needs now gives way to the one of -y, y, x, -x that admits the most, and the report and the log say so. The
  viewing pose does the same where its heading admits no view.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from src.calibration.exceptions import CalibrationDataError
from src.config.schema.robot import WorkspaceLimitsConfig
from src.geometry import Pose
from src.robot.core import MotionCommand, MotionResult, MotionStatus
from src.robot.execution import camera_aim as aim_module
from src.robot.execution.calibration import CalibrationRoutine
from src.robot.execution.camera_aim import (
    DEFAULT_VIEWS,
    HEADINGS,
    HeadingChoice,
    MarkerAim,
    MountEstimate,
    aim_camera,
    aimed_stations,
    choose_heading,
)
from src.robot.execution.hand_eye import CalibrationOutcome, SweepOptions
from tests.test_a_wrist_camera_is_aimed_not_the_tool import (
    ALONG,
    LOOK,
    MARKER,
    _Arm,
    _aim,
    _eih_result,
    _estimate,
    _in_view,
    _miss_deg,
    _mount,
    _routine,
    _seen,
)

#: The UR10 profile's box; the sweep screens it less the gate's 20 mm.
UR10_BOX = WorkspaceLimitsConfig(x_min=-760.7, x_max=760.7, y_min=-760.7, y_max=760.7, z_min=100.0, z_max=748.4)
#: The camera tilted the other way: 45 degrees toward the tool's -X, on a bracket out that way.
AGAINST = _mount((-70.0, 0.0, -150.0), tilt_deg=-45.0)
#: Where the owner of that camera stands the arm to look: the mirror of LOOK, the tool's +X along base +Y.
LOOK_AGAINST = Pose.tool_down(-100.0, -300.0, 280.0, closing_axis="y", yaw_deg=10.0)


def _truth(mount: np.ndarray) -> MountEstimate:
    return MountEstimate(camera_in_tool=aim_module._as_tuple(mount), source="truth", marker_mm=tuple(MARKER))


class _Refusing(_Arm):
    """An arm whose planning refuses the stations named, with nothing moved, as cuRobo refuses a folded forearm."""

    def __init__(self, at: Pose, *, refuse: set[str]) -> None:
        super().__init__(at)
        self.refuse = refuse
        self.refused: list[str] = []

    def move(self, pose: Pose, **_: Any) -> MotionResult:
        if pose.label in self.refuse:
            self.refused.append(pose.label or "")
            return MotionResult.failed(MotionStatus.TIMEOUT, MotionCommand.MOVE_TO, target_pose=pose,
                                       message="cuRobo found no collision-free plan; failing safe (no blind motion)")
        return super().move(pose)


# ---------------------------------------------------------------------------------------------------------------------
# The roll about the line of sight
# ---------------------------------------------------------------------------------------------------------------------


class EachViewTurnsTheCameraAboutItsOwnAxisTests(unittest.TestCase):

    def test_the_default_views_roll_by_minus_45_0_and_plus_45(self) -> None:
        rolls = [view[2] for view in DEFAULT_VIEWS]
        self.assertEqual(sorted(set(rolls)), [-45.0, 0.0, 45.0])
        self.assertEqual(dict(((e, a), r) for e, a, r in DEFAULT_VIEWS)[(0.0, 0.0)], 0.0,
                         "the camera's own view with the tool down keeps the heading")
        self.assertLessEqual(max(rolls.count(r) for r in set(rolls)) - min(rolls.count(r) for r in set(rolls)), 1)

    def test_a_rolled_station_is_the_unrolled_one_turned_about_the_line_of_sight(self) -> None:
        """Same camera point, same optical axis, the image turned by the view's roll; the tilt is the aim's alone."""
        estimate = _estimate(ALONG)
        for station in aimed_stations(_aim(), estimate):
            assert station.pose is not None
            elevation, azimuth, roll = station.view
            self.assertEqual(station.roll_deg, roll)
            camera = station.pose.to_matrix() @ estimate.matrix()
            kept, tilt = aim_camera(camera[:3, 3], MARKER, estimate, closing_axis="-y")
            unrolled = kept.to_matrix() @ estimate.matrix()
            np.testing.assert_allclose(camera[:3, 3], unrolled[:3, 3], atol=1e-6)
            np.testing.assert_allclose(camera[:3, 2], unrolled[:3, 2], atol=1e-9)
            turned = math.degrees(math.atan2(float(np.dot(np.cross(unrolled[:3, 0], camera[:3, 0]), camera[:3, 2])),
                                             float(np.dot(unrolled[:3, 0], camera[:3, 0]))))
            self.assertAlmostEqual(turned, roll, places=6, msg=station.label)
            self.assertAlmostEqual(station.tilt_deg, tilt, places=6)

    def test_the_relative_turns_span_every_axis_of_the_tool(self) -> None:
        """Before: the weakest axis of every pair's relative turn carried 1/70 of the strongest (0.014), 6 degrees from
        the optical axis (review 4, verify/f1_axis.py); rolled, 0.17. AX=XB reads the camera's rotation about an axis
        only from turns about it."""
        poses = [station.pose.to_matrix() for station in aimed_stations(_aim(), _estimate(ALONG))
                 if station.pose is not None]
        scatter = np.zeros((3, 3))
        for i, a in enumerate(poses):
            for b in poses[i + 1:]:
                turn = aim_module._log(a[:3, :3].T @ b[:3, :3])
                scatter += np.outer(turn, turn)
        weakest, _, strongest = np.linalg.eigvalsh(scatter)
        self.assertGreater(weakest / strongest, 0.1)

    def test_the_solve_places_a_point_half_a_metre_out_within_a_few_millimetres(self) -> None:
        """At K2's noise (0.5 deg and 1 mm a view, depth 0.4 %), 12 seeds: before, the heading kept, the point 500 mm
        down the optical axis landed 5.6 mm off at the median; rolled, 1.3."""
        stations = [station.pose for station in aimed_stations(_aim(), _truth(ALONG)) if station.pose is not None]
        marker = np.eye(4)
        marker[:3, :3] = aim_module._rz(math.radians(20.0))
        marker[:3, 3] = MARKER
        misses = []
        for seed in range(12):
            rng = np.random.default_rng(1000 + seed)
            arm = _Arm(LOOK)

            def seen() -> np.ndarray:
                pose = np.linalg.inv(arm.at.to_matrix() @ ALONG) @ marker
                axis = rng.normal(size=3)
                pose[:3, :3] = aim_module._exp(axis / np.linalg.norm(axis) * math.radians(rng.normal() * 0.5)) @ \
                    pose[:3, :3]
                pose[:3, 3] += np.array([rng.normal(), rng.normal(), rng.normal() * 0.004 * pose[2, 3]])
                return pose

            routine = CalibrationRoutine(
                arm=arm, marker_source=seen, workspace_limits=UR10_BOX,  # type: ignore[arg-type]
                calibration_mode="eye_in_hand", settle_time_s=0.0,
                eth_settings=SimpleNamespace(mode="eye_in_hand", min_samples=6, min_distance_mm=1.0,  # type: ignore
                                             min_angle=1.0, min_angle_deg=1.0))
            solved = routine.run_with_poses(stations).T_cam_to_tool
            assert solved is not None
            out = np.array([0.0, 0.0, 500.0, 1.0])
            misses.append(float(np.linalg.norm((solved @ out)[:3] - (ALONG @ out)[:3])))
        self.assertLess(float(np.median(misses)), 3.0, misses)

    def test_every_rolled_station_still_sees_the_marker_from_one_look(self) -> None:
        for station in aimed_stations(_aim(), _estimate(ALONG), start_tool_mm=LOOK.position_mm):
            assert station.pose is not None
            self.assertLess(_miss_deg(station.pose, ALONG), 1.0, station.label)
            self.assertTrue(_in_view(_seen(station.pose, ALONG)), station.label)

    def test_a_roll_the_box_refuses_gives_way_to_the_view_unrolled(self) -> None:
        """A roll never costs a view: where the rolled station is refused and the unrolled one is not, the view is
        visited unrolled."""
        estimate = _estimate(ALONG)
        view = next(view for view in DEFAULT_VIEWS if view[2] != 0.0)
        (rolled,) = aimed_stations(_aim(views=(view,)), estimate)
        (kept,) = aimed_stations(_aim(views=(view[:2],)), estimate)
        assert rolled.pose is not None and kept.pose is not None
        moved = np.asarray(rolled.pose.position_mm) - np.asarray(kept.pose.position_mm)
        axis = int(np.argmax(np.abs(moved)))
        self.assertGreater(abs(float(moved[axis])), 10.0, "the roll moves the TCP, which stands off the camera")
        cut = float(kept.pose.position_mm[axis]) + float(moved[axis]) / 2.0
        bounds = {f"{name}_{end}": (2000.0 if end == "max" else -2000.0) for name in "xyz" for end in ("min", "max")}
        bounds["xyz"[axis] + ("_max" if moved[axis] > 0 else "_min")] = cut
        box = WorkspaceLimitsConfig(**bounds)
        self.assertTrue(aimed_stations(_aim(views=(view,)), estimate, box=box, skip_labels=())[0].pose is not None)
        self.assertEqual(aim_module._outside_box(box, 0.0, rolled.pose.to_matrix(), None)[:7], "its TCP")
        (station,) = aimed_stations(_aim(views=(view,)), estimate, box=box)
        assert station.pose is not None
        self.assertEqual((station.view, station.roll_deg), (view, 0.0))
        np.testing.assert_allclose(station.pose.to_matrix(), kept.pose.to_matrix(), atol=1e-9)

    def test_a_rolled_station_the_arm_refuses_is_tried_once_more_unrolled(self) -> None:
        """On the real cuRobo sidecar (2026-09-23) the rolled views folded the forearm onto wrist 2 more often than the
        heading-kept ones, and the planner refused them with nothing moved. The view is then tried once more with the
        tool's heading; a view that was not rolled, or was refused unrolled too, is not tried again."""
        rolled = [aim_module._view_label(view) for view in DEFAULT_VIEWS if view[2] != 0.0][:2]
        flat = next(aim_module._view_label(view) for view in DEFAULT_VIEWS if view[2] == 0.0)
        arm = _Refusing(LOOK, refuse={*rolled, flat, f"{rolled[1]}_r0"})
        result = _routine(arm, ALONG).run_aimed(_aim())
        self.assertEqual(arm.refused.count(rolled[0]), 1)
        self.assertEqual([label for label in arm.refused if label.startswith(rolled[1])],
                         [rolled[1], f"{rolled[1]}_r0"], "refused unrolled too: not tried a third time")
        self.assertEqual(arm.refused.count(flat), 1)
        self.assertNotIn(f"{flat}_r0", arm.refused + [pose.label for pose in arm.moves], "a view with no roll")
        verdicts = {verdict.label: verdict for verdict in result.pose_log}
        self.assertEqual(verdicts[rolled[0]].reason, "move_rejected")
        self.assertTrue(verdicts[f"{rolled[0]}_r0"].counted)
        (retried,) = [pose for pose in arm.moves if pose.label == f"{rolled[0]}_r0"]
        camera = retried.to_matrix() @ ALONG
        kept, _ = aim_camera(camera[:3, 3], MARKER, _truth(ALONG), closing_axis="-y")
        np.testing.assert_allclose(retried.to_matrix(), kept.to_matrix(), atol=1e-3)
        self.assertEqual(result.num_samples, len(DEFAULT_VIEWS) - 2)

    def test_what_a_rolled_view_refuses(self) -> None:
        for views in (((0.0, 0.0, 95.0),), ((0.0, 0.0, float("nan")),), ((0.0, 0.0, 45.0), (0.0, 0.0, -45.0)),
                      ((0.0, 0.0, 0.0, 1.0),)):
            with self.subTest(views=views), self.assertRaises(ValueError):
                MarkerAim(marker_mm=(0.0, 0.0, 0.0), views=views)
        self.assertEqual(MarkerAim(marker_mm=(0.0, 0.0, 0.0), views=((0.0, 0.0), (10.0, 0.0, 45.0))).views,
                         ((0.0, 0.0), (10.0, 0.0, 45.0)))


# ---------------------------------------------------------------------------------------------------------------------
# The heading
# ---------------------------------------------------------------------------------------------------------------------


class AHeadingIsChosenAmongFourTests(unittest.TestCase):

    def test_the_one_asked_is_kept_while_it_admits_enough(self) -> None:
        asked: list[str] = []

        def admitted(heading: str) -> int:
            asked.append(heading)
            return {"-y": 6, "y": 17}.get(heading, 0)

        choice = choose_heading("-y", admitted, need=6, views=17, what="the solve")
        self.assertEqual((choice.chosen, choice.changed, asked), ("-y", False, ["-y"]))
        self.assertIn("'-y' admits 6 of the 17 views", choice.line())

    def test_too_few_gives_way_to_the_most_the_one_asked_winning_a_tie_then_the_order(self) -> None:
        counts = {"-y": 2, "y": 10, "x": 10, "-x": 3}
        choice = choose_heading("-y", counts.__getitem__, need=6, views=17, what="the solve")
        self.assertEqual((choice.chosen, choice.changed), ("y", True), "y before x, in HEADINGS order")
        self.assertEqual(dict(choice.admitted), counts)
        line = choice.line()
        for part in ("'y', not the '-y' asked", "'-y' admits 2 of the 17 views, fewer than the 6 the solve needs",
                     "-y 2, y 10, x 10, -x 3"):
            self.assertIn(part, line)
        tie = choose_heading("x", {"-y": 0, "y": 5, "x": 5, "-x": 0}.__getitem__, need=6, views=17, what="the solve")
        self.assertEqual((tie.chosen, tie.changed), ("x", False))
        self.assertIn("no other heading admits more", tie.line())
        self.assertEqual(HEADINGS, ("-y", "y", "x", "-x"))

    def test_a_leading_plus_or_a_heading_outside_the_four_is_the_one_asked(self) -> None:
        tried: list[str] = []

        def admitted(heading: str) -> int:
            tried.append(heading)
            return {"+x": 1, "-y": 3, "y": 3, "-x": 2}[heading]

        plus = choose_heading("+x", admitted, need=6, views=17, what="the solve")
        self.assertEqual(tried, ["+x", "-y", "y", "-x"], "x is the one asked, and is not tried twice")
        self.assertEqual((plus.chosen, plus.changed), ("-y", True))
        counts = {"tangential": 0, "-y": 1, "y": 9, "x": 0, "-x": 0}
        other = choose_heading("tangential", counts.__getitem__, need=6, views=17, what="the solve")
        self.assertEqual((other.chosen, list(dict(other.admitted))), ("y", ["tangential", "-y", "y", "x", "-x"]))
        self.assertEqual(other.to_dict()["chosen"], "y")


class TheSweepFindsItsHeadingTests(unittest.TestCase):

    def test_the_premise_the_camera_tilted_the_other_way_sees_the_marker_from_its_look(self) -> None:
        self.assertTrue(_in_view(_seen(LOOK_AGAINST, AGAINST)))
        refused = aimed_stations(_aim(), _truth(AGAINST), box=UR10_BOX, margin_mm=20.0)
        self.assertTrue(all(station.reason == "outside_workspace" for station in refused))

    def test_a_heading_that_admits_no_view_gives_way_to_the_one_that_admits_most(self) -> None:
        """Before: 0 of 17 moved, and the sweep ended 'too few samples: got 0, need >= 4' naming no heading."""
        arm = _Arm(LOOK_AGAINST)
        routine = _routine(arm, AGAINST, box=UR10_BOX)
        with self.assertLogs("CalibrationRoutine", level="WARNING") as said:
            result = routine.run_aimed(_aim(), margin_mm=20.0)
        choice = result.aim_heading
        assert choice is not None
        self.assertEqual((choice.asked, choice.chosen, choice.changed), ("-y", "y", True))
        self.assertEqual(dict(choice.admitted)["-y"], 0)
        self.assertEqual(dict(choice.admitted)["y"], len(DEFAULT_VIEWS))
        self.assertIn("not the '-y' asked", "\n".join(said.output))
        self.assertEqual(len(arm.moves), len(DEFAULT_VIEWS))
        assert result.T_cam_to_tool is not None
        self.assertLess(float(np.linalg.norm(result.T_cam_to_tool[:3, 3] - AGAINST[:3, 3])), 0.5)
        self.assertIs(routine.aimed.heading, choice)  # type: ignore[union-attr]

    def test_a_heading_that_admits_enough_is_kept(self) -> None:
        arm = _Arm(LOOK)
        result = _routine(arm, ALONG, box=UR10_BOX).run_aimed(_aim(), margin_mm=20.0)
        choice = result.aim_heading
        assert choice is not None
        self.assertEqual((choice.chosen, choice.changed), ("-y", False))
        self.assertEqual([heading for heading, _ in choice.admitted], ["-y"], "nothing else was tried")

    def test_no_heading_admitting_enough_ends_in_the_solves_refusal_and_names_every_heading(self) -> None:
        tiny = WorkspaceLimitsConfig(x_min=-50.0, x_max=50.0, y_min=-50.0, y_max=50.0, z_min=100.0, z_max=200.0)
        arm = _Arm(LOOK)
        routine = _routine(arm, ALONG, box=tiny)
        with self.assertRaisesRegex(CalibrationDataError, "too few samples"):
            routine.run_aimed(_aim(), margin_mm=20.0)
        self.assertEqual(arm.moves, [])
        choice = routine.aimed.heading  # type: ignore[union-attr]
        self.assertEqual((choice.chosen, choice.changed), ("-y", False))
        self.assertIn("-y 0, y 0, x 0, -x 0", choice.line())


class TheReportSaysWhichHeadingTests(unittest.TestCase):

    def _run(self, **patches: Any) -> Any:
        from tests.test_hand_eye_calibration import _Doubles, _no_geometry

        doubles = _Doubles()
        doubles.setUp()
        self.addCleanup(doubles.doCleanups)
        out = str(self.enterContext(tempfile.TemporaryDirectory()))
        noun = doubles._noun(mode="eye_in_hand", rig_id="wrist", robot_config=_no_geometry(),  # noqa: SLF001
                             options=SweepOptions(out_dir=out, aim=_aim(), target="aruco:50:150:DICT_4X4_100"))
        with patch.object(CalibrationRoutine, "run_aimed", autospec=True, **patches):
            return noun.run()

    _CHOICE = HeadingChoice(asked="-y", chosen="y", admitted=(("-y", 0), ("y", 17), ("x", 10), ("-x", 10)),
                            views=17, need=6, what="the solve")

    def test_a_written_sweep_says_which_heading_it_took(self) -> None:
        report = self._run(return_value=_eih_result(aim_heading=self._CHOICE))
        self.assertIs(report.outcome, CalibrationOutcome.WRITTEN, report.render())
        self.assertIn("  heading           'y', not the '-y' asked", report.render())
        self.assertEqual(report.to_dict()["aim_heading"]["chosen"], "y")

    def test_a_failed_sweep_says_which_headings_it_tried(self) -> None:
        choice = HeadingChoice(asked="-y", chosen="-y", admitted=(("-y", 0), ("y", 0), ("x", 0), ("-x", 0)),
                               views=17, need=6, what="the solve")

        def refused(routine: Any, *_: Any, **__: Any) -> Any:
            routine.aimed = SimpleNamespace(estimates=(), heading=choice)
            raise CalibrationDataError("too few samples: got 0, need >= 6")

        report = self._run(side_effect=refused)
        self.assertIs(report.outcome, CalibrationOutcome.SWEEP_FAILED)
        self.assertIn("  heading           '-y' admits 0 of the 17 views", report.render())
        self.assertIn("-y 0, y 0, x 0, -x 0", report.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
