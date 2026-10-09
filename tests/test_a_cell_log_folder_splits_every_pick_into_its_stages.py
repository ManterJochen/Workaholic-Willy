"""A cell's logs folder splits every task into its survey, picks and places, and each into its stages, as its lines say.

``scripts/cell/pick_timeline.py`` reads the folders under ``tests/data/cell_logs``:

* ``2026_10_08``: the owner's cell on the morning of the presentation, copied whole with its private paths replaced:
  two waves and two tasks that stopped ``part_still_held``, the second a restart of the first.
* ``2026_10_07``: the demo day's run-3b3c8e8082 from its connect to the start of its second pick: a pick, a place into
  the yellow bin and the way home, on the code before mesh first. The copy ends inside the second pick.
* ``new_code_2026_10_12``: written to the formats of today's log calls, not copied from a cell: an until-empty task that
  picks one cube, places it and ends on its check look, its robot log rotated once.
* ``ursim_2026_10_06``: three tasks against URSim that place at a taught pose, with no models' logs.

Every number asserted below is read off those lines by hand, to the millisecond.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import shutil
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "tests" / "data" / "cell_logs"
_MORNING = _DATA / "2026_10_08"
_DEMO_DAY = _DATA / "2026_10_07"
_TODAY = _DATA / "new_code_2026_10_12"
_URSIM = _DATA / "ursim_2026_10_06"


def _timeline() -> Any:
    spec = importlib.util.spec_from_file_location("_pick_timeline_under_test", _REPO / "scripts/cell/pick_timeline.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


T = _timeline()


def _picks(folder: Any) -> list[Any]:
    return [unit for run in folder.runs for unit in run.units if unit.kind == "pick"]


def _places(folder: Any) -> list[Any]:
    return [unit for run in folder.runs for unit in run.units if unit.kind == "place"]


def _run(argv: list[str]) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        try:
            code = T.main(argv)
        except SystemExit as exc:
            code = exc.code if isinstance(exc.code, int) else 2
            out.write(str(exc.code))
    return int(code), out.getvalue()


class _Copies(unittest.TestCase):
    """A folder copied into a scratch directory, to take files away from."""

    def copy(self, source: Path, *, without: tuple[str, ...] = ()) -> Path:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        target = root / source.name
        shutil.copytree(source, target)
        for name in without:
            path = target / name
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        return target


class TheMorningOfThePresentationIsSplitAsItsLinesSayTests(unittest.TestCase):
    folder = T.load(str(_MORNING))

    def test_its_waves_and_its_two_tasks_are_found_the_second_a_restart_of_the_first(self) -> None:
        kinds = [(run.kind, run.code) for run in self.folder.runs]
        self.assertEqual(kinds, [("planner", "planner_ready"), ("wave", "finished"), ("wave", "finished"),
                                 ("task", "part_still_held"), ("task", "part_still_held"), ("home", "finished")])
        first, second = (run for run in self.folder.runs if run.kind == "task")
        self.assertEqual((first.object, first.target, first.scope),
                         ("grauer Würfel auf der schwarzen Matte.", "yellow bin", "until_empty"))
        self.assertEqual(second.restarts, first.id)
        self.assertEqual(second.object, first.object, "a restart's sentence is its first run's")

    def test_the_first_pick_splits_into_the_stages_its_lines_mark(self) -> None:
        pick = _picks(self.folder)[0]
        self.assertEqual(pick.outcome, "succeeded")
        self.assertEqual(pick.start, datetime(2026, 10, 8, 7, 37, 29, 249000), "it starts where the pick's first line is")
        expected = {"look": 1.4404, "grounding": 13.1656, "grasp_search": 30.174, "judging": 13.064, "approach": 7.265,
                    "standoff": 3.301, "line_down": 2.2, "at_part": 0.185, "close": 4.032, "lift": 3.367}
        self.assertEqual(list(pick.stages), list(expected))
        for stage, seconds in expected.items():
            with self.subTest(stage=stage):
                self.assertAlmostEqual(pick.stages[stage], seconds, places=3)
        self.assertAlmostEqual(pick.total, 78.194, places=3)
        self.assertAlmostEqual(sum(pick.stages.values()), pick.total, places=6, msg="the stages fill the pick")

    def test_what_said_its_own_length_is_summed_inside_the_pick(self) -> None:
        pick = _picks(self.folder)[0]
        self.assertAlmostEqual(pick.inside["qwen"], 12.853, places=3)
        self.assertAlmostEqual(pick.inside["sam2"], 0.103 + 0.098 + 0.096, places=3)
        self.assertEqual((pick.counts["tries"], pick.counts["refused"], pick.counts["parts"]), (2, 1, 3))

    def test_a_refused_try_is_named_by_where_and_what_refused_it(self) -> None:
        pick = _picks(self.folder)[0]
        self.assertEqual(dict(pick.reasons), {"line down: guard: lfinger vs a box the camera saw": 1})

    def test_the_sentence_and_the_click_before_the_first_task_are_timed(self) -> None:
        first = next(run for run in self.folder.runs if run.kind == "task")
        values = T.stage_values(self.folder.runs)
        self.assertAlmostEqual(first.command.ms, 9838.0)
        self.assertEqual(values["command"], [9.838])
        self.assertAlmostEqual(values["to_start"][0], 25.177, places=3, msg="07:36:44.921 to 07:37:10.098")
        self.assertAlmostEqual(values["survey"][0], 19.151, places=3, msg="07:37:10.098 to the pick's 07:37:29.249")

    def test_the_connect_carries_its_build_its_planner_and_qwens_load(self) -> None:
        (session,) = self.folder.sessions
        self.assertEqual(session.fingerprint, "0ba1a0b2e7c3f824")
        self.assertEqual((session.facts["cell_build_s"], session.facts["planner_start_s"],
                          session.facts["qwen_load_s"]), (196.7, 75.7, 144.1))

    def test_a_task_that_stopped_with_the_part_in_the_jaws_times_no_place(self) -> None:
        values = T.stage_values(self.folder.runs)
        self.assertNotIn("place", values)
        self.assertNotIn("cycle", values)
        self.assertEqual(T.quality(self.folder.runs)["parts placed"], 0)


class ADemoDayPickAndPlaceIsSplitIntoItsStagesTests(unittest.TestCase):
    folder = T.load(str(_DEMO_DAY))

    def test_the_place_splits_into_its_carry_its_bin_check_its_drop_and_its_way_home(self) -> None:
        (place,) = _places(self.folder)
        expected = {"place_judging": 6.286, "carry": 6.226, "bin_check": 3.189, "drop_judging": 11.069,
                    "to_drop": 6.109, "drop_standoff": 3.42, "line_in": 1.048, "release": 5.0, "line_out": 1.6349,
                    "return_judging": 11.6351, "return": 6.052}
        self.assertEqual(list(place.stages), list(expected))
        for stage, seconds in expected.items():
            with self.subTest(stage=stage):
                self.assertAlmostEqual(place.stages[stage], seconds, places=3)
        self.assertAlmostEqual(place.total, 61.669, places=3)

    def test_the_pick_and_its_place_make_one_cycle(self) -> None:
        values = T.stage_values(self.folder.runs)
        pick = _picks(self.folder)[0]
        self.assertAlmostEqual(values["cycle"][0], pick.total + 61.669, places=3)
        self.assertAlmostEqual(pick.stages["judging"], 94.487, places=3, msg="three tries refused by the planner")
        self.assertEqual(dict(pick.reasons), {"lift: planner": 3})

    def test_a_pick_the_copy_cut_short_is_counted_and_not_timed(self) -> None:
        cut = _picks(self.folder)[-1]
        self.assertEqual(cut.outcome, "")
        self.assertEqual(T.quality(self.folder.runs)["picks cut short (no end logged)"], 1)
        self.assertEqual(len(T.stage_values(self.folder.runs)["pick"]), 1)


class TodaysLinesAreReadTests(unittest.TestCase):
    folder = T.load(str(_TODAY))

    def test_a_rotated_robot_log_is_read_oldest_first(self) -> None:
        self.assertEqual(T.log_files(_TODAY)["robot/robot.log"],
                         [_TODAY / "robot" / "robot.log.1", _TODAY / "robot" / "robot.log"])
        pick = _picks(self.folder)[0]
        expected = {"look": 0.6, "grounding": 6.5, "grasp_search": 0.9, "judging": 2.31, "approach": 7.001,
                    "standoff": 1.5, "line_down": 1.1, "at_part": 0.1, "close": 2.089, "lift": 1.25}
        for stage, seconds in expected.items():
            with self.subTest(stage=stage):
                self.assertAlmostEqual(pick.stages[stage], seconds, places=3)

    def test_the_lines_of_todays_code_are_counted_in_the_pick_that_wrote_them(self) -> None:
        pick, empty = _picks(self.folder)
        self.assertEqual({name: pick.counts[name] for name in ("looks_skipped", "route_reused", "lift_reused", "held",
                                                               "mesh_first", "refreshes", "records")},
                         {"looks_skipped": 1, "route_reused": 1, "lift_reused": 1, "held": 2, "mesh_first": 3,
                          "refreshes": 1, "records": 0})
        (place,) = _places(self.folder)
        self.assertEqual(place.counts["records"], 1, "a record written in the background lands where it was written")

    def test_a_recheck_by_depth_ends_the_carry_and_no_bin_check_is_timed(self) -> None:
        (place,) = _places(self.folder)
        self.assertNotIn("bin_check", place.stages)
        self.assertAlmostEqual(place.stages["carry"], 7.09, places=3)
        self.assertAlmostEqual(place.total, 29.05, places=3)

    def test_the_check_look_after_the_last_part_is_an_empty_pick(self) -> None:
        values = T.stage_values(self.folder.runs)
        self.assertEqual(len(values["pick"]), 1, "the pick that tried a grasp")
        self.assertAlmostEqual(values["empty pick"][0], 6.8, places=3)
        self.assertAlmostEqual(values["cycle"][0], 23.35 + 29.05, places=3)
        self.assertEqual(T.quality(self.folder.runs)["pick outcomes"], {"succeeded": 1, "no_perception": 1})

    def test_a_known_sentence_is_read_in_its_milliseconds(self) -> None:
        (task,) = (run for run in self.folder.runs if run.kind == "task")
        self.assertEqual((task.command.ms, task.command.how), (3.0, "known sentence"))
        self.assertEqual(task.phrase, "each separate gray cube")

    def test_a_traceback_is_one_line_with_the_line_it_continues(self) -> None:
        records = T.read_folder(_TODAY)
        (error,) = (record for record in records if record.level == "ERROR")
        self.assertTrue(error.text.startswith("a line nobody reads"))
        self.assertIn("ValueError: only to be skipped", error.text)

    def test_the_connect_says_its_grasp_workers_qwens_lookups_and_the_switches_that_log_their_choice(self) -> None:
        (session,) = self.folder.sessions
        self.assertEqual(session.facts["workers"], "7 ready (pids 4101, 4102, 4103, 4104, 4105, 4106, 4107)")
        self.assertEqual(session.facts["qwen"],
                         "prompt_lookup_tokens=10, text_prompt_lookup_tokens=10, stop_at_answer_end=True")
        (said,) = session.facts["said"]
        self.assertTrue(said.startswith("The steady gate reads the joint speeds"))

    def test_a_line_solved_on_the_controllers_own_kinematics_is_counted_where_the_arm_stands(self) -> None:
        pick = _picks(self.folder)[0]
        self.assertEqual(pick.counts["local_ik"], 1)
        self.assertAlmostEqual(pick.stages["standoff"], 1.5, places=3, msg="said at the standoff, it moves no stage")


class AFollowedLookIsTimedAsItsGroundingTests(unittest.TestCase):
    """A pick whose parts were followed (robot.grasping.follow_parts) says so with its own length, and that is its
    grounding: no detector was asked."""

    def test_the_followed_frame_is_the_grounding_and_is_counted(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        day = "2026-10-12"
        (root / "api").mkdir()
        (root / "robot").mkdir()
        (root / "api" / "runs.log").write_text(
            f"{day} 11:00:00,000 - RunRegistry - INFO - Run run-f1 starting: task.\n"
            f"{day} 11:00:30,000 - RunRegistry - INFO - Run run-f1 (task) finished: part_still_held in 30.0 s.\n",
            encoding="utf-8")
        (root / "api" / "task_run.log").write_text(
            f"{day} 11:00:29,000 - TaskRun - INFO - Task run run-f1, part 1, pick 1: succeeded.\n"
            f"{day} 11:00:29,900 - TaskRun - INFO - Task run run-f1 ended part_still_held: held.\n", encoding="utf-8")
        loop = "src.robot.grasping.loop.pick_loop"
        (root / "robot" / "robot.log").write_text(
            f"{day} 11:00:01,000 - src.robot.execution.autonomous_grasp.service - INFO - recovery rescans nothing for "
            f"a wrist pick handed looks: its looks were its rescans\n"
            f"{day} 11:00:01,500 - src.robot.perception.realsense_source - INFO - 2 part(s) followed, nothing new in "
            f"depth in 300 ms: the detector was not asked\n"
            f"{day} 11:00:02,000 - {loop} - INFO - the looks stop at look home, whose grasp is safe enough\n"
            f"{day} 11:00:02,100 - {loop} - INFO - try 1 of 4: the grasp the look ranked first (rank 0 of the look's "
            f"grasps)\n", encoding="utf-8")
        (pick,) = _picks(T.load(str(root)))
        self.assertAlmostEqual(pick.stages["look"], 0.2, places=3)
        self.assertAlmostEqual(pick.stages["grounding"], 0.3, places=3)
        self.assertEqual(pick.counts["followed"], 1)
        self.assertNotIn("qwen", pick.inside)


class TheTasksOwnWayHomeAtItsEndOpensNoPickTests(unittest.TestCase):
    """A task that ends on a pick that found nothing drives home after it: that judgement and that move are the task's
    end, not another pick's look."""

    def test_the_way_home_after_the_last_pick_is_no_pick(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        day = "2026-10-12"
        (root / "api").mkdir()
        (root / "robot").mkdir()
        (root / "api" / "runs.log").write_text(
            f"{day} 13:00:00,000 - RunRegistry - INFO - Run run-h1 starting: task.\n"
            f"{day} 13:00:30,000 - RunRegistry - INFO - Run run-h1 (task) finished: nothing_left in 30.0 s.\n",
            encoding="utf-8")
        (root / "api" / "task_run.log").write_text(
            f"{day} 13:00:10,000 - TaskRun - INFO - Task run run-h1, part 1, pick 1: no_perception.\n"
            f"{day} 13:00:29,990 - TaskRun - INFO - Task run run-h1 ended nothing_left: nothing to pick is left.\n",
            encoding="utf-8")
        (root / "robot" / "robot.log").write_text(
            f"{day} 13:00:01,000 - src.robot.execution.autonomous_grasp.service - INFO - recovery rescans nothing for a "
            f"wrist pick handed looks: its looks were its rescans\n"
            f"{day} 13:00:09,000 - src.robot.grasping.loop.pick_loop - INFO - no look saw anything to pick (looked from "
            f"look 2)\n"
            f"{day} 13:00:12,000 - CuroboUrPlanner - INFO - planner world refreshed: 9 box(es), frame 5 ms old, 400.0 ms "
            f"to build, 100.0 ms to register\n"
            f"{day} 13:00:13,000 - URRobotArm - INFO - move_home: direct line; 2 waypoint(s) dense, 2 executed (1 "
            f"leg(s)); joint travel in deg, total/largest leg: x\n", encoding="utf-8")
        folder = T.load(str(root))
        (run,) = folder.runs
        self.assertEqual([unit.kind for unit in run.units], ["survey", "pick", "between"])
        self.assertAlmostEqual(run.units[-1].total, 19.99, places=3, msg="the way home, to the task's end line")
        self.assertEqual(T.quality(folder.runs)["picks cut short (no end logged)"], 0)


class AJudgementOnASecondThreadEndsNoLineTests(unittest.TestCase):
    """robot.motion.judge_next_leg in_settles_and_motion judges the way home on a second thread while the line out
    runs: what that thread writes then is no sign the line ended; the main thread's "runs as it was judged" is."""

    def test_the_line_out_runs_until_the_way_home_runs_as_judged(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        day = "2026-10-12"
        arm = "URRobotArm"
        (root / "api").mkdir()
        (root / "robot").mkdir()
        (root / "api" / "runs.log").write_text(
            f"{day} 12:00:00,000 - RunRegistry - INFO - Run run-s1 starting: task.\n"
            f"{day} 12:00:40,000 - RunRegistry - INFO - Run run-s1 (task) finished: finished in 40.0 s.\n",
            encoding="utf-8")
        (root / "api" / "task_run.log").write_text(
            f"{day} 12:00:10,000 - TaskRun - INFO - Task run run-s1, part 1, pick 1: succeeded.\n"
            f"{day} 12:00:39,900 - TaskRun - INFO - Task run run-s1 ended finished: placed.\n", encoding="utf-8")
        lines = [
            ("12:00:01,000", "src.robot.grasping.loop.pick_loop", "try 1 of 2: the grasp the look ranked first"),
            ("12:00:12,000", arm, "move to standoff: direct line to goal 1 of 4 (branch shoulder -); 2 waypoint(s) "
                                  "dense, 2 executed (1 leg(s)); joint travel in deg, total/largest leg: x"),
            ("12:00:18,000", "CuroboUrPlanner", "judged path executed on UR: 2 moveJ"),
            ("12:00:19,000", "MotionController", "Moving to 'drop over yellow bin' [1.0, 2.0, 3.0 mm] (linear) ..."),
            ("12:00:20,000", "JawIOGripper", "actuating jaws OPEN via single_toggle"),
            ("12:00:22,000", arm, "the joint move declared next is judged on a second thread while the line to "
                                  "standoff runs"),
            ("12:00:22,010", "MotionController", "Moving to 'standoff' [1.0, 2.0, 83.0 mm] (linear) ..."),
            ("12:00:22,300", arm, "the world is held (the drop): judged against the 9 box(es) the last refresh "
                                  "registered"),
            ("12:00:22,400", arm, "mesh first: the exact guard alone judged 880 sample(s) of this straight joint line "
                                  "clear; cuRobo was not asked"),
            ("12:00:23,110", arm, "move_home: the joint move judged ahead runs as it was judged, 900 ms ago, in the "
                                  "world held (the drop); it is not judged a second time"),
            ("12:00:23,120", arm, "move_home: direct line; 2 waypoint(s) dense, 2 executed (1 leg(s)); joint travel "
                                  "in deg, total/largest leg: x"),
        ]
        (root / "robot" / "robot.log").write_text(
            "".join(f"{day} {at} - {logger} - INFO - {text}\n" for at, logger, text in lines), encoding="utf-8")
        (place,) = _places(T.load(str(root)))
        self.assertAlmostEqual(place.stages["line_out"], 1.1, places=3, msg="not ended by the second thread's lines")
        self.assertAlmostEqual(place.stages["return_judging"], 0.01, places=3)
        self.assertAlmostEqual(place.stages["return"], 39.9 - 23.12, places=3)
        self.assertEqual((place.counts["leg_reused"], place.counts["held"]), (1, 1))


class APlaceAtATaughtPoseIsSplitTests(unittest.TestCase):
    folder = T.load(str(_URSIM))

    def test_a_place_at_a_taught_pose_goes_straight_to_the_drop(self) -> None:
        places = _places(self.folder)
        self.assertEqual(len(places), 3)
        first = places[0]
        self.assertEqual(list(first.stages), ["place_judging", "to_drop", "drop_standoff", "line_in", "release",
                                              "line_out", "return"])
        self.assertAlmostEqual(first.stages["to_drop"], 5.403, places=3)
        self.assertAlmostEqual(first.stages["return"], 5.274, places=3, msg="the return ends where the task did")

    def test_a_folder_with_no_models_logs_says_so(self) -> None:
        out: list[str] = []
        T.describe(self.folder, out)
        self.assertTrue(any("no models/*.log" in line for line in out))
        self.assertEqual(T.quality(self.folder.runs)["parts placed"], 3)


class AFolderWithFilesMissingStillSplitsTests(_Copies):
    def test_a_line_the_aggregate_and_its_module_file_both_hold_is_read_once(self) -> None:
        alone = self.copy(_MORNING, without=("robot/modules",))
        self.assertEqual(len(T.read_folder(_MORNING)), len(T.read_folder(alone)))

    def test_the_module_files_alone_give_the_lines_the_aggregate_would(self) -> None:
        modules = self.copy(_MORNING, without=("robot/robot.log",))
        texts = {record.text for record in T.read_folder(modules) if record.file.startswith("robot/")}
        aggregate = {record.text for record in T.read_folder(_MORNING) if record.file == "robot/robot.log"}
        self.assertTrue(texts)
        self.assertLessEqual(texts, aggregate)

    def test_without_the_models_logs_a_grounding_is_timed_inside_its_look(self) -> None:
        folder = T.load(str(self.copy(_MORNING, without=("models",))))
        pick = _picks(folder)[0]
        self.assertNotIn("grounding", pick.stages)
        self.assertAlmostEqual(pick.stages["look"], 14.842, places=3, msg="to the calculator's first line, 07:37:44.091")
        self.assertAlmostEqual(pick.total, 78.194, places=3)

    def test_without_the_console_logs_the_picks_end_at_their_grasp_records(self) -> None:
        folder = T.load(str(self.copy(_MORNING, without=("api",))))
        (run,) = folder.runs
        self.assertEqual(run.kind, "logs")
        picks = _picks(folder)
        self.assertEqual([pick.outcome for pick in picks], ["succeeded", "succeeded"])
        self.assertEqual(picks[0].start, datetime(2026, 10, 8, 7, 37, 29, 249000))
        self.assertEqual(picks[0].end, datetime(2026, 10, 8, 7, 38, 46, 682000), "its record's line")

    def test_a_folder_with_no_log_says_so_and_exits_two(self) -> None:
        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        code, said = _run([str(empty)])
        self.assertEqual(code, 2)
        self.assertIn("holds no log file", said)

    def test_a_pick_runs_picks_are_split_without_a_survey(self) -> None:
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        day = "2026-10-12"
        (root / "api").mkdir()
        (root / "robot").mkdir()
        (root / "api" / "runs.log").write_text(
            f"{day} 10:00:00,000 - RunRegistry - INFO - Run run-p1 starting: 1 pick(s), prompt 'gray cube'.\n"
            f"{day} 10:00:20,000 - RunRegistry - INFO - Run run-p1 pick 1/1: succeeded (gripped).\n"
            f"{day} 10:00:20,100 - RunRegistry - INFO - Run run-p1 finished (finished): 1/1 succeeded in 20.1 s.\n",
            encoding="utf-8")
        (root / "robot" / "robot.log").write_text(
            f"{day} 10:00:01,000 - CuroboUrPlanner - INFO - planner world refreshed: 9 box(es), frame 5 ms old, "
            f"400.0 ms to build, 100.0 ms to register\n"
            f"{day} 10:00:02,000 - src.robot.grasping.loop.pick_loop - INFO - try 1 of 3: the grasp the look ranked "
            f"first (rank 0 of the look's grasps)\n", encoding="utf-8")
        folder = T.load(str(root))
        (run,) = folder.runs
        self.assertEqual((run.kind, run.phrase, run.code), ("pick", "gray cube", "finished"))
        (pick,) = _picks(folder)
        self.assertEqual(pick.start, datetime(2026, 10, 12, 10, 0, 0, 500000), "its first work, dated back")
        self.assertEqual(pick.outcome, "succeeded")


class TwoFoldersCompareStageByStageTests(unittest.TestCase):
    def test_a_folder_against_itself_differs_by_nothing(self) -> None:
        code, said = _run([str(_MORNING), str(_MORNING)])
        self.assertEqual(code, 0)
        rows = [line for line in said.splitlines()
                if line.split()[:1] in (["approach"], ["judging"], ["pick"]) and line.split()[1].isdigit()]
        self.assertTrue(rows)
        for row in rows:
            with self.subTest(row=row):
                self.assertIn("+0.0", row)

    def test_the_demo_day_against_the_morning_names_each_stage_s_difference(self) -> None:
        code, said = _run([str(_DEMO_DAY), str(_MORNING)])
        self.assertEqual(code, 0)
        (approach,) = (line for line in said.splitlines() if line.strip().startswith("approach"))
        self.assertIn("6.2", approach)
        self.assertIn("7.0", approach)
        self.assertIn("+0.8", approach)
        self.assertIn("quality", said)
        self.assertIn("parts placed", said)


class ASelectionKeepsTheRunsItNamesTests(unittest.TestCase):
    def test_hash_last_keeps_the_runs_of_the_last_connect_that_ran_any(self) -> None:
        folder = T.load(f"{_MORNING}#last")
        self.assertEqual(len(folder.runs), 6)
        self.assertEqual({run.session for run in folder.runs}, {1})

    def test_a_connect_by_its_number_keeps_its_runs_alone(self) -> None:
        self.assertEqual(T.load(f"{_MORNING}#2").runs, [])

    def test_a_window_keeps_the_runs_that_start_in_it(self) -> None:
        folder = T.load(f"{_MORNING}@07:37-07:40")
        self.assertEqual([run.id for run in folder.runs], ["run-1f9a82860d", "run-ca72c832aa"])
        self.assertEqual([run.id for run in T.load(f"{_MORNING}@07:39").runs], ["run-ca72c832aa", "run-77c6528fca"])

    def test_a_selector_it_cannot_read_is_refused_with_a_sentence(self) -> None:
        with self.assertRaises(SystemExit) as said:
            T.load(f"{_MORNING}#second")
        self.assertIn("#last", str(said.exception))
        with self.assertRaises(SystemExit) as said:
            T.load(f"{_MORNING}@half past nine")
        self.assertIn("@HH:MM-HH:MM", str(said.exception))


class TheCommandLineSaysWhatItReadTests(unittest.TestCase):
    def test_it_prints_every_stage_it_timed_and_what_the_speed_was_bought_with(self) -> None:
        code, said = _run([str(_DEMO_DAY), "--picks"])
        self.assertEqual(code, 0)
        for word in ("connect #1", "run-3b3c8e8082", "grounding", "return_judging", "cycle", "parts placed",
                     "place: 61.7 s"):
            with self.subTest(word=word):
                self.assertIn(word, said)

    def test_json_holds_every_run_and_every_unit(self) -> None:
        target = Path(self.enterContext(tempfile.TemporaryDirectory())) / "out.json"
        code, said = _run([str(_TODAY), "--json", str(target)])
        self.assertEqual(code, 0)
        written = json.loads(target.read_text(encoding="utf-8"))
        (task,) = (run for run in written["runs"] if run["kind"] == "task")
        self.assertEqual([unit["kind"] for unit in task["units"]], ["survey", "pick", "place", "pick", "between"])
        self.assertAlmostEqual(task["units"][1]["stages_s"]["approach"], 7.001, places=3)
        self.assertEqual(written["quality"]["parts placed"], 1)

    def test_stages_says_what_every_stage_holds(self) -> None:
        code, said = _run(["--stages"])
        self.assertEqual(code, 0)
        for name, _, _ in T.STAGES:
            with self.subTest(stage=name):
                self.assertIn(name, said)

    def test_the_readme_names_every_stage_the_script_times(self) -> None:
        readme = (_REPO / "scripts" / "cell" / "README.md").read_text(encoding="utf-8")
        for name, _, _ in T.STAGES:
            with self.subTest(stage=name):
                self.assertIn(f"`{name}`", readme)


if __name__ == "__main__":
    unittest.main()
