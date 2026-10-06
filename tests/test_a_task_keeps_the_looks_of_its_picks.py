"""A task hands its picks the operator's switches, and puts back everything it set on the cell (OD 12, Q11).

The Advanced drawer's switches are the task's, never the cell's: multi-view (on: the looks as configured; off: the first
look only, no generated view), both jaw faces, the closing axis (an opt-in filter the pick loop reads off the policy),
the push distance (the campaign's), recordings of each pick's looks, and the grasp overlay (on for the task, a
measuring aid). Each is set before the first pick and put back as found when the task ends, whatever ended it, as the
prompt is: ``set_prompt`` and ``set_closing_axis`` return what they replaced, and the overlay switch reads back.

A pick's looks are kept for training where the task asks (``record_views``), through the same function a campaign uses
(``pick_run.keep_pick_views``, lifted out of ``PickRun`` with no change of behaviour): one file per pick, named after the
pick's record, and a file that cannot be written is said and the task goes on. Where it went rides on the report the
console is handed (``telemetry['views_file']``).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from src.geometry.closing_axis import closing_axis_of
from tests._task_fakes import Pick, RecordingHooks, run


class KeepPickViewsTests(unittest.TestCase):
    def _report(self, looks: tuple[str, ...] = ("home",), attempt_id: str = "pick-0123") -> Any:
        return SimpleNamespace(looks=looks, telemetry={"attempt_id": attempt_id})

    def _service(self, views: tuple[Any, ...] = ("view",)) -> Any:
        judged = SimpleNamespace(target_cloud_base_mm="cloud")
        return SimpleNamespace(looked_around=SimpleNamespace(views=views, judged=judged))

    def test_the_looks_of_a_pick_are_kept_under_its_record_name(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        with mock.patch("src.robot.execution.record_views.record_views", return_value="D:/views/pick-0123.npz") as kept:
            where = keep_pick_views(self._service(), self._report(), name="fallback")

        self.assertEqual("D:/views/pick-0123.npz", where)
        kept.assert_called_once_with(("view",), target_cloud_base_mm="cloud", name="pick-0123")

    def test_a_pick_with_no_record_name_takes_the_one_given(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        with mock.patch("src.robot.execution.record_views.record_views", return_value="x") as kept:
            keep_pick_views(self._service(), self._report(attempt_id=""), name="task-part2-pick3")

        self.assertEqual("task-part2-pick3", kept.call_args.kwargs["name"])

    def test_a_pick_that_did_not_look_or_kept_no_views_writes_nothing(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        with mock.patch("src.robot.execution.record_views.record_views") as kept:
            self.assertEqual("", keep_pick_views(self._service(), self._report(looks=()), name="x"))
            self.assertEqual("", keep_pick_views(self._service(views=()), self._report(), name="x"))
        kept.assert_not_called()

    def test_a_file_that_cannot_be_written_is_said_and_nothing_raises(self) -> None:
        from src.robot.execution.pick_run import keep_pick_views

        with mock.patch("src.robot.execution.record_views.record_views", side_effect=OSError("disk full")), \
                self.assertLogs("src.robot.execution.pick_run", level="WARNING") as said:
            self.assertEqual("", keep_pick_views(self._service(), self._report(), name="x"))
        self.assertIn("disk full", said.output[0])

    def test_a_campaign_keeps_its_looks_through_the_same_function(self) -> None:
        from src.robot.execution.pick_run import PickRun, Recording

        class _Service:
            def pick(self, **_: Any) -> Any:
                return SimpleNamespace(outcome="succeeded", fault=None, looks=("home",),
                                       telemetry={"attempt_id": "pick-1"}, failure_summary=lambda: "")

        with mock.patch("src.robot.execution.pick_run.keep_pick_views", return_value="kept") as kept:
            run_report = PickRun.from_service(_Service(), runs=1, recording=Recording.off(),
                                              record_views=True).execute()

        self.assertEqual("kept", run_report.attempts[0].views_file)
        self.assertEqual("run000", kept.call_args.kwargs["name"])


class AFailedPicksPicturesKeepItsGraspsTests(unittest.TestCase):
    """The debug pictures draw the judged look's grasps; where it ranked none, as after a blocker was set aside, the
    grasps of the pick's last look that ranked any (``ranked_looked``), so a failed pick's pictures still show what it
    tried (the owner's debug pictures, 2026-10-05; missing on failed picks, 2026-10-06)."""

    def _drawn(self, tmp: str, judged: Any, ranked: Any) -> Any:
        import cv2
        import numpy as np

        from src.robot.execution.pick_run import _keep_debug_images
        from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

        intrinsics = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
        frame = SimpleNamespace(rgb=np.zeros((240, 320, 3), dtype=np.uint8), intrinsics=intrinsics, segmentations=())
        view = SimpleNamespace(frame=frame, camera_to_base=np.eye(4), name="look", label="part")
        grasp = GraspPoint(position=np.array([0.0, 0.0, 400.0]), approach=np.array([0.0, 0.0, 1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=40.0, score=0.9, frame=GraspFrame.BASE)
        with_grasp = SimpleNamespace(result=SimpleNamespace(candidates=(grasp,)))
        none = SimpleNamespace(result=SimpleNamespace(candidates=()))
        from pathlib import Path

        written = Path(tmp) / "pick.npz"
        paths = _keep_debug_images((view,), with_grasp if judged else none, written,
                                   ranked=with_grasp if ranked else None)
        return cv2.imread(str(paths[0]))

    def test_a_look_that_ranked_none_draws_the_last_that_did(self) -> None:
        import tempfile

        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            bare = self._drawn(tmp, judged=False, ranked=False)
            fallen_back = self._drawn(tmp, judged=False, ranked=True)
            judged = self._drawn(tmp, judged=True, ranked=False)
        self.assertTrue(np.array_equal(fallen_back, judged), "the last ranked look's grasp was not drawn")
        self.assertFalse(np.array_equal(bare, judged), "the control: a grasp draws a rectangle")


class TheTaskSetsAndPutsBackTests(unittest.TestCase):
    def test_the_prompt_the_axis_the_overlay_and_the_cancel_check_are_the_tasks_alone(self) -> None:
        from src.robot.execution.task import TaskOptions

        ran = run(["part", "empty", "empty"], scope="until_empty",
                  options=TaskOptions(closing_axis="-y", overlay=True))

        service = ran.service
        # Every red cube in a box of its own, each mapped onto the object the task named (2026-10-06).
        self.assertEqual(["each separate red cube", "object"], [prompt.phrase for prompt in service.prompts],
                         "the task's prompt was not set, or the cell's own not put back")
        self.assertEqual(("red cube", ("red cube",)), (service.prompts[0].target_label, service.prompts[0].object_labels))
        self.assertEqual([closing_axis_of("-y"), None], service.closing_axes)
        self.assertEqual([True, True, True], service.rendering_seen)
        self.assertIs(False, service.rendering, "the overlay outlived the task")
        self.assertTrue(all(callable(check) for check in service.cancel_seen))
        self.assertIsNone(service.cancel_check)

    def test_everything_is_put_back_when_a_problem_ends_the_task(self) -> None:
        from src.robot.execution.task import TaskOptions, TaskStop

        ran = run(["fault"], options=TaskOptions(closing_axis="x"))

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop)
        self.assertEqual("object", ran.service.prompt.phrase)
        self.assertIsNone(ran.service.closing_axis)
        self.assertIs(False, ran.service.rendering)
        self.assertIsNone(ran.service.cancel_check)

    def test_an_overlay_switched_off_stays_off_and_one_found_on_stays_on(self) -> None:
        from src.robot.execution.task import TaskOptions

        off = run(["part"], options=TaskOptions(overlay=False))
        self.assertEqual([False], off.service.rendering_seen)

    def test_the_campaign_starts_with_the_tasks_push_distance(self) -> None:
        from src.robot.execution.task import TaskOptions

        self.assertEqual([{"push_mm": 40.0}], run(["part"], options=TaskOptions(push_mm=40.0)).service.campaigns)
        self.assertEqual([{"push_mm": None}], run(["part"]).service.campaigns)

    def test_a_task_that_names_no_object_grounds_every_part_in_a_box_of_its_own(self) -> None:
        """Qwen3-VL-4B grounded one box of 20 parts for "object" and all 20 for "each separate object" (2026-10-06): a
        task that names no object asks for the latter on a cell that grounds a phrase, every part a target and called
        "object" where its words allow, and puts the cell's prompt back after; a cell that grounds none is asked
        nothing."""
        from src.robot.execution.task import EVERY_PART_PHRASE, PlaceAt, TaskOptions, TaskPlan

        plan = TaskPlan(object="", place=PlaceAt(pose="drop_left"), return_to="home", scope="once",
                        options=TaskOptions(pick_anything=True))
        grounding = run(["part"], plan=plan, grounds=True).service
        asked, put_back = grounding.prompts
        self.assertEqual((EVERY_PART_PHRASE, None, ("object",)),
                         (asked.phrase, asked.target_label, asked.object_labels))
        self.assertEqual("object", put_back.phrase)
        self.assertEqual([], run(["part"], plan=plan, grounds=False).service.prompts)

    def test_a_task_that_takes_every_part_takes_a_blocker_as_the_pick(self) -> None:
        """The owner, 2026-10-06: where the task takes every part into one place, a blocker goes there too (the
        cell's ``recovery.blocker_into_the_place``, on); a task that names an object, or says otherwise, sets it aside."""
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan

        def plan(obj: str, **options: object) -> TaskPlan:
            return TaskPlan(object=obj, place=PlaceAt(pose="drop_left"), return_to="home", scope="once",
                            options=TaskOptions(pick_anything=True, **options))  # type: ignore[arg-type]

        self.assertEqual([{"push_mm": None, "blocker_is_the_pick": True}],
                         run(["part"], plan=plan("")).service.campaigns)
        self.assertEqual([{"push_mm": None}], run(["part"], plan=plan("red cube")).service.campaigns)
        self.assertEqual([{"push_mm": None}],
                         run(["part"], plan=plan("", blocker_into_the_place=False)).service.campaigns)


class ThePickIsHandedTheSwitchesTests(unittest.TestCase):
    def test_multi_view_off_and_both_faces_reach_every_pick(self) -> None:
        from src.robot.execution.task import TaskOptions

        ran = run(["empty", "empty"], options=TaskOptions(multi_view=False, both_faces=True))

        for call in ran.service.calls:
            self.assertIs(False, call["multi_view"])
            self.assertIs(True, call["both_faces"])

    def test_the_defaults_hand_the_pick_neither_switch(self) -> None:
        ran = run(["part"])
        self.assertNotIn("multi_view", ran.service.calls[0])
        self.assertNotIn("both_faces", ran.service.calls[0])

    def test_a_wrist_camera_looks_from_its_configured_looks_else_from_home(self) -> None:
        from src.robot.core import JointPositions

        look = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
        configured = run(["part"], wrist=True, looks=(look,))
        bare = run(["part"], wrist=True)
        fixed = run(["part"])

        self.assertEqual((look,), configured.service.calls[0]["look"])
        self.assertEqual("home", bare.service.calls[0]["look"])
        self.assertNotIn("look", fixed.service.calls[0])

    def test_the_looks_of_every_pick_are_kept_where_the_task_asks_and_said_on_its_report(self) -> None:
        from src.robot.execution.task import TaskOptions

        hooks = RecordingHooks()
        with mock.patch("src.robot.execution.task.keep_pick_views", return_value="D:/views/a.npz") as kept:
            run(["part", Pick("empty"), "empty"], scope="until_empty", hooks=hooks,
                options=TaskOptions(record_views=True))

        self.assertEqual(3, kept.call_count)
        self.assertEqual(["task-part1-pick1", "task-part2-pick2", "task-part2-pick3"],
                         [call.kwargs["name"] for call in kept.call_args_list])
        self.assertEqual("D:/views/a.npz", hooks.picks[0][2].telemetry["views_file"])

    def test_no_looks_are_kept_unless_asked(self) -> None:
        hooks = RecordingHooks()
        with mock.patch("src.robot.execution.task.keep_pick_views") as kept:
            run(["part"], hooks=hooks)

        kept.assert_not_called()
        self.assertNotIn("views_file", hooks.picks[0][2].telemetry)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
