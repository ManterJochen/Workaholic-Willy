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


class TheTaskSetsAndPutsBackTests(unittest.TestCase):
    def test_the_prompt_the_axis_the_overlay_and_the_cancel_check_are_the_tasks_alone(self) -> None:
        from src.robot.execution.task import TaskOptions

        ran = run(["part", "empty", "empty"], scope="until_empty",
                  options=TaskOptions(closing_axis="-y", overlay=True))

        service = ran.service
        self.assertEqual(["red cube", "object"], [prompt.phrase for prompt in service.prompts],
                         "the task's prompt was not set, or the cell's own not put back")
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
