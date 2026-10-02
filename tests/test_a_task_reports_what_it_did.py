"""A task says what it did, in the words the console speaks, and a program reaches it through the library's door.

* Every way a task ends (``TaskStop``) is a stop code of the console (``api.codes.StopCode``), and every event it says
  (``TaskEvent``) is an event type of the console, so the console publishes the library's words as they are and the
  frontend translates codes, never sentences.
* The hooks a task calls are exactly the five the console answers (``api.task_run.ConsoleTaskHooks``): what it says,
  a pick done, stop after this part, halt, and a cell being taken down. Every event carries the library's own English
  sentence (``said``), which the console publishes as the envelope's ``human`` and keeps out of ``data``.
* ``TaskReport`` prints as it renders, plain ASCII, and is plain data as a dict, with the last pick's own report in it.
* ``from willy import run_task, TaskPlan, PlaceAt, TaskOptions, TaskReport, TaskStop`` reaches the library's objects,
  as does ``src.robot.execution``.
"""

from __future__ import annotations

import inspect
import json
import unittest

from tests._task_fakes import run


class TheConsolesWordsTests(unittest.TestCase):
    def test_every_way_a_task_ends_is_a_stop_code_of_the_console(self) -> None:
        from api.codes import StopCode
        from src.robot.execution.task import TaskStop

        self.assertLessEqual({stop.value for stop in TaskStop}, {code.value for code in StopCode})
        self.assertEqual(19, len(TaskStop))

    def test_the_library_sorts_its_stops_as_the_console_does(self) -> None:
        """A benign end returns the arm and a problem stop leaves it where it stands: the library decides that, and the
        console's classes (``STOP_CLASS``) say the same of every stop."""
        from api.codes import STOP_CLASS, StopCode
        from src.robot.execution.task import TaskStop

        for stop in TaskStop:
            with self.subTest(stop=stop):
                self.assertEqual(STOP_CLASS[StopCode(stop.value)].value, stop.stop_class)

    def test_every_event_a_task_says_is_an_event_type_of_the_console(self) -> None:
        from api.codes import EventType
        from src.robot.execution.task import TaskEvent

        task_events = {event.value for event in EventType if event.value.startswith("task.")}
        self.assertEqual(task_events, {event.value for event in TaskEvent})

    def test_every_refusal_a_task_raises_is_a_refusal_code_of_the_console(self) -> None:
        """The console maps ``TaskRefused.code`` onto its own refusal (``api.codes.RefusalCode`` and its status), so a
        code the catalog does not know would turn the refusal of a run that moved nothing into a software error. Read
        off every ``TaskRefused(...)`` the library raises, each with its code written out."""
        import ast

        from api.codes import RefusalCode
        from src.robot.execution import task

        calls = [node for node in ast.walk(ast.parse(inspect.getsource(task)))
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "TaskRefused"]
        codes = {call.args[0].value for call in calls if call.args and isinstance(call.args[0], ast.Constant)}

        self.assertEqual(len(calls), sum(1 for call in calls if call.args and isinstance(call.args[0], ast.Constant)),
                         "a refusal whose code is not written out")
        self.assertEqual({"unknown_pose", "route_refused", "carried_part_not_modelled", "object_required",
                          "closing_axis_refused", "bad_request", "push_distance_refused", "camera_target_unavailable"},
                         codes)
        self.assertLessEqual(codes, {code.value for code in RefusalCode})

    def test_the_hooks_a_task_calls_are_the_ones_the_console_answers(self) -> None:
        from api.task_run import ConsoleTaskHooks
        from src.robot.execution.task import TaskHooks

        called = {name for name, member in vars(TaskHooks).items() if not name.startswith("_") and callable(member)}
        self.assertEqual({"event", "pick_done", "stop_after_part", "halted", "abandoned"}, called)
        self.assertLessEqual(called, {name for name in dir(ConsoleTaskHooks)})
        self.assertEqual(["self", "name", "data"], list(inspect.signature(TaskHooks.event).parameters))

    def test_every_event_carries_its_sentence_and_its_data_is_plain(self) -> None:
        ran = run(["part", "empty", "empty"], scope="until_empty")

        self.assertEqual(len(ran.hooks.events), len(ran.hooks.said))
        for sentence in ran.hooks.said:
            self.assertEqual(sentence, sentence.encode("ascii", "backslashreplace").decode("ascii"))
        json.dumps([data for _name, data in ran.hooks.events])


class TheReportTests(unittest.TestCase):
    def test_it_prints_as_it_renders_and_is_plain_data(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "empty", "empty"], scope="until_empty")
        report = ran.report

        self.assertEqual(report.render(), str(report))
        self.assertIn("NOTHING_LEFT", report.render())
        self.assertIn("1 part(s) placed", report.render())
        as_data = report.to_dict()
        json.dumps(as_data)
        self.assertEqual("nothing_left", as_data["stop"])
        self.assertEqual("done", as_data["stop_class"])
        self.assertEqual((1, 3, 1, False), (as_data["parts_placed"], as_data["picks"], as_data["succeeded"],
                                            as_data["holding"]))
        self.assertEqual("no_target", as_data["last_report"]["outcome"])
        self.assertIs(TaskStop.NOTHING_LEFT, report.stop)

    def test_a_problem_stop_says_its_class(self) -> None:
        ran = run(["fault"])
        self.assertEqual("problem", ran.report.to_dict()["stop_class"])
        self.assertIn("the arm stays where it stands", ran.report.render())
        self.assertFalse(ran.report.at_return)

    def test_a_benign_end_says_where_the_arm_stands(self) -> None:
        """A benign end after a motion stands at the return pose; one before any motion moved nothing, and says so
        rather than claim a return that never ran."""
        from src.robot.execution.task import TaskStop
        from src.robot.safety.planning.band import PoseVerdict
        from tests._task_fakes import PLACE_TCP, TaskArm

        done = run(["part"])
        self.assertTrue(done.report.at_return)
        self.assertIn("the arm is back at its return pose", done.report.render())
        self.assertIs(True, done.report.to_dict()["at_return"])

        joints = TaskArm([]).nearest_configuration(PLACE_TCP)
        key = tuple(round(v, 1) for v in joints.degrees())
        refused = run(["part"], arm=TaskArm([], screens={key: PoseVerdict.PLANNER_REFUSED}))
        self.assertIs(TaskStop.POSE_REFUSED, refused.report.stop)
        self.assertFalse(refused.report.at_return)
        self.assertIn("the task moved nothing", refused.report.render())
        self.assertNotIn("return pose", refused.report.render())

    def test_a_task_that_picked_nothing_reports_no_pick(self) -> None:
        from tests._task_fakes import RecordingHooks

        ran = run(["part"], hooks=RecordingHooks(halt=True))
        self.assertIsNone(ran.report.last_report)
        self.assertIsNone(ran.report.to_dict()["last_report"])


class TheDoorsTests(unittest.TestCase):
    def test_willy_and_the_execution_package_hand_out_the_task(self) -> None:
        import willy
        from src.robot import execution
        from src.robot.execution import task

        for name in ("run_task", "TaskPlan", "PlaceAt", "TaskOptions", "TaskReport", "TaskStop"):
            with self.subTest(name):
                self.assertIs(getattr(task, name), getattr(willy, name))
                self.assertIs(getattr(task, name), getattr(execution, name))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
