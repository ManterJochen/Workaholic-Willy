"""A task moves nothing while a person is needed, or while it was handed what it cannot do (build plan 1.3.3).

* **The service's latch.** A recovery of an earlier pick that stopped where the arm stands (a push that stopped) leaves
  the pick service a person to wait for (``stopped_where_the_arm_stands``). A task checks it FIRST and ends
  ``recovery_needs_person`` with nothing read off the cell, asked or commanded: it never calls ``start_campaign``,
  which would acknowledge the latch, and it never clears it, so a program that imports ``run_task`` from ``willy``
  meets the gate the console meets. A pick of the task that sets the latch ends the task there, the latch kept.
* **Halt and Disconnect.** A task asked to halt, or whose cell is being taken down, before it began moves nothing; nor
  does a task on an arm whose halt latch is set (L9, read where the arm has it).
* **What it was handed.** An arm whose place and return cannot run (no planner judges its motions: a UR on ik, a
  KUKA), a carried part the planner cannot model (``payload_declined_reason``), a pose name it does not know, an empty
  object on a cell that grounds a phrase without the operator's ``pick_anything`` (Q7 A+), a closing axis that names
  none, both jaw faces asked of a cell that turns every grasp off the faces judged, a push distance the cell refuses,
  and a camera place on a cell with no camera to locate with are refused with :class:`TaskRefused`, carrying the
  console's refusal code, before anything is commanded. The console refuses each of them before it starts a run; this
  is the library's own backstop, which a program meets too.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any

from src.contracts import UNSET
from tests._task_fakes import RecordingHooks, TaskArm, TaskService, do0_changes, motions, run, toggle


class TheLatchTests(unittest.TestCase):
    def test_a_task_with_the_service_latch_set_moves_nothing_and_leaves_the_latch_set(self) -> None:
        """Red before: there was no task; the console's pick run called start_campaign first, which cleared it."""
        from src.robot.execution.task import TaskStop

        ran = run(["part"], needs_person="a push stopped where the arm stands")

        self.assertIs(TaskStop.RECOVERY_NEEDS_PERSON, ran.report.stop)
        self.assertEqual("a push stopped where the arm stands", ran.service.stopped_where_the_arm_stands)
        self.assertEqual([], ran.service.campaigns, "start_campaign was called, which acknowledges the latch")
        self.assertEqual([], ran.service.calls)
        self.assertEqual([], motions(ran.log))
        self.assertEqual(0, do0_changes(ran.log))
        self.assertEqual([], ran.names(), "the task said a step although it did not start")
        self.assertIn("a push stopped where the arm stands", ran.report.sentence)

    def test_a_pick_that_stops_where_the_arm_stands_ends_the_task_there_with_the_latch_kept(self) -> None:
        from src.robot.execution.task import TaskStop
        from tests._task_fakes import Pick

        log: list[Any] = []
        arm = TaskArm(log)
        at_the_stop: list[int] = []
        ran = run(["part", Pick("needs_person", then=lambda: at_the_stop.append(len(motions(log)))), "part"],
                  scope="until_empty", arm=arm)

        self.assertIs(TaskStop.RECOVERY_NEEDS_PERSON, ran.report.stop)
        self.assertEqual(2, ran.report.picks)
        self.assertTrue(ran.service.stopped_where_the_arm_stands, "the task cleared the latch on its way out")
        self.assertEqual(1, len(ran.service.campaigns))
        self.assertEqual(at_the_stop, [len(motions(log))], "the arm moved after the pick that stopped")
        self.assertEqual(2, do0_changes(log), "the jaws changed after the pick that stopped")


class HaltedOrGoneBeforeTheStartTests(unittest.TestCase):
    def test_a_halt_before_the_start_moves_nothing(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"], hooks=RecordingHooks(halt=True))

        self.assertIs(TaskStop.HALTED, ran.report.stop)
        self.assertEqual([], motions(ran.log))
        self.assertEqual([], ran.service.calls)

    def test_an_arm_whose_halt_latch_is_set_moves_nothing(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log)
        arm.halted = "the console halted the arm"
        ran = run(["part"], arm=arm)

        self.assertIs(TaskStop.HALTED, ran.report.stop)
        self.assertIn("the console halted the arm", ran.report.sentence)
        self.assertEqual([], motions(log))

    def test_a_halt_latch_that_cannot_be_read_moves_nothing(self) -> None:
        """A latch that cannot be read is no latch that is open: the task reads it as a halt."""
        from src.robot.execution.task import TaskStop

        class _Unreadable(TaskArm):
            def halt_state(self) -> Any:
                raise OSError("the link dropped")

        log: list[Any] = []
        ran = run(["part"], arm=_Unreadable(log))

        self.assertIs(TaskStop.HALTED, ran.report.stop, ran.report.sentence)
        self.assertIn("could not be read", ran.report.sentence)
        self.assertEqual([], motions(log))
        self.assertEqual([], ran.service.campaigns)
        self.assertEqual([], ran.service.calls)

    def test_a_cell_being_taken_down_moves_nothing_and_says_why_first(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"], hooks=RecordingHooks(gone="the cell is being disconnected", halt=True))

        self.assertIs(TaskStop.DISCONNECTED, ran.report.stop, "the halt was read before the disconnect")
        self.assertIn("the cell is being disconnected", ran.report.sentence)
        self.assertEqual([], motions(ran.log))


class RefusedBeforeAnythingMovesTests(unittest.TestCase):
    def _refused(self, code: str, **keywords: Any) -> Any:
        from src.robot.execution.task import TaskRefused

        with self.assertRaises(TaskRefused) as raised:
            run(["part"], **keywords)
        self.assertEqual(code, raised.exception.code)
        self.assertIsInstance(raised.exception, ValueError)
        return raised.exception

    def test_a_carried_part_the_planner_cannot_model_is_refused(self) -> None:
        log: list[Any] = []
        arm = TaskArm(log, declined="safety.planning_world.payload.length_mm is undeclared")
        refused = self._refused("carried_part_not_modelled", arm=arm)

        self.assertIn("length_mm", str(refused))
        self.assertEqual([], motions(log))

    def test_a_cell_that_chose_no_carried_part_runs_its_task(self) -> None:
        """The owner, 2026-10-07: "er lässt mich nicht fahren, wenn ich payload nicht aktiviert habe". A cell whose
        ``payload.enabled`` is false judges a closed hand as an empty one (2026-10-05), and its task carries so."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, declined="safety.planning_world.payload.enabled is false, so neither the planner nor the "
                                    "self filter models a part in the gripper")
        arm.config = SimpleNamespace(safety=SimpleNamespace(planning_world=SimpleNamespace(payload=SimpleNamespace(enabled=False))))  # type: ignore[attr-defined]
        ran = run(["part"], arm=arm)
        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.picks)

    def test_an_arm_whose_place_and_return_cannot_run_is_refused_before_it_picks(self) -> None:
        """``Robot.place`` and ``Robot.home`` refuse an arm whose motions no planner judges (``route_of``), so a part
        picked there could not be set down. Red before: the task picked it and stopped ``part_still_held`` with the
        part in the jaws."""
        from src.robot.core.arm_capabilities import LineMotion, LineReading

        class _Ik(TaskArm):
            plans_paths = False

            def line_motion(self) -> LineReading:
                return LineReading(LineMotion.CONTROLLER_LINE, "the controller draws the line, judged at its end only")

        log: list[Any] = []
        refused = self._refused("route_refused", arm=_Ik(log))

        self.assertIn("judged at its end only", str(refused))
        self.assertEqual([], motions(log))
        self.assertEqual(0, do0_changes(log))

    def test_both_jaw_faces_on_a_cell_that_turns_every_grasp_off_them_are_refused_before_anything_moves(self) -> None:
        """Red before: the pick raised it, after a Restart's move home had already run, and the task let it out raw."""
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, TaskRefused, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        arm = TaskArm(log)
        service = TaskService(arm, toggle(log, arm=arm), ["part"])
        service.runtime.orchestrator.policy.align_closing_to_base_x = True
        hooks = RecordingHooks()

        with self.assertRaises(TaskRefused) as raised:
            run_task(service, TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"), first_motion="return",
                                       options=TaskOptions(both_faces=True)), hooks=hooks, poses=POSES)
        self.assertEqual("bad_request", raised.exception.code)
        self.assertIn("both_faces", str(raised.exception))
        self.assertEqual([], hooks.events)
        self.assertEqual([], motions(log))
        self.assertEqual([], service.calls)

    def test_a_pose_name_it_does_not_know_is_refused(self) -> None:
        from src.robot.execution.task import PlaceAt

        self._refused("unknown_pose", place=PlaceAt(pose="nowhere"))
        self._refused("unknown_pose", return_to="nowhere")

    def test_an_empty_object_on_a_cell_that_grounds_a_phrase_needs_the_operators_word(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, TaskStop

        self._refused("object_required", plan=TaskPlan(object=" ", place=PlaceAt(pose="drop_left")))

        anything = TaskPlan(object="", place=PlaceAt(pose="drop_left"), options=TaskOptions(pick_anything=True))
        ran = run(["part"], plan=anything)
        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertTrue(all(prompt.phrase for prompt in ran.service.prompts), "an empty phrase was set as the prompt")
        self.assertEqual("each separate object", ran.service.prompts[0].phrase, "every part is grounded on its own")

    def test_an_empty_object_on_a_cell_that_grounds_nothing_picks_what_it_shows(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop

        ran = run(["part"], plan=TaskPlan(object="", place=PlaceAt(pose="drop_left")), grounds=False)
        self.assertIs(TaskStop.FINISHED, ran.report.stop)

    def test_a_closing_axis_that_names_none_is_refused(self) -> None:
        from src.robot.execution.task import TaskOptions

        self._refused("closing_axis_refused", options=TaskOptions(closing_axis="sideways"))

    def test_a_closing_axis_the_cells_own_rule_refuses_is_refused_before_the_task_says_anything(self) -> None:
        """A cell that turns every grasp to base x itself (``align_closing_to_base_x``) refuses a second axis beside
        it: the task is refused before its first event, with nothing set and nothing moved."""
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, TaskRefused, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        arm = TaskArm(log)
        service = TaskService(arm, toggle(log, arm=arm), ["part"])
        service.runtime.orchestrator.policy.align_closing_to_base_x = True
        hooks = RecordingHooks()

        with self.assertRaises(TaskRefused) as raised:
            run_task(service, TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"),
                                       options=TaskOptions(closing_axis="y")), hooks=hooks, poses=POSES)
        self.assertEqual("closing_axis_refused", raised.exception.code)
        self.assertEqual([], hooks.events)
        self.assertEqual([], service.campaigns)
        self.assertEqual([], service.prompts)
        self.assertEqual([], motions(log))

    def test_a_closing_axis_the_setter_still_refuses_puts_the_setup_back(self) -> None:
        """The backstop: a setter that refuses what the read let through refuses the task too, and the prompt the task
        had set is put back."""
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, TaskRefused, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        arm = TaskArm(log)
        service = TaskService(arm, toggle(log, arm=arm), ["part"])
        before = service.prompt

        def refuse(axis: Any) -> None:
            raise ValueError("this cell aligns every grasp to base x (align_closing_to_base_x)")

        service.set_closing_axis = refuse  # type: ignore[method-assign]
        with self.assertRaises(TaskRefused) as raised:
            run_task(service, TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"),
                                       options=TaskOptions(closing_axis="y")), hooks=RecordingHooks(), poses=POSES)
        self.assertEqual("closing_axis_refused", raised.exception.code)
        self.assertIn("align_closing_to_base_x", str(raised.exception))
        self.assertEqual([], motions(log))
        self.assertEqual([], service.calls)
        self.assertEqual(before, service.prompt, "the prompt the task set outlived it")

    def test_a_push_distance_the_cell_refuses_is_refused_before_anything_moves(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan, TaskRefused, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        arm = TaskArm(log)
        service = TaskService(arm, toggle(log, arm=arm), ["part"])

        def refuse(*, push_mm: Any = UNSET) -> None:
            raise ValueError("a push of 60 mm is above the cell's ceiling of 50 mm")

        service.start_campaign = refuse  # type: ignore[method-assign]
        with self.assertRaises(TaskRefused) as raised:
            run_task(service, TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"),
                                       options=TaskOptions(push_mm=60.0)), hooks=RecordingHooks(), poses=POSES)
        self.assertEqual("push_distance_refused", raised.exception.code)
        self.assertIn("ceiling", str(raised.exception))
        self.assertEqual([], motions(log))
        self.assertEqual([], service.calls)

    def test_a_camera_place_on_a_cell_with_no_camera_to_locate_with_is_refused(self) -> None:
        from src.robot.execution.task import PlaceAt

        self._refused("camera_target_unavailable", place=PlaceAt(camera="blue bin"))

    def test_a_wrist_camera_place_on_an_arm_with_no_live_world_to_hold_its_frames_is_refused(self) -> None:
        from src.robot.execution.task import PlaceAt
        from tests._task_fakes import ScriptedLocator

        log: list[Any] = []
        arm = TaskArm(log, world=False)
        self._refused("camera_target_unavailable", place=PlaceAt(camera="blue bin"), arm=arm, wrist=True,
                      locators=[ScriptedLocator({}, arm=arm)])
        self.assertEqual([], motions(log))


class APlanThatIsNoPlanTests(unittest.TestCase):
    def test_a_place_names_a_pose_or_a_camera_phrase_and_only_one(self) -> None:
        from src.robot.execution.task import PlaceAt

        for build in (lambda: PlaceAt(), lambda: PlaceAt(pose="drop_left", camera="blue bin"),
                      lambda: PlaceAt(camera=" "), lambda: PlaceAt(pose="drop_left", air_mm=20.0),
                      lambda: PlaceAt(camera="blue bin", air_mm=5.0), lambda: PlaceAt(camera="blue bin", air_mm=60.0)):
            with self.subTest(), self.assertRaises(ValueError):
                build()
        self.assertEqual(20.0, PlaceAt(camera="blue bin", air_mm=20.0).air_mm)

    def test_a_plan_with_a_scope_or_a_limit_it_cannot_keep_is_refused(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan

        place = PlaceAt(pose="drop_left")
        for keywords in ({"scope": "twice"}, {"first_motion": "grasp"}, {"max_parts": 0}, {"max_failed_in_a_row": 0},
                         {"empty_looks_to_end": 0}, {"pose_keep_out_mm": 0.0}):
            with self.subTest(keywords=keywords), self.assertRaises(ValueError):
                TaskPlan(object="red cube", place=place, **keywords)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
