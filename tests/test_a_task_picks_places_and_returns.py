"""A task picks a part, places it, returns, and looks again: once, or until nothing is left (the owner's OD 7, OD 9).

``run_task(service, plan, hooks=, poses=)`` is the console's task as a library verb, which a program can call as the
console does (``from willy import run_task``). Each part is one turn of the loop: a pick that looks fresh, a place (a
taught pose here; the camera's bin in its own file), and the return to the return pose (home, or a pose the operator
taught). The scope ends it:

* ``once``: when the one part is placed and the arm is back;
* ``until_empty``: when two looks in a row saw nothing to pick (a look that saw only what the task keeps out is one of
  them); a ``once`` task that sees nothing looks one more time too;
* both: after three picks in a row failed (the arm stays where it stands), and at 100 parts placed (``part_limit``,
  since the rehearsal scene never empties). The two rows count apart, as the build plan writes them (1.3.3): an empty
  look adds to the empty row, a failed pick to the failed row, neither starts the other's again, and only a part
  placed starts both.

A benign end returns the arm to its return pose first; a problem stop leaves it where it stands, with nothing commanded
after (``tests/test_a_task_stops_where_a_problem_found_the_arm.py``). The operator's "stop after this part" lets the
part in hand finish: its pick, its place, the return; then the task ends.

The owner's hand is a toggle on DO0 with no sensor (``jaw_io`` single_toggle): every change of the output moves the
jaws. A task changes it exactly twice per part, the close at the pick and the open at the place, and the place leaves
the program's count OPEN, so the next pick starts on jaws the program knows open and asks nobody. That is read off the
recorded outputs here, on the scripted service and on the real one.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.core.errors import RobotError
from tests._task_fakes import (
    HOME,
    PARK_JOINTS,
    PLACE_TCP,
    GRASP_Z_MM,
    MeasuringHand,
    Pick,
    RecordingHooks,
    TaskArm,
    do0_changes,
    motions,
    run,
    toggle,
)


def _key(joints: Any) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


class OncePartTests(unittest.TestCase):
    def test_one_part_is_picked_placed_and_the_arm_returns_home(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"])

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual((1, 1, 1), (ran.report.parts_placed, ran.report.picks, ran.report.succeeded))
        self.assertFalse(ran.report.holding)
        self.assertEqual(("home",), motions(ran.log)[-1], "the arm did not end at home")
        self.assertEqual(["task.pose_screened", "task.part_started", "task.drop_planned", "task.place_started",
                          "task.placed", "task.return_started", "task.returned", "task.part_finished"], ran.names())
        self.assertEqual([{"part": 1, "of": 1, "pick": 1}], ran.hooks.of("task.part_started"))
        self.assertEqual([(1, 1)], [(part, pick) for part, pick, _ in ran.hooks.picks])
        self.assertEqual({"place": "pose:drop_left"}, ran.hooks.of("task.place_started")[0])
        self.assertEqual({"part": 1, "placed": True}, {k: v for k, v in ran.hooks.of("task.part_finished")[0].items()
                                                       if k != "duration_s"})

    def test_the_toggle_changes_twice_and_the_place_leaves_its_count_open(self) -> None:
        ran = run(["part"])

        self.assertEqual(2, do0_changes(ran.log), "a toggle hand changed DO0 other than once at the part and once "
                                                  "at the place")
        self.assertIs(False, ran.jaws.jaws_closed, "the place left the program's count closed")
        self.assertEqual("", ran.jaws.why_jaws_unknown())

    def test_the_part_is_let_go_where_its_bottom_meets_the_taught_pose(self) -> None:
        ran = run(["part"])

        line_in = [entry for entry in motions(ran.after("task.place_started")) if entry[0] == "move"][1]
        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), line_in[1], atol=1e-6)
        self.assertTrue(line_in[2], "the drop was not a line in")

    def test_a_return_to_a_taught_pose_goes_there(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"], return_to="park")

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertEqual(("joints", _key(PARK_JOINTS)), motions(ran.log)[-1])
        self.assertEqual([{"to": "park", "note": ""}], ran.hooks.of("task.returned"))

    def test_a_once_task_that_sees_nothing_looks_once_more_and_ends_nothing_left_at_home(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["empty", "empty"])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop)
        self.assertEqual(2, ran.report.picks)
        self.assertEqual([1, 2], [event["empty_in_a_row"] for event in ran.hooks.of("task.nothing_found")])
        self.assertEqual(("home",), motions(ran.log)[-1])
        self.assertEqual(0, do0_changes(ran.log))

    def test_a_once_task_whose_second_look_finds_the_part_places_it(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["empty", "part"])

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertEqual([{"part": 1, "of": 1, "pick": 1}, {"part": 1, "of": 1, "pick": 2}],
                         ran.hooks.of("task.part_started"))


class UntilEmptyTests(unittest.TestCase):
    def test_two_parts_then_two_empty_looks_end_nothing_left(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "part", "empty", "empty"], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((2, 4, 2), (ran.report.parts_placed, ran.report.picks, ran.report.succeeded))
        self.assertEqual(4, do0_changes(ran.log))
        self.assertEqual(("home",), motions(ran.log)[-1])
        self.assertEqual([None, None, None, None], [event["of"] for event in ran.hooks.of("task.part_started")])
        self.assertEqual([1, 2, 3, 3], [event["part"] for event in ran.hooks.of("task.part_started")])

    def test_exactly_two_changes_of_do0_for_every_part(self) -> None:
        ran = run(["part", "empty", "part", "part", "empty", "empty"], scope="until_empty")

        per_part: list[int] = []
        count = 0
        for entry in ran.log:
            if entry == ("event", "task.part_finished"):
                per_part.append(count)
                count = 0
            elif isinstance(entry, tuple) and entry[:1] == ("DO",) and getattr(entry, "changed", False):
                count += 1
        self.assertEqual([2, 2, 2], per_part)
        self.assertEqual(0, count, "DO0 changed after the last part")

    def test_an_empty_look_between_two_is_no_row(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "empty", "part", "empty", "empty"], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop)
        self.assertEqual(2, ran.report.parts_placed)

    def test_the_part_limit_ends_a_task_that_never_empties(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop

        plan = TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"), scope="until_empty", max_parts=3)
        ran = run(["part"] * 3, plan=plan)

        self.assertIs(TaskStop.PART_LIMIT, ran.report.stop)
        self.assertEqual(3, ran.report.parts_placed)
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_the_owners_limit_is_100_parts(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan

        plan = TaskPlan(object="red cube", place=PlaceAt(pose="drop_left"))
        self.assertEqual((100, 3, 2, 150.0), (plan.max_parts, plan.max_failed_in_a_row, plan.empty_looks_to_end,
                                              plan.pose_keep_out_mm))


class FailedPicksTests(unittest.TestCase):
    def test_three_failed_picks_in_a_row_end_the_task_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["failed", "failed", "failed"], scope="until_empty")

        self.assertIs(TaskStop.FAILED_IN_A_ROW, ran.report.stop)
        self.assertEqual(3, ran.report.picks)
        self.assertEqual([], motions(ran.log), "the arm was sent somewhere after three failed picks")
        self.assertNotIn("task.return_started", ran.names())

    def test_a_success_starts_the_row_again(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["failed", "failed", "part", "failed", "failed", "part", "empty", "empty"], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop)
        self.assertEqual(2, ran.report.parts_placed)

    def test_the_failed_row_and_the_empty_row_count_apart_and_a_part_placed_starts_both_again(self) -> None:
        """The build plan's rule (1.3.3): a failed pick does not start the empty row again, nor an empty look the
        failed row; only a part placed does. Red before: a failed pick started the empty row again, so a task that
        alternated the two ran on to three failures, a problem stop where the arm stands, instead of ending
        ``nothing_left`` at home at its second empty look."""
        from src.robot.execution.task import TaskStop

        for picks, stop, count in ((["empty", "failed", "empty"], TaskStop.NOTHING_LEFT, 3),
                                   (["failed", "empty", "failed", "empty"], TaskStop.NOTHING_LEFT, 4),
                                   (["failed", "empty", "failed", "failed"], TaskStop.FAILED_IN_A_ROW, 4),
                                   (["failed", "failed", "part", "empty", "failed", "empty"], TaskStop.NOTHING_LEFT,
                                    6)):
            with self.subTest(picks=picks):
                ran = run(picks, scope="until_empty")

                self.assertIs(stop, ran.report.stop, ran.report.sentence)
                self.assertEqual(count, ran.report.picks)
                if stop is TaskStop.NOTHING_LEFT:
                    self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_look_that_saw_only_parts_skipped_after_a_failed_pick_is_a_failed_pick_not_an_empty_look(self) -> None:
        """Only a part the task keeps out (its bin, its drop) makes a look empty. A part ``next_target`` skips because
        a pick failed on it still stands there to be picked. Red before: any exclusion counted as an empty look, and
        two of them ended the task ``nothing_left`` at home, "done", with the failed part on the bench."""
        from src.robot.execution.task import TaskStop

        ran = run(["zoned", "zoned", "zoned"], scope="until_empty")

        self.assertIs(TaskStop.FAILED_IN_A_ROW, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.picks)
        self.assertEqual([], ran.hooks.of("task.nothing_found"))
        self.assertNotIn("task.return_started", ran.names())

    def test_a_pick_that_failed_with_the_part_closed_in_the_jaws_ends_the_task_holding_it(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["failed_holding", "part"], scope="until_empty")

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, ran.report.picks, "a pick started with a part in the jaws")
        self.assertEqual(1, do0_changes(ran.log))


class StopAfterThisPartTests(unittest.TestCase):
    def test_the_part_in_hand_is_placed_the_arm_returns_and_the_task_ends(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()

        def press(name: str, _data: dict[str, Any]) -> None:
            if name == "task.part_started":
                hooks.stop = True

        hooks.on_event = press
        ran = run(["part", "part"], scope="until_empty", hooks=hooks)

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop)
        self.assertEqual((1, 1), (ran.report.parts_placed, ran.report.picks))
        self.assertEqual(2, do0_changes(ran.log))
        self.assertEqual(1, len([m for m in motions(ran.log) if m == ("home",)]), "the arm returned twice")
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_a_stop_before_the_task_moved_ends_it_with_nothing_moved(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks(stop=True)
        ran = run(["part"], hooks=hooks)

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop)
        self.assertEqual([], motions(ran.log))
        self.assertEqual(0, ran.report.picks)

    def test_a_stop_during_a_pick_that_found_nothing_returns_and_ends(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        ran = run([Pick("empty", then=lambda: setattr(hooks, "stop", True))], scope="until_empty", hooks=hooks)

        self.assertIs(TaskStop.STOPPED_AFTER_PART, ran.report.stop)
        self.assertEqual(("home",), motions(ran.log)[-1])


class TheHandBeforeEachPickTests(unittest.TestCase):
    def test_a_toggle_the_program_believes_closed_ends_the_task_before_anything_moves(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part"], closed=True)

        self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop)
        self.assertEqual(0, ran.report.picks)
        self.assertEqual([], motions(ran.log))
        self.assertEqual(0, do0_changes(ran.log))
        self.assertIn("CLOSED", ran.report.sentence)

    def test_a_toggle_the_program_believes_closed_moves_no_survey_and_no_restart_home(self) -> None:
        """The hand is read before the task's first motion of any kind, not only before its first pick: a wrist survey
        and a Restart's move home carry whatever the jaws hold too."""
        from src.robot.execution.task import PlaceAt, TaskStop
        from tests._task_fakes import ScriptedLocator

        restart = run(["part"], closed=True, first_motion="return")
        self.assertIs(TaskStop.HAND_NEEDS_PERSON, restart.report.stop, restart.report.sentence)
        self.assertEqual([], motions(restart.log))

        log: list[Any] = []
        arm = TaskArm(log)
        survey = run(["part"], closed=True, arm=arm, place=PlaceAt(camera="blue bin"), wrist=True,
                     looks=(PARK_JOINTS,), locators=[ScriptedLocator({}, arm=arm)])
        self.assertIs(TaskStop.HAND_NEEDS_PERSON, survey.report.stop, survey.report.sentence)
        self.assertEqual([], motions(log))
        self.assertNotIn("task.survey_started", survey.names())

    def test_a_hand_that_still_holds_a_part_moves_nothing_and_starts_no_pick(self) -> None:
        """The console's ``part_still_held`` gate, the library's own copy: a hand that measures a part, or a planner
        that still models one, before the task's first motion of any kind. Red before: only a toggle's count was read,
        so a Restart drove home with the part and a pick opened the jaws over the bench, dropping it."""
        from src.robot.execution.task import TaskStop

        def measured(log: list[Any]) -> tuple[TaskArm, Any]:
            return TaskArm(log), MeasuringHand(holding=True)

        def modelled(log: list[Any]) -> tuple[TaskArm, Any]:
            arm = TaskArm(log)
            arm.attach_payload(40.0)
            return arm, toggle(log, arm=arm)

        for name, held in (("measured", measured), ("modelled", modelled)):
            for first_motion in ("look", "return"):
                with self.subTest(held=name, first_motion=first_motion):
                    log: list[Any] = []
                    arm, hand = held(log)
                    ran = run(["part"], arm=arm, jaws=hand, first_motion=first_motion)

                    self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
                    self.assertTrue(ran.report.holding)
                    self.assertEqual([], motions(log))
                    self.assertEqual([], ran.service.calls)
                    self.assertEqual(0, do0_changes(log))
                    self.assertEqual([], getattr(hand, "commands", []), "the hand was commanded")

    def test_a_part_the_planner_still_models_after_a_place_starts_no_next_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log)
        ran = run(["part", "part"], arm=arm, scope="until_empty", hooks=RecordingHooks(
            on_event=lambda name, _data: arm.attach_payload(40.0) if name == "task.part_finished" else None))

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertEqual((1, 1), (ran.report.parts_placed, ran.report.picks))
        self.assertEqual([], motions(ran.after("task.part_finished")))

    def test_an_output_switched_at_the_pendant_between_two_parts_starts_no_next_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log)
        jaws = toggle(log, arm=arm)

        def pendant(name: str, _data: Any) -> None:
            if name == "task.part_finished":
                jaws._io.do[0] = not jaws._io.do[0]  # noqa: SLF001 (the pendant switched DO0)

        ran = run(["part", "part"], arm=arm, jaws=jaws, scope="until_empty", hooks=RecordingHooks(on_event=pendant))
        self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop, ran.report.sentence)
        self.assertEqual((1, 1), (ran.report.parts_placed, ran.report.picks))
        self.assertEqual([], motions(ran.after("task.part_finished")))

    def test_a_toggle_whose_output_was_switched_by_hand_ends_the_task_and_asks_nobody(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log)
        jaws = toggle(log, arm=arm)
        jaws._io.do[0] = not jaws._io.do[0]  # noqa: SLF001 (the pendant switched DO0)
        ran = run(["part"], arm=arm, jaws=jaws)

        self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop)
        self.assertEqual(0, ran.report.picks)
        self.assertIn("nobody can say where", ran.report.sentence)

    def test_the_library_copy_of_the_consoles_check_answers_as_the_console_did(self) -> None:
        from src.robot.execution.task import jaws_before_a_pick

        log: list[Any] = []
        self.assertEqual("", jaws_before_a_pick(toggle(log)))
        self.assertIn("CLOSED", jaws_before_a_pick(toggle(log, closed=True)))
        self.assertEqual("", jaws_before_a_pick(None))
        self.assertEqual("", jaws_before_a_pick(object()))

        class _Raising:
            toggles_without_sensor = True
            edge_unknown = False
            jaws_closed = False
            is_connected = True

            def why_jaws_unknown(self) -> str:
                raise RobotError("the link dropped")

            def jaws_open_for_a_pick(self) -> str:  # pragma: no cover - never asked
                raise AssertionError

            def asking_nobody(self, why: str) -> Any:  # pragma: no cover
                raise AssertionError

        said = jaws_before_a_pick(_Raising())
        self.assertIn("could not be read", said)


class TheRealServiceTests(unittest.TestCase):
    """The task on the real pick service: its pick loop and the guarded execution policy on the arm, the owner's toggle
    closing once at each part, ``Robot.place`` opening it once at the place, ``Robot.home`` returning."""

    def _service(self, log: list[Any], frames: list[bool]) -> tuple[Any, Any, Any]:
        from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode
        from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy
        from tests.test_a_stopped_controller_moves_no_jaws import _Calculator, _Frames

        class _Emptying(_Frames):
            def acquire(self) -> Any:
                frame = super().acquire()
                if frames and frames.pop(0):
                    return frame
                from dataclasses import replace

                return replace(frame, segmentations=())

        arm = TaskArm(log)
        jaws = toggle(log, arm=arm)
        policy = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99,  # type: ignore[arg-type]
                                      require_steady_before_motion=True)
        service = AutonomousGraspService.from_components(
            arm=arm, calculator=_Calculator(), perception=_Emptying(), mode=GraspMode.EASY,  # type: ignore[arg-type]
            gripper=jaws, policy=policy, max_attempts=1)
        return service, arm, jaws

    def test_until_empty_on_the_real_service_changes_do0_twice_per_part_and_ends_at_home(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        service, arm, jaws = self._service(log, [True, True, False, False])
        hooks = RecordingHooks()

        report = run_task(service, TaskPlan(object="", place=PlaceAt(pose="drop_left"), scope="until_empty"),
                          hooks=hooks, poses=POSES)

        self.assertIs(TaskStop.NOTHING_LEFT, report.stop, report.sentence)
        self.assertEqual((2, 4), (report.parts_placed, report.picks))
        self.assertEqual(4, do0_changes(log))
        self.assertIs(False, jaws.jaws_closed)
        self.assertEqual(HOME, arm.get_tcp_pose())
        self.assertEqual(("home",), motions(log)[-1])
        self.assertTrue(report.last_report is not None and not report.last_report.succeeded)
        self.assertEqual("", service.stopped_where_the_arm_stands)

    def test_a_toggle_switched_by_hand_as_the_pick_starts_asks_nobody_and_stops_where_the_arm_stands(self) -> None:
        """The pendant switches DO0 after the task's own check and before the pick's: the pick would ask where the jaws
        stand, and on a task's thread nobody is asked, so the pick refuses and the task stops where the arm stands."""
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop, run_task
        from tests._task_fakes import POSES

        log: list[Any] = []
        service, arm, jaws = self._service(log, [True])
        asked: list[str] = []

        def ask(question: str) -> str:
            asked.append(question)
            return "open"

        jaws._ask = ask  # noqa: SLF001 (a person at the terminal who would answer)

        def pendant(name: str, _data: Any) -> None:
            if name == "task.part_started":
                jaws._io.do[0] = not jaws._io.do[0]  # noqa: SLF001 (the pendant switched DO0)

        report = run_task(service, TaskPlan(object="", place=PlaceAt(pose="drop_left"), scope="until_empty"),
                          hooks=RecordingHooks(on_event=pendant), poses=POSES)

        self.assertIs(TaskStop.HAND_NEEDS_PERSON, report.stop, report.sentence)
        self.assertEqual([], asked, "a task's pick asked a person where the jaws stand")
        self.assertEqual(0, do0_changes(log))
        self.assertEqual([], motions(log))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
