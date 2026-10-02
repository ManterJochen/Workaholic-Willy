"""A task that meets a problem stops where it found the arm, and commands nothing after (OD 9, build plan 1.3.3).

The owner's rule: after a problem the robot moves on nobody's say but a person's, and never on its own. So a task's
problem stop sends the arm nowhere, a return home included, and switches no output: a person clears the cell, then
Restart's first motion is the planned move home (the console's half). What ends a task that way:

* ``halted``: "halt now" was pressed, or the arm's own halt latch is set; read before every motion and after every
  failed one, so a halt is never mistaken for a refused motion or a stopped controller;
* ``disconnected``: the cell is being taken down under the task; read first;
* ``controller_stopped``: a protective or emergency stop met a pick or a motion of the task;
* ``hand_needs_person``: a gripper that raised, or a toggle's count nobody can vouch for, or a release the gripper
  did not confirm;
* ``recovery_needs_person``: a pick's push stopped where the arm stands (``tests/test_a_task_moves_nothing_while_a_
  person_is_needed.py``);
* ``part_still_held``: a place refused before its release, so the part is still in the jaws; and a part the pick
  gripped that the planner declined to carry (``payload_model`` not ``planner_and_filter``), so every carry would be
  planned as if the hand were empty;
* ``return_failed``: the move to the return pose was refused;
* ``cell_fault``: a camera that could not vouch for the cell, a link that dropped;
* ``failed_in_a_row`` and ``detector_failed``: three failed picks, or a detector that failed where nothing was found.

Halt and Disconnect are read before every motion the task sends, not only by the arm: an arm with no halt latch of its
own (a sim or KUKA arm, a UR before its latch) sends whatever it is told, so a halt pressed between two motions of the
task is read by the task before the next one.

Each test reads the log of the arm's motions and the hand's outputs, so "nothing after" is what the arm and the hand
were told, not what the task says it did.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.geometry import Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.arm_capabilities import PayloadModel
from src.robot.core.errors import CameraWorldUnavailable
from tests._task_fakes import (
    PROTECTIVE,
    MeasuringHand,
    Pick,
    RecordingHooks,
    ScriptedLocator,
    TaskArm,
    bin_object,
    do0_changes,
    motions,
    run,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _camera_unavailable() -> CameraWorldUnavailable:
    return CameraWorldUnavailable(camera="wrist", verdict="stale", attempts=3,
                                  reason="every frame was older than the cell allows")


def _placing(hooks: RecordingHooks) -> bool:
    return any(name == "task.place_started" for name, _ in hooks.events)


def _after(log: list[Any], mark: int) -> list[Any]:
    return motions(log[mark:])


class HaltNowTests(unittest.TestCase):
    def test_a_halt_during_the_line_in_releases_nothing_and_sends_no_line_out_and_no_return(self) -> None:
        """The brake route latches the arm (L9) and the task's halt: the release that would follow the line in is
        refused by the latched arm's output, and the task sends nothing after."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()
        at: list[int] = []

        def brake(kind: str, index: int) -> None:
            if hooks.events and hooks.events[-1][0] == "task.place_started" and kind == "move" and not at:
                at.append(index)
            if at and index == at[0] + 1:
                hooks.halt = True
                arm.halted = "halt now was pressed"

        arm = TaskArm(log, on_motion=brake)
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.HALTED, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding, "the part was released or counted released after the halt")
        self.assertEqual(0, ran.report.parts_placed)
        self.assertEqual(1, do0_changes(log), "the jaws changed after the halt")
        place = motions(ran.after("task.place_started"))
        self.assertEqual(2, len(place), "a line out or a return was sent after the halt")
        self.assertNotIn("task.return_started", ran.names())

    def test_a_halt_while_the_pick_runs_ends_the_task_after_it_with_nothing_more_sent(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        ran = run([Pick("part", then=lambda: setattr(hooks, "halt", True)), "part"], scope="until_empty", hooks=hooks)

        self.assertIs(TaskStop.HALTED, ran.report.stop)
        self.assertEqual(1, ran.report.picks)
        self.assertTrue(ran.report.holding)
        self.assertNotIn("task.place_started", ran.names())
        self.assertEqual(1, do0_changes(ran.log))

    def test_the_services_cancel_check_reads_the_halt_and_the_disconnect_but_not_stop_after_this_part(self) -> None:
        hooks = RecordingHooks()
        ran = run(["part"], hooks=hooks)
        check = ran.service.cancel_seen[0]

        self.assertTrue(callable(check))
        self.assertFalse(check())
        hooks.stop = True
        self.assertFalse(check(), "stop after this part ended the pick in flight")
        hooks.halt = True
        self.assertTrue(check())
        hooks.halt, hooks.gone = False, "disconnecting"
        self.assertTrue(check())
        self.assertIsNone(ran.service.cancel_check, "the cancel check outlived the task")

    def test_a_pick_the_controller_stopped_while_halted_reads_halted(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        ran = run([Pick("controller", then=lambda: setattr(hooks, "halt", True))], hooks=hooks)

        self.assertIs(TaskStop.HALTED, ran.report.stop)


class TheControllerAndTheCellTests(unittest.TestCase):
    def test_a_stopped_controller_ends_the_task_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["part", "controller", "part"], scope="until_empty")

        self.assertIs(TaskStop.CONTROLLER_STOPPED, ran.report.stop)
        self.assertEqual(2, ran.report.picks)
        self.assertEqual(2, do0_changes(ran.log))
        self.assertEqual(1, len([m for m in motions(ran.log) if m == ("home",)]), "the arm was sent home again")

    def test_a_fault_of_the_cell_ends_the_task_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["fault"])

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop)
        self.assertIn("the camera stopped delivering", ran.report.sentence)
        self.assertEqual([], motions(ran.log))

    def test_a_cell_taken_down_during_a_pick_is_disconnected_whatever_the_pick_said(self) -> None:
        from src.robot.execution.task import TaskStop

        hooks = RecordingHooks()
        ran = run([Pick("fault", then=lambda: setattr(hooks, "gone", "the operator disconnected the cell"))],
                  hooks=hooks)

        self.assertIs(TaskStop.DISCONNECTED, ran.report.stop)
        self.assertIn("the operator disconnected the cell", ran.report.sentence)

    def test_a_gripper_fault_ends_the_task_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run(["gripper"])

        self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop)
        self.assertIn("the gripper raised", ran.report.sentence)
        self.assertEqual([], motions(ran.log))

    def test_a_detector_that_failed_where_nothing_was_found_is_not_nothing_left(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = run([Pick("empty", detector_failed=True), "empty"], scope="until_empty")

        self.assertIs(TaskStop.DETECTOR_FAILED, ran.report.stop)
        self.assertEqual(1, ran.report.picks)
        self.assertNotIn("task.return_started", ran.names())


class ThePlaceAndTheReturnTests(unittest.TestCase):
    def test_a_place_refused_before_its_release_leaves_the_part_held_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()

        def refuse_the_standoff(kind: str, index: int, target: Any) -> "MotionStatus | None":
            placing = bool(hooks.events) and hooks.events[-1][0] == "task.place_started"
            return MotionStatus.WORKSPACE_REJECTED if placing else None

        arm = TaskArm(log, refuse=refuse_the_standoff)
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(log))
        self.assertEqual([], motions(ran.after("task.place_failed")))
        self.assertEqual("motion_refused", ran.hooks.of("task.place_failed")[0]["outcome"])

    def test_a_release_the_gripper_does_not_confirm_leaves_the_arm_where_it_stands(self) -> None:
        """A hand that closes and opens by intent and still measures the part after it opened."""
        from src.robot.execution.task import TaskStop

        ran = run(["part"], jaws=MeasuringHand(sticks=True))

        self.assertIs(TaskStop.HAND_NEEDS_PERSON, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual("release_not_confirmed", ran.hooks.of("task.place_failed")[0]["outcome"])
        self.assertEqual([], motions(ran.after("task.place_failed")))

    def test_a_return_that_is_refused_ends_the_task_where_the_arm_stands_with_the_part_counted(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log, refuse=lambda kind, index, target: MotionStatus.SELF_COLLISION_REJECTED
                      if kind == "home" else None)
        ran = run(["part", "part"], arm=arm, scope="until_empty")

        self.assertIs(TaskStop.RETURN_FAILED, ran.report.stop)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertFalse(ran.report.holding)
        self.assertEqual(1, ran.report.picks)
        self.assertEqual([], motions(ran.after("task.return_failed")))
        self.assertEqual("self_collision_rejected", ran.hooks.of("task.return_failed")[0]["status"])

    def test_a_return_a_halt_refused_is_halted_not_a_failed_return(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()
        arm = TaskArm(log)

        def halt_on_return(name: str, _data: dict[str, Any]) -> None:
            if name == "task.return_started":
                arm.halted = "halt now was pressed"

        hooks.on_event = halt_on_return
        ran = run(["part"], arm=arm, hooks=hooks)

        self.assertIs(TaskStop.HALTED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)

    def test_a_return_refused_on_a_controller_that_stopped_reads_controller_stopped(self) -> None:
        """A motion that failed is read as what stopped it: the controller, asked after the refusal, is stopped."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []

        def refuse_home(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "home":
                arm._statuses = [PROTECTIVE]  # noqa: SLF001 (a protective stop met the move home)
                return MotionStatus.CONTROLLER_REJECTED
            return None

        arm = TaskArm(log, refuse=refuse_home)
        ran = run(["part"], arm=arm)

        self.assertIs(TaskStop.CONTROLLER_STOPPED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertFalse(ran.report.holding)
        self.assertEqual([], motions(ran.after("task.return_failed")))
        self.assertIn("PROTECTIVE_STOP", ran.report.sentence)

    def test_a_camera_that_could_not_vouch_on_the_return_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []

        def refuse_home(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "home":
                raise _camera_unavailable()
            return None

        ran = run(["part"], arm=TaskArm(log, refuse=refuse_home))

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertEqual([], motions(ran.after("task.return_failed")))
        self.assertEqual(2, do0_changes(log))

    def test_a_camera_that_could_not_vouch_on_the_line_out_counts_the_part_placed_and_sends_no_return(self) -> None:
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        hooks = RecordingHooks()
        placing: list[int] = []

        def refuse_the_line_out(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if _placing(hooks):
                placing.append(index)
                if len(placing) == 3:
                    raise _camera_unavailable()
            return None

        ran = run(["part"], arm=TaskArm(log, refuse=refuse_the_line_out), hooks=hooks)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertFalse(ran.report.holding)
        self.assertEqual({"outcome": "camera_world_unavailable", "no_sensor": True, "line_out_refused": True},
                         ran.hooks.of("task.placed")[0])
        self.assertEqual([], motions(ran.after("task.placed")))
        self.assertNotIn("task.return_started", ran.names())


class TheCarriedPartTests(unittest.TestCase):
    """Between the close and the release only the planner holds the part (build plan item 9). A task refuses to start
    on an arm whose configuration models no carried part; and after every pick it reads what the attach left in force,
    because the planner can still decline the part there (its sidecar answers no, the hand model is missing): a part
    nobody models is carried nowhere, not to the drop, not back where it was gripped."""

    def test_a_part_the_planner_declined_at_the_pick_is_carried_nowhere(self) -> None:
        """Red before: the task read ``payload_declined_reason`` once, at its start, and carried a part the planner
        declined to the drop and home, every motion planned as if the hand were empty."""
        from src.robot.execution.task import TaskStop

        for attaches in (PayloadModel.FILTER_ONLY, PayloadModel.NONE):
            with self.subTest(attaches=attaches.value):
                log: list[Any] = []
                ran = run(["part", "part"], arm=TaskArm(log, attaches=attaches), scope="until_empty")

                self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
                self.assertTrue(ran.report.holding)
                self.assertEqual((0, 1), (ran.report.parts_placed, ran.report.picks))
                self.assertEqual(1, do0_changes(log), "the jaws changed after the pick")
                carried = motions(log[log.index(("attach",)):])
                self.assertEqual(1, len(carried), f"the task moved the part after the pick's own lift: {carried}")
                self.assertNotIn("task.drop_planned", ran.names())
                self.assertNotIn("task.return_started", ran.names())
                self.assertIn("planner", ran.report.sentence)

    def test_a_part_the_planner_declined_is_not_carried_to_the_bins_look_either(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1}, attaches=PayloadModel.FILTER_ONLY)
        locator = ScriptedLocator({"blue bin": [bin_object()]}, arm=arm, log=log)
        ran = run(["part"], arm=arm, place=PlaceAt(camera="blue bin"), wrist=True, looks=(L1,), locators=[locator])

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertNotIn("task.carry_started", ran.names())
        self.assertEqual(1, len(motions(log[log.index(("attach",)):])))


class HaltBetweenTheMotionsTests(unittest.TestCase):
    """"Halt now" and a Disconnect pressed between two motions of the task, on an arm whose own halt latch is left open
    (an arm with no latch sends whatever it is told): the task reads them before its next motion and sends nothing
    more, an output included."""

    def _pressed(self, at: str, *, gone: bool, camera: bool = False) -> tuple[Any, int]:
        from src.robot.execution.task import PlaceAt

        log: list[Any] = []
        hooks = RecordingHooks()
        mark: list[int] = []

        def press(name: str, _data: dict[str, Any]) -> None:
            if name == at and not mark:
                mark.append(len(log))
                if gone:
                    hooks.gone = "the operator disconnected the cell"
                else:
                    hooks.halt = True

        hooks.on_event = press
        if camera:
            arm = TaskArm(log, fk_table={_key(L1): AT_L1})
            locator = ScriptedLocator({"blue bin": [bin_object()]}, arm=arm, log=log)
            ran = run(["part", "part"], arm=arm, hooks=hooks, scope="until_empty", place=PlaceAt(camera="blue bin"),
                      wrist=True, looks=(L1,), locators=[locator])
        else:
            ran = run(["part", "part"], arm=TaskArm(log), hooks=hooks, scope="until_empty")
        self.assertTrue(mark, f"{at} was never said")
        return ran, mark[0]

    def test_a_halt_or_a_disconnect_between_two_motions_ends_the_task_with_nothing_more_sent(self) -> None:
        """Pinned against a mutation: before these, deleting the read before the return or before the place left
        every test of the task green, and an arm with no latch of its own would have driven on."""
        from src.robot.execution.task import TaskStop

        for at, camera in (("task.drop_planned", False), ("task.placed", False), ("task.carry_started", True),
                           ("task.part_finished", False), ("task.target_checked", True)):
            for gone in (False, True):
                with self.subTest(at=at, gone=gone):
                    ran, mark = self._pressed(at, gone=gone, camera=camera)

                    self.assertIs(TaskStop.DISCONNECTED if gone else TaskStop.HALTED, ran.report.stop,
                                  ran.report.sentence)
                    self.assertEqual([], motions(ran.log[mark:]), "a motion was sent after the press")
                    self.assertEqual(0, do0_changes(ran.log[mark:]), "the jaws changed after the press")
                    self.assertEqual(1, len(ran.service.calls), "a pick started after the press")
                    self.assertFalse(ran.arm.halted, "the arm's own latch did the stopping")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
