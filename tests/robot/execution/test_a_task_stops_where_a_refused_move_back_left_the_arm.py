"""A task stops for a person where its pick's move back to the look did not run, rather than pick again from there (S2 of
the stuck map, 2026-10-08).

The arm never stays where a refused last try left it: the pick drives it back to its look inside the pick's world. Where
that move back is refused too, the pick says where the arm stands (``PickReport.stands_at``, ``"standoff of grasp 3"``)
and the task's loop ends ``recovery_needs_person`` there, naming it, with nothing more commanded: picking again would
drive the first look from the standoff, judged on its frame alone (no depth over a bin), and a Home would meet the same
judgement. On the cell (2026-10-07) the task went on and refused three picks in a row from the standoff. A failed pick
whose arm got back to its look is a failed pick as before: the task picks again.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Any
from unittest import mock

from src.robot.execution.autonomous_grasp.report import AutonomousGraspOutcome
from src.robot.grasping.loop.pick_loop import PickOutcome
from tests._task_fakes import Pick, TaskArm, TaskService, _report, do0_changes, motions, run

_PLAY = TaskService._play


def _play(service: TaskService, pick: Pick) -> Any:
    """The scripted pick, and two more: ``stood`` failed with the arm left at the standoff of its third grasp, its move
    back refused; ``got_back`` failed with the arm back at its look."""
    if pick.kind not in ("stood", "got_back"):
        return _PLAY(service, pick)
    look = service.configured_looks[0] if service.configured_looks else None
    if look is not None and not isinstance(look, str):
        service.arm.move_to_joints(look)
    attempt = SimpleNamespace(reasons=("motion_plan_refused",), excluded=None, motion_status="workspace_rejected",
                              motion_message="the move back to the look comes within 2.1 mm of seen_03")
    return _report(AutonomousGraspOutcome.EXECUTION_FAILED, pick_report=SimpleNamespace(
        outcome=PickOutcome.EXECUTION_FAILED, attempts=(attempt,),
        stands_at="standoff of grasp 3" if pick.kind == "stood" else ""))


class ATaskStopsWhereTheArmWasLeftTests(unittest.TestCase):
    def test_a_pick_whose_move_back_was_refused_ends_the_task_for_a_person_naming_the_standoff(self) -> None:
        """Red before: the task counted a failed pick and picked again, its first look driven from the standoff."""
        from src.robot.execution.task import TaskStop

        log: list[Any] = []
        arm = TaskArm(log)
        at_the_stop: list[int] = []
        with mock.patch.object(TaskService, "_play", _play):
            ran = run([Pick("stood", then=lambda: at_the_stop.append(len(motions(log)))), "part"],
                      scope="until_empty", arm=arm)

        self.assertIs(TaskStop.RECOVERY_NEEDS_PERSON, ran.report.stop)
        self.assertEqual("problem", ran.report.stop.stop_class)
        self.assertIn("standoff of grasp 3", ran.report.sentence)
        self.assertIn("move back to its look did not run", ran.report.sentence)
        self.assertEqual(1, ran.report.picks, "a pick started from where the arm was left")
        self.assertEqual(at_the_stop, [len(motions(log))], "the arm moved after the pick that left it there")
        self.assertEqual(0, do0_changes(log))

    def test_a_task_of_one_part_stops_so_too(self) -> None:
        from src.robot.execution.task import TaskStop

        with mock.patch.object(TaskService, "_play", _play):
            ran = run(["stood"], scope="once")

        self.assertIs(TaskStop.RECOVERY_NEEDS_PERSON, ran.report.stop)

    def test_a_failed_pick_whose_arm_got_back_to_its_look_is_picked_again(self) -> None:
        """The control: the move back ran, so the arm stands at its look, and the task goes on as it always did."""
        from src.robot.execution.task import TaskStop

        with mock.patch.object(TaskService, "_play", _play):
            ran = run(["got_back", "part"], scope="once")

        self.assertIs(TaskStop.FINISHED, ran.report.stop)
        self.assertEqual(2, ran.report.picks)


if __name__ == "__main__":
    unittest.main()
