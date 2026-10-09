"""An "until empty" task ends without a long search: one empty pass over its looks, or one check look from home.

Measured on the owner's cell (2026-10-07 and 08): after its last part, a task waited for two empty picks, and each empty
pick drove all four looks: eight look moves and eight groundings, 90 to 115 s before ``nothing_left``. Two rules of the
owner's (2026-10-08) end it sooner, and neither adds a motion:

* **An empty pick counts the looks it perceived from** (its report's ``looks``), so one empty pass over four looks ends
  the task, ``once`` or ``until_empty``. A pick of one look, or one that names none (a fixed camera), counts one: such a
  cell still looks twice.
* **The check look.** Where the part placed was the only target its pick's first look counted (the report's
  ``telemetry['targets_by_look']``), the next pick looks from that first look alone, home, where the return left the
  arm, with no generated view (``multi_view=False``). Seeing nothing there, or only parts the task keeps out, ends the
  task at once; a part it sees is picked as any other; a part it sees and fails on counts no failure, and the pick after
  it looks from every look again. A detector that failed during it is still ``detector_failed``.

The doubles of ``tests/_task_fakes.py`` play each pick: the looks its report names and the targets its first look
counted, on the real arm and toggle.
"""

from __future__ import annotations

import unittest

from src.robot.core import JointPositions
from src.robot.execution.task import TaskOptions, TaskStop
from tests._task_fakes import Pick, do0_changes, motions, run

#: The owner's four looks: home, then three taught joints.
LOOKS = ("home", JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0),
         JointPositions.deg(-10.0, -90.0, -110.0, -60.0, 90.0, 0.0),
         JointPositions.deg(0.0, -80.0, -120.0, -60.0, 90.0, 0.0))
#: What a report names the four looks it perceived from.
FOUR = ("home", "look 1", "look 2", "look 3")


class AnEmptyPickCountsItsLooksTests(unittest.TestCase):
    def test_one_empty_pick_over_four_looks_ends_the_task(self) -> None:
        ran = run(["part", Pick("empty", looks=FOUR)], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((1, 2), (ran.report.parts_placed, ran.report.picks))
        [found] = ran.hooks.of("task.nothing_found")
        self.assertEqual((4, 4, False), (found["looks"], found["empty_in_a_row"], found["check_look"]))
        self.assertIn("4 looks in a row", ran.report.sentence)
        self.assertEqual(("home",), motions(ran.log)[-1])

    def test_an_empty_pick_over_one_look_still_needs_a_second(self) -> None:
        ran = run(["part", Pick("empty", looks=("home",)), Pick("empty", looks=("home",))], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.picks)
        self.assertEqual([(1, 1), (1, 2)], [(found["looks"], found["empty_in_a_row"])
                                            for found in ran.hooks.of("task.nothing_found")])

    def test_a_pick_that_names_no_look_counts_one(self) -> None:
        """A fixed camera's pick handed no look: the cell still looks twice, as it always did."""
        ran = run(["part", "empty", "empty"], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.picks)
        self.assertEqual([1, 1], [found["looks"] for found in ran.hooks.of("task.nothing_found")])

    def test_only_parts_the_task_keeps_out_over_four_looks_end_it(self) -> None:
        ran = run(["part", Pick("excluded", looks=FOUR)], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.picks)
        [found] = ran.hooks.of("task.nothing_found")
        self.assertTrue(found["only_excluded"])
        self.assertEqual(4, found["looks"])

    def test_a_once_task_that_sees_nothing_over_four_looks_ends_after_one_pass(self) -> None:
        ran = run([Pick("empty", looks=FOUR)], scope="once")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.picks)

    def test_a_failed_pick_between_two_empty_ones_leaves_the_empty_row_standing(self) -> None:
        ran = run([Pick("empty", looks=("home",)), "failed", Pick("empty", looks=("home",))], scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(3, ran.report.picks)


class TheCheckLookTests(unittest.TestCase):
    def _run(self, picks: list[object], **keywords: object) -> object:
        return run(picks, scope=str(keywords.pop("scope", "until_empty")), wrist=True, looks=LOOKS,  # type: ignore[arg-type]
                   **keywords)

    def test_after_the_counted_last_part_the_next_pick_looks_from_home_alone_and_seeing_nothing_ends_it(self) -> None:
        ran = self._run([Pick("part", looks=("home",), targets_first_look=1), Pick("empty", looks=("home",))])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((1, 2), (ran.report.parts_placed, ran.report.picks))
        first, check = ran.service.calls
        self.assertNotIn("multi_view", first)
        self.assertIs(False, check["multi_view"], "the check is a pick with multi-view off: home alone, no view")
        self.assertEqual(LOOKS, check["look"])
        [found] = ran.hooks.of("task.nothing_found")
        self.assertTrue(found["check_look"])
        self.assertEqual(1, found["looks"])
        self.assertIn("check look", ran.report.sentence)
        self.assertEqual(("home",), motions(ran.log)[-1], "a done end returns home first")
        self.assertEqual(2, do0_changes(ran.log), "the check adds no change of DO0")

    def test_a_check_that_sees_a_part_picks_it_and_the_pick_after_looks_from_every_look(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), Pick("part", targets_first_look=2),
                         Pick("empty", looks=FOUR)])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((2, 3), (ran.report.parts_placed, ran.report.picks))
        first, check, after = ran.service.calls
        self.assertIs(False, check["multi_view"])
        self.assertNotIn("multi_view", after)
        self.assertEqual(4, do0_changes(ran.log))

    def test_a_check_after_a_check_where_each_part_was_the_last_one_counted(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), Pick("part", targets_first_look=1), Pick("empty")])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual((2, 3), (ran.report.parts_placed, ran.report.picks))
        self.assertEqual([None, False, False], [call.get("multi_view") for call in ran.service.calls])

    def test_a_check_that_sees_a_part_and_fails_counts_no_failure(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), "failed", "failed", "failed", "failed"])

        self.assertIs(TaskStop.FAILED_IN_A_ROW, ran.report.stop, ran.report.sentence)
        self.assertEqual(5, ran.report.picks, "the check's failure was counted")
        self.assertEqual([None, False, None, None, None], [call.get("multi_view") for call in ran.service.calls])

    def test_a_check_that_sees_only_the_task_s_own_parts_ends_it(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), Pick("excluded", looks=("home",))])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.picks)

    def test_a_detector_that_failed_during_the_check_is_never_nothing_left(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), Pick("empty", detector_failed=True)])

        self.assertIs(TaskStop.DETECTOR_FAILED, ran.report.stop, ran.report.sentence)

    def test_a_part_counted_beside_others_is_followed_by_a_full_pick(self) -> None:
        ran = self._run([Pick("part", targets_first_look=2), Pick("empty", looks=FOUR)])

        self.assertNotIn("multi_view", ran.service.calls[1])
        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)

    def test_a_pick_that_counts_no_targets_is_never_followed_by_a_check(self) -> None:
        """A fixed camera's pick names no count: its task searches as before."""
        ran = self._run(["part", Pick("empty", looks=FOUR)])

        self.assertNotIn("multi_view", ran.service.calls[1])

    def test_the_switch_off_looks_from_every_look_after_the_last_part(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1), Pick("empty", looks=FOUR)],
                        options=TaskOptions(check_look=False))

        self.assertNotIn("multi_view", ran.service.calls[1])
        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertFalse(ran.hooks.of("task.nothing_found")[0]["check_look"])

    def test_a_once_task_takes_no_check_look(self) -> None:
        ran = self._run([Pick("part", targets_first_look=1)], scope="once")

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.picks)

    def test_the_defaults_switch_the_check_look_on(self) -> None:
        self.assertIs(True, TaskOptions().check_look)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
