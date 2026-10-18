"""A part still standing where the last pick took it from is counted not placed, and picked again (the owner,
2026-10-12: until the URCap reports the grip).

The owner's Hand-E switches on one tool output and reports nothing: a grip that closes on nothing carries nothing to the
drop, the release is not measured, and the task counted the part placed. A task that follows its parts
(every cell built from a tree) reads the spot the last pick took its part from on the next first look: where it still
reads a part over the support (``KeptScene``, ``TAKEN_STILL_STANDS``), the last part was not taken.

* the kept scene says so in the words a task reads, where the taken part still stands where it stood;
* the task counts that part not placed, takes its number again for this pick and says the part's end once more, not
  placed; a look that says nothing of the kind, or a task that placed nothing yet, changes nothing.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

from src.robot.execution.task import TaskEvent, _Task
from src.robot.perception.kept_scene import TAKEN_STILL_STANDS
from tests.robot.perception.test_a_kept_scene_follows_only_what_stands_where_it_stood import kept_of, shoot, two_grey_cubes, why


class TheKeptSceneSaysTheTakenPartStillStandsTests(unittest.TestCase):
    def test_a_taken_cube_still_in_its_place_is_said_to_stand_there(self) -> None:
        scene = two_grey_cubes()
        kept = kept_of(shoot(scene))
        taken = kept.without(kept.parts[0].cloud_base_mm)
        assert taken is not None
        self.assertIn(TAKEN_STILL_STANDS, why(taken, shoot(scene)))
        # Gone from its place, the spot reads the bench and nothing is said.
        self.assertNotIn(TAKEN_STILL_STANDS, why(taken, shoot([scene[1]])) or "")


def _task(placed: int) -> Any:
    said: list[tuple[Any, str, dict[str, Any]]] = []
    task = types.SimpleNamespace(parts_placed=placed, part_number=placed + 1, picks=7, said=said)
    task._say = lambda event, line, **data: said.append((event, line, data))
    return task


def _report(follow_why: str) -> Any:
    return types.SimpleNamespace(telemetry={"follow_why": follow_why})


class TheTaskCountsThePartNotPlacedTests(unittest.TestCase):
    def test_the_last_part_placed_is_counted_not_placed_and_picked_again(self) -> None:
        task = _task(3)
        _Task._the_last_part_not_taken(task, _report(f"the taken part's spot reads 31.0 mm over the support (at most "
                                                     f"10): {TAKEN_STILL_STANDS}"))
        self.assertEqual((2, 3), (task.parts_placed, task.part_number))
        [(event, line, data)] = task.said
        self.assertIs(TaskEvent.PART_FINISHED, event)
        self.assertEqual({"part": 3, "placed": False, "taken_spot_stands": True, "duration_s": None}, data)
        self.assertIn("picked again", line)

    def test_a_look_that_says_nothing_of_it_or_a_task_that_placed_nothing_changes_nothing(self) -> None:
        for placed, follow_why in ((3, "2 part(s) followed, nothing new in depth"), (3, ""), (0, TAKEN_STILL_STANDS)):
            with self.subTest(placed=placed, follow_why=follow_why):
                task = _task(placed)
                _Task._the_last_part_not_taken(task, _report(follow_why))
                self.assertEqual((placed, placed + 1, []), (task.parts_placed, task.part_number, task.said))


if __name__ == "__main__":
    unittest.main()
