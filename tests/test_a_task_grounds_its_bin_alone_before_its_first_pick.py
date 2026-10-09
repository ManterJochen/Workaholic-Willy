"""Before its first pick a task grounds its bin alone, never the part (2026-10-08).

The survey used to ground the part's phrase too, after the bin's, at every look: one more grounding, 4.5 to 6 s on the
cell, for one chat line, "N parts seen". Nothing else read the count, it grounded one box where "each separate" grounds
every part, and on 10-08 it said 1 of 3 cubes. The pick grounds every part with its own phrase and its own frame and
says how many it saw, so the task's survey grounds the bin alone: ``task.target_found`` carries no count
(``parts_seen`` null) and its sentence says none. A program that asks ``survey`` for a count still gets one
(``object_phrase``, ``tests/test_a_task_finds_its_bin_before_its_first_pick.py``).
"""

from __future__ import annotations

import unittest
from typing import Any

from src.geometry import Pose
from src.robot.core import JointPositions
from tests._task_fakes import PART_XY, ScriptedLocator, TaskArm, bin_object, bin_points, run

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _parts(_tcp: Any) -> list[Any]:
    return [bin_object(bin_points(PART_XY, (40.0, 40.0), 40.0, inside=False), label="red cube", score=0.7)]


class TheSurveyGroundsTheBinAloneTests(unittest.TestCase):
    def _ran(self, *, wrist: bool) -> Any:
        from src.robot.execution.task import PlaceAt

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1})
        locator = ScriptedLocator({"blue bin": [bin_object()], "red cube": _parts, "each separate red cube": _parts},
                                  arm=arm, log=log, wrist=wrist, rig_id="wrist" if wrist else "overhead")
        ran = run(("part",), place=PlaceAt(camera="blue bin"), arm=arm, wrist=wrist, looks=(L1,), locators=[locator])
        ran.locator = locator  # type: ignore[attr-defined]
        return ran

    def test_the_survey_grounds_the_bins_phrase_and_never_the_parts(self) -> None:
        for wrist in (True, False):
            with self.subTest(wrist=wrist):
                ran = self._ran(wrist=wrist)

                before_the_pick = ran.log[:ran.log.index(("event", "task.target_found"))]
                self.assertEqual([("locate", "blue bin")], [entry for entry in before_the_pick if entry[0] == "locate"])
                self.assertNotIn("red cube", ran.locator.asked)

    def test_the_bin_found_says_no_count_of_parts(self) -> None:
        ran = self._ran(wrist=True)

        found = ran.hooks.of("task.target_found")[0]
        self.assertIn("parts_seen", found, "the event lost its key: the console reads it as null")
        self.assertIsNone(found["parts_seen"])
        said = ran.hooks.said[ran.names().index("task.target_found")]
        self.assertNotIn("seen outside it", said)
        self.assertNotIn("red cube", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
