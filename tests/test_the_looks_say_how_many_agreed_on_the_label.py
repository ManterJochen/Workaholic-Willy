"""The account of a pick's looks says how many of them agreed on the part's label (owner's decision 4, 2026-09-30).

Looks that call the part by one label add to the confidence, and a look that calls it otherwise makes its grasp
uncertain (2026-09-29, addendum 7.3). Only the disagreement acts; the agreement is said: the pick loop's one INFO
account of a pick's looks, and the Locator's INFO account of the part after each look, name it as "label agreed in N of
M looks", M the looks whose views of the part make up its cloud and N those that call it what the judged look calls it.
Nothing a pick does changes with it.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.loop.pick_loop import PickOutcome
from src.robot.grasping.multiview.association import label_agreement_said
from tests._wrist_views import CUBE, Box
from tests.test_a_located_part_is_looked_at_from_every_look import (
    EAST,
    NORTH,
    WEST,
    _arm,
    _locator,
    _looked,
    _robot,
    _Wrist,
)
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import (
    LOOK_ELSEWHERE,
    LOOK_MINUS_X,
    LOOK_PLUS_X,
    LOOK_PLUS_Y,
    _Cell,
    _label,
)

_PICK_LOOP = "src.robot.grasping.loop.pick_loop"
_LOCATOR = "src.robot.perception.locator"
#: A part like the cube 400 mm along +y of it, which the look from elsewhere sees and the others do not.
_OTHER = Box((-20.0, -320.0, 0.0), (20.0, -280.0, 30.0), "part")


def _accounts(output: list[str]) -> list[str]:
    return [line for line in output if "label agreed in" in line]


class TheOneSentenceTests(unittest.TestCase):
    def test_the_looks_that_call_the_part_what_the_judged_look_calls_it_are_counted_among_all_that_saw_it(self) -> None:
        """Compared trimmed and in lower case; a look that gave no label is counted among the looks and agrees with
        nothing, and the judged look counts itself where it has a label."""
        self.assertEqual("label agreed in 1 of 1 look", label_agreement_said("part", []))
        self.assertEqual("label agreed in 3 of 4 looks", label_agreement_said(
            "Part ", [("a", "part"), ("b", " PART"), ("c", "bolt")]))
        self.assertEqual("label agreed in 1 of 2 looks", label_agreement_said("part", [("a", "")]))
        self.assertEqual("label agreed in 0 of 2 looks", label_agreement_said("", [("a", "part")]))


class ThePickLoopsAccountTests(unittest.TestCase):
    def test_a_pick_that_stops_at_a_safe_grasp_says_how_many_looks_agreed_on_the_label(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1, 2))

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        (account,) = _accounts(said.output)
        self.assertTrue(account.startswith(f"INFO:{_PICK_LOOP}:"), account)
        self.assertIn("label agreed in 2 of 2 looks", account)
        self.assertIn(_label(LOOK_MINUS_X), account, "the account names the look it stopped at")

    def test_a_look_that_calls_the_part_otherwise_is_counted_out_and_the_pick_goes_on_as_before(self) -> None:
        """The second look calls the part a bolt: uncertain, so the third look is visited (as it always was); it calls it
        a part, as the first did."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X, LOOK_PLUS_Y), grasps_on=(1, 2), labels={1: {"part": "bolt"}})

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        (account,) = _accounts(said.output)
        self.assertIn("label agreed in 2 of 3 looks", account)

    def test_a_pick_whose_looks_ranked_no_grasp_says_it_on_the_same_line(self) -> None:
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), grasps_on=())

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            cell.run()

        (account,) = _accounts(said.output)
        self.assertIn("no look ranked a grasp", account)
        self.assertIn("label agreed in 2 of 2 looks", account)

    def test_a_pick_handed_no_looks_gives_no_account(self) -> None:
        cell = _Cell(looks=())

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            cell.run()

        self.assertEqual([], _accounts(said.output))

    def test_a_look_that_did_not_see_the_part_is_not_counted_among_the_looks(self) -> None:
        """The second look sees only another part, 400 mm off: it is left out of the part's cloud, so it is not among
        the looks that saw the part. The third look's grasp is safe, on the views of the first and the third."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_ELSEWHERE, LOOK_MINUS_X), boxes=(CUBE, _OTHER), grasps_on=(2,))

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.EXECUTED, report.outcome)
        self.assertEqual(3, len(cell.looked.visited))
        (account,) = _accounts(said.output)
        self.assertIn("label agreed in 2 of 2 looks", account)

    def test_a_pick_whose_grasp_is_withheld_says_it_on_the_same_line(self) -> None:
        """Both jaw faces asked for, and neither look shows the face at jaw 1: the grasp is not gripped, and the one
        account says so, naming how the looks agreed."""
        cell = _Cell(looks=(LOOK_PLUS_X, LOOK_PLUS_Y), both_faces=True)

        with self.assertLogs(_PICK_LOOP, level="INFO") as said:
            report = cell.run()

        self.assertIs(PickOutcome.RESCANNED_EXHAUSTED, report.outcome)
        (account,) = _accounts(said.output)
        self.assertIn(f"the looks end on the grasp of look {_label(LOOK_PLUS_Y)}, which is not gripped", account)
        self.assertIn("label agreed in 2 of 2 looks", account)


class TheLocatorsAccountTests(unittest.TestCase):
    def test_the_account_of_the_part_after_each_look_says_how_many_looks_agreed_on_the_label(self) -> None:
        with self.assertLogs(_LOCATOR, level="INFO") as said:
            _looked(EAST, WEST, both_faces=True)

        east, west = _accounts(said.output)
        self.assertTrue(east.startswith(f"INFO:{_LOCATOR}:"), east)
        self.assertIn(f"look {EAST.label}:", east)
        self.assertIn("label agreed in 1 of 1 look", east)
        self.assertIn(f"look {WEST.label}:", west)
        self.assertIn("label agreed in 2 of 2 looks", west)

    def test_a_look_that_calls_the_part_otherwise_is_counted_out(self) -> None:
        built = _arm(EAST, WEST, NORTH)
        wrist = _Wrist(built, labels={1: {"part": "bolt"}})

        with self.assertLogs(_LOCATOR, level="INFO") as said:
            _locator(wrist).look_around("a part", [EAST.joints, WEST.joints, NORTH.joints], robot=_robot(built),
                                        both_faces=True)

        accounts = _accounts(said.output)
        self.assertIn("label agreed in 1 of 2 looks", accounts[1], "WEST calls the part a bolt")
        self.assertIn("label agreed in 2 of 3 looks", accounts[2], "NORTH calls it a part, as EAST did")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
