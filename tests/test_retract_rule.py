"""The retract pose is chosen by a rule on the exact meshes, with every hand (UM5, B4).

A descriptor's retract used to be Isaac's Lula ``default_q``, copied verbatim. Measured in U2: on ur5 that pose keeps
9.5 mm to the Hand-E and 6.9 mm to the EGU-50, inside the guard's 10 mm margin, so the arm starts in a pose its own
guard refuses. The rule searches from the Lula pose outwards and takes the first candidate that clears EVERY hand the
registry holds, stays above the table, and lies inside the planner's envelope.

Deterministic by construction: a fixed candidate order, no random draw, first pass wins. That is what makes the
committed table reproducible on another box with the same engine, so this holds the order and the refusals rather than
the numbers, which only the box can measure.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_ANCHOR = [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]
#: The planner's envelope after cuRobo's clip: the elbow at +-(pi - 0.1), everything else at +-(2pi - 0.1).
_ENVELOPE = (
    tuple(-(limit - 0.1) for limit in (2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi)),
    tuple(limit - 0.1 for limit in (2 * math.pi, 2 * math.pi, math.pi, 2 * math.pi, 2 * math.pi, 2 * math.pi)),
)


def _rule():
    path = _ROOT / "scripts" / "curobo" / "_retract_rule.py"
    spec = importlib.util.spec_from_file_location("_retract_rule_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _Judge:
    """A hand under test: it answers a distance and a height for a pose, and counts what it was asked."""

    def __init__(self, label: str, distance, height=100.0) -> None:
        self.label = label
        self._distance = distance
        self._height = height
        self.asked = 0

    def nearest(self, joints):
        self.asked += 1
        value = self._distance(joints) if callable(self._distance) else self._distance
        return value, f"forearm|{self.label}"

    def lowest_z_mm(self, joints):
        return self._height(joints) if callable(self._height) else self._height


class TheCandidateOrderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _rule()

    def test_the_anchor_comes_first_and_the_family_is_closed(self) -> None:
        seen = list(self.rule.candidates(_ANCHOR, max_steps=2))
        self.assertEqual(seen[0][1], (0, 0, 0, 0, 0))
        self.assertEqual(seen[0][0], _ANCHOR)
        self.assertEqual(len(seen), 5 ** 5)
        self.assertEqual(len({steps for _, steps in seen}), len(seen))

    def test_the_full_family_is_the_declared_one(self) -> None:
        """13 steps per joint on five joints: the pan is fixed, because turning about the base moves no distance."""
        self.assertEqual(sum(1 for _ in self.rule.candidates(_ANCHOR)), 13 ** 5)

    def test_one_joint_at_a_time_comes_first(self) -> None:
        """The property the search rests on: what opens a pair is usually ONE joint turning far, and a candidate that
        moves four joints a little is both slower to reach and harder to read."""
        moved = [sum(1 for step in steps if step) for _, steps in self.rule.candidates(_ANCHOR, max_steps=2)]
        self.assertEqual(moved, sorted(moved), "the order must never put a two joint move before a one joint move")
        self.assertEqual(moved[:11], [0] + [1] * 10)

    def test_two_calls_give_one_order(self) -> None:
        first = [steps for _, steps in self.rule.candidates(_ANCHOR, max_steps=1)]
        second = [steps for _, steps in self.rule.candidates(_ANCHOR, max_steps=1)]
        self.assertEqual(first, second)
        self.assertEqual(first[1], (-1, 0, 0, 0, 0), "the smallest step comes first, lexicographically")

    def test_a_step_moves_one_joint_by_the_step_size(self) -> None:
        poses = {steps: pose for pose, steps in self.rule.candidates(_ANCHOR, step_rad=0.05, max_steps=1)}
        moved = poses[(0, 0, 0, 0, -1)]
        self.assertAlmostEqual(moved[5], _ANCHOR[5] - 0.05)
        self.assertEqual(moved[0], _ANCHOR[0], "the pan never moves")
        self.assertEqual(moved[:5], _ANCHOR[:5])


class TheRuleChoosesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _rule()

    def _choose(self, judges, **kwargs):
        return self.rule.choose(
            "ur5", _ANCHOR, judges, planner_envelope=_ENVELOPE, guard_margin_mm=10.0, reserve_mm=5.0,
            table_clearance_mm=10.0, **kwargs,
        )

    def test_an_anchor_that_clears_every_hand_is_kept(self) -> None:
        judges = [_Judge("a", 20.0), _Judge("b", 18.0), _Judge("c", 16.0)]
        chosen = self._choose(judges)
        self.assertEqual(chosen.steps, (0, 0, 0, 0, 0))
        self.assertEqual(chosen.retract, _ANCHOR)
        self.assertEqual(chosen.tried, 1)
        self.assertEqual({row["hand"]: row["clearance_mm"] for row in chosen.rows}, {"a": 20.0, "b": 18.0, "c": 16.0})

    def test_one_hand_below_the_margin_moves_the_arm_to_the_first_pose_that_clears_all_three(self) -> None:
        """⭐ The case the rule exists for: ur5 reads 9.5 mm to the Hand-E at Isaac's pose. An implementation that
        judged only the first hand, or combined them with any(), would keep the anchor."""
        wrist = [_Judge("a", 20.0), _Judge("b", 18.0),
                 _Judge("c", lambda q: 9.5 if abs(q[5] - _ANCHOR[5]) < 1e-9 else 16.0)]
        chosen = self._choose(wrist)
        self.assertEqual(chosen.steps, (0, 0, 0, 0, -1))
        self.assertEqual(chosen.tried, 6, "the search stops at the first candidate that passes every judge")
        self.assertAlmostEqual(chosen.retract[5], _ANCHOR[5] - 0.05)

    def test_a_pose_the_planner_refuses_is_not_a_retract(self) -> None:
        """⭐ THE SECOND JUDGE, and the reason the rule needed one.

        The rule judged the exact meshes only, and a pose can clear them by 13.4 mm while the planner's own
        sphere model calls it a self collision: measured 2026-09-16, that was true of ur3 with the Hand-E at
        Isaac's pose, and of five other arm and hand pairs. A retract the planner refuses is not a retract:
        the sidecar never becomes ready, so the cell does not come up at all.

        The planner is asked in ONE batch over the whole candidate family, because asking it pose by pose
        means a round trip each, and because that is what its explain command is for.
        """
        asked: list = []

        def admits(poses):
            asked.append(len(poses))
            return [abs(pose[5] - _ANCHOR[5]) > 0.04 for pose in poses]

        chosen = self._choose([_Judge("a", 20.0)], planner_admits=admits)

        self.assertNotEqual(chosen.steps, (0, 0, 0, 0, 0), "the anchor was kept although the planner refuses it")
        self.assertGreater(abs(chosen.retract[5] - _ANCHOR[5]), 0.04)
        self.assertEqual(len(asked), 1, f"the planner was asked {len(asked)} times, not once")
        self.assertGreater(asked[0], 100, "the whole candidate family goes in one batch")

    def test_a_planner_that_refuses_everything_refuses_the_arm_by_name(self) -> None:
        """⭐ THE CONTROL. ur3 and ur5 with the EGU-50 are in exactly this state: 0 of 433 candidates. The rule
        has to say so rather than fall back to a pose the planner will not hold.
        """
        with self.assertRaises(self.rule.RetractRefused) as caught:
            self._choose([_Judge("a", 20.0)], planner_admits=lambda poses: [False] * len(poses))
        self.assertIn("planner", str(caught.exception))

    def test_without_a_planner_judge_the_rule_is_what_it_was(self) -> None:
        """The control on the control: every mesh-only case in this file still reads the same."""
        chosen = self._choose([_Judge("a", 20.0)])
        self.assertEqual(chosen.steps, (0, 0, 0, 0, 0))

    def test_a_pose_hanging_into_the_table_is_skipped(self) -> None:
        """Clearance alone is not enough: a retract that clears itself can still stand in the bench."""
        low = _Judge("a", 20.0, height=lambda q: 5.0 if abs(q[5] - _ANCHOR[5]) < 1e-9 else 40.0)
        chosen = self._choose([low])
        self.assertNotEqual(chosen.steps, (0, 0, 0, 0, 0))
        self.assertGreaterEqual(min(row["lowest_z_mm"] for row in chosen.rows), 10.0)

    def test_a_pose_outside_the_planner_envelope_is_skipped(self) -> None:
        """The elbow at UR's limit is where this bites: a retract the planner may not even hold is not a retract."""
        anchor = [0.0, -1.0, math.pi - 0.1 - 0.02, 0.0, 0.0, 0.0]
        judges = [_Judge("a", lambda q: 20.0 if q[2] < math.pi - 0.15 else 9.0)]
        chosen = self.rule.choose(
            "ur5", anchor, judges, planner_envelope=_ENVELOPE, guard_margin_mm=10.0, reserve_mm=5.0,
            table_clearance_mm=10.0,
        )
        self.assertLessEqual(chosen.retract[2], _ENVELOPE[1][2] - 0.05 + 1e-9)

    def test_a_search_that_finds_nothing_refuses_and_says_what_was_closest(self) -> None:
        judges = [_Judge("a", 20.0), _Judge("hande", 9.5)]
        with self.assertRaises(self.rule.RetractRefused) as ctx:
            self._choose(judges, max_steps=1)
        message = str(ctx.exception)
        for token in ("ur5", "hande", "9.5", "forearm|hande"):
            with self.subTest(token=token):
                self.assertIn(token, message)


class TheTableIsReadStrictlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rule = _rule()
        self.table = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "ur_retract.yaml"
        self.table.write_text(
            "rule:\n"
            "  placement: +Y+X\n"
            "  guard_margin_mm: 10.0\n"
            "  reserve_mm: 5.0\n"
            "  pairs_with_no_retract: []\n"
            "retracts:\n"
            "  - arm: ur5\n    hand: robotiq_2f85\n    plate_mm: 0.0\n    planner_margin_mm: 10.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, -0.05, 0.0]\n    steps: [0, 0, 0, -1, 0]\n"
            "    anchor: [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]\n"
            "  - arm: ur5\n    hand: robotiq_hande\n    plate_mm: 20.0\n    planner_margin_mm: 10.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, -0.75]\n    steps: [0, 0, 0, 0, -15]\n"
            "    anchor: [0.0, -1.0, 0.9, 0.0, 0.0, 0.0]\n",
            encoding="utf-8",
        )

    def test_a_judged_arm_hand_and_plate_reads_back_its_own_pose(self) -> None:
        """⭐ Each hand reads ITS pose. Measured 2026-09-16: ur3 needs one pose for the 2F-85 and another three
        steps away for the Hand-E, because the planner's sphere model refuses the first with that hand on.
        """
        self.assertEqual(self.rule.read_retract("ur5", "robotiq_2f85", 0.0, placement="+Y+X", table_path=self.table),
                         [0.0, -1.0, 0.9, 0.0, -0.05, 0.0])
        self.assertEqual(self.rule.read_retract("ur5", "robotiq_hande", 20.0, placement="+Y+X", table_path=self.table),
                         [0.0, -1.0, 0.9, 0.0, 0.0, -0.75])

    def test_a_margin_the_pair_was_not_judged_at_refuses(self) -> None:
        """⭐ THE CONTROL. A pose the planner admits while keeping 10 mm it can refuse while keeping 4: the
        margin is part of the question, so it is part of the key.
        """
        with self.assertRaises(self.rule.RetractMissing) as caught:
            self.rule.read_retract("ur5", "robotiq_2f85", 0.0, 4.0, placement="+Y+X", table_path=self.table)
        self.assertIn("10.0", str(caught.exception))

    def test_an_arm_a_hand_or_a_plate_the_table_never_judged_refuses(self) -> None:
        """⭐ THE CONTROL, three ways. A retract judged for another hand is not this cell's retract, and a silent
        fallback to Isaac's pose is exactly what UM5 removes."""
        cases = (("ur10", "robotiq_2f85", 0.0), ("ur5", "schunk_egu50", 0.0), ("ur5", "robotiq_hande", 15.0))
        for arm, hand, plate in cases:
            with self.subTest(arm=arm, hand=hand, plate=plate):
                with self.assertRaises(self.rule.RetractMissing) as ctx:
                    self.rule.read_retract(arm, hand, plate, placement="+Y+X", table_path=self.table)
                self.assertIn("choose_ur_retract.py", str(ctx.exception))


class TheRetractBelongsToThePlacementTests(unittest.TestCase):
    """UM lane S23. The same joints hold a +Z hand somewhere else than a +Y one, so a row judged at one placement is not
    a retract for the other, and the table says which placement each row was judged at."""

    def setUp(self) -> None:
        self.rule = _rule()
        self.table = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "ur_retract.yaml"
        self.table.write_text(
            "rule:\n"
            "  placement: +Y+X\n"
            "  pairs_with_no_retract:\n"
            "  - arm: ur5\n    hand: robotiq_hande\n    plate_mm: 20.0\n    planner_margin_mm: 4.0\n"
            "    placement: +Z+X\n    reason: 'the rule found no pose'\n"
            "retracts:\n"
            "  - arm: ur5\n    hand: robotiq_2f85\n    plate_mm: 0.0\n    planner_margin_mm: 4.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, -0.05, 0.0]\n"
            "  - arm: ur5\n    hand: robotiq_2f85\n    plate_mm: 0.0\n    planner_margin_mm: 4.0\n"
            "    placement: +Z+X\n    tool_rotation_xyzw: [0.0, 0.0, 0.0, 1.0]\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, 0.5]\n"
            "  - arm: ur5\n    hand: robotiq_hande\n    plate_mm: 20.0\n    planner_margin_mm: 4.0\n"
            "    retract: [0.0, -1.0, 0.9, 0.0, 0.0, -0.75]\n",
            encoding="utf-8",
        )

    def test_each_placement_reads_its_own_row(self) -> None:
        self.assertEqual(self.rule.read_retract("ur5", "robotiq_2f85", 0.0, 4.0, placement="+Y+X", table_path=self.table),
                         [0.0, -1.0, 0.9, 0.0, -0.05, 0.0])
        self.assertEqual(self.rule.read_retract("ur5", "robotiq_2f85", 0.0, 4.0, placement="+Z+X", table_path=self.table),
                         [0.0, -1.0, 0.9, 0.0, 0.0, 0.5])

    def test_a_placement_the_pair_was_not_judged_at_refuses_and_names_both(self) -> None:
        """⭐ THE CONTROL. Before S23 the reader ignored the placement, and a +Z cell started from the +Y pose."""
        with self.assertRaises(self.rule.RetractMissing) as caught:
            self.rule.read_retract("ur5", "robotiq_2f85", 0.0, 4.0, placement="+Z+Y", table_path=self.table)
        for part in ("+Y+X", "+Z+X", "+Z+Y", "--tool-rotation-xyzw"):
            self.assertIn(part, str(caught.exception))

    def test_a_refusal_counts_only_at_its_own_placement(self) -> None:
        with self.assertRaises(self.rule.RetractMissing) as caught:
            self.rule.read_retract("ur5", "robotiq_hande", 20.0, 4.0, placement="+Z+X", table_path=self.table)
        self.assertIn("found no retract", str(caught.exception))
        self.assertEqual(self.rule.read_retract("ur5", "robotiq_hande", 20.0, 4.0, placement="+Y+X", table_path=self.table),
                         [0.0, -1.0, 0.9, 0.0, 0.0, -0.75])

    def test_the_committed_table_says_which_placement_it_judged(self) -> None:
        import yaml

        from src.robot.safety.planning.robot.retract_table import TABLE_PATH, placement_of

        table = yaml.safe_load(Path(TABLE_PATH).read_text(encoding="utf-8"))
        self.assertEqual(table["rule"]["placement"], "+Y+X")
        self.assertTrue(all(placement_of(row, table) in ("+Y+X", "+Z+X") for row in table["retracts"]))
        self.assertEqual(len(self.rule.read_retract("ur5e", "robotiq_2f85", 0.0, 4.0, placement="+Y+X")), 6)


class OneRunGoesIntoTheTableNotOverItTests(unittest.TestCase):
    """``merge_judged``: a run at a real flange's +Z must never throw away the +Y rows the sim cell starts from."""

    @staticmethod
    def _row(arm: str, hand: str, plate: float, retract: float) -> dict:
        return {"arm": arm, "hand": hand, "plate_mm": plate, "planner_margin_mm": 4.0,
                "anchor": [0.0] * 6, "retract": [retract] * 6}

    def _base(self) -> dict:
        return {"rule": {"guard_margin_mm": 10.0, "placement": "+Y+X", "tool_rotation_xyzw": [-0.7, 0.0, 0.0, 0.7],
                         "engine": "coal", "planner_margin_mm": 4.0, "pairs_with_no_retract": []},
                "retracts": [self._row("ur3", "robotiq_2f85", 0.0, 1.0), self._row("ur5e", "robotiq_2f85", 0.0, 2.0),
                             self._row("ur5e", "robotiq_hande", 20.0, 3.0)]}

    def _run(self, placement: str, rows: list, *, refused: "list | None" = None, **rule: object) -> dict:
        rotation = [0.0, 0.0, 0.0, 1.0] if placement == "+Z+X" else [-0.7, 0.0, 0.0, 0.7]
        return {"rule": {"guard_margin_mm": 10.0, "placement": placement, "tool_rotation_xyzw": rotation,
                         "engine": "coal", "planner_margin_mm": 4.0, "pairs_with_no_retract": refused or [], **rule},
                "retracts": rows}

    def test_the_committed_table_merged_with_itself_is_the_same_bytes(self) -> None:
        """⭐ The merge must not reformat what it keeps: a full run at the table's own placement that finds the same
        rows writes the committed file again, byte for byte."""
        import yaml

        from src.robot.safety.planning.robot.retract_table import TABLE_PATH, merge_judged

        text = Path(TABLE_PATH).read_text(encoding="utf-8")
        table = yaml.safe_load(text)
        base = table["rule"]["placement"]
        # What a full run at the table's own placement hands the merge: its rows and refusals, naming no placement.
        run = {"rule": {**table["rule"], "pairs_with_no_retract": [row for row in table["rule"]["pairs_with_no_retract"]
                                                                   if not row.get("placement")]},
               "retracts": [row for row in table["retracts"] if not row.get("placement")]}
        merged = merge_judged(yaml.safe_load(text), run)
        self.assertEqual(yaml.safe_dump(merged, sort_keys=False, default_flow_style=False), text)
        self.assertTrue(any(row.get("placement") not in (None, base) for row in merged["retracts"]),
                        "the control needs a table that holds another placement, or it proves nothing about keeping one")

    def test_a_run_carrying_another_placement_is_refused(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        stray = {**self._row("ur5e", "robotiq_2f85", 0.0, 9.0), "placement": "+Z+X"}
        with self.assertRaises(ValueError) as caught:
            merge_judged(self._base(), self._run("+Y+X", [stray]))
        self.assertIn("one placement", str(caught.exception))

    def test_a_plus_z_run_keeps_every_plus_y_row_and_says_where_its_own_sit(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        base = self._base()
        merged = merge_judged(base, self._run("+Z+X", [self._row("ur5e", "robotiq_2f85", 0.0, 9.0)]))
        self.assertEqual(merged["rule"]["placement"], "+Y+X")
        self.assertEqual(merged["retracts"][:3], self._base()["retracts"])
        added = merged["retracts"][3]
        self.assertEqual(added["placement"], "+Z+X")
        self.assertEqual(added["tool_rotation_xyzw"], [0.0, 0.0, 0.0, 1.0])
        self.assertEqual(list(added)[:6], ["arm", "hand", "plate_mm", "planner_margin_mm", "placement", "tool_rotation_xyzw"])

    def test_a_second_plus_z_run_replaces_its_rows_rather_than_adding_them(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        once = merge_judged(self._base(), self._run("+Z+X", [self._row("ur5e", "robotiq_2f85", 0.0, 9.0)]))
        twice = merge_judged(once, self._run("+Z+X", [self._row("ur5e", "robotiq_2f85", 0.0, 8.0)]))
        self.assertEqual(len(twice["retracts"]), 4)
        self.assertEqual(twice["retracts"][3]["retract"], [8.0] * 6)

    def test_a_run_for_one_arm_keeps_the_other_arms_where_they_were(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        merged = merge_judged(self._base(), self._run("+Y+X", [self._row("ur5e", "robotiq_2f85", 0.0, 7.0),
                                                              self._row("ur5e", "robotiq_hande", 20.0, 3.0)]))
        self.assertEqual([row["retract"][0] for row in merged["retracts"]], [1.0, 7.0, 3.0])
        self.assertNotIn("placement", merged["retracts"][1])

    def test_rows_under_another_rule_are_not_kept_beside_new_ones(self) -> None:
        """⭐ THE CONTROL. A table is one rule; a +Z run at another guard margin would sit beside +Y rows it contradicts."""
        from src.robot.safety.planning.robot.retract_table import merge_judged

        with self.assertRaises(ValueError) as caught:
            merge_judged(self._base(), self._run("+Z+X", [], guard_margin_mm=8.0))
        self.assertIn("guard_margin_mm", str(caught.exception))

    def test_a_refusal_of_one_placement_does_not_erase_the_other(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        refused = [{"arm": "ur5e", "hand": "robotiq_hande", "plate_mm": 20.0, "planner_margin_mm": 4.0, "reason": "none"}]
        merged = merge_judged(self._base(), self._run("+Z+X", [], refused=refused))
        self.assertEqual(len(merged["retracts"]), 3)
        self.assertEqual(merged["rule"]["pairs_with_no_retract"][0]["placement"], "+Z+X")


class ARunReplacesOnlyWhatItJudgedTests(unittest.TestCase):
    """C3c. A run for one hand, one plate or one margin replaces exactly the pairs it judged, and a pair the planner
    could not be asked about is no judgement at all."""

    @staticmethod
    def _committed() -> dict:
        import yaml

        from src.robot.safety.planning.robot.retract_table import TABLE_PATH

        return yaml.safe_load(Path(TABLE_PATH).read_text(encoding="utf-8"))

    @classmethod
    def _run(cls, rows: list, *, refused: "list | None" = None, **rule: object) -> dict:
        base = dict(cls._committed()["rule"])
        base.update({"placement": "+Z+X", "tool_rotation_xyzw": [0.0, 0.0, 0.0, 1.0], "pairs_with_no_retract": refused or []})
        base.update(rule)
        return {"rule": base, "retracts": rows}

    @staticmethod
    def _row(hand: str, plate: float, margin: float, value: float) -> dict:
        return {"arm": "ur5e", "hand": hand, "plate_mm": plate, "planner_margin_mm": margin, "anchor": [0.0] * 6,
                "retract": [value] * 6}

    @staticmethod
    def _z_rows(table: dict, arm: str = "ur5e") -> list:
        return [row for row in table["retracts"] if row.get("placement") == "+Z+X" and row["arm"] == arm]

    def test_a_run_for_one_hand_keeps_every_other_hands_row_on_that_arm(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        committed = self._committed()
        merged = merge_judged(self._committed(), self._run([self._row("acme_2f", 0.0, 4.0, 7.0)]))
        self.assertEqual(self._z_rows(merged)[:len(self._z_rows(committed))], self._z_rows(committed))
        self.assertEqual(merged["retracts"][-1]["hand"], "acme_2f")
        self.assertEqual(merged["retracts"][:len(committed["retracts"])], committed["retracts"])

    def test_a_pair_the_planner_could_not_be_asked_keeps_its_committed_row(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        committed = self._committed()
        refusal = {"arm": "ur5e", "hand": "robotiq_2f85", "plate_mm": 0.0, "planner_margin_mm": 4.0,
                   "reason": "the planner could not be asked: no descriptor"}
        merged = merge_judged(self._committed(), self._run([], refused=[refusal]),
                              not_asked=[("ur5e", "robotiq_2f85", 0.0, 4.0)])
        self.assertEqual(merged["retracts"], committed["retracts"])
        self.assertEqual(merged["rule"]["pairs_with_no_retract"], committed["rule"]["pairs_with_no_retract"])

    def test_a_judged_refusal_does_replace_its_row(self) -> None:
        """⭐ THE CONTROL: a pair the rule really refused is a judgement, and the row it replaces goes."""
        from src.robot.safety.planning.robot.retract_table import merge_judged

        refusal = {"arm": "ur5e", "hand": "robotiq_2f85", "plate_mm": 0.0, "planner_margin_mm": 4.0,
                   "reason": "every candidate refused"}
        merged = merge_judged(self._committed(), self._run([], refused=[refusal]))
        self.assertNotIn("robotiq_2f85", [row["hand"] for row in self._z_rows(merged)])
        self.assertIn("every candidate refused", [row["reason"] for row in merged["rule"]["pairs_with_no_retract"]])

    def test_a_run_at_another_planner_margin_goes_in_beside(self) -> None:
        import yaml

        from src.robot.safety.planning.robot.retract_table import merge_judged, read_retract

        merged = merge_judged(self._committed(), self._run([self._row("robotiq_2f85", 0.0, 6.0, 6.0)], planner_margin_mm=6.0))
        self.assertEqual(merged["rule"]["planner_margin_mm"], 4.0)
        folder = Path(self.enterContext(__import__("tempfile").TemporaryDirectory()))
        table = folder / "ur_retract.yaml"
        table.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
        self.assertEqual(read_retract("ur5e", "robotiq_2f85", 0.0, 6.0, placement="+Z+X", table_path=table), [6.0] * 6)
        self.assertNotEqual(read_retract("ur5e", "robotiq_2f85", 0.0, 4.0, placement="+Z+X", table_path=table), [6.0] * 6)

    def test_a_run_at_another_plate_goes_in_beside_and_the_rule_lists_every_plate(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        committed = self._committed()
        merged = merge_judged(self._committed(), self._run([self._row("robotiq_hande", 35.0, 4.0, 3.5)],
                                                           plates_mm_judged_for_mounting_face_hands=[35.0]))
        plates = sorted(row["plate_mm"] for row in self._z_rows(merged) if row["hand"] == "robotiq_hande")
        self.assertEqual(plates, [0.0, 20.0, 35.0])
        self.assertEqual(merged["rule"]["plates_mm_judged_for_mounting_face_hands"], [0.0, 20.0, 35.0])
        self.assertEqual(len(merged["retracts"]), len(committed["retracts"]) + 1)

    def test_a_shared_rule_key_that_differs_names_both_values(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        with self.assertRaises(ValueError) as caught:
            merge_judged(self._committed(), self._run([self._row("acme_2f", 0.0, 4.0, 1.0)], engine="fcl"))
        self.assertIn("coal", str(caught.exception))
        self.assertIn("fcl", str(caught.exception))

    def test_a_not_asked_pair_the_run_recorded_nothing_for_is_refused(self) -> None:
        from src.robot.safety.planning.robot.retract_table import merge_judged

        with self.assertRaises(ValueError):
            merge_judged(self._committed(), self._run([]), not_asked=[("ur5e", "robotiq_2f85", 0.0, 4.0)])


if __name__ == "__main__":
    unittest.main()
