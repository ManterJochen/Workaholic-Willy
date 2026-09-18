"""The retract chooser judges the hands and plates it is told to, and never dies on a hand it cannot judge (lane C3d).

Adding one customer hand re-judged every registry hand on the named arms (finding 8 of the audit of 2026-09-17), a hand
with a body and no map killed the whole run (finding 6), a plate other than 0 or 20 mm could not be judged at all
(finding 21), and a run for some arms wrote every other arm down as having no bundle (finding 29).
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import yaml

from tests._retract_chooser import TABLE, run

_Z = ["--tool-rotation-xyzw", "0", "0", "0", "1"]


def _table(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _key(row: dict) -> tuple:
    return (row["arm"], row["hand"], float(row["plate_mm"]), float(row["planner_margin_mm"]), row.get("placement"))


class OneHandOnePlateTests(unittest.TestCase):
    def test_one_hand_is_judged_and_every_other_row_stays(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_hande", *_Z])
        self.assertEqual(done.code, 0, done.out)
        self.assertEqual(sorted({hand for _, hand, _ in done.judges_built}), ["robotiq_hande"])
        before = {_key(row): row for row in _table(TABLE)["retracts"]}
        after = {_key(row): row for row in _table(done.table)["retracts"]}
        self.assertEqual(set(before), set(after))
        moved = sorted(key for key in before if before[key] != after[key])
        self.assertTrue(all(key[0] == "ur5e" and key[1] == "robotiq_hande" and key[4] == "+Z+X" for key in moved), moved)

    def test_a_named_plate_is_judged_for_a_mounting_face_hand(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_hande", "--plate-mm", "35", *_Z])
        self.assertEqual(done.code, 0, done.out)
        table = _table(done.table)
        plates = sorted(float(row["plate_mm"]) for row in table["retracts"]
                        if row["arm"] == "ur5e" and row["hand"] == "robotiq_hande" and row.get("placement") == "+Z+X")
        self.assertEqual(plates, [0.0, 20.0, 35.0])
        self.assertEqual(table["rule"]["plates_mm_judged_for_mounting_face_hands"], [0.0, 20.0, 35.0])

    def test_a_plate_on_a_flange_hand_is_refused_before_anything_is_judged(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_2f85", "--plate-mm", "20", *_Z])
        self.assertEqual(done.code, 2, done.out)
        self.assertIn("robotiq_2f85", done.out)
        self.assertIn("flange", done.out)
        self.assertEqual(done.judges_built, [])
        self.assertEqual(done.table.read_bytes(), TABLE.read_bytes())


class WhatCannotBeJudgedTests(unittest.TestCase):
    def test_a_hand_with_a_bundle_and_no_sphere_map_is_skipped_by_name(self) -> None:
        maps = Path(self.enterContext(tempfile.TemporaryDirectory()))
        source = TABLE.parent
        for name in ("robotiq_2f85_gripper_spheres.yml",):
            shutil.copy(source / name, maps / name)
        done = run(["ur5e", *_Z], maps=maps)
        self.assertEqual(done.code, 0, done.out)
        self.assertIn("robotiq_hande", done.out)
        self.assertIn("scripts/curobo/fit_cover_spheres.py", done.out)
        self.assertEqual(sorted({hand for _, hand, _ in done.judges_built}), ["robotiq_2f85"])
        after = {_key(row) for row in _table(done.table)["retracts"]}
        self.assertIn(("ur5e", "robotiq_hande", 20.0, 4.0, "+Z+X"), after)

    def test_a_named_hand_the_chooser_cannot_judge_is_refused_up_front(self) -> None:
        done = run(["ur5e", "--hand", "acme_2f", *_Z])
        self.assertEqual(done.code, 2, done.out)
        self.assertIn("robotiq_hande", done.out)
        self.assertEqual(done.judges_built, [])
        maps = Path(self.enterContext(tempfile.TemporaryDirectory()))
        done = run(["ur5e", "--hand", "robotiq_hande", *_Z], maps=maps)
        self.assertEqual(done.code, 2, done.out)
        self.assertIn("fit_cover_spheres.py --hand robotiq_hande --write", done.out)
        self.assertEqual(done.judges_built, [])

    def test_a_pair_the_planner_cannot_be_asked_about_keeps_its_row_through_the_chooser(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_2f85", *_Z], planner_refuses=(("ur5e", "robotiq_2f85"),))
        self.assertEqual(done.code, 0, done.out)
        before = [row for row in _table(TABLE)["retracts"] if _key(row) == ("ur5e", "robotiq_2f85", 0.0, 4.0, "+Z+X")]
        after = [row for row in _table(done.table)["retracts"] if _key(row) == ("ur5e", "robotiq_2f85", 0.0, 4.0, "+Z+X")]
        self.assertEqual(after, before)
        self.assertEqual(_table(done.table)["rule"]["pairs_with_no_retract"], _table(TABLE)["rule"]["pairs_with_no_retract"])


class TheRuleSaysWhatIsTrueTests(unittest.TestCase):
    def test_arms_without_a_bundle_is_read_from_the_files(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_2f85"])
        self.assertEqual(done.code, 0, done.out)
        self.assertEqual(_table(done.table)["rule"]["arms_without_a_bundle"], [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
