"""The retract chooser names every committed row it moved, and the evidence that row invalidates (lane C3e).

A retract is ``default_q`` inside the composed config a sidecar hashes, so a re-judged pair whose pose drifted refuses
every committed evidence file for it at the next planner start, on ``composed_sha256`` (finding 25 of the audit of
2026-09-17). The chooser says so when it writes, instead of the planner saying it at the cell.
"""

from __future__ import annotations

import copy
import unittest

import yaml

from tests._retract_chooser import TABLE, run

_Z = ["--tool-rotation-xyzw", "0", "0", "0", "1"]


class ARetractChangeIsNamedTests(unittest.TestCase):
    def test_a_moved_row_is_named_with_its_evidence_files(self) -> None:
        from src.robot.safety.planning.evidence import evidence_path
        from src.robot.safety.planning.robot.retract_table import retract_changes

        committed = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
        merged = copy.deepcopy(committed)
        (row,) = [r for r in merged["retracts"] if r["arm"] == "ur5e" and r["hand"] == "robotiq_2f85"
                  and r.get("placement") == "+Z+X"]
        row["retract"] = [v + 0.1 for v in row["retract"]]
        (change,) = retract_changes(committed, merged)
        said = change.render()
        expected = evidence_path(arm="ur5e", hand="robotiq_2f85", coupling_mm=0.0, approach="+Z", closing="+X",
                                 planner_margin_mm=4.0, attach_spheres=0).name
        for part in ("ur5e", "robotiq_2f85", "+Z+X", "4 mm", expected):
            self.assertIn(part, said)

    def test_an_unmoved_table_names_nothing(self) -> None:
        """⭐ THE CONTROL: the comparison does not report every row."""
        from src.robot.safety.planning.robot.retract_table import retract_changes

        committed = yaml.safe_load(TABLE.read_text(encoding="utf-8"))
        self.assertEqual(retract_changes(committed, copy.deepcopy(committed)), [])

    def test_the_chooser_prints_what_it_moved(self) -> None:
        done = run(["ur5e", "--hand", "robotiq_2f85", *_Z], moved=(("ur5e", "robotiq_2f85"),))
        self.assertEqual(done.code, 0, done.out)
        self.assertIn("ur5e_robotiq_2f85_c0mm_+Z+X_m4mm_a0.json", done.out)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
