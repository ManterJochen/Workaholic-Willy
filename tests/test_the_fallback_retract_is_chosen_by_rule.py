"""A descriptor's fallback retract is chosen by rule, not by the order of the table's rows (customer chain lane C3f).

``read_arm_retract`` took the first row written for an arm. A table rebuilt with a customer hand whose name sorts before
the 2F-85 would have moved every arm descriptor's ``default_q``, and with it every evidence file's
``arm_descriptor_sha256`` (finding 27 of the audit of 2026-09-17).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import yaml

_ROW = "  - arm: ur5\n    hand: {hand}\n    plate_mm: {plate}\n    planner_margin_mm: 4.0\n    retract: [{v}, 0.0, 0.0, 0.0, 0.0, 0.0]\n    steps: [0]\n    anchor: [0.0]\n"


def _table(folder: Path, rows: list) -> Path:
    path = folder / "ur_retract.yaml"
    path.write_text("rule:\n  placement: +Y+X\n  planner_margin_mm: 4.0\nretracts:\n" + "".join(rows), encoding="utf-8")
    return path


class TheFallbackIsChosenByRuleTests(unittest.TestCase):
    def test_a_hand_sorted_before_the_2f85_does_not_become_the_fallback(self) -> None:
        from src.robot.safety.planning.robot.retract_table import read_arm_retract

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = _table(folder, [_ROW.format(hand="acme_2f", plate=0.0, v=9.0), _ROW.format(hand="robotiq_2f85", plate=0.0, v=1.0)])
        row = read_arm_retract(path, "ur5")
        self.assertEqual(row.retract[0], 1.0)
        self.assertEqual(sorted(row.hands), ["acme_2f", "robotiq_2f85"])

    def test_the_fallback_hand_is_the_hand_every_arm_bundle_carries(self) -> None:
        from src.robot.safety.planning.hand import ARM_BUNDLE_HAND
        from src.robot.safety.planning.robot.retract_table import FALLBACK_HAND

        self.assertEqual(FALLBACK_HAND, ARM_BUNDLE_HAND)

    def test_an_arm_with_no_fallback_row_refuses_and_names_the_command(self) -> None:
        from src.robot.safety.planning.robot.retract_table import RetractMissing, read_arm_retract

        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = _table(folder, [_ROW.format(hand="robotiq_hande", plate=20.0, v=2.0)])
        with self.assertRaises(RetractMissing) as caught:
            read_arm_retract(path, "ur5")
        self.assertIn("choose_ur_retract.py ur5 --hand robotiq_2f85", str(caught.exception))

    def test_every_committed_arm_keeps_the_fallback_its_descriptor_was_built_with(self) -> None:
        """⭐ THE CONTROL: on the committed table the rule picks exactly the row the first-row reading picked, so no
        descriptor and no arm_descriptor_sha256 moves."""
        from src.robot.safety.planning.robot.retract_table import TABLE_PATH, read_arm_retract

        table = yaml.safe_load(Path(TABLE_PATH).read_text(encoding="utf-8"))
        for arm in sorted({row["arm"] for row in table["retracts"]}):
            with self.subTest(arm=arm):
                first = next(row for row in table["retracts"] if row["arm"] == arm)
                self.assertEqual(read_arm_retract(TABLE_PATH, arm).retract, [float(v) for v in first["retract"]])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
