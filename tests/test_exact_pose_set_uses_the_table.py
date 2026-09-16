"""The poses the exact judge measures start at the retract the descriptor carries (UM5, B4).

The judge's pose 0 used to be Isaac's Lula ``default_q``, which is also what the descriptor carried. Now the descriptor
carries the retract the rule chose, so the judge has to read the same table: a fidelity measurement whose first pose is
not the pose the planner starts from is measuring a robot nobody runs.

The anchor is appended as the LAST pose instead of replacing anything, so indices 1 to 1,482 stay exactly what the U2
baseline measured and the file still answers what the arm's old pose was worth.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
_TABLE = _ROOT / "src" / "robot" / "safety" / "planning" / "robot" / "ur_retract.yaml"


def _judge_module():
    path = _ROOT / "scripts" / "curobo" / "exact_self_collision_poses.py"
    spec = importlib.util.spec_from_file_location("exact_self_collision_poses_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(_ROOT / "scripts" / "curobo"))
    spec.loader.exec_module(module)
    return module


class ThePoseSetStartsAtTheCommittedRetractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.judge = _judge_module()
        self.table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))
        #: The row the arm's own fallback comes from: the first one written for that arm. Since 2026-09-16
        #: there is one row per arm, hand, plate and margin, and the pose a CELL plans from is its pair's,
        #: applied when the sidecar composes the hand. A fidelity measurement carries no hand, so it reads
        #: the fallback, and that is what these assertions are about.
        self.first: dict = {}
        for row in self.table["retracts"]:
            self.first.setdefault(row["arm"], row)

    def test_pose_zero_is_the_table_row_and_not_isaacs_pose(self) -> None:
        for arm, row in sorted(self.first.items()):
            with self.subTest(arm=arm):
                self.assertEqual([round(v, 6) for v in self.judge._retract(arm)],  # noqa: SLF001
                                 [round(float(v), 6) for v in row["retract"]])

    def test_the_two_arms_the_rule_moved_no_longer_start_at_the_anchor(self) -> None:
        """⭐ THE CONTROL: on an arm the rule moved, reading the table has to give a different pose than Isaac's."""
        moved = [arm for arm, row in sorted(self.first.items()) if any(row["steps"])]
        self.assertTrue(moved, "no arm moved, so this assertion can no longer fail")
        for arm in moved:
            with self.subTest(arm=arm):
                row = self.first[arm]
                self.assertNotEqual([round(v, 6) for v in self.judge._retract(arm)],  # noqa: SLF001
                                    [round(float(v), 6) for v in row["anchor"]])

    def test_the_anchor_is_appended_and_the_indices_in_between_do_not_move(self) -> None:
        arm = "ur5"
        poses, kinds = self.judge.judged_pose_set(arm, random_n=1000, seed=20260915)
        self.assertEqual(len(poses), 1 + 1000 + 481 + 1)
        self.assertEqual((kinds[0], kinds[-1]), ("retract", "anchor"))
        self.assertEqual([round(v, 6) for v in poses[-1]],
                         [round(float(v), 6) for v in self.first[arm]["anchor"]])
        drawn = np.random.default_rng(20260915).uniform(-np.pi, np.pi, size=(1000, 6))
        self.assertTrue(np.allclose(np.asarray(poses[1:1001]), drawn),
                        "the random block must be exactly the U2 baseline's, or nothing compares to it")

    def test_an_arm_with_no_row_refuses_rather_than_guessing(self) -> None:
        """⭐ THE CONTROL, rebuilt because it rotted by growth.

        It asked for ur16e, which had no row until B5 baked its bundle and the rule judged its retract on
        2026-09-16. An arm nobody has measured is now something to NAME rather than something to find lying about,
        so the case is made from a name no table will ever carry.
        """
        rule = sys.modules["exact_self_collision_poses_under_test"]
        with self.assertRaises(Exception) as ctx:  # noqa: B017 - the module maps it to SystemExit for the CLI
            rule._retract("ur42")  # noqa: SLF001
        self.assertIn("ur42", str(ctx.exception))

        # And the control on the control: an arm that IS in the table answers with its judged pose.
        self.assertEqual(len(rule._retract("ur5e")), 6)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
