"""Every gate judges the same poses, by one rule (UM lane S13).

The exact judge, the fidelity probe, the hand link equivalence probe and the matrix gate each measure one half of the
same question. Two of them generating their own poses would compare measurements of different configurations, and the
difference would read as a finding about the geometry. So the rule lives once, in ``scripts/curobo/_matrix_gate.py``,
and this holds it: the arm's retract first, then seeded random draws, then the sweep over the two joints between wrist_1
and the hand.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_RETRACT = [0.0, -1.5707, 1.5707, -1.5707, -1.5707, 0.0]


def _gate():
    path = _ROOT / "scripts" / "curobo" / "_matrix_gate.py"
    spec = importlib.util.spec_from_file_location("_matrix_gate_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OnePoseSetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.gate = _gate()

    def test_the_retract_comes_first_and_the_counts_are_the_measured_ones(self) -> None:
        """1 plus 1,000 plus 481: the number every U2 measurement is quoted over."""
        poses, kinds = self.gate.pose_set(_RETRACT)
        self.assertEqual(len(poses), 1482)
        self.assertEqual(poses[0], _RETRACT)
        self.assertEqual(kinds[0], "retract")
        self.assertEqual([kinds.count(k) for k in ("retract", "random", "sweep")], [1, 1000, 481])
        self.assertEqual(len(kinds), len(poses))

    def test_one_seed_is_one_set(self) -> None:
        self.assertEqual(self.gate.pose_set(_RETRACT), self.gate.pose_set(_RETRACT))

    def test_another_seed_moves_the_random_poses_and_leaves_the_sweep(self) -> None:
        """⭐ THE CONTROL. The seed has to reach the random draws, and nothing else: a sweep that moved with the seed
        would mean two gates sweeping different joints."""
        first, kinds = self.gate.pose_set(_RETRACT)
        second, _ = self.gate.pose_set(_RETRACT, seed=1)
        random_first = [p for p, k in zip(first, kinds) if k == "random"]
        random_second = [p for p, k in zip(second, kinds) if k == "random"]
        sweep_first = [p for p, k in zip(first, kinds) if k == "sweep"]
        sweep_second = [p for p, k in zip(second, kinds) if k == "sweep"]
        self.assertNotEqual(random_first, random_second)
        self.assertEqual(sweep_first, sweep_second)
        self.assertEqual(first[0], second[0])

    def test_the_sweep_is_joint_5_and_joint_6_only(self) -> None:
        poses, kinds = self.gate.pose_set(_RETRACT)
        sweep = [p for p, k in zip(poses, kinds) if k == "sweep"]
        self.assertEqual({tuple(p[:4]) for p in sweep}, {(0.0, -1.57, 0.0, 0.0)})
        self.assertEqual(len({p[4] for p in sweep}), 37)
        self.assertEqual(len({p[5] for p in sweep}), 13)


class EveryGateAsksTheSameRuleTests(unittest.TestCase):
    """Source checks: the scripts run under two interpreters, and neither can be imported from the other."""

    def test_the_exact_judge_and_the_equivalence_probe_use_it(self) -> None:
        for name in ("exact_self_collision_poses.py", "probe_hand_link_equivalence.py"):
            with self.subTest(script=name):
                source = (_ROOT / "scripts" / "curobo" / name).read_text(encoding="utf-8")
                self.assertIn("from _matrix_gate import pose_set", source)
                self.assertNotIn("np.arange(-180, 181, 10)", source)


if __name__ == "__main__":
    unittest.main()
