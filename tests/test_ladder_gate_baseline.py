"""The grasp-eval gate must be able to FIND the baseline that is committed for it.

MEASURED 2026-09-10: ``run_gate`` resolved ``GATE_BASELINE`` to ``datagen/eval/gate_baseline.json``
and the only committed baseline in the tree was ``datagen/grasps/gate_baseline.json``. A check run
therefore raised ``FileNotFoundError("no baseline at ...; run the gate once with --write-baseline")``
after paying for the whole evaluation, and ``--write-baseline`` wrote a SECOND file next to the
first, so a repo could carry two baselines and compare against neither.

The gate itself costs minutes and needs a rendered dataset, so what is pinned here is the part that
was actually broken: the path, and that exactly one baseline exists to point at.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from datagen.eval.ladder import GATE_BASELINE, GATE_CONFIGS, GATE_SCENE_STEP

_REPO_ROOT = Path(__file__).resolve().parents[1]


class GateBaselineIsReachableTests(unittest.TestCase):
    def test_the_constant_points_at_a_file_that_exists(self) -> None:
        """The failure this replaces was a FileNotFoundError raised AFTER the measurement ran."""
        self.assertTrue(
            GATE_BASELINE.is_file(),
            f"run_gate reads {GATE_BASELINE}, which is not on disk",
        )

    def test_there_is_exactly_one_gate_baseline_in_the_tree(self) -> None:
        """Two baselines is the state a mis-pointed constant produces, and the second one looks
        just as authoritative as the first."""
        found = sorted(
            p.relative_to(_REPO_ROOT).as_posix()
            for p in (_REPO_ROOT / "datagen").rglob("gate_baseline.json")
        )
        self.assertEqual(len(found), 1, f"expected one gate baseline, found {found}")
        self.assertEqual(found[0], GATE_BASELINE.relative_to(_REPO_ROOT).as_posix())

    def test_the_committed_baseline_matches_the_gate_contract(self) -> None:
        """``run_gate`` refuses a baseline taken with other rungs or another scene step. A file the
        gate would refuse is no more usable than a missing one."""
        payload = json.loads(GATE_BASELINE.read_text(encoding="utf-8"))
        self.assertEqual(payload.get("scene_step"), GATE_SCENE_STEP)
        self.assertEqual(payload.get("rungs"), list(GATE_CONFIGS))
        self.assertEqual(set(payload.get("rates", {})), set(GATE_CONFIGS))


if __name__ == "__main__":
    unittest.main()
