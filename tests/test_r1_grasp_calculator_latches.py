"""The calculator's one cross-call latch: ``camera_matrix=None`` warns exactly once in the calculator's life.

A calculator built with no intrinsics synthesizes them (focal = width / 2), which places nothing metric right, and
says so once; a duplicated lens helper would warn on every call. (The dense sampler's runtime cooldown, the other
latch this file pinned, left with that sampler on 2026-10-16.)
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping import GraspCalculator


def _camera_matrix() -> np.ndarray:
    return np.array([[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _box_seg(shape: tuple[int, int] = (24, 24)) -> SimpleNamespace:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return SimpleNamespace(mask=mask, label="box")


class SyntheticIntrinsicsWarnOnceTests(unittest.TestCase):
    """camera_matrix=None must warn exactly once across the calculator's lifetime (the warn-once latch)."""

    def test_warns_exactly_once_across_calls(self) -> None:
        calc = GraspCalculator(
            min_grip_width_mm=1.0, max_grip_width_mm=200.0, max_candidates=3,
            camera_matrix=None,  # no K -> synthesized intrinsics -> the loud-once warning
        )
        seg = _box_seg()
        depth = np.full((24, 24), 1000.0, dtype=np.float64)
        with self.assertLogs("GraspCalculator", level="WARNING") as cm:
            calc.compute(seg, depth, unit="mm")
            calc.compute(seg, depth, unit="mm")
        synth = [r for r in cm.records if "synthesizing intrinsics" in r.getMessage()]
        self.assertEqual(
            len(synth), 1,
            "the synthetic-intrinsics warning must fire exactly once across calls (the warn-once latch)",
        )


if __name__ == "__main__":
    unittest.main()
