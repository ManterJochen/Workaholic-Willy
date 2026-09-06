"""Phase T4 inline synthetic clutter scenarios for the target selector.

Three small hand-built scenes prove the selector behaves correctly
on the canonical clutter topologies operators care about:

1. **3-object pyramid**: one front object adjacent to two back
   objects. The selector with mask_adjacency + depth enabled must
   choose the front object even when its local score is lower
   (within ``max_local_score_drop``), because removing it unblocks
   two downstream picks.
2. **Tower**: all three objects vertically stacked along the camera
   ray with the top object having the *highest* local score. The
   legacy ``argmax`` already chooses the right thing; the selector
   should not regress \u2014 it should reach the same chosen index
   with reason ``LOCAL_MAX`` (no swap needed).
3. **Side-by-side**: three coplanar objects with no overlap and no
   adjacency. Unlock scores must all be 0; the selector reduces to
   legacy ``argmax``.

These scenarios are intentionally small (32x32 masks, 3 candidates)
so they run in microseconds and stay readable. They complement the
unit tests in ``test_t4_target_selector.py``.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.loop.target_selector import (
    BlockerGraphConfig,
    OrderingReason,
    TargetCandidate,
    TargetOrderingConfig,
    select_target,
)


def _square_mask(h: int, w: int, x0: int, y0: int, side: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y0 : y0 + side, x0 : x0 + side] = True
    return m


def _candidate(
    *,
    idx: int,
    mask: np.ndarray,
    depth_mm: float,
    score: float,
) -> TargetCandidate:
    ys, xs = np.where(mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    return TargetCandidate(
        segmentation_index=idx,
        mask=mask,
        centroid_depth_mm=float(depth_mm),
        local_score=float(score),
        bbox_px=bbox,
    )


class PyramidSceneTests(unittest.TestCase):
    """Front object adjacent to two back ones \u2192 unlock swap wins."""

    def test_pyramid_selects_front_when_swap_within_guard(self) -> None:
        front = _candidate(
            idx=0,
            mask=_square_mask(32, 32, 12, 12, 6),
            depth_mm=300.0,
            score=0.55,
        )
        back_left = _candidate(
            idx=1,
            mask=_square_mask(32, 32, 9, 12, 4),
            depth_mm=600.0,
            score=0.6,
        )
        back_right = _candidate(
            idx=2,
            mask=_square_mask(32, 32, 19, 12, 4),
            depth_mm=600.0,
            score=0.58,
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.5,
            max_local_score_drop=0.2,
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                depth_only_enabled=True,
                adjacency_radius_px=3,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(
            candidates=(front, back_left, back_right), config=cfg
        )
        self.assertEqual(d.chosen_index, 0)
        self.assertEqual(d.reason, OrderingReason.UNLOCK_SWAP)


class TowerSceneTests(unittest.TestCase):
    """Stacked objects: top has highest local score \u2192 no swap needed."""

    def test_tower_keeps_top_local_winner(self) -> None:
        top = _candidate(
            idx=0,
            mask=_square_mask(32, 32, 12, 12, 4),
            depth_mm=300.0,
            score=0.85,
        )
        middle = _candidate(
            idx=1,
            mask=_square_mask(32, 32, 12, 12, 4),
            depth_mm=400.0,
            score=0.5,
        )
        bottom = _candidate(
            idx=2,
            mask=_square_mask(32, 32, 12, 12, 4),
            depth_mm=500.0,
            score=0.4,
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.5,
            max_local_score_drop=0.1,
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                depth_only_enabled=True,
                adjacency_radius_px=3,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(candidates=(top, middle, bottom), config=cfg)
        self.assertEqual(d.chosen_index, 0)
        # No swap needed; reason should be LOCAL_MAX even though the
        # blocker graph is fully enabled.
        self.assertEqual(d.reason, OrderingReason.LOCAL_MAX)


class SideBySideSceneTests(unittest.TestCase):
    """Coplanar, non-adjacent objects \u2192 zero unlock, legacy argmax."""

    def test_no_blocking_reduces_to_local_max(self) -> None:
        a = _candidate(
            idx=0,
            mask=_square_mask(32, 32, 2, 2, 4),
            depth_mm=500.0,
            score=0.5,
        )
        b = _candidate(
            idx=1,
            mask=_square_mask(32, 32, 14, 2, 4),
            depth_mm=500.0,
            score=0.7,
        )
        c = _candidate(
            idx=2,
            mask=_square_mask(32, 32, 26, 2, 4),
            depth_mm=500.0,
            score=0.6,
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.9,
            max_local_score_drop=0.5,
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                depth_only_enabled=True,
                adjacency_radius_px=3,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(candidates=(a, b, c), config=cfg)
        self.assertEqual(d.chosen_index, 1)
        self.assertEqual(d.reason, OrderingReason.LOCAL_MAX)
        for u in d.unlock_scores:
            self.assertEqual(u, 0.0)


if __name__ == "__main__":
    unittest.main()
