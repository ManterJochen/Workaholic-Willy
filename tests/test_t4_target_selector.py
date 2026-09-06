"""Phase T4 target_selector pure-function tests.

These tests pin the *behavioural contract* of the clutter-aware
target selector landed in T4. They exercise only pure functions on
the new ``target_selector`` module; the orchestrator threading is
covered by ``test_t4_pick_loop_ordering.py``.

Contracts asserted
------------------
1. ``OrderingMode`` is a ``str + Enum`` with stable lowercase values
   so telemetry consumers can compare against literal strings.
2. ``OrderingReason`` likewise carries stable strings explaining
   *why* a particular candidate won (or why a swap was blocked).
3. ``BlockerGraphConfig`` and ``TargetOrderingConfig`` are frozen,
   slotted, and validate their numeric inputs at construction.
4. ``TargetCandidate`` is a frozen container carrying everything the
   selector needs (segmentation index, mask, centroid depth, local
   score, bbox).
5. ``select_target`` is byte-identical to ``argmax(local_score)``
   when the config is disabled, regardless of unlock signals. This
   is the locked T3 \u2192 T4 compatibility guarantee.
6. The selector emits :class:`OrderingDecision` with parallel-indexed
   tuples (``local_scores``, ``unlock_scores``, ``priority_scores``)
   so log scrapers can correlate by ``segmentation_index`` without
   special-casing the chosen entry.
7. The hard-swap guard (``max_local_score_drop``) is enforced: a
   candidate with strictly lower local score than the legacy
   ``argmax`` choice can only win when its local score is within
   ``max_local_score_drop`` of that top local score. Otherwise the
   decision reverts to ``LOCAL_MAX`` with reason
   ``GUARD_BLOCKED_SWAP`` so operators can audit blocked swaps.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.loop.target_selector import (
    BlockerGraphConfig,
    OrderingDecision,
    OrderingMode,
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


class OrderingEnumTests(unittest.TestCase):
    def test_ordering_mode_string_values(self) -> None:
        self.assertEqual(OrderingMode.SINGLE_BEST.value, "single_best")
        self.assertEqual(OrderingMode.CLUTTER_AWARE.value, "clutter_aware")
        self.assertEqual(OrderingMode.SINGLE_BEST, "single_best")

    def test_ordering_reason_string_values(self) -> None:
        # Locked telemetry strings; consumers compare against these.
        self.assertEqual(OrderingReason.LOCAL_MAX.value, "local_max")
        self.assertEqual(OrderingReason.UNLOCK_SWAP.value, "unlock_swap")
        self.assertEqual(
            OrderingReason.GUARD_BLOCKED_SWAP.value, "guard_blocked_swap"
        )
        self.assertEqual(OrderingReason.NO_CANDIDATES.value, "no_candidates")
        self.assertEqual(OrderingReason.DISABLED.value, "disabled")


class ConfigValidationTests(unittest.TestCase):
    def test_blocker_graph_defaults(self) -> None:
        cfg = BlockerGraphConfig()
        self.assertFalse(cfg.mask_adjacency_enabled)
        self.assertFalse(cfg.depth_only_enabled)
        self.assertFalse(cfg.corridor_overlap_enabled)
        self.assertEqual(cfg.adjacency_radius_px, 5)
        self.assertEqual(cfg.depth_tolerance_mm, 10.0)

    def test_blocker_graph_rejects_negative_radius(self) -> None:
        with self.assertRaises(ValueError):
            BlockerGraphConfig(adjacency_radius_px=-1)

    def test_blocker_graph_rejects_negative_depth_tol(self) -> None:
        with self.assertRaises(ValueError):
            BlockerGraphConfig(depth_tolerance_mm=-0.1)

    def test_ordering_config_defaults(self) -> None:
        cfg = TargetOrderingConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.unlock_weight, 0.0)
        self.assertEqual(cfg.max_local_score_drop, 0.1)
        self.assertIsInstance(cfg.blocker_graph, BlockerGraphConfig)

    def test_ordering_config_rejects_bad_weights(self) -> None:
        with self.assertRaises(ValueError):
            TargetOrderingConfig(unlock_weight=-0.1)
        with self.assertRaises(ValueError):
            TargetOrderingConfig(max_local_score_drop=-0.1)
        with self.assertRaises(ValueError):
            TargetOrderingConfig(max_local_score_drop=1.5)

    def test_ordering_config_is_frozen(self) -> None:
        cfg = TargetOrderingConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.enabled = True  # type: ignore[misc]


class EmptyAndSingleCandidateTests(unittest.TestCase):
    def test_empty_returns_no_candidates_reason(self) -> None:
        d = select_target(candidates=(), config=TargetOrderingConfig())
        self.assertEqual(d.reason, OrderingReason.NO_CANDIDATES)
        self.assertIsNone(d.chosen_index)

    def test_single_candidate_always_wins(self) -> None:
        c = _candidate(
            idx=0,
            mask=_square_mask(32, 32, 8, 8, 4),
            depth_mm=500.0,
            score=0.6,
        )
        d = select_target(candidates=(c,), config=TargetOrderingConfig())
        self.assertEqual(d.chosen_index, 0)


class DisabledPathByteIdenticalTests(unittest.TestCase):
    """Locked T3 \u2192 T4 default: disabled config = argmax(local_score)."""

    def test_disabled_picks_argmax_local(self) -> None:
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 4, 4, 4), depth_mm=400.0, score=0.4
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 20, 20, 4), depth_mm=600.0, score=0.9
        )
        d = select_target(candidates=(a, b), config=TargetOrderingConfig())
        self.assertEqual(d.chosen_index, 1)
        self.assertEqual(d.reason, OrderingReason.DISABLED)
        self.assertEqual(d.mode, OrderingMode.SINGLE_BEST)

    def test_disabled_ignores_unlock_signals(self) -> None:
        # Even with all blocker signals on, disabled = legacy path.
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 4, 4, 8), depth_mm=300.0, score=0.4
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 4, 4, 8), depth_mm=600.0, score=0.9
        )
        cfg = TargetOrderingConfig(
            enabled=False,
            unlock_weight=1.0,  # ignored
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                depth_only_enabled=True,
            ),
        )
        d = select_target(candidates=(a, b), config=cfg)
        self.assertEqual(d.chosen_index, 1)
        self.assertEqual(d.reason, OrderingReason.DISABLED)


class EnabledZeroWeightTests(unittest.TestCase):
    def test_enabled_zero_weight_still_local_max(self) -> None:
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 4, 4, 4), depth_mm=400.0, score=0.4
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 20, 20, 4), depth_mm=600.0, score=0.9
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.0,
            blocker_graph=BlockerGraphConfig(mask_adjacency_enabled=True),
        )
        d = select_target(candidates=(a, b), config=cfg)
        self.assertEqual(d.chosen_index, 1)
        self.assertEqual(d.reason, OrderingReason.LOCAL_MAX)
        self.assertEqual(d.mode, OrderingMode.CLUTTER_AWARE)


class UnlockSwapTests(unittest.TestCase):
    """Hard-swap path: lower local but clear unlock value wins."""

    def test_unlock_swap_chooses_blocker(self) -> None:
        # 'a' sits in front of 'b' and 'c'; its mask is adjacent to
        # both. Picking 'a' unblocks two others.
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 12, 12, 6), depth_mm=300.0, score=0.5
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 10, 10, 4), depth_mm=600.0, score=0.55
        )
        c = _candidate(
            idx=2, mask=_square_mask(32, 32, 16, 16, 4), depth_mm=600.0, score=0.55
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.5,
            max_local_score_drop=0.2,
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                adjacency_radius_px=3,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(candidates=(a, b, c), config=cfg)
        self.assertEqual(d.chosen_index, 0)
        self.assertEqual(d.reason, OrderingReason.UNLOCK_SWAP)
        # Tuples are parallel-indexed by segmentation_index.
        self.assertEqual(len(d.local_scores), 3)
        self.assertEqual(len(d.unlock_scores), 3)
        self.assertEqual(len(d.priority_scores), 3)
        self.assertGreater(d.unlock_scores[0], d.unlock_scores[1])

    def test_guard_blocks_swap_when_local_too_low(self) -> None:
        # 'a' would win on unlock value but its local score is more
        # than max_local_score_drop below the top local score, so the
        # guard reverts to LOCAL_MAX and records GUARD_BLOCKED_SWAP.
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 12, 12, 6), depth_mm=300.0, score=0.2
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 10, 10, 4), depth_mm=600.0, score=0.9
        )
        c = _candidate(
            idx=2, mask=_square_mask(32, 32, 16, 16, 4), depth_mm=600.0, score=0.8
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=1.0,
            max_local_score_drop=0.1,
            blocker_graph=BlockerGraphConfig(
                mask_adjacency_enabled=True,
                adjacency_radius_px=3,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(candidates=(a, b, c), config=cfg)
        self.assertEqual(d.chosen_index, 1)
        self.assertEqual(d.reason, OrderingReason.GUARD_BLOCKED_SWAP)


class DepthOnlySignalTests(unittest.TestCase):
    def test_depth_only_unlock_with_bbox_overlap(self) -> None:
        # 'a' is in front of 'b' and bboxes overlap; should unblock.
        a = _candidate(
            idx=0, mask=_square_mask(32, 32, 8, 8, 8), depth_mm=300.0, score=0.6
        )
        b = _candidate(
            idx=1, mask=_square_mask(32, 32, 10, 10, 8), depth_mm=600.0, score=0.55
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.5,
            max_local_score_drop=0.2,
            blocker_graph=BlockerGraphConfig(
                depth_only_enabled=True,
                depth_tolerance_mm=50.0,
            ),
        )
        d = select_target(candidates=(a, b), config=cfg)
        self.assertGreater(d.unlock_scores[0], 0.0)
        self.assertEqual(d.unlock_scores[1], 0.0)


class DeterminismTests(unittest.TestCase):
    def test_same_inputs_same_choice(self) -> None:
        cands = (
            _candidate(
                idx=0,
                mask=_square_mask(32, 32, 12, 12, 6),
                depth_mm=300.0,
                score=0.5,
            ),
            _candidate(
                idx=1,
                mask=_square_mask(32, 32, 10, 10, 4),
                depth_mm=600.0,
                score=0.55,
            ),
        )
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.5,
            blocker_graph=BlockerGraphConfig(mask_adjacency_enabled=True),
        )
        a = select_target(candidates=cands, config=cfg)
        b = select_target(candidates=cands, config=cfg)
        self.assertEqual(a.chosen_index, b.chosen_index)
        self.assertEqual(a.priority_scores, b.priority_scores)


class OrderingDecisionTelemetryTests(unittest.TestCase):
    def test_to_dict_keys(self) -> None:
        d = OrderingDecision(
            chosen_index=1,
            local_scores=(0.4, 0.9),
            unlock_scores=(0.0, 0.0),
            priority_scores=(0.4, 0.9),
            mode=OrderingMode.SINGLE_BEST,
            reason=OrderingReason.DISABLED,
        )
        out = d.to_dict()
        self.assertEqual(out["chosen_index"], 1)
        self.assertEqual(out["mode"], "single_best")
        self.assertEqual(out["reason"], "disabled")
        self.assertEqual(out["local_scores"], [0.4, 0.9])
        self.assertEqual(out["unlock_scores"], [0.0, 0.0])
        self.assertEqual(out["priority_scores"], [0.4, 0.9])

    def test_decision_is_frozen(self) -> None:
        d = OrderingDecision(
            chosen_index=0,
            local_scores=(0.5,),
            unlock_scores=(0.0,),
            priority_scores=(0.5,),
            mode=OrderingMode.SINGLE_BEST,
            reason=OrderingReason.LOCAL_MAX,
        )
        with self.assertRaises((AttributeError, TypeError)):
            d.chosen_index = 9  # type: ignore[misc]


if __name__ == "__main__":
    unittest.main()
