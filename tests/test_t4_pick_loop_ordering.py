"""Phase T4 orchestrator threading tests for clutter-aware ordering.

These tests pin the *behavioural* contract of plumbing
``TargetOrderingConfig`` into :class:`BinPickingOrchestrator`:

1. **Default path is byte-identical to T3.** When
   ``target_ordering`` is ``None`` (the default), or is supplied with
   ``enabled=False``, the orchestrator's
   ``_best_result_over_segmentations`` runs the legacy code path
   verbatim. Existing :file:`tests/test_pick_loop.py` regressions
   must continue to pass; this file only asserts the *additive*
   surface.
2. **Enabled path returns a new chosen index when unlock value
   compensates a small local-score deficit.** With the
   ``mask_adjacency_enabled`` signal on and the guard threshold
   ``max_local_score_drop=0.2``, the orchestrator should pick the
   front object (lower local score) over the back object (higher
   local score) when the front unblocks the back.
3. **Per-attempt telemetry carries an :class:`OrderingDecision`.**
   The new :attr:`PickAttempt.ordering_decision` field is populated
   when ordering is enabled, and remains ``None`` otherwise to keep
   log scrapers backward compatible.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.geometry import Frame, Pose
from src.robot.grasping.types.feedback import GraspResult
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from src.robot.grasping.loop.pick_loop import (
    BinPickingOrchestrator,
    PerceptionFrame,
    PickOutcome,
)
from src.robot.grasping.loop.target_selector import (
    BlockerGraphConfig,
    OrderingReason,
    TargetOrderingConfig,
)


# ---------------------------------------------------------------------------
# Test doubles (mirror those in test_pick_loop.py)
# ---------------------------------------------------------------------------


class _FakeArm:
    def __init__(self) -> None:
        self._tcp = Pose(
            position_mm=np.array([0.0, 0.0, 500.0]),
            quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
            frame=Frame.BASE,
            label="home",
        )
        self.move_to_calls: list[Pose] = []

    def get_tcp_pose(self) -> Pose:
        return self._tcp

    def move_to(self, pose: Pose, **_: object) -> None:
        self.move_to_calls.append(pose)
        if pose.frame == Frame.BASE:
            self._tcp = pose


class _FakePerception:
    def __init__(self, frame: PerceptionFrame) -> None:
        self.frame = frame

    def acquire(self) -> PerceptionFrame:
        return self.frame


def _grasp_point(score: float) -> GraspPoint:
    return GraspPoint(
        position=np.array([0.0, 0.0, 400.0]),
        approach=np.array([0.0, 0.0, 1.0]),
        axis=np.array([1.0, 0.0, 0.0]),
        grip_width_mm=40.0,
        score=score,
        frame=GraspFrame.BASE,
        label="t",
    )


def _success(score: float) -> GraspResult:
    return GraspResult(
        candidates=(_grasp_point(score),),
        reasons=(),
        top_score=score,
    )


class _IndexedCalculator:
    """Returns scripted results per segmentation index (round-robin).

    The orchestrator calls ``compute_result`` once per segmentation in
    a frame, so this stub maps ``self.calls`` modulo the number of
    segmentations to pick the right result.
    """

    def __init__(self, scores_per_seg: list[float]) -> None:
        self._scores = scores_per_seg
        self.calls = 0

    def compute_result(self, *_args: object, **_kwargs: object) -> GraspResult:
        idx = self.calls % len(self._scores)
        self.calls += 1
        return _success(self._scores[idx])


def _square_mask(h: int, w: int, x0: int, y0: int, side: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y0 : y0 + side, x0 : x0 + side] = 1
    return m


def _seg_with_mask(mask: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(mask=mask)


def _clutter_frame() -> PerceptionFrame:
    # Three objects: 0 front-centre (300mm), 1 back-left (600mm,
    # adjacent), 2 back-right (600mm, adjacent). Depth map encodes
    # the per-object depth at the mask region.
    h, w = 32, 32
    depth = np.full((h, w), 1000.0, dtype=np.float64)
    m0 = _square_mask(h, w, 12, 12, 6)
    m1 = _square_mask(h, w, 6, 12, 4)
    m2 = _square_mask(h, w, 22, 12, 4)
    depth[m0.astype(bool)] = 300.0
    depth[m1.astype(bool)] = 600.0
    depth[m2.astype(bool)] = 600.0
    intrinsics = np.array(
        [[400.0, 0.0, 16.0], [0.0, 400.0, 16.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return PerceptionFrame(
        depth_map=depth,
        intrinsics=intrinsics,
        segmentations=(
            _seg_with_mask(m0),
            _seg_with_mask(m1),
            _seg_with_mask(m2),
        ),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class OrchestratorDefaultByteIdenticalTests(unittest.TestCase):
    """``target_ordering=None`` \u21d2 legacy argmax path verbatim."""

    def test_default_picks_top_local_score(self) -> None:
        # Per-segmentation scores: 0.5, 0.6, 0.9 \u2192 legacy picks idx 2.
        calc = _IndexedCalculator([0.5, 0.6, 0.9])
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=calc,
            perception=_FakePerception(_clutter_frame()),
            max_attempts=1,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(report.attempts[0].target_index, 2)
        # ordering_decision must be None when ordering is disabled
        # (additive surface: existing callers see no change).
        self.assertIsNone(report.attempts[0].ordering_decision)


class OrchestratorEnabledThreadingTests(unittest.TestCase):
    """``target_ordering.enabled=True`` \u21d2 unlock-aware selection."""

    def test_enabled_swaps_to_unlock_candidate(self) -> None:
        # Local scores: front=0.55, back-left=0.6, back-right=0.58.
        # Legacy picks back-left (0.6). With mask-adjacency enabled
        # the front object unblocks the two back ones \u2192 swap to idx 0.
        calc = _IndexedCalculator([0.55, 0.6, 0.58])
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
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=calc,
            perception=_FakePerception(_clutter_frame()),
            max_attempts=1,
            target_ordering=cfg,
        )
        report = orch.run()
        self.assertEqual(report.outcome, PickOutcome.EXECUTED)
        self.assertEqual(report.attempts[0].target_index, 0)
        decision = report.attempts[0].ordering_decision
        self.assertIsNotNone(decision)
        assert decision is not None  # type narrowing
        self.assertEqual(decision.chosen_index, 0)
        self.assertEqual(decision.reason, OrderingReason.UNLOCK_SWAP)

    def test_enabled_zero_weight_still_local_max(self) -> None:
        # Same scene but weight=0 \u2192 reduce to local argmax.
        calc = _IndexedCalculator([0.55, 0.6, 0.58])
        cfg = TargetOrderingConfig(
            enabled=True,
            unlock_weight=0.0,
            blocker_graph=BlockerGraphConfig(mask_adjacency_enabled=True),
        )
        orch = BinPickingOrchestrator(
            arm=_FakeArm(),
            calculator=calc,
            perception=_FakePerception(_clutter_frame()),
            max_attempts=1,
            target_ordering=cfg,
        )
        report = orch.run()
        self.assertEqual(report.attempts[0].target_index, 1)
        decision = report.attempts[0].ordering_decision
        assert decision is not None
        self.assertEqual(decision.reason, OrderingReason.LOCAL_MAX)


if __name__ == "__main__":
    unittest.main()
