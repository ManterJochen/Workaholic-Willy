"""Tests for the semantic / task-aware grasp policy (Phase G3)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping import SemanticDecision, SemanticPolicy
from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.generation.calculator import GraspCalculator


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _convex_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return mask


def _depth(shape, value_mm=1000.0) -> np.ndarray:
    return np.full(shape, value_mm, dtype=np.float64)


def _make_calc() -> GraspCalculator:
    return GraspCalculator(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
        camera_matrix=_camera_matrix(),
    )


def _segmentation(label: str | None, score: float | None) -> SimpleNamespace:
    return SimpleNamespace(mask=_convex_mask(), label=label, score=score)


# ---------------------------------------------------------------------------
# Pure policy decisions
# ---------------------------------------------------------------------------


class PolicyEvaluationTests(unittest.TestCase):
    def test_no_op_policy_is_noop(self) -> None:
        policy = SemanticPolicy()
        self.assertTrue(policy.is_noop)
        self.assertEqual(
            policy.evaluate("anything", 0.5), SemanticDecision(True, None)
        )

    def test_allow_list_accepts_listed_label(self) -> None:
        policy = SemanticPolicy(allow_labels=frozenset({"can"}))
        self.assertTrue(policy.evaluate("can", 0.9).accept)

    def test_allow_list_rejects_unlisted_label(self) -> None:
        policy = SemanticPolicy(allow_labels=frozenset({"can"}))
        decision = policy.evaluate("bottle", 0.9)
        self.assertFalse(decision.accept)
        self.assertEqual(decision.reason, "label_not_in_allow_list")

    def test_allow_list_rejects_missing_label(self) -> None:
        policy = SemanticPolicy(allow_labels=frozenset({"can"}))
        self.assertFalse(policy.evaluate(None, 0.9).accept)

    def test_deny_list_wins_over_allow(self) -> None:
        policy = SemanticPolicy(
            allow_labels=frozenset({"can"}), deny_labels=frozenset({"can"})
        )
        decision = policy.evaluate("can", 0.99)
        self.assertFalse(decision.accept)
        self.assertEqual(decision.reason, "label_in_deny_list")

    def test_confidence_threshold(self) -> None:
        policy = SemanticPolicy(min_label_confidence=0.5)
        self.assertTrue(policy.evaluate("any", 0.8).accept)
        d = policy.evaluate("any", 0.3)
        self.assertFalse(d.accept)
        self.assertEqual(d.reason, "label_confidence_below_threshold")

    def test_confidence_fail_closed_on_missing_score(self) -> None:
        policy = SemanticPolicy(min_label_confidence=0.5)
        self.assertFalse(policy.evaluate("any", None).accept)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SemanticPolicy(min_label_confidence=1.5)
        with self.assertRaises(ValueError):
            SemanticPolicy(min_label_confidence=-0.1)


# ---------------------------------------------------------------------------
# Calculator integration
# ---------------------------------------------------------------------------


class CalculatorSemanticGateTests(unittest.TestCase):
    def test_no_policy_path_unchanged(self) -> None:
        # No policy at all: telemetry must not gain semantic_decision.
        calc = _make_calc()
        calc.compute(
            _segmentation("bottle", 0.9), _depth((24, 24)), pixel_to_mm=5.0
        )
        self.assertNotIn("semantic_decision", calc.last_telemetry)
        self.assertNotIn(
            GraspFailureReason.SEMANTIC_REJECTED, calc.last_failure_reasons
        )

    def test_noop_policy_path_unchanged(self) -> None:
        # An empty policy is treated as if None had been passed.
        calc = _make_calc()
        calc.compute(
            _segmentation("bottle", 0.9),
            _depth((24, 24)),
            pixel_to_mm=5.0,
            semantic_policy=SemanticPolicy(),
        )
        self.assertNotIn("semantic_decision", calc.last_telemetry)

    def test_deny_list_rejects_with_typed_reason(self) -> None:
        calc = _make_calc()
        policy = SemanticPolicy(deny_labels=frozenset({"bottle"}))
        out = calc.compute(
            _segmentation("bottle", 0.9),
            _depth((24, 24)),
            pixel_to_mm=5.0,
            semantic_policy=policy,
        )
        self.assertEqual(out, [])
        self.assertIn(
            GraspFailureReason.SEMANTIC_REJECTED, calc.last_failure_reasons
        )
        self.assertEqual(calc.last_telemetry["semantic_decision"], "label_in_deny_list")

    def test_allow_list_passes_listed_label(self) -> None:
        calc = _make_calc()
        policy = SemanticPolicy(allow_labels=frozenset({"can"}))
        calc.compute(
            _segmentation("can", 0.9),
            _depth((24, 24)),
            pixel_to_mm=5.0,
            semantic_policy=policy,
        )
        self.assertEqual(calc.last_telemetry["semantic_decision"], "accepted")
        self.assertNotIn(
            GraspFailureReason.SEMANTIC_REJECTED, calc.last_failure_reasons
        )

    def test_confidence_gate_rejects_low_score(self) -> None:
        calc = _make_calc()
        policy = SemanticPolicy(min_label_confidence=0.7)
        out = calc.compute(
            _segmentation("can", 0.3),
            _depth((24, 24)),
            pixel_to_mm=5.0,
            semantic_policy=policy,
        )
        self.assertEqual(out, [])
        self.assertIn(
            GraspFailureReason.SEMANTIC_REJECTED, calc.last_failure_reasons
        )
        self.assertEqual(
            calc.last_telemetry["semantic_decision"],
            "label_confidence_below_threshold",
        )


if __name__ == "__main__":
    unittest.main()
