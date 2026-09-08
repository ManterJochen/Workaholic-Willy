"""Tests for the dense-sampling heuristic and runtime budget (Phase M)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.robot.grasping.generation.calculator import (
    GraspCalculator,
    _mask_non_convexity,
)


# ---------------------------------------------------------------------------
# Mask shape helpers
# ---------------------------------------------------------------------------


def _convex_mask(shape: tuple[int, int] = (24, 24)) -> np.ndarray:
    """Simple rectangular mask -- solidity 1.0 (non-convexity 0.0)."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[7:17, 6:18] = 1
    return mask


def _u_shape_mask(shape: tuple[int, int] = (64, 64)) -> np.ndarray:
    """U-shape mask -- strongly non-convex."""
    mask = np.zeros(shape, dtype=np.uint8)
    mask[16:50, 16:24] = 1   # left leg
    mask[16:50, 40:48] = 1   # right leg
    mask[42:50, 16:48] = 1   # bottom bar
    return mask


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 12.0], [0.0, 200.0, 12.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _make_calc(**kwargs):
    defaults = dict(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
        camera_matrix=_camera_matrix(),
    )
    defaults.update(kwargs)
    return GraspCalculator(**defaults)


def _segmentation(mask: np.ndarray) -> SimpleNamespace:
    return SimpleNamespace(mask=mask, label="t")


def _depth(shape: tuple[int, int], value_mm: float = 1000.0) -> np.ndarray:
    return np.full(shape, value_mm, dtype=np.float64)


def _compute_kwargs() -> dict:
    return dict(pixel_to_mm=5.0, unit="mm")


# ---------------------------------------------------------------------------
# Non-convexity primitive
# ---------------------------------------------------------------------------


class MaskNonConvexityTests(unittest.TestCase):
    def test_convex_mask_yields_zero(self) -> None:
        nc = _mask_non_convexity(_convex_mask())
        self.assertIsNotNone(nc)
        self.assertLess(nc, 0.05)  # near-perfect solidity

    def test_u_shape_is_non_convex(self) -> None:
        nc = _mask_non_convexity(_u_shape_mask())
        self.assertIsNotNone(nc)
        self.assertGreater(nc, 0.3)

    def test_empty_mask_returns_none(self) -> None:
        self.assertIsNone(
            _mask_non_convexity(np.zeros((32, 32), dtype=np.uint8))
        )


# ---------------------------------------------------------------------------
# Heuristic: explicit overrides + auto signals
# ---------------------------------------------------------------------------


class DenseDecisionTests(unittest.TestCase):
    def test_explicit_true_is_respected(self) -> None:
        calc = _make_calc()
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=True,
            mask=_convex_mask(),
            other_object_masks=None,
            telemetry=telemetry,
        )
        self.assertTrue(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "explicit")

    def test_explicit_false_is_respected_even_with_clutter(self) -> None:
        calc = _make_calc()
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=False,
            mask=_u_shape_mask(),
            other_object_masks=[_convex_mask()],
            telemetry=telemetry,
        )
        self.assertFalse(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "explicit")

    def test_auto_enables_dense_on_clutter(self) -> None:
        calc = _make_calc()
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=None,
            mask=_convex_mask(),
            other_object_masks=[_convex_mask()],
            telemetry=telemetry,
        )
        self.assertTrue(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "clutter")

    def test_auto_enables_dense_on_non_convex_mask(self) -> None:
        calc = _make_calc(dense_non_convexity_threshold=0.15)
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=None,
            mask=_u_shape_mask(),
            other_object_masks=None,
            telemetry=telemetry,
        )
        self.assertTrue(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "non_convex_mask")

    def test_auto_skips_dense_for_simple_isolated_object(self) -> None:
        calc = _make_calc()
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=None,
            mask=_convex_mask(),
            other_object_masks=None,
            telemetry=telemetry,
        )
        self.assertFalse(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "default_silhouette")

    def test_cooldown_after_dense_overrun(self) -> None:
        calc = _make_calc()
        calc._dense_last_overran = True
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=None,
            mask=_u_shape_mask(),
            other_object_masks=[_convex_mask()],
            telemetry=telemetry,
        )
        self.assertFalse(decision)
        self.assertEqual(telemetry["dense_auto_reason"], "degraded_after_overrun")

    def test_cooldown_can_be_overridden_by_explicit_true(self) -> None:
        calc = _make_calc()
        calc._dense_last_overran = True
        telemetry: dict = {}
        decision = calc._decide_dense_sampling(
            dense_sampling=True,
            mask=_convex_mask(),
            other_object_masks=None,
            telemetry=telemetry,
        )
        self.assertTrue(decision)


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class ConstructorValidationTests(unittest.TestCase):
    def test_rejects_non_positive_budget(self) -> None:
        with self.assertRaises(ValueError):
            _make_calc(dense_runtime_budget_ms=0.0)
        with self.assertRaises(ValueError):
            _make_calc(dense_runtime_budget_ms=-1.0)

    def test_rejects_threshold_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            _make_calc(dense_non_convexity_threshold=-0.1)
        with self.assertRaises(ValueError):
            _make_calc(dense_non_convexity_threshold=1.5)


# ---------------------------------------------------------------------------
# Runtime budget end-to-end via compute()
# ---------------------------------------------------------------------------


class DenseRuntimeBudgetTests(unittest.TestCase):
    """Force a budget overrun via monkeypatching ``time.monotonic``."""

    def test_overrun_falls_back_and_flags_telemetry(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)

        # Each monotonic() call inside the dense block returns a value
        # 500 ms apart, forcing the overrun branch. A generator-based
        # side_effect keeps subsequent unrelated calls happy.
        clock = [0.0]

        def _fake_monotonic() -> float:
            t = clock[0]
            clock[0] += 0.5
            return t

        with patch("time.monotonic", side_effect=_fake_monotonic):
            calc.compute(
                _segmentation(mask), depth, dense_sampling=True, **_compute_kwargs()
            )
        self.assertTrue(calc.last_telemetry.get("dense_timeout"))
        self.assertTrue(calc._dense_last_overran)
        self.assertGreater(calc.last_telemetry["dense_runtime_ms"], 10.0)

    def test_within_budget_does_not_flag_timeout(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10_000.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        calc.compute(
            _segmentation(mask), depth, dense_sampling=True, **_compute_kwargs()
        )
        self.assertFalse(calc.last_telemetry.get("dense_timeout", False))
        self.assertFalse(calc._dense_last_overran)


# ---------------------------------------------------------------------------
# G1: telemetry contract guarantees
# ---------------------------------------------------------------------------


class DenseTelemetryContractTests(unittest.TestCase):
    """Every dense execution must emit the canonical telemetry keys."""

    _REQUIRED_KEYS_WHEN_DENSE_RAN = (
        "dense_decision",
        "dense_auto_reason",
        "dense_runtime_ms",
        "dense_timeout",
        "dense_fallback",
        "dense_sampling_points",
        "grasp_sampling_mode",
    )

    def test_keys_present_on_normal_dense_run(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10_000.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        calc.compute(
            _segmentation(mask), depth, dense_sampling=True, **_compute_kwargs()
        )
        for key in self._REQUIRED_KEYS_WHEN_DENSE_RAN:
            self.assertIn(key, calc.last_telemetry, f"missing key {key!r}")
        self.assertEqual(calc.last_telemetry["dense_fallback"], "none")

    def test_keys_present_on_forced_overrun(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        clock = [0.0]

        def _fake_monotonic() -> float:
            t = clock[0]
            clock[0] += 0.5
            return t

        with patch("time.monotonic", side_effect=_fake_monotonic):
            calc.compute(
                _segmentation(mask), depth, dense_sampling=True, **_compute_kwargs()
            )
        for key in self._REQUIRED_KEYS_WHEN_DENSE_RAN:
            self.assertIn(key, calc.last_telemetry, f"missing key {key!r}")
        self.assertEqual(
            calc.last_telemetry["dense_fallback"], "silhouette_after_overrun"
        )
        self.assertTrue(calc.last_telemetry["dense_timeout"])


# ---------------------------------------------------------------------------
# G5: confidence taxonomy marker
# ---------------------------------------------------------------------------


class ConfidenceTaxonomyTests(unittest.TestCase):
    """Every candidate must be tagged as a heuristic score, not a calibrated
    probability."""

    def test_geometry_first_candidates_marked_heuristic(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10_000.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        candidates = calc.compute(
            _segmentation(mask), depth, dense_sampling=False, **_compute_kwargs()
        )
        for cand in candidates:
            self.assertEqual(
                cand.metadata.get("confidence_kind"), "heuristic"
            )

    def test_dense_candidates_marked_heuristic(self) -> None:
        calc = _make_calc(dense_runtime_budget_ms=10_000.0)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        candidates = calc.compute(
            _segmentation(mask), depth, dense_sampling=True, **_compute_kwargs()
        )
        for cand in candidates:
            self.assertEqual(
                cand.metadata.get("confidence_kind"), "heuristic"
            )


if __name__ == "__main__":
    unittest.main()
