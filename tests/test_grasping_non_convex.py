"""Tests for non-convex topology-risk handling (Phase G2)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.robot.grasping.types.feedback import GraspFailureReason
from src.robot.grasping.generation.calculator import GraspCalculator
from src.robot.grasping.scoring.topology_risk import (
    depth_continuity_risk,
    topology_risk_from_mask,
)


def _camera_matrix() -> np.ndarray:
    return np.array(
        [[200.0, 0.0, 32.0], [0.0, 200.0, 32.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _convex_mask(shape=(64, 64)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[20:44, 20:44] = 1
    return mask


def _ring_mask(shape=(64, 64)) -> np.ndarray:
    """A C-shape: ring outline with a 90 deg wedge removed.

    The outer contour is markedly non-convex (the wedge cut introduces a
    deep indent), which is what the topology-risk score must surface.
    A closed ring would have a near-circular outer contour and
    therefore near-zero ``1 - solidity`` despite the topological hole;
    that limitation is documented in the calculator README.
    """
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    cy, cx = shape[0] // 2, shape[1] // 2
    r = (yy - cy) ** 2 + (xx - cx) ** 2
    annulus = (r <= 22**2) & (r >= 10**2)
    # Remove a 90 deg wedge so the OUTER contour goes concave.
    wedge = (xx >= cx) & (yy >= cy)
    return (annulus & ~wedge).astype(np.uint8)


def _u_mask(shape=(64, 64)) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    mask[16:50, 16:24] = 1
    mask[16:50, 40:48] = 1
    mask[42:50, 16:48] = 1
    return mask


def _make_calc(**kwargs) -> GraspCalculator:
    defaults = dict(
        min_grip_width_mm=1.0,
        max_grip_width_mm=200.0,
        max_candidates=3,
        camera_matrix=_camera_matrix(),
    )
    defaults.update(kwargs)
    return GraspCalculator(**defaults)


def _depth(shape, value_mm=1000.0) -> np.ndarray:
    return np.full(shape, value_mm, dtype=np.float64)


# ---------------------------------------------------------------------------
# Pure topology-risk primitives
# ---------------------------------------------------------------------------


class TopologyRiskPrimitiveTests(unittest.TestCase):
    def test_convex_mask_is_low_risk(self) -> None:
        risk = topology_risk_from_mask(_convex_mask())
        self.assertIsNotNone(risk)
        self.assertLess(risk, 0.1)

    def test_ring_mask_is_high_risk(self) -> None:
        risk = topology_risk_from_mask(_ring_mask())
        self.assertIsNotNone(risk)
        self.assertGreater(risk, 0.3)

    def test_u_shape_is_high_risk(self) -> None:
        risk = topology_risk_from_mask(_u_mask())
        self.assertIsNotNone(risk)
        self.assertGreater(risk, 0.3)

    def test_empty_mask_returns_none(self) -> None:
        self.assertIsNone(topology_risk_from_mask(np.zeros((32, 32), dtype=np.uint8)))

    def test_depth_continuity_risk_flags_jumps(self) -> None:
        depth = np.full((32, 32), 500.0, dtype=np.float64)
        # Cliff at column 16 -- the right half is 200 mm farther.
        depth[:, 16:] = 700.0
        mask = np.ones((32, 32), dtype=np.uint8)
        risk = depth_continuity_risk(depth, mask, jump_mm=25.0)
        self.assertIsNotNone(risk)
        self.assertGreater(risk, 0.0)

    def test_depth_continuity_risk_smooth_scene(self) -> None:
        depth = np.full((32, 32), 500.0, dtype=np.float64)
        mask = np.ones((32, 32), dtype=np.uint8)
        risk = depth_continuity_risk(depth, mask, jump_mm=25.0)
        self.assertEqual(risk, 0.0)


# ---------------------------------------------------------------------------
# Calculator integration
# ---------------------------------------------------------------------------


class CalculatorTopologyGateTests(unittest.TestCase):
    def test_no_threshold_no_rejection(self) -> None:
        calc = _make_calc()
        mask = _ring_mask()
        depth = _depth(mask.shape)
        calc.compute(SimpleNamespace(mask=mask, label="ring"), depth, pixel_to_mm=5.0)
        # Risk is reported but no rejection happens.
        self.assertIn("topology_risk", calc.last_telemetry)
        reasons = calc.last_failure_reasons
        self.assertNotIn(GraspFailureReason.TOPOLOGY_RISK_REJECTED, reasons)

    def test_threshold_rejects_high_risk_mask(self) -> None:
        calc = _make_calc(topology_risk_threshold=0.2)
        mask = _ring_mask()
        depth = _depth(mask.shape)
        candidates = calc.compute(
            SimpleNamespace(mask=mask, label="ring"), depth, pixel_to_mm=5.0
        )
        self.assertEqual(candidates, [])
        self.assertIn(
            GraspFailureReason.TOPOLOGY_RISK_REJECTED, calc.last_failure_reasons
        )
        self.assertTrue(calc.last_telemetry.get("topology_risk_rejected"))
        self.assertEqual(calc.last_telemetry["topology_risk_threshold"], 0.2)

    def test_threshold_passes_convex_mask(self) -> None:
        calc = _make_calc(topology_risk_threshold=0.2)
        mask = _convex_mask()
        depth = _depth(mask.shape)
        calc.compute(
            SimpleNamespace(mask=mask, label="box"), depth, pixel_to_mm=5.0
        )
        # Convex masks must survive the gate -- behavior unchanged.
        self.assertNotIn(
            GraspFailureReason.TOPOLOGY_RISK_REJECTED, calc.last_failure_reasons
        )
        self.assertIn("topology_risk", calc.last_telemetry)
        self.assertLess(calc.last_telemetry["topology_risk"], 0.2)

    def test_invalid_threshold_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _make_calc(topology_risk_threshold=1.5)
        with self.assertRaises(ValueError):
            _make_calc(topology_risk_threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
