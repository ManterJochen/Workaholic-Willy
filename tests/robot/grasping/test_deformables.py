"""Phase D: deformable-handling seam tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping import (
    DeformableClass,
    DeformableHandlingDecision,
    GraspCalculator,
    GraspFailureReason,
)


def _rect_mask(H: int, W: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


def _cable_mask(H: int = 240, W: int = 320) -> np.ndarray:
    """A long, thin diagonal stripe -- a realistic cable silhouette."""
    m = np.zeros((H, W), dtype=bool)
    for i in range(-2, 3):
        ys = np.arange(40, 200)
        xs = (ys + 20 + i).clip(0, W - 1)
        m[ys, xs] = True
    return m


class _Seg:
    def __init__(self, mask: np.ndarray, label: str = "thing", conf: float = 0.9):
        self.mask = mask
        self.label = label
        self.confidence = conf


class DeformableClassEnumTests(unittest.TestCase):
    def test_taxonomy_values_are_stable(self) -> None:
        self.assertEqual(DeformableClass.RIGID.value, "rigid")
        self.assertEqual(DeformableClass.CABLE.value, "cable")
        self.assertEqual(DeformableClass.CLOTH.value, "cloth")
        self.assertEqual(DeformableClass.BAG.value, "bag")
        self.assertEqual(DeformableClass.UNKNOWN.value, "unknown")


class CalculatorDeformableGateTests(unittest.TestCase):
    """Confirm Phase D wiring is strictly additive."""

    def test_default_constructor_has_no_strategy(self) -> None:
        calc = GraspCalculator()
        self.assertIsNone(calc.deformable_strategy)

    def test_invalid_strategy_type_rejected(self) -> None:
        with self.assertRaises(TypeError):
            GraspCalculator(deformable_strategy="not-a-strategy")  # type: ignore[arg-type]

    def test_no_strategy_ignores_deformable_class(self) -> None:
        # No strategy + class=CABLE must NOT short-circuit and must not
        # add any deformable_* telemetry.
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(camera_matrix=K)
        calc.compute(
            segmentation=_Seg(target),
            depth_map=depth,
            deformable_class=DeformableClass.CABLE,
        )
        self.assertNotIn(
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
            calc.last_failure_reasons,
        )
        self.assertNotIn("deformable_class", calc.last_telemetry)


    def test_custom_strategy_protocol_compatible(self) -> None:
        class _AlwaysRefuse:
            def handle(self, *, deformable_class, mask, depth_map=None):
                return DeformableHandlingDecision(
                    proceed=False,
                    reasons=(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,),
                    telemetry={"deformable_strategy": "always_refuse"},
                )

        H, W = 60, 80
        target = _rect_mask(H, W, 10, 10, 50, 50)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(camera_matrix=K, deformable_strategy=_AlwaysRefuse())
        out = calc.compute(
            segmentation=_Seg(target),
            depth_map=depth,
            deformable_class=DeformableClass.RIGID,
        )
        self.assertEqual(out, [])
        self.assertEqual(
            calc.last_telemetry["deformable_strategy"], "always_refuse"
        )


if __name__ == "__main__":
    unittest.main()
