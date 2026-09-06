"""Phase D: deformable-handling seam tests."""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping import (
    CablePCAStrategy,
    DeformableClass,
    DeformableHandlingDecision,
    DeformableHandlingStrategy,
    GraspCalculator,
    GraspFailureReason,
    RefuseDeformableStrategy,
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


class RefuseDeformableStrategyTests(unittest.TestCase):
    def test_implements_protocol(self) -> None:
        self.assertIsInstance(
            RefuseDeformableStrategy(), DeformableHandlingStrategy
        )

    def test_rigid_proceeds_silently(self) -> None:
        strat = RefuseDeformableStrategy()
        d = strat.handle(
            deformable_class=DeformableClass.RIGID, mask=np.ones((4, 4), bool)
        )
        self.assertTrue(d.proceed)
        self.assertEqual(d.reasons, ())
        self.assertEqual(d.telemetry, {})

    def test_unknown_proceeds_by_default(self) -> None:
        d = RefuseDeformableStrategy().handle(
            deformable_class=DeformableClass.UNKNOWN, mask=np.ones((4, 4), bool)
        )
        self.assertTrue(d.proceed)
        self.assertEqual(d.telemetry["deformable_class"], "unknown")

    def test_unknown_refused_when_strict(self) -> None:
        d = RefuseDeformableStrategy(refuse_unknown=True).handle(
            deformable_class=DeformableClass.UNKNOWN, mask=np.ones((4, 4), bool)
        )
        self.assertFalse(d.proceed)
        self.assertIn(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED, d.reasons)

    def test_cable_cloth_bag_refused(self) -> None:
        strat = RefuseDeformableStrategy()
        for cls in (
            DeformableClass.CABLE,
            DeformableClass.CLOTH,
            DeformableClass.BAG,
        ):
            d = strat.handle(deformable_class=cls, mask=np.ones((4, 4), bool))
            self.assertFalse(d.proceed, msg=cls)
            self.assertEqual(
                d.reasons, (GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,)
            )
            self.assertEqual(d.telemetry["deformable_class"], cls.value)
            self.assertEqual(d.telemetry["deformable_strategy"], "refuse")


class CablePCAStrategyTests(unittest.TestCase):
    def test_implements_protocol(self) -> None:
        self.assertIsInstance(CablePCAStrategy(), DeformableHandlingStrategy)

    def test_cable_stamps_pca_and_refuses(self) -> None:
        d = CablePCAStrategy().handle(
            deformable_class=DeformableClass.CABLE, mask=_cable_mask()
        )
        self.assertFalse(d.proceed)
        self.assertIn(GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED, d.reasons)
        self.assertIn("cable_pca_axis_px", d.telemetry)
        self.assertIn("cable_pca_midpoint_px", d.telemetry)
        # A long diagonal stripe must be highly elongated.
        self.assertGreater(d.telemetry["cable_pca_elongation"], 5.0)
        # Axis is a unit 2-vector.
        ax = d.telemetry["cable_pca_axis_px"]
        self.assertAlmostEqual(
            float(ax[0]) ** 2 + float(ax[1]) ** 2, 1.0, places=5
        )

    def test_degenerate_mask_skips_pca_fields(self) -> None:
        # A single-pixel mask -> PCA undefined -> still refused but no
        # cable_pca_* fields.
        m = np.zeros((10, 10), dtype=bool)
        m[5, 5] = True
        d = CablePCAStrategy().handle(
            deformable_class=DeformableClass.CABLE, mask=m
        )
        self.assertFalse(d.proceed)
        self.assertNotIn("cable_pca_axis_px", d.telemetry)

    def test_rigid_proceeds(self) -> None:
        d = CablePCAStrategy().handle(
            deformable_class=DeformableClass.RIGID, mask=np.ones((4, 4), bool)
        )
        self.assertTrue(d.proceed)

    def test_cloth_refused_without_pca(self) -> None:
        d = CablePCAStrategy().handle(
            deformable_class=DeformableClass.CLOTH, mask=np.ones((10, 10), bool)
        )
        self.assertFalse(d.proceed)
        self.assertNotIn("cable_pca_axis_px", d.telemetry)


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

    def test_refuse_strategy_short_circuits_cable(self) -> None:
        H, W = 240, 320
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(
            camera_matrix=K,
            deformable_strategy=RefuseDeformableStrategy(),
        )
        out = calc.compute(
            segmentation=_Seg(_cable_mask(H, W)),
            depth_map=depth,
            deformable_class=DeformableClass.CABLE,
        )
        self.assertEqual(out, [])
        self.assertIn(
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
            calc.last_failure_reasons,
        )
        self.assertEqual(calc.last_telemetry["deformable_class"], "cable")
        self.assertEqual(calc.last_telemetry["deformable_strategy"], "refuse")

    def test_refuse_strategy_passes_rigid_through(self) -> None:
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(
            camera_matrix=K,
            deformable_strategy=RefuseDeformableStrategy(),
        )
        calc.compute(
            segmentation=_Seg(target),
            depth_map=depth,
            deformable_class=DeformableClass.RIGID,
        )
        self.assertNotIn(
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
            calc.last_failure_reasons,
        )

    def test_missing_class_defaults_to_rigid(self) -> None:
        # With a strategy configured but no explicit class, behaviour
        # must equal class=RIGID (i.e. proceed silently).
        H, W = 240, 320
        target = _rect_mask(H, W, 100, 80, 180, 160)
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(
            camera_matrix=K,
            deformable_strategy=RefuseDeformableStrategy(),
        )
        calc.compute(segmentation=_Seg(target), depth_map=depth)
        self.assertNotIn(
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
            calc.last_failure_reasons,
        )

    def test_cable_pca_strategy_stamps_telemetry_and_refuses(self) -> None:
        H, W = 240, 320
        depth = np.full((H, W), 600.0, dtype=np.float32)
        K = np.array(
            [[600.0, 0.0, W / 2.0], [0.0, 600.0, H / 2.0], [0.0, 0.0, 1.0]]
        )
        calc = GraspCalculator(
            camera_matrix=K,
            deformable_strategy=CablePCAStrategy(),
        )
        out = calc.compute(
            segmentation=_Seg(_cable_mask(H, W)),
            depth_map=depth,
            deformable_class=DeformableClass.CABLE,
        )
        self.assertEqual(out, [])
        self.assertIn(
            GraspFailureReason.DEFORMABLE_ROUTING_REQUIRED,
            calc.last_failure_reasons,
        )
        self.assertIn("cable_pca_axis_px", calc.last_telemetry)
        self.assertIn("cable_pca_midpoint_px", calc.last_telemetry)
        self.assertGreater(calc.last_telemetry["cable_pca_elongation"], 5.0)

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
