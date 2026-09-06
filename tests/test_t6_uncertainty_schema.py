"""Phase T6 — RED tests for the ``GraspingUncertaintyConfig`` schema.

Locked design decisions exercised:

* Q5=B: ``decision.auto_uncertainty_threshold`` is preserved (for
  T1 backward compatibility) but logically demoted; the new
  ``uncertainty.fail_closed_threshold`` wins **when**
  ``uncertainty.enabled=True``.
* Q6=A: ``apply_modes`` defaults to ``("auto", "dense_clutter",
  "dense_autonomous")``; ``"easy"`` is rejected.
* Per-channel weights default to ``1.0`` for the five always-produced
  signals and ``0.0`` for the two optional ones (``topology_risk``,
  ``semantic_confidence``).
* All flags default to OFF so a T5 ``robot.yaml`` (no
  ``uncertainty:`` block) keeps byte-identical behaviour.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot.robot_schema import (
    GraspingUncertaintyConfig,
    RobotGraspingConfig,
    UncertaintyChannelWeightsConfig,
)


class UncertaintyChannelWeightsConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        w = UncertaintyChannelWeightsConfig()
        self.assertEqual(w.depth_confidence, 1.0)
        self.assertEqual(w.mask_confidence, 1.0)
        self.assertEqual(w.occlusion_corridor_risk, 1.0)
        self.assertEqual(w.feasibility_margin, 1.0)
        self.assertEqual(w.verification_residual, 1.0)
        self.assertEqual(w.topology_risk, 0.0)
        self.assertEqual(w.semantic_confidence, 0.0)

    def test_negative_weights_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            UncertaintyChannelWeightsConfig(depth_confidence=-0.1)


class GraspingUncertaintyConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        cfg = GraspingUncertaintyConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.fail_closed_threshold, 0.4)  # mirrors legacy default
        self.assertEqual(
            cfg.apply_modes,
            ("auto", "dense_clutter", "dense_autonomous"),
        )
        self.assertIsNone(cfg.calibration_artifact_path)

    def test_threshold_bounds(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(fail_closed_threshold=-0.1)
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(fail_closed_threshold=1.5)

    def test_easy_excluded_from_apply_modes(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(apply_modes=("easy", "auto"))

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(apply_modes=("auto", "rocket"))

    def test_at_least_one_weight_positive_when_enabled(self) -> None:
        # If operator opts the layer on but zeros every weight, the
        # fused value is unconditionally zero --- a footgun. The
        # schema rejects that combination eagerly.
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(
                enabled=True,
                weights=UncertaintyChannelWeightsConfig(
                    depth_confidence=0.0,
                    mask_confidence=0.0,
                    occlusion_corridor_risk=0.0,
                    feasibility_margin=0.0,
                    verification_residual=0.0,
                    topology_risk=0.0,
                    semantic_confidence=0.0,
                ),
            )


class RobotGraspingConfigUncertaintyMountTests(unittest.TestCase):
    def test_default_mount(self) -> None:
        cfg = RobotGraspingConfig()
        self.assertIsInstance(cfg.uncertainty, GraspingUncertaintyConfig)
        self.assertFalse(cfg.uncertainty.enabled)

    def test_yaml_layer_round_trip(self) -> None:
        cfg = RobotGraspingConfig.model_validate(
            {
                "uncertainty": {
                    "enabled": True,
                    "fail_closed_threshold": 0.55,
                    "apply_modes": ["auto", "dense_clutter"],
                    "weights": {
                        "depth_confidence": 1.0,
                        "mask_confidence": 0.5,
                        "occlusion_corridor_risk": 0.5,
                        "feasibility_margin": 0.5,
                        "verification_residual": 1.5,
                        "topology_risk": 0.0,
                        "semantic_confidence": 0.0,
                    },
                }
            }
        )
        self.assertTrue(cfg.uncertainty.enabled)
        self.assertAlmostEqual(cfg.uncertainty.fail_closed_threshold, 0.55)
        self.assertEqual(cfg.uncertainty.weights.verification_residual, 1.5)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
