"""Phase T7 — RED tests for the KPI threshold schema and YAML.

Q8=A locks the canonical threshold artefact at
``backend/config/data/robot/kpi_thresholds.yaml``, parsed by a typed
schema under ``backend.config.schema.robot``.

Contract:

* The YAML loads into :class:`KpiThresholdsConfig` without error.
* Numeric values in the YAML match Phase T's locked thresholds:

  - EASY: ``false_positive_grasp_rate <= 0.5%``, ``dead_loop_rate ==
    0.0%``, ``median_cycle_time_increase_pct <= 5.0``.
  - AUTO: ``dead_loop_rate <= 0.2%``.
  - DENSE modes: ``dead_loop_rate <= 1.0%``,
    ``false_positive_grasp_rate <= 2.0%``.
  - Soak: ``min_attempts >= 1000``, ``dead_loop_rate <= 0.5%``.

* Reject negative bounds and bounds outside ``[0, 1]`` where rates
  apply.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.config.schema.robot.kpi_schema import KpiThresholdsConfig


_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "data"
    / "robot"
    / "kpi_thresholds.yaml"
)


def _load_yaml() -> dict:
    with _THRESHOLDS_PATH.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise AssertionError(
            f"expected mapping at top of {_THRESHOLDS_PATH}, got {type(data)!r}"
        )
    return data


class ThresholdsYamlTests(unittest.TestCase):
    def test_yaml_loads_into_typed_schema(self) -> None:
        cfg = KpiThresholdsConfig.model_validate(_load_yaml())
        self.assertIsInstance(cfg, KpiThresholdsConfig)

    def test_easy_locked_values(self) -> None:
        cfg = KpiThresholdsConfig.model_validate(_load_yaml())
        self.assertAlmostEqual(cfg.easy.false_positive_grasp_rate_max, 0.005)
        self.assertEqual(cfg.easy.dead_loop_rate_max, 0.0)
        self.assertAlmostEqual(cfg.easy.median_cycle_time_increase_pct_max, 5.0)

    def test_auto_locked_values(self) -> None:
        cfg = KpiThresholdsConfig.model_validate(_load_yaml())
        self.assertAlmostEqual(cfg.auto.dead_loop_rate_max, 0.002)

    def test_dense_locked_values(self) -> None:
        cfg = KpiThresholdsConfig.model_validate(_load_yaml())
        self.assertAlmostEqual(cfg.dense.dead_loop_rate_max, 0.01)
        self.assertAlmostEqual(cfg.dense.false_positive_grasp_rate_max, 0.02)

    def test_soak_locked_values(self) -> None:
        cfg = KpiThresholdsConfig.model_validate(_load_yaml())
        self.assertGreaterEqual(cfg.soak.min_attempts, 1000)
        self.assertAlmostEqual(cfg.soak.dead_loop_rate_max, 0.005)


class ThresholdsSchemaValidationTests(unittest.TestCase):
    def test_rejects_rate_above_one(self) -> None:
        bad = _load_yaml()
        bad["easy"]["false_positive_grasp_rate_max"] = 1.5
        with self.assertRaises(ValidationError):
            KpiThresholdsConfig.model_validate(bad)

    def test_rejects_negative_rate(self) -> None:
        bad = _load_yaml()
        bad["dense"]["dead_loop_rate_max"] = -0.01
        with self.assertRaises(ValidationError):
            KpiThresholdsConfig.model_validate(bad)

    def test_rejects_zero_or_negative_min_attempts(self) -> None:
        bad = _load_yaml()
        bad["soak"]["min_attempts"] = 0
        with self.assertRaises(ValidationError):
            KpiThresholdsConfig.model_validate(bad)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
