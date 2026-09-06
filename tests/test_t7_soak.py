"""Phase T7 — RED soak invariant test (the strict gate).

This is the production-readiness gate enforced by Phase T's locked
numeric KPI thresholds, its clause 10:

* ``>= 1,000`` attempts,
* ``0`` untyped terminal outcomes,
* ``0`` unbounded retry loops,
* ``dead_loop_rate <= 0.5%``.

The test consumes the locked ``kpi_thresholds.yaml`` artefact so
operators can tighten thresholds without code changes.

Q3=A + Q7=B: deterministic synthetic generator, plain unittest, runs
in-process under ``unittest discover``.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.config.schema.robot.kpi_schema import KpiThresholdsConfig
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
from src.robot.grasping.replay.kpi import compute_kpis
from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    generate_soak_records,
)
from src.robot.grasping.replay.telemetry_catalog import audit_records


_THRESHOLDS_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "robot"
    / "kpi_thresholds.yaml"
)
_VALID_OUTCOMES = frozenset(v.value for v in AutonomousGraspOutcome)


def _thresholds() -> KpiThresholdsConfig:
    with _THRESHOLDS_PATH.open() as fh:
        return KpiThresholdsConfig.model_validate(yaml.safe_load(fh))


def _build_corpus(total: int) -> tuple:
    """Mix EASY / AUTO / DENSE attempts to exercise every mode."""

    per = total // 3
    remainder = total - per * 3
    easy = generate_soak_records(
        SoakScenarioSpec(
            name="easy",
            mode="easy",
            attempts=per,
            failure_class_weights={
                "succeeded": 19.0,
                "no_valid_grasp": 1.0,
            },
            recovery_success_rate=0.0,
            cycle_time_mean_s=1.5,
            cycle_time_jitter_s=0.2,
            seed=11,
        )
    )
    auto = generate_soak_records(
        SoakScenarioSpec(
            name="auto",
            mode="auto",
            attempts=per,
            failure_class_weights={
                "succeeded": 14.0,
                "execution_failed": 2.0,
                "no_valid_grasp": 2.0,
                "decision_fail_closed": 1.0,
            },
            recovery_success_rate=0.5,
            cycle_time_mean_s=2.5,
            cycle_time_jitter_s=0.5,
            seed=22,
        )
    )
    dense = generate_soak_records(
        SoakScenarioSpec(
            name="dense",
            mode="dense_clutter",
            attempts=per + remainder,
            failure_class_weights={
                "succeeded": 990.0,
                "execution_failed": 5.0,
                "no_valid_grasp": 4.0,
                "recovery_exhausted": 1.0,
            },
            recovery_success_rate=0.6,
            cycle_time_mean_s=3.5,
            cycle_time_jitter_s=0.7,
            seed=33,
        )
    )
    return (*easy, *auto, *dense)


class SoakGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.thresholds = _thresholds()
        self.records = _build_corpus(self.thresholds.soak.min_attempts)

    def test_attempt_floor_met(self) -> None:
        self.assertGreaterEqual(
            len(self.records), self.thresholds.soak.min_attempts
        )

    def test_no_untyped_terminal_outcomes(self) -> None:
        untyped = [
            r.attempt_id
            for r in self.records
            if r.final_outcome not in _VALID_OUTCOMES
        ]
        self.assertEqual(untyped, [], f"untyped outcomes leaked: {untyped[:5]!r}")

    def test_no_telemetry_gaps(self) -> None:
        offenders = audit_records(self.records)
        self.assertEqual(
            offenders,
            [],
            f"telemetry catalog gaps: {offenders[:3]!r}",
        )

    def test_dead_loop_rate_under_soak_cap(self) -> None:
        summary = compute_kpis(self.records)
        self.assertLessEqual(
            summary.dead_loop_rate,
            self.thresholds.soak.dead_loop_rate_max,
            f"dead-loop rate {summary.dead_loop_rate:.4f} exceeds cap "
            f"{self.thresholds.soak.dead_loop_rate_max:.4f}",
        )

    def test_easy_mode_zero_dead_loop(self) -> None:
        easy_records = tuple(r for r in self.records if r.mode == "easy")
        summary = compute_kpis(easy_records)
        self.assertLessEqual(
            summary.dead_loop_rate, self.thresholds.easy.dead_loop_rate_max
        )

    def test_easy_mode_false_positive_under_cap(self) -> None:
        easy_records = tuple(r for r in self.records if r.mode == "easy")
        summary = compute_kpis(easy_records)
        self.assertLessEqual(
            summary.false_positive_grasp_rate,
            self.thresholds.easy.false_positive_grasp_rate_max,
        )

    def test_dense_mode_dead_loop_under_cap(self) -> None:
        dense_records = tuple(
            r for r in self.records if r.mode == "dense_clutter"
        )
        summary = compute_kpis(dense_records)
        self.assertLessEqual(
            summary.dead_loop_rate, self.thresholds.dense.dead_loop_rate_max
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
