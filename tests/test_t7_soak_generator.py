"""Phase T7 — RED tests for the synthetic soak generator.

Q3=A locks an in-process, deterministic synthetic generator that
produces a stream of :class:`GraspAttemptRecord` instances suitable
for KPI gate evaluation.

Contract:

* ``SoakScenarioSpec`` is a frozen dataclass with at least:
  ``name``, ``mode``, ``attempts``, ``failure_class_weights``
  (``Mapping[str, float]``), ``recovery_success_rate`` (float in
  ``[0, 1]``), ``cycle_time_mean_s`` (positive float),
  ``cycle_time_jitter_s`` (non-negative float), ``seed`` (int).
* ``generate_soak_records(spec)`` returns a tuple of records of
  length ``spec.attempts``.
* Determinism: identical specs produce identical record sequences
  (byte-equal after ``to_dict()``).
* Every produced record has all telemetry-catalog required fields
  populated for its emitted ``final_outcome``.
* Failure-class weights normalise to a probability distribution; an
  all-zero weight map raises ``ValueError``.
* ``mode`` is rejected unless it is a known
  :class:`GraspMode` value.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    generate_soak_records,
)
from src.robot.grasping.replay.telemetry_catalog import audit_record


def _spec(**overrides: object) -> SoakScenarioSpec:
    defaults: dict[str, object] = {
        "name": "default",
        "mode": "auto",
        "attempts": 100,
        "failure_class_weights": {
            "succeeded": 8.0,
            "execution_failed": 1.0,
            "no_valid_grasp": 1.0,
        },
        "recovery_success_rate": 0.6,
        "cycle_time_mean_s": 2.5,
        "cycle_time_jitter_s": 0.5,
        "seed": 42,
    }
    defaults.update(overrides)
    return SoakScenarioSpec(**defaults)  # type: ignore[arg-type]


class SoakSpecValidationTests(unittest.TestCase):
    def test_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            _spec(mode="not_a_mode")

    def test_rejects_zero_attempts(self) -> None:
        with self.assertRaises(ValueError):
            _spec(attempts=0)

    def test_rejects_negative_attempts(self) -> None:
        with self.assertRaises(ValueError):
            _spec(attempts=-3)

    def test_rejects_recovery_rate_out_of_unit_interval(self) -> None:
        with self.assertRaises(ValueError):
            _spec(recovery_success_rate=1.5)
        with self.assertRaises(ValueError):
            _spec(recovery_success_rate=-0.1)

    def test_rejects_all_zero_failure_class_weights(self) -> None:
        with self.assertRaises(ValueError):
            _spec(failure_class_weights={"succeeded": 0.0})

    def test_rejects_unknown_outcome_in_weights(self) -> None:
        with self.assertRaises(ValueError):
            _spec(failure_class_weights={"made_up": 1.0})

    def test_rejects_negative_jitter(self) -> None:
        with self.assertRaises(ValueError):
            _spec(cycle_time_jitter_s=-0.1)

    def test_rejects_non_positive_cycle_mean(self) -> None:
        with self.assertRaises(ValueError):
            _spec(cycle_time_mean_s=0.0)


class GenerateSoakRecordsTests(unittest.TestCase):
    def test_returns_requested_attempt_count(self) -> None:
        spec = _spec(attempts=250)
        records = generate_soak_records(spec)
        self.assertEqual(len(records), 250)

    def test_is_deterministic_under_fixed_seed(self) -> None:
        spec = _spec(attempts=100, seed=1234)
        a = generate_soak_records(spec)
        b = generate_soak_records(spec)
        self.assertEqual(
            tuple(r.to_dict() for r in a),
            tuple(r.to_dict() for r in b),
        )

    def test_different_seeds_diverge(self) -> None:
        a = generate_soak_records(_spec(attempts=50, seed=1))
        b = generate_soak_records(_spec(attempts=50, seed=2))
        self.assertNotEqual(
            tuple(r.to_dict() for r in a),
            tuple(r.to_dict() for r in b),
        )

    def test_every_record_passes_telemetry_catalog(self) -> None:
        records = generate_soak_records(_spec(attempts=500))
        offenders = [
            (r.attempt_id, r.final_outcome, audit_record(r))
            for r in records
            if audit_record(r)
        ]
        self.assertEqual(
            offenders,
            [],
            f"synthetic records failed telemetry audit: {offenders[:5]!r}",
        )

    def test_mode_field_matches_spec(self) -> None:
        records = generate_soak_records(_spec(mode="dense_clutter"))
        self.assertTrue(all(r.mode == "dense_clutter" for r in records))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
