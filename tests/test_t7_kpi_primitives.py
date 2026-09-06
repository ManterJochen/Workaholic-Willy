"""Phase T7 — RED tests for the KPI summary primitive.

The replay harness computes seven KPIs over a sequence of
:class:`GraspAttemptRecord` instances. The contract is:

* ``compute_kpis(records)`` returns an immutable :class:`KpiSummary`.
* Every KPI is normalised to the unit interval (rates) or seconds
  (median cycle time). ``None`` is only allowed for the cycle-time
  field when the source records lack timing extras.
* Empty input is well-defined: rates are ``0.0`` and the
  ``total_attempts`` field is ``0`` — never a divide-by-zero.
* ``KpiSummary`` is JSON-safe via ``to_dict()``.

Locked under Q2=A: the harness consumes the existing
:class:`GraspAttemptRecord` format. No new record dataclass.
"""

from __future__ import annotations

import unittest

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.kpi import KpiSummary, compute_kpis


def _record(
    *,
    mode: str,
    outcome: str,
    cycle_time_s: float | None = None,
    recovery_actions: tuple[dict, ...] = (),
    safety_rejected: bool = False,
    verification_failed_after_success: bool = False,
) -> GraspAttemptRecord:
    extra: dict[str, object] = {}
    if cycle_time_s is not None:
        extra["cycle_time_s"] = float(cycle_time_s)
    if safety_rejected:
        extra["safety_rejected"] = True
    if verification_failed_after_success:
        extra["verification_failed_after_success"] = True
    return GraspAttemptRecord.new(
        attempt_id=f"a-{mode}-{outcome}-{id(extra)}",
        mode=mode,
        final_outcome=outcome,
        recovery_actions=recovery_actions,
        extra=extra,
    )


class ComputeKpisTests(unittest.TestCase):
    def test_empty_input_returns_zero_rates(self) -> None:
        summary = compute_kpis(())
        self.assertIsInstance(summary, KpiSummary)
        self.assertEqual(summary.total_attempts, 0)
        self.assertEqual(summary.pick_success_rate, 0.0)
        self.assertEqual(summary.first_attempt_success_rate, 0.0)
        self.assertEqual(summary.dead_loop_rate, 0.0)
        self.assertEqual(summary.safety_rejection_rate, 0.0)
        self.assertEqual(summary.dense_recovery_success_rate, 0.0)
        self.assertEqual(summary.false_positive_grasp_rate, 0.0)
        self.assertIsNone(summary.median_cycle_time_s)

    def test_pick_success_rate_counts_only_succeeded(self) -> None:
        records = (
            _record(mode="easy", outcome="succeeded"),
            _record(mode="easy", outcome="succeeded"),
            _record(mode="easy", outcome="execution_failed"),
            _record(mode="easy", outcome="no_valid_grasp"),
        )
        summary = compute_kpis(records)
        self.assertEqual(summary.total_attempts, 4)
        self.assertAlmostEqual(summary.pick_success_rate, 0.5)

    def test_first_attempt_success_rate_excludes_recovered(self) -> None:
        records = (
            _record(mode="auto", outcome="succeeded"),
            _record(
                mode="auto",
                outcome="succeeded",
                recovery_actions=({"action": "next_viewpoint"},),
            ),
            _record(mode="auto", outcome="execution_failed"),
        )
        summary = compute_kpis(records)
        # one of three succeeded without any recovery action
        self.assertAlmostEqual(summary.first_attempt_success_rate, 1.0 / 3.0)

    def test_dead_loop_rate_counts_recovery_exhausted_terminal(self) -> None:
        records = (
            _record(mode="dense_clutter", outcome="succeeded"),
            _record(mode="dense_clutter", outcome="recovery_exhausted"),
            _record(mode="dense_clutter", outcome="recovery_exhausted"),
            _record(mode="dense_clutter", outcome="execution_failed"),
        )
        summary = compute_kpis(records)
        self.assertAlmostEqual(summary.dead_loop_rate, 0.5)

    def test_safety_rejection_rate_reads_extra_flag(self) -> None:
        records = (
            _record(mode="auto", outcome="execution_failed", safety_rejected=True),
            _record(mode="auto", outcome="execution_failed"),
            _record(mode="auto", outcome="succeeded"),
        )
        summary = compute_kpis(records)
        self.assertAlmostEqual(summary.safety_rejection_rate, 1.0 / 3.0)

    def test_dense_recovery_success_rate_uses_dense_modes_only(self) -> None:
        records = (
            # EASY successes never count as recovery-assisted.
            _record(mode="easy", outcome="succeeded"),
            # DENSE success with recovery -> counts.
            _record(
                mode="dense_clutter",
                outcome="succeeded",
                recovery_actions=({"action": "nudge_target"},),
            ),
            # DENSE success without recovery -> not counted as recovery success.
            _record(mode="dense_clutter", outcome="succeeded"),
            # DENSE failure with recovery attempts -> denominator.
            _record(
                mode="dense_clutter",
                outcome="recovery_exhausted",
                recovery_actions=({"action": "container_agitate"},),
            ),
        )
        summary = compute_kpis(records)
        # numerator = 1 dense success w/ recovery
        # denominator = 2 dense attempts that ran recovery
        self.assertAlmostEqual(summary.dense_recovery_success_rate, 0.5)

    def test_false_positive_rate_uses_verification_flag(self) -> None:
        records = (
            _record(
                mode="auto",
                outcome="succeeded",
                verification_failed_after_success=True,
            ),
            _record(mode="auto", outcome="succeeded"),
            _record(mode="auto", outcome="execution_failed"),
        )
        summary = compute_kpis(records)
        # 1 false positive out of 2 reported successes
        self.assertAlmostEqual(summary.false_positive_grasp_rate, 0.5)

    def test_median_cycle_time_from_extras(self) -> None:
        records = (
            _record(mode="auto", outcome="succeeded", cycle_time_s=2.0),
            _record(mode="auto", outcome="succeeded", cycle_time_s=4.0),
            _record(mode="auto", outcome="execution_failed", cycle_time_s=6.0),
        )
        summary = compute_kpis(records)
        self.assertAlmostEqual(summary.median_cycle_time_s, 4.0)

    def test_to_dict_round_trip_is_json_safe(self) -> None:
        import json

        records = (
            _record(mode="easy", outcome="succeeded", cycle_time_s=1.5),
            _record(mode="easy", outcome="execution_failed"),
        )
        summary = compute_kpis(records)
        d = summary.to_dict()
        json.dumps(d)  # must not raise
        for key in (
            "total_attempts",
            "pick_success_rate",
            "first_attempt_success_rate",
            "dead_loop_rate",
            "safety_rejection_rate",
            "median_cycle_time_s",
            "dense_recovery_success_rate",
            "false_positive_grasp_rate",
        ):
            self.assertIn(key, d)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
