"""KPIs that cannot be measured must SAY SO, and the one that can must exist.

THE FAILURE MODE THIS PINS. ``_ratio(x, 0)`` returns ``0.0``. So a rate whose denominator is empty --
because the records never ran in that mode, or never recorded that field -- does not arrive as "unknown".
It arrives as a confident zero, indistinguishable on a dashboard from a measured zero, and reading as
either *perfect* or *broken* depending on which rate it is. Measured on this repo's own console path,
three of the six rates the rollup used to publish were exactly that:

    dense_recovery_success_rate   denominator = attempts in a dense mode; shipped default mode is 'auto'
    first_attempt_success_rate    differs from pick_success_rate only by excluding recovered attempts,
                                  and the default open-loop path records no recovery actions at all
    median_cycle_time_s           reads extra['cycle_time_s'], which no production writer ever sets

And the duration that IS honest was in no KPI at all: ``extra['attempt_wall_time_s']`` is stamped by
``pick()`` on every attempt regardless of config, and only ever reached the CSV export.

The distinction being defended is between a denominator that is empty and a result that is zero. A
recovery rate of 0.0 measured over 200 dense attempts is a real and alarming number; the same 0.0 over
none of them is not a number. Only the INPUT tells them apart, which is why ``_has_support`` checks the
input and not the result.
"""

from __future__ import annotations

import unittest

from api.history import UNMEASURABLE_KPIS, rollup
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


def _record(**kwargs: object) -> GraspAttemptRecord:
    fields: dict = {
        "attempt_id": "a1",
        "mode": "auto",
        "final_outcome": "succeeded",
        "timestamp": 1_700_000_000.0,
    }
    fields.update(kwargs)  # type: ignore[arg-type]
    return GraspAttemptRecord.new(**fields)  # type: ignore[arg-type]


class TheAlwaysUnmeasurableTests(unittest.TestCase):
    def test_the_false_positive_rate_is_never_published_whatever_the_records_say(self) -> None:
        result = rollup([_record()], source="test")
        self.assertNotIn("false_positive_grasp_rate", result.kpis)
        self.assertIn("false_positive_grasp_rate", result.unmeasurable)
        self.assertIn("false_positive_grasp_rate", UNMEASURABLE_KPIS)


class TheConditionallyUnmeasurableTests(unittest.TestCase):
    """Withheld on records that cannot support them; published on records that can."""

    def test_a_default_console_record_set_withholds_all_three(self) -> None:
        result = rollup([_record(), _record(attempt_id="a2", final_outcome="no_valid_grasp")],
                        source="test")
        for name in ("dense_recovery_success_rate", "first_attempt_success_rate",
                     "median_cycle_time_s"):
            with self.subTest(kpi=name):
                self.assertNotIn(name, result.kpis, f"{name} was published over an empty denominator")
                self.assertIn(name, result.unmeasurable)

        # The two that ARE supported by any record set stay.
        self.assertIn("pick_success_rate", result.kpis)
        self.assertIn("dead_loop_rate", result.kpis)

    def test_dense_records_that_NEVER_RECOVERED_still_withhold_the_recovery_rate(self) -> None:
        """The half-a-conjunction bug, pinned.

        ``compute_kpis`` divides by ``mode in _DENSE_MODES AND bool(recovery_actions)`` (kpi.py:136).
        The first guard tested only the mode, so a log of 200 dense attempts that never recovered --
        which several shipped logs are, e.g. logs/s4_sim_soak.jsonl at 330 dense / 0 recovery --
        published ``_ratio(0, 0) = 0.0`` while the SAME response withheld first_attempt_success_rate
        for having no recovery actions. One response, two opposite answers about one fact.
        """
        records = [
            _record(attempt_id=f"a{i}", mode="dense_clutter", final_outcome="succeeded")
            for i in range(200)
        ]
        result = rollup(records, source="test")
        self.assertNotIn("dense_recovery_success_rate", result.kpis)
        self.assertIn("dense_recovery_success_rate", result.unmeasurable)
        # ...and it must not contradict its neighbour about the same absent fact.
        self.assertIn("first_attempt_success_rate", result.unmeasurable)

    def test_the_dense_mode_set_is_IMPORTED_not_re_spelled(self) -> None:
        """A prefix test would admit modes ``compute_kpis`` does not count, and drift from it."""
        from src.robot.grasping.replay.kpi import _DENSE_MODES

        # A mode that starts with "dense" but is not one kpi.py counts must NOT restore the rate.
        result = rollup(
            [_record(mode="dense_experimental", recovery_actions=("nudge",))], source="test"
        )
        self.assertNotIn("dense_experimental", _DENSE_MODES, "the fixture stopped being a stranger")
        self.assertNotIn("dense_recovery_success_rate", result.kpis)

    def test_a_dense_record_RESTORES_the_recovery_rate_even_when_it_is_zero(self) -> None:
        """The whole point: 0.0 over a real denominator is a measurement and must be shown."""
        result = rollup(
            [_record(mode="dense_clutter", final_outcome="no_valid_grasp",
                     recovery_actions=("nudge",))],
            source="test",
        )
        self.assertIn("dense_recovery_success_rate", result.kpis)
        self.assertEqual(result.kpis["dense_recovery_success_rate"], 0.0)
        self.assertNotIn("dense_recovery_success_rate", result.unmeasurable)

    def test_a_recorded_recovery_action_restores_the_first_attempt_rate(self) -> None:
        result = rollup([_record(recovery_actions=("nudge",)), _record(attempt_id="a2")],
                        source="test")
        self.assertIn("first_attempt_success_rate", result.kpis)

    def test_a_cycle_time_restores_the_cycle_time_median(self) -> None:
        result = rollup([_record(extra={"cycle_time_s": 4.0})], source="test")
        self.assertIn("median_cycle_time_s", result.kpis)
        self.assertEqual(result.kpis["median_cycle_time_s"], 4.0)

    def test_every_withheld_kpi_carries_a_reason_a_person_can_act_on(self) -> None:
        result = rollup([_record()], source="test")
        for name, reason in result.unmeasurable.items():
            with self.subTest(kpi=name):
                self.assertGreater(len(reason), 60, f"{name} was hidden without saying why")


class TheHonestDurationTests(unittest.TestCase):
    def test_the_attempt_wall_time_becomes_a_kpi(self) -> None:
        result = rollup(
            [
                _record(extra={"attempt_wall_time_s": 2.0}),
                _record(attempt_id="a2", extra={"attempt_wall_time_s": 4.0}),
                _record(attempt_id="a3", extra={"attempt_wall_time_s": 9.0}),
            ],
            source="test",
        )
        self.assertEqual(result.kpis["median_attempt_seconds"], 4.0)

    def test_records_without_it_leave_it_ABSENT_rather_than_zero(self) -> None:
        """A missing duration is not a duration of zero, and zero would be the faster-looking lie."""
        result = rollup([_record()], source="test")
        self.assertNotIn("median_attempt_seconds", result.kpis)

    def test_a_boolean_is_not_a_duration(self) -> None:
        """`isinstance(True, int)` is True in Python, and a stray flag would land as a 1.0 second."""
        result = rollup([_record(extra={"attempt_wall_time_s": True})], source="test")
        self.assertNotIn("median_attempt_seconds", result.kpis)

    def test_it_is_the_MEDIAN_so_one_slow_attempt_does_not_move_it(self) -> None:
        result = rollup(
            [
                _record(extra={"attempt_wall_time_s": 3.0}),
                _record(attempt_id="a2", extra={"attempt_wall_time_s": 3.0}),
                _record(attempt_id="a3", extra={"attempt_wall_time_s": 300.0}),
            ],
            source="test",
        )
        self.assertEqual(result.kpis["median_attempt_seconds"], 3.0)


class TheEmptyCaseTests(unittest.TestCase):
    def test_no_records_publishes_no_conditional_rate_at_all(self) -> None:
        """A fresh bench must not show a table of confident zeros before it has done anything."""
        result = rollup([], source="test")
        self.assertEqual(result.total_attempts, 0)
        for name in ("dense_recovery_success_rate", "first_attempt_success_rate",
                     "median_cycle_time_s", "median_attempt_seconds"):
            with self.subTest(kpi=name):
                self.assertNotIn(name, result.kpis)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
