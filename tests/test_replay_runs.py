"""The replay nouns: the roll-up and the gate.

Every assertion here corresponds to something that was actually wrong at some point while this
module was written, or to a rule the convention would be pointless without. Nothing is here to hold
a coverage percentage.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.robot.grasping.replay.runs import (
    GateKeyStatus,
    RecordLog,
    SoakGate,
    SoakSource,
    TelemetryVerdict,
)
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord


def _records(attempts: int = 12) -> list[GraspAttemptRecord]:
    """A clean log: one attempt in ten found no grasp, every other one succeeded and carries its execution block."""
    return [
        GraspAttemptRecord.new(
            attempt_id=f"unit-{index:04d}", mode="easy", timestamp=float(index),
            final_outcome="no_valid_grasp" if index % 10 == 9 else "succeeded",
            execution=None if index % 10 == 9 else {"outcome": "executed"},
        )
        for index in range(attempts)
    ]


class TestRecordLog(unittest.TestCase):
    def test_clean_records_are_sound_and_exit_zero(self) -> None:
        rollup = RecordLog.from_records(_records()).kpis()
        self.assertIs(rollup.verdict, TelemetryVerdict.SOUND)
        self.assertTrue(rollup.sound)
        self.assertEqual(rollup.exit_code, 0)
        self.assertEqual(rollup.records, 12)
        self.assertIn("pick_success_rate", rollup.kpi)

    def test_a_missing_log_is_a_verdict_not_an_exception(self) -> None:
        """⛔ The whole reason reading happens in the verb rather than the factory."""
        rollup = RecordLog.from_jsonl("does/not/exist.jsonl").kpis()
        self.assertIs(rollup.verdict, TelemetryVerdict.UNREADABLE)
        self.assertEqual(rollup.exit_code, 2)
        self.assertTrue(rollup.detail)

    def test_the_two_audits_stay_two_lists(self) -> None:
        """⛔ THE DEFECT THIS CAUGHT. The first version stored one merged, id-sorted tuple. Both
        audits emit in RECORD order, so a payload rebuilt from a sorted merge would not match the
        one the CLI has printed for years, and two records sharing an attempt id would collapse.
        """
        rollup = RecordLog.from_records(_records()).kpis()
        self.assertIsInstance(rollup.missing_telemetry, tuple)
        self.assertIsInstance(rollup.wrong_types, tuple)
        # The merged view is derived, never stored.
        self.assertEqual(rollup.offenders, ())

    def test_render_is_whole_ascii_and_has_no_trailing_newline(self) -> None:
        text = RecordLog.from_records(_records()).kpis().render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("SOUND", text)


class TestSoakGate(unittest.TestCase):
    def test_each_door_records_which_door_it_was(self) -> None:
        """⭐ The field the CLI spent four `note` paragraphs on. A green gate over the wrong
        population is how a stack gets credited with work it did not do.
        """
        self.assertIs(SoakGate.over_records("x.jsonl").source, SoakSource.REAL_RECORDS)
        self.assertIs(SoakGate.over_sim_records("x.jsonl").source, SoakSource.SIM_RECORDS)

    def test_every_key_is_judged_from_the_records(self) -> None:
        """The pack-dependent keys left with the synthetic soak on 2026-10-16: the six keys a door reports are the six
        the gate judges, each passed or failed."""
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "r.jsonl"
            log.write_text(
                "\n".join(r.to_json() for r in _records()) + "\n", encoding="utf-8"
            )
            verdict = SoakGate.over_records(log).evaluate()

        self.assertEqual(6, len(verdict.keys))
        self.assertTrue(all(status in (GateKeyStatus.PASSED, GateKeyStatus.FAILED) for status in verdict.keys.values()))
        wire = verdict.gate_wire()
        self.assertIs(wire["passes"], verdict.passes)

    def test_exit_code_is_derived_from_the_violations(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "r.jsonl"
            log.write_text(
                "\n".join(r.to_json() for r in _records()) + "\n", encoding="utf-8"
            )
            # 12 records against the 2000-attempt floor: this must fail, and say why.
            strict = SoakGate.over_records(log).evaluate()
            # The same records with an honestly-stated smaller floor: this must pass.
            lenient = SoakGate.over_sim_records(log, min_attempts=1).evaluate()

        self.assertFalse(strict.passes)
        self.assertEqual(strict.exit_code, 1)
        self.assertTrue(any("min_attempts" in v for v in strict.violations))
        self.assertTrue(lenient.passes)
        self.assertEqual(lenient.exit_code, 0)

    def test_render_states_what_the_verdict_is_evidence_of(self) -> None:
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "r.jsonl"
            log.write_text(
                "\n".join(r.to_json() for r in _records()) + "\n", encoding="utf-8"
            )
            text = SoakGate.over_sim_records(log, min_attempts=1).evaluate().render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("not gated", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
