"""The replay nouns: the roll-up, the gate and the baseline.

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
    PACK_DEPENDENT_GATE_KEYS,
    RecordLog,
    SoakGate,
    SoakSource,
    TelemetryVerdict,
)
from src.robot.grasping.replay.soak import (
    SoakScenarioSpec,
    generate_soak_records,
)


def _records(attempts: int = 12, seed: int = 7):
    return generate_soak_records(
        SoakScenarioSpec(
            name="unit",
            mode="easy",
            attempts=attempts,
            failure_class_weights={"succeeded": 9.0, "no_valid_grasp": 1.0},
            recovery_success_rate=0.0,
            cycle_time_mean_s=1.5,
            cycle_time_jitter_s=0.2,
            seed=seed,
        )
    )


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
        self.assertIs(SoakGate.over_synthetic("t.yaml").source, SoakSource.SYNTHETIC_STREAM)
        self.assertIs(SoakGate.over_canonical_packs().source, SoakSource.CANONICAL_PACKS)

    def test_the_pack_dependent_keys_are_present_and_not_applicable(self) -> None:
        """⛔ THE OTHER DEFECT THIS CAUGHT. `evaluate_soak_gate` returns the seven record-intrinsic
        keys ONLY; both CLI modes then write the four pack-dependent ones in. My first version
        filtered them out of the result instead of adding them, so the filter could never fire and
        the gate reported seven keys where the CLI reports eleven.
        """
        with TemporaryDirectory() as tmp:
            log = Path(tmp) / "r.jsonl"
            log.write_text(
                "\n".join(r.to_json() for r in _records()) + "\n", encoding="utf-8"
            )
            verdict = SoakGate.over_records(log).evaluate()

        for key in PACK_DEPENDENT_GATE_KEYS:
            self.assertIn(key, verdict.keys)
            self.assertIs(verdict.keys[key], GateKeyStatus.NOT_APPLICABLE)
        self.assertEqual(len(verdict.keys), 11)
        # ⚠ A gate whose keys are all not_applicable passes loudest. This is the number that says so.
        self.assertEqual(verdict.judged, 7)

        wire = verdict.gate_wire()
        self.assertEqual(wire["slo_packs_pass"], "not_applicable")
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
        self.assertIn("NOT gated", text)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
