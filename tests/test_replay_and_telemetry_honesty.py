"""The replay command lines and the record serializer, judged on what they SAY.

Every test here corresponds to a defect reproduced against the shipped code before the repair, and
each one names the failure it caught:

* three CLI legs answered a missing input file with a raw ``FileNotFoundError`` traceback while
  every neighbouring leg answered with a sentence and an exit code;
* ``--baseline-report`` silently ignored ``--out`` and rewrote the git-tracked baseline instead;
* the library KPI leg printed ``false_positive_grasp_rate: 0.0`` as a measurement, a number that
  comes from a field no writer on this stack ever sets;
* the record serializer read ``report.verification``, an attribute ``AutonomousGraspReport`` has
  never had, so the verification block was ``None`` on every real record;
* nothing wrote the ``refinement`` block, so the three refine-stage outcomes that the telemetry
  catalog requires it for could never produce a complete record. The writer that repaired it left
  on 2026-09-29 with the two-scan refinement, and its tests with it: no pick produces those
  outcomes any more, and a record logged before then still audits
  (``tests/test_no_second_look_before_the_close.py``).
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from src.robot.grasping.replay import __main__ as replay_main
from src.robot.grasping.replay.runs import SoakGate


def _run_cli(argv: list[str]) -> tuple[int, str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = replay_main.main(argv)
    return code, buffer.getvalue()


class MissingInputIsAnAnswerTests(unittest.TestCase):
    """A missing input file is an operator typo, not a crash site."""

    def test_records_gate_answers_a_missing_log(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            code, out = _run_cli(["--records-gate", str(missing)])
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_sim_soak_report_answers_a_missing_log_and_writes_no_report(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "nope.jsonl"
            report = Path(tmp) / "report.json"
            code, out = _run_cli(
                ["--sim-soak-report", str(missing), "--baseline-out", str(report)]
            )
            self.assertFalse(
                report.exists(), "a report was written for an input nobody could read"
            )
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_failure_taxonomy_answers_a_missing_pack(self) -> None:
        with TemporaryDirectory() as tmp:
            missing = Path(tmp) / "pack.jsonl"
            code, out = _run_cli(
                ["--failure-taxonomy", str(missing), "--out", str(Path(tmp) / "r.json")]
            )
        self.assertEqual(code, 2)
        payload = json.loads(out)
        self.assertIn("error", payload)
        self.assertIn(str(missing), payload["error"])

    def test_the_gate_itself_returns_a_verdict_rather_than_raising(self) -> None:
        """The library twin of the CLI leg above. ``RecordLog`` has answered this way since it was
        written; the gate raised.
        """
        verdict = SoakGate.over_records("does/not/exist.jsonl").evaluate()
        self.assertFalse(verdict.passes)
        self.assertEqual(verdict.exit_code, 2)
        self.assertTrue(verdict.violations)


class BaselineOutIsHonouredTests(unittest.TestCase):
    """``--out`` names a file. A mode that ignores it overwrites a committed one instead."""

    def test_baseline_report_writes_where_the_operator_asked(self) -> None:
        written: list[Path] = []

        class _Measured:
            report = {"stub": True}
            audit_offenders = 0
            exit_code = 0

            def write(self, path: Path) -> Path:
                written.append(Path(path))
                return Path(path)

        class _Baseline:
            @classmethod
            def canonical(cls) -> "_Baseline":
                return cls()

            def measure(self) -> _Measured:
                return _Measured()

        with TemporaryDirectory() as tmp:
            asked = Path(tmp) / "mine" / "baseline.json"
            with mock.patch.object(replay_main, "Baseline", _Baseline):
                code, out = _run_cli(["--baseline-report", "--out", str(asked)])
            self.assertEqual(code, 0)
            self.assertEqual(written, [asked.resolve()])
            self.assertEqual(json.loads(out)["report_path"], str(asked.resolve()))


class UnmeasurableRatesAreNamedTests(unittest.TestCase):
    """A rate with no source must not print as a number, in the library as well as the console."""

    def _log(self, tmp: str) -> Path:
        from src.robot.grasping.replay.soak import (
            SoakScenarioSpec,
            generate_soak_records,
        )

        records = generate_soak_records(
            SoakScenarioSpec(
                name="unit",
                mode="easy",
                attempts=12,
                failure_class_weights={"succeeded": 9.0, "no_valid_grasp": 1.0},
                recovery_success_rate=0.0,
                cycle_time_mean_s=1.5,
                cycle_time_jitter_s=0.2,
                seed=7,
            )
        )
        path = Path(tmp) / "records.jsonl"
        path.write_text("\n".join(r.to_json() for r in records) + "\n", encoding="utf-8")
        return path

    def test_the_records_leg_names_the_rate_it_cannot_measure(self) -> None:
        with TemporaryDirectory() as tmp:
            code, out = _run_cli(["--records", str(self._log(tmp))])
        payload = json.loads(out)
        self.assertIn("unmeasurable", payload)
        self.assertIn("false_positive_grasp_rate", payload["unmeasurable"])
        self.assertNotIn("false_positive_grasp_rate", payload["kpi"])
        self.assertIn(
            "post-grasp re-check", payload["unmeasurable"]["false_positive_grasp_rate"]
        )
        self.assertEqual(code, 0)

    def test_the_console_reads_the_library_rather_than_its_own_copy(self) -> None:
        from api.history import UNMEASURABLE_KPIS
        from src.robot.grasping.replay.kpi import UNMEASURABLE_KPIS as LIBRARY

        self.assertEqual(dict(UNMEASURABLE_KPIS), dict(LIBRARY))


class RecordSerializerReadsWhatExistsTests(unittest.TestCase):
    """``getattr(report, "verification", None)`` on a report with no such attribute is always
    ``None``: a lookup that can neither fail nor succeed.
    """

    def test_the_report_has_never_had_a_verification_attribute(self) -> None:
        from src.robot.execution.autonomous_grasp.report import (
            AutonomousGraspReport,
        )

        self.assertNotIn("verification", {f.name for f in fields(AutonomousGraspReport)})

    def test_no_writer_fills_the_verification_block_from_telemetry_any_more(self) -> None:
        """The service stamped these three keys from its verification stage, which left on 2026-09-29.
        A stray stamp in a report's telemetry is not a verdict: the block is written from a simulator's
        ground-truth lift only, labelled as such."""
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )

        class _Report:
            outcome = "succeeded"
            mode = "dense_clutter"
            profile = None
            pick_report = None
            telemetry = {
                "verification_outcome": "passed",
                "verification_reason": "width_delta_within_tolerance",
                "verification_telemetry": {"width_delta_mm": 3.2},
            }

        self.assertIsNone(to_attempt_record(_Report(), attempt_id="a1").verification)
        lifted = to_attempt_record(_Report(), attempt_id="a2", extra={"sim_lifted": True, "sim_lift_mm": 41.5})
        self.assertIsNotNone(lifted.verification)
        assert lifted.verification is not None
        self.assertEqual(lifted.verification["source"], "sim_ground_truth_lift")
        self.assertIs(lifted.verification["lifted"], True)
        self.assertEqual(lifted.verification["lift_mm"], 41.5)

class NoWriterFillsTheRefinementBlockTests(unittest.TestCase):
    """The block stays in the record for the records logged before 2026-09-29, and nothing fills it now."""

    def test_an_open_loop_attempt_still_carries_no_refinement_block(self) -> None:
        """⚠ THE BYTE-IDENTICAL HALF. The default pick runs no refiner, and a fabricated block there
        would be exactly the invention this writer exists to avoid.
        """
        from src.robot.execution.autonomous_grasp.record_logging import (
            to_attempt_record,
        )

        class _Report:
            outcome = "succeeded"
            mode = "auto"
            profile = None
            pick_report = None
            telemetry: dict = {}

        self.assertIsNone(to_attempt_record(_Report(), attempt_id="open-1").refinement)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
