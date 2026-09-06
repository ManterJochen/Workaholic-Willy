"""`check-dataset` as a library call, with the three exit codes it used to keep to itself.

⛔⛔ **WHAT WAS TRAPPED IN THE CLI, AND IT WAS NOT THE ASSESSMENT.** `assess_records` was already a
pure function and `format_readiness` was already the renderer. What no Python caller could reach was
everything around them: the two ways reading a log fails, and the meaning of exit 0 / 1 / 2, which
existed only as literal `return` statements inside `_cmd_check_dataset`. A caller had to open the
file themselves, invent their own error handling, and then guess at the same verdict boundary.

⚠ **THE VERDICT IS REPORTED, NEVER RAISED**, because `_commands_dataset.py:33-36` says this check is
"reported and deliberately NOT gating". A library that raised on an unreadable log would take that
decision away from the caller.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.contracts import Rendered, Structured
from src.robot.grasping.rl.runs import (
    ReadinessVerdict,
    RecordLogCheck,
    RecordLogReport,
)

#: One record with an outcome. Enough to be READ and judged, not enough to be trainable, which is
#: exactly the middle verdict this test needs and the hardest one to reach by accident.
_ONE_RECORD = {"success": True}


class TheThreeVerdictsTests(unittest.TestCase):

    def test_a_missing_log_is_unreadable_not_empty(self) -> None:
        """⛔ NOT THE SAME AS AN EMPTY DATASET. `UNREADABLE` means the question was never asked,
        because there was nothing to ask it of. Collapsing the two would report a mistyped path and
        a bad corpus as one outcome."""
        with tempfile.TemporaryDirectory() as tmp:
            report = RecordLogCheck.from_records_log(Path(tmp) / "nope.jsonl").assess()
        self.assertIs(report.verdict, ReadinessVerdict.UNREADABLE)
        self.assertEqual(report.detail, "no such record log")
        self.assertIsNone(report.readiness)

    def test_an_unparseable_log_is_unreadable_with_the_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "broken.jsonl"
            path.write_text("not json\n", encoding="utf-8")
            report = RecordLogCheck.from_records_log(path).assess()
        self.assertIs(report.verdict, ReadinessVerdict.UNREADABLE)
        self.assertIn("ValueError", report.detail)

    def test_a_readable_log_that_cannot_be_trained_on_says_so(self) -> None:
        report = RecordLogCheck.from_records([_ONE_RECORD]).assess()
        self.assertIs(report.verdict, ReadinessVerdict.NOT_TRAINABLE)
        self.assertFalse(report.trainable)
        self.assertIsNotNone(report.readiness)

    def test_asking_never_raises(self) -> None:
        """The whole point of a reported verdict: a caller deciding whether to train gets an answer,
        not a traceback."""
        with tempfile.TemporaryDirectory() as tmp:
            for name, content in (("missing", None), ("broken", "not json\n"), ("empty", "")):
                with self.subTest(name):
                    path = Path(tmp) / f"{name}.jsonl"
                    if content is not None:
                        path.write_text(content, encoding="utf-8")
                    self.assertIsInstance(RecordLogCheck.from_records_log(path).assess(),
                                          RecordLogReport)


class TheExitCodesLiveOnTheReportTests(unittest.TestCase):
    """⭐ ON THE REPORT, NOT IN THE CLI, which is what makes the command a shim.
    `safety/planning/doctor.py:154` already does this and its CLI is one line for the same reason."""

    def test_each_verdict_carries_the_code_the_cli_returned(self) -> None:
        expected = {
            ReadinessVerdict.TRAINABLE: 0,
            ReadinessVerdict.NOT_TRAINABLE: 1,
            ReadinessVerdict.UNREADABLE: 2,
        }
        for verdict, code in expected.items():
            with self.subTest(verdict):
                self.assertEqual(RecordLogReport(verdict, "x").exit_code, code)

    def test_the_codes_are_distinct(self) -> None:
        """A collapsed pair would make two different operator situations indistinguishable to a
        script that branches on the code."""
        codes = {RecordLogReport(v, "x").exit_code for v in ReadinessVerdict}
        self.assertEqual(len(codes), len(ReadinessVerdict))


class TheReportContractTests(unittest.TestCase):

    def test_it_is_rendered_and_structured(self) -> None:
        report = RecordLogCheck.from_records([_ONE_RECORD]).assess()
        self.assertIsInstance(report, Rendered)
        self.assertIsInstance(report, Structured)

    def test_render_is_the_whole_thing_and_summary_is_a_fragment(self) -> None:
        """⚠ WHY THE SHORT FORM IS NOT CALLED `render`. The convention is that `render()` returns the
        WHOLE object and takes no arguments; the CLI's `--quiet` wants less than that, so it gets its
        own name. Both go through `format_readiness`, so there is still one implementation."""
        report = RecordLogCheck.from_records([_ONE_RECORD]).assess()
        self.assertGreater(len(report.render()), len(report.summary()))
        self.assertTrue(report.summary())

    def test_render_obeys_the_contract_for_every_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            reports = [
                RecordLogCheck.from_records_log(Path(tmp) / "nope.jsonl").assess(),
                RecordLogCheck.from_records([_ONE_RECORD]).assess(),
            ]
        for report in reports:
            with self.subTest(report.verdict):
                text = report.render()
                self.assertTrue(text.isascii(), "printed to a cp1252 console")
                self.assertFalse(text.endswith("\n"), "the caller owns the line break")
                self.assertTrue(text.strip())

    def test_to_dict_survives_json_without_a_custom_encoder(self) -> None:
        for report in (RecordLogCheck.from_records([_ONE_RECORD]).assess(),
                       RecordLogReport(ReadinessVerdict.UNREADABLE, "x", detail="gone")):
            with self.subTest(report.verdict):
                payload = report.to_dict()
                self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_an_unreadable_report_still_answers_every_field(self) -> None:
        """⛔ NO `None` HOLES IN THE WIRE FORM. A consumer reading telemetry months later should not
        have to know which verdict leaves which key absent."""
        payload = RecordLogReport(ReadinessVerdict.UNREADABLE, "x", detail="gone").to_dict()
        self.assertEqual(payload["records"], 0)
        self.assertEqual(payload["blocking"], [])
        self.assertEqual(payload["exit_code"], 2)


class TheTwoDoorsAgreeTests(unittest.TestCase):
    """⛔ ONE ASSESSMENT, TWO WAYS IN. `from_records_log` reads and then judges; `from_records`
    judges what it was handed. If those ever produced different verdicts for the same content, the
    file path would be changing the answer."""

    def test_reading_a_log_gives_the_same_verdict_as_passing_its_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run.jsonl"
            path.write_text(json.dumps(_ONE_RECORD) + "\n", encoding="utf-8")
            from_file = RecordLogCheck.from_records_log(path).assess()
        in_memory = RecordLogCheck.from_records([_ONE_RECORD]).assess()

        self.assertIs(from_file.verdict, in_memory.verdict)
        self.assertEqual(from_file.render(), in_memory.render())
        self.assertNotEqual(from_file.source, in_memory.source, "the source still names each")

    def test_the_factory_does_not_read_the_file(self) -> None:
        """⚠ READING IS THE PART THAT FAILS, so it belongs in the verb. A factory that read would
        move the failure to construction, where the caller has no report to put it in."""
        with tempfile.TemporaryDirectory() as tmp:
            check = RecordLogCheck.from_records_log(Path(tmp) / "nope.jsonl")  # must not raise
            self.assertIs(check.assess().verdict, ReadinessVerdict.UNREADABLE)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
