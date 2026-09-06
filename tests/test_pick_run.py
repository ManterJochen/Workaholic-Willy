"""A campaign of picks: the four things around the loop, each one a measured disagreement.

The loop itself is six lines and is not tested here beyond what the assertions below need. What is
tested is the knowledge that used to live only in a CLI's `main()`.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.execution.pick_run import (
    PassRule,
    PickAttempt,
    PickOutcome,
    PickRun,
    PickRunReport,
    Recording,
)


class _Report:
    def __init__(self, outcome: str, lift_mm: float = 0.0) -> None:
        self.outcome = outcome
        self.lift_mm = lift_mm

    def failure_summary(self) -> str:
        return "reason=no_target"


class _Service:
    """A service that answers a scripted list, and remembers what was done to it."""

    def __init__(self, outcomes: list[str]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.label: Any = "UNTOUCHED"
        self.recording: list[Any] = []

    def pick(self) -> _Report:
        outcome = self.outcomes[self.calls]
        self.calls += 1
        if outcome == "BOOM":
            raise RuntimeError("driver lost the arm")
        return _Report(outcome, lift_mm=60.0 if outcome == "succeeded" else 0.0)

    def enable_record_logging(self, path: Any, provenance: Any = None) -> None:
        self.recording.append(path)

    def set_target_label(self, label: Any) -> None:
        self.label = label


def _run(outcomes: list[str], **kw: Any) -> tuple[_Service, PickRunReport]:
    service = _Service(outcomes)
    kw.setdefault("recording", Recording.off())
    report = PickRun.from_service(service, runs=len(outcomes), **kw).execute()
    return service, report


class VerdictTests(unittest.TestCase):
    def test_the_two_shipped_rules_disagree_about_the_same_campaign(self) -> None:
        """⛔⛔ THE REASON THE RULE IS AN ARGUMENT. The runner requires unanimity; the same repository
        configures `pass_fraction: 0.8` for the sim gate. 8 of 10 is a pass under one and a failure
        under the other, and only one of the four doors in this repo said which it meant.
        """
        eight = ["succeeded"] * 8 + ["execution_failed"] * 2
        _, strict = _run(eight)
        _, lenient = _run(eight, rule=PassRule(fraction=0.8))
        self.assertFalse(strict.passed)
        self.assertEqual(strict.exit_code, 2)
        self.assertTrue(lenient.passed)
        self.assertEqual(lenient.exit_code, 0)

    def test_a_second_opinion_can_refuse_a_reported_success(self) -> None:
        """⛔ THE SIM GATE REFUSES TO TAKE `SUCCEEDED` AS EVIDENCE and measures the lift itself. The
        real cell never has, and a cell with a NullGripper reports SUCCEEDED on every run.
        """
        rule = PassRule(confirm=lambda report: getattr(report, "lift_mm", 0.0) >= 50.0)
        _, honest = _run(["succeeded"] * 3, rule=rule)
        self.assertTrue(honest.passed)

        service = _Service(["succeeded"] * 3)
        liar = PickRun.from_service(
            service, runs=3, recording=Recording.off(),
            rule=PassRule(confirm=lambda report: False),
        ).execute()
        self.assertEqual(liar.succeeded, 0)
        self.assertEqual(liar.exit_code, 2)
        self.assertIn("independently confirmed", liar.rule.render())

    def test_a_campaign_of_zero_passes(self) -> None:
        """Matching the runner: `--runs 0` connects, moves the gripper on activation, picks nothing
        and exits 0. "0 of 0 succeeded" reads like a failure and is not one."""
        service = _Service([])
        report = PickRun.from_service(service, runs=0, recording=Recording.off()).execute()
        self.assertTrue(report.passed)
        self.assertEqual(report.exit_code, 0)


class FourNumbersTests(unittest.TestCase):
    def test_an_exception_does_not_erase_what_already_worked(self) -> None:
        """⚠ THE RUNNER PRINTED `succeeded/requested` AND LOST THE MIDDLE TWO. A campaign that raised
        on attempt 3 of 10 reported nothing about the two that had worked, and nothing about the
        seven that never ran.
        """
        _, report = _run(["succeeded", "succeeded", "BOOM"] + ["succeeded"] * 7)
        self.assertEqual(report.requested, 10)
        self.assertEqual(report.attempted, 3)
        self.assertEqual(report.succeeded, 2)
        self.assertEqual(report.cancelled, 7)
        self.assertTrue(report.raised)
        self.assertEqual(report.exit_code, 3)

    def test_cancelling_reports_the_attempts_that_never_ran(self) -> None:
        stop = {"now": False}

        def should_cancel() -> bool:
            return stop["now"]

        service = _Service(["succeeded"] * 5)

        def on_attempt(attempt: PickAttempt) -> None:
            if attempt.index == 1:
                stop["now"] = True

        report = PickRun.from_service(
            service, runs=5, recording=Recording.off(),
            should_cancel=should_cancel, on_attempt=on_attempt,
        ).execute()
        self.assertEqual(report.attempted, 2)
        self.assertEqual(report.cancelled, 3)


class PerCampaignSettingsTests(unittest.TestCase):
    def test_the_target_label_is_put_back(self) -> None:
        """⛔ MEASURED IN `api/runs.py`: not clearing it meant every later run on the shared service
        kept hunting the object an earlier one had been told to find. A per-campaign setting that
        outlives its campaign is indistinguishable from a configured one."""
        service, _ = _run(["succeeded"], target_label="a red cube")
        self.assertIsNone(service.label)

    def test_the_label_is_put_back_even_when_a_pick_raises(self) -> None:
        service, report = _run(["BOOM"], target_label="a red cube")
        self.assertTrue(report.raised)
        self.assertIsNone(service.label)

    def test_recording_is_switched_on_and_off_around_the_campaign(self) -> None:
        service, report = _run(["succeeded"], recording=Recording.to_file("logs/x.jsonl"))
        self.assertEqual(service.recording, ["logs/x.jsonl", None])
        self.assertTrue(report.recording.enabled)

    def test_recording_has_no_default_at_either_factory(self) -> None:
        """⛔⛔ THE LARGEST MEANING-CHANGING PIECE OF CONTEXT IN THE AREA. The shipped tree has
        `record_log_path: null`, so this runner writes no corpus while the console writes one, on the
        same cell and the same config. "Run ten picks and measure the rate offline" therefore works
        through the browser and silently produces nothing through the CLI.
        """
        import inspect

        for factory in (PickRun.from_cell, PickRun.from_service):
            param = inspect.signature(factory).parameters["recording"]
            self.assertIs(param.default, inspect.Parameter.empty, factory.__name__)


class ReportShapeTests(unittest.TestCase):
    def test_render_is_whole_ascii_and_summary_is_the_tail_of_it(self) -> None:
        _, report = _run(["succeeded", "execution_failed"])
        text = report.render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertTrue(text.endswith(report.summary()))
        self.assertIn("run 0: succeeded", text)
        self.assertIn("rule:", text)
        self.assertIn("recording nothing", text)

    def test_to_dict_is_json_safe(self) -> None:
        import json

        _, report = _run(["succeeded", "BOOM"])
        payload = report.to_dict()
        json.dumps(payload)
        self.assertEqual(payload["exit_code"], 3)
        self.assertEqual(len(payload["attempts"]), 2)

    def test_a_service_door_does_not_claim_to_know_the_teardown(self) -> None:
        """⚠ `from_service` did not bring the cell up and must not report how it came down. `None`
        here is honest; the alternative is a report asserting a clean teardown it never saw."""
        _, report = _run(["succeeded"])
        self.assertIsNone(report.teardown)
        self.assertTrue(report.clean_teardown)


class OrderTests(unittest.TestCase):
    def test_attempts_are_announced_as_they_land(self) -> None:
        """⭐ CAUGHT BY DIFFING THE RUNNER'S OWN OUTPUT. Without the hook the `run N:` lines moved
        BELOW the log lines of every pick, which at a bench is the difference between watching a
        campaign and reading its transcript afterwards.

        ⚠ `Cell` deliberately refuses a stage hook and that is not an inconsistency: its four steps
        are separate methods, so narration is what a caller writes between two calls. This loop lives
        inside ONE verb, so there is no "between".
        """
        seen: list[int] = []
        service = _Service(["succeeded"] * 3)
        PickRun.from_service(
            service, runs=3, recording=Recording.off(),
            on_attempt=lambda a: seen.append(service.calls),
        ).execute()
        self.assertEqual(seen, [1, 2, 3], "each attempt must be announced before the next runs")

    def test_the_outcome_enum_is_not_a_copy_of_the_services(self) -> None:
        """It has one state the service's cannot: an attempt that never ran."""
        self.assertIn(PickOutcome.CANCELLED, set(PickOutcome))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
