"""The report the owner's own sketch imitates had neither half of the contract.

⛔ MEASURED on 2026-09-04, across every report-shaped class in the in-scope tree: `render()` and
`to_dict()` were DISJOINT. Four classes had the first, two had the second, and exactly one had
neither: `AutonomousGraspReport`, the aggregate result of a pick, and the class the convention was
modelled on. That is why `backend/src/contracts/reporting.py` declares two Protocols rather than one,
and why this file exists.

⭐ **THE LAYERS LINE IS THE PART THAT EARNS THE METHOD.** Every advanced grasping block in this
repository ships `enabled: false`, so the default pick is open-loop: no AUTO decision gate, no
closed-loop refine, no fusion commit, no learned success model, no RL. A report that printed only an
outcome would read identically whether one layer fired or six.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace

from src.contracts import Rendered, Structured
from src.robot.execution.autonomous_grasp.config import GraspMode, _profile_for
from src.robot.execution.autonomous_grasp.report import (
    AutonomousGraspOutcome,
    AutonomousGraspReport,
)


def _report(**overrides: object) -> AutonomousGraspReport:
    base = AutonomousGraspReport(
        outcome=AutonomousGraspOutcome.SUCCEEDED,
        mode=GraspMode.AUTO,
        profile=_profile_for(GraspMode.AUTO),
    )
    return replace(base, **overrides) if overrides else base


class _Commit:
    """Stands in for `CommitDecision`, which is attached whenever a candidate wins."""

    def __init__(self, reason: str) -> None:
        self.reason, self.allowed = reason, True


class TheContractTests(unittest.TestCase):

    def test_it_now_satisfies_both_halves(self) -> None:
        """The one class in the tree that satisfied neither."""
        report = _report()
        self.assertIsInstance(report, Rendered)
        self.assertIsInstance(report, Structured)

    def test_render_obeys_the_three_rules(self) -> None:
        for outcome in (AutonomousGraspOutcome.SUCCEEDED,
                        AutonomousGraspOutcome.EXECUTION_FAILED,
                        AutonomousGraspOutcome.MODE_NOT_AVAILABLE):
            with self.subTest(outcome):
                text = _report(outcome=outcome).render()
                self.assertTrue(text.isascii(), "printed to a cp1252 console")
                self.assertFalse(text.endswith("\n"), "the caller owns the line break")
                self.assertTrue(text.strip())

    def test_to_dict_survives_json_without_a_custom_encoder(self) -> None:
        payload = _report().to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_to_dict_does_not_inline_the_effective_config(self) -> None:
        """⚠ IT IS THE 79-KEY TELEMETRY CONTRACT and it has its own `to_dict`. Copying it into every
        attempt record would make this dict eighty times larger than the attempt it describes."""
        self.assertNotIn("effective_config", _report().to_dict())

    def test_the_failure_line_comes_from_failure_summary(self) -> None:
        """⛔ NOT A SECOND LOOKUP. `failure_summary` exists because the CLI and the console were each
        guessing at the reason; composing it here keeps one answer rather than adding a third."""
        failed = _report(outcome=AutonomousGraspOutcome.EXECUTION_FAILED,
                         telemetry={"low_level_outcome": "approach_path_blocked"})
        self.assertIn(failed.failure_summary(), failed.render())
        self.assertEqual(failed.to_dict()["failure_summary"], failed.failure_summary())

    def test_a_successful_report_has_no_failure_line(self) -> None:
        self.assertEqual(_report().failure_summary(), "")


class TheLayersLineTests(unittest.TestCase):
    """⛔⛔ THE OVER-REPORTING THIS LINE SHIPPED WITH, AND THE READING THAT CAUGHT IT."""

    def test_a_default_open_loop_attempt_reports_no_layers(self) -> None:
        """The honest and expected answer on a default cell. Printed rather than omitted: an absent
        line reads as an oversight, an explicit one reads as a fact."""
        self.assertEqual(_report().layers_that_ran(), ())
        self.assertIn("(none)", _report().render())

    def test_a_skipped_commit_gate_does_not_count_as_having_run(self) -> None:
        """⛔⛔ THE DEFECT, IN ONE ASSERTION. `CommitDecision` is attached on EVERY pick with a
        winning candidate, and when the gate is disabled it still reports `allowed=True` with a
        `skipped_*` reason, so `commit_decision is not None` means "a candidate won", not "the gate
        ran". The first version of `layers_that_ran` tested exactly that, and a default rehearsal
        with every advanced block `enabled: false` printed "layers commit".

        MEASURED on that rehearsal: reason was `skipped_no_policy`.
        """
        for reason in ("skipped_disabled", "skipped_no_fusion", "skipped_no_policy",
                       "skipped_mode", "skipped_camera_frame"):
            with self.subTest(reason):
                report = _report(commit_decision=_Commit(reason))
                self.assertIsNotNone(report.commit_decision, "the field IS set; that is the trap")
                self.assertNotIn("commit", report.layers_that_ran())

    def test_a_gate_that_participated_does_count(self) -> None:
        for reason in ("ok", "views_below_min", "hit_fraction_below_min"):
            with self.subTest(reason):
                self.assertIn("commit", _report(commit_decision=_Commit(reason)).layers_that_ran())

    def test_an_unknown_reason_counts_as_participation(self) -> None:
        """⚠ THE ASYMMETRY IS DELIBERATE. A NEW skip reason silently reading as "it ran" is the
        failure this method is about; the opposite mistake is merely noise, so an unrecognised
        reason errs towards reporting."""
        self.assertIn("commit", _report(commit_decision=_Commit("some_future_reason")).layers_that_ran())

    def test_uncertainty_counts_only_when_a_fused_value_exists(self) -> None:
        """The default snapshot is `UncertaintySnapshot.disabled(...)`, which is present on EVERY
        report. Presence is not participation, the same trap as the commit gate."""
        self.assertNotIn("uncertainty", _report().layers_that_ran())
        self.assertFalse(_report().uncertainty.fused_available)

    def test_recovery_counts_only_when_an_action_was_taken(self) -> None:
        self.assertNotIn("recovery", _report().layers_that_ran())
        self.assertIn("recovery", _report(recovery_actions=({"kind": "nudge"},)).layers_that_ran())

    def test_the_layers_are_reported_in_the_wire_form_too(self) -> None:
        report = _report(commit_decision=_Commit("ok"))
        self.assertEqual(report.to_dict()["layers_that_ran"], list(report.layers_that_ran()))


class TheRealRehearsalTests(unittest.TestCase):
    """⭐ THE MEASUREMENT, KEPT EXECUTABLE. The over-report was found by running a rehearsal and
    reading its output, not by reading the code, so the rehearsal is the regression test."""

    def test_a_built_rehearsal_cell_reports_an_open_loop_pick(self) -> None:
        from src.config.loader import load_robot_config
        from src.robot.execution.autonomous_grasp import build_rehearsal_cell

        cfg = load_robot_config(profile=None).model_copy(update={"vendor": "dummy"})
        report = build_rehearsal_cell(cfg).pick()

        self.assertIsNotNone(report.commit_decision, "a winning candidate attaches one")
        self.assertTrue(str(report.commit_decision.reason).startswith("skipped_"),
                        "the gate did not participate on a default cell")
        self.assertEqual(report.layers_that_ran(), (),
                         "the default pick is open-loop and the report must say so")
        self.assertIn("(none)", report.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
