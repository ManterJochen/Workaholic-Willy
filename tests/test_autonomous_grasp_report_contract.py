"""The report the owner's own sketch imitates had neither half of the contract.

⛔ MEASURED on 2026-09-04, across every report-shaped class in the in-scope tree: `render()` and
`to_dict()` were DISJOINT. Four classes had the first, two had the second, and exactly one had
neither: `AutonomousGraspReport`, the aggregate result of a pick, and the class the convention was
modelled on. That is why `backend/src/contracts/reporting.py` declares two Protocols rather than one,
and why this file exists.

⭐ **THE LAYERS LINE IS THE PART THAT EARNS THE METHOD.** Every advanced grasping block in this
repository ships `enabled: false`, so the default pick is open-loop: no AUTO decision gate, no
multi-camera fusion, no learned success model, no RL. A report that printed only an
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
from src.robot.execution.runtime_pick import PickSessionReport
from src.robot.grasping.loop.pick_loop import PickAttempt, PickOutcome


def _report(**overrides: object) -> AutonomousGraspReport:
    base = AutonomousGraspReport(
        outcome=AutonomousGraspOutcome.SUCCEEDED,
        mode=GraspMode.AUTO,
        profile=_profile_for(GraspMode.AUTO),
    )
    return replace(base, **overrides) if overrides else base


def _picked(**attempt_fields: object) -> PickSessionReport:
    """A pick loop's report of one executed attempt, carrying ``attempt_fields`` (its fused views, say)."""
    attempt = PickAttempt(attempt_index=0, target_index=0, reasons=(), score=0.9, action="executed",
                          **attempt_fields)  # type: ignore[arg-type]
    return PickSessionReport(outcome=PickOutcome.EXECUTED, robot_vendor="dummy", is_simulated=True,
                             gripper_present=False, attempts=(attempt,))


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

    def test_fusion_counts_only_where_another_camera_was_fused_into_the_object(self) -> None:
        """Read off the attempt, never off a config flag: the cameras behind the cloud the picked
        object was planned on (`fused_views`). Two of them is fusion. One camera, a neighbour fused
        while the object was not, or no pick report at all is single-view, as the fused line says."""
        fused = _report(pick_report=_picked(fused_views=("cam_left", "cam_right"), fused_objects=1))
        self.assertEqual(fused.layers_that_ran(), ("fusion",))
        self.assertIn("fusion", fused.render())
        for label, pick in (("no pick report", None), ("one view", _picked()),
                            ("a neighbour fused, not the object", _picked(fused_objects=1))):
            with self.subTest(label):
                self.assertNotIn("fusion", _report(pick_report=pick).layers_that_ran())

    def test_uncertainty_counts_only_when_a_fused_value_exists(self) -> None:
        """The default snapshot is `UncertaintySnapshot.disabled(...)`, which is present on EVERY
        report. Presence is not participation."""
        self.assertNotIn("uncertainty", _report().layers_that_ran())
        self.assertFalse(_report().uncertainty.fused_available)

    def test_recovery_counts_only_when_an_action_was_taken(self) -> None:
        self.assertNotIn("recovery", _report().layers_that_ran())
        self.assertIn("recovery", _report(recovery_actions=({"kind": "nudge"},)).layers_that_ran())

    def test_the_layers_are_reported_in_the_wire_form_too(self) -> None:
        report = _report(pick_report=_picked(fused_views=("cam_left", "cam_right"), fused_objects=1))
        self.assertEqual(report.to_dict()["layers_that_ran"], list(report.layers_that_ran()))
        self.assertEqual(report.to_dict()["layers_that_ran"], ["fusion"])


class TheRealRehearsalTests(unittest.TestCase):
    """⭐ THE MEASUREMENT, KEPT EXECUTABLE. The over-report was found by running a rehearsal and
    reading its output, not by reading the code, so the rehearsal is the regression test."""

    def test_a_built_rehearsal_cell_reports_an_open_loop_pick(self) -> None:
        from src.config.loader import load_robot_config
        from src.robot.execution.autonomous_grasp import build_rehearsal_cell

        cfg = load_robot_config(profile=None).model_copy(update={"vendor": "dummy"})
        report = build_rehearsal_cell(cfg).pick()

        self.assertEqual(report.fused_views, (), "a one-camera rehearsal fuses no second view")
        self.assertEqual(report.layers_that_ran(), (),
                         "the default pick is open-loop and the report must say so")
        self.assertIn("(none)", report.render())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
