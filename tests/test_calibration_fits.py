"""Fitting a calibration through the noun, and the two doors that used to disagree in silence.

Every number here was measured on 2026-09-04 against `tests/data/uncertainty_replay.jsonl` and the
committed promotion artifact, not assumed.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.calibration import (
    FitVerdict,
    LabelPolicy,
    UncertaintyFit,
    fit_uncertainty_calibration,
)
from src.robot.grasping.uncertainty import (
    UncertaintyChannel,
    UncertaintyMonotoneMap,
)

_REPO = Path(__file__).resolve().parents[1]
_FIXTURE = _REPO / "tests" / "data" / "uncertainty_replay.jsonl"


def _records() -> list[dict]:
    return [json.loads(line) for line in _FIXTURE.read_text().splitlines() if line.strip()]


class TwoDoorsOneAnswerTests(unittest.TestCase):
    def test_the_two_doors_used_to_produce_different_calibrations_from_the_same_file(self) -> None:
        """⛔⛔ THE MEASUREMENT THIS MODULE EXISTS FOR. Same 30 records, same package, two entry
        points, two different artifacts and nothing said so:

            CLI  (`main`)                      -> seven FITTED maps, feasibility_margin spanning
                                                  the full [0.000, 1.000]
            library (`fit_uncertainty_calibration`) -> seven IDENTITY maps, a complete no-op

        The difference is the label rule the CLI applied inside its argparse handler. The library
        half is still exactly that no-op -- asserted here, because it is the behaviour the noun
        wraps rather than replaces -- and the noun now REFUSES instead of returning it silently.
        """
        raw = fit_uncertainty_calibration(_records())
        identity = UncertaintyMonotoneMap.identity()
        self.assertEqual(
            sum(1 for ch in UncertaintyChannel if raw.maps[ch] == identity), 7,
            "the raw fitter over an unlabelled replay still returns seven identity maps",
        )

        refused = UncertaintyFit.from_records(_records()).fit()
        self.assertIs(refused.verdict, FitVerdict.NO_LABELS)
        self.assertEqual(refused.exit_code, 3)
        self.assertFalse(refused.artifact_available)
        # The refusal names the way out, so nobody has to go looking for the policy that the CLI
        # applies without asking.
        self.assertIn(LabelPolicy.FEASIBILITY_MARGIN_FIXTURE.name, refused.detail)

    def test_the_cli_policy_reproduces_the_cli_byte_for_byte(self) -> None:
        """⭐ The other half of the same claim: naming the rule gets the CLI's artifact back, so the
        noun is the CLI's implementation and not a second one that happens to agree today."""
        report = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        self.assertIs(report.verdict, FitVerdict.FITTED_FROM_INVENTED_LABELS)
        self.assertEqual(report.invented_labels, 30)
        assert report.calibration is not None

        with tempfile.TemporaryDirectory() as tmp:
            through_noun = Path(tmp) / "noun.json"
            report.write(through_noun)
            through_cli = Path(tmp) / "cli.json"
            proc = subprocess.run(
                [sys.executable, "-m",
                 "src.robot.grasping.calibration.uncertainty_calibration",
                 "--replay", str(_FIXTURE), "--out", str(through_cli)],
                cwd=_REPO, capture_output=True, text=True,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertEqual(proc.stdout.strip(), f"wrote {through_cli}")
            self.assertEqual(through_cli.read_bytes(), through_noun.read_bytes())

    def test_the_records_handed_in_are_not_written_into(self) -> None:
        """⛔ `main` set `label` on every record dict IN PLACE. Invisible when the caller is argparse
        and a surprise when the caller owns the list -- a second fit over the same list would have
        taken a different branch, because the records now carry labels."""
        records = _records()
        UncertaintyFit.from_records(
            records, label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        self.assertEqual([r for r in records if "label" in r], [])


class IdentityHasTwoCausesTests(unittest.TestCase):
    def test_a_perfect_fit_and_an_empty_channel_produce_the_same_map(self) -> None:
        """⛔⛔ THE CONTROL THAT MAKES `identity` USELESS ON ITS OWN, and the reason the report
        carries the sample count beside it. A channel whose values separate the labels perfectly
        fits to breakpoints (0, 1) values (0, 1), which IS `UncertaintyMonotoneMap.identity()`. The
        best possible outcome and the total absence of data are the same object.
        """
        perfect = [
            {"depth_confidence": 0.0, "label": 0.0},
            {"depth_confidence": 1.0, "label": 1.0},
        ]
        report = UncertaintyFit.from_records(perfect).fit()
        by_name = {c.channel: c for c in report.channels}

        depth = by_name["depth_confidence"]
        self.assertTrue(depth.identity, "a perfectly separating channel fits to the identity map")
        self.assertEqual(depth.samples, 2)
        self.assertFalse(depth.starved)
        self.assertIn("came out as the identity", depth.identity_cause)

        other = by_name["mask_confidence"]
        self.assertTrue(other.identity)
        self.assertTrue(other.starved)
        self.assertEqual(other.identity_cause, "no usable sample")

        # Both are in `identity_channels`; only one is in `starved_channels`. That is the whole
        # point -- a caller that read `identity_channels` alone would call the perfect fit a failure.
        self.assertIn("depth_confidence", report.identity_channels)
        self.assertNotIn("depth_confidence", report.starved_channels)
        self.assertIn("mask_confidence", report.starved_channels)

    def test_nothing_learned_is_decided_on_starvation_and_never_on_identity(self) -> None:
        """⭐ THE SELF-FAILING CONTROL FOR THE PREVIOUS TEST. If the verdict were computed from
        `identity` -- the obvious reading, and the one this file shipped for ten minutes -- the fit
        above would be filed as NOTHING_LEARNED while being the best fit the tool can produce."""
        perfect = [
            {"depth_confidence": 0.0, "label": 0.0},
            {"depth_confidence": 1.0, "label": 1.0},
        ]
        report = UncertaintyFit.from_records(perfect).fit()
        self.assertEqual(len(report.identity_channels), 7, "every map here IS the identity")
        self.assertIsNot(report.verdict, FitVerdict.NOTHING_LEARNED)
        self.assertIs(report.verdict, FitVerdict.FITTED)

        starved = UncertaintyFit.from_records([{"label": 1.0}, {"label": 0.0}]).fit()
        self.assertIs(starved.verdict, FitVerdict.NOTHING_LEARNED)
        self.assertEqual(len(starved.starved_channels), 7)
        self.assertEqual(starved.exit_code, 0, "the CLI has always exited 0 for this")


class InertChannelsTests(unittest.TestCase):
    def test_two_of_the_seven_fitted_maps_are_multiplied_by_zero(self) -> None:
        """⛔ MEASURED: `UncertaintyWeights()` ships topology_risk=0.0 and semantic_confidence=0.0.
        On the fixture both are real fits -- 26 and 25 usable samples -- and the runtime discards
        both. Nothing put that next to the fit before, which is where somebody would look."""
        report = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        self.assertEqual(
            sorted(report.inert_channels), ["semantic_confidence", "topology_risk"],
        )
        by_name = {c.channel: c for c in report.channels}
        self.assertEqual(by_name["topology_risk"].samples, 26)
        self.assertEqual(by_name["semantic_confidence"].samples, 25)
        self.assertFalse(by_name["topology_risk"].identity, "it fitted; it is just weighted away")
        self.assertIn("weight 0.0", report.render())

    def test_the_label_channel_outspans_every_other_by_a_factor(self) -> None:
        """⚠ THE TAUTOLOGY, AS A NUMBER. The labels are `feasibility_margin > 0.5`, so that channel
        is fitted against itself: span 1.000 against at most 0.350 for the other six."""
        report = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        spans = {c.channel: c.span for c in report.channels}
        self.assertAlmostEqual(spans["feasibility_margin"], 1.0, places=6)
        others = [v for k, v in spans.items() if k != "feasibility_margin"]
        self.assertLess(max(others), 0.5)
        self.assertIn("fitted against itself", report.render())


class StrictLeverTests(unittest.TestCase):
    def test_strict_rejects_what_the_cli_accepts_without_changing_the_findings(self) -> None:
        """⭐ Same records, same channels, opposite `ok`. The lever reads the findings differently;
        it does not produce different ones. Mirrors `LeakageAudit`'s strict flag."""
        lenient = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        strict = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE, strict=True,
        ).fit()

        self.assertIs(lenient.verdict, strict.verdict)
        self.assertEqual(
            [c.to_dict() for c in lenient.channels], [c.to_dict() for c in strict.channels],
        )
        self.assertTrue(lenient.ok)
        self.assertEqual(lenient.exit_code, 0)
        self.assertFalse(strict.ok)
        self.assertEqual(strict.exit_code, 3)
        self.assertTrue(
            strict.artifact_available,
            "a strict rejection still HAS the calibration; it just should not be shipped",
        )


class OneBuilderTests(unittest.TestCase):
    def test_the_file_door_declares_no_default_the_python_door_declares(self) -> None:
        """⛔ THE SEVENTH SHAPE. A default repeated on `from_jsonl` would be a second declaration of
        what `strict` and `label_policy` mean, and the two doors would drift the first time one
        moved."""
        import inspect

        from src.contracts import UNSET

        params = inspect.signature(UncertaintyFit.from_jsonl).parameters
        for name in ("label_policy", "calibration_id", "strict"):
            self.assertIs(params[name].default, UNSET, name)

    def test_a_missing_replay_is_a_verdict_rather_than_an_exception(self) -> None:
        report = UncertaintyFit.from_jsonl(_REPO / "nope.jsonl").fit()
        self.assertIs(report.verdict, FitVerdict.REPLAY_MISSING)
        self.assertEqual(report.exit_code, 2)
        self.assertFalse(report.artifact_available)
        with self.assertRaises(ValueError):
            report.write(_REPO / "unreachable.json")

    def test_a_malformed_replay_is_a_verdict_too(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.jsonl"
            bad.write_text("{not json at all\n")
            report = UncertaintyFit.from_jsonl(bad).fit()
        self.assertIs(report.verdict, FitVerdict.REPLAY_MALFORMED)
        self.assertEqual(report.exit_code, 2)


class RenderContractTests(unittest.TestCase):
    def test_render_is_ascii_whole_and_unterminated(self) -> None:
        for report in (
            UncertaintyFit.from_records(_records()).fit(),
            UncertaintyFit.from_records(
                _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
            ).fit(),
            UncertaintyFit.from_jsonl(_REPO / "nope.jsonl").fit(),
        ):
            text = report.render()
            text.encode("ascii")
            self.assertFalse(text.endswith("\n"))
            self.assertIn(report.verdict.value, text)
            self.assertIn(f"exit code       : {report.exit_code}", text)

    def test_to_dict_reuses_the_calibrations_own_serialiser(self) -> None:
        """⛔ A second serialisation of the artifact would be a second answer about bytes that are
        compared byte-for-byte elsewhere."""
        report = UncertaintyFit.from_records(
            _records(), label_policy=LabelPolicy.FEASIBILITY_MARGIN_FIXTURE,
        ).fit()
        assert report.calibration is not None
        self.assertEqual(report.to_dict()["artifact"], report.calibration.to_artifact())


class PromotionReportRenderTests(unittest.TestCase):
    """`render()` on the report that was already frozen, already on disk, and already SHA-chained."""

    def _report(self):
        from src.robot.grasping.calibration.model_promotion import load_promotion_report

        return load_promotion_report(_REPO / "assets" / "models" / "success_probability" / "v1")

    def test_every_rendered_number_comes_from_the_payload(self) -> None:
        """⛔ THE ONE RULE FOR THIS REPORT. `promotion.json` is verified by re-hashing the files it
        names, so a render that recomputed a metric and disagreed by a float would describe an
        artifact that does not exist. Each line is checked against `to_dict()` rather than against
        a literal."""
        report = self._report()
        payload = report.to_dict()
        text = report.render()

        self.assertIn(payload["artifact_sha256"], text)
        self.assertIn(payload["model_json_sha256"], text)
        self.assertIn(payload["manifest_json_sha256"], text)
        self.assertIn(payload["validation"]["dataset_sha256"], text)
        self.assertIn(payload["promoted_by"], text)
        self.assertIn(payload["tooling_version"], text)
        for name, value in payload["metrics"].items():
            self.assertIn(f"{value:.4f}", text, name)

    def test_a_metric_with_no_gate_threshold_is_rendered_without_one(self) -> None:
        """⚠ MEASURED: the gate bounds `brier` and `log_loss` and does NOT bound `ece` or `auroc`,
        which are recorded and ungated. The render says so by having no bracket, which is the honest
        rendering of "this number was not part of the decision"."""
        report = self._report()
        lines = {
            line.strip().split()[0]: line
            for line in report.render().splitlines()
            if line.startswith("    ") and not line.strip().startswith("model.json")
        }
        self.assertIn("[<=", lines["brier"])
        self.assertIn("[<=", lines["log_loss"])
        self.assertNotIn("[", lines["ece"])
        self.assertNotIn("[", lines["auroc"])

    def test_exit_code_is_the_rule_the_handler_used_to_restate(self) -> None:
        import dataclasses

        report = self._report()
        self.assertEqual(report.verdict, "pass")
        self.assertEqual(report.exit_code, 0)
        self.assertEqual(dataclasses.replace(report, verdict="fail").exit_code, 1)

    def test_render_is_ascii_whole_and_unterminated(self) -> None:
        text = self._report().render()
        text.encode("ascii")
        self.assertFalse(text.endswith("\n"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
