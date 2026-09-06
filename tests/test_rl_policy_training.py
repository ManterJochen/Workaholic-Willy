"""Training one offline policy from Python, with the parts that lived only in the CLI.

⛔⛔ **WHAT WAS TRAPPED, AND IT IS NOT THE TRAINER.** The three trainers are already public, already
pure, and already take a path. What no Python caller could reach was everything around them: turning
a DATASET ID into a manifest path, the typed refusal when that file is absent, the exit code that
refusal carries, the default output location, and the summary that names the artifact and its hash.
Same gap as `check-dataset`: the function was public and the CONTEXT was not.

⛔⛔ **AND THE FIVE TRAINING COMMANDS ARE NOT FIVE INSTANCES OF ONE THING.** Measured by reading all
five handlers AND running them: candidate/ranking/sequencing take `--dataset-id` and exit 0/2;
perception-budget takes either source and exits 0/2/3; recovery takes packs and exits 0/2/3. This
covers the three that share a shape, and says which three.

⭐ **THE VERIFICATION THAT MATTERS IS BYTE IDENTITY.** The CLI's JSON summary was captured for all
five families BEFORE a line of this was written, and the library's output was diffed against it. The
artifact sha256 is in that summary, so an identical summary means an identical ARTIFACT, not merely
an identical report.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.contracts import Rendered, Structured
from src.robot.grasping.rl.policy_training import (
    PolicyTraining,
    PolicyTrainingReport,
    TrainingFamily,
    TrainingVerdict,
)

_REPO = Path(__file__).resolve().parent.parent
_DATASET = "v1_bootstrap"


class TheRefusalTests(unittest.TestCase):
    """⛔ THE PART THAT WAS TRAPPED. A caller who named a dataset that does not exist got, from
    Python, whatever the trainer happened to raise on a missing file."""

    def test_a_missing_manifest_is_a_verdict_not_an_exception(self) -> None:
        for family in TrainingFamily:
            with self.subTest(family):
                report = PolicyTraining.from_dataset_id(
                    family, "no_such_dataset", repo_root=_REPO).train()
                self.assertIs(report.verdict, TrainingVerdict.NO_MANIFEST)
                self.assertEqual(report.exit_code, 2)
                self.assertFalse(report.trained)

    def test_the_refusal_names_the_path_it_looked_at(self) -> None:
        """⚠ THE ABSOLUTE PATH, because that is what a library caller needs to debug. The CLI shows
        the repo-relative one because that is what an operator typed. Two audiences, one fact."""
        report = PolicyTraining.from_dataset_id(
            TrainingFamily.RANKING, "no_such_dataset", repo_root=_REPO).train()
        self.assertIn("no_such_dataset.json", report.detail)

    def test_the_factory_does_not_touch_the_disk(self) -> None:
        """⚠ RESOLUTION IS ARITHMETIC ON A STRING AND CANNOT FAIL; READING IS WHAT FAILS. A factory
        that read would move the failure to construction, where the caller has no report for it."""
        run = PolicyTraining.from_dataset_id(TrainingFamily.CANDIDATE, "no_such", repo_root=_REPO)
        self.assertFalse(run.manifest_path.is_file())
        self.assertIs(run.train().verdict, TrainingVerdict.NO_MANIFEST)


class NoDefaultsOfItsOwnTests(unittest.TestCase):
    """⛔⛔ THE NEAR MISS THIS CLASS EXISTS TO NOT REPEAT.

    The first version defaulted `seed` to 0 while every trainer declares `seed: int = 1` and every
    argparse registration declares 1. The same nominal call produced a DIFFERENT ARTIFACT HASH, and
    it was caught only because the CLI's summary had been captured before any of this was written.
    """

    def test_seed_is_unset_rather_than_a_third_literal(self) -> None:
        from src.contracts import UNSET

        run = PolicyTraining.from_dataset_id(TrainingFamily.CANDIDATE, _DATASET, repo_root=_REPO)
        self.assertIs(run.seed, UNSET, "a default here is a third declaration of one fact")

    def test_the_summary_reports_the_trainers_own_default(self) -> None:
        """⛔ READ OFF THE TRAINER'S SIGNATURE, so there is one source and drift is impossible. The
        summary must report the EFFECTIVE value, and a caller who supplied nothing still gets one."""
        import inspect

        from src.robot.grasping.rl.train_sequencing import train_sequencing_from_manifest

        expected = inspect.signature(
            train_sequencing_from_manifest).parameters["min_support_threshold"].default
        with tempfile.TemporaryDirectory() as tmp:
            report = PolicyTraining.from_dataset_id(
                TrainingFamily.SEQUENCING, _DATASET, repo_root=_REPO,
                output_path=Path(tmp) / "seq.json").train()
        self.assertEqual(report.summary["min_support_threshold"], expected)

    def test_every_trainer_declares_the_same_seed(self) -> None:
        """The fact the removed literal was competing with. If the trainers ever disagree, a caller
        omitting `seed` gets a different answer per family and nothing says so."""
        import inspect

        from src.robot.grasping.rl.train_candidate import train_from_manifest
        from src.robot.grasping.rl.train_ranking import train_ranking_from_manifest
        from src.robot.grasping.rl.train_sequencing import train_sequencing_from_manifest

        seeds = {
            inspect.signature(fn).parameters["seed"].default
            for fn in (train_from_manifest, train_ranking_from_manifest,
                       train_sequencing_from_manifest)
        }
        self.assertEqual(len(seeds), 1, f"the trainers disagree about the default seed: {seeds}")


class TheSummaryMatchesTheCliTests(unittest.TestCase):
    """⭐ THE KEY SETS ARE THE CONTRACT. Three determinism goldens shell out to the real command and
    compare its stdout, and `sort_keys=True` means a missing key changes the JSON without moving
    anything else, so a dropped field reads as one deleted line in an otherwise identical file."""

    #: Measured against the committed handlers on 2026-09-04, before any of this was written.
    EXPECTED_KEYS = {
        TrainingFamily.CANDIDATE: {
            "artifact_path", "artifact_sha256", "policy_id", "iterations", "converged",
            "num_samples", "num_positive", "final_log_loss",
        },
        TrainingFamily.RANKING: {
            "artifact_path", "artifact_sha256", "policy_id", "iterations", "converged",
            "num_pairs", "num_groups_with_pairs", "final_log_loss", "ndcg_at_1",
            "pairwise_accuracy", "ndcg_at_1_baseline", "pairwise_accuracy_baseline",
        },
        TrainingFamily.SEQUENCING: {
            "artifact_path", "artifact_sha256", "policy_id", "num_records", "num_groups",
            "num_pairs", "num_cells_observed", "num_cells_committed",
            "num_cells_below_threshold", "min_support_threshold",
        },
    }

    def test_each_family_emits_exactly_the_keys_the_cli_emitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            for family, expected in self.EXPECTED_KEYS.items():
                with self.subTest(family):
                    report = PolicyTraining.from_dataset_id(
                        family, _DATASET, repo_root=_REPO,
                        output_path=Path(tmp) / f"{family.value}.json").train()
                    self.assertIs(report.verdict, TrainingVerdict.TRAINED)
                    self.assertEqual(set(report.summary), expected)

    def test_the_cli_json_is_sorted_and_indented_as_the_handlers_printed_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = PolicyTraining.from_dataset_id(
                TrainingFamily.CANDIDATE, _DATASET, repo_root=_REPO,
                output_path=Path(tmp) / "c.json").train()
        text = report.as_cli_json()
        self.assertEqual(text, json.dumps(dict(report.summary), sort_keys=True, indent=2))
        self.assertEqual(json.loads(text), dict(report.summary))

    def test_an_output_outside_the_repo_is_reported_absolute(self) -> None:
        """⚠ `--output` MAY LIVE OUTSIDE THE REPO, a tempdir during CI being the normal case, and
        the handlers surfaced the absolute path rather than crashing on `relative_to`."""
        with tempfile.TemporaryDirectory() as tmp:
            report = PolicyTraining.from_dataset_id(
                TrainingFamily.CANDIDATE, _DATASET, repo_root=_REPO,
                output_path=Path(tmp) / "c.json").train()
        self.assertTrue(Path(report.summary["artifact_path"]).is_absolute())


class TheReportContractTests(unittest.TestCase):

    def test_it_is_rendered_and_structured(self) -> None:
        report = PolicyTrainingReport(TrainingFamily.RANKING, TrainingVerdict.NO_MANIFEST)
        self.assertIsInstance(report, Rendered)
        self.assertIsInstance(report, Structured)

    def test_render_obeys_the_contract_for_both_verdicts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            trained = PolicyTraining.from_dataset_id(
                TrainingFamily.CANDIDATE, _DATASET, repo_root=_REPO,
                output_path=Path(tmp) / "c.json").train()
        refused = PolicyTraining.from_dataset_id(
            TrainingFamily.CANDIDATE, "nope", repo_root=_REPO).train()
        for report in (trained, refused):
            with self.subTest(report.verdict):
                text = report.render()
                self.assertTrue(text.isascii())
                self.assertFalse(text.endswith("\n"))
                self.assertTrue(text.strip())

    def test_to_dict_is_json_safe(self) -> None:
        report = PolicyTraining.from_dataset_id(
            TrainingFamily.RANKING, "nope", repo_root=_REPO).train()
        payload = report.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["exit_code"], 2)


class TheFamiliesAreOnlyTheThreeThatFitTests(unittest.TestCase):
    """⚠ COVERING THREE AND SAYING WHICH THREE IS HONEST. Covering three and implying five is the
    shape this repository keeps finding and removing."""

    def test_perception_budget_and_recovery_are_not_offered_here(self) -> None:
        values = {f.value for f in TrainingFamily}
        self.assertEqual(values, {"candidate", "ranking", "sequencing"})
        self.assertNotIn("recovery", values, "recovery takes replay packs, a different contract")
        self.assertNotIn("perception_budget", values)

    def test_their_cli_handlers_are_untouched(self) -> None:
        """The two that do not fit keep their own handlers, and a later pass can lift them without
        having to undo an abstraction that was forced over them."""
        source = (_REPO / "src/robot/grasping/rl/_commands_train.py").read_text(
            encoding="utf-8")
        self.assertIn("train_perception_budget_from_packs", source)
        self.assertIn("train_recovery_from_packs", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
