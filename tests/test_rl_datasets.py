"""Assembling a policy training set, and the lever that says the audit was vacuous.

Every assertion corresponds to something measured on 2026-09-04 against the committed dataset, or to
a traceback the CLI produced where a verdict belonged.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.robot.grasping.rl.datasets import (
    DatasetVerdict,
    LeakageAudit,
    LeakageVerdict,
    PolicyDataset,
)

_REPO = Path(__file__).resolve().parents[1]
_MANIFEST = "docs/baselines/rl_datasets/v1_bootstrap.json"


class StrictLeverTests(unittest.TestCase):
    def test_the_committed_dataset_passes_by_default_and_fails_under_strict(self) -> None:
        """⛔⛔ THE MEASUREMENT THAT MAKES THIS NOUN WORTH HAVING. Same manifest, byte-identical
        stdout from the CLI, opposite verdicts:

            audit-leakage --manifest <v1_bootstrap>                  exit 0
            audit-leakage --manifest <v1_bootstrap> --strict-leakage exit 3

        Strict counts a WARNED or SKIPPED audit as a failure, and every NOTE in `leakage.py` is a
        skipped audit -- its own comment says so. So this is the only thing in the repository that
        can say the committed corpus's leakage check was partly vacuous.
        """
        lenient = LeakageAudit.from_manifest(_MANIFEST, repo_root=_REPO).audit()
        strict = LeakageAudit.from_manifest(_MANIFEST, repo_root=_REPO, strict=True).audit()

        self.assertIs(lenient.verdict, LeakageVerdict.PASSED)
        self.assertEqual(lenient.exit_code, 0)
        self.assertIs(strict.verdict, LeakageVerdict.FAILED)
        self.assertEqual(strict.exit_code, 3)

        self.assertIsNotNone(lenient.findings)
        self.assertIsNotNone(strict.findings)
        assert lenient.findings is not None and strict.findings is not None
        self.assertEqual(
            [f.severity for f in lenient.findings.findings],
            [f.severity for f in strict.findings.findings],
            "the FINDINGS are identical; only the lever's reading of them differs",
        )

    def test_a_note_is_named_as_a_skipped_audit_rather_than_a_footnote(self) -> None:
        """⭐ It reads as a pass everywhere except under strict, so the rendering has to say what it
        is: a check that did not happen."""
        text = LeakageAudit.from_manifest(_MANIFEST, repo_root=_REPO).audit().render()
        self.assertIn("SKIPPED", text)
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")

    def test_the_findings_object_is_wrapped_not_restated(self) -> None:
        """⛔ `leakage.LeakageReport` already IS the findings object. A second `passed` computed here
        would be a second answer to one question."""
        report = LeakageAudit.from_manifest(_MANIFEST, repo_root=_REPO).audit()
        assert report.findings is not None
        self.assertEqual(report.to_dict()["findings"], report.findings.to_json())


class VerdictsInsteadOfTracebacksTests(unittest.TestCase):
    def test_a_missing_manifest_is_exit_two(self) -> None:
        report = LeakageAudit.from_manifest("docs/baselines/rl_datasets/nope.json", repo_root=_REPO).audit()
        self.assertIs(report.verdict, LeakageVerdict.MANIFEST_MISSING)
        self.assertEqual(report.exit_code, 2)

    def test_missing_split_files_are_a_verdict_not_a_traceback(self) -> None:
        """⛔ The split files are GITIGNORED, so a fresh clone has the manifest and not the data --
        the ordinary state. It used to escape as an uncaught FileNotFoundError and an undocumented
        exit 1."""
        import json
        import shutil
        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="leak-"))
        self.addCleanup(shutil.rmtree, tmp, True)
        (tmp / "docs" / "baselines" / "rl_datasets").mkdir(parents=True)
        payload = json.loads((_REPO / _MANIFEST).read_text(encoding="utf-8"))
        (tmp / _MANIFEST).write_text(json.dumps(payload), encoding="utf-8")

        report = LeakageAudit.from_manifest(_MANIFEST, repo_root=tmp).audit()
        self.assertIs(report.verdict, LeakageVerdict.SPLITS_MISSING)
        self.assertEqual(report.exit_code, 3)
        self.assertIn("gitignored", report.detail)

    def test_ratios_that_do_not_sum_to_one_refuse_before_writing(self) -> None:
        """⛔ This escaped as a ValueError three frames down. It is not a programming error: it is a
        number an operator typed, and nothing has been written when it is caught."""
        report = PolicyDataset.from_sources(
            repo_root=_REPO, dataset_id="probe", ratios=(0.5, 0.3, 0.1)
        ).build()
        self.assertIs(report.verdict, DatasetVerdict.RATIOS_INVALID)
        self.assertEqual(report.exit_code, 2)
        self.assertFalse(report.artifacts_exist)
        self.assertIn("0.9", report.detail)


class ArtifactsExistTests(unittest.TestCase):
    def test_a_leaked_build_still_wrote_its_artifacts(self) -> None:
        """⛔⛔ EXIT 3 IS NOT A REFUSAL. The manifest and the split files were WRITTEN and the audit
        then failed. A caller reading 3 as "nothing happened" leaves a corpus on disk they believe
        was never created, and `artifacts_exist` is the property that says so.
        """
        from src.robot.grasping.rl.datasets import PolicyDatasetReport

        leaked = PolicyDatasetReport(verdict=DatasetVerdict.LEAKAGE_FAILED, dataset_id="x")
        refused = PolicyDatasetReport(verdict=DatasetVerdict.RATIOS_INVALID, dataset_id="x")
        self.assertTrue(leaked.artifacts_exist)
        self.assertEqual(leaked.exit_code, 3)
        self.assertFalse(refused.artifacts_exist)
        self.assertIn("WERE WRITTEN", leaked.render())


class OneImplementationTests(unittest.TestCase):
    def test_readiness_reuses_the_landed_noun(self) -> None:
        """⭐ ONE implementation of "is there anything to learn here". `RecordLogCheck` already
        answers it and already renders it; the CLI called `assess_records` and `format_readiness`
        directly, which is the same computation with a second presentation."""
        import ast
        import inspect
        import textwrap

        # ⚠ THE DOCSTRING IS STRIPPED, AND THIS TEST FAILED WITHOUT IT. `_readiness`'s docstring
        # EXPLAINS that the CLI used to call `format_readiness`, so a plain substring check found the
        # name in the prose describing its removal. That is the exact trap this repository wrote down
        # earlier the same day: a test that reads source fails on the repair whenever the fixing
        # comment quotes the old text.
        tree = ast.parse(textwrap.dedent(inspect.getsource(PolicyDataset._readiness)))
        func = tree.body[0]
        assert isinstance(func, ast.FunctionDef)
        body = func.body[1:] if isinstance(func.body[0], ast.Expr) else func.body
        code = " ".join(ast.dump(node) for node in body)
        self.assertIn("RecordLogCheck", code)
        self.assertNotIn("format_readiness", code)
        self.assertNotIn("assess_records", code)

    def test_nothing_is_defaulted_that_build_dataset_declares(self) -> None:
        """⛔ A copy of `seed` or `ratios` here would be a second declaration of one fact -- the shape
        that produced a different artifact SHA256 in this same package once already."""
        import inspect

        from src.contracts import UNSET

        params = inspect.signature(PolicyDataset.from_sources).parameters
        for name in ("seed", "ratios", "canonical_paths", "emit_parquet", "strict_leakage"):
            self.assertIs(params[name].default, UNSET, name)

    def test_the_audit_handler_no_longer_resolves_split_paths_itself(self) -> None:
        """⚠ SCOPED TO THE HANDLER I ACTUALLY CHANGED. My first version asserted over the whole file
        and failed, correctly: `replay-env-check` resolves the same `split_files` paths for its own
        purpose. That is a REMAINING duplicate, named here rather than hidden by a narrower grep --
        it needs the splits to fingerprint them, not to audit them, so `LeakageAudit` is not its
        tool and folding them together would be the wrong shape.
        """
        import ast
        import textwrap

        source = (
            _REPO / "src" / "robot" / "grasping" / "rl" / "_commands_dataset.py"
        ).read_text(encoding="utf-8")
        module = ast.parse(source)
        handler = next(
            n for n in module.body
            if isinstance(n, ast.FunctionDef) and n.name == "_cmd_audit_leakage"
        )
        code = ast.dump(handler)
        self.assertIn("LeakageAudit", code)
        self.assertNotIn("split_files", code)
        self.assertNotIn("run_leakage_audits", code)
        # And the duplicate that remains is real, so the test says where it is.
        self.assertIn("split_files", textwrap.dedent(source), "replay-env-check still has its own")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
