"""Grading a calculator from Python, and the output-file rule that keeps the reference honest.

⛔⛔ **THE RULE IS A REPAIR, NOT A PRECAUTION.** `grasp_eval.jsonl` is APPENDED across runs. A
two-scene exploratory run with `--rungs sfe_fused,deep` put 964 rows from a throwaway 2-epoch model
into the file this arc quotes as its reference, plus one torn line when the run was interrupted, and
a later `--summary-only` would have reported that model as a result. The rule lived as four lines of
string arithmetic inside an argparse handler, so a Python caller either reproduced it from memory or,
far more likely, wrote into the reference file.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.eval.service import (
    REFERENCE_OUTPUT_NAME,
    EvaluationReport,
    GateReport,
    GraspEvaluation,
    TiltReport,
    evaluation_output_name,
)


class OnlyAFullCleanRunMayWriteTheReferenceTests(unittest.TestCase):

    def test_the_default_run_writes_the_reference(self) -> None:
        self.assertEqual(REFERENCE_OUTPUT_NAME, evaluation_output_name())

    def test_a_SELECTIVE_run_never_does(self) -> None:
        """The 964-row incident, as a rule. A run that graded a subset leaves rows a later summary
        cannot tell apart from a full run's."""
        name = evaluation_output_name(selective=True)
        self.assertNotEqual(REFERENCE_OUTPUT_NAME, name)
        self.assertIn("_rungs", name)

    def test_a_DIFFERENT_MASK_SOURCE_never_does(self) -> None:
        """Two runs that differ in what the calculator SAW must not share a file, or the comparison
        becomes a run against itself."""
        self.assertNotEqual(REFERENCE_OUTPUT_NAME, evaluation_output_name(mask_source="pred"))

    def test_a_MASK_COMPLETION_policy_never_does(self) -> None:
        self.assertNotEqual(REFERENCE_OUTPUT_NAME,
                            evaluation_output_name(mask_completion="fill"))

    def test_the_three_reasons_compose_rather_than_shadowing_each_other(self) -> None:
        """A run that is selective AND predicted AND filled must not collide with one that is only
        predicted: three separate reasons, three separate files."""
        names = {
            evaluation_output_name(mask_source="pred"),
            evaluation_output_name(mask_source="pred", selective=True),
            evaluation_output_name(mask_source="pred", mask_completion="fill"),
            evaluation_output_name(mask_source="pred", mask_completion="fill", selective=True),
            evaluation_output_name(mask_completion="fill"),
            evaluation_output_name(selective=True),
            evaluation_output_name(),
        }
        self.assertEqual(7, len(names), f"two runs share a file: {sorted(names)}")

    def test_every_name_is_a_grasp_eval_jsonl(self) -> None:
        """`side-approach` globs `grasp_eval*.jsonl`, so a name outside that pattern is invisible to
        the report that reads it."""
        for kwargs in ({}, {"selective": True}, {"mask_source": "pred"},
                       {"mask_completion": "fill"}):
            name = evaluation_output_name(**kwargs)                # type: ignore[arg-type]
            self.assertTrue(name.startswith("grasp_eval"), name)
            self.assertTrue(name.endswith(".jsonl"), name)


class TheLadderIsReachableFromPythonTests(unittest.TestCase):

    def test_the_computed_name_is_the_one_handed_to_the_evaluator(self) -> None:
        """⚠ ASSERTED AT THE CALL, NOT ON THE PURE FUNCTION. A rule that is correct and not wired is
        the inert-switch shape this repository fences everywhere else."""
        seen: dict[str, object] = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"rows": 1, "by_config": {}}

        with mock.patch("datagen.eval.ladder.evaluate_dataset", side_effect=capture), \
                mock.patch("datagen.eval.ladder.select_configurations", return_value=()):
            GraspEvaluation.from_dataset("d").evaluate(rungs="deep")
        self.assertEqual("grasp_eval_rungs.jsonl", seen["out_name"])

    def test_a_default_run_is_handed_the_reference_name(self) -> None:
        """The control. Without it the test above passes for a service that never writes the
        reference at all."""
        seen: dict[str, object] = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"rows": 1, "by_config": {}}

        with mock.patch("datagen.eval.ladder.evaluate_dataset", side_effect=capture), \
                mock.patch("datagen.eval.ladder.select_configurations", return_value=()):
            GraspEvaluation.from_dataset("d").evaluate()
        self.assertEqual(REFERENCE_OUTPUT_NAME, seen["out_name"])

    def test_the_limit_reaches_the_evaluator(self) -> None:
        """⛔ IT USED TO STOP AT THE HANDLER. `evaluate_dataset` has taken `limit` since the parallel
        path landed and only the tests ever passed it, so `--limit 3` was accepted, warned about
        nothing, and evaluated the whole dataset. MEASURED 2026-08-31: a run launched as a 3-scene
        smoke test returned the full 25-scene figure, identical to the previous run's, which is the
        only reason anyone noticed."""
        seen: dict[str, object] = {}

        def capture(root, **kwargs):                   # type: ignore[no-untyped-def]
            seen.update(kwargs)
            return {"rows": 1, "by_config": {}}

        with mock.patch("datagen.eval.ladder.evaluate_dataset", side_effect=capture), \
                mock.patch("datagen.eval.ladder.select_configurations", return_value=()):
            GraspEvaluation.from_dataset("d").evaluate(limit=3)
        self.assertEqual(3, seen["limit"])

    def test_last_report_evaluates_NOTHING(self) -> None:
        """⚠ THE `--summary-only` PATH IS EASY TO MISREAD. Measured 2026-08-15: it returned in
        seconds with rungs that no longer existed, which reads exactly like a fast run."""
        with mock.patch("datagen.eval.ladder.evaluate_dataset") as evaluator, \
                mock.patch("datagen.eval.ladder.summarise", return_value={"rows": 5}):
            report = GraspEvaluation.from_dataset("d").last_report()
        evaluator.assert_not_called()
        self.assertEqual("(not written)", report.out_name,
                         "a summary that names an output file reads as a run that wrote one")


class TheTiltReportFindsItsOwnRowsTests(unittest.TestCase):
    """⭐ THE FILE I/O WAS HANDLER-ONLY. `approach_tilt` offers three functions and all three take
    in-memory row sequences: there was no `..._for_dataset(path)`."""

    def _dataset(self, name: str, *, labels: bool = True) -> Path:
        root = Path(name)
        if labels:
            (root / "grasps.jsonl").write_text('{"object_key": "a"}\n', encoding="utf-8")
        (root / "grasp_eval.jsonl").write_text('{"row": 1}\n{"row": 2}\n', encoding="utf-8")
        (root / "grasp_eval_pred.jsonl").write_text('{"row": 3}\n', encoding="utf-8")
        return root

    def test_it_reads_every_eval_file_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as name, \
                mock.patch("datagen.eval.approach_tilt.object_reach_tilt", return_value={}), \
                mock.patch("datagen.eval.approach_tilt.side_approach_report", return_value={}), \
                mock.patch("datagen.eval.approach_tilt.format_report", return_value="ok"):
            report = GraspEvaluation.from_dataset(self._dataset(name)).approach_tilt()
        self.assertEqual(3, report.rows)
        self.assertEqual(("grasp_eval.jsonl", "grasp_eval_pred.jsonl"), report.files)

    def test_naming_one_file_narrows_it(self) -> None:
        with tempfile.TemporaryDirectory() as name, \
                mock.patch("datagen.eval.approach_tilt.object_reach_tilt", return_value={}), \
                mock.patch("datagen.eval.approach_tilt.side_approach_report", return_value={}), \
                mock.patch("datagen.eval.approach_tilt.format_report", return_value="ok"):
            report = GraspEvaluation.from_dataset(self._dataset(name)).approach_tilt(
                files=["grasp_eval_pred.jsonl"])
        self.assertEqual(1, report.rows)

    def test_missing_labels_name_the_command_that_writes_them(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            with self.assertRaises(FileNotFoundError) as caught:
                GraspEvaluation.from_dataset(
                    self._dataset(name, labels=False)).approach_tilt()
        self.assertIn("label-grasps", str(caught.exception))

    def test_missing_eval_rows_name_the_command_that_writes_them(self) -> None:
        with tempfile.TemporaryDirectory() as name:
            (Path(name) / "grasps.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(FileNotFoundError) as caught:
                GraspEvaluation.from_dataset(name).approach_tilt()
        self.assertIn("eval-grasps", str(caught.exception))


class TheGatePassesOnEveryKeyNotTheMeanTests(unittest.TestCase):
    """⚠ A GATE THAT AVERAGES ITS OWN CHECKS CAN BE CARRIED BY ONE."""

    def test_one_failing_key_fails_the_gate(self) -> None:
        self.assertFalse(GateReport(deltas={"top1": {"pass": True}, "cov": {"pass": False}}).passed)

    def test_all_passing_keys_pass_it(self) -> None:
        self.assertTrue(GateReport(deltas={"top1": {"pass": True}, "cov": {"pass": True}}).passed)

    def test_a_non_verdict_entry_is_not_read_as_a_failure(self) -> None:
        """The deltas carry numbers beside the verdicts; a bare float is not a failed check."""
        self.assertTrue(GateReport(deltas={"top1": {"pass": True}, "elapsed_ms": 4120}).passed)


class TheReportsReadAsTextTests(unittest.TestCase):

    def test_every_report_renders_ascii(self) -> None:
        EvaluationReport(raw={"rows": 12, "by_config": {"sfe": {
            "precision": 0.64, "valid": 12, "candidates": 19, "coverage": 0.34,
            "objects_with_labels": 307}}}, out_name="grasp_eval.jsonl").render().encode("ascii")
        GateReport(deltas={"top1": {"pass": True}}).render().encode("ascii")
        TiltReport(text="x", rows=1, files=("a.jsonl",)).render().encode("ascii")

    def test_the_rendered_run_names_the_file_it_wrote(self) -> None:
        """A number without its file is a number a reader cannot go back to."""
        text = EvaluationReport(raw={"rows": 12, "by_config": {}},
                                out_name="grasp_eval_rungs.jsonl").render()
        self.assertIn("grasp_eval_rungs.jsonl", text)

    def test_as_dict_survives_json(self) -> None:
        json.dumps(EvaluationReport(raw={"rows": 1, "by_config": {}}, out_name="x").as_dict())
        json.dumps(GateReport(deltas={"a": {"pass": True}}).as_dict())
        json.dumps(TiltReport(text="x", rows=1, files=("a",)).as_dict())


if __name__ == "__main__":
    unittest.main()
