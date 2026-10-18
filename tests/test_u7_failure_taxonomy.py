"""Phase U7 — failure taxonomy and recommendation engine tests.

The synthetic labeled pack the classifier was graded against left on 2026-10-16 with the synthetic soak (the
owner's cleanup); the report and the command line are checked here over a handful of records built in place.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.failure_taxonomy import (
    RECOMMENDATIONS,
    ROOT_CAUSE_ORDER,
    TAXONOMY_VERSION,
    FailureRootCause,
    TaxonomyVerdict,
    build_taxonomy_report,
    classify_record,
    render_report_json,
    write_report,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _mk_record(
    *,
    final_outcome: str,
    extra: dict | None = None,
    attempt_id: str = "test-0001",
    mode: str = "auto",
) -> GraspAttemptRecord:
    return GraspAttemptRecord.new(
        timestamp=0.0,
        attempt_id=attempt_id,
        mode=mode,
        final_outcome=final_outcome,
        extra=dict(extra or {}),
    )


class TaxonomyEnumTests(unittest.TestCase):
    def test_precedence_order_matches_design_lock(self) -> None:
        expected = (
            FailureRootCause.COLLISION_REJECTION,
            FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
            FailureRootCause.SLIP_AFTER_GRASP,
            FailureRootCause.EMPTY_AIR_GRASP,
            FailureRootCause.DEFORMABLE_MISCLASSIFICATION,
            FailureRootCause.OCCLUSION_MISREAD,
            FailureRootCause.UNCLASSIFIED,
        )
        self.assertEqual(ROOT_CAUSE_ORDER, expected)

    def test_taxonomy_version_is_three(self) -> None:
        # 2 since 2026-09-29: the recommendations were rewritten to stop advising removed features.
        # 3 the same day: the occlusion recommendation says what a wrist pick does with its looks, fused.
        self.assertEqual(TAXONOMY_VERSION, 3)

    def test_the_occlusion_recommendation_says_what_a_wrist_pick_does_with_its_looks(self) -> None:
        # Version 2 told an operator a wrist pick stops at the first look that finds something. Since the
        # wrist pick fuses its looks it stops at the first safe grasp, and one generated view is the last resort.
        text = RECOMMENDATIONS[FailureRootCause.OCCLUSION_MISREAD]
        self.assertNotIn("first look that finds something", text)
        for said in ("fuses each with the ones before", "valid and certain", "one generated view",
                     "last resort", "robot.motion.look_deg"):
            with self.subTest(said=said):
                self.assertIn(said, text)

    def test_no_recommendation_advises_a_removed_feature(self) -> None:
        # Version 1 advised the multi-view fusion volume, active_perception's attempts, the commit
        # gate's corridor radius and DENSE_AUTONOMOUS escalation, all removed on 2026-09-28/29.
        removed = (
            "fusion volume", "active_perception", "active perception", "commit gate",
            "dense_autonomous", "closed_loop", "viewpoint", "verification residual", "regrasp",
        )
        for cause, text in RECOMMENDATIONS.items():
            for phrase in removed:
                with self.subTest(cause=cause.value, phrase=phrase):
                    self.assertNotIn(phrase, text.lower())

    def test_every_key_and_mode_a_recommendation_names_exists(self) -> None:
        import re

        from src.config._schema_index import schema_index
        from src.robot.execution.autonomous_grasp.config import GraspMode

        index = schema_index()
        for cause, text in RECOMMENDATIONS.items():
            for key in re.findall(r"robot\.[a-z_.]*[a-z_]", text):
                with self.subTest(cause=cause.value, key=key):
                    self.assertTrue(
                        key in index or any(path.startswith(key + ".") for path in index),
                        f"{key} is not a key the schema accepts",
                    )
        # The control: a key the recommendations name is one the scan finds and the schema holds.
        self.assertIn("robot.motion.look_deg", RECOMMENDATIONS[FailureRootCause.OCCLUSION_MISREAD])
        self.assertIn("robot.motion.look_deg", index)
        self.assertIn("dense_clutter", {mode.value for mode in GraspMode})
        self.assertIn(
            "dense_clutter", RECOMMENDATIONS[FailureRootCause.DEFORMABLE_MISCLASSIFICATION]
        )

    def test_recommendation_table_covers_every_non_unclassified(self) -> None:
        for cause in ROOT_CAUSE_ORDER:
            if cause is FailureRootCause.UNCLASSIFIED:
                self.assertNotIn(cause, RECOMMENDATIONS)
            else:
                self.assertIn(cause, RECOMMENDATIONS)
                self.assertTrue(RECOMMENDATIONS[cause].strip())


class ClassifyRecordTests(unittest.TestCase):
    def test_succeeded_record_is_unclassified(self) -> None:
        record = _mk_record(final_outcome="succeeded")
        verdict = classify_record(record)
        self.assertIs(verdict.primary, FailureRootCause.UNCLASSIFIED)
        self.assertEqual(verdict.also_matched, ())

    def test_collision_evidence_classifies_collision(self) -> None:
        record = _mk_record(
            final_outcome="execution_failed",
            extra={"collision_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.COLLISION_REJECTION,
        )

    def test_unsafe_recovery_refused_classifies_collision(self) -> None:
        record = _mk_record(final_outcome="unsafe_recovery_refused")
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.COLLISION_REJECTION,
        )

    def test_calibration_drift_flag(self) -> None:
        record = _mk_record(
            final_outcome="decision_fail_closed",
            extra={"calibration_drift_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
        )

    def test_slip_only_on_verification_failed(self) -> None:
        # slip flag on the wrong outcome must NOT fire.
        record = _mk_record(
            final_outcome="execution_failed",
            extra={"slip_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.UNCLASSIFIED,
        )
        record_ok = _mk_record(
            final_outcome="verification_failed",
            extra={"slip_evidence": True},
        )
        self.assertIs(
            classify_record(record_ok).primary,
            FailureRootCause.SLIP_AFTER_GRASP,
        )

    def test_empty_air_only_on_verification_failed(self) -> None:
        record = _mk_record(
            final_outcome="verification_failed",
            extra={"empty_air_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.EMPTY_AIR_GRASP,
        )

    def test_deformable_classifies(self) -> None:
        record = _mk_record(
            final_outcome="execution_failed",
            extra={"deformable_misclass_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.DEFORMABLE_MISCLASSIFICATION,
        )

    def test_occlusion_classifies(self) -> None:
        record = _mk_record(
            final_outcome="no_valid_grasp",
            extra={"occlusion_misread_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.OCCLUSION_MISREAD,
        )

    def test_unmatched_failure_is_unclassified(self) -> None:
        # An execution_failed record with no symptom flag falls to
        # UNCLASSIFIED (honest, no force-fit).
        record = _mk_record(final_outcome="execution_failed")
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.UNCLASSIFIED,
        )

    def test_success_extra_flag_does_not_force_classification(self) -> None:
        # A 'succeeded' record carrying a stray symptom flag must
        # still be UNCLASSIFIED — the classifier honours the
        # final_outcome gate.
        record = _mk_record(
            final_outcome="succeeded",
            extra={"collision_evidence": True},
        )
        self.assertIs(
            classify_record(record).primary,
            FailureRootCause.UNCLASSIFIED,
        )


class PrecedenceTests(unittest.TestCase):
    def test_collision_beats_calibration_drift(self) -> None:
        record = _mk_record(
            final_outcome="execution_failed",
            extra={
                "collision_evidence": True,
                "calibration_drift_evidence": True,
            },
        )
        verdict = classify_record(record)
        self.assertIs(verdict.primary, FailureRootCause.COLLISION_REJECTION)
        self.assertEqual(
            verdict.also_matched,
            (FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,),
        )

    def test_calibration_beats_deformable(self) -> None:
        record = _mk_record(
            final_outcome="execution_failed",
            extra={
                "calibration_drift_evidence": True,
                "deformable_misclass_evidence": True,
            },
        )
        verdict = classify_record(record)
        self.assertIs(
            verdict.primary,
            FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
        )
        self.assertEqual(
            verdict.also_matched,
            (FailureRootCause.DEFORMABLE_MISCLASSIFICATION,),
        )

    def test_slip_beats_empty_air_on_verification(self) -> None:
        record = _mk_record(
            final_outcome="verification_failed",
            extra={
                "slip_evidence": True,
                "empty_air_evidence": True,
            },
        )
        verdict = classify_record(record)
        self.assertIs(verdict.primary, FailureRootCause.SLIP_AFTER_GRASP)
        self.assertEqual(
            verdict.also_matched, (FailureRootCause.EMPTY_AIR_GRASP,)
        )

    def test_also_matched_is_sorted_in_enum_order(self) -> None:
        # collision + slip + empty_air + occlusion all fire on
        # execution_failed for collision; slip/empty only fire on
        # verification_failed so swap outcome to verification_failed
        # for slip+empty test.
        record = _mk_record(
            final_outcome="execution_failed",
            extra={
                "collision_evidence": True,
                "calibration_drift_evidence": True,
                "deformable_misclass_evidence": True,
                "occlusion_misread_evidence": True,
            },
        )
        verdict = classify_record(record)
        self.assertIs(verdict.primary, FailureRootCause.COLLISION_REJECTION)
        # also_matched must be in declaration order.
        also_values = [m.value for m in verdict.also_matched]
        order = [c.value for c in ROOT_CAUSE_ORDER]
        idx_seq = [order.index(v) for v in also_values]
        self.assertEqual(idx_seq, sorted(idx_seq))


class VerdictValidationTests(unittest.TestCase):
    def test_also_matched_cannot_contain_primary(self) -> None:
        with self.assertRaises(ValueError):
            TaxonomyVerdict(
                primary=FailureRootCause.COLLISION_REJECTION,
                also_matched=(FailureRootCause.COLLISION_REJECTION,),
            )

    def test_also_matched_must_be_enum_ordered(self) -> None:
        with self.assertRaises(ValueError):
            TaxonomyVerdict(
                primary=FailureRootCause.COLLISION_REJECTION,
                also_matched=(
                    FailureRootCause.SLIP_AFTER_GRASP,
                    FailureRootCause.CALIBRATION_DRIFT_SUSPECTED,
                ),
            )


def _mixed_records() -> list[GraspAttemptRecord]:
    """A success, three classified failures (two of one cause) and one failure no flag explains."""
    return [
        _mk_record(final_outcome="succeeded", attempt_id="m-0"),
        _mk_record(final_outcome="execution_failed", extra={"collision_evidence": True}, attempt_id="m-1"),
        _mk_record(final_outcome="execution_failed", extra={"collision_evidence": True}, attempt_id="m-2"),
        _mk_record(final_outcome="verification_failed", extra={"slip_evidence": True}, attempt_id="m-3"),
        _mk_record(final_outcome="execution_failed", attempt_id="m-4"),
    ]


def _write_records(path: Path) -> Path:
    path.write_text("".join(r.to_json() + "\n" for r in _mixed_records()), encoding="utf-8")
    return path


class ReportDeterminismTests(unittest.TestCase):
    def test_render_is_byte_identical_across_runs(self) -> None:
        records = _mixed_records()
        a = render_report_json(build_taxonomy_report(records))
        b = render_report_json(build_taxonomy_report(records))
        self.assertEqual(a, b)

    def test_recommendations_sorted_by_count_then_enum_order(self) -> None:
        records = _mixed_records()
        report = build_taxonomy_report(records)
        last_count = None
        last_idx = -1
        for entry in report.recommendations:
            cnt = int(entry["count"])
            cause = FailureRootCause(entry["root_cause"])
            idx = ROOT_CAUSE_ORDER.index(cause)
            if last_count is None:
                last_count = cnt
                last_idx = idx
                continue
            self.assertLessEqual(cnt, last_count)
            if cnt == last_count:
                self.assertGreater(idx, last_idx)
            last_count = cnt
            last_idx = idx

    def test_recommendations_omit_unclassified(self) -> None:
        records = _mixed_records()
        report = build_taxonomy_report(records)
        for entry in report.recommendations:
            self.assertNotEqual(
                entry["root_cause"], FailureRootCause.UNCLASSIFIED.value
            )


class CliIntegrationTests(unittest.TestCase):
    def test_cli_failure_taxonomy_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = _write_records(Path(tmp) / "records.jsonl")
            out_path = Path(tmp) / "taxonomy.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.replay",
                    "--failure-taxonomy",
                    str(pack_path),
                    "--out",
                    str(out_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}",
            )
            self.assertTrue(out_path.exists())
            payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["taxonomy_version"], TAXONOMY_VERSION)
        self.assertAlmostEqual(payload["coverage_fraction"], 0.75)

    def test_cli_requires_out_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pack_path = _write_records(Path(tmp) / "records.jsonl")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.replay",
                    "--failure-taxonomy",
                    str(pack_path),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)


class WriteReportRoundTripTests(unittest.TestCase):
    def test_write_then_read_back(self) -> None:
        records = _mixed_records()
        report = build_taxonomy_report(records)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "u7_taxonomy_roundtrip.json"
            write_report(report, out_path)
            loaded = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["taxonomy_version"], TAXONOMY_VERSION)
        self.assertEqual(
            loaded["total_records"], report.total_records
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
