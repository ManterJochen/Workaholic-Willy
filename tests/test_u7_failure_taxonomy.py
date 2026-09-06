"""Phase U7 — failure taxonomy and recommendation engine tests."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.failure_taxonomy import (
    LABELED_PACK_MANIFEST_RELATIVE_PATH,
    LABELED_PACK_RELATIVE_PATH,
    RECOMMENDATIONS,
    ROOT_CAUSE_ORDER,
    TAXONOMY_VERSION,
    FailureRootCause,
    TaxonomyVerdict,
    build_labeled_pack_manifest,
    build_taxonomy_report,
    classify_record,
    load_labeled_pack,
    render_labeled_pack_jsonl,
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

    def test_taxonomy_version_is_one(self) -> None:
        self.assertEqual(TAXONOMY_VERSION, 1)

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


class LabeledPackTests(unittest.TestCase):
    def test_pack_file_present_on_disk(self) -> None:
        pack = REPO_ROOT / LABELED_PACK_RELATIVE_PATH
        self.assertTrue(pack.exists(), msg=str(pack))

    def test_pack_manifest_present_on_disk(self) -> None:
        manifest = REPO_ROOT / LABELED_PACK_MANIFEST_RELATIVE_PATH
        self.assertTrue(manifest.exists(), msg=str(manifest))

    def test_pack_is_byte_stable(self) -> None:
        on_disk = (REPO_ROOT / LABELED_PACK_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
        regenerated = render_labeled_pack_jsonl()
        self.assertEqual(on_disk, regenerated)

    def test_manifest_sha_matches_disk(self) -> None:
        manifest_path = REPO_ROOT / LABELED_PACK_MANIFEST_RELATIVE_PATH
        on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
        rebuilt = build_labeled_pack_manifest(REPO_ROOT)
        self.assertEqual(on_disk, rebuilt)

    def test_every_record_carries_ground_truth(self) -> None:
        records = load_labeled_pack(REPO_ROOT)
        for r in records:
            self.assertIn("expected_root_cause", r.extra, msg=r.attempt_id)
            self.assertIn(
                r.extra["expected_root_cause"],
                {c.value for c in ROOT_CAUSE_ORDER},
            )

    def test_classifier_matches_ground_truth_for_labeled_failures(
        self,
    ) -> None:
        records = load_labeled_pack(REPO_ROOT)
        mismatches: list[str] = []
        for r in records:
            expected = r.extra["expected_root_cause"]
            verdict = classify_record(r)
            # Successes are labeled "unclassified" by convention;
            # the classifier also returns UNCLASSIFIED for them.
            if verdict.primary.value != expected:
                mismatches.append(
                    f"{r.attempt_id}: expected={expected!r} "
                    f"got={verdict.primary.value!r}"
                )
        self.assertEqual(mismatches, [])


class CoverageGateTests(unittest.TestCase):
    def test_coverage_meets_95_percent_on_labeled_pack(self) -> None:
        records = load_labeled_pack(REPO_ROOT)
        report = build_taxonomy_report(records)
        self.assertGreaterEqual(
            report.coverage_fraction,
            0.95,
            msg=(
                f"coverage={report.coverage_fraction:.4f}; "
                f"classified={report.classified_failure_count}/"
                f"{report.failure_count}"
            ),
        )

    def test_unclassified_failures_are_honest(self) -> None:
        # The labeled pack intentionally seeds 2 UNCLASSIFIED
        # failures (no symptom flag); the report must reflect that
        # rather than force-classifying them.
        records = load_labeled_pack(REPO_ROOT)
        report = build_taxonomy_report(records)
        self.assertEqual(report.unclassified_failure_count, 2)

    def test_success_excluded_from_coverage(self) -> None:
        records = load_labeled_pack(REPO_ROOT)
        report = build_taxonomy_report(records)
        self.assertEqual(report.success_count, 5)
        self.assertEqual(
            report.success_count + report.failure_count,
            report.total_records,
        )


class ReportDeterminismTests(unittest.TestCase):
    def test_render_is_byte_identical_across_runs(self) -> None:
        records = load_labeled_pack(REPO_ROOT)
        a = render_report_json(build_taxonomy_report(records))
        b = render_report_json(build_taxonomy_report(records))
        self.assertEqual(a, b)

    def test_recommendations_sorted_by_count_then_enum_order(self) -> None:
        records = load_labeled_pack(REPO_ROOT)
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
        records = load_labeled_pack(REPO_ROOT)
        report = build_taxonomy_report(records)
        for entry in report.recommendations:
            self.assertNotEqual(
                entry["root_cause"], FailureRootCause.UNCLASSIFIED.value
            )


class CliIntegrationTests(unittest.TestCase):
    def test_cli_failure_taxonomy_end_to_end(self) -> None:
        out_path = REPO_ROOT / "logs" / "u7_taxonomy_cli_out.json"
        if out_path.exists():
            out_path.unlink()
        pack_path = REPO_ROOT / LABELED_PACK_RELATIVE_PATH
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
        self.assertGreaterEqual(payload["coverage_fraction"], 0.95)
        out_path.unlink()

    def test_cli_requires_out_flag(self) -> None:
        pack_path = REPO_ROOT / LABELED_PACK_RELATIVE_PATH
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
        records = load_labeled_pack(REPO_ROOT)
        report = build_taxonomy_report(records)
        out_path = REPO_ROOT / "logs" / "u7_taxonomy_roundtrip.json"
        if out_path.exists():
            out_path.unlink()
        write_report(report, out_path)
        loaded = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["taxonomy_version"], TAXONOMY_VERSION)
        self.assertEqual(
            loaded["total_records"], report.total_records
        )
        out_path.unlink()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
