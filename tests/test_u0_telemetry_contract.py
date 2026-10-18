"""Phase U0 — telemetry contract tests.

This test module pins the additive U+ telemetry contract. It is intentionally hostile to silent drift:

* The U+ field-name list, version, and phase-tag map are frozen
  here; any change to those constants will fail this test loudly.
* No default mutation of :class:`GraspAttemptRecord` — the schema
  version and round-trip must remain identical to Phase T7.

The canonical replay packs and the baseline report this module also pinned left on 2026-10-16 with the
synthetic soak (the owner's cleanup).
"""

from __future__ import annotations

import unittest

from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
)
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    EXTRA_TELEMETRY_FIELDS,
    EXTRA_TELEMETRY_VERSION,
    audit_record,
    audit_extra_record,
    audit_extra_records,
    extra_field_coverage,
    extra_field_names,
    extra_field_group_map,
)




# ---------------------------------------------------------------------------
# Contract: U+ telemetry field list is frozen
# ---------------------------------------------------------------------------


EXPECTED_U_PLUS_FIELDS: tuple[tuple[str, str], ...] = (
    ("fused_view_count", "multi_view_fusion"),
    ("fusion_evidence_quality", "multi_view_fusion"),
    ("multi_view_occlusion_reduced", "multi_view_fusion"),
    ("failure_taxonomy_class", "failure_taxonomy"),
    ("failure_recommendation_id", "failure_taxonomy"),
    ("attempt_wall_time_s", "latency"),
    # By owner decision, the weakest camera world behind the attempt's typed motions, written by the record
    # serializer. Covered by ``tests/test_k1_record_logging.py``.
    ("camera_world", "camera_world"),
    ("camera_world_reason", "camera_world"),
)


class UPlusTelemetryContractTests(unittest.TestCase):
    def test_version_locked(self) -> None:
        self.assertEqual(EXTRA_TELEMETRY_VERSION, 3)

    def test_field_list_frozen(self) -> None:
        actual = tuple(
            (name, phase) for name, phase, _validator in EXTRA_TELEMETRY_FIELDS
        )
        self.assertEqual(actual, EXPECTED_U_PLUS_FIELDS)

    def test_field_names_and_phase_map(self) -> None:
        self.assertEqual(
            extra_field_names(),
            tuple(name for name, _ in EXPECTED_U_PLUS_FIELDS),
        )
        self.assertEqual(
            extra_field_group_map(),
            dict(EXPECTED_U_PLUS_FIELDS),
        )

    def test_field_names_are_unique(self) -> None:
        names = [name for name, _phase, _v in EXTRA_TELEMETRY_FIELDS]
        self.assertEqual(len(names), len(set(names)))

    def test_strict_catalog_unchanged(self) -> None:
        # The Phase T7 strict catalog must keep its exact key set so
        # downstream gates keep their meaning; a retired outcome keeps its entry by its string.
        self.assertIn("decision_fail_closed", TELEMETRY_CATALOG)
        self.assertIn(
            "extra.uncertainty_score",
            TELEMETRY_CATALOG["decision_fail_closed"],
        )


# ---------------------------------------------------------------------------
# Type-only U+ audit behaviour
# ---------------------------------------------------------------------------


def _base_record(**extra) -> GraspAttemptRecord:
    return GraspAttemptRecord.new(
        timestamp=0.0,
        attempt_id="t-0",
        mode="easy",
        final_outcome="succeeded",
        execution={"outcome": "executed", "command": "MOVE_TO"},
        verification={"outcome": "passed", "score": 0.9},
        extra=extra,
    )


class UPlusAuditTypeRulesTests(unittest.TestCase):
    def test_absent_fields_are_not_flagged(self) -> None:
        record = _base_record()
        self.assertEqual(audit_extra_record(record), ())
        self.assertEqual(audit_record(record), ())

    def test_unit_interval_rejects_out_of_range(self) -> None:
        record = _base_record(fusion_evidence_quality=1.4)
        self.assertEqual(
            audit_extra_record(record),
            ("fusion_evidence_quality",),
        )

    def test_unit_interval_rejects_bool(self) -> None:
        record = _base_record(fusion_evidence_quality=True)
        self.assertIn(
            "fusion_evidence_quality",
            audit_extra_record(record),
        )

    def test_the_camera_world_fields_are_typed(self) -> None:
        from src.robot.core.camera_world import CameraWorldUse

        for use in CameraWorldUse:
            if use is CameraWorldUse.UNSTATED:
                continue
            with self.subTest(use=use.value):
                record = _base_record(camera_world=use.value, camera_world_reason="bench check")
                self.assertEqual(audit_extra_record(record), ())
        for bad in ("sideways", 3, "unstated", "PLANNED"):
            with self.subTest(bad=bad):
                self.assertEqual(audit_extra_record(_base_record(camera_world=bad)), ("camera_world",))
        self.assertEqual(audit_extra_record(_base_record(camera_world_reason=3)), ("camera_world_reason",))

    def test_non_negative_int_rejects_negative(self) -> None:
        record = _base_record(fused_view_count=-1)
        self.assertEqual(audit_extra_record(record), ("fused_view_count",))

    def test_well_typed_payload_passes(self) -> None:
        record = _base_record(
            fused_view_count=3,
            fusion_evidence_quality=0.7,
            multi_view_occlusion_reduced=True,
            failure_taxonomy_class=None,
            failure_recommendation_id=None,
            attempt_wall_time_s=1.7,
        )
        self.assertEqual(audit_extra_record(record), ())

    def test_audit_extra_records_filters_clean(self) -> None:
        clean = _base_record()
        dirty = _base_record(fusion_evidence_quality=2.0)
        offenders = audit_extra_records([clean, dirty])
        self.assertEqual(len(offenders), 1)
        self.assertEqual(offenders[0][0], dirty.attempt_id)


# ---------------------------------------------------------------------------
# Canonical packs determinism + on-disk integrity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Baseline report determinism + scope
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# GraspAttemptRecord remains byte-identical under default kwargs
# ---------------------------------------------------------------------------


class RecordDefaultsUnchangedTests(unittest.TestCase):
    def test_schema_version_unchanged(self) -> None:
        self.assertEqual(GraspAttemptRecord.SCHEMA_VERSION, 1)

    def test_default_to_dict_keys_unchanged(self) -> None:
        record = GraspAttemptRecord.new(
            timestamp=0.0,
            attempt_id="x",
            mode="easy",
            final_outcome="no_target",
        )
        keys = list(record.to_dict().keys())
        self.assertEqual(
            keys,
            [
                "schema_version",
                "timestamp",
                "attempt_id",
                "mode",
                "final_outcome",
                "profile",
                "frame",
                "target",
                "initial_grasp",
                "initial_telemetry",
                "refined_grasp",
                "refinement",
                "selected_grasp",
                "execution",
                "verification",
                "recovery_actions",
                "extra",
            ],
        )

    def test_default_extra_is_empty_mapping(self) -> None:
        record = GraspAttemptRecord.new(
            timestamp=0.0,
            attempt_id="x",
            mode="easy",
            final_outcome="no_target",
        )
        self.assertEqual(record.extra, {})


# ---------------------------------------------------------------------------
# extra_field_coverage reports per-field adoption
# ---------------------------------------------------------------------------


class UPlusFieldCoverageTests(unittest.TestCase):
    def test_empty_input_returns_zero_per_field(self) -> None:
        coverage = extra_field_coverage([])
        self.assertEqual(
            set(coverage.keys()), set(extra_field_names())
        )
        for value in coverage.values():
            self.assertEqual(value, 0.0)

    def test_coverage_counts_non_null_only(self) -> None:
        rec_a = _base_record(fusion_evidence_quality=0.5)
        rec_b = _base_record(fusion_evidence_quality=None)
        rec_c = _base_record()
        coverage = extra_field_coverage([rec_a, rec_b, rec_c])
        # Only rec_a contributed a non-null value.
        self.assertAlmostEqual(
            coverage["fusion_evidence_quality"], 1.0 / 3.0
        )
        self.assertAlmostEqual(coverage["fused_view_count"], 0.0)


# ---------------------------------------------------------------------------
# write helpers do not mutate the canonical files in this test run
# (smoke test using a tmp output for the baseline writer)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# regenerate_all + manifest writer are idempotent
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
