"""Phase U0 — telemetry/dataset/baseline contract tests.

This test module pins the additive U+ telemetry contract, the
deterministic canonical replay packs, and the baseline report
generator. It is intentionally hostile to silent drift:

* The strict :func:`audit_record` gate (Phase T7) MUST still pass on
  every canonical record. U0 is a no-behavior-change phase.
* The U+ field-name list, version, and phase-tag map are frozen
  here; any change to those constants will fail this test loudly.
* Canonical packs are byte-stable: rendering them from spec must
  match the on-disk bytes and the manifest sha256.
* The baseline report is deterministic given the canonical packs.
* No default mutation of :class:`GraspAttemptRecord` — the schema
  version and round-trip must remain identical to Phase T7.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.robot.grasping.telemetry.outcome_logging import (
    GraspAttemptRecord,
    iter_jsonl,
)
from src.robot.grasping.replay.baseline_report import (
    BASELINE_REPORT_VERSION,
    DEFAULT_REPORT_RELATIVE_PATH,
    U_PLUS_TARGET_THRESHOLDS,
    build_baseline_report,
    write_baseline_report,
)
from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    MANIFEST_VERSION,
    build_manifest,
    find_pack,
    regenerate_all,
    render_pack_jsonl,
    repo_root_from_module,
    sha256_hex,
)
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    EXTRA_TELEMETRY_FIELDS,
    EXTRA_TELEMETRY_VERSION,
    audit_record,
    audit_records,
    audit_extra_record,
    audit_extra_records,
    extra_field_coverage,
    extra_field_names,
    extra_field_group_map,
)


REPO_ROOT = repo_root_from_module()


# ---------------------------------------------------------------------------
# Contract: U+ telemetry field list is frozen
# ---------------------------------------------------------------------------


EXPECTED_U_PLUS_FIELDS: tuple[tuple[str, str], ...] = (
    ("predicted_success_probability", "success_probability"),
    ("success_probability_model_version", "success_probability"),
    ("success_probability_calibration_bin", "success_probability"),
    ("model_lifecycle_phase", "success_probability"),
    ("fused_view_count", "multi_view_fusion"),
    ("fusion_evidence_quality", "multi_view_fusion"),
    ("multi_view_occlusion_reduced", "multi_view_fusion"),
    ("failure_taxonomy_class", "failure_taxonomy"),
    ("failure_recommendation_id", "failure_taxonomy"),
    ("uncertainty_score", "uncertainty"),
    ("uncertainty_channels", "uncertainty"),
    ("uncertainty_disagreement", "uncertainty"),
    ("drift_severity", "drift_ood"),
    ("ood_flagged", "drift_ood"),
    ("degraded_mode_active", "drift_ood"),
    ("decision_latency_ms", "latency"),
    ("ranking_latency_ms", "latency"),
    ("fusion_latency_ms", "latency"),
    ("attempt_wall_time_s", "latency"),
    ("adaptation_mode", "guarded_adaptation"),
    ("adaptation_applied_changes", "guarded_adaptation"),
    # Phase V0 — RL optimisation extension layer (contract freeze).
    # Type-only at the U+ layer; the conditional "required when
    # rl_mode is RL-active" gate lives in
    # ``telemetry_catalog.audit_rl_required_record`` and is covered by
    # ``tests/test_rl_mode_contract.py``.
    ("rl_mode", "rl_core"),
    ("rl_policy_id", "rl_core"),
    ("rl_artifact_version", "rl_core"),
    ("rl_action_proposed", "rl_core"),
    ("rl_action_applied", "rl_core"),
    ("rl_action_blocked_by_mask", "rl_core"),
    ("rl_reason_features", "rl_core"),
    ("rl_confidence", "rl_core"),
    ("rl_baseline_action", "rl_core"),
    ("rl_override", "rl_core"),
    ("rl_fallback_triggered", "rl_core"),
    ("rl_router_path", "rl_core"),
    ("rl_fallback_reason_code", "rl_core"),
    # Phase V2 — candidate-selection shadow router (type-only at
    # this layer; the producer enforces non-null when rl_mode ==
    # "rl_shadow"). Covered by
    # ``tests/test_shadow_router_and_candidate_policy.py``.
    ("rl_candidate_breakdown", "rl_candidate"),
    ("rl_candidate_agreement_top1", "rl_candidate"),
    ("rl_candidate_agreement_kendall_tau", "rl_candidate"),
    ("rl_candidate_mask_total", "rl_candidate"),
    ("rl_candidate_pruned_count", "rl_candidate"),
    # Phase V3 — ranking-shadow router (type-only at this layer;
    # producer in ``BinPickingOrchestrator`` populates these only
    # when the V3 ranking policy is wired AND the pick reaches the
    # U4 blend seam). Covered by
    # ``tests/test_ranking_shadow_and_pairwise_logistic.py``.
    ("rl_ranking_policy_id", "rl_ranking"),
    ("rl_ranking_artifact_version", "rl_ranking"),
    ("rl_ranking_regret_top1", "rl_ranking"),
    ("rl_ranking_kendall_tau", "rl_ranking"),
    # Phase V4 — sequencing-shadow router (type-only at this
    # layer; producer in ``BinPickingOrchestrator`` populates
    # these only when the V4 sequencing policy is wired AND the
    # pick has at least one failure-side attempt). Covered by
    # ``tests/test_sequencing_shadow_and_lookup.py``.
    ("rl_sequencing_policy_id", "rl_sequencing"),
    ("rl_sequencing_artifact_version", "rl_sequencing"),
    ("rl_sequencing_action_proposed", "rl_sequencing"),
    ("rl_sequencing_action_baseline", "rl_sequencing"),
    ("rl_sequencing_action_agree", "rl_sequencing"),
)


class UPlusTelemetryContractTests(unittest.TestCase):
    def test_version_locked(self) -> None:
        self.assertEqual(EXTRA_TELEMETRY_VERSION, 1)
        self.assertEqual(MANIFEST_VERSION, 1)
        self.assertEqual(BASELINE_REPORT_VERSION, 1)

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
        # downstream gates keep their meaning.
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
        record = _base_record(predicted_success_probability=1.4)
        self.assertEqual(
            audit_extra_record(record),
            ("predicted_success_probability",),
        )

    def test_unit_interval_rejects_bool(self) -> None:
        record = _base_record(predicted_success_probability=True)
        self.assertIn(
            "predicted_success_probability",
            audit_extra_record(record),
        )

    def test_non_negative_int_rejects_negative(self) -> None:
        record = _base_record(fused_view_count=-1)
        self.assertEqual(audit_extra_record(record), ("fused_view_count",))

    def test_string_enum_rejects_unknown_value(self) -> None:
        record = _base_record(drift_severity="catastrophic")
        self.assertEqual(audit_extra_record(record), ("drift_severity",))

    def test_string_float_map_rejects_string_value(self) -> None:
        record = _base_record(uncertainty_channels={"depth": "high"})
        self.assertEqual(
            audit_extra_record(record),
            ("uncertainty_channels",),
        )

    def test_well_typed_payload_passes(self) -> None:
        record = _base_record(
            predicted_success_probability=0.83,
            success_probability_model_version="logistic_isotonic_v1",
            success_probability_calibration_bin=4,
            model_lifecycle_phase="shadow",
            fused_view_count=3,
            fusion_evidence_quality=0.7,
            multi_view_occlusion_reduced=True,
            failure_taxonomy_class=None,
            failure_recommendation_id=None,
            uncertainty_score=0.2,
            uncertainty_channels={"depth": 0.1, "segmentation": 0.3},
            uncertainty_disagreement=0.2,
            drift_severity="none",
            ood_flagged=False,
            degraded_mode_active=False,
            decision_latency_ms=12.5,
            ranking_latency_ms=18.0,
            fusion_latency_ms=45.0,
            attempt_wall_time_s=1.7,
            adaptation_mode="recommend_only",
            adaptation_applied_changes=[],
        )
        self.assertEqual(audit_extra_record(record), ())

    def test_audit_extra_records_filters_clean(self) -> None:
        clean = _base_record()
        dirty = _base_record(predicted_success_probability=2.0)
        offenders = audit_extra_records([clean, dirty])
        self.assertEqual(len(offenders), 1)
        self.assertEqual(offenders[0][0], dirty.attempt_id)


# ---------------------------------------------------------------------------
# Canonical packs determinism + on-disk integrity
# ---------------------------------------------------------------------------


class CanonicalPacksTests(unittest.TestCase):
    def test_all_packs_present_on_disk(self) -> None:
        for pack in CANONICAL_PACKS:
            abs_path = (REPO_ROOT / pack.relative_path).resolve()
            self.assertTrue(
                abs_path.exists(),
                msg=(
                    f"canonical pack missing: {abs_path}. "
                    "Run `python -m src.robot.grasping.replay "
                    "--regenerate-canonical`."
                ),
            )

    def test_packs_load_via_iter_jsonl(self) -> None:
        for pack in CANONICAL_PACKS:
            records = tuple(iter_jsonl(REPO_ROOT / pack.relative_path))
            self.assertGreater(len(records), 0, msg=pack.name)

    def test_packs_pass_strict_audit(self) -> None:
        """U0 strict gate: 100% canonical records pass telemetry audit."""

        for pack in CANONICAL_PACKS:
            records = tuple(iter_jsonl(REPO_ROOT / pack.relative_path))
            offenders = audit_records(records)
            self.assertEqual(offenders, [], msg=pack.name)

    def test_packs_pass_extra_type_audit(self) -> None:
        for pack in CANONICAL_PACKS:
            records = tuple(iter_jsonl(REPO_ROOT / pack.relative_path))
            self.assertEqual(audit_extra_records(records), [], msg=pack.name)

    def test_pack_bytes_match_spec_render(self) -> None:
        """Determinism: re-rendering from spec must match on-disk bytes."""

        for pack in CANONICAL_PACKS:
            abs_path = (REPO_ROOT / pack.relative_path).resolve()
            on_disk = abs_path.read_text(encoding="utf-8")
            rendered = render_pack_jsonl(pack)
            self.assertEqual(
                rendered,
                on_disk,
                msg=(
                    f"{pack.name}: on-disk bytes differ from spec render. "
                    "Did the soak generator or pack spec change? "
                    "Regenerate via "
                    "`python -m src.robot.grasping.replay "
                    "--regenerate-canonical` and commit the diff."
                ),
            )

    def test_render_is_deterministic_across_calls(self) -> None:
        for pack in CANONICAL_PACKS:
            self.assertEqual(render_pack_jsonl(pack), render_pack_jsonl(pack))

    def test_manifest_matches_packs_on_disk(self) -> None:
        manifest_path = (
            REPO_ROOT / "tests" / "data" / "replay" / "MANIFEST.json"
        ).resolve()
        self.assertTrue(manifest_path.exists())
        on_disk_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        rebuilt = build_manifest(REPO_ROOT)
        self.assertEqual(
            on_disk_manifest,
            rebuilt,
            msg=(
                "MANIFEST.json is stale. Regenerate via "
                "`python -m src.robot.grasping.replay "
                "--regenerate-canonical`."
            ),
        )

    def test_manifest_sha_matches_recomputed(self) -> None:
        manifest = build_manifest(REPO_ROOT)
        for entry in manifest["packs"]:  # type: ignore[index]
            abs_path = (REPO_ROOT / entry["path"]).resolve()
            actual = sha256_hex(abs_path.read_bytes())
            self.assertEqual(entry["sha256"], actual, msg=entry["name"])

    def test_find_pack_lookup(self) -> None:
        pack = find_pack("replay_easy_canonical_v1")
        self.assertEqual(pack.name, "replay_easy_canonical_v1")
        with self.assertRaises(KeyError):
            find_pack("does_not_exist")


# ---------------------------------------------------------------------------
# Baseline report determinism + scope
# ---------------------------------------------------------------------------


class BaselineReportTests(unittest.TestCase):
    def test_report_is_deterministic(self) -> None:
        r1 = build_baseline_report(REPO_ROOT)
        r2 = build_baseline_report(REPO_ROOT)
        # Compare canonical JSON form to catch any dict-ordering surprises.
        self.assertEqual(
            json.dumps(r1, sort_keys=True),
            json.dumps(r2, sort_keys=True),
        )

    def test_report_committed_on_disk(self) -> None:
        report_path = (REPO_ROOT / DEFAULT_REPORT_RELATIVE_PATH).resolve()
        self.assertTrue(
            report_path.exists(),
            msg=(
                f"baseline report missing: {report_path}. "
                "Run `python -m src.robot.grasping.replay "
                "--baseline-report`."
            ),
        )
        on_disk = json.loads(report_path.read_text(encoding="utf-8"))
        rebuilt = build_baseline_report(REPO_ROOT)
        self.assertEqual(on_disk, rebuilt, msg="committed baseline is stale")

    def test_report_contains_all_packs(self) -> None:
        report = build_baseline_report(REPO_ROOT)
        names = {p["name"] for p in report["packs"]}  # type: ignore[index]
        self.assertEqual(
            names, {pack.name for pack in CANONICAL_PACKS}
        )

    def test_report_zero_audit_offenders(self) -> None:
        report = build_baseline_report(REPO_ROOT)
        for pack in report["packs"]:  # type: ignore[index]
            self.assertEqual(int(pack["telemetry_offender_count"]), 0)
            self.assertEqual(int(pack["extra_type_offender_count"]), 0)

    def test_report_runtime_slo_block_is_null_at_u0(self) -> None:
        # Phase U10 promoted the three latency p95 fields from
        # ``null`` placeholders to real aggregates over the canonical
        # packs. Phase U12 then promoted ``p95_attempt_wall_time_s_by_mode``
        # from all-null to finite per-mode p95s for every mode the
        # canonical packs cover (``closed_loop`` stays ``None``
        # because no U0 canonical pack runs that mode). We assert the
        # block keeps its locked shape, the latency p95s are finite
        # non-negative floats, and the wall-time-by-mode dict has the
        # locked key set with finite floats for the modes that have
        # samples.
        report = build_baseline_report(REPO_ROOT)
        slo = report["runtime_slo"]
        self.assertEqual(slo["capability_group"], "latency")  # type: ignore[index]
        for key in (
            "p95_decision_latency_ms",
            "p95_ranking_latency_ms",
            "p95_fusion_latency_ms",
        ):
            value = slo[key]  # type: ignore[index]
            self.assertIsInstance(value, float, msg=key)
            self.assertGreaterEqual(float(value), 0.0, msg=key)
        wall_by_mode = slo["p95_attempt_wall_time_s_by_mode"]  # type: ignore[index]
        self.assertEqual(
            set(wall_by_mode.keys()),
            {"easy", "auto", "dense_clutter", "dense_autonomous", "closed_loop"},
        )
        for mode_key in ("easy", "auto", "dense_clutter", "dense_autonomous"):
            value = wall_by_mode[mode_key]
            self.assertIsInstance(value, float, msg=mode_key)
            self.assertGreater(float(value), 0.0, msg=mode_key)
        # ``closed_loop`` is not covered by any U0 canonical pack.
        self.assertIsNone(wall_by_mode["closed_loop"])

    def test_target_thresholds_match_locked_plan(self) -> None:
        # Spot-check the locked numerics from the U+ plan so anyone
        # editing the thresholds must update this test on purpose.
        self.assertEqual(
            U_PLUS_TARGET_THRESHOLDS["easy"]["false_positive_grasp_rate_max"],
            0.005,
        )
        self.assertEqual(
            U_PLUS_TARGET_THRESHOLDS["easy"]["dead_loop_rate_max"],
            0.0,
        )
        self.assertEqual(
            U_PLUS_TARGET_THRESHOLDS["calibration"]["calibration_brier_score_max"],
            0.08,
        )
        self.assertEqual(
            U_PLUS_TARGET_THRESHOLDS["runtime_slo"]["p95_decision_latency_ms_max"],
            60.0,
        )


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
        rec_a = _base_record(predicted_success_probability=0.5)
        rec_b = _base_record(predicted_success_probability=None)
        rec_c = _base_record()
        coverage = extra_field_coverage([rec_a, rec_b, rec_c])
        # Only rec_a contributed a non-null value.
        self.assertAlmostEqual(
            coverage["predicted_success_probability"], 1.0 / 3.0
        )
        self.assertAlmostEqual(coverage["fused_view_count"], 0.0)


# ---------------------------------------------------------------------------
# write helpers do not mutate the canonical files in this test run
# (smoke test using a tmp output for the baseline writer)
# ---------------------------------------------------------------------------


class BaselineWriterSmokeTests(unittest.TestCase):
    def test_write_baseline_report_to_alt_path_is_deterministic(self) -> None:
        out_rel = "logs/_test_u0_baseline_smoke.json"
        out_path = write_baseline_report(REPO_ROOT, out_relative_path=out_rel)
        try:
            first = out_path.read_text(encoding="utf-8")
            write_baseline_report(REPO_ROOT, out_relative_path=out_rel)
            second = out_path.read_text(encoding="utf-8")
            self.assertEqual(first, second)
        finally:
            if out_path.exists():
                out_path.unlink()


# ---------------------------------------------------------------------------
# regenerate_all + manifest writer are idempotent
# ---------------------------------------------------------------------------


class RegenerateIdempotenceTests(unittest.TestCase):
    def test_regenerate_all_is_byte_idempotent(self) -> None:
        # Snapshot the canonical files, regenerate, and assert no drift.
        snapshots: dict[str, str] = {}
        manifest_path = (
            REPO_ROOT / "tests" / "data" / "replay" / "MANIFEST.json"
        ).resolve()
        for pack in CANONICAL_PACKS:
            p = (REPO_ROOT / pack.relative_path).resolve()
            snapshots[str(p)] = p.read_text(encoding="utf-8")
        snapshots[str(manifest_path)] = manifest_path.read_text(encoding="utf-8")

        regenerate_all(REPO_ROOT)
        for path_str, before in snapshots.items():
            after = Path(path_str).read_text(encoding="utf-8")
            self.assertEqual(
                before, after, msg=f"regenerate drifted: {path_str}"
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
