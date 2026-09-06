"""Guarded adaptation tests.

Covers:

* Schema discovery of ``runtime_mutable=True`` fields.
* Per-key bound enforcement (min/max, abs step, rel step) +
  allow-list rejection + rate-limit caps + duplicate detection.
* Determinism of plan IDs given fixed inputs.
* Every rule in :class:`RuleBasedStrategy`.
* Mode handling: ``off`` => empty changes; invalid => ValueError.
* Overlay nesting + ``invert_plan`` correctness.
* Audit JSONL byte-stable round-trip.
* Config-loader env-var hook: legacy / valid overlay / forbidden
  overlay paths.
* CLI plan/verify/apply/rollback round-trip on temporary paths.
* Baseline-report fold-in shape + KPI regression helper.
* Telemetry catalog accepts the adaptation fields.
"""

from __future__ import annotations

import io
import json
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from src.config.loader import (
    ConfigError,
    load_config,
    reload_config,
)
from src.config.schema.robot.robot_schema import RobotConfig
from src.robot.grasping.replay.adaptation import (
    ADAPTATION_MODES,
    AdaptationPlan,
    AdaptationStrategy,
    DEFAULT_RATE_LIMIT_MAX,
    MAX_RATE_LIMIT,
    MutableFieldSpec,
    ProposedChange,
    RuleBasedStrategy,
    compute_plan,
    discover_runtime_mutable_fields,
    flatten_overlay_paths,
    invert_plan,
    plan_to_overlay_mapping,
    validate_overlay_against_allowlist,
    validate_plan,
)
from src.robot.grasping.replay.adaptation_io import (
    AUDIT_SCHEMA_VERSION,
    active_overlay_path,
    append_audit_entry,
    build_audit_entry,
    find_plan_in_audit,
    iter_audit_entries,
    load_audit_entries,
    write_overlay_sidecar,
)
from src.robot.grasping.replay.baseline_report import (
    compare_kpi_deltas,
)
from src.robot.grasping.replay.telemetry_catalog import (
    EXTRA_TELEMETRY_FIELDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Schema discovery
# ---------------------------------------------------------------------------


class SchemaDiscoveryTests(unittest.TestCase):
    def test_discovers_expected_five_fields(self) -> None:
        specs = discover_runtime_mutable_fields(RobotConfig())
        keys = {s.dotted_key for s in specs}
        self.assertEqual(
            keys,
            {
                "robot.grasping.performance.breach_window_size",
                "robot.grasping.success_model.ranking_blend_weight",
                "robot.grasping.uncertainty.ranking_penalty_weight",
                "robot.grasping.uncertainty.recovery_aggressive_threshold",
                "robot.grasping.watchdog.window_size",
            },
        )

    def test_specs_are_sorted_lexicographically(self) -> None:
        specs = discover_runtime_mutable_fields(RobotConfig())
        keys = [s.dotted_key for s in specs]
        self.assertEqual(keys, sorted(keys))

    def test_runtime_types_are_int_or_float(self) -> None:
        for spec in discover_runtime_mutable_fields(RobotConfig()):
            self.assertIn(spec.runtime_type, {"int", "float"})

    def test_bounds_are_sane(self) -> None:
        for spec in discover_runtime_mutable_fields(RobotConfig()):
            self.assertLess(spec.min_value, spec.max_value)
            self.assertGreater(spec.max_abs_step, 0)
            self.assertGreater(spec.max_rel_step, 0)

    def test_no_safety_field_marked_mutable(self) -> None:
        """Safety/calibration/lifecycle keys must NEVER appear."""

        specs = discover_runtime_mutable_fields(RobotConfig())
        forbidden_substrings = (".safety.", ".calibration.", ".lifecycle.")
        for spec in specs:
            for sub in forbidden_substrings:
                self.assertNotIn(
                    sub,
                    spec.dotted_key,
                    msg=f"forbidden mutable field {spec.dotted_key!r}",
                )


# ---------------------------------------------------------------------------
# Plan composition + validation
# ---------------------------------------------------------------------------


def _make_spec(
    key: str,
    *,
    runtime_type: str = "float",
    min_value: float = 0.0,
    max_value: float = 1.0,
    max_abs_step: float = 0.1,
    max_rel_step: float = 0.5,
    current_value: float | int = 0.2,
) -> MutableFieldSpec:
    return MutableFieldSpec(
        dotted_key=key,
        runtime_type=runtime_type,
        min_value=min_value,
        max_value=max_value,
        max_abs_step=max_abs_step,
        max_rel_step=max_rel_step,
        current_value=current_value,
    )


def _bare_plan(
    *,
    changes: tuple[ProposedChange, ...] = (),
    rate_limit_max: int = DEFAULT_RATE_LIMIT_MAX,
    mode: str = "recommend_only",
) -> AdaptationPlan:
    return AdaptationPlan(
        plan_id="0" * 16,
        created_at_ns=1,
        mode=mode,
        strategy="rule_based_v1",
        source_baseline_sha=None,
        source_taxonomy_sha=None,
        changes=changes,
        rate_limit_max=rate_limit_max,
    )


class ValidationTests(unittest.TestCase):
    SPEC_KEY = "robot.x"
    SPEC = _make_spec(SPEC_KEY)

    def test_in_bounds_passes(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange(self.SPEC_KEY, 0.2, 0.25, "ok", "s"),
            )
        )
        result = validate_plan(plan, (self.SPEC,))
        self.assertTrue(result.ok, msg=str(result.issues))

    def test_out_of_min_rejected(self) -> None:
        plan = _bare_plan(
            changes=(ProposedChange(self.SPEC_KEY, 0.05, -0.05, "below", "s"),)
        )
        self.assertFalse(validate_plan(plan, (self.SPEC,)).ok)

    def test_out_of_max_rejected(self) -> None:
        plan = _bare_plan(
            changes=(ProposedChange(self.SPEC_KEY, 0.9, 1.1, "above", "s"),)
        )
        self.assertFalse(validate_plan(plan, (self.SPEC,)).ok)

    def test_abs_step_exceeded_rejected(self) -> None:
        plan = _bare_plan(
            changes=(
                # |0.5 - 0.2| = 0.3 > max_abs_step (0.1)
                ProposedChange(self.SPEC_KEY, 0.2, 0.5, "too far", "s"),
            )
        )
        self.assertFalse(validate_plan(plan, (self.SPEC,)).ok)

    def test_rel_step_exceeded_rejected(self) -> None:
        spec = _make_spec(
            self.SPEC_KEY, max_abs_step=10.0, max_rel_step=0.1
        )
        plan = _bare_plan(
            changes=(
                # |0.5/0.1 - 1| = 4.0 >> 0.1
                ProposedChange(self.SPEC_KEY, 0.1, 0.5, "ratio", "s"),
            )
        )
        self.assertFalse(validate_plan(plan, (spec,)).ok)

    def test_not_in_allowlist_rejected(self) -> None:
        plan = _bare_plan(
            changes=(ProposedChange("robot.nope", 1.0, 2.0, "x", "s"),)
        )
        result = validate_plan(plan, (self.SPEC,))
        self.assertFalse(result.ok)
        self.assertTrue(any("allow" in i.reason or "unknown" in i.reason.lower()
                            or "not in" in i.reason.lower()
                            for i in result.issues))

    def test_rate_limit_max_global_cap(self) -> None:
        # Build 4 distinct allow-listed specs.
        specs = tuple(_make_spec(f"robot.k{i}") for i in range(4))
        changes = tuple(
            ProposedChange(f"robot.k{i}", 0.2, 0.25, "ok", "s")
            for i in range(4)
        )
        plan = _bare_plan(changes=changes, rate_limit_max=3)
        self.assertFalse(validate_plan(plan, specs).ok)

    def test_max_rate_limit_hard_guard(self) -> None:
        specs = tuple(_make_spec(f"robot.k{i}") for i in range(MAX_RATE_LIMIT + 1))
        changes = tuple(
            ProposedChange(f"robot.k{i}", 0.2, 0.25, "ok", "s")
            for i in range(MAX_RATE_LIMIT + 1)
        )
        plan = _bare_plan(changes=changes, rate_limit_max=MAX_RATE_LIMIT)
        result = validate_plan(plan, specs)
        self.assertFalse(result.ok)
        self.assertTrue(
            any("MAX_RATE_LIMIT" in i.reason for i in result.issues)
        )

    def test_duplicate_key_path_rejected(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange(self.SPEC_KEY, 0.2, 0.25, "ok", "s"),
                ProposedChange(self.SPEC_KEY, 0.25, 0.3, "ok", "s"),
            )
        )
        result = validate_plan(plan, (self.SPEC,))
        self.assertFalse(result.ok)
        self.assertTrue(
            any("duplicate" in i.reason for i in result.issues)
        )

    def test_int_field_rejects_bool(self) -> None:
        spec = _make_spec(
            "robot.n", runtime_type="int", min_value=0, max_value=10,
            max_abs_step=10, max_rel_step=10,
        )
        plan = _bare_plan(
            changes=(ProposedChange("robot.n", 1, True, "bool", "s"),)
        )
        self.assertFalse(validate_plan(plan, (spec,)).ok)


# ---------------------------------------------------------------------------
# Strategy contract + determinism + each rule trigger
# ---------------------------------------------------------------------------


class StrategyTests(unittest.TestCase):
    def test_rule_based_implements_protocol(self) -> None:
        self.assertIsInstance(RuleBasedStrategy(), AdaptationStrategy)
        self.assertEqual(RuleBasedStrategy().name, "rule_based_v1")

    def test_plan_id_is_deterministic(self) -> None:
        rc = RobotConfig()
        tax = {"per_root_cause": {"occlusion_misread": {"count": 9}}}
        p1 = compute_plan(robot_config=rc, baseline={}, taxonomy=tax,
                          mode="recommend_only", now_ns=1234)
        p2 = compute_plan(robot_config=rc, baseline={}, taxonomy=tax,
                          mode="recommend_only", now_ns=1234)
        self.assertEqual(p1.plan_id, p2.plan_id)
        self.assertEqual(p1.to_dict(), p2.to_dict())

    def test_off_mode_yields_empty_changes(self) -> None:
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline={},
            taxonomy={"per_root_cause": {"occlusion_misread": {"count": 99}}},
            mode="off",
            now_ns=1,
        )
        self.assertEqual(plan.changes, ())

    def test_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_plan(
                robot_config=RobotConfig(),
                baseline={},
                taxonomy={},
                mode="bogus",
                now_ns=1,
            )

    def test_rule_breach_window_triggers_on_ranking_p95_close_to_gate(self) -> None:
        baseline = {
            "runtime_slo": {
                "p95_ranking_latency_ms": 76.0,
                "p95_ranking_latency_ms_gate": 80.0,
            }
        }
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline=baseline,
            taxonomy={},
            mode="recommend_only",
            now_ns=1,
        )
        keys = {c.key_path for c in plan.changes}
        self.assertIn("robot.grasping.performance.breach_window_size", keys)

    def test_rule_occlusion_triggers_recovery_threshold(self) -> None:
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline={},
            taxonomy={
                "per_root_cause": {"occlusion_misread": {"count": 5}}
            },
            mode="recommend_only",
            now_ns=1,
        )
        keys = {c.key_path for c in plan.changes}
        self.assertIn(
            "robot.grasping.uncertainty.recovery_aggressive_threshold",
            keys,
        )

    def test_rule_empty_air_triggers_blend_weight(self) -> None:
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline={},
            taxonomy={
                "per_root_cause": {"empty_air_grasp": {"count": 5}}
            },
            mode="recommend_only",
            now_ns=1,
        )
        keys = {c.key_path for c in plan.changes}
        self.assertIn(
            "robot.grasping.success_model.ranking_blend_weight", keys
        )

    def test_rule_slip_triggers_penalty_weight(self) -> None:
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline={},
            taxonomy={
                "per_root_cause": {"slip_after_grasp": {"count": 5}}
            },
            mode="recommend_only",
            now_ns=1,
        )
        keys = {c.key_path for c in plan.changes}
        self.assertIn(
            "robot.grasping.uncertainty.ranking_penalty_weight", keys
        )

    def test_rate_limit_clips_proposals(self) -> None:
        tax = {
            "per_root_cause": {
                "occlusion_misread": {"count": 9},
                "empty_air_grasp": {"count": 9},
                "slip_after_grasp": {"count": 9},
            }
        }
        baseline = {
            "runtime_slo": {
                "p95_ranking_latency_ms": 76.0,
                "p95_ranking_latency_ms_gate": 80.0,
            }
        }
        plan = compute_plan(
            robot_config=RobotConfig(),
            baseline=baseline,
            taxonomy=tax,
            mode="recommend_only",
            now_ns=1,
            rate_limit_max=2,
        )
        self.assertEqual(len(plan.changes), 2)


# ---------------------------------------------------------------------------
# Overlay nesting + invert_plan + allow-list flatten
# ---------------------------------------------------------------------------


class OverlayShapeTests(unittest.TestCase):
    def test_plan_to_overlay_mapping_nests(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange(
                    "robot.grasping.success_model.ranking_blend_weight",
                    0.2, 0.15, "x", "rule_based_v1",
                ),
            )
        )
        nested = plan_to_overlay_mapping(plan)
        self.assertEqual(
            nested,
            {"robot": {"grasping": {"success_model": {"ranking_blend_weight": 0.15}}}},
        )

    def test_invert_plan_swaps_values_and_links_rollback_of(self) -> None:
        original = _bare_plan(
            changes=(
                ProposedChange("robot.x", 0.2, 0.15, "x", "rule_based_v1"),
            )
        )
        rb = invert_plan(original, now_ns=42)
        self.assertEqual(rb.changes[0].current_value, 0.15)
        self.assertEqual(rb.changes[0].proposed_value, 0.2)
        self.assertTrue(rb.changes[0].source.startswith("rollback_of:"))
        self.assertIn(original.plan_id, rb.changes[0].source)
        self.assertNotEqual(rb.plan_id, original.plan_id)

    def test_flatten_overlay_paths(self) -> None:
        nested = {"robot": {"a": {"b": 1, "c": 2}, "d": 3}}
        flat = dict(flatten_overlay_paths(nested))
        self.assertEqual(flat, {"robot.a.b": 1, "robot.a.c": 2, "robot.d": 3})

    def test_validate_overlay_against_allowlist(self) -> None:
        good = {"robot": {"a": 1}}
        bad = {"robot": {"safety": {"limit": 1}}}
        forbidden_good = validate_overlay_against_allowlist(good, {"robot.a"})
        forbidden_bad = validate_overlay_against_allowlist(bad, {"robot.a"})
        self.assertEqual(forbidden_good, ())
        self.assertEqual(forbidden_bad, ("robot.safety.limit",))


# ---------------------------------------------------------------------------
# Audit JSONL round-trip
# ---------------------------------------------------------------------------


class AuditTests(unittest.TestCase):
    def test_audit_round_trip(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange("robot.x", 0.2, 0.15, "x", "rule_based_v1"),
            )
        )
        with TemporaryDirectory() as td:
            audit = Path(td) / "logs/adaptation/a.jsonl"
            append_audit_entry(
                build_audit_entry(plan, action="plan", applied=False), audit
            )
            append_audit_entry(
                build_audit_entry(plan, action="apply", applied=True), audit
            )
            loaded = load_audit_entries(audit)
        self.assertEqual([e["action"] for e in loaded], ["plan", "apply"])
        self.assertTrue(loaded[1]["applied"])
        self.assertEqual(loaded[0]["schema_version"], AUDIT_SCHEMA_VERSION)

    def test_find_plan_in_audit_returns_latest_apply(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange("robot.x", 0.2, 0.15, "x", "rule_based_v1"),
            )
        )
        with TemporaryDirectory() as td:
            audit = Path(td) / "a.jsonl"
            append_audit_entry(
                build_audit_entry(plan, action="plan", applied=False), audit
            )
            append_audit_entry(
                build_audit_entry(plan, action="apply", applied=True), audit
            )
            entry = find_plan_in_audit(plan.plan_id, audit)
        self.assertIsNotNone(entry)
        self.assertEqual(entry["action"], "apply")

    def test_invalid_audit_action_raises(self) -> None:
        plan = _bare_plan()
        with self.assertRaises(ValueError):
            build_audit_entry(plan, action="bogus", applied=False)


# ---------------------------------------------------------------------------
# Overlay sidecar write
# ---------------------------------------------------------------------------


class OverlaySidecarTests(unittest.TestCase):
    def test_write_overlay_sidecar_refreshes_active(self) -> None:
        plan = _bare_plan(
            changes=(
                ProposedChange(
                    "robot.grasping.success_model.ranking_blend_weight",
                    0.2, 0.15, "x", "rule_based_v1",
                ),
            )
        )
        with TemporaryDirectory() as td:
            overlay_dir = Path(td)
            sidecar = write_overlay_sidecar(plan, overlay_dir)
            self.assertTrue(sidecar.exists())
            active = active_overlay_path(overlay_dir)
            self.assertTrue(active.exists())
            sidecar_data = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
            active_data = yaml.safe_load(active.read_text(encoding="utf-8"))
            self.assertEqual(sidecar_data, active_data)
            self.assertEqual(
                sidecar_data,
                {"robot": {"grasping": {"success_model": {"ranking_blend_weight": 0.15}}}},
            )


# ---------------------------------------------------------------------------
# Config-loader env-var hook
# ---------------------------------------------------------------------------


class LoaderOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        # Save any pre-existing env value so we can restore it.
        self._prev = os.environ.pop("WILLY_ADAPTATION_OVERLAY", None)
        reload_config()

    def tearDown(self) -> None:
        os.environ.pop("WILLY_ADAPTATION_OVERLAY", None)
        if self._prev is not None:
            os.environ["WILLY_ADAPTATION_OVERLAY"] = self._prev
        reload_config()

    def test_no_env_is_byte_identical_legacy(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.robot.grasping.success_model.ranking_blend_weight, 0.2)

    def test_good_overlay_merges(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "good.yaml"
            path.write_text(
                yaml.safe_dump(
                    {
                        "robot": {
                            "grasping": {
                                "success_model": {
                                    "ranking_blend_weight": 0.15
                                }
                            }
                        }
                    }
                )
            )
            os.environ["WILLY_ADAPTATION_OVERLAY"] = str(path)
            reload_config()
            cfg = load_config()
            self.assertAlmostEqual(
                cfg.robot.grasping.success_model.ranking_blend_weight, 0.15
            )

    def test_forbidden_overlay_path_raises(self) -> None:
        with TemporaryDirectory() as td:
            path = Path(td) / "bad.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"robot": {"safety": {"limits": {"max_velocity_mps": 5.0}}}}
                )
            )
            os.environ["WILLY_ADAPTATION_OVERLAY"] = str(path)
            reload_config()
            with self.assertRaises(ConfigError):
                load_config()

    def test_missing_overlay_file_raises(self) -> None:
        os.environ["WILLY_ADAPTATION_OVERLAY"] = "/tmp/__adaptation_missing__.yaml"
        reload_config()
        with self.assertRaises(ConfigError):
            load_config()


# ---------------------------------------------------------------------------
# CLI smoke
# ---------------------------------------------------------------------------


class CLISmokeTests(unittest.TestCase):
    def _run_main(self, argv: list[str]) -> tuple[int, str]:
        from src.robot.grasping.replay.__main__ import main

        buf = io.StringIO()
        with redirect_stdout(buf):
            rc = main(argv)
        return rc, buf.getvalue()

    def test_plan_then_verify_then_apply_then_rollback(self) -> None:
        with TemporaryDirectory() as td:
            td_path = Path(td)
            tax = td_path / "tax.json"
            tax.write_text(json.dumps(
                {"per_root_cause": {"occlusion_misread": {"count": 9}}}
            ))
            # plan
            rc, out = self._run_main([
                "--adaptation-plan",
                "--taxonomy-in", str(tax),
                "--adaptation-mode", "apply_with_guardrails",
            ])
            self.assertEqual(rc, 0, msg=out)
            payload = json.loads(out)
            plan_dict = payload["plan"]
            plan_path = td_path / "plan.json"
            plan_path.write_text(json.dumps(plan_dict))
            # verify
            rc, out = self._run_main([
                "--adaptation-verify", str(plan_path),
            ])
            self.assertEqual(rc, 0, msg=out)
            # apply
            overlay_dir = td_path / "overlays"
            audit_path = td_path / "audit.jsonl"
            rc, out = self._run_main([
                "--adaptation-apply", str(plan_path),
                "--overlay-dir", str(overlay_dir),
                "--audit-path", str(audit_path),
            ])
            self.assertEqual(rc, 0, msg=out)
            self.assertTrue(
                (overlay_dir / "adaptation_active.yaml").exists()
            )
            self.assertTrue(audit_path.exists())
            # rollback
            rc, out = self._run_main([
                "--adaptation-rollback", plan_dict["plan_id"],
                "--overlay-dir", str(overlay_dir),
                "--audit-path", str(audit_path),
            ])
            self.assertEqual(rc, 0, msg=out)
            self.assertFalse(
                (overlay_dir / "adaptation_active.yaml").exists()
            )
            actions = [e["action"] for e in iter_audit_entries(audit_path)]
            self.assertEqual(actions, ["apply", "rollback"])

    def _plan_and_apply(self, td_path: Path, extra_apply_args: list[str]):
        tax = td_path / "tax.json"
        tax.write_text(json.dumps(
            {"per_root_cause": {"occlusion_misread": {"count": 9}}}
        ))
        rc, out = self._run_main([
            "--adaptation-plan", "--taxonomy-in", str(tax),
            "--adaptation-mode", "apply_with_guardrails",
        ])
        self.assertEqual(rc, 0, msg=out)
        plan_path = td_path / "plan.json"
        plan_path.write_text(json.dumps(json.loads(out)["plan"]))
        rc, out = self._run_main([
            "--adaptation-apply", str(plan_path),
            "--overlay-dir", str(td_path / "overlays"),
            "--audit-path", str(td_path / "audit.jsonl"),
            *extra_apply_args,
        ])
        self.assertEqual(rc, 0, msg=out)
        return json.loads(out)

    def test_apply_without_post_apply_report_is_signal_unavailable(self) -> None:
        # K3: with no --post-apply-report the after==before by construction, so the guardrail is a
        # structural no-op (it cannot trigger) and says so via post_apply_signal_available=false.
        with TemporaryDirectory() as td:
            payload = self._plan_and_apply(Path(td), [])
            ar = payload["auto_rollback"]
            self.assertFalse(ar["post_apply_signal_available"])
            self.assertFalse(ar["triggered"])
            self.assertEqual(list(ar["regressions"]), [])

    def test_post_apply_regression_triggers_auto_rollback(self) -> None:
        # K3: a REAL post-apply report whose runtime-SLO p95s regressed past tolerance triggers the
        # previously-unreachable in-place revert (active overlay cleared + a rollback audit entry). This is
        # the honest closed loop the --post-apply-report path enables — the no-op is gone once a signal exists.
        with TemporaryDirectory() as td:
            td_path = Path(td)
            worse = td_path / "worse.json"
            worse.write_text(json.dumps({"runtime_slo": {
                "p95_decision_latency_ms": 9999.0,
                "p95_ranking_latency_ms": 9999.0,
                "p95_fusion_latency_ms": 9999.0,
            }}))
            payload = self._plan_and_apply(
                td_path, ["--post-apply-report", str(worse)]
            )
            ar = payload["auto_rollback"]
            self.assertTrue(ar["post_apply_signal_available"])
            self.assertTrue(ar["triggered"], msg=str(ar))
            self.assertGreater(len(ar["regressions"]), 0)
            # the in-place revert cleared the active overlay + appended a rollback audit entry
            self.assertFalse(
                (td_path / "overlays" / "adaptation_active.yaml").exists()
            )
            actions = [
                e["action"]
                for e in iter_audit_entries(td_path / "audit.jsonl")
            ]
            self.assertEqual(actions, ["apply", "rollback"])

    def test_apply_rejects_non_guardrail_plan(self) -> None:
        with TemporaryDirectory() as td:
            td_path = Path(td)
            tax = td_path / "tax.json"
            tax.write_text(json.dumps(
                {"per_root_cause": {"occlusion_misread": {"count": 9}}}
            ))
            rc, out = self._run_main([
                "--adaptation-plan",
                "--taxonomy-in", str(tax),
                "--adaptation-mode", "recommend_only",
            ])
            payload = json.loads(out)
            plan_path = td_path / "plan.json"
            plan_path.write_text(json.dumps(payload["plan"]))
            rc, out = self._run_main([
                "--adaptation-apply", str(plan_path),
                "--overlay-dir", str(td_path / "overlays"),
                "--audit-path", str(td_path / "audit.jsonl"),
            ])
            self.assertEqual(rc, 2)
            self.assertIn("apply_with_guardrails", out)


# ---------------------------------------------------------------------------
# Baseline-report fold-in + KPI regression helper
# ---------------------------------------------------------------------------


class BaselineFoldInTests(unittest.TestCase):
    def test_baseline_report_contains_adaptation_block(self) -> None:
        from src.robot.grasping.replay.baseline_report import (
            build_baseline_report,
        )

        report = build_baseline_report(REPO_ROOT)
        self.assertIn("adaptation", report)
        block = report["adaptation"]
        self.assertEqual(block["capability_group"], "guarded_adaptation")
        for key in (
            "active_overlay_present",
            "audit_log_present",
            "total_plans",
            "total_applies",
            "total_rollbacks",
            "last_apply_plan_id",
            "last_apply_at_ns",
            "last_rollback_plan_id",
            "last_rollback_at_ns",
        ):
            self.assertIn(key, block)

    def test_compare_kpi_deltas_clean(self) -> None:
        v = compare_kpi_deltas(
            {"runtime_slo": {"p95_decision_latency_ms": 10.0}},
            {"runtime_slo": {"p95_decision_latency_ms": 10.0}},
        )
        self.assertTrue(v.ok)
        self.assertEqual(v.decision_p95_delta_ms, 0.0)

    def test_compare_kpi_deltas_detects_regression(self) -> None:
        v = compare_kpi_deltas(
            {"runtime_slo": {"p95_decision_latency_ms": 10.0}},
            {"runtime_slo": {"p95_decision_latency_ms": 15.0}},
        )
        self.assertFalse(v.ok)
        self.assertEqual(v.decision_p95_delta_ms, 5.0)
        self.assertEqual(len(v.regressions), 1)

    def test_compare_kpi_deltas_none_values_skipped(self) -> None:
        v = compare_kpi_deltas(
            {"runtime_slo": {"p95_decision_latency_ms": None}},
            {"runtime_slo": {"p95_decision_latency_ms": 15.0}},
        )
        self.assertTrue(v.ok)
        self.assertIsNone(v.decision_p95_delta_ms)


# ---------------------------------------------------------------------------
# Telemetry catalog acceptance
# ---------------------------------------------------------------------------


class TelemetryCatalogTests(unittest.TestCase):
    def test_adaptation_fields_present_in_catalog(self) -> None:
        names = {name for name, _phase, _check in EXTRA_TELEMETRY_FIELDS}
        self.assertIn("adaptation_mode", names)
        self.assertIn("adaptation_applied_changes", names)

    def test_adaptation_modes_constant(self) -> None:
        self.assertEqual(
            ADAPTATION_MODES,
            ("off", "recommend_only", "apply_with_guardrails"),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
