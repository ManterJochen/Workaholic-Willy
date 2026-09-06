"""RL extension-layer contract tests — schema and telemetry freeze.

Locks verified here:

* schema accepts the five RL product modes and the locked
  default ``hybrid_ml``;
* schema rejects every RL-active mode (``rl_shadow``/``rl_active``/
  ``rl_experimental``) unless both ``policy_id`` and
  ``artifact_path`` are set, and additionally requires
  ``experimental.enabled=true`` for ``rl_experimental``;
* default ``robot.yaml`` round-trips through the loader with
  ``rl.mode == "hybrid_ml"`` (byte-identical baseline behaviour);
* :func:`assert_rl_mode_supported` admits the deterministic
  subset and rejects every RL-active mode with a typed
  :class:`RLModeNotImplementedError` carrying the producer
  tag (carrier-only stance);
* the telemetry catalog ships exactly 13 RL fields tagged with
  ``capability_group = "rl_core"`` and validated type-only by the
  existing catalog audit;
* the :func:`audit_rl_required_record` gate enforces presence on
  every RL-active record and is a no-op for
  ``geometry_only`` / ``hybrid_ml`` / missing-mode records — so
  every canonical telemetry pack remains valid byte-for-byte.

No runtime producer is exercised here; that is out of scope for
this contract layer.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.loader import load_config
from src.config.schema.robot.robot_schema import (
    RL_ACTIVE_MODES,
    RL_MODE_GEOMETRY_ONLY,
    RL_MODE_HYBRID_ML,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    RL_MODE_RL_SHADOW,
    RL_MODE_VALUES,
    RobotConfig,
    RobotRLConfig,
)
from src.robot.grasping.telemetry.outcome_logging import GraspAttemptRecord
from src.robot.grasping.replay.telemetry_catalog import (
    RL_REQUIRED_TELEMETRY_FIELDS,
    audit_extra_record,
    audit_rl_required_record,
    audit_rl_required_records,
    extra_field_group_map,
)
from src.robot.grasping.rl import (
    RL_MODE_PRODUCER,
    RL_NEWEST_RUNTIME_MODE,
    RL_SUPPORTED_MODES_DETERMINISTIC,
    RL_TOOLING_VERSION,
    RLModeNotImplementedError,
    assert_rl_mode_supported,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_record(
    *,
    final_outcome: str = "succeeded",
    extra: dict | None = None,
) -> GraspAttemptRecord:
    return GraspAttemptRecord(
        timestamp=1.0,
        attempt_id="att-1",
        mode="auto",
        final_outcome=final_outcome,
        extra=extra or {},
    )


def _complete_rl_extra(mode: str) -> dict:
    return {
        "rl_mode": mode,
        "rl_policy_id": "pol-1",
        "rl_artifact_version": "art-1",
        "rl_action_proposed": "grasp",
        "rl_action_applied": "grasp",
        "rl_action_blocked_by_mask": False,
        "rl_reason_features": {"score": 0.7},
        "rl_confidence": 0.9,
        "rl_baseline_action": "grasp",
        "rl_override": False,
        "rl_fallback_triggered": False,
        "rl_router_path": mode,
        "rl_fallback_reason_code": "none",
    }


# ---------------------------------------------------------------------------
# Schema surface
# ---------------------------------------------------------------------------


class TestRLConfigSchema(unittest.TestCase):
    def test_default_is_hybrid_ml(self) -> None:
        """Shipped default preserves the locked baseline behaviour."""

        cfg = RobotConfig()
        self.assertEqual(cfg.rl.mode, RL_MODE_HYBRID_ML)
        self.assertIsNone(cfg.rl.policy_id)
        self.assertIsNone(cfg.rl.artifact_path)
        self.assertFalse(cfg.rl.experimental.enabled)

    def test_five_modes_locked(self) -> None:
        """The RL mode contract is exactly five values."""

        self.assertEqual(
            RL_MODE_VALUES,
            (
                "geometry_only",
                "hybrid_ml",
                "rl_shadow",
                "rl_active",
                "rl_experimental",
            ),
        )
        self.assertEqual(
            RL_ACTIVE_MODES,
            frozenset(
                {
                    RL_MODE_RL_SHADOW,
                    RL_MODE_RL_ACTIVE,
                    RL_MODE_RL_EXPERIMENTAL,
                }
            ),
        )

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            RobotRLConfig(mode="rl_offline")  # type: ignore[arg-type]

    def test_rl_active_requires_artifact(self) -> None:
        """Schema-side gate even before runtime rejection."""

        for mode in RL_ACTIVE_MODES:
            with self.subTest(mode=mode):
                with self.assertRaises(ValidationError) as ctx:
                    RobotRLConfig(mode=mode)
                msg = str(ctx.exception)
                self.assertIn("policy_id", msg)
                self.assertIn("artifact_path", msg)

    def test_rl_active_accepts_complete_block(self) -> None:
        rl = RobotRLConfig(
            mode=RL_MODE_RL_SHADOW,
            policy_id="p-1",
            artifact_path="/tmp/p-1.json",
        )
        self.assertEqual(rl.mode, RL_MODE_RL_SHADOW)

    def test_experimental_requires_enabled_flag(self) -> None:
        with self.assertRaises(ValidationError) as ctx:
            RobotRLConfig(
                mode=RL_MODE_RL_EXPERIMENTAL,
                policy_id="p-x",
                artifact_path="/tmp/p-x.json",
            )
        self.assertIn("experimental.enabled", str(ctx.exception))
        # And the positive path.
        rl = RobotRLConfig(
            mode=RL_MODE_RL_EXPERIMENTAL,
            policy_id="p-x",
            artifact_path="/tmp/p-x.json",
            experimental={"enabled": True},
        )
        self.assertTrue(rl.experimental.enabled)

    def test_extra_keys_forbidden(self) -> None:
        with self.assertRaises(ValidationError):
            RobotRLConfig(future_knob=True)  # type: ignore[call-arg]


class TestDefaultYamlRoundTrip(unittest.TestCase):
    def test_loader_default_yaml(self) -> None:
        cfg = load_config()
        self.assertEqual(cfg.robot.rl.mode, RL_MODE_HYBRID_ML)
        self.assertIsNone(cfg.robot.rl.policy_id)
        self.assertIsNone(cfg.robot.rl.artifact_path)
        self.assertFalse(cfg.robot.rl.experimental.enabled)
        # Runtime admission must succeed on the shipped default.
        assert_rl_mode_supported(cfg.robot.rl.mode)


# ---------------------------------------------------------------------------
# Runtime admission
# ---------------------------------------------------------------------------


class TestRuntimeAdmission(unittest.TestCase):
    def test_supported_subset_admitted(self) -> None:
        # The shadow-router layer promotes rl_shadow into the admitted set
        # alongside the two deterministic modes.
        for mode in (
            RL_MODE_GEOMETRY_ONLY,
            RL_MODE_HYBRID_ML,
            RL_MODE_RL_SHADOW,
        ):
            with self.subTest(mode=mode):
                assert_rl_mode_supported(mode)

    def test_supported_subset_is_exactly_two_modes(self) -> None:
        # Historical deterministic contract — still introspectable later.
        self.assertEqual(
            RL_SUPPORTED_MODES_DETERMINISTIC,
            frozenset({RL_MODE_GEOMETRY_ONLY, RL_MODE_HYBRID_ML}),
        )

    def test_rl_active_rejected_with_typed_error(self) -> None:
        # Only rl_active and rl_experimental are rejected; rl_shadow is admitted
        # (see TestRuntimeAdmission.test_supported_subset_admitted).
        producer_map = {
            RL_MODE_RL_SHADOW: "the log-only shadow router",
            RL_MODE_RL_ACTIVE: "bounded online->canary active control",
            RL_MODE_RL_EXPERIMENTAL: "the experimental policy lane",
        }
        self.assertEqual(RL_MODE_PRODUCER, producer_map)
        for mode in (RL_MODE_RL_ACTIVE, RL_MODE_RL_EXPERIMENTAL):
            with self.subTest(mode=mode):
                with self.assertRaises(RLModeNotImplementedError) as ctx:
                    assert_rl_mode_supported(mode)
                self.assertEqual(ctx.exception.mode, mode)
                self.assertEqual(ctx.exception.producer, producer_map[mode])

    def test_unknown_mode_value_error(self) -> None:
        with self.assertRaises(ValueError):
            assert_rl_mode_supported("rl_offline")

    def test_version_constants(self) -> None:
        # Two separate honest constants (never one conflated number): the newest runtime-admitted
        # mode, and the offline tooling version stamped into dataset manifests.
        self.assertEqual(RL_NEWEST_RUNTIME_MODE, "rl_shadow")
        self.assertEqual(RL_TOOLING_VERSION, "1.0")


# ---------------------------------------------------------------------------
# Telemetry catalog
# ---------------------------------------------------------------------------


class TestTelemetryCatalogRL(unittest.TestCase):
    def test_thirteen_rl_fields_registered(self) -> None:
        phase_map = extra_field_group_map()
        rl_fields = tuple(
            name for name, ph in phase_map.items() if ph == "rl_core"
        )
        self.assertEqual(len(rl_fields), 13)
        self.assertEqual(set(rl_fields), set(RL_REQUIRED_TELEMETRY_FIELDS))

    def test_rl_fields_have_stable_order(self) -> None:
        """Insertion order is part of the contract."""

        self.assertEqual(
            RL_REQUIRED_TELEMETRY_FIELDS,
            (
                "rl_mode",
                "rl_policy_id",
                "rl_artifact_version",
                "rl_action_proposed",
                "rl_action_applied",
                "rl_action_blocked_by_mask",
                "rl_reason_features",
                "rl_confidence",
                "rl_baseline_action",
                "rl_override",
                "rl_fallback_triggered",
                "rl_router_path",
                "rl_fallback_reason_code",
            ),
        )

    def test_type_only_audit_passes_when_fields_absent(self) -> None:
        """Canonical telemetry packs (no rl_* fields) keep validating."""

        rec = _make_record()
        self.assertEqual(audit_extra_record(rec), ())
        self.assertEqual(audit_rl_required_record(rec), ())

    def test_type_only_audit_flags_bad_type(self) -> None:
        rec = _make_record(extra={"rl_confidence": "high"})
        self.assertIn("rl_confidence", audit_extra_record(rec))

    def test_type_only_audit_rejects_unknown_mode_token(self) -> None:
        rec = _make_record(extra={"rl_mode": "rl_offline"})
        self.assertIn("rl_mode", audit_extra_record(rec))

    def test_rl_audit_noop_for_non_rl_active_modes(self) -> None:
        """Gate only fires for RL-active modes."""

        for mode in (RL_MODE_GEOMETRY_ONLY, RL_MODE_HYBRID_ML):
            with self.subTest(mode=mode):
                rec = _make_record(extra={"rl_mode": mode})
                self.assertEqual(audit_rl_required_record(rec), ())

    def test_rl_audit_requires_all_fields_for_rl_active(self) -> None:
        for mode in (
            RL_MODE_RL_SHADOW,
            RL_MODE_RL_ACTIVE,
            RL_MODE_RL_EXPERIMENTAL,
        ):
            with self.subTest(mode=mode):
                # Only rl_mode present -> 12 missing.
                rec = _make_record(extra={"rl_mode": mode})
                missing = audit_rl_required_record(rec)
                self.assertEqual(len(missing), len(RL_REQUIRED_TELEMETRY_FIELDS) - 1)
                self.assertNotIn("rl_mode", missing)
                # All fields present -> passes.
                rec_full = _make_record(extra=_complete_rl_extra(mode))
                self.assertEqual(audit_rl_required_record(rec_full), ())

    def test_rl_audit_flags_null_value_as_missing(self) -> None:
        extra = _complete_rl_extra(RL_MODE_RL_SHADOW)
        extra["rl_confidence"] = None
        rec = _make_record(extra=extra)
        self.assertEqual(audit_rl_required_record(rec), ("rl_confidence",))

    def test_rl_audit_batch(self) -> None:
        good = _make_record(extra=_complete_rl_extra(RL_MODE_RL_SHADOW))
        bad_extra = _complete_rl_extra(RL_MODE_RL_SHADOW)
        del bad_extra["rl_policy_id"]
        bad = GraspAttemptRecord(
            timestamp=2.0,
            attempt_id="att-2",
            mode="dense_autonomous",
            final_outcome="succeeded",
            extra=bad_extra,
        )
        violations = audit_rl_required_records([good, bad])
        self.assertEqual(violations, [("att-2", ("rl_policy_id",))])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
