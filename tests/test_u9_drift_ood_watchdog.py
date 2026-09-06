"""Phase U9 — focused tests for the drift + OOD watchdog stack.

These tests gate the Phase U9 subphases:

* U9-A: pure-function severity ladder + enforce logic.
* U9-B: ``GraspingWatchdogConfig`` schema validators (ladder order,
  EASY-mode rejection, mode enum).
* U9-C: new outcomes + telemetry-catalog entries.
* U9-E: ``RobotWatchdogEvent`` namespace.
* U9-F: canonical drift + OOD packs carry the expected labels and
  signal columns and pass the catalog audit.
* U9-G: watchdog KPI gates (drift precision ≥0.90 / recall ≥0.85;
  OOD precision ≥0.90 / recall ≥0.80) on the canonical packs.

Together with the standing full-suite (1266 tests pre-U9-H), these
tests pin the U9 contract end-to-end without disturbing the byte-
identical legacy default (watchdog_mode='disabled' / no snapshot).
"""

from __future__ import annotations

import json
import unittest

from src.robot.events import (
    RobotWatchdogEvent,
    RobotWatchdogEventListener,
)
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
)
from src.robot.execution.calibration_watchdog import (
    DriftMonitorBreakdown,
    DriftSeverity,
    WatchdogAction,
    WatchdogHistory,
    WatchdogMode,
    WatchdogPolicy,
    WatchdogSample,
    evaluate_drift,
    evaluate_ood,
    evaluate_watchdog,
)
from src.robot.grasping.decision import DecisionReasonCode
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.replay.canonical_datasets import (
    CANONICAL_PACKS,
    repo_root_from_module,
)
from src.robot.grasping.replay.telemetry_catalog import (
    TELEMETRY_CATALOG,
    audit_records,
    audit_extra_records,
)
from src.robot.grasping.replay.watchdog_eval import (
    BinaryKPI,
    DRIFT_PRECISION_GATE,
    DRIFT_RECALL_GATE,
    OOD_PRECISION_GATE,
    OOD_RECALL_GATE,
    build_sample_from_record,
    evaluate_drift_pack,
    evaluate_drift_pack_path,
    evaluate_ood_pack_path,
)


_REPO_ROOT = repo_root_from_module()
_DRIFT_PACK = _REPO_ROOT / "tests" / "data" / "replay" / "replay_drift_synthetic_v1.jsonl"
_OOD_PACK = _REPO_ROOT / "tests" / "data" / "replay" / "replay_ood_synthetic_v1.jsonl"


# ---------------------------------------------------------------------------
# U9-A — enums + dataclasses
# ---------------------------------------------------------------------------


class WatchdogTypeContractTests(unittest.TestCase):
    def test_drift_severity_ladder_values(self) -> None:
        self.assertEqual(
            tuple(s.value for s in DriftSeverity),
            ("none", "low", "moderate", "high", "severe"),
        )

    def test_watchdog_mode_values(self) -> None:
        self.assertEqual(
            tuple(s.value for s in WatchdogMode),
            ("disabled", "shadow", "canary", "active"),
        )

    def test_watchdog_action_values(self) -> None:
        self.assertEqual(
            tuple(s.value for s in WatchdogAction),
            ("none", "reobserve", "block_auto"),
        )

    def test_dataclasses_are_frozen(self) -> None:
        sample = WatchdogSample(ood_score=0.9)
        with self.assertRaises(Exception):
            sample.ood_score = 0.1  # type: ignore[misc]
        policy = WatchdogPolicy()
        with self.assertRaises(Exception):
            policy.window_size = 5  # type: ignore[misc]
        report = evaluate_watchdog(WatchdogHistory(()), policy)
        with self.assertRaises(Exception):
            report.enforced = True  # type: ignore[misc]

    def test_report_to_dict_shape_is_json_safe(self) -> None:
        policy = WatchdogPolicy()
        report = evaluate_watchdog(WatchdogHistory(()), policy)
        payload = report.to_dict()
        json.dumps(payload)  # must not raise
        for key in (
            "watchdog_mode",
            "drift_severity",
            "ood_severity",
            "ood_flagged",
            "aggregate_severity",
            "degraded_mode_active",
            "recommended_action",
            "enforced",
            "drift_monitors",
        ):
            self.assertIn(key, payload)
        self.assertIsInstance(payload["drift_monitors"], dict)
        self.assertEqual(
            set(payload["drift_monitors"]),
            {
                "calibration_delta",
                "depth_confidence_shift",
                "fail_closed_spike",
                "hand_eye_residual_trend",
                "verification_residual_trend",
            },
        )

    def test_sample_rejects_non_finite(self) -> None:
        with self.assertRaises(ValueError):
            WatchdogSample(ood_score=1.5)
        with self.assertRaises(ValueError):
            WatchdogSample(calibration_residual_mm=-0.1)


# ---------------------------------------------------------------------------
# Severity ladder + monitor transitions
# ---------------------------------------------------------------------------


class WatchdogEvaluatorTransitionTests(unittest.TestCase):
    """Pin the clean → moderate → high → severe progression."""

    @staticmethod
    def _hist(*samples: WatchdogSample) -> WatchdogHistory:
        return WatchdogHistory(tuple(samples))

    def test_calibration_delta_ladder(self) -> None:
        policy = WatchdogPolicy()
        # NONE: below moderate (2.0 mm)
        hist = self._hist(
            WatchdogSample(predicted_observed_calibration_delta_mm=0.5)
        )
        self.assertIs(
            evaluate_drift(hist, policy).calibration_delta,
            DriftSeverity.NONE,
        )
        # MODERATE: ≥ 2.0 mm
        hist = self._hist(
            WatchdogSample(predicted_observed_calibration_delta_mm=3.0)
        )
        self.assertIs(
            evaluate_drift(hist, policy).calibration_delta,
            DriftSeverity.MODERATE,
        )
        # HIGH: ≥ 5.0 mm
        hist = self._hist(
            WatchdogSample(predicted_observed_calibration_delta_mm=7.0)
        )
        self.assertIs(
            evaluate_drift(hist, policy).calibration_delta,
            DriftSeverity.HIGH,
        )
        # SEVERE: ≥ 10.0 mm
        hist = self._hist(
            WatchdogSample(predicted_observed_calibration_delta_mm=15.0)
        )
        self.assertIs(
            evaluate_drift(hist, policy).calibration_delta,
            DriftSeverity.SEVERE,
        )

    def test_ood_severity_ladder_inverted(self) -> None:
        policy = WatchdogPolicy()
        # In-dist → NONE
        self.assertIs(
            evaluate_ood(
                self._hist(WatchdogSample(ood_score=0.9)), policy
            ),
            DriftSeverity.NONE,
        )
        # Moderate (≤ 0.5)
        self.assertIs(
            evaluate_ood(
                self._hist(WatchdogSample(ood_score=0.4)), policy
            ),
            DriftSeverity.MODERATE,
        )
        # High (≤ 0.3)
        self.assertIs(
            evaluate_ood(
                self._hist(WatchdogSample(ood_score=0.2)), policy
            ),
            DriftSeverity.HIGH,
        )
        # Severe (≤ 0.1)
        self.assertIs(
            evaluate_ood(
                self._hist(WatchdogSample(ood_score=0.05)), policy
            ),
            DriftSeverity.SEVERE,
        )

    def test_empty_history_emits_none(self) -> None:
        policy = WatchdogPolicy()
        report = evaluate_watchdog(WatchdogHistory(()), policy)
        self.assertIs(report.drift_severity, DriftSeverity.NONE)
        self.assertIs(report.ood_severity, DriftSeverity.NONE)
        self.assertFalse(report.ood_flagged)
        self.assertFalse(report.degraded_mode_active)
        self.assertIs(report.recommended_action, WatchdogAction.NONE)


# ---------------------------------------------------------------------------
# Enforce logic — hardware + mode gating (Q4)
# ---------------------------------------------------------------------------


class WatchdogEnforcementTests(unittest.TestCase):
    @staticmethod
    def _severe_history() -> WatchdogHistory:
        # Both drift and OOD severe.
        return WatchdogHistory(
            (
                WatchdogSample(
                    predicted_observed_calibration_delta_mm=15.0,
                    ood_score=0.05,
                ),
            )
        )

    def test_active_real_hardware_auto_blocks(self) -> None:
        policy = WatchdogPolicy(mode=WatchdogMode.ACTIVE)
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="auto",
            is_simulated=False,
        )
        self.assertIs(report.recommended_action, WatchdogAction.BLOCK_AUTO)
        self.assertTrue(report.enforced)

    def test_canary_real_hardware_auto_blocks(self) -> None:
        policy = WatchdogPolicy(mode=WatchdogMode.CANARY)
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="auto",
            is_simulated=False,
        )
        self.assertIs(report.recommended_action, WatchdogAction.BLOCK_AUTO)
        self.assertTrue(report.enforced)

    def test_simulated_run_never_enforces_block(self) -> None:
        policy = WatchdogPolicy(mode=WatchdogMode.ACTIVE)
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="auto",
            is_simulated=True,
        )
        self.assertIs(report.recommended_action, WatchdogAction.BLOCK_AUTO)
        self.assertFalse(report.enforced)

    def test_shadow_never_enforces(self) -> None:
        policy = WatchdogPolicy(mode=WatchdogMode.SHADOW)
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="auto",
            is_simulated=False,
        )
        # Action still computed truthfully...
        self.assertIs(report.recommended_action, WatchdogAction.BLOCK_AUTO)
        # ...but never enforced in shadow mode.
        self.assertFalse(report.enforced)
        self.assertTrue(report.degraded_mode_active)

    def test_non_block_mode_not_enforced(self) -> None:
        policy = WatchdogPolicy(
            mode=WatchdogMode.ACTIVE, block_modes=("auto",)
        )
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="dense_clutter",
            is_simulated=False,
        )
        self.assertIs(report.recommended_action, WatchdogAction.BLOCK_AUTO)
        self.assertFalse(report.enforced)

    def test_disabled_mode_short_circuits(self) -> None:
        """Q8=A: disabled mode emits NONE, never degraded, never enforced."""

        policy = WatchdogPolicy(mode=WatchdogMode.DISABLED)
        report = evaluate_watchdog(
            self._severe_history(),
            policy,
            grasp_mode="auto",
            is_simulated=False,
        )
        self.assertIs(report.drift_severity, DriftSeverity.NONE)
        self.assertIs(report.ood_severity, DriftSeverity.NONE)
        self.assertIs(report.aggregate_severity, DriftSeverity.NONE)
        self.assertFalse(report.ood_flagged)
        self.assertFalse(report.degraded_mode_active)
        self.assertIs(report.recommended_action, WatchdogAction.NONE)
        self.assertFalse(report.enforced)
        self.assertEqual(
            report.monitors.to_dict(),
            DriftMonitorBreakdown().to_dict(),
        )


# ---------------------------------------------------------------------------
# U9-B — schema validators
# ---------------------------------------------------------------------------


class GraspingWatchdogConfigSchemaTests(unittest.TestCase):
    def test_easy_block_mode_rejected(self) -> None:
        from backend.config.schema.robot.robot_schema import (
            GraspingWatchdogConfig,
        )

        with self.assertRaises(Exception):
            GraspingWatchdogConfig(block_modes=("easy",))

    def test_disordered_ladder_rejected(self) -> None:
        from backend.config.schema.robot.robot_schema import (
            GraspingWatchdogConfig,
        )

        with self.assertRaises(Exception):
            # moderate > high violates the non-decreasing ladder.
            GraspingWatchdogConfig(
                calibration_delta_moderate_mm=10.0,
                calibration_delta_high_mm=5.0,
                calibration_delta_severe_mm=20.0,
            )


# ---------------------------------------------------------------------------
# U9-C — outcomes + reason codes + telemetry catalog
# ---------------------------------------------------------------------------


class WatchdogOutcomesAndCatalogTests(unittest.TestCase):
    def test_new_outcomes_exist(self) -> None:
        self.assertEqual(
            AutonomousGraspOutcome.DRIFT_BLOCKED_AUTO.value,
            "drift_blocked_auto",
        )
        self.assertEqual(
            AutonomousGraspOutcome.OOD_BLOCKED_AUTO.value,
            "ood_blocked_auto",
        )

    def test_new_reason_codes_exist(self) -> None:
        self.assertEqual(
            DecisionReasonCode.DRIFT_BLOCKED_AUTO.value,
            "drift_blocked_auto",
        )
        self.assertEqual(
            DecisionReasonCode.OOD_BLOCKED_AUTO.value,
            "ood_blocked_auto",
        )

    def test_telemetry_catalog_entries_present(self) -> None:
        drift_required = TELEMETRY_CATALOG["drift_blocked_auto"]
        ood_required = TELEMETRY_CATALOG["ood_blocked_auto"]
        for field in (
            "extra.decision_reason_code",
            "extra.drift_severity",
            "extra.degraded_mode_active",
        ):
            self.assertIn(field, drift_required)
        for field in (
            "extra.decision_reason_code",
            "extra.ood_flagged",
            "extra.degraded_mode_active",
        ):
            self.assertIn(field, ood_required)


# ---------------------------------------------------------------------------
# U9-E — events namespace
# ---------------------------------------------------------------------------


class RobotWatchdogEventsTests(unittest.TestCase):
    def test_event_topic_strings(self) -> None:
        self.assertEqual(
            RobotWatchdogEvent.DRIFT_DETECTED,
            "robot_watchdog_drift_detected",
        )
        self.assertEqual(
            RobotWatchdogEvent.OOD_DETECTED,
            "robot_watchdog_ood_detected",
        )
        self.assertEqual(
            RobotWatchdogEvent.DEGRADED_MODE_ENGAGED,
            "robot_watchdog_degraded_mode_engaged",
        )
        self.assertEqual(
            RobotWatchdogEvent.BLOCK_AUTO_TRIGGERED,
            "robot_watchdog_block_auto_triggered",
        )

    def test_all_tuple_is_stable(self) -> None:
        # Phase U10 added ``SLO_BREACH`` as the fifth canonical
        # watchdog event. The tuple order is part of the contract.
        self.assertEqual(
            RobotWatchdogEvent.ALL,
            (
                RobotWatchdogEvent.DRIFT_DETECTED,
                RobotWatchdogEvent.OOD_DETECTED,
                RobotWatchdogEvent.DEGRADED_MODE_ENGAGED,
                RobotWatchdogEvent.BLOCK_AUTO_TRIGGERED,
                RobotWatchdogEvent.SLO_BREACH,
            ),
        )

    def test_listener_protocol_runtime_checkable(self) -> None:
        class _Listener:
            def __call__(
                self, event_type: str, data: dict
            ) -> None:  # pragma: no cover
                pass

        self.assertIsInstance(_Listener(), RobotWatchdogEventListener)


# ---------------------------------------------------------------------------
# U9-F — canonical packs carry labels + signals + pass audit
# ---------------------------------------------------------------------------


class CanonicalWatchdogPackTests(unittest.TestCase):
    def test_drift_pack_layout(self) -> None:
        records = tuple(iter_jsonl(_DRIFT_PACK))
        self.assertEqual(len(records), 120)
        # First half: drift_label=False, second half: drift_label=True.
        first_half_labels = [
            bool(r.extra["drift_label"]) for r in records[:60]
        ]
        second_half_labels = [
            bool(r.extra["drift_label"]) for r in records[60:]
        ]
        self.assertEqual(first_half_labels, [False] * 60)
        self.assertEqual(second_half_labels, [True] * 60)
        # Stable half must keep residuals modest, degraded half must
        # ramp above the MODERATE threshold (2.0 mm).
        stable_max = max(
            float(r.extra["drift_predicted_observed_delta_mm"])
            for r in records[:60]
        )
        degraded_min = min(
            float(r.extra["drift_predicted_observed_delta_mm"])
            for r in records[60:]
        )
        self.assertLess(stable_max, 1.0)
        self.assertGreaterEqual(degraded_min, 2.0)

    def test_ood_pack_layout(self) -> None:
        records = tuple(iter_jsonl(_OOD_PACK))
        self.assertEqual(len(records), 120)
        first_half_labels = [
            bool(r.extra["ood_label"]) for r in records[:60]
        ]
        second_half_labels = [
            bool(r.extra["ood_label"]) for r in records[60:]
        ]
        self.assertEqual(first_half_labels, [False] * 60)
        self.assertEqual(second_half_labels, [True] * 60)
        stable_min = min(float(r.extra["ood_score"]) for r in records[:60])
        perturbed_max = max(
            float(r.extra["ood_score"]) for r in records[60:]
        )
        self.assertGreater(stable_min, 0.5)
        self.assertLess(perturbed_max, 0.3)

    def test_watchdog_packs_pass_catalog_audit(self) -> None:
        for pack_path in (_DRIFT_PACK, _OOD_PACK):
            records = tuple(iter_jsonl(pack_path))
            self.assertEqual(audit_records(records), [], msg=pack_path.name)
            self.assertEqual(
                audit_extra_records(records), [], msg=pack_path.name
            )

    def test_canonical_pack_specs_are_named(self) -> None:
        names = {pack.name for pack in CANONICAL_PACKS}
        self.assertIn("replay_drift_synthetic_v1", names)
        self.assertIn("replay_ood_synthetic_v1", names)


# ---------------------------------------------------------------------------
# U9-G — watchdog KPI gates
# ---------------------------------------------------------------------------


class WatchdogEvalKPIGateTests(unittest.TestCase):
    def test_drift_pack_meets_kpi_gate(self) -> None:
        report = evaluate_drift_pack_path(_DRIFT_PACK)
        self.assertEqual(report.detector, "drift")
        self.assertEqual(report.kpi.total, 120)
        self.assertGreaterEqual(report.kpi.precision, DRIFT_PRECISION_GATE)
        self.assertGreaterEqual(report.kpi.recall, DRIFT_RECALL_GATE)
        self.assertTrue(report.passes_gate)

    def test_ood_pack_meets_kpi_gate(self) -> None:
        report = evaluate_ood_pack_path(_OOD_PACK)
        self.assertEqual(report.detector, "ood")
        self.assertEqual(report.kpi.total, 120)
        self.assertGreaterEqual(report.kpi.precision, OOD_PRECISION_GATE)
        self.assertGreaterEqual(report.kpi.recall, OOD_RECALL_GATE)
        self.assertTrue(report.passes_gate)

    def test_binary_kpi_math(self) -> None:
        kpi = BinaryKPI(
            total=100,
            true_positives=45,
            false_positives=5,
            false_negatives=10,
            true_negatives=40,
        )
        self.assertEqual(kpi.positives, 55)
        self.assertEqual(kpi.negatives, 45)
        self.assertAlmostEqual(kpi.precision, 45 / 50)
        self.assertAlmostEqual(kpi.recall, 45 / 55)

    def test_build_sample_from_record_maps_fields(self) -> None:
        records = tuple(iter_jsonl(_DRIFT_PACK))
        # Pick a degraded record so every signal is populated.
        rec = records[100]
        sample = build_sample_from_record(rec)
        self.assertEqual(
            sample.calibration_residual_mm,
            float(rec.extra["drift_hand_eye_residual_mm"]),
        )
        self.assertEqual(
            sample.verification_residual_mm,
            float(rec.extra["drift_verification_residual_mm"]),
        )
        self.assertEqual(
            sample.predicted_observed_calibration_delta_mm,
            float(rec.extra["drift_predicted_observed_delta_mm"]),
        )
        self.assertEqual(
            sample.depth_confidence_mean,
            float(rec.extra["drift_depth_confidence_mean"]),
        )
        self.assertEqual(
            sample.fail_closed, bool(rec.extra["drift_fail_closed"])
        )

    def test_kpi_report_dict_is_json_safe(self) -> None:
        report = evaluate_drift_pack(tuple(iter_jsonl(_DRIFT_PACK)))
        payload = report.to_dict()
        json.dumps(payload)  # must not raise
        self.assertIn("kpi", payload)
        self.assertEqual(payload["detector"], "drift")
        self.assertTrue(payload["passes_gate"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
