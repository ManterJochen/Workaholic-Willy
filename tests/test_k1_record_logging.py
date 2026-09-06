"""L6 K1 — the production AutonomousGraspReport -> GraspAttemptRecord serializer + JSONL logger.

The live pick path persisted no records, so safety_rejection_rate had no source (a misleading 0.0). These
tests pin the bridge: a fail-closed report -> a record with extra["safety_rejected"]=True -> compute_kpis
finally counts it; and the opt-in log_record round-trips through iter_jsonl.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.record_logging import (
    log_record,
    to_attempt_record,
)
from src.robot.execution.autonomous_grasp.report import (
    AutonomousGraspOutcome,
)
from src.robot.grasping.telemetry.outcome_logging import iter_jsonl
from src.robot.grasping.replay.kpi import compute_kpis


def _report(outcome: AutonomousGraspOutcome, *, telemetry: dict | None = None):
    # Duck-typed stand-in for an AutonomousGraspReport (the serializer reads outcome/mode/profile/telemetry).
    return SimpleNamespace(
        outcome=outcome,
        mode=SimpleNamespace(value="auto"),
        profile=SimpleNamespace(),
        telemetry=telemetry or {},
        pick_report=None,
    )


class SerializerTests(unittest.TestCase):
    def test_safety_rejected_set_on_fail_closed(self) -> None:
        rec = to_attempt_record(
            _report(AutonomousGraspOutcome.DECISION_FAIL_CLOSED), attempt_id="a1"
        )
        self.assertTrue(rec.extra["safety_rejected"])
        self.assertEqual(rec.final_outcome, "decision_fail_closed")
        self.assertEqual(rec.mode, "auto")

    def test_safety_rejected_false_on_success(self) -> None:
        rec = to_attempt_record(
            _report(AutonomousGraspOutcome.SUCCEEDED), attempt_id="a2"
        )
        self.assertFalse(rec.extra["safety_rejected"])
        # The false-positive key is intentionally unset (no secondary verification detector exists yet).
        self.assertNotIn("verification_failed_after_success", rec.extra)

    def test_every_fail_closed_outcome_flags_safety(self) -> None:
        for outcome in (
            AutonomousGraspOutcome.UNCERTAINTY_FAIL_CLOSED,
            AutonomousGraspOutcome.DRIFT_BLOCKED_AUTO,
            AutonomousGraspOutcome.OOD_BLOCKED_AUTO,
            AutonomousGraspOutcome.MISSING_CAMERA_FRAME,
            AutonomousGraspOutcome.UNSAFE_RECOVERY_REFUSED,
            AutonomousGraspOutcome.NO_COMMIT_INSUFFICIENT_FUSION,
        ):
            rec = to_attempt_record(_report(outcome), attempt_id="x")
            self.assertTrue(rec.extra["safety_rejected"], msg=str(outcome))

    def test_telemetry_carried_into_extra(self) -> None:
        rec = to_attempt_record(
            _report(AutonomousGraspOutcome.SUCCEEDED, telemetry={"foo": 1}),
            attempt_id="a3",
        )
        self.assertEqual(rec.extra["foo"], 1)

    def test_extra_merges_ground_truth_labels(self) -> None:
        # P0: a sim runner stamps physically-measured ground-truth labels the pipeline cannot self-report.
        rec = to_attempt_record(
            _report(AutonomousGraspOutcome.SUCCEEDED, telemetry={"foo": 1}),
            attempt_id="a4",
            extra={"sim_lift_mm": 99.6, "sim_lifted": True},
        )
        self.assertEqual(rec.extra["sim_lift_mm"], 99.6)
        self.assertIs(rec.extra["sim_lifted"], True)
        self.assertEqual(rec.extra["foo"], 1)  # telemetry preserved underneath the merge
        self.assertIn("safety_rejected", rec.extra)  # K1 flag still stamped

    def test_extra_none_adds_no_keys(self) -> None:
        # Default-off: no caller extra -> only the telemetry-derived fields + the safety flag.
        rec = to_attempt_record(_report(AutonomousGraspOutcome.SUCCEEDED), attempt_id="a5")
        self.assertEqual(set(rec.extra), {"safety_rejected"})

    def test_extra_overrides_telemetry_last_write_wins(self) -> None:
        rec = to_attempt_record(
            _report(AutonomousGraspOutcome.SUCCEEDED, telemetry={"sim_lift_mm": 1.0}),
            attempt_id="a6",
            extra={"sim_lift_mm": 2.0},
        )
        self.assertEqual(rec.extra["sim_lift_mm"], 2.0)

    def test_recovery_actions_carried_to_top_level_field(self) -> None:
        # P4.2: the per-step recovery trail on the report is written to the record's TOP-LEVEL
        # recovery_actions field (the V6 train_recovery source) -- not into extra.
        steps = (
            {"action": "next_viewpoint", "outcome": "completed", "executed": True},
            {"action": "rescan", "outcome": "recovered_success", "executed": True},
        )
        report = SimpleNamespace(
            outcome=AutonomousGraspOutcome.SUCCEEDED,
            mode=SimpleNamespace(value="auto"),
            profile=SimpleNamespace(),
            telemetry={},
            recovery_actions=steps,
        )
        rec = to_attempt_record(report, attempt_id="rec-cl")
        self.assertEqual(len(rec.recovery_actions), 2)
        self.assertEqual(rec.recovery_actions[1]["action"], "rescan")
        self.assertEqual(rec.recovery_actions[1]["outcome"], "recovered_success")
        self.assertNotIn("recovery_actions", rec.extra)  # top-level field, not extra

    def test_execution_and_sim_ground_truth_verification_populated(self) -> None:
        # S4: the serializer fills the execution + verification telemetry BLOCKS the soak audit requires --
        # execution from the report, verification from the SIM GROUND-TRUTH lift (explicitly labelled).
        from src.robot.grasping.replay.telemetry_catalog import audit_record

        rep = SimpleNamespace(
            outcome=AutonomousGraspOutcome.SUCCEEDED, mode=SimpleNamespace(value="dense_clutter"),
            profile=SimpleNamespace(), telemetry={}, recovery_actions=(), verification=None,
            pick_report=SimpleNamespace(outcome=SimpleNamespace(value="executed"), executed_grasp=None),
        )
        rec = to_attempt_record(
            rep, attempt_id="e1",
            extra={"sim_lifted": True, "sim_lift_mm": 81.6, "sim_max_distractor_lift_mm": 0.0},
        )
        self.assertEqual(rec.execution, {"outcome": "executed"})
        self.assertEqual(rec.verification["source"], "sim_ground_truth_lift")
        self.assertTrue(rec.verification["lifted"])
        self.assertEqual(rec.verification["lift_mm"], 81.6)
        self.assertEqual(audit_record(rec), ())  # succeeded record now satisfies the soak telemetry audit

    def test_verification_none_without_real_signal(self) -> None:
        # No sim ground-truth + no pipeline verification report -> verification stays None (NO fabrication;
        # the audit then honestly flags such a succeeded record as telemetry-incomplete).
        rep = SimpleNamespace(
            outcome=AutonomousGraspOutcome.SUCCEEDED, mode=SimpleNamespace(value="auto"),
            profile=SimpleNamespace(), telemetry={}, recovery_actions=(), verification=None,
            pick_report=SimpleNamespace(outcome=SimpleNamespace(value="executed"), executed_grasp=None),
        )
        rec = to_attempt_record(rep, attempt_id="e2")
        self.assertIsNone(rec.verification)

    def test_no_recovery_actions_defaults_empty(self) -> None:
        rec = to_attempt_record(_report(AutonomousGraspOutcome.SUCCEEDED), attempt_id="r0")
        self.assertEqual(tuple(rec.recovery_actions), ())


class KpiSourcingTests(unittest.TestCase):
    def test_safety_rejection_rate_now_sourced(self) -> None:
        # The whole point of K1: a real record built from a fail-closed report makes the previously-dead
        # KPI non-zero (1 of 2 attempts was a safety rejection).
        recs = (
            to_attempt_record(
                _report(AutonomousGraspOutcome.DECISION_FAIL_CLOSED), attempt_id="a1"
            ),
            to_attempt_record(
                _report(AutonomousGraspOutcome.SUCCEEDED), attempt_id="a2"
            ),
        )
        self.assertEqual(compute_kpis(recs).safety_rejection_rate, 0.5)


class LogRecordTests(unittest.TestCase):
    def test_log_record_round_trips(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "sub" / "pick_log.jsonl"  # parent dir is created
            log_record(
                _report(AutonomousGraspOutcome.DECISION_FAIL_CLOSED),
                attempt_id="a1",
                log_path=log,
            )
            log_record(
                _report(AutonomousGraspOutcome.SUCCEEDED),
                attempt_id="a2",
                log_path=log,
            )
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 2)
        self.assertTrue(recs[0].extra["safety_rejected"])
        self.assertEqual(compute_kpis(recs).safety_rejection_rate, 0.5)

    def test_log_record_forwards_extra(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "pick_log.jsonl"
            log_record(
                _report(AutonomousGraspOutcome.SUCCEEDED),
                attempt_id="a1",
                log_path=log,
                extra={"sim_lift_mm": 50.0, "sim_lifted": True},
            )
            recs = tuple(iter_jsonl(log))
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0].extra["sim_lift_mm"], 50.0)
        self.assertIs(recs[0].extra["sim_lifted"], True)


if __name__ == "__main__":
    unittest.main()
