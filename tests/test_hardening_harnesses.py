"""Hardening-harness tests (paired soak + rollback drills).

Covers:
* :mod:`backend.src.robot.grasping.rl.paired_soak` — paired soak,
  parity verifier, determinism, JSON artefacts.
* :mod:`backend.src.robot.grasping.rl.rollback_drill` — 9
  rollback drills + report shape.
* Presence + headline keywords of the 4 hardening runbook docs.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from src.robot.grasping.rl.canary_router import RL_MODE_GEOMETRY_ONLY
from src.robot.grasping.rl.rollback_drill import (
    DRILL_KINDS,
    DRILL_PASS,
    ROUTER_BASE,
    ROUTER_SPECIALIST,
    ROLLBACK_DRILL_SCHEMA_VERSION,
    run_rollback_drills,
)
from src.robot.grasping.rl.paired_soak import (
    PARITY_FAIL,
    PARITY_PASS,
    PARITY_VERDICTS,
    SOAK_REWARD_MODEL,
    PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM,
    PAIRED_SOAK_SCHEMA_VERSION,
    SoakScenarioSpec,
    generate_rl_off_arm,
    run_paired_soak,
    verify_baseline_parity,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v6_recovery_baseline_v1.json"
)
PROMOTION_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v6_recovery_baseline_v1_promotion_v1.json"
)


def _synthetic_pass_promotion_path() -> Path:
    """R4: the committed V6 promotion report now honestly ABSTAINS (degenerate policy), and the V8/V9
    routers the paired-soak + rollback drills build internally require a ``pass`` policy. To exercise the
    harness MECHANICS we need a *promotable* policy, so this writes a synthetic ``verdict=pass`` copy once."""

    import tempfile

    data = json.loads(PROMOTION_PATH.read_text(encoding="utf-8"))
    data["verdict"] = "pass"
    data["reasons"] = []
    out = Path(tempfile.mkdtemp(prefix="hardening_pass_")) / "v6_pass_promotion_v1.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


PASS_PROMOTION_PATH = _synthetic_pass_promotion_path()
RUNBOOKS_DIR = REPO_ROOT / "docs" / "runbooks"


class SoakSpecTests(unittest.TestCase):
    def test_default_spec(self) -> None:
        spec = SoakScenarioSpec()
        self.assertGreaterEqual(spec.attempts_per_arm, PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM)
        self.assertEqual(spec.seed, 20251015)

    def test_attempts_floor_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SoakScenarioSpec(attempts_per_arm=PAIRED_SOAK_MIN_ATTEMPTS_PER_ARM - 1)

    def test_attempts_ceiling_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SoakScenarioSpec(attempts_per_arm=200_000)


class SoakArmGenerationTests(unittest.TestCase):
    def test_rl_off_arm_is_deterministic(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        a = generate_rl_off_arm(spec)
        b = generate_rl_off_arm(spec)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 200)

    def test_rl_off_arm_uses_baseline_action(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        records = generate_rl_off_arm(spec)
        for r in records:
            self.assertEqual(r.applied_action, r.baseline_action)
            self.assertFalse(r.override)
            self.assertIsNone(r.rl_action_proposed)
            self.assertEqual(r.rl_mode, RL_MODE_GEOMETRY_ONLY)

    def test_baseline_parity_pass_on_rl_off(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        records = generate_rl_off_arm(spec)
        result = verify_baseline_parity(records)
        self.assertEqual(result.verdict, PARITY_PASS)
        self.assertEqual(result.violations, ())

    def test_baseline_parity_fail_on_doctored_record(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        records = list(generate_rl_off_arm(spec))
        # Force a violation: baseline-bound row but applied != baseline.
        bad = replace(records[0], applied_action="abort_recovery")
        records[0] = bad
        result = verify_baseline_parity(records)
        self.assertEqual(result.verdict, PARITY_FAIL)
        self.assertTrue(any(records[0].attempt_id in v for v in result.violations))


class PairedSoakTests(unittest.TestCase):
    def test_run_paired_soak_writes_three_jsons(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            report = run_paired_soak(
                policy_artifact_path=POLICY_PATH,
                promotion_report_path=PASS_PROMOTION_PATH,
                spec=spec,
                output_dir=out,
            )
            self.assertEqual(report.schema_version, PAIRED_SOAK_SCHEMA_VERSION)
            self.assertIn(report.verdict, PARITY_VERDICTS)
            self.assertEqual(report.baseline_parity.verdict, PARITY_PASS)
            for name in (
                "paired_soak_rl_off_v1.json",
                "paired_soak_rl_on_v1.json",
                "paired_soak_comparison_v1.json",
            ):
                path = out / name
                self.assertTrue(path.exists(), f"missing {name}")
                payload = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(payload["schema_version"], PAIRED_SOAK_SCHEMA_VERSION)
                # R11: every harness artifact must carry the synthetic-reward honesty stamp so the deltas
                # are never mistaken for real-hardware lift.
                self.assertEqual(payload["reward_model"], SOAK_REWARD_MODEL)
            comparison = json.loads(
                (out / "paired_soak_comparison_v1.json").read_text(encoding="utf-8")
            )
            self.assertIn("interpretation", comparison)
            self.assertIn("synthetic", comparison["reward_model"])

    def test_run_paired_soak_is_deterministic(self) -> None:
        spec = SoakScenarioSpec(attempts_per_arm=200)
        with tempfile.TemporaryDirectory() as tmp_a, tempfile.TemporaryDirectory() as tmp_b:
            run_paired_soak(
                policy_artifact_path=POLICY_PATH,
                promotion_report_path=PASS_PROMOTION_PATH,
                spec=spec,
                output_dir=Path(tmp_a),
            )
            run_paired_soak(
                policy_artifact_path=POLICY_PATH,
                promotion_report_path=PASS_PROMOTION_PATH,
                spec=spec,
                output_dir=Path(tmp_b),
            )
            for name in (
                "paired_soak_rl_off_v1.json",
                "paired_soak_rl_on_v1.json",
                "paired_soak_comparison_v1.json",
            ):
                a = (Path(tmp_a) / name).read_bytes()
                b = (Path(tmp_b) / name).read_bytes()
                self.assertEqual(a, b, f"{name} not deterministic")


class RollbackDrillTests(unittest.TestCase):
    def test_run_rollback_drills_all_pass(self) -> None:
        report = run_rollback_drills(
            policy_artifact_path=POLICY_PATH,
            promotion_report_path=PASS_PROMOTION_PATH,
        )
        self.assertEqual(report.schema_version, ROLLBACK_DRILL_SCHEMA_VERSION)
        self.assertEqual(len(report.outcomes), 9)
        for outcome in report.outcomes:
            self.assertEqual(
                outcome.verdict,
                DRILL_PASS,
                msg=f"{outcome.router}/{outcome.drill} failed: {outcome.detail}",
            )
        self.assertEqual(report.overall_verdict, DRILL_PASS)

    def test_run_rollback_drills_covers_all_kinds(self) -> None:
        report = run_rollback_drills(
            policy_artifact_path=POLICY_PATH,
            promotion_report_path=PASS_PROMOTION_PATH,
        )
        seen_kinds = {o.drill for o in report.outcomes}
        # All five drill kinds appear; first four appear twice (V8+V9).
        self.assertEqual(seen_kinds, set(DRILL_KINDS))
        v8 = [o for o in report.outcomes if o.router == ROUTER_BASE]
        v9 = [o for o in report.outcomes if o.router == ROUTER_SPECIALIST]
        self.assertEqual(len(v8), 4)
        self.assertEqual(len(v9), 4)

    def test_rollback_drill_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "drill_report.json"
            report = run_rollback_drills(
                policy_artifact_path=POLICY_PATH,
                promotion_report_path=PASS_PROMOTION_PATH,
                output_path=out,
            )
            self.assertTrue(out.exists())
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                payload["schema_version"], ROLLBACK_DRILL_SCHEMA_VERSION
            )
            self.assertEqual(payload["overall_verdict"], DRILL_PASS)
            self.assertEqual(len(payload["outcomes"]), 9)
            self.assertEqual(report.overall_verdict, DRILL_PASS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
