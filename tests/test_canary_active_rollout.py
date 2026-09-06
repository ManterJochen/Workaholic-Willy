"""Bounded ``rl_active`` canary router tests.

Verifies the canary contract:

* config validation,
* deterministic SHA-1 sampling distribution and replay-stability,
* mode-scope guard (EASY/AUTO get baseline),
* override accounting + regret upper-bound proxy,
* auto-fallback on regret-rate and override-rate spikes (after warmup),
* operator-engaged fallback + reset drill,
* hard rejection of non-``pass`` promotion reports,
* end-to-end with the committed recovery baseline + promotion report,
* explainability telemetry (every telemetry field present).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.canary_router import (
    DEFAULT_ELIGIBLE_MODES,
    FALLBACK_REASON_OPERATOR,
    FALLBACK_REASON_OVERRIDE_RATE,
    FALLBACK_REASON_POLICY_ERROR,
    FALLBACK_REASON_REGRET_RATE,
    FALLBACK_REASONS,
    RL_MODE_RL_ACTIVE,
    ActiveCanaryRouter,
    CanaryConfig,
    CanaryDecision,
    CanaryInputError,
    _attempt_in_canary,
    load_active_canary_router,
)
from src.robot.grasping.rl.promotion import (
    POLICY_FAMILY_RECOVERY,
    PROMOTION_VERDICT_FAIL,
)
from src.robot.grasping.rl.action_mask import ActionMaskContext
from src.robot.grasping.rl.recovery_policy import (
    RECOVERY_ACTION_ABORT_RECOVERY,
    RECOVERY_ACTIONS,
    RecoveryRequest,
    RecoveryStateKey,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_POLICY_PATH = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies" / "v6_recovery_baseline_v1.json"
)
RECOVERY_PROMOTION_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v6_recovery_baseline_v1_promotion_v1.json"
)


def _synthetic_pass_promotion_path() -> Path:
    """The committed recovery promotion report now honestly ABSTAINS (the policy is degenerate), and the
    active-canary router correctly refuses any non-``pass`` policy. To exercise the canary MECHANICS we
    need a *promotable* policy, so this writes a synthetic ``verdict=pass`` copy of the committed report
    once. The refusal-on-abstain path is tested directly against the real committed ``RECOVERY_PROMOTION_PATH``."""

    import json
    import tempfile

    data = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
    data["verdict"] = "pass"
    data["reasons"] = []
    out = Path(tempfile.mkdtemp(prefix="canary_pass_")) / "recovery_pass_promotion_v1.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


PASS_PROMOTION_PATH = _synthetic_pass_promotion_path()


def _state() -> RecoveryStateKey:
    return RecoveryStateKey(
        failure_class_bucket="empty_air_grasp",
        attempt_index_bucket="0",
        reobserve_count_bucket="0",
        dense_bucket="dense",
        last_outcome_bucket="failed",
    )


def _req(attempt_id: str) -> RecoveryRequest:
    return RecoveryRequest(attempt_id=attempt_id, state_key=_state())


def _make_router(
    *,
    config: CanaryConfig | None = None,
) -> ActiveCanaryRouter:
    return load_active_canary_router(
        policy_artifact_path=RECOVERY_POLICY_PATH,
        promotion_report_path=PASS_PROMOTION_PATH,
        config=config,
    )


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------


class CanaryConfigTests(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        c = CanaryConfig()
        self.assertEqual(c.canary_pct, 10.0)
        self.assertEqual(c.eligible_modes, DEFAULT_ELIGIBLE_MODES)
        self.assertEqual(c.window_size, 50)
        self.assertEqual(c.warmup_n, 20)
        self.assertEqual(c.max_regret_rate, 0.30)
        self.assertEqual(c.max_override_rate, 0.95)

    def test_rejects_negative_pct(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(canary_pct=-1.0)

    def test_rejects_overlarge_pct(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(canary_pct=100.1)

    def test_rejects_nonpositive_window(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(window_size=0)

    def test_rejects_empty_eligible_modes(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(eligible_modes=())

    def test_rejects_out_of_range_regret_rate(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(max_regret_rate=1.5)

    def test_rejects_out_of_range_override_rate(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(max_override_rate=-0.1)

    def test_rejects_nonpositive_max_recovery_attempts(self) -> None:
        with self.assertRaises(CanaryInputError):
            CanaryConfig(max_recovery_attempts=0)


# ---------------------------------------------------------------------------
# Sampling.
# ---------------------------------------------------------------------------


class CanarySamplingTests(unittest.TestCase):
    def test_zero_pct_admits_nothing(self) -> None:
        for i in range(100):
            self.assertFalse(_attempt_in_canary(f"a{i}", 0.0))

    def test_full_pct_admits_everything(self) -> None:
        for i in range(100):
            self.assertTrue(_attempt_in_canary(f"a{i}", 100.0))

    def test_deterministic(self) -> None:
        for i in range(50):
            self.assertEqual(
                _attempt_in_canary(f"a{i}", 10.0),
                _attempt_in_canary(f"a{i}", 10.0),
            )

    def test_distribution_is_close_to_target(self) -> None:
        # 10% over 10000 ids should land in [800, 1200] (allow ±2%)
        n = 10000
        admitted = sum(
            1 for i in range(n) if _attempt_in_canary(f"attempt_{i}", 10.0)
        )
        self.assertGreater(admitted, 800)
        self.assertLess(admitted, 1200)

    def test_higher_pct_admits_superset(self) -> None:
        # Bucket-based admission must be monotonic in pct: any id
        # admitted at 10% must also be admitted at 50%.
        for i in range(500):
            if _attempt_in_canary(f"a{i}", 10.0):
                self.assertTrue(_attempt_in_canary(f"a{i}", 50.0))


# ---------------------------------------------------------------------------
# Promotion-report contract.
# ---------------------------------------------------------------------------


class PromotionContractTests(unittest.TestCase):
    def test_rejects_non_pass_verdict(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            doctored = Path(td) / "report.json"
            report = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
            report["verdict"] = PROMOTION_VERDICT_FAIL
            doctored.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(CanaryInputError):
                load_active_canary_router(
                    policy_artifact_path=RECOVERY_POLICY_PATH,
                    promotion_report_path=doctored,
                )

    def test_rejects_family_mismatch(self) -> None:
        # Edit the committed report's family field to provoke rejection.
        with tempfile.TemporaryDirectory() as td:
            doctored = Path(td) / "report.json"
            report = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
            report["policy_family"] = "v5_perception_budget"
            doctored.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(CanaryInputError):
                load_active_canary_router(
                    policy_artifact_path=RECOVERY_POLICY_PATH,
                    promotion_report_path=doctored,
                )

    def test_rejects_policy_id_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            doctored = Path(td) / "report.json"
            report = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
            report["policy_id"] = "some_other_policy@1"
            doctored.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(CanaryInputError):
                load_active_canary_router(
                    policy_artifact_path=RECOVERY_POLICY_PATH,
                    promotion_report_path=doctored,
                )

    def test_accepts_a_passing_policy(self) -> None:
        router = _make_router()
        self.assertEqual(
            router.promotion_report["policy_family"],
            POLICY_FAMILY_RECOVERY,
        )

    def test_committed_recovery_abstains_so_router_refuses(self) -> None:
        # The committed recovery policy is degenerate and honestly ABSTAINS, so the active-canary router
        # (which only runs a PROMOTED policy) refuses to build on it. The mechanics tests use a synthetic
        # 'pass' fixture (PASS_PROMOTION_PATH); this pins the honest refusal on the real committed report.
        with self.assertRaises(CanaryInputError):
            load_active_canary_router(
                policy_artifact_path=RECOVERY_POLICY_PATH,
                promotion_report_path=RECOVERY_PROMOTION_PATH,
            )


# ---------------------------------------------------------------------------
# Decision wiring.
# ---------------------------------------------------------------------------


class ChooseRecoveryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = _make_router()

    def test_out_of_scope_mode_returns_baseline(self) -> None:
        # EASY is not in DEFAULT_ELIGIBLE_MODES → applied == baseline,
        # in_canary False, override False.
        d = self.router.choose_recovery(
            request=_req("attempt-easy-1"),
            baseline_action="reobserve",
            mode="easy",
        )
        self.assertFalse(d.scope_eligible)
        self.assertFalse(d.in_canary)
        self.assertFalse(d.override)
        self.assertEqual(d.applied_action, "reobserve")
        self.assertIsNone(d.rl_action_proposed)

    def test_in_scope_but_not_in_canary_returns_baseline(self) -> None:
        # 0% canary => no attempt is admitted but scope is eligible.
        router = _make_router(config=CanaryConfig(canary_pct=0.0))
        d = router.choose_recovery(
            request=_req("attempt-dense-1"),
            baseline_action="re_segment",
            mode="dense",
        )
        self.assertTrue(d.scope_eligible)
        self.assertFalse(d.in_canary)
        self.assertEqual(d.applied_action, "re_segment")
        self.assertFalse(d.override)
        self.assertIsNone(d.rl_action_proposed)

    def test_in_canary_proposes_rl_action(self) -> None:
        router = _make_router(config=CanaryConfig(canary_pct=100.0))
        d = router.choose_recovery(
            request=_req("attempt-x"),
            baseline_action="abort_recovery",
            mode="dense",
        )
        self.assertTrue(d.in_canary)
        self.assertIsNotNone(d.rl_action_proposed)
        self.assertIn(d.rl_action_proposed, RECOVERY_ACTIONS)
        self.assertEqual(d.applied_action, d.rl_action_proposed)
        # Baseline was "abort_recovery" and the recovery baseline always
        # proposes "reobserve" → override is True.
        self.assertTrue(d.override)

    def test_no_override_when_rl_agrees_with_baseline(self) -> None:
        router = _make_router(config=CanaryConfig(canary_pct=100.0))
        # the recovery policy proposes "re_segment" for an empty-air dense state;
        # match the baseline to that so override should be False.
        d = router.choose_recovery(
            request=_req("attempt-y"),
            baseline_action="re_segment",
            mode="dense",
        )
        self.assertTrue(d.in_canary)
        self.assertEqual(d.applied_action, "re_segment")
        self.assertFalse(d.override)

    def test_invalid_baseline_action_rejected(self) -> None:
        with self.assertRaises(CanaryInputError):
            self.router.choose_recovery(
                request=_req("z"),
                baseline_action="not_a_real_action",
                mode="dense",
            )

    def test_decision_carries_all_telemetry_fields(self) -> None:
        router = _make_router(config=CanaryConfig(canary_pct=100.0))
        d = router.choose_recovery(
            request=_req("audit-y"),
            baseline_action="abort_recovery",
            mode="dense",
        )
        self.assertIsInstance(d, CanaryDecision)
        # Every telemetry field must be populated.
        self.assertEqual(d.rl_mode, RL_MODE_RL_ACTIVE)
        self.assertTrue(d.rl_policy_id)
        self.assertEqual(d.rl_artifact_version, 1)
        self.assertEqual(d.rl_router_path, "base")
        self.assertEqual(len(d.audit_id), 32)
        self.assertEqual(len(d.rl_reason_features), len(RECOVERY_ACTIONS))
        for action, expected in d.rl_reason_features:
            self.assertIn(action, RECOVERY_ACTIONS)
            self.assertIsInstance(expected, float)


# ---------------------------------------------------------------------------
# Fail-closed gates in front of the applied action (six-channel scene mask + anti-loop).
# ---------------------------------------------------------------------------


class MaskAndAntiLoopGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.router = _make_router(config=CanaryConfig(canary_pct=100.0))

    def test_degraded_scene_mask_blocks_rl_override(self) -> None:
        # A degraded scene (the scene-level six-channel ``degraded_mode`` channel) forbids any RL
        # override even inside the canary: the RL still PROPOSES, but the applied action stays baseline.
        d = self.router.choose_recovery(
            request=_req("masked-1"),
            baseline_action="abort_recovery",
            mode="dense",
            mask_context=ActionMaskContext(degraded_mode_active=True),
        )
        self.assertTrue(d.in_canary)
        self.assertIsNotNone(d.rl_action_proposed)  # policy still ran
        self.assertTrue(d.rl_action_blocked_by_mask)
        self.assertEqual(d.applied_action, "abort_recovery")  # baseline stands
        self.assertFalse(d.override)

    def test_non_degraded_scene_allows_override(self) -> None:
        # A non-degraded mask context is a no-op: the RL override applies as usual.
        d = self.router.choose_recovery(
            request=_req("masked-2"),
            baseline_action="abort_recovery",
            mode="dense",
            mask_context=ActionMaskContext(degraded_mode_active=False),
        )
        self.assertTrue(d.in_canary)
        self.assertFalse(d.rl_action_blocked_by_mask)
        self.assertEqual(d.applied_action, d.rl_action_proposed)
        self.assertTrue(d.override)

    def test_anti_loop_gate_clips_at_budget(self) -> None:
        # attempt_index >= max_recovery_attempts (default 3) clips any non-abort proposal to
        # ``abort_recovery`` with the budget-reached reason.
        d = self.router.choose_recovery(
            request=_req("loop-1"),
            baseline_action="reobserve",
            mode="dense",
            attempt_index=3,
        )
        self.assertTrue(d.in_canary)
        self.assertEqual(d.applied_action, RECOVERY_ACTION_ABORT_RECOVERY)
        self.assertTrue(d.anti_loop_clipped)
        self.assertEqual(d.anti_loop_clip_reason, "max_recovery_attempts_reached")

    def test_anti_loop_gate_noop_within_budget(self) -> None:
        # Default attempt_index=0 is within budget: no clip, RL override applies.
        d = self.router.choose_recovery(
            request=_req("loop-2"),
            baseline_action="abort_recovery",
            mode="dense",
        )
        self.assertFalse(d.anti_loop_clipped)
        self.assertEqual(d.anti_loop_clip_reason, "none")
        self.assertEqual(d.applied_action, d.rl_action_proposed)

    def test_mask_outranks_gate(self) -> None:
        # Safety precedence: a degraded scene blocks the override regardless of the anti-loop gate.
        d = self.router.choose_recovery(
            request=_req("prec-1"),
            baseline_action="reobserve",
            mode="dense",
            attempt_index=3,
            mask_context=ActionMaskContext(degraded_mode_active=True),
        )
        self.assertTrue(d.rl_action_blocked_by_mask)
        self.assertEqual(d.applied_action, "reobserve")  # baseline, not the gate's abort
        self.assertFalse(d.anti_loop_clipped)


# ---------------------------------------------------------------------------
# Outcome reporting + stats.
# ---------------------------------------------------------------------------


class OutcomeAndStatsTests(unittest.TestCase):
    def test_unknown_audit_id_is_silent_noop(self) -> None:
        router = _make_router()
        router.record_outcome("does-not-exist", "succeeded")  # must not raise
        self.assertFalse(router.is_fallback_active)

    def test_bad_outcome_rejected(self) -> None:
        router = _make_router()
        with self.assertRaises(CanaryInputError):
            router.record_outcome("x", "maybe")

    def test_stats_track_override_and_regret(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=10,
                max_regret_rate=0.99, max_override_rate=1.0,
            )
        )
        ids = []
        for i in range(5):
            d = router.choose_recovery(
                request=_req(f"a{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
            self.assertTrue(d.override)
            ids.append(d.audit_id)
        # Mark 3/5 overrides as failed → regret rate = 3/5 over the
        # overrides observed (note: regret_rate is over override_count
        # in this implementation).
        for i in range(3):
            router.record_outcome(ids[i], "failed")
        for i in range(3, 5):
            router.record_outcome(ids[i], "succeeded")
        s = router.stats()
        self.assertEqual(s.window_attempts, 5)
        self.assertEqual(s.override_count, 5)
        self.assertEqual(s.regret_count, 3)
        self.assertAlmostEqual(s.override_rate, 1.0)
        self.assertAlmostEqual(s.regret_rate, 3.0 / 5.0)

    def test_window_evicts_oldest(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=3,
                max_regret_rate=0.99, max_override_rate=1.0,
            )
        )
        for i in range(10):
            router.choose_recovery(
                request=_req(f"e{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
        s = router.stats()
        self.assertEqual(s.window_attempts, 3)


# ---------------------------------------------------------------------------
# Auto-fallback.
# ---------------------------------------------------------------------------


class AutoFallbackTests(unittest.TestCase):
    def test_no_fallback_before_warmup(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=10, window_size=50,
                max_regret_rate=0.01, max_override_rate=0.01,
            )
        )
        ids = []
        # 5 overrides, all failed → would trip thresholds, but warmup=10.
        for i in range(5):
            d = router.choose_recovery(
                request=_req(f"w{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
            ids.append(d.audit_id)
        for aid in ids:
            router.record_outcome(aid, "failed")
        self.assertFalse(router.is_fallback_active)

    def test_engages_on_regret_rate_after_warmup(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=50,
                max_regret_rate=0.20, max_override_rate=1.0,
            )
        )
        ids = []
        for i in range(10):
            d = router.choose_recovery(
                request=_req(f"r{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
            ids.append(d.audit_id)
        # All 10 overrides → all failed → regret rate 1.0 >> 0.20
        for aid in ids:
            router.record_outcome(aid, "failed")
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, FALLBACK_REASON_REGRET_RATE,
        )

    def test_engages_on_override_rate_after_warmup(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=50,
                max_regret_rate=0.99, max_override_rate=0.50,
            )
        )
        # Each canary attempt overrides (the recovery policy always proposes
        # reobserve; baseline is abort_recovery). Override rate = 1.0 > 0.50.
        for i in range(5):
            router.choose_recovery(
                request=_req(f"o{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, FALLBACK_REASON_OVERRIDE_RATE,
        )

    def test_post_fallback_returns_baseline(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=10,
                max_regret_rate=0.01, max_override_rate=0.99,
            )
        )
        ids = []
        for i in range(3):
            d = router.choose_recovery(
                request=_req(f"f{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
            ids.append(d.audit_id)
        for aid in ids:
            router.record_outcome(aid, "failed")
        self.assertTrue(router.is_fallback_active)
        # Subsequent calls return baseline regardless of canary sampling.
        d = router.choose_recovery(
            request=_req("after-fallback"),
            baseline_action="reobserve",
            mode="dense",
        )
        self.assertTrue(d.fallback_triggered)
        self.assertEqual(d.applied_action, "reobserve")
        self.assertFalse(d.override)
        self.assertIsNone(d.rl_action_proposed)


# ---------------------------------------------------------------------------
# Operator engage / reset.
# ---------------------------------------------------------------------------


class OperatorFallbackAndResetTests(unittest.TestCase):
    def test_operator_engage(self) -> None:
        router = _make_router()
        router.engage_fallback(FALLBACK_REASON_OPERATOR)
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(router.fallback_reason, FALLBACK_REASON_OPERATOR)

    def test_engage_rejects_unknown_reason(self) -> None:
        router = _make_router()
        with self.assertRaises(CanaryInputError):
            router.engage_fallback("bogus_reason")

    def test_reset_clears_fallback_and_window(self) -> None:
        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=10,
                max_regret_rate=0.99, max_override_rate=0.99,
            )
        )
        for i in range(3):
            router.choose_recovery(
                request=_req(f"a{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
        router.engage_fallback()
        self.assertTrue(router.is_fallback_active)
        router.reset_canary()
        self.assertFalse(router.is_fallback_active)
        self.assertIsNone(router.fallback_reason)
        s = router.stats()
        self.assertEqual(s.window_attempts, 0)
        self.assertEqual(s.override_count, 0)
        self.assertEqual(router.canary_attempts_seen, 0)

    def test_full_rollback_drill(self) -> None:
        """End-to-end rollback drill required by the active-canary contract.

        Sequence: run canary → spike regret → confirm fallback engaged
        → confirm subsequent decisions are baseline → operator reset →
        confirm canary back online (new attempts get RL action again).
        """

        router = _make_router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=20,
                max_regret_rate=0.10, max_override_rate=1.0,
            )
        )
        # 1) Drive enough overrides + failures to trip regret-rate gate.
        bad_ids = []
        for i in range(5):
            d = router.choose_recovery(
                request=_req(f"bad{i}"),
                baseline_action="abort_recovery",
                mode="dense",
            )
            bad_ids.append(d.audit_id)
        for aid in bad_ids:
            router.record_outcome(aid, "failed")
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, FALLBACK_REASON_REGRET_RATE,
        )
        # 2) During fallback, decisions stay baseline.
        d_during = router.choose_recovery(
            request=_req("during-fallback"),
            baseline_action="abort_recovery",
            mode="dense",
        )
        self.assertTrue(d_during.fallback_triggered)
        self.assertEqual(d_during.applied_action, "abort_recovery")
        # 3) Operator resets and canary comes back online.
        router.reset_canary()
        d_after = router.choose_recovery(
            request=_req("after-reset"),
            baseline_action="abort_recovery",
            mode="dense",
        )
        self.assertFalse(d_after.fallback_triggered)
        self.assertTrue(d_after.in_canary)
        # the recovery policy proposes "re_segment" for this state — override flag
        # is True because baseline was "abort_recovery".
        self.assertEqual(d_after.applied_action, "re_segment")
        self.assertTrue(d_after.override)


# ---------------------------------------------------------------------------
# Loader.
# ---------------------------------------------------------------------------


class LoaderTests(unittest.TestCase):
    def test_loader_returns_active_router(self) -> None:
        router = _make_router()
        self.assertIsInstance(router, ActiveCanaryRouter)

    def test_loader_rejects_missing_policy(self) -> None:
        with self.assertRaises(CanaryInputError):
            load_active_canary_router(
                policy_artifact_path=Path("/does/not/exist.json"),
                promotion_report_path=RECOVERY_PROMOTION_PATH,
            )

    def test_loader_rejects_missing_promotion(self) -> None:
        with self.assertRaises(CanaryInputError):
            load_active_canary_router(
                policy_artifact_path=RECOVERY_POLICY_PATH,
                promotion_report_path=Path("/does/not/exist.json"),
            )


# ---------------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------------


class ModuleConstantsTests(unittest.TestCase):
    def test_fallback_reason_set(self) -> None:
        self.assertEqual(
            set(FALLBACK_REASONS),
            {
                FALLBACK_REASON_REGRET_RATE,
                FALLBACK_REASON_OVERRIDE_RATE,
                FALLBACK_REASON_OPERATOR,
                FALLBACK_REASON_POLICY_ERROR,
            },
        )

    def test_default_eligible_modes_are_dense(self) -> None:
        for m in DEFAULT_ELIGIBLE_MODES:
            self.assertIn("dense", m)


if __name__ == "__main__":
    unittest.main()
