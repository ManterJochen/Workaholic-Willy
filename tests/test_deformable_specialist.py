"""Deformable-specialist router + rl_experimental lane tests.

Hard-rules:

* Pure stdlib, deterministic, no torch / numpy.
* Reuses the committed recovery LinUCB baseline + promotion report so
  the tests double as a *contract* check on both artifacts.
* Independent kill switch — never touches the base canary ActiveCanaryRouter.
* ``rl_experimental`` lane: typed NotImplementedError only.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from typing import Any

from src.robot.grasping.rl.canary_router import (
    RL_MODES,
    RL_MODE_RL_ACTIVE,
    RL_MODE_RL_EXPERIMENTAL,
    CanaryConfig,
    load_active_canary_router,
)
from src.robot.grasping.rl.promotion import (
    POLICY_FAMILY_RECOVERY,
    PROMOTION_VERDICT_PASS,
)
from src.robot.grasping.rl.recovery_policy import (
    RECOVERY_ACTIONS,
    RECOVERY_ACTIONS_SET,
    RecoveryRequest,
    RecoveryStateKey,
    load_linucb_recovery_policy,
)
from src.robot.grasping.rl.specialist_router import (
    DEFAULT_SPECIALIST_BLOCKED_ACTIONS,
    EXPERIMENTAL_NOT_IMPLEMENTED_REASON,
    SPECIALIST_DENSE_BUCKET,
    SPECIALIST_FAILURE_CLASS_DEFORMABLE,
    SPECIALIST_FALLBACK_OPERATOR,
    SPECIALIST_FALLBACK_OVERRIDE_RATE,
    SPECIALIST_FALLBACK_REGRET_RATE,
    SPECIALIST_ROUTER_PATH,
    DeformableSpecialistRouter,
    ExperimentalLaneNotImplementedError,
    SpecialistConfig,
    SpecialistInputError,
    is_specialist_eligible,
    load_deformable_specialist_router,
    route_rl_experimental,
)


# ---------------------------------------------------------------------------
# Paths and shared helpers.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_POLICY_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v6_recovery_baseline_v1.json"
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
    deformable specialist correctly refuses any non-``pass`` policy. To exercise the specialist MECHANICS
    (routing / masking / stats / rollback) we need a *promotable* policy, so this writes a synthetic
    ``verdict=pass`` copy of the committed report once. The refusal-on-abstain path is tested directly
    against the real committed ``RECOVERY_PROMOTION_PATH``."""

    import json
    import tempfile

    data = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
    data["verdict"] = "pass"
    data["reasons"] = []
    out = (
        Path(tempfile.mkdtemp(prefix="specialist_pass_"))
        / "recovery_pass_promotion.json"
    )
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


PASS_PROMOTION_PATH = _synthetic_pass_promotion_path()


def _deformable_dense_state(
    *, attempt_index: str = "0", reobserve: str = "0", last: str = "failed"
) -> RecoveryStateKey:
    return RecoveryStateKey(
        failure_class_bucket=SPECIALIST_FAILURE_CLASS_DEFORMABLE,
        attempt_index_bucket=attempt_index,
        reobserve_count_bucket=reobserve,
        dense_bucket=SPECIALIST_DENSE_BUCKET,
        last_outcome_bucket=last,
    )


def _build_specialist(
    *,
    config: SpecialistConfig | None = None,
) -> DeformableSpecialistRouter:
    return load_deformable_specialist_router(
        policy_artifact_path=RECOVERY_POLICY_PATH,
        promotion_report_path=PASS_PROMOTION_PATH,
        config=config,
    )


# ---------------------------------------------------------------------------
# Config validation.
# ---------------------------------------------------------------------------


class SpecialistConfigTests(unittest.TestCase):
    def test_defaults_are_strict_and_shadow(self) -> None:
        c = SpecialistConfig()
        self.assertEqual(c.canary_pct, 0.0)
        self.assertEqual(
            c.blocked_actions, DEFAULT_SPECIALIST_BLOCKED_ACTIONS
        )
        self.assertEqual(c.window_size, 50)
        self.assertEqual(c.warmup_n, 20)
        self.assertEqual(c.max_regret_rate, 0.20)
        self.assertEqual(c.max_override_rate, 0.95)

    def test_canary_pct_out_of_range(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(canary_pct=-1.0)
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(canary_pct=100.0001)

    def test_window_size_must_be_positive(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(window_size=0)

    def test_warmup_n_non_negative(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(warmup_n=-1)

    def test_max_regret_rate_in_unit_interval(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(max_regret_rate=-0.01)
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(max_regret_rate=1.01)

    def test_max_override_rate_in_unit_interval(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(max_override_rate=2.0)

    def test_blocked_actions_unknown_action_rejected(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(blocked_actions=("not_a_real_action",))

    def test_blocked_actions_cannot_be_all(self) -> None:
        with self.assertRaises(SpecialistInputError):
            SpecialistConfig(blocked_actions=tuple(RECOVERY_ACTIONS))


# ---------------------------------------------------------------------------
# Eligibility predicate.
# ---------------------------------------------------------------------------


class EligibilityTests(unittest.TestCase):
    def test_deformable_dense_is_eligible(self) -> None:
        self.assertTrue(is_specialist_eligible(_deformable_dense_state()))

    def test_deformable_non_dense_not_eligible(self) -> None:
        sk = RecoveryStateKey(
            failure_class_bucket=SPECIALIST_FAILURE_CLASS_DEFORMABLE,
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket="non_dense",
            last_outcome_bucket="failed",
        )
        self.assertFalse(is_specialist_eligible(sk))

    def test_non_deformable_dense_not_eligible(self) -> None:
        sk = RecoveryStateKey(
            failure_class_bucket="slip_after_grasp",
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket=SPECIALIST_DENSE_BUCKET,
            last_outcome_bucket="failed",
        )
        self.assertFalse(is_specialist_eligible(sk))


# ---------------------------------------------------------------------------
# Construction contract — promotion report must be PASS / family / id match.
# ---------------------------------------------------------------------------


def _doctored_promotion(
    tmp_dir: Path, *, mutate: dict[str, Any]
) -> Path:
    base = json.loads(RECOVERY_PROMOTION_PATH.read_text(encoding="utf-8"))
    base.update(mutate)
    out = tmp_dir / "doctored_promotion.json"
    out.write_text(json.dumps(base))
    return out


class ConstructionTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp_path = Path(self._tmp.name)
        self.policy = load_linucb_recovery_policy(RECOVERY_POLICY_PATH)

    def test_construction_succeeds_with_a_passing_policy(self) -> None:
        router = DeformableSpecialistRouter(
            policy=self.policy,
            promotion_report_path=PASS_PROMOTION_PATH,
        )
        self.assertEqual(
            router.promotion_report["policy_family"],
            POLICY_FAMILY_RECOVERY,
        )
        self.assertEqual(
            router.promotion_report["verdict"], PROMOTION_VERDICT_PASS
        )

    def test_committed_recovery_abstains_so_construction_refuses(self) -> None:
        # The committed recovery policy is degenerate and honestly ABSTAINS, so the specialist (which only
        # runs a PROMOTED policy) refuses to build on it. The mechanics tests use the synthetic 'pass'
        # fixture (PASS_PROMOTION_PATH); this pins the honest refusal on the real committed report.
        with self.assertRaises(SpecialistInputError):
            DeformableSpecialistRouter(
                policy=self.policy,
                promotion_report_path=RECOVERY_PROMOTION_PATH,
            )

    def test_rejects_non_pass_verdict(self) -> None:
        doctored = _doctored_promotion(
            self.tmp_path, mutate={"verdict": "warn"}
        )
        with self.assertRaises(SpecialistInputError):
            DeformableSpecialistRouter(
                policy=self.policy, promotion_report_path=doctored
            )

    def test_rejects_wrong_family(self) -> None:
        doctored = _doctored_promotion(
            self.tmp_path, mutate={"policy_family": "some_other_family"}
        )
        with self.assertRaises(SpecialistInputError):
            DeformableSpecialistRouter(
                policy=self.policy, promotion_report_path=doctored
            )

    def test_rejects_mismatched_policy_id(self) -> None:
        doctored = _doctored_promotion(
            self.tmp_path, mutate={"policy_id": "v6_recovery_baseline_v1@999"}
        )
        with self.assertRaises(SpecialistInputError):
            DeformableSpecialistRouter(
                policy=self.policy, promotion_report_path=doctored
            )


# ---------------------------------------------------------------------------
# Routing behavior.
# ---------------------------------------------------------------------------


class NonRoutedTests(unittest.TestCase):
    def test_non_routed_returns_baseline_no_rl_call(self) -> None:
        router = _build_specialist(config=SpecialistConfig(canary_pct=100.0))
        sk = RecoveryStateKey(
            failure_class_bucket="slip_after_grasp",
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket=SPECIALIST_DENSE_BUCKET,
            last_outcome_bucket="failed",
        )
        d = router.choose_recovery(
            request=RecoveryRequest(attempt_id="n1", state_key=sk),
            baseline_action="reobserve",
        )
        self.assertFalse(d.routed_to_specialist)
        self.assertFalse(d.in_canary)
        self.assertIsNone(d.rl_action_proposed)
        self.assertEqual(d.applied_action, "reobserve")
        self.assertFalse(d.override)
        self.assertEqual(d.rl_router_path, SPECIALIST_ROUTER_PATH)


class ShadowOnlyTests(unittest.TestCase):
    def test_default_canary_pct_zero_never_overrides(self) -> None:
        router = _build_specialist()  # default canary_pct=0.0
        self.assertEqual(router.config.canary_pct, 0.0)
        for i in range(20):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"sh-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="reobserve",
            )
            self.assertTrue(d.routed_to_specialist)
            self.assertFalse(d.in_canary)
            self.assertEqual(d.applied_action, "reobserve")
            self.assertFalse(d.override)
        # canary stats never advance in shadow mode.
        self.assertEqual(router.canary_attempts_seen, 0)
        self.assertEqual(router.stats().window_attempts, 0)


class MaskFilteringTests(unittest.TestCase):
    """The recovery baseline proposes ``perturb_and_retry`` for
    deformable+dense; the mask blocks it and falls through to the next
    action in ``ranked_actions`` (``reobserve``)."""

    def test_mask_filters_perturb_to_reobserve(self) -> None:
        router = _build_specialist(config=SpecialistConfig(canary_pct=100.0))
        # baseline = "abort_recovery" so override is observable when the
        # masked RL action ("reobserve") gets applied.
        d = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="m1", state_key=_deformable_dense_state()
            ),
            baseline_action="abort_recovery",
        )
        self.assertTrue(d.routed_to_specialist)
        self.assertTrue(d.in_canary)
        self.assertEqual(d.rl_action_proposed, "perturb_and_retry")
        self.assertTrue(d.mask_filtered_action)
        self.assertFalse(d.rl_action_blocked_by_mask)
        self.assertEqual(d.applied_action, "reobserve")
        self.assertTrue(d.override)

    def test_unmasked_proposal_not_filtered(self) -> None:
        # Block an action the recovery baseline does *not* propose for this
        # state: nothing is filtered.
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0, blocked_actions=("abort_recovery",)
            )
        )
        d = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="um1", state_key=_deformable_dense_state()
            ),
            baseline_action="reobserve",
        )
        self.assertEqual(d.rl_action_proposed, "perturb_and_retry")
        self.assertFalse(d.mask_filtered_action)
        self.assertEqual(d.applied_action, "perturb_and_retry")
        self.assertTrue(d.override)


class FeatureScoreShapeTests(unittest.TestCase):
    def test_decision_carries_five_action_scores(self) -> None:
        router = _build_specialist(config=SpecialistConfig(canary_pct=100.0))
        d = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="f1", state_key=_deformable_dense_state()
            ),
            baseline_action="reobserve",
        )
        actions = {a for a, _ in d.rl_reason_features}
        self.assertEqual(actions, RECOVERY_ACTIONS_SET)
        self.assertEqual(len(d.rl_reason_features), len(RECOVERY_ACTIONS))

    def test_rl_mode_and_policy_id_are_emitted(self) -> None:
        router = _build_specialist(config=SpecialistConfig(canary_pct=100.0))
        d = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="f2", state_key=_deformable_dense_state()
            ),
            baseline_action="reobserve",
        )
        self.assertEqual(d.rl_mode, RL_MODE_RL_ACTIVE)
        self.assertEqual(d.rl_policy_id, router.policy.policy_id)
        self.assertEqual(d.rl_artifact_version, router.policy.version)


# ---------------------------------------------------------------------------
# Outcome reporting + stats.
# ---------------------------------------------------------------------------


class OutcomeAndStatsTests(unittest.TestCase):
    def test_record_outcome_unknown_audit_id_is_no_op(self) -> None:
        router = _build_specialist(config=SpecialistConfig(canary_pct=100.0))
        router.record_outcome("does-not-exist", "succeeded")  # no raise
        self.assertEqual(router.stats().window_attempts, 0)

    def test_record_outcome_rejects_unknown_outcome(self) -> None:
        router = _build_specialist()
        with self.assertRaises(SpecialistInputError):
            router.record_outcome("x", "weird")

    def test_window_counts_track_override_and_regret(self) -> None:
        # baseline != masked applied action -> override=True for every
        # canary attempt.
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=0,
                max_regret_rate=1.0,
                max_override_rate=1.0,
            )
        )
        audit_ids = []
        for i in range(10):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"o-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="abort_recovery",
            )
            self.assertTrue(d.override)
            audit_ids.append(d.audit_id)
        # Mark first 3 overrides as failed -> regret count 3.
        for aid in audit_ids[:3]:
            router.record_outcome(aid, "failed")
        for aid in audit_ids[3:]:
            router.record_outcome(aid, "succeeded")
        s = router.stats()
        self.assertEqual(s.window_attempts, 10)
        self.assertEqual(s.override_count, 10)
        self.assertEqual(s.regret_count, 3)
        self.assertAlmostEqual(s.override_rate, 1.0)
        self.assertAlmostEqual(s.regret_rate, 0.3)


# ---------------------------------------------------------------------------
# Auto-fallback triggers (stricter than the base canary by design).
# ---------------------------------------------------------------------------


class FallbackTriggerTests(unittest.TestCase):
    def test_regret_rate_above_strict_cap_engages_fallback(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=5,
                max_regret_rate=0.20,
                max_override_rate=1.0,
            )
        )
        ids = []
        for i in range(10):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"r-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="abort_recovery",
            )
            ids.append(d.audit_id)
        # 30% regret > 20% cap.
        for aid in ids[:3]:
            router.record_outcome(aid, "failed")
        for aid in ids[3:]:
            router.record_outcome(aid, "succeeded")
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, SPECIALIST_FALLBACK_REGRET_RATE
        )

    def test_below_warmup_does_not_engage(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=50,
                max_regret_rate=0.20,
                max_override_rate=1.0,
            )
        )
        ids = []
        for i in range(5):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"w-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="abort_recovery",
            )
            ids.append(d.audit_id)
        for aid in ids:
            router.record_outcome(aid, "failed")
        self.assertFalse(router.is_fallback_active)

    def test_override_rate_above_cap_engages_fallback(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=5,
                max_regret_rate=1.0,
                max_override_rate=0.5,
            )
        )
        for i in range(10):
            router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"ov-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="abort_recovery",
            )
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, SPECIALIST_FALLBACK_OVERRIDE_RATE
        )

    def test_after_fallback_subsequent_decisions_return_baseline(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=0,
                max_regret_rate=0.0,
                max_override_rate=1.0,
            )
        )
        d1 = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="x1", state_key=_deformable_dense_state()
            ),
            baseline_action="abort_recovery",
        )
        router.record_outcome(d1.audit_id, "failed")
        self.assertTrue(router.is_fallback_active)
        d2 = router.choose_recovery(
            request=RecoveryRequest(
                attempt_id="x2", state_key=_deformable_dense_state()
            ),
            baseline_action="abort_recovery",
        )
        self.assertTrue(d2.fallback_triggered)
        self.assertEqual(d2.applied_action, "abort_recovery")
        self.assertIsNone(d2.rl_action_proposed)


# ---------------------------------------------------------------------------
# Independent kill switch — no cross-coupling with the base canary ActiveCanaryRouter.
# ---------------------------------------------------------------------------


class IndependentKillSwitchTests(unittest.TestCase):
    def test_specialist_fallback_does_not_engage_canary_base(self) -> None:
        base = load_active_canary_router(
            policy_artifact_path=RECOVERY_POLICY_PATH,
            promotion_report_path=PASS_PROMOTION_PATH,
            config=CanaryConfig(canary_pct=100.0),
        )
        specialist = _build_specialist(
            config=SpecialistConfig(canary_pct=100.0)
        )
        specialist.engage_fallback(SPECIALIST_FALLBACK_OPERATOR)
        self.assertTrue(specialist.is_fallback_active)
        self.assertFalse(base.is_fallback_active)

    def test_base_fallback_does_not_engage_specialist(self) -> None:
        base = load_active_canary_router(
            policy_artifact_path=RECOVERY_POLICY_PATH,
            promotion_report_path=PASS_PROMOTION_PATH,
            config=CanaryConfig(canary_pct=100.0),
        )
        specialist = _build_specialist(
            config=SpecialistConfig(canary_pct=100.0)
        )
        base.engage_fallback()
        self.assertTrue(base.is_fallback_active)
        self.assertFalse(specialist.is_fallback_active)


# ---------------------------------------------------------------------------
# Operator controls.
# ---------------------------------------------------------------------------


class OperatorTests(unittest.TestCase):
    def test_engage_fallback_with_default_reason(self) -> None:
        router = _build_specialist()
        router.engage_fallback()
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(router.fallback_reason, SPECIALIST_FALLBACK_OPERATOR)

    def test_engage_fallback_rejects_unknown_reason(self) -> None:
        router = _build_specialist()
        with self.assertRaises(SpecialistInputError):
            router.engage_fallback("not_a_real_reason")

    def test_engage_fallback_is_idempotent(self) -> None:
        router = _build_specialist()
        router.engage_fallback(SPECIALIST_FALLBACK_OPERATOR)
        router.engage_fallback(SPECIALIST_FALLBACK_REGRET_RATE)
        # First reason is preserved.
        self.assertEqual(router.fallback_reason, SPECIALIST_FALLBACK_OPERATOR)

    def test_reset_canary_clears_state(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=0,
                max_regret_rate=1.0,
                max_override_rate=1.0,
            )
        )
        for i in range(5):
            router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"rs-{i}", state_key=_deformable_dense_state()
                ),
                baseline_action="reobserve",
            )
        router.engage_fallback()
        router.reset_canary()
        self.assertFalse(router.is_fallback_active)
        self.assertIsNone(router.fallback_reason)
        self.assertEqual(router.canary_attempts_seen, 0)
        self.assertEqual(router.stats().window_attempts, 0)


# ---------------------------------------------------------------------------
# Experimental lane — typed NotImplementedError, never silent.
# ---------------------------------------------------------------------------


class ExperimentalLaneTests(unittest.TestCase):
    def test_experimental_mode_in_rl_modes(self) -> None:
        self.assertIn(RL_MODE_RL_EXPERIMENTAL, RL_MODES)

    def test_route_rl_experimental_raises_typed_error(self) -> None:
        with self.assertRaises(ExperimentalLaneNotImplementedError) as cm:
            route_rl_experimental()
        self.assertEqual(
            cm.exception.reason_code, EXPERIMENTAL_NOT_IMPLEMENTED_REASON
        )

    def test_experimental_error_is_not_implemented_subclass(self) -> None:
        with self.assertRaises(NotImplementedError):
            route_rl_experimental(state_key="anything", attempt_id="x")


# ---------------------------------------------------------------------------
# Loader edge cases.
# ---------------------------------------------------------------------------


class LoaderTests(unittest.TestCase):
    def test_missing_policy_artifact_raises(self) -> None:
        with self.assertRaises(SpecialistInputError):
            load_deformable_specialist_router(
                policy_artifact_path=Path("/nonexistent/policy.json"),
                promotion_report_path=RECOVERY_PROMOTION_PATH,
            )

    def test_missing_promotion_report_raises(self) -> None:
        with self.assertRaises(SpecialistInputError):
            load_deformable_specialist_router(
                policy_artifact_path=RECOVERY_POLICY_PATH,
                promotion_report_path=Path("/nonexistent/report.json"),
            )


# ---------------------------------------------------------------------------
# End-to-end rollback drill.
# ---------------------------------------------------------------------------


class EndToEndRollbackDrillTests(unittest.TestCase):
    """Mixed traffic, then auto-fallback, then operator-initiated reset
    and re-deployment. Verifies the kill switch is *fully* reversible."""

    def test_full_drill(self) -> None:
        router = _build_specialist(
            config=SpecialistConfig(
                canary_pct=100.0,
                warmup_n=3,
                max_regret_rate=0.20,
                max_override_rate=1.0,
            )
        )
        sk_def = _deformable_dense_state()
        sk_other = RecoveryStateKey(
            failure_class_bucket="slip_after_grasp",
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket=SPECIALIST_DENSE_BUCKET,
            last_outcome_bucket="failed",
        )
        # 5 deformable + 5 non-routed; baseline=abort to drive override.
        ids: list[str] = []
        for i in range(5):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"d-{i}", state_key=sk_def
                ),
                baseline_action="abort_recovery",
            )
            self.assertTrue(d.routed_to_specialist)
            ids.append(d.audit_id)
        for i in range(5):
            d = router.choose_recovery(
                request=RecoveryRequest(
                    attempt_id=f"n-{i}", state_key=sk_other
                ),
                baseline_action="reobserve",
            )
            self.assertFalse(d.routed_to_specialist)
            self.assertEqual(d.applied_action, "reobserve")
        # All 5 routed overrides regress -> fallback engages.
        for aid in ids:
            router.record_outcome(aid, "failed")
        self.assertTrue(router.is_fallback_active)
        self.assertEqual(
            router.fallback_reason, SPECIALIST_FALLBACK_REGRET_RATE
        )
        # Post-fallback decisions return baseline with fallback flag.
        d_post = router.choose_recovery(
            request=RecoveryRequest(attempt_id="post-1", state_key=sk_def),
            baseline_action="abort_recovery",
        )
        self.assertTrue(d_post.fallback_triggered)
        self.assertEqual(d_post.applied_action, "abort_recovery")
        # Operator reset -> back to live canary.
        router.reset_canary()
        self.assertFalse(router.is_fallback_active)
        d_resume = router.choose_recovery(
            request=RecoveryRequest(attempt_id="post-2", state_key=sk_def),
            baseline_action="reobserve",
        )
        # The recovery baseline proposes perturb_and_retry, mask falls through to reobserve.
        self.assertEqual(d_resume.applied_action, "reobserve")
        self.assertTrue(d_resume.mask_filtered_action)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
