"""Conservative offline promotion gate (tests).

Covers ``backend.src.robot.grasping.rl.promotion`` and the
``promote-policy`` CLI subcommand. All tests are stdlib-only and
deterministic. The committed promotion report for the recovery
baseline is also verified byte-for-byte to anchor the artifact format
in CI.
"""

from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.robot.grasping.rl import __main__ as rl_main
from src.robot.grasping.rl.ope import (
    BEHAVIOUR_POLICY_EPSILON,
    ConfidenceInterval,
)
from src.robot.grasping.rl.perception_budget_policy import (
    PERCEPTION_ACTIONS,
)
from src.robot.grasping.rl.promotion import (
    POLICY_FAMILIES,
    POLICY_FAMILY_SEQUENCING,
    POLICY_FAMILY_PERCEPTION,
    POLICY_FAMILY_RECOVERY,
    PROMOTION_NUMERIC_TOLERANCE,
    PROMOTION_REPORT_SCHEMA_VERSION,
    PROMOTION_VERDICT_ABSTAIN,
    PROMOTION_VERDICT_FAIL,
    PROMOTION_VERDICT_PASS,
    EvaluationTriple,
    PromotionInputError,
    PromotionThresholds,
    WISLiftEstimate,
    _per_record_weight,
    build_promotion_report_artifact,
    compute_config_hash,
    compute_dm,
    fit_tabular_dm,
    compute_wis_lift,
    evaluate_policy_for_promotion,
    evaluate_verdict,
    extract_perception_triples,
    extract_recovery_triples,
    hash_file,
    hash_pack_paths,
    load_promotion_report,
    resolve_code_commit,
    write_promotion_report,
)
from src.robot.grasping.rl.recovery_policy import (
    RECOVERY_ACTIONS,
    load_linucb_recovery_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v6_recovery_baseline_v1_promotion_v1.json"
)
COMMITTED_RECOVERY_POLICY_PATH = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies" / "v6_recovery_baseline_v1.json"
)
COMMITTED_PERCEPTION_POLICY_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v5_perception_budget_baseline_v1.json"
)
COMMITTED_SEQUENCING_REPORT_PATH = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v4_sequencing_baseline_v1_promotion_v1.json"
)
COMMITTED_SEQUENCING_POLICY_PATH = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies" / "v4_sequencing_baseline_v1.json"
)
COMMITTED_SEQUENCING_PROMOTION_ARTIFACT_SHA256 = (
    # the sequencing family's committed promotion report. The committed sequencing lookup policy is
    # degenerate (collapses to a single fallback action on the canonical packs) -> honest ABSTAIN, the
    # same pattern as recovery. Uses the RICH OPE reward; means/CIs are platform-locked.
    # re-generated to add reward_model / reward_interpretation / dataset_provenance / verdict_meaning
    # honesty stamps (schema_version 1 -> 2). WIS/DM values unchanged (surgical additive stamp).
    # Re-blessed 2026-09-10 with the tree's own regenerator, after the replay packs stopped
    # being checked out with CRLF (.gitattributes now pins tests/data/replay/** -text). Every
    # provenance hash over those packs moved from a CRLF reading to the LF bytes the committed
    # blob has always held.
    "6a7cd0e213b4ce63aad1cf89622c3c944f72572978ff23b27446f77688715076"
)
DENSE_PACK_PATH = (
    REPO_ROOT / "tests" / "data" / "replay" / "replay_dense_canonical_v1.jsonl"
)
EASY_PACK_PATH = (
    REPO_ROOT / "tests" / "data" / "replay" / "replay_easy_canonical_v1.jsonl"
)

#: Locked sha256 of the committed promotion report. If this changes,
#: the artifact format or the WIS/DM math has drifted — bump the
#: schema and re-generate intentionally.
COMMITTED_PROMOTION_ARTIFACT_SHA256 = (
    # re-generated intentionally: the report now carries the INDEPENDENT data-fit DM (fitted_q_dm) +
    # a circularity note on the legacy self-report direct_method. The verdict stays 'abstain' (the DM is
    # report-only) and wis_lift is unchanged; the new fitted_q_dm.value=0.5598 is the honest baseline rate
    # (vs the circular self-report's off-scale 2.79). Means-of-floats only, so platform-stable.
    # (Prior SHA: 107d1ff4...; the degenerate policy is ABSTAIN + the 0.01 lift floor is set.)
    # re-generated to add reward_model / reward_interpretation / dataset_provenance / verdict_meaning
    # honesty stamps (schema_version 1 -> 2). WIS/DM values unchanged (surgical additive stamp).
    # Re-blessed 2026-09-10 with the tree's own regenerator, after the replay packs stopped
    # being checked out with CRLF (.gitattributes now pins tests/data/replay/** -text). Every
    # provenance hash over those packs moved from a CRLF reading to the LF bytes the committed
    # blob has always held.
    # This report also carried a `config_hash` from when the recovery action space was smaller
    # than today's, and its WIS estimates moved in the last few ULPs with the pack bytes.
    "ecada334eebd2f8da96b79cfc32108ee0406a9f4bf941a0c07a9732fb1d0a0c5"
)


# ---------------------------------------------------------------------------
# Thresholds.
# ---------------------------------------------------------------------------


class PromotionThresholdsTests(unittest.TestCase):
    def test_defaults_are_conservative(self) -> None:
        t = PromotionThresholds()
        # a positive point-lift floor -- zero/float-noise lift no longer passes.
        self.assertEqual(t.min_lift_over_baseline, 0.01)
        self.assertEqual(t.min_lower_bound_lift, 0.0)
        self.assertEqual(t.min_n_effective, 10.0)
        self.assertEqual(t.min_records_with_weight, 5)

    def test_rejects_negative_n_effective(self) -> None:
        with self.assertRaises(PromotionInputError):
            PromotionThresholds(min_n_effective=-1.0)

    def test_rejects_negative_records_with_weight(self) -> None:
        with self.assertRaises(PromotionInputError):
            PromotionThresholds(min_records_with_weight=-1)


# ---------------------------------------------------------------------------
# Per-record weight.
# ---------------------------------------------------------------------------


class PerRecordWeightTests(unittest.TestCase):
    def test_zero_weight_on_disagreement(self) -> None:
        w, clipped = _per_record_weight(
            "a", "b", num_actions=5, epsilon=0.05, clip=100.0
        )
        self.assertEqual(w, 0.0)
        self.assertFalse(clipped)

    def test_positive_weight_on_agreement(self) -> None:
        w, clipped = _per_record_weight(
            "a", "a", num_actions=5, epsilon=0.05, clip=100.0
        )
        self.assertGreater(w, 0.0)
        self.assertFalse(clipped)

    def test_clip_applied_when_weight_exceeds(self) -> None:
        w, clipped = _per_record_weight(
            "a", "a", num_actions=5, epsilon=0.05, clip=0.5
        )
        self.assertEqual(w, 0.5)
        self.assertTrue(clipped)


# ---------------------------------------------------------------------------
# WIS lift.
# ---------------------------------------------------------------------------


class ComputeWISLiftTests(unittest.TestCase):
    def test_empty_returns_zeros(self) -> None:
        est = compute_wis_lift([], num_actions=2, seed=1)
        self.assertEqual(est.num_records, 0)
        self.assertEqual(est.target_value, 0.0)
        self.assertEqual(est.baseline_value, 0.0)
        self.assertEqual(est.lift, 0.0)
        self.assertEqual(est.effective_sample_size, 0.0)

    def test_full_agreement_reproduces_mean(self) -> None:
        triples = [
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 0.0),
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 1.0),
        ]
        est = compute_wis_lift(triples, num_actions=3, seed=7)
        self.assertAlmostEqual(est.target_value, 0.75)
        self.assertAlmostEqual(est.baseline_value, 0.75)
        self.assertAlmostEqual(est.lift, 0.0)
        self.assertEqual(est.num_records, 4)
        self.assertEqual(est.num_records_with_weight, 4)

    def test_all_disagree_yields_zero_target_value(self) -> None:
        triples = [
            EvaluationTriple("a", "b", 0.0, 1.0),
            EvaluationTriple("a", "b", 0.0, 0.0),
        ]
        est = compute_wis_lift(triples, num_actions=2, seed=0)
        self.assertEqual(est.target_value, 0.0)
        self.assertEqual(est.num_records_with_weight, 0)
        self.assertAlmostEqual(est.baseline_value, 0.5)
        self.assertAlmostEqual(est.lift, -0.5)

    def test_bootstrap_is_deterministic_with_seed(self) -> None:
        triples = [
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "b", 0.0, 0.0),
            EvaluationTriple("a", "a", 0.0, 0.0),
            EvaluationTriple("b", "a", 0.0, 1.0),
        ]
        e1 = compute_wis_lift(triples, num_actions=2, seed=42)
        e2 = compute_wis_lift(triples, num_actions=2, seed=42)
        self.assertEqual(e1.lift_ci.lower, e2.lift_ci.lower)
        self.assertEqual(e1.lift_ci.upper, e2.lift_ci.upper)

    def test_bootstrap_different_seeds_differ(self) -> None:
        # 12 triples with mixed agreement/reward so resamples vary.
        triples = [
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "b", 0.0, 0.0),
            EvaluationTriple("a", "a", 0.0, 0.0),
            EvaluationTriple("b", "a", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("b", "b", 0.0, 0.0),
            EvaluationTriple("a", "a", 0.0, 0.0),
            EvaluationTriple("b", "b", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "b", 0.0, 1.0),
            EvaluationTriple("b", "a", 0.0, 0.0),
            EvaluationTriple("a", "a", 0.0, 0.5),
        ]
        e1 = compute_wis_lift(triples, num_actions=2, seed=1)
        e2 = compute_wis_lift(triples, num_actions=2, seed=99999)
        # Two distinct seeds must produce distinct bootstrap CIs given
        # this much variability.
        self.assertNotEqual(
            (e1.lift_ci.lower, e1.lift_ci.upper),
            (e2.lift_ci.lower, e2.lift_ci.upper),
        )

    def test_weight_concentration_within_bounds(self) -> None:
        triples = [
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 1.0),
            EvaluationTriple("a", "a", 0.0, 0.0),
        ]
        est = compute_wis_lift(triples, num_actions=2, seed=0)
        # all weights equal → wci = 1/n
        self.assertAlmostEqual(est.weight_concentration_index, 1.0 / 3.0)


# ---------------------------------------------------------------------------
# DM.
# ---------------------------------------------------------------------------


class ComputeDMTests(unittest.TestCase):
    def test_empty(self) -> None:
        dm = compute_dm([])
        self.assertEqual(dm.value, 0.0)
        self.assertEqual(dm.num_records, 0)

    def test_mean_of_expected_reward(self) -> None:
        triples = [
            EvaluationTriple("a", "a", 0.2, 0.0),
            EvaluationTriple("a", "a", 0.8, 1.0),
        ]
        dm = compute_dm(triples)
        self.assertAlmostEqual(dm.value, 0.5)
        self.assertEqual(dm.num_records, 2)


class FitTabularDMTests(unittest.TestCase):
    """The INDEPENDENT data-fit DM uses LOGGED rewards, never the policy's circular self-report."""

    def test_uses_logged_reward_not_self_report(self) -> None:
        # The circular self-report (target_expected_reward) is an inflated 99.0; the LOGGED reward is 1.0.
        # The independent DM must reflect the logged 1.0 -- it must be impossible for a policy to inflate
        # this number by merely claiming a high expected reward.
        triples = [
            EvaluationTriple("a", "a", 99.0, 1.0, "s1"),
            EvaluationTriple("a", "a", 99.0, 1.0, "s1"),
        ]
        dm = fit_tabular_dm(triples)
        self.assertEqual(dm.value, 1.0)
        self.assertEqual(dm.fraction_covered, 1.0)
        self.assertEqual(dm.num_state_action_cells, 1)

    def test_fraction_covered_surfaces_unsupported_target(self) -> None:
        # In s2 the behaviour only ever took 'b'; a target of 'a' there has NO logged support, so it is
        # uncovered and fraction_covered drops -- the limitation is auditable, never silently filled.
        triples = [
            EvaluationTriple("a", "a", 0.0, 1.0, "s1"),
            EvaluationTriple("b", "a", 0.0, 0.0, "s2"),
        ]
        dm = fit_tabular_dm(triples)
        self.assertEqual(dm.fraction_covered, 0.5)
        self.assertEqual(dm.num_covered_records, 1)
        self.assertEqual(dm.value, 1.0)  # only the covered s1 cell (Q=1.0) contributes

    def test_empty(self) -> None:
        dm = fit_tabular_dm([])
        self.assertEqual(dm.value, 0.0)
        self.assertEqual(dm.fraction_covered, 0.0)
        self.assertEqual(dm.num_records, 0)


# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------


def _wis(
    *,
    n: int = 100,
    n_w: int = 50,
    ess: float = 50.0,
    lift: float = 0.1,
    lift_lower: float = 0.05,
    lift_upper: float = 0.2,
) -> WISLiftEstimate:
    return WISLiftEstimate(
        target_value=lift + 0.5,
        baseline_value=0.5,
        lift=lift,
        lift_ci=ConfidenceInterval("bootstrap", lift_lower, lift_upper, n),
        target_ci=ConfidenceInterval("bootstrap", 0.5, 0.7, n),
        num_records=n,
        num_records_with_weight=n_w,
        num_clipped=0,
        effective_sample_size=ess,
        weight_concentration_index=0.1,
    )


class EvaluateVerdictTests(unittest.TestCase):
    def test_abstain_when_no_records(self) -> None:
        wis = _wis(n=0, n_w=0, ess=0.0)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_ABSTAIN)
        self.assertIn("no_records", reasons)

    def test_abstain_on_low_ess(self) -> None:
        wis = _wis(n=20, n_w=10, ess=2.0)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_ABSTAIN)
        self.assertIn("insufficient_effective_sample_size", reasons)

    def test_abstain_on_low_records_with_weight(self) -> None:
        wis = _wis(n=20, n_w=1, ess=20.0)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_ABSTAIN)
        self.assertIn("insufficient_records_with_weight", reasons)

    def test_fail_on_negative_lift(self) -> None:
        wis = _wis(lift=-0.1, lift_lower=-0.2, lift_upper=0.0)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_FAIL)
        self.assertIn("lift_below_threshold", reasons)

    def test_fail_on_negative_lower_bound(self) -> None:
        wis = _wis(lift=0.05, lift_lower=-0.1, lift_upper=0.2)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_FAIL)
        self.assertIn("lift_lower_bound_below_threshold", reasons)

    def test_pass_on_positive_lift_with_nonneg_lower(self) -> None:
        wis = _wis(lift=0.1, lift_lower=0.0, lift_upper=0.2)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds()
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_PASS)
        self.assertEqual(reasons, ())

    def test_pass_tolerates_floating_point_noise_at_the_floor(self) -> None:
        # The PROMOTION_NUMERIC_TOLERANCE still absorbs float noise AT the (now positive) floor: a lift a
        # hair below 0.01 (within tolerance) still passes; the lower bound a hair below 0.0 still passes.
        wis = _wis(lift=0.01 - 1e-12, lift_lower=-1e-12, lift_upper=0.02)
        verdict, _ = evaluate_verdict(wis=wis, thresholds=PromotionThresholds())
        self.assertEqual(verdict, PROMOTION_VERDICT_PASS)

    def test_zero_lift_no_longer_passes(self) -> None:
        # a mathematically-zero / float-noise lift must FAIL under the positive floor -- "does no
        # harm" is not "is better".
        wis = _wis(lift=-1e-15, lift_lower=-1e-15, lift_upper=1e-15)
        verdict, reasons = evaluate_verdict(wis=wis, thresholds=PromotionThresholds())
        self.assertEqual(verdict, PROMOTION_VERDICT_FAIL)
        self.assertIn("lift_below_threshold", reasons)

    def test_degenerate_single_action_policy_abstains(self) -> None:
        # a target policy proposing <2 distinct actions is un-judgeable by WIS (its action ~always
        # matches the behaviour action, so the importance weights collapse) -> abstain, even on a "great"
        # nominal lift, rather than rubber-stamp a do-nothing policy.
        wis = _wis(lift=0.5, lift_lower=0.4, lift_upper=0.6)
        verdict, reasons = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds(), n_distinct_target_actions=1
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_ABSTAIN)
        self.assertIn("degenerate_single_action_policy", reasons)

    def test_non_degenerate_policy_is_not_abstained_for_degeneracy(self) -> None:
        wis = _wis(lift=0.1, lift_lower=0.0, lift_upper=0.2)
        verdict, _ = evaluate_verdict(
            wis=wis, thresholds=PromotionThresholds(), n_distinct_target_actions=3
        )
        self.assertEqual(verdict, PROMOTION_VERDICT_PASS)


# ---------------------------------------------------------------------------
# Triple extraction.
# ---------------------------------------------------------------------------


class ExtractRecoveryTriplesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = load_linucb_recovery_policy(COMMITTED_RECOVERY_POLICY_PATH)

    def test_handles_empty_records(self) -> None:
        res = extract_recovery_triples(
            policy=self.policy, records=[], training_scope="dense"
        )
        self.assertEqual(res.triples, ())
        self.assertEqual(res.num_records_total, 0)

    def test_extracts_from_dense_pack(self) -> None:
        with open(DENSE_PACK_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        res = extract_recovery_triples(
            policy=self.policy, records=records, training_scope="dense"
        )
        self.assertEqual(res.num_records_total, len(records))
        self.assertGreater(len(res.triples), 0)
        for t in res.triples:
            self.assertIn(t.target_action, RECOVERY_ACTIONS)
            self.assertIn(t.reward, (0.0, 1.0))


class ExtractPerceptionTriplesTests(unittest.TestCase):
    def test_extracts_from_dense_pack(self) -> None:
        from src.robot.grasping.rl.perception_budget_policy import (
            load_linucb_perception_budget_policy,
        )

        if not COMMITTED_PERCEPTION_POLICY_PATH.exists():
            self.skipTest("perception-budget baseline artifact not committed")
        policy = load_linucb_perception_budget_policy(COMMITTED_PERCEPTION_POLICY_PATH)
        with open(DENSE_PACK_PATH, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        res = extract_perception_triples(
            policy=policy, records=records, training_scope="dense"
        )
        self.assertEqual(res.num_records_total, len(records))
        for t in res.triples:
            self.assertIn(t.target_action, PERCEPTION_ACTIONS)


# ---------------------------------------------------------------------------
# Hashes.
# ---------------------------------------------------------------------------


class ConfigHashTests(unittest.TestCase):
    def test_deterministic(self) -> None:
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"a": 1, "b": 2})
        self.assertEqual(h1, h2)

    def test_order_independent(self) -> None:
        h1 = compute_config_hash({"a": 1, "b": 2})
        h2 = compute_config_hash({"b": 2, "a": 1})
        self.assertEqual(h1, h2)

    def test_differs_on_value_change(self) -> None:
        h1 = compute_config_hash({"a": 1})
        h2 = compute_config_hash({"a": 2})
        self.assertNotEqual(h1, h2)


class HashFileAndPacksTests(unittest.TestCase):
    def test_hash_file_deterministic(self) -> None:
        h1 = hash_file(COMMITTED_RECOVERY_POLICY_PATH)
        h2 = hash_file(COMMITTED_RECOVERY_POLICY_PATH)
        self.assertEqual(h1, h2)

    def test_hash_pack_paths_order_independent(self) -> None:
        h1 = hash_pack_paths([EASY_PACK_PATH, DENSE_PACK_PATH])
        h2 = hash_pack_paths([DENSE_PACK_PATH, EASY_PACK_PATH])
        self.assertEqual(h1, h2)


class CodeCommitTests(unittest.TestCase):
    def test_returns_string(self) -> None:
        out = resolve_code_commit(REPO_ROOT)
        self.assertIsInstance(out, str)
        # either a real sha or "unknown" — both valid
        self.assertTrue(len(out) > 0)

    def test_unknown_on_non_git_dir(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = resolve_code_commit(Path(td))
            self.assertEqual(out, "unknown")


# ---------------------------------------------------------------------------
# Artifact round-trip.
# ---------------------------------------------------------------------------


class ArtifactRoundTripTests(unittest.TestCase):
    def _make_report(self):
        return evaluate_policy_for_promotion(
            policy_family=POLICY_FAMILY_RECOVERY,
            policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
            pack_paths=[EASY_PACK_PATH, DENSE_PACK_PATH],
            dataset_id="v7_promotion_canonical",
            training_scope="dense",
            seed=1,
            thresholds=PromotionThresholds(),
            repo_root=REPO_ROOT,
        )

    def test_write_then_load(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "report.json"
            sha = write_promotion_report(report, out)
            self.assertEqual(len(sha), 64)
            loaded = load_promotion_report(out)
            self.assertEqual(loaded["schema_version"], 2)
            self.assertEqual(loaded["policy_family"], POLICY_FAMILY_RECOVERY)
            self.assertIn(loaded["verdict"], (
                PROMOTION_VERDICT_PASS,
                PROMOTION_VERDICT_FAIL,
                PROMOTION_VERDICT_ABSTAIN,
            ))

    def test_load_rejects_schema_drift(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            write_promotion_report(report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            data["schema_version"] = 999
            out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(PromotionInputError):
                load_promotion_report(out)

    def test_load_rejects_family_drift(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            write_promotion_report(report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            data["policy_family"] = "v99_unknown"
            out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(PromotionInputError):
                load_promotion_report(out)

    def test_load_rejects_verdict_drift(self) -> None:
        report = self._make_report()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r.json"
            write_promotion_report(report, out)
            data = json.loads(out.read_text(encoding="utf-8"))
            data["verdict"] = "maybe"
            out.write_text(json.dumps(data, sort_keys=True, indent=2) + "\n")
            with self.assertRaises(PromotionInputError):
                load_promotion_report(out)


# ---------------------------------------------------------------------------
# End-to-end on committed recovery baseline.
# ---------------------------------------------------------------------------


class EvaluatePolicyForPromotionRecoveryTests(unittest.TestCase):
    def test_recovery_dense_produces_report(self) -> None:
        report = evaluate_policy_for_promotion(
            policy_family=POLICY_FAMILY_RECOVERY,
            policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
            pack_paths=[DENSE_PACK_PATH],
            dataset_id="v7_promotion_canonical",
            training_scope="dense",
            seed=1,
            thresholds=PromotionThresholds(),
            repo_root=REPO_ROOT,
        )
        self.assertEqual(report.policy_family, POLICY_FAMILY_RECOVERY)
        self.assertEqual(report.schema_version, 2)
        self.assertGreater(report.extraction.num_records_in_scope, 0)
        self.assertIn(report.verdict, (
            PROMOTION_VERDICT_PASS,
            PROMOTION_VERDICT_FAIL,
            PROMOTION_VERDICT_ABSTAIN,
        ))

    def test_rejects_unknown_family(self) -> None:
        with self.assertRaises(PromotionInputError):
            evaluate_policy_for_promotion(
                policy_family="v99",
                policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
                pack_paths=[DENSE_PACK_PATH],
                dataset_id="x",
                training_scope="dense",
                seed=1,
            )

    def test_rejects_missing_artifact(self) -> None:
        with self.assertRaises(PromotionInputError):
            evaluate_policy_for_promotion(
                policy_family=POLICY_FAMILY_RECOVERY,
                policy_artifact_path=Path("/does/not/exist.json"),
                pack_paths=[DENSE_PACK_PATH],
                dataset_id="x",
                training_scope="dense",
                seed=1,
            )

    def test_rejects_missing_pack(self) -> None:
        with self.assertRaises(PromotionInputError):
            evaluate_policy_for_promotion(
                policy_family=POLICY_FAMILY_RECOVERY,
                policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
                pack_paths=[Path("/does/not/exist.jsonl")],
                dataset_id="x",
                training_scope="dense",
                seed=1,
            )

    def test_rejects_no_packs(self) -> None:
        with self.assertRaises(PromotionInputError):
            evaluate_policy_for_promotion(
                policy_family=POLICY_FAMILY_RECOVERY,
                policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
                pack_paths=[],
                dataset_id="x",
                training_scope="dense",
                seed=1,
            )


# ---------------------------------------------------------------------------
# Committed report.
# ---------------------------------------------------------------------------


class CommittedPromotionReportTests(unittest.TestCase):
    def test_committed_report_exists(self) -> None:
        self.assertTrue(
            COMMITTED_REPORT_PATH.exists(),
            f"committed promotion report missing at {COMMITTED_REPORT_PATH}",
        )

    def test_committed_report_sha256_matches(self) -> None:
        sha = hash_file(COMMITTED_REPORT_PATH)
        self.assertEqual(sha, COMMITTED_PROMOTION_ARTIFACT_SHA256)

    def test_committed_report_loads_clean(self) -> None:
        data = load_promotion_report(COMMITTED_REPORT_PATH)
        self.assertEqual(data["schema_version"], PROMOTION_REPORT_SCHEMA_VERSION)
        self.assertEqual(data["policy_family"], POLICY_FAMILY_RECOVERY)
        self.assertEqual(data["report_kind"], "promotion")
        self.assertEqual(data["seed"], 1)
        self.assertIn("estimators", data)
        self.assertIn("wis", data["estimators"])
        self.assertIn("direct_method", data["estimators"])
        self.assertIn("fqe", data["estimators"])
        # the legacy DM is flagged circular, and the independent data-fit DM is reported alongside.
        self.assertTrue(data["estimators"]["direct_method"]["circular"])
        self.assertIn("fitted_q_dm", data["estimators"])
        self.assertIn("fraction_covered", data["estimators"]["fitted_q_dm"])

    def test_committed_report_is_reproducible(self) -> None:
        report = evaluate_policy_for_promotion(
            policy_family=POLICY_FAMILY_RECOVERY,
            policy_artifact_path=COMMITTED_RECOVERY_POLICY_PATH,
            pack_paths=[EASY_PACK_PATH, DENSE_PACK_PATH],
            dataset_id="v7_promotion_canonical",
            training_scope="dense",
            seed=1,
            thresholds=PromotionThresholds(),
            repo_root=REPO_ROOT,
        )
        artifact = build_promotion_report_artifact(report)
        # generated_at_iso and code_commit are environment-sensitive;
        # compare the deterministic body.
        committed = load_promotion_report(COMMITTED_REPORT_PATH)
        for k in (
            "schema_version",
            "policy_family",
            "policy_id",
            "policy_version",
            "policy_artifact_sha256",
            "dataset_id",
            "dataset_hash",
            "training_scope",
            "seed",
            "config_hash",
            "thresholds",
            "estimators",
            "extraction",
            "verdict",
            "reasons",
        ):
            self.assertEqual(
                artifact[k], committed[k],
                f"mismatch on key {k!r}",
            )


class CommittedSequencingPromotionReportTests(unittest.TestCase):
    """The sequencing family is now off-policy-judgeable and gets a typed verdict. On the committed
    degenerate lookup policy (collapses to a single fallback action) the honest verdict is ABSTAIN -- the
    same pattern as recovery -- so the gate now COVERS sequencing instead of structurally excluding it."""

    def test_committed_report_exists(self) -> None:
        self.assertTrue(
            COMMITTED_SEQUENCING_REPORT_PATH.exists(),
            f"committed sequencing promotion report missing at {COMMITTED_SEQUENCING_REPORT_PATH}",
        )

    def test_committed_report_sha256_matches(self) -> None:
        sha = hash_file(COMMITTED_SEQUENCING_REPORT_PATH)
        self.assertEqual(sha, COMMITTED_SEQUENCING_PROMOTION_ARTIFACT_SHA256)

    def test_committed_report_loads_clean(self) -> None:
        data = load_promotion_report(COMMITTED_SEQUENCING_REPORT_PATH)
        self.assertEqual(data["policy_family"], POLICY_FAMILY_SEQUENCING)
        # The committed sequencing lookup policy is degenerate -> honest abstain, not a rubber-stamp.
        self.assertEqual(data["verdict"], "abstain")
        self.assertIn("degenerate_single_action_policy", data["reasons"])
        self.assertIn("fitted_q_dm", data["estimators"])

    def test_committed_report_is_reproducible(self) -> None:
        report = evaluate_policy_for_promotion(
            policy_family=POLICY_FAMILY_SEQUENCING,
            policy_artifact_path=COMMITTED_SEQUENCING_POLICY_PATH,
            pack_paths=[EASY_PACK_PATH, DENSE_PACK_PATH],
            dataset_id="v7_promotion_canonical",
            training_scope="dense",
            seed=1,
            thresholds=PromotionThresholds(),
            repo_root=REPO_ROOT,
        )
        artifact = build_promotion_report_artifact(report)
        committed = load_promotion_report(COMMITTED_SEQUENCING_REPORT_PATH)
        for k in (
            "schema_version",
            "policy_family",
            "policy_id",
            "policy_version",
            "policy_artifact_sha256",
            "dataset_id",
            "dataset_hash",
            "training_scope",
            "seed",
            "config_hash",
            "thresholds",
            "estimators",
            "extraction",
            "verdict",
            "reasons",
        ):
            self.assertEqual(
                artifact[k], committed[k],
                f"mismatch on key {k!r}",
            )


# ---------------------------------------------------------------------------
# CLI smoke.
# ---------------------------------------------------------------------------


class PromotePolicyCLITests(unittest.TestCase):
    def test_subparser_registered(self) -> None:
        parser = rl_main.build_parser()
        with redirect_stderr(io.StringIO()):
            args = parser.parse_args([
                "promote-policy",
                "--policy-family", POLICY_FAMILY_RECOVERY,
                "--policy-artifact", str(COMMITTED_RECOVERY_POLICY_PATH),
                "--replay-pack", str(DENSE_PACK_PATH),
            ])
        self.assertEqual(args.policy_family, POLICY_FAMILY_RECOVERY)
        self.assertEqual(args.training_scope, "dense")
        self.assertEqual(args.seed, 1)

    def test_smoke_run_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "smoke.json"
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = rl_main.main([
                    "promote-policy",
                    "--policy-family", POLICY_FAMILY_RECOVERY,
                    "--policy-artifact", str(COMMITTED_RECOVERY_POLICY_PATH),
                    "--replay-pack", str(DENSE_PACK_PATH),
                    "--output", str(out),
                    "--seed", "1",
                ])
            self.assertEqual(rc, 0)
            self.assertTrue(out.exists())
            summary = json.loads(buf.getvalue())
            self.assertEqual(summary["policy_family"], POLICY_FAMILY_RECOVERY)
            self.assertIn(summary["verdict"], (
                PROMOTION_VERDICT_PASS,
                PROMOTION_VERDICT_FAIL,
                PROMOTION_VERDICT_ABSTAIN,
            ))

    def test_rejects_unknown_family_via_cli(self) -> None:
        parser = rl_main.build_parser()
        with self.assertRaises(SystemExit):
            with redirect_stderr(io.StringIO()):
                parser.parse_args([
                    "promote-policy",
                    "--policy-family", "v99",
                    "--policy-artifact", str(COMMITTED_RECOVERY_POLICY_PATH),
                    "--replay-pack", str(DENSE_PACK_PATH),
                ])


# ---------------------------------------------------------------------------
# Module constants.
# ---------------------------------------------------------------------------


class ModuleConstantsTests(unittest.TestCase):
    def test_schema_version_locked(self) -> None:
        self.assertEqual(PROMOTION_REPORT_SCHEMA_VERSION, 2)

    def test_tolerance_is_small(self) -> None:
        self.assertGreater(PROMOTION_NUMERIC_TOLERANCE, 0.0)
        self.assertLess(PROMOTION_NUMERIC_TOLERANCE, 1e-6)

    def test_policy_families_tuple(self) -> None:
        self.assertEqual(
            set(POLICY_FAMILIES),
            {
                POLICY_FAMILY_SEQUENCING,
                POLICY_FAMILY_PERCEPTION,
                POLICY_FAMILY_RECOVERY,
            },
        )

    def test_epsilon_matches_ope(self) -> None:
        # the gate's epsilon must come from ope.py — single source of truth
        self.assertEqual(BEHAVIOUR_POLICY_EPSILON, 0.05)


if __name__ == "__main__":
    unittest.main()
