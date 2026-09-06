"""OPE harness unit tests.

Covers all locked design choices:
reward shape, epsilon-smoothed behaviour policy, WIS,
FQE-as-bandit + DM, bootstrap & normal CIs,
report shape + warning flags, and strict malformed-input
failure.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl import ope as ope_mod
from src.robot.grasping.rl.ope import (
    BEHAVIOUR_POLICY_EPSILON,
    BOOTSTRAP_N,
    IMPORTANCE_WEIGHT_CLIP,
    OPE_REPORT_SCHEMA_VERSION,
    OPE_REWARD_COEFFICIENTS_V1,
    MalformedOPEInputError,
    attempt_level_wis,
    behaviour_action_probability,
    build_ope_report,
    build_sequencing_section,
    compute_record_reward,
    direct_method_sequencing,
    fit_sequencing_tabular_q,
    wis_sequencing,
)
from src.robot.grasping.rl.sequencing_policy import (
    SEQUENCING_ACTIONS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _ok_record(
    *,
    attempt_id: str = "a",
    outcome: str = "succeeded",
    cycle_time_s: float = 1.0,
    extra_overrides: dict | None = None,
) -> dict:
    extra = {"cycle_time_s": cycle_time_s}
    if extra_overrides:
        extra.update(extra_overrides)
    rec: dict = {
        "schema_version": 1,
        "attempt_id": attempt_id,
        "mode": "easy",
        "final_outcome": outcome,
        "execution": {"outcome": "executed" if outcome == "succeeded" else "failed"},
        "verification": {"outcome": "passed" if outcome == "succeeded" else "failed"},
        "recovery_actions": [],
        "extra": extra,
    }
    return rec


# ---------------------------------------------------------------------------
# Constants.
# ---------------------------------------------------------------------------


class ConstantsTests(unittest.TestCase):
    def test_schema_version_locked(self) -> None:
        self.assertEqual(OPE_REPORT_SCHEMA_VERSION, 2)

    def test_reward_coefficients_locked_v1(self) -> None:
        self.assertEqual(
            dict(OPE_REWARD_COEFFICIENTS_V1),
            {
                "success": 1.0,
                "failed_attempt": -0.5,
                "cycle_time_seconds": -0.3,
                "collision_event": -1.5,
                "dead_loop": -2.0,
                "perception_cost": -0.2,
            },
        )

    def test_epsilon_smoothing_locked(self) -> None:
        self.assertEqual(BEHAVIOUR_POLICY_EPSILON, 0.05)

    def test_bootstrap_n_locked(self) -> None:
        self.assertEqual(BOOTSTRAP_N, 1000)

    def test_clip_locked(self) -> None:
        self.assertEqual(IMPORTANCE_WEIGHT_CLIP, 100.0)


# ---------------------------------------------------------------------------
# Reward.
# ---------------------------------------------------------------------------


class RewardTests(unittest.TestCase):
    def test_success_reward(self) -> None:
        r = compute_record_reward(_ok_record(outcome="succeeded", cycle_time_s=2.0))
        # +1.0 - 0.3 * 2.0 = 0.4
        self.assertAlmostEqual(r, 0.4)

    def test_failure_reward(self) -> None:
        r = compute_record_reward(_ok_record(outcome="execution_failed", cycle_time_s=1.0))
        # -0.5 - 0.3 = -0.8
        self.assertAlmostEqual(r, -0.8)

    def test_collision_reward(self) -> None:
        r = compute_record_reward(
            _ok_record(
                outcome="execution_failed",
                cycle_time_s=0.0,
                extra_overrides={"collision_event": True},
            )
        )
        # -0.5 + -1.5 = -2.0
        self.assertAlmostEqual(r, -2.0)

    def test_dead_loop_reward(self) -> None:
        r = compute_record_reward(
            _ok_record(
                outcome="recovery_exhausted",
                cycle_time_s=0.0,
                extra_overrides={"dead_loop": True},
            )
        )
        # -0.5 + -2.0 = -2.5
        self.assertAlmostEqual(r, -2.5)

    def test_perception_cost_reward(self) -> None:
        r = compute_record_reward(
            _ok_record(
                outcome="succeeded",
                cycle_time_s=0.0,
                extra_overrides={"perception_cost": 3.0},
            )
        )
        # +1.0 + -0.2 * 3 = 0.4
        self.assertAlmostEqual(r, 0.4)


# ---------------------------------------------------------------------------
# Epsilon-smoothed behaviour policy.
# ---------------------------------------------------------------------------


class BehaviourProbabilityTests(unittest.TestCase):
    def test_chosen_probability(self) -> None:
        p = behaviour_action_probability(num_actions=4, chosen=True)
        # (1 - 0.05) + 0.05/4 = 0.9625
        self.assertAlmostEqual(p, 0.9625)

    def test_not_chosen_probability(self) -> None:
        p = behaviour_action_probability(num_actions=4, chosen=False)
        # 0.05 / 4 = 0.0125
        self.assertAlmostEqual(p, 0.0125)

    def test_sums_to_one_across_actions(self) -> None:
        k = 5
        total = behaviour_action_probability(num_actions=k, chosen=True) + (
            k - 1
        ) * behaviour_action_probability(num_actions=k, chosen=False)
        self.assertAlmostEqual(total, 1.0)

    def test_invalid_num_actions(self) -> None:
        with self.assertRaises(MalformedOPEInputError):
            behaviour_action_probability(num_actions=0, chosen=True)

    def test_invalid_epsilon(self) -> None:
        with self.assertRaises(MalformedOPEInputError):
            behaviour_action_probability(num_actions=2, chosen=True, epsilon=1.5)


# ---------------------------------------------------------------------------
# Attempt-level WIS (candidate/ranking path).
# ---------------------------------------------------------------------------


class AttemptLevelWISTests(unittest.TestCase):
    def test_collapses_to_mean_reward_when_probs_default(self) -> None:
        recs = [
            _ok_record(attempt_id="a", outcome="succeeded", cycle_time_s=0.0),
            _ok_record(attempt_id="b", outcome="execution_failed", cycle_time_s=0.0),
        ]
        wis = attempt_level_wis(recs)
        # rewards = [+1.0, -0.5]; ratios = 1.0 ⇒ WIS = mean = 0.25
        self.assertAlmostEqual(wis.value, 0.25)
        self.assertEqual(wis.num_records, 2)
        self.assertEqual(wis.num_clipped, 0)

    def test_clipping_kicks_in(self) -> None:
        recs = [_ok_record(attempt_id="a", outcome="succeeded", cycle_time_s=0.0)]
        wis = attempt_level_wis(recs, target_probs=[1.0], behaviour_probs=[1e-6])
        self.assertEqual(wis.num_clipped, 1)
        # value = (clip * 1.0) / clip = 1.0
        self.assertAlmostEqual(wis.value, 1.0)

    def test_length_mismatch_raises(self) -> None:
        recs = [_ok_record()]
        with self.assertRaises(MalformedOPEInputError):
            attempt_level_wis(recs, target_probs=[1.0, 1.0], behaviour_probs=[1.0])


# ---------------------------------------------------------------------------
# Sequencing tabular Q + DM + WIS.
# ---------------------------------------------------------------------------


def _build_sequencing_failure_sequence() -> list[dict]:
    """Synthetic two-attempt episode that exposes a sequencing (s,a,r) pair."""

    # First attempt fails (gives sequencing the state), second attempt succeeds
    # with action derived from successor (a retry-with-* style action).
    rec1 = _ok_record(
        attempt_id="ep1-0",
        outcome="execution_failed",
        cycle_time_s=0.0,
    )
    # Second attempt: success → reward = +1.0; cycle_time 0 to keep math clean
    rec2 = _ok_record(attempt_id="ep1-1", outcome="succeeded", cycle_time_s=0.0)
    return [rec1, rec2]


class SequencingTabularQTests(unittest.TestCase):
    def test_q_table_records_mean_reward(self) -> None:
        recs = _build_sequencing_failure_sequence()
        q = fit_sequencing_tabular_q(recs)
        self.assertEqual(len(q.table), 1)
        # exactly one (state, action) cell
        only_state = next(iter(q.table))
        only_action = next(iter(q.table[only_state]))
        self.assertAlmostEqual(q.table[only_state][only_action], 1.0)
        self.assertEqual(q.support[only_state][only_action], 1)

    def test_q_table_empty_on_success_only(self) -> None:
        recs = [
            _ok_record(attempt_id="a-0", outcome="succeeded"),
            _ok_record(attempt_id="a-1", outcome="succeeded"),
        ]
        q = fit_sequencing_tabular_q(recs)
        self.assertEqual(q.table, {})

    def test_q_value_zero_on_unseen_cell(self) -> None:
        q = fit_sequencing_tabular_q([])
        self.assertEqual(q.value("nonexistent", "anything"), 0.0)
        self.assertFalse(q.has_support("nonexistent", "anything"))


class SequencingWISandDMTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = _build_sequencing_failure_sequence()
        self.fitted_q = fit_sequencing_tabular_q(self.records)
        # Identify the single (state, action) cell
        self.state_key_str = next(iter(self.fitted_q.table))
        self.logged_action = next(iter(self.fitted_q.table[self.state_key_str]))

    def test_dm_matches_q_when_target_matches_logged(self) -> None:
        dm = direct_method_sequencing(
            fitted_q=self.fitted_q,
            eval_records=self.records,
            target_action_for_state=lambda _s: self.logged_action,
        )
        self.assertAlmostEqual(dm.value, 1.0)
        self.assertEqual(dm.num_states_evaluated, 1)
        self.assertEqual(dm.num_states_covered, 1)

    def test_dm_zero_when_target_unseen(self) -> None:
        # Pick an action that is guaranteed not the logged action.
        other_action = next(
            a for a in SEQUENCING_ACTIONS if a != self.logged_action
        )
        dm = direct_method_sequencing(
            fitted_q=self.fitted_q,
            eval_records=self.records,
            target_action_for_state=lambda _s: other_action,
        )
        self.assertAlmostEqual(dm.value, 0.0)
        self.assertEqual(dm.num_states_covered, 0)

    def test_wis_high_when_target_matches_logged(self) -> None:
        wis = wis_sequencing(
            records=self.records,
            target_action_for_state=lambda _s: self.logged_action,
        )
        # importance weight = 1.0; WIS = reward = +1.0
        self.assertAlmostEqual(wis.value, 1.0)

    def test_wis_drops_when_target_diverges(self) -> None:
        # Build TWO independent pair-episodes. The target matches the
        # logged action only on ep_a, so importance weights are
        # severely imbalanced ⇒ ESS drops well below n=2. This is the
        # WIS pathology we want surfaced in the report.
        ep_a = _build_sequencing_failure_sequence()
        ep_b = [
            _ok_record(attempt_id="ep2-0", outcome="execution_failed", cycle_time_s=0.0),
            _ok_record(attempt_id="ep2-1", outcome="execution_failed", cycle_time_s=0.0),
        ]
        recs = ep_a + ep_b
        wis = wis_sequencing(
            records=recs,
            target_action_for_state=lambda _s: self.logged_action,
        )
        # n_pairs = 2 but only 1 has its target matched ⇒ ESS ≈ 1.
        self.assertEqual(wis.num_records, 2)
        self.assertLess(wis.effective_sample_size, 2.0)
        # weight concentration index should be > 0.5 (one weight dominates).
        self.assertGreater(wis.weight_concentration_index, 0.5)


# ---------------------------------------------------------------------------
# Confidence intervals.
# ---------------------------------------------------------------------------


class ConfidenceIntervalTests(unittest.TestCase):
    def test_bootstrap_ci_contains_mean_on_iid(self) -> None:
        # Constant samples → CI is degenerate at the mean.
        ci = ope_mod._bootstrap_mean_ci([0.5] * 10, n_resamples=200, rng_seed=1)
        self.assertAlmostEqual(ci.lower, 0.5)
        self.assertAlmostEqual(ci.upper, 0.5)

    def test_bootstrap_ci_deterministic(self) -> None:
        ci1 = ope_mod._bootstrap_mean_ci(
            [0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=200, rng_seed=42
        )
        ci2 = ope_mod._bootstrap_mean_ci(
            [0.1, 0.2, 0.3, 0.4, 0.5], n_resamples=200, rng_seed=42
        )
        self.assertEqual(ci1, ci2)

    def test_normal_approx_ci_zero_width_on_singleton(self) -> None:
        ci = ope_mod._normal_approx_mean_ci([0.7])
        self.assertAlmostEqual(ci.lower, 0.7)
        self.assertAlmostEqual(ci.upper, 0.7)

    def test_normal_approx_ci_brackets_mean(self) -> None:
        samples = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        ci = ope_mod._normal_approx_mean_ci(samples)
        mean = sum(samples) / len(samples)
        self.assertLess(ci.lower, mean)
        self.assertGreater(ci.upper, mean)


# ---------------------------------------------------------------------------
# Report builder.
# ---------------------------------------------------------------------------


class ReportBuilderTests(unittest.TestCase):
    def test_top_level_shape(self) -> None:
        recs = _build_sequencing_failure_sequence()

        def target(_s):
            return SEQUENCING_ACTIONS[0]

        report = build_ope_report(
            records=recs,
            sequencing_target_action_for_state=target,
            dataset_id="unit",
            dataset_paths=["synth.jsonl"],
            rng_seed=1,
        )
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["report_kind"], "ope")
        self.assertEqual(report["dataset_id"], "unit")
        self.assertEqual(report["num_records"], 2)
        self.assertEqual(
            report["behaviour_policy_assumption"]["kind"],
            "epsilon_smoothed_deterministic",
        )
        self.assertEqual(
            report["behaviour_policy_assumption"]["epsilon"], 0.05
        )
        self.assertEqual(
            sorted(report["sections"].keys()),
            ["candidate", "ranking", "sequencing"],
        )

    def test_candidate_ranking_have_coverage_warning(self) -> None:
        recs = _build_sequencing_failure_sequence()
        report = build_ope_report(
            records=recs,
            sequencing_target_action_for_state=lambda _s: SEQUENCING_ACTIONS[0],
            dataset_id="unit",
            dataset_paths=["synth.jsonl"],
        )
        candidate_wis = report["sections"]["candidate"]["estimators"]["wis"]
        ranking_wis = report["sections"]["ranking"]["estimators"]["wis"]
        self.assertIn("coverage_warning", candidate_wis)
        self.assertIn("coverage_warning", ranking_wis)
        self.assertIn("behaviour_action_unknown_in_logs", candidate_wis["coverage_warning"])

    def test_sequencing_section_has_all_three_estimators(self) -> None:
        recs = _build_sequencing_failure_sequence()
        section = build_sequencing_section(
            records=recs,
            target_action_for_state=lambda _s: SEQUENCING_ACTIONS[0],
        )
        ests = section["estimators"]
        self.assertEqual(ests["wis"]["estimator"], "WIS")
        self.assertEqual(ests["direct_method"]["estimator"], "DirectMethod")
        self.assertEqual(ests["fqe"]["algorithm"], "tabular_bandit_gamma_0")

    def test_empty_records_raises(self) -> None:
        with self.assertRaises(MalformedOPEInputError):
            build_ope_report(
                records=[],
                sequencing_target_action_for_state=lambda _s: SEQUENCING_ACTIONS[0],
                dataset_id="empty",
                dataset_paths=[],
            )

    def test_dataset_hash_deterministic(self) -> None:
        recs = _build_sequencing_failure_sequence()
        r1 = build_ope_report(
            records=recs,
            sequencing_target_action_for_state=lambda _s: SEQUENCING_ACTIONS[0],
            dataset_id="unit",
            dataset_paths=["synth.jsonl"],
            rng_seed=1,
        )
        r2 = build_ope_report(
            records=recs,
            sequencing_target_action_for_state=lambda _s: SEQUENCING_ACTIONS[0],
            dataset_id="unit",
            dataset_paths=["synth.jsonl"],
            rng_seed=1,
        )
        self.assertEqual(r1["dataset_hash"], r2["dataset_hash"])
        self.assertEqual(r1, r2)


# ---------------------------------------------------------------------------
# Malformed-input strictness.
# ---------------------------------------------------------------------------


class MalformedInputTests(unittest.TestCase):
    def test_missing_extra_raises(self) -> None:
        bad = {
            "attempt_id": "x",
            "final_outcome": "succeeded",
        }
        with self.assertRaises(MalformedOPEInputError):
            compute_record_reward(bad)

    def test_missing_cycle_time_raises(self) -> None:
        bad = {
            "attempt_id": "x",
            "final_outcome": "succeeded",
            "extra": {},
        }
        with self.assertRaises(MalformedOPEInputError):
            compute_record_reward(bad)

    def test_record_not_mapping_raises(self) -> None:
        with self.assertRaises(MalformedOPEInputError):
            compute_record_reward("not a record")  # type: ignore[arg-type]

    def test_load_missing_file_raises(self) -> None:
        with self.assertRaises(MalformedOPEInputError):
            ope_mod.load_records_for_ope([Path("/tmp/does_not_exist.jsonl")])


# ---------------------------------------------------------------------------
# CLI smoke test.
# ---------------------------------------------------------------------------


class CLISmokeTests(unittest.TestCase):
    def test_ope_subcommand_writes_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ope_smoke.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.rl",
                    "ope",
                    "--dataset-id",
                    "unit_smoke",
                    "--output",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                msg=f"stdout={result.stdout}\nstderr={result.stderr}",
            )
            self.assertTrue(out.exists())
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["schema_version"], 2)
            self.assertEqual(report["report_kind"], "ope")
            self.assertEqual(report["dataset_id"], "unit_smoke")
            self.assertIn("sequencing", report["sections"])


COMMITTED_OPE_REPORT_PATH = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies" / "v5_ope_report_v1.json"
)
COMMITTED_OPE_REPORT_SHA256 = (
    # Re-generated intentionally: the candidate/ranking WIS now carries an explicit ``estimable: false`` abstain
    # marker (the behaviour action was never logged, so WIS collapses to the mean — not an off-policy
    # estimate). Sequencing stays ``estimable: true``. Estimator values are unchanged; only the marker is added.
    # Re-generated to add reward_model / reward_interpretation / dataset_provenance honesty stamps
    # (schema_version 1 -> 2). Estimator values unchanged (surgical additive stamp on the committed report).
    "ce75d79040c549aeea2a8b15e39a9020014e185a82e20223b742f570d189ec4a"
)


class CommittedArtifactTests(unittest.TestCase):
    """Byte-identity test on the committed OPE report.

    Mirrors candidate/ranking/sequencing artifact discipline: regenerating the report must
    produce a byte-identical file as long as inputs and code are
    unchanged. Update the expected sha256 deliberately when intent
    changes.
    """

    def test_committed_report_exists(self) -> None:
        self.assertTrue(
            COMMITTED_OPE_REPORT_PATH.exists(),
            msg=f"missing committed artifact: {COMMITTED_OPE_REPORT_PATH}",
        )

    def test_committed_report_byte_identity(self) -> None:
        import hashlib

        data = COMMITTED_OPE_REPORT_PATH.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(
            digest,
            COMMITTED_OPE_REPORT_SHA256,
            msg=(
                "Committed V5 OPE report drifted. Re-run "
                "`python -m backend.src.robot.grasping.rl ope "
                "--dataset-id v5_ope_canonical` and update the expected "
                "sha256 if the drift is intentional."
            ),
        )

    def test_candidate_ranking_abstain_estimable_marker(self) -> None:
        # candidate/ranking WIS is NOT a real off-policy estimate (no behaviour action logged) -> estimable=false
        # (abstain), while sequencing (real importance weights) stays estimable=true.
        report = json.loads(COMMITTED_OPE_REPORT_PATH.read_text(encoding="utf-8"))
        sections = report["sections"]
        self.assertFalse(sections["candidate"]["estimators"]["wis"]["estimable"])
        self.assertFalse(sections["ranking"]["estimators"]["wis"]["estimable"])
        self.assertTrue(sections["sequencing"]["estimators"]["wis"]["estimable"])

    def test_committed_report_schema(self) -> None:
        report = json.loads(COMMITTED_OPE_REPORT_PATH.read_text(encoding="utf-8"))
        self.assertEqual(report["schema_version"], 2)
        self.assertEqual(report["report_kind"], "ope")
        self.assertEqual(
            report["behaviour_policy_assumption"]["epsilon"], 0.05
        )
        self.assertEqual(
            sorted(report["sections"].keys()),
            ["candidate", "ranking", "sequencing"],
        )


if __name__ == "__main__":
    unittest.main()
