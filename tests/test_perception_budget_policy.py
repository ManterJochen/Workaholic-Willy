"""Perception-budget policy + LinUCB + router + CLI tests.

Coverage:
* Bucket functions (mode prefix, views, candidate, occlusion).
* ``PerceptionBudgetStateKey`` construction.
* One-hot layout determinism and exactly-4-ones encoding.
* ``LinUCBPerceptionBudgetPolicy`` construction, fallback, lex tie.
* Trainer extract_state / extract_action_label / scope filter.
* Artifact write/load round-trip and schema-drift rejection.
* ``ShadowRouter.run_perception_shadow`` carrier + emit helper.
* ``SHADOW_ROUTER_TELEMETRY_VERSION == 5`` (last bumped in the recovery layer).
* CLI smoke (``train-perception-budget-policy``).
* Committed artifact byte-identity sha256 (regenerated deterministically).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.perception_budget_policy import (
    CANDIDATE_COUNT_BUCKETS,
    DEFAULT_FALLBACK_TABLE,
    DEFAULT_LINUCB_ALPHA,
    DEFAULT_LINUCB_LAMBDA,
    DEFAULT_MIN_SUPPORT_THRESHOLD,
    MODE_BUCKETS,
    MODE_BUCKET_DENSE,
    OCCLUSION_BUCKETS,
    ONEHOT_FAMILIES,
    PERCEPTION_ACTION_CONTINUE,
    PERCEPTION_ACTION_STOP,
    PERCEPTION_ACTIONS,
    PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION,
    LinUCBPerceptionBudgetPolicy,
    PerceptionBudgetRequest,
    PerceptionBudgetStateKey,
    VIEWS_SEEN_BUCKETS,
    bucket_candidate_count,
    bucket_mode,
    bucket_occlusion,
    bucket_views_seen,
    load_linucb_perception_budget_policy,
    onehot_dim,
    onehot_index_layout,
    state_key_to_onehot,
)
from src.robot.grasping.rl.train_perception_budget import (
    build_perception_budget_artifact_json,
    extract_action_label,
    extract_perception_state,
    train_linucb_perception_budget_policy,
    write_perception_budget_artifact,
)
from src.robot.grasping.rl.router import (
    ROUTER_PATH_FALLBACK_BASELINE,
    ROUTER_PATH_RL_SHADOW,
    SHADOW_ROUTER_TELEMETRY_VERSION,
    PerceptionShadowTelemetry,
    ShadowRouter,
    emit_perception_extras,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACT_REL = "docs/baselines/rl_policies/v5_perception_budget_baseline_v1.json"
ARTIFACT_PATH = REPO_ROOT / ARTIFACT_REL
# C: the committed v5 baseline is DEGENERATE (600 continue / 0 stop -> one cell, one action). It now
# carries the honest ``degeneracy_note`` (build_perception_honesty flags a single-action perception fit),
# so a reader cannot mistake the fallback-table wrapper for a trained STOP/CONTINUE policy. The origin
# dataset_hash is preserved; only the note was added.
EXPECTED_ARTIFACT_SHA256 = (
    "7543e999dc3b83194940c4a1fd5f28be96cac5b3cfb0c49fd68b61a331d46181"
)


class BucketTests(unittest.TestCase):
    def test_bucket_mode_prefix(self) -> None:
        self.assertEqual(bucket_mode("dense_clutter"), "dense")
        self.assertEqual(bucket_mode("dense"), "dense")
        self.assertEqual(bucket_mode("easy_canonical"), "easy")
        self.assertEqual(bucket_mode("auto"), "auto")
        self.assertEqual(bucket_mode("unknown"), "unknown")
        self.assertEqual(bucket_mode(""), "unknown")
        self.assertEqual(bucket_mode(None), "unknown")  # type: ignore[arg-type]

    def test_bucket_views_seen(self) -> None:
        self.assertEqual(bucket_views_seen(0), "0")
        self.assertEqual(bucket_views_seen(1), "1")
        self.assertEqual(bucket_views_seen(2), "2")
        self.assertEqual(bucket_views_seen(3), "3+")
        self.assertEqual(bucket_views_seen(99), "3+")
        # Defensive: negatives clamp to "0"; non-int safe.
        self.assertEqual(bucket_views_seen(-5), "0")
        self.assertEqual(bucket_views_seen(None), "0")  # type: ignore[arg-type]
        # bool is guarded explicitly → always "0" regardless of value.
        self.assertEqual(bucket_views_seen(True), "0")
        self.assertEqual(bucket_views_seen(False), "0")

    def test_bucket_candidate_count(self) -> None:
        self.assertEqual(bucket_candidate_count(0), "0")
        self.assertEqual(bucket_candidate_count(1), "1-2")
        self.assertEqual(bucket_candidate_count(2), "1-2")
        self.assertEqual(bucket_candidate_count(3), "3-5")
        self.assertEqual(bucket_candidate_count(5), "3-5")
        self.assertEqual(bucket_candidate_count(6), "6+")
        self.assertEqual(bucket_candidate_count(99), "6+")
        self.assertEqual(bucket_candidate_count(None), "0")  # type: ignore[arg-type]
        self.assertEqual(bucket_candidate_count(-1), "0")

    def test_bucket_occlusion(self) -> None:
        self.assertEqual(bucket_occlusion(None), "unknown")
        self.assertEqual(bucket_occlusion(float("nan")), "unknown")
        self.assertEqual(bucket_occlusion(0.0), "low")
        self.assertEqual(bucket_occlusion(0.29), "low")
        self.assertEqual(bucket_occlusion(0.3), "mid")
        self.assertEqual(bucket_occlusion(0.5), "mid")
        self.assertEqual(bucket_occlusion(0.7), "high")
        self.assertEqual(bucket_occlusion(0.99), "high")
        # Out-of-range still mapped to nearest bucket (or unknown).
        self.assertIn(bucket_occlusion(-1.0), {"low", "unknown"})
        self.assertIn(bucket_occlusion(2.0), {"high", "unknown"})


class StateKeyTests(unittest.TestCase):
    def test_construction_and_string_key(self) -> None:
        k = PerceptionBudgetStateKey(
            views_seen_bucket="2",
            last_candidate_count_bucket="3-5",
            last_occlusion_bucket="mid",
            mode_bucket="dense",
        )
        self.assertEqual(k.as_string_key(), "2|3-5|mid|dense")

    def test_invalid_bucket_rejected(self) -> None:
        with self.assertRaises(Exception):
            PerceptionBudgetStateKey(
                views_seen_bucket="bogus",
                last_candidate_count_bucket="3-5",
                last_occlusion_bucket="mid",
                mode_bucket="dense",
            )


class OneHotTests(unittest.TestCase):
    def test_layout_deterministic_and_dim(self) -> None:
        layout = onehot_index_layout()
        expected_dim = (
            len(VIEWS_SEEN_BUCKETS)
            + len(CANDIDATE_COUNT_BUCKETS)
            + len(OCCLUSION_BUCKETS)
            + len(MODE_BUCKETS)
        )
        self.assertEqual(len(layout), expected_dim)
        self.assertEqual(onehot_dim(), expected_dim)
        # Layout must be deterministic across calls.
        self.assertEqual(onehot_index_layout(), layout)
        # And must cover exactly the 4 families in canonical order.
        self.assertEqual(len(ONEHOT_FAMILIES), 4)

    def test_state_key_to_onehot_exactly_four_ones(self) -> None:
        k = PerceptionBudgetStateKey("0", "0", "low", "easy")
        v = state_key_to_onehot(k)
        self.assertEqual(len(v), onehot_dim())
        self.assertEqual(sum(v), 4)
        for x in v:
            self.assertIn(x, (0, 1))


def _make_uniform_policy(
    *,
    support: int = 10,
    a_diag_value: float = 1.0,
) -> LinUCBPerceptionBudgetPolicy:
    dim = onehot_dim()
    A = {a: tuple(a_diag_value for _ in range(dim)) for a in PERCEPTION_ACTIONS}
    b = {a: tuple(0.0 for _ in range(dim)) for a in PERCEPTION_ACTIONS}
    sup = {a: tuple(support for _ in range(dim)) for a in PERCEPTION_ACTIONS}
    return LinUCBPerceptionBudgetPolicy(
        A_diag_per_action=A,
        b_per_action=b,
        feature_support_per_action=sup,
        fallback_table=DEFAULT_FALLBACK_TABLE,
        alpha=DEFAULT_LINUCB_ALPHA,
        ridge_lambda=DEFAULT_LINUCB_LAMBDA,
        min_support_threshold=DEFAULT_MIN_SUPPORT_THRESHOLD,
        training_scope="dense",
        artifact_dataset_id="",
        artifact_dataset_hash="",
    )


class LinUCBPolicyTests(unittest.TestCase):
    def test_propose_lex_tie_break_continue(self) -> None:
        policy = _make_uniform_policy()
        k = PerceptionBudgetStateKey("0", "0", "low", "easy")
        sel = policy.propose_perception_budget(
            PerceptionBudgetRequest(attempt_id="a", state_key=k)
        )
        # Symmetric arms with zero b → exact tie → "continue" wins.
        self.assertEqual(sel.action, PERCEPTION_ACTION_CONTINUE)
        self.assertFalse(sel.used_fallback)

    def test_fallback_when_support_below_threshold(self) -> None:
        policy = _make_uniform_policy(support=0)
        k = PerceptionBudgetStateKey("3+", "6+", "high", "dense")
        sel = policy.propose_perception_budget(
            PerceptionBudgetRequest(attempt_id="a", state_key=k)
        )
        self.assertTrue(sel.used_fallback)
        # views_seen=3+ → fallback table says STOP.
        self.assertEqual(sel.action, PERCEPTION_ACTION_STOP)


class TrainerExtractionTests(unittest.TestCase):
    def test_extract_action_label_fusion_present(self) -> None:
        rec = {"extra": {"fusion_latency_ms": 12.0, "cycle_time_s": 1.0}}
        self.assertEqual(extract_action_label(rec), PERCEPTION_ACTION_CONTINUE)

    def test_extract_action_label_fusion_absent(self) -> None:
        rec = {"extra": {"cycle_time_s": 1.0}}
        self.assertEqual(extract_action_label(rec), PERCEPTION_ACTION_STOP)

    def test_extract_perception_state_dense(self) -> None:
        rec = {
            "mode": "dense_clutter",
            "extra": {
                "candidate_count": 4,
                "expected_occlusion_ratio": 0.5,
                "fusion_latency_ms": 9.0,
                "cycle_time_s": 1.2,
            },
        }
        key = extract_perception_state(rec)
        self.assertEqual(key.mode_bucket, "dense")
        self.assertEqual(key.last_candidate_count_bucket, "3-5")
        self.assertEqual(key.last_occlusion_bucket, "mid")


class TrainerScopeTests(unittest.TestCase):
    def test_dense_scope_filters_easy(self) -> None:
        records = [
            {
                "mode": "easy_canonical",
                "extra": {"cycle_time_s": 1.0},
                "final_outcome": "succeeded",
            },
            {
                "mode": "dense_clutter",
                "extra": {
                    "cycle_time_s": 1.0,
                    "fusion_latency_ms": 8.0,
                },
                "final_outcome": "succeeded",
            },
        ]
        _policy, result = train_linucb_perception_budget_policy(
            records,
            training_scope="dense",
            min_support_threshold=1,
        )
        self.assertEqual(result.num_records, 2)
        self.assertEqual(result.num_records_kept, 1)
        self.assertEqual(result.num_records_dropped_dense_scope, 1)


class ArtifactRoundTripTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        policy = _make_uniform_policy()
        # Build a minimal artifact JSON
        artifact = build_perception_budget_artifact_json(
            policy=policy,
            train_result=type(
                "R",
                (),
                {
                    "num_records": 0,
                    "num_records_kept": 0,
                    "num_records_dropped_dense_scope": 0,
                    "num_records_dropped_missing_cycle_time": 0,
                    "num_action_stop": 0,
                    "num_action_continue": 0,
                    "num_cells_observed": 0,
                    "mean_reward_stop": 0.0,
                    "mean_reward_continue": 0.0,
                },
            )(),
            seed=1,
            dataset_id="t",
            dataset_hash="h",
            offline_kpi_estimates={},
        )
        self.assertEqual(
            artifact["schema_version"],
            PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION,
        )
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "perception_budget.json"
            write_perception_budget_artifact(artifact, p)
            loaded = load_linucb_perception_budget_policy(p)
        self.assertEqual(loaded.training_scope, policy.training_scope)
        self.assertEqual(loaded.alpha, policy.alpha)
        self.assertEqual(loaded.ridge_lambda, policy.ridge_lambda)

    def test_schema_drift_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({"schema_version": 999}))
            with self.assertRaises(Exception):
                load_linucb_perception_budget_policy(p)


class RouterIntegrationTests(unittest.TestCase):
    def test_version_is_four(self) -> None:
        self.assertEqual(SHADOW_ROUTER_TELEMETRY_VERSION, 5)

    def test_run_perception_shadow_none_when_unwired(self) -> None:
        from src.robot.grasping.rl.candidate_policy import (
            LogisticCandidatePolicy,
        )

        # We don't need to call run_candidate; just need a router instance.
        dummy_policy = LogisticCandidatePolicy.__new__(LogisticCandidatePolicy)
        router = ShadowRouter(candidate_policy=dummy_policy)
        k = PerceptionBudgetStateKey("0", "0", "low", "easy")
        out = router.run_perception_shadow(
            attempt_id="a",
            state_key=k,
            baseline_action=PERCEPTION_ACTION_CONTINUE,
        )
        self.assertIsNone(out)

    def test_run_perception_shadow_emits_carrier(self) -> None:
        from src.robot.grasping.rl.candidate_policy import (
            LogisticCandidatePolicy,
        )

        dummy_policy = LogisticCandidatePolicy.__new__(LogisticCandidatePolicy)
        policy = _make_uniform_policy()
        router = ShadowRouter(
            candidate_policy=dummy_policy, perception_policy=policy
        )
        k = PerceptionBudgetStateKey("0", "0", "low", "easy")
        tele = router.run_perception_shadow(
            attempt_id="a",
            state_key=k,
            baseline_action=PERCEPTION_ACTION_CONTINUE,
        )
        self.assertIsInstance(tele, PerceptionShadowTelemetry)
        assert tele is not None
        self.assertEqual(tele.router_path, ROUTER_PATH_RL_SHADOW)
        self.assertFalse(tele.fallback_triggered)
        extras = emit_perception_extras(telemetry=tele)
        self.assertEqual(
            extras["rl_perception_action_proposed"],
            PERCEPTION_ACTION_CONTINUE,
        )
        self.assertIn("rl_perception_state_key", extras)

    def test_run_perception_shadow_typed_fallback_on_exception(self) -> None:
        from src.robot.grasping.rl.candidate_policy import (
            LogisticCandidatePolicy,
        )

        class BoomPolicy:
            name = "boom"
            version = 7

            def propose_perception_budget(self, req):  # noqa: ANN001
                raise RuntimeError("boom")

        dummy_policy = LogisticCandidatePolicy.__new__(LogisticCandidatePolicy)
        router = ShadowRouter(
            candidate_policy=dummy_policy, perception_policy=BoomPolicy()
        )
        k = PerceptionBudgetStateKey("0", "0", "low", "easy")
        tele = router.run_perception_shadow(
            attempt_id="a",
            state_key=k,
            baseline_action=PERCEPTION_ACTION_STOP,
        )
        assert tele is not None
        self.assertTrue(tele.fallback_triggered)
        self.assertEqual(tele.router_path, ROUTER_PATH_FALLBACK_BASELINE)
        self.assertEqual(tele.selection.action, PERCEPTION_ACTION_STOP)


class CLISmokeTests(unittest.TestCase):
    def test_cli_train_from_packs(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "perception_budget.json"
            cmd = [
                sys.executable,
                "-m",
                "src.robot.grasping.rl",
                "train-perception-budget-policy",
                "--replay-pack",
                "tests/data/replay/replay_dense_canonical_v1.jsonl",
                "--output",
                str(out),
            ]
            result = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out.exists())
            blob = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(
                blob["schema_version"],
                PERCEPTION_BUDGET_ARTIFACT_SCHEMA_VERSION,
            )


class CommittedArtifactSha256Tests(unittest.TestCase):
    def test_committed_artifact_matches(self) -> None:
        self.assertTrue(
            ARTIFACT_PATH.exists(),
            f"missing committed artifact at {ARTIFACT_PATH}",
        )
        data = ARTIFACT_PATH.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        self.assertEqual(digest, EXPECTED_ARTIFACT_SHA256)

    def test_committed_artifact_loads(self) -> None:
        policy = load_linucb_perception_budget_policy(ARTIFACT_PATH)
        self.assertEqual(policy.training_scope, MODE_BUCKET_DENSE)
        self.assertEqual(policy.alpha, DEFAULT_LINUCB_ALPHA)

    def test_committed_artifact_carries_honest_degeneracy_note(self) -> None:
        # The committed baseline is 600 continue / 0 stop -> a single-action fit. It MUST surface the
        # honest degeneracy_note so no reader mistakes the fallback-table wrapper for a trained policy.
        blob = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
        self.assertIn("degeneracy_note", blob)
        self.assertIn("single action dominates", blob["degeneracy_note"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
