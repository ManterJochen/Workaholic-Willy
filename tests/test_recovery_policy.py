"""Tests for the recovery optimization policy + router seam."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from typing import Any

from src.robot.grasping.rl.recovery_policy import (
    DEFAULT_FALLBACK_TABLE,
    DEFAULT_MAX_RECOVERY_ATTEMPTS,
    DEFAULT_MIN_SUPPORT_THRESHOLD,
    DENSE_BUCKET_DENSE,
    DENSE_BUCKET_NON_DENSE,
    FAILURE_CLASS_BUCKETS,
    LAST_OUTCOME_FAILED,
    LAST_OUTCOME_SUCCEEDED,
    LAST_OUTCOME_UNKNOWN,
    ONEHOT_FAMILIES,
    RECOVERY_ACTIONS,
    RECOVERY_ACTION_ABORT_RECOVERY,
    RECOVERY_ACTION_CONTAINER_AGITATE,
    RECOVERY_ACTION_NUDGE_TARGET,
    RECOVERY_ACTION_REOBSERVE,
    RECOVERY_ACTION_REPLAN_GRASP,
    RECOVERY_ACTION_RE_SEGMENT,
    RECOVERY_ARTIFACT_SCHEMA_VERSION,
    LinUCBRecoveryPolicy,
    RecoveryActionScore,
    RecoveryRequest,
    RecoverySelection,
    RecoveryStateKey,
    apply_recovery_anti_loop_gate,
    bucket_attempt_index,
    bucket_dense,
    bucket_last_outcome,
    bucket_reobserve_count,
    load_linucb_recovery_policy,
    onehot_dim,
    onehot_index_layout,
    state_key_to_onehot,
)
from src.robot.grasping.rl.sequencing_policy import (
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_UNKNOWN,
)
from src.robot.grasping.rl.train_recovery import (
    ACTION_LABEL_SOURCE,
    RECOVERY_TOKEN_MAP,
    TRAINING_SCOPE_ALL,
    TRAINING_SCOPE_DENSE,
    build_recovery_artifact_json,
    extract_recovery_tuples,
    train_linucb_recovery_policy,
    train_recovery_from_packs,
    write_recovery_artifact,
)
from src.robot.grasping.rl.router import (
    ROUTER_PATH_FALLBACK_BASELINE,
    ROUTER_PATH_RL_SHADOW,
    SHADOW_ROUTER_TELEMETRY_VERSION,
    RecoveryShadowTelemetry,
    ShadowRouter,
    emit_recovery_extras,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies"
    / "v6_recovery_baseline_v1.json"
)
DENSE_PACK = REPO_ROOT / "tests" / "data" / "replay" / "replay_dense_canonical_v1.jsonl"
EASY_PACK = REPO_ROOT / "tests" / "data" / "replay" / "replay_easy_canonical_v1.jsonl"

COMMITTED_ARTIFACT_SHA256 = (
    # Re-blessed 2026-09-10 with the tree's own regenerator, after the replay packs stopped
    # being checked out with CRLF (.gitattributes now pins tests/data/replay/** -text). Every
    # provenance hash over those packs moved from a CRLF reading to the LF bytes the committed
    # blob has always held.
    "8ae243fb1d438175b201c5972532caf939862c596a87f5b8316c7dd6d3495701"
)


# ---------------------------------------------------------------------------
# Buckets.
# ---------------------------------------------------------------------------


class BucketTests(unittest.TestCase):
    def test_bucket_attempt_index_total(self) -> None:
        self.assertEqual(bucket_attempt_index(0), "0")
        self.assertEqual(bucket_attempt_index(1), "1")
        self.assertEqual(bucket_attempt_index(2), "2")
        self.assertEqual(bucket_attempt_index(3), "3+")
        self.assertEqual(bucket_attempt_index(99), "3+")
        # Negatives + non-ints + bools clip to "0" (never raise).
        self.assertEqual(bucket_attempt_index(-1), "0")
        self.assertEqual(bucket_attempt_index(None), "0")
        self.assertEqual(bucket_attempt_index("2"), "0")
        self.assertEqual(bucket_attempt_index(True), "0")

    def test_bucket_reobserve_count_total(self) -> None:
        self.assertEqual(bucket_reobserve_count(0), "0")
        self.assertEqual(bucket_reobserve_count(1), "1")
        self.assertEqual(bucket_reobserve_count(2), "2")
        self.assertEqual(bucket_reobserve_count(3), "3+")
        self.assertEqual(bucket_reobserve_count(-5), "0")
        self.assertEqual(bucket_reobserve_count(None), "0")
        self.assertEqual(bucket_reobserve_count(False), "0")

    def test_bucket_dense_prefix(self) -> None:
        self.assertEqual(bucket_dense("dense"), DENSE_BUCKET_DENSE)
        self.assertEqual(bucket_dense("dense_clutter"), DENSE_BUCKET_DENSE)
        self.assertEqual(bucket_dense("easy"), DENSE_BUCKET_NON_DENSE)
        self.assertEqual(bucket_dense(None), DENSE_BUCKET_NON_DENSE)
        self.assertEqual(bucket_dense(123), DENSE_BUCKET_NON_DENSE)

    def test_bucket_last_outcome(self) -> None:
        self.assertEqual(bucket_last_outcome("succeeded"), LAST_OUTCOME_SUCCEEDED)
        self.assertEqual(bucket_last_outcome("success"), LAST_OUTCOME_SUCCEEDED)
        self.assertEqual(bucket_last_outcome("failed"), LAST_OUTCOME_FAILED)
        self.assertEqual(bucket_last_outcome("failure"), LAST_OUTCOME_FAILED)
        self.assertEqual(bucket_last_outcome(None), LAST_OUTCOME_UNKNOWN)
        self.assertEqual(bucket_last_outcome("???"), LAST_OUTCOME_UNKNOWN)


# ---------------------------------------------------------------------------
# State key + one-hot.
# ---------------------------------------------------------------------------


class StateKeyTests(unittest.TestCase):
    def _good(self) -> RecoveryStateKey:
        return RecoveryStateKey(
            failure_class_bucket=FAILURE_CLASS_UNKNOWN,
            attempt_index_bucket="0",
            reobserve_count_bucket="0",
            dense_bucket=DENSE_BUCKET_DENSE,
            last_outcome_bucket=LAST_OUTCOME_UNKNOWN,
        )

    def test_string_key(self) -> None:
        k = self._good()
        self.assertEqual(k.as_string_key(), "unknown|0|0|dense|unknown")

    def test_rejects_unknown_buckets(self) -> None:
        with self.assertRaises(ValueError):
            RecoveryStateKey(
                failure_class_bucket="bogus",
                attempt_index_bucket="0",
                reobserve_count_bucket="0",
                dense_bucket=DENSE_BUCKET_DENSE,
                last_outcome_bucket=LAST_OUTCOME_UNKNOWN,
            )
        with self.assertRaises(ValueError):
            RecoveryStateKey(
                failure_class_bucket=FAILURE_CLASS_UNKNOWN,
                attempt_index_bucket="9",
                reobserve_count_bucket="0",
                dense_bucket=DENSE_BUCKET_DENSE,
                last_outcome_bucket=LAST_OUTCOME_UNKNOWN,
            )


class OneHotTests(unittest.TestCase):
    def test_dim_matches_layout(self) -> None:
        self.assertEqual(onehot_dim(), len(onehot_index_layout()))
        # 8 failure_classes + 4 attempt + 4 reobserve + 2 dense + 4 outcome.
        self.assertEqual(onehot_dim(), 8 + 4 + 4 + 2 + 4)

    def test_layout_grouped_by_family(self) -> None:
        layout = onehot_index_layout()
        for name, _ in ONEHOT_FAMILIES:
            self.assertTrue(
                any(label.startswith(f"{name}.") for label in layout),
                msg=f"family {name} missing from layout",
            )

    def test_state_key_to_onehot_exactly_5_active(self) -> None:
        k = RecoveryStateKey(
            failure_class_bucket=FAILURE_CLASS_SLIP,
            attempt_index_bucket="1",
            reobserve_count_bucket="2",
            dense_bucket=DENSE_BUCKET_DENSE,
            last_outcome_bucket=LAST_OUTCOME_FAILED,
        )
        x = state_key_to_onehot(k)
        self.assertEqual(sum(x), 5)
        self.assertEqual(len(x), onehot_dim())


# ---------------------------------------------------------------------------
# LinUCB policy + fallback gate.
# ---------------------------------------------------------------------------


def _zero_policy(min_support: int = DEFAULT_MIN_SUPPORT_THRESHOLD) -> LinUCBRecoveryPolicy:
    dim = onehot_dim()
    return LinUCBRecoveryPolicy(
        A_diag_per_action={a: tuple([1.0] * dim) for a in RECOVERY_ACTIONS},
        b_per_action={a: tuple([0.0] * dim) for a in RECOVERY_ACTIONS},
        feature_support_per_action={
            a: tuple([0] * dim) for a in RECOVERY_ACTIONS
        },
        min_support_threshold=min_support,
    )


def _key(failure_class: str = FAILURE_CLASS_UNKNOWN) -> RecoveryStateKey:
    return RecoveryStateKey(
        failure_class_bucket=failure_class,
        attempt_index_bucket="0",
        reobserve_count_bucket="0",
        dense_bucket=DENSE_BUCKET_DENSE,
        last_outcome_bucket=LAST_OUTCOME_UNKNOWN,
    )


class LinUCBPolicyTests(unittest.TestCase):
    def test_zero_support_triggers_fallback(self) -> None:
        policy = _zero_policy()
        sel = policy.propose_recovery(
            RecoveryRequest(attempt_id="a", state_key=_key())
        )
        self.assertTrue(sel.used_fallback)
        # Fallback for "unknown" is reobserve per DEFAULT_FALLBACK_TABLE.
        self.assertEqual(sel.action, RECOVERY_ACTION_REOBSERVE)
        # Ranked actions must remain a permutation of the canon set.
        self.assertEqual(set(sel.ranked_actions), set(RECOVERY_ACTIONS))
        self.assertEqual(sel.ranked_actions[0], RECOVERY_ACTION_REOBSERVE)

    def test_fallback_table_covers_each_failure_class(self) -> None:
        policy = _zero_policy()
        for fc in FAILURE_CLASS_BUCKETS:
            sel = policy.propose_recovery(
                RecoveryRequest(attempt_id="a", state_key=_key(fc))
            )
            self.assertTrue(sel.used_fallback)
            self.assertEqual(sel.action, DEFAULT_FALLBACK_TABLE[fc])

    def test_learned_arm_beats_others(self) -> None:
        # Hand-craft per-action stats so RE_SEGMENT dominates on the
        # specific cell we query.
        dim = onehot_dim()
        A = {a: list([1.0] * dim) for a in RECOVERY_ACTIONS}
        b = {a: list([0.0] * dim) for a in RECOVERY_ACTIONS}
        sup = {a: list([10] * dim) for a in RECOVERY_ACTIONS}  # all supported
        # Pump positive reward signal into re_segment for the
        # FAILURE_CLASS_EMPTY_AIR cell.
        key = _key(FAILURE_CLASS_EMPTY_AIR)
        x = state_key_to_onehot(key)
        for i, xi in enumerate(x):
            if xi == 1:
                A[RECOVERY_ACTION_RE_SEGMENT][i] += 50.0
                b[RECOVERY_ACTION_RE_SEGMENT][i] += 40.0
        policy = LinUCBRecoveryPolicy(
            A_diag_per_action={a: tuple(A[a]) for a in RECOVERY_ACTIONS},
            b_per_action={a: tuple(b[a]) for a in RECOVERY_ACTIONS},
            feature_support_per_action={
                a: tuple(sup[a]) for a in RECOVERY_ACTIONS
            },
            min_support_threshold=DEFAULT_MIN_SUPPORT_THRESHOLD,
        )
        sel = policy.propose_recovery(
            RecoveryRequest(attempt_id="a", state_key=key)
        )
        self.assertFalse(sel.used_fallback)
        self.assertEqual(sel.action, RECOVERY_ACTION_RE_SEGMENT)

    def test_tie_break_uses_declaration_order(self) -> None:
        # All arms identical → reobserve must win (first in declaration order).
        dim = onehot_dim()
        policy = LinUCBRecoveryPolicy(
            A_diag_per_action={a: tuple([1.0] * dim) for a in RECOVERY_ACTIONS},
            b_per_action={a: tuple([0.5] * dim) for a in RECOVERY_ACTIONS},
            feature_support_per_action={
                a: tuple([100] * dim) for a in RECOVERY_ACTIONS
            },
            min_support_threshold=DEFAULT_MIN_SUPPORT_THRESHOLD,
        )
        sel = policy.propose_recovery(
            RecoveryRequest(attempt_id="a", state_key=_key())
        )
        self.assertFalse(sel.used_fallback)
        self.assertEqual(sel.action, RECOVERY_ACTION_REOBSERVE)
        self.assertEqual(sel.ranked_actions[0], RECOVERY_ACTION_REOBSERVE)

    def test_rejects_bad_construction(self) -> None:
        with self.assertRaises(ValueError):
            LinUCBRecoveryPolicy(
                A_diag_per_action={"x": ()},
                b_per_action={"x": ()},
                feature_support_per_action={"x": ()},
            )

    def test_selection_validates_ranked_actions(self) -> None:
        with self.assertRaises(ValueError):
            RecoverySelection(
                policy_id="p@1",
                policy_version=1,
                action=RECOVERY_ACTION_REOBSERVE,
                ranked_actions=(RECOVERY_ACTION_REOBSERVE,) * len(RECOVERY_ACTIONS),
                scores=tuple(
                    RecoveryActionScore(
                        action=a,
                        expected_reward=0.0,
                        ucb=0.0,
                        min_support_over_active=0,
                    )
                    for a in RECOVERY_ACTIONS
                ),
                used_fallback=False,
            )


# ---------------------------------------------------------------------------
# Anti-loop gate.
# ---------------------------------------------------------------------------


class AntiLoopGateTests(unittest.TestCase):
    def test_pass_through_when_under_budget(self) -> None:
        r = apply_recovery_anti_loop_gate(
            proposed_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=1,
            max_recovery_attempts=3,
        )
        self.assertEqual(r.action, RECOVERY_ACTION_REOBSERVE)
        self.assertFalse(r.clipped)
        self.assertEqual(r.clip_reason, "none")

    def test_forces_abort_at_budget(self) -> None:
        r = apply_recovery_anti_loop_gate(
            proposed_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=3,
            max_recovery_attempts=3,
        )
        self.assertEqual(r.action, RECOVERY_ACTION_ABORT_RECOVERY)
        self.assertTrue(r.clipped)
        self.assertEqual(r.clip_reason, "max_recovery_attempts_reached")

    def test_abort_at_budget_is_idempotent(self) -> None:
        r = apply_recovery_anti_loop_gate(
            proposed_action=RECOVERY_ACTION_ABORT_RECOVERY,
            attempt_index=3,
            max_recovery_attempts=3,
        )
        self.assertEqual(r.action, RECOVERY_ACTION_ABORT_RECOVERY)
        self.assertFalse(r.clipped)

    def test_invalid_token_clipped(self) -> None:
        r = apply_recovery_anti_loop_gate(
            proposed_action="bogus",
            attempt_index=0,
            max_recovery_attempts=3,
        )
        self.assertEqual(r.action, RECOVERY_ACTION_ABORT_RECOVERY)
        self.assertTrue(r.clipped)
        self.assertEqual(r.clip_reason, "invalid_proposed_action")


# ---------------------------------------------------------------------------
# Trainer extraction + scope.
# ---------------------------------------------------------------------------


def _make_record(
    *,
    mode: str = "dense_clutter",
    final_outcome: str = "succeeded",
    recovery_actions: list[dict[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "timestamp": 0.0,
        "attempt_id": "test",
        "mode": mode,
        "final_outcome": final_outcome,
        "recovery_actions": recovery_actions or [],
        "extra": extra or {},
    }


class TrainerExtractionTests(unittest.TestCase):
    def test_token_map_covers_every_recovery_action(self) -> None:
        mapped = set(RECOVERY_TOKEN_MAP.values())
        self.assertEqual(mapped, set(RECOVERY_ACTIONS))

    def test_extract_known_token(self) -> None:
        rec = _make_record(
            mode="dense_clutter",
            final_outcome="succeeded",
            recovery_actions=[
                {"action": "next_viewpoint", "outcome": "recovered_success"},
            ],
        )
        tuples, dropped_token, dropped_no = extract_recovery_tuples(rec)
        self.assertEqual(dropped_token, 0)
        self.assertEqual(dropped_no, 0)
        self.assertEqual(len(tuples), 1)
        t = tuples[0]
        self.assertEqual(t.action, RECOVERY_ACTION_REOBSERVE)
        self.assertEqual(t.reward, 1.0)
        self.assertEqual(t.state_key.dense_bucket, DENSE_BUCKET_DENSE)

    def test_drop_record_with_no_actions(self) -> None:
        rec = _make_record(recovery_actions=[])
        tuples, dropped_token, dropped_no = extract_recovery_tuples(rec)
        self.assertEqual(tuples, ())
        self.assertEqual(dropped_token, 0)
        self.assertEqual(dropped_no, 1)

    def test_drop_unknown_token_silently_per_entry(self) -> None:
        rec = _make_record(
            recovery_actions=[
                {"action": "next_viewpoint", "outcome": "recovered_success"},
                {"action": "totally_unknown", "outcome": "recovered_failure"},
            ],
        )
        tuples, dropped_token, _ = extract_recovery_tuples(rec)
        self.assertEqual(len(tuples), 1)
        self.assertEqual(dropped_token, 1)

    def test_recovery_trail_extracts_to_rewarded_tuples(self) -> None:
        # END-TO-END: a real recovery trail -> recovery_actions_from_trail -> extract_recovery_tuples
        # yields NON-DEGENERATE recovery tuples, INCLUDING the new concrete sim actions (no drops) and a
        # reward of 1.0 credited to the last (recovering) step. This is the whole point of the trail
        # extraction (recovery tuples were degenerate before).
        from src.robot.grasping.types.feedback import GraspFailureReason
        from src.robot.grasping.recovery.policy import SceneRecoveryAction
        from src.robot.grasping.recovery.orchestrator import (
            RecoveryTrail,
            RecoveryTrailEntry,
            recovery_actions_from_trail,
        )

        def _e(action: SceneRecoveryAction, outcome: str) -> RecoveryTrailEntry:
            return RecoveryTrailEntry(
                attempt_index=0,
                failure_reason=GraspFailureReason.RESCAN_RECOMMENDED,
                plan_action=action,
                plan_reason="t",
                executed=True,
                outcome=outcome,
            )

        trail = RecoveryTrail(
            entries=(
                _e(SceneRecoveryAction.NUDGE_TARGET, "completed"),
                _e(SceneRecoveryAction.CONTAINER_AGITATE, "completed"),
            ),
            terminal_reason="recovered_success",
        )
        rec = _make_record(
            mode="dense_clutter",
            final_outcome="succeeded",
            recovery_actions=list(recovery_actions_from_trail(trail)),
        )
        tuples, dropped_token, dropped_no = extract_recovery_tuples(rec)
        self.assertEqual((dropped_token, dropped_no), (0, 0))  # non-degenerate, no drops
        self.assertEqual(len(tuples), 2)
        self.assertEqual(tuples[0].action, RECOVERY_ACTION_NUDGE_TARGET)
        self.assertEqual(tuples[1].action, RECOVERY_ACTION_CONTAINER_AGITATE)
        self.assertEqual(tuples[1].reward, 1.0)  # last (recovering) step rewarded
        self.assertEqual(tuples[0].reward, 0.0)


class TrainerScopeTests(unittest.TestCase):
    def test_dense_scope_excludes_easy_records(self) -> None:
        records = [
            _make_record(
                mode="easy",
                recovery_actions=[
                    {"action": "next_viewpoint", "outcome": "recovered_success"},
                ],
            ),
            _make_record(
                mode="dense_clutter",
                recovery_actions=[
                    {"action": "next_viewpoint", "outcome": "recovered_success"},
                ],
            ),
        ]
        _, result = train_linucb_recovery_policy(
            records,
            training_scope=TRAINING_SCOPE_DENSE,
            min_support_threshold=0,
        )
        self.assertEqual(result.num_records_total, 2)
        self.assertEqual(result.num_records_in_scope, 1)
        self.assertEqual(result.num_tuples_extracted, 1)

    def test_all_scope_admits_everything(self) -> None:
        records = [
            _make_record(
                mode="easy",
                recovery_actions=[
                    {"action": "next_viewpoint", "outcome": "recovered_success"},
                ],
            ),
            _make_record(
                mode="dense_clutter",
                recovery_actions=[
                    {"action": "next_viewpoint", "outcome": "recovered_success"},
                ],
            ),
        ]
        _, result = train_linucb_recovery_policy(
            records,
            training_scope=TRAINING_SCOPE_ALL,
            min_support_threshold=0,
        )
        self.assertEqual(result.num_records_in_scope, 2)


# ---------------------------------------------------------------------------
# Artifact round-trip + schema drift.
# ---------------------------------------------------------------------------


class ArtifactRoundTripTests(unittest.TestCase):
    def test_build_and_load(self) -> None:
        policy, result = train_linucb_recovery_policy(
            [
                _make_record(
                    mode="dense_clutter",
                    recovery_actions=[
                        {"action": "next_viewpoint", "outcome": "recovered_success"}
                    ],
                )
            ],
            min_support_threshold=0,
        )
        artifact = build_recovery_artifact_json(
            policy=policy,
            train_result=result,
            seed=1,
            dataset_id="t",
            dataset_hash="h",
            offline_kpi_estimates={"fallback_rate": 0.0},
        )
        self.assertEqual(
            artifact["schema_version"], RECOVERY_ARTIFACT_SCHEMA_VERSION
        )
        self.assertEqual(
            artifact["training_metadata"]["action_label_source"],
            ACTION_LABEL_SOURCE,
        )

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "art.json"
            sha = write_recovery_artifact(artifact, p)
            self.assertEqual(len(sha), 64)
            loaded = load_linucb_recovery_policy(p)
            self.assertEqual(loaded.name, policy.name)
            self.assertEqual(loaded.version, policy.version)

    def test_schema_drift_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "art.json"
            p.write_text(
                json.dumps({"schema_version": 999, "actions": []}),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_linucb_recovery_policy(p)

    def test_action_drift_rejected(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "art.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": RECOVERY_ARTIFACT_SCHEMA_VERSION,
                        "actions": ["reobserve"],
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_linucb_recovery_policy(p)


# ---------------------------------------------------------------------------
# Router seam.
# ---------------------------------------------------------------------------


class RouterIntegrationTests(unittest.TestCase):
    def test_telemetry_version_is_5(self) -> None:
        self.assertEqual(SHADOW_ROUTER_TELEMETRY_VERSION, 5)

    def test_returns_none_when_recovery_policy_missing(self) -> None:
        # Need a candidate_policy stub but we never invoke it; bypass
        # by constructing only the recovery seam.

        class _Noop:
            name = "noop"
            version = 0

            def propose_candidate(self, *_a, **_kw):  # pragma: no cover
                raise AssertionError("unused")

        router = ShadowRouter(candidate_policy=_Noop())  # type: ignore[arg-type]
        result = router.run_recovery_shadow(
            attempt_id="a",
            state_key=_key(),
            baseline_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=0,
            max_recovery_attempts=DEFAULT_MAX_RECOVERY_ATTEMPTS,
        )
        self.assertIsNone(result)

    def test_emits_typed_telemetry_when_wired(self) -> None:
        policy, _ = train_linucb_recovery_policy(
            [
                _make_record(
                    mode="dense_clutter",
                    recovery_actions=[
                        {"action": "next_viewpoint", "outcome": "recovered_success"}
                    ],
                )
            ],
            min_support_threshold=0,
        )

        class _Noop:
            name = "noop"
            version = 0

            def propose_candidate(self, *_a, **_kw):  # pragma: no cover
                raise AssertionError("unused")

        router = ShadowRouter(
            candidate_policy=_Noop(),  # type: ignore[arg-type]
            recovery_policy=policy,
        )
        tel = router.run_recovery_shadow(
            attempt_id="a",
            state_key=_key(),
            baseline_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=0,
            max_recovery_attempts=3,
        )
        self.assertIsInstance(tel, RecoveryShadowTelemetry)
        assert tel is not None  # for type checkers
        self.assertEqual(tel.router_path, ROUTER_PATH_RL_SHADOW)
        self.assertFalse(tel.fallback_triggered)
        self.assertFalse(tel.gate_clipped)
        self.assertEqual(tel.gated_action, tel.selection.action)

    def test_anti_loop_gate_forces_abort_at_budget(self) -> None:
        policy = _zero_policy(min_support=0)  # learnable but uninformative

        class _Noop:
            name = "noop"
            version = 0

            def propose_candidate(self, *_a, **_kw):  # pragma: no cover
                raise AssertionError("unused")

        router = ShadowRouter(
            candidate_policy=_Noop(),  # type: ignore[arg-type]
            recovery_policy=policy,
        )
        tel = router.run_recovery_shadow(
            attempt_id="a",
            state_key=_key(),
            baseline_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=3,
            max_recovery_attempts=3,
        )
        assert tel is not None
        self.assertEqual(tel.gated_action, RECOVERY_ACTION_ABORT_RECOVERY)
        self.assertTrue(tel.gate_clipped)
        self.assertEqual(tel.gate_clip_reason, "max_recovery_attempts_reached")

    def test_typed_fallback_on_policy_exception(self) -> None:
        class _Boom:
            name = "boom"
            version = 7

            def propose_recovery(self, _req):
                raise RuntimeError("kaboom")

        class _Noop:
            name = "noop"
            version = 0

            def propose_candidate(self, *_a, **_kw):  # pragma: no cover
                raise AssertionError("unused")

        router = ShadowRouter(
            candidate_policy=_Noop(),  # type: ignore[arg-type]
            recovery_policy=_Boom(),  # type: ignore[arg-type]
        )
        tel = router.run_recovery_shadow(
            attempt_id="a",
            state_key=_key(),
            baseline_action=RECOVERY_ACTION_REPLAN_GRASP,
            attempt_index=0,
            max_recovery_attempts=3,
        )
        assert tel is not None
        self.assertEqual(tel.router_path, ROUTER_PATH_FALLBACK_BASELINE)
        self.assertTrue(tel.fallback_triggered)
        self.assertEqual(tel.fallback_reason_code, "RuntimeError")
        self.assertEqual(tel.gated_action, RECOVERY_ACTION_REPLAN_GRASP)

    def test_emit_recovery_extras_shape(self) -> None:
        policy, _ = train_linucb_recovery_policy(
            [
                _make_record(
                    mode="dense_clutter",
                    recovery_actions=[
                        {"action": "next_viewpoint", "outcome": "recovered_success"}
                    ],
                )
            ],
            min_support_threshold=0,
        )

        class _Noop:
            name = "noop"
            version = 0

            def propose_candidate(self, *_a, **_kw):  # pragma: no cover
                raise AssertionError("unused")

        router = ShadowRouter(
            candidate_policy=_Noop(),  # type: ignore[arg-type]
            recovery_policy=policy,
        )
        tel = router.run_recovery_shadow(
            attempt_id="a",
            state_key=_key(),
            baseline_action=RECOVERY_ACTION_REOBSERVE,
            attempt_index=0,
            max_recovery_attempts=3,
        )
        assert tel is not None
        row = emit_recovery_extras(telemetry=tel)
        for k in (
            "rl_recovery_policy_id",
            "rl_recovery_action_proposed",
            "rl_recovery_action_gated",
            "rl_recovery_action_baseline",
            "rl_recovery_action_agree",
            "rl_recovery_state_key",
            "rl_recovery_gate_clipped",
            "rl_recovery_fallback_triggered",
        ):
            self.assertIn(k, row)


# ---------------------------------------------------------------------------
# Committed artifact byte-identity.
# ---------------------------------------------------------------------------


class CommittedArtifactTests(unittest.TestCase):
    def test_committed_artifact_exists_and_loads(self) -> None:
        self.assertTrue(COMMITTED_ARTIFACT.exists(), msg=str(COMMITTED_ARTIFACT))
        policy = load_linucb_recovery_policy(COMMITTED_ARTIFACT)
        self.assertEqual(policy.name, "v6_recovery_baseline_v1")
        self.assertEqual(policy.version, 1)
        self.assertEqual(policy.training_scope, "dense")

    def test_committed_artifact_sha256(self) -> None:
        data = COMMITTED_ARTIFACT.read_bytes()
        self.assertEqual(
            hashlib.sha256(data).hexdigest(), COMMITTED_ARTIFACT_SHA256
        )

    def test_committed_artifact_reproduces_from_packs(self) -> None:
        _, _, artifact = train_recovery_from_packs(
            pack_paths=[DENSE_PACK, EASY_PACK],
            seed=1,
            dataset_id="v6_recovery_canonical",
        )
        rebuilt = json.dumps(artifact, sort_keys=True, indent=2) + "\n"
        self.assertEqual(
            hashlib.sha256(rebuilt.encode("utf-8")).hexdigest(),
            COMMITTED_ARTIFACT_SHA256,
        )


# ---------------------------------------------------------------------------
# CLI smoke.
# ---------------------------------------------------------------------------


class CLISmokeTests(unittest.TestCase):
    def test_help_lists_train_recovery_policy(self) -> None:
        out = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.robot.grasping.rl",
                "--help",
            ],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=True,
        )
        self.assertIn("train-recovery-policy", out.stdout)


if __name__ == "__main__":
    unittest.main()
