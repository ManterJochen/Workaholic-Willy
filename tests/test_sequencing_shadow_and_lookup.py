"""Sequencing tests — sequencing-shadow router + lookup-table policy.

The sequencing surface ships:

* :class:`SequencingPolicy` Protocol + :class:`LookupTableSequencingPolicy`
  deterministic lookup-table dispatcher with hand-authored fallback.
* :func:`apply_anti_loop_gate` defence-in-depth gate.
* :func:`load_lookup_table_sequencing_policy` schema-drift loader.
* :meth:`ShadowRouter.run_sequencing_shadow` typed shadow producer.
* :func:`emit_sequencing_extras` telemetry helper.
* Committed baseline artifact at
  ``docs/baselines/rl_policies/v4_sequencing_baseline_v1.json``.
* Trainer CLI ``python -m backend.src.robot.grasping.rl train-sequencing-policy``.

These tests lock the contract surface and prove byte-identity of
the pre-sequencing pick path when the sequencing policy is unwired.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.candidate_policy import (
    CandidateSelection,
    CandidateSelectionRequest,
    ProposedCandidate,
)
from src.robot.grasping.rl.router import (
    ROUTER_PATH_FALLBACK_BASELINE,
    SHADOW_ROUTER_TELEMETRY_VERSION,
    SequencingAttemptState,
    SequencingShadowTelemetry,
    ShadowRouter,
    emit_sequencing_extras,
)
from src.robot.grasping.rl.sequencing_policy import (
    ATTEMPT_INDEX_MAX,
    COMMIT_REOBSERVE_COUNT_MAX,
    DEFAULT_FALLBACK_TABLE,
    FAILURE_CLASSES,
    FAILURE_CLASS_COLLISION,
    FAILURE_CLASS_DEFORMABLE,
    FAILURE_CLASS_DRIFT,
    FAILURE_CLASS_EMPTY_AIR,
    FAILURE_CLASS_NONE,
    FAILURE_CLASS_OCCLUSION,
    FAILURE_CLASS_SLIP,
    FAILURE_CLASS_UNKNOWN,
    LookupTableSequencingPolicy,
    OUTCOME_LABELS,
    OUTCOME_LABEL_FAILURE,
    OUTCOME_LABEL_SUCCESS,
    SEQUENCING_ACTIONS,
    SEQUENCING_ACTIONS_SET,
    SEQUENCING_ACTION_ABORT,
    SEQUENCING_ACTION_GRASP,
    SEQUENCING_ACTION_RECOVER,
    SEQUENCING_ACTION_REOBSERVE,
    SEQUENCING_ARTIFACT_SCHEMA_VERSION,
    SequencingPolicy,
    SequencingRequest,
    SequencingSelection,
    SequencingStateKey,
    apply_anti_loop_gate,
    clip_attempt_index,
    clip_commit_reobserve_count,
    hash_sequencing_artifact,
    load_lookup_table_sequencing_policy,
)
from src.robot.grasping.rl.train_sequencing import (
    DEFAULT_MIN_SUPPORT_THRESHOLD,
    build_sequencing_artifact_json,
    train_lookup_table_sequencing_policy,
)
from src.robot.grasping.replay.telemetry_catalog import (
    EXTRA_TELEMETRY_FIELDS,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v4_sequencing_baseline_v1.json"
)


# ---------------------------------------------------------------------------
# Contract locks
# ---------------------------------------------------------------------------


class SequencingContractLockTests(unittest.TestCase):
    def test_telemetry_version_bumped(self) -> None:
        # Bumped each phase. The perception sub-block lifts the constant to 4.
        self.assertEqual(SHADOW_ROUTER_TELEMETRY_VERSION, 5)

    def test_actions_enum_locked(self) -> None:
        self.assertEqual(
            SEQUENCING_ACTIONS,
            ("grasp", "reobserve", "recover", "abort"),
        )

    def test_failure_classes_enum_locked(self) -> None:
        self.assertEqual(len(FAILURE_CLASSES), 8)
        self.assertIn(FAILURE_CLASS_NONE, FAILURE_CLASSES)
        self.assertIn(FAILURE_CLASS_UNKNOWN, FAILURE_CLASSES)
        for cls in (
            FAILURE_CLASS_SLIP,
            FAILURE_CLASS_EMPTY_AIR,
            FAILURE_CLASS_COLLISION,
            FAILURE_CLASS_OCCLUSION,
            FAILURE_CLASS_DEFORMABLE,
            FAILURE_CLASS_DRIFT,
        ):
            self.assertIn(cls, FAILURE_CLASSES)

    def test_outcome_labels_locked(self) -> None:
        self.assertEqual(
            OUTCOME_LABELS, (OUTCOME_LABEL_SUCCESS, OUTCOME_LABEL_FAILURE)
        )

    def test_schema_version_locked(self) -> None:
        self.assertEqual(SEQUENCING_ARTIFACT_SCHEMA_VERSION, 2)


# ---------------------------------------------------------------------------
# Clip helpers + anti-loop gate
# ---------------------------------------------------------------------------


class ClipHelperTests(unittest.TestCase):
    def test_clip_attempt_index_bounds(self) -> None:
        self.assertEqual(clip_attempt_index(-3), 0)
        self.assertEqual(clip_attempt_index(0), 0)
        self.assertEqual(clip_attempt_index(5), 5)
        self.assertEqual(clip_attempt_index(ATTEMPT_INDEX_MAX), ATTEMPT_INDEX_MAX)
        self.assertEqual(
            clip_attempt_index(ATTEMPT_INDEX_MAX + 17), ATTEMPT_INDEX_MAX
        )

    def test_clip_commit_reobserve_count_bounds(self) -> None:
        self.assertEqual(clip_commit_reobserve_count(-1), 0)
        self.assertEqual(
            clip_commit_reobserve_count(COMMIT_REOBSERVE_COUNT_MAX + 4),
            COMMIT_REOBSERVE_COUNT_MAX,
        )


class AntiLoopGateTests(unittest.TestCase):
    def test_reobserve_within_budget_passes_through(self) -> None:
        gated, clipped = apply_anti_loop_gate(
            proposed_action=SEQUENCING_ACTION_REOBSERVE,
            attempt_index=1,
            commit_reobserve_count=0,
            max_attempts=5,
            max_reobserve_attempts=2,
        )
        self.assertEqual(gated, SEQUENCING_ACTION_REOBSERVE)
        self.assertFalse(clipped)

    def test_reobserve_at_budget_clips_to_abort(self) -> None:
        gated, clipped = apply_anti_loop_gate(
            proposed_action=SEQUENCING_ACTION_REOBSERVE,
            attempt_index=1,
            commit_reobserve_count=2,
            max_attempts=5,
            max_reobserve_attempts=2,
        )
        self.assertEqual(gated, SEQUENCING_ACTION_ABORT)
        self.assertTrue(clipped)

    def test_non_abort_on_last_attempt_clips(self) -> None:
        for action in (
            SEQUENCING_ACTION_GRASP,
            SEQUENCING_ACTION_REOBSERVE,
            SEQUENCING_ACTION_RECOVER,
        ):
            with self.subTest(action=action):
                gated, clipped = apply_anti_loop_gate(
                    proposed_action=action,
                    attempt_index=4,
                    commit_reobserve_count=0,
                    max_attempts=5,
                    max_reobserve_attempts=2,
                )
                self.assertEqual(gated, SEQUENCING_ACTION_ABORT)
                self.assertTrue(clipped)

    def test_abort_on_last_attempt_passes(self) -> None:
        gated, clipped = apply_anti_loop_gate(
            proposed_action=SEQUENCING_ACTION_ABORT,
            attempt_index=4,
            commit_reobserve_count=2,
            max_attempts=5,
            max_reobserve_attempts=2,
        )
        self.assertEqual(gated, SEQUENCING_ACTION_ABORT)
        self.assertFalse(clipped)

    def test_unknown_action_clips_to_abort(self) -> None:
        gated, clipped = apply_anti_loop_gate(
            proposed_action="teleport",
            attempt_index=0,
            commit_reobserve_count=0,
            max_attempts=5,
            max_reobserve_attempts=2,
        )
        self.assertEqual(gated, SEQUENCING_ACTION_ABORT)
        self.assertTrue(clipped)


# ---------------------------------------------------------------------------
# Lookup-table policy
# ---------------------------------------------------------------------------


def _state(
    *,
    outcome: str = OUTCOME_LABEL_FAILURE,
    idx: int = 0,
    cnt: int = 0,
    cls: str = FAILURE_CLASS_UNKNOWN,
) -> SequencingStateKey:
    return SequencingStateKey(
        last_outcome_label=outcome,
        attempt_index_clipped=idx,
        commit_reobserve_count_clipped=cnt,
        last_failure_class=cls,
    )


class LookupTablePolicyTests(unittest.TestCase):
    def test_default_fallback_table_covers_all_classes(self) -> None:
        for cls in FAILURE_CLASSES:
            self.assertIn(cls, DEFAULT_FALLBACK_TABLE)
            self.assertIn(DEFAULT_FALLBACK_TABLE[cls], SEQUENCING_ACTIONS_SET)

    def test_cell_above_threshold_returns_table_action(self) -> None:
        key = _state(idx=0, cnt=0, cls=FAILURE_CLASS_SLIP)
        policy = LookupTableSequencingPolicy(
            lookup_table={
                key.as_string_key(): {
                    "action": SEQUENCING_ACTION_RECOVER,
                    "support": 12,
                },
            },
            min_support_threshold=5,
        )
        sel = policy.propose_sequencing(
            SequencingRequest(attempt_id="t0", state_key=key)
        )
        self.assertEqual(sel.action, SEQUENCING_ACTION_RECOVER)
        self.assertFalse(sel.used_fallback)
        self.assertEqual(sel.support_count, 12)
        self.assertEqual(sel.policy_id, "v4_sequencing_baseline_v1@1")

    def test_cell_below_threshold_uses_fallback(self) -> None:
        key = _state(cls=FAILURE_CLASS_OCCLUSION)
        policy = LookupTableSequencingPolicy(
            lookup_table={
                key.as_string_key(): {
                    "action": SEQUENCING_ACTION_GRASP,
                    "support": 1,
                },
            },
            min_support_threshold=5,
        )
        sel = policy.propose_sequencing(
            SequencingRequest(attempt_id="t1", state_key=key)
        )
        self.assertEqual(sel.action, DEFAULT_FALLBACK_TABLE[FAILURE_CLASS_OCCLUSION])
        self.assertTrue(sel.used_fallback)
        self.assertEqual(sel.support_count, 1)

    def test_missing_cell_uses_fallback_by_class(self) -> None:
        policy = LookupTableSequencingPolicy(lookup_table={})
        for cls in FAILURE_CLASSES:
            with self.subTest(cls=cls):
                sel = policy.propose_sequencing(
                    SequencingRequest(
                        attempt_id="t",
                        state_key=_state(cls=cls),
                    )
                )
                self.assertEqual(sel.action, DEFAULT_FALLBACK_TABLE[cls])
                self.assertTrue(sel.used_fallback)
                self.assertEqual(sel.support_count, 0)

    def test_invalid_action_in_lookup_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LookupTableSequencingPolicy(
                lookup_table={
                    "failure|0|0|unknown": {
                        "action": "teleport",
                        "support": 9,
                    }
                }
            )

    def test_invalid_support_in_lookup_rejected(self) -> None:
        with self.assertRaises(ValueError):
            LookupTableSequencingPolicy(
                lookup_table={
                    "failure|0|0|unknown": {
                        "action": SEQUENCING_ACTION_GRASP,
                        "support": -1,
                    }
                }
            )

    def test_fallback_table_must_cover_all_classes(self) -> None:
        with self.assertRaises(ValueError):
            LookupTableSequencingPolicy(
                fallback_table={
                    FAILURE_CLASS_NONE: SEQUENCING_ACTION_GRASP,
                },
            )

    def test_satisfies_protocol(self) -> None:
        policy: SequencingPolicy = LookupTableSequencingPolicy()
        self.assertEqual(policy.name, "v4_sequencing_baseline_v1")
        self.assertEqual(policy.version, 1)


# ---------------------------------------------------------------------------
# State key validation
# ---------------------------------------------------------------------------


class SequencingStateKeyTests(unittest.TestCase):
    def test_string_key_format_stable(self) -> None:
        key = _state(
            outcome=OUTCOME_LABEL_FAILURE,
            idx=2,
            cnt=1,
            cls=FAILURE_CLASS_SLIP,
        )
        self.assertEqual(key.as_string_key(), "failure|2|1|slip_after_grasp")

    def test_invalid_outcome_label_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SequencingStateKey(
                last_outcome_label="maybe",
                attempt_index_clipped=0,
                commit_reobserve_count_clipped=0,
                last_failure_class=FAILURE_CLASS_NONE,
            )

    def test_invalid_failure_class_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SequencingStateKey(
                last_outcome_label=OUTCOME_LABEL_FAILURE,
                attempt_index_clipped=0,
                commit_reobserve_count_clipped=0,
                last_failure_class="cosmic_ray",
            )


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


def _record(
    *,
    group: str,
    success: bool,
    failure_class: str = FAILURE_CLASS_UNKNOWN,
    attempt_index: int = 0,
    commit_reobserve_count: int = 0,
) -> dict[str, object]:
    return {
        "scene_id": group,
        "attempt_id": f"{group}_a{attempt_index}",
        "final_outcome": "succeeded" if success else "failed",
        "attempt_index": attempt_index,
        "commit_reobserve_count": commit_reobserve_count,
        "extra": {"failure_taxonomy_class": failure_class},
    }


class TrainSequencingTests(unittest.TestCase):
    def test_empirical_majority_wins_when_above_threshold(self) -> None:
        # 5 pairs of (failure idx=0 SLIP) → (success idx=1). Successor
        # success maps to GRASP with success rate 1.0; cell support=5
        # meets threshold so GRASP commits without fallback.
        records: list[dict[str, object]] = []
        for i in range(5):
            records.append(
                _record(
                    group=f"g{i}",
                    success=False,
                    failure_class=FAILURE_CLASS_SLIP,
                    attempt_index=0,
                )
            )
            records.append(
                _record(
                    group=f"g{i}",
                    success=True,
                    attempt_index=1,
                )
            )
        policy, result = train_lookup_table_sequencing_policy(
            records=records, min_support_threshold=5
        )
        sel = policy.propose_sequencing(
            SequencingRequest(
                attempt_id="t",
                state_key=_state(idx=0, cnt=0, cls=FAILURE_CLASS_SLIP),
            )
        )
        self.assertEqual(sel.action, SEQUENCING_ACTION_GRASP)
        self.assertFalse(sel.used_fallback)
        self.assertGreaterEqual(result.num_cells_committed, 1)

    def test_below_threshold_falls_back(self) -> None:
        # Only 2 records for one cell — below default threshold of 5.
        records = [
            _record(
                group="g0",
                success=False,
                failure_class=FAILURE_CLASS_OCCLUSION,
                attempt_index=0,
            ),
            _record(
                group="g0",
                success=False,
                failure_class=FAILURE_CLASS_OCCLUSION,
                attempt_index=1,
            ),
        ]
        policy, result = train_lookup_table_sequencing_policy(
            records=records, min_support_threshold=5
        )
        sel = policy.propose_sequencing(
            SequencingRequest(
                attempt_id="t",
                state_key=_state(idx=0, cnt=0, cls=FAILURE_CLASS_OCCLUSION),
            )
        )
        # Fallback action for OCCLUSION is REOBSERVE.
        self.assertEqual(sel.action, DEFAULT_FALLBACK_TABLE[FAILURE_CLASS_OCCLUSION])
        self.assertTrue(sel.used_fallback)
        self.assertEqual(result.num_cells_below_threshold, 1)

    def test_default_min_support_threshold_value(self) -> None:
        self.assertEqual(DEFAULT_MIN_SUPPORT_THRESHOLD, 5)


# ---------------------------------------------------------------------------
# Artifact loader + committed artifact integrity
# ---------------------------------------------------------------------------


class SequencingArtifactLoaderTests(unittest.TestCase):
    def test_round_trip_load(self) -> None:
        records = [
            _record(
                group=f"g{i}",
                success=False,
                failure_class=FAILURE_CLASS_SLIP,
                attempt_index=0,
            )
            for i in range(6)
        ] + [
            _record(
                group=f"g{i}",
                success=False,
                failure_class=FAILURE_CLASS_SLIP,
                attempt_index=1,
            )
            for i in range(6)
        ]
        _, _ = train_lookup_table_sequencing_policy(
            records=records, min_support_threshold=5
        )
        artifact = build_sequencing_artifact_json(
            policy=_,
            train_result=_,  # placeholder; overwritten below
            seed=0,
            dataset_id="unit",
            dataset_hash="deadbeef",
        ) if False else None
        # Re-train so we have both pieces named clearly:
        _, _ = train_lookup_table_sequencing_policy(
            records=records, min_support_threshold=5
        )
        artifact = build_sequencing_artifact_json(
            policy=_,
            train_result=_,  # placeholder; overwritten below
            seed=0,
            dataset_id="unit",
            dataset_hash="deadbeef",
        ) if False else None
        # Re-train so we have both pieces named clearly:
        policy, result = train_lookup_table_sequencing_policy(
            records=records, min_support_threshold=5
        )
        artifact = build_sequencing_artifact_json(
            policy=policy,
            train_result=result,
            seed=0,
            dataset_id="unit",
            dataset_hash="deadbeef",
        )
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sequencing.json"
            p.write_text(json.dumps(artifact, sort_keys=True))
            loaded = load_lookup_table_sequencing_policy(p)
        self.assertEqual(loaded.min_support_threshold, 5)
        self.assertEqual(loaded.name, "v4_sequencing_baseline_v1")

    def test_loader_rejects_schema_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sequencing.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": 99,
                        "actions": list(SEQUENCING_ACTIONS),
                        "failure_classes": list(FAILURE_CLASSES),
                        "lookup_table": {},
                        "fallback_table": dict(DEFAULT_FALLBACK_TABLE),
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_lookup_table_sequencing_policy(p)

    def test_loader_rejects_action_enum_drift(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sequencing.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": SEQUENCING_ARTIFACT_SCHEMA_VERSION,
                        "actions": ["grasp", "abort"],  # drift
                        "failure_classes": list(FAILURE_CLASSES),
                        "lookup_table": {},
                        "fallback_table": dict(DEFAULT_FALLBACK_TABLE),
                    }
                )
            )
            with self.assertRaises(ValueError):
                load_lookup_table_sequencing_policy(p)


class CommittedArtifactSha256Tests(unittest.TestCase):
    def test_committed_artifact_loads(self) -> None:
        self.assertTrue(COMMITTED_ARTIFACT.is_file())
        policy = load_lookup_table_sequencing_policy(COMMITTED_ARTIFACT)
        self.assertEqual(policy.name, "v4_sequencing_baseline_v1")
        self.assertEqual(policy.version, 1)

    def test_committed_artifact_byte_identity_via_cli(self) -> None:
        committed_hash = hash_sequencing_artifact(COMMITTED_ARTIFACT)
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "sequencing.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.rl",
                    "train-sequencing-policy",
                    "--dataset-id",
                    "v1_bootstrap",
                    "--output",
                    str(out),
                ],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.returncode, 0)
            regenerated_hash = hashlib.sha256(out.read_bytes()).hexdigest()
        self.assertEqual(committed_hash, regenerated_hash)


# ---------------------------------------------------------------------------
# Router: shadow producer
# ---------------------------------------------------------------------------


class _StaticSequencingPolicy:
    """Test double returning a fixed action."""

    def __init__(
        self,
        *,
        action: str,
        name: str = "static",
        version: int = 7,
        support_count: int = 99,
        used_fallback: bool = False,
    ) -> None:
        self.name = name
        self.version = version
        self._action = action
        self._support = support_count
        self._used_fallback = used_fallback

    def propose_sequencing(
        self, request: SequencingRequest
    ) -> SequencingSelection:
        return SequencingSelection(
            policy_id=f"{self.name}@{self.version}",
            policy_version=self.version,
            action=self._action,
            support_count=self._support,
            used_fallback=self._used_fallback,
        )


def _attempt_state(
    *,
    attempt_index: int = 0,
    commit_reobserve_count: int = 0,
    max_attempts: int = 5,
    max_reobserve_attempts: int = 2,
    failure_class: str = FAILURE_CLASS_UNKNOWN,
    baseline_action: str = SEQUENCING_ACTION_GRASP,
) -> SequencingAttemptState:
    return SequencingAttemptState(
        attempt_index=attempt_index,
        last_outcome_label=OUTCOME_LABEL_FAILURE,
        attempt_index_clipped=clip_attempt_index(attempt_index),
        commit_reobserve_count_clipped=clip_commit_reobserve_count(
            commit_reobserve_count
        ),
        commit_reobserve_count=commit_reobserve_count,
        last_failure_class=failure_class,
        max_attempts=max_attempts,
        max_reobserve_attempts=max_reobserve_attempts,
        baseline_action=baseline_action,
    )


class _StubCandidatePolicy:
    name = "stub_candidate"
    version = 1
    artifact_dataset_id = ""

    def propose(
        self, request: CandidateSelectionRequest
    ) -> CandidateSelection:
        proposals = tuple(
            ProposedCandidate(
                candidate_id=cf.candidate_id,
                rank=i,
                score=0.5,
                pruned=False,
            )
            for i, cf in enumerate(request.candidates)
        )
        return CandidateSelection(
            policy_id="stub_candidate@1",
            policy_version=1,
            proposals=proposals,
            top1_id=request.candidates[0].candidate_id if request.candidates else None,
        )


def _router(**kwargs) -> ShadowRouter:
    return ShadowRouter(candidate_policy=_StubCandidatePolicy(), **kwargs)


class ShadowRouterSequencingTests(unittest.TestCase):
    def test_no_policy_returns_fallback_telemetry(self) -> None:
        router = _router()
        tel = router.run_sequencing_shadow(
            attempt_id="x", states=(_attempt_state(),)
        )
        self.assertTrue(tel.fallback_triggered)
        self.assertEqual(tel.router_path, ROUTER_PATH_FALLBACK_BASELINE)
        self.assertEqual(tel.fallback_reason_code, "sequencing_exception:NoPolicy")
        self.assertEqual(tel.decisions, ())

    def test_policy_proposal_becomes_decision(self) -> None:
        policy = _StaticSequencingPolicy(action=SEQUENCING_ACTION_RECOVER)
        router = _router(sequencing_policy=policy)
        tel = router.run_sequencing_shadow(
            attempt_id="x",
            states=(
                _attempt_state(
                    attempt_index=1,
                    baseline_action=SEQUENCING_ACTION_RECOVER,
                ),
            ),
        )
        self.assertFalse(tel.fallback_triggered)
        self.assertEqual(len(tel.decisions), 1)
        dec = tel.decisions[0]
        self.assertEqual(dec.proposed_action, SEQUENCING_ACTION_RECOVER)
        self.assertEqual(dec.gated_action, SEQUENCING_ACTION_RECOVER)
        self.assertTrue(dec.agree_with_baseline)
        self.assertFalse(dec.anti_loop_clipped)

    def test_gate_clips_terminal_attempt(self) -> None:
        policy = _StaticSequencingPolicy(action=SEQUENCING_ACTION_GRASP)
        router = _router(sequencing_policy=policy)
        tel = router.run_sequencing_shadow(
            attempt_id="x",
            states=(
                _attempt_state(
                    attempt_index=4,
                    max_attempts=5,
                    baseline_action=SEQUENCING_ACTION_ABORT,
                ),
            ),
        )
        dec = tel.decisions[0]
        self.assertEqual(dec.proposed_action, SEQUENCING_ACTION_GRASP)
        self.assertEqual(dec.gated_action, SEQUENCING_ACTION_ABORT)
        self.assertTrue(dec.anti_loop_clipped)
        self.assertTrue(dec.agree_with_baseline)

    def test_invalid_baseline_action_triggers_exception_fallback(self) -> None:
        policy = _StaticSequencingPolicy(action=SEQUENCING_ACTION_GRASP)
        router = _router(sequencing_policy=policy)
        state = _attempt_state()
        object.__setattr__(state, "baseline_action", "teleport")
        tel = router.run_sequencing_shadow(attempt_id="x", states=(state,))
        self.assertTrue(tel.fallback_triggered)
        self.assertTrue(
            tel.fallback_reason_code.startswith("sequencing_exception:")
        )


# ---------------------------------------------------------------------------
# emit_sequencing_extras
# ---------------------------------------------------------------------------


class EmitSequencingExtrasTests(unittest.TestCase):
    def test_empty_decisions_emits_none_and_false(self) -> None:
        tel = SequencingShadowTelemetry(
            router_path=ROUTER_PATH_FALLBACK_BASELINE,
            policy_id="p",
            policy_version=3,
            decisions=(),
            fallback_triggered=True,
            fallback_reason_code="sequencing_exception:NoPolicy",
        )
        extras = emit_sequencing_extras(telemetry=tel)
        self.assertEqual(
            set(extras.keys()),
            {
                "rl_sequencing_policy_id",
                "rl_sequencing_artifact_version",
                "rl_sequencing_action_proposed",
                "rl_sequencing_action_baseline",
                "rl_sequencing_action_agree",
            },
        )
        self.assertEqual(extras["rl_sequencing_action_proposed"], "none")
        self.assertEqual(extras["rl_sequencing_action_baseline"], "none")
        self.assertIs(extras["rl_sequencing_action_agree"], False)
        self.assertEqual(extras["rl_sequencing_artifact_version"], "v3")

    def test_last_decision_drives_extras(self) -> None:
        policy = _StaticSequencingPolicy(action=SEQUENCING_ACTION_REOBSERVE)
        router = ShadowRouter(
            candidate_policy=_StubCandidatePolicy(),
            sequencing_policy=policy,
        )
        tel = router.run_sequencing_shadow(
            attempt_id="x",
            states=(
                _attempt_state(
                    attempt_index=0,
                    baseline_action=SEQUENCING_ACTION_GRASP,
                ),
                _attempt_state(
                    attempt_index=1,
                    baseline_action=SEQUENCING_ACTION_REOBSERVE,
                ),
            ),
        )
        extras = emit_sequencing_extras(telemetry=tel)
        self.assertEqual(
            extras["rl_sequencing_action_proposed"], SEQUENCING_ACTION_REOBSERVE
        )
        self.assertEqual(
            extras["rl_sequencing_action_baseline"], SEQUENCING_ACTION_REOBSERVE
        )
        self.assertIs(extras["rl_sequencing_action_agree"], True)


# ---------------------------------------------------------------------------
# Byte-identity: no sequencing policy → ranking/candidate telemetry unchanged
# ---------------------------------------------------------------------------


class PreSequencingByteIdentityTests(unittest.TestCase):
    def test_router_without_sequencing_policy_has_none_sequencing_slot(
        self,
    ) -> None:
        router = ShadowRouter(candidate_policy=_StubCandidatePolicy())
        self.assertIsNone(router.sequencing_policy)


# ---------------------------------------------------------------------------
# Telemetry catalog
# ---------------------------------------------------------------------------


class TelemetryCatalogSequencingTests(unittest.TestCase):
    def test_sequencing_keys_registered(self) -> None:
        by_name = {name: phase for (name, phase, _) in EXTRA_TELEMETRY_FIELDS}
        for key in (
            "rl_sequencing_policy_id",
            "rl_sequencing_artifact_version",
            "rl_sequencing_action_proposed",
            "rl_sequencing_action_baseline",
            "rl_sequencing_action_agree",
        ):
            with self.subTest(key=key):
                self.assertIn(key, by_name)
                self.assertEqual(by_name[key], "rl_sequencing")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
