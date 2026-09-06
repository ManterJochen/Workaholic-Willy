"""Ranking-shadow router + pairwise-logistic policy tests.

The ranking surface ships:

* :class:`RankingPolicy` Protocol + :class:`PairwiseLogisticRankingPolicy`
  deterministic dispatcher.
* :func:`kendall_tau_b` with full tie correction.
* :func:`load_pairwise_logistic_ranking_policy` schema-drift loader.
* :meth:`ShadowRouter.run_ranking_shadow` typed shadow producer.
* :func:`emit_ranking_extras` telemetry helper.
* Committed baseline artifact at
  ``docs/baselines/rl_policies/v3_ranking_baseline_v1.json``.
* Trainer CLI ``python -m backend.src.robot.grasping.rl train-ranking-policy``.

These tests lock the contract surface and prove byte-identity of
the pre-ranking pick path when the ranker is unwired.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.candidate_policy import (
    CANDIDATE_FEATURE_KEYS,
    CandidateSelection,
    CandidateSelectionRequest,
)
from src.robot.grasping.rl.ranking_policy import (
    RANKING_FEATURE_KEYS,
    PairwiseLogisticRankingPolicy,
    RankedCandidate,
    RankingCandidateFeatures,
    RankingSelection,
    RankingSelectionRequest,
    hash_ranking_artifact,
    kendall_tau_b,
    load_pairwise_logistic_ranking_policy,
)
from src.robot.grasping.rl.router import (
    KENDALL_TAU_BOUNDS,
    ROUTER_PATH_FALLBACK_BASELINE,
    SHADOW_ROUTER_TELEMETRY_VERSION,
    RankingShadowTelemetry,
    ShadowRouter,
    ShadowRouterTelemetry,
    emit_ranking_extras,
)
from src.robot.grasping.rl.train_ranking import (
    RankingTrainResult,
    train_pairwise_logistic_ranker,
)
from src.robot.grasping.loop.pick_loop import _build_candidate_log

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = (
    REPO_ROOT / "docs" / "baselines" / "rl_policies" / "v3_ranking_baseline_v1.json"
)


# ---------------------------------------------------------------------------
# Contract locks
# ---------------------------------------------------------------------------


class RankingContractLockTests(unittest.TestCase):
    def test_telemetry_version_bumped(self) -> None:
        # Bumped each phase; the perception sub-block lifts the constant to 4.
        self.assertEqual(SHADOW_ROUTER_TELEMETRY_VERSION, 5)

    def test_ranking_feature_keys_length_and_suffix(self) -> None:
        self.assertEqual(len(RANKING_FEATURE_KEYS), 15)
        self.assertEqual(
            RANKING_FEATURE_KEYS[: len(CANDIDATE_FEATURE_KEYS)],
            CANDIDATE_FEATURE_KEYS,
        )
        self.assertEqual(
            RANKING_FEATURE_KEYS[-3:],
            (
                "geometric_score",
                "feasibility_score",
                "shadow_predicted_success_probability",
            ),
        )


# ---------------------------------------------------------------------------
# kendall_tau_b
# ---------------------------------------------------------------------------


class KendallTauBTests(unittest.TestCase):
    def test_identical_order_is_one(self) -> None:
        self.assertEqual(
            kendall_tau_b(("a", "b", "c", "d"), ("a", "b", "c", "d")),
            1.0,
        )

    def test_reverse_order_is_minus_one(self) -> None:
        self.assertEqual(
            kendall_tau_b(("a", "b", "c", "d"), ("d", "c", "b", "a")),
            -1.0,
        )

    def test_single_swap_value(self) -> None:
        # 4 items, one adjacent swap → C=5 D=1 → 4/6
        tau = kendall_tau_b(
            ("a", "b", "c", "d"), ("b", "a", "c", "d")
        )
        self.assertAlmostEqual(tau, 4.0 / 6.0, places=12)

    def test_less_than_two_common_returns_zero(self) -> None:
        self.assertEqual(kendall_tau_b((), ()), 0.0)
        self.assertEqual(kendall_tau_b(("a",), ("a",)), 0.0)
        self.assertEqual(kendall_tau_b(("a",), ("b",)), 0.0)

    def test_disjoint_common_set_zero(self) -> None:
        self.assertEqual(
            kendall_tau_b(("a", "b"), ("c", "d")), 0.0
        )

    def test_in_bounds(self) -> None:
        tau = kendall_tau_b(
            ("a", "b", "c", "d", "e"), ("c", "b", "a", "e", "d")
        )
        self.assertGreaterEqual(tau, -1.0)
        self.assertLessEqual(tau, 1.0)


# ---------------------------------------------------------------------------
# PairwiseLogisticRankingPolicy
# ---------------------------------------------------------------------------


def _zero_policy() -> PairwiseLogisticRankingPolicy:
    return PairwiseLogisticRankingPolicy(
        weights=tuple(0.0 for _ in RANKING_FEATURE_KEYS),
        bias=0.0,
    )


def _bias_one_policy() -> PairwiseLogisticRankingPolicy:
    return PairwiseLogisticRankingPolicy(
        weights=tuple(0.0 for _ in RANKING_FEATURE_KEYS),
        bias=1.0,
    )


class PairwiseLogisticRankingPolicyTests(unittest.TestCase):
    def test_weights_length_mismatch_rejected(self) -> None:
        with self.assertRaises(ValueError):
            PairwiseLogisticRankingPolicy(
                weights=(0.1, 0.2), bias=0.0
            )

    def test_score_returns_sigmoid_of_bias_for_zero_weights(self) -> None:
        policy = _bias_one_policy()
        # σ(1) ≈ 0.7310585786
        self.assertAlmostEqual(
            policy.score_features({}), 1.0 / (1.0 + math.exp(-1.0)), places=10
        )

    def test_propose_ranking_is_deterministic(self) -> None:
        policy = _zero_policy()
        cands = (
            RankingCandidateFeatures(candidate_id="c0", features={}),
            RankingCandidateFeatures(candidate_id="c1", features={}),
            RankingCandidateFeatures(candidate_id="c2", features={}),
        )
        req = RankingSelectionRequest(
            attempt_id="t",
            candidates=cands,
            deterministic_ranking=("c0", "c1", "c2"),
        )
        s1 = policy.propose_ranking(req)
        s2 = policy.propose_ranking(req)
        self.assertEqual(s1, s2)
        # Tied scores → tie-break by candidate_id asc
        self.assertEqual(
            tuple(p.candidate_id for p in s1.proposals),
            ("c0", "c1", "c2"),
        )

    def test_propose_ranking_top1_via_weights(self) -> None:
        # Single positive weight on the last feature → candidate with
        # the largest value of that feature wins.
        w = [0.0] * len(RANKING_FEATURE_KEYS)
        w[-1] = 5.0
        policy = PairwiseLogisticRankingPolicy(
            weights=tuple(w), bias=0.0
        )
        cands = (
            RankingCandidateFeatures(
                candidate_id="loser",
                features={"shadow_predicted_success_probability": 0.1},
            ),
            RankingCandidateFeatures(
                candidate_id="winner",
                features={"shadow_predicted_success_probability": 0.9},
            ),
        )
        req = RankingSelectionRequest(
            attempt_id="t",
            candidates=cands,
            deterministic_ranking=("loser", "winner"),
        )
        sel = policy.propose_ranking(req)
        self.assertEqual(sel.top1_id, "winner")
        self.assertTrue(sel.regret_top1)
        self.assertAlmostEqual(sel.kendall_tau, -1.0, places=10)

    def test_empty_candidates_no_regret(self) -> None:
        policy = _zero_policy()
        req = RankingSelectionRequest(
            attempt_id="t",
            candidates=(),
            deterministic_ranking=("c0",),
        )
        sel = policy.propose_ranking(req)
        self.assertFalse(sel.regret_top1)
        self.assertEqual(sel.proposals, ())
        self.assertEqual(sel.kendall_tau, 0.0)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


class RankingArtifactLoaderTests(unittest.TestCase):
    def test_loads_committed_artifact(self) -> None:
        policy = load_pairwise_logistic_ranking_policy(COMMITTED_ARTIFACT)
        self.assertIsInstance(policy, PairwiseLogisticRankingPolicy)
        self.assertEqual(policy.name, "v3_ranking_baseline_v1")
        self.assertEqual(policy.version, 1)
        self.assertEqual(len(policy.weights), len(RANKING_FEATURE_KEYS))

    def test_schema_version_mismatch_rejected(self) -> None:
        blob = json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))
        blob["schema_version"] = 999
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as f:
            json.dump(blob, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pairwise_logistic_ranking_policy(path)
        finally:
            Path(path).unlink()

    def test_feature_keys_mismatch_rejected(self) -> None:
        blob = json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))
        blob["feature_keys"] = list(RANKING_FEATURE_KEYS[:-1])
        with tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False
        ) as f:
            json.dump(blob, f)
            path = f.name
        try:
            with self.assertRaises(ValueError):
                load_pairwise_logistic_ranking_policy(path)
        finally:
            Path(path).unlink()


# ---------------------------------------------------------------------------
# Committed artifact byte-identity (regenerate must match)
# ---------------------------------------------------------------------------


class CommittedArtifactSha256Tests(unittest.TestCase):
    EXPECTED_SHA = (
        "3aaf9d509e2128953a912de6056be9bcdaefab108409d40cfa3bce3dc12e07b8"
    )

    def test_committed_sha256_locked(self) -> None:
        self.assertEqual(
            hash_ranking_artifact(COMMITTED_ARTIFACT), self.EXPECTED_SHA
        )

    def test_cli_regenerates_byte_identical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "ranking.json"
            res = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.rl",
                    "train-ranking-policy",
                    "--dataset-id",
                    "v1_bootstrap",
                    "--output",
                    str(out_path),
                ],
                cwd=str(REPO_ROOT),
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertEqual(res.returncode, 0)
            rebuilt = hashlib.sha256(out_path.read_bytes()).hexdigest()
            self.assertEqual(rebuilt, self.EXPECTED_SHA)


# ---------------------------------------------------------------------------
# Trainer numerics
# ---------------------------------------------------------------------------


def _mk_record(
    *,
    attempt_id: str,
    bin_id: str,
    success: bool,
    sig: float,
) -> dict:
    feats = {
        k: 0.0 for k in RANKING_FEATURE_KEYS
    }
    feats["shadow_predicted_success_probability"] = sig
    return {
        "attempt_id": attempt_id,
        "bin_id": bin_id,
        "final_outcome": "succeeded" if success else "failed",
        "rl_state_features": feats,
    }


class TrainPairwiseLogisticRankerTests(unittest.TestCase):
    def test_separable_pairs_drive_weight_positive(self) -> None:
        # Three bins, each with one success at sig=1.0 and one fail at 0.0.
        records = []
        for i in range(3):
            records.append(
                _mk_record(
                    attempt_id=f"b{i}-pos",
                    bin_id=f"b{i}",
                    success=True,
                    sig=1.0,
                )
            )
            records.append(
                _mk_record(
                    attempt_id=f"b{i}-neg",
                    bin_id=f"b{i}",
                    success=False,
                    sig=0.0,
                )
            )
        policy, result = train_pairwise_logistic_ranker(
            records,
            dataset_id="unit",
            dataset_hash="0" * 64,
        )
        self.assertIsInstance(policy, PairwiseLogisticRankingPolicy)
        self.assertIsInstance(result, RankingTrainResult)
        # Weight aligned with the discriminative feature must be > 0.
        idx = RANKING_FEATURE_KEYS.index(
            "shadow_predicted_success_probability"
        )
        self.assertGreater(policy.weights[idx], 0.0)
        self.assertEqual(result.num_groups_with_pairs, 3)
        self.assertGreater(result.num_pairs, 0)


def _fam_record(attempt_id: str, family: str, scene_id: str, *, success: bool, sig: float) -> dict:
    """A record carrying a coarse scene_family_id + a UNIQUE per-episode scene_id (the real sim shape)."""
    feats = {k: 0.0 for k in RANKING_FEATURE_KEYS}
    feats["shadow_predicted_success_probability"] = sig
    return {
        "attempt_id": attempt_id,
        "extra": {"scene_family_id": family, "scene_id": scene_id},
        "final_outcome": "succeeded" if success else "failed",
        "rl_state_features": feats,
    }


class SceneFamilyGroupingTests(unittest.TestCase):
    """scene_family_id grouping forms ranking pairs ACROSS episodes of the same target object, where the
    unique-per-episode scene_id alone yields ZERO pairs (the measured blocker)."""

    def test_group_key_prefers_scene_family_id(self) -> None:
        from src.robot.grasping.rl.train_ranking import _group_key

        rec = {"attempt_id": "a-0", "extra": {"scene_family_id": "ycb:sugar_box", "scene_id": "run-ep0"}}
        self.assertEqual(_group_key(rec), "scene_family_id:ycb:sugar_box")

    def test_group_key_falls_back_to_scene_id(self) -> None:
        from src.robot.grasping.rl.train_ranking import _group_key

        # no scene_family_id -> unchanged (the canonical packs keep grouping by scene_id)
        self.assertEqual(_group_key({"extra": {"scene_id": "run-ep0"}}), "scene_id:run-ep0")

    def test_unique_scene_ids_alone_give_zero_pairs(self) -> None:
        from src.robot.grasping.rl.train_ranking import _build_pairs

        # 4 episodes, UNIQUE scene_id, NO scene_family_id -> groups of 1 -> zero pairs (the measured blocker)
        recs = [
            {"attempt_id": f"a{i}", "extra": {"scene_id": f"r-ep{i}"},
             "final_outcome": "succeeded" if i % 2 else "failed",
             "rl_state_features": {k: 0.0 for k in RANKING_FEATURE_KEYS}}
            for i in range(4)
        ]
        pairs, groups_with_pairs = _build_pairs(recs)
        self.assertEqual(len(pairs), 0)
        self.assertEqual(groups_with_pairs, 0)

    def test_scene_family_forms_pairs_across_episodes(self) -> None:
        # SAME family, distinct scene_ids, mixed success/fail -> pairs FORM
        recs = [
            _fam_record("s0", "ycb:sugar_box", "r-ep0", success=True, sig=1.0),
            _fam_record("s1", "ycb:sugar_box", "r-ep1", success=False, sig=0.0),
            _fam_record("s2", "ycb:sugar_box", "r-ep2", success=True, sig=1.0),
            _fam_record("s3", "ycb:sugar_box", "r-ep3", success=False, sig=0.0),
        ]
        policy, result = train_pairwise_logistic_ranker(recs, dataset_id="unit", dataset_hash="0" * 64)
        self.assertIsInstance(policy, PairwiseLogisticRankingPolicy)
        self.assertEqual(result.num_groups_with_pairs, 1)  # one family
        self.assertGreater(result.num_pairs, 0)            # 2 succ x 2 fail -> 4 pairs


# ---------------------------------------------------------------------------
# ShadowRouter.run_ranking_shadow
# ---------------------------------------------------------------------------


class _StubCandidatePolicy:
    name = "stub_candidate"
    version = 1
    artifact_dataset_id = ""

    def propose(
        self, request: CandidateSelectionRequest
    ) -> CandidateSelection:
        from src.robot.grasping.rl.candidate_policy import (
            ProposedCandidate,
        )

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
        )


class _ExplodingRankingPolicy:
    name = "boom"
    version = 7

    def propose_ranking(self, request):
        raise RuntimeError("kaboom")


class _UnknownIdRankingPolicy:
    name = "rogue"
    version = 1

    def propose_ranking(self, request):
        return RankingSelection(
            policy_id="rogue@1",
            policy_version=1,
            proposals=(
                RankedCandidate(candidate_id="not-in-request", rank=0, score=0.9),
            ),
            regret_top1=True,
            kendall_tau=0.0,
        )


class ShadowRouterRankingTests(unittest.TestCase):
    def _router_with(self, ranker) -> ShadowRouter:
        return ShadowRouter(
            candidate_policy=_StubCandidatePolicy(),
            ranking_policy=ranker,
        )

    def test_happy_path(self) -> None:
        router = self._router_with(_zero_policy())
        tel = router.run_ranking_shadow(
            attempt_id="t",
            candidate_ids=("c0", "c1"),
            per_candidate_features={"c0": {}, "c1": {}},
            deterministic_ranking=("c0", "c1"),
        )
        self.assertFalse(tel.fallback_triggered)
        self.assertEqual(tel.policy_id, "v3_ranking_baseline_v1@1")
        self.assertEqual(tel.policy_version, 1)
        self.assertIn(tel.selection.top1_id, {"c0", "c1"})

    def test_exception_becomes_typed_fallback(self) -> None:
        router = self._router_with(_ExplodingRankingPolicy())
        tel = router.run_ranking_shadow(
            attempt_id="t",
            candidate_ids=("c0",),
            per_candidate_features={"c0": {}},
            deterministic_ranking=("c0",),
        )
        self.assertTrue(tel.fallback_triggered)
        self.assertEqual(
            tel.fallback_reason_code, "ranking_exception:RuntimeError"
        )
        self.assertEqual(tel.router_path, ROUTER_PATH_FALLBACK_BASELINE)
        self.assertEqual(tel.policy_id, "boom")
        self.assertEqual(tel.policy_version, 7)

    def test_unknown_candidate_id_becomes_fallback(self) -> None:
        router = self._router_with(_UnknownIdRankingPolicy())
        tel = router.run_ranking_shadow(
            attempt_id="t",
            candidate_ids=("c0",),
            per_candidate_features={"c0": {}},
            deterministic_ranking=("c0",),
        )
        self.assertTrue(tel.fallback_triggered)
        self.assertTrue(
            tel.fallback_reason_code.startswith("ranking_exception:")
        )

    def test_no_policy_returns_defensive_fallback(self) -> None:
        router = ShadowRouter(candidate_policy=_StubCandidatePolicy())
        tel = router.run_ranking_shadow(
            attempt_id="t",
            candidate_ids=("c0",),
            per_candidate_features={"c0": {}},
            deterministic_ranking=("c0",),
        )
        self.assertTrue(tel.fallback_triggered)
        self.assertEqual(tel.fallback_reason_code, "ranking_exception:NoPolicy")


# ---------------------------------------------------------------------------
# emit_ranking_extras
# ---------------------------------------------------------------------------


class EmitRankingExtrasTests(unittest.TestCase):
    def _build(
        self, *, regret: bool, tau: float, version: int = 1
    ) -> RankingShadowTelemetry:
        sel = RankingSelection(
            policy_id="p@1",
            policy_version=version,
            proposals=(),
            regret_top1=regret,
            kendall_tau=tau,
        )
        return RankingShadowTelemetry(
            router_path="rl_shadow",
            policy_id="p",
            policy_version=version,
            selection=sel,
            regret_top1=regret,
            kendall_tau=tau,
        )

    def test_keys_and_values(self) -> None:
        tel = self._build(regret=True, tau=0.42)
        extras = emit_ranking_extras(telemetry=tel)
        self.assertEqual(
            set(extras.keys()),
            {
                "rl_ranking_policy_id",
                "rl_ranking_artifact_version",
                "rl_ranking_regret_top1",
                "rl_ranking_kendall_tau",
            },
        )
        self.assertEqual(extras["rl_ranking_policy_id"], "p")
        self.assertEqual(extras["rl_ranking_artifact_version"], "v1")
        self.assertTrue(extras["rl_ranking_regret_top1"])
        self.assertAlmostEqual(
            float(extras["rl_ranking_kendall_tau"]), 0.42
        )

    def test_kendall_tau_clamped_to_bounds(self) -> None:
        lo, hi = KENDALL_TAU_BOUNDS
        for raw, expected in ((-99.0, lo), (99.0, hi)):
            tel = self._build(regret=False, tau=raw)
            extras = emit_ranking_extras(telemetry=tel)
            self.assertEqual(
                float(extras["rl_ranking_kendall_tau"]), expected
            )


# ---------------------------------------------------------------------------
# Protocol separation: RankingPolicy must NOT be CandidatePolicy
# ---------------------------------------------------------------------------


class ProtocolSeparationTests(unittest.TestCase):
    def test_pairwise_ranker_is_not_candidate_policy(self) -> None:
        # PairwiseLogisticRankingPolicy has ``propose_ranking`` but
        # not ``propose`` — that is the structural difference between
        # the RankingPolicy Protocol and the CandidatePolicy
        # Protocol.
        self.assertFalse(hasattr(PairwiseLogisticRankingPolicy, "propose"))
        self.assertTrue(
            hasattr(PairwiseLogisticRankingPolicy, "propose_ranking")
        )


# ---------------------------------------------------------------------------
# ShadowRouterTelemetry candidate byte-identity (default ranking_telemetry=None)
# ---------------------------------------------------------------------------


class CandidateByteIdentityTests(unittest.TestCase):
    def test_default_ranking_telemetry_is_none(self) -> None:
        # Build a minimal candidate telemetry without referencing the ranking field.
        from src.robot.grasping.rl.candidate_policy import (
            ProposedCandidate,
        )
        from src.robot.grasping.rl.router import CandidateAgreement

        agreement = CandidateAgreement(
            top1_agree=True, kendall_tau=0.0, pruned_count=0
        )
        tel = ShadowRouterTelemetry(
            router_path="rl_shadow",
            policy_id="p@1",
            policy_version=1,
            selection=CandidateSelection(
                policy_id="p@1",
                policy_version=1,
                proposals=(
                    ProposedCandidate(
                        candidate_id="c0", rank=0, score=0.5, pruned=False
                    ),
                ),
            ),
            mask_summary={},
            agreement=agreement,
        )
        self.assertIsNone(tel.ranking_telemetry)


# ---------------------------------------------------------------------------
# pick_loop wiring smoke (orchestrator field + helper present)
# ---------------------------------------------------------------------------


class PickLoopWiringTests(unittest.TestCase):
    def test_orchestrator_has_shadow_router_field(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        fields = {f.name for f in BinPickingOrchestrator.__dataclass_fields__.values()}
        self.assertIn("shadow_router", fields)
        self.assertIn("_ranking_telemetry", fields)

    def test_orchestrator_has_ranking_helper(self) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        self.assertTrue(
            hasattr(BinPickingOrchestrator, "_maybe_run_ranking_shadow")
        )


# ---------------------------------------------------------------------------
# Telemetry catalog locks
# ---------------------------------------------------------------------------


class TelemetryCatalogRankingTests(unittest.TestCase):
    def test_ranking_fields_registered(self) -> None:
        from src.robot.grasping.replay.telemetry_catalog import (
            EXTRA_TELEMETRY_FIELDS,
        )

        phase_map = {
            name: phase for name, phase, _ in EXTRA_TELEMETRY_FIELDS
        }
        for key in (
            "rl_ranking_policy_id",
            "rl_ranking_artifact_version",
            "rl_ranking_regret_top1",
            "rl_ranking_kendall_tau",
        ):
            self.assertEqual(phase_map.get(key), "rl_ranking")


class BuildCandidateLogTests(unittest.TestCase):
    """The per-candidate log assembler (top-N full feature rows + tail aggregate + behavior id).

    Pure-function tests (the parallel of the candidate BreakdownTests). The orchestrator captures this from the
    ranking-shadow ``features_by_id`` + ``deterministic_ranking`` and ``shadow.py`` splices it into
    ``record.extra['rl_candidate_features']`` / ``['rl_behavior_candidate_id']`` for offline RL + OPE.
    """

    @staticmethod
    def _features(ids: tuple[str, ...]) -> dict[str, dict[str, object]]:
        # geometric_score varies by index so the tail mean is checkable
        return {cid: {"geometric_score": float(i), "feasibility_score": 0.5} for i, cid in enumerate(ids)}

    def test_rows_rank_executed_and_behavior(self) -> None:
        ranking = ("a0_c0", "a0_c1", "a0_c2")
        rows, behavior = _build_candidate_log(ranking, self._features(ranking))
        self.assertEqual(behavior, "a0_c0")
        self.assertEqual([r["candidate_id"] for r in rows], list(ranking))
        self.assertEqual([r["rank"] for r in rows], [0, 1, 2])
        self.assertEqual([r["executed"] for r in rows], [True, False, False])
        self.assertEqual(rows[0]["features"]["geometric_score"], 0.0)
        self.assertEqual(rows[1]["features"]["feasibility_score"], 0.5)

    def test_tail_aggregate_caps_rows(self) -> None:
        ranking = tuple(f"a0_c{i}" for i in range(5))
        rows, behavior = _build_candidate_log(ranking, self._features(ranking), top_n=2)
        self.assertEqual(behavior, "a0_c0")
        self.assertEqual(len(rows), 3)  # 2 full rows + 1 tail-summary row
        tail = rows[-1]
        self.assertEqual(tail["tail_count"], 3)
        self.assertAlmostEqual(tail["tail_mean_geometric_score"], 3.0)  # mean of c2,c3,c4 = (2+3+4)/3
        self.assertNotIn("candidate_id", tail)

    def test_no_tail_when_within_top_n(self) -> None:
        ranking = ("a0_c0", "a0_c1")
        rows, _ = _build_candidate_log(ranking, self._features(ranking), top_n=8)
        self.assertEqual(len(rows), 2)
        self.assertTrue(all("candidate_id" in r for r in rows))

    def test_empty_ranking_is_empty(self) -> None:
        rows, behavior = _build_candidate_log((), {})
        self.assertEqual(rows, [])
        self.assertIsNone(behavior)

    def test_missing_features_default_empty(self) -> None:
        rows, _ = _build_candidate_log(("a0_c0",), {})  # no feature row for the id
        self.assertEqual(rows[0]["features"], {})

    def test_non_numeric_tail_score_is_safe(self) -> None:
        # a malformed geometric_score must not crash the tail aggregate
        feats: dict[str, dict[str, object]] = {"a0_c0": {}, "a0_c1": {"geometric_score": "x"}}
        rows, _ = _build_candidate_log(("a0_c0", "a0_c1"), feats, top_n=1)
        self.assertEqual(rows[-1]["tail_count"], 1)
        self.assertEqual(rows[-1]["tail_mean_geometric_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
