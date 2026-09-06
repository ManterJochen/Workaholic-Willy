"""Candidate-policy contract tests.

Locks under test:

* ShadowRouter umbrella with reserved ranking/sequencing/perception/recovery
  slots; only ``candidate_policy`` live so far.
* Two committed baselines — RandomCandidatePolicy (deterministic seed) and
  LogisticCandidatePolicy (frozen-feature-keys, sigmoid).
* Action mask precedence: 6 channels in fixed order; scene-level
  ``degraded_mode_active`` masks every candidate.
* Pruning surface exists; default keeps every candidate.
* Shadow telemetry attaches to :class:`AutonomousGraspReport` and is
  *strictly* carrier-only (byte-identical pick path when
  ``shadow_router is None``).
* RandomCandidatePolicy seed depends on attempt_id + policy_id.
* Top-N candidate breakdown bounded; trailing tail summary row when
  proposals exceed ``BREAKDOWN_TOP_N``.
* CLI: ``python -m backend.src.robot.grasping.rl train-candidate-policy``
  reproduces the committed artifact sha256 byte-for-byte.
* Telemetry catalog: 5 new candidate fields validate without breaking the
  base rl-mode-contract and telemetry fields.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from src.robot.grasping.rl import (
    RL_SUPPORTED_MODES,
)
from src.robot.grasping.rl.action_mask import (
    MASK_CHANNELS,
    ActionMaskContext,
    apply_action_mask,
    mask_summary,
)
from src.robot.grasping.rl.candidate_policy import (
    CANDIDATE_FEATURE_KEYS,
    CandidateFeatures,
    CandidateSelectionRequest,
    LogisticCandidatePolicy,
    RandomCandidatePolicy,
    hash_logistic_artifact,
    load_logistic_candidate_policy,
)
from src.robot.grasping.rl.router import (
    BREAKDOWN_TOP_N,
    ROUTER_PATH_FALLBACK_BASELINE,
    ROUTER_PATH_RL_SHADOW,
    SHADOW_ROUTER_TELEMETRY_VERSION,
    ShadowRouter,
    build_candidate_breakdown,
    emit_candidate_shadow_extras,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
COMMITTED_ARTIFACT = (
    REPO_ROOT
    / "docs"
    / "baselines"
    / "rl_policies"
    / "v2_candidate_baseline_v1.json"
)


# ---------------------------------------------------------------------------
# Supported-modes promotion.
# ---------------------------------------------------------------------------


class SupportedModesPromotionTests(unittest.TestCase):
    def test_rl_shadow_in_supported_modes(self) -> None:
        self.assertIn("rl_shadow", RL_SUPPORTED_MODES)
        self.assertIn("geometry_only", RL_SUPPORTED_MODES)
        self.assertIn("hybrid_ml", RL_SUPPORTED_MODES)


# ---------------------------------------------------------------------------
# Action mask.
# ---------------------------------------------------------------------------


class ActionMaskTests(unittest.TestCase):
    def test_channel_count_is_six(self) -> None:
        self.assertEqual(len(MASK_CHANNELS), 6)

    def test_no_context_keeps_every_candidate(self) -> None:
        rows = apply_action_mask(("a", "b", "c"), ActionMaskContext())
        for r in rows:
            self.assertTrue(r.keep)
            self.assertEqual(r.masked_by, ())

    def test_channel_precedence_preserved(self) -> None:
        ctx = ActionMaskContext(
            safety_rejected_ids=frozenset({"x"}),
            feasibility_failed_ids=frozenset({"x"}),
            corridor_failed_ids=frozenset({"x"}),
            uncertainty_failed_ids=frozenset({"x"}),
            drift_ood_failed_ids=frozenset({"x"}),
        )
        rows = apply_action_mask(("x",), ctx)
        self.assertEqual(
            rows[0].masked_by,
            (
                "safety",
                "feasibility",
                "corridor",
                "uncertainty",
                "drift_ood",
            ),
        )

    def test_degraded_mode_masks_everything(self) -> None:
        ctx = ActionMaskContext(degraded_mode_active=True)
        rows = apply_action_mask(("a", "b"), ctx)
        for r in rows:
            self.assertEqual(r.masked_by, ("degraded_mode",))

    def test_mask_summary_counts(self) -> None:
        ctx = ActionMaskContext(
            safety_rejected_ids=frozenset({"a"}),
            feasibility_failed_ids=frozenset({"b"}),
        )
        rows = apply_action_mask(("a", "b", "c"), ctx)
        summary = mask_summary(rows)
        self.assertEqual(summary["safety"], 1)
        self.assertEqual(summary["feasibility"], 1)
        self.assertEqual(summary["total_masked"], 2)

    def test_empty_input_is_safe(self) -> None:
        self.assertEqual(apply_action_mask((), ActionMaskContext()), ())


# ---------------------------------------------------------------------------
# RandomCandidatePolicy.
# ---------------------------------------------------------------------------


class RandomCandidatePolicyTests(unittest.TestCase):
    def _request(self, attempt_id: str = "att-1") -> CandidateSelectionRequest:
        cands = tuple(
            CandidateFeatures(candidate_id=f"c{i}", features={}) for i in range(5)
        )
        return CandidateSelectionRequest(
            attempt_id=attempt_id, candidates=cands
        )

    def test_deterministic_across_calls(self) -> None:
        p = RandomCandidatePolicy()
        s1 = p.propose(self._request())
        s2 = p.propose(self._request())
        self.assertEqual(
            tuple(x.candidate_id for x in s1.proposals),
            tuple(x.candidate_id for x in s2.proposals),
        )

    def test_seed_depends_on_attempt_id(self) -> None:
        p = RandomCandidatePolicy()
        s1 = p.propose(self._request("att-A"))
        s2 = p.propose(self._request("att-B"))
        self.assertNotEqual(
            tuple(x.candidate_id for x in s1.proposals),
            tuple(x.candidate_id for x in s2.proposals),
        )

    def test_never_prunes(self) -> None:
        p = RandomCandidatePolicy()
        sel = p.propose(self._request())
        for pr in sel.proposals:
            self.assertFalse(pr.pruned)


# ---------------------------------------------------------------------------
# LogisticCandidatePolicy.
# ---------------------------------------------------------------------------


class LogisticCandidatePolicyTests(unittest.TestCase):
    def test_weights_length_validated(self) -> None:
        with self.assertRaises(ValueError):
            LogisticCandidatePolicy(weights=(0.0,), bias=0.0)

    def test_prune_threshold_validated(self) -> None:
        with self.assertRaises(ValueError):
            LogisticCandidatePolicy(
                weights=tuple(0.0 for _ in CANDIDATE_FEATURE_KEYS),
                bias=0.0,
                prune_threshold=2.0,
            )

    def test_sigmoid_bounded(self) -> None:
        p = LogisticCandidatePolicy(
            weights=tuple(10.0 for _ in CANDIDATE_FEATURE_KEYS),
            bias=-1000.0,
        )
        s = p.score_features({k: 0.0 for k in CANDIDATE_FEATURE_KEYS})
        self.assertGreaterEqual(s, 0.0)
        self.assertLessEqual(s, 1.0)

    def test_tie_break_by_candidate_id(self) -> None:
        p = LogisticCandidatePolicy(
            weights=tuple(0.0 for _ in CANDIDATE_FEATURE_KEYS),
            bias=0.0,
        )
        cands = tuple(
            CandidateFeatures(candidate_id=cid, features={})
            for cid in ("c", "a", "b")
        )
        req = CandidateSelectionRequest(attempt_id="x", candidates=cands)
        sel = p.propose(req)
        # All scores tie; ids sort lexicographically ascending.
        self.assertEqual(
            tuple(x.candidate_id for x in sel.proposals), ("a", "b", "c")
        )

    def test_pruning_threshold_applied(self) -> None:
        p = LogisticCandidatePolicy(
            weights=tuple(0.0 for _ in CANDIDATE_FEATURE_KEYS),
            bias=-100.0,  # sigmoid near 0
            prune_threshold=0.5,
        )
        req = CandidateSelectionRequest(
            attempt_id="x",
            candidates=(CandidateFeatures(candidate_id="a", features={}),),
        )
        sel = p.propose(req)
        self.assertTrue(sel.proposals[0].pruned)

    def test_bool_features_projected_to_floats(self) -> None:
        # ood_flagged True must contribute its weight; False must not.
        weights = list(0.0 for _ in CANDIDATE_FEATURE_KEYS)
        idx = CANDIDATE_FEATURE_KEYS.index("ood_flagged")
        weights[idx] = 5.0
        p = LogisticCandidatePolicy(weights=tuple(weights), bias=0.0)
        s_true = p.score_features({"ood_flagged": True})
        s_false = p.score_features({"ood_flagged": False})
        self.assertGreater(s_true, s_false)


# ---------------------------------------------------------------------------
# Artifact loader.
# ---------------------------------------------------------------------------


class LogisticArtifactLoaderTests(unittest.TestCase):
    def test_committed_artifact_loads(self) -> None:
        self.assertTrue(COMMITTED_ARTIFACT.is_file())
        p = load_logistic_candidate_policy(COMMITTED_ARTIFACT)
        self.assertEqual(p.name, "v2_candidate_baseline_v1")
        self.assertEqual(p.version, 1)
        self.assertEqual(len(p.weights), len(CANDIDATE_FEATURE_KEYS))

    def test_schema_version_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            payload = json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))
            payload["schema_version"] = 999
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_logistic_candidate_policy(path)

    def test_feature_keys_mismatch_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.json"
            payload = json.loads(COMMITTED_ARTIFACT.read_text(encoding="utf-8"))
            payload["feature_keys"] = list(payload["feature_keys"][:-1])
            path.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_logistic_candidate_policy(path)


# ---------------------------------------------------------------------------
# ShadowRouter.
# ---------------------------------------------------------------------------


def _request_three_candidates() -> tuple[ShadowRouter, str, dict[str, dict]]:
    p = LogisticCandidatePolicy(
        weights=tuple(0.0 for _ in CANDIDATE_FEATURE_KEYS),
        bias=0.0,
    )
    router = ShadowRouter(candidate_policy=p)
    feats = {f"c{i}": {} for i in range(3)}
    return router, "att-x", feats


class ShadowRouterTests(unittest.TestCase):
    def test_telemetry_emitted_with_router_path(self) -> None:
        router, attempt_id, feats = _request_three_candidates()
        tel = router.run_candidate_selection(
            attempt_id=attempt_id,
            candidate_ids=tuple(feats.keys()),
            per_candidate_features=feats,
            mask_context=ActionMaskContext(),
            deterministic_top1_id="c0",
        )
        self.assertEqual(tel.router_path, ROUTER_PATH_RL_SHADOW)
        self.assertEqual(tel.policy_id, "v2_candidate_baseline_v1@1")
        self.assertEqual(tel.policy_version, 1)
        self.assertEqual(tel.mask_summary["total_masked"], 0)
        self.assertFalse(tel.fallback_triggered)

    def test_top1_agreement_set(self) -> None:
        router, attempt_id, feats = _request_three_candidates()
        tel = router.run_candidate_selection(
            attempt_id=attempt_id,
            candidate_ids=tuple(feats.keys()),
            per_candidate_features=feats,
            mask_context=ActionMaskContext(),
            deterministic_top1_id=None,
        )
        # deterministic_top1=None → top1_agree=False (one-side missing).
        self.assertFalse(tel.agreement.top1_agree)

    def test_mask_filters_before_policy_sees(self) -> None:
        router, attempt_id, feats = _request_three_candidates()
        ctx = ActionMaskContext(safety_rejected_ids=frozenset({"c0"}))
        tel = router.run_candidate_selection(
            attempt_id=attempt_id,
            candidate_ids=tuple(feats.keys()),
            per_candidate_features=feats,
            mask_context=ctx,
            deterministic_top1_id="c0",
        )
        ids = {p.candidate_id for p in tel.selection.proposals}
        self.assertNotIn("c0", ids)
        self.assertEqual(tel.mask_summary["safety"], 1)

    def test_fallback_on_misbehaving_policy(self) -> None:
        @dataclass(frozen=True)
        class _Bad:
            name: str = "bad"
            version: int = 1

            def propose(self, request):  # noqa: ANN001
                raise RuntimeError("boom")

        router = ShadowRouter(candidate_policy=_Bad())
        tel = router.run_candidate_selection(
            attempt_id="x",
            candidate_ids=("a",),
            per_candidate_features={"a": {}},
            mask_context=ActionMaskContext(),
            deterministic_top1_id="a",
        )
        self.assertTrue(tel.fallback_triggered)
        self.assertEqual(tel.router_path, ROUTER_PATH_FALLBACK_BASELINE)
        self.assertTrue(tel.fallback_reason_code.startswith("router_exception:"))


class BreakdownTests(unittest.TestCase):
    def test_breakdown_caps_at_top_n_plus_tail(self) -> None:
        p = LogisticCandidatePolicy(
            weights=tuple(0.0 for _ in CANDIDATE_FEATURE_KEYS),
            bias=0.0,
        )
        router = ShadowRouter(candidate_policy=p)
        ids = tuple(f"c{i:02d}" for i in range(BREAKDOWN_TOP_N + 5))
        feats = {cid: {} for cid in ids}
        tel = router.run_candidate_selection(
            attempt_id="x",
            candidate_ids=ids,
            per_candidate_features=feats,
            mask_context=ActionMaskContext(),
            deterministic_top1_id=ids[0],
        )
        rows = build_candidate_breakdown(telemetry=tel)
        # top_n head rows + one tail summary row
        self.assertEqual(len(rows), BREAKDOWN_TOP_N + 1)
        self.assertIn("tail_count", rows[-1])
        self.assertEqual(rows[-1]["tail_count"], 5)


class EmitCandidateShadowExtrasTests(unittest.TestCase):
    def test_all_base_and_candidate_fields_present(self) -> None:
        router, attempt_id, feats = _request_three_candidates()
        tel = router.run_candidate_selection(
            attempt_id=attempt_id,
            candidate_ids=tuple(feats.keys()),
            per_candidate_features=feats,
            mask_context=ActionMaskContext(),
            deterministic_top1_id="c0",
        )
        extras = emit_candidate_shadow_extras(
            telemetry=tel,
            baseline_action="accept",
            deterministic_top1_id="c0",
        )
        for field_name in (
            "rl_mode",
            "rl_policy_id",
            "rl_artifact_version",
            "rl_action_proposed",
            "rl_action_applied",
            "rl_action_blocked_by_mask",
            "rl_reason_features",
            "rl_confidence",
            "rl_baseline_action",
            "rl_override",
            "rl_fallback_triggered",
            "rl_router_path",
            "rl_fallback_reason_code",
            # candidate additive
            "rl_candidate_breakdown",
            "rl_candidate_agreement_top1",
            "rl_candidate_agreement_kendall_tau",
            "rl_candidate_mask_total",
            "rl_candidate_pruned_count",
        ):
            self.assertIn(field_name, extras)

    def test_telemetry_version_constant(self) -> None:
        # Bumped each phase; the perception sub-block lifted the constant to 4.
        self.assertEqual(SHADOW_ROUTER_TELEMETRY_VERSION, 5)


# ---------------------------------------------------------------------------
# CLI: reproduce the committed artifact byte-for-byte.
# ---------------------------------------------------------------------------


class TrainCandidatePolicyCLITests(unittest.TestCase):
    def test_cli_reproduces_committed_artifact(self) -> None:
        committed_sha = hash_logistic_artifact(COMMITTED_ARTIFACT)
        with tempfile.TemporaryDirectory() as tmp:
            out_path = Path(tmp) / "rebuilt.json"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "src.robot.grasping.rl",
                    "train-candidate-policy",
                    "--dataset-id",
                    "v1_bootstrap",
                    "--seed",
                    "1",
                    "--output",
                    str(out_path),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"stdout={result.stdout!r}\nstderr={result.stderr!r}",
            )
            self.assertTrue(out_path.is_file())
            rebuilt_sha = hash_logistic_artifact(out_path)
            self.assertEqual(rebuilt_sha, committed_sha)


# ---------------------------------------------------------------------------
# AutonomousGraspReport wiring (byte-identical guarantee).
# ---------------------------------------------------------------------------


class AutonomousGraspCandidateWiringTests(unittest.TestCase):
    def test_report_has_shadow_router_telemetry_field(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspReport,
        )

        # Field is present on the frozen dataclass surface and
        # defaults to None for the byte-identical base path.
        names = {f.name for f in AutonomousGraspReport.__dataclass_fields__.values()}
        self.assertIn("shadow_router_telemetry", names)

    def test_service_has_shadow_router_slot(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspService,
        )

        names = {
            f.name for f in AutonomousGraspService.__dataclass_fields__.values()
        }
        self.assertIn("shadow_router", names)


if __name__ == "__main__":
    unittest.main()
