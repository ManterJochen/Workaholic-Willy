"""Ship-readiness tests for the perception-budget (v5) shadow feature.

These close the gaps a ship-readiness audit surfaced for the C perception-budget feature:

* G4  -- the --collect-perception runner -> trainer LABEL round-trip (the STOP/CONTINUE label is re-derived
         SOLELY from ``extra.fusion_latency_ms`` presence; a stamping regression would silently mislabel every
         CONTINUE episode as STOP and corrupt the trained corpus). Pin it against the runner's exact record shape.
* G5  -- a TRAINED, non-degenerate policy fires a NON-fallback proposal (the committed baseline + the sim corpus
         both fall back on every cell because STOP/CONTINUE never co-occupy a views_seen cell; a crafted
         co-occupying corpus proves the learned inference path works) + the rl.perception_artifact_path
         POSITIVE override-load path (only the None/negative path was covered).
* G6  -- promotion of a perception policy runs end-to-end + returns a typed verdict (parity with recovery/sequencing).
* G7  -- a co-occupying perception fixture (>=5 STOP + >=5 CONTINUE in ONE cell) the above two need.
* G8  -- the loader's fail-closed drift guards (actions / onehot_layout / per-action block), only schema_version
         was covered.

All stdlib + isaac-free.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.robot.grasping.rl.perception_budget_policy import (
    PERCEPTION_ACTION_CONTINUE,
    PERCEPTION_ACTION_STOP,
    PerceptionBudgetRequest,
    PerceptionBudgetStateKey,
    load_linucb_perception_budget_policy,
    onehot_index_layout,
)
from src.robot.grasping.rl.train_perception_budget import (
    extract_action_label,
    extract_perception_state,
    train_perception_budget_from_records,
    write_perception_budget_artifact,
)


# --------------------------------------------------------------------------------------------------------
# Shared record builders.
# --------------------------------------------------------------------------------------------------------


def _runner_record(
    *, action: str, occ: float, views: int, cands: int, cycle: float, fusion_ms: float | None,
    mode: str = "easy_canonical", lifted: bool = True,
) -> dict:
    """A record with the EXACT extra shape run_multiview_pick --collect-perception emits.

    fusion_latency_ms is present iff action == 'continue' (the trainer's sole label signal).
    """
    extra: dict = {
        "sim_runner": "multiview_perception",
        "views_seen": views,
        "candidate_count": cands,
        "expected_occlusion_ratio": occ,
        "perception_cost": float(views),
        "cycle_time_s": cycle,
        "perception_action_collect": action,
        "sim_lifted": lifted,
    }
    if action == PERCEPTION_ACTION_CONTINUE:
        extra["fusion_latency_ms"] = fusion_ms if fusion_ms is not None else 900.0
    if not lifted:
        extra["failure_taxonomy_class"] = "slip_after_grasp"
    return {"mode": mode, "final_outcome": "succeeded", "extra": extra}


def _cooccupying_corpus(n_each: int = 8) -> list[dict]:
    """>=5 STOP + >=5 CONTINUE records that ALL land in ONE state cell (views=1, cand=3-5, occ=low, easy).

    STOP and CONTINUE co-occupy the cell (same views_seen bucket) -> after training BOTH actions have >=5
    support on all 4 active features -> the propose gate can fire a LEARNED (non-fallback) action. This is a
    SYNTHETIC test fixture: the real collect logs views_seen=1 for STOP / 2 for CONTINUE (they never co-occupy),
    so the real corpus always falls back -- an honest limitation the ship docs record; this fixture exists only
    to prove the learned inference path itself works.
    """
    recs: list[dict] = []
    for i in range(n_each):
        # STOP: cheaper cycle -> higher reward (budget-optimal here). CONTINUE: longer + the extra-view cost.
        recs.append(_runner_record(action=PERCEPTION_ACTION_STOP, occ=0.1, views=1, cands=4,
                                    cycle=5.0 + 0.05 * i, fusion_ms=None))
        recs.append(_runner_record(action=PERCEPTION_ACTION_CONTINUE, occ=0.1, views=1, cands=4,
                                    cycle=6.5 + 0.05 * i, fusion_ms=900.0 + i))
    return recs


# --------------------------------------------------------------------------------------------------------
# G4 -- collect -> trainer label + state round-trip.
# --------------------------------------------------------------------------------------------------------


class LabelRoundTripTests(unittest.TestCase):
    def test_label_matches_runner_choice_both_actions(self) -> None:
        # The trainer re-derives STOP/CONTINUE from fusion_latency_ms presence; it MUST equal the runner's
        # own logged perception_action_collect for both a STOP and a CONTINUE episode.
        for action in (PERCEPTION_ACTION_STOP, PERCEPTION_ACTION_CONTINUE):
            rec = _runner_record(action=action, occ=0.5, views=(1 if action == "stop" else 2),
                                  cands=3, cycle=7.0, fusion_ms=(None if action == "stop" else 12.0))
            self.assertEqual(extract_action_label(rec), rec["extra"]["perception_action_collect"],
                             f"label mismatch for {action}")

    def test_state_key_buckets_from_runner_record(self) -> None:
        rec = _runner_record(action="continue", occ=0.5, views=2, cands=4, cycle=7.0, fusion_ms=9.0,
                             mode="dense_clutter")
        key = extract_perception_state(rec)
        self.assertEqual(key.views_seen_bucket, "2")
        self.assertEqual(key.last_candidate_count_bucket, "3-5")
        self.assertEqual(key.last_occlusion_bucket, "mid")
        self.assertEqual(key.mode_bucket, "dense")

    def test_continue_without_fusion_latency_would_mislabel(self) -> None:
        # Guard the fragility explicitly: a CONTINUE record that LOST its fusion_latency_ms is mislabeled STOP.
        rec = _runner_record(action="continue", occ=0.1, views=2, cands=3, cycle=7.0, fusion_ms=9.0)
        del rec["extra"]["fusion_latency_ms"]
        self.assertEqual(extract_action_label(rec), PERCEPTION_ACTION_STOP)  # documents the coupling


# --------------------------------------------------------------------------------------------------------
# G7 + G5a -- a trained co-occupying policy fires a NON-fallback proposal.
# --------------------------------------------------------------------------------------------------------


class TrainedNonFallbackTests(unittest.TestCase):
    def test_cooccupying_corpus_trains_non_fallback_cell(self) -> None:
        recs = _cooccupying_corpus(n_each=8)
        policy, result, _artifact = train_perception_budget_from_records(
            recs, training_scope="easy", min_support_threshold=5, dataset_id="cooccupy_test",
        )
        self.assertEqual(result.num_action_stop, 8)
        self.assertEqual(result.num_action_continue, 8)
        # The co-occupied cell (views=1, cand=3-5, occ=low, easy) has >=5 support on BOTH actions -> non-fallback.
        cell = PerceptionBudgetStateKey("1", "3-5", "low", "easy")
        sel = policy.propose_perception_budget(PerceptionBudgetRequest(attempt_id="t", state_key=cell))
        self.assertFalse(sel.used_fallback, "trained co-occupying cell must fire a LEARNED (non-fallback) action")
        self.assertGreaterEqual(sel.support_stop, 5)
        self.assertGreaterEqual(sel.support_continue, 5)
        # And an UNSEEN cell still falls back (the gate is honest about unsupported states).
        unseen = PerceptionBudgetStateKey("3+", "6+", "high", "dense")
        self.assertTrue(
            policy.propose_perception_budget(
                PerceptionBudgetRequest(attempt_id="t", state_key=unseen)
            ).used_fallback
        )


# --------------------------------------------------------------------------------------------------------
# G5b -- rl.perception_artifact_path POSITIVE override-load through maybe_build_shadow_router.
# --------------------------------------------------------------------------------------------------------


class OverrideLoadTests(unittest.TestCase):
    def test_perception_artifact_path_override_loads_trained_policy(self) -> None:
        from src.robot.execution.autonomous_grasp.shadow import maybe_build_shadow_router

        recs = _cooccupying_corpus(n_each=8)
        _policy, _result, artifact = train_perception_budget_from_records(
            recs, training_scope="easy", min_support_threshold=5, dataset_id="override_test",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "trained_perception.json"
            write_perception_budget_artifact(artifact, path)
            cfg = SimpleNamespace(
                rl=SimpleNamespace(mode="rl_shadow", perception_artifact_path=str(path))
            )
            router = maybe_build_shadow_router(cfg)
        self.assertIsNotNone(router)
        assert router is not None
        self.assertIsNotNone(router.perception_policy, "override path must load a perception policy")
        # It is the TRAINED artifact (distinct dataset id), not the committed baseline.
        self.assertEqual(getattr(router.perception_policy, "artifact_dataset_id", None), "override_test")


# --------------------------------------------------------------------------------------------------------
# G6 -- promotion of a perception policy runs end-to-end + returns a typed verdict.
# --------------------------------------------------------------------------------------------------------


class PerceptionPromotionTests(unittest.TestCase):
    def test_promotion_returns_typed_verdict(self) -> None:
        from src.robot.grasping.rl.promotion import (
            POLICY_FAMILY_PERCEPTION,
            evaluate_policy_for_promotion,
        )

        recs = _cooccupying_corpus(n_each=10)
        _policy, _result, artifact = train_perception_budget_from_records(
            recs, training_scope="easy", min_support_threshold=5, dataset_id="promo_test",
        )
        with tempfile.TemporaryDirectory() as d:
            art_path = Path(d) / "trained.json"
            write_perception_budget_artifact(artifact, art_path)
            pack = Path(d) / "pack.jsonl"
            pack.write_text("\n".join(json.dumps(r) for r in recs), encoding="utf-8")
            report = evaluate_policy_for_promotion(
                policy_family=POLICY_FAMILY_PERCEPTION,
                policy_artifact_path=art_path,
                pack_paths=[pack],
                dataset_id="promo_test",
                training_scope="easy",
                seed=1,
            )
        self.assertEqual(report.policy_family, POLICY_FAMILY_PERCEPTION)
        self.assertIn(report.verdict, {"pass", "abstain", "fail"})  # typed, non-crashing


# --------------------------------------------------------------------------------------------------------
# G8 -- loader fail-closed drift guards.
# --------------------------------------------------------------------------------------------------------


class LoaderDriftTests(unittest.TestCase):
    def _valid_artifact_bytes(self) -> dict:
        recs = _cooccupying_corpus(n_each=6)
        _policy, _result, artifact = train_perception_budget_from_records(
            recs, training_scope="easy", min_support_threshold=5, dataset_id="drift_test",
        )
        return dict(artifact)

    def _write_and_expect_raise(self, artifact: dict) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "drift.json"
            p.write_text(json.dumps(artifact), encoding="utf-8")
            with self.assertRaises(Exception):
                load_linucb_perception_budget_policy(p)

    def test_actions_drift_rejected(self) -> None:
        art = self._valid_artifact_bytes()
        art["actions"] = ["continue", "stop"]  # reordered -> drift
        self._write_and_expect_raise(art)

    def test_onehot_layout_drift_rejected(self) -> None:
        art = self._valid_artifact_bytes()
        art["onehot_layout"] = list(onehot_index_layout())[:-1] + ["mode.BOGUS"]
        self._write_and_expect_raise(art)

    def test_missing_per_action_block_rejected(self) -> None:
        art = self._valid_artifact_bytes()
        del art["A_diag_per_action"]["stop"]  # per-action data missing
        self._write_and_expect_raise(art)

    def test_valid_artifact_loads(self) -> None:
        art = self._valid_artifact_bytes()
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "ok.json"
            p.write_text(json.dumps(art), encoding="utf-8")
            pol = load_linucb_perception_budget_policy(p)
        self.assertEqual(pol.artifact_dataset_id, "drift_test")


# --------------------------------------------------------------------------------------------------------
# G3 -- the committed v5 promotion report (parity with v4/v6; structural, not byte-locked).
# --------------------------------------------------------------------------------------------------------


class CommittedV5PromotionReportTests(unittest.TestCase):
    """The committed v5 promotion report must exist (audit-trail parity with v4_sequencing / v6_recovery) and
    honestly record the ABSTAIN on the degenerate baseline. NOT byte-locked: config_hash / generated_at_iso /
    code_commit are environment/time-dependent (the v4/v6 byte-SHA tests are platform-locked determinism gates
    that only pass on the artifact-origin box); a structural assertion is portable regression protection."""

    def _report(self) -> dict:
        repo_root = Path(__file__).resolve().parent.parent
        p = repo_root / "docs" / "baselines" / "rl_policies" / "v5_perception_budget_baseline_v1_promotion_v1.json"
        self.assertTrue(p.exists(), f"missing committed v5 promotion report at {p}")
        return json.loads(p.read_text(encoding="utf-8"))

    def test_committed_v5_report_abstains_honestly(self) -> None:
        d = self._report()
        self.assertEqual(d.get("policy_family"), "v5_perception_budget")
        self.assertEqual(d.get("verdict"), "abstain")
        self.assertIn("degenerate_single_action_policy", d.get("reasons", []))
        self.assertIn("degeneracy_note", d)  # the honest note the degenerate baseline must carry


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
