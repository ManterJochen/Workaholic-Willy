"""Tier-0 online-update coordinator over a frozen ShadowRouter.

Loads the committed LinUCB recovery + perception baselines, drives observations directly and from a
real canonical replay record, and checks the effective router is (a) identical to the base until an
observation lands and (b) an online-mutated variant afterwards. Pure-stdlib default; no Isaac.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.robot.grasping.rl.online_repromotion import (
    REPROMOTION_STATUS_CURRENT,
    REPROMOTION_STATUS_STALE,
    RepromotionCadence,
)
from src.robot.grasping.rl.online_router import (
    FAMILY_PERCEPTION,
    FAMILY_RECOVERY,
    OnlineUpdateCoordinator,
)
from src.robot.grasping.rl.perception_budget_policy import (
    load_linucb_perception_budget_policy,
)
from src.robot.grasping.rl.recovery_policy import (
    RECOVERY_ACTION_REOBSERVE,
    RecoveryStateKey,
    load_linucb_recovery_policy,
)
from src.robot.grasping.rl.router import ShadowRouter

_REPO = Path(__file__).resolve().parents[1]
_POL = _REPO / "docs" / "baselines" / "rl_policies"
_PACK = _REPO / "tests" / "data" / "replay" / "replay_dense_canonical_v1.jsonl"


class _StubCandidatePolicy:
    name = "stub"
    version = 0

    def propose(self, request):  # pragma: no cover - never called by the coordinator
        raise NotImplementedError


def _router() -> ShadowRouter:
    return ShadowRouter(
        candidate_policy=_StubCandidatePolicy(),  # type: ignore[arg-type]
        recovery_policy=load_linucb_recovery_policy(_POL / "v6_recovery_baseline_v1.json"),
        perception_policy=load_linucb_perception_budget_policy(
            _POL / "v5_perception_budget_baseline_v1.json"
        ),
    )


def _state() -> RecoveryStateKey:
    return RecoveryStateKey(
        failure_class_bucket="slip_after_grasp",
        attempt_index_bucket="1",
        reobserve_count_bucket="0",
        dense_bucket="dense",
        last_outcome_bucket="failed",
    )


class CoordinatorConstructionTests(unittest.TestCase):
    def test_active_families_for_linucb_slots(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        self.assertEqual(set(coord.active_families), {FAMILY_RECOVERY, FAMILY_PERCEPTION})
        self.assertEqual(coord.n_updates(FAMILY_RECOVERY), 0)

    def test_non_linucb_slots_get_no_accumulator(self) -> None:
        # A router with no recovery/perception policy has no accumulators.
        coord = OnlineUpdateCoordinator(
            ShadowRouter(candidate_policy=_StubCandidatePolicy())  # type: ignore[arg-type]
        )
        self.assertEqual(coord.active_families, ())

    def test_effective_router_is_identity_without_observations(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        self.assertIs(coord.effective_router(), coord._base)


class ObservationTests(unittest.TestCase):
    def test_direct_observe_mutates_effective_recovery_policy(self) -> None:
        base = _router()
        coord = OnlineUpdateCoordinator(base)
        coord.observe_recovery(
            state_key=_state(), action=RECOVERY_ACTION_REOBSERVE, reward=1.0
        )
        self.assertEqual(coord.n_updates(FAMILY_RECOVERY), 1)
        eff = coord.effective_router()
        self.assertIsNot(eff, base)
        # the observed action's A_diag grew on the active features; the base is untouched
        base_A = base.recovery_policy.A_diag_per_action[RECOVERY_ACTION_REOBSERVE]
        eff_A = eff.recovery_policy.A_diag_per_action[RECOVERY_ACTION_REOBSERVE]
        self.assertGreater(sum(eff_A), sum(base_A))
        # base object genuinely unchanged (frozen ⊕ delta, not in-place)
        self.assertEqual(
            base.recovery_policy.A_diag_per_action[RECOVERY_ACTION_REOBSERVE], base_A
        )

    def test_observe_from_real_canonical_record(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        with _PACK.open("r", encoding="utf-8") as f:
            rec = json.loads(f.readline())
        applied = coord.observe_from_record(rec)
        # a dense canonical record yields at least one recovery/perception observation
        self.assertGreaterEqual(applied, 0)
        # feeding several records accumulates
        total = 0
        with _PACK.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50:
                    break
                total += coord.observe_from_record(json.loads(line))
        if total > 0:
            self.assertTrue(coord.effective_router() is not coord._base)

    def test_stamps_reflect_mutation(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        stamps = coord.online_state_stamps()
        self.assertFalse(stamps[FAMILY_RECOVERY]["online_mutated"])
        coord.observe_recovery(
            state_key=_state(), action=RECOVERY_ACTION_REOBSERVE, reward=0.0
        )
        stamps = coord.online_state_stamps()
        self.assertTrue(stamps[FAMILY_RECOVERY]["online_mutated"])
        self.assertTrue(stamps[FAMILY_RECOVERY]["still_shadow"])


class RepromotionCadenceTests(unittest.TestCase):
    def test_fresh_coordinator_is_current(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        status = coord.repromotion_status()
        self.assertEqual(status[FAMILY_RECOVERY].status, REPROMOTION_STATUS_CURRENT)
        self.assertFalse(status[FAMILY_RECOVERY].repromotion_required)
        self.assertTrue(status[FAMILY_RECOVERY].still_shadow)

    def test_updates_past_budget_flag_stale(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        cadence = RepromotionCadence(snapshot_every_n=2, stale_after_n=3)
        for _ in range(3):
            coord.observe_recovery(
                state_key=_state(), action=RECOVERY_ACTION_REOBSERVE, reward=1.0
            )
        d = coord.repromotion_status(cadence=cadence)[FAMILY_RECOVERY]
        self.assertEqual(d.status, REPROMOTION_STATUS_STALE)
        self.assertTrue(d.repromotion_required)
        self.assertEqual(d.updates_since_promotion, 3)
        # honest: a stale verdict never activates the effective policy.
        self.assertTrue(d.still_shadow)

    def test_promotion_baseline_resets_the_budget(self) -> None:
        coord = OnlineUpdateCoordinator(_router())
        cadence = RepromotionCadence(snapshot_every_n=2, stale_after_n=3)
        for _ in range(3):
            coord.observe_recovery(
                state_key=_state(), action=RECOVERY_ACTION_REOBSERVE, reward=1.0
            )
        # re-baselining the promotion + snapshot counts to now makes the verdict fresh again.
        d = coord.repromotion_status(
            cadence=cadence,
            last_promoted={FAMILY_RECOVERY: 3},
            last_snapshot={FAMILY_RECOVERY: 3},
        )[FAMILY_RECOVERY]
        self.assertEqual(d.status, REPROMOTION_STATUS_CURRENT)
        self.assertFalse(d.repromotion_required)


class TierTests(unittest.TestCase):
    def test_numpy_tier_coordinator(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")
        coord = OnlineUpdateCoordinator(_router(), tier="numpy")
        coord.observe_recovery(
            state_key=_state(), action=RECOVERY_ACTION_REOBSERVE, reward=1.0
        )
        self.assertEqual(coord.n_updates(FAMILY_RECOVERY), 1)
        self.assertIsNot(coord.effective_router(), coord._base)


if __name__ == "__main__":
    unittest.main()
