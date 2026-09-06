"""Tier-0 online-state architecture — accumulator, apply, integrity, tiers, promotion refusal.

Pure-stdlib default (numpy tier is an exact cross-check; torch tier is a deferred honest stub).
No Isaac, no GPU, no training — the accumulation math + honesty contract only.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.online_state import (
    ONLINE_TIER_NUMPY,
    ONLINE_TIER_STDLIB,
    OnlineStateError,
    StdlibLinUCBOnlineAccumulator,
    apply_online_state,
    build_online_state_stamp,
    is_online_mutated_artifact,
    make_online_accumulator,
    verify_online_state,
)

_ACTIONS = ("a", "b", "c")
_DIM = 4


def _acc(tier: str = ONLINE_TIER_STDLIB):
    return make_online_accumulator(tier, dim=_DIM, actions=_ACTIONS)


class AccumulatorTests(unittest.TestCase):
    def test_fresh_is_not_mutated(self) -> None:
        acc = _acc()
        self.assertEqual(acc.n_updates, 0)
        state = acc.snapshot()
        self.assertFalse(state.online_mutated)
        self.assertTrue(verify_online_state(state))

    def test_observe_updates_diagonal_linucb_deltas(self) -> None:
        acc = _acc()
        acc.observe(onehot=[1, 0, 1, 0], action="b", reward=2.0)
        state = acc.snapshot()
        self.assertEqual(state.n_updates, 1)
        self.assertEqual(state.high_water_mark, 1)
        self.assertTrue(state.online_mutated)
        # chosen action "b": A += x, b += r*x, support += x on active features {0, 2}
        self.assertEqual(state.delta_A_diag_per_action["b"], (1.0, 0.0, 1.0, 0.0))
        self.assertEqual(state.delta_b_per_action["b"], (2.0, 0.0, 2.0, 0.0))
        self.assertEqual(state.delta_support_per_action["b"], (1, 0, 1, 0))
        # untouched actions stay zero
        self.assertEqual(state.delta_A_diag_per_action["a"], (0.0, 0.0, 0.0, 0.0))

    def test_high_water_mark_is_monotonic(self) -> None:
        acc = _acc()
        for _ in range(3):
            acc.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        self.assertEqual(acc.snapshot().high_water_mark, 3)

    def test_rejects_bad_action_dim_and_values(self) -> None:
        acc = _acc()
        with self.assertRaises(OnlineStateError):
            acc.observe(onehot=[1, 0, 0, 0], action="zzz", reward=1.0)
        with self.assertRaises(OnlineStateError):
            acc.observe(onehot=[1, 0, 0], action="a", reward=1.0)
        with self.assertRaises(OnlineStateError):
            acc.observe(onehot=[2, 0, 0, 0], action="a", reward=1.0)


class IntegrityTests(unittest.TestCase):
    def test_chain_is_order_sensitive(self) -> None:
        acc1 = _acc()
        acc1.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        acc1.observe(onehot=[0, 1, 0, 0], action="b", reward=0.0)
        acc2 = _acc()
        acc2.observe(onehot=[0, 1, 0, 0], action="b", reward=0.0)
        acc2.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        self.assertNotEqual(
            acc1.snapshot().update_chain_hash, acc2.snapshot().update_chain_hash
        )

    def test_tamper_fails_verification_and_apply(self) -> None:
        acc = _acc()
        acc.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        state = acc.snapshot()
        self.assertTrue(verify_online_state(state))
        # mutate a delta value in place (the dict is mutable even on a frozen dataclass)
        state.delta_b_per_action["a"] = (999.0, 0.0, 0.0, 0.0)
        self.assertFalse(verify_online_state(state))
        base = ({a: (1.0,) * _DIM for a in _ACTIONS},
                {a: (0.0,) * _DIM for a in _ACTIONS},
                {a: (0,) * _DIM for a in _ACTIONS})
        with self.assertRaises(OnlineStateError):
            apply_online_state(
                A_diag_per_action=base[0],
                b_per_action=base[1],
                feature_support_per_action=base[2],
                state=state,
            )


class ApplyTests(unittest.TestCase):
    def test_apply_adds_deltas_onto_frozen(self) -> None:
        acc = _acc()
        acc.observe(onehot=[1, 0, 1, 0], action="a", reward=3.0)
        state = acc.snapshot()
        A = {a: (1.0, 1.0, 1.0, 1.0) for a in _ACTIONS}
        b = {a: (0.0, 0.0, 0.0, 0.0) for a in _ACTIONS}
        sup = {a: (0, 0, 0, 0) for a in _ACTIONS}
        newA, newb, newsup = apply_online_state(
            A_diag_per_action=A, b_per_action=b, feature_support_per_action=sup, state=state
        )
        self.assertEqual(newA["a"], (2.0, 1.0, 2.0, 1.0))
        self.assertEqual(newb["a"], (3.0, 0.0, 3.0, 0.0))
        self.assertEqual(newsup["a"], (1, 0, 1, 0))
        # non-observed action unchanged
        self.assertEqual(newA["b"], (1.0, 1.0, 1.0, 1.0))
        # inputs not mutated
        self.assertEqual(A["a"], (1.0, 1.0, 1.0, 1.0))

    def test_apply_rejects_action_mismatch(self) -> None:
        acc = _acc()
        acc.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        state = acc.snapshot()
        with self.assertRaises(OnlineStateError):
            apply_online_state(
                A_diag_per_action={"x": (1.0,) * _DIM},
                b_per_action={"x": (0.0,) * _DIM},
                feature_support_per_action={"x": (0,) * _DIM},
                state=state,
            )


class StampTests(unittest.TestCase):
    def test_pristine_stamp(self) -> None:
        stamp = build_online_state_stamp(_acc().snapshot())
        self.assertEqual(stamp, {"online_mutated": False, "n_updates": 0, "still_shadow": True})
        self.assertFalse(is_online_mutated_artifact({"online_state": stamp}))

    def test_mutated_stamp(self) -> None:
        acc = _acc()
        acc.observe(onehot=[1, 0, 0, 0], action="a", reward=1.0)
        state = acc.snapshot()
        stamp = build_online_state_stamp(state)
        self.assertTrue(stamp["online_mutated"])
        self.assertTrue(stamp["still_shadow"])
        self.assertEqual(stamp["n_updates"], 1)
        self.assertEqual(stamp["last_update_hash"], state.update_chain_hash)
        self.assertEqual(stamp["content_hash"], state.content_hash)
        self.assertTrue(is_online_mutated_artifact({"online_state": stamp}))

    def test_none_and_missing_stamp(self) -> None:
        self.assertFalse(build_online_state_stamp(None)["online_mutated"])
        self.assertFalse(is_online_mutated_artifact({}))


class TierTests(unittest.TestCase):
    def test_numpy_tier_is_exact_mirror_of_stdlib(self) -> None:
        try:
            import numpy  # noqa: F401
        except ImportError:  # pragma: no cover
            self.skipTest("numpy not installed")
        stream = [
            ([1, 0, 1, 0], "a", 1.0),
            ([0, 1, 0, 1], "b", 0.0),
            ([1, 1, 0, 0], "c", 2.5),
            ([1, 0, 1, 0], "a", -1.0),
        ]
        s = _acc(ONLINE_TIER_STDLIB)
        n = _acc(ONLINE_TIER_NUMPY)
        self.assertIsInstance(s, StdlibLinUCBOnlineAccumulator)
        for oh, act, r in stream:
            s.observe(onehot=oh, action=act, reward=r)
            n.observe(onehot=oh, action=act, reward=r)
        ss, ns = s.snapshot(), n.snapshot()
        self.assertEqual(ss.content_hash, ns.content_hash)
        self.assertEqual(ss.update_chain_hash, ns.update_chain_hash)
        self.assertEqual(ss.delta_A_diag_per_action, ns.delta_A_diag_per_action)
        self.assertEqual(ss.delta_b_per_action, ns.delta_b_per_action)
        self.assertEqual(ss.delta_support_per_action, ns.delta_support_per_action)

    def test_torch_tier_is_honest_deferred_stub(self) -> None:
        with self.assertRaises(OnlineStateError) as ctx:
            make_online_accumulator("torch", dim=_DIM, actions=_ACTIONS)
        self.assertIn("not built yet", str(ctx.exception))

    def test_unknown_tier_raises(self) -> None:
        with self.assertRaises(OnlineStateError):
            make_online_accumulator("banana", dim=_DIM, actions=_ACTIONS)


class PromotionRefusalTests(unittest.TestCase):
    def test_promotion_refuses_online_mutated_artifact(self) -> None:
        from src.robot.grasping.rl.promotion import (
            POLICY_FAMILY_RECOVERY,
            PromotionInputError,
            evaluate_policy_for_promotion,
        )

        repo_root = Path(__file__).resolve().parents[1]
        committed = (
            repo_root / "docs" / "baselines" / "rl_policies" / "v6_recovery_baseline_v1.json"
        )
        blob = json.loads(committed.read_text(encoding="utf-8"))
        # inject a mutated online_state stamp
        blob["online_state"] = {"online_mutated": True, "n_updates": 7, "still_shadow": True}
        with tempfile.TemporaryDirectory() as tmp:
            art = Path(tmp) / "mutated.json"
            art.write_text(json.dumps(blob), encoding="utf-8")
            pack = repo_root / "tests" / "data" / "replay" / "replay_dense_canonical_v1.jsonl"
            with self.assertRaises(PromotionInputError) as ctx:
                evaluate_policy_for_promotion(
                    policy_family=POLICY_FAMILY_RECOVERY,
                    policy_artifact_path=art,
                    pack_paths=[pack],
                    dataset_id="unit",
                    training_scope="dense",
                    seed=1,
                )
            self.assertIn("online-mutated", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
