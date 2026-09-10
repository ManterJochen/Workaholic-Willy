"""R3.1c precise-float gate — pins the V5/V6 ``_score_action`` outputs BIT-EXACTLY against the pre-R3.1c baseline,
so the shared ``rl/_linucb.score_linucb_action`` (the V6-lean form now used by both families) cannot drift even a
ULP from the original per-module forms.

The committed-artifact-SHA determinism tests skipped in EVERY run that had ever happened (an allow-list
waiting on an environment variable nothing in the tree set), so this exact-float assertion was not the local
gate for the LinUCB merge. It was the only one. MEASURED 2026-09-10, with the allow-list retired: 25 of those
27 run and pass here and 2 stand down on a measured drift. This stays, because it is exact and costs nothing.
``assertEqual`` on the float tuples is exact (no tolerance). If this fails, the merge is ABANDONED
(revert to per-module ``_score_action``). The baseline was captured from the ORIGINAL per-module V5/V6 forms on
the same fixed one-hot contexts before the extraction.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from src.robot.grasping.rl.perception_budget_policy import (
    PERCEPTION_ACTIONS,
    PerceptionBudgetStateKey,
    load_linucb_perception_budget_policy,
)
from src.robot.grasping.rl.perception_budget_policy import (
    state_key_to_onehot as v5_onehot,
)
from src.robot.grasping.rl.recovery_policy import (
    DENSE_BUCKET_DENSE,
    LAST_OUTCOME_FAILED,
    RECOVERY_ACTIONS,
    RecoveryStateKey,
    load_linucb_recovery_policy,
)
from src.robot.grasping.rl.recovery_policy import (
    state_key_to_onehot as v6_onehot,
)

_RL_BASELINES = Path(__file__).resolve().parent.parent / "docs" / "baselines" / "rl_policies"

# Exact-float baseline captured from the ORIGINAL per-module V5/V6 _score_action (pre-R3.1c).
_V5_BASELINE = {
    "stop": (0.0, 2.0, 0),
    "continue": (-0.4684230158301042, 0.9469666079207664, 0),
}
_V6_BASELINE = {
    "reobserve": (1.0874580609456608, 2.8222874507291236, 0),
    "re_segment": (0.0, 2.23606797749979, 0),
    "replan_grasp": (0.0, 2.23606797749979, 0),
    "perturb_and_retry": (0.0, 2.23606797749979, 0),
    "abort_recovery": (0.0, 2.23606797749979, 0),
    "next_target": (0.0, 2.23606797749979, 0),
    "nudge_target": (0.0, 2.23606797749979, 0),
    "container_agitate": (0.0, 2.23606797749979, 0),
}


class LinUCBPreciseFloatGateTests(unittest.TestCase):
    def test_v5_score_action_bit_exact(self) -> None:
        pol = load_linucb_perception_budget_policy(
            _RL_BASELINES / "v5_perception_budget_baseline_v1.json"
        )
        onehot = v5_onehot(PerceptionBudgetStateKey("0", "0", "low", "easy"))
        for action in PERCEPTION_ACTIONS:
            with self.subTest(action=action):
                got = pol._score_action(action, onehot)
                self.assertEqual(got, _V5_BASELINE[action])
                self.assertIsInstance(got[2], int)

    def test_v6_score_action_bit_exact(self) -> None:
        pol = load_linucb_recovery_policy(_RL_BASELINES / "v6_recovery_baseline_v1.json")
        onehot = v6_onehot(
            RecoveryStateKey(
                failure_class_bucket="slip_after_grasp",
                attempt_index_bucket="1",
                reobserve_count_bucket="2",
                dense_bucket=DENSE_BUCKET_DENSE,
                last_outcome_bucket=LAST_OUTCOME_FAILED,
            )
        )
        for action in RECOVERY_ACTIONS:
            with self.subTest(action=action):
                got = pol._score_action(action, onehot)
                self.assertEqual(got, _V6_BASELINE[action])
                self.assertIsInstance(got[2], int)


if __name__ == "__main__":
    unittest.main()
