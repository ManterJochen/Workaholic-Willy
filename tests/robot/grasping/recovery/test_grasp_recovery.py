"""The recovery policy: the operator-bounded configuration both the loop and the push read.

* :class:`SceneRecoveryPolicy` validates its allow-list (no duplicate, no ``NONE``, members only), its budgets and its
  modes, and :meth:`~SceneRecoveryPolicy.permits` says no while it is off;
* :func:`push_permitted` is the gate the service hands a pick attempt: the push (``NUDGE_TARGET``) runs only in
  ``dense_clutter``, with the policy on, the mode applied, the action allowed and a budget left; the automatic push
  box bounds where the part lands (the owner, 2026-10-01: the push needs no fixture);
* :func:`refused_before_motion`: every ``refused_*`` outcome is reached before the arm is commanded.

``FixtureEnvelope``, ``CONTAINER_AGITATE`` with its strategy, and the executor that drove the agitation left on
2026-10-16 (the owner's cleanup), as the scene-blind nudge strategy did on 2026-09-29: the push runs inside the pick
attempt (``push_motion.execute_push``).
"""

from __future__ import annotations

import importlib
import unittest
from typing import Any

from src.robot.execution.autonomous_grasp import GraspMode, _profile_for
from src.robot.grasping.recovery.policy import SceneRecoveryAction as A
from src.robot.grasping.recovery.policy import SceneRecoveryPolicy, push_permitted, refused_before_motion


def _policy(*actions: Any, max_actions: int = 3, **kwargs: Any) -> SceneRecoveryPolicy:
    return SceneRecoveryPolicy(enabled=True, allowed_actions=tuple(actions), max_recovery_actions=max_actions,
                               **kwargs)


class SceneRecoveryPolicyTests(unittest.TestCase):
    def test_the_default_is_off_and_rescans_only(self) -> None:
        policy = SceneRecoveryPolicy()
        self.assertFalse(policy.enabled)
        self.assertEqual((A.RESCAN,), policy.allowed_actions)
        self.assertFalse(policy.permits(A.RESCAN))

    def test_it_permits_what_it_lists_while_it_is_on(self) -> None:
        self.assertTrue(_policy(A.RESCAN, A.NUDGE_TARGET).permits(A.NUDGE_TARGET))
        self.assertFalse(_policy(A.RESCAN).permits(A.NUDGE_TARGET))

    def test_what_it_refuses(self) -> None:
        refused = {
            "a duplicate action": lambda: _policy(A.RESCAN, A.RESCAN),
            "NONE in the allow-list": lambda: _policy(A.NONE),
            "a plain string": lambda: _policy("rescan"),
            "a negative total": lambda: SceneRecoveryPolicy(max_recovery_actions=-1),
            "a negative budget": lambda: _policy(A.RESCAN, per_action_budget={A.RESCAN: -1}),
            "an empty mode": lambda: _policy(A.RESCAN, apply_modes=("",)),
        }
        for name, build in refused.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    build()


class WhereAPushMayRunTests(unittest.TestCase):
    def test_dense_clutter_with_the_push_allowed(self) -> None:
        self.assertTrue(push_permitted(_profile_for(GraspMode.DENSE_CLUTTER), _policy(A.NUDGE_TARGET, A.RESCAN)))

    def test_every_gate_closes_it(self) -> None:
        dense = _profile_for(GraspMode.DENSE_CLUTTER)
        closed = {
            "auto": (_profile_for(GraspMode.AUTO), _policy(A.NUDGE_TARGET, A.RESCAN)),
            "easy": (_profile_for(GraspMode.EASY), _policy(A.NUDGE_TARGET, A.RESCAN)),
            "off": (dense, SceneRecoveryPolicy(enabled=False, allowed_actions=(A.NUDGE_TARGET, A.RESCAN))),
            "not allowed": (dense, _policy(A.RESCAN)),
            "mode not applied": (dense, _policy(A.NUDGE_TARGET, apply_modes=("auto",))),
            "budget zero": (dense, _policy(A.NUDGE_TARGET, per_action_budget={A.NUDGE_TARGET: 0})),
            "max actions zero": (dense, _policy(A.NUDGE_TARGET, max_actions=0)),
        }
        for name, (profile, policy) in closed.items():
            with self.subTest(case=name):
                self.assertFalse(push_permitted(profile, policy))


class BeforeMotionTests(unittest.TestCase):
    def test_every_refusal_is_before_motion_and_nothing_else_is(self) -> None:
        for outcome in ("refused_push_runs_in_the_pick", "refused_no_arm", "refused_jaws_unknown"):
            self.assertTrue(refused_before_motion(outcome), outcome)
        for outcome in ("aborted_motion_failed", "completed", "pushed", ""):
            self.assertFalse(refused_before_motion(outcome), outcome)


class WhatLeftStaysGoneTests(unittest.TestCase):
    def test_nothing_offers_the_blind_nudge_the_agitation_or_its_box_any_more(self) -> None:
        for module in ("src.robot.grasping", "src.robot.grasping.recovery.policy"):
            for name in ("SmallNudgeStrategy", "ContainerAgitateStrategy", "FixtureEnvelope",
                         "execute_recovery_motion"):
                with self.subTest(module=module, name=name):
                    self.assertFalse(hasattr(importlib.import_module(module), name))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
