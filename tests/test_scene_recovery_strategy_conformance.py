"""R10.1 — behavioral conformance of every SceneRecoveryStrategy implementation to the planner contract.

This guards the ``SceneRecoveryStrategy`` Protocol (``backend/src/robot/grasping/recovery.py``), which is
the vendor-neutral *planner* surface every built-in scene-recovery strategy implements: a single
``plan(context: SceneRecoveryContext) -> SceneRecoveryPlan`` method. Mirrors the SafetyGuard conformance
suite ("no-raise + typed-decision + accept/reject anchor"). Every strategy wired into the recovery layer
must:

* return a :class:`SceneRecoveryPlan` (never a bare ``bool`` / ``None`` / raw enum) whose ``action`` is a
  valid :class:`SceneRecoveryAction` member and whose ``reason`` is a string — the typed-decision contract;
* **NOT raise on a well-formed context** — the load-bearing fail-closed invariant. The module docstring
  promises "Strategies degrade gracefully: missing inputs yield ``SceneRecoveryAction.NONE`` rather than a
  crash." A strategy that raised instead of returning a NONE plan would crash the recovery loop rather than
  *declining* to act. This suite is the cross-strategy proof that none of them does, on both a benign and a
  signal-starved context;
* **fail closed when the policy is disabled** — the documented master switch (``SceneRecoveryPolicy.enabled``
  / ``permits``): "When :data:`False` every strategy in this module returns ``SceneRecoveryAction.NONE``."

The five strategies are constructed exactly as the existing ``tests/test_grasp_recovery.py`` builds them
(same profile / policy / context / fixture idioms), so this exercises the real shipped strategy set — not
hand-built stubs.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import Optional

import numpy as np

from src.geometry import Frame, Pose
from src.robot.execution.autonomous_grasp import (
    AutonomousGraspOutcome,
    GraspBehaviorProfile,
    GraspMode,
)
from src.robot.grasping import (
    ActivePerceptionRecoveryStrategy,
    ContainerAgitateStrategy,
    FixtureEnvelope,
    NextTargetRecoveryStrategy,
    NoRecoveryStrategy,
    SceneRecoveryAction,
    SceneRecoveryContext,
    SceneRecoveryPlan,
    SceneRecoveryPolicy,
    SceneRecoveryStrategy,
    SmallNudgeStrategy,
)
from src.robot.grasping.types.modes import GraspSamplingMode
from src.robot.grasping.types.perception import PerceptionFrame

# Every valid action vocabulary member — a plan whose action is outside this set is a contract violation.
_VALID_ACTIONS = frozenset(SceneRecoveryAction)

# A fixture that arms the physical (NUDGE / AGITATE) strategies; matches the test_grasp_recovery idiom.
_FIXTURE = FixtureEnvelope(
    center_mm=(0.0, 0.0, 200.0),
    half_extents_mm=(100.0, 100.0, 100.0),
    max_nudge_mm=10.0,
    max_agitate_amplitude_mm=5.0,
)

# The full action allow-list (inner gate) so any strategy *may* plan its action when armed.
_ALL_ACTIONS = (
    SceneRecoveryAction.RESCAN,
    SceneRecoveryAction.NEXT_VIEWPOINT,
    SceneRecoveryAction.NEXT_TARGET,
    SceneRecoveryAction.NUDGE_TARGET,
    SceneRecoveryAction.CONTAINER_AGITATE,
)

# Profile allow-list (outer gate) mirrors the policy's, as plain strings (recovery_allowed_actions contract).
_ALL_ACTION_STRINGS = tuple(a.value for a in _ALL_ACTIONS)


def _strategies() -> tuple[SceneRecoveryStrategy, ...]:
    """Every production SceneRecoveryStrategy, built with its real constructor idiom."""

    return (
        NoRecoveryStrategy(),
        ActivePerceptionRecoveryStrategy(),
        NextTargetRecoveryStrategy(),
        SmallNudgeStrategy(offset_axis=(1.0, 0.0, 0.0)),
        ContainerAgitateStrategy(),
    )


def _profile(actions: tuple[str, ...]) -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.DENSE_CLUTTER,
        sampling_mode=GraspSamplingMode.DENSE_CLUTTER,
        refinement_enabled=False,
        verification_enabled=False,
        recovery_allowed_actions=actions,
    )


def _easy_profile() -> GraspBehaviorProfile:
    return GraspBehaviorProfile(
        mode=GraspMode.EASY,
        sampling_mode=GraspSamplingMode.SINGLE_OBJECT,
        recovery_allowed_actions=(),
    )


def _seg() -> SimpleNamespace:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:4, 1:4] = 1
    return SimpleNamespace(mask=mask)


def _frame_with_n_segs(n: int) -> PerceptionFrame:
    return PerceptionFrame(
        depth_map=np.zeros((8, 8), dtype=np.float64),
        intrinsics=np.eye(3, dtype=np.float64),
        segmentations=tuple(_seg() for _ in range(n)),
    )


def _tcp_at(xyz: tuple[float, float, float] = (0.0, 0.0, 200.0)) -> Pose:
    return Pose(
        position_mm=np.asarray(xyz, dtype=np.float64),
        quaternion_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        frame=Frame.BASE,
        label="tcp",
    )


def _ctx(
    *,
    profile: Optional[GraspBehaviorProfile] = None,
    policy: Optional[SceneRecoveryPolicy] = None,
    history: tuple[SceneRecoveryAction, ...] = (),
    last_frame: Optional[PerceptionFrame] = None,
    current_tcp: Optional[Pose] = None,
) -> SceneRecoveryContext:
    return SceneRecoveryContext(
        profile=profile or _profile(_ALL_ACTION_STRINGS),
        policy=policy or _armed_policy(),
        last_outcome=AutonomousGraspOutcome.NO_VALID_GRASP,
        last_frame=last_frame,
        current_tcp=current_tcp,
        history=history,
    )


def _armed_policy() -> SceneRecoveryPolicy:
    """A fully-enabled policy permitting every action, with a fixture for the physical ones."""

    return SceneRecoveryPolicy(
        enabled=True,
        allowed_actions=_ALL_ACTIONS,
        max_recovery_actions=8,
        fixture=_FIXTURE,
    )


class SceneRecoveryStrategyConformanceTests(unittest.TestCase):
    """Every production SceneRecoveryStrategy honours the planner Protocol contract."""

    def setUp(self) -> None:
        self._strategies = _strategies()

    def test_every_strategy_satisfies_runtime_protocol(self) -> None:
        # SceneRecoveryStrategy is @runtime_checkable; each built-in must structurally satisfy it.
        self.assertEqual(len(self._strategies), 5, "expected all five built-in strategies")
        for s in self._strategies:
            with self.subTest(strategy=type(s).__name__):
                self.assertIsInstance(s, SceneRecoveryStrategy)

    def test_plan_returns_typed_plan_with_valid_action(self) -> None:
        # The typed-decision contract: on a fully-armed context every strategy must RETURN a
        # SceneRecoveryPlan (not a bare bool / None / raw enum) whose .action is a real
        # SceneRecoveryAction member and whose .reason is a string. An impl returning None or a
        # bare bool fails the isinstance; one returning an out-of-vocabulary action fails the membership.
        ctx = _ctx(
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
            last_frame=_frame_with_n_segs(0),
        )
        for s in self._strategies:
            with self.subTest(strategy=type(s).__name__):
                plan = s.plan(ctx)
                self.assertIsInstance(plan, SceneRecoveryPlan)
                self.assertIsInstance(plan.action, SceneRecoveryAction)
                self.assertIn(plan.action, _VALID_ACTIONS)
                self.assertIsInstance(plan.reason, str)
                # is_motion_action is a derived bool invariant on the plan; must not raise / must be bool.
                self.assertIsInstance(plan.is_motion_action, bool)

    def test_plan_never_raises_on_benign_or_starved_context(self) -> None:
        # THE fail-closed invariant (module docstring): "missing inputs yield SceneRecoveryAction.NONE
        # rather than a crash". For both a fully-armed context AND a signal-starved one (no frame, no TCP,
        # no grasp), every strategy must RETURN a typed plan — NONE raises. A strategy that raised here
        # would crash the recovery loop instead of declining to act.
        contexts = {
            "armed": _ctx(
                current_tcp=_tcp_at((0.0, 0.0, 200.0)),
                last_frame=_frame_with_n_segs(0),
            ),
            "starved": _ctx(),  # no frame, no tcp, no grasp anchor
        }
        for label, ctx in contexts.items():
            for s in self._strategies:
                with self.subTest(strategy=type(s).__name__, context=label):
                    try:
                        plan = s.plan(ctx)
                    except Exception as exc:  # noqa: BLE001 - the whole point is to prove no strategy raises
                        self.fail(
                            f"{type(s).__name__} raised {type(exc).__name__} on the {label} context "
                            f"instead of returning a SceneRecoveryPlan: {exc!r}"
                        )
                    self.assertIsInstance(plan, SceneRecoveryPlan)
                    self.assertIn(plan.action, _VALID_ACTIONS)

    def test_disabled_policy_forces_none_for_every_strategy(self) -> None:
        # The documented master switch (SceneRecoveryPolicy.enabled=False -> "every strategy in this module
        # returns SceneRecoveryAction.NONE"). This is the fail-closed REJECT path: even with a fixture armed
        # and a multi-target frame supplied, a disabled policy must veto every strategy. A strategy that
        # planned a real action under a disabled policy would be a genuine safety regression and fails here.
        disabled = SceneRecoveryPolicy(enabled=False, fixture=_FIXTURE)
        ctx = _ctx(
            policy=disabled,
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
            last_frame=_frame_with_n_segs(3),
        )
        for s in self._strategies:
            with self.subTest(strategy=type(s).__name__):
                plan = s.plan(ctx)
                self.assertIs(
                    plan.action,
                    SceneRecoveryAction.NONE,
                    f"{type(s).__name__} planned {plan.action} under a DISABLED policy",
                )

    def test_easy_profile_outer_gate_forces_none(self) -> None:
        # The outer (profile) gate: an EASY profile has an empty recovery_allowed_actions tuple, so the
        # locked S0 contract makes it provably free of recovery motion — every strategy must short-circuit
        # to NONE regardless of how permissive the policy is. Complements the disabled-policy (inner-gate)
        # path: both independent gates must each be able to force NONE.
        ctx = _ctx(
            profile=_easy_profile(),
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
            last_frame=_frame_with_n_segs(3),
        )
        for s in self._strategies:
            with self.subTest(strategy=type(s).__name__):
                plan = s.plan(ctx)
                self.assertIs(
                    plan.action,
                    SceneRecoveryAction.NONE,
                    f"{type(s).__name__} planned {plan.action} under an EASY (no-recovery) profile",
                )

    def test_acting_strategies_plan_their_action_when_armed(self) -> None:
        # Non-vacuity / two-path anchor: the four ACTING strategies must each plan their NON-NONE action on a
        # context that fully satisfies their preconditions. Paired with the disabled-policy test, this proves
        # both the accept AND the reject path are genuinely exercised — the suite is not all-NONE (or
        # all-error-swallow) theatre. NoRecoveryStrategy is intentionally always-NONE and excluded here.
        armed_full = _ctx(
            current_tcp=_tcp_at((0.0, 0.0, 200.0)),
            last_frame=_frame_with_n_segs(3),
        )
        expected = {
            "ActivePerceptionRecoveryStrategy": SceneRecoveryAction.RESCAN,
            "NextTargetRecoveryStrategy": SceneRecoveryAction.NEXT_TARGET,
            "SmallNudgeStrategy": SceneRecoveryAction.NUDGE_TARGET,
            "ContainerAgitateStrategy": SceneRecoveryAction.CONTAINER_AGITATE,
        }
        for s in self._strategies:
            name = type(s).__name__
            if name not in expected:
                continue
            with self.subTest(strategy=name):
                plan = s.plan(armed_full)
                self.assertIs(
                    plan.action,
                    expected[name],
                    f"{name} did not plan its action on a fully-armed context (got {plan.action})",
                )
                self.assertIsNot(plan.action, SceneRecoveryAction.NONE)

    def test_no_recovery_strategy_is_always_none(self) -> None:
        # The documented always-decline strategy: returns NONE even on a fully-armed context.
        no_recovery = NoRecoveryStrategy()
        plan = no_recovery.plan(
            _ctx(current_tcp=_tcp_at(), last_frame=_frame_with_n_segs(3))
        )
        self.assertIs(plan.action, SceneRecoveryAction.NONE)
        self.assertEqual(plan.reason, "no_recovery_strategy")


if __name__ == "__main__":
    unittest.main()
