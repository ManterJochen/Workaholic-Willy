"""Live recovery-dispatch seam — the decide -> execute -> report loop with an injected executor.

No Isaac, no robot: ``execute_recovery`` is a fake that records the dispatched action and returns a
scripted outcome. Verifies the outcome reaches the canary KPI window (auto-fallback fires), the online
estimator folds the executed action, bad outcomes are rejected, and the fail-closed mask still wins.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.robot.grasping.rl.action_mask import ActionMaskContext
from src.robot.grasping.rl.canary_router import (
    CanaryConfig,
    load_active_canary_router,
)
from src.robot.grasping.rl.online_router import (
    FAMILY_RECOVERY,
    OnlineUpdateCoordinator,
)
from src.robot.grasping.rl.perception_budget_policy import (
    load_linucb_perception_budget_policy,
)
from src.robot.grasping.rl.recovery_dispatch import (
    RecoveryDispatchCoordinator,
    RecoveryDispatchError,
)
from src.robot.grasping.rl.recovery_policy import (
    RecoveryRequest,
    RecoveryStateKey,
    load_linucb_recovery_policy,
)
from src.robot.grasping.rl.router import ShadowRouter

_REPO = Path(__file__).resolve().parents[1]
_POL = _REPO / "docs" / "baselines" / "rl_policies"
_RECOVERY = _POL / "v6_recovery_baseline_v1.json"
_PROMO = _POL / "v6_recovery_baseline_v1_promotion_v1.json"


def _pass_promotion_path() -> Path:
    data = json.loads(_PROMO.read_text(encoding="utf-8"))
    data["verdict"] = "pass"
    data["reasons"] = []
    out = Path(tempfile.mkdtemp(prefix="dispatch_pass_")) / "recovery_pass_promotion_v1.json"
    out.write_text(json.dumps(data, indent=2, sort_keys=True))
    return out


_PASS = _pass_promotion_path()


def _state() -> RecoveryStateKey:
    return RecoveryStateKey(
        failure_class_bucket="empty_air_grasp",
        attempt_index_bucket="0",
        reobserve_count_bucket="0",
        dense_bucket="dense",
        last_outcome_bucket="failed",
    )


def _req(attempt_id: str) -> RecoveryRequest:
    return RecoveryRequest(attempt_id=attempt_id, state_key=_state())


def _router(config: CanaryConfig | None = None):
    return load_active_canary_router(
        policy_artifact_path=_RECOVERY,
        promotion_report_path=_PASS,
        config=config,
    )


class _FakeExecutor:
    """Records every dispatched action; returns a scripted (or constant) outcome."""

    def __init__(self, outcome: str = "succeeded") -> None:
        self.outcome = outcome
        self.actions: list[str] = []

    def __call__(self, action: str) -> str:
        self.actions.append(action)
        return self.outcome


class DispatchLoopTests(unittest.TestCase):
    def test_dispatch_applies_and_reports(self) -> None:
        router = _router(config=CanaryConfig(canary_pct=100.0))
        coord = RecoveryDispatchCoordinator(router=router)
        ex = _FakeExecutor(outcome="succeeded")
        decision = coord.dispatch(
            request=_req("d-1"),
            baseline_action="abort_recovery",
            mode="dense",
            execute_recovery=ex,
        )
        # the executor received the applied action, and the outcome reached the KPI window.
        self.assertEqual(ex.actions, [decision.applied_action])
        self.assertEqual(router.stats().window_attempts, 1)

    def test_bad_outcome_is_rejected(self) -> None:
        coord = RecoveryDispatchCoordinator(router=_router())
        with self.assertRaises(RecoveryDispatchError):
            coord.dispatch(
                request=_req("d-2"),
                baseline_action="reobserve",
                mode="dense",
                execute_recovery=lambda _a: "maybe",
            )

    def test_failures_drive_auto_fallback(self) -> None:
        router = _router(
            config=CanaryConfig(
                canary_pct=100.0, warmup_n=0, window_size=20,
                max_regret_rate=0.10, max_override_rate=1.0,
            )
        )
        coord = RecoveryDispatchCoordinator(router=router)
        ex = _FakeExecutor(outcome="failed")
        for i in range(5):
            coord.dispatch(
                request=_req(f"bad-{i}"),
                baseline_action="abort_recovery",
                mode="dense",
                execute_recovery=ex,
            )
        # every override failed -> regret spike -> the canary rolled itself back.
        self.assertTrue(router.is_fallback_active)

    def test_online_estimator_folds_executed_action(self) -> None:
        router = _router(config=CanaryConfig(canary_pct=100.0))
        base = ShadowRouter(
            candidate_policy=_StubCandidate(),  # type: ignore[arg-type]
            recovery_policy=load_linucb_recovery_policy(_RECOVERY),
            perception_policy=load_linucb_perception_budget_policy(
                _POL / "v5_perception_budget_baseline_v1.json"
            ),
        )
        online = OnlineUpdateCoordinator(base)
        coord = RecoveryDispatchCoordinator(router=router, online=online)
        self.assertEqual(online.n_updates(FAMILY_RECOVERY), 0)
        coord.dispatch(
            request=_req("on-1"),
            baseline_action="abort_recovery",
            mode="dense",
            execute_recovery=_FakeExecutor(outcome="succeeded"),
        )
        # the executed action was folded into the online estimator (shadow-only).
        self.assertEqual(online.n_updates(FAMILY_RECOVERY), 1)

    def test_degraded_scene_mask_still_wins_through_dispatch(self) -> None:
        router = _router(config=CanaryConfig(canary_pct=100.0))
        coord = RecoveryDispatchCoordinator(router=router)
        ex = _FakeExecutor(outcome="succeeded")
        decision = coord.dispatch(
            request=_req("mask-1"),
            baseline_action="abort_recovery",
            mode="dense",
            execute_recovery=ex,
            mask_context=ActionMaskContext(degraded_mode_active=True),
        )
        # the mask forced baseline; the executor ran the baseline, not an RL override.
        self.assertTrue(decision.rl_action_blocked_by_mask)
        self.assertEqual(decision.applied_action, "abort_recovery")
        self.assertEqual(ex.actions, ["abort_recovery"])


class _StubCandidate:
    name = "stub"
    version = 0

    def propose(self, request):  # pragma: no cover - never called
        raise NotImplementedError


if __name__ == "__main__":
    unittest.main()
