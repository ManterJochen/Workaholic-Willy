"""L5 R2 — V5 perception-budget + V6 recovery shadow WIRING.

The router methods / loaders / carrier slots / telemetry sub-blocks already existed; R2 wires them:
  (a) maybe_build_shadow_router now LOADS the V5/V6 policies from config (committed baselines by default)
      and passes them into the ShadowRouter -- pinned here.
  (b) attach_shadow_router FOLDS the V5/V6 shadow telemetry onto the ShadowRouterTelemetry CARRIER ONLY
      (replace(telemetry, perception_telemetry=/recovery_telemetry=)) and DELIBERATELY omits the
      ``extras.update(emit_v5/v6_*)`` half that V3/V4 do, so the replay JSONL / U+ telemetry catalog stay
      byte-frozen at V4. That contract is enforced by the load-bearing comment in shadow.py + the fact that
      no rl_perception_*/rl_recovery_* keys are ever added to ``extras`` (a full report-level fold test
      would need a realistic AutonomousGraspReport, which the no-regression suite already exercises).
  (c) the pick_loop producer (the _finalize_v5/v6 finalisers) is guarded (policy-present) and covered by
      the no-regression suite (default-off byte-identical: full mock + --soak-report unchanged).
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.execution.autonomous_grasp.shadow import (
    maybe_build_shadow_router,
)


def _rl_shadow_cfg() -> SimpleNamespace:
    rl = SimpleNamespace(
        mode="rl_shadow",
        artifact_path=None,
        ranking_artifact_path=None,
        sequencing_artifact_path=None,
        perception_artifact_path=None,
        recovery_artifact_path=None,
    )
    return SimpleNamespace(rl=rl)


class LoadV5V6Tests(unittest.TestCase):
    def test_off_when_not_rl_shadow(self) -> None:
        cfg = SimpleNamespace(rl=SimpleNamespace(mode="geometry_only"))
        self.assertIsNone(maybe_build_shadow_router(cfg))

    def test_unsupported_mode_raises_loudly(self) -> None:
        # Reconciled typed gate: a schema-valid but not-yet-shipped RL mode fails LOUDLY here instead of
        # silently returning None (which used to hide an rl_active / rl_experimental misconfiguration).
        from src.robot.grasping.rl import RLModeNotImplementedError

        for mode in ("rl_active", "rl_experimental"):
            with self.assertRaises(RLModeNotImplementedError):
                maybe_build_shadow_router(SimpleNamespace(rl=SimpleNamespace(mode=mode)))

    def test_loads_v5_and_v6_policies_from_committed_baselines(self) -> None:
        # R2: with rl.mode=rl_shadow + no overrides, the V5 perception-budget + V6 recovery policies are
        # now wired onto the ShadowRouter from their committed baselines (were entirely unwired before).
        router = maybe_build_shadow_router(_rl_shadow_cfg())
        self.assertIsNotNone(router)
        assert router is not None
        self.assertIsNotNone(router.perception_policy)
        self.assertIsNotNone(router.recovery_policy)

    def test_bogus_artifact_path_degrades_to_none(self) -> None:
        # A missing/invalid override path leaves the slot None (the V5/V6 shadow simply does not run),
        # while the rest of the router still builds -- the try/except->None degrade contract.
        rl = SimpleNamespace(
            mode="rl_shadow",
            artifact_path=None,
            ranking_artifact_path=None,
            sequencing_artifact_path=None,
            perception_artifact_path="/does/not/exist_v5.json",
            recovery_artifact_path="/does/not/exist_v6.json",
        )
        router = maybe_build_shadow_router(SimpleNamespace(rl=rl))
        self.assertIsNotNone(router)
        assert router is not None
        self.assertIsNone(router.perception_policy)
        self.assertIsNone(router.recovery_policy)


if __name__ == "__main__":
    unittest.main()
