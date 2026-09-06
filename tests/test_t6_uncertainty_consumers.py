"""Phase T6 — RED tests for T1/T2/T5 consumer carriers.

Q4=C asks for full T1+T2+T5 integration of the fused uncertainty
signal. Safety + byte-identical-defaults discipline mean T6 lands:

* **T1 (decision)** — full behavioral wiring: when
  ``uncertainty.enabled=True``, ``DecisionPolicy.auto_uncertainty_threshold``
  is overridden by ``uncertainty.fail_closed_threshold``. Confirmed in
  ``test_t6_uncertainty_integration.py``.
* **T2 (ranking) and T5 (recovery)** — carrier-only: the
  :class:`EffectiveGraspingConfig` exposes the operator knobs
  (``uncertainty_ranking_penalty_weight`` and
  ``uncertainty_recovery_threshold``) and the runtime threads them
  through to the relevant components, but the *defaults are
  byte-identical no-ops* (zero penalty, threshold above 1.0 so it
  never fires). Real behavioral integration of T2/T5 lands in a
  follow-on slice with its own RED tests.

This file pins the carrier surface for T2 and T5.
"""

from __future__ import annotations

import unittest

from src.config.schema.robot import RobotConfig
from src.config.schema.robot.robot_schema import GraspingUncertaintyConfig


class T2RankingCarrierTests(unittest.TestCase):
    def test_default_ranking_penalty_weight_is_zero(self) -> None:
        cfg = GraspingUncertaintyConfig()
        # New knob on the schema; defaults to zero ⇒ T2 ranking
        # behaviour is byte-identical to T5.
        self.assertEqual(cfg.ranking_penalty_weight, 0.0)

    def test_ranking_penalty_weight_must_be_non_negative(self) -> None:
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(ranking_penalty_weight=-0.1)


class T5RecoveryCarrierTests(unittest.TestCase):
    def test_default_recovery_threshold_above_one(self) -> None:
        cfg = GraspingUncertaintyConfig()
        # Threshold strictly above 1.0 means the fused uncertainty
        # (clamped to [0,1]) can never exceed it ⇒ recovery loop
        # priorities are unchanged.
        self.assertGreater(cfg.recovery_aggressive_threshold, 1.0)

    def test_recovery_threshold_unit_or_above_unit(self) -> None:
        # Operator may set it to any non-negative finite value; we
        # only check the lower bound here.
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            GraspingUncertaintyConfig(recovery_aggressive_threshold=-0.1)


class EffectiveConfigExposesT2T5KnobsTests(unittest.TestCase):
    def test_effective_config_carries_ranking_and_recovery_knobs(self) -> None:
        from src.robot.execution.autonomous_grasp import (
            AutonomousGraspService,
        )
        # Minimal config; defaults should propagate to the snapshot.
        cfg = RobotConfig(
            vendor="dummy", gripper={"vendor": "none"},
            grasping={"default_mode": "auto"},
        )
        # Build a tiny service double via a lightweight calc/perception
        # — reuse the doubles from the integration tests by importing
        # them lazily.
        from tests.test_t6_uncertainty_integration import _Calc, _FakePerception
        svc = AutonomousGraspService.from_robot_config(
            cfg, calculator=_Calc(), perception=_FakePerception(),
        )
        eff = svc.effective_config
        self.assertEqual(eff.uncertainty.ranking_penalty_weight, 0.0)
        self.assertGreater(eff.uncertainty.recovery_aggressive_threshold, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
