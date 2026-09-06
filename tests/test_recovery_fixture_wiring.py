"""A config that declares a physical recovery action must RUN, not raise on every pick.

THE DEFECT THIS PINS, and it is worth stating precisely because it was not a dead key — it was a
latent crash reachable from a perfectly valid config:

* the schema DEMANDS `robot.grasping.recovery.fixture` as soon as `allowed_actions` names a physical
  action (`nudge_target`, `container_agitate`), so such a config LOADS — that guard works;
* `SceneRecoveryPolicy.__post_init__` REFUSES to construct a physical action without a
  `FixtureEnvelope`, which is also right: a push with no declared bound is a robot shoving an
  unbounded workspace;
* but the builder constructed the policy without passing `fixture=` at all.

So the two guards agreed with each other and the wiring between them was missing. The config
validated at load and then raised `ValueError` on EVERY `pick()` — not once at startup, where it
would have been obvious, but on the hot path, for a cell that had been running fine until somebody
enabled recovery.

The envelope is now carried YAML -> `EffectiveRecoveryOrchestratorConfig.fixture` -> `FixtureEnvelope`.
"""

from __future__ import annotations

import unittest

from src.robot.execution.autonomous_grasp.config import (
    EffectiveRecoveryOrchestratorConfig,
)
from src.robot.grasping.recovery.policy import (
    FixtureEnvelope,
    SceneRecoveryAction,
    SceneRecoveryPolicy,
)

_CENTRE = (450.0, 0.0, 60.0)
_HALF = (150.0, 100.0, 40.0)


class TheEnvelopeReachesThePolicyTests(unittest.TestCase):
    def test_a_physical_action_without_an_envelope_still_refuses(self) -> None:
        """The guard that was firing. It is correct and stays — the wiring was what was missing."""
        with self.assertRaises(ValueError) as caught:
            SceneRecoveryPolicy(
                enabled=True,
                allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
                max_recovery_actions=2,
                fixture=None,
            )
        self.assertIn("FixtureEnvelope", str(caught.exception))

    def test_the_same_policy_constructs_once_the_envelope_is_supplied(self) -> None:
        policy = SceneRecoveryPolicy(
            enabled=True,
            allowed_actions=(SceneRecoveryAction.NUDGE_TARGET,),
            max_recovery_actions=2,
            fixture=FixtureEnvelope(
                center_mm=_CENTRE, half_extents_mm=_HALF, max_nudge_mm=8.0
            ),
        )
        self.assertIsNotNone(policy.fixture)
        assert policy.fixture is not None
        self.assertEqual(policy.fixture.center_mm, _CENTRE)
        self.assertEqual(policy.fixture.max_nudge_mm, 8.0)

    def test_the_overlay_carries_the_box_the_operator_declared(self) -> None:
        """`EffectiveRecoveryOrchestratorConfig` is the seam between YAML and the envelope."""
        overlay = EffectiveRecoveryOrchestratorConfig(
            enabled=True,
            allowed_actions=("nudge_target",),
            fixture=(_CENTRE, _HALF, 8.0),
        )
        self.assertEqual(overlay.fixture, (_CENTRE, _HALF, 8.0))

    def test_it_defaults_to_absent_so_nothing_changes_for_a_cell_without_one(self) -> None:
        """The overwhelmingly common case: no physical recovery, no fixture, no behaviour change."""
        self.assertIsNone(EffectiveRecoveryOrchestratorConfig().fixture)


class TheTelemetryContractIsUntouchedTests(unittest.TestCase):
    def test_the_new_field_is_not_in_the_flat_dict(self) -> None:
        """`to_dict()` is a frozen 79-key contract with its own order test.

        The envelope is a nested box, not a scalar, and widening that contract has telemetry-catalog
        consequences — so the field is deliberately absent from it. This asserts the absence so a
        later "tidy-up" that adds it has to argue with a test rather than with a comment.
        """
        from src.robot.execution.autonomous_grasp.config import (
            EffectiveGraspingConfig,
            GraspMode,
        )

        emitted = EffectiveGraspingConfig(
            default_mode=GraspMode.AUTO,
            max_attempts=5,
            closed_loop_enabled=False,
            verification_enabled=False,
            dense_recovery_enabled=False,
            dense_recovery_allowed_actions=(),
        ).to_dict()
        self.assertNotIn("recovery_orchestrator_fixture", emitted)
        for key in ("recovery_orchestrator_enabled", "recovery_orchestrator_allowed_actions"):
            with self.subTest(key=key):
                self.assertIn(key, emitted, "the surrounding contract moved")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
