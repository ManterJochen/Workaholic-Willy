"""One flag in YAML has to produce a cell that really does the thing.

**The bug this file exists to prevent, stated plainly.** ``build_subpolicies`` has always turned
``closed_loop`` / ``verification`` / ``dense_recovery`` into POLICIES. Nothing turned them into the
objects that consume those policies -- ``refiner``, ``verifier``, ``recovery_strategy`` were
constructor-only. So an operator who set ``closed_loop.enabled: true``, restarted the cell and
watched it run got ``mode_requires_refinement_but_no_refiner_wired`` at pick time: a flag that reads
as on, a boot log that says nothing, and a cell quietly running the old path.

It was never a capability problem. ``DefaultPreGraspRefiner`` needs only its policy, both verifiers
take no arguments, and so do all three recovery strategies -- the six lines that assemble them were
simply never written. These tests pin that they now are, that an explicit object still wins, and
that a policy left with nothing to drive it fails the BUILD rather than every later pick.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from src.config.schema.robot.grasping_schema import RobotGraspingConfig
from src.robot.execution.autonomous_grasp.builders import (
    assert_closed_loop_actors_wired,
    build_closed_loop_actors,
    build_subpolicies,
)
from src.robot.grasping.closed_loop.refinement import (
    DefaultPreGraspRefiner,
    RefinementPolicy,
)
from src.robot.grasping.closed_loop.verification import (
    CompositeGraspVerifier,
    GraspVerificationPolicy,
)
from src.robot.grasping.recovery.policy import (
    ActivePerceptionRecoveryStrategy,
    NextTargetRecoveryStrategy,
    NoRecoveryStrategy,
    SceneRecoveryPolicy,
)


def _actors(**grasping):
    config = RobotGraspingConfig(**grasping)
    refinement, verification, recovery = build_subpolicies(
        config,
        refinement_policy=None,
        verification_policy=None,
        recovery_policy=None,
        standoff_mm=80.0,
        recovery_fixture=None,
    )
    return build_closed_loop_actors(
        config,
        refinement_policy=refinement,
        verification_policy=verification,
        recovery_policy=recovery,
        refiner=None,
        verifier=None,
        recovery_strategy=None,
    )


class OneFlagIsEnoughTests(unittest.TestCase):
    def test_default_config_builds_nothing(self) -> None:
        refiner, verifier, strategy = _actors()

        self.assertIsNone(refiner)
        self.assertIsNone(verifier)
        self.assertIsNone(strategy)

    def test_closed_loop_enabled_builds_a_refiner(self) -> None:
        refiner, _, _ = _actors(closed_loop={"enabled": True})

        self.assertIsInstance(refiner, DefaultPreGraspRefiner)
        assert isinstance(refiner, DefaultPreGraspRefiner)
        self.assertTrue(refiner.policy.enabled)
        self.assertIsNotNone(refiner.tracker, "the tracker default never materialised")

    def test_verification_enabled_builds_both_verifiers(self) -> None:
        """Both, because they read different things -- object-detect and jaw width."""

        _, verifier, _ = _actors(verification={"enabled": True})

        self.assertIsInstance(verifier, CompositeGraspVerifier)
        assert isinstance(verifier, CompositeGraspVerifier)
        self.assertEqual(len(verifier.verifiers), 2)
        self.assertFalse(
            verifier.require_all_conclusive,
            "a sensorless gripper would fail every pick, including the good ones",
        )

    def test_conclusiveness_can_be_demanded(self) -> None:
        _, verifier, _ = _actors(
            verification={"enabled": True, "require_all_conclusive": True}
        )

        assert isinstance(verifier, CompositeGraspVerifier)
        self.assertTrue(verifier.require_all_conclusive)

    def test_dense_recovery_enabled_builds_the_configured_strategy(self) -> None:
        for name, expected in (
            ("active_perception", ActivePerceptionRecoveryStrategy),
            ("next_target", NextTargetRecoveryStrategy),
            ("none", NoRecoveryStrategy),
        ):
            with self.subTest(strategy=name):
                _, _, strategy = _actors(
                    dense_recovery={"enabled": True, "strategy": name}
                )
                self.assertIsInstance(strategy, expected)

    def test_the_default_strategy_never_moves_a_part(self) -> None:
        """`active_perception` rescans and relocates the camera; it cannot make the scene worse."""

        _, _, strategy = _actors(dense_recovery={"enabled": True})

        self.assertIsInstance(strategy, ActivePerceptionRecoveryStrategy)

    def test_an_explicit_object_still_wins(self) -> None:
        sentinel = NoRecoveryStrategy()
        config = RobotGraspingConfig(dense_recovery={"enabled": True})
        _, _, recovery = build_subpolicies(
            config, refinement_policy=None, verification_policy=None,
            recovery_policy=None, standoff_mm=80.0, recovery_fixture=None,
        )

        _, _, strategy = build_closed_loop_actors(
            config,
            refinement_policy=None,
            verification_policy=None,
            recovery_policy=recovery,
            refiner=None,
            verifier=None,
            recovery_strategy=sentinel,
        )

        self.assertIs(strategy, sentinel)


class BuildRefusesRatherThanEveryPickTests(unittest.TestCase):
    """The complaint belongs where the mistake is, while somebody is still looking."""

    def test_an_enabled_policy_with_no_actor_refuses(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            assert_closed_loop_actors_wired(
                refinement_policy=RefinementPolicy(enabled=True),
                verification_policy=None,
                recovery_policy=None,
                refiner=None,
                verifier=None,
                recovery_strategy=None,
            )

        self.assertIn("closed_loop", str(ctx.exception))

    def test_it_names_every_block_that_is_short(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            assert_closed_loop_actors_wired(
                refinement_policy=RefinementPolicy(enabled=True),
                verification_policy=GraspVerificationPolicy(enabled=True),
                recovery_policy=SceneRecoveryPolicy(enabled=True),
                refiner=None,
                verifier=None,
                recovery_strategy=None,
            )

        message = str(ctx.exception)
        for block in ("closed_loop", "verification", "dense_recovery"):
            self.assertIn(block, message)

    def test_a_disabled_policy_needs_nothing(self) -> None:
        assert_closed_loop_actors_wired(
            refinement_policy=RefinementPolicy(enabled=False),
            verification_policy=None,
            recovery_policy=None,
            refiner=None,
            verifier=None,
            recovery_strategy=None,
        )


class EndToEndThroughFromRobotConfigTests(unittest.TestCase):
    """The measurement that matters: does the SERVICE come out able to refine and verify."""

    def _service(self, **grasping):
        from src.config.schema.robot.robot_schema import RobotConfig
        from src.robot.execution.autonomous_grasp import AutonomousGraspService
        from src.robot.grasping.types.feedback import GraspResult
        from src.robot.grasping.types.perception import PerceptionFrame

        def _frame() -> PerceptionFrame:
            mask = np.zeros((32, 32), dtype=np.uint8)
            mask[10:22, 10:22] = 1
            return PerceptionFrame(
                depth_map=np.full((32, 32), 500.0, dtype=np.float64),
                intrinsics=np.eye(3, dtype=np.float64),
                segmentations=(SimpleNamespace(mask=mask),),
            )

        calculator = SimpleNamespace(
            compute_result=lambda *a, **k: GraspResult(
                candidates=(), reasons=(), telemetry={}
            ),
            render_debug_images=False,
        )
        return AutonomousGraspService.from_robot_config(
            RobotConfig(vendor="dummy", grasping=grasping),
            calculator=calculator,  # type: ignore[arg-type]
            perception=SimpleNamespace(acquire=_frame),  # type: ignore[arg-type]
        )

    def test_a_yaml_flag_alone_produces_a_working_refiner(self) -> None:
        service = self._service(closed_loop={"enabled": True}, default_mode="auto")

        self.assertIsNotNone(
            service.refiner,
            "closed_loop.enabled in YAML still leaves the cell unable to refine",
        )

    def test_a_yaml_flag_alone_produces_a_working_verifier(self) -> None:
        service = self._service(verification={"enabled": True}, default_mode="auto")

        self.assertIsNotNone(service.verifier)

    def test_a_yaml_flag_alone_produces_a_recovery_strategy(self) -> None:
        service = self._service(dense_recovery={"enabled": True}, default_mode="auto")

        self.assertIsInstance(service.recovery_strategy, ActivePerceptionRecoveryStrategy)


if __name__ == "__main__":
    unittest.main()
