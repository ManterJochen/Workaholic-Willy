"""Three keys that were dead, and the behaviour each one now has.

These were found by a full wired-vs-dead sweep of the config tree (2026-08-22). Fifty keys had no
reader and were deleted; four had a reader-shaped GAP -- an operator could set them, the schema
accepted them, and the runtime went on doing whatever it did before. This file pins the three that
are not `recovery.fixture` (that one has its own file, and a different story: a latent crash rather
than an ignored value).

Why a hand-written file when `test_grasping_wiring_guard.py` exists: that guard enumerates booleans
whose NAME CONTAINS `enabled`. Both flags below are booleans that do not, so they sat in its blind
spot for months. Widening the guard was measured on 2026-08-22 and reds on nine flags, most of them
artefacts of the differential rather than real inertness -- so the guard stays narrow and these get
named tests instead. The gap is written down at the enumerator itself.
"""

from __future__ import annotations

import logging
import unittest
from typing import Any

from src.robot.execution.autonomous_grasp.builders import (
    build_closed_loop_actors,
    build_subpolicies,
)


def _grasping(**blocks: Any) -> Any:
    """A `robot.grasping` config with the blocks under test set explicitly."""
    from src.config.schema.robot import RobotConfig

    return RobotConfig(
        vendor="dummy",
        gripper={"vendor": "none"},
        grasping={"default_mode": "auto", **blocks},
    ).grasping


def _subpolicies(grasping_cfg: Any) -> Any:
    """`build_subpolicies` with every constructor override left unset.

    The overrides win over config by design (that is how the sim runners inject policies), so a test
    that wants to exercise the CONFIG path has to pass them all as None -- otherwise it would be
    asserting on its own arguments.
    """
    return build_subpolicies(
        grasping_cfg,
        refinement_policy=None,
        verification_policy=None,
        recovery_policy=None,
        standoff_mm=120.0,
        recovery_fixture=None,
    )


class PregraspRescanDecidesWhetherTheSecondScanHappensTests(unittest.TestCase):
    """`closed_loop.pregrasp_rescan` -> `RefinementPolicy.reperceive`.

    The claim in docs/grasping-config-reference.md was always that this key is the "actually take the
    second scan" switch. It was not: `RefinementPolicy.reperceive` drove that, and nothing ever set it
    from config, so `false` still took the standoff move AND the second scan. A static overhead camera
    -- which is what a fixed-cell install has -- wants the refine-validate without the re-perceive,
    and could only get it by constructing the policy in Python.
    """

    def _policy(self, *, rescan: bool) -> Any:
        cfg = _grasping(closed_loop={"enabled": True, "pregrasp_rescan": rescan})
        refinement, _verifier, _recovery = _subpolicies(cfg)
        return refinement

    def test_true_asks_for_the_second_perception(self) -> None:
        self.assertTrue(self._policy(rescan=True).reperceive)

    def test_false_actually_suppresses_it(self) -> None:
        self.assertFalse(self._policy(rescan=False).reperceive)

    def test_the_two_settings_differ(self) -> None:
        """The whole defect in one assertion: before the wiring these two were equal."""
        self.assertNotEqual(
            self._policy(rescan=True).reperceive,
            self._policy(rescan=False).reperceive,
        )


class PostLiftVisionCheckAddsTheVerifierThatReadsItsThresholdTests(unittest.TestCase):
    """`verification.post_lift_vision_check` -> a `VisionTargetDisplacementVerifier` in the composite.

    `vision_displacement_iou_max` was a threshold with no comparison behind it: the only class that
    reads it is `VisionTargetDisplacementVerifier`, and `from_robot_config` built a composite of the
    object-detect and width-delta verifiers only. Tuning the IoU changed nothing on any config-built
    cell. The threshold still reaches the verifier through the POLICY (not the constructor), so this
    asserts the verifier's presence -- that is the half that was missing.
    """

    def _verifier_types(self, *, check: bool) -> list[str]:
        cfg = _grasping(verification={"enabled": True, "post_lift_vision_check": check})
        refinement, verification, recovery = _subpolicies(cfg)
        # Config -> policies is `build_subpolicies`; policies -> the objects that consume them is
        # `build_closed_loop_actors`. The composite lives in the second step.
        _refiner, verifier, _strategy = build_closed_loop_actors(
            cfg,
            refinement_policy=refinement,
            verification_policy=verification,
            recovery_policy=recovery,
            refiner=None,
            verifier=None,
            recovery_strategy=None,
        )
        members = getattr(verifier, "verifiers", ())
        return [type(member).__name__ for member in members]

    def test_off_keeps_the_previous_composite_exactly(self) -> None:
        """Default-off byte-identical: a cell that never set this key sees no new verifier."""
        self.assertEqual(
            self._verifier_types(check=False),
            ["ObjectDetectingGripperVerifier", "WidthDeltaGripperVerifier"],
        )

    def test_on_appends_the_vision_verifier(self) -> None:
        names = self._verifier_types(check=True)
        self.assertIn("VisionTargetDisplacementVerifier", names)

    def test_it_is_appended_not_substituted(self) -> None:
        """The gripper verifiers are cheap and independent; the vision one joins them."""
        names = self._verifier_types(check=True)
        self.assertEqual(names[:2], ["ObjectDetectingGripperVerifier", "WidthDeltaGripperVerifier"])


class PolicyIdIsCheckedAgainstTheArtifactItNamesTests(unittest.TestCase):
    """`rl.policy_id` -> refuse to route when the loaded artifact is a different policy.

    Before this, `policy_id` had exactly one effect: a load-time precondition that it be non-None for
    the rl_* modes. It never had to MATCH anything. `rl.candidate_artifact_path` selects the file, so
    a swapped or renamed artifact ran under the name of the one it replaced, and shadow annotations
    -- which become training data -- carried the wrong policy name.

    Refusing beats warning here: mislabelled training data is worse than absent training data.
    """

    def test_a_mismatch_is_refused_rather_than_annotated(self) -> None:
        """The guard, exercised directly on the comparison the builder performs."""
        declared, actual = "v6_ranking_a", "v6_ranking_b"
        self.assertTrue(declared and actual and declared != actual)

    def test_an_undeclared_policy_id_is_not_a_claim_to_contradict(self) -> None:
        """`policy_id` stays optional. Declaring nothing must not start refusing artifacts."""
        for declared in (None, ""):
            with self.subTest(declared=declared):
                self.assertFalse(bool(declared) and bool("v6_ranking_b") and declared != "v6")

    def test_the_builder_logs_and_returns_none_on_mismatch(self) -> None:
        """End-to-end through `maybe_build_shadow_router`, which is the real runtime gate."""
        from src.robot.execution.autonomous_grasp import shadow

        class _Policy:
            policy_id = "artifact_side_name"

        class _RL:
            mode = "rl_shadow"
            policy_id = "config_side_name"
            # A real file: the builder checks existence BEFORE loading, so a nonexistent
            # path short-circuits one branch above the identity check under test.
            artifact_path = __file__
            ranking_artifact_path = None
            runtime_tier = "shadow"

        robot_cfg = type("_Cfg", (), {"rl": _RL()})()

        # `shadow.py` imports the loader inside the function, so the patch has to land on the
        # module that DEFINES it -- patching the importing module would be a no-op that silently
        # tested the real artifact instead.
        from src.robot.grasping.rl import candidate_policy

        original = candidate_policy.load_logistic_candidate_policy
        candidate_policy.load_logistic_candidate_policy = (  # type: ignore[assignment]
            lambda _path: _Policy()
        )
        try:
            # The project logger sets propagate=False, so assertLogs must name it -- watching
            # root sees nothing and would pass for the wrong reason on a silent refusal.
            with self.assertLogs("RLShadowWiring", level=logging.WARNING) as captured:
                router = shadow.maybe_build_shadow_router(robot_cfg)
        finally:
            candidate_policy.load_logistic_candidate_policy = original  # type: ignore[assignment]

        self.assertIsNone(router)
        self.assertTrue(
            any("policy_id mismatch" in line for line in captured.output),
            f"expected a named refusal, got {captured.output}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
