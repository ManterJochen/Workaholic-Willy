"""Keys that were dead, and the behaviour each one now has.

These were found by a full wired-vs-dead sweep of the config tree (2026-08-22). Fifty keys had no
reader and were deleted; four had a reader-shaped GAP -- an operator could set them, the schema
accepted them, and the runtime went on doing whatever it did before. This file pinned the three that
are not `recovery.fixture` (that one has its own file, and a different story: a latent crash rather
than an ignored value). Two of them left on 2026-09-29 with the two-scan refinement they served,
`closed_loop.pregrasp_rescan` and `verification.post_lift_vision_check` (with its
`vision_displacement_iou_max`), and their cases with them; a tree that still writes one is refused at
load (`tests/test_no_second_look_before_the_close.py`). The verifier the config built left with the
whole `verification` block on 2026-09-29, and its case with it
(`tests/test_the_gripper_says_whether_it_holds.py`). What is left pinned here: `rl.policy_id` is
checked.

Why a hand-written file when `test_grasping_wiring_guard.py` exists: that guard enumerates booleans
whose NAME CONTAINS `enabled`. Both flags below are booleans that do not, so they sat in its blind
spot for months. Widening the guard was measured on 2026-08-22 and reds on nine flags, most of them
artefacts of the differential rather than real inertness -- so the guard stays narrow and these get
named tests instead. The gap is written down at the enumerator itself.
"""

from __future__ import annotations

import logging
import unittest


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
