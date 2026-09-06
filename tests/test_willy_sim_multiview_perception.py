"""Pure / mock-safe (no isaacsim) tests for the perception-budget COLLECT decision policy (C).

The ``run_multiview_pick`` module imports its Isaac touchpoints lazily inside the run function, so the
``perception_collect_decision`` helper (the deterministic occlusion-driven STOP/CONTINUE explorer that
produces a NON-DEGENERATE v5 training corpus) imports and runs Isaac-free. These tests pin its contract:
determinism, both-actions coverage, occlusion correlation, and clamping.
"""

from __future__ import annotations

import unittest


class PerceptionCollectDecisionTests(unittest.TestCase):
    def _fn(self):  # noqa: ANN202 - lazy import keeps the test isaacsim-free
        from src.willy_sim.run_multiview_pick import perception_collect_decision

        return perception_collect_decision

    def test_deterministic_for_same_seed_episode(self) -> None:
        decide = self._fn()
        a = decide(occlusion_ratio=0.5, seed=3, episode=7)
        b = decide(occlusion_ratio=0.5, seed=3, episode=7)
        self.assertEqual(a, b)

    def test_both_actions_appear_across_occlusion(self) -> None:
        decide = self._fn()
        acts = [
            decide(occlusion_ratio=o / 10.0, seed=0, episode=i)[0]
            for o in range(11)
            for i in range(6)
        ]
        # A non-degenerate corpus needs BOTH actions with meaningful support (no single-action collapse).
        self.assertIn("stop", acts)
        self.assertIn("continue", acts)
        self.assertGreater(acts.count("stop"), 3)
        self.assertGreater(acts.count("continue"), 3)

    def test_continue_prob_rises_with_occlusion(self) -> None:
        decide = self._fn()
        _, p_low = decide(occlusion_ratio=0.0, seed=0, episode=0)
        _, p_high = decide(occlusion_ratio=1.0, seed=0, episode=0)
        self.assertLess(p_low, p_high)
        self.assertAlmostEqual(p_low, 0.45, places=6)
        self.assertAlmostEqual(p_high, 0.85, places=6)

    def test_probability_is_clamped(self) -> None:
        decide = self._fn()
        _, p_neg = decide(occlusion_ratio=-2.0, seed=1, episode=1)
        _, p_big = decide(occlusion_ratio=5.0, seed=1, episode=1)
        self.assertGreaterEqual(p_neg, 0.0)
        self.assertLessEqual(p_big, 1.0)
        # Out-of-range occlusion clamps to the [0,1] endpoints' probabilities.
        self.assertAlmostEqual(p_neg, 0.45, places=6)
        self.assertAlmostEqual(p_big, 0.85, places=6)

    def test_action_matches_seeded_threshold(self) -> None:
        from src.willy_sim.run_multiview_pick import (
            _perception_seeded_uniform,
            perception_collect_decision,
        )

        # The action is CONTINUE iff the seeded uniform < continue_prob — pin the exact rule.
        for i in range(20):
            u = _perception_seeded_uniform(0, i, salt="decision")
            action, p = perception_collect_decision(occlusion_ratio=0.4, seed=0, episode=i)
            self.assertEqual(action, "continue" if u < p else "stop")


class OccludeSceneTests(unittest.TestCase):
    def test_occlude_scene_builds_tall_neighbour_side_occluder(self) -> None:
        # The hard-occlusion follow-up scene: a tall occluder on oblique_L's (-Y) side so a single-oblique
        # STOP is genuinely occluded. Assert the spec builder honours the optional 4th (size_mm) tuple element.
        from src.willy_sim.run_multiview_pick import SCENES, _scene_specs

        self.assertIn("occlude", SCENES)
        specs = _scene_specs("occlude")
        self.assertEqual(len(specs), 2)
        target, occluder = specs[0], specs[1]
        self.assertEqual(target.size_mm, (30.0, 30.0, 50.0))  # default cube target
        self.assertGreater(occluder.size_mm[2], 100.0)  # a TALL occluder (blocks the slanted oblique view)
        self.assertLess(occluder.position_mm[1], 0.0)  # on the -Y (oblique_L) side, between it and the target

    def test_default_scenes_still_build(self) -> None:
        # Backward-compat: the 3-tuple scenes (no size) still build with the default cube size.
        from src.willy_sim.run_multiview_pick import _scene_specs

        for scene in ("clutter", "close"):
            specs = _scene_specs(scene)
            self.assertTrue(all(s.size_mm == (30.0, 30.0, 50.0) for s in specs))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
