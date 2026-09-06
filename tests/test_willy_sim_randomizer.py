"""S2 DomainRandomizer — Isaac-free unit tests for the pure-numpy pose sampling + config + provenance.

The colour/lighting axis goes through Replicator (on-box only); here we lock the geometry diversity that
the RL dataset needs — deterministic, reproducible, bounded, and byte-identical when disabled.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.willy_sim.harness.randomizer import (
    DISTRACTOR_PALETTE,
    DomainRandomizer,
    RandomizationConfig,
    nearest_palette_name,
)

_HOME_POS = [(450.0, -120.0, 60.0), (580.0, -60.0, 60.0), (320.0, -120.0, 60.0)]
_HOME_QUAT = [(1.0, 0.0, 0.0, 0.0), (0.70710678, 0.70710678, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]


class ConfigFactoryTests(unittest.TestCase):
    def test_disabled_is_inert(self) -> None:
        c = RandomizationConfig.disabled()
        self.assertFalse(c.enabled)
        self.assertFalse(c.any_pose)
        self.assertFalse(c.any_visual)

    def test_full_enables_all_axes(self) -> None:
        c = RandomizationConfig.full(seed=7, jitter_mm=20.0)
        self.assertTrue(c.enabled and c.any_pose and c.any_visual)
        self.assertGreater(c.position_jitter_mm, 0.0)
        self.assertGreater(c.orientation_jitter_deg, 0.0)
        self.assertTrue(c.randomize_color)
        self.assertIsNotNone(c.dome_intensity_range)
        self.assertIsNotNone(c.key_intensity_range)

    def test_legacy_is_position_plus_optional_colour_only(self) -> None:
        c = RandomizationConfig.legacy(seed=3, jitter_mm=20.0, recolor=True)
        self.assertTrue(c.enabled and c.any_pose and c.any_visual)
        self.assertEqual(c.orientation_jitter_deg, 0.0)  # legacy: NO orientation
        self.assertIsNone(c.dome_intensity_range)        # legacy: NO lighting
        self.assertTrue(c.randomize_color)
        c2 = RandomizationConfig.legacy(seed=3, jitter_mm=20.0, recolor=False)
        self.assertFalse(c2.any_visual)


class SamplePoseTests(unittest.TestCase):
    def test_disabled_returns_home_unchanged(self) -> None:
        r = DomainRandomizer(RandomizationConfig.disabled())
        ep = r.sample_pose(5, _HOME_POS, _HOME_QUAT)
        self.assertEqual(ep.positions_mm, tuple(tuple(p) for p in _HOME_POS))
        self.assertEqual(ep.orientations_wxyz, tuple(tuple(q) for q in _HOME_QUAT))
        self.assertEqual(ep.yaw_deltas_deg, (0.0, 0.0, 0.0))

    def test_reproducible_by_seed_and_index(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=42, jitter_mm=20.0))
        a = r.sample_pose(3, _HOME_POS, _HOME_QUAT)
        b = r.sample_pose(3, _HOME_POS, _HOME_QUAT)
        self.assertEqual(a.positions_mm, b.positions_mm)
        self.assertEqual(a.orientations_wxyz, b.orientations_wxyz)

    def test_diversity_across_episodes(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=42, jitter_mm=20.0))
        a = r.sample_pose(0, _HOME_POS, _HOME_QUAT)
        b = r.sample_pose(1, _HOME_POS, _HOME_QUAT)
        self.assertNotEqual(a.positions_mm, b.positions_mm)

    def test_position_within_jitter_bounds_xy_z_fixed(self) -> None:
        jit = 20.0
        r = DomainRandomizer(RandomizationConfig.full(seed=1, jitter_mm=jit))
        for idx in range(25):
            ep = r.sample_pose(idx, _HOME_POS, _HOME_QUAT)
            for home, got in zip(_HOME_POS, ep.positions_mm):
                self.assertLessEqual(abs(got[0] - home[0]), jit + 1e-9)
                self.assertLessEqual(abs(got[1] - home[1]), jit + 1e-9)
                self.assertEqual(got[2], home[2])  # Z is not jittered

    def test_yaw_within_bounds_and_quat_normalized(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=9, jitter_mm=0.0))  # orientation only
        for idx in range(25):
            ep = r.sample_pose(idx, _HOME_POS, _HOME_QUAT)
            for yaw, q in zip(ep.yaw_deltas_deg, ep.orientations_wxyz):
                self.assertLessEqual(abs(yaw), 15.0 + 1e-6)  # full() uses 15 deg
                self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=6)

    def test_zero_jitter_keeps_home(self) -> None:
        # position_jitter=0 AND orientation=0 -> any_pose False -> home unchanged even when enabled
        c = RandomizationConfig(enabled=True, seed=1, position_jitter_mm=0.0, orientation_jitter_deg=0.0)
        ep = DomainRandomizer(c).sample_pose(2, _HOME_POS, _HOME_QUAT)
        self.assertEqual(ep.positions_mm, tuple(tuple(p) for p in _HOME_POS))


class PaletteAndProvenanceTests(unittest.TestCase):
    def test_nearest_palette_name(self) -> None:
        self.assertEqual(nearest_palette_name((0.9, 0.1, 0.1)), "red")
        self.assertEqual(nearest_palette_name((0.1, 0.65, 0.18)), "green")
        self.assertEqual(nearest_palette_name((0.12, 0.22, 0.9)), "blue")

    def test_palette_names_unique(self) -> None:
        names = [n for n, _rgb in DISTRACTOR_PALETTE]
        self.assertEqual(len(names), len(set(names)))

    def test_provenance_carries_applied_params(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=11, jitter_mm=20.0))
        ep = r.sample_pose(4, _HOME_POS, _HOME_QUAT)
        prov = r.provenance(4, ep, ["red", "green", None], {"dome_intensity": 175.3, "key_intensity": 320.0})
        self.assertEqual(prov["seed"], 11 + 4)  # per-episode seed
        self.assertTrue(prov["enabled"])
        self.assertEqual(len(prov["positions_mm"]), len(_HOME_POS))
        self.assertEqual(prov["dome_intensity"], 175.3)            # merged Replicator lighting readback
        self.assertEqual(prov["object_set_colors"], ["red", "green", None])  # numpy colour names
        self.assertTrue(prov["axes"]["position"] and prov["axes"]["orientation"])

    def test_apply_lighting_noop_without_setup(self) -> None:
        # apply_lighting returns {} when no lighting axis was registered (off-box safe — no Replicator import)
        r = DomainRandomizer(RandomizationConfig.full(seed=1, jitter_mm=20.0))
        self.assertEqual(r.apply_lighting(0), {})


class SampleColorsTests(unittest.TestCase):
    def test_disabled_or_off_returns_all_none(self) -> None:
        self.assertEqual(DomainRandomizer(RandomizationConfig.disabled()).sample_colors(0, 3, 0),
                         [None, None, None])
        c = RandomizationConfig(enabled=True, seed=1, randomize_color=False)
        self.assertEqual(DomainRandomizer(c).sample_colors(0, 3, 0), [None, None, None])

    def test_target_kept_distractors_coloured(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=5, jitter_mm=20.0))
        cols = r.sample_colors(0, 3, target_index=0)
        self.assertIsNone(cols[0])                       # target identity kept
        self.assertIsNotNone(cols[1])
        self.assertIsNotNone(cols[2])
        # each non-None is a (name, rgb) from the palette
        for c in cols[1:]:
            assert c is not None
            self.assertIn(c[0], {n for n, _ in DISTRACTOR_PALETTE})

    def test_reproducible_and_diverse(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=5, jitter_mm=20.0))
        a = r.sample_colors(2, 3, 0)
        b = r.sample_colors(2, 3, 0)
        self.assertEqual(a, b)                            # reproducible by (seed, idx)
        c = r.sample_colors(3, 3, 0)
        self.assertNotEqual([x[0] for x in a[1:] if x], [x[0] for x in c[1:] if x])  # diverse across episodes

    def test_names_unique_within_episode(self) -> None:
        r = DomainRandomizer(RandomizationConfig.full(seed=8, jitter_mm=20.0))
        names = [c[0] for c in r.sample_colors(0, 4, 0) if c is not None]
        self.assertEqual(len(names), len(set(names)))     # distinct distractor colours (shuffled palette)


if __name__ == "__main__":
    unittest.main()
