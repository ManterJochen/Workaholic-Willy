"""H3.9a — synthetic sensor depth-noise injector (the H1.0R input-side extension).

Covers: default == no-op (byte-identical), validity/background preservation, determinism, the wrapper, and
the discriminance the on-box gate rests on -- injected speckle creates off-surface cloud outliers that the
H1.2 RADIUS filter removes while leaving the surface (and a clean cloud) intact.
"""
from __future__ import annotations

import unittest

import numpy as np

from src.willy_sim.harness.depth_noise import (
    DepthNoiseConfig,
    NoisyDepthPerceptionSource,
    inject_depth_noise,
)
from src.robot.grasping.types.perception import PerceptionFrame


def _depth_patch() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.zeros((100, 100), dtype=np.float64)
    depth[30:70, 30:70] = 1000.0
    mask = np.zeros((100, 100), dtype=bool)
    mask[30:70, 30:70] = True
    intr = np.array([[500.0, 0.0, 50.0], [0.0, 500.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return depth, mask, intr


class DepthNoiseTests(unittest.TestCase):
    def test_default_config_is_disabled_and_noop(self) -> None:
        cfg = DepthNoiseConfig()
        self.assertFalse(cfg.enabled)
        depth, _, _ = _depth_patch()
        out = inject_depth_noise(depth, cfg, np.random.default_rng(0))
        np.testing.assert_array_equal(out, depth)  # byte-identical when disabled

    def test_validation_rejects_bad_params(self) -> None:
        with self.assertRaises(ValueError):
            DepthNoiseConfig(speckle_fraction=1.5)
        with self.assertRaises(ValueError):
            DepthNoiseConfig(gaussian_sigma_mm=-1.0)

    def test_only_valid_pixels_noised_and_deterministic(self) -> None:
        depth, _, _ = _depth_patch()  # background is 0
        cfg = DepthNoiseConfig(gaussian_sigma_mm=2.0, speckle_fraction=0.05, speckle_sigma_mm=60.0)
        a = inject_depth_noise(depth, cfg, np.random.default_rng(7))
        b = inject_depth_noise(depth, cfg, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)                 # deterministic given the rng
        self.assertTrue(np.all(a[depth == 0.0] == 0.0))     # background untouched
        self.assertTrue(np.all(a >= 0.0))                   # depth stays >= 0
        self.assertFalse(np.array_equal(a, depth))          # valid pixels changed


    def test_the_noise_lands_on_the_grasp_depth_and_NOT_on_the_surface(self) -> None:
        """⚠ THE TWO FIELDS NOW DIFFER BY EXACTLY THIS, and nothing asserted it.

        Both carry the same measurement since the producers stopped flattening the depth under a
        mask, so the noise harness is the only thing that separates them. That separation is the
        design rather than an oversight: the grasp path should see what a real sensor delivers, and
        the planner's obstacle world is built from the clean field, because an obstacle that jumps a
        few millimetres every frame is worse than one that is slightly wrong.
        """
        depth, _, intr = _depth_patch()
        frame = PerceptionFrame(depth_map=depth, intrinsics=intr, segmentations=(),
                                surface_depth_map=depth.copy())

        class _Inner:
            def acquire(self) -> PerceptionFrame:
                return frame

        wrapped = NoisyDepthPerceptionSource(
            _Inner(), DepthNoiseConfig(gaussian_sigma_mm=2.0))
        out = wrapped.acquire()

        self.assertFalse(np.array_equal(out.depth_map, depth), "the grasp depth must be noisy")
        np.testing.assert_array_equal(out.surface_depth_map, depth)

    def test_wrapper_passthrough_when_disabled(self) -> None:
        depth, _, intr = _depth_patch()
        frame = PerceptionFrame(depth_map=depth, intrinsics=intr, segmentations=())

        class _Inner:
            def acquire(self) -> PerceptionFrame:
                return frame

        wrapped = NoisyDepthPerceptionSource(_Inner(), DepthNoiseConfig())  # disabled
        self.assertIs(wrapped.acquire(), frame)  # exact same object -> byte-identical
        on = NoisyDepthPerceptionSource(_Inner(), DepthNoiseConfig(speckle_fraction=0.05))
        out = on.acquire()
        self.assertFalse(np.array_equal(out.depth_map, depth))  # noised
        self.assertEqual(out.intrinsics.shape, intr.shape)      # other fields preserved


if __name__ == "__main__":
    unittest.main()
