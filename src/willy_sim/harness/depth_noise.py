"""Synthetic sensor depth noise injected onto the noise-free Isaac depth map.

The Isaac renderer is close to noise-free, so the cloud-outlier filter
(:mod:`src.robot.grasping.geometry.filters`) has nothing to remove there. This models the noise the
sim omits: a per-pixel gaussian for surface jitter, plus a speckle spray that offsets a fraction of
pixels far off-surface, which is the kind of outlier the filter exists to remove. The same cell can
then measure whether the filter recovers picks the noise broke.

The depth map is noised before perception, so the cloud build, the outlier filter and the grasp all
see the speckle. The all-zero default does nothing. The noise is synthetic, not a measured
real-sensor profile: it probes whether the filter discriminates, and says nothing about how any
particular camera fails.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.robot.grasping.types.perception import PerceptionFrame


@dataclass(frozen=True, slots=True)
class DepthNoiseConfig:
    """Parameters for :func:`inject_depth_noise`. All-zero (default) == disabled == no-op."""

    gaussian_sigma_mm: float = 0.0   # per-pixel additive gaussian on valid depth (sensor surface jitter)
    speckle_fraction: float = 0.0    # fraction of valid pixels sprayed as outliers (the filter's target)
    speckle_sigma_mm: float = 60.0   # outlier depth offset magnitude (large -> off-surface cloud points)
    seed: int = 1234                 # deterministic; the wrapper advances it per acquire for variety

    def __post_init__(self) -> None:
        for name in ("gaussian_sigma_mm", "speckle_sigma_mm"):
            v = getattr(self, name)
            if not (np.isfinite(v) and v >= 0.0):
                raise ValueError(f"{name} must be finite and >= 0; got {v}")
        if not (np.isfinite(self.speckle_fraction) and 0.0 <= self.speckle_fraction <= 1.0):
            raise ValueError(f"speckle_fraction must be in [0, 1]; got {self.speckle_fraction}")

    @property
    def enabled(self) -> bool:
        return self.gaussian_sigma_mm > 0.0 or (self.speckle_fraction > 0.0 and self.speckle_sigma_mm > 0.0)


def inject_depth_noise(
    depth_mm: np.ndarray, config: DepthNoiseConfig, rng: np.random.Generator
) -> np.ndarray:
    """Return a noisy copy of ``depth_mm``, in mm.

    Only valid pixels, meaning finite and greater than 0, are noised. Background, meaning 0 or
    non-finite, is left alone, so the mask-restricted cloud is unaffected outside the object. The
    gaussian is near-surface jitter, which the filter mostly keeps; the speckle offsets a fraction of
    pixels by ``N(0, speckle_sigma)`` into isolated points far off-surface, which the filter removes.
    """
    out = np.array(depth_mm, dtype=np.float64, copy=True)
    if not config.enabled:
        return out
    flat = out.ravel()
    idx = np.flatnonzero(np.isfinite(flat) & (flat > 0.0))
    if idx.size == 0:
        return out
    if config.gaussian_sigma_mm > 0.0:
        flat[idx] += rng.normal(0.0, config.gaussian_sigma_mm, idx.size)
    if config.speckle_fraction > 0.0 and config.speckle_sigma_mm > 0.0:
        n_speckle = int(round(config.speckle_fraction * idx.size))
        if n_speckle > 0:
            spk = rng.choice(idx, size=n_speckle, replace=False)
            flat[spk] += rng.normal(0.0, config.speckle_sigma_mm, n_speckle)
    flat[idx] = np.clip(flat[idx], 0.0, None)  # depth stays >= 0
    return flat.reshape(out.shape)


class NoisyDepthPerceptionSource:
    """Wraps any ``PerceptionSource`` and sprays synthetic depth noise on each frame's ``depth_map``.

    With the config disabled, or with the source left unwrapped, the frame passes through unchanged.
    The seed advances on every acquire, so a multi-run gate sees different noise realizations while
    staying deterministic given the base seed.
    """

    def __init__(self, inner: Any, config: DepthNoiseConfig) -> None:
        self._inner = inner
        self._config = config
        self._n = 0

    def acquire(self) -> PerceptionFrame:
        frame = self._inner.acquire()
        if not self._config.enabled:
            return frame
        rng = np.random.default_rng(self._config.seed + self._n)
        self._n += 1
        noisy = inject_depth_noise(frame.depth_map, self._config, rng)
        # `depth_map` only, and `surface_depth_map` deliberately left clean. It reads as an
        # oversight and it is the design: the grasp path should see what a real sensor
        # delivers, noise included, while the planner's obstacle world is built from the other
        # field, and an obstacle that jumps a few millimetres every frame is worse than one
        # that is slightly wrong. The two fields carried different content until the producers
        # stopped flattening the depth under a mask. Noise is what separates them now.
        return replace(frame, depth_map=noisy)
