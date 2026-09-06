"""Randomised PBR parameters, authored here rather than taken from a licensed library.

Path tracing computes light transport correctly; on a flat diffuse colour there is almost nothing for
it to compute. Roughness and metallicity are the two levers the renderer applies, and they are what
makes an object hard: polished metal is where depth cameras fail, and a specular highlight is where a
grounding model loses a colour word.

Parameters are drawn per object and biased by family, because "randomised" does not mean "uniform
over nonsense": a cardboard carton is never chrome, and a steel bracket is never matte cardboard. The
bias is what keeps the variation plausible instead of merely wide.

``specular`` and ``clearcoat`` are sampled and recorded in ``scene.json``, but they do not reach the
shader: ``IsaacRenderer._author_material`` gives OmniPBR a colour, a roughness and a metallic value
and nothing else.

The renderer's own material library is not used: those materials carry a third-party licence, and
every rendered image is a derivative work of the materials in it. Materials authored here cost some
realism and keep the dataset's licence answerable in one sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MaterialParams", "sample_material"]


@dataclass(frozen=True, slots=True)
class MaterialParams:
    """A PBR material, in the parameters every renderer's principled shader accepts."""

    base_color_rgb: tuple[float, float, float]
    roughness: float
    metallic: float
    specular: float
    clearcoat: float

    def as_dict(self) -> dict[str, float | tuple[float, float, float]]:
        return {
            "base_color_rgb": self.base_color_rgb,
            "roughness": self.roughness,
            "metallic": self.metallic,
            "specular": self.specular,
            "clearcoat": self.clearcoat,
        }


#: Per-family parameter bands: (roughness range, metallic probability, clearcoat range). Industrial
#: parts are mostly metal and often polished, which is the case a stereo depth camera cannot see and
#: a grounding model misreads.
_BANDS: dict[str, tuple[tuple[float, float], float, tuple[float, float]]] = {
    "primitive":  ((0.15, 0.85), 0.20, (0.0, 0.3)),
    "industrial": ((0.05, 0.45), 0.75, (0.0, 0.2)),
    "packaging":  ((0.55, 0.95), 0.02, (0.0, 0.4)),   # matte board, occasionally a glossy print
    "vessel":     ((0.10, 0.60), 0.10, (0.1, 0.8)),   # plastic and glass: low roughness, high clearcoat
}


def sample_material(
    rng: np.random.Generator, family: str, base_color_rgb: tuple[float, float, float],
) -> MaterialParams:
    """One plausible material for an object of ``family``, keeping its assigned colour.

    The colour is an input, not a draw: it was fixed at layout time because a referring expression may
    name it, and a material sampler that re-rolled it would silently invalidate every colour prompt.
    """
    roughness_range, metallic_probability, clearcoat_range = _BANDS.get(family, _BANDS["primitive"])
    metallic = 1.0 if rng.random() < metallic_probability else 0.0
    # A metal's colour is its reflectance, so a metallic object keeps its hue but loses the washed-out
    # end of the range; a dielectric keeps the colour it was given.
    color = base_color_rgb
    if metallic:
        color = tuple(round(float(min(1.0, channel * 0.6 + 0.35)), 4) for channel in base_color_rgb)  # type: ignore[assignment]
    return MaterialParams(
        base_color_rgb=color,
        roughness=round(float(rng.uniform(*roughness_range)), 4),
        metallic=metallic,
        specular=round(float(rng.uniform(0.3, 0.7)), 4),
        clearcoat=round(float(rng.uniform(*clearcoat_range)), 4),
    )
