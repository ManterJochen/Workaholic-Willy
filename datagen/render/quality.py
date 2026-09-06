"""Is this actually a rendered image? One predicate, used by the renderer and the verifier.

Every other check in this package looks at geometry: where objects are, which pixels are whose,
whether a box is the nearest one. None of them looks at the colour image, and without a check that
does, a scene with all three RGB frames entirely zero passes every one of them as ``ok`` while its
depth, its masks and its labelled boxes are all perfectly correct. Only a human opening the folder
sees it.

Depth and colour lag independently, so a stability check on the depth buffer says nothing about the
colour buffer that arrived with it. Colour needs a content check of its own.

The predicate is shared rather than duplicated at each site: when two places must agree about what
"valid" means, they get one function, not two implementations that start identical.

Both conditions are kept because they fail differently. An all-zero buffer trips the lit fraction,
and a uniform grey wash, which is a lens or exposure fault rather than a missing frame, trips the
distinct-level count.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ImageContent", "describe_rgb", "rgb_is_renderable"]

#: A pixel this dark counts as black. Not 0: an 8-bit frame that is almost empty is empty, and a real
#: render has very nearly every pixel above this level.
_BLACK_LEVEL = 8
#: Half the frame. A real render lights very nearly all of it and a missing colour buffer lights none,
#: so the gate sits far below every render and far above the failure, and needs no delicacy.
_MIN_LIT_FRACTION = 0.5
#: A frame that resolves fewer than 8 distinct grey levels is a fill, not an image of anything. A real
#: render of a lit table resolves far more, and the gate is kept low so a flat scene still passes.
_MIN_DISTINCT_LEVELS = 8


@dataclass(frozen=True, slots=True)
class ImageContent:
    """What an RGB frame contains, as the numbers the decision is made on."""

    lit_fraction: float
    distinct_levels: int
    peak: int

    @property
    def renderable(self) -> bool:
        return (
            self.lit_fraction >= _MIN_LIT_FRACTION
            and self.distinct_levels >= _MIN_DISTINCT_LEVELS
        )

    def why_not(self) -> str:
        """Why this frame is not a render, in the numbers; empty when it is one."""
        if self.renderable:
            return ""
        if self.peak == 0:
            return "the frame is entirely zero; no colour buffer arrived"
        if self.lit_fraction < _MIN_LIT_FRACTION:
            return (
                f"only {self.lit_fraction * 100:.2f}% of pixels are above black (level "
                f"{_BLACK_LEVEL}), against {_MIN_LIT_FRACTION * 100:.0f}% required and "
                f"99.81% in the darkest real frame measured"
            )
        return (
            f"the frame resolves {self.distinct_levels} distinct level(s), under the "
            f"{_MIN_DISTINCT_LEVELS} required; a fill, not an image"
        )


def describe_rgb(rgb: np.ndarray | None) -> ImageContent:
    """Measure an RGB frame. ``None`` and empty arrays describe as maximally broken, never crash."""
    if rgb is None or getattr(rgb, "size", 0) == 0:
        return ImageContent(lit_fraction=0.0, distinct_levels=0, peak=0)
    array = np.asarray(rgb)
    # Brightest channel per pixel: a scene lit only in red is still lit, and the failure being caught
    # is a buffer that never arrived, which is zero in every channel at once.
    gray = array.max(axis=2) if array.ndim == 3 else array
    return ImageContent(
        lit_fraction=float(np.count_nonzero(gray > _BLACK_LEVEL)) / float(gray.size),
        distinct_levels=int(np.unique(gray).size),
        peak=int(gray.max()),
    )


def rgb_is_renderable(rgb: np.ndarray | None) -> bool:
    """True when this frame is a real render. The one-line form for call sites that only branch."""
    return describe_rgb(rgb).renderable
