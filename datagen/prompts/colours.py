"""Naming a colour, or honestly refusing to.

A referring expression is ground truth only if a person would agree with it. "The orange cup" is
not ground truth when the cup sits at hue 45 deg, halfway between orange and yellow, because half
the people asked would call it yellow and the benchmark would then be measuring the labeller, not
the model.

That is not hypothetical here. Object colours are drawn uniformly from an RGB box
(``randomization.object_color_min/max``), so the hue distribution is flat: there are no clusters
to name, only a continuum, and a naive nearest-name assignment would hand a confident label to
every boundary case.

So this module names a colour only when the name is safe, and returns ``None`` otherwise. An
unnamed object is not a lost object: it can still be referred to by shape, by size or by position.
What it may not do is carry a colour word nobody can defend.

Categories are the basic colour terms, which is also what makes the German and English prompts
line up: basic terms are the layer of the vocabulary that translates one-to-one, unlike "teal" or
"burgundy".
"""

from __future__ import annotations

import colorsys
from dataclasses import dataclass

__all__ = ["ColourName", "name_colour"]

#: Hue centres in degrees for the chromatic basic terms. Red is deliberately listed twice: the hue
#: circle wraps, and a colour at 355 deg is as red as one at 5 deg.
_HUE_CENTRES: tuple[tuple[float, str, str], ...] = (
    (0.0, "red", "rot"),
    (30.0, "orange", "orange"),
    (60.0, "yellow", "gelb"),
    (120.0, "green", "grün"),
    (200.0, "cyan", "türkis"),
    (240.0, "blue", "blau"),
    (280.0, "purple", "lila"),
    (330.0, "pink", "rosa"),
    (360.0, "red", "rot"),
)
#: How far from the nearest rival centre a hue must sit before the name is safe. The gaps between
#: centres run from 30 deg (red->orange) to 80 deg (green->cyan), so a 12 deg margin refuses a
#: 12 deg band around every transition, roughly the middle third of the narrowest one, which is
#: the region where two people would disagree.
_HUE_MARGIN_DEG = 12.0
#: Below this saturation a colour has no hue worth naming; it is grey, white or black.
_ACHROMATIC_SATURATION = 0.18
#: ...and the band in between is exactly the "is it grey or is it a washed-out blue" argument, so it is
#: refused rather than guessed.
_CHROMATIC_SATURATION = 0.35
#: Value (brightness) cuts for the achromatic names, with an unnamed band between each pair.
_BLACK_VALUE, _DARK_GREY_VALUE, _LIGHT_GREY_VALUE, _WHITE_VALUE = 0.18, 0.30, 0.72, 0.85


@dataclass(frozen=True, slots=True)
class ColourName:
    """A colour word in both languages, and the margin that made it safe to use."""

    en: str
    de: str
    #: Degrees to the nearest rival hue centre, or ``inf`` for the achromatic names. Kept because a
    #: dataset that records why a label was allowed can be re-audited under a stricter rule later.
    margin_deg: float


def _achromatic(saturation: float, value: float) -> ColourName | None:
    if value < _BLACK_VALUE:
        # Value first, and deliberately before the saturation gate. Saturation is a ratio against
        # the brightest channel, so at value 0.04 a one-hundredth difference between channels
        # reads as saturation 0.25: numerically large, visually nothing. Under a saturation-first
        # order a near-black colour such as (0.03, 0.03, 0.04) is refused a name.
        return ColourName("black", "schwarz", float("inf"))
    if saturation > _ACHROMATIC_SATURATION:
        return None
    if _DARK_GREY_VALUE < value < _LIGHT_GREY_VALUE:
        return ColourName("grey", "grau", float("inf"))
    if value > _WHITE_VALUE:
        return ColourName("white", "weiß", float("inf"))
    return None  # in a transition band: not confidently black/grey/white


def name_colour(rgb: tuple[float, float, float]) -> ColourName | None:
    """The colour's basic term in English and German, or ``None`` when no name is defensible.

    ``None`` is a normal outcome, not a failure: with hue drawn uniformly, a large share of
    objects legitimately sit between two names.
    """
    red, green, blue = (float(max(0.0, min(1.0, channel))) for channel in rgb)
    hue, saturation, value = colorsys.rgb_to_hsv(red, green, blue)
    hue *= 360.0

    if saturation <= _ACHROMATIC_SATURATION or value < _BLACK_VALUE:
        return _achromatic(saturation, value)
    if saturation < _CHROMATIC_SATURATION:
        return None  # too washed out to name by hue, too saturated to call grey

    scored = sorted(
        ((min(abs(hue - centre), 360.0 - abs(hue - centre)), en, de)
         for centre, en, de in _HUE_CENTRES),
        key=lambda row: row[0],
    )
    nearest, rival = scored[0], next(row for row in scored[1:] if row[1] != scored[0][1])
    # The margin is the separation between the two candidates, not the distance to the winner: a
    # hue 20 deg from orange and 40 deg from yellow is safely orange, while one 25 deg and 35 deg
    # away is not.
    separation = rival[0] - nearest[0]
    if separation < _HUE_MARGIN_DEG:
        return None
    return ColourName(nearest[1], nearest[2], round(separation, 1))
