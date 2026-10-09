"""Whether a part the detector grounded has the colour its phrase names: the phrase's words against the part's pixels.

Asked for "each separate grey cube", the detector on the owner's cell boxed the green and the orange parts on the mat as
well and called every box by the prompt's words (2026-10-07 and 08), and nothing compared a grounded part with the
colour the operator named: the task picked the wrong parts, or failed on them and kept searching. This compares them,
on the part's own pixels, in about 2 ms a part:

* :func:`colour_named` reads the one colour word of a phrase's head, the words before a place word ("grauer Würfel auf
  der schwarzen Matte" names grey; the black is where it lies), English or German. No colour word, or two colours,
  names none, and nothing is checked.
* :func:`judge_colour` judges a mask in CIELAB (OpenCV's 8-bit Lab, L* scaled back to 0-100): the mask eroded by 2 px,
  the pixels darker than L* 20 left out (the mat, a shadow, the edge's bleed) unless black is wanted, and a pixel
  counted as coloured from C* 12. Grey, white and black are read by how much of the part is coloured and how light its
  colourless pixels are; a colour with a hue by how much of the part lies in its hue band. The numbers come from the
  owner's cell: on the 9 home views of 2026-10-07/08 (142 sightings of a part on the mat, the cell's light and auto
  exposure) every part was at least half coloured or at most a fifth, and the 80th percentile of a colourless part's
  lightness read 52-69 on the grey parts and 80-100 on the white ones (one at 79.6).
* A pixel the camera clipped in some but not all of its channels is left out of every count (``clipped=``,
  :class:`ColourClipped`, ``robot.grasping.colour_check_clipped``; ``exclude``, the owner's choice as shipped,
  2026-10-09: "wenn das schon so gut war, dann nutzen wir das doch instant"). The colour camera runs on auto exposure,
  and where the mat around an orange part is dark it clips the part's red channel at 252 to 254: the hue is then read
  off the green channel alone and lands in the yellow band. Measured offline on the 31 looks the cell recorded on
  2026-10-07, by the review's rule and again through this function on the same parts (2026-10-09): orange judged
  right 53 of 89 times with them counted (26 unsure, 10 refused as yellow), 89 of 89 with them left out; the 69 grey
  cubes, 30 white and 99 green sightings and the yellow bin judged exactly as before. Not measured: no red part stood
  in those views, and a red part clips its red channel the same way (the test plan's sort block checks red cubes). A
  pixel clipped in all three channels, glare on a white part, is counted as before.
* An unsure verdict (a share between the rules, a lightness between grey and white, fewer than 50 pixels) is asked of
  the VLM: one colour word for a crop of the box (:func:`ask_colour`, the backend's ``name_colour``), read through the
  same words (:meth:`ColourVerdict.answered`). No answer refuses the part (the owner, 2026-10-08): it stays on the mat.

A refused part keeps its mask, its depth and its place in the planner world under a label no object carries
(:func:`refused_label`), so it stays a neighbour for every check a target gets. The check takes targets away and never
adds one, and it moves nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any, Final, Mapping

import numpy as np

__all__ = [
    "CLIPPED_AT",
    "Colour",
    "ColourCheck",
    "ColourClipped",
    "ColourVerdict",
    "ask_colour",
    "colour_named",
    "colour_of_word",
    "judge_colour",
    "refused_label",
]


class ColourCheck(StrEnum):
    """What a camera source does with a part's colour (``robot.grasping.colour_check``)."""

    #: A part whose pixels contradict the colour its object label names is no target: the owner's choice, 2026-10-08.
    ON = "on"
    #: Judged and logged, and nothing changed: the part stays a target whatever its pixels say.
    LOG = "log"
    #: Not judged.
    OFF = "off"


class ColourClipped(StrEnum):
    """What the colour check does with a pixel the camera clipped in some but not all of its channels
    (``robot.grasping.colour_check_clipped``)."""

    #: Left out of the hue and lightness counts: the owner's choice as shipped, 2026-10-09.
    EXCLUDE = "exclude"
    #: Counted as every other pixel is, as before.
    KEEP = "keep"


class Colour(StrEnum):
    """A colour a phrase can name. Pink and brown are read and never judged on pixels: they are always asked."""

    GREY = "grey"
    WHITE = "white"
    BLACK = "black"
    RED = "red"
    ORANGE = "orange"
    YELLOW = "yellow"
    GREEN = "green"
    CYAN = "cyan"
    BLUE = "blue"
    PURPLE = "purple"
    PINK = "pink"
    BROWN = "brown"


#: The colour words, English and German, folded as :func:`_folded` folds a word (lower case, umlauts and ß spelled
#: out). A German ending is stripped before a word is looked up here ("grauer", "roten", "silberne").
_WORDS: Final[Mapping[str, Colour]] = {
    "grey": Colour.GREY, "gray": Colour.GREY, "silver": Colour.GREY, "grau": Colour.GREY, "silber": Colour.GREY,
    "white": Colour.WHITE, "weiss": Colour.WHITE,
    "black": Colour.BLACK, "schwarz": Colour.BLACK,
    "red": Colour.RED, "rot": Colour.RED,
    "orange": Colour.ORANGE,
    "yellow": Colour.YELLOW, "gelb": Colour.YELLOW,
    "green": Colour.GREEN, "gruen": Colour.GREEN,
    "blue": Colour.BLUE, "blau": Colour.BLUE,
    "purple": Colour.PURPLE, "violet": Colour.PURPLE, "violett": Colour.PURPLE, "lila": Colour.PURPLE,
    "cyan": Colour.CYAN, "turquoise": Colour.CYAN, "tuerkis": Colour.CYAN,
    "pink": Colour.PINK, "rosa": Colour.PINK,
    "brown": Colour.BROWN, "braun": Colour.BROWN,
}
#: The endings a colour word may carry, the German declension and an English plural, tried longest first.
_ENDINGS: Final[tuple[str, ...]] = ("nen", "nes", "ner", "nem", "ne", "en", "em", "er", "es", "e", "n", "s")
#: The words a phrase's head ends before: where a part lies is no colour of the part.
_PLACE_WORDS: Final[frozenset[str]] = frozenset({
    "on", "in", "into", "onto", "inside", "at", "from", "under", "below", "beneath", "above", "over", "beside",
    "behind", "near", "next", "between", "by", "to",
    "auf", "im", "ins", "an", "am", "aus", "neben", "unter", "ueber", "vor", "hinter", "bei", "beim", "zwischen",
    "von", "vom", "zu", "zum", "zur", "nach",
})
_FOLD: Final = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})

#: How far the mask is eroded before its pixels are read, px: the edge bleeds the mat and the neighbours in.
ERODE_PX: Final = 2
#: Pixels darker than this L* are left out (the mat read L* 16), unless black is wanted.
DARK_L: Final = 20.0
#: A pixel is coloured from this chroma C* (grey parts read C* 1-4, green ones 30-32, orange 80).
COLOURED_C: Final = 12.0
#: Fewer pixels than this tell nothing: the verdict is unsure.
MIN_PIXELS: Final = 50
#: A channel at or above this 8-bit value is clipped. A pixel with one or two channels there and not all three has lost
#: the ratio between its channels its hue is read from (:class:`ColourClipped`); the review that measured the rule
#: (2026-10-09) drew the line at 250, where the orange parts' red channel sat at 252 to 254.
CLIPPED_AT: Final = 250
#: Grey, white or black wanted: a coloured share from this one refuses the part, one above :data:`UNSURE_COLOURED`
#: leaves it unsure.
REFUSE_COLOURED: Final = 0.5
UNSURE_COLOURED: Final = 0.2
#: The 80th percentile of a colourless part's L*: grey up to this, white from :data:`WHITE_MIN_P80`, unsure between.
GREY_MAX_P80: Final = 74.0
WHITE_MIN_P80: Final = 80.0
#: Black wanted: the median L* of the part's colourless pixels up to this.
BLACK_MAX_MEDIAN: Final = 25.0
#: A colour with a hue wanted: the part's share in its band from this agrees; up to :data:`BAND_REFUSES` while another
#: band holds :data:`BAND_AGREES` or more (or the part is at most :data:`UNSURE_COLOURED` coloured) refuses it.
BAND_AGREES: Final = 0.5
BAND_REFUSES: Final = 0.15
#: The hue bands, Lab hue in degrees, from the first bound up to the second (red wraps round 0).
HUE_BANDS: Final[Mapping[Colour, tuple[float, float]]] = {
    Colour.RED: (345.0, 45.0),
    Colour.ORANGE: (45.0, 75.0),
    Colour.YELLOW: (75.0, 115.0),
    Colour.GREEN: (115.0, 190.0),
    Colour.CYAN: (190.0, 250.0),
    Colour.BLUE: (250.0, 320.0),
    Colour.PURPLE: (320.0, 345.0),
}
_COLOURLESS: Final[frozenset[Colour]] = frozenset({Colour.GREY, Colour.WHITE, Colour.BLACK})
#: What a verdict says it saw where nothing named a colour: the VLM gave no answer of a colour word.
UNNAMED: Final = "unnamed"


def _folded(text: str) -> list[str]:
    """``text``'s words, lower case, umlauts and ß spelled out."""
    return re.findall(r"[a-z0-9]+", str(text).lower().translate(_FOLD))


def _colour_of(word: str) -> Colour | None:
    """The colour one folded word names, its ending stripped where the word itself is none."""
    found = _WORDS.get(word)
    if found is not None:
        return found
    for ending in _ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= 3:
            found = _WORDS.get(word[: -len(ending)])
            if found is not None:
                return found
    return None


def _one_colour(words: list[str]) -> Colour | None:
    """The one colour ``words`` name; ``None`` for none or for two."""
    named = {colour for colour in (_colour_of(word) for word in words) if colour is not None}
    return next(iter(named)) if len(named) == 1 else None


def colour_named(phrase: str) -> Colour | None:
    """The one colour ``phrase``'s head names, or ``None`` where it names none or two.

    The head is the phrase up to its first place word after the first word ("grey cube on the black mat" names grey,
    "an orange cube" orange), so where a part lies never names its colour. "red and white part" names two, and is
    not checked.
    """
    words = _folded(phrase)
    for index, word in enumerate(words):
        if index and word in _PLACE_WORDS:
            words = words[:index]
            break
    return _one_colour(words)


def colour_of_word(answer: str) -> Colour | None:
    """The colour a VLM's answer names ("Grey.", "silver", "light grey"), read through the phrase's words; ``None``
    where it names none or two."""
    return _one_colour(_folded(answer))


@dataclass(frozen=True)
class ColourVerdict:
    """What a part's pixels say against the colour its phrase names.

    ``agrees`` the part has that colour; ``unsure`` the pixels cannot tell, and the VLM is asked
    (:meth:`answered`); neither, it is refused. ``seen`` is the colour the part was read as (a :class:`Colour` value,
    or :data:`UNNAMED` where nothing named one), ``why`` the numbers behind the verdict, and ``asked`` what the VLM
    answered where it was asked.
    """

    wanted: Colour
    agrees: bool
    unsure: bool
    seen: str
    why: str
    asked: str | None = None

    def answered(self, answer: str) -> "ColourVerdict":
        """The verdict once the VLM answered ``answer`` (``""`` for no answer); an unsure verdict only, any other is
        kept. A colour word of the wanted colour agrees; any other colour, a word that is none, and no answer refuse."""
        if not self.unsure:
            return self
        named = colour_of_word(answer)
        return replace(self, unsure=False, agrees=named is self.wanted,
                       seen=named.value if named is not None else UNNAMED, asked=str(answer))

    def said(self, label: str) -> str:
        """One line for the log: what was wanted, what was seen, and why."""
        if self.agrees:
            head = "agrees"
        elif self.unsure:
            head = "is unsure"
        elif self.seen == UNNAMED:
            head = "refuses it, no colour named"
        else:
            head = f"refuses it as {self.seen}"
        if self.asked is None:
            asked = ""
        elif self.asked:
            asked = f"; the VLM answered {self.asked!r}"
        else:
            asked = "; no colour word came back from the VLM"
        return f"colour check of {label!r}: {self.wanted.value} wanted, the part {head} ({self.why}{asked})"


def refused_label(verdict: ColourVerdict, label: str) -> str:
    """The label a refused part goes by: never an object label, so the label gate passes it by and it stays a
    neighbour ("green object, not grey cube")."""
    seen = verdict.seen if verdict.seen != UNNAMED else "unknown colour"
    return f"{seen} object, not {label}"


def ask_colour(backend: Any, image_bgr: Any, box: Any) -> str:
    """What ``backend`` names the colour of the object in ``box`` (its ``name_colour``), ``""`` where it can name none:
    a backend with no VLM to ask, or a detection with no box. What a question that raises means is the backend's: a
    ``TwoStageBackend`` counts it as a failure and answers ``""``."""
    name = getattr(backend, "name_colour", None)
    if not callable(name) or box is None:
        return ""
    answer = name(image_bgr, box)
    return answer if isinstance(answer, str) else ""


def _in_band(hue: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    low, high = band
    return (hue >= low) & (hue < high) if low < high else (hue >= low) | (hue < high)


def _colourless_seen(lightness: np.ndarray) -> Colour:
    """Grey or white, by the 80th percentile of a colourless part's lightness (white from :data:`WHITE_MIN_P80`)."""
    if lightness.size and float(np.percentile(lightness, 80.0)) >= WHITE_MIN_P80:
        return Colour.WHITE
    return Colour.GREY


def judge_colour(image_bgr: Any, mask: Any, wanted: Colour, *,
                 clipped: ColourClipped | str = ColourClipped.EXCLUDE) -> ColourVerdict:
    """Whether the pixels of ``mask`` in ``image_bgr`` (BGR, uint8) have the colour ``wanted``; see the module's rule.

    Grey, white or black wanted: a part at least half coloured is refused as the hue band most of it lies in, one more
    than a fifth coloured is unsure; otherwise the 80th percentile of the colourless pixels' L* is grey up to 74 and
    white from 80, unsure between, and black is a median L* up to 25. A part mostly darker than L* 20 is black, refused
    where grey or white is wanted. A colour with a hue wanted: half the part in its band agrees; at most 0.15 in it while
    another band holds half, or while the part is at most a fifth coloured, refuses it; anything else is unsure, and so is
    a part of fewer than 50 pixels. Pink and brown are always unsure.

    ``clipped`` (``robot.grasping.colour_check_clipped``): ``exclude``, the default, leaves every pixel with one or two
    channels at :data:`CLIPPED_AT` or over, and not all three, out of every count above, the 50 pixels' included, so a
    part left with fewer is unsure and the VLM is asked, and no part is called black for the pixels left out; the
    verdict's numbers say how many were. ``keep`` counts them as every other pixel, as before.
    """
    import cv2  # noqa: PLC0415 - deferred: the camera source imports this module, and OpenCV is no small import

    wanted = Colour(wanted)
    clipped = ColourClipped(clipped)
    full = np.asarray(mask).astype(bool)
    if not full.any():
        return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED, why="the mask is empty")
    rows, cols = np.nonzero(full)
    pad = ERODE_PX + 1
    y0, y1 = max(0, int(rows.min()) - pad), min(full.shape[0], int(rows.max()) + pad + 1)
    x0, x1 = max(0, int(cols.min()) - pad), min(full.shape[1], int(cols.max()) + pad + 1)
    kernel = np.ones((2 * ERODE_PX + 1, 2 * ERODE_PX + 1), dtype=np.uint8)
    core = cv2.erode(full[y0:y1, x0:x1].astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT,
                     borderValue=0).astype(bool)
    total = int(core.sum())
    if total < MIN_PIXELS:
        return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED,
                             why=f"{total} px after a {ERODE_PX} px erosion, fewer than {MIN_PIXELS}")
    window = np.ascontiguousarray(np.asarray(image_bgr)[y0:y1, x0:x1, :3], dtype=np.uint8)
    lab = cv2.cvtColor(window, cv2.COLOR_BGR2LAB).astype(np.float32)[core]
    lightness = lab[:, 0] * (100.0 / 255.0)
    a, b = lab[:, 1] - 128.0, lab[:, 2] - 128.0
    chroma = np.hypot(a, b)
    hue = (np.degrees(np.arctan2(b, a)) + 360.0) % 360.0
    # A pixel clipped in one or two channels and not all three says nothing true of its hue: the owner's choice leaves
    # it out (ColourClipped.EXCLUDE). Kept, every pixel is readable, as before.
    if clipped is ColourClipped.EXCLUDE:
        channels = window[core]
        readable = ~((channels.max(axis=1) >= CLIPPED_AT) & (channels.min(axis=1) < CLIPPED_AT))
    else:
        readable = np.ones(total, dtype=bool)
    left_out = total - int(np.count_nonzero(readable))
    said_clipped = f", {left_out} clipped left out" if left_out else ""
    if wanted is Colour.BLACK:
        counted = readable
    else:
        counted = readable & (lightness >= DARK_L)
    n = int(counted.sum())
    if n < MIN_PIXELS:
        # The share darker than L* 20 among the pixels read: a clipped pixel is no dark one.
        dark = 1.0 - (n + left_out) / total
        why = (f"{n} of {total} px" if wanted is Colour.BLACK
               else f"{n} of {total} px lighter than L* {DARK_L:g}") + said_clipped
        if dark >= REFUSE_COLOURED and wanted in (Colour.GREY, Colour.WHITE):
            return ColourVerdict(wanted, agrees=False, unsure=False, seen=Colour.BLACK.value, why=why)
        return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED, why=f"{why}, fewer than {MIN_PIXELS}")
    coloured = counted & (chroma >= COLOURED_C)
    share = int(coloured.sum()) / n
    bands = {colour: int((coloured & _in_band(hue, band)).sum()) / n for colour, band in HUE_BANDS.items()}
    most = max(bands, key=lambda colour: bands[colour])
    colourless = lightness[counted & ~coloured]
    numbers = f"{n} px{said_clipped}, coloured {share:.2f}"
    if wanted in _COLOURLESS:
        if share >= REFUSE_COLOURED:
            return ColourVerdict(wanted, agrees=False, unsure=False, seen=most.value,
                                 why=f"{numbers}, {most.value} {bands[most]:.2f}")
        if share > UNSURE_COLOURED:
            return ColourVerdict(wanted, agrees=False, unsure=True, seen=most.value,
                                 why=f"{numbers}, between {UNSURE_COLOURED:g} and {REFUSE_COLOURED:g}")
        p80 = float(np.percentile(colourless, 80.0))
        if wanted is Colour.BLACK:
            median = float(np.median(colourless))
            black = median <= BLACK_MAX_MEDIAN
            return ColourVerdict(wanted, agrees=black, unsure=False,
                                 seen=Colour.BLACK.value if black else _colourless_seen(colourless).value,
                                 why=f"{numbers}, median L* {median:.0f}")
        why = f"{numbers}, p80 L* {p80:.0f}"
        if p80 <= GREY_MAX_P80:
            return ColourVerdict(wanted, agrees=wanted is Colour.GREY, unsure=False, seen=Colour.GREY.value, why=why)
        if p80 >= WHITE_MIN_P80:
            return ColourVerdict(wanted, agrees=wanted is Colour.WHITE, unsure=False, seen=Colour.WHITE.value,
                                 why=why)
        return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED,
                             why=f"{why}, between grey ({GREY_MAX_P80:g}) and white ({WHITE_MIN_P80:g})")
    if wanted not in HUE_BANDS:
        return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED,
                             why=f"{numbers}; {wanted.value} is asked, never judged on pixels")
    own = bands[wanted]
    other = max((colour for colour in bands if colour is not wanted), key=lambda colour: bands[colour])
    why = f"{numbers}, {wanted.value} {own:.2f}, {other.value} {bands[other]:.2f}"
    if own >= BAND_AGREES:
        return ColourVerdict(wanted, agrees=True, unsure=False, seen=wanted.value, why=why)
    if own <= BAND_REFUSES and bands[other] >= BAND_AGREES:
        return ColourVerdict(wanted, agrees=False, unsure=False, seen=other.value, why=why)
    if own <= BAND_REFUSES and share <= UNSURE_COLOURED:
        return ColourVerdict(wanted, agrees=False, unsure=False, seen=_colourless_seen(colourless).value, why=why)
    return ColourVerdict(wanted, agrees=False, unsure=True, seen=UNNAMED, why=why)
