"""The two readings the command reader takes of a phrase: is it English, and does it carry a quantifier.

Vocabulary in, two flags out, no model: when a phrase is read wrong, the word that caused it is in one of the lists
below. The command reader (``src/models/vlm/command.py``) asks both of every phrase it would hand the detector, and
asks the model again where a phrase is not English or carries a quantifier ("all screws"): the detector is asked in
English, and "all" belongs in the task's scope.

The vocabularies are English and German, the languages the cell is operated in. Any other language is detected only
through non-ASCII letters, so an ASCII-only phrase in, say, Dutch reads as English. Widening that needs either a
language-ID model or a vocabulary per language.

The lists are what is left of the prompt router's (removed with GroundingDINO on 2026-10-17), word for word, so a
phrase reads as it did before.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

__all__ = ["PromptSignals", "analyse"]

# --- vocabularies ---------------------------------------------------------------------------------
# Tokens that are unambiguously German. Ambiguous ones are excluded: "die" is a German article and an
# English verb, and "in", "an", "so" and "war" collide too. A false reading as German asks the model
# again for an English phrase that was English, so this set holds only tokens with no English reading.
_GERMAN_MARKERS = frozenset({
    "der", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer", "eines",
    "und", "oder", "nicht", "kein", "keine", "keinen", "alle", "jeder", "jede", "jedes",
    "welche", "welcher", "welches", "welchen", "mit", "ohne", "auf", "unter", "neben", "hinter",
    # "war", "hat" and "leg" are absent: each is also an ordinary English word ("the war memorial").
    "zwischen", "links", "rechts", "oben", "unten", "ist", "sind", "waren", "haben",
    "greif", "greife", "nimm", "hole", "lege", "bitte", "dann", "noch", "auch",
    "kaputt", "kaputte", "kaputten", "kaputter", "beschaedigt", "beschaedigte", "defekt", "defekte",
    "wuerfel", "teil", "teile", "objekt", "objekte", "kiste", "kisten", "becher", "schachtel",
    "rote", "roten", "roter", "gelbe", "gelben", "blaue", "blauen", "gruene", "gruenen",
    "schwarze", "schwarzen", "weisse", "weissen", "grosse", "grossen", "kleine", "kleinen",
    "groesste", "groessten", "kleinste", "kleinsten", "linke", "rechte", "vordere", "hintere",
})

_QUANTIFIER = frozenset({
    "all", "every", "each", "both", "any", "several", "many", "few", "none",
    "alle", "jeder", "jede", "jedes", "jeden", "beide", "mehrere", "viele", "einige", "keines",
})

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: German umlauts and eszett folded to their ASCII digraphs, so the vocabularies above can be written
#: in plain ASCII and still match "Würfel" / "beschädigt" / "größte" as typed.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})


@dataclass(frozen=True, slots=True)
class PromptSignals:
    """What :func:`analyse` read off a phrase. Typed, not a bare dict, because a reading is logged."""

    #: A German word of :data:`_GERMAN_MARKERS` stands in it, or a letter outside ASCII does.
    non_english: bool
    #: A quantifier stands in it: all, every, both, alle, beide, ...
    has_quantifier: bool

    def to_dict(self) -> dict[str, bool]:
        """Flat JSON-safe mapping, for a log line."""
        return dict(asdict(self))


def _tokens(prompt: str) -> tuple[list[str], bool]:
    """``(folded lowercase words, saw a letter outside ASCII)``.

    The non-ASCII flag is the only cross-language signal here: a phrase carrying accented letters is
    not English, whatever language it is. It is computed before folding, which is what would erase it.
    """
    lowered = prompt.lower()
    saw_non_ascii = any(
        not char.isascii() and unicodedata.category(char).startswith("L") for char in lowered
    )
    return _WORD_RE.findall(lowered.translate(_FOLD)), saw_non_ascii


def analyse(prompt: str) -> PromptSignals:
    """Read a phrase. Pure, total, and free: no model, no I/O, no exceptions for bad input.

    Args:
        prompt (str): The phrase, as the reader would hand it to the detector.

    Returns:
        PromptSignals: Whether it is not English, and whether it carries a quantifier.
    """
    words, saw_non_ascii = _tokens(prompt)
    unique = set(words)
    return PromptSignals(
        non_english=saw_non_ascii or bool(unique & _GERMAN_MARKERS),
        has_quantifier=bool(unique & _QUANTIFIER),
    )
