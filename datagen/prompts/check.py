"""Does a paraphrase still name the same object? The gate that lets a language model near the labels.

The template layer earns uniqueness structurally: it inspects the scene facts and only emits an
expression when exactly one object matches. A paraphrase is free text, so that check cannot be
re-run as-is, and this module is the honest replacement. It is weaker than the structural check
and says so: it verifies that the discriminating words survived and that no other object's
discriminating words appeared. It cannot understand a sentence.

Three word rules, all fail-closed. Two form gates and a size gate sit beside them, and
``PARAPHRASE_VERDICTS`` carries the full set of eight verdicts.

1. The kind word must survive as a whole word, not as a substring. A model turns "die rote Dose"
   into "die rote Dosenkanne", an invented compound that a substring check accepts and a person
   rejects. Kind synonyms are deliberately not allowed: the kind is what grounds the object, and
   "box" becoming "container" changes what was asked for.
2. The colour may vary within its synonym set. The opposite call, and for the opposite reason:
   colour is the attribute under test, and a good output varies it, as when "the red can" becomes
   "the crimson tin can". A checker without synonyms would reject exactly the successes. What a
   paraphrase may not do is become a different basic colour.
3. No other object's discriminating words may appear. "The can near the green carton" adds a
   second object, and a detector that finds the carton instead is not obviously wrong any more.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["PARAPHRASE_VERDICTS", "ParaphraseVerdict", "check_paraphrase"]

#: English only. German paraphrase is unusable and stays template-only, so there is nothing here
#: to check, and a half-filled German table would imply otherwise.
_COLOUR_SYNONYMS: dict[str, frozenset[str]] = {
    "red": frozenset({"red", "crimson", "scarlet", "ruby"}),
    "orange": frozenset({"orange", "amber"}),
    "yellow": frozenset({"yellow", "golden", "gold"}),
    "green": frozenset({"green", "emerald", "lime"}),
    "cyan": frozenset({"cyan", "turquoise", "teal", "aqua"}),
    "blue": frozenset({"blue", "cerulean", "azure", "navy"}),
    "purple": frozenset({"purple", "violet", "lilac"}),
    "pink": frozenset({"pink", "rose", "magenta"}),
    "black": frozenset({"black", "dark"}),
    "white": frozenset({"white", "ivory"}),
    "grey": frozenset({"grey", "gray", "silver"}),
}
_SIZE_SYNONYMS: dict[str, frozenset[str]] = {
    "large": frozenset({"large", "big", "tall"}),
    "small": frozenset({"small", "little", "tiny"}),
}
#: Every verdict this module can return, so a caller can tabulate rejection reasons without inventing
#: its own vocabulary. A rejection rate broken down by reason is the measurement; a bare count is not.
PARAPHRASE_VERDICTS = (
    "accepted", "kind_lost", "colour_lost", "colour_changed", "size_lost", "foreign_object",
    "not_a_phrase", "empty",
)
#: A referring expression is short. The failures that survive the word rules are all long: hedges,
#: asides and translations appended after the phrase.
_MAX_PHRASE_CHARS = 60
#: Markers of something that is not a noun phrase. A parenthesis or an "or" means the model offered a
#: choice instead of naming an object, and a choice cannot be ground truth.
_NOT_A_PHRASE = ("(", ")", " or ", ":", ";", "depending", "i.e.", "e.g.")


@dataclass(frozen=True, slots=True)
class ParaphraseVerdict:
    verdict: str
    detail: str = ""

    @property
    def accepted(self) -> bool:
        return self.verdict == "accepted"


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z]+", text.lower()))


def check_paraphrase(
    paraphrase: str,
    *,
    kind_word: str,
    colour: str | None,
    size: str | None,
    other_kind_words: Sequence[str],
    other_colours: Sequence[str],
) -> ParaphraseVerdict:
    """Whether ``paraphrase`` still refers to the same object, and if not, which rule it broke."""
    text = paraphrase.strip()
    if not text or len(text) > _MAX_PHRASE_CHARS or "\n" in text:
        return ParaphraseVerdict("empty", f"blank or run-on: {text[:60]!r}")
    # Form, not vocabulary. The word rules alone accept "the metal fastener (or the plastic
    # fastener, depending on the material)", which keeps every required word and is not a
    # referring expression. A phrase that hedges names no object.
    if any(marker in text.lower() for marker in _NOT_A_PHRASE):
        return ParaphraseVerdict("not_a_phrase", f"hedged or parenthetical: {text!r}")
    present = _words(text)

    # A multi-word kind ("blister pack") survives only if every word of it does.
    if not all(word in present for word in _words(kind_word)):
        return ParaphraseVerdict("kind_lost", f"{kind_word!r} is gone from {text!r}")

    if colour is not None:
        allowed = _COLOUR_SYNONYMS.get(colour, frozenset({colour}))
        if not (present & allowed):
            rival = next((name for name, words in _COLOUR_SYNONYMS.items()
                          if name != colour and (present & words)), None)
            if rival:
                return ParaphraseVerdict("colour_changed", f"{colour} became {rival} in {text!r}")
            return ParaphraseVerdict("colour_lost", f"no word for {colour} in {text!r}")

    if size is not None and not (present & _SIZE_SYNONYMS.get(size, frozenset({size}))):
        return ParaphraseVerdict("size_lost", f"no word for {size} in {text!r}")

    # Naming another object in the scene turns one referent into two.
    for other in other_kind_words:
        if other != kind_word and all(word in present for word in _words(other)):
            return ParaphraseVerdict("foreign_object", f"also names a {other} in {text!r}")
    for other in other_colours:
        if other != colour and (present & _COLOUR_SYNONYMS.get(other, frozenset({other}))):
            return ParaphraseVerdict("foreign_object", f"also names something {other} in {text!r}")
    return ParaphraseVerdict("accepted")
