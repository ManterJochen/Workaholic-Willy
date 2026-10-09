"""The deterministic rules behind :class:`RuleBasedRouter`: vocabulary in, route out, no model.

The router runs before any weights load and on every pick, so it stays free and auditable: when a
route is wrong, the word that caused it is in one of the vocabularies below. A mistake costs in both
directions, a false simple grasping the wrong object and a false VLM costing seconds and VRAM. The
:class:`~.decision.PromptRouter` seam is where a learned judge would take over.

Several rules can fire on one prompt ("nicht der kaputte Würfel" is non-English, negated and a state
word). The first match wins and the order is most-specific first, so the reason an operator sees is
the most informative one available rather than an accident.

The vocabularies are English and German, the languages the cell is operated in. Any other language is
detected only through non-ASCII letters, so an ASCII-only prompt in, say, Dutch routes simple and
grounds badly. Widening that needs either a language-ID model, which is a weight load to decide
whether to load weights, or a vocabulary per language.

A class list (``"green part | red part"``, ``src.models.vlm.qwen.class_list_prompt``: every rule's kind
of part, or every bin, found in one call, the sorting package of 2026-10-09) is routed description by
description, and goes to the VLM where any one of them needs it. Summed, three colour words of three
plain descriptions would read as one phrase binding three attributes, and two short descriptions as
one long instruction, though each is a phrase the simple path grounds on its own.
"""

from __future__ import annotations

import re
import unicodedata

from .decision import PromptSignals, Route, RouteDecision, RouteReason

__all__ = ["RuleBasedRouter", "analyse", "route", "MAX_SIMPLE_WORDS", "MULTI_ATTRIBUTE_THRESHOLD"]

#: Past this many words a prompt is an instruction rather than a noun phrase. Judgement, not
#: measurement, as :mod:`.decision` says. Six covers the target descriptions used in this cell ("the
#: blue plastic box" is four); "pick up the thing i pointed at before" is eight.
MAX_SIMPLE_WORDS = 6

#: How many attributes before binding them becomes the hard part. GroundingDINO is reliable on "small
#: red cube" and degrades once the phrase carries enough modifiers that which-adjective-goes-with-
#: which-noun starts to matter. At 2 most ordinary prompts would take the VLM route and the fast path
#: would carry nothing.
MULTI_ATTRIBUTE_THRESHOLD = 3

# --- vocabularies ---------------------------------------------------------------------------------
# Tokens that are unambiguously German. Ambiguous ones are excluded: "die" is a German article and an
# English verb, and "in", "an", "so" and "war" collide too. A false NON_ENGLISH silently doubles the
# cost of every English pick, so this set holds only tokens with no English reading at all.
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

_RELATIVE_CLAUSE = frozenset({
    "that", "which", "who", "whose", "where", "welche", "welcher", "welches", "welchen", "wo",
})
_NEGATION = frozenset({
    "not", "no", "without", "except", "neither", "nor", "avoid", "ignore", "skip",
    "nicht", "kein", "keine", "keinen", "keiner", "ohne", "ausser", "weder",
})
_STATE_WORDS = frozenset({
    "broken", "damaged", "cracked", "faulty", "defective", "dented", "bent", "chipped", "torn",
    "intact", "undamaged", "empty", "full", "open", "closed", "dirty", "clean", "wet",
    "kaputt", "kaputte", "kaputten", "kaputter", "kaputtes", "beschaedigt", "beschaedigte",
    "beschaedigten", "defekt", "defekte", "defekten", "heil", "leer", "voll", "offen",
    "geschlossen", "schmutzig", "sauber", "verbogen", "gerissen",
})
_COMPARATIVE = frozenset({
    "largest", "smallest", "biggest", "tallest", "shortest", "longest", "widest", "narrowest",
    "heaviest", "lightest", "nearest", "closest", "furthest", "farthest", "leftmost", "rightmost",
    "topmost", "larger", "smaller", "bigger", "taller", "shorter", "longer", "heavier", "lighter",
    # "next" is absent: alone it is an ordinal, while "next to" is spatial and the commoner phrasing
    # at a bench, so the bigram is handled in _SPATIAL_PHRASES.
    "nearer", "closer", "most", "least", "first", "last",
    "groesste", "groessten", "kleinste", "kleinsten", "hoechste", "laengste", "kuerzeste",
    "schwerste", "leichteste", "naechste", "vorderste", "hinterste", "oberste", "unterste",
    "groesser", "kleiner", "hoeher", "laenger", "schwerer", "leichter", "naeher",
    # Where a part stands among its kind (2026-10-08): "the upper gray cube" singles out one cube of
    # a stack, and a phrase grounder boxes every gray cube for it. The German with the endings its
    # articles leave on it ("der obere", "den oberen", "ein oberer", "ein oberes").
    "upper", "lower", "uppermost", "lowermost", "highest", "lowest",
    "obere", "oberen", "oberer", "oberes", "untere", "unteren", "unterer", "unteres",
})
_SPATIAL = frozenset({
    "above", "below", "beneath", "under", "underneath", "behind", "beside", "between", "inside",
    "outside", "near", "onto", "atop", "adjacent", "opposite", "front",
    "ueber", "unter", "hinter", "neben", "zwischen", "innerhalb", "ausserhalb", "vor", "darunter",
    "darauf", "daneben", "dahinter", "gegenueber",
})
#: Spatial relations that exist only as several words, matched against the joined token stream: their
#: parts are individually harmless, since "next" is an ordinal and "front" and "top" are nouns.
_SPATIAL_PHRASES = (
    "next to", "in front of", "on top of", "close to", "far from", "away from", "to the left of",
    "to the right of",
    # The same relations said shorter (2026-10-08): "the cup left of the bin", "the gray cube on
    # top", "the cube on the left". Each would ground every cup or cube on the simple path.
    "left of", "right of", "on top", "on the left", "on the right",
)

_QUANTIFIER = frozenset({
    "all", "every", "each", "both", "any", "several", "many", "few", "none",
    "alle", "jeder", "jede", "jedes", "jeden", "beide", "mehrere", "viele", "einige", "keines",
})
_CONJUNCTION = frozenset({"and", "or", "plus", "then", "also", "und", "oder", "sowie", "dann", "auch"})

#: Attribute-bearing modifiers the simple path handles individually. Only how many must be bound at
#: once matters here, so the set is colours, sizes and common shapes and materials rather than a full
#: adjective lexicon.
_ATTRIBUTES = frozenset({
    "red", "green", "blue", "yellow", "orange", "purple", "pink", "brown", "black", "white", "grey",
    "gray", "silver", "golden", "transparent", "shiny", "matte", "striped", "spotted",
    "big", "large", "small", "tiny", "huge", "long", "short", "tall", "flat", "thin", "thick",
    "wide", "narrow", "round", "square", "rectangular", "cylindrical", "curved", "pointed",
    "metal", "metallic", "plastic", "wooden", "cardboard", "rubber", "glass", "paper",
    "rot", "rote", "roten", "roter", "gruen", "gruene", "gruenen", "blau", "blaue", "blauen",
    "gelb", "gelbe", "gelben", "schwarz", "schwarze", "schwarzen", "weiss", "weisse", "weissen",
    "gross", "grosse", "grossen", "klein", "kleine", "kleinen", "lang", "lange", "kurz", "kurze",
    "rund", "runde", "flach", "flache", "duenn", "duenne", "dick", "dicke", "breit", "schmal",
    "metall", "metallene", "kunststoff", "holz", "holzerne", "glas", "papier", "pappe",
})

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: German umlauts and eszett folded to their ASCII digraphs, so the vocabularies above can be written
#: in plain ASCII and still match "Würfel" / "beschädigt" / "größte" as typed.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def _tokens(prompt: str) -> tuple[list[str], bool]:
    """``(folded lowercase words, sawNonAscii)``.

    The non-ASCII flag is the only cross-language signal here: a prompt carrying accented letters is
    not English, whatever language it is. It is computed before folding, which is what would erase it.
    """
    lowered = prompt.lower()
    saw_non_ascii = any(
        not char.isascii() and unicodedata.category(char).startswith("L") for char in lowered
    )
    return _WORD_RE.findall(lowered.translate(_FOLD)), saw_non_ascii


def analyse(prompt: str) -> PromptSignals:
    """Measure a prompt. Pure, total, and free: no model, no I/O, no exceptions for bad input."""
    words, saw_non_ascii = _tokens(prompt)
    unique = set(words)
    joined = " ".join(words)
    return PromptSignals(
        words=len(words),
        attributes=sum(1 for word in words if word in _ATTRIBUTES),
        conjunctions=sum(1 for word in words if word in _CONJUNCTION),
        non_english=saw_non_ascii or bool(unique & _GERMAN_MARKERS),
        has_relative_clause=bool(unique & _RELATIVE_CLAUSE),
        has_negation=bool(unique & _NEGATION),
        has_state_word=bool(unique & _STATE_WORDS),
        has_comparative=bool(unique & _COMPARATIVE),
        has_spatial_relation=bool(unique & _SPATIAL) or any(p in joined for p in _SPATIAL_PHRASES),
        has_quantifier=bool(unique & _QUANTIFIER),
    )


def _reason_for(signals: PromptSignals) -> RouteReason | None:
    """The first complex property this prompt has, most-specific first, or ``None`` for the fast path."""
    if signals.non_english:
        return RouteReason.NON_ENGLISH
    if signals.has_relative_clause:
        return RouteReason.RELATIVE_CLAUSE
    if signals.has_negation:
        return RouteReason.NEGATION
    if signals.has_state_word:
        return RouteReason.STATE_WORD
    if signals.has_comparative:
        return RouteReason.COMPARATIVE
    if signals.has_spatial_relation:
        return RouteReason.SPATIAL_RELATION
    if signals.has_quantifier:
        return RouteReason.QUANTIFIER
    if signals.conjunctions:
        return RouteReason.CONJUNCTION
    if signals.attributes >= MULTI_ATTRIBUTE_THRESHOLD:
        return RouteReason.MULTI_ATTRIBUTE
    if signals.words > MAX_SIMPLE_WORDS:
        return RouteReason.TOO_LONG
    return None


def _route_one(prompt: str) -> RouteDecision:
    """Route a prompt of one description: its signals, and the first complex property they show."""
    signals = analyse(prompt)
    if signals.words == 0:
        # Routing is not validation: an empty prompt is a caller bug, and the expensive route would
        # hide it. The simple path fails on it loudly and cheaply.
        return RouteDecision(Route.SIMPLE, RouteReason.EMPTY_PROMPT, signals)
    reason = _reason_for(signals)
    if reason is None:
        return RouteDecision(Route.SIMPLE, RouteReason.PLAIN_NOUN_PHRASE, signals)
    return RouteDecision(Route.VLM, reason, signals)


def route(prompt: str) -> RouteDecision:
    """Route one prompt. The module-level entry point; :class:`RuleBasedRouter` wraps it for the seam.

    A class list is routed description by description (``src.models.vlm.qwen.classes_of``): the
    decision of the first description, in the list's order, that needs the VLM, its reason and its
    signals; where none does, the simple path, with the whole list's signals. A prompt of one
    description is routed as it always was.
    """
    # Imported here: ``vlm.command`` imports this package, and the VLM's package imports no model at import.
    from src.models.vlm.qwen import classes_of  # noqa: PLC0415

    descriptions = classes_of(prompt)
    if not descriptions:
        return _route_one(prompt)
    for description in descriptions:
        decision = _route_one(description)
        if decision.is_vlm:
            return decision
    signals = analyse(prompt)
    reason = RouteReason.PLAIN_NOUN_PHRASE if signals.words else RouteReason.EMPTY_PROMPT
    return RouteDecision(Route.SIMPLE, reason, signals)


class RuleBasedRouter:
    """The default :class:`~.decision.PromptRouter`: stateless, deterministic, model-free."""

    __slots__ = ()

    def route(self, prompt: str) -> RouteDecision:
        return route(prompt)
