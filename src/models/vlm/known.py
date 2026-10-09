"""Known sentences: the commands a cell hears every day, answered without asking the model (the owner, 2026-10-08).

    from src.models.vlm.known import read_known

    reading = read_known("Alle grauen Würfel in die Gelbe Kiste.", poses={"Ablage links": "ablage_links"})
    reading.object.phrase    # 'gray cube'
    reading.place.phrase     # 'yellow bin'
    reading.model_id         # 'known-sentence': no model was asked, nothing was loaded

The owner, 2026-10-08: "speed first". On the cell the 8B model spends about 140 ms on every token of its answer, so
the morning's "Alle grauen Würfel in die Gelbe Kiste." cost 9.8 s and every "Hallo Willy" 7.2 s. This table answers
a closed grammar of such sentences with the answer the model gives for them, the words the model was measured to
choose (``tests/test_vlm_command.py``, ``_REAL_ANSWERS``), and the reader's own code makes the card of that answer
(:func:`~src.models.vlm.command._checked` and :func:`~src.models.vlm.command._reading`): the same checks, the same
notes, the same words found in the sentence. A sentence outside the grammar, or an answer the checks would send back,
returns ``None``, and the model reads the sentence as before.

**The grammar**, on folded words, with nothing before or after it:

    [verb] (alle|all [the] <colour>? <part> | den|die|the <colour>? <part>) [und leg ihn|sie|es | and put it|them]
        <in|ins|auf|into|on|onto> (<a taught pose's label or name> | die|den|the <colour>? <Kiste|Box|Kasten|bin|box>)

and a sentence that is nothing but a greeting (:func:`~src.models.vlm.command.greets` with no reading at all).

**Only measured words.** Würfel, Schraube(n), Mutter/Muttern, Becher, Zylinder; Kiste, Box, Kasten; rot, blau,
gelb, grün and grau with their endings, which the model gave as red, blue, yellow, green and gray ("graue Box" ->
"gray box"); the English words as they are; the definite article ("den Würfel", "the cube"). One language per
sentence. Everything else goes to the model: a count ("einen Würfel" and "a cube" included), a negation, "danach",
"neben", "bitte", a plural without "alle" (the model read "Nimm die Schrauben" as once), Home as the place, another
word for a part.

**Where to, never where from.** A place must say that the part goes there: a verb that puts (leg, pack, tu, bring,
put, place, move), the words "und leg ihn" or "and put it", "into", "onto" or "ins", or a German accusative article
("in die Kiste", "auf die Ablage links"). "Nimm den Würfel in der Kiste" names where the cube lies, and "Take the
cube in the bin" may; both go to the model. A container is put INTO ("in", "into"); "auf die Kiste" goes to the model.

Pure: no torch, no robot, no camera, no console; it reads a string and the pose labels and calls nothing.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from typing import Any, Final

from .command import (
    _LOG,
    _WORD,
    HOME_POSE,
    KNOWN_SENTENCE_MODEL,
    CommandReading,
    _checked,
    _example,
    _fold,
    _Offered,
    _reading,
    _sentence,
    greets,
)

__all__ = ["KNOWN_COLOURS", "KNOWN_PARTS", "KNOWN_PLACES", "known_answer", "read_known"]

#: The German colour stems, folded, and the English word the model gave for each.
_COLOUR_STEMS: Final[Mapping[str, str]] = {
    "rot": "red", "blau": "blue", "gelb": "yellow", "gruen": "green", "grau": "gray",
}

#: Every colour word the table reads, folded -> (the English word, the language of the word).
KNOWN_COLOURS: Final[Mapping[str, tuple[str, str]]] = {
    **{stem + ending: (english, "de") for stem, english in _COLOUR_STEMS.items()
       for ending in ("", "e", "en", "er", "es", "em")},
    **{english: (english, "en") for english in _COLOUR_STEMS.values()},
}

#: Every word for a part the table reads, folded -> (the English phrase, the number the word says, its language and,
#: in German, its gender). ``both``: the German word is the same in the singular and the plural ("Würfel"), so its
#: article says which ("den Würfel" is one, "die Würfel" are several).
KNOWN_PARTS: Final[Mapping[str, tuple[str, str, str, str]]] = {
    "wuerfel": ("cube", "both", "de", "m"), "becher": ("cup", "both", "de", "m"),
    "zylinder": ("cylinder", "both", "de", "m"),
    "schraube": ("screw", "sg", "de", "f"), "schrauben": ("screw", "pl", "de", "f"),
    "mutter": ("nut", "sg", "de", "f"), "muttern": ("nut", "pl", "de", "f"),
    **{word: (word, "sg", "en", "") for word in ("cube", "screw", "nut", "cup", "cylinder")},
    **{word + "s": (word, "pl", "en", "") for word in ("cube", "screw", "nut", "cup", "cylinder")},
}

#: Every container the table reads, folded -> (the English phrase, the languages of the word, its German gender).
KNOWN_PLACES: Final[Mapping[str, tuple[str, frozenset[str], str]]] = {
    "kiste": ("bin", frozenset({"de"}), "f"), "kasten": ("box", frozenset({"de"}), "m"),
    "box": ("box", frozenset({"de", "en"}), "f"), "bin": ("bin", frozenset({"en"}), ""),
}

#: The verbs a command may open with, folded -> (language, whether the verb itself puts the part somewhere).
_VERBS: Final[Mapping[str, tuple[str, bool]]] = {
    "nimm": ("de", False),
    **{verb: ("de", True) for verb in ("leg", "lege", "pack", "packe", "tu", "tue", "bring", "bringe")},
    "take": ("en", False), "pick": ("en", False),
    **{verb: ("en", True) for verb in ("put", "place", "move")},
}

#: "alle" or "all": every part of the kind, the scope until empty.
_EVERY: Final[Mapping[str, str]] = {"alle": "de", "all": "en"}

#: The articles of one part, folded -> (language, German gender): the definite accusative singular, as a command says
#: it. "einen Würfel" or "a cube" may be a count of one, which the model may answer as one: those go to the model.
_ARTICLES: Final[Mapping[str, tuple[str, str]]] = {"den": ("de", "m"), "die": ("de", "f"), "the": ("en", "")}

#: "und leg ihn" and "and put it": the words that hand the part from the verb that takes it to the place.
_LINKS: Final[Mapping[str, frozenset[tuple[str, str, str]]]] = {
    "de": frozenset((and_, verb, it) for and_ in ("und",) for verb in ("leg", "lege") for it in ("ihn", "sie", "es")),
    "en": frozenset((and_, verb, it) for and_ in ("and",) for verb in ("put", "place") for it in ("it", "them")),
}

#: The prepositions of a place, by language.
_PREPOSITIONS: Final[Mapping[str, frozenset[str]]] = {
    "de": frozenset({"in", "ins", "auf"}),
    "en": frozenset({"in", "into", "on", "onto"}),
}

#: The articles a place may have after its preposition, folded -> German gender: in German only the accusative, which
#: says the part goes there ("in die Kiste"; "in der Kiste" is where it lies).
_PLACE_ARTICLES: Final[Mapping[str, Mapping[str, str]]] = {
    "de": {"die": "f", "den": "m", "das": "n"},
    "en": {"the": ""},
}

#: Prepositions that say "to there" whatever the verb.
_TOWARD: Final[frozenset[str]] = frozenset({"into", "onto", "ins"})


def known_answer(sentence: str, offered: _Offered) -> dict[str, Any] | None:
    """The answer the model gives for ``sentence``, a sentence of the grammar, as the instruction's JSON object; else
    ``None``, and the model reads it.

    ``sentence`` is the reader's (:func:`~src.models.vlm.command._sentence`); ``offered`` the poses as the model is
    handed them. The words the answer says the operator used are cut from the sentence's own words, so the reader
    finds them there.
    """
    words = _WORD.findall(sentence)
    folded = [_fold(word) for word in words]
    count = len(folded)
    at = 0

    verb_language: str | None = None
    puts = False
    if at < count and folded[at] in _VERBS:
        verb_language, puts = _VERBS[folded[at]]
        at += 1
        if folded[at - 1] == "pick" and at < count and folded[at] == "up":
            at += 1

    every = at < count and folded[at] in _EVERY
    if every:
        language, gender = _EVERY[folded[at]], ""
        at += 1
        if language == "en" and at < count and folded[at] == "the":
            at += 1
        start = at  # the model leaves "alle" out of the operator's words for the part ("Schrauben")
    elif at < count and folded[at] in _ARTICLES:
        language, gender = _ARTICLES[folded[at]]
        start = at  # and keeps the article ("den grünen Würfel")
        at += 1
    else:
        return None  # a count, a plural without "alle", "irgendwas", no part at all
    if verb_language is not None and verb_language != language:
        return None

    colour: str | None = None
    if at < count and folded[at] in KNOWN_COLOURS:
        colour, colour_language = KNOWN_COLOURS[folded[at]]
        if colour_language != language:
            return None
        at += 1
    if at >= count or folded[at] not in KNOWN_PARTS:
        return None
    noun, number, noun_language, noun_gender = KNOWN_PARTS[folded[at]]
    at += 1
    if noun_language != language:
        return None
    if every and number == "sg":
        return None
    if not every and (number == "pl" or (language == "de" and noun_gender != gender)):
        return None  # "die Schrauben", "die Würfel": several parts without "alle", which the model read as once
    object_said = " ".join(words[start:at])

    linked = at + 3 <= count and (folded[at], folded[at + 1], folded[at + 2]) in _LINKS[language]
    if linked:
        at += 3
    if at >= count or folded[at] not in _PREPOSITIONS[language]:
        return None
    preposition = folded[at]
    place_from = at
    at += 1
    article: str | None = None
    if preposition != "ins" and at < count and folded[at] in _PLACE_ARTICLES[language]:
        article = folded[at]
        at += 1

    # The answer as the instruction writes it: only what the sentence says, so a scope only for "alle" (2026-10-08).
    answer = _example(object=f"{colour} {noun}" if colour else noun, object_said=object_said,
                      **({"scope": "until_empty"} if every else {}))
    toward = linked or puts or preposition in _TOWARD or (language == "de" and article is not None)

    rest = " ".join(folded[at:])
    pose = (offered.by_label.get(rest) or offered.by_name.get(rest) or offered.by_name.get(rest.replace(" ", "_"))
            if rest else None)
    if pose is not None:
        if pose == HOME_POSE or not toward:
            return None  # Home is where the arm returns; a pose that may be where the part lies is the model's
        return {**answer, "place_pose": pose}

    place_colour: str | None = None
    if at < count and folded[at] in KNOWN_COLOURS:
        place_colour, colour_language = KNOWN_COLOURS[folded[at]]
        if colour_language != language:
            return None
        at += 1
    if at >= count or folded[at] not in KNOWN_PLACES:
        return None
    container, languages, container_gender = KNOWN_PLACES[folded[at]]
    at += 1
    if at != count or language not in languages:
        return None  # words after the place: "danach", "neben", "bitte"
    if preposition not in ("in", "into") or article is None or not toward:
        return None
    if language == "de" and _PLACE_ARTICLES["de"][article] != container_gender:
        return None
    return {**answer, "place": f"{place_colour} {container}" if place_colour else container,
            "place_said": " ".join(words[place_from:at])}


def read_known(text: str, *, poses: Mapping[str, str]) -> CommandReading | None:
    """The reading of a known sentence, made by the reader from the table's answer; ``None`` for any other sentence.

    ``poses`` maps each taught pose's spoken label to its name, as :func:`~src.models.vlm.command.understand` takes
    them. The reading is the one the model's answer would make: the same checks, notes and greeting; it names
    :data:`~src.models.vlm.command.KNOWN_SENTENCE_MODEL` as its model, with no question asked (``attempts`` 0) and
    nothing loaded. Raises what the reader raises for a sentence that is no sentence (empty, too long).
    """
    sentence = _sentence(text)
    started = time.perf_counter()
    offered = _Offered.of(poses)
    answer = _example(intent="none") if greets(sentence, intent=None) else known_answer(sentence, offered)
    if answer is None:
        return None
    raw = json.dumps(answer, ensure_ascii=False)
    checked, problem = _checked(raw, offered, sentence, first=True)
    if checked is None:
        _LOG.warning("known sentence %r: the reader's checks refuse the table's answer (%s), so the model reads it",
                     sentence[:120], problem)
        return None
    reading = _reading(checked, sentence, offered, retried=False, record={
        "raw": raw,
        "attempts": 0,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "model_id": KNOWN_SENTENCE_MODEL,
        "loaded_now": False,
        "greeting": greets(sentence, intent=checked.intent),
    })
    _LOG.info("read %r as %s: a known sentence, no model asked, %.2f ms, notes %s (answer: %s)",
              sentence[:120], reading.intent, reading.latency_ms, list(reading.notes), raw)
    return reading
