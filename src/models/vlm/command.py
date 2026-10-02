"""The command reader: an operator's sentence, German or English, read by the VLM into a task card.

    from src.models.vlm import understand

    reading = understand("Leg den grünen Würfel in die blaue Kiste", ask=ask, poses={"Ablage links": "ablage_links"})
    reading.object.phrase    # 'green cube': the English phrase the detector grounds
    reading.place.phrase     # 'blue bin': a place the camera finds (or reading.place_pose, a taught pose's NAME)
    reading.scope            # 'once' (or 'until_empty')
    print(reading)           # the whole reading, as ASCII text

The owner reads commands only through the VLM (owner decision 22): no word list reads the sentence. The whole
sentence goes to the model as a text-only question, with one fixed instruction (:data:`COMMAND_INSTRUCTION`) and
the taught poses as spoken label -> name pairs, so a spoken "Ablage links" comes back as its name ``ablage_links``.
The answer must be one JSON object with exactly the instruction's keys (:class:`CommandAnswer`). The word lists
below only check that answer.

**Hard checks, one corrective retry.** An answer that holds no JSON object, misses or adds a key, names a scope
other than the two, or sets a place and a pose that contradict each other is asked again once, saying what was
wrong. So is a phrase the detector would get that is not English or that carries a quantifier ("all screws"): the
router's own analysis (:func:`src.models.routing.analyse`) must find it English with no quantifier, none of the
cell's German words (:data:`GERMAN_WORDS`; "schrauben" and "mutter" pass the router as English) may stand in it,
and in the first answer to a German command it must not be made of the command's own words alone: that phrase
was copied, not translated. A word English shares ("Box", :data:`SHARED_WORDS`) or a code ("M6") is no copy, and
a copied word the model gives again after the retry is taken as a loanword. A second bad answer is "not
understood" and fills in nothing, because a half-read card would look like a reading.

**A place that is a pose.** A place whose own words are a taught pose's label and nothing more ("auf Ablage
links") is that pose: the card gets the pose, the camera searches for nothing, and no second question is asked
(:func:`_place_as_pose`). A place without its own words is never taken for a pose, so beside a pose it is a
contradiction and asked again. A place whose own words are Home is no place: Home is where the arm returns.

**Read narrowly.** A phrase that is only a word for any part ("part", "everything": :data:`ANY_PART_WORDS`) names
no part, so the card asks: on a real cell an empty phrase takes the operator's tick (owner decision Q7 A+). A
counted command ("nimm drei Würfel") is read as scope ``once`` with the note that a count is not supported: the
card errs toward fewer motions.

**Soft notes.** An answer that passes is understood, and what the card should mark "bitte prüfen" is noted, in
the order of :data:`NOTE_ORDER`: words the sentence does not contain, a pose nobody taught (or Home as a place), a
count the task cannot honour, and that a retry was needed.

**Reading moves nothing.** :func:`understand` calls ``ask`` and nothing else, and this module imports no robot,
camera or console module: it never starts a run, never sets a detector prompt and never touches the cell. A
sentence read as "stop" stops nothing either; the card points at the stop buttons. :func:`read_command` is the
console's door: the availability rule first (:func:`~src.models.vlm.availability.reader_availability`, owner
decision Q8 A), then this reader through the process's one VLM copy.

No torch at import: the model is reached through ``ask``, and the holder that builds one loads nothing until asked.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, ValidationError

from src.models.constants import MODELS_LOG_DIR, VLM_COMMAND_LOG_FILE
from src.models.routing import analyse
from src.utility.log_cfg import create_logger

from .availability import (
    CommandRefused,
    VlmNotLoadedError,
    VlmUnavailableError,
    reader_availability,
    to_ascii,
)
from .holder import VlmHolder, shared_vlm

__all__ = [
    "ANY_PART_WORDS",
    "Ask",
    "COMMAND_EXAMPLES",
    "COMMAND_INSTRUCTION",
    "COMMAND_MAX_NEW_TOKENS",
    "CommandAnswer",
    "CommandIntent",
    "CommandReading",
    "CommandScope",
    "EXAMPLE_POSES",
    "GERMAN_WORDS",
    "HOME_POSE",
    "MAX_PHRASE_CHARS",
    "MAX_SENTENCE_CHARS",
    "NOTE_ORDER",
    "PhraseReading",
    "ReadingNote",
    "SHARED_WORDS",
    "extract_command_object",
    "read_command",
    "understand",
]

_LOG = create_logger(__name__, log_file=VLM_COMMAND_LOG_FILE, log_dir=MODELS_LOG_DIR)

CommandIntent: TypeAlias = Literal["task", "stop", "none"]
CommandScope: TypeAlias = Literal["once", "until_empty"]
#: A soft note on an understood command. The same strings are the console's ``CommandNote`` codes.
ReadingNote: TypeAlias = Literal[
    "object_not_in_sentence", "place_not_in_sentence", "pose_unknown", "count_not_supported", "retried",
]
#: The one order notes come in, whichever check raised them first.
NOTE_ORDER: Final[tuple[ReadingNote, ...]] = (
    "object_not_in_sentence", "place_not_in_sentence", "pose_unknown", "count_not_supported", "retried",
)

#: ``ask(system, user) -> answer``: one text question to the model. The holder's
#: :class:`~src.models.vlm.holder.TextAsker` is one; a test scripts its own.
Ask: TypeAlias = Callable[[str, str], str]

#: The return pose every cell has: ``robot.home_joint_positions``. No taught pose may be called this.
HOME_POSE: Final[str] = "home"
HOME_LABEL: Final[str] = "Home"
#: A phrase the detector gets is at most this long: the task's ``object`` field takes no more.
MAX_PHRASE_CHARS: Final[int] = 80
#: A sentence is at most this long, as the console's ``CommandIn.text``.
MAX_SENTENCE_CHARS: Final[int] = 500
#: The raw answer kept on a reading, for the tech view.
RAW_LIMIT: Final[int] = 500
#: Room for the longest answer the instruction asks for (about 70 tokens with long German phrases) and a margin.
#: Greedy decoding stops at the end of the object, so the bound costs nothing on a good answer.
COMMAND_MAX_NEW_TOKENS: Final[int] = 160

# --- the words that check an answer (never a reading of the sentence: owner decision 22) ---------------------------

_GERMAN_COLOURS: Final = ("rot", "blau", "gelb", "gruen", "schwarz", "weiss", "grau", "braun", "lila")

#: German words, folded to ASCII, that no English phrase for the detector uses and that the router's vocabulary
#: (:mod:`src.models.routing.rules`) does not hold: the parts, containers, colours, sizes and materials of a
#: workshop table. Without them "schrauben", "mutter" or "blau kasten" pass the router as English and go to
#: GroundingDINO, which grounds German badly.
GERMAN_WORDS: Final[frozenset[str]] = frozenset({
    "schraube", "schrauben", "mutter", "muttern", "scheibe", "scheiben", "unterlegscheibe", "unterlegscheiben",
    "beilagscheibe", "beilagscheiben", "bolzen", "nagel", "naegel", "feder", "federn", "stift", "stifte",
    "kugel", "kugeln", "zylinder", "wuerfeln", "klotz", "kloetze", "platte", "platten", "rohr", "rohre",
    "stueck", "stuecke", "bauteil", "bauteile", "werkstueck", "werkstuecke", "teilchen", "halterung",
    "halterungen", "winkel", "schiene", "schienen", "leiste", "leisten", "blech", "bleche", "kabel", "stecker",
    "gehaeuse", "spule", "spulen", "rolle", "rollen", "deckel", "flasche", "flaschen", "dosen", "tasse", "tassen",
    "glaeser", "werkzeug", "werkzeuge", "zange", "zangen", "schluessel", "buerste", "buersten",
    "kugelschreiber", "kasten", "kaesten", "schachteln", "karton", "kartons", "korb", "koerbe", "behaelter",
    "schale", "schalen", "wanne", "wannen", "tablett", "tabletts", "tuete", "tueten", "beutel", "ablage", "ablagen",
    "klein", "kleiner", "kleines", "grosser", "grosses", "metall", "metallisch", "metallische", "metallischen",
    "holz", "hoelzern", "hoelzerne", "hoelzernen", "kunststoff", "plastik", "glas", "gummi", "pappe", "papier",
    "stahl", "eisen", "kupfer", "alu",
    *(colour + ending for colour in _GERMAN_COLOURS for ending in ("", "e", "en", "er", "es", "em")),
})

#: Words German shares with English, which a translation may keep from a German command ("in die Box" -> "box").
SHARED_WORDS: Final[frozenset[str]] = frozenset({
    "box", "ball", "ring", "block", "clip", "clips", "tape", "kit", "set", "bit", "bits", "pin", "pins", "pad",
    "tube", "stick", "spray", "container", "laptop", "tablet", "hammer", "magnet", "motor", "sensor", "adapter",
    "filter", "chip", "chips", "puck", "display",
})

#: Words for any part at all. A phrase that is only one of them grounds whatever the camera sees, bin walls
#: included, so it names no part and the card asks (owner decision Q7 A+).
ANY_PART_WORDS: Final[frozenset[str]] = frozenset({
    "part", "parts", "object", "objects", "item", "items", "thing", "things", "piece", "pieces", "component",
    "components", "workpiece", "workpieces", "anything", "everything", "something", "stuff",
})

#: What may stand before a pose's label in a place's own words and leave it that pose: "auf Ablage links", "to the
#: Wartepose". Never a word that places a spot beside it ("neben", "next"): that is a place the camera finds.
_BEFORE_A_POSE: Final[frozenset[str]] = frozenset({
    "auf", "aufs", "in", "ins", "im", "an", "am", "ans", "zu", "zur", "zum", "nach", "bei", "beim",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer",
    "on", "onto", "into", "to", "at", "the", "a",
})


# --- the instruction ----------------------------------------------------------------------------------------------


def _example(**fields: Any) -> dict[str, Any]:
    answer: dict[str, Any] = {
        "intent": "task", "object": "", "object_said": None, "place": None, "place_said": None,
        "place_pose": None, "scope": "once", "count": None, "return_to": None,
    }
    answer.update(fields)
    return answer


#: The poses the examples use, label -> name. Names follow the pose name rule (an ASCII identifier); the label is
#: what an operator says.
EXAMPLE_POSES: Final[tuple[tuple[str, str], ...]] = (("Ablage links", "ablage_links"), ("Wartepose", "wartepose"))

#: The instruction's examples, each a sentence and the answer the reader accepts for it as it stands.
COMMAND_EXAMPLES: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    ("nimm den grünen Würfel und leg ihn in die blaue Kiste",
     _example(object="green cube", object_said="den grünen Würfel", place="blue bin",
              place_said="in die blaue Kiste")),
    ("alle Schrauben in die Kiste",
     _example(object="screw", object_said="Schrauben", place="bin", place_said="in die Kiste", scope="until_empty")),
    ("räum die Kiste aus", _example(scope="until_empty")),
    ("pick the red cylinder and go to Wartepose",
     _example(object="red cylinder", object_said="the red cylinder", return_to="wartepose")),
    ("leg den Becher auf Ablage links", _example(object="cup", object_said="den Becher", place_pose="ablage_links")),
    ("leg die Mutter auf Ablage links und fahr danach in die Wartepose",
     _example(object="nut", object_said="die Mutter", place_pose="ablage_links", return_to="wartepose")),
    ("nimm zwei Schrauben", _example(object="screw", object_said="Schrauben", count=2)),
    ("Stopp!", _example(intent="stop")),
    ("wie spät ist es?", _example(intent="none")),
)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _poses_line(pairs: tuple[tuple[str, str], ...]) -> str:
    """``"Ablage links" -> "ablage_links", ...``: both sides quoted, so a label cannot change the question."""
    return ", ".join(f"{_json(label)} -> {_json(name)}" for label, name in pairs)


#: The system message of every command question. Fixed, so it can be compared and cached; the poses and the
#: command travel in the user message.
COMMAND_INSTRUCTION: Final[str] = "\n".join([
    "You read ONE spoken or typed command for a pick-and-place robot. The command may be German or English.",
    'The user message gives POSES, the robot\'s taught poses as "spoken label" -> "name" pairs, and then the '
    "COMMAND.",
    "Answer with ONLY one JSON object on one line, no prose, no code fence, with exactly these keys:",
    '{"intent": "task"|"stop"|"none", "object": "<english noun phrase or empty>", '
    '"object_said": "<the command\'s own words for it, or null>", "place": "<english noun phrase or null>", '
    '"place_said": "<the command\'s own words for it, or null>", "place_pose": "<a name from POSES, or null>", '
    '"scope": "once"|"until_empty", "count": <a whole number, or null>, '
    '"return_to": "<a name from POSES, or null>"}',
    "Rules:",
    "- object: what to pick, as 1 to 4 lowercase English words, singular, no article, keeping colour, size and "
    'material ("den grünen Würfel" -> "green cube"). Always English, never German. Empty "" when the command '
    'names no kind of thing ("räum die Kiste aus", "pick anything").',
    "- object_said / place_said: the command's own words for it, copied exactly as they appear in the command. "
    "Never invent an object or a place the command does not name.",
    "- place_pose: when the command says to put the part on or into one of the POSES (by its spoken label or its "
    "name), that pose's NAME copied exactly; otherwise null. One of the POSES is never a place: then place and "
    "place_said are null.",
    "- return_to: when the command says where the robot goes AFTERWARDS (danach, dann, anschließend, zurück, "
    "afterwards, then go, go back), the NAME of that pose from POSES; otherwise null. A pose the robot only goes "
    "to afterwards is never place_pose.",
    "- place: where the part goes when the command names a container or a spot the camera must find that is not "
    'one of the POSES, in English ("in die blaue Kiste" -> "blue bin"); otherwise null. place and place_pose are '
    "never both set.",
    '- scope: "until_empty" when the command says all, every, alle, jede(n), uses a plural for the parts, or says '
    'to empty or clear something (ausräumen, leer machen); otherwise "once". Never put all/alle into object. When '
    'the command names a number of parts, scope is "once" and count is that number.',
    '- count: the number of parts when the command names one ("zwei Schrauben" -> 2, "three cubes" -> 3); '
    "otherwise null.",
    '- intent: "stop" for stop, stopp, halt, anhalten; "none" for greetings, questions, or anything that is not a '
    'pick command; otherwise "task".',
    f"Examples, with POSES: {_poses_line(EXAMPLE_POSES + ((HOME_LABEL, HOME_POSE),))}",
    *(f"{sentence} -> {_json(answer)}" for sentence, answer in COMMAND_EXAMPLES),
])


# --- the answer ---------------------------------------------------------------------------------------------------


class CommandAnswer(BaseModel):
    """The JSON object the model must answer: exactly these keys, every one present, nothing else.

    Strict, so a number is no string and a string no number: ``"count": "3"`` is asked again rather than read.
    ``object`` and ``place`` may be null or empty, and both mean "none named".
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    intent: CommandIntent
    object: str | None
    object_said: str | None
    place: str | None
    place_said: str | None
    place_pose: str | None
    scope: CommandScope
    count: int | None
    return_to: str | None


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_command_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's answer, or ``None``. Never raises.

    Object-first: a command is one object, so the answer is scanned for the first ``{`` that opens a whole one,
    whatever comes before or after it (prose, a code fence, a second object, a doubled opening brace). An array is
    not a command, though an object inside one is still found.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1), text] if fenced else [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        while start != -1:
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except ValueError:
                value = None
            if isinstance(value, dict):
                return value
            start = candidate.find("{", start + 1)
    return None


def _schema_problem(error: ValidationError) -> str:
    """What was wrong with the keys, in words the model can act on; at most three findings."""
    parts: list[str] = []
    for item in error.errors()[:3]:
        location = item.get("loc") or ("answer",)
        key = str(location[0])
        kind = item.get("type", "")
        if kind == "missing":
            parts.append(f"key {key!r} is missing")
        elif kind == "extra_forbidden":
            parts.append(f"key {key!r} is not one of the keys")
        else:
            parts.append(f"key {key!r}: {item.get('msg', 'not valid')}")
    return "the JSON does not match the keys: " + "; ".join(parts)


def _place_as_pose(answer: CommandAnswer, offered: "_Offered") -> str | None:
    """The taught pose a place really names, or ``None`` when it is a place the camera finds.

    Measured on Qwen3-VL-4B (2026-10-01) with the first wording of this instruction: for "Nimm alle Schrauben und
    leg sie auf Ablage links" it answered place "left shelf", place_said "auf Ablage links" and place_pose
    ``ablage_links`` together, and gave the same answer when asked again, so the owner's own sentence was not read.
    The place's own words are the taught pose's label: the answer means the pose, and the camera has nothing to
    search for. The instruction now says so directly, and on the owner's sentences the model leaves place null;
    this stays as the backstop, with or without place_pose beside the place.

    Only the place's OWN words decide, and only when they are the label (or the name) and nothing more, bar a
    leading preposition or article (:meth:`_Offered.pose_said`): "in die Kiste neben Ablage links" is a place beside
    the pose. A place without its own words is never taken for a pose, because the sentence may name the pose only
    as where the arm goes afterwards ("..., fahr danach zur Ablage links"). Words naming one pose while place_pose
    names another are a contradiction, and Home is never a place (:func:`_place_is_home`).
    """
    if not _phrase(answer.place):
        return None
    said = _said(answer.place_said)
    if said is None:
        return None
    named = offered.pose_said(said)
    if named is None or named == HOME_POSE:
        return None
    pose = (answer.place_pose or "").strip()
    if pose and offered.name_of(pose) != named:
        return None
    return named


def _place_is_home(answer: CommandAnswer, offered: "_Offered") -> bool:
    """A place whose own words are Home's, with no other pose beside it. Home is where the arm returns, never where
    a part is let go, so the place (and a place_pose that says Home too) is dropped and noted, not searched for by
    the camera. Measured on Qwen3-VL-4B (2026-10-01): "Leg den Becher auf Home" came back as place "home", place_said
    "auf Home" and place_pose "home", twice, so a second question only costs time."""
    if not _phrase(answer.place):
        return False
    pose = (answer.place_pose or "").strip()
    if pose and offered.name_of(pose) != HOME_POSE:
        return False
    said = _said(answer.place_said)
    return said is not None and offered.pose_said(said) == HOME_POSE


def _words(text: str) -> list[str]:
    return _fold(text).split()


def _not_english(text: str) -> bool:
    """The router's analysis finds it not English, or one of the cell's German words stands in it."""
    return analyse(text).non_english or any(word in GERMAN_WORDS for word in _words(text))


def _copied(phrase: str, sentence: str) -> bool:
    """Every word of ``phrase`` stands in ``sentence``, and not every one is a word English shares or a code: the
    phrase repeats the command's words instead of translating them. The caller passes the sentence without its
    pose labels: a pose's label is a name, in no language."""
    words = _words(phrase)
    said = set(_words(sentence))
    if not words or not all(word in said for word in words):
        return False
    return not all(word in SHARED_WORDS or any(char.isdigit() for char in word) for word in words)


def _phrase_problem(answer: CommandAnswer, offered: "_Offered", sentence: str, *, first: bool) -> str:
    """The first hard problem with the phrases of a task answer, or ``""``.

    ``first`` is the answer to the first question: there a phrase copied from a German command is sent back to be
    translated; in the answer to the corrective question the model kept it on purpose, and it is a loanword.
    """
    if answer.intent != "task":
        return ""  # its phrases are thrown away, so they are not worth a second question
    camera_place = bool(_phrase(answer.place)) and _place_as_pose(answer, offered) is None \
        and not _place_is_home(answer, offered)
    fields = [("object", answer.object)] + ([("place", answer.place)] if camera_place else [])
    own_words = offered.outside_poses(sentence)  # the command without its pose labels, which are names
    german_command = _not_english(own_words)
    for field, value in fields:
        phrase = _phrase(value)
        if not phrase:
            continue
        if len(phrase) > MAX_PHRASE_CHARS:
            return (f"the {field} phrase is longer than {MAX_PHRASE_CHARS} characters: name it in at most "
                    f"4 English words")
        if _not_english(phrase):
            return f"the {field} phrase {phrase!r} is not English: give the {field} in English words"
        if analyse(phrase).has_quantifier:
            return (f"the {field} phrase {phrase!r} carries a quantifier: all/alle/every belong in scope, "
                    f"never in {field}")
        if first and german_command and _copied(phrase, own_words):
            return (f"the {field} phrase {phrase!r} repeats the command's German words: give the {field} in "
                    f"English words (keep a word only where English uses the same word)")
    if camera_place and (answer.place_pose or "").strip():
        return ("place and place_pose are both set: when the command names one of the POSES, place and "
                "place_said are null; otherwise place_pose is null")
    return ""


def _checked(raw: str, offered: "_Offered", sentence: str, *, first: bool) -> tuple[CommandAnswer | None, str]:
    """The answer, or ``None`` and the hard problem that sends it back."""
    payload = extract_command_object(raw)
    if payload is None:
        return None, "the answer holds no JSON object"
    try:
        answer = CommandAnswer.model_validate(payload)
    except ValidationError as exc:
        return None, _schema_problem(exc)
    problem = _phrase_problem(answer, offered, sentence, first=first)
    return (None, problem) if problem else (answer, "")


# --- the sentence and the poses -----------------------------------------------------------------------------------

#: German letters folded to their ASCII spelling, so "Würfel" and "WUERFEL" are the same word to the checks.
_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})
_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)
_WORD = re.compile(r"\w+", re.UNICODE)


def _fold(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.casefold().translate(_FOLD)).split())


def _found(words: str, sentence: str) -> bool:
    """Whether ``words`` stand in ``sentence`` as whole words, whatever their case, umlauts or punctuation."""
    needle = _fold(words)
    return bool(needle) and f" {needle} " in f" {_fold(sentence)} "


def _phrase(value: str | None) -> str:
    return " ".join((value or "").split()).lower()


def _said(value: str | None) -> str | None:
    return " ".join((value or "").split()) or None


def _sentence(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError(f"a command is text, not {type(text).__name__}")
    sentence = " ".join(text.split())
    if not sentence:
        raise ValueError("there is no command to read: the sentence is empty")
    if len(sentence) > MAX_SENTENCE_CHARS:
        raise ValueError(f"a command is at most {MAX_SENTENCE_CHARS} characters, not {len(sentence)}")
    return sentence


@dataclass(frozen=True)
class _Offered:
    """The poses offered to the model as label -> name pairs, Home always among them."""

    pairs: tuple[tuple[str, str], ...]
    #: A folded label -> its name. A label two poses share (after folding) is left out: it cannot say which.
    by_label: Mapping[str, str]
    #: A folded name -> the name.
    by_name: Mapping[str, str]

    @classmethod
    def of(cls, poses: Mapping[str, str]) -> "_Offered":
        pairs: list[tuple[str, str]] = []
        for label, name in poses.items():
            name_text = str(name).strip()
            if not name_text:
                raise ValueError(f"the pose labelled {label!r} has no name; a pose field holds a name")
            pairs.append((" ".join(str(label).split()) or name_text, name_text))
        if not any(name == HOME_POSE for _, name in pairs):
            pairs.append((HOME_LABEL, HOME_POSE))
        by_label: dict[str, str] = {}
        shared: set[str] = set()
        for label, name in pairs:
            key = _fold(label)
            if key in by_label and by_label[key] != name:
                shared.add(key)
            by_label.setdefault(key, name)
        for key in shared:
            del by_label[key]
        return cls(pairs=tuple(pairs), by_label=by_label, by_name={_fold(name): name for _, name in pairs})

    def line(self) -> str:
        return _poses_line(self.pairs)

    def name_of(self, answered: str) -> str | None:
        """The NAME the model meant: a name as offered, a name in another case, or a label. Else ``None``."""
        text = answered.strip()
        if any(text == name for _, name in self.pairs):
            return text
        key = _fold(text)
        return self.by_name.get(key) or self.by_label.get(key)

    def named_in(self, name: str, sentence: str) -> bool:
        """Whether the sentence says this pose: one of its labels, or its name."""
        spoken = [label for label, owner in self.pairs if owner == name] + [name, name.replace("_", " ")]
        return any(_found(words, sentence) for words in spoken)

    def outside_poses(self, sentence: str) -> str:
        """The sentence without the offered poses' labels and names, in its own spelling: "Put the red cylinder on
        Ablage links" is an English command, whatever language its poses are labelled in."""
        words = _WORD.findall(sentence)
        folded = [_fold(word) for word in words]
        spans = {tuple(_fold(label).split()) for label, _ in self.pairs}
        spans |= {tuple(_fold(name).split()) for _, name in self.pairs}
        spans |= {tuple(_fold(name.replace("_", " ")).split()) for _, name in self.pairs}
        longest_first = sorted((span for span in spans if span), key=len, reverse=True)
        kept: list[str] = []
        index = 0
        while index < len(words):
            span = next((s for s in longest_first if tuple(folded[index:index + len(s)]) == s), None)
            if span is None:
                kept.append(words[index])
                index += 1
            else:
                index += len(span)
        return " ".join(kept)

    def pose_said(self, said: str) -> str | None:
        """The pose whose label or name ``said`` is and nothing more, bar leading prepositions and articles
        ("auf Ablage links", "to the Wartepose"); ``None`` for words that say more ("in die Kiste neben Ablage
        links") or name no pose."""
        words = _fold(said).split()
        while words:
            key = " ".join(words)
            named = self.by_label.get(key) or self.by_name.get(key) or self.by_name.get(key.replace(" ", "_"))
            if named is not None:
                return named
            if words[0] not in _BEFORE_A_POSE:
                return None
            words = words[1:]
        return None


def _question(sentence: str, offered: _Offered) -> str:
    return f"POSES: {offered.line()}\nCOMMAND: {sentence}"


def _retry_question(question: str, raw: str, problem: str) -> str:
    previous = " ".join(raw.split())[:300]
    return (f"{question}\n"
            f"Your previous answer was rejected: {problem.rstrip('.')}.\n"
            f"Previous answer: {previous}\n"
            "Answer again with ONLY the corrected JSON object.")


# --- the reading ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PhraseReading:
    """One phrase the reader filled: the English words the detector gets, and the operator's own."""

    #: English, lowercase, no quantifier: what the detector grounds.
    phrase: str
    #: The operator's words for it, as the model copied them from the sentence; ``None`` when it gave none.
    said: str | None
    #: The operator's words stand in the sentence. False shows "bitte prüfen" on the card.
    verified: bool

    def to_dict(self) -> dict[str, Any]:
        return {"phrase": self.phrase, "said": self.said, "verified": self.verified}


@dataclass(frozen=True, slots=True)
class CommandReading:
    """What the reader understood of one sentence. It starts nothing: Start on the card does, after a person looked.

    ``understood`` false means the model's answer was not usable after one corrective retry; ``reason`` says why
    and no field is filled. ``place_pose`` and ``return_to`` hold pose NAMES; ``return_to`` ``None`` is the default
    return (Home). ``scope`` is ``None`` for a sentence that is no task.
    """

    understood: bool
    intent: CommandIntent
    object: PhraseReading | None = None
    place: PhraseReading | None = None
    place_pose: str | None = None
    scope: CommandScope | None = None
    count: int | None = None
    return_to: str | None = None
    notes: tuple[ReadingNote, ...] = ()
    reason: str = ""
    #: The model's last raw answer, at most 500 characters.
    raw: str = ""
    attempts: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    #: This reading loaded the model: the first command a VLM cell read.
    loaded_now: bool = False

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        if not self.understood:
            lines = [f"command not read: {to_ascii(self.reason)}"]
        elif self.intent == "stop":
            lines = ["command read: stop (a sentence stops nothing; the stop buttons do)"]
        elif self.intent == "none":
            lines = ["command read: not a pick command"]
        else:
            lines = ["command read: task", f"  pick    : {self._pick()}", f"  place   : {self._place()}"]
            scope = str(self.scope)
            if self.count is not None and self.count > 1:
                scope += f" (the sentence names {self.count}; a count is not supported)"
            lines.append(f"  scope   : {scope}")
            lines.append(f"  then    : {self.return_to or 'home (the default)'}")
        if self.notes:
            lines.append(f"  notes   : {', '.join(self.notes)}")
        model = f"  model   : {self.model_id or 'unnamed'}, {self.attempts} question(s), {self.latency_ms:.0f} ms"
        lines.append(model + (", loaded for this command" if self.loaded_now else ""))
        if not self.understood and self.raw:
            lines.append(f"  answer  : {to_ascii(' '.join(self.raw.split())[:200])}")
        return "\n".join(lines)

    def _pick(self) -> str:
        if self.object is None:
            return "no kind of part named (the card asks)"
        return _phrase_text(self.object)

    def _place(self) -> str:
        if self.place is not None:
            return f"the camera finds {_phrase_text(self.place)}"
        if self.place_pose is not None:
            return f"the taught pose {self.place_pose}"
        return "the default place pose"

    def to_dict(self) -> dict[str, Any]:
        """The console's command card (``CommandOut``) without the route previews, which the console adds."""
        return {
            "understood": self.understood,
            "intent": self.intent,
            "object": self.object.to_dict() if self.object is not None else None,
            "place": self.place.to_dict() if self.place is not None else None,
            "place_pose": self.place_pose,
            "scope": self.scope,
            "count": self.count,
            "return_to": self.return_to,
            "notes": list(self.notes),
            "reason": self.reason,
            "model": {
                "model_id": self.model_id,
                "latency_ms": self.latency_ms,
                "attempts": self.attempts,
                "loaded_now": self.loaded_now,
            },
            "raw": self.raw,
        }


def _phrase_text(reading: PhraseReading) -> str:
    said = f'"{to_ascii(reading.said)}"' if reading.said is not None else "no words given"
    check = "in the sentence" if reading.verified else "NOT in the sentence: please check"
    return f"{to_ascii(reading.phrase)} (said {said}, {check})"


def _ordered(notes: set[ReadingNote]) -> tuple[ReadingNote, ...]:
    return tuple(note for note in NOTE_ORDER if note in notes)


def _phrase_reading(phrase: str, said_value: str | None, sentence: str) -> PhraseReading:
    said = _said(said_value)
    return PhraseReading(phrase=phrase, said=said, verified=said is not None and _found(said, sentence))


def _reading(answer: CommandAnswer, sentence: str, offered: _Offered, *, retried: bool,
             record: dict[str, Any]) -> CommandReading:
    notes: set[ReadingNote] = {"retried"} if retried else set()
    if answer.intent != "task":
        return CommandReading(understood=True, intent=answer.intent, notes=_ordered(notes), **record)
    target = None
    if (phrase := _phrase(answer.object)) and phrase not in ANY_PART_WORDS:
        target = _phrase_reading(phrase, answer.object_said, sentence)
        if not target.verified:
            notes.add("object_not_in_sentence")
    place = None
    place_pose = _place_as_pose(answer, offered)
    if place_pose is not None:
        if not offered.named_in(place_pose, sentence):
            notes.add("place_not_in_sentence")
    elif _place_is_home(answer, offered):
        notes.add("pose_unknown")  # Home is where the arm returns, never where a part is let go
    elif phrase := _phrase(answer.place):
        place = _phrase_reading(phrase, answer.place_said, sentence)
        if not place.verified:
            notes.add("place_not_in_sentence")
    if place_pose is None and (answer.place_pose or "").strip():
        name = offered.name_of(answer.place_pose or "")
        if name is None or name == HOME_POSE:
            notes.add("pose_unknown")  # Home is where the arm returns, never where a part is let go
        else:
            place_pose = name
            if not offered.named_in(name, sentence):
                notes.add("place_not_in_sentence")
    return_to = None
    if (answer.return_to or "").strip():
        return_to = offered.name_of(answer.return_to or "")
        if return_to is None:
            notes.add("pose_unknown")
    count = answer.count if answer.count is not None and answer.count >= 1 else None
    if count is not None and count > 1:
        notes.add("count_not_supported")
    # A named number of parts is read as one part: the task cannot honour a count, and "until empty" would take
    # more than the sentence asked for.
    scope: CommandScope = "once" if count is not None else answer.scope
    return CommandReading(
        understood=True, intent="task", object=target, place=place, place_pose=place_pose, scope=scope,
        count=count, return_to=return_to, notes=_ordered(notes), **record,
    )


# --- the verbs ------------------------------------------------------------------------------------------------------


def understand(text: str, *, ask: Ask, poses: Mapping[str, str]) -> CommandReading:
    """Read one sentence into a task card, asking the model once, or twice when the first answer fails a check.

    ``ask(system, user)`` puts one text question to the VLM and returns its answer; ``poses`` maps each taught
    pose's spoken label to its name. Calls ``ask`` and nothing else, at most twice. An exception from ``ask`` is
    raised as it is: "I could not ask" is not "I did not understand". A sentence that is empty or longer than
    :data:`MAX_SENTENCE_CHARS` is refused before anything is asked.
    """
    sentence = _sentence(text)
    offered = _Offered.of(poses)
    question = _question(sentence, offered)
    started = time.perf_counter()
    raw = str(ask(COMMAND_INSTRUCTION, question))
    attempts = 1
    answer, problem = _checked(raw, offered, sentence, first=True)
    if answer is None:
        _LOG.info("asking again for %r: %s (first answer: %.200s)", sentence[:120], problem, raw)
        raw = str(ask(COMMAND_INSTRUCTION, _retry_question(question, raw, problem)))
        attempts = 2
        answer, problem = _checked(raw, offered, sentence, first=False)
    record: dict[str, Any] = {
        "raw": raw[:RAW_LIMIT],
        "attempts": attempts,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "model_id": str(getattr(ask, "model_id", "") or ""),
        "loaded_now": bool(getattr(ask, "loaded_now", False)),
    }
    if answer is None:
        reading = CommandReading(
            understood=False, intent="none", notes=("retried",),
            reason=f"the model's answer was not usable after one corrective retry: {problem}", **record,
        )
        _LOG.warning("not read %r after %d question(s): %s (last answer: %.200s)",
                     sentence[:120], attempts, problem, raw)
    else:
        reading = _reading(answer, sentence, offered, retried=attempts > 1, record=record)
        _LOG.info("read %r as %s in %d question(s), %.0f ms, notes %s",
                  sentence[:120], reading.intent, attempts, reading.latency_ms, list(reading.notes))
    return reading


def read_command(
    text: str,
    *,
    models: Any,
    poses: Mapping[str, str],
    weights_present: bool | None = None,
    holder: VlmHolder | None = None,
) -> CommandReading:
    """The console's door: the availability rule, then :func:`understand` through the one VLM copy.

    ``models`` is the models section the cell was built with, and ``weights_present`` whether the checkpoint is on
    this box (see :func:`~src.models.vlm.availability.reader_availability`). Raises :class:`CommandRefused` with
    the console's code when the rule refuses, when a copy of other weights is in the way (``vlm_not_loaded``), or
    when the model cannot be loaded or fails while it answers (``vlm_unavailable``). Never loads the model on a
    cell whose detector is not the VLM, where a person loads it first, and never beside a copy of other weights.
    """
    _sentence(text)  # a sentence that cannot be read is refused before the rule is asked
    held = holder if holder is not None else shared_vlm()
    availability = reader_availability(models, weights_present=weights_present, holder=held)
    if availability.refusal:
        raise CommandRefused(availability.refusal, availability.cause, availability=availability)
    try:
        asker = held.asker_for(availability.vlm, may_load=availability.may_load,
                               max_new_tokens=COMMAND_MAX_NEW_TOKENS)
        return understand(text, ask=asker, poses=poses)
    except VlmNotLoadedError as exc:
        raise CommandRefused("vlm_not_loaded", str(exc), availability=availability) from exc
    except VlmUnavailableError as exc:
        raise CommandRefused("vlm_unavailable", str(exc), availability=availability) from exc
