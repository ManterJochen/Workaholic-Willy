"""The command reader: an operator's sentence, German or English, read by the VLM into a task card.

    from src.models.vlm import understand

    reading = understand("Leg den grünen Würfel in die blaue Kiste", ask=ask, poses={"Ablage links": "ablage_links"})
    reading.object.phrase    # 'green cube': the English phrase the detector grounds
    reading.place.phrase     # 'blue bin': a place the camera finds (or reading.place_pose, a taught pose's NAME)
    reading.scope            # 'once' (or 'until_empty')
    reading.which            # None, or the one part the sentence singles out: 'the gray cube on top of the other one'
    reading.source           # None, or where the parts lie: 'on the black mat'
    reading.rules            # every rule: rule 0 is the fields above, then one per further kind of a sorting sentence
    print(reading)           # the whole reading, as ASCII text

The owner reads commands through the VLM (owner decision 22). The whole sentence goes to the model as a text-only
question, with one fixed instruction (:data:`COMMAND_INSTRUCTION`) and the taught poses as spoken label -> name
pairs, so a spoken "Ablage links" comes back as its name ``ablage_links``. The answer is one compact JSON object with
only the keys the sentence fills, ``intent`` always (:class:`CommandAnswer`): on the cell every token of the answer
costs about 140 ms, and the nulls of a full answer cost "Hallo Willy" seven seconds (the owner's speed round,
2026-10-08). The answer ends where its object closes, one pass before the model's own end token
(``vlm.stop_at_answer_end``, :func:`src.models.vlm.qwen.answer_object_closed`); this reader takes an answer's first
whole object anyway. A key left out means "not said", never "all"; a full answer in the earlier format reads as well.
The word lists below only check that answer.

**One part, and where the parts lie.** A sentence may single out one part of its kind ("den grauen Würfel, der oben
auf dem anderen liegt": ``which``, "the gray cube on top of the other one") and say where the parts are taken from
("von der schwarzen Matte": ``from``, "on the black mat"). Both are English and checked as the object is; a which
names the object's noun, so the detector's label still maps onto the object; and both are found in the sentence
through the operator's words for the part, which hold them. The task grounds a which alone, and every part of its
kind where they lie otherwise (``src.robot.execution.task.pick_phrase``). A which reads ``once``.

**Sorting: one rule per kind of part** (the owner, 2026-10-09: "grüne Teile in die gelbe Kiste, rote in die blaue").
A sentence that sends different kinds of parts to different places is read into rules. The first kind fills the keys
above; each further kind is one entry of ``also`` with the same keys (:data:`RULE_KEYS`), checked as the first is, and
a problem with one names its entry (``also[0]``). At most :data:`MAX_FURTHER_RULES` entries; the same kind in two
rules, or a rule without its place where there are two or more, is asked again (:func:`_rules_problem`).
:attr:`CommandReading.rules` holds every rule, rule 0 being the reading's own fields, so a one-kind sentence reads as
it did; the notes of every rule are the reading's, so a note on any rule opens the card. The scope, the count and the
pose to go to after are the sentence's, one for all its rules, and the reader adds no rule the sentence does not say.
No quality (i.O./N.i.O.) yet.

**Known sentences** (the owner, 2026-10-08: "speed first"). The cell's everyday sentences ("Alle grauen Würfel in
die Gelbe Kiste.", "Hallo Willy") cost the 8B model ten seconds each time. :mod:`.known` answers a closed grammar
of them with the answer the model was measured to give, and this module makes the card of that answer with the same
checks; any other sentence goes to the model. :func:`read_command` tries it first where the console asks it to
(``runtime.commands.known_sentences``). A sentence the model read before is answered from memory while the same copy
of the weights is loaded (:class:`_Remembering`): with greedy decoding the same question gets the same answer, and
the answer is checked again as if it were new.

**Hard checks, one corrective retry.** An answer that holds no JSON object, misses ``intent`` or adds a key, names a
scope other than the two, or sets a place and a pose that contradict each other is asked again once, saying what was
wrong. So is a which that does not name the object, an object that holds where the parts lie, and a phrase the
detector would get that is not English or that carries a quantifier ("all screws", "both cubes"): the
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
counted command ("nimm drei Würfel") is read as scope ``once`` with the note that a count is not supported, and a
command that singles out one part as ``once`` too: the card errs toward fewer motions.

**Soft notes.** An answer that passes is understood, and what the card should mark "bitte prüfen" is noted, in
the order of :data:`NOTE_ORDER`: words the sentence does not contain, a pose nobody taught (or Home as a place), a
count the task cannot honour, and that a retry was needed.

**A greeting.** A sentence the model read as no command that opens with a greeting or a farewell ("Hallo Willy, wie
geht's?", "Tschüss Willy") or asks Willy to wave ("Willy, wink mal!") is a greeting (:func:`greets`); so is a sentence
that is nothing but one ("Hallo Willy!"), whatever the model made of it, bar a stop. It is the one thing a word list
reads, and only beside the model's answer, never in place of it: a command with a greeting in front ("Hallo Willy,
nimm den Würfel") stays the model's task. The console answers a greeting as its app config says (a wave at once, a
wave on a click, or a greeting back); the reading itself moves nothing.

**Reading moves nothing.** :func:`understand` calls ``ask`` and nothing else, and this module imports no robot,
camera or console module: it never starts a run, never sets a detector prompt and never touches the cell. A
sentence read as "stop" stops nothing either; the card points at the stop buttons. :func:`read_command` is the
console's door: a known sentence first, then the availability rule
(:func:`~src.models.vlm.availability.reader_availability`, owner decision Q8 A), then this reader through the
process's one VLM copy. A reading says whether a person must look at it before anything starts
(:attr:`CommandReading.startable`): the console starts a task on Enter only from a clean reading, and opens the card
for every other.

No torch at import: the model is reached through ``ask``, and the holder that builds one loads nothing until asked.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import weakref
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, ValidationError

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
from .holder import VlmHolder, VlmWeights, shared_vlm

__all__ = [
    "ANSWER_KEYS",
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
    "GREETING_WORDS",
    "HOME_POSE",
    "KNOWN_SENTENCE_MODEL",
    "MAX_FURTHER_RULES",
    "MAX_PHRASE_CHARS",
    "MAX_SENTENCE_CHARS",
    "MAX_WHICH_CHARS",
    "NOTE_ORDER",
    "PhraseReading",
    "REMEMBERED_ANSWERS",
    "RULE_KEYS",
    "ReadingNote",
    "RuleAnswer",
    "RuleReading",
    "SHARED_WORDS",
    "WAVE_WORDS",
    "extract_command_object",
    "forget_remembered_answers",
    "greets",
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
#: The reader's own phrase is at most this long: 1 to 4 English words, as the instruction asks. Not the card's field,
#: which takes what a person types (``TaskIn.object``, 200 characters). The object, the from and the place.
MAX_PHRASE_CHARS: Final[int] = 80
#: The reader's which is at most this long: up to 12 English words that single out one part ("the gray cube on top of
#: the other one" is 37 characters). The card's field takes 200 (``TaskIn.which``).
MAX_WHICH_CHARS: Final[int] = 120
#: A sentence is at most this long: the console's ``CommandIn.text`` and ``CommandProvenanceIn.text`` take this
#: constant. The sentence is the model's input, which costs almost nothing per character (measured on the cell,
#: 2026-10-08: about 140 ms per ANSWER token and no fixed cost), so a long description fits.
MAX_SENTENCE_CHARS: Final[int] = 1000
#: The ``model_id`` of a reading the known sentences' table answered (:mod:`.known`): no model was asked.
KNOWN_SENTENCE_MODEL: Final[str] = "known-sentence"
#: How many answers the reader remembers for one loaded copy of the weights (:class:`_Remembering`).
REMEMBERED_ANSWERS: Final[int] = 256
#: The raw answer kept on a reading, for the tech view.
RAW_LIMIT: Final[int] = 500
#: Room for the longest answer the instruction asks for (57 tokens for the stacked cube of 2026-10-08 with its which
#: and its bin, about 80 with a from and 15 words said too; a full answer in the earlier format, every key written,
#: about 95) and a margin. A sort of four kinds of parts is the longest (counted with the Qwen3 tokenizer, 2026-10-09):
#: 114 tokens compact, 148 with the spaces the 4B writes, 229 with every key written null; with the longest words the
#: instruction allows in every rule, a from and a pose to go to after, 213 compact, 253 spaced, 318 spaced and null. The
#: answer ends where its object closes, so the bound costs nothing on a good answer; only an answer that never closes
#: runs to it (about 45 s at 140 ms a token on the cell).
COMMAND_MAX_NEW_TOKENS: Final[int] = 320
#: The most further rules (``also``) one answer may hold: a sentence sorts at most four kinds of parts. Each rule is a
#: place the task finds before its first pick, and 30 to 40 tokens of answer (about 4 to 6 s on the cell).
MAX_FURTHER_RULES: Final[int] = 3

# --- the words that check an answer (never a reading of the sentence: owner decision 22; the known sentences'
# table in .known is the one reading besides the model's, the owner's of 2026-10-08) ---------------------------------

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

#: What a greeting or a farewell opens with, folded as :func:`_fold` folds a sentence ("Grüß Gott" is "gruess gott"):
#: the sentences Willy may wave at (:func:`greets`).
GREETING_WORDS: Final[tuple[str, ...]] = (
    "hallo", "hello", "hi", "hey", "huhu", "hallihallo", "halli hallo", "servus", "moin", "moinsen", "tach",
    "gruess gott", "gruss gott", "gruezi", "gruess dich", "gruss dich", "guten morgen", "guten tag", "guten abend",
    "good morning", "good afternoon", "good evening", "howdy", "greetings", "ahoi", "hola",
    "tschuess", "tschau", "ciao", "bye", "goodbye", "auf wiedersehen", "bis bald", "see you",
)

#: Words that ask Willy to wave, folded.
WAVE_WORDS: Final[frozenset[str]] = frozenset({"wink", "winke", "winken", "winkst", "zuwinken", "wave", "waving"})

#: The robot's name, which a greeting may carry anywhere ("Hallo Willy!", "Willy, hallo").
_ROBOT_NAMES: Final[frozenset[str]] = frozenset({"willy", "willi"})

#: What may stand beside a greeting in a sentence that is nothing but one ("Hallo zusammen!", "Hi there").
_BESIDE_A_GREETING: Final[frozenset[str]] = frozenset({
    "du", "ihr", "zusammen", "allerseits", "there", "everyone", "everybody", "all", "you", "na", "mal", "doch",
    "bitte", "please", "again", "wieder", "lieber", "mein", "my", "freund", "friend", "robot", "roboter",
})

#: A word that turns a greeting or a wave around ("nicht winken", "don't wave"): such a sentence greets nobody.
_NEGATIONS: Final[frozenset[str]] = frozenset({"nicht", "kein", "keine", "not", "dont", "don", "never", "nie", "no",
                                               "nein"})

#: What may stand before a pose's label in a place's own words and leave it that pose: "auf Ablage links", "to the
#: Wartepose". Never a word that places a spot beside it ("neben", "next"): that is a place the camera finds.
_BEFORE_A_POSE: Final[frozenset[str]] = frozenset({
    "auf", "aufs", "in", "ins", "im", "an", "am", "ans", "zu", "zur", "zum", "nach", "bei", "beim",
    "der", "die", "das", "den", "dem", "des", "ein", "eine", "einen", "einem", "einer",
    "on", "onto", "into", "to", "at", "the", "a",
})


# --- the instruction ----------------------------------------------------------------------------------------------


#: The keys of an answer, in the order the instruction lists them and its examples write them. ``also`` holds a
#: sort's further rules, after the first rule's keys, in the order the sentence says them.
ANSWER_KEYS: Final[tuple[str, ...]] = (
    "intent", "object", "from", "which", "object_said", "place", "place_said", "place_pose", "also", "return_to",
    "count", "scope",
)
#: The keys of one further rule, an entry of ``also``: the first rule's own keys, in the same order.
RULE_KEYS: Final[tuple[str, ...]] = ("object", "from", "which", "object_said", "place", "place_said", "place_pose")


def _rule(**fields: Any) -> dict[str, Any]:
    """A further rule as the instruction writes it, an entry of ``also``: only the keys the sentence fills for that
    kind, in :data:`RULE_KEYS` order. ``source`` is the key ``from``."""
    rule = {"from" if key == "source" else key: value for key, value in fields.items()}
    unknown = sorted(set(rule) - set(RULE_KEYS))
    if unknown:
        raise ValueError(f"a rule has no key {', '.join(unknown)}")
    return {key: rule[key] for key in RULE_KEYS if key in rule}


def _example(**fields: Any) -> dict[str, Any]:
    """An answer as the instruction writes it: ``intent`` (a task unless ``fields`` says otherwise) and only the keys
    the sentence fills, in :data:`ANSWER_KEYS` order. ``source`` is the key ``from``, which Python takes as no
    keyword; each entry of a list ``also`` is written as :func:`_rule` writes it."""
    answer: dict[str, Any] = {"intent": "task"}
    answer.update({"from" if key == "source" else key: value for key, value in fields.items()})
    unknown = sorted(set(answer) - set(ANSWER_KEYS))
    if unknown:
        raise ValueError(f"an answer has no key {', '.join(unknown)}")
    if isinstance(answer.get("also"), (list, tuple)):
        answer["also"] = [_rule(**entry) if isinstance(entry, Mapping) else entry for entry in answer["also"]]
    return {key: answer[key] for key in ANSWER_KEYS if key in answer}


#: The poses the examples use, label -> name. Names follow the pose name rule (an ASCII identifier); the label is
#: what an operator says.
EXAMPLE_POSES: Final[tuple[tuple[str, str], ...]] = (
    ("Ablage links", "ablage_links"), ("Ablage rechts", "ablage_rechts"), ("Wartepose", "wartepose"),
)

#: The instruction's examples, each a sentence and the answer the reader accepts for it as it stands. The four after
#: the count teach the owner's harder sentences of 2026-10-08: a part on top of another, where the parts lie, "links"
#: beside the pose "Ablage links", and a superlative, which the 4B read as a size ("small cube") without one. The three
#: before the stop sort (2026-10-09): a kind whose noun the sentence leaves out ("rote"), a bin and a pose in one sort,
#: and three kinds in English.
COMMAND_EXAMPLES: Final[tuple[tuple[str, dict[str, Any]], ...]] = (
    ("nimm den grünen Würfel und leg ihn in die blaue Kiste",
     _example(object="green cube", object_said="den grünen Würfel", place="blue bin",
              place_said="in die blaue Kiste")),
    ("alle Schrauben in die Kiste",
     _example(object="screw", object_said="Schrauben", place="bin", place_said="in die Kiste", scope="until_empty")),
    ("räum die Kiste aus", _example(scope="until_empty")),
    ("alle Objekte auf der Schaumstoffmatte in die gelbe Kiste",
     _example(source="on the foam mat", object_said="Objekte auf der Schaumstoffmatte", place="yellow bin",
              place_said="in die gelbe Kiste", scope="until_empty")),
    ("pick the red cylinder and go to Wartepose",
     _example(object="red cylinder", object_said="the red cylinder", return_to="wartepose")),
    ("leg den Becher auf Ablage links", _example(object="cup", object_said="den Becher", place_pose="ablage_links")),
    ("leg die Mutter auf Ablage links und fahr danach in die Wartepose",
     _example(object="nut", object_said="die Mutter", place_pose="ablage_links", return_to="wartepose")),
    ("nimm zwei Schrauben", _example(object="screw", object_said="Schrauben", count=2)),
    ("nimm den roten Würfel, der auf dem anderen liegt",
     _example(object="red cube", which="the red cube on top of the other one",
              object_said="den roten Würfel, der auf dem anderen liegt")),
    ("alle blauen Schrauben vom Tisch in die gelbe Kiste",
     _example(object="blue screw", source="on the table", object_said="blauen Schrauben vom Tisch",
              place="yellow bin", place_said="in die gelbe Kiste", scope="until_empty")),
    ("nimm die Mutter links neben der Kiste und leg sie auf Ablage links",
     _example(object="nut", which="the nut left of the bin", object_said="die Mutter links neben der Kiste",
              place_pose="ablage_links")),
    ("nimm die kleinste Schraube",
     _example(object="screw", which="the smallest screw", object_said="die kleinste Schraube")),
    ("grüne Teile in die gelbe Kiste, rote in die blaue",
     _example(object="green part", object_said="grüne Teile", place="yellow bin", place_said="in die gelbe Kiste",
              also=[_rule(object="red part", object_said="rote", place="blue bin", place_said="in die blaue")],
              scope="until_empty")),
    ("die Schrauben in die Kiste links, die Muttern auf Ablage rechts",
     _example(object="screw", object_said="die Schrauben", place="left bin", place_said="in die Kiste links",
              also=[_rule(object="nut", object_said="die Muttern", place_pose="ablage_rechts")], scope="until_empty")),
    ("put the red cubes into the red bin, the blue ones into the blue bin and the nuts on Ablage links",
     _example(object="red cube", object_said="the red cubes", place="red bin", place_said="into the red bin",
              also=[_rule(object="blue cube", object_said="the blue ones", place="blue bin",
                          place_said="into the blue bin"),
                    _rule(object="nut", object_said="the nuts", place_pose="ablage_links")],
              scope="until_empty")),
    ("Stopp!", _example(intent="stop")),
    ("wie spät ist es?", _example(intent="none")),
)


def _json(value: Any) -> str:
    """One line of JSON as the instruction writes it: compact, no space after a colon or a comma."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _poses_line(pairs: tuple[tuple[str, str], ...]) -> str:
    """``"Ablage links" -> "ablage_links", ...``: both sides quoted, so a label cannot change the question."""
    return ", ".join(f"{_json(label)} -> {_json(name)}" for label, name in pairs)


#: The system message of every command question. Fixed, so it can be compared and cached; the poses and the
#: command travel in the user message.
COMMAND_INSTRUCTION: Final[str] = "\n".join([
    "You read ONE spoken or typed command for a pick-and-place robot. The command may be German or English.",
    'The user message gives POSES, the robot\'s taught poses as "spoken label" -> "name" pairs, and then the '
    "COMMAND.",
    "Answer with ONLY one JSON object on one line, written compactly with no space after a colon or a comma, no "
    'prose, no code fence. Write only the keys the command fills, and "intent" always: a key left out means the '
    'command does not say it, so never write null or "". The keys:',
    '{"intent":"task"|"stop"|"none","object":"<english noun phrase>","from":"<english words>",'
    '"which":"<english words>","object_said":"<the command\'s own words>","place":"<english noun phrase>",'
    '"place_said":"<the command\'s own words>","place_pose":"<a name from POSES>",'
    '"also":[<one object per further kind of part, with its own keys object to place_pose>],'
    '"return_to":"<a name from POSES>","count":<a whole number>,"scope":"until_empty"}',
    "Rules:",
    "- object: what to pick, as 1 to 4 lowercase English words, singular, no article, keeping colour, size and "
    'material ("den grünen Würfel" -> "green cube", "den großen grauen Würfel" -> "large gray cube"). Always '
    'English, never German. Left out when the command names no kind of thing ("räum die Kiste aus", "pick '
    'anything").',
    "- from: where the parts lie that the command takes them from, as 1 to 4 English words with their preposition "
    '("von der schwarzen Matte" -> "on the black mat", "aus der Kiste" -> "in the bin"). Never in object: the '
    "mat, table or tray the parts lie on is not picked. Never the place the part goes to.",
    "- which: only when the command singles out one part of its kind, by where it lies among the others (der obere, "
    "ganz links, neben der Kiste) or as the most of them (der kleinste, der größte): at most 12 English words that "
    'name the object with its noun ("the gray cube on top of the other one", "the cup left of the bin", "the '
    'smallest cube"). A size word stays in object and is no which ("den großen Würfel" -> object "large cube"); a '
    'superlative is a which and never a size in object ("den kleinsten Würfel" -> object "cube", which "the '
    'smallest cube").',
    "- object_said: the command's own words for the part, its from and which words included, copied exactly as one "
    "piece of the command, at most 15 words. place_said: the command's own words for the place, copied exactly. "
    "Never invent an object or a place the command does not name.",
    "- place_pose: when the command says to put the part on or into one of the POSES (by its spoken label or its "
    "name), that pose's NAME copied exactly. One of the POSES is never a place: then place and place_said are left "
    "out.",
    "- return_to: when the command says where the robot goes AFTERWARDS (danach, dann, anschließend, zurück, "
    "afterwards, then go, go back), the NAME of that pose from POSES. A pose the robot only goes to afterwards is "
    "never place_pose.",
    "- place: where the part goes when the command names a container or a spot the camera must find that is not "
    'one of the POSES, in English ("in die blaue Kiste" -> "blue bin"). place and place_pose are never both set.',
    "- also: only when the command sends different kinds of parts to different places (sorting). The first kind "
    "fills the keys above; each further kind is one object in also with its own object, object_said and place or "
    "place_pose (from and which where the command says them for that kind), read by the same rules. Write the noun "
    'the command leaves out ("rote" -> object "red part", object_said "rote"). Never the same kind twice; at most '
    f"{MAX_FURTHER_RULES} entries; left out for one kind of part.",
    '- scope: "until_empty" when the command says all, every, alle, jede(n), uses a plural for the parts, or says '
    "to empty or clear something (ausräumen, leer machen); otherwise left out. Never put all/alle into object. When "
    "the command names a number of parts, scope is left out and count is that number.",
    '- count: the number of parts when the command names one ("zwei Schrauben" -> 2, "alle vier Muttern" -> 4, '
    '"three cubes" -> 3). A number is always written as count, beside alle too, and then scope is left out.',
    '- intent: "stop" for stop, stopp, halt, anhalten; "none" for greetings, questions, or anything that is not a '
    'pick command; otherwise "task".',
    f"Examples, with POSES: {_poses_line(EXAMPLE_POSES + ((HOME_LABEL, HOME_POSE),))}",
    *(f"{sentence} -> {_json(answer)}" for sentence, answer in COMMAND_EXAMPLES),
])


# --- the answer ---------------------------------------------------------------------------------------------------


class RuleAnswer(BaseModel):
    """One further rule of a sorting command, an entry of ``also``: a kind of part and where it goes, in the first
    rule's own keys (:data:`RULE_KEYS`) and nothing else, read by the same rules. Strict, as the answer is; ``from``
    is :attr:`source` here."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)

    object: str | None = None
    source: str | None = Field(default=None, alias="from")
    which: str | None = None
    object_said: str | None = None
    place: str | None = None
    place_said: str | None = None
    place_pose: str | None = None


class CommandAnswer(BaseModel):
    """The JSON object the model must answer: ``intent``, and only the keys the sentence fills; nothing else.

    Strict, so a number is no string and a string no number: ``"count": "3"`` is asked again rather than read. A key
    left out, null or empty means "not said", never "all": no scope is ``once``. A full answer in the earlier format,
    every key written, reads the same. ``from`` is :attr:`source` here, a word Python keeps for itself. ``also`` holds
    a sort's further rules (a list: strict mode reads no JSON array as a tuple); null or empty is one rule.
    """

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, populate_by_name=True)

    intent: CommandIntent
    object: str | None = None
    source: str | None = Field(default=None, alias="from")
    which: str | None = None
    object_said: str | None = None
    place: str | None = None
    place_said: str | None = None
    place_pose: str | None = None
    also: list[RuleAnswer] | None = None
    return_to: str | None = None
    count: int | None = None
    scope: CommandScope | None = None


_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def extract_command_object(text: str) -> dict[str, Any] | None:
    """The first JSON object in a model's answer, or ``None``. Never raises.

    Object-first: a command is one object, so the answer is scanned for the first ``{`` that opens a whole one,
    whatever comes before or after it (prose, a code fence, a second object, a doubled opening brace). An array is
    not a command, though an object inside one is still found. An object after the first brace is the command only
    where it names its intent: inside an answer that does not read (cut short, a comma missing), the next brace opens
    one of its own rules (``also``), and a rule is never taken for the whole command.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    fenced = _FENCE.search(text)
    candidates = [fenced.group(1), text] if fenced else [text]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        start = candidate.find("{")
        first = True
        while start != -1:
            try:
                value, _ = decoder.raw_decode(candidate, start)
            except ValueError:
                value = None
            if isinstance(value, dict) and (first or "intent" in value):
                return value
            first = False
            start = candidate.find("{", start + 1)
    return None


def _key_of(location: tuple[Any, ...]) -> str:
    """Where a key stands in the answer, as the retry names it: ``intent``, or ``also[0].object_said`` in a rule."""
    key = str(location[0])
    for step in location[1:]:
        key += f"[{step}]" if isinstance(step, int) else f".{step}"
    return key


def _schema_problem(error: ValidationError) -> str:
    """What was wrong with the keys, in words the model can act on; at most three findings."""
    parts: list[str] = []
    for item in error.errors()[:3]:
        location = item.get("loc") or ("answer",)
        key = _key_of(tuple(location))
        kind = item.get("type", "")
        if kind == "missing":
            parts.append(f"key {key!r} is missing")
        elif kind == "extra_forbidden":
            parts.append(f"key {key!r} is not one of the keys")
        else:
            parts.append(f"key {key!r}: {item.get('msg', 'not valid')}")
    return "the JSON does not match the keys: " + "; ".join(parts)


def _place_as_pose(answer: CommandAnswer | RuleAnswer, offered: "_Offered") -> str | None:
    """The taught pose a place really names, or ``None`` when it is a place the camera finds. ``answer`` is a rule:
    the answer's first, or one of its further rules (``also``), each read alone.

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


def _place_is_home(answer: CommandAnswer | RuleAnswer, offered: "_Offered") -> bool:
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


#: Each phrase of a task answer by its key: the most characters it may hold, and the words the retry asks for.
_PHRASE_LIMITS: Final[Mapping[str, tuple[int, int]]] = {
    "object": (MAX_PHRASE_CHARS, 4), "from": (MAX_PHRASE_CHARS, 4), "which": (MAX_WHICH_CHARS, 12),
    "place": (MAX_PHRASE_CHARS, 4),
}


def _head(phrase: str) -> str:
    """The noun a phrase names, its last word, folded ("small metal bracket" -> "bracket"); ``""`` for no words."""
    words = _words(phrase)
    return words[-1] if words else ""


def _names(phrase: str, noun: str) -> bool:
    """``phrase`` holds ``noun``, as it is or with a plural ending ("cubes", "boxes")."""
    return any(word == noun or (word.startswith(noun) and len(word) - len(noun) <= 2) for word in _words(phrase))


def _phrase_problem(answer: CommandAnswer, offered: "_Offered", sentence: str, *, first: bool) -> str:
    """The first hard problem with the phrases of a task answer, or ``""``.

    ``first`` is the answer to the first question: there a phrase copied from a German command is sent back to be
    translated; in the answer to the corrective question the model kept it on purpose, and it is a loanword. The from
    and the which are checked as the object is (the owner's speed round, 2026-10-08), and a which must name the
    object's noun: the detector is asked the which alone, and its label maps onto the object by that noun. A sort's
    rules are checked together first (:func:`_rules_problem`), then each rule as the first is, a further rule's
    problem naming its entry ("in also[0], ...").
    """
    if answer.intent != "task":
        return ""  # its phrases are thrown away, so they are not worth a second question
    together = _rules_problem(answer)
    if together:
        return together
    own_words = offered.outside_poses(sentence)  # the command without its pose labels, which are names
    german_command = _not_english(own_words)
    for index, rule in enumerate((answer, *(answer.also or ()))):
        problem = _rule_problem(rule, offered, own_words, german_command=german_command, first=first)
        if problem:
            return problem if index == 0 else f"in also[{index - 1}], {problem}"
    return ""


def _rule_problem(answer: CommandAnswer | RuleAnswer, offered: "_Offered", own_words: str, *, german_command: bool,
                  first: bool) -> str:
    """The first hard problem with the phrases of one rule, the answer's first or one of its further rules, or
    ``""``: the checks :func:`_phrase_problem` names. ``own_words`` is the command without its pose labels, and
    ``german_command`` whether they are German."""
    camera_place = bool(_phrase(answer.place)) and _place_as_pose(answer, offered) is None \
        and not _place_is_home(answer, offered)
    fields = [("object", answer.object), ("from", answer.source), ("which", answer.which)] \
        + ([("place", answer.place)] if camera_place else [])
    for field, value in fields:
        phrase = _phrase(value)
        if not phrase:
            continue
        limit, words = _PHRASE_LIMITS[field]
        if len(phrase) > limit:
            return f"the {field} phrase is longer than {limit} characters: name it in at most {words} English words"
        if _not_english(phrase):
            return f"the {field} phrase {phrase!r} is not English: give the {field} in English words"
        if analyse(phrase).has_quantifier:
            if field == "which":
                return (f"the which phrase {phrase!r} carries a quantifier: a which singles out one part, so "
                        "all/both/every never stand in it")
            return (f"the {field} phrase {phrase!r} carries a quantifier: all/alle/every belong in scope, "
                    f"never in {field}")
        if first and german_command and _copied(phrase, own_words):
            return (f"the {field} phrase {phrase!r} repeats the command's German words: give the {field} in "
                    f"English words (keep a word only where English uses the same word)")
    named, which, source = _phrase(answer.object), _phrase(answer.which), _phrase(answer.source)
    noun = _head(named)
    if which and noun and not _names(which, noun):
        return (f"the which phrase {which!r} does not name the object {named!r}: say which {noun} it is, in English "
                f"words that hold {noun!r}")
    if source and named and _names(named, _head(source)):
        return (f"the object phrase {named!r} holds where the parts lie ({source!r}): the object names only the "
                "part, and where it lies goes in from")
    if camera_place and (answer.place_pose or "").strip():
        return ("place and place_pose are both set: when the command names one of the POSES, place and "
                "place_said are left out; otherwise place_pose is left out")
    return ""


def _same_kind(one: str, other: str) -> bool:
    """Two phrases name the same kind of part: the same words, the last in the singular or the plural ("red part",
    "red parts"), whatever their case or umlaut spelling."""
    first, second = _words(one), _words(other)
    if not first or not second or first[:-1] != second[:-1]:
        return False
    short, long = sorted((first[-1], second[-1]), key=len)
    return long in (short, short + "s", short + "es")


def _rules_problem(answer: CommandAnswer) -> str:
    """The first hard problem with a sort's rules together, or ``""``; an answer of one rule has none.

    More further rules than :data:`MAX_FURTHER_RULES`; one kind of part in two rules (each kind goes to one place,
    and the detector's label is what sorts a part, so a which or a from cannot tell two rules of one kind apart); or,
    with two rules or more, a rule that names neither a place nor a pose: a sort places a part only where its rule
    says, never at a default. A rule that names no kind of part is the card's to ask, as a sentence of one rule is.
    """
    further = answer.also or []
    if not further:
        return ""
    if len(further) > MAX_FURTHER_RULES:
        return (f"also holds {len(further)} entries: a command sorts at most {MAX_FURTHER_RULES + 1} kinds of parts, "
                f"so also holds at most {MAX_FURTHER_RULES}")
    rules: tuple[CommandAnswer | RuleAnswer, ...] = (answer, *further)
    names = ("the first kind", *(f"also[{index}]" for index in range(len(further))))
    for index, rule in enumerate(rules):
        kind = _phrase(rule.object)
        for before in range(index):
            if kind and _same_kind(kind, _phrase(rules[before].object)):
                return (f"{names[before]} and {names[index]} both name {kind!r}: each kind of part goes to one place, "
                        "so name each kind once, with the words that tell it from the others")
    for name, rule in zip(names, rules):
        if not _phrase(rule.place) and not (rule.place_pose or "").strip():
            return (f"{name} names no place: when the command sends kinds of parts to different places, each kind "
                    "names its place or its place_pose")
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
    """The corrective question: the first, what was wrong, and the previous answer whole, a sort of four rules too
    (up to about 1,100 characters at the token bound), so the rule the problem names is there to correct."""
    previous = " ".join(raw.split())[:1200]
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
class RuleReading:
    """One rule the reader read: a kind of part and where it goes (the owner's sorting, 2026-10-09).

    A task's reading holds one rule, its own fields, and a sorting sentence one more per further kind
    (:attr:`CommandReading.rules`). ``object`` ``None`` names no part, so the card asks; ``place`` is a place the
    camera finds, ``place_pose`` a taught pose's NAME. A rule of a sort with neither carries a note (the reader asked
    again for one that named none). ``notes`` are the notes this rule raised, in :data:`NOTE_ORDER`; the reading's
    notes hold them too.
    """

    object: PhraseReading | None = None
    #: The one part of its kind the rule singles out, in English; ``None`` where it singles out none.
    which: str | None = None
    #: Where the rule's parts lie, in English with its preposition; ``None`` where the sentence says none for it.
    source: str | None = None
    place: PhraseReading | None = None
    place_pose: str | None = None
    notes: tuple[ReadingNote, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """One rule of the console's command card: a part and its place as the reading's own keys say them, and the
        rule's notes."""
        return {
            "object": self.object.to_dict() if self.object is not None else None,
            "which": self.which,
            "source": self.source,
            "place": self.place.to_dict() if self.place is not None else None,
            "place_pose": self.place_pose,
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class CommandReading:
    """What the reader understood of one sentence. It starts nothing: a person's Start on the card does, or the
    person's Enter where the reading is :attr:`startable` (the owner, 2026-10-08).

    ``understood`` false means the model's answer was not usable after one corrective retry; ``reason`` says why
    and no field is filled. ``place_pose`` and ``return_to`` hold pose NAMES; ``return_to`` ``None`` is the default
    return (Home). ``scope`` is ``None`` for a sentence that is no task. A sorting sentence's further rules are in
    :attr:`rules` after rule 0, which is this reading's own object, which, source, place and place pose.
    """

    understood: bool
    intent: CommandIntent
    object: PhraseReading | None = None
    #: The one part the sentence singles out, in English ("the gray cube on top of the other one"): the task grounds
    #: this phrase alone. Its words are the part's (``object.said``). ``None`` where the sentence singles out none.
    which: str | None = None
    #: Where the parts lie that the sentence takes them from, in English with its preposition ("on the black mat"):
    #: the task grounds its parts there. ``None`` where the sentence says none.
    source: str | None = None
    place: PhraseReading | None = None
    place_pose: str | None = None
    scope: CommandScope | None = None
    count: int | None = None
    return_to: str | None = None
    #: Every rule the sentence says, each a kind of part and where it goes, in the sentence's order: rule 0 is this
    #: reading's own object, which, source, place and place pose, then one per further kind of a sorting sentence
    #: ("rote in die blaue"). One rule for a task of one kind, filled in from those fields where a reading is built
    #: without; none for a sentence that is no task or was not read.
    rules: tuple[RuleReading, ...] = ()
    notes: tuple[ReadingNote, ...] = ()
    reason: str = ""
    #: The model's last raw answer, at most 500 characters.
    raw: str = ""
    attempts: int = 0
    latency_ms: float = 0.0
    model_id: str = ""
    #: This reading loaded the model: the first command a VLM cell read.
    loaded_now: bool = False
    #: The sentence greets Willy, bids it goodbye or asks it to wave, and is no command (:func:`greets`). The
    #: console answers it as its app config says (``CommandOut.greeting``, which the route fills in, not
    #: :meth:`to_dict`); the reading moves nothing.
    greeting: bool = False
    #: Every answer of this reading came from memory: the same copy of the weights was asked the same question
    #: before (:class:`_Remembering`), and the answer was checked again as if it were new.
    remembered: bool = False

    def __post_init__(self) -> None:
        """Rule 0 is this reading's own fields: filled in from them where a task's reading is built without rules,
        refused where it says anything else. Only an understood task has rules, at most one more than
        :data:`MAX_FURTHER_RULES`."""
        own = (self.object, self.which, self.source, self.place, self.place_pose)
        if not self.rules:
            if self.understood and self.intent == "task":
                first = RuleReading(object=self.object, which=self.which, source=self.source, place=self.place,
                                    place_pose=self.place_pose)
                object.__setattr__(self, "rules", (first,))
            return
        if not self.understood or self.intent != "task":
            raise ValueError("only an understood task has rules")
        if len(self.rules) > MAX_FURTHER_RULES + 1:
            raise ValueError(f"a reading holds at most {MAX_FURTHER_RULES + 1} rules, not {len(self.rules)}")
        first = self.rules[0]
        if (first.object, first.which, first.source, first.place, first.place_pose) != own:
            raise ValueError("rule 0 is the reading's own object, which, source, place and place pose")

    @property
    def known(self) -> bool:
        """The known sentences' table answered (:mod:`.known`): no model was asked."""
        return self.model_id == KNOWN_SENTENCE_MODEL

    @property
    def more_rules(self) -> tuple[RuleReading, ...]:
        """A sorting sentence's further rules, after rule 0, in the sentence's order; ``()`` for one kind of part."""
        return self.rules[1:]

    @property
    def startable(self) -> bool:
        """A reading the console may start on the person's Enter, with no card to look at first (the owner,
        2026-10-08): an understood task, no note (so no retry, every word found, every pose taught, no count), a part
        named and found in the sentence, and a place, where one is named, found there too. Anything else opens the
        card, which errs narrow as ever: a part named by no word, or a place nobody can check, waits for a person.
        A sort starts so only where every rule is as clean (the owner, 2026-10-09): no note on any rule, and each
        rule's part named and found and its place or pose named, found where it is a place."""
        if not self.understood or self.intent != "task" or self.notes or self.greeting:
            return False
        if self.object is None or not self.object.verified:
            return False
        if self.place is not None and not self.place.verified:
            return False
        if any(rule.notes for rule in self.rules):
            return False
        if self.more_rules and self.place is None and self.place_pose is None:
            return False  # a sort never places a part at a default
        return all(rule.object is not None and rule.object.verified
                   and (rule.place.verified if rule.place is not None else rule.place_pose is not None)
                   for rule in self.more_rules)

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        if not self.understood:
            lines = [f"command not read: {to_ascii(self.reason)}"]
        elif self.intent == "stop":
            lines = ["command read: stop (a sentence stops nothing; the stop buttons do)"]
        elif self.intent == "none":
            lines = ["command read: a greeting, not a pick command" if self.greeting
                     else "command read: not a pick command"]
        else:
            lines = ["command read: task", f"  pick    : {self._pick()}"]
            if self.which is not None:
                lines.append(f"  which   : {to_ascii(self.which)} (the detector is asked this one part)")
            if self.source is not None:
                lines.append(f"  from    : {to_ascii(self.source)}")
            lines.append(f"  place   : {self._place()}")
            for number, rule in enumerate(self.more_rules, start=2):
                lines.append(f"  rule {number}  : {_rule_text(rule)}")
            scope = str(self.scope)
            if self.count is not None and self.count > 1:
                scope += f" (the sentence names {self.count}; a count is not supported)"
            lines.append(f"  scope   : {scope}")
            lines.append(f"  then    : {self.return_to or 'home (the default)'}")
        if self.greeting and self.intent != "none":
            lines.append("  greeting: the sentence is nothing but a greeting, and the console answers it as one")
        if self.notes:
            lines.append(f"  notes   : {', '.join(self.notes)}")
        if self.known:
            lines.append(f"  model   : none asked, a known sentence, {self.latency_ms:.1f} ms")
        else:
            model = (f"  model   : {self.model_id or 'unnamed'}, {self.attempts} question(s), "
                     f"{self.latency_ms:.0f} ms")
            lines.append(model + (", loaded for this command" if self.loaded_now else "")
                         + (", every answer from memory" if self.remembered else ""))
        if self.understood and self.intent == "task":
            lines.append("  start   : " + ("on Enter, nothing to check" if self.startable else "the card first"))
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
            "which": self.which,
            "source": self.source,
            "place": self.place.to_dict() if self.place is not None else None,
            "place_pose": self.place_pose,
            "scope": self.scope,
            "count": self.count,
            "return_to": self.return_to,
            "rules": [rule.to_dict() for rule in self.rules],
            "notes": list(self.notes),
            "reason": self.reason,
            "model": {
                "model_id": self.model_id,
                "latency_ms": self.latency_ms,
                "attempts": self.attempts,
                "loaded_now": self.loaded_now,
                "remembered": self.remembered,
            },
            "raw": self.raw,
            "startable": self.startable,
        }


def _phrase_text(reading: PhraseReading) -> str:
    said = f'"{to_ascii(reading.said)}"' if reading.said is not None else "no words given"
    check = "in the sentence" if reading.verified else "NOT in the sentence: please check"
    return f"{to_ascii(reading.phrase)} (said {said}, {check})"


def _rule_text(rule: RuleReading) -> str:
    """A further rule on one line: its part, which and from where the sentence says them, and where it goes."""
    parts = [_phrase_text(rule.object) if rule.object is not None else "no kind of part named (the card asks)"]
    if rule.which is not None:
        parts.append(f"which {to_ascii(rule.which)}")
    if rule.source is not None:
        parts.append(f"from {to_ascii(rule.source)}")
    place = (f"the camera finds {_phrase_text(rule.place)}" if rule.place is not None
             else f"the taught pose {rule.place_pose}" if rule.place_pose is not None
             else "no place (the card asks)")
    return f"{', '.join(parts)} -> {place}"


def _ordered(notes: set[ReadingNote]) -> tuple[ReadingNote, ...]:
    return tuple(note for note in NOTE_ORDER if note in notes)


def _said_in(said_value: str | None, sentence: str) -> bool:
    """The operator's words, as the model copied them, stand in the sentence."""
    said = _said(said_value)
    return said is not None and _found(said, sentence)


def _phrase_reading(phrase: str, said_value: str | None, sentence: str) -> PhraseReading:
    return PhraseReading(phrase=phrase, said=_said(said_value), verified=_said_in(said_value, sentence))


def _rule_reading(answer: CommandAnswer | RuleAnswer, sentence: str, offered: _Offered) -> RuleReading:
    """One rule of a task's answer read alone, the answer's first or one of its further rules (``also``), with the
    notes it raises: its words not in the sentence, or a pose nobody taught (or Home) as its place."""
    notes: set[ReadingNote] = set()
    target = None
    if (phrase := _phrase(answer.object)) and phrase not in ANY_PART_WORDS:
        target = _phrase_reading(phrase, answer.object_said, sentence)
        if not target.verified:
            notes.add("object_not_in_sentence")
    # A which and a from are the part's words too: the operator's words for the part hold them, and a sentence that
    # does not hold those words says neither as read.
    which, source = _phrase(answer.which) or None, _phrase(answer.source) or None
    if (which or source) and not _said_in(answer.object_said, sentence):
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
    return RuleReading(object=target, which=which, source=source, place=place, place_pose=place_pose,
                       notes=_ordered(notes))


def _reading(answer: CommandAnswer, sentence: str, offered: _Offered, *, retried: bool,
             record: dict[str, Any]) -> CommandReading:
    notes: set[ReadingNote] = {"retried"} if retried else set()
    if answer.intent != "task":
        return CommandReading(understood=True, intent=answer.intent, notes=_ordered(notes), **record)
    # Each rule is read alone, the answer's own keys first and then each further kind of a sort, and the notes of
    # every rule are the reading's: a note on any rule opens the card.
    rules = tuple(_rule_reading(rule, sentence, offered) for rule in (answer, *(answer.also or ())))
    for rule in rules:
        notes.update(rule.notes)
    first = rules[0]
    return_to = None
    if (answer.return_to or "").strip():
        return_to = offered.name_of(answer.return_to or "")
        if return_to is None:
            notes.add("pose_unknown")
    count = answer.count if answer.count is not None and answer.count >= 1 else None
    if count is not None and count > 1:
        notes.add("count_not_supported")
    # A named number of parts is read as one part: the task cannot honour a count, and "until empty" would take
    # more than the sentence asked for. So is one part singled out, by any rule; no scope said is once, never "all".
    singled_out = any(rule.which for rule in rules)
    scope: CommandScope = "once" if count is not None or singled_out else (answer.scope or "once")
    return CommandReading(
        understood=True, intent="task", object=first.object, which=first.which, source=first.source,
        place=first.place, place_pose=first.place_pose, scope=scope, count=count, return_to=return_to, rules=rules,
        notes=_ordered(notes), **record,
    )


# --- a greeting -----------------------------------------------------------------------------------------------------


def _greeting_length(words: list[str]) -> int:
    """How many of ``words`` the greeting they open with takes (the longest that fits), or 0 for none."""
    best = 0
    for greeting in GREETING_WORDS:
        span = greeting.split()
        if len(span) > best and words[:len(span)] == span:
            best = len(span)
    return best


def greets(sentence: str, *, intent: CommandIntent | None) -> bool:
    """Whether ``sentence`` greets Willy, bids it goodbye or asks it to wave, beside the model's reading ``intent``
    (``None`` where its answer was not usable).

    A sentence the model read as no command (``"none"``) greets when, the robot's name aside, it opens with one of
    :data:`GREETING_WORDS` ("Hallo Willy, wie geht's?") or holds one of :data:`WAVE_WORDS` ("Kannst du winken?"). A
    sentence that is nothing but that, the name and a few words that may stand beside a greeting ("Hallo zusammen!",
    "Willy, wink mal!"), greets whatever the model made of it, bar a stop: a small model that reads "Hallo Willy" as a
    task without a part, or not at all, does not keep Willy from waving back. A sentence with a negation ("nicht
    winken", "don't wave") greets nobody.
    """
    words = _fold(sentence).split()
    if not words or intent == "stop" or any(word in _NEGATIONS for word in words):
        return False
    rest = [word for word in words if word not in _ROBOT_NAMES]
    opened = _greeting_length(rest)
    asks = any(word in WAVE_WORDS for word in rest)
    if not opened and not asks:
        return False
    if intent == "none":
        return True
    return all(word in _BESIDE_A_GREETING or word in WAVE_WORDS for word in rest[opened:])


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
        _LOG.info("asking again for %r: %s (first answer: %.1200s)", sentence[:120], problem, raw)
        raw = str(ask(COMMAND_INSTRUCTION, _retry_question(question, raw, problem)))
        attempts = 2
        answer, problem = _checked(raw, offered, sentence, first=False)
    record: dict[str, Any] = {
        "raw": raw[:RAW_LIMIT],
        "attempts": attempts,
        "latency_ms": (time.perf_counter() - started) * 1000.0,
        "model_id": str(getattr(ask, "model_id", "") or ""),
        "loaded_now": bool(getattr(ask, "loaded_now", False)),
        "remembered": bool(getattr(ask, "remembered", False)),
    }
    if answer is None:
        reading = CommandReading(
            understood=False, intent="none", notes=("retried",),
            reason=f"the model's answer was not usable after one corrective retry: {problem}",
            greeting=greets(sentence, intent=None), **record,
        )
        _LOG.warning("not read %r after %d question(s): %s (last answer: %.1200s)",
                     sentence[:120], attempts, problem, raw)
    else:
        reading = _reading(answer, sentence, offered, retried=attempts > 1,
                           record={**record, "greeting": greets(sentence, intent=answer.intent)})
        # The answer itself is logged too, a sort's whole: the cell's log is where a measured row for the tests comes
        # from.
        _LOG.info("read %r as %s in %d question(s), %.0f ms%s, %d rule(s), notes %s (answer: %.1200s)",
                  sentence[:400], reading.intent, attempts, reading.latency_ms,
                  ", from memory" if reading.remembered else "", len(reading.rules), list(reading.notes),
                  " ".join(raw.split()))
    return reading


def read_command(
    text: str,
    *,
    models: Any,
    poses: Mapping[str, str],
    weights_present: bool | None = None,
    holder: VlmHolder | None = None,
    known: bool = False,
) -> CommandReading:
    """The console's door: a known sentence, else the availability rule, then :func:`understand` through the one VLM
    copy.

    ``known`` asks the known sentences' table first (:func:`~src.models.vlm.known.read_known`, the console's
    ``runtime.commands.known_sentences``): it answers before the rule, so a known sentence is read before "Laden"
    and on a box without the weights, and it loads nothing. ``models`` is the models section the cell was built
    with, and ``weights_present`` whether the checkpoint is on this box (see
    :func:`~src.models.vlm.availability.reader_availability`). Raises :class:`CommandRefused` with the console's
    code when the rule refuses, when a copy of other weights is in the way (``vlm_not_loaded``), or when the model
    cannot be loaded or fails while it answers (``vlm_unavailable``). Never loads the model on a cell whose detector
    is not the VLM, where a person loads it first, and never beside a copy of other weights. A question the loaded
    copy answered before is answered from memory (:class:`_Remembering`).
    """
    _sentence(text)  # a sentence that cannot be read is refused before the rule is asked
    if known:
        from .known import read_known  # noqa: PLC0415 (known.py reads its answers through this module)

        reading = read_known(text, poses=poses)
        if reading is not None:
            return reading
    held = holder if holder is not None else shared_vlm()
    availability = reader_availability(models, weights_present=weights_present, holder=held)
    if availability.refusal:
        raise CommandRefused(availability.refusal, availability.cause, availability=availability)
    try:
        asker = held.asker_for(availability.vlm, may_load=availability.may_load,
                               max_new_tokens=COMMAND_MAX_NEW_TOKENS)
        return understand(text, ask=_Remembering(asker, weights=VlmWeights.of(availability.vlm)), poses=poses)
    except VlmNotLoadedError as exc:
        raise CommandRefused("vlm_not_loaded", str(exc), availability=availability) from exc
    except VlmUnavailableError as exc:
        raise CommandRefused("vlm_unavailable", str(exc), availability=availability) from exc


# --- remembered answers -----------------------------------------------------------------------------------------

#: The answers each loaded copy gave, by copy: a copy that goes (a rebuild, a test's fresh holder) takes its answers
#: with it. Inside, (weights, the instruction's sha256, the user message) -> the raw answer, oldest first.
_ANSWERS: "weakref.WeakKeyDictionary[Any, OrderedDict[tuple[VlmWeights, str, str], str]]" = \
    weakref.WeakKeyDictionary()
_ANSWERS_LOCK = threading.Lock()


def forget_remembered_answers() -> None:
    """Drop every remembered answer: the next question of each sentence goes to the model again."""
    with _ANSWERS_LOCK:
        _ANSWERS.clear()


class _Remembering:
    """An ``ask`` that answers a question the same loaded copy answered before from memory (R2, 2026-10-08).

    The model decodes greedily, so the same weights asked the same question give the same answer. The key is the
    weights, the sha256 of the instruction and the user message, which holds the poses and the sentence: a pose
    taught or renamed since, or a new instruction, is a new question. Only the raw answer is kept, so
    :func:`understand` checks a remembered answer exactly as a fresh one, and a reading that needed a retry or was not
    understood replays as such. Memory answers only while the copy that gave the answer is loaded: a VLM cell whose
    copy is not loaded loads it at its command, as before, and never later inside a run. An answer that raised is
    never remembered. At most :data:`REMEMBERED_ANSWERS` per copy, the least recently asked go first.
    """

    def __init__(self, ask: Any, *, weights: VlmWeights) -> None:
        self._ask = ask
        self._weights = weights
        #: Questions put to the model by this reading, and questions answered from memory.
        self.asked = 0
        self.replayed = 0

    @property
    def model_id(self) -> str:
        return str(getattr(self._ask, "model_id", "") or "")

    @property
    def loaded_now(self) -> bool:
        return bool(getattr(self._ask, "loaded_now", False))

    @property
    def remembered(self) -> bool:
        """Every answer of this reading came from memory."""
        return self.replayed > 0 and self.asked == 0

    def __call__(self, system: str, user: str) -> str:
        copy = getattr(self._ask, "grounder", None)
        key = (self._weights, hashlib.sha256(system.encode("utf-8")).hexdigest(), user)
        if copy is not None and bool(getattr(copy, "loaded", False)):
            with _ANSWERS_LOCK:
                answers = _answers_of(copy, keep=False)
                raw = answers.get(key) if answers is not None else None
                if answers is not None and raw is not None:
                    answers.move_to_end(key)
            if raw is not None:
                self.replayed += 1
                return raw
        raw = str(self._ask(system, user))
        self.asked += 1
        if copy is not None:
            with _ANSWERS_LOCK:
                answers = _answers_of(copy, keep=True)
                if answers is not None:
                    answers[key] = raw
                    answers.move_to_end(key)
                    while len(answers) > REMEMBERED_ANSWERS:
                        answers.popitem(last=False)
        return raw


def _answers_of(copy: Any, *, keep: bool) -> "OrderedDict[tuple[VlmWeights, str, str], str] | None":
    """The answers ``copy`` gave, a new empty record where ``keep`` asks for one; ``None`` for a copy that cannot be
    remembered by (one that cannot be referred to weakly is never kept alive by a memory of its answers). Under the
    lock."""
    try:
        answers = _ANSWERS.get(copy)
        if answers is None and keep:
            answers = _ANSWERS[copy] = OrderedDict()
        return answers
    except TypeError:
        return None
