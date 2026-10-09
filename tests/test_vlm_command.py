"""The command reader: the owner's sentences, German and English, read from canned model answers.

`understand(sentence, ask=..., poses=...)` puts one text question to the VLM and checks the answer it gets. These
tests script `ask`, so no weights and no GPU are involved: what is pinned here is what the reader does with an
answer, which is where every mistake would turn into a wrong card.

Three things matter most:

1. **A phrase the detector gets is English, with no quantifier.** A German object phrase (one the router knows,
   one of the cell's German words, or the German command's own words copied), or "all screws", gets exactly one
   corrective retry, and a second bad answer is "not understood", never a half-read card.
2. **A pose field holds only an offered NAME.** The reader is handed the taught poses as label -> name pairs, so a
   spoken "Ablage links" comes back as `ablage_links`; anything else is dropped and noted. A place whose own words
   are a pose's label is that pose; a place without words is never taken for one.
3. **The card errs narrow.** A word for any part names no part (the card asks, Q7 A+), and a counted command
   reads as one part.
4. **Reading moves nothing.** `understand` calls `ask` and nothing else, and its module imports no robot, camera,
   console or `willy` module, so a sentence cannot reach the cell.

The real model's answers from the dev box are replayed too (`TheRealModelsAnswersTests`): the reader's own checks
must not cost a good answer a second question.

The owner's speed round (2026-10-08) adds three things, each pinned against the same measured answers: the known
sentences' table answers a closed grammar of everyday sentences as the model did, word for word, and gives every
other sentence to the model (`TheKnownSentencesTests`); a question the loaded copy answered before is answered from
memory and checked again (`RememberedAnswersTests`); and a reading says whether Enter may start it without the card
(`StartableTests`).

Its second wave shortens the answer and widens the reading: the model writes only the keys a sentence fills, compact
(`SparseAnswersTests`; a full answer in the earlier format still reads, which the measured answers above pin), and a
sentence may single out one part (`which`) and say where the parts lie (`from`). `TheOwnersDayTests` reads the
sentences of 2026-10-08 that way, from canned answers in the new format.

Every test tears down with `shared_vlm().forget()`: the reader shares one process-wide VLM holder with detection,
and a holder that kept a grounder would leak it into the perception tests that assert a build loads nothing.
"""

from __future__ import annotations

import ast
import json
import pathlib
import subprocess
import sys
import unittest
from typing import Any

from src.models.vlm import (
    COMMAND_INSTRUCTION,
    CommandReading,
    PhraseReading,
    shared_vlm,
    understand,
)
from src.models.vlm.command import (
    ANSWER_KEYS,
    COMMAND_EXAMPLES,
    EXAMPLE_POSES,
    KNOWN_SENTENCE_MODEL,
    MAX_PHRASE_CHARS,
    MAX_SENTENCE_CHARS,
    MAX_WHICH_CHARS,
    REMEMBERED_ANSWERS,
    CommandAnswer,
    _example,
    _json,
    extract_command_object,
    forget_remembered_answers,
    read_command,
)
from src.models.vlm.known import known_answer, read_known

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The owner's taught poses as the console hands them over: spoken label -> name.
POSES = {"Ablage links": "ablage_links", "Wartepose": "wartepose"}


def _answer(**fields: Any) -> str:
    """One model answer with every key, as the instruction asked until 2026-10-08 and as the measured answers were
    given; ``fields`` overrides the defaults."""
    base: dict[str, Any] = {
        "intent": "task", "object": "", "object_said": None, "place": None, "place_said": None,
        "place_pose": None, "scope": "once", "count": None, "return_to": None,
    }
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


def _sparse(**fields: Any) -> str:
    """One model answer as the instruction asks now: compact, ``intent`` and only the keys given (``source`` is
    ``from``)."""
    return _json(_example(**fields))


class _Canned:
    """An ``ask`` that answers from a script, one answer per call, and records every question."""

    def __init__(self, *answers: str, model_id: str = "fake/qwen", loaded_now: bool = False) -> None:
        self.answers = list(answers)
        self.calls: list[tuple[str, str]] = []
        self.model_id = model_id
        self.loaded_now = loaded_now

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        if not self.answers:
            raise AssertionError("the reader asked more often than this test scripted")
        return self.answers.pop(0)


class _ForgetsTheSharedHolder(unittest.TestCase):
    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)


class TheOwnersSentencesTests(_ForgetsTheSharedHolder):
    """The sentences the owner says at the cell, each with the answer a well-behaved model gives."""

    def _read(self, sentence: str, *answers: str) -> tuple[CommandReading, _Canned]:
        ask = _Canned(*answers)
        return understand(sentence, ask=ask, poses=POSES), ask

    def test_the_green_cube_goes_into_the_blue_bin(self) -> None:
        reading, ask = self._read(
            "Leg den grünen Würfel in die blaue Kiste",
            _answer(object="green cube", object_said="den grünen Würfel",
                    place="blue bin", place_said="in die blaue Kiste"),
        )
        self.assertTrue(reading.understood)
        self.assertEqual(reading.intent, "task")
        self.assertEqual(reading.object, PhraseReading(phrase="green cube", said="den grünen Würfel", verified=True))
        self.assertEqual(reading.place, PhraseReading(phrase="blue bin", said="in die blaue Kiste", verified=True))
        self.assertIsNone(reading.place_pose)
        self.assertEqual(reading.scope, "once")
        self.assertIsNone(reading.return_to)
        self.assertEqual(reading.notes, ())
        self.assertEqual(reading.attempts, 1)
        self.assertEqual(len(ask.calls), 1)

    def test_clearing_the_bin_names_no_part_and_runs_until_empty(self) -> None:
        reading, _ = self._read("Räum die Kiste aus", _answer(scope="until_empty"))
        self.assertTrue(reading.understood)
        self.assertIsNone(reading.object, "no kind of thing was named: the card asks what to pick")
        self.assertIsNone(reading.place)
        self.assertIsNone(reading.place_pose)
        self.assertEqual(reading.scope, "until_empty")
        self.assertEqual(reading.notes, ())

    def test_all_screws_go_to_a_taught_pose_named_by_its_label(self) -> None:
        reading, _ = self._read(
            "Nimm alle Schrauben und leg sie auf Ablage links",
            _answer(object="screw", object_said="Schrauben", place_pose="ablage_links", scope="until_empty"),
        )
        self.assertTrue(reading.understood)
        self.assertEqual(reading.object, PhraseReading(phrase="screw", said="Schrauben", verified=True))
        self.assertEqual(reading.place_pose, "ablage_links")
        self.assertIsNone(reading.place)
        self.assertEqual(reading.scope, "until_empty")
        self.assertEqual(reading.notes, ())

    def test_a_spoken_label_comes_back_as_its_name(self) -> None:
        """The model answered the label it heard; the card needs the YAML key."""
        reading, _ = self._read(
            "Nimm alle Schrauben und leg sie auf Ablage links",
            _answer(object="screw", object_said="Schrauben", place_pose="Ablage links", scope="until_empty"),
        )
        self.assertEqual(reading.place_pose, "ablage_links")
        self.assertNotIn("pose_unknown", reading.notes)

    def test_the_same_in_english(self) -> None:
        reading, _ = self._read(
            "Put the green cube in the blue bin",
            _answer(object="green cube", object_said="the green cube", place="blue bin", place_said="in the blue bin"),
        )
        self.assertTrue(reading.understood)
        self.assertEqual(reading.object.phrase if reading.object else None, "green cube")
        self.assertEqual(reading.place.phrase if reading.place else None, "blue bin")
        self.assertEqual(reading.scope, "once")
        self.assertEqual(reading.notes, ())

    def test_all_screws_in_english(self) -> None:
        reading, _ = self._read(
            "Take all screws and put them on Ablage links",
            _answer(object="screw", object_said="screws", place_pose="ablage_links", scope="until_empty"),
        )
        self.assertEqual((reading.place_pose, reading.scope), ("ablage_links", "until_empty"))
        self.assertEqual(reading.notes, ())

    def test_the_return_pose_is_read_as_a_name(self) -> None:
        reading, _ = self._read(
            "Nimm den roten Zylinder und fahr danach in die Wartepose",
            _answer(object="red cylinder", object_said="den roten Zylinder", return_to="wartepose"),
        )
        self.assertEqual(reading.return_to, "wartepose")
        self.assertIsNone(reading.place_pose)
        self.assertEqual(reading.notes, ())

    def test_home_is_always_a_return_pose(self) -> None:
        reading, ask = self._read(
            "Leg den Becher auf Ablage links und fahr dann nach Home",
            _answer(object="cup", object_said="den Becher", place_pose="ablage_links", return_to="home"),
        )
        self.assertEqual(reading.return_to, "home")
        self.assertIn('"Home" -> "home"', ask.calls[0][1], "Home is offered even when no taught pose is called home")

    def test_stop_is_read_and_stops_nothing(self) -> None:
        """A sentence never stops the arm: the card only shows where the stop buttons are."""
        reading, _ = self._read("Stopp!", _answer(intent="stop"))
        self.assertTrue(reading.understood)
        self.assertEqual(reading.intent, "stop")
        self.assertEqual((reading.object, reading.place, reading.place_pose, reading.scope), (None, None, None, None))

    def test_a_greeting_is_no_task(self) -> None:
        reading, _ = self._read("Hallo Willy, wie geht's?", _answer(intent="none"))
        self.assertTrue(reading.understood)
        self.assertEqual(reading.intent, "none")
        self.assertIsNone(reading.scope)

    def test_a_count_is_read_and_noted_as_not_supported(self) -> None:
        reading, _ = self._read("Nimm drei Würfel", _answer(object="cube", object_said="Würfel", count=3))
        self.assertTrue(reading.understood)
        self.assertEqual(reading.count, 3)
        self.assertEqual(reading.scope, "once")
        self.assertEqual(reading.notes, ("count_not_supported",))

    def test_a_count_of_one_needs_no_note(self) -> None:
        reading, _ = self._read("Nimm einen Würfel", _answer(object="cube", object_said="Würfel", count=1))
        self.assertEqual(reading.count, 1)
        self.assertEqual(reading.notes, ())

    def test_a_count_reads_once_whatever_scope_the_model_gave(self) -> None:
        """A task cannot honour a count, so the card errs toward fewer motions: one part, and the note says why."""
        for count in (1, 3):
            with self.subTest(count=count):
                reading, _ = self._read("Nimm drei Würfel", _answer(object="cube", object_said="Würfel", count=count,
                                                                   scope="until_empty"))
                self.assertTrue(reading.understood)
                self.assertEqual((reading.scope, reading.count), ("once", count))

    def test_a_word_for_any_part_names_no_part(self) -> None:
        """A phrase like "part" grounds whatever the camera sees, bin walls included: on a real cell that takes the
        operator's tick (Q7 A+), so the card asks rather than carrying it as a part's name."""
        for sentence, phrase in (("Nimm das Teil und leg es auf Ablage links", "part"),
                                 ("Räum die Kiste aus", "everything"), ("Räum die Kiste aus", "objects"),
                                 ("Pick anything", "anything"), ("Nimm die Sachen", "things")):
            with self.subTest(phrase=phrase):
                reading, ask = self._read(sentence, _answer(object=phrase, scope="until_empty"))
                self.assertTrue(reading.understood)
                self.assertIsNone(reading.object, "the card asks what to pick")
                self.assertEqual(len(ask.calls), 1)

    def test_a_part_word_that_narrows_is_a_part(self) -> None:
        reading, _ = self._read("Nimm das kleine Metallteil",
                                _answer(object="small metal part", object_said="das kleine Metallteil"))
        self.assertEqual(reading.object.phrase if reading.object else None, "small metal part")

    def test_the_model_and_its_time_are_on_the_reading(self) -> None:
        ask = _Canned(_answer(intent="none"), model_id="Qwen/Qwen3-VL-4B-Instruct", loaded_now=True)
        reading = understand("Hallo", ask=ask, poses=POSES)
        self.assertEqual(reading.model_id, "Qwen/Qwen3-VL-4B-Instruct")
        self.assertTrue(reading.loaded_now)
        self.assertGreaterEqual(reading.latency_ms, 0.0)
        self.assertEqual(reading.raw, _answer(intent="none"))


class OneCorrectiveRetryTests(_ForgetsTheSharedHolder):
    """A hard check failed: the reader asks once more, says why, and never asks a third time."""

    SENTENCE = "Leg den grünen Würfel in die blaue Kiste"
    GOOD = _answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                   place_said="in die blaue Kiste")

    def _retried(self, first: str, *, problem_says: str) -> CommandReading:
        ask = _Canned(first, self.GOOD)
        reading = understand(self.SENTENCE, ask=ask, poses=POSES)
        self.assertEqual(len(ask.calls), 2)
        system, retry = ask.calls[1]
        self.assertEqual(system, COMMAND_INSTRUCTION, "the retry keeps the instruction")
        self.assertIn(self.SENTENCE, retry, "the retry repeats the command")
        self.assertIn(problem_says, retry, "the retry says what was wrong")
        self.assertTrue(reading.understood)
        self.assertEqual(reading.notes, ("retried",))
        self.assertEqual(reading.attempts, 2)
        self.assertEqual(reading.object.phrase if reading.object else None, "green cube")
        return reading

    def test_a_german_object_phrase_is_asked_again(self) -> None:
        self._retried(_answer(object="grüner Würfel", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste"), problem_says="not English")

    def test_a_german_object_phrase_without_umlauts_is_asked_again(self) -> None:
        self._retried(_answer(object="gruener wuerfel", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste"), problem_says="not English")

    def test_a_german_place_phrase_is_asked_again(self) -> None:
        self._retried(_answer(object="green cube", object_said="den grünen Würfel", place="blaue Kiste",
                              place_said="in die blaue Kiste"), problem_says="not English")

    def test_a_quantifier_in_the_object_is_asked_again(self) -> None:
        self._retried(_answer(object="all green cubes", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste"), problem_says="quantifier")

    def test_an_answer_without_json_is_asked_again(self) -> None:
        self._retried("Sure! You want the green cube in the blue bin.", problem_says="no JSON object")

    def test_a_missing_intent_is_asked_again(self) -> None:
        """Every other key may be left out, as "not said" (2026-10-08); the intent never."""
        incomplete = json.loads(self.GOOD)
        del incomplete["intent"]
        self._retried(json.dumps(incomplete), problem_says="'intent' is missing")

    def test_an_extra_key_is_asked_again(self) -> None:
        extra = json.loads(self.GOOD)
        extra["colour"] = "green"
        self._retried(json.dumps(extra), problem_says="'colour'")

    def test_a_scope_outside_the_two_is_asked_again(self) -> None:
        self._retried(_answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste", scope="twice"), problem_says="'scope'")

    def test_a_count_that_is_no_whole_number_is_asked_again(self) -> None:
        self._retried(_answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste", count="3"), problem_says="'count'")

    def test_a_place_and_a_pose_together_are_asked_again(self) -> None:
        self._retried(_answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                              place_said="in die blaue Kiste", place_pose="ablage_links"),
                      problem_says="both set")

    def test_an_overlong_phrase_is_asked_again(self) -> None:
        self._retried(_answer(object="green " * (MAX_PHRASE_CHARS // 6 + 2) + "cube", object_said="den grünen Würfel",
                              place="blue bin", place_said="in die blaue Kiste"),
                      problem_says=f"longer than {MAX_PHRASE_CHARS}")

    def test_two_bad_answers_are_not_understood_and_fill_nothing(self) -> None:
        second = _answer(object="grüner Würfel")
        ask = _Canned("no idea", second)
        reading = understand(self.SENTENCE, ask=ask, poses=POSES)
        self.assertFalse(reading.understood)
        self.assertEqual(reading.intent, "none")
        self.assertIn("not English", reading.reason)
        self.assertEqual(reading.notes, ("retried",))
        self.assertEqual(reading.attempts, 2)
        self.assertEqual(reading.raw, second, "the raw answer kept is the last one")
        self.assertEqual((reading.object, reading.place, reading.place_pose, reading.scope, reading.return_to),
                         (None, None, None, None, None), "a half-read card would look like a reading")

    def test_the_reader_never_asks_a_third_time(self) -> None:
        ask = _Canned("bad", "bad", "bad", "bad")
        understand(self.SENTENCE, ask=ask, poses=POSES)
        self.assertEqual(len(ask.calls), 2)

    def test_a_good_first_answer_is_asked_once(self) -> None:
        ask = _Canned(self.GOOD, self.GOOD)
        understand(self.SENTENCE, ask=ask, poses=POSES)
        self.assertEqual(len(ask.calls), 1)

    def test_stop_and_none_answers_are_not_checked_for_phrases(self) -> None:
        """Their phrases are thrown away, so a German leftover in them costs no second question."""
        ask = _Canned(_answer(intent="stop", object="Würfel"))
        reading = understand("Stopp", ask=ask, poses=POSES)
        self.assertEqual((reading.understood, reading.intent, reading.object), (True, "stop", None))
        self.assertEqual(len(ask.calls), 1)


class GermanWordsAreAskedAgainTests(_ForgetsTheSharedHolder):
    """German nouns with no umlaut and no word the router knows ("schrauben", "mutter") pass the router's analysis
    as English, and GroundingDINO grounds German badly. Two checks of the reader's own catch them: a phrase made only
    of the German command's own words was copied, not translated; and the cell's German part, colour and material
    words are never English."""

    def test_a_german_noun_copied_from_the_sentence_is_asked_again(self) -> None:
        cases = (
            ("Nimm alle Schrauben und leg sie auf Ablage links",
             _answer(object="schrauben", object_said="Schrauben", place_pose="ablage_links", scope="until_empty"),
             _answer(object="screw", object_said="Schrauben", place_pose="ablage_links", scope="until_empty"),
             "screw"),
            ("Nimm die Mutter", _answer(object="mutter", object_said="die Mutter"),
             _answer(object="nut", object_said="die Mutter"), "nut"),
        )
        for sentence, first, second, phrase in cases:
            with self.subTest(sentence=sentence):
                ask = _Canned(first, second)
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertEqual(len(ask.calls), 2, "the copied German noun went to the detector")
                self.assertIn("English", ask.calls[1][1])
                self.assertTrue(reading.understood)
                self.assertEqual(reading.object.phrase if reading.object else None, phrase)
                self.assertEqual(reading.notes, ("retried",))

    def test_a_german_part_word_is_never_english(self) -> None:
        """Not copied word for word ("blau" from "blauen"), and still German."""
        sentence = "Leg den Würfel in den blauen Kasten"
        bad = _answer(object="cube", object_said="den Würfel", place="blau kasten", place_said="in den blauen Kasten")
        ask = _Canned(bad, bad)
        reading = understand(sentence, ask=ask, poses=POSES)
        self.assertEqual(len(ask.calls), 2)
        self.assertFalse(reading.understood)
        self.assertIn("not English", reading.reason)

    def test_a_german_part_word_is_refused_on_the_second_answer_too(self) -> None:
        ask = _Canned(_answer(object="schrauben", object_said="Schrauben"),
                      _answer(object="schraube", object_said="Schrauben"))
        reading = understand("Nimm die Schrauben", ask=ask, poses=POSES)
        self.assertFalse(reading.understood)
        self.assertIsNone(reading.object)

    def test_a_copied_word_said_again_is_taken_as_a_loanword(self) -> None:
        """The model kept the word after it was asked to translate: a word German took from English stays."""
        answer = _answer(object="joystick", object_said="den Joystick")
        ask = _Canned(answer, answer)
        reading = understand("Nimm den Joystick", ask=ask, poses=POSES)
        self.assertEqual(len(ask.calls), 2)
        self.assertTrue(reading.understood)
        self.assertEqual(reading.object.phrase if reading.object else None, "joystick")
        self.assertEqual(reading.notes, ("retried",))

    def test_a_word_that_is_the_same_in_both_languages_costs_no_question(self) -> None:
        for sentence, answer in (
            ("Leg den Würfel in die Box", _answer(object="cube", object_said="den Würfel", place="box",
                                                  place_said="in die Box")),
            ("Nimm den Ball", _answer(object="ball", object_said="den Ball")),
            ("Nimm die M6", _answer(object="m6", object_said="die M6")),
        ):
            with self.subTest(sentence=sentence):
                ask = _Canned(answer)
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood)
                self.assertEqual(len(ask.calls), 1)

    def test_an_english_command_names_its_part_in_its_own_words(self) -> None:
        ask = _Canned(_answer(object="joystick", object_said="the joystick"))
        reading = understand("Pick the joystick", ask=ask, poses=POSES)
        self.assertEqual((reading.understood, len(ask.calls)), (True, 1))

    def test_a_german_pose_label_does_not_make_an_english_command_german(self) -> None:
        """ "Ablage links" carries words the router counts as German; the command around it is still English."""
        ask = _Canned(_answer(object="red cylinder", object_said="the red cylinder", place_pose="ablage_links",
                              return_to="wartepose"))
        reading = understand("Put the red cylinder on Ablage links and go back to the Wartepose", ask=ask,
                             poses=POSES)
        self.assertEqual((reading.understood, len(ask.calls)), (True, 1))
        self.assertEqual(reading.object.phrase if reading.object else None, "red cylinder")

    def test_a_translated_phrase_that_shares_one_word_is_no_copy(self) -> None:
        ask = _Canned(_answer(object="red ball", object_said="den roten Ball"))
        reading = understand("Nimm den roten Ball", ask=ask, poses=POSES)
        self.assertEqual((reading.understood, len(ask.calls)), (True, 1))


class APoseNamedAsAPlaceTests(_ForgetsTheSharedHolder):
    """Measured on Qwen3-VL-4B (2026-10-01) with the instruction's first wording: for the owner's "leg sie auf
    Ablage links" it answered a place "left shelf" with the words "auf Ablage links" AND the pose `ablage_links`,
    and gave the same answer when asked again. Those words are the pose's label: the answer means the pose, so
    the card gets the pose at once. The reworded instruction fixed the model's answer; this is the backstop.

    Only the place's own words decide, and only when they are the label and nothing more: without words, or with
    words that say more ("neben Ablage links"), a place stays a place, and beside a pose it is asked again."""

    def _read(self, sentence: str, *answers: str) -> tuple[CommandReading, _Canned]:
        ask = _Canned(*answers)
        return understand(sentence, ask=ask, poses=POSES), ask

    def test_a_pose_label_read_as_a_place_is_the_pose(self) -> None:
        reading, ask = self._read(
            "Nimm alle Schrauben und leg sie auf Ablage links",
            _answer(object="screw", object_said="Schrauben", place="left shelf", place_said="auf Ablage links",
                    place_pose="ablage_links", scope="until_empty"),
        )
        self.assertTrue(reading.understood)
        self.assertEqual(len(ask.calls), 1, "no second question for an answer that means the pose")
        self.assertIsNone(reading.place, "the camera has nothing to search for")
        self.assertEqual(reading.place_pose, "ablage_links")
        self.assertEqual(reading.scope, "until_empty")
        self.assertEqual(reading.notes, ())

    def test_the_same_in_english(self) -> None:
        reading, ask = self._read(
            "Take all screws and put them on Ablage links",
            _answer(object="screw", object_said="all screws", place="left shelf", place_said="on Ablage links",
                    place_pose="ablage_links", scope="until_empty"),
        )
        self.assertEqual((reading.understood, reading.place, reading.place_pose), (True, None, "ablage_links"))
        self.assertEqual(len(ask.calls), 1)

    def test_a_place_without_words_beside_a_pose_is_asked_again(self) -> None:
        """Without its own words a place cannot be shown to be the pose: the sentence may name the pose only as
        where the arm goes AFTERWARDS, and dropping the bin would lose the place and the return together."""
        sentence = "Leg den grünen Würfel in die blaue Kiste und fahr danach zur Ablage links"
        reading, ask = self._read(
            sentence,
            _answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                    place_pose="ablage_links"),
            _answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                    place_said="in die blaue Kiste", return_to="ablage_links"),
        )
        self.assertEqual(len(ask.calls), 2)
        self.assertIn("both set", ask.calls[1][1])
        self.assertEqual(reading.place.phrase if reading.place else None, "blue bin")
        self.assertEqual((reading.place_pose, reading.return_to), (None, "ablage_links"))

    def test_a_place_whose_own_words_are_a_pose_label_is_that_pose(self) -> None:
        """The same answer without place_pose: "auf Ablage links" names the taught pose, not a spot to search."""
        reading, ask = self._read(
            "Leg den Becher auf Ablage links",
            _answer(object="cup", object_said="den Becher", place="left shelf", place_said="auf Ablage links"),
        )
        self.assertEqual(len(ask.calls), 1)
        self.assertEqual((reading.place, reading.place_pose), (None, "ablage_links"))
        self.assertEqual(reading.notes, ())

    def test_a_place_beside_a_pose_stays_a_place(self) -> None:
        """Words that name more than the pose ("the bin next to Ablage links") are a place the camera finds."""
        reading, ask = self._read(
            "Leg den Becher in die Kiste neben Ablage links",
            _answer(object="cup", object_said="den Becher", place="bin next to left shelf",
                    place_said="in die Kiste neben Ablage links"),
        )
        self.assertEqual(len(ask.calls), 1)
        self.assertEqual(reading.place.phrase if reading.place else None, "bin next to left shelf")
        self.assertIsNone(reading.place_pose)

    def test_a_place_naming_one_pose_beside_another_pose_is_asked_again(self) -> None:
        reading, ask = self._read(
            "Leg den Becher auf die Wartepose",
            _answer(object="cup", object_said="den Becher", place="waiting pose", place_said="auf die Wartepose",
                    place_pose="ablage_links"),
            _answer(object="cup", object_said="den Becher", place_pose="wartepose"),
        )
        self.assertEqual(len(ask.calls), 2)
        self.assertEqual((reading.place, reading.place_pose), (None, "wartepose"))

    def test_a_place_whose_words_are_home_is_no_place(self) -> None:
        """Home is where the arm returns, never where a part is let go: not a camera search either."""
        reading, ask = self._read(
            "Leg den Becher auf Home",
            _answer(object="cup", object_said="den Becher", place="home", place_said="auf Home"),
        )
        self.assertEqual(len(ask.calls), 1)
        self.assertEqual((reading.place, reading.place_pose), (None, None))
        self.assertEqual(reading.notes, ("pose_unknown",))

    def test_a_german_place_that_is_the_pose_costs_no_second_question(self) -> None:
        """Its phrase is thrown away, so its language does not matter."""
        reading, ask = self._read(
            "Leg den Becher auf Ablage links",
            _answer(object="cup", object_said="den Becher", place="Ablage links", place_said="auf Ablage links",
                    place_pose="ablage_links"),
        )
        self.assertEqual((reading.understood, reading.place, reading.place_pose), (True, None, "ablage_links"))
        self.assertEqual(len(ask.calls), 1)

    def test_a_place_whose_words_name_something_else_still_contradicts_the_pose(self) -> None:
        reading, ask = self._read(
            "Leg den Becher in die blaue Kiste",
            _answer(object="cup", object_said="den Becher", place="blue bin", place_said="in die blaue Kiste",
                    place_pose="ablage_links"),
            _answer(object="cup", object_said="den Becher", place="blue bin", place_said="in die blaue Kiste"),
        )
        self.assertEqual(len(ask.calls), 2)
        self.assertIn("both set", ask.calls[1][1])
        self.assertEqual((reading.place.phrase if reading.place else None, reading.place_pose), ("blue bin", None))

    def test_home_is_never_the_pose_a_place_means(self) -> None:
        """Measured on Qwen3-VL-4B: place "home" with place_pose "home", and the same again when asked once more.
        Home is never a place, so both go and the card asks: no second question."""
        reading, ask = self._read(
            "Leg den Becher auf Home",
            _answer(object="cup", object_said="den Becher", place="home", place_said="auf Home", place_pose="home"),
        )
        self.assertEqual(len(ask.calls), 1)
        self.assertTrue(reading.understood)
        self.assertEqual((reading.place, reading.place_pose), (None, None))
        self.assertEqual(reading.notes, ("pose_unknown",))

    def test_home_as_a_place_beside_another_pose_is_asked_again_for_what_it_is(self) -> None:
        """A pose's label is a name in no language: the retry names the contradiction, not a copied German word."""
        reading, ask = self._read(
            "Leg den Becher auf Home",
            _answer(object="cup", object_said="den Becher", place="home", place_said="auf Home",
                    place_pose="ablage_links"),
            _answer(object="cup", object_said="den Becher"),
        )
        self.assertEqual(len(ask.calls), 2)
        self.assertIn("both set", ask.calls[1][1])
        self.assertNotIn("German", ask.calls[1][1])
        self.assertEqual((reading.place, reading.place_pose), (None, None))


class SoftNotesTests(_ForgetsTheSharedHolder):
    """What the card marks "bitte prüfen": understood, but not everything was found in the sentence."""

    def _read(self, sentence: str, answer: str) -> CommandReading:
        return understand(sentence, ask=_Canned(answer), poses=POSES)

    def test_an_object_the_sentence_does_not_name_is_unverified(self) -> None:
        reading = self._read("Leg das Teil in die blaue Kiste",
                             _answer(object="green cube", object_said="den grünen Würfel",
                                     place="blue bin", place_said="in die blaue Kiste"))
        self.assertTrue(reading.understood)
        self.assertFalse(reading.object.verified if reading.object else True)
        self.assertEqual(reading.notes, ("object_not_in_sentence",))

    def test_an_object_without_its_words_is_unverified(self) -> None:
        reading = self._read("Leg den grünen Würfel in die blaue Kiste",
                             _answer(object="green cube", place="blue bin", place_said="in die blaue Kiste"))
        self.assertEqual(reading.object, PhraseReading(phrase="green cube", said=None, verified=False))
        self.assertEqual(reading.notes, ("object_not_in_sentence",))

    def test_a_place_the_sentence_does_not_name_is_unverified(self) -> None:
        reading = self._read("Leg den grünen Würfel weg",
                             _answer(object="green cube", object_said="den grünen Würfel",
                                     place="blue bin", place_said="in die blaue Kiste"))
        self.assertFalse(reading.place.verified if reading.place else True)
        self.assertEqual(reading.notes, ("place_not_in_sentence",))

    def test_the_words_are_found_whatever_their_case_and_umlaut_spelling(self) -> None:
        reading = self._read("LEG DEN GRUENEN WUERFEL IN DIE BLAUE KISTE.",
                             _answer(object="green cube", object_said="den grünen Würfel",
                                     place="blue bin", place_said="in die blaue Kiste"))
        self.assertEqual(reading.notes, ())

    def test_a_pose_nobody_taught_is_dropped_and_noted(self) -> None:
        reading = self._read("Leg den Becher auf Ablage rechts",
                             _answer(object="cup", object_said="den Becher", place_pose="ablage_rechts"))
        self.assertIsNone(reading.place_pose)
        self.assertEqual(reading.notes, ("pose_unknown",))

    def test_home_is_no_place_to_put_a_part(self) -> None:
        reading = self._read("Leg den Becher auf Home",
                             _answer(object="cup", object_said="den Becher", place_pose="home"))
        self.assertIsNone(reading.place_pose)
        self.assertIn("pose_unknown", reading.notes)

    def test_an_unknown_return_pose_is_dropped_and_noted(self) -> None:
        reading = self._read("Nimm den Becher und fahr in die Parkpose",
                             _answer(object="cup", object_said="den Becher", return_to="parkpose"))
        self.assertIsNone(reading.return_to)
        self.assertEqual(reading.notes, ("pose_unknown",))

    def test_a_pose_the_sentence_does_not_name_is_unverified(self) -> None:
        reading = self._read("Leg den Becher weg",
                             _answer(object="cup", object_said="den Becher", place_pose="ablage_links"))
        self.assertEqual(reading.place_pose, "ablage_links")
        self.assertEqual(reading.notes, ("place_not_in_sentence",))

    def test_a_name_in_another_case_is_still_the_name(self) -> None:
        reading = self._read("Leg den Becher auf Ablage links",
                             _answer(object="cup", object_said="den Becher", place_pose="ABLAGE_LINKS"))
        self.assertEqual(reading.place_pose, "ablage_links")

    def test_notes_come_in_one_order(self) -> None:
        reading = understand(
            "Nimm drei Teile",
            ask=_Canned("nonsense", _answer(object="cube", object_said="Würfel", place="red tray",
                                            place_said="aufs rote Tablett", return_to="nowhere", count=3)),
            poses=POSES,
        )
        self.assertEqual(reading.notes, ("object_not_in_sentence", "place_not_in_sentence", "pose_unknown",
                                         "count_not_supported", "retried"))

    def test_a_label_two_poses_share_maps_to_neither(self) -> None:
        """Two labels that differ only in case cannot say which pose was meant; the exact name still can."""
        poses = {"Ablage": "ablage_a", "ablage": "ablage_b"}
        reading = understand("Leg den Becher auf die Ablage", ask=_Canned(
            _answer(object="cup", object_said="den Becher", place_pose="Ablage")), poses=poses)
        self.assertIsNone(reading.place_pose)
        self.assertIn("pose_unknown", reading.notes)
        reading = understand("Leg den Becher auf die Ablage", ask=_Canned(
            _answer(object="cup", object_said="den Becher", place_pose="ablage_b")), poses=poses)
        self.assertEqual(reading.place_pose, "ablage_b")


#: What Qwen3-VL-4B-Instruct answered on the dev box's RTX 5080 (2026-10-01), through the instruction as it stands:
#: a probe of the owner's sentences, a reviewer's adversarial ones, and the fix round's. Each row is the
#: sentence, the keys the model set (every other key was null, "" or "once"), and the card the reader makes of it:
#: object, place, place pose, scope, return, count and notes; or only the intent of a sentence that is no task.
_REAL_ANSWERS: tuple[tuple[str, dict[str, Any], tuple[Any, ...]], ...] = (
    ("Leg den grünen Würfel in die blaue Kiste",
     dict(object="green cube", object_said="den grünen Würfel", place="blue bin", place_said="in die blaue Kiste"),
     ("green cube", "blue bin", None, "once", None, None, ())),
    ("Räum die Kiste aus", dict(scope="until_empty"), (None, None, None, "until_empty", None, None, ())),
    ("Nimm alle Schrauben und leg sie auf Ablage links",
     dict(object="screw", object_said="Schrauben", place_pose="ablage_links", scope="until_empty"),
     ("screw", None, "ablage_links", "until_empty", None, None, ())),
    ("Put the green cube in the blue bin",
     dict(object="green cube", object_said="the green cube", place="blue bin", place_said="in the blue bin"),
     ("green cube", "blue bin", None, "once", None, None, ())),
    ("Take all screws and put them on Ablage links",
     dict(object="screw", object_said="all screws", place_pose="ablage_links", scope="until_empty"),
     ("screw", None, "ablage_links", "until_empty", None, None, ())),
    ("Nimm den roten Zylinder und fahr danach in die Wartepose",
     dict(object="red cylinder", object_said="den roten Zylinder", return_to="wartepose"),
     ("red cylinder", None, None, "once", "wartepose", None, ())),
    ("Put the red cylinder on Ablage links and go back to the Wartepose",
     dict(object="red cylinder", object_said="the red cylinder", place_pose="ablage_links", return_to="wartepose"),
     ("red cylinder", None, "ablage_links", "once", "wartepose", None, ())),
    ("Stopp!", dict(intent="stop"), ("stop",)),
    ("Hallo Willy, wie geht es dir?", dict(intent="none"), ("none",)),
    ("Nimm drei Würfel", dict(object="cube", object_said="Würfel", count=3),
     ("cube", None, None, "once", None, 3, ("count_not_supported",))),
    ("Leg alle grünen Würfel in die graue Box und fahr danach nach Home",
     dict(object="green cube", object_said="alle grünen Würfel", place="gray box", place_said="in die graue Box",
          scope="until_empty", return_to="home"),
     ("green cube", "gray box", None, "until_empty", "home", None, ())),
    ("pick the small metal bracket and put it on Ablage links",
     dict(object="small metal bracket", object_said="the small metal bracket", place_pose="ablage_links"),
     ("small metal bracket", None, "ablage_links", "once", None, None, ())),
    ("Räum die blaue Kiste leer", dict(scope="until_empty"), (None, None, None, "until_empty", None, None, ())),
    ("Nimm den Becher", dict(object="cup", object_said="den Becher"), ("cup", None, None, "once", None, None, ())),
    ("Nimm die Unterlegscheibe", dict(object="washer", object_said="die Unterlegscheibe"),
     ("washer", None, None, "once", None, None, ())),
    ("Nimm die Sechskantmutter und leg sie in die Kiste",
     dict(object="hex nut", object_said="die Sechskantmutter", place="bin", place_said="in die Kiste"),
     ("hex nut", "bin", None, "once", None, None, ())),
    ("Nimm alle Muttern", dict(object="nut", object_said="Muttern", scope="until_empty"),
     ("nut", None, None, "until_empty", None, None, ())),
    ("Nimm drei Würfel und leg sie auf Ablage links",
     dict(object="cube", object_said="Würfel", place_pose="ablage_links", count=3),
     ("cube", None, "ablage_links", "once", None, 3, ("count_not_supported",))),
    ("Nimm zwei Schrauben", dict(object="screw", object_said="Schrauben", count=2),
     ("screw", None, None, "once", None, 2, ("count_not_supported",))),
    ("Räum alles aus der Kiste", dict(scope="until_empty"), (None, None, None, "until_empty", None, None, ())),
    ("Nimm irgendwas", dict(object_said="irgendwas"), (None, None, None, "once", None, None, ())),
    # The one card that differs from the one measured: "part" grounds anything, so the card asks (Q7 A+).
    ("Nimm das Teil und leg es auf Ablage links",
     dict(object="part", object_said="das Teil", place_pose="ablage_links"),
     (None, None, "ablage_links", "once", None, None, ())),
    ("Leg den Würfel in die blaue Kiste und fahr danach zur Ablage links",
     dict(object="cube", object_said="den Würfel", place="blue bin", place_said="in die blaue Kiste",
          return_to="ablage_links"),
     ("cube", "blue bin", None, "once", "ablage_links", None, ())),
    # The model's second answer; its first (not recorded) was asked again.
    ("Leg den Becher auf die linke Ablage", dict(object="cup", object_said="den Becher", place_pose="ablage_links"),
     ("cup", None, "ablage_links", "once", None, None, ("place_not_in_sentence",))),
    ("Nimm den Kugelschreiber", dict(object="ballpoint pen", object_said="den Kugelschreiber"),
     ("ballpoint pen", None, None, "once", None, None, ())),
    ("Leg die Zange in den grauen Kasten",
     dict(object="pliers", object_said="die Zange", place="gray box", place_said="in den grauen Kasten"),
     ("pliers", "gray box", None, "once", None, None, ())),
    # The fix round's probe, through the same instruction (gpu_fix_probe, 2026-10-01).
    ("Nimm die Mutter", dict(object="nut", object_said="die Mutter"), ("nut", None, None, "once", None, None, ())),
    ("Nimm die Schrauben", dict(object="screw", object_said="die Schrauben"),
     ("screw", None, None, "once", None, None, ())),
    ("Leg den Würfel in die Box",
     dict(object="cube", object_said="den Würfel", place="box", place_said="in die Box"),
     ("cube", "box", None, "once", None, None, ())),
    ("Leg den Becher auf Ablage links", dict(object="cup", object_said="den Becher", place_pose="ablage_links"),
     ("cup", None, "ablage_links", "once", None, None, ())),
    ("Leg den Würfel in den blauen Kasten",
     dict(object="cube", object_said="den Würfel", place="blue box", place_said="in den blauen Kasten"),
     ("cube", "blue box", None, "once", None, None, ())),
    ("Leg den grünen Würfel in die blaue Kiste und fahr danach zur Ablage links",
     dict(object="green cube", object_said="den grünen Würfel", place="blue bin", place_said="in die blaue Kiste",
          return_to="ablage_links"),
     ("green cube", "blue bin", None, "once", "ablage_links", None, ())),
    # The model said until_empty and three: the card reads one part.
    ("Nimm alle drei Würfel", dict(object="cube", object_said="Würfel", scope="until_empty", count=3),
     ("cube", None, None, "once", None, 3, ("count_not_supported",))),
    # Home as the place and as the place pose: Home is no place, so the card asks.
    ("Leg den Becher auf Home",
     dict(object="cup", object_said="den Becher", place="home", place_said="auf Home", place_pose="home"),
     ("cup", None, None, "once", None, None, ("pose_unknown",))),
    # The model dropped "neben Ablage links" itself; the card shows the bin, for a person to check.
    ("Leg den Becher in die Kiste neben Ablage links",
     dict(object="cup", object_said="den Becher", place="bin", place_said="in die Kiste"),
     ("cup", "bin", None, "once", None, None, ())),
)

#: Two sentences the real model needed a second question for (gpu_fix_probe, 2026-10-01): its first answer, why it
#: went back, its second answer, and the card.
_REAL_RETRIES: tuple[tuple[str, dict[str, Any], str, dict[str, Any], tuple[Any, ...]], ...] = (
    ("Leg den Becher auf die linke Ablage",
     dict(object="cup", object_said="den Becher", place="left shelf", place_said="auf die linke Ablage",
          place_pose="ablage_links"),
     "both set",
     dict(object="cup", object_said="den Becher", place_pose="ablage_links"),
     ("cup", None, "ablage_links", "once", None, None, ("place_not_in_sentence", "retried"))),
    ("Nimm den Joystick",
     dict(object="joystick", object_said="den Joystick"),
     "German words",
     dict(object="joystick", object_said="den Joystick"),
     ("joystick", None, None, "once", None, None, ("retried",))),
)


class TheRealModelsAnswersTests(_ForgetsTheSharedHolder):
    """The real model's answers, replayed: each is read on the first question, as the model meant it.

    The reader's own checks (a copied German word, the cell's German words, a word for any part, a place that is a
    pose) exist for the answers that go wrong; on these good ones they must not cost a second question."""

    def test_each_answer_is_read_on_the_first_question(self) -> None:
        for sentence, fields, want in _REAL_ANSWERS:
            with self.subTest(sentence=sentence):
                ask = _Canned(_answer(**fields))
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood, reading.reason)
                self.assertEqual(len(ask.calls), 1, "a good answer was asked again")
                if len(want) == 1:
                    self.assertEqual(reading.intent, want[0])
                    continue
                card = (reading.object.phrase if reading.object else None,
                        reading.place.phrase if reading.place else None,
                        reading.place_pose, reading.scope, reading.return_to, reading.count, reading.notes)
                self.assertEqual(card, want)

    def test_the_answers_it_was_asked_again_for_read_on_the_second_question(self) -> None:
        for sentence, first, why, second, want in _REAL_RETRIES:
            with self.subTest(sentence=sentence):
                ask = _Canned(_answer(**first), _answer(**second))
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertEqual(len(ask.calls), 2)
                self.assertIn(why, ask.calls[1][1])
                self.assertTrue(reading.understood, reading.reason)
                card = (reading.object.phrase if reading.object else None,
                        reading.place.phrase if reading.place else None,
                        reading.place_pose, reading.scope, reading.return_to, reading.count, reading.notes)
                self.assertEqual(card, want)

    def test_each_answer_written_as_the_instruction_asks_now_reads_the_same_card(self) -> None:
        """The same keys the model set, compact and with nothing else: what the 8B is asked for since 2026-10-08."""
        for sentence, fields, want in _REAL_ANSWERS:
            with self.subTest(sentence=sentence):
                ask = _Canned(_sparse(**fields))
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertEqual((True, 1), (reading.understood, len(ask.calls)), reading.reason)
                self.assertEqual(want, (reading.intent,) if len(want) == 1 else _card_of(reading))
                self.assertEqual((None, None), (reading.which, reading.source))


def _card_of(reading: CommandReading) -> tuple[Any, ...]:
    """A task's card as the measured rows write it: object, place, place pose, scope, return, count, notes."""
    return (reading.object.phrase if reading.object else None, reading.place.phrase if reading.place else None,
            reading.place_pose, reading.scope, reading.return_to, reading.count, reading.notes)


#: The measured sentences the known sentences' table answers itself; the model reads every other one.
_KNOWN_OF_THE_MEASURED: frozenset[str] = frozenset({
    "Leg den grünen Würfel in die blaue Kiste",
    "Nimm alle Schrauben und leg sie auf Ablage links",
    "Put the green cube in the blue bin",
    "Take all screws and put them on Ablage links",
    "Leg den Würfel in die Box",
    "Leg den Becher auf Ablage links",
    "Leg den Würfel in den blauen Kasten",
})


class TheKnownSentencesTests(_ForgetsTheSharedHolder):
    """The owner, 2026-10-08 ("speed first"): the everyday sentences are answered without the model, as the model
    answered them, and the reader makes the card of that answer with its own checks; any other sentence goes to the
    model."""

    def test_each_measured_sentence_it_takes_reads_as_the_model_read_it(self) -> None:
        taken = set()
        for sentence, _fields, want in _REAL_ANSWERS:
            with self.subTest(sentence=sentence):
                reading = read_known(sentence, poses=POSES)
                if reading is None:
                    continue
                taken.add(sentence)
                self.assertEqual(want, (reading.intent,) if len(want) == 1 else _card_of(reading))
        self.assertEqual(_KNOWN_OF_THE_MEASURED, taken)

    def test_the_sentences_the_model_was_asked_twice_for_go_to_the_model(self) -> None:
        for sentence, *_ in _REAL_RETRIES:
            with self.subTest(sentence=sentence):
                self.assertIsNone(read_known(sentence, poses=POSES))

    def test_the_instruction_examples_it_takes_are_answered_word_for_word(self) -> None:
        offered = dict(EXAMPLE_POSES)
        taken = []
        for sentence, answer in COMMAND_EXAMPLES:
            reading = read_known(sentence, poses=offered)
            if reading is None:
                continue
            taken.append(sentence)
            with self.subTest(sentence=sentence):
                self.assertEqual(answer, json.loads(reading.raw))
        self.assertEqual(["nimm den grünen Würfel und leg ihn in die blaue Kiste", "alle Schrauben in die Kiste",
                          "leg den Becher auf Ablage links"], taken)

    def test_the_cells_sentence_of_the_morning_reads_without_a_model(self) -> None:
        reading = read_known("Alle grauen Würfel in die Gelbe Kiste.", poses=POSES)
        assert reading is not None
        self.assertEqual(("gray cube", "yellow bin", None, "until_empty", None, None, ()), _card_of(reading))
        self.assertEqual((PhraseReading("gray cube", "grauen Würfel", True),
                          PhraseReading("yellow bin", "in die Gelbe Kiste", True)), (reading.object, reading.place))
        self.assertEqual((KNOWN_SENTENCE_MODEL, 0, False, False),
                         (reading.model_id, reading.attempts, reading.loaded_now, reading.remembered))
        self.assertTrue(reading.known)

    def test_hallo_willy_is_a_greeting_without_a_model(self) -> None:
        for sentence in ("Hallo Willy", "Hallo Willy!", "Willy, wink mal!"):
            with self.subTest(sentence=sentence):
                reading = read_known(sentence, poses=POSES)
                assert reading is not None
                self.assertEqual((True, "none", True, KNOWN_SENTENCE_MODEL),
                                 (reading.understood, reading.intent, reading.greeting, reading.model_id))

    def test_a_command_with_a_greeting_in_front_goes_to_the_model(self) -> None:
        self.assertIsNone(read_known("Hallo Willy, nimm den roten Würfel", poses=POSES))

    def test_near_misses_go_to_the_model(self) -> None:
        for sentence in (
            "Nimm die Würfel und leg sie in die Kiste",  # several without "alle": the model read that as once
            "Nimm die Schrauben und leg sie in die Kiste",
            "Nimm zwei Würfel und leg sie in die Kiste",  # a count
            "Nimm alle drei Würfel und leg sie in die Kiste",
            "Nimm einen Würfel und leg ihn in die Kiste",  # "einen" may be a count of one
            "Put a red cube into the blue bin",
            "Leg den Würfel nicht in die Kiste",  # a negation
            "Leg den Würfel in die Kiste und fahr danach nach Home",  # where the arm goes after
            "Leg den Würfel in die Kiste neben Ablage links",  # a place beside a pose
            "Alle grauen Würfel auf der schwarzen Matte in die gelbe Kiste",  # where the parts lie
            "Alle Teile in die gelbe Kiste",  # a word for any part
            "Leg den Würfel auf Home",  # Home is where the arm returns
            "Nimm den Würfel in der Kiste",  # the cube that lies in the bin
            "Nimm den Würfel aus der Kiste",
            "Take the cube in the bin",  # the cube that may lie in the bin
            "All gray cubes in the yellow bin",
            "Nimm den Becher auf Ablage links",  # the cup that may stand on the pose
            "Leg den grauen cube in die Kiste",  # two languages
            "Put all grey cubes into the yellow bin",  # a spelling the model was not measured on
            "Leg bitte den Würfel in die Kiste",
            "Willy, leg den Würfel in die Kiste",
            "Leg den Würfel auf die Kiste",  # a bin is put into
            "Leg den Würfel in Kiste",
            "Leg den Würfel in der Kiste",
            "Nimm den Würfel",  # no place: the model reads it
            "Räum die Kiste aus",
            "Stopp!",
        ):
            with self.subTest(sentence=sentence):
                self.assertIsNone(read_known(sentence, poses=POSES))

    def test_a_pose_named_by_its_name_or_label_in_any_case_is_that_pose(self) -> None:
        for sentence in ("Leg den Becher auf ablage_links", "LEG DEN BECHER AUF ABLAGE LINKS",
                         "Leg den Becher auf die Ablage links", "Lege den Becher in die Wartepose"):
            with self.subTest(sentence=sentence):
                reading = read_known(sentence, poses=POSES)
                assert reading is not None
                self.assertIn(reading.place_pose, ("ablage_links", "wartepose"))
                self.assertEqual((), reading.notes)

    def test_an_answer_the_readers_checks_refuse_goes_to_the_model(self) -> None:
        """The table's answers pass the checks by construction; were one to fail, it would never be a card."""
        from unittest import mock

        german = {"intent": "task", "object": "grüner würfel", "object_said": "den grünen Würfel", "place": None,
                  "place_said": None, "place_pose": "ablage_links", "scope": "once", "count": None,
                  "return_to": None}
        with mock.patch("src.models.vlm.known.known_answer", return_value=german):
            self.assertIsNone(read_known("Leg den grünen Würfel auf Ablage links", poses=POSES))

    def test_the_answer_is_the_instructions_json(self) -> None:
        """As the instruction writes an answer since 2026-10-08: only what the sentence says, a scope only for all."""
        from src.models.vlm.command import _Offered

        offered = _Offered.of(POSES)
        self.assertEqual(_example(object="cube", object_said="den Würfel", place="box", place_said="in die Box"),
                         known_answer("Leg den Würfel in die Box", offered))
        self.assertEqual(_example(object="screw", object_said="Schrauben", place="bin", place_said="in die Kiste",
                                  scope="until_empty"),
                         known_answer("Alle Schrauben in die Kiste", offered))

    def test_the_door_asks_the_table_first_and_loads_nothing(self) -> None:
        """Before the load rule: a GroundingDINO cell whose model is not loaded reads a known sentence, and refuses
        any other as before."""
        from src.models.vlm import CommandRefused
        from tests.test_vlm_holder import _holder, _Loads, _models

        loads = _Loads()
        holder = _holder(loads)
        self.addCleanup(holder.forget)
        reading = read_command("Leg den Würfel in die Box", models=_models(backend="grounded_sam"), poses=POSES,
                               weights_present=True, holder=holder, known=True)
        self.assertEqual(("cube", "box", KNOWN_SENTENCE_MODEL), (reading.object.phrase if reading.object else None,
                                                                 reading.place.phrase if reading.place else None,
                                                                 reading.model_id))
        self.assertEqual(0, loads.calls)
        with self.assertRaises(CommandRefused) as refused:
            read_command("Leg den Würfel in die Box", models=_models(backend="grounded_sam"), poses=POSES,
                         weights_present=True, holder=holder)
        self.assertEqual("vlm_not_loaded", refused.exception.code, "without the switch the table is not asked")


class RememberedAnswersTests(_ForgetsTheSharedHolder):
    """R2 (2026-10-08): a question the loaded copy answered before is answered from memory, keyed by the weights, the
    instruction and the user message, and checked again; a copy that is not loaded is asked (and loaded) as before."""

    SENTENCE = "Nimm alle Schrauben und leg sie auf Ablage links"
    ANSWER = ('"intent": "task", "object": "screw", "object_said": "Schrauben", "place": null, "place_said": null, '
              '"place_pose": "ablage_links", "scope": "until_empty", "count": null, "return_to": null}')

    def setUp(self) -> None:
        super().setUp()
        forget_remembered_answers()
        self.addCleanup(forget_remembered_answers)

    def _door(self, answer: str | None = None) -> tuple[Any, Any]:
        from tests.test_vlm_holder import _holder, _Loads

        loads = _Loads(answer=answer or self.ANSWER)
        holder = _holder(loads)
        self.addCleanup(holder.forget)
        return loads, holder

    def _read(self, holder: Any, sentence: str | None = None, poses: dict[str, str] | None = None) -> CommandReading:
        from tests.test_vlm_holder import _models

        return read_command(sentence or self.SENTENCE, models=_models(backend="vlm"), poses=poses or POSES,
                            weights_present=True, holder=holder)

    def test_the_same_question_asks_the_model_once(self) -> None:
        loads, holder = self._door()
        first = self._read(holder)
        again = self._read(holder)
        self.assertEqual(1, len(loads.model.calls))
        self.assertEqual((False, True), (first.remembered, again.remembered))
        self.assertEqual(_card_of(first), _card_of(again))
        self.assertEqual((first.attempts, first.raw), (again.attempts, again.raw))

    def test_another_pose_list_is_another_question(self) -> None:
        loads, holder = self._door()
        self._read(holder)
        self._read(holder, poses={**POSES, "Ablage rechts": "ablage_rechts"})
        self.assertEqual(2, len(loads.model.calls))

    def test_another_instruction_is_another_question(self) -> None:
        from unittest import mock

        loads, holder = self._door()
        self._read(holder)
        with mock.patch("src.models.vlm.command.COMMAND_INSTRUCTION", COMMAND_INSTRUCTION + "\nOne more rule."):
            self._read(holder)
        self.assertEqual(2, len(loads.model.calls))

    def test_other_weights_are_asked_again(self) -> None:
        from src.models.vlm import VlmWeights
        from src.models.vlm.command import _Remembering

        ask = _Canned(_answer(intent="none"), _answer(intent="none"))
        ask.grounder = _LoadedCopy()  # type: ignore[attr-defined]
        for model_id in ("Qwen/Qwen3-VL-4B-Instruct", "Qwen/Qwen3-VL-8B-Instruct"):
            _Remembering(ask, weights=VlmWeights(model_id=model_id))("system", "user")
        self.assertEqual(2, len(ask.calls))

    def test_a_replay_keeps_the_notes_of_the_reading(self) -> None:
        loads, holder = self._door('"intent": "task", "object": "screw", "object_said": "Muttern", "place": null, '
                                   '"place_said": null, "place_pose": "ablage_links", "scope": "once", "count": 3, '
                                   '"return_to": null}')
        first = self._read(holder)
        again = self._read(holder)
        self.assertEqual(("object_not_in_sentence", "count_not_supported"), first.notes)
        self.assertEqual((first.notes, True, False), (again.notes, again.remembered, again.startable))

    def test_a_reading_not_understood_replays_as_not_understood(self) -> None:
        loads, holder = self._door("no json at all")
        first = self._read(holder)
        again = self._read(holder)
        self.assertEqual((False, False, 2), (first.understood, again.understood, again.attempts))
        self.assertEqual(2, len(loads.model.calls))
        self.assertTrue(again.remembered)

    def test_a_copy_that_was_unloaded_is_asked_and_loaded_again(self) -> None:
        loads, holder = self._door()
        self._read(holder)
        holder.asker_for(_vlm_block(), may_load=True).grounder.unload()
        again = self._read(holder)
        self.assertEqual((2, 2), (loads.calls, len(loads.model.calls)))
        self.assertEqual((True, False), (again.loaded_now, again.remembered))

    def test_an_answer_that_raised_is_not_remembered(self) -> None:
        from src.models.vlm import CommandRefused

        loads, holder = self._door()
        self._read(holder)  # loads the copy
        loads.model.fail = RuntimeError("CUDA out of memory")
        with self.assertRaises(CommandRefused):
            self._read(holder, "Nimm alle Schrauben")
        loads.model.fail = None
        reading = self._read(holder, "Nimm alle Schrauben")
        self.assertFalse(reading.remembered)

    def test_forgetting_asks_the_model_again(self) -> None:
        loads, holder = self._door()
        self._read(holder)
        forget_remembered_answers()
        self.assertFalse(self._read(holder).remembered)
        self.assertEqual(2, len(loads.model.calls))

    def test_at_most_the_newest_answers_are_kept_per_copy(self) -> None:
        from src.models.vlm import VlmWeights
        from src.models.vlm.command import _Remembering

        ask = _Canned(*[_answer(intent="none")] * (REMEMBERED_ANSWERS + 2))
        ask.grounder = _LoadedCopy()  # type: ignore[attr-defined]
        weights = VlmWeights(model_id="fake/qwen")
        for index in range(REMEMBERED_ANSWERS + 1):
            _Remembering(ask, weights=weights)("system", f"user {index}")
        _Remembering(ask, weights=weights)("system", f"user {REMEMBERED_ANSWERS}")
        self.assertEqual(REMEMBERED_ANSWERS + 1, len(ask.calls), "the newest answer was asked again")
        _Remembering(ask, weights=weights)("system", "user 0")
        self.assertEqual(REMEMBERED_ANSWERS + 2, len(ask.calls), "the oldest answer was kept")


class _LoadedCopy:
    """A copy of the weights that is loaded, as the asker hands it over: memory is kept per copy."""

    loaded = True


def _vlm_block() -> Any:
    from tests.test_vlm_holder import _vlm

    return _vlm()


class StartableTests(_ForgetsTheSharedHolder):
    """R3 (2026-10-08): Enter starts only a clean reading; every other opens the card."""

    def _read(self, sentence: str, *answers: str) -> CommandReading:
        return understand(sentence, ask=_Canned(*answers), poses=POSES)

    def test_a_clean_reading_may_start(self) -> None:
        reading = self._read("Leg den grünen Würfel in die blaue Kiste",
                             _answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                                     place_said="in die blaue Kiste"))
        self.assertTrue(reading.startable)
        self.assertTrue(reading.to_dict()["startable"])

    def test_a_clean_reading_with_a_pose_may_start(self) -> None:
        reading = self._read("Leg den Becher auf Ablage links",
                             _answer(object="cup", object_said="den Becher", place_pose="ablage_links"))
        self.assertTrue(reading.startable)

    def test_each_note_opens_the_card(self) -> None:
        for sentence, answers in (
            ("Leg das Teil in die blaue Kiste", (_answer(object="green cube", object_said="den grünen Würfel",
                                                         place="blue bin", place_said="in die blaue Kiste"),)),
            ("Leg den grünen Würfel weg", (_answer(object="green cube", object_said="den grünen Würfel",
                                                   place="blue bin", place_said="in die blaue Kiste"),)),
            ("Leg den Becher auf Ablage rechts", (_answer(object="cup", object_said="den Becher",
                                                          place_pose="ablage_rechts"),)),
            ("Nimm drei Würfel", (_answer(object="cube", object_said="Würfel", count=3),)),
            ("Leg den grünen Würfel in die blaue Kiste",
             ("kein JSON", _answer(object="green cube", object_said="den grünen Würfel", place="blue bin",
                                   place_said="in die blaue Kiste"))),
        ):
            with self.subTest(sentence=sentence):
                reading = self._read(sentence, *answers)
                self.assertTrue(reading.understood)
                self.assertNotEqual((), reading.notes)
                self.assertFalse(reading.startable)

    def test_an_object_without_its_words_opens_the_card(self) -> None:
        reading = self._read("Leg den grünen Würfel in die blaue Kiste",
                             _answer(object="green cube", place="blue bin", place_said="in die blaue Kiste"))
        self.assertFalse(reading.startable)

    def test_no_part_named_opens_the_card(self) -> None:
        self.assertFalse(self._read("Räum die Kiste aus", _answer(scope="until_empty")).startable)

    def test_what_is_no_task_never_starts(self) -> None:
        for sentence, answers in (("Stopp!", (_answer(intent="stop"),)),
                                  ("Hallo Willy, wie geht's?", (_answer(intent="none"),)),
                                  ("Hallo", ("x", "y"))):
            with self.subTest(sentence=sentence):
                self.assertFalse(self._read(sentence, *answers).startable)

    def test_nothing_but_a_greeting_read_as_a_task_never_starts(self) -> None:
        reading = self._read("Hallo Willy!", _answer(object="cube", object_said="Willy"))
        self.assertTrue(reading.greeting)
        self.assertFalse(reading.startable)

    def test_a_part_singled_out_in_the_sentences_words_may_start(self) -> None:
        reading = self._read("Nimm den oberen grauen Würfel und leg ihn in die gelbe Kiste",
                             _sparse(object="gray cube", which="the upper gray cube",
                                     object_said="den oberen grauen Würfel", place="yellow bin",
                                     place_said="in die gelbe Kiste"))
        self.assertTrue(reading.startable)

    def test_a_which_or_a_from_the_sentence_does_not_hold_opens_the_card(self) -> None:
        for fields in (dict(which="the upper gray cube"), dict(source="on the black mat")):
            with self.subTest(**fields):
                reading = self._read("Nimm den grauen Würfel und leg ihn in die gelbe Kiste",
                                     _sparse(object="gray cube", object_said="den oberen grauen Würfel",
                                             place="yellow bin", place_said="in die gelbe Kiste", **fields))
                self.assertEqual(("object_not_in_sentence",), reading.notes)
                self.assertFalse(reading.startable)


#: The owner's long sentence of 2026-10-08 (348 characters): where the parts lie, the one part, the bin, and the pose
#: to go to after, in one breath.
_LONG_SENTENCE = (
    "Willy, sei bitte so gut und nimm von den Teilen, die da vorne auf der schwarzen Schaumstoffmatte liegen, den "
    "kleinen grauen Würfel, der oben auf dem größeren grauen Würfel liegt, und leg ihn vorsichtig in die gelbe Kiste "
    "rechts daneben, und wenn du fertig bist, fahr bitte wieder in die Wartepose zurück, damit ich die nächsten Teile "
    "hinlegen kann."
)


class TheOwnersDayTests(_ForgetsTheSharedHolder):
    """The sentences of 2026-10-08 on the cell, each with the answer the instruction asks for now: compact, only what
    the sentence says, the one part it singles out in ``which`` and where the parts lie in ``from``. Size stays in the
    object, as it worked; a selection is a which, and reads once."""

    def _read(self, sentence: str, *answers: str) -> tuple[CommandReading, _Canned]:
        ask = _Canned(*answers)
        return understand(sentence, ask=ask, poses=POSES), ask

    def test_the_mornings_sentence_is_the_gray_cubes_into_the_yellow_bin(self) -> None:
        reading, ask = self._read("Alle grauen Würfel in die Gelbe Kiste.",
                                  _sparse(object="gray cube", object_said="grauen Würfel", place="yellow bin",
                                          place_said="in die Gelbe Kiste", scope="until_empty"))
        self.assertEqual(("gray cube", "yellow bin", None, "until_empty", None, None, ()), _card_of(reading))
        self.assertEqual((1, None, None, True), (len(ask.calls), reading.which, reading.source, reading.startable))

    def test_where_the_parts_lie_is_the_from_and_never_the_object(self) -> None:
        sentence = "Alle grauen Würfel von der schwarzen Matte in die gelbe Kiste"
        reading, ask = self._read(sentence, _sparse(
            object="gray cube", source="on the black mat", object_said="grauen Würfel von der schwarzen Matte",
            place="yellow bin", place_said="in die gelbe Kiste", scope="until_empty"))
        self.assertEqual(("gray cube", "yellow bin", None, "until_empty", None, None, ()), _card_of(reading))
        self.assertEqual(("on the black mat", None, 1), (reading.source, reading.which, len(ask.calls)))
        self.assertNotIn("mat", reading.object.phrase if reading.object else "mat")
        self.assertTrue(reading.startable)

    def test_an_object_that_holds_the_mat_is_asked_again(self) -> None:
        sentence = "Alle grauen Würfel von der schwarzen Matte in die gelbe Kiste"
        good = _sparse(object="gray cube", source="on the black mat", object_said="grauen Würfel von der schwarzen Matte",
                       place="yellow bin", place_said="in die gelbe Kiste", scope="until_empty")
        reading, ask = self._read(sentence, _sparse(
            object="gray cube on black mat", source="on the black mat",
            object_said="grauen Würfel von der schwarzen Matte", place="yellow bin", place_said="in die gelbe Kiste",
            scope="until_empty"), good)
        self.assertEqual(2, len(ask.calls))
        self.assertIn("holds where the parts lie", ask.calls[1][1])
        self.assertEqual(("gray cube", ("retried",)), (reading.object.phrase if reading.object else None, reading.notes))

    def test_the_cube_on_top_of_the_other_is_singled_out(self) -> None:
        sentence = "Nimm den grauen Würfel, der oben auf dem anderen liegt, und leg ihn in die gelbe Kiste"
        reading, ask = self._read(sentence, _sparse(
            object="gray cube", which="the gray cube on top of the other one",
            object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
            place_said="in die gelbe Kiste"))
        self.assertEqual((1, "the gray cube on top of the other one"), (len(ask.calls), reading.which))
        self.assertEqual(PhraseReading("gray cube", "den grauen Würfel, der oben auf dem anderen liegt", True),
                         reading.object)
        self.assertEqual(("once", (), True), (reading.scope, reading.notes, reading.startable))

    def test_the_upper_and_the_smallest_are_each_one_part(self) -> None:
        for sentence, fields, which in (
            ("Nimm den oberen grauen Würfel",
             dict(object="gray cube", which="the upper gray cube", object_said="den oberen grauen Würfel"),
             "the upper gray cube"),
            ("Nimm den kleinsten Würfel",
             dict(object="cube", which="the smallest cube", object_said="den kleinsten Würfel"), "the smallest cube"),
        ):
            with self.subTest(sentence=sentence):
                reading, ask = self._read(sentence, _sparse(**fields))
                self.assertEqual((which, "once", (), 1), (reading.which, reading.scope, reading.notes, len(ask.calls)))

    def test_a_size_stays_in_the_object(self) -> None:
        reading, ask = self._read("Nimm den großen grauen Würfel",
                                  _sparse(object="large gray cube", object_said="den großen grauen Würfel"))
        self.assertEqual(("large gray cube", None, 1), (reading.object.phrase if reading.object else None,
                                                        reading.which, len(ask.calls)))

    def test_links_beside_the_pose_ablage_links(self) -> None:
        reading, ask = self._read("Nimm den Becher links von der Kiste und leg ihn auf Ablage links", _sparse(
            object="cup", which="the cup left of the bin", object_said="den Becher links von der Kiste",
            place_pose="ablage_links"))
        self.assertEqual(("the cup left of the bin", "ablage_links", None, (), 1),
                         (reading.which, reading.place_pose, reading.place, reading.notes, len(ask.calls)))
        self.assertTrue(reading.startable)

    def test_the_cola_bottle_is_english_and_a_copied_colaflasche_is_asked_again(self) -> None:
        sentence = "Nimm die Colaflasche und leg sie in die gelbe Kiste"
        good = _sparse(object="cola bottle", object_said="die Colaflasche", place="yellow bin",
                       place_said="in die gelbe Kiste")
        reading, ask = self._read(sentence, good)
        self.assertEqual(("cola bottle", 1), (reading.object.phrase if reading.object else None, len(ask.calls)))
        reading, ask = self._read(sentence, _sparse(object="colaflasche", object_said="die Colaflasche",
                                                    place="yellow bin", place_said="in die gelbe Kiste"), good)
        self.assertEqual(2, len(ask.calls))
        self.assertIn("German words", ask.calls[1][1])
        self.assertEqual(("cola bottle", ("retried",)), (reading.object.phrase if reading.object else None,
                                                         reading.notes))

    def test_the_long_sentence_reads_in_one_question(self) -> None:
        self.assertEqual(348, len(_LONG_SENTENCE))
        reading, ask = self._read(_LONG_SENTENCE, _sparse(
            object="small gray cube", source="on the black foam mat",
            which="the small gray cube on top of the larger gray cube",
            object_said="den kleinen grauen Würfel, der oben auf dem größeren grauen Würfel liegt",
            place="yellow bin", place_said="in die gelbe Kiste", return_to="wartepose"))
        self.assertEqual(1, len(ask.calls))
        self.assertEqual(("small gray cube", "yellow bin", None, "once", "wartepose", None, ()), _card_of(reading))
        self.assertEqual(("the small gray cube on top of the larger gray cube", "on the black foam mat"),
                         (reading.which, reading.source))
        assert reading.object is not None and reading.object.said is not None
        self.assertTrue(reading.object.verified)
        self.assertLessEqual(len(reading.object.said.split()), 15)


class SparseAnswersTests(_ForgetsTheSharedHolder):
    """R1 (2026-10-08): ``intent`` and only the keys the sentence fills, compact. A key left out is "not said", never
    "all"; ``intent`` left out, or a key that is none of the instruction's, is asked again; a which and a from are
    checked as the object is."""

    SENTENCE = "Nimm den grauen Würfel, der oben auf dem anderen liegt, und leg ihn in die gelbe Kiste"
    GOOD = _sparse(object="gray cube", which="the gray cube on top of the other one",
                   object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
                   place_said="in die gelbe Kiste")

    def _read(self, sentence: str, *answers: str) -> tuple[CommandReading, _Canned]:
        ask = _Canned(*answers)
        return understand(sentence, ask=ask, poses=POSES), ask

    def _retried(self, first: str, *, problem_says: str) -> CommandReading:
        reading, ask = self._read(self.SENTENCE, first, self.GOOD)
        self.assertEqual(2, len(ask.calls))
        self.assertIn(problem_says, ask.calls[1][1])
        self.assertEqual(("the gray cube on top of the other one", ("retried",)), (reading.which, reading.notes))
        return reading

    def test_intent_alone_reads_in_one_question(self) -> None:
        for sentence, answer, intent in (("Hallo Willy, wie geht's?", '{"intent":"none"}', "none"),
                                         ("Stopp!", '{"intent":"stop"}', "stop")):
            with self.subTest(sentence=sentence):
                reading, ask = self._read(sentence, answer)
                self.assertEqual((True, intent, 1), (reading.understood, reading.intent, len(ask.calls)))

    def test_a_scope_or_a_count_left_out_is_not_said(self) -> None:
        reading, ask = self._read("Leg den grünen Würfel in die blaue Kiste", _sparse(
            object="green cube", object_said="den grünen Würfel", place="blue bin", place_said="in die blaue Kiste"))
        self.assertEqual(("once", None, None, None, (), 1),
                         (reading.scope, reading.count, reading.which, reading.source, reading.notes, len(ask.calls)))

    def test_no_intent_is_asked_again(self) -> None:
        self._retried("{}", problem_says="'intent' is missing")

    def test_a_key_that_is_none_of_the_instructions_is_asked_again(self) -> None:
        self._retried(self.GOOD[:-1] + ',"colour":"gray"}', problem_says="'colour'")

    def test_nulls_and_spaces_read_as_the_compact_answer_does(self) -> None:
        compact, _ = self._read(self.SENTENCE, self.GOOD)
        spaced, _ = self._read(self.SENTENCE, json.dumps(json.loads(self.GOOD), ensure_ascii=False))
        full = {key: None for key in ANSWER_KEYS} | {"object": "", "scope": "once"} | json.loads(self.GOOD)
        written, ask = self._read(self.SENTENCE, json.dumps(full, ensure_ascii=False))
        self.assertEqual(1, len(ask.calls))
        for reading in (spaced, written):
            self.assertEqual((_card_of(compact), compact.which, compact.source),
                             (_card_of(reading), reading.which, reading.source))

    def test_a_german_which_is_asked_again(self) -> None:
        self._retried(_sparse(object="gray cube", which="der obere graue würfel",
                              object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
                              place_said="in die gelbe Kiste"), problem_says="not English")

    def test_a_which_with_both_in_it_is_asked_again(self) -> None:
        self._retried(_sparse(object="gray cube", which="the upper one of both gray cubes",
                              object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
                              place_said="in die gelbe Kiste"), problem_says="quantifier")

    def test_a_which_that_does_not_name_the_object_is_asked_again(self) -> None:
        self._retried(_sparse(object="gray cube", which="the one on top",
                              object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
                              place_said="in die gelbe Kiste"), problem_says="does not name the object")

    def test_an_overlong_which_is_asked_again(self) -> None:
        self._retried(_sparse(object="gray cube", which="the gray cube " + "right on top " * 10,
                              object_said="den grauen Würfel, der oben auf dem anderen liegt", place="yellow bin",
                              place_said="in die gelbe Kiste"), problem_says=f"longer than {MAX_WHICH_CHARS}")

    def test_a_which_may_name_the_object_in_the_plural(self) -> None:
        reading, ask = self._read("Nimm den kleinsten der Würfel", _sparse(
            object="cube", which="the smallest of the cubes", object_said="den kleinsten der Würfel"))
        self.assertEqual(("the smallest of the cubes", 1), (reading.which, len(ask.calls)))

    def test_a_german_from_is_asked_again(self) -> None:
        sentence = "Alle grauen Würfel von der schwarzen Matte in die gelbe Kiste"
        good = _sparse(object="gray cube", source="on the black mat", object_said="grauen Würfel von der schwarzen Matte",
                       place="yellow bin", place_said="in die gelbe Kiste", scope="until_empty")
        reading, ask = self._read(sentence, _sparse(
            object="gray cube", source="von der schwarzen matte", object_said="grauen Würfel von der schwarzen Matte",
            place="yellow bin", place_said="in die gelbe Kiste", scope="until_empty"), good)
        self.assertEqual(2, len(ask.calls))
        self.assertIn("the from phrase 'von der schwarzen matte' is not English", ask.calls[1][1])
        self.assertEqual("on the black mat", reading.source)

    def test_a_which_reads_once_whatever_scope_the_model_gave(self) -> None:
        reading, _ = self._read(self.SENTENCE, self.GOOD[:-1] + ',"scope":"until_empty"}')
        self.assertEqual(("once", ()), (reading.scope, reading.notes))

    def test_a_which_and_a_from_reach_the_card_and_the_report(self) -> None:
        from api.schemas import CommandOut

        reading, _ = self._read("Nimm von der Matte den grauen Würfel, der oben auf dem anderen liegt", _sparse(
            object="gray cube", source="on the mat", which="the gray cube on top of the other one",
            object_said="den grauen Würfel, der oben auf dem anderen liegt"))
        card = CommandOut.model_validate(reading.to_dict())
        self.assertEqual(("the gray cube on top of the other one", "on the mat"), (card.which, card.source))
        text = reading.render()
        text.encode("ascii")
        self.assertIn("which   : the gray cube on top of the other one", text)
        self.assertIn("from    : on the mat", text)


class TheQuestionTests(_ForgetsTheSharedHolder):
    """What the model is asked: one fixed instruction, and the poses and the command in the user message."""

    def test_the_system_message_is_the_instruction_verbatim(self) -> None:
        ask = _Canned(_answer(intent="none"))
        understand("Hallo", ask=ask, poses=POSES)
        self.assertEqual(ask.calls[0][0], COMMAND_INSTRUCTION)

    def test_the_user_message_offers_each_label_with_its_name_and_then_the_command(self) -> None:
        ask = _Canned(_answer(intent="none"))
        understand("  Nimm   alle Schrauben\n", ask=ask, poses=POSES)
        user = ask.calls[0][1]
        self.assertEqual(
            user,
            'POSES: "Ablage links" -> "ablage_links", "Wartepose" -> "wartepose", "Home" -> "home"\n'
            "COMMAND: Nimm alle Schrauben",
        )

    def test_no_taught_pose_still_offers_home(self) -> None:
        ask = _Canned(_answer(intent="none"))
        understand("Hallo", ask=ask, poses={})
        self.assertIn('POSES: "Home" -> "home"', ask.calls[0][1])

    def test_a_label_is_quoted_so_it_cannot_change_the_question(self) -> None:
        ask = _Canned(_answer(intent="none"))
        understand("Hallo", ask=ask, poses={'Ablage "A"': "ablage_a"})
        self.assertIn('"Ablage \\"A\\"" -> "ablage_a"', ask.calls[0][1])

    def test_the_instruction_maps_labels_to_names_and_never_offers_a_label_as_a_name(self) -> None:
        """The draft listed "Ablage links" as a pose name, which the name rule refuses; the fix is pinned."""
        for label, name in EXAMPLE_POSES:
            with self.subTest(label=label):
                self.assertTrue(name.isidentifier(), f"{name!r} is not a pose name")
                self.assertNotEqual(label, name)
                self.assertIn(f'"{label}" -> "{name}"', COMMAND_INSTRUCTION)
        for written in ('"place_pose": "Ablage links"', '"place_pose":"Ablage links"'):
            self.assertNotIn(written, COMMAND_INSTRUCTION)

    def test_every_example_in_the_instruction_is_an_answer_the_reader_accepts(self) -> None:
        """The examples teach the model the answer format; one the reader would refuse teaches a retry."""
        example_poses = dict(EXAMPLE_POSES)
        for sentence, answer in COMMAND_EXAMPLES:
            with self.subTest(sentence=sentence):
                CommandAnswer.model_validate(answer)
                line = f"{sentence} -> {_json(answer)}"
                self.assertIn(line, COMMAND_INSTRUCTION)
                reading = understand(sentence, ask=_Canned(_json(answer)), poses=example_poses)
                self.assertTrue(reading.understood, reading.reason)
                self.assertEqual(reading.attempts, 1)
                self.assertEqual(set(reading.notes) - {"count_not_supported"}, set(), reading.notes)

    def test_every_example_writes_only_what_its_sentence_says_compactly(self) -> None:
        """The answer is the cost on the cell (about 140 ms a token, 2026-10-08): no null, no empty phrase, no "once",
        no space after a colon or a comma, and the keys in the instruction's order."""
        for sentence, answer in COMMAND_EXAMPLES:
            with self.subTest(sentence=sentence):
                self.assertNotIn(None, answer.values())
                self.assertNotIn("", answer.values())
                self.assertNotEqual("once", answer.get("scope"))
                self.assertEqual([key for key in ANSWER_KEYS if key in answer], list(answer))
                self.assertNotIn(': "', _json(answer))
                self.assertNotIn('", "', _json(answer))

    def test_the_owners_sentences_cover_both_scopes_a_place_and_a_pose(self) -> None:
        answers = [answer for _, answer in COMMAND_EXAMPLES]
        self.assertTrue(any(a.get("scope") == "until_empty" for a in answers))
        self.assertTrue(any(not a.get("scope") and a["intent"] == "task" for a in answers))
        self.assertTrue(any(a.get("place") for a in answers))
        self.assertTrue(any(a.get("place_pose") for a in answers))
        self.assertTrue(any(a.get("return_to") for a in answers))
        self.assertTrue(any(a["intent"] == "stop" for a in answers))
        self.assertTrue(any(a["intent"] == "none" for a in answers))

    def test_the_examples_teach_a_part_singled_out_and_where_the_parts_lie(self) -> None:
        """The owner's sentences of 2026-10-08: a part on top of another, a region, and "links" beside "Ablage
        links"."""
        answers = [answer for _, answer in COMMAND_EXAMPLES]
        self.assertGreaterEqual(sum(1 for a in answers if a.get("which")), 2)
        self.assertGreaterEqual(sum(1 for a in answers if a.get("from")), 2)
        self.assertTrue(any(a.get("which") and a.get("place_pose") == "ablage_links" for a in answers))
        for answer in answers:
            if answer.get("which"):
                with self.subTest(which=answer["which"]):
                    self.assertIn(answer["object"].split()[-1], answer["which"].split())


class AnswerExtractionTests(unittest.TestCase):
    """Object-first: the first JSON object in the answer, whatever surrounds it."""

    def test_a_bare_object(self) -> None:
        self.assertEqual(extract_command_object('{"intent": "none"}'), {"intent": "none"})

    def test_a_fenced_object(self) -> None:
        self.assertEqual(extract_command_object('```json\n{"intent": "none"}\n```'), {"intent": "none"})

    def test_prose_around_the_object(self) -> None:
        self.assertEqual(extract_command_object('Here: {"intent": "stop"} Hope it helps!'), {"intent": "stop"})

    def test_the_first_of_two_objects(self) -> None:
        self.assertEqual(extract_command_object('{"intent": "stop"}\n{"intent": "none"}'), {"intent": "stop"})

    def test_a_doubled_prefill_brace(self) -> None:
        """The answer is the prefill "{" plus what the model wrote; a model that opens again doubles it."""
        self.assertEqual(extract_command_object('{{"intent": "none"}'), {"intent": "none"})

    def test_an_object_inside_an_array_is_found_but_a_bare_array_is_not_a_command(self) -> None:
        self.assertEqual(extract_command_object('[{"intent": "none"}]'), {"intent": "none"})
        self.assertIsNone(extract_command_object("[1, 2, 3]"))

    def test_nothing_usable_is_none_and_never_raises(self) -> None:
        for text in ("", "   ", "no json here", '{"intent": ', "{oops}", None):
            with self.subTest(text=text):
                self.assertIsNone(extract_command_object(text))  # type: ignore[arg-type]


class TheReadingTests(_ForgetsTheSharedHolder):
    """The reading is a report: ASCII for a person, plain data for the console's card."""

    def _reading(self) -> CommandReading:
        return understand(
            "Leg den grünen Würfel in die blaue Kiste",
            ask=_Canned("kein JSON", _answer(object="green cube", object_said="den grünen Würfel",
                                             place="blue bin", place_said="in die blaue Kiste")),
            poses=POSES,
        )

    def test_render_is_ascii_without_a_trailing_newline(self) -> None:
        text = self._reading().render()
        text.encode("ascii")
        self.assertFalse(text.endswith("\n"))
        self.assertIn("green cube", text)
        self.assertIn("gruenen Wuerfel", text, "the operator's words, folded to ASCII")

    def test_printing_shows_the_render(self) -> None:
        reading = self._reading()
        self.assertEqual(str(reading), reading.render())

    def test_a_reading_that_failed_renders_its_reason(self) -> None:
        reading = understand("Hallo", ask=_Canned("x", "y"), poses=POSES)
        text = reading.render()
        text.encode("ascii")
        self.assertIn("not read", text)
        self.assertIn("no JSON object", text)

    def test_to_dict_is_plain_json(self) -> None:
        payload = self._reading().to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["notes"], ["retried"])
        self.assertEqual(payload["model"]["attempts"], 2)
        self.assertEqual(payload["object"], {"phrase": "green cube", "said": "den grünen Würfel", "verified": True})

    def test_to_dict_is_the_consoles_command_card(self) -> None:
        """The console builds `CommandOut` from this dict (adding route previews): the shapes must not drift."""
        from api.schemas import CommandOut

        for reading in (self._reading(), understand("Hallo", ask=_Canned("x", "y"), poses=POSES),
                        understand("Stopp", ask=_Canned(_answer(intent="stop")), poses=POSES)):
            with self.subTest(understood=reading.understood, intent=reading.intent):
                card = CommandOut.model_validate(reading.to_dict())
                self.assertEqual(card.understood, reading.understood)
                self.assertEqual(card.notes, list(reading.notes))


class ReadingMovesNothingTests(_ForgetsTheSharedHolder):
    """A sentence reaches the model and nothing else."""

    def test_an_empty_sentence_is_refused_before_anything_is_asked(self) -> None:
        ask = _Canned()
        for text in ("", "   ", "\n\t"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                understand(text, ask=ask, poses=POSES)
        self.assertEqual(ask.calls, [])

    def test_an_overlong_sentence_is_refused_before_anything_is_asked(self) -> None:
        ask = _Canned()
        with self.assertRaises(ValueError):
            understand("x" * (MAX_SENTENCE_CHARS + 1), ask=ask, poses=POSES)
        self.assertEqual(ask.calls, [])

    def test_a_model_that_raises_is_not_a_reading(self) -> None:
        """"I could not ask" is not "I did not understand": the caller gets the exception."""
        def broken(system: str, user: str) -> str:
            raise RuntimeError("CUDA out of memory")

        with self.assertRaises(RuntimeError):
            understand("Leg den Würfel in die Kiste", ask=broken, poses=POSES)

    def test_the_reader_imports_no_robot_camera_or_console_module(self) -> None:
        """Checked in the source: an import is how a reader could reach the cell, and none is there."""
        for module in ("command.py", "known.py", "holder.py", "availability.py", "qwen.py"):
            with self.subTest(module=module):
                tree = ast.parse((_ROOT / "src" / "models" / "vlm" / module).read_text(encoding="utf-8"))
                imported = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.update(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        imported.add(node.module)
                reaching = sorted(name for name in imported
                                  if name.split(".")[0] in {"api", "willy"}
                                  or name.startswith(("src.robot", "src.camera", "src.calibration")))
                self.assertEqual(reaching, [], f"{module} imports {reaching}")

    def test_importing_the_reader_pulls_in_no_torch(self) -> None:
        """Measured in a fresh interpreter: this one has already imported torch elsewhere."""
        probe = (
            "import sys;"
            "import src.models.vlm.command, src.models.vlm.known, src.models.vlm.holder;"
            "heavy=[m for m in ('torch','transformers') if m in sys.modules];"
            "print(','.join(heavy) or 'clean')"
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False,
                              timeout=180, cwd=str(_ROOT))
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertEqual(done.stdout.strip(), "clean")


if __name__ == "__main__":
    unittest.main()
