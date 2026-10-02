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
    COMMAND_EXAMPLES,
    EXAMPLE_POSES,
    MAX_PHRASE_CHARS,
    MAX_SENTENCE_CHARS,
    CommandAnswer,
    extract_command_object,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: The owner's taught poses as the console hands them over: spoken label -> name.
POSES = {"Ablage links": "ablage_links", "Wartepose": "wartepose"}


def _answer(**fields: Any) -> str:
    """One model answer with every key, as the instruction asks; ``fields`` overrides the defaults."""
    base: dict[str, Any] = {
        "intent": "task", "object": "", "object_said": None, "place": None, "place_said": None,
        "place_pose": None, "scope": "once", "count": None, "return_to": None,
    }
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)


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

    def test_a_missing_key_is_asked_again(self) -> None:
        incomplete = json.loads(self.GOOD)
        del incomplete["count"]
        self._retried(json.dumps(incomplete), problem_says="'count' is missing")

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
        self.assertNotIn('"place_pose": "Ablage links"', COMMAND_INSTRUCTION)

    def test_every_example_in_the_instruction_is_an_answer_the_reader_accepts(self) -> None:
        """The examples teach the model the answer format; one the reader would refuse teaches a retry."""
        example_poses = dict(EXAMPLE_POSES)
        for sentence, answer in COMMAND_EXAMPLES:
            with self.subTest(sentence=sentence):
                CommandAnswer.model_validate(answer)
                line = f"{sentence} -> {json.dumps(answer, ensure_ascii=False)}"
                self.assertIn(line, COMMAND_INSTRUCTION)
                reading = understand(sentence, ask=_Canned(json.dumps(answer, ensure_ascii=False)),
                                     poses=example_poses)
                self.assertTrue(reading.understood, reading.reason)
                self.assertEqual(reading.attempts, 1)
                self.assertEqual(set(reading.notes) - {"count_not_supported"}, set(), reading.notes)

    def test_the_owners_sentences_cover_both_scopes_a_place_and_a_pose(self) -> None:
        answers = [answer for _, answer in COMMAND_EXAMPLES]
        self.assertTrue(any(a["scope"] == "until_empty" for a in answers))
        self.assertTrue(any(a["place"] for a in answers))
        self.assertTrue(any(a["place_pose"] for a in answers))
        self.assertTrue(any(a["return_to"] for a in answers))
        self.assertTrue(any(a["intent"] == "stop" for a in answers))
        self.assertTrue(any(a["intent"] == "none" for a in answers))


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
        for module in ("command.py", "holder.py", "availability.py", "qwen.py"):
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
            "import src.models.vlm.command, src.models.vlm.holder;"
            "heavy=[m for m in ('torch','transformers') if m in sys.modules];"
            "print(','.join(heavy) or 'clean')"
        )
        done = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False,
                              timeout=180, cwd=str(_ROOT))
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        self.assertEqual(done.stdout.strip(), "clean")


if __name__ == "__main__":
    unittest.main()
