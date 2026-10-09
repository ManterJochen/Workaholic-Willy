"""Sorting by rules from one sentence: the command reader reads several kinds of parts, each to its own place.

The owner, 2026-10-09: "grüne Teile in die gelbe Kiste, rote in die blaue". The model answers a sort in today's keys
for the first kind and one entry of ``also`` per further kind, with the same keys; the reader checks each further rule
as it checks the first, naming its entry, and the rules together: at most three further kinds, no kind twice, and in a
sort every kind with its place (``src/models/vlm/command.py``). No quality (i.O./N.i.O.) yet.

These tests script ``ask`` as ``tests/test_vlm_command.py`` does: no weights, no GPU. What is pinned:

1. **Two kinds into two places read as two rules**, every word found in the sentence ("rote" too), and a sentence of
   one kind reads as it did: one rule, the reading's own fields.
2. **What a sort cannot be is asked again once, naming the rule**: the same kind twice, a kind without its place, more
   than four kinds, a German phrase or a place beside a pose in ``also[0]``; a second bad answer fills nothing.
3. **Enter starts a sort only where every rule is clean**: a note on any rule opens the card.
4. **The longest sort fits** ``COMMAND_MAX_NEW_TOKENS``, and an answer cut short is never read as one of its rules.
5. **A sorting sentence is never a known sentence**: the model reads it.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

from src.models.vlm import shared_vlm, understand
from src.models.vlm.command import (
    ANSWER_KEYS,
    COMMAND_EXAMPLES,
    COMMAND_INSTRUCTION,
    COMMAND_MAX_NEW_TOKENS,
    MAX_FURTHER_RULES,
    RULE_KEYS,
    CommandAnswer,
    CommandReading,
    PhraseReading,
    RuleReading,
    _Offered,
    extract_command_object,
    forget_remembered_answers,
    read_command,
)
from src.models.vlm.known import known_answer, read_known
from tests.test_vlm_command import _REAL_ANSWERS, _answer, _Canned, _sparse
from tests.test_vlm_command import POSES as TODAYS_POSES

#: The owner's taught poses, and a second place pose a sort may name.
POSES = {"Ablage links": "ablage_links", "Ablage rechts": "ablage_rechts", "Wartepose": "wartepose"}

#: The owner's sentence of 2026-10-09, and its second rule as a well-behaved model writes it.
SORT = "grüne Teile in die gelbe Kiste, rote in die blaue"
RED: dict[str, Any] = dict(object="red part", object_said="rote", place="blue bin", place_said="in die blaue")
GREEN_RULE = RuleReading(object=PhraseReading("green part", "grüne Teile", True),
                         place=PhraseReading("yellow bin", "in die gelbe Kiste", True))
RED_RULE = RuleReading(object=PhraseReading("red part", "rote", True), place=PhraseReading("blue bin", "in die blaue", True))


def _sort(*, also: list[dict[str, Any]] | None = None, **first: Any) -> str:
    """The answer to :data:`SORT`, compact as the instruction asks: ``first`` overrides the first kind's keys (``None``
    writes null), ``also`` the further rules (``[RED]`` unless given)."""
    fields: dict[str, Any] = dict(object="green part", object_said="grüne Teile", place="yellow bin",
                                  place_said="in die gelbe Kiste", scope="until_empty")
    fields.update(first)
    return _sparse(**fields, also=[RED] if also is None else also)


def _with(answer: str, change: Any) -> str:
    """``answer`` with ``change(payload)`` applied to its JSON, written compactly again."""
    payload = json.loads(answer)
    change(payload)
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


class _Reads(unittest.TestCase):
    """Reads with a scripted model; forgets the process's one VLM holder, as the reader's own tests do."""

    def setUp(self) -> None:
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)

    def _read(self, sentence: str, *answers: str, poses: dict[str, str] | None = None) -> tuple[CommandReading, _Canned]:
        ask = _Canned(*answers)
        return understand(sentence, ask=ask, poses=POSES if poses is None else poses), ask


class TwoKindsIntoTwoPlacesTests(_Reads):
    """A sort reads as one rule per kind, each checked and found in the sentence as the first is."""

    def test_green_into_the_yellow_bin_and_red_into_the_blue_one_are_two_rules(self) -> None:
        reading, ask = self._read(SORT, _sort())
        self.assertEqual((True, "task", 1), (reading.understood, reading.intent, len(ask.calls)))
        self.assertEqual((GREEN_RULE, RED_RULE), reading.rules)
        self.assertEqual((RED_RULE,), reading.more_rules)
        self.assertEqual((GREEN_RULE.object, GREEN_RULE.place, None, None, None),
                         (reading.object, reading.place, reading.place_pose, reading.which, reading.source))
        self.assertEqual(("until_empty", (), True), (reading.scope, reading.notes, reading.startable))

    def test_rote_is_found_in_the_sentence_and_words_it_does_not_hold_are_noted_on_their_rule(self) -> None:
        reading, _ = self._read(SORT, _sort())
        self.assertEqual(PhraseReading("red part", "rote", True), reading.rules[1].object)
        reading, _ = self._read(SORT, _sort(also=[dict(RED, object_said="roten")]))
        self.assertEqual(PhraseReading("red part", "roten", False), reading.rules[1].object)
        self.assertEqual(((), ("object_not_in_sentence",), ("object_not_in_sentence",)),
                         (reading.rules[0].notes, reading.rules[1].notes, reading.notes))
        self.assertFalse(reading.startable)

    def test_a_bin_and_a_pose_in_one_sort(self) -> None:
        sentence = "die Schrauben in die Kiste links, die Muttern auf Ablage rechts"
        reading, ask = self._read(sentence, _sparse(
            object="screw", object_said="die Schrauben", place="left bin", place_said="in die Kiste links",
            scope="until_empty", also=[dict(object="nut", object_said="die Muttern", place_pose="ablage_rechts")]))
        self.assertEqual(1, len(ask.calls))
        self.assertEqual(RuleReading(object=PhraseReading("nut", "die Muttern", True), place_pose="ablage_rechts"),
                         reading.rules[1])
        self.assertEqual(("left bin", None, (), True), (reading.place.phrase if reading.place else None,
                                                        reading.place_pose, reading.notes, reading.startable))

    def test_three_kinds_in_english(self) -> None:
        sentence = "put the red cubes into the red bin, the blue ones into the blue bin and the nuts on Ablage links"
        reading, ask = self._read(sentence, _sparse(
            object="red cube", object_said="the red cubes", place="red bin", place_said="into the red bin",
            scope="until_empty",
            also=[dict(object="blue cube", object_said="the blue ones", place="blue bin", place_said="into the blue bin"),
                  dict(object="nut", object_said="the nuts", place_pose="ablage_links")]))
        self.assertEqual((1, 3, (), True), (len(ask.calls), len(reading.rules), reading.notes, reading.startable))
        self.assertEqual([("red cube", "red bin", None), ("blue cube", "blue bin", None), ("nut", None, "ablage_links")],
                         [(rule.object.phrase if rule.object else None, rule.place.phrase if rule.place else None,
                           rule.place_pose) for rule in reading.rules])

    def test_a_spoken_label_in_a_further_rule_comes_back_as_its_name(self) -> None:
        reading, _ = self._read("grüne Teile in die gelbe Kiste, rote auf Ablage rechts",
                                _sort(also=[dict(object="red part", object_said="rote", place_pose="Ablage rechts")]))
        self.assertEqual(("ablage_rechts", ()), (reading.rules[1].place_pose, reading.notes))

    def test_a_further_place_whose_own_words_are_a_pose_label_is_that_pose(self) -> None:
        """The backstop of the first rule holds for every rule: those words name the taught pose, nothing to search."""
        reading, ask = self._read("grüne Teile in die gelbe Kiste, rote auf Ablage rechts", _sort(
            also=[dict(object="red part", object_said="rote", place="right shelf", place_said="auf Ablage rechts")]))
        self.assertEqual(1, len(ask.calls))
        self.assertEqual((None, "ablage_rechts", ()), (reading.rules[1].place, reading.rules[1].place_pose,
                                                       reading.notes))

    def test_a_pose_nobody_taught_is_dropped_and_noted_on_its_rule(self) -> None:
        reading, ask = self._read("grüne Teile in die gelbe Kiste, rote auf Ablage hinten", _sort(
            also=[dict(object="red part", object_said="rote", place_pose="ablage_hinten")]))
        self.assertEqual((True, 1), (reading.understood, len(ask.calls)))
        self.assertEqual((None, None, ("pose_unknown",)),
                         (reading.rules[1].place, reading.rules[1].place_pose, reading.rules[1].notes))
        self.assertEqual((("pose_unknown",), False), (reading.notes, reading.startable))

    def test_a_further_kind_may_say_where_its_parts_lie(self) -> None:
        sentence = "grüne Teile in die gelbe Kiste, rote von der Matte in die blaue"
        reading, _ = self._read(sentence, _sort(also=[dict(RED, source="on the mat", object_said="rote von der Matte")]))
        self.assertEqual((None, "on the mat"), (reading.rules[0].source, reading.rules[1].source))
        self.assertEqual(("until_empty", (), True), (reading.scope, reading.notes, reading.startable))

    def test_a_part_singled_out_by_any_rule_reads_once(self) -> None:
        """A which reads one part (the card errs toward fewer motions), whichever rule says it."""
        for sentence, answer in (
            ("den oberen grünen Würfel in die gelbe Kiste, den roten in die blaue",
             _sparse(object="green cube", which="the upper green cube", object_said="den oberen grünen Würfel",
                     place="yellow bin", place_said="in die gelbe Kiste",
                     also=[dict(object="red cube", object_said="den roten", place="blue bin",
                                place_said="in die blaue")])),
            ("grüne Würfel in die gelbe Kiste, den oberen roten in die blaue",
             _sparse(object="green cube", object_said="grüne Würfel", place="yellow bin",
                     place_said="in die gelbe Kiste", scope="until_empty",
                     also=[dict(object="red cube", which="the upper red cube", object_said="den oberen roten",
                                place="blue bin", place_said="in die blaue")])),
        ):
            with self.subTest(sentence=sentence):
                reading, ask = self._read(sentence, answer)
                self.assertEqual((True, 1, "once", ()), (reading.understood, len(ask.calls), reading.scope,
                                                         reading.notes))

    def test_also_null_or_empty_is_one_rule(self) -> None:
        sentence = "Leg den grünen Würfel in die blaue Kiste"
        fields: dict[str, Any] = dict(object="green cube", object_said="den grünen Würfel", place="blue bin",
                                      place_said="in die blaue Kiste")
        plain, _ = self._read(sentence, _sparse(**fields))
        nothing_more: tuple[list[dict[str, Any]] | None, ...] = (None, [])
        for also in nothing_more:
            with self.subTest(also=also):
                reading, ask = self._read(sentence, _sparse(**fields, also=also))
                self.assertEqual((1, plain.rules, (), True), (len(ask.calls), reading.rules, reading.more_rules,
                                                              reading.startable))


class ASortIsAskedAgainTests(_Reads):
    """A sort the task could not run is asked again once, and the question names the rule that was wrong."""

    GOOD = _sort()

    def _retried(self, first: str, *, says: str) -> str:
        reading, ask = self._read(SORT, first, self.GOOD)
        self.assertEqual(2, len(ask.calls))
        retry = ask.calls[1][1]
        self.assertIn(says, retry)
        self.assertIn(SORT, retry, "the retry repeats the command")
        self.assertEqual((True, ("retried",), (GREEN_RULE, RED_RULE)), (reading.understood, reading.notes,
                                                                         reading.rules))
        return retry

    def test_the_same_kind_twice(self) -> None:
        self._retried(_sort(also=[dict(RED, object="green part")]),
                      says="the first kind and also[0] both name 'green part': each kind of part goes to one place")

    def test_the_same_kind_in_the_plural_or_another_case(self) -> None:
        for kind in ("green parts", "Green Part"):
            with self.subTest(kind=kind):
                self._retried(_sort(also=[dict(RED, object=kind)]), says="the first kind and also[0] both name")

    def test_the_same_kind_in_two_further_rules(self) -> None:
        self._retried(_sort(also=[RED, dict(RED, place="black bin")]), says="also[0] and also[1] both name 'red part'")

    def test_a_further_kind_without_its_place(self) -> None:
        for rule in (dict(object="red part", object_said="rote"),
                     dict(object="red part", object_said="rote", place_said="in die blaue")):
            with self.subTest(rule=rule):
                self._retried(_sort(also=[rule]), says="also[0] names no place")

    def test_the_first_kind_without_its_place_in_a_sort(self) -> None:
        self._retried(_sort(place=None, place_said=None), says="the first kind names no place")

    def test_more_than_four_kinds_are_asked_again_and_four_are_read(self) -> None:
        further = [RED, dict(object="gray part", object_said="graue", place="black bin", place_said="in die schwarze"),
                   dict(object="white part", object_said="weiße", place_pose="ablage_links"),
                   dict(object="blue part", object_said="blaue", place="red bin", place_said="in die rote")]
        five = ("grüne Teile in die gelbe Kiste, rote in die blaue, graue in die schwarze, weiße auf Ablage links, "
                "blaue in die rote")
        answer = _sort(also=further)
        reading, ask = self._read(five, answer, answer)
        self.assertEqual(2, len(ask.calls))
        self.assertIn(f"also holds 4 entries: a command sorts at most {MAX_FURTHER_RULES + 1} kinds of parts",
                      ask.calls[1][1])
        self.assertEqual((False, (), None, None), (reading.understood, reading.rules, reading.object, reading.scope))
        four = "grüne Teile in die gelbe Kiste, rote in die blaue, graue in die schwarze, weiße auf Ablage links"
        reading, ask = self._read(four, _sort(also=further[:MAX_FURTHER_RULES]))
        self.assertEqual((True, 1, MAX_FURTHER_RULES + 1, (), True),
                         (reading.understood, len(ask.calls), len(reading.rules), reading.notes, reading.startable))

    def test_a_german_phrase_in_also_0_is_asked_again_naming_also_0(self) -> None:
        retry = self._retried(_sort(also=[dict(RED, object="rote teile")]),
                              says="in also[0], the object phrase 'rote teile' is not English")
        self.assertNotIn("the first kind", retry)
        self._retried(_sort(also=[dict(RED, place="blaue kiste")]),
                      says="in also[0], the place phrase 'blaue kiste' is not English")

    def test_a_copied_german_word_in_a_further_rule_is_asked_again(self) -> None:
        sentence = "grüne Teile in die gelbe Kiste, die Colaflaschen in die blaue"
        good = _sort(also=[dict(RED, object="cola bottle", object_said="die Colaflaschen")])
        reading, ask = self._read(sentence, _sort(also=[dict(RED, object="colaflaschen",
                                                             object_said="die Colaflaschen")]), good)
        self.assertEqual(2, len(ask.calls))
        self.assertIn("in also[0], the object phrase 'colaflaschen'", ask.calls[1][1])
        self.assertEqual(("cola bottle", ("retried",)), (reading.rules[1].object.phrase if reading.rules[1].object
                                                         else None, reading.notes))

    def test_a_place_beside_a_pose_in_also_0_is_asked_again_naming_also_0(self) -> None:
        self._retried(_sort(also=[dict(RED, place_pose="ablage_links")]),
                      says="in also[0], place and place_pose are both set")

    def test_a_quantifier_in_also_0(self) -> None:
        self._retried(_sort(also=[dict(RED, object="all red parts")]),
                      says="in also[0], the object phrase 'all red parts' carries a quantifier")

    def test_a_which_in_also_0_that_does_not_name_its_object(self) -> None:
        self._retried(_sort(also=[dict(RED, which="the upper one")]),
                      says="in also[0], the which phrase 'the upper one' does not name the object 'red part'")

    def test_a_key_no_rule_has_is_named_with_its_entry(self) -> None:
        self._retried(_with(_sort(), lambda answer: answer["also"][0].update(colour="red")),
                      says="key 'also[0].colour' is not one of the keys")
        self._retried(_with(_sort(), lambda answer: answer["also"][0].update(place=7)),
                      says="key 'also[0].place'")

    def test_also_that_is_no_list(self) -> None:
        self._retried(_with(_sort(), lambda answer: answer.update(also=answer["also"][0])), says="key 'also'")

    def test_no_quality_yet(self) -> None:
        """The owner, 2026-10-09: rules by colour or kind only; an i.O./N.i.O. key is none of the keys."""
        self._retried(_with(_sort(), lambda answer: answer.update(quality="ok")),
                      says="key 'quality' is not one of the keys")
        self._retried(_with(_sort(), lambda answer: answer["also"][0].update(quality="nok")),
                      says="key 'also[0].quality' is not one of the keys")

    def test_two_bad_answers_fill_nothing(self) -> None:
        bad = _sort(also=[dict(RED, object="green part")])
        reading, ask = self._read(SORT, bad, bad)
        self.assertEqual(2, len(ask.calls))
        self.assertEqual((False, "none", (), None, None, ("retried",)),
                         (reading.understood, reading.intent, reading.rules, reading.object, reading.place,
                          reading.notes))
        self.assertIn("both name 'green part'", reading.reason)
        self.assertFalse(reading.startable)

    def test_the_retry_quotes_a_long_sort_whole(self) -> None:
        """The rule the problem names is in the quoted answer, however far back it stands."""
        further = [RED, dict(object="gray part", object_said="graue", place="black bin", place_said="in die schwarze"),
                   dict(object="white part", object_said="weiße", place="white bin", place_said="in die weiße")]
        first = json.dumps(json.loads(_sort(also=further)), ensure_ascii=False)  # spaced, as the 4B writes
        reading, ask = self._read(SORT + ", graue in die schwarze, weiße in die weiße", first)
        self.assertGreater(len(first), 300)
        self.assertEqual((1, True), (len(ask.calls), reading.understood))
        bad = _with(_sort(also=further), lambda answer: answer["also"][2].update(object="weisse teile"))
        reading, ask = self._read(SORT, bad, self.GOOD)
        self.assertIn("in also[2]", ask.calls[1][1])
        self.assertIn(f"Previous answer: {bad}\n", ask.calls[1][1])


class StartableSortTests(_Reads):
    """The owner, 2026-10-09: Enter starts at once only on a clean reading; a note on any rule opens the card."""

    def test_a_clean_sort_may_start_on_enter(self) -> None:
        reading, _ = self._read(SORT, _sort())
        self.assertEqual((True, True), (reading.startable, reading.to_dict()["startable"]))
        self.assertIn("start   : on Enter, nothing to check", reading.render())

    def test_a_note_on_any_rule_opens_the_card(self) -> None:
        for name, answer, rule, note in (
            ("the first kind's words", _sort(object_said="grünen Teile"), 0, "object_not_in_sentence"),
            ("the first place's words", _sort(place_said="in die gelbe Box"), 0, "place_not_in_sentence"),
            ("a further kind's words", _sort(also=[dict(RED, object_said="roten")]), 1, "object_not_in_sentence"),
            ("a further place's words", _sort(also=[dict(RED, place_said="in die rote")]), 1, "place_not_in_sentence"),
            ("a further pose nobody taught",
             _sort(also=[dict(object="red part", object_said="rote", place_pose="ablage_hinten")]), 1, "pose_unknown"),
            ("Home as a further place",
             _sort(also=[dict(object="red part", object_said="rote", place_pose="home")]), 1, "pose_unknown"),
        ):
            with self.subTest(name):
                reading, ask = self._read(SORT, answer)
                self.assertEqual((True, 1), (reading.understood, len(ask.calls)))
                self.assertIn(note, reading.rules[rule].notes)
                self.assertIn(note, reading.notes)
                self.assertEqual((False, False), (reading.startable, reading.to_dict()["startable"]))

    def test_a_kind_that_names_no_part_opens_the_card(self) -> None:
        """A word for any part grounds whatever the camera sees (Q7 A+): the card asks, in a sort as alone."""
        for answer, rule in ((_sort(object="part"), 0),
                             (_sort(also=[dict(RED, object="everything", object_said="alles andere")]), 1)):
            with self.subTest(rule=rule):
                reading, _ = self._read(SORT, answer)
                self.assertEqual((True, None, False), (reading.understood, reading.rules[rule].object,
                                                       reading.startable))

    def test_a_sort_built_with_a_rule_that_names_no_place_never_starts(self) -> None:
        bin_ = PhraseReading("yellow bin", "in die gelbe Kiste", True)
        green = PhraseReading("green part", "grüne Teile", True)
        red = PhraseReading("red part", "rote", True)
        for further, startable in ((RuleReading(object=red), False),
                                   (RuleReading(object=red, place_pose="ablage_links"), True)):
            with self.subTest(startable=startable):
                reading = CommandReading(understood=True, intent="task", object=green, place=bin_,
                                         scope="until_empty", rules=(RuleReading(object=green, place=bin_), further))
                self.assertEqual(startable, reading.startable)
        reading = CommandReading(understood=True, intent="task", object=green, scope="until_empty",
                                 rules=(RuleReading(object=green), RuleReading(object=red, place_pose="ablage_links")))
        self.assertFalse(reading.startable, "a sort never places its first kind at the default place pose")


class TheRulesOnTheCardTests(_Reads):
    """The reading carries every rule: as data for the console's card, and as ASCII text for a person."""

    def test_to_dict_carries_every_rule_and_rule_0_is_the_cards_own_fields(self) -> None:
        reading, _ = self._read(SORT, _sort())
        payload = reading.to_dict()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual([
            {"object": {"phrase": "green part", "said": "grüne Teile", "verified": True}, "which": None,
             "source": None, "place": {"phrase": "yellow bin", "said": "in die gelbe Kiste", "verified": True},
             "place_pose": None, "notes": []},
            {"object": {"phrase": "red part", "said": "rote", "verified": True}, "which": None, "source": None,
             "place": {"phrase": "blue bin", "said": "in die blaue", "verified": True}, "place_pose": None,
             "notes": []},
        ], payload["rules"])
        for key in ("object", "which", "source", "place", "place_pose"):
            with self.subTest(key=key):
                self.assertEqual(payload[key], payload["rules"][0][key])

    def test_the_consoles_card_still_reads_a_sort(self) -> None:
        from api.schemas import CommandOut

        reading, _ = self._read(SORT, _sort())
        card = CommandOut.model_validate(reading.to_dict())
        self.assertEqual(("green part", "yellow bin"), (card.object.phrase if card.object else None,
                                                        card.place.phrase if card.place else None))

    def test_render_lists_every_rule_in_ascii(self) -> None:
        sentence = "put the red cubes into the red bin, the blue ones into the blue bin and the nuts on Ablage links"
        reading, _ = self._read(sentence, _sparse(
            object="red cube", object_said="the red cubes", place="red bin", place_said="into the red bin",
            scope="until_empty",
            also=[dict(object="blue cube", object_said="the blue ones", place="blue bin", place_said="into the blue bin"),
                  dict(object="nut", object_said="the nuts", place_pose="ablage_links")]))
        text = reading.render()
        text.encode("ascii")
        self.assertIn('  pick    : red cube (said "the red cubes", in the sentence)', text)
        self.assertIn('  rule 2  : blue cube (said "the blue ones", in the sentence) -> the camera finds blue bin '
                      '(said "into the blue bin", in the sentence)', text)
        self.assertIn('  rule 3  : nut (said "the nuts", in the sentence) -> the taught pose ablage_links', text)
        reading, _ = self._read(SORT, _sort(also=[dict(RED, object_said="roten", source="on the mat")]))
        self.assertIn('  rule 2  : red part (said "roten", NOT in the sentence: please check), from on the mat -> '
                      'the camera finds blue bin', reading.render())

    def test_rule_0_is_the_readings_own_fields(self) -> None:
        green = PhraseReading("green cube", "den grünen Würfel", True)
        built = CommandReading(understood=True, intent="task", object=green, scope="once")
        self.assertEqual((RuleReading(object=green),), built.rules)
        self.assertEqual((), CommandReading(understood=True, intent="stop").rules)
        self.assertEqual((), CommandReading(understood=False, intent="none", reason="x").rules)
        wrongs: tuple[dict[str, Any], ...] = (
            dict(understood=True, intent="task", object=green, rules=(RuleReading(),)),
            dict(understood=True, intent="stop", rules=(RuleReading(),)),
            dict(understood=False, intent="none", rules=(RuleReading(),)),
            dict(understood=True, intent="task", rules=(RuleReading(),) * (MAX_FURTHER_RULES + 2)),
        )
        for wrong in wrongs:
            with self.subTest(wrong=wrong), self.assertRaises(ValueError):
                CommandReading(**wrong)


class OneKindReadsAsItDidTests(_Reads):
    """Every answer of one kind is today's reading, with one rule: its own fields."""

    def test_every_measured_answer_reads_one_rule_its_own_fields(self) -> None:
        for sentence, fields, _want in _REAL_ANSWERS:
            for written, answer in (("full", _answer(**fields)), ("sparse", _sparse(**fields))):
                with self.subTest(sentence=sentence, written=written):
                    reading, ask = self._read(sentence, answer, poses=TODAYS_POSES)
                    self.assertEqual(1, len(ask.calls))
                    if reading.intent != "task":
                        self.assertEqual(((), []), (reading.rules, reading.to_dict()["rules"]))
                        continue
                    (rule,) = reading.rules
                    self.assertEqual((reading.object, reading.which, reading.source, reading.place,
                                      reading.place_pose), (rule.object, rule.which, rule.source, rule.place,
                                                            rule.place_pose))
                    self.assertLessEqual(set(rule.notes), set(reading.notes))
                    self.assertEqual(((), 1), (reading.more_rules, len(reading.to_dict()["rules"])))

    def test_the_card_of_one_kind_is_todays_with_its_rule(self) -> None:
        reading, _ = self._read("Leg den grünen Würfel in die blaue Kiste", _sparse(
            object="green cube", object_said="den grünen Würfel", place="blue bin", place_said="in die blaue Kiste"))
        self.assertEqual({"understood", "intent", "object", "which", "source", "place", "place_pose", "scope", "count",
                          "return_to", "rules", "notes", "reason", "model", "raw", "startable"},
                         set(reading.to_dict()))
        self.assertNotIn("rule 2", reading.render())

    def test_one_kind_without_a_place_still_goes_to_the_default_place_pose(self) -> None:
        reading, _ = self._read("Nimm den Becher", _sparse(object="cup", object_said="den Becher"))
        self.assertEqual(((RuleReading(object=PhraseReading("cup", "den Becher", True)),), True),
                         (reading.rules, reading.startable))
        self.assertIn("  place   : the default place pose", reading.render())


class TheInstructionTeachesASortTests(unittest.TestCase):
    """The instruction lists also after the first kind's keys, teaches it by example, and asks no quality."""

    def test_also_follows_the_first_kinds_keys_and_a_rule_has_them_all(self) -> None:
        self.assertEqual(ANSWER_KEYS.index("place_pose") + 1, ANSWER_KEYS.index("also"))
        self.assertEqual(("object", "from", "which", "object_said", "place", "place_said", "place_pose"), RULE_KEYS)
        self.assertIn('"place_pose":"<a name from POSES>","also":[', COMMAND_INSTRUCTION)

    def test_the_examples_teach_a_noun_left_out_a_pose_and_three_kinds(self) -> None:
        sorts = {sentence: answer for sentence, answer in COMMAND_EXAMPLES if answer.get("also")}
        self.assertIn("grüne Teile in die gelbe Kiste, rote in die blaue", sorts)
        self.assertEqual("red part", sorts["grüne Teile in die gelbe Kiste, rote in die blaue"]["also"][0]["object"])
        self.assertTrue(any(rule.get("place_pose") for answer in sorts.values() for rule in answer["also"]))
        self.assertTrue(any(len(answer["also"]) >= 2 for answer in sorts.values()))
        self.assertTrue(all(answer.get("scope") == "until_empty" for answer in sorts.values()))

    def test_each_further_rule_of_an_example_is_written_compactly_with_its_place(self) -> None:
        for sentence, answer in COMMAND_EXAMPLES:
            for rule in answer.get("also", ()):
                with self.subTest(sentence=sentence, rule=rule.get("object")):
                    self.assertEqual([key for key in RULE_KEYS if key in rule], list(rule))
                    self.assertNotIn(None, rule.values())
                    self.assertNotIn("", rule.values())
                    self.assertTrue(rule.get("place") or rule.get("place_pose"))

    def test_the_instruction_bounds_the_rules_and_asks_no_quality(self) -> None:
        self.assertIn("- also: only when the command sends different kinds of parts to different places",
                      COMMAND_INSTRUCTION)
        self.assertIn(f"at most {MAX_FURTHER_RULES} entries", COMMAND_INSTRUCTION)
        self.assertEqual(3, MAX_FURTHER_RULES)
        self.assertNotIn("quality", COMMAND_INSTRUCTION)
        self.assertNotIn("quality", ANSWER_KEYS + RULE_KEYS)
        self.assertNotIn("quality", CommandAnswer.model_fields)


#: Four kinds of parts with the longest words the instruction allows in every rule, a from and a pose to go to after.
_LONGEST_SENTENCE = (
    "Die kleinen grünen Metallwinkel von der schwarzen Schaumstoffmatte in die große gelbe Kunststoffkiste ganz rechts, "
    "die kleinen roten Metallwinkel in die große blaue Kunststoffkiste in der Mitte, die großen grauen "
    "Sechskantmuttern in den kleinen schwarzen Karton ganz links und die langen silbernen Sechskantschrauben auf "
    "Ablage rechts, danach in die Wartepose"
)
_LONGEST: dict[str, Any] = {
    "intent": "task", "object": "small green metal bracket", "from": "on the black foam mat",
    "object_said": "Die kleinen grünen Metallwinkel von der schwarzen Schaumstoffmatte",
    "place": "large yellow plastic bin", "place_said": "in die große gelbe Kunststoffkiste ganz rechts",
    "also": [
        {"object": "small red metal bracket", "from": "on the black foam mat",
         "object_said": "die kleinen roten Metallwinkel", "place": "large blue plastic bin",
         "place_said": "in die große blaue Kunststoffkiste in der Mitte"},
        {"object": "large gray hex nut", "object_said": "die großen grauen Sechskantmuttern",
         "place": "small black cardboard box", "place_said": "in den kleinen schwarzen Karton ganz links"},
        {"object": "long silver hex screw", "object_said": "die langen silbernen Sechskantschrauben",
         "place_pose": "ablage_rechts"},
    ],
    "return_to": "wartepose", "scope": "until_empty",
}
_TYPICAL_SENTENCE = "grüne Teile in die gelbe Kiste, rote in die blaue, graue in die schwarze, weiße auf Ablage links"
_TYPICAL: dict[str, Any] = {
    "intent": "task", "object": "green part", "object_said": "grüne Teile", "place": "yellow bin",
    "place_said": "in die gelbe Kiste",
    "also": [RED, {"object": "gray part", "object_said": "graue", "place": "black bin", "place_said": "in die schwarze"},
             {"object": "white part", "object_said": "weiße", "place_pose": "ablage_links"}],
    "scope": "until_empty",
}


def _every_key(answer: dict[str, Any]) -> dict[str, Any]:
    """``answer`` in the earlier format: every key written, null where the sentence says nothing, in every rule."""
    full = {key: answer.get(key) for key in ANSWER_KEYS}
    full["also"] = [{key: rule.get(key) for key in RULE_KEYS} for rule in answer["also"]]
    return full


#: Each four-rule answer as the model may write it after the reader's "{", which is the prompt's, and the tokens it
#: took, counted with Qwen3-VL-4B's tokenizer on the dev box (2026-10-09, CPU, the tokenizer file alone).
_FOUR_RULES: dict[str, tuple[str, str, int]] = {
    "the longest words, spaced as the 4B writes": (_LONGEST_SENTENCE, json.dumps(_LONGEST, ensure_ascii=False), 251),
    "typical words, every key written null and spaced":
        (_TYPICAL_SENTENCE, json.dumps(_every_key(_TYPICAL), ensure_ascii=False), 229),
    "the longest words, every key written null and spaced":
        (_LONGEST_SENTENCE, json.dumps(_every_key(_LONGEST), ensure_ascii=False), 316),
}


def _qwen_tokenizer() -> Any | None:
    """Qwen3-VL-4B's tokenizer from the local Hugging Face cache, never downloaded, CPU only; ``None`` without it."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from tokenizers import Tokenizer
    except ImportError:
        return None
    path = try_to_load_from_cache("Qwen/Qwen3-VL-4B-Instruct", "tokenizer.json")
    return Tokenizer.from_file(path) if isinstance(path, str) else None


class TheLongestSortFitsTests(_Reads):
    """A four-rule answer has room under the reader's token bound, and is read as four rules."""

    def test_a_four_rule_answer_fits_the_token_bound(self) -> None:
        self.assertLessEqual(max(tokens for _, _, tokens in _FOUR_RULES.values()), COMMAND_MAX_NEW_TOKENS)
        tokenizer = _qwen_tokenizer()
        if tokenizer is None:
            return  # the counts above were measured; a box with the tokenizer counts them again
        for name, (_, answer, tokens) in _FOUR_RULES.items():
            with self.subTest(name):
                self.assertEqual(tokens, len(tokenizer.encode(answer[1:], add_special_tokens=False).ids))

    def test_each_is_read_as_four_rules_in_one_question(self) -> None:
        for name, (sentence, answer, _) in _FOUR_RULES.items():
            with self.subTest(name):
                reading, ask = self._read(sentence, answer)
                self.assertEqual((True, 1, MAX_FURTHER_RULES + 1, ()),
                                 (reading.understood, len(ask.calls), len(reading.rules), reading.notes))

    def test_the_door_gives_a_sort_its_room_and_asks_the_model(self) -> None:
        """The known sentences' table takes no sort, and the model may write the whole of one."""
        from tests.test_vlm_holder import _holder, _Loads, _models

        forget_remembered_answers()
        self.addCleanup(forget_remembered_answers)
        loads = _Loads(answer=_sort()[1:])
        holder = _holder(loads)
        self.addCleanup(holder.forget)
        reading = read_command(SORT, models=_models(backend="vlm"), poses=POSES, weights_present=True, holder=holder,
                               known=True)
        self.assertEqual((False, (GREEN_RULE, RED_RULE)), (reading.known, reading.rules))
        self.assertEqual(COMMAND_MAX_NEW_TOKENS, loads.model.calls[0]["max_new_tokens"])

    def test_an_answer_cut_short_is_never_read_as_one_of_its_rules(self) -> None:
        """Cut inside a rule, the answer holds no whole object: its first rule's braces are no command."""
        whole = _sort(also=[RED, dict(object="gray part", object_said="graue", place="black bin",
                                      place_said="in die schwarze")])
        cut = whole[:whole.index('{"object":"gray part"') + 12]
        self.assertIsNone(extract_command_object(cut))
        reading, ask = self._read(SORT, cut, _sort())
        self.assertIn("the answer holds no JSON object", ask.calls[1][1])
        self.assertEqual((True, ("retried",)), (reading.understood, reading.notes))
        self.assertEqual({"intent": "none"}, extract_command_object('{{"intent": "none"}'), "a doubled brace reads")


class ASortIsNeverAKnownSentenceTests(unittest.TestCase):
    """The known sentences' table reads one kind into one place; any sort goes to the model."""

    SORTS = (
        "grüne Teile in die gelbe Kiste, rote in die blaue",
        "Grüne Teile in die gelbe Kiste, rote in die blaue.",
        "Alle grünen Würfel in die gelbe Kiste, alle roten in die blaue Kiste",
        "Alle grünen Würfel in die gelbe Kiste und alle roten Würfel in die blaue Kiste",
        "Leg den grünen Würfel in die gelbe Kiste und den roten Würfel in die blaue Kiste",
        "Leg die Schrauben auf Ablage links und die Muttern auf Ablage rechts",
        "Alle Schrauben auf Ablage links, alle Muttern auf Ablage rechts",
        "die Schrauben in die Kiste links, die Muttern auf Ablage rechts",
        "Put the green cubes into the yellow bin and the red cubes into the blue bin",
        "Put all green cubes into the yellow bin, all red cubes into the blue bin",
    )

    def test_a_sorting_sentence_goes_to_the_model(self) -> None:
        offered = _Offered.of(POSES)
        examples = tuple(sentence for sentence, answer in COMMAND_EXAMPLES if answer.get("also"))
        self.assertEqual(3, len(examples))
        for sentence in self.SORTS + examples:
            with self.subTest(sentence=sentence):
                self.assertIsNone(known_answer(sentence, offered))
                self.assertIsNone(read_known(sentence, poses=POSES))

    def test_one_kind_into_one_place_stays_known(self) -> None:
        reading = read_known("Alle grünen Würfel in die gelbe Kiste", poses=POSES)
        assert reading is not None
        self.assertEqual((True, 1), (reading.known, len(reading.rules)))


if __name__ == "__main__":
    unittest.main()
