"""The two readings the command reader takes of a phrase before the detector is asked it: English or not, and a quantifier.

``src.models.routing.analyse`` is what is left of the prompt router (removed with GroundingDINO on 2026-10-17): the
VLM grounds every phrase now, and the command reader (``src/models/vlm/command.py``) asks the model again where a
phrase it would hand the detector is not English or carries a quantifier ("all screws"), because the detector is asked
in English and "all" belongs in the task's scope. The vocabularies are the router's, word for word, so the prompts the
router's golden set read one way read the same way here. Pure Python: no weights, no GPU, no network.
"""

from __future__ import annotations

import json
import unittest

from src.models.routing import PromptSignals, analyse

#: Phrases a reader hands the detector, in the cell's two languages, and how each reads (the router's golden set).
ENGLISH = ("a red cube", "red cube", "the mug", "a screwdriver", "small red cube", "the blue plastic box",
           "a toy rhino", "cardboard tray", "the sugar box", "the gray cube", "not the red cube", "the broken part",
           "the largest cube", "the cube behind the tray", "the upper gray cube", "the cup left of the bin")
GERMAN = ("ein roter Würfel", "der Becher", "die Schachtel", "ein Schraubendreher", "das gelbe Teil",
          "greif den Würfel", "nimm die blaue Kiste", "der kaputte Würfel", "das beschädigte Bauteil",
          "nicht den roten Würfel", "der größte Würfel", "der Würfel hinter der Kiste", "welche Teile sind kaputt",
          "Würfel", "rote Kiste")


class EnglishOrNotTests(unittest.TestCase):

    def test_every_english_phrase_reads_as_english(self) -> None:
        for phrase in ENGLISH:
            with self.subTest(phrase):
                self.assertFalse(analyse(phrase).non_english)

    def test_every_german_phrase_reads_as_not_english_however_simple_it_looks(self) -> None:
        for phrase in GERMAN:
            with self.subTest(phrase):
                self.assertTrue(analyse(phrase).non_english)

    def test_the_umlaut_and_its_digraph_read_alike(self) -> None:
        self.assertTrue(analyse("den kaputten Wuerfel").non_english)
        self.assertTrue(analyse("den kaputten Würfel").non_english)

    def test_another_language_is_caught_only_by_its_letters_outside_ascii(self) -> None:
        """A stated limit, asserted so it cannot rot into a surprise: an accentless third language reads as English."""
        self.assertTrue(analyse("el cubo pequeño").non_english)
        self.assertTrue(analyse("le cube préféré").non_english)
        self.assertFalse(analyse("il cubo rosso").non_english)

    def test_words_german_shares_with_english_do_not_read_as_german(self) -> None:
        """A false reading costs a second question, so only words with no English reading count."""
        for phrase in ("the die", "a can", "the war memorial", "in the bin", "so many cubes", "leg of the table"):
            with self.subTest(phrase):
                self.assertFalse(analyse(phrase).non_english)


class AQuantifierTests(unittest.TestCase):

    def test_a_quantifier_in_either_language_is_read(self) -> None:
        for phrase in ("every cube", "all the parts", "both cups", "each screw", "alle Würfel", "beide Becher",
                       "mehrere Teile"):
            with self.subTest(phrase):
                self.assertTrue(analyse(phrase).has_quantifier)

    def test_a_phrase_naming_one_kind_carries_none(self) -> None:
        for phrase in ("the red cube", "a cup", "the gray cube on top", "der Becher"):
            with self.subTest(phrase):
                self.assertFalse(analyse(phrase).has_quantifier)


class AReadingTests(unittest.TestCase):

    def test_an_empty_or_odd_phrase_reads_as_nothing_and_never_raises(self) -> None:
        for phrase in ("", "   ", "\n\t", "123", "!!!"):
            with self.subTest(repr(phrase)):
                self.assertEqual(PromptSignals(non_english=False, has_quantifier=False), analyse(phrase))

    def test_a_reading_is_json_safe_for_a_log_line(self) -> None:
        reading = analyse("alle Würfel").to_dict()
        self.assertEqual({"non_english": True, "has_quantifier": True}, reading)
        self.assertEqual(reading, json.loads(json.dumps(reading)))

    def test_the_same_phrase_reads_the_same_way_every_time(self) -> None:
        self.assertEqual(analyse("the largest cube"), analyse("the largest cube"))
        self.assertEqual(analyse("A RED CUBE"), analyse("  a  red   cube. "))


if __name__ == "__main__":
    unittest.main()
