"""Real weights, real GPU: the command reader on the owner's sorting sentences (the `inference` tier).

    python -m pytest tests/test_vlm_command_rules_inference.py -m inference -q

Skips unless CUDA and the weights are both present, as `tests/test_vlm_command_inference.py` does, whose one-copy
base class this file uses: one model load for the one test, each sentence a subtest. NOT part of the CI gate. The
checkpoint is the dev box's 4B unless the environment names another (`WILLY_VLM_MODEL_ID`, `WILLY_VLM_MODEL_PATH`:
the cell's 8B).

WHAT THIS EXISTS TO CATCH. `tests/test_vlm_command_rules.py` pins what the reader does with a sort's answer. Only this
file tells whether the model writes one: the owner's "grüne Teile in die gelbe Kiste, rote in die blaue" (2026-10-09)
as two rules, the noun a kind leaves out written again ("rote" -> "red ..."), a pose in a sort by its NAME, and no
``also`` for a sentence of one kind. Each check names the words a card needs, not a whole answer, and the rules in
any order, so a model that words them its own way still passes where the card would be right.

Run once on the dev box's 4B (2026-10-09, 2.0 to 3.6 s a sentence): every rule as listed, and one rule for each
sentence of one kind. Three answers carried a note, so their cards open instead of starting on Enter: the English
sentence and the "Sortier die Würfel: ..." one came without ``object_said``, and "alle Muttern" came back as
``object_said`` "Mutter". The cell's 8B is still to be run.
"""

from __future__ import annotations

import unittest
from typing import Any

import pytest

from src.models.vlm import understand
from src.models.vlm.command import COMMAND_MAX_NEW_TOKENS, CommandReading, RuleReading
from tests.test_vlm_command_inference import MODEL_ID, _cuda_available, _OnTheRealModel, _weights_present

pytestmark = pytest.mark.inference

#: The owner's taught poses as the console hands them over, with a second place pose a sort may name.
POSES = {"Ablage links": "ablage_links", "Ablage rechts": "ablage_rechts", "Wartepose": "wartepose"}

#: Each sentence and its rules, in any order: the words the part's phrase holds, and the words of its place or the
#: pose it names.
_SORTS: tuple[tuple[str, tuple[tuple[tuple[str, ...], tuple[str, ...] | None, str | None], ...]], ...] = (
    ("Grüne Teile in die gelbe Kiste, rote in die blaue.",
     ((("green",), ("yellow",), None), (("red",), ("blue",), None))),
    ("Die grünen Würfel in die gelbe Kiste und die roten in die blaue Kiste",
     ((("green", "cube"), ("yellow",), None), (("red", "cube"), ("blue",), None))),
    ("Alle Schrauben in die graue Kiste, alle Muttern auf Ablage rechts",
     ((("screw",), ("gray",), None), (("nut",), None, "ablage_rechts"))),
    ("Put the green cubes into the yellow bin and the red ones into the blue bin",
     ((("green", "cube"), ("yellow",), None), (("red", "cube"), ("blue",), None))),
    ("Sortier die Würfel: grüne in die gelbe Kiste, rote in die blaue Kiste, graue auf Ablage links",
     ((("green", "cube"), ("yellow",), None), (("red", "cube"), ("blue",), None), (("gray", "cube"), None,
                                                                                    "ablage_links"))),
)


def _matches(rule: RuleReading, words: tuple[str, ...], place: tuple[str, ...] | None, pose: str | None) -> bool:
    phrase = rule.object.phrase if rule.object is not None else ""
    if not all(word in phrase for word in words):
        return False
    if pose is not None:
        return rule.place_pose == pose
    where = rule.place.phrase if rule.place is not None else ""
    return all(word in where for word in place or ())


def _rules_text(reading: CommandReading) -> str:
    return "; ".join(f"{rule.object.phrase if rule.object else None} -> "
                     f"{rule.place.phrase if rule.place else rule.place_pose}" for rule in reading.rules)


@unittest.skipUnless(_cuda_available(), "no CUDA in this interpreter")
@unittest.skipUnless(_weights_present(), f"{MODEL_ID} not on this box (scripts/model_weights/fetch.py)")
class TheOwnersSortOnTheRealModelTests(_OnTheRealModel):
    def _ask(self) -> Any:
        return self.holder.asker_for(self.vlm, may_load=False, max_new_tokens=COMMAND_MAX_NEW_TOKENS)

    def test_the_sorting_sentences_read_as_their_rules(self) -> None:
        ask = self._ask()
        for sentence, rules in _SORTS:
            with self.subTest(sentence=sentence):
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertTrue(reading.understood, f"{reading.reason} | raw: {reading.raw}")
                self.assertEqual("task", reading.intent, reading.raw)
                self.assertEqual(len(rules), len(reading.rules), f"{_rules_text(reading)} | raw: {reading.raw}")
                for words, place, pose in rules:
                    self.assertTrue(any(_matches(rule, words, place, pose) for rule in reading.rules),
                                    f"no rule {words} -> {place or pose}: {_rules_text(reading)} | raw: {reading.raw}")
                self.assertEqual("until_empty", reading.scope, reading.raw)
        for sentence in ("Alle grauen Würfel in die Gelbe Kiste.", "Leg den grünen Würfel in die blaue Kiste"):
            with self.subTest(sentence=sentence):
                reading = understand(sentence, ask=ask, poses=POSES)
                self.assertEqual((True, 1), (reading.understood, len(reading.rules)), reading.raw)


if __name__ == "__main__":
    unittest.main()
