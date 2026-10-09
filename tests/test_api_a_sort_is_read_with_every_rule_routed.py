"""A sorting sentence is read with every rule, each phrase routed, and Enter starts the whole sort (the owner, 2026-10-09).

"Grüne Teile in die gelbe Kiste, rote in die blaue": ``POST /v1/commands/parse`` answers one rule per kind of part
(``CommandOut.rules``, the first the reading's own fields), each object with the route of the phrase its picks would
ground as the task route judges it (the which, or the object where the parts lie), each place with its own. The interim
guard that opened the card for every sort is gone, as the task, the API and the console carry every rule
(``TaskIn.more_rules``): a sort is ``startable`` where every rule reads clean, and a note on any rule opens the card. The
console, the command reader and the task bound a sort by one number (``MAX_FURTHER_RULES``). What is held here:

* a clean sort comes back with every rule, each phrase routed, the first rule the reading's own, and startable;
* a note on a further rule opens the card, and the note stands on that rule;
* a rule's object is routed as its picks ground it: where its parts lie added, the one part singled out alone;
* a sentence of one kind reads one rule, as its own fields say it;
* reading a sort starts nothing;
* the console's bound on further rules is the reader's and the task's, and ``TaskIn`` takes that many and no more.

Honesty bucket (2): the real holder and the real reader over a scripted model (``tests/test_api_commands.py``'s cell),
the reading itself scripted where only the routes are asked; no weights, no GPU.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from tests.test_api_commands import _CommandCell, _continuation

#: The owner's sentence of 2026-10-09, and its second rule as a well-behaved model writes it.
SORT = "grüne Teile in die gelbe Kiste, rote in die blaue"
RED: dict[str, Any] = {"object": "red part", "object_said": "rote", "place": "blue bin", "place_said": "in die blaue"}


def _sorted(**first: Any) -> str:
    """The model's answer to :data:`SORT`: green parts into the yellow bin, ``also`` the further rules (``[RED]``)."""
    fields: dict[str, Any] = {"object": "green part", "object_said": "grüne Teile", "place": "yellow bin",
                              "place_said": "in die gelbe Kiste", "scope": "until_empty", "also": [RED]}
    fields.update(first)
    return _continuation(**fields)


class ASortIsReadWithEveryRuleTests(_CommandCell):
    def test_every_rule_comes_back_routed_and_a_clean_sort_may_start_on_enter(self) -> None:
        self.scripted(_sorted())

        answered = self.parse(SORT)

        self.assertEqual(200, answered.status_code, answered.text)
        card = answered.json()
        self.assertEqual((True, "task", "until_empty", []), (card["understood"], card["intent"], card["scope"],
                                                              card["notes"]))
        self.assertEqual(2, len(card["rules"]))
        green, red = card["rules"]
        self.assertEqual(("green part", "grüne Teile", "yellow bin", "in die gelbe Kiste", []),
                         (green["object"]["phrase"], green["object"]["said"], green["place"]["phrase"],
                          green["place"]["said"], green["notes"]))
        self.assertEqual(("red part", "rote", "blue bin", "in die blaue", []),
                         (red["object"]["phrase"], red["object"]["said"], red["place"]["phrase"], red["place"]["said"],
                          red["notes"]))
        self.assertEqual([("green part", "simple", "yellow bin"), ("red part", "simple", "blue bin")],
                         [(rule["object"]["route"]["prompt"], rule["object"]["route"]["route"],
                           rule["place"]["route"]["prompt"]) for rule in card["rules"]])
        self.assertEqual((card["object"], card["place"]), (green["object"], green["place"]),
                         "the first rule is not the reading's own object and place, routes included")
        self.assertTrue(card["startable"], "a clean sort opened the card: Enter starts the whole sort now")

    def test_a_note_on_a_further_rule_opens_the_card(self) -> None:
        self.scripted(_sorted(also=[dict(RED, object_said="roten")]))

        card = self.parse(SORT).json()

        self.assertEqual(([], ["object_not_in_sentence"]), (card["rules"][0]["notes"], card["rules"][1]["notes"]))
        self.assertEqual(["object_not_in_sentence"], card["notes"])
        self.assertFalse(card["startable"])
        self.assertEqual("red part", card["rules"][1]["object"]["route"]["prompt"], "a rule with a note lost its route")

    def test_a_rule_s_object_is_routed_as_its_picks_ground_it(self) -> None:
        """Where the parts of a rule lie is added to its kind, and the one part a rule singles out is asked alone: the
        task route judges each rule's phrase so (``api.routers.task._grounded``)."""
        from src.models.vlm.command import CommandReading, PhraseReading, RuleReading

        green = RuleReading(object=PhraseReading("green part", "grüne Teile", True),
                            place=PhraseReading("yellow bin", "in die gelbe Kiste", True))
        red = RuleReading(object=PhraseReading("red part", "rote", True), source="on the black mat",
                          place=PhraseReading("blue bin", "in die blaue", True))
        cube = RuleReading(object=PhraseReading("gray cube", "den grauen Würfel", True),
                           which="the gray cube on top of the other one", place_pose="drop_left")
        reading = CommandReading(understood=True, intent="task", object=green.object, place=green.place,
                                 scope="once", rules=(green, red, cube), model_id="scripted", attempts=1)

        with patch("api.routers.commands.read_command", return_value=reading):
            answered = self.parse(SORT)

        self.assertEqual(200, answered.status_code, answered.text)
        rules = answered.json()["rules"]
        self.assertEqual(["green part", "red part on the black mat", "the gray cube on top of the other one"],
                         [rule["object"]["route"]["prompt"] for rule in rules])
        self.assertEqual(("blue bin", None, "drop_left"),
                         (rules[1]["place"]["route"]["prompt"], rules[2]["place"], rules[2]["place_pose"]))
        self.assertEqual(("on the black mat", "the gray cube on top of the other one"),
                         (rules[1]["source"], rules[2]["which"]))

    def test_a_sentence_of_one_kind_reads_one_rule_as_its_own_fields(self) -> None:
        card = self.parse().json()

        self.assertEqual(1, len(card["rules"]))
        rule = card["rules"][0]
        self.assertEqual((card["object"], card["place"], card["place_pose"], []),
                         (rule["object"], rule["place"], rule["place_pose"], rule["notes"]))
        self.assertEqual("green cube", rule["object"]["route"]["prompt"])
        self.assertTrue(card["startable"])

    def test_reading_a_sort_starts_nothing(self) -> None:
        self.scripted(_sorted())

        self.parse(SORT)

        self.assertEqual([], self.client.get("/v1/runs").json())
        self.assertIsNone(self.cell.active_run_id)


class OneBoundForASortTests(unittest.TestCase):
    def test_the_console_the_reader_and_the_task_bound_a_sort_alike(self) -> None:
        from api.schemas import MAX_FURTHER_RULES
        from src.models.vlm.command import MAX_FURTHER_RULES as READER
        from src.robot.execution.task import MAX_FURTHER_RULES as TASK

        self.assertEqual(READER, MAX_FURTHER_RULES)
        self.assertEqual(TASK, MAX_FURTHER_RULES)
        self.assertEqual(3, MAX_FURTHER_RULES)

    def test_a_task_takes_that_many_further_rules_and_no_more(self) -> None:
        from pydantic import ValidationError

        from api.schemas import MAX_FURTHER_RULES, TaskIn

        def rules(count: int) -> list[dict[str, Any]]:
            return [{"object": f"part {n}", "place": {"kind": "pose", "pose": None}} for n in range(count)]

        body = {"object": "green part", "place": {"kind": "pose", "pose": None}}
        self.assertEqual(MAX_FURTHER_RULES,
                         len(TaskIn.model_validate({**body, "more_rules": rules(MAX_FURTHER_RULES)}).more_rules))
        with self.assertRaises(ValidationError):
            TaskIn.model_validate({**body, "more_rules": rules(MAX_FURTHER_RULES + 1)})
        self.assertEqual([], TaskIn.model_validate(body).more_rules, "a task of one kind names further rules")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
