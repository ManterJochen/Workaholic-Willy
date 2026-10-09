"""A task grounds the one part its sentence singled out, and every part of its kind where the parts lie (G1, 2026-10-08).

On 2026-10-08 the task grounded "each separate " + the object for every pick: right for every part of a kind ("each
separate green part" found both green parts where "green part" found one), wrong for a selection. "Den oberen der zwei
gestapelten Würfel" came back as both cubes of the stack, and the owner typed "grauer Würfel auf der schwarzen Matte"
into the card's object to say where the parts lay. The reader now reads both (``which``, ``from``), and the task builds
its phrase from them (``task.pick_phrase``):

* a which alone, as said ("the gray cube on top of the other one"): one part, no "each separate";
* else every part of the kind named where the parts lie ("each separate red cube on the black mat");
* else every part there ("each separate object on the black mat").

The detector's label still maps onto the object (a which names the object's noun, and the label mapping matches on
words), and the route guard judges the operator's words in the built phrase: a which only the VLM can ground is refused
on a phrase-only cell before anything moves (``prompt_not_routable``), as a German object always was. A plan kept
before the two fields existed reads with neither.

Honesty bucket (2): the library's ``run_task`` on the task tests' doubles, and the real console, route and run on the
scripted cell of ``tests/_console_task_fakes.py``.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import ConsoleCase, Scripted
from tests._task_fakes import run
from tests.test_api_commands import _CommandCell


def _plan(obj: str = "red cube", **fields: Any) -> Any:
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan

    return TaskPlan(object=obj, place=PlaceAt(pose="drop_left"), options=TaskOptions(pick_anything=not obj), **fields)


class ThePhraseIsBuiltFromWhatTheSentenceSaidTests(unittest.TestCase):
    def test_a_part_singled_out_is_grounded_alone_and_maps_onto_the_object(self) -> None:
        asked = run(["part"], plan=_plan("gray cube", which="the gray cube on top of the other one")).service.prompts[0]
        self.assertEqual(("the gray cube on top of the other one", "gray cube", ("gray cube",)),
                         (asked.phrase, asked.target_label, asked.object_labels))

    def test_where_the_parts_lie_follows_the_kind(self) -> None:
        asked = run(["part"], plan=_plan(source="on the black mat")).service.prompts[0]
        self.assertEqual(("each separate red cube on the black mat", "red cube"), (asked.phrase, asked.target_label))

    def test_no_kind_named_grounds_every_part_where_they_lie(self) -> None:
        asked = run(["part"], plan=_plan("", source="on the black mat")).service.prompts[0]
        self.assertEqual(("each separate object on the black mat", None, ("object",)),
                         (asked.phrase, asked.target_label, asked.object_labels))

    def test_a_which_is_grounded_alone_beside_where_the_parts_lie(self) -> None:
        asked = run(["part"], plan=_plan("gray cube", which="the upper gray cube", source="on the black mat"))
        self.assertEqual("the upper gray cube", asked.service.prompts[0].phrase)

    def test_without_either_the_phrase_is_todays(self) -> None:
        from src.robot.execution.task import EVERY_PART_PHRASE

        self.assertEqual("each separate red cube", run(["part"], plan=_plan()).service.prompts[0].phrase)
        self.assertEqual(EVERY_PART_PHRASE, run(["part"], plan=_plan("")).service.prompts[0].phrase)

    def test_the_cells_prompt_is_put_back_after(self) -> None:
        service = run(["part"], plan=_plan("gray cube", which="the upper gray cube")).service
        self.assertEqual(["the upper gray cube", "object"], [prompt.phrase for prompt in service.prompts])

    def test_a_cell_that_grounds_no_phrase_is_handed_the_object_as_before(self) -> None:
        service = run(["part"], plan=_plan("gray cube", which="the upper gray cube"), grounds=False).service
        self.assertEqual(["gray cube"], [prompt.phrase for prompt in service.prompts][:1])

    def test_the_phrase_takes_the_words_as_given_without_their_blanks(self) -> None:
        from src.robot.execution.task import pick_phrase

        self.assertEqual("the smallest cube", pick_phrase(" cube ", which="  the smallest cube "))
        self.assertEqual("each separate cube in the bin", pick_phrase("cube", source=" in the bin "))
        self.assertEqual("each separate cube", pick_phrase("cube", which="  ", source=""))

    def test_a_plan_says_its_which_and_where_as_words(self) -> None:
        for fields in ({"which": None}, {"source": 3}):
            with self.subTest(**fields), self.assertRaises(TypeError):
                _plan("cube", **fields)


class ATaskAtTheConsoleTests(ConsoleCase):
    """``POST /v1/task`` with ``which`` and ``source``: echoed in the plan, handed to the library, and judged by the
    route guard as the detector will get them."""

    def test_the_plan_echoes_them_and_the_detector_is_asked_the_one_part(self) -> None:
        from tests.test_api_task_restart import _stack

        cell = self.scripted([Scripted("part")])
        # "upper" singles out by height, which only the VLM grounds (src/models/routing/rules.py).
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            run_ = self.task(object="gray cube", which="the upper gray cube", source="on the black mat")
        body = self.finished(run_["id"])
        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual(("the upper gray cube", "on the black mat"), (body["plan"]["which"], body["plan"]["source"]))
        self.assertEqual("the upper gray cube", cell.service.prompts[0].phrase)

    def test_where_the_parts_lie_reaches_the_detector(self) -> None:
        cell = self.scripted([Scripted("part")])
        self.finished(self.task(object="gray cube", source="on the black mat")["id"])
        self.assertEqual("each separate gray cube on the black mat", cell.service.prompts[0].phrase)

    def test_a_which_only_the_vlm_can_ground_is_refused_on_a_phrase_only_cell(self) -> None:
        cell = self.scripted([Scripted("part")])
        refused = self.refused(self.client.post("/v1/task", json={
            "object": "gray cube", "which": "the gray cube on top of the other one",
            "place": {"kind": "pose", "pose": None}}), 422, "prompt_not_routable")
        self.assertIn("the gray cube on top of the other one", refused["message"])
        self.assertEqual([], cell.motions())
        self.assertIsNone(self.cell.registry.active())

    def test_a_which_said_by_its_height_or_its_side_is_refused_on_a_phrase_only_cell(self) -> None:
        """A phrase grounder boxes every gray cube for "the upper gray cube" (2026-10-08): the router sends these to
        the VLM, so a cell without one refuses them before anything moves."""
        cell = self.scripted([Scripted("part")])
        for obj, which in (("gray cube", "the upper gray cube"), ("gray cube", "the gray cube on top"),
                           ("cup", "the cup left of the bin")):
            with self.subTest(which=which):
                refused = self.refused(self.client.post("/v1/task", json={
                    "object": obj, "which": which, "place": {"kind": "pose", "pose": None}}), 422, "prompt_not_routable")
                self.assertIn(which, refused["message"])
        self.assertEqual([], cell.motions())
        self.assertIsNone(self.cell.registry.active())

    def test_the_same_which_runs_where_the_vlm_grounds(self) -> None:
        from tests.test_api_task_restart import _stack

        cell = self.scripted([Scripted("part")])
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            run_ = self.task(object="gray cube", which="the gray cube on top of the other one")
        self.assertEqual("finished", self.finished(run_["id"])["stop_code"])
        self.assertEqual("the gray cube on top of the other one", cell.service.prompts[0].phrase)

    def test_the_each_separate_of_every_task_is_no_quantifier_to_the_guard(self) -> None:
        """The task's own words: judged with them, every task on a phrase-only cell would be refused."""
        self.scripted([Scripted("part")])
        self.assertEqual("finished", self.finished(self.task(object="red cube")["id"])["stop_code"])

    def test_a_plan_kept_without_them_still_runs_as_before(self) -> None:
        from api.schemas import TaskPlanOut
        from api.task_run import library_plan

        kept = {"object": "red cube", "place": {"kind": "pose", "pose": "drop_left", "pose_joints_deg": [0.0] * 6}}
        self.assertEqual(("", ""), (TaskPlanOut.model_validate(kept).which, TaskPlanOut.model_validate(kept).source))
        library, _ = library_plan(kept)
        self.assertEqual(("red cube", "", ""), (library.object, library.which, library.source))

    def test_each_field_takes_a_cards_200_characters_and_no_more(self) -> None:
        from api.schemas import MAX_FIELD_CHARS, TaskIn
        from tests.test_api_text_limits import _max_length

        for field in ("which", "source"):
            with self.subTest(field=field):
                self.assertEqual(MAX_FIELD_CHARS, _max_length(TaskIn, field))
                refused = self.client.post("/v1/task", json={"object": "cube", field: "w" * (MAX_FIELD_CHARS + 1),
                                                             "place": {"kind": "pose", "pose": None}})
                self.assertEqual((422, "text_too_long"), (refused.status_code, refused.json()["code"]), refused.text)
                self.assertEqual((field, MAX_FIELD_CHARS),
                                 (refused.json()["detail"]["field"], refused.json()["detail"]["max"]))


class TheCardSaysTheOnePartAndWhereThePartsLieTests(_CommandCell):
    """``POST /v1/commands/parse``: the reading's which and from reach the card, and the object's route is that of the
    phrase its picks would ground, as the task's guard judges it."""

    SENTENCE = "Nimm von der schwarzen Matte den grauen Würfel, der oben auf dem anderen liegt"

    def _card(self, **fields: Any) -> Any:
        from src.models.vlm.command import _example, _json

        self.scripted(_json(_example(**fields))[1:])  # what the model writes after the reader's "{"
        answered = self.parse(self.SENTENCE)
        self.assertEqual(200, answered.status_code, answered.text)
        return answered.json()

    def test_the_which_and_the_from_are_on_the_card(self) -> None:
        card = self._card(object="gray cube", source="on the black mat", which="the gray cube on top of the other one",
                          object_said="den grauen Würfel, der oben auf dem anderen liegt")
        self.assertEqual(("the gray cube on top of the other one", "on the black mat", "once", []),
                         (card["which"], card["source"], card["scope"], card["notes"]))
        self.assertEqual(("the gray cube on top of the other one", "vlm"),
                         (card["object"]["route"]["prompt"], card["object"]["route"]["route"]))

    def test_where_the_parts_lie_is_routed_with_the_object(self) -> None:
        card = self._card(object="gray cube", source="on the black mat",
                          object_said="den grauen Würfel, der oben auf dem anderen liegt")
        self.assertEqual((None, "on the black mat"), (card["which"], card["source"]))
        self.assertEqual("gray cube on the black mat", card["object"]["route"]["prompt"])

    def test_a_card_without_either_says_neither(self) -> None:
        card = self._card(object="gray cube", object_said="den grauen Würfel")
        self.assertEqual((None, None, "gray cube"), (card["which"], card["source"], card["object"]["route"]["prompt"]))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
