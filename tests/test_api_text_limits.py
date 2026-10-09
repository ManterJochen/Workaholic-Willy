"""How long a command and a card's phrase may be, at the console's door (the owner, 2026-10-08).

The morning of 2026-10-08 a person typed descriptions into the card ("der obere von den zwei grauen Würfeln, die
aufeinander auf der schwarzen Matte liegen", 85 characters) and Start answered "Die Anfrage ist ungültig.": the card's
fields took 80 characters, and the refusal did not say why. The model reads what is typed as input, which costs almost
nothing per character (about 140 ms per ANSWER token on the cell, no fixed cost), so the limits were raised:

* L1: one limit for a sentence, the reader's ``MAX_SENTENCE_CHARS`` (1000), for ``CommandIn.text`` and for the record
  a Start carries (``CommandProvenanceIn.text``), so a long sentence that was read is never refused at Start;
* L2: 200 characters for the card's phrases (``TaskIn.object``, ``CameraPlaceIn.phrase``);
* L3: a text over its limit, and nothing else wrong, is 422 ``text_too_long`` with ``{field, max}``; any other mix
  stays ``bad_request``;
* L4: the console's fields stop typing at the same numbers (``frontend/src/api/limits.ts``).

Recorded first as they stood (500 and 80, ``bad_request``), then flipped. Honesty bucket (2): the real app and its
validation, on the console_dummy tree and on the command cell of ``tests/test_api_commands.py`` (a scripted model).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any

from tests._console_task_fakes import ConsoleCase
from tests.test_api_commands import _CommandCell

_LIMITS_TS = Path(__file__).resolve().parents[1] / "frontend" / "src" / "api" / "limits.ts"


def _task(**over: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"object": "green cube", "place": {"kind": "pose", "pose": None}}
    body.update(over)
    return body


class ATaskTakesLongerPhrasesTests(ConsoleCase):
    """``POST /v1/task`` on a cell that is not connected: a body that passes the door is refused by the cell
    (``not_connected``), one that does not by its length."""

    def passes(self, body: dict[str, Any]) -> None:
        answered = self.client.post("/v1/task", json=body)
        self.assertEqual((409, "not_connected"), (answered.status_code, answered.json()["code"]), answered.text)

    def too_long(self, body: dict[str, Any], field: str, limit: int) -> None:
        refused = self.client.post("/v1/task", json=body)
        self.assertEqual((422, "text_too_long"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual((field, limit), (refused.json()["detail"]["field"], refused.json()["detail"]["max"]))
        self.assertIn(str(limit), refused.json()["message"])

    def test_an_object_of_200_characters_passes_the_door(self) -> None:
        self.passes(_task(object="x" * 200))

    def test_an_object_of_201_characters_is_too_long_and_says_its_limit(self) -> None:
        self.too_long(_task(object="x" * 201), "object", 200)

    def test_a_camera_phrase_of_200_characters_passes_the_door(self) -> None:
        self.passes(_task(place={"kind": "camera", "phrase": "y" * 200}))

    def test_a_camera_phrase_of_201_characters_is_too_long(self) -> None:
        self.too_long(_task(place={"kind": "camera", "phrase": "y" * 201}), "place.camera.phrase", 200)

    def test_a_start_carries_a_sentence_as_long_as_the_reader_reads(self) -> None:
        self.passes(_task(command={"text": "z" * 1000}))

    def test_a_start_whose_sentence_is_longer_is_too_long(self) -> None:
        self.too_long(_task(command={"text": "z" * 1001}), "command.text", 1000)

    def test_a_text_too_long_beside_another_fault_stays_a_bad_request(self) -> None:
        refused = self.client.post("/v1/task", json={"object": "x" * 201})
        self.assertEqual((422, "bad_request"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual({"object", "place"}, {error["loc"][1] for error in refused.json()["detail"]["errors"]})

    def test_a_refusal_says_where_and_why_but_never_the_text(self) -> None:
        refused = self.client.post("/v1/task", json=_task(object="secret " * 40))
        self.assertNotIn("secret", refused.text)


class TheSentenceLimitTests(_CommandCell):
    """``POST /v1/commands/parse``: the reader reads 1000 characters, and a longer sentence asks nobody."""

    def test_a_sentence_of_1000_characters_is_read(self) -> None:
        answered = self.parse("w" * 1000)
        self.assertEqual(200, answered.status_code, answered.text)
        self.assertEqual(1, len(self.loads.model.calls))

    def test_a_sentence_of_1001_characters_is_too_long_and_nothing_is_asked(self) -> None:
        refused = self.parse("w" * 1001)
        self.assertEqual((422, "text_too_long"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual({"field": "text", "max": 1000},
                         {key: refused.json()["detail"][key] for key in ("field", "max")})
        self.assertEqual((0, 0), (self.loads.calls, len(self.loads.model.calls)))


class OneLimitEverywhereTests(unittest.TestCase):
    def test_the_sentence_has_one_limit_the_readers(self) -> None:
        from api.schemas import CommandIn, CommandProvenanceIn
        from src.models.vlm.command import MAX_SENTENCE_CHARS

        self.assertEqual(1000, MAX_SENTENCE_CHARS)
        for model in (CommandIn, CommandProvenanceIn):
            with self.subTest(model=model.__name__):
                self.assertEqual(MAX_SENTENCE_CHARS, _max_length(model, "text"))

    def test_the_cards_phrases_have_one_limit(self) -> None:
        from api.schemas import MAX_FIELD_CHARS, CameraPlaceIn, TaskIn

        self.assertEqual(200, MAX_FIELD_CHARS)
        self.assertEqual(MAX_FIELD_CHARS, _max_length(TaskIn, "object"))
        self.assertEqual(MAX_FIELD_CHARS, _max_length(CameraPlaceIn, "phrase"))

    def test_the_consoles_fields_stop_at_the_servers_numbers(self) -> None:
        from api.schemas import MAX_FIELD_CHARS
        from src.models.vlm.command import MAX_SENTENCE_CHARS

        text = _LIMITS_TS.read_text(encoding="utf-8")
        numbers = dict(re.findall(r"export const ([A-Z_]+) = (\d+)", text))
        self.assertEqual({"MAX_SENTENCE_CHARS": str(MAX_SENTENCE_CHARS), "MAX_FIELD_CHARS": str(MAX_FIELD_CHARS)},
                         {key: numbers.get(key) for key in ("MAX_SENTENCE_CHARS", "MAX_FIELD_CHARS")})


def _max_length(model: Any, field: str) -> int | None:
    for meta in model.model_fields[field].metadata:
        limit = getattr(meta, "max_length", None)
        if limit is not None:
            return int(limit)
    return None


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
