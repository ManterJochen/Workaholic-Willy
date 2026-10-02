"""Commands (build plan 1.9, OD 22, Q8): the VLM reads the operator's sentence into a card, and nothing moves.

``POST /v1/commands/parse`` hands the whole sentence, German or English, to the one VLM copy of the process
(``read_command`` through ``shared_vlm``), with the taught poses as label -> name pairs, and answers what it understood:
the object and the place as English phrases with the operator's own words and their route, a pose NAME for a spoken
label, the scope. What is held here:

* reading never creates a run, never sets a prompt and never touches the cell; it is refused during a run
  (``run_active``), before anything is asked, so a teach or a task never shares the card with a parse;
* the load rule is the library's (Q8 A): a cell whose detector is the VLM loads it at its first command; any other cell
  refuses (``vlm_not_loaded``) until a person loads it with ``POST /v1/commands/warmup`` ("Laden"); missing weights are
  ``vlm_model_missing``, a load that failed ``vlm_unavailable``, each at the catalog's status;
* the models section read is the one the cell was BUILT with where the console keeps it, the tree's otherwise;
* ``GET /v1/commands/status`` says where the reader stands and loads nothing;
* the reader's own words are the console's: its refusals are catalog codes, its notes the card's notes, its states the
  status's and the ready bar's commands light's; a refusal the catalog does not know is ``vlm_unavailable``, never a
  500.

Honesty bucket (2): the VLM is the real holder and the real reader over a scripted model (``tests/test_vlm_holder``'s
doubles); no weights, no GPU.
"""

from __future__ import annotations

import json
import re
import unittest
from typing import Any
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.models.vlm import shared_vlm
from tests.test_api_poses import LayerCell
from tests.test_vlm_holder import _holder, _Loads


def _continuation(**fields: Any) -> str:
    """What the model writes after the reader's ``{`` prefill: one answer with every key, ``fields`` overriding."""
    base: dict[str, Any] = {
        "intent": "task", "object": "", "object_said": None, "place": None, "place_said": None,
        "place_pose": None, "scope": "once", "count": None, "return_to": None,
    }
    base.update(fields)
    return json.dumps(base, ensure_ascii=False)[1:]


#: The owner's sentence and the answer a well-behaved model gives: a label they said comes back as the pose's name.
SENTENCE = "Nimm den grünen Würfel und leg ihn auf Ablage links"
ANSWER = _continuation(object="green cube", object_said="den grünen Würfel", place_pose="Ablage links")


class _CommandCell(LayerCell):
    """The layer cell (``drop_left`` labelled "Ablage links"), its VLM a scripted one through a fresh holder."""

    BACKEND = "vlm"
    WEIGHTS = True

    def setUp(self) -> None:
        super().setUp()
        models = self.tmp / "models" / "object.yaml"
        text = models.read_text(encoding="utf-8")
        text, hits = re.subn(r"^(\s*)backend: grounded_sam", rf"\g<1>backend: {self.BACKEND}", text, count=1,
                             flags=re.MULTILINE)
        assert hits == 1, "the backend line was not found"
        models.write_text(text, encoding="utf-8")
        shared_vlm().forget()
        self.addCleanup(shared_vlm().forget)
        self.loads = self.scripted(ANSWER)
        weights = patch("api.routers.diagnostics._vlm_weights_present", side_effect=lambda *_a: self.WEIGHTS)
        weights.start()
        self.addCleanup(weights.stop)

    def scripted(self, answer: str, **kwargs: Any) -> _Loads:
        """A fresh holder whose model writes ``answer``; the console's door asks it."""
        loads = _Loads(answer=answer, **kwargs)
        holder = _holder(loads)
        self.addCleanup(holder.forget)
        door = patch("api.routers.commands.shared_vlm", return_value=holder)
        door.start()
        self.addCleanup(door.stop)
        return loads

    def parse(self, text: str = SENTENCE) -> Any:
        return self.client.post("/v1/commands/parse", json={"text": text, "source": "spoken", "language": "de"})


class AVlmCellReadsItsFirstCommandTests(_CommandCell):
    def test_the_sentence_becomes_a_card_with_its_routes_and_a_pose_name_for_the_label(self) -> None:
        answered = self.parse()

        self.assertEqual(200, answered.status_code, answered.text)
        card = answered.json()
        self.assertEqual((True, "task", "once"), (card["understood"], card["intent"], card["scope"]))
        self.assertEqual(("green cube", "den grünen Würfel", True),
                         (card["object"]["phrase"], card["object"]["said"], card["object"]["verified"]))
        self.assertIsNotNone(card["object"]["route"], "the object's route was not previewed")
        self.assertEqual(("green cube", "simple"), (card["object"]["route"]["prompt"], card["object"]["route"]["route"]))
        self.assertEqual("drop_left", card["place_pose"], "the spoken label did not come back as the pose's name")
        self.assertIsNone(card["place"])
        self.assertTrue(card["model"]["loaded_now"])
        self.assertEqual(1, self.loads.calls)

    def test_reading_creates_no_run_sets_no_prompt_and_touches_no_cell(self) -> None:
        self.parse()
        self.assertEqual([], self.client.get("/v1/runs").json())
        self.assertIsNone(self.cell.active_run_id)
        self.assertEqual("disconnected", self.client.get("/v1/cell").json()["state"])

    def test_a_camera_place_is_previewed_too(self) -> None:
        self.scripted(_continuation(object="screw", object_said="Schrauben", place="blue bin",
                                    place_said="in die blaue Kiste", scope="until_empty"))
        card = self.parse("Leg alle Schrauben in die blaue Kiste").json()
        self.assertEqual(("blue bin", "until_empty"), (card["place"]["phrase"], card["scope"]))
        self.assertEqual("blue bin", card["place"]["route"]["prompt"])

    def test_a_sentence_read_as_stop_only_shows_where_the_stop_buttons_are(self) -> None:
        self.scripted(_continuation(intent="stop"))
        card = self.parse("Stopp!").json()
        self.assertEqual(("stop", None, None, None), (card["intent"], card["object"], card["place"], card["scope"]))
        self.assertEqual([], self.client.get("/v1/runs").json())

    def test_a_run_in_progress_refuses_before_anything_is_asked(self) -> None:
        self.cell.active_run_id = "run-teaching"
        refused = self.parse()
        self.assertEqual((409, "run_active"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual("run-teaching", refused.json()["detail"]["run_id"])
        self.assertEqual(0, self.loads.calls)

    def test_a_sentence_of_blanks_is_a_bad_request(self) -> None:
        refused = self.parse("   ")
        self.assertEqual((422, "bad_request"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(0, self.loads.calls)

    def test_the_status_says_where_the_reader_stands_and_loads_nothing(self) -> None:
        status = self.client.get("/v1/commands/status")
        self.assertEqual(200, status.status_code, status.text)
        body = status.json()
        self.assertEqual(("idle", "Qwen/Qwen3-VL-4B-Instruct", True, True),
                         (body["state"], body["model_id"], body["weights_present"], body["shared_with_detection"]))
        self.assertEqual(0, self.loads.calls)
        self.parse()
        body = self.client.get("/v1/commands/status").json()
        self.assertEqual("ready", body["state"])
        self.assertIsNotNone(body["last_latency_ms"])

    def test_a_load_that_fails_is_unavailable_and_a_command_does_not_retry_it(self) -> None:
        loads = self.scripted(ANSWER, fail=OSError("no weights on disk"))
        refused = self.parse()
        self.assertEqual((501, "vlm_unavailable"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("no weights on disk", refused.json()["message"])
        self.assertEqual("failed", self.client.get("/v1/commands/status").json()["state"])
        self.assertEqual(501, self.parse().status_code)
        self.assertEqual(1, loads.calls, "a command tried the failed load again")


class MissingWeightsTests(_CommandCell):
    WEIGHTS = False

    def test_a_command_and_laden_are_refused_and_nothing_loads(self) -> None:
        refused = self.parse()
        self.assertEqual((501, "vlm_model_missing"), (refused.status_code, refused.json()["code"]), refused.text)
        warm = self.client.post("/v1/commands/warmup")
        self.assertEqual((501, "vlm_model_missing"), (warm.status_code, warm.json()["code"]), warm.text)
        self.assertEqual("missing", self.client.get("/v1/commands/status").json()["state"])
        self.assertEqual(0, self.loads.calls)


class AGroundingDinoCellLoadsOnlyOnLadenTests(_CommandCell):
    BACKEND = "grounded_sam"

    def test_a_command_is_refused_until_a_person_loads_the_reader(self) -> None:
        refused = self.parse()
        self.assertEqual((409, "vlm_not_loaded"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(0, self.loads.calls, "a command loaded the VLM on a cell that does not detect with it")
        status = self.client.get("/v1/commands/status").json()
        self.assertEqual(("idle", False), (status["state"], status["shared_with_detection"]))

        warm = self.client.post("/v1/commands/warmup")

        self.assertEqual(200, warm.status_code, warm.text)
        self.assertEqual("ready", warm.json()["state"])
        self.assertEqual(1, self.loads.calls)
        read = self.parse()
        self.assertEqual(200, read.status_code, read.text)
        self.assertFalse(read.json()["model"]["loaded_now"])
        self.assertEqual(1, self.loads.calls)

    def test_laden_is_refused_during_a_run(self) -> None:
        self.cell.active_run_id = "run-busy"
        refused = self.client.post("/v1/commands/warmup")
        self.assertEqual((409, "run_active"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(0, self.loads.calls)

    def test_the_models_the_cell_was_built_with_decide_not_a_yaml_edited_since(self) -> None:
        """A VLM cell built, and its models YAML switched to GroundingDINO since: the first command still loads, as the
        built cell detects with the VLM; and the other way round it does not."""
        from types import SimpleNamespace

        from tests.test_vlm_holder import _models

        self.cell.built_models = _models(backend="vlm")  # type: ignore[attr-defined]
        self.assertEqual(200, self.parse().status_code)
        self.assertEqual(1, self.loads.calls)
        shared_vlm().forget()
        loads = self.scripted(ANSWER)
        self.cell.built_models = SimpleNamespace(pipeline=None)  # type: ignore[attr-defined]
        refused = self.parse()
        self.assertEqual((501, "vlm_unavailable"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual("not_configured", self.client.get("/v1/commands/status").json()["state"])
        self.assertEqual(0, loads.calls)


class TheReaderSpeaksTheConsoleSWordsTests(_CommandCell):
    """The library's command reader and the console's catalog are two lists of the same words (pinned at the merge):
    a word one has and the other lacks would answer a 500 at the first sentence that meets it."""

    def test_its_refusals_notes_and_states_are_the_console_s(self) -> None:
        from typing import get_args

        from api.codes import LIGHT_CODES, CommandNote, LightId, RefusalCode
        from api.schemas import CommandStatusOut
        from src.models.vlm.availability import ReaderRefusal, ReaderState
        from src.models.vlm.command import NOTE_ORDER

        self.assertLessEqual(set(get_args(ReaderRefusal)), {code.value for code in RefusalCode})
        self.assertEqual(NOTE_ORDER, tuple(note.value for note in CommandNote))
        self.assertEqual(get_args(ReaderState), get_args(CommandStatusOut.model_fields["state"].annotation))
        self.assertEqual(set(get_args(ReaderState)), {code.value for code in LIGHT_CODES[LightId.COMMANDS]})

    def test_a_refusal_the_catalog_does_not_know_is_unavailable_never_a_500(self) -> None:
        from types import SimpleNamespace

        from src.models.vlm import CommandRefused

        unknown = CommandRefused("vlm_on_strike", "the reader refused in a word the console does not know")  # type: ignore[arg-type]
        with patch("api.routers.commands.read_command", side_effect=unknown):
            refused = self.parse()
        self.assertEqual((501, "vlm_unavailable"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("does not know", refused.json()["message"])

        said = SimpleNamespace(warmup_refusal="vlm_on_strike", cause="the loader refused in a word nobody knows")
        with patch("api.routers.commands.reader_availability", return_value=said):
            warm = self.client.post("/v1/commands/warmup")
        self.assertEqual((501, "vlm_unavailable"), (warm.status_code, warm.json()["code"]), warm.text)
        self.assertEqual(0, self.loads.calls)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
