"""The console reads the cell's task settings and its Settings page writes them, into the cell's own layer (the owner,
2026-10-17: "die Konsole schreibt sie").

What is held here, over real HTTP on a real config tree in a scratch directory:

* ``GET /v1/task/settings`` answers ``runtime.task`` as the tree holds it, the defaults each browser had until then
  where nobody chose (Enter starts, once, the default place, the card asking for a part, the looks when needed, parts
  not critical), the file a change is written to, and why none can be (``no_layer``);
* ``PATCH /v1/task/settings`` writes the settings named, all or none, into ``app/runtime.cell.yaml``, the runtime
  overlay of the chain's last layer, and nowhere else; the next read and the next load of the tree say them, and a
  setting left out keeps its value. A place is the default, a taught pose by its name, or what the camera finds;
* refused with nothing written: a value the loader refuses (``invalid_value``, the loader's sentence), a value no
  setting takes (``bad_request``), a phrase longer than a card's field (``text_too_long``), a pose the tree holds none
  of (``unknown_pose``), a change that names nothing (``empty_patch``), a run in progress (``run_active``), and a chain
  with no layer of the cell's own (``no_layer``); the generic ``PATCH /v1/config`` refuses every task setting;
* nothing starts: a write starts no run and moves nothing;
* the console's literals are the schema's, and a camera phrase holds what a card's field takes.

Honesty bucket (2): a real config tree in a scratch directory, real writes through the task settings' door, real HTTP.
"""

from __future__ import annotations

import typing
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from src.config.loader import load_config, reload_config
from tests.api import test_api_poses as poses

#: The settings the shipped tree holds, as the console reads them.
SHIPPED = {"start": "enter", "scope": "once", "place": {"kind": "default"}, "anything": False, "looks": "when_needed",
           "critical_parts": False}


class _Cell(poses.LayerCell):
    """A console on a scratch tree whose chain ends in the cell's own layer (``--profile cell``)."""

    def runtime_layer(self) -> Path:
        return self.tmp / "app" / "runtime.cell.yaml"

    def read(self) -> dict[str, typing.Any]:
        answer = self.client.get("/v1/task/settings")
        self.assertEqual(200, answer.status_code, answer.text)
        return dict(answer.json())

    def change(self, **values: typing.Any) -> typing.Any:
        return self.client.patch("/v1/task/settings", json=values)

    def files(self) -> dict[str, bytes]:
        return {str(path.relative_to(self.tmp)): path.read_bytes() for path in sorted(self.tmp.rglob("*.yaml"))}


class TheSettingsAreReadTests(_Cell):
    def test_the_shipped_settings_the_file_a_change_goes_to_and_no_refusal(self) -> None:
        body = self.read()
        self.assertEqual(SHIPPED, {key: body[key] for key in SHIPPED})
        self.assertEqual(self.runtime_layer(), Path(body["target_file"]))
        self.assertEqual(("", ""), (body["why_not"], body["why_not_code"]))
        self.assertFalse(self.runtime_layer().exists(), "reading wrote the cell's layer")

    def test_a_cells_own_layer_is_read_as_it_holds_them(self) -> None:
        self.runtime_layer().write_text("runtime:\n  task:\n    start: card\n    place: \"pose:park\"\n    looks: first\n",
                                        encoding="utf-8")
        reload_config()
        body = self.read()
        self.assertEqual(("card", {"kind": "pose", "pose": "park"}, "first", "once"),
                         (body["start"], body["place"], body["looks"], body["scope"]))


class TheSettingsAreWrittenIntoTheCellsOwnLayerTests(_Cell):
    def test_every_setting_lands_in_the_cells_runtime_layer_and_reads_back(self) -> None:
        before = self.files()
        answer = self.change(start="card", scope="until_empty", place={"kind": "camera", "phrase": "yellow bin"},
                             anything=True, looks="every", critical_parts=True)
        self.assertEqual(200, answer.status_code, answer.text)
        chosen = {"start": "card", "scope": "until_empty", "place": {"kind": "camera", "phrase": "yellow bin"},
                  "anything": True, "looks": "every", "critical_parts": True}
        self.assertEqual(chosen, {key: answer.json()[key] for key in chosen})
        self.assertEqual(chosen, {key: self.read()[key] for key in chosen})
        after = self.files()
        self.assertEqual([str(Path("app") / "runtime.cell.yaml")],
                         sorted(name for name in after if after[name] != before.get(name)))
        written = yaml.safe_load(self.runtime_layer().read_text(encoding="utf-8"))["runtime"]["task"]
        self.assertEqual("camera:yellow bin", written["place"])
        reload_config()
        task = load_config(self.tmp, profile="cell").runtime.task
        self.assertEqual(("card", "camera:yellow bin", True), (task.start, task.place, task.critical_parts))

    def test_a_setting_left_out_keeps_its_value(self) -> None:
        self.assertEqual(200, self.change(start="card").status_code)
        answer = self.change(scope="until_empty", looks=None)
        self.assertEqual(200, answer.status_code, answer.text)
        self.assertEqual(("card", "until_empty", "when_needed"),
                         (answer.json()["start"], answer.json()["scope"], answer.json()["looks"]))

    def test_a_taught_pose_and_the_default_place(self) -> None:
        answer = self.change(place={"kind": "pose", "pose": "park"})
        self.assertEqual(200, answer.status_code, answer.text)
        self.assertEqual({"kind": "pose", "pose": "park"}, answer.json()["place"])
        self.assertIn('place: "pose:park"', self.runtime_layer().read_text(encoding="utf-8"))
        back = self.change(place={"kind": "default"})
        self.assertEqual({"kind": "default"}, back.json()["place"])

    def test_nothing_starts_and_nothing_moves(self) -> None:
        arm = self.connect_arm(poses.TeachArm())
        self.assertEqual(200, self.change(start="enter", scope="until_empty").status_code)
        self.assertIsNone(self.cell.active_run_id)
        self.assertEqual([], self.client.get("/v1/runs").json())
        self.assertEqual([], arm.log, "the arm was asked to do something")


class ABadChangeIsRefusedWithNothingWrittenTests(_Cell):
    def setUp(self) -> None:
        super().setUp()
        self.assertEqual(200, self.change(start="card").status_code)
        self.before = self.files()

    def assert_refused(self, answer: typing.Any, status: int, code: str) -> dict[str, typing.Any]:
        self.assertEqual((status, code), (answer.status_code, answer.json().get("code")), answer.text)
        self.assertEqual(self.before, self.files(), "a refused change wrote a file")
        self.assertEqual("card", self.read()["start"])
        return dict(answer.json())

    def test_a_value_the_loader_refuses_answers_its_sentence(self) -> None:
        refused = self.assert_refused(self.change(scope="until_empty", place={"kind": "camera", "phrase": "bin #2"}),
                                      422, "invalid_value")
        for part in ("runtime.task.place", "runtime.cell.yaml", "'#'"):
            self.assertIn(part, refused["message"])
        self.assertEqual("once", self.read()["scope"], "half the group was written")

    def test_a_value_no_setting_takes(self) -> None:
        for values in ({"start": "now"}, {"looks": "all of them"}, {"scope": "twice"}, {"place": {"kind": "bin"}},
                       {"place": {"kind": "pose"}}, {"place": {"kind": "camera", "phrase": ""}}):
            with self.subTest(values=values):
                self.assert_refused(self.client.patch("/v1/task/settings", json=values), 422, "bad_request")

    def test_a_phrase_longer_than_a_cards_field(self) -> None:
        refused = self.assert_refused(self.change(place={"kind": "camera", "phrase": "y" * 201}), 422, "text_too_long")
        self.assertEqual(200, refused["detail"]["max"])

    def test_a_pose_nobody_taught(self) -> None:
        refused = self.assert_refused(self.change(place={"kind": "pose", "pose": "nowhere"}), 422, "unknown_pose")
        self.assertIn("drop_left", refused["message"])

    def test_a_change_that_names_nothing(self) -> None:
        self.assert_refused(self.change(), 400, "empty_patch")
        self.assert_refused(self.change(start=None), 400, "empty_patch")

    def test_a_run_in_progress(self) -> None:
        self.cell.active_run_id = "run-busy"
        refused = self.assert_refused(self.change(start="enter"), 409, "run_active")
        self.assertEqual("run-busy", refused["detail"]["run_id"])

    def test_the_generic_config_patch(self) -> None:
        refused = self.assert_refused(self.client.patch("/v1/config", json={"runtime.task.start": "enter"}), 403,
                                      "not_writable")
        self.assertIn("PATCH /v1/task/settings", refused["message"])


class AChainWithNoLayerOfItsOwnTests(_Cell):
    """The shipped tree with no layer of the cell's own: its ``runtime.yaml`` is the one every cell reads. The scratch
    tree stands for it (``_is_shipped_tree``), so a regression writes nowhere it matters."""

    PROFILE = None
    LAYER = None

    def setUp(self) -> None:
        shipped = patch("src.config.edit._is_shipped_tree", return_value=True)
        shipped.start()
        self.addCleanup(shipped.stop)
        super().setUp()

    def test_the_settings_say_no_file_takes_a_change_and_why(self) -> None:
        body = self.read()
        self.assertEqual(SHIPPED, {key: body[key] for key in SHIPPED})
        self.assertIsNone(body["target_file"])
        self.assertEqual("no_layer", body["why_not_code"])
        self.assertIn("runtime.yaml", body["why_not"])

    def test_a_change_is_refused_and_the_shared_file_kept(self) -> None:
        before = self.files()
        answer = self.change(start="card")
        self.assertEqual((422, "no_layer"), (answer.status_code, answer.json()["code"]), answer.text)
        self.assertEqual(before, self.files())


class TheConsolesWordsAreTheSchemasTests(unittest.TestCase):
    def test_the_literals_and_the_phrase_limit(self) -> None:
        from api.schemas import MAX_FIELD_CHARS, TaskLooksName, TaskScope, TaskStartName
        from src.config.schema.runtime import TASK_PHRASE_MAX_CHARS, TaskLooks, TaskStart
        from src.config.schema.runtime import TaskScope as ConfigScope

        self.assertEqual(typing.get_args(TaskStart), typing.get_args(TaskStartName))
        self.assertEqual(typing.get_args(TaskLooks), typing.get_args(TaskLooksName))
        self.assertEqual(typing.get_args(ConfigScope), typing.get_args(TaskScope))
        self.assertEqual(MAX_FIELD_CHARS, TASK_PHRASE_MAX_CHARS)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
