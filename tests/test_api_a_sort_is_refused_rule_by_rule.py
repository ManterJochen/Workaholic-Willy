"""``POST /v1/task`` takes a sort and refuses it rule by rule before anything moves (the owner, 2026-10-09).

"Grüne Teile in die gelbe Kiste, rote in die blaue" is one task of several rules: the task's own object, which, source
and place, and up to three ``more_rules``. The route keeps every refusal of 1.3.2 for every rule, in its order: the
camera target the cell can find (once, where any place is a camera's), each rule's kind (a sort names every one, so
``object_required`` names the rule), each rule's phrase and each camera place's, each taught pose a rule places at
(``unknown_pose`` naming ``place of rule 2``), a place nobody declared; then a sort on a cell whose perception grounds
no phrase (``bad_request``) and whatever the library's plan refuses (one kind in two rules, a ``|`` or the word
``ambiguous`` in a phrase: ``bad_request``). The plan resolves each further rule's place as the first's and echoes it
in ``run_started``, the library is handed every rule, and a Restart checks every rule again and runs the sort again.
What this file pins:

* the plan's ``more_rules``, each place resolved (a taught pose's label and joints, the default place pose where a pose
  place names none, a camera's phrase and words), echoed in ``run_started`` and the run, and handed to ``run_task`` as
  the plan's ``SortRule`` entries with the poses they name and the air over a bin's rim;
* each refusal above, with what it names, nothing moved and no run started, and their order;
* a Restart of a stopped sort runs every rule again, and refuses a rule the cell can no longer ground, or a sort it can
  no longer tell apart.

Honesty bucket (2): the real console, routes, registry and plan on the scripted cell of
``tests/_console_task_fakes.py`` and console_dummy; the library's ``run_task`` replaced where only what it was handed
is read.
"""

from __future__ import annotations

import unittest
from contextlib import contextmanager
from typing import Any, Iterator
from unittest.mock import patch

from tests._console_task_fakes import POSES_DEG, ConsoleCase, SightedLocator, events_of
from tests.test_api_task_restart import _stack, clear

#: Green parts at the default place, red ones at the park pose: a sort of taught poses, which needs no camera.
GREEN = {"object": "green part", "object_said": "grüne Teile", "place": {"kind": "pose", "pose": None}}
RED = {"object": "red part", "object_said": "rote", "place": {"kind": "pose", "pose": "park"}}


def _sort(*more: dict[str, Any], **first: Any) -> dict[str, Any]:
    """A sort's body: :data:`GREEN` as the task's own rule (``first`` changing it), the further rules ``more``
    (:data:`RED` where none are given), until empty."""
    return {**GREEN, "scope": "until_empty", "more_rules": list(more) if more else [RED], **first}


def _joints(name: str) -> list[float]:
    return [float(value) for value in POSES_DEG[name][1]]


@contextmanager
def _handed(stop: str = "finished") -> Iterator[list[dict[str, Any]]]:
    """``run_task`` replaced by one that keeps what it was handed (the plan, and every keyword) and ends ``stop``."""
    from src.robot.execution.task import TaskReport, TaskStop

    handed: list[dict[str, Any]] = []

    def run_task(_service: Any, plan: Any, **keywords: Any) -> Any:
        handed.append({"plan": plan, **keywords})
        return TaskReport(stop=TaskStop(stop), sentence=f"the scripted task ended {stop}")

    with patch("src.robot.execution.task.run_task", side_effect=run_task):
        yield handed


class _SortCase(ConsoleCase):
    def assertNothingStarted(self, cell: Any = None) -> None:  # noqa: N802 (unittest's own style)
        self.assertEqual([], self.client.get("/v1/runs").json(), "a refused sort started a run")
        if cell is not None:
            self.assertEqual([], cell.motions(), "a refused sort moved the arm")
            self.assertEqual(0, cell.do0_changes(), "a refused sort switched the jaws")

    def refused_sort(self, status: int, code: str, body: dict[str, Any]) -> dict[str, Any]:
        return self.refused(self.client.post("/v1/task", json=body), status, code)

    def camera(self, cell: Any, *phrases: str) -> None:
        """The scripted wrist cell's camera, which finds nothing it is asked for: enough for the route's camera check."""
        locator = SightedLocator({phrase: lambda _tcp: () for phrase in phrases}, arm=cell.arm, wrist=True)
        for where in ("src.robot.execution.place_target.locators_for_service",
                      "src.robot.execution.task.locators_for_service"):
            patcher = patch(where, return_value=[locator])
            patcher.start()
            self.addCleanup(patcher.stop)


class ASortsPlanTests(_SortCase):
    def test_every_rule_s_place_is_resolved_echoed_and_handed_to_the_library(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule

        self.scripted()
        blue = {"object": " blue part ", "source": "on the black mat", "place": {"kind": "pose", "pose": None}}
        with _handed() as handed:
            run = self.task(**_sort(RED, blue))
            body = self.finished(run["id"])

        self.assertEqual("finished", body["stop_code"], body)
        self.assertEqual([
            {"object": "red part", "object_said": "rote", "which": "", "source": "",
             "place": {"kind": "pose", "pose": "park", "pose_label": "Parkposition", "pose_joints_deg": _joints("park"),
                       "phrase": None, "said": None}},
            {"object": "blue part", "object_said": None, "which": "", "source": "on the black mat",
             "place": {"kind": "pose", "pose": "drop_left", "pose_label": "Ablage links",
                       "pose_joints_deg": _joints("drop_left"), "phrase": None, "said": None}},
        ], body["plan"]["more_rules"])
        self.assertEqual(("pose", "drop_left"), (body["plan"]["place"]["kind"], body["plan"]["place"]["pose"]))
        started = events_of(self.cell, run["id"])[0]
        self.assertEqual(("run_started", body["plan"]), (started.type, started.data["plan"]))
        self.assertEqual(body["plan"], run["plan"])
        plan = handed[0]["plan"]
        self.assertEqual((SortRule("red part", PlaceAt(pose="park")),
                          SortRule("blue part", PlaceAt(pose="drop_left"), source="on the black mat")), plan.more_rules)
        self.assertEqual(("green part", PlaceAt(pose="drop_left"), "until_empty"), (plan.object, plan.place, plan.scope))
        self.assertEqual({"drop_left", "park"}, set(handed[0]["poses"]))
        self.assertNotIn("known_target", handed[0], "a sort of taught poses was handed a bin")

    def test_a_further_rule_s_camera_place_takes_the_air_over_its_rim(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule

        cell = self.scripted(wrist=True)
        self.camera(cell, "blue bin")
        red = {**RED, "place": {"kind": "camera", "phrase": " blue bin ", "said": "in die blaue"}}
        with _handed() as handed:
            run = self.task(**_sort(red))
            body = self.finished(run["id"])

        self.assertEqual({"kind": "camera", "pose": None, "pose_label": None, "pose_joints_deg": None,
                          "phrase": "blue bin", "said": "in die blaue"}, body["plan"]["more_rules"][0]["place"])
        self.assertEqual(20.0, body["plan"]["options"]["rim_air_mm"], "a sort into a bin left the rim air unresolved")
        self.assertEqual((SortRule("red part", PlaceAt(camera="blue bin", air_mm=20.0)),), handed[0]["plan"].more_rules)
        self.assertEqual({}, handed[0]["known_targets"], "a bin nobody kept was handed to the sort")

    def test_a_task_of_one_kind_plans_no_further_rule(self) -> None:
        self.scripted()
        with _handed() as handed:
            run = self.task(object="green part")
            body = self.finished(run["id"])

        self.assertEqual([], body["plan"]["more_rules"])
        self.assertEqual((), handed[0]["plan"].more_rules)


class ASortIsRefusedBeforeAnythingMovesTests(_SortCase):
    def test_a_camera_place_of_any_rule_needs_a_camera_to_find_it(self) -> None:
        cell = self.scripted()
        red = {**RED, "place": {"kind": "camera", "phrase": "blue bin"}}
        with patch("src.robot.execution.place_target.locators_for_service", return_value=[]):
            refused = self.refused_sort(409, "camera_target_unavailable", _sort(red))
        self.assertIn("no camera", refused["message"])
        self.assertNothingStarted(cell)

    def test_a_rule_that_names_no_kind_is_named(self) -> None:
        cell = self.scripted()
        cases = {
            "the task's own, anything ticked": (_sort(object="", options={"pick_anything": True}), 1),
            "a further one": (_sort({**RED, "object": "  "}), 2),
            "the third": (_sort(RED, {"object": " ", "place": {"kind": "pose", "pose": None}}), 3),
        }
        for name, (body, rule) in cases.items():
            with self.subTest(name):
                refused = self.refused_sort(422, "object_required", body)
                self.assertEqual({"rule": rule}, refused["detail"])
                self.assertIn(f"rule {rule} of the sort names no kind of part", refused["message"])
        self.assertNothingStarted(cell)

    def test_a_further_rule_s_phrases_the_detector_would_ground_wrong(self) -> None:
        cell = self.scripted()
        refused = self.refused_sort(422, "prompt_not_routable", _sort({**RED, "object": "den roten Würfel"}))
        self.assertIn("den roten Würfel", refused["message"])
        self.camera(cell, "die blaue Kiste")
        red = {**RED, "place": {"kind": "camera", "phrase": "die blaue Kiste"}}
        refused = self.refused_sort(422, "target_not_routable", _sort(red))
        self.assertIn("die blaue Kiste", refused["message"])
        self.assertNothingStarted(cell)

    def test_a_further_rule_s_pose_nobody_taught_is_named(self) -> None:
        cell = self.scripted()
        refused = self.refused_sort(422, "unknown_pose", _sort({**RED, "place": {"kind": "pose", "pose": "nowhere"}}))
        self.assertEqual({"which": "place of rule 2", "pose": "nowhere"}, refused["detail"])
        self.assertIn("the place of rule 2 names the pose 'nowhere'", refused["message"])
        # The task's own place is said as it always was.
        refused = self.refused_sort(422, "unknown_pose", _sort(place={"kind": "pose", "pose": "nowhere"}))
        self.assertEqual("place", refused["detail"]["which"])
        self.assertNothingStarted(cell)

    def test_a_sort_on_a_cell_whose_perception_grounds_no_phrase(self) -> None:
        cell = self.scripted()
        cell.service.grounds = False
        refused = self.refused_sort(422, "bad_request", _sort())
        self.assertEqual({"field": "more_rules"}, refused["detail"])
        self.assertIn("grounds no phrase", refused["message"])
        self.assertNothingStarted(cell)

    def test_a_sort_on_the_rehearsal_cell(self) -> None:
        """The rehearsal scene names its parts by no word a rule could go by."""
        self.build_dummy()
        self.refused_sort(422, "bad_request", _sort())
        self.assertNothingStarted()

    def test_what_the_library_cannot_sort_is_a_bad_request(self) -> None:
        cell = self.scripted()
        cases = {
            "one kind in two rules": (_sort({**RED, "object": "Green  Part"}), "both name"),
            "a bar in a kind": (_sort({**RED, "object": "red | blue part"}), "'|'"),
            "the detector's own word": (_sort({**RED, "object": "ambiguous part"}), "ambiguous"),
            "two rules that ground one phrase": (
                _sort({**RED, "which": "the big part"}, which="the big part"), "ground one phrase"),
        }
        for name, (body, said) in cases.items():
            with self.subTest(name):
                refused = self.refused_sort(422, "bad_request", body)
                self.assertIn("the task cannot be planned", refused["message"])
                self.assertIn(said, refused["message"])
        self.assertNothingStarted(cell)

    def test_more_than_three_further_rules_is_no_task(self) -> None:
        cell = self.scripted()
        rules = [{"object": f"part {n}", "place": {"kind": "pose", "pose": None}} for n in range(4)]
        self.refused_sort(422, "bad_request", _sort(*rules))
        self.assertNothingStarted(cell)

    def test_the_rules_are_refused_in_the_order_of_1_3_2(self) -> None:
        """Every rule's kind before any rule's phrase, every phrase before any pose, any pose before the cell's
        grounding; the camera before the kinds."""
        cell = self.scripted()
        nowhere = {"kind": "pose", "pose": "nowhere"}
        self.refused_sort(422, "object_required",
                          _sort({**RED, "object": "den roten Würfel"}, {"object": " ", "place": nowhere}))
        self.refused_sort(422, "prompt_not_routable",
                          _sort({**RED, "place": nowhere}, {"object": "den blauen Würfel", "place": nowhere}))
        cell.service.grounds = False
        self.refused_sort(422, "unknown_pose", _sort({**RED, "place": nowhere}))
        with patch("src.robot.execution.place_target.locators_for_service", return_value=[]):
            self.refused_sort(409, "camera_target_unavailable",
                              _sort({**RED, "object": " ", "place": {"kind": "camera", "phrase": "blue bin"}}))
        self.assertNothingStarted(cell)


class NoDefaultPlaceTests(_SortCase):
    default_place = False

    def test_a_further_rule_s_pose_place_that_names_none_on_a_cell_with_no_default(self) -> None:
        cell = self.scripted()
        refused = self.refused_sort(422, "no_place_declared",
                                    _sort({**RED, "place": {"kind": "pose", "pose": None}},
                                          place={"kind": "pose", "pose": "drop_left"}))
        self.assertEqual({"rule": 2}, refused["detail"])
        self.assertIn("rule 2 of the sort names no place pose", refused["message"])
        self.assertNothingStarted(cell)


class ARestartRunsTheSortAgainTests(_SortCase):
    def _stopped(self, **body: Any) -> dict[str, Any]:
        """A sort that stopped on a problem, and a person who said the cell is clear and answered the jaws open."""
        with _handed("failed_in_a_row"):
            stopped = self.finished(self.task(**_sort(**body))["id"])
        self.assertEqual("failed_in_a_row", stopped["stop_code"], stopped)
        clear(self)
        return stopped

    def test_a_restart_runs_every_rule_again(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule

        self.scripted()
        stopped = self._stopped()

        with _handed() as handed:
            answered = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
            self.assertEqual(202, answered.status_code, answered.text)
            restarted = self.finished(answered.json()["id"])

        self.assertEqual(("finished", stopped["id"]), (restarted["stop_code"], restarted["restart_of"]))
        self.assertEqual(stopped["plan"]["more_rules"], restarted["plan"]["more_rules"])
        plan = handed[0]["plan"]
        self.assertEqual(("return", (SortRule("red part", PlaceAt(pose="park")),)), (plan.first_motion, plan.more_rules))

    def test_a_restart_refuses_a_sort_the_cell_can_no_longer_tell_apart(self) -> None:
        cell = self.scripted()
        stopped = self._stopped()
        cell.service.grounds = False
        moved = list(cell.motions())

        refused = self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 422,
                               "bad_request")

        self.assertEqual({"field": "more_rules"}, refused["detail"])
        self.assertEqual(moved, cell.motions())
        self.assertEqual(stopped["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]

    def test_a_restart_refuses_a_further_rule_the_cell_would_now_ground_wrong(self) -> None:
        self.scripted()
        with patch("api.routers.diagnostics._perception", return_value=_stack("vlm", True)):
            stopped = self._stopped(more_rules=[{**RED, "object": "den roten Würfel"}])
        with patch("api.routers.diagnostics._perception", return_value=_stack("grounded_sam", None)):
            refused = self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 422,
                                   "prompt_not_routable")
        self.assertIn("den roten Würfel", refused["message"])
        self.assertEqual(stopped["id"], self.cell.recovery.run_id)  # type: ignore[union-attr]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
