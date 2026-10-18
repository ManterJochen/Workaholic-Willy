"""A sort through the console keeps each place's bin apart: its picture, its memory, its search (the owner, 2026-10-09).

"Grüne Teile in die gelbe Kiste, rote in die blaue": every place a camera finds is found before the first pick, and the
console keeps what the sort says of each place under that place. Its picture is the overlay of its own place: the
first place a camera finds under ``target`` (``GET /v1/runs/{id}/target/overlay``), as a task of one kind keeps its
bin, each place after it under its own key (``GET /v1/runs/{id}/targets/{n}/overlay``), so the check of one bin never
shows over another. Its bin is remembered by its own phrase and handed to the next sort (``known_targets``); a place
whose bin the sort lost is forgotten, and a task of one kind is handed its one bin as before (``known_target``). A bin
that moved and is found again (``task.target_relocated``) shows its new picture under its place; one found nowhere is a
warning, as the parts no rule claims are (``task.unsorted``). What this file pins:

* a sort on the scripted wrist cell end to end: each bin found with its picture under its place, each part by its rule
  (``task.rule``), each check's picture under its own place, the parts no rule claims a warning, both bins
  remembered, each place's overlay served and a place it does not have refused;
* a bin that moved 150 mm, found again, its picture under its place; a bin found nowhere, a warning with no picture;
* the bins handed to the next sort by phrase, each remembered from what the sort kept, a lost one forgotten; a task of
  one kind handed its one bin as before;
* the hooks: a rule puts the run on its place, a check is known by the label of what it saw, two rules into one bin
  are one place, and a bin found nowhere is a warning;
* a pick of a sort that sees none of its kinds names every one, and a pick of one kind its one label, as before;
* the overlay store's keys: each place's own, its URL, and the count of numbered overlays that leaves them aside.

Honesty bucket (2): the real console, routes, registry and ``run_task`` on the scripted cell of
``tests/_console_task_fakes.py``, its picks saying the kind they went for as a sort's pick loop does; the library's
``run_task`` replaced where only what it was handed is read.
"""

from __future__ import annotations

import dataclasses
import unittest
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

from tests._console_task_fakes import (
    LOOK_1,
    LOOK_2,
    ConsoleArm,
    ConsoleCase,
    ConsoleService,
    Scripted,
    ScriptedCell,
    SightedLocator,
    VirtualClock,
    _toggle,
    events_of,
    wait_for_event,
)
from tests._task_fakes import BIN_CENTRE, BIN_SIZE, bin_object, bin_points

#: Where the blue bin stands: beside the yellow one (``BIN_CENTRE``), clear of it and of the parts.
BLUE_CENTRE = (-300.0, -500.0)
YELLOW_PLACE = {"kind": "camera", "phrase": "yellow bin", "said": "in die gelbe Kiste"}
BLUE_PLACE = {"kind": "camera", "phrase": "blue bin", "said": "in die blaue"}


@dataclass
class SortScripted(Scripted):
    """One scripted pick of a sort, as a sort's pick loop reports it: ``label`` the kind of the part it went for
    (``PickReport.target_label``), ``unclaimed`` what its first look saw that no rule claims (``unclaimed_labels``)."""

    label: str = ""
    unclaimed: tuple[str, ...] = ()


class SortingConsoleService(ConsoleService):
    """The scripted cell's service whose picks carry what a sort's pick loop says (:class:`SortScripted`)."""

    def _play_scripted(self, pick: Scripted) -> Any:
        report = super()._play_scripted(pick)
        label, unclaimed = str(getattr(pick, "label", "")), tuple(getattr(pick, "unclaimed", ()))
        if report.pick_report is None or not (label or unclaimed):
            return report
        said = SimpleNamespace(**vars(report.pick_report), target_label=label, unclaimed_labels=unclaimed)
        return dataclasses.replace(report, pick_report=said)


def _bins(blue_bin: dict[str, Any]) -> Any:
    """What the wrist camera sees, wherever the arm stands: the yellow bin and the blue one, the blue moved
    ``blue_bin["moved"]`` mm along BASE X and of ``blue_bin["size"]``. A class list grounds both, each labelled with its
    phrase; one phrase its own bin."""

    def yellow() -> Any:
        return bin_object(bin_points(BIN_CENTRE), label="yellow bin")

    def blue() -> Any:
        centre = (BLUE_CENTRE[0] + blue_bin["moved"], BLUE_CENTRE[1])
        return bin_object(bin_points(centre, blue_bin["size"]), label="blue bin")

    return {
        "yellow bin | blue bin": lambda _tcp: (yellow(), blue()),
        "blue bin | yellow bin": lambda _tcp: (yellow(), blue()),
        "yellow bin": lambda _tcp: (yellow(),),
        "blue bin": lambda _tcp: (blue(),),
    }


class _SortingCellCase(ConsoleCase):
    def sorting_cell(self, picks: list[Scripted]) -> tuple[ScriptedCell, dict[str, Any]]:
        """A scripted wrist cell whose picks say the kind they went for, and whose camera sees both bins
        (:func:`_bins`); what moves the blue bin, or swaps it for another, is returned beside it."""
        log: list[Any] = []
        arm = ConsoleArm(log)
        jaws = _toggle(log, arm)
        clock = VirtualClock()
        service = SortingConsoleService(arm, jaws, picks, clock=clock, wrist=True, looks=(LOOK_1, LOOK_2))
        cell = ScriptedCell(log=log, arm=arm, jaws=jaws, service=service, clock=clock)
        cell.install(self.cell)
        blue_bin: dict[str, Any] = {"moved": 0.0, "size": BIN_SIZE}
        locator = SightedLocator(_bins(blue_bin), arm=arm, wrist=True)
        for where in ("src.robot.execution.place_target.locators_for_service",
                      "src.robot.execution.task.locators_for_service"):
            patcher = patch(where, return_value=[locator])
            patcher.start()
            self.addCleanup(patcher.stop)
        return cell, blue_bin

    def sort(self, **body: Any) -> dict[str, Any]:
        """Green parts into the yellow bin and red ones into the blue bin, until empty unless ``body`` says otherwise."""
        task = {"object": "green part", "object_said": "grüne Teile", "place": YELLOW_PLACE, "scope": "until_empty",
                "more_rules": [{"object": "red part", "object_said": "rote", "place": BLUE_PLACE}]}
        task.update(body)
        return self.task(**task)

    def png(self, url: str) -> bytes:
        image = self.client.get(url)
        self.assertEqual((200, "image/png"), (image.status_code, image.headers["content-type"]), url)
        self.assertTrue(image.content.startswith(b"\x89PNG"))
        return image.content


class ASortEndToEndTests(_SortingCellCase):
    def test_each_bin_s_picture_stands_under_its_own_place_and_each_part_goes_by_its_rule(self) -> None:
        from api.task_run import remembered_bin

        cell, _moved = self.sorting_cell([
            SortScripted("part", label="green part"), SortScripted("part", label="red part"), Scripted("empty"),
            SortScripted("empty", unclaimed=("orange part", "ambiguous", "orange part"))])

        run = self.sort()
        body = self.finished(run["id"])

        self.assertEqual(("finished", "nothing_left", 2), (body["state"], body["stop_code"], body["parts_placed"]), body)
        self.assertEqual(4, cell.do0_changes(), "each part closes and opens the toggle once, nothing more")
        events = events_of(self.cell, run["id"])
        first, second = (f"/v1/runs/{run['id']}/target/overlay", f"/v1/runs/{run['id']}/targets/1/overlay")
        found = [event.data for event in events if event.type == "task.target_found"]
        self.assertEqual([("yellow bin", first), ("blue bin", second)],
                         [(data["phrase"], data["target"]["overlay"]) for data in found])
        rules = [event for event in events if event.type == "task.rule"]
        self.assertEqual([(0, "target:yellow bin", "info"), (1, "target:blue bin", "info")],
                         [(event.data["rule"], event.data["place"], str(event.severity)) for event in rules])
        checked = [event.data["target"]["overlay"] for event in events if event.type == "task.target_checked"]
        self.assertEqual([first, second], checked, "a check's picture stood under another bin's place")
        unsorted = wait_for_event(self.cell, run["id"], "task.unsorted")
        self.assertEqual(("warn", 3, ["ambiguous", "orange part"]),
                         (str(unsorted.severity), unsorted.data["count"], unsorted.data["labels"]))
        self.assertEqual(self.png(first), self.png(f"/v1/runs/{run['id']}/targets/0/overlay"))
        self.png(second)
        self.refused(self.client.get(f"/v1/runs/{run['id']}/targets/2/overlay"), 404, "no_overlay")
        self.refused(self.client.get("/v1/runs/run-nobody/targets/1/overlay"), 404, "no_such_run")
        yellow, blue = remembered_bin(self.cell, "yellow bin"), remembered_bin(self.cell, "blue bin")
        assert yellow is not None and blue is not None
        self.assertEqual(("yellow bin", "blue bin"), (yellow.label, blue.label))

    def test_a_bin_that_moved_and_is_found_again_shows_its_new_picture_under_its_place(self) -> None:
        cell, blue_bin = self.sorting_cell([SortScripted("part", label="red part",
                                                         then=lambda: blue_bin.update(moved=150.0))])

        run = self.sort(scope="once")
        body = self.finished(run["id"])

        self.assertEqual(("finished", 1), (body["stop_code"], body["parts_placed"]), body)
        relocated = wait_for_event(self.cell, run["id"], "task.target_relocated")
        self.assertEqual(("info", "blue bin", True), (str(relocated.severity), relocated.data["phrase"],
                                                      relocated.data["found"]))
        self.assertAlmostEqual(150.0, relocated.data["moved_mm"], delta=1.0)
        url = relocated.data["target"]["overlay"]
        self.assertEqual(f"/v1/runs/{run['id']}/targets/1/overlay", url)
        self.png(url)
        self.assertNotIn("image_png", relocated.data)
        self.assertEqual(2, cell.do0_changes())

    def test_a_bin_found_nowhere_is_a_warning_with_no_picture(self) -> None:
        """The blue bin moved 180 mm and became one of another size: none of its size is found anywhere."""
        cell, blue_bin = self.sorting_cell([SortScripted(
            "part", label="red part", then=lambda: blue_bin.update(moved=180.0, size=(200.0, 120.0)))])

        run = self.sort(scope="once")
        body = self.finished(run["id"])

        self.assertEqual(("target_lost", 0), (body["stop_code"], body["parts_placed"]), body)
        relocated = wait_for_event(self.cell, run["id"], "task.target_relocated")
        self.assertEqual(("warn", False, None), (str(relocated.severity), relocated.data["found"],
                                                 relocated.data["target"]))
        self.assertEqual(2, cell.do0_changes(), "the part was not put back with the one change that opens the jaws")


class TheBinsOfASortAreRememberedByPhraseTests(_SortingCellCase):
    @staticmethod
    def _kept(phrase: str, centre: tuple[float, float]) -> Any:
        from src.robot.execution.place_target import KeptTarget

        kept = KeptTarget.of(bin_object(bin_points(centre), label=phrase), camera="wrist", look=None, seen_at=1.0,
                             phrase=phrase)
        assert kept is not None
        return kept

    def _handed(self, report: Any) -> tuple[list[dict[str, Any]], Any]:
        handed: list[dict[str, Any]] = []

        def run_task(_service: Any, _plan: Any, **keywords: Any) -> Any:
            handed.append(keywords)
            return report

        return handed, patch("src.robot.execution.task.run_task", side_effect=run_task)

    def test_each_place_is_handed_its_bin_and_each_is_remembered_from_what_the_sort_kept(self) -> None:
        from api.task_run import remembered_bin
        from src.robot.execution.task import TaskReport, TaskStop

        self.sorting_cell([])
        yellow, blue = self._kept("yellow bin", BIN_CENTRE), self._kept("blue bin", BLUE_CENTRE)
        first = TaskReport(stop=TaskStop.NOTHING_LEFT, sentence="sorted", kept_target=yellow,
                           kept_targets={"yellow bin": yellow, "blue bin": blue})
        handed, replaced = self._handed(first)
        with replaced:
            self.finished(self.sort()["id"])
        self.assertEqual({"hooks", "poses", "known_targets"}, set(handed[0]))
        self.assertEqual({}, handed[0]["known_targets"], "a bin nobody kept was handed on")
        self.assertIs(yellow, remembered_bin(self.cell, "yellow bin"))
        self.assertIs(blue, remembered_bin(self.cell, "blue bin"))

        moved_yellow = self._kept("yellow bin", (BIN_CENTRE[0] + 20.0, BIN_CENTRE[1]))
        second = TaskReport(stop=TaskStop.TARGET_LOST, sentence="the blue bin was lost at the drop",
                            kept_target=None, kept_targets={"yellow bin": moved_yellow})
        handed, replaced = self._handed(second)
        with replaced:
            self.finished(self.sort()["id"])
        known = handed[0]["known_targets"]
        self.assertEqual(({"yellow bin", "blue bin"}, True, True),
                         (set(known), known["yellow bin"] is yellow, known["blue bin"] is blue))
        self.assertIs(moved_yellow, remembered_bin(self.cell, "yellow bin"))
        self.assertIsNone(remembered_bin(self.cell, "blue bin"), "the bin the sort lost is still remembered")

    def test_a_task_of_one_kind_is_handed_its_one_bin_as_before(self) -> None:
        from api.task_run import remembered_bin
        from src.robot.execution.task import TaskReport, TaskStop

        self.sorting_cell([])
        blue = self._kept("blue bin", BLUE_CENTRE)
        report = TaskReport(stop=TaskStop.FINISHED, sentence="placed", kept_target=blue,
                            kept_targets={"blue bin": blue})
        handed, replaced = self._handed(report)
        with replaced:
            self.finished(self.task(object="red part", place=BLUE_PLACE)["id"])
            self.finished(self.task(object="red part", place=BLUE_PLACE)["id"])
        self.assertEqual([{"hooks", "poses", "known_target"}] * 2, [set(keywords) for keywords in handed])
        self.assertEqual((None, True), (handed[0]["known_target"], handed[1]["known_target"] is blue))
        self.assertIs(blue, remembered_bin(self.cell, "blue bin"))


class TheHooksSayASortTests(ConsoleCase):
    """What the console's task hooks make of a sort's own events, on a run that carries a sort's plan."""

    def _hooks(self) -> tuple[Any, Any]:
        from api.runs import Run
        from api.task_run import ConsoleTaskHooks

        plan = {"place": {"kind": "pose", "pose": "drop_left"}, "more_rules": [
            {"object": "green part", "place": YELLOW_PLACE}, {"object": "red part", "place": BLUE_PLACE},
            {"object": "blue part", "place": {**BLUE_PLACE, "phrase": "Blue  Bin"}}]}
        run = Run(id="run-sort", prompt="", requested_picks=0, plan=plan)
        return ConsoleTaskHooks(self.cell, run), run

    def _said(self, kind: str) -> Any:
        return [event for event in events_of(self.cell, "run-sort") if event.type == kind][-1]

    def test_a_rule_puts_the_run_on_its_place_and_the_parts_no_rule_took_are_a_warning(self) -> None:
        hooks, run = self._hooks()
        hooks.event("task.rule", said="Part 1 is a red part: it goes into the blue bin.", part=1, rule=2,
                    object="red part", place="target:blue bin", place_label="blue bin")
        self.assertEqual(("place", "info"), (run.step, str(self._said("task.rule").severity)))
        hooks.event("task.unsorted", said="2 part(s) no rule clearly claims stay where they lie.", count=2,
                    labels=["ambiguous"])
        self.assertEqual("warn", str(self._said("task.unsorted").severity))

    def test_each_place_a_camera_finds_keeps_its_picture_under_its_own_key(self) -> None:
        """A pose place first, two rules into one bin (its words spelled twice): the yellow bin is the first place a
        camera finds, the blue one the second. A check names no place and is known by the label of what it saw."""
        hooks, _run = self._hooks()
        target = {"label": "blue bin", "score": 0.8, "centre_mm": [-300.0, -500.0, 120.0], "rim_mm": 120.0}
        hooks.event("task.target_found", said="Found the blue bin.", target=dict(target), phrase="blue bin",
                    image_png=b"\x89PNG blue")
        hooks.event("task.target_found", said="Found the yellow bin.", target={**target, "label": "yellow bin"},
                    phrase="yellow bin", image_png=b"\x89PNG yellow")
        hooks.event("task.target_checked", said="The blue bin stands where it was kept.", target=dict(target),
                    followed=True, by="depth", moved_mm=0.0, image_png=None)

        self.assertEqual(b"\x89PNG yellow", self.cell.overlays.get("run-sort", "target"))
        self.assertEqual(b"\x89PNG blue", self.cell.overlays.get("run-sort", "target1"))
        self.assertEqual("/v1/runs/run-sort/targets/1/overlay", self._said("task.target_checked").data["target"]["overlay"])
        hooks.event("task.target_relocated", said="The blue bin was found nowhere.", phrase="Blue  Bin", found=False,
                    target=None, image_png=None)
        self.assertEqual(("warn", None), (str(self._said("task.target_relocated").severity),
                                          self._said("task.target_relocated").data["target"]))


class APickOfASortSaysTheKindsItWasAskedForTests(unittest.TestCase):
    """A sort's pick asks for every rule's kind at once, so a frame that holds none of them names them all; a pick of
    one kind says its one label as it always did."""

    @staticmethod
    def _said(extra: dict[str, Any]) -> str:
        from api.runs import _sentence
        from src.robot.grasping.loop.progress import PickProgress, PickStage

        _severity, sentence = _sentence(PickProgress(stage=PickStage.NO_CANDIDATE, attempt=0, extra=extra))
        return sentence

    def test_a_sort_names_every_kind_and_one_kind_its_own(self) -> None:
        seen = ["orange part", "ambiguous"]
        self.assertTrue(self._said({"target_label": None, "target_labels": ["green part", "red part"],
                                    "labels_seen": seen}).startswith(
            "Nothing in this frame is called 'green part' or 'red part'. Perception returned 2 object(s)"))
        self.assertTrue(self._said({"target_label": "green part", "labels_seen": seen}).startswith(
            "Nothing in this frame is called 'green part'. Perception returned 2 object(s)"))


class TheStoreKeepsEachPlaceApartTests(unittest.TestCase):
    def test_each_place_has_its_own_key_and_url(self) -> None:
        from api.overlays import overlay_url, target_key

        self.assertEqual(["target", "target1", "target3"], [target_key(n) for n in (0, 1, 3)])
        self.assertEqual(["/v1/runs/run-a/target/overlay", "/v1/runs/run-a/targets/2/overlay",
                          "/v1/runs/run-a/overlays/2", "/v1/runs/run-a/overlays/target0"],
                         [overlay_url("run-a", key) for key in ("target", "target2", 2, "target0")])

    def test_the_count_of_numbered_overlays_leaves_every_place_aside(self) -> None:
        from api.overlays import OverlayStore

        store = OverlayStore()
        for key in (1, 2, "target", "target1", "target2"):
            self.assertIsNotNone(store.put("run-a", key, b"\x89PNG " + str(key).encode()))
        self.assertEqual(2, store.count("run-a"))
        self.assertEqual("/v1/runs/run-a/targets/2/overlay", store.put("run-a", "target2", b"\x89PNG again"))
        self.assertEqual(b"\x89PNG again", store.get("run-a", "target2"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
