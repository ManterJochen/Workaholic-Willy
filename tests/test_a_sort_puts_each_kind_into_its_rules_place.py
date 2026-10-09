"""A sort puts each kind of part into its rule's place (the owner, 2026-10-09).

"Gruene Teile in die gelbe Kiste, rote in die blaue": a plan of several rules (``TaskPlan.more_rules``, up to three
``SortRule`` beside its own first rule) sends each kind of part to its rule's place, and two rules may share one place.
Every place a camera finds is found before the first pick, with one locate of a class list per look
(``place_target.survey_places``); a place no look saw stops the task before it picks anything, and the task names each
place it is missing. Each pick grounds every rule's kind in one call ("each separate green part | each separate red
part"), and the part it gripped goes by the rule of the kind it went for (``PickReport.target_label``), said as
``task.rule``. A gripped part no rule names goes back where it was gripped, and the task asks. A part no rule clearly
claims stays where it lies, and the end names what the last look saw of them (``task.unsorted``). A sort runs only on a
cell whose perception grounds a phrase, follows no parts from pick to pick, and judges no carry ahead of its pick, as
its place is known only once the pick says which kind it gripped. A task of one kind says no ``task.rule`` and no
``task.unsorted``.

The camera is ``tests._task_fakes.ClassListLocator``, a wrist camera over a bench it ray casts, which grounds a class list
in one locate and labels each bin with its phrase; the picks are ``SortPick``, which say the kind they went for.
"""

from __future__ import annotations

import json
import unittest
from contextlib import nullcontext
from typing import Any
from unittest.mock import patch

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    PLACE_TCP,
    BinScene,
    ClassListLocator,
    RecordingHooks,
    SeenBin,
    SortPick,
    TaskArm,
    do0_changes,
    motions,
    run_sort,
)

#: Above the yellow bin, above the blue one, and above an empty spot of the bench.
LY = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LB = JointPositions.deg(40.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LE = JointPositions.deg(70.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LOOKS = (LY, LB, LE)
AT_LY = Pose.tool_down(BIN_CENTRE[0], BIN_CENTRE[1], 450.0, label="LY")
AT_LB = Pose.tool_down(-300.0, -500.0, 450.0, label="LB")
AT_LE = Pose.tool_down(-300.0, -50.0, 450.0, label="LE")
BLUE_CENTRE = (-300.0, -500.0)


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _scene(*, yellow: bool = True, blue: bool = True) -> BinScene:
    """The yellow bin where only LY sees it, and the blue one where only LB sees it."""
    bins = ([SeenBin()] if yellow else []) + ([SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE)]
                                             if blue else [])
    return BinScene(bins=bins)


def _plan(*more: Any, place: Any = None, scope: str = "until_empty", **keywords: Any) -> Any:
    """Green parts into the yellow bin, and the further rules ``more`` (red parts into the blue bin where none given)."""
    from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

    rules = more or (SortRule(object="red part", place=PlaceAt(camera="blue bin")),)
    return TaskPlan(object="green part", place=place if place is not None else PlaceAt(camera="yellow bin"),
                    scope=scope, more_rules=rules, **keywords)


def _sort(picks: Any, *, plan: Any = None, scene: "BinScene | None" = None, hooks: "RecordingHooks | None" = None,
          **keywords: Any) -> Any:
    """A sort on a wrist camera over ``scene`` (both bins where none is given), every pick from the yellow bin's look."""
    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE})
    scene = scene if scene is not None else _scene()
    locator = ClassListLocator(scene, arm=arm, log=log)
    ran = run_sort(plan if plan is not None else _plan(), picks, arm=arm, locators=[locator], wrist=True, looks=LOOKS,
                   hooks=hooks, **keywords)
    ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
    return ran


def _line_ins(ran: Any) -> list[Any]:
    """Where each place's line in went down (the second move after each ``task.place_started``)."""
    marks = [index for index, entry in enumerate(ran.log) if entry == ("event", "task.place_started")]
    return [[m for m in motions(ran.log[mark:]) if m[0] == "move"][1] for mark in marks]


class ThePlanTests(unittest.TestCase):
    def test_the_further_rules_are_the_command_readers_bound(self) -> None:
        from src.models.vlm.command import MAX_FURTHER_RULES as READER
        from src.robot.execution.task import MAX_FURTHER_RULES

        self.assertEqual(READER, MAX_FURTHER_RULES)
        self.assertEqual(3, MAX_FURTHER_RULES)

    def test_every_rule_its_own_first_and_a_list_is_kept_as_a_tuple(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

        red = SortRule(object="red part", place=PlaceAt(camera="blue bin"), source="on the black mat")
        plan = TaskPlan(object="green part", place=PlaceAt(camera="yellow bin"), which="", more_rules=[red])

        self.assertEqual((red,), plan.more_rules)
        self.assertIsInstance(plan.more_rules, tuple)
        self.assertEqual((SortRule("green part", PlaceAt(camera="yellow bin")), red), plan.rules)
        one = TaskPlan(object="", place=PlaceAt(pose="drop_left"))
        self.assertEqual((SortRule("", PlaceAt(pose="drop_left")),), one.rules, "a task of one kind has one rule")

    def test_a_sort_that_cannot_be_sorted_is_refused(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

        blue, yellow = PlaceAt(camera="blue bin"), PlaceAt(camera="yellow bin")
        refused = {
            "one kind twice": ("green part", yellow, [SortRule(" Green  Part", blue)]),
            "an empty kind of its own": ("", yellow, [SortRule("red part", blue)]),
            "an empty further kind": ("green part", yellow, [SortRule("  ", blue)]),
            "a bar in a place": ("green part", yellow, [SortRule("red part", PlaceAt(camera="blue | red bin"))]),
            "a bar in a kind": ("green part", yellow, [SortRule("red | blue part", blue)]),
            "the word ambiguous": ("green part", PlaceAt(camera="ambiguous bin"), [SortRule("red part", blue)]),
            "four further rules": ("green part", yellow, [SortRule(f"part {n}", blue) for n in range(4)]),
            "one phrase grounded twice": ("green part", yellow, [SortRule("red part", blue, which="the cube on top"),
                                                                 SortRule("blue part", blue, which="the cube on top")]),
        }
        for name, (kind, place, more) in refused.items():
            with self.subTest(name), self.assertRaises(ValueError):
                TaskPlan(object=kind, place=place, more_rules=more)
        TaskPlan(object="green part", place=yellow, more_rules=[SortRule(f"part {n}", blue) for n in range(3)])

    def test_a_rule_that_is_no_sort_rule_is_a_programmers_error(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

        for more in (["red part"], (PlaceAt(camera="blue bin"),), SortRule("red part", PlaceAt(camera="blue bin"))):
            with self.subTest(more=more), self.assertRaises(TypeError):
                TaskPlan(object="green part", place=PlaceAt(camera="yellow bin"), more_rules=more)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            SortRule("red part", "blue bin")  # type: ignore[arg-type]

    def test_a_task_of_one_kind_keeps_todays_words(self) -> None:
        """A plan of one rule is refused nothing new: its phrases are grounded alone, as before."""
        from src.robot.execution.task import PlaceAt, TaskPlan

        TaskPlan(object="red | blue part", place=PlaceAt(camera="ambiguous bin"))

    def test_the_plans_refusals_are_the_class_lists_own(self) -> None:
        from src.models.vlm.qwen import class_list_prompt
        from src.robot.execution.task import _class_list_refusal

        for text in ("blue bin", "blue | red", "an Ambiguous part", "ambiguously red", "part|bin", "unambiguous bin"):
            with self.subTest(text=text):
                try:
                    class_list_prompt([text, "green part"])
                    refused = False
                except ValueError:
                    refused = True
                self.assertEqual(refused, bool(_class_list_refusal(text, "the phrase")))

    def test_the_sort_names_are_handed_out(self) -> None:
        import willy
        from src.robot import execution
        from src.robot.execution import task

        for name in ("SortRule", "MAX_FURTHER_RULES"):
            with self.subTest(name):
                self.assertIs(getattr(task, name), getattr(willy, name))
                self.assertIs(getattr(task, name), getattr(execution, name))
                self.assertIn(name, task.__all__)


class TwoBinsTests(unittest.TestCase):
    def test_each_kind_goes_into_its_rules_bin_and_both_bins_stay_kept_out(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _sort([SortPick("part", label="green part"), SortPick("part", label="red part"), "empty", "empty"])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        before_the_pick = ran.log[:ran.log.index(("event", "task.part_started"))]
        self.assertEqual([("locate", "yellow bin | blue bin"), ("locate", "yellow bin | blue bin")],
                         [entry for entry in before_the_pick if entry[0] == "locate"],
                         "every bin, one locate per look, before the first pick")
        self.assertEqual(["yellow bin", "blue bin"], [found["phrase"] for found in ran.hooks.of("task.target_found")])
        survey = ran.hooks.of("task.survey_started")[0]
        self.assertEqual(("yellow bin", ["yellow bin", "blue bin"]), (survey["phrase"], survey["phrases"]))
        self.assertIn("the yellow bin and the blue bin", ran.hooks.said[ran.names().index("task.survey_started")])
        rules = ran.hooks.of("task.rule")
        self.assertEqual([{"part": 1, "rule": 0, "object": "green part", "place": "target:yellow bin",
                           "place_label": "yellow bin"},
                          {"part": 2, "rule": 1, "object": "red part", "place": "target:blue bin",
                           "place_label": "blue bin"}], rules)
        said = [ran.hooks.said[index] for index, name in enumerate(ran.names()) if name == "task.rule"]
        self.assertEqual(["Part 1 is a green part: it goes into the yellow bin.",
                          "Part 2 is a red part: it goes into the blue bin."], said)
        self.assertEqual(["target:yellow bin", "target:blue bin"],
                         [placed["place"] for placed in ran.hooks.of("task.place_started")])
        yellow, blue = _line_ins(ran)
        np.testing.assert_allclose(BIN_CENTRE, yellow[1][:2], atol=10.0)
        np.testing.assert_allclose(BLUE_CENTRE, blue[1][:2], atol=10.0)
        self.assertEqual([look_label(LY), look_label(LB)], [c["to_look"] for c in ran.hooks.of("task.carry_started")])
        for regions in ran.service.zones_seen:
            self.assertEqual(["the blue bin it places into", "the yellow bin it places into"],
                             sorted(region.reason for region in regions), "a check forgot another bin's region")
        self.assertEqual(4, do0_changes(ran.log))
        self.assertNotIn("task.unsorted", ran.names())
        self.assertIn("'green part' or 'red part'", ran.report.sentence)

    def test_the_picks_ground_every_rules_kind_in_one_call(self) -> None:
        from src.robot.execution.autonomous_grasp.prompt import PickPrompt

        ran = _sort(["empty", "empty"])

        prompt = ran.service.prompts[0]
        self.assertEqual("each separate green part | each separate red part", prompt.phrase)
        self.assertEqual((None, ("green part", "red part")), (prompt.target_label, prompt.object_labels))
        if "target_labels" in getattr(PickPrompt, "__dataclass_fields__", {}):
            self.assertEqual(("green part", "red part"), prompt.target_labels)
        self.assertEqual(PickPrompt(phrase="object"), ran.service.prompt, "the cell's prompt was not put back")

    def test_a_bin_no_look_saw_stops_the_sort_before_its_first_pick_and_is_named(self) -> None:
        from src.robot.execution.task import TaskStop

        for scene, missing in ((_scene(blue=False), ["blue bin"]),
                               (_scene(yellow=False, blue=False), ["yellow bin", "blue bin"])):
            with self.subTest(missing=missing):
                ran = _sort([], scene=scene)

                self.assertIs(TaskStop.TARGET_NOT_FOUND, ran.report.stop, ran.report.sentence)
                self.assertEqual([], ran.service.calls, "a part was picked for a sort missing a bin")
                self.assertEqual(0, do0_changes(ran.log))
                self.assertEqual(missing, [event["phrase"] for event in ran.hooks.of("task.target_missing")])
                self.assertIn(f"no {' and no '.join(missing)} was seen from", ran.report.sentence)
                self.assertIn("nothing was picked", ran.report.sentence)
                self.assertNotIn("task.target_found", ran.names())
                self.assertEqual(("home",), motions(ran.log)[-1])
                self.assertEqual({}, dict(ran.report.kept_targets), "a sort that picked nothing kept a bin")

    def test_a_camera_bin_and_a_taught_pose(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule, TaskStop

        plan = _plan(SortRule(object="red part", place=PlaceAt(pose="drop_left")))
        ran = _sort([SortPick("part", label="red part"), SortPick("part", label="green part"), "empty", "empty"],
                    plan=plan)

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(["yellow bin"], [entry[1] for entry in ran.log[:ran.log.index(("event", "task.part_started"))]
                                          if entry[0] == "locate"], "one camera place is surveyed as before")
        self.assertEqual(["pose:drop_left", "target:yellow bin"],
                         [placed["place"] for placed in ran.hooks.of("task.place_started")])
        said = [ran.hooks.said[index] for index, name in enumerate(ran.names()) if name == "task.rule"]
        self.assertEqual("Part 1 is a red part: it goes to pose 'drop_left'.", said[0])
        self.assertEqual({"part": 1, "rule": 1, "object": "red part", "place": "pose:drop_left",
                          "place_label": "drop_left"}, ran.hooks.of("task.rule")[0])
        self.assertEqual("place", ran.hooks.of("task.pose_screened")[0]["role"])
        regions = ran.service.zones_seen[-1]
        self.assertEqual(["the drop at pose 'drop_left'", "the yellow bin it places into"],
                         [region.reason for region in regions])
        self.assertTrue(regions[0].contains((float(PLACE_TCP.position_mm[0]), float(PLACE_TCP.position_mm[1]), 0.0)))
        self.assertEqual({"yellow bin"}, set(ran.report.kept_targets))

    def test_two_rules_that_share_one_bin(self) -> None:
        from src.robot.execution.task import PlaceAt, SortRule, TaskStop

        plan = _plan(SortRule(object="red part", place=PlaceAt(camera="Yellow  Bin")))
        ran = _sort([SortPick("part", label="red part"), SortPick("part", label="green part"), "empty", "empty"],
                    plan=plan)

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(["yellow bin"], [entry[1] for entry in ran.log[:ran.log.index(("event", "task.part_started"))]
                                          if entry[0] == "locate"])
        self.assertEqual(1, len(ran.hooks.of("task.target_found")))
        self.assertEqual([(1, "target:yellow bin"), (0, "target:yellow bin")],
                         [(rule["rule"], rule["place"]) for rule in ran.hooks.of("task.rule")])
        self.assertEqual(1, len(ran.service.zones_seen[-1]), "one bin is one region")
        self.assertEqual(2, ran.report.parts_placed)


class TheRuleOfEachPartTests(unittest.TestCase):
    def test_a_gripped_part_no_rule_names_goes_back_and_the_task_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        for label in ("orange part", ""):
            with self.subTest(label=label):
                ran = _sort([SortPick("part", label=label)])

                self.assertIs(TaskStop.TARGET_NOT_FOUND, ran.report.stop, ran.report.sentence)
                self.assertIn("no rule of the sort names its kind", ran.report.sentence)
                self.assertIn("the part was put back where it was gripped", ran.report.sentence)
                self.assertNotIn("task.rule", ran.names())
                self.assertNotIn("task.carry_started", ran.names())
                self.assertEqual(["task.put_back", "task.return_started", "task.returned", "task.part_finished"],
                                 ran.names()[-4:])
                self.assertEqual(2, do0_changes(ran.log), "the part was not let go where it was gripped")
                self.assertEqual({"yellow bin", "blue bin"}, set(ran.report.kept_targets),
                                 "the bins stood where they were kept")

    def test_a_kind_is_read_with_case_and_blanks_aside(self) -> None:
        ran = _sort([SortPick("part", label=" Red   Part"), "empty", "empty"])

        self.assertEqual((1, "target:blue bin"), (ran.hooks.of("task.rule")[0]["rule"],
                                                  ran.hooks.of("task.rule")[0]["place"]))

    def test_the_parts_no_rule_clearly_claims_are_named_at_the_end(self) -> None:
        from src.robot.execution.task import TaskStop

        seen = ("ambiguous", "orange part", "ambiguous")
        ran = _sort([SortPick("part", label="green part"), SortPick("empty", unclaimed=seen),
                     SortPick("empty", unclaimed=seen)])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual([{"count": 3, "labels": ["ambiguous", "orange part"]}], ran.hooks.of("task.unsorted"))
        self.assertIn("; 3 part(s) no rule clearly claims stay where they lie (ambiguous, orange part)",
                      ran.report.sentence)
        self.assertEqual(3, ran.report.unsorted)
        self.assertEqual(3, ran.report.to_dict()["unsorted"])
        self.assertEqual(["task.nothing_found", "task.unsorted", "task.return_started", "task.returned"],
                         ran.names()[-4:])

    def test_a_task_of_one_kind_says_no_rule_and_names_no_unsorted_part(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan, TaskStop

        plan = TaskPlan(object="green part", place=PlaceAt(camera="yellow bin"), scope="until_empty")
        ran = _sort([SortPick("part", label="red part"), SortPick("empty", unclaimed=("ambiguous",)), "empty"],
                    plan=plan, scene=_scene(blue=False))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed, "a task of one kind places what its pick gripped")
        self.assertNotIn("task.rule", ran.names())
        self.assertNotIn("task.unsorted", ran.names())
        self.assertEqual(0, ran.report.unsorted)
        prompt = ran.service.prompts[0]
        self.assertEqual(("each separate green part", "green part"), (prompt.phrase, prompt.target_label))


class ASortsCellTests(unittest.TestCase):
    def test_a_sort_on_a_cell_that_grounds_no_phrase_is_refused_before_anything_moves(self) -> None:
        from src.robot.execution.task import TaskRefused

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE})
        with self.assertRaises(TaskRefused) as refused:
            run_sort(_plan(), ["part"], arm=arm, locators=[ClassListLocator(_scene(), arm=arm, log=log)],
                     wrist=True, looks=LOOKS, grounds=False)

        self.assertEqual("bad_request", refused.exception.code)
        self.assertIn("a sort tells its kinds apart by the detector's words", str(refused.exception))
        self.assertEqual([], motions(log))
        self.assertEqual(0, do0_changes(log))

    def test_a_sort_follows_no_parts_and_judges_no_carry_ahead_of_its_pick(self) -> None:
        from types import SimpleNamespace

        from src.config.schema.robot.place_schema import RobotPlaceConfig

        declared: list[Any] = []

        def declaring(_arm: Any, joints: Any) -> Any:
            declared.append(joints)
            return nullcontext()

        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE})
        arm.config = SimpleNamespace(  # type: ignore[attr-defined]
            place=RobotPlaceConfig(), grasping=SimpleNamespace(follow_parts=SimpleNamespace(
                enabled=True, refresh_every_picks=0, max_shift_mm=10.0, max_creep_mm=20.0)))
        with patch("src.robot.execution.motion.expecting_next", side_effect=declaring), \
                self.assertLogs("src.robot.execution.task", "INFO") as said:
            ran = run_sort(_plan(), [SortPick("part", label="red part"), "empty", "empty"], arm=arm,
                           locators=[ClassListLocator(_scene(), arm=arm, log=log)], wrist=True, looks=LOOKS)

        self.assertEqual(1, ran.report.parts_placed, ran.report.sentence)
        self.assertTrue(all("follow_looks" not in call and "follow" not in call for call in ran.service.calls))
        self.assertTrue(any("a sort follows no parts from pick to pick" in line for line in said.output))
        self.assertEqual([None, "home", None, None], declared, "a pick judged a carry ahead, or a place no return")


class WhatASortReportsTests(unittest.TestCase):
    def test_the_report_keeps_every_bin_by_its_phrase_and_the_next_sort_looks_at_them_first(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = _scene()
        first = _sort([SortPick("part", label="green part"), "empty", "empty"], scene=scene)
        kept = first.report.kept_targets
        self.assertEqual({"yellow bin", "blue bin"}, set(kept))
        self.assertIs(first.report.kept_target, kept["yellow bin"], "the first camera place's bin, as before")
        as_data = first.report.to_dict()
        json.dumps(as_data)
        self.assertEqual("blue bin", as_data["kept_targets"]["blue bin"]["label"])

        again = _sort([SortPick("part", label="red part"), "empty", "empty"], scene=scene, known_targets=kept)

        self.assertIs(TaskStop.NOTHING_LEFT, again.report.stop, again.report.sentence)
        survey = again.log[:again.log.index(("event", "task.part_started"))]
        self.assertEqual([], [entry for entry in survey if entry[0] == "locate"], "the detector was asked for bins "
                                                                                 "that stood still")
        self.assertEqual([(True, "depth"), (True, "depth")],
                         [(found["known"], found["by"]) for found in again.hooks.of("task.target_found")])

    def test_a_known_bin_that_is_no_kept_bin_is_a_programmers_error(self) -> None:
        for known in ({"yellow bin": "here"}, ["not a map"]):
            with self.subTest(known=known), self.assertRaises(TypeError):
                _sort([], known_targets=known)

    def test_a_task_of_one_kind_keeps_its_bin_by_its_phrase_too(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskPlan

        ran = _sort([SortPick("part")], plan=TaskPlan(object="green part", place=PlaceAt(camera="yellow bin")),
                    scene=_scene(blue=False))

        self.assertEqual({"yellow bin": ran.report.kept_target}, dict(ran.report.kept_targets))
        self.assertEqual(0, ran.report.to_dict()["unsorted"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
