"""A bin that moved is found again before the drop, and the drop is planned anew over it (the owner, 2026-10-09).

"wenn sie dies nicht mehr tut, dann kann er seine Ablage nochmal neu errechnen": before every drop the task checks its
bin again, as before (its rim's depth and colour, the detector where that is unsure). Where the check loses it (not
seen, moved too far, another size where it stood), the task looks for it again (``place_target.relocate``). The hold of
the bin's look ends first; then the task drives its looks once more with the part in its jaws, each through the arm's
judged verb. A bin of the size the survey found and the colour the task followed, standing in no other place, is kept in
place of the old one (``task.target_relocated``). The part is carried to the look it was found from, the bin is checked
there as every bin is, and the drop is planned anew over it. Once per part: a bin lost again, or found nowhere, puts the
part back where it was gripped and the task asks, as before. A fixed camera's bin is looked for where the camera stands,
before the pick, and nothing moves. ``robot.place.relocate`` off keeps the old rule. In a sort, every look at one bin
grounds every bin (``together``), so the other rule's bin is never taken for this one.

The camera is ``tests._task_fakes.MeasuringLocator`` (one bin) or ``ClassListLocator`` (a sort), a wrist camera over a
bench it ray casts, or a ``ScriptedLocator`` that stands fixed.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.core.errors import CameraWorldUnavailable
from src.robot.execution.looks import look_label, move_to_look
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    BinScene,
    ClassListLocator,
    MeasuringLocator,
    OnePhraseLocator,
    Pick,
    RecordingHooks,
    ScriptedLocator,
    SeenBin,
    SortPick,
    TaskArm,
    bin_object,
    bin_points,
    do0_changes,
    motions,
    placing,
    run,
    run_sort,
)

#: Above the yellow bin, above the blue one, and above an empty spot of the bench.
LY = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LB = JointPositions.deg(40.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LE = JointPositions.deg(70.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_LY = Pose.tool_down(BIN_CENTRE[0], BIN_CENTRE[1], 450.0, label="LY")
AT_LB = Pose.tool_down(-300.0, -500.0, 450.0, label="LB")
AT_LE = Pose.tool_down(-300.0, -50.0, 450.0, label="LE")
BLUE_CENTRE = (-300.0, -500.0)
#: Where only LE sees a bin.
EMPTY_SPOT = (-300.0, -50.0)
#: Where two bins stand that LY sees whole: the yellow one 140 mm from where it was kept, the blue one 70 mm from it.
YELLOW_NOW = (BIN_CENTRE[0], BIN_CENTRE[1] - 140.0)
BLUE_NOW = (BIN_CENTRE[0], BIN_CENTRE[1] + 70.0)


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _arm(log: list[Any], **keywords: Any) -> TaskArm:
    return TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE}, **keywords)


def _task(picks: Any, *, scene: BinScene, arm: "TaskArm | None" = None, hooks: "RecordingHooks | None" = None,
          locator: Any = None, **keywords: Any) -> Any:
    """A task into the yellow bin on a wrist camera over ``scene``, the bin seen from LY, the bench's empty spot from LE."""
    from src.robot.execution.task import PlaceAt

    log: list[Any] = arm.log if arm is not None else []
    arm = arm if arm is not None else _arm(log)
    locator = locator if locator is not None else MeasuringLocator(scene, arm=arm, log=log)
    ran = run(picks, place=PlaceAt(camera="yellow bin"), arm=arm, wrist=True, looks=(LY, LE), locators=[locator],
              hooks=hooks, **keywords)
    ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
    return ran


def _moved_to(scene: BinScene, centre: "tuple[float, float]") -> None:
    scene.bins[0] = replace(scene.bins[0], centre_xy=centre)


def _line_ins(ran: Any) -> list[Any]:
    marks = [index for index, entry in enumerate(ran.log) if entry == ("event", "task.place_started")]
    return [[m for m in motions(ran.log[mark:]) if m[0] == "move"][1] for mark in marks]


def _between(ran: Any, first: str, then: str) -> list[Any]:
    """The log between the first ``first`` event and the first ``then`` event after it."""
    start = ran.log.index(("event", first))
    return ran.log[start:ran.log.index(("event", then), start)]


class ATaskOfOneBinTests(unittest.TestCase):
    def test_a_bin_moved_past_the_bound_is_found_where_the_check_saw_it_and_the_drop_goes_there(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=lambda: scene.move(150.0)),), scene=scene)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(["task.survey_started", "task.target_found", "task.part_started", "task.carry_started",
                          "task.target_checked", "task.target_relocated", "task.carry_started", "task.target_checked",
                          "task.drop_planned", "task.place_started", "task.placed", "task.return_started",
                          "task.returned", "task.part_finished"], ran.names())
        checks = ran.hooks.of("task.target_checked")
        self.assertEqual([(False, "detector"), (True, "depth")], [(c["followed"], c["by"]) for c in checks])
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertEqual((True, "check", look_label(LY), "yellow bin"),
                         (found["found"], found["by"], found["look"], found["phrase"]))
        self.assertAlmostEqual(150.0, found["moved_mm"], delta=3.0)
        np.testing.assert_allclose((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]), _line_ins(ran)[0][1][:2], atol=10.0)
        self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked, "the search asked the detector again")
        self.assertEqual([], [m for m in motions(_between(ran, "task.target_relocated", "task.place_started"))],
                         "the arm was driven to the look it stood at")
        self.assertLess(ran.log.index(("forget",)), ran.log.index(("event", "task.target_relocated")),
                        "the hold of the bin's look outlived the loss")
        self.assertEqual(2, ran.log.count(("hold",)), "the frames of the look the bin was found from were not held")
        self.assertEqual(2, do0_changes(ran.log))
        kept = ran.report.kept_target
        assert kept is not None
        np.testing.assert_allclose((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]), kept.centre_xy_mm, atol=3.0)
        self.assertIs(kept, ran.report.kept_targets["yellow bin"])
        self.assertIn("Found the yellow bin again where the check saw it",
                      ran.hooks.said[ran.names().index("task.target_relocated")])

    def test_a_bin_moved_out_of_its_looks_view_is_found_from_the_look_the_task_drove_to(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=lambda: _moved_to(scene, EMPTY_SPOT)),), scene=scene)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertEqual((True, "detector", look_label(LE), [look_label(LE)]),
                         (found["found"], found["by"], found["look"], found["looks_tried"]))
        self.assertEqual([("joints", _key(LE))], motions(_between(ran, "task.target_checked", "task.target_relocated")),
                         "the search drove anywhere but the one look left")
        self.assertEqual([look_label(LY), look_label(LE)], [c["to_look"] for c in ran.hooks.of("task.carry_started")])
        np.testing.assert_allclose(EMPTY_SPOT, _line_ins(ran)[0][1][:2], atol=10.0)
        self.assertEqual(2, do0_changes(ran.log))

    def test_a_bin_taken_away_is_found_nowhere_the_part_goes_back_and_the_task_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=scene.take_away),), scene=scene)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual(["task.target_relocated", "task.target_lost", "task.put_back", "task.return_started",
                          "task.returned", "task.part_finished"], ran.names()[-6:])
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertEqual((False, None, [look_label(LE)]), (found["found"], found["target"], found["looks_tried"]))
        self.assertEqual([{"look": look_label(LY), "why": "not_seen", "phrase": "yellow bin"}],
                         ran.hooks.of("task.target_lost"))
        self.assertIn("found nowhere", ran.hooks.said[ran.names().index("task.target_lost")])
        self.assertIn("found nowhere", ran.report.sentence)
        self.assertEqual(2, do0_changes(ran.log), "the part was not let go where it was gripped")
        self.assertEqual((None, {}), (ran.report.kept_target, dict(ran.report.kept_targets)))

    def test_a_bin_lost_again_after_it_was_found_puts_the_part_back(self) -> None:
        """One search per part: the bin found again is gone at its second check, and the part goes back."""
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        hooks = RecordingHooks(on_event=lambda name, _d: scene.take_away() if name == "task.target_relocated" else None)
        ran = _task((Pick("part", then=lambda: scene.move(150.0)),), scene=scene, hooks=hooks)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.names().count("task.target_relocated"))
        self.assertEqual(["task.target_checked", "task.target_lost", "task.put_back"],
                         [name for name in ran.names() if name.startswith("task.")][-6:-3])
        self.assertNotIn("found nowhere", ran.report.sentence)
        self.assertEqual(2, do0_changes(ran.log))

    def test_a_cell_that_keeps_the_old_rule_puts_the_part_back(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        arm = placing(_arm([]), relocate=False)
        ran = _task((Pick("part", then=lambda: scene.move(150.0)),), scene=scene, arm=arm)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertNotIn("task.target_relocated", ran.names())
        self.assertEqual([{"look": look_label(LY), "why": "moved_too_far", "phrase": "yellow bin"}],
                         ran.hooks.of("task.target_lost"))
        self.assertEqual(["task.target_lost", "task.put_back", "task.return_started", "task.returned",
                          "task.part_finished"], ran.names()[-5:])
        self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked)


class WhatEndsASearchTests(unittest.TestCase):
    """The search's looks are the task's own motions, the part in its jaws: what ends a carry ends them."""

    def _searched(self, refuse: Any = None, locator: Any = None) -> Any:
        scene = BinScene()
        log: list[Any] = []
        arm = _arm(log, refuse=refuse)
        return _task((Pick("part", then=lambda: _moved_to(scene, EMPTY_SPOT)),), scene=scene, arm=arm,
                     locator=locator(scene, arm, log) if locator is not None else None)

    def test_a_look_refused_before_anything_was_sent_is_skipped_and_said(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = self._searched(refuse=lambda kind, index, target: MotionStatus.WORKSPACE_REJECTED
                             if kind == "joints" and _key(target) == _key(LE) else None)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertFalse(found["found"])
        self.assertEqual(look_label(LE), found["refused"][0][0])
        self.assertEqual(2, do0_changes(ran.log))

    def test_a_look_motion_that_failed_once_sent_ends_the_task_where_the_arm_stands(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = self._searched(refuse=lambda kind, index, target: MotionStatus.CONTROLLER_REJECTED
                             if kind == "joints" and _key(target) == _key(LE) else None)

        self.assertIs(TaskStop.PART_STILL_HELD, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(ran.log))
        self.assertNotIn("task.target_relocated", ran.names())
        self.assertNotIn("task.put_back", ran.names())

    def test_a_camera_that_could_not_vouch_on_a_search_look_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        def blind(kind: str, index: int, target: Any) -> "MotionStatus | None":
            if kind == "joints" and _key(target) == _key(LE):
                raise CameraWorldUnavailable(camera="wrist", verdict="stale", attempts=3,
                                             reason="every frame was older than the cell allows")
            return None

        ran = self._searched(refuse=blind)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertNotIn("task.put_back", ran.names())

    def test_a_locate_that_raises_in_the_search_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        class _Raising(MeasuringLocator):
            def locate(self, prompt: str) -> Any:
                if len(self.asked) == 2:  # the survey's, the check's, then the search's
                    self.asked.append(prompt)
                    raise RuntimeError("the camera stopped delivering")
                return super().locate(prompt)

        ran = self._searched(locator=lambda scene, arm, log: _Raising(scene, arm=arm, log=log))

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertIn("could not look for the yellow bin again", ran.report.sentence)
        self.assertTrue(ran.report.holding)

    def test_a_halt_before_a_search_look_ends_it_with_nothing_sent(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        hooks = RecordingHooks()
        hooks.on_event = lambda name, data: setattr(hooks, "halt", True) if (
            name == "task.target_checked" and data.get("followed") is False) else None
        ran = _task((Pick("part", then=lambda: _moved_to(scene, EMPTY_SPOT)),), scene=scene, hooks=hooks)

        self.assertIs(TaskStop.HALTED, ran.report.stop, ran.report.sentence)
        self.assertEqual([], motions(ran.after("task.target_checked")))
        self.assertTrue(ran.report.holding)


class AFixedCamerasBinTests(unittest.TestCase):
    def _task(self, after_part_1: Any, picks: Any) -> Any:
        from src.robot.execution.task import PlaceAt

        seen = [bin_object()]
        log: list[Any] = []
        arm = TaskArm(log)
        locator = ScriptedLocator({"blue bin": lambda _tcp: list(seen)}, wrist=False, rig_id="overhead", arm=arm,
                                  log=log)
        done: list[int] = []

        def changed(name: str, _data: Any) -> None:
            if name == "task.part_finished" and not done:
                done.append(1)
                seen[:] = after_part_1()

        ran = run(picks, place=PlaceAt(camera="blue bin"), scope="until_empty", arm=arm, wrist=False,
                  locators=[locator], hooks=RecordingHooks(on_event=changed))
        ran.locator = locator  # type: ignore[attr-defined]
        return ran

    def test_a_bin_moved_between_two_parts_is_found_where_the_camera_stands_and_nothing_moves(self) -> None:
        from src.robot.execution.task import TaskStop

        moved = (BIN_CENTRE[0] + 200.0, BIN_CENTRE[1])
        ran = self._task(lambda: [bin_object(bin_points(moved))], ("part", "part", "empty", "empty"))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertEqual((True, "check", None), (found["found"], found["by"], found["look"]))
        between = _between(ran, "task.part_finished", "task.part_started")
        self.assertEqual([], [m for m in motions(between) if m[0] != "home"], "the search moved the arm")
        self.assertEqual(1, sum(1 for entry in between if entry[0] == "locate"), "the search located again")
        np.testing.assert_allclose(moved, _line_ins(ran)[1][1][:2], atol=10.0)

    def test_a_bin_taken_away_between_two_parts_ends_the_task_before_the_next_pick(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = self._task(lambda: [], ("part", "part"))

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.picks)
        self.assertEqual(["task.part_finished", "task.target_checked", "task.target_relocated", "task.target_lost"],
                         ran.names()[-4:], "the arm stood home: nothing more was sent")
        self.assertIn("found nowhere", ran.report.sentence)
        self.assertEqual(2, do0_changes(ran.log))


class ASortTests(unittest.TestCase):
    """Two rules into two bins: green parts into the yellow bin, red ones into the blue."""

    def _sort(self, picks: Any, *, locator_type: Any = ClassListLocator, hooks: Any = None) -> Any:
        from src.robot.execution.task import PlaceAt, SortRule, TaskPlan

        scene = BinScene(bins=[SeenBin(), SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE)])
        log: list[Any] = []
        arm = _arm(log)
        locator = locator_type(scene, arm=arm, log=log)
        plan = TaskPlan(object="green part", place=PlaceAt(camera="yellow bin"), scope="until_empty",
                        more_rules=(SortRule(object="red part", place=PlaceAt(camera="blue bin")),))
        ran = run_sort(plan, picks(scene), arm=arm, locators=[locator], wrist=True, looks=(LY, LB, LE), hooks=hooks)
        ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
        return ran

    def test_a_bin_moved_between_two_parts_is_found_again_and_the_other_bin_stays_kept_out(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = self._sort(lambda scene: [SortPick("part", label="green part", then=lambda: scene.move(150.0)),
                                        SortPick("part", label="red part"), "empty", "empty"])

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        # The survey grounds both bins at the yellow bin's look and the blue bin's, the check of the moved bin
        # grounds both again, and the last is the end's one locate of every part, which counts what the sort left.
        self.assertEqual(["yellow bin | blue bin"] * 3 + ["each separate object"], ran.locator.asked,
                         "the check's detector grounded one bin of two alone")
        found = ran.hooks.of("task.target_relocated")
        self.assertEqual([("yellow bin", True, "check")], [(f["phrase"], f["found"], f["by"]) for f in found])
        yellow, blue = _line_ins(ran)
        np.testing.assert_allclose((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]), yellow[1][:2], atol=10.0)
        np.testing.assert_allclose(BLUE_CENTRE, blue[1][:2], atol=10.0)
        regions = ran.service.zones_seen[1]
        self.assertEqual(2, len(regions))
        self.assertTrue(any(region.contains((*BLUE_CENTRE, 30.0)) for region in regions), "the blue bin's region "
                                                                                         "went with the yellow one's")
        self.assertTrue(any(region.contains((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1], 30.0)) for region in regions))
        self.assertFalse(any(region.contains((BIN_CENTRE[0] - 140.0, BIN_CENTRE[1], 30.0)) for region in regions),
                         "the yellow bin's old region is still kept out")
        kept = ran.report.kept_targets
        np.testing.assert_allclose((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]), kept["yellow bin"].centre_xy_mm, atol=3.0)
        np.testing.assert_allclose(BLUE_CENTRE, kept["blue bin"].centre_xy_mm, atol=3.0)

    def test_the_other_rules_bin_is_never_taken_for_a_bin_found_nowhere(self) -> None:
        """The yellow bin taken away, on a detector that boxes every bin as the one phrase it is asked for: the search
        grounds both bins, so the blue bin it sees from its own look is the blue bin, and no yellow one is found."""
        from src.robot.execution.task import TaskStop

        def gone(scene: BinScene) -> None:
            scene.bins = scene.bins[1:]

        ran = self._sort(lambda scene: [SortPick("part", label="green part", then=lambda: gone(scene))],
                         locator_type=OnePhraseLocator)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        found = ran.hooks.of("task.target_relocated")[0]
        self.assertEqual((False, [look_label(LB), look_label(LE)]), (found["found"], found["looks_tried"]))
        self.assertEqual(["blue bin"], list(ran.report.kept_targets), "the lost bin was reported kept")


class TogetherTests(unittest.TestCase):
    """Where the detector looks at one bin of two, it grounds both: a second bin standing nearer is never taken."""

    def _two_bins(self) -> tuple[Any, Any, Any, Any]:
        from src.robot.execution.place_target import survey_places

        scene = BinScene(bins=[SeenBin(), SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE)])
        log: list[Any] = []
        arm = _arm(log)
        locator = OnePhraseLocator(scene, arm=arm, log=log)
        found = survey_places(arm, [locator], ("yellow bin", "blue bin"), looks=(LY, LB, LE))
        assert found.complete, found.render()
        yellow = found.place("yellow bin").kept
        assert yellow is not None
        # The yellow bin moved 140 mm, past its bound, and the blue one stands 70 mm from where the yellow one stood;
        # LY sees both whole.
        scene.bins = [replace(scene.bins[0], centre_xy=YELLOW_NOW), replace(scene.bins[1], centre_xy=BLUE_NOW)]
        assert move_to_look(arm, LY).ok
        return scene, arm, locator, yellow

    def test_a_check_that_grounds_every_bin_follows_no_other_bin(self) -> None:
        from src.robot.execution.place_target import recheck

        _scene, _arm_, locator, yellow = self._two_bins()

        alone = recheck([locator], yellow, last=yellow)
        together = recheck([locator], yellow, last=yellow, together=("blue bin",))

        assert alone.kept is not None, alone.render()
        np.testing.assert_allclose(BLUE_NOW, alone.kept.centre_xy_mm, atol=5.0,
                                   err_msg="the one-phrase locate's danger is not shown")
        self.assertIsNone(together.kept, together.render())
        self.assertEqual("moved_too_far", together.why)
        assert together.seen is not None
        np.testing.assert_allclose(YELLOW_NOW, together.seen.centre_xy_mm, atol=5.0,
                                   err_msg="the blue bin was seen as the yellow one")
        self.assertEqual("yellow bin", together.seen.phrase)
        self.assertEqual("yellow bin | blue bin", locator.asked[-1])

    def test_a_search_that_grounds_every_bin_takes_no_other_bin(self) -> None:
        from src.robot.execution.place_target import relocate

        _scene, arm, locator, yellow = self._two_bins()

        def searched(**together: Any) -> Any:
            search = relocate([locator], yellow, last=yellow, colour_delta_e=float("inf"), **together)
            for look in search:
                assert move_to_look(arm, look).ok
                search.at(look)
            return search.result()

        alone, both = searched(), searched(together=("blue bin",))

        assert alone.kept is not None and both.kept is not None, (alone.render(), both.render())
        np.testing.assert_allclose(BLUE_NOW, alone.kept.centre_xy_mm, atol=5.0,
                                   err_msg="the one-phrase locate's danger is not shown")
        np.testing.assert_allclose(YELLOW_NOW, both.kept.centre_xy_mm, atol=5.0,
                                   err_msg="the blue bin was taken for the yellow one")
        self.assertEqual("yellow bin | blue bin", locator.asked[-1])

    def test_with_no_other_place_the_locate_is_the_phrase_alone(self) -> None:
        from src.robot.execution.place_target import recheck

        _scene, _arm_, locator, yellow = self._two_bins()
        asked = len(locator.asked)

        recheck([locator], yellow, last=yellow, together=())

        self.assertEqual(["yellow bin"], locator.asked[asked:])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
