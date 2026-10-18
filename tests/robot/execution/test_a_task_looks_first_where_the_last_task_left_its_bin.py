"""A task looks first where the last task left its bin, and keeps a bin that stood still with one look at its rim.

The owner, 2026-10-08: speed first. Every task into a bin surveyed it from its looks, and grounded it there (4.5 to 6 s
on the cell), though the last task into the same bin had just kept it. A task now reports the bin it followed last
(``TaskReport.kept_target``; none where it was lost or never found), and the next task into the same phrase may be
handed it (``run_task(..., known_target=)``; the console hands its own, a Restart included). The survey then goes to
that bin's look first and looks at it again as before a drop (``place_target.recheck``): its rim's depth and colour, the
detector only where they are unsure. Followed there, the bin is kept and no other look runs, so a bin that stood still
costs one frame and no grounding. A bin that moved, or is gone, is surveyed from every look, as without one. A known
bin found for another phrase, or seen from a look this task does not look from, is not looked at.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    BinScene,
    MeasuringLocator,
    Pick,
    ScriptedLocator,
    SeenBin,
    TaskArm,
    bin_object,
    motions,
    run,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L3 = JointPositions.deg(-22.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: L1 and L3 look straight down at the bin, L2 at the parts.
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")
AT_L3 = Pose.tool_down(420.0, -300.0, 450.0, label="L3")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _task(scene: BinScene, *, phrase: str = "yellow bin", known: Any = None, looks: tuple[Any, ...] = (L1, L2),
          picks: Any = ("part",), **keywords: Any) -> Any:
    """A task into ``phrase`` on a wrist camera over ``scene`` that reads rims (:class:`MeasuringLocator`)."""
    from src.robot.execution.task import PlaceAt

    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2, _key(L3): AT_L3})
    locator = MeasuringLocator(scene, arm=arm, log=log)
    ran = run(picks, place=PlaceAt(camera=phrase), arm=arm, wrist=True, looks=looks, locators=[locator],
              known_target=known, **keywords)
    ran.locator = locator  # type: ignore[attr-defined]
    return ran


def _kept_by_a_first_task(scene: BinScene) -> Any:
    first = _task(scene)
    kept = first.report.kept_target
    assert kept is not None, first.report.sentence
    return kept


def _survey_motions(ran: Any) -> list[Any]:
    """The arm's motions from the survey's start to the first part."""
    survey = ran.log.index(("event", "task.survey_started"))
    part = ran.log.index(("event", "task.part_started"))
    return motions(ran.log[survey:part])


class AKnownBinTests(unittest.TestCase):
    def test_a_restart_into_a_bin_that_stood_still_asks_the_detector_for_no_bin(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task(scene, known=_kept_by_a_first_task(scene), first_motion="return")

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(("home",), motions(ran.log)[0])
        self.assertEqual(["task.return_started", "task.returned", "task.survey_started"], ran.names()[:3])
        self.assertEqual([], ran.locator.asked, "the detector was asked for a bin that stood still")
        self.assertEqual(2, len(ran.locator.measured), "the rim at the survey, the rim before the drop")
        found = ran.hooks.of("task.target_found")[0]
        self.assertEqual((True, "depth", look_label(L1)), (found["known"], found["by"], found["look"]))
        self.assertIn("where the last task left it", ran.hooks.said[ran.names().index("task.target_found")])

    def test_the_survey_of_a_bin_that_stood_still_goes_to_its_look_alone(self) -> None:
        scene = BinScene()
        ran = _task(scene, known=_kept_by_a_first_task(scene))

        self.assertEqual([("joints", _key(L1))], _survey_motions(ran))
        self.assertEqual(look_label(L1), ran.hooks.of("task.target_found")[0]["look"])
        np.testing.assert_allclose(BIN_CENTRE, ran.report.kept_target.centre_xy_mm, atol=3.0)

    def test_a_known_bin_that_moved_is_surveyed_from_every_look(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        kept = _kept_by_a_first_task(scene)
        scene.move(150.0)
        ran = _task(scene, known=kept)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked, "the detector at the known look, then the "
                                                                          "survey's first look")
        # The survey's first look is the known bin's, where the arm already stands: nothing is sent to it again
        # (``looks.move_to_look``).
        self.assertEqual([("joints", _key(L1))], _survey_motions(ran))
        found = ran.hooks.of("task.target_found")[0]
        self.assertEqual((False, ""), (found["known"], found["by"]))
        self.assertAlmostEqual(BIN_CENTRE[0] + 150.0, found["target"]["centre_mm"][0], delta=3.0)

    def test_a_known_bin_taken_away_is_not_found_as_without_one(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        kept = _kept_by_a_first_task(scene)
        scene.take_away()
        ran = _task(scene, known=kept)

        self.assertIs(TaskStop.TARGET_NOT_FOUND, ran.report.stop, ran.report.sentence)
        self.assertEqual([], ran.service.calls, "a part was picked for a bin nobody found")
        self.assertIsNone(ran.report.kept_target)

    def test_a_known_bin_found_for_another_phrase_is_not_looked_at(self) -> None:
        scene = BinScene()
        kept = _kept_by_a_first_task(scene)
        scene.bins = [SeenBin(label="blue bin", colour_bgr=BLUE)]
        ran = _task(scene, phrase="blue bin", known=kept)

        survey = ran.log[ran.log.index(("event", "task.survey_started")):ran.log.index(("event", "task.target_found"))]
        self.assertEqual([("locate", "blue bin")], [entry for entry in survey if entry[0] in ("locate", "measure")],
                         "the yellow bin's rim was read for the blue bin")
        self.assertIs(False, ran.hooks.of("task.target_found")[0]["known"])

    def test_a_known_bin_seen_from_a_look_the_task_does_not_look_from_is_not_looked_at(self) -> None:
        scene = BinScene()
        kept = _kept_by_a_first_task(scene)
        ran = _task(scene, known=kept, looks=(L3, L2))

        self.assertEqual([("joints", _key(L3))], _survey_motions(ran))
        self.assertEqual(["yellow bin"], ran.locator.asked)
        found = ran.hooks.of("task.target_found")[0]
        self.assertEqual((False, look_label(L3)), (found["known"], found["look"]))


class TheKeptTargetTests(unittest.TestCase):
    def test_the_report_keeps_the_bin_the_task_followed_last(self) -> None:
        scene = BinScene()
        ran = _task(scene, picks=(Pick("part", then=lambda: scene.move(40.0)),))

        kept = ran.report.kept_target
        assert kept is not None
        self.assertEqual("yellow bin", kept.label)
        self.assertEqual(look_label(L1), kept.look_label)
        np.testing.assert_allclose((BIN_CENTRE[0] + 40.0, BIN_CENTRE[1]), kept.centre_xy_mm, atol=3.0)
        said = ran.report.to_dict()
        json.dumps(said)
        self.assertEqual("yellow bin", said["kept_target"]["label"])

    def test_a_task_that_lost_its_bin_or_never_found_one_keeps_none(self) -> None:
        from src.robot.execution.task import TaskStop

        lost_scene = BinScene()
        lost = _task(lost_scene, picks=(Pick("part", then=lost_scene.take_away),))
        never = _task(BinScene(bins=[]))

        self.assertEqual((TaskStop.TARGET_LOST, None), (lost.report.stop, lost.report.kept_target))
        self.assertEqual((TaskStop.TARGET_NOT_FOUND, None), (never.report.stop, never.report.kept_target))
        self.assertIsNone(never.report.to_dict()["kept_target"])

    def test_a_pose_place_keeps_no_bin(self) -> None:
        self.assertIsNone(run(("part",)).report.kept_target)

    def test_a_known_target_that_is_no_kept_bin_is_a_programmers_error(self) -> None:
        with self.assertRaises(TypeError):
            _task(BinScene(), known=bin_object())


class TheSurveyOfAKnownBinTests(unittest.TestCase):
    def test_a_known_bin_followed_at_its_look_ends_the_survey_there(self) -> None:
        from src.robot.execution.place_target import survey

        scene = BinScene()
        kept = _kept_by_a_first_task(scene)
        log: list[Any] = []
        arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
        locator = MeasuringLocator(scene, arm=arm)

        found = survey(arm, [locator], "yellow bin", looks=(L2, L1), known=kept)

        self.assertEqual([("joints", _key(L1))], motions(log), "the survey went on past the bin it knew")
        assert found.kept is not None and found.known is not None
        self.assertEqual((True, "depth"), (found.known.followed, found.known.by))
        self.assertEqual((look_label(L1),), found.looks_tried)
        self.assertIs(L1, found.kept.look, "the bin is carried to the task's own look")
        self.assertIn("looked at again", found.render())

    def test_a_fixed_cameras_known_bin_is_looked_at_where_it_stands(self) -> None:
        from src.robot.execution.place_target import survey

        log: list[Any] = []
        arm = TaskArm(log)
        locator = ScriptedLocator({"blue bin": [bin_object()]}, wrist=False, rig_id="overhead")
        first = survey(arm, [locator], "blue bin")
        assert first.kept is not None

        found = survey(arm, [locator], "blue bin", known=first.kept)

        self.assertEqual([], motions(log))
        assert found.known is not None
        self.assertEqual((True, "detector"), (found.known.followed, found.known.by))
        self.assertEqual(["blue bin", "blue bin"], locator.asked,
                         "the first survey, then the check of the bin it kept; no survey after a bin followed")

    def test_a_known_bin_of_a_camera_the_task_does_not_hold_is_not_looked_at(self) -> None:
        from dataclasses import replace

        from src.robot.execution.place_target import survey

        scene = BinScene()
        kept = replace(_kept_by_a_first_task(scene), camera="side")
        arm = TaskArm([], fk_table={_key(L1): AT_L1})
        locator = MeasuringLocator(scene, arm=arm)

        found = survey(arm, [locator], "yellow bin", looks=(L1,), known=kept)

        self.assertIsNone(found.known)
        self.assertEqual(["yellow bin"], locator.asked)
        self.assertEqual([], locator.measured)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
