"""A sort finds every place before its first pick, with one locate per look for every place still missing (2026-10-09).

The owner: "gruene Teile in die gelbe Kiste, rote in die blaue". Every rule's bin is found before the first pick, or the
task stops and names the bin it is missing. ``place_target.survey_places`` finds them as ``survey`` finds one, look by
look through the arm's own judged verb, asking ``may_move`` before every motion. At each look it makes ONE locate of a
class list of every place still missing ("yellow bin | blue bin"), and each located object is the place its label
names, exactly. The last place missing is located alone, as today. The survey ends once every place is kept, and names
each place no look saw. A sighting that stands in a place already kept is not taken for another place, so one bin is
never two places. The bins an earlier task kept are looked at again first, each from its own look. One place is
surveyed exactly as ``survey`` surveys it.

The camera is ``tests._task_fakes.ClassListLocator``: a wrist camera over a bench it ray casts. It grounds a class list
as the cell's locator will once it maps the detector's words onto the list (``object_labels``).
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    BinScene,
    ClassListLocator,
    MeasuringLocator,
    ScriptedLocator,
    SeenBin,
    TaskArm,
    bin_object,
    bin_points,
    motions,
)

#: Above the yellow bin, above the blue one, above an empty spot of the bench, and high over both bins.
LY = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LB = JointPositions.deg(40.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LE = JointPositions.deg(70.0, -90.0, -110.0, -60.0, 90.0, 0.0)
LW = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_LY = Pose.tool_down(BIN_CENTRE[0], BIN_CENTRE[1], 450.0, label="LY")
AT_LB = Pose.tool_down(-300.0, -500.0, 450.0, label="LB")
AT_LE = Pose.tool_down(-300.0, -50.0, 450.0, label="LE")
AT_LW = Pose.tool_down(50.0, -400.0, 900.0, label="LW")
BLUE_CENTRE = (-300.0, -500.0)
PLACES = ("yellow bin", "blue bin")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _scene(*, blue: bool = True) -> BinScene:
    """The yellow bin where only LY sees it, and the blue one where only LB sees it; LW sees both."""
    bins = [SeenBin()]
    if blue:
        bins.append(SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE))
    return BinScene(bins=bins)


def _arm(log: list[Any], **keywords: Any) -> TaskArm:
    return TaskArm(log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE, _key(LW): AT_LW}, **keywords)


def _cell(scene: BinScene, locator: Any = ClassListLocator, **keywords: Any) -> tuple[list[Any], TaskArm, Any]:
    log: list[Any] = []
    arm = _arm(log, **keywords)
    return log, arm, locator(scene, arm=arm, log=log)


def _looks(*poses: JointPositions) -> tuple[str, ...]:
    return tuple(look_label(pose) for pose in poses)


class EveryPlaceTests(unittest.TestCase):
    def test_two_bins_seen_from_two_looks_are_found_with_one_locate_per_look(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene())

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB, LE))

        self.assertEqual(["yellow bin | blue bin", "yellow bin | blue bin"], locator.asked,
                         "every place in one locate per look, the one kept already too, so a second bin in view is "
                         "never taken for the one still missing")
        self.assertEqual([("joints", _key(LY)), ("locate", "yellow bin | blue bin"), ("joints", _key(LB)),
                          ("locate", "yellow bin | blue bin")],
                         [entry for entry in log if entry[0] in ("joints", "locate")],
                         "the survey went on past the last bin, or located twice at a look")
        self.assertTrue(found.complete)
        self.assertEqual((), found.missing)
        yellow, blue = found.place("yellow bin").kept, found.place("blue bin").kept
        assert yellow is not None and blue is not None
        np.testing.assert_allclose(BIN_CENTRE, yellow.centre_xy_mm, atol=3.0)
        np.testing.assert_allclose(BLUE_CENTRE, blue.centre_xy_mm, atol=3.0)
        self.assertEqual((LY, LB), (yellow.look, blue.look))
        self.assertEqual(PLACES, (yellow.phrase, blue.phrase), "a place is kept under its own phrase, not the list")
        self.assertIsNotNone(yellow.colour_lab, "the rim's colour was not read on the survey's frame")
        self.assertIsNotNone(blue.colour_lab)
        self.assertEqual(_looks(LY, LB), found.looks_tried)
        self.assertEqual(_looks(LY), found.place("yellow bin").looks_tried)
        self.assertEqual(_looks(LY, LB), found.place("blue bin").looks_tried)

    def test_two_bins_one_look_sees_are_found_with_one_locate_and_no_other_look(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene())

        found = survey_places(arm, [locator], PLACES, looks=(LW, LY, LB))

        self.assertEqual(["yellow bin | blue bin"], locator.asked)
        self.assertEqual([("joints", _key(LW))], motions(log))
        self.assertTrue(found.complete)
        for phrase, centre in zip(PLACES, (BIN_CENTRE, BLUE_CENTRE)):
            with self.subTest(phrase):
                kept = found.place(phrase).kept
                assert kept is not None
                np.testing.assert_allclose(centre, kept.centre_xy_mm, atol=3.0)
                self.assertEqual(look_label(LW), kept.look_label)

    def test_a_bin_no_look_saw_is_named_and_the_one_seen_is_kept(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene(blue=False))

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB, LE))

        self.assertEqual(("blue bin",), found.missing)
        self.assertFalse(found.complete)
        self.assertEqual(["yellow bin | blue bin"] * 3, locator.asked, "every place grounded at every look")
        self.assertEqual(_looks(LY, LB, LE), found.place("blue bin").looks_tried, "the looks it was looked for from")
        self.assertIsNotNone(found.place("yellow bin").kept)
        said = found.render()
        self.assertIn("missing: blue bin", said)
        self.assertIn("no blue bin was seen from", said)
        one = found.survey_of("blue bin")
        self.assertIsNone(one.kept)
        self.assertIsNone(one.parts_seen)
        self.assertEqual(found.place("blue bin").looks_tried, one.looks_tried)
        self.assertIs(found.place("yellow bin").kept, found.survey_of(" Yellow  Bin").kept, "case and blanks aside")
        with self.assertRaises(KeyError):
            found.place("red bin")

    def test_a_label_that_names_no_place_is_no_place_and_is_never_mapped_onto_one(self) -> None:
        """A box the class list's phrases both claimed (``ambiguous``), a bin whose pixels said another colour than its
        label: neither is a place, though "blue object, not yellow bin" holds every word of "yellow bin"."""
        from src.robot.execution.place_target import survey_places

        for called in ("ambiguous", "blue object, not yellow bin"):
            with self.subTest(called=called):
                log: list[Any] = []
                arm = _arm(log)
                locator = ClassListLocator(_scene(), arm=arm, log=log, called={"blue bin": called})

                found = survey_places(arm, [locator], PLACES, looks=(LW,))

                self.assertEqual(("blue bin",), found.missing)
                kept = found.place("yellow bin").kept
                assert kept is not None
                np.testing.assert_allclose(BIN_CENTRE, kept.centre_xy_mm, atol=3.0)

    def test_a_bin_kept_for_one_place_is_never_taken_for_another(self) -> None:
        """At the blue bin's look the detector calls the yellow bin the survey kept the blue bin, and more surely than
        the blue bin beside it: it stands in the yellow bin, so it is not the blue one."""
        from src.robot.execution.place_target import survey_places

        yellow = bin_object(label="yellow bin", score=0.8)
        in_the_yellow_bin = bin_object(label="blue bin", score=0.95)
        blue = bin_object(bin_points(BLUE_CENTRE), label="blue bin", score=0.6)
        for at_lb, kept_blue in (([in_the_yellow_bin], None), ([in_the_yellow_bin, blue], BLUE_CENTRE)):
            with self.subTest(at_lb=len(at_lb)):
                log: list[Any] = []
                arm = _arm(log)
                locator = ScriptedLocator({
                    "yellow bin | blue bin": lambda tcp, seen=at_lb: (
                        [yellow] if tcp is AT_LY else list(seen) if tcp is AT_LB else [])}, arm=arm, log=log)

                found = survey_places(arm, [locator], PLACES, looks=(LY, LB))

                kept = found.place("blue bin").kept
                if kept_blue is None:
                    self.assertIsNone(kept, "the yellow bin was kept as the blue bin too")
                else:
                    assert kept is not None
                    np.testing.assert_allclose(kept_blue, kept.centre_xy_mm, atol=3.0)
                    self.assertEqual(0.6, kept.score)

    def test_of_several_sightings_of_a_place_the_most_confident_is_kept(self) -> None:
        from src.robot.execution.place_target import survey_places

        arm = _arm([])
        sure = bin_object(bin_points((450.0, -200.0)), label="yellow bin", score=0.9)
        locator = ScriptedLocator({"yellow bin | blue bin": [
            bin_object(label="yellow bin", score=0.6), sure, bin_object(bin_points(BLUE_CENTRE), label="blue bin")]},
            arm=arm)

        found = survey_places(arm, [locator], PLACES, looks=(LY,))

        kept = found.place("yellow bin").kept
        assert kept is not None
        self.assertEqual(0.9, kept.score)
        self.assertTrue(found.complete)


class KnownPlacesTests(unittest.TestCase):
    def _kept_by_a_first_sort(self, scene: BinScene) -> list[Any]:
        from src.robot.execution.place_target import survey_places

        _log, arm, locator = _cell(scene)
        first = survey_places(arm, [locator], PLACES, looks=(LY, LB))
        assert first.complete, first.render()
        return [first.place(phrase).kept for phrase in PLACES]

    def test_a_restart_into_bins_that_stood_still_reads_each_rim_and_asks_the_detector_for_none(self) -> None:
        from src.robot.execution.place_target import survey_places

        scene = _scene()
        known = self._kept_by_a_first_sort(scene)
        log, arm, locator = _cell(scene)

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB, LE), known=known)

        self.assertEqual([], locator.asked, "the detector was asked for bins that stood still")
        self.assertEqual(2, len(locator.measured), "one frame for each bin's rim")
        self.assertEqual([("joints", _key(LY)), ("joints", _key(LB))], motions(log))
        for phrase, look in zip(PLACES, (LY, LB)):
            with self.subTest(phrase):
                place = found.place(phrase)
                assert place.known is not None and place.kept is not None
                self.assertEqual((True, "depth"), (place.known.followed, place.known.by))
                self.assertIs(look, place.kept.look)
                self.assertIn("where the last task left it", place.render())

    def test_a_known_bin_that_moved_is_surveyed_and_the_one_that_stood_still_is_followed(self) -> None:
        from src.robot.execution.place_target import survey_places

        scene = _scene()
        known = self._kept_by_a_first_sort(scene)
        scene.bins[1] = replace(scene.bins[1], centre_xy=(BLUE_CENTRE[0] + 150.0, BLUE_CENTRE[1]))
        log, arm, locator = _cell(scene)

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB), known=known)

        self.assertTrue(found.complete, found.render())
        yellow, blue = found.place("yellow bin"), found.place("blue bin")
        assert yellow.known is not None and blue.known is not None and blue.kept is not None
        self.assertEqual((True, False), (yellow.known.followed, blue.known.followed))
        np.testing.assert_allclose((BLUE_CENTRE[0] + 150.0, BLUE_CENTRE[1]), blue.kept.centre_xy_mm, atol=3.0)
        self.assertEqual(["blue bin | yellow bin", "yellow bin | blue bin", "yellow bin | blue bin"], locator.asked,
                         "the detector at the blue bin's own look, then from every look, every place grounded each "
                         "time")

    def test_a_known_bin_of_no_place_is_not_used_and_one_that_is_no_kept_bin_is_a_programmers_error(self) -> None:
        from src.robot.execution.place_target import survey_places

        scene = _scene()
        known = self._kept_by_a_first_sort(scene)
        log, arm, locator = _cell(scene)

        found = survey_places(arm, [locator], ("yellow bin",), looks=(LY,), known={"blue bin": known[1], "red": None})

        self.assertIsNone(found.place("yellow bin").known)
        self.assertEqual(["yellow bin"], locator.asked)
        with self.assertRaises(TypeError):
            survey_places(arm, [locator], PLACES, looks=(LY,), known=[bin_object()])


class LikeTheSurveyTests(unittest.TestCase):
    """What ends the survey of a sort, and what it skips, as in ``survey``."""

    def test_a_task_that_may_not_move_any_more_keeps_what_it_found_and_moves_no_further(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene())
        asked: list[bool] = []

        def may_move() -> str:
            asked.append(True)
            return "" if len(asked) == 1 else "halt now was pressed"

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB), may_move=may_move)

        self.assertEqual([("joints", _key(LY))], motions(log))
        self.assertEqual("halt now was pressed", found.halted)
        self.assertIsNotNone(found.place("yellow bin").kept)
        self.assertEqual(("blue bin",), found.missing)
        self.assertIn("stopped before a motion", found.render())

    def test_a_look_refused_before_anything_was_sent_is_skipped_and_said(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene(), refuse=lambda kind, index, target: (
            MotionStatus.WORKSPACE_REJECTED if index == 0 else None))

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB))

        self.assertEqual(look_label(LY), found.refused[0][0])
        self.assertEqual(["yellow bin | blue bin"], locator.asked, "both places were still missing at LB")
        self.assertEqual(("yellow bin",), found.missing)
        self.assertIsNone(found.stopped)

    def test_a_look_motion_that_may_have_moved_the_arm_ends_the_survey_there(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene(), refuse=lambda kind, index, target: (
            MotionStatus.CONTROLLER_REJECTED if index == 1 else None))

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB))

        assert found.stopped is not None
        self.assertEqual(look_label(LB), found.stopped_look)
        self.assertEqual(1, len(motions(log)))
        self.assertIsNotNone(found.place("yellow bin").kept, "what was kept before the stop stands")
        self.assertEqual(("blue bin",), found.missing)

    def test_a_fixed_camera_locates_the_class_list_once_and_moves_nothing(self) -> None:
        from src.robot.execution.place_target import survey_places

        log: list[Any] = []
        arm = _arm(log)
        locator = ScriptedLocator({"yellow bin | blue bin": [
            bin_object(label="yellow bin"), bin_object(bin_points(BLUE_CENTRE), label="Blue  Bin")]},
            wrist=False, rig_id="overhead")

        found = survey_places(arm, [locator], PLACES, looks=(LY, LB))

        self.assertEqual([], motions(log))
        self.assertEqual(["yellow bin | blue bin"], locator.asked)
        self.assertTrue(found.complete, "a label is read with its case and blanks aside")
        for phrase in PLACES:
            kept = found.place(phrase).kept
            assert kept is not None
            self.assertEqual((None, None, "overhead"), (kept.look, kept.look_label, kept.camera))

    def test_a_place_said_twice_is_one_and_a_phrase_that_names_nothing_is_a_programmers_error(self) -> None:
        from src.robot.execution.place_target import survey_places

        log, arm, locator = _cell(_scene())

        found = survey_places(arm, [locator], ["yellow bin", " Yellow bin "], looks=(LY,))

        self.assertEqual(("yellow bin",), tuple(place.phrase for place in found.places))
        self.assertEqual(["yellow bin"], locator.asked)
        for phrases in ([], [""], ["yellow bin | blue bin"], ["yellow bin", "  "]):
            with self.subTest(phrases=phrases), self.assertRaises(ValueError):
                survey_places(arm, [locator], phrases, looks=(LY,))


class OnePlaceTests(unittest.TestCase):
    """One place is surveyed exactly as ``survey`` surveys it: the same locates, frames, motions and bin."""

    def _both(self, scene_of: Any, looks: tuple[Any, ...], *, known: Any = None, locator: Any = MeasuringLocator,
              **keywords: Any) -> tuple[Any, Any, list[Any], list[Any], Any, Any]:
        from src.robot.execution.place_target import survey, survey_places

        log_a, arm_a, locator_a = _cell(scene_of(), locator, **keywords)
        log_b, arm_b, locator_b = _cell(scene_of(), locator, **keywords)
        today = survey(arm_a, [locator_a], "yellow bin", looks=looks, known=known)
        sort = survey_places(arm_b, [locator_b], ["yellow bin"], looks=looks, known=known)
        return today, sort.survey_of("yellow bin"), log_a, log_b, locator_a, locator_b

    def _same(self, today: Any, sort: Any, log_a: list[Any], log_b: list[Any], locator_a: Any, locator_b: Any) -> None:
        self.assertEqual(locator_a.asked, locator_b.asked)
        self.assertEqual(getattr(locator_a, "measured", None), getattr(locator_b, "measured", None))
        self.assertEqual(motions(log_a), motions(log_b))
        self.assertEqual(today.looks_tried, sort.looks_tried)
        self.assertEqual(today.refused, sort.refused)
        self.assertEqual(today.halted, sort.halted)
        self.assertEqual(today.stopped_look, sort.stopped_look)
        self.assertEqual(today.parts_seen, sort.parts_seen)
        self.assertEqual(today.known is None, sort.known is None)
        if today.known is not None:
            self.assertEqual((today.known.followed, today.known.by), (sort.known.followed, sort.known.by))
        self.assertEqual(today.kept is None, sort.kept is None)
        if today.kept is not None:
            self.assertEqual(today.kept.to_dict() | {"seen_at": 0}, sort.kept.to_dict() | {"seen_at": 0})
            self.assertIs(today.kept.look, sort.kept.look)
            self.assertEqual(today.kept.colour_lab, sort.kept.colour_lab)
            self.assertEqual(today.kept.phrase, sort.kept.phrase)

    def test_a_bin_seen_from_the_second_look(self) -> None:
        self._same(*self._both(_scene, (LE, LY, LB)))

    def test_a_bin_no_look_sees(self) -> None:
        self._same(*self._both(lambda: BinScene(bins=[]), (LY, LB)))

    def test_a_bin_seen_from_both_looks_on_the_class_list_camera(self) -> None:
        self._same(*self._both(_scene, (LW, LY), locator=ClassListLocator))

    def test_a_refused_look_and_a_stopped_one(self) -> None:
        for status in (MotionStatus.WORKSPACE_REJECTED, MotionStatus.CONTROLLER_REJECTED):
            with self.subTest(status=status.value):
                self._same(*self._both(_scene, (LE, LY), refuse=lambda kind, index, target, status=status: (
                    status if index == 0 else None)))

    def test_a_known_bin_that_stood_still_and_one_that_moved(self) -> None:
        from src.robot.execution.place_target import survey

        scene = _scene()
        _log, arm, locator = _cell(scene, MeasuringLocator)
        known = survey(arm, [locator], "yellow bin", looks=(LY,)).kept
        assert known is not None
        self._same(*self._both(_scene, (LE, LY), known=known))

        def moved() -> BinScene:
            scene = _scene()
            scene.move(150.0)
            return scene

        self._same(*self._both(moved, (LE, LY), known=known))

    def test_a_fixed_camera(self) -> None:
        from src.robot.execution.place_target import survey, survey_places

        for seen in ([bin_object(label="yellow bin")], []):
            with self.subTest(seen=len(seen)):
                locator_a = ScriptedLocator({"yellow bin": seen}, wrist=False, rig_id="overhead")
                locator_b = ScriptedLocator({"yellow bin": seen}, wrist=False, rig_id="overhead")
                today = survey(_arm([]), [locator_a], "yellow bin", looks=(LY,))
                sort = survey_places(_arm([]), [locator_b], "yellow bin", looks=(LY,)).survey_of("yellow bin")
                self.assertEqual(locator_a.asked, locator_b.asked)
                self.assertEqual(today.kept is None, sort.kept is None)
                if today.kept is not None:
                    self.assertEqual(today.kept.to_dict(), sort.kept.to_dict())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
