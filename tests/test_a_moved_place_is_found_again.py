"""A place that moved is found again, and a check of one place keeps every other place out (2026-10-09).

The owner: during a sort each bin is checked again when a part goes there. Where it moved, it is LOCATED AGAIN and the
drop is planned anew instead of stopping ("selbst wenn diese verstellt wird, findet er die Ablage"); only a bin found
nowhere asks a person. ``place_target.relocate`` is that search. Where the check before a drop lost the bin (wave 1's
rim read in depth and colour, then the detector), it tries the bin's own look and then every other look once more, for
that place alone, while the task moves the arm: the search itself moves nothing. Only a bin of the size the survey found
and the colour the task followed, standing in no other place, is taken. A check of one place replaces only its own
keep-out region (``ExclusionZones.forget_region``, the sorting map's F4). Before, a check forgot every region, so a later
pick could take sorted parts back out of the other rule's bin.

The camera is ``tests._task_fakes.ClassListLocator``, a wrist camera over a bench it ray casts.
"""

from __future__ import annotations

import json
import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.execution.looks import look_label, move_to_look
from src.robot.grasping.recovery.exclusion_zones import ExclusionRegion, ExclusionZones
from tests._task_fakes import (
    BIN_CENTRE,
    BLUE,
    GRASP_Z_MM,
    PART_XY,
    PLACE_TCP,
    BinScene,
    ClassListLocator,
    ScriptedLocator,
    SeenBin,
    TaskArm,
    bin_object,
    bin_points,
    motions,
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
#: Where only LE sees a bin.
EMPTY_SPOT = (-300.0, -50.0)


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


class _Sort:
    """A sort into the yellow and the blue bin, both found by its survey and kept out of its picks, as a task keeps
    them, with the circle about a taught drop beside them; the arm then carries a part to the yellow bin's look."""

    def __init__(self, **locator_keywords: Any) -> None:
        from src.robot.execution.place_target import survey_places

        self.scene = BinScene(bins=[SeenBin(), SeenBin(label="blue bin", centre_xy=BLUE_CENTRE, colour_bgr=BLUE)])
        self.log: list[Any] = []
        self.arm = TaskArm(self.log, fk_table={_key(LY): AT_LY, _key(LB): AT_LB, _key(LE): AT_LE})
        self.locator = ClassListLocator(self.scene, arm=self.arm, log=self.log, **locator_keywords)
        found = survey_places(self.arm, [self.locator], ("yellow bin", "blue bin"), looks=LOOKS)
        assert found.complete, found.render()
        self.yellow, self.blue = found.place("yellow bin").kept, found.place("blue bin").kept
        assert self.yellow is not None and self.blue is not None
        self.zones = ExclusionZones()
        self.yellow_region = self.zones.keep_out_region(self.yellow.keep_out_region())
        self.blue_region = self.zones.keep_out_region(self.blue.keep_out_region())
        self.circle = self.zones.keep_out_region(ExclusionRegion.circle(
            PLACE_TCP.position_mm, 150.0, reason="the drop at pose 'drop_left'"))
        assert move_to_look(self.arm, LY).ok
        self.asked = len(self.locator.asked)
        self.moved = len(motions(self.log))

    def move_yellow(self, centre: "tuple[float, float]", **changes: Any) -> None:
        from dataclasses import replace

        self.scene.bins[0] = replace(self.scene.bins[0], centre_xy=centre, **changes)

    def check(self) -> Any:
        from src.robot.execution.place_target import recheck

        return recheck([self.locator], self.yellow, last=self.yellow)

    def relocate(self, check: Any, **keywords: Any) -> Any:
        from src.robot.execution.place_target import relocate

        keywords.setdefault("avoid", [self.blue_region, self.circle])
        return relocate([self.locator], self.yellow, looks=LOOKS, check=check, last=self.yellow, **keywords)

    def drive(self, search: Any) -> list[Any]:
        """The task's part: to each look the search hands out, by the arm's judged verb, then one locate there."""
        driven = []
        for look in search:
            assert move_to_look(self.arm, look).ok
            driven.append(look)
            search.at(look)
        return driven

    def since_the_check(self) -> tuple[list[str], list[Any]]:
        """What was located, and what moved, since the arm stood at the yellow bin's look."""
        return self.locator.asked[self.asked:], motions(self.log)[self.moved:]


def _grasp() -> Pose:
    return Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM, label="grasp")


class ForgetOneRegionTests(unittest.TestCase):
    def test_forgetting_one_region_keeps_every_other_region_and_every_zone(self) -> None:
        zones = ExclusionZones()
        zones.start_pick()
        a = zones.keep_out_region(ExclusionRegion.rectangle((400.0, -300.0), (320.0, 220.0), reason="the yellow bin"))
        b = zones.keep_out_region(ExclusionRegion.rectangle((-300.0, -500.0), (320.0, 220.0), reason="the blue bin"))
        circle = zones.keep_out_region(ExclusionRegion.circle((300.0, -400.0), 150.0, reason="the drop"))
        zones.exclude(label="red cube", centre_mm=(0.0, -700.0, 20.0))

        self.assertTrue(zones.forget_region(a))

        self.assertEqual((b, circle), zones.regions())
        self.assertTrue(zones.excludes(label="red cube", centre_mm=(0.0, -700.0, 20.0)), "a zone was forgotten")
        self.assertFalse(zones.forget_region(a), "a region forgotten once is forgotten")
        self.assertEqual((b, circle), zones.regions())

    def test_a_region_equal_to_one_kept_is_forgotten_once(self) -> None:
        zones = ExclusionZones()
        first = zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 50.0, reason="the drop"))
        second = zones.keep_out_region(ExclusionRegion.circle((0.0, 0.0), 50.0, reason="the drop"))

        self.assertTrue(zones.forget_region(ExclusionRegion.circle((0.0, 0.0), 50.0, reason="the drop")))

        self.assertEqual(1, len(zones.regions()))
        self.assertIsNot(first, zones.regions()[0], "the first equal one is forgotten, the other stays")
        self.assertTrue(zones.forget_region(second))
        self.assertEqual((), zones.regions())

    def test_only_a_region_is_forgotten(self) -> None:
        with self.assertRaises(TypeError):
            ExclusionZones().forget_region((0.0, 0.0))  # type: ignore[arg-type]


class ACheckOfOnePlaceTests(unittest.TestCase):
    def test_rechecking_bin_a_keeps_bin_bs_region_and_a_taught_poses_circle(self) -> None:
        sort = _Sort()
        sort.move_yellow((BIN_CENTRE[0] + 40.0, BIN_CENTRE[1]))
        check = sort.check()
        assert check.kept is not None, check.render()
        self.assertEqual("detector", check.by)

        # As the task follows a check of one place: its own region replaced, every other one kept.
        self.assertTrue(sort.zones.forget_region(sort.yellow_region))
        followed = sort.zones.keep_out_region(check.kept.keep_out_region())

        self.assertEqual((sort.blue_region, sort.circle, followed), sort.zones.regions())
        in_blue = (BLUE_CENTRE[0], BLUE_CENTRE[1], 30.0)
        at_the_drop = (float(PLACE_TCP.position_mm[0]), float(PLACE_TCP.position_mm[1]), 20.0)
        for where in (in_blue, at_the_drop, (BIN_CENTRE[0] + 40.0, BIN_CENTRE[1], 30.0)):
            with self.subTest(where=where):
                self.assertTrue(sort.zones.kept_out_by_a_region(label="red cube", centre_mm=where))
        self.assertFalse(sort.zones.kept_out_by_a_region(label="red cube", centre_mm=(255.0, -200.0, 30.0)),
                         "the yellow bin's old edge is still kept out")

    def test_forgetting_every_region_at_a_check_would_let_a_pick_take_parts_back_out_of_the_other_bin(self) -> None:
        """What a check did before (``forget_regions``): the blue bin and the drop's circle went with the yellow
        bin's region."""
        sort = _Sort()

        sort.zones.forget_regions()
        sort.zones.keep_out_region(sort.yellow.keep_out_region())

        self.assertFalse(sort.zones.kept_out_by_a_region(label="red cube",
                                                         centre_mm=(BLUE_CENTRE[0], BLUE_CENTRE[1], 30.0)))


class AMovedBinTests(unittest.TestCase):
    def test_a_bin_moved_150_mm_is_found_again_from_the_checks_own_sighting_and_its_drop_moves_with_it(self) -> None:
        from src.robot.execution.place_target import drop_plan, nominal_drop

        sort = _Sort()
        sort.move_yellow((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]))
        check = sort.check()
        self.assertEqual(("moved_too_far", "detector"), (check.why, check.by))
        asked_by_the_check = len(sort.locator.asked)

        search = sort.relocate(check)

        self.assertTrue(search.done)
        self.assertEqual([], list(search), "a look was handed out for a bin the check already saw")
        self.assertEqual(asked_by_the_check, len(sort.locator.asked), "the detector was asked again")
        self.assertEqual([], sort.since_the_check()[1], "the search moved the arm")
        self.assertIs(search.kept, search.at(LB), "a search that found its bin located again")
        self.assertEqual(asked_by_the_check, len(sort.locator.asked))
        found = search.result()
        assert found.kept is not None
        self.assertEqual(("check", look_label(LY)), (found.by, found.kept.look_label))
        self.assertAlmostEqual(150.0, found.moved_mm, delta=3.0)
        np.testing.assert_allclose((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1]), found.kept.centre_xy_mm, atol=3.0)
        self.assertEqual("yellow bin", found.kept.phrase)
        before = nominal_drop(sort.arm, sort.yellow, hang_mm=40.0, air_mm=20.0)
        after = nominal_drop(sort.arm, found.kept, hang_mm=40.0, air_mm=20.0)
        assert before.pose is not None and after.pose is not None and before.ok and after.ok
        self.assertAlmostEqual(150.0, float(after.pose.position_mm[0] - before.pose.position_mm[0]), delta=3.0)
        old = drop_plan(sort.arm, sort.yellow, _grasp(), part_bottom_mm=0.0, air_mm=20.0)
        new = drop_plan(sort.arm, found.kept, _grasp(), part_bottom_mm=0.0, air_mm=20.0)
        assert old.pose is not None and new.pose is not None and new.ok, new.render()
        self.assertGreater(float(np.linalg.norm(new.pose.position_mm[:2] - old.pose.position_mm[:2])), 140.0)
        self.assertIn("found again", found.render())
        json.dumps(found.to_dict())

    def test_a_bin_moved_out_of_its_looks_view_is_found_from_the_look_the_task_drove_to(self) -> None:
        from src.robot.execution.place_target import nominal_drop

        sort = _Sort()
        sort.move_yellow(EMPTY_SPOT)
        check = sort.check()
        self.assertEqual(("not_seen", "detector"), (check.why, check.by))
        search = sort.relocate(check)
        self.assertEqual((LB, LE), search.looks, "the bin's own look again, where the detector just saw nothing")
        commanded_before = len(motions(sort.log))

        driven = sort.drive(search)

        self.assertEqual([LB, LE], driven)
        asked, moved = sort.since_the_check()
        self.assertEqual(["yellow bin", "yellow bin", "yellow bin"], asked, "the check, then one locate per look")
        self.assertEqual([("joints", _key(LB)), ("joints", _key(LE))], motions(sort.log)[commanded_before:],
                         "the search moved the arm")
        found = search.result()
        assert found.kept is not None
        self.assertEqual(("detector", look_label(LE)), (found.by, found.kept.look_label))
        self.assertIs(LE, found.kept.look, "the bin is kept from a look the task looks from")
        self.assertEqual((look_label(LB), look_label(LE)), found.looks_tried)
        np.testing.assert_allclose(EMPTY_SPOT, found.kept.centre_xy_mm, atol=3.0)
        drop = nominal_drop(sort.arm, found.kept, hang_mm=40.0, air_mm=20.0)
        assert drop.pose is not None
        np.testing.assert_allclose(EMPTY_SPOT, drop.pose.position_mm[:2], atol=10.0)

    def test_a_bin_taken_away_is_found_nowhere_and_says_so(self) -> None:
        sort = _Sort()
        sort.scene.bins = sort.scene.bins[1:]
        check = sort.check()

        search = sort.relocate(check)
        sort.drive(search)

        found = search.result()
        self.assertTrue(found.found_nowhere)
        self.assertIsNone(found.kept)
        self.assertEqual("", found.by)
        self.assertEqual((look_label(LB), look_label(LE)), found.looks_tried)
        said = found.render()
        self.assertIn("the yellow bin was found nowhere", said)
        self.assertIn(look_label(LE), said)
        self.assertEqual({"phrase": "yellow bin", "found": False, "moved_mm": None, "by": "", "look": None,
                          "target": None}, {key: found.to_dict()[key] for key in
                                            ("phrase", "found", "moved_mm", "by", "look", "target")})

    def test_the_other_rules_bin_is_never_taken_for_it(self) -> None:
        """The yellow bin gone, and a detector that calls the blue bin a yellow one: the blue bin stands in its own
        region and in another colour, either of which passes it over."""
        for keywords, why in (({}, "the blue bin it places into"), ({"avoid": ()}, "another colour")):
            with self.subTest(why=why):
                sort = _Sort()
                sort.locator.called = {"blue bin": "yellow bin"}
                sort.scene.bins = sort.scene.bins[1:]

                search = sort.relocate(sort.check(), **keywords)
                sort.drive(search)

                found = search.result()
                self.assertTrue(found.found_nowhere, found.render())
                self.assertTrue(any(why in passed for passed in found.passed_over), found.passed_over)

    def test_a_bin_of_another_size_is_not_taken(self) -> None:
        sort = _Sort()
        sort.move_yellow(EMPTY_SPOT, size_xy=(200.0, 120.0))

        search = sort.relocate(sort.check())
        sort.drive(search)

        found = search.result()
        self.assertTrue(found.found_nowhere)
        self.assertTrue(any("another size" in passed for passed in found.passed_over), found.passed_over)

    def test_a_look_the_task_could_not_reach_is_skipped_and_said(self) -> None:
        sort = _Sort()
        sort.move_yellow(EMPTY_SPOT)
        search = sort.relocate(sort.check())

        for look in search:
            if look is LB:
                search.skip(look, "workspace_rejected: scripted")
                continue
            assert move_to_look(sort.arm, look).ok
            search.at(look)

        found = search.result()
        self.assertEqual(((look_label(LB), "workspace_rejected: scripted"),), found.refused)
        self.assertEqual((look_label(LE),), found.looks_tried)
        assert found.kept is not None
        self.assertIn("skipped look", found.render())

    def test_a_check_that_followed_needs_no_search_and_a_search_starts_from_a_kept_bin(self) -> None:
        from src.robot.execution.place_target import relocate

        sort = _Sort()
        followed = sort.check()
        self.assertTrue(followed.followed)

        with self.assertRaises(ValueError):
            relocate([sort.locator], sort.yellow, looks=LOOKS, check=followed)
        with self.assertRaises(TypeError):
            relocate([sort.locator], bin_object(), looks=LOOKS)  # type: ignore[arg-type]


class AFixedCamerasBinTests(unittest.TestCase):
    def _kept(self, locator: ScriptedLocator) -> Any:
        from src.robot.execution.place_target import survey

        kept = survey(TaskArm([]), [locator], "blue bin").kept
        assert kept is not None
        return kept

    def test_a_bin_moved_150_mm_is_taken_from_the_checks_sighting_with_no_other_locate(self) -> None:
        from src.robot.execution.place_target import recheck, relocate

        seen = [bin_object()]
        locator = ScriptedLocator({"blue bin": lambda _tcp: list(seen)}, wrist=False, rig_id="overhead")
        kept = self._kept(locator)
        seen[:] = [bin_object(bin_points((BIN_CENTRE[0] + 150.0, BIN_CENTRE[1])))]
        check = recheck([locator], kept)
        self.assertEqual("moved_too_far", check.why)

        search = relocate([locator], kept, check=check)

        self.assertEqual([], list(search))
        found = search.result()
        assert found.kept is not None
        self.assertEqual(("check", None), (found.by, found.kept.look))
        self.assertEqual(["blue bin", "blue bin"], locator.asked, "the survey, then the check; nothing more")

    def test_a_bin_the_check_saw_nowhere_is_found_nowhere_with_no_other_locate(self) -> None:
        from src.robot.execution.place_target import recheck, relocate

        seen = [bin_object()]
        locator = ScriptedLocator({"blue bin": lambda _tcp: list(seen)}, wrist=False, rig_id="overhead")
        kept = self._kept(locator)
        seen[:] = []

        search = relocate([locator], kept, check=recheck([locator], kept))

        self.assertEqual([], list(search))
        self.assertTrue(search.result().found_nowhere)
        self.assertEqual(["blue bin", "blue bin"], locator.asked)

    def test_with_no_check_the_fixed_camera_locates_once_where_it_stands(self) -> None:
        from src.robot.execution.place_target import relocate

        seen = [bin_object()]
        locator = ScriptedLocator({"blue bin": lambda _tcp: list(seen)}, wrist=False, rig_id="overhead")
        kept = self._kept(locator)
        seen[:] = [bin_object(bin_points((BIN_CENTRE[0] - 120.0, BIN_CENTRE[1])))]

        search = relocate([locator], kept)
        looks = list(search)
        self.assertEqual([None], looks)
        search.at(None)

        found = search.result()
        assert found.kept is not None
        self.assertEqual("detector", found.by)
        np.testing.assert_allclose((BIN_CENTRE[0] - 120.0, BIN_CENTRE[1]), found.kept.centre_xy_mm, atol=3.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
