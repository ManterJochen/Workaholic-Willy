"""Before every drop the task reads its bin's rim first, and asks the detector only where that is unsure (2026-10-08).

The owner's word on the day of the presentation: about a minute per pick is too slow, speed first. Asking the detector
for the bin again before every drop cost one grounding per part, 4 to 6 s on the cell. Now the camera reads the rim band
of the bin the task follows (its walls' tops) in one frame with no detector (``Locator.measure``): where at least 80 %
of it lands in the frame unhidden, at least 90 % of that reads within 10 mm of where it should, at most 5 % reads
farther (a wall that is gone) and its median colour stands within dE 25 of the one kept, the bin stands where it stood
and is followed there (``by`` ``depth``). Anything else asks the detector, as before, with today's bounds against the
bin the survey found: a bin moved a little is followed, one moved too far, taken away, swapped for another size or
height, or for a bin of another colour, is lost, and the part goes back where it was gripped. A bin hidden by what
stands in front of it asks the detector too; so does one that creeps, which is lost once it stands too far from where
the survey found it. A frame that cannot be taken is a fault of the cell, as a locate that raises is. The motions, the
place and the guard are untouched: only who answers "is it still there" changed. The lost bins here are lost with
``robot.place.relocate`` off, as before; on (the owner, 2026-10-09), a bin lost is looked for again first
(``tests/test_a_bin_that_moved_is_found_again_before_the_drop.py``).

The camera is :class:`tests._task_fakes.MeasuringLocator`, a wrist camera over a bench it ray casts, so the rim is read
from the pixels it was placed from, as on the cell; the bounds are read on synthetic depth beside it.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.core.errors import PerceptionFrameMoved
from src.robot.execution.looks import look_label
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    BLUE,
    GRASP_Z_MM,
    BinScene,
    MeasuringLocator,
    Pick,
    ScriptedLocator,
    SeenBin,
    TaskArm,
    bin_object,
    do0_changes,
    motions,
    placing,
    run,
)

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
#: L1 looks straight down at the bin; L2 at the parts, where no bin stands.
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _task(picks: Any = ("part",), *, scene: "BinScene | None" = None, scope: str = "once",
          measure_raises: "BaseException | None" = None, relocate: bool = True, **keywords: Any) -> Any:
    """A task into the yellow bin on a wrist camera that reads its rim (:class:`MeasuringLocator`); with ``relocate``
    off, on a cell that keeps the old rule for a lost bin."""
    from src.robot.execution.task import PlaceAt

    log: list[Any] = []
    arm = TaskArm(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
    if not relocate:
        placing(arm, relocate=False)
    scene = scene if scene is not None else BinScene()
    locator = MeasuringLocator(scene, arm=arm, log=log, measure_raises=measure_raises)
    ran = run(picks, place=PlaceAt(camera="yellow bin"), scope=scope, arm=arm, wrist=True, looks=(L1, L2),
              locators=[locator], **keywords)
    ran.locator, ran.scene = locator, scene  # type: ignore[attr-defined]
    return ran


def _line_in(ran: Any) -> Any:
    return [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]


def _checks(ran: Any) -> list[tuple[bool, str]]:
    return [(check["followed"], check["by"]) for check in ran.hooks.of("task.target_checked")]


class ABinThatStoodStillTests(unittest.TestCase):
    def test_a_bin_that_stood_still_is_followed_by_its_rim_and_the_detector_is_not_asked_again(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task()

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual([(True, "depth")], _checks(ran))
        self.assertAlmostEqual(0.0, ran.hooks.of("task.target_checked")[0]["moved_mm"], delta=1e-6)
        self.assertEqual(["yellow bin"], ran.locator.asked, "the detector was asked for the bin at the drop")
        self.assertEqual(1, len(ran.locator.measured))
        self.assertIn("the detector was not asked", ran.hooks.said[ran.names().index("task.target_checked")])
        np.testing.assert_allclose(BIN_CENTRE, _line_in(ran)[1][:2], atol=10.0)
        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 20.0, _line_in(ran)[1][2], delta=0.5)
        self.assertEqual(2, do0_changes(ran.log))

    def test_the_rim_is_read_at_the_bins_look_with_every_frame_held(self) -> None:
        ran = _task()

        carried = ran.after("task.carry_started")
        self.assertEqual(("joints", _key(L1)), motions(carried)[0])
        measure = next(index for index, entry in enumerate(carried) if entry[0] == "measure")
        self.assertLess(carried.index(("hold",)), measure)
        self.assertLess(measure, carried.index(("event", "task.drop_planned")))

    def test_until_empty_every_drop_into_a_bin_that_stood_still_reads_its_rim_alone(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(("part", "part", "empty", "empty"), scope="until_empty")

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        self.assertEqual(2, ran.report.parts_placed)
        self.assertEqual([(True, "depth"), (True, "depth")], _checks(ran))
        self.assertEqual(["yellow bin"], ran.locator.asked)
        self.assertEqual(4, do0_changes(ran.log))


class ABinThatChangedTests(unittest.TestCase):
    def test_a_bin_moved_40_mm_is_looked_for_by_the_detector_and_followed(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=lambda: scene.move(40.0)),), scene=scene)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual([(True, "detector")], _checks(ran))
        self.assertAlmostEqual(40.0, ran.hooks.of("task.target_checked")[0]["moved_mm"], delta=3.0)
        self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked)
        np.testing.assert_allclose((BIN_CENTRE[0] + 40.0, BIN_CENTRE[1]), _line_in(ran)[1][:2], atol=10.0)

    def test_a_bin_moved_too_far_taken_away_or_another_in_its_place_is_lost_as_before(self) -> None:
        from src.robot.execution.task import TaskStop

        changes = (
            ("moved 150 mm", lambda scene: scene.move(150.0), "moved_too_far"),
            ("taken away", lambda scene: scene.take_away(), "not_seen"),
            ("60 mm lower", lambda scene: scene.move(0.0, rim_mm=BIN_RIM_MM - 60.0), "footprint_changed"),
            ("another footprint", lambda scene: scene.move(0.0, size_xy=(200.0, 120.0)), "footprint_changed"),
            ("a blue bin in its place", lambda scene: setattr(scene, "bins", [SeenBin(label="blue bin",
                                                                                   colour_bgr=BLUE)]), "not_seen"),
        )
        for name, change, why in changes:
            with self.subTest(name):
                scene = BinScene()
                ran = _task((Pick("part", then=lambda change=change, scene=scene: change(scene)),), scene=scene,
                            relocate=False)

                self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
                self.assertEqual([(False, "detector")], _checks(ran))
                self.assertEqual([{"look": look_label(L1), "why": why, "phrase": "yellow bin"}],
                                 ran.hooks.of("task.target_lost"))
                self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked, "the detector was not asked")
                self.assertEqual(["task.put_back", "task.return_started", "task.returned", "task.part_finished"],
                                 ran.names()[-4:])
                self.assertEqual(2, do0_changes(ran.log), "the part was not let go where it was gripped")
                self.assertIsNone(ran.report.kept_target)

    def test_a_rim_more_than_a_fifth_hidden_asks_the_detector(self) -> None:
        """A box in front of the near wall, as a hand or the carried part, hides a quarter of the rim band."""
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=lambda: scene.in_front.append(((400.0, -365.0, 290.0), (90.0, 15.0, 10.0)))),),
                    scene=scene)

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        self.assertEqual([(True, "detector")], _checks(ran))
        self.assertEqual(["yellow bin", "yellow bin"], ran.locator.asked)

    def test_a_bin_that_creeps_60_mm_twice_is_lost_against_where_the_survey_found_it(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene()
        ran = _task((Pick("part", then=lambda: scene.move(60.0)), Pick("part", then=lambda: scene.move(60.0)),
                     "part"), scene=scene, scope="until_empty", relocate=False)

        self.assertIs(TaskStop.TARGET_LOST, ran.report.stop, ran.report.sentence)
        self.assertEqual(1, ran.report.parts_placed)
        self.assertEqual([(True, "detector"), (False, "detector")], _checks(ran))
        self.assertAlmostEqual(120.0, ran.hooks.of("task.target_checked")[1]["moved_mm"], delta=3.0)
        self.assertEqual("moved_too_far", ran.hooks.of("task.target_lost")[0]["why"])
        self.assertEqual(2, len(ran.locator.measured), "the rim was not read first before each drop")

    def test_a_frame_that_cannot_be_taken_for_the_rim_is_a_fault_of_the_cell(self) -> None:
        from src.robot.execution.task import TaskStop

        moved = PerceptionFrameMoved(camera="wrist", attempts=4, moved_mm=3.0, turned_deg=0.2, tolerance_mm=1.0,
                                     tolerance_deg=0.5)
        ran = _task(measure_raises=moved)

        self.assertIs(TaskStop.CELL_FAULT, ran.report.stop, ran.report.sentence)
        self.assertIn("PerceptionFrameMoved", ran.report.sentence)
        self.assertTrue(ran.report.holding)
        self.assertEqual(1, do0_changes(ran.log))
        self.assertEqual([], motions(ran.after("task.carry_started"))[1:], "a motion was sent after the fault")
        self.assertNotIn("task.target_checked", ran.names())


class WhenTheRimIsNotReadTests(unittest.TestCase):
    def test_a_locator_that_reads_no_depth_asks_the_detector_as_before(self) -> None:
        from src.robot.execution.place_target import KeptTarget, recheck

        kept = KeptTarget.of(bin_object(), camera="wrist", look=None, seen_at=1.0, phrase="blue bin")
        assert kept is not None
        locator = ScriptedLocator({"blue bin": [bin_object()]})

        check = recheck([locator], replace(kept, colour_lab=(80.0, 0.0, 75.0)))

        self.assertEqual(("", "detector"), (check.why, check.by))
        self.assertEqual(["blue bin"], locator.asked)

    def test_a_sighting_whose_rim_colour_was_never_read_asks_the_detector_and_takes_no_frame(self) -> None:
        from src.robot.execution.place_target import recheck, survey

        arm = TaskArm([], fk_table={_key(L1): AT_L1})
        locator = MeasuringLocator(BinScene(), arm=arm)
        kept = survey(arm, [locator], "yellow bin", looks=(L1,)).kept
        assert kept is not None and kept.colour_lab is not None, "the survey read no colour of the rim"

        check = recheck([locator], replace(kept, colour_lab=None))

        self.assertEqual(("", "detector"), (check.why, check.by))
        self.assertEqual([], locator.measured, "a frame was taken for a rim whose colour nobody read")
        self.assertEqual(["yellow bin", "yellow bin"], locator.asked)

    def test_a_sighting_followed_by_its_rim_draws_no_new_picture(self) -> None:
        from src.robot.execution.place_target import recheck, survey

        arm = TaskArm([], fk_table={_key(L1): AT_L1})
        locator = MeasuringLocator(BinScene(), arm=arm)
        kept = survey(arm, [locator], "yellow bin", looks=(L1,)).kept
        assert kept is not None

        check = recheck([locator], kept)

        self.assertEqual("depth", check.by)
        self.assertIs(kept, check.kept)
        assert check.seen is not None
        self.assertIsNone(check.seen.image_png, "the old sighting's picture was said again as a new one")
        self.assertIn("depth and colour", check.render())


class TheBoundsOnSyntheticDepthTests(unittest.TestCase):
    """A bin ray cast 900 mm under a camera looking straight down (``tests/_seen_scenes.py``): its rim band placed from
    one frame, read again in another."""

    def _bin(self, dx: float = 0.0) -> list[Any]:
        from tests._seen_scenes import open_bin

        return open_bin((450.0 + dx, -250.0), (240.0, 180.0), 110.0)

    def _kept(self) -> Any:
        from src.robot.execution.place_target import KeptTarget
        from src.robot.perception.locator import _lab_of_rgb
        from tests._seen_scenes import looking_down, render, seen_points

        camera = looking_down(450.0, -250.0, 900.0)
        kept = KeptTarget.of(bin_object(seen_points(camera, render(camera, self._bin())), label="yellow bin"),
                             camera="wrist", look=None, seen_at=1.0, phrase="yellow bin")
        assert kept is not None
        return replace(kept, colour_lab=tuple(float(v) for v in _lab_of_rgb([[230, 200, 20]])[0]))

    def _reading(self, *, dx: float = 0.0, noise_mm: float = 0.0) -> Any:
        from src.robot.perception.locator import _measured
        from tests._seen_scenes import HEIGHT, WIDTH, K, looking_down, render

        camera = looking_down(450.0, -250.0, 900.0)
        depth = render(camera, self._bin(dx))
        depth = depth + np.random.default_rng(7).uniform(-noise_mm, noise_mm, depth.shape) * (depth > 0.0)
        colour = np.empty((HEIGHT, WIDTH, 3), dtype=np.uint8)
        colour[...] = (230, 200, 20)

        class _Camera:
            def measure(self, points: Any) -> Any:
                return _measured(points, camera="wrist", captured_at_s=2.0, depth_mm=depth, intrinsics=K,
                                 camera_to_base=camera, rgb=colour)

        return _Camera()

    def test_a_rim_read_again_with_3_mm_of_noise_stands_where_it_stood(self) -> None:
        from src.robot.execution.place_target import _stands_where_it_stood

        self.assertEqual("", _stands_where_it_stood(self._reading(noise_mm=3.0), self._kept()))

    def test_a_bin_moved_5_mm_reads_unsure(self) -> None:
        from src.robot.execution.place_target import _stands_where_it_stood

        why = _stands_where_it_stood(self._reading(dx=5.0), self._kept())

        self.assertNotEqual("", why)
        self.assertRegex(why, "read where it stood|farther")

    def test_a_rim_of_another_colour_reads_unsure(self) -> None:
        from src.robot.execution.place_target import _stands_where_it_stood
        from src.robot.perception.locator import _lab_of_rgb

        blue = replace(self._kept(), colour_lab=tuple(float(v) for v in _lab_of_rgb([[20, 90, 200]])[0]))

        self.assertIn("colour", _stands_where_it_stood(self._reading(), blue))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
