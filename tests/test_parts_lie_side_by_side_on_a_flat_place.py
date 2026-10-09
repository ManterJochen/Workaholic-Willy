"""Parts set down on a flat place lie side by side, never on top of each other (the owner, 2026-10-08 night).

"Sonst tuermen wir die Bauteile": with ``robot.place.side_by_side`` on, a place that is no box (a taught pose, or a
target the camera found with no inside) lays each part at a spot of its own. The spots are a grid (``robot.place.grid``:
rows x columns about the taught pose along its heading, or across a target's top about its middle), each part's reach
(its cloud's, or the open hand's where that is more) and ``spacing_margin_mm`` apart, the nearest the middle first. The
camera reads the spots first (the pick's own looks for a taught pose, the check before the drop for a target): one it
reads anything on, or whose support is another surface, is passed over; one it reads free is taken. Where the camera
reads nothing, a spot is taken that no part this task laid there stands near. A place with no spot left puts the part
back where it was gripped and asks (``part_does_not_fit``). An "until empty" task keeps each laid part out of its picks.
As shipped, every part goes to the one spot, as before.
"""

from __future__ import annotations

import math
import unittest
from typing import Any

import numpy as np

from src.geometry import Pose
from tests._place_fakes import FlatSupport, cube_cloud, hande_tree, pick_view, run_placing, what_the_looks_saw
from tests._task_fakes import (
    GRASP_Z_MM,
    PLACE_TCP,
    BinScene,
    ScriptedLocator,
    TaskArm,
    bin_object,
    bin_points,
    do0_changes,
    motions,
)

TAUGHT_XY = (float(PLACE_TCP.position_mm[0]), float(PLACE_TCP.position_mm[1]))
#: Where the camera of the pick's look over the place stood: straight over the taught pose.
OVER_THE_PLACE = Pose.tool_down(TAUGHT_XY[0], TAUGHT_XY[1], 450.0, label="over the place")


def _line_ins(ran: Any) -> list[np.ndarray]:
    """Where every place's line in went, in order."""
    marks = [index for index, entry in enumerate(ran.log) if entry == ("event", "task.place_started")]
    ins = []
    for mark in marks:
        moves = [m for m in motions(ran.log[mark + 1:]) if m[0] == "move"]
        ins.append(np.asarray(moves[1][1], dtype=np.float64))
    return ins


class TheGridTests(unittest.TestCase):
    def test_the_spots_stand_about_the_middle_the_nearest_first(self) -> None:
        from src.robot.execution.place_target import grid_spots

        spots = grid_spots((0.0, 0.0), 0.0, rows=3, columns=3, pitch_mm=100.0)

        self.assertEqual(9, len(spots))
        self.assertEqual((0.0, 0.0), spots[0])
        self.assertEqual({(-100.0, 0.0), (0.0, -100.0), (0.0, 100.0), (100.0, 0.0)},
                         {(round(x), round(y)) for x, y in spots[1:5]})
        self.assertTrue(all(math.isclose(math.hypot(x, y), 100.0 * math.sqrt(2.0)) for x, y in spots[5:]))

    def test_the_grid_runs_along_the_taught_heading(self) -> None:
        from src.robot.execution.place_target import grid_spots

        spots = grid_spots((10.0, 20.0), math.radians(90.0), rows=2, columns=1, pitch_mm=50.0)

        np.testing.assert_allclose([(10.0, -5.0), (10.0, 45.0)], sorted(spots, key=lambda xy: xy[1]), atol=1e-9)

    def test_a_flat_tops_spots_keep_the_part_on_it(self) -> None:
        from src.robot.execution.place_target import KeptTarget, top_spots

        top = KeptTarget.of(bin_object(bin_points(inside=False)), camera="wrist", look=None, seen_at=0.0)
        assert top is not None
        spots = top_spots(top, pitch_mm=80.0, reach_mm=30.0)

        self.assertGreater(len(spots), 1)
        np.testing.assert_allclose(top.centre_xy_mm, spots[0], atol=1e-6)
        for x, y in spots:
            self.assertLessEqual(abs(x - top.centre_xy_mm[0]) + 30.0, top.footprint_mm[0] / 2.0 + 1.0)
            self.assertLessEqual(abs(y - top.centre_xy_mm[1]) + 30.0, top.footprint_mm[1] / 2.0 + 1.0)


class ChoosingASpotTests(unittest.TestCase):
    def test_a_spot_near_a_part_laid_there_is_passed_over(self) -> None:
        from src.robot.execution.place_target import choose_spot

        spot, _why = choose_spot([(0.0, 0.0), (60.0, 0.0), (120.0, 0.0)], reach_mm=30.0, margin_mm=20.0,
                                 placed=[((0.0, 0.0), 30.0)])

        assert spot is not None
        self.assertEqual(((120.0, 0.0), "grid"), (spot.xy, spot.by))

    def test_the_camera_passes_over_what_it_reads_taken_and_takes_what_it_reads_free(self) -> None:
        from src.robot.execution.place_target import choose_spot

        verdicts = {(0.0, 0.0): "taken", (100.0, 0.0): "free"}
        spot, _why = choose_spot([(0.0, 0.0), (100.0, 0.0)], reach_mm=30.0, margin_mm=20.0,
                                 read=lambda xy, _r: (verdicts[xy], 0.0))

        assert spot is not None
        self.assertEqual(((100.0, 0.0), "camera"), (spot.xy, spot.by))

    def test_no_spot_left_says_why(self) -> None:
        from src.robot.execution.place_target import choose_spot

        spot, why = choose_spot([(0.0, 0.0), (100.0, 0.0)], reach_mm=30.0, margin_mm=20.0,
                                placed=[((0.0, 0.0), 30.0)], read=lambda _xy, _r: ("taken", 0.0))

        self.assertIsNone(spot)
        self.assertIn("1 hold a part this task laid there", why)
        self.assertIn("1 more taken", why)


class TheCameraReadsTheSpotsTests(unittest.TestCase):
    def _reader(self, scene: BinScene, support: Any = None) -> Any:
        from src.robot.execution.place_target import look_points, views_spot_reader

        view = pick_view(scene, OVER_THE_PLACE, "over the place")
        return views_spot_reader(look_points([view]), FlatSupport(0.0) if support is None else support, TAUGHT_XY)

    def test_a_part_lying_at_a_spot_takes_it_and_the_bench_beside_it_is_free(self) -> None:
        read = self._reader(BinScene(bins=[], in_front=[((TAUGHT_XY[0], TAUGHT_XY[1], 20.0), (20.0, 20.0, 20.0))]))

        self.assertEqual("taken", read(TAUGHT_XY, 50.0)[0])
        self.assertEqual("free", read((TAUGHT_XY[0] + 120.0, TAUGHT_XY[1]), 50.0)[0])

    def test_a_spot_on_another_surface_is_no_spot_of_the_place(self) -> None:
        mat = FlatSupport(0.0, x_mm=(TAUGHT_XY[0] - 80.0, TAUGHT_XY[0] + 80.0))
        read = self._reader(BinScene(bins=[]), support=mat)

        self.assertEqual("free", read(TAUGHT_XY, 50.0)[0])
        self.assertEqual("taken", read((TAUGHT_XY[0] + 200.0, TAUGHT_XY[1]), 50.0)[0])

    def test_a_spot_the_looks_did_not_see_is_unread(self) -> None:
        read = self._reader(BinScene(bins=[]))

        self.assertEqual("unread", read((TAUGHT_XY[0], TAUGHT_XY[1] - 600.0), 50.0)[0])

    def test_no_camera_reads_a_place_no_surface_lies_under(self) -> None:
        from src.robot.execution.place_target import views_spot_reader

        self.assertIsNone(views_spot_reader(np.zeros((0, 3)), FlatSupport(0.0), TAUGHT_XY))
        self.assertIsNone(views_spot_reader(np.ones((10, 3)), FlatSupport(0.0, x_mm=(900.0, 950.0)), TAUGHT_XY))

    def test_a_flat_tops_spots_are_read_from_what_the_check_read_of_it(self) -> None:
        from src.robot.execution.place_target import InsideRead, top_spot_reader

        xs, ys = np.meshgrid(np.arange(-150.0, 150.0, 4.0), np.arange(-100.0, 100.0, 4.0))
        points = np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, 120.0)])
        covered = np.hypot(points[:, 0], points[:, 1]) <= 25.0
        read = top_spot_reader(InsideRead("top", points, covered=covered, seen=np.ones(points.shape[0], dtype=bool)))

        assert read is not None
        self.assertEqual("taken", read((0.0, 0.0), 40.0)[0])
        self.assertEqual("free", read((90.0, 0.0), 40.0)[0])
        self.assertIsNone(top_spot_reader(None))


def _side_by_side_arm(**place: Any) -> TaskArm:
    log: list[Any] = []
    arm = TaskArm(log)
    arm.config = hande_tree(side_by_side=True, **place)  # type: ignore[attr-defined]
    return arm


class ATaskLaysItsPartsSideBySideTests(unittest.TestCase):
    def test_parts_at_a_taught_pose_lie_side_by_side_on_its_grid(self) -> None:
        from src.robot.execution.task import TaskStop

        arm = _side_by_side_arm()
        ran = run_placing(("part", "part", "part", "empty", "empty"), arm=arm, scope="until_empty",
                          looked_around=what_the_looks_saw(cloud=cube_cloud()))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        ins = _line_ins(ran)
        self.assertEqual(3, len(ins))
        np.testing.assert_allclose(TAUGHT_XY, ins[0][:2], atol=1e-6)
        reach = max(26.0, 38.0)
        for a in range(3):
            self.assertAlmostEqual(float(PLACE_TCP.position_mm[2]) + GRASP_Z_MM, ins[a][2], delta=0.5)
            for b in range(a + 1, 3):
                self.assertGreaterEqual(float(np.hypot(*(ins[a][:2] - ins[b][:2]))), 2.0 * reach + 20.0)
        said = [event["spot_by"] for event in ran.hooks.of("task.drop_planned")]
        self.assertEqual(["grid", "grid", "grid"], said)
        self.assertEqual(6, do0_changes(ran.log))

    def test_an_until_empty_task_keeps_every_part_it_laid_out_of_its_picks(self) -> None:
        arm = _side_by_side_arm()
        ran = run_placing(("part", "part", "empty", "empty"), arm=arm, scope="until_empty",
                          looked_around=what_the_looks_saw(cloud=cube_cloud()))

        ins = _line_ins(ran)
        last = ran.service.zones_seen[-1]
        for laid in ins:
            self.assertTrue(any(region.contains((laid[0], laid[1], 10.0)) for region in last))

    def test_a_spot_the_camera_reads_a_part_on_is_passed_over(self) -> None:
        scene = BinScene(bins=[], in_front=[((TAUGHT_XY[0], TAUGHT_XY[1], 20.0), (20.0, 20.0, 20.0))])
        arm = _side_by_side_arm()
        looked = what_the_looks_saw(cloud=cube_cloud(), support=FlatSupport(0.0),
                                    views=[pick_view(scene, OVER_THE_PLACE, "over the place")])
        ran = run_placing(("part",), arm=arm, looked_around=looked)

        laid = _line_ins(ran)[0]
        self.assertGreater(float(np.hypot(laid[0] - TAUGHT_XY[0], laid[1] - TAUGHT_XY[1])), 50.0)
        self.assertEqual("camera", ran.hooks.of("task.drop_planned")[0]["spot_by"])

    def test_a_place_with_no_spot_left_puts_the_part_back_and_asks(self) -> None:
        from src.robot.execution.task import TaskStop

        scene = BinScene(bins=[], in_front=[((TAUGHT_XY[0], TAUGHT_XY[1], 20.0), (20.0, 20.0, 20.0))])
        arm = _side_by_side_arm(grid={"rows": 1, "columns": 1})
        looked = what_the_looks_saw(cloud=cube_cloud(), support=FlatSupport(0.0),
                                    views=[pick_view(scene, OVER_THE_PLACE, "over the place")])
        ran = run_placing(("part",), arm=arm, looked_around=looked)

        self.assertIs(TaskStop.PART_DOES_NOT_FIT, ran.report.stop, ran.report.sentence)
        self.assertIn("no spot", ran.report.sentence)
        self.assertNotIn("task.place_started", ran.names())
        self.assertEqual(["task.put_back", "task.return_started", "task.returned", "task.part_finished"],
                         ran.names()[-4:])
        self.assertEqual(2, do0_changes(ran.log))

    def test_as_shipped_every_part_is_let_go_at_the_taught_pose(self) -> None:
        log: list[Any] = []
        arm = TaskArm(log)
        arm.config = hande_tree()  # type: ignore[attr-defined]
        ran = run_placing(("part", "part", "empty", "empty"), arm=arm, scope="until_empty",
                          looked_around=what_the_looks_saw(cloud=cube_cloud()))

        for laid in _line_ins(ran):
            np.testing.assert_allclose(TAUGHT_XY, laid[:2], atol=1e-6)
        self.assertNotIn("spot_mm", ran.hooks.of("task.drop_planned")[0])

    def test_parts_on_a_flat_target_the_camera_found_lie_side_by_side_on_its_top(self) -> None:
        from src.robot.execution.task import PlaceAt, TaskStop

        flat = bin_object(bin_points(inside=False), label="grey plate")
        arm = _side_by_side_arm()
        locator = ScriptedLocator({"grey plate": lambda _tcp: [flat]}, arm=arm, log=arm.log, wrist=False,
                                  rig_id="overhead")
        ran = run_placing(("part", "part", "empty", "empty"), arm=arm, scope="until_empty",
                          place=PlaceAt(camera="grey plate", air_mm=20.0), locators=[locator],
                          looked_around=what_the_looks_saw(cloud=cube_cloud()))

        self.assertIs(TaskStop.NOTHING_LEFT, ran.report.stop, ran.report.sentence)
        ins = _line_ins(ran)
        self.assertEqual(2, len(ins))
        self.assertGreater(float(np.hypot(*(ins[0][:2] - ins[1][:2]))), 2.0 * 38.0)
        self.assertEqual(["grid", "grid"], [e["spot_by"] for e in ran.hooks.of("task.drop_planned")])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
