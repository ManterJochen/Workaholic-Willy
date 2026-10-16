"""A bin low in front is a box, and the parts spread in it (the owner, 2026-10-15 and 16).

The owner's small bins of 2026-10-15 stand low in front: their back and sides at 125 to 130 mm, the front lip at 75. The
opening reader asked for two walls at the rim on each axis, read "no opening seen" at every look, and the bin was set
down on as a flat top. Every part went over its middle, and the ninth drop of a run was refused and the run ended. The
owner: the bin is read stably, and the parts spread, built in, no switch ("immer verteilen"): each part over the freest
spot the check reads in the box, the next where a spot's drop is refused, 15 mm higher where every one is, and only then
back where it was gripped. The frame the check read over the bin is kept beside the picks', so a refused drop can be
read again.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from src.geometry import Pose
from src.robot.core import JointPositions
from src.robot.core.errors import RobotKinematicsError
from tests._place_fakes import hande_tree
from tests._seen_scenes import Solid, looking_at, open_bin, render, seen_points
from tests._task_fakes import BIN_CENTRE, BIN_RIM_MM, GRASP_Z_MM, BinScene, MeasuringLocator, TaskArm, bin_object, motions, run

L1 = JointPositions.deg(-20.0, -90.0, -110.0, -60.0, 90.0, 0.0)
L2 = JointPositions.deg(10.0, -90.0, -110.0, -60.0, 90.0, 0.0)
AT_L1 = Pose.tool_down(400.0, -300.0, 450.0, label="L1")
AT_L2 = Pose.tool_down(150.0, -650.0, 450.0, label="L2")

#: The bin of these tests: 240 x 160 mm outside, walls 6 mm thick, its rim 120 mm, its front (-x) wall 50 mm high.
CENTRE, SIZE, RIM, WALL, LIP = (450.0, -250.0), (240.0, 160.0), 120.0, 6.0, 50.0
VIEWS = {
    "45 degrees from -x": (450.0 - 380.0, -250.0, 120.0 + 380.0),
    "45 degrees from +x": (450.0 + 380.0, -250.0, 120.0 + 380.0),
    "45 degrees from +y": (450.0, -250.0 + 380.0, 120.0 + 380.0),
    "straight down": (450.0 + 1.0, -250.0, 120.0 + 500.0),
}


def _key(joints: JointPositions) -> tuple[float, ...]:
    return tuple(round(v, 1) for v in joints.degrees())


def _low_in_front(yaw_deg: float = 0.0) -> "list[Solid]":
    """The bin with its front (-x) wall :data:`LIP` mm high, its back and its sides at the rim, a 5 mm floor."""
    half_x, half_y = SIZE[0] / 2.0, SIZE[1] / 2.0
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def at(u: float, v: float, z: float) -> "tuple[float, float, float]":
        return (CENTRE[0] + c * u - s * v, CENTRE[1] + s * u + c * v, z)

    wall = WALL / 2.0
    return [
        Solid(at(-half_x + wall, 0.0, LIP / 2.0), (wall, half_y, LIP / 2.0), yaw_deg),
        Solid(at(half_x - wall, 0.0, RIM / 2.0), (wall, half_y, RIM / 2.0), yaw_deg),
        Solid(at(0.0, -half_y + wall, RIM / 2.0), (half_x, wall, RIM / 2.0), yaw_deg),
        Solid(at(0.0, half_y - wall, RIM / 2.0), (half_x, wall, RIM / 2.0), yaw_deg),
        Solid(at(0.0, 0.0, 2.5), (half_x, half_y, 2.5), yaw_deg),
    ]


def _kept(solids: "list[Solid]", eye: "tuple[float, float, float]") -> Any:
    from src.robot.execution.place_target import KeptTarget

    camera = looking_at(eye, (*CENTRE, 40.0))
    kept = KeptTarget.of(bin_object(seen_points(camera, render(camera, solids))), camera="wrist", look=None,
                         seen_at=0.0)
    assert kept is not None
    return kept


class ABinLowInFrontIsABoxTests(unittest.TestCase):
    def test_its_opening_runs_from_its_back_wall_to_its_front_from_every_view(self) -> None:
        """Red before: "no opening seen" from every view, as at every look of the owner's small bins."""
        inside = (SIZE[0] - WALL, SIZE[1] - 2.0 * WALL)
        for yaw in (0.0, 20.0):
            for view, eye in VIEWS.items():
                with self.subTest(view=view, yaw=yaw):
                    kept = _kept(_low_in_front(yaw), eye)

                    self.assertIsNotNone(kept.opening_mm, f"a bin low in front showed no opening: {kept.render()}")
                    self.assertAlmostEqual(RIM, kept.rim_mm, delta=3.0)
                    along, across = kept.opening_mm if kept.footprint_mm[0] >= kept.footprint_mm[1] \
                        else kept.opening_mm[::-1]
                    self.assertAlmostEqual(inside[0], along, delta=8.0, msg=kept.render())
                    self.assertAlmostEqual(inside[1], across, delta=4.0, msg=kept.render())

    def test_a_closed_box_and_an_open_bin_read_as_they_did(self) -> None:
        closed = [Solid((*CENTRE, RIM / 2.0), (SIZE[0] / 2.0, SIZE[1] / 2.0, RIM / 2.0), 0.0)]
        self.assertIsNone(_kept(closed, VIEWS["straight down"]).opening_mm)
        open_one = _kept(open_bin(CENTRE, SIZE, RIM, wall_mm=WALL, floor_mm=5.0), VIEWS["45 degrees from -x"])
        assert open_one.opening_mm is not None
        np.testing.assert_allclose(sorted((SIZE[0] - 2.0 * WALL, SIZE[1] - 2.0 * WALL)), sorted(open_one.opening_mm),
                                   atol=3.0)

    def test_a_tall_part_standing_in_it_narrows_its_opening_never_widens_it(self) -> None:
        solids = _low_in_front() + [Solid((CENTRE[0] - 40.0, CENTRE[1], RIM / 2.0), (10.0, 10.0, RIM / 2.0), 0.0)]
        kept = _kept(solids, VIEWS["straight down"])
        if kept.opening_mm is not None:
            self.assertLess(max(kept.opening_mm), SIZE[0] - WALL - 30.0, kept.render())


class TheFreestSpotOfABoxTests(unittest.TestCase):
    """``place_target.inside_spots``: the spots of a box's inside, the lowest the check read within a part's reach of
    each first."""

    def _inside(self, kept: Any, top_of: Any) -> Any:
        from src.robot.execution.place_target import InsideRead

        grid = np.array([(x, y, 5.0) for x in np.arange(CENTRE[0] - 100.0, CENTRE[0] + 101.0, 10.0)
                         for y in np.arange(CENTRE[1] - 60.0, CENTRE[1] + 61.0, 10.0)])
        tops = np.array([top_of(x, y) for x, y, _ in grid])
        return InsideRead("box", grid, top_mm=float(tops.max()), said="scripted", tops_mm=tops)

    def test_the_half_a_pile_fills_is_left_for_the_free_one(self) -> None:
        from src.robot.execution.place_target import INSIDE_SPOTS_TRIED, inside_spots

        kept = _kept(open_bin(CENTRE, SIZE, RIM, wall_mm=WALL, floor_mm=5.0), VIEWS["straight down"])
        inside = self._inside(kept, lambda x, y: 80.0 if x > CENTRE[0] else 5.0)

        spots = inside_spots(kept, inside, reach_mm=30.0, margin_mm=10.0)

        self.assertTrue(spots)
        self.assertLessEqual(len(spots), INSIDE_SPOTS_TRIED)
        self.assertLess(spots[0].xy[0], CENTRE[0] - 30.0, "the first spot is not in the free half")
        self.assertEqual(5.0, spots[0].top_mm)
        self.assertEqual(sorted(s.top_mm for s in spots), [s.top_mm for s in spots])

    def test_a_box_whose_inside_was_not_read_has_no_spot_and_the_part_goes_over_its_middle(self) -> None:
        from src.robot.execution.place_target import inside_spots

        kept = _kept(open_bin(CENTRE, SIZE, RIM, wall_mm=WALL, floor_mm=5.0), VIEWS["straight down"])
        self.assertEqual([], inside_spots(kept, None, reach_mm=30.0))


class _RefusedAtTheRimAir(TaskArm):
    """The arm, refusing every pose the tool stands at one of :attr:`refused_z` (BASE z mm) for: the drops at an air."""

    refused_z: "tuple[float, ...]" = ()

    def nearest_configuration(self, pose: Pose) -> JointPositions:
        if any(abs(float(pose.position_mm[2]) - z) < 0.5 for z in self.refused_z):
            raise RobotKinematicsError(f"{pose.label or 'pose'}: refused by the endpoint gate: scripted")
        return super().nearest_configuration(pose)


def _task(*, refused_z: "tuple[float, ...]" = ()) -> Any:
    from src.robot.execution.task import PlaceAt

    log: list[Any] = []
    arm = _RefusedAtTheRimAir(log, fk_table={_key(L1): AT_L1, _key(L2): AT_L2})
    arm.refused_z = refused_z
    arm.config = hande_tree(release_in_a_box="over_the_rim", length_mm=60.0)  # type: ignore[attr-defined]
    scene = BinScene()
    locator = MeasuringLocator(scene, arm=arm, log=log)
    ran = run(("part",), place=PlaceAt(camera="yellow bin", air_mm=20.0), arm=arm, wrist=True, looks=(L1, L2),
              locators=[locator])
    ran.locator = locator  # type: ignore[attr-defined]
    return ran


class ARefusedDropTriesTheBoxAgainTests(unittest.TestCase):
    def test_every_spot_refused_at_the_rim_air_is_asked_15_mm_higher_before_the_part_goes_back(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(refused_z=(BIN_RIM_MM + 20.0 + GRASP_Z_MM,))

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        line_in = [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]
        self.assertAlmostEqual(BIN_RIM_MM + 35.0 + GRASP_Z_MM, line_in[1][2], delta=0.5)

    def test_a_drop_refused_at_every_spot_and_both_heights_puts_the_part_back_as_before(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task(refused_z=(BIN_RIM_MM + 20.0 + GRASP_Z_MM, BIN_RIM_MM + 35.0 + GRASP_Z_MM))

        self.assertIs(TaskStop.TARGET_UNREACHABLE, ran.report.stop, ran.report.sentence)
        self.assertIn("put back", ran.report.sentence + " ".join(str(s) for s in ran.hooks.said))

    def test_a_drop_the_arm_reaches_goes_over_the_freest_spot_at_the_rim_air(self) -> None:
        from src.robot.execution.task import TaskStop

        ran = _task()

        self.assertIs(TaskStop.FINISHED, ran.report.stop, ran.report.sentence)
        line_in = [m for m in motions(ran.after("task.place_started")) if m[0] == "move"][1]
        self.assertAlmostEqual(BIN_RIM_MM + 20.0 + GRASP_Z_MM, line_in[1][2], delta=0.5)
        self.assertLess(math.hypot(line_in[1][0] - BIN_CENTRE[0], line_in[1][1] - BIN_CENTRE[1]), 150.0)


class TheFrameOverTheBinIsKeptTests(unittest.TestCase):
    def test_the_frame_the_check_measured_is_written_beside_the_picks_with_the_bins_points(self) -> None:
        from src.robot.execution import record_views
        from src.robot.execution.place_target import record_the_check
        from src.robot.perception.locator import MeasuredLook

        kept = _kept(open_bin(CENTRE, SIZE, RIM, wall_mm=WALL, floor_mm=5.0), VIEWS["straight down"])
        frame = type("Frame", (), {"depth_map": np.full((4, 5), 600.0), "intrinsics": np.eye(3), "rgb": None,
                                   "surface_depth_map": None, "tool_pose": None})()
        locator = type("Locator", (), {"rig_id": "wrist", "last_measured": MeasuredLook(
            frame=frame, camera_to_base=np.eye(4), name="wrist")})()

        written = record_the_check([locator], kept, name="place-yellow bin")

        assert written is not None
        self.assertEqual(Path(record_views.RECORD_VIEWS_DIR) / "places", written.parent)
        with np.load(written, allow_pickle=False) as data:
            self.assertEqual((4, 5), data["depth_0"].shape)
            self.assertEqual(kept.points_base_mm.shape[0], data["target_cloud_base_mm"].shape[0])

    def test_no_frame_measured_writes_nothing(self) -> None:
        from src.robot.execution.place_target import record_the_check

        kept = _kept(open_bin(CENTRE, SIZE, RIM, wall_mm=WALL, floor_mm=5.0), VIEWS["straight down"])
        self.assertIsNone(record_the_check([type("Locator", (), {"rig_id": "wrist"})()], kept, name="place-x"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
