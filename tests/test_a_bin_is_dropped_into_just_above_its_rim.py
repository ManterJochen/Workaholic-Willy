"""A part goes into a bin the camera found just above its rim, over the middle of its opening (OD 19, Q3, item 6).

What a task keeps of the bin it found (``KeptTarget``) is measured, never written: its rim is the 95th percentile of
the heights of what the camera saw of it (``SetDown``'s top, a stray pixel above it lifts nothing), its footprint and
its turn are the smallest rectangle about it (``height_map.turn_of``), and its opening is the gap between the inner
edges of its walls' tops, along each of its axes, where the inside reads lower than the rim; a box with a flat top has
none, and its part is set down on that top. An object with fewer than 20 points of surface is no target: it says where
a few pixels are, not where a surface is.

The drop (``drop_plan``) is ``SetDown.onto`` the bin, then four rules, in order:

* the air over the rim is the task's rim air, 20 mm unless the operator chose 10 to 50 (Q3): ``SET_DOWN_AIR_MM`` plus the
  perceived world's margin (``rim_clearance_mm``), so the carried part clears what the planner keeps around the rim,
  held to the owner's 10 to 50 mm whatever the margin;
* a grasp within 5 degrees of vertical is turned about the vertical until it closes along the cell's natural closing
  axis (``robot.natural_closing_axis``); any other keeps the grasp's own turn. Either way the tool keeps the tilt it
  gripped with: the hang is an upper bound on how far the part reaches below the tool only while the part hangs as it
  was gripped, and a turn about the vertical leaves every height under the tool as it was;
* the arm is asked where it would stand (``nearest_configuration``) and the exact guard and the planner screen that
  (``screen_configuration``): no configuration, or an ERROR, is unreachable;
* the part, turned as it will hang, must fit the opening (or the top), else it does not fit.

A taught place pose (``pose_drop``) says where the part's BOTTOM is let go: the tool goes to the taught pose raised in
BASE Z by the part's hang, the grasp's Z less the part's declared support, or the declared ``payload.length_mm`` where
the pick reports no grasp; turned as taught about the vertical, and tilted as the grasp was (the taught turn whole only
where the pick reports no grasp), then screened the same way. So the part's lowest point never ends below the taught
point, however the tool was tilted at the grasp.

The opening is read from the rim band alone, the walls' tops, across the middle half of the band's own extent, so a bin
seen at a slant (its far wall's inner face filling the view) still shows its inside: within a few millimetres of the
truth on the ray-cast bins of ``tests/_seen_scenes.py``.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.geometry.closing_axis import closing_axis_of
from src.robot.core import JointPositions
from src.robot.core.errors import RobotKinematicsError
from src.robot.safety.planning.band import PoseVerdict
from tests._task_fakes import (
    BIN_CENTRE,
    BIN_RIM_MM,
    BIN_SIZE,
    GRASP_Z_MM,
    PART_XY,
    PLACE_JOINTS,
    PLACE_TCP,
    TaskArm,
    bin_object,
    bin_points,
)


def _kept(points: "np.ndarray | None" = None, **keywords: Any) -> Any:
    from src.robot.execution.place_target import KeptTarget

    kept = KeptTarget.of(bin_object(points, **keywords), camera="wrist", look=None, seen_at=12.5)
    assert kept is not None
    return kept


def _grasp(tilt_deg: float = 0.0, *, yaw_deg: float = 0.0) -> Pose:
    """The pose the tool closed at on a 40 mm part: straight down closing along base x, tilted about base y."""
    down = Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM, yaw_deg=yaw_deg)
    if not tilt_deg:
        return down
    turn = math.radians(tilt_deg)
    tilt = np.array([[math.cos(turn), 0.0, math.sin(turn)], [0.0, 1.0, 0.0], [-math.sin(turn), 0.0, math.cos(turn)]])
    matrix = np.asarray(down.to_matrix(), dtype=np.float64).copy()
    matrix[:3, :3] = tilt @ matrix[:3, :3]
    return Pose.from_matrix(matrix, frame=Frame.BASE)


def _heading_deg(pose: Pose) -> float:
    tool_x = np.asarray(pose.to_matrix(), dtype=np.float64)[:3, 0]
    return math.degrees(math.atan2(float(tool_x[1]), float(tool_x[0]))) % 360.0


def _part(length_mm: float, width_mm: float = 40.0) -> np.ndarray:
    """The part's cloud where it stood when it was gripped, its long side along base x."""
    xs = np.linspace(-length_mm / 2.0, length_mm / 2.0, 30)
    ys = np.linspace(-width_mm / 2.0, width_mm / 2.0, 10)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([PART_XY[0] + gx.ravel(), PART_XY[1] + gy.ravel(), np.full(gx.size, 40.0)])


def _carried(solid_base_mm: np.ndarray, grasp: Pose, drop: Pose) -> np.ndarray:
    """Where the points of a part gripped at ``grasp`` stand when the tool stands at ``drop``: rigid in the jaws."""
    at_grasp = np.asarray(grasp.to_matrix(), dtype=np.float64)
    at_drop = np.asarray(drop.to_matrix(), dtype=np.float64)
    homogeneous = np.column_stack([solid_base_mm, np.ones(solid_base_mm.shape[0])])
    return (at_drop @ np.linalg.inv(at_grasp) @ homogeneous.T).T[:, :3]


def _off_vertical_deg(pose: Pose) -> float:
    tool_z = np.asarray(pose.to_matrix(), dtype=np.float64)[:3, 2]
    return math.degrees(math.acos(max(-1.0, min(1.0, -float(tool_z[2])))))


def _box(low: "tuple[float, float, float]", high: "tuple[float, float, float]", step_mm: float = 5.0) -> np.ndarray:
    """Points over a box's whole volume, its corners included, BASE mm."""
    axes = [np.append(np.arange(lo, hi, step_mm), hi) for lo, hi in zip(low, high)]
    gx, gy, gz = np.meshgrid(*axes)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


class WhatATaskKeepsOfItsBinTests(unittest.TestCase):
    def test_the_rim_the_middle_the_footprint_and_the_opening_are_measured(self) -> None:
        kept = _kept()

        self.assertAlmostEqual(BIN_RIM_MM, kept.rim_mm, delta=0.5)
        np.testing.assert_allclose(BIN_CENTRE, kept.centre_xy_mm, atol=3.0)
        np.testing.assert_allclose(sorted(BIN_SIZE), sorted(kept.footprint_mm), atol=10.0)
        assert kept.opening_mm is not None
        np.testing.assert_allclose(sorted((BIN_SIZE[0] - 16.0, BIN_SIZE[1] - 16.0)), sorted(kept.opening_mm),
                                   atol=10.0)
        self.assertEqual("blue bin", kept.label)
        self.assertEqual(0.8, kept.score)

    def test_a_turned_bin_is_measured_along_its_own_sides(self) -> None:
        kept = _kept(bin_points(yaw_deg=30.0))

        np.testing.assert_allclose(sorted(BIN_SIZE), sorted(kept.footprint_mm), atol=10.0)
        assert kept.opening_mm is not None
        np.testing.assert_allclose(sorted((BIN_SIZE[0] - 16.0, BIN_SIZE[1] - 16.0)), sorted(kept.opening_mm),
                                   atol=12.0)
        np.testing.assert_allclose(BIN_CENTRE, kept.centre_xy_mm, atol=3.0)

    def test_a_box_with_a_flat_top_has_no_opening(self) -> None:
        kept = _kept(bin_points(inside=False))
        self.assertIsNone(kept.opening_mm)
        self.assertAlmostEqual(BIN_RIM_MM, kept.rim_mm, delta=0.5)

    def test_an_object_with_fewer_than_20_points_is_no_target(self) -> None:
        from src.robot.execution.place_target import KeptTarget

        few = bin_points()[:19]
        self.assertIsNone(KeptTarget.of(bin_object(few), camera="wrist", look=None, seen_at=0.0))

    def test_its_keep_out_region_is_its_footprint_for_every_label(self) -> None:
        kept = _kept()
        region = kept.keep_out_region()

        self.assertIsNone(region.label)
        self.assertTrue(region.contains((*BIN_CENTRE, 50.0)))
        self.assertTrue(region.contains((BIN_CENTRE[0] + BIN_SIZE[0] / 2.0 - 2.0, BIN_CENTRE[1], 50.0)))
        self.assertFalse(region.contains((BIN_CENTRE[0] + BIN_SIZE[0] / 2.0 + 40.0, BIN_CENTRE[1], 50.0)))
        self.assertIn("blue bin", region.reason)

    def test_it_says_itself_as_the_console_shows_a_target(self) -> None:
        kept = _kept()
        said = kept.to_dict()

        self.assertEqual({"label", "score", "centre_mm", "rim_mm", "footprint_mm", "opening_mm", "look", "seen_at",
                          "camera", "points"}, set(said))
        self.assertEqual(3, len(said["centre_mm"]))
        self.assertAlmostEqual(BIN_RIM_MM, said["centre_mm"][2], delta=0.5)
        self.assertEqual(kept.render(), str(kept))
        self.assertIn("blue bin", kept.render())
        from api.schemas import TargetOut

        TargetOut.model_validate({key: value for key, value in said.items() if key in TargetOut.model_fields})


class TheRimClearanceTests(unittest.TestCase):
    def test_it_is_the_set_down_air_and_the_perceived_worlds_margin_20_mm_as_shipped(self) -> None:
        from src.config import load_tree
        from src.robot.execution.place_target import rim_clearance_mm

        robot = load_tree("console_dummy").robot
        self.assertEqual(20.0, rim_clearance_mm(robot))
        margin = SimpleNamespace(safety=SimpleNamespace(planning_world=SimpleNamespace(
            perceived=SimpleNamespace(margin_mm=25.0))))
        self.assertEqual(30.0, rim_clearance_mm(margin))
        self.assertEqual(20.0, rim_clearance_mm(None), "a cell that says nothing gets the owner's 20 mm")

    def test_the_cells_own_air_is_held_to_the_owners_10_to_50_mm(self) -> None:
        """Q3 bounds the air over a rim at 10 to 50 mm; the schema lets the margin run from 0 to 500 mm. Red before:
        the cell's default ran from 5 to 505 mm."""
        from src.robot.execution.place_target import rim_clearance_mm

        for margin_mm, air_mm in ((0.0, 10.0), (4.0, 10.0), (5.0, 10.0), (15.0, 20.0), (45.0, 50.0), (60.0, 50.0),
                                  (500.0, 50.0)):
            with self.subTest(margin_mm=margin_mm):
                tree = SimpleNamespace(safety=SimpleNamespace(planning_world=SimpleNamespace(
                    perceived=SimpleNamespace(margin_mm=margin_mm))))
                self.assertEqual(air_mm, rim_clearance_mm(tree))


class TheDropOverTheRimTests(unittest.TestCase):
    def _plan(self, kept: Any = None, grasp: "Pose | None" = None, *, arm: "TaskArm | None" = None, air: float = 20.0,
              natural: Any = None, part: "np.ndarray | None" = None) -> Any:
        from src.robot.execution.place_target import drop_plan

        return drop_plan(arm or TaskArm([]), kept if kept is not None else _kept(), grasp or _grasp(),
                         part_bottom_mm=0.0, air_mm=air, standoff_mm=80.0, natural_axis=natural, part_cloud_mm=part)

    def test_the_part_is_let_go_over_the_middle_of_the_opening_at_rim_hang_and_air(self) -> None:
        plan = self._plan()

        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual("camera", plan.kind)
        assert plan.pose is not None
        np.testing.assert_allclose(BIN_CENTRE, plan.pose.position_mm[:2], atol=10.0)
        self.assertAlmostEqual(GRASP_Z_MM, plan.hang_mm)
        self.assertAlmostEqual(20.0, plan.air_mm)
        self.assertAlmostEqual(BIN_RIM_MM, plan.rim_mm, delta=0.5)
        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 20.0, float(plan.pose.position_mm[2]), delta=0.5)
        self.assertEqual(80.0, plan.standoff_mm)
        self.assertEqual("clear", plan.verdict)
        self.assertIsNotNone(plan.joints)

    def test_the_rim_air_is_the_operators_choice(self) -> None:
        plan = self._plan(air=45.0)
        assert plan.pose is not None
        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 45.0, float(plan.pose.position_mm[2]), delta=0.5)

    def test_a_vertical_grasp_is_turned_along_the_cells_natural_axis(self) -> None:
        """Turned about the vertical alone: it closes along the natural axis, tilted as it gripped. Red before for the
        tilted one: the drop was stood exactly upright, a turn about a level axis the hang does not bound."""
        for tilt in (0.0, 4.0):
            with self.subTest(tilt=tilt):
                grasp = _grasp(tilt, yaw_deg=20.0)
                plan = self._plan(grasp=grasp, natural=closing_axis_of("-y"))
                assert plan.pose is not None
                self.assertAlmostEqual(270.0, _heading_deg(plan.pose), delta=0.01)
                self.assertAlmostEqual(_off_vertical_deg(grasp), _off_vertical_deg(plan.pose), delta=1e-6)
                self.assertTrue(plan.turned)

    def test_a_turned_drop_lets_no_part_hang_below_the_rim_and_its_air(self) -> None:
        """A 200 mm bar lying on the bench, gripped 20 mm from its end at mid-height, the tool tilted 4.9 degrees along
        it: the drop is turned along -y, and the bar's lowest point still stands at least the air over the rim. Red
        before: stood upright, the bar's far end swung 15 mm down, 5.3 mm below the rim with 10 mm of air promised."""
        bar = _box((PART_XY[0] - 20.0, PART_XY[1] - 15.0, 0.0), (PART_XY[0] + 180.0, PART_XY[1] + 15.0, 20.0))
        for tilt in (0.0, 2.5, 4.9):
            with self.subTest(tilt=tilt):
                grasp = _grasp(tilt)
                grasp = Pose(position_mm=np.array([PART_XY[0], PART_XY[1], 10.0]),
                             quaternion_xyzw=grasp.quaternion_xyzw, frame=Frame.BASE)
                plan = self._plan(grasp=grasp, air=10.0, natural=closing_axis_of("-y"))
                assert plan.pose is not None and plan.rim_mm is not None
                lowest = float(_carried(bar, grasp, plan.pose)[:, 2].min())
                self.assertGreaterEqual(lowest, plan.rim_mm + 10.0 - 1e-6)
                self.assertTrue(plan.turned)

    def test_a_grasp_past_5_degrees_keeps_its_own_turn(self) -> None:
        grasp = _grasp(10.0, yaw_deg=20.0)
        plan = self._plan(grasp=grasp, natural=closing_axis_of("-y"))

        assert plan.pose is not None
        np.testing.assert_allclose(grasp.quaternion_xyzw, plan.pose.quaternion_xyzw, atol=1e-12)
        self.assertFalse(plan.turned)

    def test_with_no_natural_axis_the_grasp_keeps_its_turn(self) -> None:
        grasp = _grasp(yaw_deg=20.0)
        plan = self._plan(grasp=grasp)
        assert plan.pose is not None
        np.testing.assert_allclose(grasp.quaternion_xyzw, plan.pose.quaternion_xyzw, atol=1e-12)

    def test_no_configuration_and_an_error_verdict_are_unreachable(self) -> None:
        out_of_reach = self._plan(arm=TaskArm([], unreachable_above_mm=100.0))
        self.assertFalse(out_of_reach.ok)
        self.assertEqual("unreachable", out_of_reach.refusal)
        self.assertIn("out of reach", out_of_reach.reason)

        arm = TaskArm([])
        clear = self._plan(arm=arm)
        assert clear.joints is not None
        refused = self._plan(arm=TaskArm([], screens={
            tuple(round(v, 1) for v in clear.joints.degrees()): PoseVerdict.GUARD_REFUSED}))
        self.assertEqual("unreachable", refused.refusal)
        self.assertEqual("guard_refused", refused.verdict)

    def test_a_band_verdict_is_reachable(self) -> None:
        arm = TaskArm([])
        clear = self._plan(arm=arm)
        assert clear.joints is not None
        band = self._plan(arm=TaskArm([], screens={
            tuple(round(v, 1) for v in clear.joints.degrees()): PoseVerdict.BAND}))
        self.assertTrue(band.ok)
        self.assertEqual("band", band.verdict)

    def test_a_part_longer_than_the_opening_does_not_fit_and_a_short_one_does(self) -> None:
        long = self._plan(part=_part(320.0))
        self.assertEqual("does_not_fit", long.refusal)
        self.assertIn("opening", long.reason)
        short = self._plan(part=_part(40.0))
        self.assertTrue(short.ok, short.reason)
        self.assertEqual("fits", short.fit)

    def test_the_part_is_judged_turned_as_it_will_hang(self) -> None:
        """A part 250 mm long fits the bin's 284 mm along base x but not its 184 mm along base y: turned a quarter turn
        by the natural axis (gripped closing along x, dropped closing along -y), it no longer fits."""
        straight = self._plan(part=_part(250.0))
        self.assertTrue(straight.ok, straight.reason)
        turned = self._plan(part=_part(250.0), natural=closing_axis_of("-y"))
        self.assertEqual("does_not_fit", turned.refusal)

    def test_without_the_parts_cloud_the_fit_is_not_judged_and_says_so(self) -> None:
        plan = self._plan()
        self.assertTrue(plan.ok)
        self.assertEqual("not_judged", plan.fit)

    def test_a_flat_top_takes_the_part_on_it(self) -> None:
        plan = self._plan(kept=_kept(bin_points(inside=False)), part=_part(40.0))
        self.assertTrue(plan.ok, plan.reason)
        assert plan.pose is not None
        self.assertAlmostEqual(BIN_RIM_MM + GRASP_Z_MM + 20.0, float(plan.pose.position_mm[2]), delta=0.5)

    def test_a_grasp_that_did_not_stand_above_the_part_has_no_hang_and_is_unreachable(self) -> None:
        from src.robot.execution.place_target import drop_plan

        plan = drop_plan(TaskArm([]), _kept(), _grasp(), part_bottom_mm=GRASP_Z_MM + 5.0, air_mm=20.0,
                         standoff_mm=80.0)
        self.assertFalse(plan.ok)
        self.assertEqual("unreachable", plan.refusal)
        self.assertIsNone(plan.pose)

    def test_the_drop_reads_as_the_console_says_it(self) -> None:
        plan = self._plan()
        said = plan.to_event()
        self.assertEqual({"kind", "pose_mm", "rim_mm", "hang_mm", "air_mm", "verdict"}, set(said))
        self.assertEqual("camera", said["kind"])
        self.assertEqual(plan.render(), str(plan))


class TheDropAtATaughtPoseTests(unittest.TestCase):
    def test_the_drop_stands_the_hang_above_the_taught_pose_and_keeps_its_turn(self) -> None:
        from src.robot.execution.place_target import pose_drop

        plan = pose_drop(TaskArm([]), PLACE_JOINTS, _grasp(), part_bottom_mm=0.0, length_mm=60.0, standoff_mm=80.0)

        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual("pose", plan.kind)
        assert plan.pose is not None
        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), plan.pose.position_mm)
        np.testing.assert_allclose(PLACE_TCP.quaternion_xyzw, plan.pose.quaternion_xyzw)
        self.assertAlmostEqual(GRASP_Z_MM, plan.hang_mm)
        self.assertIsNone(plan.rim_mm)
        self.assertEqual("clear", plan.verdict)

    def test_a_tilted_grasp_keeps_its_tilt_so_the_parts_bottom_never_goes_below_the_taught_point(self) -> None:
        """A 40 mm cube on the bench gripped at its centre, the tool tilted about its closing axis: the drop keeps that
        tilt, turned about the vertical to the taught heading, and the cube's lowest corner ends at the taught point,
        where its bottom is let go. Red before: the drop took the taught turn whole, and the corner swung down, 7.3 mm
        below the taught point at a 30 degree tilt."""
        from src.robot.execution.place_target import pose_drop

        cube = _box((PART_XY[0] - 20.0, PART_XY[1] - 20.0, 0.0), (PART_XY[0] + 20.0, PART_XY[1] + 20.0, 40.0))
        taught_z = float(PLACE_TCP.position_mm[2])
        for tilt, yaw in ((5.0, 0.0), (15.0, 0.0), (30.0, 0.0), (45.0, 0.0), (30.0, 35.0)):
            with self.subTest(tilt=tilt, yaw=yaw):
                down = Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM, yaw_deg=yaw)
                turn = math.radians(tilt)
                about_x = np.array([[1.0, 0.0, 0.0], [0.0, math.cos(turn), -math.sin(turn)],
                                    [0.0, math.sin(turn), math.cos(turn)]])
                matrix = np.asarray(down.to_matrix(), dtype=np.float64).copy()
                matrix[:3, :3] = matrix[:3, :3] @ about_x
                grasp = Pose.from_matrix(matrix, frame=Frame.BASE)

                plan = pose_drop(TaskArm([]), PLACE_JOINTS, grasp, part_bottom_mm=0.0, length_mm=None,
                                 standoff_mm=80.0)

                assert plan.pose is not None
                lowest = float(_carried(cube, grasp, plan.pose)[:, 2].min())
                self.assertGreaterEqual(lowest, taught_z - 1e-6)
                self.assertAlmostEqual(taught_z, lowest, delta=1e-6, msg="the cube's bottom was not let go there")
                gap = (_heading_deg(PLACE_TCP) - _heading_deg(plan.pose) + 180.0) % 360.0 - 180.0
                self.assertAlmostEqual(0.0, gap, delta=1e-6, msg="the drop is not turned as taught")
                self.assertAlmostEqual(_off_vertical_deg(grasp), _off_vertical_deg(plan.pose), delta=1e-6)
                np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, GRASP_Z_MM), plan.pose.position_mm)

    def test_a_grasp_whose_closing_axis_points_near_the_vertical_keeps_its_own_turn(self) -> None:
        """A grasp tilted so far that its closing axis heads nowhere on the table has no heading to turn to the taught
        one: the drop keeps the grasp's turn whole, which still keeps every height under the tool."""
        from src.robot.execution.place_target import pose_drop

        down = np.asarray(Pose.tool_down(PART_XY[0], PART_XY[1], GRASP_Z_MM).to_matrix(), dtype=np.float64).copy()
        turn = math.radians(85.0)
        about_y = np.array([[math.cos(turn), 0.0, math.sin(turn)], [0.0, 1.0, 0.0],
                            [-math.sin(turn), 0.0, math.cos(turn)]])
        down[:3, :3] = down[:3, :3] @ about_y
        grasp = Pose.from_matrix(down, frame=Frame.BASE)

        plan = pose_drop(TaskArm([]), PLACE_JOINTS, grasp, part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)

        assert plan.pose is not None
        np.testing.assert_allclose(grasp.quaternion_xyzw, plan.pose.quaternion_xyzw, atol=1e-12)
        cube = _box((PART_XY[0] - 20.0, PART_XY[1] - 20.0, 0.0), (PART_XY[0] + 20.0, PART_XY[1] + 20.0, 40.0))
        self.assertGreaterEqual(float(_carried(cube, grasp, plan.pose)[:, 2].min()),
                                float(PLACE_TCP.position_mm[2]) - 1e-6)

    def test_a_grasp_that_did_not_stand_above_the_parts_bottom_has_no_hang(self) -> None:
        from src.robot.execution.place_target import pose_drop

        plan = pose_drop(TaskArm([]), PLACE_JOINTS, _grasp(), part_bottom_mm=GRASP_Z_MM + 5.0, length_mm=60.0,
                         standoff_mm=80.0)
        self.assertEqual("unreachable", plan.refusal)
        self.assertIsNone(plan.pose)
        self.assertIn("hang is unknown", plan.reason)

    def test_without_a_grasp_pose_the_declared_length_stands_in_for_the_hang(self) -> None:
        from src.robot.execution.place_target import pose_drop

        plan = pose_drop(TaskArm([]), PLACE_JOINTS, None, part_bottom_mm=0.0, length_mm=60.0, standoff_mm=80.0)
        assert plan.pose is not None
        np.testing.assert_allclose(PLACE_TCP.position_mm + (0.0, 0.0, 60.0), plan.pose.position_mm)
        self.assertEqual(60.0, plan.hang_mm)

    def test_with_neither_the_hang_is_unknown_and_nothing_is_set_down_blind(self) -> None:
        from src.robot.execution.place_target import pose_drop

        plan = pose_drop(TaskArm([]), PLACE_JOINTS, None, part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)
        self.assertFalse(plan.ok)
        self.assertEqual("unreachable", plan.refusal)
        self.assertIn("length_mm", plan.reason)

    def test_an_error_on_the_raised_drop_is_unreachable(self) -> None:
        from src.robot.execution.place_target import pose_drop

        arm = TaskArm([])
        clear = pose_drop(arm, PLACE_JOINTS, _grasp(), part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)
        assert clear.joints is not None
        key = tuple(round(v, 1) for v in clear.joints.degrees())
        refused = pose_drop(TaskArm([], screens={key: PoseVerdict.PLANNER_REFUSED}), PLACE_JOINTS, _grasp(),
                            part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)
        self.assertEqual("unreachable", refused.refusal)
        self.assertEqual("planner_refused", refused.verdict)

    def test_an_arm_that_screens_nothing_drops_unscreened(self) -> None:
        from src.robot.execution.place_target import pose_drop

        class _Plain:
            def fk(self, joints: JointPositions) -> Pose:
                return PLACE_TCP

        plan = pose_drop(_Plain(), PLACE_JOINTS, _grasp(), part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual("unscreened", plan.verdict)

    def test_a_taught_pose_the_arm_cannot_say_is_unreachable(self) -> None:
        from src.robot.execution.place_target import pose_drop

        class _NoFk:
            def fk(self, joints: JointPositions) -> Pose:
                raise RobotKinematicsError("FK failed")

        plan = pose_drop(_NoFk(), PLACE_JOINTS, _grasp(), part_bottom_mm=0.0, length_mm=None, standoff_mm=80.0)
        self.assertEqual("unreachable", plan.refusal)
        self.assertIn("FK failed", plan.reason)


class ABinSeenThroughACameraTests(unittest.TestCase):
    """The bin of ``tests/_seen_scenes.py``, ray cast through a camera as a wrist look sees it: from 45 degrees beside
    it its near wall's top, its far wall's inner face and top, part of its floor; from above its walls' tops. 240 x 180
    mm outside, walls 5 mm thick, so 230 x 170 mm inside."""

    CENTRE, SIZE, RIM = (450.0, -250.0), (240.0, 180.0), 110.0
    VIEWS = {
        "45 degrees from -x": (450.0 - 380.0, -250.0, 110.0 + 380.0),
        "45 degrees from +y": (450.0, -250.0 + 380.0, 110.0 + 380.0),
        "30 degrees off vertical": (450.0 - 220.0, -250.0, 110.0 + 380.0),
        "straight down": (450.0 + 1.0, -250.0, 110.0 + 500.0),
        "60 degrees, low": (450.0 - 450.0, -250.0, 110.0 + 260.0),
    }

    def _kept(self, eye: "tuple[float, float, float]", *, yaw_deg: float = 0.0, closed: bool = False) -> Any:
        from src.robot.execution.place_target import KeptTarget
        from tests._seen_scenes import Solid, looking_at, open_bin, render, seen_points

        camera = looking_at(eye, (450.0, -250.0, 40.0))
        solids = ([Solid((*self.CENTRE, self.RIM / 2.0), (self.SIZE[0] / 2.0, self.SIZE[1] / 2.0, self.RIM / 2.0),
                         yaw_deg)] if closed else open_bin(self.CENTRE, self.SIZE, self.RIM, yaw_deg=yaw_deg))
        kept = KeptTarget.of(bin_object(seen_points(camera, render(camera, solids))), camera="wrist", look=None,
                             seen_at=0.0)
        assert kept is not None
        return kept

    def test_the_rim_the_middle_and_the_opening_hold_from_every_view(self) -> None:
        """Red before: the opening's strip was the middle half of every point's spread, so in a slanted view the far
        wall's inner face filled it, a wall's top spanned it, and no bin seen by a camera showed an opening at all."""
        inside = sorted((self.SIZE[0] - 10.0, self.SIZE[1] - 10.0))
        for yaw in (0.0, 25.0):
            for view, eye in self.VIEWS.items():
                with self.subTest(view=view, yaw=yaw):
                    kept = self._kept(eye, yaw_deg=yaw)

                    self.assertAlmostEqual(self.RIM, kept.rim_mm, delta=3.0)
                    np.testing.assert_allclose(self.CENTRE, kept.centre_xy_mm, atol=10.0)
                    self.assertIsNotNone(kept.opening_mm, "a bin seen through a camera showed no opening")
                    np.testing.assert_allclose(inside, sorted(kept.opening_mm), atol=3.0)

    def test_a_closed_box_seen_the_same_ways_has_no_opening(self) -> None:
        for view in ("45 degrees from -x", "straight down"):
            with self.subTest(view=view):
                self.assertIsNone(self._kept(self.VIEWS[view], closed=True).opening_mm)

    def test_a_part_between_the_bins_inside_and_outside_does_not_fit_it(self) -> None:
        from src.robot.execution.place_target import drop_plan

        kept = self._kept(self.VIEWS["45 degrees from -x"])
        for length, fit in ((200.0, "fits"), (228.0, "fits"), (235.0, "does_not_fit"), (245.0, "does_not_fit")):
            with self.subTest(length=length):
                plan = drop_plan(TaskArm([]), kept, _grasp(), part_bottom_mm=0.0, air_mm=20.0, standoff_mm=80.0,
                                 part_cloud_mm=_part(length))
                self.assertEqual(fit, plan.fit, plan.reason)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
