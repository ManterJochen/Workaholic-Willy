"""A grasp is turned the way round its wrist camera keeps off what the camera saw, before the arm is asked.

The grasp bench (2026-10-05) lost its deep-bin scenes to the camera on the wrist: every grasp was turned the way the
cell's hand naturally stands, the camera came down on the bin's wall that way, and the guard refused a grasp whose twin,
half a turn about its approach, cleared the wall. ``wrist_turn`` places the guard's own housing at each grasp and along
its line in, both ways round, against the boxes the camera world builds of the neighbours. Pinned here:

* the housing stands on the TCP by the calibration's CAMERA to TOOL, whatever flange to TCP the body was placed from;
* a grasp whose camera meets the wall the way it was turned comes back turned half a turn, its twin clearing it;
* one whose camera clears it stays as it was, the very object;
* one whose camera meets it both ways keeps its turn and is tried after every one that clears, and the line in counts:
  a wall only the standoff's camera meets turns the grasp too;
* a grasp that is not in BASE is not measured, and keeps its place;
* the pick loop turns its grasps so, and not beside a named closing axis.
"""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.robot.grasping.geometry.wrist_turn import WristHousing, turned_for_the_wrist
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint

#: The owner's D415 on the TCP, near enough: 49 mm along -x, 84 mm along +y, 121 mm back up the approach.
_CAMERA_ON_TCP = (-49.0, 84.0, -121.0)
_HOUSING = WristHousing(centres=np.array([_CAMERA_ON_TCP]), turns=np.eye(3)[None, :, :],
                        halves=np.array([[55.0, 17.0, 15.0]]))


def _grasp(x: float, y: float, *, axis: tuple[float, float, float] = (0.0, -1.0, 0.0), score: float = 0.9,
           frame: GraspFrame = GraspFrame.BASE) -> GraspPoint:
    """A grasp straight down at (x, y, 80) closing along ``axis``."""
    return GraspPoint(position=np.array([x, y, 80.0]), approach=np.array([0.0, 0.0, -1.0]), axis=np.array(axis),
                      grip_width_mm=40.0, score=score, frame=frame)


def _wall(x: float, y: float, z: float, size: tuple[float, float, float]) -> Any:
    """A seen box square with BASE, as ``perceived.SeenBox`` holds one."""
    return SimpleNamespace(centre_mm=(x, y, z), rotation=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
                           half_extents_mm=(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))


def _camera_at(grasp: GraspPoint) -> np.ndarray:
    pose = np.asarray(grasp.pose().to_matrix(), dtype=np.float64)
    return pose[:3, :3] @ np.asarray(_CAMERA_ON_TCP) + pose[:3, 3]


class TheHousingStandsOnTheTcpTests(unittest.TestCase):
    def test_by_the_calibrations_camera_to_tool(self) -> None:
        camera_to_tool = np.eye(4)
        turn = math.radians(-28.0)
        camera_to_tool[:3, :3] = [[1.0, 0.0, 0.0], [0.0, math.cos(turn), -math.sin(turn)],
                                  [0.0, math.sin(turn), math.cos(turn)]]
        camera_to_tool[:3, 3] = (-48.8, 84.5, -121.1)
        flange_to_tcp = np.eye(4)
        flange_to_tcp[:3, :3] = [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        flange_to_tcp[:3, 3] = (-6.7, 2.9, 156.7)
        body = SimpleNamespace(placement=lambda: flange_to_tcp @ camera_to_tool, record_mm=flange_to_tcp,
                               boxes=(SimpleNamespace(centre_mm=(10.0, 0.0, 5.0), half_extents_mm=(54.5, 16.5, 15.0)),))

        housing = WristHousing.of([body])

        assert housing is not None
        np.testing.assert_allclose(housing.centres[0], (camera_to_tool @ [10.0, 0.0, 5.0, 1.0])[:3], atol=1e-9)
        np.testing.assert_allclose(housing.turns[0], camera_to_tool[:3, :3], atol=1e-12)
        np.testing.assert_allclose(housing.halves[0], (54.5, 16.5, 15.0))

    def test_an_arm_that_carries_no_camera_has_none(self) -> None:
        self.assertIsNone(WristHousing.of([]))


class AGraspIsTurnedOffTheWallTests(unittest.TestCase):
    def test_a_camera_on_the_wall_turns_the_grasp_half_a_turn(self) -> None:
        grasp = _grasp(0.0, 0.0)
        camera = _camera_at(grasp)
        # A wall just where the camera stands this way round: it stands on base -x, so the wall is there.
        wall = _wall(float(camera[0]), float(camera[1]), float(camera[2]), (20.0, 300.0, 400.0))

        ordered, turns = turned_for_the_wrist([grasp], _HOUSING, [wall], standoff_mm=80.0, distance_mm=3.0)

        self.assertEqual(1, turns.turned)
        np.testing.assert_allclose(ordered[0].axis, -grasp.axis)
        np.testing.assert_allclose(ordered[0].position, grasp.position)
        self.assertEqual(grasp.score, ordered[0].score)
        self.assertGreater(float(np.linalg.norm(_camera_at(ordered[0]) - camera)), 150.0, "the camera crossed sides")

    def test_a_camera_clear_of_the_wall_keeps_its_turn(self) -> None:
        grasp = _grasp(0.0, 0.0)
        wall = _wall(400.0, 0.0, 150.0, (20.0, 300.0, 300.0))

        ordered, turns = turned_for_the_wrist([grasp], _HOUSING, [wall], standoff_mm=80.0, distance_mm=3.0)

        self.assertIs(grasp, ordered[0])
        self.assertEqual((1, 0, 0), (turns.kept, turns.turned, turns.neither))

    def test_one_that_meets_the_wall_both_ways_is_tried_after_the_rest(self) -> None:
        boxed = _grasp(0.0, 0.0, score=0.95)
        free = _grasp(1000.0, 0.0, score=0.5)
        # A slab over both of the boxed grasp's camera places, nowhere near the free one.
        slab = _wall(0.0, 0.0, 201.0, (300.0, 300.0, 40.0))

        ordered, turns = turned_for_the_wrist([boxed, free], _HOUSING, [slab], standoff_mm=80.0, distance_mm=3.0)

        self.assertEqual([free, boxed], list(ordered))
        self.assertEqual((1, 0, 1), (turns.kept, turns.turned, turns.neither))

    def test_the_line_in_counts_a_wall_only_the_standoff_meets(self) -> None:
        grasp = _grasp(0.0, 0.0)
        camera = _camera_at(grasp)
        # Over the camera at the grasp, under it at the standoff 80 mm up: only the way in meets it.
        ledge = _wall(float(camera[0]), float(camera[1]), float(camera[2]) + 60.0, (40.0, 40.0, 20.0))

        at_the_grasp, _ = turned_for_the_wrist([grasp], _HOUSING, [ledge], standoff_mm=0.0, distance_mm=3.0)
        on_the_way, turns = turned_for_the_wrist([grasp], _HOUSING, [ledge], standoff_mm=80.0, distance_mm=3.0)

        self.assertIs(grasp, at_the_grasp[0])
        self.assertEqual(1, turns.turned)
        np.testing.assert_allclose(on_the_way[0].axis, -grasp.axis)

    def test_a_grasp_not_in_base_is_not_measured_and_keeps_its_place(self) -> None:
        in_camera = _grasp(0.0, 0.0, frame=GraspFrame.CAMERA)
        camera = _camera_at(in_camera)
        wall = _wall(float(camera[0]), float(camera[1]), float(camera[2]), (20.0, 300.0, 400.0))

        ordered, turns = turned_for_the_wrist([in_camera], _HOUSING, [wall], standoff_mm=80.0, distance_mm=3.0)

        self.assertIs(in_camera, ordered[0])
        self.assertEqual(1, turns.unmeasured)


class ThePickLoopTurnsItsGraspsTests(unittest.TestCase):
    """The pick loop's ``_closing_along``: turned beside the natural orientation, not beside a named axis."""

    def _loop(self, closing_axis: Any = None) -> Any:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        loop = object.__new__(BinPickingOrchestrator)
        loop.policy = SimpleNamespace(closing_axis=closing_axis, standoff_mm=80.0, align_closing_to_base_x=False)
        loop.natural_closing_axis = None
        loop.arm = SimpleNamespace(wrist_bodies=lambda: ())
        loop._wrist_housing_read = True
        loop._wrist_housing_cached = _HOUSING
        loop.calculator_for = lambda camera_id: None
        return loop

    def _result(self, grasp: GraspPoint, wall: Any) -> Any:
        from src.robot.grasping.types.feedback import GraspResult

        return GraspResult(candidates=(grasp,), reasons=(), telemetry={}, depth_confidence=1.0, mask_confidence=1.0,
                           top_score=grasp.score,
                           metadata={"scene_seen_boxes": (wall,), "scene_seen_distance_mm": 3.0})

    def test_beside_no_named_axis_the_camera_turns_the_grasp(self) -> None:
        grasp = _grasp(0.0, 0.0)
        camera = _camera_at(grasp)
        result = self._result(grasp, _wall(float(camera[0]), float(camera[1]), float(camera[2]), (20.0, 300.0, 400.0)))

        kept = self._loop()._closing_along(result, "wrist")

        np.testing.assert_allclose(kept.candidates[0].axis, -grasp.axis)

    def test_a_named_axis_keeps_its_way_round(self) -> None:
        grasp = _grasp(0.0, 0.0)
        camera = _camera_at(grasp)
        result = self._result(grasp, _wall(float(camera[0]), float(camera[1]), float(camera[2]), (20.0, 300.0, 400.0)))

        kept = self._loop(closing_axis="-y")._closing_along(result, "wrist")

        np.testing.assert_allclose(kept.candidates[0].axis, grasp.axis)


if __name__ == "__main__":
    unittest.main()
