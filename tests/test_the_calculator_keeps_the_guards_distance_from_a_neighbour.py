"""The calculator keeps the guard's distance from what the camera saw beside the part (the owner, 2026-10-03).

The camera world holds a neighbour as boxes: its points clustered, each cluster a height map of columns grown by
``planning_world.perceived.margin_mm`` (15) and carried down to the bench, and the exact guard keeps
``perceived_min_distance_mm`` (5) from every one. The calculator's own obstacle grid, 12 mm cells dilated once, let an
open finger come about 12 mm from a neighbour's points: on URSim (2026-10-03) a blocker's grasp stood 0.48 mm inside the
part's box at its standoff, and the arm went down for nothing. Now the calculator holds the boxes the world builds of the
same points (``scene_obstacles.seen_boxes``) and keeps the guard's distance from them with the open hand, at the grasp
and on its way in: what it offers beside a neighbour is what the guard admits there.

The world here is built from the neighbours' pixels of the very frame the calculator reads (``build_perceived_boxes``,
the owner's tuning), the open hand is the Hand-E's ``collision_boxes`` at full stroke, and the way in reaches 80 mm back
along the approach.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.grasping.generation.scene_obstacles import box_distances_mm
from src.robot.safety.planning.perceived import (
    DepthView,
    WorldBuildLimits,
    WorldBuildTuning,
    build_perceived_boxes,
)
from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import (
    AIM,
    BENCH,
    EYE,
    FOCAL,
    HEIGHT,
    MAT,
    MAT_MM,
    PART,
    WIDTH,
    Frame_,
    _hande,
    calculator,
    compute,
)
from tests.test_a_side_grasp_never_goes_through_unseen_space import Box, DepthCamera

#: What the guard keeps from a camera box on the owner's cell, millimetres.
GUARD_MM = 5.0
#: How far back along the approach the open hand is asked, millimetres: the default standoff.
WAY_IN_MM = (0.0, 40.0, 80.0)
_LIMITS = WorldBuildLimits(x_mm=(-700.0, 700.0), y_mm=(-1100.0, 0.0), z_mm=(-100.0, 700.0), support_plane_top_mm=0.0)
_TUNING = WorldBuildTuning(pixel_stride=2, voxel_size_mm=10.0, cluster_voxel_mm=25.0, min_points=12, margin_mm=15.0,
                           max_boxes=64, voxel_field_mm=0.0, floor_to_plane=True)


def cube(x_mm: float, y_mm: float, side_mm: float = 40.0, height_mm: float = 40.0) -> Box:
    return Box(lo=(x_mm - side_mm / 2.0, y_mm - side_mm / 2.0, MAT_MM),
               hi=(x_mm + side_mm / 2.0, y_mm + side_mm / 2.0, MAT_MM + height_mm))


def on_x(gap_mm: float, *, both: bool = True) -> tuple[Box, ...]:
    """Cubes ``gap_mm`` off the part along x, on its -x side or on both."""
    off = 20.0 + gap_mm + 20.0
    return tuple(cube(sx * off, -650.0) for sx in ((-1.0, 1.0) if both else (-1.0,)))


def l_shape(cube_gap_mm: float, block_gap_mm: float) -> tuple[Box, ...]:
    """A 40 mm cube off the part's -x side and a block wider than the hand opens off its +y side."""
    return (cube(-(20.0 + cube_gap_mm + 20.0), -650.0),
            cube(0.0, -650.0 + 20.0 + block_gap_mm + 30.0, side_mm=60.0, height_mm=40.0))


def world_boxes(neighbours: tuple[Box, ...], frame: Frame_) -> list[Any]:
    """The boxes the camera world holds for the neighbours' pixels of ``frame``: what is nearer than the scene with no
    neighbour in it, built by the world's own rules."""
    full = DepthCamera(eye=EYE, target=AIM, solids=(BENCH, MAT, *neighbours, PART), width=WIDTH, height=HEIGHT,
                       focal_px=FOCAL).depth
    bare = DepthCamera(eye=EYE, target=AIM, solids=(BENCH, MAT, PART), width=WIDTH, height=HEIGHT,
                       focal_px=FOCAL).depth
    seen = np.isfinite(full) & (~np.isfinite(bare) | (full < bare - 0.5))
    view = DepthView(surface_depth_mm=np.where(seen, full, 0.0), intrinsics=frame.intrinsics,
                     camera_to_base=frame.camera_to_base, name="wrist")
    return list(build_perceived_boxes(views=[view], limits=_LIMITS, tuning=_TUNING).boxes)


def least_mm(grasp: Any, boxes: list[Any]) -> float:
    """The open Hand-E's least distance to ``boxes`` at ``grasp`` and back along its approach, millimetres."""
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry

    cfg = _hande()
    hand = build_gripper_geometry(cfg.grasping.gripper_geometry).collision_boxes(float(cfg.gripper.max_width_mm))
    local_c = np.array([(np.asarray(b.min_corner_mm) + np.asarray(b.max_corner_mm)) / 2.0 for b in hand])
    local_h = np.array([(np.asarray(b.max_corner_mm) - np.asarray(b.min_corner_mm)) / 2.0 for b in hand])
    approach = np.asarray(grasp.approach, dtype=np.float64)
    turn = np.column_stack([grasp.axis, np.cross(approach, grasp.axis), approach])
    centres_b = np.array([b.center_mm for b in boxes], dtype=np.float64)
    turns_b = np.array([[[np.cos(b.yaw_rad), -np.sin(b.yaw_rad), 0.0], [np.sin(b.yaw_rad), np.cos(b.yaw_rad), 0.0],
                         [0.0, 0.0, 1.0]] for b in boxes], dtype=np.float64)
    halves_b = np.array([np.asarray(b.dims_mm, dtype=np.float64) / 2.0 for b in boxes])
    least = np.inf
    for back in WAY_IN_MM:
        centres = np.asarray(grasp.position, dtype=np.float64) - back * approach + local_c @ turn.T
        n, m = centres.shape[0], len(boxes)
        d = box_distances_mm(np.repeat(centres, m, axis=0), np.repeat(turn[None], n * m, axis=0),
                             np.repeat(local_h, m, axis=0), np.tile(centres_b, (n, 1)), np.tile(turns_b, (n, 1, 1)),
                             np.tile(halves_b, (n, 1)))
        least = min(least, float(d.min()))
    return least


SCENES = {
    **{f"cubes {gap:g} mm off on both sides": on_x(gap) for gap in (5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0)},
    **{f"a cube {gap:g} mm off on one side": on_x(gap, both=False) for gap in (8.0, 15.0, 20.0)},
    "a cube 20 mm off -x, a block 15 mm off +y": l_shape(20.0, 15.0),
    "a cube 10 mm off -x, a block 10 mm off +y": l_shape(10.0, 10.0),
}


class NoGraspComesNearerANeighbourThanTheGuardKeepsTests(unittest.TestCase):

    def test_every_offered_grasp_keeps_the_guards_distance_from_the_worlds_boxes(self) -> None:
        for name, neighbours in SCENES.items():
            with self.subTest(name):
                frame = Frame_((BENCH, MAT, *neighbours), PART)
                boxes = world_boxes(neighbours, frame)
                self.assertTrue(boxes, "the world holds no box of the neighbours: nothing was asked")
                result = compute(calculator(scene=True), frame)
                near = [round(least_mm(g, boxes), 2) for g in result.candidates]
                self.assertTrue(all(d >= GUARD_MM - 0.05 for d in near),
                                f"{name}: open-hand distances to the world's boxes {near} mm, the guard keeps {GUARD_MM}")

    def test_room_on_both_sides_keeps_the_grasps_across_it(self) -> None:
        result = compute(calculator(scene=True), Frame_((BENCH, MAT, *on_x(25.0)), PART))
        self.assertGreater(len(result.candidates), 0)
        self.assertTrue(all(abs(float(g.axis[0])) < 0.5 for g in result.candidates),
                        [np.round(g.axis, 2).tolist() for g in result.candidates])

    def test_room_the_guard_admits_is_not_refused_by_the_grid(self) -> None:
        """Cubes 16 mm off both sides leave the open fingers room the guard admits beside the world's boxes. The grid,
        12 mm cells grown once, refused every finger within 12 to 24 mm of a point, and with it these grasps; where the
        boxes decide, it keeps the hand off the points alone (the grasp bench, 2026-10-05)."""
        frame = Frame_((BENCH, MAT, *on_x(16.0)), PART)

        result = compute(calculator(scene=True), frame)

        self.assertGreater(len(result.candidates), 0)
        boxes = world_boxes(on_x(16.0), frame)
        near = [round(least_mm(g, boxes), 2) for g in result.candidates]
        self.assertTrue(all(d >= GUARD_MM - 0.05 for d in near), near)

    def test_a_part_the_boxes_close_in_says_a_neighbour_is_in_the_way(self) -> None:
        from src.robot.grasping.types.feedback import GraspFailureReason

        result = compute(calculator(scene=True), Frame_((BENCH, MAT, *on_x(8.0)), PART))
        self.assertEqual([], list(result.candidates))
        self.assertIn(GraspFailureReason.ALL_COLLIDED, result.reasons)


class TheCalculatorsBoxesCoverTheWorldsTests(unittest.TestCase):
    """``seen_boxes`` builds the world's boxes of every point it reads, where the world thins them to one a 10 mm cell
    first: its boxes are the world's or larger, never smaller, so the distance it keeps from them holds at the guard."""

    def test_every_world_box_lies_within_the_calculators_boxes(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import seen_boxes
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import owner_rules

        rules = owner_rules(margin_mm=15.0, voxel_size_mm=10.0, floor_to_plane=True)
        for name, neighbours in SCENES.items():
            with self.subTest(name):
                frame = Frame_((BENCH, MAT, *neighbours), PART)
                world = world_boxes(neighbours, frame)
                full = DepthCamera(eye=EYE, target=AIM, solids=(BENCH, MAT, *neighbours, PART), width=WIDTH,
                                   height=HEIGHT, focal_px=FOCAL).depth
                bare = DepthCamera(eye=EYE, target=AIM, solids=(BENCH, MAT, PART), width=WIDTH, height=HEIGHT,
                                   focal_px=FOCAL).depth
                seen = np.isfinite(full) & (~np.isfinite(bare) | (full < bare - 0.5))
                seen[1::2, :] = False
                seen[:, 1::2] = False
                rows, cols = np.nonzero(seen)
                z = full[rows, cols]
                points = (np.column_stack(((cols - WIDTH / 2.0) * z / FOCAL, (rows - HEIGHT / 2.0) * z / FOCAL, z))
                          @ frame.camera_to_base[:3, :3].T + frame.camera_to_base[:3, 3])
                mine = seen_boxes(points, rules)
                outside = [tuple(np.round(p, 1)) for a in world for p in _grid(a) if not _inside_any(p, mine)]
                self.assertEqual([], outside[:5], f"{name}: {len(outside)} point(s) of the world's boxes in none of "
                                                  "the calculator's")


def _grid(box: Any, n: int = 5) -> np.ndarray:
    """``n`` points along each axis of a world box, its faces included, BASE millimetres."""
    half = np.asarray(box.dims_mm, dtype=np.float64) / 2.0
    c, s = np.cos(box.yaw_rad), np.sin(box.yaw_rad)
    turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    t = np.linspace(-1.0, 1.0, n)
    local = np.array([(a, b, d) for a in t for b in t for d in t]) * half
    return np.asarray(box.center_mm, dtype=np.float64) + local @ turn.T


def _inside_any(point: np.ndarray, boxes: Any, tolerance_mm: float = 1.0) -> bool:
    for b in boxes:
        local = (point - np.asarray(b.centre_mm)) @ np.asarray(b.rotation, dtype=np.float64)
        if np.all(np.abs(local) <= np.asarray(b.half_extents_mm) + tolerance_mm):
            return True
    return False


if __name__ == "__main__":
    unittest.main()
