"""The parts beside the ones a pick named are read off the depth, and the fingers may brush them (the owner, 2026-10-12).

On the owner's crate (2026-10-12 13:45) "the gray cube" grounded the gray cube alone, and the green cube 30 mm from it
was a box nobody named: the fingers kept the guard's distance and the margin from it, and no grasp was admitted. The
owner's answer, "alle Teile benennen", with three guards: part-sized, standing where the parts stand, not the target.
Qwen asked for every object boxed the mat or the whole group on half the cell's frames and took 0.6 s a part, so the
owner chose the depth (2026-10-12: "Klassisch aus Tiefe"):

* what stands over the supports, outside the named masks and their rims, falls into pieces; a piece seen whole, apart
  from anything larger, near a named part and of a part's shape is a part beside (``parts_beside``);
* each rule that leaves a piece out says why (``why_no_part_beside``): a wall seen from above, a strip, a corner, a
  low lid, a hovering hand, a far part, one at the image's edge and one touching a wall stay boxes nobody named;
* the pick puts them on the frame (``PerceptionFrame.parts_beside``) where the cell asks and outside the regions its
  task keeps out, and the planner world's named parts and the calculator's neighbours read them there.
"""

from __future__ import annotations

import types
import unittest
from typing import Any

import numpy as np

from src.robot.grasping.generation.parts_beside import LABEL, parts_beside, why_no_part_beside
from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator, _named_points_base
from src.robot.grasping.types.perception import PerceptionFrame

H, W = 480, 640
F = 600.0
K = np.array([[F, 0.0, W / 2.0], [0.0, F, H / 2.0], [0.0, 0.0, 1.0]])
#: The camera 700 mm over the bench looking straight down: BASE x = camera x, BASE y = -camera y, BASE z = 700 - depth.
CAMERA_TO_BASE = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 700.0], [0.0, 0.0, 0.0, 1.0]])


def _px(x_mm: float, y_mm: float, z_mm: float = 0.0) -> "tuple[int, int]":
    """The pixel (row, column) a BASE point over the bench at height ``z_mm`` falls on."""
    depth = 700.0 - z_mm
    return int(round(H / 2.0 - y_mm * F / depth)), int(round(W / 2.0 + x_mm * F / depth))


class _Scene:
    """The bench at 700 mm and boxes standing on it, each its footprint in BASE (x0, y0, x1, y1) and its height."""

    def __init__(self) -> None:
        self.depth = np.full((H, W), 700.0)

    def box(self, x0: float, y0: float, x1: float, y1: float, top_mm: float) -> np.ndarray:
        r0, c0 = _px(x0, y1, top_mm)
        r1, c1 = _px(x1, y0, top_mm)
        mask = np.zeros((H, W), dtype=bool)
        mask[max(r0, 0):min(r1, H), max(c0, 0):min(c1, W)] = True
        self.depth[mask] = np.minimum(self.depth[mask], 700.0 - top_mm)
        return mask


def _found(scene: _Scene, named: "list[np.ndarray]") -> "tuple[list[Any], dict[str, int]]":
    return parts_beside(scene.depth, K, CAMERA_TO_BASE, named, support_model=None, bench_mm=0.0)


def _near(parts: "list[Any]", x_mm: float, y_mm: float) -> bool:
    return any(abs(part.centre_mm[0] - x_mm) < 8.0 and abs(part.centre_mm[1] - y_mm) < 8.0 for part in parts)


class TheDepthShowsThePartsBesideTests(unittest.TestCase):
    def test_the_cubes_beside_the_named_one_are_parts_and_nothing_else_is(self) -> None:
        scene = _Scene()
        target = scene.box(-20, -20, 20, 20, 40)  # the gray cube, named
        scene.box(50, -20, 90, 20, 30)  # 30 mm right of it
        scene.box(-15, -85, 15, -55, 25)  # 35 mm under it
        scene.box(-20, 180, 20, 220, 40)  # 160 mm away: no finger meets it
        scene.box(-100, -60, -92, 60, 80)  # a wall seen from above, 8 mm thick
        scene.box(-80, -100, -40, -60, 8)  # a lid 8 mm over the bench
        found, left = _found(scene, [target])
        self.assertEqual(2, len(found), (left, [part.centre_mm for part in found]))
        self.assertTrue(_near(found, 70, 0) and _near(found, 0, -70), [part.centre_mm for part in found])
        self.assertEqual({"too far from the parts named": 1, "thinner than a part": 1, "lower than a part": 1}, left)
        # Each part beside is a mask of the frame, of its own pixels and not the named cube's.
        for part in found:
            self.assertEqual((H, W), part.mask.shape)
            self.assertFalse(bool((part.mask & target).any()))

    def test_a_piece_of_a_wall_a_part_at_the_edge_or_a_hand_stays_unnamed(self) -> None:
        scene = _Scene()
        target = scene.box(-20, -20, 20, 20, 40)
        scene.box(-300, -100, 20, -92, 100)  # a wall longer than a part
        scene.box(24, -100, 80, -92, 100)  # a piece of it the depth broke off, 4 mm on
        top, left_col = _px(-262, 190, 40)
        scene.depth[:top + 30, :left_col + 30] = 660.0  # a part the image's corner cuts
        scene.box(40, 20, 80, 60, 420)  # a hand 420 mm over the bench
        found, left = _found(scene, [target])
        self.assertEqual([], found, left)
        self.assertEqual({"larger than a part": 1, "a piece of something larger": 1,
                          "seen only in part, at the image's edge": 1, "it hovers over the parts": 1}, left)

    def test_nothing_named_or_nothing_read_finds_nothing(self) -> None:
        scene = _Scene()
        scene.box(50, -20, 90, 20, 30)
        self.assertEqual(([], {}), _found(scene, []))
        target = scene.box(-20, -20, 20, 20, 40)
        self.assertEqual(([], {}), parts_beside(np.zeros((H, W)), K, CAMERA_TO_BASE, [target], support_model=None,
                                                bench_mm=0.0))


class TheShapeOfAPartTests(unittest.TestCase):
    def test_each_rule_says_why(self) -> None:
        rng = np.random.default_rng(1)

        def block(x: float, y: float, z0: float, z1: float, n: int = 400) -> np.ndarray:
            return np.column_stack((rng.uniform(0, x, n), rng.uniform(0, y, n), rng.uniform(z0, z1, n)))

        self.assertEqual("", why_no_part_beside(block(40, 40, 0, 40), bench_mm=0.0))
        self.assertEqual("too few of its points were measured", why_no_part_beside(block(40, 40, 0, 40, 10), bench_mm=0.0))
        self.assertEqual("a surface the parts lie on", why_no_part_beside(block(400, 300, 55, 58), bench_mm=0.0))
        self.assertEqual("larger than a part", why_no_part_beside(block(300, 60, 0, 180), bench_mm=0.0))
        self.assertEqual("it hovers over the parts", why_no_part_beside(block(40, 40, 300, 340), bench_mm=0.0))
        self.assertEqual("thinner than a part", why_no_part_beside(block(120, 8, 0, 80), bench_mm=0.0))
        self.assertEqual("a strip of something, not a part", why_no_part_beside(block(120, 20, 0, 40), bench_mm=0.0))
        corner = np.vstack([block(100, 8, 0, 50), block(8, 100, 0, 50)])
        self.assertEqual("hollow, not a part", why_no_part_beside(corner, bench_mm=0.0))
        self.assertEqual("lower than a part", why_no_part_beside(block(40, 40, 0, 6), bench_mm=0.0))

    def test_what_a_part_stands_on_is_the_support_under_it(self) -> None:
        """A cube 30 mm tall on a crate floor 25 mm over the bench: the floor is what it stands on, not the bench."""
        rng = np.random.default_rng(2)
        cube = np.column_stack((rng.uniform(0, 40, 400), rng.uniform(0, 40, 400), rng.uniform(25, 31, 400)))
        floor = types.SimpleNamespace(height_under=lambda points: 25.0)
        self.assertEqual("", why_no_part_beside(cube, bench_mm=0.0))
        self.assertEqual("lower than a part", why_no_part_beside(cube, bench_mm=0.0, support_model=floor))


class _Placed:
    def to_matrix(self) -> np.ndarray:
        return CAMERA_TO_BASE


class _Zones:
    """Keeps out everything whose middle lies at BASE x over ``x_mm``."""

    def __init__(self, x_mm: float) -> None:
        self.x_mm = x_mm

    def kept_out_by_a_region(self, *, label: str, centre_mm: Any) -> bool:
        return float(centre_mm[0]) > self.x_mm


def _orchestrator(*, on: bool = True, zones: Any = None) -> Any:
    loop = types.SimpleNamespace(
        name_neighbours=on, arm=None, exclusion_zones=zones, target_labels=(), target_label="gray cube",
        frame_resolver=types.SimpleNamespace(camera_to_base_for_frame=lambda frame, arm=None: _Placed()))
    loop._bench_top_mm = lambda: 0.0
    loop._support_model_of_frame = lambda frame, transform: None
    loop._kept_out_by_a_region = types.MethodType(BinPickingOrchestrator._kept_out_by_a_region, loop)
    return loop


class _Seg:
    def __init__(self, mask: np.ndarray, label: str) -> None:
        self.mask = mask.astype(np.uint8)
        self.label = label


def _frame() -> PerceptionFrame:
    scene = _Scene()
    target = scene.box(-20, -20, 20, 20, 40)
    scene.box(50, -20, 90, 20, 30)
    scene.box(-15, -85, 15, -55, 25)
    return PerceptionFrame(depth_map=scene.depth, intrinsics=K, segmentations=(_Seg(target, "gray cube"),),
                           surface_depth_map=scene.depth, timestamp=1.0)


class ThePickPutsThemOnTheFrameTests(unittest.TestCase):
    def test_the_frame_carries_the_parts_beside_where_the_cell_asks(self) -> None:
        frame = _frame()
        beside = BinPickingOrchestrator._with_the_parts_beside(_orchestrator(), frame)
        self.assertEqual(2, len(beside.parts_beside))
        self.assertEqual(frame.segmentations, beside.segmentations)
        self.assertIs(frame, BinPickingOrchestrator._with_the_parts_beside(_orchestrator(on=False), frame))

    def test_a_part_in_a_region_the_task_keeps_out_is_left_out(self) -> None:
        beside = BinPickingOrchestrator._with_the_parts_beside(_orchestrator(zones=_Zones(40.0)), _frame())
        self.assertEqual(1, len(beside.parts_beside))

    def test_the_world_names_their_boxes_by_their_points(self) -> None:
        beside = BinPickingOrchestrator._with_the_parts_beside(_orchestrator(), _frame())
        named = _named_points_base(beside, beside.segmentations, 0, CAMERA_TO_BASE)
        self.assertEqual([LABEL, LABEL], [label for label, _ in named])
        for _, points in named:
            self.assertGreater(float(np.median(points[:, 2])), 20.0)


if __name__ == "__main__":
    unittest.main()
