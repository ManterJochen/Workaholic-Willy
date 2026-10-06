"""A finger may come to the measured surface of a neighbour part the detector named (the owner, 2026-10-05).

The owner's decision: "Finger dürfen streifen". Only the fingers, and only against a part the detector named: the
camera world's box of such a part, no larger than a part (``perceived.PART_SIZED_MM``), holds a finger link to its
measured surface, the box less its margin, at no distance kept (``PerceivedBox.soft_mm``,
``perceived.fingers_touch_parts``). Every other link, the hand's housing among them, keeps the guard's distance from the
whole box, and a box nobody named (a wall, a bin, the support, what the detector did not see) stays whole for every link.
Pinned here, on the exact meshes where the engine loads:

* the exact guard admits a finger inside a named part's margin and refuses one inside its measured surface; the same box
  unnamed refuses the finger inside its margin; the housing keeps the whole box either way;
* the camera world marks a named part's boxes soft, not an unnamed one's, not a named box larger than a part, and none
  where the fingers may not touch parts;
* the calculator offers the grasps the guard admits so: a cylinder with named cubes 18 mm off either side gets grasps
  closing between them, the same cubes unnamed or the rule off get none there.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.robot.safety._capsule import AxisAlignedBox
from src.robot.safety._ur_kinematics import ur_link_transforms_mm
from tests.test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards import AWAY, _arm, _backend

#: The world's margin round a box the cameras saw, and the guard's distance from it, as in this file's scenes.
MARGIN_MM = 8.0
GUARD_MM = 3.0
#: A 30 mm part, its box grown by the margin.
_HALF = 15.0 + MARGIN_MM


class TheExactGuardHoldsAFingerToANamedPartsSurfaceTests(unittest.TestCase):
    """The Hand-E's fingers on the owner-like UR10, a 30 mm part's box marched in along the closing direction."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.backend = _backend(_arm())
        cls.transforms = ur_link_transforms_mm("ur10", np.asarray(AWAY, dtype=np.float64))
        flange = np.asarray(cls.transforms[-1], dtype=np.float64)
        cls.tcp = flange[:3, :3] @ np.array([0.0, 0.0, 155.75]) + flange[:3, 3]
        # The direction a box meets a finger first from: along the tool's x or y, either way round, at the fingertips.
        cls.direction, cls.first = None, None
        for column in (0, 1):
            for sign in (1.0, -1.0):
                way = sign * flange[:3, column]
                start = cls._march(way)
                if start is not None and "finger" in start[1]:
                    cls.direction, cls.first = way, start[0]
                    return

    @classmethod
    def _box(cls, at_mm: float, soft_mm: float, direction: "np.ndarray | None" = None) -> AxisAlignedBox:
        way = cls.direction if direction is None else direction
        centre = cls.tcp + way * at_mm
        return AxisAlignedBox(center_mm=centre, half_extents_mm=np.array([_HALF, _HALF, _HALF]), name="seen_00_part",
                              soft_mm=soft_mm)

    @classmethod
    def _hit(cls, box: AxisAlignedBox) -> "tuple[str, float] | None":
        return cls.backend.evaluate(cls.transforms, 0.0, (box,), GUARD_MM, arm_pairs=False)

    @classmethod
    def _march(cls, way: np.ndarray) -> "tuple[float, str] | None":
        """The farthest centre along ``way`` at which a whole box is refused, and the pair that refuses it."""
        for at in np.arange(160.0, 0.0, -0.5):
            hit = cls.backend.evaluate(cls.transforms, 0.0, (cls._box(float(at), 0.0, way),), GUARD_MM,
                                       arm_pairs=False)
            if hit is not None:
                return float(at), str(hit[0])
        return None

    def setUp(self) -> None:
        if self.direction is None:
            self.skipTest("no direction meets a finger first at this pose")

    def test_inside_a_named_parts_margin_a_finger_is_admitted_and_unnamed_refused(self) -> None:
        # 6 mm further in: the finger stands inside the grown box, still a few millimetres off the measured surface.
        at = float(self.first) - 6.0
        unnamed = self._hit(self._box(at, 0.0))
        self.assertIsNotNone(unnamed)
        assert unnamed is not None
        self.assertIn("finger", unnamed[0])
        self.assertIsNone(self._hit(self._box(at, MARGIN_MM)), "a finger to a named part's margin is refused")

    def test_inside_a_named_parts_measured_surface_a_finger_is_refused(self) -> None:
        # 25 mm further in: the box meets the finger at a slant, so this is well past the margin and into the part.
        at = float(self.first) - 25.0
        hit = self._hit(self._box(at, MARGIN_MM))
        self.assertIsNotNone(hit, "a finger through a named part's measured surface is admitted")
        assert hit is not None
        self.assertIn("finger", hit[0])
        self.assertLess(hit[1], 0.0)

    def test_the_housing_keeps_the_whole_box(self) -> None:
        """A named part's box marched in until it meets anything but a finger: the housing keeps the guard's distance
        from the whole box, soft or not."""
        flange = np.asarray(self.transforms[-1], dtype=np.float64)
        # Higher up, where the housing reaches out further than the fingers: 60 mm back from the TCP.
        higher = self.tcp - flange[:3, 2] * 60.0
        for at in np.arange(160.0, 0.0, -0.5):
            box = AxisAlignedBox(center_mm=higher + self.direction * float(at),
                                 half_extents_mm=np.array([_HALF, _HALF, _HALF]), name="seen_00_part", soft_mm=MARGIN_MM)
            hit = self._hit(box)
            if hit is not None:
                self.assertNotIn("finger", hit[0], "the soft box let the housing through and met a finger")
                self.assertGreaterEqual(hit[1], 0.0, "the housing was let into the box")
                return
        self.fail("no box met the hand")


def _scene() -> "tuple[Any, list[Any], Any, Any]":
    """A camera straight down over a 30 mm cube, a 300 mm wall and a second cube, the first cube and the wall named."""
    from tests._seen_scenes import Solid, looking_down, render

    cube = Solid((0.0, -400.0, 15.0), (15.0, 15.0, 15.0))
    wall = Solid((0.0, -300.0, 30.0), (150.0, 3.0, 30.0))
    other = Solid((120.0, -450.0, 15.0), (15.0, 15.0, 15.0))
    camera = looking_down(0.0, -380.0)
    full = render(camera, [cube, wall, other])

    def mask_of(solid: Any) -> np.ndarray:
        without = render(camera, [s for s in (cube, wall, other) if s is not solid])
        return (full > 0.0) & ((without <= 0.0) | (full < without - 0.5))

    return camera, [cube, wall, other], full, {"cube": mask_of(cube), "wall": mask_of(wall)}


class TheCameraWorldMarksANamedPartSoftTests(unittest.TestCase):
    def _world(self, part_soft_mm: float) -> Any:
        from src.robot.safety.planning.perceived import DepthView, WorldBuildTuning, build_perceived_boxes
        from tests._seen_scenes import K, LIMITS

        camera, _solids, depth, masks = _scene()
        view = DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=camera, name="overhead", timestamp=100.0,
                         labelled_masks=(("part", masks["cube"]), ("part", masks["wall"])))
        return build_perceived_boxes(views=(view,), limits=LIMITS, tuning=WorldBuildTuning(
            pixel_stride=2, margin_mm=MARGIN_MM, part_soft_mm=part_soft_mm))

    def test_a_named_part_is_soft_a_wall_and_an_unnamed_part_are_not(self) -> None:
        world = self._world(MARGIN_MM)
        named = [box for box in world.boxes if box.label and max(box.dims_mm[:2]) < 100.0]
        wall = [box for box in world.boxes if box.label and max(box.dims_mm[:2]) > 250.0]
        unnamed = [box for box in world.boxes if not box.label and box.kind == "seen"]
        self.assertTrue(named and wall and unnamed, [(b.name, b.dims_mm, b.label) for b in world.boxes])
        self.assertTrue(all(box.soft_mm == MARGIN_MM for box in named))
        self.assertTrue(all(box.soft_mm == 0.0 for box in wall), "a named wall longer than a part went soft")
        self.assertTrue(all(box.soft_mm == 0.0 for box in unnamed))

    def test_a_part_named_by_its_points_in_base_is_soft_without_a_mask(self) -> None:
        """A wrist camera offers no mask that holds once the arm moved: the pick offers the named parts' points in BASE
        (``SegmentationOffer.named_points_base_mm``), and a box most of whose points lie on them is that part's."""
        from src.robot.safety.planning.perceived import DepthView, WorldBuildTuning, build_perceived_boxes
        from tests._seen_scenes import K, LIMITS, seen_points

        camera, _solids, depth, masks = _scene()
        cube_points = seen_points(camera, np.where(masks["cube"], depth, 0.0))
        view = DepthView(surface_depth_mm=depth, intrinsics=K, camera_to_base=camera, name="wrist", timestamp=100.0)
        world = build_perceived_boxes(views=(view,), limits=LIMITS,
                                      tuning=WorldBuildTuning(pixel_stride=2, margin_mm=MARGIN_MM,
                                                              part_soft_mm=MARGIN_MM),
                                      named_points_base_mm=(("part", cube_points),))
        soft = [box for box in world.boxes if box.soft_mm > 0.0]
        self.assertTrue(soft, [(b.name, b.dims_mm, b.label) for b in world.boxes])
        self.assertTrue(all(box.label == "part" and max(box.dims_mm[:2]) < 100.0 for box in soft))
        self.assertEqual(len(soft), len([box for box in world.boxes if box.label]), "a box named and not soft")

    def test_where_the_fingers_may_not_touch_parts_nothing_is_soft(self) -> None:
        self.assertTrue(all(box.soft_mm == 0.0 for box in self._world(0.0).boxes))

    def test_the_guard_is_handed_the_soft_boxes(self) -> None:
        from src.robot.safety.planning.live_world import _guard_boxes

        world = self._world(MARGIN_MM)
        soft = {box.name for box in world.boxes if box.soft_mm > 0.0}
        self.assertTrue(soft)
        self.assertEqual(soft, {box.name for box in _guard_boxes(world) if box.soft_mm > 0.0})


def _compute(calc: Any, frame: Any, others: "list[np.ndarray]") -> Any:
    """``compute_result`` as the boxed-in tests' ``compute`` calls it, with the other parts' masks handed in."""
    from types import SimpleNamespace

    from src.geometry import Frame, Transform
    from src.robot.execution.autonomous_grasp.builders import build_gripper_geometry
    from src.robot.grasping.collision import SupportPlane
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import MAT_MM, _hande

    cfg = _hande()
    seg = SimpleNamespace(mask=frame.mask, label="part", score=1.0)
    return calc.compute_result(
        seg, frame.depth, pixel_to_mm=None, dense_sampling=False, other_object_masks=list(others),
        camera_to_base=Transform.from_matrix(frame.camera_to_base, from_frame=Frame.CAMERA, to_frame=Frame.BASE),
        support_plane=SupportPlane(normal=np.array([0.0, 0.0, 1.0]), offset_mm=float(MAT_MM), frame=Frame.BASE),
        min_table_clearance_mm=float(cfg.grasping.support.min_clearance_mm),
        gripper_model=build_gripper_geometry(cfg.grasping.gripper_geometry))


class TheCalculatorOffersTheGraspsTheGuardAdmitsTests(unittest.TestCase):
    """The calculator's scenes (``test_the_calculator_keeps_the_guards_distance_from_a_neighbour``): cubes 18 mm off the
    40 mm cylinder on both sides. The open finger needs 15.8 mm beside the part; the margin and the guard's distance
    make 31 mm of it a whole box."""

    def _result(self, *, named: bool, soft_mm: float) -> Any:
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import (
            AIM, BENCH, EYE, FOCAL, HEIGHT, MAT, PART, WIDTH, Frame_, calculator, owner_rules,
        )
        from tests.test_a_side_grasp_never_goes_through_unseen_space import DepthCamera
        from tests.test_the_calculator_keeps_the_guards_distance_from_a_neighbour import on_x

        cubes = on_x(18.0)
        frame = Frame_((BENCH, MAT, *cubes), PART)
        masks = []
        for cube in cubes:
            others = tuple(c for c in cubes if c is not cube)
            without = DepthCamera(eye=EYE, target=AIM, solids=(BENCH, MAT, *others, PART), width=WIDTH,
                                  height=HEIGHT, focal_px=FOCAL).depth
            depth = np.where(frame.depth > 0.0, frame.depth, np.inf)
            masks.append(np.isfinite(depth) & (~np.isfinite(without) | (depth < without - 0.5)))
        rules = owner_rules(margin_mm=15.0, support_distance_mm=5.0, part_soft_mm=soft_mm)
        return _compute(calculator(scene=True, rules=rules), frame, masks if named else [])

    @staticmethod
    def _across(result: Any) -> list[Any]:
        """The grasps closing toward the cubes, along x: their fingers go between the part and a cube."""
        return [g for g in result.candidates if abs(float(g.axis[0])) > 0.9]

    def test_named_cubes_18_mm_off_leave_grasps_closing_between_them(self) -> None:
        self.assertTrue(self._across(self._result(named=True, soft_mm=15.0)))

    def test_the_same_cubes_unnamed_leave_none_between_them(self) -> None:
        self.assertEqual([], self._across(self._result(named=False, soft_mm=15.0)))

    def test_with_the_fingers_kept_off_parts_none_between_them(self) -> None:
        self.assertEqual([], self._across(self._result(named=True, soft_mm=0.0)))


if __name__ == "__main__":
    unittest.main()
