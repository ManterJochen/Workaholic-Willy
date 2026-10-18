"""The hand grasps the part's own body where SFE's convex prism has no grasp (lever A, 2026-10-13).

``body_grasps`` plans on the part as the cameras measured it: a 2 mm grid of columns, each as high as its highest point.
Pinned on the owner's Hand-E (the shipped ``hande`` profile, 49.99 mm stroke, 10.8 mm fingers 29.24 mm wide), on
synthetic parts read as a camera straight above reads them, 0.4 mm of noise:

* a 30 mm cube: closed across a face, the width the cube's and 2 mm, the contact square, robust;
* a lying 40 mm cylinder: closed across its axis with the pads over its equator; never with the pads wholly over it,
  where it leans out of the friction cone;
* a ring 70 mm across with a 44 mm bore: too wide as a whole, held by one wall with a finger in the bore; a ring with a
  30 mm bore offers none, a 29 by 10.8 mm finger does not go into a 30 mm bore;
* a six-lobed star 70 mm across, which SFE's hull makes wider than the stroke: held;
* a neighbour standing where an open finger would come down refuses that grasp.

SFE's prism offered no grasp on the ring and the star; it left with its search on 2026-10-16 (the owner's cleanup).
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.grasping.generation.body_grasps import body_grasps, body_map
from src.robot.grasping.generation.hand import SupportFootprintCandidate, SupportFootprintJaw


@lru_cache(maxsize=None)
def _jaw() -> SupportFootprintJaw:
    from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

    return SupportFootprintJaw.from_robot_config(hande_cell())


def _surface(height: Any, half_mm: float, *, n: int = 12000, seed: int = 0) -> np.ndarray:
    """A part's top as a camera above reads it: ``height(xy)`` over the square ``half_mm`` about the origin, NaN where
    the part is not, 0.4 mm of noise."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-half_mm, half_mm, (n, 2))
    z = height(xy)
    keep = np.isfinite(z)
    return np.column_stack([xy[keep], z[keep] + rng.normal(0.0, 0.4, int(keep.sum()))])


def _cube(xy: np.ndarray) -> np.ndarray:
    return np.where((np.abs(xy[:, 0]) <= 15.0) & (np.abs(xy[:, 1]) <= 15.0), 30.0, np.nan)


def _lying(xy: np.ndarray) -> np.ndarray:
    """A cylinder 40 mm across lying along x, its axis 20 mm over the support."""
    return np.where((np.abs(xy[:, 0]) <= 30.0) & (np.abs(xy[:, 1]) <= 20.0),
                    20.0 + np.sqrt(np.maximum(400.0 - xy[:, 1] ** 2, 0.0)), np.nan)


def _ring(outer: float, bore: float):
    def height(xy: np.ndarray) -> np.ndarray:
        r = np.hypot(xy[:, 0], xy[:, 1])
        return np.where((r <= outer / 2.0) & (r >= bore / 2.0), 20.0, np.nan)
    return height


def _star(xy: np.ndarray) -> np.ndarray:
    angle = np.arctan2(xy[:, 1], xy[:, 0])
    return np.where(np.hypot(xy[:, 0], xy[:, 1]) <= 18.0 + 17.0 * np.abs(np.cos(3.0 * angle)), 15.0, np.nan)


def _yaw_deg(candidate: SupportFootprintCandidate) -> float:
    return math.degrees(math.atan2(float(candidate.closing_axis[1]), float(candidate.closing_axis[0]))) % 180.0


class TheBodyOfAPart(unittest.TestCase):
    def test_a_cube_is_held_across_a_face(self) -> None:
        found = body_grasps(_surface(_cube, 20.0), support_height_mm=0.0, jaw=_jaw(), max_candidates=6)
        self.assertTrue(found)
        best = found[0]
        self.assertAlmostEqual(best.grip_width_mm, 32.0, delta=1.5)
        self.assertLess(math.degrees(best.contact_angle_rad), 1.0)
        self.assertTrue(best.robust)
        self.assertLess(min(_yaw_deg(best) % 90.0, 90.0 - _yaw_deg(best) % 90.0), 1e-6)
        self.assertLess(float(np.hypot(*best.position_mm[:2])), 3.0)
        # The fingertips over the support by its clearance.
        self.assertGreaterEqual(best.clearance_mm, _jaw().table_clearance_mm - 1e-9)

    def test_a_part_tall_enough_is_held_at_least_ten_mm_under_its_top(self) -> None:
        """The owner's cell, 2026-10-14: "sehr knapp an der Oberseite", grasps 6 to 15 mm into the part's top. Every
        grasp of a block tall enough holds it ``HOLD_OVERLAP_MM`` under its top at the least; a block too short for it is
        held as deep as the support lets the fingertips go, the table's clearance over it."""
        from src.robot.grasping.generation.body_grasps import HOLD_OVERLAP_MM

        jaw = _jaw()
        for height in (9.0, 12.0, 25.0, 45.0):
            with self.subTest(height=height):
                block = _surface(lambda xy, h=height: np.where((np.abs(xy[:, 0]) <= 15.0) & (np.abs(xy[:, 1]) <= 15.0),
                                                              h, np.nan), 20.0)
                found = body_grasps(block, support_height_mm=0.0, jaw=jaw, max_candidates=12)
                self.assertTrue(found)
                deepest = height - jaw.table_clearance_mm
                for candidate in found:
                    held = height - (float(candidate.position_mm[2]) - jaw.pad_ahead_mm)
                    # The block's top reads up to about 1.2 mm over its height in the 0.4 mm noise.
                    self.assertGreaterEqual(held, min(HOLD_OVERLAP_MM, deepest) - 1.5, candidate)
                    self.assertGreaterEqual(candidate.clearance_mm, jaw.table_clearance_mm - 1e-9)

    def test_a_hand_that_opens_as_needed_is_offered_each_place_narrow_and_fully_open(self) -> None:
        """Lever G: the narrow opening first, the span and 7 mm beside each jaw (``NARROW_ROOM_MM``: 2 mm until the
        owner's Robotiq came down onto the parts, 2026-10-14); the same place fully open after it, where the guard holds
        the part's smeared edge beside the narrow fingers (2026-10-13). A toggle opens fully, and so does a hand that
        opens as needed where the narrow room leaves no stroke past the span."""
        from src.robot.grasping.generation.hand import NARROW_ROOM_MM

        jaw = _jaw()
        self.assertEqual(NARROW_ROOM_MM, 7.0)
        cube = _surface(_cube, 20.0)
        found = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=6, opens_as_needed=True)
        narrow = [c for c in found if c.open_width_mm is not None]
        fully = [c for c in found if c.open_width_mm is None]
        self.assertTrue(narrow and fully)
        for c in narrow:
            self.assertAlmostEqual(c.open_width_mm, c.grip_width_mm - 2.0 + 2.0 * NARROW_ROOM_MM, places=6)
            self.assertLess(c.open_width_mm, jaw.aperture_mm)
        # A 40 mm cube: its span and the narrow room pass the stroke, so it is grasped fully open alone.
        wide = body_grasps(_surface(lambda xy: np.where((np.abs(xy[:, 0]) <= 20.0) & (np.abs(xy[:, 1]) <= 20.0), 30.0,
                                                        np.nan), 26.0),
                           support_height_mm=0.0, jaw=jaw, max_candidates=6, opens_as_needed=True)
        self.assertTrue(wide and all(c.open_width_mm is None for c in wide))
        self.assertTrue(any(np.allclose(n.position_mm, f.position_mm) and np.allclose(n.closing_axis, f.closing_axis)
                            for n in narrow for f in fully))
        self.assertIsNotNone(found[0].open_width_mm, "the narrow opening ranks first")
        toggle = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=6)
        self.assertTrue(toggle and all(c.open_width_mm is None for c in toggle))

    def test_a_lying_cylinder_is_held_with_the_pads_over_its_equator(self) -> None:
        jaw = _jaw()
        found = body_grasps(_surface(_lying, 35.0), support_height_mm=0.0, jaw=jaw, max_candidates=12)
        self.assertTrue(found)
        for candidate in found:
            self.assertAlmostEqual(_yaw_deg(candidate), 90.0, delta=0.5)   # across its axis, never along it
            low = float(candidate.position_mm[2]) - jaw.pad_ahead_mm
            high = float(candidate.position_mm[2]) + jaw.pad_behind_mm
            self.assertLessEqual(low, 20.0)          # the pads reach down to its equator
            self.assertGreaterEqual(high, 20.0)

    def test_a_lying_cylinder_leans_out_of_the_cone_over_its_equator(self) -> None:
        from src.robot.grasping.generation.body_grasps import BODY_MIN_OVERLAP_MM

        jaw = _jaw()
        cloud = _surface(_lying, 35.0)
        # The support raised so that the lowest pad meets the cylinder over its equator: it leans there by the angle
        # whose sine is the height over the axis over the radius, and past the friction cone nothing is offered.
        refused: dict[str, int] = {}
        found = body_grasps(cloud, support_height_mm=12.0, jaw=jaw, max_candidates=12, refusals=refused)
        for candidate in found:
            meets = float(candidate.position_mm[2]) - jaw.pad_ahead_mm + BODY_MIN_OVERLAP_MM
            lean = math.asin(min(1.0, max(0.0, meets - 20.0) / 20.0))
            self.assertLessEqual(lean, jaw.cone_rad + math.radians(4.0))   # the 2 mm grid's own resolution
        self.assertGreater(refused.get("cone", 0), 0)

    def test_a_ring_is_held_by_one_wall_with_a_finger_in_its_bore(self) -> None:
        jaw = _jaw()
        wide = _surface(_ring(70.0, 44.0), 40.0, n=16000)
        found = body_grasps(wide, support_height_mm=0.0, jaw=jaw, max_candidates=6)
        self.assertTrue(found)
        for candidate in found:
            radius = float(np.hypot(*candidate.position_mm[:2]))
            self.assertGreater(radius, 22.0)          # between the bore's wall and the ring's outside
            self.assertLess(radius, 35.0)
            self.assertLess(candidate.grip_width_mm, 26.0)
        narrow = _surface(_ring(60.0, 30.0), 35.0, n=14000)
        self.assertFalse(body_grasps(narrow, support_height_mm=0.0, jaw=jaw, max_candidates=6))

    def test_a_finger_goes_into_a_bore_only_where_the_camera_saw_it_free(self) -> None:
        """The guard does not hold the part it grasps: a finger over the part's own outline comes down only through
        space a ray of the frame reached. Outside the outline, beside a cube, nothing changes."""
        jaw = _jaw()
        ring = _surface(_ring(70.0, 44.0), 40.0, n=16000)
        everything = body_grasps(ring, support_height_mm=0.0, jaw=jaw, max_candidates=6,
                                 corridor_seen=lambda points: np.ones(points.shape[0], dtype=bool))
        self.assertTrue(everything)
        refused: dict[str, int] = {}
        # A camera that saw the ring's top and nothing under it: the bore is unseen.
        hidden = body_grasps(ring, support_height_mm=0.0, jaw=jaw, max_candidates=6, refusals=refused,
                             corridor_seen=lambda points: points[:, 2] > 20.0)
        self.assertFalse(hidden)
        self.assertGreater(refused.get("unseen_corridor", 0), 0)
        cube = _surface(_cube, 20.0)
        free = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=6)
        blind = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=6,
                            corridor_seen=lambda points: points[:, 2] > 30.0)
        self.assertEqual([c.position_mm.tolist() for c in free], [c.position_mm.tolist() for c in blind])

    def test_where_the_scene_refuses_straight_down_the_hand_leans_away_and_only_through_seen_space(self) -> None:
        """A beam 25 to 60 mm over a 30 mm cube's +y half, over the fingers' way down from above: straight down every
        grasp closing along x meets it; leaning toward -y the fingers come in under it (``BODY_TILTS_DEG``). Without a
        seen test no tilt is offered, as SFE offers none."""
        jaw = _jaw()
        cube = _surface(_cube, 20.0)
        rng = np.random.default_rng(7)
        beam = np.column_stack([rng.uniform(-60.0, 60.0, 4000), rng.uniform(2.0, 30.0, 4000),
                                rng.uniform(55.0, 90.0, 4000)])
        along_x = lambda c: abs(float(c.closing_axis[0])) > math.cos(math.radians(10.0))  # noqa: E731
        seen_all = lambda points: np.ones(points.shape[0], dtype=bool)  # noqa: E731
        leaning = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=24, obstacle_points_base_mm=beam,
                              corridor_seen=seen_all)
        tilted = [c for c in leaning if along_x(c) and float(c.approach[2]) > -0.999]
        self.assertTrue(tilted, "no grasp closing along x leans under the beam")
        for c in tilted:
            # The approach is the way in: a hand coming in toward +y stands on the -y side, away from the beam.
            self.assertGreater(float(c.approach[1]), 0.0, "the hand stands on the beam's side")
            self.assertGreaterEqual(math.degrees(math.acos(-float(c.approach[2]))), 15.0 - 1e-6)
        self.assertFalse(any(along_x(c) and float(c.approach[2]) <= -0.999 for c in leaning
                             if float(c.position_mm[1]) > -5.0), "a grasp straight down under the beam")
        blind = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=24, obstacle_points_base_mm=beam)
        self.assertFalse(any(float(c.approach[2]) > -0.999 for c in blind))

    def test_a_star_wider_than_the_stroke_as_a_hull_is_held(self) -> None:
        jaw = _jaw()
        star = _surface(_star, 40.0, n=16000)
        found = body_grasps(star, support_height_mm=0.0, jaw=jaw, max_candidates=6)
        self.assertTrue(found)
        for candidate in found:
            self.assertLessEqual(candidate.grip_width_mm - 2.0, jaw.aperture_mm - jaw.width_safety_mm + 1e-6)
            self.assertLessEqual(candidate.contact_angle_rad, jaw.cone_rad + 1e-9)

    def test_a_neighbour_under_an_open_finger_refuses_the_grasp(self) -> None:
        jaw = _jaw()
        cube = _surface(_cube, 20.0)
        free = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=12)
        self.assertTrue(any(abs(_yaw_deg(c) - 0.0) < 1e-6 or abs(_yaw_deg(c) - 180.0) < 1e-6 for c in free))
        # A post 8 mm across standing 21 to 29 mm off the cube's middle along x, where an open finger closing along x
        # comes down (the open stroke's half, 25 mm, and the finger beyond it).
        rng = np.random.default_rng(3)
        post = np.column_stack([rng.uniform(22.0, 28.0, 400), rng.uniform(-4.0, 4.0, 400), rng.uniform(0.0, 30.0, 400)])
        refused: dict[str, int] = {}
        boxed = body_grasps(cube, support_height_mm=0.0, jaw=jaw, max_candidates=12, obstacle_points_base_mm=post,
                            refusals=refused)
        for candidate in boxed:
            along_x = abs(float(candidate.closing_axis[0])) > math.cos(math.radians(20.0))
            near_post = abs(float(candidate.position_mm[1])) < 16.0
            self.assertFalse(along_x and near_post, "a grasp closing along x kept its open finger on the post")
        self.assertGreater(refused.get("seen_corridor", 0) + refused.get("seen_fingers", 0), 0)

    def test_the_body_map_keeps_the_low_fragments_apart(self) -> None:
        cube = _surface(_cube, 20.0)
        fragment = np.column_stack([np.full(40, 60.0), np.linspace(-5, 5, 40), np.full(40, 3.0)])
        body = body_map(np.vstack([cube, fragment]), 0.0, floor_margin_mm=2.0)
        assert body is not None
        self.assertGreater(body.trimmed.shape[0], 0)
        self.assertLess(float(body.points[:, 0].max()), 20.0)


class SeenByAnyLook(unittest.TestCase):
    def test_a_point_one_look_saw_past_is_seen_whichever_look_it_was(self) -> None:
        """A bore one look sees only the near wall of, the next looks into (``scene_obstacles.SeenInAnyFrame``): a point
        is seen where any frame's ray reached it."""
        from src.robot.grasping.generation.scene_obstacles import SeenInAnyFrame, corridor_seen_in

        k = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
        down = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, -1.0, 500.0], [0.0, 0.0, 0.0, 1.0]])
        # Two frames of one camera over a floor 500 mm away: the left saw only its left half, the right its right half.
        left, right = np.full((480, 640), 500.0), np.full((480, 640), 500.0)
        left[:, 320:] = 0.0
        right[:, :320] = 0.0
        points = np.array([[-50.0, 0.0, 20.0], [50.0, 0.0, 20.0], [0.5, 0.0, 600.0]])
        one, other = corridor_seen_in(left, k, down), corridor_seen_in(right, k, down)
        self.assertEqual([True, False, False], one(points).tolist())
        self.assertEqual([False, True, False], other(points).tolist())
        both = SeenInAnyFrame((one, other))
        self.assertEqual([True, True, False], both(points).tolist())


class ThePartTheGuardHoldsOut(unittest.TestCase):
    def test_a_sliver_of_a_rim_the_mask_ran_onto_is_no_part_to_plan_on(self) -> None:
        """The owner's crate (2026-10-13, 10:00): a sliver of the rim 60 mm off the part, in its mask, raised SFE's prism
        to the rim and its grasps closed on air. The body plans on the largest cluster, the part the guard holds out
        (``perceived.target_body_points``)."""
        from src.robot.grasping.generation._support_footprint_stage import support_footprint_breakdowns

        cube = _surface(_cube, 20.0)
        rng = np.random.default_rng(5)
        rim = np.column_stack([rng.uniform(55.0, 58.0, 60), rng.uniform(-15.0, 15.0, 60), rng.uniform(95.0, 98.0, 60)])

        def positions(cloud: np.ndarray) -> list[list[float]]:
            breakdowns, _ = support_footprint_breakdowns(cloud, camera_to_base=np.eye(4), support_height_mm=0.0,
                                                         jaw=_jaw(), obstacle_points_base_mm=None, max_candidates=12)
            return [np.round(b.pose.position_mm, 6).tolist() for b in breakdowns]

        alone = positions(cube)
        self.assertTrue(alone)
        self.assertEqual(alone, positions(np.vstack([cube, rim])))


if __name__ == "__main__":
    unittest.main()
