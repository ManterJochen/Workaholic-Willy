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
* a neighbour standing where an open finger would come down refuses that grasp;
* both stages' grasps ranked together, robust first, by score, every grasp of SFE kept and a body's near one left out
  unless it opens the jaws otherwise (``merged_ranking``).
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.grasping.generation.body_grasps import body_grasps, body_map, merged_ranking
from src.robot.grasping.generation.support_footprint import (
    SupportFootprintCandidate,
    SupportFootprintJaw,
    generate_support_footprint_grasps,
)


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
        self.assertFalse(generate_support_footprint_grasps(wide, support_height_mm=0.0, jaw=jaw))
        found = body_grasps(wide, support_height_mm=0.0, jaw=jaw, max_candidates=6)
        self.assertTrue(found)
        for candidate in found:
            radius = float(np.hypot(*candidate.position_mm[:2]))
            self.assertGreater(radius, 22.0)          # between the bore's wall and the ring's outside
            self.assertLess(radius, 35.0)
            self.assertLess(candidate.grip_width_mm, 26.0)
        narrow = _surface(_ring(60.0, 30.0), 35.0, n=14000)
        self.assertFalse(body_grasps(narrow, support_height_mm=0.0, jaw=jaw, max_candidates=6))

    def test_a_star_wider_than_the_stroke_as_a_hull_is_held(self) -> None:
        jaw = _jaw()
        star = _surface(_star, 40.0, n=16000)
        self.assertFalse(generate_support_footprint_grasps(star, support_height_mm=0.0, jaw=jaw))
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


class TheBodyMakesAPartWhole(unittest.TestCase):
    """Lever D: where SFE's fine search waits and its coarse grid finds few, the body's grasps make the part whole, so a
    lazy ranking takes it without searching every part again in full; where neither finds enough, it still waits."""

    def test_a_star_the_coarse_grid_finds_nothing_on_is_whole_with_the_bodys_grasps(self) -> None:
        from src.robot.grasping.generation._support_footprint_stage import (
            FINE_SEARCH_KEY,
            support_footprint_breakdowns,
        )

        breakdowns, telemetry = support_footprint_breakdowns(
            _surface(_star, 40.0, n=16000), camera_to_base=np.eye(4), support_height_mm=0.0, jaw=_jaw(),
            obstacle_points_base_mm=None, max_candidates=12, fine_pass=False)
        self.assertEqual(0, telemetry["support_footprint_candidates"])
        self.assertGreaterEqual(len(breakdowns), 3)
        self.assertTrue(all(b.components["support_footprint"]["stage"] == "body" for b in breakdowns))
        self.assertNotIn(FINE_SEARCH_KEY, telemetry)


class BothStagesRankedTogether(unittest.TestCase):
    @staticmethod
    def _candidate(score: float, robust: bool, x: float, yaw_deg: float = 0.0) -> SupportFootprintCandidate:
        axis = np.array([math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg)), 0.0])
        return SupportFootprintCandidate(score=score, position_mm=np.array([x, 0.0, 20.0]),
                                         approach=np.array([0.0, 0.0, -1.0]), closing_axis=axis, grip_width_mm=30.0,
                                         contact_angle_rad=0.0, clearance_mm=5.0, robust=robust)

    def test_robust_first_then_by_score_and_a_near_one_left_out_unless_it_opens_otherwise(self) -> None:
        sfe = [self._candidate(0.9, False, 0.0), self._candidate(0.7, True, 10.0)]
        body = [self._candidate(0.85, True, 20.0), self._candidate(0.8, True, 11.0)]
        self.assertEqual([c.score for c in merged_ranking(sfe, body, most=12)], [0.85, 0.7, 0.9])
        self.assertEqual([c.score for c in merged_ranking(sfe, body, most=12, apart=True)], [0.85, 0.8, 0.7, 0.9])
        across = [self._candidate(0.8, True, 11.0, yaw_deg=90.0)]
        self.assertEqual([c.score for c in merged_ranking(sfe, across, most=12)], [0.8, 0.7, 0.9])

    def test_every_grasp_of_sfe_stays_and_the_body_adds_most_at_most(self) -> None:
        sfe = [self._candidate(0.5 - k / 100.0, True, 10.0 * k) for k in range(12)]
        body = [self._candidate(0.9 - k / 100.0, True, 500.0 + 10.0 * k) for k in range(20)]
        ranked = merged_ranking(sfe, body, most=12)
        self.assertEqual(len(ranked), 24)
        self.assertTrue(all(any(c is s for c in ranked) for s in sfe))
        self.assertEqual([c.score for c in ranked[:12]], [c.score for c in body[:12]])


if __name__ == "__main__":
    unittest.main()
