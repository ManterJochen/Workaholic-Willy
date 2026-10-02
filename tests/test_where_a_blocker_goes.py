"""What clears a blocker reads, pure (cell fixes Track R): the neighbours as separate objects, which of them may be
gripped as a blocker, a free spot the camera saw to set one down on, and the pose that sets it down there.

* Seen points group into objects where they stand apart; a speck is no object.
* A blocker stands on what the part stands on (its foot within reach of the support's reading), apart from the part, away
  from the robot's base, and no wider than the hand opens across it: a cluster in the air, the part's own unmasked rest,
  the base, or a pile too wide to grip is no blocker, and each says why.
* A free spot lies on table the camera saw, with nothing seen within the hand's reach plus the guard's clearance round
  it, at least 150 mm from the part, inside the workspace; the nearest such spot to where the blocker stands is taken.
  Where none is, there is none.
* The pose that sets the blocker down keeps the grasp's turn and moves it by the spot's offset.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.robot.grasping.recovery.blocker import (
    SPOT_FROM_THE_PART_MM,
    Cluster,
    clusters_of,
    free_spot,
    mask_of,
    not_a_blocker,
    release_pose,
)
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint


def _block(low: tuple[float, float, float], high: tuple[float, float, float], step: float = 2.0) -> np.ndarray:
    """The top and sides of an upright box, every ``step`` mm."""
    xs = np.arange(low[0], high[0] + 1e-9, step)
    ys = np.arange(low[1], high[1] + 1e-9, step)
    zs = np.arange(low[2], high[2] + 1e-9, step)
    top = [(x, y, high[2]) for x in xs for y in ys]
    sides = [(x, y, z) for z in zs for x in xs for y in (ys[0], ys[-1])]
    sides += [(x, y, z) for z in zs for y in ys for x in (xs[0], xs[-1])]
    return np.asarray(top + sides, dtype=np.float64)


def _table(x0: float, x1: float, y0: float, y1: float, z: float = 0.0, step: float = 2.5) -> np.ndarray:
    xs, ys = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step))
    return np.column_stack([xs.ravel(), ys.ravel(), np.full(xs.size, z)])


PART = _block((-20.0, -720.0, 0.0), (20.0, -680.0, 40.0))
POST = _block((-55.0, -715.0, 0.0), (-27.0, -685.0, 40.0))


class SeenPointsGroupIntoObjectsTests(unittest.TestCase):
    def test_two_posts_apart_are_two_objects(self) -> None:
        other = _block((-55.0, -650.0, 0.0), (-27.0, -620.0, 30.0))

        clusters = clusters_of(np.vstack([POST, other]))

        self.assertEqual(2, len(clusters))
        centres = sorted(round(float(c.centre_mm[1])) for c in clusters)
        self.assertEqual([-700, -635], centres)

    def test_a_speck_is_no_object(self) -> None:
        speck = np.array([[100.0, -700.0, 20.0], [101.0, -700.0, 21.0]])

        clusters = clusters_of(np.vstack([POST, speck]))

        self.assertEqual(1, len(clusters))

    def test_nothing_seen_is_no_object(self) -> None:
        self.assertEqual((), clusters_of(np.zeros((0, 3))))


class WhatMayBeGrippedAsABlockerTests(unittest.TestCase):
    def test_a_post_on_the_support_beside_the_part_is_a_blocker(self) -> None:
        (cluster,) = clusters_of(POST)

        self.assertEqual("", not_a_blocker(cluster, support_mm=0.0, target_points_mm=PART, base_radius_mm=110.0,
                                           open_width_mm=50.0))

    def test_a_cluster_in_the_air_is_no_blocker(self) -> None:
        (cluster,) = clusters_of(_block((-55.0, -715.0, 30.0), (-27.0, -685.0, 70.0)))

        said = not_a_blocker(cluster, support_mm=0.0, target_points_mm=PART, base_radius_mm=110.0, open_width_mm=50.0)

        self.assertIn("support", said)

    def test_the_robot_s_base_is_no_blocker(self) -> None:
        (cluster,) = clusters_of(_block((60.0, -40.0, 0.0), (90.0, -10.0, 40.0)))

        said = not_a_blocker(cluster, support_mm=0.0, target_points_mm=PART, base_radius_mm=110.0, open_width_mm=50.0)

        self.assertIn("base", said)

    def test_a_pile_wider_than_the_hand_opens_every_way_is_no_blocker(self) -> None:
        (cluster,) = clusters_of(_block((-200.0, -800.0, 0.0), (-60.0, -660.0, 40.0)))

        said = not_a_blocker(cluster, support_mm=0.0, target_points_mm=PART, base_radius_mm=110.0, open_width_mm=50.0)

        self.assertIn("wide", said)

    def test_the_part_s_own_unmasked_rest_is_no_blocker(self) -> None:
        rest = PART[PART[:, 0] > 10.0]
        (cluster,) = clusters_of(rest)

        said = not_a_blocker(cluster, support_mm=0.0, target_points_mm=PART[PART[:, 0] <= 16.0], base_radius_mm=110.0,
                             open_width_mm=50.0)

        self.assertIn("part", said)


class AFreeSpotTests(unittest.TestCase):
    def test_the_nearest_free_seen_spot_away_from_the_part(self) -> None:
        spot = free_spot(table_points_mm=_table(-300.0, 300.0, -950.0, -450.0), occupied_points_mm=np.vstack([PART, POST]),
                         blocker_points_mm=POST, target_points_mm=PART,
                         workspace_xy=((-460.0, -935.0), (440.0, -235.0)), hand_reach_mm=40.0)

        assert spot is not None
        part_xy = PART[:, :2]
        nearest = float(np.min(np.hypot(part_xy[:, 0] - spot.centre_xy[0], part_xy[:, 1] - spot.centre_xy[1])))
        self.assertGreaterEqual(nearest, SPOT_FROM_THE_PART_MM)
        # Nothing seen within the hand's reach and the clearance of it.
        seen = np.vstack([PART, POST])[:, :2]
        self.assertGreater(float(np.min(np.hypot(seen[:, 0] - spot.centre_xy[0], seen[:, 1] - spot.centre_xy[1]))), 40.0)
        # The nearest such spot to where the post stands: no farther than the part's distance plus the post's reach.
        self.assertLess(spot.travel_mm, SPOT_FROM_THE_PART_MM + 120.0)

    def test_no_spot_where_the_seen_table_ends_near_the_part(self) -> None:
        spot = free_spot(table_points_mm=_table(-100.0, 100.0, -800.0, -600.0), occupied_points_mm=np.vstack([PART, POST]),
                         blocker_points_mm=POST, target_points_mm=PART,
                         workspace_xy=((-460.0, -935.0), (440.0, -235.0)), hand_reach_mm=40.0)

        self.assertIsNone(spot)

    def test_no_spot_where_everything_round_is_taken(self) -> None:
        crowd = np.vstack([_block((x, y, 0.0), (x + 10.0, y + 10.0, 30.0), step=5.0)
                           for x in np.arange(-300.0, 300.0, 60.0) for y in np.arange(-950.0, -450.0, 60.0)])
        spot = free_spot(table_points_mm=_table(-300.0, 300.0, -950.0, -450.0),
                         occupied_points_mm=np.vstack([PART, POST, crowd]), blocker_points_mm=POST,
                         target_points_mm=PART, workspace_xy=((-460.0, -935.0), (440.0, -235.0)), hand_reach_mm=40.0)

        self.assertIsNone(spot)

    def test_a_spot_stays_inside_the_workspace(self) -> None:
        spot = free_spot(table_points_mm=_table(-300.0, 300.0, -950.0, -450.0), occupied_points_mm=np.vstack([PART, POST]),
                         blocker_points_mm=POST, target_points_mm=PART,
                         workspace_xy=((-460.0, -800.0), (440.0, -600.0)), hand_reach_mm=40.0)

        assert spot is not None
        self.assertTrue(-800.0 <= spot.centre_xy[1] <= -600.0)


class TheReleasePoseTests(unittest.TestCase):
    def test_the_grasp_s_turn_is_kept_and_its_point_moved_by_the_spot(self) -> None:
        grasp = GraspPoint(position=np.array([-41.0, -700.0, 20.0]), approach=np.array([0.0, 0.0, -1.0]),
                           axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=28.0, score=0.9, frame=GraspFrame.BASE)

        pose = release_pose(grasp, shift_mm=(200.0, 100.0, 3.0))

        np.testing.assert_allclose(pose.position_mm, (159.0, -600.0, 23.0))
        rotation = pose.to_matrix()[:3, :3]
        np.testing.assert_allclose(rotation[:, 0], (1.0, 0.0, 0.0), atol=1e-9)
        np.testing.assert_allclose(rotation[:, 2], (0.0, 0.0, -1.0), atol=1e-9)


class TheBlockersMaskTests(unittest.TestCase):
    def test_the_cluster_s_points_mark_their_pixels_in_the_frame(self) -> None:
        intrinsics = np.array([[500.0, 0.0, 160.0], [0.0, 500.0, 120.0], [0.0, 0.0, 1.0]])
        camera_to_base = np.eye(4)
        camera_to_base[:3, 3] = (0.0, 0.0, -500.0)
        points = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]])

        mask = mask_of(points, shape=(240, 320), intrinsics=intrinsics, camera_to_base=camera_to_base, grow_px=1)

        self.assertTrue(mask[120, 160])
        self.assertTrue(mask[120, 170])
        self.assertFalse(mask[0, 0])
        self.assertEqual(Cluster, type(clusters_of(POST)[0]))


if __name__ == "__main__":
    unittest.main()
