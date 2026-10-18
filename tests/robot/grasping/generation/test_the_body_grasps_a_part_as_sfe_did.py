"""The part's own body grasped as SFE grasped (the owner, 2026-10-16: "alle der vielen features von dem SFE Stand
übernehmen und das auf die neue Punktwolke werfen").

* Every grasp is asked leaning about its closing axis too, a lean offered where it keeps the scene and ranked by the
  room it keeps, vertical first on SFE's tie (``hand._upright_first``), and a vertical grasp kept in the
  list where one exists (``body_grasps._with_a_vertical``).
* The way in is asked seen from SFE's 15 degrees off vertical on (``hand._SEEN_FROM_TILT_DEG``).
* Where straight down leaves fewer than three square grasps, the closing axis is rolled +-10/20/30 degrees out of the
  horizontal and the part searched again in that hand's own frame: so the pads lie on a tilted part's leaning faces.
  From a camera straight over a tilted box the lower finger's way under it is never seen, and no rolled grasp is
  offered there; from a camera that sees that side it is.
"""

from __future__ import annotations

import math
import unittest
from functools import lru_cache
from typing import Any

import numpy as np

from src.robot.grasping.generation.body_grasps import BODY_ROLLS_DEG, body_grasps
from src.robot.grasping.generation.scene_obstacles import SeenInAnyFrame, corridor_seen_in
from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _jaw

_K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])


def _camera(eye: "tuple[float, float, float]", target: "tuple[float, float, float]" = (0.0, 0.0, 15.0)) -> np.ndarray:
    eye_at, aim = np.asarray(eye, dtype=np.float64), np.asarray(target, dtype=np.float64)
    forward = (aim - eye_at) / np.linalg.norm(aim - eye_at)
    up = np.array([0.0, 0.0, 1.0]) if abs(float(forward[2])) < 0.99 else np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    placed = np.eye(4)
    placed[:3, :3] = np.column_stack((right, np.cross(forward, right), forward))
    placed[:3, 3] = eye_at
    return placed


@lru_cache(maxsize=None)
def _box(tilt_deg: float, eye: "tuple[float, float, float]") -> "tuple[np.ndarray, Any]":
    """A box 40 x 60 x 30 mm turned ``tilt_deg`` about BASE y, resting on its lowest edge on the bench at z = 0, ray
    cast from a camera at ``eye``: the box's points it sees, and its seen test. Its +x face leans over the bench."""
    half = np.array([20.0, 30.0, 15.0])
    t = math.radians(tilt_deg)
    turn = np.array([[math.cos(t), 0.0, math.sin(t)], [0.0, 1.0, 0.0], [-math.sin(t), 0.0, math.cos(t)]])
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * half
    centre = np.array([0.0, 0.0, -float((corners @ turn.T)[:, 2].min())])
    camera = _camera(eye)
    rows, cols = np.mgrid[0:480, 0:640].astype(np.float64)
    in_camera = np.stack([(cols - _K[0, 2]) / _K[0, 0], (rows - _K[1, 2]) / _K[1, 1], np.ones_like(cols)], axis=-1)
    rays = in_camera @ camera[:3, :3].T
    origin = camera[:3, 3]
    local_origin, local_rays = (origin - centre) @ turn, rays @ turn
    with np.errstate(divide="ignore", invalid="ignore"):
        low, high = (-half - local_origin) / local_rays, (half - local_origin) / local_rays
    near, far = np.nanmax(np.minimum(low, high), axis=-1), np.nanmin(np.maximum(low, high), axis=-1)
    floor = np.where(rays[..., 2] < 0.0, -origin[2] / rays[..., 2], np.inf)
    on_box = (near <= far) & (far > 0.0) & (near <= floor)
    reach = np.where(on_box, near, floor)
    depth = np.where(np.isfinite(reach), reach * in_camera[..., 2], 0.0)
    points = (origin + reach[..., None] * rays)[on_box]
    return points[points[:, 2] > 1.0], corridor_seen_in(depth, _K, camera)


def _grasps(cloud: np.ndarray, seen: Any) -> "list[Any]":
    return body_grasps(cloud, support_height_mm=0.0, jaw=_jaw(), corridor_seen=seen, measured_base_mm=cloud,
                       max_candidates=12)


def _rolled(candidates: "list[Any]") -> "list[Any]":
    return [c for c in candidates if abs(float(c.closing_axis[2])) > math.sin(math.radians(5.0))]


def _tilt_deg(candidate: Any) -> float:
    return math.degrees(math.acos(min(1.0, -float(candidate.approach[2]))))


class ATiltedPartIsGraspedByItsFacesWhereItsWayInIsSeenTests(unittest.TestCase):
    OVER = (0.0, 0.0, 500.0)
    SHADOW_SIDE = (350.0, 0.0, 370.0)

    def test_from_straight_over_it_no_rolled_grasp_reaches_under_it(self) -> None:
        cloud, seen = _box(25.0, self.OVER)
        self.assertEqual([], _rolled(_grasps(cloud, seen)),
                         "a rolled grasp put its lower finger where the camera over the box saw nothing")

    def test_a_second_camera_that_sees_its_leaning_side_lets_the_hand_roll_onto_its_faces(self) -> None:
        """Two cameras fused (the owner, 2026-10-16: "mehrere Kameras verbinden"): the one over the box and one on the
        side it leans over, their points one cloud and a point seen where either saw it (``SeenInAnyFrame``). Red
        before: nothing held straight down, and the search ended there."""
        for tilt in (25.0, 35.0):
            with self.subTest(tilt=tilt):
                over, over_seen = _box(tilt, self.OVER)
                side, side_seen = _box(tilt, self.SHADOW_SIDE)
                rolled = _rolled(_grasps(np.vstack([over, side]), SeenInAnyFrame((over_seen, side_seen))))
                self.assertTrue(rolled, "no rolled grasp where the second camera saw the way under the box")
                for c in rolled:
                    roll = math.degrees(math.asin(min(1.0, abs(float(c.closing_axis[2])))))
                    self.assertIn(round(roll), {abs(r) for r in BODY_ROLLS_DEG})
                    # The closing axis runs across the leaning faces (along x), rolled toward them.
                    self.assertGreater(abs(float(c.closing_axis[0])), 0.8)


def _tilts(candidates: "list[Any]") -> "list[float]":
    return [round(_tilt_deg(c), 1) for c in candidates]


class TheHandLeansOnSfesWholeLadderTests(unittest.TestCase):
    """SFE's ladder to 90 degrees (``hand._TILTS_DEG``, the owner, 2026-10-16: "alles von SFE"): a lean wins
    only with more room, and a vertical grasp stays listed. Red before: the body leaned to 45 at most."""

    def _beside_a_wall(self, gap_mm: "float | None") -> "list[Any]":
        from src.robot.grasping.generation.hand import SIDE_APPROACH_SCORE_WEIGHTS
        from tests.robot.grasping.generation.test_a_side_grasp_never_goes_through_unseen_space import CYLINDER, HIGH_OVERHEAD, WALL, camera
        from tests._part_scenes import cylinder_beside_a_wall, hande_jaw

        cloud, wall = cylinder_beside_a_wall(gap_mm)
        solids = (CYLINDER,) if gap_mm is None else (CYLINDER, WALL)
        return body_grasps(cloud, support_height_mm=0.0, jaw=hande_jaw(), obstacle_points_base_mm=wall,
                           corridor_seen=camera(HIGH_OVERHEAD, *solids).seen, max_candidates=12,
                           score_weights=SIDE_APPROACH_SCORE_WEIGHTS)

    def test_beside_a_wall_the_hand_leans_away_past_45_and_a_vertical_stays_listed(self) -> None:
        found = self._beside_a_wall(12.0)
        self.assertTrue(found)
        self.assertGreater(_tilt_deg(found[0]), 45.0, _tilts(found))
        self.assertGreater(float(found[0].approach[1]), 0.0, "the tool travels toward the wall: the hand leans away")
        self.assertTrue(any(_tilt_deg(c) < 1.0 for c in found), "a vertical candidate is still listed")

    def test_without_the_wall_vertical_wins(self) -> None:
        found = self._beside_a_wall(None)
        self.assertTrue(found)
        self.assertLess(_tilt_deg(found[0]), 1.0, _tilts(found))


def _under_a_roof(**kwargs: Any) -> "list[Any]":
    """A 30 mm cube under a roof 76 mm over the support, reaching 50 mm either side of it along y and 120 along x, every
    point seen: the open fingers pass under the roof leaning 60 degrees, and at 45 meet it."""
    from src.robot.grasping.generation.hand import SIDE_APPROACH_SCORE_WEIGHTS
    from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _cube, _surface

    xs, ys = np.arange(-120.0, 120.0 + 1e-9, 2.0), np.arange(-50.0, 50.0 + 1e-9, 2.0)
    roof = np.array([(x, y, 76.0) for x in xs for y in ys])
    return body_grasps(_surface(_cube, 40.0), support_height_mm=0.0, jaw=_jaw(), obstacle_points_base_mm=roof,
                       corridor_seen=lambda points: np.ones(np.asarray(points).reshape(-1, 3).shape[0], dtype=bool),
                       score_weights=SIDE_APPROACH_SCORE_WEIGHTS, max_candidates=60, **kwargs)


class ThePalmStaysOverTheSupportTests(unittest.TestCase):
    """Leaning past 45 the palm becomes the hand's lowest part: SFE's ``palm_aware`` (``hand._palm_low_mm``)
    is asked of every grasp. Red before: under the roof every lean of 75 and 90 degrees was offered, its palm 7.5 and
    20 mm under the support, and ranked first for its room."""

    def test_under_a_roof_only_the_lean_whose_palm_clears_the_support_is_offered(self) -> None:
        from src.robot.grasping.generation.hand import _palm_low_mm

        found = _under_a_roof()
        self.assertTrue(found)
        self.assertEqual({60.0}, set(_tilts(found)))
        jaw = _jaw()
        for c in found:
            approach, axis = np.asarray(c.approach), np.asarray(c.closing_axis)
            binormal = np.cross(approach, axis)
            binormal /= np.linalg.norm(binormal)
            self.assertGreaterEqual(_palm_low_mm(np.asarray(c.position_mm), approach, binormal, jaw, axis),
                                    jaw.table_clearance_mm)


class SfesFinePassRunsWhereItRanTests(unittest.TestCase):
    """SFE's fine pass (``hand._FINE_BELOW``): where straight down and the ladder's leans leave fewer than 3
    square grasps, the leans every 7.5 degrees (``BODY_FINE_TILTS_DEG``) and the rolled frames, each leaning on the
    ladder as SFE rolls on every tilt of it."""

    def _asked(self, run: Any) -> "list[tuple[float, ...]]":
        from unittest import mock

        from src.robot.grasping.generation import body_grasps as bg

        asked: "list[tuple[float, ...]]" = []
        original = bg._side_approaches

        def side_approaches(grasps: Any, upright: float, checks: Any, tilts: "tuple[float, ...]") -> Any:
            asked.append(tuple(tilts))
            return original(grasps, upright, checks, tilts)

        with mock.patch.object(bg, "_side_approaches", side_approaches):
            run()
        return asked

    def test_with_few_grasps_the_fine_tilts_are_asked_and_with_many_they_are_not(self) -> None:
        from src.robot.grasping.generation.body_grasps import BODY_FINE_TILTS_DEG, BODY_TILTS_DEG
        from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _cube, _surface

        self.assertEqual((7.5, 22.5, 37.5, 52.5, 67.5, 82.5), BODY_FINE_TILTS_DEG)
        self.assertEqual((15.0, 30.0, 45.0, 60.0, 75.0, 90.0), BODY_TILTS_DEG)
        self.assertIn(BODY_FINE_TILTS_DEG, self._asked(_under_a_roof))
        cloud = _surface(_cube, 40.0)
        free = self._asked(lambda: _grasps(cloud, lambda p: np.ones(np.asarray(p).reshape(-1, 3).shape[0], dtype=bool)))
        self.assertEqual([BODY_TILTS_DEG], free)

    def test_a_rolled_grasp_leans_as_well_where_its_way_in_is_seen(self) -> None:
        over, over_seen = _box(25.0, ATiltedPartIsGraspedByItsFacesWhereItsWayInIsSeenTests.OVER)
        side, side_seen = _box(25.0, ATiltedPartIsGraspedByItsFacesWhereItsWayInIsSeenTests.SHADOW_SIDE)
        found = body_grasps(np.vstack([over, side]), support_height_mm=0.0, jaw=_jaw(),
                            corridor_seen=SeenInAnyFrame((over_seen, side_seen)),
                            measured_base_mm=np.vstack([over, side]), max_candidates=40)
        rolled = _rolled(found)
        self.assertTrue(rolled)
        # A roll alone tilts the approach as far as the closing axis; leaning on it tilts the approach further.
        both = [c for c in rolled
                if abs(_tilt_deg(c) - math.degrees(math.asin(min(1.0, abs(float(c.closing_axis[2])))))) > 1.0]
        self.assertTrue(both, [(round(_tilt_deg(c), 1), round(math.degrees(math.asin(abs(float(c.closing_axis[2])))), 1))
                               for c in rolled])


class AFingerGoesDownOnlyWhereACameraSawFreeTests(unittest.TestCase):
    """The owner, 2026-10-16: "nur Sachen anbieten, die auch wirklich funktionieren". Space no camera saw counts as
    standing up to the height a neighbour stands within ``_HIDDEN_BESIDE_MM`` of it, straight down too; beside a part
    alone its own shadow stays free. The scene: a 30 mm cube, a strip beside its +x face no camera saw up to 60 mm, and
    in the taller case a neighbour 60 mm high whose seen top begins 25 mm off the cube's face. Red before: a grasp
    closing along x stood its open finger in that strip, its outer face 4.2 mm from the neighbour's seen top (the owner's
    5 mm reach), as it did beside the towers on URSim (2026-10-16)."""

    @staticmethod
    def _seen(points: Any) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        return ~((p[:, 0] > 15.5) & (p[:, 0] < 40.0) & (p[:, 2] < 60.0))

    def _grasps(self, taller_beside: bool) -> "list[Any]":
        from src.robot.grasping.generation.hand import SIDE_APPROACH_SCORE_WEIGHTS
        from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _cube, _surface

        neighbour = None
        if taller_beside:
            xs, ys = np.arange(40.0, 49.0 + 1e-9, 2.0), np.arange(-30.0, 30.0 + 1e-9, 2.0)
            neighbour = np.array([(x, y, 60.0) for x in xs for y in ys])
        return body_grasps(_surface(_cube, 40.0), support_height_mm=0.0, jaw=_jaw(), obstacle_points_base_mm=neighbour,
                           corridor_seen=self._seen, max_candidates=60, score_weights=SIDE_APPROACH_SCORE_WEIGHTS)

    @staticmethod
    def _into_the_strip(found: "list[Any]") -> "list[Any]":
        return [c for c in found if abs(float(c.closing_axis[0])) > 0.9 and _tilt_deg(c) < 15.0]

    def test_beside_a_taller_neighbour_no_finger_goes_down_where_no_camera_saw(self) -> None:
        found = self._grasps(taller_beside=True)
        self.assertTrue(found, "the grasps closing along y keep their fingers out of the strip")
        self.assertEqual([], self._into_the_strip(found))

    def test_beside_a_part_alone_its_own_shadow_stays_free(self) -> None:
        self.assertTrue(self._into_the_strip(self._grasps(taller_beside=False)))


class AFreeCubeIsGraspedStraightDownFirstTests(unittest.TestCase):
    def test_vertical_wins_the_tie_the_leans_are_offered_and_nothing_rolls(self) -> None:
        from tests.robot.grasping.generation.test_the_hand_grasps_the_parts_own_body import _cube, _surface

        cloud = _surface(_cube, 40.0)
        found = _grasps(cloud, lambda points: np.ones(np.asarray(points).reshape(-1, 3).shape[0], dtype=bool))

        self.assertTrue(found)
        self.assertLess(_tilt_deg(found[0]), 1.0, "a lean took rank 0 on a cube nothing stands beside")
        self.assertEqual([], _rolled(found), "a cube with square grasps straight down was searched rolled")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
