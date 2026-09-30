"""The push planner on synthetic scenes: where the finger comes down, which way the part goes, and when it refuses.

`plan_push` is pure numpy. It runs inside the pick attempt, on BASE points the attempt already has: the failed
part, its neighbours, the support plane, the table the camera saw and the workspace box. These scenes are built
by hand on a level table at z = 0, so every number below can be checked with a ruler:

* The part is a 30 x 30 x 30 mm block centred on the origin (x and y from -15 to +15) unless a test says
  otherwise.
* The hand is the Hand-E as the registry gives it: finger 10.80 mm thick and 29.24 mm wide, fingertip
  10.45 mm past the TCP and 36.53 mm behind it, palm 75 mm wide, jaws open 49.99 mm. Its outer faces are
  35.8 mm from the TCP.
* The workspace is generous. Its floor is at z = -100 mm, so only the z_min tests reach it.
* ``SIDE`` is a 10 x 10 mm post whose near face is 15 mm off the part's +Y face, set back along -X. It is close
  enough to count as the neighbour that blocked the grasp (25 mm). It is outside the fingers' band when the part
  slides along X. Sliding +X takes the part out of its reach; sliding -X keeps it beside the post.

The owner's rules pinned here (2026-09-29):

* The push goes away from the neighbour that blocked the grasp.
* The fingertip rides at support + clamp(part height / 2, 10, 20) mm.
* A part under 15 mm is not pushed.
* Every TCP point stays 20 mm above z_min.
* The landing plus 15 mm stays inside the push box. That box is the workspace intersected with the seen table,
  shrunk by (push distance + 30 mm), or else a declared container.
* A push above 50 mm is refused.

The findings of the first review are pinned here too:

* The table the camera saw is kept cell by cell, not as the box around it. A landing past a diagonal table edge,
  into a hole or a shadow, or across a gap between two seen patches is refused, and so is a hand coming down
  where no table was seen. The part and its neighbours stand on the table and hide it, so their own footprints
  count as seen.
* A part must rest on the support: its lowest points lie within 10 mm of it. A part on something unsegmented,
  or seen only from above, is not pushed. The 15 mm rule applies to the part's own height.
* A part that reaches the palm's underside is not pushed: only the finger may touch it. The Hand-E housing is
  75 mm along the closing axis, wider than the open fingers, and the planner keeps it clear of tall neighbours.
* The push distance is resolved against the config: ``push_distance_mm`` (30 mm) when nobody asks, a request up to
  ``max_nudge_mm`` (50 mm) as asked, above it or above 50 mm refused, and never less than the 10 mm a push must
  open.
"""

from __future__ import annotations

import math
import tracemalloc
import unittest

import numpy as np

from src.robot.grasping.recovery import push_planner
from src.robot.grasping.recovery.push_planner import (
    AUTO_BOX_EXTRA_SHRINK_MM,
    BACK_OFF_MM,
    BACK_OFF_SPEED_MM_S,
    CONTACT_GAP_MM,
    DEFAULT_PUSH_DISTANCE_MM,
    DESCENT_SPEED_MM_S,
    FINGER_HEIGHT_MAX_MM,
    FINGER_HEIGHT_MIN_MM,
    LANDING_MARGIN_MM,
    LEG_ACCEL_M_S2,
    LIFT_SPEED_MM_S,
    MIN_CLEARANCE_GAIN_MM,
    MIN_PUSHABLE_PART_HEIGHT_MM,
    NEIGHBOUR_EVIDENCE_RADIUS_MM,
    PUSH_ACCEL_M_S2,
    PUSH_APPROACH_RISE_MM,
    PUSH_DISTANCE_CAP_MM,
    PUSH_SPEED_MM_S,
    WORKSPACE_MARGIN_MM,
    AxisBox,
    PushHand,
    PushPlan,
    PushRefusal,
    neighbour_evidence,
    plan_push,
    resolve_push_distance,
    swept_target_points_mm,
    table_points_from_cloud,
)

HAND_E = PushHand(
    finger_thickness_mm=10.80,
    finger_width_mm=29.24,
    fingertip_depth_mm=10.45,
    finger_length_mm=36.53,
    palm_width_mm=75.0,
    open_width_mm=49.99,
)
HALF_OUTER = 49.99 / 2 + 10.80

WORKSPACE = AxisBox(min_mm=(-800.0, -800.0, -100.0), max_mm=(800.0, 800.0, 600.0))
SHRINK = DEFAULT_PUSH_DISTANCE_MM + AUTO_BOX_EXTRA_SHRINK_MM
X_ONLY = ((1.0, 0.0),)


def _block(
    centre_xy: tuple[float, float] = (0.0, 0.0),
    size_xy: tuple[float, float] = (30.0, 30.0),
    height: float = 30.0,
    step: float = 2.0,
    z0: float = 0.0,
) -> np.ndarray:
    """The surface of a box whose bottom is at ``z0`` (the table unless a test says otherwise), top and four
    sides, sampled every ``step`` mm."""
    cx, cy = centre_xy
    sx, sy = size_xy
    xs = np.arange(cx - sx / 2, cx + sx / 2 + 1e-9, step)
    ys = np.arange(cy - sy / 2, cy + sy / 2 + 1e-9, step)
    zs = np.arange(z0, z0 + height + 1e-9, step)
    top = np.array([(x, y, z0 + height) for x in xs for y in ys])
    sides = [np.array([(x, cy - sy / 2, z) for x in xs for z in zs]),
             np.array([(x, cy + sy / 2, z) for x in xs for z in zs]),
             np.array([(cx - sx / 2, y, z) for y in ys for z in zs]),
             np.array([(cx + sx / 2, y, z) for y in ys for z in zs])]
    return np.vstack([top, *sides])


# Near face 15 mm off the part's +Y face (y = 30), spanning x = -25 .. -15.
SIDE = _block(centre_xy=(-20.0, 35.0), size_xy=(10.0, 10.0))
# The same post, mirrored to +X.
SIDE_MIRRORED = _block(centre_xy=(20.0, 35.0), size_xy=(10.0, 10.0))


def _table(x: tuple[float, float] = (-400.0, 400.0), y: tuple[float, float] = (-400.0, 400.0),
           step: float = 5.0) -> np.ndarray:
    xs = np.arange(x[0], x[1] + 1e-9, step)
    ys = np.arange(y[0], y[1] + 1e-9, step)
    return np.array([(a, b, 0.0) for a in xs for b in ys])


TABLE = _table()


def _plan(**overrides: object) -> PushPlan | PushRefusal:
    kwargs: dict[str, object] = dict(
        target_points_mm=_block(),
        neighbour_points_mm=SIDE,
        support_normal=(0.0, 0.0, 1.0),
        support_offset_mm=0.0,
        workspace=WORKSPACE,
        table_points_mm=TABLE,
        hand=HAND_E,
    )
    kwargs.update(overrides)
    return plan_push(**kwargs)  # type: ignore[arg-type]


def _refused(case: unittest.TestCase, result: PushPlan | PushRefusal, code: str) -> PushRefusal:
    if isinstance(result, PushPlan):
        case.fail(f"planned a push toward {result.direction} where {code!r} was expected")
    assert isinstance(result, PushRefusal)
    case.assertEqual(result.code, code, result.sentence)
    case.assertTrue(result.sentence.endswith("."), result.sentence)
    return result


def _planned(case: unittest.TestCase, result: PushPlan | PushRefusal) -> PushPlan:
    if isinstance(result, PushRefusal):
        case.fail(f"refused {result.code}: {result.sentence} {[v.to_dict() for v in result.candidates]}")
    assert isinstance(result, PushPlan)
    return result


def _verdict(refusal: PushRefusal, direction_xy: tuple[float, float]) -> str:
    for candidate in refusal.candidates:
        if np.allclose(candidate.direction_xy, direction_xy, atol=1e-9):
            return candidate.verdict
    raise AssertionError(f"no candidate for {direction_xy}")


class TheDirectionGoesAwayFromTheNeighbour(unittest.TestCase):
    def test_along_x_the_part_slides_out_of_the_neighbours_reach(self) -> None:
        # +X: the finger comes down on the free -X side and the part leaves the post behind (gap 15 -> 33.5 mm).
        # -X: the part stays beside the post, the gap stays 15 mm, nothing gained.
        plan = _planned(self, _plan(push_axes_xy=X_ONLY))
        np.testing.assert_allclose(plan.direction, (1.0, 0.0, 0.0), atol=1e-9)
        self.assertAlmostEqual(plan.clearance_before_mm, 15.0, delta=1.0)
        self.assertAlmostEqual(plan.clearance_after_mm, math.hypot(30.0, 15.0), delta=1.5)
        verdicts = {v.direction_xy: v.verdict for v in plan.candidates}
        self.assertEqual(verdicts[(-1.0, 0.0)], "no_clearance_gain")

    def test_the_mirrored_scene_pushes_the_mirrored_way(self) -> None:
        plan = _planned(self, _plan(neighbour_points_mm=SIDE_MIRRORED, push_axes_xy=X_ONLY))
        np.testing.assert_allclose(plan.direction, (-1.0, 0.0, 0.0), atol=1e-9)

    def test_with_every_direction_the_part_moves_away_from_a_neighbour_at_its_corner(self) -> None:
        # A post 12 mm diagonally off the part's (-X, +Y) corner. Straight away, (+1, -1), would bring the
        # finger down right on the post, so the planner slides the part +X or -Y. Both lead away from the post
        # and both gain about 27 mm. The diagonals across the post's line gain 16 mm and are not chosen.
        corner = _block(centre_xy=(-32.0, 32.0), size_xy=(10.0, 10.0))
        plan = _planned(self, _plan(neighbour_points_mm=corner))
        away = np.array([1.0, -1.0]) / math.sqrt(2.0)
        self.assertGreater(float(np.dot(plan.direction[:2], away)), 0.5)
        self.assertGreater(plan.clearance_gain_mm, 20.0)
        mirrored = _planned(self, _plan(neighbour_points_mm=_block(centre_xy=(32.0, -32.0), size_xy=(10.0, 10.0))))
        self.assertGreater(float(np.dot(mirrored.direction[:2], -away)), 0.5)

    def test_a_neighbour_close_behind_leaves_no_direction_free(self) -> None:
        # A 30 mm block 20 mm off the part's -X face. Pushing +X needs the finger in that 20 mm gap. Pushing -X
        # runs the part into the block. Sliding along Y keeps the block beside it. Nothing is guessed.
        refusal = _refused(self, _plan(neighbour_points_mm=_block(centre_xy=(-50.0, 0.0))), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_blocked")
        self.assertEqual(_verdict(refusal, (-1.0, 0.0)), "path_blocked")
        self.assertEqual(_verdict(refusal, (0.0, 1.0)), "no_clearance_gain")
        self.assertEqual(len(refusal.candidates), 8)
        self.assertIn("hand_blocked", refusal.sentence)

    def test_only_the_axes_the_caller_offers_are_tried(self) -> None:
        refusal = _refused(self, _plan(push_axes_xy=((0.0, 1.0),)), "no_free_direction")
        self.assertEqual(sorted(v.direction_xy for v in refusal.candidates), [(0.0, -1.0), (0.0, 1.0)])

    def test_a_tall_neighbour_under_the_palm_blocks_what_a_short_one_does_not(self) -> None:
        # The palm (75 mm wide) passes over the post while the fingers slide the part +X. A post taller than
        # the palm's underside (15 + 10.45 + 36.53 = 62 mm above the table, less the 10 mm clearance) is hit.
        tall = _block(centre_xy=(-20.0, 35.0), size_xy=(10.0, 10.0), height=70.0)
        refusal = _refused(self, _plan(neighbour_points_mm=tall, push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_blocked")


class WhereTheFingerGoes(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _planned(self, _plan(push_axes_xy=X_ONLY))

    def test_the_leading_finger_comes_down_the_gap_behind_the_part(self) -> None:
        # The part's trailing face is x = -15. The leading finger's outer face comes down CONTACT_GAP_MM behind
        # it, and the TCP sits HALF_OUTER behind that face.
        start = self.plan.contact_start_tcp_mm
        self.assertAlmostEqual(start[0], -15.0 - CONTACT_GAP_MM - HALF_OUTER, places=6)
        self.assertAlmostEqual(start[1], 0.0, delta=1.0)  # on the line through the part's centre

    def test_the_stations_follow_the_push_line(self) -> None:
        plan = self.plan
        self.assertAlmostEqual(plan.end_tcp_mm[0] - plan.contact_start_tcp_mm[0],
                               CONTACT_GAP_MM + DEFAULT_PUSH_DISTANCE_MM, places=6)
        self.assertAlmostEqual(plan.end_tcp_mm[0] - plan.back_off_tcp_mm[0], 5.0, places=6)
        self.assertAlmostEqual(plan.lift_tcp_mm[2] - plan.back_off_tcp_mm[2], PUSH_APPROACH_RISE_MM, places=6)
        self.assertAlmostEqual(plan.approach_tcp_mm[2] - plan.contact_start_tcp_mm[2], PUSH_APPROACH_RISE_MM,
                               places=6)
        np.testing.assert_allclose(plan.approach_tcp_mm[:2], plan.contact_start_tcp_mm[:2], atol=1e-9)
        np.testing.assert_allclose(plan.lift_tcp_mm[:2], plan.back_off_tcp_mm[:2], atol=1e-9)
        np.testing.assert_allclose(plan.approach, (0.0, 0.0, -1.0), atol=1e-12)
        for point in (plan.contact_start_tcp_mm, plan.end_tcp_mm, plan.back_off_tcp_mm):
            self.assertAlmostEqual(point[2], plan.finger_height_mm + 10.45, places=6)

    def test_the_finger_height_is_half_the_part_clamped_to_ten_and_twenty(self) -> None:
        for height, expected in ((16.0, FINGER_HEIGHT_MIN_MM), (30.0, 15.0), (60.0, FINGER_HEIGHT_MAX_MM)):
            with self.subTest(height=height):
                plan = _planned(self, _plan(target_points_mm=_block(height=height), push_axes_xy=X_ONLY))
                self.assertAlmostEqual(plan.part_height_mm, height, delta=0.5)
                self.assertAlmostEqual(plan.finger_height_mm, expected, delta=0.25)
                # The TCP rides the fingertip's reach above the fingertip.
                self.assertAlmostEqual(plan.contact_start_tcp_mm[2], plan.finger_height_mm + 10.45, places=6)

    def test_the_part_is_predicted_to_land_the_push_distance_along_the_direction(self) -> None:
        plan = _planned(self, _plan(push_axes_xy=X_ONLY, push_distance_mm=20.0))
        np.testing.assert_allclose(
            np.asarray(plan.predicted_landing_centre_mm) - np.asarray(plan.target_centre_mm),
            (20.0, 0.0, 0.0), atol=1e-9)
        self.assertEqual(plan.push_distance_mm, 20.0)

    def test_the_legs_carry_the_owners_speeds_in_the_units_the_arm_takes(self) -> None:
        # arm.move(linear=True, vel=..., acc=...) takes m/s and m/s^2. A leg carries nothing else, so a
        # millimetre speed can never reach the controller as 25 m/s.
        legs = {leg.name: leg for leg in self.plan.legs}
        self.assertEqual(tuple(legs), ("down", "push", "back", "up"))
        self.assertEqual((legs["push"].vel_m_s, legs["push"].acc_m_s2), (0.025, 0.1))
        for name in ("down", "back", "up"):
            with self.subTest(leg=name):
                self.assertEqual((legs[name].vel_m_s, legs[name].acc_m_s2), (0.05, 0.1))
        for leg in legs.values():
            self.assertLessEqual(leg.vel_m_s, 0.05)
            self.assertFalse(hasattr(leg, "speed_mm_s"))
        self.assertEqual(legs["down"].start_tcp_mm, self.plan.approach_tcp_mm)
        self.assertEqual(legs["push"].start_tcp_mm, self.plan.contact_start_tcp_mm)
        self.assertEqual(legs["up"].end_tcp_mm, self.plan.lift_tcp_mm)

    def test_the_swept_points_cover_the_whole_travel(self) -> None:
        swept = swept_target_points_mm(_block(), self.plan.direction, self.plan.push_distance_mm)
        self.assertAlmostEqual(float(swept[:, 0].min()), -15.0, places=6)
        self.assertAlmostEqual(float(swept[:, 0].max()), 15.0 + self.plan.push_distance_mm, places=6)
        np.testing.assert_allclose(swept[:, 1:].min(axis=0), (-15.0, 0.0))

    def test_the_plan_serialises_to_plain_values(self) -> None:
        data = self.plan.to_dict()
        self.assertEqual(data["direction"], [1.0, 0.0, 0.0])
        self.assertEqual(len(data["candidates"]), 2)
        self.assertIsInstance(data["clearance_after_mm"], float)


class WhenThePlannerRefuses(unittest.TestCase):
    def test_a_flat_part_is_not_pushed(self) -> None:
        refusal = _refused(self, _plan(target_points_mm=_block(height=12.0)), "part_too_flat")
        self.assertIn("15 mm", refusal.sentence)

    def test_a_push_above_fifty_millimetres_is_refused(self) -> None:
        refusal = _refused(self, _plan(push_distance_mm=50.5), "push_distance_above_cap")
        self.assertIn("50 mm", refusal.sentence)
        plan = _planned(self, _plan(push_distance_mm=PUSH_DISTANCE_CAP_MM, push_axes_xy=X_ONLY))
        self.assertEqual(plan.push_distance_mm, 50.0)

    def test_a_push_distance_that_is_not_a_positive_number_is_refused(self) -> None:
        for bad in (0.0, -5.0, math.nan, math.inf):
            with self.subTest(distance=bad):
                _refused(self, _plan(push_distance_mm=bad), "push_distance_invalid")

    def test_every_push_height_stays_twenty_millimetres_above_z_min(self) -> None:
        # The TCP rides at 15 + 10.45 = 25.45 mm. A z_min of 5.45 is exactly 20 below that and passes; 6 does not.
        ok = AxisBox(min_mm=(-800.0, -800.0, 5.45), max_mm=(800.0, 800.0, 600.0))
        _planned(self, _plan(workspace=ok, push_axes_xy=X_ONLY))
        low = AxisBox(min_mm=(-800.0, -800.0, 6.0), max_mm=(800.0, 800.0, 600.0))
        refusal = _refused(self, _plan(workspace=low), "below_z_min")
        self.assertIn("z_min", refusal.sentence)

    def test_the_repo_z_min_of_one_hundred_millimetres_refuses_every_push(self) -> None:
        repo = AxisBox(min_mm=(-500.0, -500.0, 100.0), max_mm=(500.0, 500.0, 600.0))
        _refused(self, _plan(workspace=repo), "below_z_min")

    def test_a_lift_above_z_max_is_refused(self) -> None:
        # The lift point is 25.45 + 80 = 105.45 mm; z_max must be at least 20 mm above that.
        low_ceiling = AxisBox(min_mm=(-800.0, -800.0, -100.0), max_mm=(800.0, 800.0, 120.0))
        _refused(self, _plan(workspace=low_ceiling), "above_z_max")

    def test_no_neighbour_within_twenty_five_millimetres_means_no_reason_to_push(self) -> None:
        far = _block(centre_xy=(0.0, 15.0 + 26.0 + 15.0))
        refusal = _refused(self, _plan(neighbour_points_mm=far), "no_blocking_neighbour")
        self.assertIn("25 mm", refusal.sentence)
        _refused(self, _plan(neighbour_points_mm=np.zeros((0, 3))), "no_blocking_neighbour")

    def test_a_landing_outside_the_box_is_refused(self) -> None:
        # The table the camera saw ends 40 mm past the part's +X face: shrunk by 30 + 30 mm, the box stops
        # short of the part itself.
        table = _table(x=(-400.0, 55.0))
        refusal = _refused(self, _plan(table_points_mm=table, push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "landing_outside_box")
        self.assertIn("landing_outside_box", refusal.sentence)

    def test_no_table_seen_and_no_container_means_no_box(self) -> None:
        _refused(self, _plan(table_points_mm=None), "no_table_seen")
        _refused(self, _plan(table_points_mm=_table(x=(-10.0, 10.0), y=(-10.0, 10.0), step=10.0)), "no_table_seen")

    def test_a_table_region_smaller_than_the_shrink_leaves_no_box(self) -> None:
        _refused(self, _plan(table_points_mm=_table(x=(-50.0, 50.0), y=(-50.0, 50.0))), "push_box_empty")

    def test_a_tilted_support_is_not_pushed_on(self) -> None:
        tilt = math.radians(8.0)
        refusal = _refused(self, _plan(support_normal=(math.sin(tilt), 0.0, math.cos(tilt))), "support_tilted")
        self.assertIn("5 deg", refusal.sentence)

    def test_a_slight_tilt_is_planned_on_with_the_push_in_the_plane(self) -> None:
        tilt = math.radians(2.0)
        normal = (math.sin(tilt), 0.0, math.cos(tilt))
        plan = _planned(self, _plan(support_normal=normal, push_axes_xy=X_ONLY))
        self.assertAlmostEqual(float(np.dot(plan.direction, normal)), 0.0, places=9)
        np.testing.assert_allclose(plan.approach, -np.asarray(normal), atol=1e-12)

    def test_no_target_points_above_the_support(self) -> None:
        _refused(self, _plan(target_points_mm=_table(x=(-20.0, 20.0), y=(-20.0, 20.0), step=2.0)),
                 "no_target_points")

    def test_no_axis_is_no_push(self) -> None:
        _refused(self, _plan(push_axes_xy=((0.0, 0.0),)), "no_push_axis")

    def test_a_refusal_serialises_with_what_each_direction_met(self) -> None:
        refusal = _refused(self, _plan(neighbour_points_mm=_block(centre_xy=(-40.0, 0.0))), "no_free_direction")
        data = refusal.to_dict()
        self.assertEqual(data["code"], "no_free_direction")
        self.assertEqual(len(data["candidates"]), 8)
        self.assertTrue(all(c["verdict"] != "ok" for c in data["candidates"]))


class ThePushBox(unittest.TestCase):
    def test_the_automatic_box_is_workspace_and_seen_table_shrunk_by_the_push_plus_thirty(self) -> None:
        # The table was seen from x = -400 to +300, the workspace reaches +500. So the box ends near
        # 300 - (30 + 30) = 240. The seen region is trimmed by half a percent of its points on each side,
        # which removes a stray point and nothing from a table seen evenly.
        table = _table(x=(-400.0, 300.0))
        plan = _planned(self, _plan(table_points_mm=table, push_axes_xy=X_ONLY,
                                    workspace=AxisBox(min_mm=(-500.0, -500.0, -100.0),
                                                      max_mm=(500.0, 500.0, 600.0))))
        (lo_x, lo_y), (hi_x, hi_y) = plan.landing_box_xy_mm
        self.assertAlmostEqual(hi_x, 300.0 - SHRINK, delta=6.0)
        self.assertAlmostEqual(lo_x, -400.0 + SHRINK, delta=6.0)
        self.assertAlmostEqual(hi_y, 400.0 - SHRINK, delta=6.0)
        self.assertAlmostEqual(lo_y, -400.0 + SHRINK, delta=6.0)

    def test_the_shrink_grows_with_the_push_distance(self) -> None:
        near = _planned(self, _plan(push_axes_xy=X_ONLY, push_distance_mm=20.0))
        far = _planned(self, _plan(push_axes_xy=X_ONLY, push_distance_mm=40.0))
        self.assertAlmostEqual(near.landing_box_xy_mm[1][0] - far.landing_box_xy_mm[1][0], 20.0, places=6)

    def test_the_workspace_limits_the_box_where_it_is_tighter_than_the_table(self) -> None:
        plan = _planned(self, _plan(push_axes_xy=X_ONLY,
                                    workspace=AxisBox(min_mm=(-200.0, -800.0, -100.0),
                                                      max_mm=(200.0, 800.0, 600.0))))
        (lo_x, _), (hi_x, _) = plan.landing_box_xy_mm
        self.assertAlmostEqual(hi_x, 200.0 - SHRINK, places=6)
        self.assertAlmostEqual(lo_x, -200.0 + SHRINK, places=6)

    def test_the_landing_margin_is_fifteen_millimetres(self) -> None:
        # The landing's +X face is 15 + 30 = 45 mm; with the 15 mm margin the box has to reach 60 mm, which
        # a workspace edge at 60 + SHRINK gives.
        edge = 45.0 + LANDING_MARGIN_MM + SHRINK
        for ws_x, fits in ((edge + 0.5, True), (edge - 0.5, False)):
            with self.subTest(workspace_x_max=ws_x):
                result = _plan(push_axes_xy=X_ONLY,
                               workspace=AxisBox(min_mm=(-800.0, -800.0, -100.0), max_mm=(ws_x, 800.0, 600.0)))
                if fits:
                    _planned(self, result)
                else:
                    self.assertEqual(_verdict(_refused(self, result, "no_free_direction"), (1.0, 0.0)),
                                     "landing_outside_box")

    def test_the_hand_comes_down_only_over_table_the_camera_saw(self) -> None:
        # The table was seen only from x = -70: the part lands well inside, but the hand, which comes down
        # 96.6 mm behind the part's centre, would come down where the camera saw no table.
        table = _table(x=(-70.0, 400.0))
        refusal = _refused(self, _plan(table_points_mm=table, push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_outside_region")

    def test_a_declared_container_is_the_box_and_no_table_is_needed(self) -> None:
        interior = AxisBox(min_mm=(-150.0, -150.0, 0.0), max_mm=(150.0, 150.0, 120.0))
        plan = _planned(self, _plan(table_points_mm=None, container_interior=interior, push_axes_xy=X_ONLY))
        self.assertEqual(plan.landing_box_xy_mm, ((-150.0, -150.0), (150.0, 150.0)))

    def test_a_container_too_small_for_the_hand_refuses(self) -> None:
        interior = AxisBox(min_mm=(-60.0, -60.0, 0.0), max_mm=(60.0, 60.0, 120.0))
        refusal = _refused(self, _plan(table_points_mm=None, container_interior=interior), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_outside_region")

    def test_an_operator_box_only_narrows(self) -> None:
        narrow = AxisBox(min_mm=(-300.0, -300.0, -100.0), max_mm=(300.0, 300.0, 600.0))
        plan = _planned(self, _plan(operator_box=narrow, push_axes_xy=X_ONLY))
        (lo_x, _), (hi_x, _) = plan.landing_box_xy_mm
        self.assertAlmostEqual(hi_x, 300.0 - SHRINK, places=6)
        self.assertAlmostEqual(lo_x, -300.0 + SHRINK, places=6)
        wide = AxisBox(min_mm=(-5000.0, -5000.0, -100.0), max_mm=(5000.0, 5000.0, 600.0))
        self.assertEqual(_planned(self, _plan(operator_box=wide, push_axes_xy=X_ONLY)).landing_box_xy_mm,
                         _planned(self, _plan(push_axes_xy=X_ONLY)).landing_box_xy_mm)


class NeighbourEvidence(unittest.TestCase):
    def test_a_neighbour_within_twenty_five_millimetres_is_evidence(self) -> None:
        near = neighbour_evidence(target_points_mm=_block(), neighbour_points_mm=_block(centre_xy=(0.0, 30.0 + 24.0)))
        self.assertTrue(near.found)
        self.assertAlmostEqual(near.nearest_gap_mm, 24.0, delta=0.5)
        far = neighbour_evidence(target_points_mm=_block(), neighbour_points_mm=_block(centre_xy=(0.0, 30.0 + 26.0)))
        self.assertFalse(far.found)
        self.assertAlmostEqual(far.nearest_gap_mm, 26.0, delta=0.5)
        self.assertEqual(far.radius_mm, 25.0)

    def test_table_points_are_not_a_neighbour(self) -> None:
        # A mask that bled onto the table right next to the part: its points lie within 5 mm of the support.
        bleed = np.array([(x, 17.0, 1.0) for x in np.arange(-15.0, 15.0, 1.0)])
        self.assertFalse(neighbour_evidence(target_points_mm=_block(), neighbour_points_mm=bleed).found)

    def test_no_neighbours_is_no_evidence(self) -> None:
        result = neighbour_evidence(target_points_mm=_block(), neighbour_points_mm=np.zeros((0, 3)))
        self.assertFalse(result.found)
        self.assertEqual(result.nearest_gap_mm, math.inf)


class TablePointsFromACloud(unittest.TestCase):
    def test_only_points_on_the_support_are_table(self) -> None:
        table = _table(x=(-50.0, 50.0), y=(-50.0, 50.0), step=10.0)
        found = table_points_from_cloud(np.vstack([table, _block()]),
                                        support_normal=(0.0, 0.0, 1.0), support_offset_mm=0.0)
        self.assertTrue(np.all(np.abs(found[:, 2]) <= 5.0))
        self.assertGreaterEqual(len(found), len(table))

    def test_points_below_the_support_are_not_table_either(self) -> None:
        # A hole, a step down or depth noise under the plane: within 5 mm either side is table, beyond it is not.
        below = np.array([(0.0, 0.0, -4.9), (10.0, 0.0, -5.1), (20.0, 0.0, -30.0)])
        found = table_points_from_cloud(below, support_normal=(0.0, 0.0, 1.0), support_offset_mm=0.0)
        np.testing.assert_allclose(found, [(0.0, 0.0, -4.9)])


class TheHand(unittest.TestCase):
    def test_the_hand_reads_a_gripper_model(self) -> None:
        class _Model:
            finger_thickness_mm = 10.80
            finger_width_mm = 29.24
            fingertip_depth_mm = 10.45
            finger_length_mm = 36.53
            palm_width_mm = 75.0

        self.assertEqual(PushHand.from_gripper_model(_Model(), open_width_mm=49.99), HAND_E)

    def test_a_hand_with_a_non_positive_size_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            PushHand(finger_thickness_mm=0.0, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                     finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99)

    def test_a_degenerate_box_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            AxisBox(min_mm=(0.0, 0.0, 0.0), max_mm=(10.0, 0.0, 10.0))



def _tilted(points: np.ndarray, tilt_deg: float) -> np.ndarray:
    """A level scene turned about BASE Y so the support rises toward +X by ``tilt_deg``."""
    t = math.radians(tilt_deg)
    turn = np.array([[math.cos(t), 0.0, -math.sin(t)], [0.0, 1.0, 0.0], [math.sin(t), 0.0, math.cos(t)]])
    return np.asarray(points, dtype=np.float64) @ turn.T


def _tilted_normal(tilt_deg: float) -> tuple[float, float, float]:
    t = math.radians(tilt_deg)
    return (-math.sin(t), 0.0, math.cos(t))


class TheOwnersNumbers(unittest.TestCase):
    def test_every_number_the_owner_set_is_pinned_by_its_value(self) -> None:
        # The other tests read these from the module, so a changed number would move them along. Not here.
        self.assertEqual((DEFAULT_PUSH_DISTANCE_MM, PUSH_DISTANCE_CAP_MM), (30.0, 50.0))
        self.assertEqual(MIN_PUSHABLE_PART_HEIGHT_MM, 15.0)
        self.assertEqual((FINGER_HEIGHT_MIN_MM, FINGER_HEIGHT_MAX_MM), (10.0, 20.0))
        self.assertEqual(WORKSPACE_MARGIN_MM, 20.0)
        self.assertEqual(NEIGHBOUR_EVIDENCE_RADIUS_MM, 25.0)
        self.assertEqual(AUTO_BOX_EXTRA_SHRINK_MM, 30.0)
        self.assertEqual(LANDING_MARGIN_MM, 15.0)
        self.assertEqual(BACK_OFF_MM, 5.0)
        self.assertEqual((PUSH_SPEED_MM_S, PUSH_ACCEL_M_S2), (25.0, 0.1))
        self.assertEqual((DESCENT_SPEED_MM_S, BACK_OFF_SPEED_MM_S, LIFT_SPEED_MM_S, LEG_ACCEL_M_S2),
                         (50.0, 50.0, 50.0, 0.1))
        self.assertEqual(MIN_CLEARANCE_GAIN_MM, 10.0)

    def test_every_reason_code_is_exported_for_the_pick_attempt_and_the_telemetry(self) -> None:
        codes = [name for name in dir(push_planner) if name.startswith(("REFUSED_", "DIRECTION_"))]
        self.assertGreaterEqual(len(codes), 20)
        for name in codes + ["MAX_SUPPORT_TILT_DEG", "DESCENT_SPEED_MM_S", "BACK_OFF_SPEED_MM_S",
                             "LIFT_SPEED_MM_S", "LEG_ACCEL_M_S2", "resolve_push_distance"]:
            with self.subTest(name=name):
                self.assertIn(name, push_planner.__all__)


class TheChoiceAndTheStations(unittest.TestCase):
    def test_the_direction_that_gains_the_most_room_wins_not_the_first_that_passes(self) -> None:
        # With every default axis, +X passes first (the part slides past the post, 18.5 mm gained) and the
        # diagonal away from the post passes later with more (21.2 mm). The diagonal is chosen.
        plan = _planned(self, _plan())
        passed = [v for v in plan.candidates if v.verdict == "ok"]
        self.assertGreaterEqual(len(passed), 2)
        gains = [float(v.clearance_gain_mm or 0.0) for v in passed]
        self.assertGreater(max(gains), gains[0] + 1.0)
        self.assertAlmostEqual(plan.clearance_gain_mm, max(gains), places=6)
        np.testing.assert_allclose(plan.direction[:2], (-math.sqrt(0.5), -math.sqrt(0.5)), atol=1e-9)

    def test_on_a_tilted_support_every_station_stays_twenty_millimetres_above_z_min(self) -> None:
        # A support rising 4 deg toward +X. Pushing +X, the finger comes down about 61 mm behind the part's
        # centre, where the support is 4.2 mm lower. A z_min that the centre clears by 3 mm refuses the push:
        # the contact start would run the TCP less than 20 mm above it.
        tilt = 4.0
        normal = _tilted_normal(tilt)
        scene = dict(target_points_mm=_tilted(_block(), tilt), neighbour_points_mm=_tilted(SIDE, tilt),
                     table_points_mm=_tilted(TABLE, tilt), support_normal=normal, push_axes_xy=X_ONLY)
        free = _planned(self, _plan(**scene))
        centre_tcp_z = (free.finger_height_mm + 10.45) * math.cos(math.radians(tilt))
        stations_z = (free.contact_start_tcp_mm[2], free.end_tcp_mm[2], free.back_off_tcp_mm[2])
        self.assertLess(min(stations_z), centre_tcp_z - 4.0)
        z_min = centre_tcp_z - 3.0 - WORKSPACE_MARGIN_MM
        workspace = AxisBox(min_mm=(-800.0, -800.0, z_min), max_mm=(800.0, 800.0, 600.0))
        refusal = _refused(self, _plan(workspace=workspace, **scene), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "tcp_outside_workspace")


class TheSeenTableIsTheCellsTheCameraSaw(unittest.TestCase):
    """The table the camera saw is kept as 5 mm cells, not as the box around its points."""

    def test_a_landing_past_a_diagonal_table_edge_is_refused(self) -> None:
        # The table ends along x + y = 100. The box around its points would reach (300, 300) and let the part
        # land 7 mm past the edge; the cells do not.
        table = _table(x=(-300.0, 300.0), y=(-300.0, 300.0))
        table = table[table[:, 0] + table[:, 1] < 100.0]
        for distance in (30.0, 50.0):
            with self.subTest(distance=distance):
                refusal = _refused(self, _plan(target_points_mm=_block(centre_xy=(40.0, 20.0)),
                                               neighbour_points_mm=_block(centre_xy=(20.0, 55.0), size_xy=(10.0, 10.0)),
                                               table_points_mm=table, push_distance_mm=distance, push_axes_xy=X_ONLY),
                                   "no_free_direction")
                self.assertEqual(_verdict(refusal, (1.0, 0.0)), "landing_over_unseen_table")

    def test_a_landing_into_a_hole_the_camera_did_not_see_is_refused(self) -> None:
        table = TABLE[~((TABLE[:, 0] > 40.0) & (TABLE[:, 0] < 120.0) & (np.abs(TABLE[:, 1]) < 60.0))]
        refusal = _refused(self, _plan(table_points_mm=table, push_distance_mm=50.0, push_axes_xy=X_ONLY),
                           "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "landing_over_unseen_table")

    def test_a_shadow_behind_the_part_is_not_table(self) -> None:
        # The 45 deg wrist camera cannot see the table right behind a 30 mm part. Whatever lies there, the
        # part is not pushed into it: another view that sees it has to add its points first.
        shadow = (TABLE[:, 0] > 15.0) & (TABLE[:, 0] < 45.0) & (np.abs(TABLE[:, 1]) < 15.0)
        refusal = _refused(self, _plan(table_points_mm=TABLE[~shadow], push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "landing_over_unseen_table")
        _planned(self, _plan(table_points_mm=np.vstack([TABLE[~shadow], TABLE[shadow]]), push_axes_xy=X_ONLY))

    def test_a_landing_across_a_gap_between_two_seen_patches_is_refused(self) -> None:
        table = np.vstack([_table(x=(-400.0, 0.0)), _table(x=(200.0, 400.0))])
        refusal = _refused(self, _plan(target_points_mm=_block(centre_xy=(-40.0, 0.0)),
                                       neighbour_points_mm=_block(centre_xy=(-60.0, 35.0), size_xy=(10.0, 10.0)),
                                       table_points_mm=table, push_distance_mm=50.0, push_axes_xy=X_ONLY),
                           "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "landing_over_unseen_table")

    def test_a_hand_coming_down_over_a_gap_is_refused(self) -> None:
        # The part stands on the right patch; pushing +X the hand comes down 97 mm behind its centre, over
        # the gap where no table was seen.
        table = np.vstack([_table(x=(-400.0, 0.0)), _table(x=(200.0, 400.0))])
        refusal = _refused(self, _plan(target_points_mm=_block(centre_xy=(240.0, 0.0)),
                                       neighbour_points_mm=_block(centre_xy=(220.0, 35.0), size_xy=(10.0, 10.0)),
                                       table_points_mm=table, push_axes_xy=X_ONLY), "no_free_direction")
        self.assertIn(_verdict(refusal, (1.0, 0.0)), ("landing_over_unseen_table", "hand_over_unseen_table"))

    def test_a_hole_only_under_the_hand_is_refused_for_the_hand(self) -> None:
        # The landing and its 75 mm window are all seen; only where the fingers come down, 70 to 90 mm behind
        # the part's centre, the camera saw no table.
        hole = (TABLE[:, 0] > -90.0) & (TABLE[:, 0] < -70.0) & (np.abs(TABLE[:, 1]) < 20.0)
        refusal = _refused(self, _plan(table_points_mm=TABLE[~hole], push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_over_unseen_table")

    def test_twenty_points_at_the_rim_of_the_view_are_not_a_table(self) -> None:
        rim = np.array([(x, y, 0.0) for x in np.linspace(-400.0, 400.0, 5) for y in (-400.0, 400.0)]
                       + [(x, y, 0.0) for x in (-400.0, 400.0) for y in np.linspace(-300.0, 300.0, 5)])
        self.assertEqual(len(rim), 20)
        _refused(self, _plan(table_points_mm=rim, push_axes_xy=X_ONLY), "no_table_seen")

    def test_the_table_under_the_part_and_its_neighbours_counts_as_seen(self) -> None:
        # They stand on the table and hide it: no camera sees the table under a part.
        under_part = (np.abs(TABLE[:, 0]) <= 15.0) & (np.abs(TABLE[:, 1]) <= 15.0)
        under_post = (TABLE[:, 0] >= -25.0) & (TABLE[:, 0] <= -15.0) & (TABLE[:, 1] >= 30.0) & (TABLE[:, 1] <= 40.0)
        _planned(self, _plan(table_points_mm=TABLE[~(under_part | under_post)], push_axes_xy=X_ONLY))


class ThePartMustRestOnTheSupport(unittest.TestCase):
    def test_a_part_on_something_unsegmented_is_not_pushed(self) -> None:
        # A 30 mm part on a 40 mm box nobody segmented: counted from the table it looks 70 mm tall, and the
        # finger would ride at 20 mm, under the part, into the box.
        refusal = _refused(self, _plan(target_points_mm=_block(z0=40.0), push_axes_xy=X_ONLY), "part_not_on_support")
        self.assertIn("10 mm", refusal.sentence)

    def test_a_thin_washer_on_an_unsegmented_block_is_not_pushed(self) -> None:
        washer = _block(height=4.0, z0=25.0, step=1.0)
        _refused(self, _plan(target_points_mm=washer, push_axes_xy=X_ONLY), "part_not_on_support")

    def test_a_part_seen_only_from_above_is_not_pushed(self) -> None:
        # Where it rests cannot be told from its top face alone, so it fails closed. Another view that sees a
        # side lets it through.
        block = _block()
        top_only = block[block[:, 2] >= 29.9]
        _refused(self, _plan(target_points_mm=top_only, push_axes_xy=X_ONLY), "part_not_on_support")
        _planned(self, _plan(target_points_mm=block, push_axes_xy=X_ONLY))

    def test_the_fifteen_millimetre_rule_is_the_parts_own_height(self) -> None:
        # A 12 mm part on an 8 mm plate: it rests within the finger's reach of the support, and its top is
        # 20 mm up, but the part itself is 12 mm tall.
        refusal = _refused(self, _plan(target_points_mm=_block(height=12.0, z0=8.0), push_axes_xy=X_ONLY),
                           "part_too_flat")
        self.assertIn("12 mm", refusal.sentence)

    def test_the_plan_says_where_the_part_rests(self) -> None:
        plan = _planned(self, _plan(push_axes_xy=X_ONLY))
        self.assertAlmostEqual(plan.part_base_mm, 0.0, delta=0.5)
        self.assertAlmostEqual(plan.to_dict()["part_base_mm"], 0.0, delta=0.5)


class OnlyTheFingerTouchesThePart(unittest.TestCase):
    def test_a_part_that_reaches_the_palm_is_not_pushed(self) -> None:
        # A 70 mm part: the fingertip rides at 20 mm and the palm's underside is 20 + 10.45 + 36.53 = 67 mm up.
        # The housing would push it too.
        refusal = _refused(self, _plan(target_points_mm=_block(height=70.0), push_axes_xy=X_ONLY),
                           "part_reaches_the_palm")
        self.assertIn("palm", refusal.sentence)

    def test_the_housing_wider_than_the_open_fingers_keeps_clear_of_a_tall_neighbour(self) -> None:
        # The Hand-E housing is 75 mm along the closing axis, the open fingers 71.6 mm across their outer faces.
        # A 70 mm post 46.7 mm behind the TCP where the finger comes down: clear of the fingers by more than
        # 10 mm, closer than 10 mm to the housing.
        u_start = -15.0 - CONTACT_GAP_MM - HALF_OUTER
        near_face = u_start - HALF_OUTER - 10.0 - 0.9
        tall = _block(centre_xy=(near_face - 5.0, 0.0), size_xy=(10.0, 10.0), height=70.0)
        neighbours = np.vstack([SIDE, tall])
        _planned(self, _plan(neighbour_points_mm=neighbours, push_axes_xy=X_ONLY))
        housing = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                           finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=75.0)
        self.assertEqual(housing.palm_half_along_mm, 37.5)
        refusal = _refused(self, _plan(neighbour_points_mm=neighbours, hand=housing, push_axes_xy=X_ONLY),
                           "no_free_direction")
        self.assertEqual(_verdict(refusal, (1.0, 0.0)), "hand_blocked")

    def test_the_housing_is_read_from_the_gripper_registry_when_it_says(self) -> None:
        class _Registry:
            finger_thickness_mm = 10.80
            finger_width_mm = 29.24
            fingertip_depth_mm = 10.45
            finger_length_mm = 36.53
            palm_width_mm = 75.0
            palm_thickness_mm = 75.0

        self.assertEqual(PushHand.from_gripper_model(_Registry(), open_width_mm=49.99).palm_thickness_mm, 75.0)
        self.assertEqual(PushHand.from_gripper_model(_Registry(), open_width_mm=49.99,
                                                     palm_thickness_mm=80.0).palm_half_along_mm, 40.0)
        # Unset, the housing is taken as wide as the open fingers, never narrower.
        self.assertAlmostEqual(HAND_E.palm_half_along_mm, HALF_OUTER, places=9)
        narrow = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                          finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=40.0)
        self.assertAlmostEqual(narrow.palm_half_along_mm, HALF_OUTER, places=9)
        with self.assertRaises(ValueError):
            PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45,
                     finger_length_mm=36.53, palm_width_mm=75.0, open_width_mm=49.99, palm_thickness_mm=0.0)


class ThePushDistanceIsResolvedAgainstTheConfig(unittest.TestCase):
    """``recovery.fixture.push_distance_mm`` (30) is the push when nobody asks; ``max_nudge_mm`` (50) is the ceiling.

    The owner (2026-09-29): 30 mm by default, adjustable through the API and a ``PickRun`` up to 50 mm, refused
    above, never shortened; under 10 mm no push can open room for a finger.
    """

    @staticmethod
    def _resolve(requested: float | None, default: float = 30.0, ceiling: float = 50.0) -> float | PushRefusal:
        return resolve_push_distance(requested, default_mm=default, ceiling_mm=ceiling)

    def test_nothing_asked_is_the_configs_push_distance(self) -> None:
        self.assertEqual(self._resolve(None), 30.0)
        self.assertEqual(self._resolve(None, default=20.0), 20.0)
        self.assertEqual(self._resolve(None, default=50.0, ceiling=50.0), 50.0)

    def test_a_request_up_to_the_ceiling_is_taken_as_asked(self) -> None:
        for requested in (10.0, 15.0, 30.0, 40.0, 50.0):
            with self.subTest(requested=requested):
                self.assertEqual(self._resolve(requested), requested)
        self.assertEqual(self._resolve(20.0, ceiling=20.0), 20.0)

    def test_a_request_above_the_ceiling_is_refused_never_shortened(self) -> None:
        refusal = self._resolve(40.0, default=20.0, ceiling=30.0)
        assert isinstance(refusal, PushRefusal)
        self.assertEqual(refusal.code, "push_distance_above_config")
        self.assertIn("max_nudge_mm", refusal.sentence)
        self.assertIn("30 mm", refusal.sentence)
        self.assertTrue(refusal.sentence.endswith("."))

    def test_a_request_above_fifty_is_refused_whatever_the_config_says(self) -> None:
        for ceiling in (50.0, 80.0):
            with self.subTest(ceiling=ceiling):
                refusal = self._resolve(50.5, ceiling=ceiling)
                assert isinstance(refusal, PushRefusal)
                self.assertEqual(refusal.code, "push_distance_above_cap")
        self.assertEqual(self._resolve(None, ceiling=80.0), 30.0)

    def test_a_config_push_above_fifty_is_refused_whatever_the_ceiling_says(self) -> None:
        # The hard cap holds on the config's own push too, not only on a request: a ceiling written above 50 mm
        # (the schema refuses one at load; a caller that did not load one) does not let a 60 mm default through.
        refusal = self._resolve(None, default=60.0, ceiling=80.0)
        assert isinstance(refusal, PushRefusal)
        self.assertEqual(refusal.code, "push_distance_above_config")
        self.assertIn("50 mm", refusal.sentence)

    def test_a_config_whose_push_lies_above_its_own_ceiling_is_refused(self) -> None:
        refusal = self._resolve(None, default=30.0, ceiling=20.0)
        assert isinstance(refusal, PushRefusal)
        self.assertEqual(refusal.code, "push_distance_above_config")
        self.assertIn("push_distance_mm", refusal.sentence)

    def test_a_push_too_short_to_open_room_for_a_finger_is_refused(self) -> None:
        # A push can open at most its own length, and the planner asks for 10 mm. The old 5 mm default would
        # never have pushed; now that is said, not silent.
        for requested, default in ((8.0, 30.0), (9.99, 30.0), (None, 5.0)):
            with self.subTest(requested=requested, default=default):
                refusal = self._resolve(requested, default=default)
                assert isinstance(refusal, PushRefusal)
                self.assertEqual(refusal.code, "push_distance_too_short")
                self.assertTrue(refusal.sentence.endswith("."))
        _refused(self, _plan(push_distance_mm=8.0), "push_distance_too_short")

    def test_what_is_not_a_positive_number_is_refused(self) -> None:
        cases = ((0.0, 30.0, 50.0), (-5.0, 30.0, 50.0), (math.nan, 30.0, 50.0), (math.inf, 30.0, 50.0),
                 (None, math.nan, 50.0), (None, -1.0, 50.0), (None, 30.0, math.nan), (None, 30.0, 0.0),
                 (20.0, 30.0, -1.0))
        for requested, default, ceiling in cases:
            with self.subTest(requested=requested, default=default, ceiling=ceiling):
                refusal = self._resolve(requested, default=default, ceiling=ceiling)
                assert isinstance(refusal, PushRefusal)
                self.assertEqual(refusal.code, "push_distance_invalid")


class TheDistancesStayInBoundedMemory(unittest.TestCase):
    def test_a_large_part_against_many_neighbour_cells_is_measured_in_small_blocks(self) -> None:
        # A 200 x 200 mm part is about 10,000 planner cells. The distances are taken in blocks of a bounded
        # number of pairs, so the pick attempt does not reach for a gigabyte.
        rng = np.random.default_rng(7)
        a = rng.uniform(0.0, 200.0, size=(6000, 2))
        b = rng.uniform(260.0, 460.0, size=(6000, 2))
        expected = min(float(np.sqrt(((a[i:i + 200, None, :] - b[None, :, :]) ** 2).sum(axis=2)).min())
                       for i in range(0, len(a), 200))
        tracemalloc.start()
        try:
            found = push_planner._min_distance(a, b)
            swept = push_planner._min_swept_distance(a, b, 30.0)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertAlmostEqual(found, expected, places=9)
        self.assertLessEqual(swept, found)
        self.assertLess(peak, 64 * 2 ** 20)

if __name__ == "__main__":
    unittest.main()
