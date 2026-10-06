"""A push may rearrange the scene, and critical parts are cleared instead (the owner, 2026-10-03).

On 160 renders of the owner's mat (the URSim stand-in, 2026-10-03) the push planned in none: the open Hand-E is 72 mm
long along the push, the guard keeps 20 mm from a neighbour, the camera's shadows counted as unseen, and the landing
window grew with the push. The owner: "Allgemein Schieben erlauben, sofern nicht ein Abgrund oder so da ist", and then
"Er darf die Szene ruhig dolle verändern ... Erst wenn es um Teile geht, welche kritisch sind, dort soll dann Wegräumen
kommen ... das muss dann als Schalter eingestellt werden können". Pinned here, on the hand-built scenes of
``test_a_push_is_planned_from_what_the_camera_saw.py`` (a 30 mm block at the origin on a table at z = 0):

* **The foot.** A part no look saw the lower sides of (from above alone, or past a neighbour that hides its foot) is
  taken to stand on the support where the support was seen round it, or hidden there by what stands on it, and nothing
  says otherwise: not the edge of what was seen within 15 mm of it (where the support may drop away), not the table
  seen under it, not neighbours alone round it (parts side by side can pass for the ground).
* **The ground.** A patch the camera saw nothing in among seen table, beside something standing (its shadow) or no
  wider than 50 mm (a dip of the mat, a dropout), is ground for a landing; a larger patch beside nothing may be a hole.
  The fingers still come down only where the camera saw table.
* **The edge.** A landing keeps 30 + 15 mm of seen table round it, however far the push goes.
* **Longer pushes.** Where nobody asked for a distance and the cell's opens too little room, 10 mm longer pushes are
  tried up to the cell's longest, the shortest that works first; a distance asked for is taken as asked.
* **The rearranging push** (``may_shove``): the part may brush and shove what stands in its way and the fingers what
  stands beside their stroke; the fingers come down 10 mm from every neighbour, on seen table; the housing keeps the
  hand's clearance from every neighbour tall enough to reach it, and such a neighbour stays in the guard's world, the
  fingers keeping the hand's clearance from it; a push that would end with a neighbour against the part opens no room
  and is refused. Every lower neighbour it may touch is held out of the camera world whole, a box of its own beside the
  part's: a box round the stroke alone left the rest of a 60 mm block for the world to box again, and the guard refused
  the push leg at 4.5 mm (URSim, 2026-10-03).
* **Judged ahead**: before P0, the move there and every contact leg are judged from where the arm stands, each from
  where the one before ends (``JudgesLinesAhead``); a refusal there moves nothing.
* **The switch** (``robot.grasping.recovery.critical_parts``, and a console run's own word): not critical, a boxed-in
  part is pushed first and a blocker is cleared where no push plans; critical, nothing is pushed and a blocker is
  cleared instead.
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Any

import numpy as np

from src.robot.grasping.recovery.push_planner import (
    AUTO_BOX_EXTRA_SHRINK_MM,
    FOOT_RING_MM,
    HIDDEN_PATCH_MAX_MM,
    PushPlan,
)
from tests.test_a_push_is_planned_from_what_the_camera_saw import (
    SIDE,
    TABLE,
    X_ONLY,
    _block,
    _plan,
    _planned,
    _refused,
    _verdict,
)

#: The table less what lies under the block at the origin, which no camera sees.
UNDER_THE_PART = (np.abs(TABLE[:, 0]) <= 15.5) & (np.abs(TABLE[:, 1]) <= 15.5)
#: The pick's numbers beside the part: the owner's 10 mm, inside the world's 15 mm margin round what the push keeps out.
BESIDE = {"beside_part_clearance_mm": 10.0, "beside_part_mm": 15.0}


def _top_only() -> np.ndarray:
    block = _block()
    return block[block[:, 2] >= 29.9]


def _without(table: np.ndarray, x: tuple[float, float], y: tuple[float, float]) -> np.ndarray:
    gone = (table[:, 0] > x[0]) & (table[:, 0] < x[1]) & (table[:, 1] > y[0]) & (table[:, 1] < y[1])
    return table[~gone]


class TheFootIsInferredUnlessADropOffTests(unittest.TestCase):

    def test_a_part_seen_from_above_alone_stands_where_the_table_round_it_says(self) -> None:
        plan = _planned(self, _plan(target_points_mm=_top_only(), table_points_mm=TABLE[~UNDER_THE_PART],
                                    push_axes_xy=X_ONLY))
        self.assertTrue(plan.foot_inferred)
        self.assertEqual(0.0, plan.part_base_mm)
        self.assertAlmostEqual(30.0, plan.part_height_mm, delta=0.5)
        self.assertAlmostEqual(15.0, plan.finger_height_mm, delta=0.5)
        self.assertEqual("inferred", plan.to_dict()["foot"])
        self.assertEqual("seen", _planned(self, _plan(push_axes_xy=X_ONLY)).to_dict()["foot"])

    def test_the_edge_of_what_was_seen_beside_it_is_a_possible_drop_off(self) -> None:
        # The table ends 5 mm past its +x face: the support may drop away right there.
        table = TABLE[~UNDER_THE_PART & (TABLE[:, 0] <= 20.0)]
        refusal = _refused(self, _plan(target_points_mm=_top_only(), table_points_mm=table, push_axes_xy=X_ONLY),
                           "part_not_on_support")
        self.assertIn("edge of what the camera saw", refusal.sentence)
        self.assertIn(f"{FOOT_RING_MM:g} mm", refusal.sentence)

    def test_neighbours_alone_round_it_say_nothing_of_what_it_stands_on(self) -> None:
        # Four 16 mm walls 2 mm off its faces cover the whole ring; no table is seen within 15 mm of it.
        walls = np.vstack([_block(centre_xy=(25.0, 0.0), size_xy=(16.0, 66.0)),
                           _block(centre_xy=(-25.0, 0.0), size_xy=(16.0, 66.0)),
                           _block(centre_xy=(0.0, 25.0), size_xy=(34.0, 16.0)),
                           _block(centre_xy=(0.0, -25.0), size_xy=(34.0, 16.0))])
        table = TABLE[~((np.abs(TABLE[:, 0]) <= 33.5) & (np.abs(TABLE[:, 1]) <= 33.5))]
        refusal = _refused(self, _plan(target_points_mm=_top_only(), neighbour_points_mm=walls, table_points_mm=table,
                                       push_axes_xy=X_ONLY), "part_not_on_support")
        self.assertIn("neighbours alone", refusal.sentence)

    def test_a_neighbour_that_hides_its_foot_does_not_keep_it_from_being_pushed(self) -> None:
        # The owner's L-scene from the camera's side: a wall 12 mm off its +y face hides the gap and its foot; the
        # table is seen on its other sides. Pushed 50 mm along x, it slides past the wall's end.
        wall = _block(centre_xy=(-10.0, 32.0), size_xy=(40.0, 10.0), height=40.0)
        table = _without(_without(TABLE[~UNDER_THE_PART], (-15.5, 15.5), (15.0, 27.5)), (-30.5, 10.5), (26.5, 37.5))
        plan = _planned(self, _plan(target_points_mm=_top_only(), neighbour_points_mm=wall, table_points_mm=table,
                                    push_axes_xy=X_ONLY, push_distance_mm=50.0))
        self.assertTrue(plan.foot_inferred)

    def test_a_part_seen_down_to_its_foot_is_not_inferred(self) -> None:
        self.assertFalse(_planned(self, _plan(push_axes_xy=X_ONLY)).foot_inferred)


class TheGroundIsWhatIsNoDropOffTests(unittest.TestCase):

    def test_a_dip_no_wider_than_fifty_millimetres_is_ground_and_a_larger_hole_is_not(self) -> None:
        # Where a 50 mm push lands (x 35 to 65): a patch the camera read no table in, beside nothing standing.
        dip = _without(TABLE, (45.0, 45.0 + HIDDEN_PATCH_MAX_MM - 10.0), (-20.0, 20.0))
        _planned(self, _plan(table_points_mm=dip, push_axes_xy=X_ONLY, push_distance_mm=50.0))
        hole = _without(TABLE, (45.0, 45.0 + HIDDEN_PATCH_MAX_MM + 20.0), (-35.0, 35.0))
        refusal = _refused(self, _plan(table_points_mm=hole, push_axes_xy=X_ONLY, push_distance_mm=50.0),
                           "no_free_direction")
        self.assertEqual("landing_over_unseen_table", _verdict(refusal, (1.0, 0.0)))

    def test_the_landing_keeps_its_margin_from_the_edge_however_far_the_push_goes(self) -> None:
        # The table ends at x = 115. A 50 mm push lands at x 35 to 65, 50 mm from the edge: the box (the table shrunk by
        # 30 mm) holds it with its 15 mm margin. Shrunk by the push as well, as before, it would not.
        table = TABLE[TABLE[:, 0] <= 115.0]
        plan = _planned(self, _plan(table_points_mm=table, push_axes_xy=X_ONLY, push_distance_mm=50.0))
        self.assertAlmostEqual(115.0 - AUTO_BOX_EXTRA_SHRINK_MM, plan.landing_box_xy_mm[1][0], delta=6.0)
        near = TABLE[TABLE[:, 0] <= 100.0]
        refusal = _refused(self, _plan(table_points_mm=near, push_axes_xy=X_ONLY, push_distance_mm=50.0),
                           "no_free_direction")
        self.assertIn(_verdict(refusal, (1.0, 0.0)), ("landing_outside_box", "landing_over_unseen_table"))


#: A wall 15 mm off the block's +y face, from x = -40 to 0: a 30 mm push along +x leaves the block beside its end.
WALL = _block(centre_xy=(-20.0, 35.0), size_xy=(40.0, 10.0))


class ALongerPushIsTriedWhereNobodyAskedTests(unittest.TestCase):

    def test_the_shortest_longer_push_that_opens_the_room_is_taken(self) -> None:
        refusal = _refused(self, _plan(neighbour_points_mm=WALL, push_axes_xy=X_ONLY), "no_free_direction")
        self.assertEqual("no_clearance_gain", _verdict(refusal, (1.0, 0.0)))
        plan = _planned(self, _plan(neighbour_points_mm=WALL, push_axes_xy=X_ONLY, longest_push_mm=50.0))
        self.assertEqual(40.0, plan.push_distance_mm)
        self.assertGreaterEqual(plan.clearance_gain_mm, 10.0)

    def test_a_refusal_says_every_distance_it_tried(self) -> None:
        long_wall = _block(centre_xy=(-20.0, 35.0), size_xy=(120.0, 10.0))
        refusal = _refused(self, _plan(neighbour_points_mm=long_wall, push_axes_xy=X_ONLY, longest_push_mm=50.0),
                           "no_free_direction")
        self.assertIn("at 30, 40 or 50 mm", refusal.sentence)

    def test_never_past_the_hard_cap(self) -> None:
        plan = _planned(self, _plan(neighbour_points_mm=WALL, push_axes_xy=X_ONLY, longest_push_mm=80.0))
        self.assertLessEqual(plan.push_distance_mm, 50.0)


#: A low cube 3 mm off the block's -y face, just past its +x face: the block brushes past it along +x.
BRUSHED = _block(centre_xy=(18.0, -23.0), size_xy=(4.0, 10.0), height=10.0)


class ARearrangingPushMayTouchWhatStandsInItsWayTests(unittest.TestCase):

    def _scene(self, neighbours: np.ndarray, **changes: Any) -> dict[str, Any]:
        values: dict[str, Any] = dict(neighbour_points_mm=neighbours, push_axes_xy=X_ONLY, push_distance_mm=50.0,
                                      **BESIDE)
        values.update(changes)
        return values

    def test_the_part_brushes_past_a_neighbour_the_careful_push_keeps_clear_of(self) -> None:
        scene = self._scene(np.vstack([SIDE, BRUSHED]))
        careful = _refused(self, _plan(**scene), "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(careful, (1.0, 0.0)))
        plan = _planned(self, _plan(may_shove=True, **scene))
        np.testing.assert_allclose(plan.direction[:2], (1.0, 0.0), atol=1e-9)
        self.assertTrue(plan.shoves)
        self.assertTrue(plan.to_dict()["shoves"])
        (held,) = plan.kept_out_neighbours_mm
        above_the_band = BRUSHED[BRUSHED[:, 2] > 5.0]
        self.assertEqual(len(above_the_band), len(held), "the brushed cube is not held out of the world whole")

    def test_a_push_that_ends_with_a_neighbour_against_the_part_opens_no_room(self) -> None:
        # A cube 6 mm off its +x face: pushed along +x the part shoves it ahead and ends against it.
        ahead = _block(centre_xy=(26.0, 0.0), size_xy=(10.0, 20.0), height=20.0)
        refusal = _refused(self, _plan(may_shove=True, **self._scene(ahead)), "no_free_direction")
        self.assertEqual("no_clearance_gain", _verdict(refusal, (1.0, 0.0)))

    def test_the_fingers_never_come_down_beside_a_neighbour(self) -> None:
        # Where the fingers come down for +x, 25 to 97 mm behind the part's -x face.
        under_the_fingers = _block(centre_xy=(-40.0, 0.0), size_xy=(10.0, 10.0), height=10.0)
        refusal = _refused(self, _plan(may_shove=True, **self._scene(np.vstack([SIDE, under_the_fingers]))),
                           "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_the_housing_keeps_the_hands_clearance_from_a_tall_neighbour(self) -> None:
        # 70 mm tall, 25 mm beside the push line: clear of the fingers, within reach of the 75 mm housing.
        tall = _block(centre_xy=(-55.0, 30.0), size_xy=(10.0, 10.0), height=70.0)
        refusal = _refused(self, _plan(may_shove=True, **self._scene(np.vstack([SIDE, BRUSHED, tall]))),
                           "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_a_neighbour_tall_enough_to_reach_the_housing_stays_in_the_guards_world(self) -> None:
        # The brushed cube 60 mm tall: it could reach the housing, so it is not held out, and the hand keeps its clearance
        # from it as the guard will.
        tall = _block(centre_xy=(18.0, -23.0), size_xy=(4.0, 10.0), height=60.0)
        refusal = _refused(self, _plan(may_shove=True, **self._scene(np.vstack([SIDE, tall]))), "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_the_fingers_come_down_on_seen_table_and_may_cross_a_shadow(self) -> None:
        # A dip where the fingers come down: ground for a landing, never for a finger.
        dip = _without(TABLE, (-60.0, -40.0), (-10.0, 10.0))
        refusal = _refused(self, _plan(may_shove=True, table_points_mm=dip, **self._scene(SIDE)), "no_free_direction")
        self.assertEqual("hand_over_unseen_table", _verdict(refusal, (1.0, 0.0)))
        # The part's own shadow between where the fingers come down and the part: crossed, not landed in.
        shadow = _without(TABLE, (-24.0, -15.5), (-10.0, 10.0))
        careful = _refused(self, _plan(table_points_mm=shadow, **self._scene(SIDE)), "no_free_direction")
        self.assertEqual("hand_over_unseen_table", _verdict(careful, (1.0, 0.0)))
        _planned(self, _plan(may_shove=True, table_points_mm=shadow, **self._scene(SIDE)))

    def test_a_push_that_touches_nothing_holds_out_nothing_but_the_part(self) -> None:
        plan = _planned(self, _plan(may_shove=True, **self._scene(SIDE)))
        self.assertFalse(plan.shoves)
        self.assertEqual((), plan.kept_out_neighbours_mm)
        self.assertEqual((), _planned(self, _plan(**self._scene(SIDE))).kept_out_neighbours_mm)


#: A low cube 5 mm beside where the fingers come down for +x: inside the owner's 10 mm, outside a grasp's 1 mm.
BESIDE_THE_FINGERS = _block(centre_xy=(-70.0, 0.5 * 29.24 + 5.0 + 5.0), size_xy=(10.0, 10.0), height=10.0)


class BesideANamedPartTheFingersComeDownAsAGraspsDoTests(unittest.TestCase):
    """The owner, 2026-10-06 ("Wie beim Greifen: 1 mm"): on a rearranging push the fingers come down 1 mm from a part
    the detector named, as a grasp's finger comes to one, and the owner's 10 mm from everything nobody named."""

    def _scene(self, *, named: bool, clearance: "float | None" = 1.0) -> dict[str, Any]:
        flags = np.concatenate([np.zeros(len(SIDE), dtype=bool), np.full(len(BESIDE_THE_FINGERS), named)])
        return dict(neighbour_points_mm=np.vstack([SIDE, BESIDE_THE_FINGERS]), push_axes_xy=X_ONLY,
                    push_distance_mm=50.0, may_shove=True, neighbour_named=flags, named_part_clearance_mm=clearance,
                    **BESIDE)

    def test_a_named_part_5_mm_beside_the_fingers_lets_them_come_down(self) -> None:
        plan = _planned(self, _plan(**self._scene(named=True)))
        np.testing.assert_allclose(plan.direction[:2], (1.0, 0.0), atol=1e-9)

    def test_what_nobody_named_keeps_the_owners_10_mm(self) -> None:
        refusal = _refused(self, _plan(**self._scene(named=False)), "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_with_no_named_clearance_every_neighbour_keeps_10_mm(self) -> None:
        refusal = _refused(self, _plan(**self._scene(named=True, clearance=None)), "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_a_named_part_under_the_fingers_still_blocks_them(self) -> None:
        under = _block(centre_xy=(-70.0, 0.0), size_xy=(10.0, 10.0), height=10.0)
        flags = np.concatenate([np.zeros(len(SIDE), dtype=bool), np.ones(len(under), dtype=bool)])
        refusal = _refused(self, _plan(neighbour_points_mm=np.vstack([SIDE, under]), push_axes_xy=X_ONLY,
                                       push_distance_mm=50.0, may_shove=True, neighbour_named=flags,
                                       named_part_clearance_mm=1.0, **BESIDE), "no_free_direction")
        self.assertEqual("hand_blocked", _verdict(refusal, (1.0, 0.0)))

    def test_flags_that_do_not_match_the_points_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            _plan(neighbour_named=np.ones(3, dtype=bool), named_part_clearance_mm=1.0)

    def test_the_cell_takes_a_grasps_slack_where_the_fingers_touch_parts(self) -> None:
        from src.robot.grasping.generation.scene_obstacles import SOFT_SLACK_MM
        from src.robot.grasping.recovery.push_gate import PushCell
        from tests.test_a_boxed_in_part_gets_no_grasp_and_says_so import hande_cell

        cell = PushCell.from_robot_config(hande_cell())
        assert isinstance(cell, PushCell), cell
        self.assertEqual(SOFT_SLACK_MM, cell.named_part_clearance_mm)
        on = hande_cell()
        perceived = on.safety.planning_world.perceived.model_copy(update={"fingers_touch_parts": False})
        world = on.safety.planning_world.model_copy(update={"perceived": perceived})
        off = on.model_copy(update={"safety": on.safety.model_copy(update={"planning_world": world})})
        cell = PushCell.from_robot_config(off)
        assert isinstance(cell, PushCell), cell
        self.assertIsNone(cell.named_part_clearance_mm)


def _brushing_plan() -> PushPlan:
    plan = _plan(may_shove=True, neighbour_points_mm=np.vstack([SIDE, BRUSHED]), push_axes_xy=X_ONLY,
                 push_distance_mm=50.0, **BESIDE)
    assert isinstance(plan, PushPlan)
    return plan


class TheRearrangingPushHoldsWhatItTouchesOutOfTheWorldTests(unittest.TestCase):

    def test_every_motion_runs_with_each_touched_neighbour_held_out_as_a_box_of_its_own(self) -> None:
        from tests.test_a_push_moves_only_when_every_guard_holds import OFFER, _Cell

        plan = _brushing_plan()
        cell = _Cell(self)
        outcome = cell.push(plan, offer=OFFER)
        self.assertEqual("pushed", outcome.code.value, outcome.reason)
        offers = cell.log.of("offer")
        self.assertEqual(["part", "push_neighbour_0"], [offer["target_label"] for offer in offers])
        (neighbour,) = plan.kept_out_neighbours_mm
        np.testing.assert_allclose(np.asarray(offers[1]["target_points_base_mm"]), neighbour)
        kinds = cell.log.kinds()
        moves = [index for index, kind in enumerate(kinds) if kind == "move"]
        self.assertLess(max(index for index, kind in enumerate(kinds) if kind == "offer"), min(moves))


class _Ahead:
    """A mixin for the push tests' arm: it judges a push's lines ahead and answers ``refusal`` for them."""

    refusal = ""
    asked: list[tuple[Any, tuple[Any, ...]]]

    def lines_refusal_ahead(self, *, approach: Any, lines: Any) -> str:
        self.asked = [*getattr(self, "asked", []), (approach, tuple(lines))]
        return self.refusal


class EveryLineOfAPushIsJudgedBeforeP0Tests(unittest.TestCase):

    def _cell(self, refusal: str) -> Any:
        from tests.test_a_push_moves_only_when_every_guard_holds import _UR, _Cell

        class _JudgingUR(_Ahead, _UR):
            pass

        _JudgingUR.refusal = refusal
        return _Cell(self, _JudgingUR)

    def test_a_refusal_ahead_moves_nothing(self) -> None:
        from src.robot.grasping.recovery.push_motion import PushOutcomeCode
        from tests.test_a_push_moves_only_when_every_guard_holds import OFFER, PLAN

        cell = self._cell("the line to push_push would be refused (self_collision_rejected: lfinger 4.5 mm)")
        outcome = cell.push(PLAN, offer=OFFER)
        # Judged inside the push's keep-out, as the motions would run: offered and forgotten, and nothing moved.
        self.assertEqual([], cell.motions(), "a refusal ahead moved the arm")
        self.assertEqual([], cell.log.of("DO"), "a refusal ahead wrote the tool output")
        self.assertEqual(["offer", "forget"], [kind for kind in cell.log.kinds() if kind in ("offer", "forget")])
        self.assertIs(PushOutcomeCode.REFUSED_AHEAD, outcome.code)
        self.assertIn("push_push", outcome.reason)
        self.assertFalse(outcome.motion_started)

    def test_p0_and_the_four_legs_are_what_it_judges(self) -> None:
        from tests.test_a_push_moves_only_when_every_guard_holds import OFFER, PLAN

        cell = self._cell("")
        outcome = cell.push(PLAN, offer=OFFER)
        self.assertEqual("pushed", outcome.code.value, outcome.reason)
        ((approach, lines),) = cell.arm.asked
        np.testing.assert_allclose(approach.position_mm, PLAN.approach_tcp_mm)
        self.assertEqual(["push_down", "push_push", "push_back", "push_up"], [pose.label for pose in lines])
        for pose, leg in zip(lines, PLAN.legs):
            np.testing.assert_allclose(pose.position_mm, leg.end_tcp_mm)


class TheArmJudgesEveryLineFromWhereTheOneBeforeEndsTests(unittest.TestCase):
    """``URRobotArm.lines_refusal_ahead`` on a stand-in of the arm's own judgements."""

    def _arm(self, *, route: Any = None, refuse_at: int | None = None) -> Any:
        from types import SimpleNamespace

        from src.robot.core import MotionCommand, MotionResult, MotionStatus
        from src.robot.drivers.ur.arm import _Route

        asked: list[tuple[str, Any]] = []
        lines = {"count": 0}

        def route_to(planner_: Any, goal: Any, pose: Any, *, velocity: float) -> Any:
            asked.append(("route", pose.label))
            return route if route is not None else _Route(waypoints=[[0.0] * 6, [0.1] * 6], dense=2, how="direct line",
                                                           planned=False)

        def judge_line(pose: Any, *, command: Any, commanded: bool = True, start: Any = None) -> Any:
            lines["count"] += 1
            asked.append(("line", (pose.label, commanded, start[0].label, list(start[1]))))
            if refuse_at is not None and lines["count"] == refuse_at:
                return MotionResult.failed(MotionStatus.SELF_COLLISION_REJECTED, MotionCommand.MOVE_TO,
                                           message="lfinger 4.5 mm")
            return None

        def ik(pose: Any, *, seed: Any) -> Any:
            asked.append(("ik", pose.label))
            value = [round(float(seed.tolist()[0]) + 0.1, 3)] * 6
            return SimpleNamespace(tolist=lambda: value)

        self.asked = asked
        return SimpleNamespace(
            _motion_planner="curobo", _preflight=object(), _conn=SimpleNamespace(is_connected=True),
            config=SimpleNamespace(motion_limits=SimpleNamespace(max_velocity=0.5)),
            _curobo_ur_planner=lambda: "planner", _pose_to_flange=lambda pose: pose,
            _route_to_the_nearest_goal=route_to, _judge_linear_move=judge_line, ik=ik,
        )

    def _ask(self, arm: Any) -> str:
        from src.geometry import Pose
        from src.robot.drivers.ur.arm import URRobotArm

        def pose(z: float, label: str) -> Pose:
            return Pose.tool_down(-150.0, -700.0, z, label=label)

        return URRobotArm.lines_refusal_ahead(arm, approach=pose(180.0, "push_p0"),
                                              lines=[pose(100.0, "push_down"), pose(100.5, "push_push")])

    def test_each_line_from_where_the_one_before_ends(self) -> None:
        self.assertEqual("", self._ask(self._arm()))
        self.assertEqual(["route", "line", "ik", "line", "ik"], [kind for kind, _ in self.asked])
        down, push = self.asked[1][1], self.asked[3][1]
        self.assertEqual(("push_down", False, "push_p0", [0.1] * 6), down)
        self.assertEqual(("push_push", False, "push_down", [0.2] * 6), push)

    def test_the_first_refusal_is_said_and_ends_it(self) -> None:
        said = self._ask(self._arm(refuse_at=2))
        self.assertIn("the line to push_push would be refused", said)
        self.assertIn("lfinger 4.5 mm", said)
        self.assertEqual(["route", "line", "ik", "line"], [kind for kind, _ in self.asked])


class TheSwitchSaysWhetherToPushOrToClearTests(unittest.TestCase):
    """The pick's order: a part that is not critical is pushed first; a critical one is never pushed."""

    def _cell(self, *, critical: bool) -> tuple[Any, list[str]]:
        from tests.test_a_blocker_is_set_aside_before_the_part import _Cell

        cell = _Cell()
        assert cell.orchestrator.push_gate is not None
        cell.orchestrator.push_gate = replace(cell.orchestrator.push_gate, critical_parts=critical)
        order: list[str] = []
        push, clear = cell.orchestrator._push_the_failed_part, cell.orchestrator._clear_the_blockers  # noqa: SLF001

        def pushing(*args: Any, **keywords: Any) -> Any:
            order.append("push")
            return push(*args, **keywords)

        def clearing(*args: Any, **keywords: Any) -> Any:
            order.append("clear")
            return clear(*args, **keywords)

        cell.orchestrator._push_the_failed_part = pushing  # type: ignore[method-assign]
        cell.orchestrator._clear_the_blockers = clearing  # type: ignore[method-assign]
        return cell, order

    def test_parts_that_are_not_critical_are_pushed_first(self) -> None:
        cell, order = self._cell(critical=False)
        cell.run()
        self.assertEqual("push", order[0], order)

    def test_critical_parts_are_never_pushed_and_their_blocker_is_cleared(self) -> None:
        cell, order = self._cell(critical=True)
        report = cell.run()
        self.assertNotIn("push", order)
        self.assertEqual((), cell.orchestrator.pushes, "a push was considered for a critical part")
        self.assertEqual("set_aside", cell.orchestrator.blockers[0].code, cell.orchestrator.blockers[0].sentence)
        self.assertEqual("executed", report.attempts[-1].action)


class TheCampaignCarriesTheSwitchAndTheLongestPushTests(unittest.TestCase):

    def _service(self, **recovery: Any) -> Any:
        from tests.test_a_push_needs_no_fixture import _HAND_E, _service

        service = _service(gripper=_HAND_E)
        if recovery:
            service.effective_config = replace(service.effective_config, recovery_orchestrator=replace(
                service.effective_config.recovery_orchestrator, **recovery))
        return service

    def _gate(self, service: Any, **campaign: Any) -> Any:
        from src.robot.execution.autonomous_grasp import GraspMode, _profile_for

        return service._push_gate(_profile_for(GraspMode.DENSE_CLUTTER), service._build_recovery_orchestrator_policy(),
                                  service.start_campaign(**campaign))

    def test_a_distance_nobody_asked_for_may_go_as_far_as_the_cell_allows(self) -> None:
        service = self._service()
        gate = self._gate(service)
        self.assertEqual((30.0, 50.0), (gate.distance_mm, gate.longest_mm))
        asked = self._gate(service, push_mm=40.0)
        self.assertEqual((40.0, None), (asked.distance_mm, asked.longest_mm))

    def test_the_cells_switch_stands_unless_the_run_says_otherwise(self) -> None:
        self.assertFalse(self._gate(self._service()).critical_parts)
        critical = self._service(critical_parts=True)
        self.assertTrue(self._gate(critical).critical_parts)
        self.assertFalse(self._gate(critical, critical_parts=False).critical_parts)
        self.assertTrue(self._gate(critical, critical_parts=None).critical_parts)
        self.assertTrue(self._gate(self._service(), critical_parts=True).critical_parts)

    def test_the_config_key_reaches_the_effective_config(self) -> None:
        from src.config.schema.robot.grasping_schema import GraspingRecoveryConfig, RobotGraspingConfig
        from src.robot.execution.autonomous_grasp.builders import build_effective_config
        from src.robot.execution.autonomous_grasp.config import GraspMode

        self.assertFalse(GraspingRecoveryConfig().critical_parts)
        cfg = RobotGraspingConfig(recovery=GraspingRecoveryConfig(enabled=True, critical_parts=True))
        effective = build_effective_config(cfg, resolved_mode=GraspMode.DENSE_CLUTTER, resolved_max_attempts=3)
        assert effective is not None
        self.assertTrue(effective.recovery_orchestrator.critical_parts)


if __name__ == "__main__":
    unittest.main()
