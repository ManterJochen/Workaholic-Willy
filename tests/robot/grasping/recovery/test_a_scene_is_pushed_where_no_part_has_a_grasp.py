"""Where no part a task taking every part wants has a grasp, the scene is pushed (the owner, 2026-10-15: "ist das nicht
genau der Punkt wo man extreme Veraenderung braucht?").

In a tight pile even the closed fingers find no room beside a part, so no part can be pushed on its own. Any part the
detector named may be pushed instead, as a rearranging push with every rule it keeps, judged by the room it opens for a
grasp of a part the task wants: the most room a wanted part has on two opposite faces along the best of four closing
axes, each face counted up to a finger's thickness and 3 mm. What is pinned:

* the planner (``plan_scene_push``): in a 3 x 3 grid of 30 mm cubes 3 mm apart no cube has room on two opposite faces;
  pushing a middle cube of an edge into the grid frees the two corners beside it on the faces that faced it; a part
  nobody wants is pushed too where that opens room for one that is wanted; where no push opens 10 mm more, the one that
  opens the most room at all, and where none opens any, the one that moves the most parts, another with it (the owner,
  2026-10-16: "einfach schieben"); a part alone, and a scene no part of which is wanted, is never pushed;
* the budgets: 3 scene pushes per pick, 10 per campaign, apart from the 1/2/5 of a part's own push;
* the routing: a task taking every part pushes the scene and sets no neighbour aside (that is a part it takes next); a
  task of one part, and critical parts, clear a blocker instead; a console task that says "until empty" says so to its
  campaign.
"""

from __future__ import annotations

import unittest
from typing import Any
from unittest import mock

import numpy as np

from src.robot.grasping.recovery.push_budgets import (
    BUDGET_CAMPAIGN_SPENT,
    BUDGET_PICK_SPENT,
    SCENE_PUSHES_PER_CAMPAIGN,
    SCENE_PUSHES_PER_PICK,
)
from src.robot.grasping.recovery.push_gate import PushCampaign
from src.robot.grasping.recovery.push_planner import (
    ROOM_BESIDE_A_FINGER_MM,
    PushHand,
    PushRefusal,
    plan_scene_push,
)
from tests._console_task_fakes import ConsoleCase, Scripted
from tests.robot.grasping.recovery.test_a_push_is_planned_from_what_the_camera_saw import TABLE, WORKSPACE, _block

#: The cell's Robotiq Hand-E pushing closed: its fingers 0 mm apart and the guard's 0.5 mm.
CLOSED = PushHand(finger_thickness_mm=10.80, finger_width_mm=29.24, fingertip_depth_mm=10.45, finger_length_mm=36.53,
                  palm_width_mm=75.0, open_width_mm=0.5, palm_thickness_mm=75.0)
#: The room one face of a wanted part may count: a finger's thickness and 3 mm.
FACE = CLOSED.finger_thickness_mm + ROOM_BESIDE_A_FINGER_MM
#: The grid's pitch: 30 mm cubes 3 mm apart.
PITCH = 33.0
GRID = [(column * PITCH, row * PITCH) for row in (-1, 0, 1) for column in (-1, 0, 1)]
#: Where the middle cubes of the grid's four edges stand.
EDGE_MIDDLES = {(-PITCH, 0.0), (PITCH, 0.0), (0.0, -PITCH), (0.0, PITCH)}


def _grid() -> list[np.ndarray]:
    return [_block(centre_xy=centre) for centre in GRID]


def _scene(parts: list[np.ndarray], wanted: list[bool], unnamed: np.ndarray | None = None) -> Any:
    return plan_scene_push(parts_mm=parts, wanted=wanted,
                           unnamed_points_mm=np.zeros((0, 3)) if unnamed is None else unnamed,
                           push_distance_mm=50.0, support_normal=(0.0, 0.0, 1.0), support_offset_mm=0.0,
                           workspace=WORKSPACE, hand=CLOSED, table_points_mm=TABLE, hand_clearance_mm=25.0,
                           beside_part_clearance_mm=10.0, beside_part_mm=15.0, named_part_clearance_mm=1.0)


class TheScenePushOpensRoomForAGraspTests(unittest.TestCase):

    def test_a_middle_cube_of_an_edge_is_pushed_into_the_grid(self) -> None:
        answer = _scene(_grid(), [True] * 9)
        assert not isinstance(answer, PushRefusal), answer.sentence
        which, plan = answer
        self.assertIn(GRID[which], EDGE_MIDDLES)
        # Into the grid: from its edge toward the middle cube.
        inward = -np.asarray(GRID[which]) / np.linalg.norm(GRID[which])
        np.testing.assert_allclose(plan.direction[:2], inward, atol=1e-9)
        # Before: a corner has a free outer face and 3 mm on the inner one. After: the two corners beside the pushed
        # cube have room on both faces that faced it.
        self.assertAlmostEqual(FACE + 3.0, plan.clearance_before_mm, delta=2.5)
        self.assertAlmostEqual(2.0 * FACE, plan.clearance_after_mm, delta=0.5)
        self.assertTrue(plan.shoves)
        self.assertGreaterEqual(len(plan.kept_out_neighbours_mm), 1)

    def test_a_part_nobody_wants_is_pushed_where_that_opens_room_for_one_that_is_wanted(self) -> None:
        # Only the corner at -x, -y is wanted: one of the two cubes beside it, neither wanted, is pushed into the grid.
        wanted = [centre == (-PITCH, -PITCH) for centre in GRID]
        answer = _scene(_grid(), wanted)
        assert not isinstance(answer, PushRefusal), answer.sentence
        which, plan = answer
        self.assertIn(GRID[which], {(-PITCH, 0.0), (0.0, -PITCH)})
        self.assertAlmostEqual(2.0 * FACE, plan.clearance_after_mm, delta=0.5)

    def test_where_no_push_opens_room_the_one_that_moves_another_part_too_is_taken(self) -> None:
        # A row of three: every cube has its two long faces free, so no push gains room on two opposite faces. Red
        # before: nothing was pushed. Now the scene changes all the same: a cube pushed with the one ahead of it.
        row = [_block(centre_xy=(column * PITCH, 0.0)) for column in (-1, 0, 1)]
        answer = _scene(row, [True, True, True])
        assert not isinstance(answer, PushRefusal), answer.sentence
        _which, plan = answer
        self.assertGreaterEqual(plan.parts_moved, 2)
        self.assertTrue(plan.shoves)
        self.assertLessEqual(plan.clearance_after_mm, plan.clearance_before_mm + 1e-6)

    def test_where_no_push_opens_10_mm_the_one_that_opens_the_most_is_taken(self) -> None:
        # The grid 8 mm apart: a corner has 8 mm on its inner face, and the edge's middle cube pushed into the grid opens
        # 5.8 mm more, under the 10 mm a part's own push asks. Red before: nothing was pushed.
        pitch = 38.0
        grid = [(column * pitch, row * pitch) for row in (-1, 0, 1) for column in (-1, 0, 1)]
        answer = _scene([_block(centre_xy=centre) for centre in grid], [True] * 9)
        assert not isinstance(answer, PushRefusal), answer.sentence
        which, plan = answer
        self.assertIn(grid[which], {(-pitch, 0.0), (pitch, 0.0), (0.0, -pitch), (0.0, pitch)})
        gain = plan.clearance_after_mm - plan.clearance_before_mm
        self.assertGreater(gain, 0.0)
        self.assertLess(gain, 10.0)

    def test_a_part_alone_is_not_pushed_for_nothing(self) -> None:
        refusal = _scene([_block(centre_xy=(0.0, 0.0))], [True])
        assert isinstance(refusal, PushRefusal)
        self.assertEqual("no_free_direction", refusal.code)
        self.assertTrue(refusal.sentence.endswith("."), refusal.sentence)

    def test_what_nobody_named_is_never_pushed_and_never_wanted(self) -> None:
        # The grid's middle row as what nobody named: only six named cubes are left to push, none of them wanted.
        parts = [cloud for cloud, (_x, y) in zip(_grid(), GRID) if y != 0.0]
        middle = np.vstack([cloud for cloud, (_x, y) in zip(_grid(), GRID) if y == 0.0])
        refusal = _scene(parts, [False] * len(parts), unnamed=middle)
        assert isinstance(refusal, PushRefusal)
        self.assertEqual("no_free_direction", refusal.code)


class TheScenePushHasItsOwnBudgetsTests(unittest.TestCase):

    def test_three_per_pick_and_ten_per_campaign_beside_a_parts_own(self) -> None:
        campaign = PushCampaign(distance_mm=30.0)
        self.assertEqual((3, 10), (SCENE_PUSHES_PER_PICK, SCENE_PUSHES_PER_CAMPAIGN))
        pushed = 0
        for pick in range(4):
            campaign.start_pick()
            for push in range(SCENE_PUSHES_PER_PICK):
                centre = (200.0 * pick, 200.0 * push, 0.0)  # a part of its own each time
                verdict = campaign.scene_budgets.check(centre)
                if not verdict.allowed:
                    self.assertEqual(BUDGET_CAMPAIGN_SPENT, verdict.code)
                    break
                campaign.scene_budgets.record(centre_mm=centre, landing_mm=(centre[0] + 50.0, centre[1], 0.0))
                pushed += 1
            else:
                self.assertEqual(BUDGET_PICK_SPENT, campaign.scene_budgets.check((900.0, 900.0, 0.0)).code)
        self.assertEqual(SCENE_PUSHES_PER_CAMPAIGN, pushed)
        # A part's own push keeps its budgets.
        self.assertTrue(campaign.budgets.check((0.0, 0.0, 0.0)).allowed)
        self.assertEqual(0, campaign.budgets.pushes_this_campaign)


class _Routed:
    """Which way the pick changed the scene, its own push refused: the scene pushed or a blocker cleared."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def patch(self, case: unittest.TestCase) -> None:
        from src.robot.grasping.loop.pick_loop import BinPickingOrchestrator

        for name, said in (("_push_the_failed_part", "part"), ("_push_the_scene", "scene"),
                           ("_clear_the_blockers", "blocker")):
            case.enterContext(mock.patch.object(BinPickingOrchestrator, name, autospec=True,
                                                side_effect=lambda *_a, _said=said, **_k: self.calls.append(_said)
                                                or False))


class TheSceneIsPushedOnlyWhereEveryPartIsTakenTests(unittest.TestCase):

    def _routed(self, **campaign: Any) -> list[str]:
        from tests.robot.grasping.recovery.test_a_boxed_in_part_is_pushed_inside_the_pick import LOOKS, _PushCell

        cell = _PushCell(self)
        routed = _Routed()
        routed.patch(self)
        cell.service.start_campaign(**campaign)
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            cell.service.pick(look=list(LOOKS))
        return routed.calls

    def test_a_task_taking_every_part_pushes_the_scene_and_clears_nothing(self) -> None:
        calls = self._routed(every_part=True)
        self.assertIn("scene", calls)
        self.assertNotIn("blocker", calls)
        self.assertEqual("part", calls[0], "the part's own push comes first")

    def test_a_task_of_one_part_clears_a_blocker(self) -> None:
        calls = self._routed()
        self.assertIn("blocker", calls)
        self.assertNotIn("scene", calls)

    def test_critical_parts_clear_a_blocker_and_push_nothing(self) -> None:
        calls = self._routed(every_part=True, critical_parts=True)
        self.assertIn("blocker", calls)
        self.assertNotIn("scene", calls)


class TheGuardRefusingEveryGraspPushesTooTests(unittest.TestCase):
    """The owner, 2026-10-16: "ja er soll dann schieben". The part has a grasp, and the guard refuses every one: judged
    ahead from the look, the part is pushed first and then, for a task taking every part, the scene (red before: the
    pick ended there); refused at the standoff before the line down was sent, the pick looks again instead of ending,
    and goes on to its grasp."""

    def _cell(self) -> Any:
        from tests.robot.grasping.recovery.test_a_boxed_in_part_is_pushed_inside_the_pick import _PushCell

        return _PushCell(self, freed_on=tuple(range(40)))

    def _routed(self, **campaign: Any) -> list[str]:
        from tests.robot.grasping.recovery.test_a_boxed_in_part_is_pushed_inside_the_pick import LOOKS

        cell = self._cell()
        cell.policy.refusal_ahead = lambda grasp: "the line down would be refused (test)"
        routed = _Routed()
        routed.patch(self)
        cell.service.start_campaign(**campaign)
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            cell.service.pick(look=list(LOOKS))
        return routed.calls

    def test_refused_ahead_a_task_taking_every_part_pushes_the_part_then_the_scene(self) -> None:
        calls = self._routed(every_part=True)
        self.assertEqual(["part", "scene"], calls[:2])
        self.assertNotIn("blocker", calls)

    def test_refused_ahead_a_task_of_one_part_clears_a_blocker_and_pushes_no_scene(self) -> None:
        calls = self._routed()
        self.assertIn("blocker", calls)
        self.assertNotIn("scene", calls)

    def test_refused_at_the_standoff_the_pick_looks_again_and_goes_on(self) -> None:
        from src.robot.core import MotionStatus
        from src.robot.grasping.motion.execution_policy import PolicyOutcome, PolicyReport
        from tests.robot.grasping.recovery.test_a_boxed_in_part_is_pushed_inside_the_pick import LOOKS

        cell = self._cell()
        real = cell.policy.execute
        refused: list[Any] = []

        def execute(grasp: Any) -> Any:
            if not refused:
                refused.append(grasp)
                return PolicyReport(outcome=PolicyOutcome.MOTION_FAILED,
                                    motion_status=MotionStatus.SELF_COLLISION_REJECTED,
                                    motion_message="[safety:self_collision] gripper at a box the camera saw (test)")
            return real(grasp)

        cell.policy.execute = execute  # type: ignore[method-assign]
        cell.service.start_campaign(every_part=True)
        with mock.patch("builtins.input", side_effect=AssertionError("asked at the terminal")):
            report = cell.service.pick(look=list(LOOKS))
        attempts = report.pick_report.attempts
        self.assertEqual("rescan", attempts[0].action, [a.action for a in attempts])
        self.assertEqual("executed", attempts[-1].action, [a.action for a in attempts])
        self.assertEqual(1, len(refused))


class AConsoleTaskSaysItTakesEveryPartTests(ConsoleCase):

    def test_until_empty_reaches_the_campaign(self) -> None:
        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube", scope="until_empty")
        self.finished(run["id"])
        self.assertIs(True, cell.service.campaigns[0].get("every_part"))

    def test_a_task_of_one_part_says_nothing_of_it(self) -> None:
        cell = self.scripted([Scripted("part")])
        run = self.task(object="green cube")
        self.finished(run["id"])
        self.assertNotIn("every_part", cell.service.campaigns[-1])


if __name__ == "__main__":
    unittest.main()
