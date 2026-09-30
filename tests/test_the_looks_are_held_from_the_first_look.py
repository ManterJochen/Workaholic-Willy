"""A pick holds its wrist frames from the moment the arm stands at its first look, not from where it started.

The investigation of 2026-09-30 (scratchpad selfcol_seen_answer.md, finding C3): the pick loop and the locator started
holding the pick's frames before the first look's motion, so the frame grabbed for that motion, at the pose the pick
started from (home, a retreat, the last place), was held for the whole pick and never aged. Whatever that frame saw
that the looks never did, a cable at the start pose, a hand in the cell, stayed an obstacle in every world of the pick.

Now the hold starts once the arm stands at the first look it reaches: the frame of the start pose is not held, and a
look refused before anything was sent does not start it, because the arm still stands where it started. A new look
around lets go of what an earlier one held before it moves. The worlds here write down where the arm stood whenever
they were asked to hold or to let go.
"""

from __future__ import annotations

import unittest
from typing import Any

from tests._wrist_views import LookingArm, joints_key
from tests.test_a_located_part_is_looked_at_from_every_look import EAST, WEST, _key, _locator, _robot, _Wrist
from tests.test_a_located_part_is_looked_at_from_every_look import _arm as _locator_arm
from tests.test_a_wrist_pick_looks_until_its_grasp_is_safe import LOOK_MINUS_X, LOOK_PLUS_X, _Cell, _poses


class _Where:
    """A live planner world that writes down where the arm stood each time it was asked to hold or to let go."""

    def __init__(self, arm: Any, *, holding: bool = False) -> None:
        self.arm = arm
        self.calls: list[tuple[str, tuple[float, ...], int]] = []
        self.holds_pick_views = holding

    @property
    def holding(self) -> bool:
        """What the pick loop's recording policy reads when it is handed a grasp."""
        return self.holds_pick_views

    def _say(self, what: str) -> None:
        self.calls.append((what, joints_key(self.arm.get_joint_positions()), len(self.arm.motions)))

    def hold_pick_views(self) -> bool:
        self._say("hold")
        self.holds_pick_views = True
        return True

    def forget_pick_views(self) -> None:
        self._say("forget")
        self.holds_pick_views = False

    def offer_segmentation(self, **_offer: Any) -> None:
        return None

    def forget_segmentation(self) -> None:
        return None


class ThePickLoopHoldsFromItsFirstLookTests(unittest.TestCase):
    def test_the_hold_starts_with_the_arm_at_the_first_look(self) -> None:
        """⛔ Before, the pick held from where it started: the first look's motion grabbed the start pose's frame."""
        arm = LookingArm(_poses())
        world = _Where(arm)
        arm.live_planner_world = world  # type: ignore[attr-defined]
        start = joints_key(arm.get_joint_positions())
        self.assertNotEqual(start, joints_key(LOOK_PLUS_X))

        _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, grasps_on=(1,)).run()

        holds = [call for call in world.calls if call[0] == "hold"]
        self.assertEqual(len(holds), 1, world.calls)
        self.assertEqual(holds[0][1], joints_key(LOOK_PLUS_X), "the hold did not start at the first look")
        self.assertEqual(holds[0][2], 1, "the hold started before the arm reached its first look")
        self.assertEqual([what for what, _, _ in world.calls], ["hold", "forget"])

    def test_a_first_look_refused_before_anything_was_sent_does_not_start_it(self) -> None:
        arm = LookingArm(_poses(), refuse=(LOOK_PLUS_X,))
        world = _Where(arm)
        arm.live_planner_world = world  # type: ignore[attr-defined]

        _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, grasps_on=(1,)).run()

        holds = [call for call in world.calls if call[0] == "hold"]
        self.assertEqual(len(holds), 1, world.calls)
        self.assertEqual(holds[0][1], joints_key(LOOK_MINUS_X), "the hold started where the arm started")

    def test_a_pick_that_reaches_no_look_holds_nothing(self) -> None:
        arm = LookingArm(_poses(), refuse=(LOOK_PLUS_X, LOOK_MINUS_X))
        world = _Where(arm)
        arm.live_planner_world = world  # type: ignore[attr-defined]

        _Cell(looks=(LOOK_PLUS_X, LOOK_MINUS_X), arm=arm, grasps_on=(1,)).run()

        self.assertEqual([call for call in world.calls if call[0] == "hold"], [])


class TheLocatorHoldsFromItsFirstLookTests(unittest.TestCase):
    def test_the_hold_starts_with_the_arm_at_the_first_look(self) -> None:
        """⛔ Before, the look around held from where the arm stood when it was asked."""
        built = _locator_arm(EAST, WEST)
        world = _Where(built)
        built.live_planner_world = world
        _locator(_Wrist(built)).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(built))

        holds = [call for call in world.calls if call[0] == "hold"]
        self.assertEqual(len(holds), 1, world.calls)
        self.assertEqual(holds[0][1], _key(EAST), "the hold did not start at the first look")
        self.assertEqual(holds[0][2], 1, "the hold started before the arm reached its first look")

    def test_a_new_look_around_lets_go_of_the_last_before_it_moves(self) -> None:
        built = _locator_arm(EAST, WEST)
        world = _Where(built, holding=True)
        built.live_planner_world = world
        _locator(_Wrist(built)).look_around("a part", [WEST.joints, EAST.joints], robot=_robot(built))

        self.assertEqual([what for what, _, _ in world.calls][:2], ["forget", "hold"], world.calls)
        self.assertEqual(world.calls[0][2], 0, "the earlier frames were let go only after the arm moved")
        self.assertEqual(world.calls[1][1], _key(WEST))

    def test_a_look_around_that_reaches_no_look_holds_nothing(self) -> None:
        built = _locator_arm(EAST, WEST, refuse=(EAST.joints, WEST.joints))
        world = _Where(built)
        built.live_planner_world = world
        located = _locator(_Wrist(built)).look_around("a part", [EAST.joints, WEST.joints], robot=_robot(built))

        self.assertIn("reached none of the looks", located.refused)
        self.assertEqual([call for call in world.calls if call[0] == "hold"], [])


class TheWorldSaysWhetherItHoldsTests(unittest.TestCase):
    def test_holds_pick_views_follows_the_hold(self) -> None:
        import numpy as np

        from src.robot.safety.planning.live_world import CameraView, LivePlannerWorld
        from src.robot.safety.planning.perceived import WorldBuildLimits

        world = LivePlannerWorld(
            cameras=(CameraView(name="wrist", depth_source=object(), camera_to_tool=np.eye(4)),),  # type: ignore[arg-type]
            declared=(), limits=WorldBuildLimits(x_mm=(-1.0, 1.0), y_mm=(-1.0, 1.0), z_mm=(-1.0, 1.0),
                                                 support_plane_top_mm=0.0),
        )
        self.assertFalse(world.holds_pick_views)  # type: ignore[attr-defined]
        world.hold_pick_views()
        self.assertTrue(world.holds_pick_views)  # type: ignore[attr-defined]
        world.forget_pick_views()
        self.assertFalse(world.holds_pick_views)  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()

