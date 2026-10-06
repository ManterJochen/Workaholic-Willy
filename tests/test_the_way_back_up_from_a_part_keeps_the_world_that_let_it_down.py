"""The way back up from a part is judged against the world that let the line down run (the owner, 2026-10-06: "Ja,
beides").

On the grasp bench a pick in a bin of 22 parts went down its judged line to the part, and stood there: the lift judged
before the jaws close was refused, and so was the line back up it had just come down, each at its first sample, a finger
or the wrist camera 0.4 to 1.1 mm from a box (the guard keeps 3 mm). The line down had left the region between the jaws
out of its world at its end, the part; the lines back up left it out at theirs, high above it, and their world, refreshed
at the part, held the part between the fingers. What this file pins:

* a line leaves the region between the jaws out at its lower end, as its world is built about it: at the end of a line
  down, at the start of a line up (``URRobotArm._judge_linear_move``);
* inside ``held_world`` a line refreshes no world: it is judged against the one the last refresh built, and stamped
  with that refresh; outside it refreshes as ever (``HoldsItsWorld``);
* a grasp holds the arm's world from the part until the arm is back up: the lift judged before the jaws close, the
  close, the attach and the lift, or the line back up with the jaws open; the move to the standoff and the line down
  refresh as ever, and an arm that does not hold its world grasps as before.

Honesty bucket (1) for the grasp, the real policy over a scripted arm and the owner's toggle hand; (2) for the driver,
the real UR driver, its real guard and planner client over the test bench's camera and sidecar.
"""

from __future__ import annotations

import unittest
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import numpy as np

from src.geometry import Pose
from src.robot.core.arm_capabilities import HoldsItsWorld
from src.robot.core.camera_world import CameraWorldUse
from src.robot.drivers.ur import curobo_motion
from src.robot.drivers.ur.curobo_motion import UR_ARM_JOINT_NAMES
from src.robot.grasping.motion.execution_policy import PolicyOutcome
from tests.test_a_grasp_judges_its_lift_carrying_before_it_closes import (
    _HELD,
    _LIFT,
    _PART,
    _STANDOFF,
    _grasp,
    _JudgingArm,
    _policy,
)
from tests.test_a_stopped_controller_moves_no_jaws import _toggle
from tests.test_every_verb_meets_the_camera_world import _Camera as _BenchCamera
from tests.test_every_verb_meets_the_camera_world import _Client, _mesh_backend_available, _ur
from tests.test_every_verb_meets_the_camera_world import _world as _bench_world
from tests.test_the_goal_keeps_its_jaws_clear import _expected_at, _regions


class _HoldingArm(_JudgingArm):
    """The judging arm, holding its world in a block it records: ``("held", reason)`` as it opens, ``"released"`` as it
    closes, on the log the hand shares."""

    @contextmanager
    def held_world(self, reason: str) -> Iterator[None]:
        self.events.append(("held", reason))
        try:
            yield
        finally:
            self.events.append("released")


def _between(events: "list[Any]") -> "list[Any]":
    """What happened inside the held block."""
    opened = next(i for i, e in enumerate(events) if isinstance(e, tuple) and e[0] == "held")
    closed = events.index("released")
    return events[opened + 1:closed]


def _before(events: "list[Any]") -> "list[Any]":
    opened = next(i for i, e in enumerate(events) if isinstance(e, tuple) and e[0] == "held")
    return events[:opened]


class AGraspHoldsItsWorldFromThePartUntilItIsBackUpTests(unittest.TestCase):
    def test_the_holding_arm_is_one(self) -> None:
        self.assertIsInstance(_HoldingArm([]), HoldsItsWorld)
        self.assertNotIsInstance(_JudgingArm([]), HoldsItsWorld)

    def test_a_lift_that_runs_is_judged_closed_on_attached_and_run_in_the_held_world(self) -> None:
        events: list[Any] = []
        report = _policy(_HoldingArm(events), _toggle(events)).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        before = [e for e in _before(events) if isinstance(e, tuple) and e[0] == "move"]
        self.assertEqual([("move", _STANDOFF, False), ("move", _PART, True)], before,
                         "the move to the standoff and the line down refresh as ever, outside the block")
        inside = _between(events)
        kinds = [e[0] if isinstance(e, tuple) else e for e in inside]
        self.assertIn("judge", kinds, "the lift was not judged inside the held world")
        self.assertIn("attach", kinds, "the part was not attached inside the held world")
        self.assertIn(("move", _LIFT, True), inside, "the lift did not run inside the held world")
        self.assertLess(kinds.index("judge"), kinds.index("attach"))
        self.assertEqual(1, sum(1 for e in events if isinstance(e, tuple) and e[0] == "held"))

    def test_a_lift_refused_backs_out_up_the_line_in_the_held_world(self) -> None:
        events: list[Any] = []
        report = _policy(_HoldingArm(events, carried=_HELD), _toggle(events)).execute(_grasp())

        self.assertIs(PolicyOutcome.CARRIED_RETREAT_REFUSED, report.outcome, report.motion_message)
        inside = _between(events)
        self.assertIn(("move", _STANDOFF, True), inside, "the line back up did not run inside the held world")
        self.assertFalse([e for e in inside if isinstance(e, tuple) and e[0] == "attach"], "a part was attached")

    def test_the_held_block_says_why(self) -> None:
        events: list[Any] = []
        _policy(_HoldingArm(events), _toggle(events)).execute(_grasp())
        (reason,) = [e[1] for e in events if isinstance(e, tuple) and e[0] == "held"]
        self.assertIn("the world the line down was judged in", reason)

    def test_an_arm_that_does_not_hold_its_world_grasps_as_before(self) -> None:
        events: list[Any] = []
        report = _policy(_JudgingArm(events), _toggle(events)).execute(_grasp())
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "held"])

    def test_an_arm_that_keeps_no_line_holds_nothing(self) -> None:
        """No line motion reading, no line down: there is no world a line down was judged in to hold."""

        class _Plain(_HoldingArm):
            def line_motion(self) -> Any:  # type: ignore[override]
                return None

        events: list[Any] = []
        _policy(_Plain(events), _toggle(events)).execute(_grasp())
        self.assertFalse([e for e in events if isinstance(e, tuple) and e[0] == "held"])


def _shifted(pose: Pose, dz_mm: float) -> Pose:
    position = np.asarray(pose.position_mm, dtype=np.float64).copy()
    position[2] += float(dz_mm)
    return Pose(position_mm=position, quaternion_xyzw=np.asarray(pose.quaternion_xyzw, dtype=np.float64),
                frame=pose.frame, label=f"{dz_mm:+.0f} mm")


class TheUrDriverTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _arm(self) -> Any:
        return _ur(_bench_world(_BenchCamera(silent=False)), _Client(joint_names=UR_ARM_JOINT_NAMES), [])

    def _region_of(self, arm: Any, goal: Pose) -> np.ndarray:
        with patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world) as refresh:
            arm.move(goal, linear=True)
        goals = _regions(refresh)
        self.assertEqual(1, len(goals), "the line refreshed its world once")
        (goal_keep_out,) = goals
        assert goal_keep_out.region is not None, goal_keep_out.reason  # type: ignore[attr-defined]
        return goal_keep_out.region.matrix()  # type: ignore[attr-defined]

    def test_a_line_up_leaves_out_the_region_between_the_jaws_where_it_starts(self) -> None:
        arm = self._arm()
        start = arm.get_tcp_pose()
        region = self._region_of(arm, _shifted(start, +40.0))
        np.testing.assert_allclose(region, _expected_at(start.to_matrix()), atol=1e-6)

    def test_a_line_down_leaves_it_out_where_it_ends(self) -> None:
        arm = self._arm()
        start = arm.get_tcp_pose()
        goal = _shifted(start, -40.0)
        region = self._region_of(arm, goal)
        np.testing.assert_allclose(region, _expected_at(goal.to_matrix()), atol=1e-6)

    def test_inside_a_held_world_a_line_refreshes_nothing_and_is_stamped_with_the_world_it_was_judged_in(self) -> None:
        arm = self._arm()
        start = arm.get_tcp_pose()
        arm.move(_shifted(start, -30.0), linear=True)  # the line down: its refresh builds the world
        built = arm._curobo_ur_planner().last_world_refresh
        self.assertIsNotNone(built)
        with patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world) as refresh:
            with arm.held_world("the way back up from the part"):
                held = arm.move(start, linear=True)
        self.assertEqual(0, refresh.call_count, "a line inside the held world refreshed its world")
        self.assertIs(built, arm._curobo_ur_planner().last_world_refresh)
        self.assertTrue(held.ok, held.message)
        self.assertIs(CameraWorldUse.PLANNED, held.camera_world.use)
        with patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world) as refresh:
            arm.move(_shifted(arm.get_tcp_pose(), -30.0), linear=True)
        self.assertEqual(1, refresh.call_count, "outside the block a line refreshes as ever")

    def test_a_held_world_with_no_world_built_yet_refreshes_as_ever(self) -> None:
        arm = self._arm()
        self.assertIsNone(arm._curobo_ur_planner().last_world_refresh)
        with patch.object(curobo_motion, "refresh_planner_world", wraps=curobo_motion.refresh_planner_world) as refresh:
            with arm.held_world("the way back up from the part"):
                arm.move(_shifted(arm.get_tcp_pose(), -30.0), linear=True)
        self.assertEqual(1, refresh.call_count)

    def test_a_held_world_says_why(self) -> None:
        arm = self._arm()
        with self.assertRaises(ValueError):
            with arm.held_world("  "):
                pass
        self.assertIsNone(arm._held_world_reason)


if __name__ == "__main__":
    unittest.main()
