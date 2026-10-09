"""The approach to a part holds the world its grasp was judged in at the look, and takes no frame at the standoff (O2).

The owner, 2026-10-09 (map2 O2, ``safety.planning_world.hold.standoff``): "altes Bild, weil sich in der Kiste so oder so
nichts verändert". A standoff frame in a bin had no depth at all on 12 of 88 arrivals, and on the 10-07 cell and the B
bench it flipped a line down 0.05 mm short of the guard distance that had passed when judged ahead. With the switch on:

* the grasp judged ahead judges its route in the world its line down was judged in, one refresh for the three;
* the move to the standoff and the line down hold that world (``holding_the_approach``): the route runs as judged, and
  the line down is judged in it, every sample by the exact guard, with no frame at the standoff, stamped with the refresh
  the world was built by;
* where the route judged ahead does not stand ready (another standoff, more than half a second, the arm moved), nothing
  is held and every motion refreshes as ever; with the switch off nothing changes at all.

Pinned on the real UR driver, its real guard and planner glue (the fixture of
``test_a_grasp_judged_ahead_runs_as_it_was_judged``), and through the pick's own policy. Honesty bucket (2).
"""

from __future__ import annotations

import dataclasses
import time
import unittest
from typing import Any

import numpy as np

from src.config.schema.robot.safety_schema import HeldWorldConfig
from src.robot.core import MotionStatus
from src.robot.core.camera_world import CameraWorldUse
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grasping.types.grasp_point import GraspFrame, GraspPoint
from tests.test_a_grasp_judged_ahead_runs_as_it_was_judged import (
    _GRASP,
    _STANDOFF,
    _WIDTH_MM,
    _cell,
    _configuration,
    _judged_ahead,
    _moved_1_mm,
    _refreshes,
    _shifted,
    _stand,
)
from tests.test_every_verb_meets_the_camera_world import _mesh_backend_available

_REASON = "the approach, judged in the world the grasp was judged in ahead"


def _holding(arm: Any, standoff: bool = True) -> None:
    arm._holds = HeldWorldConfig(standoff=standoff)


class TheGraspIsJudgedAheadInOneWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def test_the_route_is_judged_in_the_world_its_line_down_was_judged_in(self) -> None:
        arm, events, _solved = _cell()
        _holding(arm)
        with _refreshes() as refresh:
            self.assertEqual("", _judged_ahead(arm))
        self.assertEqual(1, refresh.call_count, "the line down's world alone: the route was judged in it")
        self.assertEqual(1, events.count("gate_planned_path"))
        ahead = arm._judged_ahead
        assert ahead is not None, "the route was not kept for the move to the standoff"
        self.assertIs(arm._planner_refresh(), ahead.refresh)

    def test_off_the_route_refreshes_its_own_world_as_before(self) -> None:
        arm, _events, _solved = _cell()
        with _refreshes() as refresh:
            self.assertEqual("", _judged_ahead(arm))
        self.assertEqual(2, refresh.call_count)


class TheApproachHoldsThatWorldTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _ready(self, *, standoff: bool = True) -> "tuple[Any, list[str]]":
        arm, events, _solved = _cell()
        _holding(arm, standoff)
        self.assertEqual("", _judged_ahead(arm))
        events.clear()
        return arm, events

    def test_the_route_and_the_line_down_take_no_frame_at_the_standoff(self) -> None:
        """⭐ Red before: a frame at the standoff, a world built from it, then the line down judged in it."""
        arm, events = self._ready()
        ahead = arm._judged_ahead
        with arm.holding_the_approach(_STANDOFF, _REASON) as held, _refreshes() as refresh:
            self.assertTrue(held)
            to_standoff = arm.move(_STANDOFF)
            self.assertIs(MotionStatus.EXECUTED, to_standoff.status, to_standoff.message)
            _stand(arm, [float(v) for v in ahead.route.waypoints[-1]])
            down = arm.move(_GRASP, linear=True)
        self.assertIs(MotionStatus.EXECUTED, down.status, down.message)
        self.assertEqual(0, refresh.call_count, "a frame was taken on the approach")
        self.assertNotIn("gate_planned_path", events, "the route was judged a second time")
        self.assertEqual(1, events.count("gate_joint_path"), "the line down was not judged sample by sample")
        for result in (to_standoff, down):
            self.assertIs(CameraWorldUse.PLANNED, result.camera_world.use)
            self.assertEqual(ahead.refresh.captured_at_s, result.camera_world.captured_at_s)
        self.assertIsNone(arm._held_world_reason, "the world is let go after the block")

    def test_off_nothing_is_held_and_the_line_down_takes_its_frame(self) -> None:
        arm, _events = self._ready(standoff=False)
        ahead = arm._judged_ahead
        with arm.holding_the_approach(_STANDOFF, _REASON) as held, _refreshes() as refresh:
            self.assertFalse(held)
            self.assertTrue(arm.move(_STANDOFF).ok)
            _stand(arm, [float(v) for v in ahead.route.waypoints[-1]])
            self.assertTrue(arm.move(_GRASP, linear=True).ok)
        self.assertEqual(1, refresh.call_count, "the line down's own frame at the standoff")

    def test_a_route_that_does_not_stand_ready_holds_nothing(self) -> None:
        cases = {
            "another standoff": lambda arm: _shifted(_STANDOFF, 5.0),
            "more than half a second": lambda arm: (
                setattr(arm, "_judged_ahead", dataclasses.replace(arm._judged_ahead, at=time.monotonic() - 0.6))
                or _STANDOFF),
            "the arm moved": lambda arm: (_moved_1_mm(arm) or _STANDOFF),
        }
        for label, standoff in cases.items():
            with self.subTest(label):
                arm, _events = self._ready()
                with arm.holding_the_approach(standoff(arm), _REASON) as held:
                    self.assertFalse(held)
                    self.assertIsNone(arm._held_world_reason)

    def test_no_route_judged_ahead_holds_nothing(self) -> None:
        arm, _events, _solved = _cell()
        _holding(arm)
        with arm.holding_the_approach(_STANDOFF, _REASON) as held:
            self.assertFalse(held)


class ThePicksOwnApproachTests(unittest.TestCase):
    def setUp(self) -> None:
        if not _mesh_backend_available():
            self.skipTest("no exact mesh backend on this box")

    def _grasp(self) -> GraspPoint:
        return GraspPoint(position=np.array([350.0, -150.0, 120.0]), approach=np.array([0.0, 0.0, -1.0]),
                          axis=np.array([1.0, 0.0, 0.0]), grip_width_mm=_WIDTH_MM, score=0.9, frame=GraspFrame.BASE,
                          label="part")

    def _execute(self, *, standoff: bool) -> "tuple[Any, list[str], Any, int]":
        arm, events, _solved = _cell()
        _holding(arm, standoff)
        policy = GraspExecutionPolicy(arm=arm, gripper=None, standoff_mm=80.0, retreat_mm=100.0)
        grasp = self._grasp()
        self.assertEqual("", policy.refusal_ahead(grasp))
        events.clear()
        at_standoff = _configuration(arm, _STANDOFF, [float(v) for v in arm._conn.get_joint_positions()])
        moved = arm._conn.moveJ.side_effect

        def arrives(*asked: Any, **said: Any) -> bool:
            _stand(arm, [float(v) for v in asked[0]])
            return True if moved is None else moved(*asked, **said)

        arm._conn.moveJ.side_effect = arrives
        with _refreshes() as refresh:
            report = policy.execute(grasp)
        self.assertIsNotNone(at_standoff)
        return arm, events, report, refresh.call_count

    def test_the_pick_takes_no_frame_at_the_standoff(self) -> None:
        arm, events, report, refreshes = self._execute(standoff=True)
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertEqual(0, refreshes, "the pick took a frame between the look and the lift")
        self.assertEqual(0, events.count("gate_planned_path"))

    def test_off_the_pick_takes_its_frame_at_the_standoff_as_before(self) -> None:
        _arm, _events, report, refreshes = self._execute(standoff=False)
        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertEqual(1, refreshes, "the line down's world alone")


if __name__ == "__main__":
    unittest.main()
