"""A cuRobo path halted in leg k sends no later waypoint, braked or not.

``CuroboUrPlanner.execute`` runs a judged path as one ``moveJ`` per waypoint, in order. A leg that answered ``True``
before it arrived would let the next waypoint's move start while it still ran, cutting the leg short off the path both
authorities judged; a leg the halt ended must be the last one sent. So:

* brakes on: a halt in leg k brakes it (``stopJ`` from the thread that sent it) and the glue answers CANCELLED naming
  waypoint k of n, with no ``moveJ`` after it;
* brakes off, as shipped (Q1 = A): leg k runs to its end as it always did, and waypoint k+1 is never sent; CANCELLED
  names it;
* halted before the first: nothing is sent at all;
* halted during leg k and cleared before it ended: the path still ends there. The glue counts the halts the connection
  was asked for (``halt_requests``) before the first waypoint and reads the count before each, so a halt that came and
  went inside one leg (or inside one 8 ms read of a watched leg) is not read as no halt.

A refusal that is not the halt's stays what it was (``tests/test_a_halted_arm_with_brakes_off_sends_what_it_always_sent.py``
pins it). The controller is ``tests._halt_fakes.SimulatedUr`` on a virtual clock.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionCommand, MotionStatus
from src.robot.drivers.ur.arm import _Route
from src.robot.drivers.ur.connection import URConnection
from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner
from tests._halt_fakes import SimulatedUr, VirtualClock, no_client, ur_arm_on

_HERE = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
#: Four waypoints, the first where the arm stands (a plan starts there): legs of 0, 0.4, 0.4 and 0.4 s at 0.5 rad/s.
_PATH = [
    list(_HERE),
    [0.2, -1.2, 1.3, -0.4, 1.5, 0.2],
    [0.4, -1.2, 1.3, -0.4, 1.5, 0.2],
    [0.6, -1.2, 1.3, -0.4, 1.5, 0.2],
]
_POSE = Pose(position_mm=np.array([450.0, -120.0, 320.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
             frame=Frame.BASE, label="goal")
_REASON = "the operator pressed halt now"


def _glue(*, brake: bool) -> tuple[CuroboUrPlanner, URConnection, SimulatedUr, VirtualClock]:
    clock = VirtualClock()
    sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
    conn = sim.attach(URConnection("127.0.0.1", vel=0.5, acc=0.8, brake_on_halt=brake, clock=clock,
                                   sleep=clock.sleep))
    return CuroboUrPlanner(conn, client_factory=no_client, vel=0.5, acc=0.8), conn, sim, clock


class TheGlueSendsNoWaypointAfterTheHaltTests(unittest.TestCase):
    def test_brakes_on_a_halt_in_leg_three_brakes_it_and_sends_nothing_after(self) -> None:
        """Red before: there was no halt to read, and every waypoint went out."""
        planner, conn, sim, clock = _glue(brake=True)
        clock.at(0.6, lambda: conn.request_halt(_REASON))  # leg 3 runs from 0.4 s to 0.8 s
        result = planner.execute(_PATH, _POSE)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("waypoint 3 of 4", result.message)
        self.assertIn("braked", result.message)
        self.assertIn(_REASON, result.message)
        self.assertEqual(_PATH[:3], [call[3][0] for call in sim.named("moveJ")])
        self.assertEqual(1, len(sim.named("stopJ")))
        self.assertLess(sim.q[0], _PATH[2][0], "leg 3 was not braked")
        self.assertGreater(sim.q[0], _PATH[1][0])

    def test_brakes_off_the_leg_in_flight_runs_out_and_the_next_waypoint_is_never_sent(self) -> None:
        """Red before: the next waypoint went out after the halt."""
        planner, conn, sim, clock = _glue(brake=False)
        clock.at(0.6, lambda: conn.request_halt(_REASON))
        result = planner.execute(_PATH, _POSE)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("waypoint 4 of 4", result.message)
        self.assertIn("not sent", result.message)
        self.assertEqual(_PATH[:3], [call[3][0] for call in sim.named("moveJ")])
        self.assertEqual([], sim.named("stopJ"))
        np.testing.assert_allclose(sim.q, _PATH[2], err_msg="the leg in flight was cut short with brakes off")

    def test_a_path_halted_before_it_starts_sends_nothing(self) -> None:
        for brake in (True, False):
            with self.subTest(brake_on_halt=brake):
                planner, conn, sim, _clock = _glue(brake=brake)
                conn.request_halt(_REASON)
                result = planner.execute(_PATH, command=MotionCommand.MOVE_JOINTS,
                                         target_joints=JointPositions(_PATH[-1]))
                self.assertIs(MotionStatus.CANCELLED, result.status)
                self.assertEqual(MotionCommand.MOVE_JOINTS, result.command)
                self.assertIn("waypoint 1 of 4", result.message)
                self.assertEqual([], sim.calls)

    def test_a_path_nobody_halts_runs_every_waypoint(self) -> None:
        planner, _conn, sim, _clock = _glue(brake=True)
        self.assertIs(MotionStatus.EXECUTED, planner.execute(_PATH, _POSE).status)
        self.assertEqual(_PATH, [call[3][0] for call in sim.named("moveJ")])
        np.testing.assert_allclose(sim.q, _PATH[-1], atol=2e-3)

    def test_a_brake_nobody_saw_end_says_press_the_emergency_stop_and_sends_nothing_after(self) -> None:
        planner, conn, sim, clock = _glue(brake=True)
        clock.at(0.6, lambda: conn.request_halt(_REASON))
        sim.getActualQd = lambda: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]  # type: ignore[method-assign]
        result = planner.execute(_PATH, _POSE)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("waypoint 3 of 4", result.message)
        self.assertIn("emergency stop", result.message)
        self.assertEqual(3, len(sim.named("moveJ")))

    def test_a_halt_cleared_while_a_leg_runs_still_ends_the_path(self) -> None:
        """Brakes off: halted during leg 2 and cleared before it ended. Red before: the latch was read as it stood before
        each waypoint, so the clear let waypoints 3 and 4 go out after a halt."""
        planner, conn, sim, clock = _glue(brake=False)
        clock.at(0.2, lambda: conn.request_halt(_REASON))  # leg 2 runs from 0 to 0.4 s
        clock.at(0.3, conn.clear_halt)
        result = planner.execute(_PATH, _POSE)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("waypoint 3 of 4", result.message)
        self.assertIn("cleared since", result.message)
        self.assertEqual(_PATH[:2], [call[3][0] for call in sim.named("moveJ")])
        np.testing.assert_allclose(sim.q, _PATH[1])

    def test_with_brakes_on_a_halt_cleared_before_the_poll_read_it_still_ends_the_path(self) -> None:
        """Halted and cleared between two reads of the watch: no brake, the leg arrives, and the path ends there."""
        planner, conn, sim, clock = _glue(brake=True)
        clock.at(0.2, lambda: conn.request_halt(_REASON))
        clock.at(0.2, conn.clear_halt)  # the same instant: the 8 ms poll never sees the latch set
        result = planner.execute(_PATH, _POSE)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("waypoint 3 of 4", result.message)
        self.assertEqual(_PATH[:2], [call[3][0] for call in sim.named("moveJ")])
        self.assertEqual([], sim.named("stopJ"))

    def test_a_halt_requested_before_the_path_began_and_cleared_does_not_end_it(self) -> None:
        """Only a halt during the path ends it: one a person cleared before it began is the past."""
        planner, conn, sim, _clock = _glue(brake=False)
        conn.request_halt(_REASON)
        conn.clear_halt()
        self.assertIs(MotionStatus.EXECUTED, planner.execute(_PATH, _POSE).status)
        self.assertEqual(4, len(sim.named("moveJ")))

    def test_the_connection_counts_the_halts_it_was_asked_for(self) -> None:
        conn = URConnection("127.0.0.1")
        self.assertEqual(0, conn.halt_requests)
        conn.request_halt("one")
        conn.request_halt("a second press keeps the first record")
        self.assertEqual(1, conn.halt_requests)
        conn.clear_halt()
        self.assertEqual(1, conn.halt_requests, "a clear is no request")
        conn.request_halt("two")
        self.assertEqual(2, conn.halt_requests)

    def test_a_connection_double_whose_latch_cannot_be_read_runs_as_before(self) -> None:
        """The glue's own read of the latch never raises: the connection refuses on its own latch regardless."""
        _planner, _conn, sim, _clock = _glue(brake=False)

        class _Broken:
            is_connected = True

            def halt_state(self) -> None:
                raise RuntimeError("the double broke")

            def moveJ(self, joints: list[float], vel: float | None = None, acc: float | None = None) -> bool:
                sim.moveJ(joints, vel or 0.5, acc or 0.8, False)
                return True

        planner = CuroboUrPlanner(_Broken(), client_factory=no_client)  # type: ignore[arg-type]
        self.assertIs(MotionStatus.EXECUTED, planner.execute(_PATH, _POSE).status)
        self.assertEqual(4, len(sim.named("moveJ")))


class TheArmSaysTheBrakedPathTests(unittest.TestCase):
    def test_a_planned_joint_move_halted_mid_path_is_cancelled_with_no_later_waypoint(self) -> None:
        clock = VirtualClock()
        sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
        arm = ur_arm_on(sim, "curobo", brake_on_halt=True)
        arm._conn._clock, arm._conn._sleep = clock, clock.sleep
        route = _Route(waypoints=_PATH, dense=len(_PATH), how="the test's plan", planned=True)
        arm._judge_joint_move = lambda joints, *, command, plan_around=True: (joints, route, None)
        clock.at(0.3, lambda: arm.halt(_REASON))  # 0.6 rad/s: leg 2 runs from 0 to 0.33 s
        with arm.without_camera_world("the halt tests wire no camera world"):
            result = arm.move_to_joints(JointPositions(_PATH[-1]))
        self.assertIs(MotionStatus.CANCELLED, result.status, result.message)
        self.assertIn("waypoint 2 of 4", result.message)
        self.assertEqual(_PATH[:2], [call[3][0] for call in sim.named("moveJ")])
        self.assertEqual(1, len(sim.named("stopJ")))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
