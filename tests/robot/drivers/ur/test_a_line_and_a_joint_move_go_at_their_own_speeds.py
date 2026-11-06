"""A line and a joint move go at their own speeds: a moveL at ``robot.motion.linear``, a moveJ at ``robot.motion.joint``.

ur_rtde reads a moveJ's speed as rad/s at the joints and a moveL's as m/s of the TCP. Until 2026-11-06 one pair,
``robot.motion.max_velocity`` and ``max_acceleration``, served both: on the owner's cell (1.0 and 1.0) every line down
to a grasp and up from it ran at up to 1.0 m/s and 1.0 m/s^2, and raising the joints' speed raised every line's with it
(the owner, 2026-11-06: "Liniear und Joint unterschiedlich steuern").

Pinned on the simulated UR, each send read off what the controller was told: a line where its caller names no speed,
a joint move likewise, a caller asking more held to its own motion's ceiling and not the other's, a caller asking less
keeping it; on the ik path, on a cuRobo cell's checked line and through ``move_linear``.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus
from src.robot.drivers.ur.connection import URConnection
from tests._halt_fakes import SimulatedUr, VirtualClock, ur_arm_on

_HERE = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
#: The joints' pair and the line's, apart from each other and from every default, so a send names which it read.
_JOINT = (0.6, 0.9)
_LINE = (0.1, 0.3)
_SPEEDS = {"joint": {"max_velocity": _JOINT[0], "max_acceleration": _JOINT[1]},
           "linear": {"max_velocity": _LINE[0], "max_acceleration": _LINE[1]}}
#: A line's end and a pose the controller solves, as the halt tests drive them on this double.
_LINE_END = Pose(position_mm=np.array([400.0, 400.0, 168.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                 frame=Frame.BASE, label="line")
_POSE = Pose(position_mm=np.array([450.0, -120.0, 120.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
             frame=Frame.BASE, label="pose")
_FAR = [0.6, -1.2, 1.3, -0.4, 1.5, 0.2]


def _arm(planner: str = "ik") -> tuple[Any, SimulatedUr]:
    """A ur5e on ``planner`` over the simulated UR on a clock of its own, its joints at :data:`_JOINT` and its lines at
    :data:`_LINE`; on cuRobo its line judges pass, the judgement being no part of what speed is sent."""
    clock = VirtualClock()
    sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
    arm = ur_arm_on(sim, planner, motion=_SPEEDS)
    arm._conn._clock, arm._conn._sleep = clock, clock.sleep
    if planner == "curobo":
        arm._judge_linear_move = lambda pose, *, command: None
    return arm, sim


def _sent(sim: SimulatedUr, method: str) -> list[tuple[float, float]]:
    """The speed and acceleration of every ``method`` the controller was told, in order."""
    return [(float(call[3][1]), float(call[3][2])) for call in sim.named(method)]


class TheConnectionTests(unittest.TestCase):
    """Where its caller names no speed, a line takes the line's pair and a joint move the joints'."""

    def _connection(self) -> tuple[URConnection, SimulatedUr]:
        clock = VirtualClock()
        sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
        conn = URConnection("127.0.0.1", vel=_JOINT[0], acc=_JOINT[1], line_vel=_LINE[0], line_acc=_LINE[1],
                            clock=clock, sleep=clock.sleep)
        sim.attach(conn)
        return conn, sim

    def test_a_line_with_no_speed_runs_at_the_lines_pair(self) -> None:
        conn, sim = self._connection()
        self.assertTrue(conn.moveL([0.4, 0.1, 0.25, 0.0, 3.1416, 0.0]))
        self.assertEqual([_LINE], _sent(sim, "moveL"))

    def test_a_joint_move_with_no_speed_runs_at_the_joints_pair(self) -> None:
        conn, sim = self._connection()
        self.assertTrue(conn.moveJ(_FAR))
        self.assertEqual([_JOINT], _sent(sim, "moveJ"))

    def test_a_connection_built_with_no_line_pair_runs_its_lines_at_urs_own(self) -> None:
        """Red before: a line ran at the joints' pair, 1.0 rad/s read as 1.0 m/s, wherever nothing else was said."""
        clock = VirtualClock()
        sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
        conn = URConnection("127.0.0.1", vel=1.0, acc=1.0, clock=clock, sleep=clock.sleep)
        sim.attach(conn)
        self.assertTrue(conn.moveL([0.4, 0.1, 0.25, 0.0, 3.1416, 0.0]))
        self.assertEqual([(0.25, 1.2)], _sent(sim, "moveL"))


class TheIkPathTests(unittest.TestCase):
    """``move(pose, linear=...)`` on an ik cell: the controller's moveL or moveJ, each at its own pair."""

    def test_a_line_asking_nothing_runs_at_the_lines_pair(self) -> None:
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True).status)
        self.assertEqual([_LINE], _sent(sim, "moveL"))
        self.assertEqual([], _sent(sim, "moveJ"))

    def test_a_joint_move_asking_nothing_runs_at_the_joints_pair(self) -> None:
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move(_POSE).status)
        self.assertEqual([_JOINT], _sent(sim, "moveJ"))
        self.assertEqual([], _sent(sim, "moveL"))

    def test_a_line_asking_more_is_held_to_the_lines_ceiling_not_the_joints(self) -> None:
        """Red before: held to the joints' 0.6 and 0.9, which a line reads as m/s."""
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True, vel=0.5, acc=0.8).status)
        self.assertEqual([_LINE], _sent(sim, "moveL"))

    def test_a_joint_move_asking_more_is_held_to_the_joints_ceiling(self) -> None:
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move(_POSE, vel=3.0, acc=4.0).status)
        self.assertEqual([_JOINT], _sent(sim, "moveJ"))

    def test_a_line_asking_less_keeps_what_it_asked_for(self) -> None:
        """A push's legs ask 25 and 50 mm/s at 0.1 m/s^2, under any line's ceiling."""
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True, vel=0.05, acc=0.1).status)
        self.assertEqual([(0.05, 0.1)], _sent(sim, "moveL"))

    def test_move_linear_runs_at_the_lines_pair_and_is_held_to_it(self) -> None:
        arm, sim = _arm()
        arm.move_linear(_LINE_END)
        arm.move_linear(_LINE_END, velocity=2.0, acceleration=3.0)
        self.assertEqual([_LINE, _LINE], _sent(sim, "moveL"))

    def test_a_move_to_joints_runs_at_the_joints_pair(self) -> None:
        arm, sim = _arm()
        self.assertIs(MotionStatus.EXECUTED, arm.move_to_joints(JointPositions(_FAR)).status)
        self.assertEqual([_JOINT], _sent(sim, "moveJ"))


class TheCuroboCellTests(unittest.TestCase):
    """A cuRobo cell runs a line as one judged moveL, at the line's pair, wherever it is asked from."""

    def test_a_checked_line_runs_at_the_lines_pair_and_is_held_to_it(self) -> None:
        arm, sim = _arm("curobo")
        with arm.without_camera_world("the speed tests wire no camera world"):
            self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True).status)
            self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True, vel=0.9, acc=2.0).status)
            self.assertIs(MotionStatus.EXECUTED, arm.move(_LINE_END, linear=True, vel=0.05, acc=0.1).status)
        self.assertEqual([_LINE, _LINE, (0.05, 0.1)], _sent(sim, "moveL"))


if __name__ == "__main__":
    unittest.main()
