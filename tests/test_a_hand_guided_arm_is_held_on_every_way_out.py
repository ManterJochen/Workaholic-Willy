"""A UR arm freed for hand guiding is held again on every way out of its session, behind a watchdog while free.

Owner decision 2 (freedrive calibration): the software frees the arm, a person moves it, Enter holds it, and a
watchdog locks the arm if the Python process dies. ur_rtde 1.6.5 on the owner's CB3 (PolyScope 3.15) offers teach
mode (``teachMode``/``endTeachMode``; ``freedriveMode`` is a ``$5.10`` no-op there) and ``setWatchdog`` with the stop
action, which lives as long as the control program, so ``hold()`` starts a fresh program to clear it before a gap in
the kicks or a later blocking moveJ could trip it.

Measured on the CB3 URSim 3.15.8 (``src/robot/drivers/ur/freedrive.py`` has the numbers): with no watchdog a killed
client leaves the arm free, with it the arm is protective-stopped within 0.7 s, and every trip is a protective stop.
``reuploadScript`` on a control interface a few seconds old uploaded programs that never ran, so the fresh program
comes from a new control interface (``URConnection.restart_control_script``).
The connection here is a recording double of :class:`URConnection`'s hand-guiding surface, so the order of the
controller calls is read off one event list; nothing here moves anything.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    FreedriveSample,
    FreedriveSession,
    RobotConnectionError,
    RobotEmergencyStop,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
)
from src.robot.drivers.ur.freedrive import WATCHDOG_MIN_HZ, URFreedriveSession

_RUNNING = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False)
_PROTECTIVE = RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                          emergency_stopped=False, message="PROTECTIVE_STOP")
_IDLE = RobotStatus(RobotMode.IDLE, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False)


class _Conn:
    """The hand-guiding surface of ``URConnection``, recording every controller call in order."""

    def __init__(self, *, speeds: "list[list[float]] | None" = None) -> None:
        self.events: list[object] = []
        self.is_connected = True
        self.running = True
        self.protective = False
        self.watchdog_ok = True
        self.teach_ok = True
        self.end_teach_ok = True
        #: Whether the new control interface comes up; where it does not, every interface is closed.
        self.restart_ok = True
        self.in_teach_mode = False
        self.watchdog_armed = False
        #: The joint speeds each read returns in turn; the last one repeats.
        self.speeds = speeds or [[0.0] * 6]

    def set_watchdog(self, hz: float) -> bool:
        self.events.append(("set_watchdog", hz))
        self.watchdog_armed = self.watchdog_ok
        return self.watchdog_ok

    def teach_mode(self) -> bool:
        self.events.append("teach_mode")
        self.in_teach_mode = self.teach_ok
        return self.teach_ok

    def end_teach_mode(self) -> bool:
        self.events.append("end_teach_mode")
        if self.end_teach_ok:
            self.in_teach_mode = False
        return self.end_teach_ok

    def kick_watchdog(self) -> bool:
        self.events.append("kick_watchdog")
        return self.running

    def restart_control_script(self) -> None:
        self.events.append("restart_control_script")
        # The old program stops, and its teach mode and its watchdog end with it.
        self.running = self.in_teach_mode = self.watchdog_armed = False
        if not self.restart_ok:
            self.is_connected = False
            raise RuntimeError("RTDE control script is not running!")
        self.running = True

    def is_program_running(self) -> bool:
        return self.running

    def is_protective_stopped(self) -> bool:
        return self.protective

    def is_emergency_stopped(self) -> bool:
        return False

    def get_joint_positions(self) -> list[float]:
        return [0.1, -1.2, 1.3, -0.4, 1.5, 0.2]

    def get_joint_speeds(self) -> list[float]:
        self.events.append("get_joint_speeds")
        return list(self.speeds.pop(0) if len(self.speeds) > 1 else self.speeds[0])


class _Arm:
    """What the arm hands a session: its TCP, its controller state, and the record every motion verb reads."""

    def __init__(self) -> None:
        self.status = _RUNNING
        self.open: list[URFreedriveSession] = []
        self.closed: list[URFreedriveSession] = []

    def tcp_pose(self) -> Pose:
        # Tool down: 180 degrees about base Y.
        return Pose(position_mm=np.array([400.0, -120.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE)

    def robot_status(self) -> RobotStatus:
        return self.status

    def on_open(self, session: URFreedriveSession) -> None:
        self.open.append(session)

    def on_close(self, session: URFreedriveSession) -> None:
        self.closed.append(session)


def _session(
    conn: _Conn, arm: "_Arm | None" = None, *, sleep: "Callable[[float], None] | None" = None,
) -> "tuple[URFreedriveSession, _Arm]":
    arm = arm or _Arm()
    clock = iter(float(i) * 0.01 for i in range(100000))
    session = URFreedriveSession(
        conn,  # type: ignore[arg-type]
        tcp_pose=arm.tcp_pose, robot_status=arm.robot_status, on_open=arm.on_open, on_close=arm.on_close,
        clock=lambda: next(clock), sleep=sleep or (lambda _s: None),
    )
    return session, arm


def _controller_calls(conn: _Conn) -> list[object]:
    """The calls that change the controller's state, without the reads."""
    return [e for e in conn.events if e not in ("get_joint_speeds", "kick_watchdog")]


#: What free() and then hold() ask of the controller, in this order.
_FREE = [("set_watchdog", WATCHDOG_MIN_HZ), "teach_mode"]
_HOLD = ["end_teach_mode", "restart_control_script"]


class TheOrderOfTheCallsTests(unittest.TestCase):
    def test_it_is_a_freedrive_session(self) -> None:
        session: FreedriveSession = _session(_Conn())[0]
        for member in ("is_free", "free", "hold", "sample", "__enter__", "__exit__"):
            self.assertTrue(hasattr(session, member), member)

    def test_the_watchdog_is_armed_before_teach_mode_and_cleared_when_the_arm_holds(self) -> None:
        conn = _Conn()
        session, arm = _session(conn)
        with session:
            self.assertEqual(arm.open, [session])
            session.free()
            self.assertTrue(session.is_free)
            self.assertTrue(conn.in_teach_mode)
            self.assertTrue(conn.watchdog_armed)
        self.assertEqual(_controller_calls(conn), _FREE + _HOLD)
        self.assertFalse(session.is_free)
        self.assertFalse(conn.in_teach_mode)
        self.assertFalse(conn.watchdog_armed, "a watchdog left armed stops the program during the next blocking moveJ")
        self.assertEqual(arm.closed, [session])

    def test_a_fresh_program_runs_after_the_hold_so_the_next_move_works(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            session.free()
            session.hold()
            self.assertTrue(conn.running)
            self.assertFalse(conn.watchdog_armed)

    def test_a_new_interface_that_comes_up_with_no_program_is_raised(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)

        def silent() -> None:
            conn.events.append("restart_control_script")
            conn.running = conn.in_teach_mode = conn.watchdog_armed = False

        conn.restart_control_script = silent  # type: ignore[method-assign]
        with self.assertRaisesRegex(RobotConnectionError, "came up with no program running"), session:
            session.free()
        self.assertFalse(session.is_free)

    def test_a_watchdog_the_controller_did_not_take_frees_nothing(self) -> None:
        conn = _Conn()
        conn.watchdog_ok = False
        session, _ = _session(conn)
        with session:
            with self.assertRaisesRegex(RobotConnectionError, "watchdog"):
                session.free()
            self.assertFalse(session.is_free)
        self.assertNotIn("teach_mode", conn.events)

    def test_a_refused_teach_mode_leaves_no_watchdog_behind(self) -> None:
        conn = _Conn()
        conn.teach_ok = False
        session, _ = _session(conn)
        with session, self.assertRaisesRegex(RobotConnectionError, "refused teach mode"):
            session.free()
        self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(conn), _FREE + _HOLD[1:])
        self.assertFalse(conn.watchdog_armed)

    def test_an_sdk_fault_while_freeing_is_named_and_leaves_no_watchdog(self) -> None:
        conn = _Conn()

        def teach_mode() -> bool:
            conn.events.append("teach_mode")
            raise RuntimeError("RTDE control script is not running!")

        conn.teach_mode = teach_mode  # type: ignore[method-assign]
        session, _ = _session(conn)
        with session, self.assertRaisesRegex(RobotConnectionError, "RTDE control script is not running"):
            session.free()
        self.assertIn("restart_control_script", conn.events)
        self.assertFalse(session.is_free)

    def test_free_and_hold_are_idempotent(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            session.hold()
            session.free()
            session.free()
            session.hold()
            session.hold()
        self.assertEqual(_controller_calls(conn), _FREE + _HOLD)

    def test_the_arm_is_freed_again_after_a_hold(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            session.free()
            session.hold()
            session.free()
        self.assertEqual(_controller_calls(conn), _FREE + _HOLD + _FREE + _HOLD)


class EveryWayOutHoldsTests(unittest.TestCase):
    def test_an_exception_in_the_block_holds_and_propagates(self) -> None:
        conn = _Conn()
        session, arm = _session(conn)
        with self.assertRaisesRegex(ValueError, "the board fell over"), session:
            session.free()
            raise ValueError("the board fell over")
        self.assertEqual(_controller_calls(conn)[-2:], _HOLD)
        self.assertFalse(conn.in_teach_mode)
        self.assertEqual(arm.closed, [session])

    def test_ctrl_c_in_the_block_holds(self) -> None:
        conn = _Conn()
        session, arm = _session(conn)
        with self.assertRaises(KeyboardInterrupt), session:
            session.free()
            raise KeyboardInterrupt
        self.assertEqual(_controller_calls(conn)[-2:], _HOLD)
        self.assertFalse(conn.in_teach_mode)
        self.assertEqual(arm.closed, [session])

    def test_ctrl_c_while_the_arm_settles_finds_the_watchdog_cleared_already(self) -> None:
        conn = _Conn(speeds=[[0.0] * 6, [0.2, 0, 0, 0, 0, 0]])  # still when freed, moving when held

        def interrupted(_s: float) -> None:
            raise KeyboardInterrupt

        session, _ = _session(conn, sleep=interrupted)
        with session:
            session.free()
            with self.assertRaises(KeyboardInterrupt):
                session.hold()
            self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(conn)[-2:], _HOLD)
        self.assertFalse(conn.watchdog_armed)

    def test_a_refused_end_of_teach_mode_is_ended_by_the_restart(self) -> None:
        conn = _Conn()
        conn.end_teach_ok = False
        session, _ = _session(conn)
        with session:
            session.free()
        self.assertEqual(_controller_calls(conn)[-2:], _HOLD)
        self.assertFalse(conn.in_teach_mode)

    def test_hold_waits_until_the_joints_stand_still(self) -> None:
        moving = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        conn = _Conn(speeds=[[0.0] * 6, moving, moving, [0.0, 0.001, 0.0, 0.0, 0.0, 0.0]])
        naps: list[float] = []
        session, _ = _session(conn, sleep=naps.append)
        with session:
            session.free()
            conn.events.clear()
            session.hold()
        # The restart comes first: waiting with an armed watchdog and nothing kicking it is a protective stop.
        self.assertEqual(conn.events, _HOLD + ["get_joint_speeds"] * 3)
        self.assertEqual(len(naps), 2)

    def test_a_failed_restart_is_raised_where_the_block_ended_normally(self) -> None:
        conn = _Conn()
        session, arm = _session(conn)
        with self.assertRaisesRegex(RobotConnectionError, "could not be restarted.*reconnects"), session:
            session.free()
            conn.restart_ok = False
        self.assertFalse(session.is_free)
        self.assertFalse(conn.is_connected, "the arm reads as disconnected, so nothing moves before a reconnect")
        self.assertEqual(arm.closed, [session])

    def test_a_failed_restart_does_not_replace_the_error_that_ended_the_block(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with self.assertLogs("URFreedrive", level="ERROR") as logs, self.assertRaisesRegex(ValueError, "first"), \
                session:
            session.free()
            conn.restart_ok = False
            raise ValueError("first")
        self.assertIn("could not hold the arm cleanly", "\n".join(logs.output))

    def test_a_closed_connection_counts_as_held(self) -> None:
        """disconnect() stops the control script, and teach mode ends with it."""
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            session.free()
            conn.is_connected = False
        self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(conn), [("set_watchdog", WATCHDOG_MIN_HZ), "teach_mode"])


class WhatIsRefusedTests(unittest.TestCase):
    def test_free_outside_the_block_is_refused(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with self.assertRaisesRegex(RobotMotionRejected, "inside `with arm.freedrive"):
            session.free()
        with self.assertRaises(RobotMotionRejected):
            session.sample()
        self.assertEqual(conn.events, [])

    def test_a_session_is_used_once(self) -> None:
        session, _ = _session(_Conn())
        with session:
            pass
        with self.assertRaisesRegex(RobotMotionRejected, "left already"):
            session.__enter__()

    def test_an_arm_moving_by_itself_is_not_freed(self) -> None:
        conn = _Conn(speeds=[[0.0, 0.0, 0.05, 0.0, 0.0, 0.0]])
        session, _ = _session(conn)
        with session, self.assertRaisesRegex(RobotMotionRejected, "still moves by itself"):
            session.free()
        self.assertEqual(_controller_calls(conn), [])

    def test_a_controller_that_is_not_operational_is_not_freed(self) -> None:
        for status, error in ((_PROTECTIVE, RobotEmergencyStop), (_IDLE, RobotMotionRejected)):
            with self.subTest(status=status.safety_mode.value, mode=status.robot_mode.value):
                conn = _Conn()
                session, arm = _session(conn)
                arm.status = status
                with session, self.assertRaises(error):
                    session.free()
                self.assertEqual(_controller_calls(conn), [])


class SampleTests(unittest.TestCase):
    def test_a_sample_while_free_kicks_the_watchdog_first(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            held = session.sample()
            self.assertNotIn("kick_watchdog", conn.events, "a held arm has no watchdog to kick")
            session.free()
            conn.events.clear()
            sample = session.sample()
            self.assertEqual(conn.events[0], "kick_watchdog")
        for reading in (held, sample):
            self.assertIsInstance(reading, FreedriveSample)
        self.assertEqual(sample.joints_rad, (0.1, -1.2, 1.3, -0.4, 1.5, 0.2))
        self.assertEqual(sample.tcp_xyz_mm, (400.0, -120.0, 300.0))
        np.testing.assert_allclose(sample.tcp_rotvec_rad, (0.0, math.pi, 0.0), atol=1e-12)
        self.assertEqual(sample.peak_joint_speed_rad_s, 0.0)
        self.assertGreater(sample.t_s, held.t_s)

    def test_a_program_that_ended_while_free_is_raised_after_the_session_marked_itself_held(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with session:
            session.free()
            conn.running = False  # the watchdog stopped it, or a stop on the pendant did
            conn.in_teach_mode = False
            conn.events.clear()
            with self.assertRaisesRegex(RobotConnectionError, "no longer hand-guided: the control program ended"):
                session.sample()
            self.assertFalse(session.is_free)
            self.assertIn("restart_control_script", conn.events, "control is given back where no stop forbids it")
            conn.events.clear()
        self.assertEqual(_controller_calls(conn), [], "the exit finds the arm held already")

    def test_a_protective_stop_while_free_is_raised_as_a_stop(self) -> None:
        conn = _Conn()
        session, arm = _session(conn)
        with session:
            session.free()
            conn.protective = True
            conn.running = False
            arm.status = _PROTECTIVE
            conn.events.clear()
            with self.assertRaisesRegex(RobotEmergencyStop, "Clear the stop at the pendant") as caught:
                session.sample()
            self.assertFalse(session.is_free)
        self.assertNotIn("restart_control_script", conn.events, "a stopped controller starts no program")
        self.assertIn(f"not called for {1.0 / WATCHDOG_MIN_HZ:.2f} s", str(caught.exception),
                      "a watchdog trip reads as a protective stop, so the stop names it")

    def test_holding_a_stopped_arm_says_so_at_once_rather_than_waiting_for_a_program(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)
        with self.assertRaisesRegex(RobotEmergencyStop, "controller is stopped"), session:
            session.free()
            conn.protective = True  # nobody sampled, so the session still believes the arm is free
            conn.running = False
        self.assertNotIn("restart_control_script", conn.events)
        self.assertFalse(session.is_free)

    def test_a_read_that_fails_while_free_holds_the_arm(self) -> None:
        conn = _Conn()
        session, _ = _session(conn)

        def broken() -> list[float]:
            raise RuntimeError("receive interface lost")

        with session:
            session.free()
            conn.get_joint_positions = broken  # type: ignore[method-assign]
            with self.assertRaisesRegex(RobotConnectionError, "receive interface lost"):
                session.sample()
            self.assertFalse(session.is_free)
            self.assertFalse(conn.in_teach_mode, "the restart ends the teach mode the program was still in")


if __name__ == "__main__":
    unittest.main()
