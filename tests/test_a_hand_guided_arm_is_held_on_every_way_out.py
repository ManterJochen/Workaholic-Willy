"""An arm freed for hand guiding is held again on every way out of its session, behind a watchdog while free.

The owner's decision of 2026-09-24 (freedrive calibration): the software frees the arm, a person moves it, Enter holds it, and a
watchdog locks the arm if the Python process dies. On the UR, the one controller that offers it so far, that is teach
mode behind the RTDE watchdog with the stop action, which lives as long as the control program, so ``hold()`` starts a
fresh program to clear it before a gap in the kicks or a later blocking moveJ could trip it
(``src/robot/drivers/planned/freedrive.py`` and ``URConnection.restart_control_script`` have the URSim
measurements).

The transport here is a recording double of the ``ControllerFreedrive`` surface and the two ``ArmTransport`` reads a
session makes, so the order of the controller calls is read off one event list; nothing here moves anything.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import (
    ArmFault,
    FreedriveSample,
    FreedriveSession,
    RobotConnectionError,
    RobotMode,
    RobotMotionRejected,
    RobotStatus,
    SafetyMode,
)
from src.robot.core.transport import ControllerFreedrive
from src.robot.drivers.planned.freedrive import WATCHDOG_MIN_HZ, PlannedFreedriveSession

_RUNNING = RobotStatus(RobotMode.RUNNING, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False,
                       program_running=True)
_PROTECTIVE = RobotStatus(RobotMode.RUNNING, SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                          emergency_stopped=False, message="PROTECTIVE_STOP", program_running=True)
_IDLE = RobotStatus(RobotMode.IDLE, SafetyMode.NORMAL, protective_stopped=False, emergency_stopped=False,
                    program_running=True)


class _Transport:
    """The hand-guiding surface of a transport, recording every controller call in order."""

    def __init__(self, *, speeds: "list[list[float]] | None" = None) -> None:
        self.events: list[object] = []
        self.is_connected = True
        self.running = True
        self.protective = False
        self.watchdog_ok = True
        self.guiding_ok = True
        self.end_guiding_ok = True
        #: Whether the fresh control program comes up; where it does not, the link is torn down.
        self.restart_ok = True
        self.guided = False
        self.watchdog_armed = False
        #: The joint speeds each read returns in turn; the last one repeats.
        self.speeds = speeds or [[0.0] * 6]

    def arm_watchdog(self, min_frequency_hz: float) -> bool:
        self.events.append(("arm_watchdog", min_frequency_hz))
        self.watchdog_armed = self.watchdog_ok
        return self.watchdog_ok

    def begin_hand_guiding(self) -> bool:
        self.events.append("begin_hand_guiding")
        self.guided = self.guiding_ok
        return self.guiding_ok

    def end_hand_guiding(self) -> bool:
        self.events.append("end_hand_guiding")
        if self.end_guiding_ok:
            self.guided = False
        return self.end_guiding_ok

    def kick_watchdog(self) -> bool:
        self.events.append("kick_watchdog")
        return self.running

    def restart_control_program(self) -> None:
        self.events.append("restart_control_program")
        # The old program stops, and its hand guiding and its watchdog end with it.
        self.running = self.guided = self.watchdog_armed = False
        if not self.restart_ok:
            self.is_connected = False
            raise RuntimeError("RTDE control script is not running!")
        self.running = True

    def control_program_running(self) -> bool:
        return self.running

    def stop_latched(self) -> bool:
        return self.protective

    def read_joints(self) -> list[float]:
        return [0.1, -1.2, 1.3, -0.4, 1.5, 0.2]

    def read_joint_speeds(self) -> list[float]:
        self.events.append("read_joint_speeds")
        return list(self.speeds.pop(0) if len(self.speeds) > 1 else self.speeds[0])

    def read_payload(self) -> None:
        return None


class _Arm:
    """What the arm hands a session: its TCP, its controller state, and the record every motion verb reads."""

    def __init__(self) -> None:
        self.status = _RUNNING
        self.open: list[PlannedFreedriveSession] = []
        self.closed: list[PlannedFreedriveSession] = []

    def tcp_pose(self) -> Pose:
        # Tool down: 180 degrees about base Y.
        return Pose(position_mm=np.array([400.0, -120.0, 300.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE)

    def robot_status(self) -> RobotStatus:
        return self.status

    def on_open(self, session: PlannedFreedriveSession) -> None:
        self.open.append(session)

    def on_close(self, session: PlannedFreedriveSession) -> None:
        self.closed.append(session)


def _session(
    transport: _Transport, arm: "_Arm | None" = None, *, sleep: "Callable[[float], None] | None" = None,
) -> "tuple[PlannedFreedriveSession, _Arm]":
    arm = arm or _Arm()
    clock = iter(float(i) * 0.01 for i in range(100000))
    session = PlannedFreedriveSession(
        transport, tcp_pose=arm.tcp_pose, robot_status=arm.robot_status, on_open=arm.on_open, on_close=arm.on_close,
        clock=lambda: next(clock), sleep=sleep or (lambda _s: None),
    )
    return session, arm


def _controller_calls(transport: _Transport) -> list[object]:
    """The calls that change the controller's state, without the reads."""
    return [e for e in transport.events if e not in ("read_joint_speeds", "kick_watchdog")]


#: What free() and then hold() ask of the controller, in this order.
_FREE = [("arm_watchdog", WATCHDOG_MIN_HZ), "begin_hand_guiding"]
_HOLD = ["end_hand_guiding", "restart_control_program"]


class TheOrderOfTheCallsTests(unittest.TestCase):
    def test_it_is_a_freedrive_session_and_the_double_is_the_transport_surface(self) -> None:
        session: FreedriveSession = _session(_Transport())[0]
        for member in ("is_free", "free", "hold", "sample", "__enter__", "__exit__"):
            self.assertTrue(hasattr(session, member), member)
        self.assertIsInstance(_Transport(), ControllerFreedrive)

    def test_the_watchdog_is_armed_before_hand_guiding_and_cleared_when_the_arm_holds(self) -> None:
        transport = _Transport()
        session, arm = _session(transport)
        with session:
            self.assertEqual(arm.open, [session])
            session.free()
            self.assertTrue(session.is_free)
            self.assertTrue(transport.guided)
            self.assertTrue(transport.watchdog_armed)
        self.assertEqual(_controller_calls(transport), _FREE + _HOLD)
        self.assertFalse(session.is_free)
        self.assertFalse(transport.guided)
        self.assertFalse(transport.watchdog_armed,
                         "a watchdog left armed stops the program during the next blocking move")
        self.assertEqual(arm.closed, [session])

    def test_a_fresh_program_runs_after_the_hold_so_the_next_move_works(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            session.free()
            session.hold()
            self.assertTrue(transport.running)
            self.assertFalse(transport.watchdog_armed)

    def test_a_fresh_program_that_comes_up_not_running_is_raised(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)

        def silent() -> None:
            transport.events.append("restart_control_program")
            transport.running = transport.guided = transport.watchdog_armed = False

        transport.restart_control_program = silent  # type: ignore[method-assign]
        with self.assertRaisesRegex(RobotConnectionError, "came up not running"), session:
            session.free()
        self.assertFalse(session.is_free)

    def test_a_watchdog_the_controller_did_not_take_frees_nothing(self) -> None:
        transport = _Transport()
        transport.watchdog_ok = False
        session, _ = _session(transport)
        with session:
            with self.assertRaisesRegex(RobotConnectionError, "watchdog"):
                session.free()
            self.assertFalse(session.is_free)
        self.assertNotIn("begin_hand_guiding", transport.events)

    def test_refused_hand_guiding_leaves_no_watchdog_behind(self) -> None:
        transport = _Transport()
        transport.guiding_ok = False
        session, _ = _session(transport)
        with session, self.assertRaisesRegex(RobotConnectionError, "refused hand guiding"):
            session.free()
        self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(transport), _FREE + _HOLD[1:])
        self.assertFalse(transport.watchdog_armed)

    def test_an_sdk_fault_while_freeing_is_named_and_leaves_no_watchdog(self) -> None:
        transport = _Transport()

        def begin() -> bool:
            transport.events.append("begin_hand_guiding")
            raise RuntimeError("RTDE control script is not running!")

        transport.begin_hand_guiding = begin  # type: ignore[method-assign]
        session, _ = _session(transport)
        with session, self.assertRaisesRegex(RobotConnectionError, "RTDE control script is not running"):
            session.free()
        self.assertIn("restart_control_program", transport.events)
        self.assertFalse(session.is_free)

    def test_free_and_hold_are_idempotent(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            session.hold()
            session.free()
            session.free()
            session.hold()
            session.hold()
        self.assertEqual(_controller_calls(transport), _FREE + _HOLD)

    def test_the_arm_is_freed_again_after_a_hold(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            session.free()
            session.hold()
            session.free()
        self.assertEqual(_controller_calls(transport), _FREE + _HOLD + _FREE + _HOLD)


class EveryWayOutHoldsTests(unittest.TestCase):
    def test_an_exception_in_the_block_holds_and_propagates(self) -> None:
        transport = _Transport()
        session, arm = _session(transport)
        with self.assertRaisesRegex(ValueError, "the board fell over"), session:
            session.free()
            raise ValueError("the board fell over")
        self.assertEqual(_controller_calls(transport)[-2:], _HOLD)
        self.assertFalse(transport.guided)
        self.assertEqual(arm.closed, [session])

    def test_ctrl_c_in_the_block_holds(self) -> None:
        transport = _Transport()
        session, arm = _session(transport)
        with self.assertRaises(KeyboardInterrupt), session:
            session.free()
            raise KeyboardInterrupt
        self.assertEqual(_controller_calls(transport)[-2:], _HOLD)
        self.assertFalse(transport.guided)
        self.assertEqual(arm.closed, [session])

    def test_ctrl_c_while_the_arm_settles_finds_the_watchdog_cleared_already(self) -> None:
        transport = _Transport(speeds=[[0.0] * 6, [0.2, 0, 0, 0, 0, 0]])  # still when freed, moving when held

        def interrupted(_s: float) -> None:
            raise KeyboardInterrupt

        session, _ = _session(transport, sleep=interrupted)
        with session:
            session.free()
            with self.assertRaises(KeyboardInterrupt):
                session.hold()
            self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(transport)[-2:], _HOLD)
        self.assertFalse(transport.watchdog_armed)

    def test_a_refused_end_of_hand_guiding_is_ended_by_the_restart(self) -> None:
        transport = _Transport()
        transport.end_guiding_ok = False
        session, _ = _session(transport)
        with session:
            session.free()
        self.assertEqual(_controller_calls(transport)[-2:], _HOLD)
        self.assertFalse(transport.guided)

    def test_an_end_of_hand_guiding_that_raises_is_ended_by_the_restart_all_the_same(self) -> None:
        """``hold`` restarts the program in a ``finally``: a controller error while hand guiding ends, and a Ctrl-C
        arriving during it, still stop the old program, and its hand guiding and its watchdog end with it (review of
        2026-09-25: only a refused end, which returns, was held)."""
        for raised in (RuntimeError("RTDE control interface stopped"), KeyboardInterrupt()):
            with self.subTest(raised=type(raised).__name__):
                transport = _Transport()
                session, _ = _session(transport)

                def end(error: BaseException = raised, calls: _Transport = transport) -> bool:
                    calls.events.append("end_hand_guiding")
                    raise error

                transport.end_hand_guiding = end  # type: ignore[method-assign]
                if isinstance(raised, KeyboardInterrupt):
                    with self.assertRaises(KeyboardInterrupt), session:
                        session.free()
                else:
                    with session:
                        session.free()
                self.assertEqual(_controller_calls(transport)[-2:], _HOLD)
                self.assertFalse(transport.guided)
                self.assertFalse(transport.watchdog_armed)

    def test_hold_waits_until_the_joints_stand_still(self) -> None:
        moving = [0.0, 0.3, 0.0, 0.0, 0.0, 0.0]
        transport = _Transport(speeds=[[0.0] * 6, moving, moving, [0.0, 0.001, 0.0, 0.0, 0.0, 0.0]])
        naps: list[float] = []
        session, _ = _session(transport, sleep=naps.append)
        with session:
            session.free()
            transport.events.clear()
            session.hold()
        # The restart comes first: waiting with an armed watchdog and nothing kicking it is a protective stop.
        self.assertEqual(transport.events, _HOLD + ["read_joint_speeds"] * 3)
        self.assertEqual(len(naps), 2)

    def test_a_failed_restart_is_raised_where_the_block_ended_normally(self) -> None:
        transport = _Transport()
        session, arm = _session(transport)
        with self.assertRaisesRegex(RobotConnectionError, "could not be restarted.*reconnects"), session:
            session.free()
            transport.restart_ok = False
        self.assertFalse(session.is_free)
        self.assertFalse(transport.is_connected, "the arm reads as disconnected, so nothing moves before a reconnect")
        self.assertEqual(arm.closed, [session])

    def test_a_failed_restart_does_not_replace_the_error_that_ended_the_block(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with self.assertLogs("PlannedFreedrive", level="ERROR") as logs, self.assertRaisesRegex(ValueError, "first"), \
                session:
            session.free()
            transport.restart_ok = False
            raise ValueError("first")
        self.assertIn("could not hold the arm cleanly", "\n".join(logs.output))

    def test_a_closed_connection_counts_as_held(self) -> None:
        """A disconnect stops the control program, and hand guiding ends with it."""
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            session.free()
            transport.is_connected = False
        self.assertFalse(session.is_free)
        self.assertEqual(_controller_calls(transport), [("arm_watchdog", WATCHDOG_MIN_HZ), "begin_hand_guiding"])


class WhatIsRefusedTests(unittest.TestCase):
    def test_free_outside_the_block_is_refused(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with self.assertRaisesRegex(RobotMotionRejected, "inside `with arm.freedrive"):
            session.free()
        with self.assertRaises(RobotMotionRejected):
            session.sample()
        self.assertEqual(transport.events, [])

    def test_a_session_is_used_once(self) -> None:
        session, _ = _session(_Transport())
        with session:
            pass
        with self.assertRaisesRegex(RobotMotionRejected, "left already"):
            session.__enter__()

    def test_an_arm_moving_by_itself_is_not_freed(self) -> None:
        transport = _Transport(speeds=[[0.0, 0.0, 0.05, 0.0, 0.0, 0.0]])
        session, _ = _session(transport)
        with session, self.assertRaisesRegex(RobotMotionRejected, "still moves by itself"):
            session.free()
        self.assertEqual(_controller_calls(transport), [])

    def test_a_controller_that_is_not_operational_is_not_freed(self) -> None:
        for status, error in ((_PROTECTIVE, ArmFault), (_IDLE, RobotMotionRejected)):
            with self.subTest(status=status.safety_mode.value, mode=status.robot_mode.value):
                transport = _Transport()
                session, arm = _session(transport)
                arm.status = status
                with session, self.assertRaises(error):
                    session.free()
                self.assertEqual(_controller_calls(transport), [])


class SampleTests(unittest.TestCase):
    def test_a_sample_while_free_kicks_the_watchdog_first(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            held = session.sample()
            self.assertNotIn("kick_watchdog", transport.events, "a held arm has no watchdog to kick")
            session.free()
            transport.events.clear()
            sample = session.sample()
            self.assertEqual(transport.events[0], "kick_watchdog")
        for reading in (held, sample):
            self.assertIsInstance(reading, FreedriveSample)
        self.assertEqual(sample.joints_rad, (0.1, -1.2, 1.3, -0.4, 1.5, 0.2))
        self.assertEqual(sample.tcp_xyz_mm, (400.0, -120.0, 300.0))
        np.testing.assert_allclose(sample.tcp_rotvec_rad, (0.0, math.pi, 0.0), atol=1e-12)
        self.assertEqual(sample.peak_joint_speed_rad_s, 0.0)
        self.assertGreater(sample.t_s, held.t_s)

    def test_a_program_that_ended_while_free_is_raised_after_the_session_marked_itself_held(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with session:
            session.free()
            transport.running = False  # the watchdog stopped it, or a stop on the pendant did
            transport.guided = False
            transport.events.clear()
            with self.assertRaisesRegex(RobotConnectionError, "no longer hand-guided: the control program ended"):
                session.sample()
            self.assertFalse(session.is_free)
            self.assertIn("restart_control_program", transport.events,
                          "control is given back where no stop forbids it")
            transport.events.clear()
        self.assertEqual(_controller_calls(transport), [], "the exit finds the arm held already")

    def test_a_protective_stop_while_free_is_raised_as_a_fault(self) -> None:
        transport = _Transport()
        session, arm = _session(transport)
        with session:
            session.free()
            transport.protective = True
            transport.running = False
            arm.status = _PROTECTIVE
            transport.events.clear()
            with self.assertRaisesRegex(ArmFault, "Clear the stop at the pendant") as caught:
                session.sample()
            self.assertFalse(session.is_free)
        self.assertNotIn("restart_control_program", transport.events, "a stopped controller starts no program")
        self.assertIn(f"not called for {1.0 / WATCHDOG_MIN_HZ:.2f} s", str(caught.exception),
                      "a watchdog trip reads as a protective stop, so the stop names it")
        self.assertEqual("protective_stop", caught.exception.kind.value)

    def test_holding_a_stopped_arm_says_so_at_once_rather_than_waiting_for_a_program(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)
        with self.assertRaisesRegex(ArmFault, "controller is stopped"), session:
            session.free()
            transport.protective = True  # nobody sampled, so the session still believes the arm is free
            transport.running = False
        self.assertNotIn("restart_control_program", transport.events)
        self.assertFalse(session.is_free)

    def test_a_read_that_fails_while_free_holds_the_arm(self) -> None:
        transport = _Transport()
        session, _ = _session(transport)

        def broken() -> list[float]:
            raise RuntimeError("receive interface lost")

        with session:
            session.free()
            transport.read_joints = broken  # type: ignore[method-assign]
            with self.assertRaisesRegex(RobotConnectionError, "receive interface lost"):
                session.sample()
            self.assertFalse(session.is_free)
            self.assertFalse(transport.guided, "the restart ends the hand guiding the program was still in")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
