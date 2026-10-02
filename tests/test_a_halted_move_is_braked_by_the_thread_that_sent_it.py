"""With ``robot.ur.brake_on_halt`` on, a halted move is braked under control by the thread that sent it.

The ur_rtde docstring of ``moveJ``: "If async is true it is possible to stop a move command using either the stopJ or stopL
function. Default is false, this means the function will block until the movement has completed." A synchronous move
cannot be stopped from another thread (the control script runs it in its main loop, and a ``stopJ`` waits behind it), so
with the brake on every move is sent asynchronously and the thread that sent it polls it every 8 ms: the async operation
register, the stop flags, the program state. A halt is a flag that poll reads, and the same thread then brakes with
``stopJ`` for a joint move or ``stopL`` for a line, each decelerating along the line it was running, at max(2.0, the
move's own acceleration). ``halt()`` itself sends nothing from the thread that calls it.

``True`` comes only at the target. The operation register is read only once the operation this move started shows in
it (its id changed), so a register that still shows the move before is no arrival; a finished operation is an arrival only
where the joints stand within 2e-3 rad of the target (the TCP within 1 mm and 2e-3 rad for a line), waited for up to
half a second for a servo trailing its set point. A move cut short, a protective stop, a program that ended: ``False``,
never an early ``True``, which would let the next waypoint's move cut the leg short. ``last_stop_cause`` says which stop
ended it, the protective and emergency stops asked before the program, which a protective stop ends as well.

Whatever else ends the watch (a read that raises, a garbled register, a ``KeyboardInterrupt`` in the moving thread), the
move would run on in the controller with nobody watching it, so the moving thread stops it first and only then says
nothing is in flight. The latch's check and the mark of a move in flight are one step right before the send, so a halt
either refuses the send or finds the move in flight. The record the console polls says what became of that move
(``HaltState.brake``): pending while it runs or brakes, then braked, unconfirmed or ran out.

The controller here is ``tests._halt_fakes.SimulatedUr`` on a virtual clock, except where the threads themselves are the
question.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any

import numpy as np

from src.geometry import Frame, Pose
from src.robot.core import JointPositions, MotionStatus, RobotMotionRejected
from src.robot.drivers.ur.connection import MoveEnd, URConnection
from tests._halt_fakes import SimulatedUr, VirtualClock, ur_arm_on

_HERE = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
_FAR = [0.6, -1.2, 1.3, -0.4, 1.5, 0.2]  # 0.6 rad on the shoulder: 1.2 s at 0.5 rad/s
_REASON = "the operator pressed halt now"


def _conn(sim: SimulatedUr, clock: VirtualClock | None, *, acc: float = 0.8, brake: bool = True) -> URConnection:
    kwargs: dict[str, Any] = {} if clock is None else {"clock": clock, "sleep": clock.sleep}
    conn = URConnection("127.0.0.1", vel=0.5, acc=acc, brake_on_halt=brake, **kwargs)
    return sim.attach(conn)


def _virtual(**sim_kwargs: Any) -> tuple[VirtualClock, SimulatedUr]:
    clock = VirtualClock()
    return clock, SimulatedUr(_HERE, clock=clock, wait=clock.sleep, **sim_kwargs)


class TheBrakeTests(unittest.TestCase):
    def test_a_halt_mid_movej_brakes_it_with_stopj_and_the_arm_stands_on_its_line(self) -> None:
        """Red before: there was no brake; a synchronous moveJ ran to its end whatever was pressed."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.BRAKED, conn.last_move_end)
        self.assertEqual([[_FAR, 0.5, 0.8, True]], [call[3] for call in sim.named("moveJ")])
        self.assertEqual([[2.0]], [call[3] for call in sim.named("stopJ")], "max(2.0, a) for a = 0.8")
        self.assertEqual([], sim.named("stopL"))
        travelled = (sim.q[0] - _HERE[0]) / (_FAR[0] - _HERE[0])
        self.assertGreater(travelled, 0.2)
        self.assertLess(travelled, 0.5, "the arm was not braked: it went on towards the target")
        np.testing.assert_allclose(sim.q[1:], _HERE[1:], err_msg="the stop point left the joint line")
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((_REASON, True, True, "braked"), (state.reason, state.in_motion, state.braked, state.brake))
        assert state.brake_s is not None
        self.assertGreater(state.brake_s, 0.0)
        self.assertLess(state.brake_s, 1.0)

    def test_the_deceleration_is_the_moves_own_where_that_is_stronger(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock, acc=3.0)
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        self.assertFalse(conn.moveJ(_FAR))
        self.assertEqual([[3.0]], [call[3] for call in sim.named("stopJ")])

    def test_a_halt_mid_movel_brakes_it_with_stopl(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        target = [0.4, 0.4, 0.3, 0.0, 3.1416, 0.0]  # 0.3 m along base y at 0.5 m/s
        clock.at(0.2, lambda: conn.request_halt(_REASON))
        self.assertFalse(conn.moveL(target, 0.5, 0.8))
        self.assertIs(MoveEnd.BRAKED, conn.last_move_end)
        self.assertEqual([[2.0]], [call[3] for call in sim.named("stopL")])
        self.assertEqual([], sim.named("stopJ"))
        self.assertLess(sim.tcp[1], 0.4)

    def test_after_a_brake_nothing_more_is_sent(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        conn.moveJ(_FAR)
        self.assertFalse(conn.moveJ(_HERE))
        self.assertIs(MoveEnd.REFUSED_HALTED, conn.last_move_end)
        self.assertEqual(1, len(sim.named("moveJ")))

    def test_a_clear_during_the_brake_does_not_bring_the_halt_back(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        clock.at(0.31, conn.clear_halt)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIsNone(conn.halt_state())

    def test_a_brake_the_arm_is_not_seen_to_finish_is_unconfirmed(self) -> None:
        """The joints never read still after the stop: the latch says not braked, and the arm says press the e-stop."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        sim.getActualQd = lambda: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.BRAKE_UNCONFIRMED, conn.last_move_end)
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, None, "unconfirmed"), (state.in_motion, state.braked, state.brake_s, state.brake))
        self.assertIn("emergency stop", state.render())

    def test_a_brake_that_raises_is_unconfirmed_and_says_so(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))

        def lost(*_a: Any) -> None:
            raise RuntimeError("RTDE control script is not running!")

        sim.stopJ = lost  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.BRAKE_UNCONFIRMED, conn.last_move_end)
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, "unconfirmed"), (state.in_motion, state.braked, state.brake),
                         "the record the console polls read a failed brake as one still in progress")

    def test_a_brake_whose_stillness_cannot_be_read_is_unconfirmed(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt(_REASON))

        def unreadable() -> list[float]:
            raise RuntimeError("RTDE receive timed out")

        sim.getActualQd = unreadable  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.BRAKE_UNCONFIRMED, conn.last_move_end)

    def test_a_new_halt_during_a_brake_after_a_clear_keeps_its_own_record(self) -> None:
        """Halted, cleared and halted again while one brake runs: the record the latch keeps is the second one, braked,
        its time counted from the second press. The first record does not come back."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: conn.request_halt("the first press"))
        clock.at(0.31, conn.clear_halt)
        clock.at(0.32, lambda: conn.request_halt("the second press"))
        self.assertFalse(conn.moveJ(_FAR))
        state = conn.halt_state()
        assert state is not None
        self.assertEqual(("the second press", True, True, "braked"),
                         (state.reason, state.in_motion, state.braked, state.brake))
        assert state.brake_s is not None
        # The brake from 0.5 rad/s at 2.0 rad/s^2 takes 0.25 s from about 0.30 s: still at about 0.55 s.
        self.assertGreater(state.brake_s, 0.15)
        self.assertLess(state.brake_s, 0.25, "the time was counted from the first press")

    def test_refuse_if_halted_is_the_one_door_every_send_meets(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        self.assertFalse(conn.refuse_if_halted("moveJ"))
        self.assertIs(MoveEnd.NONE, conn.last_move_end)
        conn.request_halt(_REASON)
        self.assertTrue(conn.halted)
        self.assertTrue(conn.refuse_if_halted("moveJ"))
        self.assertIs(MoveEnd.REFUSED_HALTED, conn.last_move_end)
        conn.clear_halt()
        conn.clear_halt()  # a clear with nothing latched is no fault
        self.assertFalse(conn.halted)
        self.assertEqual([], sim.calls)


class TheThreadsTests(unittest.TestCase):
    def test_the_brake_is_sent_by_the_thread_that_moved_the_arm_and_halt_sends_nothing(self) -> None:
        """Red before: the only stop there was came from the calling thread, and did nothing to a synchronous move."""
        sim = SimulatedUr(_HERE)
        conn = _conn(sim, None)
        answered: list[bool] = []
        mover = threading.Thread(target=lambda: answered.append(conn.moveJ(_FAR)), name="moving")
        started = time.monotonic()
        mover.start()
        time.sleep(0.15)
        conn.request_halt(_REASON)
        mover.join(timeout=5.0)
        self.assertFalse(mover.is_alive())
        self.assertEqual([False], answered)
        self.assertLess(time.monotonic() - started, 1.0, "the move ran to its end: nothing braked it")
        stops = sim.named("stopJ")
        self.assertEqual(1, len(stops))
        self.assertEqual(mover.ident, stops[0][0], "the brake was not sent by the thread that moved the arm")
        self.assertNotIn(threading.get_ident(), {call[0] for call in sim.calls}, "request_halt reached the controller")


class TheArrivalTests(unittest.TestCase):
    def test_a_move_nobody_halts_answers_true_at_its_target(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        self.assertTrue(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ARRIVED, conn.last_move_end)
        np.testing.assert_allclose(sim.q, _FAR, atol=2e-3)
        self.assertGreaterEqual(clock.t, 1.2, "True came before the move could have arrived")
        self.assertEqual([], sim.named("stopJ"))

    def test_a_register_that_still_shows_the_move_before_is_no_arrival(self) -> None:
        """The operation id changes only after the reporting delay: until then the register reads "not running"."""
        clock, sim = _virtual(report_delay_s=0.05)
        conn = _conn(sim, clock)
        self.assertTrue(conn.moveJ([0.1, *_HERE[1:]]))
        start = clock.t
        self.assertTrue(conn.moveJ(_FAR))
        self.assertGreaterEqual(clock.t - start, 1.0, "an early True: the register still showed the move before")
        np.testing.assert_allclose(sim.q, _FAR, atol=2e-3)

    def test_a_move_that_ends_short_of_its_target_is_not_an_arrival(self) -> None:
        clock, sim = _virtual()
        sim.kill_move_at = 0.5
        conn = _conn(sim, clock)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ENDED_SHORT, conn.last_move_end)
        self.assertIsNone(conn.halt_state())

    def test_a_servo_trailing_its_set_point_is_waited_for(self) -> None:
        clock, sim = _virtual(lag=0.01, lag_s=0.1)
        conn = _conn(sim, clock)
        self.assertTrue(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ARRIVED, conn.last_move_end)

    def test_a_servo_that_never_arrives_is_not_an_arrival(self) -> None:
        clock, sim = _virtual(lag=0.01, lag_s=5.0)
        conn = _conn(sim, clock)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ENDED_SHORT, conn.last_move_end)

    def test_a_line_arrives_only_with_its_tcp_at_the_target(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        target = [0.4, 0.2, 0.3, 0.0, 3.1416, 0.0]
        self.assertTrue(conn.moveL(target, 0.5, 0.8))
        np.testing.assert_allclose(sim.tcp, target, atol=1e-3)

    def test_a_protective_stop_mid_move_is_never_an_arrival(self) -> None:
        clock, sim = _virtual()
        sim.protective_at = 0.3
        conn = _conn(sim, clock)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.STOPPED, conn.last_move_end)
        self.assertEqual("protective", conn.last_stop_cause)
        self.assertEqual([], sim.named("stopJ"), "a stopped controller is not sent a brake")

    def test_a_program_that_ends_mid_move_is_never_an_arrival(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        clock.at(0.3, lambda: setattr(sim, "program_running", False))
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.STOPPED, conn.last_move_end)
        self.assertEqual("program", conn.last_stop_cause)

    def test_a_protective_stop_that_ends_the_program_too_is_said_as_the_protective_stop(self) -> None:
        """What a probe on a real controller has to tell apart: a protective stop stops the control program as well, and
        the stop is the cause. Red before: the program was asked first, so a protective stop read as a program end."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)

        def protective_stop() -> None:
            sim.protective = True
            sim.program_running = False

        clock.at(0.3, protective_stop)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.STOPPED, conn.last_move_end)
        self.assertEqual("protective", conn.last_stop_cause)

    def test_a_stop_during_the_settle_ends_the_move_stopped(self) -> None:
        """The operation finished short of the target and the controller stops while the arm is waited for: STOPPED, at
        once, not an arrival and not a move that ended short after the settle time."""
        for flag, cause in (("protective", "protective"), ("emergency", "emergency")):
            with self.subTest(stop=flag):
                clock, sim = _virtual(lag=0.01, lag_s=5.0)  # finished at 1.2 s, 0.01 rad short for 5 s
                conn = _conn(sim, clock)
                clock.at(1.3, lambda flag=flag: setattr(sim, flag, True))  # type: ignore[misc]
                self.assertFalse(conn.moveJ(_FAR))
                self.assertIs(MoveEnd.STOPPED, conn.last_move_end)
                self.assertEqual(cause, conn.last_stop_cause)
                self.assertLess(clock.t, 1.2 + 0.5, "the settle ran out its time instead of reading the stop")

    def test_a_line_at_its_position_but_turned_off_its_orientation_is_not_an_arrival(self) -> None:
        """The orientation half of a line's arrival: the TCP at the target within 1 mm, turned 0.01 rad off it."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        target = [0.4, 0.2, 0.3, 0.0, 3.1416, 0.0]
        actual = sim.getActualTCPPose

        def turned() -> list[float]:
            pose = actual()
            return [*pose[:3], pose[3], pose[4], pose[5] + 0.01]

        sim.getActualTCPPose = turned  # type: ignore[method-assign]
        self.assertFalse(conn.moveL(target, 0.5, 0.8))
        self.assertIs(MoveEnd.ENDED_SHORT, conn.last_move_end)
        np.testing.assert_allclose(sim.tcp[:3], target[:3], atol=1e-3)

    def test_an_operation_the_controller_never_shows_is_stopped_and_refused(self) -> None:
        clock, sim = _virtual(report_delay_s=10.0)
        conn = _conn(sim, clock)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.NOT_STARTED, conn.last_move_end)
        self.assertEqual([[2.0]], [call[3] for call in sim.named("stopJ")], "a move nobody can watch is stopped")

    def test_a_send_the_controller_refuses_is_refused(self) -> None:
        clock, sim = _virtual()
        sim.refuse_sends = True
        conn = _conn(sim, clock)
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.NOT_STARTED, conn.last_move_end)
        self.assertEqual(1, len(sim.named("getAsyncOperationProgressEx")), "a refused send was watched: only the "
                         "register read taken before the send is expected")
        self.assertEqual([], sim.named("stopJ"))

    def test_a_line_with_no_turn_arrives_too(self) -> None:
        clock, sim = _virtual(tcp=(0.4, 0.1, 0.3, 0.0, 0.0, 0.0))
        conn = _conn(sim, clock)
        self.assertTrue(conn.moveL([0.4, 0.2, 0.3, 0.0, 0.0, 0.0], 0.5, 0.8))

    def test_a_reading_that_is_no_configuration_is_no_arrival(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        sim.getActualQ = lambda: [0.0, 0.0, 0.0]  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ENDED_SHORT, conn.last_move_end)
        clock2, sim2 = _virtual()
        conn2 = _conn(sim2, clock2)
        sim2.getActualTCPPose = lambda: [0.0, 0.0, 0.0]  # type: ignore[method-assign]
        self.assertFalse(conn2.moveL([0.4, 0.2, 0.3, 0.0, 3.1416, 0.0], 0.5, 0.8))
        self.assertIs(MoveEnd.ENDED_SHORT, conn2.last_move_end)

    def test_a_stop_that_fails_after_a_failed_read_still_raises_the_read(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        original = sim.getAsyncOperationProgressEx
        reads = {"n": 0}

        def failing() -> Any:
            reads["n"] += 1
            if reads["n"] > 3:
                raise OSError("connection reset")
            return original()

        def lost(*_a: Any) -> None:
            raise RuntimeError("RTDE control script is not running!")

        sim.getAsyncOperationProgressEx = failing  # type: ignore[method-assign]
        sim.stopJ = lost  # type: ignore[method-assign]
        with self.assertRaises(OSError):
            conn.moveJ(_FAR)

    def test_a_read_that_fails_mid_move_stops_the_arm_and_raises(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        reads = {"n": 0}
        original = sim.getAsyncOperationProgressEx

        def failing() -> Any:
            reads["n"] += 1
            if reads["n"] > 10:
                raise RuntimeError("RTDE receive timed out")
            return original()

        sim.getAsyncOperationProgressEx = failing  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            conn.moveJ(_FAR)
        self.assertEqual(1, len(sim.named("stopJ")), "the move in flight was left running")


class WhateverEndsTheWatchStopsTheMoveTests(unittest.TestCase):
    """The watched move runs in the controller until somebody stops it: whatever ends the watch, the thread that sent the
    move stops it before the fault goes on, and only then says nothing is in flight."""

    def _recording_stops(self, sim: SimulatedUr, conn: URConnection) -> list[bool]:
        """Whether the connection still said a move was in flight at each stopJ."""
        seen: list[bool] = []
        stop = sim.stopJ

        def recording(*args: Any) -> None:
            seen.append(conn.in_motion)
            stop(*args)

        sim.stopJ = recording  # type: ignore[method-assign]
        return seen

    def test_an_interrupt_mid_move_stops_it_from_the_moving_thread_and_raises_on(self) -> None:
        """Ctrl-C in a CLI arrives in the moving main thread. Red before: it went on with the move still running in the
        controller and nobody watching it, in_motion read False, and a halt then braked nothing."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        in_flight_at_the_stop = self._recording_stops(sim, conn)

        def interrupt() -> None:
            raise KeyboardInterrupt

        clock.at(0.3, interrupt)
        with self.assertRaises(KeyboardInterrupt):
            conn.moveJ(_FAR)
        stops = sim.named("stopJ")
        self.assertEqual(1, len(stops), "the move in flight was left running")
        self.assertEqual(threading.get_ident(), stops[0][0])
        self.assertEqual([True], in_flight_at_the_stop, "the move read as not in flight before its stop went out")
        self.assertFalse(conn.in_motion)
        clock.sleep(2.0)
        sim.update()
        self.assertLess(sim.q[0], _FAR[0] - 0.1, "the move ran on to its target with nobody watching it")

    def test_any_fault_of_a_read_mid_move_stops_it(self) -> None:
        """Red before: only RuntimeError and OSError stopped the move; a garbled register (ValueError) left it running."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        reads = {"n": 0}
        original = sim.getAsyncOperationProgressEx

        def garbled() -> Any:
            reads["n"] += 1
            if reads["n"] > 5:
                raise ValueError("garbled register")
            return original()

        sim.getAsyncOperationProgressEx = garbled  # type: ignore[method-assign]
        with self.assertRaises(ValueError):
            conn.moveJ(_FAR)
        self.assertEqual(1, len(sim.named("stopJ")))
        sim.update()
        self.assertFalse(sim._running(), "the controller still runs the move nobody watches")

    def test_a_send_that_raises_is_stopped_where_it_may_have_started(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        send = sim.moveJ

        def sent_then_raised(*args: Any) -> bool:
            send(*args)
            raise RuntimeError("RTDE control script is not running!")

        sim.moveJ = sent_then_raised  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            conn.moveJ(_FAR)
        self.assertEqual(1, len(sim.named("stopJ")))
        self.assertFalse(conn.in_motion)

    def test_a_fault_after_the_connection_was_dropped_raises_on(self) -> None:
        """A disconnect from another thread mid-move: nothing is left to stop through, and the fault itself goes on."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        reads = {"n": 0}
        original = sim.getAsyncOperationProgressEx

        def dropped() -> Any:
            reads["n"] += 1
            if reads["n"] > 3:
                conn._ctrl = None
                raise RuntimeError("RTDE control script is not running!")
            return original()

        sim.getAsyncOperationProgressEx = dropped  # type: ignore[method-assign]
        with self.assertRaises(RuntimeError):
            conn.moveJ(_FAR)
        self.assertEqual([], sim.named("stopJ"))
        self.assertFalse(conn.in_motion)

    def test_an_interrupt_inside_the_brake_leaves_the_record_unconfirmed(self) -> None:
        """Halted, and the brake interrupted before the arm was seen to stand: stopped again where it can be, and the
        record the console polls says unconfirmed, never that the move ran out."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)

        def interrupt() -> None:
            raise KeyboardInterrupt

        clock.at(0.3, lambda: conn.request_halt(_REASON))
        clock.at(0.35, interrupt)  # inside the brake: the stopJ waits for the arm to stand
        with self.assertRaises(KeyboardInterrupt):
            conn.moveJ(_FAR)
        self.assertEqual(2, len(sim.named("stopJ")), "the brake, then the stop of a move nobody watches any more")
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, "unconfirmed"), (state.in_motion, state.braked, state.brake))
        self.assertFalse(conn.in_motion)

    def test_a_halt_after_an_interrupt_finds_nothing_in_flight(self) -> None:
        clock, sim = _virtual()
        conn = _conn(sim, clock)

        def interrupt() -> None:
            raise KeyboardInterrupt

        clock.at(0.3, interrupt)
        with self.assertRaises(KeyboardInterrupt):
            conn.moveJ(_FAR)
        state = conn.request_halt(_REASON)
        self.assertEqual((False, "none"), (state.in_motion, state.brake))


class TheLatchAndTheSendAreOneStepTests(unittest.TestCase):
    """A halt that lands between the latch's check and the send either refuses the send or finds the move in flight:
    never a move sent that the record calls not in flight."""

    def test_brakes_on_a_halt_during_the_read_before_the_send_sends_nothing(self) -> None:
        """Red before: the latch was read before the register, the halt landed during the register read, and the move
        was sent and braked while the record said no move was in flight."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        original = sim.getAsyncOperationProgressEx

        def read_then_halt() -> Any:
            status = original()
            if conn.halt_state() is None:
                conn.request_halt(_REASON)
            return status

        sim.getAsyncOperationProgressEx = read_then_halt  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.REFUSED_HALTED, conn.last_move_end)
        self.assertEqual([], sim.named("moveJ") + sim.named("stopJ"))
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((False, False, "none"), (state.in_motion, state.braked, state.brake))

    def test_brakes_off_a_halt_right_after_the_latch_was_asked_sends_nothing(self) -> None:
        """Red before: with brakes off a halt landing after the check went out with the move."""
        clock = VirtualClock()
        sim = SimulatedUr(_HERE, clock=clock, wait=clock.sleep)
        conn = sim.attach(URConnection("127.0.0.1", vel=0.5, acc=0.8, clock=clock, sleep=clock.sleep))
        asked = conn.refuse_if_halted

        def asked_then_halted(what: str) -> bool:
            refused = asked(what)
            conn.request_halt(_REASON)
            return refused

        conn.refuse_if_halted = asked_then_halted  # type: ignore[method-assign]
        self.assertFalse(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.REFUSED_HALTED, conn.last_move_end)
        self.assertEqual([], sim.named("moveJ"))

    def test_a_halt_during_the_settle_records_that_the_move_ran_out(self) -> None:
        """The operation had finished and the arm was being waited for at its target: nothing was braked, and the record
        says the move ended without a brake rather than staying a brake in progress for good."""
        clock, sim = _virtual(lag=0.01, lag_s=0.3)
        conn = _conn(sim, clock)
        clock.at(1.25, lambda: conn.request_halt(_REASON))
        self.assertTrue(conn.moveJ(_FAR))
        self.assertIs(MoveEnd.ARRIVED, conn.last_move_end)
        state = conn.halt_state()
        assert state is not None
        self.assertEqual((True, False, None, "ran_out"), (state.in_motion, state.braked, state.brake_s, state.brake))
        self.assertEqual([], sim.named("stopJ"))

    def test_a_halt_while_a_brake_is_awaited_reads_pending_from_another_thread(self) -> None:
        """What the console polls while the moving thread brakes: pending, then braked once the arm stands."""
        clock, sim = _virtual()
        conn = _conn(sim, clock)
        polled: list[str] = []
        clock.at(0.3, lambda: conn.request_halt(_REASON))
        clock.at(0.4, lambda: polled.append(getattr(conn.halt_state(), "brake", "?")))
        conn.moveJ(_FAR)
        self.assertEqual(["pending"], polled)
        state = conn.halt_state()
        assert state is not None
        self.assertEqual("braked", state.brake)


class TheArmAnswersABrakedMoveCancelledTests(unittest.TestCase):
    """The arm's verbs say CANCELLED for a move the halt braked, where they said CONTROLLER_REJECTED for a moveJ failure."""

    def _arm(self, planner: str = "ik", *, brake: bool = True, halt_at: float = 0.3) -> tuple[Any, VirtualClock,
                                                                                             SimulatedUr]:
        clock, sim = _virtual()
        arm = ur_arm_on(sim, planner, brake_on_halt=brake)
        arm._conn._clock, arm._conn._sleep = clock, clock.sleep
        clock.at(halt_at, lambda: arm.halt(_REASON))
        return arm, clock, sim

    def test_a_braked_joint_move_is_cancelled_and_says_so(self) -> None:
        arm, _clock, sim = self._arm()
        result = arm.move_to_joints(JointPositions(_FAR))
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("braked", result.message)
        self.assertIn(_REASON, result.message)
        self.assertEqual(1, len(sim.named("stopJ")))

    def test_move_joint_raises_the_cancelled_result(self) -> None:
        arm, _clock, _sim = self._arm()
        with self.assertRaises(RobotMotionRejected) as raised:
            arm.move_joint(JointPositions(_FAR))
        assert raised.exception.result is not None
        self.assertIs(MotionStatus.CANCELLED, raised.exception.result.status)

    def test_a_braked_ik_move_is_cancelled(self) -> None:
        arm, _clock, sim = self._arm(halt_at=0.1)  # the controller's IK answer is 0.14 rad away: 0.23 s at 0.6 rad/s
        pose = Pose(position_mm=np.array([450.0, -120.0, 120.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE, label="far")
        result = arm.move(pose)
        self.assertIs(MotionStatus.CANCELLED, result.status, result.message)
        self.assertIn(_REASON, result.message)
        self.assertEqual(1, len(sim.named("stopJ")))

    def test_a_braked_checked_line_is_cancelled(self) -> None:
        arm, _clock, sim = self._arm("curobo")
        arm._judge_linear_move = lambda pose, *, command: None
        pose = Pose(position_mm=np.array([400.0, 400.0, 168.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE, label="line")
        with arm.without_camera_world("the halt tests wire no camera world"):
            result = arm.move(pose, linear=True)
        self.assertIs(MotionStatus.CANCELLED, result.status, result.message)
        self.assertIn("braked", result.message)
        self.assertEqual(1, len(sim.named("stopL")))

    def test_a_braked_move_linear_raises_the_cancelled_result(self) -> None:
        arm, _clock, _sim = self._arm()
        pose = Pose(position_mm=np.array([400.0, 400.0, 168.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE, label="line")
        with self.assertRaises(RobotMotionRejected) as raised:
            arm.move_linear(pose)
        assert raised.exception.result is not None
        self.assertIs(MotionStatus.CANCELLED, raised.exception.result.status)

    def test_a_brake_nobody_saw_end_tells_the_person_to_press_the_emergency_stop(self) -> None:
        arm, _clock, sim = self._arm()
        sim.getActualQd = lambda: [0.3, 0.0, 0.0, 0.0, 0.0, 0.0]  # type: ignore[method-assign]
        result = arm.move_to_joints(JointPositions(_FAR))
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("emergency stop", result.message)

    def test_a_halt_that_lands_after_the_gate_is_refused_unsent_and_said_so(self) -> None:
        """Halted between the arm's own gate and the send: the connection refuses, and the verb says nothing was sent."""
        arm, _clock, sim = self._arm()
        arm.halt(_REASON)
        with self.assertRaises(RobotMotionRejected) as raised:
            arm._drive_joints(JointPositions(_FAR))  # the drive after the gate, as a judged move reaches it
        assert raised.exception.result is not None
        self.assertIs(MotionStatus.CANCELLED, raised.exception.result.status)
        self.assertIn("was not sent", raised.exception.result.message)
        pose = Pose(position_mm=np.array([400.0, 400.0, 168.0]), quaternion_xyzw=np.array([0.0, 1.0, 0.0, 0.0]),
                    frame=Frame.BASE, label="line")
        arm._judge_linear_move = lambda pose, *, command: None
        result = arm._drive_checked_line(pose)
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("moveL was not sent", result.message)
        self.assertEqual([], sim.named("moveJ") + sim.named("moveL"))

    def test_a_halt_cleared_before_the_answer_still_reads_cancelled(self) -> None:
        arm, clock, _sim = self._arm()
        clock.at(0.6, arm.clear_halt)  # inside the brake, before the verb answers
        result = arm.move_to_joints(JointPositions(_FAR))
        self.assertIs(MotionStatus.CANCELLED, result.status)
        self.assertIn("cleared since", result.message)

    def test_with_brakes_off_the_move_in_flight_runs_out_and_the_next_verb_is_cancelled(self) -> None:
        arm, _clock, sim = self._arm(brake=False)
        self.assertIs(MotionStatus.EXECUTED, arm.move_to_joints(JointPositions(_FAR)).status)
        np.testing.assert_allclose(sim.q, _FAR)
        self.assertIs(MotionStatus.CANCELLED, arm.move_to_joints(JointPositions(_HERE)).status)
        self.assertEqual(1, len(sim.named("moveJ")))
        self.assertEqual([], sim.named("stopJ"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
