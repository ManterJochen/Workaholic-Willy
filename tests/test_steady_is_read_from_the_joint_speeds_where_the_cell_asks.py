"""A UR's steady gate reads the joint speeds where the cell asks (``robot.safety.dwell.steady_signal: joint_speeds``, the
owner, 2026-10-09), and the controller's ``isSteady`` as always where it does not.

``isSteady`` is a script command: about 33 ms a call on URSim CB3 (measured), polled every 20 ms, and the gate before a
line cost 0.54 to 0.61 s on the cell after every motion. The joint speeds are a local read of what the receive interface
last got: steady is every joint at most 0.01 rad/s, the brake's own stillness, on three reads in a row 8 ms apart.

What this file pins: three quiet reads are steady, 16 ms; a read that moves starts the count again; a wait that runs out
times out with ``False`` and the warning it always had; a wait of 0 asks once, three reads, and the first read that moves
is a no; only a sample the controller sent since the read before counts, so a stream that stopped is never an arm that
stands; speeds that cannot be read leave the answer to ``isSteady``; ``isSteady`` stays the default and the joint speeds
are not read then; a signal of another name is refused; and the key loads.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from src.robot.drivers.ur.connection import URConnection
from tests._halt_fakes import VirtualClock

_STILL = [0.0, 0.001, -0.002, 0.0, 0.0, 0.009]
_MOVING = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]


def _conn(signal: str, speeds: "list[list[float]] | None" = None) -> tuple[URConnection, VirtualClock]:
    """A connection on a virtual clock whose receive interface gets a new sample, stamped by the clock, every read."""
    clock = VirtualClock()
    conn = URConnection("127.0.0.1", steady_signal=signal, clock=clock, sleep=clock.sleep)
    conn._ctrl = MagicMock()  # noqa: SLF001
    conn._ctrl.isSteady.return_value = True  # noqa: SLF001
    conn._recv = MagicMock()  # noqa: SLF001
    conn._recv.getTimestamp.side_effect = lambda: 1000.0 + clock.t  # noqa: SLF001
    if speeds is not None:
        conn._recv.getActualQd.side_effect = speeds  # noqa: SLF001
    return conn, clock


class SteadyFromTheJointSpeedsTests(unittest.TestCase):
    def test_three_quiet_reads_8_ms_apart_are_steady_with_no_round_trip(self) -> None:
        """⭐ Red before: the gate asked ``isSteady``, 33 ms a call, until it said so."""
        conn, clock = _conn("joint_speeds", [_STILL] * 3)
        self.assertTrue(conn.wait_until_steady(5.0))
        self.assertAlmostEqual(0.016, clock.t)
        self.assertEqual(3, conn._recv.getActualQd.call_count)  # noqa: SLF001
        conn._ctrl.isSteady.assert_not_called()  # noqa: SLF001

    def test_a_read_that_moves_starts_the_count_again(self) -> None:
        conn, clock = _conn("joint_speeds", [_STILL, _STILL, _MOVING, _STILL, _STILL, _STILL])
        self.assertTrue(conn.wait_until_steady(5.0))
        self.assertEqual(6, conn._recv.getActualQd.call_count)  # noqa: SLF001
        self.assertAlmostEqual(0.040, clock.t)

    def test_a_speed_just_over_the_brakes_stillness_is_motion(self) -> None:
        conn, _clock = _conn("joint_speeds", [[0.0, 0.0, 0.0101, 0.0, 0.0, 0.0]] * 3 + [_STILL] * 3)
        self.assertTrue(conn.wait_until_steady(5.0))
        self.assertEqual(6, conn._recv.getActualQd.call_count)  # noqa: SLF001

    def test_a_wait_that_runs_out_times_out_with_false_and_its_warning(self) -> None:
        conn, clock = _conn("joint_speeds")
        conn._recv.getActualQd.return_value = _MOVING  # noqa: SLF001
        with self.assertLogs("URConnection", level="WARNING") as said:
            self.assertFalse(conn.wait_until_steady(0.1))
        self.assertIn("wait_until_steady timed out after 0.10s", "\n".join(said.output))
        self.assertGreaterEqual(clock.t, 0.1)
        self.assertLess(clock.t, 0.1 + 0.009)
        conn._ctrl.isSteady.assert_not_called()  # noqa: SLF001

    def test_a_wait_of_0_asks_once_in_three_reads(self) -> None:
        conn, clock = _conn("joint_speeds", [_STILL] * 3)
        self.assertTrue(conn.wait_until_steady(0.0))
        self.assertAlmostEqual(0.016, clock.t)
        conn, clock = _conn("joint_speeds", [_STILL, _MOVING, _STILL])
        self.assertFalse(conn.wait_until_steady(0.0))
        self.assertEqual(2, conn._recv.getActualQd.call_count)  # noqa: SLF001
        conn._ctrl.isSteady.assert_not_called()  # noqa: SLF001

    def test_a_stream_that_stopped_is_never_an_arm_that_stands(self) -> None:
        """The receive interface hands back its last sample once the stream stops: only a sample the controller sent
        since the read before counts."""
        conn, clock = _conn("joint_speeds")
        conn._recv.getActualQd.return_value = _STILL  # noqa: SLF001
        conn._recv.getTimestamp.side_effect = None  # noqa: SLF001
        conn._recv.getTimestamp.return_value = 1234.5  # noqa: SLF001
        with self.assertLogs("URConnection", level="WARNING"):
            self.assertFalse(conn.wait_until_steady(0.2))
        self.assertGreaterEqual(clock.t, 0.2)
        self.assertFalse(conn.wait_until_steady(0.0))
        conn._ctrl.isSteady.assert_not_called()  # noqa: SLF001

    def test_a_sample_read_twice_counts_once(self) -> None:
        conn, _clock = _conn("joint_speeds", [_STILL] * 4)
        stamps = iter([1.000, 1.000, 1.008, 1.016])
        conn._recv.getTimestamp.side_effect = lambda: next(stamps)  # noqa: SLF001
        self.assertTrue(conn.wait_until_steady(5.0))
        self.assertEqual(4, conn._recv.getActualQd.call_count)  # noqa: SLF001

    def test_speeds_that_cannot_be_read_leave_the_answer_to_is_steady(self) -> None:
        for unreadable in (RuntimeError("unable to get state data"), [0.0, 0.0]):
            with self.subTest(read=repr(unreadable)):
                conn, _clock = _conn("joint_speeds")
                if isinstance(unreadable, Exception):
                    conn._recv.getActualQd.side_effect = unreadable  # noqa: SLF001
                else:
                    conn._recv.getActualQd.return_value = unreadable  # noqa: SLF001
                conn._ctrl.isSteady.return_value = False  # noqa: SLF001
                with self.assertLogs("URConnection", level="WARNING"):
                    self.assertFalse(conn.wait_until_steady(0.0))
                conn._ctrl.isSteady.assert_called_once()  # noqa: SLF001

    def test_a_receive_interface_without_a_timestamp_leaves_the_answer_to_is_steady(self) -> None:
        """A binding or a stand-in without ``getTimestamp`` (the desk bench's, 2026-10-09) is one whose speeds cannot
        be read: isSteady decides, and nothing raises out of the gate."""
        conn, _clock = _conn("joint_speeds")
        conn._recv.getTimestamp.side_effect = AttributeError("'RTDEReceiveInterface' object has no attribute 'getTimestamp'")  # noqa: SLF001,E501
        conn._ctrl.isSteady.return_value = True  # noqa: SLF001

        with self.assertLogs("URConnection", level="WARNING"):
            self.assertTrue(conn.wait_until_steady(0.0))
        conn._ctrl.isSteady.assert_called_once()  # noqa: SLF001


class IsSteadyStaysTheDefaultTests(unittest.TestCase):
    def test_the_controllers_is_steady_decides_and_no_speed_is_read(self) -> None:
        conn, _clock = _conn("is_steady")
        conn._ctrl.isSteady.side_effect = [False, False, True]  # noqa: SLF001
        self.assertTrue(conn.wait_until_steady(5.0, poll_interval_s=0.001))
        self.assertEqual(3, conn._ctrl.isSteady.call_count)  # noqa: SLF001
        conn._recv.getActualQd.assert_not_called()  # noqa: SLF001
        self.assertEqual("is_steady", URConnection("127.0.0.1").steady_signal)

    def test_a_signal_of_another_name_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            URConnection("127.0.0.1", steady_signal="joint_positions")

    def test_the_key_loads_and_defaults_to_is_steady(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot import RobotConfig

        self.assertEqual("is_steady", RobotConfig.model_validate({"vendor": "ur"}).safety.dwell.steady_signal)
        asked = RobotConfig.model_validate({"vendor": "ur", "safety": {"dwell": {"steady_signal": "joint_speeds"}}})
        self.assertEqual("joint_speeds", asked.safety.dwell.steady_signal)
        with self.assertRaises(ValidationError):
            RobotConfig.model_validate({"vendor": "ur", "safety": {"dwell": {"steady_signal": "isSteady"}}})


if __name__ == "__main__":
    unittest.main()
