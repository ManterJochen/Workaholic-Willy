"""The controller's own kinematics are read from its primary interface, once per connection, and nothing is sent there.

A UR controller computes its forward and inverse kinematics on its own DH rows, the nominal table plus the arm's factory
calibration, and says what they are in the kinematics info of the first robot state its primary client interface sends a
new client (URSim CB3 3.15.8, measured: later robot states leave it out). ``URConnection.controller_kinematics`` reads
them there over a connection of its own, the read-only port 30011 first and the primary port 30001 after it, and only
ever reads: a primary interface runs any URScript it is sent. ``active_tcp_offset`` reads the TCP those kinematics apply.

What this file pins: the rows of a calibrated URSim are read from the bytes it sent (a recording), and not one byte goes
back; they are read once per connection and again after a reconnect; a port that refuses hands over to the next; robot
states without the kinematics info, a garbled stream and a stream cut short read as ``None`` and say so; a disconnect
during the read keeps nothing; the TCP is read as the controller reports it, zero included, once per connection and
again for a fresh program.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.robot.drivers.ur import connection as conn_mod
from src.robot.drivers.ur.connection import ControllerKinematics, URConnection

_RECORDED = Path(__file__).parent / "data" / "ur_primary" / "ursim_cb3_3.15.8_calibrated.bin"


def _message(kind: int, body: bytes) -> bytes:
    return struct.pack(">iB", 5 + len(body), kind) + body


def _robot_state(*parts: bytes) -> bytes:
    return _message(16, b"".join(parts))


def _part(kind: int, body: bytes) -> bytes:
    return struct.pack(">iB", 5 + len(body), kind) + body


class _Primary:
    """A primary interface on localhost that sends ``payload`` to every client and records what each sends back."""

    def __init__(self, payload: bytes, *, close_after: bool = False) -> None:
        self.payload = payload
        self.close_after = close_after
        self.received = b""
        self.accepted = 0
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(4)
        self.port = self.server.getsockname()[1]
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self) -> None:
        self.server.settimeout(0.1)
        while not self._stop.is_set():
            try:
                client, _ = self.server.accept()
            except OSError:
                continue
            self.accepted += 1
            with client:
                client.sendall(self.payload)
                client.settimeout(0.3)
                if self.close_after:
                    continue
                try:
                    while not self._stop.is_set():
                        data = client.recv(4096)
                        if not data:
                            break
                        self.received += data
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        self.server.close()
        self.thread.join(timeout=2.0)


def _connected() -> URConnection:
    conn = URConnection("127.0.0.1")
    conn._ctrl = MagicMock()  # noqa: SLF001
    conn._recv = MagicMock()  # noqa: SLF001
    return conn


def _a_closed_port() -> int:
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    return port


class TheRowsAreReadFromThePrimaryInterfaceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.primary = _Primary(_RECORDED.read_bytes())
        self.addCleanup(self.primary.close)

    def _ports(self, *ports: int):  # noqa: ANN202
        return patch.object(conn_mod, "_PRIMARY_PORTS", tuple(ports))

    def test_the_rows_of_a_calibrated_ursim_are_read_and_not_a_byte_goes_back(self) -> None:
        """⭐ The recording: URSim CB3 3.15.8, a UR10 with a calibration.conf, port 30011 from its first byte."""
        conn = _connected()
        with self._ports(self.primary.port):
            read = conn.controller_kinematics()
        assert read is not None
        self.assertEqual((0.0012, -0.0021, 0.0035, -0.0009, 0.0017, 0.0008), read.theta_rad)
        self.assertEqual((0.00031, -0.61117, -0.57357, 0.00022, -0.00041, 0.0), read.a_m)
        self.assertEqual((0.12797, 0.0011, -0.0009, 0.164461, 0.11608, 0.09191), read.d_m)
        self.assertAlmostEqual(1.570796327 + 0.0011, read.alpha_rad[0], places=12)
        self.assertEqual((0xFFFFFFFF,) * 6, read.checksums)
        self.assertEqual(0, read.calibration_status)
        self.assertEqual(self.primary.port, read.port)
        self.primary.close()
        self.assertEqual(b"", self.primary.received, "nothing may be sent to a primary interface")

    def test_the_rows_are_read_once_per_connection_and_again_after_a_reconnect(self) -> None:
        conn = _connected()
        with self._ports(self.primary.port):
            first = conn.controller_kinematics()
            self.assertIs(first, conn.controller_kinematics())
            self.assertEqual(1, self.primary.accepted)
            epoch = conn.kinematics_epoch
            conn._forget_the_kinematics()  # noqa: SLF001 (what connect and disconnect do)
            self.assertEqual(epoch + 1, conn.kinematics_epoch)
            self.assertEqual(first, conn.controller_kinematics())
        self.assertEqual(2, self.primary.accepted)

    def test_a_port_that_refuses_hands_over_to_the_next(self) -> None:
        conn = _connected()
        with self._ports(_a_closed_port(), self.primary.port):
            read = conn.controller_kinematics()
        assert read is not None
        self.assertEqual(self.primary.port, read.port)

    def test_a_disconnect_during_the_read_keeps_nothing(self) -> None:
        conn = _connected()
        real = conn_mod._read_kinematics_info  # noqa: SLF001

        def read_meanwhile_disconnected(host: str, port: int, **asked: object) -> ControllerKinematics:
            answer = real(host, port, **asked)  # type: ignore[arg-type]
            conn._forget_the_kinematics()  # noqa: SLF001
            return answer

        with self._ports(self.primary.port), patch.object(conn_mod, "_read_kinematics_info", read_meanwhile_disconnected):
            self.assertIsNone(conn.controller_kinematics())
        with self._ports(self.primary.port):
            self.assertIsNotNone(conn.controller_kinematics(), "the next ask reads again")

    def test_the_render_names_every_row_and_the_checksums(self) -> None:
        conn = _connected()
        with self._ports(self.primary.port):
            read = conn.controller_kinematics()
        assert read is not None
        said = read.render()
        for word in ("theta [0.0012", "a [0.00031, -0.61117", "d [0.12797", "alpha [1.5718963", "0xFFFFFFFF",
                     "calibration status 0"):
            self.assertIn(word, said)


class WhatCannotBeReadIsNoneTests(unittest.TestCase):
    def _read(self, payload: bytes, *, close_after: bool = False) -> "tuple[ControllerKinematics | None, URConnection]":
        primary = _Primary(payload, close_after=close_after)
        self.addCleanup(primary.close)
        conn = _connected()
        with patch.object(conn_mod, "_PRIMARY_PORTS", (primary.port,)), \
                self.assertLogs("URConnection", level="WARNING") as said:
            read = conn.controller_kinematics()
        self.assertIn("could not be read", "\n".join(said.output))
        return read, conn

    def test_robot_states_without_the_kinematics_info_read_as_none_once(self) -> None:
        joint_data = _part(1, b"\x00" * 40)
        read, conn = self._read(_message(20, b"version") + _robot_state(joint_data) + _robot_state(joint_data))
        self.assertIsNone(read)
        with patch.object(conn_mod, "_read_kinematics_info", side_effect=AssertionError("read again")):
            self.assertIsNone(conn.controller_kinematics(), "a failure is kept until the next connect")

    def test_a_garbled_stream_reads_as_none(self) -> None:
        read, _ = self._read(struct.pack(">iB", 2, 16) + b"\x00" * 64)
        self.assertIsNone(read)

    def test_a_stream_cut_short_reads_as_none(self) -> None:
        whole = _RECORDED.read_bytes()
        read, _ = self._read(whole[: len(whole) // 3], close_after=True)
        self.assertIsNone(read)

    def test_a_kinematics_info_too_short_reads_as_none(self) -> None:
        read, _ = self._read(_robot_state(_part(5, b"\x00" * 100)))
        self.assertIsNone(read)


class TheActiveTcpIsReadAsTheControllerReportsItTests(unittest.TestCase):
    def test_a_zero_tcp_stays_zero_and_is_read_once(self) -> None:
        conn = _connected()
        conn._ctrl.getTCPOffset.return_value = [0.0] * 6  # noqa: SLF001
        self.assertEqual([0.0] * 6, conn.active_tcp_offset())
        self.assertEqual([0.0] * 6, conn.active_tcp_offset())
        self.assertEqual(1, conn._ctrl.getTCPOffset.call_count)  # noqa: SLF001

    def test_a_fresh_program_reads_the_tcp_again_and_keeps_the_rows(self) -> None:
        conn = _connected()
        conn._ctrl.getTCPOffset.return_value = [0.0, 0.0, 0.157, 0.0, 0.0, -1.5707963267948966]  # noqa: SLF001
        conn._kinematics, conn._kinematics_settled = "rows", True  # noqa: SLF001
        conn.active_tcp_offset()
        conn._forget_the_kinematics(tcp_only=True)  # noqa: SLF001 (what restart_control_script does)
        self.assertEqual(0.157, conn.active_tcp_offset()[2])  # type: ignore[index]
        self.assertEqual(2, conn._ctrl.getTCPOffset.call_count)  # noqa: SLF001
        self.assertEqual("rows", conn._kinematics)  # noqa: SLF001

    def test_a_tcp_that_cannot_be_read_is_none_and_says_so(self) -> None:
        conn = _connected()
        conn._ctrl.getTCPOffset.side_effect = RuntimeError("getTCPOffset() function did not succeed!")  # noqa: SLF001
        with self.assertLogs("URConnection", level="WARNING"):
            self.assertIsNone(conn.active_tcp_offset())
        self.assertIsNone(conn.active_tcp_offset())
        self.assertEqual(1, conn._ctrl.getTCPOffset.call_count)  # noqa: SLF001

    def test_neither_is_read_without_a_connection(self) -> None:
        conn = URConnection("127.0.0.1")
        with self.assertRaises(RuntimeError):
            conn.controller_kinematics()
        with self.assertRaises(RuntimeError):
            conn.active_tcp_offset()


if __name__ == "__main__":
    unittest.main()
