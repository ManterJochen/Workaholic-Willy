"""A CB3 says its serial, so the console can prove URSim CB3 a simulator, and it asks once per connection.

ur_rtde's ``getSerialNumber()`` refuses PolyScope below 5.6, a CB3 among them, so the console's simulator proof
(api/telemetry.py) never saw a serial there: the top bar could not say "URSim" for a CB3, and
scripts/ursim/probe_console_task.py refused it (exit 2). MEASURED 2026-10-02: URSim CB3 (PolyScope 3.15.8) answers the
dashboard's own "get serial number" with ``2018309999``. The driver now asks that over a dashboard connection of its
own, never the driver's shared one, takes only digits for a serial, and keeps a settled answer for the connection,
because the console asks on every status read. What proves a simulator is the owner's rule of 2026-10-02, pinned in
tests/test_ur_controller_identity.py: 10 digits ending 9999 or 11 ending 99999, on a loopback host only.
"""

from __future__ import annotations

import socket
import threading
import unittest
from unittest.mock import MagicMock, patch

from src.robot.drivers.ur import connection as conn_mod

_BANNER = b"Connected: Universal Robots Dashboard Server\n"


class _Dashboard:
    """A dashboard server on a loopback port: the banner, then one answer per command line."""

    def __init__(self, answer: bytes | None) -> None:
        self.answer = answer
        self.commands: list[str] = []
        self._server = socket.create_server(("127.0.0.1", 0))
        self.port = self._server.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        try:
            client, _ = self._server.accept()
        except OSError:  # closed before anybody called
            return
        with client:
            if self.answer is None:  # a dashboard that never speaks
                client.recv(256)
                return
            client.sendall(_BANNER)
            line = b""
            while not line.endswith(b"\n"):
                chunk = client.recv(256)
                if not chunk:
                    return
                line += chunk
            self.commands.append(line.decode().strip())
            client.sendall(self.answer)

    def close(self) -> None:
        self._server.close()
        self._thread.join(timeout=2)


class TheDashboardIsAskedOverItsOwnConnectionTests(unittest.TestCase):

    def test_one_command_is_answered_by_one_line(self) -> None:
        dashboard = _Dashboard(b"2018309999\n")
        try:
            answer = conn_mod._ask_dashboard("127.0.0.1", "get serial number", port=dashboard.port)
        finally:
            dashboard.close()

        self.assertEqual(answer, "2018309999")
        self.assertEqual(dashboard.commands, ["get serial number"])

    def test_a_dashboard_that_never_speaks_is_an_OSError_not_a_hang(self) -> None:
        dashboard = _Dashboard(None)
        try:
            with self.assertRaises(OSError):
                conn_mod._ask_dashboard("127.0.0.1", "get serial number", port=dashboard.port, timeout_s=0.2)
        finally:
            dashboard.close()


def _connection(*, ur_rtde_answer: object = "", host: str = "127.0.0.1") -> conn_mod.URConnection:
    """A connected-looking connection: only the shared dashboard ``controller_serial`` reads is set."""
    c = conn_mod.URConnection(ip=host)
    c._dashboard = MagicMock()
    if isinstance(ur_rtde_answer, Exception):
        c._dashboard.getSerialNumber.side_effect = ur_rtde_answer
    else:
        c._dashboard.getSerialNumber.return_value = ur_rtde_answer
    return c


_BELOW_5_6 = RuntimeError("getSerialNumber is not supported on PolyScope versions less than 5.6.0")


class ControllerSerialTests(unittest.TestCase):

    def test_an_e_series_answer_is_used_and_no_connection_of_its_own_is_opened(self) -> None:
        c = _connection(ur_rtde_answer="20195399999")
        with patch.object(conn_mod, "_ask_dashboard") as ask:
            self.assertEqual(c.controller_serial(), "20195399999")
        ask.assert_not_called()

    def test_a_cb3_is_asked_over_a_connection_of_its_own(self) -> None:
        c = _connection(ur_rtde_answer=_BELOW_5_6)
        with patch.object(conn_mod, "_ask_dashboard", return_value="2018309999") as ask:
            self.assertEqual(c.controller_serial(), "2018309999")
        ask.assert_called_once_with("127.0.0.1", "get serial number")

    def test_an_answer_from_ur_rtde_that_is_no_serial_is_asked_again_over_its_own(self) -> None:
        for answer in ("", "None", "could not understand: 'get serial number'"):
            with self.subTest(answer=answer):
                c = _connection(ur_rtde_answer=answer)
                with patch.object(conn_mod, "_ask_dashboard", return_value="2018309999"):
                    self.assertEqual(c.controller_serial(), "2018309999")

    def test_a_settled_answer_is_kept_for_the_connection(self) -> None:
        c = _connection(ur_rtde_answer=_BELOW_5_6)
        with patch.object(conn_mod, "_ask_dashboard", return_value="2018309999") as ask:
            for _ in range(3):
                self.assertEqual(c.controller_serial(), "2018309999")
        ask.assert_called_once()
        c._dashboard.getSerialNumber.assert_called_once()

    def test_an_answer_that_is_no_serial_is_None_and_is_not_asked_again(self) -> None:
        c = _connection(ur_rtde_answer=_BELOW_5_6)
        with patch.object(conn_mod, "_ask_dashboard", return_value="could not understand: 'get serial number'") as ask:
            self.assertIsNone(c.controller_serial())
            self.assertIsNone(c.controller_serial())
        ask.assert_called_once()

    def test_a_socket_that_failed_is_None_and_is_asked_again_next_time(self) -> None:
        c = _connection(ur_rtde_answer=_BELOW_5_6)
        with patch.object(conn_mod, "_ask_dashboard", side_effect=[OSError("timed out"), "2018309999"]) as ask:
            self.assertIsNone(c.controller_serial())
            self.assertEqual(c.controller_serial(), "2018309999")
        self.assertEqual(ask.call_count, 2)

    def test_no_dashboard_is_None_and_nothing_is_opened(self) -> None:
        c = conn_mod.URConnection(ip="127.0.0.1")
        with patch.object(conn_mod, "_ask_dashboard") as ask:
            self.assertIsNone(c.controller_serial())
        ask.assert_not_called()

    def test_a_teardown_forgets_the_serial(self) -> None:
        c = _connection(ur_rtde_answer=_BELOW_5_6)
        with patch.object(conn_mod, "_ask_dashboard", return_value="2018309999"):
            c.controller_serial()
        c._safe_teardown()
        c._dashboard = MagicMock()
        c._dashboard.getSerialNumber.return_value = "20195399999"
        self.assertEqual(c.controller_serial(), "20195399999")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
