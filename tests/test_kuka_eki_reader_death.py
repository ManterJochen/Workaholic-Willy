"""Iteration-2 regression test: when the EKI reader thread dies on its own
(dropped KRC link / peer close), every in-flight waiter is woken so callers fail
fast instead of blocking for their full timeout.

KUKA hardware is not required — a fake peer-close socket reproduces the condition.
"""

from __future__ import annotations

import threading
import unittest

from src.robot.drivers.kuka.eki_client import (
    EkiClient,
    _PendingAck,
    _PendingRequest,
)


class _PeerClosedSocket:
    """``recv()`` returns ``b''`` immediately, simulating a peer that closed."""

    def recv(self, _bufsize: int) -> bytes:
        return b""


class EkiReaderDeathWakesWaitersTests(unittest.TestCase):
    def test_wake_all_pending_sets_events_and_clears(self) -> None:
        client = EkiClient()
        fk, ik, ack = _PendingRequest(), _PendingRequest(), _PendingAck()
        echo = threading.Event()
        client._pending_fk["fk1"] = fk
        client._pending_ik["ik1"] = ik
        client._pending_acks.append(ack)
        client._pending_echo["tok1"] = echo

        client._wake_all_pending()

        self.assertTrue(fk.event.is_set())
        self.assertTrue(ik.event.is_set())
        self.assertTrue(ack.event.is_set())
        self.assertTrue(echo.is_set())
        self.assertEqual(client._pending_fk, {})
        self.assertEqual(client._pending_ik, {})
        self.assertEqual(client._pending_acks, [])
        self.assertEqual(client._pending_echo, {})

    def test_reader_loop_peer_close_wakes_pending_ack(self) -> None:
        client = EkiClient()
        client._sock = _PeerClosedSocket()
        ack = _PendingAck()
        client._pending_acks.append(ack)

        # Runs synchronously: recv() -> b'' -> loop breaks -> _wake_all_pending().
        client._reader_loop()

        self.assertTrue(
            ack.event.wait(timeout=1.0),
            "in-flight ack waiter was not woken on reader-thread death",
        )
        self.assertEqual(client._pending_acks, [])


if __name__ == "__main__":
    unittest.main()
