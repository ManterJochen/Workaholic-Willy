r"""The Robotiq URCap socket client, against a server that speaks the documented protocol.

⚠ **HONESTY BUCKET (2), NOT (3), AND THE DIFFERENCE IS THIS FILE.** No UR controller exists on a dev
box, so the client cannot be run against the real URCap daemon here. What it CAN be run against is a
real TCP server implementing the grammar as documented -- so framing, parsing, the activation
sequence, the polarity handling and every timeout are exercised over an actual socket rather than
asserted. What that does NOT establish is whether the documentation is right, which is what
`probe()` is for and what a bench with the gripper bolted on will settle.

⛔ **EVERY ASSERTION BELOW CORRESPONDS TO SOMETHING AN ADVERSARIAL CHECK FLAGGED**, not to a shape I
liked. The list it produced: `POS` means two different registers depending on direction; `SPE 0` is
minimum speed rather than stop; `ack` is exactly three bytes with no newline; the reply may arrive
split across reads; and the ecosystem's own reference client contradicts itself about which end is
open.
"""

from __future__ import annotations

import socket
import threading
import unittest

from src.robot.grippers.robotiq_socket import (
    ActivationStatus,
    ObjectStatus,
    RobotiqSocket,
    RobotiqSocketError,
)


class FakeURCap(threading.Thread):
    """A TCP server speaking the port-63352 grammar: `SET a 1 b 2\\n` -> `ack`, `GET a\\n` -> `a 1`.

    ⭐ IT MODELS THE `POS` SPLIT, which is the trap: written, POS is the TARGET (rPR); read, POS is
    the ACTUAL position (gPO). A fake that stored one value would let the client's biggest known
    hazard pass unnoticed.
    """

    daemon = True

    def __init__(
        self, *, split_replies: bool = False, silent: bool = False, lying: bool = False
    ) -> None:
        super().__init__()
        self.vars: dict[str, int] = {
            "ACT": 0, "ATR": 0, "GTO": 0, "SPE": 0, "FOR": 0,
            "STA": 0, "OBJ": 0, "FLT": 0,
            "POS": 137,   # gPO: where the fingers ARE
            "PRE": 0,     # gPR: the echo of what was last requested
        }
        self.commands: list[str] = []
        self._split = split_replies
        self._silent = silent
        #: Answer a GET with the WRONG variable name. The real daemon has no request ids, so a
        #: client that trusted the reply it happened to get would pair answers with questions by
        #: position alone.
        self._lying = lying
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(1)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()

    def run(self) -> None:  # pragma: no cover - exercised through the socket
        try:
            conn, _ = self._sock.accept()
        except OSError:
            return
        with conn:
            buffer = b""
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(1024)
                except OSError:
                    return
                if not chunk:
                    return
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    reply = self._handle(line.decode("ascii").strip())
                    if reply is None:
                        continue
                    if self._split and len(reply) > 2:
                        conn.sendall(reply[:2])
                        conn.sendall(reply[2:])
                    else:
                        conn.sendall(reply)

    def _handle(self, line: str) -> bytes | None:
        self.commands.append(line)
        if self._silent:
            return None
        parts = line.split()
        if not parts:
            return None
        if parts[0] == "GET":
            name = parts[1]
            answered = "SOMETHING_ELSE" if self._lying else name
            return f"{answered} {self.vars.get(name, 0)}".encode()
        if parts[0] == "SET":
            pairs = parts[1:]
            for name, value in zip(pairs[0::2], pairs[1::2]):
                if name == "POS":
                    # ⛔⛔ THE WRITTEN `POS` IS rPR AND MUST NOT TOUCH THE READ-BACK. It echoes into
                    # PRE (gPR) and leaves gPO alone. My first version of this fake stored it in
                    # `vars["POS"]` like every other variable, which made `GET POS` return the
                    # target -- so the fake modelled AWAY the exact trap the test below exists for,
                    # and that test failed against the fake rather than against the client. A fake
                    # that gets the system wrong proves nothing about the code under it.
                    self.vars["PRE"] = int(value)
                    continue
                self.vars[name] = int(value)
                if name == "ACT" and int(value) == 1:
                    self.vars["STA"] = int(ActivationStatus.ACTIVE)
                if name == "ACT" and int(value) == 0:
                    self.vars["STA"] = int(ActivationStatus.RESET)
            return b"ack"
        return b"ack"

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


class _Served(unittest.TestCase):
    split_replies = False
    silent = False
    lying = False

    def setUp(self) -> None:
        self.server = FakeURCap(
            split_replies=self.split_replies, silent=self.silent, lying=self.lying
        )
        self.server.start()
        self.addCleanup(self.server.close)
        self.client = RobotiqSocket(timeout_s=1.0)
        self.client.connect("127.0.0.1", self.server.port)
        self.addCleanup(self.client.disconnect)


class GrammarTests(_Served):
    r"""⭐⭐ **THE WIRE FORMAT IS PINNED AGAINST THE REFERENCE IMPLEMENTATION, QUOTED.** The
    Robotiq URCap's port-63352 grammar is not vendor-documented anywhere, so "is our framing right"
    cannot be answered by reading a manual. It CAN be answered by comparing against the client the
    whole ecosystem uses. Fetched 2026-09-04 from
    https://sdurobotics.gitlab.io/ur_rtde/_static/robotiq_gripper.py, verbatim:

        cmd = "SET"
        for variable, value in var_dict.items():
            cmd += f" {variable} {str(value)}"
        cmd += '
'

        cmd = f"GET {variable}
"

        return data == b'ack'

    and its reset sends ACT 0 then ATR 0, looping until ACT and STA both read 0, then ACT 1, waiting
    until ACT reads 1 and STA reads 3. Every byte this class asserts below is that format. Two
    independently written clients producing identical traffic is the strongest evidence available
    short of the daemon itself, and it is what makes this file bucket (2) rather than a hope.

    ⚠ THE ONE DELIBERATE DIFFERENCE: the reference declares an `ADR` constant (auto-release
    direction) that this client omits, because nothing in our path performs an auto-release. An
    unused constant is not a compatibility gap.
    """

    def test_a_get_parses_the_name_value_pair(self) -> None:
        self.assertEqual(self.client.get("STA"), 0)
        self.assertEqual(self.server.commands[-1], "GET STA")

    def test_a_move_is_one_line_with_gto_last(self) -> None:
        """⛔ ONE LINE, BECAUSE `GTO` IS WHAT STARTS THE MOTION. Sent separately there is a window in
        which the gripper is told to go to a stale target."""
        self.client.move(200, 128, 100)
        self.assertEqual(self.server.commands[-1], "SET POS 200 SPE 128 FOR 100 GTO 1")



class MismatchedReplyTests(_Served):
    lying = True

    def test_a_reply_naming_a_different_variable_is_refused(self) -> None:
        """⛔ THE PROTOCOL HAS NO REQUEST IDS, so a client that took whatever arrived would pair
        answers with questions by arrival order alone. One interleaved command and every reading
        after it belongs to a different question."""
        with self.assertRaises(RobotiqSocketError) as caught:
            self.client.get("STA")
        self.assertIn("expected", str(caught.exception))


class PosIsTwoRegistersTests(_Served):
    def test_writing_pos_does_not_change_what_reading_pos_returns(self) -> None:
        """⛔⛔ THE TRAP THE RESEARCH FOUND. Written, `POS` is rPR, the TARGET. Read, it is gPO, where
        the fingers ARE. `SET POS 200` then `GET POS` does NOT read back 200, and a client that
        assumed it did would report a move as complete the instant it was commanded.
        """
        before = self.client.get_current_position()
        self.client.move(200, 128, 100)
        self.assertEqual(self.client.get_current_position(), before, "GET POS must read gPO")
        self.assertEqual(self.client.get_requested_position(), 200, "PRE is the echo of the target")


class PolarityTests(_Served):
    def test_the_client_speaks_device_counts_only(self) -> None:
        """⛔ 0 = OPEN, 255 = CLOSED, and ur_rtde's OWN unit systems contradict each other about it:
        its `UNIT_DEVICE` says 0 is open while `UNIT_NORMALIZED` says 0.0 is CLOSED. Any snippet
        lifted from there without checking the unit mode inverts the jaws. This client has no unit
        mode, and no millimetres, so there is nothing to invert.
        """
        import inspect

        from src.robot.grippers import robotiq_socket

        source = inspect.getsource(robotiq_socket)
        self.assertNotIn("_mm_to_count", source)
        self.assertNotIn("max_width_mm", source, "this layer must not invent a millimetre map")

    def test_counts_are_clipped_to_the_command_range(self) -> None:
        self.client.move(999, 999, 999)
        self.assertEqual(self.server.commands[-1], "SET POS 255 SPE 255 FOR 255 GTO 1")


class ActivationTests(_Served):
    def test_activation_resets_first_and_waits_for_the_status(self) -> None:
        """The vendor states that after a power loss rACT must be CLEARED and set again, and the
        reference client at sdurobotics.gitlab.io sends exactly this pair in exactly this order:
        ACT 0 then ATR 0, loop until both ACT and STA read 0, then ACT 1 and wait for STA 3.
        """
        self.client.activate(timeout_s=2.0)
        sets = [c for c in self.server.commands if c.startswith("SET")]
        self.assertEqual(sets[0], "SET ACT 0 ATR 0")
        self.assertEqual(sets[1], "SET ACT 1")
        self.assertTrue(self.client.is_active())

    def test_activate_if_needed_is_one_round_trip_when_already_active(self) -> None:
        """⭐ THE PROTOCOL MAKES THIS CHEAP -- `GET STA` answers it -- so idempotence is real here
        rather than faked in Python."""
        self.server.vars["STA"] = int(ActivationStatus.ACTIVE)
        self.server.commands.clear()
        self.client.activate_if_needed()
        self.assertEqual(self.server.commands, ["GET STA"])

    def test_stop_clears_gto_because_speed_zero_does_not_stop(self) -> None:
        """⚠ `SPE 0` IS MINIMUM SPEED, NOT STOP: the register maps 0-100 % onto a non-zero physical
        band (20-150 mm/s on a 2F-85)."""
        self.client.stop()
        self.assertEqual(self.server.commands[-1], "SET GTO 0")


class FramingTests(_Served):
    split_replies = True

    def test_a_reply_split_across_reads_is_still_parsed(self) -> None:
        """⚠ There is no length prefix and no reply terminator to rely on, so trusting a single
        `recv` is trusting the network to frame for you. Both reference clients do exactly that."""
        self.assertEqual(self.client.get("POS"), 137)
        self.client.set(GTO=0)


class SilenceTests(_Served):
    silent = True

    def test_a_daemon_that_never_answers_is_a_typed_refusal(self) -> None:
        """The daemon can be up while its RS-485 link to the gripper is dead, and that is the
        commonest bring-up symptom. It must not read as a hang."""
        with self.assertRaises(RobotiqSocketError) as caught:
            self.client.get("STA")
        self.assertIn("did not answer", str(caught.exception))


class ProbeTests(_Served):
    def test_probe_commands_nothing(self) -> None:
        """⭐ THE READ-ONLY FIRST CONTACT. Every unknown this client could not source is answered by
        running it, moving the gripper from the pendant, and running it again."""
        reading = self.client.probe()
        self.assertTrue(reading.reachable)
        self.assertTrue(all(c.startswith("GET") for c in self.server.commands), self.server.commands)
        self.assertEqual(reading.exit_code, 0)

    def test_the_reading_says_what_it_does_not_prove(self) -> None:
        text = self.client.probe().render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("confirm that on this unit", text)

    def test_a_fault_is_reported_not_raised(self) -> None:
        self.server.vars["FLT"] = 9
        reading = self.client.probe()
        self.assertTrue(reading.faulted)
        self.assertEqual(reading.exit_code, 1)

    def test_object_status_distinguishes_holding_from_arrived(self) -> None:
        """⛔ `AT_POSITION` MEANS THE GRIPPER IS HOLDING NOTHING: the fingers met, or reached the
        target unobstructed. A caller reading "not moving" as "gripped" reads it as success."""
        self.server.vars["OBJ"] = int(ObjectStatus.AT_POSITION)
        self.assertIs(self.client.object_status(), ObjectStatus.AT_POSITION)
        self.server.vars["OBJ"] = int(ObjectStatus.STOPPED_CLOSING)
        self.assertIs(self.client.object_status(), ObjectStatus.STOPPED_CLOSING)


class ConnectionTests(unittest.TestCase):
    def test_an_unreachable_port_names_the_three_things_to_check(self) -> None:
        client = RobotiqSocket(timeout_s=0.3)
        with self.assertRaises(RobotiqSocketError) as caught:
            client.connect("127.0.0.1", 1)
        message = str(caught.exception)
        self.assertIn("URCap", message)
        self.assertIn("Tool I/O", message)

    def test_using_it_before_connecting_is_a_typed_refusal(self) -> None:
        with self.assertRaises(RobotiqSocketError):
            RobotiqSocket().get("STA")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
