r"""The OnRobot RG2/RG6 client, against a Modbus TCP server that answers the documented frames.

⚠ **HONESTY BUCKET (2) FOR THE FRAMING, (3) FOR THE MEANING.** No Compute Box exists here, so what
is exercised is the wire: MBAP framing, transaction-id pairing, split reads, exception replies, and
the register block layout. What is NOT exercised is whether register 267 really is the actual width
on a real RG6 -- the frames reproduce verbatim from the vendor manual and were re-derived by hand in
an adversarial pass, and two independent open-source drivers agree, but agreement is not measurement.
`probe()` is read-only for exactly that reason.

⛔ **THE FAKE ANSWERS THE MANUAL'S OWN EXAMPLE FRAMES**, so a client that framed Modbus wrongly fails
here rather than on a machine with a gripper bolted to it.
"""

from __future__ import annotations

import socket
import struct
import threading
import unittest

from src.robot.grippers.onrobot_modbus import (
    UNIT_DUAL_PRIMARY,
    UNIT_QUICK_CHANGER,
    ModbusError,
    OnRobotRG,
    RGStatus,
)


class FakeComputeBox(threading.Thread):
    """A Modbus TCP server holding one register bank per unit id."""

    daemon = True

    def __init__(self, *, units: set[int] | None = None, split: bool = False) -> None:
        super().__init__()
        self.units = units if units is not None else {UNIT_QUICK_CHANGER}
        self.regs: dict[int, int] = {267: 655, 268: int(RGStatus.AT_POSITION)}
        self.writes: list[tuple[int, list[int]]] = []
        self._split = split
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
            while not self._stop.is_set():
                try:
                    head = self._read(conn, 6)
                    if head is None:
                        return
                    tid, _proto, length = struct.unpack(">HHH", head)
                    rest = self._read(conn, length)
                    if rest is None:
                        return
                except OSError:
                    return
                unit, pdu = rest[0], rest[1:]
                reply = self._handle(unit, pdu)
                if reply is None:
                    continue  # this unit id is silent, as an absent tool is
                frame = struct.pack(">HHHB", tid, 0, len(reply) + 1, unit) + reply
                if self._split and len(frame) > 4:
                    conn.sendall(frame[:4])
                    conn.sendall(frame[4:])
                else:
                    conn.sendall(frame)

    @staticmethod
    def _read(conn: socket.socket, count: int) -> bytes | None:
        out = b""
        while len(out) < count:
            chunk = conn.recv(count - len(out))
            if not chunk:
                return None
            out += chunk
        return out

    def _handle(self, unit: int, pdu: bytes) -> bytes | None:
        if unit not in self.units:
            return None
        if pdu[0] == 0x03:
            start, qty = struct.unpack(">HH", pdu[1:5])
            if start + qty > 65535:
                return bytes([0x83, 0x02])  # illegal data address
            data = b"".join(struct.pack(">H", self.regs.get(start + i, 0)) for i in range(qty))
            return bytes([0x03, len(data)]) + data
        if pdu[0] == 0x10:
            start, qty = struct.unpack(">HH", pdu[1:5])
            values = [struct.unpack(">H", pdu[6 + 2 * i:8 + 2 * i])[0] for i in range(qty)]
            self.writes.append((start, values))
            for i, v in enumerate(values):
                self.regs[start + i] = v
            return struct.pack(">BHH", 0x10, start, qty)
        return bytes([pdu[0] | 0x80, 0x01])

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


class _Served(unittest.TestCase):
    units: set[int] | None = None
    split = False

    def setUp(self) -> None:
        self.box = FakeComputeBox(units=self.units, split=self.split)
        self.box.start()
        self.addCleanup(self.box.close)
        self.rg = OnRobotRG(timeout_s=1.0)
        self.rg.connect("127.0.0.1", self.box.port)
        self.addCleanup(self.rg.disconnect)


class FramingTests(_Served):
    def test_a_read_frames_the_manuals_example(self) -> None:
        """Registers 267 and 268 are adjacent, so one transaction answers "how open" and "what state"."""
        self.assertAlmostEqual(self.rg.width_mm(), 65.5)
        self.assertIs(self.rg.status(), RGStatus.AT_POSITION)

    def test_a_modbus_exception_is_a_typed_refusal(self) -> None:
        with self.assertRaises(ModbusError) as caught:
            self.rg.read_registers(65535, 4)
        self.assertIn("exception", str(caught.exception))

    def test_using_it_before_connecting_refuses(self) -> None:
        with self.assertRaises(ModbusError):
            OnRobotRG().status()


class SplitFrameTests(_Served):
    split = True

    def test_a_frame_split_across_reads_is_reassembled(self) -> None:
        """⚠ Modbus TCP has a length field precisely because it is a stream. Assuming one recv is one
        frame works on a quiet LAN and stops working on a busy one."""
        self.assertAlmostEqual(self.rg.width_mm(), 65.5)


class GripTests(_Served):
    def test_a_grip_writes_force_width_control_in_one_block(self) -> None:
        """One FC 0x10 of three consecutive registers does the whole move: force in tenths of a
        newton, width in tenths of a millimetre, then the control value that commits it."""
        self.rg.grip(width_mm=40.0, force_n=60.0)
        self.assertEqual(self.box.writes[-1], (0, [600, 400, 1]))

    def test_the_fingertip_offset_uses_a_different_control_value(self) -> None:
        self.rg.grip(width_mm=40.0, force_n=60.0, use_fingertip_offset=True)
        self.assertEqual(self.box.writes[-1], (0, [600, 400, 16]))

    def test_there_is_no_speed_argument(self) -> None:
        """⛔ NOT AN OMISSION. RG2/RG6 have no speed register anywhere in the writable map, and
        OnRobot's own library exposes only a read-only rg_get_speed. Accepting a speed and dropping
        it would be a lie to the caller, which is the defect shape this convention keeps finding.
        """
        import inspect

        params = set(inspect.signature(OnRobotRG.grip).parameters)
        self.assertNotIn("speed", params)
        self.assertNotIn("speed_mm_s", params)

    def test_bigger_width_means_more_open_unlike_robotiq(self) -> None:
        """⛔ THE POLARITY IS THE OPPOSITE OF THE OTHER CLIENT IN THIS PACKAGE. Robotiq counts run
        0 = open, 255 = closed; here the number is an OPENING in tenths of a millimetre. Porting a
        helper between the two files inverts a gripper."""
        self.rg.grip(width_mm=10.0, force_n=20.0)
        narrow = self.box.writes[-1][1][1]
        self.rg.grip(width_mm=80.0, force_n=20.0)
        wide = self.box.writes[-1][1][1]
        self.assertLess(narrow, wide)

    def test_a_tripped_safety_switch_refuses_the_command(self) -> None:
        """⛔ The gripper is dead until TOOL POWER is cycled, and a command sent meanwhile is
        silently discarded. Refusing beats writing into a void and reporting success."""
        self.box.regs[268] = int(RGStatus.SAFETY_1_TRIPPED)
        before = len(self.box.writes)
        with self.assertRaises(ModbusError) as caught:
            self.rg.grip(width_mm=40.0, force_n=60.0)
        self.assertIn("TOOL POWER", str(caught.exception))
        self.assertEqual(len(self.box.writes), before, "nothing may be written after a refusal")

    def test_stop_writes_the_stop_control_value(self) -> None:
        self.rg.stop()
        self.assertEqual(self.box.writes[-1], (0, [0, 0, 8]))


class ProbeTests(_Served):
    units = {UNIT_QUICK_CHANGER, UNIT_DUAL_PRIMARY}

    def test_the_probe_commands_nothing_and_sweeps_the_unit_ids(self) -> None:
        """⭐ THE SWEEP IS FREE AND REPLACES A TABLE LOOKUP WITH A MEASUREMENT: which side of a Dual
        Quick Changer this tool is on is answered by which unit ids answer."""
        reading = self.rg.probe()
        self.assertTrue(reading.reachable)
        self.assertEqual(reading.answering_units, (UNIT_QUICK_CHANGER, UNIT_DUAL_PRIMARY))
        self.assertEqual(self.box.writes, [], "a probe must never write")

    def test_the_reading_says_it_cannot_identify_the_tool(self) -> None:
        """⛔ AN RG2-FT ANSWERS ON THE SAME UNIT ID WITH AN INCOMPATIBLE MAP, because the id is chosen
        by the MOUNTING and not by the tool. There is no way to tell from Modbus alone, so the
        reading says so instead of implying otherwise."""
        text = self.rg.probe().render()
        self.assertFalse(text.endswith("\n"))
        text.encode("ascii")
        self.assertIn("RG2-FT", text)
        self.assertIn("calipers", text)

    def test_a_blocked_gripper_reports_exit_one(self) -> None:
        self.box.regs[268] = int(RGStatus.SAFETY_2_TRIPPED)
        reading = self.rg.probe()
        self.assertTrue(reading.blocked)
        self.assertEqual(reading.exit_code, 1)
        self.assertIn("Cycle TOOL POWER", reading.render())

    def test_to_dict_is_json_safe(self) -> None:
        import json

        json.dumps(self.rg.probe().to_dict())


class AbsentToolTests(_Served):
    units: set[int] = set()

    def test_a_silent_unit_id_is_reported_not_hung(self) -> None:
        """Silence is the documented symptom of EVERY misconfiguration here: wrong unit id, Modbus
        not served by this firmware, or no tool on that Quick Changer side."""
        reading = self.rg.probe(sweep_units=False)
        self.assertFalse(reading.reachable)
        self.assertIn("did not answer", reading.detail)
        self.assertEqual(reading.exit_code, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
