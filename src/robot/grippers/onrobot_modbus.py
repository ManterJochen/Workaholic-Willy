r"""A dependency-free Modbus TCP client for OnRobot RG2 and RG6 through the Compute Box.

Two function codes and one header are the whole surface, so `pymodbus` would buy
nothing and cost a dependency this repository has decided not to take. FC 0x03 reads
holding registers, FC 0x10 writes several, and the mbap header is seven big-endian
bytes.

Three things are the opposite of Robotiq, and each one is a way to break a gripper.

  1. There is no activation. Nothing here corresponds to the Robotiq `rACT`, and no
     calibration stroke exists. Readiness is reads only. A client that activated an
     RG6 would be writing into whatever register 0 happens to be, which is the target
     force.
  2. There is no speed register. The writable map is force(0), width(1), control(2)
     and nothing else below 1031, and OnRobot's own XML-RPC library offers only a
     read-only `rg_get_speed`. So :meth:`OnRobotRG.grip` takes no speed, and a `speed`
     argument would be a lie to the caller.
  3. Width is in tenths of a millimetre, so bigger is more open. Robotiq counts run
     the other way, 0 for open and 255 for closed. Porting a helper between the two
     files inverts a gripper.

The one that fails silently is that an RG2-FT answers on the same unit id with an
incompatible map. Unit 65 is chosen by the mounting rather than by the tool, so a
force-torque variant sitting in the same Quick Changer replies to every read looking
perfectly healthy while meaning something else. That is a write of a force value into
a width field, and Modbus alone cannot detect it. :meth:`OnRobotRG.probe` says so, and
the identity check is a step on the bench, in the Web Client.

An FC 0x10 echo means the write was accepted, not that the gripper is moving and not
that it finished. OnRobot states that the gripper ignores a grip command while the
busy flag is set, and an ignored grip still returns a perfectly normal echo.
Confirming a move means polling the status word.

Honesty bucket 3. No Compute Box exists on this machine. Every frame below reproduces
from the vendor manual, was re-derived by hand, and agrees with two independent
open-source drivers, but nothing here has spoken to hardware. The 2FG7 is deliberately
absent: no public register map could be sourced for it, and a guessed Modbus address is
the most expensive kind of wrong, because it reads a plausible number or writes into
something else and both look like success.
"""

from __future__ import annotations

import socket
import struct
import threading
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any

__all__ = [
    "ModbusError",
    "OnRobotRG",
    "RGProbeReading",
    "RGStatus",
]

#: The Compute Box Modbus TCP port. The box default address is 192.168.1.1, but do not
#: assume it: the documented Dynamic IP mode falls back to that only after a 60-second
#: DHCP timeout.
DEFAULT_PORT = 502

#: Unit id, chosen by the mounting rather than by the tool. Quick Changer and HEX-E/H QC
#: both use 65; a Dual Quick Changer uses 66 for the primary side and 67 for the
#: secondary. The Compute Box answers on 63, which is where a tripped safety switch is
#: reset.
UNIT_QUICK_CHANGER = 0x41
UNIT_DUAL_PRIMARY = 0x42
UNIT_DUAL_SECONDARY = 0x43
UNIT_COMPUTE_BOX = 0x3F

#: Write block, three consecutive holding registers, 0-based as the PDU wants and as
#: OnRobot prints them. The manual decimal and hex agree throughout and the map starts
#: at register 0, which a 1-based 4xxxx scheme cannot have.
_REG_TARGET_FORCE = 0      # uint16, tenths of a newton
_REG_TARGET_WIDTH = 1      # uint16, tenths of a millimetre
_REG_CONTROL = 2           # 1 grip, 16 grip using the fingertip offset, 8 stop

#: Read block. 267 and 268 are adjacent, so one transaction answers how open the jaws
#: are and what state they are in.
_REG_ACTUAL_WIDTH = 267    # 0x010B, tenths of a millimetre, without the fingertip offset
_REG_STATUS = 268          # 0x010C, bitfield

_CTRL_GRIP = 1
_CTRL_GRIP_WITH_OFFSET = 16
_CTRL_STOP = 8

_FC_READ_HOLDING = 0x03
_FC_WRITE_MULTIPLE = 0x10


class RGStatus(IntFlag):
    """The status word, register 268. Bit meanings verbatim from the vendor manual."""

    #: A motion is ongoing. The gripper ignores a new grip while this is set and still
    #: answers the write with a normal echo, so a caller that does not check this loses
    #: a command.
    BUSY = 1 << 0
    #: A part is detected between the fingers.
    GRIP_DETECTED = 1 << 1
    #: The commanded position was reached without meeting anything.
    AT_POSITION = 1 << 2
    #: Safety switch 1 tripped. The gripper is dead until tool power is cycled.
    SAFETY_1_TRIPPED = 1 << 3
    SAFETY_1_ACTIVE = 1 << 4
    #: Safety switch 2 tripped. Same consequence.
    SAFETY_2_TRIPPED = 1 << 5
    SAFETY_2_ACTIVE = 1 << 6


#: The three bits that mean this gripper must not be commanded. Fail-closed, and the
#: first two need a tool power cycle rather than a retry.
_BLOCKING = RGStatus.SAFETY_1_TRIPPED | RGStatus.SAFETY_2_TRIPPED | RGStatus.SAFETY_2_ACTIVE


class ModbusError(RuntimeError):
    """The box refused, framed something unparseable, or went away."""


@dataclass(frozen=True, slots=True)
class RGProbeReading:
    """A read-only interrogation of one gripper. Commands nothing, moves nothing."""

    reachable: bool
    unit: int
    width_mm: float | None = None
    status: RGStatus = RGStatus(0)
    detail: str = ""
    #: Which unit ids answered a read at all. The sweep costs nothing and settles the
    #: primary or secondary Dual Quick Changer question by measurement rather than by
    #: reading the manual.
    answering_units: tuple[int, ...] = ()
    raw: dict[str, int] = field(default_factory=dict)

    @property
    def blocked(self) -> bool:
        return bool(self.status & _BLOCKING)

    @property
    def exit_code(self) -> int:
        return 0 if self.reachable and not self.blocked else 1

    def render(self) -> str:
        """The reading and what it cannot establish. ASCII, no trailing newline."""
        if not self.reachable:
            return f"  UNREACHABLE on unit {self.unit}: {self.detail}"
        flags = ", ".join(f.name for f in RGStatus if f in self.status and f.name) or "none"
        lines = [
            f"  unit {self.unit} answered: width {self.width_mm:.1f} mm, status 0x{int(self.status):04x}",
            f"    flags: {flags}",
        ]
        if self.answering_units:
            lines.append(f"    units that answered a read: {list(self.answering_units)}")
        if self.blocked:
            lines.append("    BLOCKED: a safety switch is tripped. Cycle tool power; a retry will not help.")
        # The caveat is part of the reading, because every number above looks equally
        # healthy on the wrong tool.
        lines.append(
            "    This cannot tell an RG2/RG6 from an RG2-FT: the unit id is chosen by the mounting,"
        )
        lines.append(
            "    and an FT variant answers here with an incompatible map. Confirm the model in the"
        )
        lines.append(
            "    Compute Box Web Client before commanding anything, and check the width above"
        )
        lines.append("    against the jaw gap with calipers.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "unit": self.unit,
            "width_mm": self.width_mm,
            "status": int(self.status),
            "blocked": self.blocked,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "answering_units": list(self.answering_units),
            "raw": dict(self.raw),
        }


class OnRobotRG:
    """An RG2 or RG6 on a Compute Box.

        rg = OnRobotRG()
        rg.connect("192.168.1.1")
        print(rg.probe().render())      # reads only
        rg.grip(width_mm=40.0, force_n=60.0)
        rg.disconnect()

    One connection. OnRobot documents a limit of one concurrent connection, so this
    holds the socket for its lifetime and serialises every transaction. A second
    client, or a robot controller that itself speaks Modbus to the same box, is a hard
    conflict rather than a slow one.
    """

    def __init__(self, *, unit: int = UNIT_QUICK_CHANGER, timeout_s: float = 1.0) -> None:
        self.unit = int(unit)
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        self._tid = 0
        #: OnRobot documents no timeout at all. This is one second per transaction, and
        #: expiry is fail-closed, because silence is the documented symptom of every
        #: misconfiguration here.
        self._timeout_s = float(timeout_s)

    # --- lifecycle -----------------------------------------------------------------------------

    def connect(self, host: str, port: int = DEFAULT_PORT) -> None:
        """Open the socket. There is no login, no session and no activation frame in Modbus."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout_s)
        try:
            sock.connect((host, int(port)))
        except OSError as exc:
            sock.close()
            raise ModbusError(
                f"cannot reach the OnRobot Compute Box at {host}:{port}: {exc}. Find its real "
                f"address in the Web Client rather than assuming 192.168.1.1; Dynamic IP mode only "
                f"falls back to that after a 60 s DHCP timeout."
            ) from exc
        self._socket = sock

    def disconnect(self) -> None:
        """Close the socket.

        This releases nothing and stops nothing. There is no Modbus goodbye and no
        register to clear, so a gripper holding a part goes on holding it and a move in
        flight goes on moving.
        """
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None

    # --- the wire ------------------------------------------------------------------------------

    def _transact(self, pdu: bytes, *, unit: int | None = None) -> bytes:
        """One Modbus TCP request/response. Returns the response PDU without the mbap header."""
        with self._lock:
            sock = self._socket
            if sock is None:
                raise ModbusError("not connected; call connect() first")
            self._tid = (self._tid + 1) % 0x10000
            uid = self.unit if unit is None else int(unit)
            # Mbap: transaction id, protocol id (always 0), length (unit id plus PDU),
            # unit id.
            header = struct.pack(">HHHB", self._tid, 0, len(pdu) + 1, uid)
            try:
                sock.sendall(header + pdu)
                # Read the 6-byte prefix, then exactly `length` more. One recv is not
                # one frame: Modbus TCP has a length field precisely because it is a
                # stream.
                head = self._recv_exactly(sock, 6)
                tid, proto, length = struct.unpack(">HHH", head)
                if tid != self._tid or proto != 0:
                    raise ModbusError(
                        f"reply is for transaction {tid} protocol {proto}, expected {self._tid} / 0"
                    )
                rest = self._recv_exactly(sock, length)
            except socket.timeout as exc:
                raise ModbusError(
                    f"the Compute Box did not answer within {self._timeout_s}s. Silence is the "
                    f"documented symptom of every misconfiguration here: wrong unit id, Modbus not "
                    f"served by this firmware, or no tool on that Quick Changer side."
                ) from exc
            except OSError as exc:
                raise ModbusError(f"socket error: {exc}") from exc

        body = rest[1:]  # drop the echoed unit id
        if body and body[0] & 0x80:
            code = body[1] if len(body) > 1 else 0
            raise ModbusError(f"Modbus exception 0x{code:02x} for function 0x{body[0] & 0x7f:02x}")
        return body

    @staticmethod
    def _recv_exactly(sock: socket.socket, count: int) -> bytes:
        out = b""
        while len(out) < count:
            chunk = sock.recv(count - len(out))
            if not chunk:
                raise ModbusError("the Compute Box closed the connection mid-frame")
            out += chunk
        return out

    def read_registers(self, start: int, quantity: int, *, unit: int | None = None) -> list[int]:
        """FC 0x03. Reads; commands nothing."""
        pdu = struct.pack(">BHH", _FC_READ_HOLDING, int(start), int(quantity))
        body = self._transact(pdu, unit=unit)
        if len(body) < 2 or body[0] != _FC_READ_HOLDING:
            raise ModbusError(f"expected a read reply, got {body!r}")
        byte_count = body[1]
        if byte_count != quantity * 2:
            raise ModbusError(f"byte count {byte_count} does not match {quantity} register(s)")
        data = body[2:2 + byte_count]
        return [struct.unpack(">H", data[i:i + 2])[0] for i in range(0, byte_count, 2)]

    def write_registers(self, start: int, values: "list[int]") -> None:
        """FC 0x10.

        The echo means the write was accepted and nothing more. The gripper ignores a
        grip command while it is busy and still answers normally.
        """
        payload = b"".join(struct.pack(">H", int(v) & 0xFFFF) for v in values)
        pdu = struct.pack(">BHHB", _FC_WRITE_MULTIPLE, int(start), len(values), len(payload)) + payload
        body = self._transact(pdu)
        if len(body) < 5 or body[0] != _FC_WRITE_MULTIPLE:
            raise ModbusError(f"expected a write echo, got {body!r}")

    # --- the read-only door --------------------------------------------------------------------

    def probe(self, *, sweep_units: bool = True) -> RGProbeReading:
        """Read width and status, and optionally see which unit ids answer at all.

        The sweep is free and settles a question the manual answers only in prose:
        whether this tool is on the primary or the secondary side of a Dual Quick
        Changer. Reading three unit ids costs three frames, moves nothing, and replaces
        a table lookup with a measurement.
        """
        answering: list[int] = []
        if sweep_units:
            for candidate in (UNIT_QUICK_CHANGER, UNIT_DUAL_PRIMARY, UNIT_DUAL_SECONDARY):
                try:
                    self.read_registers(_REG_ACTUAL_WIDTH, 2, unit=candidate)
                    answering.append(candidate)
                except ModbusError:
                    continue
        try:
            width_raw, status_raw = self.read_registers(_REG_ACTUAL_WIDTH, 2)
        except ModbusError as exc:
            return RGProbeReading(
                reachable=False, unit=self.unit, detail=str(exc),
                answering_units=tuple(answering),
            )
        return RGProbeReading(
            reachable=True,
            unit=self.unit,
            width_mm=width_raw / 10.0,
            status=RGStatus(status_raw),
            answering_units=tuple(answering),
            raw={"width_raw": width_raw, "status_raw": status_raw},
        )

    def status(self) -> RGStatus:
        return RGStatus(self.read_registers(_REG_STATUS, 1)[0])

    def width_mm(self) -> float:
        """Actual opening in millimetres, without the fingertip offset.

        Register 275 carries the corrected value and is not adjacent, so reading it
        costs a second transaction.
        """
        return self.read_registers(_REG_ACTUAL_WIDTH, 1)[0] / 10.0

    # --- the one commanding call ---------------------------------------------------------------

    def grip(self, *, width_mm: float, force_n: float, use_fingertip_offset: bool = False) -> None:
        """Command a target width at a target force. One write of three registers.

        There is no speed argument, and that is not an omission. RG2 and RG6 have no
        speed register anywhere in the writable map, and OnRobot's own library exposes
        only a read-only `rg_get_speed`. Accepting a speed and dropping it would be a
        lie to the caller.

        Width is the opening, so a larger number is more open. Robotiq counts run the
        other way.

        This refuses when a safety switch is tripped, because the gripper is dead until
        tool power is cycled and a command sent meanwhile is discarded silently.
        """
        current = self.status()
        if current & _BLOCKING:
            raise ModbusError(
                f"refusing to command this gripper: status 0x{int(current):04x} has a safety switch "
                f"tripped. It stays dead until tool power is cycled; retrying will not help."
            )
        control = _CTRL_GRIP_WITH_OFFSET if use_fingertip_offset else _CTRL_GRIP
        self.write_registers(
            _REG_TARGET_FORCE,
            [round(force_n * 10), round(width_mm * 10), control],
        )

    def stop(self) -> None:
        """Halt a motion. Control value 8, written into the same block."""
        self.write_registers(_REG_TARGET_FORCE, [0, 0, _CTRL_STOP])
