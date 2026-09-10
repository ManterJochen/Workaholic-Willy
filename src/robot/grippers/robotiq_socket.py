r"""A dependency-free client for the Robotiq URCap gripper socket, TCP 63352 on the UR controller.

Why this exists. `robotiq_gripper.RobotiqGripper` is not a package: `pip index
versions robotiq_gripper` answers that no matching distribution was found, and the
installed `ur_rtde` ships nothing named `robotiq*`. It is a single example file from
SDU's ur_rtde repository that an operator drops on the path by hand, so a driver that
imported it would be gated behind a dependency that cannot be installed. This is about
a hundred lines of `socket` instead, and there is no SDK gate for Robotiq.

What this client deliberately does not do is millimetres or newtons. It speaks counts,
0 to 255, and nothing else, because the arithmetic does not close. The 2F-85 manual
states 0.4 mm per count over an 85 mm stroke, and 255 x 0.4 = 102 mm. The 2F-140
states 0.65 mm per count over 140 mm, and 255 x 0.65 = 166 mm. So
`mm = stroke * (255 - count) / 255` is wrong on both two-finger models, and ur_rtde
ships `auto_calibrate()` precisely because the real endpoints are a per-unit
measurement rather than 0 and 255. Force is worse: the register table gives only a
minimum and a maximum, and the 2F-140 span of 10 to 125 N is lower than the 2F-85 span
of 20 to 235 N, so the same count means a different force on a different model.
`GripperController` keeps its own millimetre map, anchored on config, and this layer
refuses to invent one.

Polarity is 0 for open and 255 for closed, and the ecosystem contradicts itself about
it. The vendor says so three times for the same register: rPR is documented as
"0x00: Open position", gPR as "0x00: Full opening", gPO as "0x00: Fully opened". But
the ur_rtde C++ header declares `UNIT_DEVICE` as 0 for open while `UNIT_NORMALIZED` is
0.0 for closed and `UNIT_PERCENT` is 0% for closed. Lifting a snippet from that
project without checking which unit mode it is in inverts the jaws. This module is
device units only, stated here so nobody corrects it against a normalised example.

The grammar is de facto rather than vendor-documented. Robotiq documents the register
semantics thoroughly and documents no string grammar for port 63352 at all. The
grammar below is taken from three readable implementations that agree: SDU's Python
client, SDU's C++ client, and URScript captured from a URCap-generated program. The
vendor article that looks like the spec is about a different interface. "How to Control
the Robotiq Gripper via Socket Communication" on blog.robotiq.com is about ports 30002
and 30003 and uploading a `gripper.script` of some 650 lines. It contains no GET or SET
grammar and no 63352. A future reader reaching for the Robotiq socket documentation
will land on the wrong protocol, so do not correct this file against it.

This has never run against hardware, which is honesty bucket 3, and the first steps
are read-only on purpose. :meth:`RobotiqSocket.probe` commands nothing and exists so
an operator's first contact answers the questions this file could not: whether the port
is even open from an external PC, since on PolyScope X it may not exist at all, and
whether the polarity above holds on this unit.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum, StrEnum
from typing import Any

__all__ = [
    "ActivationStatus",
    "ObjectStatus",
    "ProbeReading",
    "RobotiqSocket",
    "RobotiqSocketError",
]

#: The URCap daemon port on the UR controller, not on the gripper: a Robotiq 2F has no
#: IP. The daemon owns the RS-485 Modbus RTU link and mirrors the register map as named
#: variables, which is why this client computes no CRC and packs no struct.
DEFAULT_PORT = 63352

#: Exactly what a `SET` is answered with: three bytes, no trailing newline.
_ACK = b"ack"

#: The command range, not the achievable range. A real unit does not reach the
#: mechanical stops at the extremes, which is why ur_rtde ships a calibration routine.
#: `get_current_position()` on a healthy gripper may never return exactly 0 or 255, and
#: that is not a fault.
POS_OPEN = 0
POS_CLOSED = 255


#: Measured against URSim, and it is the most useful sentence in this file. With port
#: 63352 published from the container and the Robotiq URCap not installed, the TCP
#: connect succeeds and the connection dies on the first byte: Docker forwards the port
#: and nothing inside listens. It arrives as an empty read on Linux and as a reset on
#: Windows, so both paths say this.
#:
#: It earns its own sentence precisely because the connect having worked is what sends
#: an operator looking somewhere else.
_NOTHING_LISTENING = (
    "the URCap daemon accepted the connection and then closed it without answering. A TCP connect "
    "that succeeds and dies on the first byte means something is forwarding the port while nothing "
    "is listening behind it; on a UR controller, that the Robotiq_Grippers URCap is not installed "
    "or not running. Connecting is not evidence of a gripper."
)


#: `gFLT` values from the 2-finger register map. ⚠ Robotiq numbers these differently on the
#: 3-Finger and the EPick, so :meth:`RobotiqSocket.fault_text` reports an unlisted code by number
#: rather than inventing a meaning for it.
_FAULTS = {
    5: "action delayed, activation must be completed first",
    7: "activation bit not set",
    9: "communication chip not ready",
    10: "changing mode fault, the automatic release is still in progress",
    11: "automatic release completed",
    13: "activation fault, check for a mechanical obstruction",
    14: "changing mode fault, check for an obstruction",
    15: "automatic release fault, review the emergency-release procedure",
}


#: `gFLT` values from the 2-finger register map. ⚠ Robotiq numbers these differently on the
#: 3-Finger and the EPick, so :meth:`RobotiqSocket.fault_text` reports an unlisted code by number
#: rather than inventing a meaning for it.
_FAULTS = {
    5: "action delayed, activation must be completed first",
    7: "activation bit not set",
    9: "communication chip not ready",
    10: "changing mode fault, the automatic release is still in progress",
    11: "automatic release completed",
    13: "activation fault, check for a mechanical obstruction",
    14: "changing mode fault, check for an obstruction",
    15: "automatic release fault, review the emergency-release procedure",
}


class ActivationStatus(IntEnum):
    """`STA` (gSTA): where the activation routine is. 2 is documented as not used."""

    RESET = 0
    ACTIVATING = 1
    ACTIVE = 3


class ObjectStatus(IntEnum):
    """`OBJ` (gOBJ): why the fingers stopped. The only post-grasp evidence this protocol offers."""

    MOVING = 0
    #: Stopped early while opening: something is in the way.
    STOPPED_OPENING = 1
    #: Stopped early while closing: something is held. This is a successful grasp.
    STOPPED_CLOSING = 2
    #: Reached the commanded position, holding nothing: the fingers met, or arrived at
    #: the target with no obstruction. A caller that reads not moving as gripped reads
    #: this as success.
    AT_POSITION = 3


class RobotiqSocketError(RuntimeError):
    """The daemon refused, answered something unparseable, or went away."""


class _Var(StrEnum):
    """The variables this client uses.

    `MOD` is deliberately absent. It exists on the wire but is meaningless for the
    two-finger map, because it selects grasp mode on the 3-Finger and vacuum mode on
    the EPick. Sending it to a 2F is at best ignored.
    """

    ACT = "ACT"   # rACT: activate. Must stay 1; clearing it deactivates.
    ATR = "ATR"   # rATR: automatic release. Set to 0 alongside ACT 0 when resetting.
    GTO = "GTO"   # rGTO: go-to. This is what starts motion; POS/SPE/FOR alone move nothing.
    POS = "POS"   # write: rPR target. read: gPO actual. Not the same register.
    PRE = "PRE"   # gPR: the gripper's echo of the requested position.
    SPE = "SPE"   # rSP: speed. 0 is minimum speed, not stop. Use GTO 0 to stop.
    FOR = "FOR"   # rFR: force. Opaque counts; the Newton span differs per model.
    STA = "STA"   # gSTA: activation status.
    OBJ = "OBJ"   # gOBJ: why the fingers stopped.
    FLT = "FLT"   # gFLT: fault code. 0 = no fault.


@dataclass(frozen=True, slots=True)
class ProbeReading:
    """What a read-only interrogation found. Commands nothing.

    This is what turns the unknowns above into a measurement. Every claim the research
    could not source, whether the port exists from an external PC, whether 0 really
    means open on this unit, what its real endpoints are, is answered by running this,
    moving the gripper from the teach pendant, and running it again.
    """

    reachable: bool
    #: Raw variable readings, verbatim, so a reader sees what the wire said before any
    #: interpretation.
    raw: dict[str, int] = field(default_factory=dict)
    detail: str = ""

    @property
    def activated(self) -> bool:
        return self.raw.get("STA") == int(ActivationStatus.ACTIVE)

    @property
    def faulted(self) -> bool:
        return bool(self.raw.get("FLT", 0))

    @property
    def exit_code(self) -> int:
        return 0 if self.reachable and not self.faulted else 1

    def render(self) -> str:
        """The reading and what it does not prove. ASCII, no trailing newline."""
        if not self.reachable:
            return f"  UNREACHABLE: {self.detail}"
        lines = [f"  reachable on the URCap socket, {len(self.raw)} variable(s) read"]
        lines.extend(f"    {name:<5}{value}" for name, value in sorted(self.raw.items()))
        sta = self.raw.get("STA")
        lines.append(
            f"    -> {'activated' if self.activated else f'not activated (STA {sta})'}"
            f"{', FAULT ' + str(self.raw['FLT']) if self.faulted else ''}"
        )
        # The caveat is part of the reading. A number that looks plausible is exactly
        # how a wrong polarity survives to the first commanded move.
        lines.append(
            "    POS is the actual position (gPO), 0 = open and 255 = closed by the manual. To"
        )
        lines.append(
            "    confirm that on this unit: move the fingers from the pendant and read it again."
        )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reachable": self.reachable,
            "activated": self.activated,
            "faulted": self.faulted,
            "exit_code": self.exit_code,
            "detail": self.detail,
            "raw": dict(self.raw),
        }


class RobotiqSocket:
    """The five calls `GripperController` needs, spoken directly to the URCap daemon.

        client = RobotiqSocket()
        client.connect("192.168.1.10", 63352)
        print(client.probe().render())      # reads only; commands nothing
        client.activate_if_needed()          # this moves: a full-travel calibration sweep
        client.move(200, 128, 100)
        client.disconnect()

    One connection, serialised. The protocol has no request ids, so two interleaved
    commands would mismatch replies. Every exchange is held behind one lock, and both
    readable reference clients do the same.
    """

    def __init__(self, *, timeout_s: float = 2.0) -> None:
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()
        #: 2.0 s matches the SDU client socket timeout. The protocol documents none.
        self._timeout_s = float(timeout_s)

    # --- lifecycle -----------------------------------------------------------------------------

    def connect(self, hostname: str, port: int = DEFAULT_PORT) -> None:
        """Open the socket. It sends nothing: the first bytes on the wire are the caller's command.

        A successful connect is not evidence that a gripper exists. Three failures look
        different and mean different things: connection refused, which is the URCap not
        installed or Tool I/O not set to Robotiq_Grippers; a timeout, which is a wrong
        address or a firewall; and a connect where every GET times out, which is the
        daemon up with its RS-485 link to the gripper dead.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout_s)
        try:
            sock.connect((hostname, int(port)))
        except OSError as exc:
            sock.close()
            raise RobotiqSocketError(
                f"cannot reach the Robotiq URCap socket at {hostname}:{port}: {exc}. Check that the "
                f"Robotiq_Grippers URCap is installed, that Tool I/O is set to Robotiq_Grippers, and "
                f"that the RS485 URCap is not installed (they are mutually exclusive)."
            ) from exc
        self._socket = sock

    def disconnect(self) -> None:
        """Close the socket.

        Closing releases nothing and stops nothing. There is no goodbye in this
        protocol and no register to clear, so a gripper holding a part goes on holding
        it. That is deliberate: dropping a workpiece because a process exited would be
        worse.
        """
        with self._lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                finally:
                    self._socket = None

    # --- the wire ------------------------------------------------------------------------------

    def _exchange(self, line: bytes) -> bytes:
        with self._lock:
            sock = self._socket
            if sock is None:
                raise RobotiqSocketError("not connected; call connect() first")
            try:
                sock.sendall(line)
                # Read until parseable rather than once. There is no length prefix and
                # no reply terminator to rely on, so trusting a single read is trusting
                # the network to frame for you. Both reference clients do exactly that
                # and get away with it on a LAN.
                chunks: list[bytes] = []
                deadline = time.monotonic() + self._timeout_s
                while time.monotonic() < deadline:
                    chunk = sock.recv(1024)
                    if not chunk:
                        # An empty read and a reset are the same situation reached two
                        # ways, so they get the same sentence rather than one useful
                        # message and one bare one.
                        raise RobotiqSocketError(_NOTHING_LISTENING)
                    chunks.append(chunk)
                    joined = b"".join(chunks).strip()
                    if joined == _ACK or len(joined.split()) >= 2:
                        return joined
                raise RobotiqSocketError(f"no parseable reply to {line!r} within {self._timeout_s}s")
            except socket.timeout as exc:
                raise RobotiqSocketError(
                    f"the URCap daemon did not answer {line!r} within {self._timeout_s}s. The daemon "
                    f"can be up while its RS-485 link to the gripper is dead."
                ) from exc
            except ConnectionError as exc:
                # `ConnectionError`, not `ConnectionResetError`. Windows raises
                # `ConnectionAbortedError` (WinError 10053) where Linux gives a reset
                # or an empty read, so catching only the reset makes the sentence above
                # reachable on one platform and emits a bare `[WinError 10053]` on the
                # other.
                raise RobotiqSocketError(_NOTHING_LISTENING) from exc
            except OSError as exc:
                raise RobotiqSocketError(f"socket error sending {line!r}: {exc}") from exc

    def get(self, name: str) -> int:
        """`GET NAME` returns the integer. Reads; commands nothing."""
        reply = self._exchange(f"GET {name}\n".encode())
        parts = reply.decode("ascii", errors="replace").split()
        if len(parts) != 2 or parts[0] != name:
            raise RobotiqSocketError(f"GET {name} answered {reply!r}, expected '{name} <int>'")
        try:
            return int(parts[1])
        except ValueError as exc:
            raise RobotiqSocketError(f"GET {name} answered a non-integer: {reply!r}") from exc

    def set(self, **values: int) -> None:
        """`SET A 1 B 2` returns `ack`. Several variables in one line, which is how a move is atomic.

        `ack` means the daemon accepted the line. It does not mean the gripper received
        it and it does not mean anything moved. Confirming a command reached the
        gripper means polling `PRE`, its own echo of the requested position.
        """
        line = "SET" + "".join(f" {k} {int(v)}" for k, v in values.items()) + "\n"
        reply = self._exchange(line.encode())
        if reply != _ACK:
            raise RobotiqSocketError(f"{line.strip()!r} was answered {reply!r}, expected b'ack'")

    # --- the read-only door --------------------------------------------------------------------

    def probe(self) -> ProbeReading:
        """Read every status variable. Commands nothing, so it is safe as a first contact."""
        raw: dict[str, int] = {}
        for name in (_Var.STA, _Var.FLT, _Var.POS, _Var.PRE, _Var.ACT, _Var.GTO, _Var.OBJ):
            try:
                raw[str(name)] = self.get(str(name))
            except RobotiqSocketError as exc:
                return ProbeReading(reachable=False, raw=raw, detail=str(exc))
        return ProbeReading(reachable=True, raw=raw)

    # --- the five calls `GripperController` uses -----------------------------------------------

    def is_active(self) -> bool:
        """One round trip, which is what makes idempotent activation cheap rather than faked."""
        return self.get(str(_Var.STA)) == int(ActivationStatus.ACTIVE)

    def activate_if_needed(self, *, timeout_s: float = 10.0) -> None:
        """Activate unless the gripper already reports ACTIVE.

        Activation moves the fingers through their entire travel, at speed. It is a
        calibration routine and not a handshake, so nothing may be between the jaws and
        no hands near them.
        """
        if self.is_active():
            return
        self.activate(timeout_s=timeout_s)

    def activate(self, *, timeout_s: float = 10.0) -> None:
        """Reset, then activate, then wait for the routine to report ACTIVE.

        The reset is not optional: the vendor states that after a power loss `rACT`
        must be cleared and set again before the gripper will operate.
        """
        self.set(ACT=0, ATR=0)
        self._wait_for(lambda: self.get(str(_Var.ACT)) == 0 and self.get(str(_Var.STA)) == 0,
                       timeout_s=timeout_s, what="the reset to take")
        time.sleep(0.5)
        self.set(ACT=1)
        time.sleep(1.0)
        try:
            self._wait_for(
                lambda: (self.get(str(_Var.ACT)) == 1
                         and self.get(str(_Var.STA)) == int(ActivationStatus.ACTIVE)),
                timeout_s=timeout_s, what="activation to complete",
            )
        except RobotiqSocketError as exc:
            # ⛔ THE REGISTER THAT EXPLAINS IT IS ONE ROUND TRIP AWAY. A bare "timed out after 10s
            # waiting for activation to complete" sends an operator to the network; `gFLT 13` sends
            # them to the obstruction that is actually holding the fingers.
            try:
                fault = self.fault_text()
            except RobotiqSocketError:
                fault = ""
            sta = self._quiet_get(_Var.STA)
            raise RobotiqSocketError(
                f"{exc} (gSTA {sta}"
                + (f", {fault}" if fault else ", no fault reported")
                + "). Activation sweeps the fingers through their full travel, so an obstruction "
                  "between the jaws is the usual cause."
            ) from exc

    def move(self, position: int, speed: int, force: int) -> None:
        """Command a position in counts. 0 is open, 255 is closed. No millimetres here.

        A `speed` of 0 is minimum speed, not stop. The register maps 0-100 %
        onto a non-zero physical band, which on the 2F-85 is 20 to 150 mm/s. Holding
        the fingers still means sending `GTO 0`.

        All four variables go in one line, because `GTO` is what starts the motion and
        a separate line would mean a window in which the gripper moves to a stale
        target.
        """
        self.set(
            POS=max(POS_OPEN, min(POS_CLOSED, int(position))),
            SPE=max(0, min(255, int(speed))),
            FOR=max(0, min(255, int(force))),
            GTO=1,
        )

    def stop(self) -> None:
        """Clear `GTO`. It is the only way to halt a move, and `SPE 0` does not do it."""
        self.set(GTO=0)

    def get_current_position(self) -> int:
        """The actual finger position, `gPO`, in counts.

        `POS` is two registers and that is the trap. Written, it is `rPR`, the target.
        Read, it is `gPO`, where the fingers actually are. So `SET POS 200` followed by
        `GET POS` does not read back 200; it reads where the gripper got to. The echo
        of the requested value is a different variable, `PRE` (`gPR`), which is what a
        caller polls to confirm the command reached the gripper at all.
        """
        return self.get(str(_Var.POS))

    def get_requested_position(self) -> int:
        """`PRE` (`gPR`): the gripper's own echo of the last requested position."""
        return self.get(str(_Var.PRE))

    def object_status(self) -> ObjectStatus:
        """Why the fingers stopped.

        This is the only post-grasp evidence the protocol offers, and `AT_POSITION`
        means the gripper is holding nothing.
        """
        return ObjectStatus(self.get(str(_Var.OBJ)))

    def fault_code(self) -> int:
        """`FLT` (`gFLT`): 0 when healthy. ⛔ READ IT ON THE OPERATING PATH, NOT ONLY IN `probe`.

        Until 2026-09-10 this register was read by :meth:`probe` alone, and nothing calls `probe`
        during a pick. A gripper that is faulted or has been deactivated by a controller-side event
        still answers `ack` to every `SET`, because `ack` means the DAEMON accepted the line. So the
        fingers stayed where they were, every close was reported as done, and the one register that
        would have said why was never asked.
        """
        return self.get(str(_Var.FLT))

    def fault_text(self) -> str:
        """The fault as a sentence, or `""` when healthy. Vendor semantics, quoted not classified.

        ⚠ The codes are the 2-finger map; a 3-Finger or an EPick numbers them differently. Anything
        unlisted is reported by number rather than guessed at, because a wrong explanation costs more
        than a bare code.
        """
        code = self.fault_code()
        if not code:
            return ""
        return f"gFLT {code}" + (f": {_FAULTS[code]}" if code in _FAULTS else " (not in the 2-finger table)")

    def wait_for_motion(self, *, timeout_s: float = 5.0) -> ObjectStatus:
        """Block until the fingers stop, and say WHY they stopped. ⭐ THE EVIDENCE A CLOSE NEEDS.

        `move()` returns when the daemon has accepted the line. This returns when `gOBJ` has left
        `MOVING`, which is the gripper's own statement that the motion is over, together with the
        distinction that matters:

            STOPPED_CLOSING   the fingers stalled on something      -> a part is held
            AT_POSITION       the fingers reached the target        -> holding NOTHING

        ⛔ A FAULT IS CHECKED FIRST AND EVERY POLL. A faulted gripper never leaves `MOVING`, so
        without this the caller would wait out the whole deadline and then be told about a timeout
        instead of about the fault.

        Raises :class:`RobotiqSocketError` on a fault or on the deadline. Both are honest failures:
        a caller that cannot learn why the fingers stopped has no grasp evidence at all.
        """
        deadline = time.monotonic() + float(timeout_s)
        last = ObjectStatus.MOVING
        while time.monotonic() < deadline:
            fault = self.fault_text()
            if fault:
                raise RobotiqSocketError(f"the gripper reports a fault while moving: {fault}")
            last = self.object_status()
            if last is not ObjectStatus.MOVING:
                return last
            time.sleep(0.02)
        raise RobotiqSocketError(
            f"the fingers were still moving {timeout_s}s after the command (gOBJ {int(last)}). The "
            f"gripper is powered and answering, so this is a jaw that cannot reach its target: check "
            f"for an obstruction, and that the commanded speed is not below what the load needs."
        )

    # -- internals -----------------------------------------------------------------------------

    # --- internals -----------------------------------------------------------------------------

    def _quiet_get(self, name: "_Var") -> "int | str":
        """A read for an ERROR MESSAGE. Never raises: a diagnostic that fails is not a diagnostic."""
        try:
            return self.get(str(name))
        except RobotiqSocketError:
            return "unreadable"

    def _wait_for(self, predicate: "Any", *, timeout_s: float, what: str) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.01)
        raise RobotiqSocketError(f"timed out after {timeout_s}s waiting for {what}")
