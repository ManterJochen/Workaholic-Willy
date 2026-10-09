"""URConnection, the thin RTDE wrapper around a Universal Robots controller.

It uses the ``ur_rtde`` library, installed with ``pip install ur_rtde``, to talk to the
robot over the Real-Time Data Exchange protocol.

It provides:
    - the connect and disconnect lifecycle, usable as a context manager;
    - the current TCP pose, through ``get_tcp_pose``;
    - the current joint angles, through ``get_joint_positions``;
    - forward kinematics, through ``fk``;
    - inverse kinematics, through ``ik``;
    - the DH rows and the TCP the controller computes those with, ``controller_kinematics`` (read from its primary
      interface, nothing sent) and ``active_tcp_offset``;
    - the steady gate, ``wait_until_steady``, on ``isSteady`` or on the joint speeds (``robot.safety.dwell.steady_signal``);
    - the low-level move wrappers ``moveJ`` and ``moveL``;
    - the halt latch ("halt now"), and with ``robot.ur.brake_on_halt`` the watched move a halt brakes;
    - the hand-guiding primitives, teach mode, the RTDE watchdog and a fresh control
      program, which :mod:`.freedrive` puts together into a session.
"""

from __future__ import annotations

import dataclasses
import math
import re
import socket
import struct
import threading
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from ...constants import UR_CONNECTION_LOG_FILE, create_robot_logger
from ...core import RobotConnectionError
from ...core.arm_capabilities import ArmHalted, HaltState, halted_refusal

def _absent(exc: Exception) -> str:
    """The SDK is not on this machine. Installing it is the fix."""
    return f"ur_rtde is not installed. Run: pip install ur_rtde  ({exc})"


def _unloadable(exc: Exception) -> str:
    """The SDK is here and the operating system refused to load it, which is a different
    fault with a different remedy.

    ``ur_rtde`` is a native extension. A missing package raises ``ImportError`` and a
    present one whose DLL cannot be loaded raises ``OSError``, so a guard that catches
    only the first misses this case entirely. It is a real one:

        import mujoco
        OSError: [WinError 4551] Eine Anwendungssteuerungsrichtlinie hat diese Datei blockiert

    Windows Smart App Control re-evaluates an unsigned native build by reputation, days
    after it was installed. The same policy applies to the ur_rtde extension, and here
    the consequence is that the import of the UR driver dies raw, so the degrade path
    these four lines build, and the actionable message
    :meth:`URConnection.connect` raises, never run at all. That is the path a real arm
    is brought up on.

    Widening the catch is only half of it. Saying that ur_rtde is not installed and to
    run pip install is false where it is installed and blocked, and sending an operator
    to reinstall a package they already have, next to a powered arm, is worse than the
    raw OSError, which at least names the policy. So the reason is recorded here and
    reported later rather than assumed.
    """
    return (
        f"ur_rtde IS installed, and its native library could not be LOADED "
        f"({type(exc).__name__}: {exc}). This is NOT a missing package and reinstalling will not "
        f"help. On Windows this is usually Smart App Control or an application-control policy "
        f"refusing an unsigned extension; elsewhere it is a missing system library or an ABI "
        f"mismatch. Check the CodeIntegrity event log before changing anything."
    )


#: Why the move core is unusable, or empty where it loaded.
#: :meth:`URConnection.connect` reads it, so the operator gets the remedy that matches
#: their actual fault.
_CORE_UNAVAILABLE: str = ""
try:
    import rtde_control  # ur_rtde
    import rtde_receive  # ur_rtde
except ImportError as exc:
    rtde_control = None        # type: ignore[assignment]
    rtde_receive = None        # type: ignore[assignment]
    _CORE_UNAVAILABLE = _absent(exc)
except OSError as exc:
    rtde_control = None        # type: ignore[assignment]
    rtde_receive = None        # type: ignore[assignment]
    _CORE_UNAVAILABLE = _unloadable(exc)

# rtde_io, for digital and analog output, and dashboard_client, for the safety text an
# operator reads and for protective-stop recovery, ship in the same ur_rtde wheel and are
# optional to a given deployment. Each is guarded independently, so I/O and status
# degrade gracefully on a slimmer build while control and receive, the move and read
# core, still work.
#
# Their reasons are recorded rather than raised, because these two are best-effort by
# design and a blocked one degrades exactly as an absent one does. What must not happen
# is a degradation with no reason attached: the `_require_io` message asks whether the
# ur_rtde rtde_io is present, which is precisely the question these strings answer.
_IO_UNAVAILABLE: str = ""
try:
    import rtde_io  # ur_rtde
except ImportError as exc:
    rtde_io = None             # type: ignore[assignment]
    _IO_UNAVAILABLE = _absent(exc)
except OSError as exc:
    rtde_io = None             # type: ignore[assignment]
    _IO_UNAVAILABLE = _unloadable(exc)

_DASHBOARD_UNAVAILABLE: str = ""
try:
    import dashboard_client  # ur_rtde
except ImportError as exc:
    dashboard_client = None    # type: ignore[assignment]
    _DASHBOARD_UNAVAILABLE = _absent(exc)
except OSError as exc:
    dashboard_client = None    # type: ignore[assignment]
    _DASHBOARD_UNAVAILABLE = _unloadable(exc)


#: The first PolyScope version whose dashboard answers the remote-control question. A CB3 runs
#: PolyScope 3.x and has no Remote/Local control mode at all. Below 5.6, ur_rtde 1.6.5 prints
#: "Warning! isInRemoteControl() function is not supported on the dashboard server for PolyScope
#: versions less than 5.6.0" (read from the installed library's own strings) and still hands back
#: an answer, so a bool from it there is not the controller's.
_REMOTE_CONTROL_SINCE: tuple[int, int] = (5, 6)

#: The dashboard robot modes in which a controller runs no external control program, each with
#: what it means. External control needs RUNNING: the arm powered and its brakes released.
_NOT_RUNNING_MODES: dict[str, str] = {
    "POWER_OFF": "the arm is powered off",
    "BOOTING": "the arm is still powering on",
    "POWER_ON": "the arm is still powering on",
    "IDLE": "the arm is powered on and its brakes are still engaged",
}


def _rotation_matrix(rotvec: "Sequence[float]") -> list[list[float]]:
    """The rotation of an axis-angle vector (Rodrigues), in plain Python: this module imports no numpy."""
    x, y, z = (float(v) for v in rotvec)
    angle = math.sqrt(x * x + y * y + z * z)
    if angle < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    kx, ky, kz = x / angle, y / angle, z / angle
    c, s = math.cos(angle), math.sin(angle)
    t = 1.0 - c
    return [
        [c + kx * kx * t, kx * ky * t - kz * s, kx * kz * t + ky * s],
        [ky * kx * t + kz * s, c + ky * ky * t, ky * kz * t - kx * s],
        [kz * kx * t - ky * s, kz * ky * t + kx * s, c + kz * kz * t],
    ]


def _turn_between(a: "Sequence[float]", b: "Sequence[float]") -> float:
    """The angle, in rad, of the rotation between two axis-angle orientations."""
    ra, rb = _rotation_matrix(a), _rotation_matrix(b)
    # trace(ra^T rb) = 1 + 2 cos(angle)
    trace = sum(ra[i][j] * rb[i][j] for i in range(3) for j in range(3))
    return math.acos(max(-1.0, min(1.0, (trace - 1.0) / 2.0)))


def _polyscope_version(dash: object) -> tuple[int, int] | None:
    """The (major, minor) PolyScope version a dashboard reports, or ``None`` where it will not say.

    The dashboard answers with text such as ``URSoftware 3.15.4.106291 (...)`` on a CB3 and
    ``URSoftware 5.11.1.108318 (...)`` on an e-Series. An answer with no version in it, and a
    dashboard that raises, are no evidence.
    """
    try:
        text = dash.polyscopeVersion()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 (cannot ask => no version)
        return None
    match = re.search(r"(\d+)\.(\d+)", text) if isinstance(text, str) else None
    return (int(match.group(1)), int(match.group(2))) if match else None


def _answers_remote_control(dash: object) -> bool:
    """Whether this dashboard's answer to ``isInRemoteControl`` is the controller's own.

    Only a controller that says it runs PolyScope 5.6 or later. A version that cannot be read
    answers ``False`` here, because an answer nobody can vouch for is not one to act on.
    """
    version = _polyscope_version(dash)
    return version is not None and version >= _REMOTE_CONTROL_SINCE


def _robot_mode_diagnosis(dash: object) -> tuple[str, str] | None:
    """The dashboard's robot mode and what it means, where it is one that runs no external program.

    ``None`` for RUNNING, for a mode this does not name, and where the dashboard will not say.
    """
    try:
        text = dash.robotmode()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 (cannot ask => cannot claim)
        return None
    if not isinstance(text, str):
        return None
    mode = text.split(":", 1)[-1].strip().upper()
    meaning = _NOT_RUNNING_MODES.get(mode)
    return (text.strip(), meaning) if meaning is not None else None


#: The dashboard server's port, on a CB3 and an e-Series alike.
_DASHBOARD_PORT = 29999


def _ask_dashboard(host: str, command: str, *, port: int = _DASHBOARD_PORT, timeout_s: float = 2.0) -> str:
    """One dashboard command over a connection of its own, answered by one line.

    Its own connection, never the driver's shared one, so the question cannot interleave with a read another thread
    is making. Raises ``OSError`` (a timeout included) where the dashboard does not answer.
    """
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.settimeout(timeout_s)
        _read_line(sock)  # the banner: "Connected: Universal Robots Dashboard Server"
        sock.sendall(command.encode("ascii") + b"\n")
        return _read_line(sock)


def _read_line(sock: socket.socket) -> str:
    """One line from the dashboard; a connection that closes or says nothing in time raises ``OSError``."""
    data = b""
    while not data.endswith(b"\n") and len(data) < 4096:
        chunk = sock.recv(256)
        if not chunk:
            raise OSError("the dashboard closed the connection")
        data += chunk
    return data.decode("utf-8", errors="replace").strip()


#: What :meth:`URConnection.wait_until_steady` may read (``robot.safety.dwell.steady_signal``).
STEADY_SIGNALS: tuple[str, ...] = ("is_steady", "joint_speeds")
#: How many reads in a row, :data:`_POLL_S` apart, every joint has to be slower than :data:`_STILL_RAD_S` for the arm to
#: count as steady from its joint speeds.
_QUIET_READS = 3

#: Where the controller's kinematics are read, in order: the primary client interface's read-only port (CB3 3.x and
#: e-Series alike; URSim CB3 3.15.8 answers there, measured), then the primary interface itself. Either is only read:
#: nothing is ever sent on it, because a primary interface runs any URScript it is sent.
_PRIMARY_PORTS: tuple[int, ...] = (30011, 30001)
#: The primary interface's robot state message, and its kinematics info sub-package. The controller sends that
#: sub-package in the first robot state after a client connects, and in none after it (URSim CB3 3.15.8, measured).
_ROBOT_STATE = 16
_KINEMATICS_INFO = 5
#: The kinematics info is six uint32 joint checksums, six doubles each of DH theta, a, d and alpha, and a uint32
#: calibration status: 220 bytes on a CB3.
_KINEMATICS_INFO_BYTES = 6 * 4 + 24 * 8
#: How many robot states without the kinematics info are read before the read gives up, and how many bytes at most.
_ROBOT_STATES_READ = 2
_PRIMARY_BYTES_READ = 1 << 20


@dataclasses.dataclass(frozen=True, slots=True)
class ControllerKinematics:
    """The DH rows a UR controller computes its kinematics with, as its primary interface reports them.

    The nominal table of the arm plus the arm's own factory calibration: on a calibrated arm every row differs from the
    support page's by a little, a theta offset included (URSim CB3 3.15.8 reports its nominal UR10 table plus exactly
    what a ``calibration.conf`` adds, measured). Lengths in metres, angles in radians, the joint variable added to
    ``theta_rad``. ``checksums`` are the six joint checksums the controller reports beside them (URSim: ``0xFFFFFFFF``
    each), ``calibration_status`` its status word, and ``port`` where they were read.
    """

    theta_rad: tuple[float, ...]
    a_m: tuple[float, ...]
    d_m: tuple[float, ...]
    alpha_rad: tuple[float, ...]
    checksums: tuple[int, ...]
    calibration_status: int | None
    port: int

    def render(self) -> str:
        """The rows and the checksums as one log line reads them."""
        def row(values: "tuple[float, ...]") -> str:
            return "[" + ", ".join(f"{v:.9g}" for v in values) + "]"

        status = "not reported" if self.calibration_status is None else str(self.calibration_status)
        return (f"theta {row(self.theta_rad)}, a {row(self.a_m)}, d {row(self.d_m)}, alpha {row(self.alpha_rad)}; "
                f"joint checksums {', '.join(f'0x{c:08X}' for c in self.checksums)}; calibration status {status}")


def _kinematics_info(data: bytes, port: int) -> ControllerKinematics:
    """The kinematics info sub-package's payload, after its five header bytes; ``ValueError`` where it is not one."""
    if len(data) < _KINEMATICS_INFO_BYTES:
        raise ValueError(f"the kinematics info holds {len(data)} bytes, fewer than the {_KINEMATICS_INFO_BYTES} it needs")
    checksums = struct.unpack_from(">6I", data, 0)
    values = struct.unpack_from(">24d", data, 24)
    if not all(math.isfinite(v) for v in values):
        raise ValueError("the kinematics info holds a DH value that is not finite")
    status = struct.unpack_from(">I", data, _KINEMATICS_INFO_BYTES)[0] if len(data) >= _KINEMATICS_INFO_BYTES + 4 else None
    return ControllerKinematics(
        theta_rad=tuple(values[0:6]), a_m=tuple(values[6:12]), d_m=tuple(values[12:18]), alpha_rad=tuple(values[18:24]),
        checksums=tuple(int(c) for c in checksums), calibration_status=status, port=port,
    )


def _read_kinematics_info(host: str, port: int, *, timeout_s: float = 2.0) -> ControllerKinematics:
    """The kinematics info of the first robot state a primary interface at ``host:port`` sends; nothing is sent to it.

    Raises ``OSError`` (a timeout included) where the port does not answer or closes, and ``ValueError`` where what it
    sends is no message stream, or holds no readable kinematics info in its first robot states.
    """
    deadline = time.monotonic() + timeout_s
    states = 0
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        data = b""
        while True:
            # Every complete message so far; a robot state's sub-packages are length-prefixed like the messages.
            while len(data) >= 5:
                size, kind = struct.unpack_from(">iB", data, 0)
                if size < 5 or size > _PRIMARY_BYTES_READ:
                    raise ValueError(f"port {port} sent a message {size} bytes long, which no UR primary interface does")
                if len(data) < size:
                    break
                message, data = data[5:size], data[size:]
                if kind != _ROBOT_STATE:
                    continue
                states += 1
                at = 0
                while at + 5 <= len(message):
                    part, part_kind = struct.unpack_from(">iB", message, at)
                    if part < 5 or at + part > len(message):
                        raise ValueError(f"port {port} sent a robot state whose sub-package {part_kind} is garbled")
                    if part_kind == _KINEMATICS_INFO:
                        return _kinematics_info(message[at + 5:at + part], port)
                    at += part
                if states >= _ROBOT_STATES_READ:
                    raise ValueError(f"port {port} sent {states} robot states and no kinematics info in them")
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise OSError(f"port {port} sent no kinematics info within {timeout_s:.1f} s")
            sock.settimeout(remaining)
            chunk = sock.recv(65536)
            if not chunk:
                raise OSError(f"port {port} closed the connection before it sent its kinematics info")
            data += chunk


if TYPE_CHECKING:  # pragma: no cover (import only for static analysis)
    from collections.abc import Callable, Sequence

    from dashboard_client import DashboardClient
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface


class MoveEnd(StrEnum):
    """How the last move a connection was asked for ended, for the sentence a refused verb says.

    Read by the thread that asked for the move, right after it returned, never across threads.
    """

    #: No move was asked for yet.
    NONE = "none"
    #: The move was sent as it always was, and the answer is ur_rtde's own (brakes off, or a caller's asynchronous move).
    SENT = "sent"
    #: A watched move stood at its target: joints within 2e-3 rad, a line's TCP within 1 mm and 2e-3 rad.
    ARRIVED = "arrived"
    #: The halt latch was set: nothing was sent.
    REFUSED_HALTED = "refused_halted"
    #: The halt braked the move in flight (stopJ or stopL) and the arm stood still.
    BRAKED = "braked"
    #: The halt's brake was sent and the arm was not seen to stand still, or the brake itself failed.
    BRAKE_UNCONFIRMED = "brake_unconfirmed"
    #: A watched move ended away from its target with nobody braking it.
    ENDED_SHORT = "ended_short"
    #: A protective or emergency stop, or the end of the control program, ended a watched move.
    STOPPED = "stopped"
    #: The controller refused to start the move, or never showed it running; a move it never showed was stopped.
    NOT_STARTED = "not_started"


#: The ends of a move the halt explains: a verb that meets one answers CANCELLED, not a controller refusal.
HALT_ENDS: frozenset[MoveEnd] = frozenset({MoveEnd.REFUSED_HALTED, MoveEnd.BRAKED, MoveEnd.BRAKE_UNCONFIRMED})

_JOINT, _LINE = "moveJ", "moveL"
#: How often a watched move is read: one RTDE cycle of a CB3, which publishes at 125 Hz.
_POLL_S = 0.008
#: How long a sent move may go unseen in the controller's async operation register before it is stopped and refused.
_SEEN_WITHIN_S = 1.0
#: How long a finished operation may take to stand within the arrival tolerance: a servo trails its set point.
_SETTLE_S = 0.5
#: A joint move has arrived with every joint within this of its target, in rad.
_ARRIVED_RAD = 2e-3
#: A line has arrived with its TCP within this of the target, in mm, and turned within :data:`_ARRIVED_TURN_RAD`.
_ARRIVED_MM = 1.0
_ARRIVED_TURN_RAD = 2e-3
#: The weakest deceleration a brake uses, in rad/s^2 for stopJ and m/s^2 for stopL; a move that accelerated harder is
#: braked as hard as it accelerated. 2.0 is ur_rtde's own stopJ default.
_BRAKE_MIN_DECEL = 2.0
#: A joint slower than this, in rad/s, stands still; a braked arm is waited for at most :data:`_STILL_WAIT_S`.
_STILL_RAD_S = 0.01
_STILL_WAIT_S = 2.0


class URConnection:
    """Manage the RTDE connection to a UR robot.

    Parameters:
        ip:           The IP address of the robot controller.
        vel:          The default joint velocity in rad/s.
        acc:          The default joint acceleration in rad/s^2.
        frequency:    The RTDE exchange frequency in Hz. 0 means the default, 125 Hz.
        brake_on_halt: ``robot.ur.brake_on_halt``. Off: every move is the synchronous ur_rtde call it always was, and a
                      halt latches. On: a move is sent asynchronously and watched, and a halt brakes it (:meth:`moveJ`).
        steady_signal: ``robot.safety.dwell.steady_signal``, what :meth:`wait_until_steady` reads: ``"is_steady"``, the
                      controller's ``isSteady`` as always, or ``"joint_speeds"``, every joint slower than 0.01 rad/s on
                      three reads in a row 8 ms apart.
        clock, sleep: The monotonic clock and the sleep a watched move polls with; a test hands in its own. The clock is
                      ``perf_counter``: ``time.monotonic`` ticks in 15.6 ms steps on Windows, two RTDE cycles.

    The halt latch (:meth:`request_halt`) lives on this object, so a disconnect and a connect keep it: only
    :meth:`clear_halt` ends it. While it is set no move is sent and no output is switched.
    """

    def __init__(
        self,
        ip: str,
        vel: float = 1.0,
        acc: float = 0.5,
        frequency: float = 0.0,
        *,
        brake_on_halt: bool = False,
        steady_signal: str = "is_steady",
        clock: "Callable[[], float]" = time.perf_counter,
        sleep: "Callable[[float], None]" = time.sleep,
    ):
        # An ur_rtde import failure is deferred to connect(), so this module imports on
        # a machine without the native library.
        self.ip = ip
        self.vel = vel
        self.acc = acc
        self.frequency = frequency
        self.brake_on_halt = bool(brake_on_halt)
        if steady_signal not in STEADY_SIGNALS:
            raise ValueError(f"steady_signal is one of {', '.join(STEADY_SIGNALS)}, got {steady_signal!r}")
        self.steady_signal = steady_signal
        self.logger = create_robot_logger("URConnection", UR_CONNECTION_LOG_FILE)
        if steady_signal == "joint_speeds":
            self.logger.info("The steady gate reads the joint speeds (robot.safety.dwell.steady_signal: joint_speeds): "
                             "every joint under %.2f rad/s on %d reads %.0f ms apart, isSteady where they cannot be read.",
                             _STILL_RAD_S, _QUIET_READS, _POLL_S * 1000.0)

        self._ctrl: RTDEControlInterface | None = None
        self._recv: RTDEReceiveInterface | None = None
        self._io: RTDEIOInterface | None = None
        self._dashboard: DashboardClient | None = None
        #: Cached controller TCP offset for :meth:`_fk_tcp_offset`; cleared on every (dis)connect.
        self._tcp_offset_cache: list[float] | None = None
        #: What :meth:`controller_kinematics` and :meth:`active_tcp_offset` read, each settled once per connection (an
        #: answer, or ``None`` with the reason logged); cleared on every (dis)connect, the TCP also on a fresh program.
        #: The epoch counts the clearings, so a read that ends after one keeps nothing.
        self._kinematics: ControllerKinematics | None = None
        self._kinematics_settled = False
        self._active_tcp: list[float] | None = None
        self._active_tcp_settled = False
        self._kinematics_epoch = 0
        self._kinematics_lock = threading.Lock()
        #: The serial :meth:`controller_serial` read, whether that read is settled (a serial, or an answer that is
        #: none), and how often its own dashboard connection failed; all cleared on every (dis)connect.
        self._serial: str | None = None
        self._serial_settled = False
        self._serial_failures = 0
        self._clock = clock
        self._sleep = sleep
        #: The halt latch: the flag a watched move polls, and the record :meth:`halt_state` reads. Written under the
        #: lock, by :meth:`request_halt`, :meth:`clear_halt` and the thread whose move a halt found in flight, once that
        #: move ended; read without it.
        self._halt = threading.Event()
        self._halt_state: HaltState | None = None
        self._halt_lock = threading.Lock()
        #: How many times the latch was set: a path reads it to learn of a halt that came and was cleared mid-leg.
        self._halt_requests = 0
        self._halt_requested_mono = 0.0
        #: Whether a move is in flight now: set by the thread that sent it, under the halt lock with the latch's check,
        #: and read by the one that halts.
        self._in_motion = False
        #: When the brake of the move in flight saw the arm stand still, on :attr:`_clock`; ``None`` until it did.
        self._still_at: float | None = None
        self._last_end = MoveEnd.NONE
        #: What ended the last watched move the controller stopped: ``protective``, ``emergency`` or ``program``.
        self._last_stop_cause = ""

    # ------------------------------------------------------------------
    # The halt latch
    # ------------------------------------------------------------------

    def request_halt(self, reason: str) -> HaltState:
        """Latch: no move is sent and no output switched from now on, until :meth:`clear_halt`. Never raises.

        Nothing is sent from the calling thread, which is usually not the one moving the arm: ur_rtde runs a
        synchronous move in the control script's main loop, and a ``stopJ`` from another thread waits behind it. The
        move in flight is the moving thread's to end: braked under control with ``brake_on_halt``, run to its end
        without. A second request keeps the first record.
        """
        with self._halt_lock:
            if self._halt_state is not None:
                return self._halt_state
            state = HaltState(reason=str(reason).strip() or "halt requested", requested_at=time.time(),
                              in_motion=self._in_motion)
            self._halt_requests += 1
            self._halt_requested_mono = self._clock()
            self._halt_state = state
            self._halt.set()
        self.logger.warning("Halt requested (%s); %s.", state.reason,
                            "the move in flight is braked by the thread that sent it" if state.in_motion
                            and self.brake_on_halt else "the move in flight runs to its end and nothing after it is "
                            "sent" if state.in_motion else "no move is in flight")
        return state

    def clear_halt(self) -> None:
        """End the latch: moves and outputs are sent again. The console calls it once a person confirmed the cell."""
        with self._halt_lock:
            was = self._halt_state
            self._halt.clear()
            self._halt_state = None
        if was is not None:
            self.logger.info("Halt cleared (%s).", was.reason)

    def halt_state(self) -> HaltState | None:
        """The latch while it is set, else ``None``: one attribute read, never a lock, never a raise."""
        return self._halt_state

    @property
    def halted(self) -> bool:
        """Whether the latch is set."""
        return self._halt.is_set()

    @property
    def halt_requests(self) -> int:
        """How many times the latch was set since this connection was built; a clear does not count, nor a second press.

        A caller that runs several moves in a row (a judged path's waypoints) takes it before the first and ends where it
        changed: a halt that came and was cleared while one move ran still ends the moves after it.
        """
        return self._halt_requests

    @property
    def in_motion(self) -> bool:
        """Whether a move is in flight now, as the thread that sent it says."""
        return self._in_motion

    @property
    def last_move_end(self) -> MoveEnd:
        """How the last move this connection was asked for ended; for the thread that asked."""
        return self._last_end

    @property
    def last_stop_cause(self) -> str:
        """What ended the last watched move the controller stopped (:attr:`MoveEnd.STOPPED`): ``protective``,
        ``emergency`` or ``program`` (the control program no longer ran); ``""`` before any such stop."""
        return self._last_stop_cause

    def refuse_if_halted(self, what: str) -> bool:
        """``True`` where the latch is set: ``what`` is not sent, and counts as a move the halt refused.

        For a caller that would read the controller before sending ``what`` (``MotionController``): it refuses first,
        so nothing at all reaches the controller, and :attr:`last_move_end` says why rather than how an earlier move
        ended.
        """
        if not self._halt.is_set():
            return False
        self._refused_while_halted(what)
        return True

    def _refused_while_halted(self, what: str) -> bool:
        self._last_end = MoveEnd.REFUSED_HALTED
        state = self._halt_state
        self.logger.warning("%s not sent: %s.", what, state.render() if state is not None else "the arm is halted")
        return False

    def _begin_move(self, what: str, *, in_flight: bool) -> bool:
        """``True`` where ``what`` may be sent now; ``False`` where the latch refused it, with nothing sent.

        The latch's check and the mark of a move in flight are one step under the halt lock, right before the send: a
        halt either came first and nothing is sent, or finds the move in flight and records it so. Without the step a
        halt landing between the check and the send let a move go out that its record called not in flight.
        ``in_flight`` is for a move that blocks until it ends (a synchronous or a watched one); a caller's own
        asynchronous move is not marked.
        """
        with self._halt_lock:
            refused = self._halt.is_set()
            if not refused and in_flight:
                self._in_motion = True
                self._still_at = None
        if refused:
            self._refused_while_halted(what)
        return not refused

    def _end_move(self, *, unconfirmed: bool = False) -> None:
        """The move :meth:`_begin_move` marked is over: nothing is in flight, and a halt that found it in flight records
        what became of it, once, from how it ended (:attr:`last_move_end`).

        ``unconfirmed`` is a move that was stopped where nobody could wait for it to stand (a fault ended its watch, or a
        synchronous move raised). Under the halt lock with the mark's clearing, so a halt either found the move in
        flight and is told how it ended, or came after it and found nothing in flight. A cleared latch stays cleared.
        """
        with self._halt_lock:
            self._in_motion = False
            state = self._halt_state
            if state is None or state.brake != "pending":
                return
            end = self._last_end
            if unconfirmed or end is MoveEnd.BRAKE_UNCONFIRMED:
                self._halt_state = dataclasses.replace(state, in_motion=True, brake="unconfirmed")
            elif end is MoveEnd.BRAKED:
                still = self._still_at
                brake_s = max(0.0, still - self._halt_requested_mono) if still is not None else None
                self._halt_state = dataclasses.replace(state, in_motion=True, braked=True, brake_s=brake_s,
                                                       brake="braked")
            else:
                self._halt_state = dataclasses.replace(state, in_motion=True, brake="ran_out")

    def _refuse_output_while_halted(self, what: str) -> None:
        state = self._halt_state
        if state is not None:
            raise ArmHalted(halted_refusal(state.reason, f"{what} was not switched, and nothing was sent"))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        if rtde_control is None or rtde_receive is None:
            # The recorded reason rather than a fixed sentence: a package that is not
            # installed and one that is installed and refused by the OS need different
            # actions, and this is the line an operator reads.
            raise ImportError(_CORE_UNAVAILABLE or _absent(ImportError("ur_rtde")))
        if self.is_connected:
            self.logger.debug("connect() called but already connected; ignored.")
            return
        self._tcp_offset_cache = None
        self._forget_the_kinematics()
        self.logger.info("Connecting to UR robot at %s ...", self.ip)
        try:
            self._ctrl = self._open_control()
            self._recv = rtde_receive.RTDEReceiveInterface(self.ip)
        except Exception as exc:  # noqa: BLE001 (roll back a partial connect, then reraise the transport fault)
            # Roll back a partially-opened connection.
            self.logger.error("Failed to connect to %s: %s", self.ip, exc)
            self._safe_teardown()
            # Name the cause where it can be proved. `RTDEControlInterface` is built
            # first above, and on a controller in local mode it fails with a failure to
            # start RTDE data synchronization before timeout, which says nothing about
            # the actual problem. Measured against URSim 5.26.0: the control script is
            # uploaded and visible in the PolyScope log, but a controller in local mode
            # never runs an externally-sent program, so the synchronisation that script
            # would set up never starts.
            #
            # It is claimed only where the dashboard answers false. A dashboard that is
            # absent or unhappy re-raises the original fault rather than blaming a mode
            # nobody verified.
            blocked = self._diagnose_connect_failure()
            if blocked is not None:
                raise RobotConnectionError(f"{blocked} The underlying transport error was: {exc}") from exc
            raise
        # I/O and the dashboard are best-effort extras, covering digital and analog
        # output, the safety text an operator reads, and protective-stop recovery. A
        # failure here does not tear down the move core, and the capability wrappers
        # below report the extra as unavailable instead.
        #
        # One bucket-3 item for the first real-UR bring-up: RTDEIOInterface and
        # RTDEControlInterface share the RTDE register space, and on some controller
        # configurations a concurrent output write can clash unless both use
        # `use_upper_range_registers`.
        if rtde_io is not None:
            try:
                self._io = rtde_io.RTDEIOInterface(self.ip)
            except Exception as exc:  # noqa: BLE001 (I/O is optional; log and continue without it)
                self.logger.warning("RTDEIOInterface unavailable (%s); digital/analog I/O disabled.", exc)
        else:
            # A capability that disappears with no reason is what sends somebody hunting
            # a controller fault for a package that was never imported.
            self.logger.warning("Digital/analog I/O disabled: %s", _IO_UNAVAILABLE)
        if dashboard_client is not None:
            try:
                self._dashboard = dashboard_client.DashboardClient(self.ip)
                self._dashboard.connect()
            except Exception as exc:  # noqa: BLE001 (dashboard is optional; log and continue without it)
                self.logger.warning("DashboardClient unavailable (%s); safety text/recovery disabled.", exc)
                self._dashboard = None
        else:
            self.logger.warning("Safety text and protective-stop recovery disabled: %s",
                                _DASHBOARD_UNAVAILABLE)
        self.logger.info("Connected (control + receive%s%s).",
                         " + io" if self._io is not None else "",
                         " + dashboard" if self._dashboard is not None else "")

    def _open_control(self) -> RTDEControlInterface:
        """A new control interface, whose constructor uploads the control script and returns once it runs."""
        assert rtde_control is not None  # connect() refused a missing ur_rtde before this
        # 0.0 is the not-configured sentinel of this stack and not of ur_rtde. The
        # schema ships `ur.rtde_frequency: 0.0` with `ge=0.0`, meaning the controller
        # decides; the ur_rtde sentinel for the same thing is -1.0, and it takes 0.0
        # literally as zero hertz. Measured against URSim 5.26.0, one variable at a
        # time:
        #
        #     frequency=0.0   -> failed in 6.1 s: "Failed to start RTDE data synchronization"
        #     frequency=-1.0  -> connected in 0.1 s
        #     frequency=500.0 -> connected in 0.6 s
        #
        # Without this translation the shipped default could never connect to a real
        # UR, and would fail with a message naming nothing, next to a powered arm.
        # Translating the sentinel here rather than changing the schema default keeps
        # every existing config loadable and puts the vendor convention inside the
        # driver boundary, which is where this repo keeps one.
        frequency = float(self.frequency) if float(self.frequency) > 0.0 else -1.0
        return rtde_control.RTDEControlInterface(
            self.ip, frequency,
        )

    def disconnect(self) -> None:
        """Idempotent: safe to call repeatedly, and after a failed connect()."""
        self._safe_teardown()
        self.logger.info("Disconnected.")

    def _safe_teardown(self) -> None:
        """Best-effort release of both RTDE interfaces; never raises."""
        self._tcp_offset_cache = None
        self._forget_the_kinematics()
        self._serial, self._serial_settled, self._serial_failures = None, False, 0
        if self._ctrl is not None:
            try:
                self._ctrl.stopScript()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown: log and continue, never raise)
                self.logger.warning("stopScript() failed: %s", exc)
            try:
                self._ctrl.disconnect()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown: log and continue, never raise)
                self.logger.warning("control.disconnect() failed: %s", exc)
            self._ctrl = None
        if self._recv is not None:
            try:
                self._recv.disconnect()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown: log and continue, never raise)
                self.logger.warning("receive.disconnect() failed: %s", exc)
            self._recv = None
        if self._io is not None:
            try:
                self._io.disconnect()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown: log and continue, never raise)
                self.logger.warning("io.disconnect() failed: %s", exc)
            self._io = None
        if self._dashboard is not None:
            try:
                self._dashboard.disconnect()
            except Exception as exc:  # noqa: BLE001 (best-effort teardown: log and continue, never raise)
                self.logger.warning("dashboard.disconnect() failed: %s", exc)
            self._dashboard = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *exc):
        self.disconnect()

    @property
    def is_connected(self) -> bool:
        return self._ctrl is not None and self._recv is not None

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_tcp_pose(self) -> list[float]:
        """The current TCP pose as ``[x, y, z, rx, ry, rz]``.

        The translations are metres and the rotation is axis-angle in radians.
        """
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return self._recv.getActualTCPPose()

    def get_joint_positions(self) -> list[float]:
        """Current joint angles ``[q0 ... q5]`` in radians."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return self._recv.getActualQ()

    # ------------------------------------------------------------------
    # Kinematics helpers
    # ------------------------------------------------------------------

    #: A TCP offset of exactly zero cannot be passed to ``getForwardKinematics``:
    #: measured, that call fails, and on one attempt it hung until its timeout. One
    #: nanometre in tool Z is the smallest value that works and is physically nil, being
    #: 1e-9 m against a tolerance measured in millimetres.
    _FK_EPSILON_OFFSET: "tuple[float, ...]" = (0.0, 0.0, 1e-9, 0.0, 0.0, 0.0)

    def _fk_tcp_offset(self) -> list[float]:
        """The TCP offset handed to :meth:`fk` explicitly, so it never reads a stale register.

        It is explicit because ``getForwardKinematics(q)`` without an offset reads the
        controller float registers, which moveJ and moveL also write, so after any
        commanded motion it returns a pose wrong by hundreds of millimetres. That is the
        mechanism GitLab #348 describes, still present in 1.6.5. Measured across three
        consecutive move and return cycles:

            q-only form          1113.207 mm wrong, every time
            explicit offset          0.000 mm, every time

        It is the controller own offset rather than zero, because
        ``getForwardKinematics(q, offset)`` returns FK with that offset applied, so
        passing something near zero would return the flange pose. On URSim the
        controller carries no tool and the two coincide, which is exactly the
        coincidence that hides a defect until a real cell sets a TCP on the pendant.
        Reading the actual offset keeps this method returning the TCP, which is what
        every caller and the ``RobotArm`` Protocol mean by a pose.

        It is read once per connection and cached: the tool register does not change
        mid-session, and this sits on the motion path where a second round trip per FK
        call is not free.
        """
        if self._tcp_offset_cache is None:
            assert self._ctrl is not None  # guaranteed by the caller's _require_connected()
            try:
                offset = [float(v) for v in self._ctrl.getTCPOffset()]
            except Exception as exc:  # noqa: BLE001 (fall back to the epsilon rather than to a stale register)
                self.logger.warning(
                    "getTCPOffset() failed (%s); using a 1 nm offset so FK still bypasses the "
                    "stale-register path. A controller with a TCP set would report the FLANGE here.",
                    exc,
                )
                offset = list(self._FK_EPSILON_OFFSET)
            if not any(offset):
                offset = list(self._FK_EPSILON_OFFSET)
            self._tcp_offset_cache = offset
        return list(self._tcp_offset_cache)

    def fk(self, joint_positions: list[float]) -> list[float]:
        """Forward kinematics from joints to the TCP pose ``[x, y, z, rx, ry, rz]``.

        It uses the built-in FK of the controller, ``getForwardKinematics(q, tcp_offset)``,
        and always passes the TCP offset explicitly (:meth:`_fk_tcp_offset`). The q-only form
        reads the offset from the controller float registers, which moveJ and moveL also
        write, so after any commanded move it returned a pose off by hundreds of millimetres,
        the mechanism GitLab #348 describes. With the offset passed it measured 0.000 mm
        across three consecutive move and return cycles, so FK of an arbitrary q, as the
        singularity guard, ``move_home`` and MotionController ask it, is not corrupted by the
        motion before it.

        Where FK of the current joints is meant, :meth:`fk_current` is still the call: it asks
        the controller for the TCP it stands at and needs no offset at all.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return self._ctrl.getForwardKinematics(joint_positions, self._fk_tcp_offset())

    def fk_current(self) -> list[float]:
        """The controller current TCP, through ``getForwardKinematics()`` with no arguments.

        It is deliberately separate from :meth:`fk`, because the two are not equally
        trustworthy and the difference is the difference between a cell that reconnects
        and one that refuses to. Measured against URSim 5.26.0 with ur_rtde 1.6.5,
        comparing both against ``getActualTCPPose()`` with the arm steady:

            state                    fk(q)          fk()  (no argument)
            fresh controller         0.000 mm       0.000 mm
            after one moveJ        656.780 mm       0.000 mm
            after further motion  1000.093 mm       0.000 mm

        ``getForwardKinematics(q)`` shares the controller float registers with the motion
        commands, so any moveJ or moveL leaves values there that the next q-form call
        reads as a TCP offset. GitLab #348 describes the mechanism: the registers can
        hold values other than a zero tool offset, because other functions use the same
        ones. The no-argument form does not touch them and was correct in every state
        tried.

        The consequence reproduces end to end: the first ``connect()`` verifies the tool
        frame at 0.00 mm, the cell moves once, and the next connect is refused with a
        delta of over a metre, turning a perfectly good arm away on the second run.

        Use this wherever FK of the joints the robot is at right now is meant. It is not
        a substitute for :meth:`fk`, which answers a different question, FK of an
        arbitrary q, and has no immune equivalent.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return list(self._ctrl.getForwardKinematics())

    def ik(
        self,
        tcp_pose: list[float],
        q_near: list[float] | None = None,
    ) -> list[float]:
        """Inverse kinematics from a TCP pose to joint angles.

        Parameters:
            tcp_pose: ``[x, y, z, rx, ry, rz]``
            q_near:   Optional seed joints, for nearest-solution selection.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        if q_near is not None:
            return self._ctrl.getInverseKinematics(tcp_pose, q_near)
        return self._ctrl.getInverseKinematics(tcp_pose)

    def controller_kinematics(self) -> ControllerKinematics | None:
        """The DH rows this controller computes its kinematics with, or ``None`` where they cannot be read.

        Read once per connection from the kinematics info of the first robot state the controller's primary client
        interface sends (:data:`_PRIMARY_PORTS`: the read-only port first, then the primary one), over a connection of
        its own that is only ever read: nothing is sent, nothing moves, no setting changes. They are the rows its
        ``getForwardKinematics`` and ``getInverseKinematics`` compute on, so a solver here on them answers what the
        controller answers (``robot.safety.ik_quality.line_ik``, ``singularity_fk``). Both the rows and a failure are
        logged once and kept until the next connect; a failure keeps every caller on the controller's own answers.
        """
        self._require_connected()
        with self._kinematics_lock:
            if self._kinematics_settled:
                return self._kinematics
            epoch = self._kinematics_epoch
            read: ControllerKinematics | None = None
            tried: list[str] = []
            for port in _PRIMARY_PORTS:
                try:
                    read = _read_kinematics_info(self.ip, port)
                    break
                except (OSError, ValueError, struct.error) as exc:
                    tried.append(f"port {port}: {exc}")
            if epoch != self._kinematics_epoch:
                return None  # a disconnect came while it was read: nothing is kept, and nothing vouched for
            self._kinematics, self._kinematics_settled = read, True
            if read is None:
                self.logger.warning("The controller's kinematics could not be read from its primary interface (%s); "
                                    "whatever asks for them asks the controller instead.", "; ".join(tried))
            else:
                self.logger.info("The controller's kinematics, read from %s:%d (read only, nothing sent): %s.", self.ip,
                                 read.port, read.render())
            return read

    def active_tcp_offset(self) -> list[float] | None:
        """The controller's active TCP offset, ``[x, y, z, rx, ry, rz]`` in metres and a rotation vector, exactly as
        ``getTCPOffset`` reports it, zero included; ``None`` where it cannot be read.

        The TCP the controller's inverse and forward kinematics apply to a pose with no TCP of its own. Read once per
        connection and once per fresh control program, and kept: a failure as well, logged once. Unlike
        :meth:`_fk_tcp_offset`, a zero offset stays zero here.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        with self._kinematics_lock:
            if self._active_tcp_settled:
                return None if self._active_tcp is None else list(self._active_tcp)
            epoch = self._kinematics_epoch
            offset: list[float] | None = None
            try:
                answer = [float(v) for v in self._ctrl.getTCPOffset()]
                offset = answer if len(answer) == 6 and all(math.isfinite(v) for v in answer) else None
                why = f"it answered {answer!r}"
            except Exception as exc:  # noqa: BLE001 (no offset read => nothing here solves as the controller does)
                why = str(exc)
            if epoch != self._kinematics_epoch:
                return None  # a disconnect or a fresh program came while it was read
            self._active_tcp, self._active_tcp_settled = offset, True
            if offset is None:
                self.logger.warning("The controller's active TCP offset could not be read (%s); whatever needs it "
                                    "asks the controller instead.", why)
            return None if offset is None else list(offset)

    @property
    def kinematics_epoch(self) -> int:
        """How often what :meth:`controller_kinematics` and :meth:`active_tcp_offset` read was forgotten (a connect, a
        disconnect, a fresh program): whatever is checked against those reads is checked again once it moved on."""
        return self._kinematics_epoch

    def _forget_the_kinematics(self, *, tcp_only: bool = False) -> None:
        """Read the active TCP, and unless ``tcp_only`` the kinematics, again on the next ask: a new connection may be
        another controller, and a fresh program another TCP.

        Without the lock, so a teardown never waits behind a read: the epoch it counts up makes a read that ends after it
        keep nothing.
        """
        self._kinematics_epoch += 1
        if not tcp_only:
            self._kinematics, self._kinematics_settled = None, False
        self._active_tcp, self._active_tcp_settled = None, False

    # ------------------------------------------------------------------
    # Motion primitives
    # ------------------------------------------------------------------

    def moveJ(
        self,
        joint_positions: list[float],
        vel: float | None = None,
        acc: float | None = None,
        asynchronous: bool = False,
    ) -> bool:
        """A joint-space move.

        It returns ``True`` when a synchronous move completes, or an asynchronous one
        starts. While the halt latch is set it returns ``False`` and sends nothing.

        With ``brake_on_halt`` off, as shipped, the call is the one it always was: ur_rtde's synchronous ``moveJ``,
        whose answer is returned. On, a synchronous move is sent asynchronously and watched by this thread
        (:meth:`_move_and_wait`): a halt brakes it, and ``True`` comes only with the arm at the target.
        :attr:`last_move_end` says how it ended.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        if self.refuse_if_halted(_JOINT):
            return False
        if self.brake_on_halt and not asynchronous:
            return self._move_and_wait(_JOINT, [float(q) for q in joint_positions], v, a)
        return self._send_as_always(_JOINT, self._ctrl.moveJ, joint_positions, v, a, asynchronous)

    def moveL(
        self,
        tcp_pose: list[float],
        vel: float | None = None,
        acc: float | None = None,
        asynchronous: bool = False,
    ) -> bool:
        """A linear, meaning Cartesian, move.

        ``tcp_pose`` is ``[x, y, z, rx, ry, rz]`` in metres with an axis-angle rotation. The halt latch and
        ``brake_on_halt`` work as for :meth:`moveJ`; a line is braked with ``stopL``, along the line.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        if self.refuse_if_halted(_LINE):
            return False
        if self.brake_on_halt and not asynchronous:
            return self._move_and_wait(_LINE, [float(p) for p in tcp_pose], v, a)
        return self._send_as_always(_LINE, self._ctrl.moveL, tcp_pose, v, a, asynchronous)

    def _send_as_always(self, kind: str, send: "Callable[..., bool]", target: "Sequence[float]", v: float, a: float,
                        asynchronous: bool) -> bool:
        """The move as ur_rtde always took it: one call, its answer returned (``brake_on_halt`` off, or the caller's own
        asynchronous move). The latch is asked again in one step with the mark of the move in flight, right before it."""
        if not self._begin_move(kind, in_flight=not asynchronous):
            return False
        self._last_end = MoveEnd.SENT
        if asynchronous:
            return send(target, v, a, asynchronous)
        ended = False
        try:
            answer = send(target, v, a, asynchronous)
            ended = True
            return answer
        finally:
            self._end_move(unconfirmed=not ended)

    # ------------------------------------------------------------------
    # The watched move (robot.ur.brake_on_halt)
    # ------------------------------------------------------------------
    #
    # ur_rtde's control script runs a synchronous move in its main loop, so nothing reaches it until the move ends: a
    # stopJ sent from another thread waits behind it (measured on the CB3 URSim, scripts/ursim/probe_halt.py M0). An
    # asynchronous move runs in a thread of the script instead, a stopJ or stopL kills that thread and decelerates
    # along the line it ran, and the script writes the move's progress into an output register: an operation id that
    # changes at every start, and a running bit. So a watched move is sent asynchronously and read every 8 ms by the
    # thread that sent it, and that thread brakes it, so no two threads ever drive one control interface.

    def _move_and_wait(self, kind: str, target: "list[float]", v: float, a: float) -> bool:
        """Send ``kind`` to ``target`` asynchronously and watch it: ``True`` only at the target, ``False`` otherwise.

        A halt brakes it (:meth:`_brake`); a protective or emergency stop or the end of the control program ends it
        (``False``, the arm wherever the controller stopped it); a move the register never shows within
        :data:`_SEEN_WITHIN_S` is stopped and refused. Whatever else ends the watch, a read that raises, a garbled
        register, a ``KeyboardInterrupt`` in the thread that moves the arm: the move may still run in the controller with
        nobody watching it, so it is stopped where it can be, before anything says it is no longer in flight, and the
        fault goes on, as a synchronous move's raise did.
        """
        ctrl = self._ctrl
        assert ctrl is not None  # the caller checked the connection
        before = ctrl.getAsyncOperationProgressEx().operationId()
        # The latch again, in one step with the mark: a halt during the read above refuses the send.
        if not self._begin_move(kind, in_flight=True):
            return False
        self._last_end = MoveEnd.NOT_STARTED
        watched = False
        try:
            send = ctrl.moveJ if kind == _JOINT else ctrl.moveL
            if not send(target, v, a, True):
                self.logger.error("%s was not started: the controller refused the asynchronous send.", kind)
                watched = True
                return False
            self._last_end = MoveEnd.SENT
            answer = self._watch(kind, target, a, before)
            watched = True
            return answer
        except BaseException:
            self._stop_quietly(kind, a)
            raise
        finally:
            self._end_move(unconfirmed=not watched)

    def _watch(self, kind: str, target: "list[float]", a: float, before: int) -> bool:
        ctrl = self._ctrl
        assert ctrl is not None
        sent_at = self._clock()
        seen = False
        while True:
            if self._halt.is_set():
                return self._brake(kind, a)
            if self._stopped_by_the_controller(kind):
                return False
            status = ctrl.getAsyncOperationProgressEx()
            # Only once the operation this move started shows: until then the register can still read the move
            # before, which had finished, and that is no arrival.
            seen = seen or status.operationId() != before
            if seen and not status.isAsyncOperationRunning():
                return self._settle(kind, target)
            if not seen and self._clock() - sent_at > _SEEN_WITHIN_S:
                self._stop_quietly(kind, a)
                self._last_end = MoveEnd.NOT_STARTED
                self.logger.error("%s never showed in the controller's async operation register within %.1f s: "
                                  "it was stopped, and is refused.", kind, _SEEN_WITHIN_S)
                return False
            self._sleep(_POLL_S)

    def _stopped_by_the_controller(self, kind: str) -> bool:
        """Whether a protective or emergency stop, or the end of the control program, ended the watched move.

        The stops are asked before the program, because a protective stop ends the control program as well and is the
        cause: :attr:`last_stop_cause` says which, so a probe can tell a protective stop from a program that died.
        """
        ctrl, recv = self._ctrl, self._recv
        assert ctrl is not None and recv is not None
        if recv.isProtectiveStopped():
            cause, why = "protective", "the controller is in a protective stop"
        elif recv.isEmergencyStopped():
            cause, why = "emergency", "the controller is in an emergency stop"
        elif not ctrl.isProgramRunning():
            cause, why = "program", "the control program is no longer running"
        else:
            return False
        self._last_end = MoveEnd.STOPPED
        self._last_stop_cause = cause
        self.logger.error("%s ended where the controller stopped it: %s.", kind, why)
        return True

    def _settle(self, kind: str, target: "list[float]") -> bool:
        """The move's operation finished: ``True`` once the arm stands within the arrival tolerance, else ``False``."""
        deadline = self._clock() + _SETTLE_S
        while True:
            off = self._off_target(kind, target)
            if off is None:
                self._last_end = MoveEnd.ARRIVED
                return True
            if self._stopped_by_the_controller(kind):
                return False
            if self._clock() >= deadline:
                self._last_end = MoveEnd.ENDED_SHORT
                self.logger.error("%s finished %s from its target: it did not arrive.", kind, off)
                return False
            self._sleep(_POLL_S)

    def _off_target(self, kind: str, target: "list[float]") -> str | None:
        """How far the arm stands from ``target``, or ``None`` within the arrival tolerance."""
        recv = self._recv
        assert recv is not None
        if kind == _JOINT:
            here = [float(v) for v in recv.getActualQ()]
            if len(here) != len(target):
                return f"{len(here)} joints read for {len(target)} targets"
            worst = max((abs(h - t) for h, t in zip(here, target)), default=0.0)
            return None if worst <= _ARRIVED_RAD else f"{worst:.4f} rad"
        tcp = [float(v) for v in recv.getActualTCPPose()]
        if len(tcp) != 6 or len(target) != 6:
            return "a pose that is not six numbers"
        mm = 1000.0 * math.dist(tcp[:3], target[:3])
        turn = _turn_between(tcp[3:], target[3:])
        return None if mm <= _ARRIVED_MM and turn <= _ARRIVED_TURN_RAD else f"{mm:.2f} mm and {turn:.4f} rad"

    def _brake(self, kind: str, a: float) -> bool:
        """Brake the move in flight under control, from this thread, and wait until the arm stands still.

        ``stopJ`` for a joint move and ``stopL`` for a line, each decelerating along the line it ran, at max(2.0, the
        move's own acceleration). The latch records the brake once the move is over (:meth:`_end_move`): braked, with
        the time from the request until the arm stood, or unconfirmed. A clear meanwhile leaves the latch cleared.
        """
        ctrl = self._ctrl
        assert ctrl is not None
        decel = max(_BRAKE_MIN_DECEL, float(a))
        stop = ctrl.stopJ if kind == _JOINT else ctrl.stopL
        try:
            stop(decel)
        except (RuntimeError, OSError) as exc:
            self._last_end = MoveEnd.BRAKE_UNCONFIRMED
            self.logger.error("The halt's %s raised (%s): whether the arm stopped is not known.",
                              "stopJ" if kind == _JOINT else "stopL", exc)
            return False
        still = self._wait_still()
        self._last_end = MoveEnd.BRAKED if still else MoveEnd.BRAKE_UNCONFIRMED
        if still:
            self._still_at = self._clock()
            self.logger.warning("%s braked by the halt at %.1f, standing still %.3f s after the request.", kind,
                                decel, self._still_at - self._halt_requested_mono)
        else:
            self.logger.error("%s braked by the halt at %.1f, and the arm was not seen to stand still within %.1f s.",
                              kind, decel, _STILL_WAIT_S)
        return False

    def _wait_still(self) -> bool:
        """Whether every joint stands still within :data:`_STILL_WAIT_S`; a read that fails says it did not."""
        recv = self._recv
        assert recv is not None
        deadline = self._clock() + _STILL_WAIT_S
        while True:
            try:
                fastest = max((abs(float(v)) for v in recv.getActualQd()), default=0.0)
            except (RuntimeError, OSError):
                return False
            if fastest <= _STILL_RAD_S:
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(_POLL_S)

    def _stop_quietly(self, kind: str, a: float) -> None:
        """Stop a move that cannot be watched, best effort: a failure is logged, never raised over the first fault.

        At max(2.0, the move's own acceleration), the halt's deceleration. A connection dropped meanwhile has nothing
        to stop through, and the first fault goes on.
        """
        ctrl = self._ctrl
        if ctrl is None:
            return
        try:
            (ctrl.stopJ if kind == _JOINT else ctrl.stopL)(max(_BRAKE_MIN_DECEL, float(a)))
        except Exception as exc:  # noqa: BLE001 (best effort on a path that already failed)
            self.logger.error("Stopping a move that could not be watched failed too: %s", exc)

    def stop(self, deceleration: float = 2.0) -> None:
        """Send ``stopJ`` and then ``stopL`` from the calling thread. Best effort; nothing it meets is re-raised.

        It stops an asynchronous move of the caller's own. It does not stop a synchronous move another thread is in:
        the control script runs that move in its main loop and reads the stop only after it ended, which is why the
        arm's ``stop()`` latches the halt instead (:meth:`request_halt`), and why a move ``brake_on_halt`` is to brake
        is sent asynchronously. ``scripts/ursim/probe_halt.py`` M0 measures it.

        Parameters:
            deceleration: The rate, in rad/s^2 for stopJ and m/s^2 for stopL.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        try:
            self._ctrl.stopJ(deceleration)
        except Exception as exc:  # noqa: BLE001 (best-effort stop: log and try the next stop variant)
            self.logger.warning("stopJ failed: %s", exc)
        try:
            self._ctrl.stopL(deceleration)
        except Exception as exc:  # noqa: BLE001 (best-effort stop: log and continue)
            self.logger.warning("stopL failed: %s", exc)

    def is_steady(self) -> bool:
        """True when the robot has finished its current motion."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return self._ctrl.isSteady()

    def set_payload(
        self,
        mass_kg: float,
        cog_mm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Push the payload mass and centre of gravity to the controller over RTDE.

        It calls ``RTDEControlInterface.setPayload(mass_kg, cog_m)``. The centre of
        gravity is converted from millimetres to metres at the boundary, because the
        vendor-neutral units here are millimetres and kilograms while ur_rtde expects
        metres and kilograms.

        An error from the controller is surfaced verbatim, and the Python
        ``rtde_control`` binding tends to raise :class:`RuntimeError` on a protocol
        fault. A caller treats an exception here as a hard failure: it means the
        protective-stop calculations of the controller do not reflect the configured
        payload.

        Parameters
        ----------
        mass_kg
            The tool or payload mass in kilograms. It is at least 0, and the controller
            refuses a negative value.
        cog_mm
            The centre-of-gravity offset from the flange in millimetres, converted to
            metres for the ``setPayload`` call.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        if mass_kg < 0.0:
            raise ValueError(
                f"set_payload: mass_kg must be >= 0; got {mass_kg}"
            )
        cog_m = (
            float(cog_mm[0]) * 1e-3,
            float(cog_mm[1]) * 1e-3,
            float(cog_mm[2]) * 1e-3,
        )
        self.logger.info(
            "Pushing payload to controller: mass=%.3f kg cog_mm=%s",
            mass_kg, cog_mm,
        )
        self._ctrl.setPayload(float(mass_kg), list(cog_m))

    def wait_until_steady(
        self,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> bool:
        """Block until the robot is steady, or until ``timeout_s`` elapses.

        It is what a pipeline that needs a still frame after a move uses instead of a
        blind ``time.sleep(settle_time_s)``. Polling at about 50 Hz keeps the wait close
        to the real settle time, and it never returns before the controller reports
        ``isSteady``.

        With :attr:`steady_signal` ``"joint_speeds"`` (``robot.safety.dwell.steady_signal``) the joint
        speeds decide instead, read every 8 ms (:meth:`_steady_from_joint_speeds`), and ``isSteady``
        only where they cannot be read; the timeout and its ``False`` are the same.

        Parameters
        ----------
        timeout_s : float
            The longest wait in seconds. It is at least 0, and a value at or below 0
            returns immediately with the current ``is_steady()`` value (with joint
            speeds, after the three reads that make one).
        poll_interval_s : float
            The sleep between polls of ``isSteady``, capped at ``timeout_s`` for a short wait.

        Returns
        -------
        bool
            ``True`` where the robot became steady before the timeout, and ``False``
            where the timeout fired first.
        """
        import time as _time

        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()

        if self.steady_signal == "joint_speeds":
            steady = self._steady_from_joint_speeds(timeout_s)
            if steady is not None:
                return steady

        if timeout_s <= 0:
            return bool(self._ctrl.isSteady())

        interval = max(1e-3, min(poll_interval_s, timeout_s))
        deadline = _time.monotonic() + float(timeout_s)
        while True:
            if self._ctrl.isSteady():
                return True
            if _time.monotonic() >= deadline:
                self.logger.warning(
                    "wait_until_steady timed out after %.2fs.", timeout_s,
                )
                return False
            _time.sleep(interval)

    def _steady_from_joint_speeds(self, timeout_s: float) -> bool | None:
        """Steady from the joint speeds (``robot.safety.dwell.steady_signal: joint_speeds``); ``None`` where they cannot
        be read, and ``isSteady`` decides as it always did.

        Steady is every joint slower than :data:`_STILL_RAD_S`, the brake's own stillness, on :data:`_QUIET_READS` reads
        in a row :data:`_POLL_S` apart: a local read of what the receive interface last got, no round trip, where
        ``isSteady`` is a script command of 33 ms (URSim CB3, measured) polled every 20 ms. Only a sample the controller
        sent since the read before counts (its timestamp moved on), so a receive stream that stopped never reads as an arm
        that stands. A wait of at most 0 asks whether the arm is steady now: its three reads, 16 ms, the first that is
        not quiet a no, and no for a stream that sends nothing new within twice that. A longer wait reads until the arm
        is steady or ``timeout_s`` is over, and times out as the controller's signal does.
        """
        recv = self._recv
        assert recv is not None  # the caller checked the connection
        deadline = self._clock() + (float(timeout_s) if timeout_s > 0 else 2.0 * _QUIET_READS * _POLL_S)
        quiet = 0
        last: float | None = None
        while True:
            try:
                speeds = [abs(float(v)) for v in recv.getActualQd()]
                stamp = float(recv.getTimestamp())
            # AttributeError too: a receive interface without these reads (an older binding, a stand-in) is one whose
            # speeds cannot be read, and isSteady decides, as the switch promises (2026-10-09, the desk bench).
            except (AttributeError, RuntimeError, OSError, TypeError, ValueError) as exc:
                self.logger.warning("The joint speeds could not be read (%s); isSteady decides whether the arm is "
                                    "steady.", exc)
                return None
            if len(speeds) != 6 or not all(math.isfinite(v) for v in [*speeds, stamp]):
                self.logger.warning("The joint speeds read %r at %r, not six numbers and a time; isSteady decides "
                                    "whether the arm is steady.", speeds, stamp)
                return None
            if last is None or stamp > last:
                last = stamp
                quiet = quiet + 1 if max(speeds) <= _STILL_RAD_S else 0
                if quiet >= _QUIET_READS:
                    return True
                if timeout_s <= 0 and quiet == 0:
                    return False
            if self._clock() >= deadline:
                if timeout_s > 0:
                    self.logger.warning("wait_until_steady timed out after %.2fs (joint speeds).", timeout_s)
                return False
            self._sleep(_POLL_S)

    # ------------------------------------------------------------------
    # Digital / analog I/O  (RTDEIOInterface set-side + RTDEReceiveInterface read-side)
    # ------------------------------------------------------------------
    # UR groups digital pins into banks: standard 0 to 7, configurable 0 to 7, tool 0 to
    # 1. The set side takes a per-bank id through a bank-specific method, and the read
    # side uses a global id, standard 0 to 7, configurable 8 to 15 and tool 16 to 17.
    # ``_global_din_id`` bridges that asymmetry.
    _PORT_BASE = {"standard": 0, "configurable": 8, "tool": 16}

    def set_digital_out(self, pin: int, value: bool, port: str = "standard") -> None:
        """Drive a digital output pin (bank ``port``) to ``value``. Refused with ``ArmHalted`` while halted.

        A toggle hand on the owner's tool DO0 moves its jaws at every change of the output, so a halted arm switches
        none: nothing is sent, and the refusal says why.
        """
        self._refuse_output_while_halted(f"{port} digital output {pin}")
        self._write_digital_out(pin, value, port)

    def end_output_pulse(self, pin: int, port: str = "standard") -> None:
        """Drive ``pin`` low to end a pulse that began before a halt: the one output write a halted arm makes.

        A bistable valve's coil, or a vacuum blow-off, is pulsed high and must go back low: a coil left energised until
        the cell is cleared is what cooks it. A halt that lands inside the pulse refuses every other write, so the pulse's
        own end comes through here, and only ever as low. Never for a toggle hand, whose every change of the output moves
        the jaws: its writes go through :meth:`set_digital_out` and are refused while halted. Unhalted it is the plain
        write of low.
        """
        state = self._halt_state
        if state is not None:
            self.logger.warning("%s digital output %d driven low to end a pulse that began before the halt (%s).", port,
                                pin, state.reason)
        self._write_digital_out(pin, False, port)

    def _write_digital_out(self, pin: int, value: bool, port: str) -> None:
        self._require_io()
        assert self._io is not None  # guaranteed by _require_io()
        if port == "standard":
            self._io.setStandardDigitalOut(pin, value)
        elif port == "configurable":
            self._io.setConfigurableDigitalOut(pin, value)
        elif port == "tool":
            self._io.setToolDigitalOut(pin, value)
        else:
            raise ValueError(f"set_digital_out: unknown port {port!r}")

    def set_analog_out(self, pin: int, value: float, current: bool = False) -> None:
        """Set an analog output: voltage (V) unless ``current=True`` (A). Refused with ``ArmHalted`` while halted."""
        self._refuse_output_while_halted(f"analog output {pin}")
        self._require_io()
        assert self._io is not None  # guaranteed by _require_io()
        if current:
            self._io.setAnalogOutputCurrent(pin, value)
        else:
            self._io.setAnalogOutputVoltage(pin, value)

    def get_digital_in(self, pin: int, port: str = "standard") -> bool:
        """Read a digital input pin's level (bank ``port``)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.getDigitalInState(self._global_din_id(pin, port)))

    def get_digital_out(self, pin: int, port: str = "standard") -> bool:
        """Read back a digital output pin's commanded level (bank ``port``)."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.getDigitalOutState(self._global_din_id(pin, port)))

    def _global_din_id(self, pin: int, port: str) -> int:
        base = self._PORT_BASE.get(port)
        if base is None:
            raise ValueError(f"digital I/O: unknown port {port!r}")
        return base + int(pin)

    # ------------------------------------------------------------------
    # Force / torque
    # ------------------------------------------------------------------

    def get_tcp_force(self) -> list[float]:
        """The generalized force and torque at the TCP, ``[Fx,Fy,Fz,Tx,Ty,Tz]`` in N and Nm, BASE frame."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return list(self._recv.getActualTCPForce())

    def get_joint_torques(self) -> list[float]:
        """The torque at each joint, ``[t0 ... t5]`` in newton-metres, from the control interface."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return list(self._ctrl.getJointTorques())

    # ------------------------------------------------------------------
    # Robot / safety status  (RTDEReceiveInterface + DashboardClient)
    # ------------------------------------------------------------------

    def get_robot_mode(self) -> int:
        """The raw UR robot-mode integer. The driver holds the RobotMode mapping."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getRobotMode())

    def get_safety_mode(self) -> int:
        """The raw UR safety-mode integer. The driver holds the SafetyMode mapping."""
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getSafetyMode())

    def is_protective_stopped(self) -> bool:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.isProtectiveStopped())

    def is_emergency_stopped(self) -> bool:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return bool(self._recv.isEmergencyStopped())

    def get_safety_status_bits(self) -> int:
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return int(self._recv.getSafetyStatusBits())

    def dashboard_safety_status(self) -> str:
        """The controller safety text an operator reads, empty where the dashboard is unavailable."""
        if self._dashboard is None:
            return ""
        try:
            return str(self._dashboard.safetystatus()).strip()
        except Exception as exc:  # noqa: BLE001 (dashboard is best-effort: return no text on a hiccup)
            self.logger.warning("dashboard.safetystatus() failed: %s", exc)
            return ""

    def controller_model(self) -> str | None:
        """What the controller says it is, such as ``'UR3'``, or ``None`` where it will not say.

        It is a dashboard read, so it costs a socket round trip and never sits on the
        motion path. It exists because ``ur.model`` keys the safety DH chain, the
        exact-mesh collision bundle and the cuRobo robot config, and nothing else in
        this driver asks the controller whether it agrees.

        ``None`` is returned for every failure, including an unparseable answer, and a
        caller treats it as no evidence rather than as a mismatch. Absence of evidence
        is the one thing a fail-closed check must not act on here: refusing a good cell
        at connect is not a safe failure but a cell that will not start.
        """
        if self._dashboard is None:
            return None
        try:
            model = str(self._dashboard.getRobotModel()).strip()
        except Exception as exc:  # noqa: BLE001 (best-effort: no answer is not a wrong answer)
            self.logger.warning("dashboard.getRobotModel() failed: %s", exc)
            return None
        return model or None

    def controller_serial(self) -> str | None:
        """The controller serial number, or ``None``. Best-effort, on the contract above.

        ur_rtde's ``getSerialNumber()`` refuses PolyScope below 5.6, a CB3 among them; there the dashboard's own
        "get serial number" is asked over a connection of its own (MEASURED 2026-10-02: URSim CB3 3.15.8 answers
        ``2018309999``). Only digits are a serial. A settled answer is kept for the connection, because the console
        asks on every status read and a serial does not change while connected; a connection that failed is tried
        again on the next read.
        """
        if self._dashboard is None:
            return None
        if self._serial_settled:
            return self._serial
        try:
            answer = str(self._dashboard.getSerialNumber()).strip()
        except Exception as exc:  # noqa: BLE001 (below PolyScope 5.6 ur_rtde refuses; the dashboard is asked itself)
            self.logger.debug("dashboard.getSerialNumber() refused (%s); asking 'get serial number' itself.", exc)
            answer = ""
        if not answer.isdigit():
            try:
                answer = _ask_dashboard(self.ip, "get serial number")
            except OSError as exc:
                self._serial_failures += 1
                # Once per connection at warning level: the console asks on every status read.
                log = self.logger.warning if self._serial_failures == 1 else self.logger.debug
                log("The dashboard's 'get serial number' failed: %s", exc)
                return None
        self._serial = answer if answer.isdigit() else None
        self._serial_settled = True
        if self._serial is None:
            self.logger.warning("The controller answers no serial number (%r); no claim is made about it.", answer)
        return self._serial

    def _diagnose_connect_failure(self) -> str | None:
        """Name why the control interface refused, where a dashboard can prove it.

        It returns ``None`` otherwise. Both causes below produce the same nameless
        transport error, either a failure to start RTDE data synchronization or a
        failure to start the control script before timeout, and neither is derivable
        from it. Both are measured against URSim, and both are things a real cell will
        do:

        * the controller is in local mode, so it never runs an externally-sent program;
        * the controller is in a protective stop, for instance because the last run
          ended in one and nobody cleared it before restarting the cell.

        A third is named from the dashboard's robot mode: an arm powered off, still powering
        on, or powered with its brakes engaged runs no external program either. That one is
        not measured here; it is the mode a CB3 is in after a power cycle.

        The local-mode question is asked only of a controller that says it runs PolyScope 5.6
        or later. A CB3 (PolyScope 3.x) has no Remote/Local mode, and below 5.6 ur_rtde's
        answer is not the controller's, so there the diagnosis never claims local mode.

        It is claimed only where the dashboard answers. A dashboard that is absent or
        unhappy returns ``None`` and the caller re-raises the original fault, because
        being unable to ask is not an answer of no, and sending an operator to fix a
        state that was never wrong is worse than an opaque error: it looks like an
        answer.
        """
        if dashboard_client is None:
            return None
        dash = None
        try:
            dash = dashboard_client.DashboardClient(self.ip)
            dash.connect()
            try:
                safety = str(dash.safetystatus()).strip().upper()
            except Exception:  # noqa: BLE001
                safety = ""
            if "PROTECTIVE_STOP" in safety or "EMERGENCY" in safety or "VIOLATION" in safety:
                return (
                    f"Could not open the UR control interface at {self.ip}: the controller reports "
                    f"{safety!r}. A stopped controller will not start an external control program, so "
                    "every motion this cell commands would fail the same way. Clear the stop where "
                    "the arm is VISIBLE; that is a deliberate design decision in this stack, which "
                    "is why nothing here offers to clear it for you. Then connect again."
                )
            mode = _robot_mode_diagnosis(dash)
            if mode is not None:
                reported, meaning = mode
                return (
                    f"Could not open the UR control interface at {self.ip}: the controller reports "
                    f"{reported!r}, so {meaning}. External control runs only with the arm powered on and "
                    "its brakes released (robot mode RUNNING). On the teach pendant, where the arm is "
                    "VISIBLE, power it on and release the brakes, then connect again."
                )
            if not _answers_remote_control(dash):
                return None
            try:
                if not bool(dash.isInRemoteControl()):
                    return (
                        f"Could not open the UR control interface at {self.ip}: the controller is in "
                        "LOCAL control mode. External control (script upload, RTDE control, "
                        "dashboard power/brake/protective-stop commands) requires REMOTE control. "
                        "On the teach pendant: hamburger menu -> Settings -> System -> Remote Control "
                        "-> Enable, then switch the top-right selector from Local to Remote. Note what "
                        "that costs: in Remote mode the pendant is locked for motion (no jogging, no "
                        "freedrive, no manual program start) because ISO requires a single source of "
                        "control; switch back to Local to teach."
                    )
            except Exception:  # noqa: BLE001 (cannot ask => cannot claim)
                return None
            return None
        except Exception:  # noqa: BLE001
            return None
        finally:
            if dash is not None:
                try:
                    dash.disconnect()
                except Exception:  # noqa: BLE001 (teardown of a diagnostic must not mask the fault)
                    pass

    def is_in_remote_control(self) -> bool | None:
        """Is the controller in remote control? ``None`` where no dashboard can answer.

        It is three-valued on purpose. ``False`` means the controller said so and
        external control will be refused. ``None`` means nobody could be asked, which is
        a different fact and must not be reported as local, because a caller that
        refuses on an unanswered question refuses cells that are perfectly fine.

        A controller below PolyScope 5.6, every CB3 among them, answers ``None`` as well:
        it has no dashboard answer to this question, and a CB3 has no Remote/Local mode.
        So does one whose version cannot be read.
        """
        if self._dashboard is None:
            return None
        if not _answers_remote_control(self._dashboard):
            return None
        try:
            return bool(self._dashboard.isInRemoteControl())
        except Exception as exc:  # noqa: BLE001 (an unanswerable question is not a "no")
            self.logger.warning("dashboard.isInRemoteControl() failed: %s", exc)
            return None

    def _remote_control_is_off(self) -> bool:
        """``True`` only where a throwaway dashboard connection proves the controller is in local mode.

        It is used on the connect-failure path, where ``self._dashboard`` does not exist
        yet: the control interface is built before the dashboard, so the diagnosis opens
        its own.
        """
        if dashboard_client is None:
            return False
        dash = None
        try:
            dash = dashboard_client.DashboardClient(self.ip)
            dash.connect()
            if not _answers_remote_control(dash):
                return False
            return not bool(dash.isInRemoteControl())
        except Exception:  # noqa: BLE001 (cannot ask => cannot claim)
            return False
        finally:
            if dash is not None:
                try:
                    dash.disconnect()
                except Exception:  # noqa: BLE001 (teardown of a diagnostic must not mask the fault)
                    pass

    def unlock_protective_stop(self) -> bool:
        """Ask the dashboard to release a protective stop; ``True`` if a dashboard is connected."""
        if self._dashboard is None:
            return False
        try:
            self._dashboard.closeSafetyPopup()
            self._dashboard.unlockProtectiveStop()
            return True
        except Exception as exc:  # noqa: BLE001 (surface the failure as "not recovered", never raise)
            self.logger.warning("unlockProtectiveStop() failed: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Hand guiding: teach mode, the RTDE watchdog, the control script
    # ------------------------------------------------------------------
    # The primitives :class:`~.freedrive.URFreedriveSession` is made of; the order they run in, and
    # why, is that module's. Read from the installed ur_rtde 1.6.5 and measured on the CB3 URSim
    # (PolyScope 3.15.8 in Docker under WSL, 2026-09-24):
    #
    # * ``teachMode``/``endTeachMode`` and not ``freedriveMode``. The control script ur_rtde uploads
    #   prefixes every freedrive_mode line with ``$5.10``, so a controller below PolyScope 5.10, every
    #   CB3 among them, gets a script without them and ``freedriveMode()`` does nothing there.
    #   teach_mode() has no version prefix and runs on the CB3 3.15 the owner's cell has.
    # * Teach mode lives in the control program: a program that ends (a stop on the pendant, a
    #   protective stop, a new control interface, a reconnect) ends it and the arm holds.
    # * ``setWatchdog`` asks the script for ``rtde_set_watchdog("input_int_register_0", f, "stop")``:
    #   every command the control interface sends updates that register, and a program that sees no
    #   update for 1/f seconds or a little more is stopped, which ends teach mode (at 2 Hz: 0.5 s on a
    #   bare ur_rtde connection, 1.0 s on this one with its I/O and dashboard clients attached). That
    #   stop is a protective stop. It stays armed for the life of the program, so a later blocking
    #   moveJ, which sends nothing while it runs, would trip it; a fresh program is what clears it.
    #   ``setWatchdog(0)`` does not disarm it: it protective-stopped the arm 0.05 s later. The I/O
    #   interface drives tool DO0 as before on the fresh program.
    # * A fresh program comes from a fresh control interface, not from ``reuploadScript``. On a control
    #   interface a few seconds old ``reuploadScript`` answered ``True`` and no program ran, with teach
    #   mode and without it, after a ``stopScript`` and without one, and no later upload on that
    #   interface ran either (it still ran 1 s after connecting without teach mode, and not 2 s after;
    #   once teach mode had been used, 0.8 s was already too late). ``reconnect()`` hung for over a
    #   minute. A new interface ran its program every time, about 2 s after the stop, and the payload
    #   the controller compensates for outlived the old program.

    def teach_mode(self) -> bool:
        """Free the arm for a person to move by hand (``teachMode``). ``True`` where the controller took it."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return bool(self._ctrl.teachMode())

    def end_teach_mode(self) -> bool:
        """Put the arm back under position control where it stands (``endTeachMode``)."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return bool(self._ctrl.endTeachMode())

    def set_watchdog(self, min_frequency_hz: float) -> bool:
        """Arm the RTDE watchdog: the control program stops when no command arrives for ``1/min_frequency_hz`` s."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return bool(self._ctrl.setWatchdog(float(min_frequency_hz)))

    def kick_watchdog(self) -> bool:
        """Tell the armed watchdog this program is alive. ``False`` where the control script did not answer."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return bool(self._ctrl.kickWatchdog())

    def restart_control_script(self) -> None:
        """Stop the control program and start a fresh one on a new control interface.

        ``stopScript`` ends the old program, and its teach mode and watchdog with it, so the arm holds;
        the control interface is closed and a new one built, whose constructor uploads the script and
        returns once it runs. The fresh program carries no watchdog and no teach mode, and it is what a
        moveJ after a hand-guided session needs. Where the new interface does not come up, every
        interface is closed before the error is raised, so the arm reads as disconnected and the next
        ``connect()`` starts clean.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        self._tcp_offset_cache = None
        self._forget_the_kinematics(tcp_only=True)
        try:
            for step in ("stopScript", "disconnect"):
                try:
                    getattr(self._ctrl, step)()
                except Exception as exc:  # noqa: BLE001 (the new interface below is the check that counts)
                    self.logger.warning("%s() on the old control interface failed: %s", step, exc)
            self._ctrl = None
            self._ctrl = self._open_control()
        except BaseException:  # a Ctrl-C included: half a restart leaves nothing open
            self._safe_teardown()
            raise
        self.logger.info("A fresh control program runs on a new control interface.")

    def is_program_running(self) -> bool:
        """Whether the ur_rtde control program runs on the controller, read on the control interface."""
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        return bool(self._ctrl.isProgramRunning())

    def get_joint_speeds(self) -> list[float]:
        """The actual joint speeds ``[qd0 ... qd5]`` in rad/s (``getActualQd``).

        ``isSteady`` is no stillness test for a hand-guided arm: it reads the script's own motion state,
        which is never steady in teach mode.
        """
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        return [float(v) for v in self._recv.getActualQd()]

    def get_payload(self) -> tuple[float, tuple[float, float, float]] | None:
        """The payload the controller compensates for, as mass in kg and CoG from the flange in mm.

        ``None`` where the receive recipe does not carry it: ur_rtde raises "unable to get state data for
        specified key: payload" for an output this controller did not publish. The CoG is converted from
        the controller's metres at this boundary, as :meth:`set_payload` converts the other way.
        """
        self._require_connected()
        assert self._recv is not None  # guaranteed by _require_connected()
        try:
            mass_kg = float(self._recv.getPayload())
            cog_m = [float(v) for v in self._recv.getPayloadCog()]
        except RuntimeError as exc:
            self.logger.warning("The controller payload cannot be read over RTDE (%s).", exc)
            return None
        if len(cog_m) != 3:
            self.logger.warning("The controller payload CoG has %d values, not 3; not read.", len(cog_m))
            return None
        return mass_kg, (cog_m[0] * 1e3, cog_m[1] * 1e3, cog_m[2] * 1e3)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _require_connected(self) -> None:
        if not self.is_connected:
            raise RuntimeError("Not connected: call connect() first.")

    def _require_io(self) -> None:
        if self._io is None:
            # The recorded import reason answers the first half of that question
            # outright, so it is attached rather than left for the operator to work out.
            why = f" {_IO_UNAVAILABLE}" if _IO_UNAVAILABLE else ""
            raise RuntimeError(
                "Digital/analog I/O unavailable: the RTDEIOInterface did not come up "
                f"(is ur_rtde's rtde_io present + the controller reachable?).{why}"
            )
