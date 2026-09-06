"""URConnection, the thin RTDE wrapper around a Universal Robots controller.

It uses the ``ur_rtde`` library, installed with ``pip install ur_rtde``, to talk to the
robot over the Real-Time Data Exchange protocol.

It provides:
    - the connect and disconnect lifecycle, usable as a context manager;
    - the current TCP pose, through ``get_tcp_pose``;
    - the current joint angles, through ``get_joint_positions``;
    - forward kinematics, through ``fk``;
    - inverse kinematics, through ``ik``;
    - the low-level move wrappers ``moveJ`` and ``moveL``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING



from ...constants import UR_CONNECTION_LOG_FILE, create_robot_logger
from ...core import RobotConnectionError

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


if TYPE_CHECKING:  # pragma: no cover (import only for static analysis)
    from dashboard_client import DashboardClient
    from rtde_control import RTDEControlInterface
    from rtde_io import RTDEIOInterface
    from rtde_receive import RTDEReceiveInterface


class URConnection:
    """Manage the RTDE connection to a UR robot.

    Parameters:
        ip:           The IP address of the robot controller.
        vel:          The default joint velocity in rad/s.
        acc:          The default joint acceleration in rad/s^2.
        frequency:    The RTDE exchange frequency in Hz. 0 means the default, 125 Hz.
    """

    def __init__(
        self,
        ip: str,
        vel: float = 1.0,
        acc: float = 0.5,
        frequency: float = 0.0,
    ):
        # An ur_rtde import failure is deferred to connect(), so this module imports on
        # a machine without the native library.
        self.ip = ip
        self.vel = vel
        self.acc = acc
        self.frequency = frequency
        self.logger = create_robot_logger("URConnection", UR_CONNECTION_LOG_FILE)

        self._ctrl: RTDEControlInterface | None = None
        self._recv: RTDEReceiveInterface | None = None
        self._io: RTDEIOInterface | None = None
        self._dashboard: DashboardClient | None = None
        #: Cached controller TCP offset for :meth:`_fk_tcp_offset`; cleared on every (dis)connect.
        self._tcp_offset_cache: list[float] | None = None

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
        self.logger.info("Connecting to UR robot at %s ...", self.ip)
        try:
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
            self._ctrl = rtde_control.RTDEControlInterface(
                self.ip, frequency,
            )
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

    def disconnect(self) -> None:
        """Idempotent: safe to call repeatedly, and after a failed connect()."""
        self._safe_teardown()
        self.logger.info("Disconnected.")

    def _safe_teardown(self) -> None:
        """Best-effort release of both RTDE interfaces; never raises."""
        self._tcp_offset_cache = None
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

        It uses the built-in FK of the controller.

        Measured, this form is corrupted by motion. It shares the controller float
        registers with moveJ and moveL, so after any commanded move it returns a pose
        offset by hundreds of millimetres until those registers are rewritten, which is
        the mechanism GitLab #348 describes. It is correct on a freshly powered
        controller and after IK, and wrong after a move.

        Where FK of the current joints is meant, use :meth:`fk_current`, which is
        immune. Where FK of an arbitrary q is genuinely needed, as in the singularity
        guard, move_home and MotionController, there is no immune equivalent and it
        remains an open question for the real cell. Do not assume those call sites are
        fine.
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
        starts.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        return self._ctrl.moveJ(joint_positions, v, a, asynchronous)

    def moveL(
        self,
        tcp_pose: list[float],
        vel: float | None = None,
        acc: float | None = None,
        asynchronous: bool = False,
    ) -> bool:
        """A linear, meaning Cartesian, move.

        ``tcp_pose`` is ``[x, y, z, rx, ry, rz]`` in metres with an axis-angle rotation.
        """
        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()
        v = vel if vel is not None else self.vel
        a = acc if acc is not None else self.acc
        return self._ctrl.moveL(tcp_pose, v, a, asynchronous)

    def stop(self, deceleration: float = 2.0) -> None:
        """Stop all movement immediately.

        It calls both ``stopJ`` and ``stopL``, so the robot halts whether the active
        command is a joint-space or a linear move. An error from the controller is
        logged and never re-raised, because stopping is best-effort and a cleanup path
        must never fail.

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

        Parameters
        ----------
        timeout_s : float
            The longest wait in seconds. It is at least 0, and a value at or below 0
            returns immediately with the current ``is_steady()`` value.
        poll_interval_s : float
            The sleep between polls, capped at ``timeout_s`` for a short wait.

        Returns
        -------
        bool
            ``True`` where the robot became steady before the timeout, and ``False``
            where the timeout fired first.
        """
        import time as _time

        self._require_connected()
        assert self._ctrl is not None  # guaranteed by _require_connected()

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

    # ------------------------------------------------------------------
    # Digital / analog I/O  (RTDEIOInterface set-side + RTDEReceiveInterface read-side)
    # ------------------------------------------------------------------
    # UR groups digital pins into banks: standard 0 to 7, configurable 0 to 7, tool 0 to
    # 1. The set side takes a per-bank id through a bank-specific method, and the read
    # side uses a global id, standard 0 to 7, configurable 8 to 15 and tool 16 to 17.
    # ``_global_din_id`` bridges that asymmetry.
    _PORT_BASE = {"standard": 0, "configurable": 8, "tool": 16}

    def set_digital_out(self, pin: int, value: bool, port: str = "standard") -> None:
        """Drive a digital output pin (bank ``port``) to ``value``."""
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
        """Set an analog output: voltage (V) unless ``current=True`` (A)."""
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
        """The controller serial number, or ``None``. Best-effort, on the contract above."""
        if self._dashboard is None:
            return None
        try:
            serial = str(self._dashboard.getSerialNumber()).strip()
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("dashboard.getSerialNumber() failed: %s", exc)
            return None
        return serial or None

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
        """
        if self._dashboard is None:
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
