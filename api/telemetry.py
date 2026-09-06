"""What the status panel is allowed to read, and what each read actually costs.

The rule this module encodes: the panel polls the receive stream and nothing else. On a UR cell the
RTDE output stream is a broadcast the controller is already producing, so reading pose, joints and TCP
wrench costs nothing a running motion notices. Two things that look identical from a Python signature
do not behave that way, and both are excluded from the default reading:

* ``get_joint_torques()`` goes through the control interface, the same register space a running pick
  is using. It sits next to ``get_tcp_wrench()`` in the same capability Protocol and is measured in
  the same units, which is why it is worth naming here rather than trusting to memory.
* ``get_robot_status()`` looks like four cheap enum reads, and it is, plus a dashboard socket round
  trip on every single call, for the human-readable message. At a panel's poll rate that is a TCP
  request/response per tick, forever. It is available, opt-in, and says what it cost.

Everything is read through the vendor-neutral ``RobotArm`` surface and the optional capability
Protocols. A cell whose driver implements none of them renders empty fields rather than an error: sim,
dummy and KUKA advertise no force/torque or status capability at all, and a panel that raised on those
would be a panel that only works on one vendor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.constants import API_LOG_DIR, TELEMETRY_LOG_FILE
from src.utility.log_cfg import create_logger

__all__ = ["CellTelemetry", "read_telemetry"]

logger = create_logger("CellTelemetry", TELEMETRY_LOG_FILE, log_dir=API_LOG_DIR)

#: What the last logged reading said. ``GET /v1/cell/status`` is polled at 1 Hz by the cell screen, so
#: a line per call would be about 86 000 identical entries a day and the one tick where the reading
#: changed would be invisible among them. These gate the log call and nothing else: a lost race writes
#: one extra line and changes no answer. Deliberately not reset on disconnect, because the same arm
#: reconnecting produces the same evidence and ``api.lifecycle`` logs the transition itself.
_LAST_UNAVAILABLE: frozenset[str] = frozenset()
_LAST_PROVENANCE: tuple[Any, ...] | None = None


@dataclass(frozen=True, slots=True)
class CellTelemetry:
    """One snapshot. Every field is optional because every field comes from an optional capability."""

    connected: bool
    #: Whether this is a simulated arm. On the panel deliberately, so nobody reads a sim run as the
    #: real cell: the numbers look identical and only this says which one produced them.
    simulated: bool = False
    vendor: str = ""
    model: str = ""

    #: BASE-frame TCP pose, millimetres + XYZW quaternion. Receive stream.
    tcp_position_mm: tuple[float, float, float] | None = None
    tcp_quaternion_xyzw: tuple[float, float, float, float] | None = None
    #: Radians, base-to-tool order. Receive stream.
    joint_positions: tuple[float, ...] | None = None
    #: Newtons / newton-metres at the TCP, BASE frame. Receive stream (``getActualTCPForce``).
    tcp_force_n: tuple[float, float, float] | None = None
    tcp_torque_nm: tuple[float, float, float] | None = None

    #: Only populated when the caller asked for it; see the module docstring.
    robot_mode: str | None = None
    safety_mode: str | None = None
    protective_stopped: bool | None = None
    emergency_stopped: bool | None = None
    controller_message: str | None = None

    #: Three-valued, and never ``False``. ``True`` means this console can prove the thing on the other
    #: end is a simulator; ``None`` means it cannot tell, which is not the same as "physical".
    #:
    #: It exists because :attr:`simulated` above answers a different question than an operator (or an
    #: audience) reads it as. That flag is about the driver: dummy and Isaac are simulators, the UR
    #: driver is not, and it is correctly ``False`` for URSim, which is genuine UR controller software
    #: speaking the genuine protocol on the genuine ports, driving a robot that does not exist. A
    #: URSim UR3e therefore produces a full, plausible telemetry panel with ``simulated: False``,
    #: which a demo screen renders as "physical arm".
    #:
    #: Proof, not a hint: the controller's serial is URSim's ``...99999`` placeholder and its address
    #: is a loopback. Either alone is suggestive; a real cell's controller is not on 127.0.0.1.
    #: Anything less than both yields ``None``, because the opposite error, labelling a real arm a
    #: simulator, is the worse one.
    controller_is_simulator: bool | None = None
    #: The evidence itself, so a reader can judge the line above rather than trust it.
    controller_serial: str | None = None
    controller_host: str | None = None

    #: Which reads were attempted and failed, and why. A field that is simply unsupported is not an
    #: error and does not appear here; this is for a read that should have worked and did not, which on
    #: a live cell usually means the connection dropped between two ticks.
    unavailable: dict[str, str] = field(default_factory=dict)


def read_telemetry(arm: Any, *, include_controller_state: bool = False) -> CellTelemetry:
    """One snapshot of ``arm``, reading only what the caller is willing to pay for.

    Never raises for a missing capability or a dropped connection: a status panel that throws when the
    arm goes away shows a stack trace at exactly the moment an operator needs to see that the arm has
    gone away. Failures land in ``unavailable`` with their reason.
    """
    from src.robot.core import SupportsForceTorque, SupportsRobotStatus

    if arm is None:
        return CellTelemetry(connected=False)

    unavailable: dict[str, str] = {}
    capabilities = getattr(arm, "capabilities", None)
    connected = bool(getattr(arm, "is_connected", False))
    snapshot: dict[str, Any] = {
        "connected": connected,
        "simulated": bool(getattr(capabilities, "is_simulated", False)),
        "vendor": str(getattr(capabilities, "vendor", "") or ""),
        "model": str(getattr(capabilities, "model", "") or ""),
    }
    if not connected:
        # Nothing below is meaningful on a closed connection, and attempting it would bottom out in the
        # driver's bare RuntimeError rather than the typed error the arm-level methods raise.
        return CellTelemetry(**snapshot)

    try:
        pose = arm.get_tcp_pose()
        snapshot["tcp_position_mm"] = tuple(float(v) for v in pose.position_mm)
        snapshot["tcp_quaternion_xyzw"] = tuple(float(v) for v in pose.quaternion_xyzw)
    except Exception as exc:  # noqa: BLE001 (a panel reports the failure, it does not propagate it)
        unavailable["tcp_pose"] = f"{type(exc).__name__}: {exc}"

    try:
        snapshot["joint_positions"] = tuple(float(v) for v in arm.get_joint_positions())
    except Exception as exc:  # noqa: BLE001
        unavailable["joint_positions"] = f"{type(exc).__name__}: {exc}"

    if isinstance(arm, SupportsForceTorque):
        try:
            # get_tcp_wrench only. get_joint_torques is the control-interface read; see the module
            # docstring. It is not offered here at any price.
            wrench = arm.get_tcp_wrench()
            snapshot["tcp_force_n"] = (
                float(wrench.fx), float(wrench.fy), float(wrench.fz),
            )
            snapshot["tcp_torque_nm"] = (
                float(wrench.tx), float(wrench.ty), float(wrench.tz),
            )
        except Exception as exc:  # noqa: BLE001
            unavailable["tcp_wrench"] = f"{type(exc).__name__}: {exc}"

    if include_controller_state and isinstance(arm, SupportsRobotStatus):
        try:
            status = arm.get_robot_status()
            snapshot["robot_mode"] = str(status.robot_mode)
            snapshot["safety_mode"] = str(status.safety_mode)
            snapshot["protective_stopped"] = bool(status.protective_stopped)
            snapshot["emergency_stopped"] = bool(status.emergency_stopped)
            snapshot["controller_message"] = status.message
        except Exception as exc:  # noqa: BLE001
            unavailable["controller_state"] = f"{type(exc).__name__}: {exc}"

    _add_simulator_evidence(arm, snapshot)
    _log_changes(snapshot, unavailable)
    return CellTelemetry(**snapshot, unavailable=unavailable)


def _log_changes(snapshot: dict[str, Any], unavailable: dict[str, str]) -> None:
    """Write a line for what changed since the last reading, and nothing for the reading itself.

    Two things in a snapshot outlive the panel that renders it: a read that should have worked and did
    not, and the evidence behind the simulator verdict. The rest is numbers a panel draws and forgets,
    at 1 Hz, forever.
    """
    global _LAST_UNAVAILABLE, _LAST_PROVENANCE

    failed = frozenset(unavailable)
    if failed != _LAST_UNAVAILABLE:
        _LAST_UNAVAILABLE = failed
        if failed:
            # Warning rather than error: the call still returns a usable snapshot, so this is a
            # degraded reading rather than a failed one. On a live cell it usually means the
            # connection dropped between two ticks, which the panel shows as blank fields and
            # nothing else records.
            logger.warning(
                "Telemetry read(s) failed: %s.",
                "; ".join(f"{name} ({reason})" for name, reason in sorted(unavailable.items())),
            )
        else:
            logger.info("Telemetry reads are answering again; no field is unavailable.")

    provenance = (
        snapshot.get("simulated"),
        snapshot.get("controller_is_simulator"),
        snapshot.get("controller_serial"),
        snapshot.get("controller_host"),
    )
    if provenance != _LAST_PROVENANCE:
        _LAST_PROVENANCE = provenance
        # Both flags on one line because they answer different questions and only together are they
        # honest: ``simulated`` is about the driver and is correctly False for URSim, so on its own it
        # renders a URSim UR3e as a physical arm.
        logger.info(
            "Cell provenance: driver simulated=%s, controller_is_simulator=%s "
            "(serial %s, host %s).",
            provenance[0], provenance[1],
            provenance[2] or "unknown", provenance[3] or "unknown",
        )


def _add_simulator_evidence(arm: Any, snapshot: dict[str, Any]) -> None:
    """Fill ``controller_is_simulator`` / ``controller_serial`` / ``controller_host`` where provable.

    Best-effort and never-raising: a driver with no dashboard, a controller that will not answer, or
    any exception leaves all three ``None``, which the UI must render as "cannot tell" rather than as
    either verdict. Costs one dashboard round trip and is therefore only attempted on a driver that
    advertises the reads.
    """
    conn = getattr(arm, "_conn", None)
    if conn is None or not hasattr(conn, "controller_serial"):
        return
    try:
        serial = conn.controller_serial()
        host = str(getattr(conn, "ip", "") or "") or None
    except Exception as exc:  # noqa: BLE001 (evidence gathering must never break a status panel)
        # Debug, and therefore silent at the default level: this is on the polled path, so a
        # controller whose dashboard will not answer would write one line per tick. It is here at all
        # because "cannot tell" and "never asked" render identically, and this says which one it was.
        logger.debug("No provenance evidence this tick: %s: %s", type(exc).__name__, exc)
        return
    snapshot["controller_serial"] = serial
    snapshot["controller_host"] = host
    loopback = host in {"127.0.0.1", "::1", "localhost"} if host else False
    if serial and serial.endswith("99999") and loopback:
        snapshot["controller_is_simulator"] = True
