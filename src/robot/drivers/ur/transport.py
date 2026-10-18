"""The UR transport: RTDE and the UR motion controller behind the planned arm's verbs (P0e E7).

A translation and nothing else. Every conversion happens exactly where the arm made it before E7: a controller list
becomes a :class:`URPose` and then a :class:`~src.geometry.Pose` through ``urpose_to_pose``, a pose goes back
through ``pose_to_urpose``, and each RTDE call is the call the arm sent itself, with the same arguments in the same
order. The P0e command tape holds that to the byte.

``connection`` and ``motion`` are plain attributes on purpose: ``URRobotArm._conn`` and ``._motion`` read and write
them, so a test that injects a connection into the arm injects it here (``tests/_injection_guard.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.geometry import Frame
from src.robot.core import (
    ControllerPayload,
    ControllerProvenance,
    RobotConnectionError,
    RobotMode,
    RobotStatus,
    SafetyMode,
    Wrench,
)

from .pose import URPose
from .pose_adapter import pose_to_urpose, urpose_to_pose

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from collections.abc import Sequence

    import numpy as np

    from src.geometry import Pose
    from src.robot.core import HaltState, MotionStatus
    from src.robot.core.arm_capabilities import DigitalIOPort, MoveEnd

    from .connection import URConnection
    from .motion import MotionController

__all__ = ["URTransport"]

# UR controller mode integers to the vendor-neutral enums. UR getRobotMode runs from -1
# for no controller and 0 for disconnected up to 8 for updating firmware; getSafetyMode
# runs from 1 for normal to 9 for fault, and 10 to 13, the validate, undefined, auto-mode
# and three-position stops, map to unknown here. An unmapped integer falls back to
# unknown.
_UR_ROBOT_MODE: dict[int, RobotMode] = {
    0: RobotMode.DISCONNECTED, 1: RobotMode.CONFIRM_SAFETY, 2: RobotMode.BOOTING,
    3: RobotMode.POWER_OFF, 4: RobotMode.POWER_ON, 5: RobotMode.IDLE,
    6: RobotMode.BACKDRIVE, 7: RobotMode.RUNNING, 8: RobotMode.UPDATING_FIRMWARE,
}
_UR_SAFETY_MODE: dict[int, SafetyMode] = {
    1: SafetyMode.NORMAL, 2: SafetyMode.REDUCED, 3: SafetyMode.PROTECTIVE_STOP,
    4: SafetyMode.RECOVERY, 5: SafetyMode.SAFEGUARD_STOP, 6: SafetyMode.SYSTEM_EMERGENCY_STOP,
    7: SafetyMode.ROBOT_EMERGENCY_STOP, 8: SafetyMode.VIOLATION, 9: SafetyMode.FAULT,
}


def _is_ursim_serial(serial: "str | None") -> bool:
    """URSim's serial: 10 digits ending 9999 or 11 ending 99999, and nothing else (the owner's rule of 2026-10-02).

    Measured: URSim CB3 (PolyScope 3.15.8) answers ``2018309999`` over the dashboard's "get serial number", URSim
    e-Series ``20195399999``. A real controller's serial could end the same way only by chance, which is why the
    loopback address is the other half of the proof (:meth:`URTransport.read_provenance`).
    """
    if not serial or not serial.isdigit():
        return False
    return (len(serial) == 10 and serial.endswith("9999")) or (len(serial) == 11 and serial.endswith("99999"))


class URTransport:
    """Every transport Protocol of :mod:`src.robot.core.transport` over one :class:`URConnection` and the
    :class:`MotionController` that shares it: ``ArmTransport``, ``ControllerKinematics``, ``ControllerMotion``,
    ``TakesPayload``, ``ReportsControllerModel``, ``HoldsAHaltLatch``, ``ControllerIO``, ``ReadsForceTorque``,
    ``ReportsControllerStatus``, ``ReleasesProtectiveStop``, ``ReadsControllerProvenance``, ``ControllerFreedrive``
    and ``WatchesController``."""

    def __init__(self, connection: URConnection, motion: MotionController) -> None:
        self.connection = connection
        self.motion = motion

    # ---- ArmTransport ---------------------------------------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return bool(self.connection.is_connected)

    def connect(self) -> None:
        self.connection.connect()

    def disconnect(self) -> None:
        self.connection.disconnect()

    # ---- WatchesController ----------------------------------------------------------------------------------------------

    @property
    def is_watching(self) -> bool:
        return bool(self.connection.is_watching)

    def connect_watch(self) -> None:
        """The receive interface and the dashboard, never the control interface (P0i): what a stopped UR still opens."""
        self.connection.connect_watch()

    def read_joints(self) -> list[float]:
        return self.connection.get_joint_positions()

    def run_trajectory(
        self, waypoints: Sequence[Sequence[float]], *, vel: float | None, acc: float | None,
    ) -> bool:
        for waypoint in waypoints:
            if not self.connection.moveJ(list(waypoint), vel=vel, acc=acc):
                return False
        return True

    def stop(self) -> None:
        self.connection.stop()

    def wait_until_steady(self, timeout_s: float, poll_interval_s: float) -> bool:
        return bool(self.connection.wait_until_steady(timeout_s, poll_interval_s))

    # ---- ControllerKinematics -------------------------------------------------------------------------------------------

    def is_steady(self) -> bool:
        return bool(self.connection.is_steady())

    def read_tcp(self, *, label: str = "") -> Pose:
        # Through the motion controller, as the arm read it: it labels its own reading "current".
        urpose = self.motion.get_current_pose()
        return urpose_to_pose(urpose, frame=Frame.BASE, label=label)

    def controller_tcp_now(self) -> np.ndarray:
        return URPose.from_ur_list(self.connection.fk_current()).to_T()

    def controller_fk(self, joints: Sequence[float], *, label: str = "") -> Pose:
        urpose = URPose.from_ur_list(self.connection.fk(list(joints)), label=label)
        return urpose_to_pose(urpose, frame=Frame.BASE, label=label)

    def controller_ik(self, pose: Pose, *, seed: Sequence[float] | None = None) -> list[float] | None:
        seed_list = list(seed) if seed is not None else None
        joints = self.connection.ik(pose_to_urpose(pose).to_ur_list(), q_near=seed_list)
        # ur_rtde answers [] for a pose with no solution, and the Protocol says None (P0g slice 7). PlannedArm read
        # both as no solution, so no motion changes.
        return joints if joints else None

    # ---- ControllerMotion -------------------------------------------------------------------------------------------------

    def move_pose(
        self, pose: Pose, *, linear: bool, vel: float | None, acc: float | None, register: bool,
        workspace_pose: Pose,
    ) -> bool:
        return self.motion.move_to(
            pose_to_urpose(pose), linear=linear, vel=vel, acc=acc, register=register,
            workspace_pose=pose_to_urpose(workspace_pose),
        )

    @property
    def last_reject_status(self) -> MotionStatus | None:
        return self.motion.last_reject_status

    # ---- TakesPayload, ReportsControllerModel ------------------------------------------------------------------------

    def set_payload(self, mass_kg: float, cog_mm: Sequence[float]) -> None:
        self.connection.set_payload(mass_kg, cog_mm)  # type: ignore[arg-type]

    def controller_model(self) -> str | None:
        return self.connection.controller_model()

    # ---- HoldsAHaltLatch ----------------------------------------------------------------------------------------------
    #
    # The latch is the connection's (URConnection.request_halt): a connection a test built by hand may keep no record of
    # how its last move ended, or no count, and reads None there, as the arm read it before the transport.

    def request_halt(self, reason: str) -> HaltState:
        return self.connection.request_halt(reason)

    def clear_halt(self) -> None:
        self.connection.clear_halt()

    def halt_state(self) -> HaltState | None:
        return self.connection.halt_state()

    @property
    def brake_on_halt(self) -> bool:
        # Every UR brakes the move in flight on a halt since 2026-10-17, as the owner's cell ran it.
        return True

    @property
    def last_move_end(self) -> MoveEnd | None:
        return getattr(self.connection, "last_move_end", None)

    @property
    def halt_requests(self) -> int | None:
        count = getattr(self.connection, "halt_requests", None)
        return count if isinstance(count, int) and not isinstance(count, bool) else None

    # ---- ControllerIO -------------------------------------------------------------------------------------------------
    #
    # The bank goes to the connection by its name, which picks the RTDE setter and the read offset there. A write is
    # the bool ur_rtde answered; a read is the level as the connection answers it, unconverted, as the arm read it.

    def write_digital(self, pin: int, value: bool, *, port: DigitalIOPort) -> bool:
        return bool(self.connection.set_digital_out(int(pin), bool(value), str(port)))

    def read_digital_input(self, pin: int, *, port: DigitalIOPort) -> bool:
        return self.connection.get_digital_in(int(pin), str(port))

    def read_digital_output(self, pin: int, *, port: DigitalIOPort) -> bool:
        return self.connection.get_digital_out(int(pin), str(port))

    def write_analog(self, pin: int, value: float, *, current: bool) -> bool:
        return bool(self.connection.set_analog_out(int(pin), float(value), bool(current)))

    def end_pulse(self, pin: int, *, port: DigitalIOPort) -> bool:
        return bool(self.connection.end_output_pulse(int(pin), str(port)))

    # ---- ReadsForceTorque, ReportsControllerStatus, ReleasesProtectiveStop, ReadsControllerProvenance ---------------

    def read_tcp_wrench(self) -> Wrench:
        f = self.connection.get_tcp_force()
        return Wrench(float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[4]), float(f[5]),
                      frame=Frame.BASE)

    def read_joint_torques(self) -> tuple[float, ...]:
        return tuple(float(t) for t in self.connection.get_joint_torques())

    def read_status(self) -> RobotStatus:
        """UR's ``getRobotMode`` / ``getSafetyMode`` / ``isProtectiveStopped`` / ``isEmergencyStopped`` /
        ``isProgramRunning`` plus the dashboard ``safetystatus`` text, as the core's status."""
        return self._status(quick=False)

    def read_status_quickly(self) -> RobotStatus:
        """The same from the RTDE streams alone, with no dashboard round trip (``message`` empty). Raises
        :class:`RobotConnectionError` on a closed connection."""
        if not self.connection.is_connected:
            raise RobotConnectionError("quick_robot_status() requires an open connection.")
        return self._status(quick=True)

    def _status(self, *, quick: bool) -> RobotStatus:
        return RobotStatus(
            robot_mode=_UR_ROBOT_MODE.get(self.connection.get_robot_mode(), RobotMode.UNKNOWN),
            safety_mode=_UR_SAFETY_MODE.get(self.connection.get_safety_mode(), SafetyMode.UNKNOWN),
            protective_stopped=self.connection.is_protective_stopped(),
            emergency_stopped=self.connection.is_emergency_stopped(),
            message="" if quick else self.connection.dashboard_safety_status(),
            # On a watch no control program runs, by design: it is never started there (P0i).
            program_running=bool(self.connection.is_program_running()) if self.connection.is_connected else False,
        )

    def release_protective_stop(self) -> bool:
        return self.connection.unlock_protective_stop()

    def read_provenance(self) -> ControllerProvenance:
        """The serial and the address, and URSim proven only where both say so.

        The proof rule is the controller's: a URSim controller answers the dashboard with a placeholder serial
        (:func:`_is_ursim_serial`), and it is reachable on a loopback address. Either half alone proves nothing, because
        a real controller can sit behind a local tunnel and a serial can be anything; both together have never been a
        physical arm.
        """
        serial = self.connection.controller_serial()
        host = str(getattr(self.connection, "ip", "") or "") or None
        loopback = host in {"127.0.0.1", "::1", "localhost"} if host else False
        proven = True if (_is_ursim_serial(serial) and loopback) else None
        return ControllerProvenance(serial=serial, host=host, is_simulator=proven)

    # ---- ControllerFreedrive ----------------------------------------------------------------------------------------------

    def arm_watchdog(self, min_frequency_hz: float) -> bool:
        return self.connection.set_watchdog(min_frequency_hz)

    def kick_watchdog(self) -> bool:
        return self.connection.kick_watchdog()

    def begin_hand_guiding(self) -> bool:
        return self.connection.teach_mode()

    def end_hand_guiding(self) -> bool:
        return self.connection.end_teach_mode()

    def restart_control_program(self) -> None:
        self.connection.restart_control_script()

    def control_program_running(self) -> bool:
        return self.connection.is_program_running()

    def stop_latched(self) -> bool:
        return bool(self.connection.is_protective_stopped() or self.connection.is_emergency_stopped())

    def read_joint_speeds(self) -> list[float]:
        return self.connection.get_joint_speeds()

    def read_payload(self) -> ControllerPayload | None:
        payload = self.connection.get_payload()
        if payload is None:
            return None
        mass_kg, cog_mm = payload
        return ControllerPayload(mass_kg=float(mass_kg), cog_mm=(float(cog_mm[0]), float(cog_mm[1]), float(cog_mm[2])))
