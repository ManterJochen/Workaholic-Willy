"""An instant UR controller for the bench: what ur_rtde would talk to, answered at once, so a cell's real driver runs.

The bench drives the owner's own stack, the UR driver with its judged lines, its cuRobo routes, its exact guard and its
camera world, and only the controller is stood in for. A ``moveJ`` or a ``moveL`` the driver sends is taken at once:
the joints are the target the moment it returns, where URSim takes the seconds a real arm takes. Nothing here judges a
motion; the driver judged it before it sent it, as it does on a cell.

:func:`install` puts the four stand-in modules where :mod:`src.robot.drivers.ur.connection` imported ``rtde_control``,
``rtde_receive``, ``rtde_io`` and ``dashboard_client``, and returns the :class:`ControllerState` they share. Kinematics
are the stack's own UR tables (:mod:`src.robot.safety._ur_kinematics`, :mod:`src.robot.safety._ur_ik`), the TCP the
controller holds is the cell's declared tool frame. A tool output that changes calls ``on_tool_output(pin, level, tcp)``,
which the bench hands to the scene so the jaws grip and set down.

The controller never stops on its own: no protective stop, no safety fault. A bench that needs one asks for it through
:meth:`ControllerState.protective_stop`.
"""

from __future__ import annotations

import math
import threading
import types
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["ControllerState", "install", "pose_matrix", "ur_pose"]


def pose_matrix(pose: Sequence[float]) -> np.ndarray:
    """A UR pose ``[x, y, z, rx, ry, rz]`` (metres, axis-angle) as a 4x4 in millimetres."""
    x, y, z, rx, ry, rz = (float(v) for v in pose)
    angle = math.sqrt(rx * rx + ry * ry + rz * rz)
    out = np.eye(4)
    if angle > 1e-12:
        k = np.array([rx, ry, rz]) / angle
        kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
        out[:3, :3] = np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)
    out[:3, 3] = [x * 1000.0, y * 1000.0, z * 1000.0]
    return out


def ur_pose(matrix: np.ndarray) -> list[float]:
    """A 4x4 in millimetres as a UR pose ``[x, y, z, rx, ry, rz]`` (metres, axis-angle)."""
    m = np.asarray(matrix, dtype=np.float64)
    r = m[:3, :3]
    cos = max(-1.0, min(1.0, (float(np.trace(r)) - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        axis = np.zeros(3)
    elif math.pi - angle < 1e-6:
        # Near a half turn the antisymmetric part vanishes: the axis is the column of r + I with the largest norm.
        b = r + np.eye(3)
        col = int(np.argmax(np.linalg.norm(b, axis=0)))
        axis = b[:, col] / np.linalg.norm(b[:, col]) * angle
    else:
        axis = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (2.0 * math.sin(angle)) * angle
    t = m[:3, 3] / 1000.0
    return [float(t[0]), float(t[1]), float(t[2]), float(axis[0]), float(axis[1]), float(axis[2])]


@dataclass
class ControllerState:
    """The one controller every stand-in interface reads and writes."""

    model: str
    joints: list[float]
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    payload_kg: float = 0.0
    payload_cog: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    #: Digital outputs and inputs as the controller numbers them: standard 0-7, configurable 8-15, tool 16-17.
    outputs: list[bool] = field(default_factory=lambda: [False] * 18)
    inputs: list[bool] = field(default_factory=lambda: [False] * 18)
    protective_stopped: bool = False
    on_tool_output: Callable[[int, bool, np.ndarray], None] | None = None
    #: Every command sent, ``(verb, target)``: joints in radians, or a UR pose.
    commands: list[tuple[str, Any]] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    # --- kinematics ------------------------------------------------------------------------------------------------
    def flange_mm(self, joints: Sequence[float]) -> np.ndarray:
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm  # noqa: PLC0415

        frames = ur_link_transforms_mm(self.model, np.asarray(joints, dtype=np.float64))
        if frames is None:
            raise RuntimeError(f"no kinematics table for {self.model!r}")
        return np.asarray(frames[-1], dtype=np.float64)

    def tcp_mm(self, joints: Sequence[float] | None = None, offset: Sequence[float] | None = None) -> np.ndarray:
        q = self.joints if joints is None else list(joints)
        return self.flange_mm(q) @ pose_matrix(self.tcp_offset if offset is None else offset)

    def ik(self, tcp_pose: Sequence[float], near: Sequence[float] | None = None) -> list[float]:
        """The configuration nearest ``near`` (the current joints where none) that puts the TCP at ``tcp_pose``."""
        from src.robot.safety._ur_ik import ur_flange_ik  # noqa: PLC0415

        flange = pose_matrix(tcp_pose) @ np.linalg.inv(pose_matrix(self.tcp_offset))
        solutions = ur_flange_ik(self.model, flange)
        if not solutions:
            raise RuntimeError("no inverse kinematics solution for this pose")
        seed = np.asarray(self.joints if near is None else near, dtype=np.float64)
        best, best_cost = None, math.inf
        for solution in solutions:
            q = np.asarray(solution, dtype=np.float64)
            # Every joint the controller may turn a full turn either way: the representative nearest the seed.
            q = seed + (q - seed + math.pi) % (2.0 * math.pi) - math.pi
            cost = float(np.sum((q - seed) ** 2))
            if cost < best_cost:
                best, best_cost = q, cost
        assert best is not None
        return [float(v) for v in best]

    # --- what the bench may do -------------------------------------------------------------------------------------
    def teleport(self, joints: Sequence[float]) -> None:
        with self.lock:
            self.joints = [float(v) for v in joints]

    def protective_stop(self) -> None:
        with self.lock:
            self.protective_stopped = True

    def set_output(self, index: int, level: bool) -> None:
        with self.lock:
            before = self.outputs[index]
            self.outputs[index] = bool(level)
            tcp = self.tcp_mm()
        if before != bool(level) and index >= 16 and self.on_tool_output is not None:
            self.on_tool_output(index - 16, bool(level), tcp)


class _Progress:
    def isAsyncOperationRunning(self) -> bool:  # noqa: N802 (ur_rtde's name)
        return False

    def __int__(self) -> int:
        return -1


def install(state: ControllerState) -> dict[str, Any]:
    """Put the stand-in ``ur_rtde`` modules into the UR connection module; returns what was there, for :func:`remove`."""
    from src.robot.drivers.ur import connection  # noqa: PLC0415

    class RTDEControlInterface:
        def __init__(self, ip: str, *args: Any, **kwargs: Any) -> None:
            self.ip = ip

        # motion: taken at once
        def moveJ(self, target: Any, speed: float = 1.05, acceleration: float = 1.4,  # noqa: N802
                  asynchronous: bool = False) -> bool:
            if state.protective_stopped:
                return False
            q = target
            if len(target) and isinstance(target[0], (list, tuple)):
                q = target[-1][:6]
            with state.lock:
                state.commands.append(("moveJ", [float(v) for v in q[:6]]))
                state.joints = [float(v) for v in q[:6]]
            return True

        def moveL(self, pose: Any, speed: float = 0.25, acceleration: float = 1.2,  # noqa: N802
                  asynchronous: bool = False) -> bool:
            if state.protective_stopped:
                return False
            target = pose
            if len(pose) and isinstance(pose[0], (list, tuple)):
                target = pose[-1][:6]
            with state.lock:
                state.commands.append(("moveL", [float(v) for v in target[:6]]))
                state.joints = state.ik(target[:6])
            return True

        def stopJ(self, *args: Any, **kwargs: Any) -> bool:  # noqa: N802
            return True

        def stopL(self, *args: Any, **kwargs: Any) -> bool:  # noqa: N802
            return True

        def getAsyncOperationProgressEx(self) -> _Progress:  # noqa: N802
            return _Progress()

        # kinematics
        def getForwardKinematics(self, q: Any = None, tcp_offset: Any = None) -> list[float]:  # noqa: N802
            if q is None:
                return ur_pose(state.tcp_mm())
            return ur_pose(state.tcp_mm(q, tcp_offset))

        def getInverseKinematics(self, x: Any, qnear: Any = None, *args: Any) -> list[float]:  # noqa: N802
            return state.ik(x, qnear)

        def getTCPOffset(self) -> list[float]:  # noqa: N802
            return list(state.tcp_offset)

        def setTcp(self, offset: Any) -> bool:  # noqa: N802
            state.tcp_offset = [float(v) for v in offset]
            return True

        # state and housekeeping
        def isSteady(self) -> bool:  # noqa: N802
            return True

        def isProgramRunning(self) -> bool:  # noqa: N802
            return not state.protective_stopped

        def isConnected(self) -> bool:  # noqa: N802
            return True

        def setPayload(self, mass: float, cog: Any = None) -> bool:  # noqa: N802
            state.payload_kg = float(mass)
            if cog is not None:
                state.payload_cog = [float(v) for v in cog]
            return True

        def getJointTorques(self) -> list[float]:  # noqa: N802
            return [0.0] * 6

        def teachMode(self) -> bool:  # noqa: N802
            return True

        def endTeachMode(self) -> bool:  # noqa: N802
            return True

        def setWatchdog(self, *args: Any) -> bool:  # noqa: N802
            return True

        def kickWatchdog(self) -> bool:  # noqa: N802
            return True

        def stopScript(self) -> None:  # noqa: N802
            return None

        def reuploadScript(self) -> bool:  # noqa: N802
            return True

        def disconnect(self) -> None:
            return None

    class RTDEReceiveInterface:
        def __init__(self, ip: str, *args: Any, **kwargs: Any) -> None:
            self.ip = ip

        def getActualQ(self) -> list[float]:  # noqa: N802
            with state.lock:
                return list(state.joints)

        def getActualQd(self) -> list[float]:  # noqa: N802
            return [0.0] * 6

        def getActualTCPPose(self) -> list[float]:  # noqa: N802
            return ur_pose(state.tcp_mm())

        def getActualTCPForce(self) -> list[float]:  # noqa: N802
            return [0.0] * 6

        def getRobotMode(self) -> int:  # noqa: N802
            return 7  # RUNNING

        def getSafetyMode(self) -> int:  # noqa: N802
            return 3 if state.protective_stopped else 1  # PROTECTIVE_STOP or NORMAL

        def getSafetyStatusBits(self) -> int:  # noqa: N802
            return 4 if state.protective_stopped else 1

        def isProtectiveStopped(self) -> bool:  # noqa: N802
            return bool(state.protective_stopped)

        def isEmergencyStopped(self) -> bool:  # noqa: N802
            return False

        def getDigitalOutState(self, index: int) -> bool:  # noqa: N802
            return bool(state.outputs[int(index)])

        def getDigitalInState(self, index: int) -> bool:  # noqa: N802
            return bool(state.inputs[int(index)])

        def getActualDigitalOutputBits(self) -> int:  # noqa: N802
            return sum(1 << i for i, on in enumerate(state.outputs) if on)

        def getPayload(self) -> float:  # noqa: N802
            return float(state.payload_kg)

        def getPayloadCog(self) -> list[float]:  # noqa: N802
            return list(state.payload_cog)

        def isConnected(self) -> bool:  # noqa: N802
            return True

        def disconnect(self) -> None:
            return None

    class RTDEIOInterface:
        def __init__(self, ip: str, *args: Any, **kwargs: Any) -> None:
            self.ip = ip

        def setToolDigitalOut(self, pin: int, level: bool) -> bool:  # noqa: N802
            state.set_output(16 + int(pin), level)
            return True

        def setStandardDigitalOut(self, pin: int, level: bool) -> bool:  # noqa: N802
            state.set_output(int(pin), level)
            return True

        def setConfigurableDigitalOut(self, pin: int, level: bool) -> bool:  # noqa: N802
            state.set_output(8 + int(pin), level)
            return True

        def setAnalogOutputVoltage(self, *args: Any) -> bool:  # noqa: N802
            return True

        def setAnalogOutputCurrent(self, *args: Any) -> bool:  # noqa: N802
            return True

        def disconnect(self) -> None:
            return None

    class DashboardClient:
        def __init__(self, ip: str, *args: Any, **kwargs: Any) -> None:
            self.ip = ip

        def connect(self, *args: Any) -> None:
            return None

        def disconnect(self) -> None:
            return None

        def isConnected(self) -> bool:  # noqa: N802
            return True

        def isInRemoteControl(self) -> bool:  # noqa: N802
            return True

        def safetystatus(self) -> str:
            return "Safetystatus: PROTECTIVE_STOP" if state.protective_stopped else "Safetystatus: NORMAL"

        def robotmode(self) -> str:
            return "Robotmode: RUNNING"

        def polyscopeVersion(self) -> str:  # noqa: N802
            return "URSoftware 3.15.8.106339 (Nov 01 2023)"

        def getSerialNumber(self) -> str:  # noqa: N802
            return "BENCH"

        def getRobotModel(self) -> str:  # noqa: N802
            return state.model.upper()

        def unlockProtectiveStop(self) -> None:  # noqa: N802
            state.protective_stopped = False

        def closeSafetyPopup(self) -> None:  # noqa: N802
            return None

    modules = {
        "rtde_control": types.SimpleNamespace(RTDEControlInterface=RTDEControlInterface),
        "rtde_receive": types.SimpleNamespace(RTDEReceiveInterface=RTDEReceiveInterface),
        "rtde_io": types.SimpleNamespace(RTDEIOInterface=RTDEIOInterface),
        "dashboard_client": types.SimpleNamespace(DashboardClient=DashboardClient),
    }
    before = {name: getattr(connection, name, None) for name in modules}
    for name, module in modules.items():
        setattr(connection, name, module)
    return before


def remove(before: dict[str, Any]) -> None:
    """Put back what :func:`install` replaced."""
    from src.robot.drivers.ur import connection  # noqa: PLC0415

    for name, module in before.items():
        setattr(connection, name, module)
