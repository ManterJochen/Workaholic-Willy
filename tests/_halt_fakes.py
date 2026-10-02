"""Doubles for the halt tests: ur_rtde as interfaces that write every call down, and a controller whose moves take time.

Two doubles, for two kinds of question.

:class:`RecordingRtde` answers "what did the driver send, in what order, with what arguments". It stands in for the four
ur_rtde interfaces a :class:`~src.robot.drivers.ur.connection.URConnection` holds (control, receive, I/O, dashboard) and
writes every call onto one log, with plain JSON values, so a table of moves can be compared call for call against what the
code sent before the halt was built. A method it was not told how to answer raises ``AttributeError``: a call the driver
did not make before is a failure of the comparison, never a silent ``MagicMock``.

:class:`SimulatedUr` answers "what does the driver do while a move runs". Its moves take time on a clock the test owns,
an asynchronous move runs until it arrives or a stop ends it, the async operation register behaves as the ur_rtde control
script writes it (an operation id that changes at every start, a running bit, visible only after a reporting delay), and
the stop flags, the program state and the joints read as a controller's would. Every call is written down with the thread
it came from, which is how a test sees that a brake was sent by the thread that moved the arm.

Neither double knows anything about the halt: they are the controller, and the halt is the driver's.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------------------------------------------
# A UR arm on a double controller
# ------------------------------------------------------------------------------------------------------------------


class AcceptingPreflight:
    """Every gate passes: the halt tests are about what reaches the controller after the gates, not the gates."""

    def context_for_pose(self, *_a: Any, **_k: Any) -> object:
        return object()

    def evaluate(self, *_a: Any, **_k: Any) -> Any:
        from src.robot.safety import SafetyDecision, SafetyReason

        return SafetyDecision(accepted=True, reason=SafetyReason.OK, guard="halt-tests", message="")

    def gate_joint_target(self, *_a: Any, **_k: Any) -> None:
        return None

    def set_perceived_obstacles(self, *_a: Any, **_k: Any) -> int:
        return 0


def ur_config(planner: str = "ik", **ur: Any) -> Any:
    """A ur5e cell on ``planner`` with a declared 132 mm tool and a box nothing in these tests leaves."""
    from src.config.schema.robot import RobotConfig

    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"motion_planner": planner, **ur},
        "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": "willy", "offset_mm": [0.0, 0.0, 132.0], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
        "workspace_limits": {"x_min": -2000.0, "x_max": 2000.0, "y_min": -2000.0, "y_max": 2000.0,
                             "z_min": -2000.0, "z_max": 2000.0},
        "motion_limits": {"max_velocity": 0.6, "max_acceleration": 0.9},
    })


def no_client() -> Any:
    raise AssertionError("executing a judged path never starts the planner")


def ur_arm_on(controller: Any, planner: str = "ik", **ur: Any) -> Any:
    """A ur5e on ``planner`` whose controller is ``controller`` (any double with ``attach``), every gate accepting.

    On cuRobo its planner glue is the real ``CuroboUrPlanner`` on the arm's own connection, built with no client: a
    judged path runs without starting a planner.
    """
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner

    arm = URRobotArm(ur_config(planner, **ur))
    controller.attach(arm._conn)
    arm._preflight = AcceptingPreflight()
    if planner == "curobo":
        arm._curobo_ur = CuroboUrPlanner(arm._conn, client_factory=no_client,
                                         vel=arm.config.motion_limits.max_velocity,
                                         acc=arm.config.motion_limits.max_acceleration)
    return arm


# ------------------------------------------------------------------------------------------------------------------
# The recording interfaces
# ------------------------------------------------------------------------------------------------------------------


def plain(value: Any) -> Any:
    """``value`` as JSON: tuples and arrays become lists, numpy scalars Python numbers, a URPose its list."""
    if isinstance(value, (bool, int, str)) or value is None:
        return value
    if isinstance(value, float):
        return float(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    to_list = getattr(value, "to_ur_list", None)
    if callable(to_list):
        return plain(to_list())
    raise TypeError(f"the recording double cannot write down a {type(value).__name__}: {value!r}")


class _Interface:
    """One ur_rtde interface: every call is written onto the shared log and answered from ``answers``.

    An answer is a value, or a callable given the call's arguments. A method with no answer raises ``AttributeError``,
    which is how a call the driver never made before shows up.
    """

    def __init__(self, name: str, log: list[list[Any]], answers: dict[str, Any]) -> None:
        self._name = name
        self._log = log
        self._answers = answers

    def __getattr__(self, method: str) -> Callable[..., Any]:
        if method.startswith("_") or method not in self._answers:
            raise AttributeError(f"the recording {self._name} interface was never asked {method}()")
        answer = self._answers[method]

        def call(*args: Any, **kwargs: Any) -> Any:
            self._log.append([self._name, method, plain(list(args)), plain(kwargs)])
            if isinstance(answer, BaseException):
                raise answer
            if callable(answer):
                return answer(*args, **kwargs)
            return answer

        return call


class RecordingRtde:
    """The four ur_rtde interfaces of one connection, writing every call onto :attr:`calls`.

    ``fk`` answers ``getForwardKinematics`` with the flange of the model's DH chain, so the singularity check of the ik
    path reads a real arm. ``here`` is where the arm stands (``getActualQ``) and ``ik`` what the controller's inverse
    kinematics answers for every pose. ``overrides`` replaces an answer per interface and method, a list being answered
    one item per call.
    """

    def __init__(
        self,
        *,
        model: str = "ur5e",
        here: Sequence[float] = (0.0, -1.2, 1.3, -0.4, 1.5, 0.2),
        ik: Sequence[float] = (0.1, -1.1, 1.2, -0.5, 1.4, 0.3),
        overrides: "dict[str, dict[str, Any]] | None" = None,
    ) -> None:
        self.calls: list[list[Any]] = []
        self.model = model
        self.here = [float(v) for v in here]
        flange = self.flange(self.here)
        ctrl: dict[str, Any] = {
            "moveJ": True,
            "moveL": True,
            "stopJ": None,
            "stopL": None,
            "getInverseKinematics": lambda *a: [float(v) for v in ik],
            "getForwardKinematics": lambda *a: self.flange(a[0]) if a else list(flange),
            "getTCPOffset": [0.0] * 6,
            "isSteady": True,
            "isProgramRunning": True,
            "setPayload": True,
            "stopScript": None,
            "disconnect": None,
        }
        recv: dict[str, Any] = {
            "getActualQ": lambda: list(self.here),
            "getActualQd": [0.0] * 6,
            "getActualTCPPose": list(flange),
            "getRobotMode": 7,
            "getSafetyMode": 1,
            "isProtectiveStopped": False,
            "isEmergencyStopped": False,
            "getDigitalOutState": False,
            "getDigitalInState": False,
            "disconnect": None,
        }
        io: dict[str, Any] = {
            "setToolDigitalOut": True,
            "setStandardDigitalOut": True,
            "setConfigurableDigitalOut": True,
            "setAnalogOutputVoltage": True,
            "setAnalogOutputCurrent": True,
            "disconnect": None,
        }
        dash: dict[str, Any] = {
            "safetystatus": "Safetystatus: NORMAL",
            "disconnect": None,
        }
        answers = {"ctrl": ctrl, "recv": recv, "io": io, "dash": dash}
        for name, table in (overrides or {}).items():
            for method, answer in table.items():
                answers[name][method] = _one_per_call(answer) if isinstance(answer, list) else answer
        self.ctrl = _Interface("ctrl", self.calls, ctrl)
        self.recv = _Interface("recv", self.calls, recv)
        self.io = _Interface("io", self.calls, io)
        self.dash = _Interface("dash", self.calls, dash)

    def flange(self, q: Sequence[float]) -> list[float]:
        """The flange of the DH chain at ``q``, as the controller answers a pose: metres and a rotation vector."""
        from src.robot.drivers.ur.pose import URPose
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        links = ur_link_transforms_mm(self.model, np.asarray([float(v) for v in q], dtype=np.float64))
        assert links is not None, f"no DH chain for {self.model}"
        return [float(v) for v in URPose.from_T(links[-1]).to_ur_list()]

    def attach(self, conn: Any) -> Any:
        """Hand ``conn`` (a ``URConnection``) these four interfaces, as a connected controller would."""
        conn._ctrl, conn._recv, conn._io, conn._dashboard = self.ctrl, self.recv, self.io, self.dash
        return conn

    def named(self, method: str) -> list[list[Any]]:
        """Every call of ``method`` on any interface, in order."""
        return [call for call in self.calls if call[1] == method]


def _one_per_call(answers: list[Any]) -> Callable[..., Any]:
    """An answer that takes the next item of ``answers`` at each call, raising it where it is an exception."""
    left = list(answers)

    def answer(*_a: Any, **_k: Any) -> Any:
        item = left.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    return answer


# ------------------------------------------------------------------------------------------------------------------
# A clock the test owns
# ------------------------------------------------------------------------------------------------------------------


class VirtualClock:
    """Time that passes only when somebody sleeps on it, and things that happen at a given time.

    ``at(t, action)`` runs ``action`` once the clock has passed ``t`` seconds, from inside the ``sleep`` that passed it:
    a halt requested "while the arm moves" without a second thread.
    """

    def __init__(self) -> None:
        self.t = 0.0
        self._events: list[tuple[float, Callable[[], None]]] = []
        self.slept = 0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept += 1
        self.t += max(0.0, float(seconds))
        due = [event for event in self._events if event[0] <= self.t]
        self._events = [event for event in self._events if event[0] > self.t]
        for _when, action in sorted(due, key=lambda event: event[0]):
            action()

    def at(self, t: float, action: Callable[[], None]) -> None:
        self._events.append((float(t), action))


# ------------------------------------------------------------------------------------------------------------------
# A controller whose moves take time
# ------------------------------------------------------------------------------------------------------------------


class AsyncStatus:
    """``rtde_control.AsyncOperationStatus`` as the control script writes it: an id per operation, a running bit."""

    def __init__(self, op_id: int, running: bool, change_count: int) -> None:
        self._op_id, self._running, self._count = op_id, running, change_count

    def operationId(self) -> int:  # noqa: N802 (ur_rtde's name)
        return self._op_id

    def isAsyncOperationRunning(self) -> bool:  # noqa: N802 (ur_rtde's name)
        return self._running

    def changeCount(self) -> int:  # noqa: N802 (ur_rtde's name)
        return self._count

    def progress(self) -> int:
        return 0 if self._running else -1

    def value(self) -> int:
        return (self._op_id << 24) | (self._count << 16) | (32768 if self._running else 0)


class SimulatedUr:
    """A UR controller with one arm: a move runs on ``clock`` at its speed along its straight line, joint or tool.

    * ``moveJ(q, v, a, True)`` / ``moveL(p, v, a, True)`` start a move and return at once; the operation id changes and
      the running bit sets, both visible only ``report_delay_s`` after the send, as an RTDE output register is.
    * ``moveJ(..., False)`` runs the move to its end before it returns, the clock passing as it runs.
    * ``stopJ(d)`` / ``stopL(d)`` end the move in flight: it decelerates at ``d`` along its line, and the call returns
      once the arm stands, the clock passing as it brakes.
    * ``kill_move_at`` ends a move part of the way at that fraction with nobody asking (a killed move thread), and
      ``protective_at`` raises a protective stop at that time; ``lag`` leaves the arm that far short of the target for
      ``lag_s`` after the operation reports finished, as a servo trails its set point.

    Every call is written onto :attr:`calls` as ``(thread ident, interface, method, args)``.
    """

    def __init__(
        self,
        here: Sequence[float] = (0.0, -1.2, 1.3, -0.4, 1.5, 0.2),
        *,
        clock: Callable[[], float] = time.monotonic,
        wait: Callable[[float], None] | None = None,
        report_delay_s: float = 0.0,
        lag: float = 0.0,
        lag_s: float = 0.0,
        tcp: Sequence[float] = (0.4, 0.1, 0.3, 0.0, 3.1416, 0.0),
    ) -> None:
        self.clock = clock
        self._wait = wait or time.sleep
        self.report_delay_s = float(report_delay_s)
        self.lag, self.lag_s = float(lag), float(lag_s)
        self.q = np.asarray([float(v) for v in here], dtype=np.float64)
        self.tcp = np.asarray([float(v) for v in tcp], dtype=np.float64)
        self.calls: list[tuple[int, str, str, list[Any]]] = []
        self.op_id = 0
        self.change_count = 0
        self.program_running = True
        self.protective = False
        self.emergency = False
        self.kill_move_at: float | None = None
        self.protective_at: float | None = None
        self.refuse_sends = False
        self._move: dict[str, Any] | None = None
        self._op_visible_at = -1.0
        self._visible_op = 0
        self._lock = threading.RLock()
        self.ctrl = _Bound(self, "ctrl")
        self.recv = _Bound(self, "recv")
        self.io = _Bound(self, "io")
        self.dash = _Bound(self, "dash")
        self.outputs: dict[int, bool] = {}
        self.model = "ur5e"
        self.ik_base = [float(v) for v in here]

    # -- bookkeeping --------------------------------------------------------------------------------------------

    def _log(self, iface: str, method: str, args: Sequence[Any]) -> None:
        self.calls.append((threading.get_ident(), iface, method, plain(list(args))))

    def attach(self, conn: Any) -> Any:
        conn._ctrl, conn._recv, conn._io, conn._dashboard = self.ctrl, self.recv, self.io, self.dash
        return conn

    def named(self, method: str) -> list[tuple[int, str, str, list[Any]]]:
        return [call for call in self.calls if call[2] == method]

    # -- the motion model ---------------------------------------------------------------------------------------

    def _space(self, kind: str) -> np.ndarray:
        return self.q if kind == "J" else self.tcp

    def _start(self, kind: str, target: Sequence[float], speed: float) -> None:
        start = self._space(kind).copy()
        goal = np.asarray([float(v) for v in target], dtype=np.float64)
        span = float(np.max(np.abs(goal - start))) if kind == "J" else float(np.linalg.norm(goal[:3] - start[:3]))
        duration = span / max(float(speed), 1e-9)
        now = self.clock()
        self._move = {"kind": kind, "start": start, "goal": goal, "t0": now, "duration": duration,
                      "speed": float(speed), "stop": None, "ended": False}
        self.op_id = (self.op_id % 127) + 1
        self.change_count = (self.change_count + 1) % 128
        self._op_visible_at = now + self.report_delay_s

    def _fraction(self, move: dict[str, Any], t: float) -> float:
        if move["stop"] is not None:
            t_stop, s_stop, s_end, t_end = move["stop"]
            if t >= t_end:
                return s_end
            # decelerating: s runs from s_stop to s_end along a falling speed
            u = (t - t_stop) / max(t_end - t_stop, 1e-12)
            return s_stop + (s_end - s_stop) * (1.0 - (1.0 - u) ** 2)
        if move["duration"] <= 0.0:
            return 1.0
        return min(1.0, max(0.0, (t - move["t0"]) / move["duration"]))

    def update(self) -> None:
        """Bring the arm to where it stands now on the clock."""
        with self._lock:
            move = self._move
            now = self.clock()
            if self.protective_at is not None and now >= self.protective_at and not self.protective:
                self.protective = True
                if move is not None and not move["ended"]:
                    self._freeze(move, now)
            if move is None or move["ended"]:
                return
            if self.kill_move_at is not None and move["stop"] is None:
                t_kill = move["t0"] + move["duration"] * self.kill_move_at
                if now >= t_kill:
                    self._place(move, self.kill_move_at)
                    move["ended"] = True
                    self.change_count = (self.change_count + 1) % 128
                    return
            s = self._fraction(move, now)
            self._place(move, s)
            finished = (move["stop"] is not None and now >= move["stop"][3]) or (move["stop"] is None and s >= 1.0)
            if finished:
                move["ended"] = True
                move["t_end"] = now
                self.change_count = (self.change_count + 1) % 128

    def _place(self, move: dict[str, Any], s: float) -> None:
        here = move["start"] + (move["goal"] - move["start"]) * s
        if move["kind"] == "J":
            self.q = here
        else:
            self.tcp = here

    def _freeze(self, move: dict[str, Any], now: float) -> None:
        s = self._fraction(move, now)
        self._place(move, s)
        move["ended"] = True
        self.change_count = (self.change_count + 1) % 128

    def _running(self) -> bool:
        move = self._move
        return move is not None and not move["ended"]

    def speeds(self) -> list[float]:
        move = self._move
        if move is None or move["ended"] or move["kind"] != "J":
            return [0.0] * 6
        span = move["goal"] - move["start"]
        rate = 1.0 / max(move["duration"], 1e-9)
        if move["stop"] is not None:
            t_stop, s_stop, s_end, t_end = move["stop"]
            u = (self.clock() - t_stop) / max(t_end - t_stop, 1e-12)
            rate = 2.0 * (s_end - s_stop) * max(0.0, 1.0 - u) / max(t_end - t_stop, 1e-12)
        return [float(v) * rate for v in span]

    def at_rest(self, seconds: float) -> None:
        """Let ``seconds`` pass on the clock, as a sync call waiting on the controller would."""
        self._wait(seconds)
        self.update()

    # -- the control interface ----------------------------------------------------------------------------------

    def moveJ(self, q: Sequence[float], speed: float = 1.05, accel: float = 1.4,  # noqa: N802 (ur_rtde's name)
              asynchronous: bool = False) -> bool:
        return self._move_cmd("J", "moveJ", q, speed, accel, asynchronous)

    def moveL(self, pose: Sequence[float], speed: float = 0.25, accel: float = 1.2,  # noqa: N802 (ur_rtde's name)
              asynchronous: bool = False) -> bool:
        return self._move_cmd("L", "moveL", pose, speed, accel, asynchronous)

    def _move_cmd(self, kind: str, name: str, target: Sequence[float], speed: float, accel: float,
                  asynchronous: bool) -> bool:
        self._log("ctrl", name, [target, speed, accel, asynchronous])
        if self.refuse_sends or not self.program_running or self.protective or self.emergency:
            return False
        with self._lock:
            self.update()
            if self._running():
                self._freeze(self._move, self.clock())  # a new move kills the one in flight, as stop_async_move()
            self._start(kind, target, speed)
        if asynchronous:
            return True
        move = self._move
        assert move is not None
        while True:
            self.update()
            if move["ended"]:
                return not self.protective and self._fraction(move, self.clock()) >= 1.0
            if self.protective or self.emergency:
                return False
            self._wait(0.004)

    def _stop_cmd(self, name: str, decel: float, asynchronous: bool) -> None:
        self._log("ctrl", name, [decel, asynchronous] if asynchronous else [decel])
        with self._lock:
            self.update()
            move = self._move
            if move is None or move["ended"]:
                return
            now = self.clock()
            s_now = self._fraction(move, now)
            span = move["goal"] - move["start"]
            length = float(np.max(np.abs(span))) if move["kind"] == "J" else float(np.linalg.norm(span[:3]))
            speed = move["speed"]
            t_brake = speed / max(float(decel), 1e-9)
            ds = (speed * speed / (2.0 * max(float(decel), 1e-9))) / max(length, 1e-12)
            s_end = min(1.0, s_now + ds)
            move["stop"] = (now, s_now, s_end, now + t_brake)
        if not asynchronous:
            while True:
                self.update()
                if move["ended"]:
                    return
                self._wait(0.004)

    def stopJ(self, a: float = 2.0, asynchronous: bool = False) -> None:  # noqa: N802 (ur_rtde's name)
        self._stop_cmd("stopJ", a, asynchronous)

    def stopL(self, a: float = 10.0, asynchronous: bool = False) -> None:  # noqa: N802 (ur_rtde's name)
        self._stop_cmd("stopL", a, asynchronous)

    def getAsyncOperationProgressEx(self) -> AsyncStatus:  # noqa: N802 (ur_rtde's name)
        self._log("ctrl", "getAsyncOperationProgressEx", [])
        with self._lock:
            self.update()
            now = self.clock()
            if now >= self._op_visible_at:
                self._visible_op = self.op_id
            if self._visible_op != self.op_id:
                # the register still shows the operation before this one, which had finished
                return AsyncStatus(self._visible_op, False, self.change_count)
            return AsyncStatus(self.op_id, self._running(), self.change_count)

    def isProgramRunning(self) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("ctrl", "isProgramRunning", [])
        return self.program_running

    def isSteady(self) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("ctrl", "isSteady", [])
        self.update()
        return not self._running()

    def getInverseKinematics(self, tcp: Sequence[float], *_rest: Any) -> list[float]:  # noqa: N802 (ur_rtde's name)
        """A configuration near :attr:`ik_base` for every pose: lower poses bend the shoulder and the elbow further."""
        self._log("ctrl", "getInverseKinematics", [tcp, *_rest])
        dz = 0.45 - float(tcp[2])
        base = list(self.ik_base)
        return [base[0] + 0.5 * float(tcp[1]) + 0.2, base[1] + 0.6 * dz, base[2] - 0.6 * dz, *base[3:]]

    def getForwardKinematics(self, q: Sequence[float] | None = None,  # noqa: N802 (ur_rtde's name)
                             offset: Sequence[float] | None = None) -> list[float]:
        self._log("ctrl", "getForwardKinematics", [] if q is None else [q, offset])
        from src.robot.drivers.ur.pose import URPose
        from src.robot.safety._ur_kinematics import ur_link_transforms_mm

        at = self.q if q is None else np.asarray([float(v) for v in q], dtype=np.float64)
        links = ur_link_transforms_mm(self.model, at)
        assert links is not None
        return [float(v) for v in URPose.from_T(links[-1]).to_ur_list()]

    def getTCPOffset(self) -> list[float]:  # noqa: N802 (ur_rtde's name)
        self._log("ctrl", "getTCPOffset", [])
        return [0.0] * 6

    # -- the receive interface ----------------------------------------------------------------------------------

    def getActualQ(self) -> list[float]:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getActualQ", [])
        self.update()
        q = [float(v) for v in self.q]
        move = self._move
        if move is not None and move["ended"] and move["kind"] == "J" and self.lag:
            t_end = move.get("t_end", -math.inf)
            if self.clock() < t_end + self.lag_s:
                q = [v - self.lag for v in q]
        return q

    def getActualQd(self) -> list[float]:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getActualQd", [])
        self.update()
        return self.speeds()

    def getActualTCPPose(self) -> list[float]:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getActualTCPPose", [])
        self.update()
        return [float(v) for v in self.tcp]

    def isProtectiveStopped(self) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "isProtectiveStopped", [])
        self.update()
        return self.protective

    def isEmergencyStopped(self) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "isEmergencyStopped", [])
        return self.emergency

    def getRobotMode(self) -> int:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getRobotMode", [])
        return 7

    def getSafetyMode(self) -> int:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getSafetyMode", [])
        self.update()
        return 3 if self.protective else 1

    def getDigitalOutState(self, index: int) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("recv", "getDigitalOutState", [index])
        return self.outputs.get(int(index), False)

    # -- the I/O interface --------------------------------------------------------------------------------------

    def setToolDigitalOut(self, pin: int, value: bool) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("io", "setToolDigitalOut", [pin, value])
        self.outputs[16 + int(pin)] = bool(value)
        return True

    def setStandardDigitalOut(self, pin: int, value: bool) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("io", "setStandardDigitalOut", [pin, value])
        self.outputs[int(pin)] = bool(value)
        return True

    def setAnalogOutputVoltage(self, pin: int, value: float) -> bool:  # noqa: N802 (ur_rtde's name)
        self._log("io", "setAnalogOutputVoltage", [pin, value])
        return True

    def safetystatus(self) -> str:
        self._log("dash", "safetystatus", [])
        return "Safetystatus: PROTECTIVE_STOP" if self.protective else "Safetystatus: NORMAL"

    def disconnect(self) -> None:
        return None

    def stopScript(self) -> None:  # noqa: N802 (ur_rtde's name)
        self._log("ctrl", "stopScript", [])


class _Bound:
    """One interface of a :class:`SimulatedUr`: the methods of that interface and nothing else."""

    _METHODS = {
        "ctrl": {"moveJ", "moveL", "stopJ", "stopL", "getAsyncOperationProgressEx", "isProgramRunning", "isSteady",
                 "stopScript", "disconnect", "getInverseKinematics", "getForwardKinematics", "getTCPOffset"},
        "recv": {"getActualQ", "getActualQd", "getActualTCPPose", "isProtectiveStopped", "isEmergencyStopped",
                 "getRobotMode", "getSafetyMode", "getDigitalOutState", "disconnect"},
        "io": {"setToolDigitalOut", "setStandardDigitalOut", "setAnalogOutputVoltage", "disconnect"},
        "dash": {"safetystatus", "disconnect"},
    }

    def __init__(self, sim: SimulatedUr, name: str) -> None:
        self._sim, self._name = sim, name

    def __getattr__(self, method: str) -> Any:
        if method not in self._METHODS[self._name]:
            raise AttributeError(f"the simulated {self._name} interface has no {method}()")
        return getattr(self._sim, method)
