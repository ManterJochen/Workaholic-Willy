"""Measure "halt now" against a real UR controller (URSim CB3): M0-M9 of the build plan's halt checklist (6.5, item 1).

The console's "Sofort anhalten" latches the arm (``arm.halt``): nothing is sent and no output switched until a person
says the cell is clear. ``robot.ur.brake_on_halt`` decides what happens to the move in flight. Off, as shipped, it runs
to its end. On, every move is sent asynchronously and watched by the thread that sent it, which brakes it with
``stopJ`` or ``stopL`` at max(2.0, the move's own acceleration) and answers ``True`` only at the target. The owner
switches it on in the cell's own profile only after these measurements have passed:

  M0  baseline, brakes off. (a) A cross-thread ``URConnection.stop()`` during a synchronous moveJ: does the move stop,
      or arrive anyway? (b) A halt during a synchronous moveJ: the move runs out, the latch's record says it ran out
      (``HaltState.brake``), and the next one is refused.
  M1  brake latency, brakes on: a halt during a joint move at cruise. Request to stopJ sent (Python), to the
      controller's target speed falling (the deceleration starts), to standstill. Pass: deceleration within 50 ms, the
      stop sent by the moving thread, and the latch's record ends ``braked``.
  M2  the stop point on the judged leg: M1's braked moves, on a diagonal joint line, measured against that line in
      joint space. Pass: within 1e-3 rad of the line. Also the stopping distance at max(2.0, a) and at the move's own a,
      the deceleration choice the plan leaves to this measurement.
  M3  200-move parity, brakes on: 200 watched moves with no halt. Pass: zero early returns (a True with the arm off
      its target or still moving) and zero refusals. The first 20 again with brakes off, for the durations.
  M3b the same for lines, brakes on: 50 watched moveL (``--lines``) with no halt, each turning the tool up to 0.2 rad
      about its axis. Pass: every one True and ``arrived`` (so the controller showed every asynchronous moveL in its
      operation register: a line it never showed is stopped and refused after 1 s), the TCP within 1 mm and 2e-3 rad
      of the target and the arm still, and zero refusals. The first 10 again with brakes off, for the durations. Every
      line of a pick (the approach, the line in, the line out) is a moveL, so this is what the owner's picks meet.
  M4  a multi-waypoint path (``CuroboUrPlanner.execute``, 5 waypoints) halted in leg 3, brakes on. Pass: CANCELLED
      naming waypoint 3, exactly 3 moveJ on the wire, and the arm never heads for waypoint 4.
  M5  the owner's toggle on tool DO0 (``JawIOGripper`` single_toggle): a halt mid-approach, then the close the pick
      would send. Pass: zero DO0 edges after the halt, read on a second RTDE connection, brakes on and off.
  M6  a line braked with stopL, brakes on, halted 1.2 s into a 2 s line: past the 1 s a watched move has to show in
      the register, so ``braked`` also says the register showed the line. Pass: ``braked``, one stopL, and the TCP
      stops within 1 mm of the line.
  M7  a protective stop during a watched move, brakes on: the controller's own (``triggerProtectiveStop``), because a
      CB3's IK refuses the on-axis pose probe_protective_stop.py drives through. Pass: False, never True, no moveJ
      after it, and a protective stop really seen: the connection names it as the cause (``last_stop_cause``), or the
      second RTDE connection read it. A program that merely ended is no protective stop and fails M7. The stop is
      released at the end through the dashboard, as the URSim probes do; a real cell releases it at the pendant, and
      the console never does.
  M8  alternative B, the program stop: the dashboard ``stop`` during a synchronous moveJ. How far the arm runs on,
      and whether DO0 changes when the program stops.
  M9  the latch outlives a disconnect and a connect of the same arm: a move and a DO write refused, until cleared.

Everything runs on 127.0.0.1 and nowhere else: the probe moves the arm and switches tool DO0 without asking, so it has no
host flag, and it refuses a profile whose ``ur.ip`` is not the loopback. It writes its own scratch profile layer into a
copy of ``config/`` in a temporary directory (``brake_on_halt: true``, ``payload.length_mm``, the toggle on tool DO0)
and loads ``ursim,<model>,haltprobe`` from there; no shipped profile changes. Before any check it confirms the layer
reached the arm (``brakes_in_motion()`` true from the loaded profile: the config-to-connection wiring); each check then
sets the brake on its connection itself, on or off as that check measures.

Where it departs from the build plan's checklist (6.5): the chain is ``ursim,<model>,haltprobe``, an ik arm, not
``ursim,ursim_curobo``, because every check drives the connection, the arm's halt and the glue's ``execute`` directly and
none needs a planner; and M4 runs ``CuroboUrPlanner.execute`` on five waypoints the probe writes (the legs a planned path
runs, one moveJ each), with a planner that refuses to start, not on a path cuRobo planned. A planned M4 needs the
cuRobo sidecar beside URSim.

Start the CB3 controller as ``scripts/ursim/README.md`` and the UR section of ``docs/runbooks/cell_bringup.md`` say
(``URSIM_IMAGE=universalrobots/ursim_cb3:latest URSIM_NAME=ursim_cb3 URSIM_FRESH=1 ursim.sh up UR10``), confirm its
"Confirm Safety Configuration" dialog at the noVNC pendant, then:

    python scripts/ursim/probe_halt.py                    # M0-M6 (M3b included) and M9
    python scripts/ursim/probe_halt.py --only M7,M8       # the two that stop the controller's program
    python scripts/ursim/probe_halt.py --json halt.json   # the measurements as one JSON record as well

Exit codes: 0 every check that ran passed, 1 a check failed, 2 no controller reachable or not a loopback UR.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import socket
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import src...` fails without
# the repository root on a checkout that was never installed with `pip install -e . --no-deps`; the
# install puts `src` on the path from anywhere.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

from src.config.loader import load_robot_config  # noqa: E402
from src.robot.core import ArmHalted, MotionCommand, MotionStatus, RobotVendor  # noqa: E402
from src.robot.core.arm_capabilities import DigitalIOPort  # noqa: E402
from src.robot.drivers import create_arm  # noqa: E402
from src.robot.drivers.ur import connection as connection_module  # noqa: E402
from src.robot.drivers.ur.arm import URRobotArm  # noqa: E402
from src.robot.drivers.ur.connection import MoveEnd  # noqa: E402
from src.robot.drivers.ur.pose import URPose  # noqa: E402
from src.robot.drivers.ur.curobo_motion import CuroboUrPlanner  # noqa: E402
from src.robot.execution.handling import _controller_refusal  # noqa: E402
from src.robot.grippers.jaw_io import JawIOGripper  # noqa: E402

HOST = "127.0.0.1"
_DASHBOARD_PORT = 29999
_OK, _FAILED, _BAD_REQUEST = 0, 1, 2

#: Tool down, upper arm up, forearm level: free space on every URSim model, and the start of every move here.
HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
#: A diagonal joint line from HOME: four joints turn together, so a stop point off the line shows on any of them.
DIAGONAL = [1.2, -1.2708, 1.2708, -1.0708, -1.5708, 0.6]
#: The tool output the owner's toggle hand is on, as RTDE numbers digital outputs: tool 0 is bit 16.
_TOOL_DO0_BIT = 16

_LAYER = """\
# probe_halt.py's own layer, written into a copy of config/ in a temporary directory; never into config/.
robot:
  ur:
    model: "{model}"
    brake_on_halt: true
  safety:
    planning_world:
      payload:
        length_mm: 60.0
  gripper:
    jaw_io:
      actuation: single_toggle
      close_output_pin: 0
      io_port: tool
      pulse_s: 0.05
      close_settle_s: 0.0
"""


# ------------------------------------------------------------------------------------------------------------------
# What is measured with
# ------------------------------------------------------------------------------------------------------------------


def dash(*commands: str) -> list[str]:
    """Speak the dashboard protocol over a raw socket: the probe's own door, before and beside any connection."""
    sock = socket.create_connection((HOST, _DASHBOARD_PORT), timeout=10)
    sock.settimeout(10)
    sock.recv(256)
    answers = []
    for command in commands:
        sock.sendall((command + "\n").encode())
        time.sleep(0.3)
        answers.append(sock.recv(1024).decode(errors="replace").strip())
    sock.close()
    return answers


class Watcher:
    """A second RTDE receive connection sampling the arm at the controller's rate, with ``perf_counter`` times.

    It reads what the probe's own connection never asks: the controller's target joint speeds, which fall the moment a
    stop starts, every digital output bit, so an edge of tool DO0 is seen whoever switched it, and the protective-stop
    flag, so M7 sees a protective stop whatever ended the move.
    """

    def __init__(self) -> None:
        import rtde_receive

        self._recv = rtde_receive.RTDEReceiveInterface(HOST)
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="probe-watcher", daemon=True)
        self._lock = threading.Lock()

    def __enter__(self) -> Watcher:
        self._thread.start()
        time.sleep(0.2)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(2.0)
        try:
            self._recv.disconnect()
        except Exception:  # noqa: BLE001 (a probe's teardown)
            pass

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            stamp = self._recv.getTimestamp()
            if stamp != last:
                last = stamp
                sample = {
                    "t": time.perf_counter(), "stamp": stamp, "q": list(self._recv.getActualQ()),
                    "qd": list(self._recv.getActualQd()), "target_qd": list(self._recv.getTargetQd()),
                    "tcp": list(self._recv.getActualTCPPose()),
                    "do": int(self._recv.getActualDigitalOutputBits()),
                    "protective": bool(self._recv.isProtectiveStopped()),
                }
                with self._lock:
                    self.samples.append(sample)
            time.sleep(0.001)

    def since(self, t: float) -> list[dict[str, Any]]:
        with self._lock:
            return [s for s in self.samples if s["t"] >= t]

    def latest(self) -> dict[str, Any]:
        with self._lock:
            return dict(self.samples[-1])

    def deceleration_after(self, t: float) -> float | None:
        """When, after ``t``, the controller's target speed first fell below 97 % of what it was at ``t``."""
        rows = self.since(t - 0.05)
        before = [s for s in rows if s["t"] <= t]
        if not before:
            return None
        cruise = max(abs(v) for v in before[-1]["target_qd"])
        if cruise < 1e-3:
            return None
        for s in rows:
            if s["t"] > t and max(abs(v) for v in s["target_qd"]) < 0.97 * cruise:
                return float(s["t"])
        return None

    def still_after(self, t: float, *, rad_s: float = 0.01) -> float | None:
        for s in self.since(t):
            if max(abs(v) for v in s["qd"]) <= rad_s:
                return float(s["t"])
        return None

    def do0_edges(self, t0: float, t1: float | None = None) -> int:
        rows = [s for s in self.since(t0 - 0.05) if t1 is None or s["t"] <= t1]
        bits = [(s["do"] >> _TOOL_DO0_BIT) & 1 for s in rows]
        return sum(1 for a, b in zip(bits, bits[1:]) if a != b)

    def protective_after(self, t: float) -> bool:
        """Whether the controller reported a protective stop at any sample after ``t``."""
        return any(s["protective"] for s in self.since(t))


class Timed:
    """The control interface, with every motion command written down: time, thread, name and arguments."""

    _WATCHED = {"moveJ", "moveL", "stopJ", "stopL", "stopScript"}

    def __init__(self, ctrl: Any, log: list[tuple[float, int, str, tuple[Any, ...]]]) -> None:
        self._ctrl = ctrl
        self._log = log

    def __getattr__(self, name: str) -> Any:
        attribute = getattr(self._ctrl, name)
        if name not in self._WATCHED:
            return attribute

        def call(*args: Any) -> Any:
            self._log.append((time.perf_counter(), threading.get_ident(), name, args))
            return attribute(*args)

        return call


class Probe:
    """One arm on the loopback controller, built from the probe's scratch layer, and the checks' results."""

    def __init__(self, model: str) -> None:
        self.model = model
        self.tree = Path(tempfile.mkdtemp(prefix="willy_probe_halt_")) / "config"
        shutil.copytree(_REPO_ROOT / "config", self.tree)
        (self.tree / "robot" / "robot.haltprobe.yaml").write_text(_LAYER.format(model=model), encoding="utf-8")
        self.cfg = load_robot_config(self.tree, profile=f"ursim,{model},haltprobe")
        if self.cfg.ur.ip not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit(f"profile ur.ip is {self.cfg.ur.ip!r}: this probe moves the arm and runs on the loopback")
        built = create_arm(RobotVendor.from_string(self.cfg.vendor), config=self.cfg)
        if not isinstance(built, URRobotArm):
            raise SystemExit(f"the scratch profile builds {type(built).__name__}, not a UR arm")
        self.arm = built
        self.commands: list[tuple[float, int, str, tuple[Any, ...]]] = []
        self.results: dict[str, Any] = {}
        self.passed: dict[str, bool] = {}
        # The layer's brake_on_halt reached the arm through the config, before any check sets it on the connection.
        wired = self.arm.brakes_in_motion()
        self.results["profile"] = {"chain": f"ursim,{model},haltprobe", "brake_on_halt": self.cfg.ur.brake_on_halt,
                                   "brakes_in_motion": wired}
        self.passed["profile"] = wired is True

    @property
    def conn(self) -> Any:
        return self.arm.connection

    def connect(self) -> None:
        self.arm.connect()
        self.conn._ctrl = Timed(self.conn._ctrl, self.commands)

    def reconnect(self) -> None:
        try:
            self.arm.disconnect()
        except Exception:  # noqa: BLE001 (a probe's teardown)
            pass
        time.sleep(1.0)
        self.connect()

    def healthy(self) -> None:
        """Back to HOME, brakes off, no latch, the program running: the start of every check."""
        self.conn.clear_halt()
        self.conn.brake_on_halt = False
        if not self.conn.is_program_running():
            self.reconnect()
        if not self.conn.moveJ(HOME, 0.8, 1.2):
            raise RuntimeError(f"could not return HOME: {self.conn.last_move_end}")

    def sent(self, name: str, since: float) -> list[tuple[float, int, str, tuple[Any, ...]]]:
        return [c for c in self.commands if c[2] == name and c[0] >= since]

    def moving(self, verb: Callable[[], Any]) -> tuple[threading.Thread, list[Any], float]:
        """Run ``verb`` on a thread of its own, as the console's run thread does; its answer lands in the list."""
        answer: list[Any] = []
        started = time.perf_counter()
        thread = threading.Thread(target=lambda: answer.append(verb()), name="probe-move", daemon=True)
        thread.start()
        return thread, answer, started


def _no_planner() -> Any:
    """M4 runs a waypoint list the probe wrote: executing it never starts a planner."""
    raise RuntimeError("probe_halt.py executes its own waypoints and starts no planner")


def _off_line(q: list[float], start: list[float], target: list[float]) -> tuple[float, float]:
    """How far joint vector ``q`` stands off the line start->target (rad, worst joint), and how far along it (0..1)."""
    span = [b - a for a, b in zip(start, target)]
    norm2 = sum(v * v for v in span) or 1e-12
    s = sum((qq - a) * v for qq, a, v in zip(q, start, span)) / norm2
    off = max(abs(qq - (a + s * v)) for qq, a, v in zip(q, start, span))
    return off, s


def _ms(seconds: float | None) -> float | None:
    return None if seconds is None else round(1000.0 * seconds, 1)


# ------------------------------------------------------------------------------------------------------------------
# The checks
# ------------------------------------------------------------------------------------------------------------------


def m0(p: Probe, w: Watcher) -> None:
    """Brakes off: a cross-thread stop() does not stop a synchronous move; the latch lets the move in flight run out."""
    p.healthy()
    target = list(HOME)
    target[0] += 1.2
    thread, answer, started = p.moving(lambda: p.conn.moveJ(target, 0.5, 1.0))
    time.sleep(0.8)
    asked = time.perf_counter()
    p.conn.stop()  # stopJ and stopL from this thread, as the arm's stop() did before the halt
    returned = time.perf_counter()
    thread.join(10.0)
    ended = w.still_after(asked + 0.05)
    arrived = abs(w.latest()["q"][0] - target[0]) < 2e-3
    decel = w.deceleration_after(asked)
    natural_end = started + 1.2 / 0.5 + 0.5  # cruise plus the ramps at 1.0 rad/s^2
    p.results["M0a"] = {
        "moveJ_returned": answer[:1], "arrived_at_target": arrived,
        "stop_returned_after_ms": _ms(returned - asked),
        "deceleration_after_stop_ms": _ms(None if decel is None else decel - asked),
        "deceleration_before_the_natural_end": decel is not None and decel < natural_end - 0.6,
        "still_after_stop_ms": _ms(None if ended is None else ended - asked),
        "program_running_after": p.conn.is_program_running(),
    }
    print(f"  M0a cross-thread stop(): the move {'ARRIVED anyway' if arrived else 'was cut short'}, stop() returned "
          f"after {_ms(returned - asked)} ms")
    p.passed["M0a"] = True  # a measurement of the baseline, not a check

    p.healthy()
    thread, answer, started = p.moving(lambda: p.conn.moveJ(target, 0.5, 1.0))
    time.sleep(0.8)
    t_req = time.perf_counter()
    state = p.conn.request_halt("probe M0b: halt with brakes off")
    thread.join(10.0)
    arrived = abs(w.latest()["q"][0] - target[0]) < 2e-3
    after = p.conn.halt_state()
    refused = p.conn.moveJ(HOME, 0.5, 1.0)
    end = p.conn.last_move_end
    time.sleep(0.5)
    moved_after = max(abs(v) for s in w.since(time.perf_counter() - 0.4) for v in s["qd"])
    record = None if after is None else after.brake
    p.results["M0b"] = {"in_motion": state.in_motion, "move_in_flight_returned": answer[:1],
                        "arrived_at_target": arrived, "record": record, "next_move_answer": refused,
                        "next_move_end": end.value, "moveJ_sent_after_halt": len(p.sent("moveJ", t_req)),
                        "speed_after_rad_s": moved_after}
    ok = answer[:1] == [True] and arrived and refused is False and end is MoveEnd.REFUSED_HALTED and \
        len(p.sent("moveJ", t_req)) == 0 and record == "ran_out"
    p.passed["M0b"] = ok
    print(f"  M0b halt, brakes off: in flight ran out={arrived}, next move refused={refused is False}: "
          f"{'PASS' if ok else 'FAIL'}")
    p.conn.clear_halt()


def m1_m2(p: Probe, w: Watcher) -> None:
    """Brakes on: the latency of the brake, and where the arm stops, against the diagonal joint line it ran."""
    runs: list[dict[str, Any]] = []
    for speed, accel, floor in ((0.5, 1.0, 2.0), (1.0, 1.4, 2.0), (1.0, 1.4, 0.0)):
        for _ in range(3):
            p.healthy()
            p.conn.brake_on_halt = True
            saved = connection_module._BRAKE_MIN_DECEL
            connection_module._BRAKE_MIN_DECEL = floor  # 0.0: the move's own acceleration, for the comparison
            try:
                thread, answer, started = p.moving(lambda: p.conn.moveJ(DIAGONAL, speed, accel))
                time.sleep(speed / accel + 0.35)
                t_req = time.perf_counter()
                p.conn.request_halt("probe M1: halt now")
                thread.join(10.0)
            finally:
                connection_module._BRAKE_MIN_DECEL = saved
            stops = p.sent("stopJ", t_req)
            decel = w.deceleration_after(t_req)
            still = w.still_after(t_req + 0.02)
            time.sleep(0.2)
            q = w.latest()["q"]
            off, along = _off_line(q, HOME, DIAGONAL)
            state = p.conn.halt_state()
            runs.append({
                "speed": speed, "accel": accel, "decel": max(floor, accel), "answer": answer[:1],
                "end": p.conn.last_move_end.value,
                "stop_sent_ms": _ms(stops[0][0] - t_req) if stops else None,
                "stop_thread_is_moving_thread": bool(stops) and stops[0][1] == thread.ident,
                "deceleration_ms": _ms(None if decel is None else decel - t_req),
                "still_ms": _ms(None if still is None else still - t_req),
                "brake_s": None if state is None else state.brake_s,
                "record": None if state is None else state.brake,
                "off_line_rad": off, "along": along,
            })
    p.healthy()
    p.results["M1"] = runs
    braked = [r for r in runs if r["decel"] >= 2.0]
    latencies = [r["deceleration_ms"] for r in braked if r["deceleration_ms"] is not None]
    ok1 = (all(r["answer"] == [False] and r["end"] == "braked" and r["stop_thread_is_moving_thread"]
               and r["record"] == "braked" for r in runs)
           and len(latencies) == len(braked) and max(latencies) <= 50.0)
    p.passed["M1"] = ok1
    print(f"  M1 brake latency: stop sent {[r['stop_sent_ms'] for r in runs]} ms, deceleration "
          f"{[r['deceleration_ms'] for r in runs]} ms, still {[r['still_ms'] for r in runs]} ms: "
          f"{'PASS' if ok1 else 'FAIL'}")
    worst = max(r["off_line_rad"] for r in runs)
    by_decel: dict[float, list[float]] = {}
    for r in runs:
        by_decel.setdefault(r["decel"], []).append(r["along"])
    p.results["M2"] = {"worst_off_line_rad": worst,
                       "along_the_line_at_stop": {str(k): v for k, v in by_decel.items()}}
    ok2 = worst <= 1e-3
    p.passed["M2"] = ok2
    print(f"  M2 stop point: worst {worst:.2e} rad off the judged joint line: {'PASS' if ok2 else 'FAIL'}")


def m3(p: Probe, w: Watcher, moves: int) -> None:
    """Brakes on: ``moves`` watched moves, no halt: every True at the target with the arm still."""
    rng = random.Random(20261001)
    targets = []
    for _ in range(10):
        t = list(HOME)
        for joint in (0, 1, 2, 5):
            t[joint] += rng.uniform(-0.25, 0.25)
        targets.append(t)
    p.healthy()
    p.conn.brake_on_halt = True
    early, refused, durations = [], 0, []
    for index in range(moves):
        target = targets[index % len(targets)] if index % 2 == 0 else HOME
        t0 = time.perf_counter()
        ok = p.conn.moveJ(target, 1.0, 1.4)
        durations.append(time.perf_counter() - t0)
        if not ok:
            refused += 1
            continue
        q = p.conn.get_joint_positions()
        speed = max(abs(v) for v in p.conn.get_joint_speeds())
        off = max(abs(a - b) for a, b in zip(q, target))
        if off > 2e-3 or speed > 0.01:
            early.append({"move": index, "off_rad": off, "speed_rad_s": speed})
    p.conn.brake_on_halt = False
    synchronous = []
    for index in range(min(20, moves)):
        target = targets[index % len(targets)] if index % 2 == 0 else HOME
        t0 = time.perf_counter()
        p.conn.moveJ(target, 1.0, 1.4)
        synchronous.append(time.perf_counter() - t0)
    paired = [a - b for a, b in zip(durations, synchronous)]
    p.results["M3"] = {"moves": moves, "early_returns": early, "refused": refused,
                       "median_watched_ms": _ms(statistics.median(durations)),
                       "median_synchronous_ms": _ms(statistics.median(synchronous)),
                       "median_extra_ms": _ms(statistics.median(paired))}
    ok = not early and refused == 0
    p.passed["M3"] = ok
    print(f"  M3 parity: {moves} moves, {len(early)} early returns, {refused} refused, median watched "
          f"{_ms(statistics.median(durations))} ms against {_ms(statistics.median(synchronous))} ms synchronous: "
          f"{'PASS' if ok else 'FAIL'}")


def _turned_about_the_tool(pose: URPose, dx_mm: float, dy_mm: float, dz_mm: float, turn_rad: float) -> list[float]:
    """``pose`` moved by (dx, dy, dz) mm in the base and turned ``turn_rad`` about its own tool z, as a UR list."""
    c, s = math.cos(turn_rad), math.sin(turn_rad)
    about_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    moved = pose.to_T()
    moved[:3, :3] = moved[:3, :3] @ about_z
    moved[:3, 3] += [dx_mm, dy_mm, dz_mm]
    return URPose.from_T(moved).to_ur_list()


def m3b(p: Probe, w: Watcher, lines: int) -> None:
    """Brakes on: ``lines`` watched moveL, no halt: every True at the target, every line shown in the register."""
    rng = random.Random(20261002)
    p.healthy()
    start = URPose.from_ur_list(list(p.conn.get_tcp_pose()))
    targets = [_turned_about_the_tool(start, rng.uniform(-50, 50), rng.uniform(-50, 50), rng.uniform(-40, 40),
                                      rng.uniform(-0.2, 0.2)) for _ in range(10)]
    home_line = start.to_ur_list()
    p.conn.brake_on_halt = True
    early: list[dict[str, Any]] = []
    ends: dict[str, int] = {}
    refused, durations = 0, []
    for index in range(lines):
        target = targets[index % len(targets)] if index % 2 == 0 else home_line
        t0 = time.perf_counter()
        ok = p.conn.moveL(target, 0.25, 1.2)
        durations.append(time.perf_counter() - t0)
        end = p.conn.last_move_end.value
        ends[end] = ends.get(end, 0) + 1
        if not ok:
            refused += 1
            continue
        here = URPose.from_ur_list(list(p.conn.get_tcp_pose()))
        goal = URPose.from_ur_list(target)
        off_mm, turn_rad = here.distance_to(goal), math.radians(here.angle_to(goal))
        speed = max(abs(v) for v in p.conn.get_joint_speeds())
        if off_mm > 1.0 or turn_rad > 2e-3 or speed > 0.01 or end != MoveEnd.ARRIVED.value:
            early.append({"line": index, "end": end, "off_mm": off_mm, "turn_rad": turn_rad, "speed_rad_s": speed})
    p.conn.brake_on_halt = False
    synchronous = []
    for index in range(min(10, lines)):
        target = targets[index % len(targets)] if index % 2 == 0 else home_line
        t0 = time.perf_counter()
        p.conn.moveL(target, 0.25, 1.2)
        synchronous.append(time.perf_counter() - t0)
    p.conn.moveL(home_line, 0.25, 1.2)
    p.results["M3b"] = {"lines": lines, "ends": ends, "early_returns": early, "refused": refused,
                        "median_watched_ms": _ms(statistics.median(durations)),
                        "median_synchronous_ms": _ms(statistics.median(synchronous))}
    ok = not early and refused == 0 and ends.get(MoveEnd.ARRIVED.value, 0) == lines
    p.passed["M3b"] = ok
    print(f"  M3b lines: {lines} moveL, ends {ends}, {len(early)} early returns, {refused} refused, median watched "
          f"{_ms(statistics.median(durations))} ms against {_ms(statistics.median(synchronous))} ms synchronous: "
          f"{'PASS' if ok else 'FAIL'}")


def m4(p: Probe, w: Watcher) -> None:
    """Brakes on: a 5-waypoint path halted in leg 3 sends no later waypoint."""
    p.healthy()
    p.conn.brake_on_halt = True
    path = [list(HOME)] + [[HOME[0] + 0.25 * k, *HOME[1:]] for k in range(1, 5)]
    planner = CuroboUrPlanner(p.conn, client_factory=_no_planner, vel=0.5, acc=1.0)
    started = time.perf_counter()

    def halt_in_leg_three() -> None:
        while len(p.sent("moveJ", started)) < 3:
            time.sleep(0.002)
        time.sleep(0.3)
        p.conn.request_halt("probe M4: halt in leg 3")

    halter = threading.Thread(target=halt_in_leg_three, daemon=True)
    halter.start()
    result = planner.execute(path, command=MotionCommand.MOVE_JOINTS)
    halter.join(5.0)
    time.sleep(0.5)
    q0 = w.latest()["q"][0]
    sent = p.sent("moveJ", started)
    p.results["M4"] = {"status": result.status.value, "message": result.message, "moveJ_sent": len(sent),
                       "stopJ_sent": len(p.sent("stopJ", started)), "base_joint_at_rest": q0,
                       "waypoint_3": path[2][0], "waypoint_4": path[3][0]}
    ok = (result.status is MotionStatus.CANCELLED and "waypoint 3 of 5" in result.message and len(sent) == 3
          and q0 <= path[2][0] + 1e-3)
    p.passed["M4"] = ok
    print(f"  M4 braked path: {result.status.value}, {len(sent)} moveJ sent, base at {q0:.4f} rad (waypoint 3 at "
          f"{path[2][0]:.4f}): {'PASS' if ok else 'FAIL'}")
    p.conn.clear_halt()


def m5(p: Probe, w: Watcher) -> None:
    """The toggle on tool DO0: a halt mid-approach, then the close; no edge of DO0 after the halt, brakes on and off."""
    jaw = p.cfg.gripper.jaw_io
    rows: dict[str, dict[str, Any]] = {}
    for brake in (True, False):
        p.healthy()
        jaws = JawIOGripper(p.arm, actuation="single_toggle", close_output_pin=jaw.close_output_pin,
                            io_port=DigitalIOPort.TOOL, pulse_s=jaw.pulse_s, close_settle_s=0.0,
                            min_width_mm=5.0, max_width_mm=49.99, ask=lambda _question: "open",
                            sleep=time.sleep)
        jaws.connect()
        p.conn.brake_on_halt = brake
        approach = list(HOME)
        approach[0] += 0.8
        approach[1] += 0.2
        t_start = time.perf_counter()
        thread, answer, started = p.moving(lambda: p.conn.moveJ(approach, 0.5, 1.0))
        time.sleep(0.9)
        t_req = time.perf_counter()
        p.arm.halt("probe M5: halt mid-approach")
        thread.join(10.0)
        try:
            jaws.set_closed(True)  # the close the pick sends at the part
            closed = "sent"
        except ArmHalted as exc:
            closed = f"refused: {exc}"
        except Exception as exc:  # noqa: BLE001
            closed = f"raised {type(exc).__name__}: {exc}"
        gate = _controller_refusal(p.arm)
        time.sleep(0.5)
        rows["brakes_on" if brake else "brakes_off"] = {
            "move_answer": answer[:1], "close": closed, "hand_gate": gate,
            "do0_edges_before_halt": w.do0_edges(t_start, t_req), "do0_edges_after_halt": w.do0_edges(t_req),
        }
        try:
            jaws.disconnect()
        except Exception:  # noqa: BLE001
            pass
    p.healthy()
    p.results["M5"] = rows
    ok = all(row["do0_edges_after_halt"] == 0 and row["close"].startswith("refused") and "halted" in row["hand_gate"]
             for row in rows.values())
    p.passed["M5"] = ok
    print(f"  M5 toggle: DO0 edges after the halt {[r['do0_edges_after_halt'] for r in rows.values()]}, close "
          f"{[r['close'][:7] for r in rows.values()]}: {'PASS' if ok else 'FAIL'}")


def m6(p: Probe, w: Watcher) -> None:
    """Brakes on: a line braked with stopL stops on the line; halted past the register's 1 s, so it showed the line."""
    p.healthy()
    p.conn.brake_on_halt = True
    start = list(p.conn.get_tcp_pose())
    target = list(start)
    target[1] += 0.3
    thread, answer, started = p.moving(lambda: p.conn.moveL(target, 0.15, 1.2))  # 0.3 m at 0.15 m/s: 2 s
    time.sleep(1.2)  # past _SEEN_WITHIN_S: a line the register never showed was stopped and refused at 1 s
    t_req = time.perf_counter()
    p.conn.request_halt("probe M6: halt on a line")
    thread.join(10.0)
    end = p.conn.last_move_end
    time.sleep(0.3)
    tcp = w.latest()["tcp"]
    span = [b - a for a, b in zip(start[:3], target[:3])]
    length = math.sqrt(sum(v * v for v in span))
    along = sum((t - a) * v for t, a, v in zip(tcp[:3], start[:3], span)) / (length * length)
    off_mm = 1000.0 * math.dist(tcp[:3], [a + along * v for a, v in zip(start[:3], span)])
    decel = w.deceleration_after(t_req)
    p.results["M6"] = {"answer": answer[:1], "end": end.value, "halted_after_s": round(t_req - started, 3),
                       "stopL_sent": len(p.sent("stopL", t_req)), "off_line_mm": off_mm, "along": along,
                       "deceleration_ms": _ms(None if decel is None else decel - t_req)}
    ok = (answer[:1] == [False] and end is MoveEnd.BRAKED and off_mm <= 1.0 and len(p.sent("stopL", t_req)) == 1
          and 0.0 < along < 1.0)
    p.passed["M6"] = ok
    print(f"  M6 line: {end.value}, stopped {off_mm:.3f} mm off the line at {along:.2f} of it: "
          f"{'PASS' if ok else 'FAIL'}")
    p.conn.clear_halt()


def m7(p: Probe, w: Watcher) -> None:
    """Brakes on: a protective stop during a watched move answers False, never True."""
    p.healthy()
    p.conn.brake_on_halt = True
    target = list(HOME)
    target[0] += 1.2
    thread, answer, started = p.moving(lambda: p.conn.moveJ(target, 0.5, 1.0))
    time.sleep(0.8)
    t_req = time.perf_counter()
    # The controller's own protective stop, from this thread while the watched move runs: a CB3 has it since 3.8, and
    # its IK refuses the on-axis poses probe_protective_stop.py drives through (see SINGULAR_Z_MM there). The moving
    # thread only reads while it watches, so this command is the only one on the interface.
    p.conn._ctrl.triggerProtectiveStop()
    thread.join(10.0)
    end, cause = p.conn.last_move_end, p.conn.last_stop_cause
    status = dash("safetystatus", "robotmode")
    still = w.still_after(t_req)
    seen = w.protective_after(t_req)
    p.results["M7"] = {"answer": answer[:1], "end": end.value, "cause": cause, "protective_seen": seen,
                       "controller": status, "still_ms": _ms(None if still is None else still - t_req),
                       "moveJ_sent_after": len(p.sent("moveJ", t_req)), "stops_sent": len(p.sent("stopJ", t_req))}
    # A protective stop, really: named as the cause, or read on the second connection. A program that only ended
    # (cause "program", no flag ever read) is the end of ur_rtde's script, not the stop this check is about.
    ok = (answer[:1] == [False] and end is MoveEnd.STOPPED and len(p.sent("moveJ", t_req)) == 0
          and (cause == "protective" or seen))
    p.passed["M7"] = ok
    print(f"  M7 protective stop mid-move: answered {answer[:1]}, {end.value} ({cause or 'no cause'}), protective "
          f"stop {'seen' if seen else 'NEVER seen'}, controller {status}: {'PASS' if ok else 'FAIL'}")
    _release_and_reconnect(p)


def _release_and_reconnect(p: Probe) -> None:
    """Release a protective stop through the dashboard (URSim only) and wait until a fresh control script runs.

    A CB3 releases a protective stop only some seconds after it, and the first control script uploaded right after the
    release can time out: measured 2026-10-01, "Failed to start control script, before timeout of 5 seconds".
    """
    time.sleep(6.0)
    dash("close safety popup", "unlock protective stop")
    for _ in range(10):
        if "NORMAL" in dash("safetystatus")[0].upper():
            break
        time.sleep(1.0)
    _reconnect_patiently(p)


def _reconnect_patiently(p: Probe) -> None:
    """Reconnect after the controller's program ended, retried while it comes back; the last failure raises."""
    for attempt in range(3):
        try:
            p.reconnect()
            return
        except Exception as exc:  # noqa: BLE001 (retried: the controller is still coming back)
            print(f"  reconnect attempt {attempt + 1} after the stop failed: {exc}")
            time.sleep(3.0)
    p.reconnect()


def m8(p: Probe, w: Watcher) -> None:
    """Alternative B: the program stopped from the dashboard during a synchronous moveJ; DO0 before and after."""
    p.healthy()
    target = list(HOME)
    target[0] += 1.2
    level_before = p.conn.get_digital_out(0, "tool")
    thread, answer, started = p.moving(lambda: p.conn.moveJ(target, 0.5, 1.0))
    time.sleep(0.8)
    t_req = time.perf_counter()
    dash("stop")
    thread.join(10.0)
    decel = w.deceleration_after(t_req)
    still = w.still_after(t_req + 0.02)
    level_after = bool((w.latest()["do"] >> _TOOL_DO0_BIT) & 1)
    p.results["M8"] = {"move_answer": answer[:1], "deceleration_ms": _ms(None if decel is None else decel - t_req),
                       "still_ms": _ms(None if still is None else still - t_req), "do0_before": level_before,
                       "do0_after": level_after, "do0_edges": w.do0_edges(t_req)}
    p.passed["M8"] = True  # a measurement for the alternative, not a check
    print(f"  M8 program stop: deceleration after {p.results['M8']['deceleration_ms']} ms, still after "
          f"{p.results['M8']['still_ms']} ms, DO0 {level_before} -> {level_after}")
    _reconnect_patiently(p)


def m9(p: Probe, w: Watcher) -> None:
    """The latch outlives a disconnect and a connect of the same arm."""
    p.healthy()
    p.arm.halt("probe M9: halt, then disconnect and connect")
    p.reconnect()
    refused_move = p.conn.moveJ(HOME, 0.5, 1.0)
    try:
        p.arm.set_digital_output(0, True, port=DigitalIOPort.TOOL)
        output = "switched"
    except ArmHalted:
        output = "refused"
    still_halted = p.arm.halt_state() is not None
    p.arm.clear_halt()
    moved = p.conn.moveJ(HOME, 0.5, 1.0)
    p.results["M9"] = {"halted_after_reconnect": still_halted, "move": refused_move, "output": output,
                       "move_after_clear": moved}
    ok = still_halted and refused_move is False and output == "refused" and moved is True
    p.passed["M9"] = ok
    print(f"  M9 latch across a reconnect: halted={still_halted}, move refused={refused_move is False}, DO "
          f"{output}, after clear moved={moved}: {'PASS' if ok else 'FAIL'}")


_CHECKS: dict[str, Callable[..., None]] = {
    "M0": m0, "M1": m1_m2, "M2": m1_m2, "M3": m3, "M3B": m3b, "M4": m4, "M5": m5, "M6": m6, "M7": m7, "M8": m8,
    "M9": m9,
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/ursim/probe_halt.py", description=__doc__.splitlines()[0])
    ap.add_argument("--only", default="M0,M1,M2,M3,M3b,M4,M5,M6,M9",
                    help="the checks to run, comma separated (default every one but M7 and M8)")
    ap.add_argument("--moves", type=int, default=200, help="M3's number of watched moves")
    ap.add_argument("--lines", type=int, default=50, help="M3b's number of watched lines")
    ap.add_argument("--json", type=Path, default=None, help="write the measurements here as one JSON record")
    args = ap.parse_args(argv)

    try:
        robotmode, model = dash("robotmode", "get robot model")
    except OSError as exc:
        print(f"dashboard UNREACHABLE at {HOST}:{_DASHBOARD_PORT}: {exc}")
        return _BAD_REQUEST
    model_key = {"UR3": "ur3", "UR5": "ur5", "UR10": "ur10"}.get(model.strip().upper(), "")
    if not model_key:
        print(f"the controller says {model!r}: this probe knows the CB3 models UR3, UR5 and UR10")
        return _BAD_REQUEST
    if "RUNNING" not in robotmode.upper():
        print(f"powering on and releasing the brakes ({robotmode})")
        dash("power on")
        time.sleep(5.0)
        dash("brake release")
        time.sleep(5.0)

    probe = Probe(model_key)
    print(f"controller {model.strip()} at {HOST}, profile ursim,{model_key},haltprobe from {probe.tree}")
    print(f"  the layer's brake_on_halt reached the arm: brakes_in_motion() is {probe.results['profile']['brakes_in_motion']}"
          f": {'PASS' if probe.passed['profile'] else 'FAIL'}")
    try:
        probe.connect()
    except Exception as exc:  # noqa: BLE001
        print(f"arm.connect() failed: {type(exc).__name__}: {exc}")
        return _BAD_REQUEST
    wanted = [name.strip().upper() for name in args.only.split(",") if name.strip()]
    ran: set[Callable[..., None]] = set()
    try:
        with Watcher() as watcher:
            for name in wanted:
                check = _CHECKS.get(name)
                if check is None:
                    print(f"  no check named {name}")
                    continue
                if check in ran:
                    continue
                ran.add(check)
                print(f"== {name}: {(check.__doc__ or '').strip().splitlines()[0]}")
                try:
                    if check is m3:
                        check(probe, watcher, args.moves)
                    elif check is m3b:
                        check(probe, watcher, args.lines)
                    else:
                        check(probe, watcher)
                except Exception as exc:  # noqa: BLE001 (one check that breaks is a failed check, not a lost run)
                    probe.passed[name] = False
                    probe.results[f"{name}_error"] = f"{type(exc).__name__}: {exc}"
                    print(f"  {name} raised {type(exc).__name__}: {exc}")
                    try:
                        probe.reconnect()
                    except Exception as again:  # noqa: BLE001
                        print(f"  reconnect failed too: {again}")
                        break
            try:
                probe.healthy()
            except Exception:  # noqa: BLE001 (leave the cell as found, best effort)
                pass
    finally:
        try:
            probe.arm.disconnect()
        except Exception:  # noqa: BLE001
            pass
        shutil.rmtree(probe.tree.parent, ignore_errors=True)
    record = {"controller": model.strip(), "results": probe.results, "passed": probe.passed}
    if args.json is not None:
        args.json.write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    failed = [name for name, ok in probe.passed.items() if not ok]
    print("PROBE_HALT_DONE " + ("all passed" if not failed else f"FAILED: {', '.join(failed)}"))
    return _OK if not failed else _FAILED


if __name__ == "__main__":
    raise SystemExit(main())
