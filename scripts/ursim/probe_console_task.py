"""The console's task, its stops and its way back, against a real UR controller (URSim CB3): build plan 6.5, item 2.

The API tests drive the console on doubles. This drives the same console, in this process, against UR controller
software: the routes through ``TestClient``, the real ``RunRegistry`` on its threads, the real ``URRobotArm`` on the
loopback with cuRobo planning every motion, and the owner's hand, a toggle on tool DO0 (``jaw_io`` ``single_toggle``,
every change of the output moves the jaws). Perception is the rehearsal scene, because URSim has no camera: one part,
its stand-in camera hung over the base where the arm reaches it. A second RTDE connection reads every edge of tool DO0,
whoever switched it.

  C1  three task cycles, a pose place and the return home each: every one ``finished``, one part placed, and exactly
      2 DO0 edges per part (the close at the grasp, the open at the drop).
  C2  halt now during the approach: the run ends ``halted`` where the arm stands, its recovery record stands, no DO0
      edge after the halt, and every moving route answers 409 ``halted``, never ``controller_stopped``. Then the cell
      is said clear, a new task is refused ``restart_required``, the Restart ``jaws_not_confirmed`` until the jaws
      question answered "open", and the Restart's first motion is the planned move home; it ends the record and places
      the part with 2 DO0 edges.
  C3  halt now during the carry: the run ends ``halted`` holding the part, and the Restart is refused
      ``part_still_held`` until the jaws question answered "open now", which is exactly 1 DO0 edge (the count says
      closed, so the check asks no "where"). The Restart then counts down 3 s, hands off, before its first motion, the
      move home.
  C4  a protective stop mid-move (the controller's own, ``triggerProtectiveStop``): the run ends on a problem where
      the arm stands, and "the cell is clear" and the Restart are refused ``controller_stopped`` while the controller
      stands stopped: the console never clears a protective stop. The probe clears it through the dashboard, as the
      other URSim probes do (a real cell clears it at the pendant), connects again where the controller ended the
      control script, and the Restart goes home first.
  C5  Disconnect during the planner's start: it waits for the start to end (the start moves nothing), then the cell
      comes down, and nothing moved and no output changed meanwhile.

Not measured here, and said so when it runs:

* the jaws question in the browser (``api.jaws``), Disconnect during a jaws question, and during a teach, and a pose
  taught by URSim freedrive: where ``GET /v1/cell/jaws`` or the teach routes answer 501 ``not_built_yet``, the
  console has no browser seam yet. The probe then answers the toggle's questions itself, through the hand's structured
  seam (``answer_questions_with``), as a person at a simulator would, and stamps what the browser's answer would stamp;
* a halt with the hand inside a bin, the path home planned against the current frame: the bed has no camera world.
  Every verb of a run here declines it, as the URSim bed probe does (``without_camera_world``), so a path home is
  judged against no frame at all.

It refuses to run, before any task or Home, unless the console says ``controller_is_simulator is True``
(``GET /v1/cell/status``): its controller's serial is URSim's placeholder and its address the loopback. A profile whose
``ur.ip`` is not the loopback is refused before anything connects. Everything it writes is a scratch layer in a copy of
``config/`` in a temporary directory (``ursim,<model>,ursim_curobo,consoleprobe``: ``brake_on_halt``, the toggle, a
declared ``payload.length_mm``, a home inside the workspace box, two taught poses); no shipped profile changes.

Start the CB3 controller as ``scripts/ursim/README.md`` says (UR10, the safety dialogs confirmed), with the cuRobo
environment the planner's sidecar runs in on this machine, then:

    python scripts/ursim/probe_console_task.py                      # C5, then C1-C4
    python scripts/ursim/probe_console_task.py --only C2,C3 --json console_task.json

It also refuses to start where the arm stands outside the window the console's joint guard keeps (half a turn either
side of the layer's home, less the guard's margin): every path from there is refused, so C1 would end
``failed_in_a_row`` and its stop record would refuse C2-C4. ``probe_halt.py`` leaves wrist 2 at -90 deg, which is such
a place: move URSim to the layer's home, [-90, -90, -100, -80, 90, 0] deg, first.

Exit codes: 0 every check that ran passed, 1 a check failed, 2 no controller, not the loopback, no simulator proven,
an arm that stands outside the guard's window, or a cell that did not come up (the build, the connect, the planner).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Running this as a file puts only `scripts/ursim/` on sys.path, so `import api...` fails without the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402

HOST = "127.0.0.1"
_DASHBOARD_PORT = 29999
_OK, _FAILED, _BAD_REQUEST = 0, 1, 2
#: The tool output the owner's toggle hand is on, as RTDE numbers digital outputs: tool 0 is bit 16.
_TOOL_DO0_BIT = 16
#: Where the rehearsal scene's stand-in camera hangs over the base, looking straight down: its one part then stands at
#: about (-130, -650) mm with its top 150 mm up, within the UR10's workspace box (its floor is 100 mm).
CAMERA_MM = (-130.0, -650.0, 900.0)
CHECKS = ("C5", "C1", "C2", "C3", "C4")
#: How long a run, a planner start and a reconnect may take on the bed before a check gives up on it.
_RUN_S, _PLANNER_S = 240.0, 300.0

_LAYER = """\
# probe_console_task.py's own layer, written into a copy of config/ in a temporary directory; never into config/.
robot:
  ur:
    model: "{model}"
    brake_on_halt: true
  # Tool down, the grasp centre (-164, -679, 392) mm on a UR10: inside the workspace box, which the shipped default home,
  # the arm standing straight up, is not.
  home_joint_positions_deg: [-90.0, -90.0, -100.0, -80.0, 90.0, 0.0]
  gripper:
    vendor: "jaw_io"
    model: robotiq_hande
    coupling_plates:
      - name: hande_adapter
        thickness_mm: 20.0
    tool_frame:
      source: "willy"
      offset_mm: [0.0, 0.0, 155.75]
      rotation_quat_xyzw: [0.0, 0.0, 0.0, 1.0]
    jaw_io:
      actuation: single_toggle
      close_output_pin: 0
      io_port: tool
  safety:
    planning_world:
      payload:
        length_mm: 40.0
  named_poses:
    drop:
      joints_deg: [-70.0, -95.0, -110.0, -65.0, 90.0, 0.0]
      label: "Ablage"
      screen: clear
    park:
      joints_deg: [-100.0, -95.0, -110.0, -65.0, 90.0, 0.0]
      label: "Parkposition"
      screen: clear
  default_place_pose: drop
"""


# ------------------------------------------------------------------------------------------------------------------
# What is measured with
# ------------------------------------------------------------------------------------------------------------------


def dash(*commands: str) -> list[str]:
    """Speak the dashboard protocol over a raw socket: the probe's own door, beside the console's connection."""
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


#: One reading of the second connection: time, the digital output bits, the joints, the fastest joint's speed (rad/s),
#: and whether the controller stands protective-stopped.
_Sample = tuple[float, int, list[float], float, bool]


class Watcher:
    """A second RTDE receive connection: every digital output bit, the joints, their speed and the protective-stop
    flag, with ``perf_counter`` times, so an edge of tool DO0 is seen whoever switched it, and a halt is pressed while
    the arm really moves."""

    def __init__(self) -> None:
        import rtde_receive

        self._recv = rtde_receive.RTDEReceiveInterface(HOST)
        self._samples: list[_Sample] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="probe-watcher", daemon=True)
        self._thread.start()
        time.sleep(0.2)

    def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            stamp = self._recv.getTimestamp()
            if stamp != last:
                last = stamp
                sample = (time.perf_counter(), int(self._recv.getActualDigitalOutputBits()),
                          list(self._recv.getActualQ()), max(abs(v) for v in self._recv.getActualQd()),
                          bool(self._recv.isProtectiveStopped()))
                with self._lock:
                    self._samples.append(sample)
            time.sleep(0.002)

    def moving_after(self, t0: float, *, timeout: float = 60.0, rad_s: float = 0.05) -> float | None:
        """When the arm first moved faster than ``rad_s`` after ``t0``, waited for; ``None`` where it did not."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            moving = [s[0] for s in self._since(t0, None) if s[3] > rad_s]
            if moving:
                return moving[0]
            time.sleep(0.005)
        return None

    def joints_now(self) -> list[float] | None:
        """The joints the controller reported last (radians), or ``None`` before its first sample."""
        with self._lock:
            return list(self._samples[-1][2]) if self._samples else None

    def close(self) -> None:
        self._stop.set()
        self._thread.join(2.0)
        try:
            self._recv.disconnect()
        except Exception:  # noqa: BLE001 (a probe's teardown)
            pass

    def _since(self, t0: float, t1: float | None) -> list[_Sample]:
        with self._lock:
            return [s for s in self._samples if s[0] >= t0 and (t1 is None or s[0] <= t1)]

    def do0_edges(self, t0: float, t1: float | None = None) -> int:
        bits = [(s[1] >> _TOOL_DO0_BIT) & 1 for s in self._since(t0 - 0.05, t1)]
        return sum(1 for a, b in zip(bits, bits[1:]) if a != b)

    def travel_rad(self, t0: float, t1: float | None = None) -> float:
        """The largest joint excursion between ``t0`` and ``t1``: zero where nothing moved."""
        rows = [s[2] for s in self._since(t0, t1)]
        if len(rows) < 2:
            return 0.0
        q = np.asarray(rows)
        return float(np.max(np.max(q, axis=0) - np.min(q, axis=0)))

    def protective_now(self) -> bool:
        with self._lock:
            return bool(self._samples and self._samples[-1][4])


class ProbeAnswers:
    """The toggle's questions where the console has no browser seam yet: answered from a script, as a person at a
    simulator would, every question written down. Out of answers is no answer (``EOFError``), never "open"."""

    def __init__(self) -> None:
        self.script: list[str] = []
        self.asked: list[tuple[str, str]] = []

    def __call__(self, asking: Any) -> str:
        self.asked.append((str(asking.stage), str(asking.reason)))
        if not self.script:
            raise EOFError("the probe scripted no answer for this question")
        return self.script.pop(0)


@dataclass
class Check:
    name: str
    passed: bool | None = None
    results: dict[str, Any] = field(default_factory=dict)
    skipped: str = ""


# ------------------------------------------------------------------------------------------------------------------
# The console on the bed
# ------------------------------------------------------------------------------------------------------------------


class Bed:
    """The console in this process on a scratch tree, its cell built from the profile and the rehearsal scene."""

    def __init__(self, model: str) -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.cell import Console, set_console
        from api.events import EventHub
        from src.config.loader import load_robot_config

        self.tmp = Path(tempfile.mkdtemp(prefix="willy_probe_console_"))
        self.tree = self.tmp / "config"
        shutil.copytree(_REPO_ROOT / "config", self.tree)
        (self.tree / "robot" / "robot.consoleprobe.yaml").write_text(_LAYER.format(model=model), encoding="utf-8")
        self.chain = f"ursim,{model},ursim_curobo,consoleprobe"
        cfg = load_robot_config(self.tree, profile=self.chain)
        if cfg.ur.ip not in ("127.0.0.1", "localhost", "::1"):
            raise SystemExit(f"profile ur.ip is {cfg.ur.ip!r}: this probe moves the arm and runs on the loopback")
        self.hub = EventHub()
        self.console = Console(root=self.tree, profile=self.chain, hub=self.hub)
        self.console.record_log_path = self.tmp / "grasp_records.jsonl"
        self._previous = set_console(self.console)
        self.client = TestClient(create_app())
        self.answers = ProbeAnswers()
        self.browser_jaws = False

    def close(self) -> None:
        from api.cell import set_console

        try:
            self.client.post("/v1/cell/disconnect")
        except Exception:  # noqa: BLE001 (a probe's teardown)
            pass
        try:
            self.console.session.release()
        finally:
            set_console(self._previous)
            shutil.rmtree(self.tmp, ignore_errors=True)

    # ---- the cell ----------------------------------------------------------------------------------------------

    def build(self) -> None:
        """What ``Console.build`` does, with the rehearsal scene's perception beside the configured UR arm and hand
        (``build_rehearsal_cell`` would swap the arm for the dummy)."""
        from api import jaws as browser_jaws
        from src.geometry import Frame, Transform
        from src.robot.execution.autonomous_grasp import AutonomousGraspService
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components
        from src.robot.grasping.motion.frame_resolver import StaticCameraToBaseResolver

        _app, robot = self.console.resolved()
        calculator, perception, _desk, _rigs, _calculators = build_rehearsal_components(robot, data_dir=self.tree)
        resolver = StaticCameraToBaseResolver(transform=Transform(
            translation_mm=np.array(CAMERA_MM), quaternion_xyzw=np.array([1.0, 0.0, 0.0, 0.0]),
            from_frame=Frame.CAMERA, to_frame=Frame.BASE))
        service = AutonomousGraspService.from_robot_config(robot, calculator=calculator, perception=perception,
                                                           frame_resolver=resolver)
        self.console.session.may_adopt()
        self.console.session.release()
        self.console.session.adopt(service)
        browser_jaws.install(self.console)
        self.browser_jaws = self.client.get("/v1/cell/jaws").status_code != 501
        if not self.browser_jaws:
            self.console.session.gripper.answer_questions_with(self.answers)

    def connect(self, *, jaws: str = "open") -> Any:
        """Connect through the routes; the toggle's question at connect is answered ``jaws``."""
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        if self.browser_jaws:
            return self._answering_in_the_browser(lambda: self.client.post("/v1/cell/connect", json={"token": token}),
                                                  [jaws])
        self.answers.script = [jaws]
        return self.client.post("/v1/cell/connect", json={"token": token})

    def simulator(self) -> tuple[bool | None, Any, Any]:
        status = self.client.get("/v1/cell/status").json()
        return status.get("controller_is_simulator"), status.get("controller_serial"), status.get("controller_host")

    def planner(self, timeout: float = _PLANNER_S) -> tuple[str, str]:
        """The planner light once it is no longer starting: ``(code, message)``."""
        deadline = time.monotonic() + timeout
        light: dict[str, Any] = {}
        while time.monotonic() < deadline:
            lights = {item["id"]: item for item in self.client.get("/v1/cell/readiness").json()["lights"]}
            light = lights["planner"]
            if light["code"] not in ("starting", "off"):
                break
            if lights["robot"]["code"] in ("not_built", "not_connected"):
                return str(lights["robot"]["code"]), str(lights["robot"]["message"])  # nothing will start it
            time.sleep(1.0)
        return str(light.get("code")), str(light.get("message"))

    # ---- the jaws ----------------------------------------------------------------------------------------------

    def jaws_check(self, answers: list[str], reason: str) -> str:
        """Ask the toggle where its jaws stand, as Restart's way back does; ``""`` where they stand open by the end.

        Through ``POST /v1/cell/jaws/check`` where the browser seam exists; else the hand is asked through the probe's
        seam, with the controller read right before the one change, and what the browser's answer stamps is stamped:
        the jaws confirmed open, opened just now after "open now", and the planner told the hand is empty.
        """
        if self.browser_jaws:
            answered = self._answering_in_the_browser(lambda: self.client.post("/v1/cell/jaws/check"), answers)
            if answered is None:
                return "the jaws check did not answer within 120 s"
            return "" if answered.status_code == 200 else str(answered.text)
        from api.readiness import controller_gate
        from src.robot.core.arm_capabilities import CarriesPayload

        arm, hand = self.console.session.arm, self.console.session.gripper
        self.answers.script = list(answers)
        refused = hand.confirm_where_the_jaws_stand(reason, before_change=lambda: controller_gate(arm))
        if refused:
            return str(refused)
        self.console.stamp_jaws_open(opened="open_now" in answers)
        if isinstance(arm, CarriesPayload):
            arm.detach_payload()
        return ""

    def _answering_in_the_browser(self, request: Callable[[], Any], answers: list[str]) -> Any:
        """Send ``request`` (it blocks while the question waits) and answer each question it asks, in order."""
        answered: list[Any] = []
        thread = threading.Thread(target=lambda: answered.append(request()), daemon=True)
        thread.start()
        pending = list(answers)
        seen: set[str] = set()
        deadline = time.monotonic() + 120.0
        while thread.is_alive() and time.monotonic() < deadline:
            question = self.client.get("/v1/cell/jaws").json().get("question")
            if question and question["question_id"] not in seen and pending:
                seen.add(question["question_id"])
                self.client.post("/v1/cell/jaws/answer",
                                 json={"question_id": question["question_id"], "choice": pending.pop(0)})
            time.sleep(0.05)
        thread.join(5.0)
        return answered[0] if answered else None

    # ---- runs --------------------------------------------------------------------------------------------------

    def task(self, *, scope: str = "once") -> Any:
        return self.client.post("/v1/task", json={"object": "", "place": {"kind": "pose", "pose": None},
                                                  "scope": scope})

    def wait_event(self, stream: str, event_type: str, timeout: float = _RUN_S) -> Any:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for event in self.hub.since(stream, 0)[0]:
                if event.type == event_type:
                    return event
            time.sleep(0.01)
        raise RuntimeError(f"no {event_type} on {stream} within {timeout:.0f} s")

    def finished(self, run_id: str, timeout: float = _RUN_S) -> dict[str, Any]:
        self.wait_event(run_id, "run_finished", timeout)
        return dict(self.client.get(f"/v1/runs/{run_id}").json())

    def types(self, run_id: str) -> list[str]:
        return [event.type for event in self.hub.since(run_id, 0)[0]]

    def recovery(self) -> Any:
        return self.client.get("/v1/cell").json().get("recovery")

    def code(self, answered: Any) -> str:
        """A refusal's code, or the status where the answer is no refusal."""
        try:
            return str(answered.json().get("code") or answered.status_code)
        except ValueError:
            return str(answered.status_code)

    @contextmanager
    def declining_the_camera_world(self) -> Iterator[None]:
        """Every task and Home run declines the camera world on its own thread: the bed has no camera, so a verb that
        asks for one is refused before planning. The decline follows code, not threads, so it is entered in the run."""
        from unittest.mock import patch

        import api.task_run as task_run
        from src.robot.core.camera_world import without_camera_world

        def declining(driver: Callable[[Any, Any], Any]) -> Callable[[Any, Any], Any]:
            def run(console: Any, run: Any) -> Any:
                with without_camera_world(console.session.arm, "the URSim bed has no camera"):
                    return driver(console, run)
            return run

        # Looked up by the routes at call time (``from api.task_run import drive_task`` inside each), so patched here.
        with patch.object(task_run, "drive_task", declining(task_run.drive_task)), \
                patch.object(task_run, "drive_home", declining(task_run.drive_home)):
            yield


# ------------------------------------------------------------------------------------------------------------------
# The checks
# ------------------------------------------------------------------------------------------------------------------


def c1(bed: Bed, w: Watcher, check: Check, cycles: int = 3) -> None:
    """Three task cycles, each a part placed at the default pose and the return home, with 2 DO0 edges per part."""
    rows = []
    for _ in range(cycles):
        t0 = time.perf_counter()
        started = bed.task()
        if started.status_code != 202:
            rows.append({"refused": bed.code(started), "message": started.text[:300]})
            break
        record = bed.finished(started.json()["id"])
        rows.append({"stop_code": record["stop_code"], "parts_placed": record["parts_placed"],
                     "do0_edges": w.do0_edges(t0), "seconds": round(time.perf_counter() - t0, 1),
                     "error": record["error"][:300]})
    check.results["cycles"] = rows
    check.passed = len(rows) == cycles and all(
        row.get("stop_code") == "finished" and row.get("parts_placed") == 1 and row.get("do0_edges") == 2
        for row in rows)
    print(f"  C1 {len(rows)} cycle(s): " + "; ".join(
        f"{row.get('stop_code', row.get('refused'))}, {row.get('do0_edges')} DO0 edge(s)" for row in rows)
        + f": {'PASS' if check.passed else 'FAIL'}")


def _moving_after(bed: Bed, w: Watcher, run_id: str, event_type: str, delay_s: float) -> float | None:
    """Wait for ``event_type`` of ``run_id``, then for the arm to move (cuRobo plans first), then ``delay_s`` more:
    the moment to act on a motion in flight. ``None`` where the arm did not move."""
    bed.wait_event(run_id, event_type)
    moving = w.moving_after(time.perf_counter())
    if moving is not None:
        time.sleep(delay_s)
    return moving


def _halt_on(bed: Bed, w: Watcher, check: Check, event_type: str, delay_s: float) -> tuple[str, float] | None:
    """Start a task and press halt now ``delay_s`` into the first motion after ``event_type``; the stopped run's id and
    the press time."""
    started = bed.task()
    if started.status_code != 202:
        check.results["start"] = {"refused": bed.code(started), "message": started.text[:300]}
        return None
    run_id = started.json()["id"]
    check.results["arm_moved_before_the_halt"] = _moving_after(bed, w, run_id, event_type, delay_s) is not None
    pressed = time.perf_counter()
    brake = bed.client.post("/v1/cell/brake")
    stopped = bed.finished(run_id)
    check.results["brake"] = brake.json() if brake.status_code == 200 else {"status": brake.status_code}
    # The latch's own record of the move in flight: braked (brake_on_halt), ran out, or nothing was moving.
    check.results["latch"] = bed.client.get("/v1/cell").json().get("halted")
    check.results["stopped"] = {key: stopped[key] for key in ("stop_code", "stop_class", "holding", "error")}
    check.results["moving_routes_while_halted"] = {
        "task": bed.code(bed.task()),
        "home": bed.code(bed.client.post("/v1/cell/home", json={"to": "home"})),
        "restart": bed.code(bed.client.post("/v1/task/restart", json={"run_id": run_id})),
    }
    check.results["record"] = bed.recovery()
    check.results["do0_edges_after_the_halt"] = w.do0_edges(pressed + 0.05)
    return run_id, pressed


def _restart(bed: Bed, w: Watcher, check: Check, run_id: str) -> dict[str, Any]:
    """The Restart of ``run_id``, to its end: its record, its first events, and its DO0 edges."""
    t0 = time.perf_counter()
    answered = bed.client.post("/v1/task/restart", json={"run_id": run_id})
    if answered.status_code != 202:
        return {"refused": bed.code(answered), "message": answered.text[:300]}
    restart = answered.json()["id"]
    record = bed.finished(restart)
    types = bed.types(restart)
    moves = [t for t in types if t.startswith(("task.return", "task.part", "pick.", "task.place"))]
    ended = [e.data for e in bed.hub.since("cell", 0)[0] if e.type == "cell.recovery_ended"]
    return {"stop_code": record["stop_code"], "parts_placed": record["parts_placed"], "restart_of": record.get(
        "restart_of"), "first_moves": moves[:2], "countdown": types.count("run_countdown"),
        "recovery_ended": ended[-1] if ended else None, "do0_edges": w.do0_edges(t0)}


def c2(bed: Bed, w: Watcher, check: Check) -> None:
    """Halt now during the approach, then the way back: acknowledge, the jaws question, Restart home first."""
    stopped = _halt_on(bed, w, check, "pick.executing", 0.3)
    if stopped is None:
        check.passed = False
        print(f"  C2 the task did not start: {check.results['start']}: FAIL")
        return
    run_id, _pressed = stopped
    acknowledged = bed.client.post("/v1/cell/acknowledge", json={"cell_clear": True})
    check.results["acknowledge"] = bed.code(acknowledged)
    check.results["after_acknowledge"] = {
        "task": bed.code(bed.task()),
        "restart": bed.code(bed.client.post("/v1/task/restart", json={"run_id": run_id})),
    }
    check.results["jaws"] = bed.jaws_check(["open"], "the probe's check after halt now during the approach")
    check.results["restart"] = restart = _restart(bed, w, check, run_id)
    results = check.results
    check.passed = (
        results["arm_moved_before_the_halt"] is True and results["brake"].get("in_motion") is True
        and results["stopped"]["stop_code"] == "halted" and results["record"] is not None
        and results["do0_edges_after_the_halt"] == 0
        and set(results["moving_routes_while_halted"].values()) == {"halted"}
        and results["acknowledge"] == "200"
        and results["after_acknowledge"] == {"task": "restart_required", "restart": "jaws_not_confirmed"}
        and results["jaws"] == ""
        and restart.get("stop_code") == "finished" and restart.get("restart_of") == run_id
        and restart.get("first_moves", [None])[0] == "task.return_started"
        and (restart.get("recovery_ended") or {}).get("by") == "restart" and restart.get("do0_edges") == 2)
    print(f"  C2 halt in the approach (in motion {results['brake'].get('in_motion')}, latch "
          f"{(results['latch'] or {}).get('brake')}): {results['stopped']['stop_code']}, "
          f"{results['do0_edges_after_the_halt']} DO0 edge(s) after it, routes {results['moving_routes_while_halted']}, "
          f"then {results['after_acknowledge']}; Restart {restart.get('stop_code', restart.get('refused'))}, first "
          f"{restart.get('first_moves')}: {'PASS' if check.passed else 'FAIL'}")


def c3(bed: Bed, w: Watcher, check: Check) -> None:
    """Halt now during the carry: the part held, the jaws question opens them with one edge, the Restart counts down."""
    stopped = _halt_on(bed, w, check, "task.place_started", 0.3)
    if stopped is None:
        check.passed = False
        print(f"  C3 the task did not start: {check.results['start']}: FAIL")
        return
    run_id, _pressed = stopped
    check.results["acknowledge"] = bed.code(bed.client.post("/v1/cell/acknowledge", json={"cell_clear": True}))
    check.results["restart_while_held"] = bed.code(bed.client.post("/v1/task/restart", json={"run_id": run_id}))
    t0 = time.perf_counter()
    # The count says CLOSED and DO0 still stands where the close left it: the check asks only "open now" (never
    # "where", whose "open" would contradict the count), one change of DO0 while a person holds the part.
    check.results["jaws"] = bed.jaws_check(["open_now"], "the probe's check after halt now in the carry")
    check.results["do0_edges_in_the_question"] = w.do0_edges(t0)
    check.results["restart"] = restart = _restart(bed, w, check, run_id)
    results = check.results
    check.passed = (
        results["arm_moved_before_the_halt"] is True and results["brake"].get("in_motion") is True
        and results["stopped"]["stop_code"] == "halted" and results["stopped"]["holding"] is True
        and results["do0_edges_after_the_halt"] == 0
        and set(results["moving_routes_while_halted"].values()) == {"halted"}
        and results["acknowledge"] == "200" and results["restart_while_held"] == "part_still_held"
        and results["jaws"] == "" and results["do0_edges_in_the_question"] == 1
        and restart.get("stop_code") == "finished" and restart.get("countdown") == 3
        and restart.get("first_moves", [None])[0] == "task.return_started")
    print(f"  C3 halt in the carry (in motion {results['brake'].get('in_motion')}, latch "
          f"{(results['latch'] or {}).get('brake')}): {results['stopped']['stop_code']} holding="
          f"{results['stopped']['holding']}, Restart while held {results['restart_while_held']}, "
          f"{results['do0_edges_in_the_question']} DO0 edge(s) in the jaws question; Restart "
          f"{restart.get('stop_code', restart.get('refused'))} after {restart.get('countdown')} countdown step(s), "
          f"first {restart.get('first_moves')}: {'PASS' if check.passed else 'FAIL'}")


def c4(bed: Bed, w: Watcher, check: Check) -> None:
    """A protective stop mid-move: a problem stop where the arm stands, and the console never clears it."""
    started = bed.task()
    if started.status_code != 202:
        check.passed = False
        check.results["start"] = {"refused": bed.code(started), "message": started.text[:300]}
        print(f"  C4 the task did not start: {check.results['start']}: FAIL")
        return
    run_id = started.json()["id"]
    check.results["arm_moved_before_the_stop"] = _moving_after(bed, w, run_id, "pick.executing", 0.3) is not None
    # The controller's own protective stop, through the console's connection from this thread, as probe_halt.py M7
    # does on its own: a CB3's IK refuses the on-axis pose probe_protective_stop.py drives through.
    bed.console.session.arm.connection._ctrl.triggerProtectiveStop()
    stopped = bed.finished(run_id)
    check.results["stopped"] = {key: stopped[key] for key in ("stop_code", "stop_class", "holding", "error")}
    check.results["record"] = bed.recovery()
    check.results["while_stopped"] = {
        "acknowledge": bed.code(bed.client.post("/v1/cell/acknowledge", json={"cell_clear": True})),
        "restart": bed.code(bed.client.post("/v1/task/restart", json={"run_id": run_id})),
    }
    check.results["still_protective_before_the_probe_clears_it"] = (
        w.protective_now() or "PROTECTIVE" in dash("safetystatus")[0].upper())
    # A CB3 releases a protective stop only some seconds after it (probe_halt.py, _release_and_reconnect).
    time.sleep(6.0)
    dash("close safety popup", "unlock protective stop")
    for _ in range(15):
        if "NORMAL" in dash("safetystatus")[0].upper():
            break
        time.sleep(1.0)
    running = bed.console.session.arm.connection.is_program_running()
    check.results["control_script_running_after_the_release"] = running
    if not running:
        # The controller ended the control script with the stop: the way back is Disconnect and Connect, which the
        # recovery record outlives. A CB3 can time out the first control script uploaded right after a release
        # (probe_halt.py, _release_and_reconnect), so the connect is tried again while the controller comes back.
        check.results["disconnect"] = bed.code(bed.client.post("/v1/cell/disconnect"))
        tries = []
        for _ in range(4):
            time.sleep(3.0)
            tries.append(bed.code(bed.connect(jaws="open")))
            if tries[-1] == "200":
                break
        check.results["connect"] = tries
        check.results["planner"] = bed.planner()[0]
    check.results["acknowledge"] = bed.code(bed.client.post("/v1/cell/acknowledge", json={"cell_clear": True}))
    check.results["jaws"] = bed.jaws_check(["open"], "the probe's check after the protective stop")
    check.results["restart"] = restart = _restart(bed, w, check, run_id)
    results = check.results
    check.passed = (
        results["arm_moved_before_the_stop"] is True
        and results["stopped"]["stop_class"] == "problem" and results["record"] is not None
        and results["while_stopped"] == {"acknowledge": "controller_stopped", "restart": "controller_stopped"}
        and results["still_protective_before_the_probe_clears_it"] is True
        and results["acknowledge"] == "200" and results["jaws"] == ""
        and restart.get("stop_code") == "finished" and restart.get("first_moves", [None])[0] == "task.return_started")
    print(f"  C4 protective stop: {results['stopped']['stop_code']} ({results['stopped']['stop_class']}), while stopped "
          f"{results['while_stopped']}, control script after the release {'running' if running else 'ended'}; "
          f"Restart {restart.get('stop_code', restart.get('refused'))}: {'PASS' if check.passed else 'FAIL'}")


def c5(bed: Bed, w: Watcher, check: Check) -> None:
    """Disconnect while the planner starts: the start is waited out, then the cell comes down; nothing moves."""
    t0 = time.perf_counter()
    light = next(item for item in bed.client.get("/v1/cell/readiness").json()["lights"] if item["id"] == "planner")
    check.results["planner_at_the_disconnect"] = light["code"]
    answered = bed.client.post("/v1/cell/disconnect")
    t1 = time.perf_counter()
    planners = [run for run in bed.console.registry.recent(10) if str(run.kind) == "planner"]
    check.results.update({
        "disconnect": bed.code(answered), "state": answered.json().get("state"),
        "disconnect_s": round(t1 - t0, 1),
        "planner_runs": [(run.id, str(run.state), str(run.stop_code)) for run in planners],
        "joint_travel_rad": round(w.travel_rad(t0, t1), 5), "do0_edges": w.do0_edges(t0, t1),
    })
    check.passed = (light["code"] == "starting" and answered.status_code == 200
                    and check.results["state"] == "built" and bool(planners) and str(planners[0].state) != "running"
                    and check.results["joint_travel_rad"] < 1e-3 and check.results["do0_edges"] == 0)
    print(f"  C5 Disconnect during the planner start ({light['code']}): answered {check.results['disconnect']} after "
          f"{check.results['disconnect_s']} s, planner {check.results['planner_runs'][:1]}, joints moved "
          f"{check.results['joint_travel_rad']} rad, {check.results['do0_edges']} DO0 edge(s): "
          f"{'PASS' if check.passed else 'FAIL'}")


# ------------------------------------------------------------------------------------------------------------------
# The run
# ------------------------------------------------------------------------------------------------------------------


# ------------------------------------------------------------------------------------------------------------------
# Where the arm starts
# ------------------------------------------------------------------------------------------------------------------


def outside_the_window(q_rad: Sequence[float], lower_deg: Sequence[float], upper_deg: Sequence[float],
                       margin_deg: float) -> str:
    """The first axis of ``q_rad`` (radians) outside ``[lower + margin, upper - margin]`` (degrees), in a clause, or
    ``""`` where every axis is inside: the joint guard's own test (``JointLimitGuard.evaluate``), read before anything
    moves."""
    for axis, (q, low, high) in enumerate(zip(q_rad, lower_deg, upper_deg)):
        deg, lower, upper = math.degrees(float(q)), float(low) + margin_deg, float(high) - margin_deg
        if not lower <= deg <= upper:
            return f"axis {axis} stands at {deg:.1f} deg, outside the joint guard's [{lower:.1f}, {upper:.1f}] deg"
    return ""


def where_it_starts(bed: Bed, q_rad: Sequence[float] | None) -> str:
    """Why the checks cannot start where the arm stands, or ``""``: the window the console's joint guard keeps every
    path inside (``safety.joint_limits``, half a turn either side of the home the built arm states), read with the
    guard's own limits and margin. A probe started outside it would have every path refused, so C1 would end
    ``failed_in_a_row`` and its stop record would refuse C2-C4."""
    from src.robot.safety.joint_limits import JointLimitGuard, stated_home_rad

    if q_rad is None:
        return "the second RTDE connection has read no joints yet"
    _app, robot = bed.console.resolved()
    if not robot.safety.joint_limits.enforce:
        return ""
    arm = bed.console.session.arm
    guard = JointLimitGuard(robot.safety.joint_limits)
    limits = guard.limits_for_arm(arm)
    if limits is None:
        return ""
    said = outside_the_window(q_rad, limits[0], limits[1], guard.margin_deg)
    if not said:
        return ""
    home = stated_home_rad(arm)
    home_deg = ", ".join(f"{math.degrees(v):.0f}" for v in home) if home is not None else "the layer's home"
    return (f"{said}, so every path from there is refused: move URSim to the layer's home [{home_deg}] deg first (the "
            "pendant's Move tab, or a moveJ through ur_rtde), then run the probe again")


def _not_measured(bed: Bed) -> list[str]:
    said = ["a halt with the hand in a bin, the path home planned against the current frame: the bed has no camera "
            "world, and every verb here declines one"]
    if not bed.browser_jaws:
        said.append("the jaws question in the browser, and Disconnect during one: GET /v1/cell/jaws answers 501 "
                    "not_built_yet, so the probe answered the toggle's questions through the hand's seam")
    if bed.client.get("/v1/teach/payload").status_code == 501:
        said.append("teaching a pose by freedrive, and Disconnect during a teach: the teach routes answer 501 "
                    "not_built_yet")
    return said


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python scripts/ursim/probe_console_task.py",
                                     description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="ur10", help="the URSim model's layer (default ur10: the owner's arm)")
    parser.add_argument("--only", default="", help=f"comma-separated checks of {', '.join(CHECKS)} (default: all)")
    parser.add_argument("--cycles", type=int, default=3, help="C1's task cycles (default 3)")
    parser.add_argument("--json", type=Path, default=None, help="write every measurement to this file as well")
    parser.add_argument("--curobo-python", default="", help="the cuRobo environment's python (WILLY_CUROBO_PYTHON)")
    parser.add_argument("--coal-prefix", default="", help="the Coal environment (WILLY_COAL_PREFIX)")
    args = parser.parse_args(argv)
    wanted = [name.strip().upper() for name in args.only.split(",") if name.strip()] or list(CHECKS)
    unknown = sorted(set(wanted) - set(CHECKS))
    if unknown:
        parser.error(f"no check {', '.join(unknown)}; there are {', '.join(CHECKS)}")
    if args.curobo_python:
        os.environ["WILLY_CUROBO_PYTHON"] = args.curobo_python
    if args.coal_prefix:
        os.environ["WILLY_COAL_PREFIX"] = args.coal_prefix
    try:
        serial = dash("get serial number", "PolyscopeVersion")
    except OSError as exc:
        print(f"no controller answers the dashboard on {HOST}:{_DASHBOARD_PORT} ({exc}): is URSim up?")
        return _BAD_REQUEST
    print(f"controller: {serial}")

    checks = {name: Check(name) for name in CHECKS if name in wanted}
    report: dict[str, Any] = {"controller": serial, "model": args.model}
    bed = Bed(args.model)
    watcher = Watcher()
    try:
        try:
            bed.build()
        except Exception as exc:  # noqa: BLE001 (a cell that does not build is the probe's answer, said in full)
            print(f"the cell did not build: {type(exc).__name__}: {exc}")
            return _BAD_REQUEST
        connected = bed.connect()
        if connected.status_code != 200:
            print(f"the cell did not connect: {connected.status_code} {connected.text[:400]}")
            return _BAD_REQUEST
        proven, serial_seen, host = bed.simulator()
        report["simulator"] = {"controller_is_simulator": proven, "serial": serial_seen, "host": host}
        if proven is not True:
            print(f"REFUSED: the console does not prove this controller a simulator (controller_is_simulator={proven}, "
                  f"serial {serial_seen!r}, host {host!r}); this probe halts, restarts and stops the arm, and runs "
                  "only where the console says it is URSim.")
            return _BAD_REQUEST
        print(f"the console proves a simulator: serial {serial_seen}, host {host}; jaws answered "
              f"{'in the browser' if bed.browser_jaws else 'through the hand seam (no browser seam yet)'}")
        start = where_it_starts(bed, watcher.joints_now())
        report["start"] = {"joints_rad": watcher.joints_now(), "refused": start}
        if start:
            print(f"REFUSED: {start}.")
            return _BAD_REQUEST
        if "C5" in checks:
            c5(bed, watcher, checks["C5"])
            connected = bed.connect()
            if connected.status_code != 200:
                print(f"the cell did not connect again after C5: {connected.status_code} {connected.text[:400]}")
                return _BAD_REQUEST
        code, message = bed.planner()
        report["planner"] = {"code": code, "message": message}
        if code != "ready":
            print(f"the planner did not come up ({code}): {message}")
            return _BAD_REQUEST
        with bed.declining_the_camera_world():
            for name, run in (("C1", lambda c: c1(bed, watcher, c, args.cycles)), ("C2", lambda c: c2(bed, watcher, c)),
                              ("C3", lambda c: c3(bed, watcher, c)), ("C4", lambda c: c4(bed, watcher, c))):
                if name not in checks:
                    continue
                try:
                    run(checks[name])
                except Exception as exc:  # noqa: BLE001 (one check that raised fails, and the others still run)
                    checks[name].passed = False
                    checks[name].results["raised"] = f"{type(exc).__name__}: {exc}"
                    print(f"  {name} raised {type(exc).__name__}: {exc}: FAIL")
        report["not_measured"] = _not_measured(bed)
    finally:
        watcher.close()
        bed.close()
        report["checks"] = {name: {"passed": c.passed, "skipped": c.skipped, **c.results} for name, c in checks.items()}
        if args.json is not None:
            args.json.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    for said in report.get("not_measured", []):
        print(f"  not measured: {said}")
    ran = [c for c in checks.values() if c.passed is not None]
    print(f"CONSOLE_TASK_DONE: {sum(bool(c.passed) for c in ran)}/{len(ran)} passed")
    return _OK if ran and all(c.passed for c in ran) else _FAILED


if __name__ == "__main__":
    raise SystemExit(main())
