"""Capture the console's event logs, the ones the frontend's run model replays (``frontend/src/test/fixtures/*.json``).

Each scenario is driven through the real console as a browser drives it: the app's routes through ``TestClient``, the
real ``RunRegistry`` on its threads, the library's ``run_task``, and every event exactly as the hub published it, in
arrival order across the run's streams and the cell's own. Nothing in a log is written by hand but its run ids, which
are fixed per scenario (``RUN_IDS``), and its clock: the events' ``ts`` count ticks of a fixed clock, every other
wall-clock instant (``started_at``, ``finished_at``, ``requested_at``, a record's ``at``, ...) reads the tick of the
first event at or after it (:func:`stable_clock`), and a part's seconds come from a scripted clock. So a regenerated log
is byte for byte the one before, except where the console's words changed. A jaws question's id is fixed per scenario
too (``QUESTION_IDS``). A fixture is

    {"about": "...", "scenario": "<file name>", "events": [<every envelope, every stream, in arrival order>],
     "runs": {"<run id>": <the run as GET /v1/runs/<id> answers once it ended>}}

which ``tests/test_api_contract.py`` checks against the contract, and ``frontend/src/model/runModel.test.ts`` replays.

Two cells, and each fixture says in its ``about`` which one told it:

* **console_dummy** (the shipped tree, its arm and hand the dummies, two taught poses): the console end to end at a
  desk, the rehearsal scene, the real pick loop on the dummy arm. Its logs are what a desk run says, word for word:
  ``console_dummy_task_once``, ``console_dummy_until_empty``, ``console_dummy_home_counts_down``,
  ``console_dummy_stop_after_part`` (plan 6.2 scenario 4) and ``console_dummy_halted_backen_leer`` (scenario 7).
* **the scripted cell** of ``tests/_console_task_fakes.py``, where a story needs what console_dummy cannot show: the
  owner's toggle hand on tool DO0 (no sensor), a part pushed free, a bin a wrist camera finds, a halt during the grasp.
  Its arm is the task tests' double (``tests/_task_fakes.py``), its hand the real ``JawIOGripper``, its picks scripted
  as the pick loop plays them (their ``pick.*`` events are the double's, in the real loop's words and fields), and the
  seconds a part takes come from a scripted clock. The three logs that replaced the contract's hand-written ones are
  its: ``task_two_parts_nothing_left``, ``task_camera_target_lost`` and ``task_halted_then_restart``; and so is
  ``task_drop_area_holds_the_rest`` (scenario 3), which needs an arm that knows where a pose puts the tool, as the
  dummy's fk does not.

Run it from the repository root after any change to what the console says, and commit the fixtures it writes:

    python scripts/console/capture_event_log.py                 # every scenario this server can show
    python scripts/console/capture_event_log.py --only task_halted_then_restart
    python scripts/console/capture_event_log.py --out /tmp/logs  # somewhere else

The jaws question goes round through the browser seam (``api.jaws``): ``jaws_question_round_trip`` connects the scripted
cell's toggle and answers in the browser, 'closed' then 'open now' (on a server whose seam still answers
``not_built_yet`` it is skipped, and its fixture kept as it is). ``teach_one_pose`` is not captured here: teaching needs
a hand-guided arm double; its fixture stays as written until a scenario for it lands.
"""

from __future__ import annotations

import argparse
import bisect
import importlib
import json
import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

# Running this as a file puts only `scripts/console/` on sys.path, so `import api...` would fail without the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from api.events import EventEnvelope, EventHub  # noqa: E402

_FIXTURES = _REPO_ROOT / "frontend" / "src" / "test" / "fixtures"


class RecordingHub(EventHub):
    """The console's hub, keeping every envelope in the order it was published, whatever its stream.

    ``after`` is called with each envelope once it is kept, outside the hub's locks: a scenario presses a button there
    when the console says a step began (the button's own events are published from inside it).
    """

    def __init__(self) -> None:
        super().__init__()
        self.log: list[EventEnvelope] = []
        self._log_lock = threading.Lock()
        self.after: Callable[[EventEnvelope], None] | None = None

    def publish(self, run_id: str, event_type: str, /, **keywords: Any) -> EventEnvelope:
        with self._log_lock:
            envelope = super().publish(run_id, event_type, **keywords)
            self.log.append(envelope)
        if self.after is not None:
            self.after(envelope)
        return envelope


#: The fields that carry a wall-clock instant (Unix seconds) beside an envelope's own ``ts``: a run's record, an event's
#: data. Only values past 1e9 are instants; a sighting's ``seen_at`` on a scripted camera counts frames.
_INSTANTS = frozenset({"started_at", "finished_at", "requested_at", "at", "cleared_at", "seen_at", "captured_at"})
#: Where a captured log's fixed clock starts (2026-10-01 12:00:00 UTC), and the step from one event to the next.
_EPOCH_S = 1_790_856_000.0
_TICK_S = 0.05


def _is_instant(key: str, value: Any) -> bool:
    return key in _INSTANTS and isinstance(value, (int, float)) and not isinstance(value, bool) and value > 1e9


def _on_the_fixed_clock(value: Any, clock: Callable[[float], float]) -> Any:
    """``value`` with every instant (:data:`_INSTANTS`) read off ``clock``."""
    if isinstance(value, list):
        return [_on_the_fixed_clock(item, clock) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: clock(float(item)) if _is_instant(key, item) else _on_the_fixed_clock(item, clock)
            for key, item in value.items()}


def _anchors(fixture: dict[str, Any], fixed: list[float]) -> dict[float, float]:
    """The fixed time of each run's own instants, by the event that says them: a run starts at its ``run_started``,
    ends at its record's ``cell.recovery`` or else its ``run_finished``, and a halt is requested at its
    ``run_halt_requested``. Two readings of the wall clock in one tick would otherwise decide which event an instant
    falls on."""
    anchored: dict[float, float] = {}
    runs = fixture["runs"]
    for index, event in enumerate(fixture["events"]):
        data = event.get("data") if isinstance(event.get("data"), dict) else {}
        if event["type"] == "run_started":
            started = runs.get(event["run_id"], {}).get("started_at")
            if _is_instant("started_at", started):
                anchored.setdefault(float(started), fixed[index])
        elif event["type"] == "cell.recovery":
            ended = runs.get(str(data.get("run_id")), {}).get("finished_at")
            if _is_instant("finished_at", ended):
                anchored.setdefault(float(ended), fixed[index])
        elif event["type"] == "run_finished" and _is_instant("finished_at", data.get("finished_at")):
            anchored.setdefault(float(data["finished_at"]), fixed[index])
        elif event["type"] == "run_halt_requested" and _is_instant("requested_at", data.get("requested_at")):
            anchored.setdefault(float(data["requested_at"]), fixed[index])
    return anchored


def stable_clock(fixture: dict[str, Any]) -> dict[str, Any]:
    """``fixture`` on a fixed clock, the same at every capture: the n-th event's ``ts`` is ``n`` ticks past the epoch; a
    run's start and end and a halt's request read the tick of the event that says them (:func:`_anchors`), and any
    other instant the tick of the first event at or after it. A deadline (``expires_at``) keeps its distance from its
    event, to the tenth."""
    events = fixture["events"]
    raw = [float(event["ts"]) for event in events]
    fixed = [round(_EPOCH_S + index * _TICK_S, 2) for index in range(len(events))]
    after_the_last = round(_EPOCH_S + len(events) * _TICK_S, 2)
    anchored = _anchors(fixture, fixed)

    def clock(instant: float) -> float:
        if instant in anchored:
            return anchored[instant]
        at = bisect.bisect_left(raw, instant)
        return fixed[at] if at < len(fixed) else after_the_last

    stable = dict(_on_the_fixed_clock(fixture, clock))
    for index, event in enumerate(stable["events"]):
        event["ts"] = fixed[index]
        data = event.get("data")
        if isinstance(data, dict) and isinstance(data.get("expires_at"), (int, float)):
            data["expires_at"] = round(fixed[index] + float(data["expires_at"]) - raw[index], 1)
    return stable


class SteppingClock:
    """The task's clock in a console_dummy capture (``src.robot.execution.task.time``, which reads ``monotonic`` alone):
    each reading is ``step_s`` past the one before, so a part's ``duration_s`` is the same at every capture."""

    def __init__(self, step_s: float = 12.5) -> None:
        self.step_s = step_s
        self.now = 1000.0
        self._lock = threading.Lock()

    def monotonic(self) -> float:
        with self._lock:
            self.now += self.step_s
            return self.now


def _fakes() -> Any:
    """The scripted cell's doubles, read from the checkout's tests at run time (they are no part of the shipped
    console)."""
    return importlib.import_module("tests._console_task_fakes")


class Scene:
    """One scenario's console: a scratch tree, a fresh console with a recording hub installed as the app's, a client."""

    def __init__(self, *, gripper: str = "dummy") -> None:
        from fastapi.testclient import TestClient

        from api.app import create_app
        from api.cell import Console, set_console
        from src.config.loader import active_profile

        self._tmp = Path(tempfile.mkdtemp(prefix="willy-capture-"))
        tree = self._tmp / "config"
        profile = _fakes().dummy_tree(tree, gripper=gripper)
        self.hub = RecordingHub()
        self.console = Console(root=tree, profile=profile, hub=self.hub)
        self.console.record_log_path = self._tmp / "grasp_records.jsonl"
        self._previous_profile = active_profile()
        self._previous_console = set_console(self.console)
        self.client = TestClient(create_app())

    def close(self) -> None:
        from api.cell import set_console
        from src.config.loader import reload_config, set_active_profile

        try:
            self.console.session.release()
        finally:
            set_console(self._previous_console)
            set_active_profile(self._previous_profile)
            reload_config()
            shutil.rmtree(self._tmp, ignore_errors=True)

    # ---- driving -----------------------------------------------------------------------------------------------

    def ok(self, answered: Any, status: int = 200) -> Any:
        if answered.status_code != status:
            raise RuntimeError(f"{answered.request.method} {answered.request.url.path} answered "
                               f"{answered.status_code}: {answered.text}")
        return answered.json()

    def build_dummy(self) -> None:
        self.ok(self.client.post("/v1/cell/build", params={"rehearse": True}))
        token = self.ok(self.client.get("/v1/cell/connect-preview"))["token"]
        self.ok(self.client.post("/v1/cell/connect", json={"token": token}))

    def finished(self, run_id: str, timeout: float = 60.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if any(event.type == "run_finished" for event in self.hub.since(run_id, 0)[0]):
                return dict(self.ok(self.client.get(f"/v1/runs/{run_id}")))
            time.sleep(0.01)
        raise RuntimeError(f"run {run_id} did not finish within {timeout} s")

    def log(self, scenario: str, about: str, run_ids: list[str]) -> dict[str, Any]:
        """The fixture: every event of the runs' streams and of the cell's, in arrival order, and each run's record, on
        the fixed clock (:func:`stable_clock`)."""
        streams = set(run_ids) | {"cell"}
        events = [event.to_dict() for event in self.hub.log if event.run_id in streams]
        runs = {run_id: self.ok(self.client.get(f"/v1/runs/{run_id}")) for run_id in run_ids}
        return stable_clock({"about": about, "scenario": scenario, "events": events, "runs": runs})


@contextmanager
def _scene(**keywords: Any) -> Iterator[Scene]:
    scene = Scene(**keywords)
    try:
        yield scene
    finally:
        scene.close()


_SPOKEN_DE = {"source": "spoken", "language": "de", "parsed": True, "edited": []}
_TYPED_DE = {"source": "typed", "language": "de", "parsed": True, "edited": []}


# ---------------------------------------------------------------------------------------------------------------------
# The scenarios
# ---------------------------------------------------------------------------------------------------------------------


def task_two_parts_nothing_left() -> dict[str, Any]:
    """Until empty into 'Ablage links': two parts placed, the second pushed 30 mm free first, then two empty looks."""
    fakes = _fakes()
    from src.robot.execution import task as task_module

    with _scene() as scene:
        cell = fakes.ScriptedCell.build([
            fakes.Scripted("part", looks=("look_1", "look_2"), seconds=21.4),
            fakes.Scripted("push", looks=("look_1",), seconds=27.9),
            fakes.Scripted("empty", looks=("look_1",), seconds=4.0),
            fakes.Scripted("empty", looks=("look_1",), seconds=4.0),
        ])
        cell.install(scene.console)
        with patch.object(task_module, "time", cell.clock):
            run = scene.ok(scene.client.post("/v1/task", json={
                "object": "green cube", "object_said": "alle grünen Würfel",
                "place": {"kind": "pose", "pose": "drop_left"}, "scope": "until_empty",
                "options": {"push_mm": 30},
                "command": {"text": "Räum alle grünen Würfel auf die Ablage links", **_SPOKEN_DE},
            }), 202)
            ended = scene.finished(run["id"])
        if (ended["stop_code"], ended["parts_placed"]) != ("nothing_left", 2):
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("task_two_parts_nothing_left", (
            "Captured by scripts/console/capture_event_log.py on the scripted cell of tests/_console_task_fakes.py (the "
            "owner's toggle hand on tool DO0, the task tests' arm double, scripted picks and a scripted clock) through the "
            "real "
            "console: until empty into the taught pose 'Ablage links', two parts placed, the second pushed 30 mm free "
            "and looked at again, then two empty looks and the return home."), [run["id"]])


def task_drop_area_holds_the_rest() -> dict[str, Any]:
    """Plan 6.2 scenario 3: until empty into 'Ablage links'; once the part is placed, two looks see only parts standing
    in the circle the task keeps out about its drop, so nothing is left to pick: nothing_left and the return."""
    fakes = _fakes()
    task_fakes = importlib.import_module("tests._task_fakes")
    from src.robot.execution import task as task_module

    drop = task_fakes.PLACE_TCP.position_mm
    at_the_drop = (float(drop[0]) + 20.0, float(drop[1]) + 20.0)
    with _scene() as scene:
        cell = fakes.ScriptedCell.build([
            fakes.Scripted("part", looks=("look_1",), seconds=19.5),
            fakes.Scripted("kept_out", xy=at_the_drop, looks=("look_1",), seconds=4.0),
            fakes.Scripted("kept_out", xy=at_the_drop, looks=("look_1",), seconds=4.0),
        ])
        cell.install(scene.console)
        with patch.object(task_module, "time", cell.clock):
            run = scene.ok(scene.client.post("/v1/task", json={
                "object": "green cube", "object_said": "die grünen Würfel",
                "place": {"kind": "pose", "pose": "drop_left"}, "scope": "until_empty",
                "command": {"text": "Leg die grünen Würfel auf die Ablage links", **_SPOKEN_DE},
            }), 202)
            ended = scene.finished(run["id"])
        if (ended["stop_code"], ended["parts_placed"]) != ("nothing_left", 1):
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("task_drop_area_holds_the_rest", (
            "Captured by scripts/console/capture_event_log.py on the scripted cell of tests/_console_task_fakes.py (its "
            "arm knows where a pose puts the tool) through the real console: until empty into the taught pose 'Ablage "
            "links', one part placed, then two looks that see only parts standing in the 150 mm the task keeps out about "
            "its drop: nothing is left to pick, so the task returns home and ends nothing_left, never failed_in_a_row."),
            [run["id"]])


def task_camera_target_lost() -> dict[str, Any]:
    """A wrist camera finds the blue bin, the part is picked, the bin has moved 180 mm by the drop: put back, ask."""
    fakes = _fakes()
    task_fakes = importlib.import_module("tests._task_fakes")
    import numpy as np

    from src.robot.perception.locator import LocatedObject

    bin_at = {"centre": (120.0, -610.0)}

    def the_bin(_tcp: Any) -> Any:
        return (task_fakes.bin_object(task_fakes.bin_points(bin_at["centre"], (300.0, 400.0), 95.0)),)

    def the_cubes(_tcp: Any) -> Any:
        cubes = []
        for x, y in ((450.0, -150.0), (480.0, -80.0), (420.0, -260.0)):
            points = np.array([[x + dx, y + dy, 40.0] for dx in (-10.0, 0.0, 10.0) for dy in (-10.0, 0.0, 10.0)])
            cubes.append(LocatedObject(label="red cube", score=0.9, box_px=None, mask=np.ones((4, 4), dtype=bool),
                                       points_base_mm=points, centre_mm=(x, y, 40.0)))
        return tuple(cubes)

    def the_bin_moves() -> None:
        bin_at["centre"] = (300.0, -610.0)

    with _scene() as scene:
        cell = fakes.ScriptedCell.build(
            [fakes.Scripted("part", xy=(450.0, -150.0), looks=("look_1", "look_2"), seconds=31.0,
                            then=the_bin_moves)],
            wrist=True, looks=(fakes.LOOK_1, fakes.LOOK_2))
        cell.install(scene.console)
        locator = fakes.SightedLocator({"blue bin": the_bin, "red cube": the_cubes}, arm=cell.arm, wrist=True)
        with patch("src.robot.execution.place_target.locators_for_service", return_value=[locator]), \
             patch("src.robot.execution.task.locators_for_service", return_value=[locator]), \
             patch("src.robot.execution.task.time", cell.clock):
            run = scene.ok(scene.client.post("/v1/task", json={
                "object": "red cube", "object_said": "den roten Würfel",
                "place": {"kind": "camera", "phrase": "blue bin", "said": "in die blaue Kiste"}, "scope": "once",
                "command": {"text": "Leg den roten Würfel in die blaue Kiste", **_TYPED_DE},
            }), 202)
            ended = scene.finished(run["id"])
        if ended["stop_code"] != "target_lost":
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("task_camera_target_lost", (
            "Captured by scripts/console/capture_event_log.py on the scripted wrist cell of "
            "tests/_console_task_fakes.py through the real console: the survey finds the blue bin at the first look, "
            "the part is picked, the bin has moved 180 mm by the drop, more than it is followed, so the part is put "
            "back where it was gripped, the arm returns, and the task asks."), [run["id"]])


def task_halted_then_restart() -> dict[str, Any]:
    """Halt now during the grasp's line in; the cell said clear, the jaws answered open; Restart, home first."""
    fakes = _fakes()
    from src.robot.execution import task as task_module

    with _scene() as scene:
        cell = fakes.ScriptedCell.build([fakes.Scripted("part", seconds=18.0), fakes.Scripted("part", seconds=24.6)])
        cell.install(scene.console)
        pressed: list[int] = []

        def brake_during_the_line_in(kind: str, index: int) -> None:
            if kind == "move" and index == 1 and not pressed:
                pressed.append(scene.client.post("/v1/cell/brake").status_code)

        cell.arm.on_motion = brake_during_the_line_in
        with patch.object(task_module, "time", cell.clock):
            first = scene.ok(scene.client.post("/v1/task", json={
                "object": "green cube", "object_said": "den grünen Würfel",
                "place": {"kind": "pose", "pose": "drop_left"}, "scope": "once",
                "command": {"text": "Leg den grünen Würfel auf die Ablage links", **_SPOKEN_DE},
            }), 202)
            stopped = scene.finished(first["id"])
            if stopped["stop_code"] != "halted":
                raise RuntimeError(f"the story did not happen: {stopped}")
            cell.arm.on_motion = None
            scene.ok(scene.client.post("/v1/cell/acknowledge", json={"cell_clear": True}))
            # The browser's jaws question ends open (api.jaws stamps it so): a toggle's way back asks it first.
            scene.console.stamp_jaws_open(opened=False)
            restart = scene.ok(scene.client.post("/v1/task/restart", json={"run_id": first["id"]}), 202)
            ended = scene.finished(restart["id"])
        if ended["stop_code"] != "finished":
            raise RuntimeError(f"the restart did not finish: {ended}")
        return scene.log("task_halted_then_restart", (
            "Captured by scripts/console/capture_event_log.py on the scripted cell of tests/_console_task_fakes.py "
            "through the real console: halt now pressed during the grasp's line in (the move in flight runs to its "
            "end, nothing after it is sent), the stop card's record, 'the cell is clear', the jaws answered open, and "
            "the Restart, whose first motion is the planned move home, placing the part."),
            [first["id"], restart["id"]])


def console_dummy_task_once() -> dict[str, Any]:
    """console_dummy: one part of the rehearsal scene into the default place, and home."""
    with _scene() as scene, patch("src.robot.execution.task.time", SteppingClock()):
        scene.build_dummy()
        run = scene.ok(scene.client.post("/v1/task", json={
            "object": "", "place": {"kind": "pose", "pose": None}, "scope": "once",
            "command": {"text": "Nimm das Teil und leg es ab", **_SPOKEN_DE},
        }), 202)
        ended = scene.finished(run["id"])
        if ended["stop_code"] != "finished":
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("console_dummy_task_once", (
            "Captured by scripts/console/capture_event_log.py on console_dummy (the dummy arm and hand, the rehearsal "
            "scene, the real pick loop): one part into the default place pose, and the return home."), [run["id"]])


def console_dummy_until_empty() -> dict[str, Any]:
    """console_dummy: until empty on a rehearsal scene that empties after two frames."""
    fakes = _fakes()
    with _scene() as scene, patch("src.robot.execution.task.time", SteppingClock()):
        scene.build_dummy()
        fakes.EmptyingScene.install(scene.console.session.service, after=2)
        run = scene.ok(scene.client.post("/v1/task", json={
            "object": "", "place": {"kind": "pose", "pose": "drop_left"}, "scope": "until_empty",
        }), 202)
        ended = scene.finished(run["id"])
        if (ended["stop_code"], ended["parts_placed"]) != ("nothing_left", 2):
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("console_dummy_until_empty", (
            "Captured by scripts/console/capture_event_log.py on console_dummy, its rehearsal scene emptied after two "
            "frames: two parts placed at 'Ablage links', two empty looks, nothing left."), [run["id"]])


def console_dummy_stop_after_part() -> dict[str, Any]:
    """console_dummy, plan 6.2 scenario 4: until empty, and "stop after this part" while the first part is grasped."""
    with _scene() as scene, patch("src.robot.execution.task.time", SteppingClock()):
        scene.build_dummy()
        pressed: list[int] = []

        def stop_at_the_grasp(event: EventEnvelope) -> None:
            if event.type == "pick.executing" and not pressed:
                pressed.append(scene.client.post("/v1/task/stop", params={"run_id": event.run_id}).status_code)

        scene.hub.after = stop_at_the_grasp
        run = scene.ok(scene.client.post("/v1/task", json={
            "object": "", "place": {"kind": "pose", "pose": None}, "scope": "until_empty",
            "command": {"text": "Räum alles auf die Ablage", **_SPOKEN_DE},
        }), 202)
        ended = scene.finished(run["id"])
        scene.hub.after = None
        if pressed != [200] or (ended["stop_code"], ended["parts_placed"]) != ("stopped_after_part", 1):
            raise RuntimeError(f"the story did not happen: {pressed}, {ended}")
        return scene.log("console_dummy_stop_after_part", (
            "Captured by scripts/console/capture_event_log.py on console_dummy: until empty into the default place; "
            "'stop after this part' pressed while the first part is grasped, so that part is still placed, the arm "
            "returns home, and the task ends stopped_after_part."), [run["id"]])


def console_dummy_halted_backen_leer() -> dict[str, Any]:
    """console_dummy, plan 6.2 scenario 7: halt right after the pick, so the stopped run holds the part; the cell said
    clear, "Backen leer" (the dummy hand measures nothing), and the Restart, which counts down first."""
    with _scene() as scene, patch("src.robot.execution.task.time", SteppingClock()):
        scene.build_dummy()
        pressed: list[int] = []

        def halt_after_the_pick(event: EventEnvelope) -> None:
            if event.type == "pick_result" and not pressed:
                pressed.append(scene.client.post("/v1/cell/brake").status_code)

        scene.hub.after = halt_after_the_pick
        first = scene.ok(scene.client.post("/v1/task", json={
            "object": "", "place": {"kind": "pose", "pose": None}, "scope": "once",
            "command": {"text": "Nimm das Teil und leg es ab", **_SPOKEN_DE},
        }), 202)
        stopped = scene.finished(first["id"])
        scene.hub.after = None
        if pressed != [200] or (stopped["stop_code"], stopped["holding"]) != ("halted", True):
            raise RuntimeError(f"the story did not happen: {pressed}, {stopped}")
        scene.ok(scene.client.post("/v1/cell/acknowledge", json={"cell_clear": True}))
        scene.ok(scene.client.post("/v1/cell/acknowledge", json={"cell_clear": True, "jaws_empty": True}))
        restart = scene.ok(scene.client.post("/v1/task/restart", json={"run_id": first["id"]}), 202)
        ended = scene.finished(restart["id"])
        if ended["stop_code"] != "finished" or not restart["plan"]["countdown"]:
            raise RuntimeError(f"the restart did not count down and finish: {restart}, {ended}")
        return scene.log("console_dummy_halted_backen_leer", (
            "Captured by scripts/console/capture_event_log.py on console_dummy: halt now pressed right after the pick, "
            "so the stopped run holds the part where the arm stands; 'the cell is clear', then 'Backen leer' (the "
            "dummy hand measures nothing, so the person's word empties it, and the planner is told so); the Restart "
            "counts down three seconds, hands off, before its first motion, the planned move home, and places a part."),
            [first["id"], restart["id"]])


def console_dummy_home_counts_down() -> dict[str, Any]:
    """console_dummy: a pose was taught by hand, so Home counts down three seconds before its first motion."""
    with _scene() as scene:
        scene.build_dummy()
        scene.console.stamp_teach_ended()
        run = scene.ok(scene.client.post("/v1/cell/home", json={"to": "park"}), 202)
        ended = scene.finished(run["id"])
        if ended["stop_code"] != "finished":
            raise RuntimeError(f"the story did not happen: {ended}")
        return scene.log("console_dummy_home_counts_down", (
            "Captured by scripts/console/capture_event_log.py on console_dummy: after a pose was taught by hand, Home "
            "to 'Parkposition' counts down three seconds, hands off, before its first motion."), [run["id"]])


def jaws_question_round_trip() -> dict[str, Any] | None:
    """A toggle's connect asks where the jaws stand in the browser: closed, then open now. ``None`` where this server
    has no browser seam yet."""
    fakes = _fakes()
    from api import jaws as browser_jaws

    with _scene() as scene:
        if scene.client.get("/v1/cell/jaws").status_code == 501:
            return None
        cell = fakes.ScriptedCell.build()
        cell.install(scene.console, connect=False)
        browser_jaws.install(scene.console)
        token = scene.ok(scene.client.get("/v1/cell/connect-preview"))["token"]
        answered: dict[str, Any] = {}

        def connect() -> None:
            answered["connect"] = scene.client.post("/v1/cell/connect", json={"token": token})

        connecting = threading.Thread(target=connect, daemon=True)
        connecting.start()
        for choice in ("closed", "open_now"):
            question = _waiting_question(scene)
            scene.ok(scene.client.post("/v1/cell/jaws/answer",
                                       json={"question_id": question["question_id"], "choice": choice}))
        connecting.join(timeout=30)
        if answered.get("connect") is None or answered["connect"].status_code != 200:
            raise RuntimeError(f"the connect did not come up: {answered}")
        return scene.log("jaws_question_round_trip", (
            "Captured by scripts/console/capture_event_log.py on the scripted cell of tests/_console_task_fakes.py "
            "through the real console: a toggle's connect asks in the browser where the jaws stand, 'closed', then "
            "'open now', one change of tool DO0 while a person holds the part."), [])


def _waiting_question(scene: Scene, timeout: float = 30.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        question = scene.ok(scene.client.get("/v1/cell/jaws")).get("question")
        if question:
            return dict(question)
        time.sleep(0.02)
    raise RuntimeError("no jaws question came")


SCENARIOS: dict[str, Callable[[], dict[str, Any] | None]] = {
    "task_two_parts_nothing_left": task_two_parts_nothing_left,
    "task_drop_area_holds_the_rest": task_drop_area_holds_the_rest,
    "task_camera_target_lost": task_camera_target_lost,
    "task_halted_then_restart": task_halted_then_restart,
    "console_dummy_task_once": console_dummy_task_once,
    "console_dummy_until_empty": console_dummy_until_empty,
    "console_dummy_home_counts_down": console_dummy_home_counts_down,
    "console_dummy_stop_after_part": console_dummy_stop_after_part,
    "console_dummy_halted_backen_leer": console_dummy_halted_backen_leer,
    "jaws_question_round_trip": jaws_question_round_trip,
}

#: The ids each scenario's runs are given, in the order they start. The three task logs keep the ids of the contract's
#: hand-written logs they replaced, so the capture differed from those only where the console's words did; and since
#: every id is fixed, a regenerated log changes where the console's words change, never in its ids.
RUN_IDS: dict[str, tuple[str, ...]] = {
    "task_two_parts_nothing_left": ("run-1f3a9c0e21",),
    "task_drop_area_holds_the_rest": ("run-3b7e10c4d2",),
    "task_camera_target_lost": ("run-a84c2e7d05",),
    "task_halted_then_restart": ("run-7c1d00aa42", "run-9e2b44f0c3"),
    "console_dummy_task_once": ("run-5e1d0a7c01",),
    "console_dummy_until_empty": ("run-5e1d0a7c02",),
    "console_dummy_home_counts_down": ("run-5e1d0a7c03",),
    "console_dummy_stop_after_part": ("run-5e1d0a7c04",),
    "console_dummy_halted_backen_leer": ("run-5e1d0a7c05", "run-5e1d0a7c06"),
    "jaws_question_round_trip": (),
}


#: The ids each scenario's jaws questions are given, in the order they are asked (``api.jaws`` names a question
#: ``q-<hex>``): the contract's hand-written log's, which the frontend's cell model pins.
QUESTION_IDS: dict[str, tuple[str, ...]] = {
    "jaws_question_round_trip": ("5d1e0c", "8a2071"),
}


@contextmanager
def _named_runs(scenario: str) -> Iterator[None]:
    """Name the scenario's runs from ``RUN_IDS``, and its jaws questions from ``QUESTION_IDS``. A run or a question
    beyond its ids is a story that changed, and stops the capture."""
    import types

    ids = iter(RUN_IDS[scenario])
    questions = iter(QUESTION_IDS.get(scenario, ()))

    def next_id() -> str:
        try:
            return next(ids)
        except StopIteration:
            raise RuntimeError(f"{scenario} started more runs than RUN_IDS names for it") from None

    def next_question() -> Any:
        try:
            return types.SimpleNamespace(hex=next(questions))
        except StopIteration:
            raise RuntimeError(f"{scenario} asked more jaws questions than QUESTION_IDS names for it") from None

    with patch("api.runs.new_run_id", next_id), patch("api.jaws.uuid", types.SimpleNamespace(uuid4=next_question)):
        yield


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=_FIXTURES, help="where the fixtures are written")
    parser.add_argument("--only", default="", help="comma-separated scenario names (default: every one)")
    args = parser.parse_args(argv)
    wanted = [name.strip() for name in args.only.split(",") if name.strip()] or list(SCENARIOS)
    unknown = sorted(set(wanted) - set(SCENARIOS))
    if unknown:
        parser.error(f"no scenario {', '.join(unknown)}; there are {', '.join(SCENARIOS)}")
    args.out.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        with _named_runs(name):
            fixture = SCENARIOS[name]()
        if fixture is None:
            print(f"skipped  {name}: this server does not show it yet; its fixture is kept as it is")
            continue
        target = args.out / f"{name}.json"
        target.write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"captured {name}: {len(fixture['events'])} event(s), {len(fixture['runs'])} run(s) -> {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
