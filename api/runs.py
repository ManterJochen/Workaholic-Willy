"""A run: one operator instruction, executed on a thread that outlives the request that started it.

``pick()`` blocks for as long as the arm takes. A request handler that waited for it would hold the
HTTP connection open across a physical motion, and the browser, a proxy or a sleeping laptop would
time out somewhere in the middle. So starting a run returns immediately with an id, the work happens
on its own thread, and everything the operator sees arrives over the event stream.

The run finishes whether or not anyone is watching. A tab that closes is no reason to abandon a motion
halfway, so the run goes on as it would have, and the event history is what makes that survivable for the UI.

Stop is not kill. ``stop()`` sets a flag the pick loop reads between attempts, and a task's "stop after this
part" (:meth:`RunRegistry.stop_after_part`) lets the part in hand be placed and the arm return first. Neither
interrupts a motion in flight. "Halt now" (``POST /v1/cell/brake``, :meth:`RunRegistry.halt`) is the strongest
thing a socket asks: the run stops commanding, and where the arm latches, every next motion and output is refused
before anything is sent; where the arm also brakes (``robot.ur.brake_on_halt``) it brakes the move in flight under
control. None of them is an emergency stop: a request over a socket depends on latency, an open tab and an awake
laptop, and the red button on the wall does not.

A run ends early where a campaign does (``PickRun``, ``src/robot/execution/pick_run.py``): on a fault
of the cell, and on the three things a pick reports rather than raises, a controller that cannot move,
a hand that needs a person, and a recovery that stopped where the arm stands (``needs_person``: a push of
a failed part that stopped once something may have moved). Each ends it FAILED with the reason in
``error``, because the next pick would meet the same stopped arm or the same hand, or drive the arm back
to its look from where the push left it. And each pick looks from where a campaign's does: from home on
a wrist camera, from where the arm stands on a fixed one. A run is a campaign, too: its picks share the
push budgets and the parts ``next_target`` skips, and ``push_mm`` sets how far a push moves a part.

Two stops a campaign does not need, because a campaign runs in a terminal and a run does not. A run
never asks a person anything: on a hand that toggles with no sensor, a pick that starts on jaws the
program believes closed asks where they stand, and from a run's thread that question goes to the
server's terminal, which nobody watching the console sees and Stop cannot reach. A console run sets
none of its parts down, so the run ends FAILED before such a pick instead (``_why_no_pick_starts``).
And a Disconnect stops the run before the cell comes down (:meth:`RunRegistry.abandon`); the run keeps
the console's run lock until its thread ends, and Connect and Build refuse while it does.

A pick run is one kind of five (``api.codes.RunKind``). The others, a task, the Home button, teaching a pose and
starting the planner, start through :meth:`RunRegistry.start_kind`: the same one-run lock and the same thread, a
countdown before the first motion where a person's hands were last at the arm, and one way to end. The driver
answers a typed stop code; its class decides the run's state (``STATE_OF_CLASS``); and a moving kind that ends on a
problem code leaves the console's recovery record BEFORE it hands the lock back, so nothing new can start between
the stop and the record that gates it. The registry reads that record itself, under the same lock, before it starts a
moving run (:class:`RunRefused`): a route's gates read it a moment earlier, and a run that stopped on a problem in
between has written it by then, so no motion starts from where a problem stop left the arm but the two ways back,
Restart and Home, and those only once a person said the cell is clear.

A pick run keeps its own body (``_drive``, which programs and pinned tests call) and the same rules: the countdown
where one is due, a typed stop code beside its pinned sentences, the recovery record for a problem stop, and two more
of its own. A run started on a service whose earlier pick stopped where the arm stands, a person to decide, ends
``recovery_needs_person`` with nothing commanded, and the latch kept: its campaign, which would acknowledge it, is
never started before it is read. And a pick a halt cut short ends ``halted`` with its own sentence, whatever else it
reports (a stopped controller, a hand that refused on the latch, a recovery the latch ended): the halt's remedy is
neither the pendant nor the hand's.

The countdown (:meth:`RunRegistry._countdown`) is the owner's "hands off": after a teach, after "open now" (a person
held the part at the jaws) or after "Backen leer" (a person emptied the hand), the next moving run counts down three
seconds before its first motion, one ``run_countdown`` a second, and reads stop, halt and abandon every 100 ms;
stopped, the run ends ``cancelled`` with nothing moved. Only a run that moved the arm ends it: its driver stamps the
console once it knows the arm moved (a pick that gripped, a part placed, an arrival), so a run that counted down and
then moved nothing (a pose its screen refused, a move the planner refused before sending) leaves it due.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, TypeAlias

from api.codes import MOVING_KINDS, REFUSAL_STATUS, STOP_CLASS, RefusalCode, RunKind, StopClass, StopCode
from api.constants import API_LOG_DIR, RUNS_LOG_FILE
from api.events import EventHub, Severity
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.robot.grasping.loop.progress import PickProgress

__all__ = [
    "RETAINED_RUNS",
    "STATE_OF_CLASS",
    "Run",
    "RunConflict",
    "RunDriver",
    "RunKind",
    "RunRefused",
    "RunRegistry",
    "RunState",
    "kept_run",
    "motion_started",
    "run_from_kept",
]

#: How many runs the console remembers. Beyond this the oldest is forgotten whole (record, event
#: history, sequence counter, thread), so that ``/v1/runs/<id>`` and the event stream agree about
#: what exists rather than one of them answering an empty replay for a run the other still names.
#:
#: 200 because of what a run costs and what the console is for. Measured in this tree: 35 events
#: (one pick of five attempts) replay as about 14 kB of JSON, so 200 runs is roughly 2.8 MB; the
#: per-run ring caps a pathological run (2048 events, about 290 attempts in one run) at under 1 MB.
#: Before this, nothing was ever dropped: a console next to a robot is started on Monday and asked
#: about on Friday.
#:
#: ``/v1/history/runs.csv`` defaults to ``limit=500`` and therefore now returns at most 200 rows.
#: That export was always the perishable in-memory view; the durable one is the grasp record log
#: (``/v1/history/records.csv``), which this does not touch.
RETAINED_RUNS = 200

logger = create_logger("RunRegistry", RUNS_LOG_FILE, log_dir=API_LOG_DIR)


def new_run_id() -> str:
    """A new run's id: ``run-`` and ten hex digits, drawn at random. Every run of every kind is named by it, in the event
    stream, the history and the recovery record; ``scripts/console/capture_event_log.py`` names its runs in their place,
    so a regenerated event log keeps its ids."""
    return f"run-{uuid.uuid4().hex[:10]}"


class RunState(StrEnum):
    RUNNING = "running"
    #: All requested picks finished; some may have failed, so read the counts.
    FINISHED = "finished"
    #: An operator asked it to stop; it ended after the attempt that was running.
    CANCELLED = "cancelled"
    #: The run raised, or a pick reported a cell that needs a person (a controller that cannot move,
    #: a hand that needs one), or the next pick would have had to ask one (a toggle's jaws believed
    #: closed), or the cell was disconnected under it. The cell may need attention; ``Run.error`` says
    #: why.
    FAILED = "failed"


#: The state each class of stop code ends a run in (build plan 1.2): the done and ask classes finish, an operator's
#: stop cancels, and a teach, planner or problem stop fails.
STATE_OF_CLASS: Final[Mapping[StopClass, RunState]] = MappingProxyType({
    StopClass.DONE: RunState.FINISHED,
    StopClass.OPERATOR: RunState.CANCELLED,
    StopClass.ASK: RunState.FINISHED,
    StopClass.TEACH: RunState.FAILED,
    StopClass.PLANNER: RunState.FAILED,
    StopClass.PROBLEM: RunState.FAILED,
})

#: The body of a run of any kind but a pick: it runs on the run's own thread, drives the library, and answers why the
#: run ended as a stop code (a ``StopCode`` or its string). It writes ``run.error`` where the code is a problem, and it
#: never raises for a domain refusal; what it raises ends the run ``software_error``.
RunDriver: TypeAlias = Callable[[Any, "Run"], "StopCode | str"]

#: The fields :meth:`RunRegistry.start_kind` sets on a new run; everything else a run records, the run itself fills.
_START_FIELDS: Final[frozenset[str]] = frozenset({"prompt", "requested_picks", "plan", "restart_of", "push_mm"})

#: The hands-off countdown before a moving run's first motion, where one is due (build plan item 11): three steps of one
#: second, the run's stop, halt and abandon read every 100 ms.
COUNTDOWN_STEPS: Final[int] = 3
COUNTDOWN_STEP_S: Final[float] = 1.0
COUNTDOWN_POLL_S: Final[float] = 0.1

#: The timeline step each pick stage puts a run on (``RunOut.step``).
_STEP_OF_STAGE: Final[Mapping[str, str]] = MappingProxyType({
    "pick_started": "look", "attempt_started": "look", "perceived": "look", "ranked": "detect",
    "no_candidate": "detect", "executing": "grasp",
})


#: Maps a library stage to a severity and a sentence for the operator. The sentence lives here rather
#: than in the library because it is UI copy: the library's job is to say what happened, not how to
#: phrase it.
_STAGE_COPY: dict[str, tuple[Severity, str]] = {
    "pick_started": (Severity.INFO, "Starting a pick."),
    "attempt_started": (Severity.INFO, "Attempt {attempt_1} of {attempt_total}."),
    "perceived": (Severity.INFO, "Camera frame acquired: {segmentation_count} object(s) segmented."),
    "ranked": (Severity.INFO, "Ranked {candidate_count} grasp candidate(s), best score {score_2}."),
    "no_candidate": (Severity.WARN, "No usable grasp this attempt ({reasons_joined}); next: {action}."),
    "executing": (Severity.INFO, "Moving to the grasp at {position_short}."),
    "attempt_finished": (Severity.INFO, "Attempt ended: {action}."),
    "pick_finished": (Severity.INFO, "Pick finished: {outcome}."),
    "cancelled": (Severity.WARN, "Stopped by the operator before attempt {attempt_1}."),
}

#: The camera-world uses a sentence speaks: a planned motion with no camera world, and a caller's
#: decline. A dummy or ik cell stamps ``unplanned`` on every motion, and a sentence repeating that on
#: every event would bury the line that matters; the payload still carries it.
_SPOKEN_CAMERA_WORLDS = frozenset({"missing", "declined"})


def _sentence(event: "PickProgress") -> tuple[Severity, str]:
    """Render one library event for a person. Never raises: a missing field yields a plainer line."""
    severity, template = _STAGE_COPY.get(str(event.stage), (Severity.INFO, str(event.stage)))
    fields: dict[str, Any] = {
        "attempt_1": (event.attempt + 1) if event.attempt is not None else "?",
        "attempt_total": event.attempt_total if event.attempt_total is not None else "?",
        "segmentation_count": event.segmentation_count,
        "candidate_count": event.candidate_count,
        "score_2": f"{event.score:.2f}" if event.score is not None else "?",
        "action": event.action or "?",
        "outcome": event.outcome or "?",
        "reasons_joined": ", ".join(event.reasons) if event.reasons else "no reason given",
        "position_short": (
            "x={:.0f} y={:.0f} z={:.0f} mm".format(*event.position_mm)
            if event.position_mm else "an unreported pose"
        ),
    }
    try:
        sentence = template.format(**fields)
    except (KeyError, IndexError, ValueError):  # pragma: no cover (copy bug, not an operator's problem)
        return severity, str(event.stage)
    if event.route:
        # Appended rather than baked into the template: the route is present only on a routed cell, and
        # a template with a permanent "via ..." would read as a missing value everywhere else.
        sentence = f"{sentence} Grounded via the {event.route} route ({event.route_reason})."
    push_reason = (event.extra or {}).get("push_reason")
    if str(event.stage) == "attempt_finished" and event.action == "push" and isinstance(push_reason, str):
        # A push of the failed part says what it did in its own words: pushed, and back at the look, or stopped where
        # the arm stands, a person to decide. Its motion fields are those of a leg that ran after the arm left the
        # look, so "Nothing moved" would be untrue of it.
        sentence = f"{sentence} {push_reason}"
        if event.outcome != "pushed":
            severity = Severity.WARN
    elif event.motion_message or event.motion_error:
        # A refused motion says why, in the guard's own words. Appended rather than templated,
        # because these fields are present only when something declined to move. Without them a
        # precise workspace refusal such as "pose 'approach_00' outside workspace box" reaches the
        # operator's screen as the bare word `execution_failed`.
        why = event.motion_message or event.motion_error or ""
        sentence = f"{sentence} Nothing moved: {why}"
        if event.motion_status:
            sentence = f"{sentence} ({event.motion_status})"
        severity = Severity.WARN
    if event.camera_world in _SPOKEN_CAMERA_WORLDS:
        # Appended at the event's own severity: a declined camera world is a fact about how the
        # motions were planned, and a missing one rides on a refused motion whose status already set
        # the severity.
        sentence = (
            f"{sentence} Camera world: {str(event.camera_world).upper()} "
            f"({event.camera_world_reason})."
        )
    if "labels_seen" in (event.extra or {}):
        # A label the scene does not contain is the one rejection an operator can fix in a second,
        # but only if the line says what perception did return. "no usable grasp" sends them to the
        # lighting; this sends them to the prompt. The library names the failure
        # (``target_label_not_found``); phrasing it is this module's job, as for ``_STAGE_COPY``.
        seen = [str(x) for x in event.extra.get("labels_seen") or ()]
        named = ", ".join(repr(x) for x in seen if x) or "nothing (this source emits no labels)"
        # A sort's pick asks for every rule's kind at once (``target_labels``, 2026-10-09), and names no one label.
        kinds = [str(x) for x in event.extra.get("target_labels") or () if x]
        called = " or ".join(repr(x) for x in kinds) if kinds else repr(event.extra.get("target_label"))
        sentence = (
            f"Nothing in this frame is called {called}. "
            f"Perception returned {len(seen)} object(s), labelled: {named}. "
            f"Re-prompt with a name the detector actually uses; the cell will not substitute "
            f"another object for the one you asked for."
        )
    return severity, sentence


@dataclass
class Run:
    """One operator instruction and everything that came of it."""

    id: str
    prompt: str
    requested_picks: int
    state: RunState = RunState.RUNNING
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    succeeded: int = 0
    attempted: int = 0
    #: Per-pick outcome strings, in order. The typed reason surface, not free text.
    outcomes: list[str] = field(default_factory=list)
    #: Set when the run raised, or ended on a pick that reported a cell that needs a person.
    error: str = ""
    #: Set when an operator asked the run to stop.
    stop_requested: bool = False
    #: Set when the cell was taken down under the run (a Disconnect): why nothing of it may go on. Not a field of
    #: :meth:`to_dict`: it reaches the operator as :attr:`error` when the run ends, which is the one place a UI reads
    #: why a run stopped.
    abandoned: str = ""
    #: How far a push of a failed part moves it, in mm, as the request asked (``None``: the cell's own distance). The run
    #: starts its campaign with it (``service.start_campaign``). Not a field of :meth:`to_dict`.
    push_mm: float | None = None
    #: Which body the run drives.
    kind: RunKind = RunKind.PICK
    #: Why the run ended, typed; ``None`` while it runs. :attr:`error` carries the sentence beside it.
    stop_code: StopCode | None = None
    #: The class of :attr:`stop_code`: what the operator sees, and whether a recovery record was written.
    stop_class: StopClass | None = None
    #: The resolved task plan, as ``TaskPlanOut`` serialises it (a task, a Restart); ``None`` for the other kinds.
    plan: dict[str, Any] | None = None
    #: A task: the parts released at the place.
    parts_placed: int = 0
    #: The program believed a part was in the jaws at the end. Every gate reads the live hand first; the stop record
    #: carries this belief, which keeps ``part_still_held`` standing for a hand that can say nothing itself.
    holding: bool = False
    #: The run a Restart continues.
    restart_of: str | None = None
    #: "Halt now" was pressed during the run.
    halt_requested: bool = False
    #: The last timeline step (``countdown``, ``survey``, ``look``, ``detect``, ``grasp``, ``place``, ``return``).
    step: str = ""
    #: The operator asked a task to stop after the part in hand: the part in flight is placed and the arm returns. Not a
    #: field of :meth:`to_dict`: :attr:`stop_requested` says it.
    stop_after_part: bool = False
    #: What a kind's driver reads besides the plan (a Home run's ``to``, a teach's ``name`` and ``label``), JSON only;
    #: ``run_started`` says it. Not a field of :meth:`to_dict`.
    inputs: dict[str, Any] = field(default_factory=dict)
    #: A refusal met inside the run before anything moved, ``{"code", "status"}`` as ``RunRefusalOut`` says it: the
    #: library's (a task), or ``jaws_question_pending`` where a moving run gave way to a jaws check that began as it
    #: started; ``None`` otherwise.
    refusal: dict[str, Any] | None = None
    #: The run as it was accepted, before its thread started: what a 202 answers (a fast run may already have ended by
    #: the time the request answers). Not a field of :meth:`to_dict`.
    accepted: dict[str, Any] | None = field(default=None, repr=False)

    def as_accepted(self) -> dict[str, Any]:
        """The run as it was accepted, before its thread started (what a 202 answers); :meth:`to_dict` where none
        was kept."""
        return dict(self.accepted) if self.accepted is not None else self.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "prompt": self.prompt,
            "requested_picks": self.requested_picks,
            "state": str(self.state),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "succeeded": self.succeeded,
            "attempted": self.attempted,
            "outcomes": list(self.outcomes),
            "error": self.error,
            "stop_requested": self.stop_requested,
            "kind": str(self.kind),
            "stop_code": "" if self.stop_code is None else str(self.stop_code),
            "stop_class": "" if self.stop_class is None else str(self.stop_class),
            "plan": None if self.plan is None else dict(self.plan),
            "parts_placed": self.parts_placed,
            "holding": self.holding,
            "restart_of": self.restart_of,
            "halt_requested": self.halt_requested,
            "step": self.step,
            "refusal": None if self.refusal is None else dict(self.refusal),
        }


def kept_run(run: Run) -> dict[str, Any]:
    """``run`` as the console keeps it beside its stop record across a restart (``Console.stop_file``): what
    :meth:`Run.to_dict` says, with what a Restart reads besides (``inputs``, ``push_mm``). Kept as it ends, while the run
    still holds the lock and still reads as running: the state it ends in is its stop class's (:data:`STATE_OF_CLASS`)."""
    said = run.to_dict()
    if run.state is RunState.RUNNING and run.stop_class is not None:
        said["state"] = str(STATE_OF_CLASS[run.stop_class])
    said["inputs"] = dict(run.inputs)
    said["push_mm"] = run.push_mm
    return said


def run_from_kept(said: Mapping[str, Any]) -> Run:
    """The run :func:`kept_run` kept, as a finished run of this console; ``ValueError`` (or ``KeyError``,
    ``TypeError``) for anything else, a run still running included: a kept run is one that ended."""

    def text(key: str) -> str:
        value = said[key]
        if not isinstance(value, str):
            raise TypeError(f"the kept run's {key} is no text: {value!r}")
        return value

    def number(key: str, *, optional: bool = False) -> float | None:
        value = said.get(key)
        if value is None and optional:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"the kept run's {key} is no number: {value!r}")
        return float(value)

    def mapping(key: str) -> dict[str, Any] | None:
        value = said.get(key)
        if value is None:
            return None
        if not isinstance(value, dict):
            raise TypeError(f"the kept run's {key} is no mapping: {value!r}")
        return dict(value)

    state = RunState(text("state"))
    if state is RunState.RUNNING:
        raise ValueError("a kept run is one that ended, and this one says it is still running")
    code = text("stop_code")
    stop_code = StopCode(code) if code else None
    outcomes = said.get("outcomes") or []
    if not isinstance(outcomes, list) or not all(isinstance(outcome, str) for outcome in outcomes):
        raise TypeError(f"the kept run's outcomes are no list of words: {outcomes!r}")
    restart_of = said.get("restart_of")
    return Run(
        id=text("id"), prompt=text("prompt"), requested_picks=int(number("requested_picks") or 0), state=state,
        started_at=float(number("started_at") or 0.0), finished_at=number("finished_at", optional=True),
        succeeded=int(number("succeeded") or 0), attempted=int(number("attempted") or 0), outcomes=list(outcomes),
        error=text("error"), stop_requested=bool(said.get("stop_requested", False)), kind=RunKind(text("kind")),
        stop_code=stop_code, stop_class=STOP_CLASS[stop_code] if stop_code is not None else None, plan=mapping("plan"),
        parts_placed=int(number("parts_placed") or 0), holding=bool(said.get("holding", False)),
        restart_of=restart_of if isinstance(restart_of, str) else None,
        halt_requested=bool(said.get("halt_requested", False)), step=str(said.get("step") or ""),
        inputs=mapping("inputs") or {}, refusal=mapping("refusal"), push_mm=number("push_mm", optional=True),
    )


class RunRegistry:
    """Starts runs, keeps their history, and is the only thing that flips the console's run lock."""

    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self._lock = threading.RLock()
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._threads: dict[str, threading.Thread] = {}
        #: The hands-off countdown (:meth:`_countdown`): its steps, each step's length and how often it reads the run.
        self.countdown_steps = COUNTDOWN_STEPS
        self.countdown_step_s = COUNTDOWN_STEP_S
        self.countdown_poll_s = COUNTDOWN_POLL_S
        #: Called with the id of every run the registry forgets (:data:`RETAINED_RUNS`): what else keeps something of
        #: a run, its overlays, lets go of it in the same breath (``Console`` wires its overlay store here).
        self.on_forget: Callable[[str], None] | None = None

    # --- reading ------------------------------------------------------------------------------------

    def get(self, run_id: str) -> Run | None:
        with self._lock:
            return self._runs.get(run_id)

    def recent(self, limit: int = 50) -> list[Run]:
        """The newest runs first, at most ``limit`` and never more than :data:`RETAINED_RUNS`."""
        with self._lock:
            return [self._runs[i] for i in reversed(self._order[-limit:])]

    def active(self) -> Run | None:
        with self._lock:
            return next(
                (r for r in self._runs.values() if r.state is RunState.RUNNING), None
            )

    def restore(self, run: Run) -> bool:
        """Know a run again that a console before this one ran: the stopped run its stop record names, read back after a
        restart of the server (``Console.stop_file``), so ``GET /v1/runs/<id>`` answers it and Restart replays its plan.

        Kept as the oldest run, finished, with no thread and no events in this process (its stream replays nothing).
        ``False``, and nothing kept, for a run still running or one whose id this registry knows already.
        """
        if run.state is RunState.RUNNING:
            return False
        with self._lock:
            if run.id in self._runs:
                return False
            self._runs[run.id] = run
            self._order.insert(0, run.id)
            self._forget_runs_beyond_the_window()
        logger.warning("Run %s (%s, %s) is known again from the stop a console before this one kept.", run.id, run.kind,
                       run.stop_code)
        return True

    # --- driving ------------------------------------------------------------------------------------

    def start(self, console: Any, *, prompt: str, picks: int, push_mm: float | None = None) -> Run:
        """Begin a run on its own thread and return immediately.

        Refuses if one is already running: a second concurrent pick would drive the same arm from two
        threads, which is not a queue but a collision. ``push_mm`` is the run's push distance, which its
        campaign starts with; the router refuses one the cell would refuse before calling this. And refuses
        (:class:`RunRefused`) while the console's recovery record stands: a pick is no way back from a stop.
        """
        with self._lock:
            if (existing := self.active()) is not None:
                logger.warning(
                    "Run refused: %s is still active (%d/%d done).",
                    existing.id, existing.succeeded, existing.attempted,
                )
                raise RunConflict(existing)
            _admit_past_the_stop(console, RunKind.PICK, None)
            run = Run(id=new_run_id(), prompt=prompt, requested_picks=picks, push_mm=push_mm)
            run.accepted = run.to_dict()
            self._runs[run.id] = run
            self._order.append(run.id)
            console.active_run_id = run.id
            thread = threading.Thread(
                target=self._drive, args=(console, run), name=f"willy-{run.id}", daemon=True
            )
            self._threads[run.id] = thread
            self._forget_runs_beyond_the_window()
        logger.info(
            "Run %s starting: %d pick(s), prompt %r.", run.id, run.requested_picks, run.prompt
        )
        thread.start()
        return run

    def _forget_runs_beyond_the_window(self) -> None:
        """Drop the runs that have fallen out of :data:`RETAINED_RUNS`. Caller holds the lock.

        Whole, and in one place: the record, the thread handle and the hub's history for that id go
        together. Dropping the history alone would leave ``since()`` answering "no events, none
        dropped" for a run ``/v1/runs/<id>`` still returns: a silently short replay, which is the
        one thing the sequence numbers exist to prevent.

        Only finished runs can be reached: one arm means one active run, and it is always the
        newest.
        """
        while len(self._order) > RETAINED_RUNS:
            forgotten = self._order.pop(0)
            self._runs.pop(forgotten, None)
            self._threads.pop(forgotten, None)
            self.hub.forget(forgotten)
            if self.on_forget is not None:
                try:
                    self.on_forget(forgotten)
                except Exception as exc:  # noqa: BLE001 (forgetting reports, it does not fail a new run)
                    logger.warning("Forgetting what else kept run %s failed: %s: %s", forgotten,
                                   type(exc).__name__, exc)

    def stop(self, run_id: str) -> bool:
        """Ask a run to stop between attempts. Returns False if it was not running.

        Not a kill: the flag is read by the pick loop before it begins the next attempt, so a motion
        already in flight completes.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.state is not RunState.RUNNING:
                logger.info("Stop asked for %s, which is not running; nothing to do.", run_id)
                return False
            run.stop_requested = True
        logger.warning(
            "Stop requested for %s. The attempt in flight finishes; this is not a kill.", run_id
        )
        self.hub.publish(
            run_id, "run_stop_requested", severity=Severity.WARN,
            human="Stop requested. The current attempt will finish; no new one will start.",
            # A pick run's stop acts between its attempts; a task's stop after its part and a Disconnect's say their own.
            data={"scope": "between_attempts"} if run.kind is RunKind.PICK else None,
        )
        return True

    def abandon(self, because: str) -> Run | None:
        """Stop the active run because the cell is coming down under it; the run it stopped, or ``None``.

        Called by Disconnect BEFORE the arm and the hand come down (``api/routers/cell.py``), which is what makes the
        order hold: by the time the hardware is gone the run's flag is set, so whatever of the run returns next meets
        the stop before it starts anything new. ``pick()`` refuses to begin on it, the pick loop begins no attempt on
        it (the first one of a pick that was waiting on a person's answer at its start included), and this run loop
        begins no next pick. The run ends FAILED with ``because`` as its error, not CANCELLED: nobody pressed Stop,
        and the cell went down with the run in it, so the operator is sent to the arm and the jaws, not to the next
        prompt.

        Not a kill, as :meth:`stop` is not: what is already inside an attempt meets the disconnected arm. And the run
        keeps the console's run lock until its thread has ended, which is the other half of the guarantee: Connect and
        Build refuse while it is held, so the old run can never go on with an arm or a hand a later Connect brings up.
        """
        with self._lock:
            run = self.active()
            if run is None:
                return None
            run.stop_requested = True
            run.abandoned = because
        logger.warning(
            "Run %s abandoned (%d/%d done): %s The pick in flight finishes against a disconnected cell; the run "
            "keeps the console's run lock until its thread ends.",
            run.id, run.succeeded, run.requested_picks, because,
        )
        self.hub.publish(
            run.id, "run_stop_requested", severity=Severity.WARN,
            human=(f"The cell is being disconnected, so the run stops: {because} Nothing of it starts again; a pick "
                   f"in flight meets the disconnected arm."),
            reason=because, scope="disconnect",
        )
        return run

    def stop_after_part(self, run_id: str) -> bool:
        """Ask a task to stop once the part in hand is placed and the arm is back. False if it was not running.

        Not wired to the service's cancel check: the pick in flight finishes its attempts and pushes, a held part is
        placed, and the arm returns before the task ends ``stopped_after_part``. During the countdown, before the first
        motion, the run ends ``cancelled`` with nothing moved.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.state is not RunState.RUNNING:
                logger.info("Stop after this part asked for %s, which is not running; nothing to do.", run_id)
                return False
            run.stop_after_part = True
            run.stop_requested = True
        logger.warning("Stop after this part requested for %s: the part in hand is placed and the arm returns first.",
                       run_id)
        self.hub.publish(
            run_id, "run_stop_requested", severity=Severity.WARN,
            human=("Stop after this part: the part in hand is still placed and the arm returns, then the task ends. "
                   "Not an emergency stop; the red button is."),
            scope="after_part",
        )
        return True

    def stop_home(self, run_id: str) -> bool:
        """Ask a Home run not to send its move, or a wave its next swing. False if it was not running.

        During its countdown, or any time before its one move is sent, the run ends ``cancelled`` with nothing moved
        (the countdown and ``drive_home`` read the flag); a move already sent runs to its end, since this is not a kill.
        A wave reads the flag before every swing (``drive_wave``) and ends ``cancelled`` where the arm stands. Nothing
        is latched, so no "the cell is clear" is owed, as the brake would owe one.
        """
        with self._lock:
            run = self._runs.get(run_id)
            if run is None or run.state is not RunState.RUNNING:
                logger.info("Stop asked for Home run %s, which is not running; nothing to do.", run_id)
                return False
            run.stop_requested = True
        logger.warning("Stop requested for Home run %s: its move is not sent unless it is already under way.", run_id)
        self.hub.publish(
            run_id, "run_stop_requested", severity=Severity.WARN,
            human=("Stop: the move is not sent; a move already under way runs to its end. Not an emergency stop; the "
                   "red button is."),
            # A Home run's stop acts before its one move is sent, as a task's after its part and a pick's between its
            # attempts: the browser words each scope its own way.
            scope="before_motion",
        )
        return True

    def halt(self, because: str, *, in_motion: bool = False, braking: bool = False,
             requested_at: float | None = None) -> Run | None:
        """"Halt now" for the active run: it commands nothing more. The run it halted, or ``None`` where none was active.

        The run-level half of ``POST /v1/cell/brake``, which latches the arm first. The run reads the flag after every
        motion (a task, Home) and its service's cancel check reads it between attempts (a pick, a task's pick); a
        countdown reads it every 100 ms. ``in_motion`` and ``braking`` say what the arm's latch said of the move in
        flight; ``run_halt_requested`` carries them.
        """
        with self._lock:
            run = self.active()
            if run is None:
                return None
            run.halt_requested = True
        at = time.time() if requested_at is None else float(requested_at)
        logger.warning("Halt requested for run %s (%s): %s; in motion %s, braking %s.", run.id, run.kind, because,
                       in_motion, braking)
        self.hub.publish(
            run.id, "run_halt_requested", severity=Severity.WARN,
            human=("Halt now: the run commands nothing more, and the arm refuses every next motion until a person "
                   "confirms the cell is clear. Not an emergency stop; the red button is."),
            reason=because, in_motion=bool(in_motion), requested_at=at, braking=bool(braking),
        )
        return run

    def join(self, run_id: str, timeout_s: float) -> bool:
        """Wait at most ``timeout_s`` for the thread of ``run_id`` to end; ``True`` once none is left."""
        with self._lock:
            thread = self._threads.get(run_id)
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout=timeout_s)
        return not thread.is_alive()

    def _drive(self, console: Any, run: Run) -> None:
        """The pick run's body. Everything it can raise is caught: a dead thread must still close its stream.

        Its end is typed (``stop_code``) beside the pinned sentences, and a problem stop leaves the console's recovery
        record while the run still holds the lock: the state flips only after the record stands.
        """
        service = console.session.service
        hub = self.hub
        # What the run's prompt replaced, put back in the `finally`. `None` when the run named none.
        previous_prompt: Any = None
        # How the run ends, decided in the body and applied in the `finally`, after the recovery record.
        code: StopCode = StopCode.SOFTWARE_ERROR
        state = RunState.FAILED

        def _on_progress(event: "PickProgress") -> None:
            severity, human = _sentence(event)
            step = _STEP_OF_STAGE.get(str(event.stage))
            if step:
                run.step = step
            hub.publish(
                run.id, f"pick.{event.stage}", severity=severity, human=human,
                step=str(event.stage), step_index=event.attempt, step_total=event.attempt_total,
                **{k: v for k, v in _payload(event).items()},
            )

        try:
            started: dict[str, Any] = {"kind": str(run.kind), "prompt": run.prompt, "requested_picks": run.requested_picks}
            if run.push_mm is not None:
                # The push distance the run asked for, where it asked for one: the one place its record keeps it.
                started["push_mm"] = run.push_mm
            hub.publish(
                run.id, "run_started", severity=Severity.INFO,
                human=f"Run started: {run.requested_picks} pick(s), prompt {run.prompt!r}.",
                **started,
            )
            # Read again now that this run holds the cell: a jaws check that began after the route read the question.
            began = _jaws_question_began(console)
            if began:
                code, state = StopCode.CANCELLED, RunState.CANCELLED
                _give_way_to_the_jaws_question(run, began)
                logger.warning("Run %s (%s) ends before its first motion: %s.", run.id, run.kind, began)
                return
            # Hands off first, where a person's hands were last at the arm (a teach, "open now", "Backen leer"); a
            # console that keeps no stamps counts nothing down.
            if not self._countdown(console, run):
                code, state = StopCode.CANCELLED, RunState.CANCELLED
                return
            # A recovery of an earlier pick that stopped where the arm stands needs a person first. Read BEFORE the
            # campaign, which would acknowledge it: no run clears that latch silently, and nothing is commanded.
            waiting = _latch_of(service)
            if waiting:
                stopped_by = ("a recovery of an earlier pick stopped where the arm stands, and a person decides "
                              f"first; the run started nothing and commanded nothing: {waiting}")
                code, state = StopCode.RECOVERY_NEEDS_PERSON, RunState.FAILED
                run.error = stopped_by
                logger.error("Run %s FAILED before its first pick: %s", run.id, stopped_by)
                hub.publish(run.id, "run_error", severity=Severity.ERROR, human=f"The run stopped: {stopped_by}",
                            error=stopped_by, stop_code=str(code))
                return
            service.attach_progress_listener(_on_progress)
            # Between attempts the pick loop stops on the operator's stop, on "halt now" and on a cell taken down.
            service.set_cancel_check(lambda: run.stop_requested or run.halt_requested or bool(run.abandoned))
            # A run is a campaign: fresh push budgets (1 per part, 2 per pick, 5 per run), no part skipped by
            # next_target, and the run's push distance. A distance the cell refuses ends the run here, before any pick
            # (the router refused it already; this says it again for a caller that did not go through the router).
            start_campaign = getattr(service, "start_campaign", None)
            if callable(start_campaign):
                start_campaign(**({} if run.push_mm is None else {"push_mm": run.push_mm}))
            # The prompt reaches the detector, for this run only. A label filter alone is not enough:
            # every frame would still be grounded with the build phrase ("object"), and "the red
            # cube", typed or spoken, would filter on a label no phrase grounder returns. `set_prompt`
            # sets the phrase, the labels the detector's words map onto and the filter together, and
            # the `finally` below puts all three back: the console keeps one cell for the whole
            # process, and a prompt that outlived its run would have every later run hunting it. An
            # empty prompt keeps the build phrase and filters nothing.
            if run.prompt.strip():
                previous_prompt = service.set_prompt(run.prompt)

            # Asked once, as `PickRun` asks it: neither the cell profile's looks nor whether a camera sits
            # on the wrist changes between two picks. A wrist camera sees what the arm points it at, so
            # each pick looks from the looks the cell profile configures, else from the arm's home;
            # before this the console handed no look, and a wrist camera perceived from wherever the last
            # pick had left the arm. A service handed no look is called as it always was, so a fixed
            # camera, and a service that takes none, still runs. `both_faces` stays off: the console runs
            # the fast rule, and the switch is a program's (`PickRun`, `pick`).
            look = _look_for(service)

            # Why the run ended before its last pick although nothing raised; "" while it has not.
            stopped_by = ""
            stopped_code: StopCode | None = None
            for _ in range(run.requested_picks):
                if run.stop_requested or run.halt_requested:
                    break
                # Before every pick, the first included: a toggle hand the program believes CLOSED, or whose jaws
                # nobody can place (a change that failed, DO0 switched by hand at the pendant), would have `pick()` ask
                # a person where the jaws stand. From this thread that
                # question goes to the SERVER's terminal, where nobody watching the console sees it, Stop cannot
                # reach `input()`, and a 'p' typed later drops the part wherever the arm then stands. A console run
                # never releases what it lifted, so on a toggle every pick after a success would ask. The run ends
                # here instead, as the other stops end it.
                stopped_by = _why_no_pick_starts(service)
                if stopped_by:
                    stopped_code = StopCode.HAND_NEEDS_PERSON
                    break
                # And inside the pick nobody is asked either: the pick asks the hand at its start and again before its
                # approach, seconds after the check above, and DO0 switched at the pendant in between made it ask at
                # the server's terminal (review of 2026-09-28). There a question is a refusal, nothing is sent, and the
                # pick ends as a hand that needs a person, which ends the run.
                with _asking_nobody(service):
                    report = service.pick() if look is None else service.pick(look=look)
                fault = getattr(report, "fault", None)
                if fault is not None:
                    # A fault of the cell still ends the run: the next pick would meet the same dead
                    # camera or the same dropped link. `pick()` reports it, so the run raises it here.
                    raise _CellFault(fault)
                run.attempted += 1
                outcome = str(getattr(report, "outcome", "unknown"))
                run.outcomes.append(outcome)
                ok = bool(getattr(report, "succeeded", False))
                run.succeeded += int(ok)
                # A console pick run never sets its part down: what the last pick lifted is still in the jaws.
                run.holding = ok
                if ok:
                    # The arm moved: it gripped and lifted. What ends the hands-off countdown; a pick that moved
                    # nothing anybody can vouch for leaves it due, and the next run counts down once more.
                    motion_started(console)
                # One line per pick, not per progress event: a pick is seconds of physical motion, so
                # this is the natural grain. The per-stage stream lives in the event hub.
                logger.info(
                    "Run %s pick %d/%d: %s (%s).",
                    run.id, run.attempted, run.requested_picks, outcome,
                    "succeeded" if ok else report.failure_summary() or "failed",
                )
                hub.publish(
                    run.id, "pick_result",
                    severity=Severity.SUCCESS if ok else Severity.WARN,
                    human=(
                        f"Pick {run.attempted}: {outcome}."
                        if ok else f"Pick {run.attempted}: {outcome}. {report.failure_summary()}"
                    ),
                    outcome=outcome, succeeded=ok, reason=report.failure_summary(),
                    # Where it looked from (a fixed camera handed looks: those it moved to, the last maybe not
                    # reached, as the reason says) and which looks its grasp was fused from (a wrist camera's).
                    looks=_names(getattr(report, "looks", ())),
                    looks_fused=_names(getattr(report, "looks_fused", ())),
                    # What else the looks came to, each key only where the report says it.
                    **_what_the_looks_came_to(report),
                    # And what else the report says, each key only where it says it, in its type.
                    **pick_fields(report),
                    pick=run.attempted,
                )
                # After the pick is counted and said: it ran, and its outcome is part of the run's
                # record. What ends the run is that the next pick must not start.
                stopped_code, stopped_by = _stop_of(report, _arm_of(service), halt_requested=run.halt_requested)
                if stopped_by:
                    break
            if run.abandoned:
                # The cell went down with the run in it. That is the reason the operator must read first; what the
                # last pick said, where it said anything, follows it.
                stopped_by = (f"{run.abandoned} The last pick also said: {stopped_by}" if stopped_by
                              else run.abandoned)
                stopped_code = StopCode.DISCONNECTED
            elif not stopped_by and run.halt_requested:
                stopped_by = _halted_sentence(_arm_of(service), "the pick run started no further attempt")
                stopped_code = StopCode.HALTED
            if stopped_by:
                # FAILED, not CANCELLED: nobody asked for this stop, the cell needs a person, and a run
                # that read as finished or cancelled would send the operator to the next prompt instead
                # of to the arm. Whatever a stop request said, this is what the operator must read.
                code, state = stopped_code or StopCode.SOFTWARE_ERROR, RunState.FAILED
                run.error = stopped_by
                logger.error(
                    "Run %s FAILED after pick %d/%d: %s",
                    run.id, run.attempted, run.requested_picks, stopped_by,
                )
                hub.publish(
                    run.id, "run_error", severity=Severity.ERROR,
                    human=f"The run stopped: {stopped_by}", error=stopped_by, stop_code=str(code),
                )
            elif run.stop_requested:
                code, state = StopCode.CANCELLED, RunState.CANCELLED
            else:
                code, state = StopCode.FINISHED, RunState.FINISHED
        except BaseException as exc:  # noqa: BLE001 (the stream must close whatever happened)
            raised = exc.fault if isinstance(exc, _CellFault) else exc
            state = RunState.FAILED
            code = (StopCode.DISCONNECTED if run.abandoned
                    else StopCode.CELL_FAULT if isinstance(exc, _CellFault) else StopCode.SOFTWARE_ERROR)
            run.error = f"{type(raised).__name__}: {raised}"
            if run.abandoned:
                # A pick in flight when the cell went down usually raises on the arm that went away; the disconnect
                # is why, and the raise is how it showed.
                run.error = f"{run.abandoned} The pick in flight then raised {run.error}"
            # With the traceback: this thread is the only place it exists. The envelope published
            # below carries the one-line reason to the browser and nothing carries the stack, so
            # without this a run that dies in the library leaves the operator a sentence and no
            # thread to pull on.
            logger.exception("Run %s FAILED: %s", run.id, run.error)
            hub.publish(
                run.id, "run_error", severity=Severity.ERROR,
                human=f"The run stopped with an error: {run.error}", error=run.error, stop_code=str(code),
            )
        finally:
            run.finished_at = time.time()
            # Detach before releasing the lock: a listener left attached would publish into a finished
            # run's stream on the next pick, which a UI would render as a run resurrecting itself.
            try:
                service.attach_progress_listener(None)
                service.set_cancel_check(None)
            except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
                # Worth a line of its own: a detach that does not take is exactly how a listener
                # stays attached, which is what the ordering above exists to prevent.
                logger.warning(
                    "Detaching the progress listener from run %s failed: %s: %s",
                    run.id, type(exc).__name__, exc,
                )
            if previous_prompt is not None:
                # Before the run lock is released, so the next run starts on the cell's own phrase.
                # Teardown reports and does not propagate.
                try:
                    service.set_prompt(previous_prompt)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Putting back the prompt run %s replaced failed: %s: %s. The cell may still "
                        "ground %r.", run.id, type(exc).__name__, exc, run.prompt,
                    )
            run.stop_code, run.stop_class = code, STOP_CLASS[code]
            # The record before the lock: nothing new can start between a problem stop and the record that gates it.
            self._record(console, run)
            with self._lock:
                run.state = state
                if getattr(console, "active_run_id", None) == run.id:
                    console.active_run_id = None
                # The Thread object is useless the moment its run ends and nothing reads this dict
                # again, so it goes now rather than at the retention boundary. It was never popped
                # at all, which is how a dict with no reader grew one entry per run for the life of
                # the process.
                self._threads.pop(run.id, None)
            logger.info(
                "Run %s %s (%s): %d/%d succeeded in %.1f s.",
                run.id, run.state, code, run.succeeded, run.attempted,
                (run.finished_at or time.time()) - run.started_at,
            )
            hub.publish(
                run.id, "run_finished",
                severity=Severity.SUCCESS if run.state is RunState.FINISHED else Severity.WARN,
                human=(
                    f"Run {run.state}: {run.succeeded}/{run.attempted} succeeded."
                ),
                # The whole record, as data: a run's own `step` is not the envelope's.
                data=run.to_dict(),
            )

    def _record(self, console: Any, run: Run) -> None:
        """Write the console's recovery record for ``run`` where its end calls for one; a console that keeps none (a
        program's own) is left alone. Teardown reports, it does not propagate: the lock must still come back."""
        recorder = getattr(console, "record_problem_stop", None)
        if not callable(recorder):
            return
        try:
            recorder(run)
        except Exception as exc:  # noqa: BLE001
            logger.error("Writing the recovery record of run %s failed: %s: %s", run.id, type(exc).__name__, exc)

    # --- the other kinds: a task, Home, a teach, the planner -----------------------------------------------------

    def start_kind(
        self,
        console: Any,
        kind: RunKind | str,
        driver: RunDriver,
        *,
        inputs: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> Run:
        """Begin a run of ``kind`` on its own thread and return at once, as :meth:`start` does for a pick.

        One lock for every kind: refused with :class:`RunConflict` while any run is active, a pick included. A moving
        kind is refused (:class:`RunRefused`) where the console's recovery record does not let it start, read under the
        same lock: until a person said the cell is clear nothing moving starts, and after it only Home and the Restart
        of the record's own run (``restart_of``). Then, on the run's thread, in this order:

        1. ``run_started`` (``kind``, the ``plan``, ``restart_of``, and ``inputs``);
        2. for a moving kind (pick, task, home), the hands-off countdown where it is due (:meth:`_countdown`); a run
           stopped during it ends ``cancelled`` with nothing moved;
        3. ``driver(console, run)``, which answers the stop code, and stamps the console's ``last_motion_started_at``
           (:func:`motion_started`) once it knows the arm moved, so a run that moved nothing leaves the countdown due;
        4. :meth:`_finish`: the class and the state, ``run_error`` for a failed run, the recovery record for a problem
           stop of a moving kind, and only then the lock handed back and ``run_finished``.

        ``fields`` are the run's own fields a caller may set (``prompt``, ``requested_picks``, ``plan``,
        ``restart_of``, ``push_mm``); ``inputs`` what the driver reads besides them, JSON only, since
        ``run_started`` says it. The gates (a halted arm, a recovery record, a question waiting) are the routes', and
        are checked before this is called; the record is read here once more, under the lock.
        """
        kind = RunKind(kind)
        unknown = sorted(set(fields) - _START_FIELDS)
        if unknown:
            raise TypeError(f"start_kind() takes no run field(s) {', '.join(unknown)}; it sets {sorted(_START_FIELDS)}")
        said = dict(inputs or {})
        # Refused here rather than on the run's thread: a run_started that cannot be serialised would reach no browser.
        json.dumps({"inputs": said, "plan": fields.get("plan")})
        with self._lock:
            if (existing := self.active()) is not None:
                logger.warning(
                    "%s run refused: %s (%s) is still active.", kind.capitalize(), existing.id, existing.kind,
                )
                raise RunConflict(existing)
            _admit_past_the_stop(console, kind, fields.get("restart_of"))
            run = Run(
                id=new_run_id(),
                prompt=str(fields.pop("prompt", "")),
                requested_picks=int(fields.pop("requested_picks", 0)),
                kind=kind,
                inputs=said,
                **fields,
            )
            run.accepted = run.to_dict()
            self._runs[run.id] = run
            self._order.append(run.id)
            console.active_run_id = run.id
            thread = threading.Thread(
                target=self._drive_kind, args=(console, run, driver), name=f"willy-{run.id}", daemon=True
            )
            self._threads[run.id] = thread
            self._forget_runs_beyond_the_window()
        logger.info("Run %s starting: %s%s.", run.id, kind, f", restarting {run.restart_of}" if run.restart_of else "")
        thread.start()
        return run

    def _drive_kind(self, console: Any, run: Run, driver: RunDriver) -> None:
        """The thread of a :meth:`start_kind` run. Everything it can raise is caught: a dead thread must still close
        its stream, write its record and hand the lock back."""
        code = StopCode.SOFTWARE_ERROR
        try:
            self.hub.publish(
                run.id, "run_started", severity=Severity.INFO, human=f"{run.kind.capitalize()} run started.",
                data=_started(run),
            )
            began = _jaws_question_began(console) if run.kind in MOVING_KINDS else ""
            if began:
                # Read again now that this run holds the cell: a jaws check that began after the route read the
                # question. Nothing moved, and nothing is sent while a person may have a hand at the jaws.
                _give_way_to_the_jaws_question(run, began)
                logger.warning("Run %s (%s) ends before its first motion: %s.", run.id, run.kind, began)
                code = StopCode.CANCELLED
            elif run.kind in MOVING_KINDS and not self._countdown(console, run):
                code = StopCode.CANCELLED
            else:
                # The driver stamps the motion itself (`motion_started`), once it knows the arm moved.
                code = StopCode(driver(console, run))
                if run.abandoned:
                    # The cell went down with the run in it: that is what the operator must read first, and what the
                    # run said follows it, as a pick run says it.
                    said = f" The run also said: {run.error}" if run.error else ""
                    code, run.error = StopCode.DISCONNECTED, f"{run.abandoned}{said}"
        except BaseException as exc:  # noqa: BLE001 (the stream must close and the lock come back whatever happened)
            code = StopCode.DISCONNECTED if run.abandoned else StopCode.SOFTWARE_ERROR
            run.error = f"{type(exc).__name__}: {exc}"
            if run.abandoned:
                run.error = f"{run.abandoned} The run then raised {run.error}"
            # With the traceback: this thread is the only place it exists.
            logger.exception("Run %s (%s) FAILED: %s", run.id, run.kind, run.error)
        finally:
            self._finish(console, run, code)

    def _countdown(self, console: Any, run: Run) -> bool:
        """The hands-off countdown before a moving run's first motion: ``True`` to go on, ``False`` when it stopped.

        Due where a person's hands were at the arm since its last motion (a teach, "open now", "Backen leer":
        ``Console.countdown_because``): ``run_countdown`` :attr:`countdown_steps` times, :attr:`countdown_step_s`
        apart, each saying the seconds left and why, while the run's stop, halt and abandon are read every
        :attr:`countdown_poll_s`. Stopped, the run ends ``cancelled`` with nothing moved, and the countdown stays due: no
        motion took the person's hands away from the arm. A console that keeps no stamps counts nothing down.
        """
        because = _countdown_because(console)
        if because is None:
            return True
        run.step = "countdown"
        logger.warning("Run %s (%s) counts down %d s before its first motion: %s.", run.id, run.kind,
                       self.countdown_steps, because)
        for left in range(self.countdown_steps, 0, -1):
            self.hub.publish(
                run.id, "run_countdown", severity=Severity.WARN,
                human=(f"Hands off: the arm moves by itself in {left} s "
                       + ("(a pose was taught by hand)." if because == "teach"
                          else "(a person's hands were at the jaws).")),
                seconds_left=left, because=because,
            )
            deadline = time.monotonic() + max(0.0, float(self.countdown_step_s))
            while True:
                why = _stopped_before_the_first_motion(run)
                if why:
                    run.error = f"{why} during the hands-off countdown, so the run ended before its first motion; " \
                                "nothing moved"
                    logger.warning("Run %s (%s) ends before its first motion: %s.", run.id, run.kind, run.error)
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(max(0.001, float(self.countdown_poll_s)), remaining))
        why = _stopped_before_the_first_motion(run)
        if why:
            run.error = f"{why} as the countdown ended, so the run ended before its first motion; nothing moved"
            return False
        return True

    def _finish(self, console: Any, run: Run, code: StopCode) -> None:
        """End a :meth:`start_kind` run on ``code``: its class and state, ``run_error``, the recovery record, the lock.

        The record is written while the run still holds the lock and still reads as running, so no new run can start
        between the stop and the record that gates it; ``RunRegistry.active`` looks at the state, and the state flips
        with the lock, under one lock.
        """
        stop_class = STOP_CLASS[code]
        state = STATE_OF_CLASS[stop_class]
        run.stop_code, run.stop_class = code, stop_class
        run.finished_at = time.time()
        if state is RunState.FAILED:
            if not run.error:
                run.error = f"the {run.kind} run stopped: {code}"
            logger.error("Run %s (%s) FAILED (%s): %s", run.id, run.kind, code, run.error)
            self.hub.publish(
                run.id, "run_error", severity=Severity.ERROR, human=f"The run stopped: {run.error}",
                error=run.error, stop_code=str(code),
            )
        self._record(console, run)
        with self._lock:
            run.state = state
            if getattr(console, "active_run_id", None) == run.id:
                console.active_run_id = None
            self._threads.pop(run.id, None)
        logger.info(
            "Run %s (%s) %s: %s in %.1f s.", run.id, run.kind, state, code, run.finished_at - run.started_at,
        )
        self.hub.publish(
            run.id, "run_finished",
            severity=Severity.SUCCESS if state is RunState.FINISHED else Severity.WARN,
            human=f"{run.kind.capitalize()} run {state}: {code}.",
            data=run.to_dict(),
        )


def _started(run: Run) -> dict[str, Any]:
    """What ``run_started`` says of a :meth:`RunRegistry.start_kind` run: its kind, plan, restart and inputs."""
    said: dict[str, Any] = {"kind": str(run.kind)}
    if run.plan is not None:
        said["plan"] = dict(run.plan)
    if run.restart_of:
        said["restart_of"] = run.restart_of
    if run.kind is RunKind.PICK:
        said.update(prompt=run.prompt, requested_picks=run.requested_picks)
    if run.push_mm is not None:
        said["push_mm"] = run.push_mm
    for key, value in run.inputs.items():
        said.setdefault(key, value)
    return said


def motion_started(console: Any) -> None:
    """Stamp ``console`` with the arm's motion (``Console.stamp_motion_started``): what ends the hands-off countdown.

    Called by a moving run's driver once it knows the arm moved (a pick that gripped and lifted, a part placed, an
    arrival at a return pose), never before: a run that counted down and then moved nothing (a pose its screen refused,
    a move the planner refused before anything was sent, a gate inside the library) leaves the countdown due for the
    next run, and a run that moved without saying so costs the next one three seconds at most. A console that keeps no
    stamps (a program's own, a test's) is left alone.
    """
    stamp = getattr(console, "stamp_motion_started", None)
    if callable(stamp):
        stamp()


def _admit_past_the_stop(console: Any, kind: RunKind, restart_of: str | None) -> None:
    """Refuse (:class:`RunRefused`) a moving run the console's recovery record does not let start; the caller holds the
    run lock.

    A route reads the record among its gates before it starts the run, and a run that stopped on a problem in between
    writes its record before it lets go of the lock (``_drive``, :meth:`RunRegistry._finish`), so read here, under the
    lock, it misses none. Until a person said the cell is clear (``cell_not_cleared``) no moving run starts; after it
    only the two ways back, whose first motion is the planned move to the return pose: Home, and the Restart of the
    record's own run (another run is ``not_restartable``; a new task or pick ``restart_required``). A run that moves
    nothing (the planner, a teach, which its route gates) and a console that keeps no record (a program's own) are
    admitted as ever.
    """
    if kind not in MOVING_KINDS:
        return
    record = getattr(console, "recovery", None)
    if record is None:
        return
    run_id = str(getattr(record, "run_id", "") or "")
    stopped = f"the {getattr(record, 'kind', 'moving')} run {run_id} stopped where the arm stands " \
              f"({getattr(record, 'stop_code', '')})"
    if getattr(record, "cleared", False) is not True:
        refused = RunRefused("cell_not_cleared", f"{stopped}: a person confirms the cell is clear first.",
                             run_id=run_id, stop_code=str(getattr(record, "stop_code", "")))
    elif kind is RunKind.HOME:
        return
    elif kind is RunKind.TASK and restart_of is not None:
        if restart_of == run_id:
            return
        refused = RunRefused("not_restartable", (f"run {restart_of} is not the stop to come back from: the console's "
                                                 f"stop record names run {run_id}."), run_id=restart_of)
    else:
        refused = RunRefused("restart_required", (f"{stopped}: the way back is Restart or Home, whose first motion is "
                                                  "the planned move to the return pose; nothing new starts before it."),
                             run_id=run_id, stop_code=str(getattr(record, "stop_code", "")))
    logger.warning("%s run refused under the run lock (%s): %s", kind.capitalize(), refused.code, refused)
    raise refused


def _countdown_because(console: Any) -> str | None:
    """Why the next moving run counts down (``teach``, ``jaws_opened``), or ``None``: what the console says, read so a
    console that keeps no stamps (a program's own, a test's) counts nothing down."""
    because_of = getattr(console, "countdown_because", None)
    if callable(because_of):
        because = because_of()
        return because if because in ("teach", "jaws_opened") else None
    due = getattr(console, "countdown_due", None)
    return "teach" if callable(due) and due() is True else None


def _jaws_question_began(console: Any) -> str:
    """Why a moving run must end before its first motion although its route let it start, or ``""``: a jaws question
    is in progress (``api.jaws.pending``), a check that began after the route read it.

    Read on the run's thread, once the run holds the cell (``active_run_id`` set under the run lock). A check marks
    itself pending before it reads the run lock, and the run reads the question after it set the lock, so a run and a
    check never go on together, in either order: the later of the two sees the other and gives way. A console that
    keeps no question (a program's own) has none in progress; one that cannot be read may have one.
    """
    from api import jaws as browser_jaws  # noqa: PLC0415

    try:
        pending = browser_jaws.pending(console)
    except Exception as exc:  # noqa: BLE001 (a question nobody can read may be in progress)
        return (f"whether a jaws question is in progress could not be read ({type(exc).__name__}: {exc}), so the run "
                "ended before its first motion; nothing moved")
    if pending is None:
        return ""
    return ("a jaws question began as the run started (a check of the jaws): answer it in the browser, then start "
            "again; the run ended before its first motion, and nothing moved")


def _give_way_to_the_jaws_question(run: Run, began: str) -> None:
    """End a moving run before its first motion because a jaws question is (or may be) in progress: the sentence of
    :func:`_jaws_question_began`, and beside it the refusal its route would have answered a moment earlier
    (``RunOut.refusal``: ``jaws_question_pending``, as the route answers it, an unreadable question included). The run
    still ends ``cancelled``, with no ``run_error`` and no recovery record: nothing moved."""
    run.error = began
    run.refusal = {"code": str(RefusalCode.JAWS_QUESTION_PENDING),
                   "status": REFUSAL_STATUS[RefusalCode.JAWS_QUESTION_PENDING]}


def _stopped_before_the_first_motion(run: Run) -> str:
    """Why ``run`` may not make its first motion, in words; ``""`` where it may: a Disconnect, a halt, a stop."""
    if run.abandoned:
        return "the cell was disconnected"
    if run.halt_requested:
        return "halt now was pressed"
    if run.stop_requested:
        return "the operator stopped the run"
    return ""


class _CellFault(Exception):
    """A fault of the cell a pick reported (``report.fault``), raised by the pick run's loop so it ends ``cell_fault``."""

    def __init__(self, fault: BaseException) -> None:
        super().__init__(str(fault))
        self.fault = fault


def _latch_of(service: Any) -> str:
    """Why an earlier pick of ``service`` stopped where the arm stands, a person to decide; ``""`` where none waits.
    Strict, as ``PickRun`` reads it: only a non-empty string, so a double that answers every attribute waits for no
    one."""
    said = getattr(service, "stopped_where_the_arm_stands", "")
    return said if isinstance(said, str) else ""


def _arm_of(service: Any) -> Any:
    return getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "arm", None)


def _halted_sentence(arm: Any, what: str) -> str:
    """The sentence a run halted by "halt now" ends with: the console halted the arm, why, and what became of it."""
    from api.readiness import halt_reason  # noqa: PLC0415

    reason = halt_reason(arm) or "halt now was pressed"
    return (f"the console halted the arm ({reason}): {what}, and nothing more is commanded, the jaws included; a "
            "person confirms the cell is clear, then Restart or Home")


def _stop_of(report: Any, arm: Any, *, halt_requested: bool = False) -> tuple[StopCode | None, str]:
    """Why a pick's report ends a pick run, typed and said (``_why_the_run_stops``'s sentences); ``(None, "")`` where
    the next pick may start.

    A pick that ended on a halt is the halt's, read first, as the library's task reads it: where "halt now" was pressed
    during the run (``halt_requested``) or the arm's latch is set, the run ends ``halted`` in its own sentence, whatever
    the report says it met (a controller that refused a latched move, a hand that refused to switch on the latch, a
    recovery the latch stopped). Their remedies, the pendant, the hand, the recovery, are not the halt's."""
    said = _why_the_run_stops(report)
    if not said:
        return None, ""
    from api.readiness import halt_reason  # noqa: PLC0415

    if halt_requested or halt_reason(arm):
        return StopCode.HALTED, _halted_sentence(arm, f"the pick stopped ({report.failure_summary()})")
    if getattr(report, "controller_stopped", False) is True:
        return StopCode.CONTROLLER_STOPPED, said
    hand = getattr(report, "gripper_fault", "")
    if isinstance(hand, str) and hand:
        return StopCode.HAND_NEEDS_PERSON, said
    return StopCode.RECOVERY_NEEDS_PERSON, said


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if number == number and abs(number) != float("inf") else None


def pick_fields(report: Any) -> dict[str, Any]:
    """What a ``pick_result`` adds beyond its pinned keys (build plan 1.4), each only where the report says it, in its
    type: the hold, the stops, the grasp and the part's middle, both faces, the pushes, the fused views, whether it
    found nothing or only parts its campaign keeps out, the hand-eye bound, where its looks were kept. A double that
    answers every attribute adds nothing a page could misread."""
    from src.robot.grasping.multiview.association import HAND_EYE_DRIFT_WARN_MM  # noqa: PLC0415

    said: dict[str, Any] = {"hand_eye_warn_mm": float(HAND_EYE_DRIFT_WARN_MM)}
    hold = getattr(report, "hold_measured", None)
    if isinstance(hold, bool):
        said["hold_measured"] = hold
    said["controller_stopped"] = getattr(report, "controller_stopped", False) is True
    fault = getattr(report, "gripper_fault", "")
    said["gripper_fault"] = fault if isinstance(fault, str) else ""
    said["needs_person"] = getattr(report, "needs_person", False) is True
    both = getattr(report, "both_faces", None)
    if isinstance(both, bool):
        said["both_faces"] = both
    pose = getattr(report, "grasp_pose", None)
    position = getattr(pose, "position_mm", None)
    quaternion = getattr(pose, "quaternion_xyzw", None)
    try:
        if position is not None and quaternion is not None:
            said["grasp_pose"] = {"position_mm": [round(float(v), 2) for v in position],
                                  "quaternion_xyzw": [round(float(v), 6) for v in quaternion]}
    except (TypeError, ValueError):
        pass
    centre = getattr(report, "object_centre_mm", None)
    if isinstance(centre, tuple) and len(centre) == 3 and all(_number(v) is not None for v in centre):
        said["object_centre_mm"] = [round(float(v), 2) for v in centre]
    fused = getattr(report, "fused_views", ())
    if isinstance(fused, tuple) and all(isinstance(view, str) for view in fused):
        said["fused_views"] = list(fused)
    telemetry = getattr(report, "telemetry", None)
    if isinstance(telemetry, Mapping):
        pushes = telemetry.get("pushes")
        if isinstance(pushes, (list, tuple)):
            said["pushes"] = len(pushes)
        stopped = telemetry.get("push_stopped")
        said["push_stopped"] = isinstance(stopped, str) and bool(stopped)
        views = telemetry.get("views_file")
        if isinstance(views, str) and views:
            said["views_file"] = views
    from src.robot.execution.autonomous_grasp.report import AutonomousGraspReport  # noqa: PLC0415

    if isinstance(report, AutonomousGraspReport):
        from src.robot.execution.autonomous_grasp.service import found_nothing, only_excluded  # noqa: PLC0415

        said["found_nothing"] = bool(found_nothing(report))
        said["only_excluded"] = bool(only_excluded(report))
    return said


def _look_for(service: Any) -> Any:
    """What every pick of a console run looks from: the cell profile's looks, else home on a wrist camera, else
    ``None``, no look at all.

    The rule ``PickRun._looks_for`` keeps for a campaign that names no look of its own
    (``src/robot/execution/pick_run.py``), and the console names none: the looks the cell profile
    configures (``robot.look_joint_positions_deg``, read as ``service.configured_looks``), a fixed
    camera's arm sent to them too; else, as a wrist camera sees what the arm points it at, the arm's configured home
    (``src.robot.execution.looks.HOME``); else a fixed camera perceives from where the arm stands.
    Type-checked (``configured_looks_of``) and ``is True``, so a double that answers every attribute is
    read as neither looks nor a wrist camera, and a service that does not say is asked as it always was.

    Imported here, not at the top: the console imports this module to start, and the look vocabulary
    brings the motion layer with it.
    """
    from src.robot.execution.looks import HOME  # noqa: PLC0415
    from src.robot.execution.pick_run import configured_looks_of  # noqa: PLC0415

    configured = configured_looks_of(service)
    if configured:
        return configured
    return HOME if getattr(service, "perceives_from_the_wrist", False) is True else None


def _names(value: Any) -> list[str]:
    """A report's tuple of look names as the wire carries it; a report that does not say names none."""
    return list(value) if isinstance(value, tuple) and all(isinstance(name, str) for name in value) else []


def _what_the_looks_came_to(report: Any) -> dict[str, Any]:
    """The keys a ``pick_result`` adds for a wrist pick's looks, each only where the report says it, in its type.

    ``jaw_faces_seen`` (jaw 1, jaw 2: whether each contact face of the chosen grasp was seen), ``generated_view_deg``
    (how far the one view the looks generated turned about the part), ``hand_eye_gap_mm`` (how far apart the looks
    measure the part's shared surface: the hand-eye check) and ``refused_look`` (the look, generated view or move back
    whose motion ended the pick). Additive: a report that says none of them, a fixed camera's among them, adds no key,
    and a double that answers every attribute adds nothing.
    """
    said: dict[str, Any] = {}
    faces = getattr(report, "jaw_faces_seen", None)
    if isinstance(faces, tuple) and len(faces) == 2 and all(isinstance(seen, bool) for seen in faces):
        said["jaw_faces_seen"] = list(faces)
    for key in ("generated_view_deg", "hand_eye_gap_mm"):
        value = getattr(report, key, None)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            said[key] = float(value)
    telemetry = getattr(report, "telemetry", None)
    refused = telemetry.get("refused_look") if isinstance(telemetry, Mapping) else None
    if isinstance(refused, str) and refused:
        said["refused_look"] = refused
    return said


def _why_the_run_stops(report: Any) -> str:
    """Why a pick's report ends the run although it raised nothing; ``""`` where the next pick may start.

    The rule ``PickRun`` keeps for a campaign (``src/robot/execution/pick_run.py``), in its words, so
    the console and a program stop on the same reports and say them alike. A fault of the cell is not
    here: the run loop raises it. Three more things a pick reports rather than raises:

    * a controller that cannot move (``controller_stopped``): a protective or emergency stop, or a
      power-off. The pick ends CANCELLED with no fault, and the next pick would perceive, and on a
      toggle switch the jaws, for an arm a person has to walk up to;
    * a hand that needs a person (``gripper_fault``): a gripper that raised, whose jaws nobody can now
      name, a toggle that would not start a pick on jaws it believes closed with nobody at a
      terminal, a hand a push found nobody can vouch for (a toggle's count, a gripper that measures
      its width not connected or unreadable), or a toggle's count found so before ``next_target``
      drove the looks again. The pick ends EXECUTION_FAILED with no fault, and the next one meets the
      same hand;
    * a recovery that stopped where the arm stands (``needs_person``): a push of a failed part that
      stopped once something may have moved, or whose move back to the look failed. The next pick would
      drive the arm back to its look from wherever the push left it, the escape the owner ruled out.

    ``is True`` and a non-empty string, as ``PickRun`` reads them, so a double that answers every
    attribute stops nothing.
    """
    if getattr(report, "controller_stopped", False) is True:
        return (
            "the controller cannot move (a protective or emergency stop, or a power-off), so the run "
            "stops; a person clears the stop where the arm is visible: "
            + str(report.failure_summary())
        )
    hand = getattr(report, "gripper_fault", "")
    if isinstance(hand, str) and hand:
        return f"the gripper needs a person, so the run stops: {hand}"
    if getattr(report, "needs_person", False) is True:
        return (
            "a recovery stopped where the arm stands, so the run stops; nothing more is commanded and a person "
            "decides what happens next: " + str(report.failure_summary())
        )
    return ""


#: How a console operator gets a toggle's count back to OPEN: the one question a person answers about the jaws is the
#: one a Connect asks, at program start, before anything moves (the owner's rule), and it offers the change that opens
#: them. The console has no release of its own to offer.
_HOW_THE_JAWS_COME_BACK = (
    "The program's count says OPEN again only once a person has said so: Disconnect and Connect the cell before that "
    "run; the connect asks where the jaws stand, and 'closed' offers one change to open them. A run started from the "
    "console never asks at the server's terminal, where nobody watching the console sees the question and Stop "
    "cannot reach it."
)


def _why_no_pick_starts(service: Any) -> str:
    """Why the next pick of a console run must not start on this cell's hand; ``""`` where it may.

    Only a hand that toggles with no sensor is asked (``toggle_without_sensor_of``, which reads
    ``toggles_without_sensor is True``, so a double that answers every attribute is not taken for one), the hand
    ``pick()`` itself asks: the orchestrator's gripper. Two beliefs stop the run before ``pick()`` is called, because in
    both ``pick()`` would ask a person where the jaws stand (``jaws_open_for_a_pick``), and from a run's thread that
    question goes to the server's terminal:

    * the program believes the jaws stand CLOSED: its last command closed them, on the part the last pick lifted,
      which a console run never sets down;
    * nobody can say where the jaws stand: the last change of the output failed, or somebody switched the output by
      hand since the program's last command (``why_jaws_unknown``, which reads the output once, without asking or
      sending anything).

    Fail closed: anything but a plain ``False`` from ``jaws_closed``, and anything but a plain ``False`` from
    ``edge_unknown`` (``TogglesWithoutSensor``), stops the run, so neither is ever asked about from this thread.
    The pick itself asks nobody either (:func:`_asking_nobody`).
    """
    toggle = _toggle_of(service)
    if toggle is None:
        return ""
    reader = getattr(toggle, "why_jaws_unknown", None)
    why = reader() if callable(reader) else ""
    why = why if isinstance(why, str) else ""
    if why or getattr(toggle, "edge_unknown", True) is not False:
        return (
            "the gripper needs a person, so the run stops: nobody can say where they stand (a toggle hand with no "
            f"sensor): {why or 'its last change failed, so nobody can say which way it moved them'}. Look at them, "
            f"release or place any part they hold, then start a new run. {_HOW_THE_JAWS_COME_BACK}"
        )
    if toggle.jaws_closed is not False:
        return (
            "the gripper needs a person, so the run stops: the jaws still hold the last part (a toggle hand with no "
            "sensor): release or place it, then start a new run. The program believes they stand CLOSED, and a "
            f"console run never sets its part down. {_HOW_THE_JAWS_COME_BACK}"
        )
    return ""


def _toggle_of(service: Any) -> Any:
    """The hand ``pick()`` asks where it toggles with no sensor (``toggle_without_sensor_of``), else ``None``.

    Imported here, not at the top, as :func:`_look_for` imports: the console imports this module to start.
    """
    from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    return toggle_without_sensor_of(getattr(orchestrator, "gripper", None))


#: What a question the hand would ask inside a run's pick answers instead: nobody is asked from a run's thread.
_NOBODY_ASKED_FROM_A_RUN = (
    "a run started from the console never asks at the server's terminal; the pick stops here, and the run with it"
)


def _asking_nobody(service: Any) -> Any:
    """A block in which the hand ``pick()`` asks asks nobody on this thread (``asking_nobody``); a no-op otherwise."""
    toggle = _toggle_of(service)
    asking = getattr(toggle, "asking_nobody", None) if toggle is not None else None
    return asking(_NOBODY_ASKED_FROM_A_RUN) if callable(asking) else nullcontext()


def _payload(event: "PickProgress") -> dict[str, Any]:
    """The machine half of the envelope: every populated field, and none of the empty ones."""
    fields = {
        "attempt": event.attempt,
        "attempt_total": event.attempt_total,
        "segmentation_count": event.segmentation_count,
        "candidate_count": event.candidate_count,
        "target_index": event.target_index,
        "score": event.score,
        "position_mm": list(event.position_mm) if event.position_mm else None,
        "reasons": list(event.reasons) if event.reasons else None,
        "action": event.action,
        "outcome": event.outcome,
        "route": event.route,
        "route_reason": event.route_reason,
        "motion_status": event.motion_status,
        "motion_message": event.motion_message,
        "motion_error": event.motion_error,
        "camera_world": event.camera_world,
        "camera_world_reason": event.camera_world_reason,
    }
    payload = {k: v for k, v in fields.items() if v is not None}
    # Stage-specific extras (`labels_seen`, ...) belong in the machine half too. Without this they
    # would exist only inside the rendered sentence, which is the one place a program cannot read
    # them: exactly the drift the two-audience envelope exists to prevent.
    for key, value in (event.extra or {}).items():
        payload.setdefault(key, value)
    return payload


class RunConflict(RuntimeError):
    """Raised when a second run is requested while one is active."""

    def __init__(self, existing: Run) -> None:
        super().__init__(
            f"run {existing.id} is already active ({existing.succeeded}/{existing.attempted} so far). "
            f"One arm, one run: a second would not queue, it would collide. Stop it first."
        )
        self.existing = existing


class RunRefused(RuntimeError):
    """Raised where the console's recovery record does not let a moving run start, read under the run lock.

    ``code`` is the refusal the route answers (``cell_not_cleared``, ``restart_required`` or ``not_restartable``, each a
    409 of ``api.codes.RefusalCode``), ``detail`` its envelope's detail; the message is the sentence.
    """

    def __init__(self, code: str, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail
