"""A run: one operator instruction, executed on a thread that outlives the request that started it.

``pick()`` blocks for as long as the arm takes. A request handler that waited for it would hold the
HTTP connection open across a physical motion, and the browser, a proxy or a sleeping laptop would
time out somewhere in the middle. So starting a run returns immediately with an id, the work happens
on its own thread, and everything the operator sees arrives over the event stream.

The run finishes whether or not anyone is watching. The arm is holding a part, so the correct
behaviour when a tab closes is to complete the pick and put the object down, not to abandon a motion
halfway. The event history is what makes that survivable for the UI.

Stop is not kill. ``stop()`` sets a flag the pick loop reads between attempts. It never interrupts a
motion in flight: this process has no safe way to do that, and the thing that does is the red button
on the wall.

A run ends early where a campaign does (``PickRun``, ``src/robot/execution/pick_run.py``): on a fault
of the cell, and on the two things a pick reports rather than raises, a controller that cannot move and
a hand that needs a person. Each ends it FAILED with the reason in ``error``, because the next pick
would meet the same stopped arm or the same hand. And each pick looks from where a campaign's does:
from home on a wrist camera, from where the arm stands on a fixed one.

Two stops a campaign does not need, because a campaign runs in a terminal and a run does not. A run
never asks a person anything: on a hand that toggles with no sensor, a pick that starts on jaws the
program believes closed asks where they stand, and from a run's thread that question goes to the
server's terminal, which nobody watching the console sees and Stop cannot reach. A console run sets
none of its parts down, so the run ends FAILED before such a pick instead (``_why_no_pick_starts``).
And a Disconnect stops the run before the cell comes down (:meth:`RunRegistry.abandon`); the run keeps
the console's run lock until its thread ends, and Connect and Build refuse while it does.
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from api.constants import API_LOG_DIR, RUNS_LOG_FILE
from api.events import EventHub, Severity
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.robot.grasping.loop.progress import PickProgress

__all__ = ["RETAINED_RUNS", "Run", "RunRegistry", "RunState"]

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
    if event.motion_message or event.motion_error:
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
        sentence = (
            f"Nothing in this frame is called {event.extra.get('target_label')!r}. "
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
        }


class RunRegistry:
    """Starts runs, keeps their history, and is the only thing that flips the console's run lock."""

    def __init__(self, hub: EventHub) -> None:
        self.hub = hub
        self._lock = threading.RLock()
        self._runs: dict[str, Run] = {}
        self._order: list[str] = []
        self._threads: dict[str, threading.Thread] = {}

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

    # --- driving ------------------------------------------------------------------------------------

    def start(self, console: Any, *, prompt: str, picks: int) -> Run:
        """Begin a run on its own thread and return immediately.

        Refuses if one is already running: a second concurrent pick would drive the same arm from two
        threads, which is not a queue but a collision.
        """
        with self._lock:
            if (existing := self.active()) is not None:
                logger.warning(
                    "Run refused: %s is still active (%d/%d done).",
                    existing.id, existing.succeeded, existing.attempted,
                )
                raise RunConflict(existing)
            run = Run(id=f"run-{uuid.uuid4().hex[:10]}", prompt=prompt, requested_picks=picks)
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
            reason=because,
        )
        return run

    def _drive(self, console: Any, run: Run) -> None:
        """The run body. Everything it can raise is caught: a dead thread must still close its stream."""
        service = console.session.service
        hub = self.hub
        # What the run's prompt replaced, put back in the `finally`. `None` when the run named none.
        previous_prompt: Any = None

        def _on_progress(event: "PickProgress") -> None:
            severity, human = _sentence(event)
            hub.publish(
                run.id, f"pick.{event.stage}", severity=severity, human=human,
                step=str(event.stage), step_index=event.attempt, step_total=event.attempt_total,
                **{k: v for k, v in _payload(event).items()},
            )

        try:
            hub.publish(
                run.id, "run_started", severity=Severity.INFO,
                human=f"Run started: {run.requested_picks} pick(s), prompt {run.prompt!r}.",
                prompt=run.prompt, requested_picks=run.requested_picks,
            )
            service.attach_progress_listener(_on_progress)
            service.set_cancel_check(lambda: run.stop_requested)
            # The prompt reaches the detector, for this run only. A label filter alone is not enough:
            # every frame would still be grounded with the build phrase ("object"), and "the red
            # cube", typed or spoken, would filter on a label no phrase grounder returns. `set_prompt`
            # sets the phrase, the labels the detector's words map onto and the filter together, and
            # the `finally` below puts all three back: the console keeps one cell for the whole
            # process, and a prompt that outlived its run would have every later run hunting it. An
            # empty prompt keeps the build phrase and filters nothing.
            if run.prompt.strip():
                previous_prompt = service.set_prompt(run.prompt)

            # Asked once, as `PickRun` asks it: whether a camera sits on the wrist does not change
            # between two picks. A wrist camera sees what the arm points it at, so each pick looks from
            # the arm's home first; before this the console handed no look, and a wrist camera
            # perceived from wherever the last pick had left the arm. A service handed no look is
            # called as it always was, so a fixed camera, and a service that takes none, still runs.
            look = _look_for(service)

            # Why the run ended before its last pick although nothing raised; "" while it has not.
            stopped_by = ""
            for _ in range(run.requested_picks):
                if run.stop_requested:
                    break
                # Before every pick, the first included: a toggle hand the program believes CLOSED, or whose last
                # pulse nobody can place, would have `pick()` ask a person where the jaws stand. From this thread that
                # question goes to the SERVER's terminal, where nobody watching the console sees it, Stop cannot
                # reach `input()`, and a 'p' typed later drops the part wherever the arm then stands. A console run
                # never releases what it lifted, so on a toggle every pick after a success would ask. The run ends
                # here instead, as the other stops end it.
                stopped_by = _why_no_pick_starts(service)
                if stopped_by:
                    break
                report = service.pick() if look is None else service.pick(look=look)
                fault = getattr(report, "fault", None)
                if fault is not None:
                    # A fault of the cell still ends the run: the next pick would meet the same dead
                    # camera or the same dropped link. `pick()` reports it, so the run raises it here.
                    raise fault
                run.attempted += 1
                outcome = str(getattr(report, "outcome", "unknown"))
                run.outcomes.append(outcome)
                ok = bool(getattr(report, "succeeded", False))
                run.succeeded += int(ok)
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
                )
                # After the pick is counted and said: it ran, and its outcome is part of the run's
                # record. What ends the run is that the next pick must not start.
                stopped_by = _why_the_run_stops(report)
                if stopped_by:
                    break
            if run.abandoned:
                # The cell went down with the run in it. That is the reason the operator must read first; what the
                # last pick said, where it said anything, follows it.
                stopped_by = (f"{run.abandoned} The last pick also said: {stopped_by}" if stopped_by
                              else run.abandoned)
            if stopped_by:
                # FAILED, not CANCELLED: nobody asked for this stop, the cell needs a person, and a run
                # that read as finished or cancelled would send the operator to the next prompt instead
                # of to the arm. Whatever a stop request said, this is what the operator must read.
                run.state = RunState.FAILED
                run.error = stopped_by
                logger.error(
                    "Run %s FAILED after pick %d/%d: %s",
                    run.id, run.attempted, run.requested_picks, stopped_by,
                )
                hub.publish(
                    run.id, "run_error", severity=Severity.ERROR,
                    human=f"The run stopped: {stopped_by}", error=stopped_by,
                )
            else:
                run.state = RunState.CANCELLED if run.stop_requested else RunState.FINISHED
        except BaseException as exc:  # noqa: BLE001 (the stream must close whatever happened)
            run.state = RunState.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
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
                human=f"The run stopped with an error: {run.error}", error=run.error,
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
            with self._lock:
                console.active_run_id = None
                # The Thread object is useless the moment its run ends and nothing reads this dict
                # again, so it goes now rather than at the retention boundary. It was never popped
                # at all, which is how a dict with no reader grew one entry per run for the life of
                # the process.
                self._threads.pop(run.id, None)
            logger.info(
                "Run %s %s: %d/%d succeeded in %.1f s.",
                run.id, run.state, run.succeeded, run.attempted,
                (run.finished_at or time.time()) - run.started_at,
            )
            hub.publish(
                run.id, "run_finished",
                severity=Severity.SUCCESS if run.state is RunState.FINISHED else Severity.WARN,
                human=(
                    f"Run {run.state}: {run.succeeded}/{run.attempted} succeeded."
                ),
                **run.to_dict(),
            )


def _look_for(service: Any) -> Any:
    """What every pick of a console run looks from: home on a wrist camera, else ``None``, no look at all.

    The rule ``PickRun._looks_for`` keeps for a campaign that names no look of its own
    (``src/robot/execution/pick_run.py``), and the console names none: a wrist camera sees what the
    arm points it at, so it looks from the arm's configured home (``src.robot.execution.looks.HOME``);
    a fixed camera perceives from where the arm stands. ``is True``, so a double that answers every
    attribute is not read as a wrist camera, and a service that does not say is asked as it always was.

    Imported here, not at the top: the console imports this module to start, and the look vocabulary
    brings the motion layer with it.
    """
    from src.robot.execution.looks import HOME  # noqa: PLC0415

    return HOME if getattr(service, "perceives_from_the_wrist", False) is True else None


def _why_the_run_stops(report: Any) -> str:
    """Why a pick's report ends the run although it raised nothing; ``""`` where the next pick may start.

    The rule ``PickRun`` keeps for a campaign (``src/robot/execution/pick_run.py``), in its words, so
    the console and a program stop on the same reports and say them alike. A fault of the cell is not
    here: the run loop raises it. Two more things a pick reports rather than raises:

    * a controller that cannot move (``controller_stopped``): a protective or emergency stop, or a
      power-off. The pick ends CANCELLED with no fault, and the next pick would perceive, and on a
      toggle pulse the jaws, for an arm a person has to walk up to;
    * a hand that needs a person (``gripper_fault``): a gripper that raised, whose jaws nobody can now
      name, or a toggle that would not start a pick on jaws it believes closed with nobody at a
      terminal. The pick ends EXECUTION_FAILED with no fault, and the next one meets the same hand.

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
    return ""


#: How a console operator gets a toggle's count back to OPEN: the one question a person answers about the jaws is the
#: one a Connect asks, at program start, before anything moves (the owner's rule), and it offers the pulse that opens
#: them. The console has no release of its own to offer.
_HOW_THE_JAWS_COME_BACK = (
    "The program's count says OPEN again only once a person has said so: Disconnect and Connect the cell before that "
    "run; the connect asks where the jaws stand, and 'closed' offers one pulse to open them. A run started from the "
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
    * the last pulse failed on its high write, so nobody can say which way it moved them.

    Fail closed: anything but a plain ``False`` from ``jaws_closed``, and anything but a plain ``False`` from
    ``edge_unknown`` (``TogglesWithoutSensor``), stops the run, so neither is ever asked about from this thread.

    Imported here, not at the top, as :func:`_look_for` imports: the console imports this module to start.
    """
    from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    toggle = toggle_without_sensor_of(getattr(orchestrator, "gripper", None))
    if toggle is None:
        return ""
    if getattr(toggle, "edge_unknown", True) is not False:
        return (
            "the gripper needs a person, so the run stops: the last pulse on the jaws (a toggle hand with no sensor) "
            "failed on its high write, so nobody can say where they stand. Look at them, release or place any part "
            f"they hold, then start a new run. {_HOW_THE_JAWS_COME_BACK}"
        )
    if toggle.jaws_closed is not False:
        return (
            "the gripper needs a person, so the run stops: the jaws still hold the last part (a toggle hand with no "
            "sensor): release or place it, then start a new run. The program believes they stand CLOSED, and a "
            f"console run never sets its part down. {_HOW_THE_JAWS_COME_BACK}"
        )
    return ""


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
