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
    #: The run raised. The cell may need attention.
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
    #: Set when the run raised.
    error: str = ""
    #: Set when an operator asked the run to stop.
    stop_requested: bool = False

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

    def _drive(self, console: Any, run: Run) -> None:
        """The run body. Everything it can raise is caught: a dead thread must still close its stream."""
        service = console.session.service
        hub = self.hub

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
            # Always set it, including to None. `set_target_label` documents "Pass None to clear",
            # and setting it only when a prompt is present leaves the previous run's label on the
            # shared service: every later run on that cell keeps hunting the earlier object, however
            # it was prompted. The console keeps one cell for the whole process, so "later" means
            # "for the rest of the session".
            service.set_target_label(run.prompt or None)

            for _ in range(run.requested_picks):
                if run.stop_requested:
                    break
                report = service.pick()
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
            run.state = RunState.CANCELLED if run.stop_requested else RunState.FINISHED
        except BaseException as exc:  # noqa: BLE001 (the stream must close whatever happened)
            run.state = RunState.FAILED
            run.error = f"{type(exc).__name__}: {exc}"
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
