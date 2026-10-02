"""Teaching one pose by hand from the browser: the arm is freed, a person guides it, the pose is screened and saved.

Build plan 1.8, the owner's decision 15 and answers Q10, Q14 and Q15. One pose per session, on its own run (kind
``teach``, which moves nothing by itself), driven by the library's ``teach_one`` (``src/robot/execution/teach.py``):
everything that would make the pose unwritable is refused before the arm is freed, the pose is captured once the arm
stands still, held, screened at once by the exact guard and the planner, and written into the cell's own layer only
where both clear it or it lies in the planner's band. This module is the browser's half:

* :func:`start` refuses a teach in the plan's order before it starts the run: the cell's state (not connected, a run, a
  halt, a stopped controller, a stop not cleared, a jaws question), a part in the hand and a toggle count nobody can
  vouch for, the library's own refusals (``teach_refusal``: no hand guiding, a planner that is not ready, an arm that
  screens nothing, a name that is none), the pose store's (a label that is none, a name taken, a chain with no layer of
  the cell's own), and a controller payload that is not the one the person confirmed (one that is no number never is);
* a jaws check and a teach never go on together: the run reads the jaws question again before it frees anything, so a
  check that began after the route's own read ends the teach with nothing freed (a check marks itself before it reads
  the run lock, so it is refused by a teach that started first);
* the session's console and view (:class:`BrowserConsole`, :class:`BrowserView`) are what ``HandGuide`` polls: every
  key the browser sends is emitted exactly once, Save only once the arm is sampled (a line sent before the wait would be
  forgotten by it), and a poll never answers end-of-input;
* **every poll is the heartbeat.** Without one for :data:`TEACH_HEARTBEAT_S`, or at :data:`TEACH_TIME_LIMIT_S` of free
  time (announced :data:`TEACH_WARNING_S` before), the session asks for a hold, and the arm is held only once it stands
  still (``StillnessGate``), never while it moves in a person's hands (Q14). Save, Hold, Cancel, closing the tab (its
  beacon), a halt (the run's flag, or the arm's own latch however it was set), a stop request and a Disconnect hold at
  once (the view's ``finish``). A Disconnect holds the arm before it takes it down: the session asks every open teach
  to hold first (``CellSession.teaches``), whichever way the Disconnect came;
* the RTDE watchdog of the freedrive session still stops a server that died (``src/robot/drivers/ur/freedrive.py``);
* the console's microphone is the browser's to switch off during a teach (Q15); a sentence typed meanwhile is refused,
  because ``POST /v1/commands/parse`` refuses during any run.

An ERROR verdict and an UNSCREENED one are never written; CLEAR and BAND are, through the pose door alone. Once a
session has freed the arm, the console is stamped (``Console.stamp_teach_ended``): a person's hands were on the arm, so
the next moving run counts down first.

The poses themselves are read and chosen here too (:func:`poses`, :func:`choose_default_place`): Home from the config,
read-only; the named poses with their labels and screens; the default place, written into the cell's own layer through
the pose door; and the file a taught pose goes to, said before anybody frees the arm (Q10).
"""

from __future__ import annotations

import math
import secrets
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

from api.codes import REFUSAL_STATUS, RefusalCode, RunKind, StopCode
from api.constants import API_LOG_DIR
from api.events import Severity
from api.schemas import PayloadOut, PoseOut, PosesOut, TeachIn, TeachStateOut
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from api.cell import Console
    from api.runs import Run

__all__ = [
    "TEACH_HEARTBEAT_S",
    "TEACH_HOLD_WAIT_S",
    "TEACH_LOG_FILE",
    "TEACH_TIME_LIMIT_S",
    "TEACH_WARNING_S",
    "BrowserConsole",
    "BrowserView",
    "TeachSessionRefused",
    "cancel",
    "capture",
    "choose_default_place",
    "drive_teach",
    "end_and_join",
    "hold_open_teaches",
    "payload",
    "poses",
    "refusal_before_a_teach",
    "snapshot",
    "start",
]

#: Without a poll for this long the session holds the arm, once it stands still.
TEACH_HEARTBEAT_S = 3.0
#: The longest a person may guide the arm in one session.
TEACH_TIME_LIMIT_S = 300.0
#: The time warning comes this long before the limit.
TEACH_WARNING_S = 30.0
#: How long a Disconnect waits for an open teach to let go of the arm (held at once), before it takes the arm down
#: anyway (build plan item 18: at most 10 s). A hold at once comes within one sample, a fiftieth of a second.
TEACH_HOLD_WAIT_S = 10.0
#: This module's log, beside the console's others under ``logs/api``.
TEACH_LOG_FILE: Final[str] = "teach.log"

logger = create_logger("BrowserTeach", TEACH_LOG_FILE, log_dir=API_LOG_DIR)

#: How many sessions are kept for their last poll once they ended (a browser reads the verdict after the run ended).
_KEPT_SESSIONS = 16
#: How many of the session's said lines a poll carries.
_LINES_KEPT = 40
#: The states a session ends in.
_ENDED: Final[frozenset[str]] = frozenset({"saved", "refused", "not_saved", "ended"})
#: How close a payload must stay to the one the person confirmed: a hundredth of a kilogram, a millimetre.
_MASS_TOLERANCE_KG = 0.01
_COG_TOLERANCE_MM = 1.0


class TeachSessionRefused(Exception):
    """A teach route refused, with its code from the catalog (``api.codes``) and one sentence; answered at
    ``REFUSAL_STATUS[code]`` in the one envelope."""

    def __init__(self, code: RefusalCode, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail or {})

    @property
    def status(self) -> int:
        return REFUSAL_STATUS[self.code]


@dataclass(frozen=True, slots=True)
class _Spec:
    """What a teach was asked for, resolved before its run starts: the pose, and the store it is written to."""

    name: str
    label: str
    role: str
    make_default_place: bool
    replace: bool
    store: Any
    target: str


class BrowserConsole:
    """The console ``HandGuide`` polls during a teach: what the session says goes to the browser (``teach.say``), and
    Save is the line ``""``, emitted exactly once and only once the free arm is sampled. A poll never answers
    end-of-input: the browser going quiet is the heartbeat's to judge, never a hold of its own."""

    def __init__(self, session: "_TeachSession") -> None:
        self._session = session

    def say(self, line: str) -> None:
        self._session.said(line)

    def bell(self) -> None:
        return None

    def poll(self) -> str | None:
        return self._session.console_line()


class BrowserView:
    """The view ``HandGuide`` polls during a teach: the keys that hold at once (Hold, Cancel, the tab's beacon, a halt,
    a Disconnect), each emitted as the view's ``finish``, exactly once; the guide's own lines are kept for the poll."""

    def __init__(self, session: "_TeachSession") -> None:
        self._session = session

    def guide(self, lines: Any, tone: str, bar: float | None = None) -> None:
        self._session.guided(list(lines), tone)

    def poll_key(self) -> str | None:
        return self._session.view_key()


class _UnlessHeldAtOnce:
    """The pose store, as the session hands it to ``teach_one``: a pose captured while a hold-at-once came in (a Cancel
    sent while the pose was screened, a halt, a Disconnect) is not written. Cancel saves nothing."""

    def __init__(self, store: Any, session: "_TeachSession") -> None:
        self._store = store
        self._session = session

    @property
    def target(self) -> str:
        return str(self._store.target)

    def write(self, pose: Any, screen: Any) -> str:
        why = self._session.held_at_once()
        if why:
            return f"the teach was ended before the pose was written ({why})"
        return str(self._store.write(pose, screen))


class _TeachSession:
    """One teach: the browser's token and heartbeat, what the arm does, and the run that drives it."""

    def __init__(self, console: "Console", spec: _Spec) -> None:
        self.console = console
        self.spec = spec
        self.token = secrets.token_urlsafe(16)
        self.run: Any = None
        self.thread: threading.Thread | None = None
        #: Set once the session let go of the arm: held on its way out (``teach_one`` holds on every way out), or never
        #: freed. A Disconnect waits for this, never for the run's own end, which takes the console's lock.
        self.left = threading.Event()
        #: Set once the run's body returned: the session is over, whatever its run still writes.
        self.done = threading.Event()
        self._lock = threading.Lock()
        self._state = "freeing"
        self._lines: deque[str] = deque(maxlen=_LINES_KEPT)
        self._guide: list[str] = []
        self._joints_deg: list[float] | None = None
        self._tcp_mm: list[float] | None = None
        self._outside = ""
        self._verdict: str | None = None
        self._nearby_deg: list[float] | None = None
        self._message = ""
        self._contact = time.monotonic()
        self._freed_at: float | None = None
        self._freed = False
        self._sampled = False
        self._hold = ""
        self._hold_said = False
        self._warned = False
        self._save = False
        self._finish = ""
        self._finish_sent = False
        self._limits: Any = None

    # --- the browser's side ---------------------------------------------------------------------------------------

    def allows(self, token: str) -> None:
        """Refuse a token that is not this session's (``wrong_token``); a matching one is the heartbeat."""
        if not (isinstance(token, str) and secrets.compare_digest(token, self.token)):
            raise TeachSessionRefused(RefusalCode.WRONG_TOKEN, (
                "that token is not this teach session's: only the browser that started it polls, saves and cancels "
                "it"))
        with self._lock:
            self._contact = time.monotonic()

    def state_out(self) -> TeachStateOut:
        with self._lock:
            return TeachStateOut(
                state=self._state,  # type: ignore[arg-type]
                outside=bool(self._outside), lines=list(self._lines), joints_deg=self._joints_deg,
                tcp_mm=self._tcp_mm, verdict=self._verdict,  # type: ignore[arg-type]
                nearby_deg=self._nearby_deg, message=self._message, time_left_s=self._time_left(),
            )

    def capture(self) -> TeachStateOut:
        """Save: captured by the session once the arm stands still. Refused (``not_free``) where the arm is not free."""
        with self._lock:
            if self._state != "free":
                raise TeachSessionRefused(RefusalCode.NOT_FREE, (
                    f"the arm is not free to be saved: the session is {self._state}. Save captures only while a person "
                    "guides the arm"), {"state": self._state})
            self._save = True
        logger.info("Teach %s: Save asked; captured once the arm stands still.", self.spec.name)
        return self.state_out()

    def hold_at_once(self, why: str) -> TeachStateOut:
        """Hold, Cancel, the tab's beacon, a Disconnect: the arm holds at once and nothing is saved. Idempotent."""
        with self._lock:
            if not self._finish and self._state not in _ENDED:
                self._finish = why
                logger.warning("Teach %s: hold at once (%s).", self.spec.name, why)
        return self.state_out()

    def held_at_once(self) -> str:
        """Why the session was asked to hold at once (a person, a halt, a Disconnect), or ``""``."""
        with self._lock:
            finish = self._finish
        return finish or _hold_asked_by(self.run, self._arm())

    def _arm(self) -> Any:
        """The arm the session's cell holds now: its own latch holds the teach at once too."""
        try:
            return self.console.session.arm
        except Exception:  # noqa: BLE001 (a session that cannot name its arm has no latch to read)
            return None

    # --- the session's console and view ---------------------------------------------------------------------------

    def said(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
        self._publish("teach.say", line, data={"line": line})

    def guided(self, lines: list[str], tone: str) -> None:
        with self._lock:
            self._guide = lines

    def console_line(self) -> str | None:
        with self._lock:
            if self._save and self._sampled:
                self._save = False
                return ""
        return None

    def view_key(self) -> str | None:
        # Read outside the session's lock: the arm's latch is the arm's to lock.
        why = "" if self._finish else _hold_asked_by(self.run, self._arm())
        with self._lock:
            if not self._finish and why:
                self._finish = why
            if self._finish and not self._finish_sent:
                self._finish_sent = True
                return "finish"
        return None

    def hold_when_still(self) -> str:
        """What asks the session to hold once the arm stands still: ``heartbeat`` or ``time_limit``; ``""`` for none.
        Once asked it stays asked."""
        with self._lock:
            return self._hold_reason(time.monotonic())

    def watch(self, sample: Any) -> None:
        """Every sample of the free arm: the readout, the boundaries, the time warning and a hold asked for."""
        now = time.monotonic()
        said: list[tuple[str, str, dict[str, Any], Severity]] = []
        with self._lock:
            self._sampled = True
            self._joints_deg = [round(math.degrees(float(v)), 4) for v in sample.joints_rad]
            self._tcp_mm = [round(float(v), 2) for v in sample.tcp_xyz_mm]
            which, why = self._limits.crossed(sample) if self._limits is not None else ("", "")
            if which != self._outside:
                if why:
                    said.append(("teach.outside", f"Outside: {why}.", {"sentence": why}, Severity.WARN))
                elif self._outside:
                    said.append(("teach.inside", "Back inside.", {}, Severity.INFO))
                self._outside = which
            if self._freed_at is not None and not self._warned:
                left = TEACH_TIME_LIMIT_S - (now - self._freed_at)
                if left <= TEACH_WARNING_S:
                    self._warned = True
                    seconds = round(max(0.0, left), 1)
                    said.append(("teach.time_warning", f"{seconds:g} s left to save this pose; then the arm is held "
                                 "once it stands still.", {"seconds_left": seconds}, Severity.WARN))
            reason = self._hold_reason(now)
            if reason and not self._hold_said and self._state == "free":
                self._hold_said = True
                self._state = "holding_when_still"
                said.append(("teach.holding_when_still", _holding_said(reason), {"because": reason}, Severity.WARN))
        for event_type, human, data, severity in said:
            self._publish(event_type, human, data=data, severity=severity)

    def on_step(self, step: str) -> None:
        """``free``, ``holding`` and ``screening`` from ``teach_one``: the state and its event."""
        with self._lock:
            if step == "free":
                self._freed, self._freed_at = True, time.monotonic()
                if self._state == "freeing":
                    self._state = "free"
            elif step in ("holding", "screening"):
                self._state = step
        if step == "free":
            self._publish("teach.free", f"The arm is free: guide it to where {self.spec.name} should be, then save it.",
                          data={"name": self.spec.name, "label": self.spec.label, "time_limit_s": TEACH_TIME_LIMIT_S})
        elif step == "holding":
            self._publish("teach.holding", "The arm holds where it stands.")
        elif step == "screening":
            self._publish("teach.screening", "The pose is screened by the exact guard and the planner.")

    # --- the run --------------------------------------------------------------------------------------------------

    def drive(self, console: "Console", run: "Run") -> StopCode:
        """The run's body: one ``teach_one`` session, its outcome the run's stop code."""
        self.run = run
        self.thread = threading.current_thread()
        # Kept from the run's first moment, so a Disconnect finds it however early it comes (the route keeps it too).
        _keep(run.id, self)
        try:
            return self._drive(console, run)
        finally:
            self.left.set()
            if self._freed:
                console.stamp_teach_ended()
            with self._lock:
                if self._state not in _ENDED:
                    self._state = "ended"
            self.done.set()

    def _drive(self, console: "Console", run: "Run") -> StopCode:
        from api import jaws as browser_jaws  # noqa: PLC0415
        from src.robot.core import RobotError  # noqa: PLC0415
        from src.robot.execution.hand_guiding import HandGuide, HandGuidingLimits, HandGuidingRefused  # noqa: PLC0415
        from src.robot.execution.teach import teach_one  # noqa: PLC0415

        early = self.held_at_once()
        if early:
            return self._ended(run, StopCode.CANCELLED, f"held before the arm was freed ({early}); nothing was taught")
        # Read again now that this run holds the cell: a jaws check that began after the route read the question (it
        # marks itself before it reads the run lock) must never ask, or open the jaws, while a person guides the arm.
        if browser_jaws.pending(console) is not None:
            message = ("a jaws question came up as the teach started (a check of the jaws began meanwhile): answer it "
                       "in the browser, then teach; the arm was not freed")
            self._publish("teach.not_saved", f"The pose was not saved: {message}", data={"message": message},
                          severity=Severity.WARN)
            return self._ended(run, StopCode.TEACH_NOT_SAVED, message, state="not_saved")
        arm = console.session.arm
        app = console.config()
        robot = app.robot
        box = getattr(getattr(arm, "config", None), "workspace_limits", None)
        if box is None and robot is not None:
            box = robot.workspace_limits
        self._limits = HandGuidingLimits.of(arm, box)
        guide = HandGuide(BrowserConsole(self), BrowserView(self))
        # The person confirmed the payload in the browser before the arm was freed (``payload_seen``, checked by
        # :func:`start` against the controller's own): the guide asks nothing at the console.
        guide.payload_confirmed = True
        try:
            taught = teach_one(arm, guide, self._limits, self.spec.name, store=_UnlessHeldAtOnce(self.spec.store, self),
                               tree=app, hold_when_still=self.hold_when_still, watch=self.watch, on_step=self.on_step)
        except (HandGuidingRefused, RobotError) as refused:
            if self.held_at_once():
                return self._ended(run, StopCode.CANCELLED, f"held at once: {refused}")
            message = f"{refused}"
            self._publish("teach.not_saved", f"The pose was not saved: {message}", data={"message": message},
                          severity=Severity.WARN)
            return self._ended(run, StopCode.TEACH_NOT_SAVED, message, state="not_saved")
        return self._taught(run, taught)

    def _taught(self, run: "Run", taught: Any) -> StopCode:
        said = taught.to_dict()
        outcome = taught.outcome
        if outcome == "saved":
            with self._lock:
                self._verdict = said.get("verdict")
                self._message = taught.render()
            self._publish("teach.saved", f"{self.spec.name} ({self.spec.label}) was saved: {said.get('verdict')}.",
                          data={"name": self.spec.name, "label": self.spec.label, "verdict": said.get("verdict")},
                          severity=Severity.SUCCESS)
            return self._ended(run, StopCode.TAUGHT, "", state="saved")
        if outcome == "refused":
            with self._lock:
                self._verdict, self._nearby_deg = said.get("verdict"), said.get("nearby_deg")
            self._publish("teach.refused", f"The pose was refused and not saved: {said.get('detail')}",
                          data={"verdict": said.get("verdict"), "detail": said.get("detail"),
                                "nearby_deg": said.get("nearby_deg")}, severity=Severity.WARN)
            return self._ended(run, StopCode.TEACH_REFUSED, taught.render(), state="refused")
        if outcome == "not_saved" and self.held_at_once():
            return self._ended(run, StopCode.CANCELLED, taught.not_written or taught.render())
        if outcome == "not_saved":
            with self._lock:
                self._verdict = said.get("verdict")
            message = taught.not_written or taught.render()
            self._publish("teach.not_saved", f"The pose was not saved: {message}", data={"message": message},
                          severity=Severity.WARN)
            return self._ended(run, StopCode.TEACH_NOT_SAVED, message, state="not_saved")
        if outcome == "held_when_still":
            code = StopCode.TEACH_TIME_LIMIT if taught.because == "time_limit" else StopCode.HEARTBEAT_LOST
            return self._ended(run, code, taught.render())
        held = self.held_at_once()
        return self._ended(run, StopCode.CANCELLED, f"held at once ({held}): {taught.render()}" if held
                           else taught.render())

    def _ended(self, run: "Run", code: StopCode, message: str, *, state: str = "ended") -> StopCode:
        with self._lock:
            self._state = state
            if message:
                self._message = message
        if message:
            run.error = message
        logger.info("Teach %s ended %s%s", self.spec.name, code, f": {message}" if message else ".")
        return code

    # --- helpers --------------------------------------------------------------------------------------------------

    def _hold_reason(self, now: float) -> str:
        """Under the lock: a lapsed heartbeat or the time limit, kept once found."""
        if not self._hold:
            if now - self._contact > TEACH_HEARTBEAT_S:
                self._hold = "heartbeat"
            elif self._freed_at is not None and now - self._freed_at >= TEACH_TIME_LIMIT_S:
                self._hold = "time_limit"
        return self._hold

    def _time_left(self) -> float | None:
        """Under the lock: the free time left, the whole of it before the arm is freed, none once it holds."""
        if self._state == "freeing":
            return float(TEACH_TIME_LIMIT_S)
        if self._state in ("free", "holding_when_still") and self._freed_at is not None:
            return round(max(0.0, TEACH_TIME_LIMIT_S - (time.monotonic() - self._freed_at)), 1)
        return None

    def _publish(self, event_type: str, human: str, *, data: dict[str, Any] | None = None,
                 severity: Severity = Severity.INFO) -> None:
        run = self.run
        if run is None:
            return
        self.console.hub.publish(run.id, event_type, human=human, severity=severity, data=data or {})


# --- the doors the routes call ----------------------------------------------------------------------------------------


_SESSIONS: dict[str, _TeachSession] = {}
_SESSIONS_LOCK = threading.Lock()


def start(console: "Console", request: TeachIn) -> "tuple[Run, str]":
    """Start one teach session on its own run, or refuse it before anything is freed (:class:`TeachSessionRefused`).

    The refusals, in order: ``not_connected``, ``run_active``, ``halted``, ``controller_stopped``,
    ``cell_not_cleared``, ``jaws_question_pending``, ``part_in_hand``, ``jaws_not_confirmed``; the library's
    (``invalid_name``, ``no_hand_guiding``, ``screen_unavailable``, ``planner_not_ready``); the store's
    (``invalid_label``, ``name_taken``, ``no_layer``); then ``payload_changed``. Answers the run and the token its polls
    carry.
    """
    from api.runs import RunConflict  # noqa: PLC0415
    from src.robot.execution.teach import ProfilePoseStore, TeachRefused  # noqa: PLC0415

    app = console.config()
    refused = refusal_before_a_teach(console, app, name=request.name)
    if refused is not None:
        raise refused
    try:
        store = ProfilePoseStore(_tree(console), request.name, request.label,
                                 make_default_place=request.make_default_place, replace=request.replace)
        target = store.target
    except TeachRefused as refusal:
        raise TeachSessionRefused(_code(refusal.code), str(refusal)) from refusal
    changed = _payload_changed(console.session.arm, request.payload_seen)
    if changed:
        raise TeachSessionRefused(RefusalCode.PAYLOAD_CHANGED, changed)
    session = _TeachSession(console, _Spec(name=request.name, label=request.label, role=request.role,
                                           make_default_place=request.make_default_place, replace=request.replace,
                                           store=store, target=target))
    # A Disconnect asks the session to hold every open teach before it takes the arm down, whichever way it came.
    console.session.teaches = _SessionTeaches(console)
    try:
        run = console.registry.start_kind(
            console, RunKind.TEACH, session.drive,
            inputs={"name": request.name, "label": request.label, "role": request.role,
                    "make_default_place": request.make_default_place, "replace": request.replace, "target": target},
        )
    except RunConflict as conflict:
        raise TeachSessionRefused(RefusalCode.RUN_ACTIVE, str(conflict), {"run_id": conflict.existing.id}) from conflict
    session.run = session.run or run
    _keep(run.id, session)
    logger.info("Teach %s (%s) started on run %s; it is written to %s.", request.name, request.label, run.id, target)
    return run, session.token


def refusal_before_a_teach(console: "Console", app: Any, *, name: str | None = None) -> TeachSessionRefused | None:
    """Why no pose could be taught now, as :func:`start` refuses it before the store and the payload, or ``None``.

    Reads only: the session, the run lock, the arm's latch and controller, the recovery record, the jaws question, the
    hand, and the library's own rule (``teach_refusal``: the arm, its planner, its screen, ``name`` where one is given).
    ``GET /v1/poses`` asks it without a name for ``teachable`` and ``why_not``.

    The hand: a part that may be in it (``part_in_hand``: the task's own rule), and a toggle whose count nobody can vouch
    for (``jaws_not_confirmed``, the jaws question's code: a change whose write failed may have closed the jaws on a
    part, uncompensated by the payload the person confirms, and a place pose taught then would be wrong).
    """
    from api import jaws as browser_jaws  # noqa: PLC0415
    from src.robot.core.gripper import why_toggle_count_unknown  # noqa: PLC0415
    from src.robot.execution.task import _part_may_be_held  # noqa: PLC0415 (one rule with the task's own check)
    from src.robot.execution.teach import teach_refusal  # noqa: PLC0415

    session = console.session
    if not session.connected:
        return TeachSessionRefused(RefusalCode.NOT_CONNECTED, "the cell is not connected: connect it, then teach")
    active = console.active_run_id
    if active is not None:
        return TeachSessionRefused(RefusalCode.RUN_ACTIVE, (
            f"run {active} is active: a pose is taught with the arm to nobody else, once it has ended"),
            {"run_id": active})
    arm, gripper = session.arm, session.gripper
    halted = browser_jaws.halt_refusal(arm, "the arm was not freed")
    if halted:
        return TeachSessionRefused(RefusalCode.HALTED, halted)
    stopped = browser_jaws.controller_stop(arm)
    if stopped:
        return TeachSessionRefused(RefusalCode.CONTROLLER_STOPPED, f"{stopped}; the arm was not freed")
    if console.recovery_gate() == "cell_not_cleared":
        return TeachSessionRefused(RefusalCode.CELL_NOT_CLEARED, (
            "a run stopped where the arm stands, and nobody has confirmed the cell is clear since: confirm it "
            "(POST /v1/cell/acknowledge), then teach"))
    if browser_jaws.pending(console) is not None:
        return TeachSessionRefused(RefusalCode.JAWS_QUESTION_PENDING, (
            "a jaws question is in progress: answer it in the browser first; the arm was not freed"))
    held = _part_may_be_held(arm, gripper)
    if held:
        return TeachSessionRefused(RefusalCode.PART_IN_HAND, (
            f"{held}: a pose is taught with empty jaws (a place pose puts the fingertips where a part's bottom is let "
            "go). Empty the hand first; the arm was not freed"))
    unknown = why_toggle_count_unknown(gripper)
    if unknown:
        return TeachSessionRefused(RefusalCode.JAWS_NOT_CONFIRMED, (
            f"nobody can say where the jaws stand ({unknown}): a part may be between them. Answer the jaws question "
            "first (POST /v1/cell/jaws/check); a pose is taught with empty jaws, and the arm was not freed"))
    refusal = teach_refusal(arm, tree=app, name=name, store=None)
    if refusal is not None:
        return TeachSessionRefused(_code(refusal.code), str(refusal))
    return None


def poses(console: "Console") -> PosesOut:
    """Home (read-only, from ``robot.home_joint_positions``, in degrees), every named pose with its label and screen, the
    default place, the file a taught pose is written to, and whether a pose can be taught now and why not.

    A pose the console wrote carries when it was taught (``source: taught``); one a person wrote in the YAML does not
    (``config``). ``why_not`` is the teach's own refusal, asked without freeing anything, or the chain's (no layer of
    the cell's own).
    """
    from src.robot.constants import HOME_JOINTS_DEFAULT  # noqa: PLC0415

    app, robot = console.resolved()
    home = robot.home_joint_positions or HOME_JOINTS_DEFAULT
    named = [
        PoseOut(name=name, label=pose.label, joints_deg=[float(v) for v in pose.joints_deg],
                source="taught" if pose.taught_at else "config", screen=pose.screen, note=pose.note,
                taught_at=pose.taught_at)
        for name, pose in robot.named_poses.items()
    ]
    tree = _tree(console)
    target = tree.pose_file()
    refused = refusal_before_a_teach(console, app)
    layer = "" if refused is not None else tree.pose_layer_refusal()
    why_not = refused.message if refused is not None else layer
    # The code beside the sentence, so the browser says it in its own words ("Planer startet noch"): the teach's own
    # refusal, or the pose door's for a chain whose last layer is not the cell's own.
    why_not_code = str(refused.code) if refused is not None else (str(RefusalCode.NO_LAYER) if layer else "")
    return PosesOut(
        home=PoseOut(name="home", label="Home", joints_deg=[round(math.degrees(float(v)), 6) for v in home],
                     source="config"),
        poses=named, default_place=robot.default_place_pose, teachable=not why_not, why_not=why_not,
        why_not_code=why_not_code,  # type: ignore[arg-type]
        target_file=str(target) if target is not None else None,
    )


def choose_default_place(console: "Console", name: str | None) -> PosesOut:
    """Write the default place pose, or ``None`` for none, into the cell's own layer through the pose door.

    Refused (:class:`TeachSessionRefused`): ``run_active``; ``unknown_pose`` for a name that is no pose of the tree;
    and the pose door's own refusals at their catalog statuses (``no_layer`` for a chain with no layer of the cell's
    own, ``invalid_value`` where the tree refuses the result, which is then rolled back).
    """
    from api.cell import RunLocked  # noqa: PLC0415

    try:
        console.require_idle()
    except RunLocked as locked:
        raise TeachSessionRefused(RefusalCode.RUN_ACTIVE, str(locked), {"run_id": locked.run_id}) from locked
    robot = console.robot()
    if name is not None and name not in robot.named_poses:
        known = ", ".join(sorted(robot.named_poses)) or "none"
        raise TeachSessionRefused(RefusalCode.UNKNOWN_POSE, (
            f"{name!r} is no pose of this cell (poses: {known}): teach it first, or choose one of them"),
            {"which": "place"})
    result = _tree(console).write_default_place_pose(name)
    if not result.applied:
        logger.warning("Default place %r refused (%s): %s", name, result.refused, result.message)
        raise TeachSessionRefused(_code(str(result.refused)), result.message or "the pose door refused the write",
                                  {"key": result.refused_key, "keys": list(result.keys)})
    logger.info("Default place pose set to %r in %s.", name, ", ".join(str(f) for f in result.files))
    return poses(console)


def payload(console: "Console") -> PayloadOut:
    """The payload the controller compensates for, read now (``arm.controller_payload``); ``not_connected`` without a
    connected cell. An arm or a controller that does not report it says so, and the person checks the pendant."""
    session = console.session
    if not session.connected:
        raise TeachSessionRefused(RefusalCode.NOT_CONNECTED, "the cell is not connected, so no payload can be read")
    read = getattr(session.arm, "controller_payload", None)
    if not callable(read):
        return PayloadOut(readable=False, source=(
            "this arm does not report the payload its controller compensates for: check it on the pendant"))
    try:
        found = read()
    except Exception as exc:  # noqa: BLE001 (said to the person, who checks the pendant)
        return PayloadOut(readable=False, source=(
            f"the controller's payload could not be read ({type(exc).__name__}: {exc}): check it on the pendant"))
    if found is None:
        return PayloadOut(readable=False, source="the controller does not report its payload: check it on the pendant")
    return PayloadOut(mass_kg=float(found.mass_kg), cog_mm=[float(v) for v in found.cog_mm], readable=True,
                      source="the controller, read now")


def drive_teach(console: "Console", run: "Run") -> StopCode:
    """The body of a teach run started with ``RunRegistry.start_kind``: the session :func:`start` opened for ``run``.

    :func:`start` hands the session's own body to the run, so this is for a caller that started a teach run itself; a
    run with no session frees nothing and ends ``cancelled``.
    """
    session = _session_of(console, run.id)
    if session is None:
        run.error = "no teach session was opened for this run; the arm was not freed"
        return StopCode.CANCELLED
    return session.drive(console, run)


def snapshot(console: "Console", run_id: str, token: str) -> TeachStateOut | None:
    """The session of ``run_id`` as a poll answers it (the poll is the heartbeat); ``None`` for no such session, and
    ``wrong_token`` (:class:`TeachSessionRefused`) for a token that is not its own."""
    session = _session_of(console, run_id)
    if session is None:
        return None
    session.allows(token)
    return session.state_out()


def capture(console: "Console", run_id: str, token: str) -> TeachStateOut | None:
    """Save: the pose is captured once the arm stands still. ``None`` for no such session; ``wrong_token``, and
    ``not_free`` where the arm is not free."""
    session = _session_of(console, run_id)
    if session is None:
        return None
    session.allows(token)
    return session.capture()


def cancel(console: "Console", run_id: str, token: str) -> TeachStateOut | None:
    """Hold, Cancel, the tab's beacon: the arm holds at once and nothing is saved. ``None`` for no such session;
    ``wrong_token``. A session that already ended answers as it stands."""
    session = _session_of(console, run_id)
    if session is None:
        return None
    session.allows(token)
    return session.hold_at_once("the person cancelled the teach")


def end_and_join(console: "Console", timeout_s: float = 10.0) -> bool:
    """End every open teach of ``console`` (the arm holds at once) and wait at most ``timeout_s`` for its thread; ``True``
    once none is left. A teach that does not end keeps the run lock, as any run does."""
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    with _SESSIONS_LOCK:
        live = [s for s in _SESSIONS.values() if s.console is console and not s.done.is_set()]
    for session in live:
        session.hold_at_once("the cell is being taken down")
    for session in live:
        session.done.wait(max(0.0, deadline - time.monotonic()))
        thread = session.thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(max(0.0, deadline - time.monotonic()))
    return all(s.done.is_set() and (s.thread is None or not s.thread.is_alive()) for s in live)


def hold_open_teaches(console: "Console", why: str, *, timeout_s: float = TEACH_HOLD_WAIT_S) -> bool:
    """Hold every open teach of ``console`` at once and wait at most ``timeout_s`` until each has let go of the arm
    (held, or never freed); ``True`` once none is free.

    What a Disconnect asks before it takes the arm down (``CellSession.teaches``): an arm taken down in teach mode is
    stopped by its watchdog in a person's hands, never held by the console. It waits for the hold alone, never for the
    run's end, which takes the console's own lock.
    """
    deadline = time.monotonic() + max(0.0, float(timeout_s))
    with _SESSIONS_LOCK:
        live = [s for s in _SESSIONS.values() if s.console is console and not s.left.is_set()]
    for session in live:
        session.hold_at_once(why)
    for session in live:
        if session.thread is not threading.current_thread():
            session.left.wait(max(0.0, deadline - time.monotonic()))
    held = all(s.left.is_set() for s in live)
    if live:
        logger.warning("%d open teach(es) held at once (%s)%s.", len(live), why,
                       "" if held else f"; one did not let go of the arm within {timeout_s:g} s")
    return held


class _SessionTeaches:
    """What the cell's session asks of the console's teaches (``CellSession.teaches``): hold every open one at once
    before the arm comes down."""

    def __init__(self, console: "Console") -> None:
        self._console = console

    def hold_at_once(self, why: str) -> bool:
        return hold_open_teaches(self._console, why)


# --- helpers ----------------------------------------------------------------------------------------------------------


def _keep(run_id: str, session: _TeachSession) -> None:
    """Keep ``session`` under its run, and only so many ended ones."""
    with _SESSIONS_LOCK:
        _SESSIONS[run_id] = session
        ended = [key for key, kept in _SESSIONS.items() if kept.done.is_set()]
        for key in ended[: max(0, len(ended) - _KEPT_SESSIONS)]:
            _SESSIONS.pop(key, None)


def _session_of(console: "Console", run_id: str) -> _TeachSession | None:
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(run_id)
    return session if session is not None and session.console is console else None


def _tree(console: "Console") -> Any:
    """The chain the console runs, as the pose door writes it (its own root, its own layers)."""
    from src.config.tree import ConfigTree  # noqa: PLC0415

    return ConfigTree(root=console.root, profile=console.profile, layers=tuple(console.layers),
                      named_root=str(console.root))


def _code(code: str) -> RefusalCode:
    """A library refusal's code as the catalog's; one the catalog does not know is a request the cell cannot do."""
    try:
        return RefusalCode(code)
    except ValueError:
        return RefusalCode.BAD_REQUEST


def _hold_asked_by(run: Any, arm: Any = None) -> str:
    """Why the teach must hold at once: a halt (the run's flag, or the arm's own latch however it was set), a
    Disconnect, a stop; ``""`` for none."""
    from src.robot.core.arm_capabilities import halt_state_of  # noqa: PLC0415

    if run is not None:
        if getattr(run, "halt_requested", False) is True:
            return "halt now was pressed"
        abandoned = getattr(run, "abandoned", "")
        if isinstance(abandoned, str) and abandoned:
            return "the cell is being disconnected"
        if getattr(run, "stop_requested", False) is True:
            return "the run was asked to stop"
    halted = halt_state_of(arm)
    if halted is not None:
        return f"the arm is halted ({halted.reason})"
    return ""


def _within(seen: float, now: float, tolerance: float) -> bool:
    """``seen`` within ``tolerance`` of ``now``. Written as the one comparison that holds, so a value that is no number
    on either side fails it: NaN compares false against every tolerance (``abs(x - inf)`` is inf or NaN)."""
    return abs(seen - now) <= tolerance


def _payload_changed(arm: Any, seen: Any) -> str:
    """Why the controller's payload is not the one the person confirmed, or ``""``: a payload the arm reports is
    compared within a hundredth of a kilogram and a millimetre; one it does not report must have been seen as none. A
    value that is no number (JSON's ``NaN`` or ``Infinity``, on either side) confirms nothing."""
    read = getattr(arm, "controller_payload", None)
    try:
        now = read() if callable(read) else None
    except Exception:  # noqa: BLE001 (a payload nobody can read now is compared as none)
        now = None
    if now is None:
        if seen is None:
            return ""
        return ("the controller does not report its payload now, and a payload was confirmed: read it again "
                "(GET /v1/teach/payload); the arm was not freed")
    shown = f"{float(now.mass_kg):.2f} kg at ({', '.join(f'{float(v):.0f}' for v in now.cog_mm)}) mm"
    if seen is None or seen.mass_kg is None or seen.cog_mm is None:
        return (f"the controller compensates for {shown}, and no payload was confirmed: read it (GET /v1/teach/payload) "
                "and confirm it; the arm was not freed")
    cog_seen = [float(v) for v in seen.cog_mm]
    cog_now = [float(v) for v in now.cog_mm]
    if not all(math.isfinite(v) for v in (float(seen.mass_kg), *cog_seen)):
        return (f"the payload confirmed is no number, and the controller compensates for {shown}: read it again "
                "(GET /v1/teach/payload) and confirm it; the arm was not freed")
    if (not _within(float(seen.mass_kg), float(now.mass_kg), _MASS_TOLERANCE_KG) or len(cog_seen) != len(cog_now)
            or not all(_within(a, b, _COG_TOLERANCE_MM) for a, b in zip(cog_seen, cog_now))):
        return (f"the controller now compensates for {shown}, not the payload that was confirmed: a wrong payload makes "
                "the freed arm sink or rise in a person's hands. Read it again and confirm it; the arm was not freed")
    return ""


def _holding_said(reason: str) -> str:
    if reason == "time_limit":
        return "The time to teach this pose is up: the arm is held once it stands still."
    return "The browser has gone quiet: the arm is held once it stands still, never while it moves."
