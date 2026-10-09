"""The console's session: which config tree, which profile chain, and whether a run owns the cell.

One place touches the loader's global profile. ``set_active_profile`` mutates process state, so a
request handler doing it inline would let two concurrent requests read each other's chain and report
a UR3e verdict for a simulator tree. Everything here funnels through :meth:`Console.config`, under a
lock.

One place holds the run flag. Configuration may not be written while the cell is moving, and that
rule is the reason the write endpoint is allowed to exist at all. :meth:`Console.require_idle` is
the guard; ``RunRegistry`` in ``api/runs.py`` is the only thing that sets ``active_run_id``.

And one place remembers a stop. A moving run that ended on a problem code leaves the arm where it stopped and a
:class:`RecoveryRecord` here, on the console rather than on the run or the cell: it survives Disconnect, a rebuild
and a page reload, and while it stands nothing new starts. A person first confirms "the cell is clear"; then the way
back is Restart or Home, whose first motion is the planned move to the return pose, and only arriving there ends it.
A halt a person pressed is carried here too (:attr:`Console.carried_halt`), and latches the arm of every later build:
only "the cell is clear" ends a halt, never a rebuild. The server's console keeps both in a small file
(:attr:`Console.stop_file`, :func:`default_stop_file`), so they outlive a restart of the server as well (review of
2026-10-02): read back, the record stands again uncleared, since a restart is no "the cell is clear".
The console also keeps the stamps the next motion reads: when a person's hands were last at the arm (a teach, "open
now", "Backen leer") against when the arm last moved by itself, which is when a moving run counts down first.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from api.codes import MOVING_KINDS, RunKind, StopClass, StopCode, stop_class_of
from api.constants import API_LOG_DIR, CELL_LOG_FILE
from api.events import CELL_STREAM, EventHub, Severity
from api.jaws import BrowserJawQuestions
from api.lifecycle import CellSession
from api.live import LiveFrames
from api.overlays import OverlayStore
from src.config.loader import (
    active_profile,
    load_config,
    profile_layers,
    reload_config,
    set_active_profile,
)
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.config.schema import AppConfig
    from src.config.schema.robot.robot_schema import RobotConfig
    from api.runs import RunRegistry
    from src.robot.execution.real_cell.preflight import PreflightReport

__all__ = [
    "STOP_FILE_ENV",
    "Console",
    "NoRobotConfigured",
    "RecoveryRecord",
    "RunLocked",
    "console",
    "default_stop_file",
    "set_console",
    "take_down",
]

logger = create_logger("Console", CELL_LOG_FILE, log_dir=API_LOG_DIR)

#: Why the next moving run counts down: a teach ended, or a person's hands were at the jaws (they held the part while
#: "open now" opened them, or they emptied the hand and said "Backen leer").
CountdownBecause = Literal["teach", "jaws_opened"]


@dataclass(frozen=True, slots=True)
class RecoveryRecord:
    """A moving run ended on a problem code; the arm stands where it stopped (``CellOut.recovery``)."""

    run_id: str
    kind: RunKind
    stop_code: StopCode
    #: When the run ended, Unix seconds.
    at: float
    #: The program believed a part was in the jaws. Every gate reads the live hand first; for a hand that can say
    #: nothing itself (no toggle count, no measurement) this belief keeps ``part_still_held`` standing until "Backen leer".
    holding: bool = False
    #: When a person confirmed "the cell is clear" since the stop; ``None`` until then.
    cleared_at: float | None = None

    @property
    def cleared(self) -> bool:
        """Whether "the cell is clear" was confirmed after the stop."""
        return self.cleared_at is not None and self.cleared_at > self.at

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "kind": str(self.kind),
            "stop_code": str(self.stop_code),
            "at": self.at,
            "holding": self.holding,
            "cleared_at": self.cleared_at,
        }

    @classmethod
    def from_dict(cls, said: Any) -> "RecoveryRecord":
        """The record :meth:`to_dict` said; ``ValueError`` or ``TypeError`` for anything else (a record of a kind that
        does not move, a stop that is no problem, a time that is no number)."""
        if not isinstance(said, dict):
            raise TypeError(f"a stop record is a mapping, not {said!r}")
        kind, code = RunKind(str(said["kind"])), StopCode(str(said["stop_code"]))
        if kind not in MOVING_KINDS or stop_class_of(str(code)) is not StopClass.PROBLEM:
            raise ValueError(f"a {kind} run that ended {code} leaves no stop record")
        at, cleared = said["at"], said.get("cleared_at")
        if isinstance(at, bool) or not isinstance(at, (int, float)):
            raise TypeError(f"the stop's time is no number: {at!r}")
        if cleared is not None and (isinstance(cleared, bool) or not isinstance(cleared, (int, float))):
            raise TypeError(f"the clear's time is no number: {cleared!r}")
        run_id = said["run_id"]
        if not isinstance(run_id, str) or not run_id:
            raise TypeError(f"the stopped run's id is no text: {run_id!r}")
        return cls(run_id=run_id, kind=kind, stop_code=code, at=float(at), holding=bool(said.get("holding", False)),
                   cleared_at=None if cleared is None else float(cleared))

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "config"

#: The environment variable that names the file a console keeps its stop in (:attr:`Console.stop_file`): ``python -m api``
#: sets it for ``--reload``'s worker, whose console is this module's own, and a person names another file with it (a
#: console on a tree named with ``--data``). Unset or empty, a console keeps nothing on disk.
STOP_FILE_ENV: Final[str] = "WILLY_CONSOLE_STOP_FILE"
#: The stop file's format. A file of another, or one that does not read, is a stop of unknown origin.
_STOP_FILE_VERSION: Final[int] = 1
#: The run a stop of unknown origin names: a stop file nobody could read. Home is its way back; nothing restarts it.
_UNKNOWN_STOP_RUN: Final[str] = "run-unknown"


def default_stop_file(profile: str | None) -> Path:
    """Where ``python -m api`` keeps its console's stop across a restart: under ``logs/console``, beside the console's
    grasp records, one file per profile chain (``stop.json``, ``stop.console_dummy.json``, ``stop.ur10+tiltcam.json``),
    so a desk rehearsal's stop never gates the cell's own console, nor the other way round."""
    chain = re.sub(r"[^A-Za-z0-9_.+-]", "_", (profile or "").strip().replace(",", "+")).strip("._")
    return _REPO_ROOT / "logs" / "console" / (f"stop.{chain}.json" if chain else "stop.json")


def _stop_file_from_the_environment() -> Path | None:
    """The file :data:`STOP_FILE_ENV` names, or ``None`` where it names none: a console nobody named a file for (a
    program's own, a test's) keeps nothing on disk."""
    said = os.environ.get(STOP_FILE_ENV, "").strip()
    return Path(said) if said else None


class NoRobotConfigured(RuntimeError):
    """Raised when the tree the console was pointed at configures no robot at all."""

    def __init__(self, root: Path, profile: str | None) -> None:
        chain = profile or "(no profile)"
        super().__init__(
            f"the config tree at {root} (profile chain: {chain}) has no robot section, so there is "
            f"no cell to check or prepare. Point the console at a tree that configures an arm, or "
            f"start it with the profile that does."
        )
        self.root = root
        self.profile = profile


class RunLocked(RuntimeError):
    """Raised when something would change the cell while a run owns it.

    Carries the run id so the console can name the run rather than answer a bare "busy": an operator
    looking at a stalled browser needs to know whether the thing holding the lock is theirs.
    """

    def __init__(self, run_id: str) -> None:
        super().__init__(
            f"run {run_id} is active. Configuration changes are refused while the cell is moving; "
            f"this is not a queue; re-submit after the run ends."
        )
        self.run_id = run_id


@dataclass
class Console:
    """What the server knows about the cell between requests.

    Owns exactly two things a request handler must never own itself: the loader's process-global
    profile, and the built cell. Both are shared mutable state that two browser tabs reach at the
    same time, so both are guarded by this class's ``_lock`` rather than by a handler.
    """

    root: Path = _DEFAULT_ROOT
    #: The profile chain string, e.g. ``"ur3e,tiltcam"``. Defaults to ``WILLY_PROFILE``, the same
    #: environment variable every other entry point in this repo obeys: a shell already set up for a
    #: UR3e cell configures the console for one too, and ``--reload``'s worker subprocess (which
    #: re-imports this module and cannot see the parent's objects) still comes up on the right chain.
    profile: str | None = field(default_factory=active_profile)
    #: The text prompt a real perception source is built with, and the phrase it grounds between
    #: runs. It lives here because ``build_real_components`` needs one at construction time; a run's
    #: own prompt replaces it for that run through ``service.set_prompt``, and ``api/runs.py`` puts
    #: it back when the run ends. Nothing is rebuilt.
    prompt: str = "object"
    #: Set while a run owns the cell. ``RunRegistry`` assigns and clears it; everything else only
    #: refuses against it.
    active_run_id: str | None = None
    #: Where this console appends grasp records: the default below when the config names no path of
    #: its own, and the configured path once :meth:`build` has read one. The default sits under
    #: ``logs/`` rather than the config tree: it is data the console produced, not configuration,
    #: and it must not end up in a diff of the cell's settings.
    record_log_path: Path = field(
        default_factory=lambda: _REPO_ROOT / "logs" / "console" / "grasp_records.jsonl"
    )
    #: The built cell and its connection state.
    session: CellSession = field(default_factory=CellSession)
    #: The event stream. One hub per console: sequence numbers are per run, so runs never collide.
    hub: EventHub = field(default_factory=EventHub)
    #: Runs, and the only thing that sets :attr:`active_run_id`.
    runs: "RunRegistry | None" = None
    #: The recovery record of the newest problem stop of a moving run, until a Restart or Home arrives; ``None`` while
    #: nothing stopped mid-task. Here, not on the cell: it outlives Disconnect and a rebuild.
    recovery: RecoveryRecord | None = None
    #: The stamps the next motion reads, set only by the ``stamp_*`` methods and read through the read-only properties
    #: of the same name without the underscore (:attr:`jaws_confirmed_at`, ...): the countdown orders them by the
    #: methods' own counter, so a stamp written directly would turn the hands-off countdown off unseen.
    _jaws_confirmed_at: float | None = field(default=None, init=False, repr=False)
    _jaws_opened_at: float | None = field(default=None, init=False, repr=False)
    _teach_ended_at: float | None = field(default=None, init=False, repr=False)
    _last_motion_started_at: float | None = field(default=None, init=False, repr=False)
    #: The jaws question in the browser (``api.jaws``). Built in ``__post_init__``: it needs the hub.
    jaws: "BrowserJawQuestions | None" = None
    #: Each camera's last display frame (``GET /v1/camera/live``).
    live: LiveFrames = field(default_factory=LiveFrames)
    #: The grasp overlays task runs captured (``GET /v1/runs/{id}/overlays/{n}``).
    overlays: OverlayStore = field(default_factory=OverlayStore)
    #: The models section the cell was BUILT with (``AppConfig.models`` at the last build), ``None`` before one: the
    #: command reader and the commands light read it, not the tree as it reads now, which may name weights the cell does
    #: not hold.
    built_models: Any = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    #: Which stamp came last, by a counter: two stamps within one tick of a coarse wall clock (Windows) still say which
    #: came first, and "a teach since the last motion" must never be decided by a tie.
    _stamp_order: dict[str, int] = field(default_factory=dict, repr=False)
    _stamp_counter: "itertools.count[int]" = field(default_factory=lambda: itertools.count(1), repr=False)
    #: How often a person pressed "halt now" (``POST /v1/cell/brake``): a Disconnect gives back the latch it set itself
    #: only where no press came since, because a press then lands on that very latch (the arm keeps the first record).
    _halt_presses: int = field(default=0, init=False, repr=False)
    #: The file this console keeps its stop in, across a restart of the server (:meth:`_keep_the_stop`): the recovery
    #: record, the stopped run it names, and a halt nobody has said the cell is clear of. ``None`` keeps nothing on disk
    #: (a program's own console, a test's); ``python -m api`` names one (:func:`default_stop_file`), and
    #: :data:`STOP_FILE_ENV` names one for a console built without it. Read back once, as the console is built.
    stop_file: Path | None = field(default_factory=_stop_file_from_the_environment)
    #: A halt that latched the arm (``POST /v1/cell/brake``, a teardown's latch) and that nobody has said the cell is clear
    #: of, ``(reason, requested_at)``: carried to the arm of every later build (:meth:`build`) and across a restart, so
    #: only a person's "the cell is clear" ends a halt, never a rebuild (review of 2026-10-02).
    _carried_halt: tuple[str, float] | None = field(default=None, init=False, repr=False)

    # --- the cell ----------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        # The registry needs the hub, and a `default_factory` cannot see a sibling field. Built here
        # rather than lazily so `console().runs` is never None for a caller.
        if self.runs is None:
            from api.runs import RunRegistry

            self.runs = RunRegistry(self.hub)
        if self.jaws is None:
            self.jaws = BrowserJawQuestions(self.hub)
        # A run the registry forgets takes its overlays with it, in the same breath as its events.
        if getattr(self.runs, "on_forget", None) is None:
            self.runs.on_forget = self.overlays.forget
        self._read_the_kept_stop()

    @property
    def registry(self) -> "RunRegistry":
        """The run registry, non-optional for callers. ``__post_init__`` guarantees it exists."""
        assert self.runs is not None
        return self.runs

    def fingerprint(self) -> str:
        """A short hash of the tree in force, used to bind a connect token to the config it described.

        Content-based, not a timestamp: an edit that changes nothing (a reformatted comment, a rewritten
        file with the same values) must not invalidate an operator's acknowledgement, and an edit that
        changes one number must.
        """
        payload = self.config().model_dump_json().encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:16]

    def build(self, *, rehearse: bool = False) -> Any:
        """Build the cell through the same path the CLI runner uses. Touches no robot.

        ``from_robot_config`` deliberately does not connect the arm, because the caller owns the
        lifecycle, so this is safe to call at a desk. For a real cell it is not cheap: it opens the
        camera and loads the detector and segmenter onto the GPU, which takes tens of seconds.

        Building twice is a normal thing to do (build, read the refusal, fix a key, build again), so
        the second build must not be poisoned by the first. Three rules make that hold, in this
        order: the state is checked before anything is acquired, the previous cell's camera is
        released before the new one is opened (a cell has one device), and :meth:`CellSession.adopt`
        releases again as a backstop for any service that reached it another way.
        """
        from src.robot.execution.autonomous_grasp import (
            build_real_cell,
            build_rehearsal_cell,
        )

        with self._lock:
            # One read for both halves (see `resolved`). `config()` sets and restores the process
            # profile around each call, so reading twice was also two dances where one is correct.
            app_cfg, robot = self.resolved()
            # The one line that says what the pause before the next entry was spent on. `rehearse`
            # is on it because the two paths cost wildly differently, and a log that does not say
            # which one ran cannot explain either duration.
            logger.info(
                "Building the cell (rehearse=%s) from %s, profile chain %s.",
                rehearse, self.root, self.profile or "(none)",
            )
            started = time.perf_counter()
            # Before the camera is opened and two models are loaded onto the GPU. Asking afterwards
            # would let a refused rebuild claim the device it was refused for.
            self.session.may_adopt()
            # And hand the old device back before asking for it again. A cell has one camera, and
            # `pipeline.start()` on a device another pipeline is still streaming fails, so releasing
            # inside `adopt()`, after the new streamer is already open, closes the leak but not the
            # reopen.
            #
            # The cost is deliberate: a build that then fails leaves the cell unbuilt rather than
            # holding a service whose camera has been closed underneath it. "Not built" is a state
            # the console can render and an operator can act on; "built, but blind" is not.
            self.session.release()
            # One VLM copy per process, shared by detection and the command reader (Q8 A). A cell that will not
            # detect with it (the rehearsal, a GroundingDINO cell) hands the copy back here, unless a person loaded it
            # ("Laden") and the new config still names its weights; a VLM cell's own build reuses or replaces it.
            self._release_the_vlm_unless_this_cell_needs_it(app_cfg.models, rehearse=rehearse)
            # One call, and the same one the CLI runner makes. Building the perception and grasp
            # side here and then calling `from_robot_config` separately is how a browser and a
            # terminal come to disagree about what "the cell" is.
            service = (build_rehearsal_cell(robot, data_dir=self.root) if rehearse
                       else build_real_cell(robot, prompt=self.prompt, app_config=app_cfg, data_dir=self.root))
            try:
                self._log_records_of(service, robot)
                self.session.adopt(service)
            except BaseException:
                # The new service was never adopted, so nothing else would close the camera it opened.
                from src.robot.execution.lifecycle import release_perception  # noqa: PLC0415

                release_perception(service)
                raise
            self.built_models = app_cfg.models
            # The jaws question goes to the browser, never to this server's terminal: the seam is handed to the hand
            # of every build. A failed install fails the build and leaves the cell as every failed build leaves it,
            # unbuilt, never built with a hand that would ask at the terminal; Connect refuses a toggle without the
            # seam only as a backstop. Looked up at call time, so the module's function is the one that runs.
            from api import jaws as browser_jaws  # noqa: PLC0415

            try:
                browser_jaws.install(self)
            except BaseException as refused:
                logger.error(
                    "The jaws question could not be handed to the browser (%s: %s); the build is undone.",
                    type(refused).__name__, refused,
                )
                self.session.release()
                self.built_models = None
                raise
            # A halt nobody has said the cell is clear of latches this build's arm too: a rebuild ends no halt.
            try:
                self._latch_the_carried_halt()
            except BaseException as refused:
                logger.error(
                    "The halt this console carries could not latch the arm of the new build (%s: %s); the build is "
                    "undone.", type(refused).__name__, refused,
                )
                self.session.release()
                self.built_models = None
                raise
            logger.info(
                "Cell built in %.1f s: arm=%s, gripper=%s, records -> %s.",
                time.perf_counter() - started,
                type(self.session.arm).__name__,
                type(self.session.gripper).__name__,
                self.record_log_path,
            )
            return service

    def _log_records_of(self, service: Any, robot: "RobotConfig") -> None:
        """Record logging is off in the shipped config (``grasping.record_log_path: null``), so a console-driven cell
        would keep no history at all and a bring-up that goes wrong would leave nothing to diagnose from afterwards. The
        console therefore turns it on itself, into its own file, without touching the YAML: a cell that never runs the
        console is unaffected, and an operator who did configure a path keeps it.

        ``from_robot_config`` has already wired the configured path when there is one; calling
        ``enable_record_logging`` again would replace it, so this only fills a gap.
        """
        if getattr(robot.grasping, "record_log_path", None) is None:
            self.record_log_path.parent.mkdir(parents=True, exist_ok=True)
            # Warning rather than info: the console is overriding what the YAML says (nothing), and
            # an operator who later cannot find their records in the configured place should be
            # able to grep for the moment the console decided otherwise.
            logger.warning(
                "The config names no grasp record path; the console is logging its own to %s.",
                self.record_log_path,
            )
            service.enable_record_logging(
                self.record_log_path,
                provenance={
                    "robot_vendor": str(robot.vendor),
                    "robot_model": str(getattr(getattr(robot, "ur", None), "model", "") or ""),
                    "written_by": "operator console",
                },
            )
        else:
            self.record_log_path = Path(str(robot.grasping.record_log_path))
            logger.info(
                "Grasp records go to the configured path %s.", self.record_log_path
            )

    @staticmethod
    def _release_the_vlm_unless_this_cell_needs_it(models: Any, *, rehearse: bool) -> None:
        """Hand the process's one VLM copy back before a build whose cell does not detect with it (the rehearsal, a
        GroundingDINO cell), unless a person loaded it ("Laden") and the new config still names its weights. Builds and
        loads nothing; a failure is said, never a reason to refuse the build."""
        try:
            from src.models.vlm import shared_vlm, vlm_detects  # noqa: PLC0415

            if rehearse or not vlm_detects(models):
                pipeline = getattr(models, "pipeline", None)
                shared_vlm().release_unless_requested(pipeline.zero_shot.vlm if pipeline is not None else None)
        except Exception as exc:  # noqa: BLE001 (the copy stays; the commands light says what is held)
            logger.warning("Handing the VLM copy back before the build failed: %s: %s", type(exc).__name__, exc)

    def lock_key(self) -> str | None:
        """The cross-process lock key for this cell, or ``None`` when it claims no controller."""
        from src.robot.execution.cell_lock import cell_lock_key

        return cell_lock_key(self.robot())

    @property
    def layers(self) -> tuple[str, ...]:
        return profile_layers(self.profile)

    def config(self) -> "AppConfig":
        """The tree as a runner would load it under this console's profile chain.

        The previous global profile is restored afterwards: the console shares a process with library
        code that reads ``active_profile()``, and leaving the chain set behind would change what a
        later, unrelated call believes it is configuring.
        """
        with self._lock:
            previous = active_profile()
            if self.profile != previous:
                # Debug, not info: this fires on nearly every request, and it is only interesting when
                # somebody is asking why a value resolved the way it did.
                logger.debug(
                    "Reading the tree under %s (process profile was %s); it is restored afterwards.",
                    self.profile or "(none)", previous or "(none)",
                )
                set_active_profile(self.profile)
                reload_config()
            try:
                return load_config(self.root)
            finally:
                if self.profile != previous:
                    set_active_profile(previous)
                    reload_config()

    def robot(self) -> "RobotConfig":
        """The robot section, or a refusal naming the tree that has none.

        ``AppConfig.robot`` is optional, since a tree may configure cameras and models and no arm at
        all, so every consumer here would otherwise fail with an ``AttributeError`` on ``None`` at
        request time. Saying which directory configures no robot turns that into an answer.
        """
        return self.resolved()[1]

    def resolved(self) -> "tuple[AppConfig, RobotConfig]":
        """The tree and its robot section, from one read, for a caller that needs both.

        A cell has two halves and they came from different trees. :meth:`build` used to take the
        robot half from here and let `build_real_components` read the camera half itself, with a
        bare `load_config()` that consults neither :attr:`root` nor :attr:`profile`. A console
        pointed at a deployment tree therefore ran that tree's arm and this checkout's cameras, and
        the profile dance :meth:`config` performs was undone one call later. Reading once and
        handing both halves down is what makes the two agree by construction rather than by
        coincidence.
        """
        config = self.config()
        if config.robot is None:
            raise NoRobotConfigured(self.root, self.profile)
        return config, config.robot

    def preflight(self) -> "PreflightReport":
        """The same checklist the CLI prints, from the same function. Not a second opinion."""
        from src.robot.execution.real_cell.preflight import run_config_preflight

        config, robot = self.resolved()
        return run_config_preflight(robot, camera=config.camera, data_dir=self.root)

    def require_idle(self) -> None:
        """Guard for anything that changes the cell. Raises :class:`RunLocked`, never queues."""
        with self._lock:
            if self.active_run_id is not None:
                raise RunLocked(self.active_run_id)

    @property
    def halt_presses(self) -> int:
        """How often a person pressed "halt now" since this console started (:meth:`note_halt_pressed`)."""
        return self._halt_presses

    def note_halt_pressed(self) -> None:
        """A person pressed "halt now" (``POST /v1/cell/brake``), before the arm is latched: a latch the teardown set
        itself is then the person's too, and stays (:func:`take_down`)."""
        with self._lock:
            self._halt_presses += 1

    # --- the stamps the next motion reads ------------------------------------------------------------
    #
    # Read-only: each is set by its ``stamp_*`` method alone, which also orders it (``_stamp_order``), and that order is
    # what the countdown reads. A track that assigns one fails at once instead of turning the countdown off unseen.

    @property
    def jaws_confirmed_at(self) -> float | None:
        """When the jaws were last answered open in the browser (a toggle), Unix seconds (:meth:`stamp_jaws_open`)."""
        return self._jaws_confirmed_at

    @property
    def jaws_opened_at(self) -> float | None:
        """When a person's hands were last at the jaws: "open now" opened them while a person held the part
        (:meth:`stamp_jaws_open`), or a person emptied the hand, "Backen leer" (:meth:`stamp_jaws_emptied`)."""
        return self._jaws_opened_at

    @property
    def teach_ended_at(self) -> float | None:
        """When the last teach session ended: a person's hands were on the arm (:meth:`stamp_teach_ended`)."""
        return self._teach_ended_at

    @property
    def last_motion_started_at(self) -> float | None:
        """When a moving run last moved the arm, as its driver knows it (:meth:`stamp_motion_started`)."""
        return self._last_motion_started_at

    def _stamp(self, name: str, at: float | None) -> float:
        """Order the stamp ``name`` after every earlier one and answer its wall-clock time. The caller holds the lock."""
        self._stamp_order[name] = next(self._stamp_counter)
        return time.time() if at is None else at

    def stamp_jaws_open(self, *, opened: bool, at: float | None = None) -> None:
        """The jaws question ended with the jaws standing open. ``opened``: "open now" opened them just now, while a
        person held the part, so the next moving run counts down first."""
        with self._lock:
            self._jaws_confirmed_at = now = self._stamp("jaws_confirmed", at)
            if opened:
                self._stamp_order["jaws_opened"] = self._stamp_order["jaws_confirmed"]
                self._jaws_opened_at = now

    def stamp_jaws_emptied(self, at: float | None = None) -> None:
        """A person emptied the hand ("Backen leer", ``POST /v1/cell/acknowledge`` with ``jaws_empty``): their hands were
        at the jaws, so the next moving run counts down first, as after "open now" (owner decision Q2 = A).

        Ordered as the ``jaws_opened`` stamp, the countdown's word for a person's hands at the jaws, and read through
        :attr:`jaws_opened_at`. Never the ``jaws_confirmed`` stamp: the person's word answers no jaws question, so a
        toggle's way back still asks where its jaws stand since the stop."""
        with self._lock:
            self._jaws_opened_at = self._stamp("jaws_opened", at)

    def stamp_teach_ended(self, at: float | None = None) -> None:
        """A teach session ended: a person's hands were on the arm, so the next moving run counts down first."""
        with self._lock:
            self._teach_ended_at = self._stamp("teach", at)

    def stamp_motion_started(self, at: float | None = None) -> None:
        """A moving run moved the arm: called by its driver once it knows it did (``api.runs.motion_started``: a pick
        that gripped, a part placed, an arrival), never before, so a run that counted down and then moved nothing leaves
        the countdown due. No person's hands can come to the arm while a run holds it (every route that stamps them
        refuses ``run_active``), so a stamp made after the motion orders as one made before it."""
        with self._lock:
            self._last_motion_started_at = self._stamp("motion", at)

    def countdown_because(self) -> CountdownBecause | None:
        """Why the next moving run counts down 3 s before its first motion, or ``None`` where it does not.

        A teach, or a person's hands at the jaws ("open now", "Backen leer"), after the last motion: the owner's rule,
        hands off before the robot drives by itself. The later of the two names it. Stamped through the ``stamp_*``
        methods, whose order decides, never the wall clock.
        """
        with self._lock:
            since = self._stamp_order.get("motion", 0)
            stamps = [(self._stamp_order[name], why) for name, why in (("teach", "teach"), ("jaws_opened", "jaws_opened"))
                      if self._stamp_order.get(name, 0) > since]
        if not stamps:
            return None
        return "teach" if max(stamps)[1] == "teach" else "jaws_opened"

    def countdown_due(self) -> bool:
        """Whether the next moving run counts down first (:meth:`countdown_because`)."""
        return self.countdown_because() is not None

    # --- the recovery record -------------------------------------------------------------------------

    def record_problem_stop(self, run: Any) -> RecoveryRecord | None:
        """Write the recovery record for ``run`` where it is a moving kind that ended on a problem code; else nothing.

        Called by the run as it ends, while it still holds the run lock (``RunRegistry._finish``), so the record
        stands before anything new can start. The newest problem stop replaces an older record. ``cell.recovery`` goes
        on the cell's stream. No record for a benign end, an ask, a teach or a planner run: nothing was stopped
        mid-motion, and the arm's own latch gates a halt with no run.
        """
        try:
            kind = RunKind(str(getattr(run, "kind", "")))
        except ValueError:
            return None
        code = str(getattr(run, "stop_code", "") or "")
        if kind not in MOVING_KINDS or stop_class_of(code) is not StopClass.PROBLEM:
            return None
        finished = getattr(run, "finished_at", None)
        record = RecoveryRecord(
            run_id=str(run.id), kind=kind, stop_code=StopCode(code),
            at=float(finished) if isinstance(finished, (int, float)) else time.time(),
            holding=bool(getattr(run, "holding", False)),
        )
        with self._lock:
            self.recovery = record
            # Ordered among the stamps, so "the jaws were answered open since the stop" is decided by the order the
            # two happened in, never by two wall-clock readings within one tick.
            self._stamp("recovery", record.at)
            self._keep_the_stop(run)
        logger.warning(
            "Recovery record written: %s run %s stopped on %s where the arm stands; a person confirms the cell is clear, "
            "then Restart or Home.", kind, record.run_id, code,
        )
        self.hub.publish(
            CELL_STREAM, "cell.recovery", severity=Severity.ERROR,
            human=(f"The {kind} run {record.run_id} stopped where the arm stands ({code}). Nothing moves until a person "
                   f"confirms the cell is clear; then Restart or Home, whose first motion is the planned move home."),
            run_id=record.run_id, kind=str(kind), stop_code=code, at=record.at,
        )
        return record

    def mark_cell_clear(self, at: float | None = None) -> RecoveryRecord | None:
        """A person confirmed "the cell is clear" (``POST /v1/cell/acknowledge``): the record stays, and is cleared.

        Stamped after the stop, strictly, so ``cleared_at > at`` holds for a browser comparing the two. ``None`` where
        no record stands.
        """
        with self._lock:
            record = self.recovery
            if record is None:
                return None
            now = time.time() if at is None else at
            self.recovery = replace(record, cleared_at=max(now, math.nextafter(record.at, math.inf)))
            self._keep_the_stop()
            return self.recovery

    def mark_jaws_emptied(self) -> RecoveryRecord | None:
        """A person said the hand holds nothing ("Backen leer", ``POST /v1/cell/acknowledge`` with ``jaws_empty``): the
        record no longer believes a part is held. ``None`` where no record stands."""
        with self._lock:
            record = self.recovery
            if record is None:
                return None
            self.recovery = replace(record, holding=False)
            self._keep_the_stop()
            return self.recovery

    def jaws_answered_since(self, record: RecoveryRecord) -> bool:
        """Whether the jaws question ended with the jaws open after ``record``'s stop (``stamp_jaws_open``), by the
        stamps' own order: a toggle's way back asks where its jaws stand first."""
        with self._lock:
            answered = self._stamp_order.get("jaws_confirmed", 0)
            stopped = self._stamp_order.get("recovery", 0)
            standing = self.recovery
        if standing is not record and standing is not None and standing.run_id != record.run_id:
            return False
        return answered > stopped

    def end_recovery(self, run_id: str, by: Literal["restart", "home"]) -> RecoveryRecord | None:
        """End the record: run ``run_id`` arrived at its return pose (a Restart's first return, a Home run's arrival).

        ``cell.recovery_ended`` names the stopped run, how it ended, and the run that ended it. ``None`` where no
        record stood.
        """
        with self._lock:
            record, self.recovery = self.recovery, None
            if record is not None:
                self._keep_the_stop()
        if record is None:
            return None
        logger.info("Recovery record of %s ended: run %s (%s) arrived at its return pose.", record.run_id, run_id, by)
        self.hub.publish(
            CELL_STREAM, "cell.recovery_ended", severity=Severity.SUCCESS,
            human=f"The arm is back at its return pose ({by}); the stop of run {record.run_id} is behind it.",
            run_id=record.run_id, by=by, ended_by=run_id,
        )
        return record

    # --- the halt carried to every later build, and the stop kept across a restart ---------------------------------

    @property
    def carried_halt(self) -> tuple[str, float] | None:
        """The halt this console carries until a person says the cell is clear, ``(reason, requested_at)``, or ``None``
        (:meth:`carry_halt`)."""
        return self._carried_halt

    def carry_halt(self, reason: str, at: float | None = None) -> None:
        """A halt latched the arm (``POST /v1/cell/brake``, a teardown's latch): carry it to every later build, and across
        a restart, until a person says the cell is clear (:meth:`forget_halt`). The first is kept, as the arm keeps the
        first record of its latch."""
        with self._lock:
            if self._carried_halt is None:
                self._carried_halt = (str(reason).strip() or "halt requested", time.time() if at is None else float(at))
                self._keep_the_stop()

    def forget_halt(self) -> bool:
        """A person said the cell is clear (or a teardown gave back the latch it set itself): the halt is carried no
        longer. Whether one was."""
        with self._lock:
            carried, self._carried_halt = self._carried_halt, None
            if carried is not None:
                self._keep_the_stop()
        return carried is not None

    def _latch_the_carried_halt(self) -> bool:
        """Latch the arm a build just adopted with the halt this console carries: a rebuild, a restart, ends no halt.
        Whether it is latched now. An arm with no latch (a sim or KUKA arm) is said and left as it is: it refuses nothing
        it ever refused, and the carried halt stays until a person says the cell is clear."""
        from src.robot.core.arm_capabilities import SupportsHalt, halt_state_of  # noqa: PLC0415

        carried = self._carried_halt
        if carried is None:
            return False
        arm = self.session.arm
        if not isinstance(arm, SupportsHalt):
            logger.warning("The halt carried since %.0f (%s) cannot latch this build's arm (%s has no latch); a person "
                           "confirms the cell is clear all the same.", carried[1], carried[0], type(arm).__name__)
            return False
        if halt_state_of(arm) is None:
            arm.halt(carried[0])
        if halt_state_of(arm) is None:
            raise RuntimeError(f"the arm ({type(arm).__name__}) did not latch with the halt carried since a person "
                               f"pressed it ({carried[0]})")
        logger.warning("The arm of this build is latched with the halt carried since %.0f (%s): a person confirms the "
                       "cell is clear before it moves.", carried[1], carried[0])
        return True

    def _keep_the_stop(self, run: Any = None) -> None:
        """Write what stands of the stop to :attr:`stop_file`, or remove the file where nothing stands; the caller holds
        the lock, so writes land in the order the stop changed. ``run`` is the stopped run where the caller holds it (as it
        ends), else the registry's. A write that fails is said, never raised: the stop still gates this process, and the
        log says a restart may not see it."""
        path = self.stop_file
        if path is None:
            return
        from api.runs import kept_run  # noqa: PLC0415

        record, halt = self.recovery, self._carried_halt
        try:
            if record is None and halt is None:
                path.unlink(missing_ok=True)
                return
            stopped = run if run is not None and record is not None and getattr(run, "id", None) == record.run_id \
                else (self.registry.get(record.run_id) if record is not None else None)
            kept = {
                "version": _STOP_FILE_VERSION,
                "written_by": "operator console",
                "profile": self.profile,
                "written_at": time.time(),
                "recovery": record.to_dict() if record is not None else None,
                "run": kept_run(stopped) if stopped is not None else None,
                "halt": {"reason": halt[0], "requested_at": halt[1]} if halt is not None else None,
            }
            path.parent.mkdir(parents=True, exist_ok=True)
            scratch = path.with_name(f"{path.name}.writing")
            scratch.write_text(json.dumps(kept, indent=1), encoding="utf-8")
            os.replace(scratch, path)
        except Exception as exc:  # noqa: BLE001 (the stop gates this process whatever the disk said)
            logger.error("The stop could not be kept in %s (%s: %s): it gates this console, but a restart of the server "
                         "may not see it.", path, type(exc).__name__, exc)

    def _read_the_kept_stop(self) -> None:
        """Read back the stop a console before this one kept (:attr:`stop_file`), once, as this one is built.

        The record stands again UNCLEARED, whatever it said: a restart is no "the cell is clear", and nobody knows what
        happened in the cell meanwhile. The stopped run is known again (Restart replays its plan), and a carried halt
        latches the arm of the next build. A file that does not read gates as a stop of unknown origin, whose way back
        is "the cell is clear", then Home. Nothing is written here: the file changes when the stop next does.
        """
        from api.runs import run_from_kept  # noqa: PLC0415

        path = self.stop_file
        if path is None or not path.exists():
            return
        try:
            kept = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(kept, dict) or kept.get("version") != _STOP_FILE_VERSION:
                raise ValueError(f"no stop file of version {_STOP_FILE_VERSION}")
            record = RecoveryRecord.from_dict(kept["recovery"]) if kept.get("recovery") is not None else None
            run = run_from_kept(kept["run"]) if record is not None and kept.get("run") is not None else None
            halt = kept.get("halt")
            carried: tuple[str, float] | None = None
            if halt is not None:
                reason, at = halt["reason"], halt["requested_at"]
                if not isinstance(reason, str) or isinstance(at, bool) or not isinstance(at, (int, float)):
                    raise TypeError(f"the carried halt reads {halt!r}")
                carried = (reason or "halt requested", float(at))
        except Exception as exc:  # noqa: BLE001 (a stop nobody can read gates as one)
            logger.error("The stop kept in %s could not be read (%s: %s): it gates as a stop of unknown origin until a "
                         "person confirms the cell is clear and Home brings the arm back.", path, type(exc).__name__,
                         exc)
            record = RecoveryRecord(run_id=_UNKNOWN_STOP_RUN, kind=RunKind.PICK, stop_code=StopCode.SOFTWARE_ERROR,
                                    at=time.time(), holding=True)
            run, carried = None, None
        if record is not None and run is not None and run.id == record.run_id:
            self.registry.restore(run)
        with self._lock:
            if record is not None:
                self.recovery = replace(record, cleared_at=None)
                self._stamp("recovery", record.at)
            self._carried_halt = carried
        if carried is not None:
            logger.warning("A halt kept in %s stands (%s): the next build's arm is latched until a person confirms the "
                           "cell is clear.", path, carried[0])
        if record is not None:
            logger.warning("A stop kept in %s stands: %s run %s ended %s where the arm stood. A person confirms the cell "
                           "is clear, then Restart or Home.", path, record.kind, record.run_id, record.stop_code)
            self.hub.publish(
                CELL_STREAM, "cell.recovery", severity=Severity.ERROR,
                human=(f"The {record.kind} run {record.run_id} stopped where the arm stands ({record.stop_code}) before "
                       "the console started again. Nothing moves until a person confirms the cell is clear; then "
                       "Restart or Home, whose first motion is the planned move home."),
                run_id=record.run_id, kind=str(record.kind), stop_code=str(record.stop_code), at=record.at,
            )

    def recovery_gate(self) -> Literal["", "cell_not_cleared", "restart_required"]:
        """What the recovery record refuses now: ``cell_not_cleared`` until a person confirmed the cell is clear, then
        ``restart_required`` (a new task or pick; Restart and Home remain) until a return ends it; ``""`` without one.
        """
        with self._lock:
            record = self.recovery
        if record is None:
            return ""
        return "restart_required" if record.cleared else "cell_not_cleared"


#: How long a Disconnect waits for a planner start to end before it goes on (the sidecar's own ready timeout bounds
#: the start; the arm's disconnect then waits for it as well, and closes what it started).
PLANNER_JOIN_S = 150.0
#: How long a Disconnect waits for a braked move to stand still before it disconnects.
BRAKE_WAIT_S = 2.0
#: Why a Disconnect latched the arm of a moving run it abandoned: what ``CellOut.halted`` and Connect's refusal say.
DISCONNECT_HALT = "the cell was disconnected while a run was moving the arm"


def take_down(console: Console, why: str) -> None:
    """The ordered teardown for Disconnect and the server's shutdown (build plan item 18).

    1. cancel a waiting jaws question (its connect is refused, never "open"), before anything takes the session lock a
       connect holds while the question waits;
    2. end and join a teach (the arm holds at once); the teach is abandoned first, so it ends ``disconnected`` as any
       run the cell is taken down under does, not as a person's Cancel;
    3. join a planner start (it moves nothing): the planner run ends as itself, ready or failed;
    4. abandon the run;
    5. latch the arm of an abandoned moving run (:data:`DISCONNECT_HALT`), so nothing of that run is sent while the cell
       comes down, and give a brake up to 2 s to stand the arm still where the arm brakes a move in flight
       (``robot.ur.brake_on_halt``). The latch stays, and the next Connect waits for "the cell is clear", where a move
       was in flight or the arm braked; on an arm that lets a move run to its end with none in flight it is given back
       once the cell is down, so that Connect connects as it always did. A halt already set (a person's) is kept, and
       so is the teardown's own latch where a person pressed "halt now" while the cell came down;
    6. disconnect.

    Every step is tried, and the disconnect happens whatever an earlier step raised: the one thing an operator must
    always be able to do is put the cell down.
    """
    from api import jaws as browser_jaws  # noqa: PLC0415
    from api.task_run import forget_bins  # noqa: PLC0415

    lent: Any = None
    try:
        try:
            browser_jaws.cancel_pending(console, why)
        finally:
            try:
                _end_a_teach(console, why)
            finally:
                try:
                    _join_a_planner_start(console)
                finally:
                    abandoned = console.registry.abandon(why)
                    if abandoned is not None and abandoned.kind in MOVING_KINDS:
                        lent = _latch_what_moves(console, abandoned)
    finally:
        try:
            console.session.disconnect()
        finally:
            # A bin a task kept stood in the cell that is coming down: the next task looks for it again. A cell with a
            # controller lock forgets it anyway (``remembered_bin``); a dummy or sim cell claims none.
            forget_bins(console, f"the cell was taken down ({why})")
        # Only once the cell is down: a disconnect that raised keeps the latch, and a person says the cell is clear.
        if lent is not None:
            _give_back_the_teardown_latch(console, *lent)


def _end_a_teach(console: Console, why: str) -> None:
    """Step 2: hold an open teach at once and join it (``api.teach.end_and_join``, at most 10 s). A teach run that holds
    the cell is abandoned first, as step 4 abandons any other run: the teach reads the abandon as a Disconnect, holds at
    once, and ends ``disconnected``, the cell taken down under it, never ``cancelled`` as a person's Cancel."""
    from api import teach as browser_teach  # noqa: PLC0415

    active = console.registry.active()
    if active is not None and active.kind is RunKind.TEACH:
        console.registry.abandon(why)
    browser_teach.end_and_join(console, timeout_s=10.0)


def _join_a_planner_start(console: Console) -> None:
    """Wait for a planner run to end before the cell comes down: it moves nothing, and abandoning it would only leave a
    sidecar half started for the disconnect to wait on anyway."""
    active = console.registry.active()
    if active is None or active.kind is not RunKind.PLANNER:
        return
    logger.info("Disconnect waits for planner run %s to end (at most %.0f s); it moves nothing.", active.id,
                PLANNER_JOIN_S)
    if not console.registry.join(active.id, PLANNER_JOIN_S):
        logger.warning("Planner run %s did not end within %.0f s; the disconnect goes on and retires the planner.",
                       active.id, PLANNER_JOIN_S)


def _latch_what_moves(console: Console, run: Any) -> "tuple[Any, int] | None":
    """Step 5: latch the arm of the abandoned moving ``run``, and give a brake up to :data:`BRAKE_WAIT_S`.

    Answers the latch this teardown set, with the count of "halt now" presses before it, where it is to be given back
    once the cell is down (an arm that lets a move run to its end, with none in flight: the latch only kept the run's
    next motion from going out while the cell came down), else ``None``: kept, where a move was in flight or the arm
    brakes, and never touched where a halt was set already, a person's above all, which only "the cell is clear" ends.
    """
    from src.robot.core.arm_capabilities import SupportsHalt, brakes_in_motion_of, halt_state_of  # noqa: PLC0415

    arm = console.session.arm
    if not isinstance(arm, SupportsHalt) or halt_state_of(arm) is not None:
        return None
    brakes = brakes_in_motion_of(arm)
    presses = console.halt_presses
    try:
        state = arm.halt(DISCONNECT_HALT)
    except Exception as exc:  # noqa: BLE001 (the disconnect goes on whatever the latch said)
        logger.error("Latching the arm of run %s before the disconnect failed: %s: %s", run.id, type(exc).__name__, exc)
        return None
    # Carried while it stands, as every latch is: a rebuild or a restart before it is given back ends no halt.
    console.carry_halt(DISCONNECT_HALT, getattr(state, "requested_at", None))
    in_motion = getattr(state, "in_motion", False) is True
    if not brakes:
        if not in_motion:
            logger.info("Run %s was abandoned with no move in flight: the arm is latched while the cell comes down, and "
                        "the latch is given back once it is down.", run.id)
            return state, presses
        logger.warning("Run %s was abandoned with a move in flight: the arm is latched, the move runs to its end and "
                       "nothing after it is sent; a person confirms the cell is clear before it connects again.", run.id)
        return None
    logger.warning("Run %s was abandoned with the arm under way: the arm is latched and its move braked before the "
                   "disconnect; a person confirms the cell is clear before it moves again.", run.id)
    deadline = time.monotonic() + BRAKE_WAIT_S
    while time.monotonic() < deadline:
        latch = halt_state_of(arm)
        if latch is None or latch.brake != "pending":
            return None
        time.sleep(0.02)
    logger.error("The brake of run %s's move did not report the arm still within %.1f s; disconnecting anyway. If it "
                 "still moves, the emergency stop is the answer.", run.id, BRAKE_WAIT_S)
    return None


def _give_back_the_teardown_latch(console: Console, lent: Any, presses: int) -> None:
    """End the latch :func:`_latch_what_moves` set with nothing in flight, now the cell is down: only while it is still
    that very latch (a halt set since is a new record, and is kept) and no person pressed "halt now" since it was set,
    whose press landed on it."""
    from src.robot.core.arm_capabilities import SupportsHalt, halt_state_of  # noqa: PLC0415

    arm = console.session.arm
    if not isinstance(arm, SupportsHalt) or halt_state_of(arm) is not lent:
        return
    if console.halt_presses != presses:
        logger.warning("A person pressed halt now while the cell came down: the arm stays latched until a person "
                       "confirms the cell is clear.")
        return
    try:
        arm.clear_halt()
    except Exception as exc:  # noqa: BLE001 (a latch left set is the safe side: a person says the cell is clear)
        logger.error("Giving back the latch the disconnect set failed (%s: %s); a person confirms the cell is clear "
                     "before the next connect.", type(exc).__name__, exc)
        return
    console.forget_halt()
    logger.info("The cell is down with nothing in flight: the latch the disconnect set is given back.")


_CONSOLE = Console()


def console() -> Console:
    """The process-wide console. A FastAPI dependency, and the seam :func:`set_console` replaces."""
    return _CONSOLE


def set_console(replacement: Console) -> Console:
    """Install a console, e.g. one pointed at a scratch tree. Returns the one it displaced."""
    global _CONSOLE
    previous, _CONSOLE = _CONSOLE, replacement
    return previous
