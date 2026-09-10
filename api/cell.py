"""The console's session: which config tree, which profile chain, and whether a run owns the cell.

One place touches the loader's global profile. ``set_active_profile`` mutates process state, so a
request handler doing it inline would let two concurrent requests read each other's chain and report
a UR3e verdict for a simulator tree. Everything here funnels through :meth:`Console.config`, under a
lock.

One place holds the run flag. Configuration may not be written while the cell is moving, and that
rule is the reason the write endpoint is allowed to exist at all. :meth:`Console.require_idle` is
the guard; ``RunRegistry`` in ``api/runs.py`` is the only thing that sets ``active_run_id``.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from api.constants import API_LOG_DIR, CELL_LOG_FILE
from api.events import EventHub
from api.lifecycle import CellSession
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

__all__ = ["Console", "NoRobotConfigured", "RunLocked", "console", "set_console"]

logger = create_logger("Console", CELL_LOG_FILE, log_dir=API_LOG_DIR)

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / "config"


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
    #: The text prompt a real perception source is built with. It lives here because
    #: ``build_real_components`` needs one at construction time; a run's own prompt reaches the
    #: built service later, through ``set_target_label``.
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
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # --- the cell ----------------------------------------------------------------------------------

    def __post_init__(self) -> None:
        # The registry needs the hub, and a `default_factory` cannot see a sibling field. Built here
        # rather than lazily so `console().runs` is never None for a caller.
        if self.runs is None:
            from api.runs import RunRegistry

            self.runs = RunRegistry(self.hub)

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
            # One call, and the same one the CLI runner makes. Building the perception and grasp
            # side here and then calling `from_robot_config` separately is how a browser and a
            # terminal come to disagree about what "the cell" is.
            service = (build_rehearsal_cell(robot) if rehearse
                       else build_real_cell(robot, prompt=self.prompt, app_config=app_cfg))
            # Record logging is off in the shipped config (`grasping.record_log_path: null`), so a
            # console-driven cell would keep no history at all and a bring-up that goes wrong would
            # leave nothing to diagnose from afterwards. The console therefore turns it on itself,
            # into its own file, without touching the YAML: a cell that never runs the console is
            # unaffected, and an operator who did configure a path keeps it.
            #
            # `from_robot_config` has already wired the configured path when there is one; calling
            # enable_record_logging again would replace it, so this only fills a gap.
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
            self.session.adopt(service)
            logger.info(
                "Cell built in %.1f s: arm=%s, gripper=%s, records -> %s.",
                time.perf_counter() - started,
                type(self.session.arm).__name__,
                type(self.session.gripper).__name__,
                self.record_log_path,
            )
            return service

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

        return run_config_preflight(self.robot())

    def require_idle(self) -> None:
        """Guard for anything that changes the cell. Raises :class:`RunLocked`, never queues."""
        with self._lock:
            if self.active_run_id is not None:
                raise RunLocked(self.active_run_id)


_CONSOLE = Console()


def console() -> Console:
    """The process-wide console. A FastAPI dependency, and the seam :func:`set_console` replaces."""
    return _CONSOLE


def set_console(replacement: Console) -> Console:
    """Install a console, e.g. one pointed at a scratch tree. Returns the one it displaced."""
    global _CONSOLE
    previous, _CONSOLE = _CONSOLE, replacement
    return previous
