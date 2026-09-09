"""Bringing a cell up and down from a long-running process, as a transaction.

The CLI runner brings a cell up once and lets process exit take it down. A server cannot: it brings
the same cell up and down repeatedly, in one process, while something else on the machine might also
want it. Three things follow, and each is a rule here rather than a convention:

Connect is all-or-nothing. Arm first (the order is a lifecycle contract: ``VacuumGripper.connect()``
drives digital I/O immediately and ``from_robot_config`` never connects the arm), then gripper. If
the gripper refuses, the arm is disconnected again. There is no half-connected state to render, to
reason about, or to leave a UR controller's single control script held by.

Connecting is motion, and the server makes the operator say so. Robotiq activation is a calibration
sweep of the full finger travel; a vacuum cup's connect asserts the ejector pin immediately and drops
whatever it is holding. A bare post must not be able to cause that, so connect requires a token
issued by a preview that named those specific motions for this cell. A stray curl, a replayed request
or a reloaded tab cannot produce one.

A substituted gripper blocks the connect. ``from_robot_config`` answers a misconfigured end-effector
with a working ``NullGripper``, so the cell would come up, every pick would report success, and the
jaws would close on nothing. The typed reason travels on the gripper, and this refuses on it.

The library refuses it too, in the function this one calls. ``connect_cell`` raises
``NoRealGripper`` on the same record, so a Python caller using ``Cell.connected()`` gets the same
answer this console gives, which it did not before 2026-09-09, when ``real_cell --rehearse``
reported ``3/3 succeeded`` on a cell this console refused to connect at all. What stays here is what
a server has and a library does not: a typed code for the UI, a preview whose token must be
invalidated, and a lock that was already taken. The sentence both print now has one author.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from api.constants import API_LOG_DIR, LIFECYCLE_LOG_FILE
from src.robot.execution.lifecycle import (
    ConnectStage,
    connect_cell,
    disconnect_cell,
    no_real_gripper_reason,
    release_perception,
)
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from src.config.schema.robot.robot_schema import RobotConfig
    from src.robot.execution.cell_lock import CellLock

__all__ = [
    "CellState",
    "release_perception",
    "ConnectRefused",
    "MotionWarning",
    "CellSession",
    "ConnectPreview",
]

logger = create_logger("CellSession", LIFECYCLE_LOG_FILE, log_dir=API_LOG_DIR)

#: How long a connect token stays valid. Long enough to read the warning and decide; short enough that
#: a token found in a log or a stale tab is not a way to move a robot an hour later.
_TOKEN_TTL = timedelta(minutes=5)


class CellState(StrEnum):
    """Where the cell is. Deliberately four states, and there is no ``degraded``.

    A half-connected cell is not a state the console reports, because connect rolls back rather than
    stopping halfway; anything else would mean rendering a cell that looks almost ready and is not.
    """

    DISCONNECTED = "disconnected"
    #: Built but not connected: the service exists, the models are loaded, nothing has been commanded.
    BUILT = "built"
    CONNECTING = "connecting"
    CONNECTED = "connected"


class ConnectRefused(StrEnum):
    """Why a connect did not happen. Each renders as a different thing for the operator to do."""

    #: Nothing has been built yet.
    NOT_BUILT = "not_built"
    #: No valid acknowledgement token: the operator has not seen what will move.
    NOT_ACKNOWLEDGED = "not_acknowledged"
    #: The token expired, or the config changed since it was issued; either way, the warning it
    #: acknowledged is no longer true.
    STALE_TOKEN = "stale_token"
    #: Another process holds the cell.
    CELL_BUSY = "cell_busy"
    #: The build produced a substitute end-effector; connecting would look like it worked.
    NO_REAL_GRIPPER = "no_real_gripper"
    #: The driver refused: payload, tool frame, network. Its own message is carried through.
    DRIVER_REFUSED = "driver_refused"
    #: Already connected, or a connect is already in flight.
    WRONG_STATE = "wrong_state"


@dataclass(frozen=True, slots=True)
class MotionWarning:
    """One thing that will physically move when connect is pressed."""

    #: ``"gripper"`` / ``"arm"``: what moves.
    subject: str
    what: str
    #: What an operator must do about it before pressing.
    precaution: str


@dataclass(frozen=True, slots=True)
class ConnectPreview:
    """What connecting this cell will do, plus the token that acknowledges it."""

    token: str
    expires_at: str
    arm: str
    gripper: str
    warnings: tuple[MotionWarning, ...]
    #: Populated when the build substituted a NullGripper. Non-empty means connect is refused.
    blocking: tuple[str, ...] = ()


def motion_warnings(robot_config: "RobotConfig", gripper: object) -> tuple[MotionWarning, ...]:
    """Everything that physically moves during connect, gated on what was actually built.

    The built gripper holds the veto: a config asking for a Robotiq that fell back to a
    ``NullGripper`` moves nothing, and warning about a finger sweep that cannot happen trains an
    operator to skip the warning that can. Past that veto the branch is chosen by
    ``robot_config.gripper.vendor``, a config string; only the ``jaw_io`` sentence is read off the
    built gripper, and a new branch whose wording depends on the wiring should do the same.
    """
    from src.robot.grippers.null import NullGripper

    warnings: list[MotionWarning] = []
    if isinstance(gripper, NullGripper) or gripper is None:
        return ()

    vendor = str(robot_config.gripper.vendor)
    if vendor == "robotiq":
        warnings.append(MotionWarning(
            subject="gripper",
            what="The Robotiq activates on connect, and activation is a CALIBRATION: the fingers "
                 "travel their entire range, at full speed, immediately.",
            precaution="Keep hands clear of the jaws. Make sure nothing is between them: activation "
                       "with a workpiece in the gripper is how a part gets crushed or launched.",
        ))
    elif vendor == "vacuum":
        warnings.append(MotionWarning(
            subject="gripper",
            what="Connecting a vacuum cup asserts the ejector output immediately, and fires a blow-off "
                 "pulse if one is configured.",
            precaution="If the cup is currently holding anything, it will be RELEASED. Clear the "
                       "workspace below the tool before connecting.",
        ))
    elif vendor == "onrobot":
        # No warning about connecting, which is the honest answer rather than an omission: an RG
        # has no activation stroke, so `connect()` reads two registers and commands nothing. The
        # hazard here is a different one, and it gets its own sentence because no amount of
        # software can detect it.
        warnings.append(MotionWarning(
            subject="gripper",
            what="Connecting an OnRobot RG moves NOTHING: it has no activation stroke. But the "
                 "Modbus unit id is chosen by the MOUNTING, not by the tool, so an RG2-FT in the "
                 "same Quick Changer answers on the same address with an incompatible register map.",
            precaution="Confirm the model in the Compute Box Web Client before running. A wrong "
                       "identity here writes a FORCE value into a WIDTH field, and every reading "
                       "looks healthy.",
        ))
    elif vendor == "jaw_io":
        # This dispatch is hand-kept: a gripper vendor without a branch here gets no warning at all.
        # `JawIOGripper.connect()` calls `_actuate(close=False)`, which opens the jaws, whenever the
        # state reads empty, and unconditionally when there is no feedback and the operator opted
        # in. It is the one vendor whose connect behaviour depends on the wiring.
        #
        # The sentence is read off the built gripper, which is the rule this function's own
        # docstring states and which matters more here than for the other two: with feedback wired
        # the driver refuses to drop a held part, and without it the behaviour is whatever the
        # operator opted into. A flat "the jaws will open" would be false half the time, and a
        # warning that is false half the time is one an operator learns to skip.
        has_feedback = bool(getattr(gripper, "has_feedback", False))
        opts_in = bool(getattr(gripper, "_open_on_connect_without_feedback", False))
        if has_feedback:
            warnings.append(MotionWarning(
                subject="gripper",
                what="Connecting the jaws reads the end-stop switches first. If they report EMPTY the "
                     "jaws are opened immediately; if they report a part, the driver holds it and "
                     "warns instead of dropping it.",
                precaution="Keep hands clear of the jaws. If the switches are miswired an empty "
                           "reading is what a held part looks like.",
            ))
        elif opts_in:
            warnings.append(MotionWarning(
                subject="gripper",
                what="Connecting OPENS THE JAWS UNCONDITIONALLY. This cell has no end-stop feedback "
                     "wired and open_on_connect_without_feedback is set, so the driver cannot check "
                     "whether anything is held first.",
                precaution="Anything between the jaws WILL BE DROPPED where the arm is standing. "
                           "Clear the tool before connecting.",
            ))
    return tuple(warnings)


@dataclass
class CellSession:
    """The built cell and its connection, owned by one console process.

    Not thread-safe by accident: every transition takes ``_lock``, because a browser with two tabs is
    two concurrent requests, and "connect" is not an operation that tolerates being started twice.
    """

    #: The built :class:`AutonomousGraspService`, or ``None`` before a build.
    service: Any = None
    state: CellState = CellState.DISCONNECTED
    #: The cross-process lock, held only while connected.
    cell_lock: "CellLock | None" = None
    #: Set by :meth:`preview`, consumed by :meth:`connect`.
    _token: str | None = None
    _token_expires: datetime | None = None
    #: Fingerprint of the config the token was issued against; a write invalidates the token.
    _token_fingerprint: str = ""
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    # --- what was built ----------------------------------------------------------------------------

    @property
    def arm(self) -> Any:
        orchestrator = getattr(getattr(self.service, "runtime", None), "orchestrator", None)
        return getattr(orchestrator, "arm", None)

    @property
    def gripper(self) -> Any:
        orchestrator = getattr(getattr(self.service, "runtime", None), "orchestrator", None)
        return getattr(orchestrator, "gripper", None)

    @property
    def substitution(self) -> Any:
        """The reason a real end-effector could not be built, or ``None``."""
        return getattr(self.gripper, "substitution", None)

    @property
    def connected(self) -> bool:
        return self.state is CellState.CONNECTED

    # --- transitions -------------------------------------------------------------------------------

    def may_adopt(self) -> None:
        """Refuse a rebuild that would orphan a live connection. Raises; returns nothing.

        Split out of :meth:`adopt` so the caller can ask before it spends anything. A real build opens
        a camera and loads two models onto the GPU, and doing that first and asking afterwards means a
        refused rebuild has already claimed the device it was refused for.
        """
        with self._lock:
            if self.state in (CellState.CONNECTED, CellState.CONNECTING):
                logger.warning("Rebuild refused: the cell is %s, not idle.", self.state.value)
                raise CellTransitionError(
                    ConnectRefused.WRONG_STATE,
                    "the cell is connected; disconnect before rebuilding. Rebuilding under a live "
                    "connection would leave the old arm and gripper connected with nothing holding "
                    "them.",
                )

    def adopt(self, service: Any) -> None:
        """Install a freshly built service, releasing the one it replaces.

        The release is the point. A build opens a camera; nothing else in this server ever closes
        one. Overwriting ``self.service`` without releasing drops the old one on the floor with its
        ``rs.pipeline`` still streaming, and on real hardware the next build's ``pipeline.start()``
        then fails on the device the abandoned one is holding, so an ordinary loop (build, get
        refused, fix the config, build again) leaves the console unable to rebuild until the process
        is restarted. The old service also sits in a reference cycle
        (``service.runtime.orchestrator``), so refcounting would not run a destructor even if one
        existed.
        """
        with self._lock:
            self.may_adopt()
            previous, self.service = self.service, service
            self.state = CellState.BUILT
            self._invalidate_token()
            if previous is not None:
                logger.info("Replacing the built cell; the previous one's device is handed back now.")
        # Outside the lock: closing a device can block, and nothing else may wait on this session
        # while it does.
        release_perception(previous)

    def release(self) -> None:
        """Take the cell down and close its camera. For process shutdown, not for a rebuild."""
        self.disconnect()
        with self._lock:
            previous, self.service = self.service, None
            self.state = CellState.DISCONNECTED
        logger.info("Cell released: nothing is built, nothing is connected.")
        release_perception(previous)

    def preview(self, robot_config: "RobotConfig", fingerprint: str) -> ConnectPreview:
        """Describe what connecting will move, and issue the token that acknowledges it."""
        with self._lock:
            if self.service is None:
                logger.warning("Connect preview refused: nothing is built yet.")
                raise CellTransitionError(
                    ConnectRefused.NOT_BUILT, "nothing is built yet; build the cell first."
                )
            substitution = self.substitution
            blocking: tuple[str, ...] = ()
            if substitution is not None:
                blocking = (f"{substitution.detail} {substitution.fix}",)

            token = secrets.token_urlsafe(24)
            expires = datetime.now(timezone.utc) + _TOKEN_TTL
            self._token, self._token_expires, self._token_fingerprint = token, expires, fingerprint
            warnings = motion_warnings(robot_config, self.gripper)
            # The token itself is never logged. It is the thing that authorises a motion, and a log
            # file is read by more people, and kept longer, than a browser tab.
            logger.info(
                "Connect preview issued for arm=%s gripper=%s: %d motion warning(s), %d blocker(s); "
                "config fingerprint %s, valid until %s.",
                type(self.arm).__name__, type(self.gripper).__name__, len(warnings), len(blocking),
                fingerprint, expires.isoformat(timespec="seconds"),
            )
            return ConnectPreview(
                token=token,
                expires_at=expires.isoformat(timespec="seconds"),
                arm=type(self.arm).__name__,
                gripper=type(self.gripper).__name__,
                warnings=warnings,
                blocking=blocking,
            )

    def connect(self, token: str, fingerprint: str, cell_lock: "CellLock | None") -> None:
        """Bring the cell up as a transaction, or leave it exactly as it was.

        ``cell_lock`` is acquired by the caller (it needs the config to know the key) and handed over
        here. This method releases it on the path that reached the driver; the refusals raised before
        that (wrong state, bad or stale token, substituted gripper) leave it held, which is why the
        caller releases it as well. A lock held by a cell that is not up is a lock nobody can explain.
        """
        from src.robot.execution.cell_lock import CellBusy  # noqa: F401 (documents the caller)

        with self._lock:
            if self.state is not CellState.BUILT:
                logger.warning(
                    "Connect refused: the cell is %s, not built-and-idle.", self.state.value
                )
                raise CellTransitionError(
                    ConnectRefused.WRONG_STATE,
                    f"the cell is {self.state.value}, not built-and-idle; nothing to connect.",
                )
            self._check_token(token, fingerprint)
            if (substitution := self.substitution) is not None:
                # The refusal an operator is most likely to argue with, so the log keeps the reason
                # rather than the verdict: a NullGripper cell connects fine and picks nothing.
                logger.warning(
                    "Connect refused: no real gripper (%s). %s",
                    substitution.reason, substitution.detail,
                )
                # The sentence is the library's, and so is the rule. `connect_cell` refuses this
                # same gripper underneath (`NoRealGripper`), so this branch is what the console has
                # that the library does not: a typed code for the UI, a preview to invalidate and a
                # lock it never took, rather than a second answer to the same question. The wording
                # came from here; keeping a copy of it here is how the two would drift.
                raise CellTransitionError(
                    ConnectRefused.NO_REAL_GRIPPER, no_real_gripper_reason(substitution),
                )

            self.state = CellState.CONNECTING
            self.cell_lock = cell_lock
            arm, gripper = self.arm, self.gripper
            logger.info(
                "Connecting: arm=%s, then gripper=%s. THIS MOVES; acknowledged by token.",
                type(arm).__name__, type(gripper).__name__,
            )
            def _narrate(stage: ConnectStage) -> None:
                """The console's voice. `connect_cell` owns the order and the rollback."""
                if stage is ConnectStage.ARM_CONNECTED:
                    logger.info("Arm connected (%s).", type(arm).__name__)
                elif stage is ConnectStage.GRIPPER_CONNECTED:
                    # This moves. The preview said so and the token proves the operator read it.
                    logger.info("Gripper connected (%s).", type(gripper).__name__)

            try:
                # One implementation, shared with the CLI, for the same reason the teardown is: two
                # copies of the same transaction drift, and they drift first in the half that
                # decides whether the gripper comes down at all. `connect_cell` owns
                # arm-then-gripper and the roll-back-on-refusal; this method keeps the token, the
                # lock and the state machine, which are the console's.
                connect_cell(arm, gripper, announce=_narrate)
            except BaseException as exc:
                # The arm is already rolled back by `connect_cell`. What is left here is the state
                # this console keeps around it.
                logger.error("Connect FAILED (%s: %s); rolled back.", type(exc).__name__, exc)
                if cell_lock is not None:
                    cell_lock.release()
                self.cell_lock = None
                self.state = CellState.BUILT
                self._invalidate_token()
                if isinstance(exc, Exception):
                    raise CellTransitionError(
                        ConnectRefused.DRIVER_REFUSED, f"{type(exc).__name__}: {exc}"
                    ) from exc
                raise  # KeyboardInterrupt / SystemExit propagate, rolled back first
            self.state = CellState.CONNECTED
            logger.info("Cell connected.")
            self._invalidate_token()

    def disconnect(self) -> None:
        """Take the cell down. Idempotent, and it releases the cross-process lock.

        Gripper first, then arm: the reverse of connect. A vacuum cup's ``disconnect`` releases its
        output, and doing that while the arm is still up means the release actually reaches the I/O.
        """
        with self._lock:
            was = self.state
            # One implementation, shared with the CLI. Two copies of this teardown drift, and they
            # drift first over which end comes down and whether the gripper comes down at all;
            # `disconnect_cell` is the order, once, for this method and for
            # `real_cell/__main__.py`.
            #
            # No `service` argument, on purpose. `disconnect_cell` also closes the cameras when
            # given one, and this console deliberately keeps them open across a disconnect: it
            # releases them on rebuild instead (see the two `release_perception` calls above), so a
            # reconnect does not pay to reopen a device it never gave up.
            teardown = disconnect_cell(self.arm, self.gripper)
            if self.cell_lock is not None:
                self.cell_lock.release()
                self.cell_lock = None
            self.state = CellState.BUILT if self.service is not None else CellState.DISCONNECTED
            if not teardown.clean:
                logger.warning("Teardown was NOT clean: %s", teardown.to_dict()["detail"])
            if was is CellState.CONNECTED:
                # Only when something was actually up: disconnect() is idempotent and is called on
                # every release, and logging the no-op would bury the transition that mattered.
                logger.info("Cell disconnected (gripper first, then arm); now %s.", self.state.value)
            self._invalidate_token()

    # --- token ------------------------------------------------------------------------------------

    def _check_token(self, token: str, fingerprint: str) -> None:
        if not self._token or not token or not secrets.compare_digest(token, self._token):
            raise CellTransitionError(
                ConnectRefused.NOT_ACKNOWLEDGED,
                "connect needs a token from GET /v1/cell/connect-preview. The preview names what will "
                "physically move on THIS cell, and the token is the record that someone read it.",
            )
        if self._token_expires is None or datetime.now(timezone.utc) > self._token_expires:
            raise CellTransitionError(
                ConnectRefused.STALE_TOKEN,
                "that acknowledgement has expired. Read the preview again; the cell may not be in the "
                "state it was in when you last looked.",
            )
        if fingerprint != self._token_fingerprint:
            raise CellTransitionError(
                ConnectRefused.STALE_TOKEN,
                "the configuration changed after that acknowledgement was issued, so what it described "
                "is no longer what will happen. Read the preview again.",
            )

    def _invalidate_token(self) -> None:
        self._token, self._token_expires, self._token_fingerprint = None, None, ""


class CellTransitionError(RuntimeError):
    """A refused transition, carrying the typed reason the console renders on."""

    def __init__(self, reason: ConnectRefused, message: str) -> None:
        super().__init__(message)
        self.reason = reason
