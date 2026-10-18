"""Taking a cell up and down, once, so the CLI and the console cannot do it differently.

Up is a transaction: arm first, then gripper, and a gripper that refuses rolls the arm back rather
than leaving a UR controller's single control script held with a payload already pushed. Down is
the exact reverse, then the cameras, then the lock.

Gripper before arm on the way down is what makes a release reach the I/O: a vacuum cup's
``disconnect`` releases its output, and a teardown that skips the gripper leaves a vacuum line
asserted after the run ends. The cameras come down in the same call, because on real hardware a
second ``pipeline.start()`` on a streaming device fails and a run that closes nothing cannot be
repeated.

Teardown reports, it never raises, and it is never silent. ``except Exception: pass`` around a
disconnect leaves a gripper that would not release with no trace anywhere. Every step's outcome is
on :class:`TeardownReport` instead, so a caller that ignores it still cannot make it silent.

``ConnectedCell`` holds a pick service and ``ConnectedRobot`` holds an arm and a gripper with no
service. Both run the same enter and the same exit (``_bring_up`` and ``_take_down``), so a robot
connected for calibration takes the lock and keeps the order a cell connected for a pick does.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import TracebackType
from typing import Any, Callable

from src.robot.constants import ROBOT_LOG_FILE, create_robot_logger
from src.robot.core.camera_world import live_planner_world_of

__all__ = [
    "ConnectedCell",
    "ConnectedRobot",
    "ConnectStage",
    "NoRealGripper",
    "StepOutcome",
    "TeardownReport",
    "WatchedRobot",
    "connect_cell",
    "disconnect_cell",
    "no_real_gripper_reason",
    "release_perception",
    "service_cameras",
]

logger = create_robot_logger("ConnectedCell", ROBOT_LOG_FILE)


class NoRealGripper(RuntimeError):
    """This cell has no end-effector and one was asked for. Raised instead of connecting.

    Measured before this existed: ``real_cell --rehearse --runs 3`` on the shipped ``robot.yaml``
    printed ``gripper  NullGripper`` and then ``RESULT: 3/3 succeeded``. Five configuration
    combinations cannot produce the end-effector they name, one per ``SubstitutionReason`` member,
    and none of them raises: the build hands back a working ``NullGripper``, which accepts every
    commanded width and answers ``get_width_mm()`` with the configured maximum forever.

    No pick path refuses it for you. The grasp verifier that refused a substituted gripper by name
    (``WidthDeltaGripperVerifier``) was removed on 2026-09-29 with the post-grasp verification stage,
    which no pick path ran, and the execution policy reads a close's hold from the gripper's own
    ``is_object_detected`` and ``hold_evidence``, which a ``NullGripper`` does not claim, so its close
    is reported unmeasured rather than empty. This refusal is what guards the cell.

    An ``Exception``, for the reason ``CellBuildRefused`` is one: both callers of this module catch
    ``Exception`` and turn it into an operator-facing refusal (``api/routers/cell.py`` answers the
    console, ``real_cell/__main__.py`` returns its configuration exit code). A ``BaseException``
    would walk past both and arrive as a 500 with no message and a bare traceback at a bench.
    """


class StepOutcome(StrEnum):
    """What happened to one part of a teardown."""

    #: It was there and it went down.
    RELEASED = "released"
    #: There was nothing to release. A cell with no gripper, a perception source owning no device.
    ABSENT = "absent"
    #: It was there and it refused. The reason is in the report's ``detail``.
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class TeardownReport:
    """What actually came down, per part. Satisfies both halves of the report contract.

    Teardown must not raise, or it masks whatever result was being reported when it ran. A failure
    still has a place to go: a value that says ``gripper=FAILED`` cannot be swallowed by a bare
    ``except Exception: pass``.
    """

    gripper: StepOutcome
    arm: StepOutcome
    perception: StepOutcome
    #: Per part, why it failed. Empty for parts that did not.
    detail: tuple[tuple[str, str], ...] = ()

    @property
    def clean(self) -> bool:
        """Did every part that existed come down?"""
        return not any(o is StepOutcome.FAILED for o in (self.gripper, self.arm, self.perception))

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """One line per part, ASCII, no trailing newline."""
        reasons = dict(self.detail)
        lines = [
            f"  {part:<11}{outcome.value}"
            + (f": {reasons[part]}" if part in reasons else "")
            for part, outcome in (
                ("gripper", self.gripper), ("arm", self.arm), ("perception", self.perception),
            )
        ]
        if not self.clean:
            lines.append("  ^ something did not come down. On a real cell that can mean an output "
                         "is still asserted.")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "gripper": self.gripper.value,
            "arm": self.arm.value,
            "perception": self.perception.value,
            "clean": self.clean,
            "detail": {part: why for part, why in self.detail},
        }


def release_perception(service: Any) -> StepOutcome:
    """Close whatever device a built service opened. Duck-typed, idempotent, never raises.

    Duck-typed because only some perception sources own a device: the rehearsal source owns nothing,
    the Isaac sources belong to a simulator this process does not own, and only the RealSense adapter
    has a streamer to hand back. A source without ``close()`` is not an error, it is the normal case.

    Every camera, not the primary one. A fused cell also opens a device for every other rig it
    fuses, each held by its ``Camera`` owner, so both the
    ``perception`` and ``multi_camera_perception`` slots are walked. On real hardware a second
    ``pipeline.start()`` on a streaming device fails, so a camera left open makes the next build of
    a multi-camera cell impossible without a restart.

    A cell whose live planner world takes a rig that neither of those holds opens it for the world
    alone and keeps it in ``planner_world_cameras``, the third slot, so a rebuild meets none of
    them still held.
    """
    if service is None:
        return StepOutcome.ABSENT
    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    outcome = StepOutcome.ABSENT
    for slot in ("perception", "multi_camera_perception", "planner_world_cameras"):
        holder = getattr(orchestrator, slot, None)
        close = getattr(holder, "close", None)
        if not callable(close):
            continue
        try:
            close()
            logger.info("Closed the perception device(s) held by %s.", type(holder).__name__)
            if outcome is StepOutcome.ABSENT:
                outcome = StepOutcome.RELEASED
        except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
            # A camera that would not close is the reason the next build cannot open one, so the
            # failure is logged as well as reported.
            logger.warning(
                "Closing the perception device (%s) failed: %s: %s",
                type(holder).__name__, type(exc).__name__, exc,
            )
            outcome = StepOutcome.FAILED
    return outcome


def service_cameras(service: Any) -> tuple[Any, ...]:
    """The camera owners a built service holds, each once: the primary first, then the fused ones, then those the
    planner world opened for itself. The three slots :func:`release_perception` gives back, read and not closed.

    Duck-typed as that function is, so this module loads no camera code: an owner is what a RealSense source's
    ``streamer.camera`` is, or what ``planner_world_cameras`` holds, and it answers ``rig_id`` with a string and has a
    ``peek`` to look through. A rehearsal's source holds none, and a double that answers every attribute holds none
    either. Nothing is opened, closed or read.
    """
    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    found: list[Any] = []

    def keep(camera: Any) -> None:
        if (isinstance(getattr(camera, "rig_id", None), str) and callable(getattr(camera, "peek", None))
                and all(camera is not held for held in found)):
            found.append(camera)

    keep(getattr(getattr(getattr(orchestrator, "perception", None), "streamer", None), "camera", None))
    fused = getattr(getattr(orchestrator, "multi_camera_perception", None), "sources", None)
    if isinstance(fused, Mapping):
        for source in fused.values():
            keep(getattr(getattr(source, "streamer", None), "camera", None))
    opened = getattr(getattr(orchestrator, "planner_world_cameras", None), "cameras", None)
    if isinstance(opened, (tuple, list)):
        for camera in opened:
            keep(camera)
    return tuple(found)


def no_real_gripper_reason(substitution: Any) -> str:
    """The one sentence both refusals say, built from the record the build attached.

    One sentence, two doors. ``api/lifecycle.py`` refuses this connect with its own typed
    ``ConnectRefused.NO_REAL_GRIPPER`` (it has a preview and a token to invalidate, which this
    module knows nothing about) and ``connect_cell`` refuses it underneath. Those are two renderings
    of one rule, and the prose was the part most likely to drift into being two rules: the console's
    copy already carried the "close on nothing" clause that makes the refusal arguable with, and a
    second hand-written copy here would have been the version the command line printed.
    """
    detail = str(getattr(substitution, "detail", "") or "")
    fix = str(getattr(substitution, "fix", "") or "")
    return (f"{detail} Connecting would look like it worked: the cell would come up, every pick "
            f"would report success, and the gripper would close on nothing. {fix}")


class ConnectStage(StrEnum):
    """Where a connect has got to, for a caller that narrates it.

    A stage, not a sentence: the stage is what happened and the caller owns how it reads, so this
    module never authors text it cannot see.

    ``GRIPPER_MOVING`` fires before the call, which is why the hook exists. Robotiq activation is a
    calibration sweep of the full finger travel, so the warning has to reach a person at the bench
    before the fingers move, not in a summary afterwards.
    """

    ARM_CONNECTED = "arm_connected"
    #: About to connect the gripper. This moves.
    GRIPPER_MOVING = "gripper_moving"
    GRIPPER_CONNECTED = "gripper_connected"


def connect_cell(
    arm: Any,
    gripper: Any,
    *,
    announce: "Callable[[ConnectStage], None] | None" = None,
) -> None:
    """Bring a cell up as a transaction: arm, then gripper, or neither.

    Arm first is a contract. ``VacuumGripper.connect()`` drives digital I/O the moment it runs, and
    ``from_robot_config`` never connects the arm (the caller owns the lifecycle), so the reverse
    order commands the vacuum line with no arm to command it through.

    A gripper that refuses rolls the arm back, and so does any failure after the arm came up, an announce
    that raises included; one after the gripper came up takes the gripper down first. Leaving a
    half-connected cell means a UR controller's single control script is held by a process that has
    already reported a failure.

    A cell whose end-effector could not be built does not come up at all. Measured on this tree:
    ``real_cell --rehearse --runs 3`` on the shipped ``robot.yaml`` printed ``gripper  NullGripper``
    at build and ``RESULT: 3/3 succeeded`` at the end. The substituted gripper accepts every
    commanded width and answers ``get_width_mm()`` with the configured 85.0 mm maximum forever, so
    nothing downstream disagrees: the grasp verifier that refused it by name left on 2026-09-29 with
    the post-grasp verification stage, and the pick reads its hold as unmeasured. The fact is
    decidable before the arm is commanded, so it is decided there.

    Connecting is motion. Robotiq activation is a calibration sweep of the full finger travel; a
    vacuum cup's connect asserts the ejector pin immediately and drops whatever it is holding. This
    function does not ask whether that is allowed. Its callers do: the console with an
    acknowledgement token, the CLI with a printed banner.

    Once both are up, an arm that reads the hand it carries (``set_hand``, the UR driver) is handed it, ``None`` for an
    arm alone: the UR driver runs what it judged ahead as judged only while that hand reads as it did then.
    """
    # No success logging here, on purpose: both callers already narrate a successful connect, so a
    # third line from this module would appear in the console's log twice. Failures are logged here,
    # because a rollback that did not take must not depend on a caller remembering to write it.
    say = announce or (lambda _stage: None)
    # Before the first motion of the run, and read off the record rather than off a class, so any
    # driver that grows the same field participates and this module keeps knowing nothing about the
    # gripper package. `gripper.vendor: none` is the same jawless object with `substitution=None`
    # and is a legitimate cell (a calibration rig, a camera-only bring-up); it still connects.
    substitution = getattr(gripper, "substitution", None)
    if substitution is not None:
        # The refusal an operator is most likely to argue with, so the log keeps the reason rather
        # than the verdict: a substituted cell connects fine and picks nothing.
        logger.error("Connect refused: no real gripper (%s). %s",
                     getattr(substitution, "reason", ""), getattr(substitution, "detail", ""))
        raise NoRealGripper(no_real_gripper_reason(substitution))
    arm.connect()
    hand_to = getattr(arm, "set_hand", None)
    # Every announce is inside the transaction. One that ran before the `try` and raised (a bench `print` on a closed
    # pipe, or any hook a library caller passes) left the arm connected and its control script held, and one that
    # raised after the gripper came up rolled back the arm alone.
    gripper_asked = gripper_up = False
    try:
        say(ConnectStage.ARM_CONNECTED)
        if gripper is None:
            if callable(hand_to):
                hand_to(None)
            return
        say(ConnectStage.GRIPPER_MOVING)
        gripper_asked = True
        gripper.connect()
        gripper_up = True
        say(ConnectStage.GRIPPER_CONNECTED)
        if callable(hand_to):
            hand_to(gripper)
    except BaseException:
        if gripper_up:
            # Gripper first, the order a cell comes down in: a vacuum cup's disconnect releases its output, and it
            # reaches the I/O only while the arm is still up.
            logger.error("Connect failed after the gripper came up; rolling the gripper back, then the arm.")
            try:
                gripper.disconnect()
            except Exception as rollback:  # noqa: BLE001 (must not mask the refusal being reported)
                logger.error("Rollback of the gripper also failed, its output may still be asserted: %s: %s",
                             type(rollback).__name__, rollback)
        elif gripper_asked:
            logger.error("Gripper connect failed; rolling the arm back to disconnected.")
        else:
            logger.error("Connect failed after the arm came up; rolling the arm back to disconnected.")
        try:
            arm.disconnect()
        except Exception as rollback:  # noqa: BLE001 (must not mask the refusal being reported)
            # The rollback itself did not take, so the controller may still hold a control script
            # this process no longer tracks.
            logger.error("Rollback disconnect also failed: %s: %s",
                         type(rollback).__name__, rollback)
        raise


def disconnect_cell(arm: Any, gripper: Any, service: Any = None) -> TeardownReport:
    """Take a cell down in the reverse order, and say what came down. Never raises.

    Gripper first, then arm, then the cameras. A vacuum cup's ``disconnect`` releases its output, and
    doing that while the arm is still up is what makes the release reach the I/O.

    The lock is not released here. It belongs to whoever acquired it, and the two callers hold it
    differently: the console keeps one across many requests, the CLI holds one for a run.
    :class:`ConnectedCell` releases it for the caller that used the context manager.
    """
    detail: list[tuple[str, str]] = []

    def _down(part: str, handle: Any) -> StepOutcome:
        if handle is None:
            return StepOutcome.ABSENT
        try:
            handle.disconnect()
        except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
            logger.warning("%s disconnect failed: %s: %s", part, type(exc).__name__, exc)
            detail.append((part, f"{type(exc).__name__}: {exc}"))
            return StepOutcome.FAILED
        return StepOutcome.RELEASED

    gripper_outcome = _down("gripper", gripper)
    arm_outcome = _down("arm", arm)
    perception_outcome = release_perception(service)
    if perception_outcome is StepOutcome.FAILED:
        detail.append(("perception", "see the log for the device that refused"))

    report = TeardownReport(gripper_outcome, arm_outcome, perception_outcome, tuple(detail))
    logger.info("Cell down (gripper first, then arm, then cameras): %s",
                "clean" if report.clean else "not clean")
    return report


def _bring_up(session: "ConnectedCell | ConnectedRobot") -> None:
    """The one enter: take the lock, then :func:`connect_cell`, and give the lock back on a refusal."""
    if session.lock is not None:
        session.lock.acquire()
    try:
        connect_cell(session.arm, session.gripper, announce=session.announce)
    except BaseException:
        # `connect_cell` has already rolled the arm back; the lock is this object's to give up.
        if session.lock is not None:
            session.lock.release()
        raise


def let_go_of_held_views(arm: Any) -> None:
    """Drop the frames a wrist camera's ``Locator.look_around`` held in ``arm``'s live planner world
    (``LivePlannerWorld.forget_pick_views``), where it holds any; an arm with no live world holds none.

    ``Robot.pick`` calls it when the pick ends, and the one exit of a ``with`` block (:func:`_take_down`) before the
    cell comes down, so a reconnect in the same process starts with no frames of a look around it did not take. Never
    raises: a pick's report and a teardown are what the caller reads, and a world that would not let go is logged.
    """
    world = live_planner_world_of(arm)
    forget = getattr(world, "forget_pick_views", None)
    if not callable(forget):
        return
    try:
        forget()
    except Exception as exc:  # noqa: BLE001 (a report or a teardown is not to be masked by it)
        logger.warning("the live planner world did not let go of the frames of the looks: %s: %s",
                       type(exc).__name__, exc)


def _take_down(session: "ConnectedCell | ConnectedRobot", service: Any) -> None:
    """The one exit: the frames a look around held let go (:func:`let_go_of_held_views`), then :func:`disconnect_cell`,
    its report kept on the session, then the lock."""
    let_go_of_held_views(session.arm)
    session.teardown = disconnect_cell(session.arm, session.gripper, service)
    if session.lock is not None:
        session.lock.release()


@dataclass
class ConnectedCell:
    """A connected cell, for the duration of a ``with`` block.

        with ConnectedCell(service, lock=cell_lock) as session:
            report = session.service.pick()
        print(session.teardown.render())

    The exit is the point: teardown runs after a refusal, after an exception and after a keyboard
    interrupt without a caller remembering to write it.
    """

    service: Any
    #: Acquired before anything is touched and released after everything is down. ``None`` for a cell
    #: that owns no controller (a rehearsal, a simulator), which is why two of those can run at once.
    lock: Any = None
    #: Narration hook, forwarded to :func:`connect_cell`. See :class:`ConnectStage`.
    announce: "Callable[[ConnectStage], None] | None" = None
    #: Filled in on exit. ``None`` while the block is running.
    teardown: TeardownReport | None = None
    _live: Any = field(default=None, repr=False)

    @property
    def arm(self) -> Any:
        return getattr(self._orchestrator, "arm", None)

    @property
    def gripper(self) -> Any:
        return getattr(self._orchestrator, "gripper", None)

    @property
    def _orchestrator(self) -> Any:
        return getattr(getattr(self.service, "runtime", None), "orchestrator", None)

    def halt_every_arm(self, reason: str) -> None:
        """Halt this cell's arm: what ``together()`` does where a step fails.

        Args:
            reason (str): Why, as the arm's latch keeps it.
        """
        from src.robot.execution.together import halt_arms  # noqa: PLC0415

        halt_arms((self.arm,), reason)

    def __enter__(self) -> "ConnectedCell":
        from src.robot.execution.together import live_cell  # noqa: PLC0415

        live = live_cell(self)
        live.__enter__()
        try:
            _bring_up(self)
        except BaseException:
            live.__exit__(None, None, None)
            raise
        self._live = live
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            _take_down(self, self.service)
        finally:
            live, self._live = self._live, None
            if live is not None:
                live.__exit__(None, None, None)


@dataclass
class ConnectedRobot:
    """An arm and its gripper, connected for the duration of a ``with`` block, with no pick service.

        with ConnectedRobot(arm, gripper, lock=cell_lock) as live:
            live.gripper.set_width_mm(40.0)
        print(live.teardown.render())

    The enter and the exit are the ones :class:`ConnectedCell` runs: the lock, the refusal of a
    substituted gripper, the arm before the gripper, and the reverse on the way out. With no service
    there is no camera to close, so the teardown reports perception as absent.
    """

    arm: Any
    #: ``None`` for an arm-only robot, which connects the arm alone.
    gripper: Any
    #: Acquired before anything is touched and released after everything is down. ``None`` for a
    #: robot that owns no controller.
    lock: Any = None
    #: Narration hook, forwarded to :func:`connect_cell`. See :class:`ConnectStage`.
    announce: "Callable[[ConnectStage], None] | None" = None
    #: Filled in on exit. ``None`` while the block is running.
    teardown: TeardownReport | None = None

    def __enter__(self) -> "ConnectedRobot":
        _bring_up(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        _take_down(self, None)


@dataclass
class WatchedRobot:
    """An arm watched, not connected, for the duration of a ``with`` block (P0i, owner 2026-09-26).

        with WatchedRobot(arm, lock=cell_lock):
            status = arm.get_robot_status()

    The door a person clears a fault through: the cell lock first, as :class:`ConnectedRobot` takes it, then
    ``SupportsWatching.watch()``, a link that reads the controller's state and releases a protective stop and never
    starts a control program, so it opens on the stopped controller a full connect is refused by (measured on URSim,
    2026-09-26). Every motion verb of the arm refuses meanwhile, since it is not connected. On the way out, on every way
    out, the arm is disconnected and the lock given back. An arm that cannot be watched is refused before the lock is
    taken.
    """

    arm: Any
    #: Acquired before the watch opens and released after it is closed. ``None`` for an arm that owns no controller.
    lock: Any = None

    def __enter__(self) -> "WatchedRobot":
        from src.robot.core import SupportsWatching

        watched = self.arm
        if not isinstance(watched, SupportsWatching):
            raise TypeError(
                f"{type(watched).__name__} cannot be watched without being driven (SupportsWatching): its status "
                "and its release need a connection, which a stopped controller refuses")
        if self.lock is not None:
            self.lock.acquire()
        try:
            watched.watch()
        except BaseException:
            try:
                self.arm.disconnect()
            finally:
                if self.lock is not None:
                    self.lock.release()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        try:
            self.arm.disconnect()
        finally:
            if self.lock is not None:
                self.lock.release()
