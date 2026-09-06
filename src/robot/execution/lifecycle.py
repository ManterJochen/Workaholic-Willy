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
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import TracebackType
from typing import Any, Callable

from src.robot.constants import ROBOT_LOG_FILE, create_robot_logger

__all__ = [
    "ConnectedCell",
    "ConnectStage",
    "StepOutcome",
    "TeardownReport",
    "connect_cell",
    "disconnect_cell",
    "release_perception",
]

logger = create_robot_logger("ConnectedCell", ROBOT_LOG_FILE)


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

    Every camera, not the primary one. A fused cell opens one device per camera in
    ``grasping.fusion.cameras``, so both the ``perception`` and ``multi_camera_perception`` slots
    are walked. On real hardware a second ``pipeline.start()`` on a streaming device fails, so a
    camera left open makes the next build of a multi-camera cell impossible without a restart.
    """
    if service is None:
        return StepOutcome.ABSENT
    orchestrator = getattr(getattr(service, "runtime", None), "orchestrator", None)
    outcome = StepOutcome.ABSENT
    for slot in ("perception", "multi_camera_perception"):
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

    A gripper that refuses rolls the arm back. Leaving a half-connected cell means a UR controller's
    single control script is held by a process that has already reported a failure.

    Connecting is motion. Robotiq activation is a calibration sweep of the full finger travel; a
    vacuum cup's connect asserts the ejector pin immediately and drops whatever it is holding. This
    function does not ask whether that is allowed. Its callers do: the console with an
    acknowledgement token, the CLI with a printed banner.
    """
    # No success logging here, on purpose: both callers already narrate a successful connect, so a
    # third line from this module would appear in the console's log twice. Failures are logged here,
    # because a rollback that did not take must not depend on a caller remembering to write it.
    say = announce or (lambda _stage: None)
    arm.connect()
    say(ConnectStage.ARM_CONNECTED)
    if gripper is None:
        return
    try:
        say(ConnectStage.GRIPPER_MOVING)
        gripper.connect()
        say(ConnectStage.GRIPPER_CONNECTED)
    except BaseException:
        logger.error("Gripper connect failed; rolling the arm back to disconnected.")
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

    @property
    def arm(self) -> Any:
        return getattr(self._orchestrator, "arm", None)

    @property
    def gripper(self) -> Any:
        return getattr(self._orchestrator, "gripper", None)

    @property
    def _orchestrator(self) -> Any:
        return getattr(getattr(self.service, "runtime", None), "orchestrator", None)

    def __enter__(self) -> "ConnectedCell":
        if self.lock is not None:
            self.lock.acquire()
        try:
            connect_cell(self.arm, self.gripper, announce=self.announce)
        except BaseException:
            # `connect_cell` has already rolled the arm back; the lock is this object's to give up.
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
        self.teardown = disconnect_cell(self.arm, self.gripper, self.service)
        if self.lock is not None:
            self.lock.release()
