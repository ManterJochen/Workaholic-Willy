"""The bodies of the console's moving runs: a task, the Home button, and the planner start.

Each driver runs on the run's own thread, started by ``RunRegistry.start_kind``, which takes the one-run lock, counts
down where a person's hands were last at the arm, and, once the driver returns its stop code, maps the class, writes
the recovery record for a problem stop of a moving kind, and lets go of the lock. A driver never raises for a domain
refusal: it returns the stop code and writes the run's ``error`` where the code is a problem.

* :func:`drive_task` drives the library's ``run_task`` (``src/robot/execution/task.py``) with :class:`ConsoleTaskHooks`:
  the task's events on the run's stream (its English sentence the envelope's ``human``, the rest the ``data``), the
  pick loop's own stages beside them, the extended ``pick_result``, the grasp overlays (captured at ``pick.executing``
  and after each pick, by the identity rule) and the target's, "stop after this part", "halt now" and a cell taken
  down read by the library between its motions, and the recovery record ended at a Restart's first ``task.returned``;
* :func:`drive_home` runs ``robot.home()`` or ``robot.move_joints(taught)`` through the arm's own judged verbs, unless
  the run was stopped before its move was sent; its arrival ends the recovery record;
* :func:`drive_planner` starts cuRobo, which moves nothing.

A moving driver stamps the console's motion (``api.runs.motion_started``) once it knows the arm moved: a pick that
gripped, a part placed, an arrival at the return pose. That ends the hands-off countdown; a run that counted down and
then moved nothing leaves it due for the next.

The plan a task runs is the resolved plan the route echoed (``TaskPlanOut``), joints included, so a Restart runs exactly
the plan the stopped run ran, its first motion the planned move to the return pose.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING, Any, Protocol

from api.codes import REFUSAL_STATUS, STOP_CLASS, RefusalCode, StopClass, StopCode
from api.constants import API_LOG_DIR, TASK_RUN_LOG_FILE
from api.events import Severity
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from api.cell import Console
    from api.runs import Run
    from src.robot.core import JointPositions
    from src.robot.execution.task import TaskPlan

__all__ = ["ConsoleTaskHooks", "TaskHooks", "drive_home", "drive_planner", "drive_task", "library_plan"]

logger = create_logger("TaskRun", TASK_RUN_LOG_FILE, log_dir=API_LOG_DIR)


class TaskHooks(Protocol):
    """What the library's ``run_task`` calls on its caller between its motions: library L1's ``TaskHooks``, as the plan
    names it. :class:`ConsoleTaskHooks` declares it, so a type checker holds the console's hooks to it, and
    ``tests/test_api_contract.py`` pins that every hook the library calls is one the console answers."""

    def event(self, name: str, /, **data: Any) -> None:
        """One of the task's events (``task.*``), to say on the run's stream."""
        ...

    def pick_done(self, part: int, pick: int, report: Any) -> None:
        """One pick of the task ended, with its report."""
        ...

    def stop_after_part(self) -> bool:
        """Whether the operator asked to stop once the part in hand is placed."""
        ...

    def halted(self) -> bool:
        """Whether "halt now" was pressed: no further motion is sent."""
        ...

    def abandoned(self) -> str:
        """Why the cell is being taken down under the run; ``""`` while it is not."""
        ...


# ---------------------------------------------------------------------------------------------------------------------
# What a task's events carry
# ---------------------------------------------------------------------------------------------------------------------

#: The timeline step each task event puts the run on (``RunOut.step``).
_STEP_OF_EVENT: Mapping[str, str] = {
    "task.survey_started": "survey", "task.part_started": "look", "task.carry_started": "place",
    "task.drop_planned": "place", "task.place_started": "place", "task.return_started": "return",
}
#: The task's events an operator should notice.
_WARNINGS = frozenset({"task.target_missing", "task.target_lost", "task.place_failed", "task.return_failed"})
#: The task's events that say the arm moved: a part let go at the drop, an arrival at the return pose. With a pick that
#: gripped (``pick_done``), what ends the hands-off countdown.
_MOVED = frozenset({"task.placed", "task.returned"})


def _jsonable(value: Any) -> Any:
    """``value`` as plain JSON data: tuples as lists, enums as their values, numpy scalars and arrays as numbers and
    lists, a number that is no number (NaN, infinite) as null, bytes left out. The wire carries nothing else."""
    from src.contracts import UNSET  # noqa: PLC0415

    if value is None or value is UNSET:
        return None
    if isinstance(value, enum.Enum):
        return _jsonable(value.value)
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()
                if not isinstance(item, (bytes, bytearray)) and item is not UNSET}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (bytes, bytearray)):
        return None
    if is_dataclass(value) and not isinstance(value, type):
        # Field by field, not ``asdict``: a deep copy would hand back copies of the UNSET sentinel nobody could tell.
        return _jsonable({item.name: getattr(value, item.name) for item in fields(value)})
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        return _jsonable(tolist())
    item = getattr(value, "item", None)
    if callable(item):
        return _jsonable(item())
    return str(value)


def _rounded_target(target: dict[str, Any]) -> dict[str, Any]:
    """A target as the wire shows it: millimetres to a tenth, the score to a thousandth (``TargetOut``)."""
    for key in ("centre_mm", "footprint_mm", "opening_mm"):
        values = target.get(key)
        if isinstance(values, list):
            target[key] = [round(float(v), 1) if isinstance(v, (int, float)) else v for v in values]
    for key, digits in (("rim_mm", 1), ("score", 3)):
        value = target.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            target[key] = round(float(value), digits)
    return target


def _severity(kind: str, data: Mapping[str, Any]) -> Severity:
    if kind == "task.placed":
        return Severity.SUCCESS
    if kind == "task.part_finished":
        return Severity.SUCCESS if data.get("placed") is True else Severity.WARN
    if kind in _WARNINGS:
        return Severity.WARN
    if kind == "task.pose_screened" and data.get("verdict") in ("guard_refused", "planner_refused"):
        return Severity.WARN
    if kind == "task.target_checked" and data.get("followed") is not True:
        return Severity.WARN
    if kind == "task.put_back" and data.get("outcome") != "executed":
        return Severity.WARN
    return Severity.INFO


class ConsoleTaskHooks(TaskHooks):
    """The library's ``TaskHooks`` seen from the console: what the task says, and what the operator asked of it.

    It also listens to the pick loop (:meth:`on_progress`, attached to the service for the task's length), so the
    task's stream carries every pick's stages and the overlay the calculator rendered when the grasp was decided.
    """

    def __init__(self, console: "Console", run: "Run") -> None:
        self.console = console
        self.run = run
        #: The overlay that stood before the current pick (the identity rule), the last one kept, how many were kept,
        #: and the current pick's own.
        self._before: Any = None
        self._kept: Any = None
        self._overlays = 0
        self._pick_overlay: str | None = None
        self._failures_before = 0
        self._recovery_ended = False
        self._moved = False

    # ---- what the operator asked ------------------------------------------------------------------------------

    def stop_after_part(self) -> bool:
        """The operator asked to stop after the part in hand (``POST /v1/task/stop``)."""
        return bool(self.run.stop_after_part)

    def halted(self) -> bool:
        """"Halt now" was pressed: no further motion is sent."""
        return bool(self.run.halt_requested)

    def abandoned(self) -> str:
        """Why the cell is being taken down under the run; ``""`` while it is not."""
        return str(self.run.abandoned)

    # ---- what the task says -------------------------------------------------------------------------------------

    def event(self, name: str, /, **data: Any) -> None:
        """Publish one of the task's events on the run's stream: its sentence as ``human``, the rest as ``data``; a
        target's picture kept as the run's target overlay and named by its URL."""
        kind = str(name)
        said = str(data.pop("said", "") or kind)
        image = data.pop("image_png", None)
        payload = _jsonable(data)
        if kind in ("task.target_found", "task.target_checked") and isinstance(payload.get("target"), dict):
            target = _rounded_target(payload["target"])
            keep = kind == "task.target_found" or payload.get("followed") is True
            url = None
            if keep and isinstance(image, (bytes, bytearray)) and image:
                url = self.console.overlays.put(self.run.id, "target", bytes(image))
            target["overlay"] = url
        if kind == "task.part_started":
            self._next_pick()
        step = _STEP_OF_EVENT.get(kind)
        if step:
            self.run.step = step
        self.console.hub.publish(self.run.id, kind, severity=_severity(kind, payload), human=said, data=payload)
        if kind in _MOVED:
            self._the_arm_moved()
        if kind == "task.returned":
            self._arrived()

    def _the_arm_moved(self) -> None:
        """The arm moved (a pick gripped, a part was let go, an arrival): stamped once, what ends the countdown."""
        if not self._moved:
            self._moved = True
            from api.runs import motion_started  # noqa: PLC0415

            motion_started(self.console)

    def _arrived(self) -> None:
        """A Restart's first arrival at its return pose ends the console's recovery record of the run it restarts."""
        if self._recovery_ended or not self.run.restart_of:
            return
        record = self.console.recovery
        if record is not None and record.run_id == self.run.restart_of:
            self.console.end_recovery(self.run.id, "restart")
        self._recovery_ended = True

    # ---- the pick loop beside it ----------------------------------------------------------------------------------

    def _service(self) -> Any:
        return self.console.session.service

    def _next_pick(self) -> None:
        service = self._service()
        self._before = getattr(service, "last_debug_image_png", None)
        self._pick_overlay = None
        count = getattr(service, "detector_failures", None)
        value = count() if callable(count) else 0
        self._failures_before = value if isinstance(value, int) and not isinstance(value, bool) else 0

    def _capture(self) -> str | None:
        """Keep the service's overlay where it is a new image of this pick (never one that stood before it, never the
        one kept last), and answer its URL; ``None`` where there is none."""
        png = getattr(self._service(), "last_debug_image_png", None)
        if not isinstance(png, (bytes, bytearray)) or not png or png is self._before or png is self._kept:
            return None
        url = self.console.overlays.put(self.run.id, self._overlays + 1, bytes(png))
        if url is None:
            return None
        self._overlays += 1
        self._kept = png
        self._pick_overlay = url
        return url

    def on_progress(self, event: Any) -> None:
        """One stage of the pick loop, said on the run's stream as a pick run says it; at ``executing`` with the
        overlay of the grasp as it was decided, before the arm moved."""
        from api.runs import _STEP_OF_STAGE, _payload, _sentence  # noqa: PLC0415

        stage = str(event.stage)
        severity, human = _sentence(event)
        payload = _payload(event)
        step = _STEP_OF_STAGE.get(stage)
        if step:
            self.run.step = step
        if stage == "executing":
            url = self._capture()
            if url is not None:
                payload["overlay"] = url
        self.console.hub.publish(self.run.id, f"pick.{stage}", severity=severity, human=human, step=stage,
                                 step_index=event.attempt, step_total=event.attempt_total, data=_jsonable(payload))

    def pick_done(self, part: int, pick: int, report: Any) -> None:
        """One pick of the task ended: its ``pick_result`` (the pinned keys of a pick run's, and what a task adds), its
        overlay, and the run's counts."""
        from api.runs import _names, _what_the_looks_came_to, pick_fields  # noqa: PLC0415

        self._capture()
        run = self.run
        outcome = str(getattr(report, "outcome", "unknown"))
        ok = bool(getattr(report, "succeeded", False))
        run.attempted += 1
        run.outcomes.append(outcome)
        run.succeeded += int(ok)
        summary = getattr(report, "failure_summary", None)
        reason = str(summary() if callable(summary) else "")
        count = getattr(self._service(), "detector_failures", None)
        failures = count() if callable(count) else 0
        failed = isinstance(failures, int) and not isinstance(failures, bool) and failures > self._failures_before
        data: dict[str, Any] = {
            "outcome": outcome, "succeeded": ok, "reason": reason,
            "looks": _names(getattr(report, "looks", ())), "looks_fused": _names(getattr(report, "looks_fused", ())),
            **_what_the_looks_came_to(report), **pick_fields(report),
            "part": part, "pick": pick, "detector_failed": failed, "overlay": self._pick_overlay,
        }
        logger.info("Task run %s, part %d, pick %d: %s.", run.id, part, pick, outcome)
        self.console.hub.publish(
            run.id, "pick_result", severity=Severity.SUCCESS if ok else Severity.WARN,
            human=f"Pick {pick}: {outcome}." if ok else f"Pick {pick}: {outcome}. {reason}",
            data=_jsonable(data),
        )
        if ok:
            self._the_arm_moved()


# ---------------------------------------------------------------------------------------------------------------------
# The task
# ---------------------------------------------------------------------------------------------------------------------


def library_plan(plan: Mapping[str, Any]) -> "tuple[TaskPlan, dict[str, JointPositions]]":
    """The library's ``TaskPlan`` and the taught poses it names, from the resolved plan a run carries (``TaskPlanOut``).

    The joints are the plan's own, as the route resolved them when the task was started, so a Restart runs exactly the
    plan that stopped. A camera place takes its air over the rim (``air_mm``); a pose place takes none.
    """
    from api.schemas import TaskPlanOut  # noqa: PLC0415
    from src.robot.core import JointPositions  # noqa: PLC0415
    from src.robot.execution.task import PlaceAt, TaskOptions, TaskPlan  # noqa: PLC0415

    resolved = TaskPlanOut.model_validate(dict(plan))
    poses: dict[str, JointPositions] = {}
    place = resolved.place
    if place.kind == "pose":
        if place.pose is None or place.pose_joints_deg is None:
            raise ValueError("a pose place names no pose or no joints")
        poses[place.pose] = JointPositions.deg(*place.pose_joints_deg)
        place_at = PlaceAt(pose=place.pose)
    else:
        place_at = PlaceAt(camera=str(place.phrase), air_mm=resolved.options.rim_air_mm)
    if resolved.return_to != "home":
        if resolved.return_joints_deg is None:
            raise ValueError(f"the return pose {resolved.return_to!r} carries no joints")
        poses[resolved.return_to] = JointPositions.deg(*resolved.return_joints_deg)
    options = resolved.options
    library = TaskPlan(
        object=resolved.object, place=place_at, return_to=resolved.return_to, scope=resolved.scope,
        options=TaskOptions(multi_view=options.multi_view, both_faces=options.both_faces,
                            closing_axis=options.closing_axis,
                            # A distance nobody asked for is the cell's, which may go longer where it opens too little.
                            push_mm=options.push_mm if options.push_asked else None,
                            critical_parts=options.critical_parts,
                            rescan=options.rescan, push=options.push, clear=options.clear,
                            blocker_into_the_place=options.blocker_into_the_place,
                            record_views=options.record_views, overlay=options.overlay,
                            pick_anything=options.pick_anything),
        first_motion=resolved.first_motion,
    )
    return library, poses


def _halted_error(console: "Console", run: "Run", sentence: str) -> str:
    from api.readiness import halt_reason  # noqa: PLC0415

    reason = halt_reason(console.session.arm) or "halt now was pressed"
    if run.halt_requested:
        return f"the console halted the arm ({reason}): {sentence}"
    return f"the arm is halted ({reason}): {sentence}"


def drive_task(console: "Console", run: "Run") -> StopCode:
    """The body of a task run: the library's ``run_task`` on the connected cell, said on the run's stream."""
    from api.readiness import halt_reason  # noqa: PLC0415
    from src.robot.execution.task import TaskRefused, run_task  # noqa: PLC0415

    service = console.session.service
    if run.plan is None:
        raise ValueError("a task run carries no plan")
    plan, poses = library_plan(run.plan)
    hooks = ConsoleTaskHooks(console, run)
    attach = getattr(service, "attach_progress_listener", None)
    if callable(attach):
        attach(hooks.on_progress)
    try:
        report = run_task(service, plan, hooks=hooks, poses=poses)
    except TaskRefused as refused:
        # The route refuses all of these before a run starts; met here, the cell changed in between (or the library
        # knows a rule the route does not). Nothing was commanded: no recovery record, and the code is said.
        try:
            refusal_code = RefusalCode(refused.code)
        except ValueError:
            refusal_code = RefusalCode.BAD_REQUEST  # a refusal the catalog does not know: a request the cell cannot do
        status = REFUSAL_STATUS[refusal_code]
        run.error = f"refused before anything moved ({refused.code}, {status}): {refused}"
        # Typed beside the sentence (``RunOut.refusal``): the browser says the refusal in its own words.
        run.refusal = {"code": str(refusal_code), "status": status}
        logger.warning("Task run %s refused by the library: %s", run.id, run.error)
        return StopCode.CANCELLED
    finally:
        if callable(attach):
            try:
                attach(None)
            except Exception as exc:  # noqa: BLE001 (teardown reports, it does not propagate)
                logger.warning("Detaching task run %s from the pick loop failed: %s: %s", run.id,
                               type(exc).__name__, exc)
    run.parts_placed = int(report.parts_placed)
    run.holding = bool(report.holding)
    run.attempted = max(run.attempted, int(report.picks))
    run.succeeded = max(run.succeeded, int(report.succeeded))
    code = StopCode(report.stop.value)
    if code is StopCode.CONTROLLER_STOPPED and halt_reason(console.session.arm):
        # A halted arm refuses as a controller would; it is the halt, whose remedy is not the pendant's.
        code = StopCode.HALTED
    if code is StopCode.HALTED:
        run.error = _halted_error(console, run, report.sentence)
    elif STOP_CLASS[code] is StopClass.PROBLEM:
        run.error = report.sentence
    logger.info("Task run %s ended %s: %s", run.id, code, report.sentence)
    return code


# ---------------------------------------------------------------------------------------------------------------------
# Home
# ---------------------------------------------------------------------------------------------------------------------


def drive_home(console: "Console", run: "Run") -> StopCode:
    """The body of a Home run: one planned move to Home or a taught pose, through the arm's own judged verb.

    Before it: a cell taken down, a halt or a part still held ends it with nothing sent, and so does the operator's
    stop (``POST /v1/task/stop``), which ends it ``cancelled``. A refused move ends it where the arm stands: as the halt
    where the arm is halted, as the controller where that stopped, else ``return_failed``. Its arrival ends the
    console's recovery record, and the hands-off countdown with it.
    """
    from api import readiness  # noqa: PLC0415
    from api.jaws import hand_of  # noqa: PLC0415
    from api.runs import motion_started  # noqa: PLC0415
    from src.robot.core import JointPositions  # noqa: PLC0415
    from src.robot.execution.robot import Robot  # noqa: PLC0415

    session = console.session
    arm, gripper = session.arm, session.gripper
    to = str(run.inputs.get("to", "home"))
    joints = run.inputs.get("joints_deg")
    run.step = "return"
    if run.abandoned:
        run.error = f"{run.abandoned} The move to {to} was not sent."
        return StopCode.DISCONNECTED
    halted = readiness.halt_reason(arm)
    if run.halt_requested or halted:
        run.error = _halted_error(console, run, f"the move to {to} was not sent")
        return StopCode.HALTED
    if run.stop_requested:
        # The operator's stop, read once more right before the move: one that came as the countdown ended still sends
        # nothing. Not a problem: the arm stands where the person last saw it.
        logger.warning("Home run %s was stopped before its move to %s was sent; nothing moved.", run.id, to)
        return StopCode.CANCELLED
    held = readiness.part_held(arm, gripper, hand_of(gripper), record=console.recovery)
    if held:
        run.error = f"{held}: the move to {to} was not sent, and no move outside a task carries a part"
        return StopCode.PART_STILL_HELD
    console.hub.publish(run.id, "home.started", severity=Severity.INFO,
                        human="Moving home." if to == "home" else f"Moving to {run.inputs.get('label') or to}.",
                        to=to, note="")
    robot = Robot.from_parts(arm=arm, gripper=gripper, lock_key=None)
    moved = robot.home() if to == "home" or joints is None else robot.move_joints(JointPositions.deg(*joints))
    note = str(getattr(moved, "message", "") or "")
    note = "" if note in ("", "move_home", "move_to_home", "move_to_joints", "move_joint", "home", "joints") else note
    if moved.ok:
        motion_started(console)
        console.hub.publish(run.id, "home.arrived", severity=Severity.SUCCESS,
                            human="Back home." if to == "home" else f"At {run.inputs.get('label') or to}.",
                            to=to, note=note)
        console.end_recovery(run.id, "home")
        return StopCode.FINISHED
    status = moved.status.value
    console.hub.publish(run.id, "home.refused", severity=Severity.ERROR,
                        human=f"The move to {to} did not run: {status}: {moved.message}", status=status,
                        message=str(moved.message))
    if run.abandoned:
        run.error = f"{run.abandoned} The move to {to} said: {moved.message}"
        return StopCode.DISCONNECTED
    if run.halt_requested or readiness.halt_reason(arm):
        run.error = _halted_error(console, run, f"the move to {to} ended ({status})")
        return StopCode.HALTED
    stopped = readiness.controller_gate(arm)
    if stopped:
        run.error = f"the move to {to} was refused ({status}: {moved.message}); {stopped}"
        return StopCode.CONTROLLER_STOPPED
    run.error = (f"the move to {to} was refused ({status}: {moved.message}), and the arm stays where it stands; a "
                 "person decides what happens next")
    return StopCode.RETURN_FAILED


# ---------------------------------------------------------------------------------------------------------------------
# The wave
# ---------------------------------------------------------------------------------------------------------------------


def _wave_stopped(run: "Run") -> bool:
    """Whether the wave sends no next swing: the operator's stop, a halt, or a cell taken down."""
    return bool(run.stop_requested or run.halt_requested or run.abandoned)


def drive_wave(console: "Console", run: "Run") -> StopCode:
    """The body of a wave run: Willy waves back at a greeting (``src.robot.execution.gestures.wave``), the second
    wrist joint swung out and back twice, each swing a straight joint line the exact guard judges before it is sent.

    Before it, as Home: a cell taken down, a halt or a part still held ends it with nothing sent, and so does the
    operator's stop, which ends it ``cancelled``. The stop, the halt and the cell are read again before every swing, and
    end the wave where the arm stands: ``cancelled``, ``halted``, ``disconnected``. A swing the arm refuses ends it
    there too: before the first one ran, nothing moved and the run ends ``cancelled`` with the refusal said, as a task
    the library refused before its first motion; after one ran, the arm stands a swing off where it started and the run
    ends as Home's refused move does (the halt, the controller, else ``return_failed``), so a person decides.
    """
    from api import readiness  # noqa: PLC0415
    from api.jaws import hand_of  # noqa: PLC0415
    from api.runs import motion_started  # noqa: PLC0415
    from src.robot.execution import gestures  # noqa: PLC0415

    session = console.session
    arm, gripper = session.arm, session.gripper
    run.step = "wave"
    if run.abandoned:
        run.error = f"{run.abandoned} The wave was not sent."
        return StopCode.DISCONNECTED
    if run.halt_requested or readiness.halt_reason(arm):
        run.error = _halted_error(console, run, "the wave was not sent")
        return StopCode.HALTED
    if run.stop_requested:
        logger.warning("Wave run %s was stopped before its first swing was sent; nothing moved.", run.id)
        return StopCode.CANCELLED
    held = readiness.part_held(arm, gripper, hand_of(gripper), record=console.recovery)
    if held:
        run.error = f"{held}: the wave was not sent, and no move outside a task carries a part"
        return StopCode.PART_STILL_HELD
    console.hub.publish(run.id, "wave.started", severity=Severity.INFO, human="Waving.",
                        swings=gestures.WAVE_SWINGS, swing_deg=gestures.WAVE_SWING_DEG)
    waved = gestures.wave(arm, should_stop=lambda: _wave_stopped(run))
    if waved.moved:
        motion_started(console)
    if waved.ok:
        console.hub.publish(run.id, "wave.done", severity=Severity.SUCCESS, human="Waved.",
                            swings=gestures.WAVE_SWINGS)
        return StopCode.FINISHED
    ran = sum(1 for report in waved.reports if report.ok)
    if waved.ended == "stopped":
        if run.abandoned:
            run.error = f"{run.abandoned} The wave ended after {ran} swing(s)."
            return StopCode.DISCONNECTED
        if run.halt_requested or readiness.halt_reason(arm):
            run.error = _halted_error(console, run, f"the wave ended after {ran} swing(s)")
            return StopCode.HALTED
        logger.warning("Wave run %s was stopped after %d swing(s); the arm stands where it stopped.", run.id, ran)
        return StopCode.CANCELLED
    refused = waved.refusal
    status = refused.status.value if refused is not None else "unknown"
    message = str(refused.message) if refused is not None else "no report"
    console.hub.publish(run.id, "wave.refused", severity=Severity.ERROR,
                        human=f"The wave did not run on: {status}: {message}", status=status, message=message,
                        moved=waved.moved)
    if run.abandoned:
        run.error = f"{run.abandoned} The wave said: {message}"
        return StopCode.DISCONNECTED
    if run.halt_requested or readiness.halt_reason(arm):
        run.error = _halted_error(console, run, f"the wave ended after {ran} swing(s) ({status})")
        return StopCode.HALTED
    stopped = readiness.controller_gate(arm)
    if stopped:
        run.error = f"the wave was refused after {ran} swing(s) ({status}: {message}); {stopped}"
        return StopCode.CONTROLLER_STOPPED
    if not waved.moved:
        # Nothing moved: the arm stands where the greeting found it, and the cell stays ready.
        run.error = f"refused before anything moved ({status}): {message}"
        logger.warning("Wave run %s refused before its first swing: %s", run.id, run.error)
        return StopCode.CANCELLED
    run.error = (f"the wave was refused after {ran} swing(s) ({status}: {message}), and the arm stays where it stands, "
                 "a swing off where it started; a person decides what happens next")
    return StopCode.RETURN_FAILED


# ---------------------------------------------------------------------------------------------------------------------
# The planner
# ---------------------------------------------------------------------------------------------------------------------


def drive_planner(console: "Console", run: "Run") -> StopCode:
    """The body of a planner run: start cuRobo (about a minute on a cell). It moves nothing either way."""
    from api.events import CELL_STREAM  # noqa: PLC0415

    arm = console.session.arm
    console.hub.publish(run.id, "planner.starting", severity=Severity.INFO,
                        human="Starting the planner; this takes about a minute and moves nothing.")
    console.hub.publish(CELL_STREAM, "cell.planner", severity=Severity.INFO, human="The planner is starting.",
                        state="starting")
    start = getattr(arm, "start_planner", None)
    try:
        if not callable(start):
            raise RuntimeError("this arm has no planner to start")
        loaded = start()
    except Exception as exc:  # noqa: BLE001 (a planner that did not start is the run's answer, not a crash)
        refusal = f"{type(exc).__name__}: {exc}"
        run.error = f"the planner did not start: {refusal}"
        console.hub.publish(run.id, "planner.failed", severity=Severity.ERROR,
                            human=f"The planner did not start: {refusal}", refusal=refusal)
        console.hub.publish(CELL_STREAM, "cell.planner", severity=Severity.WARN, human="The planner is off.",
                            state="off")
        return StopCode.PLANNER_FAILED
    said = _jsonable(loaded) if loaded is not None else None
    console.hub.publish(run.id, "planner.ready", severity=Severity.SUCCESS, human="The planner is ready.",
                        loaded=said if isinstance(said, (dict, list, str)) else None)
    console.hub.publish(CELL_STREAM, "cell.planner", severity=Severity.SUCCESS, human="The planner is ready.",
                        state="ready")
    return StopCode.PLANNER_READY
