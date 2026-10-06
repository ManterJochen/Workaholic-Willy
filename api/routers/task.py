"""A task: pick, then place, then return, then look again; once, or until nothing matching is left.

A command is a task. The operator's sentence is read into an editable card, and Start on that card is the
confirmation: ``POST /v1/task`` is the only thing that starts one, and it answers 202 with the run at once, like a
pick. Everything after that arrives on the run's event stream.

Before anything moves the route refuses what the cell or the request is not fit for, in the order of build plan 1.3.2:
the cell's state first (409), then the request's own (422). It resolves the plan the run will run (the place pose and
its joints, the return, the push distance, the air over a bin's rim) and echoes it; the run then drives the library's
``run_task`` on its own thread (``api.task_run``).

Two ways to stop, and neither is an emergency stop: ``POST /v1/task/stop`` lets the part in hand finish (a held part
is still placed, the arm returns, then the task ends; a Home run it stops before its move is sent), and "halt now" is
``POST /v1/cell/brake``. After a problem stop the way back is ``POST /v1/task/restart``, whose first motion is the
planned move to the return pose; it keeps every refusal of a new task but ``restart_required``, read against the cell as
it is now.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, Response

from api import readiness as gates
from api.cell import Console, console
from api.codes import RunKind
from api.constants import API_LOG_DIR, ROUTER_TASK_LOG_FILE
from api.readiness import refusal
from api.routers.pick import _refuse_unroutable_prompt
from api.runs import RunConflict, RunRefused
from api.schemas import RestartIn, RunOut, TaskIn
from src.utility.log_cfg import create_logger

router = APIRouter(tags=["task"])

logger = create_logger("TaskRouter", ROUTER_TASK_LOG_FILE, log_dir=API_LOG_DIR)

#: How the two overlay routes answer: a PNG, or the one failure envelope.
_PNG: dict[int | str, dict[str, Any]] = {200: {"content": {"image/png": {}}, "description": "The overlay, as a PNG."}}


# ---------------------------------------------------------------------------------------------------------------------
# What a task is refused for, and the plan it runs
# ---------------------------------------------------------------------------------------------------------------------


def _refuse_a_camera_target_this_cell_cannot_find(cell: Console) -> None:
    """``409 camera_target_unavailable``: a camera place needs the cell's own cameras and detector to find its target
    (none on the rehearsal cell), and on a wrist camera an arm whose live world holds the frames of the bin's look."""
    from src.robot.execution import place_target  # noqa: PLC0415

    service = cell.session.service
    try:
        locators = list(place_target.locators_for_service(service))
    except Exception as exc:  # noqa: BLE001 (LocatorRefused and every other reason: no camera to find it with)
        raise refusal("camera_target_unavailable", (
            f"a camera place finds its target with the cell's own camera, and this cell has none to lend: {exc}")) from None
    if not locators:
        raise refusal("camera_target_unavailable", "this cell holds no camera to find a target with.")
    if any(getattr(locator, "on_the_wrist", False) is True for locator in locators):
        world = getattr(cell.session.arm, "live_planner_world", None)
        if not (callable(getattr(world, "hold_pick_views", None)) and callable(getattr(world, "forget_pick_views",
                                                                                         None))):
            raise refusal("camera_target_unavailable", (
                "a wrist camera's place holds every frame of the bin's look while the part goes in, and this arm has no "
                "live planner world to hold them: nothing would keep the bin's walls in the world."))


def _refuse_an_empty_object(cell: Console, phrase: str, pick_anything: bool) -> None:
    """``422 object_required`` (Q7 A+): an empty object picks anything the camera sees, bin walls included, which a
    cell whose detector grounds a phrase takes only with the operator's word (``pick_anything``)."""
    if phrase.strip() or pick_anything:
        return
    grounds = getattr(cell.session.service, "grounds_a_phrase", None)
    if callable(grounds) and grounds() is False:
        return  # the rehearsal scene grounds no phrase: it picks what it shows
    raise refusal("object_required", (
        "the task names no object, and this cell's detector grounds a phrase: an empty one picks anything the camera "
        "sees, bin walls included. Name the object, or confirm 'anything the camera sees' (options.pick_anything)."))


def _resolved_push(cell: Console, push_mm: float | None) -> float | None:
    """How far a push of this task moves a part, as the cell resolves it; ``422 push_distance_refused`` where the cell
    refuses the operator's number (never shortened). ``None`` for a cell that says no distance."""
    distance = getattr(cell.session.service, "push_distance", None)
    if not callable(distance):
        return push_mm
    if push_mm is None:
        try:
            return float(distance())
        except (TypeError, ValueError):
            return None
    try:
        return float(distance(push_mm))
    except ValueError as refused:
        logger.warning("Task refused before anything moved: push_mm=%r (%s).", push_mm, refused)
        raise refusal("push_distance_refused", str(refused), push_mm=push_mm) from None


def _named(cell: Console, name: str, which: str) -> tuple[str | None, list[float]]:
    """A taught pose's label and joints (degrees); ``422 unknown_pose`` naming ``which`` (place, return) for a name
    nobody taught."""
    robot = cell.robot()
    pose = robot.named_poses.get(name)
    if pose is None:
        known = ", ".join(sorted(robot.named_poses)) or "none is"
        raise refusal("unknown_pose", f"the {which} names the pose {name!r}, and no pose of that name was taught "
                                      f"({known}).", which=which, pose=name)
    return (pose.label or None), [float(value) for value in pose.joints_deg]


def _closing_axis(cell: Console, axis: str | None) -> str | None:
    """The closing axis a task's grasps keep to, by its name (``+y`` reads ``y``); ``422 closing_axis_refused`` for one
    that names no direction, or that this cell's own motion turns every grasp off (``align_closing_to_base_x``)."""
    if axis is None or not axis.strip():
        return None
    from src.geometry.closing_axis import closing_axis_of  # noqa: PLC0415

    try:
        named = closing_axis_of(axis.strip())
    except (TypeError, ValueError) as exc:
        raise refusal("closing_axis_refused", str(exc), closing_axis=axis) from None
    if not named.name:
        raise refusal("closing_axis_refused", f"{axis!r} names no direction by name", closing_axis=axis)
    refusal_of = getattr(cell.session.service, "closing_axis_refusal", None)
    if callable(refusal_of):
        why = refusal_of(named.name)
        if isinstance(why, str) and why:
            raise refusal("closing_axis_refused", why, closing_axis=axis)
    return str(named.name)


def _refuse_both_faces_the_cell_turns_away(cell: Console, both_faces: bool) -> None:
    """``422 bad_request``: both jaw faces asked of a cell whose motion turns every grasp off the faces judged."""
    if not both_faces:
        return
    from src.robot.grasping.loop.pick_loop import judged_faces_turned_away  # noqa: PLC0415

    orchestrator = getattr(getattr(cell.session.service, "runtime", None), "orchestrator", None)
    try:
        turned = judged_faces_turned_away(orchestrator)
    except Exception as exc:  # noqa: BLE001 (a cell that cannot say is asked nothing it cannot keep)
        turned = f"whether this cell keeps both faces could not be read ({type(exc).__name__}: {exc})"
    if turned:
        raise refusal("bad_request", turned, field="options.both_faces")


def _plan(cell: Console, body: TaskIn) -> dict[str, Any]:
    """The resolved plan (``TaskPlanOut``): what is echoed in ``run_started`` and the run, and what a Restart runs again.
    Refuses, in the order of 1.3.2, what the request names that the cell cannot do."""
    from api.schemas import TaskPlanOut  # noqa: PLC0415
    from src.robot.execution.place_target import rim_clearance_mm  # noqa: PLC0415

    options = body.options
    push_mm = _resolved_push(cell, options.push_mm)
    robot = cell.robot()
    place: dict[str, Any]
    # 1.3.2's order: a pose nobody taught (the place's, then the return's) before a place nobody declared.
    named_place = _named(cell, body.place.pose, "place") if body.place.kind == "pose" and body.place.pose else None
    return_label: str | None = None
    return_joints: list[float] | None = None
    return_to = body.return_to.strip() or "home"
    if return_to != "home":
        return_label, return_joints = _named(cell, return_to, "return")
    if body.place.kind == "pose":
        name = body.place.pose or robot.default_place_pose
        if not name:
            raise refusal("no_place_declared", ("the task names no place pose, and this cell declares no default "
                                                "place (robot.default_place_pose): teach one, or name the place."))
        label, joints = named_place if named_place is not None else _named(cell, name, "place")
        place = {"kind": "pose", "pose": name, "pose_label": label, "pose_joints_deg": joints, "phrase": None,
                 "said": None}
        rim_air = None
    else:
        place = {"kind": "camera", "pose": None, "pose_label": None, "pose_joints_deg": None,
                 "phrase": body.place.phrase.strip(), "said": body.place.said}
        rim_air = (float(options.rim_air_mm) if options.rim_air_mm is not None
                   else rim_clearance_mm(getattr(cell.session.arm, "config", None)))
    axis = _closing_axis(cell, options.closing_axis)
    _refuse_both_faces_the_cell_turns_away(cell, options.both_faces)
    plan = TaskPlanOut.model_validate({
        "object": body.object.strip(),
        "object_said": body.object_said,
        "place": place,
        "return_to": return_to,
        "return_label": return_label,
        "return_joints_deg": return_joints,
        "scope": body.scope,
        "options": {"multi_view": options.multi_view, "both_faces": options.both_faces, "closing_axis": axis,
                    "push_mm": push_mm, "push_asked": options.push_mm is not None,
                    "critical_parts": options.critical_parts, "record_views": options.record_views,
                    "rescan": options.rescan, "push": options.push, "clear": options.clear,
                    "blocker_into_the_place": options.blocker_into_the_place,
                    "rim_air_mm": rim_air, "pick_anything": options.pick_anything, "overlay": options.overlay},
        "command": body.command.model_dump() if body.command is not None else None,
        "first_motion": "look",
        "countdown": cell.countdown_due(),
    })
    return plan.model_dump(mode="json")


def _refuse_what_the_plan_asks_that_this_cell_cannot(cell: Console, plan: dict[str, Any]) -> None:
    """A Restart's refusals of 1.3.2 beyond the cell's state, in their order, read against the cell as it is now: the
    record outlives a rebuild, and the detector, the cameras, the push ceiling or the hand's motion may have changed
    since the stop. The plan's resolved values are asked of the cell again; nothing is resolved anew."""
    from api.schemas import TaskPlanOut  # noqa: PLC0415

    try:
        resolved = TaskPlanOut.model_validate(plan)
    except ValueError as exc:
        raise refusal("bad_request", f"the stopped run's plan cannot be read again: {exc}") from None
    options = resolved.options
    if resolved.place.kind == "camera":
        _refuse_a_camera_target_this_cell_cannot_find(cell)
    _refuse_an_empty_object(cell, resolved.object, options.pick_anything)
    _refuse_unroutable_prompt(cell, resolved.object)
    if resolved.place.kind == "camera":
        _refuse_unroutable_prompt(cell, str(resolved.place.phrase or ""), code="target_not_routable")
    if options.push_mm is not None and options.push_asked:
        _resolved_push(cell, options.push_mm)
    _closing_axis(cell, options.closing_axis)
    _refuse_both_faces_the_cell_turns_away(cell, options.both_faces)


def _built_plan(plan: dict[str, Any]) -> None:
    """Build the library's plan from the resolved one here, before the run starts: a plan the library refuses is the
    request's error (``422 bad_request``), never a ``software_error`` inside a run."""
    from api.task_run import library_plan  # noqa: PLC0415

    try:
        library_plan(plan)
    except ValueError as exc:
        raise refusal("bad_request", f"the task cannot be planned: {exc}") from None


# ---------------------------------------------------------------------------------------------------------------------
# The routes
# ---------------------------------------------------------------------------------------------------------------------


@router.post("/task", response_model=RunOut, status_code=202, summary="Start a task: pick, place, return (THIS MOVES)")
def post_task(cell: Annotated[Console, Depends(console)], body: TaskIn) -> RunOut:
    """202: the task is accepted and runs on its own thread. Its first motion is its first look, after a 3 s
    countdown where a person's hands were last at the arm. Refused before anything moves (409, 422) where the cell or
    the request is not fit for it."""
    from api.task_run import drive_task  # noqa: PLC0415

    gates.refuse_unless(cell, gates.TASK_GATES)
    if body.place.kind == "camera":
        _refuse_a_camera_target_this_cell_cannot_find(cell)
    _refuse_an_empty_object(cell, body.object, body.options.pick_anything)
    _refuse_unroutable_prompt(cell, body.object)
    if body.place.kind == "camera":
        _refuse_unroutable_prompt(cell, body.place.phrase, code="target_not_routable")
    plan = _plan(cell, body)
    _built_plan(plan)
    try:
        run = cell.registry.start_kind(cell, RunKind.TASK, drive_task, plan=plan, prompt=plan["object"])
    except RunConflict as busy:
        raise refusal("run_active", str(busy), run_id=busy.existing.id) from busy
    except RunRefused as stopped:
        raise refusal(stopped.code, str(stopped), **stopped.detail) from stopped
    logger.info("Task run %s started: %r into %s, %s.", run.id, plan["object"],
                plan["place"]["pose"] or plan["place"]["phrase"], plan["scope"])
    return RunOut(**run.as_accepted())


@router.post("/task/stop", response_model=RunOut, summary="Stop the task after the part in hand")
def post_task_stop(cell: Annotated[Console, Depends(console)], run_id: str | None = None) -> RunOut:
    """Not a kill, and not an emergency stop: the part in hand is still placed and the arm returns, then the task
    ends ``stopped_after_part``. During the countdown it ends ``cancelled`` with nothing moved.

    A Home run is stopped too, before its one move is sent: during its countdown, or after it as long as the move has
    not gone out, it ends ``cancelled`` with nothing moved; a move already under way runs to its end. A wave is stopped
    the same way before its next swing, and the arm stands where the last swing left it. Nothing is latched, so no "the
    cell is clear" is owed, as after a halt."""
    registry = cell.registry
    run = registry.get(run_id) if run_id else registry.active()
    if run is None:
        raise refusal("no_such_run", "no task to stop." if not run_id else f"no run {run_id!r}.")
    if run.kind in (RunKind.HOME, RunKind.WAVE):
        registry.stop_home(run.id)
        return RunOut(**run.to_dict())
    if run.kind is not RunKind.TASK:
        raise refusal("not_a_task", (f"run {run.id} is a {run.kind} run, not a task: stop a pick run with "
                                     "POST /v1/pick/stop, or halt now with POST /v1/cell/brake."), run_id=run.id,
                      kind=str(run.kind))
    registry.stop_after_part(run.id)
    return RunOut(**run.to_dict())


@router.post(
    "/task/restart", response_model=RunOut, status_code=202,
    summary="Restart after a stop (THIS MOVES: first the planned move to the return pose)",
)
def post_task_restart(cell: Annotated[Console, Depends(console)], body: RestartIn) -> RunOut:
    """A new run of the stopped run's plan, whose first motion is the planned move to the return pose. Only the
    console's recovery record restarts, once a person confirmed the cell is clear and the hand is empty.

    A stopped task restarts as a new task (``restart_of`` the stopped run, ``first_motion`` ``return``); a stopped Home
    run restarts as a Home run to the same pose. A toggle hand is asked where its jaws stand since the stop first.
    Every refusal of a new task but ``restart_required`` holds, read against the cell as it is now.
    """
    from api.task_run import drive_home, drive_task  # noqa: PLC0415

    stopped = cell.registry.get(body.run_id)
    if stopped is None:
        raise refusal("no_such_run", f"no run {body.run_id!r}.")
    record = cell.recovery
    if record is None or record.run_id != stopped.id or stopped.kind not in (RunKind.TASK, RunKind.HOME):
        newest = f"the console's stop record names run {record.run_id}" if record is not None else \
            "no stop record stands"
        raise refusal("not_restartable", (f"run {stopped.id} ({stopped.kind}) is not the stop to come back from: "
                                          f"{newest}; only a stopped task or Home run restarts."), run_id=stopped.id)
    try:
        if stopped.kind is RunKind.TASK:
            gates.refuse_unless(cell, gates.RESTART_GATES, record=record)
            plan = dict(stopped.plan or {})
            if not plan:
                raise refusal("not_restartable", f"run {stopped.id} keeps no plan to run again.", run_id=stopped.id)
            _refuse_what_the_plan_asks_that_this_cell_cannot(cell, plan)
            plan["first_motion"] = "return"
            plan["countdown"] = cell.countdown_due()
            _built_plan(plan)
            run = cell.registry.start_kind(cell, RunKind.TASK, drive_task, plan=plan, prompt=stopped.prompt,
                                           restart_of=stopped.id)
        else:
            gates.refuse_unless(cell, gates.HOME_GATES, record=record)
            run = cell.registry.start_kind(cell, RunKind.HOME, drive_home, inputs=dict(stopped.inputs),
                                           restart_of=stopped.id)
    except RunConflict as busy:
        raise refusal("run_active", str(busy), run_id=busy.existing.id) from busy
    except RunRefused as refused:
        raise refusal(refused.code, str(refused), **refused.detail) from refused
    logger.warning("Run %s restarts run %s (%s): its first motion is the planned move to the return pose.", run.id,
                   stopped.id, stopped.kind)
    return RunOut(**run.as_accepted())


def _overlay(cell: Console, run_id: str, key: int | str) -> Response:
    if cell.registry.get(run_id) is None:
        raise refusal("no_such_run", f"no run {run_id!r}.")
    png = cell.overlays.get(run_id, key)
    if png is None:
        raise refusal("no_overlay", f"run {run_id} kept no overlay {key!r}.", run_id=run_id, key=str(key))
    return Response(content=png, media_type="image/png")


@router.get(
    "/runs/{run_id}/overlays/{n}", response_class=Response, responses=_PNG,
    summary="The n-th grasp overlay a task captured",
)
def get_run_overlay(cell: Annotated[Console, Depends(console)], run_id: str, n: int) -> Response:
    """The grasp as decided, rendered before the arm moved: its own image, never drawn over the live picture."""
    return _overlay(cell, run_id, n)


@router.get(
    "/runs/{run_id}/target/overlay", response_class=Response, responses=_PNG,
    summary="The place target a task found",
)
def get_target_overlay(cell: Annotated[Console, Depends(console)], run_id: str) -> Response:
    """The target the camera found (a bin), drawn over the frame it was seen in."""
    return _overlay(cell, run_id, "target")
