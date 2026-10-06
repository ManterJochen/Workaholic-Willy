"""Bringing the cell up and down.

Four endpoints and one shape: look, then acknowledge, then act. ``GET /v1/cell`` says where the cell
is; ``POST /v1/cell/build`` assembles it without touching a robot; ``GET /v1/cell/connect-preview``
names every motion connecting will cause and hands back a token; ``POST /v1/cell/connect`` accepts only
that token.

The token is not ceremony. On this cell "connect" is motion: a Robotiq activates by sweeping its full
finger travel, and a vacuum cup asserts its ejector and drops whatever it holds. A browser button is
reachable by a stray curl, a replayed request and a reloaded tab, so requiring a token that a preview
issued for this configuration means a connect cannot happen without something having read what it
would do.

A run is the other thing these routes answer to. It lives on its own thread and outlives the request
that started it, so taking the cell down under it and bringing it up again would hand the old run a
new arm. Disconnect stops the run before the cell comes down, and Connect and Build are refused while
its thread is still alive.

After a stop, these routes are the way back, and none of them is an emergency stop: "halt now" brakes the arm under
control (``POST /v1/cell/brake``) and the red button stays the safety halt; a person confirms "the cell is clear"
(``POST /v1/cell/acknowledge``), which never clears a protective stop, since that is done at the pendant; and Home
(``POST /v1/cell/home``) is a planned move that starts only on a click.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from api import cell as cell_module
from api import jaws as browser_jaws
from api import readiness as gates
from api.cell import Console, RunLocked, console
from api.codes import RunKind
from api.constants import API_LOG_DIR, ROUTER_CELL_LOG_FILE
from api.events import CELL_STREAM, Severity
from api.lifecycle import CellState, CellTransitionError, ConnectRefused
from api.readiness import HaltStateLike, ReadsHaltState, payload_model_of, refusal
from api.runs import RunConflict, RunRefused
from api.schemas import (
    AcknowledgeIn,
    BrakeOut,
    CellFactsOut,
    CellOut,
    ConnectIn,
    ConnectPreviewOut,
    HaltStateOut,
    HomeIn,
    MotionWarningOut,
    PayloadModelName,
    PlannerOut,
    ReadinessOut,
    RecoveryOut,
    RunOut,
    TelemetryOut,
)
from api.telemetry import read_telemetry
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/cell", tags=["cell"])

#: The typed refusals themselves are logged where they are decided (``api.lifecycle``), and every
#: failure envelope is logged once more as it leaves the server (``api.app``). What is left for this
#: file is what only it knows: the cross-process lock, and why a build refused.
logger = create_logger("CellRouter", ROUTER_CELL_LOG_FILE, log_dir=API_LOG_DIR)

#: Each typed refusal maps to one HTTP status. Each is a different thing for the operator to do, which
#: is the whole reason the reasons are typed: 409 means "fix the cell", 403 means "fix the config",
#: 428 means "read the preview first".
_STATUS: dict[ConnectRefused, int] = {
    ConnectRefused.NOT_BUILT: 409,
    ConnectRefused.WRONG_STATE: 409,
    ConnectRefused.CELL_BUSY: 409,
    ConnectRefused.NOT_ACKNOWLEDGED: 428,   # Precondition Required
    ConnectRefused.STALE_TOKEN: 428,
    ConnectRefused.NO_REAL_GRIPPER: 403,
    ConnectRefused.DRIVER_REFUSED: 502,     # the refusal came from the controller, not from this server
}


def _refuse(error: CellTransitionError) -> HTTPException:
    return HTTPException(
        status_code=_STATUS.get(error.reason, 409),
        detail={"code": str(error.reason), "message": str(error), "detail": {}},
    )


def _refuse_while_a_run_is_alive(cell: Console, what: str, would: str) -> None:
    """Refuse ``what`` while a run's thread still holds the cell: 409 ``run_active``, as a config write is refused.

    A run lives on its own thread and ends only when that thread does, a Disconnect or a Stop notwithstanding: both set
    a flag it reads, neither can pull it out of a motion or out of a pick waiting on a person's answer. Until it has
    ended, a Connect would hand it a live arm and hand to go on with, and a Build would take the cell and its camera
    from under it. So both wait for ``active_run_id`` to clear, which ``RunRegistry`` does as the thread's last act,
    and never queue: the operator re-submits once the run has ended.
    """
    try:
        cell.require_idle()
    except RunLocked as locked:
        logger.warning("%s refused: run %s is still active.", what.capitalize(), locked.run_id)
        raise HTTPException(
            status_code=409,
            detail={
                "code": "run_active",
                "message": (
                    f"run {locked.run_id} is still active, so {what} is refused: {would}. A Disconnect or a Stop "
                    f"ends it once the pick in flight returns, and a pick waiting on a question at the server's "
                    f"terminal returns once someone answers there; the run does not go on after it. Try again when "
                    f"GET /v1/runs/{locked.run_id} no longer says running; this is not a queue."
                ),
                "detail": {"run_id": locked.run_id},
            },
        ) from locked


def _render(cell: Console) -> CellOut:
    from src.robot.execution.cell_lock import peek

    # Reused, not re-derived: one reading of the perception stack, so /v1/cell and /v1/diagnostics can
    # never disagree about which models this cell would use.
    from api.routers.diagnostics import _perception

    session = cell.session
    substitution = session.substitution
    key = cell.lock_key()
    holder = peek(key) if key else None
    # Read once: a Restart's or a Home run's arrival ends the record on its run's thread while a browser polls.
    record = cell.recovery
    return CellOut(
        state=str(session.state),
        arm=type(session.arm).__name__ if session.arm is not None else None,
        gripper=type(session.gripper).__name__ if session.gripper is not None else None,
        vendor=str(cell.robot().vendor),
        profile=cell.profile,
        perception=_perception(cell),
        # A substituted gripper is reported whether or not anything is connected: it is a property of
        # what was built, and the operator needs it before they reach for the connect button.
        gripper_substitution=(
            {
                "reason": str(substitution.reason),
                "requested": substitution.requested,
                "detail": substitution.detail,
                "fix": substitution.fix,
            }
            if substitution is not None else None
        ),
        lock_key=key,
        # Who holds the cell right now: this console, or the CLI runner in another terminal.
        # A snapshot by nature: it can be stale the instant it is read, which is why it never gates
        # anything. The OS arbitrates the actual acquisition.
        lock_holder=holder.describe() if holder is not None else None,
        active_run_id=cell.active_run_id,
        # Read, never commanded: the hand chip and the stop card's gates read these on every poll, and none of them
        # may cost the arm anything (no dashboard round trip, no output written).
        hand=browser_jaws.hand_of(session.gripper),
        needs_person=_needs_person(session.service),
        halted=_halted(session.arm),
        planner=_planner(session.arm),
        payload_model=_payload_model(session.arm),
        recovery=RecoveryOut(**record.to_dict()) if record is not None else None,
        jaws_confirmed_at=cell.jaws_confirmed_at,
        countdown_due=cell.countdown_due(),
        jaws_question=browser_jaws.pending(cell) is not None,
    )


# The panel reads below run on every poll of ``GET /v1/cell``, and build, connect and disconnect answer with the same
# render: a read that raises is never a 500 after a transition that happened. Each fails closed instead.


def _needs_person(service: Any) -> str:
    """Why a pick stopped where the arm stands, a person to decide (the service's latch); ``""`` while none waits.

    A latch that cannot be read is said, never ``""``: nobody can say a person is not needed.
    """
    try:
        said = getattr(service, "stopped_where_the_arm_stands", "")
    except Exception as exc:  # noqa: BLE001 (a panel read that fails is said, never raised)
        return (f"the service's stop could not be read ({type(exc).__name__}: {exc}), so a person looks at the arm "
                "before anything moves")
    return said if isinstance(said, str) else ""


#: What a halt did to the move in flight, in the library's words (``HALT_BRAKE_OUTCOMES``): passed on as said, never a
#: word guessed from ``in_motion`` and ``braked``, where a move that ran out with the brake off would read as braking.
_BRAKE_WORDS = frozenset({"none", "pending", "braked", "unconfirmed", "ran_out"})


def _halted(arm: Any) -> HaltStateOut | None:
    """The arm's halt latch ("halt now"), where the arm has one (:class:`ReadsHaltState`, library L9); read, never
    cleared.

    Typed as the latch says it, so a double that answers every attribute reads as not halted. A read that raises
    confirms no halt, and every gate reads the latch itself. ``brake`` (what became of the move in flight) is passed on
    only where it is one of the library's five words.
    """
    try:
        state: HaltStateLike | None = arm.halt_state() if isinstance(arm, ReadsHaltState) else None
        if state is None:
            return None
        reason, at = getattr(state, "reason", None), getattr(state, "requested_at", None)
        in_motion, braked = getattr(state, "in_motion", False), getattr(state, "braked", False)
        brake_s = getattr(state, "brake_s", None)
        brake = getattr(state, "brake", None)
    except Exception:  # noqa: BLE001 (a panel read that fails says nothing rather than failing the panel)
        return None
    if not isinstance(reason, str) or not isinstance(at, (int, float)) or isinstance(at, bool):
        return None
    return HaltStateOut(
        reason=reason, requested_at=float(at), in_motion=in_motion is True, braked=braked is True,
        brake_s=float(brake_s) if isinstance(brake_s, (int, float)) and not isinstance(brake_s, bool) else None,
        brake=brake if isinstance(brake, str) and brake in _BRAKE_WORDS else None,  # type: ignore[arg-type]
    )


def _planner(arm: Any) -> PlannerOut:
    """cuRobo's state on the arm (``planner_state``, library L9: a property or a method); ``not_used`` for an arm that
    says nothing. An arm whose planner cannot say its state now reads ``off``, not ready, never ``not_used``, which would
    read as nothing to wait for."""
    state: Any
    try:
        state = getattr(arm, "planner_state", None)
        state = state() if callable(state) else state
    except Exception:  # noqa: BLE001 (a panel read that fails is not ready, never a 500)
        state = "off"
    return PlannerOut(state=state if state in ("off", "starting", "ready") else "not_used")


def _payload_model(arm: Any) -> PayloadModelName:
    """What models the part the gripper carries now, as the arm says it; ``not_applicable`` for an arm that models no
    part at all (the dummy, sim). ``unknown`` where it models parts and its answer could not be read, or is one this
    console does not know: ``unknown`` is not in ``NO_PART_MODELLED``, so a part nobody can rule out keeps the stop card's
    "no part held, now" gate shut. The gates read the same (``api.readiness.payload_model_of``)."""
    return payload_model_of(arm)


@router.get("", response_model=CellOut, summary="Where is the cell?")
def get_cell(cell: Annotated[Console, Depends(console)]) -> CellOut:
    return _render(cell)


@router.post("/build", response_model=CellOut, summary="Assemble the cell (no robot is touched)")
def post_build(
    cell: Annotated[Console, Depends(console)],
    rehearse: bool = False,
) -> CellOut:
    """Build through the same functions the CLI runner uses.

    ``rehearse=true`` builds the desk scene instead of opening a camera and loading models, which is how
    the whole console stays playable with no hardware attached. Not cheap otherwise: a real build takes
    tens of seconds while the detector and segmenter load onto the GPU.

    Refused while a run is active, before anything is released: a build takes the cell down and closes its camera
    first, and a run that outlived a Disconnect still has its thread in that cell.

    A halt nobody has said the cell is clear of latches the new arm as well (``Console.carried_halt``), so Connect
    answers ``halted`` after a rebuild too: only "the cell is clear" ends a halt, never building again.
    """
    _refuse_while_a_run_is_alive(
        cell, "a build", "a build takes the cell down and closes its camera under that run's thread"
    )
    try:
        cell.build(rehearse=rehearse)
    except CellTransitionError as error:
        raise _refuse(error) from error
    except Exception as exc:  # noqa: BLE001 (a build refusal is a designed outcome, not a crash)
        # The message travels to the browser and nowhere else. On a real cell this is the camera that
        # would not open or the weights that are not on the box, and it is the first thing anyone
        # asks about afterwards.
        logger.error("Build refused: %s: %s", type(exc).__name__, exc)
        raise HTTPException(
            status_code=422,
            detail={
                "code": "build_refused",
                "message": f"{type(exc).__name__}: {exc}",
                "detail": {},
            },
        ) from exc
    return _render(cell)


@router.get(
    "/connect-preview",
    response_model=ConnectPreviewOut,
    summary="What will move, and the token that acknowledges it",
)
def get_connect_preview(cell: Annotated[Console, Depends(console)]) -> ConnectPreviewOut:
    try:
        preview = cell.session.preview(cell.robot(), cell.fingerprint())
    except CellTransitionError as error:
        raise _refuse(error) from error
    return ConnectPreviewOut(
        token=preview.token,
        expires_at=preview.expires_at,
        arm=preview.arm,
        gripper=preview.gripper,
        warnings=[
            MotionWarningOut(subject=w.subject, what=w.what, precaution=w.precaution)
            for w in preview.warnings
        ],
        blocking=list(preview.blocking),
    )


@router.post("/connect", response_model=CellOut, summary="Bring the cell up (THIS MOVES)")
def post_connect(cell: Annotated[Console, Depends(console)], body: ConnectIn) -> CellOut:
    """Arm first, then gripper, and roll the arm back if the gripper refuses.

    The cross-process lock is taken before the arm, so a cell already owned by the CLI runner is
    refused without a single command reaching the controller.

    And a run still active is refused before either: a run that outlived a Disconnect keeps its thread,
    and a connect now would hand that thread a live arm and hand to go on with, the jaws' question
    included, on a count a person had not answered for this connect.
    """
    from src.robot.execution.cell_lock import CellBusy, CellLock

    _refuse_while_a_run_is_alive(
        cell, "a connect", "its thread would go on with the arm and the hand this connect brings up"
    )
    # A built arm whose halt latch is set stays latched across a Disconnect and a Connect: "halt now" is ended by a
    # person's "the cell is clear" alone (POST /v1/cell/acknowledge, allowed while built), never by connecting again,
    # and never by building again either: the console carries the halt to the arm of every later build (Console.build).
    latched = gates.halt_reason(cell.session.arm)
    if latched:
        raise refusal("halted", (f"the arm is halted ({latched}): a person confirms the cell is clear first "
                                 "(POST /v1/cell/acknowledge); connecting again does not end a halt."), reason=latched)
    key = cell.lock_key()
    lock = CellLock(key, owner="operator console") if key else None
    if lock is not None:
        try:
            lock.acquire()
            logger.info("Cell lock %s acquired for the operator console.", key)
        except CellBusy as busy:
            # Which other process owns the cell, usually the CLI runner in another terminal. The
            # HTTP envelope carries the holder to the browser; this keeps it after the tab is closed.
            logger.warning(
                "Connect refused: cell %s is held by %s.",
                key, busy.holder.describe() if busy.holder else "another process",
            )
            raise HTTPException(
                status_code=409,
                detail={
                    "code": str(ConnectRefused.CELL_BUSY),
                    "message": str(busy),
                    "detail": {"holder": busy.holder.describe() if busy.holder else None},
                },
            ) from busy

    try:
        cell.session.connect(body.token, cell.fingerprint(), lock)
    except CellTransitionError as error:
        # connect() already released the lock on the paths it took; releasing again is a no-op, and
        # doing it here covers the refusals raised before it ever got that far.
        if lock is not None:
            lock.release()
        raise _refuse(error) from error
    _start_the_planner_after_connect(cell)
    return _render(cell)


def _start_the_planner_after_connect(cell: Console) -> None:
    """Build plan item 19: a cuRobo arm whose planner is off starts it at once, as a planner run that moves nothing and
    holds the run lock, so no task races a half-started planner; decided before Connect answers. The dummy and an ik arm
    start nothing. A start that cannot begin is said, never a reason to undo a connect that happened."""
    from src.robot.execution.motion import MotionRoute, route_of  # noqa: PLC0415

    arm = cell.session.arm
    if _planner(arm).state != "off" or route_of(arm).route is not MotionRoute.PLANNED:
        return
    from api.task_run import drive_planner  # noqa: PLC0415

    try:
        run = cell.registry.start_kind(cell, RunKind.PLANNER, drive_planner)
    except RunConflict as busy:  # pragma: no cover - Connect refused while a run was alive
        logger.warning("The planner was not started after Connect: %s", busy)
        return
    logger.info("Connect started the planner as run %s (about a minute; it moves nothing).", run.id)


@router.get("/status", response_model=TelemetryOut, summary="Live telemetry (receive stream only)")
def get_status(
    cell: Annotated[Console, Depends(console)],
    include_controller_state: bool = False,
) -> TelemetryOut:
    """A snapshot cheap enough to poll.

    The default reading is the RTDE output stream: pose, joints, TCP wrench, which the controller is
    already broadcasting, so polling it costs a running motion nothing. ``get_joint_torques`` is
    deliberately absent: it goes through the control interface a pick is using, and it is never
    offered here.

    ``include_controller_state=true`` adds robot mode, safety mode and the human-readable safety text,
    and costs a dashboard socket round trip per call, which is why it is opt-in rather than part of
    the default tick. Poll it at seconds, not at frames.

    Read-only in the strong sense: there is no counterpart that clears a protective stop. That is done
    where the arm is visible, at the pendant, for the same reason there is no E-stop button here.
    """
    telemetry = read_telemetry(cell.session.arm, include_controller_state=include_controller_state)
    return TelemetryOut(
        state=str(cell.session.state),
        connected=telemetry.connected,
        simulated=telemetry.simulated,
        vendor=telemetry.vendor,
        model=telemetry.model,
        tcp_position_mm=list(telemetry.tcp_position_mm) if telemetry.tcp_position_mm else None,
        tcp_quaternion_xyzw=(
            list(telemetry.tcp_quaternion_xyzw) if telemetry.tcp_quaternion_xyzw else None
        ),
        joint_positions=list(telemetry.joint_positions) if telemetry.joint_positions else None,
        tcp_force_n=list(telemetry.tcp_force_n) if telemetry.tcp_force_n else None,
        tcp_torque_nm=list(telemetry.tcp_torque_nm) if telemetry.tcp_torque_nm else None,
        robot_mode=telemetry.robot_mode,
        safety_mode=telemetry.safety_mode,
        protective_stopped=telemetry.protective_stopped,
        emergency_stopped=telemetry.emergency_stopped,
        controller_message=telemetry.controller_message,
        controller_is_simulator=telemetry.controller_is_simulator,
        controller_serial=telemetry.controller_serial,
        controller_host=telemetry.controller_host,
        unavailable=telemetry.unavailable,
        controller_state_included=include_controller_state,
    )


@router.post("/disconnect", response_model=CellOut, summary="Take the cell down")
def post_disconnect(cell: Annotated[Console, Depends(console)]) -> CellOut:
    """Gripper first, then arm: the reverse of connect, so a cup's release still reaches the I/O.

    Idempotent: disconnecting an already-disconnected cell is a no-op rather than an error, because the
    one thing an operator must always be able to do is put the cell down.

    An active run is stopped FIRST, and the order is the guarantee. Its flag is set before the arm and
    the hand come down, so whatever of the run returns next meets the stop before it starts anything
    new: ``pick()`` will not begin on it, the pick loop begins no attempt on it (the first one of a pick
    that was waiting on a person's answer at its start included), and the run begins no next pick.
    What is already inside an attempt meets a disconnected arm. The run keeps the console's run lock
    until its thread ends, and Connect and Build refuse while it does, so the old run never meets a
    reconnected cell. The disconnect is never held up by a moving run: putting the cell down does not
    wait on one.

    The whole order (build plan item 18, ``api.cell.take_down``): a jaws question waiting is cancelled first, before
    anything takes the session lock its connect holds; a teach is ended and joined; a planner start is joined (it moves
    nothing); the run is abandoned; the arm of an abandoned moving run is latched, and braked where it brakes a move in
    flight; then the cell comes down. Where a move was in flight, or the arm braked one, the latch stays and the next
    Connect waits for "the cell is clear"; with none in flight on an arm that lets a move run to its end, it is given
    back once the cell is down.
    """
    cell_module.take_down(
        cell,
        "the cell was disconnected from the console while the run was active, so the run stops and nothing "
        "of it starts again. Look at the arm and the jaws before you connect: a part may still be held.",
    )
    return _render(cell)


# --- the console of commit 2: facts, readiness, and the way back after a stop -----------------------------------


@router.get("/facts", response_model=CellFactsOut, summary="What this cell is (read after build and after connect)")
def get_facts(cell: Annotated[Console, Depends(console)]) -> CellFactsOut:
    """Its cameras and looks, its hand's natural axis, push, the detector, the brake, the carried part and how its
    moves are planned. Moves nothing."""
    return gates.facts(cell)


@router.get("/readiness", response_model=ReadinessOut, summary="The ready bar: can the first task start?")
def get_readiness(cell: Annotated[Console, Depends(console)]) -> ReadinessOut:
    """Six lights and the blockers. Always answers, moves nothing, and reads the controller from the receive stream
    only. The server enforces its gates on its own; this only says them before a button is pressed."""
    return gates.readiness(cell)


@router.post("/acknowledge", response_model=CellOut, summary="The cell is clear (a person's word)")
def post_acknowledge(cell: Annotated[Console, Depends(console)], body: AcknowledgeIn) -> CellOut:
    """A person confirms the cell is clear after a stop: it clears the arm's halt latch (and the halt the console carries
    to every later build), the service's latch of a recovery that needs a person, and stamps the recovery record
    cleared. It moves nothing, and it never clears a protective stop: that is done at the pendant, where the arm is
    visible.

    ``jaws_empty: true`` is the person's word that the hand holds nothing: taken for a hand that cannot say it itself
    (a toggle answers the jaws question instead, and is taken only while its count says open; a hand that measures a
    part refuses it), and then the planner is told the hand is empty too. A person's hands were at the jaws for it, so
    the next motion counts down 3 s first (owner decision Q2 = A); it answers no jaws question.
    """
    from src.robot.core.arm_capabilities import CarriesPayload, SupportsHalt, halt_state_of  # noqa: PLC0415
    from src.robot.core.gripper import HoldEvidence, hold_evidence_of  # noqa: PLC0415

    session = cell.session
    if session.service is None or session.state not in (CellState.BUILT, CellState.CONNECTED):
        raise refusal("not_built", f"the cell is {session.state}: nothing is built to say clear of.")
    active = cell.registry.active()
    if active is not None:
        # A run thread, an abandoned one included, may still be inside a verb: a latch cleared between two of its
        # moves would let the next one go.
        raise refusal("run_active", (f"run {active.id} ({active.kind}) is still active: the cell is said clear once "
                                     "it has ended."), run_id=active.id)
    arm, gripper, service = session.arm, session.gripper, session.service
    halted = halt_state_of(arm)
    if session.state is CellState.CONNECTED:
        code, why = gates.controller_reading(arm)
        if code == "controller_stopped":
            raise refusal("controller_stopped", (f"{why} The console never clears a protective stop: clear it at the "
                                                 "pendant first, then say the cell is clear."))
    if body.jaws_empty:
        hand = browser_jaws.hand_of(gripper)
        if hand.kind == "toggle" and hand.jaws != "open":
            raise refusal("jaws_not_open", (f"the toggle's count says its jaws stand {hand.jaws}: answer the jaws "
                                            "question instead (POST /v1/cell/jaws/check), which opens them with one "
                                            "change of the output while a person holds the part."), jaws=hand.jaws)
        try:
            measured = gripper is not None and hand.connected and hold_evidence_of(gripper) is HoldEvidence.HELD
        except Exception:  # noqa: BLE001 (a hand nobody can read may hold anything)
            measured = True
        if measured:
            raise refusal("jaws_not_open", "the gripper still measures a part in its jaws: take it out first.")
    cleared: list[str] = []
    if gates._needs_person(service):  # noqa: SLF001
        acknowledge = getattr(service, "acknowledge_needs_person", None)
        if callable(acknowledge):
            acknowledge()
            cleared.append("needs_person")
    if halted is not None and isinstance(arm, SupportsHalt):
        arm.clear_halt()
        cleared.append("halted")
    # The halt the console carries to every later build ends with the arm's: a person said the cell is clear.
    if cell.forget_halt() and "halted" not in cleared:
        cleared.append("halted")
    if cell.mark_cell_clear() is not None:
        cleared.append("recovery")
    emptied = False
    if body.jaws_empty:
        if isinstance(arm, CarriesPayload):
            arm.detach_payload()
        cell.mark_jaws_emptied()
        # Hands were at the jaws: the next motion the robot makes by itself counts down first.
        cell.stamp_jaws_emptied()
        emptied = True
    logger.warning("A person said the cell is clear: cleared %s%s.", ", ".join(cleared) or "nothing that was set",
                   "; the jaws are empty" if emptied else "")
    cell.hub.publish(
        CELL_STREAM, "cell.acknowledged", severity=Severity.INFO,
        human=("A person confirmed the cell is clear" + (f": {', '.join(cleared)} cleared" if cleared else "")
               + ("; the hand holds nothing" if emptied else "") + "."),
        cleared=cleared, jaws_emptied=emptied,
    )
    return _render(cell)


def _brake_answer(arm: Any, reason: str) -> tuple[bool, bool, bool]:
    """Latch ``arm`` where it has the latch (``SupportsHalt``): whether it latched, whether a move was in flight, and
    whether that move is being braked (an arm that brakes, whose brake has not ended yet). Sends nothing from here."""
    from src.robot.core.arm_capabilities import SupportsHalt, brakes_in_motion_of  # noqa: PLC0415

    if not isinstance(arm, SupportsHalt):
        return False, False, False
    try:
        state = arm.halt(reason)
    except Exception as exc:  # noqa: BLE001 (a latch that raised latched nothing anybody can count on)
        logger.error("The arm's halt raised %s: %s; the run is told to halt regardless.", type(exc).__name__, exc)
        return False, False, False
    in_motion = getattr(state, "in_motion", False) is True
    braking = brakes_in_motion_of(arm) and getattr(state, "brake", "") == "pending"
    return True, in_motion, braking


@router.post("/brake", response_model=BrakeOut,
             summary="Halt now: stops the run and latches the arm; brakes a move in flight where enabled; not an "
                     "emergency stop")
def post_brake(cell: Annotated[Console, Depends(console)]) -> BrakeOut:
    """Halt now: stops the run and latches the arm; not an emergency stop; the red button is.

    The run stops commanding, and where the arm latches, every next motion and output switch is refused before
    anything is sent, until a person confirms the cell is clear. Only where the arm also brakes a move in flight
    (``robot.ur.brake_on_halt``, off as shipped) is that move braked under control; elsewhere the move in flight runs
    to its end and nothing after it is sent. A request over a socket depends on latency, an open tab and an awake
    laptop, which is why the red button stays the safety halt."""
    session = cell.session
    # No session lock: a connect may hold it while a jaws question waits, and a halt must never wait for that.
    if session.state is not CellState.CONNECTED:
        raise refusal("not_connected", f"the cell is {session.state}: nothing is connected to halt.",
                      state=str(session.state))
    reason = "the operator pressed halt now in the console"
    # Counted before the latch: a press that lands on a latch a Disconnect set for its teardown keeps that latch.
    cell.note_halt_pressed()
    # The arm first: from here on it refuses every next motion and output, whatever the run does next.
    latched, in_motion, braking = _brake_answer(session.arm, reason)
    if latched:
        # Carried to the arm of every later build and across a restart of the server, as the arm keeps its first record:
        # only a person's "the cell is clear" ends a halt, never a rebuild.
        from src.robot.core.arm_capabilities import halt_state_of  # noqa: PLC0415

        state = halt_state_of(session.arm)
        cell.carry_halt(state.reason if state is not None else reason,
                        state.requested_at if state is not None else None)
    run = cell.registry.halt(reason, in_motion=in_motion, braking=braking)
    if run is None:
        cell.hub.publish(
            CELL_STREAM, "cell.halted", severity=Severity.WARN,
            human=("Halt now: the arm refuses every next motion until a person confirms the cell is clear."
                   if latched else "Halt now: no run was active, and this arm has no latch to set."),
            reason=reason, in_motion=in_motion,
        )
    if latched:
        message = ("Halted: the run commands nothing more, and the arm refuses every next motion and output until a "
                   "person confirms the cell is clear. "
                   + ("The move in flight is braked under control. " if braking else
                      "A move in flight runs to its end; nothing after it is sent. " if in_motion else "")
                   + "Not an emergency stop; the red button is.")
    else:
        message = ("Halted as far as this arm allows: it has no latch, so the run commands nothing more and the "
                   "current motion ends first. Not an emergency stop; the red button is.")
    logger.warning("Halt now (%s): run %s, latched %s, in motion %s, braking %s.", reason,
                   run.id if run is not None else "none", latched, in_motion, braking)
    return BrakeOut(run_halted=run is not None, latched=latched, braking=braking, in_motion=in_motion,
                    run_id=run.id if run is not None else None, message=message)


def _named_pose(cell: Console, name: str) -> tuple[str, list[float]] | None:
    """A taught pose's label and joints (degrees), or ``None`` for a name nobody taught."""
    pose = cell.robot().named_poses.get(name)
    if pose is None:
        return None
    return pose.label or name, [float(value) for value in pose.joints_deg]


@router.post("/home", response_model=RunOut, status_code=202, summary="Drive to Home or a taught pose (THIS MOVES)")
def post_home(cell: Annotated[Console, Depends(console)], body: HomeIn) -> RunOut:
    """A planned move, started only by this click, after a 3 s countdown where a person's hands were last at the arm.
    Refused during a run, on a halted or stopped controller, while a person is needed, and while a part is held.

    Its arrival ends the console's recovery record: with Restart, it is the way back after a stop, once a person
    confirmed the cell is clear.
    """
    from api.task_run import drive_home  # noqa: PLC0415

    record = cell.recovery
    gates.refuse_unless(cell, gates.HOME_GATES, record=record)
    to = body.to.strip()
    inputs: dict[str, Any] = {"to": to}
    if to != "home":
        named = _named_pose(cell, to)
        if named is None:
            known = ", ".join(sorted(cell.robot().named_poses)) or "none is"
            raise refusal("unknown_pose", f"no pose called {to!r} was taught ({known}); Home is 'home'.", which="home")
        inputs["label"], inputs["joints_deg"] = named
    try:
        run = cell.registry.start_kind(cell, RunKind.HOME, drive_home, inputs=inputs)
    except RunConflict as busy:
        raise refusal("run_active", str(busy), run_id=busy.existing.id) from busy
    except RunRefused as stopped:
        raise refusal(stopped.code, str(stopped), **stopped.detail) from stopped
    return RunOut(**run.as_accepted())


@router.post("/wave", response_model=RunOut, status_code=202, summary="Wave back at a greeting (THIS MOVES)")
def post_wave(cell: Annotated[Console, Depends(console)]) -> RunOut:
    """Willy waves: the second wrist joint swings 15 degrees either way, twice, and back, each swing a straight joint
    line the exact guard judges against the camera world before it is sent (``src.robot.execution.gestures``). The
    console's answer to a greeting in the chat, at once or on a click as the app config says
    (``runtime.greeting.wave``), after a 3 s countdown where a person's hands were last at the arm.

    Refused as a new task is, bar a carried part: during a run, on a halted or stopped controller, before the cell is
    cleared after a stop and while a Restart is required (a wave is no way back), while a person is needed, while the
    jaws question waits or a toggle's jaws are not confirmed, while a part is held, and on an arm whose motions are
    refused.
    """
    from api.task_run import drive_wave  # noqa: PLC0415

    gates.refuse_unless(cell, gates.WAVE_GATES, record=cell.recovery)
    try:
        run = cell.registry.start_kind(cell, RunKind.WAVE, drive_wave)
    except RunConflict as busy:
        raise refusal("run_active", str(busy), run_id=busy.existing.id) from busy
    except RunRefused as stopped:
        raise refusal(stopped.code, str(stopped), **stopped.detail) from stopped
    return RunOut(**run.as_accepted())


@router.post("/planner", response_model=RunOut, status_code=202, summary="Start the planner (moves nothing)")
def post_planner(cell: Annotated[Console, Depends(console)]) -> RunOut:
    """Starts cuRobo, which takes about a minute on a cell and moves nothing. Connect starts it by itself on a cuRobo
    arm whose planner is off; this starts it again after it stopped or failed."""
    from api.task_run import drive_planner  # noqa: PLC0415

    session = cell.session
    if session.service is None:
        raise refusal("not_built", "nothing is built yet: build the cell first.")
    active = cell.registry.active()
    if active is not None:
        raise refusal("run_active", f"run {active.id} ({active.kind}) is active.", run_id=active.id)
    if _planner(session.arm).state == "not_used":
        raise refusal("planner_not_used", ("this arm's motions go through no cuRobo planner (the dummy, an ik arm, a "
                                           "KUKA): there is nothing to start."))
    try:
        run = cell.registry.start_kind(cell, RunKind.PLANNER, drive_planner)
    except RunConflict as busy:
        raise refusal("run_active", str(busy), run_id=busy.existing.id) from busy
    return RunOut(**run.as_accepted())


__all__ = ["CellState", "router"]
