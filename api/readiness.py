"""The ready bar, the cell's facts, and the gates every moving route keeps: can the next motion start, and what is this
cell.

One module, because they are one question asked three ways. The ready bar says before a button is pressed what the
server refuses when it is pressed, so both read the same things here, in the same words:

* :func:`readiness` answers ``GET /v1/cell/readiness`` (build plan 1.5): six lights (the robot, the cameras, the
  planner, the gripper, the carried part, the commands) and the blockers. It always answers and moves nothing;
* :func:`refuse_unless` is every moving route's gate (1.3.2, 1.3.5, 1.3.6): the cell's state checked in one fixed
  order, before anything is started, and the first refusal raised in the one error envelope;
* :func:`facts` answers ``GET /v1/cell/facts`` (1.10).

**The controller is read from the receive stream alone** (:func:`controller_reading`): ``quick_robot_status()`` where
the arm has it, never ``get_robot_status()`` on an arm that has both, which asks the dashboard on every call. An arm
that reports no controller at all (the dummy) is read as fine; an arm that reports one only through
``get_robot_status`` (a double, a sim) is read through that.

**A halt is checked before the controller and apart from it**: the arm's latch (``halt_state_of``, library L9) first,
then the controller's own fields (``RobotStatus.controller_operational``), so a halted arm reads ``halted`` everywhere
and never as a stopped controller, whose remedy (the pendant) is not the halt's (a person says the cell is clear).

**One rule for a part still held** (:func:`part_held`), the library's (``task._part_may_be_held``): a toggle's count
that says CLOSED, a planner that still models a part (``payload_model`` not one of ``NO_PART_MODELLED``), or a
measuring hand that measures one. Where a recovery record stands, the record's own belief counts too for a hand that
can say nothing (no toggle, no measurement): only the person's "Backen leer" ends it.

What the console reads of the halt latch is named in :class:`HaltStateLike` and :class:`ReadsHaltState`; the latch is
read on every poll of ``GET /v1/cell`` and by every gate, so ``halt_state()`` must be a lock-free read that never raises
and never blocks.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from fastapi import HTTPException

from api.codes import REFUSAL_STATUS, RefusalCode, RunKind, StopCode
from api.constants import API_LOG_DIR, READINESS_LOG_FILE
from api.schemas import (
    NO_PART_MODELLED,
    BlockerOut,
    BrakeFactsOut,
    CellFactsOut,
    DetectorFactsOut,
    HandOut,
    LightOut,
    PayloadFactsOut,
    PayloadModelName,
    PushFactsOut,
    ReadinessOut,
    RigOut,
    RouteFactsOut,
)
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from api.cell import Console, RecoveryRecord

__all__ = [
    "HaltStateLike",
    "ReadsHaltState",
    "controller_gate",
    "controller_reading",
    "detector_precision",
    "facts",
    "halt_reason",
    "part_held",
    "payload_model_of",
    "readiness",
    "refuse_unless",
    "refusal",
]

logger = create_logger("Readiness", READINESS_LOG_FILE, log_dir=API_LOG_DIR)

#: How old a rig's last frame may be for the camera light to read live (``api.live.STALE_AFTER_S``).
_CAMERA_FRESH_S = 2.0
#: How long a "weights on this box" answer is kept for the commands light: the check walks the model cache.
_WEIGHTS_CACHE_S = 30.0
_WEIGHTS_SEEN: dict[tuple[str, str | None], tuple[float, bool]] = {}


class HaltStateLike(Protocol):
    """The arm's halt latch while it is set, as the console reads it: the library's ``HaltState`` (library L9).

    Read-only, so a frozen dataclass answers it.
    """

    @property
    def reason(self) -> str:
        """Why "halt now" was pressed."""
        ...

    @property
    def requested_at(self) -> float:
        """When, Unix seconds."""
        ...

    @property
    def in_motion(self) -> bool:
        """A motion was in flight."""
        ...

    @property
    def braked(self) -> bool:
        """The move in flight was braked (``robot.ur.brake_on_halt``); false where it ran to its end first."""
        ...

    @property
    def brake_s(self) -> float | None:
        """How long the brake took, where it braked."""
        ...

    @property
    def brake(self) -> str:
        """What became of the move in flight: ``none``, ``pending``, ``braked``, ``unconfirmed`` or ``ran_out``."""
        ...


@runtime_checkable
class ReadsHaltState(Protocol):
    """An arm with the halt latch (library L9's ``SupportsHalt``), as the console reads it; the console never clears
    it here, only ``POST /v1/cell/acknowledge`` does."""

    def halt_state(self) -> HaltStateLike | None:
        """The latch while it is set, else ``None``."""
        ...


# ---------------------------------------------------------------------------------------------------------------------
# What every gate reads
# ---------------------------------------------------------------------------------------------------------------------


def refusal(code: RefusalCode | str, message: str, **detail: Any) -> HTTPException:
    """The one error envelope for ``code``, at the one status the catalog gives it."""
    typed = RefusalCode(code)
    return HTTPException(status_code=REFUSAL_STATUS[typed],
                         detail={"code": str(typed), "message": message, "detail": dict(detail)})


def halt_reason(arm: Any) -> str:
    """Why the arm's halt latch is set, or ``""`` (an arm that is not halted or has none). Strict: only the library's
    ``HaltState`` counts, so a double that answers every attribute is not halted."""
    from src.robot.core.arm_capabilities import halt_state_of  # noqa: PLC0415

    state = halt_state_of(arm)
    if state is None:
        return ""
    return str(state.reason) or "the arm's halt latch is set"


def controller_reading(arm: Any) -> tuple[str, str]:
    """The controller's own fields, the halt latch aside: ``("", "")`` where it can move or the arm reports nothing,
    ``("controller_stopped", why)`` where its fields say it cannot, ``("controller_unreadable", why)`` where they
    cannot be read or do not say. From the receive stream (``quick_robot_status``) where the arm has it; never the
    dashboard there.

    The console's one rule for the controller: only ``controller_operational`` true lets anything go. Every gate reads
    it here, the moving routes' and the ready bar's, and the jaws' own (``api.jaws.controller_stop``, the gate before
    the one change of the output, the jaws check and a teach)."""
    quick = getattr(arm, "quick_robot_status", None)
    read = quick if callable(quick) else None
    if read is None:
        from src.robot.core.arm_capabilities import SupportsRobotStatus  # noqa: PLC0415

        if not isinstance(arm, SupportsRobotStatus):
            return "", ""
        read = arm.get_robot_status
    try:
        status = read()
    except Exception as exc:  # noqa: BLE001 (a controller nobody can read moves nothing anybody can vouch for)
        return "controller_unreadable", (f"the controller's state could not be read ({type(exc).__name__}: {exc}), so "
                                         "nothing is started on it")
    operational = getattr(status, "controller_operational", None)
    if operational is True:
        return "", ""
    said = _status_said(status)
    if operational is None:
        return "controller_unreadable", (f"the controller's status does not say whether it can move{said}, and only a "
                                         "controller that says it can is trusted: look at the pendant first; nothing "
                                         "is started")
    return "controller_stopped", (
        f"the controller cannot move{said}: a protective or emergency stop, or a power-off. A person clears it at the "
        "pendant, where the arm is visible; the console never does")


def _status_said(status: Any) -> str:
    """The controller's fields as one bracketed clause (`` (robot mode ..., a protective stop)``): modes and stops, as
    the receive stream says them; ``""`` where it says none."""
    parts = []
    for name, label in (("robot_mode", "robot mode"), ("safety_mode", "safety mode")):
        value = getattr(status, name, None)
        if value is not None:
            parts.append(f"{label} {getattr(value, 'value', value)}")
    if getattr(status, "protective_stopped", False) is True:
        parts.append("a protective stop")
    if getattr(status, "emergency_stopped", False) is True:
        parts.append("an emergency stop")
    return f" ({', '.join(parts)})" if parts else ""


def controller_gate(arm: Any) -> str:
    """Why the controller cannot move, from its own fields alone; ``""`` where it can, or where the arm reports nothing.

    The halt latch is checked before this and apart from it (:func:`halt_reason`).
    """
    return controller_reading(arm)[1]


def payload_model_of(arm: Any) -> PayloadModelName:
    """What models the part the gripper carries now, as the arm says it; ``not_applicable`` for an arm that models no
    part at all (the dummy, sim). ``unknown`` where it models parts and its answer could not be read, or is one this
    console does not know: ``unknown`` is not in ``NO_PART_MODELLED``, so a part nobody can rule out keeps every gate
    shut."""
    from src.robot.core.arm_capabilities import CarriesPayload  # noqa: PLC0415

    try:
        if arm is None or not isinstance(arm, CarriesPayload):
            return "not_applicable"
        answer = arm.payload_model()
        model = str(getattr(answer, "value", answer))
    except Exception:  # noqa: BLE001 (a read that fails rules out no part)
        return "unknown"
    if model == "none":
        return "none"
    if model == "filter_only":
        return "filter_only"
    if model == "planner_and_filter":
        return "planner_and_filter"
    return "unknown"


def _hold_unmeasured(gripper: Any) -> bool:
    """Whether the hand measures nothing about a hold (no toggle count, no hold evidence): only a person can say."""
    from src.robot.core.gripper import HoldEvidence, hold_evidence_of  # noqa: PLC0415

    try:
        return hold_evidence_of(gripper) is HoldEvidence.UNMEASURED
    except Exception:  # noqa: BLE001 (a hand nobody can read measures nothing)
        return True


def part_held(arm: Any, gripper: Any, hand: HandOut, *, record: "RecoveryRecord | None" = None) -> str:
    """Why a part may still be in the jaws now, or ``""``: one rule with the library.

    A toggle's count that says CLOSED, a planner that still models a part (``payload_model`` not in
    ``NO_PART_MODELLED``), or a measuring hand that measures one. With ``record`` (a recovery record that stands): the
    record's own belief too, for a hand that can say nothing (not a toggle, and no measurement), until the person's
    "Backen leer" (``POST /v1/cell/acknowledge`` with ``jaws_empty``) ends it.
    """
    from src.robot.core.gripper import HoldEvidence, hold_evidence_of  # noqa: PLC0415

    if hand.kind == "toggle" and hand.jaws == "closed":
        return "the program's count says the toggle's jaws stand CLOSED: a part may be in them"
    model = payload_model_of(arm)
    if model not in NO_PART_MODELLED:
        return f"the planner still models a part in the hand (payload_model {model})"
    try:
        measured = gripper is not None and hand.connected and hold_evidence_of(gripper) is HoldEvidence.HELD
    except Exception as exc:  # noqa: BLE001 (a hand nobody can read may hold anything)
        return f"the hand could not be read ({type(exc).__name__}: {exc})"
    if measured:
        return "the gripper measures a part in its jaws"
    if record is not None and record.holding and hand.kind != "toggle" and _hold_unmeasured(gripper):
        return ("the stopped run left a part in the jaws, and this hand measures nothing: empty the hand and say so "
                "(\"Backen leer\")")
    return ""


def _needs_person(service: Any) -> str:
    """Why a pick stopped where the arm stands, a person to decide (the service's latch); ``""`` while none waits.

    Strict: only a string; a latch that cannot be read is said, never ``""``.
    """
    try:
        said = getattr(service, "stopped_where_the_arm_stands", "")
    except Exception as exc:  # noqa: BLE001
        return f"the service's stop could not be read ({type(exc).__name__}: {exc})"
    return said if isinstance(said, str) else ""


def _question_waiting(console: "Console") -> bool:
    from api import jaws as browser_jaws  # noqa: PLC0415

    try:
        return browser_jaws.pending(console) is not None
    except Exception:  # noqa: BLE001 (a question nobody can read may wait)
        return True


def _hand(gripper: Any) -> HandOut:
    from api import jaws as browser_jaws  # noqa: PLC0415

    return browser_jaws.hand_of(gripper)


# ---------------------------------------------------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------------------------------------------------

#: The gates a new task keeps, in the order of build plan 1.3.2 (the cell's state; the request's own come after).
TASK_GATES: tuple[str, ...] = (
    "not_connected", "run_active", "halted", "controller_stopped", "cell_not_cleared", "restart_required",
    "needs_person", "jaws_question_pending", "part_still_held", "jaws_not_confirmed", "route_refused",
    "carried_part_not_modelled",
)
#: A Restart's: every gate of a task but ``restart_required``, the hand read against the record (1.3.5).
RESTART_GATES: tuple[str, ...] = tuple(gate for gate in TASK_GATES if gate != "restart_required")
#: Home's (1.3.6): it is the way back while a Restart is required, and it carries no part.
HOME_GATES: tuple[str, ...] = (
    "not_connected", "run_active", "halted", "controller_stopped", "cell_not_cleared", "needs_person",
    "jaws_question_pending", "part_still_held", "jaws_not_confirmed", "route_refused",
)
#: A wave's: a task's, bar a carried part (no wave carries one): a greeting never starts the way back after a stop.
WAVE_GATES: tuple[str, ...] = (
    "not_connected", "run_active", "halted", "controller_stopped", "cell_not_cleared", "restart_required",
    "needs_person", "jaws_question_pending", "part_still_held", "jaws_not_confirmed", "route_refused",
)
#: ``POST /v1/pick``'s (item 17), after its own ``not_connected``, in 1.3.2's order: a run holding the lock first, so a
#: run that ends while these are read cannot slip past them unseen (the registry reads the record once more as it
#: starts the run, under the lock).
PICK_GATES: tuple[str, ...] = (
    "run_active", "halted", "cell_not_cleared", "restart_required", "needs_person", "jaws_question_pending",
)


def refuse_unless(console: "Console", gates: Sequence[str], *, record: "RecoveryRecord | None" = None) -> None:
    """Raise the first of ``gates`` the cell fails now, in their order, as its refusal; return where none does.

    ``record`` is the recovery record a Restart or a Home reads the hand against (the jaws answered open since the
    stop; the record's belief that a part is held). Reads only: nothing is asked, cleared or commanded.
    """
    from api.lifecycle import CellState  # noqa: PLC0415

    session = console.session
    arm, gripper, service = session.arm, session.gripper, session.service
    hand: HandOut | None = None
    for gate in gates:
        if gate == "not_connected":
            if session.state is not CellState.CONNECTED:
                raise refusal(gate, f"the cell is {session.state}; connect it first.", state=str(session.state))
        elif gate == "run_active":
            active = console.registry.active()
            if active is not None:
                raise refusal(gate, (f"run {active.id} ({active.kind}) is active: one arm, one run. A second would not "
                                     "queue, it would collide; wait until it ended."), run_id=active.id)
        elif gate == "halted":
            why = halt_reason(arm)
            if why:
                raise refusal(gate, (f"the arm is halted ({why}): every motion and every output is refused until a "
                                     "person confirms the cell is clear (POST /v1/cell/acknowledge)."), reason=why)
        elif gate == "controller_stopped":
            code, why = controller_reading(arm)
            if code:
                raise refusal("controller_stopped", why, reading=code)
        elif gate in ("cell_not_cleared", "restart_required"):
            gated = console.recovery_gate()
            standing = console.recovery
            if gated == gate and standing is not None:
                if gate == "cell_not_cleared":
                    message = (f"the {standing.kind} run {standing.run_id} stopped where the arm stands "
                               f"({standing.stop_code}): a person confirms the cell is clear first.")
                else:
                    message = (f"the {standing.kind} run {standing.run_id} stopped where the arm stands "
                               f"({standing.stop_code}): the way back is Restart or Home, whose first motion is the "
                               "planned move to the return pose; nothing new starts before it.")
                raise refusal(gate, message, run_id=standing.run_id, stop_code=str(standing.stop_code))
        elif gate == "needs_person":
            why = _needs_person(service)
            if why:
                raise refusal(gate, (f"a recovery stopped where the arm stands and a person decides first: {why}. "
                                     "Confirm the cell is clear (POST /v1/cell/acknowledge)."))
        elif gate == "jaws_question_pending":
            if _question_waiting(console):
                raise refusal(gate, "a jaws question waits for its answer (GET /v1/cell/jaws); nothing moves before.")
        elif gate == "part_still_held":
            hand = hand if hand is not None else _hand(gripper)
            why = part_held(arm, gripper, hand, record=record)
            if why:
                raise refusal(gate, f"{why}; nothing that moves by itself carries a part outside a task.")
        elif gate == "jaws_not_confirmed":
            hand = hand if hand is not None else _hand(gripper)
            if hand.kind == "toggle":
                if hand.jaws != "open":
                    raise refusal(gate, ("nobody can say where the toggle's jaws stand ("
                                         f"{hand.why_unknown or hand.jaws}): answer the jaws question first "
                                         "(POST /v1/cell/jaws/check)."), jaws=hand.jaws)
                if record is not None and not console.jaws_answered_since(record):
                    raise refusal(gate, ("the toggle's jaws were not answered open since the stop: the way back "
                                         "asks where they stand first (POST /v1/cell/jaws/check)."), jaws=hand.jaws)
        elif gate == "route_refused":
            from src.robot.execution.motion import route_of  # noqa: PLC0415

            route = route_of(arm)
            if not route.runs:
                raise refusal(gate, (f"this arm's place and return go through no planner that judges them, so a part "
                                     f"picked here could not be set down: {route.reason}"))
        elif gate == "carried_part_not_modelled":
            from src.robot.core.arm_capabilities import CarriesPayload, chose_no_carried_part  # noqa: PLC0415

            # A cell that chose to model no carried part carries its parts as an empty hand (the owner, 2026-10-07).
            if isinstance(arm, CarriesPayload) and not chose_no_carried_part(arm):
                try:
                    declined = arm.payload_declined_reason()
                except Exception as exc:  # noqa: BLE001 (a model nobody can read models nothing)
                    declined = f"the arm could not say ({type(exc).__name__}: {exc})"
                if declined is not None:
                    raise refusal(gate, (
                        f"a task carries every part it picks, and this arm models no carried part ({declined}): "
                        "between the close and the release only the planner holds it. Declare "
                        "safety.planning_world.payload.length_mm in the cell profile."), reason=declined)
        else:  # pragma: no cover - a programmer's error
            raise ValueError(f"no gate {gate!r}")


# ---------------------------------------------------------------------------------------------------------------------
# The ready bar
# ---------------------------------------------------------------------------------------------------------------------


def _light(light: str, state: str, code: str, message: str, *, blocks: bool = True) -> LightOut:
    return LightOut(id=light, state=state, code=code, message=message, blocks=blocks)  # type: ignore[arg-type]


def _robot_light(console: "Console") -> LightOut:
    from api.lifecycle import CellState  # noqa: PLC0415

    session = console.session
    if session.service is None:
        return _light("robot", "blocked", "not_built", "nothing is built yet: build the cell")
    if session.state is not CellState.CONNECTED:
        return _light("robot", "blocked", "not_connected", f"the cell is {session.state}: connect it")
    why = halt_reason(session.arm)
    if why:
        return _light("robot", "blocked", "halted", f"the arm is halted ({why}): a person confirms the cell is clear")
    code, said = controller_reading(session.arm)
    if code:
        return _light("robot", "blocked", code, said)
    return _light("robot", "ok", "connected", "connected, and the controller can move")


def rehearsal_of(service: Any) -> bool:
    """Whether the cell runs the rehearsal scene: a picture this process draws, no camera behind it."""
    perception = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "perception", None)
    return getattr(perception, "colour_source_kind", None) == "synthetic"


def _camera_light(console: "Console") -> LightOut:
    from src.robot.execution.lifecycle import service_cameras  # noqa: PLC0415

    service = console.session.service
    if service is None:
        return _light("cameras", "blocked", "none", "nothing is built yet")
    if rehearsal_of(service):
        return _light("cameras", "ok", "rehearsal", "the rehearsal scene: a picture this console draws, no camera")
    rigs = [str(camera.rig_id) for camera in service_cameras(service)]
    if not rigs:
        return _light("cameras", "blocked", "none", "this cell holds no camera to look with")
    try:
        console.live.refresh_stale(service, max_age_s=_CAMERA_FRESH_S)
        ages = console.live.ages()
    except Exception as exc:  # noqa: BLE001 (a camera nobody can read shows nothing)
        return _light("cameras", "blocked", "no_frame", f"the cameras could not be read ({type(exc).__name__}: {exc})")
    stale = [rig for rig in rigs if not (isinstance(ages.get(rig), (int, float)) and ages[rig] < _CAMERA_FRESH_S)]
    if stale:
        return _light("cameras", "blocked", "no_frame", f"no frame younger than {_CAMERA_FRESH_S:g} s from "
                                                        f"{', '.join(stale)}")
    return _light("cameras", "ok", "live", f"{len(rigs)} camera(s) live")


def _planner_said(arm: Any) -> str:
    try:
        state = getattr(arm, "planner_state", None)
        state = state() if callable(state) else state
    except Exception:  # noqa: BLE001 (a planner that cannot say is not ready)
        return "off"
    return state if state in ("off", "starting", "ready") else "not_used"


def _planner_light(console: "Console") -> LightOut:
    from src.robot.execution.motion import MotionRoute, route_of  # noqa: PLC0415

    arm = console.session.arm
    if console.session.service is None:
        return _light("planner", "blocked", "off", "nothing is built yet")
    route = route_of(arm)
    if route.route is MotionRoute.REFUSED:
        return _light("planner", "blocked", "route_refused", route.reason)
    if route.route is MotionRoute.UNPLANNED:
        return _light("planner", "ok", "unplanned", f"a desk arm: nothing real moves ({route.reason})")
    state = _planner_said(arm)
    if state == "ready":
        return _light("planner", "ok", "ready", "cuRobo is ready")
    if state == "starting":
        return _light("planner", "wait", "starting", "cuRobo is starting; this takes about a minute")
    if state == "not_used":
        return _light("planner", "ok", "ready", f"this arm plans its own paths: {route.reason}")
    last = next((run for run in console.registry.recent(50) if run.kind is RunKind.PLANNER), None)
    if last is not None and last.stop_code is StopCode.PLANNER_FAILED:
        return _light("planner", "blocked", "failed", f"the planner did not start: {last.error}")
    return _light("planner", "blocked", "off", "cuRobo is not started: start the planner (about a minute)")


def _gripper_light(console: "Console", hand: HandOut, waiting: bool) -> LightOut:
    if console.session.gripper is None or hand.kind == "none":
        return _light("gripper", "blocked", "none", "this cell has no hand to pick with")
    if waiting:
        return _light("gripper", "blocked", "question_pending", "a jaws question waits for its answer")
    if hand.kind == "toggle":
        if hand.jaws == "open":
            return _light("gripper", "ok", "open_confirmed", "the jaws stand open, as counted (no sensor)")
        if hand.jaws == "closed":
            return _light("gripper", "blocked", "jaws_closed", "the program's count says the jaws stand closed")
        return _light("gripper", "blocked", "jaws_unknown", hand.why_unknown or "nobody can say where the jaws stand")
    if not hand.connected:
        return _light("gripper", "blocked", "none", "the hand is not connected")
    return _light("gripper", "ok", "connected", f"{hand.driver} connected")


def _carried_part_light(console: "Console") -> LightOut:
    from src.robot.core.arm_capabilities import CarriesPayload, chose_no_carried_part  # noqa: PLC0415

    arm = console.session.arm
    if arm is None or not isinstance(arm, CarriesPayload):
        return _light("carried_part", "ok", "not_applicable", "this arm carries nothing real (a desk arm)")
    if chose_no_carried_part(arm):
        # A grey fact, no block: the cell chose it (the owner, 2026-10-05 and 2026-10-07).
        return _light("carried_part", "info", "not_modelled", (
            "safety.planning_world.payload.enabled is false: a closed hand is judged as an empty one, by the exact "
            "guard at full stroke, and the part it holds by nobody"), blocks=False)
    try:
        declined = arm.payload_declined_reason()
    except Exception as exc:  # noqa: BLE001
        declined = f"the arm could not say ({type(exc).__name__}: {exc})"
    if declined is not None:
        return _light("carried_part", "blocked", "not_modelled", declined)
    return _light("carried_part", "ok", "modelled", "the planner carries the part between the close and the release")


def _built_models(console: "Console") -> Any:
    models = getattr(console, "built_models", None)
    if models is not None:
        return models
    try:
        return console.config().models
    except Exception:  # noqa: BLE001
        return None


def _weights_present(model_id: str, model_path: str | None) -> bool:
    from api.routers.diagnostics import _vlm_weights_present  # noqa: PLC0415

    key = (model_id, model_path)
    seen = _WEIGHTS_SEEN.get(key)
    now = time.monotonic()
    if seen is not None and now - seen[0] < _WEIGHTS_CACHE_S:
        return seen[1]
    present = bool(_vlm_weights_present(model_id, model_path))
    _WEIGHTS_SEEN[key] = (now, present)
    return present


def _commands_light(console: "Console") -> LightOut:
    """What reading a command can do now: informational only, it never blocks Start (owner decision 22)."""
    try:
        from src.models.vlm import reader_availability  # noqa: PLC0415

        models = _built_models(console)
        pipeline = getattr(models, "pipeline", None)
        present: bool | None = None
        if pipeline is not None:
            vlm = pipeline.zero_shot.vlm
            present = _weights_present(str(vlm.model_id), getattr(vlm, "model_path", None))
        available = reader_availability(models, weights_present=present)
    except Exception as exc:  # noqa: BLE001 (an informational light never fails the bar)
        return _light("commands", "info", "failed", f"the command reader could not be asked ({type(exc).__name__}: "
                                                    f"{exc})", blocks=False)
    state = {"ready": "ok", "loading": "wait"}.get(available.state, "info")
    return _light("commands", state, available.state, available.cause, blocks=False)


def _blockers(console: "Console", hand: HandOut, waiting: bool) -> list[BlockerOut]:
    blockers: list[BlockerOut] = []
    active = console.registry.active()
    if active is not None:
        blockers.append(BlockerOut(code="run_active", message=f"run {active.id} ({active.kind}) is active",
                                   run_id=active.id))
    gate = console.recovery_gate()
    record = console.recovery
    if gate and record is not None:
        message = ("a person confirms the cell is clear first" if gate == "cell_not_cleared"
                   else "the way back is Restart or Home")
        blockers.append(BlockerOut(code=gate, message=f"the {record.kind} run {record.run_id} stopped "
                                                      f"({record.stop_code}): {message}", run_id=record.run_id))
    needs = _needs_person(console.session.service)
    if needs:
        blockers.append(BlockerOut(code="needs_person", message=needs))
    session = console.session
    if session.service is not None:
        held = part_held(session.arm, session.gripper, hand, record=record)
        if held:
            blockers.append(BlockerOut(code="part_still_held", message=held))
    if waiting:
        blockers.append(BlockerOut(code="jaws_question_pending", message="a jaws question waits for its answer"))
    return blockers


def readiness(console: "Console") -> ReadinessOut:
    """Every light and every blocker, read now: ``ready`` where every light that blocks is ok and nothing blocks.
    Always answers, moves nothing, and reads the controller from the receive stream alone."""
    session = console.session
    hand = _hand(session.gripper)
    waiting = _question_waiting(console)
    lights = [
        _robot_light(console),
        _camera_light(console),
        _planner_light(console),
        _gripper_light(console, hand, waiting),
        _carried_part_light(console),
        _commands_light(console),
    ]
    blockers = _blockers(console, hand, waiting)
    ready = all(light.state == "ok" for light in lights if light.blocks) and not blockers
    return ReadinessOut(ready=ready, lights=lights, blockers=blockers)


# ---------------------------------------------------------------------------------------------------------------------
# The cell's facts
# ---------------------------------------------------------------------------------------------------------------------


def _mounting(camera: Any) -> str:
    said = str(getattr(getattr(camera, "rig", None), "mounting", "") or getattr(camera, "mounting", "") or "")
    if said in ("eye_in_hand", "wrist"):
        return "wrist"
    if said in ("eye_to_hand", "fixed"):
        return "fixed"
    return "unknown"


def _push_facts(service: Any) -> PushFactsOut:
    from src.robot.grasping.recovery.push_gate import PushCell  # noqa: PLC0415
    from src.robot.grasping.recovery.push_planner import PUSH_DISTANCE_CAP_MM  # noqa: PLC0415

    cell = getattr(service, "push_cell", None)
    default: float | None = None
    distance = getattr(service, "push_distance", None)
    if callable(distance):
        try:
            default = float(distance())
        except Exception:  # noqa: BLE001
            default = None
    fixture = getattr(getattr(getattr(service, "effective_config", None), "recovery_orchestrator", None), "fixture",
                      None)
    ceiling: float | None = None
    if isinstance(fixture, (tuple, list)) and len(fixture) > 2 and isinstance(fixture[2], (int, float)):
        ceiling = float(fixture[2])
    elif callable(distance):
        # The service's own rule (``push_distance``): a cell that declares no fixture is held to the hard cap alone.
        ceiling = float(PUSH_DISTANCE_CAP_MM)
    recovery = getattr(getattr(service, "effective_config", None), "recovery_orchestrator", None)
    critical = bool(getattr(recovery, "critical_parts", False) is True)
    into = bool(getattr(recovery, "blocker_into_the_place", True) is not False)
    on = bool(getattr(recovery, "enabled", False))
    named = tuple(str(a) for a in (getattr(recovery, "allowed_actions", ()) or ())) if on else ()
    rescan_allowed, push_allowed = "rescan" in named, "nudge_target" in named
    if isinstance(cell, PushCell):
        return PushFactsOut(can_push=True, default_mm=default, ceiling_mm=ceiling, critical_parts=critical,
                            blocker_into_the_place=into, rescan_allowed=rescan_allowed, push_allowed=push_allowed)
    why = str(getattr(cell, "sentence", "") or "") if cell is not None else (
        "this cell's recovery pushes no part (grasping.recovery.allowed_actions names no nudge_target)")
    return PushFactsOut(can_push=False, why_not=why, default_mm=default, ceiling_mm=ceiling, critical_parts=critical,
                        blocker_into_the_place=into, rescan_allowed=rescan_allowed, push_allowed=push_allowed)


#: What a phrase grounder's weights and arithmetic are where its config leaves ``optim.torch_dtype`` unset: the HF
#: default of fp32 weights, run under CUDA's fp16 autocast (``src/models/_inference.py``, ``autocast_ctx``).
_UNSET_DTYPE = "fp32 weights, fp16 autocast"
#: The VLM loads in the checkpoint's own dtype (``src/models/vlm/qwen.py``: ``dtype="auto"``), FP8 for an -FP8 id.
_VLM_DTYPE = "auto (the checkpoint's own)"


def detector_precision(models: Any, backend: str, router_enabled: bool) -> str | None:
    """The precision the cell's detector runs at, read from the models config it was built with (``optim.torch_dtype``,
    as ``build_load_kwargs`` reads it); ``None`` where there is no config to read. A routed stack names both of its
    grounders. Loads nothing."""
    if models is None:
        return None

    def configured(block: str) -> str:
        dtype = getattr(getattr(getattr(models, block, None), "optim", None), "torch_dtype", None)
        return str(dtype) if dtype else _UNSET_DTYPE

    if backend == "vlm":
        return f"VLM {_VLM_DTYPE}; GroundingDINO {configured('objectdetector')}" if router_enabled else _VLM_DTYPE
    return configured("rtdetr" if backend == "rtdetr" else "objectdetector")


def _detector_facts(console: "Console") -> DetectorFactsOut:
    try:
        from api.routers.diagnostics import _perception  # noqa: PLC0415

        stack = _perception(console)
    except Exception:  # noqa: BLE001
        return DetectorFactsOut()
    try:
        precision = detector_precision(_built_models(console), stack.backend, stack.router_enabled)
    except Exception:  # noqa: BLE001 (a fact nobody can read is not claimed)
        precision = None
    return DetectorFactsOut(backend=stack.backend, router_enabled=stack.router_enabled,
                            vlm_model_id=stack.vlm_model_id, precision=precision)


def facts(console: "Console") -> CellFactsOut:
    """What this cell is: its cameras and looks, its hand's natural axis, push, detector, brake, carried part, route.
    Read once after build and once after connect; moves nothing."""
    from src.robot.core.arm_capabilities import CarriesPayload, SupportsHalt, brakes_in_motion_of  # noqa: PLC0415
    from src.robot.execution.lifecycle import service_cameras  # noqa: PLC0415
    from src.robot.execution.looks import look_label  # noqa: PLC0415
    from src.robot.execution.motion import route_of  # noqa: PLC0415
    from src.robot.execution.pick_run import configured_looks_of  # noqa: PLC0415
    from src.robot.grasping.multiview.association import HAND_EYE_DRIFT_WARN_MM  # noqa: PLC0415

    session = console.session
    service, arm = session.service, session.arm
    if service is None:
        return CellFactsOut(route=RouteFactsOut(route="refused", sentence="nothing is built yet"),
                            hand_eye_warn_mm=HAND_EYE_DRIFT_WARN_MM, detector=_detector_facts(console))
    route = route_of(arm)
    cameras = [RigOut(rig_id=str(camera.rig_id), mounting=_mounting(camera), primary=index == 0)  # type: ignore[arg-type]
               for index, camera in enumerate(service_cameras(service))]
    looks = [look_label(look) for look in (configured_looks_of(service) or ())]
    axis = getattr(getattr(getattr(service, "runtime", None), "orchestrator", None), "natural_closing_axis", None)
    declined: str | None = None
    modelled = False
    if isinstance(arm, CarriesPayload):
        try:
            declined = arm.payload_declined_reason()
        except Exception as exc:  # noqa: BLE001
            declined = f"the arm could not say ({type(exc).__name__}: {exc})"
        modelled = declined is None
    length = getattr(getattr(getattr(getattr(getattr(arm, "config", None), "safety", None), "planning_world", None),
                             "payload", None), "length_mm", None)
    return CellFactsOut(
        wrist_camera=getattr(service, "perceives_from_the_wrist", False) is True,
        cameras=cameras,
        looks=looks,
        natural_closing_axis=None if axis is None else str(getattr(axis, "name", None) or axis),
        push=_push_facts(service),
        detector=_detector_facts(console),
        hand_eye_warn_mm=HAND_EYE_DRIFT_WARN_MM,
        brake=BrakeFactsOut(latches=isinstance(arm, SupportsHalt), brakes_in_motion=brakes_in_motion_of(arm)),
        payload=PayloadFactsOut(modelled=modelled, declined_reason=declined,
                                length_mm=float(length) if isinstance(length, (int, float)) else None),
        route=RouteFactsOut(route=route.route.value, sentence=route.reason),  # type: ignore[arg-type]
        rehearsal=rehearsal_of(service),
    )
