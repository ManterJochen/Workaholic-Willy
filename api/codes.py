"""Every typed code the console answers with, in one place: run kinds, stop codes and their classes, events,
refusals, readiness lights and blockers, the jaws question's stages and choices, and the command reader's notes.

A code is a promise to the frontend. The browser translates codes, not sentences (the backend's sentences stay
English and show as details in the tech view), so a code that meant two things would be translated as one of them
and read wrong in the other. Hence the rules this module keeps:

* **One code, one meaning.** ``not_acknowledged`` is Connect's 428, "the operator has not seen what will move", and
  nothing else; the stop card's gate is ``cell_not_cleared``. A code may appear in two catalogs (``halted`` is a stop
  code, a refusal and a light) only where it says the same thing in each.
* **Typed twice, on purpose.** Each catalog is a ``Literal`` (so the OpenAPI document carries the union and the
  frontend's ``schema.d.ts`` types every i18n map over it exhaustively) and a ``StrEnum`` of the same values, in the
  same order (so Python code names members instead of repeating strings). ``tests/test_api_contract.py`` asserts the
  two are equal, and that ``frontend/src/api/codes.ts`` lists the same values.
* **Served whole.** ``GET /v1/codes`` answers :func:`catalog`, so a client can read the vocabulary instead of copying
  it.

The stop codes carry the run's end: which class a code belongs to decides what the operator sees and whether the
console writes a recovery record (``STOP_CLASS``, ``api/runs.py``).
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from types import MappingProxyType
from typing import Final, Literal, TypeAlias

__all__ = [
    "BlockerCode",
    "BlockerCodeName",
    "CommandNote",
    "CommandNoteName",
    "EventType",
    "EventTypeName",
    "JawsChoice",
    "JawsChoiceName",
    "JawsStage",
    "JawsStageName",
    "LIGHT_CODES",
    "LightCode",
    "LightCodeName",
    "LightId",
    "LightIdName",
    "LightState",
    "LightStateName",
    "MOVING_KINDS",
    "REFUSAL_STATUS",
    "RefusalCode",
    "RefusalCodeName",
    "RunKind",
    "RunKindName",
    "STOP_CLASS",
    "StopClass",
    "StopClassName",
    "StopCode",
    "StopCodeName",
    "catalog",
    "not_built_yet",
    "stop_class_of",
]


# --- runs --------------------------------------------------------------------------------------------------------

RunKindName: TypeAlias = Literal["pick", "task", "home", "teach", "planner"]


class RunKind(StrEnum):
    """Which body a run drives. All kinds share the console's one-run lock."""

    #: ``POST /v1/pick``: picks only, nothing placed.
    PICK = "pick"
    #: ``POST /v1/task``: pick, then place, then return, then look again.
    TASK = "task"
    #: ``POST /v1/cell/home``: the manual Home button, a planned move to a pose.
    HOME = "home"
    #: ``POST /v1/teach``: a person guides the arm by hand to one pose, which is screened and saved.
    TEACH = "teach"
    #: ``POST /v1/cell/planner``: starting cuRobo. It moves nothing.
    PLANNER = "planner"


#: The kinds that move the arm by themselves. A problem stop of one of them writes the console's recovery record, and
#: each of them counts down first where a person's hands were last at the arm (a teach, "open now").
MOVING_KINDS: Final[frozenset[RunKind]] = frozenset({RunKind.PICK, RunKind.TASK, RunKind.HOME})


StopCodeName: TypeAlias = Literal[
    # done: the arm stands at its return pose (or never left it)
    "finished", "nothing_left", "part_limit", "taught", "planner_ready",
    # operator: somebody asked it to stop
    "stopped_after_part", "cancelled",
    # ask: a question for the operator in the chat, every option that moves a new run
    "target_not_found", "target_lost", "target_unreachable", "part_does_not_fit", "pose_refused", "teach_refused",
    # teach: the pose was not saved, the arm holds where the person left it
    "heartbeat_lost", "teach_time_limit", "teach_not_saved",
    # planner: cuRobo did not start
    "planner_failed",
    # problem: the arm stays where it stopped, and a person decides what happens next
    "halted", "controller_stopped", "hand_needs_person", "recovery_needs_person", "part_still_held",
    "return_failed", "failed_in_a_row", "detector_failed", "cell_fault", "disconnected", "software_error",
]


class StopCode(StrEnum):
    """Why a run ended, typed. The run's ``error`` carries the backend's sentence beside it."""

    FINISHED = "finished"
    NOTHING_LEFT = "nothing_left"
    PART_LIMIT = "part_limit"
    TAUGHT = "taught"
    PLANNER_READY = "planner_ready"
    STOPPED_AFTER_PART = "stopped_after_part"
    CANCELLED = "cancelled"
    TARGET_NOT_FOUND = "target_not_found"
    TARGET_LOST = "target_lost"
    TARGET_UNREACHABLE = "target_unreachable"
    PART_DOES_NOT_FIT = "part_does_not_fit"
    POSE_REFUSED = "pose_refused"
    TEACH_REFUSED = "teach_refused"
    HEARTBEAT_LOST = "heartbeat_lost"
    TEACH_TIME_LIMIT = "teach_time_limit"
    TEACH_NOT_SAVED = "teach_not_saved"
    PLANNER_FAILED = "planner_failed"
    HALTED = "halted"
    CONTROLLER_STOPPED = "controller_stopped"
    HAND_NEEDS_PERSON = "hand_needs_person"
    RECOVERY_NEEDS_PERSON = "recovery_needs_person"
    PART_STILL_HELD = "part_still_held"
    RETURN_FAILED = "return_failed"
    FAILED_IN_A_ROW = "failed_in_a_row"
    DETECTOR_FAILED = "detector_failed"
    CELL_FAULT = "cell_fault"
    DISCONNECTED = "disconnected"
    SOFTWARE_ERROR = "software_error"


StopClassName: TypeAlias = Literal["done", "operator", "ask", "teach", "planner", "problem"]


class StopClass(StrEnum):
    """What the frontend does with a stop code (build plan 1.2)."""

    #: A neutral line ("Fertig, 7 Teile"), then the next-instruction question. RunState ``finished``.
    DONE = "done"
    #: A neutral line, then the next-instruction question. RunState ``cancelled``.
    OPERATOR = "operator"
    #: An ask card with options; every option that moves starts a new run on a click. RunState ``finished``.
    ASK = "ask"
    #: A teach card: the pose was not saved, and why. RunState ``failed``.
    TEACH = "teach"
    #: The planner light and its "start the planner" action. RunState ``failed``.
    PLANNER = "planner"
    #: The stop card. The arm stays where it stopped; a moving kind writes the recovery record. RunState ``failed``.
    PROBLEM = "problem"


#: Each stop code's class. Total: every :class:`StopCode` has exactly one.
STOP_CLASS: Final[Mapping[StopCode, StopClass]] = MappingProxyType({
    StopCode.FINISHED: StopClass.DONE,
    StopCode.NOTHING_LEFT: StopClass.DONE,
    StopCode.PART_LIMIT: StopClass.DONE,
    StopCode.TAUGHT: StopClass.DONE,
    StopCode.PLANNER_READY: StopClass.DONE,
    StopCode.STOPPED_AFTER_PART: StopClass.OPERATOR,
    StopCode.CANCELLED: StopClass.OPERATOR,
    StopCode.TARGET_NOT_FOUND: StopClass.ASK,
    StopCode.TARGET_LOST: StopClass.ASK,
    StopCode.TARGET_UNREACHABLE: StopClass.ASK,
    StopCode.PART_DOES_NOT_FIT: StopClass.ASK,
    StopCode.POSE_REFUSED: StopClass.ASK,
    StopCode.TEACH_REFUSED: StopClass.ASK,
    StopCode.HEARTBEAT_LOST: StopClass.TEACH,
    StopCode.TEACH_TIME_LIMIT: StopClass.TEACH,
    StopCode.TEACH_NOT_SAVED: StopClass.TEACH,
    StopCode.PLANNER_FAILED: StopClass.PLANNER,
    StopCode.HALTED: StopClass.PROBLEM,
    StopCode.CONTROLLER_STOPPED: StopClass.PROBLEM,
    StopCode.HAND_NEEDS_PERSON: StopClass.PROBLEM,
    StopCode.RECOVERY_NEEDS_PERSON: StopClass.PROBLEM,
    StopCode.PART_STILL_HELD: StopClass.PROBLEM,
    StopCode.RETURN_FAILED: StopClass.PROBLEM,
    StopCode.FAILED_IN_A_ROW: StopClass.PROBLEM,
    StopCode.DETECTOR_FAILED: StopClass.PROBLEM,
    StopCode.CELL_FAULT: StopClass.PROBLEM,
    StopCode.DISCONNECTED: StopClass.PROBLEM,
    StopCode.SOFTWARE_ERROR: StopClass.PROBLEM,
})


def stop_class_of(code: str) -> StopClass | None:
    """The class of ``code``, or ``None`` for a string that is no stop code (the empty one of a running run)."""
    try:
        return STOP_CLASS[StopCode(code)]
    except ValueError:
        return None


# --- events ------------------------------------------------------------------------------------------------------

EventTypeName: TypeAlias = Literal[
    # every run
    "run_started", "run_countdown", "run_stop_requested", "run_halt_requested", "run_error", "run_finished",
    # a pick, inside a pick run or a task: the library's PickStage, prefixed, and the API's per-pick summary
    "pick.pick_started", "pick.attempt_started", "pick.perceived", "pick.ranked", "pick.no_candidate",
    "pick.executing", "pick.attempt_finished", "pick.pick_finished", "pick.cancelled", "pick_result",
    # a task
    "task.pose_screened", "task.survey_started", "task.target_found", "task.target_missing", "task.part_started",
    "task.nothing_found", "task.carry_started", "task.target_checked", "task.target_lost", "task.drop_planned",
    "task.place_started", "task.placed", "task.place_failed", "task.put_back", "task.return_started",
    "task.returned", "task.return_failed", "task.part_finished",
    # the Home button
    "home.started", "home.arrived", "home.refused",
    # teaching one pose by hand
    "teach.free", "teach.say", "teach.outside", "teach.inside", "teach.time_warning", "teach.holding_when_still",
    "teach.holding", "teach.screening", "teach.saved", "teach.refused", "teach.not_saved",
    # starting the planner
    "planner.starting", "planner.ready", "planner.failed",
    # the cell's own stream (run_id "cell"): what belongs to no run
    "cell.jaws_question", "cell.jaws_answered", "cell.jaws_ended", "cell.halted", "cell.recovery",
    "cell.recovery_ended", "cell.acknowledged", "cell.planner",
]


class EventType(StrEnum):
    """Every event type the console publishes. The envelope is ``api.events.EventEnvelope``, unchanged."""

    RUN_STARTED = "run_started"
    RUN_COUNTDOWN = "run_countdown"
    RUN_STOP_REQUESTED = "run_stop_requested"
    RUN_HALT_REQUESTED = "run_halt_requested"
    RUN_ERROR = "run_error"
    RUN_FINISHED = "run_finished"
    PICK_STARTED = "pick.pick_started"
    PICK_ATTEMPT_STARTED = "pick.attempt_started"
    PICK_PERCEIVED = "pick.perceived"
    PICK_RANKED = "pick.ranked"
    PICK_NO_CANDIDATE = "pick.no_candidate"
    PICK_EXECUTING = "pick.executing"
    PICK_ATTEMPT_FINISHED = "pick.attempt_finished"
    PICK_FINISHED = "pick.pick_finished"
    PICK_CANCELLED = "pick.cancelled"
    PICK_RESULT = "pick_result"
    TASK_POSE_SCREENED = "task.pose_screened"
    TASK_SURVEY_STARTED = "task.survey_started"
    TASK_TARGET_FOUND = "task.target_found"
    TASK_TARGET_MISSING = "task.target_missing"
    TASK_PART_STARTED = "task.part_started"
    TASK_NOTHING_FOUND = "task.nothing_found"
    TASK_CARRY_STARTED = "task.carry_started"
    TASK_TARGET_CHECKED = "task.target_checked"
    TASK_TARGET_LOST = "task.target_lost"
    TASK_DROP_PLANNED = "task.drop_planned"
    TASK_PLACE_STARTED = "task.place_started"
    TASK_PLACED = "task.placed"
    TASK_PLACE_FAILED = "task.place_failed"
    TASK_PUT_BACK = "task.put_back"
    TASK_RETURN_STARTED = "task.return_started"
    TASK_RETURNED = "task.returned"
    TASK_RETURN_FAILED = "task.return_failed"
    TASK_PART_FINISHED = "task.part_finished"
    HOME_STARTED = "home.started"
    HOME_ARRIVED = "home.arrived"
    HOME_REFUSED = "home.refused"
    TEACH_FREE = "teach.free"
    TEACH_SAY = "teach.say"
    TEACH_OUTSIDE = "teach.outside"
    TEACH_INSIDE = "teach.inside"
    TEACH_TIME_WARNING = "teach.time_warning"
    TEACH_HOLDING_WHEN_STILL = "teach.holding_when_still"
    TEACH_HOLDING = "teach.holding"
    TEACH_SCREENING = "teach.screening"
    TEACH_SAVED = "teach.saved"
    TEACH_REFUSED = "teach.refused"
    TEACH_NOT_SAVED = "teach.not_saved"
    PLANNER_STARTING = "planner.starting"
    PLANNER_READY = "planner.ready"
    PLANNER_FAILED = "planner.failed"
    CELL_JAWS_QUESTION = "cell.jaws_question"
    CELL_JAWS_ANSWERED = "cell.jaws_answered"
    CELL_JAWS_ENDED = "cell.jaws_ended"
    CELL_HALTED = "cell.halted"
    CELL_RECOVERY = "cell.recovery"
    CELL_RECOVERY_ENDED = "cell.recovery_ended"
    CELL_ACKNOWLEDGED = "cell.acknowledged"
    CELL_PLANNER = "cell.planner"


# --- refusals ----------------------------------------------------------------------------------------------------

RefusalCodeName: TypeAlias = Literal[
    # the request itself
    "bad_request", "no_robot_configured", "no_such_run", "not_built_yet",
    # bringing the cell up (Connect's own: ConnectRefused)
    "not_built", "not_acknowledged", "stale_token", "cell_busy", "no_real_gripper", "driver_refused",
    "wrong_state", "build_refused", "jaws_seam_missing",
    # the cell's state, before anything moves
    "not_connected", "run_active", "halted", "controller_stopped", "cell_not_cleared", "restart_required",
    "needs_person", "jaws_question_pending", "part_still_held", "jaws_not_confirmed", "route_refused",
    "carried_part_not_modelled", "camera_target_unavailable",
    # what a task or a pick was asked to do
    "object_required", "prompt_not_routable", "target_not_routable", "push_distance_refused", "unknown_pose",
    "no_place_declared", "closing_axis_refused", "not_a_task", "not_a_pick", "not_restartable",
    # the jaws question
    "jaws_not_open", "no_question", "question_changed", "choice_not_offered",
    # the planner, overlays
    "planner_not_used", "no_overlay",
    # poses and teaching
    "no_layer", "no_hand_guiding", "part_in_hand", "planner_not_ready", "screen_unavailable", "payload_changed",
    "name_taken", "invalid_name", "invalid_label", "wrong_token", "not_free",
    # commands (the VLM reader)
    "vlm_not_loaded", "vlm_unavailable", "vlm_model_missing",
    # config writes (WriteRefused)
    "unknown_key", "not_writable", "invalid_value", "no_target", "cell_connected", "empty_patch",
    # speech (the owner's voice routes)
    "empty_audio", "audio_format_unsupported", "audio_undecodable", "audio_too_long", "speech_model_missing",
    "speech_unavailable", "transcription_failed", "listen_busy", "talk_not_pressed", "microphone_ended",
    "nothing_recorded", "microphone_unavailable", "listen_failed",
]


class RefusalCode(StrEnum):
    """Every ``code`` the one error envelope ``{code, message, detail}`` can carry, but FastAPI's own ``http_<status>``.

    A refusal names what the operator does next; the ``message`` is the backend's sentence and ``detail`` the machine
    half (which key, which run holds the lock).
    """

    BAD_REQUEST = "bad_request"
    NO_ROBOT_CONFIGURED = "no_robot_configured"
    NO_SUCH_RUN = "no_such_run"
    #: A route of the console's contract whose body is not built yet (the build's contract stage). Never a refusal of
    #: the cell: it says the server is older than the page.
    NOT_BUILT_YET = "not_built_yet"
    NOT_BUILT = "not_built"
    #: Connect's 428 and nothing else: the operator has not seen what connecting will move (no preview token).
    NOT_ACKNOWLEDGED = "not_acknowledged"
    STALE_TOKEN = "stale_token"
    CELL_BUSY = "cell_busy"
    NO_REAL_GRIPPER = "no_real_gripper"
    DRIVER_REFUSED = "driver_refused"
    WRONG_STATE = "wrong_state"
    BUILD_REFUSED = "build_refused"
    JAWS_SEAM_MISSING = "jaws_seam_missing"
    NOT_CONNECTED = "not_connected"
    RUN_ACTIVE = "run_active"
    #: The arm's halt latch is set ("halt now" was pressed). Checked before the controller, so a halt never reads as a
    #: controller stop.
    HALTED = "halted"
    CONTROLLER_STOPPED = "controller_stopped"
    #: The recovery record is newer than the last "the cell is clear": the stop card's gate.
    CELL_NOT_CLEARED = "cell_not_cleared"
    #: The recovery record stands: the way back is Restart or Home, whose first motion is the planned move home.
    RESTART_REQUIRED = "restart_required"
    NEEDS_PERSON = "needs_person"
    JAWS_QUESTION_PENDING = "jaws_question_pending"
    PART_STILL_HELD = "part_still_held"
    JAWS_NOT_CONFIRMED = "jaws_not_confirmed"
    ROUTE_REFUSED = "route_refused"
    CARRIED_PART_NOT_MODELLED = "carried_part_not_modelled"
    CAMERA_TARGET_UNAVAILABLE = "camera_target_unavailable"
    OBJECT_REQUIRED = "object_required"
    PROMPT_NOT_ROUTABLE = "prompt_not_routable"
    TARGET_NOT_ROUTABLE = "target_not_routable"
    PUSH_DISTANCE_REFUSED = "push_distance_refused"
    UNKNOWN_POSE = "unknown_pose"
    NO_PLACE_DECLARED = "no_place_declared"
    CLOSING_AXIS_REFUSED = "closing_axis_refused"
    NOT_A_TASK = "not_a_task"
    NOT_A_PICK = "not_a_pick"
    NOT_RESTARTABLE = "not_restartable"
    JAWS_NOT_OPEN = "jaws_not_open"
    NO_QUESTION = "no_question"
    QUESTION_CHANGED = "question_changed"
    CHOICE_NOT_OFFERED = "choice_not_offered"
    PLANNER_NOT_USED = "planner_not_used"
    NO_OVERLAY = "no_overlay"
    NO_LAYER = "no_layer"
    NO_HAND_GUIDING = "no_hand_guiding"
    PART_IN_HAND = "part_in_hand"
    PLANNER_NOT_READY = "planner_not_ready"
    SCREEN_UNAVAILABLE = "screen_unavailable"
    PAYLOAD_CHANGED = "payload_changed"
    NAME_TAKEN = "name_taken"
    INVALID_NAME = "invalid_name"
    INVALID_LABEL = "invalid_label"
    WRONG_TOKEN = "wrong_token"
    NOT_FREE = "not_free"
    VLM_NOT_LOADED = "vlm_not_loaded"
    VLM_UNAVAILABLE = "vlm_unavailable"
    VLM_MODEL_MISSING = "vlm_model_missing"
    UNKNOWN_KEY = "unknown_key"
    NOT_WRITABLE = "not_writable"
    INVALID_VALUE = "invalid_value"
    NO_TARGET = "no_target"
    CELL_CONNECTED = "cell_connected"
    EMPTY_PATCH = "empty_patch"
    EMPTY_AUDIO = "empty_audio"
    AUDIO_FORMAT_UNSUPPORTED = "audio_format_unsupported"
    AUDIO_UNDECODABLE = "audio_undecodable"
    AUDIO_TOO_LONG = "audio_too_long"
    SPEECH_MODEL_MISSING = "speech_model_missing"
    SPEECH_UNAVAILABLE = "speech_unavailable"
    TRANSCRIPTION_FAILED = "transcription_failed"
    LISTEN_BUSY = "listen_busy"
    TALK_NOT_PRESSED = "talk_not_pressed"
    MICROPHONE_ENDED = "microphone_ended"
    NOTHING_RECORDED = "nothing_recorded"
    MICROPHONE_UNAVAILABLE = "microphone_unavailable"
    LISTEN_FAILED = "listen_failed"


#: The HTTP status each refusal is answered with. One code, one status: a code that came back as 409 here and 422
#: there would mean two things. 409 "fix the cell", 422 "fix the request", 428 "read the preview first", 403 "fix
#: the config", 404 "no such thing", 501 "this server cannot", 502 "the controller refused".
REFUSAL_STATUS: Final[Mapping[RefusalCode, int]] = MappingProxyType({
    RefusalCode.BAD_REQUEST: 422,
    RefusalCode.NO_ROBOT_CONFIGURED: 409,
    RefusalCode.NO_SUCH_RUN: 404,
    RefusalCode.NOT_BUILT_YET: 501,
    RefusalCode.NOT_BUILT: 409,
    RefusalCode.NOT_ACKNOWLEDGED: 428,
    RefusalCode.STALE_TOKEN: 428,
    RefusalCode.CELL_BUSY: 409,
    RefusalCode.NO_REAL_GRIPPER: 403,
    RefusalCode.DRIVER_REFUSED: 502,
    RefusalCode.WRONG_STATE: 409,
    RefusalCode.BUILD_REFUSED: 422,
    RefusalCode.JAWS_SEAM_MISSING: 409,
    RefusalCode.NOT_CONNECTED: 409,
    RefusalCode.RUN_ACTIVE: 409,
    RefusalCode.HALTED: 409,
    RefusalCode.CONTROLLER_STOPPED: 409,
    RefusalCode.CELL_NOT_CLEARED: 409,
    RefusalCode.RESTART_REQUIRED: 409,
    RefusalCode.NEEDS_PERSON: 409,
    RefusalCode.JAWS_QUESTION_PENDING: 409,
    RefusalCode.PART_STILL_HELD: 409,
    RefusalCode.JAWS_NOT_CONFIRMED: 409,
    RefusalCode.ROUTE_REFUSED: 409,
    RefusalCode.CARRIED_PART_NOT_MODELLED: 409,
    RefusalCode.CAMERA_TARGET_UNAVAILABLE: 409,
    RefusalCode.OBJECT_REQUIRED: 422,
    RefusalCode.PROMPT_NOT_ROUTABLE: 422,
    RefusalCode.TARGET_NOT_ROUTABLE: 422,
    RefusalCode.PUSH_DISTANCE_REFUSED: 422,
    RefusalCode.UNKNOWN_POSE: 422,
    RefusalCode.NO_PLACE_DECLARED: 422,
    RefusalCode.CLOSING_AXIS_REFUSED: 422,
    RefusalCode.NOT_A_TASK: 409,
    RefusalCode.NOT_A_PICK: 409,
    RefusalCode.NOT_RESTARTABLE: 409,
    RefusalCode.JAWS_NOT_OPEN: 409,
    RefusalCode.NO_QUESTION: 404,
    RefusalCode.QUESTION_CHANGED: 409,
    RefusalCode.CHOICE_NOT_OFFERED: 422,
    RefusalCode.PLANNER_NOT_USED: 409,
    RefusalCode.NO_OVERLAY: 404,
    RefusalCode.NO_LAYER: 422,
    RefusalCode.NO_HAND_GUIDING: 409,
    RefusalCode.PART_IN_HAND: 409,
    RefusalCode.PLANNER_NOT_READY: 409,
    RefusalCode.SCREEN_UNAVAILABLE: 409,
    RefusalCode.PAYLOAD_CHANGED: 409,
    RefusalCode.NAME_TAKEN: 409,
    RefusalCode.INVALID_NAME: 422,
    RefusalCode.INVALID_LABEL: 422,
    RefusalCode.WRONG_TOKEN: 403,
    RefusalCode.NOT_FREE: 409,
    RefusalCode.VLM_NOT_LOADED: 409,
    RefusalCode.VLM_UNAVAILABLE: 501,
    RefusalCode.VLM_MODEL_MISSING: 501,
    RefusalCode.UNKNOWN_KEY: 404,
    RefusalCode.NOT_WRITABLE: 403,
    RefusalCode.INVALID_VALUE: 422,
    RefusalCode.NO_TARGET: 409,
    RefusalCode.CELL_CONNECTED: 409,
    RefusalCode.EMPTY_PATCH: 400,
    RefusalCode.EMPTY_AUDIO: 400,
    RefusalCode.AUDIO_FORMAT_UNSUPPORTED: 415,
    RefusalCode.AUDIO_UNDECODABLE: 422,
    RefusalCode.AUDIO_TOO_LONG: 422,
    RefusalCode.SPEECH_MODEL_MISSING: 501,
    RefusalCode.SPEECH_UNAVAILABLE: 501,
    RefusalCode.TRANSCRIPTION_FAILED: 422,
    RefusalCode.LISTEN_BUSY: 409,
    RefusalCode.TALK_NOT_PRESSED: 409,
    RefusalCode.MICROPHONE_ENDED: 409,
    RefusalCode.NOTHING_RECORDED: 422,
    RefusalCode.MICROPHONE_UNAVAILABLE: 501,
    RefusalCode.LISTEN_FAILED: 422,
})


def not_built_yet(route: str) -> Exception:
    """The answer of a contract route whose body is not built yet: 501 ``not_built_yet``, naming the route.

    Raised, not returned: ``raise not_built_yet("POST /v1/task")``. Imported lazily so this catalog does not need the
    web framework to be read.
    """
    from fastapi import HTTPException  # noqa: PLC0415

    return HTTPException(
        status_code=REFUSAL_STATUS[RefusalCode.NOT_BUILT_YET],
        detail={
            "code": str(RefusalCode.NOT_BUILT_YET),
            "message": (f"{route} is part of the console's contract and is not built yet on this server; nothing was "
                        "done."),
            "detail": {"route": route},
        },
    )


# --- readiness (GET /v1/cell/readiness) ----------------------------------------------------------------------------

LightIdName: TypeAlias = Literal["robot", "cameras", "planner", "gripper", "carried_part", "commands"]


class LightId(StrEnum):
    """One light of the ready bar."""

    ROBOT = "robot"
    CAMERAS = "cameras"
    PLANNER = "planner"
    GRIPPER = "gripper"
    CARRIED_PART = "carried_part"
    #: Informational only: it never blocks ``ready`` (owner decision 22).
    COMMANDS = "commands"


LightStateName: TypeAlias = Literal["ok", "wait", "blocked", "info"]


class LightState(StrEnum):
    """How a light reads. Whether it takes part in ``ready`` is its own ``blocks`` field."""

    #: Green: this part of the cell is ready (a rehearsal cell's camera and planner included, labelled "Probe").
    OK = "ok"
    #: Amber: it becomes ready by itself (the planner starting, the VLM loading).
    WAIT = "wait"
    #: Red: something or somebody must act first; its action is named by its code.
    BLOCKED = "blocked"
    #: Grey: a fact that gates nothing.
    INFO = "info"


LightCodeName: TypeAlias = Literal[
    "connected", "not_built", "not_connected", "halted", "controller_stopped", "controller_unreadable",
    "live", "rehearsal", "no_frame", "none",
    "ready", "starting", "off", "failed", "unplanned", "route_refused",
    "open_confirmed", "jaws_unknown", "jaws_closed", "question_pending",
    "modelled", "not_modelled", "not_applicable",
    "idle", "loading", "missing", "not_configured",
]


class LightCode(StrEnum):
    """Why a light reads as it does. :data:`LIGHT_CODES` says which light may carry which code."""

    CONNECTED = "connected"
    NOT_BUILT = "not_built"
    NOT_CONNECTED = "not_connected"
    HALTED = "halted"
    CONTROLLER_STOPPED = "controller_stopped"
    CONTROLLER_UNREADABLE = "controller_unreadable"
    LIVE = "live"
    REHEARSAL = "rehearsal"
    NO_FRAME = "no_frame"
    NONE = "none"
    READY = "ready"
    STARTING = "starting"
    OFF = "off"
    FAILED = "failed"
    UNPLANNED = "unplanned"
    ROUTE_REFUSED = "route_refused"
    OPEN_CONFIRMED = "open_confirmed"
    JAWS_UNKNOWN = "jaws_unknown"
    JAWS_CLOSED = "jaws_closed"
    QUESTION_PENDING = "question_pending"
    MODELLED = "modelled"
    NOT_MODELLED = "not_modelled"
    NOT_APPLICABLE = "not_applicable"
    IDLE = "idle"
    LOADING = "loading"
    MISSING = "missing"
    NOT_CONFIGURED = "not_configured"


#: The codes each light may carry (build plan 1.5). A code shared by two lights says the same thing of each.
LIGHT_CODES: Final[Mapping[LightId, tuple[LightCode, ...]]] = MappingProxyType({
    LightId.ROBOT: (LightCode.CONNECTED, LightCode.NOT_BUILT, LightCode.NOT_CONNECTED, LightCode.HALTED,
                    LightCode.CONTROLLER_STOPPED, LightCode.CONTROLLER_UNREADABLE),
    LightId.CAMERAS: (LightCode.LIVE, LightCode.REHEARSAL, LightCode.NO_FRAME, LightCode.NONE),
    LightId.PLANNER: (LightCode.READY, LightCode.STARTING, LightCode.OFF, LightCode.FAILED, LightCode.UNPLANNED,
                      LightCode.ROUTE_REFUSED),
    LightId.GRIPPER: (LightCode.OPEN_CONFIRMED, LightCode.JAWS_UNKNOWN, LightCode.JAWS_CLOSED,
                      LightCode.QUESTION_PENDING, LightCode.CONNECTED, LightCode.NONE),
    LightId.CARRIED_PART: (LightCode.MODELLED, LightCode.NOT_MODELLED, LightCode.NOT_APPLICABLE),
    LightId.COMMANDS: (LightCode.READY, LightCode.IDLE, LightCode.LOADING, LightCode.MISSING, LightCode.FAILED,
                       LightCode.NOT_CONFIGURED),
})


BlockerCodeName: TypeAlias = Literal[
    "run_active", "cell_not_cleared", "restart_required", "needs_person", "part_still_held", "jaws_question_pending",
]


class BlockerCode(StrEnum):
    """Why Start is disabled although every light may be green. Each is also the refusal of the same name."""

    RUN_ACTIVE = "run_active"
    CELL_NOT_CLEARED = "cell_not_cleared"
    RESTART_REQUIRED = "restart_required"
    NEEDS_PERSON = "needs_person"
    PART_STILL_HELD = "part_still_held"
    JAWS_QUESTION_PENDING = "jaws_question_pending"


# --- the jaws question (a toggle hand, answered in the browser) ---------------------------------------------------

JawsStageName: TypeAlias = Literal["where", "open_now"]


class JawsStage(StrEnum):
    """Which question a toggle hand asks."""

    #: "Where do the jaws stand?": open or closed.
    WHERE = "where"
    #: "They stand closed: open them now (one change of the output) or abort?"
    OPEN_NOW = "open_now"


JawsChoiceName: TypeAlias = Literal["open", "closed", "open_now", "abort"]


class JawsChoice(StrEnum):
    """An answer to the jaws question. There is no default: a question nobody answers is refused, never "open"."""

    OPEN = "open"
    CLOSED = "closed"
    #: One change of the hand's output, which opens the jaws where the arm stands. Asked only after "closed".
    OPEN_NOW = "open_now"
    ABORT = "abort"


# --- commands (POST /v1/commands/parse) ----------------------------------------------------------------------------

CommandNoteName: TypeAlias = Literal[
    "object_not_in_sentence", "place_not_in_sentence", "pose_unknown", "count_not_supported", "retried",
]


class CommandNote(StrEnum):
    """A soft note the command reader adds to what it understood; the card shows it beside the field it concerns."""

    OBJECT_NOT_IN_SENTENCE = "object_not_in_sentence"
    PLACE_NOT_IN_SENTENCE = "place_not_in_sentence"
    POSE_UNKNOWN = "pose_unknown"
    COUNT_NOT_SUPPORTED = "count_not_supported"
    RETRIED = "retried"


# --- the whole catalog ---------------------------------------------------------------------------------------------


def catalog() -> dict[str, object]:
    """Every catalog as plain lists, in declaration order: what ``GET /v1/codes`` answers (``CodesOut``)."""
    return {
        "run_kinds": [str(kind) for kind in RunKind],
        "moving_kinds": [str(kind) for kind in RunKind if kind in MOVING_KINDS],
        "stop_codes": [str(code) for code in StopCode],
        "stop_classes": [str(cls) for cls in StopClass],
        "stop_class_of": {str(code): str(cls) for code, cls in STOP_CLASS.items()},
        "event_types": [str(event) for event in EventType],
        "refusal_codes": [str(code) for code in RefusalCode],
        "light_ids": [str(light) for light in LightId],
        "light_states": [str(state) for state in LightState],
        "light_codes": [str(code) for code in LightCode],
        "light_codes_of": {str(light): [str(code) for code in codes] for light, codes in LIGHT_CODES.items()},
        "blocker_codes": [str(code) for code in BlockerCode],
        "jaws_stages": [str(stage) for stage in JawsStage],
        "jaws_choices": [str(choice) for choice in JawsChoice],
        "command_notes": [str(note) for note in CommandNote],
    }
