"""Commands: the VLM reads the operator's sentence (German or English) into an editable card.

Reading never starts anything: it creates no run, sets no prompt and touches no cell. The card shows what was
understood, a person corrects it, and Start on the card starts the task. A sentence read as "stop" only shows where
the stop buttons are. A greeting says how the console answers it (``CommandOut.greeting``, the app config's
``runtime.greeting.wave``): the console starts the wave, ``POST /v1/cell/wave``, never this route. Refused during a
run, whose chat input is disabled anyway.

One VLM copy per process reads commands and, on a cell whose detector is the VLM, detects too (``shared_vlm``). The
load rule is the library's (owner decision Q8 A): a cell that detects with the VLM loads it at its first command; any
other cell refuses (``vlm_not_loaded``) until a person loads it with ``POST /v1/commands/warmup``, "Laden" in the
ready bar. The models section read is the one the cell was built with, where the console keeps it: a models YAML
edited since names weights the cell does not hold.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException

from api.cell import Console, console
from api.codes import REFUSAL_STATUS, RefusalCode
from api.constants import API_LOG_DIR
from api.routers import diagnostics
from api.schemas import CommandIn, CommandOut, CommandStatusOut
from src.models.vlm import CommandRefused, read_command, reader_availability, shared_vlm
from src.utility.log_cfg import create_logger

router = APIRouter(prefix="/commands", tags=["commands"])

#: A parse is an operator's act, logged once each: what was read, how long it took, whether it loaded the model.
logger = create_logger("CommandsRouter", "router_commands.log", log_dir=API_LOG_DIR)


def _refuse(code: RefusalCode, message: str, **detail: Any) -> HTTPException:
    return HTTPException(status_code=REFUSAL_STATUS[code],
                         detail={"code": str(code), "message": message, "detail": detail})


def _reader_code(code: str) -> RefusalCode:
    """The reader's refusal as the catalog's code. One the catalog does not know (a library that grew a word the
    console has not) is the reader being unavailable (501), never a 500; ``tests/test_api_commands.py`` pins the two
    lists against each other."""
    try:
        return RefusalCode(code)
    except ValueError:
        logger.error("The command reader refused with %r, which the console's catalog does not know; answered as "
                     "vlm_unavailable.", code)
        return RefusalCode.VLM_UNAVAILABLE


def _no_run(cell: Console, what: str) -> None:
    """``run_active`` while a run holds the cell: a command is read between runs, never during one (a teach included,
    whose microphone the browser switches off)."""
    active = cell.active_run_id
    if active is not None:
        raise _refuse(RefusalCode.RUN_ACTIVE, (
            f"run {active} is active, so {what} is refused: a command is read between runs, and the stop buttons are "
            "how a run is stopped"), run_id=active)


def _models(cell: Console) -> Any:
    """The models section the cell was built with (``Console.built_models``), else the tree's: before any build the
    tree's is the one the next build reads."""
    built = getattr(cell, "built_models", None)
    return built if built is not None else cell.config().models


def _weights_present(models: Any) -> bool | None:
    """Whether the VLM checkpoint the models section names is on this box, checked without loading; ``None`` where
    none is named."""
    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        return None
    vlm = pipeline.zero_shot.vlm
    return diagnostics._vlm_weights_present(str(vlm.model_id), getattr(vlm, "model_path", None))


def _poses(cell: Console) -> dict[str, str]:
    """The taught poses as the reader is handed them: each pose's label (its name where it has none) -> its name."""
    robot = cell.config().robot
    named = getattr(robot, "named_poses", None) or {}
    return {(pose.label or name): name for name, pose in named.items()}


def _greeting(cell: Console) -> str:
    """How the console answers a greeting: ``runtime.greeting.wave`` of the app config the cell reads now."""
    return str(cell.config().runtime.greeting.wave)


def _status(availability: Any) -> CommandStatusOut:
    return CommandStatusOut(
        state=availability.state, model_id=availability.model_id, weights_present=availability.weights_present,
        cause=availability.cause, last_latency_ms=availability.last_answer_ms,
        shared_with_detection=availability.shared_with_detection,
    )


@router.post("/parse", response_model=CommandOut, summary="Read a sentence into a task card (moves nothing)")
def post_parse(cell: Annotated[Console, Depends(console)], body: CommandIn) -> CommandOut:
    """The whole sentence goes to the VLM, which answers the object and the place as English phrases with the
    operator's own words, a taught pose by its NAME (a spoken label comes back as the name), the scope and where to go
    after. Each phrase carries the route the detector would take.

    A ``def`` route: a VLM cell's first command loads the model, several seconds, on the thread pool. Refused:
    ``run_active`` (409) before anything is asked, ``vlm_not_loaded`` (409), ``vlm_unavailable`` (501),
    ``vlm_model_missing`` (501), and a sentence that is no sentence (422 ``bad_request``).
    """
    _no_run(cell, "reading a command")
    models = _models(cell)
    try:
        reading = read_command(body.text, models=models, poses=_poses(cell), weights_present=_weights_present(models),
                               holder=shared_vlm())
    except CommandRefused as refused:
        logger.warning("Command not read (%s): %s", refused.code, refused.cause)
        raise _refuse(_reader_code(refused.code), refused.cause) from refused
    except ValueError as exc:
        raise _refuse(RefusalCode.BAD_REQUEST, f"there is no command to read: {exc}") from exc
    said = reading.to_dict()
    said["greeting"] = _greeting(cell) if reading.greeting else None
    for field in ("object", "place"):
        phrase = said.get(field)
        if isinstance(phrase, dict) and phrase.get("phrase"):
            phrase["route"] = diagnostics.preview_route(str(phrase["phrase"]), cell).model_dump()
    logger.info("Command read (%s source, %s): intent %s, %d question(s), %.0f ms%s%s.", body.source,
                body.language or "language unsaid", reading.intent, reading.attempts, reading.latency_ms,
                ", the model loaded for it" if reading.loaded_now else "",
                f", a greeting (wave {said['greeting']})" if reading.greeting else "")
    return CommandOut.model_validate(said)


@router.get("/status", response_model=CommandStatusOut, summary="Is the command reader loaded?")
def get_status(cell: Annotated[Console, Depends(console)]) -> CommandStatusOut:
    """Where the reader stands (``ready``, ``idle``, ``loading``, ``missing``, ``failed``, ``not_configured``), and why.
    It loads nothing."""
    models = _models(cell)
    return _status(reader_availability(models, weights_present=_weights_present(models), holder=shared_vlm()))


@router.post("/warmup", response_model=CommandStatusOut, summary="Load the command reader (refused during a run)")
def post_warmup(cell: Annotated[Console, Depends(console)]) -> CommandStatusOut:
    """"Laden": load the VLM now, about six seconds on the cell PC, and answer where it stands then. A load that fails
    answers ``failed`` with its cause. Refused during a run, and where loading is refused (missing weights, another
    checkpoint on the card: rebuild the cell)."""
    _no_run(cell, "loading the command reader")
    models = _models(cell)
    holder = shared_vlm()
    availability = reader_availability(models, weights_present=_weights_present(models), holder=holder)
    if availability.warmup_refusal:
        raise _refuse(_reader_code(availability.warmup_refusal), availability.cause)
    status = holder.load_for(availability.vlm)
    logger.info("Laden: the VLM is %s%s.", status.state, f" ({status.cause})" if status.cause else "")
    return _status(reader_availability(models, weights_present=_weights_present(models), holder=holder))
