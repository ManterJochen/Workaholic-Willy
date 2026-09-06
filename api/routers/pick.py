"""Starting a pick, stopping one, and watching it happen.

Starting returns an id immediately. ``pick()`` blocks for as long as the arm takes, so a handler that
waited for it would hold an HTTP connection open across a physical motion, and the browser, a proxy or
a sleeping laptop would give up somewhere in the middle, leaving an operator with no idea whether the
arm was still moving.

The WebSocket is where the answer lives, and it is built around one question a reconnecting client must
be able to ask: "I last saw event 41; what did I miss?"
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect

from api.cell import Console, console
from api.constants import API_LOG_DIR, ROUTER_PICK_LOG_FILE
from api.lifecycle import CellState
from api.runs import RunConflict, RunRegistry
from api.schemas import PickIn, RunOut
from src.utility.log_cfg import create_logger

router = APIRouter(tags=["pick"])

#: The run itself is logged by ``api.runs`` (start, each pick, the end). This file logs what only it
#: sees: a prompt refused before anything moved, and the event socket's replay.
logger = create_logger("PickRouter", ROUTER_PICK_LOG_FILE, log_dir=API_LOG_DIR)

#: How long a quiet socket waits before sending a keep-alive. A socket that is merely quiet looks
#: exactly like a dead one from the far end, and a UI that cannot tell will show a stale run as live.
_KEEPALIVE_S = 15.0


def _registry(cell: Console) -> RunRegistry:
    return cell.registry


def _refuse_unroutable_prompt(cell: Console, prompt: str) -> None:
    """Refuse a prompt this cell cannot ground correctly, before anything moves.

    The failure this guards is the one the whole perception arc exists to remove, and it is silent: a
    complex prompt handed to the phrase grounder does not come back empty, it comes back with a
    confident box on the wrong object. The cell then grasps something the operator did not ask for, and
    every log line looks like a successful pick.

    So the check happens at the API boundary, where refusing is free, rather than after the arm has
    moved. It costs nothing, since the router is pure text analysis with no model and no image, and it
    stays quiet on every prompt the configured stack can actually handle.

    An operator who wants the pick anyway has two honest routes, both named in the refusal: configure
    the VLM, or set ``on_unavailable: degrade`` and accept a loudly-logged wrong answer. What they do
    not get is a silent one.
    """
    if not prompt.strip():
        return  # "whatever the source already targets": there is no prompt to route
    from api.routers.diagnostics import _perception

    from src.models.routing import Route, route

    decision = route(prompt)
    if decision.route is not Route.VLM:
        return
    stack = _perception(cell)
    if stack.backend == "vlm" and stack.vlm_weights_present is not False:
        return
    if stack.vlm_on_unavailable == "degrade":
        return  # a deliberate, configured choice; the library logs every degraded grounding

    reason = ("this cell configures no VLM route"
              if stack.backend != "vlm"
              else f"the VLM route is configured but {stack.vlm_model_id} is not on this box")
    # The refusal that prevents a wrong pick, so it is the one most worth having a record of.
    logger.warning(
        "Pick refused before anything moved: %r needs the VLM route (%s) and %s.",
        prompt, decision.reason, reason,
    )
    raise HTTPException(
        status_code=422,
        detail={
            "code": "prompt_not_routable",
            "message": (
                f"{prompt!r} needs the VLM route ({decision.describe()}), and {reason}. The phrase "
                f"grounder does not fail loudly on prompts like this; it returns a confident box for "
                f"the WRONG object, so this pick is refused rather than run. Fix it by configuring "
                f"models.pipeline.zero_shot.backend='vlm' (and fetching the weights with "
                f"scripts/model_weights/fetch.py), or set vlm.on_unavailable='degrade' to accept a "
                f"loudly-logged fallback."
            ),
            "detail": {
                "route": str(decision.route),
                "reason": str(decision.reason),
                "backend": stack.backend,
                "vlm_weights_present": stack.vlm_weights_present,
            },
        },
    )


@router.post("/pick", response_model=RunOut, status_code=202, summary="Start a run (THIS MOVES)")
def post_pick(cell: Annotated[Console, Depends(console)], body: PickIn) -> RunOut:
    """202, not 200: the work is accepted and happening elsewhere, not finished when this returns."""
    if cell.session.state is not CellState.CONNECTED:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "not_connected",
                "message": (
                    f"the cell is {cell.session.state}; connect it before asking it to pick."
                ),
                "detail": {"state": str(cell.session.state)},
            },
        )
    _refuse_unroutable_prompt(cell, body.prompt)
    try:
        run = _registry(cell).start(cell, prompt=body.prompt, picks=body.picks)
    except RunConflict as conflict:
        raise HTTPException(
            status_code=409,
            detail={
                "code": "run_active",
                "message": str(conflict),
                "detail": {"run_id": conflict.existing.id},
            },
        ) from conflict
    return RunOut(**run.to_dict())


@router.post("/pick/stop", response_model=RunOut, summary="Do not start the next attempt")
def post_stop(cell: Annotated[Console, Depends(console)], run_id: str | None = None) -> RunOut:
    """Stop, in the only sense this process can honour.

    The flag is read by the pick loop before it begins the next attempt, so a motion already in flight
    completes. There is deliberately nothing stronger here: a stop that travels over a socket depends on
    latency, an open tab and an awake laptop, and the one that does not is the button on the wall.
    """
    registry = _registry(cell)
    run = registry.get(run_id) if run_id else registry.active()
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "no_such_run", "message": "no run to stop.", "detail": {}},
        )
    registry.stop(run.id)
    return RunOut(**run.to_dict())


@router.get("/runs", response_model=list[RunOut], summary="Recent runs, newest first")
def get_runs(cell: Annotated[Console, Depends(console)], limit: int = 50) -> list[RunOut]:
    return [RunOut(**run.to_dict()) for run in _registry(cell).recent(limit)]


@router.get("/runs/{run_id}", response_model=RunOut, summary="One run")
def get_run(cell: Annotated[Console, Depends(console)], run_id: str) -> RunOut:
    run = _registry(cell).get(run_id)
    if run is None:
        raise HTTPException(
            status_code=404,
            detail={"code": "no_such_run", "message": f"no run {run_id!r}.", "detail": {}},
        )
    return RunOut(**run.to_dict())


@router.websocket("/events")
async def events_socket(socket: WebSocket, run_id: str, since_seq: int = 0) -> None:
    """Replay what was missed, then go live.

    ``since_seq=0`` means "from the beginning". A client reconnecting after a sleep sends the last seq
    it rendered and receives everything after it, and, if it slept longer than the ring buffer, a
    ``gap`` frame saying how many events it will never see. A short replay that looked complete would
    be worse than the gap: the UI would draw a run that never had those steps.
    """
    from api.cell import console as _console

    cell = _console()
    hub = cell.hub
    await socket.accept()

    replay, dropped = hub.since(run_id, since_seq)
    logger.info(
        "Event socket opened for %s from seq %d: replaying %d event(s).",
        run_id, since_seq, len(replay),
    )
    if dropped:
        # The ring buffer is the only thing that knows it lost events, and this is its only caller.
        logger.warning(
            "%d event(s) of run %s scrolled out of the buffer before the client reconnected.",
            dropped, run_id,
        )
        await socket.send_json({
            "type": "gap",
            "run_id": run_id,
            "dropped": dropped,
            "human": (
                f"{dropped} event(s) scrolled out of the buffer before you reconnected and cannot be "
                f"replayed. What follows is complete from here on."
            ),
        })
    for event in replay:
        await socket.send_json(event.to_dict())
    cursor = replay[-1].seq if replay else since_seq

    try:
        while True:
            # The blocking wait runs off the event loop: it parks a thread on a condition variable, and
            # doing that on the loop itself would stall every other request on the server.
            pending = await asyncio.to_thread(hub.wait_for, run_id, cursor, timeout=_KEEPALIVE_S)
            if not pending:
                await socket.send_json({"type": "keepalive", "run_id": run_id, "seq": cursor})
                continue
            for event in pending:
                await socket.send_json(event.to_dict())
            cursor = pending[-1].seq
    except WebSocketDisconnect:
        # Expected: a closed tab. The run keeps going, which is the whole design, and the log says so
        # explicitly, because "the browser went away" and "the run stopped" look identical from a UI.
        logger.info(
            "Event socket for %s closed at seq %d; the run is unaffected.", run_id, cursor
        )
        return
