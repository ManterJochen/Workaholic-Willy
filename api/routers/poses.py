"""Poses: Home and the taught poses, the default place, and teaching one pose by hand.

Home comes from the config and is read-only here. A taught pose is guided by a person with the arm freed, captured
where it stands still, screened by the exact guard and the planner at once, and written, only when it is clear or in
the planner's band, into the cell profile's own layer through the pose writer alone. An unscreened or refused pose is
never written.

Every poll of a teach session is the browser's heartbeat: without one for 3 s, or at the 5-minute limit, the arm is
held once it stands still, never while it moves in a person's hands.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api import teach as browser_teach
from api.cell import Console, console
from api.codes import REFUSAL_STATUS, RefusalCode
from api.schemas import DefaultPlaceIn, PayloadOut, PosesOut, RunOut, TeachIn, TeachStartOut, TeachStateOut

router = APIRouter(tags=["poses"])


def _refuse(refused: browser_teach.TeachSessionRefused) -> HTTPException:
    return HTTPException(
        status_code=refused.status,
        detail={"code": str(refused.code), "message": refused.message, "detail": refused.detail},
    )


def _no_such_run(run_id: str) -> HTTPException:
    return HTTPException(
        status_code=REFUSAL_STATUS[RefusalCode.NO_SUCH_RUN],
        detail={"code": str(RefusalCode.NO_SUCH_RUN), "message": f"no teach session runs as {run_id}.",
                "detail": {"run_id": run_id}},
    )


@router.get("/poses", response_model=PosesOut, summary="Home and the taught poses")
def get_poses(cell: Annotated[Console, Depends(console)]) -> PosesOut:
    """Home (read-only), the named poses with their labels and screens, the default place, the file a taught pose is
    written to, and whether one can be taught now (``teachable``; ``why_not`` says why not, ``why_not_code`` as its
    code). Moves nothing."""
    return browser_teach.poses(cell)


@router.put("/poses/default-place", response_model=PosesOut, summary="Choose the default place pose")
def put_default_place(cell: Annotated[Console, Depends(console)], body: DefaultPlaceIn) -> PosesOut:
    """Where a task puts its part when the command names no target; null to have none.

    Written into the cell's own layer, all or nothing. Refused: ``run_active`` (409), ``unknown_pose`` (422),
    ``no_layer`` (422: the chain has no layer of the cell's own), ``invalid_value`` (422).
    """
    try:
        return browser_teach.choose_default_place(cell, body.name)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused


# Declared before ``/teach/{run_id}``: a path parameter would otherwise take "payload" for a run id.
@router.get("/teach/payload", response_model=PayloadOut, summary="The payload the controller compensates for")
def get_teach_payload(cell: Annotated[Console, Depends(console)]) -> PayloadOut:
    """It decides whether a freed arm floats, sinks or rises; the person confirms it before the arm is freed."""
    try:
        return browser_teach.payload(cell)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused


@router.post("/teach", response_model=TeachStartOut, status_code=202, summary="Teach one pose (FREES THE ARM for a person)")
def post_teach(cell: Annotated[Console, Depends(console)], body: TeachIn) -> TeachStartOut:
    """The arm is freed for a person to guide by hand. Refused while the planner is not ready, since every new pose
    is screened at once.

    Refused before anything is freed, in order: ``not_connected``, ``run_active``, ``halted``, ``controller_stopped``,
    ``cell_not_cleared``, ``jaws_question_pending``, ``part_in_hand``, ``jaws_not_confirmed`` (a toggle count nobody can
    vouch for), ``invalid_name``, ``no_hand_guiding``, ``screen_unavailable``, ``planner_not_ready``, ``invalid_label``,
    ``name_taken``, ``no_layer`` and ``payload_changed``. The token in the answer goes with every poll, Save and Cancel.
    """
    try:
        run, token = browser_teach.start(cell, body)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused
    return TeachStartOut(run=RunOut.model_validate(run.to_dict()), token=token)


@router.get("/teach/{run_id}", response_model=TeachStateOut, summary="A teach session (every poll is the heartbeat)")
def get_teach(cell: Annotated[Console, Depends(console)], run_id: str, token: str) -> TeachStateOut:
    """Where the session stands. Without a poll for 3 s the arm is held, once it stands still."""
    try:
        state = browser_teach.snapshot(cell, run_id, token)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused
    if state is None:
        raise _no_such_run(run_id)
    return state


@router.post("/teach/{run_id}/capture", response_model=TeachStateOut, summary="Save the pose where the arm stands")
def post_teach_capture(cell: Annotated[Console, Depends(console)], run_id: str, token: str) -> TeachStateOut:
    """Captured once the arm stands still, then held, screened and written where the verdict allows. Refused
    ``not_free`` where the arm is not free."""
    try:
        state = browser_teach.capture(cell, run_id, token)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused
    if state is None:
        raise _no_such_run(run_id)
    return state


@router.post("/teach/{run_id}/cancel", response_model=TeachStateOut, summary="Hold the arm and save nothing")
def post_teach_cancel(cell: Annotated[Console, Depends(console)], run_id: str, token: str) -> TeachStateOut:
    """Holds at once. The page also sends it when it is closed; a session that already ended answers as it stands."""
    try:
        state = browser_teach.cancel(cell, run_id, token)
    except browser_teach.TeachSessionRefused as refused:
        raise _refuse(refused) from refused
    if state is None:
        raise _no_such_run(run_id)
    return state
