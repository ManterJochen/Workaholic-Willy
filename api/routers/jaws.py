"""The jaws question of a toggle hand, answered in the browser.

A toggle on one output with no sensor moves its jaws at every change of the output, and only a person can say where
they stand. Connect asks, and so does a check (Restart, Setup, the ready bar); the question waits here, never at the
server's terminal, and it has no default: a question nobody answers within 120 s is refused, never "open".

``open_now`` is the one answer that moves something: one change of the output, which opens the jaws where the arm
stands, sent only after the controller said it can move.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api import jaws as browser_jaws
from api.cell import Console, console
from api.schemas import JawAnswerIn, JawsOut

router = APIRouter(prefix="/cell/jaws", tags=["jaws"])


def _refuse(refused: browser_jaws.JawsRefused) -> HTTPException:
    return HTTPException(
        status_code=refused.status,
        detail={"code": str(refused.code), "message": refused.message, "detail": refused.detail},
    )


@router.get("", response_model=JawsOut, summary="The hand, and the jaws question waiting for its answer")
def get_jaws(cell: Annotated[Console, Depends(console)]) -> JawsOut:
    """Takes no session lock, like ``GET /v1/cell``: a connect waits on the question while this is polled.

    ``question`` is the one waiting for the browser's answer, with its choices and no default; null once it is
    answered, also while the hand still acts on the answer, and while a check reads the latch and the controller before
    it asks (``CellOut.jaws_question`` stays true throughout).
    """
    seam = getattr(cell, "jaws", None)
    waiting = seam.waiting() if isinstance(seam, browser_jaws.BrowserJawQuestions) else None
    return JawsOut(hand=browser_jaws.hand_of(cell.session.gripper), question=waiting)


@router.post("/answer", response_model=JawsOut, summary="Answer the jaws question (open_now: ONE change of the output)")
def post_jaws_answer(cell: Annotated[Console, Depends(console)], body: JawAnswerIn) -> JawsOut:
    """``open`` or ``closed`` where the question asks where they stand; ``open_now`` or ``abort`` after "closed".

    Answers once the hand moved on (at most a few seconds): ``question`` is the next one where it asks another, and
    null where its connect or check has ended. Refused: ``no_question`` (404), ``question_changed`` (409, a stale id)
    and ``choice_not_offered`` (422); the question goes on waiting.
    """
    try:
        nxt = browser_jaws.answer(cell, body.question_id, body.choice)
    except browser_jaws.JawsRefused as refused:
        raise _refuse(refused) from refused
    question = nxt.to_out() if nxt is not None and nxt.waiting else None
    return JawsOut(hand=browser_jaws.hand_of(cell.session.gripper), question=question)


@router.post("/check", response_model=JawsOut, summary="Ask where the jaws stand now (blocks until answered)")
def post_jaws_check(cell: Annotated[Console, Depends(console)]) -> JawsOut:
    """Always asks, even where the count says open. A hand that is not a toggle answers at once, asking nothing.

    Blocks until the question has ended: up to 120 s per question, which the browser answers through
    ``POST /v1/cell/jaws/answer``. Refused, in order: ``not_connected``, ``run_active``, ``halted`` (confirm the cell is
    clear first: a latched arm switches no output), ``controller_stopped``, ``cell_not_cleared`` (a stop record nobody
    has said the cell is clear of: the jaws wait for it, as every motion does), ``jaws_question_pending``,
    ``jaws_seam_missing``, and ``jaws_not_open`` with the hand's own sentence (closed and aborted, no answer, the change
    refused). Where the hand's count says closed and nothing moved the jaws since, its first question is ``open_now``:
    "open" is no answer there. A check and a run never go on together: the check is a jaws question in progress from its start, so a
    moving route or a teach that comes in meanwhile is refused, and a run that took the cell all the same ends the
    check's question at once (``run_active``, nothing sent).
    """
    try:
        return browser_jaws.check(cell)
    except browser_jaws.JawsRefused as refused:
        raise _refuse(refused) from refused
