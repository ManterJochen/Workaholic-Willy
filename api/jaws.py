"""The hand as the console reads it, and the jaws question answered in the browser.

A toggle hand (the owner's Hand-E on one tool output, no sensor) counts its own changes from where a person said the
jaws stood. At Connect, and at a check (Restart, Setup, the ready bar), it asks where they stand. The console asks
only in the browser: a blocking ``input()`` on the server's terminal cannot be cancelled, and nobody watching the
console sees it (build plan 1.6, item 10).

:func:`hand_of` is how the hand chip, the ready bar's gripper light and the stop card's gates read the hand, and it
only ever reads.

The question in the browser (:class:`BrowserJawQuestions`) is the structured seam the library's hand takes
(``JawIOGripper.answer_questions_with``), handed over at every build (:func:`install`). It keeps one promise above all,
**a question nobody answered is never "open"**:

* each question the hand asks (one per stage and attempt: where do the jaws stand, then, after "closed", open them
  now or abort) is published on the cell's own stream (``cell.jaws_question``) and shown by ``GET /v1/cell/jaws``, with no
  default; it waits at most :data:`JAW_QUESTION_TIMEOUT_S` for ``POST /v1/cell/jaws/answer``, and a timeout, a cancel
  or a Disconnect is ``EOFError``, which the hand takes as no answer and refuses with nothing sent. At a check where the
  hand's own count says CLOSED and nothing moved the jaws since (a task stopped holding its part), the hand asks no
  "where": its first question is "open them now", so no click on "open" can turn round a count a part is clamped
  against (``JawIOGripper.confirm_where_the_jaws_stand``, review of 2026-10-02);
* "open now" is the one answer that moves something, ONE change of the output, and the hand sends it only once the
  gate :func:`before_change` let it out: no run holding the cell, the arm not halted, a stop record that a person has
  said the cell is clear of (or none), and the controller's own fields saying it can move;
* an answer that leaves the jaws open stamps the console (``jaws_confirmed_at``; after "open now" also
  ``jaws_opened_at``, so the next moving run counts down while a person's hand may still be at the jaws) and tells the
  planner the hand is empty (``arm.detach_payload()``), but only where the hand itself says its jaws stand open by the
  end: a solenoid whose feedback reads a part keeps them closed, and nothing is stamped then;
* while a connect's or a check's questions are in progress, from the first question until the one change they led to
  is done and its stroke waited out, a question is pending (:func:`pending`), and every moving route refuses meanwhile.
  A check is pending from its very start: it marks itself before it reads the run lock, so a check and a run never go on
  together. A run that started all the same is read again before anybody is asked, and ends a check's waiting question
  at once (no answer, never "open").

The hand is read on every poll of ``GET /v1/cell``: whatever the driver exposes to :func:`hand_of` (its ``where_pin``
included) must be a lock-free read that never blocks. One that raises anyway is said, never raised.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol, runtime_checkable

from api.codes import REFUSAL_STATUS, JawsChoice, RefusalCode
from api.constants import API_LOG_DIR
from api.events import CELL_STREAM, EventHub, Severity
from api.schemas import HandOut, JawQuestionOut, JawsOut
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover
    from api.cell import Console

__all__ = [
    "ANSWER_SETTLES_S",
    "JAW_QUESTION_TIMEOUT_S",
    "JAWS_LOG_FILE",
    "BrowserJawQuestions",
    "JawAskingLike",
    "JawsRefused",
    "PendingJawQuestion",
    "answer",
    "before_change",
    "cancel_pending",
    "check",
    "controller_stop",
    "ended_open",
    "halt_refusal",
    "hand_of",
    "install",
    "pending",
    "uncleared_stop",
]

#: How long a question waits for the browser's answer. Unanswered, it is refused: never "open".
JAW_QUESTION_TIMEOUT_S = 120.0
#: How long ``POST /v1/cell/jaws/answer`` waits for the hand to move on after an answer: the next question asked, or
#: the question's connect or check at its end (the one change sent and its stroke waited out). Then it answers anyway.
ANSWER_SETTLES_S = 5.0
#: How often a check's waiting question looks whether a run took the cell meanwhile; such a run ends the question.
_RUN_LOOK_S = 0.1
#: This module's log, beside the console's others under ``logs/api`` (``api/constants.py`` names theirs).
JAWS_LOG_FILE: Final[str] = "jaws.log"

logger = create_logger("BrowserJaws", JAWS_LOG_FILE, log_dir=API_LOG_DIR)

#: The answers the browser may give, in the console's words (``api.codes.JawsChoice``).
_CHOICES: Final[frozenset[str]] = frozenset(choice.value for choice in JawsChoice)
#: Why a check asks, as the hand puts it into its question.
_CHECK_REASON = "the console asks where they stand before the arm moves again"


def hand_of(gripper: object) -> HandOut:
    """The hand as the console shows it, read now; never raises and never commands.

    A toggle with no sensor (``toggle_without_sensor_of``) says where the program's count puts its jaws: ``open`` or
    ``closed`` where the count stands, ``unknown`` with the driver's sentence where nobody can vouch for it (a change
    that failed, the output switched at the pendant, an output that cannot be read), ``not_counted`` while it is not
    connected, because the count ends at a disconnect and the next connect asks a person. Every other hand keeps no
    count: ``not_counted``, with what reads it back (a width sensor, feedback switches, a vacuum switch) in
    ``no_sensor``.

    Every read is guarded, whether the hand is a toggle and whether it measures its width included: a read that raises
    is said, and fails closed. A toggle nobody can read stands ``unknown``, never ``open``; a sensor nobody can read is
    not claimed.
    """
    from src.robot.core.gripper import (  # noqa: PLC0415 (the console imports this module to start)
        TwoStateGripper,
        toggle_without_sensor_of,
        why_toggle_count_unknown,
        width_is_measured_of,
    )
    from src.robot.grippers.null import NullGripper  # noqa: PLC0415

    if gripper is None or isinstance(gripper, NullGripper):
        return HandOut(kind="none", driver="" if gripper is None else type(gripper).__name__)
    driver = type(gripper).__name__
    connected = _read(lambda: getattr(gripper, "is_connected", False) is True, False)
    where = _where(gripper)
    sent = _read(lambda: getattr(gripper, "commands_sent", None), None)
    commands_sent = sent if isinstance(sent, int) and not isinstance(sent, bool) else None

    try:
        toggle = toggle_without_sensor_of(gripper) is not None
    except Exception as exc:  # noqa: BLE001 (a panel must not fail because one read did)
        # Only a hand with a toggle's properties can raise reading them: a hand without them answers AttributeError,
        # which the check takes as no toggle. So this is a toggle whose count nobody can vouch for now.
        return HandOut(kind="toggle", driver=driver, where=where, connected=connected, jaws="unknown",
                       why_unknown=f"the jaws could not be read ({type(exc).__name__}: {exc})", no_sensor=True,
                       commands_sent=commands_sent)
    if toggle:
        why = _read(lambda: why_toggle_count_unknown(gripper), "the jaws could not be read")
        if not connected:
            jaws = "not_counted"
        elif why:
            jaws = "unknown"
        else:
            closed = _read(lambda: getattr(gripper, "jaws_closed", None), None)
            jaws = "closed" if closed is True else "open" if closed is False else "unknown"
            if jaws == "unknown":
                why = "the hand did not say where its count puts the jaws"
        return HandOut(kind="toggle", driver=driver, where=where, connected=connected, jaws=jaws,  # type: ignore[arg-type]
                       why_unknown=why, no_sensor=True, commands_sent=commands_sent)

    if _read(lambda: _is_suction(gripper), False):
        # The vacuum switch is the only thing that reads a suction hold back; a cup without one measures nothing.
        return HandOut(kind="suction", driver=driver, where=where, connected=connected,
                       no_sensor=_read(lambda: getattr(gripper, "_ok_pin", None) is None, True),
                       commands_sent=commands_sent)
    measured = _read(lambda: width_is_measured_of(gripper), False)
    if _read(lambda: isinstance(gripper, TwoStateGripper), False) and not measured:
        return HandOut(kind="jaw", driver=driver, where=where, connected=connected,
                       no_sensor=not bool(_read(lambda: getattr(gripper, "has_feedback", False), False)),
                       commands_sent=commands_sent)
    return HandOut(kind="width", driver=driver, where=where, connected=connected, no_sensor=not measured,
                   commands_sent=commands_sent)


def _read(read: Callable[[], Any], fallback: Any) -> Any:
    """One read of the hand; a read that raises is the fallback, said by the caller, never raised to the page."""
    try:
        return read()
    except Exception:  # noqa: BLE001 (a panel must not fail because one read did)
        return fallback


def _where(gripper: object) -> str:
    """The bank and pin the jaws are driven on, as a person reads it on the pendant (``tool output 0``).

    The driver says it in its own questions; a public ``where_pin`` (a property or a method) is preferred where a
    driver offers one. A read that raises says nothing.
    """
    for name in ("where_pin", "_where_pin"):
        value = _read(partial(_said, gripper, name), "")
        if isinstance(value, str) and value:
            return value
    return ""


def _said(gripper: object, name: str) -> Any:
    """What ``gripper`` says under ``name``: the property's value, or the method's answer."""
    said = getattr(gripper, name, None)
    return said() if callable(said) else said


def _is_suction(gripper: object) -> bool:
    from src.robot.grippers.vacuum import VacuumGripper  # noqa: PLC0415

    return isinstance(gripper, VacuumGripper)


# --- the question in the browser ----------------------------------------------------------------------------------


@runtime_checkable
class JawAskingLike(Protocol):
    """One question a toggle hand asks, as the browser seam reads it: the library's ``JawAsking`` (library L10), which
    the hand hands to the seam its ``answer_questions_with`` installed, once per attempt. Read-only, so a frozen
    dataclass answers it; ``tests/test_api_contract.py`` pins these names against the library's once it has landed.
    The library's ``of`` (how many attempts there are) is read where it is there, and is 3 where not."""

    @property
    def stage(self) -> str:
        """``where`` (where do the jaws stand) or ``open_now`` (open them with one change of the output, or abort)."""
        ...

    @property
    def at_connect(self) -> bool:
        """Asked by a Connect; otherwise by a check (Restart, Setup, the ready bar)."""
        ...

    @property
    def reason(self) -> str:
        """Why the hand asks, in its own words."""
        ...

    @property
    def where(self) -> str:
        """The bank and pin the jaws are driven on (``tool output 0``)."""
        ...

    @property
    def choices(self) -> tuple[str, ...]:
        """The driver's own answers this stage takes."""
        ...

    @property
    def text(self) -> str:
        """The question as the terminal would have asked it."""
        ...

    @property
    def attempt(self) -> int:
        """Which attempt this is, from 1."""
        ...

    @property
    def why_again(self) -> str:
        """Why the hand asks again; ``""`` on the first attempt."""
        ...


class JawsRefused(Exception):
    """A jaws route refused, with its code from the catalog (``api.codes``) and one sentence; ``detail`` is the machine
    half. The route answers it at ``REFUSAL_STATUS[code]`` in the one envelope."""

    def __init__(self, code: RefusalCode, message: str, detail: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = dict(detail or {})

    @property
    def status(self) -> int:
        return REFUSAL_STATUS[self.code]


@dataclass(frozen=True, slots=True)
class PendingJawQuestion:
    """The jaws question in progress: what ``cell.jaws_question`` carried.

    ``waiting`` while it waits for the browser's answer (``GET /v1/cell/jaws`` offers it); ``False`` once it is
    answered and the hand still acts on it (the next stage not asked yet, or the one change it led to going out and its
    stroke waited out), and for a check that has begun and asked nothing yet (``question`` is ``None`` then: it reads
    the run lock, the latch and the controller first). Pending either way: a moving route started meanwhile would drive
    the arm while a person's hand is at the jaws.
    """

    question: JawQuestionOut | None
    waiting: bool = True

    def to_out(self) -> JawQuestionOut | None:
        return self.question


@dataclass(eq=False)
class _Asked:
    """One question, and how it ended: ``answered`` (with ``choice``), ``no_answer`` or ``cancelled`` (with ``why``)."""

    out: JawQuestionOut
    timeout_s: float
    ended: str = ""
    choice: str = ""
    why: str = ""


@dataclass(eq=False)
class _Flow:
    """The questions one connect or one check asks, from its start until the console said how it ended."""

    at: Literal["connect", "check"]
    asked: list[_Asked] = field(default_factory=list)
    #: A cancel (a Disconnect) came while this flow was open: every later question of it is no answer at once.
    cancelled: str = ""
    open: bool = True
    #: What ends this flow's questions at once, said in a clause (a run that took the cell, for a check), ``""`` while
    #: nothing does. Read under the seam's lock, so it takes no lock of its own.
    blocked_by: Callable[[], str] | None = None

    @property
    def last(self) -> _Asked | None:
        return self.asked[-1] if self.asked else None

    @property
    def in_progress(self) -> bool:
        """Pending to the console: a check from its very start (it marks itself before it reads the run lock, so a run
        and a check never go on together), a connect once it asked anything (a connecting cell starts no run)."""
        return self.open and (self.at == "check" or bool(self.asked))

    def blocked(self) -> str:
        """Why this flow's questions must end now (:attr:`blocked_by`), ``""`` for nothing. A read that raises blocks:
        nobody can say then that no run holds the cell."""
        if self.blocked_by is None:
            return ""
        try:
            return str(self.blocked_by() or "")
        except Exception as exc:  # noqa: BLE001 (a guard nobody can read lets nothing be asked)
            return f"whether a run holds the cell could not be read ({type(exc).__name__}: {exc})"

    def unanswered(self) -> str:
        """How the flow's questions went unanswered: ``no_answer``, ``cancelled`` (its question, or a cancel between two
        of them, which ends every later one at once), or ``""`` where they were answered."""
        last = self.last
        if last is not None and last.ended == "no_answer":
            return "no_answer"
        if self.cancelled or (last is not None and last.ended == "cancelled"):
            return "cancelled"
        return ""

    def unanswered_said(self) -> str:
        """What a refusal adds where the flow's questions went unanswered: nobody answered, or they were cancelled."""
        how = self.unanswered()
        last = self.last
        if how == "no_answer" and last is not None:
            return (f"Nobody answered the jaws question in the browser within {last.timeout_s:g} s, and no answer is "
                    "never 'open'.")
        if how == "cancelled":
            why = self.cancelled or (last.why if last is not None else "")
            return f"The jaws question in the browser was cancelled ({why}), and no answer is never 'open'."
        return ""


class BrowserJawQuestions:
    """The structured question seam the console hands a toggle hand (the library's ``answer_questions_with``).

    Called by the hand with one ``JawAsking`` per stage and attempt, it publishes the question on the cell's stream
    (``cell.jaws_question``), waits at most :data:`JAW_QUESTION_TIMEOUT_S` for :meth:`answer`, and returns the choice as
    the browser gave it, one of the question's own choices (``open``/``closed``, then ``open_now``/``abort``). A timeout,
    :meth:`cancel` (a Disconnect, a shutdown) and a question asked while another waits raise ``EOFError``, which the hand
    takes as no answer and refuses, never as "open". :meth:`pending` answers the question in progress, and
    ``tests/test_api_contract.py`` asks on a thread and cancels once the question waits.

    A connect's or a check's questions are one flow (:meth:`open_flow`, :meth:`close_flow`): the question stays pending
    from the first one (a check: from its start) until the console closed the flow, the change it led to sent and
    stamped, and ``cell.jaws_ended`` says how it ended. One flow at a time: a second is refused ``jaws_question_pending``.
    A flow's ``blocked_by`` (a check's: a run that took the cell) ends its questions at once, shown or not.
    """

    def __init__(self, hub: EventHub, *, timeout_s: float | None = None) -> None:
        self.hub = hub
        #: ``None``: :data:`JAW_QUESTION_TIMEOUT_S` as it stands when a question is asked.
        self._timeout_s = timeout_s
        self._cond = threading.Condition(threading.RLock())
        self._waiting: _Asked | None = None
        self._flow: _Flow | None = None
        #: Every question ever asked, counted: an answer waits for the hand to ask the next one.
        self._asked = 0

    @property
    def timeout_s(self) -> float:
        """How long a question waits for its answer."""
        return float(JAW_QUESTION_TIMEOUT_S if self._timeout_s is None else self._timeout_s)

    # --- the hand's side ------------------------------------------------------------------------------------------

    def __call__(self, asking: JawAskingLike) -> str:
        timeout = self.timeout_s
        with self._cond:
            flow = self._flow if self._flow is not None and self._flow.open else None
            if flow is not None and flow.cancelled:
                raise EOFError(f"the jaws question was cancelled ({flow.cancelled})")
            if self._waiting is not None:
                raise EOFError("another jaws question already waits for its answer in the browser")
            blocked = flow.blocked() if flow is not None else ""
            if blocked:
                # Nothing is shown: a question about the jaws while a run drives the arm would call a person to it.
                raise EOFError(f"the jaws question was not asked: {blocked}")
            out = _question_out(asking, at=flow.at if flow is not None else None, expires_at=time.time() + timeout)
            asked = _Asked(out, timeout)
            self._waiting = asked
            self._asked += 1
            if flow is not None:
                flow.asked.append(asked)
            # Published under the lock, so the question is on the stream before anybody can answer it.
            self.hub.publish(
                CELL_STREAM, "cell.jaws_question", severity=Severity.WARN, human=_asked_said(out, timeout),
                data=out.model_dump(),
            )
            self._cond.notify_all()
            logger.warning("Jaws question %s (%s, %s) waits in the browser for at most %g s: %s", out.question_id,
                           out.stage, out.at, timeout, out.reason)
            deadline = time.monotonic() + timeout
            guarded = flow is not None and flow.blocked_by is not None
            while not asked.ended:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    asked.ended, asked.why = "no_answer", f"nobody answered within {timeout:g} s"
                    break
                if guarded:
                    assert flow is not None  # guarded says so
                    blocked = flow.blocked()
                    if blocked:
                        # A run took the cell while the question waited: it ends now, as no answer, never "open".
                        asked.ended, asked.why = "cancelled", blocked
                        logger.warning("Jaws question %s ended: %s.", out.question_id, blocked)
                        break
                    self._cond.wait(min(remaining, _RUN_LOOK_S))
                else:
                    self._cond.wait(remaining)
            if self._waiting is asked:
                self._waiting = None
            self._cond.notify_all()
        if asked.ended == "answered":
            return asked.choice
        logger.warning("Jaws question %s ended unanswered (%s): the hand refuses, nothing is sent.", out.question_id,
                       asked.why)
        raise EOFError(asked.why)

    # --- the console's side ---------------------------------------------------------------------------------------

    def pending(self) -> PendingJawQuestion | None:
        """The question in progress: the one waiting for its answer, else the last one of a connect or check still
        acting on it, or a check that has begun and asked nothing yet (``question`` ``None``); ``None`` where none
        is."""
        with self._cond:
            if self._waiting is not None:
                return PendingJawQuestion(self._waiting.out)
            flow = self._flow
            if flow is not None and flow.in_progress:
                last = flow.last
                return PendingJawQuestion(last.out if last is not None else None, waiting=False)
            return None

    def waiting(self) -> JawQuestionOut | None:
        """The question waiting for the browser's answer, or ``None``: what ``GET /v1/cell/jaws`` offers."""
        with self._cond:
            return self._waiting.out if self._waiting is not None else None

    def cancel(self, why: str) -> bool:
        """End the waiting question as no answer (``EOFError`` to the hand), never "open"; ``True`` where one waited.

        A connect or a check whose question this ends asks nothing more: every later question of it is no answer at
        once, so a Disconnect never waits behind its next stage.
        """
        with self._cond:
            ended = False
            asked = self._waiting
            if asked is not None and not asked.ended:
                asked.ended, asked.why = "cancelled", why
                ended = True
            if self._flow is not None and self._flow.open and not self._flow.cancelled:
                self._flow.cancelled = why
            self._cond.notify_all()
        if ended:
            logger.warning("The waiting jaws question was cancelled: %s", why)
        return ended

    def answer(self, question_id: str, choice: str, *, settle_s: float | None = None) -> PendingJawQuestion | None:
        """Answer the waiting question with one of its choices, then wait (at most ``settle_s``, by default
        :data:`ANSWER_SETTLES_S`) for the hand to move on; the next question waiting, or ``None``.

        Refused (:class:`JawsRefused`): ``no_question`` where none waits, ``question_changed`` for another question's
        id (a stale tab), ``choice_not_offered`` for a choice the question does not offer; the question goes on waiting.
        """
        with self._cond:
            asked = self._waiting
            if asked is None or asked.ended:
                raise JawsRefused(RefusalCode.NO_QUESTION, (
                    "no jaws question waits for an answer: it was answered, ran out or was cancelled, or none was "
                    "asked; GET /v1/cell/jaws shows the one that waits"))
            if asked.out.question_id != question_id:
                raise JawsRefused(RefusalCode.QUESTION_CHANGED, (
                    f"that answer is for question {question_id}, and the question waiting now is "
                    f"{asked.out.question_id}: read it again before answering"),
                    {"question_id": asked.out.question_id})
            if choice not in asked.out.choices:
                raise JawsRefused(RefusalCode.CHOICE_NOT_OFFERED, (
                    f"{choice!r} is not offered by this question ({asked.out.stage}): it takes "
                    f"{', '.join(asked.out.choices)}"), {"choices": list(asked.out.choices)})
            asked.ended, asked.choice = "answered", choice
            self._waiting = None
            self.hub.publish(CELL_STREAM, "cell.jaws_answered", human=f"Answered in the browser: {choice}.",
                             question_id=question_id, choice=choice)
            logger.warning("Jaws question %s answered in the browser: %s", question_id, choice)
            self._cond.notify_all()
            before, flow = self._asked, self._flow
            deadline = time.monotonic() + (ANSWER_SETTLES_S if settle_s is None else settle_s)
            while flow is not None and flow.open and self._asked == before:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            waiting = self._waiting
            return PendingJawQuestion(waiting.out) if waiting is not None else None

    # --- flows ----------------------------------------------------------------------------------------------------

    def open_flow(self, at: Literal["connect", "check"], *, blocked_by: Callable[[], str] | None = None) -> _Flow:
        """Start the questions of one connect or check; ``jaws_question_pending`` while another is open or a question
        waits.

        ``blocked_by`` answers why the flow's questions must end now, in a clause (a run that took the cell), or
        ``""``: read before each question is shown and while it waits, under this seam's lock, so it must take none.
        """
        with self._cond:
            if (self._flow is not None and self._flow.open) or self._waiting is not None:
                raise JawsRefused(RefusalCode.JAWS_QUESTION_PENDING, (
                    "a jaws question is in progress: answer it in the browser (GET /v1/cell/jaws) and let the hand "
                    "finish what it leads to first"))
            flow = _Flow(at=at, blocked_by=blocked_by)
            self._flow = flow
            return flow

    def close_flow(self, flow: _Flow, *, opened: bool, refusal: str = "", detached: bool = False) -> None:
        """End ``flow``, saying how (``cell.jaws_ended``, where it asked anything). Idempotent: once closed it stays so.

        ``opened``: the jaws stand open by its end. Otherwise its last question says why it ended: ``no_answer``,
        ``cancelled``, or ``refused`` (an abort, a gate that refused the change, a connection that changed, the change
        itself refused).
        """
        with self._cond:
            if not flow.open:
                return
            flow.open = False
            if self._flow is flow:
                self._flow = None
            last = flow.last
            if last is not None:
                outcome = "open" if opened else (flow.unanswered() or "refused")
                self.hub.publish(
                    CELL_STREAM, "cell.jaws_ended",
                    severity=Severity.SUCCESS if opened else Severity.WARN,
                    human=_ended_said(outcome, refusal, detached),
                    question_id=last.out.question_id, outcome=outcome, refusal="" if opened else refusal,
                    detached=bool(detached),
                )
                logger.info("Jaws questions of the %s ended %s%s.", flow.at, outcome,
                            f" ({refusal})" if refusal and not opened else "")
            self._cond.notify_all()


def _question_out(asking: JawAskingLike, *, at: Literal["connect", "check"] | None, expires_at: float) -> JawQuestionOut:
    """The question as the browser reads it; one that cannot be shown is no answer (``EOFError``), never a guess."""
    try:
        choices = [str(choice) for choice in asking.choices if str(choice) in _CHOICES]
        if not choices:
            raise ValueError(f"none of the hand's choices {list(asking.choices)!r} is an answer the browser gives")
        said_at: Literal["connect", "check"] = at if at is not None else (
            "connect" if asking.at_connect is True else "check")
        return JawQuestionOut(
            question_id=f"q-{uuid.uuid4().hex[:10]}", stage=asking.stage,  # type: ignore[arg-type]
            at=said_at, where=str(asking.where), reason=str(asking.reason),
            choices=choices,  # type: ignore[arg-type]
            attempt=int(asking.attempt), of=int(getattr(asking, "of", 3) or 3), why_again=str(asking.why_again),
            expires_at=expires_at, text=str(asking.text),
        )
    except Exception as exc:  # noqa: BLE001 (a question the browser cannot show is a question nobody answered)
        logger.error("A jaws question could not be shown in the browser (%s: %s); it is no answer.",
                     type(exc).__name__, exc)
        raise EOFError(f"the jaws question could not be shown in the browser ({type(exc).__name__}: {exc})") from exc


def _asked_said(out: JawQuestionOut, timeout_s: float) -> str:
    """``cell.jaws_question`` for a person."""
    if out.stage == "open_now":
        return (f"The jaws on {out.where or 'the hand'} stand closed: open them now with ONE change of the output, "
                f"which releases what they hold, or abort. Answer in the browser within {timeout_s:g} s.")
    return (f"Where do the jaws on {out.where or 'the hand'} stand? Answer in the browser within {timeout_s:g} s; no "
            "answer is never 'open'.")


def _ended_said(outcome: str, refusal: str, detached: bool) -> str:
    """``cell.jaws_ended`` for a person."""
    if outcome == "open":
        return "The jaws stand open" + ("; the planner was told the hand is empty." if detached else ".")
    if outcome == "no_answer":
        return "Nobody answered the jaws question, so nothing was sent and the jaws are not taken for open."
    if outcome == "cancelled":
        return "The jaws question was cancelled, so nothing was sent and the jaws are not taken for open."
    return f"The jaws do not stand open: {refusal}" if refusal else "The jaws do not stand open."


# --- the console's doors ------------------------------------------------------------------------------------------


def install(console: "Console") -> None:
    """Hand the browser question to the hand of every build; raises where a hand that asks cannot take it.

    Every hand that takes a structured seam (``answer_questions_with``) gets this console's, with :func:`before_change`
    as its gate; a hand that asks a person (a toggle, or a solenoid that opted into ``confirm_open_at_start``) and
    cannot take one is refused with ``RuntimeError``, and ``Console.build`` then undoes the build: its question could
    only go to the terminal of the server. The session gets the hooks a connect and a Disconnect use either way
    (``CellSession.questions``).
    """
    session = console.session
    seam = _seam_of(console)
    session.questions = _SessionQuestions(console)
    gripper = session.gripper
    asks = _asks_a_person(gripper)
    installer = getattr(gripper, "answer_questions_with", None)
    if not callable(installer):
        if asks:
            raise RuntimeError(
                f"the hand ({type(gripper).__name__}) asks a person where its jaws stand and takes no question seam, "
                "so the console could not ask it in the browser; its question would go to the terminal of the server, "
                "which the console never uses")
        return
    if seam.pending() is not None:
        # The hand's install waits for a question in flight on it: end one first, as no answer.
        seam.cancel("the hand's question was handed to the browser anew")
    installer(seam, before_change=before_change(console))
    installed = getattr(gripper, "question_seam_installed", None)
    if asks and not (callable(installed) and installed() is True):
        raise RuntimeError(
            f"the hand ({type(gripper).__name__}) did not take the browser question: it would ask at the terminal of "
            "the server, which the console never uses")
    logger.info("The jaws question of %s is asked in the browser%s.", type(gripper).__name__,
                "" if asks else " (this hand asks nothing at connect)")


def cancel_pending(console: "Console", why: str) -> bool:
    """End a waiting question as unanswered (a Disconnect, a shutdown): its connect or check is refused, never "open".

    ``True`` where one was waiting. The questions of the connect or check it ends ask nothing more.
    """
    seam = getattr(console, "jaws", None)
    return bool(seam.cancel(why)) if isinstance(seam, BrowserJawQuestions) else False


def pending(console: "Console") -> PendingJawQuestion | None:
    """The jaws question in progress, or ``None``: waiting for its answer, answered while the hand still acts on it, or
    a check that has begun and asks nothing yet (:class:`PendingJawQuestion`). Takes no session lock; every moving route
    and a teach refuse while it is not ``None``, and a run reads it again once it holds the cell (a teach does, before
    it frees the arm)."""
    seam = getattr(console, "jaws", None)
    return seam.pending() if isinstance(seam, BrowserJawQuestions) else None


def answer(console: "Console", question_id: str, choice: str) -> PendingJawQuestion | None:
    """Answer the waiting question; the next question waiting once the hand moved on, or ``None``.

    Refusals (:class:`JawsRefused`): ``no_question``, ``question_changed`` and ``choice_not_offered``.
    """
    seam = getattr(console, "jaws", None)
    if not isinstance(seam, BrowserJawQuestions):
        raise JawsRefused(RefusalCode.NO_QUESTION, "no jaws question waits for an answer on this console")
    return seam.answer(question_id, choice)


def ended_open(console: "Console", opened: bool) -> None:
    """A question ended with the jaws standing open: stamp it through ``console.stamp_jaws_open(opened=opened)``
    (``opened`` after "open now": a person held the part at the jaws, so the next moving run counts down; the stamps are
    read-only, and the countdown reads the order their methods keep) and tell the planner the hand is empty."""
    _ended_open(console, opened=opened)


def before_change(console: "Console") -> Callable[[], str]:
    """The gate the toggle asks immediately before its one change of the output: ``""`` lets it send.

    Read at the moment of the change, on the arm the session holds then. A run holding the cell refuses first, whatever
    it is: a change while a run drives the arm would open the jaws under it, and while a teach has it free, in a
    person's hands (no run asks the jaws: a run's thread asks nobody). Then a halted arm (a halt is never a controller
    stop), then the controller's own fields (``controller_operational``, read from the receive stream where the arm
    offers ``quick_robot_status``); a controller that cannot be read refuses too. Then a stop record nobody has said
    the cell is clear of: after a problem stop nothing moves before "Zelle ist frei", the jaws included (review of
    2026-10-02). An arm that reports no controller at all (the dummy) lets it go.
    """

    def gate() -> str:
        active = getattr(console, "active_run_id", None)
        if active is not None:
            return (f"run {active} holds the cell, and a change of the jaws while it runs would open them under it; the "
                    "output was not changed")
        arm = console.session.arm
        halted = halt_refusal(arm, "the output was not changed")
        if halted:
            return halted
        stopped = controller_stop(arm)
        if stopped:
            return f"{stopped}; the output was not changed"
        uncleared = uncleared_stop(console)
        return f"{uncleared}; the output was not changed" if uncleared else ""

    return gate


def controller_stop(arm: Any) -> str:
    """Why the controller cannot move, from its own fields alone, or ``""`` where it can or the arm reports none.

    The console's one rule, read where every other gate reads it (``api.readiness.controller_reading``): the receive
    stream (``quick_robot_status``) where the arm offers it, otherwise ``get_robot_status`` (``SupportsRobotStatus``).
    The halt latch is the caller's to read first: ``controller_operational`` is the controller's answer with the halt
    aside. A read that raises refuses, and so does an answer that does not say the controller can move: only
    ``controller_operational`` true lets anything go.
    """
    from api.readiness import controller_reading  # noqa: PLC0415

    return controller_reading(arm)[1]


def uncleared_stop(console: "Console") -> str:
    """Why nothing may move yet after a stop, in a clause, or ``""``: a stop record stands that nobody has said the cell is
    clear of (what ``Console.recovery_gate`` calls ``cell_not_cleared``). The jaws wait for "Zelle ist frei" as every
    motion does: the stop card's order, a person in the cell first. A record that cannot be read refuses too.

    One read of ``Console.recovery`` and no lock: the gate before the one change asks this under the hand's lock, while
    a connect or a check holds the session (the record is a frozen value, replaced whole, never changed in place).
    """
    try:
        record = getattr(console, "recovery", None)
        cleared = record is None or record.cleared is True
    except Exception as exc:  # noqa: BLE001 (a stop nobody can read may stand)
        return f"whether a stop still waits for 'the cell is clear' could not be read ({type(exc).__name__}: {exc})"
    if cleared or record is None:
        return ""
    return (f"the {record.kind} run {record.run_id} stopped where the arm stands ({record.stop_code}), and nobody has "
            "said the cell is clear since: a person confirms it first (POST /v1/cell/acknowledge)")


def check(console: "Console") -> JawsOut:
    """``POST /v1/cell/jaws/check``: ask where the jaws stand now, whatever the count says (a Restart, Setup, the ready
    bar), and answer once the question has ended.

    Refused (:class:`JawsRefused`), in this order: ``not_connected``, ``run_active``, ``halted`` (a latched arm switches
    no output: "the cell is clear" first), ``controller_stopped`` (the controller first, as the pick service orders it),
    ``cell_not_cleared`` (a stop record nobody has said the cell is clear of: the jaws wait for it as every motion does),
    ``jaws_question_pending``, ``jaws_seam_missing`` and ``jaws_not_open`` (the hand's own sentence). A hand that is not a
    toggle answers at once, asking nothing. The check holds the session while it asks (``CellSession.while_connected``),
    so a Disconnect ends its question first and never brings the arm down in the middle of the one change.

    What the hand asks is its own: where the jaws stand, both answers offered; or, where its count says CLOSED and
    nothing moved the jaws since, only whether to open them now (``JawIOGripper.confirm_where_the_jaws_stand``), so
    "open" is no answer there and a refused answer leaves the question waiting.

    A check and a run never go on together. A toggle's check is pending (:func:`pending`) from before it reads the run
    lock, so a moving route or a teach that starts while it reads the latch and the controller is refused
    ``jaws_question_pending``; a question in progress found then is answered in the plan's place. A run that started all
    the same (its route read the question before the check began) is read again right before anybody is asked, and
    ends the check's waiting question at once: ``run_active``, nothing sent. And the gate refuses the one change while
    any run holds the cell. Only where the hand itself says its jaws stand open by the end is anything stamped open.
    """
    from src.robot.core import RobotConnectionError, RobotError  # noqa: PLC0415
    from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

    from api.lifecycle import CellTransitionError  # noqa: PLC0415

    session = console.session
    if not session.connected:
        raise JawsRefused(RefusalCode.NOT_CONNECTED, "the cell is not connected, so there are no jaws to ask about")
    gripper, arm = session.gripper, session.arm
    try:
        toggle = toggle_without_sensor_of(gripper) is not None
    except Exception:  # noqa: BLE001 (a hand nobody can read is asked about as a toggle: it may move at a change)
        toggle = True
    seam = _seam_of(console)
    flow: _Flow | None = None
    in_progress: JawsRefused | None = None
    if toggle:
        # Marked before the run lock is read: from here every moving route and a teach refuse jaws_question_pending.
        try:
            flow = seam.open_flow("check", blocked_by=partial(_run_holding, console))
        except JawsRefused as refusal:
            in_progress = refusal                      # answered in the plan's place, below
    try:
        _refuse_a_run(console, started=False)
        if not toggle:
            return JawsOut(hand=hand_of(gripper), question=None)
        halted = halt_refusal(arm, "nobody was asked and the output was not changed")
        if halted:
            raise JawsRefused(RefusalCode.HALTED, halted)
        stopped = controller_stop(arm)
        if stopped:
            raise JawsRefused(RefusalCode.CONTROLLER_STOPPED, f"{stopped}; nobody was asked")
        uncleared = uncleared_stop(console)
        if uncleared:
            record = console.recovery
            raise JawsRefused(RefusalCode.CELL_NOT_CLEARED, f"{uncleared}; nobody was asked and the output was not "
                              "changed", {"run_id": record.run_id, "stop_code": str(record.stop_code)}
                              if record is not None else {})
        if in_progress is not None:
            raise in_progress
        assert flow is not None  # a toggle's check opened its flow, or was refused just above
        installed = getattr(gripper, "question_seam_installed", None)
        if not (callable(installed) and installed() is True):
            raise JawsRefused(RefusalCode.JAWS_SEAM_MISSING, (
                f"the hand ({type(gripper).__name__}) has no question in the browser installed, and the console never "
                "asks at the terminal of the server it runs in: build the cell again; nobody was asked"))
        # A run whose route read the question before this check began, and started while the latch and the controller
        # were read (a dashboard round trip on a cell): nobody is asked while it runs.
        _refuse_a_run(console, started=True)
        sent = _commands_sent(gripper)
        try:
            with session.while_connected():
                refused = gripper.confirm_where_the_jaws_stand(_CHECK_REASON, before_change=before_change(console))
                # Read while the session is held, before a Disconnect waiting for it can take the hand down.
                not_open = "" if refused else _not_open(gripper)
        except (CellTransitionError, RobotConnectionError) as exc:
            raise JawsRefused(RefusalCode.NOT_CONNECTED, (
                f"the cell is no longer connected ({exc}): nothing was sent")) from exc
        except RobotError as exc:
            refused, not_open = f"{type(exc).__name__}: {exc}", ""
        if refused:
            _refuse_a_run(console, started=True)       # the run that ended the question, or refused the change
            said = flow.unanswered_said()
            raise JawsRefused(RefusalCode.JAWS_NOT_OPEN, f"{refused}{f' {said}' if said else ''}")
        if not_open:
            raise JawsRefused(RefusalCode.JAWS_NOT_OPEN, not_open)
        detached = _ended_open(console, opened=_rose(sent, _commands_sent(gripper)))
        seam.close_flow(flow, opened=True, detached=detached)
    except JawsRefused as refusal:
        if flow is not None:
            seam.close_flow(flow, opened=False, refusal=refusal.message)
        raise
    finally:
        if flow is not None:
            seam.close_flow(flow, opened=False, refusal="the check ended without saying how")
    return JawsOut(hand=hand_of(gripper), question=None)


def _run_holding(console: "Console") -> str:
    """A check's guard (``_Flow.blocked_by``): the run that took the cell, in a clause, or ``""``. Reads one attribute
    and takes no lock: it is read under the seam's."""
    active = getattr(console, "active_run_id", None)
    return f"run {active} took the cell while the check asked" if active is not None else ""


def _refuse_a_run(console: "Console", *, started: bool) -> None:
    """``run_active`` while a run holds the cell: ``started`` for one that began after the check did."""
    active = console.active_run_id
    if active is None:
        return
    if started:
        raise JawsRefused(RefusalCode.RUN_ACTIVE, (
            f"run {active} started while the check was under way, so nobody is asked and nothing is sent while it runs: "
            "check the jaws once it has ended"), {"run_id": active})
    raise JawsRefused(RefusalCode.RUN_ACTIVE, (
        f"run {active} is active, and a question about the jaws while it runs would answer for a hand the run is "
        "using; check them once it has ended"), {"run_id": active})


def _not_open(gripper: Any) -> str:
    """``""`` where the hand itself says its jaws stand open (its count or command open, and no feedback that reads a
    part), else why they are not taken for open. Reads only; a read that raises takes nothing for open."""
    from src.robot.core.gripper import HoldEvidence, hold_evidence_of  # noqa: PLC0415

    try:
        closed = getattr(gripper, "jaws_closed", None)
        if closed is True:
            return ("the hand keeps its jaws closed by the end (its feedback reads a part between them, or they were "
                    "not opened), so they are not taken for open and the planner keeps its part")
        if closed is not False:
            return "the hand does not say its jaws stand open, so they are not taken for open"
        if hold_evidence_of(gripper) is HoldEvidence.HELD:
            return ("the hand's feedback reads a part between its jaws, so they are not taken for open and the planner "
                    "keeps its part")
    except Exception as exc:  # noqa: BLE001 (a hand nobody can read holds whatever it holds)
        return f"the hand could not be read ({type(exc).__name__}: {exc}), so its jaws are not taken for open"
    return ""


# --- the session's hooks (``CellSession.questions``) ----------------------------------------------------------------


class _SessionQuestions:
    """What the session asks of the browser question: watch a connect, end a waiting question (a Disconnect)."""

    def __init__(self, console: "Console") -> None:
        self._console = console

    def connecting(self, gripper: Any) -> "_ConnectWatch":
        return _ConnectWatch(self._console, gripper)

    def cancel(self, why: str) -> bool:
        return cancel_pending(self._console, why)


class _ConnectWatch:
    """One connect's questions: opened as the connect starts, closed with how it ended."""

    def __init__(self, console: "Console", gripper: Any) -> None:
        self._console = console
        self._gripper = gripper
        self._seam = _seam_of(console)
        self._sent = _commands_sent(gripper)
        self._flow = self._seam.open_flow("connect")

    def succeeded(self) -> None:
        """The connect came up: where a question was answered and the hand says its jaws stand open by the end, stamp
        it and detach.

        A connect that comes up need not have opened them: a solenoid that asks first and then connects by its feedback
        keeps them closed where the feedback reads a part between them. The person's answer alone is no open jaws, so
        the hand is read again here; where it does not say open, nothing is stamped and the planner keeps its part.
        """
        try:
            if self._flow.asked:
                not_open = _not_open(self._gripper)
                if not_open:
                    logger.warning("The connect came up with the jaws not open: %s.", not_open)
                    self._seam.close_flow(self._flow, opened=False, refusal=not_open)
                    return
                opened = _rose(self._sent, _commands_sent(self._gripper))
                detached = _ended_open(self._console, opened=opened)
                self._seam.close_flow(self._flow, opened=True, detached=detached)
        finally:
            self._seam.close_flow(self._flow, opened=False, refusal="the connect ended without saying how")

    def failed(self, error: BaseException) -> str:
        """The connect was refused: what the refusal adds about the question (nobody answered, it was cancelled)."""
        said = self._flow.unanswered_said()
        self._seam.close_flow(self._flow, opened=False, refusal=f"{type(error).__name__}: {error}")
        return said


# --- helpers ----------------------------------------------------------------------------------------------------------


def _seam_of(console: "Console") -> BrowserJawQuestions:
    """The console's question seam; one is built where a console has none yet (it needs only the hub)."""
    seam = getattr(console, "jaws", None)
    if not isinstance(seam, BrowserJawQuestions):
        seam = BrowserJawQuestions(console.hub)
        console.jaws = seam
    return seam


def _asks_a_person(gripper: Any) -> bool:
    """Whether ``gripper`` asks a person where its jaws stand: a toggle always, a solenoid that opted in at connect."""
    from src.robot.core.gripper import toggle_without_sensor_of  # noqa: PLC0415

    try:
        return toggle_without_sensor_of(gripper) is not None or getattr(gripper, "asks_at_connect", False) is True
    except Exception:  # noqa: BLE001 (a hand nobody can read may ask: it is asked about in the browser or not at all)
        return True


def _ended_open(console: "Console", *, opened: bool) -> bool:
    """Stamp a question that ended with the jaws open, then tell the planner the hand is empty; whether it was told."""
    console.stamp_jaws_open(opened=opened)
    detach = getattr(console.session.arm, "detach_payload", None)
    if not callable(detach):
        return False
    try:
        return bool(detach())
    except Exception as exc:  # noqa: BLE001 (the jaws stand open whatever the planner answered; said, not raised)
        logger.error("Telling the planner the hand is empty failed (%s: %s); the next pick attaches anew.",
                     type(exc).__name__, exc)
        return False


def _commands_sent(gripper: Any) -> int | None:
    """How many commands the hand has sent, where it counts them (``commands_sent``)."""
    try:
        sent = getattr(gripper, "commands_sent", None)
    except Exception:  # noqa: BLE001 (a count nobody can read is no count)
        return None
    return sent if isinstance(sent, int) and not isinstance(sent, bool) else None


def _rose(before: int | None, after: int | None) -> bool:
    """Whether a command went out between two counts; where either is unknown, it may have (the countdown is due)."""
    return True if before is None or after is None else after > before


def halt_refusal(arm: Any, what: str) -> str:
    """The halt latch's own sentence (``halted_refusal``) while the arm is latched, else ``""``: read first by every gate
    here, so a halt never reads as a controller stop. ``what`` says what was not done."""
    from src.robot.core.arm_capabilities import halt_state_of, halted_refusal  # noqa: PLC0415

    halted = halt_state_of(arm)
    return halted_refusal(halted.reason, what) if halted is not None else ""
