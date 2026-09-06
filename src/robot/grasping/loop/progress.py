"""What a pick is doing while it does it.

``BinPickingOrchestrator.run()`` is a single blocking call that returns one report at the end. The
loop inside it emits progress events, so a subscriber learns something while the pick is running and
not only once it is over. A CLI printing a result line needs nothing here; an operator console does.

Default-off and byte-identical. With no listener attached, every emit is one ``is None`` test. The
payload is not constructed, no string is formatted, nothing is measured that was not already measured.
That is why :func:`emit` takes the pieces rather than a built :class:`PickProgress`: building the
event to then discard it would be the cost this design exists to avoid.

Typed. The stage is a :class:`StrEnum` and the payload a frozen dataclass, so the fields a console
renders cannot change shape unnoticed. ``CalibrationRoutine`` carries the untyped form of the same
seam, ``(event_type: str, data: dict)``.

Emission cannot break a pick. Listener exceptions are swallowed, exactly as the calibration seam
does. A browser that disconnects mid-render, a subscriber with a bug, a full disk in a log listener:
none of those may abort a motion that is already in flight.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final, Protocol, runtime_checkable

__all__ = [
    "PickProgress",
    "PickProgressListener",
    "PickStage",
    "ShouldCancel",
    "emit",
]

_LOG = logging.getLogger(__name__)


class PickStage(StrEnum):
    """The canonical stages of one pick, in the order they occur.

    The member set is a fixed contract. Adding a member is a breaking change for a console's replay
    buffer and for any UI that maps stages to labels, so it is a deliberate act, not growth.
    """

    #: One pick has begun. Carries ``attempt_total``.
    PICK_STARTED = "pick_started"
    #: One attempt within that pick has begun. Carries ``attempt``/``attempt_total``.
    ATTEMPT_STARTED = "attempt_started"
    #: A camera frame was acquired. Carries ``segmentation_count``.
    PERCEIVED = "perceived"
    #: Candidates were generated and ranked. Carries ``candidate_count``, ``score``, ``target_index``.
    RANKED = "ranked"
    #: No usable candidate this attempt. Carries ``reasons`` and the chosen ``action``.
    NO_CANDIDATE = "no_candidate"
    #: About to command motion. Carries the executed grasp's ``position_mm`` and ``score``.
    EXECUTING = "executing"
    #: The attempt ended. Carries ``action`` and, on the executing path, whether it succeeded.
    ATTEMPT_FINISHED = "attempt_finished"
    #: The pick ended. Carries the typed ``outcome``.
    PICK_FINISHED = "pick_finished"
    #: The loop stopped because a caller asked it to, between attempts. Carries ``attempt``.
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class PickProgress:
    """One progress event. Every field beyond ``stage`` is optional and stage-dependent.

    Deliberately flat and JSON-shaped, so an event serialises onto a wire with no translation layer;
    a nested structure would mean the wire format and this class drift apart the first time one of
    them changes.
    """

    stage: PickStage
    #: 0-based index of the attempt this event belongs to. ``None`` for pick-level events.
    attempt: int | None = None
    attempt_total: int | None = None

    segmentation_count: int | None = None
    candidate_count: int | None = None
    target_index: int | None = None
    score: float | None = None
    #: BASE-frame grasp position, millimetres, on :attr:`PickStage.EXECUTING`.
    position_mm: tuple[float, float, float] | None = None
    #: Typed rejection reasons, as strings. Never free text: these come from ``GraspFailureReason``.
    reasons: tuple[str, ...] = ()
    #: ``executed`` / ``next_target`` / ``rescan`` / ``relocate`` / ``exhausted``.
    action: str | None = None
    #: The typed ``PickOutcome`` on :attr:`PickStage.PICK_FINISHED`.
    outcome: str | None = None
    #: Which perception route grounded this frame (``simple`` / ``vlm``), and the rule that chose it.
    #: Both ``None`` unless the cell runs a routed pipeline; an un-routed cell is byte-identical.
    #:
    #: Named fields rather than entries in :attr:`extra`: a console renders a sentence from them, and
    #: the route is usually what explains a slow pick or a grasp on the wrong object.
    route: str | None = None
    route_reason: str | None = None
    #: Why a motion did not happen, on :attr:`PickStage.ATTEMPT_FINISHED`. All three ``None`` on the
    #: healthy path.
    #:
    #: Named fields for the same reason as ``route`` above. The progress event is the only thing a
    #: console sees while a run is happening; without these three, a safety guard that refuses a pose
    #: with a precise, actionable message surfaces in the browser as the bare word
    #: ``execution_failed``. ``PickAttempt`` carries the same three.
    #:
    #: ``motion_status`` is typed (``workspace_rejected``, ``ik_failed``, ...); ``motion_message`` is
    #: the guard's own sentence; ``motion_error`` is a driver exception, when there was one.
    motion_status: str | None = None
    motion_message: str | None = None
    motion_error: str | None = None
    #: Anything a stage needs that does not deserve a field of its own.
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PickProgressListener(Protocol):
    """Callable the orchestrator notifies. Must be fast and must not raise.

    Exceptions are swallowed, but a slow listener still blocks the pick loop: it is called inline,
    on the thread driving the robot. A console listener therefore hands the event to a queue and
    returns; it does not write to a socket here.
    """

    def __call__(self, event: PickProgress) -> None: ...


@runtime_checkable
class ShouldCancel(Protocol):
    """Asked at the top of every attempt: should the loop stop rather than start another one?

    "Stop" means do not begin the next attempt. It never interrupts a motion that is already in
    flight: there is no safe way to do that from here, and the physical stop is the one that can.
    """

    def __call__(self) -> bool: ...


#: Sentinel for "no listener attached"; the ``is None`` test against it keeps the default path free.
_NO_LISTENER: Final[None] = None


def emit(listener: "PickProgressListener | None", stage: PickStage, **fields: Any) -> None:
    """Build and deliver one event; with no listener, do nothing at all.

    The ``is None`` test comes first and the :class:`PickProgress` is constructed after it, so the
    default path costs one comparison per call site and allocates nothing. Callers pass keyword pieces
    rather than a built event for exactly that reason.
    """
    if listener is _NO_LISTENER:
        return
    try:
        listener(PickProgress(stage=stage, **fields))  # type: ignore[misc]
    except Exception:  # noqa: BLE001 (a subscriber must never be able to abort a motion in flight)
        _LOG.debug("pick progress listener raised on %s; ignored", stage, exc_info=True)
