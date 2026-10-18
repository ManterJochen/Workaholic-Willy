"""What a VLM that cannot be loaded means, and when a command may be read.

A cell whose VLM cannot be loaded (its weights missing, a dependency absent, no VRAM) refuses: a pick's grounding
comes back empty with the failure counted, which the console's task ends ``detector_failed`` on, and a command is
refused with :class:`VlmUnavailableError`'s sentence. There is no other model to fall back to (the owner,
2026-10-17: GroundingDINO left, and with it the fallback the owner's cell never ran).

The command reader has its own availability rule here, :func:`reader_availability` (owner decision
Q8 A): one shared VLM copy, which a command may load only on a cell whose detector it is. On any
other cell a person loads it first ("Laden" in the ready bar), because the copy holds gigabytes of
VRAM that the cell's own detector does not need. Neither a command nor Laden loads it beside a copy
of other weights: only a rebuild switches the checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .holder import VlmHolder

__all__ = [
    "CommandRefused",
    "ReaderAvailability",
    "ReaderRefusal",
    "ReaderState",
    "VlmAnswerFailedError",
    "VlmCopyConflictError",
    "VlmNotLoadedError",
    "VlmUnavailableError",
    "reader_availability",
    "vlm_detects",
]

class VlmUnavailableError(RuntimeError):
    """The VLM was needed and could not be loaded.

    Carries the underlying cause, so an operator sees whether the weights are missing, a dependency
    is absent or the GPU is out of VRAM. Raised rather than returned as an empty answer: "nothing was
    understood" and "I could not read" are different answers, and a caller must not take the second
    for the first.
    """

    def __init__(self, model_id: str, cause: BaseException | None = None) -> None:
        detail = f": {type(cause).__name__}: {cause}" if cause is not None else ""
        super().__init__(
            f"the VLM {model_id!r} could not be loaded{detail}. Fetch its weights, or free the card's memory; "
            f"nothing else grounds a prompt or reads a command on this cell, so it refuses rather than guessing."
        )
        self.model_id = model_id
        self.cause = cause


class VlmAnswerFailedError(VlmUnavailableError):
    """The VLM is loaded and failed while it answered a text question: out of VRAM on a shared card, mostly.

    A :class:`VlmUnavailableError`, so whoever maps "unavailable" maps this too (the console's
    ``vlm_unavailable``); its sentence says what happened instead of telling a person to install
    weights that are there.
    """

    def __init__(self, model_id: str, cause: BaseException) -> None:
        super(VlmUnavailableError, self).__init__(
            f"the VLM {model_id!r} is loaded but could not answer: {type(cause).__name__}: {cause}. "
            f"On a card shared with SAM2, Whisper and the cuRobo sidecar this is mostly VRAM; send the "
            f"command again, or fill in the card by hand."
        )
        self.model_id = model_id
        self.cause = cause


# --- the command reader's availability (owner decision Q8 A) ---------------------------------------------------

#: The command reader's state, as the console's commands light and ``GET /v1/commands/status`` show it.
ReaderState: TypeAlias = Literal["ready", "idle", "loading", "missing", "failed"]
#: Why a command is not read. The same strings are the console's refusal codes (``api/codes.py``).
ReaderRefusal: TypeAlias = Literal["vlm_not_loaded", "vlm_unavailable", "vlm_model_missing"]


class VlmNotLoadedError(Exception):
    """The VLM is not loaded, and the caller may not load it.

    Raised by a text asker built with ``may_load=False`` before anything is loaded, so a cell whose
    detector is not the VLM never pays for the model because a command arrived (owner decision Q8 A).
    """

    def __init__(self, model_id: str) -> None:
        super().__init__(
            f"the VLM {model_id!r} is not loaded, and reading a command does not load it on this cell: "
            f"a person loads it first (Laden in the ready bar, POST /v1/commands/warmup)."
        )
        self.model_id = model_id


class VlmCopyConflictError(VlmNotLoadedError):
    """Reading would load a second VLM copy beside one of other weights; only a rebuild switches them.

    A :class:`VlmNotLoadedError` (the console's ``vlm_not_loaded``), raised before anything is built
    or loaded. ``cause`` names both copies and says to rebuild the cell.
    """

    def __init__(self, model_id: str, cause: str) -> None:
        super(VlmNotLoadedError, self).__init__(cause)
        self.model_id = model_id
        self.cause = cause


class CommandRefused(Exception):
    """The command reader refused before it read anything: the console's code, and one sentence why."""

    def __init__(self, code: ReaderRefusal, cause: str, *,
                 availability: "ReaderAvailability | None" = None) -> None:
        super().__init__(f"{code}: {cause}")
        self.code: ReaderRefusal = code
        self.cause = cause
        self.availability = availability


@dataclass(frozen=True, slots=True)
class ReaderAvailability:
    """Whether a command can be read on this cell now, and why; decided without loading anything.

    ``refusal`` is what reading a command answers and ``warmup_refusal`` what an explicit load
    answers: a failed load and an idle copy on a cell whose detector is not the VLM refuse the first
    and allow the second, which is how a person gets the model loaded.
    """

    state: ReaderState
    refusal: ReaderRefusal | Literal[""]
    warmup_refusal: ReaderRefusal | Literal[""]
    cause: str
    model_id: str | None
    weights_present: bool | None
    #: The VLM is this cell's detector, so the one copy reads commands and detects.
    shared_with_detection: bool
    #: A command may load the weights itself (only where the VLM is the detector, Q8 A).
    may_load: bool
    load_s: float | None = None
    last_answer_ms: float | None = None
    #: The ``models.pipeline.zero_shot.vlm`` block the reader asks with.
    vlm: Any = field(default=None, repr=False, compare=False)
    #: The folder its weights are read from (``models.weights_root``), for a copy a command or Laden builds.
    weights_root: str | None = field(default=None, repr=False, compare=False)

    @property
    def may_parse(self) -> bool:
        return not self.refusal

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        lines = [f"command reader: {self.state}" + (f" ({self.model_id})" if self.model_id else "")]
        if self.refusal:
            lines.append(f"  a command is refused: {self.refusal}")
        if self.warmup_refusal:
            lines.append(f"  loading is refused  : {self.warmup_refusal}")
        lines.append(f"  shared with detection: {'yes' if self.shared_with_detection else 'no'}")
        lines.append(f"  {to_ascii(self.cause)}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "refusal": self.refusal,
            "warmup_refusal": self.warmup_refusal,
            "cause": self.cause,
            "model_id": self.model_id,
            "weights_present": self.weights_present,
            "shared_with_detection": self.shared_with_detection,
            "may_load": self.may_load,
            "may_parse": self.may_parse,
            "load_s": self.load_s,
            "last_answer_ms": self.last_answer_ms,
        }


def vlm_detects(models: Any) -> bool:
    """Whether a cell built from ``models`` detects with the VLM (``models.pipeline.kind: zero_shot``): the cell of
    Q8 A whose first command may load the copy. One rule for the availability below and for the console, which
    releases the copy when it builds a cell that does not
    (:meth:`~src.models.vlm.holder.VlmHolder.release_unless_requested`).
    """
    pipeline = getattr(models, "pipeline", None)
    return bool(pipeline is not None and pipeline.kind == "zero_shot")


def reader_availability(
    models: Any, *, weights_present: bool | None = None, holder: "VlmHolder | None" = None,
) -> ReaderAvailability:
    """Can a command be read now? Owner decision Q8 A, decided from config and the holder alone.

    ``models`` is the models section the cell was BUILT with (anything with a ``pipeline`` field):
    a models YAML edited since names weights the cell does not hold. ``weights_present`` is whether
    the checkpoint is on this box (the console checks it without loading: the weights folder's copy,
    or the Hugging Face cache); ``None`` means not checked, and a load that then finds no weights
    fails with the cause. ``holder`` defaults to the process's shared one.

    Loads nothing and builds nothing: it reads what the holder already holds. In this order:

    * The copy is loaded: commands are read (``ready``), on any cell.
    * A load is running: a VLM cell waits for it; any other cell is refused until it is done.
    * The weights are missing: refused (``missing``), loading too, whatever an earlier load said.
    * A copy of other weights is held, or still loaded in a cell built with it: refused
      (``vlm_not_loaded``), loading too. One copy per process: only a rebuild switches the weights.
    * The last load failed: refused until a person loads again; a command never retries it.
    * Not loaded on a cell whose detector is the VLM: the first command loads it.
    * Not loaded on any other cell: refused (``vlm_not_loaded``) until a person loads it.
    """
    from src.models.weights import fetch_hint  # noqa: PLC0415 (stdlib only)

    pipeline = models.pipeline
    if holder is None:
        from .holder import shared_vlm

        holder = shared_vlm()
    vlm = pipeline.zero_shot.vlm
    model_id = str(vlm.model_id)
    detector = vlm_detects(models)
    status = holder.status_for(vlm)
    common: dict[str, Any] = dict(
        model_id=model_id, weights_present=weights_present, shared_with_detection=detector,
        load_s=status.load_s, last_answer_ms=status.last_answer_ms, vlm=vlm,
        weights_root=getattr(models, "weights_root", None),
    )
    if status.state == "ready":
        return ReaderAvailability(
            state="ready", refusal="", warmup_refusal="", may_load=False,
            cause=(f"the VLM {model_id!r} is loaded" + (
                " and is this cell's detector: one copy reads commands and detects." if detector
                else "; it reads commands, and this cell detects with its own detector.")),
            **common,
        )
    if status.state == "loading":
        return ReaderAvailability(
            state="loading", refusal="" if detector else "vlm_not_loaded", warmup_refusal="",
            may_load=detector,
            cause=(f"the VLM {model_id!r} is loading; "
                   + ("the command is read once it is ready." if detector
                      else "read the command again once the commands light is green.")),
            **common,
        )
    if weights_present is False:
        # Before a failed load: a load that failed for want of the weights is answered by fetching them,
        # and Laden would only fail again.
        return ReaderAvailability(
            state="missing", refusal="vlm_model_missing", warmup_refusal="vlm_model_missing",
            may_load=False,
            cause=(f"the VLM checkpoint {model_id!r} is not on this box, and nothing is downloaded at run time: "
                   f"{fetch_hint(model_id).replace('`', '')}."),
            **common,
        )
    conflict = holder.conflict_for(vlm)
    if conflict:
        return ReaderAvailability(
            state="idle", refusal="vlm_not_loaded", warmup_refusal="vlm_not_loaded", may_load=False,
            cause=conflict, **common,
        )
    if status.state == "failed":
        return ReaderAvailability(
            state="failed", refusal="vlm_unavailable", warmup_refusal="", may_load=False,
            cause=(f"the VLM {model_id!r} failed to load ({status.cause}). A command does not try again: "
                   f"fix the cause, then load it with Laden (POST /v1/commands/warmup)."),
            **common,
        )
    if detector:
        return ReaderAvailability(
            state="idle", refusal="", warmup_refusal="", may_load=True,
            cause=(f"the VLM {model_id!r} is this cell's detector and loads at the first command it reads; "
                   f"one copy reads commands and detects."),
            **common,
        )
    return ReaderAvailability(
        state="idle", refusal="vlm_not_loaded", warmup_refusal="", may_load=False,
        cause=(f"the VLM {model_id!r} is not this cell's detector (kind {pipeline.kind}), so a command does "
               f"not load it: a person loads it with Laden in the ready bar (POST /v1/commands/warmup). It holds "
               f"gigabytes of VRAM beside the cell's own models."),
        **common,
    )


#: German letters folded to their ASCII spelling; anything else non-ASCII is escaped.
_ASCII_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def to_ascii(text: str) -> str:
    """``text`` as ASCII, for a ``render()`` that a cp1252 console prints: "Würfel" reads "Wuerfel"."""
    return text.translate(_ASCII_FOLD).encode("ascii", "backslashreplace").decode("ascii")
