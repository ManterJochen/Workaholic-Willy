"""What happens when the VLM cannot be loaded: refuse the prompt, or degrade with a warning.

``refuse`` means the prompt is not taken at all; ``degrade`` means the fallback answers it and every
such answer is warned about. The default is ``refuse``, because a silent degrade is the failure this
route exists to prevent.

The complex prompts that reach this route are the ones the phrase grounder gets confidently wrong:
it returns a high-scoring box for the wrong object rather than admitting defeat. A quiet fallback
therefore does not mean slightly worse perception; it means the cell grasps something the operator
did not ask for, with nothing in the log to say why.

The command reader has its own availability rule here, :func:`reader_availability` (owner decision
Q8 A): one shared VLM copy, which a command may load only on a cell whose detector it is. On any
other cell a person loads it first ("Laden" in the ready bar), because the copy holds gigabytes of
VRAM that the cell's own detector does not need. Neither a command nor Laden loads it beside a copy
of other weights: only a rebuild switches the checkpoint.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

from src.models.constants import MODELS_LOG_DIR, VLM_AVAILABILITY_LOG_FILE
from src.models.perception_backend import PerceivedObject, failures_of, last_failure_of
from src.utility.log_cfg import create_logger

if TYPE_CHECKING:  # pragma: no cover (typing only)
    from .holder import VlmHolder

__all__ = [
    "CommandRefused",
    "GuardedVlmBackend",
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

#: The default sink for this module's two lines. Under a bare ``getLogger`` the degrade warning
#: below has no handler and is discarded, and that is the one line here that must not be lost. The
#: name stays ``__name__``; an injected ``logger=`` overrides it per instance.
_LOG = create_logger(__name__, log_file=VLM_AVAILABILITY_LOG_FILE, log_dir=MODELS_LOG_DIR)


class VlmUnavailableError(RuntimeError):
    """The VLM route was required and the model could not be loaded.

    Carries the underlying cause, so an operator sees whether the weights are missing, a dependency
    is absent or the GPU is out of VRAM. Raised rather than returned as an empty result: "no
    objects found" and "I could not look" are different answers, and the pick loop must not treat
    the second as the first.
    """

    def __init__(self, model_id: str, cause: BaseException | None = None) -> None:
        detail = f": {type(cause).__name__}: {cause}" if cause is not None else ""
        super().__init__(
            f"the prompt needs the VLM route and {model_id!r} could not be loaded{detail}. "
            f"Install the weights, or set models.pipeline.zero_shot.vlm.on_unavailable=degrade to fall "
            f"back to the phrase grounder (which will be WRONG on prompts of this kind, loudly logged)."
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


class GuardedVlmBackend:
    """Wraps the VLM backend with the ``on_unavailable`` contract.

    Delegates to ``vlm`` while it works. The first time loading fails it either raises
    :class:`VlmUnavailableError` (``refuse``) or hands over to the fallback (``degrade``), and the
    degrade path warns on every use rather than once, so a run that has fallen back never looks
    normal again.

    The fallback arrives as a factory, not an instance, and is built only if degradation happens.
    Constructing it eagerly would load GroundingDINO and SAM2 on every cell that merely might
    degrade, spending seconds and gigabytes of VRAM on a path a healthy cell never runs.
    """

    def __init__(
        self,
        *,
        vlm: Any,
        model_id: str,
        degrade: bool = False,
        fallback_factory: Callable[[], Any] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        if degrade and fallback_factory is None:
            raise ValueError(
                "on_unavailable='degrade' needs a fallback backend to degrade to; refusing to "
                "construct a backend whose degraded path is also unavailable."
            )
        self._vlm = vlm
        self._model_id = model_id
        self._degrade = degrade
        self._fallback_factory = fallback_factory
        self._fallback: Any = None
        self._log = logger or _LOG
        #: Set once the VLM has failed to load, so a multi-second load is not retried per pick.
        self._unavailable: BaseException | None = None
        self._last_failure = ""

    @property
    def degraded(self) -> bool:
        """True once this backend has fallen back; surfaced by the console and the run report."""
        return self._unavailable is not None

    @property
    def failures(self) -> int:
        """The wrapped backends' counted failures: the VLM route's, plus the fallback's once built.

        A perceive that returned nothing because a model raised, as
        :attr:`~src.models.perception_backend.TwoStageBackend.failures` counts it. Only grows.
        """
        total = failures_of(self._vlm)
        if self._fallback is not None:
            total += failures_of(self._fallback)
        return total

    @property
    def last_failure(self) -> str:
        """The latest counted failure, from the VLM route or the fallback, or ``""``."""
        return self._last_failure

    def _through(self, backend: Any, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        """Perceive with ``backend``, keeping its failure sentence when its count rose."""
        before = failures_of(backend)
        try:
            return tuple(backend.perceive(image_bgr, prompt))
        finally:
            if failures_of(backend) > before:
                self._last_failure = last_failure_of(backend)

    def perceive(self, image_bgr: Any, prompt: str) -> tuple[PerceivedObject, ...]:
        if self._unavailable is None:
            try:
                return self._through(self._vlm, image_bgr, prompt)
            except VlmUnavailableError as exc:
                self._unavailable = exc.cause or exc
            except (ImportError, OSError, RuntimeError) as exc:
                # The three shapes a missing or unloadable model takes: no dependency, no weights on
                # disk, no VRAM. Anything else is a bug and must not be swallowed by a fallback.
                self._unavailable = exc
            if self._unavailable is not None:
                self._log.error(
                    "VLM %s became unavailable: %s: %s",
                    self._model_id, type(self._unavailable).__name__, self._unavailable,
                )

        if self._unavailable is None:  # pragma: no cover (unreachable; kept for narrowing)
            return ()
        if not self._degrade:
            raise VlmUnavailableError(self._model_id, self._unavailable)

        self._log.warning(
            "DEGRADED: %r needs the VLM route but %s is unavailable; grounding with the phrase "
            "detector instead. It does not fail loudly on prompts of this kind; it returns a "
            "confident box for the WRONG object. Treat this result as unverified.",
            prompt, self._model_id,
        )
        if self._fallback is None:
            assert self._fallback_factory is not None  # guaranteed by __init__
            self._fallback = self._fallback_factory()
        return self._through(self._fallback, image_bgr, prompt)


# --- the command reader's availability (owner decision Q8 A) ---------------------------------------------------

#: The command reader's state, as the console's commands light and ``GET /v1/commands/status`` show it.
ReaderState: TypeAlias = Literal["ready", "idle", "loading", "missing", "failed", "not_configured"]
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
    #: The ``models.pipeline.zero_shot.vlm`` block the reader asks with; ``None`` when none is named.
    vlm: Any = field(default=None, repr=False, compare=False)

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
    """Whether a cell built from ``models`` detects with the VLM, routed or not: the cell of Q8 A whose
    first command may load the copy. One rule for the availability below and for the console, which
    releases the copy when it builds a cell that does not (:meth:`~src.models.vlm.holder.VlmHolder.release_unless_requested`).
    """
    pipeline = getattr(models, "pipeline", None)
    return bool(pipeline is not None and pipeline.kind == "zero_shot" and pipeline.zero_shot.backend == "vlm")


def reader_availability(
    models: Any, *, weights_present: bool | None = None, holder: "VlmHolder | None" = None,
) -> ReaderAvailability:
    """Can a command be read now? Owner decision Q8 A, decided from config and the holder alone.

    ``models`` is the models section the cell was BUILT with (anything with a ``pipeline`` field):
    a models YAML edited since names weights the cell does not hold. ``weights_present`` is whether
    the checkpoint is on this box (the console checks it without loading: a local path that exists,
    or the Hugging Face cache); ``None`` means not checked, and a load that then finds no weights
    fails with the cause. ``holder`` defaults to the process's shared one.

    Loads nothing and builds nothing: it reads what the holder already holds. In this order:

    * No ``models.pipeline`` block: no checkpoint is named (``not_configured``).
    * The copy is loaded: commands are read (``ready``), on any cell.
    * A load is running: a VLM cell waits for it; any other cell is refused until it is done.
    * The weights are missing: refused (``missing``), loading too, whatever an earlier load said.
    * A copy of other weights is held, or still loaded in a cell built with it: refused
      (``vlm_not_loaded``), loading too. One copy per process: only a rebuild switches the weights.
    * The last load failed: refused until a person loads again; a command never retries it.
    * Not loaded on a cell whose detector is the VLM (routed or not): the first command loads it.
    * Not loaded on any other cell: refused (``vlm_not_loaded``) until a person loads it.
    """
    pipeline = getattr(models, "pipeline", None)
    if pipeline is None:
        return ReaderAvailability(
            state="not_configured", refusal="vlm_unavailable", warmup_refusal="vlm_unavailable",
            cause=("no VLM is named for this cell: the models config has no models.pipeline block, so "
                   "there is no models.pipeline.zero_shot.vlm checkpoint to read commands with. Add the "
                   "pipeline block (its backend may stay grounded_sam) and fetch the weights."),
            model_id=None, weights_present=weights_present, shared_with_detection=False, may_load=False,
        )
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
            cause=(f"the VLM checkpoint {model_id!r} is not on this box; fetch it first "
                   f"(python scripts/model_weights/fetch.py vlm-4b)."),
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
        cause=(f"the VLM {model_id!r} is not this cell's detector (backend {pipeline.zero_shot.backend}, "
               f"kind {pipeline.kind}), so a command does not load it: a person loads it with Laden in the "
               f"ready bar (POST /v1/commands/warmup). It holds gigabytes of VRAM beside the cell's own models."),
        **common,
    )


#: German letters folded to their ASCII spelling; anything else non-ASCII is escaped.
_ASCII_FOLD = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "Ae", "Ö": "Oe", "Ü": "Ue"})


def to_ascii(text: str) -> str:
    """``text`` as ASCII, for a ``render()`` that a cp1252 console prints: "Würfel" reads "Wuerfel"."""
    return text.translate(_ASCII_FOLD).encode("ascii", "backslashreplace").decode("ascii")
