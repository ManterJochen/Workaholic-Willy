"""The process's one Qwen3-VL copy, shared by detection and the command reader.

    from src.models.vlm import shared_vlm

    vlm = models.pipeline.zero_shot.vlm
    grounder = shared_vlm().grounder_for(vlm)      # the build's door: built once per process, never loaded here
    shared_vlm().status_for(vlm)                   # idle | loading | ready | failed, and why; builds nothing
    shared_vlm().load_for(vlm)                     # the explicit load, "Laden": loads, or says why it did not
    ask = shared_vlm().asker_for(vlm, may_load=True)
    ask(system, user)                              # the model's text answer to one question
    shared_vlm().release_unless_requested(vlm)     # the console builds a cell that does not detect with it

The 4B checkpoint held 8.93 GB of VRAM on the dev box's RTX 5080, beside SAM2, Whisper and the cuRobo sidecar,
so a second copy is not an option. The perception build takes its grounder from here
(:func:`src.models.factory.build_perception`) and the command reader asks the same object: a cell whose
detector is the VLM reads commands with the copy it detects with.

**One copy per process**, kept per weights (``model_id``, ``model_path``, ``local``). A block that differs only in
``preload`` or ``on_unavailable`` names the same weights and gets the same grounder.

**Only a build switches the weights.** :meth:`VlmHolder.grounder_for` is the build's door: for other weights it
builds a new grounder and unloads the one it replaces, so the card never holds both. A command
(:meth:`VlmHolder.asker_for`) and Laden (:meth:`VlmHolder.load_for`) never load beside a copy of other weights:
while the holder holds one, or a cell built with a replaced one has loaded it again, they refuse before anything is
built or loaded, and the reason says to rebuild the cell (:meth:`VlmHolder.conflict_for`). A models YAML edited
from 4B to 8B without a rebuild costs a refusal, never a second copy.

**Released when no cell needs it.** A cell whose detector is not the VLM holds the copy only after a person
loaded it (owner decision Q8 A), so the console's build of such a cell calls
:meth:`VlmHolder.release_unless_requested`, which unloads a copy that a VLM cell, a command or a look loaded. A
routed cell's build lets a copy of other weights go itself (:meth:`VlmHolder.release_other_than`), because its VLM
route takes its grounder only at the first prompt that needs it.

Nothing loads here unless a caller asks: building a grounder never loads it, :meth:`VlmHolder.status_for`,
:meth:`VlmHolder.loaded_for` and :meth:`VlmHolder.conflict_for` never build one, and the shared holder starts
empty. :meth:`VlmHolder.forget` drops what the holder knows and unloads nothing (a test's teardown). It follows the
pattern of the speech holder (``src/models/speech/holder.py``) without importing it.
"""

from __future__ import annotations

import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from src.contracts import UNSET, Maybe, resolve
from src.models.constants import MODELS_LOG_DIR, VLM_HOLDER_LOG_FILE
from src.utility.log_cfg import create_logger

from .availability import (
    VlmAnswerFailedError,
    VlmCopyConflictError,
    VlmNotLoadedError,
    VlmUnavailableError,
    to_ascii,
)
from .qwen import Qwen3VLGrounder

__all__ = [
    "GrounderBuilder",
    "TextAsker",
    "VlmHolder",
    "VlmState",
    "VlmStatus",
    "VlmWeights",
    "shared_vlm",
]

_LOG = create_logger(__name__, log_file=VLM_HOLDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

#: Where the held copy stands: built but not loaded, loading, loaded, or the last load failed.
VlmState: TypeAlias = Literal["idle", "loading", "ready", "failed"]

#: The shapes a missing or unloadable model takes, as ``GuardedVlmBackend`` reads them: no dependency, no
#: weights on disk, no VRAM. Anything else raised by a load is a bug and surfaces as itself.
_UNAVAILABLE: Final = (ImportError, OSError, RuntimeError)


@dataclass(frozen=True, slots=True)
class VlmWeights:
    """Which weights a VLM block names: what one held copy stands for."""

    model_id: str
    model_path: str | None = None
    local: bool = False

    @classmethod
    def of(cls, vlm: Any) -> "VlmWeights":
        """The weights of a ``models.pipeline.zero_shot.vlm`` block, or of anything with its three fields."""
        model_path = getattr(vlm, "model_path", None)
        return cls(
            model_id=str(vlm.model_id),
            model_path=str(model_path) if model_path else None,
            local=bool(getattr(vlm, "local", False)),
        )

    def describe(self) -> str:
        where = self.model_path or ("the local Hugging Face cache" if self.local else "the Hugging Face hub")
        return f"{self.model_id} from {where}"


@dataclass(frozen=True, slots=True)
class VlmStatus:
    """The held copy for one set of weights: its state, why it failed, what loading and answering cost."""

    state: VlmState
    model_id: str
    cause: str = ""
    load_s: float | None = None
    last_answer_ms: float | None = None

    def __str__(self) -> str:
        return self.render()

    def render(self) -> str:
        line = f"VLM {self.model_id}: {self.state}"
        if self.load_s is not None:
            line += f", loaded in {self.load_s:.1f} s"
        if self.last_answer_ms is not None:
            line += f", last text answer {self.last_answer_ms:.0f} ms"
        if self.cause:
            line += f"\n  cause: {to_ascii(self.cause)}"
        return line

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "model_id": self.model_id,
            "cause": self.cause,
            "load_s": self.load_s,
            "last_answer_ms": self.last_answer_ms,
        }


#: Builds the grounder for one set of weights. It must not load them.
GrounderBuilder: TypeAlias = Callable[[VlmWeights], Qwen3VLGrounder]


def _qwen(weights: VlmWeights) -> Qwen3VLGrounder:
    return Qwen3VLGrounder(model_id=weights.model_id, model_path=weights.model_path, local=weights.local)


def _status_of(grounder: Qwen3VLGrounder) -> VlmStatus:
    error = grounder.load_error
    if grounder.loaded:
        state: VlmState = "ready"
    elif grounder.loading:
        state = "loading"
    elif error is not None:
        state = "failed"
    else:
        state = "idle"
    return VlmStatus(
        state=state, model_id=grounder.model_id,
        cause=f"{type(error).__name__}: {error}" if state == "failed" and error is not None else "",
        load_s=grounder.load_s, last_answer_ms=grounder.last_answer_ms,
    )


def _on_the_card(grounder: Qwen3VLGrounder) -> bool:
    return grounder.loaded or grounder.loading


class TextAsker:
    """One way to put a text question to the held copy: ``asker(system, user)`` is the model's answer.

    The answer opens with ``{`` (the prefill) and continues with what the model wrote, so a JSON object comes back
    whole. ``may_load`` decides what an unloaded copy means: load it first (a cell whose detector is the VLM), or
    refuse with :class:`VlmNotLoadedError` before anything is loaded (any other cell, owner decision Q8 A). A copy
    whose last load failed is never loaded here: Laden or a pick's look tries again, a command does not, and it gets
    :class:`VlmUnavailableError` with that failure. A load that fails for want of a dependency, the weights or VRAM
    is raised as :class:`VlmUnavailableError` with its cause, and an answer that fails the same way (out of VRAM
    mid-answer, mostly) as :class:`VlmAnswerFailedError`; anything else is a bug and surfaces as itself.
    """

    def __init__(self, grounder: Qwen3VLGrounder, *, may_load: bool, max_new_tokens: int = 96) -> None:
        self._grounder = grounder
        self._may_load = bool(may_load)
        self._max_new_tokens = int(max_new_tokens)
        #: True once a call of this asker loaded the weights: the first command a VLM cell read.
        self.loaded_now = False

    @property
    def grounder(self) -> Qwen3VLGrounder:
        """The copy this asker questions, the one the cell detects with where the VLM is its detector."""
        return self._grounder

    @property
    def model_id(self) -> str:
        return self._grounder.model_id

    def __call__(self, system: str, user: str) -> str:
        grounder = self._grounder
        if not grounder.loaded:
            if not self._may_load:
                raise VlmNotLoadedError(grounder.model_id)
            failed = grounder.load_error
            if failed is not None and not grounder.loading:
                # Laden or a pick's look tries a failed load again; a command does not. While one of them is
                # trying, the command waits for that load below, and a waiter never loads (Qwen3VLGrounder).
                raise VlmUnavailableError(grounder.model_id, failed)
            try:
                if grounder.ensure_loaded():
                    self.loaded_now = True
            except _UNAVAILABLE as exc:
                raise VlmUnavailableError(grounder.model_id, exc) from exc
        try:
            # `load=False`: loading was decided above, and weights released since then are not brought back
            # by a command (on a cell that may not load, or beside a copy a rebuild just put on the card).
            return grounder.answer_text(system, user, prefill="{", max_new_tokens=self._max_new_tokens, load=False)
        except _UNAVAILABLE as exc:
            raise VlmAnswerFailedError(grounder.model_id, exc) from exc


class VlmHolder:
    """One grounder per process, for one set of weights at a time. Build it with :meth:`from_parts`;
    :func:`shared_vlm` returns the process's."""

    def __init__(self, *, build: GrounderBuilder) -> None:
        self._build = build
        self._lock = threading.Lock()
        self._weights: VlmWeights | None = None
        self._grounder: Qwen3VLGrounder | None = None
        #: A person loaded the held copy (Laden): the build of a cell that does not detect with it keeps it.
        self._requested = False
        #: Every grounder this holder built, with its weights, for as long as anything holds it. A cell built
        #: with a grounder the holder has since replaced keeps it, and its next look loads it again.
        self._issued: weakref.WeakKeyDictionary[Qwen3VLGrounder, VlmWeights] = weakref.WeakKeyDictionary()

    @classmethod
    def from_parts(cls, *, build: Maybe[GrounderBuilder] = UNSET) -> "VlmHolder":
        """The Python door. Builds nothing yet. ``build`` UNSET is a :class:`Qwen3VLGrounder` for the weights."""
        return cls(build=resolve("build", build, _qwen))

    # --- the build's door ---------------------------------------------------------------------------------------

    def grounder_for(self, vlm: Any) -> Qwen3VLGrounder:
        """The build's door: the grounder for ``vlm``'s weights. Never loads it.

        The held one for the same weights. For other weights a new one, and the one it replaces is unloaded: a
        build is how a cell switches checkpoints, and the console releases the old cell before it builds. A
        program that keeps a cell built with the replaced grounder still has it; its next look loads it again,
        and from then on a command or Laden for the new weights is refused rather than loading beside it.
        """
        weights = VlmWeights.of(vlm)
        with self._lock:
            held, held_weights = self._grounder, self._weights
            if held is not None and held_weights == weights:
                return held
            grounder = self._hold(weights)
        if held is not None:
            self._retire(held, held_weights, why=f"a build names {weights.describe()}")
        return grounder

    def release_other_than(self, vlm: Any) -> bool:
        """Unload a held copy of other weights than ``vlm``'s. Builds and loads nothing; ``True`` when one went.

        For the build of a routed cell, whose VLM route takes its grounder only at the first prompt that needs
        it: the cell is configured with ``vlm``'s weights, so a copy of others is nobody's any more.
        """
        weights = VlmWeights.of(vlm)
        with self._lock:
            held, held_weights = self._grounder, self._weights
            if held is None or held_weights == weights:
                return False
            self._drop_held()
        self._retire(held, held_weights, why=f"a routed build names {weights.describe()}")
        return True

    def release_unless_requested(self, vlm: Any | None) -> bool:
        """Unload the held copy unless a person loaded it (Laden) and ``vlm`` still names its weights.

        For the console's build of a cell whose detector is not the VLM: owner decision Q8 A has such a cell hold
        the VLM only after a person loaded it, so a copy that a VLM cell, a command or a look loaded goes, and its
        gigabytes with it. ``vlm`` is the new cell's ``models.pipeline.zero_shot.vlm`` block, ``None`` when it names
        none. A cell whose detector IS the VLM needs no call: its build takes the copy (:meth:`grounder_for`).
        Builds and loads nothing; ``True`` when a copy was released.
        """
        weights = VlmWeights.of(vlm) if vlm is not None else None
        with self._lock:
            held, held_weights = self._grounder, self._weights
            if held is None:
                return False
            if self._requested and weights == held_weights:
                return False
            why = ("the config now names other weights" if self._requested
                   else "the cell built does not detect with it, and no person loaded it")
            self._drop_held()
        self._retire(held, held_weights, why=why)
        return True

    # --- reading, without building ------------------------------------------------------------------------------

    def held_for(self, vlm: Any) -> Qwen3VLGrounder | None:
        """The grounder held for ``vlm``'s weights, or ``None``. Builds nothing."""
        weights = VlmWeights.of(vlm)
        with self._lock:
            return self._grounder if self._weights == weights else None

    def loaded_for(self, vlm: Any) -> bool:
        """Whether ``vlm``'s weights are loaded. Builds and loads nothing."""
        held = self.held_for(vlm)
        return held is not None and held.loaded

    def status_for(self, vlm: Any) -> VlmStatus:
        """Where ``vlm``'s copy stands. Builds and loads nothing: weights nobody asked for are ``idle``."""
        held = self.held_for(vlm)
        if held is None:
            return VlmStatus(state="idle", model_id=VlmWeights.of(vlm).model_id)
        return _status_of(held)

    def conflict_for(self, vlm: Any) -> str:
        """Why loading ``vlm``'s weights now would put a second copy on the card, or ``""``. Builds and loads
        nothing.

        A copy of other weights is in the way when the holder holds it (in any state: a cell built with it may
        load it at its next look), or when a cell built with a grounder the holder has since replaced loaded it
        again. A copy of ``vlm``'s weights already loaded or loading is never in the way of itself.
        """
        weights = VlmWeights.of(vlm)
        with self._lock:
            return self._in_the_way(weights)

    # --- the doors of a command and of Laden ----------------------------------------------------------------------

    def load_for(self, vlm: Any) -> VlmStatus:
        """The explicit load ("Laden"): load ``vlm``'s weights unless loaded, and say how that went.

        A failed load is tried again here, and only here or by a pick: a command never retries one. A load that
        fails for want of a dependency, the weights or VRAM comes back as a ``failed`` status with its cause;
        anything else raised is a bug and propagates. Beside a copy of other weights it builds and loads nothing
        and answers ``idle`` with the reason (:meth:`conflict_for`). A copy loaded here is one a person asked for,
        and the console's build of a cell that does not detect with it keeps it (:meth:`release_unless_requested`).
        """
        weights = VlmWeights.of(vlm)
        with self._lock:
            conflict = self._in_the_way(weights)
            if conflict:
                _LOG.warning("Laden for VLM %s refused: %s", weights.describe(), conflict)
                return VlmStatus(state="idle", model_id=weights.model_id, cause=conflict)
            grounder = self._grounder if self._grounder is not None else self._hold(weights)
        try:
            if grounder.ensure_loaded():
                _LOG.info("VLM %s loaded on request in %.1f s", grounder.model_id, grounder.load_s or 0.0)
        except _UNAVAILABLE as exc:
            _LOG.error("VLM %s could not be loaded on request: %s: %s", grounder.model_id, type(exc).__name__, exc)
        with self._lock:
            if self._grounder is grounder and grounder.loaded:
                self._requested = True
        return _status_of(grounder)

    def asker_for(self, vlm: Any, *, may_load: bool, max_new_tokens: int = 96) -> TextAsker:
        """A text asker over ``vlm``'s copy. ``may_load`` is the availability rule's answer (Q8 A).

        Builds a grounder only for a command that may load, on an empty holder; never beside a copy of other
        weights. Raises :class:`VlmCopyConflictError` (a :class:`VlmNotLoadedError`) when a copy of other weights is
        in the way (:meth:`conflict_for`), and :class:`VlmNotLoadedError` when nothing is held and the command may
        not load. Either way nothing was built, loaded or taken away.
        """
        weights = VlmWeights.of(vlm)
        with self._lock:
            conflict = self._in_the_way(weights)
            if conflict:
                raise VlmCopyConflictError(weights.model_id, conflict)
            grounder = self._grounder
            if grounder is None:
                if not may_load:
                    raise VlmNotLoadedError(weights.model_id)
                grounder = self._hold(weights)
        return TextAsker(grounder, may_load=may_load, max_new_tokens=max_new_tokens)

    def forget(self) -> None:
        """Drop what it knows, so the next call builds anew: the held copy, who asked for it, and every grounder
        it built. Unloads nothing: a cell built with a grounder keeps it, loaded or not. A test's teardown."""
        with self._lock:
            self._drop_held()
            self._issued.clear()

    # --- under the lock -------------------------------------------------------------------------------------------

    def _hold(self, weights: VlmWeights) -> Qwen3VLGrounder:
        """Build and hold a grounder for ``weights``. Under the lock; loads nothing."""
        grounder = self._build(weights)
        self._grounder, self._weights, self._requested = grounder, weights, False
        self._issued[grounder] = weights
        _LOG.info("holding VLM %s (not loaded)", weights.describe())
        return grounder

    def _drop_held(self) -> None:
        self._grounder, self._weights, self._requested = None, None, False

    def _in_the_way(self, weights: VlmWeights) -> str:
        held = self._grounder
        if held is not None and self._weights == weights and _on_the_card(held):
            return ""  # already on the card: reading it or loading it again adds nothing
        other: VlmWeights | None = None
        where = ""
        if held is not None and self._weights != weights:
            other, where = self._weights, f"held for this process ({_status_of(held).state})"
        else:
            for grounder, its in list(self._issued.items()):
                if its != weights and _on_the_card(grounder):
                    other, where = its, "still loaded in a cell built with it"
                    break
        if other is None:
            return ""
        return (f"the VLM copy {other.describe()} is {where}, and the config names {weights.describe()}. One copy "
                f"per process: neither a command nor Laden loads a second one beside it. Rebuild the cell to "
                f"switch the weights; the rebuild releases the old copy.")

    # --- outside the lock -----------------------------------------------------------------------------------------

    def _retire(self, grounder: Qwen3VLGrounder, weights: VlmWeights | None, *, why: str) -> None:
        """Unload a copy the holder no longer holds. Outside the lock: it waits for an inference or load in flight."""
        unloaded = grounder.unload()
        name = weights.describe() if weights is not None else grounder.model_id
        _LOG.info("released VLM %s (%s)%s", name, why, "; its weights left the card" if unloaded else "")


_SHARED: VlmHolder | None = None
_SHARED_LOCK: Final[threading.Lock] = threading.Lock()


def shared_vlm() -> VlmHolder:
    """The process's holder. The perception build asks it, and so does the command reader."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = VlmHolder.from_parts()
        return _SHARED
