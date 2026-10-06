"""Qwen3-VL as a detector: ``detect_all(bgr, prompt) -> [Detection]``, the shape GroundingDINO has.

A grounding VLM produces boxes, and boxes are what
:class:`~src.models.perception_backend.TwoStageBackend` already hands to SAM2. So this is a drop-in
replacement for the detector stage rather than a second kind of pipeline, and the two-stage backend
composes it with the existing segmenter unchanged.

The same loaded model also answers text: :meth:`Qwen3VLGrounder.answer_text` asks it a question
with no image, which is how the console reads an operator's command
(:mod:`src.models.vlm.command`). One copy serves both, held once per process
(:mod:`src.models.vlm.holder`), so the load is guarded by a lock (two callers load it once, and a
caller that waited for a load that failed does not try again on its heels) and the two uses take
turns on one generate lock: the model runs one inference at a time. :meth:`Qwen3VLGrounder.unload`
hands the memory back when the holder retires the copy.

No torch, transformers or weights are imported at module import. Everything heavy is loaded inside
:meth:`Qwen3VLGrounder._load`, so this module imports on CI, on macOS and on a box with no
GPU, and a cell that never sends a complex prompt never pays for the model.

Measured against the real 4B weights on the dev box's RTX 5080 (2026-10-01): the load takes about
6.3-6.5 s and holds 8.93 GB, a command's text answer peaks at 9.8-9.9 GB (a retried question is the
longest) and takes about 2.2 s, an unload hands all of it back (0.01 GB still allocated), and the
grounding tests pass (`tests/test_vlm_inference.py`, `tests/test_vlm_command_inference.py`). The
grounding quality on a real cell's scenes, the VRAM on any other card and the choice between the 4B
and 8B checkpoints are unmeasured, so the config default has to follow a measurement rather than the
other way round.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from src.models.constants import MODELS_LOG_DIR, VLM_GROUNDER_LOG_FILE
from src.models.detection.types import Detection
from src.utility.log_cfg import create_logger

from .parsing import CoordinateSpace, parse_grounding_response

__all__ = ["Qwen3VLGrounder", "GROUNDING_INSTRUCTION", "GROUNDING_MAX_NEW_TOKENS"]

#: A file sink: nothing in this repo configures the root logger, so a bare ``getLogger`` would
#: format the lines below and then discard them. The name stays ``__name__``, matching its siblings
#: in this package rather than the class-named loggers elsewhere in ``models``. ``create_logger`` is
#: stdlib-only, so the no-torch-at-import promise above holds.
_LOG = create_logger(__name__, log_file=VLM_GROUNDER_LOG_FILE, log_dir=MODELS_LOG_DIR)

#: How many tokens a grounding answer may run to. Qwen3-VL spends about 33 tokens on a box written compactly and
#: 47 on one fenced and indented, as it often answers: 512 tokens, the earlier cap, held 10 to 15 boxes, and a pile of
#: more parts on the mat came back cut off mid-list, which no JSON reading takes (measured on the checkpoint's own
#: tokenizer, 2026-10-06). 2048 holds about 40; greedy decoding stops at the answer's end, so a short answer costs
#: no more than before.
GROUNDING_MAX_NEW_TOKENS = 2048

#: The instruction wrapped around every operator prompt.
#:
#: It asks for Qwen's documented ``bbox_2d`` grounding format and for an empty list when the object
#: is absent. Instruct-tuned models are agreeable by default: without that sentence they invent a
#: plausible box rather than return nothing, and an invented box becomes a grasp at the wrong place.
GROUNDING_INSTRUCTION = (
    "Locate every object matching this description and return ONLY a JSON array, no prose:\n"
    '[{{"bbox_2d": [x0, y0, x1, y1], "label": "<short name>"}}]\n'
    "Coordinates are absolute pixels in this image. "
    "If no object matches, return an empty array []. Do not guess.\n\n"
    "Description: {prompt}"
)


def _hand_back_memory(torch: Any) -> None:
    """Collect what a dropped model left behind and empty torch's CUDA cache, so the card itself has
    the memory back: the caching allocator would otherwise keep it reserved for this process, where
    the cuRobo sidecar or the next checkpoint cannot use it. Best effort: a failure is logged."""
    import gc  # noqa: PLC0415

    gc.collect()
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if not callable(empty_cache):
        return
    try:
        empty_cache()
    except Exception as exc:  # noqa: BLE001 - emptying a cache must not fail the release it follows
        _LOG.warning("emptying the CUDA cache after an unload failed: %s: %s", type(exc).__name__, exc)


class Qwen3VLGrounder:
    """Ground a free-form prompt to boxes with a Qwen3-VL-Instruct checkpoint.

    Implements the ``detect_all`` detector contract, so ``TwoStageBackend`` takes it wherever a
    phrase detector goes.
    """

    def __init__(
        self,
        *,
        model_id: str,
        model_path: str | None = None,
        local: bool = False,
        device: str = "cuda",
        max_new_tokens: int = GROUNDING_MAX_NEW_TOKENS,
        preload: bool = False,
        coordinate_space: CoordinateSpace = CoordinateSpace.GRID_1000,
    ) -> None:
        #: What this checkpoint's box numbers mean, measured for Qwen3-VL-4B-Instruct. A parameter
        #: because a future checkpoint may differ and the failure is silent: grid values look like
        #: plausible pixels on a large frame. See :class:`CoordinateSpace`.
        self.coordinate_space = coordinate_space
        self.model_id = model_id
        self._source = model_path or model_id
        self._local = local
        self._device = device
        self._max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None
        self._torch: Any = None
        #: Taken around the load: two callers that both need the weights (a pick's look and a
        #: command being read, or two requests) load them once.
        self._load_lock = threading.Lock()
        #: One inference at a time on the one model. Grounding a frame and answering a command take
        #: turns, and the span covers tokenising and decoding too: a fast tokenizer shared between
        #: threads raises "Already borrowed".
        self._generate_lock = threading.Lock()
        self._loading = False
        self._load_error: BaseException | None = None
        #: Load attempts that have ended, well or not. A caller that waited on the load lock compares
        #: it with what it read before waiting, so it knows the load it waited for has been tried.
        self._loads_ended = 0
        self._load_s: float | None = None
        self._last_answer_ms: float | None = None
        if preload:
            # Predictable latency and VRAM held from cell build, against nothing paid until the
            # first complex prompt arrives. Config chooses; both suit different cells.
            self._ensure_loaded()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def loading(self) -> bool:
        """True while a load is in progress on some thread."""
        return self._loading

    @property
    def load_error(self) -> BaseException | None:
        """What the last load raised, until a load succeeds; ``None`` otherwise."""
        return self._load_error

    @property
    def load_s(self) -> float | None:
        """How long the successful load took, in seconds; ``None`` before one."""
        return self._load_s

    @property
    def last_answer_ms(self) -> float | None:
        """How long the latest :meth:`answer_text` took, in milliseconds; ``None`` before one."""
        return self._last_answer_ms

    def ensure_loaded(self) -> bool:
        """Load the weights unless they are loaded. ``True`` when this call loaded them.

        Raises what the load raised, unchanged, and records it in :attr:`load_error`; the next call
        tries again, except one that was already waiting for that load (see :meth:`_ensure_loaded`).
        Deciding what a failed load means is the caller's: ``GuardedVlmBackend`` for a pick, the
        holder's text asker for a command.
        """
        return self._ensure_loaded()

    def _ensure_loaded(self) -> bool:
        """Load weights on first use, once however many threads ask at the same time.

        Double-checked: the unlocked read is the fast path of every inference after the first, and
        the second read under the lock is what makes a caller that waited for another's load find
        the weights there instead of loading them again. A caller that waited for a load that
        FAILED takes that failure as its own answer rather than trying again on its heels: a second
        multi-GB attempt right after an out-of-memory or a missing file only repeats it, and a
        command that arrived during a failing Laden must not become the retry. The next caller that
        was not waiting tries again (a pick's next look, Laden). Raises the underlying error
        unchanged; ``GuardedVlmBackend`` decides what it means.
        """
        if self._model is not None:
            return False
        ended = self._loads_ended
        with self._load_lock:
            if self._model is not None:
                return False
            failure = self._load_error
            if failure is not None and self._loads_ended != ended:
                # Re-raised as it is, from every caller that waited for it, as a Future's result is.
                raise failure
            self._loading = True
            started = time.perf_counter()
            try:
                processor, model, torch = self._load()
            except BaseException as exc:
                self._load_error = exc
                _LOG.error("VLM %s failed to load: %s: %s", self.model_id, type(exc).__name__, exc)
                raise
            finally:
                self._loading = False
                self._loads_ended += 1
            self._processor = processor
            self._torch = torch
            self._load_error = None
            self._load_s = time.perf_counter() - started
            # Published last: `loaded` reads True only once the processor and torch are in place, so
            # the unlocked fast path above never meets a half-loaded model.
            self._model = model
            # The load cost is what `preload` trades against first-prompt latency.
            _LOG.info("VLM %s ready in %.1f s", self.model_id, self._load_s)
            return True

    def _load(self) -> tuple[Any, Any, Any]:
        """The heavy part: ``(processor, model on its device, torch)``. Called under the load lock only."""
        # Imported here, not at module scope: this module must import with no torch and no GPU.
        import torch  # noqa: PLC0415
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415

        _LOG.info("loading VLM %s (device=%s)", self._source, self._device)
        processor = AutoProcessor.from_pretrained(self._source, local_files_only=self._local)
        # Not `device_map=`: that routes through `accelerate`, which is absent from the validated
        # Isaac environment and fails at load with an error naming accelerate rather than the real
        # situation. `.to(device)` needs no extra dependency and is the right call for a single GPU;
        # device_map earns its keep only for multi-GPU sharding or CPU offload.
        # `model` is typed Any because transformers annotates `.to()` as taking a PreTrainedModel
        # rather than a device string, which mypy then rejects.
        model: Any = AutoModelForImageTextToText.from_pretrained(
            self._source,
            local_files_only=self._local,
            dtype="auto",
        )
        model = model.to(self._device)
        model.eval()
        return processor, model, torch

    def unload(self) -> bool:
        """Drop the loaded weights, so the card gets their memory back. ``True`` when there were some.

        Waits for a load or an inference in flight, so nothing loses its model halfway; an inference
        that starts afterwards loads the weights again (one that may not load refuses instead). The
        holder unloads a copy when it retires it: a rebuild that names other weights, or the
        console's release of a copy nobody asked for (:mod:`src.models.vlm.holder`).
        """
        with self._load_lock, self._generate_lock:
            model, torch = self._model, self._torch
            if model is None:
                return False
            # `loaded` reads False from here on, and an inference that takes the generate lock next
            # finds no model rather than half of one.
            self._model = None
            self._processor = None
            self._torch = None
            self._load_s = None
            self._last_answer_ms = None
        del model
        _hand_back_memory(torch)
        _LOG.info("VLM %s unloaded", self.model_id)
        return True

    def _in_hand(self) -> tuple[Any, Any, Any]:
        """``(processor, model, torch)``, read under the generate lock; refused when the weights are gone."""
        model = self._model
        if model is None:
            raise RuntimeError(
                f"the VLM {self.model_id!r} is not loaded: its weights were not there to answer with "
                f"(never loaded, or released since)"
            )
        return self._processor, model, self._torch

    def _generate(self, image_rgb: Any, prompt: str) -> str:
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_rgb},
                {"type": "text", "text": GROUNDING_INSTRUCTION.format(prompt=prompt)},
            ],
        }]
        with self._generate_lock:
            processor, model, torch = self._in_hand()
            inputs = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)

            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=self._max_new_tokens, do_sample=False,
                )
            # Strip the prompt tokens: decoding the whole sequence would feed the instruction, which
            # contains a bbox_2d example, back into the parser and yield a fabricated detection.
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            return str(processor.batch_decode(trimmed, skip_special_tokens=True)[0])

    def answer_text(
        self, system: str, user: str, *, prefill: str = "{", max_new_tokens: int = 96,
        load: bool = True,
    ) -> str:
        """Ask the loaded model a text-only question: no image, greedy, at most ``max_new_tokens``.

        Returns ``prefill`` followed by what the model wrote. With a prefill, the answer is a
        continuation of an assistant turn that already starts with it (``continue_final_message``),
        which is how a JSON answer is made to open with ``{`` rather than with prose. Without one,
        the model starts a new turn. Loads the weights first if needed, and raises what the load
        raised; with ``load=False`` it never loads, and weights that are not there raise
        :class:`RuntimeError` (the holder's text asker decides loading itself).

        Every message is a list of typed text parts: the processor's tokenising path iterates the
        content looking for images, and a bare string is not iterable that way.
        """
        if max_new_tokens < 1:
            raise ValueError(
                f"an answer needs room for at least one token, not max_new_tokens={max_new_tokens}"
            )
        if load:
            self._ensure_loaded()
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        if prefill:
            messages.append({"role": "assistant", "content": [{"type": "text", "text": prefill}]})
        started = time.perf_counter()
        with self._generate_lock:
            processor, model, torch = self._in_hand()
            inputs = processor.apply_chat_template(
                messages, tokenize=True,
                add_generation_prompt=not prefill, continue_final_message=bool(prefill),
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                )
            # Stripped as in `_generate`: the instruction carries example answers, and decoding the
            # prompt back would hand one of them over as the model's own.
            trimmed = generated[:, inputs["input_ids"].shape[1]:]
            text = str(processor.batch_decode(trimmed, skip_special_tokens=True)[0])
            # Under the lock, so an unload that follows clears it rather than being followed by it.
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self._last_answer_ms = elapsed_ms
        _LOG.info("VLM %s answered %d character(s) of text in %.0f ms",
                  self.model_id, len(text), elapsed_ms)
        return prefill + text

    def detect_all(self, image_bgr: Any, prompt: str) -> list[Detection]:
        """Ground ``prompt``; ``[]`` when nothing matches. Same contract as the phrase detectors."""
        self._ensure_loaded()
        import numpy as np  # noqa: PLC0415

        array = np.asarray(image_bgr)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"expected an HxWx3 BGR image, got shape {array.shape}")
        height, width = int(array.shape[0]), int(array.shape[1])
        # The processor expects RGB; every caller here speaks BGR. `ascontiguousarray` is required:
        # a bare `[..., ::-1]` is a negative-stride view, and torch refuses those with "At least one
        # stride in the given numpy array is negative", raised deep inside the image processor and
        # several frames away from anything naming this file.
        image_rgb = np.ascontiguousarray(array[..., ::-1])

        # Timed around generate and parse together, because that span is what a caller waits for.
        started = time.perf_counter()
        answer = self._generate(image_rgb, prompt)
        detections = parse_grounding_response(
            answer, image_width=width, image_height=height, fallback_label=prompt,
            space=self.coordinate_space,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if not detections:
            # "The model answered and nothing survived parsing" and "the model said the object is
            # absent" look identical downstream, and only this line distinguishes them.
            _LOG.info(
                "VLM grounded nothing for %r in %.0f ms (raw answer: %.200s)",
                prompt, elapsed_ms, answer,
            )
        else:
            # The other half of the same event, so exactly one line per inference either way.
            _LOG.info(
                "VLM grounded %d box(es) for %r in %.0f ms", len(detections), prompt, elapsed_ms,
            )
        return detections
