"""Shared inference-time helpers for torch-based model wrappers, and how each model loads.

The CUDA-versus-CPU handling lives here, so a wrapper carries only the logic of its own model.

How a model loads is fixed per model (the owner, 2026-10-17), and :data:`RTDETR`, :data:`SAM2` and :data:`WHISPER`
are the one place it is written: the values the owner's cell ran (its tree of 2026-10-14), which no cell had a reason
to change. The spread a cell could choose from was narrow: the dtype and the attention follow from the model and the
card, ``flash_attention_2`` would need a package this repository does not install, and ``torch.compile`` costs
warm-up time and is fragile on Windows, so every cell ran it off. The VLM picks its own dtype from its checkpoint
(``src/models/vlm/qwen.py``: ``dtype="auto"``).

``build_load_kwargs`` produces the kwargs for ``from_pretrained``, adding ``torch_dtype`` and
``attn_implementation`` as a model's settings ask. ``finalize_model`` applies the post-load steps, ``.eval()`` and
channels-last. ``weight_load_errors`` names the exception types an unusable set of weights actually arrives as, for
the ``except`` clause that says which model failed. ``autocast_ctx`` is a no-op context on CPU and ``torch.autocast``
on CUDA.

torch is imported inside the helpers, never at import, so the settings above are read without it (the console's
cell facts name the detector's precision before a model loads).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import torch

_log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class LoadSettings:
    """How one model loads and runs: its weights' dtype, its attention, and its memory format.

    Attributes:
        torch_dtype (str | None): ``"auto"`` loads fp16 weights on CUDA and fp32 elsewhere; ``None`` keeps the
            checkpoint's fp32 weights and runs the pass under CUDA's fp16 autocast, the path that holds recall on
            small objects.
        attn_implementation (str | None): Forwarded to ``from_pretrained``; ``None`` keeps transformers' default.
        channels_last (bool): Channels-last memory format for a vision model on CUDA, its weights and its input.
    """

    torch_dtype: str | None
    attn_implementation: str | None
    channels_last: bool


#: RT-DETR, the closed-set detector: fp16 weights on CUDA, the eager attention, channels last.
RTDETR: Final = LoadSettings(torch_dtype="auto", attn_implementation="eager", channels_last=True)
#: SAM2, the segmenter: fp32 weights under fp16 autocast, which keeps small parts whole, sdpa, channels last.
SAM2: Final = LoadSettings(torch_dtype=None, attn_implementation="sdpa", channels_last=True)
#: Whisper: fp16 weights on CUDA, sdpa; an audio model, so no channels-last.
WHISPER: Final = LoadSettings(torch_dtype="auto", attn_implementation="sdpa", channels_last=False)


def build_load_kwargs(
    settings: LoadSettings | None,
    device: torch.device,
    base_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return ``from_pretrained`` kwargs honouring ``settings``; ``None`` is transformers' own defaults."""
    import torch  # noqa: PLC0415 (loaded with the model, never at import)

    from src.utility.device import resolve_torch_dtype  # noqa: PLC0415 (torch)

    kwargs: dict[str, Any] = dict(base_kwargs or {})
    if settings is None:
        return kwargs

    # A non-default torch_dtype only where the settings name one ("auto" or a concrete dtype). Left unset, the HF
    # default of fp32 weights applies, and the pass runs under the CUDA autocast (`autocast_ctx`).
    if settings.torch_dtype is not None:
        resolved = resolve_torch_dtype(settings.torch_dtype, device)
        if resolved == torch.float16:
            # Said once per load: fp16 weights are what the settings ask, and fp16 is where small-object recall
            # is lost first, so the line names it beside the model's own load line.
            _log.info("Loading model under float16 (torch_dtype=%r on %s).", settings.torch_dtype, device.type)
        kwargs["torch_dtype"] = resolved

    if settings.attn_implementation:
        kwargs["attn_implementation"] = settings.attn_implementation

    return kwargs


def finalize_model(
    model: torch.nn.Module,
    device: torch.device,
    settings: LoadSettings | None,
    vision: bool = False,
) -> torch.nn.Module:
    """Apply the post-load steps and return the model: ``.eval()``, and channels-last where ``settings`` ask."""
    import torch  # noqa: PLC0415 (loaded with the model, never at import)

    model.eval()

    if settings is None:
        return model

    if vision and settings.channels_last and device.type == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)  # type: ignore[call-overload]  # torch stub lacks the memory_format overload
        except (RuntimeError, ValueError) as exc:
            # Not every submodule supports channels_last; skip it and log.
            _log.warning("channels_last conversion skipped: %s", exc)

    return model


@cache
def weight_load_errors() -> tuple[type[BaseException], ...]:
    """The exception types an unusable set of weights actually arrives as.

    It is for the ``except`` clause around a ``from_pretrained`` pair whose only job is to
    name which model failed before re-raising. That clause caught ``RuntimeError``, and
    measured 2026-09-10 on this box, with transformers 5.5.4, torch 2.7.1+cu128 and
    safetensors 0.8.0, driving the real loaders against three throwaway model directories,
    not one of the three ways weights are unusable is a RuntimeError:

        a directory holding only a README   ValueError (GroundingDINO), OSError (SAM2)
        a half-downloaded snapshot          OSError, both
        a corrupted model.safetensors       safetensors.SafetensorError,
                                            "Error while deserializing header: header too large"

    So the one log line connecting a transformers traceback to the config key that pointed
    there was dead in exactly the situations it was written for. ``RuntimeError`` stays in
    the set anyway, because a CUDA OOM on the trailing ``.to(device)`` is a load failure
    worth naming too.

    The set is named rather than ``Exception`` on purpose. ``except Exception`` would also
    catch a defect in our own code and report it to the operator as unusable weights, which
    is a worse lie than the silence it replaces, and a defect has to come through
    unlabelled.

    ``SafetensorError`` lives in a compiled extension, ``safetensors_rust``, and is imported
    here rather than at module scope, the way this repository treats heavy vendor libraries.
    It costs nothing at the moment it matters: an ``except`` clause is evaluated only while
    an exception is already propagating out of ``from_pretrained``, by which point
    transformers has imported safetensors itself and this is a ``sys.modules`` lookup. The
    result is cached, so a load that never fails never pays for it either.
    """
    errors: tuple[type[BaseException], ...] = (OSError, ValueError, RuntimeError)
    try:
        import safetensors
    except ImportError:  # pragma: no cover (safetensors ships with transformers)
        # A missing safetensors cannot produce a SafetensorError, so the shorter set is complete.
        return errors
    return (*errors, safetensors.SafetensorError)


def autocast_ctx(device: torch.device, dtype: torch.dtype | None = None):
    """Return an autocast context, or a no-op context on non-CUDA devices."""
    if device.type != "cuda":
        return contextlib.nullcontext()
    import torch  # noqa: PLC0415 (loaded with the model, never at import)

    return torch.autocast(device_type="cuda", dtype=dtype)


__all__ = [
    "RTDETR",
    "SAM2",
    "WHISPER",
    "LoadSettings",
    "autocast_ctx",
    "build_load_kwargs",
    "finalize_model",
    "weight_load_errors",
]
