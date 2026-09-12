"""Silero VAD as TorchScript: the voice detector behind `VoiceActivityDetector`, loaded on first use.

Silero runs as TorchScript on the torch this interpreter already has, with no new package: ONNX would
need onnxruntime, and the silero-vad package adds nothing the model file does not carry. The file is
`silero_vad.jit` from the silero-vad 6.2.1 wheel (MIT), pinned by the wheel's sha256 and by its own in
`scripts/model_weights/fetch.py`; `models.stt.vad_model_path` names where it lies.

It is called the way the package's own `VADIterator` and `get_speech_timestamps` call it
(`silero_vad/utils_vad.py`): windows of 512 samples at 16 kHz, ``model(window, 16000).item()`` per window,
and ``model.reset_states()`` before a new stretch of audio. The model carries its state from one window to
the next, so one detector serves one stream at a time: `SpeechGate` holds a lock around a whole recording,
and a `Listener` builds a detector of its own.

The probabilities the real model gives silence and a tone are in this package's `README.md`.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import UNSET, Maybe, chosen
from src.models.speech.engine import SpeechModelMissing, SpeechStackUnavailable

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = ["SILERO_SAMPLERATE", "SILERO_WINDOW", "SileroVoiceActivityDetector"]

#: The rate the detector scores at. Silero also accepts 8 kHz; this stack runs at Whisper's 16 kHz.
SILERO_SAMPLERATE: Final[int] = 16000
#: The window Silero scores at 16 kHz.
SILERO_WINDOW: Final[int] = 512


def _import_torch() -> Any:
    """torch, or the refusal that names it."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="torch", cause=exc) from exc
    return torch


class SileroVoiceActivityDetector:
    """Silero VAD through `torch.jit` on the CPU. Build it with `from_config` or `from_parts`."""

    def __init__(self, *, model_path: Path, model: Maybe[Any]) -> None:
        self._model_path = model_path
        self._model: Any = model if chosen(model) else None
        self._lock = threading.Lock()

    @classmethod
    def from_config(cls, *, config: SpeechToTextConfig) -> SileroVoiceActivityDetector:
        """The YAML door: `models.stt.vad_model_path`. Loads nothing."""
        return cls.from_parts(model_path=config.vad_model_path)

    @classmethod
    def from_parts(cls, *, model_path: str | Path, model: Maybe[Any] = UNSET) -> SileroVoiceActivityDetector:
        """The Python door. Loads nothing.

        ``model`` hands in a TorchScript module that is already loaded, and ``model_path`` then only
        names it.
        """
        return cls(model_path=Path(model_path), model=model)

    @property
    def name(self) -> str:
        """The detector's name as a `SpeechCheck` reports it."""
        return "silero-vad"

    @property
    def model_path(self) -> Path:
        """The TorchScript file this detector loads."""
        return self._model_path

    @property
    def samplerate(self) -> int:
        """The only rate the detector accepts."""
        return SILERO_SAMPLERATE

    @property
    def window_samples(self) -> int:
        """The only window length the detector accepts."""
        return SILERO_WINDOW

    def load(self) -> None:
        """Load the TorchScript file on the CPU now, rather than on the first window. Loads once."""
        with self._lock:
            if self._model is not None:
                return
            torch = _import_torch()
            if not self._model_path.is_file():
                raise SpeechModelMissing(
                    key="models.stt.vad_model_path", path=str(self._model_path), kind="file",
                    fetch="fetch it with `python scripts/model_weights/fetch.py silero-vad`",
                )
            model = torch.jit.load(str(self._model_path), map_location="cpu")
            model.eval()
            self._model = model

    def reset(self) -> None:
        """Forget the state carried between windows, before a new stretch of audio. Loads first."""
        self.load()
        self._model.reset_states()

    def speech_probability(self, window: np.ndarray) -> float:
        """A probability in [0, 1] that ``window``, mono float32 of 512 samples at 16 kHz, is speech."""
        import numpy as np

        samples = np.asarray(window, dtype=np.float32)
        if samples.shape != (SILERO_WINDOW,):
            raise ValueError(
                f"Silero scores mono windows of exactly {SILERO_WINDOW} samples at {SILERO_SAMPLERATE} Hz, "
                f"not an array of shape {samples.shape}."
            )
        self.load()
        torch = _import_torch()
        with torch.inference_mode():
            return float(self._model(torch.tensor(samples), SILERO_SAMPLERATE).item())
