"""Stand-ins for Whisper, Silero, the microphone, the voice detector and the engine, so speech tests need
no weights, no sound card and no PortAudio.

The token ids are the ones openai/whisper-small's generation config uses; large-v3-turbo keeps the same
ids for the start token and for German, English and French, and moves the task tokens up by one. The
fake model answers the three calls the engine makes: the encoder once, one decoder step whose logits
score the languages, and `generate`, whose sequence starts with the decoder prompt so the language token
sits at index 1 (`transformers/models/whisper/generation_whisper.py:913-934`).
"""

from __future__ import annotations

import types
from collections.abc import Iterable
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np

START = 50258
END = 50257
ENGLISH = 50259
GERMAN = 50261
FRENCH = 50265
TRANSCRIBE = 50359
NO_TIMESTAMPS = 50363
VOCABULARY = 51865

SAMPLERATE = 16000


def whisper_parts(
    *,
    text: str = "  Pick the red cube  ",
    language_token: int = GERMAN,
    language_scores: dict[int, float] | None = None,
) -> tuple[MagicMock, MagicMock]:
    """A processor and a model that decode every recording to ``text``.

    ``language_scores`` are the decoder's logits at the first step, by token id; left out, the language
    the prompt carries scores 5 and every other token 0. ``language_token`` is the language token the
    returned prompt carries.
    """
    import torch

    processor = MagicMock(name="WhisperProcessor")
    processor.return_value = SimpleNamespace(input_features=torch.zeros(1, 80, 3000))
    processor.batch_decode.return_value = [text]

    model = MagicMock(name="WhisperForConditionalGeneration")
    model.to.return_value = model
    model.generation_config = SimpleNamespace(
        lang_to_id={"<|en|>": ENGLISH, "<|de|>": GERMAN},
        language=None,
        task=None,
        forced_decoder_ids=[[1, None], [2, TRANSCRIBE]],
        decoder_start_token_id=START,
    )
    model.get_encoder.return_value = MagicMock(
        name="WhisperEncoder", return_value=SimpleNamespace(last_hidden_state=torch.zeros(1, 1500, 8))
    )
    logits = torch.zeros(1, 1, VOCABULARY)
    for token, score in (language_scores or {language_token: 5.0}).items():
        logits[0, 0, token] = score
    model.return_value = SimpleNamespace(logits=logits)
    model.generate.return_value = SimpleNamespace(
        sequences=torch.tensor([[START, language_token, TRANSCRIBE, NO_TIMESTAMPS, 11, END]])
    )
    return processor, model


class FakeSileroModel:
    """What `torch.jit.load` returns for Silero, as a test double.

    A window is speech when its peak reaches 0.1; ``speech`` True or False makes every window speech or
    none. ``calls`` records the shape and rate of every window scored.
    """

    def __init__(self, *, speech: bool | None = None) -> None:
        self.speech = speech
        self.resets = 0
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def eval(self) -> FakeSileroModel:
        return self

    def reset_states(self) -> None:
        self.resets += 1

    def __call__(self, window: Any, samplerate: int) -> Any:
        import torch

        self.calls.append((tuple(window.shape), samplerate))
        if self.speech is None:
            probability = 1.0 if float(window.abs().max()) >= 0.1 else 0.0
        else:
            probability = 1.0 if self.speech else 0.0
        return torch.tensor([[probability]])


# Audio.


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * SAMPLERATE)), dtype=np.float32)


def tone(seconds: float, amplitude: float = 0.5, frequency: float = 220.0) -> np.ndarray:
    count = int(round(seconds * SAMPLERATE))
    return (amplitude * np.sin(np.arange(count) * 2 * np.pi * frequency / SAMPLERATE)).astype(np.float32)


def blocks_of(*segments: np.ndarray, block_s: float = 0.1) -> list[np.ndarray]:
    """The segments joined and cut into blocks the size a sound card hands over."""
    joined = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    size = int(round(block_s * SAMPLERATE))
    return [joined[i:i + size] for i in range(0, joined.size, size)]


class FakeClock:
    """A monotonic clock that only moves when a fake source says audio time has passed."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class ScriptedSource:
    """An `AudioSource` that plays back blocks and advances a fake clock by each block's length.

    ``endless`` keeps delivering ``endless`` (a block) after the script runs out; ``silent_after``
    makes `read` return nothing, advancing the clock by the timeout it was given, without ever ending.
    """

    def __init__(
        self,
        blocks: Iterable[np.ndarray],
        *,
        clock: FakeClock,
        endless: np.ndarray | None = None,
        silent_after: bool = False,
        overflow_at: frozenset[int] = frozenset(),
        samplerate: int = SAMPLERATE,
    ) -> None:
        self.remaining = list(blocks)
        self._clock = clock
        self._endless = endless
        self._silent_after = silent_after
        self._overflow_at = overflow_at
        self._samplerate = samplerate
        self._served = 0
        self.started = 0
        self.stopped = 0

    @property
    def samplerate(self) -> int:
        return self._samplerate

    @property
    def ended(self) -> bool:
        return not self.remaining and self._endless is None and not self._silent_after

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def discard_buffered(self) -> int:
        return 0

    def read(self, *, timeout_s: float) -> Any:
        from src.models.speech.capture import AudioBlock

        if self.remaining:
            block = self.remaining.pop(0)
        elif self._endless is not None:
            block = self._endless
        else:
            self._clock.now += timeout_s
            return None
        self._clock.now += block.size / self._samplerate
        overflowed = self._served in self._overflow_at
        self._served += 1
        return AudioBlock(samples=block, overflowed=overflowed)


class LoudnessDetector:
    """A test double for `VoiceActivityDetector`: a window is speech when its peak reaches 0.1.

    Not a product detector; `SileroVoiceActivityDetector` is. This one lets the endpointing, the gate
    and the listener be tested with audio whose speech is known. ``windows`` records every window's
    length."""

    name = "loudness"
    samplerate = SAMPLERATE
    window_samples = 512

    def __init__(self) -> None:
        self.resets = 0
        self.windows: list[int] = []

    def reset(self) -> None:
        self.resets += 1

    def speech_probability(self, window: np.ndarray) -> float:
        self.windows.append(int(window.shape[0]))
        return 1.0 if float(np.max(np.abs(window))) >= 0.1 else 0.0


class RecordingEngine:
    """A `SpeechEngine` that records what it was asked to transcribe and answers with fixed text."""

    name = "recording"

    def __init__(self, text: str = "Nimm den roten Würfel", language: str = "de") -> None:
        self.text = text
        self.language = language
        self.calls: list[tuple[np.ndarray, int]] = []

    def load(self) -> None:
        pass

    def transcribe(self, samples: np.ndarray, *, samplerate: int) -> Any:
        from src.models.speech.transcript import LanguageSource, Transcript

        self.calls.append((np.array(samples, copy=True), samplerate))
        return Transcript(
            text=self.text, language=self.language, language_source=LanguageSource.DETECTED,
            duration_s=samples.shape[0] / samplerate, engine=self.name, model="none", device="cpu",
            latency_ms=1.0,
        )


# sounddevice.


def fake_sounddevice(*, hostapi: str = "MME", default_samplerate: float = 48000.0) -> types.ModuleType:
    """A stand-in `sounddevice` module. It has an `InputStream` and deliberately no `RawInputStream`.

    ``module.opened`` lists the streams opened, and ``stream.feed(block)`` calls the stream's callback
    the way PortAudio would, with a (frames, channels) array and a status carrying ``input_overflow``.
    """
    module = types.ModuleType("sounddevice")
    opened: list[Any] = []

    class PortAudioError(Exception):
        pass

    class InputStream:
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs
            self.active = False
            self.closed = False
            opened.append(self)

        def start(self) -> None:
            self.active = True

        def stop(self) -> None:
            self.active = False

        def close(self) -> None:
            self.closed = True

        def feed(self, block: np.ndarray, *, overflow: bool = False) -> None:
            status = SimpleNamespace(input_overflow=overflow)
            self.kwargs["callback"](block, block.shape[0], None, status)

    class WasapiSettings:
        def __init__(
            self, exclusive: bool = False, auto_convert: bool = False,
            explicit_sample_format: bool = False,
        ) -> None:
            self.auto_convert = auto_convert

    def query_devices(device: Any = None, kind: Any = None) -> dict[str, Any]:
        return {"name": "fake microphone", "hostapi": 0, "default_samplerate": default_samplerate,
                "max_input_channels": 2}

    def query_hostapis(index: Any = None) -> dict[str, Any]:
        return {"name": hostapi}

    module.InputStream = InputStream  # type: ignore[attr-defined]
    module.PortAudioError = PortAudioError  # type: ignore[attr-defined]
    module.WasapiSettings = WasapiSettings  # type: ignore[attr-defined]
    module.query_devices = query_devices  # type: ignore[attr-defined]
    module.query_hostapis = query_hostapis  # type: ignore[attr-defined]
    module.opened = opened  # type: ignore[attr-defined]
    return module
