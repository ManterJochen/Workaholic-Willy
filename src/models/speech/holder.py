"""The process's one speech engine and voice gate, keyed on the `models.stt` section.

`SpeechHolder` builds a speech engine and a `SpeechGate` for a section, hands the same pair back while
the section is unchanged, and builds a new pair when it changes, so an edited config takes effect without
a restart. Neither model loads until it is first used. The slot lives in the library rather than inside
the console's router, so a cell verb reaches the same pair; loading Whisper from disk per request would
put a whole model load inside the 2 s budget from the end of speaking.

`HeldSpeech` is what the holder hands back, and `propose()` is the upload path's verb: refuse a recording
longer than Whisper's window, ask the gate whether it holds speech, and only then ask the engine what was
said. A `Listener` caller takes `held.engine`, and builds its own voice detector: a detector carries state
from one window to the next, and the gate's belongs to uploads.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Final

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import UNSET, Maybe, resolve
from src.models.speech.engine import WHISPER_WINDOW_S, RecordingTooLong, SpeechEngine
from src.models.speech.gate import SpeechGate
from src.models.speech.transcript import Proposal

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = ["EngineBuilder", "GateBuilder", "HeldSpeech", "SpeechHolder", "shared_speech"]

#: Builds the engine for a section.
EngineBuilder = Callable[[SpeechToTextConfig], SpeechEngine]
#: Builds the gate for a section.
GateBuilder = Callable[[SpeechToTextConfig], SpeechGate]


@dataclass(frozen=True, slots=True)
class HeldSpeech:
    """One section's engine and gate, as the holder keeps them."""

    config: SpeechToTextConfig
    engine: SpeechEngine
    gate: SpeechGate

    def propose(self, samples: np.ndarray, *, samplerate: int) -> Proposal:
        """What one recording proposes for the prompt box. Loads each model on its first use.

        Raises `RecordingTooLong` for a recording longer than Whisper's window, before either model runs.
        """
        import numpy as np

        if samplerate <= 0:
            raise ValueError(f"a recording needs a positive sample rate, not {samplerate} Hz.")
        array = np.asarray(samples)
        frames = int(array.shape[0]) if array.ndim else 0
        if frames == 0:
            raise ValueError("there is nothing to transcribe: the recording holds zero frames.")
        duration_s = frames / float(samplerate)
        if duration_s > WHISPER_WINDOW_S:
            raise RecordingTooLong(duration_s=duration_s)
        speech = self.gate.check(array, samplerate=samplerate)
        if not speech.heard_speech:
            return Proposal(
                text="",
                reason=(
                    f"no speech was heard in {speech.duration_s:.2f} s of audio (peak speech probability "
                    f"{speech.peak_probability:.2f}; an utterance needs a window at {speech.onset:.2f} or "
                    f"above and {speech.min_speech_s:.2f} s of speech), so Whisper was not asked."
                ),
                speech=speech,
                transcript=None,
            )
        transcript = self.engine.transcribe(array, samplerate=samplerate)
        reason = None if transcript.text else "the voice detector heard speech, and Whisper decoded no words from it."
        return Proposal(text=transcript.text, reason=reason, speech=speech, transcript=transcript)


def _whisper(config: SpeechToTextConfig) -> SpeechEngine:
    from src.models.speech.whisper_transformers import WhisperTransformersEngine

    return WhisperTransformersEngine.from_config(config=config)


def _silero_gate(config: SpeechToTextConfig) -> SpeechGate:
    return SpeechGate.from_config(config=config)


class SpeechHolder:
    """One engine and one gate per `models.stt` section. Build it with `from_parts`; `shared_speech()`
    returns the process's."""

    def __init__(self, *, build_engine: EngineBuilder, build_gate: GateBuilder) -> None:
        self._build_engine = build_engine
        self._build_gate = build_gate
        self._lock = threading.Lock()
        self._held: HeldSpeech | None = None

    @classmethod
    def from_parts(
        cls, *, build_engine: Maybe[EngineBuilder] = UNSET, build_gate: Maybe[GateBuilder] = UNSET
    ) -> SpeechHolder:
        """The Python door. Builds nothing yet.

        ``build_engine`` UNSET is `WhisperTransformersEngine.from_config`; ``build_gate`` UNSET is
        `SpeechGate.from_config`, which scores with Silero.
        """
        return cls(
            build_engine=resolve("build_engine", build_engine, _whisper),
            build_gate=resolve("build_gate", build_gate, _silero_gate),
        )

    def for_config(self, *, config: SpeechToTextConfig) -> HeldSpeech:
        """The engine and gate for ``config``: the pair already built while the section is unchanged."""
        with self._lock:
            held = self._held
            if held is None or held.config != config:
                held = HeldSpeech(config=config, engine=self._build_engine(config), gate=self._build_gate(config))
                self._held = held
            return held

    def for_tree(
        self, data_dir: str | Path | None = None, *, profile: Maybe[str | None] = UNSET
    ) -> HeldSpeech:
        """`models.stt` read alone from ``data_dir`` under ``profile`` (`load_speech_section`), then
        `for_config`. A camera or detector file that does not load cannot refuse a recording."""
        from src.config.loader import load_speech_section

        return self.for_config(config=load_speech_section(data_dir, profile=profile))

    def forget(self) -> None:
        """Drop what it holds, so the next call builds a new engine and gate."""
        with self._lock:
            self._held = None


_SHARED: SpeechHolder | None = None
_SHARED_LOCK: Final[threading.Lock] = threading.Lock()


def shared_speech() -> SpeechHolder:
    """The process's holder. The console's upload path asks it, and so can a `Listener` caller."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = SpeechHolder.from_parts()
        return _SHARED
