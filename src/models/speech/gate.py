"""Whether a recording holds speech, asked before Whisper is asked what was said.

Whisper answers a recording without speech with a word: whisper-small answers 0.2 s of a tone with
"you", and that word would land in the prompt box. `SpeechGate` runs
a `VoiceActivityDetector` over the recording window by window, through the same `UtteranceCutter` the
microphone path cuts utterances with, and stops at the first utterance that closes. A recording in which
none closes holds no speech, and `HeldSpeech.propose` then proposes nothing and says why.

The rule is the microphone path's rule on purpose: an upload the gate lets through is one `listen()`
would have heard, and every `Endpointing` default is the same placeholder until it is measured on
recorded cell audio.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import UNSET, Maybe, resolve
from src.models.speech.capture import to_mono_at_rate
from src.models.speech.endpointing import Endpointing, UtteranceCutter, VoiceActivityDetector
from src.models.speech.transcript import SpeechCheck

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = ["SpeechGate"]


class SpeechGate:
    """A voice detector and the endpointing rule. Build it with `from_config` or `from_parts`; the verb
    is `check()`."""

    def __init__(self, *, detector: VoiceActivityDetector, endpointing: Endpointing) -> None:
        self._detector = detector
        self._endpointing = endpointing
        self._lock = threading.Lock()

    @classmethod
    def from_config(
        cls, *, config: SpeechToTextConfig, endpointing: Maybe[Endpointing] = UNSET
    ) -> SpeechGate:
        """The YAML door: Silero from `models.stt.vad_model_path`. Loads nothing."""
        from src.models.speech.silero import SileroVoiceActivityDetector

        return cls.from_parts(
            detector=SileroVoiceActivityDetector.from_config(config=config), endpointing=endpointing
        )

    @classmethod
    def from_parts(
        cls, *, detector: VoiceActivityDetector, endpointing: Maybe[Endpointing] = UNSET
    ) -> SpeechGate:
        """The Python door. ``endpointing`` UNSET is `Endpointing()`, whose values are placeholders."""
        return cls(detector=detector, endpointing=resolve("endpointing", endpointing, Endpointing()))

    @property
    def detector(self) -> VoiceActivityDetector:
        """The detector this gate scores windows with."""
        return self._detector

    def check(self, samples: np.ndarray, *, samplerate: int) -> SpeechCheck:
        """Whether ``samples`` (frames, or frames by channels) recorded at ``samplerate`` hold an utterance.

        The recording is mixed to mono, brought to the detector's rate and cut into its windows; a last
        partial window is filled with silence. The detector is reset first, and the lock is held for the
        whole recording, because the detector carries state from one window to the next.
        """
        import numpy as np

        if samplerate <= 0:
            raise ValueError(f"a recording needs a positive sample rate, not {samplerate} Hz.")
        array = np.asarray(samples)
        frames = int(array.shape[0]) if array.ndim else 0
        rate, size = self._detector.samplerate, self._detector.window_samples
        load = getattr(self._detector, "load", None)
        with self._lock:
            if callable(load):
                load()
            started = time.perf_counter()
            audio = to_mono_at_rate(array, samplerate=samplerate, target=rate)
            if audio.size % size:
                audio = np.concatenate((audio, np.zeros(size - audio.size % size, dtype=np.float32)))
            cutter = UtteranceCutter(endpointing=self._endpointing, window_samples=size, samplerate=rate)
            self._detector.reset()
            heard, peak, scored = False, 0.0, 0
            for offset in range(0, audio.size, size):
                window = audio[offset:offset + size]
                probability = float(self._detector.speech_probability(window))
                peak = max(peak, probability)
                scored += 1
                if cutter.push(window, probability) is not None:
                    heard = True
                    break
            if not heard and cutter.finish() is not None:
                heard = True
            latency_ms = (time.perf_counter() - started) * 1000.0
        duration_s = frames / float(samplerate)
        return SpeechCheck(
            heard_speech=heard,
            duration_s=duration_s,
            checked_s=min(scored * size / rate, duration_s),
            peak_probability=peak,
            onset=self._endpointing.onset,
            min_speech_s=self._endpointing.min_speech_s,
            detector=str(getattr(self._detector, "name", type(self._detector).__name__)),
            latency_ms=latency_ms,
        )
