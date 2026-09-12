"""`listen()`: the first utterance a voice stream closes, transcribed, as a frozen report.

A `Listener` holds a source of audio (the cell PC's microphone, `MicrophoneSource`), a
`VoiceActivityDetector` and a `SpeechEngine`. `listen()` drops the audio captured before the call, reads
blocks, cuts them into detector windows, feeds an `UtteranceCutter`, and hands the first utterance that
closes to the engine. The answer is an `Utterance` carrying the engine's `Transcript`, and it is bounded:
silence, or a microphone that delivers nothing, returns `TIMED_OUT` at the bound instead of waiting.

Nothing here acts. The transcript is a proposal a human confirms, "Stopp" included.

`from_config` builds `SileroVoiceActivityDetector` from `models.stt.vad_model_path` when no detector is
chosen, and takes the engine it is handed, which is the one `shared_speech()` holds for uploads. The
detector is its own: it carries state from one window to the next.

Not built yet, on purpose: push-to-talk (a console button and a USB hand or foot switch raising one
event) and a caller. No console route or cell verb opens a `Listener` today.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import UNSET, Maybe, chosen, resolve
from src.models.speech.capture import AudioSource, MicrophoneSource
from src.models.speech.endpointing import (
    CutUtterance,
    Endpointing,
    UtteranceCutter,
    UtteranceEnd,
    VoiceActivityDetector,
)
from src.models.speech.engine import SpeechEngine
from src.models.speech.transcript import Transcript

if TYPE_CHECKING:  # pragma: no cover, typing only
    from types import TracebackType

__all__ = ["ListenOutcome", "Listener", "Utterance"]

#: How long `listen()` waits for an utterance to close when nobody chose a bound.
_DEFAULT_TIMEOUT_S: Final[float] = 30.0
#: The longest one read waits, so the bound is checked at least this often.
_READ_WAIT_S: Final[float] = 0.1


class ListenOutcome(StrEnum):
    """How one `listen()` ended."""

    #: An utterance closed and the engine transcribed it.
    HEARD = "heard"
    #: The bound elapsed before an utterance closed.
    TIMED_OUT = "timed_out"
    #: The source ended holding no utterance with enough speech.
    SOURCE_ENDED = "source_ended"


@dataclass(frozen=True, slots=True)
class Utterance:
    """What one `listen()` heard, or why it heard nothing."""

    outcome: ListenOutcome
    #: The engine's answer; None unless `outcome` is HEARD.
    transcript: Transcript | None
    #: What closed the utterance; None unless `outcome` is HEARD.
    ended_by: UtteranceEnd | None
    #: Seconds of audio handed to the engine, pre-roll and trailing silence included; 0 when none.
    speech_s: float
    #: Seconds from the start of `listen()` until the utterance closed, the bound elapsed or the source
    #: ended. The decode is not part of it; `transcript.latency_ms` is.
    waited_s: float
    #: The bound this `listen()` ran under; None means it waited for the source.
    timeout_s: float | None
    #: Audio was lost while listening: the driver overflowed, or the capture ring dropped its oldest.
    overflowed: bool

    def render(self) -> str:
        """The outcome in a line, the transcript indented under it, and a line when audio was lost."""
        if self.outcome is ListenOutcome.HEARD and self.transcript is not None:
            ended = "unknown" if self.ended_by is None else self.ended_by.value.replace("_", " ")
            lines = [
                f"Heard an utterance of {self.speech_s:.2f} s, ended by {ended}, "
                f"after {self.waited_s:.2f} s of listening"
            ]
            lines.extend(f"  {line}" for line in self.transcript.render().split("\n"))
        elif self.outcome is ListenOutcome.TIMED_OUT:
            bound = "" if self.timeout_s is None else f" within {self.timeout_s:.1f} s"
            lines = [f"Nothing heard{bound}"]
        else:
            lines = [f"The source ended after {self.waited_s:.2f} s without an utterance"]
        if self.overflowed:
            lines.append("  audio was dropped while listening (the capture overflowed)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data, the transcript as its own `to_dict()`."""
        return {
            "outcome": self.outcome.value,
            "transcript": None if self.transcript is None else self.transcript.to_dict(),
            "ended_by": None if self.ended_by is None else self.ended_by.value,
            "speech_s": self.speech_s,
            "waited_s": self.waited_s,
            "timeout_s": self.timeout_s,
            "overflowed": self.overflowed,
        }


def _checked_bound(bound: float | None) -> float | None:
    if bound is not None and not bound > 0:
        raise ValueError(f"listen() needs a positive bound in seconds, or None for none; got {bound}.")
    return bound


class Listener:
    """A voice stream cut into utterances. Build it with `from_config` or `from_parts`; the verb is
    `listen()`, and `start()`/`stop()` (or `with`) open and close the source."""

    def __init__(
        self,
        *,
        source: AudioSource,
        detector: VoiceActivityDetector,
        engine: SpeechEngine,
        endpointing: Endpointing,
        timeout_s: float | None,
        clock: Callable[[], float],
    ) -> None:
        self._source = source
        self._detector = detector
        self._engine = engine
        self._endpointing = endpointing
        self._timeout_s = timeout_s
        self._clock = clock

    @classmethod
    def from_config(
        cls,
        *,
        config: SpeechToTextConfig,
        engine: SpeechEngine,
        detector: Maybe[VoiceActivityDetector] = UNSET,
        endpointing: Maybe[Endpointing] = UNSET,
        timeout_s: Maybe[float | None] = UNSET,
    ) -> Listener:
        """The YAML door: the microphone keys of `models.stt` describe the cell PC's microphone.

        The engine is handed in rather than built, because it is the one the upload path already holds
        (`shared_speech().for_config(config=config).engine`). ``detector`` UNSET is Silero from
        `models.stt.vad_model_path`, a detector of this listener's own. Opens and loads nothing.
        """
        if not chosen(detector):
            from src.models.speech.silero import SileroVoiceActivityDetector

            detector = SileroVoiceActivityDetector.from_config(config=config)
        return cls.from_parts(
            source=MicrophoneSource.from_config(config=config),
            detector=detector,
            engine=engine,
            endpointing=endpointing,
            timeout_s=timeout_s,
        )

    @classmethod
    def from_parts(
        cls,
        *,
        source: AudioSource,
        detector: VoiceActivityDetector,
        engine: SpeechEngine,
        endpointing: Maybe[Endpointing] = UNSET,
        timeout_s: Maybe[float | None] = UNSET,
        clock: Callable[[], float] = time.monotonic,
    ) -> Listener:
        """The Python door. Opens nothing.

        ``timeout_s`` is the bound `listen()` uses when a call chooses none: UNSET means 30 s, None
        means no bound. ``endpointing`` UNSET is `Endpointing()`, whose values are placeholders.
        """
        if detector.samplerate != source.samplerate:
            raise ValueError(
                f"the detector scores audio at {detector.samplerate} Hz and the source delivers "
                f"{source.samplerate} Hz; open the source at {detector.samplerate} Hz."
            )
        return cls(
            source=source,
            detector=detector,
            engine=engine,
            endpointing=resolve("endpointing", endpointing, Endpointing()),
            timeout_s=_checked_bound(resolve("timeout_s", timeout_s, _DEFAULT_TIMEOUT_S)),
            clock=clock,
        )

    @property
    def source(self) -> AudioSource:
        """Where this listener reads audio from."""
        return self._source

    @property
    def detector(self) -> VoiceActivityDetector:
        """The voice detector this listener cuts utterances with."""
        return self._detector

    @property
    def engine(self) -> SpeechEngine:
        """The engine this listener hands each utterance to."""
        return self._engine

    def start(self) -> None:
        """Open the source and begin capturing."""
        self._source.start()

    def stop(self) -> None:
        """Close the source."""
        self._source.stop()

    def __enter__(self) -> Listener:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def listen(self, *, timeout_s: Maybe[float | None] = UNSET) -> Utterance:
        """The first utterance that closes from now on, transcribed; or why there was none.

        ``timeout_s`` UNSET uses the listener's own bound; None waits for the source to end.
        """
        import numpy as np

        bound = _checked_bound(resolve("timeout_s", timeout_s, self._timeout_s))
        started = self._clock()
        self._source.discard_buffered()
        self._detector.reset()
        size = self._detector.window_samples
        cutter = UtteranceCutter(
            endpointing=self._endpointing, window_samples=size, samplerate=self._source.samplerate
        )
        pending = np.zeros(0, dtype=np.float32)
        overflowed = False
        while True:
            waited = self._clock() - started
            if bound is not None and waited >= bound:
                return self._nothing(ListenOutcome.TIMED_OUT, waited, bound, overflowed)
            wait = _READ_WAIT_S if bound is None else min(_READ_WAIT_S, bound - waited)
            block = self._source.read(timeout_s=wait)
            if block is None:
                if not self._source.ended:
                    continue
                cut = cutter.finish()
                waited = self._clock() - started
                if cut is None:
                    return self._nothing(ListenOutcome.SOURCE_ENDED, waited, bound, overflowed)
                return self._heard(cut, waited, bound, overflowed)
            overflowed = overflowed or block.overflowed
            pending = np.concatenate((pending, block.samples)) if pending.size else block.samples
            offset = 0
            while pending.size - offset >= size:
                window = pending[offset:offset + size]
                offset += size
                cut = cutter.push(window, float(self._detector.speech_probability(window)))
                if cut is not None:
                    return self._heard(cut, self._clock() - started, bound, overflowed)
            pending = pending[offset:]

    def _heard(
        self, cut: CutUtterance, waited: float, bound: float | None, overflowed: bool
    ) -> Utterance:
        samplerate = self._source.samplerate
        transcript = self._engine.transcribe(cut.samples, samplerate=samplerate)
        return Utterance(
            outcome=ListenOutcome.HEARD,
            transcript=transcript,
            ended_by=cut.ended_by,
            speech_s=cut.samples.shape[0] / samplerate,
            waited_s=waited,
            timeout_s=bound,
            overflowed=overflowed,
        )

    @staticmethod
    def _nothing(
        outcome: ListenOutcome, waited: float, bound: float | None, overflowed: bool
    ) -> Utterance:
        return Utterance(
            outcome=outcome,
            transcript=None,
            ended_by=None,
            speech_s=0.0,
            waited_s=waited,
            timeout_s=bound,
            overflowed=overflowed,
        )
