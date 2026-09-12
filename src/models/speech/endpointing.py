"""Where an utterance starts and ends: the voice detector's seam and the endpointing state machine.

`VoiceActivityDetector` scores one window of audio. `SileroVoiceActivityDetector` (`silero.py`) fills
it: Silero VAD 6.2.1 as TorchScript on the CPU, with windows of 512 samples at 16 kHz.

`UtteranceCutter` turns those scores into utterances: an onset threshold with hysteresis, a minimum
amount of speech, a trailing silence, a pre-roll kept from before the onset, and a maximum length. It is
pure: windows and probabilities in, utterances out, no clock and no device.

Every default in `Endpointing` is a placeholder. No source gives the right values for short commands on
a shop floor: Silero's own iterator ends an utterance after 100 ms of silence, which is short for a
hesitant command, and faster-whisper's file filter after 2000 ms, which is long. They are measured on
recorded cell audio. The trailing silence counts against the 2 s budget from the end of speaking.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = ["CutUtterance", "Endpointing", "UtteranceCutter", "UtteranceEnd", "VoiceActivityDetector"]


@runtime_checkable
class VoiceActivityDetector(Protocol):
    """How likely one window of audio is speech."""

    @property
    def samplerate(self) -> int:
        """The only rate the detector accepts."""
        ...

    @property
    def window_samples(self) -> int:
        """The only window length the detector accepts (Silero: 512 samples at 16 kHz)."""
        ...

    def reset(self) -> None:
        """Forget the state carried between windows, before a new stretch of audio."""
        ...

    def speech_probability(self, window: np.ndarray) -> float:
        """A probability in [0, 1] that ``window``, mono float32 and `window_samples` long, is speech."""
        ...


class UtteranceEnd(StrEnum):
    """What closed an utterance."""

    #: The silence after the speech lasted `Endpointing.trailing_silence_s`.
    SILENCE = "silence"
    #: The utterance reached `Endpointing.max_utterance_s` while speech went on.
    MAX_LENGTH = "max_length"
    #: The source ended while the utterance was open.
    SOURCE_END = "source_end"


@dataclass(frozen=True, slots=True)
class Endpointing:
    """When an utterance starts and ends. Every default is a placeholder until measured on cell audio."""

    #: A window at or above this probability opens an utterance.
    onset: float = 0.5
    #: Inside an utterance a window at or above this still counts as speech. Below `onset` on purpose:
    #: a word that dips between the two does not start the trailing silence.
    offset: float = 0.35
    #: An utterance with less speech than this is a click or a cough, and it is dropped.
    min_speech_s: float = 0.25
    #: Silence this long after speech closes the utterance.
    trailing_silence_s: float = 0.5
    #: Audio kept from before the onset, so the first syllable is not cut off.
    pre_roll_s: float = 0.2
    #: An utterance still speaking at this length is closed anyway.
    max_utterance_s: float = 15.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.offset <= self.onset <= 1.0:
            raise ValueError(
                f"endpointing needs 0 <= offset <= onset <= 1; got offset={self.offset}, "
                f"onset={self.onset}."
            )
        if min(self.min_speech_s, self.trailing_silence_s, self.max_utterance_s) <= 0.0 or (
            self.pre_roll_s < 0.0
        ):
            raise ValueError(
                f"endpointing durations must be positive, and the pre-roll at least 0; got "
                f"min_speech_s={self.min_speech_s}, trailing_silence_s={self.trailing_silence_s}, "
                f"pre_roll_s={self.pre_roll_s}, max_utterance_s={self.max_utterance_s}."
            )
        if self.max_utterance_s < self.min_speech_s:
            raise ValueError(
                f"max_utterance_s={self.max_utterance_s} is shorter than min_speech_s="
                f"{self.min_speech_s}, so no utterance could ever be kept."
            )


@dataclass(frozen=True, slots=True)
class CutUtterance:
    """The samples of one closed utterance, pre-roll and trailing silence included, and what closed it."""

    samples: np.ndarray
    ended_by: UtteranceEnd


def _windows(seconds: float, window_s: float) -> int:
    """How many whole windows cover ``seconds``, at least one, forgiving a hair of float error."""
    return max(1, math.ceil(seconds / window_s - 1e-6))


class UtteranceCutter:
    """The endpointing state machine. Push every window with its speech probability; a closed
    utterance comes back from the push that closed it."""

    def __init__(self, *, endpointing: Endpointing, window_samples: int, samplerate: int) -> None:
        if window_samples <= 0 or samplerate <= 0:
            raise ValueError(
                f"a window needs a positive length and rate; got window_samples={window_samples}, "
                f"samplerate={samplerate}."
            )
        window_s = window_samples / samplerate
        self._endpointing = endpointing
        self._min_speech = _windows(endpointing.min_speech_s, window_s)
        self._trailing = _windows(endpointing.trailing_silence_s, window_s)
        self._longest = _windows(endpointing.max_utterance_s, window_s)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=round(endpointing.pre_roll_s / window_s))
        self._open: list[np.ndarray] = []
        self._voiced = 0
        self._silent = 0

    @property
    def in_speech(self) -> bool:
        """True while an utterance is open."""
        return bool(self._open)

    def reset(self) -> None:
        """Forget the pre-roll and any open utterance."""
        self._pre_roll.clear()
        self._open = []
        self._voiced = self._silent = 0

    def push(self, window: np.ndarray, probability: float) -> CutUtterance | None:
        """Take one window. Returns the utterance this window closed, if it closed one worth keeping."""
        if not self._open:
            if probability < self._endpointing.onset:
                self._pre_roll.append(window)
                return None
            self._open = [*self._pre_roll, window]
            self._pre_roll.clear()
            self._voiced, self._silent = 1, 0
            return self._close(UtteranceEnd.MAX_LENGTH) if len(self._open) >= self._longest else None
        self._open.append(window)
        if probability >= self._endpointing.offset:
            self._voiced += 1
            self._silent = 0
        else:
            self._silent += 1
        if self._silent >= self._trailing:
            return self._close(UtteranceEnd.SILENCE)
        if len(self._open) >= self._longest:
            return self._close(UtteranceEnd.MAX_LENGTH)
        return None

    def finish(self) -> CutUtterance | None:
        """The source ended: close the open utterance, if there is one worth keeping."""
        return self._close(UtteranceEnd.SOURCE_END) if self._open else None

    def _close(self, ended_by: UtteranceEnd) -> CutUtterance | None:
        import numpy as np

        windows, voiced = self._open, self._voiced
        self._open = []
        self._voiced = self._silent = 0
        if voiced < self._min_speech:
            return None
        return CutUtterance(samples=np.concatenate(windows).astype(np.float32, copy=False), ended_by=ended_by)
