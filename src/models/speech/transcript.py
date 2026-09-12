"""What the speech stack answers with, as frozen reports: a transcript, a voice check, and a proposal.

`Transcript` is the one shape the engine returns on both paths. It carries what a bare string cannot:
which language Whisper decided it heard, how long the recording was, and how many milliseconds the
decode took against the 2 s budget from the end of speaking, so all three can be checked afterwards.
`SpeechCheck` is what the voice detector found before Whisper was asked, and `Proposal` is what one
upload proposes for the prompt box: the words, or an empty text and the reason there are none.

Every one of them is a proposal and nothing more. Voice never acts on its own: the text lands in the
prompt box and a human confirms it, "Stopp" included.

stdlib only, so a caller can name the types without importing the speech stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

__all__ = ["LanguageSource", "Proposal", "SpeechCheck", "Transcript"]


class LanguageSource(StrEnum):
    """Where the language of a transcript came from."""

    #: The engine picked it from the recording (`models.stt.language: auto`): German or English, whichever
    #: of the two Whisper's language scores rank higher.
    DETECTED = "detected"
    #: The config forced it (`models.stt.language: german` or `english`); nothing was detected.
    CONFIGURED = "configured"


def _ascii(text: str) -> str:
    """``text`` with every non-ASCII character escaped, so a rendered line survives a cp1252 console."""
    return text.encode("ascii", "backslashreplace").decode("ascii")


@dataclass(frozen=True, slots=True)
class Transcript:
    """The most likely text for one recording, with the language, the duration and the cost."""

    #: The words as decoded, stripped of surrounding whitespace and otherwise untouched: never
    #: lower-cased and never translated.
    text: str
    #: The Whisper language code the decoder prompt carried (``de``, ``en``), or ``None`` when the
    #: output carried no language token at all.
    language: str | None
    language_source: LanguageSource
    #: Length of the recording in seconds, at the rate it arrived in.
    duration_s: float
    #: Which engine decoded it, e.g. ``whisper-transformers``.
    engine: str
    #: The weights it ran: a local directory or a Hub id.
    model: str
    #: The torch device type the decode ran on (``cuda``, ``cpu``).
    device: str
    #: Milliseconds from the audio to the text. The one-time weight load is not part of it.
    latency_ms: float

    def render(self) -> str:
        """Three ASCII lines: what was heard in which language, what it cost, which weights."""
        heard = f"Heard '{_ascii(self.text)}'" if self.text else "Heard no words"
        language = self.language if self.language is not None else "language unknown"
        return "\n".join((
            f"{heard} ({_ascii(language)}, {self.language_source.value})",
            f"  {self.duration_s:.2f} s of audio, decoded in {self.latency_ms:.0f} ms "
            f"by {_ascii(self.engine)} on {_ascii(self.device)}",
            f"  model {_ascii(self.model)}",
        ))

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data that survives ``json.dumps`` with no custom encoder."""
        return {
            "text": self.text,
            "language": self.language,
            "language_source": self.language_source.value,
            "duration_s": self.duration_s,
            "engine": self.engine,
            "model": self.model,
            "device": self.device,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class SpeechCheck:
    """What the voice detector found in one recording, before Whisper was asked about it."""

    #: An utterance closed: a window at or above `onset`, and at least `min_speech_s` of speech in it.
    heard_speech: bool
    #: Length of the recording in seconds, at the rate it arrived in.
    duration_s: float
    #: Seconds of audio the detector scored before it answered. It stops at the first utterance.
    checked_s: float
    #: The highest speech probability of any window it scored, in [0, 1].
    peak_probability: float
    #: The probability a window needs to open an utterance (`Endpointing.onset`).
    onset: float
    #: The speech an utterance needs to be kept (`Endpointing.min_speech_s`).
    min_speech_s: float
    #: Which detector scored it, e.g. ``silero-vad``.
    detector: str
    #: Milliseconds the check took. The one-time model load is not part of it.
    latency_ms: float

    def render(self) -> str:
        """One ASCII line: whether speech was heard, on what evidence, by which detector, at what cost."""
        if self.heard_speech:
            head = f"Speech heard within {self.checked_s:.2f} s of {self.duration_s:.2f} s of audio"
        else:
            head = f"No speech in {self.duration_s:.2f} s of audio"
        return (
            f"{head} (peak speech probability {self.peak_probability:.2f}, onset {self.onset:.2f}, "
            f"at least {self.min_speech_s:.2f} s of speech) by {_ascii(self.detector)} "
            f"in {self.latency_ms:.0f} ms"
        )

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data that survives ``json.dumps`` with no custom encoder."""
        return {
            "heard_speech": self.heard_speech,
            "duration_s": self.duration_s,
            "checked_s": self.checked_s,
            "peak_probability": self.peak_probability,
            "onset": self.onset,
            "min_speech_s": self.min_speech_s,
            "detector": self.detector,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class Proposal:
    """What one recording proposes for the prompt box: the words, or an empty text and why."""

    #: The words Whisper decoded, or ``""`` when there are none to propose.
    text: str
    #: Why `text` is empty, as a sentence; ``None`` when it holds words.
    reason: str | None
    #: What the voice detector found before Whisper was asked.
    speech: SpeechCheck
    #: Whisper's report; ``None`` when the voice detector heard no speech and Whisper was not asked.
    transcript: Transcript | None

    def render(self) -> str:
        """The proposal in a line, the voice check and the transcript indented under it. ASCII."""
        head = f"Proposed '{_ascii(self.text)}'" if self.text else f"Nothing proposed: {_ascii(self.reason or '')}"
        lines = [head, f"  {self.speech.render()}"]
        if self.transcript is not None:
            lines.extend(f"  {line}" for line in self.transcript.render().split("\n"))
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data, the check and the transcript as their own `to_dict()`."""
        return {
            "text": self.text,
            "reason": self.reason,
            "speech": self.speech.to_dict(),
            "transcript": None if self.transcript is None else self.transcript.to_dict(),
        }
