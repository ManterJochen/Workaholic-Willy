"""Push to talk: a talk switch gates the cell PC's microphone, and a turn is what is said while it is held.

The trigger is push to talk, from a console button and from a hand or foot switch at the cell PC, both
raising the same event. `TalkSwitch` is the seam that event comes through. `TalkButton` fills it in
software, and `shared_talk_button()` is the process's one: `POST /v1/voice/talk` presses and releases
it, and a switch reader at the cell PC presses the same one.

`PushToTalkSource` is an `AudioSource` around another, normally `MicrophoneSource`. The microphone
captures from `start()` on, and the source serves nothing until the switch goes down. At the press it
drops what was captured before it, while the switch is held it serves what arrives, and at the release
it serves what the ring still holds and then reports that it has ended. It never stops the microphone to
end a turn: `MicrophoneSource.stop()` drops the audio not yet read, and that audio is the end of the
command.

Two callers read a turn. A `Listener` built on the source listens across it, and the release closes the
utterance at the latest; a trailing silence still closes it earlier. `record()` is the turn without a
voice detector: every sample said while the switch was held, which is what `HeldSpeech.propose()` takes.
So a turn at the cell PC and a recording uploaded from the browser go through one gate and one engine.

Nothing here acts. What a turn says is a proposal, and a person confirms it (`confirm.py`).

Not built: a reader for a USB HID hand or foot switch at the cell PC, because the HID library it needs
gets a licence check first. A foot switch that sends a key works in the browser, through the console's
talk key.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import UNSET, Maybe, chosen, resolve
from src.models.speech.capture import AudioBlock, AudioSource, MicrophoneSource
from src.models.speech.engine import WHISPER_WINDOW_S

if TYPE_CHECKING:  # pragma: no cover, typing only
    from types import TracebackType

    import numpy as np

__all__ = [
    "PushToTalkSource",
    "TalkButton",
    "TalkOutcome",
    "TalkRecording",
    "TalkSwitch",
    "shared_talk_button",
]

#: The longest one read waits while the switch is held, so a release is noticed at least this often.
_HELD_POLL_S: Final[float] = 0.1
#: The longest `record()` hands one read, so its bounds are checked at least this often.
_READ_WAIT_S: Final[float] = 0.1


@runtime_checkable
class TalkSwitch(Protocol):
    """The push to talk switch: down while a person talks, up when they are done."""

    @property
    def pressed(self) -> bool:
        """True while the switch is held down."""
        ...

    @property
    def presses(self) -> int:
        """How often the switch went down since it was built. A press counts even when it came up
        again before anyone looked."""
        ...

    def wait_for_press(self, *, after: int, timeout_s: float) -> int:
        """Wait up to ``timeout_s`` until the switch has gone down more than ``after`` times, and
        return `presses` then."""
        ...


class TalkButton:
    """A talk switch pressed and released in software. Thread safe.

    The console's `POST /v1/voice/talk` presses the process's one (`shared_talk_button()`). A press
    while the button is already down is not counted, so a key that repeats while it is held is one
    press.
    """

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._pressed = False
        self._presses = 0

    @classmethod
    def from_parts(cls) -> TalkButton:
        """The Python door: a button that is up and has never been pressed."""
        return cls()

    @property
    def pressed(self) -> bool:
        """True while the button is down."""
        with self._changed:
            return self._pressed

    @property
    def presses(self) -> int:
        """How often the button went down since it was built."""
        with self._changed:
            return self._presses

    def press(self) -> None:
        """Put the button down. On a button that is already down it changes nothing."""
        with self._changed:
            if self._pressed:
                return
            self._pressed = True
            self._presses += 1
            self._changed.notify_all()

    def release(self) -> None:
        """Let the button up. On a button that is already up it changes nothing."""
        with self._changed:
            if not self._pressed:
                return
            self._pressed = False
            self._changed.notify_all()

    def wait_for_press(self, *, after: int, timeout_s: float) -> int:
        """Wait up to ``timeout_s`` until the button has gone down more than ``after`` times, and
        return `presses` then."""
        with self._changed:
            self._changed.wait_for(lambda: self._presses > after, timeout=max(timeout_s, 0.0))
            return self._presses


_SHARED: TalkButton | None = None
_SHARED_LOCK: Final[threading.Lock] = threading.Lock()


def shared_talk_button() -> TalkButton:
    """The process's talk button: the console's talk route presses it, and a `PushToTalkSource` built
    without a chosen switch listens to it."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = TalkButton.from_parts()
        return _SHARED


class TalkOutcome(StrEnum):
    """How one push to talk turn ended."""

    #: The switch went down and came up again, and audio arrived while it was held.
    RELEASED = "released"
    #: The bound elapsed before the switch went down.
    NOT_PRESSED = "not_pressed"
    #: The switch came up before the microphone delivered any audio.
    NOTHING_CAPTURED = "nothing_captured"
    #: The switch was still down at the longest turn. What arrived is kept and is not a command: a
    #: command cut off at the limit is a different command.
    HELD_TOO_LONG = "held_too_long"
    #: The microphone ended while the turn was open.
    SOURCE_ENDED = "source_ended"


@dataclass(frozen=True, slots=True)
class TalkRecording:
    """What one `record()` caught: the audio said while the switch was held, or why there is none."""

    outcome: TalkOutcome
    #: Mono float32 in [-1, 1] at `samplerate`: everything served between the press and the end of the
    #: turn. Empty when the switch never went down.
    samples: np.ndarray
    samplerate: int
    #: Seconds from the start of `record()` until the press was noticed, or until the bound.
    waited_s: float
    #: Seconds from the press until the turn ended; 0 when the switch never went down.
    held_s: float
    #: The bound on the wait for the press.
    timeout_s: float
    #: The longest turn this `record()` accepted.
    longest_s: float
    #: Audio was lost while the switch was held: the driver overflowed, or the ring dropped its oldest.
    overflowed: bool

    @property
    def ok(self) -> bool:
        """True when the turn is a command to propose: the switch came up, and audio arrived."""
        return self.outcome is TalkOutcome.RELEASED

    @property
    def frames(self) -> int:
        """Samples caught."""
        return int(self.samples.shape[0])

    @property
    def duration_s(self) -> float:
        """Seconds of audio caught."""
        return self.frames / self.samplerate

    def __str__(self) -> str:
        """What ``print()`` shows: the text :meth:`render` returns."""
        return self.render()

    def render(self) -> str:
        """What the turn caught, or why nothing, in a line; a second line when audio was lost. ASCII."""
        if self.outcome is TalkOutcome.RELEASED:
            head = (
                f"Recorded {self.duration_s:.2f} s while the talk switch was held for {self.held_s:.2f} s, "
                f"after {self.waited_s:.2f} s of waiting for it"
            )
        elif self.outcome is TalkOutcome.NOT_PRESSED:
            head = f"Nothing recorded: the talk switch was not pressed within {self.timeout_s:.1f} s"
        elif self.outcome is TalkOutcome.NOTHING_CAPTURED:
            head = (
                f"Nothing recorded: the talk switch came up after {self.held_s:.2f} s, before the "
                f"microphone delivered any audio. Hold it while you speak"
            )
        elif self.outcome is TalkOutcome.HELD_TOO_LONG:
            head = (
                f"Not proposed: the talk switch was still held after {self.longest_s:.1f} s, the "
                f"longest turn. Release it once the command is said"
            )
        elif self.held_s > 0.0 or self.frames:
            head = (
                f"The microphone ended {self.held_s:.2f} s into the turn, with {self.duration_s:.2f} s "
                f"recorded"
            )
        else:
            head = f"The microphone ended after {self.waited_s:.2f} s, before the talk switch went down"
        lines = [head]
        if self.overflowed:
            lines.append("  audio was dropped while the switch was held (the capture overflowed)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """The machine half: plain data, and the samples as their count, never the audio itself."""
        return {
            "outcome": self.outcome.value,
            "frames": self.frames,
            "samplerate": self.samplerate,
            "duration_s": self.duration_s,
            "waited_s": self.waited_s,
            "held_s": self.held_s,
            "timeout_s": self.timeout_s,
            "longest_s": self.longest_s,
            "overflowed": self.overflowed,
        }


class _Turn(StrEnum):
    """Where the source is in a turn."""

    WAITING = "waiting"
    HELD = "held"
    ENDED = "ended"


class PushToTalkSource:
    """An `AudioSource` that serves audio only while a talk switch is held.

    Build it with `from_config` or `from_parts`; `start()`/`stop()` (or `with`) open and close the
    microphone, and the verb is `record()`. One thread reads it.
    """

    def __init__(
        self, *, source: AudioSource, switch: TalkSwitch, clock: Callable[[], float]
    ) -> None:
        self._source = source
        self._switch = switch
        self._clock = clock
        self._turn = _Turn.WAITING
        #: The presses this turn does not count: they came up before the turn was armed.
        self._seen = 0

    @classmethod
    def from_config(
        cls,
        *,
        config: SpeechToTextConfig,
        switch: Maybe[TalkSwitch] = UNSET,
        device: Maybe[int | str] = UNSET,
    ) -> PushToTalkSource:
        """The YAML door: the cell PC's microphone from the keys of `models.stt`
        (`MicrophoneSource.from_config`), behind ``switch``. Opens nothing.

        ``switch`` UNSET is `shared_talk_button()`, the one the console's talk route presses.
        """
        return cls.from_parts(
            source=MicrophoneSource.from_config(config=config, device=device), switch=switch
        )

    @classmethod
    def from_parts(
        cls,
        *,
        source: AudioSource,
        switch: Maybe[TalkSwitch] = UNSET,
        clock: Callable[[], float] = time.monotonic,
    ) -> PushToTalkSource:
        """The Python door. Opens nothing. ``switch`` UNSET is `shared_talk_button()`."""
        if not chosen(switch):
            switch = shared_talk_button()
        return cls(source=source, switch=switch, clock=clock)

    @property
    def source(self) -> AudioSource:
        """The microphone behind the switch."""
        return self._source

    @property
    def switch(self) -> TalkSwitch:
        """The switch a turn waits for."""
        return self._switch

    @property
    def samplerate(self) -> int:
        """The microphone's rate, and so the rate of every block."""
        return self._source.samplerate

    @property
    def ended(self) -> bool:
        """True once a turn is over (the switch came up and the ring was served), or when the microphone
        is not open."""
        return self._turn is _Turn.ENDED or self._source.ended

    def start(self) -> None:
        """Open the microphone and arm a turn. On an open microphone it changes nothing."""
        if not self._source.ended:
            return
        self._source.start()
        self._arm()

    def stop(self) -> None:
        """Close the microphone. A turn still open ends with it, and audio not yet read is dropped."""
        self._source.stop()

    def __enter__(self) -> PushToTalkSource:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.stop()

    def discard_buffered(self) -> int:
        """Arm a new turn and drop the audio captured before now; while the switch is held, drop nothing.

        A `Listener` calls this at the start of every `listen()`. During a held turn the press has already
        dropped what came before it, and what came since is the command.
        """
        if self._turn is _Turn.HELD:
            return 0
        self._arm()
        return self._source.discard_buffered()

    def read(self, *, timeout_s: float) -> AudioBlock | None:
        """What the microphone caught while the switch is held; None before the press and after the turn.

        Before the press it waits up to ``timeout_s`` for one, and at the press drops what was captured
        before it. While the switch is held it waits up to 0.1 s for audio. At the release it serves what
        the ring still holds, once, and from then on the source has ended.
        """
        if self._turn is _Turn.WAITING:
            presses = self._switch.wait_for_press(after=self._seen, timeout_s=timeout_s)
            if presses <= self._seen:
                return None
            self._seen = presses
            self._source.discard_buffered()
            self._turn = _Turn.HELD
            return None
        if self._turn is _Turn.HELD:
            if self._switch.pressed and self._switch.presses == self._seen:
                return self._source.read(timeout_s=min(timeout_s, _HELD_POLL_S))
            # Up, or up and down again: the turn ended at the release. The microphone is not stopped,
            # because stopping drops what it has not handed over yet, and that is the end of the command.
            self._turn = _Turn.ENDED
            return self._source.read(timeout_s=0.0)
        return None

    def record(self, *, timeout_s: float, longest_s: Maybe[float] = UNSET) -> TalkRecording:
        """One turn: wait up to ``timeout_s`` for the switch, then everything caught until it comes up.

        ``longest_s`` UNSET is Whisper's window, 30 s; a switch still held then ends the turn as
        HELD_TOO_LONG. The microphone must be open (`with source:`). A turn armed before this call is
        armed again, and a press still held when it starts counts.
        """
        import numpy as np

        if not timeout_s > 0:
            raise ValueError(
                f"record() needs a positive bound in seconds on the wait for the talk switch; got "
                f"{timeout_s}."
            )
        longest = resolve("longest_s", longest_s, WHISPER_WINDOW_S)
        if not longest > 0:
            raise ValueError(f"record() needs a positive longest turn in seconds; got {longest}.")
        if self._source.ended:
            raise RuntimeError(
                "the microphone is not open: call start(), or enter the source with `with`, before "
                "record()."
            )
        started = self._clock()
        self.discard_buffered()
        parts: list[np.ndarray] = []
        overflowed = False
        pressed_at: float | None = None

        def recording(outcome: TalkOutcome, now: float) -> TalkRecording:
            samples = (
                np.concatenate(parts).astype(np.float32, copy=False)
                if parts
                else np.zeros(0, dtype=np.float32)
            )
            return TalkRecording(
                outcome=outcome,
                samples=samples,
                samplerate=self.samplerate,
                waited_s=(now if pressed_at is None else pressed_at) - started,
                held_s=0.0 if pressed_at is None else now - pressed_at,
                timeout_s=timeout_s,
                longest_s=longest,
                overflowed=overflowed,
            )

        while True:
            now = self._clock()
            if pressed_at is None:
                if now - started >= timeout_s:
                    return recording(TalkOutcome.NOT_PRESSED, now)
                wait = min(_READ_WAIT_S, timeout_s - (now - started))
            else:
                if now - pressed_at >= longest:
                    return recording(TalkOutcome.HELD_TOO_LONG, now)
                wait = min(_READ_WAIT_S, longest - (now - pressed_at))
            block = self.read(timeout_s=wait)
            if pressed_at is None and self._turn is not _Turn.WAITING:
                pressed_at = self._clock()
            if block is not None:
                parts.append(block.samples)
                overflowed = overflowed or block.overflowed
                continue
            if self._turn is _Turn.ENDED:
                caught = any(part.shape[0] for part in parts)
                outcome = TalkOutcome.RELEASED if caught else TalkOutcome.NOTHING_CAPTURED
                return recording(outcome, self._clock())
            if self._source.ended:
                return recording(TalkOutcome.SOURCE_ENDED, self._clock())

    def _arm(self) -> None:
        """Wait for a new press. A press still held now counts; one that already came up does not."""
        self._turn = _Turn.WAITING
        self._seen = self._switch.presses - (1 if self._switch.pressed else 0)
