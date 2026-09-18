"""The microphone as a continuous stream of mono blocks, and the one conversion every block goes through.

`MicrophoneSource` opens an `InputStream`, whose callback receives a numpy array of (frames, channels),
rather than a `RawInputStream`, whose callback receives a plain buffer with no shape and no dtype. The
callback only copies that array into a ring allocated at `start()` and counts what was lost; `read()`
scales each block by the full scale of its own format and averages the channels, on the reader's thread.

Each format is divided by its own full scale, never by 32768 throughout: that constant makes int32 audio
65536 times too loud and float32 audio 32768 times too quiet.

sounddevice is imported in `start()`, never at module level. A machine without PortAudio gets
`SpeechStackUnavailable` naming `requirements.txt`, and still transcribes uploads. A machine whose
input device is missing or refuses the stream gets `MicrophoneUnavailable`, a capability the host lacks.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

from src.config.schema.models.models_schema import SampleFormat, SpeechToTextConfig
from src.contracts import UNSET, Maybe, resolve
from src.models.speech.engine import SpeechStackUnavailable

if TYPE_CHECKING:  # pragma: no cover, typing only
    import numpy as np

__all__ = [
    "AudioBlock",
    "AudioSource",
    "MicrophoneSource",
    "MicrophoneUnavailable",
    "to_mono_at_rate",
    "to_mono_float32",
]

#: The full scale of every sample format the stream opens, so each block lands in [-1, 1].
_FULL_SCALE: Final[dict[str, float]] = {"int16": 32768.0, "int32": 2147483648.0, "float32": 1.0}


def to_mono_float32(block: np.ndarray, sample_format: SampleFormat) -> np.ndarray:
    """One block from the sound card as mono float32 in [-1, 1].

    ``block`` is (frames,) or (frames, channels) in ``sample_format``. Channels are averaged: a laptop's
    second microphone often carries most of the voice, and dropping one would hear the quiet side.
    """
    import numpy as np

    samples = np.asarray(block)
    if samples.dtype != np.dtype(sample_format):
        raise ValueError(
            f"a {samples.dtype} block reached a stream opened as {sample_format}; scaling it by the "
            f"wrong full scale would make it silently too loud or too quiet."
        )
    if samples.ndim not in (1, 2):
        raise ValueError(f"a block is (frames,) or (frames, channels), not {samples.shape}.")
    scaled = samples.astype(np.float32) / np.float32(_FULL_SCALE[sample_format])
    if scaled.ndim == 2:
        scaled = scaled.mean(axis=1, dtype=np.float32)
    return scaled


def to_mono_at_rate(samples: np.ndarray, *, samplerate: int, target: int) -> np.ndarray:
    """A recording, (frames,) or (frames, channels) in [-1, 1], as mono float32 at ``target`` Hz.

    Channels are averaged, as for every block. Another rate goes through scipy's polyphase resampler,
    imported here and never at module level; a machine that cannot import scipy gets
    `SpeechStackUnavailable` naming it. Whisper and the voice detector both read audio through this.
    """
    import numpy as np

    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1, dtype=np.float32)
    if samplerate != target:
        try:
            from scipy.signal import resample_poly
        except (ImportError, OSError) as exc:
            raise SpeechStackUnavailable(package="scipy", cause=exc) from exc
        audio = resample_poly(audio, target, samplerate).astype(np.float32)
    return audio


@dataclass(frozen=True, slots=True)
class AudioBlock:
    """Mono float32 samples in [-1, 1] at the source's rate, and whether audio was lost before them."""

    samples: np.ndarray
    #: The driver reported an input overflow, or the capture ring was full and dropped its oldest
    #: audio, since the previous block.
    overflowed: bool = False


@runtime_checkable
class AudioSource(Protocol):
    """Where a `Listener` reads audio from: the cell PC's microphone, or a scripted source in a test."""

    @property
    def samplerate(self) -> int:
        """The rate every block arrives at."""
        ...

    @property
    def ended(self) -> bool:
        """True once the source will deliver nothing more. A microphone has ended when it is stopped."""
        ...

    def start(self) -> None:
        """Begin capturing. On a source that is already capturing it changes nothing."""
        ...

    def stop(self) -> None:
        """Stop capturing and release the device."""
        ...

    def discard_buffered(self) -> int:
        """Drop the audio captured before now, and say how many frames that was."""
        ...

    def read(self, *, timeout_s: float) -> AudioBlock | None:
        """Everything captured since the last read, waiting up to ``timeout_s`` for any; None when none came."""
        ...


class MicrophoneUnavailable(RuntimeError):
    """This machine has no microphone the stream can open: no usable input device, or one that refused
    the rate, the channels or the format. A capability the host lacks, not a fault of the caller. It
    stays a `RuntimeError`, so a handler that catches `RuntimeError` still catches it."""


def _import_sounddevice() -> Any:
    """sounddevice, or the refusal naming the file that installs it. A missing PortAudio raises OSError."""
    try:
        import sounddevice
    except (ImportError, OSError) as exc:
        raise SpeechStackUnavailable(package="sounddevice", cause=exc) from exc
    return sounddevice


class MicrophoneSource:
    """The cell PC's microphone through sounddevice, capturing continuously into a ring buffer.

    Build it with `from_config` or `from_parts`; `start()` opens the device and nothing opens it at
    construction. The device is the host's default input unless one is chosen: which microphone the
    cell PC gets is still open, and the endpointing is tuned on the real one.
    """

    def __init__(
        self,
        *,
        samplerate: int,
        blocksize: int,
        channels: int,
        sample_format: SampleFormat,
        device: Maybe[int | str],
        buffer_s: float,
    ) -> None:
        self._samplerate = samplerate
        self._blocksize = blocksize
        self._channels = channels
        self._sample_format: SampleFormat = sample_format
        self._device = device
        self._buffer_s = buffer_s
        self._ready = threading.Condition()
        self._stream: Any = None
        self._ring: Any = None
        #: Index of the oldest frame held, and how many frames are held.
        self._first = 0
        self._count = 0
        #: Overflows reported plus frames the full ring dropped, since the last read.
        self._lost = 0

    @classmethod
    def from_config(
        cls, *, config: SpeechToTextConfig, device: Maybe[int | str] = UNSET
    ) -> MicrophoneSource:
        """The YAML door: `models.stt.samplerate`, `blocksize`, `channels` and `dtype`. Opens nothing."""
        return cls.from_parts(
            samplerate=config.samplerate,
            blocksize=config.blocksize,
            channels=config.channels,
            sample_format=config.dtype,
            device=device,
        )

    @classmethod
    def from_parts(
        cls,
        *,
        samplerate: int = 16000,
        blocksize: int = 0,
        channels: int = 1,
        sample_format: SampleFormat = "int16",
        device: Maybe[int | str] = UNSET,
        buffer_s: float = 30.0,
    ) -> MicrophoneSource:
        """The Python door. Opens nothing.

        ``device`` left UNSET is the host's default input. ``buffer_s`` bounds the unread audio the ring
        holds; beyond it the oldest audio is dropped and the next block says so.
        """
        if samplerate <= 0 or channels < 1 or blocksize < 0 or buffer_s <= 0:
            raise ValueError(
                f"a microphone stream needs a positive rate, at least one channel, a block size of 0 or "
                f"more and a positive buffer; got samplerate={samplerate}, channels={channels}, "
                f"blocksize={blocksize}, buffer_s={buffer_s}."
            )
        if sample_format not in _FULL_SCALE:
            raise ValueError(
                f"the microphone stream opens int16, int32 or float32, not {sample_format!r}."
            )
        return cls(
            samplerate=samplerate,
            blocksize=blocksize,
            channels=channels,
            sample_format=sample_format,
            device=device,
            buffer_s=buffer_s,
        )

    @property
    def samplerate(self) -> int:
        """The rate the stream opens at, and so the rate of every block."""
        return self._samplerate

    @property
    def ended(self) -> bool:
        """True while the microphone is not open."""
        with self._ready:
            return self._stream is None

    def start(self) -> None:
        """Open the microphone and begin capturing. On an open microphone it changes nothing."""
        with self._ready:
            if self._stream is not None:
                return
        sd = _import_sounddevice()
        import numpy as np

        device: int | str | None = resolve("device", self._device, None)
        info = self._input_device(sd, device)
        # A rate the WASAPI mixer does not run at is refused as an invalid device unless the mixer may
        # convert it (PortAudio issue 314). Other host APIs resample on their own.
        hostapi = str(sd.query_hostapis(info["hostapi"])["name"])
        extra = sd.WasapiSettings(auto_convert=True) if "WASAPI" in hostapi else None
        with self._ready:
            capacity = max(int(self._samplerate * self._buffer_s), 1)
            self._ring = np.zeros((capacity, self._channels), dtype=self._sample_format)
            self._first = self._count = self._lost = 0
        try:
            stream = sd.InputStream(
                samplerate=self._samplerate,
                blocksize=self._blocksize,
                device=device,
                channels=self._channels,
                dtype=self._sample_format,
                callback=self._callback,
                extra_settings=extra,
            )
            stream.start()
        except (sd.PortAudioError, ValueError) as exc:
            raise MicrophoneUnavailable(
                f"the microphone '{info['name']}' did not open at {self._samplerate} Hz, "
                f"{self._channels} channel(s), {self._sample_format} ({type(exc).__name__}: {exc}). "
                f"The device runs at {float(info['default_samplerate']):.0f} Hz by default."
            ) from exc
        with self._ready:
            self._stream = stream

    @staticmethod
    def _input_device(sd: Any, device: int | str | None) -> Any:
        try:
            return sd.query_devices(device, "input")
        except (sd.PortAudioError, ValueError) as exc:
            chosen_device = "default input device" if device is None else f"input device {device!r}"
            raise MicrophoneUnavailable(
                f"this machine has no usable {chosen_device} ({type(exc).__name__}: {exc})."
            ) from exc

    def stop(self) -> None:
        """Close the microphone. Audio not yet read is dropped."""
        with self._ready:
            stream, self._stream = self._stream, None
            self._count = 0
            self._ready.notify_all()
        if stream is not None:
            stream.stop()
            stream.close()

    def discard_buffered(self) -> int:
        """Drop the audio captured before now, and say how many frames that was."""
        with self._ready:
            dropped, self._count = self._count, 0
            self._first = 0
            self._lost = 0
            return dropped

    def read(self, *, timeout_s: float) -> AudioBlock | None:
        """Everything captured since the last read, waiting up to ``timeout_s`` for any; None when none came."""
        with self._ready:
            if self._stream is None:
                raise RuntimeError(
                    "the microphone is not open: call start(), or enter the Listener with `with`, "
                    "before reading it."
                )
            if self._count == 0 and timeout_s > 0:
                self._ready.wait(timeout=timeout_s)
            if self._count == 0:
                return None
            raw = self._take(self._count)
            lost, self._lost = self._lost, 0
        return AudioBlock(samples=to_mono_float32(raw, self._sample_format), overflowed=lost > 0)

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio's thread: copy the block into the ring and count what was lost, nothing else."""
        with self._ready:
            ring = self._ring
            if ring is None:
                return
            if getattr(status, "input_overflow", False):
                self._lost += 1
            capacity = ring.shape[0]
            arrived = indata.shape[0]
            block = indata[arrived - capacity:] if arrived > capacity else indata
            if arrived > capacity:
                self._lost += arrived - capacity
            size = block.shape[0]
            overflow = self._count + size - capacity
            if overflow > 0:
                # The reader fell behind: keep the newest audio, since a live command is what matters.
                self._first = (self._first + overflow) % capacity
                self._count -= overflow
                self._lost += overflow
            end = (self._first + self._count) % capacity
            head = min(size, capacity - end)
            ring[end:end + head] = block[:head]
            if size > head:
                ring[:size - head] = block[head:]
            self._count += size
            self._ready.notify()

    def _take(self, size: int) -> np.ndarray:
        import numpy as np

        capacity = self._ring.shape[0]
        head = min(size, capacity - self._first)
        parts = [self._ring[self._first:self._first + head]]
        if size > head:
            parts.append(self._ring[:size - head])
        taken = np.concatenate(parts, axis=0)
        self._first = (self._first + size) % capacity
        self._count -= size
        return taken
