"""Turn an uploaded recording into the mono float32 samples Whisper wants.

`POST /v1/voice/transcribe` accepts more than one container, and no browser records WAV:
`MediaRecorder` produces `audio/webm;codecs=opus` in Chrome and Firefox and `audio/mp4` in Safari. A
handler that only called `wave.open()`, which reads WAV and nothing else, would refuse every one of
them and surface as "transcription failed", with the real cause, a container the server cannot open,
nowhere in the message.

Two decoders, in the order that keeps the common case free:

  1. WAV via the standard library. No dependency, no install step, and it is what the console itself
     uploads, since it encodes in the browser rather than shipping whatever the platform chose. A
     console on a machine with no extras still transcribes.
  2. Everything else via PyAV. This is what makes the endpoint's own description true, and what lets
     a client other than the console, a phone app or `curl` with a voice memo, send the file it has.

PyAV's absence is reported as a missing capability, not as a broken request. The distinction is the
same one the Whisper import already draws: "this host cannot do that yet, here is the install line"
is an answer an operator can act on; "transcription failed" is not.
"""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

__all__ = ["AudioDecodeError", "AudioFormatUnsupported", "decode_audio", "describe_upload"]

#: What the endpoint may honestly claim it accepts, given both decoders.
describe_upload: Final[str] = (
    "A recording to transcribe. WAV is decoded with no extra dependency; any other container "
    "(webm/opus, mp4, ogg, flac, mp3) needs `av`, which requirements.txt installs."
)


class AudioDecodeError(ValueError):
    """The bytes open as audio and are not usable: empty, or a stream that cannot be read."""


class AudioFormatUnsupported(AudioDecodeError):
    """Not WAV, and the decoder that would handle it is not installed on this host.

    Separate from `AudioDecodeError` because the two need different answers: this one is a missing
    capability (install the decoder), the other is a bad request (send different bytes). Collapsing
    them would send an operator hunting through their microphone settings for a `pip install`.
    """


def decode_audio(raw: bytes) -> tuple["np.ndarray", int]:
    """``(mono float32 samples in [-1, 1], sample rate)`` from an uploaded recording.

    Raises `AudioFormatUnsupported` when the container needs PyAV and PyAV is absent, and
    `AudioDecodeError` when the bytes are empty or the stream carries no audio.
    """
    if not raw:
        raise AudioDecodeError("the upload was empty.")
    try:
        return _decode_wav(raw)
    except wave.Error:
        # Not a WAV. That is the ordinary case for a browser upload, so it is a fall-through and not
        # a failure: the error only matters if the second decoder is also unavailable.
        pass
    return _decode_with_av(raw)


def _decode_wav(raw: bytes) -> tuple["np.ndarray", int]:
    import numpy as np

    with wave.open(io.BytesIO(raw)) as handle:
        rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())
    if width != 2:
        # 16-bit only, and said out loud. Reading the frames as `int16` whatever `getsampwidth()`
        # returns would reinterpret an 8-bit or 24-bit WAV rather than reject it, transcribing noise
        # while reporting success. Refusing is the honest answer, and the console never produces
        # anything but 16-bit.
        raise AudioDecodeError(
            f"the WAV is {width * 8}-bit; only 16-bit PCM is decoded without the `av` extra.")
    samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
    return _to_mono(samples, channels), rate


def _decode_with_av(raw: bytes) -> tuple["np.ndarray", int]:
    import numpy as np

    try:
        import av  # noqa: PLC0415 (optional extra, imported on use)
        from av.audio.resampler import AudioResampler  # noqa: PLC0415
    except ImportError as error:
        raise AudioFormatUnsupported(
            "this recording is not 16-bit WAV, and the decoder for other formats is not installed "
            f"on this machine ({error}). pip install -r requirements.txt, or upload WAV."
        ) from error

    try:
        # `mode="r"` is for the type checker as much as the reader. Without it `av.open` is typed as
        # returning `InputContainer | OutputContainer` and `.decode` exists on only one of them, so
        # the call type-checks as an error on any host that has PyAV installed and passes silently on
        # one that does not. A check that only fires on some machines is worse than none.
        with av.open(io.BytesIO(raw), mode="r") as container:
            streams = container.streams.audio
            if not streams:
                raise AudioDecodeError("the upload carries no audio stream.")
            stream = streams[0]
            # Resample in the decoder, not afterwards. PyAV hands back whatever layout and sample
            # format the container used (planar float, interleaved s16, 5.1) and every one of those
            # would need its own reshape. Asking the resampler for mono float32 at the source rate
            # makes the rest of this function one concatenate.
            resampler = AudioResampler(format="flt", layout="mono")
            chunks: list[np.ndarray] = []
            for frame in container.decode(stream):
                for resampled in resampler.resample(frame):
                    chunks.append(resampled.to_ndarray().reshape(-1))
            for resampled in resampler.resample(None):          # flush
                chunks.append(resampled.to_ndarray().reshape(-1))
            rate = int(stream.rate or 16000)
    except AudioDecodeError:
        raise
    except Exception as error:                                   # noqa: BLE001 (report the container)
        raise AudioDecodeError(
            f"the recording could not be decoded ({type(error).__name__}: {error}).") from error

    if not chunks:
        raise AudioDecodeError("the recording decoded to zero samples.")
    return np.concatenate(chunks).astype(np.float32), rate


def _to_mono(samples: "np.ndarray", channels: int) -> "np.ndarray":
    if channels <= 1:
        return samples
    # Averaging rather than dropping a channel: a laptop's second microphone often carries most of
    # the voice, and picking one at random would transcribe the quiet side.
    usable = (samples.size // channels) * channels
    return samples[:usable].reshape(-1, channels).mean(axis=1)
