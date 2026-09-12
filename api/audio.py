"""Turn an uploaded recording into the mono float32 samples Whisper wants. WAV only.

`POST /v1/voice/transcribe` accepts one format, and the endpoint says so. No browser records WAV:
`MediaRecorder` produces `audio/webm;codecs=opus` in Chrome and Firefox and `audio/mp4` in Safari, so
the console encodes 16-bit PCM WAV in the browser (`frontend/src/prompt/recordWav.ts`) rather than
shipping whatever the platform chose.

A second decoder for every other container, PyAV, is gone: the pinned av 13.1.0 bundles a GPL build of
FFmpeg, with x264 and x265 beside it, not the LGPL build it was admitted under. Uploads are WAV only
and decoded with the standard library. Any other format is refused with a sentence saying the console
records WAV and naming what arrived, as `AudioFormatUnsupported`: a format this server will never
decode, which the endpoint answers with 415.
"""

from __future__ import annotations

import io
import wave
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:  # pragma: no cover (typing only)
    import numpy as np

__all__ = ["AudioDecodeError", "AudioFormatUnsupported", "decode_audio", "describe_upload"]

#: What the endpoint may honestly claim it accepts.
describe_upload: Final[str] = (
    "A recording to transcribe, as 16-bit PCM WAV: the format the operator console records. Any other "
    "format is refused with 415."
)

#: The shortest header of a RIFF WAVE: ``"RIFF" + size + "WAVE"``. Below it there is nothing to identify,
#: which is why it is a truncated upload rather than a format to name.
_SHORTEST_HEADER: Final[int] = 12

#: What the first bytes of an upload say it is, for the refusal: (offset, magic, name). The magic numbers
#: are each container's own: EBML for WebM and Matroska, Ogg's capture pattern, FLAC's stream marker, the
#: ISO base media `ftyp` box at offset 4 for MP4 and M4A, and an ID3 tag for MP3.
_SIGNATURES: Final[tuple[tuple[int, bytes, str], ...]] = (
    (0, b"\x1aE\xdf\xa3", "a WebM or Matroska container"),
    (0, b"OggS", "an Ogg container"),
    (0, b"fLaC", "FLAC"),
    (4, b"ftyp", "an MP4 container"),
    (0, b"ID3", "MP3"),
)


class AudioDecodeError(ValueError):
    """The upload is empty, truncated, or a WAV this path cannot use (not 16-bit PCM, or no samples)."""


class AudioFormatUnsupported(AudioDecodeError):
    """The upload is not WAV. The console records WAV, and nothing else is decoded here.

    Separate from `AudioDecodeError` because the answer differs: this one is a format the server will
    never decode (415), the other a WAV that is broken or cut short (422).
    """


def decode_audio(raw: bytes) -> tuple["np.ndarray", int]:
    """``(mono float32 samples in [-1, 1], sample rate)`` from an uploaded 16-bit PCM WAV.

    Raises `AudioFormatUnsupported` when the bytes are not WAV, and `AudioDecodeError` when they are
    empty, truncated, not 16-bit PCM, or carry no samples. Nothing else: a caller that catches those two
    has caught everything this function answers with.
    """
    if not raw:
        raise AudioDecodeError("the upload was empty.")
    if len(raw) < _SHORTEST_HEADER:
        # A bad request, not a format. Below RIFF's 12-byte header there is nothing to identify, and
        # naming a format for a handful of bytes would be a guess. Measured: a connection dropped after
        # the first packet delivers 1 to 7 bytes.
        raise AudioDecodeError(
            f"the upload is {len(raw)} byte(s); a WAV header alone is {_SHORTEST_HEADER}, so these bytes "
            f"are a truncated upload rather than a recording. Record again."
        )
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise AudioFormatUnsupported(
            f"the upload is {_describe(raw)}, and only WAV is decoded here: the operator console records "
            f"16-bit PCM WAV. Upload WAV."
        )
    try:
        return _decode_wav(raw)
    except EOFError as error:
        # `EOFError` is not a `wave.Error`. `wave.open` parses the RIFF header with `struct.unpack`
        # and answers a short read with a bare `EOFError`. Measured on an 8044-byte WAV truncated to
        # every length from 0 to 59: lengths 20 to 35 raised it.
        raise AudioDecodeError(
            f"the WAV ends inside its header after {len(raw)} byte(s); the upload was cut short. "
            f"Record again."
        ) from error
    except wave.Error as error:
        raise AudioDecodeError(
            f"the WAV cannot be read as 16-bit PCM ({error}); the console records 16-bit PCM WAV."
        ) from error


def _describe(raw: bytes) -> str:
    """What the first bytes of a non-WAV upload say it is, as a phrase for the refusal."""
    for offset, magic, name in _SIGNATURES:
        if raw[offset:offset + len(magic)] == magic:
            return name
    if raw[:4] == b"RIFF":
        form = raw[8:12].decode("ascii", "replace").strip()
        return f"a RIFF container holding {form!r} rather than WAVE"
    if raw[0] == 0xFF and raw[1] & 0xE0 == 0xE0:
        return "MP3 (an MPEG audio frame)"
    return f"bytes that begin {raw[:8].hex(' ')}, which is no audio format the console knows"


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
        # while reporting success.
        raise AudioDecodeError(
            f"the WAV is {width * 8}-bit; only 16-bit PCM is decoded, which is what the console records."
        )
    # A trailing half-sample is dropped, not raised on. `np.frombuffer` refuses a buffer whose size is
    # not a multiple of the element size, and that ValueError is not an `AudioDecodeError`. A recording
    # cut mid-sample is an ordinary dropped connection, and one abandoned sample is 1/16000 s.
    whole = frames[: len(frames) - len(frames) % 2]
    samples = np.frombuffer(whole, dtype=np.int16).astype(np.float32) / 32768.0
    if samples.size == 0:
        # A 44-byte upload is a complete header and no frames. It used to return an empty array and
        # report success, so Whisper was handed nothing and the operator read "transcription failed".
        raise AudioDecodeError("the recording decoded to zero samples.")
    return _to_mono(samples, channels), rate


def _to_mono(samples: "np.ndarray", channels: int) -> "np.ndarray":
    if channels <= 1:
        return samples
    # Averaging rather than dropping a channel: a laptop's second microphone often carries most of the
    # voice, and picking one at random would transcribe the quiet side.
    usable = (samples.size // channels) * channels
    return samples[:usable].reshape(-1, channels).mean(axis=1)
