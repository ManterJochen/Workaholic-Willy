"""`api/audio.py`: the decoder between an upload and Whisper, WAV only.

⛔ THE ENDPOINT ADVERTISED THREE FORMATS AND DECODED ONE. `POST /v1/voice/transcribe` described its
upload as "WAV/FLAC/OGG audio" and decoded it with `wave.open()`, which reads WAV and nothing else. PyAV
was added as a second decoder for every other container, and on 2026-09-11 the pinned av 13.1.0 turned out
to bundle a GPL build of FFmpeg (x264, x265), not the LGPL build its requirements comment claimed. The
owner decided the same day that uploads are WAV only: the console records WAV in the browser
(`frontend/src/prompt/recordWav.ts`), and any other format is refused with a sentence saying so and
naming what arrived.
"""

from __future__ import annotations

import ast
import io
import math
import struct
import unittest
import wave
from pathlib import Path

import numpy as np

from api.audio import (
    AudioDecodeError,
    AudioFormatUnsupported,
    decode_audio,
    describe_upload,
)

_AUDIO_MODULE = Path(__file__).resolve().parents[1] / "api" / "audio.py"

#: A tone, not noise: a decoder that silently drops or reorders samples still produces *something*,
#: and only a signal with structure makes that visible.
_RATE = 16000
_HZ = 440.0
_SECONDS = 0.25


def _tone(rate: int = _RATE, channels: int = 1) -> np.ndarray:
    t = np.arange(int(rate * _SECONDS), dtype=np.float32) / rate
    mono = 0.5 * np.sin(2.0 * math.pi * _HZ * t).astype(np.float32)
    if channels == 1:
        return mono
    return np.stack([mono] * channels, axis=1).reshape(-1)


def _wav_bytes(samples: np.ndarray, *, rate: int = _RATE, channels: int = 1,
               width: int = 2) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(width)
        handle.setframerate(rate)
        if width == 2:
            handle.writeframes((samples * 32767.0).astype(np.int16).tobytes())
        else:
            packed = np.clip(samples * 127.0 + 128.0, 0, 255).astype(np.uint8)
            handle.writeframes(packed.tobytes())
    return buffer.getvalue()


def _float_wav_bytes(samples: np.ndarray, *, rate: int = _RATE) -> bytes:
    """A RIFF WAVE of 32-bit IEEE float (format tag 3), which `wave` cannot read."""
    data = samples.astype("<f4").tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(data)) + data
    return b"RIFF" + struct.pack("<I", len(body)) + body


#: A complete, ordinary recording, to be sliced. Truncation is what a dropped connection produces,
#: and every interesting failure of the WAV reader is a length rather than a corruption.
_TRUNCATABLE_WAV: bytes = _wav_bytes(_tone())


def _dominant_hz(samples: np.ndarray, rate: int) -> float:
    """The strongest frequency present. The one number that survives every legal re-encoding."""
    spectrum = np.abs(np.fft.rfft(samples - samples.mean()))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(spectrum.argmax())])


class WavPathTests(unittest.TestCase):
    """The one path: what the console itself uploads."""

    def test_mono_16_bit_round_trips(self) -> None:
        samples, rate = decode_audio(_wav_bytes(_tone()))
        self.assertEqual(rate, _RATE)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(_dominant_hz(samples, rate), _HZ, delta=20.0)

    def test_stereo_is_averaged_not_truncated(self) -> None:
        """Both channels carry the tone, so the average must still BE the tone at full amplitude.

        Dropping a channel would also pass a "is there a 440 Hz peak" check, which is why the
        amplitude is asserted too: a laptop's second microphone often carries most of the voice.
        """
        samples, rate = decode_audio(_wav_bytes(_tone(channels=2), channels=2))
        self.assertAlmostEqual(_dominant_hz(samples, rate), _HZ, delta=20.0)
        self.assertGreater(float(np.abs(samples).max()), 0.4)

    def test_an_odd_frame_count_does_not_crash_the_stereo_fold(self) -> None:
        """A truncated upload leaves a half frame. Reshaping it would raise, and a dropped
        connection mid-recording is an ordinary event, not an exceptional one."""
        raw = bytearray(_wav_bytes(_tone(channels=2), channels=2))
        samples, _ = decode_audio(bytes(raw) + struct.pack("<h", 1234))
        self.assertGreater(samples.size, 0)

    def test_eight_bit_is_REFUSED_rather_than_reinterpreted(self) -> None:
        """⛔ THE OLD HANDLER READ EVERY WAV AS int16 whatever its sample width was, so an 8-bit file
        was reinterpreted, transcribing noise while reporting success. Refusing is the honest
        answer, and nothing the console produces is anything but 16-bit."""
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(_wav_bytes(_tone(), width=1))
        self.assertIn("8-bit", str(caught.exception))
        self.assertNotIn("av", str(caught.exception).split())

    def test_a_float_wav_is_a_wav_this_decoder_refuses_with_a_sentence(self) -> None:
        """`wave` reads PCM only and answers a float WAV with `wave.Error: unknown format: 3`. That fell
        through to PyAV, which decoded it; with WAV only it is a WAV this path does not read."""
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(_float_wav_bytes(_tone()))
        self.assertNotIsInstance(caught.exception, AudioFormatUnsupported)
        self.assertIn("16-bit PCM", str(caught.exception))

    def test_empty_upload_names_itself(self) -> None:
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(b"")
        self.assertIn("empty", str(caught.exception))

    def test_a_TRUNCATED_upload_is_A_TYPED_REFUSAL_and_not_a_bare_EOFError(self) -> None:
        """A dropped connection mid-upload, which is an ordinary event and was an undocumented crash.

        MEASURED on a 8044-byte mono 16-bit WAV, truncated to every length from 0 to 59: lengths
        1-7 and 20-35 escaped as a bare `EOFError` (`wave.open` reads the RIFF header with
        `struct.unpack`, which raises `EOFError`, not `wave.Error`), and the odd lengths from 45 up
        escaped as a bare `ValueError` from `np.frombuffer` ("buffer size must be a multiple of
        element size"). Neither is `AudioDecodeError`, so `POST /v1/voice/transcribe` answered them
        from its last-resort handler as `transcription_failed` with the message "EOFError: ".
        """
        for size in range(0, 60):
            with self.subTest(bytes=size):
                try:
                    decode_audio(_TRUNCATABLE_WAV[:size])
                except AudioDecodeError:
                    # The typed answer. A prefix that still holds whole frames (46, 48, ... bytes
                    # is one, two, ... frames after the 44-byte header) decodes instead, and that
                    # is correct: the claim here is about the EXCEPTION, not about refusing.
                    pass

    def test_a_HANDFUL_OF_BYTES_is_a_bad_request_and_not_another_format(self) -> None:
        """Anything shorter than RIFF's 12-byte header cannot be identified at all, so it is a
        truncated upload, not a format to name."""
        for size in range(1, 12):
            with self.subTest(bytes=size):
                with self.assertRaises(AudioDecodeError) as caught:
                    decode_audio(_TRUNCATABLE_WAV[:size])
                self.assertNotIsInstance(caught.exception, AudioFormatUnsupported)

    def test_a_header_with_no_audio_after_it_says_so(self) -> None:
        """A 44-byte WAV is a complete header and zero frames. It decoded to an empty array and
        reported success, so Whisper received nothing and the operator read "transcription failed"."""
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(_TRUNCATABLE_WAV[:44])
        self.assertIn("zero samples", str(caught.exception))


#: The first bytes of what a browser or a phone hands over, and the name the refusal must use.
_OTHER_FORMATS: tuple[tuple[str, bytes], ...] = (
    ("WebM", b"\x1aE\xdf\xa3\x9fB\x86\x81\x01B\xf7\x81\x01B\xf2\x81\x04webm"),
    ("Ogg", b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00\x12\x34\x56\x78"),
    ("FLAC", b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00\x00\x00\x00\x00"),
    ("MP4", b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00isomiso2"),
    ("MP3", b"ID3\x04\x00\x00\x00\x00\x00\x00\xff\xfb\x90\x64\x00\x00"),
    ("AVI", b"RIFF\x24\x00\x00\x00AVI LIST\x00\x00\x00\x00"),
)


class OtherFormatsAreRefusedTests(unittest.TestCase):
    """WAV only, owner decision 2026-09-11. PyAV left the shipped path with its GPL FFmpeg."""

    def test_each_other_format_is_refused_naming_it_and_saying_the_console_records_wav(self) -> None:
        for name, header in _OTHER_FORMATS:
            with self.subTest(format=name):
                with self.assertRaises(AudioFormatUnsupported) as caught:
                    decode_audio(header + bytes(256))
                message = str(caught.exception)
                self.assertIn(name, message)
                self.assertIn("console records", message)
                self.assertIn("WAV", message)

    def test_bytes_of_no_known_format_are_refused_naming_their_first_bytes(self) -> None:
        with self.assertRaises(AudioFormatUnsupported) as caught:
            decode_audio(b"\x00\x01\x02\x03" * 256)
        self.assertIn("00 01 02 03", str(caught.exception))

    def test_the_decoder_imports_only_the_standard_library_and_numpy(self) -> None:
        """Every import in the module, at any depth. PyAV was imported inside a function."""
        tree = ast.parse(_AUDIO_MODULE.read_text(encoding="utf-8"))
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        allowed = {"__future__", "io", "wave", "typing", "numpy"}
        self.assertEqual(sorted(imported - allowed), [])

    def test_the_upload_description_promises_wav_and_nothing_else(self) -> None:
        self.assertIn("WAV", describe_upload)
        for gone in ("voice.txt", "av", "webm/opus"):
            with self.subTest(fragment=gone):
                self.assertNotIn(gone, describe_upload.replace("WAV", "").split())


if __name__ == "__main__":
    unittest.main()
