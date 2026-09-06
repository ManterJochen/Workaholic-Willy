"""`api/audio.py` -- the decoder that stands between a browser's microphone and Whisper.

⛔ THE DEFECT THESE TESTS EXIST FOR. `POST /v1/voice/transcribe` described its upload as
"WAV/FLAC/OGG audio" and decoded it with `wave.open()`, which reads WAV and nothing else. Nothing had
ever uploaded to it, so the gap was invisible -- and the first real client would have hit it every
time, because **no browser records WAV**: `MediaRecorder` gives `audio/webm;codecs=opus` in Chrome and
Firefox and `audio/mp4` in Safari. The operator would have seen "transcription failed" with the
actual cause -- a container the server cannot open -- nowhere in the message.

⚠ THE WEBM CASES BUILD REAL WEBM, not a fixture blob and not a mock. A test that asserts the decoder
handles Opus by feeding it bytes a mock produced proves the mock. These encode a known tone through
PyAV and decode it back, so a container change, a layout change or a resampler regression fails here
rather than in front of an operator.
"""

from __future__ import annotations

import importlib.util
import io
import math
import struct
import unittest
import wave

import numpy as np

from api.audio import (
    AudioDecodeError,
    AudioFormatUnsupported,
    decode_audio,
    describe_upload,
)


def _av_is_loadable() -> tuple[bool, str]:
    """Whether PyAV can be IMPORTED, not merely found.

    ⚠ `find_spec` answers "is it installed", and on Windows that is a different question. Smart App
    Control refuses freshly-published unsigned binaries until they accumulate reputation, so `av`
    resolves and then `import av` dies with `DLL load failed ... Eine Anwendungssteuerungsrichtlinie
    hat diese Datei blockiert`. That is a fact about the box, not a defect here -- and CI does not
    install `requirements/voice.txt` at all, so this skip is the ordinary case there.
    """
    if importlib.util.find_spec("av") is None:
        return (False, "av is not installed (it is an optional extra: requirements/voice.txt)")
    try:
        import av  # noqa: F401,PLC0415
    except Exception as error:                                   # noqa: BLE001 - report, not fail
        return (False, f"av is installed but will not load: {type(error).__name__}: {error}")
    return (True, "")


_HAS_AV, _WHY_NOT = _av_is_loadable()

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


def _dominant_hz(samples: np.ndarray, rate: int) -> float:
    """The strongest frequency present. The one number that survives every legal re-encoding."""
    spectrum = np.abs(np.fft.rfft(samples - samples.mean()))
    return float(np.fft.rfftfreq(samples.size, 1.0 / rate)[int(spectrum.argmax())])


class WavPathTests(unittest.TestCase):
    """The dependency-free path -- what the console itself uploads."""

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
        was reinterpreted -- transcribing noise while reporting success. Refusing is the honest
        answer, and nothing the console produces is anything but 16-bit."""
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(_wav_bytes(_tone(), width=1))
        self.assertIn("8-bit", str(caught.exception))

    def test_empty_upload_names_itself(self) -> None:
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(b"")
        self.assertIn("empty", str(caught.exception))


class BrowserFormatTests(unittest.TestCase):
    """What a browser actually hands over. Skipped where the optional decoder is absent."""

    @staticmethod
    def _encode(container_format: str, codec: str, samples: np.ndarray, rate: int) -> bytes:
        import av

        buffer = io.BytesIO()
        with av.open(buffer, mode="w", format=container_format) as container:
            stream = container.add_stream(codec, rate=rate)
            frame = av.AudioFrame.from_ndarray(
                samples.reshape(1, -1).astype(np.float32), format="flt", layout="mono")
            frame.rate = rate
            for packet in stream.encode(frame):
                container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)
        return buffer.getvalue()

    @unittest.skipUnless(_HAS_AV, _WHY_NOT)
    def test_webm_opus_decodes(self) -> None:
        """What Chrome and Firefox record. Opus is lossy and resamples to 48 kHz internally, so the
        assertion is the tone that survives, not the samples."""
        raw = self._encode("webm", "libopus", _tone(rate=48000), 48000)
        samples, rate = decode_audio(raw)
        self.assertGreater(samples.size, 0)
        self.assertEqual(samples.dtype, np.float32)
        self.assertAlmostEqual(_dominant_hz(samples, rate), _HZ, delta=40.0)

    @unittest.skipUnless(_HAS_AV, _WHY_NOT)
    def test_ogg_vorbis_decodes(self) -> None:
        """One of the three the endpoint's description has always claimed."""
        raw = self._encode("ogg", "libvorbis", _tone(rate=44100), 44100)
        samples, rate = decode_audio(raw)
        self.assertAlmostEqual(_dominant_hz(samples, rate), _HZ, delta=40.0)

    @unittest.skipUnless(_HAS_AV, _WHY_NOT)
    def test_flac_decodes(self) -> None:
        raw = self._encode("flac", "flac", _tone(), _RATE)
        samples, rate = decode_audio(raw)
        self.assertAlmostEqual(_dominant_hz(samples, rate), _HZ, delta=20.0)

    @unittest.skipUnless(_HAS_AV, _WHY_NOT)
    def test_bytes_that_are_not_audio_are_a_DECODE_error(self) -> None:
        """Not `AudioFormatUnsupported`: the decoder is present and the bytes are simply wrong. The
        two map to 422 and 501 respectively, so confusing them sends an operator to `pip`."""
        with self.assertRaises(AudioDecodeError) as caught:
            decode_audio(b"\x00\x01\x02\x03" * 256)
        self.assertNotIsInstance(caught.exception, AudioFormatUnsupported)


class MissingDecoderTests(unittest.TestCase):
    """The answer on a host that has no optional extra -- which is every CI runner."""

    def test_a_missing_decoder_is_a_CAPABILITY_error_naming_the_fix(self) -> None:
        """⚠ `AudioFormatUnsupported`, not `AudioDecodeError`, and the endpoint maps it to 501 rather
        than 422. The operator did nothing wrong -- their browser recorded webm because that is what
        browsers do -- and an "unprocessable" answer would send them hunting through microphone
        settings for something that is a `pip install`."""
        import builtins

        real_import = builtins.__import__

        def refuse_av(name: str, *args: object, **kwargs: object) -> object:
            if name == "av":
                raise ImportError("No module named 'av'")
            return real_import(name, *args, **kwargs)                    # type: ignore[arg-type]

        builtins.__import__ = refuse_av
        try:
            with self.assertRaises(AudioFormatUnsupported) as caught:
                decode_audio(b"\x1aE\xdf\xa3not-a-wav")
        finally:
            builtins.__import__ = real_import
        message = str(caught.exception)
        self.assertIn("requirements/voice.txt", message)
        self.assertIn("WAV", message)

    def test_the_upload_description_no_longer_promises_what_it_cannot_do(self) -> None:
        """The old one said "WAV/FLAC/OGG" flatly. It now says which needs nothing and which needs
        the extra -- the whole defect in one sentence, on the endpoint that had it."""
        self.assertIn("WAV", describe_upload)
        self.assertIn("voice.txt", describe_upload)
        self.assertIn("webm", describe_upload)


if __name__ == "__main__":
    unittest.main()
