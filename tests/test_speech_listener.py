"""The microphone as a continuous stream, cut into utterances, and `listen()` handing over the first one.

The only microphone path was `WhisperSpeechToText.listen_once`, and it had no caller. It opened a
`RawInputStream`, whose callback receives a plain buffer; the callback called `flatten()` on it, so it
raised on the first block and PortAudio never called it again. It divided by 32768 whatever the sample
format was, cut the recording into fixed 3 s windows, and lower-cased the text the upload path did not.
Red evidence against that code, captured before it was retired:

    AttributeError: 'memoryview' object has no attribute 'flatten'
    AssertionError: np.float32(65536.0) not less than or equal to 1.0
    AssertionError: 'pick the red cube' != 'Pick the red cube'

What replaces it: `MicrophoneSource` captures continuously through an `InputStream`, a
`VoiceActivityDetector` scores each window, `UtteranceCutter` decides where an utterance starts and ends,
and `Listener.listen()` returns the first utterance that closes as a frozen `Utterance` carrying the
engine's `Transcript`. Every test here uses a scripted source, a loudness double for the detector and a
recording engine; the sound card is a fake `sounddevice` module. The product detector,
`SileroVoiceActivityDetector`, has its own tests in `tests/test_speech_silero.py`, and `from_config`
builds it when no detector is chosen.
"""

from __future__ import annotations

import builtins
import json
import sys
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml
from pydantic import ValidationError

from src.config.schema.models.models_schema import SpeechToTextConfig
from tests._speech_fakes import (
    SAMPLERATE,
    FakeClock,
    LoudnessDetector,
    RecordingEngine,
    ScriptedSource,
    blocks_of,
    fake_sounddevice,
    silence,
    tone,
    whisper_parts,
)

_SHIPPED_STT = Path(__file__).resolve().parents[1] / "config" / "models" / "stt.yaml"
_WINDOW = 512
_WINDOW_S = _WINDOW / SAMPLERATE


def _shipped_stt() -> dict[str, Any]:
    block = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]
    assert isinstance(block, dict)
    return block


def _config(**overrides: Any) -> SpeechToTextConfig:
    return SpeechToTextConfig.model_validate({**_shipped_stt(), **overrides})


def _listener(source: Any, engine: Any = None, **kwargs: Any) -> Any:
    from src.models.speech.listener import Listener

    return Listener.from_parts(
        source=source, detector=LoudnessDetector(), engine=engine or RecordingEngine(),
        clock=source._clock, **kwargs,
    )


class TheMicrophoneKeysTests(unittest.TestCase):
    def test_chunk_duration_left_with_the_fixed_window_path(self) -> None:
        self.assertNotIn("chunk_duration", _shipped_stt())
        with self.assertRaises(ValidationError) as caught:
            _config(chunk_duration=3)
        self.assertIn("chunk_duration", str(caught.exception))

    def test_the_sample_format_is_one_the_stream_can_scale(self) -> None:
        for sample_format in ("int16", "int32", "float32"):
            with self.subTest(dtype=sample_format):
                self.assertEqual(_config(dtype=sample_format).dtype, sample_format)
        for sample_format in ("int61", "uint8", "float64"):
            with self.subTest(dtype=sample_format):
                with self.assertRaises(ValidationError):
                    _config(dtype=sample_format)

    def test_a_stream_needs_a_channel_and_a_rate(self) -> None:
        for overrides in ({"channels": 0}, {"samplerate": 0}, {"blocksize": -1}):
            with self.subTest(**overrides):
                with self.assertRaises(ValidationError):
                    _config(**overrides)


class EveryBlockIsScaledByItsOwnFormatTests(unittest.TestCase):
    def test_every_sample_format_lands_in_the_unit_range(self) -> None:
        from src.models.speech.capture import to_mono_float32

        cases = (
            ("int16", np.array([[32767], [-32768], [16384]], dtype=np.int16), (32767 / 32768, -1.0, 0.5)),
            ("int32", np.array([[2**31 - 1], [-(2**31)], [2**30]], dtype=np.int32),
             ((2**31 - 1) / 2**31, -1.0, 0.5)),
            ("float32", np.array([[0.25], [-0.5], [1.0]], dtype=np.float32), (0.25, -0.5, 1.0)),
        )
        for sample_format, block, expected in cases:
            with self.subTest(dtype=sample_format):
                mono = to_mono_float32(block, sample_format)
                self.assertEqual(mono.dtype, np.float32)
                np.testing.assert_allclose(mono, np.array(expected, dtype=np.float32), rtol=1e-6)

    def test_channels_are_averaged_not_interleaved(self) -> None:
        from src.models.speech.capture import to_mono_float32

        stereo = np.array([[16384, 0], [0, -16384]], dtype=np.int16)
        np.testing.assert_allclose(to_mono_float32(stereo, "int16"), [0.25, -0.25])

    def test_a_block_in_another_format_than_the_stream_was_opened_with_is_refused(self) -> None:
        from src.models.speech.capture import to_mono_float32

        with self.assertRaises(ValueError) as caught:
            to_mono_float32(np.zeros((4, 1), dtype=np.int32), "int16")
        self.assertIn("int16", str(caught.exception))


class TheMicrophoneSourceTests(unittest.TestCase):
    def _open(self, module: Any, **parts: Any) -> Any:
        from src.models.speech.capture import MicrophoneSource

        source = MicrophoneSource.from_parts(**parts)
        with patch.dict(sys.modules, {"sounddevice": module}):
            source.start()
        self.addCleanup(source.stop)
        return source

    def test_the_stream_is_an_input_stream_opened_with_the_configured_keys(self) -> None:
        from src.models.speech.capture import MicrophoneSource

        module = fake_sounddevice()
        source = MicrophoneSource.from_config(config=_config())
        with patch.dict(sys.modules, {"sounddevice": module}):
            source.start()
        self.addCleanup(source.stop)
        (stream,) = module.opened  # type: ignore[attr-defined]
        config = _config()
        self.assertEqual(
            {key: stream.kwargs[key] for key in ("samplerate", "blocksize", "channels", "dtype")},
            {"samplerate": config.samplerate, "blocksize": config.blocksize,
             "channels": config.channels, "dtype": config.dtype},
        )
        self.assertTrue(stream.active)
        self.assertEqual(source.samplerate, config.samplerate)

    def test_a_block_the_callback_receives_comes_back_as_mono_float32(self) -> None:
        module = fake_sounddevice()
        source = self._open(module, channels=2, sample_format="int32")
        (stream,) = module.opened  # type: ignore[attr-defined]
        stream.feed(np.array([[2**30, 0], [0, -(2**30)], [2**30, 2**30]], dtype=np.int32))
        block = source.read(timeout_s=0.0)
        self.assertIsNotNone(block)
        self.assertEqual(block.samples.dtype, np.float32)
        np.testing.assert_allclose(block.samples, [0.25, -0.25, 0.5])
        self.assertFalse(block.overflowed)

    def test_an_overflow_is_reported_once_on_the_next_block(self) -> None:
        module = fake_sounddevice()
        source = self._open(module)
        (stream,) = module.opened  # type: ignore[attr-defined]
        stream.feed(np.zeros((4, 1), dtype=np.int16), overflow=True)
        self.assertTrue(source.read(timeout_s=0.0).overflowed)
        stream.feed(np.zeros((4, 1), dtype=np.int16))
        self.assertFalse(source.read(timeout_s=0.0).overflowed)

    def test_a_full_ring_keeps_the_newest_audio_and_says_it_dropped_some(self) -> None:
        module = fake_sounddevice()
        source = self._open(module, samplerate=16, buffer_s=0.5)
        (stream,) = module.opened  # type: ignore[attr-defined]
        stream.feed(np.arange(10, dtype=np.int16).reshape(-1, 1))
        block = source.read(timeout_s=0.0)
        np.testing.assert_allclose(block.samples * 32768.0, np.arange(2, 10))
        self.assertTrue(block.overflowed)

    def test_audio_buffered_before_listening_can_be_discarded(self) -> None:
        module = fake_sounddevice()
        source = self._open(module)
        (stream,) = module.opened  # type: ignore[attr-defined]
        stream.feed(np.zeros((1000, 1), dtype=np.int16))
        self.assertEqual(source.discard_buffered(), 1000)
        self.assertIsNone(source.read(timeout_s=0.0))

    def test_a_wasapi_device_is_opened_with_the_mixer_converting_the_rate(self) -> None:
        """A rate the WASAPI mixer does not run at is refused as 'Invalid device' unless auto_convert is
        on (PortAudio issue 314)."""
        for hostapi, converts in (("Windows WASAPI", True), ("MME", False)):
            with self.subTest(hostapi=hostapi):
                module = fake_sounddevice(hostapi=hostapi)
                self._open(module)
                (stream,) = module.opened  # type: ignore[attr-defined]
                settings = stream.kwargs.get("extra_settings")
                self.assertEqual(settings is not None and settings.auto_convert, converts)

    def test_reading_a_microphone_that_is_not_open_is_refused_with_a_sentence(self) -> None:
        from src.models.speech.capture import MicrophoneSource

        with self.assertRaises(RuntimeError) as caught:
            MicrophoneSource.from_parts().read(timeout_s=0.0)
        self.assertIn("start()", str(caught.exception))

    def test_a_missing_portaudio_is_named_with_the_file_that_installs_sounddevice(self) -> None:
        from src.models.speech.capture import MicrophoneSource
        from src.models.speech.engine import SpeechStackUnavailable

        real_import = builtins.__import__

        def no_portaudio(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sounddevice":
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, "__import__", no_portaudio):
            with self.assertRaises(SpeechStackUnavailable) as caught:
                MicrophoneSource.from_parts().start()
        self.assertIn("requirements.txt", str(caught.exception))
        self.assertIn("PortAudio", str(caught.exception))


class TheCutterTests(unittest.TestCase):
    """The endpointing state machine alone: windows and their speech probabilities in, utterances out."""

    def _cutter(self, **seconds: float) -> Any:
        from src.models.speech.endpointing import Endpointing, UtteranceCutter

        windows = {"pre_roll_s": 3, "trailing_silence_s": 10, "min_speech_s": 5, "max_utterance_s": 100}
        windows.update({key: value for key, value in seconds.items()})
        endpointing = Endpointing(**{key: value * _WINDOW_S for key, value in windows.items()})
        return UtteranceCutter(endpointing=endpointing, window_samples=_WINDOW, samplerate=SAMPLERATE)

    @staticmethod
    def _window(index: int) -> np.ndarray:
        return np.full(_WINDOW, index / 1000.0, dtype=np.float32)

    def _push(self, cutter: Any, probabilities: list[float]) -> list[tuple[int, Any]]:
        return [(i, cut) for i, p in enumerate(probabilities)
                if (cut := cutter.push(self._window(i), p)) is not None]

    def test_speech_then_trailing_silence_closes_one_utterance_with_its_pre_roll(self) -> None:
        from src.models.speech.endpointing import UtteranceEnd

        cuts = self._push(self._cutter(), [0.0] * 10 + [1.0] * 20 + [0.0] * 20)
        self.assertEqual(len(cuts), 1)
        index, cut = cuts[0]
        self.assertEqual(index, 39, "closed on the tenth silent window")
        self.assertEqual(cut.ended_by, UtteranceEnd.SILENCE)
        self.assertEqual(cut.samples.size, (3 + 20 + 10) * _WINDOW)
        self.assertAlmostEqual(float(cut.samples[0]), 0.007, places=6, msg="the pre-roll starts 3 windows early")

    def test_a_dip_between_offset_and_onset_stays_inside_the_utterance(self) -> None:
        cuts = self._push(self._cutter(), [0.9] * 6 + [0.4] * 15 + [0.1] * 10)
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0][1].samples.size, (6 + 15 + 10) * _WINDOW)

    def test_a_click_shorter_than_the_minimum_speech_is_discarded(self) -> None:
        cutter = self._cutter()
        self.assertEqual(self._push(cutter, [1.0] * 2 + [0.0] * 12), [])
        self.assertFalse(cutter.in_speech)

    def test_the_maximum_length_cuts_an_utterance_and_says_so(self) -> None:
        from src.models.speech.endpointing import UtteranceEnd

        cuts = self._push(self._cutter(), [1.0] * 150)
        self.assertEqual(cuts[0][0], 99)
        self.assertEqual(cuts[0][1].ended_by, UtteranceEnd.MAX_LENGTH)
        self.assertEqual(cuts[0][1].samples.size, 100 * _WINDOW)

    def test_finish_hands_over_an_open_utterance_that_holds_enough_speech(self) -> None:
        from src.models.speech.endpointing import UtteranceEnd

        cutter = self._cutter()
        self._push(cutter, [1.0] * 8)
        cut = cutter.finish()
        self.assertEqual(cut.ended_by, UtteranceEnd.SOURCE_END)
        self.assertEqual(cut.samples.size, 8 * _WINDOW)

    def test_endpointing_refuses_thresholds_that_cannot_mean_anything(self) -> None:
        from src.models.speech.endpointing import Endpointing

        for bad in ({"onset": 0.3, "offset": 0.5}, {"onset": 1.5}, {"trailing_silence_s": 0.0},
                    {"min_speech_s": 20.0, "max_utterance_s": 15.0}):
            with self.subTest(**bad):
                with self.assertRaises(ValueError):
                    Endpointing(**bad)


class ListenTests(unittest.TestCase):
    def test_listen_returns_the_first_complete_utterance_transcribed(self) -> None:
        from src.models.speech.endpointing import UtteranceEnd
        from src.models.speech.listener import ListenOutcome

        clock = FakeClock()
        source = ScriptedSource(
            blocks_of(silence(0.5), tone(1.0), silence(1.0), tone(1.0), silence(1.0)), clock=clock
        )
        engine = RecordingEngine()
        heard = _listener(source, engine).listen()
        self.assertEqual(heard.outcome, ListenOutcome.HEARD)
        self.assertEqual(heard.ended_by, UtteranceEnd.SILENCE)
        self.assertEqual(len(engine.calls), 1, "one utterance, one decode")
        self.assertEqual(heard.transcript.text, "Nimm den roten Würfel")
        self.assertEqual(engine.calls[0][1], SAMPLERATE)
        self.assertTrue(1.4 < heard.speech_s < 2.0, heard.speech_s)
        self.assertTrue(1.9 <= heard.waited_s <= 2.3, heard.waited_s)
        self.assertTrue(source.remaining, "the second utterance is left for the next listen()")

    def test_the_microphone_path_returns_the_same_text_as_the_upload_path(self) -> None:
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        processor, model = whisper_parts(text="  Pick the red cube  ")
        engine = WhisperTransformersEngine.from_parts(
            source="org/whisper", device="cpu", processor=processor, model=model
        )
        upload = engine.transcribe(tone(1.0), samplerate=SAMPLERATE)
        clock = FakeClock()
        source = ScriptedSource(blocks_of(tone(1.0), silence(1.0)), clock=clock)
        heard = _listener(source, engine).listen()
        self.assertEqual(heard.transcript.text, upload.text)
        self.assertEqual(heard.transcript.text, "Pick the red cube", "never lower-cased")

    def test_listen_is_bounded_and_says_it_timed_out(self) -> None:
        """The property the retired poll helper pinned: silence, or a microphone that delivers nothing,
        never keeps listen() waiting past its bound."""
        from src.models.speech.listener import ListenOutcome

        for label, kwargs in (("silence keeps arriving", {"endless": silence(0.1)}),
                              ("nothing arrives", {"silent_after": True})):
            with self.subTest(source=label):
                clock = FakeClock()
                engine = RecordingEngine()
                heard = _listener(ScriptedSource([], clock=clock, **kwargs), engine).listen(timeout_s=2.0)
                self.assertEqual(heard.outcome, ListenOutcome.TIMED_OUT)
                self.assertIsNone(heard.transcript)
                self.assertGreaterEqual(heard.waited_s, 2.0)
                self.assertLess(heard.waited_s, 2.2)
                self.assertEqual(engine.calls, [])

    def test_the_bound_left_unset_is_the_listeners_own(self) -> None:
        from src.models.speech.listener import ListenOutcome

        clock = FakeClock()
        source = ScriptedSource([], clock=clock, endless=silence(0.1))
        heard = _listener(source, timeout_s=3.0).listen()
        self.assertEqual((heard.outcome, heard.timeout_s), (ListenOutcome.TIMED_OUT, 3.0))

    def test_none_is_a_chosen_bound_meaning_wait_for_the_source(self) -> None:
        from src.models.speech.listener import ListenOutcome

        clock = FakeClock()
        source = ScriptedSource(blocks_of(silence(40.0)), clock=clock)
        heard = _listener(source, timeout_s=3.0).listen(timeout_s=None)
        self.assertEqual(heard.outcome, ListenOutcome.SOURCE_ENDED)
        self.assertIsNone(heard.timeout_s)
        self.assertGreaterEqual(heard.waited_s, 40.0)

    def test_a_source_that_ends_mid_utterance_still_hands_over_what_was_said(self) -> None:
        from src.models.speech.endpointing import UtteranceEnd
        from src.models.speech.listener import ListenOutcome

        clock = FakeClock()
        heard = _listener(ScriptedSource(blocks_of(silence(0.2), tone(1.0)), clock=clock)).listen()
        self.assertEqual((heard.outcome, heard.ended_by), (ListenOutcome.HEARD, UtteranceEnd.SOURCE_END))

    def test_a_source_that_ends_with_no_speech_says_so(self) -> None:
        from src.models.speech.listener import ListenOutcome

        clock = FakeClock()
        engine = RecordingEngine()
        heard = _listener(ScriptedSource(blocks_of(silence(1.0)), clock=clock), engine).listen()
        self.assertEqual(heard.outcome, ListenOutcome.SOURCE_ENDED)
        self.assertIsNone(heard.transcript)
        self.assertEqual(engine.calls, [])

    def test_an_overflow_while_listening_is_carried_in_the_report(self) -> None:
        clock = FakeClock()
        source = ScriptedSource(blocks_of(tone(1.0), silence(1.0)), clock=clock,
                                overflow_at=frozenset({3}))
        heard = _listener(source).listen()
        self.assertTrue(heard.overflowed)
        self.assertIn("dropped", heard.render())

    def test_a_detector_at_another_rate_is_refused_with_a_sentence(self) -> None:
        from src.models.speech.listener import Listener

        detector = LoudnessDetector()
        detector.samplerate = 8000
        with self.assertRaises(ValueError) as caught:
            Listener.from_parts(source=ScriptedSource([], clock=FakeClock()), detector=detector,
                                engine=RecordingEngine())
        self.assertIn("8000", str(caught.exception))

    def test_the_listener_opens_and_closes_its_source(self) -> None:
        clock = FakeClock()
        source = ScriptedSource(blocks_of(tone(1.0), silence(1.0)), clock=clock)
        listener = _listener(source)
        with listener:
            self.assertEqual((source.started, source.stopped), (1, 0))
        self.assertEqual((source.started, source.stopped), (1, 1))

    def test_from_config_opens_the_cell_microphone_with_the_speech_keys(self) -> None:
        from src.models.speech.capture import MicrophoneSource
        from src.models.speech.listener import Listener

        listener = Listener.from_config(
            config=_config(), engine=RecordingEngine(), detector=LoudnessDetector()
        )
        self.assertIsInstance(listener.source, MicrophoneSource)
        self.assertEqual(listener.source.samplerate, _config().samplerate)

    def test_from_config_builds_the_silero_detector_when_none_is_chosen(self) -> None:
        import torch

        from src.models.speech.listener import Listener
        from src.models.speech.silero import SileroVoiceActivityDetector

        with patch.object(torch.jit, "load") as load:
            listener = Listener.from_config(config=_config(), engine=RecordingEngine())
        load.assert_not_called()
        self.assertIsInstance(listener.detector, SileroVoiceActivityDetector)
        self.assertEqual(listener.detector.model_path, Path(_config().vad_model_path))


class TheUtteranceReportTests(unittest.TestCase):
    def _heard(self) -> Any:
        clock = FakeClock()
        return _listener(ScriptedSource(blocks_of(tone(1.0), silence(1.0)), clock=clock)).listen()

    def test_to_dict_is_plain_data_carrying_the_transcript(self) -> None:
        data = self._heard().to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)
        self.assertEqual(data["outcome"], "heard")
        self.assertEqual(data["ended_by"], "silence")
        self.assertEqual(data["transcript"]["text"], "Nimm den roten Würfel")

    def test_render_is_ascii_and_ends_without_a_newline(self) -> None:
        text = self._heard().render()
        self.assertTrue(text.isascii(), text)
        self.assertFalse(text.endswith("\n"))
        self.assertIn("W\\xfcrfel", text, "the transcript is rendered inside the utterance")

    def test_a_timed_out_listen_renders_what_it_waited_for(self) -> None:
        clock = FakeClock()
        heard = _listener(ScriptedSource([], clock=clock, silent_after=True)).listen(timeout_s=2.0)
        self.assertIn("Nothing heard within 2.0 s", heard.render())
        self.assertIsNone(heard.to_dict()["transcript"])

    def test_it_is_frozen_and_both_halves_of_the_report_contract(self) -> None:
        import dataclasses

        from src.contracts import Rendered, Structured

        heard = self._heard()
        self.assertIsInstance(heard, Rendered)
        self.assertIsInstance(heard, Structured)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            heard.outcome = None


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
