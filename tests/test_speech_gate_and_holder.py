"""An upload is proposed only when it holds speech, and the process keeps one engine per speech section.

Whisper answers a recording without speech with a word. Through the console on 2026-09-11, whisper-small
answered 0.2 s of a tone with ``{"text":"you","language":"en",...}``, and that word would have landed in
the prompt box. `SpeechGate` runs the voice detector over the recording first, through the same
`UtteranceCutter` the microphone path uses, and a recording in which no utterance closes comes back as an
empty `Proposal` with a reason; Whisper is not asked. A recording longer than Whisper's 30 s window is
refused with a sentence before either model runs.

`SpeechHolder` is the one engine the console used to keep in a private router global, now in the library
where a `Listener` caller can take the same engine.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from src.config.schema.models.models_schema import SpeechToTextConfig
from tests._speech_fakes import SAMPLERATE, LoudnessDetector, RecordingEngine, silence, tone

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


def _config(**overrides: Any) -> SpeechToTextConfig:
    block = yaml.safe_load((_SHIPPED / "models" / "stt.yaml").read_text(encoding="utf-8"))["stt"]
    return SpeechToTextConfig.model_validate({**block, **overrides})


def _gate(detector: Any = None) -> Any:
    from src.models.speech.gate import SpeechGate

    return SpeechGate.from_parts(detector=detector or LoudnessDetector())


def _held(engine: Any = None, detector: Any = None) -> Any:
    from src.models.speech.holder import HeldSpeech

    return HeldSpeech(config=_config(), engine=engine or RecordingEngine(), gate=_gate(detector))


class TheGateTests(unittest.TestCase):
    def test_a_recording_without_speech_is_not_heard_and_says_so(self) -> None:
        check = _gate().check(silence(2.0), samplerate=SAMPLERATE)
        self.assertFalse(check.heard_speech)
        self.assertEqual(check.peak_probability, 0.0)
        self.assertIn("no speech", check.render().lower())

    def test_speech_long_enough_to_be_an_utterance_is_heard(self) -> None:
        audio = np.concatenate((silence(0.3), tone(0.6), silence(0.3)))
        check = _gate().check(audio, samplerate=SAMPLERATE)
        self.assertTrue(check.heard_speech)
        self.assertEqual(check.peak_probability, 1.0)

    def test_a_click_shorter_than_the_minimum_speech_is_not_heard(self) -> None:
        """The same rule the microphone path drops a cough by: `Endpointing.min_speech_s`."""
        audio = np.concatenate((silence(0.5), tone(0.1), silence(0.5)))
        self.assertFalse(_gate().check(audio, samplerate=SAMPLERATE).heard_speech)

    def test_stereo_at_another_rate_reaches_the_detector_as_its_own_windows(self) -> None:
        detector = LoudnessDetector()
        stereo = np.stack((tone(1.0), np.zeros(SAMPLERATE, dtype=np.float32)), axis=1)
        at_48k = np.repeat(stereo, 3, axis=0)
        check = _gate(detector).check(at_48k, samplerate=48000)
        self.assertTrue(check.heard_speech)
        self.assertEqual(set(detector.windows), {512}, "every window is the detector's own length")
        self.assertAlmostEqual(check.duration_s, 1.0, places=3)

    def test_the_detector_starts_fresh_for_every_recording(self) -> None:
        detector = LoudnessDetector()
        gate = _gate(detector)
        gate.check(silence(1.0), samplerate=SAMPLERATE)
        gate.check(silence(1.0), samplerate=SAMPLERATE)
        self.assertEqual(detector.resets, 2)

    def test_from_config_builds_silero_from_the_section_and_loads_nothing(self) -> None:
        import torch

        from src.models.speech.gate import SpeechGate
        from src.models.speech.silero import SileroVoiceActivityDetector

        with patch.object(torch.jit, "load") as load:
            gate = SpeechGate.from_config(config=_config())
        load.assert_not_called()
        self.assertIsInstance(gate.detector, SileroVoiceActivityDetector)

    def test_the_check_is_a_frozen_report_with_both_halves(self) -> None:
        from src.contracts import Rendered, Structured

        check = _gate().check(tone(1.0), samplerate=SAMPLERATE)
        self.assertIsInstance(check, Rendered)
        self.assertIsInstance(check, Structured)
        data = check.to_dict()
        self.assertEqual(json.loads(json.dumps(data)), data)
        text = check.render()
        self.assertTrue(text.isascii() and not text.endswith("\n"), text)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            check.heard_speech = False


class TheProposalTests(unittest.TestCase):
    def test_no_speech_is_an_empty_proposal_with_a_reason_and_whisper_is_not_asked(self) -> None:
        engine = RecordingEngine()
        proposal = _held(engine).propose(silence(2.0), samplerate=SAMPLERATE)
        self.assertEqual(proposal.text, "")
        self.assertIsNotNone(proposal.reason)
        self.assertIn("no speech", str(proposal.reason).lower())
        self.assertIsNone(proposal.transcript)
        self.assertEqual(engine.calls, [], "Whisper was asked about a recording without speech")

    def test_speech_is_transcribed_and_the_proposal_carries_the_transcript(self) -> None:
        engine = RecordingEngine()
        proposal = _held(engine).propose(tone(1.0), samplerate=SAMPLERATE)
        self.assertEqual(proposal.text, "Nimm den roten Würfel")
        self.assertIsNone(proposal.reason)
        self.assertIsNotNone(proposal.transcript)
        self.assertEqual(len(engine.calls), 1)

    def test_speech_that_whisper_decodes_to_no_words_says_so(self) -> None:
        proposal = _held(RecordingEngine(text="")).propose(tone(1.0), samplerate=SAMPLERATE)
        self.assertEqual(proposal.text, "")
        self.assertIn("no words", str(proposal.reason))

    def test_a_recording_longer_than_whispers_window_is_refused_before_either_model_runs(self) -> None:
        from src.models.speech.engine import RecordingTooLong

        engine, detector = RecordingEngine(), LoudnessDetector()
        with self.assertRaises(RecordingTooLong) as caught:
            _held(engine, detector).propose(tone(30.5), samplerate=SAMPLERATE)
        self.assertIsInstance(caught.exception, ValueError)
        self.assertIn("30 s", str(caught.exception))
        self.assertIn("30.5", str(caught.exception))
        self.assertEqual((detector.windows, engine.calls), ([], []))

    def test_exactly_thirty_seconds_is_accepted(self) -> None:
        proposal = _held().propose(tone(30.0), samplerate=SAMPLERATE)
        self.assertTrue(proposal.speech.heard_speech)

    def test_the_proposal_is_a_frozen_report_with_both_halves(self) -> None:
        from src.contracts import Rendered, Structured

        for audio in (silence(1.0), tone(1.0)):
            proposal = _held().propose(audio, samplerate=SAMPLERATE)
            with self.subTest(heard=proposal.speech.heard_speech):
                self.assertIsInstance(proposal, Rendered)
                self.assertIsInstance(proposal, Structured)
                data = proposal.to_dict()
                self.assertEqual(json.loads(json.dumps(data)), data)
                self.assertEqual(set(data), {"text", "reason", "speech", "transcript"})
                text = proposal.render()
                self.assertTrue(text.isascii() and not text.endswith("\n"), text)
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    proposal.text = "something else"


class TheHolderTests(unittest.TestCase):
    def _counting(self) -> tuple[Any, list[str]]:
        from src.models.speech.holder import SpeechHolder

        built: list[str] = []

        def engine(config: SpeechToTextConfig) -> Any:
            built.append(f"engine:{config.language}")
            return RecordingEngine()

        def gate(config: SpeechToTextConfig) -> Any:
            built.append(f"gate:{config.language}")
            return _gate()

        return SpeechHolder.from_parts(build_engine=engine, build_gate=gate), built

    def test_the_same_section_gets_the_same_engine_and_gate(self) -> None:
        holder, built = self._counting()
        first = holder.for_config(config=_config())
        second = holder.for_config(config=_config())
        self.assertIs(first.engine, second.engine)
        self.assertIs(first.gate, second.gate)
        self.assertEqual(built, ["engine:auto", "gate:auto"])

    def test_a_changed_section_builds_new_ones(self) -> None:
        holder, built = self._counting()
        first = holder.for_config(config=_config())
        second = holder.for_config(config=_config(language="german"))
        self.assertIsNot(first.engine, second.engine)
        self.assertEqual(built, ["engine:auto", "gate:auto", "engine:german", "gate:german"])

    def test_forget_drops_what_it_holds(self) -> None:
        holder, built = self._counting()
        holder.for_config(config=_config())
        holder.forget()
        holder.for_config(config=_config())
        self.assertEqual(len(built), 4)

    def test_for_tree_reads_the_speech_section_alone(self) -> None:
        from src.config.loader import load_speech_section

        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        root = tmp / "data"
        shutil.copytree(_SHIPPED, root)
        (root / "camera" / "cam.yaml").write_text("cameras: [never closed\n", encoding="utf-8")
        holder, _ = self._counting()
        held = holder.for_tree(root, profile=None)
        self.assertEqual(held.config, load_speech_section(root, profile=None))

    def test_the_default_holder_builds_whisper_and_silero_and_loads_nothing(self) -> None:
        import torch
        import transformers

        from src.models.speech.holder import SpeechHolder
        from src.models.speech.silero import SileroVoiceActivityDetector
        from src.models.speech.whisper_transformers import WhisperTransformersEngine

        with patch.object(torch.jit, "load") as load, \
                patch.object(transformers.WhisperForConditionalGeneration, "from_pretrained") as models:
            held = SpeechHolder.from_parts().for_config(config=_config())
        self.assertIsInstance(held.engine, WhisperTransformersEngine)
        self.assertIsInstance(held.gate.detector, SileroVoiceActivityDetector)
        load.assert_not_called()
        models.assert_not_called()

    def test_there_is_one_holder_per_process(self) -> None:
        from src.models.speech.holder import shared_speech

        self.assertIs(shared_speech(), shared_speech())

    def test_a_listener_can_take_the_held_engine(self) -> None:
        from src.models.speech.listener import Listener

        holder, _ = self._counting()
        held = holder.for_config(config=_config())
        listener = Listener.from_config(config=held.config, engine=held.engine, detector=LoudnessDetector())
        self.assertIs(listener.engine, held.engine)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
