"""Push to talk: the talk switch gates the microphone, and a turn is what is said while it is held.

The owner decided the trigger twice (2026-09-11 and 2026-09-18): push to talk, from a console button and
from a switch at the cell PC, raising the same event. `PushToTalkSource` serves nothing before the press,
drops what was captured before it, serves what arrives while the switch is held, and at the release
serves what the ring still holds before it reports that it has ended. The last part is the one these tests
lean on: `MicrophoneSource.stop()` drops the audio not yet read, so a source that ended a turn by
stopping the microphone would cut off the end of every command.

Every test uses a ring double or a fake `sounddevice` module; none needs a sound card.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

from src.config.schema.models.models_schema import SpeechToTextConfig
from src.contracts import Rendered, Structured
from src.models.speech.capture import MicrophoneSource, MicrophoneUnavailable
from src.models.speech.endpointing import UtteranceEnd
from src.models.speech.listener import ListenOutcome, Listener
from src.models.speech.push_to_talk import (
    PushToTalkSource,
    TalkButton,
    TalkOutcome,
    TalkRecording,
    TalkSwitch,
    shared_talk_button,
)
from tests._speech_fakes import (
    SAMPLERATE,
    ClockedSwitch,
    FakeClock,
    LoudnessDetector,
    RecordingEngine,
    RingSource,
    blocks_of,
    fake_sounddevice,
    silence,
    tone,
)

_SHIPPED_STT = Path(__file__).resolve().parents[1] / "config" / "models" / "stt.yaml"


def _config(**overrides: Any) -> SpeechToTextConfig:
    block = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]
    return SpeechToTextConfig.model_validate({**block, **overrides})


def _started(ring: RingSource, switch: Any, clock: FakeClock | None = None) -> PushToTalkSource:
    source = PushToTalkSource.from_parts(
        source=ring, switch=switch, **({} if clock is None else {"clock": clock})
    )
    source.start()
    return source


def _feeder(samples: np.ndarray) -> Any:
    """A ring step that feeds one block, as the sound card does between two reads."""
    return lambda ring: ring.feed(samples)


def _last_word(samples: np.ndarray, switch: Any) -> Any:
    """A ring step that feeds the last block and lets the switch up, as a hand does after the last word."""

    def step(ring: RingSource) -> None:
        ring.feed(samples)
        switch.release()

    return step


class TheTalkButtonTests(unittest.TestCase):
    def test_a_press_counts_once_while_the_button_is_held(self) -> None:
        """A foot switch that sends a key repeats it while held; that is one press, not twenty."""
        button = TalkButton.from_parts()
        button.press()
        button.press()
        self.assertTrue(button.pressed)
        self.assertEqual(button.presses, 1)
        button.release()
        button.release()
        self.assertFalse(button.pressed)
        self.assertEqual(button.presses, 1)
        button.press()
        self.assertEqual(button.presses, 2)

    def test_a_press_that_came_and_went_is_still_counted(self) -> None:
        button = TalkButton.from_parts()
        button.press()
        button.release()
        self.assertEqual(button.wait_for_press(after=0, timeout_s=0.0), 1)

    def test_waiting_for_a_press_ends_at_the_bound(self) -> None:
        button = TalkButton.from_parts()
        started = time.monotonic()
        self.assertEqual(button.wait_for_press(after=0, timeout_s=0.05), 0)
        self.assertGreaterEqual(time.monotonic() - started, 0.04)

    def test_a_press_from_another_thread_ends_the_wait(self) -> None:
        button = TalkButton.from_parts()
        timer = threading.Timer(0.05, button.press)
        timer.start()
        self.addCleanup(timer.cancel)
        started = time.monotonic()
        self.assertEqual(button.wait_for_press(after=0, timeout_s=5.0), 1)
        self.assertLess(time.monotonic() - started, 4.0)

    def test_the_process_has_one_talk_button_and_it_is_a_talk_switch(self) -> None:
        self.assertIs(shared_talk_button(), shared_talk_button())
        self.assertIsInstance(shared_talk_button(), TalkSwitch)


class ThePushToTalkSourceTests(unittest.TestCase):
    def test_nothing_is_served_before_the_press(self) -> None:
        ring = RingSource()
        source = _started(ring, TalkButton.from_parts())
        ring.feed(tone(0.2))
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertFalse(source.ended)
        self.assertEqual(len(ring.buffered), 1, "audio before the press was served or dropped early")

    def test_the_press_drops_what_was_captured_before_it(self) -> None:
        button = TalkButton.from_parts()
        ring = RingSource()
        source = _started(ring, button)
        ring.feed(np.full(1600, 0.9, dtype=np.float32))  # captured before the press
        button.press()
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertEqual(ring.buffered, [])
        during = tone(0.1)
        ring.feed(during)
        block = source.read(timeout_s=0.01)
        self.assertIsNotNone(block)
        np.testing.assert_array_equal(block.samples, during)

    def test_the_release_serves_what_the_ring_still_holds_and_then_ends(self) -> None:
        """MicrophoneSource.stop() drops the audio not yet read (capture.py), and at the release that
        audio is the end of the command. The source serves it and only then reports that it has ended;
        the microphone stays open."""
        button = TalkButton.from_parts()
        ring = RingSource()
        source = _started(ring, button)
        button.press()
        self.assertIsNone(source.read(timeout_s=0.01))
        during, tail = tone(0.3), tone(0.2, amplitude=0.3)
        ring.feed(during)
        np.testing.assert_array_equal(source.read(timeout_s=0.01).samples, during)
        ring.feed(tail)  # said just before the release, not read yet
        button.release()
        self.assertFalse(source.ended, "the source ended before it served the ring")
        block = source.read(timeout_s=0.01)
        self.assertIsNotNone(block, "the audio still in the ring at the release was dropped")
        np.testing.assert_array_equal(block.samples, tail)
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertTrue(source.ended)
        self.assertEqual(ring.stops, 0, "the turn was ended by stopping the microphone")
        self.assertTrue(ring.open)

    def test_the_real_ring_is_drained_at_the_release_and_the_stream_stays_open(self) -> None:
        """The same drain against `MicrophoneSource` itself, fed through a fake sounddevice stream."""
        module = fake_sounddevice()
        button = TalkButton.from_parts()
        source = PushToTalkSource.from_parts(source=MicrophoneSource.from_parts(), switch=button)
        with patch.dict(sys.modules, {"sounddevice": module}):
            source.start()
        self.addCleanup(source.stop)
        (stream,) = module.opened  # type: ignore[attr-defined]
        stream.feed(np.full((800, 1), 20000, dtype=np.int16))  # before the press
        button.press()
        self.assertIsNone(source.read(timeout_s=0.01))
        stream.feed(np.full((1600, 1), 1000, dtype=np.int16))
        self.assertEqual(source.read(timeout_s=0.01).samples.shape[0], 1600)
        stream.feed(np.full((400, 1), -1000, dtype=np.int16))
        button.release()
        tail = source.read(timeout_s=0.01)
        self.assertIsNotNone(tail)
        np.testing.assert_allclose(tail.samples, np.full(400, -1000 / 32768.0, dtype=np.float32))
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertTrue(source.ended)
        self.assertTrue(stream.active, "the microphone was stopped to end the turn")

    def test_a_press_held_when_the_turn_is_armed_counts(self) -> None:
        """The console sends the press and the listen at the same moment; the press may win."""
        button = TalkButton.from_parts()
        button.press()
        ring = RingSource()
        source = _started(ring, button)
        self.assertIsNone(source.read(timeout_s=0.01))
        ring.feed(tone(0.1))
        self.assertIsNotNone(source.read(timeout_s=0.01))

    def test_a_press_that_came_up_before_the_turn_was_armed_does_not_count(self) -> None:
        button = TalkButton.from_parts()
        button.press()
        button.release()
        ring = RingSource()
        source = _started(ring, button)
        ring.feed(tone(0.1))
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertIsNone(source.read(timeout_s=0.01))
        self.assertFalse(source.ended)

    def test_discarding_while_the_switch_is_held_drops_nothing(self) -> None:
        """`Listener.listen()` discards at its start; during a held turn that would cut the command."""
        button = TalkButton.from_parts()
        ring = RingSource()
        source = _started(ring, button)
        button.press()
        source.read(timeout_s=0.01)
        during = tone(0.2)
        ring.feed(during)
        self.assertEqual(source.discard_buffered(), 0)
        np.testing.assert_array_equal(source.read(timeout_s=0.01).samples, during)

    def test_discarding_after_a_turn_arms_the_next_one(self) -> None:
        button = TalkButton.from_parts()
        ring = RingSource()
        source = _started(ring, button)
        button.press()
        source.read(timeout_s=0.01)
        button.release()
        source.read(timeout_s=0.01)
        self.assertTrue(source.ended)
        source.discard_buffered()
        self.assertFalse(source.ended)
        button.press()
        self.assertIsNone(source.read(timeout_s=0.01))
        ring.feed(tone(0.1))
        self.assertIsNotNone(source.read(timeout_s=0.01))

    def test_with_opens_and_closes_the_microphone(self) -> None:
        ring = RingSource()
        with PushToTalkSource.from_parts(source=ring, switch=TalkButton.from_parts()) as source:
            self.assertTrue(ring.open)
            self.assertFalse(source.ended)
        self.assertFalse(ring.open)
        self.assertTrue(source.ended)


class TheRecordTests(unittest.TestCase):
    def test_a_turn_is_everything_said_while_the_switch_was_held(self) -> None:
        clock = FakeClock()
        button = TalkButton.from_parts()
        button.press()
        chunks = blocks_of(tone(0.3))
        tail = tone(0.1, amplitude=0.2)
        ring = RingSource(
            clock=clock,
            steps=[*(_feeder(chunk) for chunk in chunks), _last_word(tail, button)],
        )
        source = _started(ring, button, clock)
        ring.feed(np.full(3200, 0.9, dtype=np.float32))  # before the turn was armed
        recording = source.record(timeout_s=5.0)
        self.assertIs(recording.outcome, TalkOutcome.RELEASED)
        self.assertTrue(recording.ok)
        np.testing.assert_array_equal(recording.samples, np.concatenate([*chunks, tail]))
        self.assertEqual(recording.samplerate, SAMPLERATE)
        self.assertAlmostEqual(recording.waited_s, 0.0)
        self.assertAlmostEqual(recording.held_s, 0.4, places=6)
        self.assertEqual(ring.stops, 0)

    def test_a_switch_never_pressed_ends_at_the_bound(self) -> None:
        clock = FakeClock()
        ring = RingSource(clock=clock)
        source = _started(ring, ClockedSwitch(clock), clock)
        recording = source.record(timeout_s=1.0)
        self.assertIs(recording.outcome, TalkOutcome.NOT_PRESSED)
        self.assertFalse(recording.ok)
        self.assertEqual(recording.frames, 0)
        self.assertAlmostEqual(recording.waited_s, 1.0, places=6)
        self.assertIn("not pressed within 1.0 s", recording.render())

    def test_a_switch_held_past_the_longest_turn_is_not_a_command(self) -> None:
        clock = FakeClock()
        switch = ClockedSwitch(clock)
        switch.press()
        ring = RingSource(clock=clock, refill=tone(0.1))
        source = _started(ring, switch, clock)
        recording = source.record(timeout_s=1.0, longest_s=0.5)
        self.assertIs(recording.outcome, TalkOutcome.HELD_TOO_LONG)
        self.assertFalse(recording.ok)
        self.assertAlmostEqual(recording.duration_s, 0.5, places=6)

    def test_a_release_before_any_audio_captured_nothing(self) -> None:
        button = TalkButton.from_parts()
        button.press()
        ring = RingSource(steps=[lambda ring: button.release()])
        source = _started(ring, button)
        recording = source.record(timeout_s=1.0)
        self.assertIs(recording.outcome, TalkOutcome.NOTHING_CAPTURED)
        self.assertFalse(recording.ok)
        self.assertIn("Hold it while you speak", recording.render())

    def test_a_microphone_that_ends_mid_turn_says_so(self) -> None:
        clock = FakeClock()
        switch = ClockedSwitch(clock)
        switch.press()
        ring = RingSource(clock=clock, steps=[_feeder(tone(0.1)), lambda ring: ring.stop()])
        source = _started(ring, switch, clock)
        recording = source.record(timeout_s=1.0)
        self.assertIs(recording.outcome, TalkOutcome.SOURCE_ENDED)
        self.assertEqual(recording.frames, 1600)

    def test_record_needs_an_open_microphone(self) -> None:
        source = PushToTalkSource.from_parts(source=RingSource(), switch=TalkButton.from_parts())
        with self.assertRaises(RuntimeError) as caught:
            source.record(timeout_s=1.0)
        self.assertIn("start()", str(caught.exception))

    def test_record_refuses_bounds_that_mean_nothing(self) -> None:
        source = _started(RingSource(), TalkButton.from_parts())
        for bounds in ({"timeout_s": 0.0}, {"timeout_s": 1.0, "longest_s": -1.0}):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    source.record(**bounds)


class AListenerOverAPushToTalkSourceTests(unittest.TestCase):
    def test_the_release_hands_over_what_was_said(self) -> None:
        """The owner's cell PC path: a `Listener` whose source is the microphone behind the switch.
        Speech that runs up to the release is closed by the release, not lost."""
        clock = FakeClock()
        button = TalkButton.from_parts()
        button.press()
        spoken = blocks_of(silence(0.2), tone(0.6))
        ring = RingSource(
            clock=clock,
            steps=[
                *(_feeder(block) for block in spoken[:-1]),
                _last_word(spoken[-1], button),
            ],
        )
        engine = RecordingEngine()
        source = PushToTalkSource.from_parts(source=ring, switch=button, clock=clock)
        listener = Listener.from_parts(
            source=source, detector=LoudnessDetector(), engine=engine, clock=clock
        )
        with listener:
            heard = listener.listen(timeout_s=5.0)
        self.assertIs(heard.outcome, ListenOutcome.HEARD)
        self.assertIs(heard.ended_by, UtteranceEnd.SOURCE_END)
        self.assertEqual(heard.transcript.text, engine.text)
        self.assertEqual(len(engine.calls), 1)

    def test_from_config_takes_a_chosen_source(self) -> None:
        source = PushToTalkSource.from_parts(source=RingSource(), switch=TalkButton.from_parts())
        listener = Listener.from_config(
            config=_config(), engine=RecordingEngine(), source=source, detector=LoudnessDetector()
        )
        self.assertIs(listener.source, source)

    def test_from_config_still_opens_the_cell_microphone_when_no_source_is_chosen(self) -> None:
        listener = Listener.from_config(
            config=_config(), engine=RecordingEngine(), detector=LoudnessDetector()
        )
        self.assertIsInstance(listener.source, MicrophoneSource)


class TheConfigDoorTests(unittest.TestCase):
    def test_from_config_puts_the_cell_microphone_behind_the_process_talk_button(self) -> None:
        config = _config()
        source = PushToTalkSource.from_config(config=config)
        self.assertIsInstance(source.source, MicrophoneSource)
        self.assertIs(source.switch, shared_talk_button())
        self.assertEqual(source.samplerate, config.samplerate)
        self.assertTrue(source.ended, "building the source opened the microphone")

    def test_a_chosen_switch_is_the_one_used(self) -> None:
        button = TalkButton.from_parts()
        self.assertIs(PushToTalkSource.from_config(config=_config(), switch=button).switch, button)

    def test_both_doors_are_keyword_only(self) -> None:
        with self.assertRaises(TypeError):
            PushToTalkSource.from_config(_config())  # type: ignore[misc]
        with self.assertRaises(TypeError):
            PushToTalkSource.from_parts(RingSource())  # type: ignore[misc]


class TheMicrophoneRefusalTests(unittest.TestCase):
    def test_a_machine_without_a_usable_input_is_refused_as_a_missing_microphone(self) -> None:
        module = fake_sounddevice()

        def no_input(device: Any = None, kind: Any = None) -> Any:
            raise module.PortAudioError("Error querying device -1")  # type: ignore[attr-defined]

        module.query_devices = no_input  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"sounddevice": module}):
            with self.assertRaises(MicrophoneUnavailable) as caught:
                MicrophoneSource.from_parts().start()
        self.assertIn("no usable default input device (", str(caught.exception))
        self.assertIsInstance(caught.exception, RuntimeError, "callers that caught RuntimeError still do")

    def test_a_device_that_refuses_the_stream_is_refused_as_a_missing_microphone(self) -> None:
        module = fake_sounddevice()

        class RefusingStream:
            def __init__(self, **kwargs: Any) -> None:
                raise module.PortAudioError("Invalid sample rate")  # type: ignore[attr-defined]

        module.InputStream = RefusingStream  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"sounddevice": module}):
            with self.assertRaises(MicrophoneUnavailable) as caught:
                MicrophoneSource.from_parts().start()
        self.assertIn("did not open", str(caught.exception))


class TheTalkRecordingReportTests(unittest.TestCase):
    def _recording(self, outcome: TalkOutcome, frames: int = 1600, held_s: float = 0.4) -> TalkRecording:
        return TalkRecording(
            outcome=outcome,
            samples=np.zeros(frames, dtype=np.float32),
            samplerate=SAMPLERATE,
            waited_s=0.5,
            held_s=held_s,
            timeout_s=10.0,
            longest_s=30.0,
            overflowed=outcome is TalkOutcome.RELEASED,
        )

    def test_render_is_ascii_and_ends_without_a_newline_for_every_outcome(self) -> None:
        for outcome in TalkOutcome:
            with self.subTest(outcome=outcome):
                text = self._recording(outcome).render()
                text.encode("ascii")
                self.assertFalse(text.endswith("\n"))
        self.assertIn("overflowed", self._recording(TalkOutcome.RELEASED).render())
        self.assertIn(
            "before the talk switch went down",
            self._recording(TalkOutcome.SOURCE_ENDED, frames=0, held_s=0.0).render(),
        )

    def test_to_dict_is_plain_data_and_never_the_audio(self) -> None:
        wire = json.loads(json.dumps(self._recording(TalkOutcome.RELEASED).to_dict()))
        self.assertEqual(
            wire,
            {
                "outcome": "released", "frames": 1600, "samplerate": SAMPLERATE, "duration_s": 0.1,
                "waited_s": 0.5, "held_s": 0.4, "timeout_s": 10.0, "longest_s": 30.0,
                "overflowed": True,
            },
        )

    def test_it_is_frozen_and_both_halves_of_the_report_contract(self) -> None:
        recording = self._recording(TalkOutcome.RELEASED)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            recording.held_s = 1.0  # type: ignore[misc]
        self.assertIsInstance(recording, Rendered)
        self.assertIsInstance(recording, Structured)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
