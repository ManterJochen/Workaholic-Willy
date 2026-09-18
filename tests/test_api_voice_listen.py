"""The console's push to talk at the cell PC: `/v1/voice/talk` raises the talk event, and
`/v1/voice/listen` proposes what was said while the switch was held.

The answer is the upload's `Proposal`, through the same voice gate and engine, and like an upload it
starts nothing. These run on a fake held speech (a loudness detector and a recording engine) and a ring
double for the microphone, so they need no weights, no sound card and no PortAudio.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import yaml

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover, the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.schema.models.models_schema import SpeechToTextConfig
from tests._speech_fakes import (
    ClockedSwitch,
    FakeClock,
    LoudnessDetector,
    RecordingEngine,
    RingSource,
    blocks_of,
    silence,
    tone,
)

_SHIPPED_STT = Path(__file__).resolve().parents[1] / "config" / "models" / "stt.yaml"


def _config() -> SpeechToTextConfig:
    block = yaml.safe_load(_SHIPPED_STT.read_text(encoding="utf-8"))["stt"]
    return SpeechToTextConfig.model_validate(block)


def _held(engine: RecordingEngine) -> Any:
    from src.models.speech.gate import SpeechGate
    from src.models.speech.holder import HeldSpeech

    return HeldSpeech(
        config=_config(), engine=engine, gate=SpeechGate.from_parts(detector=LoudnessDetector())
    )


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class ThePushToTalkRoutesTests(unittest.TestCase):
    def setUp(self) -> None:
        from api.app import create_app

        self.client = TestClient(create_app())

    def test_listen_proposes_what_was_said_while_the_switch_was_held(self) -> None:
        from src.models.speech.push_to_talk import PushToTalkSource, TalkButton

        button = TalkButton.from_parts()
        button.press()  # the operator holds the switch as the listen starts
        spoken = blocks_of(silence(0.2), tone(0.6), silence(0.2))

        def last_word(ring: RingSource) -> None:
            ring.feed(spoken[-1])
            button.release()  # the hand comes up after the last word

        ring = RingSource(
            steps=[*((lambda ring, block=block: ring.feed(block)) for block in spoken[:-1]), last_word]
        )
        engine = RecordingEngine(text="pick the red cube", language="en")
        source = PushToTalkSource.from_parts(source=ring, switch=button)
        with patch("api.routers.media._held_speech", return_value=_held(engine)), \
                patch("api.routers.media._talk_source", return_value=source) as built:
            response = self.client.post("/v1/voice/listen", json={"timeout_s": 5.0})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(set(body), {"text", "reason", "speech", "transcript"}, "not the ProposalOut shape")
        self.assertEqual(body["text"], "pick the red cube")
        self.assertIsNone(body["reason"])
        self.assertTrue(body["speech"]["heard_speech"])
        self.assertEqual(body["transcript"]["language"], "en")
        (heard, rate), = engine.calls
        self.assertEqual(rate, 16000)
        np.testing.assert_array_equal(heard, np.concatenate(spoken), "the turn lost audio")
        built.assert_called_once()
        self.assertEqual((ring.starts, ring.stops), (1, 1), "the microphone was left open")

    def test_a_switch_that_was_never_pressed_is_an_answer_that_says_so(self) -> None:
        from src.models.speech.push_to_talk import PushToTalkSource

        clock = FakeClock()
        engine = RecordingEngine()
        source = PushToTalkSource.from_parts(
            source=RingSource(clock=clock), switch=ClockedSwitch(clock), clock=clock
        )
        with patch("api.routers.media._held_speech", return_value=_held(engine)), \
                patch("api.routers.media._talk_source", return_value=source):
            response = self.client.post("/v1/voice/listen")  # no body: the default wait
        self.assertEqual(response.status_code, 409, response.text)
        body = response.json()
        self.assertEqual(body["code"], "talk_not_pressed")
        self.assertIn("not pressed within 10.0 s", body["message"])
        self.assertEqual(body["detail"]["outcome"], "not_pressed")
        self.assertEqual(engine.calls, [], "Whisper was asked about a turn nobody spoke")

    def test_a_microphone_this_host_cannot_open_is_a_missing_capability(self) -> None:
        from src.models.speech.capture import MicrophoneUnavailable
        from src.models.speech.push_to_talk import PushToTalkSource, TalkButton

        ring = RingSource(fail_to_open=MicrophoneUnavailable("this machine has no usable input device."))
        source = PushToTalkSource.from_parts(source=ring, switch=TalkButton.from_parts())
        with patch("api.routers.media._held_speech", return_value=_held(RecordingEngine())), \
                patch("api.routers.media._talk_source", return_value=source):
            response = self.client.post("/v1/voice/listen", json={"timeout_s": 1.0})
        self.assertEqual(response.status_code, 501, response.text)
        self.assertEqual(response.json()["code"], "microphone_unavailable")
        self.assertIn("Type the prompt instead", response.json()["message"])

    def test_talk_presses_and_releases_the_process_talk_switch(self) -> None:
        from src.models.speech.push_to_talk import shared_talk_button

        button = shared_talk_button()
        button.release()
        self.addCleanup(button.release)
        before = button.presses
        down = self.client.post("/v1/voice/talk", json={"pressed": True})
        self.assertEqual(down.status_code, 200, down.text)
        self.assertEqual(down.json(), {"pressed": True, "presses": before + 1})
        repeated = self.client.post("/v1/voice/talk", json={"pressed": True})
        self.assertEqual(repeated.json(), {"pressed": True, "presses": before + 1},
                         "a key that repeats while held counted as a second press")
        self.assertTrue(button.pressed)
        up = self.client.post("/v1/voice/talk", json={"pressed": False})
        self.assertEqual(up.json(), {"pressed": False, "presses": before + 1})
        self.assertFalse(button.pressed)

    def test_talk_needs_to_say_which_edge(self) -> None:
        response = self.client.post("/v1/voice/talk", json={})
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["code"], "bad_request")

    def test_both_routes_are_in_the_openapi_document_with_the_proposal_shape(self) -> None:
        spec = self.client.get("/openapi.json").json()
        listen = spec["paths"]["/v1/voice/listen"]["post"]
        self.assertEqual(
            listen["responses"]["200"]["content"]["application/json"]["schema"]["$ref"],
            "#/components/schemas/ProposalOut",
        )
        self.assertIn("/v1/voice/talk", spec["paths"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
