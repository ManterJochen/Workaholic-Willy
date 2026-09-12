"""Driving a pick from the console, and watching one you were not there for.

The property this file exists to pin: **a run survives the browser.** An operator starts a pick, the
laptop sleeps, the tab closes. The arm is holding a part, so the correct behaviour is to finish the pick
and put the object down -- not to abandon a motion halfway because a socket dropped. What makes that
survivable for the UI is the sequence number: a client reconnects saying *"I last saw 41"* and gets 42
onwards, or an honest gap marker if it slept longer than the buffer.

The second property: **stop is not kill.** It sets a flag the pick loop reads between attempts. A motion
already in flight completes. Anything stronger from a browser would be a promise this process cannot
keep, and an operator who believed it would stop reaching for the button on the wall.

Honesty bucket (2): a real dummy cell, real threads, real sockets, no hardware.
"""

from __future__ import annotations

import contextlib
import re
import shutil
import tempfile
import time
import types
import unittest
import unittest.mock
from collections.abc import Iterator
from pathlib import Path
from typing import Any

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile

_SHIPPED = Path(__file__).resolve().parents[1] / "config"


def _dummy_tree(target: Path) -> None:
    shutil.copytree(_SHIPPED, target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(
        r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
    )
    text, gripper_hits = re.subn(
        r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "none"', text, count=1, flags=re.MULTILINE
    )
    assert arm_hits == 1 and gripper_hits == 1, "the dummy substitution found nothing"
    robot.write_text(text, encoding="utf-8")


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class PickRunTests(unittest.TestCase):
    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        _dummy_tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.disconnect()
        except Exception:  # pragma: no cover
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def _connected(self) -> None:
        self.assertEqual(
            self.client.post("/v1/cell/build", params={"rehearse": True}).status_code, 200
        )
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.assertEqual(
            self.client.post("/v1/cell/connect", json={"token": token}).status_code, 200
        )

    def _await_run(self, run_id: str, *, timeout: float = 20.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            body = self.client.get(f"/v1/runs/{run_id}").json()
            if body["state"] != "running":
                return body
            time.sleep(0.02)
        self.fail(f"run {run_id} did not finish within {timeout}s")

    # -- refusals ----------------------------------------------------------------------------------

    def test_a_pick_on_a_disconnected_cell_is_refused(self) -> None:
        response = self.client.post("/v1/pick", json={"prompt": "cube", "picks": 1})
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["code"], "not_connected")

    def test_a_second_run_is_refused_rather_than_queued(self) -> None:
        """One arm, one run. A second would not wait its turn -- it would drive the same arm."""
        self._connected()
        first = self.client.post("/v1/pick", json={"picks": 20}).json()
        try:
            second = self.client.post("/v1/pick", json={"picks": 1})
            self.assertEqual(second.status_code, 409)
            self.assertEqual(second.json()["code"], "run_active")
            self.assertEqual(second.json()["detail"]["run_id"], first["id"])
        finally:
            self.client.post("/v1/pick/stop")
            self._await_run(first["id"])

    # -- the run -----------------------------------------------------------------------------------

    def test_starting_a_run_returns_immediately(self) -> None:
        """202, not 200: the work is accepted and happening elsewhere, not finished when this returns.

        A handler that waited would hold the connection open across a physical motion, and something in
        the middle -- browser, proxy, sleeping laptop -- would give up without telling the operator
        whether the arm was still moving.
        """
        self._connected()
        started = time.monotonic()
        response = self.client.post("/v1/pick", json={"prompt": "cube", "picks": 3})
        elapsed = time.monotonic() - started

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["state"], "running")
        self.assertLess(elapsed, 2.0, "starting a run must not wait for it")
        self._await_run(response.json()["id"])

    def test_a_finished_run_reports_typed_outcomes(self) -> None:
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 2}).json()["id"]
        body = self._await_run(run_id)

        self.assertEqual(body["state"], "finished")
        self.assertEqual(body["attempted"], 2)
        self.assertEqual(len(body["outcomes"]), 2)
        # Typed strings from the outcome enum, never free text.
        self.assertTrue(all(isinstance(o, str) and o for o in body["outcomes"]))

    def test_the_run_lock_blocks_config_writes_while_it_runs(self) -> None:
        """The rule Stage 1 shipped against a flag nothing set. This is the flag being set."""
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 20}).json()["id"]
        try:
            response = self.client.patch("/v1/config", json={"robot.safety.payload.mass_kg": 1.15})
            self.assertEqual(response.status_code, 409)
            self.assertEqual(response.json()["code"], "run_active")
        finally:
            self.client.post("/v1/pick/stop")
            self._await_run(run_id)

        # ...and it is released afterwards, or the console would be permanently read-only.
        self.assertEqual(self.client.get("/v1/cell").json()["active_run_id"], None)

    def test_stopping_ends_the_run_as_cancelled_not_failed(self) -> None:
        """A stop is a DECISION. Counting it as a failure would make every press of the button lower
        the measured pick rate -- an operator's caution showing up as the cell's incompetence."""
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 50}).json()["id"]
        self.assertEqual(self.client.post("/v1/pick/stop").status_code, 200)
        body = self._await_run(run_id)

        self.assertEqual(body["state"], "cancelled")
        self.assertTrue(body["stop_requested"])
        self.assertLess(body["attempted"], 50, "a stop must actually stop something")

    # -- the event stream --------------------------------------------------------------------------

    def test_the_stream_carries_a_sentence_and_a_payload_for_every_event(self) -> None:
        """Two audiences, one envelope: the demo audience reads one field, the diagnosis reads the
        other. An envelope with only one of them forces the other to guess."""
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 1}).json()["id"]
        self._await_run(run_id)

        events, dropped = self.cell.hub.since(run_id, 0)
        self.assertEqual(dropped, 0)
        self.assertGreater(len(events), 5, "a pick produces a trail, not one message at the end")
        for event in events:
            self.assertTrue(event.human, f"{event.type} has no sentence for the operator")
            self.assertIsInstance(event.data, dict)
        # Sequence numbers are dense and start at 1 -- the reconnect story rests on it.
        self.assertEqual([e.seq for e in events], list(range(1, len(events) + 1)))

    def test_a_socket_replays_what_it_missed_then_goes_live(self) -> None:
        """The whole reason for `since_seq`: connect AFTER the run finished and miss nothing."""
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 1}).json()["id"]
        self._await_run(run_id)

        with self.client.websocket_connect(f"/v1/events?run_id={run_id}&since_seq=0") as socket:
            first = socket.receive_json()
            self.assertEqual(first["run_id"], run_id)
            self.assertEqual(first["seq"], 1)
            self.assertIn("human", first)

    def test_a_reconnect_receives_only_what_came_after_its_cursor(self) -> None:
        self._connected()
        run_id = self.client.post("/v1/pick", json={"picks": 1}).json()["id"]
        self._await_run(run_id)
        total = self.cell.hub.latest_seq(run_id)
        self.assertGreater(total, 3)

        with self.client.websocket_connect(
            f"/v1/events?run_id={run_id}&since_seq={total - 2}"
        ) as socket:
            self.assertEqual(socket.receive_json()["seq"], total - 1)

    def test_a_client_that_slept_too_long_is_told_what_it_lost(self) -> None:
        """A short replay that LOOKED complete would be worse than the gap: the UI would draw a run
        that never had those steps, and nobody would know to distrust it."""
        from api.events import EventHub

        hub = EventHub(capacity=4)
        for _ in range(10):
            hub.publish("run-x", "noise", human="something happened")

        events, dropped = hub.since("run-x", since_seq=1)
        self.assertEqual(dropped, 5, "events 2-6 fell out of a 4-deep buffer")
        self.assertEqual([e.seq for e in events], [7, 8, 9, 10])


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class EventHubTests(unittest.TestCase):
    """The sequencing guarantees, without a robot in the way."""

    def test_sequence_numbers_are_per_run(self) -> None:
        """Two runs must not share a counter, or a reconnect on one would skip events on the other."""
        from api.events import EventHub

        hub = EventHub()
        hub.publish("a", "x")
        hub.publish("b", "x")
        hub.publish("a", "x")
        self.assertEqual([e.seq for e in hub.since("a", 0)[0]], [1, 2])
        self.assertEqual([e.seq for e in hub.since("b", 0)[0]], [1])

    def test_waiting_returns_empty_on_timeout_rather_than_blocking_forever(self) -> None:
        """It is what lets a quiet socket send a keep-alive: quiet and dead look identical otherwise."""
        from api.events import EventHub

        hub = EventHub()
        started = time.monotonic()
        self.assertEqual(hub.wait_for("nobody", 0, timeout=0.05), [])
        self.assertLess(time.monotonic() - started, 2.0)

    def test_a_waiter_is_woken_by_a_publish(self) -> None:
        import threading

        from api.events import EventHub

        hub = EventHub()
        received: list[int] = []

        def _wait() -> None:
            received.extend(e.seq for e in hub.wait_for("r", 0, timeout=5.0))

        waiter = threading.Thread(target=_wait)
        waiter.start()
        time.sleep(0.05)
        hub.publish("r", "x")
        waiter.join(timeout=5.0)
        self.assertEqual(received, [1])


class RunRetentionTests(unittest.TestCase):
    """What a console left running for a week is still holding.

    ⛔ THE DEFECT. ``EventHub.forget`` existed for exactly this and had no caller anywhere in the
    tree, so every finished run's event history stayed in the hub for the life of the process, and
    the registry kept its ``Run`` record, its sequence counter and its dead ``Thread`` object beside
    it. MEASURED in this tree: 1000 runs started, 1000 kept, nothing ever dropped; one 35-event run
    replays as about 14 kB of JSON (~402 bytes per envelope) and its sequence-counter entry costs 91
    bytes, and the ring lets a single run hold 2048 envelopes. A console is a long-lived process
    standing next to a robot; nothing here ever ended.

    No fastapi and no cell: the registry and the hub are the two things under test, and neither
    needs the optional extra.
    """

    #: More runs than a console is expected to keep, and fewer than a bring-up day produces.
    _A_LONG_SESSION = 250

    def setUp(self) -> None:
        from api.events import EventHub
        from api.runs import RunRegistry

        self.hub = EventHub()
        self.registry = RunRegistry(self.hub)
        service = types.SimpleNamespace(
            attach_progress_listener=lambda _listener: None,
            set_cancel_check=lambda _check: None,
            set_target_label=lambda _label: None,
        )
        self.console = types.SimpleNamespace(
            active_run_id=None, session=types.SimpleNamespace(service=service)
        )

    def _finished_run(self) -> str:
        """One run of zero picks: it starts, publishes, finishes. No arm, no attempt, real threads."""
        run = self.registry.start(self.console, prompt="", picks=0)
        # Waiting on `run_finished` rather than on `active()`: the state flips to FINISHED before
        # the thread publishes its last envelope, so a helper that watched the state would race the
        # publish and the NEXT run would start mid-teardown.
        deadline = time.monotonic() + 10.0
        while not any(e.type == "run_finished" for e in self.hub.since(run.id, 0)[0]):
            if time.monotonic() > deadline:  # pragma: no cover - a hung run is a different failure
                self.fail(f"run {run.id} never finished")
            time.sleep(0.001)
        return run.id

    def test_a_console_that_has_run_all_week_lets_go_of_its_oldest_runs(self) -> None:
        ids = [self._finished_run() for _ in range(self._A_LONG_SESSION)]
        oldest = ids[0]

        self.assertLess(
            len(self.registry.recent(limit=100_000)), self._A_LONG_SESSION,
            "every run ever started is still in the registry",
        )
        self.assertIsNone(self.registry.get(oldest), "the oldest run's record was never dropped")
        self.assertEqual(
            self.hub.since(oldest, 0), ([], 0), "the oldest run's history was never dropped"
        )
        self.assertEqual(self.hub.latest_seq(oldest), 0, "its sequence counter outlived it")

    def test_what_it_keeps_is_WHOLE_so_a_reconnect_still_replays(self) -> None:
        """The bound must not cost the property the sequence numbers exist for: a browser that comes
        back after the run finished still gets every event of a run the console still names."""
        from api.runs import RETAINED_RUNS

        ids = [self._finished_run() for _ in range(self._A_LONG_SESSION)]

        events, dropped = self.hub.since(ids[-1], 0)
        self.assertEqual(dropped, 0)
        self.assertEqual(
            [e.type for e in events], ["run_started", "run_finished"], "a kept run lost events"
        )
        kept = [run.id for run in self.registry.recent(limit=100_000)]
        self.assertEqual(kept, list(reversed(ids[-RETAINED_RUNS:])), "the window is not the last N")

    def test_a_finished_run_does_not_leave_its_thread_behind(self) -> None:
        """A ``Thread`` object per run, in a dict nothing reads again. Bounding it is not the answer:
        the object is useless the moment the run ends, so it goes then."""
        for _ in range(5):
            self._finished_run()
        # Private on purpose: this dict has no reader, which is exactly why it could grow unnoticed.
        self.assertEqual(
            dict(self.registry._threads), {}, "finished runs are still holding their threads"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class MediaTests(unittest.TestCase):
    """The overlay stream and speech-to-text: optional, and neither one moves anything."""

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        _dummy_tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    @staticmethod
    def _wav(*, channels: int = 1, rate: int = 16000, seconds: float = 1.0) -> bytes:
        import io
        import math
        import struct
        import wave

        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as handle:
            handle.setnchannels(channels)
            handle.setsampwidth(2)
            handle.setframerate(rate)
            frames = int(rate * seconds)
            handle.writeframes(
                b"".join(
                    struct.pack("<h", int(3000 * math.sin(i / 8)))
                    for i in range(frames * channels)
                )
            )
        return buffer.getvalue()

    def test_the_overlay_is_off_until_someone_asks(self) -> None:
        """It costs real time per pick -- the calculator only forwards the RGB when it is on."""
        self.assertEqual(self.client.post("/v1/overlay/enable").status_code, 409)

        self.client.post("/v1/cell/build", params={"rehearse": True})
        response = self.client.post("/v1/overlay/enable")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.json()["enabled"])

    def test_the_overlay_stream_says_what_the_picture_is(self) -> None:
        """It is the last SEGMENTATION the calculator rendered, not a camera feed and not necessarily
        the object that was grasped. A UI that called it a video would let an operator conclude the arm
        is where the picture shows."""
        with self.client.websocket_connect("/v1/overlay") as socket:
            info = socket.receive_json()
        self.assertEqual(info["type"], "overlay_info")
        self.assertIn("not a camera feed", info["human"])

    def test_empty_audio_is_refused_before_a_model_is_loaded(self) -> None:
        response = self.client.post(
            "/v1/voice/transcribe", files={"audio": ("empty.wav", b"", "audio/wav")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["code"], "empty_audio")

    def _forget_speech_engine(self) -> None:
        """The process holds one speech engine per section; a test must not leave its fakes in it."""
        from src.models.speech.holder import shared_speech

        shared_speech().forget()

    @contextlib.contextmanager
    def _speech(self, *, speech: bool = True, weights: bool = True) -> Iterator[dict[str, Any]]:
        """Whisper and Silero behind their real loaders, from paths written into this test's own tree.

        ⛔ THE FAILURE TEST BELOW WAS NOT HERMETIC. It counted on no Whisper weights being on the box
        and went red on the first box that had them, measured with whisper-small present:
        `AssertionError: 200 not found in (422, 501) : {"text":"you","language":"en",...}`. Every model
        path is now written into the temporary tree, both loaders are stand-ins, and the device is the
        CPU. ``speech`` is what the stand-in voice detector hears; ``weights`` False leaves the Whisper
        directory absent.
        """
        import os

        import torch
        import transformers

        from tests._speech_fakes import FakeSileroModel, whisper_parts

        whisper = self.tmp.parent / "whisper"
        if weights:
            whisper.mkdir(exist_ok=True)
        vad = self.tmp.parent / "silero_vad.jit"
        vad.write_bytes(b"stand-in")
        stt = self.tmp / "models" / "stt.yaml"
        text, hits = re.subn(r"(?m)^(\s*)model_path:.*$", rf'\g<1>model_path: "{whisper.as_posix()}"',
                             stt.read_text(encoding="utf-8"))
        text, vad_hits = re.subn(r"(?m)^(\s*)vad_model_path:.*$",
                                 rf'\g<1>vad_model_path: "{vad.as_posix()}"', text)
        assert (hits, vad_hits) == (1, 1), "the shipped stt.yaml stopped naming its two model paths"
        stt.write_text(text, encoding="utf-8")
        self.addCleanup(self._forget_speech_engine)
        processor, model = whisper_parts()
        silero = FakeSileroModel(speech=speech)
        with contextlib.ExitStack() as stack:
            stack.enter_context(unittest.mock.patch.dict(os.environ, {"WILLY_DEVICE": "cpu"}))
            yield {
                "processors": stack.enter_context(unittest.mock.patch.object(
                    transformers.WhisperProcessor, "from_pretrained", return_value=processor)),
                "models": stack.enter_context(unittest.mock.patch.object(
                    transformers.WhisperForConditionalGeneration, "from_pretrained", return_value=model)),
                "loads": stack.enter_context(unittest.mock.patch.object(
                    torch.jit, "load", return_value=silero)),
                "silero": silero,
            }

    @staticmethod
    def _proposal(text: str = "pick the red cube") -> Any:
        from src.models.speech.transcript import LanguageSource, Proposal, SpeechCheck, Transcript

        return Proposal(
            text=text,
            reason=None,
            speech=SpeechCheck(
                heard_speech=True, duration_s=1.0, checked_s=0.5, peak_probability=0.97, onset=0.5,
                min_speech_s=0.25, detector="silero-vad", latency_ms=3.0,
            ),
            transcript=Transcript(
                text=text, language="en", language_source=LanguageSource.DETECTED, duration_s=1.0,
                engine="whisper-transformers", model="org/whisper", device="cpu", latency_ms=12.5,
            ),
        )

    def test_a_transcription_failure_is_an_answer_not_a_crash(self) -> None:
        """No Whisper weights where the section says, which is the state a fresh bench machine is in.

        The endpoint must name the reason and stay up: a console that 500s on a missing optional model
        tells an operator nothing about what to fetch. A model this host has not fetched is a missing
        capability (501), and the message names the directory and the fetch.
        """
        with self._speech(weights=False) as fakes:
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
        self.assertEqual(response.status_code, 501, response.text)
        self.assertEqual(response.json()["code"], "speech_model_missing")
        self.assertIn("whisper", response.json()["message"])
        self.assertIn("fetch.py", response.json()["message"])
        self.assertEqual(fakes["processors"].call_count, 0)

    def test_speech_returns_text_and_never_starts_a_run(self) -> None:
        """A misheard word must not be able to move an arm. Speech lands in the prompt box; a human
        presses the same button, with the same acknowledgement, as for a typed prompt."""
        from unittest.mock import patch

        proposal = self._proposal()
        with patch("api.routers.media._transcribe", return_value=proposal):
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), proposal.to_dict(), "the wire answer is the proposal, whole")
        self.assertEqual(self.client.get("/v1/runs").json(), [], "no run may have been started")

    def test_stereo_audio_is_mixed_rather_than_half_discarded(self) -> None:
        """A laptop's second microphone often carries most of the voice; picking one at random would
        transcribe the quiet side."""
        import numpy as np

        from api.routers import media
        from src.models.speech import holder

        captured: dict[str, object] = {}
        proposal = self._proposal("ok")

        class _Held:
            def propose(self, samples: "np.ndarray", *, samplerate: int) -> object:
                captured["samples"], captured["rate"] = samples, samplerate
                return proposal

        class _Holder:
            def for_tree(self, data_dir: object, *, profile: object) -> _Held:
                return _Held()

        with unittest.mock.patch.object(holder, "shared_speech", return_value=_Holder()):
            media._transcribe(self.cell, self._wav(channels=2, rate=16000, seconds=0.1))

        samples = captured["samples"]
        assert isinstance(samples, np.ndarray)
        self.assertEqual(samples.ndim, 1, "Whisper takes mono")
        self.assertEqual(captured["rate"], 16000)
        self.assertEqual(len(samples), 1600, "one sample per FRAME, not per channel")

    def test_the_speech_model_loads_once_across_two_requests(self) -> None:
        """Whisper was loaded from disk on every request, against a 2 s budget from the end of speaking.

        The two `from_pretrained` holders and `torch.jit.load` are faked, so this needs no weights and
        counts real loads."""
        with self._speech() as fakes:
            answers = [
                self.client.post(
                    "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
                )
                for _ in range(2)
            ]
        for answer in answers:
            self.assertEqual(answer.status_code, 200, answer.text)
            self.assertEqual(answer.json()["text"], "Pick the red cube", "and never lower-cased")
        self.assertEqual((fakes["processors"].call_count, fakes["models"].call_count), (1, 1),
                         "the weights were loaded per request")
        self.assertEqual(fakes["loads"].call_count, 1, "the voice detector was loaded per request")

    def _evict_speech_modules(self) -> None:
        """Make the next request import the speech package afresh, as a newly started console would."""
        import sys

        evicted = {name: module for name, module in sys.modules.items()
                   if name.startswith("src.models.speech")}
        for name in evicted:
            del sys.modules[name]
        self.addCleanup(sys.modules.update, evicted)

    def test_a_missing_portaudio_leaves_the_upload_path_alone(self) -> None:
        """`import sounddevice` raises OSError when the system PortAudio library is missing. An upload
        opens no microphone, so it must not need one; the speech module imported sounddevice at its top
        and the endpoint answered 422 on such a machine."""
        import builtins

        real_import = builtins.__import__

        def no_portaudio(name: str, *args: object, **kwargs: object) -> object:
            if name == "sounddevice" or name.startswith("sounddevice."):
                raise OSError("PortAudio library not found")
            return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

        with self._speech():
            self._evict_speech_modules()
            with unittest.mock.patch.object(builtins, "__import__", no_portaudio):
                response = self.client.post(
                    "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
                )
        self.assertEqual(response.status_code, 200, response.text)

    def test_a_speech_stack_that_cannot_load_answers_501_naming_the_file_that_holds_it(self) -> None:
        """A DLL the OS refuses arrives as OSError, a missing package as ModuleNotFoundError. Both mean
        this machine cannot run speech, which is a missing capability (501), not a bad request (422).
        A missing package is answered with the requirements file that really holds it; a refused DLL
        with the sentence that the package is installed and Windows refused the named file."""
        import builtins

        real_import = builtins.__import__
        refused_dll = OSError(
            "[WinError 4551] An Application Control policy has blocked this file. Error loading "
            '"D:\\venv\\Lib\\site-packages\\torch\\lib\\c10.dll" or one of its dependencies.'
        )
        cases: tuple[tuple[str, BaseException, tuple[str, ...]], ...] = (
            ("torch", refused_dll, ("torch is installed", "Windows refused", "c10.dll")),
            ("transformers",
             ModuleNotFoundError("No module named 'transformers'", name="transformers"),
             ("requirements.txt",)),
        )
        with self._speech():
            for package, failure, fragments in cases:
                with self.subTest(package=package):
                    self._evict_speech_modules()

                    def blocked(name: str, *args: object, _package: str = package,
                                _failure: BaseException = failure, **kwargs: object) -> object:
                        if name == _package or name.startswith(_package + "."):
                            raise _failure
                        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

                    with unittest.mock.patch.object(builtins, "__import__", blocked):
                        response = self.client.post(
                            "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
                        )
                    self.assertEqual(response.status_code, 501, response.text)
                    self.assertEqual(response.json()["code"], "speech_unavailable")
                    for fragment in fragments:
                        self.assertIn(fragment, response.json()["message"])
                    self.assertNotIn("voice.txt", response.json()["message"],
                                     "there is no voice.txt in this tree to send anyone to")

    def test_a_broken_camera_file_does_not_refuse_transcription(self) -> None:
        """Speech reads `models.stt` alone. It read the whole tree, so a camera file with a typo refused
        a recording that needs no camera."""
        (self.tmp / "camera" / "cam.yaml").write_text("cameras: [never closed\n", encoding="utf-8")
        reload_config()
        with self._speech():
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["text"], "Pick the red cube")

    def test_an_upload_without_speech_proposes_nothing_and_says_why(self) -> None:
        """Whisper answered 0.2 s of a tone with "you" through this endpoint (2026-09-11, whisper-small),
        and the word would have landed in the prompt box. The voice detector runs first; a recording in
        which it hears no speech is an empty proposal with a reason, and Whisper is not asked."""
        with self._speech(speech=False) as fakes:
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["text"], "")
        self.assertIn("no speech", str(body["reason"]).lower())
        self.assertIsNone(body["transcript"])
        self.assertFalse(body["speech"]["heard_speech"])
        self.assertEqual(fakes["processors"].call_count, 0, "Whisper was loaded for a recording without speech")

    def test_a_recording_longer_than_whispers_window_is_refused_with_a_sentence(self) -> None:
        with self._speech() as fakes:
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(seconds=31.0), "audio/wav")}
            )
        self.assertEqual(response.status_code, 422, response.text)
        self.assertEqual(response.json()["code"], "audio_too_long")
        self.assertIn("30 s", response.json()["message"])
        self.assertEqual(fakes["silero"].calls, [], "the voice detector ran on a recording it refuses")

    def test_an_upload_that_is_not_wav_is_refused_saying_the_console_records_wav(self) -> None:
        response = self.client.post(
            "/v1/voice/transcribe",
            files={"audio": ("a.webm", b"\x1aE\xdf\xa3" + bytes(64), "audio/webm")},
        )
        self.assertEqual(response.status_code, 415, response.text)
        self.assertEqual(response.json()["code"], "audio_format_unsupported")
        for fragment in ("WAV", "WebM", "audio/webm"):
            self.assertIn(fragment, response.json()["message"])

    def test_the_console_takes_its_engine_from_the_library_holder(self) -> None:
        """The one engine lived in a private router global, where a `Listener` caller could not reach it.
        It lives in `src.models.speech.holder` now, and the console asks the holder."""
        import numpy as np

        from src.models.speech.holder import shared_speech

        with self._speech() as fakes:
            answer = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
            held = shared_speech().for_tree(self.tmp, profile=None)
            held.engine.transcribe(np.zeros(16000, dtype=np.float32), samplerate=16000)
        self.assertEqual(answer.status_code, 200, answer.text)
        self.assertEqual(fakes["processors"].call_count, 1, "the holder did not hold the console's engine")


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class HistoryTests(unittest.TestCase):
    """What survives the process, what does not, and the one KPI that cannot be measured."""

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp())
        _dummy_tree(self.tmp / "data")
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp / "data", profile=None)
        # A scratch record log: the console's real default is under logs/, and a test must not append
        # to the file an operator's history lives in.
        self.cell.record_log_path = self.tmp / "records.jsonl"
        self._previous_console = set_console(self.cell)
        self.client = TestClient(create_app())
        self.addCleanup(self._restore)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.disconnect()
        except Exception:  # pragma: no cover
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_two_picks(self) -> None:
        self.client.post("/v1/cell/build", params={"rehearse": True})
        token = self.client.get("/v1/cell/connect-preview").json()["token"]
        self.client.post("/v1/cell/connect", json={"token": token})
        run_id = self.client.post("/v1/pick", json={"picks": 2}).json()["id"]
        deadline = time.monotonic() + 20.0
        while time.monotonic() < deadline:
            if self.client.get(f"/v1/runs/{run_id}").json()["state"] != "running":
                return
            time.sleep(0.02)
        self.fail("the run did not finish")

    def test_an_empty_history_is_a_state_not_a_fault(self) -> None:
        """A fresh bench has logged nothing. A history view that 500s there sends an operator hunting
        for a fault that is not there."""
        body = self.client.get("/v1/history/kpis").json()
        self.assertEqual(body["total_attempts"], 0)
        self.assertFalse(body["record_log_exists"])
        self.assertEqual(self.client.get("/v1/history/records").json(), [])

    def test_the_console_logs_records_although_the_config_names_no_path(self) -> None:
        """`grasping.record_log_path` is null in the shipped config, so a console-driven cell would
        keep no history at all -- and a bring-up that went wrong would have nothing to diagnose from."""
        self.assertFalse(self.cell.record_log_path.exists())
        self._run_two_picks()

        self.assertTrue(self.cell.record_log_path.exists(), "the console must have logged something")
        records = self.client.get("/v1/history/records").json()
        self.assertEqual(len(records), 2)
        for record in records:
            self.assertTrue(record["attempt_id"])
            # The provenance stamp: a corpus from two robots must not be silently mixed.
            self.assertEqual(record["extra"]["robot_vendor"], "dummy")
            self.assertEqual(record["extra"]["written_by"], "operator console")

    def test_a_configured_path_wins_over_the_console_default(self) -> None:
        """An operator who deliberately configured a path keeps it; the console only fills a gap."""
        configured = self.tmp / "configured.jsonl"
        robot = self.tmp / "data" / "robot" / "robot.yaml"
        # The existing line is REPLACED, not preceded by a second one. Inserting a duplicate key looks
        # like it works and does not: PyYAML silently keeps the LAST occurrence, so the shipped
        # `record_log_path: null` would have won and this test would have failed for a reason with
        # nothing to do with what it tests. (It did, on the first attempt.)
        text, hits = re.subn(
            r"^(\s*)record_log_path:.*$",
            lambda m: f'{m.group(1)}record_log_path: "{configured.as_posix()}"',
            robot.read_text(encoding="utf-8"), count=1, flags=re.MULTILINE,
        )
        self.assertEqual(hits, 1, "the record_log_path line was not found to replace")
        robot.write_text(text, encoding="utf-8")
        reload_config()

        self._run_two_picks()
        self.assertTrue(configured.exists(), "the configured path must be the one written")
        self.assertEqual(self.cell.record_log_path, configured)

    def test_kpis_come_from_the_same_function_the_offline_gate_uses(self) -> None:
        """A console with its own success-rate arithmetic would eventually disagree with
        `python -m ...replay --records`, and nobody could then say which number was real."""
        from src.robot.grasping.replay import compute_kpis
        from src.robot.grasping.telemetry.outcome_logging import iter_jsonl

        self._run_two_picks()
        body = self.client.get("/v1/history/kpis").json()
        expected = compute_kpis(list(iter_jsonl(self.cell.record_log_path)))

        self.assertEqual(body["total_attempts"], expected.total_attempts)
        self.assertAlmostEqual(body["kpis"]["pick_success_rate"], expected.pick_success_rate)
        # And the typed outcomes are tallied, so a UI never has to parse free text for a verdict.
        self.assertEqual(sum(body["outcomes"].values()), expected.total_attempts)

    def test_the_unmeasurable_kpi_is_named_rather_than_shown_as_zero(self) -> None:
        """`false_positive_grasp_rate` is computed from a field the serializer deliberately never sets,
        because no independent post-grasp re-check exists. The arithmetic returns a confident 0.0 --
        and a 0.0% false-positive rate on a demo screen is a claim nobody here can make."""
        self._run_two_picks()
        body = self.client.get("/v1/history/kpis").json()

        self.assertNotIn("false_positive_grasp_rate", body["kpis"])
        self.assertIn("false_positive_grasp_rate", body["unmeasurable"])
        self.assertIn("independent post-grasp re-check", body["unmeasurable"]["false_positive_grasp_rate"])

    def test_both_csv_exports_have_a_header_and_one_row_per_thing(self) -> None:
        self._run_two_picks()

        runs = self.client.get("/v1/history/runs.csv")
        self.assertEqual(runs.status_code, 200)
        self.assertIn("text/csv", runs.headers["content-type"])
        run_lines = runs.text.strip().splitlines()
        self.assertEqual(len(run_lines), 2, "one header + one run")
        self.assertIn("succeeded", run_lines[0])

        attempts = self.client.get("/v1/history/records.csv")
        attempt_lines = attempts.text.strip().splitlines()
        self.assertEqual(len(attempt_lines), 3, "one header + two attempts")
        self.assertIn("final_outcome", attempt_lines[0])

    def test_the_two_sources_are_reported_separately(self) -> None:
        """Runs are rich and perishable; records survive and are thinner. Blurring them would let an
        operator conclude that something is stored which is not."""
        self._run_two_picks()
        session_runs = self.client.get("/v1/history/runs").json()
        records = self.client.get("/v1/history/records").json()

        self.assertEqual(len(session_runs), 1, "one run...")
        self.assertEqual(len(records), 2, "...that produced two logged attempts")
        self.assertIn("outcomes", session_runs[0])
        self.assertNotIn("outcomes", records[0])
