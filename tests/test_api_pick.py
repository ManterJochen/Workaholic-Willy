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

import re
import shutil
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

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
    def _wav(*, channels: int = 1, rate: int = 16000, seconds: float = 0.2) -> bytes:
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

    def test_a_transcription_failure_is_an_answer_not_a_crash(self) -> None:
        """No Whisper model on this box, which is exactly the state a fresh bench machine is in.

        The endpoint must name the reason and stay up: a console that 500s on a missing optional model
        tells an operator nothing about what to install.
        """
        response = self.client.post(
            "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
        )
        self.assertIn(response.status_code, (422, 501), response.text)
        self.assertIn(response.json()["code"], {"transcription_failed", "speech_unavailable"})
        self.assertTrue(response.json()["message"])

    def test_speech_returns_text_and_never_starts_a_run(self) -> None:
        """A misheard word must not be able to move an arm. Speech lands in the prompt box; a human
        presses the same button, with the same acknowledgement, as for a typed prompt."""
        from unittest.mock import patch

        with patch("api.routers.media._transcribe", return_value="  pick the red cube  "):
            response = self.client.post(
                "/v1/voice/transcribe", files={"audio": ("a.wav", self._wav(), "audio/wav")}
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"text": "pick the red cube"})
        self.assertEqual(self.client.get("/v1/runs").json(), [], "no run may have been started")

    def test_stereo_audio_is_mixed_rather_than_half_discarded(self) -> None:
        """A laptop's second microphone often carries most of the voice; picking one at random would
        transcribe the quiet side."""
        import numpy as np

        from api.routers.media import _transcribe

        captured: dict[str, object] = {}

        class _FakeWhisper:
            def __init__(self, _config: object) -> None:
                pass

            def transcribe_array(self, audio: "np.ndarray", rate: int) -> str:
                captured["samples"], captured["rate"] = audio, rate
                return "ok"

        with unittest.mock.patch(
            "src.models.speech.speech_to_text.WhisperSpeechToText", _FakeWhisper
        ):
            _transcribe(self.cell, self._wav(channels=2, rate=16000, seconds=0.1))

        samples = captured["samples"]
        assert isinstance(samples, np.ndarray)
        self.assertEqual(samples.ndim, 1, "Whisper takes mono")
        self.assertEqual(captured["rate"], 16000)
        self.assertEqual(len(samples), 1600, "one sample per FRAME, not per channel")


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
