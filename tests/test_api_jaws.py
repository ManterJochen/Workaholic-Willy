"""The jaws question of a toggle hand, answered in the browser (build plan 1.6, item 10).

The owner's hand is a Hand-E on tool DO0, a ``single_toggle`` with no sensor: only a person can say where its jaws
stand, and every change of DO0 moves them. The console asks that person in the browser, never at the terminal of the
server it runs in: a blocking ``input()`` cannot be cancelled, and nobody watching the browser sees it. What is held
here, over HTTP, on a rehearsal cell whose hand is the real ``JawIOGripper`` on a recording tool I/O (the dummy arm has
no DO of its own):

* a connect's question goes round through HTTP: it is published on the cell's stream and in ``GET /v1/cell/jaws``,
  answered with ``POST /v1/cell/jaws/answer``, and the connect answers once the hand is connected;
* "closed" offers ``open_now``, which is ONE change of DO0, sent only after the controller said it can move: a stopped
  controller refuses it with no DO0 change, and the connect is refused;
* an answer that leaves the jaws open tells the planner the hand is empty (``detach_payload``) and stamps the console,
  so the next moving run counts down after "open now";
* a question nobody answers in time is no answer, never "open": the connect is refused with nothing sent;
* a hand that would ask, built without the browser seam, refuses the connect; it never falls back to the terminal;
* a halted arm is refused at connect: a latched arm switches no output;
* Disconnect ends a waiting question before it takes the session lock, so it returns at once;
* ``POST /v1/cell/jaws/check`` reads the controller first and always asks, even where the count says open; where the
  count says closed and nothing moved the jaws since, it asks only "open now" (never "where": an answer of open would
  contradict the count); its refusals come in the plan's order;
* while a question is in progress, the waiting one and the change it leads to, a jaws question is pending, so every
  moving route can refuse it; a check is pending from its start, before it reads the run lock, so a run and a check
  never go on together: a run that started all the same refuses the check (``run_active``), ends its waiting question
  at once, and the one change never goes out while any run is active;
* a connect that came up with the jaws kept closed (a solenoid's feedback reads a part between them) stamps nothing
  open and tells the planner nothing;
* a Disconnect waits for the one change a check is sending, so the arm never comes down in the middle of it.

Honesty bucket (2): real objects, a real rehearsal build on a dummy tree, real threads, real HTTP; no controller.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest.mock import patch

try:
    from fastapi.testclient import TestClient
except ImportError:  # pragma: no cover - the console is an optional extra
    TestClient = None  # type: ignore[assignment,misc]

from src.config.loader import active_profile, reload_config, set_active_profile
from src.robot.core.arm_capabilities import RobotMode, RobotStatus, SafetyMode
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_a_toggle_asks_where_its_jaws_stand import _IO, Person, _pulses, _toggle

_ROOT = Path(__file__).resolve().parents[1]

#: A controller in a protective stop, as the receive stream reads it.
STOPPED = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.PROTECTIVE_STOP, protective_stopped=True,
                      emergency_stopped=False)
#: A controller that can move.
RUNNING = RobotStatus(robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL, protective_stopped=False,
                      emergency_stopped=False)


def dummy_tree(target: Path) -> None:
    """The shipped tree re-pointed at a dummy arm and a dummy hand: the one cell that builds with no hardware."""
    shutil.copytree(_ROOT / "config", target)
    robot = target / "robot" / "robot.yaml"
    text = robot.read_text(encoding="utf-8")
    text, arm_hits = re.subn(r'^(\s*)vendor:\s*"ur"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE)
    text, hand_hits = re.subn(
        r'^(\s*)vendor:\s*"robotiq"$', r'\g<1>vendor: "dummy"', text, count=1, flags=re.MULTILINE
    )
    assert arm_hits == 1 and hand_hits == 1, "the dummy substitution found nothing"
    robot.write_text(text, encoding="utf-8")


def wait_for(condition: Callable[[], Any], *, timeout: float = 10.0, what: str = "the condition") -> Any:
    """Poll ``condition`` until it answers something truthy; fail the test after ``timeout`` seconds."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        found = condition()
        if found:
            return found
        time.sleep(0.01)
    raise AssertionError(f"{what} did not happen within {timeout} s")


class _InBackground:
    """One HTTP request on its own thread and its own client, for a request that blocks on a person's answer."""

    def __init__(self, app: Any, method: str, url: str, **kwargs: Any) -> None:
        self.response: Any = None
        self.error: BaseException | None = None

        def run() -> None:
            try:
                self.response = TestClient(app).request(method, url, **kwargs)
            except BaseException as exc:  # noqa: BLE001 (reported by the test that waits for it)
                self.error = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def result(self, timeout: float = 20.0) -> Any:
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise AssertionError("the request is still blocked")
        if self.error is not None:
            raise self.error
        return self.response


@unittest.skipIf(TestClient is None, "fastapi is unavailable; requirements.txt pins fastapi and httpx")
class ScratchCell(unittest.TestCase):
    """A console on a scratch dummy tree, built in rehearsal, installed as the process console for the app."""

    def setUp(self) -> None:
        from api.app import create_app
        from api.cell import Console, set_console

        self.tmp = Path(tempfile.mkdtemp()) / "data"
        dummy_tree(self.tmp)
        self._previous_profile = active_profile()
        self.cell = Console(root=self.tmp, profile=None)
        self.cell.record_log_path = self.tmp.parent / "grasp_records.jsonl"
        self._previous_console = set_console(self.cell)
        self.addCleanup(self._restore)
        self.app = create_app()
        self.client = TestClient(self.app)

    def _restore(self) -> None:
        from api.cell import set_console

        try:
            self.cell.session.release()
        except Exception:  # pragma: no cover - cleanup must not mask a failure
            pass
        set_console(self._previous_console)
        set_active_profile(self._previous_profile)
        reload_config()
        shutil.rmtree(self.tmp.parent, ignore_errors=True)

    def build(self) -> None:
        built = self.client.post("/v1/cell/build", params={"rehearse": True})
        self.assertEqual(200, built.status_code, built.text)

    def token(self) -> str:
        preview = self.client.get("/v1/cell/connect-preview")
        self.assertEqual(200, preview.status_code, preview.text)
        return str(preview.json()["token"])

    def cell_events(self, event_type: str | None = None) -> list[Any]:
        from api.events import CELL_STREAM

        events = self.cell.hub.since(CELL_STREAM, 0)[0]
        return [e for e in events if event_type is None or e.type == event_type]


class ToggleCell(ScratchCell):
    """The rehearsal cell with the owner's hand: a ``single_toggle`` on tool DO0, the browser seam installed on it."""

    #: Whether the browser seam is handed to the toggle after the swap, as every build hands it.
    INSTALL = True

    def setUp(self) -> None:
        from api import jaws as browser_jaws

        super().setUp()
        self.build()
        self.events: list[Any] = []
        self.slept: list[float] = []
        # Nobody at the terminal: the string seam fails the test if the hand ever asks there.
        self.jaws = _toggle(self.events, Person(), sleep=self.slept.append)
        self.cell.session.service.runtime.orchestrator.gripper = self.jaws
        self.arm = self.cell.session.arm
        self.detached: list[str] = []
        self.arm.detach_payload = lambda: self.detached.append("detach") or True  # type: ignore[attr-defined]
        if self.INSTALL:
            browser_jaws.install(self.cell)

    def question(self, *, stage: str | None = None, after: str = "", timeout: float = 10.0) -> dict[str, Any]:
        """The question ``GET /v1/cell/jaws`` shows, once one waits (of ``stage``, and another than ``after``)."""

        def shown() -> dict[str, Any] | None:
            body = self.client.get("/v1/cell/jaws").json()
            asked = body.get("question")
            if asked and asked["question_id"] != after and (stage is None or asked["stage"] == stage):
                return asked
            return None

        return wait_for(shown, timeout=timeout, what="a jaws question in GET /v1/cell/jaws")

    def answer(self, question: dict[str, Any], choice: str) -> Any:
        return self.client.post("/v1/cell/jaws/answer", json={"question_id": question["question_id"],
                                                                "choice": choice})

    def connect_in_background(self) -> _InBackground:
        return _InBackground(self.app, "POST", "/v1/cell/connect", json={"token": self.token()})

    def connected(self) -> None:
        """Connect, answering "open" in the browser."""
        connecting = self.connect_in_background()
        self.assertEqual(200, self.answer(self.question(), "open").status_code)
        self.assertEqual(200, connecting.result().status_code)
        self.events.clear()
        self.detached.clear()


# ---------------------------------------------------------------------------------------------------------------------
# Connect: the question goes round through HTTP
# ---------------------------------------------------------------------------------------------------------------------


class AConnectAsksInTheBrowserTests(ToggleCell):
    def test_the_question_goes_round_through_http_and_the_connect_answers_once_it_is_answered(self) -> None:
        connecting = self.connect_in_background()
        asked = self.question()
        self.assertEqual(("where", "connect", "tool output 0", ["open", "closed"], 1, 3, ""),
                         (asked["stage"], asked["at"], asked["where"], asked["choices"], asked["attempt"], asked["of"],
                          asked["why_again"]))
        self.assertIn("every change of its output moves them", asked["reason"])
        self.assertIn("Do they stand OPEN?", asked["text"])
        self.assertAlmostEqual(time.time() + 120.0, asked["expires_at"], delta=10.0)
        # The question is the cell's, so the panel says one waits; the session lock the connect holds blocks no read.
        self.assertTrue(self.client.get("/v1/cell").json()["jaws_question"])
        self.assertTrue(connecting.thread.is_alive(), "the connect did not wait for the answer")

        answered = self.answer(asked, "open")

        self.assertEqual(200, answered.status_code, answered.text)
        body = answered.json()
        self.assertIsNone(body["question"])
        connected = connecting.result()
        self.assertEqual(200, connected.status_code, connected.text)
        self.assertEqual("connected", connected.json()["state"])
        self.assertEqual(("toggle", "open"), (connected.json()["hand"]["kind"], connected.json()["hand"]["jaws"]))
        self.assertFalse(connected.json()["jaws_question"])
        self.assertEqual(0, _pulses(self.events), "an answer of open changed DO0")
        # Open jaws hold nothing: the planner is told, and the answer is stamped; nothing was opened, so no countdown.
        self.assertEqual(["detach"], self.detached)
        self.assertIsNotNone(self.cell.jaws_confirmed_at)
        self.assertFalse(self.cell.countdown_due())
        self.assertEqual(["cell.jaws_question", "cell.jaws_answered", "cell.jaws_ended"],
                         [e.type for e in self.cell_events() if e.type.startswith("cell.jaws")])
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual((asked["question_id"], "open", True), (ended.data["question_id"], ended.data["outcome"],
                                                                 ended.data["detached"]))
        (published,) = self.cell_events("cell.jaws_question")
        self.assertEqual(asked, published.data, "the stream and the panel showed two questions")
        (said,) = self.cell_events("cell.jaws_answered")
        self.assertEqual({"question_id": asked["question_id"], "choice": "open"}, said.data)

    def test_closed_offers_open_now_which_is_one_change_of_do0_and_the_next_motion_counts_down(self) -> None:
        connecting = self.connect_in_background()
        where = self.question(stage="where")

        closed = self.answer(where, "closed")

        self.assertEqual(200, closed.status_code, closed.text)
        offered = closed.json()["question"]
        self.assertIsNotNone(offered, "the answer did not carry the next question")
        self.assertEqual(("open_now", ["open_now", "abort"]), (offered["stage"], offered["choices"]))
        self.assertEqual(0, _pulses(self.events), "closed alone changed DO0")

        opened = self.answer(offered, "open_now")

        self.assertEqual(200, opened.status_code, opened.text)
        self.assertEqual(200, connecting.result().status_code)
        self.assertEqual(1, _pulses(self.events), "open now is exactly ONE change of DO0")
        self.assertEqual("open", self.client.get("/v1/cell/jaws").json()["hand"]["jaws"])
        self.assertEqual(["detach"], self.detached)
        self.assertEqual("jaws_opened", self.cell.countdown_because(), "a person held the part: the next run counts down")
        self.assertTrue(self.client.get("/v1/cell").json()["countdown_due"])
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual(("open", True, offered["question_id"]),
                         (ended.data["outcome"], ended.data["detached"], ended.data["question_id"]))

    def test_a_stopped_controller_refuses_open_now_with_no_change_of_do0(self) -> None:
        self.arm.quick_robot_status = lambda: STOPPED  # type: ignore[attr-defined]
        connecting = self.connect_in_background()
        self.answer(self.question(stage="where"), "closed")
        self.answer(self.question(stage="open_now"), "open_now")

        refused = connecting.result()

        self.assertEqual((502, "driver_refused"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("controller cannot move", refused.json()["message"])
        self.assertIn("nothing was sent", refused.json()["message"])
        self.assertEqual(0, _pulses(self.events), "a stopped controller got its change of DO0")
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual([], self.detached, "a refused answer told the planner the hand is empty")
        self.assertIsNone(self.cell.jaws_confirmed_at)
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual(("refused", False), (ended.data["outcome"], ended.data["detached"]))
        self.assertIn("controller cannot move", ended.data["refusal"])

    def test_a_halted_arm_gets_no_open_now_either(self) -> None:
        connecting = self.connect_in_background()
        self.answer(self.question(stage="where"), "closed")
        offered = self.question(stage="open_now")
        self.arm.halt("the operator pressed halt now")
        self.answer(offered, "open_now")

        refused = connecting.result()

        self.assertEqual(502, refused.status_code, refused.text)
        self.assertIn("halted", refused.json()["message"])
        self.assertEqual(0, _pulses(self.events))

    def test_a_question_nobody_answers_is_refused_and_never_open(self) -> None:
        from unittest.mock import patch

        with patch("api.jaws.JAW_QUESTION_TIMEOUT_S", 0.3):
            refused = self.client.post("/v1/cell/connect", json={"token": self.token()})

        self.assertEqual((502, "driver_refused"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("nobody answered", refused.json()["message"].lower())
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual(0, _pulses(self.events))
        self.assertFalse(self.jaws.is_connected)
        self.assertIsNone(self.client.get("/v1/cell/jaws").json()["question"])
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual(("no_answer", False), (ended.data["outcome"], ended.data["detached"]))

    def test_disconnect_while_a_question_waits_ends_it_and_returns_at_once(self) -> None:
        connecting = self.connect_in_background()
        self.question()
        started = time.monotonic()

        down = self.client.post("/v1/cell/disconnect")

        self.assertLess(time.monotonic() - started, 5.0, "Disconnect waited on the question")
        self.assertEqual(200, down.status_code, down.text)
        refused = connecting.result()
        self.assertEqual(502, refused.status_code, refused.text)
        self.assertIn("cancelled", refused.json()["message"])
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual(0, _pulses(self.events))
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual("cancelled", ended.data["outcome"])

    def test_a_build_while_a_connect_waits_on_its_question_is_refused_at_once(self) -> None:
        """A connect holds the session while its question waits, minutes perhaps; a build that queued behind it would
        hold the console's lock all that time, and every read of the panel with it."""
        connecting = self.connect_in_background()
        asked = self.question()
        building = _InBackground(self.app, "POST", "/v1/cell/build", params={"rehearse": True})
        refused = building.result(timeout=5.0)
        self.assertEqual((409, "wrong_state"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(200, self.client.get("/v1/cell").status_code)
        self.answer(asked, "open")
        self.assertEqual(200, connecting.result().status_code)

    def test_a_connect_stamps_the_console_only_after_it_let_go_of_the_session(self) -> None:
        """``Console.build`` holds the console's lock and then asks for the session's; a connect that came up stamps the
        console. Stamped while the connect still held the session, the two would wait for each other for ever: a build
        racing a connect would freeze every read of ``GET /v1/cell``."""
        connecting = self.connect_in_background()
        asked = self.question()
        holding, got_session, done = threading.Event(), threading.Event(), threading.Event()
        self.addCleanup(done.set)

        def as_a_build_does() -> None:
            with self.cell._lock:                     # the console's lock, held as Console.build holds it
                holding.set()
                if self.cell.session._lock.acquire(timeout=5.0):
                    got_session.set()
                    self.cell.session._lock.release()
            done.set()

        builder = threading.Thread(target=as_a_build_does, daemon=True)
        builder.start()
        self.assertTrue(holding.wait(5.0))
        self.answer(asked, "open")
        self.assertTrue(done.wait(10.0))
        self.assertTrue(got_session.is_set(), "the connect held the session while it waited for the console's lock")
        self.assertEqual(200, connecting.result().status_code)
        self.assertIsNotNone(self.cell.jaws_confirmed_at, "the connect that came up was not stamped")

    def test_the_answer_is_refused_for_no_question_a_stale_one_and_a_choice_not_offered(self) -> None:
        nothing = self.client.post("/v1/cell/jaws/answer", json={"question_id": "q-nobody", "choice": "open"})
        self.assertEqual((404, "no_question"), (nothing.status_code, nothing.json()["code"]), nothing.text)
        connecting = self.connect_in_background()
        asked = self.question()

        stale = self.client.post("/v1/cell/jaws/answer", json={"question_id": "q-stale", "choice": "open"})
        self.assertEqual((409, "question_changed"), (stale.status_code, stale.json()["code"]), stale.text)
        not_offered = self.answer(asked, "open_now")
        self.assertEqual((422, "choice_not_offered"), (not_offered.status_code, not_offered.json()["code"]))
        self.assertEqual(asked["question_id"], self.question()["question_id"], "a refused answer ended the question")
        no_default = self.client.post("/v1/cell/jaws/answer", json={"question_id": asked["question_id"], "choice": ""})
        self.assertEqual((422, "bad_request"), (no_default.status_code, no_default.json()["code"]))

        self.answer(asked, "open")
        self.assertEqual(200, connecting.result().status_code)


class AMissingSeamRefusesTheConnectTests(ToggleCell):
    INSTALL = False

    def test_a_toggle_without_the_browser_seam_is_refused_before_the_arm_connects(self) -> None:
        connects: list[str] = []
        original = type(self.arm).connect
        self.arm.connect = lambda: connects.append("arm") or original(self.arm)  # type: ignore[method-assign]

        refused = self.client.post("/v1/cell/connect", json={"token": self.token()})

        self.assertEqual((409, "jaws_seam_missing"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("browser", refused.json()["message"])
        self.assertEqual([], connects, "the arm connected for a hand nobody could ask")
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual(0, _pulses(self.events))

    def test_a_solenoid_that_asks_at_connect_is_refused_without_the_browser_seam_too(self) -> None:
        """``confirm_open_at_start`` makes a solenoid ask a person before anything moves, as a toggle always does: with
        no browser seam its question could only go to the terminal of the server, so the connect is refused."""
        events: list[Any] = []
        hand = JawIOGripper(_IO(events), actuation="double_solenoid", close_output_pin=0, open_output_pin=1,
                            confirm_open_at_start=True, pulse_s=0.0, close_settle_s=0.0, ask=Person(),
                            sleep=lambda _s: None)
        self.cell.session.service.runtime.orchestrator.gripper = hand

        refused = self.client.post("/v1/cell/connect", json={"token": self.token()})

        self.assertEqual((409, "jaws_seam_missing"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertFalse(hand.is_connected)
        self.assertEqual([], events, "a hand nobody could ask was commanded")


class _PartBetweenTheJaws(_IO):
    """A tool I/O whose part-present sensor (input 1) reads a part between the jaws."""

    def get_digital_input(self, pin: int, *, port: Any = None) -> bool:
        return pin == 1


class AConnectThatKeepsThePartIsNeverStampedOpenTests(ScratchCell):
    def test_a_solenoid_whose_sensor_reads_a_part_stamps_nothing_and_keeps_the_planner_s_part(self) -> None:
        """A solenoid that asks first (``confirm_open_at_start``) and then connects by its feedback keeps its jaws
        closed where the sensor reads a part between them: the connect comes up holding it. The jaws do not stand open,
        so nothing is stamped open and the planner is not told the hand is empty, whatever the person answered."""
        from api import jaws as browser_jaws

        self.build()
        events: list[Any] = []
        hand = JawIOGripper(_PartBetweenTheJaws(events), actuation="single_solenoid", close_output_pin=0,
                            part_present_input_pin=1, confirm_open_at_start=True, pulse_s=0.0, close_settle_s=0.0,
                            ask=Person(), sleep=lambda _s: None)
        self.cell.session.service.runtime.orchestrator.gripper = hand
        detached: list[str] = []
        self.cell.session.arm.detach_payload = lambda: detached.append("detach") or True  # type: ignore[attr-defined]
        browser_jaws.install(self.cell)
        connecting = _InBackground(self.app, "POST", "/v1/cell/connect", json={"token": self.token()})
        asked = wait_for(lambda: self.client.get("/v1/cell/jaws").json().get("question"), what="the question")

        self.client.post("/v1/cell/jaws/answer", json={"question_id": asked["question_id"], "choice": "open"})
        connected = connecting.result()

        self.assertEqual(200, connected.status_code, connected.text)
        self.assertEqual("connected", connected.json()["state"])
        self.assertTrue(hand.jaws_closed, "the driver let go of a part its sensor reads")
        self.assertIsNone(self.cell.jaws_confirmed_at, "jaws the hand keeps closed were stamped open")
        self.assertEqual([], detached, "the planner was told a hand that holds a part is empty")
        (ended,) = self.cell_events("cell.jaws_ended")
        self.assertEqual(("refused", False), (ended.data["outcome"], ended.data["detached"]))
        self.assertIn("closed", ended.data["refusal"])
        self.assertFalse(self.client.get("/v1/cell").json()["jaws_question"])


class AHaltedArmIsRefusedAtConnectTests(ToggleCell):
    def test_connect_is_refused_while_the_built_arm_is_latched(self) -> None:
        self.arm.halt("the operator pressed halt now")

        refused = self.client.post("/v1/cell/connect", json={"token": self.token()})

        self.assertEqual((409, "halted"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("the cell is clear", refused.json()["message"])
        self.assertEqual("built", self.client.get("/v1/cell").json()["state"])
        self.assertEqual([], self.cell_events("cell.jaws_question"), "a latched arm's hand was asked")


class TheToggleSaysTheQuestionComesInTheBrowserTests(unittest.TestCase):
    def test_the_preview_of_a_toggle_names_the_browser_and_keeps_its_pinned_words(self) -> None:
        from api.lifecycle import motion_warnings
        from src.config.schema.robot import RobotConfig

        config = RobotConfig.model_validate({"vendor": "ur", "gripper": {"vendor": "jaw_io"}})
        (warning,) = motion_warnings(config, _toggle([], Person()))
        for part in ("ASKS where the jaws stand", "in the browser", "terminal", "refused", "120 s",
                     "every change of the output, on or off, moves the jaws"):
            self.assertIn(part, warning.what)
        self.assertIn("ONE CHANGE", warning.precaution)
        self.assertNotIn("pulse", (warning.what + warning.precaution).lower())


# ---------------------------------------------------------------------------------------------------------------------
# The check (Restart, Setup, the ready bar)
# ---------------------------------------------------------------------------------------------------------------------


class TheCheckAlwaysAsksTests(ToggleCell):
    def setUp(self) -> None:
        super().setUp()
        self.connected()

    def check_in_background(self) -> _InBackground:
        return _InBackground(self.app, "POST", "/v1/cell/jaws/check")

    def test_it_asks_even_where_the_count_says_open_and_an_open_answer_stamps_and_detaches(self) -> None:
        confirmed = self.cell.jaws_confirmed_at
        checking = self.check_in_background()
        asked = self.question()
        self.assertEqual(("where", "check"), (asked["stage"], asked["at"]))
        self.answer(asked, "open")

        checked = checking.result()

        self.assertEqual(200, checked.status_code, checked.text)
        self.assertEqual(("toggle", "open"), (checked.json()["hand"]["kind"], checked.json()["hand"]["jaws"]))
        self.assertIsNone(checked.json()["question"])
        self.assertEqual(0, _pulses(self.events))
        self.assertEqual(["detach"], self.detached)
        self.assertGreater(self.cell.jaws_confirmed_at, confirmed)
        self.assertFalse(self.cell.countdown_due())

    def test_open_now_on_a_count_that_says_closed_is_one_change_and_counts_down_the_next_motion(self) -> None:
        self.jaws.set_closed(True)                     # the last pick closed them on a part
        self.events.clear()
        checking = self.check_in_background()
        offered = self.question()
        self.assertEqual(("open_now", ["open_now", "abort"]), (offered["stage"], offered["choices"]),
                         "a count that says closed was asked where the jaws stand")
        self.answer(offered, "open_now")

        self.assertEqual(200, checking.result().status_code)
        self.assertEqual(1, _pulses(self.events))
        self.assertFalse(self.jaws.jaws_closed)
        self.assertEqual("jaws_opened", self.cell.countdown_because())

    def test_closed_and_abort_is_jaws_not_open_with_the_hand_s_sentence(self) -> None:
        checking = self.check_in_background()
        self.answer(self.question(stage="where"), "closed")
        self.answer(self.question(stage="open_now"), "abort")

        refused = checking.result()

        self.assertEqual((409, "jaws_not_open"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("CLOSED", refused.json()["message"])
        self.assertEqual(0, _pulses(self.events))
        self.assertTrue(self.jaws.jaws_closed, "a person said closed: the count stands closed")
        self.assertEqual([], self.detached)

    def test_an_output_switched_at_the_pendant_while_the_person_decided_sends_nothing(self) -> None:
        """The hand refuses its one change where DO0 no longer stands where the program left it: the count is nobody's
        then, and the check answers ``jaws_not_open`` with the hand's own words, nothing sent and nothing stamped."""
        self.jaws.set_closed(True)
        self.events.clear()
        confirmed = self.cell.jaws_confirmed_at
        checking = self.check_in_background()
        offered = self.question(stage="open_now")
        self.jaws._io.do[0] = not self.jaws._io.do[0]  # type: ignore[attr-defined] # somebody at the pendant
        self.answer(offered, "open_now")

        refused = checking.result()

        self.assertEqual((409, "jaws_not_open"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("RobotError", refused.json()["message"])
        self.assertEqual(0, _pulses(self.events), "a change went out on a count nobody keeps")
        self.assertEqual(confirmed, self.cell.jaws_confirmed_at)
        self.assertEqual("unknown", self.client.get("/v1/cell/jaws").json()["hand"]["jaws"])

    def test_a_stopped_controller_is_refused_before_anybody_is_asked(self) -> None:
        self.arm.quick_robot_status = lambda: STOPPED  # type: ignore[attr-defined]
        refused = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "controller_stopped"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual([], self.cell_events("cell.jaws_question")[1:], "the check asked a stopped cell")

    def test_a_controller_that_cannot_be_read_is_refused_as_one_that_cannot_move(self) -> None:
        def unreadable() -> RobotStatus:
            raise ConnectionError("RTDE receive dropped")

        self.arm.quick_robot_status = unreadable  # type: ignore[attr-defined]
        refused = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "controller_stopped"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertIn("could not be read", refused.json()["message"])

    def test_the_refusals_come_in_the_plan_s_order(self) -> None:
        self.arm.quick_robot_status = lambda: STOPPED  # type: ignore[attr-defined]
        self.arm.halt("the operator pressed halt now")
        self.cell.active_run_id = "run-busy"
        self.assertEqual("run_active", self.client.post("/v1/cell/jaws/check").json()["code"])
        self.cell.active_run_id = None
        self.assertEqual("halted", self.client.post("/v1/cell/jaws/check").json()["code"],
                         "a halt read as a controller stop")
        self.arm.clear_halt()
        self.assertEqual("controller_stopped", self.client.post("/v1/cell/jaws/check").json()["code"])
        self.arm.quick_robot_status = lambda: RUNNING  # type: ignore[attr-defined]
        checking = self.check_in_background()
        waiting = self.question()
        second = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "jaws_question_pending"), (second.status_code, second.json()["code"]), second.text)
        self.answer(waiting, "open")
        self.assertEqual(200, checking.result().status_code)
        self.jaws.answer_questions_with(None)
        missing = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "jaws_seam_missing"), (missing.status_code, missing.json()["code"]), missing.text)
        self.client.post("/v1/cell/disconnect")
        down = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "not_connected"), (down.status_code, down.json()["code"]), down.text)

    def test_a_question_is_pending_until_the_one_change_it_led_to_is_done(self) -> None:
        """Between the answer and the end of the stroke no question waits, and yet the jaws are moving where a person's
        hand is: a moving route started then would drive the arm while the jaws open. So the check stays pending until
        the change has gone out and its stroke is waited out."""
        from api import jaws as browser_jaws

        in_stroke, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def stroke(_seconds: float) -> None:
            in_stroke.set()
            release.wait(10.0)

        self.jaws.set_closed(True)                     # the last pick closed them on a part
        self.jaws._sleep = stroke  # type: ignore[attr-defined] # the stroke after the one change, held by the test
        checking = self.check_in_background()
        self.answer(self.question(stage="open_now"), "open_now")
        self.assertTrue(in_stroke.wait(10.0), "the change never went out")

        self.assertIsNotNone(browser_jaws.pending(self.cell), "nothing pending while the jaws open")
        self.assertTrue(self.client.get("/v1/cell").json()["jaws_question"])
        self.assertIsNone(self.client.get("/v1/cell/jaws").json()["question"], "an answered question was offered again")
        release.set()
        self.assertEqual(200, checking.result().status_code)
        self.assertIsNone(browser_jaws.pending(self.cell))

    def test_disconnect_during_a_check_ends_its_question(self) -> None:
        checking = self.check_in_background()
        self.question()
        down = self.client.post("/v1/cell/disconnect")
        self.assertEqual(200, down.status_code)
        refused = checking.result()
        self.assertEqual(409, refused.status_code, refused.text)
        self.assertEqual(0, _pulses(self.events))

    def test_a_disconnect_waits_for_the_one_change_the_check_is_sending(self) -> None:
        """The check holds the session while the hand acts on the answer: a Disconnect that comes in while the one
        change is out and its stroke runs waits for it, and never takes the hand down in the middle of it."""
        in_stroke, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)

        def stroke(_seconds: float) -> None:
            in_stroke.set()
            release.wait(10.0)

        self.jaws.set_closed(True)                     # the last pick closed them on a part
        self.events.clear()
        self.jaws._sleep = stroke  # type: ignore[attr-defined] # the stroke after the one change, held by the test
        checking = self.check_in_background()
        self.answer(self.question(stage="open_now"), "open_now")
        self.assertTrue(in_stroke.wait(10.0), "the change never went out")

        down = _InBackground(self.app, "POST", "/v1/cell/disconnect")
        down.thread.join(0.6)

        self.assertTrue(down.thread.is_alive(), "the Disconnect took the cell down in the middle of the one change")
        self.assertTrue(self.jaws.is_connected)
        release.set()
        self.assertEqual(200, down.result().status_code)
        self.assertEqual(200, checking.result().status_code)
        self.assertEqual(1, _pulses(self.events))
        self.assertFalse(self.jaws.is_connected)

    def _a_run(self) -> tuple[Any, threading.Event, threading.Event]:
        """What a moving route's start does once its gates have passed: a Home run that drives until ``done``.

        A moving run reads the question once more on its own thread, once it holds the cell, and gives way to a check
        that began meanwhile (``api.runs._jaws_question_began``, pinned in ``tests/test_api_pick_gates.py``). This one
        stands for a run whose read came just before the check marked itself, and found nothing: the check's own guards
        are what is tested here."""
        from api.codes import RunKind, StopCode

        moving, done = threading.Event(), threading.Event()
        self.addCleanup(done.set)

        def drive_home(_console: Any, _run: Any) -> StopCode:
            moving.set()
            done.wait(10.0)                            # the arm drives meanwhile
            return StopCode.FINISHED

        with patch("api.runs._jaws_question_began", return_value=""):
            run = self.cell.registry.start_kind(self.cell, RunKind.HOME, drive_home)
            self.assertTrue(moving.wait(5.0), "the run never started")
        return run, moving, done

    def test_a_check_is_pending_from_its_start_and_a_run_that_started_meanwhile_refuses_it(self) -> None:
        """Found 2026-10-01: the check read the run lock first, then the latch and the controller (a dashboard round
        trip on a cell), and only then marked its question: a moving route that started in between passed every gate,
        and the check asked about the jaws, and sent "open now", while the run drove the arm. Now the check is pending
        before it reads the run lock, so a moving route refuses it meanwhile; and a run that started all the same is
        read again before anybody is asked: ``run_active``, nothing asked, nothing sent."""
        from api import jaws as browser_jaws

        self.jaws.set_closed(True)                     # a pick closed them on a part
        self.events.clear()
        asked_before = len(self.cell_events("cell.jaws_question"))
        in_read, release = threading.Event(), threading.Event()
        self.addCleanup(release.set)
        real = browser_jaws.controller_stop
        reads: list[str] = []

        def slow_controller(arm: Any) -> str:
            reads.append("read")
            if len(reads) == 1:                        # the check's own read, held by the test
                in_read.set()
                release.wait(10.0)
            return real(arm)

        with patch("api.jaws.controller_stop", side_effect=slow_controller):
            checking = self.check_in_background()
            self.assertTrue(in_read.wait(10.0), "the check never read the controller")
            self.assertIsNotNone(browser_jaws.pending(self.cell), "the check was not marked before it read the run lock")
            self.assertTrue(self.client.get("/v1/cell").json()["jaws_question"])
            run, _moving, done = self._a_run()         # a route whose gate read pending() before the check began
            release.set()
            refused = checking.result()

        self.assertEqual((409, "run_active"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(run.id, refused.json()["detail"]["run_id"])
        self.assertEqual(asked_before, len(self.cell_events("cell.jaws_question")), "the check asked during a run")
        self.assertEqual(0, _pulses(self.events), "a change of DO0 went out while a run drove the arm")
        self.assertIsNone(browser_jaws.pending(self.cell))
        done.set()
        wait_for(lambda: self.cell.active_run_id is None, what="the Home run's end")

    def test_a_run_that_starts_while_the_check_s_question_waits_ends_the_question(self) -> None:
        """A run that started past every gate while the check's question waits: the question is ended at once (no
        answer, never "open"), the check answers ``run_active``, and nothing is sent; a late answer finds no question."""
        self.jaws.set_closed(True)
        self.events.clear()
        checking = self.check_in_background()
        asked = self.question(stage="open_now")

        run, _moving, done = self._a_run()
        refused = checking.result(timeout=5.0)

        self.assertEqual((409, "run_active"), (refused.status_code, refused.json()["code"]), refused.text)
        self.assertEqual(run.id, refused.json()["detail"]["run_id"])
        self.assertIsNone(self.client.get("/v1/cell/jaws").json()["question"], "the question stayed up during a run")
        late = self.answer(asked, "open_now")
        self.assertEqual((404, "no_question"), (late.status_code, late.json()["code"]), late.text)
        (ended,) = [e for e in self.cell_events("cell.jaws_ended") if e.data["question_id"] == asked["question_id"]]
        self.assertEqual(("cancelled", False), (ended.data["outcome"], ended.data["detached"]))
        self.assertEqual(0, _pulses(self.events))
        self.assertTrue(self.jaws.jaws_closed, "the count moved for a question nobody answered")
        done.set()
        wait_for(lambda: self.cell.active_run_id is None, what="the Home run's end")


class AHandThatIsNotAToggleIsAskedNothingTests(ScratchCell):
    def test_the_check_answers_at_once_with_nothing_asked(self) -> None:
        self.build()
        self.assertEqual(200, self.client.post("/v1/cell/connect", json={"token": self.token()}).status_code)
        checked = self.client.post("/v1/cell/jaws/check")
        self.assertEqual(200, checked.status_code, checked.text)
        self.assertEqual("width", checked.json()["hand"]["kind"])
        self.assertIsNone(checked.json()["question"])
        self.assertEqual([], self.cell_events("cell.jaws_question"))
        self.assertIsNone(self.cell.jaws_confirmed_at, "a hand nobody asked about was stamped open")

    def test_the_hand_and_no_question_are_read_with_no_cell_at_all(self) -> None:
        body = self.client.get("/v1/cell/jaws")
        self.assertEqual(200, body.status_code, body.text)
        self.assertEqual(("none", None), (body.json()["hand"]["kind"], body.json()["question"]))
        refused = self.client.post("/v1/cell/jaws/check")
        self.assertEqual((409, "not_connected"), (refused.status_code, refused.json()["code"]))


class TheSeamItselfTests(unittest.TestCase):
    """``api.jaws.BrowserJawQuestions`` on its own, as the hand calls it: every way a question ends without a person's
    answer is ``EOFError``, which the hand takes as no answer, never "open"."""

    def setUp(self) -> None:
        from api.events import EventHub
        from api.jaws import BrowserJawQuestions

        self.hub = EventHub()
        self.seam = BrowserJawQuestions(self.hub, timeout_s=30.0)

    def asking(self, **fields: Any) -> Any:
        from tests.test_api_contract import _JawAsking

        return _JawAsking(**fields)

    def test_a_cancel_between_two_questions_ends_the_next_one_at_once(self) -> None:
        """A Disconnect that lands after "closed" was answered and before "open now" is asked finds no question
        waiting; the next question of that check is no answer at once, so the Disconnect never waits 120 s for it."""
        flow = self.seam.open_flow("check")
        self.assertFalse(self.seam.cancel("the cell was disconnected while the question waited"))
        started = time.monotonic()
        with self.assertRaises(EOFError):
            self.seam(self.asking(stage="open_now", choices=("open_now", "abort"), at_connect=False))
        self.assertLess(time.monotonic() - started, 2.0, "the next question waited for an answer")
        self.assertIn("cancelled", flow.unanswered_said())
        self.seam.close_flow(flow, opened=False, refusal="nothing was sent")
        self.assertIsNone(self.seam.pending())

    def test_a_second_question_while_one_waits_is_no_answer(self) -> None:
        first: dict[str, Any] = {}
        asking = threading.Thread(target=lambda: first.setdefault("said", self._ask_quietly()), daemon=True)
        asking.start()
        waiting = wait_for(lambda: self.seam.waiting(), what="the first question")
        started = time.monotonic()
        with self.assertRaises(EOFError):
            self.seam(self.asking())
        self.assertLess(time.monotonic() - started, 2.0, "the second question waited instead of being refused")
        self.assertEqual(waiting.question_id, self.seam.waiting().question_id, "the second question took its place")
        self.seam.cancel("the test ends the first question")
        asking.join(5.0)
        self.assertEqual("no answer", first["said"])

    def _ask_quietly(self) -> str:
        try:
            return self.seam(self.asking())
        except EOFError:
            return "no answer"

    def test_a_question_the_browser_cannot_show_is_no_answer(self) -> None:
        started = time.monotonic()
        with self.assertRaises(EOFError):
            self.seam(self.asking(choices=("p", "a")))         # letters the browser never offers
        with self.assertRaises(EOFError):
            self.seam(self.asking(stage="pulse"))               # a stage the console does not know
        self.assertLess(time.monotonic() - started, 2.0, "a question nobody could answer waited for an answer")
        self.assertIsNone(self.seam.pending())
        self.assertEqual([], self.hub.since("cell", 0)[0], "a question nobody could answer was shown")

    def test_a_flow_is_one_at_a_time(self) -> None:
        from api.jaws import JawsRefused

        flow = self.seam.open_flow("check")
        with self.assertRaises(JawsRefused) as caught:
            self.seam.open_flow("check")
        self.assertEqual("jaws_question_pending", str(caught.exception.code))
        self.seam.close_flow(flow, opened=False)
        self.seam.close_flow(self.seam.open_flow("check"), opened=False)


class InstallingTheSeamTests(unittest.TestCase):
    def test_a_hand_that_asks_and_cannot_take_the_seam_fails_the_install(self) -> None:
        from types import SimpleNamespace

        from api import jaws as browser_jaws
        from api.cell import Console

        class _ToggleWithoutSeam:
            toggles_without_sensor = True
            jaws_closed = False
            edge_unknown = False
            is_connected = False

            def jaws_open_for_a_pick(self) -> str:
                return ""

        console = Console(root=_ROOT / "config", profile=None)
        console.session.service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            arm=None, gripper=_ToggleWithoutSeam())))
        with self.assertRaises(RuntimeError) as caught:
            browser_jaws.install(console)
        self.assertIn("browser", str(caught.exception))

    def test_the_gate_reads_the_latch_before_the_controller_and_lets_only_a_moving_controller_through(self) -> None:
        from types import SimpleNamespace

        from api import jaws as browser_jaws
        from api.cell import Console
        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm()
        arm.connect()
        self.addCleanup(arm.disconnect)
        console = Console(root=_ROOT / "config", profile=None)
        console.session.service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            arm=arm, gripper=None)))
        gate = browser_jaws.before_change(console)
        self.assertEqual("", gate(), "an arm that reports nothing was refused")
        arm.quick_robot_status = lambda: RUNNING  # type: ignore[attr-defined]
        self.assertEqual("", gate())
        arm.quick_robot_status = lambda: STOPPED  # type: ignore[attr-defined]
        self.assertIn("controller cannot move", gate())
        arm.halt("the operator pressed halt now")
        said = gate()
        self.assertIn("halted", said)
        self.assertNotIn("controller cannot move", said, "a halt read as a controller stop")

    def _console_on(self, arm: Any) -> Any:
        from api.cell import Console

        console = Console(root=_ROOT / "config", profile=None)
        console.session.service = SimpleNamespace(runtime=SimpleNamespace(orchestrator=SimpleNamespace(
            arm=arm, gripper=None)))
        return console

    def _dummy_arm(self) -> Any:
        from src.robot.drivers.dummy.arm import DummyRobotArm

        arm = DummyRobotArm()
        arm.connect()
        self.addCleanup(arm.disconnect)
        arm.quick_robot_status = lambda: RUNNING  # type: ignore[attr-defined]
        return arm

    def test_the_gate_lets_no_change_out_while_any_run_is_active(self) -> None:
        """Whatever a route or a race let start: while a run holds the cell, the one change of "open now" never goes
        out, so it never opens the jaws under a run that drives the arm or a teach whose arm is free."""
        from api import jaws as browser_jaws

        console = self._console_on(self._dummy_arm())
        gate = browser_jaws.before_change(console)
        self.assertEqual("", gate())
        console.active_run_id = "run-moving"

        said = gate()

        self.assertIn("run-moving", said)
        self.assertIn("not changed", said)
        console.active_run_id = None
        self.assertEqual("", gate())

    def test_a_check_whose_hand_does_not_say_its_jaws_stand_open_stamps_nothing(self) -> None:
        """The hand's answer that the jaws stand open is read off the hand itself, while the check still holds the
        session: a hand that answers its check and still says closed is not taken for open, and nothing is stamped."""
        from api import jaws as browser_jaws
        from api.lifecycle import CellState

        class _SaysOpenKeepsClosed:
            toggles_without_sensor = True
            jaws_closed = True
            edge_unknown = False
            is_connected = True
            commands_sent = 0

            def jaws_open_for_a_pick(self) -> str:
                return ""

            def question_seam_installed(self) -> bool:
                return True

            def confirm_where_the_jaws_stand(self, reason: str, *, before_change: Any = None) -> str:
                return ""

        arm = self._dummy_arm()
        detached: list[str] = []
        arm.detach_payload = lambda: detached.append("detach") or True  # type: ignore[attr-defined]
        console = self._console_on(arm)
        console.session.service.runtime.orchestrator.gripper = _SaysOpenKeepsClosed()
        console.session.state = CellState.CONNECTED

        with self.assertRaises(browser_jaws.JawsRefused) as caught:
            browser_jaws.check(console)

        self.assertEqual("jaws_not_open", str(caught.exception.code))
        self.assertIn("closed", caught.exception.message)
        self.assertIsNone(console.jaws_confirmed_at)
        self.assertEqual([], detached)
        self.assertIsNone(browser_jaws.pending(console))

    def test_a_controller_status_that_does_not_say_it_can_move_lets_nothing_out(self) -> None:
        """Only a status that says the controller can move lets the change out: one that does not say is no answer."""
        from api import jaws as browser_jaws

        arm = self._dummy_arm()
        console = self._console_on(arm)
        arm.quick_robot_status = lambda: SimpleNamespace(  # type: ignore[attr-defined]
            robot_mode=RobotMode.RUNNING, safety_mode=SafetyMode.NORMAL)

        said = browser_jaws.before_change(console)()

        self.assertTrue(said, "a status that does not say the controller can move let the change out")
        self.assertIn("does not say", said)
        self.assertIn("does not say", browser_jaws.controller_stop(arm))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
