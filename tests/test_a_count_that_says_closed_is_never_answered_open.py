"""A count that says CLOSED is never answered "open" at the check after a stop (review of 2026-10-02, safety, high).

The owner's hand is a Hand-E on tool DO0, a ``single_toggle`` with no sensor: the program counts its own changes of the
output. A task halted in the carry ends with the count CLOSED, DO0 still at the level of the program's own close, and
the stop record saying a part is held. The check before the way back used to ask "where do the jaws stand?" all the
same, with "Offen" as its first button: one wrong click took the count to OPEN with nothing sent, told the planner the
hand was empty, and let Restart and Home drive with the part still clamped; the next "close" then opened the jaws and
dropped it. What is held here:

* where the program's own count says CLOSED and the output still stands where its last change left it, the check
  (``JawIOGripper.confirm_where_the_jaws_stand``) asks no "where": it asks only what follows from closed, "open them now
  with one change, while you hold the part" or "abort". "open" is no answer to that question, through the console's
  seam and at a terminal alike;
* an abort keeps the count CLOSED and says how a count starts anew where the jaws stand open all the same: connect
  again, and the connect asks where they stand;
* where the count cannot vouch for itself (the output switched at the pendant, a change that failed) or says OPEN,
  the check asks where they stand, both answers offered, as before;
* through the console: the halted task's way back stays shut (``part_still_held``) until "Jetzt öffnen" sent its one
  change of DO0; "open" is refused ``choice_not_offered``, an abort is ``jaws_not_open``, and neither tells the planner
  the hand is empty.

Honesty bucket (2): the real ``JawIOGripper`` on a recording tool I/O; the real console, routes and ``run_task`` on the
scripted owner's cell.
"""

from __future__ import annotations

import threading
import time
import unittest
from typing import Any
from unittest.mock import patch

from src.robot.core import RobotError
from src.robot.grippers.jaw_io import JawIOGripper
from tests._console_task_fakes import ConsoleCase, Scripted
from tests.test_a_toggle_asks_where_its_jaws_stand import Person, _pulses, _toggle
from tests.test_the_jaws_question_reaches_whoever_the_console_names import Browser

_CHECK = "the console asks where they stand before the arm moves again"


def _closed_on_a_part(events: list[Any], browser: Any) -> JawIOGripper:
    """The owner's toggle, connected open, then closed on a part by the program; the log cleared, ``browser`` asking."""
    jaws = _toggle(events, Person())
    jaws.answer_questions_with(Browser("open"))
    jaws.connect()
    jaws.set_closed(True)
    events.clear()
    jaws.answer_questions_with(browser)
    return jaws


class ACountThatSaysClosedIsAskedOnlyWhetherToOpenThemTests(unittest.TestCase):
    def test_the_check_asks_only_open_now_or_abort(self) -> None:
        events: list[Any] = []
        browser = Browser("open_now")
        jaws = _closed_on_a_part(events, browser)

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK))

        (asked,) = browser.asked
        self.assertEqual(("open_now", ("open_now", "abort"), False), (asked.stage, asked.choices, asked.at_connect))
        self.assertIn("count", asked.text)
        self.assertIn("CLOSED", asked.text)
        self.assertEqual(1, _pulses(events), "open now is exactly one change of DO0")
        self.assertFalse(jaws.jaws_closed)

    def test_open_is_no_answer_to_it_and_nothing_is_sent(self) -> None:
        events: list[Any] = []
        browser = Browser("open", "open", "open")
        jaws = _closed_on_a_part(events, browser)

        refused = jaws.confirm_where_the_jaws_stand(_CHECK)

        self.assertTrue(refused, "an answer of open against a count that says closed was taken")
        self.assertEqual(3, len(browser.asked))
        for asked in browser.asked:
            self.assertNotIn("open", asked.choices)
        self.assertEqual("'open' is none of the choices.", browser.asked[1].why_again)
        self.assertTrue(jaws.jaws_closed, "the count was turned round with nothing sent")
        self.assertEqual(0, _pulses(events))

    def test_an_abort_keeps_the_count_closed_and_says_how_a_count_starts_anew(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events, Browser("abort"))

        refused = jaws.confirm_where_the_jaws_stand(_CHECK)

        self.assertIn("count says the jaws on tool output 0 stand CLOSED", refused)
        self.assertIn("chose not to open them", refused)
        self.assertIn("connect again", refused)
        self.assertNotIn("\n", refused)
        self.assertTrue(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))

    def test_no_answer_is_a_refusal_never_open(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events, Browser("EOF"))
        refused = jaws.confirm_where_the_jaws_stand(_CHECK)
        self.assertIn("gave no clear answer to whether to open them", refused)
        self.assertTrue(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))

    def test_the_terminal_is_asked_the_same_and_takes_no_open(self) -> None:
        """A program at the terminal (a string seam): the same question, its letters, and no "open" among them."""
        events: list[Any] = []
        person = Person("", "open", "p")
        jaws = _toggle(events, person)
        jaws.connect()
        jaws.set_closed(True)
        events.clear()

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK))

        self.assertEqual(3, len(person.asked), "the connect, then the check twice: 'open' is no answer to it")
        self.assertIn("count says they stand CLOSED", person.asked[1])
        self.assertIn("[p] open them now", person.asked[1])
        self.assertNotIn("Do they stand OPEN?", person.asked[1])
        self.assertIn("'open' is none of the choices", person.asked[2])
        self.assertEqual(1, _pulses(events))
        self.assertFalse(jaws.jaws_closed)

    def test_the_gate_is_still_asked_right_before_the_one_change(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events, Browser("open_now"))
        said = "the controller reads a protective stop: clear it at the pendant first"

        refused = jaws.confirm_where_the_jaws_stand(_CHECK, before_change=lambda: said)

        self.assertIn(said, refused)
        self.assertIn("nothing was sent", refused)
        self.assertIn("count says the jaws on tool output 0 stand CLOSED", refused)
        self.assertEqual(0, _pulses(events))
        self.assertTrue(jaws.jaws_closed)

    def test_an_output_switched_at_the_pendant_while_open_now_waits_sends_nothing(self) -> None:
        events: list[Any] = []

        def seam(_asking: Any) -> str:
            jaws._io.do[0] = not jaws._io.do.get(0, False)  # type: ignore[attr-defined] # at the pendant, meanwhile
            return "open_now"

        jaws = _closed_on_a_part(events, seam)
        with self.assertRaises(RobotError) as caught:
            jaws.confirm_where_the_jaws_stand(_CHECK)
        self.assertIn("Nothing was sent", str(caught.exception))
        self.assertEqual(0, _pulses(events))
        self.assertTrue(jaws.edge_unknown, "the next question has to vouch for the count again")


class ACountThatCannotVouchOrSaysOpenIsAskedWhereTests(unittest.TestCase):
    """⭐ THE CONTROLS: the check still asks where the jaws stand, both answers offered, where nothing contradicts one."""

    def test_an_output_switched_at_the_pendant_is_asked_where_and_open_is_taken(self) -> None:
        events: list[Any] = []
        browser = Browser("open")
        jaws = _closed_on_a_part(events, browser)
        jaws._io.do[0] = not jaws._io.do.get(0, False)      # type: ignore[attr-defined] # opened at the pendant

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK))

        (asked,) = browser.asked
        self.assertEqual(("where", ("open", "closed")), (asked.stage, asked.choices))
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))

    def test_a_change_that_failed_is_asked_where(self) -> None:
        events: list[Any] = []
        browser = Browser("closed", "open_now")
        jaws = _closed_on_a_part(events, browser)
        jaws._io.refuse_all = True                          # type: ignore[attr-defined] # the open's write fails
        with self.assertRaises(RobotError):
            jaws.set_closed(False)
        jaws._io.refuse_all = False                         # type: ignore[attr-defined]
        self.assertTrue(jaws.edge_unknown)

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK))

        self.assertEqual(["where", "open_now"], [asked.stage for asked in browser.asked])

    def test_a_count_that_says_open_is_asked_where(self) -> None:
        events: list[Any] = []
        browser = Browser("open")
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("open"))
        jaws.connect()
        jaws.answer_questions_with(browser)

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK))

        self.assertEqual(["where"], [asked.stage for asked in browser.asked])
        self.assertEqual(("open", "closed"), browser.asked[0].choices)


# ---------------------------------------------------------------------------------------------------------------------
# Through the console: a task halted in the carry (scenario P4/P10 of the 2026-10-02 safety review)
# ---------------------------------------------------------------------------------------------------------------------


class AStopInTheCarryOpensOnlyByOneChangeTests(ConsoleCase):
    def _halted_in_the_carry(self) -> tuple[Any, dict[str, Any]]:
        """The scripted owner's cell: "Sofort anhalten" while the task carries its part; then "Zelle ist frei"."""
        from api import jaws as browser_jaws

        pressed: dict[str, Any] = {}
        cell = self.scripted([Scripted("part"), Scripted("part")])

        def brake_in_the_carry(_kind: str, _index: int) -> None:
            if pressed or cell.arm.payload_model().value != "planner_and_filter":
                return
            pressed["answer"] = self.client.post("/v1/cell/brake").json()

        cell.arm.on_motion = brake_in_the_carry
        run = self.task(object="green cube", scope="once")
        body = self.finished(run["id"])
        self.assertEqual(("halted", True), (body["stop_code"], body["holding"]), body)
        self.assertEqual("closed", self.client.get("/v1/cell").json()["hand"]["jaws"])
        self.assertEqual(200, self.client.post("/v1/cell/acknowledge", json={"cell_clear": True}).status_code)
        # The browser's seam in place of the scripted cell's own, as every build hands it (``api.jaws.install``).
        cell.jaws.answer_questions_with(self.cell.jaws, before_change=browser_jaws.before_change(self.cell))
        return cell, body

    def _check(self, *choices: str) -> tuple[Any, list[tuple[dict[str, Any], int]]]:
        """``POST /v1/cell/jaws/check``, each question it asks answered in the browser with the next of ``choices``:
        what the check answered, and every question with the status its answer got."""
        answered: list[tuple[dict[str, Any], int]] = []
        pending = list(choices)

        def answer() -> None:
            deadline = time.monotonic() + 20.0
            while pending and time.monotonic() < deadline:
                waiting = self.client.get("/v1/cell/jaws").json().get("question")
                if waiting and (not answered or answered[-1][1] != 200 or waiting["question_id"]
                                != answered[-1][0]["question_id"]):
                    choice = pending.pop(0)
                    status = self.client.post("/v1/cell/jaws/answer", json={
                        "question_id": waiting["question_id"], "choice": choice}).status_code
                    answered.append((dict(waiting, answered=choice), status))
                    continue
                time.sleep(0.02)

        helper = threading.Thread(target=answer, daemon=True)
        helper.start()
        # A question nobody answers ends in seconds here, not in two minutes: no answer is never "open" either way.
        with patch("api.jaws.JAW_QUESTION_TIMEOUT_S", 5.0):
            checked = self.client.post("/v1/cell/jaws/check")
        helper.join(timeout=20.0)
        return checked, answered

    def test_open_is_not_offered_and_the_way_back_stays_shut_until_open_now(self) -> None:
        cell, stopped = self._halted_in_the_carry()
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "part_still_held")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")
        before, detached = cell.do0_changes(), cell.arm.calls.count("detach_payload")

        checked, answered = self._check("open", "abort")

        first, status = answered[0]
        self.assertEqual(("open_now", ["open_now", "abort"]), (first["stage"], first["choices"]))
        self.assertEqual(422, status, "an answer of open against the count was taken")
        self.assertEqual(("abort", 200), (answered[1][0]["answered"], answered[1][1]))
        self.assertEqual(first["question_id"], answered[1][0]["question_id"], "the refused answer ended the question")
        self.assertEqual((409, "jaws_not_open"), (checked.status_code, checked.json()["code"]), checked.text)
        self.assertEqual(0, cell.do0_changes() - before)
        now = self.client.get("/v1/cell").json()
        self.assertEqual(("closed", "planner_and_filter"), (now["hand"]["jaws"], now["payload_model"]))
        self.assertEqual(detached, cell.arm.calls.count("detach_payload"), "the planner was told the hand is empty")
        self.refused(self.client.post("/v1/task/restart", json={"run_id": stopped["id"]}), 409, "part_still_held")
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")

    def test_open_now_is_one_change_of_do0_and_then_restart_moves(self) -> None:
        cell, stopped = self._halted_in_the_carry()
        before = cell.do0_changes()

        checked, answered = self._check("open_now")

        self.assertEqual(200, checked.status_code, checked.text)
        self.assertEqual([("open_now", 200)], [(asked["stage"], status) for asked, status in answered])
        self.assertEqual(1, cell.do0_changes() - before, "open now is exactly one change of DO0")
        now = self.client.get("/v1/cell").json()
        self.assertEqual(("open", "none"), (now["hand"]["jaws"], now["payload_model"]))
        self.assertTrue(now["countdown_due"], "a person held the part at the jaws: the next motion counts down")
        restart = self.client.post("/v1/task/restart", json={"run_id": stopped["id"]})
        self.assertEqual(202, restart.status_code, restart.text)
        self.assertEqual("finished", self.finished(restart.json()["id"])["stop_code"])

    def test_home_too_waits_for_open_now(self) -> None:
        cell, _stopped = self._halted_in_the_carry()
        motions = len(cell.motions())
        checked, _answered = self._check("open", "abort")
        self.assertEqual((409, "jaws_not_open"), (checked.status_code, checked.json()["code"]), checked.text)
        self.refused(self.client.post("/v1/cell/home", json={"to": "home"}), 409, "part_still_held")
        self.assertEqual(motions, len(cell.motions()), "a refused way back moved the arm")
        checked, _answered = self._check("open_now")
        self.assertEqual(200, checked.status_code, checked.text)
        home = self.client.post("/v1/cell/home", json={"to": "home"})
        self.assertEqual(202, home.status_code, home.text)
        self.assertEqual("finished", self.finished(home.json()["id"])["stop_code"])
        self.assertIsNone(self.client.get("/v1/cell").json()["recovery"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
