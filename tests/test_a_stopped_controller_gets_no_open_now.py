"""A stopped controller gets no "open now": the controller is asked right before the one change of DO0, and a refusal
sends nothing (build plan 1.6, item 10; the order of ``service.py``: the controller before the hand).

"Jetzt öffnen" in the jaws question is ONE change of the toggle's output, which releases whatever is between the jaws.
On a stopped arm (a protective stop, a halt) that drops the part from wherever the arm stopped, likely while a person
walks to the pendant (the owner-cell audit of 2026-09-23). Today ``connect_cell`` asks the controller nothing between
the arm and the hand, so the hand takes a gate, ``before_change``, that is asked under its lock immediately before that
one change: installed with the console's seam (``answer_questions_with(..., before_change=)``), which reaches the
connect, and handed to the check (``confirm_where_the_jaws_stand(..., before_change=)``). What is held here:

* a gate that answers anything refuses the change with nothing sent, and the refusal names what it said;
* only the empty sentence lets the change go out: a gate that answers no sentence at all (``None``, ``False``, a
  number, a blank) refuses too, so a gate that forgot its return fails closed;
* the gate is asked after the person chose "open now" and before the write, once, and never where nothing is sent;
* a gate that raises refuses too, with nothing sent;
* every gate there is is asked (the installed one and the check's own), and any one of them refuses;
* a disconnect that lands while a gate reads the controller sends nothing: the connection is read again after the
  gates, right before the change;
* the gate comes and goes with the console's seam: setting ``before_change`` on the hand, the plan's spelling, is
  refused pointing at ``answer_questions_with``, so it can never be set and silently ignored;
* a person's "closed" is where the count stands from then on, whatever comes after it.

A check on a part the program closed the jaws on asks no "where" (the count says CLOSED, and nothing moved them since):
its first question is "open now" (``tests/test_a_count_that_says_closed_is_never_answered_open.py``).
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import RobotError
from src.robot.grippers.jaw_io import JawAsking, JawIOGripper
from tests.test_a_toggle_asks_where_its_jaws_stand import _IO, Person, _pulses, _toggle, _Write
from tests.test_the_jaws_question_reaches_whoever_the_console_names import Browser

_STOPPED = "the controller reads a protective stop: clear it at the pendant first"
_CHECK = "a Restart after a stop"


class _Gate:
    """The console's controller check: answers ``said`` and writes "gate" on the shared log when asked."""

    def __init__(self, events: list[Any], said: str = "") -> None:
        self.events = events
        self.said = said
        self.asked = 0

    def __call__(self) -> str:
        self.asked += 1
        self.events.append("gate")
        return self.said


class _LoggedBrowser(Browser):
    """The console's seam, writing each question onto the shared log."""

    def __init__(self, events: list[Any], *answers: str) -> None:
        super().__init__(*answers)
        self.events = events

    def __call__(self, asking: JawAsking) -> str:
        self.events.append(("asked", asking.stage))
        return super().__call__(asking)


def _closed_on_a_part(events: list[Any]) -> JawIOGripper:
    """The owner's toggle, connected open and then closed on a part, the log cleared."""
    jaws = _toggle(events, Person())
    jaws.answer_questions_with(Browser("open"))
    jaws.connect()
    jaws.set_closed(True)
    events.clear()
    return jaws


class AtTheCheckTests(unittest.TestCase):
    def test_a_refusing_gate_sends_no_change_and_the_refusal_names_it(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events)
        jaws.answer_questions_with(_LoggedBrowser(events, "open_now"))
        gate = _Gate(events, _STOPPED)

        refused = jaws.confirm_where_the_jaws_stand(_CHECK, before_change=gate)

        self.assertIn(_STOPPED, refused)
        self.assertIn("nothing was sent", refused)
        self.assertNotIn("\n", refused)
        self.assertEqual(0, _pulses(events))
        self.assertEqual(1, gate.asked)
        self.assertTrue(jaws.jaws_closed, "the count says closed and nothing opened them")
        self.assertTrue(jaws.is_connected)

    def test_the_gate_is_asked_after_the_choice_and_right_before_the_change(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events)
        jaws.answer_questions_with(_LoggedBrowser(events, "open_now"))

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_CHECK, before_change=_Gate(events)))

        self.assertEqual([("asked", "open_now"), "gate", ("DO", 0, False)], events)
        self.assertIsInstance(events[-1], _Write)
        self.assertFalse(jaws.jaws_closed)

    def test_no_gate_is_asked_where_nothing_is_sent(self) -> None:
        for script in (("abort",), ("EOF",), ("open", "open", "open"), ("closed", "closed", "closed")):
            with self.subTest(script=script):
                events: list[Any] = []
                jaws = _closed_on_a_part(events)
                jaws.answer_questions_with(Browser(*script))
                gate = _Gate(events, _STOPPED)
                jaws.confirm_where_the_jaws_stand(_CHECK, before_change=gate)
                self.assertEqual(0, gate.asked)
                self.assertEqual(0, _pulses(events))

    def test_a_gate_that_raises_refuses_with_nothing_sent(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events)
        jaws.answer_questions_with(Browser("open_now"))

        def broken() -> str:
            raise ConnectionError("RTDE receive dropped")

        refused = jaws.confirm_where_the_jaws_stand(_CHECK, before_change=broken)

        self.assertIn("ConnectionError", refused)
        self.assertIn("RTDE receive dropped", refused)
        self.assertIn("nothing was sent", refused)
        self.assertEqual(0, _pulses(events))
        self.assertTrue(jaws.jaws_closed)

    def test_the_installed_gate_and_the_checks_own_are_both_asked(self) -> None:
        for installed, own, sent in (("", "", 1), (_STOPPED, "", 0), ("", _STOPPED, 0)):
            with self.subTest(installed=installed, own=own):
                events: list[Any] = []
                jaws = _closed_on_a_part(events)
                jaws.answer_questions_with(Browser("open_now"), before_change=_Gate(events, installed))
                jaws.confirm_where_the_jaws_stand(_CHECK, before_change=_Gate(events, own))
                self.assertEqual(sent, _pulses(events))

    def test_a_gate_that_answers_no_sentence_lets_nothing_through(self) -> None:
        """⛔ The gate's contract is a sentence, ``""`` to let the change go: a gate that forgot its return when the
        controller is stopped answers ``None``, and that used to let the one change of DO0 out (review of 2026-10-01)."""
        for said in (None, False, 0, [], " \n ", b""):
            with self.subTest(said=said):
                events: list[Any] = []
                jaws = _closed_on_a_part(events)
                jaws.answer_questions_with(Browser("open_now"))

                refused = jaws.confirm_where_the_jaws_stand(_CHECK, before_change=lambda said=said: said)

                self.assertIn("nothing was sent", refused)
                self.assertIn(repr(said), refused)
                self.assertEqual(0, _pulses(events))
                self.assertTrue(jaws.jaws_closed)


class AConnectionTakenDownDuringTheGateTests(unittest.TestCase):
    """⛔ The gate reads the controller (an RTDE read) while the person's answer stands: a Disconnect that lands then
    used to get the one change of DO0 sent all the same, and the check answered "open" for a hand no longer
    connected (review of 2026-10-01). The connection is read again after the gates, right before the change."""

    def test_at_the_check_it_sends_nothing_and_is_refused(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events)
        jaws.answer_questions_with(Browser("open_now"))

        def gate() -> str:
            jaws.disconnect()                               # the console's Disconnect, while the controller is read
            return ""

        refused = jaws.confirm_where_the_jaws_stand(_CHECK, before_change=gate)

        self.assertIn("changed while", refused)
        self.assertIn("nothing was sent", refused)
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)

    def test_at_connect_it_sends_nothing_and_the_connect_is_refused(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())

        def gate() -> str:
            jaws.disconnect()
            return ""

        jaws.answer_questions_with(Browser("closed", "open_now"), before_change=gate)
        with self.assertRaises(RobotError) as caught:
            jaws.connect()
        self.assertIn("nothing was sent", str(caught.exception))
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)


class TheGateComesWithTheSeamTests(unittest.TestCase):
    """The build plan names the gate ``before_change`` beside the seam. On the hand it is set through the seam alone:
    a gate set on its own would outlive the seam it belongs to, or be dropped by the next install without a word."""

    def test_setting_the_gate_on_the_hand_is_refused_pointing_at_the_seam(self) -> None:
        jaws = _toggle([], Person())
        with self.assertRaises(AttributeError) as caught:
            jaws.before_change = _Gate([])  # type: ignore[misc]
        self.assertIn("answer_questions_with", str(caught.exception))
        self.assertIsNone(jaws.before_change)

    def test_the_installed_gate_reads_back_and_goes_with_its_seam(self) -> None:
        jaws = _toggle([], Person())
        gate = _Gate([])
        jaws.answer_questions_with(Browser(), before_change=gate)
        self.assertIs(gate, jaws.before_change)
        jaws.answer_questions_with(None)
        self.assertIsNone(jaws.before_change)


class AtConnectTests(unittest.TestCase):
    def test_the_installed_gate_refuses_the_connect_with_nothing_sent(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        gate = _Gate(events, _STOPPED)
        jaws.answer_questions_with(_LoggedBrowser(events, "closed", "open_now"), before_change=gate)

        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        self.assertIn("not connecting", str(caught.exception))
        self.assertIn(_STOPPED, str(caught.exception))
        self.assertEqual(1, gate.asked)
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)

    def test_a_gate_that_lets_it_through_opens_them_with_one_change(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(_LoggedBrowser(events, "closed", "open_now"), before_change=_Gate(events))
        jaws.connect()
        self.assertEqual([("asked", "where"), ("asked", "open_now"), "gate", ("DO", 0, True)], events)
        self.assertTrue(jaws.is_connected)
        self.assertFalse(jaws.jaws_closed)

    def test_an_open_answer_at_connect_asks_no_gate(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        gate = _Gate(events, _STOPPED)
        jaws.answer_questions_with(Browser("open"), before_change=gate)
        jaws.connect()
        self.assertEqual(0, gate.asked)
        self.assertTrue(jaws.is_connected)

    def test_taking_the_seam_away_takes_its_gate_away(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person("closed", "p"))
        jaws.answer_questions_with(Browser(), before_change=_Gate(events, _STOPPED))
        jaws.answer_questions_with(None)
        jaws.connect()                                      # at the terminal, as a CLI program asks
        self.assertEqual(1, _pulses(events))

    def test_a_solenoid_that_asks_keeps_the_same_gate(self) -> None:
        events: list[Any] = []
        jaws = JawIOGripper(_IO(events), close_output_pin=0, confirm_open_at_start=True, sleep=lambda _s: None)
        jaws.answer_questions_with(Browser("closed", "open_now"), before_change=_Gate(events, _STOPPED))
        with self.assertRaises(RobotError):
            jaws.connect()
        self.assertEqual(["gate"], events, "the solenoid was driven past a refusing gate")


class AtAPickStartTests(unittest.TestCase):
    def test_the_installed_gate_reaches_a_pick_starts_question(self) -> None:
        events: list[Any] = []
        jaws = _closed_on_a_part(events)
        jaws.answer_questions_with(Browser("closed", "open_now"), before_change=_Gate(events, _STOPPED))

        refused = jaws.jaws_open_for_a_pick()

        self.assertIn("not starting the pick", refused)
        self.assertIn(_STOPPED, refused)
        self.assertEqual(0, _pulses(events))


class APersonsClosedIsTheCountTests(unittest.TestCase):
    """⛔ Found writing the check: a person said the jaws stand CLOSED and then aborted, and the count went on saying
    OPEN, so the next close would have been a change that opens them. A person's word is the count from then on."""

    def test_closed_and_abort_at_the_check_counts_them_closed(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("open"))
        jaws.connect()
        self.assertFalse(jaws.jaws_closed)
        jaws.answer_questions_with(Browser("closed", "abort"))

        self.assertTrue(jaws.confirm_where_the_jaws_stand(_CHECK))

        self.assertTrue(jaws.jaws_closed)
        jaws.set_closed(False)                              # the open the person did not choose then is one change now
        self.assertEqual(1, _pulses(events))
        self.assertFalse(jaws.jaws_closed)

    def test_a_count_nobody_vouched_for_is_vouched_for_by_closed(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("open"))
        jaws.connect()
        jaws._io.do[0] = True                               # type: ignore[attr-defined] # switched at the pendant
        jaws.answer_questions_with(Browser("closed", "abort"))

        self.assertTrue(jaws.jaws_open_for_a_pick())       # refused: the person chose not to open them

        self.assertFalse(jaws.edge_unknown)
        self.assertEqual("", jaws.why_jaws_unknown(), "the count starts from the output a person answered about")
        self.assertTrue(jaws.jaws_closed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
