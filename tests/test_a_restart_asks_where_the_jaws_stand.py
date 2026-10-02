"""A Restart asks where the jaws stand, even where the program's count says open (owner decision 9, build plan 1.6).

After a stop a person may have held the part and opened the jaws, or emptied the hand, or not: the count of a toggle
with no sensor (the owner's Hand-E on tool DO0) says only what the program commanded. So the check before a Restart,
before Home after a stop and from Setup asks a person every time, through
``JawIOGripper.confirm_where_the_jaws_stand``, and never assumes the jaws open. What is held here:

* it asks even on a known OPEN count, at the check's own wording: only the word ``open`` or ``closed`` answers it,
  an empty answer is no answer;
* where the count says CLOSED and nothing has moved the jaws since, it asks only "open now" or abort: an answer of
  open there would contradict the program's own evidence
  (``tests/test_a_count_that_says_closed_is_never_answered_open.py``);
* it answers ``""`` where the jaws stand open by the end and the one-line refusal otherwise; "closed" and
  "open now" open them with exactly one change; no answer is a refusal, never open, and leaves the count as it was;
* the caller learns whether the one change went out from ``commands_sent`` before and after, so it can tell the
  planner the hand is empty (``arm.detach_payload``) and count down before the next motion;
* an output switched at the pendant is cleared by a person's answer, and the count starts from the output read then;
  one switched at the pendant WHILE the person decides on "open now" refuses that change with nothing sent
  (``RobotError``), and the count is nobody's to vouch for until a person is asked again;
* a hand that is not a toggle answers ``""`` at once; a hand that is not connected is refused; a thread on which nobody
  may be asked is refused without a question.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.robot.core import RobotConnectionError, RobotError
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_a_toggle_asks_where_its_jaws_stand import _IO, Person, _pulses, _toggle
from tests.test_the_jaws_question_reaches_whoever_the_console_names import Browser

_RESTART = "a Restart after a stop: a person may have opened them or emptied the hand"


def _connected(events: list[Any], browser: Browser, *, closed: bool = False) -> JawIOGripper:
    """The owner's toggle, connected open through the console's seam (``closed`` closes it on a part after), the log
    cleared, the seam scripted from here with ``browser``'s answers."""
    connect = Browser("open")
    jaws = _toggle(events, Person())
    jaws.answer_questions_with(connect)
    jaws.connect()
    if closed:
        jaws.set_closed(True)
    events.clear()
    jaws.answer_questions_with(browser)
    return jaws


class ARestartAlwaysAsksTests(unittest.TestCase):
    def test_a_count_that_says_open_is_asked_all_the_same(self) -> None:
        events: list[Any] = []
        browser = Browser("open")
        jaws = _connected(events, browser)
        self.assertFalse(jaws.jaws_closed, "the count says open before the check")

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))

        (asked,) = browser.asked
        self.assertEqual("where", asked.stage)
        self.assertFalse(asked.at_connect)
        self.assertEqual(("open", "closed"), asked.choices)
        self.assertIn(_RESTART, asked.reason)
        self.assertIn("type 'open' or 'closed'", asked.text)
        self.assertEqual(0, _pulses(events))

    def test_the_check_takes_the_word_and_no_empty_answer(self) -> None:
        events: list[Any] = []
        browser = Browser("", "open")
        jaws = _connected(events, browser)
        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))
        self.assertEqual(2, len(browser.asked))
        self.assertEqual("An empty line is no answer here.", browser.asked[1].why_again)

    def test_open_now_opens_them_with_one_change_and_the_caller_sees_it_went_out(self) -> None:
        events: list[Any] = []
        browser = Browser("open_now")
        jaws = _connected(events, browser, closed=True)
        before = jaws.commands_sent

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))

        self.assertEqual(["open_now"], [asked.stage for asked in browser.asked], "the count says closed: no 'where'")
        self.assertEqual(1, _pulses(events))
        self.assertEqual(1, jaws.commands_sent - before, "the caller cannot tell the jaws were opened at the question")
        self.assertFalse(jaws.jaws_closed)
        self.assertTrue(jaws.is_connected)

    def test_an_open_answer_sends_nothing_and_the_caller_sees_nothing_went_out(self) -> None:
        """⭐ THE CONTROL: the jaws were open already, so no hand was at them and no countdown is owed for it. Asked
        where nothing the program knows says otherwise: the output was switched at the pendant since it closed them."""
        events: list[Any] = []
        jaws = _connected(events, Browser("open"), closed=True)
        jaws._io.do[0] = not jaws._io.do.get(0, False)     # type: ignore[attr-defined] # opened at the pendant
        before = jaws.commands_sent
        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))
        self.assertEqual(0, jaws.commands_sent - before)
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.jaws_closed, "a person said open: the count starts open from here")

    def test_closed_and_abort_is_the_refusal_with_nothing_sent(self) -> None:
        events: list[Any] = []
        jaws = _connected(events, Browser("closed", "abort"))

        refused = jaws.confirm_where_the_jaws_stand(_RESTART)

        self.assertIn("stand CLOSED and chose not to open them", refused)
        self.assertNotIn("\n", refused)
        self.assertTrue(jaws.jaws_closed, "a person said they stand closed: the count says so")
        self.assertTrue(jaws.is_connected)
        self.assertEqual(0, _pulses(events))

    def test_no_answer_is_a_refusal_never_open_and_leaves_the_count_as_it_was(self) -> None:
        for script in (("EOF",), ("closed", "EOF"), ("x", "y", "z")):
            with self.subTest(script=script):
                events: list[Any] = []
                jaws = _connected(events, Browser(*script), closed=True)

                refused = jaws.confirm_where_the_jaws_stand(_RESTART)

                self.assertTrue(refused)
                self.assertNotIn("chose", refused)
                self.assertTrue(jaws.jaws_closed, "no answer opened nothing, and is never taken for open")
                self.assertEqual(0, _pulses(events))

    def test_an_output_switched_at_the_pendant_is_cleared_by_a_persons_answer(self) -> None:
        events: list[Any] = []
        jaws = _connected(events, Browser("open"))
        jaws._io.do[0] = True                               # type: ignore[attr-defined] # somebody at the pendant
        self.assertTrue(jaws.why_jaws_unknown())

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))

        self.assertEqual("", jaws.why_jaws_unknown())
        self.assertFalse(jaws.edge_unknown)
        jaws.set_closed(True)                               # the count starts from the output read at the answer
        self.assertEqual(1, _pulses(events))
        self.assertFalse(jaws._io.do[0])                    # type: ignore[attr-defined] # one change: HIGH to LOW


class APendantSwitchWhileOpenNowWaitsTests(unittest.TestCase):
    """The owner opens the Hand-E at the pendant by hand. Switched there while the person decides on "Jetzt öffnen",
    the output no longer stands where the person's "closed" started the count, so the one change would move the jaws
    the other way: it is refused with nothing sent, at a check and at a connect, and the console maps the raise to
    ``jaws_not_open`` (review of 2026-10-01: the behaviour the contract note relies on, pinned)."""

    @staticmethod
    def _switched_meanwhile(jaws: JawIOGripper) -> Any:
        def seam(asking: Any) -> str:
            if asking.stage == "where":
                return "closed"
            jaws._io.do[0] = not jaws._io.do.get(0, False)  # type: ignore[attr-defined] # at the pendant, meanwhile
            return "open_now"

        return seam

    def test_at_the_check_the_change_is_refused_with_nothing_sent(self) -> None:
        events: list[Any] = []
        jaws = _connected(events, Browser(), closed=True)
        jaws.answer_questions_with(self._switched_meanwhile(jaws))

        with self.assertRaises(RobotError) as caught:
            jaws.confirm_where_the_jaws_stand(_RESTART)

        self.assertIn("switched it", str(caught.exception))
        self.assertIn("Nothing was sent", str(caught.exception))
        self.assertEqual(0, _pulses(events))
        self.assertTrue(jaws.edge_unknown, "the next question has to vouch for the count again")
        self.assertTrue(jaws.is_connected)

    def test_at_connect_the_change_is_refused_and_the_connect_with_it(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(self._switched_meanwhile(jaws))

        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        self.assertIn("switched it", str(caught.exception))
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)


class WhereNobodyIsAskedTests(unittest.TestCase):
    def test_a_hand_that_is_not_a_toggle_answers_at_once(self) -> None:
        jaws = JawIOGripper(_IO([]), close_output_pin=0, sleep=lambda _s: None)
        jaws.connect()
        browser = Browser()
        jaws.answer_questions_with(browser)
        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))
        self.assertEqual([], browser.asked)

    def test_a_hand_that_is_not_connected_is_refused_with_nothing_asked(self) -> None:
        browser = Browser()
        jaws = _toggle([], Person())
        jaws.answer_questions_with(browser)
        with self.assertRaises(RobotConnectionError):
            jaws.confirm_where_the_jaws_stand(_RESTART)
        self.assertEqual([], browser.asked)

    def test_a_thread_on_which_nobody_may_be_asked_is_refused_without_a_question(self) -> None:
        events: list[Any] = []
        browser = Browser()
        jaws = _connected(events, browser)
        with jaws.asking_nobody("a run's thread never waits on a person"):
            refused = jaws.confirm_where_the_jaws_stand(_RESTART)
        self.assertIn("a run's thread never waits on a person", refused)
        self.assertEqual([], browser.asked)
        self.assertEqual(0, _pulses(events))

    def test_a_check_with_nobody_to_ask_is_refused(self) -> None:
        """No seam, no string seam and no terminal: the check is refused, never taken for open."""
        from unittest.mock import patch
        import io

        events: list[Any] = []
        jaws = _connected(events, Browser())
        jaws.answer_questions_with(None)
        jaws._ask = None
        with patch("sys.stdin", io.StringIO("")):
            refused = jaws.confirm_where_the_jaws_stand(_RESTART)
        self.assertIn("nobody can be asked", refused)
        self.assertEqual(0, _pulses(events))


class TheCheckKeepsTheConnectsRulesTests(unittest.TestCase):
    def test_a_disconnect_while_the_check_waits_sends_nothing(self) -> None:
        events: list[Any] = []
        browser = Browser("open_now")
        jaws = _connected(events, browser, closed=True)
        browser.meanwhile = jaws.disconnect

        refused = jaws.confirm_where_the_jaws_stand(_RESTART)

        self.assertIn("while the question waited", refused)
        self.assertEqual(0, _pulses(events))

    def test_a_check_after_a_failed_change_is_how_the_count_is_vouched_for_again(self) -> None:
        events: list[Any] = []
        jaws = _connected(events, Browser("open"))
        jaws._io.refuse_high = True                         # type: ignore[attr-defined] # the change fails
        with self.assertRaises(RobotError):
            jaws.set_closed(True)
        jaws._io.refuse_high = False                        # type: ignore[attr-defined]
        self.assertTrue(jaws.edge_unknown)

        self.assertEqual("", jaws.confirm_where_the_jaws_stand(_RESTART))

        self.assertFalse(jaws.edge_unknown)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
