"""The jaws question of a toggle hand reaches whoever the console names: a structured seam, asked stage by stage.

The owner's hand is a Hand-E on tool DO0, a ``single_toggle`` with no sensor: the program counts its own changes from
where a person said the jaws stood (2026-09-24, 2026-09-28). The console cannot ask at the server's terminal: a blocking
``input()`` cannot be cancelled, and nobody watching the browser sees it (build plan, item 10). So the hand takes a
structured seam (``JawIOGripper.answer_questions_with``) and hands it one :class:`~src.robot.grippers.jaw_io.JawAsking`
per attempt. What is held here, on the real driver over a recording tool I/O:

* the seam gets the stage (``where``, then ``open_now`` after "closed"), the choices that stage takes, which attempt it
  is and why it is asked again, the reason and the bank and pin; the hand acts only on a choice it offered, or on the
  build plan's own letter for one (``p`` for ``open_now``, ``a`` for ``abort``): the terminal's ``o``, ``c`` and
  ``pulse`` are none of the choices there;
* ``EOFError`` from the seam (a timeout, a cancel, a Disconnect) is no answer and never "open": the connect is refused
  with nothing sent. An empty answer is no answer either, at connect too: the browser offers no default, and the
  question's own text says so;
* the seam comes before the string seam and the terminal, and taking it away gives them back, so CLI programs ask as
  they always did;
* an answer that comes back to a connection that changed while it waited sends nothing, through this seam as through
  the terminal;
* the hand names the bank and pin it drives (``where_pin``), read without its lock: the console reads it on every poll.
"""

from __future__ import annotations

import dataclasses
import io
import threading
import unittest
from typing import Any
from unittest.mock import patch

from src.robot.core import RobotError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.grippers.jaw_io import JawAsking, JawIOGripper
from tests.test_a_toggle_asks_where_its_jaws_stand import _IO, Person, _pulses, _toggle


class Browser:
    """The console's seam: answers each question from a script, and keeps every question it was asked.

    ``"EOF"`` in the script raises ``EOFError``, as the console's seam does on a timeout, a cancel or a Disconnect.
    ``meanwhile`` runs inside the next question, once, before its answer comes back.
    """

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[JawAsking] = []
        self.meanwhile: Any = None

    def __call__(self, asking: JawAsking) -> str:
        self.asked.append(asking)
        act, self.meanwhile = self.meanwhile, None
        if act is not None:
            act()
        if not self.answers:
            raise AssertionError(f"asked once more than scripted: {asking!r}")
        answer = self.answers.pop(0)
        if answer == "EOF":
            raise EOFError("nobody answered within 120 s")
        return answer


def _nobody_at_the_terminal(*_a: Any, **_k: Any) -> str:
    raise AssertionError("the terminal was asked while the console's seam was installed")


class TheSeamGetsTheQuestionStageByStageTests(unittest.TestCase):
    def test_the_first_question_names_its_stage_its_choices_and_its_attempt(self) -> None:
        events: list[Any] = []
        browser = Browser("open")
        jaws = _toggle(events, Person())                    # the string seam is never asked: no answer scripted
        jaws.answer_questions_with(browser)

        jaws.connect()

        self.assertTrue(jaws.is_connected)
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))
        (asked,) = browser.asked
        self.assertEqual("where", asked.stage)
        self.assertTrue(asked.at_connect)
        self.assertEqual(("open", "closed"), asked.choices)
        self.assertEqual(1, asked.attempt)
        self.assertEqual("", asked.why_again)
        self.assertEqual("tool output 0", asked.where)
        self.assertIn("every change of its output moves them", asked.reason)
        self.assertIn("Do they stand OPEN?", asked.text)
        self.assertIn("tool output 0", asked.text)

    def test_closed_then_open_now_opens_them_with_one_change(self) -> None:
        events: list[Any] = []
        browser = Browser("closed", "open_now")
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(browser)

        jaws.connect()

        self.assertTrue(jaws.is_connected)
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(1, _pulses(events), "one change of DO0 opens them, and nothing else is sent")
        first, second = browser.asked
        self.assertEqual(("where", "open_now"), (first.stage, second.stage))
        self.assertEqual(("open_now", "abort"), second.choices)
        self.assertTrue(second.at_connect)
        self.assertEqual(1, second.attempt)
        self.assertIn("one change of tool output 0", second.text)

    def test_the_plans_word_for_open_now_opens_them_too(self) -> None:
        """The console may hand back the driver's own word, ``p``, for "Jetzt öffnen" (build plan 1.6)."""
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("closed", "p"))
        jaws.connect()
        self.assertEqual(1, _pulses(events))
        self.assertFalse(jaws.jaws_closed)

    def test_closed_then_abort_refuses_the_connect_with_nothing_sent(self) -> None:
        for abort in ("abort", "a"):                        # "a" is the build plan's letter for abort
            with self.subTest(abort=abort):
                events: list[Any] = []
                jaws = _toggle(events, Person())
                jaws.answer_questions_with(Browser("closed", abort))
                with self.assertRaises(RobotError) as caught:
                    jaws.connect()
                self.assertIn("chose not to open them", str(caught.exception))
                self.assertFalse(jaws.is_connected)
                self.assertEqual(0, _pulses(events))

    def test_the_terminals_letters_are_none_of_the_choices_through_the_seam(self) -> None:
        """⛔ The browser offers ``open``/``closed`` and then ``open_now``/``abort``; a seam that hands back a word it was
        never offered (``o``, ``c``, ``pulse``) is asked again, never acted on (review of 2026-10-01)."""
        for script, why in ((("o", "o", "o"), "no answer said where"),
                            (("c", "c", "c"), "no answer said where"),
                            (("closed", "pulse", "pulse", "pulse"), "gave no clear answer")):
            with self.subTest(script=script):
                events: list[Any] = []
                browser = Browser(*script)
                jaws = _toggle(events, Person())
                jaws.answer_questions_with(browser)
                with self.assertRaises(RobotError) as caught:
                    jaws.connect()
                self.assertIn(why, str(caught.exception))
                self.assertIn(f"'{script[-1]}' is none of the choices.", browser.asked[-1].why_again)
                self.assertEqual(0, _pulses(events))
                self.assertFalse(jaws.is_connected)

    def test_the_connect_question_through_the_seam_offers_no_default(self) -> None:
        """The terminal's connect question offers Enter for open; through the seam an empty answer is no answer, so the
        question the browser shows (and the tech view prints) must not say otherwise."""
        browser = Browser("open")
        jaws = _toggle([], Person())
        jaws.answer_questions_with(browser)
        jaws.connect()
        (asked,) = browser.asked
        self.assertIn("Enter alone is no answer here", asked.text)
        self.assertNotIn("Enter or 'open'", asked.text)
        person = Person("")                                 # ⭐ THE CONTROL: the terminal keeps its default
        _toggle([], person).connect()
        self.assertIn("Enter or 'open' = open", person.asked[0])

    def test_an_answer_that_is_none_of_the_choices_is_asked_again_saying_why(self) -> None:
        events: list[Any] = []
        browser = Browser("maybe", "open_now", "?")         # open_now is no answer to "where do they stand"
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(browser)

        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        self.assertIn("no answer said where the jaws on tool output 0 stand", str(caught.exception))
        self.assertEqual([1, 2, 3], [asked.attempt for asked in browser.asked])
        self.assertEqual({"where"}, {asked.stage for asked in browser.asked})
        self.assertEqual("'maybe' is none of the choices.", browser.asked[1].why_again)
        self.assertEqual("'open_now' is none of the choices.", browser.asked[2].why_again)
        self.assertIn("'maybe' is none of the choices.", browser.asked[1].text)
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))


class NoAnswerIsNeverOpenTests(unittest.TestCase):
    """⛔ A question nobody answered is refused, never taken for "open" (build plan, item 10): a count started OPEN on
    jaws that are closed inverts every command after it."""

    def test_a_seam_that_raises_eoferror_refuses_the_connect_with_nothing_sent(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("EOF"))
        with self.assertRaises(RobotError) as caught:
            jaws.connect()
        self.assertIn("no answer said where the jaws on tool output 0 stand", str(caught.exception))
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))

    def test_no_answer_to_open_now_sends_nothing(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(Browser("closed", "EOF"))
        with self.assertRaises(RobotError) as caught:
            jaws.connect()
        self.assertIn("gave no clear answer to whether to open them", str(caught.exception))
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))

    def test_an_empty_answer_is_no_answer_even_at_connect(self) -> None:
        """The terminal takes Enter for open at connect, the owner's agreed answer there; the browser offers no
        default, so an empty answer through the seam is asked again, and three of them are no answer."""
        events: list[Any] = []
        browser = Browser("", "", "")
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(browser)

        with self.assertRaises(RobotError):
            jaws.connect()

        self.assertEqual(3, len(browser.asked))
        self.assertEqual("An empty line is no answer here.", browser.asked[1].why_again)
        self.assertNotIn("", browser.asked[0].choices)
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))

    def test_enter_at_the_terminal_still_means_open_at_connect(self) -> None:
        """⭐ THE CONTROL: without the seam, a CLI program asks at its terminal as it always did."""
        person = Person("")
        jaws = _toggle([], person)
        jaws.connect()
        self.assertEqual(1, len(person.asked))
        self.assertFalse(jaws.jaws_closed)


class TheSeamComesFirstTests(unittest.TestCase):
    def test_the_seam_is_asked_and_neither_the_string_seam_nor_the_terminal(self) -> None:
        class _Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        browser = Browser("open")
        person = Person()                                   # asked once, it fails the test: nothing scripted
        jaws = _toggle([], person)
        jaws.answer_questions_with(browser)
        with patch("sys.stdin", _Terminal("")), patch("builtins.input", _nobody_at_the_terminal):
            jaws.connect()
        self.assertEqual(1, len(browser.asked))
        self.assertEqual([], person.asked)

    def test_installing_says_what_was_installed_before_and_none_takes_it_away(self) -> None:
        person = Person("")
        jaws = _toggle([], person)
        self.assertFalse(jaws.question_seam_installed())
        first, second = Browser(), Browser()
        self.assertIsNone(jaws.answer_questions_with(first))
        self.assertTrue(jaws.question_seam_installed())
        self.assertIs(first, jaws.answer_questions_with(second))
        self.assertIs(second, jaws.answer_questions_with(None))
        self.assertFalse(jaws.question_seam_installed())

        jaws.connect()                                      # the string seam it was built with asks again
        self.assertEqual(1, len(person.asked))
        self.assertEqual([], first.asked + second.asked)

    def test_a_solenoid_asks_through_the_seam_where_it_opted_in(self) -> None:
        events: list[Any] = []
        browser = Browser("closed", "open_now")
        jaws = JawIOGripper(_IO(events), close_output_pin=0, confirm_open_at_start=True, sleep=lambda _s: None)
        jaws.answer_questions_with(browser)
        jaws.connect()
        self.assertEqual(["where", "open_now"], [asked.stage for asked in browser.asked])
        self.assertIn("tool output 0 set low", browser.asked[1].text)
        self.assertEqual(("DO", 0, False), events[-1])


class AStaleAnswerSendsNothingTests(unittest.TestCase):
    """An answer is acted on only where the connection it was asked on still stands when it comes back."""

    def test_a_disconnect_while_the_seam_waits_refuses_the_connect_with_nothing_sent(self) -> None:
        events: list[Any] = []
        browser = Browser("closed", "open_now")
        jaws = _toggle(events, Person())
        jaws.answer_questions_with(browser)
        browser.meanwhile = jaws.disconnect

        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        self.assertIn("while the question waited", str(caught.exception))
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))
        self.assertEqual(["open_now"], browser.answers, "open now was offered on a connection that had changed")

    def test_a_disconnect_while_open_now_waits_sends_nothing(self) -> None:
        """⛔ The second question waits too: "Jetzt öffnen" that comes back to a connection taken down meanwhile is a
        stale answer, and the one change it asked for is never sent."""
        events: list[Any] = []
        jaws = _toggle(events, Person())
        stages: list[str] = []

        def browser(asking: JawAsking) -> str:
            stages.append(asking.stage)
            if asking.stage == "open_now":
                jaws.disconnect()                           # the console came down while the person decided
                return "open_now"
            return "closed"

        jaws.answer_questions_with(browser)
        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        self.assertEqual(["where", "open_now"], stages)
        self.assertIn("while the question waited", str(caught.exception))
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)


class TheHandNamesItsPinTests(unittest.TestCase):
    def test_the_bank_and_the_pin_as_the_pendant_shows_them(self) -> None:
        self.assertEqual("tool output 0", _toggle([], Person()).where_pin)
        standard = JawIOGripper(_IO([]), close_output_pin=3, io_port=DigitalIOPort.STANDARD, sleep=lambda _s: None)
        self.assertEqual("standard output 3", standard.where_pin)

    def test_it_is_read_while_another_thread_holds_the_hand(self) -> None:
        """The console reads it on every ``GET /v1/cell``, also while a question holds the hand's lock for up to 120 s:
        it must never wait on that lock."""
        jaws = _toggle([], Person())
        held, release = threading.Event(), threading.Event()

        def hold() -> None:
            with jaws._lock:
                held.set()
                release.wait(timeout=10)

        holder = threading.Thread(target=hold, daemon=True)
        holder.start()
        self.addCleanup(release.set)
        self.assertTrue(held.wait(timeout=5))
        read: list[str] = []
        reader = threading.Thread(target=lambda: read.append(jaws.where_pin), daemon=True)
        reader.start()
        reader.join(timeout=2)
        self.assertEqual(["tool output 0"], read, "where_pin waited on the hand's lock")


class TheQuestionCarriesWhatTheConsoleReadsTests(unittest.TestCase):
    def test_every_field_the_browser_seam_reads_is_carried(self) -> None:
        from api.jaws import JawAskingLike

        read = {name for name, member in vars(JawAskingLike).items() if isinstance(member, property)}
        carried = {field.name for field in dataclasses.fields(JawAsking)}
        self.assertLessEqual(read, carried)

    def test_a_question_cannot_be_changed_once_asked(self) -> None:
        asking = JawAsking(stage="where", at_connect=True, reason="r", where="tool output 0", choices=("open", "closed"),
                           text="t", attempt=1, why_again="")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            asking.stage = "open_now"  # type: ignore[misc]


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
