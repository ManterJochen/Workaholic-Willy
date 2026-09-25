"""A single toggle asks a person where its jaws stand, counts its own pulses from the answer, and never pulses before
the arm moves at the start of a pick (owner's decision, 2026-09-24).

The owner's hand is a Hand-E on the Robotiq I/O Coupling: ``jaw_io`` single_toggle on tool DO0, each short pulse flips
the jaws, 24 V, nothing wired back. A persistent record of the pulses (``logs/robot/state``) was tried on 2026-09-23 and
dropped: a pulse nobody counted still inverted it, and a person had to declare the jaws on the bench to recover. Now:

* the gripper's connect asks, once per program start and before anything moves, whether the jaws stand open; closed is
  answered with one pulse to open them or an abort, and with no terminal the connect is refused;
* every pulse flips the count, whatever verb sent it; a pick sends none before the arm moves, exactly one at the part,
  and a place one where the count says closed and none where it says open ("already open");
* a pick that starts on jaws the program believes closed asks again instead of pulsing, and with nobody to ask it is
  refused before any motion;
* nothing measures a toggle, so a pick counts as grasped and its report says the hold was not checked (no sensor);
* a driver that raises in the pick loop's policy becomes a report outcome, never an escaping exception.

Every pulse is read off the recorded rising edges on DO0, on one log the arm writes its motions to.
"""

from __future__ import annotations

import io
import sys
import unittest
from typing import Any, Callable
from unittest.mock import patch

import numpy as np

from src.geometry import Pose
from src.robot.core import RobotConnectionError, RobotError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.execution import handling
from src.robot.execution.robot import Robot
from src.robot.grasping.motion.execution_policy import GraspExecutionPolicy, PolicyOutcome
from src.robot.grippers import jaw_io
from src.robot.grippers.jaw_io import JawIOGripper
from tests.test_a_stopped_controller_moves_no_jaws import _Arm, _grasp
from tests.test_robot_pick_and_place import _Log, _pose, _RecordingArm

_BENCH = "unit double: a bench with no cameras"
_NO_TERMINAL = io.StringIO("")  # isatty() is False: nobody to ask


class _IO:
    """The tool I/O the toggle pulses, writing every output onto the log the arm shares."""

    def __init__(self, events: list[Any], *, refuse_high: bool = False) -> None:
        self.events = events
        self.do: dict[int, bool] = {}
        self.refuse_high = refuse_high
        self.refuse_all = False

    def set_digital_output(self, pin: int, value: bool, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> None:
        if self.refuse_all or (value and self.refuse_high):
            raise RobotError("the controller refused the write on tool output 0")
        self.do[pin] = bool(value)
        self.events.append(("DO", pin, bool(value)))

    def get_digital_output(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return self.do.get(pin, False)

    def get_digital_input(self, pin: int, *, port: DigitalIOPort = DigitalIOPort.STANDARD) -> bool:
        return False

    def set_analog_output(self, pin: int, value: float, *, current: bool = False) -> None:
        return None


class Person:
    """The person at the terminal: answers from a script, and keeps every question asked."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        if not self.answers:
            raise AssertionError(f"asked once more than scripted: {question!r}")
        return self.answers.pop(0)


def _toggle(events: list[Any], ask: Any = None, **kw: Any) -> JawIOGripper:
    """The owner's hand: single_toggle on tool DO0, no feedback, 49.99 / 5.0 mm, a 2 s stroke."""
    return JawIOGripper(_IO(events, refuse_high=kw.pop("refuse_high", False)), actuation="single_toggle",
                        close_output_pin=0, pulse_s=0.0, close_settle_s=2.0, min_width_mm=5.0, max_width_mm=49.99,
                        ask=ask, sleep=kw.pop("sleep", lambda _s: None), **kw)


def _pulses(events: list[Any]) -> int:
    """Rising edges on DO0: each one flips the jaws."""
    return sum(1 for e in events if e == ("DO", 0, True))


def _moves(events: list[Any]) -> list[int]:
    """Where on the log each motion is."""
    return [i for i, e in enumerate(events) if isinstance(e, tuple) and e[0] == "move"]


def _edges(events: list[Any]) -> list[int]:
    return [i for i, e in enumerate(events) if e == ("DO", 0, True)]


def _robot(events: list[Any], jaws: JawIOGripper) -> Robot:
    log = _Log()
    log.entries = events
    arm = _RecordingArm(log)
    arm.connect()
    jaws.connect()
    return Robot.from_parts(arm=arm, gripper=jaws, lock_key=None)


# ---------------------------------------------------------------------------------------------------
# The question at connect
# ---------------------------------------------------------------------------------------------------


class TheStartQuestionTests(unittest.TestCase):
    def test_open_is_no_pulse_and_the_count_starts_open(self) -> None:
        for answer in ("", "open", "  OPEN "):
            with self.subTest(answer=answer):
                events: list[Any] = []
                person = Person(answer)
                jaws = _toggle(events, person)
                jaws.connect()
                self.assertTrue(jaws.is_connected)
                self.assertFalse(jaws.jaws_closed)
                self.assertEqual(0, _pulses(events))
                self.assertEqual(1, len(person.asked))

    def test_the_question_names_the_bank_and_the_pin(self) -> None:
        person = Person("")
        _toggle([], person).connect()
        self.assertIn("tool output 0", person.asked[0])
        self.assertIn("Do they stand OPEN?", person.asked[0])

    def test_closed_and_a_pulse_opens_them_with_one_pulse_and_waits_the_stroke(self) -> None:
        events: list[Any] = []
        slept: list[float] = []
        person = Person("closed", "p")
        jaws = _toggle(events, person, sleep=slept.append)
        jaws.connect()
        self.assertTrue(jaws.is_connected)
        self.assertEqual(1, _pulses(events))
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(2.0, slept[-1], "the stroke was not waited out after the pulse")
        self.assertIn("one pulse on tool output 0", person.asked[1])

    def test_closed_and_abort_refuses_the_connect_with_nothing_pulsed(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person("closed", "a"))
        with self.assertRaises(RobotError) as caught:
            jaws.connect()
        self.assertIn("CLOSED", str(caught.exception))
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))

    def test_no_terminal_and_no_question_refuses_the_connect_in_one_line(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events)
        with patch("sys.stdin", _NO_TERMINAL), self.assertRaises(RobotError) as caught:
            jaws.connect()
        message = str(caught.exception)
        self.assertNotIn("\n", message)
        for part in ("not connecting", "tool output 0", "terminal"):
            self.assertIn(part, message)
        self.assertFalse(jaws.is_connected)
        self.assertEqual([], events)

    def test_a_terminal_is_asked_through_input_by_default(self) -> None:
        class _Terminal(io.StringIO):
            def isatty(self) -> bool:
                return True

        jaws = _toggle([])
        with patch("sys.stdin", _Terminal("")), patch("builtins.input", return_value="") as typed:
            jaws.connect()
        self.assertTrue(jaws.is_connected)
        typed.assert_called_once()

    def test_a_connect_that_raised_leaves_nothing_connected_and_the_next_one_asks_again(self) -> None:
        events: list[Any] = []
        person = Person("closed", "p", "")
        jaws = _toggle(events, person)
        jaws._io.refuse_all = True                          # type: ignore[attr-defined] # the pulse the person chose fails
        with self.assertRaises(RobotError):
            jaws.connect()
        self.assertFalse(jaws.is_connected)
        jaws._io.refuse_all = False                         # type: ignore[attr-defined]
        jaws.connect()
        self.assertTrue(jaws.is_connected)
        self.assertEqual(3, len(person.asked), "the second connect did not ask again")
        jaws.connect()                                      # connected: nothing is asked twice
        self.assertEqual(3, len(person.asked))

    def test_answers_that_are_none_of_the_choices_are_an_abort(self) -> None:
        events: list[Any] = []
        person = Person("maybe", "dunno", "?")
        jaws = _toggle(events, person)
        with self.assertRaises(RobotError):
            jaws.connect()
        self.assertEqual(3, len(person.asked))
        self.assertIn("'dunno' is none of the choices", person.asked[2])
        self.assertEqual(0, _pulses(events))

    def test_a_refused_connect_rolls_the_arm_back(self) -> None:
        from src.robot.execution.lifecycle import connect_cell

        class _Cell:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def connect(self) -> None:
                self.calls.append("connect")

            def disconnect(self) -> None:
                self.calls.append("disconnect")

        arm = _Cell()
        with self.assertRaises(RobotError):
            connect_cell(arm, _toggle([], Person("closed", "a")))
        self.assertEqual(["connect", "disconnect"], arm.calls)


class TheQuestionIsOptInForASolenoidTests(unittest.TestCase):
    def _solenoid(self, events: list[Any], **kw: Any) -> JawIOGripper:
        return JawIOGripper(_IO(events), close_output_pin=0, sleep=lambda _s: None, **kw)

    def test_a_solenoid_asks_nothing_by_default(self) -> None:
        jaws = self._solenoid([], ask=Person())             # a question would raise: none scripted
        jaws.connect()
        self.assertTrue(jaws.is_connected)

    def test_a_solenoid_that_opted_in_asks_and_opens_by_command(self) -> None:
        events: list[Any] = []
        person = Person("closed", "p")
        jaws = self._solenoid(events, ask=person, confirm_open_at_start=True)
        jaws.connect()
        self.assertEqual(2, len(person.asked))
        self.assertIn("tool output 0 set low", person.asked[1])
        self.assertEqual(("DO", 0, False), events[-1])
        self.assertFalse(jaws.jaws_closed)

    def test_the_config_flag_reaches_the_built_gripper(self) -> None:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.robot_parts import build_gripper

        class _UR(_IO):
            def connect(self) -> None:
                return None

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "gripper": {"vendor": "jaw_io", "min_width_mm": 5.0, "max_width_mm": 49.99,
                                        "jaw_io": {"close_output_pin": 0, "confirm_open_at_start": True}},
        })
        jaws = build_gripper(cfg, arm=_UR([]))  # type: ignore[arg-type]
        with patch("sys.stdin", _NO_TERMINAL), self.assertRaises(RobotError):
            jaws.connect()


# ---------------------------------------------------------------------------------------------------
# A stray Enter answers nothing
# ---------------------------------------------------------------------------------------------------


class _Console:
    """A stdin that is the real console: it has a descriptor and it is a terminal."""

    def fileno(self) -> int:
        return 0

    def isatty(self) -> bool:
        return True


class _Keyboard:
    """``msvcrt`` on Windows, holding what was typed before anybody asked."""

    def __init__(self, typed: str) -> None:
        self.typed = list(typed)

    def kbhit(self) -> bool:
        return bool(self.typed)

    def getwch(self) -> str:
        return self.typed.pop(0)


class _Termios:
    """``termios`` on POSIX: records the queue each flush discarded."""

    TCIFLUSH = 0

    def __init__(self) -> None:
        self.flushed: list[tuple[int, int]] = []

    def tcflush(self, fd: int, queue: int) -> None:
        self.flushed.append((fd, queue))


class TheTypeaheadIsDrainedBeforeTheQuestionTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D1). The question was read with the builtin ``input`` straight off the console, so an
    Enter typed while the models loaded (examples 14 and 15 use Enter as push-to-talk) answered it unseen: with the jaws
    really closed the count started OPEN and every command after it ran inverted, reproduced on a real Windows
    console. What was typed before a question is discarded before it is asked, wherever the terminal is asked."""

    def test_keys_typed_before_the_question_are_discarded_on_windows(self) -> None:
        keyboard = _Keyboard("\r\r")
        with patch.dict(sys.modules, {"msvcrt": keyboard}):
            self.assertEqual(2, jaw_io._drain_typeahead(_Console()))
        self.assertEqual([], keyboard.typed, "a stray Enter is still waiting to answer the question")

    def test_the_input_queue_is_flushed_on_posix(self) -> None:
        termios = _Termios()
        with patch.dict(sys.modules, {"msvcrt": None, "termios": termios}):
            jaw_io._drain_typeahead(_Console())
        self.assertEqual([(0, _Termios.TCIFLUSH)], termios.flushed)

    def test_a_stdin_that_is_not_the_console_is_left_alone(self) -> None:
        """⭐ THE CONTROL: a pipe or a replaced stdin holds answers somebody meant, and has no console to drain."""
        keyboard = _Keyboard("\r")
        with patch.dict(sys.modules, {"msvcrt": keyboard}):
            for stdin in (io.StringIO(""), _Terminal("")):   # no descriptor, whatever isatty() says
                with self.subTest(stdin=type(stdin).__name__):
                    self.assertEqual(0, jaw_io._drain_typeahead(stdin))
        self.assertEqual(["\r"], keyboard.typed)

    def test_a_console_that_cannot_be_drained_is_asked_as_it_stands(self) -> None:
        class _Broken:
            def kbhit(self) -> bool:
                raise OSError("the console handle is gone")

        with patch.dict(sys.modules, {"msvcrt": _Broken()}):
            self.assertEqual(0, jaw_io._drain_typeahead(_Console()))

    def test_the_terminal_is_drained_before_every_question(self) -> None:
        order: list[str] = []
        answers = ["closed", "p"]

        def typed(_question: str) -> str:
            order.append("input")
            return answers.pop(0)

        def drained(*_a: Any, **_k: Any) -> int:
            order.append("drain")
            return 0

        jaws = _toggle([])
        with patch("sys.stdin", _Terminal("")), patch.object(jaw_io, "_drain_typeahead", drained), \
                patch("builtins.input", typed):
            jaws.connect()
        self.assertEqual(["drain", "input", "drain", "input"], order)

    def test_a_question_handed_in_is_never_drained(self) -> None:
        """⭐ THE CONTROL: a console or a test that hands in its own question owns its input."""
        drained: list[object] = []
        with patch.object(jaw_io, "_drain_typeahead", lambda *a, **k: drained.append(a) or 0):
            _toggle([], Person("")).connect()
        self.assertEqual([], drained)


class APickStartTakesAWordNotAnEnterTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D1). Where the count says closed at a pick start, or cannot say, an empty line counted as
    open, so an Enter meant for something else sent the pick on jaws a person never looked at. At connect Enter still
    means open, the owner's agreed answer; at a pick start only the word does, and an empty line is asked again."""

    def _closed(self, *answers: str) -> "tuple[JawIOGripper, Person, list[Any]]":
        events: list[Any] = []
        person = Person("", *answers)
        jaws = _toggle(events, person)
        jaws.connect()
        jaws.set_closed(True)
        events.clear()
        return jaws, person, events

    def test_an_empty_line_at_a_pick_start_is_no_answer_and_is_asked_again(self) -> None:
        jaws, person, events = self._closed("", "open")
        self.assertEqual("", jaws.jaws_open_for_a_pick())
        self.assertEqual(3, len(person.asked), "the empty line was taken as an answer")
        self.assertIn("An empty line is no answer here", person.asked[2])
        self.assertFalse(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))

    def test_the_pick_start_question_asks_for_the_word(self) -> None:
        jaws, person, _events = self._closed("closed", "a")
        jaws.jaws_open_for_a_pick()
        self.assertIn("type 'open' or 'closed'", person.asked[1])
        self.assertNotIn("Enter or 'open'", person.asked[1])

    def test_only_empty_lines_at_a_pick_start_refuse_the_pick_with_nothing_pulsed(self) -> None:
        jaws, person, events = self._closed("", "", "")
        refused = jaws.jaws_open_for_a_pick()
        self.assertIn("not starting the pick", refused)
        self.assertEqual(4, len(person.asked))
        self.assertTrue(jaws.jaws_closed)
        self.assertEqual(0, _pulses(events))

    def test_an_unknowable_edge_takes_the_word_too(self) -> None:
        events: list[Any] = []
        person = Person("", "", "open")
        jaws = _toggle(events, person)
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined]
        with self.assertRaises(RobotError):
            jaws.set_closed(True)
        jaws._io.refuse_high = False                        # type: ignore[attr-defined]
        self.assertEqual("", jaws.jaws_open_for_a_pick())
        self.assertEqual(3, len(person.asked))
        self.assertIn("failed on its high write", person.asked[1])

    def test_enter_at_connect_still_means_open(self) -> None:
        """⭐ THE CONTROL: the owner's agreed answer at program start."""
        person = Person("")
        jaws = _toggle([], person)
        jaws.connect()
        self.assertIn("[Enter or 'open' = open", person.asked[0])
        self.assertFalse(jaws.jaws_closed)


# ---------------------------------------------------------------------------------------------------
# Nothing moves while the question waits
# ---------------------------------------------------------------------------------------------------


class _Meanwhile:
    """A person at the terminal, and what another thread does while the next question waits: ``meanwhile`` runs inside
    that question, once, before its answer comes back. A thread-free double of a console whose second thread
    disconnects, reconnects or commands the hand while a person is still looking at the jaws."""

    def __init__(self, *answers: str) -> None:
        self.person = Person(*answers)
        self.meanwhile: "Callable[[], object] | None" = None

    def __call__(self, question: str) -> str:
        act, self.meanwhile = self.meanwhile, None
        if act is not None:
            act()
        return self.person(question)


class NothingMovesWhileTheQuestionWaitsTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D4, a race found in the console). connect marked the hand connected and started the count
    OPEN before the person had answered, so while the connect question waited another thread saw a connected hand it
    believed open and could pulse it. The hand is connected, and its count started, only once the answer is in; a
    question whose connection changed while it waited is refused without a pulse."""

    def test_while_the_connect_question_waits_the_hand_is_not_connected_and_nothing_pulses(self) -> None:
        events: list[Any] = []
        person = _Meanwhile("open")
        jaws = _toggle(events, person)
        seen: dict[str, object] = {}

        def another_thread() -> None:
            seen["connected"] = jaws.is_connected
            try:
                jaws.set_closed(True)
            except RobotConnectionError as refused:
                seen["refused"] = refused

        person.meanwhile = another_thread
        jaws.connect()

        self.assertIs(False, seen["connected"], "a hand still being asked about read as connected")
        self.assertIn("refused", seen, "a command while the question waited was not refused")
        self.assertEqual(0, _pulses(events))
        self.assertTrue(jaws.is_connected)
        self.assertFalse(jaws.jaws_closed)

    def test_a_disconnect_while_the_connect_question_waits_refuses_the_connect_with_nothing_pulsed(self) -> None:
        events: list[Any] = []
        person = _Meanwhile("closed", "p")
        jaws = _toggle(events, person)
        person.meanwhile = jaws.disconnect

        with self.assertRaises(RobotError) as caught:
            jaws.connect()

        for part in ("not connecting", "while the question waited"):
            self.assertIn(part, str(caught.exception))
        self.assertFalse(jaws.is_connected)
        self.assertEqual(0, _pulses(events))
        self.assertEqual(["p"], person.person.answers, "the pulse was offered on a connection that had changed")

    def test_a_reconnect_while_a_pick_start_question_waits_refuses_the_pick_with_nothing_pulsed(self) -> None:
        events: list[Any] = []
        person = _Meanwhile("", "open", "closed", "p")     # the connect, the reconnect, the pick's stale answers
        jaws = _toggle(events, person)
        jaws.connect()
        jaws.set_closed(True)
        events.clear()

        def reconnect() -> None:
            jaws.disconnect()
            jaws.connect()

        person.meanwhile = reconnect
        refused = jaws.jaws_open_for_a_pick()

        for part in ("not starting the pick", "while the question waited"):
            self.assertIn(part, refused)
        self.assertEqual(0, _pulses(events), "a stale answer pulsed jaws a newer connect had counted")
        self.assertTrue(jaws.is_connected)
        self.assertFalse(jaws.jaws_closed, "the stale answer overwrote the reconnect's count")
        self.assertEqual(["p"], person.person.answers)

    def test_a_disconnect_while_a_pick_start_question_waits_refuses_the_pick(self) -> None:
        events: list[Any] = []
        person = _Meanwhile("", "closed", "p")
        jaws = _toggle(events, person)
        jaws.connect()
        jaws.set_closed(True)
        events.clear()
        person.meanwhile = jaws.disconnect

        refused = jaws.jaws_open_for_a_pick()

        self.assertIn("not starting the pick", refused)
        self.assertEqual(0, _pulses(events))
        self.assertFalse(jaws.is_connected)

    def test_a_solenoid_that_asks_keeps_the_same_rule(self) -> None:
        events: list[Any] = []
        person = _Meanwhile("closed", "p")
        jaws = JawIOGripper(_IO(events), close_output_pin=0, confirm_open_at_start=True, ask=person,
                            sleep=lambda _s: None)
        person.meanwhile = jaws.disconnect
        with self.assertRaises(RobotError):
            jaws.connect()
        self.assertFalse(jaws.is_connected)
        self.assertEqual([], events, "the solenoid was driven on a connection that had changed")


# ---------------------------------------------------------------------------------------------------
# A pick and a place
# ---------------------------------------------------------------------------------------------------


class APickPulsesOnceAtThePartTests(unittest.TestCase):
    def test_no_pulse_before_the_arm_moves_and_one_at_the_part(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))

        report = robot.pick(_pose(), 40.0, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        moves, edges = _moves(events), _edges(events)
        self.assertEqual(3, len(moves))
        self.assertEqual(1, len(edges), "exactly one pulse, the close")
        self.assertLess(moves[1], edges[0], "a pulse before the line down reached the part")
        self.assertLess(edges[0], moves[2], "the lift left before the close")
        self.assertTrue(robot.gripper.jaws_closed)

    def test_an_explicit_pre_open_still_sends_no_pulse(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))

        report = robot.pick(_pose(), 40.0, pre_open_mm=49.99, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual(1, _pulses(events))
        self.assertLess(_moves(events)[1], _edges(events)[0])

    def test_the_pick_report_says_nothing_was_checked(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))

        report = robot.pick(_pose(), 40.0, decline=_BENCH)

        assert report.hand is not None
        self.assertTrue(report.hand.no_sensor)
        self.assertIsNone(report.hand.commanded_width_mm)
        self.assertIsNone(report.hand.reported_width_mm)
        self.assertIn("hold not checked (no sensor)", report.render())
        self.assertNotIn(" mm ", report.hand.render())
        self.assertTrue(report.hand.to_dict()["no_sensor"])

    def test_a_place_pulses_once_after_the_line_down(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))
        robot.pick(_pose(), 40.0, decline=_BENCH)
        events.clear()

        report = robot.place(_pose(x=300.0), decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        moves, edges = _moves(events), _edges(events)
        self.assertEqual(1, len(edges))
        self.assertLess(moves[1], edges[0])
        self.assertLess(edges[0], moves[2])
        self.assertFalse(robot.gripper.jaws_closed)
        self.assertIn("release not checked (no sensor)", report.render())

    def test_a_release_on_open_jaws_sends_nothing_and_says_already_open(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))

        report = robot.release()

        self.assertIs(handling.HandOutcome.RELEASED, report.outcome, report.render())
        self.assertEqual(0, _pulses(events))
        self.assertEqual("already open: no pulse", report.note)
        self.assertIn("already open", report.render())

    def test_a_place_on_open_jaws_sends_nothing_and_says_already_open(self) -> None:
        events: list[Any] = []
        robot = _robot(events, _toggle(events, Person("")))

        report = robot.place(_pose(), decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual(0, _pulses(events))
        self.assertIn("already open", report.render())


class APickOnJawsBelievedClosedAsksTests(unittest.TestCase):
    """A pick with no place before it: the program believes the jaws closed, and asks rather than pulsing."""

    def _closed(self, events: list[Any], *after_connect: str) -> Robot:
        person = Person("", *after_connect)
        robot = _robot(events, _toggle(events, person))
        robot.gripper.set_closed(True)
        events.clear()
        self._person = person
        return robot

    def test_the_person_says_open_and_the_pick_runs_with_one_pulse(self) -> None:
        events: list[Any] = []
        robot = self._closed(events, "open")

        report = robot.pick(_pose(), 40.0, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        self.assertEqual(2, len(self._person.asked), "the pick did not ask")
        self.assertIn("believes they stand CLOSED", self._person.asked[1])
        self.assertEqual(1, _pulses(events))
        self.assertLess(_moves(events)[1], _edges(events)[0])

    def test_the_person_says_closed_and_chooses_the_pulse(self) -> None:
        events: list[Any] = []
        robot = self._closed(events, "closed", "p")

        report = robot.pick(_pose(), 40.0, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.EXECUTED, report.outcome, report.render())
        edges, moves = _edges(events), _moves(events)
        self.assertEqual(2, len(edges), "the pulse the person chose, then the close at the part")
        self.assertLess(edges[0], moves[0])
        self.assertLess(moves[1], edges[1])

    def test_the_person_aborts_and_nothing_moves(self) -> None:
        events: list[Any] = []
        robot = self._closed(events, "closed", "a")

        report = robot.pick(_pose(), 40.0, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
        self.assertIn("not starting the pick", report.message)
        self.assertEqual([], events)

    def test_with_nobody_to_ask_the_pick_is_refused_before_any_motion(self) -> None:
        events: list[Any] = []
        robot = self._closed(events)
        robot.gripper._ask = None

        with patch("sys.stdin", _NO_TERMINAL):
            report = robot.pick(_pose(), 40.0, decline=_BENCH)

        self.assertIs(handling.HandlingOutcome.REFUSED, report.outcome, report.render())
        for part in ("not starting the pick", "CLOSED", "terminal"):
            self.assertIn(part, report.message)
        self.assertEqual([], events, "a motion, a detach or a pulse came before the refusal")
        self.assertTrue(robot.gripper.jaws_closed)


# ---------------------------------------------------------------------------------------------------
# The pick loop's policy
# ---------------------------------------------------------------------------------------------------


class ThePolicyKeepsTheSameRulesTests(unittest.TestCase):
    def test_no_pulse_before_the_approach_and_one_at_the_part(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        arm = _Arm(events)

        report = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99).execute(_grasp())

        self.assertIs(PolicyOutcome.EXECUTED, report.outcome, report.motion_message)
        self.assertIsNone(report.object_detected, "nothing measures a toggle")
        edges = _edges(events)
        moves = [i for i, e in enumerate(events) if e == "move"]
        self.assertEqual(1, len(edges))
        self.assertLess(moves[-2], edges[0], "a pulse before the approach reached the part")
        self.assertLess(edges[0], moves[-1])

    def test_jaws_believed_closed_with_nobody_to_ask_end_the_pick_before_anything(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws.set_closed(True)
        jaws._ask = None
        events.clear()
        arm = _Arm(events)

        with patch("sys.stdin", _NO_TERMINAL):
            report = GraspExecutionPolicy(arm=arm, gripper=jaws, pre_open_width_mm=49.99).execute(_grasp())

        self.assertIs(PolicyOutcome.GRIPPER_FAULT, report.outcome)
        self.assertIn("not starting the pick", report.motion_message)
        self.assertEqual(["status"], events, "only the controller was asked")
        self.assertEqual((), report.waypoints)

    def test_a_driver_error_at_the_close_is_an_outcome_not_a_raise(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined] # the controller refuses from here
        policy = GraspExecutionPolicy(arm=_Arm(events), gripper=jaws)

        report = policy.execute(_grasp())

        self.assertIs(PolicyOutcome.GRIPPER_FAULT, report.outcome)
        self.assertIsInstance(report.error, RobotError)
        self.assertIn("the gripper raised", report.motion_message)
        self.assertEqual(policy.approach_steps, len(report.waypoints), "the motions before the fault were lost")

    def test_a_driver_error_at_a_solenoids_pre_open_is_an_outcome_not_a_raise(self) -> None:
        events: list[Any] = []
        wiring = _IO(events)
        jaws = JawIOGripper(wiring, close_output_pin=0, sleep=lambda _s: None)
        jaws.connect()
        wiring.refuse_all = True

        report = GraspExecutionPolicy(arm=_Arm(events), gripper=jaws, pre_open_width_mm=49.99).execute(_grasp())

        self.assertIs(PolicyOutcome.GRIPPER_FAULT, report.outcome)
        self.assertEqual((), report.waypoints)
        self.assertNotIn("move", events)

    def test_the_services_policy_drops_the_pre_open_for_a_toggle(self) -> None:
        from src.robot.grasping.motion.grasp_motion import GraspMotion, build_execution_policy

        jaws = _toggle([], Person(""))
        for motion in (GraspMotion(), GraspMotion(pre_open_width_mm=80.0)):  # 80 mm: wider than the hand, unchecked
            with self.subTest(motion=motion.to_dict()):
                policy = build_execution_policy(motion, arm=_Arm([]), gripper=jaws, standoff_mm=80.0,
                                                retreat_mm=100.0, base_frame_required=False)
                self.assertIsNone(policy.pre_open_width_mm)


class ACampaignAsksAtEveryPickAfterAClose(unittest.TestCase):
    def test_a_campaign_of_picks_without_a_place_asks_before_each_later_pick(self) -> None:
        from src.robot.execution.pick_run import PickRun, Recording
        from tests.test_a_stopped_controller_moves_no_jaws import _service

        events: list[Any] = []
        person = Person("", "closed", "p", "closed", "p")
        jaws = _toggle(events, person)
        jaws.connect()

        run = PickRun.from_service(_service(_Arm(events), jaws), runs=3, recording=Recording.off()).execute()

        self.assertEqual(3, run.succeeded)
        self.assertEqual(5, len(person.asked), "the connect, then each later pick asked where the jaws stand")
        self.assertEqual(5, _pulses(events), "a close per pick, and the open each person chose before a later pick")

    def test_a_campaign_on_jaws_believed_closed_with_nobody_to_ask_stops_before_anything(self) -> None:
        from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
        from tests.test_a_stopped_controller_moves_no_jaws import _Counted, _service

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws.set_closed(True)                               # the last pick of an earlier campaign, never placed
        jaws._ask = None
        events.clear()
        service = _Counted(_service(_Arm(events), jaws))

        with patch("sys.stdin", _NO_TERMINAL):
            run = PickRun.from_service(service, runs=3, recording=Recording.off()).execute()

        self.assertEqual(1, service.picks, "the campaign went on after a hand that needs a person")
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED, PickOutcome.CANCELLED],
                         [a.outcome for a in run.attempts], run.render())
        for part in ("the gripper needs a person", "not starting the pick", "terminal"):
            self.assertIn(part, run.attempts[0].detail)
        self.assertEqual(3, run.exit_code)
        # Only the controller was asked, whether it stands stopped, before the person (a stopped controller moves no
        # jaws, 2026-09-24): no motion and no camera came before the refusal.
        self.assertEqual(["status"], events, "the arm moved or the camera was asked before the refusal")
        self.assertTrue(jaws.jaws_closed)
        last = run.last
        self.assertIsNone(last.pick_report)
        self.assertIn("not starting the pick", last.gripper_fault)
        self.assertIn("gripper    needs a person", last.render())
        self.assertIn("not starting the pick", last.to_dict()["gripper_fault"])

    def test_a_driver_error_at_the_close_stops_the_campaign_with_the_motions_on_the_report(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from src.robot.execution.pick_run import PickOutcome, PickRun, Recording
        from src.robot.grasping.loop.pick_loop import PickOutcome as LoopOutcome
        from tests.test_a_stopped_controller_moves_no_jaws import _Counted, _service

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined] # the controller refuses from here
        service = _Counted(_service(_Arm(events), jaws))

        run = PickRun.from_service(service, runs=2, recording=Recording.off()).execute()

        self.assertEqual(1, service.picks)
        self.assertEqual([PickOutcome.RAISED, PickOutcome.CANCELLED], [a.outcome for a in run.attempts], run.render())
        self.assertIn("the gripper raised: RobotError", run.attempts[0].detail)
        last = run.last
        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, last.outcome)
        self.assertIsNone(last.fault, "a raise the policy reported is not a raise out of the pick")
        self.assertIs(LoopOutcome.GRIPPER_FAULT, last.pick_report.outcome)
        self.assertEqual("gripper_fault", last.pick_report.attempts[-1].action)
        self.assertIn("move", events, "the approach before the fault is gone from the log")

    def test_the_two_scan_path_asks_before_its_standoff(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from tests.test_a_stopped_controller_moves_no_jaws import two_scan_service

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws.set_closed(True)
        jaws._ask = None
        events.clear()

        with patch("sys.stdin", _NO_TERMINAL):
            report = two_scan_service(_Arm(events), jaws).pick()

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome, report.render())
        self.assertIn("not starting the pick", report.gripper_fault)
        self.assertEqual(["status"], events, "the standoff move came before the question")  # only the controller

    def test_the_two_scan_path_names_a_driver_error_at_the_close(self) -> None:
        from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome
        from tests.test_a_stopped_controller_moves_no_jaws import two_scan_service

        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined]

        report = two_scan_service(_Arm(events), jaws).pick()

        self.assertIs(AutonomousGraspOutcome.EXECUTION_FAILED, report.outcome, report.render())
        self.assertIsNone(report.fault)
        self.assertIn("the gripper raised: RobotError", report.gripper_fault)


# ---------------------------------------------------------------------------------------------------
# The stroke
# ---------------------------------------------------------------------------------------------------


class AnOpenWaitsTheStrokeTests(unittest.TestCase):
    """⛔ The owner's Hand-E takes up to about 2 s to open, and a place backed out the moment the valve was told."""

    def test_an_open_after_a_close_waits_the_travel_time(self) -> None:
        for actuation in ("single_toggle", "single_solenoid"):
            with self.subTest(actuation=actuation):
                slept: list[float] = []
                jaws = JawIOGripper(_IO([]), actuation=actuation, close_output_pin=0, pulse_s=0.0, close_settle_s=2.0,
                                    min_width_mm=5.0, max_width_mm=49.99, ask=Person(""), sleep=slept.append)
                jaws.connect()
                jaws.set_closed(True)
                slept.clear()
                jaws.set_closed(False)
                self.assertIn(2.0, slept, "the open did not wait the stroke")

    def test_an_open_on_open_jaws_waits_nothing(self) -> None:
        slept: list[float] = []
        jaws = _toggle([], Person(""), sleep=slept.append)
        jaws.connect()
        jaws.set_closed(False)
        self.assertNotIn(2.0, slept)


class TheDeskAsksForTheStrokeTests(unittest.TestCase):
    """⛔ An open with no open switch waits close_settle_s, still the 0.3 s schema default unless measured, so a release
    returned 0.5 s after the edge on a Hand-E that takes about 2 s, and nothing at the desk asked for the time."""

    def _row(self, **jaw_io: object) -> Any:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import run_config_preflight

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "ik"},
            "gripper": {"vendor": "jaw_io", "jaw_io": {"close_output_pin": 0, **jaw_io}},
            "safety": {"self_collision": {"backend": "capsule"}},
        })
        rows = {check.name: check for check in run_config_preflight(cfg, curobo_available=True,
                                                                    collision_engine="coal").checks}
        return rows.get("jaw travel time")

    def test_the_owners_toggle_at_the_default_settle_is_warned(self) -> None:
        row = self._row(actuation="single_toggle")
        self.assertEqual("warn", str(row.status))
        self.assertIn("both ways", row.detail)
        self.assertIn("close_settle_s", row.fix)
        self.assertIn("2 s", row.fix)

    def test_a_measured_settle_reads_ok(self) -> None:
        """⭐ THE CONTROL: the owner's layer sets 2.0 s."""
        row = self._row(actuation="single_toggle", close_settle_s=2.0)
        self.assertEqual("ok", str(row.status))
        self.assertIn("2.0 s", row.detail)

    def test_a_solenoid_with_both_switches_is_not_warned(self) -> None:
        row = self._row(actuation="single_solenoid", closed_confirm_input_pin=1, open_confirm_input_pin=2,
                        io_port="standard")
        self.assertNotEqual("warn", str(row.status))


class TheDeskAsksForAPulseTheHandRegistersTests(unittest.TestCase):
    """Review of 2026-09-24 (D3). The load refuses a toggle's pulse under 0.05 s, a handful of CB3 controller cycles; the
    desk warns below 0.1 s, where a pulse the controller did apply may still be too short for the device to flip on."""

    def _row(self, **jaw_io: object) -> Any:
        from src.config.schema.robot import RobotConfig
        from src.robot.execution.real_cell.preflight import run_config_preflight

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "ik"},
            "gripper": {"vendor": "jaw_io", "jaw_io": {"close_output_pin": 0, **jaw_io}},
            "safety": {"self_collision": {"backend": "capsule"}},
        })
        rows = {check.name: check for check in run_config_preflight(cfg, curobo_available=True,
                                                                    collision_engine="coal").checks}
        return rows.get("toggle pulse")

    def test_a_toggle_pulse_under_a_tenth_of_a_second_is_warned(self) -> None:
        row = self._row(actuation="single_toggle", pulse_s=0.06)
        self.assertEqual("warn", str(row.status))
        self.assertIn("0.06 s", row.detail)
        self.assertIn("--pulse 0 --for", row.fix)

    def test_the_default_pulse_reads_ok(self) -> None:
        """⭐ THE CONTROL: the schema default, 0.2 s."""
        row = self._row(actuation="single_toggle")
        self.assertEqual("ok", str(row.status))
        self.assertIn("0.20 s", row.detail)

    def test_a_solenoid_has_no_such_row(self) -> None:
        self.assertIsNone(self._row(actuation="single_solenoid"))


class TheDriverCountsWhatItSentTests(unittest.TestCase):
    """What the bench's summary reads: every command the driver sent the jaws, counted where it is sent."""

    def test_a_toggle_counts_one_per_rising_edge_the_connect_included(self) -> None:
        events: list[Any] = []
        jaws = _toggle(events, Person("closed", "p"))
        self.assertEqual(0, jaws.commands_sent)
        jaws.connect()                                      # the person chose the pulse: one edge
        self.assertEqual(1, jaws.commands_sent)
        jaws.set_closed(True)
        jaws.set_closed(True)                               # already closed: nothing sent
        jaws.set_closed(False)
        self.assertEqual(3, jaws.commands_sent)
        self.assertEqual(_pulses(events), jaws.commands_sent)

    def test_a_high_write_that_raised_is_counted_as_sent(self) -> None:
        """A write that raised may still have reached the controller, so the count never says less went out."""
        events: list[Any] = []
        jaws = _toggle(events, Person(""))
        jaws.connect()
        jaws._io.refuse_high = True                         # type: ignore[attr-defined]
        with self.assertRaises(RobotError):
            jaws.set_closed(True)
        self.assertEqual(1, jaws.commands_sent)

    def test_a_solenoid_counts_every_actuation(self) -> None:
        jaws = JawIOGripper(_IO([]), actuation="single_solenoid", close_output_pin=0, pulse_s=0.0, close_settle_s=0.0,
                            min_width_mm=5.0, max_width_mm=49.99, sleep=lambda _s: None)
        jaws.connect()                                      # no feedback and not opted in: nothing actuated
        self.assertEqual(0, jaws.commands_sent)
        jaws.set_closed(True)
        jaws.set_closed(False)
        self.assertEqual(2, jaws.commands_sent)


# ---------------------------------------------------------------------------------------------------
# The bench
# ---------------------------------------------------------------------------------------------------


class _BenchArm(_IO):
    """What ``create_arm`` hands the bench: the controller's digital I/O, a connect, and where the arm stands."""

    def __init__(self) -> None:
        super().__init__([])

    def connect(self) -> None:
        return None

    def disconnect(self) -> None:
        return None

    def get_joint_positions(self) -> Any:
        from src.robot.core import JointPositions

        return JointPositions(np.radians([-45.0, -100.2, -110.0, -60.0, 90.0, -0.0]))

    def get_tcp_pose(self) -> Pose:
        return Pose.tool_down(-130.0, -700.0, 250.0)


class _Terminal(io.StringIO):
    """A stdin that is a terminal, so the driver asks through ``input``, which the test answers."""

    def isatty(self) -> bool:
        return True


#: The owner's hand as the bench builds it from config: single_toggle on tool DO0, no stroke to wait out here.
_BENCH_TOGGLE = {"vendor": "jaw_io", "min_width_mm": 5.0, "max_width_mm": 49.99,
                 "jaw_io": {"actuation": "single_toggle", "close_output_pin": 0, "close_settle_s": 0.0,
                            "pulse_s": 0.05}}


class TheBenchTests(unittest.TestCase):
    def _bench(self, *argv: str, gripper: "dict[str, Any] | None" = None,
               arm: "_BenchArm | None" = None) -> tuple[int, str]:
        from contextlib import redirect_stdout

        from src.config.schema.robot import RobotConfig
        from src.robot.drivers.ur import __main__ as bench_cli

        cfg = RobotConfig.model_validate({"vendor": "ur", "gripper": gripper or {"vendor": "robotiq"}})
        bench_arm = arm if arm is not None else _BenchArm()
        out = io.StringIO()
        with patch.object(bench_cli, "_load_robot_config", lambda *_a: cfg), \
                patch("src.robot.drivers.create_arm", lambda *_a, **_k: bench_arm), \
                patch.dict("os.environ", {"WILLY_PROFILE": ""}), redirect_stdout(out):
            code = bench_cli.main(list(argv))
        return code, out.getvalue()

    def _jaws(self, command: str, *answers: str) -> tuple[int, str, int]:
        """``--jaws command --yes`` on the owner's toggle, a person at the terminal answering ``answers``.

        The exit code, the last line printed, and the rising edges the controller saw on DO0.
        """
        arm = _BenchArm()
        person = Person(*answers)
        with patch("sys.stdin", _Terminal("")), patch("builtins.input", person):
            code, printed = self._bench("--jaws", command, "--yes", gripper=_BENCH_TOGGLE, arm=arm)
        self.assertEqual([], person.answers, "a scripted answer was never asked for")
        return code, printed.strip().splitlines()[-1], _pulses(arm.events)

    def test_where_prints_the_joints_as_a_line_to_paste_and_the_tcp(self) -> None:
        code, printed = self._bench("--where")
        self.assertEqual(0, code)
        lines = printed.splitlines()
        self.assertEqual("JointPositions.deg(-45.0, -100.2, -110.0, -60.0, 90.0, 0.0)", lines[0])
        self.assertTrue(lines[1].startswith("TCP (base)  x -130.0  y -700.0  z 250.0 mm  rx "), lines[1])
        self.assertTrue(lines[1].endswith(" deg (rotation vector)"), lines[1])

    def test_a_refusal_names_the_config_it_read_and_no_placeholder(self) -> None:
        code, printed = self._bench("--jaws", "open", "--yes")
        self.assertEqual(1, code)
        self.assertIn("REFUSED: --jaws drives a jaw_io hand", printed)
        self.assertIn("neither --profile nor WILLY_PROFILE was given", printed)
        self.assertIn("names gripper vendor 'robotiq'", printed)
        self.assertIn("--profile NAME", printed)
        self.assertNotIn("<cell>", printed)
        code, printed = self._bench("--profile", "ur10,hande", "--jaws", "open", "--yes")
        self.assertIn("profile chain 'ur10,hande' (from --profile)", printed)

    def test_jaws_through_the_driver_asks_first(self) -> None:
        toggle = {"vendor": "jaw_io", "min_width_mm": 5.0, "max_width_mm": 49.99,
                  "jaw_io": {"actuation": "single_toggle", "close_output_pin": 0, "close_settle_s": 0.0,
                             "pulse_s": 0.05}}
        with patch("sys.stdin", _NO_TERMINAL):
            code, printed = self._bench("--jaws", "closed", "--yes", gripper=toggle)
        self.assertEqual(1, code)
        self.assertIn("nobody can be asked", printed)

    def test_a_pulse_the_connect_sent_is_counted_in_the_summary(self) -> None:
        """⛔ Review of 2026-09-24. The person says closed and chooses the pulse, so the connect opens the jaws; then
        ``--jaws open`` has nothing left to do. Red before: one rising edge went out, and the summary read "from OPEN
        to OPEN (they already stood there: nothing was pulsed)", because it looked only after the connect."""
        code, line, edges = self._jaws("open", "closed", "p")

        self.assertEqual(0, code)
        self.assertEqual(1, edges)
        self.assertNotIn("nothing was pulsed", line)
        self.assertIn("1 pulse went out", line)
        self.assertIn("1 pulse at connect", line)
        self.assertIn("nothing for --jaws open", line)

    def test_the_summary_counts_the_pulses_of_the_connect_and_of_the_command_together(self) -> None:
        code, line, edges = self._jaws("closed", "closed", "p")

        self.assertEqual(0, code)
        self.assertEqual(2, edges)
        self.assertIn("from OPEN to CLOSED", line)
        self.assertIn("2 pulses went out", line)
        self.assertIn("1 pulse at connect", line)
        self.assertIn("1 pulse for --jaws closed", line)

    def test_nothing_was_pulsed_is_said_only_where_nothing_was(self) -> None:
        """⭐ THE CONTROLS: jaws answered open. ``--jaws open`` sends nothing and says so; ``--jaws closed`` sends one."""
        code, line, edges = self._jaws("open", "")
        self.assertEqual((0, 0), (code, edges))
        self.assertIn("from OPEN to OPEN (they already stood there: nothing was pulsed)", line)

        code, line, edges = self._jaws("closed", "open")
        self.assertEqual((0, 1), (code, edges))
        self.assertIn("from OPEN to CLOSED", line)
        self.assertIn("1 pulse went out", line)
        self.assertNotIn("at connect", line)
        self.assertNotIn("nothing was pulsed", line)

    def test_a_refusal_after_a_pulse_was_tried_says_so(self) -> None:
        """The person chose the pulse at connect and the controller refused its high write: the connect is refused,
        and the refusal says a pulse was tried, which may have flipped the jaws, rather than the driver's words alone."""
        arm = _BenchArm()
        arm.refuse_high = True
        with patch("sys.stdin", _Terminal("")), patch("builtins.input", Person("closed", "p")):
            code, printed = self._bench("--jaws", "open", "--yes", gripper=_BENCH_TOGGLE, arm=arm)

        self.assertEqual(1, code)
        self.assertIn("REFUSED: --jaws open", printed)
        self.assertIn("Before this refusal the driver sent 1 pulse at connect", printed)
        self.assertIn("look at the jaws", printed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
