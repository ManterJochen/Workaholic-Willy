"""A parallel-jaw gripper a real cell can actually be configured with — over digital I/O.

The September cell drives its gripper through the controller's digital I/O; that is how it is wired.
Willy had two jaw paths and neither was that one: the Robotiq driver speaks a socket the URCap opens
on port 63352, and `schunk` / `franka_hand` were empty enum slots. Suction over I/O existed. Jaws
over I/O did not.

So this driver is vendor-neutral for the same reason ``vacuum.py`` is: a pneumatic or electric
two-finger gripper on a UR is one or two output pins plus, usually, reed switches on inputs. No SDK,
nothing manufacturer-specific — the cell can be configured and tested before the gripper is bought,
and bring-up becomes measuring pin numbers rather than writing code.

Two things here are worth more than the driver:

* **The reed pair.** ``closed_confirm`` after a close command means the jaws MET — an empty grasp.
  Neither switch means they stopped in between — a part is held. That makes this an
  ``ObjectDetectingGripper``, so the jaw path gets real post-close verification for the first time.
* **``connect()`` does not blindly open.** Suction asserts OFF on connect because releasing a cup is
  cheap. Opening a jaw that was left closed drops a rigid part wherever the arm is standing. The
  rule (operator, 2026-08-17) is: open only when the sensor proves the jaws are empty.

STATUS: bucket (3). This has never touched a real gripper. These tests are evidence that the LOGIC
is right; they say nothing about the wiring.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.schema.robot.robot_schema import GripperConfig, JawIOGripperConfig, VacuumGripperConfig
from src.robot.core import Gripper, GripperVendor, RobotConnectionError, RobotError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.core.gripper import HoldEvidence, ObjectDetectingGripper
from src.robot.grippers import create_gripper
from src.robot.grippers.jaw_io import JawIOGripper, JawState

CLOSE_PIN, OPEN_PIN = 4, 5
CLOSED_SW, OPEN_SW, PART_SW = 0, 1, 2


class FakeIO:
    """A controller's digital I/O, recorded. Stands in for the arm driver."""

    def __init__(self, inputs: dict[int, bool] | None = None) -> None:
        self.outputs: dict[int, bool] = {}
        self.inputs = dict(inputs or {})
        self.writes: list[tuple[int, bool, str]] = []

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        self.outputs[pin] = bool(value)
        self.writes.append((pin, bool(value), str(port)))

    def get_digital_input(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        return bool(self.inputs.get(pin, False))

    def get_digital_output(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        return bool(self.outputs.get(pin, False))

    def set_analog_output(self, pin, value, *, current: bool = False) -> None:
        pass


def _reed_gripper(io: FakeIO, **kw) -> JawIOGripper:
    """The strongest wiring: both end-stop reed switches."""
    return JawIOGripper(
        io, close_output_pin=CLOSE_PIN, closed_confirm_input_pin=CLOSED_SW,
        open_confirm_input_pin=OPEN_SW, sleep=lambda _s: None, **kw,
    )


class ProtocolAndRegistryTests(unittest.TestCase):
    def test_it_is_a_gripper_and_advertises_object_detection(self) -> None:
        g = _reed_gripper(FakeIO())
        self.assertIsInstance(g, Gripper)
        # The whole point: the jaw path can now be ASKED, not just trusted.
        self.assertIsInstance(g, ObjectDetectingGripper)

    def test_the_registry_builds_it_by_vendor(self) -> None:
        g = create_gripper(
            GripperVendor.JAW_IO, io=FakeIO(), close_output_pin=CLOSE_PIN, sleep=lambda _s: None,
        )
        self.assertIsInstance(g, JawIOGripper)

    def test_it_refuses_an_io_source_that_cannot_do_io(self) -> None:
        with self.assertRaises(TypeError):
            JawIOGripper(object())  # type: ignore[arg-type]

    def test_commands_before_connect_are_refused(self) -> None:
        g = _reed_gripper(FakeIO())
        for call in (lambda: g.set_width_mm(0.0), g.activate, g.is_object_detected):
            with self.assertRaises(RobotConnectionError):
                call()


class SingleSolenoidTests(unittest.TestCase):
    """One output, spring return: high = close, low = open."""

    def setUp(self) -> None:
        self.io = FakeIO({OPEN_SW: True})
        self.g = _reed_gripper(self.io)
        self.g.connect()

    def test_close_drives_the_pin_high_and_open_drops_it(self) -> None:
        self.io.inputs = {CLOSED_SW: True}          # jaws meet: empty
        self.g.set_width_mm(0.0)
        self.assertTrue(self.io.outputs[CLOSE_PIN])
        self.io.inputs = {OPEN_SW: True}
        self.g.set_width_mm(80.0)
        self.assertFalse(self.io.outputs[CLOSE_PIN])

    def test_the_width_threshold_is_what_decides(self) -> None:
        self.io.inputs = {CLOSED_SW: True}
        self.g.set_width_mm(5.0)                    # == closed_below_mm -> closes
        self.assertTrue(self.io.outputs[CLOSE_PIN])
        self.io.inputs = {OPEN_SW: True}
        self.g.set_width_mm(5.01)                   # just above -> opens
        self.assertFalse(self.io.outputs[CLOSE_PIN])

    def test_width_is_reported_as_the_two_bands(self) -> None:
        # Binary hardware: there is no encoder, so the only honest answer is which band was commanded.
        self.io.inputs = {CLOSED_SW: True}
        self.g.set_width_mm(0.0)
        self.assertEqual(self.g.get_width_mm(), self.g.min_width_mm)
        self.io.inputs = {OPEN_SW: True}
        self.g.set_width_mm(80.0)
        self.assertEqual(self.g.get_width_mm(), self.g.max_width_mm)

    def test_an_open_pin_on_a_single_solenoid_is_refused(self) -> None:
        # Opening IS dropping the close pin; a second pin would never be driven, so a config that
        # declares one is describing hardware this actuation mode does not have.
        with self.assertRaises(ValueError):
            JawIOGripper(FakeIO(), close_output_pin=CLOSE_PIN, open_output_pin=OPEN_PIN,
                         actuation="single_solenoid")


class DoubleSolenoidTests(unittest.TestCase):
    """Two outputs, pulsed, bistable — it latches, so the coil must not be held."""

    def setUp(self) -> None:
        self.io = FakeIO({OPEN_SW: True})
        self.g = JawIOGripper(
            self.io, actuation="double_solenoid", close_output_pin=CLOSE_PIN,
            open_output_pin=OPEN_PIN, closed_confirm_input_pin=CLOSED_SW,
            open_confirm_input_pin=OPEN_SW, sleep=lambda _s: None,
        )
        self.g.connect()
        self.io.writes.clear()

    def test_closing_pulses_the_close_coil_and_leaves_it_low(self) -> None:
        self.io.inputs = {CLOSED_SW: True}
        self.g.set_width_mm(0.0)
        # A latching valve is thrown, not held: holding the coil high is what cooks it.
        self.assertFalse(self.io.outputs[CLOSE_PIN])
        self.assertIn((CLOSE_PIN, True, str(DigitalIOPort.TOOL)), self.io.writes)

    def test_the_opposite_coil_is_dropped_before_the_other_is_driven(self) -> None:
        # Both coils energised at once stalls a real spool. Order matters, so assert the order.
        self.io.inputs = {CLOSED_SW: True}
        self.g.set_width_mm(0.0)
        pins = [(pin, val) for pin, val, _port in self.io.writes]
        self.assertEqual(pins[0], (OPEN_PIN, False))
        self.assertEqual(pins[1], (CLOSE_PIN, True))

    def test_a_double_solenoid_without_an_open_coil_is_refused(self) -> None:
        # It would latch closed on the first grasp and never release.
        with self.assertRaises(ValueError):
            JawIOGripper(FakeIO(), actuation="double_solenoid", close_output_pin=CLOSE_PIN)

    def test_an_unknown_actuation_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            JawIOGripper(FakeIO(), actuation="hydraulic")


class ReedPairTests(unittest.TestCase):
    """The measurement the jaw path never had: closed-on-nothing vs closed-on-a-part."""

    def setUp(self) -> None:
        self.io = FakeIO({OPEN_SW: True})
        self.g = _reed_gripper(self.io)
        self.g.connect()

    def _close_with(self, closed: bool, open_: bool) -> None:
        self.io.inputs = {CLOSED_SW: closed, OPEN_SW: open_}
        self.g.set_width_mm(0.0)

    def test_fully_closed_after_a_close_is_an_EMPTY_grasp(self) -> None:
        # The jaws met each other -> nothing was between them. This is the case a trusted close
        # command reports as a success, and it is exactly the one that loses parts downstream.
        self._close_with(closed=True, open_=False)
        self.assertIs(self.g.read_jaw_state(), JawState.CLOSED_EMPTY)
        self.assertFalse(self.g.is_object_detected())

    def test_stopped_between_the_end_stops_is_a_HELD_part(self) -> None:
        self._close_with(closed=False, open_=False)
        self.assertIs(self.g.read_jaw_state(), JawState.HOLDING)
        self.assertTrue(self.g.is_object_detected())

    def test_both_switches_active_is_impossible_and_fails_closed(self) -> None:
        # Mechanically it cannot happen, so it is the wiring or a dead sensor. Never a grasp.
        self._close_with(closed=True, open_=True)
        self.assertIs(self.g.read_jaw_state(), JawState.CONTRADICTORY)
        self.assertFalse(self.g.is_object_detected())

    def test_open_jaws_never_report_a_held_part(self) -> None:
        # During travel the reed pair reads "between the end stops", which looks identical to a
        # grasp. Gating on the commanded state is what keeps a moving gripper from claiming one.
        self.io.inputs = {CLOSED_SW: False, OPEN_SW: False}
        self.g.set_width_mm(80.0)
        self.assertFalse(self.g.is_object_detected())


class ThinnerWiringTests(unittest.TestCase):
    """What each cheaper wiring can and cannot answer — stated, not glossed over."""

    def test_one_closed_switch_still_detects_an_empty_grasp(self) -> None:
        io = FakeIO()
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, closed_confirm_input_pin=CLOSED_SW,
                         sleep=lambda _s: None)
        g.connect()
        io.inputs = {CLOSED_SW: True}
        g.set_width_mm(0.0)
        self.assertFalse(g.is_object_detected())     # jaws met -> empty
        io.inputs = {CLOSED_SW: False}
        g.set_width_mm(0.0)
        self.assertTrue(g.is_object_detected())      # short of the stop -> holding

    def test_one_OPEN_switch_cannot_detect_an_empty_grasp_and_says_so(self) -> None:
        # Off the open stop, "holding" and "closed on nothing" are the same reading. A driver that
        # picked one would report every empty close as a successful grasp -- so it reports UNKNOWN
        # and falls back to the commanded state, which is the honest "as far as I know".
        io = FakeIO({OPEN_SW: True})
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, open_confirm_input_pin=OPEN_SW,
                         sleep=lambda _s: None)
        g.connect()
        io.inputs = {OPEN_SW: False}
        g.set_width_mm(0.0)
        self.assertIs(g.read_jaw_state(), JawState.UNKNOWN)

    def test_a_part_present_sensor_wins_over_the_reed_inference(self) -> None:
        # It answers the question directly; the reed pair answers it second-hand.
        io = FakeIO({OPEN_SW: True})
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, part_present_input_pin=PART_SW,
                         closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW,
                         sleep=lambda _s: None)
        g.connect()
        io.inputs = {PART_SW: True, CLOSED_SW: True, OPEN_SW: False}  # reeds say EMPTY
        g.set_width_mm(0.0)
        self.assertTrue(g.is_object_detected())

    def test_without_any_feedback_it_reports_the_commanded_state(self) -> None:
        io = FakeIO()
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, sleep=lambda _s: None)
        g.connect()
        self.assertFalse(g.has_feedback)
        g.set_width_mm(0.0)
        self.assertTrue(g.is_object_detected())      # "as far as this driver knows"
        g.set_width_mm(80.0)
        self.assertFalse(g.is_object_detected())


class ConnectSafetyTests(unittest.TestCase):
    """The rule that differs from suction on purpose: never drop an unchecked workpiece."""

    def test_a_held_part_is_NOT_released_on_connect(self) -> None:
        io = FakeIO({CLOSED_SW: False, OPEN_SW: False})   # between the stops -> holding
        g = _reed_gripper(io)
        g.connect()
        self.assertNotIn(CLOSE_PIN, io.outputs)          # nothing actuated at all
        self.assertTrue(g.jaws_closed)                   # and it knows it is holding

    def test_proven_empty_jaws_ARE_opened_on_connect(self) -> None:
        io = FakeIO({CLOSED_SW: True, OPEN_SW: False})   # fully closed on nothing
        g = _reed_gripper(io)
        g.connect()
        self.assertIs(io.outputs[CLOSE_PIN], False)      # safe, known state
        self.assertFalse(g.jaws_closed)

    def test_contradictory_sensors_stop_it_from_actuating(self) -> None:
        io = FakeIO({CLOSED_SW: True, OPEN_SW: True})
        g = _reed_gripper(io)
        g.connect()
        self.assertNotIn(CLOSE_PIN, io.outputs)

    def test_without_feedback_it_does_not_actuate_by_default(self) -> None:
        # "Proven empty" is unreachable with no sensor, and the whole point of the rule is not to
        # drop what has not been checked. This is the difference from the suction driver.
        io = FakeIO()
        JawIOGripper(io, close_output_pin=CLOSE_PIN, sleep=lambda _s: None).connect()
        self.assertNotIn(CLOSE_PIN, io.outputs)

    def test_the_suction_style_behaviour_is_available_as_an_opt_in(self) -> None:
        io = FakeIO()
        JawIOGripper(io, close_output_pin=CLOSE_PIN, open_on_connect_without_feedback=True,
                     sleep=lambda _s: None).connect()
        self.assertIs(io.outputs[CLOSE_PIN], False)

    def test_disconnect_does_NOT_release(self) -> None:
        # Teardown is often an error path, and that is the worst moment to drop a workpiece.
        io = FakeIO({CLOSED_SW: False, OPEN_SW: False})
        g = _reed_gripper(io)
        g.connect()
        io.writes.clear()
        g.disconnect()
        self.assertEqual(io.writes, [])
        self.assertFalse(g.is_connected)


class TimingTests(unittest.TestCase):
    def test_a_close_that_never_settles_is_a_missed_grasp_not_a_crash(self) -> None:
        # The jaws stay off both stops forever -> HOLDING is reached immediately here, so drive the
        # genuinely unreadable case: a lone open switch that never leaves UNKNOWN.
        io = FakeIO({OPEN_SW: False})
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, open_confirm_input_pin=OPEN_SW,
                         close_timeout_s=0.01, sleep=lambda _s: None)
        g.connect()
        g.set_width_mm(0.0)                              # must return, not raise
        self.assertTrue(io.outputs[CLOSE_PIN])

    def test_a_feedback_free_close_waits_the_configured_travel_time(self) -> None:
        # With nothing to poll, the wait IS the only thing between "commanded" and "gripped".
        slept: list[float] = []
        g = JawIOGripper(FakeIO(), close_output_pin=CLOSE_PIN, close_settle_s=0.42,
                         sleep=slept.append)
        g.connect()
        g.set_width_mm(0.0)
        self.assertIn(0.42, slept)


class _Timeline(FakeIO):
    """FakeIO that also records the driver's sleeps, so a pulse reads as low, wait, high, wait, low in order."""

    def __init__(self, inputs: dict[int, bool] | None = None) -> None:
        super().__init__(inputs)
        self.events: list[tuple[str, object]] = []

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        super().set_digital_output(pin, value, port=port)
        self.events.append((f"out{pin}", bool(value)))

    def sleep(self, seconds: float) -> None:
        self.events.append(("sleep", seconds))

    def outputs_written(self) -> list[tuple[str, object]]:
        return [event for event in self.events if event[0] != "sleep"]


PULSE_S, SETTLE_S = 0.25, 0.3
HIGH, LOW = (f"out{CLOSE_PIN}", True), (f"out{CLOSE_PIN}", False)
#: One flip: the pin low first (a flip is an EDGE, and an edge needs a low before it), the rising edge, the high
#: held for pulse_s, and low again.
PULSE = [LOW, ("sleep", PULSE_S), HIGH, ("sleep", PULSE_S), LOW]
WRITES = [LOW, HIGH, LOW]


class Person:
    """The person at the terminal: answers the driver's questions from a script, and keeps what was asked."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        if not self.answers:
            raise AssertionError(f"asked once more than scripted: {question!r}")
        return self.answers.pop(0)


def _toggle(io: _Timeline, *answers: str, **kw) -> JawIOGripper:
    """The owner's cell (2026-09-24): one tool output, every pulse on it flips the jaws, nothing wired back.

    ``answers`` is what the person types when asked where the jaws stand; Enter (open) by default, once per connect.
    """
    return JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S,
                        ask=Person(*(answers or ("",) * 4)), sleep=io.sleep, **kw)


class SingleToggleTests(unittest.TestCase):
    """One output, and each pulse flips the jaws: the pin level says nothing, so WHEN to pulse is the logic.

    Nothing is wired back, which is the owner's cell: the driver counts its own pulses from the person's answer.
    """

    def setUp(self) -> None:
        self.io = _Timeline()
        self.g = _toggle(self.io)
        self.g.connect()

    def test_a_close_from_open_is_one_pulse_that_leaves_the_pin_low_then_the_stroke(self) -> None:
        self.g.set_closed(True)
        self.assertEqual(self.io.events, [*PULSE, ("sleep", SETTLE_S)])   # the edge, then the jaws' travel time
        self.assertEqual(self.io.outputs_written(), WRITES)
        self.assertFalse(self.io.outputs[CLOSE_PIN])
        self.assertTrue(self.g.jaws_closed)

    def test_a_second_close_is_not_a_pulse(self) -> None:
        # A pulse on closed jaws OPENS them, so a repeated close must write nothing at all.
        self.g.set_closed(True)
        self.io.events.clear()
        self.g.set_closed(True)
        self.assertEqual(self.io.events, [])
        self.assertTrue(self.g.jaws_closed)

    def test_an_open_after_a_close_is_one_pulse(self) -> None:
        self.g.set_closed(True)
        self.io.events.clear()
        self.g.set_closed(False)
        # One pulse, then the jaws' travel time: nothing else says when they have opened, and a place must not back
        # out of jaws still opening.
        self.assertEqual(self.io.events, [*PULSE, ("sleep", SETTLE_S)])
        self.assertFalse(self.g.jaws_closed)

    def test_an_open_on_open_jaws_is_not_a_pulse(self) -> None:
        self.g.set_closed(False)
        self.assertEqual(self.io.events, [])
        self.assertFalse(self.g.jaws_closed)

    def test_a_width_is_refused(self) -> None:
        """A width read against closed_below_mm is how a grasp became an open on the owner's cell: a toggle takes none."""
        for width in (0.0, 80.0):
            with self.subTest(width=width), self.assertRaises(RobotError) as caught:
                self.g.set_width_mm(width)
            self.assertIn("set_closed", str(caught.exception))
        self.assertEqual(self.io.events, [])

    def test_it_says_it_toggles_without_a_sensor(self) -> None:
        from src.robot.core.gripper import TogglesWithoutSensor, toggle_without_sensor_of

        self.assertIsInstance(self.g, TogglesWithoutSensor)
        self.assertIs(toggle_without_sensor_of(self.g), self.g)
        solenoid = JawIOGripper(FakeIO(), close_output_pin=CLOSE_PIN, sleep=lambda _s: None)
        self.assertIsNone(toggle_without_sensor_of(solenoid), "a solenoid is told open, not asked")


class _ToggleJaws(_Timeline):
    """The owner's device, modelled: every rising edge on the close pin flips the jaws. ``lose`` drops that many of the
    next edges, as an e-stop mid-pulse or a cable would."""

    def __init__(self, *, closed: bool = False, lose: int = 0) -> None:
        super().__init__()
        self.closed = closed
        self.lose = lose

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        rising = pin == CLOSE_PIN and bool(value) and not self.outputs.get(pin, False)
        super().set_digital_output(pin, value, port=port)
        if rising and self.lose:
            self.lose -= 1
        elif rising:
            self.closed = not self.closed


class SingleToggleReconnectTests(unittest.TestCase):
    """The count is set at EVERY connect from a person's answer, never carried over from before a disconnect.

    The console keeps one gripper object across disconnect and connect, and a person may move the jaws in between.
    """

    def test_a_reconnect_asks_again_and_counts_from_the_answer(self) -> None:
        io = _ToggleJaws()
        person = Person("", "")
        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S, ask=person,
                         sleep=io.sleep)
        g.connect()
        g.set_closed(True)
        g.disconnect()
        self.assertTrue(io.closed)
        io.closed = False                                   # someone opened them from the pendant in between
        g.connect()                                         # ... and the person says so
        self.assertEqual(2, len(person.asked))
        self.assertFalse(g.jaws_closed)
        g.set_closed(False)
        self.assertFalse(io.closed, "an open after a reconnect closed jaws a person had said stood open")
        g.set_closed(True)
        self.assertTrue(io.closed)

    def test_a_second_connect_while_connected_asks_nothing(self) -> None:
        io = _ToggleJaws()
        person = Person("")
        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, ask=person, sleep=io.sleep)
        g.connect()
        g.set_closed(True)
        g.connect()                                         # idempotent: the count stands
        self.assertEqual(1, len(person.asked))
        self.assertTrue(g.jaws_closed)


class SingleTogglePulseTests(unittest.TestCase):
    """A flip is an edge: the pulse leaves the pin low whatever happens, and the count moves at the edge."""

    def test_an_interrupted_pulse_drops_the_pin_and_keeps_the_flip(self) -> None:
        io = _ToggleJaws()

        def sleep(seconds: float) -> None:
            io.events.append(("sleep", seconds))
            if io.outputs.get(CLOSE_PIN):
                raise KeyboardInterrupt                     # Ctrl-C while the pin is high

        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S,
                         ask=Person(""), sleep=sleep)
        g.connect()
        with self.assertRaises(KeyboardInterrupt):
            g.set_closed(True)
        self.assertFalse(io.outputs[CLOSE_PIN], "the pin was left high")
        self.assertTrue(io.closed)
        self.assertTrue(g.jaws_closed, "the edge flipped the jaws and the driver does not know it")

    def test_a_pin_left_high_still_gives_the_next_pulse_an_edge(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        io.outputs[CLOSE_PIN] = True                        # a low write that never arrived
        g.set_closed(True)
        self.assertTrue(io.closed)
        self.assertEqual(io.outputs_written(), WRITES)

    def test_a_pin_left_high_is_dropped_at_connect(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        io.outputs[CLOSE_PIN] = True                        # a low write that never arrived
        with self.assertLogs(g.logger, level="WARNING"):
            g.connect()
        self.assertFalse(io.outputs[CLOSE_PIN])
        self.assertFalse(io.closed, "a falling edge must not flip a device that flips on the rising one")

    def test_back_to_back_commands_leave_a_low_between_the_pulses(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_closed(True)
        g.set_closed(False)
        self.assertFalse(io.closed)
        first_low_after_high = io.events.index(LOW, io.events.index(HIGH))
        second_high = io.events.index(HIGH, first_low_after_high)
        self.assertIn(("sleep", PULSE_S), io.events[first_low_after_high:second_high])

    def test_a_lost_pulse_inverts_the_count(self) -> None:
        """The risk the owner takes with no sensor, measured: nothing notices a lost edge until a person looks."""
        io = _ToggleJaws(lose=1)
        g = _toggle(io)
        g.connect()
        g.set_closed(True)                                  # the edge is lost: the jaws stay open
        self.assertFalse(io.closed)
        g.set_closed(False)                                 # the driver believes closed, pulses, and CLOSES them
        self.assertTrue(io.closed)


class _ReplyLost(_ToggleJaws):
    """A high write the controller applies and whose reply is lost: the edge happened, and the call raised."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.lose_reply = True

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        super().set_digital_output(pin, value, port=port)
        if pin == CLOSE_PIN and bool(value) and self.lose_reply:
            self.lose_reply = False
            raise RobotConnectionError("the reply to the high write was lost")


class SingleToggleUnknowableEdgeTests(unittest.TestCase):
    """A high write that raised may have flipped the jaws: the driver stops pulsing until a person has said."""

    def test_a_failed_high_write_drops_the_pin_and_blocks_the_next_pulse(self) -> None:
        io = _ReplyLost()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_closed(True)
        self.assertTrue(io.closed)                          # the edge reached the jaws
        self.assertFalse(io.outputs[CLOSE_PIN], "the pin was left high")
        writes = len(io.outputs_written())
        with self.assertRaises(RobotError) as caught:
            g.set_closed(True)                              # a retry would flip the jaws back open
        self.assertIn("unknowable", str(caught.exception))
        self.assertEqual(len(io.outputs_written()), writes, "a pulse was sent on an unknowable count")
        self.assertTrue(io.closed)

    def test_the_next_pick_asks_and_the_answer_lifts_the_block(self) -> None:
        io = _ReplyLost()
        person = Person("", "closed", "p")
        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S, ask=person,
                         sleep=io.sleep)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_closed(True)
        self.assertEqual("", g.jaws_open_for_a_pick())     # the person looked: closed, and chose one pulse
        self.assertIn("failed on its high write", person.asked[1])
        self.assertFalse(io.closed)
        self.assertFalse(g.jaws_closed)

    def test_an_unknowable_edge_is_said_at_disconnect(self) -> None:
        io = _ReplyLost()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_closed(True)
        with self.assertLogs(g.logger, level="INFO") as logs:
            g.disconnect()
        self.assertTrue(any("UNKNOWN" in line for line in logs.output), "the disconnect said 'open'")


class _NeverHigh(_ToggleJaws):
    """An output the controller accepts and never reports HIGH: a wrong bank, a reserved pin, a write that never reached
    the pin. The device on it sees no edge either, so its jaws stay where they stood."""

    def __init__(self, **kw) -> None:
        super().__init__(lose=1_000, **kw)

    def get_digital_output(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        return False


class _Lagging(_ToggleJaws):
    """The RTDE receive stream: after each write, the first ``lag`` reads still report the level before it (measured on
    URSim 5.26.0, ``drivers/ur/io_bench.set_output``)."""

    def __init__(self, *, lag: int, **kw) -> None:
        super().__init__(**kw)
        self.lag = lag
        self.stale: dict[int, tuple[bool, int]] = {}

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        before = bool(self.outputs.get(pin, False))
        super().set_digital_output(pin, value, port=port)
        self.stale[pin] = (before, self.lag)

    def get_digital_output(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        before, left = self.stale.get(pin, (False, 0))
        if left > 0:
            self.stale[pin] = (before, left - 1)
            return before
        return bool(self.outputs.get(pin, False))


def _after_the_high(io: _Timeline) -> list[tuple[str, object]]:
    """What happened between the rising edge and the next write, the low."""
    high = io.events.index(HIGH)
    low = io.events.index(LOW, high)
    return io.events[high + 1:low]


class SingleTogglePulseIsReadBackTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D2). The count flipped as soon as the HIGH write returned, and a write that returns says
    the command was accepted, not that the pin moved (``drivers/ur/io_bench.set_output`` reads back for exactly this
    reason). The pin is read back until it reads HIGH, about a pulse and two 8 ms CB3 cycles and never under 50 ms,
    and the count flips only then; a pin that never reads HIGH leaves the count unknowable and the next pick asks."""

    def test_an_output_that_never_reads_high_is_refused_and_leaves_the_count_unknowable(self) -> None:
        io = _NeverHigh()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotError) as caught:
            g.set_closed(True)
        message = str(caught.exception)
        for part in (f"tool output {CLOSE_PIN}", "never read HIGH", "nobody can say whether the jaws flipped"):
            self.assertIn(part, message)
        self.assertFalse(g.jaws_closed, "the count flipped on a pulse the pin never showed")
        self.assertEqual(LOW, io.outputs_written()[-1], "the low was not attempted")
        writes = len(io.outputs_written())
        with self.assertRaises(RobotError):
            g.set_closed(True)                              # nobody can say which way a pulse would move them now
        self.assertEqual(writes, len(io.outputs_written()), "a pulse went out on an unknowable count")

    def test_the_next_pick_asks_after_an_output_that_never_read_high(self) -> None:
        io = _NeverHigh()
        person = Person("", "open")
        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S, ask=person,
                         sleep=io.sleep)
        g.connect()
        with self.assertRaises(RobotError):
            g.set_closed(True)
        self.assertEqual("", g.jaws_open_for_a_pick())
        self.assertEqual(2, len(person.asked))
        self.assertFalse(g.jaws_closed)

    def test_the_read_back_waits_about_one_pulse_and_two_controller_cycles(self) -> None:
        for pulse_s, bound_s in ((PULSE_S, PULSE_S + 0.016), (0.0, 0.05)):
            with self.subTest(pulse_s=pulse_s):
                io = _NeverHigh()
                g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=pulse_s,
                                 ask=Person(""), sleep=io.sleep)
                g.connect()
                with self.assertRaises(RobotError):
                    g.set_closed(True)
                waited = sum(float(s) for kind, s in _after_the_high(io) if kind == "sleep")  # type: ignore[arg-type]
                self.assertGreaterEqual(waited, bound_s - 1e-9, "gave up before the controller could answer")
                self.assertLess(waited, bound_s + 0.0081, "held the pin high long past the bound")

    def test_a_read_back_that_lags_two_cycles_still_counts_the_flip(self) -> None:
        """⭐ THE CONTROL: the stream reports the old level for a cycle or two, and the pulse is still one flip."""
        io = _Lagging(lag=2)
        g = _toggle(io)
        g.connect()
        io.events.clear()
        g.set_closed(True)
        self.assertTrue(io.closed)
        self.assertTrue(g.jaws_closed)
        between = _after_the_high(io)
        self.assertEqual(("sleep", PULSE_S), between[-1], "the high was not held pulse_s once it read HIGH")
        self.assertEqual(2, sum(1 for kind, _s in between[:-1] if kind == "sleep"))

    def test_a_pin_that_reads_high_at_once_costs_no_wait(self) -> None:
        """⭐ THE CONTROL: the timeline of one flip is unchanged where the read-back agrees at once."""
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        io.events.clear()
        g.set_closed(True)
        self.assertEqual(io.events, [*PULSE, ("sleep", SETTLE_S)])


class _CoilNeverHigh(FakeIO):
    """A solenoid output the controller accepts and never reports HIGH."""

    def get_digital_output(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        return False


class SolenoidsAreReadBackTooTests(unittest.TestCase):
    """The same read-back on a solenoid: a coil or a level that never reads back is refused rather than believed."""

    def test_a_double_solenoid_coil_that_never_reads_high_is_refused_and_dropped(self) -> None:
        io = _CoilNeverHigh()
        g = JawIOGripper(io, actuation="double_solenoid", close_output_pin=CLOSE_PIN, open_output_pin=OPEN_PIN,
                         pulse_s=0.1, sleep=lambda _s: None)
        g.connect()
        with self.assertRaises(RobotError) as caught:
            g.set_closed(True)
        self.assertIn(f"tool output {CLOSE_PIN}", str(caught.exception))
        self.assertIn("never read HIGH", str(caught.exception))
        self.assertEqual((CLOSE_PIN, False), io.writes[-1][:2], "the coil was left energised")

    def test_a_single_solenoid_level_that_never_reads_back_is_refused(self) -> None:
        io = _CoilNeverHigh()
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, sleep=lambda _s: None)
        g.connect()
        with self.assertRaises(RobotError) as caught:
            g.set_closed(True)
        self.assertIn("never read HIGH", str(caught.exception))
        g.set_closed(False)                                 # LOW reads back LOW: an open is not refused

    def test_a_solenoid_whose_pins_read_back_is_unchanged(self) -> None:
        """⭐ THE CONTROL."""
        io = FakeIO()
        g = JawIOGripper(io, actuation="double_solenoid", close_output_pin=CLOSE_PIN, open_output_pin=OPEN_PIN,
                         sleep=lambda _s: None)
        g.connect()
        g.set_closed(True)
        self.assertEqual([(OPEN_PIN, False), (CLOSE_PIN, True), (CLOSE_PIN, False)],
                         [(pin, value) for pin, value, _port in io.writes])
        self.assertTrue(g.jaws_closed)


class TheDecisionIsInTheLogTests(unittest.TestCase):
    """Whether a command pulsed is said at INFO, the level the cell's log keeps (owner cell, 2026-09-23).

    Both lines were DEBUG, and the robot logger keeps INFO, so jaw_io_gripper.log could not answer the one question
    a toggle cell asks of it when the jaws do the wrong thing: did that command pulse or not.
    """

    def test_a_pulse_and_a_skipped_pulse_are_both_in_the_log(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        with self.assertLogs(g.logger, level="INFO") as logs:
            g.set_closed(True)
            g.set_closed(True)
        text = "\n".join(logs.output)
        self.assertIn("actuating jaws CLOSED via single_toggle", text)
        self.assertIn("jaws already stand CLOSED: no pulse", text)

    def test_the_persons_answer_is_in_the_log(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        with self.assertLogs(g.logger, level="WARNING") as logs:
            g.connect()
        self.assertTrue(any("a person said the jaws on tool output 4 stand OPEN" in line for line in logs.output))


class _Stroke(FakeIO):
    """Jaws that close ``travel_s`` of the driver's own sleeps after the close pin goes high, off both stops meanwhile.

    ``part`` stops them short of the closed stop, as a held part does, and holds the part pin high all along, as a
    part-present sensor sees a part between open jaws too.
    """

    def __init__(self, *, travel_s: float, part: bool = False) -> None:
        super().__init__()
        self.clock = 0.0
        self.travel_s = travel_s
        self.part = part
        self.closing_since: float | None = None

    def sleep(self, seconds: float) -> None:
        self.clock += seconds

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        super().set_digital_output(pin, value, port=port)
        if pin == CLOSE_PIN and bool(value) and self.closing_since is None:
            self.closing_since = self.clock

    def get_digital_input(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        if pin == PART_SW:
            return self.part
        if self.closing_since is None:
            return pin == OPEN_SW
        arrived = self.clock - self.closing_since >= self.travel_s
        return pin == CLOSED_SW and arrived and not self.part


class CloseSettlesBeforeAVerdictTests(unittest.TestCase):
    """A reading that also occurs mid-stroke is not a verdict until the jaws have had their travel time (2026-09-23).

    Off both stops is what a reed pair reads while the jaws travel, a lone closed switch off its stop repeats the
    command, and a part-present sensor sees a part between open jaws; each was taken as HOLDING on the first read after
    the close, so the grasp returned and the arm lifted before the jaws had moved. Only the closed stop, reached, proves
    anything at once.
    """

    TRAVEL_S, SETTLE_S = 0.2, 0.3

    def _grasp(self, io: _Stroke, **pins: int) -> JawIOGripper:
        g = JawIOGripper(io, close_output_pin=CLOSE_PIN, close_settle_s=self.SETTLE_S, close_timeout_s=5.0,
                         sleep=io.sleep, **pins)
        g._connected = True
        g.set_width_mm(0.0)
        return g

    def test_a_reading_from_mid_stroke_is_not_taken_as_a_grasp(self) -> None:
        wirings = {
            "reed pair": {"closed_confirm_input_pin": CLOSED_SW, "open_confirm_input_pin": OPEN_SW},
            "closed switch alone": {"closed_confirm_input_pin": CLOSED_SW},
        }
        for name, pins in wirings.items():
            with self.subTest(wiring=name):
                io = _Stroke(travel_s=self.TRAVEL_S)
                g = self._grasp(io, **pins)
                self.assertGreaterEqual(io.clock, self.TRAVEL_S, "the grasp returned before the jaws arrived")
                self.assertIs(g.hold_evidence(), HoldEvidence.EMPTY)

    def test_a_part_sensor_alone_still_waits_the_travel_time(self) -> None:
        io = _Stroke(travel_s=self.TRAVEL_S, part=True)
        g = self._grasp(io, part_present_input_pin=PART_SW)
        self.assertGreaterEqual(io.clock, self.SETTLE_S, "the part pin read high between open jaws and ended the wait")
        self.assertIs(g.hold_evidence(), HoldEvidence.HELD)

    def test_a_part_held_short_of_the_stop_is_still_held(self) -> None:
        io = _Stroke(travel_s=self.TRAVEL_S, part=True)
        g = self._grasp(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW)
        self.assertGreaterEqual(io.clock, self.SETTLE_S)
        self.assertIs(g.hold_evidence(), HoldEvidence.HELD)

    def test_the_closed_stop_reached_is_taken_at_once(self) -> None:
        """⭐ THE CONTROL: a proven stop costs no settle time, so the wait is added only where nothing proves anything."""
        io = _Stroke(travel_s=0.0)
        g = self._grasp(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW)
        self.assertEqual(io.clock, 0.0)
        self.assertIs(g.hold_evidence(), HoldEvidence.EMPTY)


class TheWidthsMeanWhatTheyAreForTests(unittest.TestCase):
    """A jaw_io cell whose widths make a release close, or leave no width that closes, does not load (2026-09-23).

    jaw_io reads every commanded width as open or closed: closed at or below ``closed_below_mm``. A release and a
    pick's pre-open command ``max_width_mm``, and a grasp commands at least ``min_width_mm``. So with
    ``closed_below_mm`` at or above ``max_width_mm`` every release closes the jaws, and below ``min_width_mm`` no grasp
    can close them. The first is what 84, the 2F-85's number, does on a Hand-E's 49.99: the pick closes at its
    start, the symptom the owner's toggle cell reported on 2026-09-23.
    """

    @staticmethod
    def _cell(vendor: str = "jaw_io", **jaw: object) -> dict[str, object]:
        return {"vendor": vendor, "max_width_mm": 49.99, "min_width_mm": 5.0, "jaw_io": jaw}

    def test_a_threshold_at_or_above_the_opening_is_refused(self) -> None:
        for closed_below in (49.99, 84.0):
            with self.subTest(closed_below_mm=closed_below):
                with self.assertRaises(ValidationError) as caught:
                    GripperConfig.model_validate(self._cell(closed_below_mm=closed_below))
                for part in ("closed_below_mm", "max_width_mm", "49.99", "release"):
                    self.assertIn(part, str(caught.exception))

    def test_a_threshold_below_the_narrowest_grasp_is_refused(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            GripperConfig.model_validate(self._cell(closed_below_mm=4.0))
        for part in ("closed_below_mm", "min_width_mm", "never close"):
            self.assertIn(part, str(caught.exception))

    def test_the_owners_numbers_load(self) -> None:
        """⭐ THE CONTROL: 49.0 between 5.0 and 49.99, and the schema default of 5.0 at the Hand-E's floor."""
        for closed_below in (49.0, 5.0):
            with self.subTest(closed_below_mm=closed_below):
                GripperConfig.model_validate(self._cell(closed_below_mm=closed_below))

    def test_the_block_is_inert_under_another_vendor(self) -> None:
        GripperConfig.model_validate(self._cell(vendor="robotiq", closed_below_mm=84.0))

    def test_a_toggle_takes_no_width_so_the_band_does_not_apply(self) -> None:
        """Owner's decision (2026-09-24): a single_toggle refuses a width, so no threshold can turn a verb round."""
        GripperConfig.model_validate(self._cell(actuation="single_toggle", closed_below_mm=84.0))


class SingleToggleRefusalTests(unittest.TestCase):
    """What a toggle cannot do, refused by the schema and again by the constructor reachable without it."""

    def _schema_refusal(self, **fields: object) -> str:
        with self.assertRaises(ValidationError) as caught:
            JawIOGripperConfig.model_validate({"actuation": "single_toggle", **fields})
        (error,) = caught.exception.errors()
        self.assertEqual(error["type"], "value_error")      # the validator's sentence, not the Literal's
        return str(error["msg"])

    def test_the_owners_cell_validates(self) -> None:
        cfg = JawIOGripperConfig.model_validate(
            {"actuation": "single_toggle", "close_output_pin": 0, "io_port": "tool"})
        self.assertEqual(cfg.actuation, "single_toggle")
        self.assertIsNone(cfg.confirm_open_at_start, "None asks for a toggle")

    def test_the_schema_refuses_an_open_output_pin(self) -> None:
        # The one pin opens the jaws too, so a second one would never be driven.
        message = self._schema_refusal(open_output_pin=1)
        for part in ("single_toggle", "open_output_pin", "close_output_pin"):
            self.assertIn(part, message)

    def test_the_schema_refuses_asserting_open_on_connect(self) -> None:
        # A toggle cannot assert a state, only flip one: a pulse on open jaws closes them.
        message = self._schema_refusal(open_on_connect_without_feedback=True)
        for part in ("single_toggle", "open_on_connect_without_feedback"):
            self.assertIn(part, message)

    def test_the_constructor_refuses_an_open_output_pin(self) -> None:
        with self.assertRaises(ValueError) as caught:
            JawIOGripper(FakeIO(), actuation="single_toggle", close_output_pin=CLOSE_PIN,
                         open_output_pin=OPEN_PIN)
        self.assertIn("open_output_pin is never driven", str(caught.exception))

    def test_the_constructor_refuses_asserting_open_on_connect(self) -> None:
        with self.assertRaises(ValueError) as caught:
            JawIOGripper(FakeIO(), actuation="single_toggle", close_output_pin=CLOSE_PIN,
                         open_on_connect_without_feedback=True)
        self.assertIn("open_on_connect_without_feedback", str(caught.exception))

    def test_the_schema_refuses_every_feedback_input(self) -> None:
        # The toggle reads nothing back: an input is a wire nobody reads, and a cell that believes it is sensed is not.
        for name in ("part_present_input_pin", "closed_confirm_input_pin", "open_confirm_input_pin"):
            with self.subTest(input=name):
                message = self._schema_refusal(**{name: 1})
                for part in ("single_toggle", name, "reads no sensor"):
                    self.assertIn(part, message)

    def test_the_constructor_refuses_every_feedback_input(self) -> None:
        for name in ("part_present_input_pin", "closed_confirm_input_pin", "open_confirm_input_pin"):
            with self.subTest(input=name), self.assertRaises(ValueError) as caught:
                JawIOGripper(FakeIO(), actuation="single_toggle", close_output_pin=CLOSE_PIN, **{name: 1})
            self.assertIn("reads no sensor", str(caught.exception))

    def test_the_question_at_connect_cannot_be_switched_off(self) -> None:
        """Owner's decision (2026-09-24): nothing else can tell a toggle where its jaws start."""
        message = self._schema_refusal(confirm_open_at_start=False)
        for part in ("single_toggle", "confirm_open_at_start", "always asks"):
            self.assertIn(part, message)
        with self.assertRaises(ValueError):
            JawIOGripper(FakeIO(), actuation="single_toggle", close_output_pin=CLOSE_PIN, confirm_open_at_start=False)
        JawIOGripperConfig.model_validate({"actuation": "single_toggle", "confirm_open_at_start": True})
        JawIOGripperConfig.model_validate({"actuation": "single_solenoid", "confirm_open_at_start": False})

    def test_an_unknown_actuation_names_all_three(self) -> None:
        with self.assertRaises(ValueError) as caught:
            JawIOGripper(FakeIO(), actuation="hydraulic")
        for known in ("single_solenoid", "double_solenoid", "single_toggle"):
            self.assertIn(known, str(caught.exception))


class TheToolBankHasTwoPinsEachWayTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D3). The pins were 0 to 7 whatever the bank, and the UR tool connector has digital outputs
    0 and 1 and inputs 0 and 1 only (``drivers/ur/connection.py``): a tool pin above 1 loaded, and every write to it
    went to a global id that is some other bank's pin, or none."""

    def _refusal(self, model: type, **fields: object) -> str:
        with self.assertRaises(ValidationError) as caught:
            model.model_validate(fields)  # type: ignore[attr-defined]
        (error,) = caught.exception.errors()
        self.assertEqual(error["type"], "value_error")      # the validator's sentence, not a bound's
        return str(error["msg"])

    def test_a_jaw_output_above_1_on_the_tool_bank_is_refused(self) -> None:
        cases = {
            "close_output_pin": {"close_output_pin": 2},
            "open_output_pin": {"actuation": "double_solenoid", "close_output_pin": 0, "open_output_pin": 2},
        }
        for name, fields in cases.items():
            with self.subTest(pin=name):
                message = self._refusal(JawIOGripperConfig, io_port="tool", **fields)
                for part in (name, "tool", "0 and 1", "standard"):
                    self.assertIn(part, message)

    def test_a_jaw_input_above_1_on_the_tool_bank_is_refused(self) -> None:
        for name in ("part_present_input_pin", "closed_confirm_input_pin", "open_confirm_input_pin"):
            with self.subTest(pin=name):
                message = self._refusal(JawIOGripperConfig, io_port="tool", **{name: 3})
                for part in (name, "tool", "0 and 1"):
                    self.assertIn(part, message)

    def test_the_default_bank_is_the_tool_bank(self) -> None:
        self.assertIn("close_output_pin", self._refusal(JawIOGripperConfig, close_output_pin=3))

    def test_a_vacuum_on_the_tool_bank_is_held_to_the_same_pins(self) -> None:
        for name in ("vacuum_output_pin", "blow_off_output_pin", "vacuum_ok_input_pin"):
            with self.subTest(pin=name):
                message = self._refusal(VacuumGripperConfig, io_port="tool", **{name: 2})
                for part in (name, "tool", "0 and 1"):
                    self.assertIn(part, message)

    def test_the_same_pins_in_the_control_box_load(self) -> None:
        """⭐ THE CONTROL: standard and configurable have eight of each, and the tool bank's own two load."""
        for port in ("standard", "configurable"):
            with self.subTest(io_port=port):
                JawIOGripperConfig.model_validate({"io_port": port, "actuation": "double_solenoid",
                                                   "close_output_pin": 6, "open_output_pin": 7,
                                                   "closed_confirm_input_pin": 5})
                VacuumGripperConfig.model_validate({"io_port": port, "vacuum_output_pin": 7, "vacuum_ok_input_pin": 5})
        JawIOGripperConfig.model_validate({"io_port": "tool", "actuation": "double_solenoid", "close_output_pin": 0,
                                           "open_output_pin": 1, "closed_confirm_input_pin": 0,
                                           "open_confirm_input_pin": 1})
        VacuumGripperConfig.model_validate({"io_port": "tool", "vacuum_output_pin": 1, "blow_off_output_pin": 0,
                                            "vacuum_ok_input_pin": 1})


class APulseLongEnoughToRegisterTests(unittest.TestCase):
    """⛔ Review of 2026-09-24 (D3). ``pulse_s`` was only above zero, so a pulse of a millisecond loaded: shorter than one
    8 ms CB3 controller cycle, it may never reach the pin, and a toggle then never flips while the count does. The load
    refuses a pulse under 0.05 s, several controller cycles, wherever the pulse is what moves the jaws."""

    def test_a_pulse_under_50_ms_is_refused_where_it_moves_the_jaws(self) -> None:
        for actuation, extra in (("single_toggle", {}), ("double_solenoid", {"open_output_pin": 1})):
            for pulse_s in (0.001, 0.049):
                with self.subTest(actuation=actuation, pulse_s=pulse_s):
                    with self.assertRaises(ValidationError) as caught:
                        JawIOGripperConfig.model_validate({"actuation": actuation, "pulse_s": pulse_s, **extra})
                    (error,) = caught.exception.errors()
                    for part in ("pulse_s", "0.05", "controller cycle", actuation):
                        self.assertIn(part, str(error["msg"]))

    def test_50_ms_loads_and_a_single_solenoid_is_not_held_to_it(self) -> None:
        """⭐ THE CONTROL: the floor itself, and a single solenoid, which holds a level and pulses nothing."""
        JawIOGripperConfig.model_validate({"actuation": "single_toggle", "pulse_s": 0.05})
        JawIOGripperConfig.model_validate({"actuation": "double_solenoid", "open_output_pin": 1, "pulse_s": 0.05})
        JawIOGripperConfig.model_validate({"actuation": "single_solenoid", "pulse_s": 0.001})

    def test_the_driver_built_directly_keeps_any_pulse(self) -> None:
        """The floor is the load's: a driver built in a test with a zero pulse still pulses."""
        io = _ToggleJaws()
        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=0.0, ask=Person(""),
                         sleep=io.sleep)
        g.connect()
        g.set_closed(True)
        self.assertTrue(io.closed)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
