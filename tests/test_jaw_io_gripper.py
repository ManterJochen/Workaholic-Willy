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

from src.config.schema.robot.robot_schema import GripperConfig, JawIOGripperConfig
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


PULSE_S = 0.25
HIGH, LOW = (f"out{CLOSE_PIN}", True), (f"out{CLOSE_PIN}", False)
#: One flip: the pin low first (a flip is an EDGE, and an edge needs a low before it), the rising edge, the high
#: held for pulse_s, and low again.
PULSE = [LOW, ("sleep", PULSE_S), HIGH, ("sleep", PULSE_S), LOW]
WRITES = [LOW, HIGH, LOW]


def _toggle(io: _Timeline, **kw) -> JawIOGripper:
    """The owner's cell (2026-09-23): one tool output, and every pulse on it flips the jaws."""
    return JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S,
                        sleep=io.sleep, **kw)


class SingleToggleTests(unittest.TestCase):
    """One output, and each pulse flips the jaws: the pin level says nothing, so WHEN to pulse is the logic.

    Nothing is wired back here, which is the owner's cell: the driver can only trust what it last commanded.
    """

    def setUp(self) -> None:
        self.io = _Timeline()
        self.g = _toggle(self.io)
        self.g.connect()

    def test_a_close_from_open_is_one_pulse_that_leaves_the_pin_low(self) -> None:
        self.g.set_width_mm(0.0)
        self.assertEqual(self.io.events[:5], PULSE)        # low, pulse_s, high, pulse_s, low: in that order
        self.assertEqual(self.io.outputs_written(), WRITES)
        self.assertFalse(self.io.outputs[CLOSE_PIN])
        self.assertTrue(self.g.jaws_closed)

    def test_a_second_close_is_not_a_pulse(self) -> None:
        # A pulse on closed jaws OPENS them, so a repeated close must write nothing at all.
        self.g.set_width_mm(0.0)
        self.io.events.clear()
        self.g.set_width_mm(0.0)
        self.assertEqual(self.io.outputs_written(), [])
        self.assertTrue(self.g.jaws_closed)

    def test_an_open_after_a_close_is_one_pulse(self) -> None:
        self.g.set_width_mm(0.0)
        self.io.events.clear()
        self.g.set_width_mm(80.0)
        # One pulse, then the jaws' travel time (close_settle_s, 0.3 by default): with no open switch nothing
        # else says when they have opened, and a place must not back out of jaws still opening.
        self.assertEqual(self.io.events, [*PULSE, ("sleep", 0.3)])
        self.assertFalse(self.g.jaws_closed)

    def test_an_open_on_open_jaws_is_not_a_pulse(self) -> None:
        self.g.set_width_mm(80.0)
        self.assertEqual(self.io.events, [])
        self.assertFalse(self.g.jaws_closed)


class SingleToggleSensorTests(unittest.TestCase):
    """Where an end-stop switch is wired it says how the jaws stand, and a lost pulse stops mattering."""

    def test_the_open_switch_wins_over_what_was_last_commanded(self) -> None:
        io = _Timeline({OPEN_SW: True})
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        # A pulse the driver lost track of closed the jaws: it still believes them open, and the
        # switch says they are off the open stop. Pulsing on the belief would OPEN them.
        io.inputs = {OPEN_SW: False}
        g.set_width_mm(0.0)
        self.assertEqual(io.outputs_written(), [])
        self.assertTrue(g.jaws_closed)
        g.set_width_mm(80.0)
        self.assertEqual(io.outputs_written(), WRITES)

    def test_a_lost_close_pulse_does_not_turn_the_next_open_into_a_close(self) -> None:
        io = _Timeline({OPEN_SW: True})
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        g.set_width_mm(0.0)                                 # pulsed, and the jaws never moved
        self.assertEqual(io.outputs_written(), WRITES)
        io.events.clear()
        g.set_width_mm(80.0)                                # the switch still reads open
        self.assertEqual(io.outputs_written(), [])

    def test_the_reed_pair_wins_over_the_belief_too(self) -> None:
        io = _Timeline({OPEN_SW: True})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        io.inputs = {CLOSED_SW: True}                       # closed on nothing; the driver believes open
        g.set_width_mm(0.0)
        self.assertEqual(io.outputs_written(), [])
        g.set_width_mm(80.0)
        self.assertEqual(io.outputs_written(), WRITES)

    def test_a_part_sensor_does_not_stop_a_close(self) -> None:
        # A part-present sensor reads a part between OPEN jaws as well as between closed ones, and
        # between open jaws is exactly where every close is commanded. It says nothing about how the
        # jaws stand, so it must not suppress the pulse that closes them.
        io = _Timeline()
        g = _toggle(io, part_present_input_pin=PART_SW)
        g.connect()
        io.inputs = {PART_SW: True}
        g.set_width_mm(0.0)
        self.assertEqual(io.outputs_written(), WRITES)

    def test_contradictory_switches_refuse_to_pulse(self) -> None:
        io = _Timeline({OPEN_SW: True})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        io.inputs = {CLOSED_SW: True, OPEN_SW: True}
        for width in (0.0, 80.0):
            with self.subTest(width=width):
                with self.assertRaises(RobotError) as caught:
                    g.set_width_mm(width)
                # Not a dropped link: the controller answered, and what it said is impossible.
                self.assertNotIsInstance(caught.exception, RobotConnectionError)
                self.assertIn("switch", str(caught.exception))
        self.assertEqual(io.outputs_written(), [])
        self.assertFalse(g.jaws_closed)                     # a refusal does not move the belief


class SingleToggleConnectTests(unittest.TestCase):
    """connect() through a toggle: it pulses only where both switches prove the jaws closed on nothing."""

    def test_without_feedback_connect_never_touches_the_output(self) -> None:
        io = _Timeline()
        g = _toggle(io)
        g.connect()
        self.assertEqual(io.events, [])
        self.assertFalse(g.jaws_closed)                     # it starts from "open", which a person makes true

    def test_jaws_the_switches_prove_closed_on_nothing_are_opened_with_one_pulse(self) -> None:
        # Off the open stop AND on the closed stop: the jaws met, so nothing can be dropped. The only case a
        # toggle pulses at connect; a lone closed switch is refused (SingleToggleRefusalTests).
        io = _ToggleJaws(closed=True, open_switch=True, closed_switch=True)
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        self.assertEqual(io.outputs_written(), WRITES)
        self.assertFalse(io.closed)
        self.assertFalse(g.jaws_closed)

    def test_a_connect_pulse_the_open_stop_never_answers_leaves_them_closed(self) -> None:
        """⭐ THE CONTROL: the same pulse on jaws that do not move, and the switch, not the command, is believed."""
        io = _Timeline({CLOSED_SW: True})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        self.assertEqual(io.outputs_written(), WRITES)
        self.assertTrue(g.jaws_closed)

    def test_off_the_open_stop_with_no_proof_of_empty_is_refused(self) -> None:
        """An open switch that does not read open is a held part or a switch that is not there, and both refuse.

        Held and warned about until 2026-09-23. An unwired input reads low, which is exactly "off the open stop", so
        the driver then believed every pair of jaws closed: each close was swallowed and each open pulsed, and every
        report said EXECUTED (the owner's cell, diagnosed that day).
        """
        io = _Timeline({OPEN_SW: False})
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        with self.assertRaises(RobotError) as caught:
            g.connect()
        self.assertNotIsInstance(caught.exception, RobotConnectionError)
        for part in (f"input {OPEN_SW}", "not wired", "with a pulse"):
            self.assertIn(part, str(caught.exception))
        self.assertEqual(io.events, [])
        self.assertFalse(g.is_connected)

    def test_jaws_the_switch_reads_open_are_not_pulsed(self) -> None:
        io = _Timeline({OPEN_SW: True})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        self.assertEqual(io.events, [])

    def test_a_reed_pair_between_the_stops_is_refused(self) -> None:
        # A held part and a pair of switches nobody wired read the same: neither stop.
        io = _Timeline({CLOSED_SW: False, OPEN_SW: False})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        with self.assertRaises(RobotError):
            g.connect()
        self.assertEqual(io.events, [])
        self.assertFalse(g.is_connected)

    def test_contradictory_switches_at_connect_are_not_pulsed(self) -> None:
        io = _Timeline({CLOSED_SW: True, OPEN_SW: True})
        g = _toggle(io, closed_confirm_input_pin=CLOSED_SW, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        self.assertEqual(io.events, [])


class _ToggleJaws(_Timeline):
    """The owner's device, modelled: every rising edge on the close pin flips the jaws, and an open switch, where
    one is wired, reads them. ``lose`` drops that many of the next edges, as an e-stop mid-pulse or a cable would."""

    def __init__(self, *, closed: bool = False, open_switch: bool = False, closed_switch: bool = False,
                 lose: int = 0) -> None:
        super().__init__()
        self.closed = closed
        self.open_switch = open_switch
        self.closed_switch = closed_switch                  # reads the closed stop: jaws closed on nothing
        self.lose = lose

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        rising = pin == CLOSE_PIN and bool(value) and not self.outputs.get(pin, False)
        super().set_digital_output(pin, value, port=port)
        if rising and self.lose:
            self.lose -= 1
        elif rising:
            self.closed = not self.closed

    def get_digital_input(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        if self.open_switch and pin == OPEN_SW:
            return not self.closed
        if self.closed_switch and pin == CLOSED_SW:
            return self.closed
        return super().get_digital_input(pin, port=port)


class SingleToggleReconnectTests(unittest.TestCase):
    """The belief is set at EVERY connect, never carried over from before a disconnect (review, 2026-09-23).

    The console keeps one gripper object across disconnect and connect, and a person may move the jaws in
    between. Carried over, the belief of a cell that disconnected closed would turn the next "open" into a close
    after the person had stood the jaws open, as the desk row tells them to.
    """

    def test_a_reconnect_takes_the_jaws_to_stand_open_again(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        g.disconnect()
        self.assertTrue(io.closed)
        io.closed = False                                   # the person stands them open, as the desk says
        g.connect()
        self.assertFalse(g.jaws_closed)
        g.set_width_mm(80.0)
        self.assertFalse(io.closed, "an open after a reconnect closed jaws a person had stood open")
        g.set_width_mm(0.0)
        self.assertTrue(io.closed)

    def test_the_old_carried_belief_is_the_defect_this_fixes(self) -> None:
        """⭐ THE CONTROL: with the belief carried over, the same sequence closes the jaws on an open."""
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        g.disconnect()
        io.closed = False
        g._connected = True                                 # a connect that keeps the old belief
        g.set_width_mm(80.0)
        self.assertTrue(io.closed)

    def test_an_open_switch_sets_the_belief_at_a_reconnect(self) -> None:
        io = _ToggleJaws(open_switch=True)
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        g.set_width_mm(0.0)
        g.disconnect()
        io.closed = False
        g.connect()
        self.assertFalse(g.jaws_closed)

    def test_a_part_at_connect_does_not_set_the_belief(self) -> None:
        # A part-present sensor reads a part between OPEN jaws too, so at connect it says nothing about the jaws.
        io = _ToggleJaws()
        io.inputs = {PART_SW: True}
        g = _toggle(io, part_present_input_pin=PART_SW)
        g.connect()
        self.assertEqual(io.events, [])
        self.assertFalse(g.jaws_closed)
        g.set_width_mm(0.0)
        self.assertTrue(io.closed)


class SingleTogglePulseTests(unittest.TestCase):
    """A flip is an edge: the pulse leaves the pin low whatever happens, and the belief moves at the edge."""

    def test_an_interrupted_pulse_drops_the_pin_and_keeps_the_flip(self) -> None:
        io = _ToggleJaws()

        def sleep(seconds: float) -> None:
            io.events.append(("sleep", seconds))
            if io.outputs.get(CLOSE_PIN):
                raise KeyboardInterrupt                     # Ctrl-C while the pin is high

        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S, sleep=sleep)
        g.connect()
        with self.assertRaises(KeyboardInterrupt):
            g.set_width_mm(0.0)
        self.assertFalse(io.outputs[CLOSE_PIN], "the pin was left high")
        self.assertTrue(io.closed)
        self.assertTrue(g.jaws_closed, "the edge flipped the jaws and the driver does not know it")

    def test_a_pin_left_high_still_gives_the_next_pulse_an_edge(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        io.outputs[CLOSE_PIN] = True                        # a low write that never arrived
        g.set_width_mm(0.0)
        self.assertTrue(io.closed)
        self.assertEqual(io.outputs_written(), WRITES)

    def test_back_to_back_commands_leave_a_low_between_the_pulses(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        g.set_width_mm(80.0)
        self.assertFalse(io.closed)
        first_low_after_high = io.events.index(LOW, io.events.index(HIGH))
        second_high = io.events.index(HIGH, first_low_after_high)
        self.assertIn(("sleep", PULSE_S), io.events[first_low_after_high:second_high])

    def test_a_lost_pulse_inverts_a_cell_with_no_open_switch(self) -> None:
        """What the schema doc warns about, measured: the risk the owner takes without an open switch."""
        io = _ToggleJaws(lose=1)
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)                                 # the edge is lost: the jaws stay open
        self.assertFalse(io.closed)
        g.set_width_mm(80.0)                                # the driver believes closed, pulses, and CLOSES them
        self.assertTrue(io.closed)

    def test_an_open_switch_makes_the_same_lost_pulse_harmless(self) -> None:
        io = _ToggleJaws(open_switch=True, lose=1)
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=0.01)
        g.connect()
        g.set_width_mm(0.0)                                 # lost: still open, and the switch says so
        g.set_width_mm(0.0)                                 # the next close is pulsed again
        self.assertTrue(io.closed)
        g.set_width_mm(80.0)
        self.assertFalse(io.closed)


class SingleToggleOpenWaitTests(unittest.TestCase):
    """With an open switch an open pulse waits for the open stop, so a close right after is not read mid-stroke."""

    def test_an_open_waits_for_the_open_stop(self) -> None:
        io = _ToggleJaws(open_switch=True)
        polls = {"n": 0}
        travel = 3

        def get(pin, *, port=DigitalIOPort.STANDARD) -> bool:
            if pin == OPEN_SW and not io.closed:
                polls["n"] += 1
                return polls["n"] > travel                  # the stop is reached after a few polls
            return _ToggleJaws.get_digital_input(io, pin, port=port)

        io.get_digital_input = get                          # type: ignore[method-assign]
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, close_timeout_s=5.0)
        io.closed = True
        g._connected = True
        g._closed = True
        g.set_width_mm(80.0)
        self.assertFalse(io.closed)
        self.assertGreaterEqual(io.events.count(("sleep", 0.02)), travel - 1)
        self.assertTrue(get(OPEN_SW))

    def test_no_open_switch_means_no_wait(self) -> None:
        io = _ToggleJaws(closed=True)
        g = _toggle(io)
        g.connect()
        g._closed = True
        g.set_width_mm(80.0)
        self.assertNotIn(("sleep", 0.02), io.events)


class SingleToggleReconcileTests(unittest.TestCase):
    """With an open switch the belief follows what it measured after each flip (second review, 2026-09-23).

    A lost edge left the belief where the command put it, while the switch had just measured otherwise, and a
    part-present sensor, which sees a part between OPEN jaws too, then reported a grasp on jaws that never closed,
    or a release of a part the jaws still held.
    """

    def _cell(self, io: "_ToggleJaws") -> JawIOGripper:
        io.inputs = {PART_SW: True}                         # a part between the jaws, open or closed
        return _toggle(io, open_confirm_input_pin=OPEN_SW, part_present_input_pin=PART_SW, close_timeout_s=0.01)

    def test_a_lost_close_edge_reports_no_grasp(self) -> None:
        io = _ToggleJaws(open_switch=True, lose=1)
        g = self._cell(io)
        g.connect()
        g.set_width_mm(0.0)                                 # the edge is lost: the jaws never left the open stop
        self.assertFalse(io.closed)
        self.assertFalse(g.jaws_closed)
        self.assertFalse(g.is_object_detected(), "a grasp reported on jaws that never closed")
        self.assertIs(g.hold_evidence(), HoldEvidence.UNMEASURED)

    def test_a_lost_open_edge_keeps_the_part_held(self) -> None:
        io = _ToggleJaws(open_switch=True)
        g = self._cell(io)
        g.connect()
        g.set_width_mm(0.0)
        self.assertTrue(io.closed)
        io.lose = 1
        g.set_width_mm(80.0)                                # the edge is lost: the jaws still hold the part
        self.assertTrue(io.closed)
        self.assertTrue(g.jaws_closed, "a release reported while the jaws still hold the part")
        self.assertIs(g.hold_evidence(), HoldEvidence.HELD)
        self.assertEqual(g.get_width_mm(), g.min_width_mm)

    def test_the_switch_that_answers_leaves_the_belief_alone(self) -> None:
        """⭐ THE CONTROL: with no edge lost, the same cell closes and opens as commanded."""
        io = _ToggleJaws(open_switch=True)
        g = self._cell(io)
        g.connect()
        g.set_width_mm(0.0)
        self.assertTrue(g.jaws_closed and io.closed)
        g.set_width_mm(80.0)
        self.assertFalse(g.jaws_closed or io.closed)


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
    """A high write that raised may have flipped the jaws: a cell with no open switch stops pulsing until a connect."""

    def test_a_failed_high_write_drops_the_pin_and_blocks_the_next_pulse(self) -> None:
        io = _ReplyLost()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(0.0)
        self.assertTrue(io.closed)                          # the edge reached the jaws
        self.assertFalse(io.outputs[CLOSE_PIN], "the pin was left high")
        writes = len(io.outputs_written())
        with self.assertRaises(RobotError) as caught:
            g.set_width_mm(0.0)                             # a retry would flip the jaws back open
        self.assertIn("unknowable", str(caught.exception))
        self.assertIn("with a pulse", str(caught.exception))
        self.assertNotRegex(str(caught.exception), r"(?<!never )by hand")
        self.assertEqual(len(io.outputs_written()), writes, "a pulse was sent on an unknowable belief")
        self.assertTrue(io.closed)

    def test_a_connect_lifts_the_block(self) -> None:
        io = _ReplyLost()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(0.0)
        io.closed = False                                   # a person stands the jaws open, as the refusal says
        g.connect()
        g.set_width_mm(0.0)
        self.assertTrue(io.closed)
        self.assertTrue(g.jaws_closed)


class SingleToggleReconnectWarningTests(unittest.TestCase):
    """The same instance reconnected after a disconnect that left the jaws closed says so above INFO."""

    def test_a_reconnect_after_a_closed_disconnect_warns(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        g.disconnect()
        with self.assertLogs(g.logger, level="WARNING") as logs:
            g.connect()
        self.assertTrue(any("last took the jaws to stand CLOSED" in line for line in logs.output))

    def test_a_reconnect_after_an_open_disconnect_does_not(self) -> None:
        """⭐ THE CONTROL: the warning is about the closed case only."""
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.disconnect()
        with self.assertNoLogs(g.logger, level="WARNING"):
            g.connect()


class _Travelling(_ToggleJaws):
    """Jaws that take ``travel_reads`` reads of the open switch to arrive after a flip: off both stops meanwhile."""

    def __init__(self, travel_reads: int, **kw) -> None:
        super().__init__(open_switch=True, **kw)
        self.travel_reads = travel_reads
        self.pending = 0

    def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
        before = self.closed
        super().set_digital_output(pin, value, port=port)
        if self.closed != before:
            self.pending = self.travel_reads

    def get_digital_input(self, pin, *, port=DigitalIOPort.STANDARD) -> bool:
        if pin == OPEN_SW and self.pending:
            self.pending -= 1
            return False
        return super().get_digital_input(pin, port=port)


class SingleToggleFailurePathTests(unittest.TestCase):
    """The paths the third review demonstrated (2026-09-23): writes that raise after reaching the controller, and a
    stroke interrupted before its wait finished."""

    def test_an_unknowable_edge_is_said_at_disconnect_and_warned_at_connect(self) -> None:
        io = _ReplyLost()
        g = _toggle(io)
        g.connect()
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(0.0)
        with self.assertLogs(g.logger, level="INFO") as logs:
            g.disconnect()
        self.assertTrue(any("UNKNOWN" in line for line in logs.output), "the disconnect said 'open'")
        with self.assertLogs(g.logger, level="WARNING") as logs:
            g.connect()
        self.assertTrue(any("unknowable" in line for line in logs.output))

    def test_a_second_connect_without_a_disconnect_does_not_claim_one(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        with self.assertLogs(g.logger, level="WARNING") as logs:
            g.connect()
        self.assertFalse(any("disconnect" in line for line in logs.output))

    def test_the_open_switch_decides_the_grasp_evidence_after_a_failed_write(self) -> None:
        io = _ReplyLost(open_switch=True)
        io.lose_reply = False
        io.inputs = {PART_SW: True}                         # a part between the jaws, open or closed
        g = _toggle(io, open_confirm_input_pin=OPEN_SW, part_present_input_pin=PART_SW, close_timeout_s=0.01)
        g.connect()
        g.set_width_mm(0.0)
        io.lose_reply = True
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(80.0)                            # the release reached the jaws; its reply did not
        self.assertFalse(io.closed)
        self.assertFalse(g.is_object_detected(), "a grasp reported while the open switch reads open")
        self.assertIs(g.hold_evidence(), HoldEvidence.UNMEASURED)

    def test_an_interrupted_open_is_waited_out_before_the_next_decision(self) -> None:
        io = _Travelling(travel_reads=5, closed=True)
        interrupted = {"done": False}

        def sleep(seconds: float) -> None:
            io.events.append(("sleep", seconds))
            if seconds == 0.02 and not interrupted["done"]:
                interrupted["done"] = True
                raise KeyboardInterrupt                     # Ctrl-C while the jaws travel open

        g = JawIOGripper(io, actuation="single_toggle", close_output_pin=CLOSE_PIN, pulse_s=PULSE_S,
                         open_confirm_input_pin=OPEN_SW, close_timeout_s=5.0, sleep=sleep)
        g._connected = True                                 # closed jaws, as a close in this session left them
        g._closed = True
        with self.assertRaises(KeyboardInterrupt):
            g.set_width_mm(80.0)
        g.set_width_mm(80.0)                                # the retry: the jaws are still travelling open
        self.assertFalse(io.closed, "the retry pulsed opening jaws back closed")
        self.assertEqual(io.outputs_written().count(HIGH), 1)

    def test_a_pin_left_high_is_dropped_at_connect(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        io.outputs[CLOSE_PIN] = True                        # a low write that never arrived
        with self.assertLogs(g.logger, level="WARNING"):
            g.connect()
        self.assertFalse(io.outputs[CLOSE_PIN])
        self.assertFalse(io.closed, "a falling edge must not flip a device that flips on the rising one")


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
            g.set_width_mm(0.0)
            g.set_width_mm(0.0)
        text = "\n".join(logs.output)
        self.assertIn("actuating jaws CLOSED via single_toggle", text)
        self.assertIn("jaws already stand CLOSED: no pulse", text)


class TheAdviceIsAPulseTests(unittest.TestCase):
    """A toggle's jaws are stood open with a pulse, never by hand (owner cell, 2026-09-23).

    A device that keeps its own flip state is not moved by a hand that pushes its jaws open: the next pulse then only
    brings the device's state round, nothing visibly moves, and every command after it is inverted. That is the
    leading explanation of the owner's cell closing at the tray. A pulse keeps the device and the jaws in step on
    every kind of toggle, so it is the only advice the driver gives.
    """

    def test_the_connect_line_says_pulse(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        with self.assertLogs(g.logger, level="INFO") as logs:
            g.connect()
        text = "\n".join(logs.output)
        self.assertIn("with a pulse", text)
        self.assertNotRegex(text, r"(?<!never )by hand")

    def test_the_reconnect_warning_says_pulse(self) -> None:
        io = _ToggleJaws()
        g = _toggle(io)
        g.connect()
        g.set_width_mm(0.0)
        g.disconnect()
        with self.assertLogs(g.logger, level="WARNING") as logs:
            g.connect()
        text = "\n".join(logs.output)
        self.assertIn("with a pulse", text)
        self.assertNotRegex(text, r"(?<!never )by hand")


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
    def _cell(vendor: str = "jaw_io", **jaw: float) -> dict[str, object]:
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

    def test_the_schema_refuses_a_lone_closed_switch(self) -> None:
        # Off its stop it only repeats the belief, so a lost pulse would be read back as a grasp.
        message = self._schema_refusal(closed_confirm_input_pin=0)
        for part in ("single_toggle", "closed_confirm_input_pin", "open_confirm_input_pin"):
            self.assertIn(part, message)
        JawIOGripperConfig.model_validate({"actuation": "single_toggle", "closed_confirm_input_pin": 0,
                                           "open_confirm_input_pin": 1})    # both switches are fine

    def test_the_constructor_refuses_a_lone_closed_switch(self) -> None:
        with self.assertRaises(ValueError) as caught:
            JawIOGripper(FakeIO(), actuation="single_toggle", close_output_pin=CLOSE_PIN,
                         closed_confirm_input_pin=CLOSED_SW)
        self.assertIn("open_confirm_input_pin", str(caught.exception))

    def test_an_unknown_actuation_names_all_three(self) -> None:
        with self.assertRaises(ValueError) as caught:
            JawIOGripper(FakeIO(), actuation="hydraulic")
        for known in ("single_solenoid", "double_solenoid", "single_toggle"):
            self.assertIn(known, str(caught.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
