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

from src.robot.core import Gripper, GripperVendor, RobotConnectionError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.core.gripper import ObjectDetectingGripper
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


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
