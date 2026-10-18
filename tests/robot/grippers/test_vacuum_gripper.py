"""A suction end-effector a real cell can actually be configured with.

Willy could simulate suction, model its seal physics and score suction grasps, but ``GripperVendor``
listed only jaw vendors -- so the entire suction stack was reachable in sim and unreachable on
hardware. Tim's September cell needs BOTH a 2F-85 and a suction gripper, and the suction model is not
chosen yet.

That is exactly why this driver is vendor-neutral: an ejector on a UR is an output pin, a blow-off
pulse and (optionally) a vacuum switch on an input. No SDK, nothing manufacturer-specific -- so the
cell can be configured and tested before the cup exists, and bring-up becomes measuring pin numbers
rather than writing code.

STATUS: bucket (3). This has never touched a real vacuum generator; these tests are the evidence that
the LOGIC is right, and they say nothing about the wiring.
"""

from __future__ import annotations

import unittest

from src.robot.core import Gripper, GripperVendor, RobotConnectionError
from src.robot.core.arm_capabilities import DigitalIOPort
from src.robot.grippers import create_gripper
from src.robot.grippers.vacuum import VacuumGripper


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


def _gripper(io: FakeIO, **kw) -> VacuumGripper:
    kw.setdefault("sleep", lambda _s: None)
    return VacuumGripper(io, **kw)


class ProtocolConformanceTests(unittest.TestCase):
    def test_it_satisfies_the_gripper_protocol(self) -> None:
        """The whole point of the vendor-neutral Protocol: the pick path must not know it is suction."""
        self.assertIsInstance(_gripper(FakeIO()), Gripper)

    def test_the_registry_can_build_one(self) -> None:
        gripper = create_gripper(GripperVendor.VACUUM, io=FakeIO())
        self.assertIsInstance(gripper, VacuumGripper)

    def test_it_refuses_an_io_source_that_cannot_do_io(self) -> None:
        """Fail at construction with a readable message, not at the first pick with an AttributeError."""
        with self.assertRaises(TypeError) as ctx:
            VacuumGripper(object())  # type: ignore[arg-type]
        self.assertIn("SupportsDigitalIO", str(ctx.exception))


class EngageReleaseTests(unittest.TestCase):
    def test_width_below_the_threshold_engages_vacuum(self) -> None:
        """Width is reinterpreted exactly as the simulated cup does, so the two agree about the
        width-based Protocol and a runner can swap them without knowing which it holds."""
        io = FakeIO()
        g = _gripper(io, vacuum_output_pin=3, vacuum_on_below_mm=5.0)
        g.connect()
        g.set_width_mm(2.0)
        self.assertTrue(io.outputs[3])
        self.assertTrue(g.vacuum_on)

    def test_width_above_the_threshold_releases(self) -> None:
        io = FakeIO()
        g = _gripper(io, vacuum_output_pin=3)
        g.connect()
        g.set_width_mm(2.0)
        g.set_width_mm(40.0)
        self.assertFalse(io.outputs[3])
        self.assertFalse(g.vacuum_on)

    def test_reported_width_follows_the_vacuum_state(self) -> None:
        g = _gripper(FakeIO(), min_width_mm=0.0, max_width_mm=30.0)
        g.connect()
        self.assertEqual(g.get_width_mm(), 30.0)
        g.set_width_mm(1.0)
        self.assertEqual(g.get_width_mm(), 0.0)

    def test_connect_asserts_the_safe_state(self) -> None:
        """A cell that starts with the ejector latched from a previous run would hold a part it does not
        know about. The safe state is commanded, not assumed."""
        io = FakeIO()
        io.outputs[0] = True  # left on by whatever ran last
        _gripper(io).connect()
        self.assertFalse(io.outputs[0])

    def test_pins_are_driven_on_the_configured_bank(self) -> None:
        """A tool-mounted ejector is on the TOOL block; writing to STANDARD would silently do nothing
        useful on the real cell."""
        io = FakeIO()
        g = _gripper(io, io_port="tool")
        g.connect()
        g.set_width_mm(1.0)
        self.assertTrue(all(port == "tool" for _, _, port in io.writes))


class BlowOffTests(unittest.TestCase):
    def test_release_pulses_the_blow_off_output(self) -> None:
        """Residual vacuum keeps a light part stuck to the cup after the ejector stops, so it lets go
        somewhere unintended. The pulse pushes it off deliberately -- and must return LOW."""
        io = FakeIO()
        g = _gripper(io, vacuum_output_pin=0, blow_off_output_pin=1)
        g.connect()
        io.writes.clear()
        g.set_width_mm(1.0)
        g.set_width_mm(40.0)
        blow = [(v) for pin, v, _ in io.writes if pin == 1]
        self.assertEqual(blow, [True, False], "blow-off must pulse, not latch on")

    def test_no_blow_off_configured_means_no_extra_writes(self) -> None:
        io = FakeIO()
        g = _gripper(io, vacuum_output_pin=0)
        g.connect()
        io.writes.clear()
        g.set_width_mm(40.0)
        self.assertEqual({pin for pin, _, _ in io.writes}, {0})


class VacuumSwitchTests(unittest.TestCase):
    def test_a_wired_switch_makes_object_detection_a_MEASUREMENT(self) -> None:
        """The real payoff of suction: a cup can be ASKED whether it is holding something, which opts the
        cell into post-close verification. A jaw gripper mostly has to be trusted."""
        io = FakeIO(inputs={5: True})
        g = _gripper(io, vacuum_ok_input_pin=5)
        g.connect()
        g.set_width_mm(1.0)
        self.assertTrue(g.is_object_detected())
        io.inputs[5] = False  # part fell off
        self.assertFalse(g.is_object_detected())

    def test_without_a_switch_it_reports_only_what_it_COMMANDED(self) -> None:
        """The honest answer is "as far as this driver knows" -- matching a jaw gripper with no feedback.
        Claiming a part is held with nothing to read it from would be worse than saying less."""
        g = _gripper(FakeIO(), vacuum_ok_input_pin=None)
        g.connect()
        self.assertFalse(g.is_object_detected())
        g.set_width_mm(1.0)
        self.assertTrue(g.is_object_detected())

    def test_engaging_waits_for_the_seal(self) -> None:
        """Commanding the pin and moving away immediately is how a cell drops parts: the ejector needs
        time to build. Engage must not return before the switch confirms."""
        io = FakeIO(inputs={5: False})
        polls = {"n": 0}

        def _sleep(_s: float) -> None:
            polls["n"] += 1
            if polls["n"] >= 3:
                io.inputs[5] = True  # seal builds on the third poll

        g = _gripper(io, vacuum_ok_input_pin=5, sleep=_sleep)
        g.connect()
        g.set_width_mm(1.0)
        self.assertGreaterEqual(polls["n"], 3)
        self.assertTrue(g.is_object_detected())

    def test_a_seal_that_never_builds_is_a_missed_grasp_not_a_crash(self) -> None:
        """A cup that does not catch is an ordinary outcome the verification stage judges. Raising would
        turn a normal miss into a cell-stopping fault."""
        io = FakeIO(inputs={5: False})
        g = _gripper(io, vacuum_ok_input_pin=5, engage_timeout_s=0.01)
        g.connect()
        g.set_width_mm(1.0)  # must not raise
        self.assertFalse(g.is_object_detected())


class LifecycleTests(unittest.TestCase):
    def test_commands_before_connect_are_rejected(self) -> None:
        g = _gripper(FakeIO())
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(1.0)

    def test_disconnect_releases_the_part(self) -> None:
        io = FakeIO()
        g = _gripper(io, vacuum_output_pin=2)
        g.connect()
        g.set_width_mm(1.0)
        g.disconnect()
        self.assertFalse(io.outputs[2])
        self.assertFalse(g.is_connected)

    def test_disconnect_never_raises_even_if_the_io_is_gone(self) -> None:
        """Teardown must not mask whatever actually tore the cell down."""
        class DeadIO(FakeIO):
            def set_digital_output(self, pin, value, *, port=DigitalIOPort.STANDARD) -> None:
                raise OSError("controller gone")

        g = _gripper(FakeIO(), vacuum_output_pin=0)
        g.connect()
        g._io = DeadIO()  # the controller drops out mid-teardown
        g.disconnect()  # must not raise
        self.assertFalse(g.is_connected)


class ConfigTests(unittest.TestCase):
    def test_the_vendor_is_selectable_from_config(self) -> None:
        """THE gap this closes: a real cell could not previously say it had a suction gripper at all."""
        from src.config.schema.robot.robot_schema import GripperConfig

        cfg = GripperConfig.model_validate({"vendor": "vacuum", "max_width_mm": 30.0, "min_width_mm": 0.0})
        self.assertEqual(cfg.vendor, "vacuum")

    def test_width_limits_come_from_the_gripper_config(self) -> None:
        from src.config.schema.robot.robot_schema import GripperConfig

        cfg = GripperConfig.model_validate({"vendor": "vacuum", "max_width_mm": 25.0, "min_width_mm": 1.0})
        g = _gripper(FakeIO(), config=cfg)
        self.assertEqual((g.min_width_mm, g.max_width_mm), (1.0, 25.0))

    def test_the_wiring_block_is_inert_for_a_jaw_cell(self) -> None:
        """Defaults must not imply a claim about any particular cell."""
        from src.config.schema.robot.robot_schema import GripperConfig

        cfg = GripperConfig()
        self.assertEqual(cfg.vendor, "robotiq")
        self.assertEqual(cfg.vacuum.io_port, "tool")
        self.assertIsNone(cfg.vacuum.vacuum_ok_input_pin)

    def test_out_of_range_pins_are_rejected_at_load(self) -> None:
        from pydantic import ValidationError

        from src.config.schema.robot.robot_schema import VacuumGripperConfig

        with self.assertRaises(ValidationError):
            VacuumGripperConfig.model_validate({"vacuum_output_pin": 99})


if __name__ == "__main__":
    unittest.main()
