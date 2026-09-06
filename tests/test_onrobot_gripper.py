"""The OnRobot RG2/RG6 as a configured vendor, and the branch whose absence fails silently.

⛔⛔ **THE MOST IMPORTANT TEST HERE IS `test_a_configured_onrobot_cell_is_not_silently_gripper_less`.**
`runtime_pick.py` dispatches on the gripper vendor and ends in an `else` that substitutes a
`NullGripper`: the cell connects, reports every pick a success and holds nothing. That `else` carries
`# pragma: no cover` and nothing enumerated it. So adding an enum member WITHOUT its branch produces
a cell that looks healthy and grips air, and no existing test would have noticed.
"""

from __future__ import annotations

import unittest
from typing import Any

from src.config.schema.robot import GripperConfig
from src.robot.core.gripper_vendor import GripperVendor
from src.robot.grippers.onrobot import OnRobotGripper
from src.robot.grippers.onrobot_modbus import ModbusError, RGStatus


class FakeRG:
    """Stands in for `OnRobotRG`, recording what the driver asked of the wire."""

    def __init__(self, *, status: RGStatus = RGStatus.AT_POSITION, reachable: bool = True) -> None:
        self.connected: tuple[str, int] | None = None
        self.grips: list[dict[str, Any]] = []
        self._status = status
        self._reachable = reachable
        self.width = 65.5

    def connect(self, host: str, port: int) -> None:
        self.connected = (host, port)

    def disconnect(self) -> None:
        self.connected = None

    def probe(self, *, sweep_units: bool = True) -> Any:
        from src.robot.grippers.onrobot_modbus import RGProbeReading

        if not self._reachable:
            return RGProbeReading(reachable=False, unit=65, detail="no answer")
        return RGProbeReading(
            reachable=True, unit=65, width_mm=self.width, status=self._status
        )

    def status(self) -> RGStatus:
        return self._status

    def width_mm(self) -> float:
        return self.width

    def grip(self, *, width_mm: float, force_n: float, use_fingertip_offset: bool) -> None:
        self.grips.append(
            {"width_mm": width_mm, "force_n": force_n, "offset": use_fingertip_offset}
        )


def _gripper(**kw: Any) -> tuple[OnRobotGripper, FakeRG]:
    fake = FakeRG(**{k: v for k, v in kw.items() if k in ("status", "reachable")})
    g = OnRobotGripper(
        GripperConfig(vendor="onrobot", max_width_mm=160.0, min_width_mm=0.0),
        host="10.0.0.5",
        client_factory=lambda: fake,
    )
    return g, fake


class ProtocolMismatchTests(unittest.TestCase):
    """The three places the vendor-neutral Protocol does not fit an RG."""

    def test_activate_is_a_no_op_rather_than_a_refusal(self) -> None:
        """⛔ AN RG HAS NO ACTIVATION AND NO REGISTER THAT MEANS ONE. Raising would make every
        vendor-neutral caller special-case this gripper; commanding something would mean writing
        into whatever register 0 happens to be, which is the TARGET FORCE."""
        g, fake = _gripper()
        g.connect()
        g.activate()
        self.assertEqual(fake.grips, [], "activate must command nothing at all")

    def test_speed_is_dropped_and_said_once(self) -> None:
        """⛔ RG2/RG6 HAVE NO SPEED REGISTER. Raising breaks callers on a parameter the Protocol
        requires them to be able to pass; dropping it silently is a lie. So it is dropped and
        logged, once per connection rather than once per command -- a warning on every close is one
        an operator stops reading."""
        g, fake = _gripper()
        g.connect()
        with self.assertLogs("OnRobotGripper", level="WARNING") as caught:
            g.set_width_mm(40.0, speed=0.5)
            g.set_width_mm(30.0, speed=0.5)
        self.assertEqual(len(caught.records), 1, "said once, not per command")
        self.assertIn("no speed register", caught.output[0])
        self.assertEqual(len(fake.grips), 2, "both commands still went out")

    def test_force_is_newtons_not_a_count(self) -> None:
        """⚠ Robotiq's force is an opaque 0-255 whose physical span differs per model; an RG's
        register is tenths of a newton, so this is exact rather than a calibration guess."""
        g, fake = _gripper()
        g.connect()
        g.set_width_mm(40.0, force=35.0)
        self.assertEqual(fake.grips[-1]["force_n"], 35.0)
        g.set_width_mm(40.0)
        self.assertEqual(fake.grips[-1]["force_n"], 20.0, "the configured default, in newtons")


class PolarityTests(unittest.TestCase):
    def test_open_commands_a_larger_width_than_close(self) -> None:
        """⛔ THE OPPOSITE OF THE ROBOTIQ CLIENT IN THE SAME PACKAGE, where 0 is open and 255 is
        closed. Here the number is an OPENING, so bigger is more open. Two grippers in one package
        with opposite polarity is a real hazard."""
        g, fake = _gripper()
        g.connect()
        g.close()
        closed = fake.grips[-1]["width_mm"]
        g.open()
        opened = fake.grips[-1]["width_mm"]
        self.assertLess(closed, opened)


class ConnectRefusalTests(unittest.TestCase):
    def test_a_tripped_safety_switch_refuses_at_connect(self) -> None:
        """⛔ Finding this out at connect costs a message; finding it out mid-pick costs a part. The
        gripper stays dead until TOOL POWER is cycled and every command meanwhile is discarded."""
        g, _ = _gripper(status=RGStatus.SAFETY_1_TRIPPED)
        with self.assertRaises(ModbusError) as caught:
            g.connect()
        self.assertIn("TOOL POWER", str(caught.exception))
        self.assertFalse(g.is_connected)

    def test_a_unit_that_does_not_answer_refuses(self) -> None:
        g, _ = _gripper(reachable=False)
        with self.assertRaises(ModbusError):
            g.connect()
        self.assertFalse(g.is_connected)

    def test_commanding_before_connecting_refuses(self) -> None:
        g, _ = _gripper()
        with self.assertRaises(ModbusError):
            g.set_width_mm(40.0)


class VendorWiringTests(unittest.TestCase):
    def test_a_configured_onrobot_cell_is_not_silently_gripper_less(self) -> None:
        """⛔⛔ THE ONE THAT MATTERS. `runtime_pick` dispatches on the gripper vendor and ends in an
        `else` that substitutes a NullGripper -- the cell CONNECTS, reports every pick a success and
        holds nothing. That `else` carries `# pragma: no cover` and nothing enumerated it, so an
        enum member added without its branch produces a healthy-looking cell that grips air.

        This asserts the branch exists by NAME in the dispatch, because the alternative -- building a
        whole service to find out -- needs an arm, and the failure this guards against is precisely
        that a silent fallback looks like success.
        """
        from pathlib import Path

        body = (
            Path(__file__).resolve().parents[1]
            / "src" / "robot" / "execution" / "runtime_pick.py"
        ).read_text(encoding="utf-8")
        self.assertIn("GripperVendor.ONROBOT", body, "no dispatch branch: cells would grip air")
        self.assertLess(
            body.index("GripperVendor.ONROBOT"),
            body.index("SubstitutionReason.NO_DRIVER"),
            "the branch must come BEFORE the NullGripper fallback",
        )

    def test_the_registry_builds_it(self) -> None:
        from src.robot.grippers import create_gripper

        built = create_gripper(
            GripperVendor.ONROBOT,
            config=GripperConfig(vendor="onrobot"),
            host="10.0.0.5",
        )
        self.assertIsInstance(built, OnRobotGripper)

    def test_the_config_block_exists_and_does_not_offer_a_speed(self) -> None:
        """⚠ A key that reached nothing would be worse than its absence."""
        block = GripperConfig().onrobot
        self.assertEqual(block.port, 502)
        self.assertEqual(block.unit_id, 65)
        self.assertNotIn("speed", block.model_dump())

    def test_the_host_is_not_the_robots(self) -> None:
        """⛔ The Compute Box is a SEPARATE DEVICE. The Robotiq branch reuses robot.ur.ip because a
        URCap daemon runs on the controller; defaulting this one to the arm would point every
        command at a machine that has never heard of it."""
        import inspect

        from src.robot.execution import runtime_pick

        source = inspect.getsource(runtime_pick.RuntimePickService.from_robot_config)
        # ⚠ FROM THE `create_gripper` CALL, NOT FROM THE ENUM NAME. A window measured from the enum
        # was 900 characters of explanatory comment and never reached the code, so the assertion
        # failed on prose rather than on wiring -- a guard that measures the wrong span.
        marker = source.index("create_gripper(\n                GripperVendor.ONROBOT")
        branch = source[marker:marker + 600]
        self.assertIn("rg.host", branch)
        self.assertNotIn("robot_cfg.ur.ip", branch)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
