"""Tests for the PayloadGuard and URConnection.set_payload.

What connect does with the payload (refused before the socket opens, pushed once, rolled back on any failure) is the
planned arm's since P0g slice 15: ``tests/test_a_planned_arm_connects_through_its_transport.py``.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock


from src.config.schema.robot import (
    PayloadSafetyConfig,
    RobotSafetyConfig,
    WorkspaceLimitsConfig,
)

from src.robot.core import MotionCommand
from src.robot.drivers.ur.connection import URConnection


from src.robot.safety import (
    PayloadGuard,
    SafetyContext,
    SafetyPreflight,
    SafetyReason,
)


def _ctx() -> SafetyContext:
    return SafetyContext(command=MotionCommand.MOVE_TO)


class PayloadGuardTests(unittest.TestCase):
    def test_default_envelope_accepts(self) -> None:
        cfg = PayloadSafetyConfig()
        d = PayloadGuard(cfg).evaluate(_ctx())
        self.assertTrue(d.accepted)

    def test_over_mass_rejects(self) -> None:
        # Schema validator enforces mass_kg <= max_mass_kg at construction.
        # To exercise the guard's runtime check we bypass via model_construct.
        cfg = PayloadSafetyConfig.model_construct(
            mass_kg=10.0,
            max_mass_kg=5.0,
            cog_mm=(0.0, 0.0, 0.0),
            inertia_kgm2=(0.0, 0.0, 0.0),
        )
        d = PayloadGuard(cfg).evaluate(_ctx())
        self.assertFalse(d.accepted)
        self.assertIs(d.reason, SafetyReason.PAYLOAD)
        self.assertEqual(d.detail["reason"], "over_mass")

    def test_negative_inertia_rejects(self) -> None:
        cfg = PayloadSafetyConfig.model_construct(
            mass_kg=1.0,
            max_mass_kg=5.0,
            cog_mm=(0.0, 0.0, 0.0),
            inertia_kgm2=(-1.0, 0.0, 0.0),
        )
        d = PayloadGuard(cfg).evaluate(_ctx())
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "negative_inertia")

    def test_negative_mass_rejects(self) -> None:
        cfg = PayloadSafetyConfig.model_construct(
            mass_kg=-1.0,
            max_mass_kg=5.0,
            cog_mm=(0.0, 0.0, 0.0),
            inertia_kgm2=(0.0, 0.0, 0.0),
        )
        d = PayloadGuard(cfg).evaluate(_ctx())
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "negative_mass")


class PayloadWiringTests(unittest.TestCase):
    def test_every_cell_wires_the_guard(self) -> None:
        pf = SafetyPreflight.from_safety_config(RobotSafetyConfig(), WorkspaceLimitsConfig(), arm_model=None)
        self.assertIn("payload", pf.guard_names)

    def test_the_switch_that_left_it_out_is_refused(self) -> None:
        """No cell switches a guard off since 2026-10-17 (the owner's cleanup)."""
        from pydantic import ValidationError

        for value in (True, False):
            with self.subTest(enforce=value), self.assertRaises(ValidationError):
                RobotSafetyConfig.model_validate({"payload": {"enforce": value}})


class URConnectionSetPayloadTests(unittest.TestCase):
    def _conn_with_mock_ctrl(self) -> tuple[URConnection, MagicMock]:
        conn = URConnection(ip="127.0.0.1")
        ctrl = MagicMock()
        recv = MagicMock()
        conn._ctrl = ctrl
        conn._recv = recv
        return conn, ctrl

    def test_mm_to_metres_conversion(self) -> None:
        conn, ctrl = self._conn_with_mock_ctrl()
        conn.set_payload(2.5, (10.0, -20.0, 30.0))
        ctrl.setPayload.assert_called_once()
        args = ctrl.setPayload.call_args.args
        self.assertAlmostEqual(args[0], 2.5)
        # CoG converted to metres.
        cog = args[1]
        self.assertAlmostEqual(cog[0], 0.010)
        self.assertAlmostEqual(cog[1], -0.020)
        self.assertAlmostEqual(cog[2], 0.030)

    def test_negative_mass_raises(self) -> None:
        conn, _ = self._conn_with_mock_ctrl()
        with self.assertRaises(ValueError):
            conn.set_payload(-0.1)


if __name__ == "__main__":
    unittest.main()
