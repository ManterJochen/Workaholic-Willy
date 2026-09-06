"""Tests for the PayloadGuard and URConnection.set_payload."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from src.config.schema.robot import (
    PayloadSafetyConfig,
    RobotConfig,
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


def _fake_bare_flange_conn(model: str = "ur5e"):
    """A mocked URConnection whose kinematics look like a controller running a BARE FLANGE.

    connect() now derives the controller's active tool frame (inv(DH flange) @ getForwardKinematics)
    and refuses a mismatch, so a connection mock has to answer those two calls plausibly. A bare
    flange is what `tool_frame.source: "willy"` expects, since that mode composes on Willy's side.
    """
    from src.robot.drivers.ur.pose import URPose
    from src.robot.safety._ur_kinematics import ur_link_transforms_mm

    q = [0.0, -1.2, 1.3, -0.4, 1.5, 0.2]
    conn = MagicMock()
    conn.is_connected = True
    conn.get_joint_positions.return_value = q
    links = ur_link_transforms_mm(model, np.asarray(q, dtype=np.float64))
    assert links is not None
    conn.fk.return_value = URPose.from_T(links[-1]).to_ur_list()
    # connect()'s tool-frame check reads the controller's CURRENT TCP via fk_current (the
    # no-argument getForwardKinematics), because the q-form is corrupted by any preceding
    # motion -- see URConnection.fk_current for the measurement. Same active frame here.
    conn.fk_current.return_value = URPose.from_T(links[-1]).to_ur_list()
    conn.is_steady.return_value = True
    return conn


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
            enforce=True,
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
            enforce=True,
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
            enforce=True,
            mass_kg=-1.0,
            max_mass_kg=5.0,
            cog_mm=(0.0, 0.0, 0.0),
            inertia_kgm2=(0.0, 0.0, 0.0),
        )
        d = PayloadGuard(cfg).evaluate(_ctx())
        self.assertFalse(d.accepted)
        self.assertEqual(d.detail["reason"], "negative_mass")


class PayloadWiringTests(unittest.TestCase):
    def test_enforce_true_registers_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"payload": {"enforce": True}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertIn("payload", pf.guard_names)

    def test_enforce_false_omits_guard(self) -> None:
        cfg = RobotSafetyConfig.model_validate({"payload": {"enforce": False}})
        pf = SafetyPreflight.from_safety_config(cfg, WorkspaceLimitsConfig())
        self.assertNotIn("payload", pf.guard_names)


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


class URConnectPayloadRefuseTests(unittest.TestCase):
    """S0 / decision D4: ``connect()`` must fail closed on ``enforce: true`` + ``mass_kg: 0.0``.

    That is the SHIPPED default, and pushing ``setPayload(0.0)`` to a real controller silently
    overwrites the payload a mounted tool was configured for -- fail-OPEN in the dangerous direction.
    These assert on BEHAVIOUR (did the socket open? was set_payload pushed?), not just that connect
    returned, because a guard that is accepted and does nothing is the exact failure this closes.
    """

    @staticmethod
    def _arm(**payload):  # type: ignore[no-untyped-def]
        from backend.src.robot.drivers.ur.arm import URRobotArm

        cfg = RobotConfig.model_validate({
            "vendor": "ur", "safety": {"payload": payload},
            # A real cell must declare its tool frame before connect() lets it move; these tests are
            # about the PAYLOAD gates, so declare a valid one and let them exercise what they mean to.
            "gripper": {"tool_frame": {
                "source": "willy", "offset_mm": (0.0, 132.0, 0.0),
                "rotation_quat_xyzw": (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476),
            }},
        })
        arm = URRobotArm(cfg)
        conn = _fake_bare_flange_conn()
        arm._conn = conn
        return arm, conn

    def test_enforce_true_zero_mass_refuses_before_opening_the_socket(self) -> None:
        from backend.src.robot.core import RobotConnectionError

        arm, conn = self._arm(enforce=True, mass_kg=0.0)
        with self.assertRaises(RobotConnectionError) as cm:
            arm.connect()
        conn.connect.assert_not_called()          # refused BEFORE opening anything
        conn.set_payload.assert_not_called()
        self.assertIn("mass_kg", str(cm.exception))

    def test_enforce_true_real_mass_connects_and_pushes(self) -> None:
        arm, conn = self._arm(enforce=True, mass_kg=1.5, cog_mm=(0.0, 0.0, 40.0))
        arm.connect()
        conn.connect.assert_called_once()
        conn.set_payload.assert_called_once_with(1.5, (0.0, 0.0, 40.0))

    def test_enforce_false_bare_flange_connects_without_pushing(self) -> None:
        arm, conn = self._arm(enforce=False, mass_kg=0.0)
        arm.connect()
        conn.connect.assert_called_once()
        conn.set_payload.assert_not_called()      # enforce:false never touches controller payload


if __name__ == "__main__":
    unittest.main()
