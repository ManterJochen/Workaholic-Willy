"""What a caller-supplied arm does, and does not, say about this host's readiness.

``RuntimePickService.from_robot_config`` takes a live ``arm`` handle because some end-effectors and
arms are not describable by config (the Isaac arm carries the simulator session it lives on). The
question this file pins is what that handle is allowed to switch off.

MEASURED on the host that runs this suite: ``URRobotArm`` CONSTRUCTS without ``ur_rtde`` (the SDK is
imported inside ``connect()``), which `tests/test_s4_from_robot_config.py:30` already relies on to
build a UR arm in CI. So "the caller holds a constructed arm" does NOT prove the vendor SDK is on
this box, and it is the SDK the readiness gate is about. What a supplied handle proves is only that
``create_arm`` will not be called; whether a real device gets driven is decided by what the arm IS.

The stand-in arms below declare their vendor through ``RobotCapabilities.vendor``, which is the
identifier every driver already advertises (`drivers/sim/arm.py:197` declares ``"sim"``,
`drivers/ur/arm.py:64` declares ``"ur"``), and `core/capabilities.py` says in as many words that
pipeline code should branch on these flags rather than on concrete driver classes.

Env-independent: the SIM row's SDK requirement is monkeypatched, so the result does not depend on
whether the box running the suite happens to have Isaac installed.
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.core.capabilities import RobotCapabilities
from src.robot.core.errors import RobotConnectionError
from src.robot.core.vendor import RobotVendor
from src.robot.drivers import doctor
from src.robot.drivers.dummy.arm import DummyRobotArm
from src.robot.execution.runtime_pick import RuntimePickService
from tests.test_grasping_config_wiring import _calc_and_perception

#: A module name no interpreter can import, so "this host is not ready" is a fact of the test rather
#: than a fact of the box.
_ABSENT_SDK = ("nope_xyz_123",)


def _sim_cfg(**sim: object) -> RobotConfig:
    """A sim cell with ``mock_mode`` false: the profile the Isaac runners boot under ``--boot config``."""
    return RobotConfig(  # type: ignore[arg-type]
        vendor="sim",
        gripper={"vendor": "none"},
        grasping={"default_mode": "auto", "max_attempts": 5},
        sim={"mock_mode": False, **sim},
    )


def _arm_declaring(vendor: str) -> DummyRobotArm:
    """A working arm that advertises ``vendor``.

    A test double rather than the real driver on purpose: importing ``IsaacRobotArm`` to get an
    object that says ``"sim"`` would drag the Isaac SDK into the mock suite, which is the one thing
    the whole doctor module exists to avoid.
    """
    arm = DummyRobotArm()
    arm._capabilities = RobotCapabilities(vendor=vendor, model="stand-in", is_simulated=True)
    return arm


def _build(cfg: RobotConfig, arm: DummyRobotArm | None) -> RuntimePickService:
    calc, perception = _calc_and_perception()
    return RuntimePickService.from_robot_config(
        cfg, calculator=calc, perception=perception, arm=arm,  # type: ignore[arg-type]
    )


class SuppliedArmAndTheVendorReadinessGate(unittest.TestCase):
    def test_supplied_driver_for_the_configured_vendor_is_still_gated(self) -> None:
        # The arm the caller holds IS this vendor's driver, so this cell drives the real device and
        # the host has to be able to. The gate must fire, exactly as it does when the root builds the
        # arm itself.
        with patch.dict(doctor._ARM_VENDOR_SDKS, {RobotVendor.SIM: _ABSENT_SDK}):
            with self.assertRaises(RobotConnectionError) as raised:
                _build(_sim_cfg(), _arm_declaring("sim"))
        self.assertIn("host not ready", str(raised.exception))
        self.assertIn("nope_xyz_123", str(raised.exception))

    def test_supplied_stand_in_arm_is_not_gated(self) -> None:
        # A handle that is NOT this vendor's driver drives no real device, so the vendor's SDK is
        # beside the point. This is the case that must keep working: refusing it would gate a cell
        # on an SDK nothing in it would ever call.
        with patch.dict(doctor._ARM_VENDOR_SDKS, {RobotVendor.SIM: _ABSENT_SDK}):
            svc = _build(_sim_cfg(), DummyRobotArm())
        self.assertIsInstance(svc.orchestrator.arm, DummyRobotArm)

    def test_the_configured_vendor_driver_builds_when_the_host_is_ready(self) -> None:
        # The Isaac path, stood in for: the runner passes a live sim arm on a box where the sim SDK
        # imports. The gate runs and PASSES, so `--boot config` keeps building.
        with patch.dict(doctor._ARM_VENDOR_SDKS, {RobotVendor.SIM: ()}):
            svc = _build(_sim_cfg(), _arm_declaring("sim"))
        self.assertEqual(svc.orchestrator.arm.capabilities.vendor, "sim")

    def test_a_partial_double_counts_as_a_stand_in(self) -> None:
        # An object that implements no capabilities is not any vendor's driver, and the facade
        # already tolerates such doubles (`RuntimePickService._capabilities` falls back to "unknown").
        # Reading the vendor off it must not turn into an AttributeError at the gate.
        arm = DummyRobotArm()
        del arm._capabilities  # the partial double, without inventing a second fake arm class
        with patch.dict(doctor._ARM_VENDOR_SDKS, {RobotVendor.SIM: _ABSENT_SDK}):
            svc = _build(_sim_cfg(), arm)
        self.assertIs(svc.orchestrator.arm, arm)

    def test_mock_sim_profile_stays_ungated(self) -> None:
        # `sim.mock_mode: true` is the offline/CI cell: no Isaac boots, so no SDK is required, with
        # or without a supplied handle.
        with patch.dict(doctor._ARM_VENDOR_SDKS, {RobotVendor.SIM: _ABSENT_SDK}):
            _build(_sim_cfg(mock_mode=True), _arm_declaring("sim"))
            _build(_sim_cfg(mock_mode=True), None)


if __name__ == "__main__":
    unittest.main()
