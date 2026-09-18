"""The desk names the gripper driver a config builds, and a hand says which drivers can actuate it (lane C4g).

``robot.gripper.model`` selects no driver, and the base tree states ``vendor: robotiq``, so a profile naming a customer
hand that forgets ``gripper.vendor`` aimed the Robotiq socket driver at it, and a vendor with no driver was refused only
at connect. The desk now states the driver the build constructs and BLOCKs exactly where the build would substitute a
``NullGripper``, from the one function both read. The owner decided on 2026-09-17 (decision 7) that a registry hand
may list the ``gripper.vendor`` drivers that can actuate it, and a real UR profile naming another is refused at load;
``none`` and ``dummy`` are always admitted.
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from src.config.schema.robot import RobotConfig

_DATA = Path(__file__).resolve().parents[1] / "config"


def _cfg(vendor: str, model: "str | None" = None, **gripper: object) -> RobotConfig:
    # The capsule guard reads no hand geometry, so a cell naming no hand still builds its arm: the driver is the
    # question here, and the hand rules have their own tests.
    return RobotConfig.model_validate({
        "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "ik"},
        "gripper": {"vendor": vendor, "model": model, **gripper},
        "safety": {"self_collision": {"backend": "capsule"}},
    })


def _rows(cfg: RobotConfig) -> dict:
    from src.robot.execution.real_cell.preflight import run_config_preflight

    report = run_config_preflight(cfg, curobo_available=True, collision_engine="coal")
    return {check.name: check for check in report.checks}


def _built(cfg: RobotConfig) -> object:
    """What the build constructs for ``cfg`` on the arm a config builds: the UR driver, not connected."""
    from src.robot.drivers.ur.arm import URRobotArm
    from src.robot.execution.robot_parts import build_gripper

    return build_gripper(cfg, arm=URRobotArm(cfg))


class TheDriverRowTests(unittest.TestCase):
    def test_a_vendor_with_no_driver_blocks_at_the_desk_with_the_builds_sentence(self) -> None:
        cfg = _cfg("schunk", "schunk_egu50")
        row = _rows(cfg)["gripper driver"]
        self.assertEqual(str(row.status), "block")
        substitution = getattr(_built(cfg), "substitution")
        self.assertIn(substitution.detail, row.detail)
        self.assertEqual(row.fix, substitution.fix)

    def test_robotiq_on_a_ur_arm_reads_ok_naming_it(self) -> None:
        """⭐ THE CONTROL."""
        row = _rows(_cfg("robotiq", "robotiq_2f85"))["gripper driver"]
        self.assertEqual(str(row.status), "ok")
        for part in ("robotiq", "robotiq_2f85"):
            self.assertIn(part, row.detail)

    def test_the_desk_and_the_build_agree_for_every_vendor(self) -> None:
        from src.robot.core import GripperVendor

        for member in GripperVendor:
            with self.subTest(vendor=member.value):
                cfg = _cfg(member.value)
                substituted = getattr(_built(cfg), "substitution", None) is not None
                self.assertEqual(str(_rows(cfg)["gripper driver"].status) == "block", substituted)

    def test_no_gripper_and_a_dummy_on_a_real_arm_warn(self) -> None:
        for vendor in ("none", "dummy"):
            with self.subTest(vendor=vendor):
                self.assertEqual(str(_rows(_cfg(vendor))["gripper driver"].status), "warn")

    def test_the_bench_row_names_the_wiring_of_the_configured_vendor(self) -> None:
        jaw = _rows(_cfg("jaw_io", jaw_io={"actuation": "double_solenoid", "close_output_pin": 3,
                                           "open_output_pin": 4}))["end-effector wiring"]
        for part in ("close", "open", "3", "4"):
            self.assertIn(part, jaw.detail)
        self.assertNotIn("URCap", jaw.detail)
        onrobot = _rows(_cfg("onrobot", onrobot={"host": "192.168.1.1"}))["end-effector wiring"]
        self.assertIn("Compute Box", onrobot.detail)
        self.assertIn("192.168.1.1", onrobot.detail)
        self.assertIn("URCap", _rows(_cfg("robotiq"))["end-effector wiring"].detail)


class _Tree:
    def __init__(self, case: unittest.TestCase, layer: str) -> None:
        from src.config.loader import reload_config

        self.root = Path(case.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_DATA, self.root)
        (self.root / "robot" / "robot.trial.yaml").write_text(layer, encoding="utf-8")
        reload_config()

    def load(self):
        from src.config.loader import load_robot_config

        return load_robot_config(self.root, profile="trial")


class TheHandListsItsDriversTests(unittest.TestCase):
    def test_a_real_ur_profile_whose_driver_the_hand_does_not_list_is_refused_at_load(self) -> None:
        from src.config.loader import ConfigError

        tree = _Tree(self, "robot:\n  gripper:\n    model: robotiq_hande\n    vendor: jaw_io\n")
        with self.assertRaises(ConfigError) as caught:
            tree.load()
        for part in ("robotiq_hande", "jaw_io", "robotiq", "config/grippers/robotiq_hande.yaml"):
            self.assertIn(part, str(caught.exception))

    def test_no_gripper_and_a_dummy_are_always_admitted(self) -> None:
        """⭐ THE CONTROL: a real arm with nothing on the flange yet, and a desk stand-in, both load."""
        for vendor in ("none", "dummy"):
            with self.subTest(vendor=vendor):
                layer = f"robot:\n  gripper:\n    model: robotiq_hande\n    vendor: {vendor}\n"
                self.assertEqual(str(_Tree(self, layer).load().gripper.vendor), vendor)

    def test_a_listed_driver_and_an_arm_that_is_not_a_real_ur_are_admitted(self) -> None:
        self.assertEqual(str(_Tree(self, "robot:\n  gripper:\n    model: robotiq_hande\n").load().gripper.vendor),
                         "robotiq")
        layer = "robot:\n  vendor: dummy\n  gripper:\n    model: robotiq_hande\n    vendor: jaw_io\n"
        self.assertEqual(str(_Tree(self, layer).load().gripper.vendor), "jaw_io")

    def test_a_hand_that_lists_no_drivers_admits_any(self) -> None:
        from src.config.grippers import load_gripper

        self.assertIsNone(load_gripper("schunk_egu50", aliases=False).drivers)
        layer = "robot:\n  gripper:\n    model: schunk_egu50\n    vendor: jaw_io\n"
        self.assertEqual(str(_Tree(self, layer).load().gripper.vendor), "jaw_io")

    def test_the_robotiq_hands_list_the_robotiq_driver(self) -> None:
        from src.config.grippers import load_gripper

        for model in ("robotiq_2f85", "robotiq_hande"):
            with self.subTest(model=model):
                self.assertEqual(load_gripper(model, aliases=False).drivers, ("robotiq",))

    def test_a_list_naming_no_driver_this_stack_knows_is_refused(self) -> None:
        from pydantic import ValidationError

        from src.config.grippers import load_gripper
        from src.config.schema.grippers import GripperSpec

        data = load_gripper("robotiq_hande", aliases=False).model_dump(mode="json")
        data["drivers"] = ["robotiq_socket"]
        with self.assertRaises(ValidationError):
            GripperSpec.model_validate(data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
