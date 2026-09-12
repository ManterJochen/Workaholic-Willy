"""The arm and the gripper a robot config describes, pinned as literals before the builder moved.

The builder that turns ``robot.vendor`` and ``robot.gripper`` into two live handles lived inside
``RuntimePickService.from_robot_config``. It moved into ``execution/robot_parts.py`` so a robot can be
built without a pick service, and the move changed no handle and no substitution record. This file is
the proof: every expected value below is a literal read off the tree before the move.

The literals are the point. After the move the service calls the builder, so a table computed through
one and compared with the other would compare a function with itself and could not fail.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.core import RobotCapabilities, RobotVendor
from src.robot.drivers import create_arm
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grippers import SubstitutionReason
from tests.test_grasping_config_wiring import _calc_and_perception

#: A UR arm constructs without ur_rtde (the SDK import waits for connect()), and the readiness gate
#: refuses first on a box without it. Patched off only for the cases whose arm reports ``ur``.
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"

#: Widths no default carries, so a NullGripper that kept the schema defaults cannot pass.
_WIDTHS = {"min_width_mm": 7.5, "max_width_mm": 61.0}


class _ArmThatClaimsUrAndHoldsNoAddress:
    """Says ``ur`` and carries no config: the one shape with no controller address anywhere on it."""

    capabilities = RobotCapabilities(vendor="ur", model="ur5e")


@dataclass(frozen=True)
class _Substituted:
    reason: SubstitutionReason
    requested: str
    detail: str
    fix: str


@dataclass(frozen=True)
class _Case:
    name: str
    tree: dict[str, Any]
    #: A live arm handed in, or ``None`` when the arm is built from the tree.
    handle: Callable[[], object] | None
    #: Patch the readiness gate off, for the cases whose arm reports ``ur``.
    gated: bool
    arm: str
    gripper: str
    #: Attribute name on the built gripper, and its literal value.
    attributes: dict[str, object] = field(default_factory=dict)
    #: The digital I/O grippers switch the arm's pins, so their ``io`` is the arm itself.
    io_is_arm: bool = False
    substitution: _Substituted | None = None


_UR = {"vendor": "ur", "ur": {"ip": "10.9.9.9"}}

_TABLE: tuple[_Case, ...] = (
    _Case(
        name="robotiq on a UR arm",
        tree={**_UR, "gripper": {"vendor": "robotiq"}},
        handle=None, gated=True,
        arm="URRobotArm", gripper="GripperController",
        attributes={"ip": "10.9.9.9", "port": 63352},
    ),
    _Case(
        name="robotiq on a dummy handle under a ur tree",
        tree={**_UR, "gripper": {"vendor": "robotiq", **_WIDTHS}},
        handle=lambda: create_arm(RobotVendor.DUMMY), gated=False,
        arm="DummyRobotArm", gripper="NullGripper",
        attributes=dict(_WIDTHS),
        substitution=_Substituted(
            SubstitutionReason.ROBOTIQ_NEEDS_UR, "robotiq",
            "gripper.vendor='robotiq' but the arm in hand reports vendor 'dummy', which is not a "
            "UR. A Robotiq lives on the UR controller's tool I/O, so an arm that is not on such a "
            "controller cannot reach one.",
            "For the Isaac sim use AutonomousGraspService.from_components (the IsaacGripper needs "
            "the shared session). On a real cell, set robot.vendor: ur and either let this build "
            "the arm or hand in the UR arm you built yourself.",
        ),
    ),
    _Case(
        name="robotiq on an arm claiming ur with no address",
        tree={**_UR, "gripper": {"vendor": "robotiq"}},
        handle=_ArmThatClaimsUrAndHoldsNoAddress, gated=True,
        arm="_ArmThatClaimsUrAndHoldsNoAddress", gripper="NullGripper",
        substitution=_Substituted(
            SubstitutionReason.ROBOTIQ_NEEDS_UR, "robotiq",
            "gripper.vendor='robotiq' and the arm in hand does report vendor 'ur', but it exposes "
            "no controller address: reading arm.config.ur.ip off it found nothing. The Robotiq is "
            "reached over a socket on that arm's controller, and robot.ur.ip in this tree "
            "('10.9.9.9') describes whatever the config names, not the arm that was handed in.",
            "Hand in an arm built from a config tree (URRobotArm keeps the one it was constructed "
            "with), or supply no arm at all and set robot.vendor: ur so this builds the arm from "
            "this tree and both halves come from the same place.",
        ),
    ),
    _Case(
        name="vacuum on a UR arm",
        tree={**_UR, "gripper": {"vendor": "vacuum", "vacuum": {"vacuum_output_pin": 2}}},
        handle=None, gated=True,
        arm="URRobotArm", gripper="VacuumGripper",
        attributes={"_pin": 2}, io_is_arm=True,
    ),
    _Case(
        name="vacuum on a dummy arm",
        tree={"vendor": "dummy", "gripper": {"vendor": "vacuum", **_WIDTHS}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="NullGripper",
        attributes=dict(_WIDTHS),
        substitution=_Substituted(
            SubstitutionReason.VACUUM_NEEDS_DIGITAL_IO, "vacuum",
            "gripper.vendor='vacuum' but the arm in hand reports vendor 'dummy' and does not "
            "advertise SupportsDigitalIO. Suction switches the controller's digital I/O, which "
            "only the real UR driver exposes.",
            "Set robot.vendor: ur for a real suction cell, or gripper.vendor: none.",
        ),
    ),
    _Case(
        name="jaw_io on a UR arm",
        tree={**_UR, "gripper": {"vendor": "jaw_io", "jaw_io": {"close_output_pin": 3}}},
        handle=None, gated=True,
        arm="URRobotArm", gripper="JawIOGripper",
        attributes={"_close_pin": 3}, io_is_arm=True,
    ),
    _Case(
        name="jaw_io on a dummy arm",
        tree={"vendor": "dummy", "gripper": {"vendor": "jaw_io", **_WIDTHS}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="NullGripper",
        attributes=dict(_WIDTHS),
        substitution=_Substituted(
            SubstitutionReason.JAW_IO_NEEDS_DIGITAL_IO, "jaw_io",
            "gripper.vendor='jaw_io' but the arm in hand reports vendor 'dummy' and does not "
            "advertise SupportsDigitalIO. A solenoid jaw switches the controller's digital I/O, "
            "which only the real UR driver exposes.",
            "Set robot.vendor: ur for a real I/O jaw cell, or gripper.vendor: none.",
        ),
    ),
    _Case(
        name="onrobot",
        tree={"vendor": "dummy",
              "gripper": {"vendor": "onrobot",
                          "onrobot": {"host": "10.0.0.5", "port": 5020, "unit_id": 66}}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="OnRobotGripper",
        attributes={"host": "10.0.0.5", "port": 5020, "unit": 66},
    ),
    _Case(
        name="franka_hand, the branch with no driver",
        tree={"vendor": "dummy", "gripper": {"vendor": "franka_hand", **_WIDTHS}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="NullGripper",
        attributes=dict(_WIDTHS),
        substitution=_Substituted(
            SubstitutionReason.NO_DRIVER, "franka_hand",
            "gripper.vendor='franka_hand' is a recognised name with no driver in this repo.",
            "Use robotiq, vacuum, jaw_io or onrobot, or write the driver and register it in "
            "grippers/registry.py.",
        ),
    ),
    _Case(
        name="none, declared on purpose",
        tree={"vendor": "dummy", "gripper": {"vendor": "none", **_WIDTHS}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="NullGripper",
        attributes=dict(_WIDTHS),
    ),
    _Case(
        name="dummy",
        tree={"vendor": "dummy", "gripper": {"vendor": "dummy"}},
        handle=None, gated=False,
        arm="DummyRobotArm", gripper="DummyGripper",
    ),
)


class TheBuilderTableTests(unittest.TestCase):
    """Every gripper branch, read through the service and through the builder, against literals."""

    def _check(self, case: _Case, arm: object, gripper: object) -> None:
        self.assertEqual(type(arm).__name__, case.arm)
        self.assertEqual(type(gripper).__name__, case.gripper)
        for name, value in case.attributes.items():
            self.assertEqual(getattr(gripper, name), value, name)
        if case.io_is_arm:
            self.assertIs(getattr(gripper, "_io"), arm)
        substitution = getattr(gripper, "substitution", None)
        if case.substitution is None:
            self.assertIsNone(substitution)
            return
        assert substitution is not None, "the cell degraded and recorded no reason"
        expected = case.substitution
        self.assertIs(substitution.reason, expected.reason)
        self.assertEqual(substitution.requested, expected.requested)
        self.assertEqual(substitution.detail, expected.detail)
        self.assertEqual(substitution.fix, expected.fix)

    @staticmethod
    def _gate(case: _Case) -> contextlib.AbstractContextManager[object]:
        return patch(_READY) if case.gated else contextlib.nullcontext()

    def test_the_service_builds_the_table(self) -> None:
        for case in _TABLE:
            with self.subTest(case.name):
                calculator, perception = _calc_and_perception()
                with self._gate(case):
                    service = RuntimePickService.from_robot_config(
                        RobotConfig(**case.tree),
                        calculator=calculator,  # type: ignore[arg-type]
                        perception=perception,  # type: ignore[arg-type]
                        arm=case.handle() if case.handle else None,  # type: ignore[arg-type]
                    )
                self._check(case, service.orchestrator.arm, service.orchestrator.gripper)

    def test_the_builder_builds_the_same_table(self) -> None:
        from src.robot.execution.robot_parts import build_gripper, resolve_arm

        for case in _TABLE:
            with self.subTest(case.name):
                config = RobotConfig(**case.tree)
                with self._gate(case):
                    arm = resolve_arm(
                        config, arm=case.handle() if case.handle else None,  # type: ignore[arg-type]
                    )
                    gripper = build_gripper(config, arm=arm)
                self._check(case, arm, gripper)


class TheSubstitutionNamesTheArmInHandTests(unittest.TestCase):
    """A supplied arm replaces the arm the tree describes, so a detail read off ``robot.vendor``
    named an arm that was not there. The Robotiq detail already named the arm in hand."""

    def test_a_dummy_handle_under_a_ur_tree_is_named_in_the_detail(self) -> None:
        from src.robot.execution.robot_parts import build_gripper

        for vendor, reason in (("vacuum", SubstitutionReason.VACUUM_NEEDS_DIGITAL_IO),
                               ("jaw_io", SubstitutionReason.JAW_IO_NEEDS_DIGITAL_IO)):
            with self.subTest(vendor):
                gripper = build_gripper(RobotConfig(**{**_UR, "gripper": {"vendor": vendor}}),
                                        arm=create_arm(RobotVendor.DUMMY))
                substitution = getattr(gripper, "substitution", None)
                assert substitution is not None, "the cell degraded and recorded no reason"
                self.assertIs(substitution.reason, reason)
                self.assertIn("reports vendor 'dummy'", substitution.detail)
                self.assertNotIn("'ur'", substitution.detail)

    def test_the_warning_names_the_builder(self) -> None:
        from src.robot.execution.robot_parts import build_gripper

        with self.assertLogs("src.robot.execution.robot_parts", level="WARNING") as logs:
            build_gripper(RobotConfig(vendor="dummy", gripper={"vendor": "vacuum"}),
                          arm=create_arm(RobotVendor.DUMMY))
        self.assertEqual(len(logs.records), 1)
        message = logs.records[0].getMessage()
        self.assertTrue(message.startswith("build_gripper: gripper.vendor='vacuum'"), message)


_REPO = Path(__file__).resolve().parents[1]

#: What building a robot must not pay for. The pick service reaches every one of these.
_BUILDER_MUST_NOT_LOAD: tuple[str, ...] = (
    "torch",
    "src.robot.grasping",
    "src.robot.perception",
    "src.models",
    "src.camera",
    "src.robot.execution.runtime_pick",
)


def _loaded_after(script: str, probe: tuple[str, ...]) -> list[str]:
    """Which names of ``probe`` a fresh interpreter holds after ``script``, submodules included.

    A subprocess, because this process has almost certainly imported the grasping stack already.
    """
    code = "\n".join((
        "import json, sys",
        script,
        f"probe = {list(probe)!r}",
        "held = [p for p in probe if any(m == p or m.startswith(p + '.') for m in sys.modules)]",
        "print(json.dumps(sorted(held)))",
    ))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            check=False, cwd=str(_REPO))
    if result.returncode != 0:
        raise AssertionError(f"the probe exited {result.returncode}:\n{result.stderr[-2000:]}")
    return list(json.loads(result.stdout.strip().splitlines()[-1]))


class TheBuilderImportsNoGraspingStackTests(unittest.TestCase):

    def test_the_builder_imports_no_grasping_stack(self) -> None:
        loaded = _loaded_after("import src.robot.execution.robot_parts",
                               _BUILDER_MUST_NOT_LOAD)
        self.assertEqual(loaded, [], "building an arm and a gripper pulls in the pick service")

    def test_the_probe_sees_the_grasping_stack_where_it_is(self) -> None:
        """The self-failing control: ``runtime_pick`` imports the grasping stack at module top, so a
        probe that reported nothing here would be looking in the wrong place."""
        loaded = _loaded_after("import src.robot.execution.runtime_pick",
                               _BUILDER_MUST_NOT_LOAD)
        self.assertIn("src.robot.grasping", loaded)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
