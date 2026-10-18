"""Rule II, the gripper half (P0g slice 8): no code outside the grippers package branches on a gripper vendor by hand.

The build, the desk check, the console and the bench each branched on ``robot.gripper.vendor``; a fifth gripper meant
four dispatches to find, and one that was forgotten built a ``NullGripper`` that reported every pick a success. Since
slice 8 each reads the vendor's :class:`~src.robot.grippers.spec.GripperDriver`, and this holds it there: no
``GripperVendor.<MEMBER>`` and no comparison against a gripper vendor's value outside the enum, the config schemas and
the grippers package, which is the registry. The value set is generated from the enums, so a new vendor is covered
without editing this file. Values a robot vendor shares (``dummy``) and ``none`` count only where the comparison names
a gripper, because a robot vendor comparison is the other half of rule II and lands later.
"""

from __future__ import annotations

import ast
import functools
import unittest
from pathlib import Path

from src.robot.core import GripperVendor, RobotVendor

_ROOT = Path(__file__).resolve().parents[3]
_SCANNED = ("src", "api", "scripts", "examples", "willy", "datagen")
#: Where a gripper vendor may be named: the package that is the registry, the enum, and the config schemas.
_HOME = ("src/robot/grippers/", "src/robot/core/gripper_vendor.py", "src/config/schema/")
#: Counted exceptions, each with its reason. The list may only shrink.
_EXEMPT: dict[str, str] = {
    "src/config/hand_numbers.py":
        "the hand registry in the config layer, which may not import the drivers: it refuses a real UR cell whose "
        "hand has no actuating driver before any driver loads, and NONE and DUMMY are the two that actuate nothing",
}
#: Gripper vendor values no robot vendor shares, which name a gripper wherever they are compared.
_OWN = frozenset(v.value for v in GripperVendor) - frozenset(v.value for v in RobotVendor) - {"none"}
_ALL = frozenset(v.value for v in GripperVendor)


def _findings(source: str) -> list[str]:
    """Every ``GripperVendor.<MEMBER>`` and every comparison against a gripper vendor's value in ``source``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "GripperVendor"
                and node.attr.isupper()):
            found.append(f"{node.lineno}: GripperVendor.{node.attr}")
        if isinstance(node, ast.Compare):
            sides = [node.left, *node.comparators]
            values = {c.value for side in sides for c in ast.walk(side)
                      if isinstance(c, ast.Constant) and isinstance(c.value, str)} & _ALL
            if not values:
                continue
            names_a_gripper = any("gripper" in ast.unparse(side).lower() for side in sides)
            if values & _OWN or names_a_gripper:
                found.append(f"{node.lineno}: {ast.unparse(node)[:80]}")
    return found


@functools.lru_cache(maxsize=1)
def _tree() -> dict[str, list[str]]:
    offenders: dict[str, list[str]] = {}
    for folder in _SCANNED:
        for path in sorted((_ROOT / folder).rglob("*.py")):
            rel = path.relative_to(_ROOT).as_posix()
            if "__pycache__" in rel or rel.startswith(_HOME):
                continue
            found = _findings(path.read_text(encoding="utf-8"))
            if found:
                offenders[rel] = found
    return offenders


class NoGripperVendorIsDispatchedByHandTests(unittest.TestCase):
    def test_nothing_outside_the_registry_branches_on_a_gripper_vendor(self) -> None:
        stray = {rel: found for rel, found in _tree().items() if rel not in _EXEMPT}
        self.assertEqual({}, stray, "ask the vendor's GripperDriver spec (grippers.gripper_driver) instead")

    def test_every_exemption_is_still_needed(self) -> None:
        """An exemption whose file no longer names a vendor would admit the next one there without a reason."""
        self.assertEqual(set(), set(_EXEMPT) - set(_tree()))

    def test_the_scan_finds_each_shape(self) -> None:
        """The positive control, on a synthetic source: every real target left in slice 8."""
        source = (
            "a = GripperVendor.ROBOTIQ\n"
            "b = vendor == 'jaw_io'\n"
            "c = str(cfg.gripper.vendor) in ('none', 'dummy')\n"
            "d = robot_vendor in ('sim', 'dummy')\n"
            "e = GripperVendor.from_string(name)\n"
        )
        self.assertEqual([1, 2, 3], sorted(int(f.split(":")[0]) for f in _findings(source)))

    def test_every_vendor_with_a_driver_has_a_spec_and_every_spec_a_driver(self) -> None:
        from src.robot.grippers import available_gripper_vendors, gripper_drivers

        self.assertEqual(sorted(available_gripper_vendors()), sorted(d.vendor.value for d in gripper_drivers()))


class EveryReaderAsksTheSpecTests(unittest.TestCase):
    """The lookups are live: a spec that says something else is what every reader says (a sabotage control)."""

    def _swapped(self, **changes: object) -> object:
        import dataclasses

        from src.robot.grippers import registry

        original = registry._SPECS[GripperVendor.ONROBOT]
        registry._SPECS[GripperVendor.ONROBOT] = dataclasses.replace(original, **changes)
        self.addCleanup(registry._SPECS.__setitem__, GripperVendor.ONROBOT, original)
        return original

    def _cfg(self) -> object:
        from src.config.schema.robot import RobotConfig

        return RobotConfig.model_validate({"vendor": "ur", "ur": {"ip": "192.168.1.10", "model": "ur5e"},
                                           "gripper": {"vendor": "onrobot", "model": "robotiq_2f85"}})

    def test_the_verdict_and_the_build_read_the_refusal(self) -> None:
        from src.robot.drivers.dummy.arm import DummyRobotArm
        from src.robot.execution.robot_parts import build_gripper, gripper_driver_verdict
        from src.robot.grippers import Refusal, SubstitutionReason
        from src.robot.grippers.null import NullGripper

        refusal = Refusal(SubstitutionReason.NO_DRIVER, "swapped in by the test", "put the real spec back")
        self._swapped(refuse=lambda _cfg, _facts: refusal)
        verdict = gripper_driver_verdict(self._cfg(), arm_vendor="ur", arm_has_digital_io=True,  # type: ignore[arg-type]
                                         arm_ip="192.168.1.10")
        self.assertEqual("swapped in by the test", verdict.detail)
        self.assertIsInstance(build_gripper(self._cfg(), arm=DummyRobotArm()), NullGripper)  # type: ignore[arg-type]

    def test_the_desk_and_the_console_read_the_wiring_and_the_hazards(self) -> None:
        from api.lifecycle import motion_warnings
        from src.robot.execution.real_cell import preflight as desk
        from src.robot.grippers import Hazard, Row

        self._swapped(wiring=lambda _cfg: Row("bench", "the swapped wiring", "the swapped fix"),
                      hazards=lambda _cfg, _gripper, _asked: (Hazard("the swapped hazard", "the swapped precaution"),))
        row = desk._end_effector_wiring_row(self._cfg())  # type: ignore[arg-type]
        self.assertEqual(("the swapped wiring", "the swapped fix"), (row.detail, row.fix))
        warnings = motion_warnings(self._cfg(), object())  # type: ignore[arg-type]
        self.assertEqual(["the swapped hazard"], [w.what for w in warnings])


if __name__ == "__main__":
    unittest.main()
