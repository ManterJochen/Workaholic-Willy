"""A UR arm is an arm: the gripper on a UR cell is built once, by the builder, and nowhere else.

``URRobotArm.__init__`` constructed a Robotiq ``GripperController`` from ``config.gripper`` and exposed
it as ``arm.gripper`` whatever ``gripper.vendor`` said. Nothing in the repository read the property,
``connect()`` never connected it, and a UR cell built from config held two controllers for one hand:
the builder's, which connects, and the arm's, which never did.
"""

from __future__ import annotations

import ast
import contextlib
import unittest
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.config.schema.robot import RobotConfig
from src.robot.drivers.ur.arm import URRobotArm
from src.robot.execution.runtime_pick import RuntimePickService
from src.robot.grippers.robotiq import GripperController
from tests.test_grasping_config_wiring import _calc_and_perception

_REPO = Path(__file__).resolve().parents[1]
_READY = "src.robot.drivers.doctor.require_arm_vendor_ready"
_GRIPPERS = "src.robot.grippers"


@contextlib.contextmanager
def _controllers_built() -> Iterator[list[str]]:
    """The address of every ``GripperController`` constructed inside the block, in order."""
    built: list[str] = []
    real = GripperController.__init__

    def counting(controller: GripperController, *args: Any, **kwargs: Any) -> None:
        real(controller, *args, **kwargs)
        built.append(controller.ip)

    with patch.object(GripperController, "__init__", counting):
        yield built


class TheArmBuildsNoGripperTests(unittest.TestCase):

    def test_constructing_a_ur_arm_constructs_no_gripper(self) -> None:
        with _controllers_built() as built:
            arm = URRobotArm(RobotConfig.model_validate({"vendor": "ur"}))
        self.assertEqual(built, [])
        self.assertFalse(hasattr(arm, "gripper"))

    def test_a_ur_cell_built_from_config_holds_exactly_one_controller(self) -> None:
        calculator, perception = _calc_and_perception()
        tree = {"vendor": "ur", "ur": {"ip": "10.9.9.9"}, "gripper": {"vendor": "robotiq"}}
        with _controllers_built() as built, patch(_READY):
            service = RuntimePickService.from_robot_config(
                RobotConfig.model_validate(tree),
                calculator=calculator,  # type: ignore[arg-type]
                perception=perception,  # type: ignore[arg-type]
            )
        self.assertEqual(built, ["10.9.9.9"])
        self.assertIsInstance(service.orchestrator.gripper, GripperController)


def _imports_of_grippers(path: Path) -> list[int]:
    """The line of every import in ``path`` that resolves into the grippers package, at any depth.

    Relative imports are resolved against the module's own package, so ``from ...grippers import``
    counts as well as the absolute form.
    """
    package = list(path.resolve().relative_to(_REPO).with_suffix("").parts[:-1])
    lines: list[int] = []
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom):
            base = package[:len(package) - node.level + 1] if node.level else []
            module = ".".join(base + ([node.module] if node.module else []))
            names = [module] + [f"{module}.{alias.name}" for alias in node.names]
        elif isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        else:
            continue
        if any(name == _GRIPPERS or name.startswith(_GRIPPERS + ".") for name in names):
            lines.append(node.lineno)
    return lines


class NoUrDriverModuleImportsAGripperTests(unittest.TestCase):

    def test_no_module_under_drivers_ur_imports_the_grippers_package(self) -> None:
        modules = sorted((_REPO / "src" / "robot" / "drivers" / "ur").rglob("*.py"))
        self.assertIn("arm.py", {path.name for path in modules}, "the scan found no UR driver")
        offenders = {path.relative_to(_REPO).as_posix(): lines for path in modules
                     if (lines := _imports_of_grippers(path))}
        self.assertEqual(offenders, {})

    def test_the_scan_finds_the_builders_function_local_import(self) -> None:
        """The self-failing control: the builder imports the grippers package inside a function, so
        a scan that found nothing there would only be looking at module top level."""
        builder = _REPO / "src" / "robot" / "execution" / "robot_parts.py"
        self.assertNotEqual(_imports_of_grippers(builder), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
