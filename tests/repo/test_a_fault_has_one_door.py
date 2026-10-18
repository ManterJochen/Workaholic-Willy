"""``acknowledge_fault`` is reached from one place in this repository's code: ``src/robot/faults`` (P0i i4).

The owner's rule of 2026-09-19 (``tests/robot/drivers/test_robot_boundaries.py``) forbids a call from the execution, grasping and
safety layers and from the console; the owner's answer of 2026-09-26 gave the acknowledgement one door, a CLI verb a
person runs at the cell. This holds the rest of the code to that door: every reference to the method and every lookup
by name, anywhere under ``src``, ``api``, ``willy``, ``datagen`` and ``examples``, sits in that package. A reference
counts, not only a call (found by review, 2026-09-26): ``ack = arm.acknowledge_fault`` and a call through ``ack`` later
passed a scan of calls unseen. The URSim probe under ``scripts/`` is a bench tool with a person at the pendant, and
outside this scan.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SCANNED = ("src", "api", "willy", "datagen", "examples")
_DOOR = "src/robot/faults/"


def _references(tree: ast.AST, rel: str) -> list[str]:
    """Where ``tree`` reaches ``acknowledge_fault``: an attribute of that name, read in any way, or a lookup by name."""
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == "acknowledge_fault":
            found.append(f"{rel}:{node.lineno}")
        if (isinstance(node, ast.Call) and getattr(node.func, "id", "") in ("getattr", "hasattr")
                and len(node.args) > 1 and isinstance(node.args[1], ast.Constant)
                and node.args[1].value == "acknowledge_fault"):
            found.append(f"{rel}:{node.lineno} (by name)")
    return found


def _callers() -> tuple[list[str], int]:
    callers: list[str] = []
    files = 0
    for top in _SCANNED:
        for path in sorted((_REPO / top).rglob("*.py")):
            files += 1
            callers += _references(ast.parse(path.read_text(encoding="utf-8")), path.relative_to(_REPO).as_posix())
    return callers, files


class AFaultHasOneDoorTests(unittest.TestCase):
    def test_every_reference_is_the_faults_package(self) -> None:
        callers, files = _callers()
        self.assertGreater(files, 500, "the walk found almost nothing, so it proves nothing")
        self.assertTrue(callers, "the door itself was not found, so the scan would pass over any caller")
        self.assertEqual([caller for caller in callers if not caller.startswith(_DOOR)], [])

    def test_a_reference_that_is_no_call_is_found(self) -> None:
        """THE CONTROL: a bound method taken first and called later reaches the fault as a call does."""
        self.assertEqual(_references(ast.parse("ack = arm.acknowledge_fault\nack(fault)\n"), "x.py"), ["x.py:1"])
        self.assertEqual(_references(ast.parse('getattr(arm, "acknowledge_fault")(fault)\n'), "x.py"),
                         ["x.py:1 (by name)"])


if __name__ == "__main__":
    unittest.main()
