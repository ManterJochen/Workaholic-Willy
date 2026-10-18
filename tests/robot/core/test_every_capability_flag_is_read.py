"""Every flag a RobotCapabilities states is read by something (P0i i6).

Owner decision 6 (2026-09-26) and its follow-up the same day: ``supports_joint_move``, ``supports_linear_move``,
``has_force_control`` and ``has_native_ik`` were set by every driver and read by nothing in ``src``, ``api``,
``scripts``, ``examples``, ``willy`` or ``datagen``. A flag nothing reads is a claim nobody checks, and a Franka or KUKA driver would
have had to pick a value with no consequence. What an arm offers beyond ``RobotArm`` is asked through its Protocols
(``SupportsForceTorque`` for the force readings, ``SupportsFreedrive``, ``SupportsDigitalIO``, ``KeepsLines``).

The flags are the boolean fields. The identity fields (``vendor``, ``model``, ``dof``, ``controller``) are read by the
lock and the registries under names too common for a scan to tell apart, so they are not asked here. The scan counts a
name, not an owner: a same-named attribute read on another object counts too (the arm selection's
``owns_a_controller``), which is enough for what a dead flag fails, a name read nowhere at all.
"""

from __future__ import annotations

import ast
import unittest
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

from src.robot.core.capabilities import RobotCapabilities

_REPO = Path(__file__).resolve().parents[3]
_ROOTS = ("src", "api", "scripts", "examples", "willy", "datagen")
_DECLARED_IN = _REPO / "src" / "robot" / "core" / "capabilities.py"


def _flags() -> list[str]:
    hints = get_type_hints(RobotCapabilities)
    return [f.name for f in fields(RobotCapabilities) if hints[f.name] is bool]


def _read_in(source: str, names: set[str]) -> set[str]:
    """The names in ``names`` that ``source`` reads as an attribute. A keyword argument states a value and an
    assignment stores one; neither is a read."""
    return {node.attr for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load) and node.attr in names}


def _read_anywhere(names: set[str]) -> set[str]:
    read: set[str] = set()
    for root in _ROOTS:
        for path in (_REPO / root).rglob("*.py"):
            if path == _DECLARED_IN or "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8-sig")
            if any(name in text for name in names - read):
                read |= _read_in(text, names)
    return read


class EveryCapabilityFlagIsReadTests(unittest.TestCase):
    def test_every_flag_is_read_by_something(self) -> None:
        flags = set(_flags())
        self.assertTrue(flags, "RobotCapabilities states no boolean flag, so this scan would ask nothing")
        unread = sorted(flags - _read_anywhere(flags))
        self.assertEqual(unread, [], f"flags set by drivers and read by nothing: {unread}. Delete the flag, or ask "
                                     "the question through a Protocol the arm implements")

    def test_the_scan_counts_a_read_and_not_a_stated_or_stored_value(self) -> None:
        names = {"flag_read", "flag_stated", "flag_replaced", "flag_stored"}
        source = ("caps = RobotCapabilities(flag_stated=True)\n"
                  "other = replace(caps, flag_replaced=False)\n"
                  "caps.flag_stored = True\n"
                  "if arm.capabilities.flag_read:\n"
                  "    pass\n")
        self.assertEqual(_read_in(source, names), {"flag_read"})


if __name__ == "__main__":
    unittest.main()
