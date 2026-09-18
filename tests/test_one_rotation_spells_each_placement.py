"""One quaternion spells each placement, and the gate and the sweep take it in one spelling (customer chain lane C3a, C3b).

A refusal that knows only ``+Z+X`` could not print a rotation a customer can paste, and the gate took the rotation as
one comma string while the chooser and the descriptor check took four numbers (finding 11 of the audit of 2026-09-17).
"""

from __future__ import annotations

import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

_REPO = Path(__file__).resolve().parents[1]
_CUROBO = _REPO / "scripts" / "curobo"


def _by_path(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(module)
    return module


class OneQuaternionPerPlacementTests(unittest.TestCase):
    def test_every_admitted_placement_round_trips(self) -> None:
        from src.robot.safety.planning._hand_placement import HandPlacement, placement_quaternion_xyzw

        for approach in ("+Y", "+Z"):
            for closing in ("+X", "-X", "+Y", "-Y", "+Z", "-Z"):
                if closing[1] == approach[1]:
                    continue
                with self.subTest(approach=approach, closing=closing):
                    q = placement_quaternion_xyzw(approach, closing)
                    placement = HandPlacement.from_quaternion_xyzw(q)
                    self.assertEqual((placement.approach, placement.closing), (approach, closing))
                    self.assertEqual(placement.residual_deg, 0.0)
                    self.assertEqual(placement.quaternion_xyzw, q)
                    self.assertGreaterEqual(q[3], 0.0)

    def test_the_committed_spellings_come_out_exactly(self) -> None:
        from src.robot.safety.planning._hand_placement import placement_quaternion_xyzw

        self.assertEqual(repr(placement_quaternion_xyzw("+Z", "+X")), repr((0.0, 0.0, 0.0, 1.0)))
        sim = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)
        self.assertEqual(repr(placement_quaternion_xyzw("+Y", "+X")), repr(sim))
        self.assertEqual(sim[0], -math.sqrt(0.5))

    def test_a_placement_nobody_can_declare_is_refused(self) -> None:
        from src.robot.safety.planning._hand_placement import PlacementRefused, RefusalKind, placement_quaternion_xyzw

        with self.assertRaises(PlacementRefused) as caught:
            placement_quaternion_xyzw("-Z", "+X")
        self.assertEqual(caught.exception.kind, RefusalKind.APPROACH_NOT_ADMITTED)
        with self.assertRaises(PlacementRefused):
            placement_quaternion_xyzw("+Z", "+Z")


class OneRotationSpellingTests(unittest.TestCase):
    def _gate(self) -> Any:
        return _by_path("_gate_for_spelling", _CUROBO / "matrix_gate.py")

    def test_the_gate_takes_the_chooser_spelling(self) -> None:
        gate = self._gate()
        seen: dict[str, Any] = {}

        class _Result:
            path = Path("x.json")

            def render(self) -> str:
                return "x: b1 ("

        def recorder(**kwargs: Any) -> Any:
            seen.update(kwargs)
            return _Result()

        with mock.patch.object(gate, "measure", recorder):
            code = gate.main(["--arm", "ur5e", "--hand", "robotiq_2f85", "--planner-margin-mm", "4",
                              "--tool-rotation-xyzw", "0", "0", "0", "1"])
        self.assertEqual(code, 0)
        self.assertEqual(seen["rotation_xyzw"], [0.0, 0.0, 0.0, 1.0])

    def test_the_comma_spelling_is_refused(self) -> None:
        """⭐ THE CONTROL: the old spelling does not quietly still work beside the new one."""
        gate = self._gate()
        with mock.patch.object(gate, "measure", side_effect=AssertionError("measured")), \
                self.assertRaises(SystemExit) as caught:
            gate.main(["--arm", "ur5e", "--hand", "robotiq_2f85", "--planner-margin-mm", "4",
                       "--tool-rotation-xyzw", "0,0,0,1"])
        self.assertEqual(caught.exception.code, 2)

    def test_every_command_the_sweep_builds_reaches_the_gate_with_its_rows_rotation(self) -> None:
        sweep = _by_path("_sweep_for_spelling", _CUROBO / "matrix_sweep.py")
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        table = folder / "ur_retract.yaml"
        table.write_text(
            "rule:\n  placement: +Y+X\nretracts:\n"
            "- arm: ur5e\n  hand: robotiq_2f85\n  plate_mm: 0.0\n  planner_margin_mm: 4.0\n  retract: [0, 0, 0, 0, 0, 0]\n"
            "- arm: ur5e\n  hand: robotiq_2f85\n  plate_mm: 0.0\n  planner_margin_mm: 4.0\n  placement: +Z+X\n"
            "  tool_rotation_xyzw: [0.0, 0.0, 0.0, 1.0]\n  retract: [0, 0, 0, 0, 0, 0]\n",
            encoding="utf-8",
        )
        commands: list[list[str]] = []

        class _Done:
            returncode = 0
            stdout = "[gate] x: b1 (\n"
            stderr = ""

        def capture(command: list[str], **_: Any) -> Any:
            commands.append(command)
            return _Done()

        with mock.patch.object(sweep.subprocess, "run", capture), mock.patch("builtins.print"):
            sweep.main(["--table", str(table)])
        self.assertEqual(len(commands), 2)
        gate = self._gate()
        reached: list[Any] = []

        class _Result:
            path = Path("x.json")

            def render(self) -> str:
                return "x"

        def recorder(**kwargs: Any) -> Any:
            reached.append(kwargs["rotation_xyzw"])
            return _Result()

        with mock.patch.object(gate, "measure", recorder), mock.patch("builtins.print"):
            for command in commands:
                self.assertEqual(gate.main(command[2:]), 0)
        self.assertEqual(sorted(reached, key=lambda r: r is not None), [None, [0.0, 0.0, 0.0, 1.0]])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
