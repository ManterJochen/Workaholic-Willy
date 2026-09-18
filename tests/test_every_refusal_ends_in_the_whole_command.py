"""Every retract and evidence refusal ends in the whole command that judges or measures what the cell asked for (C3h).

The refusals said "choose_ur_retract.py <arm>" and "Measure it with matrix_gate.py": pasted, the first judges the Isaac
cell's +Y placement and the second measures an undeclared frame at no plate and no payload, so a +Z customer cell got
a file that could never admit it (finding 10 of the audit of 2026-09-17). Each command here is parsed back by the
script's own parser and compared with what the cell asked.
"""

from __future__ import annotations

import ast
import importlib.util
import shlex
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[1]
_CUROBO = _REPO / "scripts" / "curobo"


def _script(name: str) -> Any:
    if str(_CUROBO) not in sys.path:
        sys.path.insert(0, str(_CUROBO))
    spec = importlib.util.spec_from_file_location(f"_parse_back_{name}", _CUROBO / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _after(text: str, script: str) -> list[str]:
    tail = text.split(script, 1)[1]
    return shlex.split(tail.strip().rstrip("."))


class ARetractRefusalParsesBackTests(unittest.TestCase):
    def _table(self, body: str) -> Path:
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = folder / "ur_retract.yaml"
        path.write_text("rule:\n  placement: +Y+X\n  planner_margin_mm: 4.0\n  pairs_with_no_retract: []\nretracts:\n" + body,
                        encoding="utf-8")
        return path

    def _asked(self, text: str) -> Any:
        return _script("choose_ur_retract").parser().parse_args(_after(text, "choose_ur_retract.py"))

    def test_every_retract_refusal_parses_back_to_the_asked_key(self) -> None:
        from src.robot.safety.planning.robot.retract_table import RetractMissing, read_retract

        row = "  - arm: ur5e\n    hand: robotiq_hande\n    plate_mm: 20.0\n    planner_margin_mm: 4.0\n    retract: [0, 0, 0, 0, 0, 0]\n"
        cases = {
            "no row": (self._table(""), ("ur5e", "robotiq_hande", 20.0, 4.0, "+Z+X")),
            "another placement": (self._table(row), ("ur5e", "robotiq_hande", 20.0, 4.0, "+Z+X")),
            "another margin": (self._table(row), ("ur5e", "robotiq_hande", 20.0, 6.0, "+Y+X")),
        }
        for label, (table, (arm, hand, plate, margin, placement)) in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(RetractMissing) as caught:
                    read_retract(arm, hand, plate, margin, placement=placement, table_path=table)
                args = self._asked(str(caught.exception))
                self.assertEqual(args.arms, [arm])
                self.assertEqual(args.hand, [hand])
                self.assertEqual(args.plate_mm, [plate])
                self.assertEqual(args.planner_margin_mm, margin)
                from src.robot.safety.planning._hand_placement import HandPlacement

                turned = HandPlacement.from_quaternion_xyzw(tuple(args.tool_rotation_xyzw))
                self.assertEqual(f"{turned.approach}{turned.closing}", placement)


class AnEvidenceRefusalParsesBackTests(unittest.TestCase):
    _PLATE = [{"name": "adapter", "thickness_mm": 20.0, "cross_section_mm": [31.5, 31.5]}]

    def _cell(self) -> Any:
        from src.config.schema.robot import RobotConfig

        return RobotConfig.model_validate({
            "vendor": "ur", "ur": {"model": "ur5e", "motion_planner": "curobo"},
            "gripper": {"model": "robotiq_hande", "coupling_plates": self._PLATE,
                        "tool_frame": {"source": "polyscope", "offset_mm": [0.0, 0.0, 175.75],
                                       "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}},
            "safety": {"self_collision": {"kinematics_model": "ur5e", "planner_margin_mm": 4.0},
                       "planning_world": {"enabled": True, "support_plane": {"height_mm": 0.0},
                                          "payload": {"enabled": True, "sphere_slots": 4, "length_mm": 60.0}}},
        })

    def test_a_missing_evidence_file_names_the_gate_command_for_this_cell(self) -> None:
        from src.robot.safety.planning.evidence import desk_evidence_refusal, evidence_path

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        refused, _ = desk_evidence_refusal(self._cell(), evidence_dir=empty)
        assert refused is not None
        args = _script("matrix_gate").parser().parse_args(_after(refused, "matrix_gate.py"))
        self.assertEqual((args.arm, args.hand), ("ur5e", "robotiq_hande"))
        self.assertEqual(args.plate, self._PLATE)
        self.assertEqual((args.planner_margin_mm, args.guard_margin_mm, args.attach), (4.0, 10.0, 4))
        self.assertEqual(list(args.tool_rotation_xyzw), [0.0, 0.0, 0.0, 1.0])
        self.assertTrue(args.write)
        looked_for = evidence_path(arm="ur5e", hand="robotiq_hande", coupling_mm=20.0, approach="+Z", closing="+X",
                                   planner_margin_mm=4.0, attach_spheres=4, evidence_dir=empty)
        self.assertIn(looked_for.name, refused)

    def test_the_desk_fix_names_the_command(self) -> None:
        from src.robot.execution.real_cell.preflight import run_config_preflight

        empty = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (row,) = [c for c in run_config_preflight(self._cell(), curobo_available=True, evidence_dir=empty).checks
                  if c.name == "planner margin"]
        self.assertIn("matrix_gate.py --arm", row.detail)
        self.assertIn("command", row.fix)

    def test_a_hand_carries_its_plates_in_order(self) -> None:
        from src.robot.safety.planning.hand import planner_hand

        self.assertEqual(planner_hand(self._cell()).plates, (("adapter", 20.0, (31.5, 31.5)),))


class EveryRetractReadNamesItsPlacementTests(unittest.TestCase):
    def test_every_read_retract_call_names_its_placement(self) -> None:
        missing = []
        for root in (_REPO / "src", _REPO / "scripts"):
            for path in root.rglob("*.py"):
                try:
                    tree = ast.parse(path.read_text(encoding="utf-8"))
                except (SyntaxError, UnicodeDecodeError):
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call) and getattr(node.func, "id", getattr(node.func, "attr", "")) == "read_retract":
                        if not any(keyword.arg == "placement" for keyword in node.keywords):
                            missing.append(f"{path.relative_to(_REPO)}:{node.lineno}")
        self.assertEqual(missing, [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
