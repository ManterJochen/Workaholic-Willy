"""Adding a hand adds its retract rows and moves nothing, and two hands' guards agree where they must (lane C5c).

``scripts/trial/retract_checks.py`` is a trial instrument. Each check here has the control that shows it can say no.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_TABLE = _REPO / "src" / "robot" / "safety" / "planning" / "robot" / "ur_retract.yaml"
_PLUS_Z = (0.0, 0.0, 0.0, 1.0)


def _tool():
    name = "_trial_retract_checks"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "retract_checks.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class TableDiffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.table = yaml.safe_load(_TABLE.read_text(encoding="utf-8"))

    def _synthetic_row(self) -> dict:
        row = copy.deepcopy(next(row for row in self.table["retracts"] if row["arm"] == "ur10e"))
        row["hand"] = "acme_dims"
        return row

    def test_a_table_diff_names_a_moved_row_and_an_undeclared_one(self) -> None:
        tool = _tool()
        self.assertEqual(tool.table_diff(self.table, self.table, added=set()).exit_code, 0)
        grown = copy.deepcopy(self.table)
        grown["retracts"].append(self._synthetic_row())
        plate = round(float(self._synthetic_row().get("plate_mm", 0.0)), 6)
        self.assertEqual(tool.table_diff(self.table, grown, added={("ur10e", "acme_dims", plate)}).exit_code, 0)

    def test_the_diff_says_no_for_a_moved_row_and_for_an_undeclared_one(self) -> None:
        """⭐ THE CONTROL: a retract moved by a micro radian, and an added row nobody declared, are each named."""
        tool = _tool()
        moved = copy.deepcopy(self.table)
        moved["retracts"][0]["retract"][0] = float(moved["retracts"][0]["retract"][0]) + 1e-6
        report = tool.table_diff(self.table, moved, added=set())
        self.assertEqual(report.exit_code, 1)
        self.assertIn(self.table["retracts"][0]["arm"], report.render())
        grown = copy.deepcopy(self.table)
        grown["retracts"].append(self._synthetic_row())
        report = tool.table_diff(self.table, grown, added=set())
        self.assertEqual(report.exit_code, 1)
        self.assertIn("acme_dims", " ".join(report.undeclared))


class GuardAgreementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.robot.safety.planning.environment import import_collision_engine

        if import_collision_engine()[1] is None:
            raise unittest.SkipTest("no collision engine: neither Coal nor python-fcl imports in this interpreter")

    def test_one_hand_agrees_with_itself_and_not_with_another(self) -> None:
        tool = _tool()
        same = tool.guard_agree(arm="ur5e", hand="robotiq_hande", reference="robotiq_hande", coupling_mm=0.0,
                                rotation_xyzw=_PLUS_Z, random_n=20, tolerance_mm=1e-3)
        self.assertEqual(same.max_abs_diff_mm, 0.0)
        # ⭐ THE CONTROL: another hand on the same poses is told apart.
        other = tool.guard_agree(arm="ur5e", hand="schunk_egu50", reference="robotiq_hande", coupling_mm=0.0,
                                 rotation_xyzw=_PLUS_Z, random_n=20, tolerance_mm=1e-3)
        self.assertGreater(other.max_abs_diff_mm, 1.0)
        self.assertEqual(other.exit_code, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
