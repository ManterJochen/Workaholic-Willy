"""Cell probes for the customer chain trial (customer chain lane C5d).

``scripts/trial/cell_probes.py`` is a trial instrument. ``derived`` holds a loaded chain's hand keys to the registry, and
``build-arm`` builds the UR driver from a chain without connecting.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "config"


def _tool():
    name = "_trial_cell_probes"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _REPO / "scripts" / "trial" / "cell_probes.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class DerivedTests(unittest.TestCase):
    def test_derived_reads_the_hande_chain_as_its_registry_numbers(self) -> None:
        report = _tool().derived("hande")
        self.assertEqual(report.exit_code, 0, report.render())
        self.assertEqual((report.hand, report.checked), ("robotiq_hande", 13))

    def test_a_chain_stating_another_number_does_not_load(self) -> None:
        """⭐ THE CONTROL: the probe cannot agree with a number the loader let through, because it refuses it."""
        root = Path(self.enterContext(tempfile.TemporaryDirectory())) / "data"
        shutil.copytree(_DATA, root)
        (root / "robot" / "robot.trial.yaml").write_text(
            "robot:\n  gripper:\n    model: robotiq_hande\n    max_width_mm: 85.0\n", encoding="utf-8")
        report = _tool().derived("trial", data_dir=str(root))
        self.assertEqual(report.exit_code, 1)
        for part in ("max_width_mm", "85.0", "49.99"):
            self.assertIn(part, report.render())


class BuildArmTests(unittest.TestCase):
    def test_build_arm_builds_a_chain_that_names_its_hand(self) -> None:
        code, said = _tool().build_arm("ursim")
        self.assertEqual(code, 0, said)
        self.assertIn("hand robotiq_2f85", said)

    def test_build_arm_refuses_a_chain_that_names_no_hand(self) -> None:
        """⭐ THE CONTROL: the base tree names no hand on an exact mesh guard, and nothing is built with a default."""
        code, said = _tool().build_arm("")
        self.assertEqual(code, 1, said)
        self.assertIn("robot.gripper.model is unset", said)
        self.assertNotIn("built", said)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
