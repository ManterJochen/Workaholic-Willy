"""P6 step 1: per-robot calibration artifact directories (destructive-by-omission if flat)."""

from __future__ import annotations

import unittest

from src.willy_sim.calibration.paths import CALIBRATION_ROOT, calibration_dir


class CalibrationDirTests(unittest.TestCase):
    def test_each_robot_gets_its_own_subdir(self) -> None:
        self.assertEqual(calibration_dir("ur5e"), CALIBRATION_ROOT / "ur5e")
        self.assertEqual(calibration_dir("ur3e"), CALIBRATION_ROOT / "ur3e")
        self.assertNotEqual(calibration_dir("ur5e"), calibration_dir("ur3e"))

    def test_none_or_empty_falls_back_to_the_shared_root(self) -> None:
        # A mis-wired cell degrades to the legacy flat layout, never a dir literally named "None".
        self.assertEqual(calibration_dir(None), CALIBRATION_ROOT)
        self.assertEqual(calibration_dir(""), CALIBRATION_ROOT)
        self.assertEqual(calibration_dir("  "), CALIBRATION_ROOT)

    def test_root_ends_at_logs_calibration(self) -> None:
        self.assertEqual(CALIBRATION_ROOT.parts[-2:], ("logs", "calibration"))


if __name__ == "__main__":
    unittest.main()
