"""IsaacGripper + GripperProfile — Isaac-free unit tests (run everywhere).

The on-box behaviour (real 2F-85 open/close/set_width) is validated separately on the
workstation; here we cover the vendor-neutral surface: Protocol conformance, the calibrated
Robotiq 2F-85 profile interpolation, and the pure-Python mock path.
"""

from __future__ import annotations

import unittest

from src.robot.core import Gripper, RobotConnectionError
from src.robot.drivers.sim import IsaacSimSession, SimRobotConfig
from src.robot.grippers.sim import (
    ROBOTIQ_2F85_PROFILE,
    SCHUNK_EGU50_PROFILE,
    IsaacGripper,
)


class GripperProfileTests(unittest.TestCase):
    def test_robotiq_2f85_range_and_interpolation(self) -> None:
        p = ROBOTIQ_2F85_PROFILE
        self.assertEqual(p.driven_joint, "finger_joint")
        self.assertAlmostEqual(p.min_width_mm, 2.2)
        self.assertAlmostEqual(p.max_width_mm, 87.1)
        # The jaw gap shrinks as finger_joint grows: 0 rad -> OPEN ~87 mm, 0.8 rad -> CLOSED ~2 mm.
        self.assertAlmostEqual(p.width_to_angle(87.1), 0.0, places=2)
        self.assertAlmostEqual(p.width_to_angle(2.2), 0.8, places=2)
        # Round-trip a mid width.
        angle = p.width_to_angle(40.0)
        self.assertAlmostEqual(p.angle_to_width(angle), 40.0, delta=0.5)
        # Out-of-range widths clamp into the calibrated range.
        self.assertEqual(p.width_to_angle(999.0), p.width_to_angle(p.max_width_mm))
        self.assertEqual(p.width_to_angle(-50.0), p.width_to_angle(p.min_width_mm))

    def test_schunk_egu50_range_and_interpolation(self) -> None:
        # H4.2: a different-vendor parallel gripper behind the SAME profile abstraction; the driven joint is
        # PRISMATIC ("Jaw_Drive", metres) but the table interpolates either way.
        p = SCHUNK_EGU50_PROFILE
        self.assertEqual(p.driven_joint, "Jaw_Drive")
        self.assertAlmostEqual(p.min_width_mm, 4.1)
        self.assertAlmostEqual(p.max_width_mm, 84.1)
        # Jaw_Drive grows with the opening: 0.0 m -> CLOSED ~4 mm, 0.04 m -> OPEN ~84 mm.
        self.assertAlmostEqual(p.width_to_angle(4.1), 0.0, places=3)
        self.assertAlmostEqual(p.width_to_angle(84.1), 0.04, places=3)
        # Round-trip a mid width (linear table).
        angle = p.width_to_angle(44.1)
        self.assertAlmostEqual(angle, 0.02, places=3)
        self.assertAlmostEqual(p.angle_to_width(angle), 44.1, delta=0.5)
        # Out-of-range widths clamp.
        self.assertEqual(p.width_to_angle(999.0), p.width_to_angle(p.max_width_mm))
        self.assertEqual(p.width_to_angle(-50.0), p.width_to_angle(p.min_width_mm))


class IsaacGripperMockTests(unittest.TestCase):
    def _gripper(self) -> IsaacGripper:
        cfg = SimRobotConfig(mock_mode=True, gripper_prim_path="/World/Gripper")
        return IsaacGripper(
            session=IsaacSimSession(cfg), gripper_prim_path="/World/Gripper", mock_mode=True
        )

    def test_satisfies_gripper_protocol(self) -> None:
        self.assertIsInstance(self._gripper(), Gripper)

    def test_mock_open_close_set_width(self) -> None:
        g = self._gripper()
        g.connect()
        self.assertTrue(g.is_connected)
        self.assertAlmostEqual(g.min_width_mm, 2.2)
        self.assertAlmostEqual(g.max_width_mm, 87.1)
        g.open()
        self.assertAlmostEqual(g.get_width_mm(), 87.1, places=2)
        g.close()
        self.assertAlmostEqual(g.get_width_mm(), 2.2, places=2)
        g.set_width_mm(40.0)
        self.assertAlmostEqual(g.get_width_mm(), 40.0, places=2)
        # Clamp above max.
        g.set_width_mm(999.0)
        self.assertAlmostEqual(g.get_width_mm(), 87.1, places=2)

    def test_commands_before_connect_raise(self) -> None:
        g = self._gripper()
        with self.assertRaises(RobotConnectionError):
            g.set_width_mm(40.0)
        with self.assertRaises(RobotConnectionError):
            g.get_width_mm()


if __name__ == "__main__":
    unittest.main()
