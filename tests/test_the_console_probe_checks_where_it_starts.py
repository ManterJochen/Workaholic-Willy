"""The URSim console probe refuses to start where every path would be refused (review of 2026-10-02).

``scripts/ursim/probe_halt.py`` leaves wrist 2 at -90 deg. The console probe's layer homes it at +90 deg, and the
console's joint guard keeps every path within half a turn of home, less its 5 deg margin, so a probe started there had
every path refused: C1 ended ``failed_in_a_row`` and its stop record refused C2-C4 (measured on URSim CB3,
2026-10-02). The probe now reads where the arm stands before anything moves, with the guard's own window, and exits 2
saying where to move it.
"""

from __future__ import annotations

import math
import re
import unittest

from scripts.ursim.probe_console_task import _LAYER, outside_the_window
from src.robot.safety.joint_limits import UR_JOINT_LIMITS_DEG, half_turn_about_home_deg

#: The home ``_LAYER`` gives the probe's UR10, in degrees.
_HOME_DEG = (-90.0, -90.0, -100.0, -80.0, 90.0, 0.0)
#: The shipped ``robot.safety.joint_limits.margin_deg``.
_MARGIN_DEG = 5.0


def _window() -> tuple[list[float], list[float]]:
    """The guard's window about the probe's home: the UR table, narrowed to half a turn about home but the elbow."""
    lower, upper = UR_JOINT_LIMITS_DEG["ur10"]
    return half_turn_about_home_deg(lower, upper, [math.radians(v) for v in _HOME_DEG], vendor="ur")


def _rad(deg: list[float] | tuple[float, ...]) -> list[float]:
    return [math.radians(v) for v in deg]


class TheProbeChecksWhereTheArmStartsTests(unittest.TestCase):
    def test_the_layer_homes_the_arm_where_this_test_reads_it(self) -> None:
        said = re.search(r"home_joint_positions_deg: \[([^\]]+)\]", _LAYER)
        self.assertIsNotNone(said, "the probe's layer declares no home")
        assert said is not None
        self.assertEqual(_HOME_DEG, tuple(float(v) for v in said.group(1).split(",")))

    def test_the_layers_home_is_inside_the_window(self) -> None:
        lower, upper = _window()
        self.assertEqual("", outside_the_window(_rad(_HOME_DEG), lower, upper, _MARGIN_DEG))

    def test_wrist_2_where_probe_halt_leaves_it_is_refused_and_named(self) -> None:
        lower, upper = _window()
        start = list(_HOME_DEG)
        start[4] = -90.0
        said = outside_the_window(_rad(start), lower, upper, _MARGIN_DEG)
        self.assertIn("axis 4", said)
        self.assertIn("-90.0 deg", said)
        self.assertIn("[-85.0, 265.0]", said)

    def test_the_margin_is_kept_off_each_end(self) -> None:
        lower, upper = _window()
        inside, outside = list(_HOME_DEG), list(_HOME_DEG)
        inside[0], outside[0] = -90.0 + 174.0, -90.0 + 176.0  # home + 180 is the seam; the margin keeps 5 deg off it
        self.assertEqual("", outside_the_window(_rad(inside), lower, upper, _MARGIN_DEG))
        self.assertIn("axis 0", outside_the_window(_rad(outside), lower, upper, _MARGIN_DEG))

    def test_the_elbow_keeps_its_table(self) -> None:
        """A UR elbow is never narrowed to the window: bent far the other way it is still inside its table."""
        lower, upper = _window()
        bent = list(_HOME_DEG)
        bent[2] = 170.0
        self.assertEqual("", outside_the_window(_rad(bent), lower, upper, _MARGIN_DEG))


if __name__ == "__main__":
    unittest.main()
