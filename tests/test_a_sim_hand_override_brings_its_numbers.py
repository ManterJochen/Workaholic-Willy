"""A per-run hand in the sim bootstrap brings its own numbers (customer chain lane C4e).

``bootstrap_sim_cell(hand=...)`` renames the hand without the loader. It dumped the tree's robot with every key set and
revalidated it, so a ``--hand schunk_egu50`` run kept the 2F-85's stroke and envelope while the same hand named in YAML
derives its own (lane C4b). Both doors give one answer now. No test here starts Isaac: the reach check is patched to
stop the boot.
"""

from __future__ import annotations

import unittest
import unittest.mock as mock
from typing import Any

from src.config import reload_config


class _StopBoot(RuntimeError):
    """Raised by the patched reach check so the bootstrap never reaches Isaac."""


def _booted_robot(hand: str) -> Any:
    import src.willy_sim.harness.bootstrap as boot

    seen: list[Any] = []

    def stop(robot: Any) -> None:
        seen.append(robot)
        raise _StopBoot

    reload_config()
    with mock.patch.object(boot, "warn_if_unreachable", stop):
        try:
            boot.bootstrap_sim_cell(None, headless=True, hand=hand)
        except _StopBoot:
            pass
    assert seen, "the bootstrap did not reach its reach check"
    return seen[0]


def _read(robot: Any, dotted: str) -> Any:
    node = robot
    for part in dotted.split("."):
        node = getattr(node, part)
    return str(node) if dotted.endswith(".kind") else node


class TheOverrideDerivesTests(unittest.TestCase):
    def test_the_override_derives_the_egu50_widths_and_envelope(self) -> None:
        robot = _booted_robot("schunk_egu50")
        self.assertEqual(
            (robot.gripper.max_width_mm, robot.gripper.closed_width_mm,
             robot.grasping.gripper_geometry.parallel_jaw.fingertip_depth_mm),
            (84.1, 4.1, 7.5),
        )

    def test_every_hand_key_is_the_one_the_loader_derives(self) -> None:
        """Both doors, one answer: all thirteen keys, not the three the red test names."""
        from src.config.hand_numbers import HandNumbers

        robot = _booted_robot("schunk_egu50")
        for key, _, value in HandNumbers.from_registry(model="schunk_egu50").values:
            with self.subTest(key=key):
                self.assertEqual(_read(robot, key), value)

    def test_the_tree_s_own_hand_keeps_its_numbers(self) -> None:
        """⭐ THE CONTROL: the 2F-85 the sim tree names yields 85.0, 0.0 and 28.72 before and after."""
        robot = _booted_robot("robotiq_2f85")
        self.assertEqual(
            (robot.gripper.max_width_mm, robot.gripper.closed_width_mm,
             robot.grasping.gripper_geometry.parallel_jaw.fingertip_depth_mm),
            (85.0, 0.0, 28.72),
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
