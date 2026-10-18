"""The capsule guard skips the tool against the arm's last link, whichever index that is (P0f F9d).

The capsule fallback labels the arm's links ``link_0`` up and skips the pairs that overlap by construction, the tool
against the link it is bolted to among them. That pair was written ``{"tool", "link_5"}``, a UR's last link. On an arm
with seven links it would have refused every pose (the tool touching ``link_6`` by construction) and never checked the
tool against ``link_5``, which it can hit. The skip is now the tool against the last link the guard built, so a UR's
pairs are exactly what they were.
"""

from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from src.config.schema.robot import RobotConfig
from src.robot.safety._capsule import Capsule
from src.robot.safety.self_collision import SelfCollisionGuard


def _capsule(x_mm: float) -> Capsule:
    return Capsule(p0=np.array([x_mm, 0.0, 0.0]), p1=np.array([x_mm, 0.0, 100.0]), radius_mm=10.0)


def _verdict(links: int, tool_at: int) -> bool:
    """Whether the capsule guard accepts ``links`` links a metre apart and a tool touching link ``tool_at``."""
    guard = SelfCollisionGuard(RobotConfig.model_validate({"vendor": "ur"}).safety.self_collision)
    guard._backend = "capsule"
    with mock.patch.object(guard, "refusal_for", return_value=None), \
            mock.patch.object(guard, "_base_capsule", return_value=_capsule(-10_000.0)), \
            mock.patch.object(guard, "_ur_arm_capsules", return_value=[_capsule(1000.0 * i) for i in range(links)]), \
            mock.patch.object(guard, "_tool_capsule", return_value=_capsule(1000.0 * tool_at + 5.0)):
        return guard.evaluate(mock.MagicMock()).accepted


class TheToolIsSkippedAgainstTheLastLinkOnlyTests(unittest.TestCase):
    def test_on_seven_links_the_tool_on_the_last_is_skipped(self) -> None:
        self.assertTrue(_verdict(7, 6))

    def test_on_seven_links_the_tool_against_the_sixth_is_checked(self) -> None:
        self.assertFalse(_verdict(7, 5))

    def test_on_a_ur_the_tool_on_link_5_is_skipped_and_on_link_4_checked(self) -> None:
        """THE CONTROL: a UR's six links keep the pair the guard always skipped."""
        self.assertTrue(_verdict(6, 5))
        self.assertFalse(_verdict(6, 4))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
