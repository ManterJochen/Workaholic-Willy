"""The half turn at an arm's base is stated once, in its description, as 0 or 180 and nothing else (P0f F6d, F7).

UR's ``ur_macro.xacro`` roots the robot at ``base_link`` and turns the frame the joints start in, ``base_link_inertia``,
half a turn about z. The bake (Rodrigues' form) and the descriptor's sphere placement (an exact matrix) read that turn
from ``base.chain_yaw_deg`` since P0f F7, each mapping the value to its own expression, which is why a general angle is
refused: it would move a hash in the last bits. ``tests/robot/safety/test_arm_descriptions_are_urs_numbers.py`` holds the
descriptions to UR's rendered URDF; this file holds what the schema admits.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.arms import load_arm
from src.config.schema.arms import ArmSpec


class TheSchemaHoldsOnlyAHalfTurnOrNoneTests(unittest.TestCase):
    def _refusal(self, **base: object) -> str:
        data = load_arm("ur5e").model_dump(mode="json")
        data["base"].update(base)
        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate(data)
        return str(caught.exception)

    def test_a_quarter_turn_a_negative_half_turn_and_a_string_are_refused(self) -> None:
        for yaw in (45, 90, -180, "180"):
            with self.subTest(chain_yaw_deg=yaw):
                self.assertIn("chain_yaw_deg", self._refusal(chain_yaw_deg=yaw))

    def test_a_description_without_a_base_is_refused(self) -> None:
        data = load_arm("ur5e").model_dump(mode="json")
        del data["base"]
        with self.assertRaises(ValidationError) as caught:
            ArmSpec.model_validate(data)
        self.assertIn("base", str(caught.exception))

    def test_a_base_on_a_joints_link_or_the_flange_is_refused(self) -> None:
        for link in ("shoulder_link", "wrist_3_link", "tool0"):
            with self.subTest(link=link):
                self.assertIn("the root the joints hang from", self._refusal(link=link))


if __name__ == "__main__":
    unittest.main()
