"""An arm description names the link a hand hangs from and the axis it approaches along (P0e E12, owner decision 7).

Both were the UR's by assumption: the planner hung every hand from ``tool0`` (``HAND_PARENT``), and the desk checklist
warned on a +Y tool frame only for ``vendor == "ur"``, against a +Z it wrote into its sentence. The description holds
both now, required, and those two readers ask it: the planner hangs the hand from the link the description names
(P0f F9b, held in ``tests/robot/safety/planning/test_a_body_hangs_from_the_arms_own_flange.py``; until then it refused any other link), and
the checklist warns on a declared approach the flange does not have, for any arm that drives a controller.
"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.config.arms import available_arms, load_arm
from src.config.schema.arms.arm_schema import ArmSpec
from src.config.schema.robot import RobotConfig
from src.robot.execution.real_cell import CheckStatus, run_config_preflight
from tests.robot.safety.test_ur_urdf_renders_the_dh_chain import _LINK_FRAMES, _load
_PLUS_Y = (-0.7071067811865476, 0.0, 0.0, 0.7071067811865476)  # the 2F-85 as the Isaac asset mounts it


class EveryDescriptionNamesItsFlangeTests(unittest.TestCase):
    def test_the_link_is_the_one_ur_s_description_puts_at_the_end_of_the_dh_chain(self) -> None:
        """``tests/robot/safety/test_ur_urdf_renders_the_dh_chain.py`` proves ``tool0`` is the DH flange; each description says so."""
        dh_frame_of = dict(_LINK_FRAMES)
        for model in available_arms():
            with self.subTest(model=model):
                description = load_arm(model)
                self.assertEqual(description.dof, dh_frame_of[description.flange.link])

    def test_the_approach_is_the_axis_ur_s_description_names_for_that_link(self) -> None:
        """UR's macro, vendored and pinned: ``tool0`` with the comment ``default toolframe - X+ left, Y+ up, Z+ front``."""
        text = (_load().ROOT / "urdf" / "ur_macro.xacro").read_text(encoding="utf-8")
        tool0 = text.index('<link name="${tf_prefix}tool0"/>')
        self.assertIn("Z+ front", text[tool0:tool0 + 400])
        for model in available_arms():
            with self.subTest(model=model):
                self.assertEqual("+Z", load_arm(model).flange.approach)

    def test_a_description_without_a_flange_is_refused(self) -> None:
        data = load_arm("ur5e").model_dump(mode="json")
        del data["flange"]
        with self.assertRaises(ValidationError):
            ArmSpec.model_validate(data)


def _tree(*, vendor: str = "ur", quaternion: tuple[float, float, float, float] = _PLUS_Y,
          offset: tuple[float, float, float] = (0.0, 132.0, 0.0)) -> RobotConfig:
    tree: dict[str, object] = {
        "vendor": vendor, "gripper": {"model": "robotiq_2f85", "tool_frame": {
            "source": "willy", "offset_mm": offset, "rotation_quat_xyzw": quaternion}},
    }
    if vendor == "ur":
        tree["ur"] = {"ip": "10.1.2.3"}
    return RobotConfig.model_validate(tree)


def _hand_rows(tree: RobotConfig) -> list[object]:
    return [check for check in run_config_preflight(tree).checks if check.name == "hand"]


class TheChecklistAsksTheFlangeTests(unittest.TestCase):
    def test_a_ur_cell_declaring_plus_y_is_warned_with_the_description_s_axis(self) -> None:
        rows = _hand_rows(_tree())
        self.assertEqual([CheckStatus.WARN], [row.status for row in rows])  # type: ignore[attr-defined]
        self.assertIn("approaches along +Z, as its description ur5e.yaml states it", rows[0].detail)  # type: ignore[attr-defined]
        self.assertIn("Isaac UR asset", rows[0].detail)  # type: ignore[attr-defined]

    def test_a_ur_cell_declaring_the_flange_s_own_axis_is_not_warned(self) -> None:
        rows = _hand_rows(_tree(quaternion=(0.0, 0.0, 0.0, 1.0), offset=(0.0, 0.0, 132.0)))
        self.assertNotIn(CheckStatus.WARN, [row.status for row in rows])  # type: ignore[attr-defined]


if __name__ == "__main__":
    unittest.main()
