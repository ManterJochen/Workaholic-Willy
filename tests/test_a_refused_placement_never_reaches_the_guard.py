"""A hand whose declared frame places no model refuses the cell, instead of being modelled where it is not (UM lane).

⛔ THE HOLE THIS CLOSES, found by review on 2026-09-16. The build gate read the declared APPROACH axis
(``approach_refusal``) and nothing else, while ``_hand_placement`` refuses three further ways: a mirror, an oblique
frame, and a clocking that is not a quarter turn. A cell can declare a quaternion whose approach is exactly the hand
model's +Y and whose closing is 45 degrees away: the approach check passes, ``planner_hand`` leaves ``placement`` UNSET
with a sentence, and the exact mesh guard then read ``placement=None``, which ``place_hand_vertices`` treats as the
identity. The guard measured a hand 45 degrees from where the cell says it is, and said nothing. On a cuRobo cell the
planner refuses it at start (``HandLink.from_hand``); an ik cell never starts a planner, so nothing refused at all.

That is the implied default for a safety identity the owner refused on 13.09. The cell refuses at build now, and the
guard refuses to be constructed with a hand it cannot place. Since UM lane S23 ``approach_refusal`` is gone and the
placement refusal is the only one a declared frame meets, so a +Z frame builds and everything below still refuses.
"""

from __future__ import annotations

import unittest

from src.config.loader import ConfigError
from src.config.schema.robot import RobotConfig
from src.contracts import UNSET
from src.robot.safety.preflight import SafetyPreflight
from tests.test_the_hand_is_placed_by_the_declared_frame import _SIM, _clocked

#: The Isaac frame turned 45 degrees about its own approach: the approach axis is unchanged and the closing is not.
_CLOCKED_45 = _clocked(_SIM, 0.5)


def _cell(quaternion: "tuple[float, float, float, float]", planner: str = "ik") -> RobotConfig:
    return RobotConfig.model_validate({
        "vendor": "ur",
        "ur": {"model": "ur5e", "motion_planner": planner},
        "gripper": {
            "model": "robotiq_hande",
            "coupling_plates": [{"name": "plate", "thickness_mm": 20.0}],
            "tool_frame": {"source": "willy", "offset_mm": [0.0, 132.0, 0.0],
                           "rotation_quat_xyzw": list(quaternion)},
        },
        "safety": {"self_collision": {"enforce": True, "backend": "fcl", "kinematics_model": "ur5e"}},
    })


def _hand(config: RobotConfig):
    from src.robot.safety.planning.hand import planner_hand

    return planner_hand(config)


class TheCellRefusesAPlacementItCannotModelTests(unittest.TestCase):
    def test_a_clocked_frame_refuses_at_build(self) -> None:
        config = _cell(_CLOCKED_45)
        hand = _hand(config)
        self.assertIs(hand.placement, UNSET, "the fixture must be a frame the derivation refuses")
        self.assertIsNotNone(hand.placement_refusal)
        with self.assertRaises(ConfigError) as ctx:
            SafetyPreflight.from_safety_config(config.safety, config.workspace_limits, arm_model="ur5e", hand=hand)
        self.assertIn("quarter turn", str(ctx.exception))

    def test_the_frame_the_cells_run_still_builds(self) -> None:
        """⭐ THE CONTROL. The Isaac cell declares the frame that places the hand on the identity, and it must build."""
        config = _cell(_SIM)
        hand = _hand(config)
        self.assertTrue(hand.placement is not UNSET)
        preflight = SafetyPreflight.from_safety_config(config.safety, config.workspace_limits, arm_model="ur5e", hand=hand)
        self.assertIn("self_collision", [guard.name for guard in preflight.guards])

    def test_an_ik_cell_is_refused_as_well(self) -> None:
        """The hole was widest here: an ik cell never starts a planner, so the planner's refusal never runs."""
        config = _cell(_CLOCKED_45, planner="ik")
        with self.assertRaises(ConfigError):
            SafetyPreflight.from_safety_config(config.safety, config.workspace_limits, arm_model="ur5e", hand=_hand(config))


class TheGuardRefusesAHandItCannotPlaceTests(unittest.TestCase):
    def test_a_guard_built_directly_on_a_refused_placement_raises(self) -> None:
        """Defence in depth: the runners and the tests build this guard directly, without the cell's build gate."""
        from src.robot.safety.self_collision import SelfCollisionGuard

        config = _cell(_CLOCKED_45)
        with self.assertRaises(ValueError) as ctx:
            SelfCollisionGuard(config.safety.self_collision, hand=_hand(config))
        self.assertIn("quarter turn", str(ctx.exception))

    def test_a_placed_hand_and_no_hand_at_all_both_build(self) -> None:
        """⭐ THE CONTROL, both ways: the guard still builds for a placed hand, and for none (the arm's own bundle)."""
        from src.robot.safety.self_collision import SelfCollisionGuard

        config = _cell(_SIM)
        self.assertIsNotNone(SelfCollisionGuard(config.safety.self_collision, hand=_hand(config)))
        self.assertIsNotNone(SelfCollisionGuard(config.safety.self_collision))


if __name__ == "__main__":
    unittest.main()
