"""The hand a push plans with comes from the cell's config and the gripper registry, housing thickness included.

``PushHand`` keeps the housing clear of tall neighbours with ``palm_thickness_mm``, the housing along the closing axis.
The cell's gripper model (``ParallelJawGripperModel``, or the ``grasping.gripper_geometry.parallel_jaw`` block the
loader fills from the named hand) does not carry it; only the registry file does (``config/grippers/robotiq_hande.yaml``:
75.0). Without it the planner takes the housing as wide as the open fingers' outer faces, 71.6 mm on the Hand-E, 1.7 mm
short on each side (Recovery A's open point). So the push's hand is built here, from both:

* the owner's cell (``ur10,hande``): the Hand-E's fingers, its 49.99 mm aperture as the open width, the 75 mm housing;
* a cell that names no hand, or names one the registry does not hold: no push, with a sentence;
* a suction cell: no finger to push with, no push.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.robot.grasping.collision.gripper_model import ParallelJawGripperModel
from src.robot.grasping.recovery.push_hand import (
    REFUSED_PUSH_HAND_NOT_A_JAW,
    REFUSED_PUSH_HAND_UNKNOWN,
    push_hand_from_registry,
    push_hand_from_robot_config,
)
from src.robot.grasping.recovery.push_planner import PushHand, PushRefusal


class _Loaded(unittest.TestCase):
    def _robot(self, profile: str):  # type: ignore[no-untyped-def]
        from src.config.loader import load_robot_config, reload_config

        reload_config()
        self.addCleanup(reload_config)
        return load_robot_config(None, profile=profile)


class TheOwnersHandE(_Loaded):
    def test_the_hand_e_pushes_with_its_own_fingers_and_its_seventy_five_millimetre_housing(self) -> None:
        hand = push_hand_from_robot_config(self._robot("ur10,hande"))
        assert isinstance(hand, PushHand), hand
        self.assertEqual(
            (hand.finger_thickness_mm, hand.finger_width_mm, hand.fingertip_depth_mm, hand.finger_length_mm,
             hand.palm_width_mm, hand.open_width_mm, hand.palm_thickness_mm),
            (10.80, 29.24, 10.45, 36.53, 75.0, 49.99, 75.0),
        )
        # The housing reaches 37.5 mm from the TCP along the push, past the open fingers' 35.8 mm.
        self.assertAlmostEqual(hand.half_outer_mm, 49.99 / 2 + 10.80)
        self.assertEqual(hand.palm_half_along_mm, 37.5)

    def test_the_registry_supplies_what_the_gripper_model_does_not_carry(self) -> None:
        model = ParallelJawGripperModel(finger_length_mm=36.53, finger_thickness_mm=10.80, finger_width_mm=29.24,
                                        fingertip_depth_mm=10.45, pad_length_mm=20.91, pad_ahead_mm=10.45,
                                        palm_depth_mm=104.12, palm_width_mm=75.0)
        # The model carries the housing only where a cell hands it one (``build_gripper_geometry``); this one has none.
        self.assertIsNone(model.palm_thickness_mm)
        hand = push_hand_from_registry(model, hand="robotiq_hande", open_width_mm=49.99)
        assert isinstance(hand, PushHand), hand
        self.assertEqual(hand.palm_thickness_mm, 75.0)
        self.assertEqual(hand.open_width_mm, 49.99)

    def test_a_registry_hand_without_a_housing_thickness_leaves_the_planners_fallback(self) -> None:
        # The 2F-85's registry file carries no palm_thickness_mm; its open fingers' outer faces stand for the housing.
        hand = push_hand_from_registry(ParallelJawGripperModel(), hand="robotiq_2f85", open_width_mm=85.0)
        assert isinstance(hand, PushHand), hand
        self.assertIsNone(hand.palm_thickness_mm)
        self.assertEqual(hand.palm_half_along_mm, hand.half_outer_mm)


class NoHandNoPush(_Loaded):
    def _refused(self, result: object, code: str) -> PushRefusal:
        assert isinstance(result, PushRefusal), result
        self.assertEqual(result.code, code)
        self.assertTrue(result.sentence.endswith("."), result.sentence)
        return result

    def test_a_cell_that_names_no_hand(self) -> None:
        refusal = self._refused(push_hand_from_robot_config(self._robot("ur10")), REFUSED_PUSH_HAND_UNKNOWN)
        self.assertIn("robot.gripper.model", refusal.sentence)

    def test_a_hand_the_registry_does_not_hold_and_an_alias(self) -> None:
        for name in ("no_such_hand", "2f85"):
            with self.subTest(name=name):
                refusal = self._refused(push_hand_from_registry(ParallelJawGripperModel(), hand=name,
                                                                open_width_mm=85.0), REFUSED_PUSH_HAND_UNKNOWN)
                self.assertIn(repr(name), refusal.sentence)

    def test_a_suction_cell_has_no_finger_to_push_with(self) -> None:
        robot = SimpleNamespace(gripper=SimpleNamespace(model="robotiq_hande", max_width_mm=49.99),
                                grasping=SimpleNamespace(gripper_geometry=SimpleNamespace(kind="suction")))
        self._refused(push_hand_from_robot_config(robot), REFUSED_PUSH_HAND_NOT_A_JAW)

    def test_a_config_with_no_gripper_block(self) -> None:
        self._refused(push_hand_from_robot_config(SimpleNamespace(gripper=None, grasping=None)),
                      REFUSED_PUSH_HAND_UNKNOWN)


if __name__ == "__main__":
    unittest.main()
