"""Where the hand sits on the flange follows from the declared tool frame, by one derivation (UM lane S03).

Every committed hand model lives in frame M: x closing, y approach out of the mounting face, z = x cross y. A declared
tool frame is the grasp frame G: x closing, y binormal, z approach, so G's axes in M are fixed, R_MG = R_x(-90). The
declaration gives R_TG, G's axes in tool0, and the hand model goes onto the flange by

    R_TM = R_TG @ R_MG^T,     p_tool0 = R_TM (p_model + plate along model y).

The sim's Isaac declaration is R_x(-90) itself, so R_TM is the identity and every array the cells use today stays the
same bytes (S02's golden). A UR convention declaration, the identity quaternion, gives R_x(+90): approach along tool0 +Z,
closing +X. What still refuses is typed, in this order: a matrix that is not orthonormal, a mirror, an approach more
than a degree from every flange axis, an axis approach other than +Y or +Z, and a clocking that is not a quarter turn.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.geometry.quaternion import from_rotation_matrix, to_rotation_matrix

_SIM = (-0.70710678, 0.0, 0.0, 0.70710678)
_UR = (0.0, 0.0, 0.0, 1.0)
_IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _placement():
    from src.robot.safety.planning import _hand_placement

    return _hand_placement


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _clocked(base_q: tuple[float, float, float, float], quarter_turns: int) -> tuple[float, float, float, float]:
    """The declared frame turned about its own approach (G's z) by quarter turns, at full precision."""
    r = to_rotation_matrix(np.asarray(base_q, dtype=np.float64)) @ _rz(quarter_turns * math.pi / 2.0)
    q = from_rotation_matrix(r)
    return (float(q[0]), float(q[1]), float(q[2]), float(q[3]))


class TheDeclaredRotationIsDerivedTests(unittest.TestCase):
    def test_the_sim_declaration_is_exactly_the_identity(self) -> None:
        p = _placement().HandPlacement.from_quaternion_xyzw(_SIM)
        self.assertEqual(p.rotation, _IDENTITY)
        for row in p.rotation:
            for value in row:
                self.assertEqual(math.copysign(1.0, value), 1.0 if value == 0.0 else math.copysign(1.0, value))
        self.assertTrue(p.is_identity)
        self.assertEqual((p.approach, p.closing), ("+Y", "+X"))
        self.assertLess(p.residual_deg, 1e-5)
        self.assertEqual(p.source, "declared")

    def test_no_zero_is_negative(self) -> None:
        """-0.0 == 0.0 in Python, so the check above cannot see it; a repr of -0.0 would change S02's golden."""
        for q in (_SIM, _UR):
            p = _placement().HandPlacement.from_quaternion_xyzw(q)
            self.assertNotIn("-0.0", repr(p.rotation))

    def test_the_ur_convention_approaches_along_z(self) -> None:
        p = _placement().HandPlacement.from_quaternion_xyzw(_UR)
        self.assertEqual(p.rotation, ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)))
        self.assertEqual((p.approach, p.closing), ("+Z", "+X"))
        self.assertEqual(p.approach_in_tool0, (0.0, 0.0, 1.0))
        self.assertEqual(p.closing_in_tool0, (1.0, 0.0, 0.0))
        self.assertFalse(p.is_identity)
        self.assertIn("declared", p.render())
        self.assertIn("not measured", p.render())

    def test_the_hand_offset_lands_on_the_declared_offset(self) -> None:
        """R_TM carries the model's approach onto the declared one: the sim's 132 mm along +Y and URSim's along +Z."""
        for q, expected in ((_SIM, (0.0, 132.0, 0.0)), (_UR, (0.0, 0.0, 132.0))):
            p = _placement().HandPlacement.from_quaternion_xyzw(q)
            moved = np.asarray(p.rotation) @ np.array([0.0, 132.0, 0.0])
            with self.subTest(q=q):
                np.testing.assert_allclose(moved, expected, atol=1e-12)

    def test_the_transposed_product_would_be_caught(self) -> None:
        """The control for the test above: R_TG @ R_MG instead of R_TG @ R_MG^T sends URSim's hand into the wrist."""
        r_mg = np.array(_placement().R_MG)
        wrong = to_rotation_matrix(np.asarray(_UR)) @ r_mg
        self.assertLess(float((wrong @ np.array([0.0, 132.0, 0.0]))[2]), -131.0)

    def test_eight_placements_are_admitted(self) -> None:
        seen = set()
        for base in (_SIM, _UR):
            for turns in range(4):
                q = _clocked(base, turns)
                p = _placement().HandPlacement.from_quaternion_xyzw(q)
                declared_approach = to_rotation_matrix(np.asarray(q))[:, 2]
                with self.subTest(base=base, turns=turns):
                    self.assertAlmostEqual(float(np.linalg.det(np.asarray(p.rotation))), 1.0, delta=1e-12)
                    np.testing.assert_allclose(p.approach_in_tool0, declared_approach, atol=1e-8)
                seen.add((p.approach, p.closing))
        self.assertEqual(len(seen), 8, sorted(seen))

    def test_the_quaternion_arithmetic_is_the_stack_s(self) -> None:
        rng = np.random.default_rng(20260915)
        mod = _placement()
        for _ in range(64):
            q = rng.normal(size=4)
            q = q / np.linalg.norm(q)
            mine = np.asarray(mod.quaternion_xyzw_to_matrix(tuple(float(v) for v in q)))
            np.testing.assert_allclose(mine, to_rotation_matrix(q), atol=1e-12)

    def test_an_undeclared_frame_is_the_identity_and_says_so(self) -> None:
        p = _placement().HandPlacement.undeclared()
        self.assertEqual(p.rotation, _IDENTITY)
        self.assertEqual(p.source, "undeclared")
        self.assertIn("undeclared", p.render())
        self.assertEqual(p.to_dict()["source"], "undeclared")


class WhatStillRefusesTests(unittest.TestCase):
    def _kind(self, call) -> str:
        mod = _placement()
        with self.assertRaises(mod.PlacementRefused) as caught:
            call()
        return caught.exception.kind.value

    def test_a_matrix_that_is_not_a_rotation(self) -> None:
        mod = _placement()
        self.assertEqual(self._kind(lambda: mod.HandPlacement.from_matrix(((1.0, 0.1, 0.0), (0.0, 1.0, 0.0),
                                                                           (0.0, 0.0, 1.0)))), "not_proper")

    def test_a_mirror(self) -> None:
        mod = _placement()
        self.assertEqual(self._kind(lambda: mod.HandPlacement.from_matrix(((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
                                                                           (0.0, 0.0, 1.0)))), "mirror")

    def test_an_oblique_approach(self) -> None:
        mod = _placement()
        tilt = math.radians(30.0)
        r = to_rotation_matrix(np.asarray(_UR)) @ np.array(
            [[1.0, 0.0, 0.0], [0.0, math.cos(tilt), -math.sin(tilt)], [0.0, math.sin(tilt), math.cos(tilt)]])
        q = from_rotation_matrix(r)
        with self.assertRaises(mod.PlacementRefused) as caught:
            mod.HandPlacement.from_quaternion_xyzw(tuple(float(v) for v in q))
        self.assertEqual(caught.exception.kind.value, "oblique_approach")
        self.assertIn("30", str(caught.exception))

    def test_approaches_nobody_measured(self) -> None:
        mod = _placement()
        into_the_wrist = (1.0, 0.0, 0.0, 0.0)  # a half turn about x: approach -Z
        sideways = tuple(float(v) for v in from_rotation_matrix(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0],
                                                                           [-1.0, 0.0, 0.0]])))  # approach +X
        for q in (into_the_wrist, sideways):
            with self.subTest(q=q):
                self.assertEqual(self._kind(lambda q=q: mod.HandPlacement.from_quaternion_xyzw(q)),
                                 "approach_not_admitted")

    def test_a_clocking_that_is_not_a_quarter_turn(self) -> None:
        mod = _placement()
        with self.assertRaises(mod.PlacementRefused) as caught:
            mod.HandPlacement.from_quaternion_xyzw((0.0, 0.0, 0.38268343, 0.92387953))
        self.assertEqual(caught.exception.kind.value, "clocking_not_quarter_turn")
        self.assertIn("45", str(caught.exception))

    def test_a_quaternion_the_schema_admits_is_not_refused_for_its_norm(self) -> None:
        mod = _placement()
        q = tuple(v * (1.0 + 1e-6) for v in _UR[:3]) + (_UR[3] * (1.0 + 1e-6),)
        self.assertEqual(mod.HandPlacement.from_quaternion_xyzw(q).approach, "+Z")


class TheResolvedHandCarriesItsPlacementTests(unittest.TestCase):
    """S04: ``planner_hand`` resolves the placement once, so no consumer derives its own. The Step 4g refusal stayed
    in S04 and was narrowed in S23 to the placements this derivation refuses (tests/test_the_declared_frame_agrees_
    with_the_hand.py)."""

    _UR_FRAME = {"source": "polyscope", "offset_mm": [0.0, 0.0, 155.75], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}

    def _hand(self, model: str, frame: dict) -> object:
        from src.config.schema.robot import RobotConfig
        from src.robot.safety.planning.hand import planner_hand

        gripper: dict = {"model": model, "tool_frame": frame}
        if model == "robotiq_hande":
            gripper["coupling_plates"] = [{"name": "plate", "thickness_mm": 20.0}]
        return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))

    def test_a_ur_convention_frame_places_the_hand_along_z(self) -> None:
        hand = self._hand("robotiq_hande", self._UR_FRAME)
        self.assertEqual((hand.placement.approach, hand.placement.closing), ("+Z", "+X"))
        self.assertIsNone(hand.placement_refusal)

    def test_an_oblique_frame_carries_no_placement_and_says_why(self) -> None:
        from src.contracts import UNSET

        half = math.radians(30.0) / 2.0
        frame = dict(self._UR_FRAME, rotation_quat_xyzw=[math.sin(half), 0.0, 0.0, math.cos(half)])
        hand = self._hand("robotiq_hande", frame)
        self.assertIs(hand.placement, UNSET)
        self.assertIn("30 degrees", hand.placement_refusal)

    def test_an_undeclared_frame_is_the_undeclared_placement(self) -> None:
        hand = self._hand("robotiq_2f85", {"source": "undeclared"})
        self.assertEqual(hand.placement, _placement().HandPlacement.undeclared())
        self.assertIsNone(hand.placement_refusal)

    def test_the_sim_hand_is_placed_on_the_identity(self) -> None:
        """The control: the frame robot.sim.yaml declares resolves to the exact identity, as S02's golden needs."""
        frame = {"source": "willy", "offset_mm": [0.0, 132.0, 0.0], "rotation_quat_xyzw": list(_SIM)}
        hand = self._hand("robotiq_2f85", frame)
        self.assertTrue(hand.placement.is_identity)
        self.assertEqual(hand.placement.source, "declared")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
