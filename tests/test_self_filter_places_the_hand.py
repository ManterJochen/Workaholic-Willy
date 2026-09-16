"""The self filter, the hand tip and the carried part follow the declared hand, not a hard coded tool0 +Y (UM lane S07).

The self filter takes the robot's own body back out of what the cameras see. Its hand was the sphere map on the flange,
each sphere grown to hold the hand's mesh, and the carried part hung from the fingertips, all along tool0 +Y. On a hand
declared along the UR convention (+Z) that is a body 90 degrees from where the hand is: the real hand stays in the
camera world as an obstacle and empty space is filtered out.

So the hand is placed exactly as the guard places it (S06): the plate along the model's approach, then the rotation
the declaration derives. The identity does no arithmetic, and S02's golden holds the sim cells byte for byte.
"""

from __future__ import annotations

import unittest
from typing import Any

import numpy as np

_UR_FRAME = {"source": "polyscope", "offset_mm": [0.0, 0.0, 155.75], "rotation_quat_xyzw": [0.0, 0.0, 0.0, 1.0]}
_SIM_FRAME = {"source": "willy", "offset_mm": [0.0, 155.75, 0.0], "rotation_quat_xyzw": [-0.70710678, 0.0, 0.0, 0.70710678]}


def _hand(frame: dict[str, Any], model: str = "robotiq_hande") -> Any:
    from src.config.schema.robot import RobotConfig
    from src.robot.safety.planning.hand import planner_hand

    gripper: dict[str, Any] = {"model": model, "tool_frame": frame}
    if model == "robotiq_hande":
        gripper["coupling_plates_mm"] = [20.0]
    return planner_hand(RobotConfig.model_validate({"vendor": "ur", "gripper": gripper}))


def _placed_hand_vertices(hand: Any) -> np.ndarray:
    from src.robot.safety._fcl_self_collision import place_hand_vertices
    from src.robot.safety.planning.environment import hand_mesh_bundle

    with np.load(hand_mesh_bundle(hand.model)) as data:
        origin = str(np.asarray(data["gripper__origin"]).reshape(-1)[0]) if "gripper__origin" in data.files else ""
        parts = [place_hand_vertices(part, np.asarray(data[f"{part}__v"], dtype=np.float64), origin=origin,
                                     coupling_mm=hand.coupling_mm, placement=hand.placement)
                 for part in ("gripper", "lfinger", "rfinger")]
    return np.vstack(parts)


class TheFilterHoldsTheDeclaredHandTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from src.robot.safety.planning.self_envelope import hand_spheres

        cls.along_y = _hand(_SIM_FRAME)
        cls.along_z = _hand(_UR_FRAME)
        cls.spheres_y = hand_spheres(cls.along_y, "ur5e")
        cls.spheres_z = hand_spheres(cls.along_z, "ur5e")

    def test_the_spheres_are_the_same_spheres_turned(self) -> None:
        rotation = np.asarray(self.along_z.placement.rotation)
        self.assertEqual(len(self.spheres_z), len(self.spheres_y))
        for a, b in zip(self.spheres_y, self.spheres_z):
            np.testing.assert_allclose(np.asarray(b.start_mm), rotation @ np.asarray(a.start_mm), atol=1e-9)
            self.assertAlmostEqual(b.radius_mm, a.radius_mm, delta=1e-9)

    def test_every_placed_vertex_of_the_hand_is_inside_its_spheres(self) -> None:
        vertices = _placed_hand_vertices(self.along_z)
        centres = np.asarray([s.start_mm for s in self.spheres_z])
        radii = np.asarray([s.radius_mm for s in self.spheres_z])
        outside = np.min(np.linalg.norm(vertices[:, None, :] - centres[None, :, :], axis=2) - radii[None, :], axis=1)
        self.assertLessEqual(float(outside.max()), 1e-6)

    def test_a_hand_turned_the_other_way_would_fall_outside(self) -> None:
        """The control for the containment above: the same vertices left in the hand model's own axes, as the filter
        held them before this step, are not held by the placed spheres."""
        vertices = _placed_hand_vertices(self.along_z)
        rotation = np.asarray(self.along_z.placement.rotation)
        wrong = vertices @ rotation  # rows turned by R^T: the placement undone, the hand back along tool0 +Y
        centres = np.asarray([s.start_mm for s in self.spheres_z])
        radii = np.asarray([s.radius_mm for s in self.spheres_z])
        outside = np.min(np.linalg.norm(wrong[:, None, :] - centres[None, :, :], axis=2) - radii[None, :], axis=1)
        self.assertGreater(float(outside.max()), 10.0)


class TheCarriedPartHangsFromTheDeclaredHandTests(unittest.TestCase):
    def test_the_box_lies_along_z(self) -> None:
        from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres

        hand = _hand(_UR_FRAME)
        spheres = hand_spheres(hand, "ur5e")
        dims, centre = carried_part_box(hand, spheres, grip_width_mm=60.0, length_mm=120.0, lateral_margin_mm=10.0)
        tip = max(float(s.start_mm[2]) + s.radius_mm for s in spheres)
        self.assertEqual(dims, (70.0, 70.0, 120.0))
        np.testing.assert_allclose(centre, (0.0, 0.0, tip + 60.0), atol=1e-9)

    def test_the_capsule_lies_along_z(self) -> None:
        from src.robot.safety.planning.self_envelope import hand_spheres, payload_capsule

        hand = _hand(_UR_FRAME)
        capsule = payload_capsule(hand, hand_spheres(hand, "ur5e"), length_mm=120.0, lateral_margin_mm=10.0)
        axis = np.asarray(capsule.end_mm) - np.asarray(capsule.start_mm)
        np.testing.assert_allclose(axis, (0.0, 0.0, 120.0), atol=1e-9)

    def test_the_sim_hand_still_hangs_along_y(self) -> None:
        """The control: the identity placement is today's +Y, and S02's golden holds its numbers."""
        from src.robot.safety.planning.self_envelope import carried_part_box, hand_spheres

        hand = _hand(_SIM_FRAME)
        dims, _ = carried_part_box(hand, hand_spheres(hand, "ur5e"), grip_width_mm=60.0, length_mm=120.0,
                                   lateral_margin_mm=10.0)
        self.assertEqual(dims, (70.0, 120.0, 70.0))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
