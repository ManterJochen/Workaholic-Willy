"""The retract a sidecar starts from belongs to the arm AND the hand on it (B4 revisited, owner 2026-09-16).

A descriptor is per arm since S11, and the hand is a body link the sidecar adds when it starts. The retract was still
per arm, chosen on the exact meshes alone, and that turned out to be two mistakes in one.

It is not a property of the arm. Measured 2026-09-16 over the rule's own candidate family: at ur3's committed retract
the exact meshes clear the Hand-E by 13.4 mm and the planner's sphere model calls the same pose a self collision, so
the sidecar never becomes ready and the cell does not come up at all. One wrist turn away, 74 of 433 candidate poses
are clear. The same is true of ur5e, ur10e and ur3e with the EGU-50, at a different turn each.

So the retract is chosen per arm, hand, plate and planner margin, and the composition that adds the hand carries it:
one place decides both, and neither can be right about a robot the other is not describing.

``cspace.retract_config`` is what cuRobo seeds its graph search and its IK regularisation from, and what the ready
gate judges, so this is not a cosmetic field.
"""

from __future__ import annotations

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _body_links():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_body_links.py"
    spec = importlib.util.spec_from_file_location("_curobo_body_links_retract", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


_ARM = {
    "robot_cfg": {"kinematics": {
        "urdf_path": "robot/ur5e.urdf",
        "base_link": "base_link",
        "ee_link": "tool0",
        "tool_frames": ["tool0"],
        # An ARM descriptor: the tool frame is not a collision link, because the hand is added as a body (S11).
        "collision_link_names": ["wrist_1_link"],
        "collision_spheres": {"wrist_1_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}]},
        "collision_sphere_buffer": 0.0,
        "self_collision_ignore": {"wrist_1_link": []},
        "self_collision_buffer": {"wrist_1_link": 0.0},
        "cspace": {
            "joint_names": ["a", "b", "c", "d", "e", "f"],
            # What THIS repository's descriptors carry, measured 2026-09-16: default_joint_position and no
            # retract_config. cuRobo parses this block into CSpaceParams, which refuses a keyword it does not
            # know, so a fixture carrying both would hide exactly the failure that cost an M1 run.
            "default_joint_position": [0.0, -1.0, 0.9, 0.0, 0.0, 0.0],
            "null_space_weight": [1.0] * 6,
            "cspace_distance_weight": [1.0] * 6,
            "max_jerk": 500.0,
            "max_acceleration": 15.0,
        },
    }},
}

_HAND = {
    "link": "hand",
    "parent": "tool0",
    "joint": "tool0_to_hand",
    "fixed_transform": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    "spheres": [{"center": [0.0, 0.05, 0.0], "radius": 0.03}],
    "ignore": ["tool0"],
}

_CHOSEN = [0.0, -1.0, 0.9, 0.0, 0.0, -0.75]


class TheCompositionCarriesTheRetractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _body_links()

    def test_the_hand_brings_its_own_retract(self) -> None:
        composed = self.module.compose_sidecar_config(
            copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0, default_q=_CHOSEN,
        )
        cspace = composed["robot_cfg"]["kinematics"]["cspace"]
        self.assertEqual(cspace["default_joint_position"], _CHOSEN)
        self.assertNotIn("retract_config", cspace,
                         "a field the descriptor does not carry was invented: cuRobo parses this block "
                         "into CSpaceParams and refuses a keyword it does not know")

    def test_without_one_the_descriptor_keeps_its_own(self) -> None:
        """⭐ THE CONTROL. Every cell that names no retract, and every test that composes without one, is unchanged."""
        composed = self.module.compose_sidecar_config(
            copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0,
        )
        cspace = composed["robot_cfg"]["kinematics"]["cspace"]
        self.assertEqual(cspace["default_joint_position"], [0.0, -1.0, 0.9, 0.0, 0.0, 0.0])

    def test_a_descriptor_that_names_the_other_field_gets_that_one(self) -> None:
        """Older cuRobo configs call it retract_config. Whichever the descriptor uses is the one written."""
        config = copy.deepcopy(_ARM)
        cspace = config["robot_cfg"]["kinematics"]["cspace"]
        cspace["retract_config"] = cspace.pop("default_joint_position")

        composed = self.module.compose_sidecar_config(
            config, bodies=[_HAND], margin_mm=0.0, attach_spheres=0, default_q=_CHOSEN,
        )
        written = composed["robot_cfg"]["kinematics"]["cspace"]
        self.assertEqual(written["retract_config"], _CHOSEN)
        self.assertNotIn("default_joint_position", written)

    def test_a_cspace_with_neither_field_is_refused(self) -> None:
        config = copy.deepcopy(_ARM)
        config["robot_cfg"]["kinematics"]["cspace"].pop("default_joint_position")
        with self.assertRaises(self.module.BodyLinkError):
            self.module.compose_sidecar_config(
                config, bodies=[_HAND], margin_mm=0.0, attach_spheres=0, default_q=_CHOSEN,
            )

    def test_a_retract_of_the_wrong_length_is_refused_rather_than_padded(self) -> None:
        """⭐ THE CONTROL. Six joints is what this arm has; anything else is a pose for another robot."""
        with self.assertRaises(self.module.BodyLinkError) as caught:
            self.module.compose_sidecar_config(
                copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0, default_q=[0.0, 0.0, 0.0],
            )
        self.assertIn("6", str(caught.exception))

    def test_a_retract_that_is_not_numbers_is_refused(self) -> None:
        with self.assertRaises(self.module.BodyLinkError):
            self.module.compose_sidecar_config(
                copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0,
                default_q=[0.0, "somewhere", 0.0, 0.0, 0.0, 0.0],
            )

    def test_the_retract_is_the_only_thing_it_changes(self) -> None:
        """A composition that quietly edited anything else would be impossible to review."""
        without = self.module.compose_sidecar_config(
            copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0,
        )
        with_retract = self.module.compose_sidecar_config(
            copy.deepcopy(_ARM), bodies=[_HAND], margin_mm=0.0, attach_spheres=0, default_q=_CHOSEN,
        )
        for config in (without, with_retract):
            config["robot_cfg"]["kinematics"]["cspace"].pop("default_joint_position")
        self.assertEqual(without, with_retract)


if __name__ == "__main__":
    unittest.main()
