"""Which two links overlap, and by how much, derived from the config the planner loaded (B1, S14).

A cuRobo refusal says "no collision-free plan". The operator's next question is always the same: what touched what. The
planner knows it only as a number over a flat array of spheres, so the answer has to come from sphere OWNERSHIP in the
loaded descriptor: ``collision_link_names`` in order, the count per link from ``collision_spheres``, the spare slots
from ``extra_collision_spheres``, the padding from ``self_collision_buffer``, and the pairs ``self_collision_ignore``
takes out.

Pure arithmetic on plain data, so the python 3.11 suite tests exactly what the python 3.10 sidecar runs. What it cannot
prove here is that cuRobo lays its ``robot_spheres`` out in that order: only the box can, by naming a pair and its depth
where a probe measured the same one.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _pairs():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "_curobo_pairs.py"
    spec = importlib.util.spec_from_file_location("_curobo_pairs_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(*, buffer: float = 0.0, ignore: "dict | None" = None, links: "list[str] | None" = None) -> dict:
    """Two links, one sphere each, 12 mm inside one another: radii 0.05 m at 0.088 m apart."""
    return {
        "robot_cfg": {"kinematics": {
            "collision_link_names": list(links or ["wrist_1_link", "tool0"]),
            "collision_spheres": {
                "wrist_1_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05}],
                "tool0": [{"center": [0.088, 0.0, 0.0], "radius": 0.05}],
            },
            "self_collision_buffer": {"wrist_1_link": buffer, "tool0": buffer},
            "self_collision_ignore": dict(ignore or {}),
        }},
    }


def _spheres(*centres_and_radii: "tuple[list[float], float]") -> np.ndarray:
    """One pose, one row per sphere: ``(1, S, 4)`` in metres, as cuRobo hands them over."""
    return np.asarray([[[*centre, radius] for centre, radius in centres_and_radii]], dtype=np.float64)


_TOUCHING = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05))


class TheDeepestPairIsNamedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = _pairs()

    def test_two_overlapping_links_are_named_with_their_depth(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_config())
        (deepest,) = self.module.deepest_pairs(_TOUCHING, layout)
        self.assertEqual((deepest.link_a, deepest.link_b), ("tool0", "wrist_1_link"))
        self.assertAlmostEqual(deepest.depth_mm, 12.0, places=6)
        self.assertIn("12.0", deepest.render())
        self.assertEqual(deepest.to_dict()["depth_mm"], deepest.depth_mm)

    def test_the_guard_margin_in_the_buffers_deepens_it(self) -> None:
        """cuRobo subtracts both links' buffers from a pair's distance, so 5 mm each reads as 10 mm more overlap."""
        layout = self.module.SphereLayout.from_robot_config(_config(buffer=0.005))
        (deepest,) = self.module.deepest_pairs(_TOUCHING, layout)
        self.assertAlmostEqual(deepest.depth_mm, 22.0, places=6)

    def test_an_ignored_pair_a_single_link_and_an_empty_slot_are_not_collisions(self) -> None:
        module = self.module
        ignored = module.SphereLayout.from_robot_config(_config(ignore={"wrist_1_link": ["tool0"]}))
        self.assertEqual(module.deepest_pairs(_TOUCHING, ignored), (None,))

        one_link = module.SphereLayout.from_robot_config({
            "robot_cfg": {"kinematics": {
                "collision_link_names": ["wrist_1_link"],
                "collision_spheres": {"wrist_1_link": [{"center": [0, 0, 0], "radius": 0.05},
                                                       {"center": [0.088, 0, 0], "radius": 0.05}]},
            }},
        })
        self.assertEqual(module.deepest_pairs(_TOUCHING, one_link), (None,))

        empty = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], -100.0))
        self.assertEqual(module.deepest_pairs(empty, module.SphereLayout.from_robot_config(_config())), (None,))

    def test_the_owner_of_a_slot_follows_the_link_order(self) -> None:
        """⭐ THE CONTROL. The array is flat: only ``collision_link_names``, read in order and counted per link,
        says whose sphere slot 0 is. Here ``wrist_1_link`` owns two slots and ``tool0`` one, and the same three
        spheres are a single link's own business under one order and a real collision under the other, so an
        implementation that ignored the order could not answer both."""
        module = self.module
        two_and_one = {
            "collision_spheres": {
                "wrist_1_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.05},
                                 {"center": [0.088, 0.0, 0.0], "radius": 0.05}],
                "tool0": [{"center": [0.4, 0.0, 0.0], "radius": 0.05}],
            },
            "self_collision_ignore": {},
        }
        spheres = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05), ([0.4, 0.0, 0.0], 0.05))

        straight = module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": ["wrist_1_link", "tool0"], **two_and_one}}})
        self.assertEqual(straight.owners, ("wrist_1_link", "wrist_1_link", "tool0"))
        self.assertEqual(module.deepest_pairs(spheres, straight), (None,))

        swapped = module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": ["tool0", "wrist_1_link"], **two_and_one}}})
        self.assertEqual(swapped.owners, ("tool0", "wrist_1_link", "wrist_1_link"))
        (deepest,) = module.deepest_pairs(spheres, swapped)
        assert deepest is not None
        self.assertEqual((deepest.link_a, deepest.link_b), ("tool0", "wrist_1_link"))
        self.assertAlmostEqual(deepest.depth_mm, 12.0, places=6)

    def test_spare_slots_are_counted_so_a_payload_does_not_shift_every_owner(self) -> None:
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_link_names"].append("attached_object")
        config["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 4}
        layout = self.module.SphereLayout.from_robot_config(config)
        self.assertEqual(layout.owners, ("wrist_1_link", "tool0", "attached_object", "attached_object",
                                         "attached_object", "attached_object"))
        self.assertEqual(layout.slots, 6)

    def test_a_descriptor_whose_spheres_are_a_file_has_no_layout_and_does_not_refuse(self) -> None:
        """A stock cuRobo descriptor resolves ``collision_spheres`` from a file as it loads, and the ownership is gone
        by the time anything here sees it. That costs the operator a pair NAME. It must not cost them a planner, so
        there is no layout and no exception."""
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_spheres"] = "spheres/ur5e.yml"
        self.assertIsNone(self.module.SphereLayout.from_robot_config(config))
        self.assertIsNone(self.module.SphereLayout.from_robot_config({"robot_cfg": {}}))
        self.assertIsNone(self.module.SphereLayout.from_robot_config(
            {"robot_cfg": {"kinematics": {"collision_link_names": [], "collision_spheres": {}}}}))

    def test_a_sphere_array_the_layout_does_not_describe_is_refused(self) -> None:
        """⭐ THE CONTROL that a wrong layout cannot be papered over: cuRobo's array and the owners must agree in
        length, or every name after the mismatch is somebody else's link."""
        layout = self.module.SphereLayout.from_robot_config(_config())
        with self.assertRaises(self.module.SphereLayoutError):
            self.module.deepest_pairs(_spheres(([0.0, 0.0, 0.0], 0.05)), layout)

    def test_only_the_deepest_pair_of_a_pose_is_reported(self) -> None:
        """Three links overlap at once. The operator gets the pair that decides the refusal, not a list."""
        config = _config()
        config["robot_cfg"]["kinematics"]["collision_link_names"] = ["wrist_1_link", "tool0", "wrist_3_link"]
        config["robot_cfg"]["kinematics"]["collision_spheres"]["wrist_3_link"] = [
            {"center": [0.0, 0.0, 0.06], "radius": 0.05},
        ]
        layout = self.module.SphereLayout.from_robot_config(config)
        # wrist_1 and tool0 overlap by 12 mm, wrist_1 and wrist_3 by 40 mm, tool0 and wrist_3 not at all.
        spheres = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.088, 0.0, 0.0], 0.05), ([0.0, 0.0, 0.06], 0.05))
        (deepest,) = self.module.deepest_pairs(spheres, layout)
        assert deepest is not None
        self.assertEqual((deepest.link_a, deepest.link_b), ("wrist_1_link", "wrist_3_link"))
        self.assertAlmostEqual(deepest.depth_mm, 40.0, places=6)

    def test_one_verdict_per_pose(self) -> None:
        layout = self.module.SphereLayout.from_robot_config(_config())
        clear = _spheres(([0.0, 0.0, 0.0], 0.05), ([0.5, 0.0, 0.0], 0.05))
        poses = np.concatenate([_TOUCHING, clear], axis=0)
        verdicts = self.module.deepest_pairs(poses, layout)
        self.assertEqual(len(verdicts), 2)
        self.assertIsNotNone(verdicts[0])
        self.assertIsNone(verdicts[1])


if __name__ == "__main__":
    unittest.main()
