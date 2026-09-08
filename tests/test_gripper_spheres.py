"""The gripper the planner models is the gripper the cell has, and the committed maps prove it.

The planner represents a robot as spheres and nothing else, so a sphere map IS the gripper as far as
every plan is concerned. Two things can go wrong with that and neither shows up anywhere: the map can
describe a different hand than the one bolted on, and the committed map can drift away from the
generator that claims to produce it.

Both happened here. The on-box config builder carried a second copy of the fit and always used the
Robotiq bundle, under a comment calling it model independent, while the safety guard already selected
a per-gripper bundle through `collision_mesh_variant`. A Schunk cell therefore had a guard that knew
its hand and a planner that did not.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from src.robot.safety.planning.robot.build_gripper_spheres import build
from src.robot.safety.planning.robot.gripper_spheres import (
    GripperSpheres,
    SphereFitError,
    fit_gripper_spheres,
    fit_spheres_from_mesh,
    grid_fit_spheres,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLES = _ROOT / "src" / "robot" / "safety" / "data"
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"

#: Every gripper this repository ships a bundle and a committed map for.
_SHIPPED = ("ur5e", "schunk_egu50")


def _committed(name: str) -> list[dict]:
    path = _MAPS / f"{name}_gripper_spheres.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))["collision_spheres"]["tool0"]


class CommittedMapsTests(unittest.TestCase):
    def test_every_committed_map_is_what_the_fitter_produces(self) -> None:
        """The drift this ends: two fitters, one calling itself a mirror of the other, nothing checking."""
        for name in _SHIPPED:
            with self.subTest(gripper=name):
                fitted = fit_gripper_spheres(_BUNDLES / f"{name}_collision_meshes.npz")
                self.assertEqual(
                    _committed(name), fitted.to_dict()["tool0"],
                    f"{name}_gripper_spheres.yml is not what the fitter produces from its bundle. "
                    "Regenerate it with build_gripper_spheres.py.",
                )

    def test_every_hand_in_a_bundle_has_a_committed_map_somewhere(self) -> None:
        """The property, not the filename: whatever hand a bundle carries, the planner has its map.

        A map describes a HAND and the bundles are named after ARMS, so the same Robotiq appears in
        the ur5e bundle and in the ur3e one and needs one map between them. What must never happen is
        a bundle carrying an end effector that no committed map describes: the on-box builder reads
        these files, so that hand simply cannot be planned with.
        """
        committed = {name: _committed(name) for name in _SHIPPED}
        for bundle in sorted(_BUNDLES.glob("*_collision_meshes.npz")):
            name = bundle.name.replace("_collision_meshes.npz", "")
            with np.load(bundle, allow_pickle=True) as data:
                if "gripper__v" not in data:
                    continue
            with self.subTest(bundle=name):
                fitted = fit_gripper_spheres(bundle).to_dict()["tool0"]
                self.assertIn(
                    fitted, list(committed.values()),
                    f"{bundle.name} carries an end effector that no committed sphere map describes. "
                    f"Write one: build_gripper_spheres.py --variant {name}",
                )

    def test_two_grippers_do_not_produce_the_same_spheres(self) -> None:
        """The negative half. Without it, every assertion above would pass on one hand for all."""
        robotiq = fit_gripper_spheres(_BUNDLES / "ur5e_collision_meshes.npz")
        schunk = fit_gripper_spheres(_BUNDLES / "schunk_egu50_collision_meshes.npz")

        self.assertNotEqual(robotiq.count, schunk.count)
        self.assertNotEqual(robotiq.to_dict()["tool0"], schunk.to_dict()["tool0"])

    def test_the_provenance_says_which_hand_and_which_file(self) -> None:
        """A sphere map looks like any other list of numbers a year later."""
        for name in _SHIPPED:
            with self.subTest(gripper=name):
                block = yaml.safe_load(
                    (_MAPS / f"{name}_gripper_spheres.yml").read_text(encoding="utf-8")
                )["_provenance"]
                self.assertIn("gripper", block)
                self.assertIn("tool0", block["frame"])
                self.assertIn(name, block["source"])


class FitTests(unittest.TestCase):
    def test_a_sphere_covers_the_vertices_of_its_own_voxel(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
        (sphere,) = grid_fit_spheres(verts, cell_mm=100.0, rmax_mm=100.0)

        centre = np.asarray(sphere["center"]) * 1000.0
        radius = float(sphere["radius"]) * 1000.0
        for vertex in verts:
            self.assertLessEqual(
                float(np.linalg.norm(vertex - centre)), radius + 1e-6,
                "a vertex outside its own sphere is geometry the planner cannot see",
            )

    def test_the_radius_is_capped_so_one_stray_vertex_cannot_inflate_a_voxel(self) -> None:
        verts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 90.0]])
        (sphere,) = grid_fit_spheres(verts, cell_mm=1000.0, rmax_mm=20.0)

        self.assertAlmostEqual(float(sphere["radius"]) * 1000.0, 20.0, places=3)

    def test_a_bundle_with_no_end_effector_is_refused_by_name(self) -> None:
        """An arm bundle fitted as a gripper hangs the whole robot off the flange."""
        with self.assertRaises(SphereFitError) as caught:
            fit_gripper_spheres(_BUNDLES / "does_not_exist_collision_meshes.npz")
        self.assertIn("bake", str(caught.exception).lower())

    def test_a_mesh_with_no_scale_answer_is_refused_rather_than_guessed(self) -> None:
        """Metres read as millimetres is a hand the size of a grain of rice, and it plans."""
        with self.assertRaises(SphereFitError):
            fit_spheres_from_mesh(Path("nowhere.stl"), gripper="x", scale_to_mm=0.0)

    def test_a_missing_mesh_is_refused(self) -> None:
        with self.assertRaises(SphereFitError):
            fit_spheres_from_mesh(Path("nowhere.stl"), gripper="x")

    def test_the_written_file_carries_the_map_under_the_key_curobo_reads(self) -> None:
        fitted = GripperSpheres(
            spheres=({"center": [0.0, 0.0, 0.1], "radius": 0.02},), source="test", gripper="test"
        )
        written = build(fitted)

        self.assertEqual(list(written["collision_spheres"]), ["tool0"])
        self.assertEqual(written["collision_spheres"]["tool0"][0]["radius"], 0.02)
        self.assertIn("test", written["_provenance"]["source"])

    def test_the_description_names_the_hand_and_the_file(self) -> None:
        fitted = fit_gripper_spheres(_BUNDLES / "ur5e_collision_meshes.npz", gripper="Robotiq 2F-85")
        rendered = fitted.render()

        self.assertIn("Robotiq 2F-85", rendered)
        self.assertIn("ur5e_collision_meshes.npz", rendered)
        self.assertIn("sphere(s)", rendered)


if __name__ == "__main__":
    unittest.main()
