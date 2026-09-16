"""The gripper the planner models is the gripper the cell has, and the committed maps prove it.

The planner represents a robot as spheres and nothing else, so a sphere map IS the gripper as far as
every plan is concerned. Two things can go wrong with that and neither shows up anywhere: the map can
describe a different hand than the one bolted on, and the committed map can drift away from the
generator that claims to produce it.

Both happened here. The on-box config builder carried a second copy of the fit and always used the
Robotiq bundle, under a comment calling it model independent, while the safety guard already selected
a per-gripper bundle through its variant key. A Schunk cell therefore had a guard that knew
its hand and a planner that did not.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

from src.robot.safety.planning.robot.build_gripper_spheres import build
from src.robot.safety.planning.robot.gripper_spheres import (
    FLANGE,
    MOUNTING_FACE,
    GripperSpheres,
    SphereFitError,
    bundle_origin,
    fit_gripper_spheres,
    fit_spheres_from_mesh,
    grid_fit_spheres,
)

_ROOT = Path(__file__).resolve().parents[1]
_BUNDLES = _ROOT / "src" / "robot" / "safety" / "data"
_MAPS = _ROOT / "src" / "robot" / "safety" / "planning" / "robot"

#: Every gripper this repository ships a committed map for, read from the directory rather than
#: listed beside it. A hand-kept list has to be remembered. Since UM lane S08 a map and its bundle
#: do pair by name, one `{hand}_gripper_spheres.yml` over one `{hand}_hand_meshes.npz`.
_SHIPPED = tuple(sorted(
    path.name.replace("_gripper_spheres.yml", "")
    for path in _MAPS.glob("*_gripper_spheres.yml")
))


def _ruler():
    """The one ruler that judges any sphere map against the exact meshes (`scripts/curobo/_mesh_body.py`).

    Loaded by path because it lives beside the cuRobo work rather than in the backend package, and used here
    for the same reason the fitter uses it: a claim about a map is worth nothing unless the thing that made
    the map and the thing that checks it are not the same code.
    """
    path = _ROOT / "scripts" / "curobo" / "_mesh_body.py"
    spec = importlib.util.spec_from_file_location("_mesh_body_for_gripper_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _map(name: str) -> dict:
    path = _MAPS / f"{name}_gripper_spheres.yml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _committed(name: str) -> list[dict]:
    return _map(name)["collision_spheres"]["tool0"]


def _source_bundle(name: str) -> Path:
    """The bundle a committed map says it was fitted from.

    Read from the map's own provenance rather than reconstructed from its name. That is what makes
    the drift check work for a hand with one map and several bundles, and it gives the provenance
    block a reader: until now nothing read it, so a wrong `source` was undetectable.
    """
    return _BUNDLES / _map(name)["_provenance"]["source"]


class CommittedMapsTests(unittest.TestCase):
    def test_there_is_at_least_one_committed_map(self) -> None:
        """`_SHIPPED` is read from a directory now, and a glob that matches nothing makes every
        assertion below pass over an empty loop. A test over no gripper passes loudest."""
        self.assertGreaterEqual(len(_SHIPPED), 2, f"only found {_SHIPPED}")

    def test_every_committed_map_names_a_bundle_that_is_here(self) -> None:
        """A map whose source is missing describes nothing anybody can check."""
        for name in _SHIPPED:
            with self.subTest(gripper=name):
                bundle = _source_bundle(name)
                self.assertTrue(
                    bundle.is_file(),
                    f"{name}_gripper_spheres.yml names {bundle.name} as its source and that file "
                    "is not here, so nothing can check what the map describes.",
                )

    def test_every_committed_map_records_no_hole_and_the_reach_it_was_fitted_at(self) -> None:
        """⭐ WHAT A COMMITTED MAP NOW CLAIMS (B6). Not "this is what generator X emits", which was true and
        said nothing about the geometry: every surface sample of the hand is INSIDE a sphere, and no sphere
        reaches further past the hand than the reach the fit was given.

        Read here from what the fit measured on a fresh sample it never fitted against. The independent
        re-measurement is the test below; this one holds every hand, cheaply.
        """
        for name in _SHIPPED:
            rows = _map(name)["_provenance"]["bodies"]
            self.assertEqual(sorted(row["body"] for row in rows), ["gripper", "lfinger", "rfinger"],
                             f"{name} does not describe the three bodies its bundle carries")
            for row in rows:
                with self.subTest(gripper=name, body=row["body"]):
                    self.assertLess(row["fresh_uncovered_max_mm"], 0.0,
                                    "a surface point of this body lies outside every sphere")
                    self.assertLessEqual(row["fresh_reach_max_mm"], row["reach_mm"] + 0.1,
                                         "a sphere reaches further past the hand than it was fitted at")

    def test_the_map_really_does_cover_the_hand_measured_here_rather_than_read_from_the_file(self) -> None:
        """⭐ THE INDEPENDENT ONE. Everything above reads numbers the fitter wrote about itself, and a file
        that lies about its own quality would pass all of it. This measures the committed spheres against the
        committed mesh with the shared ruler, on the hand that blocks two arms.

        Coarser than the fit's own check, because it runs in the suite: the reading is a lower bound on the
        reach and an under-estimate of any hole, which is the safe direction for both.
        """
        ruler = _ruler()
        name = "schunk_egu50"
        body = ruler.MeshBody(
            ruler.load_meshes(_BUNDLES / f"{name}_hand_meshes.npz", ["gripper", "lfinger", "rfinger"]),
            samples=4_000, seed=11)
        centres = np.asarray([sphere["center"] for sphere in _committed(name)])
        radii = np.asarray([sphere["radius"] for sphere in _committed(name)])

        held = body.measure(centres, radii, directions=48)

        self.assertFalse(held.has_hole, held.render())
        self.assertLessEqual(held.reach_max_mm, 12.5, held.render())

    def test_the_measurement_would_notice_a_map_that_did_not_cover_the_hand(self) -> None:
        """⭐ THE CONTROL. Without it the test above passes on any map at all, including an empty one."""
        ruler = _ruler()
        name = "schunk_egu50"
        body = ruler.MeshBody(
            ruler.load_meshes(_BUNDLES / f"{name}_hand_meshes.npz", ["gripper", "lfinger", "rfinger"]),
            samples=4_000, seed=11)
        # Half the map, which is what a truncated file looks like and what the vendor's own map measured as:
        # 46 spheres, an 18.9 mm hole and 22.6 mm of reach (2026-09-16).
        half = _committed(name)[::2]

        held = body.measure(np.asarray([s["center"] for s in half]), np.asarray([s["radius"] for s in half]),
                            directions=48)

        self.assertTrue(held.has_hole, held.render())

    def test_every_hand_in_a_bundle_has_a_committed_map_somewhere(self) -> None:
        """The property, not the filename: whatever hand a bundle carries, the planner has its map.

        A map describes a HAND and the bundles are named after ARMS, so the same Robotiq appears in
        the ur5e bundle and in the ur3e one and needs one map between them. What must never happen is
        a bundle carrying an end effector that no committed map describes: the on-box builder reads
        these files, so that hand simply cannot be planned with.
        """
        # Arm bundles carry the 2F-85, and every other hand is its own bundle since UM lane S05. Reading only the arm
        # bundles would drop the Hand-E and the EGU-50 out of this property without a single assertion failing.
        bundles = sorted([*_BUNDLES.glob("*_collision_meshes.npz"), *_BUNDLES.glob("*_hand_meshes.npz")])
        self.assertTrue(any(b.name.endswith("_hand_meshes.npz") for b in bundles),
                        "no hand bundle was read, so no hand but the 2F-85 is checked here")
        # Matched on the GEOMETRY, not on a fitter's output: the maps are cover fits now (B6) and refitting one
        # in a test costs minutes. Two bundles describe the same hand when their gripper arrays are the same
        # arrays, which is the fact the old comparison was standing in for.
        described = {}
        for name in _SHIPPED:
            with np.load(_source_bundle(name), allow_pickle=True) as data:
                described[name] = data["gripper__v"].tobytes()
        for bundle in bundles:
            name = bundle.name.replace("_collision_meshes.npz", "").replace("_hand_meshes.npz", "")
            with np.load(bundle, allow_pickle=True) as data:
                if "gripper__v" not in data:
                    continue
                carried = data["gripper__v"].tobytes()
            with self.subTest(bundle=name):
                self.assertIn(
                    carried, described.values(),
                    f"{bundle.name} carries an end effector that no committed sphere map describes. "
                    f"Fit one: scripts/curobo/fit_cover_spheres.py --hand {name} --write",
                )

    def test_two_grippers_do_not_produce_the_same_spheres(self) -> None:
        """The negative half. Without it, every assertion above would pass on one hand for all."""
        robotiq = fit_gripper_spheres(_BUNDLES / "ur5e_collision_meshes.npz")
        schunk = fit_gripper_spheres(_BUNDLES / "schunk_egu50_hand_meshes.npz")

        self.assertNotEqual(robotiq.count, schunk.count)
        self.assertNotEqual(robotiq.to_dict()["tool0"], schunk.to_dict()["tool0"])

    def test_the_provenance_says_which_hand_and_which_file(self) -> None:
        """A sphere map looks like any other list of numbers a year later."""
        for name in _SHIPPED:
            with self.subTest(gripper=name):
                block = _map(name)["_provenance"]
                self.assertIn("gripper", block)
                self.assertIn("tool0", block["frame"])
                # Every hand is fitted from its OWN bundle now, the 2F-85 included. It used to be fitted from an
                # arm's bundle because that is where it was baked; measured 2026-09-16, the gripper, lfinger
                # and rfinger arrays of robotiq_2f85_hand_meshes.npz are byte for byte the ur5e bundle's, so
                # the change is which file is named and not which geometry was fitted.
                self.assertIn(name, block["source"])


class WhereTheNumbersStartTests(unittest.TestCase):
    """A sphere map is a list of numbers in the tool0 frame, and whether they already sit where
    the hand is bolted is the one thing about them a reader cannot recover by looking.

    Two hands, two answers. The 2F-85 bundle came out of a composed arm asset, so the arm had
    already placed it and its origin is the flange. The Hand-E was read from a standalone vendor
    asset that holds no coupling at all, so its numbers start at the gripper own mounting face and
    the plate between that face and the flange is a bench measurement. Getting it wrong puts every
    sphere one plate too close to the flange, which is optimistic in the one direction a planner
    must not be, and the file looks entirely reasonable.
    """

    def test_every_committed_map_declares_where_it_starts(self) -> None:
        for name in _SHIPPED:
            with self.subTest(gripper=name):
                self.assertIn(_map(name)["_provenance"].get("origin"), (FLANGE, MOUNTING_FACE))

    def test_the_two_kinds_of_bundle_answer_differently(self) -> None:
        """The discriminating pair. Without it the field could be constant and every assertion
        above would still pass."""
        self.assertEqual(bundle_origin(_BUNDLES / "ur5e_collision_meshes.npz"), FLANGE)
        self.assertEqual(bundle_origin(_BUNDLES / "robotiq_hande_hand_meshes.npz"), MOUNTING_FACE)

    def test_a_hand_that_needs_a_coupling_says_so_when_rendered(self) -> None:
        """The operator reads `render()`, not the npz key."""
        text = fit_gripper_spheres(_BUNDLES / "robotiq_hande_hand_meshes.npz", gripper="Robotiq Hand-E").render()
        self.assertIn("coupling", text)
        self.assertNotIn(
            "coupling",
            fit_gripper_spheres(_BUNDLES / "ur5e_collision_meshes.npz").render(),
        )


class OneHandOnEveryArmTests(unittest.TestCase):
    """⛔ A HAND IS ONE BUNDLE, COMPOSED ONTO EVERY ARM WHEN THE GUARD LOADS (UM lane S05, S08).

    It used to be one arm plus hand file per arm, and naming the arm in a config key is how a cell that changes arms
    kept the bundle for the old one. Measured before this: `schunk_egu50` on a ur3e tripped `variant_model_mismatch`,
    which dropped the whole cell to the capsule proxy. A hand bundle carries no arm, and records the arms it was proven
    on instead.
    """

    def test_the_hand_composes_onto_both_arms(self) -> None:
        from src.robot.safety.planning.environment import compose_collision_meshes

        for arm in ("ur5e", "ur3e"):
            with self.subTest(arm=arm):
                composed = compose_collision_meshes(arm, "robotiq_hande")
                self.assertIn("gripper__v", composed)
                self.assertIn("forearm__v", composed)

    def test_a_hand_proven_on_one_arm_is_not_offered_to_another(self) -> None:
        """The control. Composing the arm in must not turn into finding something for every
        combination: the Schunk was proven on a ur5e only, and a ur3e cell must still be told."""
        from src.robot.safety._fcl_self_collision import mesh_backend_status

        self.assertEqual(
            mesh_backend_status("ur3e", None, "schunk_egu50"), "variant_model_mismatch"
        )
        for arm in ("ur5e", "ur3e"):
            with self.subTest(arm=arm):
                self.assertEqual(mesh_backend_status(arm, None, "robotiq_hande"), "ok")

    def test_the_arm_links_are_the_committed_ones_byte_for_byte(self) -> None:
        """Composing replaces a hand. If it changed an arm link too, a difference in behaviour
        could come from either, and the arm bundle would stop being a control."""
        from src.robot.safety.planning.environment import compose_collision_meshes

        for arm in ("ur5e", "ur3e"):
            with self.subTest(arm=arm):
                composed = compose_collision_meshes(arm, "robotiq_hande")
                with np.load(_BUNDLES / f"{arm}_collision_meshes.npz") as base:
                    links = [k for k in base.files if not k.startswith(("gripper__", "lfinger__", "rfinger__"))]
                    self.assertGreaterEqual(len(links), 6, "no arm links to compare")
                    for key in links:
                        self.assertTrue(np.array_equal(base[key], composed[key]),
                                        f"{key} differs between {arm} and {arm} with the Hand-E")

    def test_the_hand_is_not_the_arms_own_hand(self) -> None:
        """The other half: the three gripper arrays must have changed, or the composition is a copy."""
        from src.robot.safety.planning.environment import compose_collision_meshes

        for arm in ("ur5e", "ur3e"):
            with self.subTest(arm=arm):
                composed = compose_collision_meshes(arm, "robotiq_hande")
                with np.load(_BUNDLES / f"{arm}_collision_meshes.npz") as base:
                    self.assertFalse(np.array_equal(base["gripper__v"], composed["gripper__v"]))


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
            fit_spheres_from_mesh(
                Path("nowhere.stl"), gripper="x", scale_to_mm=0.0, cell_mm=34.0, rmax_mm=17.0
            )

    def test_a_missing_mesh_is_refused(self) -> None:
        with self.assertRaises(SphereFitError):
            fit_spheres_from_mesh(
                Path("nowhere.stl"), gripper="x", scale_to_mm=1.0, cell_mm=34.0, rmax_mm=17.0
            )

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
