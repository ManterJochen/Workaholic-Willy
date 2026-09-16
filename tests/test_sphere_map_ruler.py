"""One ruler judges every collision sphere map, and it is right on a body that does not close (B6).

A sphere model makes two errors and only one is safe. A HOLE is a false clear: the planner believes part of the robot
is not there, clears a configuration the exact mesh guard then refuses, and on a cell without that guard clears one
where the arm actually is. REACH past the body is a false collide: picks and reachable workspace lost, nothing worse.

The claim "the refit is better than the vendor's map" is a claim about two maps, and it is worth nothing unless one
ruler measured both. So the ruler lives on its own (``scripts/curobo/_mesh_body.py``), the fitter measures its own work
with it, and the comparison CLI measures anybody's work with it.

What is tested here is the ruler against bodies whose answer is exact arithmetic rather than a mesh library's opinion:
a cube whose covering sphere reaches a distance anyone can write down, a cube whose inscribed sphere leaves a hole of
the same size, and a cube with a face taken out, which is where the generalised winding number earns its cost.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]

#: A cube of this half extent, in metres. Every number below is arithmetic on it.
_HALF = 0.05
#: The sphere through its corners, and the sphere through its face centres.
_CIRCUMSCRIBED = _HALF * math.sqrt(3.0)
#: How far the circumscribed sphere reaches past the face it is furthest from, and equally how deep the hole the
#: inscribed sphere leaves at a corner: both are (corner distance - face distance).
_GAP_MM = (_CIRCUMSCRIBED - _HALF) * 1000.0


def _load(name: str):
    path = _ROOT / "scripts" / "curobo" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _cube(centre=(0.0, 0.0, 0.0)):
    import trimesh

    mesh = trimesh.creation.box(extents=(2 * _HALF, 2 * _HALF, 2 * _HALF))
    mesh.apply_translation(np.asarray(centre, dtype=np.float64))
    return mesh


def _cube_with_a_face_taken_out():
    """The same cube, missing the two triangles of its +Z face: a body that does not close.

    A ray cast upward from the inside leaves through the hole without crossing anything and reads "outside". The
    generalised winding number sees 5 of 6 faces worth of solid angle and reads 5/6, which is inside.
    """
    mesh = _cube()
    keep = [index for index, normal in enumerate(mesh.face_normals) if normal[2] < 0.99]
    return mesh.submesh([keep], append=True)


class MeasureAMapAgainstTheBody(unittest.TestCase):
    """The two questions, asked of maps whose answer is known before the ruler runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load("_mesh_body")
        cls.body = cls.module.MeshBody(_cube(), samples=8_000, seed=3)

    def test_a_sphere_through_the_corners_covers_the_cube_and_says_how_far_it_reaches(self) -> None:
        """No hole, and a reach that is the distance from the sphere to the face it is furthest from."""
        fidelity = self.body.measure(np.zeros((1, 3)), np.asarray([_CIRCUMSCRIBED]))

        self.assertFalse(fidelity.has_hole, fidelity.render())
        self.assertLess(fidelity.uncovered_max_mm, 0.0)
        # A finite set of directions can only find a reach the body really has, so the reading is a lower bound and
        # may never exceed the arithmetic. 256 directions sit about 13 degrees apart, which cannot lose a fifth of it.
        self.assertLessEqual(fidelity.reach_max_mm, _GAP_MM + 1e-6)
        self.assertGreater(fidelity.reach_max_mm, 0.8 * _GAP_MM)

    def test_a_sphere_through_the_face_centres_leaves_the_corners_open(self) -> None:
        """The unsafe error, and the one the fitter exists to make impossible: a hole the planner cannot see."""
        fidelity = self.body.measure(np.zeros((1, 3)), np.asarray([_HALF]))

        self.assertTrue(fidelity.has_hole, fidelity.render())
        # The deepest point is the corner itself, which a surface sample lands near rather than on, so the reading is
        # again a lower bound on a gap of _GAP_MM.
        self.assertLessEqual(fidelity.uncovered_max_mm, _GAP_MM + 1e-6)
        self.assertGreater(fidelity.uncovered_max_mm, 0.7 * _GAP_MM)
        self.assertAlmostEqual(fidelity.reach_max_mm, 0.0, places=6,
                               msg="a sphere inside the body reaches past nothing")

    def test_a_map_the_ruler_cannot_read_is_refused_rather_than_measured(self) -> None:
        with self.assertRaises(ValueError):
            self.body.measure(np.zeros((2, 3)), np.asarray([_HALF]))
        with self.assertRaises(ValueError):
            self.body.measure(np.zeros((0, 3)), np.asarray([]))

    def test_the_reading_carries_its_own_sample_count_and_renders_as_a_sentence(self) -> None:
        fidelity = self.body.measure(np.zeros((1, 3)), np.asarray([_CIRCUMSCRIBED]))

        self.assertEqual(fidelity.samples, len(self.body.points))
        self.assertEqual(fidelity.spheres, 1)
        self.assertIn("no hole", fidelity.render())
        self.assertEqual(sorted(fidelity.to_dict()), ["reach_max_mm", "samples", "spheres", "uncovered_max_mm"])


class AGroupOfMeshesIsOneBody(unittest.TestCase):
    """A hand's three parts share the flange frame: a sphere leaving the gripper for the finger has not left the hand."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load("_mesh_body")
        cls.apart = (0.0, 0.0, 0.3)
        cls.body = cls.module.MeshBody([_cube(), _cube(cls.apart)], samples=8_000, seed=5)

    def test_a_point_in_either_mesh_is_inside_the_group(self) -> None:
        inside = self.body.inside(np.asarray([(0.0, 0.0, 0.0), self.apart, (0.0, 0.0, 0.15)]))

        self.assertTrue(bool(inside[0]))
        self.assertTrue(bool(inside[1]), "a point inside the second mesh is inside the group")
        self.assertFalse(bool(inside[2]), "the gap between them is not part of the body")

    def test_a_sphere_inside_the_second_mesh_reaches_past_nothing(self) -> None:
        """Measured against one mesh at a time this sphere would read as 300 mm of reach, which would be a lie."""
        fidelity = self.body.measure(np.asarray([self.apart]), np.asarray([_HALF]))

        self.assertAlmostEqual(fidelity.reach_max_mm, 0.0, places=6)

    def test_both_meshes_are_sampled(self) -> None:
        """A group sampled from only one mesh would report the other as one enormous hole, or miss it entirely."""
        upper = self.body.points[:, 2] > 0.15

        self.assertGreater(int(upper.sum()), 0)
        self.assertGreater(int((~upper).sum()), 0)


class TheInsideTestOnABodyThatDoesNotClose(unittest.TestCase):
    """7 of 45 bodies in the committed bundles are not watertight; there a ray test is wrong by up to 22 mm."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load("_mesh_body")

    def test_the_winding_number_reads_the_inside_of_an_open_box_as_inside(self) -> None:
        open_box = _cube_with_a_face_taken_out()

        wound = self.module.winding(open_box, np.asarray([(0.0, 0.0, 0.0)]))

        # Five sixths of the solid angle around the centre is still covered, which is inside by any threshold at 0.5.
        self.assertAlmostEqual(float(wound[0]), 5.0 / 6.0, places=3)
        self.assertTrue(bool(self.module.MeshBody(open_box, samples=500, seed=1)
                             .inside(np.asarray([(0.0, 0.0, 0.0)]))[0]))

    def test_a_point_outside_an_open_box_is_still_outside(self) -> None:
        """The other half: a test that says "inside" everywhere would pass the one above and be worthless."""
        open_box = _cube_with_a_face_taken_out()

        wound = self.module.winding(open_box, np.asarray([(0.0, 0.0, 0.4), (0.4, 0.0, 0.0)]))

        self.assertLess(abs(float(wound[0])), 0.5)
        self.assertLess(abs(float(wound[1])), 0.5)


class ReadAnybodysMap(unittest.TestCase):
    """The comparison is worth nothing unless the ruler can read the map it is comparing against."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _load("measure_sphere_map")

    def test_the_vendor_shape_a_flat_list_under_one_key(self) -> None:
        held = self.module.read_spheres(
            {"collision_spheres": {"tool0": [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]}})

        self.assertEqual(sorted(held), ["tool0"])
        self.assertEqual(held["tool0"][1].tolist(), [0.02])

    def test_the_fitted_shape_a_block_per_body(self) -> None:
        held = self.module.read_spheres({"collision_spheres": {
            "gripper": {"frame": 6, "spheres": [{"center": [0.0, 0.0, 0.0], "radius": 0.02}]},
            "lfinger": {"frame": 6, "spheres": [{"center": [0.0, 0.1, 0.0], "radius": 0.01}]},
        }})

        self.assertEqual(sorted(held), ["gripper", "lfinger"])
        self.assertEqual(held["lfinger"][0].tolist(), [[0.0, 0.1, 0.0]])

    def test_a_built_descriptor(self) -> None:
        held = self.module.read_spheres({"robot_cfg": {"kinematics": {"collision_spheres": {
            "forearm_link": [{"center": [0.0, 0.0, 0.0], "radius": 0.04}]}}}})

        self.assertEqual(sorted(held), ["forearm_link"])

    def test_spheres_that_are_a_path_to_another_file_are_refused_by_name(self) -> None:
        """A stock descriptor names a file. Reading that as an empty map would report a body with no spheres as fine."""
        with self.assertRaises(SystemExit) as raised:
            self.module.read_spheres({"robot_cfg": {"kinematics": {"collision_spheres": "spheres/ur5e.yml"}}})

        self.assertIn("spheres/ur5e.yml", str(raised.exception))

    def test_a_document_with_no_spheres_at_all_is_refused(self) -> None:
        with self.assertRaises(SystemExit):
            self.module.read_spheres({"robot_cfg": {"kinematics": {}}})


if __name__ == "__main__":
    unittest.main()
