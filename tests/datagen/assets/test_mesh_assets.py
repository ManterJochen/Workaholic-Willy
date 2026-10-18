"""Scanned meshes as solids: closed exactly, intersected exactly, and labelled soundly.

THE MEASUREMENT THAT SHAPED THIS MODULE, and the reason the closing step exists at all: only 44 of
the 181 placeable meshes (24.3 %) are watertight as shipped, and trimesh's own repair fixes exactly
none of them. 135 of the 137 broken ones fail on OPEN BOUNDARIES only -- clean simple loops, the
underside a scanner never saw -- and capping those loops takes the bank to 148 / 181 (81.8 %).

Watertightness is not a nicety here. A ray span and a nearest-surface normal are exact on any triangle
soup; *is this point inside the object* is only defined for a closed surface, and that is the question
the finger-collision test asks thousands of times per object.

The sharpest test in this file is `MeshLabelsAreSoundTests`: a BOX handed to the mesh path must produce
grasps that the closed-form box verdict independently accepts. The two implementations share no code,
so agreement is evidence. It is also how the float32 tie defect was caught -- see the test's docstring.

Every mesh here is built with trimesh rather than read from the library, so the file runs anywhere.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from datagen.assets.mesh_geometry import MeshShape, close_mesh, load_mesh_shape
from datagen.grasps.labels import SceneGeometry, label_jaw_grasps
from datagen.grasps.shapes import Solid, quat_to_matrix, solid_from_mesh
from datagen.grasps.verdict import JawGrasp, JawModel, check_jaw_grasp

EXTENT_MM = (70.0, 50.0, 90.0)


def _box(extent_mm=EXTENT_MM):
    import trimesh

    return trimesh.creation.box(extents=np.asarray(extent_mm, dtype=float) / 1000.0)


def _shape_from(mesh) -> MeshShape:
    """A `MeshShape` straight from a trimesh object, the way `load_mesh_shape` builds one."""
    closed = close_mesh(mesh)
    assert closed is not None
    vertices = np.asarray(closed.vertices, dtype=float) * 1000.0
    lower, upper = vertices.min(axis=0), vertices.max(axis=0)
    centre = (lower + upper) / 2.0
    return MeshShape(
        vertices_mm=vertices - centre,
        faces=np.asarray(closed.faces),
        face_normals=np.asarray(closed.face_normals),
        extent_mm=tuple(upper - lower),  # type: ignore[arg-type]
        origin_offset_mm=tuple(-centre),  # type: ignore[arg-type]
        source_path="box",
    )


def _pair(extent_mm=EXTENT_MM, position_mm=(450.0, 10.0, 45.0), quat=(0.0, 0.0, 0.0, 1.0)):
    """The same box twice: once as the closed-form primitive, once through the mesh path."""
    analytic = Solid(
        kind="box", half_extent_mm=np.asarray(extent_mm, dtype=float) / 2.0,
        rotation=quat_to_matrix(quat), centre_mm=np.asarray(position_mm, dtype=float), instance_id=0,
    )
    meshy = solid_from_mesh(_shape_from(_box(extent_mm)), position_mm, quat, instance_id=0)
    return analytic, meshy


class ClosingAMeshTests(unittest.TestCase):
    def test_an_open_box_is_capped_into_a_solid(self) -> None:
        """The bank's failure mode in miniature: a scan missing the face it stood on."""
        import trimesh

        box = _box()
        open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[2:], process=False)
        self.assertFalse(open_box.is_watertight, "the fixture must actually be open")

        closed = close_mesh(open_box)
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertTrue(closed.is_watertight)
        self.assertGreater(closed.volume, 0.0)

    def test_capping_keeps_the_original_bounds(self) -> None:
        """The fan adds interior points only, so extents measured before and after must agree.

        Load-bearing: layout spaces objects by the extent it measured from the RAW mesh, while the
        labeller works from the closed one. A cap that grew the object would place it by one size and
        label it as another.
        """
        import trimesh

        box = _box()
        open_box = trimesh.Trimesh(vertices=box.vertices, faces=box.faces[2:], process=False)
        closed = close_mesh(open_box)
        assert closed is not None
        np.testing.assert_allclose(closed.bounds, box.bounds, atol=1e-9)

    def test_a_mesh_with_meeting_boundaries_is_REFUSED(self) -> None:
        """Not a best effort. Two openings that share a vertex cannot be capped without guessing."""
        import trimesh

        # A vertex shared by two separate triangle fans: its boundary vertices have degree 4.
        vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                             [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        faces = np.array([[0, 1, 2], [0, 3, 4]])
        self.assertIsNone(close_mesh(trimesh.Trimesh(vertices=vertices, faces=faces, process=False)))

    def test_a_flat_sheet_is_refused_rather_than_labelled(self) -> None:
        """Capping an open flat scan closes it, and the result has no inside to answer questions about.

        A SHEET, not a thin box: a thin box is a perfectly good solid and is deliberately still
        accepted -- the bank contains real flat objects. What is refused is a surface with no interior,
        which is what a single-sided scan becomes once its boundary is capped.
        """
        import trimesh

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "sheet.obj"
            vertices = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.1, 0.0], [0.0, 0.1, 0.0]])
            trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 1, 2], [0, 2, 3]]),
                            process=False).export(path)
            self.assertIsNone(load_mesh_shape(str(path)))

    def test_a_thin_but_solid_object_is_KEPT(self) -> None:
        """The other half of that claim, so the sheet guard cannot quietly become a thinness filter."""
        import trimesh

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "plate.obj"
            trimesh.creation.box(extents=(0.1, 0.1, 0.004)).export(path)
            shape = load_mesh_shape(str(path))
            self.assertIsNotNone(shape)
            assert shape is not None
            np.testing.assert_allclose(shape.extent_mm, (100.0, 100.0, 4.0), atol=1e-6)

    def test_the_shape_is_centred_on_its_bounds_and_says_where_the_origin_went(self) -> None:
        """The 'air grab' guard: a mesh whose file origin is not its centre must record the shift."""
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "offset.obj"
            box = _box()
            box.apply_translation([0.5, 0.0, 0.0])       # 500 mm off its own origin
            box.export(path)

            shape = load_mesh_shape(str(path))
            self.assertIsNotNone(shape)
            assert shape is not None
            np.testing.assert_allclose(shape.vertices_mm.mean(axis=0), 0.0, atol=1e-6)
            np.testing.assert_allclose(shape.extent_mm, EXTENT_MM, atol=1e-6)
            np.testing.assert_allclose(shape.origin_offset_mm, [-500.0, 0.0, 0.0], atol=1e-6)


class TheMeshSolidAnswersWhatTheClosedFormAnswersTests(unittest.TestCase):
    """A box through both paths. They share no code, so agreement is evidence rather than tautology."""

    def setUp(self) -> None:
        self.analytic, self.meshy = _pair()
        self.rng = np.random.default_rng(7)

    def test_containment_agrees_exactly(self) -> None:
        points = np.asarray([450.0, 10.0, 45.0]) + self.rng.uniform(-60, 60, size=(3000, 3))
        for margin in (0.0, -0.5):
            with self.subTest(margin=margin):
                np.testing.assert_array_equal(
                    self.analytic.contains_many(points, margin_mm=margin),
                    self.meshy.contains_many(points, margin_mm=margin),
                )

    def test_an_inflation_margin_differs_ONLY_at_the_corners(self) -> None:
        """And the mesh is the more correct of the two, which is worth stating rather than hiding.

        The primitive grows a box into a bigger box, with sharp corners. Inflating a solid is really a
        Minkowski sum, whose corners are ROUNDED, and that is what a signed distance reports. So the
        two differ exactly where a point is outside the un-inflated box on two or three axes at once --
        measured: 46 of 4000 probe points, every one of them analytic-inside / mesh-outside.
        """
        points = np.asarray([450.0, 10.0, 45.0]) + self.rng.uniform(-60, 60, size=(4000, 3))
        analytic = self.analytic.contains_many(points, margin_mm=12.0)
        meshy = self.meshy.contains_many(points, margin_mm=12.0)
        disagree = np.nonzero(analytic != meshy)[0]

        self.assertTrue(len(disagree), "the fixture must produce corner cases to be meaningful")
        self.assertTrue(bool(analytic[disagree].all()),
                        "only the box-shaped inflation may over-report; the rounded one may not")
        body = self.analytic.to_body(points[disagree])
        outside_axes = (np.abs(body) > np.asarray(EXTENT_MM) / 2.0).sum(axis=1)
        self.assertTrue(bool((outside_axes >= 2).all()),
                        "a disagreement on a FACE would mean the two solids differ, not their corners")

    def test_line_spans_agree(self) -> None:
        centre = np.asarray([450.0, 10.0, 45.0])
        for _ in range(400):
            origin = centre + self.rng.uniform(-80, 80, size=3)
            direction = self.rng.normal(size=3)
            direction /= np.linalg.norm(direction)
            first, second = (self.analytic.line_span(origin, direction),
                             self.meshy.line_span(origin, direction))
            self.assertEqual(first is None, second is None)
            if first is not None:
                assert second is not None
                np.testing.assert_allclose(first, second, atol=2e-3)

    def test_normals_support_widths_and_bounds_agree(self) -> None:
        half = np.asarray(EXTENT_MM) / 2.0
        for point in self.analytic.to_world(np.array([
                [half[0], 10.0, 5.0], [-half[0], -5.0, 20.0],
                [0.0, half[1], -8.0], [3.0, 0.0, half[2]]])):
            with self.subTest(point=tuple(np.round(point, 2))):
                np.testing.assert_allclose(
                    self.analytic.normal_at(point), self.meshy.normal_at(point), atol=1e-6)

        for direction in (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]),
                          np.array([0.3, -0.7, 0.65])):
            direction = direction / np.linalg.norm(direction)
            with self.subTest(direction=tuple(np.round(direction, 3))):
                self.assertAlmostEqual(self.analytic.support_width_mm(direction),
                                       self.meshy.support_width_mm(direction), places=6)

        self.assertAlmostEqual(self.analytic.top_z_mm(), self.meshy.top_z_mm(), places=6)
        self.assertAlmostEqual(self.analytic.bounding_radius_mm(),
                               self.meshy.bounding_radius_mm(), places=6)

    def test_top_z_is_not_half_the_support_width_for_a_ROTATED_mesh(self) -> None:
        """``centre + half the support width`` holds only for a shape symmetric about its centre.

        Rotated, and that is the point -- my first version of this test used an unrotated lopsided
        mesh and FAILED, because a shape centred on its axis-aligned bounds is symmetric along each
        body axis by construction, whatever its shape. The asymmetry only reaches world +z once the
        object is turned, which is the state every settled object in a dataset is in.
        """
        import trimesh

        lopsided = trimesh.util.concatenate([
            _box(),
            trimesh.creation.box(extents=(0.01, 0.01, 0.04)).apply_translation([0.0, 0.0, 0.06]),
        ])
        angle = np.pi / 5.0
        quat = (float(np.sin(angle / 2.0)), 0.0, 0.0, float(np.cos(angle / 2.0)))  # about +x
        solid = solid_from_mesh(_shape_from(lopsided), (450.0, 0.0, 200.0), quat)

        naive = solid.centre_mm[2] + 0.5 * solid.support_width_mm(np.array([0.0, 0.0, 1.0]))
        self.assertGreater(abs(solid.top_z_mm() - naive), 1.0)

        # And the value itself is right: the highest vertex, in world.
        world = solid.to_world(solid.mesh.vertices_mm)  # type: ignore[union-attr]
        self.assertAlmostEqual(solid.top_z_mm(), float(world[:, 2].max()), places=6)


class MeshLabelsAreSoundTests(unittest.TestCase):
    """Every mesh-path label must survive the closed-form verdict. Completeness is NOT claimed.

    THE DEFECT THIS CAUGHT. `_pad_contacts` probes nine pad positions and keeps whichever reports the
    extreme span; on a flat face all nine reach the same plane, and the winner decides where the
    CONTACT is -- which the finger samples, the table check and the collision check are all built from.
    Embree answers in float32, so nine identical surfaces came back differing by about a micron, and
    that micron moved the contact up to 38 mm across the pad. Measured on a 60 mm cube: identical spans
    (-30.0000, 30.0000), contact z 10.788 vs 50.447, and 7 of 26 labels passing a table check the
    closed form rejected at -25.6 mm. `_ray_mesh` now quantises to a micrometre, which restores the
    exact tie and makes the mesh path break it the same way the primitives do.
    """

    def _sound(self, extent_mm) -> None:
        analytic, meshy = _pair(extent_mm, position_mm=(450.0, 0.0, extent_mm[2] / 2.0))
        labels, _ = label_jaw_grasps(SceneGeometry("t", "sparse", {0: meshy}, ()), 0)
        self.assertTrue(labels, "the fixture must produce labels for this to mean anything")

        model = JawModel()
        for label in labels:
            verdict = check_jaw_grasp(
                JawGrasp(np.asarray(label.position_mm), np.asarray(label.approach),
                         np.asarray(label.closing_axis), label.width_mm),
                analytic, (), model=model)
            with self.subTest(position=tuple(np.round(label.position_mm, 1))):
                self.assertTrue(verdict.ok,
                                f"mesh label rejected by the closed form: {verdict.reason} "
                                f"{verdict.detail}")

    def test_a_flat_box(self) -> None:
        self._sound((70.0, 50.0, 90.0))

    def test_a_tall_box(self) -> None:
        self._sound((40.0, 40.0, 120.0))

    def test_a_cube(self) -> None:
        self._sound((60.0, 60.0, 60.0))


class OneConversionForTheRendererAndThePhysicsTests(unittest.TestCase):
    """Two places author this mesh into USD -- the render pass and the shake cell. One conversion.

    They used to hold a copy each. A mesh drawn one way and collided another is precisely the drift
    this package exists to prevent, and it is the least visible kind: a physics outcome carries no
    shape in it that anyone could check afterwards.
    """

    def test_the_arrays_describe_the_same_triangles_in_metres(self) -> None:
        shape = _shape_from(_box())
        points, counts, indices, lower, upper = shape.usd_arrays()

        np.testing.assert_allclose(points * 1000.0, shape.vertices_mm, atol=1e-9)
        self.assertEqual(counts, [3] * len(shape.faces))
        self.assertEqual(len(indices), 3 * len(shape.faces))
        self.assertEqual(max(indices), len(shape.vertices_mm) - 1)
        np.testing.assert_allclose((upper - lower) * 1000.0, shape.extent_mm, atol=1e-6)


class TheMeshEnumerationIsDeterministicTests(unittest.TestCase):
    def test_two_runs_produce_the_same_labels(self) -> None:
        """No RNG anywhere on this path -- a reference that differs between runs is not a reference."""
        _, meshy = _pair()
        first, _ = label_jaw_grasps(SceneGeometry("t", "sparse", {0: meshy}, ()), 0)
        second, _ = label_jaw_grasps(SceneGeometry("t", "sparse", {0: meshy}, ()), 0)
        self.assertEqual([label.as_row() for label in first], [label.as_row() for label in second])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
