"""A body declared as boxes becomes collision geometry, once, for everything that declares one (B7).

Three things in this stack are described by measurements rather than by a mesh, and each of them has to reach the
planner and the exact mesh guard as geometry:

* a GRIPPER a customer describes in config and has never baked (B7: jaws and palm from `gripper.jaw`),
* the PLATE and the tool changer between the flange and the hand (UM8),
* the wrist CAMERA's housing and its bracket (Step 6b).

The owner settled on one writer for all three, so a box becomes geometry in one place rather than in three that can
disagree. What the three share is exactly this: a named box, its centre, its half extents, and the frame it sits in.

⚠ **A BOX IS NOT THE HAND.** The EGU-50's own config records what a box leaves out: along the closing axis its
housing reaches 2.25 mm past the finger's outer face, and an envelope built from the numbers does not carry that
corner. So a body written from dimensions carries a measured INFLATION, and says in its provenance that it came from
dimensions rather than from a scan. The inflation is measured against the three hands that have both a bundle and a
full set of dimensions, which is the only reason a number can be put on it at all.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]


def _module():
    path = _ROOT / "src" / "robot" / "safety" / "planning" / "_declared_body.py"
    spec = importlib.util.spec_from_file_location("_declared_body_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class OneBoxBecomesOneMesh(unittest.TestCase):
    """The arithmetic, against a box whose answers can be written down before the code runs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def test_a_box_has_eight_corners_and_twelve_triangles(self) -> None:
        box = self.module.Box(name="palm", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 20.0, 30.0))

        vertices, faces = self.module.box_mesh(box)

        self.assertEqual(vertices.shape, (8, 3))
        self.assertEqual(faces.shape, (12, 3))
        self.assertEqual(sorted(np.unique(faces).tolist()), list(range(8)))

    def test_the_corners_are_where_the_measurements_put_them(self) -> None:
        box = self.module.Box(name="finger", centre_mm=(5.0, 0.0, -2.0), half_extents_mm=(1.0, 2.0, 3.0))

        vertices, _ = self.module.box_mesh(box)

        self.assertEqual(np.round(vertices.min(axis=0), 6).tolist(), [4.0, -2.0, -5.0])
        self.assertEqual(np.round(vertices.max(axis=0), 6).tolist(), [6.0, 2.0, 1.0])

    def test_the_surface_is_closed_and_wound_outward(self) -> None:
        """⭐ THE ONE THAT MATTERS FOR A COLLISION MESH. An open or inside-out box is not a solid, and the cover fit
        and the exact guard both read it as one: the fit's inside test is a winding number, which an inverted surface
        answers with the wrong sign, and Coal builds a convex body from the faces it is given."""
        box = self.module.Box(name="cube", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(4.0, 4.0, 4.0))

        vertices, faces = self.module.box_mesh(box)

        # Every edge of a closed surface is shared by exactly two triangles, once in each direction.
        edges: dict = {}
        for a, b, c in faces:
            for start, end in ((a, b), (b, c), (c, a)):
                edges[(int(start), int(end))] = edges.get((int(start), int(end)), 0) + 1
        self.assertTrue(all(count == 1 for count in edges.values()), "an edge is used twice the same way round")
        for (start, end) in edges:
            self.assertIn((end, start), edges, f"edge {start}->{end} has no partner, so the surface is open")

        # The signed volume of an outward wound closed surface is positive and is the box's own volume.
        volume = 0.0
        for a, b, c in faces:
            volume += float(np.dot(vertices[a], np.cross(vertices[b], vertices[c]))) / 6.0
        self.assertAlmostEqual(volume, 8.0 * 8.0 * 8.0, places=6)

    def test_a_box_with_no_thickness_is_refused(self) -> None:
        """A zero half extent is a plate nobody measured, not a thin plate: it has no inside for a fit to march into."""
        with self.assertRaises(ValueError):
            self.module.box_mesh(self.module.Box(name="flat", centre_mm=(0.0, 0.0, 0.0),
                                                 half_extents_mm=(10.0, 0.0, 10.0)))
        with self.assertRaises(ValueError):
            self.module.box_mesh(self.module.Box(name="negative", centre_mm=(0.0, 0.0, 0.0),
                                                 half_extents_mm=(10.0, -1.0, 10.0)))


class ABodyIsSeveralBoxes(unittest.TestCase):
    """A hand is a palm and two fingers, a camera is a housing and a bracket, a tool stack is plate after plate."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def test_the_boxes_become_one_mesh_per_named_part(self) -> None:
        boxes = [
            self.module.Box(name="gripper", centre_mm=(0.0, 40.0, 0.0), half_extents_mm=(30.0, 40.0, 25.0)),
            self.module.Box(name="lfinger", centre_mm=(-20.0, 100.0, 0.0), half_extents_mm=(5.0, 20.0, 12.0)),
            self.module.Box(name="rfinger", centre_mm=(20.0, 100.0, 0.0), half_extents_mm=(5.0, 20.0, 12.0)),
        ]

        parts = self.module.boxes_to_parts(boxes, frame=6)

        self.assertEqual(sorted(parts), ["gripper__f", "gripper__frame", "gripper__v",
                                         "lfinger__f", "lfinger__frame", "lfinger__v",
                                         "rfinger__f", "rfinger__frame", "rfinger__v"])
        self.assertEqual(parts["gripper__v"].shape, (8, 3))
        self.assertEqual(int(parts["lfinger__frame"][0]), 6)

    def test_two_boxes_of_one_name_are_one_part(self) -> None:
        """A camera is a housing AND a bracket, and both are the camera. One name, one mesh, two boxes in it."""
        boxes = [
            self.module.Box(name="camera", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(45.0, 12.5, 12.5)),
            self.module.Box(name="camera", centre_mm=(0.0, -30.0, 0.0), half_extents_mm=(20.0, 20.0, 5.0)),
        ]

        parts = self.module.boxes_to_parts(boxes, frame=6)

        self.assertEqual(sorted(parts), ["camera__f", "camera__frame", "camera__v"])
        self.assertEqual(parts["camera__v"].shape, (16, 3))
        self.assertEqual(parts["camera__f"].shape, (24, 3))
        # The second box's faces index its own vertices, not the first box's.
        self.assertEqual(int(parts["camera__f"].max()), 15)

    def test_a_body_with_no_boxes_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            self.module.boxes_to_parts([], frame=6)


class TheInflationIsMeasuredRatherThanChosen(unittest.TestCase):
    """⭐ WHAT A BOX HAND MAY CLAIM. The owner's answer: equal standing, with a measured inflation and a provenance
    that says it came from dimensions.

    The inflation cannot be argued, only measured, and it can be measured here because all three shipped hands carry
    BOTH a baked bundle and a full set of declared dimensions. That is a coincidence of this moment and the reason to
    take the measurement now rather than later.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.module = _module()

    def test_an_inflation_grows_every_half_extent_by_the_same_amount(self) -> None:
        box = self.module.Box(name="palm", centre_mm=(1.0, 2.0, 3.0), half_extents_mm=(10.0, 20.0, 30.0))

        grown = self.module.inflated(box, by_mm=2.5)

        self.assertEqual(grown.centre_mm, box.centre_mm)
        self.assertEqual(grown.half_extents_mm, (12.5, 22.5, 32.5))
        self.assertEqual(grown.name, box.name)

    def test_a_negative_inflation_is_refused(self) -> None:
        """Shrinking a declared body is the unsafe direction and is never what a caller meant."""
        box = self.module.Box(name="palm", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))

        with self.assertRaises(ValueError):
            self.module.inflated(box, by_mm=-0.1)

    def test_zero_is_allowed_because_it_is_a_declaration(self) -> None:
        """A cell that measured its own body and found nothing to add says 0.0, which is not the same as saying
        nothing. The caller that writes a bundle refuses an inflation nobody stated."""
        box = self.module.Box(name="palm", centre_mm=(0.0, 0.0, 0.0), half_extents_mm=(10.0, 10.0, 10.0))

        self.assertEqual(self.module.inflated(box, by_mm=0.0).half_extents_mm, (10.0, 10.0, 10.0))


if __name__ == "__main__":
    unittest.main()
