"""A mesh that arrives watertight must not leave `close_mesh` refused.

⛔ MEASURED 2026-09-01 on our own YCB bank: **all 65 meshes are watertight as loaded** and `close_mesh`
accepted **49**. The 16 it refused are the canonical parallel-jaw objects -- mug, bowl, pear, mustard
bottle, tuna fish can, gelatin box, sponge, spoon, power drill, adjustable wrench -- from the
collection that measures BEST for us, at 91 % jaw-graspable against GSO's 32 %. They were dropped
before the mesh bank ever saw them, so no filter, no report and no rejection histogram recorded their
absence.

The mechanism: `merge_vertices()` welds coincident vertices, and on a scan whose two halves meet at a
seam that welding produces NON-MANIFOLD edges out of a surface that was already closed.
`_boundary_loops` then returns None and the mesh is refused by the very function whose job is to close
it.

⚠ THE FIX IS A FALLBACK, NOT AN EARLY RETURN, and the difference matters. Returning the raw mesh up
front would also change every mesh the current path ACCEPTS, and those sit inside shipped corpora whose
manifest hash the evaluation now refuses to see move. Falling back only where the current path FAILS is
additive.

⚠ AND IT WENT ON THE WRONG EXIT TWICE. The refusal happens at `_boundary_loops`, not at the watertight
check further down, so the first two versions of the guard were inert and the count did not move. The
number is what said so, not the intent. That is the fourth defect of this shape this week.
"""

from __future__ import annotations

import unittest

import numpy as np


def _open_box() -> object:
    """A box with one face removed: genuinely open, and `close_mesh` must still repair it."""
    import trimesh

    box = trimesh.creation.box(extents=(0.05, 0.04, 0.03))
    faces = np.asarray(box.faces)
    return trimesh.Trimesh(vertices=np.asarray(box.vertices), faces=faces[:-2], process=False)


class TheWatertightFallbackTests(unittest.TestCase):
    def test_an_already_watertight_mesh_is_never_refused(self) -> None:
        """⭑ THE ONE. A closed surface in, a closed surface out."""
        import trimesh

        from datagen.assets.mesh_geometry import close_mesh

        sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
        self.assertTrue(sphere.is_watertight, "the fixture must start watertight")
        self.assertIsNotNone(close_mesh(sphere))

    def test_a_seam_that_welds_into_a_non_manifold_edge_still_survives(self) -> None:
        """The exact YCB shape: watertight on arrival, broken by the weld, rescued by the fallback.

        Built by duplicating every vertex so `merge_vertices` has something to weld, which is what a
        scan's seam looks like to trimesh.
        """
        import trimesh

        from datagen.assets.mesh_geometry import close_mesh

        base = trimesh.creation.icosphere(subdivisions=3, radius=0.03)
        vertices = np.vstack([base.vertices, base.vertices])
        faces = np.vstack([base.faces, base.faces + len(base.vertices)])
        doubled = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self.assertIsNotNone(close_mesh(doubled),
                             "a mesh whose weld makes it non-manifold must fall back to the original")

    def test_a_GENUINELY_open_mesh_is_still_closed_or_refused_on_its_merits(self) -> None:
        """The fallback must not become a bypass: an open surface still goes through the repair."""
        from datagen.assets.mesh_geometry import close_mesh

        result = close_mesh(_open_box())
        if result is not None:
            self.assertTrue(result.is_watertight, "a repaired mesh must actually be closed")

    def test_a_degenerate_sheet_is_still_refused(self) -> None:
        """A surface enclosing no volume has no inside for the labeller to ask about."""
        import trimesh

        from datagen.assets.mesh_geometry import close_mesh

        flat = trimesh.Trimesh(
            vertices=np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.04, 0.0]]),
            faces=np.array([[0, 1, 2]]), process=False)
        self.assertIsNone(close_mesh(flat))

    def test_the_guard_sits_on_the_exit_the_meshes_actually_take(self) -> None:
        """⛔ THE INERT-GUARD CHECK. Two earlier versions put the fallback on the watertight test,
        which these meshes never reach, so the count did not move. The refusal is at the boundary-loop
        exit."""
        import inspect

        from datagen.assets import mesh_geometry

        source = inspect.getsource(mesh_geometry.close_mesh)
        loops = source.index("loops = _boundary_loops(closed)")
        guard = source.index("raw_is_watertight")
        self.assertLess(guard, loops + source[loops:].index("if loops is None:") + 200,
                        "the fallback does not cover the boundary-loop exit")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
