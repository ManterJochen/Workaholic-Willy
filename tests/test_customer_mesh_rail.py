"""WS5: a customer's CAD export, from the inbox to a screened asset. Walked, not assumed.

⛔⛔ **EVERY STAGE WAS BUILT AND WIRED AND NO FILE HAD EVER RUN.** `sha256.csv` carried zero `custom`
rows. Walking it once on 2026-09-04 found THREE defects, in ascending order of how badly they would
have hurt a customer:

1. `.glb` and `.gltf` were imported, counted ready by `available()` and `is_ready()`, placed by the
   layout, and then SKIPPED IN SILENCE by both normalise and screen: the library admitted seven
   suffixes and `prepare` handled four.
2. A source at scale 1.0 with an implausible size got no warning at all. Three parts exported in
   millimetres landed at 70,000 / 120,000 / 180,000 mm and screened `status: ok`. The warning that
   was then added was DISCARDED by the "already prepared" early return, which is worse than never
   computing it.
3. ⛔ **RESCALING AN `.stl` DESTROYED IT.** Open3D refuses to write an STL without vertex normals: it
   logs, returns `False`, and leaves a zero-byte file. Nothing read that return value, so the mesh
   came back with 0 faces and the next stage reported `unclosable` -- a geometry verdict on a file
   this pipeline had emptied itself. `.stl` is the default CAD export, so this sat on the single most
   likely customer path.

⚠ THE PARTS ARE SYNTHESISED, NOT A CUSTOMER'S. They are shaped like real ones and exported in the two
formats a customer sends, but this tests the RAIL and not anyone's geometry, and the difference is
stated rather than blurred.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

_SLOW = "walks import, normalise and screen; seconds rather than milliseconds"


def _parts():
    import trimesh

    # ⚠ IN MILLIMETRES, which is what a CAD package emits, while the pipeline reads METRES. That
    # mismatch is the point of `test_millimetres_are_NAMED_not_silently_accepted`.
    return {
        "bracket": (trimesh.creation.box(extents=(70.0, 40.0, 25.0)), "stl"),
        "plate": (trimesh.creation.box(extents=(120.0, 90.0, 3.0)), "glb"),
    }


def _inbox(directory: Path) -> Path:
    inbox = directory / "inbox"
    inbox.mkdir(parents=True)
    for name, (mesh, suffix) in _parts().items():
        mesh.export(inbox / f"{name}.{suffix}")
    return inbox


class TheRailTests(unittest.TestCase):

    def test_a_CAD_export_reaches_a_screened_asset(self) -> None:
        """⭐ THE CLAUSE ITSELF: one example CAD measured end to end. MEASURED on the walk that fixed
        the three defects above: bracket 70.0 mm with 1,633 jaw labels, plate 120.0 mm with 670."""
        from datagen.assets.library import MeshLibrary, import_from_directory
        from datagen.assets.prepare import normalise_meshes, screen_meshes

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            inbox = _inbox(root)
            library = root / "library"
            import_from_directory("custom", inbox, destination=library, license="own")
            self.assertEqual(len(MeshLibrary(library).available("custom")), len(_parts()))

            # `--scale 0.001` is what the implausible-size warning tells a customer to pass.
            normalise_meshes("custom", library=library, faces=20000, jobs=1, scale=0.001,
                             report=lambda _line: None)
            rows = screen_meshes(["custom"], library=library, out=root / "screen.json",
                                 density="grid", jobs=1, report=lambda _line: None)

        self.assertEqual(len(rows), len(_parts()),
                         "a format was imported and then skipped by the screen")
        for row in rows:
            with self.subTest(row.get("asset_id")):
                self.assertEqual(row.get("status"), "ok")
        self.assertTrue(any(int(r.get("jaw", 0)) > 0 for r in rows),
                        "no part earned a jaw label; the rail produced an unusable library")

    def test_a_GLB_is_not_silently_skipped(self) -> None:
        """⛔ It was: imported, counted ready, and handled by neither normalise nor screen."""
        from datagen.assets.library import CUSTOM_SUFFIXES
        from datagen.assets.prepare import MESH_SUFFIXES

        self.assertIn("glb", CUSTOM_SUFFIXES)
        self.assertEqual(set(MESH_SUFFIXES), {f".{s}" for s in CUSTOM_SUFFIXES},
                         "the library and the preparer disagree about what a mesh is, which is how "
                         "a customer's file gets imported and then skipped")

    def test_DAE_is_refused_rather_than_accepted_and_skipped(self) -> None:
        """⚠ MEASURED: trimesh round-trips obj, stl, ply, off, glb and gltf on this machine, and
        `.dae` raises `ImportError: missing 'pip install pycollada'`. Accepting a format nothing can
        open is the same silent skip wearing a different name."""
        from datagen.assets.library import CUSTOM_SUFFIXES

        self.assertNotIn("dae", CUSTOM_SUFFIXES)

    def test_millimetres_are_NAMED_not_silently_accepted(self) -> None:
        """⛔ A source at scale 1.0 fell through both plausibility branches, so a CAD export in
        millimetres landed 1000x too large with `status: ok` and nothing said a word."""
        from datagen.assets.prepare import _implausible

        note = _implausible(70_000.0)
        self.assertIn("IMPLAUSIBLE", note)
        self.assertIn("millimetres", note)
        self.assertIn("--scale 0.001", note, "the warning does not say how to fix it")

    def test_a_PLAUSIBLE_size_is_not_warned_about(self) -> None:
        """The control. A warning that fires on everything is noise, and noise gets filtered."""
        from datagen.assets.prepare import PLAUSIBLE_EXTENT_MM

        self.assertLess(PLAUSIBLE_EXTENT_MM[0], 70.0)
        self.assertGreater(PLAUSIBLE_EXTENT_MM[1], 70.0)

    def test_the_warning_SURVIVES_the_early_return(self) -> None:
        """⛔ It did not. A mesh needing no decimation returned "already prepared" and threw the
        implausible-size note away. A warning computed and discarded is worse than one never written,
        because the code reads as if the check exists."""
        source = Path("datagen/assets/prepare.py").read_text(encoding="utf-8")
        self.assertIn('already prepared{note}', source)

    def test_writing_a_rescaled_mesh_CHECKS_that_it_worked(self) -> None:
        """⛔⛔ THE WORST OF THE THREE. Open3D returns False and leaves a zero-byte STL when normals
        are missing, and nothing read the return value."""
        source = Path("datagen/assets/prepare.py").read_text(encoding="utf-8")
        self.assertIn("mesh.compute_vertex_normals()", source)
        self.assertIn("if not o3d.io.write_triangle_mesh(", source)

    def test_open3d_really_does_refuse_an_STL_without_normals(self) -> None:
        """The claim above, verified against the library rather than taken from a log line."""
        import open3d as o3d

        mesh = o3d.geometry.TriangleMesh.create_box(0.07, 0.04, 0.025)
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "x.stl"
            self.assertFalse(o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False))
            mesh.compute_vertex_normals()
            self.assertTrue(o3d.io.write_triangle_mesh(str(path), mesh, write_ascii=False))
            self.assertEqual(len(o3d.io.read_triangle_mesh(str(path)).triangles), 12)


if __name__ == "__main__":
    unittest.main()
