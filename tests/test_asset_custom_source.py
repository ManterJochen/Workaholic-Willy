"""The `custom` source: a customer's own meshes, and the licence they must declare for them.

⭑ WHY THIS FILE EXISTS. `SUPPORTED_SOURCES` held exactly `{gso, ycb}` -- the two research datasets --
which meant a customer could not bring a single mesh of their own into the scene generator. Their only
route was to call their parts "gso" and rename them to `.obj`, which would have put a CC-BY attribution
on a company's internal CAD. Scene generation is the USER's job now, so this path carries half the
product, and the tests below hold its two load-bearing properties: the meshes get in, and the licence
that travels with them is the operator's declaration rather than our guess.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import trimesh

from datagen.assets.library import (
    CUSTOM_SUFFIXES,
    LICENSES,
    OPTIONAL_SOURCES,
    SUPPORTED_SOURCES,
    MeshLibrary,
    declared_license,
    import_from_directory,
)
from datagen.assets.licensing import audit_asset_rows
from datagen.assets.meshes import measure_mesh


def _origin(suffixes: tuple[str, ...] = ("stl", "obj")) -> Path:
    directory = Path(tempfile.mkdtemp()) / "custom"
    directory.mkdir()
    for index, suffix in enumerate(suffixes):
        box = trimesh.creation.box(extents=(0.05 + index * 0.01, 0.04, 0.06))
        box.export(directory / f"part_{index}.{suffix}")
    return directory


class CustomSourceTests(unittest.TestCase):
    def test_custom_is_a_supported_but_optional_source(self) -> None:
        """Present enough to import, optional enough that its absence is not a broken machine."""
        self.assertIn("custom", SUPPORTED_SOURCES)
        self.assertIn("custom", OPTIONAL_SOURCES)
        # ⛔ NO LICENCE MAP ENTRY, and that is the design: see `declared_license`.
        self.assertNotIn("custom", LICENSES)

    def test_import_refuses_without_a_declared_licence(self) -> None:
        library = Path(tempfile.mkdtemp())
        with self.assertRaises(ValueError) as caught:
            import_from_directory("custom", _origin(), destination=library)
        self.assertIn("licence", str(caught.exception))
        # And it refused BEFORE writing anything -- a half-imported library is worse than none.
        self.assertEqual(MeshLibrary(library).counts().get("custom", 0), 0)

    def test_import_accepts_mixed_suffixes_a_cad_tool_actually_writes(self) -> None:
        """`.stl` beside `.obj`. The two research datasets happen to ship one suffix each; a
        customer's exports are whatever their tool writes, and that is not a compatibility question."""
        library = Path(tempfile.mkdtemp())
        entries = import_from_directory(
            "custom", _origin(("stl", "obj")), destination=library, license="own")
        self.assertEqual(len(entries), 2)
        self.assertEqual({e.path.suffix.lstrip(".") for e in entries}, {"stl", "obj"})

    def test_the_declaration_survives_the_command_that_made_it(self) -> None:
        """⭑ THE POINT OF THE WHOLE FILE. A licence passed as a command-line argument is gone by the
        next session; the manifest, the audit and the attribution text all need it months later."""
        library = Path(tempfile.mkdtemp())
        import_from_directory("custom", _origin(), destination=library,
                              license="CC0-1.0", attribution="ACME GmbH")
        self.assertEqual(declared_license(library / "custom"), ("CC0-1.0", "ACME GmbH"))

        asset = measure_mesh("custom_part_0", "custom", MeshLibrary(library).available("custom")[0])
        assert asset is not None
        self.assertEqual(asset.license, "CC0-1.0")
        self.assertIn("ACME GmbH", asset.attribution)

    def test_the_licence_file_is_not_mistaken_for_a_mesh(self) -> None:
        """`custom` matches many suffixes, and its directory also holds the LICENSE.txt we write."""
        library = Path(tempfile.mkdtemp())
        import_from_directory("custom", _origin(), destination=library, license="own")
        found = MeshLibrary(library).available("custom")
        self.assertTrue(found)
        self.assertNotIn("LICENSE.txt", [path.name for path in found])
        for path in found:
            self.assertIn(path.suffix.lstrip("."), CUSTOM_SUFFIXES)

    def test_an_undeclared_directory_is_refused_by_the_licence_audit(self) -> None:
        """Meshes dropped into the library BY HAND carry no declaration, and the audit must say so
        rather than pass them. This is the gap `declared_license` returning "" keeps open on purpose."""
        library = Path(tempfile.mkdtemp())
        (library / "custom").mkdir(parents=True)
        trimesh.creation.box(extents=(0.05, 0.05, 0.05)).export(library / "custom" / "hand.stl")

        asset = measure_mesh("custom_hand", "custom", library / "custom" / "hand.stl")
        assert asset is not None
        self.assertEqual(asset.license, "")
        problems = audit_asset_rows([{"id": asset.asset_id, "source": "custom",
                                      "license": asset.license, "attribution": ""}])
        self.assertTrue(any("missing license" in p for p in problems), problems)

    def test_a_declared_licence_passes_the_audit(self) -> None:
        for licence, attribution in (("own", ""), ("CC0-1.0", ""), ("CC-BY-4.0", "ACME GmbH")):
            with self.subTest(licence=licence):
                self.assertEqual(
                    audit_asset_rows([{"id": "custom_part_0", "source": "custom",
                                       "license": licence, "attribution": attribution}]), [])

    def test_a_noncommercial_declaration_is_still_refused(self) -> None:
        """The operator declares; the gate still judges. Declaring is not a way past it."""
        problems = audit_asset_rows([{"id": "custom_part_0", "source": "custom",
                                      "license": "CC-BY-NC-4.0", "attribution": "x"}])
        self.assertTrue(problems)


if __name__ == "__main__":                                           # pragma: no cover
    unittest.main()
