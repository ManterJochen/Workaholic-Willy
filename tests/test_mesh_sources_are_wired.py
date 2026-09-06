"""A source the fetcher can download must be a source the pipeline can see.

⛔ THE DEFECT THIS EXISTS FOR, and it shipped for several hours on 2026-09-03. Objaverse was added to
the fetcher with 724,500 licence-clean objects, and 297 of them were downloaded, normalised and
reported as ready. It was never added to `SUPPORTED_SOURCES`.

`screen_meshes` and `prepare_assets` iterate that dict when no collection is named, so the meshes sat
on disk while every command reported success and screened nothing. There is no error to notice: the
count is zero because the loop never reached the directory, and zero is a perfectly ordinary number.

That is the inert-switch shape this repository fences everywhere else: a flag that is set, a value
that is read, and a path that never runs. The wiring guard on the config schema catches it for boolean
flags. This is the same guard for mesh collections.
"""

from __future__ import annotations

import unittest
from pathlib import Path

# ⭐ A PLAIN PACKAGE IMPORT SINCE 2026-09-04. The module moved from `scripts/meshes/fetch.py`
# into `datagen/assets/fetch.py`, so the path insertion this file carried is gone. The old form
# also risked importing some other `fetch` that happened to be on the path first.

from datagen.assets.fetch import SOURCES

from datagen.assets.library import SUPPORTED_SOURCES  # noqa: E402
from datagen.assets.prepare import MESH_SUFFIXES  # noqa: E402


class WiringTests(unittest.TestCase):

    def test_every_fetchable_source_is_one_the_pipeline_iterates(self) -> None:
        """⭐ THE WHOLE GUARD. Anything downloadable must be reachable by a run that names no
        collection, because that is the run a new user makes."""
        unwired = sorted(set(SOURCES) - set(SUPPORTED_SOURCES))
        self.assertEqual(unwired, [], f"downloadable but invisible to the pipeline: {unwired}")

    def test_every_pipeline_source_can_actually_be_obtained(self) -> None:
        """The other direction. A collection the pipeline offers and nothing can fetch is a promise
        with no way to keep it. `custom` is the exception: it IS the user's own directory."""
        unobtainable = sorted(set(SUPPORTED_SOURCES) - set(SOURCES) - {"custom"})
        self.assertEqual(unobtainable, [],
                         f"the pipeline offers these and nothing fetches them: {unobtainable}")

    def test_each_source_declares_an_extension_the_enumerator_accepts(self) -> None:
        """A source mapped to an extension the mesh filter rejects would find nothing, which looks
        exactly like an empty directory."""
        for source, suffix in SUPPORTED_SOURCES.items():
            with self.subTest(source):
                if suffix == "*":
                    continue
                self.assertIn(f".{suffix}", MESH_SUFFIXES)


class EnumerationTests(unittest.TestCase):

    def test_a_non_mesh_file_beside_the_meshes_is_not_screened(self) -> None:
        """⚠ NOT HYPOTHETICAL, AND THE FILE IS REQUIRED RATHER THAN INCIDENTAL. CC-BY obliges naming
        the author, so the Objaverse fetch writes an ATTRIBUTION.tsv beside the meshes it credits.
        Under a `*` glob that became a screening task: it failed harmlessly, and it also inflated the
        reported mesh count by one. A count that is wrong by one is still a number somebody quotes.
        """
        import tempfile

        from datagen.assets.prepare import screen_meshes

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "custom").mkdir()
            (root / "custom" / "ATTRIBUTION.tsv").write_text("uid\tauthor\n", encoding="utf-8")
            (root / "custom" / "notes.md").write_text("hello", encoding="utf-8")
            seen: list[str] = []
            screen_meshes(sources=["custom"], library=root, out=root / "out",
                          report=seen.append)
        self.assertTrue(any("0 mesh(es)" in line or "nothing to screen" in line for line in seen),
                        f"non-mesh files were counted as meshes: {seen}")

    def test_the_two_enumeration_sites_share_one_definition(self) -> None:
        """They disagreed until 2026-09-03: one filtered by a literal tuple and the other did not
        filter at all.

        ⚠ THIS GUARD WAS ITSELF A SECOND SPELLING. It pinned the literal
        `(".ply", ".obj", ".stl", ".off")` and so it failed the moment the customer-mesh rail found
        the drift it was written to prevent: the LIBRARY admitted seven suffixes while the preparer
        handled four, and `.glb` files were imported, counted ready, and then skipped in silence. The
        repair derives one list from the other, so the assertion is now about the DERIVATION.
        """
        from datagen.assets.library import CUSTOM_SUFFIXES
        from datagen.assets.prepare import MESH_SUFFIXES

        text = Path("datagen/assets/prepare.py").read_text(encoding="utf-8")
        self.assertGreaterEqual(text.count("MESH_SUFFIXES"), 3)
        self.assertEqual(set(MESH_SUFFIXES), {f".{s}" for s in CUSTOM_SUFFIXES},
                         "the preparer and the library disagree about what a mesh is")
        self.assertNotIn('".ply", ".obj", ".stl"', text,
                         "the mesh extensions are spelled out a second time")


if __name__ == "__main__":
    unittest.main()
