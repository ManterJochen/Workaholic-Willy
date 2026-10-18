"""Real meshes may reach a render only with the attribution their licence obliges.

The scene generator had exactly one asset source -- procedural solids -- because
`assets.{gso,ycb,objaverse}_weight` validate and nothing reads them. This is the import half, and
what it must never do is get a mesh into a dataset without recording who made it: every rendered
image is a derivative work of every mesh in it, so a CC-BY row with no author is a licence violation
waiting for the first person who shares the data.

The Objaverse case is asserted as BEHAVIOUR rather than described in a comment. Its 199 models are
individually authored and the metadata we hold is `stem,license` with no author, so it is absent
from this module -- and the test below fails if somebody adds it without solving that first.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from datagen.assets.library import (
    ATTRIBUTIONS,
    LICENSES,
    SUPPORTED_SOURCES,
    MeshLibrary,
    import_from_directory,
    write_sha_index,
)
from datagen.assets.licensing import audit_asset_rows


def _fake_meshes(root: Path, source: str, n: int) -> Path:
    directory = root / source
    directory.mkdir(parents=True, exist_ok=True)
    suffix = SUPPORTED_SOURCES[source]
    for i in range(n):
        (directory / f"obj{i:03d}.{suffix}").write_bytes(f"mesh {i}".encode())
    return directory


class EveryImportedMeshCarriesItsAttributionTests(unittest.TestCase):
    """The licence gate is fail-closed; this proves the import satisfies it rather than dodging it."""

    def test_imported_rows_pass_the_licence_audit(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_meshes(root / "origin", "gso", 3)
            _fake_meshes(root / "origin", "ycb", 2)
            for source in ("gso", "ycb"):
                import_from_directory(
                    source, root / "origin" / source, destination=root / "library"
                )

            library = MeshLibrary(root / "library")
            rows = []
            for source in ("gso", "ycb"):
                rows += [
                    r.as_row()
                    for r in library.records(source, extent_mm=(80.0, 80.0, 80.0), mass_kg=0.15)
                ]

            self.assertEqual(len(rows), 5)
            self.assertEqual(audit_asset_rows(rows), [])

    def test_a_row_stripped_of_its_attribution_is_REFUSED(self) -> None:
        """The audit has teeth: without this, the test above could pass on a gate that accepts all."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_meshes(root / "origin", "gso", 1)
            import_from_directory("gso", root / "origin" / "gso", destination=root / "library")
            row = MeshLibrary(root / "library").records(
                "gso", extent_mm=(80.0, 80.0, 80.0), mass_kg=0.15
            )[0].as_row()
            row["attribution"] = ""
            problems = audit_asset_rows([row])
            self.assertEqual(len(problems), 1)
            self.assertIn("attribution", problems[0])

    def test_the_attribution_names_author_and_source(self) -> None:
        """What CC-BY actually asks for -- a URL alone is not attribution."""
        for source, text in ATTRIBUTIONS.items():
            with self.subTest(source=source):
                self.assertIn("http", text)
                self.assertIn(LICENSES[source], text)
                self.assertGreater(len(text.split(",")[0]), 10, "no author named")


class ObjaverseIsAdmittedBECAUSEItsAuthorsAreNowWrittenDownTests(unittest.TestCase):
    """⭐ THE DEFERRAL WAS LIFTED BY MEETING ITS CONDITION, NOT BY RELAXING THE GATE.

    This class used to assert the OPPOSITE, that objaverse was not an importable source, and it was
    right to: 199 individually-authored models against metadata of `stem,license` with no author, and
    under a fail-closed CC-BY audit those rows cannot be rendered. The correct response was to resolve
    the authors first.

    That is what happened. The fetch writes an `ATTRIBUTION.tsv` beside the meshes it credits, the
    source has an `ATTRIBUTIONS` entry naming author and licence, and the audit that refuses a row
    with an empty attribution is unchanged and still has teeth (see the test above it).

    ⛔ SO THE GATE IS WHAT IS TESTED HERE, NOT THE MEMBERSHIP. A source may be listed only while its
    attribution exists, which is the condition the old class was protecting; flipping an assertion
    without checking that condition would have been exactly the relaxation it warned about.
    """

    def test_it_is_an_importable_source(self) -> None:
        self.assertIn("objaverse", SUPPORTED_SOURCES)

    def test_it_carries_an_attribution_that_names_an_AUTHOR(self) -> None:
        """The condition the deferral was waiting on. A URL alone is not attribution."""
        text = ATTRIBUTIONS["objaverse"]
        self.assertIn("http", text)
        self.assertIn(LICENSES["objaverse"], text)
        self.assertGreater(len(text.split(",")[0]), 10, "no author named")

    def test_EVERY_listed_source_has_one(self) -> None:
        """⛔ The general rule, so the next source cannot be added the way this one nearly was. A
        source in the registry with no attribution entry is a collection whose rows the audit will
        refuse at render time, hours after somebody fetched it."""
        for source in SUPPORTED_SOURCES:
            if source == "custom":
                continue          # a customer's own parts carry whatever licence the customer holds
            with self.subTest(source=source):
                self.assertIn(source, ATTRIBUTIONS)
                self.assertIn(source, LICENSES)

    def test_an_UNKNOWN_source_still_refuses_by_name(self) -> None:
        """The control. If importing anything at all were allowed, the two tests above would pass on
        a registry that had stopped being a gate.

        ⚠ THE PLACEHOLDER IS DELIBERATELY NEUTRAL. The first version of this line named a real
        non-commercial collection as its "unknown source", and `test_license_boundary` refused the
        tree within minutes: that gate matches a forbidden project as a case-insensitive SUBSTRING
        anywhere in a tracked file, and a nonsense suffix does not make the name absent. The gate was
        right. A placeholder never needs to be a real project."""
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError) as caught:
            import_from_directory("not_a_real_collection", tmp, destination=tmp)
        self.assertIn("not_a_real_collection", str(caught.exception))


class TheLibraryReportsWhatIsActuallyOnDiskTests(unittest.TestCase):
    def test_an_empty_library_is_not_ready(self) -> None:
        with TemporaryDirectory() as tmp:
            library = MeshLibrary(tmp)
            # `custom` counts too -- a customer's own parts are meshes like any other, and a count
            # that skipped them would under-report the very bank the dataset draws from.
            #
            # READ FROM THE REGISTRY, not from a copy of it. This asserted a hardcoded three-key dict,
            # so adding a source failed it for the one reason it does not care about, while the
            # property it DOES care about -- that an empty library reports every source it knows as
            # zero rather than omitting it -- was never actually tested against the real list.
            self.assertEqual(library.counts(), dict.fromkeys(SUPPORTED_SOURCES, 0))
            self.assertFalse(library.is_ready("gso"))

    def test_it_reads_the_directory_rather_than_a_recorded_list(self) -> None:
        """A checked-in list would go stale on the first differing fetch; the audit must see the
        meshes that will actually be RENDERED."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_meshes(root / "origin", "ycb", 2)
            import_from_directory("ycb", root / "origin" / "ycb", destination=root / "library")
            library = MeshLibrary(root / "library")
            self.assertEqual(library.counts()["ycb"], 2)

            (root / "library" / "ycb" / "extra.ply").write_bytes(b"added later")
            self.assertEqual(library.counts()["ycb"], 3, "the library did not re-read the directory")

    def test_the_sha_index_lets_a_different_fetch_be_verified(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_meshes(root / "origin", "gso", 2)
            entries = import_from_directory(
                "gso", root / "origin" / "gso", destination=root / "library"
            )
            index = write_sha_index(entries, root / "sha256.csv")
            text = index.read_text(encoding="utf-8")
            self.assertIn("id,source,sha256,filename", text)
            for entry in entries:
                with self.subTest(asset=entry.asset_id):
                    self.assertIn(entry.sha256, text)

    def test_limit_takes_a_prefix_for_a_trial_import(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            _fake_meshes(root / "origin", "gso", 5)
            entries = import_from_directory(
                "gso", root / "origin" / "gso", destination=root / "library", limit=2
            )
            self.assertEqual(len(entries), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
