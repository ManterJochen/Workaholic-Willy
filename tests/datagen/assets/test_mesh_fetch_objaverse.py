"""Fetching a collection whose licence sits on each OBJECT rather than on the collection.

⛔ TWO THINGS MAKE THIS DIFFERENT FROM EVERY OTHER SOURCE IN THE FETCHER, and both are refusals
rather than features.

**The licence is per object.** MEASURED over the collection: CC-BY 721k, CC0 3.5k, CC-BY-NC 25k,
CC-BY-NC-SA 52k, CC-BY-SA 16k. So 93,000 objects must be refused, and **68,000 of those carry
ShareAlike** -- which a `startswith("cc-by")` test would have admitted. That hole was closed on
2026-09-01 after `cc-by-sa` and `cc-by-nd` passed the audit; this is the first source where the fix
has anything to catch.

**Attribution is a FETCH CONDITION.** CC-BY obliges naming the author. The sibling collection in the
same file is capped at 30 designs precisely because its metadata carries no author, and that cap is a
documented refusal rather than an oversight. This collection does carry one, so it is usable -- but
only for the rows that actually have it, and a row without an author must be skipped rather than
downloaded and credited to nobody.
"""

from __future__ import annotations

import unittest
from pathlib import Path

# ⭐ A PLAIN PACKAGE IMPORT SINCE 2026-09-04. This used to put `scripts/meshes/` on `sys.path`
# and `import fetch`, because the module lived outside any package. It is `datagen/assets/fetch.py`
# now, so the import needs no help and cannot pick up a different `fetch` from somewhere else.

from datagen.assets.fetch import (
    _OBJAVERSE_LICENCES,
    _licence_is_acceptable,
    SOURCES,
)


class LicenceMappingTests(unittest.TestCase):

    def test_only_attribution_and_public_domain_are_accepted(self) -> None:
        """⭐ THE WHOLE GATE, on the collection's own codes. `by-sa` and `by-nc-sa` are the 68,000
        objects a naive prefix test would have waved through."""
        accepted = {code for code, full in _OBJAVERSE_LICENCES.items()
                    if _licence_is_acceptable(full)}
        self.assertEqual(accepted, {"by", "cc0"})

    def test_every_restricted_variant_is_refused(self) -> None:
        for code in ("by-nc", "by-sa", "by-nc-sa", "by-nd", "by-nc-nd"):
            with self.subTest(code):
                self.assertFalse(_licence_is_acceptable(_OBJAVERSE_LICENCES[code]))

    def test_the_short_codes_map_to_strings_the_shared_gate_can_read(self) -> None:
        """⛔ The gate reads TEXT. `"by"` on its own is not in its acceptable list, so an unmapped
        code would be refused silently, as though the object were non-commercial. The mapping is what
        turns a collection's private vocabulary into the repository's own."""
        for code, full in _OBJAVERSE_LICENCES.items():
            with self.subTest(code):
                self.assertTrue(full.startswith(("cc-by", "cc0")), full)
                self.assertNotEqual(code, full, "a code that maps to itself is not a mapping")

    def test_an_unmapped_code_is_not_silently_permitted(self) -> None:
        """A code the collection adds later must not read as acceptable by accident."""
        for invented in ("by-x", "gpl", "", "proprietary"):
            with self.subTest(invented):
                self.assertNotIn(invented, _OBJAVERSE_LICENCES)
                self.assertFalse(_licence_is_acceptable(invented))


class SourceTests(unittest.TestCase):

    def test_the_source_is_registered_and_declares_its_licence(self) -> None:
        source = SOURCES["objaverse"]
        self.assertTrue(source.licence_is_verified)
        self.assertIn("CC-BY", source.licence)
        self.assertIn("ODC-By", source.licence)

    def test_the_declared_count_is_the_usable_one_not_the_headline(self) -> None:
        """⚠ 800k is the collection; 724.5k is what the licence leaves. A source that advertised the
        headline would have an operator budget disk for objects that will be refused."""
        self.assertEqual(SOURCES["objaverse"].objects, 724_500)

    def test_rows_require_an_author_and_a_profile(self) -> None:
        """Read off the source: the fetch is the expensive half and this test must not perform it."""
        text = Path("datagen/assets/fetch.py").read_text(encoding="utf-8")
        self.assertIn("if not author or not profile:", text)
        self.assertIn("cannot be attributed, so cannot be fetched", text)

    def test_an_unknown_licence_code_raises_rather_than_dropping_rows(self) -> None:
        """A code we have never seen means the collection changed and somebody should look, not that
        thousands of objects should vanish from a fetch without a word."""
        text = Path("datagen/assets/fetch.py").read_text(encoding="utf-8")
        self.assertIn("unmapped licence code(s)", text)
        self.assertIn("needs a decision rather than a default", text)

    def test_the_fetch_writes_an_attribution_row_per_object(self) -> None:
        """Collection-level attribution is a line in `ATTRIBUTIONS`; per-object attribution needs a
        file, and CC-BY is not satisfied without it."""
        text = Path("datagen/assets/fetch.py").read_text(encoding="utf-8")
        self.assertIn("ATTRIBUTION.tsv", text)
        self.assertIn("row['author']", text)
        self.assertIn("row['profile']", text)


if __name__ == "__main__":
    unittest.main()
