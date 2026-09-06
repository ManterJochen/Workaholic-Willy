"""The licence gate, probed with every string the four mesh sources actually publish.

THE PUNCTUATION USED TO DO THE GATING, and by accident in both directions.

`"CC BY 4.0"`, which is exactly what the Coles record's API returns, did not contain the token
`"cc-by"` and was REFUSED. `"Creative Commons - Attribution"` did not contain
`"creative commons attribution"` either. Two plainly permissive licences read as forbidden because
of one space and one hyphen, and the Coles fetcher would have refused its own source on the first
run. Nothing had noticed because the one source in the file happened to publish a third spelling.

Teaching the gate to read separators then opens the opposite hole: `"Creative Commons - Attribution
- Share Alike"` was refused only because those hyphens broke the token test, so normalising would
have ADMITTED 3,680 ShareAlike files of one collection that were previously refused by luck.
ShareAlike is copyleft and a rendered corpus is a derivative work, so the rule is explicit now.

The cases below are real strings, copied from what each source publishes, not invented ones.
"""

from __future__ import annotations

import unittest

# ⭐ A PLAIN PACKAGE IMPORT SINCE 2026-09-04. The module moved from `scripts/meshes/fetch.py`
# into `datagen/assets/fetch.py`, so the path insertion this file carried is gone. The old form
# also risked importing some other `fetch` that happened to be on the path first.

from datagen.assets import fetch


#: `(licence string, may we fetch it, where the string comes from)`.
CASES: tuple[tuple[str, bool, str], ...] = (
    ("CC BY 4.0", True, "figshare API, the Coles record"),
    ("Creative Commons Attribution 4.0 International", True, "Fuel, every GSO model"),
    ("Creative Commons - Attribution", True, "Thingi10K metadata, 2,945 rows"),
    ("Creative Commons - Public Domain Dedication", True, "Thingi10K metadata, 99 rows"),
    ("Public Domain", True, "Thingi10K metadata, 88 rows"),
    ("CC0-1.0", True, "SPDX"),
    ("Creative Commons - Attribution - Share Alike", False, "Thingi10K, 3,680 rows, copyleft"),
    ("CC BY-SA 4.0", False, "the same licence spelled as SPDX"),
    ("Creative Commons - Attribution - Non-Commercial", False, "Thingi10K, 1,581 rows"),
    ("Attribution - Non-Commercial - Share Alike", False, "Thingi10K, 975 rows"),
    ("Attribution - Non-Commercial - No Derivatives", False, "Thingi10K, 330 rows"),
    ("Creative Commons - Attribution - No Derivatives", False, "Thingi10K, 84 rows"),
    ("CC BY-ND 4.0", False, "the same licence spelled as SPDX"),
    ("GNU - GPL", False, "Thingi10K, 202 rows"),
    ("GNU - LGPL", False, "Thingi10K, 2 rows"),
    ("BSD License", False, "Thingi10K, 10 rows"),
    ("unknown_license", False, "Thingi10K, 4 rows"),
    ("", False, "silence is not permission"),
)


class TheLicenceGateReadsSeparatorsTests(unittest.TestCase):
    def test_every_published_string_is_read_correctly(self) -> None:
        for text, allowed, origin in CASES:
            with self.subTest(licence=text):
                self.assertEqual(
                    fetch._licence_is_acceptable(text), allowed,  # noqa: SLF001 - the gate IS the subject
                    f"{text!r} ({origin}) should be {'allowed' if allowed else 'refused'}")

    def test_spelling_does_not_change_the_answer(self) -> None:
        """The bug in one sentence: the same licence, spelled three ways, must answer once."""
        spellings = ("CC BY 4.0", "CC-BY-4.0", "cc_by_4.0",
                     "Creative Commons - Attribution", "Creative Commons Attribution")
        answers = {fetch._licence_is_acceptable(s) for s in spellings}  # noqa: SLF001
        self.assertEqual(answers, {True}, f"the same licence answered differently: {spellings}")

    def test_sharealike_is_refused_on_purpose_and_not_by_punctuation(self) -> None:
        """Pin the rule that normalising would otherwise have removed."""
        for text in ("Creative Commons - Attribution - Share Alike", "CC BY-SA 4.0",
                     "cc by sa 4.0", "ShareAlike"):
            with self.subTest(licence=text):
                self.assertFalse(fetch._licence_is_acceptable(text))  # noqa: SLF001


class TheThingiSliceAdmitsOnlyWhatWeCanHonourTests(unittest.TestCase):
    """A second, stricter filter sits in front of that collection, and it is exact-match by design.

    The collection's metadata has no author column. Its columns are ID, Thing ID, License, Link and
    geometry flags. So a CC-BY row cannot be credited from anything we hold, and the library's own
    ATTRIBUTIONS map already records that this is a refusal rather than a gap to route around. Only
    the two licences that oblige nothing are fetched.
    """

    def test_only_the_two_attribution_free_licences_are_in_the_set(self) -> None:
        self.assertEqual(
            fetch._THINGI_NO_ATTRIBUTION_REQUIRED,  # noqa: SLF001
            frozenset({"Creative Commons - Public Domain Dedication", "Public Domain"}))

    def test_a_substring_test_would_admit_the_forbidden_superstring(self) -> None:
        """MEASURED: `in` instead of `==` admits 8,290 of 10,000 rows, 5,345 of them NC, ND or SA."""
        prefix = "Creative Commons - Attribution"
        superstring = "Creative Commons - Attribution - Non-Commercial"
        self.assertIn(prefix, superstring, "the trap this test exists for has changed shape")
        self.assertNotIn(superstring, fetch._THINGI_NO_ATTRIBUTION_REQUIRED)  # noqa: SLF001
        self.assertFalse(fetch._licence_is_acceptable(superstring))  # noqa: SLF001


class TheRegistryAgreesWithTheLibraryTests(unittest.TestCase):
    def test_every_fetchable_source_is_one_the_library_can_read(self) -> None:
        from datagen.assets.library import ATTRIBUTIONS, LICENSES, SUPPORTED_SOURCES

        for key, source in fetch.SOURCES.items():
            with self.subTest(source=key):
                self.assertIn(key, SUPPORTED_SOURCES, "fetched into a directory nothing reads")
                self.assertIn(key, LICENSES, "no licence recorded for a source we fetch")
                self.assertIn(key, ATTRIBUTIONS, "MeshEntry.attribution would raise KeyError")
                self.assertTrue(fetch._licence_is_acceptable(source.licence),  # noqa: SLF001
                                f"{key} declares {source.licence!r}, which its own gate refuses")

    def test_the_declared_licence_survives_the_asset_audit(self) -> None:
        from datagen.assets.library import ATTRIBUTIONS, LICENSES
        from datagen.assets.licensing import audit_asset_rows

        rows = [{"id": f"{key}_probe", "source": key, "license": LICENSES[key],
                 "attribution": ATTRIBUTIONS[key]} for key in fetch.SOURCES]
        self.assertEqual(audit_asset_rows(rows), [])
