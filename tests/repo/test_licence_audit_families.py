"""The asset audit must refuse ShareAlike and NoDerivatives, and must not do it by punctuation.

⛔ BOTH FAMILIES PASSED THIS AUDIT until 2026-09-01, verified against the real gate rather than
assumed. The rule was `license_id.lower().startswith(("cc0", "cc-by", "own"))`, and
`"cc-by-nd-4.0"` starts with `"cc-by"`.

    PASS   'cc-by-sa-4.0'      copyleft, admitted
    PASS   'cc-by-nd-4.0'      NO DERIVATIVES, admitted
    REFUSE 'CC BY-ND 4.0'      the same licence, refused only because of the space

NoDerivatives is not a subtle problem: every rendered image of a mesh is a derivative work, and ND
forbids distributing derivatives at all. ShareAlike is copyleft, so a corpus built from one would
have to ship under the same terms.

The refusals of the space-separated spellings looked like the rule working and were an accident of
string formatting. That accident is what makes a normaliser dangerous to add on its own: the moment
somebody normalises so that `"CC BY 4.0"` is recognised, the two families stop being refused unless
their rules are explicit. Both halves therefore land together, and both are pinned here.

Every string below is one a real source publishes or one SPDX defines. None are invented.
"""

from __future__ import annotations

import logging
import unittest

from datagen.assets.licensing import (
    audit_asset_rows,
    is_noncommercial,
    is_noderivatives,
    is_sharealike,
    normalise_license,
)

#: `(licence, may an asset under it be rendered and shipped)`.
CASES: tuple[tuple[str, bool], ...] = (
    ("CC0-1.0", True),
    ("cc0-1.0", True),
    ("CC-BY-4.0", True),
    ("CC BY 4.0", True),                 # the figshare record spelling, refused before 2026-09-01
    ("own", True),
    ("cc-by-sa-4.0", False),             # PASSED before this test existed
    ("CC BY-SA 4.0", False),
    ("cc-by-nd-4.0", False),             # PASSED before this test existed
    ("CC BY-ND 4.0", False),
    ("cc-by-nc-nd-4.0", False),
    ("CC BY-NC 4.0", False),
    ("BSD-3-Clause", False),             # refused; extending the allowlist is an owner decision
    ("Public Domain", False),            # refused; same, and recorded as such in the corpus spec
    ("", False),
)


class TheAuditRefusesBothCopyleftFamiliesTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)      # the audit logs its refusals; the test asserts them

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    @staticmethod
    def _accepted(licence: str) -> bool:
        return not audit_asset_rows(
            [{"id": "probe", "source": "gso", "license": licence, "attribution": "someone"}])

    def test_every_published_spelling_is_judged_correctly(self) -> None:
        for licence, allowed in CASES:
            with self.subTest(licence=licence):
                self.assertEqual(self._accepted(licence), allowed)

    def test_spelling_does_not_change_the_answer(self) -> None:
        """The defect in one sentence: the same licence, spelled two ways, answered two ways."""
        for a, b in (("cc-by-nd-4.0", "CC BY-ND 4.0"),
                     ("cc-by-sa-4.0", "CC BY-SA 4.0"),
                     ("cc-by-4.0", "CC BY 4.0")):
            with self.subTest(pair=(a, b)):
                self.assertEqual(self._accepted(a), self._accepted(b))

    def test_the_refusal_says_which_family_and_why(self) -> None:
        """A refusal nobody can read gets worked around instead of understood."""
        row = [{"id": "probe", "source": "gso", "license": "cc-by-nd-4.0", "attribution": "a"}]
        message = " ".join(audit_asset_rows(row)).lower()
        self.assertIn("no-derivatives", message)
        self.assertIn("derivative", message)


class ThePredicatesAreSpellingIndependentTests(unittest.TestCase):
    def test_normalisation_collapses_every_separator(self) -> None:
        for text in ("CC BY-ND 4.0", "cc_by_nd_4.0", "CC  BY - ND - 4.0", "Cc-By-Nd-4-0"):
            with self.subTest(text=text):
                self.assertEqual(normalise_license(text), "cc-by-nd-4-0")

    def test_sharealike(self) -> None:
        for yes in ("cc-by-sa-4.0", "CC BY-SA 4.0", "Creative Commons - Attribution - Share Alike",
                    "ShareAlike"):
            self.assertTrue(is_sharealike(yes), yes)
        for no in ("cc-by-4.0", "CC BY 4.0", "CC0-1.0", "own", ""):
            self.assertFalse(is_sharealike(no), no)

    def test_noderivatives(self) -> None:
        for yes in ("cc-by-nd-4.0", "CC BY-ND 4.0",
                    "Creative Commons - Attribution - No Derivatives", "NoDerivatives"):
            self.assertTrue(is_noderivatives(yes), yes)
        for no in ("cc-by-4.0", "CC BY 4.0", "CC0-1.0", "own", ""):
            self.assertFalse(is_noderivatives(no), no)

    def test_noncommercial_still_catches_what_it_always_did(self) -> None:
        """The one predicate that already worked. Pinned so the normalisation cannot weaken it."""
        for yes in ("cc-by-nc-4.0", "CC BY-NC 4.0", "Creative Commons - Attribution - Non-Commercial",
                    "cc-by-nc-sa-4.0"):
            self.assertTrue(is_noncommercial(yes), yes)
        for no in ("cc-by-4.0", "CC0-1.0", "own"):
            self.assertFalse(is_noncommercial(no), no)

    def test_the_three_families_do_not_overlap_into_a_false_pass(self) -> None:
        """A licence carrying two clauses must be refused for at least one of them."""
        for text in ("cc-by-nc-sa-4.0", "cc-by-nc-nd-4.0", "CC BY-NC-ND 4.0"):
            with self.subTest(licence=text):
                self.assertTrue(is_noncommercial(text) or is_sharealike(text)
                                or is_noderivatives(text))


class TheSourcesWeShipStillPassTests(unittest.TestCase):
    def setUp(self) -> None:
        logging.disable(logging.CRITICAL)

    def tearDown(self) -> None:
        logging.disable(logging.NOTSET)

    def test_every_registered_collection_survives_the_tightened_audit(self) -> None:
        """The tightening must not have refused something we already ship."""
        from datagen.assets.library import ATTRIBUTIONS, LICENSES

        rows = [{"id": f"{key}_probe", "source": key, "license": licence,
                 "attribution": ATTRIBUTIONS[key]} for key, licence in LICENSES.items()]
        self.assertEqual(audit_asset_rows(rows), [])
