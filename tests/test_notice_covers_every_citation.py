"""Every paper the code cites is attributed in NOTICE, not only in a comment.

⛔⛔ THE GAP THIS CLOSES. Four borrowings entered the restarted spine and none reached `NOTICE`: a
gripper-conditioning representation, the rule that decides the rotation representation, a crop
normalisation, and the serialized-transformer lineage the backbone is a simplification of. Each was
cited by arXiv ID in a source comment, and a citation in a comment is not an attribution.

⚠ AND `NOTICE` IS THE ONLY FILE ALLOWED TO NAME A SOURCE. `test_license_boundary` refuses forbidden
project names anywhere in the tree, with `NOTICE` exempt, precisely so that attribution has one home.
That makes the two guards a pair: one says a name may appear only there, this one says it must.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
#: An arXiv identifier as the code writes it: `2410.04826`, with or without an `arXiv ` prefix.
_ARXIV = re.compile(r"\b(\d{4}\.\d{4,5})\b")

#: Where a citation may appear. Deliberately narrow: this guard is about work the LEARNED stack
#: borrowed, and widening it to the whole tree would sweep up version numbers and dates.
_SOURCES = (Path("src/robot/grasping/deep"),)

#: ⚠ NUMBERS THAT LOOK LIKE arXiv IDS AND ARE NOT. Listed rather than filtered by heuristic, because a
#: heuristic that silently drops a real citation is the failure this file exists to prevent.
_NOT_CITATIONS: frozenset[str] = frozenset()


def _cited() -> dict[str, list[str]]:
    """Every arXiv id in the learned stack, and which files carry it."""
    found: dict[str, list[str]] = {}
    for root in _SOURCES:
        for path in (_ROOT / root).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            for match in _ARXIV.finditer(text):
                identifier = match.group(1)
                if identifier in _NOT_CITATIONS:
                    continue
                # Only count it when the line reads like a citation rather than a bare number.
                line = text[text.rfind("\n", 0, match.start()) + 1:
                            text.find("\n", match.end())]
                if "arxiv" in line.lower() or "`" + identifier + "`" in line:
                    found.setdefault(identifier, []).append(
                        path.relative_to(_ROOT).as_posix())
    return found


class CitationTests(unittest.TestCase):

    def test_every_cited_paper_is_in_NOTICE(self) -> None:
        """⛔ THE DEFECT ITSELF. Three arXiv ids sat in source comments and in no attribution."""
        notice = (_ROOT / "NOTICE").read_text(encoding="utf-8")
        cited = _cited()
        self.assertTrue(cited, "no citations found at all; the pattern or the paths are wrong")
        missing = sorted(i for i in cited if i not in notice)
        self.assertEqual(missing, [], "\n".join([
            "these papers are cited in the code and attributed nowhere. NOTICE is the only file "
            "allowed to name a source, which is exactly why it has to name this one:",
            *(f"  {i} — cited in {', '.join(cited[i])}" for i in missing)]))

    def test_the_guard_can_actually_FAIL(self) -> None:
        """A guard nobody has seen fail is a guard nobody can trust."""
        notice = (_ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertNotIn("9999.99999", notice)
        self.assertTrue(_ARXIV.search("see arXiv 9999.99999 for the derivation"))

    def test_the_DATASET_the_importer_can_fetch_is_attributed(self) -> None:
        """⚠ A dataset is not a dependency, and it still needs naming: the importer streams it at run
        time, so a reader has to be able to find out what it is and under what terms."""
        from src.robot.grasping.deep.foreign.grasp_anything import SOURCE

        notice = (_ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn(SOURCE.licence, notice)
        self.assertIn("Grasp-Anything-6D", notice)
        self.assertIn("https://", notice, "NOTICE names no address for this dataset")
        self.assertIn("not vendored or redistributed", notice,
                      "NOTICE does not say the importer streams this rather than shipping it")
        self.assertEqual(SOURCE.scenes, 994_860)

    def test_the_simplified_backbone_says_so_in_NOTICE_too(self) -> None:
        """⚠ The module admits it is a simplification of a published design. That admission belongs
        where a reader looks for provenance, not only where a maintainer looks for code."""
        notice = (_ROOT / "NOTICE").read_text(encoding="utf-8")
        self.assertIn("serialized", notice.lower())
        self.assertIn("independently implemented", notice)
        self.assertIn("rather than reproducing the exact published implementation", notice)
        self.assertIn("No source code from the referenced work is included", notice)


if __name__ == "__main__":
    unittest.main()
