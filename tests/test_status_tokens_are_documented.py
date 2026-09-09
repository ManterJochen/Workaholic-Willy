"""Every status token the guard can return has to be findable by an operator who reads the guide.

⛔ **WHY THIS FILE EXISTS.** Three separate places listed the tokens in prose, and on 2026-09-09 all
three were behind the code by one token: `variant_model_mismatch` had shipped without any of them
noticing, and the comment in `self_collision.py` still named three of the five. Each sentence was
correct the day it was written; the change underneath turned it over, and nothing connected the two.
That is not a doc problem, it is a missing edge. A grep cannot find a sentence that has drifted,
because a drifted sentence still reads perfectly well.

⭐ **DERIVED, NOT DECLARED.** The tokens are read out of the RETURN STATEMENTS of
``mesh_backend_status`` with the ast module, so a new token is caught by the act of returning it. A
list here would be a fourth place to fall behind, and this file would then be part of the problem it
was written to solve.

The check is deliberately weak: it asks that the token APPEARS in the paragraph, not that the
paragraph is good. What a token means is a judgement no test can make; whether the prose mentions
it at all is exactly what kept going wrong.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_SOURCE = _ROOT / "src" / "robot" / "safety" / "_fcl_self_collision.py"
_GUIDE = _ROOT / "docs" / "guide" / "04-robot-and-safety.md"

#: The phrase that opens the enumeration in the guide. A PHRASE and not a line number, because a
#: line number is the same kind of unanchored claim this file exists to prevent, and not a line
#: PREFIX either: one tree writes the sentence at the start of a line and the other continues it
#: from the previous one, and a test that only reads line starts finds nothing in the second.
#:
#: A PARAGRAPH and not a line: the first run of this test failed on a token that had simply wrapped
#: onto the next line. Reading one line would have made this a rule about where sentences may break,
#: which is not what anyone here cares about.
_GUIDE_ANCHOR = "okens are "


def _returned_tokens() -> set[str]:
    """Every string literal ``mesh_backend_status`` can return, read from the function itself."""
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "mesh_backend_status":
            found: set[str] = set()
            for stmt in ast.walk(node):
                if not isinstance(stmt, ast.Return) or stmt.value is None:
                    continue
                # walk the returned EXPRESSION, so a conditional return contributes both branches
                for part in ast.walk(stmt.value):
                    if isinstance(part, ast.Constant) and isinstance(part.value, str):
                        found.add(part.value)
            return found
    raise AssertionError(f"mesh_backend_status is no longer defined in {_SOURCE}")


def _guide_paragraph() -> str:
    """The paragraph the enumeration opens, from its anchor to the next blank line."""
    lines = _GUIDE.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if _GUIDE_ANCHOR not in line:
            continue
        out = []
        for rest in lines[i:]:
            if not rest.strip():
                break
            out.append(rest)
        return "\n".join(out)
    raise AssertionError(f"no line containing {_GUIDE_ANCHOR!r} in {_GUIDE}")


class TheTokensAreDiscoverableTests(unittest.TestCase):
    def test_there_are_several_tokens(self) -> None:
        """The control. Reading zero tokens would make every assertion below pass loudly."""
        self.assertGreaterEqual(len(_returned_tokens()), 5, "the ast read found almost nothing, "
                                                            "which means it found the wrong function")

    def test_the_guide_names_every_token(self) -> None:
        paragraph = _guide_paragraph()
        missing = sorted(t for t in _returned_tokens() if f"`{t}`" not in paragraph)
        self.assertEqual(
            missing, [],
            f"the guard can return these and the guide does not mention them, so an operator who "
            f"sees one in a log has nowhere to look it up:\n  {paragraph.strip()}",
        )

    def test_the_check_can_fail(self) -> None:
        """⚠ THE SELF-FAILURE CONTROL. A token the code cannot return must NOT be in the paragraph.
        Without this, a guide paragraph listing every plausible word would pass forever."""
        self.assertNotIn("`no_such_token`", _guide_paragraph())


class EveryFailingTokenCarriesAdviceTests(unittest.TestCase):
    """A token in a log with no hint beside it tells an operator that something is wrong and not what
    to do about it. ``ok`` is the one token that needs none."""

    def test_every_non_ok_token_has_a_hint(self) -> None:
        from src.robot.safety._fcl_self_collision import _STATUS_HINTS

        missing = sorted(t for t in _returned_tokens() if t != "ok" and t not in _STATUS_HINTS)
        self.assertEqual(missing, [], "these degrade the guard and say nothing about the remedy")

    def test_no_hint_describes_a_token_that_cannot_happen(self) -> None:
        """The other direction. A hint for a retired token is advice nobody will ever be given."""
        from src.robot.safety._fcl_self_collision import _STATUS_HINTS

        extra = sorted(set(_STATUS_HINTS) - _returned_tokens())
        self.assertEqual(extra, [], "mesh_backend_status can no longer return these")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
