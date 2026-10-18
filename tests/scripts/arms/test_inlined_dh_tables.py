"""No UR link lengths are inlined anywhere any more: the build reads the arm description (P0f F7).

They were written out three times until P0d D3b and twice until P0f F7: the arm description, the descriptor builder's
``_DH_TABLES`` and the bake's ``UR_DH``, which this file compared value for value because the interpreters that read
them cannot import one another. Since F7a and F7b both scripts read the description through
``scripts/arms/_arm_file.py``, stdlib and PyYAML, which the cuRobo environment's Python 3.10 runs too.

What this file keeps is the half that still matters. A copy nothing compares is how the old ones drifted: the bake once
inlined a fourth, said in a comment that this file compared it, and it did not. These numbers place collision geometry,
and a wrong ``d`` writes a bundle whose links sit where the arm is not while everything reports healthy. So a literal
table of UR link lengths anywhere under ``scripts/`` or ``backend/`` is found by its shape and fails here the day it is
written.
"""

from __future__ import annotations

import ast
import pathlib
import re
import tempfile
import unittest
from collections.abc import Iterator

_ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Where an inlined copy could live. The authority is YAML under config/arms, not a Python literal.
_SCANNED = (_ROOT / "scripts", _ROOT / "src")

#: A UR model key as the tables spelled it: ur3, ur5e, ur16e, and ur8long once it exists.
_MODEL_KEY = re.compile(r"ur\d+(?:e|long)?\Z")


def _is_number(node: ast.expr) -> bool:
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        node = node.operand
    return isinstance(node, ast.Constant) and type(node.value) in (int, float)


def _is_dh_table(value: ast.expr | None) -> bool:
    """A dict literal keyed only by UR model names, each holding six rows of three numbers."""
    if not isinstance(value, ast.Dict) or len(value.keys) < 2:
        return False
    for key, rows in zip(value.keys, value.values):
        if not (isinstance(key, ast.Constant) and isinstance(key.value, str) and _MODEL_KEY.match(key.value)):
            return False
        if not (isinstance(rows, (ast.Tuple, ast.List)) and len(rows.elts) == 6):
            return False
        for row in rows.elts:
            if not (isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) == 3
                    and all(_is_number(x) for x in row.elts)):
                return False
    return True


def _inlined_tables(roots: "tuple[pathlib.Path, ...]" = _SCANNED) -> Iterator[tuple[pathlib.Path, str]]:
    """Every assignment of a literal DH table under ``roots``, found by shape, not by name."""
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                    value = node.value
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    names = [node.target.id]
                    value = node.value
                else:
                    continue
                if _is_dh_table(value):
                    for name in names:
                        yield path, name


class NoLinkLengthsAreInlinedTests(unittest.TestCase):
    def test_no_script_or_module_inlines_a_table_of_ur_link_lengths(self) -> None:
        found = sorted(f"{path.relative_to(_ROOT).as_posix()}:{name}" for path, name in _inlined_tables())
        self.assertEqual([], found, "read the arm description through scripts/arms/_arm_file.py instead")

    def test_the_scan_sees_a_table_of_that_shape(self) -> None:
        """⭐ THE CONTROL: a scan that matched nothing would call every repository clean, so it is shown one table."""
        row = "(0.0, 0.1625, 1.570796327)"
        rows = ", ".join([row] * 6)
        with tempfile.TemporaryDirectory() as tmp:
            planted = pathlib.Path(tmp) / "planted.py"
            planted.write_text(f"UR_DH = {{'ur5e': ({rows}), 'ur3e': ({rows})}}\n", encoding="utf-8")
            self.assertEqual([(planted, "UR_DH")], list(_inlined_tables((pathlib.Path(tmp),))))
            planted.write_text(f"NOT_UR = {{'arm': ({rows}), 'hand': ({rows})}}\n", encoding="utf-8")
            self.assertEqual([], list(_inlined_tables((pathlib.Path(tmp),))), "keyed by something other than a UR")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
