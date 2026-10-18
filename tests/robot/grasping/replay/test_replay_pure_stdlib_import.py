"""Guard: the replay tier never imports a numeric/ML library.

The replay/KPI tier is required to be pure standard library — no numpy, torch, scikit-learn, scipy, or pandas — so a
KPI rollup can never pull a heavy numeric dependency onto a hot path. numpy legitimately lives one tier down (geometry /
calibration / scoring), never here. This AST scan turns that convention into a CI contract: it parses every module
under ``grasping/replay/`` and fails if any actually imports a forbidden library. (AST, not a string scan, so the honest
docstring MENTIONS of "no numpy" do not false-positive.) The RL tier it guarded beside it left on 2026-10-16 (the
owner's cleanup).
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_REPLAY = pathlib.Path(__file__).resolve().parents[4] / "src" / "robot" / "grasping" / "replay"
_FORBIDDEN_TOP_LEVEL = {"numpy", "torch", "sklearn", "scipy", "pandas"}


def _imported_top_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".", 1)[0]


class ReplayPureStdlibImportTests(unittest.TestCase):
    def test_replay_never_imports_a_numeric_library(self) -> None:
        offenders: list[str] = []
        files = sorted(_REPLAY.rglob("*.py"))
        self.assertTrue(files, f"no module found under {_REPLAY}")
        for py in files:
            rel = py.relative_to(_REPLAY.parent).as_posix()
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            for mod in _imported_top_modules(tree):
                if mod in _FORBIDDEN_TOP_LEVEL:
                    offenders.append(f"{rel} imports {mod!r}")
        self.assertEqual(
            offenders,
            [],
            "the replay tier must stay pure-stdlib (no numpy/torch/sklearn/scipy/pandas); "
            "offenders: " + "; ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
