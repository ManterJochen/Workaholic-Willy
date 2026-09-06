"""Guard: the RL + replay tiers never import a numeric/ML library.

The RL runtime and the replay/KPI tier are required to be pure standard library — no numpy, torch,
scikit-learn, scipy, or pandas — so a policy decision or a KPI rollup can never pull a heavy numeric
dependency onto a hot path. numpy legitimately lives one tier down (geometry / calibration / scoring),
never here. This AST scan turns that convention into a CI contract: it parses every module under
``grasping/rl/`` and ``grasping/replay/`` and fails if any actually imports a forbidden library.
(AST, not a string scan, so the honest docstring MENTIONS of "no numpy" do not false-positive.)
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_GRASPING = pathlib.Path(__file__).resolve().parents[1] / "src" / "robot" / "grasping"
_GUARDED_DIRS = ("rl", "replay")
_FORBIDDEN_TOP_LEVEL = {"numpy", "torch", "sklearn", "scipy", "pandas"}

# Opt-in "heavy" runtime tiers (robot.rl.runtime_tier = numpy|torch). These are EXEMPT from the
# numpy/torch ban because they exist precisely to use it — but the exemption is only honest if they
# are never eagerly imported: they must be reachable exclusively via the lazy import inside
# ``online_state.make_online_accumulator``. The second test enforces that, so activating a heavy tier
# is a deliberate opt-in and the DEFAULT import path stays byte-for-byte numpy-free.
_HEAVY_TIER_MODULES = {"_online_numpy", "_online_torch"}
_HEAVY_TIER_FILES = {f"rl/{m}.py" for m in _HEAVY_TIER_MODULES}


def _imported_top_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".", 1)[0]


def _module_level_import_froms(tree: ast.AST):
    """Yield ``ImportFrom`` nodes that execute at module import time — i.e. module-body
    statements (incl. top-level if/try/with/for), but NOT imports nested inside a
    ``def``/``async def``/``class`` (those are lazy)."""

    stack = list(getattr(tree, "body", []))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(node, ast.ImportFrom):
            yield node
        for fieldname in ("body", "orelse", "finalbody"):
            stack.extend(getattr(node, fieldname, []) or [])
        for handler in getattr(node, "handlers", []) or []:
            stack.extend(getattr(handler, "body", []) or [])


class RlReplayPureStdlibImportTests(unittest.TestCase):
    def test_rl_and_replay_never_import_a_numeric_library(self) -> None:
        offenders: list[str] = []
        for sub in _GUARDED_DIRS:
            for py in (_GRASPING / sub).rglob("*.py"):
                rel = py.relative_to(_GRASPING).as_posix()
                if rel in _HEAVY_TIER_FILES:
                    continue  # opt-in heavy tier; honesty enforced by the lazy-import test below
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
            "the RL + replay tiers must stay pure-stdlib (no numpy/torch/sklearn/scipy/pandas); "
            "offenders: " + "; ".join(offenders),
        )

    def test_heavy_tiers_are_never_eagerly_imported(self) -> None:
        # The heavy-tier modules are exempt from the numeric-library ban only because they are
        # reachable exclusively via a lazy import (make_online_accumulator). If any module imports
        # one at top level, the exemption becomes a hole: importing rl/ would pull numpy/torch onto
        # the default path. Guard against that regression.
        offenders: list[str] = []
        for sub in _GUARDED_DIRS:
            for py in (_GRASPING / sub).rglob("*.py"):
                rel = py.relative_to(_GRASPING).as_posix()
                if rel in _HEAVY_TIER_FILES:
                    continue
                try:
                    tree = ast.parse(py.read_text(encoding="utf-8"))
                except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
                    continue
                for node in _module_level_import_froms(tree):
                    if node.level == 1 and node.module:
                        if node.module.split(".", 1)[0] in _HEAVY_TIER_MODULES:
                            offenders.append(f"{rel} top-level imports {node.module!r}")
        self.assertEqual(
            offenders,
            [],
            "heavy-tier modules must only be lazily imported (inside a function), never at module "
            "top level; offenders: " + "; ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
