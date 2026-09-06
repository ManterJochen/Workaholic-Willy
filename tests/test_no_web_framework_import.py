"""L7 D8 — the library never imports a web framework, and the console never leaks into it.

Workaholic-Willy is a library + CLI stack. R10 removed the dead optional web dependencies and this
test made "no web framework in the library" a contract: an AST scan of `backend/**/*.py` asserts no
module actually `import`s `fastapi`/`uvicorn`/`starlette`. (AST, not a string scan, so the many honest
docstring MENTIONS don't false-positive.)

2026-08-09: the operator console arrived, so the rule was NARROWED rather than dropped. CLAUDE.md always
said "not unless you are deliberately building the API layer"; we did. Three assertions now, and they
only mean something together:

  1. `backend/**` still imports no web framework          -- the library claim, unchanged
  2. `api/**` is the ONLY tree allowed to                 -- one exception, named, not "somewhere"
     (plus `tests/test_api_*.py`: testing an HTTP surface needs a client for it)
  3. nothing under `backend/**` imports `api`             -- the dependency arrow points ONE way

Without (3), (1) would be satisfiable by a library module that reaches the framework through the console
package -- the claim would read as true while being hollow. The console may depend on the library; the
library may never depend on the console.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_BACKEND = _ROOT / "src"
_API = _ROOT / "api"
_FORBIDDEN_TOP_LEVEL = {"fastapi", "uvicorn", "starlette"}


def _imported_top_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".", 1)[0]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                yield node.module.split(".", 1)[0]


class NoWebFrameworkImportTests(unittest.TestCase):
    def test_backend_never_imports_a_web_framework(self) -> None:
        offenders: list[str] = []
        for py in _BACKEND.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            for mod in _imported_top_modules(tree):
                if mod in _FORBIDDEN_TOP_LEVEL:
                    offenders.append(f"{py.relative_to(_BACKEND)} imports {mod!r}")
        self.assertEqual(
            offenders,
            [],
            "the library must not import a web framework (D8); offenders: "
            + "; ".join(offenders),
        )

    def test_the_console_is_the_only_tree_that_may(self) -> None:
        """One named exception, not an open door.

        Every web-framework import in the repo must live under `api/` -- plus the console's own tests,
        which cannot exercise an HTTP surface without a client for it. That exemption is scoped to
        files named `test_api_*.py` rather than to `tests/` as a whole, so a library test cannot pick
        up a framework import by accident; and tests ship with the repo but never with the library, so
        it does not widen what an installed Willy depends on.
        """
        allowed = _API.resolve()
        offenders: list[str] = []
        for py in _ROOT.rglob("*.py"):
            if any(part in {".venv", "__pycache__", "node_modules", "ext_deps"} for part in py.parts):
                continue
            if allowed in py.resolve().parents:
                continue
            if py.parent.name == "tests" and py.name.startswith("test_api_"):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            for mod in _imported_top_modules(tree):
                if mod in _FORBIDDEN_TOP_LEVEL:
                    offenders.append(f"{py.relative_to(_ROOT).as_posix()} imports {mod!r}")
        self.assertEqual(
            offenders, [],
            "only api/** may import a web framework; offenders: " + "; ".join(offenders),
        )

    def test_the_library_never_imports_the_console(self) -> None:
        """The dependency arrow points ONE way, and this is the assertion that makes the other two mean
        something: without it, a `backend` module could reach fastapi through `api` and still pass the
        scan above.
        """
        offenders: list[str] = []
        for py in _BACKEND.rglob("*.py"):
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):  # pragma: no cover - defensive
                continue
            if "api" in set(_imported_top_modules(tree)):
                offenders.append(py.relative_to(_ROOT).as_posix())
        self.assertEqual(
            offenders, [],
            "the library must not import the console package; offenders: " + "; ".join(offenders),
        )


if __name__ == "__main__":
    unittest.main()
