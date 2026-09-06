"""Every import of a `deep` module in live code has to name a module that still exists.

⛔⛔ **THE HOLE THIS CLOSES, AND IT WAS MEASURED THE HARD WAY.** The cleanup of 2026-09-04 deleted
fifteen modules of a retired architecture. Two instruments were run against the result and both said
green, and one live production file was broken the whole time: `calculator_factory.py` imported
`deep.generator_loop` from INSIDE a function body.

Neither instrument could see it, for two different reasons that both matter:

  * **pytest collection** imports a test module, so it catches a top-level import of a vanished
    module. An import inside a function body is not executed at collection, so nothing looked at it
    until a config actually asked for a deep calculator.
  * **mypy** would normally report `Cannot find implementation or library stub`, but this project
    sets `ignore_missing_imports = true` in `pyproject.toml` (line 32) so that the lazy vendor SDKs
    type-check without stubs. That setting cannot distinguish a vendor SDK that is absent by design
    from a first-party module that was deleted by mistake. MEASURED: `mypy` on the broken file
    reported "Success: no issues found in 1 source file".

So the ONE class of defect a deletion sweep produces is exactly the class both gates are blind to.
This test walks the AST instead, at every nesting depth, and resolves each `deep.*` module name.

⚠ **AST, NOT A TEXT SEARCH.** A grep for a deleted module name matches the comment that explains why
it was deleted, so it passes on the defect and fails on the repair. That trap was hit four times in
one day on this branch before it was written down.
"""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

#: Trees that ship. Tests are excluded on purpose: a test may legitimately assert that a module is
#: GONE, and it would then have to name it.
LIVE_TREES = ("src", "datagen", "api", "scripts")

PREFIX = "src.robot.grasping.deep"


def _deep_imports(path: Path) -> list[tuple[str, int]]:
    """Every `deep`-package module named by an import in `path`, with its line number."""
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError):  # not our problem here; ruff owns it
        return []
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            # a relative import carries no module path we can resolve from here
            if node.level or not node.module or not node.module.startswith(PREFIX):
                continue
            found.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith(PREFIX):
                    found.append((alias.name, node.lineno))
    return found


class DeepImportsResolveTests(unittest.TestCase):
    def test_every_deep_import_in_live_code_names_a_module_that_exists(self) -> None:
        root = Path(__file__).resolve().parents[1]
        broken: list[str] = []
        checked = 0
        for tree in LIVE_TREES:
            for path in sorted((root / tree).rglob("*.py")):
                for module, line in _deep_imports(path):
                    checked += 1
                    if importlib.util.find_spec(module) is None:
                        broken.append(f"{path.relative_to(root)}:{line} imports {module}")
        self.assertGreater(checked, 20, "the walk found almost nothing; it is not looking properly")
        self.assertEqual(
            [], broken,
            "these live imports name a module that no longer exists. mypy cannot see this "
            "(ignore_missing_imports) and collection cannot see it inside a function body:\n  "
            + "\n  ".join(broken))

    def test_it_would_catch_an_import_nested_inside_a_function(self) -> None:
        """The control. Without this the test could be passing for the wrong reason.

        The defect that motivated the file was nested two levels deep, so a walker that only reads
        `tree.body` would report a clean sweep over a broken tree.
        """
        source = (
            "def outer():\n"
            "    def inner():\n"
            "        from src.robot.grasping.deep.gone_for_good import Thing\n"
            "        return Thing\n"
            "    return inner\n"
        )
        path = Path(self.enterContext(__import__("tempfile").TemporaryDirectory())) / "probe.py"
        path.write_text(source, encoding="utf-8")
        found = _deep_imports(path)
        self.assertEqual([(f"{PREFIX}.gone_for_good", 3)], found)
        self.assertIsNone(importlib.util.find_spec(found[0][0]))


if __name__ == "__main__":
    unittest.main()
