"""`datagen` may depend on the robot. The robot may never depend on `datagen`.

A data generator that becomes a runtime dependency of a robot is a generator nobody can delete, refuse
to install, or run on a different machine — and the direction erodes one import at a time, each of them
locally reasonable. So it is asserted, in the same spirit as the web-framework guard.

Also asserted here: the layout half stays **importable without Isaac, torch or a mesh library**. That
is the property the whole test strategy rests on — if `datagen.scenes.layout` ever needs a GPU to
import, every cheap test above it becomes an on-box test, and layout bugs go back to costing a night of
path tracing to find.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATAGEN = REPO / "datagen"
BACKEND = REPO / "src"

#: Modules the Isaac-free half must not need at import time. The renderer will import these; it lives
#: behind a lazy boundary precisely so that this list can stay enforced here.
_HEAVY = frozenset({"isaacsim", "omni", "pxr", "carb", "torch", "torchvision", "trimesh", "open3d"})

#: The parts of datagen that must import on any machine (the layout half + its config and assets).
_LIGHT_MODULES = (
    "datagen",
    "datagen.config",
    "datagen.provenance",
    "datagen.assets",
    "datagen.assets.licensing",
    "datagen.assets.manifest",
    "datagen.assets.procedural",
    "datagen.scenes",
    "datagen.scenes.spec",
    "datagen.scenes.layout",
)


def _imported_top_levels(path: Path) -> set[str]:
    """Top-level module names imported at MODULE scope (function-local imports are the lazy escape)."""
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        # only module-scope imports count; a deferred import inside a function is the sanctioned way
        # to reach a heavy dependency
        if isinstance(node, ast.Module):
            for child in node.body:
                if isinstance(child, ast.Import):
                    names.update(alias.name.split(".")[0] for alias in child.names)
                elif isinstance(child, ast.ImportFrom) and child.module and child.level == 0:
                    names.add(child.module.split(".")[0])
    return names


class LayeringTests(unittest.TestCase):
    def test_the_backend_never_imports_datagen(self) -> None:
        offenders = []
        for path in sorted(BACKEND.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "import datagen" in text or "from datagen" in text:
                offenders.append(path.relative_to(REPO).as_posix())
        self.assertEqual(
            offenders, [],
            "the robot must not depend on the data generator:\n  " + "\n  ".join(offenders)
            + "\nIf the backend needs something from datagen, that something belongs in the backend.",
        )

    def test_the_api_never_imports_datagen(self) -> None:
        api = REPO / "api"
        if not api.exists():
            self.skipTest("no api package")
        offenders = [
            path.relative_to(REPO).as_posix()
            for path in sorted(api.rglob("*.py"))
            if "__pycache__" not in path.parts
            and ("import datagen" in path.read_text(encoding="utf-8", errors="replace"))
        ]
        self.assertEqual(offenders, [])

    def test_the_layout_half_imports_nothing_heavy(self) -> None:
        offenders: list[str] = []
        for module in _LIGHT_MODULES:
            path = DATAGEN / (module.removeprefix("datagen").strip(".").replace(".", "/") or "__init__")
            path = path.with_suffix(".py") if path.suffix != ".py" else path
            if path.is_dir():
                path = path / "__init__.py"
            if not path.exists():
                path = DATAGEN / module.removeprefix("datagen.").replace(".", "/") / "__init__.py"
            self.assertTrue(path.exists(), f"cannot locate {module}")
            heavy = _imported_top_levels(path) & _HEAVY
            if heavy:
                offenders.append(f"{module}: {sorted(heavy)}")
        self.assertEqual(
            offenders, [],
            "the Isaac-free half imported a heavy dependency at module scope:\n  "
            + "\n  ".join(offenders)
            + "\nMove it inside the function that needs it -- the whole cheap-test strategy depends on "
              "layout importing on a laptop.",
        )

    def test_the_light_modules_really_do_import(self) -> None:
        """The guard above is static; this proves the modules load in this environment for real."""
        import importlib

        for module in _LIGHT_MODULES:
            with self.subTest(module=module):
                importlib.import_module(module)


if __name__ == "__main__":
    unittest.main()
