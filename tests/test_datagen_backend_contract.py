"""Every name `datagen` imports from `src` still exists, and the private ones are listed.

⛔⛔ **THE REASON THIS GUARD EXISTS: NOTHING ELSE CAN SEE THIS BREAK.** `datagen` reaches into
`src` at 119 name-imports across 37 modules, and MEASURED, **68 of those sit inside a function
body**. (Counted as NAMES; counted as import statements it is 97 and 52, which is the same surface
seen two ways. The name is what breaks, so the name is what this counts.) Two instruments that look
like they cover it do not:

  * **pytest collection never executes a function body**, so an import that would raise `ImportError`
    is not reached by importing the module. The test suite goes green.
  * **mypy is configured with `ignore_missing_imports = true`** (`pyproject.toml:32`), so a module
    that has VANISHED is a silent pass, not an error.

So a rename on the `src` side surfaces only when that code path actually runs, which for this
tool can be several hours into a render. That is not a hypothetical: it is the failure that occurred
on 2026-09-04, in exactly this shape.

⚠ **THE DEPENDENCY IS ONE-WAY AND STAYS THAT WAY.** `datagen` imports `src`; `src` never
imports `datagen`. That direction is asserted here too, because the day it reverses this guard is
measuring a cycle instead of a contract.

⭐ **THE PRIVATE CROSSINGS ARE THE POINT.** Six underscore-prefixed names cross the package boundary.
A refactor is entitled to assume an underscore means "nobody outside uses this", and no tool
contradicts it. They are enumerated below so the assumption is contradicted in writing, by a test
that fails when one of them moves.

⚠ **THIS PARSES, IT NEVER IMPORTS.** Importing the surface would pull torch through several modules,
and this repository's recorded lesson is that one pytest file plus mypy killed a live training run.
A guard that cannot be run while the box is busy gets skipped exactly when the tree is moving
fastest, which is when it is needed.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_DATAGEN = _REPO / "datagen"

#: MEASURED 2026-09-04. Floors, not targets: a scan that silently stops matching fails here rather
#: than passing over an empty set, which is the way a guard in this repository has gone inert three
#: times before.
_MIN_EDGES = 100
_MIN_MODULES = 30


def _module_file(dotted: str) -> Path | None:
    """The file behind `src.a.b`, whether it is a module or a package."""
    base = _REPO.joinpath(*dotted.split("."))
    for candidate in (base.with_suffix(".py"), base / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _defined_names(path: Path) -> set[str]:
    """Every name a module binds at top level, including what it re-exports.

    A re-export counts: `from x import y` at module level genuinely makes `y` importable from here,
    and several `src` packages are deliberately built that way.
    """
    names: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update(a.asname or a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.If):  # TYPE_CHECKING blocks and version gates
            for inner in node.body:
                if isinstance(inner, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    names.add(inner.name)
                elif isinstance(inner, ast.ImportFrom):
                    names.update(a.asname or a.name for a in inner.names)
    return names


def _edges() -> list[tuple[Path, str, str, bool]]:
    """(file, target module, imported name, is it inside a function body)."""
    out: list[tuple[Path, str, str, bool]] = []
    for path in sorted(_DATAGEN.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        in_function = {
            id(node)
            for func in ast.walk(tree)
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef))
            for node in ast.walk(func)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module \
                    and (node.module == "src" or node.module.startswith("src.")):
                deferred = id(node) in in_function
                for alias in node.names:
                    out.append((path, node.module, alias.name, deferred))
    return out


class TheSurfaceStillResolvesTests(unittest.TestCase):

    def test_every_target_module_exists(self) -> None:
        missing = sorted({module for _, module, _, _ in _edges() if _module_file(module) is None})
        self.assertEqual(missing, [], "datagen imports src modules that are not on disk")

    def test_every_imported_name_exists(self) -> None:
        """⛔ THE ONE THAT CATCHES A RENAME. A module that still exists but no longer defines the
        name is the common case, and it is invisible to both pytest and mypy."""
        broken = []
        for path, module, name, deferred in _edges():
            target = _module_file(module)
            if target is None:
                continue  # reported by the test above
            if name == "*" or name in _defined_names(target):
                continue
            # A submodule import (`from src.pkg import submodule`) binds a module, not a name.
            if _module_file(f"{module}.{name}") is not None:
                continue
            where = "deferred" if deferred else "module level"
            broken.append(f"{path.relative_to(_REPO)} ({where}): {module}.{name}")
        self.assertEqual(broken, [], "datagen imports names src no longer defines")

    def test_the_scan_actually_found_the_surface(self) -> None:
        """A test that asserts over an empty set passes loudest of all."""
        edges = _edges()
        self.assertGreaterEqual(len(edges), _MIN_EDGES)
        self.assertGreaterEqual(len({module for _, module, _, _ in edges}), _MIN_MODULES)

    def test_the_guard_can_fail(self) -> None:
        """⭐ THE SELF-FAILING CONTROL. Both halves of the check must be able to reject."""
        self.assertIsNone(_module_file("src.robot.this_module_does_not_exist"))
        real = _module_file("src.robot.grasping.rl.train_ranking")
        assert real is not None
        self.assertNotIn("this_name_does_not_exist", _defined_names(real))
        self.assertIn("_group_key", _defined_names(real))


class PrivateCrossingsAreWrittenDownTests(unittest.TestCase):
    """⛔⛔ AN UNDERSCORE MEANS "NOBODY OUTSIDE USES THIS", AND HERE IT IS NOT TRUE.

    A refactor renaming any of these is doing the correct thing by every convention this repository
    states, and would break a tool it cannot see. The list is the contradiction, in writing.

    ⚠ NOTE THE TWO PRIVATE MODULES. `loop/_shadow_aggregator.py` and `safety/_ur_kinematics.py` are
    themselves underscore-prefixed, so the whole file reads as internal while another package
    depends on it.
    """

    #: MEASURED 2026-09-04. Change this list only together with the import it describes.
    EXPECTED = {
        ("src.robot.grasping.loop._shadow_aggregator", "_candidate_geometry"),
        ("src.robot.grasping.rl.train_ranking", "_group_key"),
        ("src.robot.grasping.rl.train_ranking", "_is_success"),
        ("src.robot.grasping.rl.train_ranking", "_pairwise_accuracy"),
        ("src.robot.grasping.rl.train_ranking", "_record_features"),
        ("src.robot.grasping.rl.train_ranking", "_train_pairwise_newton_irls"),
    }

    def test_the_private_crossings_are_exactly_the_ones_on_record(self) -> None:
        found = {(m, n) for _, m, n, _ in _edges() if n.startswith("_")}
        new = found - self.EXPECTED
        self.assertEqual(new, set(),
                         "a NEW private name now crosses the package boundary; add it here on "
                         "purpose, or give it a public name")
        gone = self.EXPECTED - found
        self.assertEqual(gone, set(),
                         "a recorded private crossing is gone; delete the row if that was intended")


class NoPathTricksTests(unittest.TestCase):
    """⛔⛔ THE SHAPE THIS SCAN CANNOT SEE, FENCED OFF SO IT CANNOT ARRIVE.

    Everything above reads `from src... import X`, a DOTTED import. A module that instead pushes
    a directory onto `sys.path` and then writes a bare `import sibling` is invisible to this guard,
    to a dotted-string grep, and to a prose sweep, because the module's own path never appears as a
    string anywhere.

    That is not hypothetical. On 2026-09-04 four test files reached `scripts/meshes/fetch.py` through
    `Path(...) / "scripts" / "meshes"` plus `import fetch`, and when the file moved into
    `datagen/assets/` all four broke at COLLECTION, aborting the whole pytest run. The path was
    assembled from segments, so the string `scripts/meshes/fetch.py` existed in none of them and
    three separate sweeps missed it. The same shape had already been hit once that day in the deep
    restructure.

    ⚠ THE DISCRIMINATOR IS A STRING LITERAL, and it is exact rather than clever. A safe insert names
    the repo root positionally: `sys.path.insert(0, str(Path(__file__).resolve().parents[2]))`, and
    carries no string literal at all, which is why dotted `from src...` imports keep working and
    this scan keeps seeing them. An insert that reaches INTO a package has to spell a directory,
    and spelling it is what makes a bare sibling import possible.
    """

    @staticmethod
    def _path_inserts(path: Path) -> list[str]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        found: list[str] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in ("insert", "append"):
                continue
            target = node.func.value
            if not (isinstance(target, ast.Attribute) and target.attr == "path"):
                continue
            literals = [n.value for n in ast.walk(node)
                        if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            if literals:
                found.append(f"sys.path.{node.func.attr}(... {literals} ...)")
        return found

    def test_datagen_never_pushes_a_named_directory_onto_the_path(self) -> None:
        offenders = {str(p.relative_to(_REPO)): found
                     for p in sorted(_DATAGEN.rglob("*.py"))
                     if (found := self._path_inserts(p))}
        self.assertEqual(offenders, {},
                         "a named directory on sys.path enables bare sibling imports, which this "
                         "guard cannot see and a rename breaks silently")

    def test_the_one_insert_that_exists_is_the_safe_shape(self) -> None:
        """⭐ THE POSITIVE CONTROL, and it doubles as proof the scan is looking in the right place.
        `datagen/assets/fetch.py` DOES insert on `sys.path`, deliberately and with a measured reason
        (an operator running the FILE rather than the module gets only its own directory, and a full
        fetch once died with ModuleNotFoundError after listing all 1,033 models). It names no
        directory, so it is not an offender, and a scan that found nothing at all would be looking
        somewhere else."""
        fetch = _DATAGEN / "assets" / "fetch.py"
        self.assertTrue(fetch.is_file(), f"{fetch} moved; repoint this control")
        source = fetch.read_text(encoding="utf-8")
        self.assertIn("sys.path.insert", source, "the control no longer inserts anything")
        self.assertEqual(self._path_inserts(fetch), [], "and it still names no directory")


class TheDependencyIsOneWayTests(unittest.TestCase):
    """`src` must never import `datagen`. The moment it does, this is a cycle, not a contract."""

    def test_backend_never_imports_datagen(self) -> None:
        offenders = []
        for path in (_REPO / "src").rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                imported = (
                    [node.module] if isinstance(node, ast.ImportFrom) and node.module
                    else [a.name for a in node.names] if isinstance(node, ast.Import)
                    else []
                )
                if any(m == "datagen" or m.startswith("datagen.") for m in imported):
                    offenders.append(str(path.relative_to(_REPO)))
        self.assertEqual(sorted(set(offenders)), [], "src must not import datagen")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
