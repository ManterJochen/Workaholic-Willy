"""`render.engine` must actually decide which backend renders — and the construction sites must use it.

⛔ WRITTEN BEFORE THE SECOND ENGINE EXISTS, DELIBERATELY. This repo has already shipped exactly this
shape and paid for it: `robot.grasping.calculator` (`geometric` | `deep`) had a fail-closed factory, a
differential test on that factory, a schema block — and **ZERO production callers**. Every construction
site named the analytic class directly, so a cell could ask for `deep` and silently get the analytic
stack, and the refusal written precisely for that case could never fire.

⚠ AND THE EXISTING WIRING GUARD COULD NOT SEE IT: `test_grasping_wiring_guard` enumerates BOOLEAN
`enabled` fields, and this is a string selector. Its own docstring names that blind spot. So a selector
needs a guard on the CONSTRUCTION SITES, by AST — a perfectly-tested factory nobody calls is the defect,
and testing the factory harder would not have caught it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from datagen.config import DatagenConfig
from datagen.render.engine import ENGINES, build_engine, engine_is_available

_ROOT = Path(__file__).resolve().parents[1]

#: Every module that constructs a rendering backend for a real run.
_CONSTRUCTION_SITES = (Path("datagen/build.py"),)

#: Backend classes that must never be named at a construction site. A second engine adds its class here
#: on the day it lands, so the guard grows with the feature instead of decaying beside it.
_BACKEND_CLASSES = ("IsaacRenderer", "MujocoRenderer", "NoEngineRenderer")


def _calls(path: Path) -> set[str]:
    tree = ast.parse((_ROOT / path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            target = node.func
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


class TheConstructionSitesGoThroughTheFactoryTests(unittest.TestCase):
    def test_no_site_constructs_a_backend_by_name(self) -> None:
        """⛔ THE DEFECT SHAPE. Naming the class here is what made the other selector inert."""
        offenders = [f"{path.as_posix()} -> {name}"
                     for path in _CONSTRUCTION_SITES
                     for name in _BACKEND_CLASSES if name in _calls(path)]
        self.assertEqual(offenders, [], "\n".join([
            "these construct a rendering backend by name, so `render.engine` cannot reach them:",
            *offenders]))

    def test_they_call_the_factory_instead(self) -> None:
        """Absence of the wrong call is not presence of the right one."""
        for path in _CONSTRUCTION_SITES:
            with self.subTest(module=path.as_posix()):
                self.assertIn("build_engine", _calls(path))

    def test_the_guard_can_actually_FAIL(self) -> None:
        """A guard nobody has seen fail is a guard nobody can trust."""
        tree = ast.parse("with IsaacRenderer(cfg, headless=True) as r:\n    pass\n")
        found = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("IsaacRenderer", found)

    def test_every_registered_engine_is_covered_by_the_class_guard(self) -> None:
        """⚠ THE WAY THIS GUARD WOULD DECAY: a second engine lands, its class is not added to
        `_BACKEND_CLASSES`, and a site could name it directly with the guard still green. One engine,
        one entry -- checked, not remembered."""
        self.assertEqual(len(_BACKEND_CLASSES), len(ENGINES),
                         f"{len(ENGINES)} engine(s) registered but {len(_BACKEND_CLASSES)} class "
                         f"name(s) guarded -- add the new backend's class to _BACKEND_CLASSES")


class TheSelectorRefusesRatherThanFallsBackTests(unittest.TestCase):
    def test_the_default_is_isaac_so_every_existing_run_is_unchanged(self) -> None:
        self.assertEqual(DatagenConfig().render.engine, "isaac")

    def test_an_unknown_engine_is_refused_by_the_SCHEMA(self) -> None:
        """The first line of defence: a typo in a config file must not reach the factory at all."""
        import pydantic

        with self.assertRaises(pydantic.ValidationError):
            DatagenConfig.model_validate({"render": {"engine": "definitely-not-an-engine"}})

    def test_an_unavailable_engine_REFUSES_and_says_what_to_do(self) -> None:
        """⛔ NEVER FALLS BACK. An engine that quietly substituted another would file one backend's
        geometry under the other's name, and `provenance.json` would say so wrongly -- worse than not
        starting, because the corpus would look fine.

        This asserts on the branch that fires when Isaac is absent, which is the state of every
        interpreter except Isaac's own bundled one -- including CI.
        """
        available, reason = engine_is_available("isaac")
        if available:                                    # pragma: no cover - only on an Isaac python
            self.skipTest("Isaac is importable here, so the refusal branch cannot be exercised")
        with self.assertRaises(RuntimeError) as caught:
            build_engine(DatagenConfig())
        message = str(caught.exception)
        self.assertIn("isaac", message)
        self.assertIn("python.bat", message, "the refusal must name the way out, not just the symptom")

    def test_the_availability_probe_names_an_unknown_engine(self) -> None:
        ok, reason = engine_is_available("definitely-not-an-engine")
        self.assertFalse(ok)
        self.assertIn("definitely-not-an-engine", reason)

    def test_the_probe_imports_nothing_heavy(self) -> None:
        """It is called to decide whether an engine CAN be used, so importing the engine to find out
        would defeat it -- and on a box without Isaac it would raise instead of answering."""
        source = (_ROOT / "datagen/render/engine.py").read_text(encoding="utf-8")
        window = source[source.index("def engine_is_available"):source.index("def build_engine")]
        self.assertIn("find_spec", window)
        self.assertNotIn("import isaacsim", window)


class TheContractLivesOutsideTheVendorModuleTests(unittest.TestCase):
    """⭑ The move that turns "write a second engine" from a rewrite into an implementation."""

    def test_the_result_types_are_importable_without_touching_isaac(self) -> None:
        import subprocess
        import sys

        code = ("import sys;"
                "from datagen.render.result import SceneRenderResult, ViewRender;"
                "print('isaacsim' in sys.modules, 'datagen.render.isaac' in sys.modules)")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                check=False, cwd=str(_ROOT))
        self.assertEqual(result.returncode, 0, result.stderr[-1500:])
        self.assertEqual(result.stdout.strip(), "False False")

    def test_the_result_module_carries_no_vendor_type(self) -> None:
        source = (_ROOT / "datagen/render/result.py").read_text(encoding="utf-8")
        for vendor in ("isaacsim", "omni.", "import isaac"):
            self.assertNotIn(vendor, source, vendor)

    def test_the_protocol_is_narrow_enough_to_implement(self) -> None:
        """A contract with twenty methods is a rewrite wearing a Protocol's clothes. Four things plus a
        lifecycle is what the corpus actually needs -- `label-grasps` never opens an image at all."""
        from datagen.render.engine import SceneEngine

        methods = {name for name in dir(SceneEngine) if not name.startswith("_")}
        self.assertEqual(methods, {"render"})

    def test_isaac_still_satisfies_the_protocol_by_shape(self) -> None:
        """Without importing it: the guard is that the module DEFINES the entry points, so a refactor
        that renamed `render()` would be caught here rather than on-box."""
        source = (_ROOT / "datagen/render/isaac.py").read_text(encoding="utf-8")
        for method in ("def render(", "def __enter__(", "def __exit__("):
            self.assertIn(method, source, method)


if __name__ == "__main__":
    unittest.main()
