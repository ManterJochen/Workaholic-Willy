"""`robot.grasping.calculator` must actually decide which generator a cell runs.

⛔ THE DEFECT THIS PINS. The key has existed since WS0 and **nothing consumed it**. Every construction
site named the analytic class directly — `cells.py` twice, and the sim runners — so a cell could set
`grasping.calculator: deep` and silently get the analytic stack. The factory's fail-closed refusal,
written precisely for that case, could never fire because nothing called the factory.

⚠ AND THE EXISTING GUARD COULD NOT SEE IT, which its own docstring now says out loud:
`test_grasping_wiring_guard` enumerates BOOLEAN `enabled` flags, and this one is a string selector. A
selector is exactly as capable of being inert as a switch. `test_deep_calculator.FactoryTests` already
covers the factory's *behaviour* — that each value builds a different class and an unknown value
refuses. What was missing, and what this module adds, is the other half: that the production
construction sites GO THROUGH it. A perfectly-tested factory nobody calls is the defect.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest import mock

from src.config.schema.robot import RobotConfig

_ROOT = Path(__file__).resolve().parents[1]

#: ⛔⛔ **THE ONLY MODULES ALLOWED TO NAME THE ANALYTIC CLASS, EACH WITH ITS REASON.**
#:
#: ⚠ `generation/calculator.py` IS NOT HERE and must not be added: it DEFINES the class rather than
#: importing it, so the sweep never sees it. Listing it was the first thing the bidirectional check
#: caught, on its first run.
#:
#: This used to be a one-element tuple with a comment promising a sim-runner list that did not exist,
#: so seven runners plus a datagen sweep named `GraspCalculator` directly and nothing noticed. The
#: guard below now DISCOVERS every site and compares against this ledger in BOTH directions: a new
#: offender fails, and a name that stops offending must be removed from here. A list that only grows
#: is a list nobody maintains.
_ALLOWED: dict[str, str] = {
    "src/robot/grasping/calculator_factory.py":
        "the factory itself; it is what everything else goes through",
    "src/willy_sim/run_suction_pick.py":
        "A NEGATIVE CONTROL, NOT A PRODUCER. It exits 0 precisely when the JAW generator finds "
        "nothing (`thesis = (not jaw_ok) and suction_ok`). Under a learned generator that sentence "
        "would be produced by an undertrained net emitting zero candidates for an unrelated reason, "
        "and the deep calculator is contractually forbidden from raising, so 'too wide for 85 mm' "
        "and 'the model is bad' become indistinguishable. A control that passes because the "
        "instrument is broken is worse than no control.",
    "datagen/eval/ladder.py":
        "THE LADDER GRADES THE ANALYTIC GENERATOR AS A NAMED RUNG. Its deep rung already goes "
        "through `build_calculator` in `_make_deep`. Routing the analytic rung too would leave the "
        "ladder unable to compare the two.",
}

#: Where the sweep looks. Tests are excluded: a test may name any class it likes.
#:
#: ⚠ THE LIBRARY ROOT IS `src`, NOT `backend/src`. A root that does not exist is skipped in
#: silence by `_offenders`, so spelling it wrong shrinks the sweep to `datagen` and `api` and the
#: whole library stops being checked. The stale-ledger test is what notices: its two `src/...`
#: entries stop being found and report as stale.
_ROOTS = (Path("src"), Path("datagen"), Path("api"))


def _calls(path: Path) -> set[str]:
    """Every function name called in the file, however it is spelled."""
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


def _names_the_class(path: Path) -> bool:
    """Does this file USE `GraspCalculator` as a value, rather than only as a type?

    ⚠ VALUE POSITION, NOT ONLY `ast.Call`. `datagen/eval/ladder.py` has
    `factory = config.make or GraspCalculator`, which constructs the analytic generator without ever
    appearing as a call, and a call-only sweep walks straight past it.

    ⚠ AND ANNOTATIONS ARE LEGAL. Several modules take a `GraspCalculator` parameter for typing and
    never build one; flagging those would make the guard noise, and a noisy guard gets an exemption
    list that quietly grows until it means nothing.

    ⚠ EXACT NAMES ONLY. `datagen/eval/floors.py` defines `RandomGraspCalculator`,
    `TopDownGraspCalculator` and `NormalGraspCalculator`; a substring test would flag all three, and
    this repository has already shipped an `in`-instead-of-`==` hole in its licence audit.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and (node.module or "").endswith("generation.calculator"):
            bound |= {alias.asname or alias.name for alias in node.names
                      if alias.name == "GraspCalculator"}
    if not bound:
        return False
    annotations: set[int] = set()
    for node in ast.walk(tree):
        for holder in (getattr(node, "annotation", None), getattr(node, "returns", None)):
            if holder is not None:
                annotations |= {id(x) for x in ast.walk(holder)}
    return any(isinstance(node, ast.Name) and node.id in bound and id(node) not in annotations
               for node in ast.walk(tree))


def _offenders() -> set[str]:
    """Every module under the roots that names the analytic class in value position."""
    found: set[str] = set()
    for root in _ROOTS:
        directory = _ROOT / root
        if not directory.is_dir():
            continue
        for path in directory.rglob("*.py"):
            if _names_the_class(path):
                found.add(path.relative_to(_ROOT).as_posix())
    return found


class TheProductionSitesGoThroughTheFactoryTests(unittest.TestCase):
    def test_NO_UNLISTED_module_constructs_the_analytic_class_by_name(self) -> None:
        """⛔ THE DEFECT ITSELF. Naming `GraspCalculator` is what made the selector inert, and this
        now sweeps the whole tree rather than one hand-kept path."""
        offenders = sorted(_offenders() - set(_ALLOWED))
        self.assertEqual(offenders, [], "\n".join([
            "these build the analytic generator by name, so `grasping.calculator: deep` cannot reach "
            "them and the factory's fail-closed refusal never fires. Route them through "
            "`build_calculator`, or add them to _ALLOWED with the reason:", *offenders]))

    def test_the_ledger_carries_NOTHING_STALE(self) -> None:
        """⚠ BOTH DIRECTIONS. A ledger that only grows is a ledger nobody maintains: an entry whose
        module has since been routed would sit there forever, exempting a site that no longer needs
        it and hiding the day it regresses."""
        found = _offenders()
        stale = sorted(name for name in _ALLOWED
                       if name not in found and (_ROOT / name).is_file())
        self.assertEqual(stale, [], "\n".join([
            "these are exempted but no longer name the class; remove them from _ALLOWED:", *stale]))

    def test_every_exemption_states_a_REASON(self) -> None:
        """An exemption without a reason is a hole with a comment character in front of it."""
        for name, reason in _ALLOWED.items():
            with self.subTest(name):
                self.assertGreater(len(reason), 30, f"{name} is exempt without saying why")

    def test_the_routed_sites_call_the_factory(self) -> None:
        """The other half: absence of the wrong call is not presence of the right one."""
        for name in ("src/robot/execution/autonomous_grasp/cells.py",
                     "src/willy_sim/run_m1_pick.py",
                     "src/willy_sim/run_m2_pick.py",
                     "src/willy_sim/run_attribute_pick.py",
                     "src/willy_sim/run_eih_pick.py",
                     "src/willy_sim/run_multiview_pick.py",
                     "src/willy_sim/run_dense_pick.py",
                     "datagen/rl/occupancy.py"):
                with self.subTest(module=name):
                    self.assertIn("build_calculator", _calls(Path(name)))

    def test_the_sweep_can_actually_SEE_a_value_use(self) -> None:
        """⚠ A guard nobody has seen fail is a guard nobody can trust, and the value-position case is
        the one a call-only sweep misses."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offender.py"
            path.write_text(
                "from src.robot.grasping.generation.calculator import GraspCalculator\n"
                "factory = None or GraspCalculator\n", encoding="utf-8")
            self.assertTrue(_names_the_class(path))
            path.write_text(
                "from src.robot.grasping.generation.calculator import GraspCalculator\n"
                "def f(c: GraspCalculator) -> GraspCalculator: return c\n", encoding="utf-8")
            self.assertFalse(_names_the_class(path), "an annotation was flagged as a construction")

    def test_the_guard_can_actually_FAIL(self) -> None:
        """A guard nobody has seen fail is a guard nobody can trust."""
        tree = ast.parse("x = GraspCalculator(camera_matrix=None)\n")
        found = {n.func.id for n in ast.walk(tree)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertIn("GraspCalculator", found)


class TheSelectorReachesTheCellTests(unittest.TestCase):
    """Differential, at the level the defect lived: the SAME builder, two config values, two runtimes."""

    @staticmethod
    def _cfg(choice: str, **grasping) -> RobotConfig:
        return RobotConfig.model_validate(
            {"vendor": "dummy", "grasping": {"calculator": choice, **grasping}})

    def test_geometric_builds_the_analytic_generator(self) -> None:
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        calculator, _perception, _resolver, _cams, _lenses = build_rehearsal_components(
            self._cfg("geometric"))
        self.assertEqual(type(calculator).__name__, "GraspCalculator")

    def test_deep_without_an_artifact_REFUSES_rather_than_falling_back(self) -> None:
        """⛔ THE WHOLE POINT OF THE SEAM. A cell that asked for the learned generator and quietly got
        the analytic one would file the analytic one's numbers under the learned one's name."""
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        with self.assertRaises(FileNotFoundError) as caught:
            build_rehearsal_components(self._cfg("deep"))
        message = str(caught.exception)
        self.assertIn("calculator is 'deep'", message)
        self.assertIn("calculator: geometric", message, "the refusal must name the way out")

    def test_the_refusal_names_a_command_that_EXISTS(self) -> None:
        """⚠ It used to say `datagen train-generator`, which has never existed — verified against
        `datagen --help`. A fail-closed message that sends the operator to a phantom command turns a
        two-minute fix into a hunt.

        ⚠ THE SUBCOMMAND IS `train-set`, SPELLED IN FULL. `deep/__main__.py` registers `train-set`
        and no bare `train`, so asserting the prefix would pass on a message naming a command that
        does not exist. The module is `src.robot.grasping.deep`: the library root is `src`."""
        from src.robot.execution.autonomous_grasp.cells import build_rehearsal_components

        with self.assertRaises(FileNotFoundError) as caught:
            build_rehearsal_components(self._cfg("deep"))
        self.assertNotIn("datagen train-generator", str(caught.exception))
        self.assertIn("python -m src.robot.grasping.deep train-set", str(caught.exception))

    def test_an_unknown_selector_refuses_by_name(self) -> None:
        from src.robot.grasping.calculator_factory import build_calculator

        with self.assertRaises(ValueError) as caught:
            build_calculator(self._cfg("neural"), camera_matrix=None)
        self.assertIn("neural", str(caught.exception))

    def test_the_REAL_cell_builder_is_on_the_same_seam(self) -> None:
        """The rehearsal path is convenient to test; the real one is the one that drives hardware."""
        from src.robot.execution.autonomous_grasp import cells

        app_cfg = mock.Mock()
        app_cfg.camera.cameras.rigs = []
        with mock.patch("src.config.load_config", return_value=app_cfg), \
             self.assertRaises(cells.CellBuildRefused):
            cells.build_real_components(self._cfg("deep"), "an object")
        # It refuses on the CAMERA first, which is correct ordering -- no models load for a cell that
        # has no rig. The seam itself is pinned by the AST guard above.
        self.assertIn("build_calculator", _calls(
            Path("src/robot/execution/autonomous_grasp/cells.py")))


class TheJawIsCarriedIntoTheDeepPathTests(unittest.TestCase):
    """⚠ MEASURED GAP: `min/max_grip_width_mm` reached `GraspCalculator` at every call site and were
    DROPPED by the factory's deep branch, so a deep cell could command a width its gripper cannot open
    to. Invisible only because the analytic generator filters at generation and this one had no filter
    at all."""

    def test_the_config_carries_the_jaw(self) -> None:
        from src.robot.grasping.deep.calculator import DeepCalculatorConfig

        cfg = DeepCalculatorConfig(artifact_path="x", min_grip_width_mm=5.0, max_grip_width_mm=85.0)
        self.assertEqual(cfg.max_grip_width_mm, 85.0)

    def test_zero_means_NOT_TOLD_and_refuses_nothing(self) -> None:
        """A caller from before these existed must behave exactly as it did."""
        from src.robot.grasping.deep.calculator import (
            DeepCalculatorConfig,
            DeepGraspCalculator,
        )

        calc = DeepGraspCalculator(DeepCalculatorConfig(artifact_path="x"))
        for width in (0.5, 40.0, 10_000.0):
            self.assertTrue(calc._fits_the_jaw(width), width)      # noqa: SLF001 - the unit under test

    def test_a_span_wider_than_the_jaw_is_REFUSED(self) -> None:
        from src.robot.grasping.deep.calculator import (
            DeepCalculatorConfig,
            DeepGraspCalculator,
        )

        calc = DeepGraspCalculator(DeepCalculatorConfig(
            artifact_path="x", min_grip_width_mm=5.0, max_grip_width_mm=85.0))
        self.assertFalse(calc._fits_the_jaw(99.0))                 # noqa: SLF001
        self.assertFalse(calc._fits_the_jaw(1.0))                  # noqa: SLF001
        self.assertTrue(calc._fits_the_jaw(40.0))                  # noqa: SLF001

    def test_the_refusal_is_COUNTED_not_silent(self) -> None:
        """"Proposed nothing" and "proposed nothing this gripper can hold" call for opposite repairs,
        and only the second is fixed by a different gripper."""
        source = (_ROOT / "src/robot/grasping/deep/calculator.py").read_text(encoding="utf-8")
        self.assertIn("rejected_grip_width", source)

    def test_the_factory_passes_the_jaw_through(self) -> None:
        source = (_ROOT / "src/robot/grasping/calculator_factory.py").read_text(
            encoding="utf-8")
        for key in ("min_grip_width_mm", "max_grip_width_mm"):
            self.assertIn(key, source, key)


class TheSchemaDescribesWhatTheCodeDoesTests(unittest.TestCase):
    """⚠ The schema told the operator the opposite of the code: that `deep` "consumes everything the
    geometric stack produces … an ADDITION to the pipeline rather than a replacement". The factory is
    strictly either/or, and the deep decoder seeds only from its own graspability head. Prose that
    contradicts the code is worse than no prose: it is the sentence that makes a missing filter tail
    easy to under-rate."""

    def test_the_selector_is_not_described_as_an_addition(self) -> None:
        source = (_ROOT / "src/config/schema/robot/grasping_schema.py").read_text(
            encoding="utf-8")
        window = source[source.index("``deep``"):source.index("``deep``") + 1400]
        self.assertNotIn("ADDITION to the pipeline", window,
                         "the schema still claims deep ADDS to the analytic stack; the factory "
                         "returns one or the other")


if __name__ == "__main__":
    unittest.main()
