"""Four seams the carry-forward audit found: each one built, each one reaching nobody.

None of these is a crash. Every one is a thing the code appears to do and does not, which is the
class of defect this repository keeps paying for.

1. `load_set_generator` passed `weights_only=False` while every other reader of the same file passes
   True, so the one reader a customer's cell uses handed the whole file to `pickle`.
2. `propose` PRINTED "from the artifact's trained mixture" while printing a class default, because
   the reader strips `shares` out of the payload on the way in.
3. `grasping.deep_generator.minimum_score` is a live config key that decides nothing for the set
   family, and nothing said so.
4. `preload()`, the only check that an artifact's weights FIT its config, had no production caller
   at all. It exists for "a caller that can still refuse" and no caller ever called it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

_ARTIFACT = Path("src/robot/grasping/deep/set_artifact.py")
_PROPOSE = Path("src/robot/grasping/deep/eval/propose.py")
_CALCULATOR = Path("src/robot/grasping/deep/calculator.py")
_CELLS = Path("src/robot/execution/autonomous_grasp/cells.py")


class TheLoaderIsAsSafeAsTheOthersTests(unittest.TestCase):

    def test_every_torch_load_in_the_deep_package_is_weights_only(self) -> None:
        """⛔ One reader out of six passed False, and it was the one a customer's cell reaches.

        ⚠ THE RESUME READER IS EXEMPT, and it is the only exemption. A training CHECKPOINT
        legitimately carries optimiser and scheduler state, which are not plain containers. An
        ARTIFACT is a state_dict, and the package docstring says so.

        ⛔ THE EXEMPTION IS A PATH, AND IT IS ASSERTED TO EXIST. It used to be a set of bare
        FILENAMES, one of which (`export_checkpoint.py`) had been deleted with the retired family.
        A name that matches nothing looks identical to a name that matches something, so the entry
        sat in a SAFETY test as a permanent fail-open: any future file that happened to take that
        name would have skipped the check with nobody deciding it. The other entry, `set_loop.py`,
        broke the moment the file was renamed to `train/trainer.py`, which is how this was found.
        """
        exempt = {Path("src/robot/grasping/deep/train/trainer.py")}
        for path in exempt:
            self.assertTrue(path.is_file(), f"the exemption names a file that does not exist: {path}")
        offenders: list[str] = []
        for path in Path("src/robot/grasping/deep").rglob("*.py"):
            if path in exempt:
                continue
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                # ⚠ `torch.load` ONLY. The first version matched any `.load` attribute call and
                # reported `np.load` and `json.load` as offenders, which is a test finding its own
                # pattern rather than the defect.
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr == "load"
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "torch"):
                    flag = next((kw.value for kw in node.keywords if kw.arg == "weights_only"), None)
                    if not (isinstance(flag, ast.Constant) and flag.value is True):
                        offenders.append(f"{path.name}:{node.lineno}")

        self.assertEqual(offenders, [], f"torch.load without weights_only=True: {offenders}")

    def test_a_real_artifact_still_loads_under_the_stricter_flag(self) -> None:
        """⚠ THE CONTROL, and it is why the change was safe to make: a payload carrying anything
        exotic would start refusing. Skipped when no artifact is on disk."""
        from src.robot.grasping.deep.set_artifact import load_set_generator

        artifact = Path("logs/dl/models/chain_proof/set_grasp_generator_v1.pt")
        if not artifact.is_file():
            self.skipTest("no artifact on disk in this checkout")
        self.assertEqual(load_set_generator(artifact).gripper, "2f85")


class TheProvenanceLineIsHonestTests(unittest.TestCase):

    def test_propose_does_not_claim_the_mixture_came_from_the_artifact(self) -> None:
        """⛔ It did. `load_set_generator` drops `shares` from the payload, so `loaded.step.shares`
        is a class default. The number happened to be right because nothing can set a mixture from
        the CLI yet; the sentence was false either way, and would have become wrong numerically the
        first time anyone exposed the flag.

        ⚠ READ OFF THE `say(...)` CALL, NOT THE FILE. The comment that explains this defect quotes
        the sentence it replaced, so a substring search over the source passes on the defect and
        fails on the repair. That exact trap has cost four tests in one day in this repository,
        because its comments explain defects by quoting them.
        """
        tree = ast.parse(_PROPOSE.read_text(encoding="utf-8"))
        printed = " ".join(
            part.value for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "say"
            for argument in node.args
            for part in ast.walk(argument) if isinstance(part, ast.Constant)
            and isinstance(part.value, str))

        self.assertNotIn("trained mixture", printed,
                         "propose still claims a provenance the reader does not carry")
        self.assertIn("not carried through the reader", printed)

    def test_the_reader_really_does_strip_it(self) -> None:
        """The claim above, verified against the reader rather than taken from a comment."""
        self.assertIn('if k not in ("shares", "weights")', _ARTIFACT.read_text(encoding="utf-8"))


class TheInertKeyIsNamedTests(unittest.TestCase):

    def test_the_set_loader_says_minimum_score_does_nothing_here(self) -> None:
        """A live config key that decides nothing is a key an operator will tune for an afternoon."""
        source = _CALCULATOR.read_text(encoding="utf-8")

        self.assertIn("is not consulted by the set family", source)

    def test_it_was_NOT_wired_across_and_the_reason_is_a_measurement(self) -> None:
        """⛔ THE TEMPTING REPAIR WOULD HAVE BEEN HARMFUL. `decode_set_prediction` takes a
        `min_confidence`, so wiring `minimum_score` into it looks obvious. On this repository's own
        640 proposals the set head's confidences spanned 0.389 to 0.401, so the key's default of 0.5
        would have dropped every candidate and the cell would emit nothing while looking like a bad
        model.

        ⚠ THE COMMENT NO LONGER QUOTES THAT RANGE, AND MAY NOT. A confidence range measured on our
        artifact describes nothing a customer training on their own cell will see
        (`.commits/robot/08-grasping-deep.md`, and the log line is recorded as a deliberate change
        in `.migration/allow/robot.txt`). What has to survive is the argument, which holds for any
        artifact: the negation, the uncalibrated head that makes the threshold inapplicable, the
        shipped default the threshold actually is, and the consequence of applying it anyway. All
        four are pinned here, so the reason cannot quietly thin out to the bare assertion.
        """
        source = _CALCULATOR.read_text(encoding="utf-8")

        self.assertIn("It is not wired across, deliberately.", source,
                      "the negation that justifies leaving it alone is missing")
        self.assertIn("is not calibrated", source,
                      "the property that makes the threshold inapplicable is missing")
        self.assertIn("`minimum_score`'s default of 0.5", source,
                      "the threshold the argument turns on is missing")
        self.assertIn("can drop every candidate", source,
                      "the consequence of wiring it across anyway is missing")
        # ⚠ THE SLICE IS TAKEN BY AST, NOT BY THE NEXT FUNCTION'S NAME. It used to end at
        # `def _decode(`, the binned decoder, which was deleted with its family and took the slice's
        # right-hand edge with it. A boundary named after a neighbour breaks when the neighbour goes.
        import ast as _ast

        method = next(
            node for node in _ast.walk(_ast.parse(source))
            if isinstance(node, _ast.FunctionDef) and node.name == "_propose_set")
        self.assertNotIn("min_confidence", _ast.unparse(method),
                         "minimum_score was wired into the set decoder after all")


def _constants_reachable_from(tree: ast.Module, entry: str) -> list[str]:
    """Every constant in `entry` and in the module functions it can reach, transitively.

    ⛔ **THIS USED TO READ ONE FUNCTION BODY, AND A REFACTOR MOVED THE LINE IT LOOKED FOR.**
    `build_real_components` grew a `try/except` that gives a refused build its camera back, and the
    tail was lifted into a helper so the diff would not re-indent sixty lines of commentary. The
    preload call went with it, and this test failed while the behaviour it guards was untouched.

    ⭐ **AND THE ABSENCE CLAIM BELOW NEEDED IT MORE THAN THE PRESENCE CLAIM DID.** A test that says
    "the rehearsal path does NOT preload" and looks at one function stops meaning anything the day
    the rehearsal path grows a helper, and it stops meaning it SILENTLY, because absence is what a
    narrowed search returns. The presence claim fails loudly when it narrows; the absence claim does
    not. Both walk the graph now, so neither can be defeated by the same move.
    """
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    seen: set[str] = set()
    pending = [entry]
    constants: list[str] = []
    while pending:
        name = pending.pop()
        if name in seen or name not in functions:
            continue
        seen.add(name)
        body = functions[name]
        constants += [ast.unparse(n) for n in ast.walk(body) if isinstance(n, ast.Constant)]
        # Plain-name calls only: a method call belongs to whatever object it was handed, which this
        # module cannot see, and following attribute names would walk into unrelated functions that
        # happen to share a name.
        pending += [n.func.id for n in ast.walk(body)
                    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    return constants


class PreloadHasACallerTests(unittest.TestCase):

    def test_the_physical_cell_builder_preloads(self) -> None:
        """⛔ `preload` exists for "a caller that can still refuse" and had none. `_compute_result`
        catches everything, because the protocol forbids raising, so a mismatched artifact reached
        the operator as an empty candidate list at 3 a.m. instead of a refusal at build time."""
        tree = ast.parse(_CELLS.read_text(encoding="utf-8"))
        names = _constants_reachable_from(tree, "build_real_components")

        self.assertIn("'preload'", names, "the physical cell builder never asks for preload")

    def test_it_is_DUCK_TYPED_so_the_analytic_calculator_still_builds(self) -> None:
        """The analytic calculator has no `preload`, and that is the normal case rather than an
        error. A hard call would refuse every geometric cell."""
        source = _CELLS.read_text(encoding="utf-8")

        self.assertIn('getattr(calculator, "preload", None)', source)
        self.assertIn("if callable(_preload):", source)

    def test_the_rehearsal_path_does_NOT_preload(self) -> None:
        """⚠ A desk rehearsal builds a calculator it may never call, and a rehearsal that refuses
        because a weights file is missing would stop testing the wiring it exists to test."""
        tree = ast.parse(_CELLS.read_text(encoding="utf-8"))
        names = _constants_reachable_from(tree, "build_rehearsal_components")
        self.assertNotIn("'preload'", names)


if __name__ == "__main__":
    unittest.main()
