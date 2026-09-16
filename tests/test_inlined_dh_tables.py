"""The UR link lengths exist four times in this repository, and nothing compared them until now.

    src/robot/safety/_ur_kinematics.py     UR_DH_TABLES_M   the one the stack plans with
    scripts/isaac/bake_ur_collision_meshes.py      UR_DH            runs under the Isaac interpreter
    scripts/curobo/build_ur_config.py              _DH_TABLES       runs under the cuRobo sidecar
    scripts/isaac/bake_ur_meshes_from_urdf.py      UR_DH            the URDF bake and the fidelity probe

The fourth copy said in a comment that this file compared it, and for a while it did not: the list
of copies was written by hand. The copies are now also found by their shape, so a fifth one fails
here the day it is written.

⛔ **THE DUPLICATION IS DELIBERATE AND THE SILENCE WAS NOT.** Three interpreters that cannot import
each other: Isaac ships its own Python, the cuRobo sidecar is 3.10 and cannot import a module using
``enum.StrEnum``, and the project venv is neither. Both scripts say so where they inline the table,
and both are right to. What was missing is anything that notices when one copy drifts.

⚠ **WHAT DRIFT WOULD DO.** These numbers place collision geometry. A wrong ``d`` in the bake writes a
bundle whose links sit somewhere the arm is not, and the guard then clears poses that collide, at a
distance equal to the error. Nothing raises: the bundle loads, the shapes are the right shapes, and
the cell reports a healthy exact-mesh backend. The 0.05 mm between a UR3 and a UR3e shoulder is the
scale at which a typo stops looking like one.

The tables are read as TEXT and parsed with ast, not imported, because two of the three files cannot
be imported by the interpreter running this test. That is the same constraint that created the
duplication, so the check has to live with it rather than wish it away.
"""

from __future__ import annotations

import ast
import pathlib
import re
import unittest
from collections.abc import Iterator

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (file, the name holding the table). Each is a dict of model -> six (a, d, alpha) rows.
_COPIES = (
    (_ROOT / "scripts" / "isaac" / "bake_ur_collision_meshes.py", "UR_DH"),
    (_ROOT / "scripts" / "curobo" / "build_ur_config.py", "_DH_TABLES"),
    (_ROOT / "scripts" / "isaac" / "bake_ur_meshes_from_urdf.py", "UR_DH"),
)

#: Where an inlined copy could live. The authority itself holds DH row objects, not literals, so the
#: scan below cannot mistake it for a copy.
_SCANNED = (_ROOT / "scripts", _ROOT / "src")

#: A UR model key as the tables spell it: ur3, ur5e, ur16e, and ur8long once it exists.
_MODEL_KEY = re.compile(r"ur\d+(?:e|long)?\Z")

#: How close two spellings of the same link length have to be. They are literals in both places, so
#: the honest tolerance is float noise and not an engineering margin.
_TOLERANCE_M = 1e-12


def _literal_table(path: pathlib.Path, name: str) -> dict[str, list[tuple[float, float, float]]]:
    """The named table, read out of the file as a literal.

    Walks every assignment rather than only module-level ones: the cuRobo copy lives inside a
    function-shaped branch, because it may only be evaluated where cuRobo and trimesh import.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if name not in targets:
            continue
        table = ast.literal_eval(node.value)
        return {k: [tuple(float(x) for x in row) for row in rows] for k, rows in table.items()}
    raise AssertionError(f"no assignment to {name} in {path}")


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


def _inlined_tables() -> Iterator[tuple[pathlib.Path, str]]:
    """Every assignment of a literal DH table under the scanned trees, found by shape, not by name."""
    for root in _SCANNED:
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


class EveryCopyOfTheLinkLengthsAgreesTests(unittest.TestCase):
    def test_the_copies_are_found_at_all(self) -> None:
        """The control. A reader that silently found nothing would make every test below pass over an
        empty set, which is the failure mode this file is otherwise built to catch."""
        for path, name in _COPIES:
            with self.subTest(path=path.name):
                table = _literal_table(path, name)
                self.assertGreaterEqual(len(table), 6, f"{name} holds {sorted(table)}")

    def test_the_scan_sees_every_listed_copy(self) -> None:
        """The control for the scan below: a shape test that matched nothing would find no unlisted
        copy anywhere, and say so as if the repository were clean."""
        found = {(p.resolve(), n) for p, n in _inlined_tables()}
        for path, name in _COPIES:
            with self.subTest(copy=path.name):
                self.assertIn((path.resolve(), name), found)

    def test_every_inlined_table_in_the_repository_is_compared(self) -> None:
        """The list above is written by hand, and a hand list is a claim about absence that only grows
        stale. The URDF bake inlined a fourth copy, said in a comment that this file compares it, and
        this file did not. So the copies are found by their shape and each must be on the list."""
        listed = {(p.resolve(), n) for p, n in _COPIES}
        unlisted = sorted(f"{p.relative_to(_ROOT).as_posix()}:{n}"
                          for p, n in {(p.resolve(), n) for p, n in _inlined_tables()} - listed)
        self.assertEqual(unlisted, [], "inlined UR link lengths that nothing compares against the authority")

    def test_every_inlined_row_matches_the_authority(self) -> None:
        for path, name in _COPIES:
            table = _literal_table(path, name)
            for model, rows in table.items():
                with self.subTest(copy=path.name, model=model):
                    self.assertIn(model, UR_DH_TABLES_M,
                                  f"{path.name} inlines a model _ur_kinematics does not have")
                    authority = UR_DH_TABLES_M[model]
                    self.assertEqual(len(rows), len(authority))
                    for i, (row, ref) in enumerate(zip(rows, authority)):
                        for value, expected, field in zip(row, (ref.a_m, ref.d_m, ref.alpha_rad),
                                                          ("a_m", "d_m", "alpha_rad")):
                            self.assertAlmostEqual(
                                value, expected, delta=_TOLERANCE_M,
                                msg=f"{path.name} {model} row {i} {field}: {value} vs {expected}. "
                                    f"This places collision geometry, so a copy that drifts clears "
                                    f"poses that collide by exactly this much, and says ok.",
                            )

    def test_the_comparison_can_fail(self) -> None:
        """⚠ THE SELF-FAILURE CONTROL. Perturb one number and the comparison must reject it, or the
        test above is asserting that floats equal themselves."""
        table = _literal_table(*_COPIES[0])
        model = next(iter(table))
        perturbed = table[model][0][1] + 1e-6  # a micrometre, far below anything an eye would catch
        self.assertNotAlmostEqual(perturbed, UR_DH_TABLES_M[model][0].d_m, delta=_TOLERANCE_M)

    def test_every_copy_covers_every_model_that_already_has_a_bundle(self) -> None:
        """B5. A committed collision bundle that no inlined table can place is a bundle nobody can rebuild.

        The bundle is baked in the arm's DH frames and read back out of them by the descriptor builder, so a model
        whose row is missing from either copy is geometry with no chain to hang it on. This is the assertion that
        turns red the moment a new arm's bundle lands without its row, which is exactly when it can still be fixed
        cheaply.
        """
        data = _ROOT / "src" / "robot" / "safety" / "data"
        bundled = sorted(
            path.name.split("_collision_meshes")[0] for path in data.glob("*_collision_meshes.npz")
        )
        arms = sorted(name for name in bundled if name in UR_DH_TABLES_M)
        self.assertTrue(arms, "no arm bundle was found, so this test would pass by having nothing to check")
        for path, name in _COPIES:
            with self.subTest(copy=path.name):
                self.assertEqual(sorted(set(arms) - set(_literal_table(path, name))), [],
                                 f"{path.name} cannot place a bundle this repository ships")

    def test_the_scripts_cover_every_model_the_stack_can_configure(self) -> None:
        """A model the stack can be configured for and a script cannot bake is a model whose bundle
        or descriptor simply never gets built, with the script reporting an unsupported model rather
        than anything about geometry."""
        from src.config.schema.robot._ur_models import UR_MODEL_KEYS

        for path, name in _COPIES:
            with self.subTest(copy=path.name):
                missing = sorted(set(UR_MODEL_KEYS) - set(_literal_table(path, name)))
                self.assertEqual(missing, [], f"{path.name} cannot build these")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
