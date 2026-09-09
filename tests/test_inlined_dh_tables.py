"""The UR link lengths exist three times in this repository, and nothing compared them until now.

    src/robot/safety/_ur_kinematics.py     UR_DH_TABLES_M   the one the stack plans with
    scripts/isaac/bake_ur_collision_meshes.py      UR_DH            runs under the Isaac interpreter
    scripts/curobo/build_ur_config.py              _DH_TABLES       runs under the cuRobo sidecar

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
import unittest

from src.robot.safety._ur_kinematics import UR_DH_TABLES_M

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: (file, the module-level name holding the table). Both are dicts of model -> six (a, d, alpha) rows.
_COPIES = (
    (_ROOT / "scripts" / "isaac" / "bake_ur_collision_meshes.py", "UR_DH"),
    (_ROOT / "scripts" / "curobo" / "build_ur_config.py", "_DH_TABLES"),
)

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


class EveryCopyOfTheLinkLengthsAgreesTests(unittest.TestCase):
    def test_the_copies_are_found_at_all(self) -> None:
        """The control. A reader that silently found nothing would make every test below pass over an
        empty set, which is the failure mode this file is otherwise built to catch."""
        for path, name in _COPIES:
            with self.subTest(path=path.name):
                table = _literal_table(path, name)
                self.assertGreaterEqual(len(table), 6, f"{name} holds {sorted(table)}")

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
