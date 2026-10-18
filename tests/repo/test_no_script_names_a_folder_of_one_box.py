"""No script under ``scripts/`` opens a folder of one box by its drive letter (P0h h8).

``probe_sphere_fidelity.py`` read Isaac's configs from ``D:/isaacsim/...`` and nowhere else, on every platform. A script
reads a flag or an environment variable for where a thing lives; on Windows it may then guess the reference
workstation's folder, and such a guess is a constant whose name says so, listed here. Docstrings and comments, which
show commands as they are typed on that box, are not paths a script opens and are not read here.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
_DRIVE = re.compile(r"^[A-Za-z]:[/\\]")

#: (script, constant) pairs that hold Windows guesses, each tried only after the flag or the variable and only on Windows.
_WINDOWS_GUESSES = {
    ("grippers/bake_gripper_variant.py", "_WINDOWS_ASSET_GUESSES"),
    ("curobo/probe_sphere_fidelity.py", "_WINDOWS_ISAAC_ROOT_GUESS"),
    # Willy's own (P0h h8 on Willy's scripts): the bench's tree and calibration, and the replay bank.
    ("bench/run_bench.py", "_WINDOWS_TREE_GUESS"),
    ("bench/run_bench.py", "_WINDOWS_CALIBRATION_GUESS"),
    ("curobo/ab_planning_spheres.py", "_WINDOWS_TREE_GUESS"),
    ("curobo/ab_planning_spheres.py", "_WINDOWS_CALIBRATION_GUESS"),
    ("replay_bank/bank.py", "_WINDOWS_BANK_GUESS"),
}


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                ids.add(id(first.value))
    return ids


def drive_paths(source: str, script: str) -> list[str]:
    """The string constants in ``source`` that start with a drive letter, but docstrings and the listed guesses."""
    tree = ast.parse(source)
    skip = _docstrings(tree)
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and (script, t.id) in _WINDOWS_GUESSES for t in targets):
            skip |= {id(inner) for inner in ast.walk(node)}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip
            and _DRIVE.match(node.value)]


class NoScriptOpensAFolderOfOneBoxTests(unittest.TestCase):
    def test_every_script(self) -> None:
        for path in sorted(_SCRIPTS.rglob("*.py")):
            script = path.relative_to(_SCRIPTS).as_posix()
            with self.subTest(script):
                self.assertEqual(drive_paths(path.read_text(encoding="utf-8"), script), [])

    def test_each_listed_guess_is_still_there_and_windows_only(self) -> None:
        for script, name in sorted(_WINDOWS_GUESSES):
            with self.subTest(script):
                source = (_SCRIPTS / script).read_text(encoding="utf-8")
                self.assertIn(f"{name} = ", source)
                self.assertIn('sys.platform == "win32"', source)

    def test_the_guard_can_fail(self) -> None:
        """The line probe_sphere_fidelity.py held until P0h h8."""
        held = 'ISAAC_MP = Path("D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts")\n'
        self.assertEqual(drive_paths(held, "curobo/probe_sphere_fidelity.py"),
                         ["D:/isaacsim/isaac-sim-standalone-5.1.0-windows-x86_64/exts"])
        self.assertEqual(drive_paths('"""run D:/x/python.bat"""\n', "any.py"), [])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
