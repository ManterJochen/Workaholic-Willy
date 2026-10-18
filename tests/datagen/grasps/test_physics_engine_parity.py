"""Both physics engines are the SAME SHAPE, checked without either simulator installed.

⭐ WHY THIS EXISTS. There are two referees now. `physics_isaac` and `physics_mujoco` each define a
class called `PhysicsCell`, the engine switch binds one of them to a single name, and until the
`PhysicsCell` Protocol was written nothing said they were the same shape: a type checker saw two
unrelated classes, and an engine that grew a method or dropped one would have been caught by nobody.

⛔ IT IS PARSED, NOT IMPORTED, and that is the point. Importing `physics_isaac` needs a simulator this
machine only has on one box and CI has on none, so an import-based check would be skipped exactly
where it is needed. Reading the source with `ast` costs nothing and runs everywhere.

⚠ SHAPE IS NOT AGREEMENT. Two harnesses with identical method names can still disagree about whether
a grasp holds. That is a measurement on the same grasps, it has not been taken, and this file does not
pretend otherwise: `engine` is written into every physics row so a corpus can always say which
referee shook it.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from datagen.grasps.physics import PhysicsCell

_ENGINES = {
    "mujoco": Path("datagen/grasps/physics_mujoco.py"),
    "isaac": Path("datagen/grasps/physics_isaac.py"),
}


def _methods(path: Path) -> set[str]:
    """Every method `class PhysicsCell` defines in that file, without importing it."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "PhysicsCell":
            return {child.name for child in node.body
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))}
    raise AssertionError(f"{path} defines no class PhysicsCell")


def _required() -> set[str]:
    """What the Protocol asks for, read off the Protocol rather than copied beside it.

    ⚠ Read, not listed. A hand-written list here would be a second source of truth that drifts from
    the first, which is the shape of defect this whole file exists to catch one level down.
    """
    return {name for name in vars(PhysicsCell) if not name.startswith("_")} | {
        "__enter__", "__exit__"}


class ParityTests(unittest.TestCase):

    def test_every_engine_implements_the_whole_protocol(self) -> None:
        for engine, path in _ENGINES.items():
            with self.subTest(engine):
                missing = _required() - _methods(path)
                self.assertFalse(missing, f"{engine} is missing {sorted(missing)}")

    def test_the_two_engines_agree_on_their_PUBLIC_surface(self) -> None:
        """⛔ Both directions. A method one engine has and the other does not is a caller that works
        on one referee and fails on the other, whichever way round it is."""
        public = {engine: {m for m in _methods(path) if not m.startswith("_")}
                  for engine, path in _ENGINES.items()}
        self.assertEqual(public["mujoco"], public["isaac"],
                         f"only in mujoco: {sorted(public['mujoco'] - public['isaac'])}; "
                         f"only in isaac: {sorted(public['isaac'] - public['mujoco'])}")

    def test_the_protocol_is_not_empty(self) -> None:
        """The control. An empty Protocol would make both tests above pass and mean nothing, and this
        arc has already shipped a guard that enumerated the wrong thing and stayed green."""
        self.assertGreaterEqual(len(_required()), 8)

    def test_run_controls_is_required_of_both(self) -> None:
        """⭐ NAMED EXPLICITLY because it is the one method that makes a hold rate a statement about
        the grasp rather than about the harness. A run that returned 0 of 48 is why the controls are
        read first and recorded rather than assumed, so an engine without them is not a referee."""
        self.assertIn("run_controls", _required())
        for engine, path in _ENGINES.items():
            with self.subTest(engine):
                self.assertIn("run_controls", _methods(path))


if __name__ == "__main__":
    unittest.main()
