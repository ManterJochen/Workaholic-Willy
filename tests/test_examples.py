"""The examples: three folders, read like a library's, and none of them rots unseen.

The owner, 2026-09-18 afternoon: the examples present the API clearly, above all for real robots, keep simulation
apart, and read "so wie man sie bei einer richtigen Bibliothek aufrufen wuerde". So they live in three folders:

* `examples/real_robot/` drives a real cell. CI cannot run it, so it is held to what CI can check: it compiles, every
  name it imports from `willy` is one the door exports, and mypy (a CI step over `examples`) checks every call against
  the library's own signatures. It is never executed here.
* `examples/simulation/` drives a dummy arm, a rehearsal or Isaac Sim, and `examples/offline/` evaluates data,
  training and models at a desk with no robot and no camera attached. Both run here on a machine with nothing
  attached, exit 0 and print something; where one needs Isaac, weights, a GPU or a corpus it says which and exits.

An example is not asserted to be correct here, only to still exist as a program: that every name resolves, every call
has its signature and every report still renders. That is the decay a rename produces, and what no reader notices
until they try the file.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import willy

_ROOT = Path(__file__).resolve().parents[1]
_EXAMPLES = _ROOT / "examples"
_REAL = _EXAMPLES / "real_robot"
_SIMULATION = _EXAMPLES / "simulation"
_OFFLINE = _EXAMPLES / "offline"

#: Long enough for a desk rehearsal or a model-free evaluation, short enough that a hung example fails.
_TIMEOUT_S = 180
#: A screen. An example that needs more is two examples, or a tool.
_MAX_LINES = 70

#: What an example may import beside `willy`: the standard library and numpy.
_ALLOWED_ROOTS = frozenset(sys.stdlib_module_names) | {"numpy", "willy"}

#: Words that tie an example to one robot or one hand. The owner: runbooks and examples must not close on one robot,
#: because other robots follow and the convention starts now. The profile comes from WILLY_PROFILE instead.
_MODEL_WORDS = re.compile(
    r"\b(ur3e?|ur5e?|ur10e?|ur16e?|ur20|ur30|robotiq|hand-?e|2f-?85|2f-?140|onrobot|schunk|egu|kuka|franka|fanuc|abb)\b",
    re.IGNORECASE,
)


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*.py") if not p.name.startswith("_"))


def _all_files() -> list[Path]:
    return _files(_REAL) + _files(_SIMULATION) + _files(_OFFLINE)


def _name(path: Path) -> str:
    return path.relative_to(_EXAMPLES).as_posix()


def _bare_env() -> dict[str, str]:
    """No profile from the operator's shell: what CI measures must not depend on it."""
    env = dict(os.environ)
    env.pop("WILLY_PROFILE", None)
    return env


#: The line `logging` prints before a traceback it reports and carries on from; that one is not a crash.
_LOGGING_ERROR = "--- Logging error ---"


def _crash(output: str) -> str | None:
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            if index and lines[index - 1].strip() == _LOGGING_ERROR:
                continue
            return "\n".join(lines[index:index + 25])
    return None


class TheLayoutTests(unittest.TestCase):

    def test_the_three_folders_hold_every_example(self) -> None:
        self.assertGreaterEqual(len(_files(_REAL)), 10)
        self.assertGreaterEqual(len(_files(_SIMULATION)), 2)
        self.assertGreaterEqual(len(_files(_OFFLINE)), 10)
        stray = sorted(_name(p) for p in _EXAMPLES.rglob("*.py")
                       if not p.is_relative_to(_REAL) and not p.is_relative_to(_SIMULATION)
                       and not p.is_relative_to(_OFFLINE))
        self.assertEqual([], stray)

    def test_the_old_folder_is_gone(self) -> None:
        self.assertFalse((_ROOT / "scripts" / "examples").exists(), "scripts/examples moved to examples/")

    def test_the_readme_says_what_offline_means(self) -> None:
        text = (_EXAMPLES / "README.md").read_text(encoding="utf-8")
        for folder in ("real_robot", "simulation", "offline"):
            self.assertIn(folder, text)
        self.assertRegex(text, r"(?is)offline.{0,400}no robot")


class TheShapeTests(unittest.TestCase):
    """What makes these examples rather than tools, checked as text on every file."""

    def test_every_example_fits_on_a_screen(self) -> None:
        for path in _all_files():
            with self.subTest(_name(path)):
                lines = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(lines, _MAX_LINES, f"{lines} lines")

    def test_no_example_is_a_tool_or_touches_the_process(self) -> None:
        for path in _all_files():
            with self.subTest(_name(path)):
                source = path.read_text(encoding="utf-8")
                for word in ("argparse", "def main(", "os.environ[", "environ.setdefault", "putenv",
                             "sys.path.insert", "sys.path.append"):
                    self.assertNotIn(word, source)

    def test_no_check_touches_the_process_environment(self) -> None:
        """The checks beside the examples take one flag at most; a profile is set around the command, never inside."""
        for path in sorted((_ROOT / "scripts" / "checks").rglob("*.py")):
            with self.subTest(path.relative_to(_ROOT).as_posix()):
                source = path.read_text(encoding="utf-8")
                for word in ("os.environ[", "environ.setdefault", "putenv"):
                    self.assertNotIn(word, source)

    def test_every_example_imports_the_library_through_willy_alone(self) -> None:
        for path in _all_files():
            with self.subTest(_name(path)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        root = node.module.split(".")[0]
                    elif isinstance(node, ast.Import):
                        root = node.names[0].name.split(".")[0]
                    else:
                        continue
                    self.assertIn(root, _ALLOWED_ROOTS, f"line {node.lineno} imports {root}")

    def test_every_name_taken_from_willy_is_one_the_door_exports(self) -> None:
        exported = set(willy.__all__)
        for path in _all_files():
            with self.subTest(_name(path)):
                tree = ast.parse(path.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module == "willy":
                        missing = [alias.name for alias in node.names if alias.name not in exported]
                        self.assertEqual([], missing, f"line {node.lineno}")

    def test_no_real_robot_or_simulation_example_names_a_robot_or_a_hand(self) -> None:
        for path in _files(_REAL) + _files(_SIMULATION):
            with self.subTest(_name(path)):
                found = sorted({m.group(0) for m in _MODEL_WORDS.finditer(path.read_text(encoding="utf-8"))})
                self.assertEqual([], found, "the cell's profile names the robot; an example does not")


class RealRobotExamplesCompileTests(unittest.TestCase):
    """Never executed: they drive a real cell. mypy checks their calls in CI; here they must compile."""

    def test_every_real_robot_example_compiles(self) -> None:
        for path in _files(_REAL):
            with self.subTest(_name(path)):
                compile(path.read_text(encoding="utf-8"), str(path), "exec")

    def test_every_real_robot_example_says_how_to_run_it_at_the_cell(self) -> None:
        for path in _files(_REAL):
            with self.subTest(_name(path)):
                docstring = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8"))) or ""
                self.assertIn("WILLY_PROFILE", docstring)


class SimulationAndOfflineExamplesRunTests(unittest.TestCase):
    """Executed, on a machine with nothing attached: exit 0, no traceback, something printed."""

    def test_every_simulation_and_offline_example_runs(self) -> None:
        for path in _files(_SIMULATION) + _files(_OFFLINE):
            with self.subTest(_name(path)):
                proc = subprocess.run([sys.executable, str(path)], cwd=_ROOT, capture_output=True, text=True,
                                      timeout=_TIMEOUT_S, env=_bare_env())
                output = proc.stdout + proc.stderr
                self.assertIsNone(_crash(output), f"{_name(path)} raised:\n{_crash(output)}")
                self.assertEqual(0, proc.returncode, f"{_name(path)} exited {proc.returncode}:\n{output[-1500:]}")
                self.assertTrue(proc.stdout.strip(), f"{_name(path)} printed nothing")


if __name__ == "__main__":
    unittest.main()
