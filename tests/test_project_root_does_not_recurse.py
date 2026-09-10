"""The warning about a guessed project root must not need a guessed project root to be printed.

⛔⛔ **MEASURED 2026-09-10, AND IT IS A HARD CRASH IN THE CUSTOMER-INSTALL CASE.** `project_root()`
walks up from its own file looking for a directory that holds `src/` plus `requirements.txt` or
`pyproject.toml`. When nothing matches it guesses from path depth and warns, and the comment above
that warning says why: "every caller that builds a path off the root then inherits the mistake
silently. Say it once, here, with the fix."

The operator never got to read it. Building the warning goes

    project_root -> _log -> utility_logger -> create_logger -> resolve_log_dir -> logs_dir
                 -> project_root -> ...

and `logs_dir()` asks `project_root()` for the very root that has just failed to resolve. Measured on
a copy of `src/utility` placed under a directory holding `src/` and no `pyproject.toml`,
which is what a vendored or site-packages install looks like: `RecursionError: maximum recursion
depth exceeded`, 506 frames, instead of one sentence naming `WILLY_PROJECT_ROOT`.

⭐ **THE SHAPE, ONE TURN FURTHER THAN THE REST OF TODAY.** A guard that cannot fire is invisible; a
DIAGNOSTIC that cannot run takes the process down with it, and it does so exactly in the situation it
was written for. Everywhere else the failure is silence, so nobody looks; here the failure is loud
and names `RecursionError` rather than the missing environment variable.

What is pinned here is the property rather than the chain: `project_root()` may be re-entered from
anywhere inside its own fallback branch, whatever the logging stack grows to depend on later.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile
import unittest

_ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Deliberately not `sys.executable`: the child must import a COPY, and pytest's interpreter already
#: has the repository on its path. The venv interpreter with an explicit PYTHONPATH is the honest
#: reproduction of an install that is not a checkout.
_PY = sys.executable


class AVendoredInstallGetsASentenceNotARecursionErrorTests(unittest.TestCase):
    _CODE = (
        "import sys; sys.setrecursionlimit(200);"
        "from src.utility.paths import project_root;"
        "print('ROOT=' + str(project_root()))"
    )

    def _vendored(self, tmp: pathlib.Path) -> pathlib.Path:
        """A copy that holds `src/` and no build file, which is the whole precondition.

        Only `src/utility` is copied: the import chain needs nothing else, and copying the
        whole tree would take a minute and bring a `pyproject.toml` with it, which is exactly the
        file whose absence makes this case.
        """
        shutil.copytree(_ROOT / "src" / "utility", tmp / "src" / "utility")
        (tmp / "src" / "__init__.py").write_text("", encoding="utf-8")
        return tmp

    def _run(self, cwd: pathlib.Path, env: dict) -> subprocess.CompletedProcess:
        return subprocess.run([_PY, "-c", self._CODE], cwd=cwd, capture_output=True, text=True,
                              timeout=180, env=env)

    def test_no_matching_ancestor_answers_instead_of_recursing(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            tmp = self._vendored(pathlib.Path(raw))
            self.assertFalse((tmp / "pyproject.toml").exists(), "the precondition was not built")
            done = self._run(tmp, {"SYSTEMROOT": "C:/Windows", "PYTHONPATH": str(tmp), "PATH": ""})
            output = done.stdout + done.stderr

            self.assertNotIn("RecursionError", output, output[-800:])
            self.assertEqual(done.returncode, 0, output[-1200:])
            self.assertIn("ROOT=", output, output[-800:])

    def test_the_environment_variable_still_wins_and_short_circuits(self) -> None:
        """The control. `WILLY_PROJECT_ROOT` is the fix the warning names, so it must not be the
        thing that was broken: it is read before the walk and never reaches the fallback at all."""
        with tempfile.TemporaryDirectory() as raw:
            tmp = self._vendored(pathlib.Path(raw))
            chosen = tmp / "somewhere_else"
            chosen.mkdir()
            done = self._run(tmp, {"SYSTEMROOT": "C:/Windows", "PYTHONPATH": str(tmp), "PATH": "",
                                   "WILLY_PROJECT_ROOT": str(chosen)})
            output = done.stdout + done.stderr

            self.assertEqual(done.returncode, 0, output[-1200:])
            # `.resolve()`: `_from_env` resolves what it reads, and on Windows a temp directory
            # arrives here as an 8.3 short name (`TIMKAC~1`) while the resolved form is the long one.
            # Comparing the raw string made this control red for a reason that was mine, not the
            # code's, which is the one kind of red that must never be accepted as evidence.
            self.assertIn(f"ROOT={chosen.resolve()}", output, output[-800:])


class TheGuardIsAboutReEntryRatherThanAboutLoggingTests(unittest.TestCase):
    """⚠ **WHY THIS IS NOT A TEST ABOUT `_log`.** The obvious repair is to make the fallback warn
    through `sys.stderr` instead of through the logger, which fixes today's chain and leaves the next
    one open: anything `logs_dir()` comes to depend on can close the loop again. The repair is a
    re-entry guard, so this asserts on re-entry from an arbitrary caller rather than on the identity
    of the one that happened to be there.
    """

    def test_re_entering_the_fallback_returns_rather_than_recursing(self) -> None:
        from unittest import mock

        from src.utility import paths

        seen: list[int] = []

        def re_enter(*_args, **_kwargs):
            # Stands in for anything reached while the warning is being built. One level is enough:
            # the defect is that the second entry takes the same branch and calls this again.
            seen.append(1)
            if len(seen) < 30:
                paths.project_root()
            return mock.MagicMock()

        with mock.patch.object(paths, "_from_env", return_value=None), \
             mock.patch.object(paths, "_log", side_effect=re_enter), \
             mock.patch.object(paths.Path, "is_dir", return_value=False):
            root = paths.project_root()

        self.assertIsInstance(root, paths.Path)
        self.assertEqual(len(seen), 1,
                         "the warning was built more than once, so the branch re-entered itself")


if __name__ == "__main__":
    unittest.main()
