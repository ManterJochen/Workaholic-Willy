"""Every example under `scripts/examples/api/` runs to completion on a box with nothing attached.

⛔ **THIS IS THE TEST THAT DID NOT EXIST.** The old `scripts/examples/_common.py` said its spine was
kept honest by "the tests that keep them from decaying into prose". No test imported an example.
`grep -rn 'scripts/examples' tests/` found four hits and every one of them was prose in a docstring.
So twenty-nine executable files had no executor, and the two defects this replaced them over had
both been sitting in `calibration/12_two_cameras.py` unnoticed: it caught `SystemExit` where the
library had started raising `CellBuildRefused`, and its headline lesson had been reversed in the
library and never in the file.

⭐ **THE CONTRACT IS DELIBERATELY BLUNT: exit 0, no traceback, on any machine.** An example may not
require a camera, a robot, a GPU, a downloaded model or a generated corpus. Where it needs one it
checks for it and prints what is missing, because a reader who cannot run the example learns nothing
from a stack trace and everything from a sentence naming the file that is absent. That rule is what
lets this test run in CI on a machine that has none of those things, which is the only place a rot
guard is any use.

An example is not asserted to be CORRECT here. It is asserted to still exist as a program: that
every name it imports resolves, every call it makes still has that signature, and every report it
renders still renders. That is exactly the decay a rename produces and exactly what no reader
notices until they try the file.
"""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_API = _ROOT / "scripts" / "examples" / "api"

#: Long enough for a model-free example to build a cell and run a synthetic pick on this box
#: (the reference example measures 6 s), short enough that a hung example fails rather than hangs.
_TIMEOUT_S = 180


def _examples() -> list[Path]:
    return sorted(p for p in _API.rglob("*.py") if not p.name.startswith("_"))


class EveryExampleRunsTests(unittest.TestCase):
    def test_there_are_examples_to_run(self) -> None:
        """The sweep found files at all.

        Without this the suite goes green by discovering nothing, which is how a directory rename
        turns a rot guard into a no-op that still reports PASS.
        """
        found = _examples()
        self.assertGreater(len(found), 10, f"only found {[p.name for p in found]}")

    def test_every_example_runs_and_exits_zero(self) -> None:
        for path in _examples():
            with self.subTest(example=str(path.relative_to(_API)).replace("\\", "/")):
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=_ROOT, capture_output=True, text=True, timeout=_TIMEOUT_S,
                )
                output = proc.stdout + proc.stderr
                self.assertNotIn(
                    "Traceback", output,
                    f"{path.name} raised:\n{output[-2000:]}",
                )
                self.assertEqual(
                    proc.returncode, 0,
                    f"{path.name} exited {proc.returncode}:\n{output[-2000:]}",
                )

    def test_every_example_prints_something(self) -> None:
        """An example that runs and says nothing is a file nobody can learn from.

        Cheap, and it catches the failure mode where a refactor leaves the imports and the step
        comments in place while the calls that produced the output are gone.
        """
        for path in _examples():
            with self.subTest(example=path.name):
                proc = subprocess.run(
                    [sys.executable, str(path)],
                    cwd=_ROOT, capture_output=True, text=True, timeout=_TIMEOUT_S,
                )
                self.assertGreater(len(proc.stdout.strip()), 40, f"{path.name} printed nothing")


class TheShapeHoldsTests(unittest.TestCase):
    """The properties that make these examples rather than tools, checked as text.

    Not style policing. Each rule below is one of the things that made the previous generation
    unreadable, and each was measured on it: 8,492 lines across thirty files, an average of 283 per
    example, 1,563 lines of pure output calls, and a 232-line shared spine imported by all of them.
    """

    #: The reference example is 44 lines. The old generation averaged 283. A ceiling rather than a
    #: target: the point is that an example fits on a screen, not that it hits a number.
    _MAX_LINES = 70

    def test_no_example_grows_back_into_a_tool(self) -> None:
        for path in _examples():
            with self.subTest(example=path.name):
                lines = len(path.read_text(encoding="utf-8").splitlines())
                self.assertLessEqual(
                    lines, self._MAX_LINES,
                    f"{path.name} is {lines} lines; an example that needs more than "
                    f"{self._MAX_LINES} is either two examples or a check",
                )

    def test_no_example_takes_arguments(self) -> None:
        """argparse is the tell that a file has become a tool.

        A tool has options because an operator has a situation. An example has a subject, and every
        flag it grows is a fork the reader has to resolve before they can read the code.
        """
        for path in _examples():
            with self.subTest(example=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("argparse", source)
                self.assertNotIn("def main(", source)

    def test_no_example_imports_a_shared_spine(self) -> None:
        """Each file stands alone.

        The previous generation shared `_common.py`, and the cost was that reading one example meant
        reading a 232-line framework first. Twenty-nine files that read as one tool is a good
        property for a tool and a bad one for twenty-nine examples.
        """
        for path in _examples():
            with self.subTest(example=path.name):
                source = path.read_text(encoding="utf-8")
                self.assertNotIn("_common", source)

    def test_nothing_here_mutates_the_process_environment(self) -> None:
        """⛔ A MEASURED LEAK, KEPT AS A RULE AFTER ITS CAUSE WAS REMOVED.

        The previous generation took `--profile` and did `os.environ["WILLY_PROFILE"] = ...` without
        putting it back. Harmless in a script, because the process exits. These files are also
        imported and called, and one `--profile ur3e` run left the variable set: a safety-guard test
        three files away then failed inside the suite while passing in isolation, because it was
        reading a different cell's workspace box.

        The new generation cannot reproduce it, because no example takes a profile and none writes to
        the environment: `WILLY_PROFILE` is set by the operator around the command. That makes this a
        static rule rather than the old dynamic probe, which is strictly stronger. It also covers the
        checks, which is where the flag would grow back first.
        """
        roots = (_ROOT / "scripts" / "examples", _ROOT / "scripts" / "checks")
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                with self.subTest(file=str(path.relative_to(_ROOT)).replace("\\", "/")):
                    source = path.read_text(encoding="utf-8")
                    self.assertNotIn('os.environ[', source)
                    self.assertNotIn("environ.setdefault", source)
                    self.assertNotIn("putenv", source)

    def test_every_api_example_has_both_cli_twins(self) -> None:
        """The api half and the cli half are the same answer in two costumes.

        `src/contracts/README.md` states it as the repository's rule: every capability reaches an
        operator through `python -m <pkg>` and through Python. A missing twin means one of the two
        doors was never opened for that capability, which is the drift the rule exists against.
        """
        cli = _ROOT / "scripts" / "examples" / "cli"
        for path in _examples():
            stem = path.relative_to(_API).with_suffix("")
            for suffix in (".ps1", ".sh"):
                with self.subTest(example=str(stem).replace("\\", "/"), shell=suffix):
                    self.assertTrue(
                        (cli / stem).with_suffix(suffix).exists(),
                        f"no {suffix} twin for {stem}",
                    )


if __name__ == "__main__":
    unittest.main()
