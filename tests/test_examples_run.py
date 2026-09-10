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

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_API = _ROOT / "scripts" / "examples" / "api"
_CLI = _ROOT / "scripts" / "examples" / "cli"
_DATA = _ROOT / "config"

#: Long enough for a model-free example to build a cell and run a synthetic pick on this box
#: (the reference example measures 6 s), short enough that a hung example fails rather than hangs.
_TIMEOUT_S = 180


def _examples() -> list[Path]:
    return sorted(p for p in _API.rglob("*.py") if not p.name.startswith("_"))


#: The line `logging` prints immediately before the traceback it is about to REPORT rather than
#: raise. It is the only thing that separates the two, and they need separating.
_LOGGING_ERROR = "--- Logging error ---"


def _unhandled_traceback(output: str) -> str | None:
    """The first traceback in `output` that killed the program, or ``None``.

    ⚠ **MEASURED 2026-09-10: NOT EVERY TRACEBACK ON stderr IS A CRASH.** A
    `logging.handlers.RotatingFileHandler` printed a full traceback and the program carried on to
    exit 0, because the rollover renamed a log file a second process on the same checkout held
    open and Windows refused the rename. `logging` catches that, reports it under
    `--- Logging error ---`, and keeps going.

    "No traceback" in this file means the reader met a stack trace instead of a sentence. A log
    handler that complains and continues did not do that, and counting it as one would make the
    contract fail for a reason no reader of the example would ever see.
    """
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if line.startswith("Traceback (most recent call last)"):
            if index and lines[index - 1].strip() == _LOGGING_ERROR:
                continue
            return "\n".join(lines[index:index + 25])
    return None


def _bare_env() -> dict[str, str]:
    """The environment a bare box hands an example.

    ``WILLY_PROFILE`` is dropped for the reason `TheShapeHoldsTests` keeps as a rule: the profile is
    set by the operator AROUND the command, so the operator's shell must not decide what this suite
    measures. ``sys.executable``'s directory goes on the front of PATH because the cli half spells
    the interpreter ``python``, and on a box with several of those that word does not otherwise mean
    this one.
    """
    env = dict(os.environ)
    env.pop("WILLY_PROFILE", None)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    return env



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
                # ⛔ THROUGH `_unhandled_traceback`, AND FOR AN HOUR THIS LINE WAS A RAW GREP while
                # the careful version sat fifty lines above it. MEASURED 2026-09-10 in this tree: a
                # `RotatingFileHandler` could not rename `logs/robot/robot.log` because a parallel
                # example runner held it, printed a full traceback under `--- Logging error ---`, and
                # the example carried on to exit 0 with its three candidates. `assertNotIn` reported
                # that as the example raising. The repair for that had been written the same day, in
                # this file, and applied only to the three hostile-tree sweeps that prompted it: two
                # checks for one question, one of them blind.
                crash = _unhandled_traceback(output)
                self.assertIsNone(crash, f"{path.name} raised:\n{crash}")
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


class TheContractHoldsOnAHostileTreeTests(unittest.TestCase):
    """The same contract, on the boxes `EveryExampleRunsTests` cannot reach from this checkout.

    That class runs every example against THIS repository: a config tree with a `robot` block in it,
    a working directory it may write, and YAML that parses. "Any machine" is the wider claim, and
    the three ways it was measured to break on 2026-09-10 are the three tests below. Each was red
    before the examples were changed, and each asserts a count first, because a hostile-box sweep
    that stops finding its subjects is a rot guard turning into a no-op that still reports PASS.

    ⛔ **THE THIRD ONE WAS FOUND BY THE FIRST TWO BEING FIXED.** Repairing the missing-`robot`-block
    case put a `ConfigError` handler on every example that calls `load_robot_config`, and that looked
    like the whole job. Five examples reach the tree through `load_config` instead, and a config
    error is a config error whichever loader raised it: on a tree with one malformed line they still
    ended in a `yaml` traceback while all four checks next door exited 2 with a sentence. A repair
    written from the case that prompted it, missing the property that mattered.
    """

    #: A floor rather than an expectation: the point is that the sweep still reaches a body of
    #: examples, not that it reaches exactly the twenty-four that exist today.
    _MIN_EXAMPLES = 10

    #: Run one example against a DIFFERENT config tree. `load_robot_config()` inside an example takes
    #: no `data_dir` (that is the point of it: an example must not carry a plumbing argument), so the
    #: only door to another tree is the loader's own default, and this is the one place in the
    #: repository that reaches through it. A test may; nothing shipped does.
    _LAUNCHER = (
        "import pathlib, runpy, sys;"
        "root, tree, target = sys.argv[1], sys.argv[2], sys.argv[3];"
        "sys.path.insert(0, root);"
        "import src.config.loader as loader;"
        "loader._DEFAULT_DATA_DIR = pathlib.Path(tree);"
        "sys.argv = [target];"
        "runpy.run_path(target, run_name='__main__')"
    )

    def test_every_example_runs_on_a_tree_with_no_robot_block(self) -> None:
        """⛔ **`robot` IS OPTIONAL ON `AppConfig`, SO A TREE WITHOUT ONE IS AN ANSWER.**

        `load_robot_config` says so itself: ``robot: RobotConfig | None = None`` makes a tree with no
        cell in it a configuration answer rather than a crash, and it raises `ConfigError` to say so.
        The four checks under `scripts/checks/` catch exactly that and exit 2 with a sentence.
        MEASURED 2026-09-10 in this tree: eight examples did not, and a reader who has not written
        their `robot` block yet is precisely the reader an example is for.

        The tree here is a copy of the shipped one with `robot/` removed, which also removes every
        `robot.*.yaml` overlay, so an example that names a profile meets the refusal from the other
        side. `ConfigError` is one class for both.
        """
        found = _examples()
        self.assertGreater(len(found), self._MIN_EXAMPLES, f"only found {len(found)}")
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "data"
            shutil.copytree(_DATA, tree)
            shutil.rmtree(tree / "robot")
            for path in found:
                with self.subTest(example=str(path.relative_to(_API)).replace("\\", "/")):
                    proc = subprocess.run(
                        [sys.executable, "-c", self._LAUNCHER, str(_ROOT), str(tree), str(path)],
                        cwd=_ROOT, capture_output=True, text=True, timeout=_TIMEOUT_S,
                        env=_bare_env(),
                    )
                    output = proc.stdout + proc.stderr
                    crash = _unhandled_traceback(output)
                    self.assertIsNone(crash, f"{path.name} raised:\n{crash}")
                    self.assertEqual(proc.returncode, 0,
                                     f"{path.name} exited {proc.returncode}:\n{output[-1500:]}")

    def test_every_example_runs_on_a_tree_whose_yaml_is_malformed(self) -> None:
        """⛔ **THE STATE AN OPERATOR IS IN FIVE SECONDS AFTER A BAD EDIT.**

        One line of `robot/robot.yaml` is given a wrong indent and an unclosed list, which is what a
        human produces daily and what every one of these examples exists to be run beside. MEASURED
        2026-09-10 in this tree, after the other two hostile trees were repaired: five of the
        twenty-four still ended in a `yaml.scanner.ScannerError` traceback, and all four
        `scripts/checks/` exited 2 with a sentence against the same tree.

        The five are the ones that read the tree through `load_config` rather than
        `load_robot_config`: `02_calibration/calibrate_fixed_camera`, `calibrate_wrist_camera`,
        `03_perception/read_depth_rig`, `resolve_perception_stack`, `route_hard_prompts`. Nothing
        distinguishes them to a reader; they were simply outside the shape of the earlier repair.

        The corruption is applied to the BASE robot file rather than to a profile overlay, so it is
        reached whether or not an example selects a chain.
        """
        found = _examples()
        self.assertGreater(len(found), self._MIN_EXAMPLES, f"only found {len(found)}")
        with tempfile.TemporaryDirectory() as tmp:
            tree = Path(tmp) / "data"
            shutil.copytree(_DATA, tree)
            robot = tree / "robot" / "robot.yaml"
            lines = robot.read_text(encoding="utf-8").splitlines()
            for index, line in enumerate(lines):
                if line.strip().startswith("vendor:"):
                    lines[index] = line + "\n      this_line_is_not_valid_yaml: [unclosed"
                    break
            else:  # pragma: no cover - the shipped tree has always had one
                self.skipTest("no `vendor:` line to corrupt in the shipped robot.yaml")
            robot.write_text("\n".join(lines), encoding="utf-8")

            for path in found:
                with self.subTest(example=str(path.relative_to(_API)).replace("\\", "/")):
                    proc = subprocess.run(
                        [sys.executable, "-c", self._LAUNCHER, str(_ROOT), str(tree), str(path)],
                        cwd=_ROOT, capture_output=True, text=True, timeout=_TIMEOUT_S,
                        env=_bare_env(),
                    )
                    output = proc.stdout + proc.stderr
                    crash = _unhandled_traceback(output)
                    self.assertIsNone(crash, f"{path.name} raised:\n{crash}")
                    self.assertEqual(proc.returncode, 0,
                                     f"{path.name} exited {proc.returncode}:\n{output[-1500:]}")

    def test_every_example_runs_when_the_working_directory_cannot_be_written(self) -> None:
        """⛔ **FOUR EXAMPLES WROTE INTO THE WORKING DIRECTORY WITHOUT ASKING WHETHER THEY COULD.**

        `logs/examples/...` is where the datagen and training examples put what they make, and each
        of them called `mkdir` straight at it. MEASURED 2026-09-10 in this tree: four of the
        twenty-four ended in a `FileNotFoundError` out of `os.mkdir` rather than in a sentence.

        `logs` and `assets` exist here as FILES, which is a portable way to make exactly those paths
        unwritable: the OS refuses the mkdir the way a read-only mount or a full disk does, on every
        platform, needing no privileges and leaving nothing behind.
        """
        found = _examples()
        self.assertGreater(len(found), self._MIN_EXAMPLES, f"only found {len(found)}")
        with tempfile.TemporaryDirectory() as tmp:
            cwd = Path(tmp)
            (cwd / "logs").write_text("not a directory", encoding="utf-8")
            (cwd / "assets").write_text("not a directory", encoding="utf-8")
            for path in found:
                with self.subTest(example=str(path.relative_to(_API)).replace("\\", "/")):
                    proc = subprocess.run(
                        [sys.executable, str(path)], cwd=cwd, capture_output=True, text=True,
                        timeout=_TIMEOUT_S, env=_bare_env(),
                    )
                    output = proc.stdout + proc.stderr
                    crash = _unhandled_traceback(output)
                    self.assertIsNone(crash, f"{path.name} raised:\n{crash}")
                    self.assertEqual(proc.returncode, 0,
                                     f"{path.name} exited {proc.returncode}:\n{output[-1500:]}")


#: What each cli subject needs beyond a bare box, and therefore why it is not executed here. One
#: measured REASON per subject rather than a category, because a category is how a list like this
#: stops being read. Every line was measured in THIS tree on 2026-09-10 and says what the
#: measurement was.
#:
#: The api half has no equivalent list and cannot have one: an api example that needed any of this
#: would be breaking the contract `EveryExampleRunsTests` holds it to. A CLI is allowed to need a
#: corpus, a network or a GPU, which is why the two halves are measured separately rather than one
#: being taken as evidence for the other.
_CLI_NEEDS_MORE_THAN_A_BARE_BOX: dict[str, str] = {
    "01_first_cell/planner_or_ik":
        "runs `safety.planning --doctor`, whose exit code is a property of the HOST rather than of "
        "the example: it answers out of the sidecar cuRobo env under ext_deps/ and the baked mesh "
        "bundles, measured 0 here while neither curobo nor coal imports in this checkout's venv",
    "04_safety/gate_the_whole_path":
        "carries no set -e, so its exit code is its last line alone, `safety.planning --check`, "
        "which is the host's answer and not the example's: measured 0 here, out of the same sidecar "
        "env",
    "04_safety/self_collision_backend":
        "asks that same CLI three times and ends on `--model ur10 --check`; the same host-dependent "
        "exit code, measured 0 here",
    "06_datagen/bring_your_own_parts":
        "step 1 is `datagen.assets --check`, which reads the mesh library: measured 1 against an "
        "empty one, and 0 here only because assets/meshes is a link into a shared 1,475-mesh bank "
        "that is deliberately not vendored",
    "06_datagen/choose_an_engine":
        "`datagen build --engine none` renders a corpus into logs/examples and leaves it on disk, "
        "which is what the two subjects below read; measured 2 s here, where that corpus was "
        "already present, and a render on a bare box is minutes",
    "06_datagen/extract_a_corpus":
        "`datagen build-cloud-corpus` reads that corpus and extracts two more from it "
        "(measured 45 s here)",
    "06_datagen/label_and_shake":
        "reads that same corpus, and `why-no-jaw` needs a fetched mesh (measured 6 s here)",
    "06_datagen/plan_scenes":
        "`init-config` refuses to overwrite logs/examples/tipped.json, so a second run in one tree "
        "exits 2, which is correct of the CLI and not something a rot guard can re-run",
    "07_training/import_a_public_corpus":
        "`deep import-foreign` fetches from the network",
    "07_training/read_a_training_run":
        "`deep report --run` reads a trained run directory, which nothing on a bare box has "
        "produced",
    "07_training/train_on_your_own_meshes":
        "builds a corpus with `datagen build --name my_parts` and trains on it with "
        "`deep train-set`",
    "08_sim/record_a_pick":
        "`willy_sim.run_eih_demo` drives Isaac Sim, which is not importable in a plain interpreter "
        "(measured here: IsaacNotAvailableError)",
    "08_sim/sim_pick_rate":
        "`willy_sim.run_m1_pick` drives Isaac Sim, the same",
}


class EveryCliExampleRunsTests(unittest.TestCase):
    """⛔ **NOTHING RAN THE CLI HALF.** Twenty-four subjects, forty-eight shell files, no executor.

    `EveryExampleRunsTests` has covered the api half since the day it was written, and the twin rule
    in `TheShapeHoldsTests` only checks that the cli files EXIST. A file that exists and no longer
    runs is exactly the decay a rename produces, and it had already happened: MEASURED 2026-09-10 in
    this tree, three of the twenty-four subjects exited 1 on this box, each by inheriting the exit
    code of a check it called instead of reporting what the check said.

    The subjects that need more than a bare box are named in `_CLI_NEEDS_MORE_THAN_A_BARE_BOX` with
    their reason, and the tests below hold that list to naming real subjects and to leaving something
    behind to run. A skip list nobody checks is how a suite ends up executing two files and reporting
    a sweep.
    """

    def _subjects(self) -> list[str]:
        return sorted(str(p.relative_to(_API).with_suffix("")).replace("\\", "/")
                      for p in _examples())

    def test_the_skip_list_names_subjects_that_exist(self) -> None:
        """A skipped name that no longer matches a subject stops skipping and starts lying.

        Renaming a subject would otherwise leave its reason behind pointing at nothing, while the
        renamed file quietly joins the executed set or does not, and either way the list would still
        read as deliberate.
        """
        unknown = sorted(set(_CLI_NEEDS_MORE_THAN_A_BARE_BOX) - set(self._subjects()))
        self.assertEqual(unknown, [], f"skip list names subjects that do not exist: {unknown}")

    def test_every_skip_reason_quotes_a_command_the_subject_runs(self) -> None:
        """A reason gets no edge from prose. Every one must quote a line out of the file it excuses.

        ⛔ **MEASURED 2026-09-10: TWO OF THE THIRTEEN REASONS DESCRIBED A DIFFERENT FILE.**
        `04_safety/gate_the_whole_path` was excused as ending "on that same doctor" and ends on
        `safety.planning --check`; `04_safety/self_collision_backend` was excused as asking "the
        doctor twice" and asks it once, between two `--check` runs. Both were written from memory of
        the neighbouring subject, both read as deliberate, and no rule could tell.

        So a reason must name its subject in a span this test can look for, and the span has to still
        be in the file. That is the whole rule: it costs one pair of backticks per entry and it turns
        a renamed flag from a sentence nobody re-reads into a red subtest.
        """
        for subject, reason in sorted(_CLI_NEEDS_MORE_THAN_A_BARE_BOX.items()):
            with self.subTest(example=subject):
                quoted = re.findall(r"`([^`]+)`", reason)
                self.assertTrue(quoted, f"{subject} is skipped for a reason that quotes nothing out "
                                        f"of the file, so nothing holds it to it: {reason!r}")
                files = [(_CLI / subject).with_suffix(s) for s in (".sh", ".ps1")]
                text = "\n".join(f.read_text(encoding="utf-8") for f in files if f.exists())
                for span in quoted:
                    self.assertIn(span, text,
                                  f"{subject} is skipped because of `{span}`, which no longer "
                                  f"appears in either of its cli files")

    def test_something_is_left_to_run(self) -> None:
        """The skip list may not grow until it covers everything.

        MEASURED 2026-09-10: eleven of the twenty-four subjects run on a box with no camera, no
        robot, no GPU and no Isaac. That is the floor.
        """
        runnable = [s for s in self._subjects() if s not in _CLI_NEEDS_MORE_THAN_A_BARE_BOX]
        self.assertGreaterEqual(len(runnable), 11, f"only {len(runnable)} subject(s) left to run")

    def test_every_cli_example_that_needs_nothing_exits_zero(self) -> None:
        """Both costumes of every runnable subject: exit 0, no traceback.

        Run from the repository root, because that is the door these files document: the repository
        is not pip installable, so `python -m src...` needs the root as the working directory. A
        shell this host does not have is skipped rather than counted as a pass.
        """
        shells = {
            ".sh": shutil.which("bash"),
            ".ps1": shutil.which("pwsh") or shutil.which("powershell"),
        }
        for subject in self._subjects():
            if subject in _CLI_NEEDS_MORE_THAN_A_BARE_BOX:
                continue
            for suffix, interpreter in shells.items():
                with self.subTest(example=subject, shell=suffix):
                    if interpreter is None:
                        self.skipTest(f"no interpreter for {suffix} on this host")
                    script = (_CLI / subject).with_suffix(suffix)
                    argv = ([interpreter, str(script)] if suffix == ".sh" else
                            [interpreter, "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(script)])
                    proc = subprocess.run(argv, cwd=_ROOT, capture_output=True, text=True,
                                          timeout=_TIMEOUT_S, env=_bare_env())
                    output = proc.stdout + proc.stderr
                    crash = _unhandled_traceback(output)
                    self.assertIsNone(crash, f"{script.name} raised:\n{crash}")
                    self.assertEqual(proc.returncode, 0,
                                     f"{script.name} exited {proc.returncode}:\n{output[-1500:]}")


if __name__ == "__main__":
    unittest.main()
