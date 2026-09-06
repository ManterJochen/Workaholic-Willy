"""Log files land next to the installation, and `WILLY_LOG_DIR` decides where.

⛔⛔ **EVERY LOG FILE USED TO LAND RELATIVE TO THE PROCESS'S WORKING DIRECTORY.** All nine package
constants are bare relative strings (`"logs/backend/robot"`, `"logs/datagen"`, `"logs/api"`, ...) and
were handed straight to `os.makedirs`. Loggers are built at MODULE level, so the mkdir ran during
import. Started from two directories, one cell wrote two divergent log trees and neither sat next to
the checkout. MEASURED before the fix: importing one module from an empty scratch directory created
the whole `logs/backend/robot/modules` tree THERE, and a plain file named `logs` in the way aborted
the import with `FileNotFoundError` from inside a logging helper.

⛔⛔ **AND `WILLY_LOG_DIR` WAS DOCUMENTED, IMPLEMENTED, AND REACHED NOTHING.** `utility_README.md`
lists it as "override the logs directory" with no caveat and `paths.logs_dir()` honours it correctly.
The logging system never called it: `logs_dir()` had no caller outside `debug_dir()` and one
re-export. An operator points logs at a mounted volume, restarts, and gets a healthy-looking service
writing its whole log tree somewhere else, with no error and no warning.

⚠ **ONE SEAM, NOT NINE CONSTANTS.** `create_logger` resolves a relative directory rather than each
package restating an absolute one, so adding a package cannot reintroduce this.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.utility.log_cfg import resolve_log_dir

_REPO = Path(__file__).resolve().parents[1]

#: Importing this pulls two module-level loggers, one per-module and one aggregate, which is the
#: shape that makes both directories observable in one run.
_IMPORT = "import src.robot.execution.cell_lock as c"


def _run(script: str, *, cwd: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script], cwd=cwd, capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(_REPO), **env})


class TheResolvedPathIsUnchangedFromTheRepoRootTests(unittest.TestCase):
    """⚠ THE BYTE-IDENTITY CLAIM, CHECKED RATHER THAN ASSERTED IN A COMMENT. Every constant already
    begins with `logs`, and `logs_dir()` IS the logs directory, so joining them blindly would give
    `<root>/logs/logs/backend/robot`. Stripping the leading segment is what keeps a run from the
    repo root landing exactly where it always did."""

    def test_each_shipped_constant_resolves_under_the_checkout(self) -> None:
        for relative in ("logs/src/robot", "logs/src/robot/modules", "logs/datagen",
                         "logs/api", "logs/src/utility", "logs"):
            with self.subTest(relative=relative):
                self.assertEqual(str(_REPO / relative), resolve_log_dir(relative))

    def test_the_logs_segment_is_stripped_exactly_once(self) -> None:
        """A doubled `logs` would be the tell-tale of a blind join."""
        self.assertNotIn(os.path.join("logs", "logs"), resolve_log_dir("logs/datagen"))

    def test_an_absolute_directory_is_returned_unchanged(self) -> None:
        """A caller who names an absolute directory has already decided; overriding that would make
        `WILLY_LOG_DIR` capture paths their owner deliberately pinned."""
        absolute = str(Path(tempfile.mkdtemp()) / "somewhere")
        self.assertEqual(absolute, resolve_log_dir(absolute))

    def test_every_package_constant_is_still_relative(self) -> None:
        """⚠ THE PREMISE OF THE SEAM. If a package started shipping an absolute log directory it
        would silently opt out of `WILLY_LOG_DIR`, and this test is where that shows up."""
        from api.constants import API_LOG_DIR
        from src.calibration.constants import CALIBRATION_LOG_DIR
        from src.models.constants import MODELS_LOG_DIR
        from src.robot.constants import ROBOT_LOG_DIR, ROBOT_MODULES_LOG_DIR
        from src.robot.grasping.constants import GRASPING_MODULES_LOG_DIR
        from src.utility.constants import UTILITY_LOG_DIR
        from src.willy_sim.constants import WILLY_SIM_LOG_DIR
        from datagen.constants import DATAGEN_LOG_DIR

        for name, value in (
            ("API_LOG_DIR", API_LOG_DIR), ("CALIBRATION_LOG_DIR", CALIBRATION_LOG_DIR),
            ("MODELS_LOG_DIR", MODELS_LOG_DIR), ("ROBOT_LOG_DIR", ROBOT_LOG_DIR),
            ("ROBOT_MODULES_LOG_DIR", ROBOT_MODULES_LOG_DIR),
            ("GRASPING_MODULES_LOG_DIR", GRASPING_MODULES_LOG_DIR),
            ("UTILITY_LOG_DIR", UTILITY_LOG_DIR), ("WILLY_SIM_LOG_DIR", WILLY_SIM_LOG_DIR),
            ("DATAGEN_LOG_DIR", DATAGEN_LOG_DIR),
        ):
            with self.subTest(constant=name):
                self.assertFalse(os.path.isabs(value), f"{name} is absolute")
                self.assertEqual("logs", Path(value).parts[0], f"{name} does not start with logs/")


class TheWorkingDirectoryNoLongerDecidesTests(unittest.TestCase):
    """⚠ THESE SPAWN A SUBPROCESS, because the defect is in what happens at IMPORT time in a
    different working directory. Nothing in-process can observe that."""

    def test_importing_from_elsewhere_leaves_that_directory_alone(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            result = _run(f"{_IMPORT}", cwd=scratch)
            self.assertEqual(0, result.returncode, result.stderr[-400:])
            self.assertEqual([], sorted(os.listdir(scratch)),
                             "the import scattered a log tree into the working directory")

    def test_a_file_named_logs_in_the_way_no_longer_aborts_the_import(self) -> None:
        """⛔ IT DID. A bare relative `os.makedirs` walked into it and raised `FileNotFoundError`
        from a logging helper, during an import, before any `main()`."""
        with tempfile.TemporaryDirectory() as scratch:
            Path(scratch, "logs").write_text("not a directory", encoding="utf-8")
            result = _run(f"{_IMPORT}", cwd=scratch)
            self.assertEqual(0, result.returncode, result.stderr[-400:])


class TheDocumentedOverrideFinallyReachesTheLogsTests(unittest.TestCase):

    def test_both_the_module_file_and_the_aggregate_land_under_it(self) -> None:
        """⚠ BOTH, BECAUSE THERE ARE TWO DIRECTORIES. `create_logger` takes `aggregate_dir` as well,
        and a fix that resolved only the first would send a package's shared `robot.log` somewhere
        else entirely."""
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch, "dest")
            result = _run(f"{_IMPORT}\nc.logger.warning('x')",
                          cwd=scratch, WILLY_LOG_DIR=str(destination))
            self.assertEqual(0, result.returncode, result.stderr[-400:])
            written = sorted(p.name for p in destination.rglob("*.log"))
            self.assertIn("cell_lock.log", written, written)
            self.assertIn("robot.log", written, written)

    def test_nothing_is_left_in_the_working_directory(self) -> None:
        """The control. Without it "the files are in the destination" would pass for a run that
        wrote them in BOTH places."""
        with tempfile.TemporaryDirectory() as scratch:
            destination = Path(scratch, "dest")
            _run(f"{_IMPORT}\nc.logger.warning('x')", cwd=scratch,
                 WILLY_LOG_DIR=str(destination))
            self.assertEqual(["dest"], sorted(os.listdir(scratch)))

    def test_an_unusable_override_refuses_and_names_the_variable(self) -> None:
        """⚠ LOUD, NOT SILENT, AND DELIBERATELY SO. A cell whose logs cannot be written is a cell
        whose evidence is gone. What was wrong before was not the raising: it was a bare
        `FileExistsError` from a logging helper with nothing naming the variable that chose the path.
        """
        with tempfile.TemporaryDirectory() as scratch:
            blocked = Path(scratch, "a_file_not_a_dir")
            blocked.write_text("x", encoding="utf-8")
            result = _run(_IMPORT, cwd=scratch, WILLY_LOG_DIR=str(blocked))
            self.assertNotEqual(0, result.returncode)
            self.assertIn("WILLY_LOG_DIR", result.stderr)


if __name__ == "__main__":
    unittest.main()
