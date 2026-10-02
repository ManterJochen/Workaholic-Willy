"""The pick's own log lines reach the robot log: the service's, PickRun's and the generated view's.

RC6 of the cell-fix plan (2026-10-01): the pick service, ``PickRun`` and ``generated_view`` logged through bare
``logging.getLogger(__name__)`` loggers, which nothing configures, so their INFO lines reached no file: E6 proved "support
plane observed" fired five times out of five and is in no file. Each now logs through ``create_robot_logger`` under its
own module name, to ``robot.log`` and to a per-module file beside the other robot loggers. The names stay, so a test that
captures ``assertLogs("src.robot...")`` still does; a capture through the root logger no longer sees them, because a
robot logger does not propagate.

These spawn a subprocess, because the loggers are built at import and ``WILLY_LOG_DIR`` is read then.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

#: Each module whose lines must reach the robot log, and the per-module file it writes beside it.
_MODULES = {
    "src.robot.execution.pick_run": "pick_run.log",
    "src.robot.execution.autonomous_grasp.service": "grasp_service.log",
    "src.robot.execution.generated_view": "generated_view.log",
}


def _emit_and_read(destination: Path) -> subprocess.CompletedProcess[str]:
    lines = ["import importlib, logging"]
    for module in _MODULES:
        lines.append(f"importlib.import_module({module!r})")
        lines.append(f"logging.getLogger({module!r}).info('probe from %s', {module!r})")
    lines.append("logging.shutdown()")
    return subprocess.run([sys.executable, "-c", "\n".join(lines)], cwd=str(_REPO), capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(_REPO), "WILLY_LOG_DIR": str(destination)},
                          timeout=300)


class ThePickLogsReachTheRobotLogTests(unittest.TestCase):
    _scratch: tempfile.TemporaryDirectory[str]
    destination: Path
    result: subprocess.CompletedProcess[str]

    @classmethod
    def setUpClass(cls) -> None:
        cls._scratch = tempfile.TemporaryDirectory()
        cls.destination = Path(cls._scratch.name, "logs")
        cls.result = _emit_and_read(cls.destination)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._scratch.cleanup()

    def _read(self, name: str) -> str:
        found = sorted(self.destination.rglob(name))
        self.assertTrue(found, f"no {name} under {self.destination}: {sorted(p.name for p in self.destination.rglob('*'))}")
        return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in found)

    def test_an_info_from_each_lands_in_robot_log(self) -> None:
        """⭐ Red before: robot.log held none of the three lines."""
        self.assertEqual(0, self.result.returncode, self.result.stderr[-800:])
        robot_log = self._read("robot.log")
        for module in _MODULES:
            with self.subTest(module):
                self.assertIn(f"probe from {module}", robot_log)

    def test_each_writes_its_own_file_beside_the_others(self) -> None:
        self.assertEqual(0, self.result.returncode, self.result.stderr[-800:])
        for module, name in _MODULES.items():
            with self.subTest(module):
                self.assertIn(f"probe from {module}", self._read(name))

    def test_the_names_stay_so_a_named_capture_still_hears_them(self) -> None:
        """``assertLogs("src.robot.execution.pick_run")`` is how the suite listens; the name is the module's."""
        import logging

        from src.robot.execution import generated_view, pick_run
        from src.robot.execution.autonomous_grasp import service

        for module, logger in (("src.robot.execution.pick_run", pick_run.logger),
                               ("src.robot.execution.autonomous_grasp.service", service._LOG),
                               ("src.robot.execution.generated_view", generated_view._LOG)):
            with self.subTest(module):
                self.assertEqual(module, logger.name)
                with self.assertLogs(module, level=logging.INFO) as said:
                    logger.info("heard")
                self.assertEqual([f"INFO:{module}:heard"], said.output)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
