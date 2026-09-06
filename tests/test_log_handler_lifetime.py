"""A log file opens when something is written to it, not when a module is imported.

WHY THIS MATTERS ENOUGH FOR A TEST. `create_logger` is called at MODULE SCOPE — that is the house
idiom, and it is the right one. The consequence is that every process which imports a logging module
runs the handler's constructor, and `RotatingFileHandler` opens its file there unless told otherwise.

That is invisible in one process and a real defect across several. MEASURED 2026-08-21 on this repo,
before the fix: three `ProcessPoolExecutor` workers importing `datagen.eval.ladder` each held
three log-file handles — `grasp_eval.log`, `grasp_labels.log`, `licensing.log` — none of which they
ever wrote to. The parent could then not rotate any of them:

    PermissionError: [WinError 32] ... grasp_eval.log -> grasp_eval.log.rotatetest

Windows refuses to rename a file another process holds open. Rotation failing is SILENT — `logging`
swallows handler errors and prints to stderr — so the symptom is not a crash but a "rotating" log
that quietly grows past its 5 MB cap forever. `delay=True` fixes it by construction: a process that
never logs never opens the file, and the workers never log (every call site in the evaluator is on
the parent's side of the pool).

The second, smaller property this buys is honesty: a log file that EXISTS now means something was
logged there. Before, 40-odd empty files appeared the moment anything imported the package.
"""

from __future__ import annotations

import io
import logging
import tempfile
import unittest
from pathlib import Path

from src.utility.log_cfg import create_logger


class LogFilesOpenOnFirstWriteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(self._drop_handlers)

    def _drop_handlers(self) -> None:
        """Release the temp files before the directory goes away (Windows will not delete open ones)."""
        for name in list(logging.Logger.manager.loggerDict):
            logger = logging.getLogger(name)
            for handler in list(getattr(logger, "handlers", [])):
                stream = getattr(handler, "stream", None)
                if stream is not None and str(getattr(stream, "name", "")).startswith(str(self.tmp)):
                    handler.close()
                    logger.removeHandler(handler)

    def test_creating_a_logger_does_not_create_its_file(self) -> None:
        """The property. An imported-but-silent module must hold nothing."""
        target = self.tmp / "quiet.log"
        create_logger("test.quiet", "quiet.log", log_dir=str(self.tmp))
        self.assertFalse(
            target.exists(),
            "the log file was opened at construction — `delay=True` has been lost from "
            "`create_logger`, and every pool worker will hold a handle again",
        )

    def test_the_file_appears_on_the_first_message(self) -> None:
        target = self.tmp / "loud.log"
        logger = create_logger("test.loud", "loud.log", log_dir=str(self.tmp))
        logger.info("the first line")
        for handler in logger.handlers:
            handler.flush()
        self.assertTrue(target.exists(), "writing a line did not create the file")
        self.assertIn("the first line", io.open(target, encoding="utf-8").read())

    def test_the_aggregate_sink_is_lazy_too(self) -> None:
        """The robot package writes to BOTH a per-module file and a package aggregate."""
        module_file = self.tmp / "sub.log"
        aggregate = self.tmp / "package.log"
        logger = create_logger(
            "test.aggregate", "sub.log", log_dir=str(self.tmp),
            aggregate_file="package.log", aggregate_dir=str(self.tmp),
        )
        self.assertFalse(module_file.exists())
        self.assertFalse(aggregate.exists())

        logger.warning("something degraded")
        for handler in logger.handlers:
            handler.flush()
        self.assertTrue(module_file.exists(), "the per-module sink never opened")
        self.assertTrue(aggregate.exists(), "the aggregate sink never opened")

    def test_two_loggers_on_one_path_still_share_a_handler(self) -> None:
        """`delay` must not break the path-keyed sharing the Windows write race depends on."""
        first = create_logger("test.share.a", "shared.log", log_dir=str(self.tmp))
        second = create_logger("test.share.b", "shared.log", log_dir=str(self.tmp))

        def file_handlers(logger: logging.Logger) -> list[logging.Handler]:
            return [h for h in logger.handlers if hasattr(h, "baseFilename")]

        self.assertTrue(file_handlers(first))
        self.assertIs(
            file_handlers(first)[0], file_handlers(second)[0],
            "two loggers on one path got two handlers — rotation is no longer atomic",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
