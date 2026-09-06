"""Iteration: ``log_cfg.create_logger`` is thread-safe.

Concurrent calls for the same logger/file must produce exactly one shared
``RotatingFileHandler`` and one console handler — no duplicates. Duplicate file
handlers on the same path are precisely the Windows multi-writer / rotation race
this module exists to prevent.
"""

from __future__ import annotations

import logging
import tempfile
import threading
import unittest
import uuid
from logging.handlers import RotatingFileHandler

from src.utility.log_cfg import create_logger


class CreateLoggerThreadSafetyTests(unittest.TestCase):
    def test_concurrent_create_logger_is_idempotent(self) -> None:
        name = f"logcfg_race_{uuid.uuid4().hex}"
        self.addCleanup(self._cleanup, name)
        n = 48
        barrier = threading.Barrier(n)
        loggers: list[logging.Logger] = []
        append_lock = threading.Lock()

        with tempfile.TemporaryDirectory() as tmp:
            def worker() -> None:
                barrier.wait(timeout=10)  # release all threads together = max contention
                lg = create_logger(name, "race.log", log_dir=tmp)
                with append_lock:
                    loggers.append(lg)

            threads = [threading.Thread(target=worker) for _ in range(n)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            logger = logging.getLogger(name)
            # Every call returned the same process-global logger object.
            self.assertEqual(len(loggers), n)
            self.assertTrue(all(lg is logger for lg in loggers))
            # Exactly one file handler + one console handler — no duplicates.
            file_handlers = [
                h for h in logger.handlers if isinstance(h, RotatingFileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(len(logger.handlers), 2)

            # Release the open RotatingFileHandler BEFORE TemporaryDirectory teardown:
            # Windows cannot delete a file that is still held open (POSIX can), so the
            # ``with`` exit would otherwise raise PermissionError [WinError 32]. The
            # ``addCleanup`` registration above stays as an idempotent backstop.
            self._cleanup(name)

    @staticmethod
    def _cleanup(name: str) -> None:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()


if __name__ == "__main__":
    unittest.main()
