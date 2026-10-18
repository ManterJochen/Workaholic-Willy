"""Top-level pytest configuration for the Workaholic-Willy suite.

The byte-identity family that stood its float-locked goldens down here (``tests/_determinism.py``) left on
2026-10-16 with every golden it guarded: the canonical replay packs, the RL ranking artifact and the success
model (the owner's cleanup).

(The integration ``isaac`` marker is gated separately in ``tests/integration/conftest.py``.)
"""

from __future__ import annotations

import shutil
import tempfile

import pytest

from tests import _injection_guard as injection_guard


#: Where the run's picks keep their looks (:func:`pytest_configure`); ``""`` before it.
_VIEWS = ""


def pytest_configure(config: pytest.Config) -> None:
    """Guard injection, and give the run's picks a folder of their own for their looks."""

    del config
    # A test may only inject what the arm already carries; see tests/_injection_guard.py.
    injection_guard.install()
    # Every pick keeps its looks (record_views, the owner, 2026-10-15): the suite's go to a folder of its own, never into
    # the cell's logs/robot/views, and the folder goes when the run ends.
    from src.robot.execution import record_views

    global _VIEWS
    _VIEWS = tempfile.mkdtemp(prefix="willy-test-views-")
    record_views.RECORD_VIEWS_DIR = _VIEWS  # type: ignore[misc]
    # So does every pick's grasp record where the tree names no file (record_log_path, 2026-10-17).
    from src.robot.execution.autonomous_grasp import service

    service.RECORD_LOG_DIR = _VIEWS


def pytest_unconfigure(config: pytest.Config) -> None:
    """Take the run's folder of looks away."""

    del config
    if _VIEWS:
        shutil.rmtree(_VIEWS, ignore_errors=True)
