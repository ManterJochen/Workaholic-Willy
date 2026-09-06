"""Shared constants for the operator console.

Centralising the log paths here is the same move ``src/models/constants.py`` makes, for the same
reason: a directory referenced by fifteen magic strings drifts, and the drift is invisible until
somebody goes looking for a file that was written somewhere else.
"""

from __future__ import annotations

from typing import Final

# ----------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------
#: Directory for every rotating log this package writes. Resolved relative to the process working
#: directory by :func:`src.utility.log_cfg.create_logger`.
#:
#: Deliberately not ``logs/console/``: that directory is where the console appends grasp records
#: (``Console.record_log_path``), and those are data the KPI roll-up parses. Keeping the rotating
#: ``*.log`` / ``*.log.1`` files out of it means an operator looking at either one is looking at only
#: one kind of thing.
API_LOG_DIR: Final[str] = "logs/api"

#: Per-module log-file names. Separate files rather than the consolidated single-file pattern the
#: robot package uses: the console's subsystems are independent (a build, a config write, a run, a
#: websocket) and an operator debugging one of them wants to read it without the others interleaved.
#: The ``router_*`` prefix disambiguates the two ``cell`` modules: ``api/cell.py`` holds the session,
#: ``api/routers/cell.py`` is its HTTP surface.
#:
#: The list is exactly the modules that log. ``events.py``, ``schemas.py``, ``routers/history.py``
#: and ``routers/preflight.py`` are absent on purpose: they hold data or delegate, and a constant
#: here for one of them would promise a file that is never written.
APP_LOG_FILE: Final[str] = "app.log"
CELL_LOG_FILE: Final[str] = "cell.log"
HISTORY_LOG_FILE: Final[str] = "history.log"
LIFECYCLE_LOG_FILE: Final[str] = "lifecycle.log"
RUNS_LOG_FILE: Final[str] = "runs.log"
TELEMETRY_LOG_FILE: Final[str] = "telemetry.log"
VIEWFINDER_LOG_FILE: Final[str] = "viewfinder.log"
ROUTER_CELL_LOG_FILE: Final[str] = "router_cell.log"
ROUTER_CONFIG_LOG_FILE: Final[str] = "router_config.log"
ROUTER_DIAGNOSTICS_LOG_FILE: Final[str] = "router_diagnostics.log"
ROUTER_MEDIA_LOG_FILE: Final[str] = "router_media.log"
ROUTER_PICK_LOG_FILE: Final[str] = "router_pick.log"
