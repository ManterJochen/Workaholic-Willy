"""Logger construction: one rotating file handler per path, shared process-wide.

Every package builds its loggers through :func:`create_logger`, which resolves
the directory against ``WILLY_LOG_DIR``, creates it, and returns a logger whose
handlers are keyed by absolute path, so rotation stays atomic however many
components write to the same file.
"""

import logging
import os
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Guards the process-wide handler registries below against concurrent
# create_logger() calls.
_LOCK = threading.RLock()

# Process-wide registry of file handlers, keyed by absolute log file path.
_FILE_HANDLERS: dict[str, RotatingFileHandler] = {}

# One shared console handler, so a record reaches stdout exactly once however
# many create_logger calls were made.
_CONSOLE_HANDLER: logging.StreamHandler | None = None


def _get_file_handler(
    log_path: str,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    abs_path = os.path.abspath(log_path)
    with _LOCK:
        handler = _FILE_HANDLERS.get(abs_path)
        if handler is not None:
            return handler

        handler = RotatingFileHandler(
            abs_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
            delay=True,
        )
        handler.setFormatter(formatter)
        _FILE_HANDLERS[abs_path] = handler
        return handler


def _get_console_handler(formatter: logging.Formatter) -> logging.StreamHandler:
    global _CONSOLE_HANDLER
    with _LOCK:
        if _CONSOLE_HANDLER is None:
            _CONSOLE_HANDLER = logging.StreamHandler()
            _CONSOLE_HANDLER.setFormatter(formatter)
        return _CONSOLE_HANDLER


def _make_log_dir(path: str) -> None:
    """Create a log directory, or refuse with a sentence an operator can act on.

    Raises rather than swallowing: a cell whose logs cannot be written is a cell
    whose evidence is gone. The message names the directory and the variable
    that chooses it, because this runs during an import, where a bare
    ``os.makedirs`` failure names neither.
    """
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as exc:
        raise OSError(
            f"cannot create the log directory {path!r} ({type(exc).__name__}: {exc}). Logging is "
            f"set up at import time, so this stops the process before anything runs. Point "
            f"WILLY_LOG_DIR at a writable directory, or clear whatever occupies that path."
        ) from exc


def resolve_log_dir(log_dir: str) -> str:
    """Turn a log directory into an absolute one, honouring ``WILLY_LOG_DIR``.

    Every package constant is a bare relative string such as ``"logs/robot"``.
    Handed straight to ``os.makedirs`` those follow the working directory, so a
    cell started from two directories writes two divergent log trees and neither
    sits next to the checkout. Every logger passes through here, so this is the
    one seam at which ``WILLY_LOG_DIR`` reaches the logs at all.

    A leading ``logs`` segment is stripped: the constants already begin with it
    and ``logs_dir()`` is itself the logs directory, so joining them blindly
    would give ``<root>/logs/logs/robot``. Run from the repository root with no
    override, ``"logs/robot"`` therefore resolves to ``<root>/logs/robot``.

    An absolute ``log_dir`` is returned unchanged.
    """
    if os.path.isabs(log_dir):
        return log_dir
    from src.utility.paths import logs_dir  # noqa: PLC0415

    # logs_dir() creates the base itself, so a WILLY_LOG_DIR that points at a
    # file raises here and never reaches _make_log_dir below.
    try:
        base = Path(logs_dir())
    except OSError as exc:
        raise OSError(
            f"cannot use the log directory ({type(exc).__name__}: {exc}). Logging is set up at "
            f"import time, so this stops the process before anything runs. Point WILLY_LOG_DIR at a "
            f"writable directory, or unset it to use the checkout's own logs/."
        ) from exc
    parts = Path(log_dir).parts
    if parts and parts[0] == "logs":
        parts = parts[1:]
    return str(base.joinpath(*parts))


def create_logger(
    name: str,
    log_file: str,
    level: int = logging.INFO,
    log_dir: str = "logs",
    max_bytes: int = 5_000_000,
    backup_count: int = 3,
    *,
    aggregate_file: str | None = None,
    aggregate_dir: str | None = None,
) -> logging.Logger:
    """Create and configure a logger with file rotation and console output.

    Loggers are keyed by ``name``, so a second call with the same name returns
    the logger already configured. Handlers are keyed by absolute path, so every
    logger pointing at one file shares a single ``RotatingFileHandler`` and its
    rotation stays atomic.

    Args:
        name:           Logger name, typically the class or module.
        log_file:       Log file name, relative to ``log_dir``.
        level:          Logging level.
        log_dir:        Directory for log files, created if missing.
        max_bytes:      Maximum size per log file before rotation.
        backup_count:   Number of rotated files to keep.
        aggregate_file: Second file sink shared across many loggers, such as a
                        package-wide ``robot.log``. When set, the logger writes
                        to both its per-module ``log_file`` and this aggregate.
                        ``None`` keeps the single-file behaviour.
        aggregate_dir:  Directory for ``aggregate_file``, defaulting to ``log_dir``.

    Returns:
        A configured ``logging.Logger``.
    """
    logger = logging.getLogger(name)

    with _LOCK:
        if logger.handlers:
            return logger  # already configured for this name

        logger.setLevel(level)

        resolved = resolve_log_dir(log_dir)
        _make_log_dir(resolved)
        log_path = os.path.join(resolved, log_file)

        formatter = logging.Formatter(FORMAT)

        logger.addHandler(
            _get_file_handler(log_path, max_bytes, backup_count, formatter)
        )
        if aggregate_file is not None:
            agg_dir = resolve_log_dir(aggregate_dir) if aggregate_dir else resolved
            _make_log_dir(agg_dir)
            agg_path = os.path.join(agg_dir, aggregate_file)
            logger.addHandler(
                _get_file_handler(agg_path, max_bytes, backup_count, formatter)
            )
        logger.addHandler(_get_console_handler(formatter))

        logger.propagate = False

    return logger
