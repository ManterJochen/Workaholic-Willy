"""A progress bar for a person watching a terminal, and nothing anywhere else.

A long run that logs one line per epoch looks the same working and hung between those lines; a bar closes that gap.
It goes to ``stderr`` and only when ``stderr`` is a terminal: redirected to a file, run under ``nohup`` or in a CI job it
writes nothing, so a log keeps its shape (a bar's carriage returns would turn a log file into thousands of lines).

ASCII, always: tqdm's default bar is Unicode block characters, and a Windows console under cp1252 raises on them. A bar
never ends a run: tqdm missing, or a terminal that refuses the write, is silence.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["is_interactive", "progress_bar"]


def is_interactive() -> bool:
    """Whether a person is watching ``stderr``: ``nohup``, a redirect, a CI job and a container without a TTY say no.

    Returns:
        bool: ``True`` when ``stderr`` is a terminal.
    """
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        # A closed or replaced stream: nobody is watching.
        return False


class _Silent:
    """The bar nobody sees: the surface of tqdm's, so a caller never branches."""

    def update(self, _n: int = 1) -> None:
        return

    def set_postfix_str(self, _text: str) -> None:
        return

    def close(self) -> None:
        return


@contextmanager
def progress_bar(total: int, *, desc: str, unit: str = "batch", enabled: bool | None = None) -> Iterator[Any]:
    """A bar over ``total`` steps that disappears when it is done; yields something with ``update`` and
    ``set_postfix_str``.

    Args:
        total (int): How many steps the bar counts: batches, not images, so it moves on every step.
        desc (str): What it is a bar of, before the bar, ASCII.
        unit (str): What one step is called (default: "batch").
        enabled (bool | None): ``None`` shows it where :func:`is_interactive` says a person watches (default: None).

    Yields:
        Any: tqdm's bar, or one that writes nothing where none is shown or tqdm is not installed.
    """
    if enabled is None:
        enabled = is_interactive()
    if not enabled or total <= 0:
        yield _Silent()
        return
    try:
        from tqdm import tqdm  # noqa: PLC0415 (optional at runtime by design)
    except ImportError:
        yield _Silent()
        return
    bar = tqdm(total=total, ascii=True, file=sys.stderr, desc=desc, unit=unit, leave=False, dynamic_ncols=True)
    try:
        yield bar
    finally:
        try:
            bar.close()
        except Exception:  # noqa: BLE001 (closing a bar must never end a run)
            pass
