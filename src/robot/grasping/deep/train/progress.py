"""What a customer sees while a training run is working: a bar that moves, on the terminal only.

A `train-set` run is hours long and logs one line per epoch, about ten minutes apart. Between those
lines a working run and a hung one look identical on the terminal, which is the gap this closes.

The bar goes to the terminal, the epoch lines go to the log, and they must not swap. The training
logger writes to `logs/deep_set_loop.log` as well as to the console. A progress bar emits a
carriage return per update, so a bar written into that file turns it into thousands of unreadable
lines. This writes to `stderr` and only when `stderr` is a terminal; redirected to a file, or run
under `nohup` as every long arm here is, it emits nothing at all and the log keeps its shape.

ASCII, always. tqdm's default bar is Unicode block characters and this project's terminal is
cp1252, where printing one raises `UnicodeEncodeError`, and a crash in a progress bar would kill a
six-hour run for a decoration.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from typing import Any, Iterator

__all__ = ["epoch_bar", "is_interactive"]


def is_interactive() -> bool:
    """True when a person is watching, which is the only case a bar is for.

    `nohup`, a redirect, a CI job and a container without a TTY all answer False. A bar that
    ignored this would write control characters into the file somebody later greps.
    """
    try:
        return bool(sys.stderr.isatty())
    except (AttributeError, ValueError):
        # A closed or replaced stream. Not an error: it means nobody is watching.
        return False


class _Silent:
    """The no-op bar. Same surface as tqdm's, so the caller never branches."""

    def update(self, _n: int = 1) -> None:
        return

    def set_postfix_str(self, _text: str) -> None:
        return

    def close(self) -> None:
        return


@contextmanager
def epoch_bar(total: int, *, epoch: int, epochs: int, fold: int = 0,
              enabled: bool | None = None) -> Iterator[Any]:
    """A bar over the batches of one epoch. Yields something with `update` and `set_postfix_str`.

    `total` is the batch count, not the unit count: a bar that counted units would jump by the batch
    size and look stuck between jumps.

    If tqdm is missing, or the terminal refuses the write, the run must continue. A progress bar
    must never cost a six-hour run, so every path here degrades to silence rather than raising.
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

    bar = tqdm(
        total=total,
        # ascii=True is not a style choice; see the module docstring.
        ascii=True,
        # stderr so a `> run.log` redirect keeps the log clean and the bar disappears with it.
        file=sys.stderr,
        desc=f"fold {fold} epoch {epoch}/{epochs}",
        unit="batch",
        leave=False,
        dynamic_ncols=True,
    )
    try:
        yield bar
    finally:
        try:
            bar.close()
        except Exception:  # noqa: BLE001 (closing a bar must never end a run)
            pass


def postfix(totals: dict[str, float], batches: int) -> str:
    """The running numbers to show beside the bar, as ASCII.

    A short selection, not every metric: a postfix carrying twelve metrics wraps the line on a
    narrow terminal and nobody reads it. `total` says whether the run is descending at all; the
    target's own angle says whether the thing being fitted is moving. Only keys present in `totals`
    are printed, and an epoch's running totals carry `total` alone: the bare angle names are
    resolved from their `_sum` and `_count` pairs after the epoch, not during it.
    """
    if batches <= 0:
        return ""
    parts = []
    for key in ("total", "approach_error_deg", "axis_error_deg"):
        if key in totals:
            value = totals[key] / batches
            parts.append(f"{key.replace('_error_deg', '')} {value:.3f}")
    return "  ".join(parts)
