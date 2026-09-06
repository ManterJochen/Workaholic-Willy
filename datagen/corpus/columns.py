"""What one column of a corpus actually contains: the arithmetic every check here is built on.

A second implementation on purpose. `src.robot.grasping.rl.readiness` computes the same statistics
and reaches the same verdict on a constant column, but importing it would make a datagen package
depend on the RL layer for twelve lines of arithmetic, and its pure-Python loop takes minutes over a
corpus this one measures in milliseconds.

So this one is numpy and lives where the data is made. The rule that keeps the two from drifting is
that they agree row for row: a change here that moves a verdict has to move `readiness` with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["ColumnStats", "Table", "column_stats"]

#: A corpus in memory: canonical column name -> values, one entry per row. Numeric columns are float
#: arrays; the group column may be strings. Every loader in `sources.py` produces this shape.
Table = dict[str, np.ndarray]


@dataclass(frozen=True, slots=True)
class ColumnStats:
    """What one column did across the rows it was measured on."""

    key: str
    rows: int
    distinct: int
    nonzero: int
    stdev: float
    minimum: float
    maximum: float
    #: Rows whose value is NaN or infinite. Separate from everything else because a column can be
    #: perfectly varied and still be unusable, and a mean would simply come back NaN without saying why.
    nonfinite: int = 0
    #: False for ids and labels. Carried because without it a string column reports "constant at 0",
    #: naming a number it never had.
    numeric: bool = True

    @property
    def verdict(self) -> str:
        """``absent``, ``nonfinite``, ``constant`` or ``live``: the four words `readiness` uses."""
        if self.rows == 0:
            return "absent"
        if self.nonfinite:
            return "nonfinite"
        if self.distinct <= 1:
            return "constant"
        return "live"

    def describe(self) -> str:
        if self.rows == 0:
            return f"{self.key}: absent"
        if self.nonfinite:
            return f"{self.key}: {self.nonfinite}/{self.rows} row(s) NaN or infinite"
        if self.distinct <= 1:
            where = f"at {self.minimum:.6g}" if self.numeric else "at its one value"
            return f"{self.key}: CONSTANT {where} across {self.rows} row(s)"
        if not self.numeric:
            return f"{self.key}: {self.distinct} distinct value(s) over {self.rows} row(s)"
        return (f"{self.key}: {self.distinct} distinct, {self.minimum:.4g} .. {self.maximum:.4g}, "
                f"sd {self.stdev:.4g}, {self.nonzero}/{self.rows} nonzero")


def column_stats(key: str, values: Any) -> ColumnStats:
    """Measure one column. Non-numeric values are counted as rows and nothing else.

    ``distinct`` rounds to nine decimals, matching `readiness._column`, so a column that differs only
    in floating-point noise counts as the one value it really is.
    """
    array = np.asarray(values)
    rows = int(array.size)
    if rows == 0:
        return ColumnStats(key, 0, 0, 0, 0.0, 0.0, 0.0)

    if array.dtype.kind in "US O".replace(" ", ""):
        # A label or an id: distinct is the only meaningful statistic, and it is the one the group
        # check needs.
        return ColumnStats(key, rows, int(np.unique(array).size), rows, 0.0, 0.0, 0.0,
                           numeric=False)

    numeric = array.astype(np.float64, copy=False)
    finite = np.isfinite(numeric)
    nonfinite = int(rows - finite.sum())
    if nonfinite == rows:
        return ColumnStats(key, rows, 0, 0, 0.0, 0.0, 0.0, nonfinite=nonfinite)

    usable = numeric[finite]
    return ColumnStats(
        key=key,
        rows=rows,
        distinct=int(np.unique(np.round(usable, 9)).size),
        nonzero=int(np.count_nonzero(usable)),
        stdev=float(usable.std()),
        minimum=float(usable.min()),
        maximum=float(usable.max()),
        nonfinite=nonfinite,
    )
