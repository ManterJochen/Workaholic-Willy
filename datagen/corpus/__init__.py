"""Judging a corpus before anyone trains on it: four checks, each one a measured defect.

The gate itself is `gate.assess_corpus`; `sources.load_corpus` reads a file into canonical columns and
`columns.column_stats` is the arithmetic underneath. `python -m datagen check-dataset` is the CLI.

Read `gate.py`'s docstring for what each check is there for. The short version: a column that looks
like a feature and carries no information stays invisible until it is quoted, and then every
diagnosis drawn from it describes the column rather than the world.
"""

from __future__ import annotations

from datagen.corpus.build import (
    attach_physics_labels,
    build_ranker_corpus,
    write_corpus,
)
from datagen.corpus.columns import ColumnStats, Table, column_stats
from datagen.corpus.gate import CorpusVerdict, assess_corpus, format_verdict
from datagen.corpus.sources import SOURCES, load_corpus

__all__ = [
    "ColumnStats",
    "CorpusVerdict",
    "SOURCES",
    "Table",
    "assess_corpus",
    "attach_physics_labels",
    "build_ranker_corpus",
    "column_stats",
    "format_verdict",
    "load_corpus",
    "write_corpus",
]
