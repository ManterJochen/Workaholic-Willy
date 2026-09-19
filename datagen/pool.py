"""The one way this package starts a process pool: workers spawn on every platform, never fork.

A worker started by ``fork`` inherits a copy of every lock the parent held at that instant, the ones torch's
and OpenMP's thread pools hold included, and nothing ever releases them in the child. MEASURED 2026-09-19 on
Linux: the mesh rail's first worker hung for 55 minutes inside ``import open3d``, in the extension's own
initialisation, while the same test passes on Windows, where ``spawn`` is the only start method there is. So
``spawn`` everywhere makes Linux run what Windows has always run, and a worker function that reaches this
pool must be importable at module scope, which Windows already demanded of it.
"""

from __future__ import annotations

import multiprocessing
from collections.abc import Callable
from concurrent import futures
from typing import Any


def process_pool(
    max_workers: int,
    *,
    initializer: Callable[..., object] | None = None,
    initargs: tuple[Any, ...] = (),
) -> futures.ProcessPoolExecutor:
    """A ``ProcessPoolExecutor`` whose workers start by ``spawn``, on Windows and on Linux alike."""
    return futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initializer,
        initargs=initargs,
    )
