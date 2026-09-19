"""Every process pool in the product starts its workers by spawn, through ``datagen.pool.process_pool``.

A pool opened directly takes the platform's default start method: spawn on Windows, fork on Linux. MEASURED
2026-09-19: under fork, the mesh rail's first worker hung for 55 minutes in ``import open3d`` on Linux, while
Windows, which only spawns, had passed the same test for months. A direct pool is therefore a Linux-only hang
that no Windows run can see, and this test is where it shows on both.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

from datagen.pool import process_pool

ROOT = Path(__file__).resolve().parents[1]
SCANNED = ("src", "datagen", "scripts", "api", "willy", "examples")
THE_ONE_POOL = Path("datagen/pool.py")
POOL_CALLS = frozenset({"ProcessPoolExecutor", "Pool"})


def _pool_calls(source: str, filename: str) -> list[str]:
    """Every call in ``source`` that opens a process pool, as ``filename:line name(...)``."""
    found: list[str] = []
    for node in ast.walk(ast.parse(source, filename=filename)):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in POOL_CALLS:
                found.append(f"{filename}:{node.lineno} {name}(...)")
    return found


def _direct_pools() -> list[str]:
    found: list[str] = []
    for top in SCANNED:
        # A folder that is not there scans as clean, so a renamed tree would pass on nothing.
        if not (ROOT / top).is_dir():
            raise AssertionError(f"{top}/ is not in this tree; name the folder that holds the code")
        for path in sorted((ROOT / top).rglob("*.py")):
            rel = path.relative_to(ROOT)
            if rel == THE_ONE_POOL or "__pycache__" in rel.parts:
                continue
            found += _pool_calls(path.read_text(encoding="utf-8"), rel.as_posix())
    return found


class EveryProcessPoolSpawnsTests(unittest.TestCase):
    def test_the_one_pool_spawns(self) -> None:
        pool = process_pool(1)
        try:
            self.assertEqual(pool._mp_context.get_start_method(), "spawn")
        finally:
            pool.shutdown()

    def test_no_pool_is_opened_directly(self) -> None:
        self.assertEqual(
            _direct_pools(), [],
            "open process pools through datagen.pool.process_pool, which spawns on every platform",
        )

    def test_the_scan_sees_a_direct_pool(self) -> None:
        # The negative control: the reader the scan uses reports each call it exists to forbid.
        source = ("import concurrent.futures as f\nfrom multiprocessing import Pool\n"
                  "f.ProcessPoolExecutor(max_workers=2)\nPool(2)\n")
        self.assertEqual(_pool_calls(source, "x.py"),
                         ["x.py:3 ProcessPoolExecutor(...)", "x.py:4 Pool(...)"])


if __name__ == "__main__":
    unittest.main()
