"""How fine should the corpus be labelled, and what does each step cost?

Two knobs decide which grasp lines get tried, and refusals concentrate on the line rather than
on the table:

    anchor_fractions   where along the free plane the closing line is anchored (n x n with the grid)
    mesh_closing_axes  how many directions that line is tried in, on top of the three principal axes

Both have been swept on isolated meshes, and this measures something different: a corpus scene has
neighbours, occlusion and a table, so the yield per candidate is lower and the cost per scene is
higher. A cost extrapolated from isolated meshes is about a different population.

Reported per arm: wall clock, labels, and objects earning at least one label. The last one is the
number that moves a corpus, because a label count alone can rise while the same few objects get
denser.
"""
from __future__ import annotations

import dataclasses
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path


from datagen.grasps.labels import DENSITIES, label_dataset

#: The default `main()` falls back to when the caller names nothing.
#:
#: `main()` parses the argument; nothing here reads `sys.argv` at import. A module that did would
#: take its value from the caller's command line under pytest or inside a service process.
#:
#: `logging.disable(logging.INFO)` runs inside `main()` for the same reason: a script may quieten
#: itself, a module may not quieten its importer's whole process.
DEFAULT_ROOT = Path("logs/dl/v5_probe")

_F5 = (0.0, -0.2, 0.2, -0.4, 0.4)
_F7 = (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45)
_F9 = (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.45, 0.45)

#: `(label, anchor fractions, closing axes)`. The first row is what ships.
ARMS: tuple[tuple[str, tuple, int], ...] = (
    ("grid, as it ships (5x5, 30 axes)", _F5, 30),
    ("7x7 anchors, 30 axes", _F7, 30),
    ("9x9 anchors, 30 axes", _F9, 30),
    ("5x5 anchors, 60 axes", _F5, 60),
    ("9x9 anchors, 60 axes", _F9, 60),
    ("9x9 anchors, 120 axes", _F9, 120),
)


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` is the probe corpus directory, optional."""
    logging.disable(logging.INFO)
    args = list(sys.argv[1:] if argv is None else argv)
    root = Path(args[0]) if args else DEFAULT_ROOT

    scenes = len(list((root / "scenes").iterdir()))
    print(f"{root}: {scenes} scenes\n", flush=True)
    print(f"  {'arm':34s} {'seconds':>8} {'jaw':>10} {'suction':>9} "
          f"{'obj w/ jaw':>11} {'of':>6} {'share':>7}", flush=True)

    base = DENSITIES["grid"]
    first: dict | None = None
    for label, fractions, axes in ARMS:
        density = dataclasses.replace(base, anchor_fractions=tuple(fractions),
                                      mesh_closing_axes=axes)
        started = time.perf_counter()
        report = label_dataset(root, density=density)
        seconds = time.perf_counter() - started
        objects = int(report.get("objects", 0))
        with_jaw = sum(int(b.get("objects_with_jaw", 0))
                       for b in report.get("by_family", {}).values())
        share = 100.0 * with_jaw / max(objects, 1)
        print(f"  {label:34s} {seconds:>8.1f} {report.get('jaw', 0):>10,} "
              f"{report.get('suction', 0):>9,} {with_jaw:>11} {objects:>6} {share:>6.1f} %",
              flush=True)
        if first is None:
            first = {"seconds": seconds, "jaw": report.get("jaw", 0), "with_jaw": with_jaw}

    if first:
        print(f"\n  Against the shipped arm: multiply its {first['seconds']:.1f} s by the corpus, "
              f"which is 20,000 scenes over {scenes} here.")
        print(f"  A shipped-arm pass over the whole corpus is about "
              f"{first['seconds'] * 20000 / max(scenes, 1) / 3600:.2f} h.")


if __name__ == "__main__":
    main()
