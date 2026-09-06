"""Does a denser closing-axis sample find grasps the current one misses?

Refusals on meshes that earn no label concentrate on the width along the line tried, and barely
involve the table, so the choice of line is the lever. `_closing_axes` tries the mesh's three
principal axes plus a Fibonacci hemisphere of `mesh_closing_axes` directions; at 30 that hemisphere
has about 23 degrees between neighbours.

The principal axes already cover the obvious grasp on an elongated object. What a coarse hemisphere
can miss is a grasp on a local feature, a neck, a rim, a handle, whose good direction is not a
principal axis of the whole mesh.

Swept on meshes that score zero in every rest pose, because those are the ones a denser sample would
have to rescue for the lever to be worth its cost. Reported as yield per second, since the cost is
linear in the axis count and the point is to choose a number, not to admire a curve.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from datagen.assets.meshes import load_mesh_shape
from datagen.grasps.labels import DENSITIES, SceneGeometry, label_jaw_grasps
from datagen.grasps.shapes import Solid

#: How many of the zero-scoring meshes to sweep, when the caller names no other number.
#:
#: `main()` parses the argument; nothing here reads `sys.argv` at import. A module that did would
#: take its value from the caller's command line under pytest or inside a service process.
DEFAULT_SAMPLE = 60

#: The screen report this reads, when the caller names no other one. `screen-meshes` writes it.
DEFAULT_SCREEN = Path("logs/dl/screen_v5.json")
COUNTS = (30, 60, 120, 240)

POSES = (np.eye(3),
         np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]),
         np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]))


def _best_over_poses(shape, density) -> int:
    extent = np.asarray(shape.extent_mm, dtype=float)
    best = 0
    for rotation in POSES:
        rested = np.abs(rotation @ extent)
        solid = Solid(kind="mesh", half_extent_mm=extent / 2.0, rotation=rotation,
                      centre_mm=np.array([0.0, 0.0, float(rested[2]) / 2.0]),
                      instance_id=0, asset_id="probe", mesh=shape)
        geometry = SceneGeometry(scene_id="probe", family="alone", objects={0: solid}, walls=())
        labels, _ = label_jaw_grasps(geometry, 0, density=density)
        best = max(best, len(labels))
    return best


def main(argv: Sequence[str] | None = None) -> None:
    """Run the sweep. `argv` is the sample size and the screen report, both optional.

    `logging.disable(logging.INFO)` runs here, not at import: at module level it would silence info
    for the whole process of anyone who imported this file, including a caller who never asked. Only
    a person running this module pays for it.
    """
    logging.disable(logging.INFO)
    args = list(sys.argv[1:] if argv is None else argv)
    sample = int(args[0]) if args else DEFAULT_SAMPLE
    screen = Path(args[1]) if len(args) > 1 else DEFAULT_SCREEN

    rows = json.loads(screen.read_text(encoding="utf-8"))
    zero = [r for r in rows if r.get("status") == "ok" and r.get("jaw", 0) == 0]
    print(f"{len(zero)} mesh(es) scoring zero in every rest pose at {COUNTS[0]} closing axes")
    picks = [zero[i] for i in np.linspace(0, len(zero) - 1, min(sample, len(zero))).astype(int)]
    print(f"sweeping {len(picks)} of them, evenly sampled across the sorted list\n", flush=True)

    base = DENSITIES["grid"]
    results: dict[int, tuple[int, int, float]] = {}
    for count in COUNTS:
        density = dataclasses.replace(base, mesh_closing_axes=count)
        started = time.perf_counter()
        gained = labels = 0
        for row in picks:
            shape = load_mesh_shape(str(Path(f"assets/meshes/{row['source']}") / row["file"]))
            if shape is None:
                continue
            found = _best_over_poses(shape, density)
            gained += int(found > 0)
            labels += found
        seconds = time.perf_counter() - started
        results[count] = (gained, labels, seconds)
        print(f"  {count:>4} axes: {gained:>3}/{len(picks)} rescued  {labels:>8,} label(s)  "
              f"{seconds:6.1f} s  ({seconds / max(len(picks), 1):.2f} s/mesh)", flush=True)

    print("\n  what each doubling buys, against what it costs:")
    previous = None
    for count in COUNTS:
        gained, labels, seconds = results[count]
        if previous is not None:
            pg, _pl, ps = results[previous]
            extra = gained - pg
            cost = seconds / max(ps, 1e-9)
            print(f"    {previous:>4} -> {count:<4}  +{extra:>3} mesh(es) rescued   "
                  f"{cost:.2f}x the time")
        previous = count
    total_zero = len(zero)
    best = max(COUNTS)
    share = results[best][0] / max(len(picks), 1)
    print(f"\n  extrapolated to the whole zero pool at {best} axes: "
          f"about {int(round(share * total_zero))} of {total_zero} objects")


if __name__ == "__main__":
    main()
