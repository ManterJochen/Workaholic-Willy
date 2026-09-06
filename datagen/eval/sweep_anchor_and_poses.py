"""Two levers on the meshes that earn nothing in any rest pose, measured side by side.

The anchor pitch. `anchor_fractions` is five values, so the silhouette grid is 5 x 5 = 25 points on
the plane the closing axis leaves free. Seven gives 49, nine gives 81. If a mesh has one graspable
region and the grid steps over it, a finer grid finds it and a coarser one never will.

The remaining rest poses. The screen tries upright, x-down and y-down; the other three are the
180-degree flips. Little is expected of them, because refusals concentrate on the width along the
line tried rather than on the table or a blocked approach.

Both run against the same pool and are reported as objects rescued per second, because the choice is
a number to put in a config, not a curve to admire.
"""
from __future__ import annotations

import concurrent.futures as futures
import dataclasses
import json
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np


#: The defaults `main()` falls back to when the caller names nothing.
#:
#: `main()` parses the arguments; nothing here reads `sys.argv` at import. A module that did would
#: take its value from the caller's command line under pytest or inside a service process.
DEFAULT_SAMPLE = 80
DEFAULT_JOBS = 6

#: Anchor fractions along the closing axis. `_SPREAD_5` is the shipped pitch, and it is the first
#: row of `ARMS` below, whose rows are `(label, anchor fractions, rest poses)`.
_SPREAD_5 = (0.0, -0.2, 0.2, -0.4, 0.4)
_SPREAD_7 = (0.0, -0.15, 0.15, -0.3, 0.3, -0.45, 0.45)
_SPREAD_9 = (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3, -0.45, 0.45)

_FLIP_X = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
_FLIP_Y = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]])
_FLIP_Z = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]])
_X_DOWN = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
_Y_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])

THREE = (np.eye(3), _X_DOWN, _Y_DOWN)
SIX = (*THREE, _FLIP_X, _FLIP_Y, _FLIP_Z)

ARMS: tuple[tuple[str, tuple, tuple], ...] = (
    ("shipped: 5 fractions, 3 poses", _SPREAD_5, THREE),
    ("7 fractions, 3 poses", _SPREAD_7, THREE),
    ("9 fractions, 3 poses", _SPREAD_9, THREE),
    ("5 fractions, 6 poses", _SPREAD_5, SIX),
    ("9 fractions, 6 poses", _SPREAD_9, SIX),
)


def _one(task: tuple[str, str, int]) -> int:
    source, filename, arm_index = task
    logging.disable(logging.INFO)
    from datagen.assets.meshes import load_mesh_shape  # noqa: PLC0415
    from datagen.grasps.labels import DENSITIES, SceneGeometry, label_jaw_grasps  # noqa: PLC0415
    from datagen.grasps.shapes import Solid  # noqa: PLC0415

    _label, fractions, poses = ARMS[arm_index]
    density = dataclasses.replace(DENSITIES["grid"], anchor_fractions=tuple(fractions))
    shape = load_mesh_shape(str(Path(f"assets/meshes/{source}") / filename))
    if shape is None:
        return 0
    extent = np.asarray(shape.extent_mm, dtype=float)
    best = 0
    for rotation in poses:
        rested = np.abs(rotation @ extent)
        solid = Solid(kind="mesh", half_extent_mm=extent / 2.0, rotation=rotation,
                      centre_mm=np.array([0.0, 0.0, float(rested[2]) / 2.0]),
                      instance_id=0, asset_id="probe", mesh=shape)
        geometry = SceneGeometry(scene_id="probe", family="alone", objects={0: solid}, walls=())
        labels, _ = label_jaw_grasps(geometry, 0, density=density)
        best = max(best, len(labels))
        if best:
            break                       # rescued; the count does not matter, only that it is > 0
    return best


def main(argv: Sequence[str] | None = None) -> None:
    """`argv` is the sample size and the worker count, both optional."""
    args = list(sys.argv[1:] if argv is None else argv)
    sample = int(args[0]) if args else DEFAULT_SAMPLE
    jobs = int(args[1]) if len(args) > 1 else DEFAULT_JOBS

    rows = json.loads(Path("logs/dl/screen_v5.json").read_text(encoding="utf-8"))
    zero = [r for r in rows if r.get("status") == "ok" and r.get("jaw", 0) == 0]
    picks = [zero[i] for i in np.linspace(0, len(zero) - 1, min(sample, len(zero))).astype(int)]
    print(f"{len(zero)} mesh(es) score zero in every shipped rest pose")
    print(f"sweeping {len(picks)} of them, evenly across the sorted list, {jobs} job(s)\n",
          flush=True)

    for index, (label, fractions, poses) in enumerate(ARMS):
        started = time.perf_counter()
        tasks = [(r["source"], r["file"], index) for r in picks]
        with futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            found = list(pool.map(_one, tasks))
        seconds = time.perf_counter() - started
        rescued = sum(1 for v in found if v > 0)
        share = 100.0 * rescued / max(len(picks), 1)
        print(f"  {label:32s} {rescued:>3}/{len(picks)} rescued ({share:5.1f} %)  "
              f"{len(fractions)}x{len(fractions)} anchors, {len(poses)} poses  {seconds:6.1f} s",
              flush=True)
        if index == 0 and rescued:
            print("    the shipped arm rescued something, so the pool is not what it claims")

    print(f"\n  extrapolate any share above to the whole pool of {len(zero)} by multiplying.")


if __name__ == "__main__":
    main()
