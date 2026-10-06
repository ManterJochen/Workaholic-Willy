"""Build a world the bench wrote out again (``worlds/<label>.pkl``), and say which step filled the cells the robot hid.

The bench writes the last few calls of ``build_perceived_boxes`` whole when a pick is refused for cells the robot's own
body hid from the cameras (``run_bench._write_worlds``). This builds one of them again with the height map's steps
watched: how many cells ``_fill_hidden`` filled, how many of them ``_under_the_robot`` lowered or left, how many a row
ran on into (``_run_on``), what ``bridge_columns`` built between two parts, and for every box with hidden cells in the
world it came to, where the robot's body stands in it.

    python scripts/bench/replay_world.py logs/bench/run12/worlds/tray_cube40.pkl [--world -1]
"""

from __future__ import annotations

import argparse
import math
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _box_corners_grid(centre: np.ndarray, rotation: np.ndarray, half: np.ndarray, step: float = 4.0) -> np.ndarray:
    """Points filling a box on a ``step`` grid, BASE."""
    axes = [np.arange(-h, h + 1e-9, step) if h > step else np.array([0.0]) for h in half]
    a, b, c = np.meshgrid(*axes, indexing="ij")
    local = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    return local @ rotation.T + centre


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("path", type=Path)
    parser.add_argument("--world", type=int, default=-1, help="which recorded world, -1 the last")
    args = parser.parse_args(argv)

    from src.robot.safety.planning import height_map as hm
    from src.robot.safety.planning import perceived as pc

    with args.path.open("rb") as handle:
        worlds = pickle.load(handle)
    print(f"{len(worlds)} world(s) recorded; building {args.world}")
    call_args, call_kwargs = worlds[args.world]
    self_body = call_kwargs.get("self_body")

    steps: list[dict[str, Any]] = []
    original_fill, original_under, original_run = hm._fill_hidden, hm._under_the_robot, hm._run_on
    original_bridge = pc.bridge_columns

    def fill(top: Any, bottom: Any, grid: Any, hidden: Any, **kw: Any) -> Any:
        t, b, f = original_fill(top, bottom, grid, hidden, **kw)
        steps.append({"step": "fill_hidden", "grid": grid, "filled": int(f.sum()), "cells": np.argwhere(f),
                      "tops": t[f], "seen": top[f]})
        return t, b, f

    def under(top: Any, bottom: Any, filled: Any, seen_top: Any, seen_bottom: Any, grid: Any, robot_on: Any,
              margin_mm: float, floor_mm: Any) -> Any:
        t, b, f = original_under(top, bottom, filled, seen_top, seen_bottom, grid, robot_on, margin_mm, floor_mm)
        lowered = int(np.count_nonzero(filled & f & (t < top)))
        unfilled = int(np.count_nonzero(filled & ~f))
        steps.append({"step": "under_the_robot", "grid": grid, "kept": int(f.sum()), "lowered": lowered,
                      "unfilled": unfilled, "cells": np.argwhere(f), "tops": t[f], "floor": floor_mm,
                      "robot_on": robot_on, "margin": margin_mm, "bottoms": b[f]})
        return t, b, f

    def run_on(top: Any, bottom: Any, grid: Any, hidden: Any, **kw: Any) -> Any:
        t, b, ran, sources = original_run(top, bottom, grid, hidden, **kw)
        steps.append({"step": "run_on", "grid": grid, "ran": int(ran.sum()), "cells": np.argwhere(ran),
                      "tops": t[ran]})
        return t, b, ran, sources

    def bridge(*a: Any, **kw: Any) -> Any:
        columns = original_bridge(*a, **kw)
        steps.append({"step": "bridge", "columns": [(c.low.copy(), c.high.copy(), c.hidden_cells) for c in columns]})
        return columns

    hm._fill_hidden, hm._under_the_robot, hm._run_on, pc.bridge_columns = fill, under, run_on, bridge
    try:
        world = pc.build_perceived_boxes(*call_args, **call_kwargs)
    finally:
        hm._fill_hidden, hm._under_the_robot, hm._run_on, pc.bridge_columns = (original_fill, original_under,
                                                                              original_run, original_bridge)

    print(f"{len(world.boxes)} box(es)")
    for step in steps:
        if step["step"] == "fill_hidden" and step["filled"]:
            yaw, along_low, along_cell, across_low, across_cell = step["grid"]
            print(f"fill_hidden: {step['filled']} cell(s), cell {along_cell:.1f} x {across_cell:.1f} mm, yaw "
                  f"{math.degrees(yaw):.1f} deg")
            for (r, c), top, seen in zip(step["cells"], step["tops"], step["seen"]):
                a = along_low + (c + 0.5) * along_cell
                b = across_low + (r + 0.5) * across_cell
                x, y = math.cos(yaw) * a - math.sin(yaw) * b, math.sin(yaw) * a + math.cos(yaw) * b
                print(f"   cell ({x:.0f}, {y:.0f}) filled to {top:.1f} (seen {seen:.1f})")
        elif step["step"] == "under_the_robot":
            print(f"under_the_robot: kept {step['kept']}, lowered {step['lowered']}, unfilled {step['unfilled']}")
            yaw, along_low, along_cell, across_low, across_cell = step["grid"]
            grow = 2.0 * float(step["margin"])
            for (r, c), top, bottom in zip(step["cells"], step["tops"], step["bottoms"]):
                # Where the robot stands over the kept cell's column, its grown footprint up to the box's top and the
                # guard's distance past it.
                a0, b0 = along_low + c * along_cell - grow, across_low + r * across_cell - grow
                alongs = np.arange(a0, a0 + along_cell + 2 * grow + 1e-9, 5.0)
                acrosses = np.arange(b0, b0 + across_cell + 2 * grow + 1e-9, 5.0)
                floor = float(step["floor"]) if step["floor"] is not None else float(bottom)
                heights = np.arange(floor, top + 60.0 + 1e-9, 5.0)
                ga, gb, gz = np.meshgrid(alongs, acrosses, heights, indexing="ij")
                pts = np.column_stack((math.cos(yaw) * ga.ravel() - math.sin(yaw) * gb.ravel(),
                                       math.sin(yaw) * ga.ravel() + math.cos(yaw) * gb.ravel(), gz.ravel()))
                on = np.asarray(step["robot_on"](pts), dtype=bool)
                low = float(pts[on, 2].min()) if on.any() else math.inf
                a = along_low + (c + 0.5) * along_cell
                b = across_low + (r + 0.5) * across_cell
                x, y = math.cos(yaw) * a - math.sin(yaw) * b, math.sin(yaw) * a + math.cos(yaw) * b
                print(f"   kept ({x:.0f}, {y:.0f}) top {top:.1f}: robot's lowest point over the grown column, up to "
                      f"60 mm over the top: {low:.1f}")
        elif step["step"] == "run_on" and step["ran"]:
            yaw, along_low, along_cell, across_low, across_cell = step["grid"]
            print(f"run_on: {step['ran']} cell(s)")
            for (r, c), top in zip(step["cells"], step["tops"]):
                a = along_low + (c + 0.5) * along_cell
                b = across_low + (r + 0.5) * across_cell
                x, y = math.cos(yaw) * a - math.sin(yaw) * b, math.sin(yaw) * a + math.cos(yaw) * b
                print(f"   ran ({x:.0f}, {y:.0f}) to {top:.1f}")
        elif step["step"] == "bridge" and step["columns"]:
            print(f"bridge: {len(step['columns'])} column(s)")
            for low, high, cells in step["columns"]:
                print(f"   {np.round(low, 1)} .. {np.round(high, 1)}, {cells} hidden cell(s)")

    for box in world.boxes:
        if not getattr(box, "hidden_cells", 0):
            continue
        centre = np.asarray(box.center_mm, dtype=np.float64)
        if box.rotation is not None:
            rotation = np.asarray(box.rotation, dtype=np.float64).reshape(3, 3)
        else:
            c, s = math.cos(float(box.yaw_rad)), math.sin(float(box.yaw_rad))
            rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        half = np.asarray(box.dims_mm, dtype=np.float64) / 2.0
        line = f"box {box.name}: centre {np.round(centre, 1)}, size {np.round(2 * half, 1)}, {box.hidden_cells} hidden"
        if self_body is not None:
            pts = _box_corners_grid(centre, rotation, half)
            inside = self_body.on_itself(pts, within_mm=0.0, from_frame=0)
            line += f"; the robot inside it at {int(inside.sum())} of {pts.shape[0]} grid point(s)"
            if inside.any():
                line += f", lowest {pts[inside, 2].min():.1f}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
