"""Registering a world of boxes in one call fills the planner's storage exactly as ``update_world`` does, faster.

Part one runs cuRobo's own storage in this process, with the cuRobo environment's interpreter, on the GPU or the CPU::

    <cuRobo-env-python> scripts/curobo/probe_world_in_place.py                      # cuRobo's storage on the GPU
    CUDA_VISIBLE_DEVICES= <cuRobo-env-python> scripts/curobo/probe_world_in_place.py --device cpu

Two scene stores are built the way the sidecar builds its planner's (the boot table, ``collision_cache`` box slots),
and a run of worlds is registered into both, one through ``load_collision_model`` (what ``MotionPlanner.update_world``
calls) and one through ``_curobo_world.register_boxes_in_place``: 129 boxes as the owner's cell holds them, 193, 65,
2, none, 129 again, each turned and sized differently. After every world the two stores are compared bit for bit:
the sizes, the inverse poses, the enable flags, the count, the names and the scene each keeps. Then a world one box
larger than the slots: refused in place before any slot is written, where the box-by-box way writes the first boxes
and raises. Then both ways are timed over the 129 boxes. Exit 0 when every comparison holds.

Part two runs two sidecars through the client, the project venv's interpreter, on the GPU::

    .venv/Scripts/python.exe scripts/curobo/probe_world_in_place.py --sidecars [--arm ur10]

one started with ``WILLY_CUROBO_WORLD_IN_PLACE`` and one without, the same bare arm descriptor, the same world of 129
boxes, and ``check_js`` with every refused sample reported over 1,000 configurations, in four batches: the reports have
to be the same row for row, and the round trip of ``set_world`` is timed both ways. Each starts a sidecar of its own:
stop the console and every other planner first.

Nothing here moves anything, and nothing is written but the probe's own JSON (``--json``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[2]
PLANNING = REPO / "src" / "robot" / "safety" / "planning"

#: The worlds of part one, in box counts: the owner's cell (129 at 2026-10-08's refreshes), the most it held on
#: 2026-10-07 (193), the fewest of a busy look (65), a bench alone (2), nothing, and the cell's again after nothing.
WORLDS = (129, 193, 65, 2, 0, 129)
SLOTS = 200
TIMED_ROUNDS = 50


def _boxes(count: int, seed: int) -> "list[dict[str, Any]]":
    """``count`` boxes as a set_world request carries them: turned about z, sized like the camera world's columns."""
    import random  # noqa: PLC0415

    rng = random.Random(seed)
    out = []
    for index in range(count):
        yaw = rng.uniform(-math.pi, math.pi)
        out.append({"name": f"seen_{seed:02d}_{index:03d}",
                    "dims_m": [rng.uniform(0.01, 0.12), rng.uniform(0.01, 0.12), rng.uniform(0.005, 0.15)],
                    "pose": [rng.uniform(-0.5, 0.4), rng.uniform(-0.9, -0.2), rng.uniform(0.0, 0.2),
                             math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)]})
    return out


def _scene(boxes: "list[dict[str, Any]]") -> "dict[str, Any]":
    """The dict the sidecar's set_world branch hands ``SceneCfg.create``."""
    return {"cuboid": {box["name"]: {"dims": list(box["dims_m"]), "pose": list(box["pose"])} for box in boxes}}


def storage_part(device: str) -> "tuple[bool, dict[str, Any]]":
    """Part one: cuRobo's own storage, both ways, bit for bit; and the timing."""
    import torch  # noqa: PLC0415
    from curobo._src.geom.collision.collision_scene import (  # noqa: PLC0415
        SceneCollisionCfg,
        create_scene_collision,
    )
    from curobo._src.geom.types import SceneCfg  # noqa: PLC0415
    from curobo._src.types.device_cfg import DeviceCfg  # noqa: PLC0415

    sys.path.insert(0, str(PLANNING))
    from _curobo_world import register_boxes_in_place  # noqa: PLC0415

    table = {"table": {"dims": [1.6, 1.6, 0.05], "pose": [0.0, 0.0, -0.026, 1, 0, 0, 0]}}
    device_cfg = DeviceCfg(device=torch.device(device))

    def store() -> Any:
        return create_scene_collision(SceneCollisionCfg(device_cfg=device_cfg, scene_model=SceneCfg.create(
            {"cuboid": table}), cache={"cuboid": SLOTS}))

    def sync() -> None:
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    today, in_place = store(), store()
    held = True
    rows: list[dict[str, Any]] = []
    for step, count in enumerate(WORLDS):
        boxes = _boxes(count, seed=step)
        today.load_collision_model(SceneCfg.create(_scene(boxes)))
        register_boxes_in_place(in_place, SceneCfg.create(_scene(boxes)), None)
        sync()
        a, b = today.data.cuboids, in_place.data.cuboids
        same = {name: bool(torch.equal(getattr(a, name), getattr(b, name)))
                for name in ("dims", "inv_pose", "enable", "count")}
        same["names"] = a.names == b.names
        same["scene_model_kept"] = in_place.scene_model is in_place.data.scene_model is not None
        rows.append({"boxes": count, **same})
        held = held and all(same.values())
        print(f"  world of {count:3d} box(es): " + ", ".join(f"{name} {'same' if ok else 'DIFFERENT'}"
                                                         for name, ok in same.items()), flush=True)

    too_many = _boxes(SLOTS, seed=99) + _boxes(1, seed=98)
    before = list(in_place.data.cuboids.names[0])
    refused_in_place = False
    try:
        register_boxes_in_place(in_place, SceneCfg.create(_scene(too_many)), None)
    except Exception as exc:  # noqa: BLE001 (any refusal is the answer; which one is said)
        refused_in_place = True
        print(f"  {len(too_many)} boxes in place: refused ({type(exc).__name__}: {str(exc)[:80]})", flush=True)
    untouched = in_place.data.cuboids.names[0] == before
    written_box_by_box = None
    try:
        today.load_collision_model(SceneCfg.create(_scene(too_many)))
    except Exception:  # noqa: BLE001
        written_box_by_box = sum(1 for name in today.data.cuboids.names[0] if name is not None)
    print(f"  in place: refused {refused_in_place}, every slot as it was {untouched}; box by box: "
          f"{written_box_by_box} box(es) written before it raised", flush=True)
    held = held and refused_in_place and untouched

    boxes = _boxes(129, seed=7)
    timings: dict[str, list[float]] = {"update_world": [], "in_place": []}
    for _ in range(TIMED_ROUNDS):
        for name, register in (("update_world", lambda: today.load_collision_model(SceneCfg.create(_scene(boxes)))),
                               ("in_place", lambda: register_boxes_in_place(in_place, SceneCfg.create(_scene(boxes)),
                                                                            None))):
            sync()
            began = time.perf_counter()
            register()
            sync()
            timings[name].append((time.perf_counter() - began) * 1000.0)
    p50 = {name: statistics.median(values) for name, values in timings.items()}
    print(f"  129 boxes on {device}: update_world p50 {p50['update_world']:.2f} ms, in place p50 "
          f"{p50['in_place']:.2f} ms ({p50['update_world'] / max(p50['in_place'], 1e-9):.1f}x)", flush=True)
    return held, {"device": device, "worlds": rows, "too_many": {"refused_in_place": refused_in_place,
                                                                 "untouched": untouched,
                                                                 "written_box_by_box": written_box_by_box},
                  "p50_ms": p50}


def sidecar_part(arm: str, configurations: int) -> "tuple[bool, dict[str, Any]]":
    """Part two: two sidecars, one registering in place, the same world and the same 1,000 configurations judged."""
    sys.path.insert(0, str(REPO))
    import random  # noqa: PLC0415

    from src.robot.drivers.sim.robot_models import curobo_arm_descriptor  # noqa: PLC0415
    from src.robot.safety.planning.curobo_client import CuroboPlanClient  # noqa: PLC0415

    rng = random.Random(20261009)
    world = _boxes(129, seed=11)
    samples = [[rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, 0.0), rng.uniform(-2.8, 2.8),
                rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi), rng.uniform(-math.pi, math.pi)]
               for _ in range(configurations)]
    reports: dict[str, Any] = {}
    for label, in_place in (("update_world", False), ("in_place", True)):
        client = CuroboPlanClient(robot_config=curobo_arm_descriptor(arm), scene_config=str(len(world) + 8),
                                  world_in_place=in_place)
        began = time.perf_counter()
        client.start()
        print(f"  {label}: sidecar ready in {time.perf_counter() - began:.1f} s", flush=True)
        try:
            round_trips = []
            for _ in range(20):
                began = time.perf_counter()
                count = client.set_world(world)
                round_trips.append((time.perf_counter() - began) * 1000.0)
            judged = client.judge_joints(samples, name_pairs=False)
            reports[label] = {
                "registered": count,
                "set_world_p50_ms": statistics.median(round_trips),
                "refused": [[row.index, row.bound_ok, row.self_ok, row.world_ok] for row in judged.refused],
            }
            print(f"  {label}: {count} box(es) registered, set_world p50 {reports[label]['set_world_p50_ms']:.1f} ms, "
                  f"{len(judged.refused)} of {configurations} configuration(s) refused", flush=True)
        finally:
            client.close()
    same = reports["update_world"]["refused"] == reports["in_place"]["refused"]
    counted = reports["update_world"]["registered"] == reports["in_place"]["registered"] == len(world)
    print(f"  the two reports are {'the same, row for row' if same else 'DIFFERENT'}", flush=True)
    return same and counted, reports


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--device", default="cuda:0", help="where part one keeps cuRobo's storage: cuda:0 or cpu")
    parser.add_argument("--sidecars", action="store_true", help="part two: two sidecars through the client (GPU)")
    parser.add_argument("--arm", default="ur10", help="the arm whose bare descriptor the sidecars load")
    parser.add_argument("--configurations", type=int, default=1000)
    parser.add_argument("--json", default=None, help="where to write the numbers")
    args = parser.parse_args(argv)
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    if args.sidecars:
        print(f"[sidecars] two sidecars on the bare {args.arm}, one registering in place", flush=True)
        held, report = sidecar_part(args.arm, args.configurations)
    else:
        print(f"[storage] cuRobo's own storage on {args.device}, both ways", flush=True)
        held, report = storage_part(args.device)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("[result] " + ("every comparison held" if held else "a comparison did NOT hold"), flush=True)
    return 0 if held else 1


if __name__ == "__main__":
    raise SystemExit(main())
