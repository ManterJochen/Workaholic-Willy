"""Judge any collision sphere map against the exact meshes it stands for, with one ruler.

    .venv/Scripts/python.exe scripts/curobo/measure_sphere_map.py --hand schunk_egu50
    .venv/Scripts/python.exe scripts/curobo/measure_sphere_map.py --hand schunk_egu50 --map <a fitted map>
    .venv/Scripts/python.exe scripts/curobo/measure_sphere_map.py --arm ur10 --map willy_ur10.yml

It is separate from the fitter because a claim that one map is better than another is worth nothing unless one ruler
measured both. This reads whatever shape a map is written in, the vendor's flat list under ``tool0``, a fitted map with
a block per body, or a built cuRobo descriptor, and asks the same two questions of all of them:

* the largest hole: how far a surface point can lie outside every sphere. Positive is a false clear, the unsafe error,
  where the planner believes part of the robot is not there.
* the furthest reach: how far a sphere's surface lies past the body. A false collide, which costs picks and reachable
  workspace and nothing worse.

A hand is measured as one body, because its three parts share the flange frame and a sphere leaving the gripper for the
finger has not left the hand. Arm links each sit in their own DH frame, so they are measured one at a time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _mesh_body import DEFAULT_DIRECTIONS, DEFAULT_SAMPLES, MeshBody, load_meshes  # noqa: E402

DATA = REPO / "src/robot/safety/data"
MAPS = REPO / "src/robot/safety/planning/robot"

#: The three parts of a hand, measured together as one body in the flange frame.
HAND_BODIES = ("gripper", "lfinger", "rfinger")
#: Arm link names as a cuRobo descriptor spells them, against the body each stands for in the bundle.
ARM_LINKS = {
    "shoulder_link": "shoulder", "upper_arm_link": "upper_arm", "forearm_link": "forearm",
    "wrist_1_link": "wrist_1", "wrist_2_link": "wrist_2", "wrist_3_link": "wrist_3",
}


def read_spheres(document: dict, keys: "list[str] | None" = None) -> "dict[str, tuple[np.ndarray, np.ndarray]]":
    """Centres and radii per key, from any of the three shapes a map in this repository is written in.

    A vendor hand map is one flat list under ``tool0``; a fitted map is a block per body carrying ``spheres``; a built
    descriptor keeps the same lists under ``robot_cfg.kinematics.collision_spheres``. A descriptor whose spheres are a
    path to another file is refused by name rather than read as empty.
    """
    block = document.get("collision_spheres")
    if block is None:
        block = document.get("robot_cfg", {}).get("kinematics", {}).get("collision_spheres")
    if isinstance(block, str):
        raise SystemExit(f"this map's spheres are a path ({block!r}), not a map: measure the file it names")
    if not isinstance(block, dict):
        raise SystemExit("no collision_spheres in this document")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for key, value in block.items():
        if keys is not None and key not in keys:
            continue
        listed = value["spheres"] if isinstance(value, dict) else value
        centres = np.asarray([sphere["center"] for sphere in listed], dtype=np.float64)
        radii = np.asarray([sphere["radius"] for sphere in listed], dtype=np.float64)
        out[key] = (centres, radii)
    return out


def measure_hand(hand: str, map_path: Path, *, samples: int, directions: int, seed: int) -> "list[dict]":
    """The whole hand as one body, against every sphere the map holds, whatever key they are filed under."""
    bundle = DATA / f"{hand}_hand_meshes.npz"
    if not bundle.is_file():
        raise SystemExit(f"no bundle at {bundle}")
    body = MeshBody(load_meshes(bundle, list(HAND_BODIES)), samples=samples, seed=seed)

    per_key = read_spheres(yaml.safe_load(map_path.read_text(encoding="utf-8")))
    centres = np.concatenate([held[0] for held in per_key.values()])
    radii = np.concatenate([held[1] for held in per_key.values()])
    fidelity = body.measure(centres, radii, directions=directions)
    print(f"  {hand:16s} {fidelity.render()}", flush=True)
    return [{"body": hand, "keys": sorted(per_key), **fidelity.to_dict()}]


def measure_arm(arm: str, map_path: Path, *, samples: int, directions: int, seed: int) -> "list[dict]":
    """Each arm link against its own mesh, because each sits in a different DH frame."""
    bundle = DATA / f"{arm}_collision_meshes.npz"
    if not bundle.is_file():
        raise SystemExit(f"no bundle at {bundle}")
    document = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    per_key = read_spheres(document)

    rows: list[dict] = []
    for key, (centres, radii) in sorted(per_key.items()):
        name = ARM_LINKS.get(key, key)
        if name not in {body for body in ARM_LINKS.values()}:
            print(f"  {key:16s} skipped: not an arm link this bundle carries", flush=True)
            continue
        body = MeshBody(load_meshes(bundle, [name]), samples=samples, seed=seed)
        fidelity = body.measure(centres, radii, directions=directions)
        print(f"  {name:16s} {fidelity.render()}", flush=True)
        rows.append({"body": name, "keys": [key], **fidelity.to_dict()})
    if not rows:
        raise SystemExit(f"{map_path.name} holds no sphere list for any link of {arm}")
    return rows


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Measure a collision sphere map against the exact meshes.")
    parser.add_argument("--arm", default=None, help="an arm model, e.g. ur10")
    parser.add_argument("--hand", default=None, help="a hand, e.g. schunk_egu50")
    parser.add_argument("--map", default=None, help="the map to judge; defaults to the committed one for a hand")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--directions", type=int, default=DEFAULT_DIRECTIONS, help="rays per sphere when reading reach")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json-out", default=None, help="where to append one JSON row per body")
    args = parser.parse_args(argv)

    if bool(args.arm) == bool(args.hand):
        raise SystemExit("name exactly one of --arm or --hand")
    if args.map:
        map_path = Path(args.map)
        if not map_path.is_file():
            map_path = MAPS / args.map
    elif args.hand:
        map_path = MAPS / f"{args.hand}_gripper_spheres.yml"
    else:
        raise SystemExit("--arm needs --map: arm spheres live in a descriptor built on the box")
    if not map_path.is_file():
        raise SystemExit(f"no map at {map_path}")

    print(f"[measure] {map_path.name} against {args.arm or args.hand}, {args.samples} samples, "
          f"{args.directions} rays per sphere", flush=True)
    kwargs = {"samples": args.samples, "directions": args.directions, "seed": args.seed}
    rows = measure_arm(args.arm, map_path, **kwargs) if args.arm else measure_hand(args.hand, map_path, **kwargs)

    holes = [row for row in rows if row["uncovered_max_mm"] > 0.0]
    print(f"[measure] {sum(row['spheres'] for row in rows)} spheres, worst hole "
          f"{max(row['uncovered_max_mm'] for row in rows):+.2f} mm, worst reach "
          f"{max(row['reach_max_mm'] for row in rows):.2f} mm, {len(holes)} of {len(rows)} bodies with a hole",
          flush=True)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"map": map_path.name, "model": args.arm or args.hand, **row}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
