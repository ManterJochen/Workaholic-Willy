"""Fit the planning-only sphere model of an arm or a hand the way NVIDIA's own UR configs are made, in the project venv.

    .venv/Scripts/python.exe scripts/curobo/fit_planning_spheres.py --nvidia            # NVIDIA's ur10e, measured
    .venv/Scripts/python.exe scripts/curobo/fit_planning_spheres.py --arm ur10           # report, write nothing
    .venv/Scripts/python.exe scripts/curobo/fit_planning_spheres.py --arm ur10 --write
    .venv/Scripts/python.exe scripts/curobo/fit_planning_spheres.py --hand robotiq_hande --write

The owner, 2026-10-08: "wir müssen CuRobo richtig verwenden ... die Kugeln richtig fitten, wie es bei den anderen
Modellen ist ... nicht direkt alles von NVIDIA nehmen, sondern schauen, wie es bei den anderen UR-Robotern ist, und eine
Art Imitation machen". This is that imitation, as a planning-only model: the planner plans with it, and every check, the
evidence and the exact guard stay on the cover fit (``safety/planning/_curobo_planning.py``).

What NVIDIA's UR configs do, read in the pinned checkout (``content/configs/robot/ur10e.yml``, the same in ``.xrdf``)
and measured against the ur10e bundle by ``--nvidia``:

* a few spheres per link, 20 in all: one on the shoulder, seven along the upper arm, seven along the forearm, one on
  each of wrist_1 and wrist_2, two on wrist_3, where the second (70 mm, 60 mm out along the flange axis) stands in for
  a tool; a tool0 sphere of negative radius holds a slot;
* centres on the link's own axis: the upper arm's on the line between the shoulder and the elbow joint at the arm's
  offset (180 mm), the forearm's on the line from the elbow to wrist_1 (30 mm off), the wrists' on the joint;
* radii about the link's own: 50 mm along the e-series tubes and on the wrists, 90 and 70 mm at the upper arm's two
  ends, where the joint housings stand;
* holes accepted: measured on the ur10e's own meshes, a surface point lies up to 71 mm outside every sphere (the
  shoulder housing), 51 to 70 mm on the arm, 31 to 32 mm on the wrists, and the spheres reach up to 29 mm past the
  links (130 mm on the tool stand-in);
* buffers: ``collision_sphere_buffer`` 0, ``self_collision_buffer`` 70 mm on the shoulder (it stands for the robot's
  base as well) and 0 elsewhere; the ``.xrdf`` adds 10 mm against the world on every link;
* the ignored self pairs: every link and its neighbour, and wrist_1 against wrist_3; 83 sphere pairs in all.

The imitation keeps the structure and fits the size. Each body of the arm's committed bundle (the meshes the exact guard
judges, in their DH frames) and of the hand's is covered by the repository's cover fit used as a tool
(``_cover_fit._march``: centres marched in from the surface to where the body is thickest, which on a tube is its axis,
radius the thickness plus a reach) with a greedy that accepts a sample within a hole allowance outside its sphere
(``_cover_fit._greedy``), so the few large spheres along each link's axis are kept and the small ones a complete cover
spends on edges and bolts are not. Per body the plan below says the reach and the hole it is fitted at, as
``fit_cover_spheres.py`` says the evidence model's, and a body that needs more spheres than its ceiling there walks on
to the coarser rungs of ``COARSER``, as the cover fit walks its ladder; the record carries the rung it was fitted at and
what a fresh sample measures of both. The ignored pairs are the composition's (``planning_model.py``): the pairs the exact guard decides, every sample of every
plan, are left to it.

Written, an arm map is ``src/robot/safety/planning/robot/{arm}_arm_lean_spheres.yml``, in the bundle's DH frames with the
DH rows the sidecar places it with, and a hand map ``{hand}_gripper_lean_spheres.yml``, flat under ``tool0`` in the hand
map's own frame with its origin, as the cover maps are written. No GPU and no simulator: the project venv's trimesh.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cover_fit import _greedy, _march  # noqa: E402
from _mesh_body import MeshBody, load_meshes  # noqa: E402

DATA = REPO / "src/robot/safety/data"
MAPS = REPO / "src/robot/safety/planning/robot"
#: The model these maps are named after (``planning_model.LEAN``).
MODEL = "lean"
#: Surface samples per body. A lean map accepts holes of 10 mm and more, so the claim needs no finer sampling than this,
#: and the fresh draw the measurement is taken on is as large.
SAMPLES = 20_000
#: Candidate seeds per round, as the cover fit draws them.
SEEDS_PER_ROUND = 4096
#: Rounds before a body that still has samples uncovered is given up: a body to look at, not a number to raise.
MAX_ROUNDS = 50


@dataclass(frozen=True)
class Ask:
    """What one body is asked for: its frame, how far a sphere may reach past it, how far a surface point may lie
    outside every sphere, a ceiling on the count, and why."""

    frame: int
    reach_mm: float
    hole_mm: float
    cap: int
    why: str


# The arm, per body. NVIDIA's ur10e gives the shoulder one sphere, the tubes seven each and every wrist one; the CB3's
# housings are larger than the e-series' and its tubes thicker, so the reach a sphere may have past them is what sets the
# count here, and the hole is bounded where NVIDIA's are not. Measured over a grid of reach and hole (2026-10-09):
#     shoulder   30/20 mm: 4 spheres     40/15: 1          upper_arm 30/20: 9    40/15: 5     30/15: 11
#     forearm    25/15:   10             30/15: 8          wrist_1/2 25/15: 1 each (one sphere, NVIDIA's own count)
#     wrist_3    30/15:    1 (hole 6.8)
# A forearm, a wrist and the hand meet a bin; a shoulder and an upper arm almost never, so those carry the larger hole.
ARM_PLAN = {
    "shoulder": Ask(1, 30.0, 20.0, 12, "NVIDIA's one shoulder sphere; the CB3's housings take four at 30 mm reach"),
    "upper_arm": Ask(2, 30.0, 20.0, 16, "NVIDIA's chain of seven along the arm's axis; nine at 30 mm reach"),
    "forearm": Ask(3, 25.0, 15.0, 16, "NVIDIA's chain of seven; the forearm meets a bin wall, so 25 mm reach"),
    "wrist_1": Ask(4, 25.0, 15.0, 4, "NVIDIA's one sphere on the joint"),
    "wrist_2": Ask(5, 25.0, 15.0, 4, "NVIDIA's one sphere on the joint"),
    "wrist_3": Ask(6, 30.0, 15.0, 4, "NVIDIA's flange sphere; the hand and the camera are bodies of their own"),
}

# The hand. NVIDIA's own gripper (franka.yml) is a row of spheres along the palm and two on each finger, 11 mm, the
# radius of the finger itself. The fingers go into the bin, so they are held to the reach the evidence model's hand has
# (12 mm), and the housing behind them, which passes a bin's rim on a deep grasp, to 15. Measured on the Hand-E
# (2026-10-09): the housing takes 3 spheres at 20/10 mm, 6 at 15/8, 12 at 12/8; a finger 3 at 12/8 and 5 at 10/5. The
# 2F-85's and the EGU-50's housings take more than 24 at 15/8, and walk on to the next rung of COARSER.
HAND_PLAN = {
    "gripper": Ask(6, 15.0, 8.0, 24, "the housing: a row along the approach, as franka's palm"),
    "lfinger": Ask(6, 12.0, 8.0, 6, "one jaw, at the evidence model's reach"),
    "rfinger": Ask(6, 12.0, 8.0, 6, "the other jaw"),
}

#: Where a body does not fit its ceiling at the reach and hole it is asked for, the rungs it walks on, reach and hole in
#: millimetres, as the cover fit walks its ladder; the record says the rung it was fitted at.
COARSER = ((20.0, 10.0), (25.0, 12.0), (30.0, 15.0), (40.0, 20.0))


def lean_cover(body: MeshBody, ask: Ask, *, seed: int = 0) -> "tuple[np.ndarray, np.ndarray] | None":
    """The few large spheres that hold every sample of ``body`` within ``ask.hole_mm`` outside one of them, or ``None``
    where more than ``ask.cap`` would be needed.

    The cover fit's candidates: each sample marched in along its normal to where the body is thickest, its sphere that
    thickness plus ``ask.reach_mm``, so no sphere reaches past the body by more. Its greedy, with a sample counted held
    within the hole allowance outside a sphere rather than inside it by the sample spacing.
    """
    reach, hole = ask.reach_mm / 1000.0, ask.hole_mm / 1000.0
    points, normals = body.points, body.normals
    rng = np.random.default_rng(seed)
    span = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    covered = np.zeros(len(points), dtype=bool)
    centres: list[np.ndarray] = []
    radii: list[float] = []
    pool = rng.choice(len(points), size=min(SEEDS_PER_ROUND, len(points)), replace=False)
    for _ in range(MAX_ROUNDS):
        found, sizes, _flat = _march(points[pool], normals[pool], distance=body.distance, inside=body.inside,
                                     reach_m=reach, spacing_m=body.spacing_m, span_m=0.5 * span)
        # _greedy counts a sample covered within `radius - spacing`; handed `radius + hole + spacing`, that is the hole.
        picked = _greedy(points, covered, found, sizes + hole + body.spacing_m, spacing_m=body.spacing_m,
                         cap=ask.cap - len(centres) + 1)
        centres.extend(found[index] for index in picked)
        radii.extend(float(sizes[index]) for index in picked)
        if len(centres) > ask.cap:
            return None
        left = np.nonzero(~covered)[0]
        if len(left) == 0:
            break
        pool = left if len(left) <= SEEDS_PER_ROUND else rng.choice(left, size=SEEDS_PER_ROUND, replace=False)
    else:
        raise SystemExit(f"samples still outside every sphere after {MAX_ROUNDS} rounds: look at the body")
    return along_the_axis(np.asarray(centres), np.asarray(radii))


def along_the_axis(centres: np.ndarray, radii: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """The spheres in order along their own principal axis, as NVIDIA lists a link's, one end to the other."""
    if len(centres) < 2:
        return centres, radii
    middle = centres.mean(axis=0)
    axis = np.linalg.svd(centres - middle, full_matrices=False)[2][0]
    order = np.argsort((centres - middle) @ axis, kind="stable")
    return centres[order], radii[order]


def fit_bodies(bundle: Path, plan: "dict[str, Ask]", *, seed: int = 0) -> "tuple[dict, list[dict]]":
    """Every body of one bundle at its ask, measured again on a fresh sample: the map and one row per body."""
    spheres: dict = {}
    rows: list[dict] = []
    for name, ask in plan.items():
        started = time.time()
        body = MeshBody(load_meshes(bundle, [name]), samples=SAMPLES, seed=seed)
        rungs = [ask] + [replace(ask, reach_mm=reach, hole_mm=hole) for reach, hole in COARSER
                         if reach > ask.reach_mm or hole > ask.hole_mm]
        fitted = None
        for rung in rungs:
            fitted = lean_cover(body, rung, seed=seed)
            if fitted is not None:
                ask = rung
                break
            print(f"  {name:10s} more than {rung.cap} spheres at {rung.reach_mm:g} mm reach and a {rung.hole_mm:g} mm "
                  "hole: the next rung", flush=True)
        if fitted is None:
            raise SystemExit(f"{name}: no rung up to {rungs[-1].reach_mm:g} mm reach holds it in {ask.cap} spheres: "
                             f"{ask.why}; look at the body")
        centres, radii = fitted
        fresh = body.measure(centres, radii, points=body.fresh_points())
        spheres[name] = {"frame": ask.frame, "spheres": [
            {"center": [round(float(v), 6) for v in centre], "radius": round(float(radius), 6)}
            for centre, radius in zip(centres, radii)]}
        rows.append({
            "body": name, "frame": ask.frame, "why": ask.why, "spheres": len(radii),
            "reach_mm": ask.reach_mm, "hole_mm": ask.hole_mm,
            "radii_mm": [round(float(radius) * 1000.0, 1) for radius in radii],
            "fresh_uncovered_max_mm": round(fresh.uncovered_max_mm, 3), "fresh_reach_max_mm": round(fresh.reach_max_mm, 3),
            "fresh_samples": fresh.samples, "sample_spacing_mm": round(body.spacing_m * 1000.0, 3),
            "seconds": round(time.time() - started, 1),
        })
        print(f"  {name:10s} {len(radii):2d} sphere(s), radii {rows[-1]['radii_mm']} mm  |  fresh: {fresh.render()}  "
              f"({rows[-1]['seconds']:.0f} s)", flush=True)
    return spheres, rows


def nvidia_ur10e() -> "list[dict]":
    """NVIDIA's ur10e spheres placed into the ur10e bundle's DH frames and measured with the same ruler, per link."""
    import _ur_description as urd  # noqa: PLC0415 (beside this script, stdlib plus PyYAML)
    from _arm_spheres import link_frames, place_spheres  # noqa: PLC0415

    config = REPO / "ext_deps" / "curobo" / "curobo" / "content" / "configs" / "robot" / "ur10e.yml"
    if not config.is_file():
        raise SystemExit(f"no pinned cuRobo checkout with NVIDIA's ur10e.yml at {config}")
    kinematics = yaml.safe_load(config.read_text(encoding="utf-8"))["robot_cfg"]["kinematics"]
    place = link_frames(urd.render_urdf("ur10e"), urd.dh_rows("ur10e"))
    rows = []
    for body, frame in (("shoulder", 1), ("upper_arm", 2), ("forearm", 3), ("wrist_1", 4), ("wrist_2", 5),
                        ("wrist_3", 6)):
        placed = place_spheres(kinematics["collision_spheres"][f"{body}_link"],
                               np.linalg.inv(place(f"{body}_link", frame)))
        centres = np.asarray([sphere["center"] for sphere in placed])
        radii = np.asarray([sphere["radius"] for sphere in placed])
        mesh = MeshBody(load_meshes(DATA / "ur10e_collision_meshes.npz", [body]), samples=SAMPLES, seed=0)
        fidelity = mesh.measure(centres, radii, points=mesh.fresh_points())
        rows.append({"body": body, "spheres": len(radii), "radii_mm": [round(r * 1000.0) for r in radii.tolist()],
                     "centres_mm": [[round(v * 1000.0) for v in centre] for centre in centres.tolist()],
                     "uncovered_max_mm": round(fidelity.uncovered_max_mm, 1),
                     "reach_max_mm": round(fidelity.reach_max_mm, 1)})
        print(f"  {body:10s} {len(radii)} sphere(s), radii {rows[-1]['radii_mm']} mm  |  {fidelity.render()}",
              flush=True)
    return rows


def _dh_rows(arm: str) -> "list[list[float]]":
    import _ur_description as urd  # noqa: PLC0415

    return [[float(v) for v in row] for row in urd.dh_rows(arm)]


def _origin(bundle: Path) -> str:
    """Whether the hand's bundle starts at the flange or at its mounting face, by the one rule every reader asks."""
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location("willy_gripper_spheres", MAPS / "gripper_spheres.py")
    assert spec is not None and spec.loader is not None  # noqa: S101
    module = importlib.util.module_from_spec(spec)
    sys.modules["willy_gripper_spheres"] = module
    spec.loader.exec_module(module)
    return str(module.bundle_origin(bundle))


def provenance(*, what: str, source: Path, rows: "list[dict]", seed: int) -> dict:
    return {
        "what": what,
        "model": MODEL,
        "source": source.name,
        "generated_by": "scripts/curobo/fit_planning_spheres.py",
        "recipe": ("NVIDIA's UR pattern, fitted: a few spheres along each body's axis, each the body's thickness there "
                   "plus the reach asked, held to a hole allowance instead of a complete cover; a planning model only, "
                   "every check, the evidence and the exact guard stay on the cover fit"),
        "asked_for": "scripts/curobo/fit_planning_spheres.py ARM_PLAN / HAND_PLAN, per body",
        "seed": seed,
        "bodies": rows,
    }


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Fit a planning-only sphere model the way NVIDIA's UR configs are made.")
    parser.add_argument("--arm", default=None, help="an arm model with a committed bundle, e.g. ur10")
    parser.add_argument("--hand", default=None, help="a hand with a committed bundle, e.g. robotiq_hande")
    parser.add_argument("--nvidia", action="store_true", help="measure NVIDIA's own ur10e spheres and stop")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="write the committed map; otherwise report only")
    parser.add_argument("--json-out", default=None, help="where to append one JSON row per body")
    args = parser.parse_args(argv)

    if args.nvidia:
        print("[nvidia] ur10e.yml placed into the ur10e bundle's DH frames", flush=True)
        rows = nvidia_ur10e()
        print(f"[nvidia] {sum(row['spheres'] for row in rows)} spheres over {len(rows)} links", flush=True)
        if args.json_out:
            with Path(args.json_out).open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps({"nvidia": "ur10e", **row}) + "\n")
        return 0
    if bool(args.arm) == bool(args.hand):
        raise SystemExit("name exactly one of --arm or --hand, or ask for --nvidia")
    if args.arm:
        bundle, plan = DATA / f"{args.arm}_collision_meshes.npz", ARM_PLAN
        out, what = MAPS / f"{args.arm}_arm_{MODEL}_spheres.yml", f"{args.arm} arm links, planning only"
    else:
        bundle, plan = DATA / f"{args.hand}_hand_meshes.npz", HAND_PLAN
        out, what = MAPS / f"{args.hand}_gripper_{MODEL}_spheres.yml", f"{args.hand} hand bodies, planning only"
    if not bundle.is_file():
        raise SystemExit(f"no bundle at {bundle}")
    print(f"[fit] {bundle.name}, the {MODEL} planning model", flush=True)
    spheres, rows = fit_bodies(bundle, plan, seed=args.seed)
    total = sum(row["spheres"] for row in rows)
    print(f"[fit] {total} spheres over {len(rows)} bodies, a hole of at most "
          f"{max(row['fresh_uncovered_max_mm'] for row in rows):.1f} mm and a reach of at most "
          f"{max(row['fresh_reach_max_mm'] for row in rows):.1f} mm on a fresh sample", flush=True)
    if args.json_out:
        with Path(args.json_out).open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"bundle": bundle.name, **row}) + "\n")
    document = {"_provenance": provenance(what=what, source=bundle, rows=rows, seed=args.seed)}
    if args.arm:
        document["_provenance"]["dh_m"] = _dh_rows(args.arm)
        document["collision_spheres"] = spheres
    else:
        flat = [sphere for block in spheres.values() for sphere in block["spheres"]]
        document["_provenance"].update({"origin": _origin(bundle),
                                        "frame": "tool0 (Y=approach, X=closing, Z=depth), metres"})
        document["collision_spheres"] = {"tool0": flat}
    if not args.write:
        print(f"[dry-run] pass --write to save {out.name}", flush=True)
        return 0
    out.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
