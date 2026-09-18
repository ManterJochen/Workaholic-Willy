"""Fit collision spheres to the committed bundles, covering every body, in the project venv.

    .venv/Scripts/python.exe scripts/curobo/fit_cover_spheres.py --arm ur5e --dry-run
    .venv/Scripts/python.exe scripts/curobo/fit_cover_spheres.py --hand schunk_egu50 --write

Why this replaces the vendor map. A planner's spheres make two errors and only one of them is safe: a hole is a false
clear, where the planner believes part of the arm is not there, and reach past the body is a false collide, which costs
picks. Isaac's Lula maps optimise a compromise and are measured afterwards, and measured here they reach 0 to 61 mm past
their own links, sit 60.8 mm off the wrong arm entirely on ur16e, and land 75.9 mm out when the description comes from
the vendor rather than from the simulator.

The fit in ``_cover_fit.py`` makes the safe half a property of the construction: every surface sample ends up inside a
sphere, or there is no fit at all. What the caller trades is count against reach, and this walks a ladder of reaches to
find the smallest that fits a per body budget.

No simulator and no GPU. The body is an oracle of two functions. The unsigned distance comes from trimesh's own
proximity query, and the inside test from the generalised winding number, which is what makes it right on the bodies
that are not watertight: 7 of 45 of them, where a ray test disagrees with the winding number by up to 22 mm.

The spheres are written in the body's own DH frame, which is the frame the bundle holds and the frame the descriptor
builder places from, so nothing here has to know how an arm is posed.

A hand nobody shipped gets its map here once its body is written (``scripts/grippers/write_hand_from_dimensions.py``,
``write_hand_from_mesh.py`` or ``bake_gripper_variant.py``). Read its trade first, then write the map:

    .venv/Scripts/python.exe scripts/curobo/fit_cover_spheres.py --hand <name> --curve
    .venv/Scripts/python.exe scripts/curobo/fit_cover_spheres.py --hand <name> --write

The plan asks every hand for a 12 mm reach within 256 spheres for the gripper and 128 per finger. A body that does not
fit says so, and says whether the reach was not tried (below twice its sample spacing) or did not fit within the cap.
A ``--cap`` or ``--reach-mm`` passed to ``--write`` is recorded in the map's provenance under ``overrides``, so a map
fitted to other numbers than the plan's says which.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _cover_fit import cover_body  # noqa: E402
from _mesh_body import DEFAULT_SAMPLES, MeshBody, load_meshes  # noqa: E402


def _display_name(existing: Path, hand: str) -> str:
    """The human name for this hand, kept from the map being replaced rather than invented here.

    A sphere map looks like any other list of numbers a year later, so the provenance names the hand in words.
    Those words were chosen by a person; a refit is not the moment to re-choose them.
    """
    if existing.is_file():
        named = (yaml.safe_load(existing.read_text(encoding="utf-8")) or {}).get("_provenance", {}).get("gripper")
        if named:
            return str(named)
    return hand


def _bundle_origin(bundle: Path) -> str:
    """Whether a hand bundle starts at the flange or at the hand's own mounting face.

    The rule has one owner, ``robot/gripper_spheres.py``, because the reader on the other side decides from it
    whether a coupling may be added, and a second opinion here would move a hand by the thickness of a plate.
    Loaded by path, because this script runs beside the cuRobo work rather than inside the package tree.
    """
    import importlib.util

    path = REPO / "src/robot/safety/planning/robot/gripper_spheres.py"
    spec = importlib.util.spec_from_file_location("willy_gripper_spheres", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["willy_gripper_spheres"] = module
    spec.loader.exec_module(module)
    return str(module.bundle_origin(bundle))

DATA = REPO / "src/robot/safety/data"


def _hand_bundle():
    """``planning/_hand_bundle.py`` by path: numpy only, the one rule every hand bundle reader and writer asks."""
    import importlib.util

    name = "willy_hand_bundle"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, REPO / "src/robot/safety/planning/_hand_bundle.py")
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def _arrays(bundle: Path) -> dict:
    import numpy as np

    with np.load(bundle, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
MAPS = REPO / "src/robot/safety/planning/robot"

@dataclass(frozen=True)
class Ask:
    """What one body is asked for: the frame it lives in, the reach it must not exceed, and a ceiling."""

    frame: int
    reach_mm: float
    #: Not a target. A body needing more than this is a body to look at, so the fit refuses instead of
    #: quietly spending the plan time.
    cap: int
    why: str


# The budget goes where the pairs bind, and which pairs bind is read rather than remembered: the retract
# table records the nearest pair of every row it chose (`robot/ur_retract.yaml`), counted as
#
#     exact meshes, nearest pair      sphere model, deepest pair
#       25  wrist_1 | wrist_3           11  hand | wrist_1_link
#       11  forearm | wrist_2            1  forearm_link | wrist_2_link
#        8  wrist_1 | gripper
#
# shoulder and upper_arm appear in no row at all, so they take the coarse end of the budget. The counts
# below are what the ur5e costs under --curve; another arm of the family may need a few more, which is
# what the cap allows for without letting a body run away.
ARM_PLAN = {
    "shoulder": Ask(1, 20.0, 128, "in no binding pair; 41 spheres at 20 mm"),
    "upper_arm": Ask(2, 30.0, 128, "in no binding pair, and the most expensive body: 320 spheres at 15 mm against 37 at 30"),
    # The ceiling is 320 because ur10e, the longest forearm in the family, does not fit under 256, and ur10
    # needs 240 at the same reach: the family growing rather than one body running away.
    "forearm": Ask(3, 15.0, 320, "forearm|wrist_2 is the nearest exact pair in 11 rows; 155 spheres at 15 mm on the ur5e, 240 on the ur10"),
    "wrist_1": Ask(4, 8.0, 256, "in every binding pair there is, exact and sphere alike; 160 spheres at 8 mm"),
    "wrist_2": Ask(5, 8.0, 256, "forearm|wrist_2; 122 spheres at 8 mm"),
    "wrist_3": Ask(6, 6.0, 128, "wrist_1|wrist_3, and cheap: 57 spheres at 6 mm"),
}

# The hand is the body that comes closest to the arm at a retract pose, so a millimetre here is worth two.
# 12 mm, from the measured curve: the EGU-50 gripper costs 149 spheres there, against 34 at 20 mm and 246
# at 10, and the map it replaces has 46 spheres, 22.6 mm of reach and an 18.9 mm hole.
HAND_PLAN = {
    "gripper": Ask(6, 12.0, 256, "the hand body nearest the arm at a retract pose"),
    "lfinger": Ask(6, 12.0, 128, "one jaw"),
    "rfinger": Ask(6, 12.0, 128, "the other jaw"),
}

#: Reaches to try, in millimetres, smallest first. The fit takes the first that meets the budget.
LADDER_MM = (2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0)

#: Surface samples per body, from the shared oracle. Coverage is a claim at the samples, checked afterwards on a
#: fresh draw with another seed.
SAMPLES = DEFAULT_SAMPLES
#: Candidate seeds per round. More is a better cover and a slower fit; rounds re-seed from whatever is left.
SEEDS_PER_ROUND = 1024


class Body(MeshBody):
    """One body of a bundle, plus the ladder walk that finds the smallest reach it can be covered at."""

    def below_sampling(self, reach_mm: float) -> bool:
        """Whether ``reach_mm`` lies below twice this body's sample spacing, where the fitter is never asked.

        The one rule the fit, the curve and a refusal read, so a rung skipped is never reported as a rung missed.
        """
        return reach_mm / 1000.0 < 2.0 * self.spacing_m

    def fit(self, *, cap: int, ladder_mm=LADDER_MM, seed: int = 0):
        """The smallest reach on the ladder whose cover fits ``cap`` spheres, or ``None``.

        A ladder of one rung is how a caller names the reach instead. On the EGU-50 gripper the same body costs
        246 spheres at 10 mm, 149 at 12 and 34 at 20, so what a cap buys is not a thing to guess at.
        """
        for reach_mm in ladder_mm:
            reach = reach_mm / 1000.0
            if self.below_sampling(reach_mm):
                continue  # below twice the sampling the fitter refuses, and it is right to
            fitted = cover_body(
                self.points, self.normals, distance=self.distance, inside=self.inside,
                reach_m=reach, spacing_m=self.spacing_m, cap=cap, seed=seed,
                seeds_per_round=SEEDS_PER_ROUND,
            )
            if fitted is not None:
                return fitted
        return None

    def curve(self, *, cap: int, ladder_mm=LADDER_MM, seed: int = 0):
        """Every rung of the ladder rather than the first that fits: the count against reach trade, measured.

        The fitter's one real choice is how many spheres to spend, and a body does not announce what that buys.
        This is the curve a person reads before picking a cap."""
        for reach_mm in ladder_mm:
            reach = reach_mm / 1000.0
            if self.below_sampling(reach_mm):
                # Never tried, which is a different answer from did not fit, and has to read as one.
                yield reach_mm, None, True
                continue
            yield reach_mm, cover_body(
                self.points, self.normals, distance=self.distance, inside=self.inside,
                reach_m=reach, spacing_m=self.spacing_m, cap=cap, seed=seed,
                seeds_per_round=SEEDS_PER_ROUND,
            ), False

    def verify(self, fitted, seed: int = 12345):
        """The same two questions on points the fit never saw, which is the only thing that says the sampling
        was fine enough."""
        return self.measure(fitted.centres_m, fitted.radii_m, points=self.fresh_points(seed))


def fit_bundle(bundle: Path, plan: "dict[str, Ask]", *, seed: int = 0,
               cap: "int | None" = None, reach_mm: "float | None" = None,
               target: str = "--hand <name> or --arm <model>") -> "tuple[dict, list[dict]]":
    """Every body of one bundle, fitted to what the plan asks of it and verified on points it never saw.

    ``cap`` and ``reach_mm`` override the plan for the whole bundle, which is how a measurement is taken. A
    map that gets committed is written from the plan. ``target`` is the ``--hand`` or ``--arm`` this bundle was
    named by, for the ``--curve`` command a refusal ends in.
    """
    spheres: dict = {}
    rows: list[dict] = []
    for name, ask in plan.items():
        started = time.time()
        asked_cap = cap if cap is not None else ask.cap
        ladder = (reach_mm,) if reach_mm else (ask.reach_mm,)
        body = Body(load_meshes(bundle, [name]), samples=SAMPLES, seed=seed)
        curve = (f".venv/Scripts/python.exe scripts/curobo/fit_cover_spheres.py {target} --bodies {name} --curve")
        if body.below_sampling(ladder[0]):
            # Never tried, which is a different answer from did not fit: a larger cap cannot help it.
            raise SystemExit(
                f"{bundle.name}/{name}: a reach of {ladder[0]:g} mm was not tried: it is below twice this body's "
                f"{body.spacing_m * 1000.0:.2f} mm sample spacing, where the fitter cannot tell a hole from a gap "
                f"between samples. Ask for a reach of at least {2.0 * body.spacing_m * 1000.0:.2f} mm; {curve} "
                f"reports every reach this body can be asked for.")
        fitted = body.fit(cap=asked_cap, seed=seed, ladder_mm=ladder)
        if fitted is None:
            raise SystemExit(
                f"{bundle.name}/{name}: a reach of {ladder[0]:g} mm does not cover this body within "
                f"{asked_cap} spheres. It is asked for that reach because: {ask.why}. Ask for more reach, or "
                f"raise the cap on purpose; {curve} reports the whole count against reach trade first.")
        checked = body.verify(fitted)
        spheres[name] = {
            "frame": ask.frame,
            "spheres": [{"center": [round(float(v), 6) for v in centre], "radius": round(float(radius), 6)}
                        for centre, radius in zip(fitted.centres_m, fitted.radii_m)],
        }
        rows.append({
            "body": name, "frame": ask.frame, "why": ask.why, **fitted.to_dict(),
            # The fit guarantees its own samples by construction, so what it is worth is the same two
            # questions asked of points it never saw. Kept apart from the fit's own numbers on purpose.
            "fresh_uncovered_max_mm": checked.uncovered_max_mm, "fresh_reach_max_mm": checked.reach_max_mm,
            "fresh_samples": checked.samples,
            "sample_spacing_mm": round(body.spacing_m * 1000.0, 3), "seconds": round(time.time() - started, 1),
        })
        print(f"  {name:10s} {fitted.render()}  |  fresh: {checked.render()}"
              f"  ({rows[-1]['seconds']:.0f} s)", flush=True)
    return spheres, rows


def walk_curve(bundle: Path, plan: "dict[str, Ask]", *, cap: int, seed: int = 0) -> 'list[dict]':
    """Each body over the whole ladder, so the count against reach trade is measured rather than assumed."""
    rows: list[dict] = []
    for name in plan:
        body = Body(load_meshes(bundle, [name]), samples=SAMPLES, seed=seed)
        print(f"  {name} (sample spacing {body.spacing_m * 1000.0:.2f} mm)", flush=True)
        for reach_mm, fitted, skipped in body.curve(cap=cap, seed=seed):
            started = time.time()
            if fitted is None:
                why = (f"not tried: below twice the {body.spacing_m * 1000.0:.2f} mm sample spacing"
                       if skipped else f"no fit within {cap} spheres")
                print(f"    reach {reach_mm:5.1f} mm  {why}", flush=True)
                rows.append({"body": name, "reach_mm": reach_mm, "spheres": None, "why": why})
                continue
            checked = body.verify(fitted)
            print(f"    reach {reach_mm:5.1f} mm  {len(fitted):4d} spheres  |  fresh: {checked.render()}",
                  flush=True)
            rows.append({
                "body": name, "reach_mm": reach_mm, **fitted.to_dict(),
                "fresh_uncovered_max_mm": checked.uncovered_max_mm, "fresh_reach_max_mm": checked.reach_max_mm,
                "seconds": round(time.time() - started, 1),
            })
    return rows


def write_arm_map(path: Path, *, provenance: dict, spheres: dict) -> None:
    """The arm map, one block per link, in each link's own DH frame. Read by the descriptor builder."""
    _write(path, {"_provenance": provenance, "collision_spheres": spheres})


def write_hand_map(path: Path, *, provenance: dict, spheres: dict, origin: str, gripper: str) -> None:
    """The hand map, in the shape the stack already reads: one flat list under ``tool0``.

    This overwrites the committed vendor map rather than sitting beside it. Three readers name this file
    (``planning/hand.py``, ``scripts/curobo/_hand_body.py``, the retract chooser), and a better map nobody reads
    is not a better map. The three bodies become one list because the hand is one body link on the flange, and
    what they were separately is kept in the provenance.

    ``origin`` comes from the bundle and never from here, because it decides whether the reader may add a
    coupling plate, and ``gripper`` is the human name the file already carried, kept rather than invented.
    """
    flat = [sphere for body in spheres.values() for sphere in body["spheres"]]
    _write(path, {
        "_provenance": {**provenance, "gripper": gripper, "origin": origin,
                        "frame": "tool0 (Y=approach, X=closing, Z=depth), metres"},
        "collision_spheres": {"tool0": flat},
    })


def provenance_of(*, what: str, source: Path, rows: list[dict], seed: int, overrides: dict) -> dict:
    """What a reader needs to know about a map to trust it or to rebuild it."""
    return {
        "what": what,
        "source": source.name,
        "generated_by": "scripts/curobo/fit_cover_spheres.py",
        "recipe": "cover fit: every surface sample inside a sphere, no sphere reaching further past the body "
                  "than the reach it was fitted at",
        "asked_for": "scripts/curobo/fit_cover_spheres.py ARM_PLAN / HAND_PLAN, per body",
        "overrides": {key: value for key, value in overrides.items() if value is not None} or None,
        "seed": seed,
        "bodies": rows,
    }


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(document, sort_keys=False, default_flow_style=False), encoding="utf-8")
    print(f"wrote {path}", flush=True)


def main(argv: "list[str] | None" = None) -> int:
    parser = argparse.ArgumentParser(description="Fit cover spheres to a committed collision bundle.")
    parser.add_argument("--arm", default=None, help="an arm model, e.g. ur5e")
    parser.add_argument("--hand", default=None, help="a hand, e.g. schunk_egu50")
    parser.add_argument("--cap", type=int, default=None,
                        help="override the planned ceiling for every body")
    parser.add_argument("--reach-mm", type=float, default=None,
                        help="override the planned reach for every body")
    parser.add_argument("--bodies", default=None, help="only these bodies, comma separated")
    parser.add_argument("--curve", action="store_true",
                        help="report every rung of the ladder instead of fitting the first that fits")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true", help="write the committed map; otherwise report only")
    parser.add_argument("--json-out", default=None, help="where to append one JSON row per body")
    args = parser.parse_args(argv)

    if bool(args.arm) == bool(args.hand):
        raise SystemExit("name exactly one of --arm or --hand")

    if args.arm:
        bundle, plan = DATA / f"{args.arm}_collision_meshes.npz", dict(ARM_PLAN)
        out, what = MAPS / f"{args.arm}_arm_spheres.yml", f"{args.arm} arm links"
    else:
        bundle, plan = DATA / f"{args.hand}_hand_meshes.npz", dict(HAND_PLAN)
        # The name the stack already reads. A map under a new name would be measured, committed and ignored.
        out, what = MAPS / f"{args.hand}_gripper_spheres.yml", f"{args.hand} hand bodies"
    if not bundle.is_file():
        raise SystemExit(f"no bundle at {bundle}")
    if args.hand:
        # A hand the guard refuses to compose gets no map either: the planner would model a hand the guard cannot.
        refused = _hand_bundle().hand_bundle_refusal(_arrays(bundle), name=bundle.name)
        if refused is not None:
            raise SystemExit(refused)
    if args.bodies:
        wanted = [name.strip() for name in args.bodies.split(",")]
        unknown = [name for name in wanted if name not in plan]
        if unknown:
            raise SystemExit(f"{unknown} is not a body of this bundle; it carries {sorted(plan)}")
        plan = {name: plan[name] for name in wanted}

    asked = (f"at most {args.cap} spheres per body" if args.cap else "the budget each body is planned for")
    if args.reach_mm:
        asked += f", all at a reach of {args.reach_mm:g} mm"
    print(f"[fit] {bundle.name}, {asked}", flush=True)
    if args.curve:
        rows = walk_curve(bundle, plan, cap=args.cap or 400, seed=args.seed)
        if args.json_out:
            path = Path(args.json_out)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps({"bundle": bundle.name, "curve": True, **row}) + "\n")
        return 0

    target = f"--arm {args.arm}" if args.arm else f"--hand {args.hand}"
    spheres, rows = fit_bundle(bundle, plan, seed=args.seed, cap=args.cap, reach_mm=args.reach_mm, target=target)
    total = sum(row["spheres"] for row in rows)
    print(f"[fit] {total} spheres over {len(rows)} bodies, reaching at most "
          f"{max(row['fresh_reach_max_mm'] for row in rows):.1f} mm past the geometry", flush=True)

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps({"bundle": bundle.name, **row}) + "\n")

    provenance = provenance_of(what=what, source=bundle, rows=rows, seed=args.seed,
                               overrides={"cap": args.cap, "reach_mm": args.reach_mm})
    if args.write:
        if args.bodies:
            raise SystemExit(
                "--write needs every body of the bundle: a map holding some of them would leave the rest unguarded")
        if args.arm:
            write_arm_map(out, provenance=provenance, spheres=spheres)
        else:
            write_hand_map(out, provenance=provenance, spheres=spheres, origin=_bundle_origin(bundle),
                           gripper=_display_name(out, args.hand))
    else:
        print(f"[dry-run] pass --write to save {out.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
