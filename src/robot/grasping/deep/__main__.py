"""The learned generator's operator surface: the only way to probe, train or inspect one.

    python -m src.robot.grasping.deep train-set --clouds logs/dl/clouds/v5 --out logs/dl/models/v5
    python -m src.robot.grasping.deep report --run logs/dl/models/v5
    python -m src.robot.grasping.deep inspect --artifact logs/dl/models/v1/generator.pt

Every probe and every training run is reachable from a shell here: a measurement nobody can
re-run from a shell is not a measurement anyone can check, so a new one gets a subcommand rather
than a function only a Python session can call.

Torch is imported lazily, inside the subcommands. `deep/__init__` does not re-export the network,
so that a cell importing the pick-loop seam does not pay for a 2 GB import; running this module
explicitly is the one place that cost is expected.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Callable, Sequence

from src.robot.grasping.deep.corpus.discovery import scene_files, stratified_scenes
from src.robot.grasping.deep.train.plan import UNSET

# Four defaults are `UNSET` rather than their value, which is what lets this command also be
# called from code. A recipe or tier applies only to settings the operator did not choose, and
# "did not choose" must not be decided by scanning `sys.argv` for the flag: in a process not
# started by this shell command, `sys.argv` holds the host's arguments, `--epochs` never appears,
# and the recipe silently overrides a value the caller passed explicitly.
#
# The four values live on `SetTrainingPlan`, which supplies them (epochs 12, train_units 4000,
# refit False, collapse_floor_deg 0.0), so a run without a recipe resolves to those. `--refit`
# needs `default=UNSET` on a store_true so that "not passed" stays distinguishable from an
# explicit False: `--tier smoke` has to be able to turn it off, and it cannot if absence already
# means False.

_EXIT_OK = 0
_EXIT_USAGE = 2
_EXIT_PROBLEM = 3


# `scene_files` and `stratified_scenes` live in `corpus/discovery.py`, not here, because both
# encode corpus facts a caller gets wrong by default: the walk must be recursive, and a prefix of
# a sorted list is one family. They belong where any caller can import them by name.





def _cmd_asset_consistency(args: argparse.Namespace) -> int:
    """The ceiling that bounds a per-object head, from labels alone."""

    from src.robot.grasping.deep.corpus.index import scene_family  # noqa: PLC0415
    from src.robot.grasping.deep.eval import (  # noqa: PLC0415
        format_consistency,
        measure_asset_consistency,
    )

    try:
        files = scene_files(args.clouds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    keep = tuple(f.strip() for f in str(args.families).split(",") if f.strip())
    if keep:
        before = len(files)
        files = [f for f in files if scene_family(f.stem) in keep]
        print(f"families {'+'.join(keep)}: {len(files)} of {before} scenes kept", flush=True)
        if not files:
            print("no scene left after the family filter", file=sys.stderr)
            return _EXIT_USAGE

    print(f"reading labels from {len(files)} scene(s)", flush=True)
    result = measure_asset_consistency(files, min_scenes=args.min_scenes)
    # Persist before presenting, for the reason recorded on every other command here.
    if args.out and result is not None:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(dataclasses.asdict(result), indent=2),
                                  encoding="utf-8")
    print("")
    print(format_consistency(result))
    if args.out and result is not None:
        print("")
        print(f"wrote {args.out}")
    return _EXIT_OK if result is not None else _EXIT_USAGE


def _cmd_geometric_floor(args: argparse.Namespace) -> int:
    """Plan proposal B: training-free rules on the split the trained arms saw."""
    import collections  # noqa: PLC0415
    import random  # noqa: PLC0415
    import statistics  # noqa: PLC0415

    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.sample import (  # noqa: PLC0415
        SampleSpec,
        build_sample,
        load_scene,
    )
    from src.robot.grasping.deep.corpus.index import scene_family  # noqa: PLC0415
    from src.robot.grasping.deep.eval import (  # noqa: PLC0415
        format_directions,
        format_geometric,
        object_direction_report,
        run_geometric_rules,
    )
    from src.robot.grasping.deep.corpus.index import collate  # noqa: PLC0415
    from src.robot.grasping.deep.corpus.index import units_with_grasps  # noqa: PLC0415

    try:
        files = scene_files(args.clouds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    keep = tuple(f.strip() for f in str(args.families).split(",") if f.strip())
    if keep:
        before = len(files)
        files = [f for f in files if scene_family(f.stem) in keep]
        print(f"families {'+'.join(keep)}: {len(files)} of {before} scenes kept", flush=True)
        if not files:
            print("no scene left after the family filter", file=sys.stderr)
            return _EXIT_USAGE

    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    wanted = args.train_units + args.test_units
    scenes: list[dict] = []
    units: list[tuple[int, int]] = []
    for path in files:
        scenes.append(load_scene(path))
        units = units_with_grasps(scenes, wanted)
        if len(units) >= wanted:
            break
    if len(units) < wanted:
        print(f"only {len(units)} labelled unit(s) available, wanted {wanted}", file=sys.stderr)
        return _EXIT_USAGE

    # The split is fixed so a training-free number lands on the same units a trained arm sees:
    # `random.Random(seed).shuffle` over the labelled units, sliced `[:train_units]` and
    # `[train_units:wanted]`, with the held-out batches drawn at `seed + 1000`. A number measured
    # on a different split cannot be compared with the trained arms `--reference` carries.
    device = args.device or ("cuda" if _cuda_available() else "cpu")
    spec = SampleSpec(approach_multilabel=True)
    print(f"{len(seeds)} seed(s), {args.train_units} train / {args.test_units} test units, "
          f"device {device}", flush=True)

    rules = []
    directions = []
    for seed in seeds:
        order = list(units)
        random.Random(seed).shuffle(order)
        train_units, test_units = order[: args.train_units], order[args.train_units: wanted]

        def batches(chosen, batch_seed):
            rng = np.random.default_rng(batch_seed)
            samples = [build_sample(scenes[s], rng, spec, target_instance=i) for s, i in chosen]
            return [collate(samples[k:k + args.batch_size], device=device)
                    for k in range(0, len(samples), args.batch_size)]

        # The held-out batches are built once and used by both reports. Rebuilding them for
        # the direction report would redraw the augmentation, so the two tables would describe
        # slightly different objects while appearing to describe the same ones.
        held_out = batches(test_units, seed + 1000)
        found = run_geometric_rules(batches(train_units, seed), held_out,
                                    approach_bins=spec.approach_bins, seed=seed)
        rules.extend(found)
        report = object_direction_report(held_out, approach_bins=spec.approach_bins, seed=seed)
        if report is not None:
            directions.append(report)
        best = max((r for r in found if not r.oracle), key=lambda r: r.hit, default=None)
        if best is None:
            print(f"  seed {seed}: no supervised point in the held-out units", flush=True)
        else:
            print(f"  seed {seed}: best honest rule {best.name} at {best.hit:.4f}", flush=True)

    reference = None
    if args.reference:
        try:
            arms = json.loads(Path(args.reference).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"could not read --reference: {exc}", file=sys.stderr)
            return _EXIT_USAGE
        pooled_hits: dict[str, list[float]] = collections.defaultdict(list)
        for arm in arms:
            pooled_hits[str(arm.get("architecture", "trained"))].append(float(arm["hit"]))
        reference = {name: statistics.fmean(values) for name, values in pooled_hits.items()}

    # Persist before presenting, so a formatting failure cannot cost a finished run its results.
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps({"rules": [dataclasses.asdict(r) for r in rules],
                        "directions": [dataclasses.asdict(d) for d in directions]},
                       indent=2), encoding="utf-8")
    print("")
    print(format_geometric(rules, reference))
    print("")
    print("=== where each object's best direction actually points ===")
    print(format_directions(directions))
    if args.out:
        print("")
        print(f"wrote {args.out}")
    return _EXIT_OK




def _cuda_available() -> bool:
    import torch  # noqa: PLC0415

    return bool(torch.cuda.is_available())


def _cmd_sanity(args: argparse.Namespace) -> int:
    """Steps 1 and 2 of the diagnosis, before any training question is worth asking."""

    import numpy as np  # noqa: PLC0415

    from src.robot.grasping.deep.corpus.sample import SampleSpec, build_sample, load_scene  # noqa: PLC0415, E501
    from src.robot.grasping.deep.eval.label_coverage import (  # noqa: PLC0415
        format_sampling_report,
        sampling_report,
        write_visual,
    )
    try:
        files = scene_files(args.clouds)
    except (FileNotFoundError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_USAGE
    # No sampling schedule: `levels` stays empty. The family that ships is a serialized
    # transformer with no set-abstraction levels, so there are no per-level reach columns to
    # report. What this command answers is whether a labelled contact lands anywhere near a
    # sampled point, which is a fact about the corpus rather than about a network.
    levels: list[tuple[int, float, int]] = []
    spec = SampleSpec()
    rng = np.random.default_rng(args.seed)
    reports = []
    for index, path in enumerate(files[: args.scenes]):
        scene = load_scene(path)
        sample = build_sample(scene, rng, spec)
        # The contacts are read from the scene, not from the sample. `build_sample` consumes them
        # into the graspability field and does not carry them out, so a report built only from the
        # sample could never ask whether a contact landed anywhere near a point.
        contacts = np.asarray(scene.get("contact_points_mm", np.zeros((0, 3))), dtype=np.float64)
        raw_points = np.asarray(scene.get('points_mm', np.zeros((0, 3))), dtype=np.float64)
        reports.append(sampling_report(sample, scene=path.stem, levels=levels,
                                       labelling_radius_mm=spec.graspability_radius_mm,
                                       contacts_mm=contacts, raw_points_mm=raw_points,
                                       seed=args.seed))
        if index == 0 and args.visual:
            try:
                written = write_visual(sample, contacts, args.visual, scene=path.stem)
                print(f"drew {path.stem} to {written}")
            except ImportError as exc:
                print(f"no drawing: {exc}", file=sys.stderr)

    print(format_sampling_report(reports))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps([dataclasses.asdict(r) for r in reports], indent=2), encoding="utf-8")
        print("")
        print(f"wrote {args.out}")
    return _EXIT_OK








def _cmd_inspect(args: argparse.Namespace) -> int:
    """What a weights file says about itself, before a cell is pointed at it.

    Reads the artifacts `train-set` writes, and the card beside them is what tells two `.pt`
    files apart. Every field printed here is a fact about the run, never a quality claim: which
    recipe, which corpus, whether the weights are one fold or a refit on everything, and whether
    it was a control run, which fits synthetic labels derived from an input channel and must
    never be served.
    """
    import json  # noqa: PLC0415

    path = Path(args.artifact)
    if not path.is_file():
        print(f"no such artifact: {path}", file=sys.stderr)
        return _EXIT_USAGE

    from src.robot.grasping.deep.set_artifact import load_set_generator  # noqa: PLC0415

    try:
        loaded = load_set_generator(path)
    except (KeyError, ValueError, RuntimeError) as exc:
        # The refusal is the product here. `load_set_generator` checks the kind, the version,
        # the gripper and whether the state dict fits its config; if a cell would refuse this file,
        # the customer should learn it from `inspect` rather than from a pick that returns nothing.
        print(f"this file cannot be loaded: {type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM

    net = loaded.net
    head = net.config.head
    print(f"{path.name}")
    print(f"  parameters      {net.parameter_count:,}")
    print(f"  gripper         {loaded.gripper}")
    print(f"  slots           {head.slots}   mixing {head.slot_mixing}   axis {head.axis_mode}")
    print(f"  part roles      {', '.join(head.part_roles) if head.part_roles else 'none'}")
    print(f"  target          {loaded.step.target}")
    print(f"  points          {loaded.sample.points}")

    card = path.with_suffix(".card.json")
    if card.is_file():
        report = json.loads(card.read_text(encoding="utf-8")).get("report") or {}
        if report:
            print(f"  recipe          {report.get('recipe') or 'none (hand-assembled)'}")
            # The tier is printed louder than the recipe, because it is the one that can mislead:
            # `--tier smoke` is two epochs on four hundred units with the refit off, and
            # `recipes.py` says in its own words that it "says nothing about grasp quality, and
            # is not a model to deploy". Without this line such a file is indistinguishable at
            # load time from one that took hours on the full corpus.
            tier = report.get("tier")
            if tier == "smoke":
                print(f"  smoke tier ({tier}): a chain-closes check, not a model to deploy. "
                      f"It says nothing about grasp quality.")
            elif tier:
                print(f"  tier            {tier}")
            print(f"  weights are     {report.get('weights_are', 'unknown')}")
            corpus = report.get("corpus") or {}
            if corpus:
                print(f"  trained on      {corpus.get('units')} unit(s), "
                      f"{corpus.get('groups')} asset group(s)")
            if report.get("control"):
                print(f"  control run ({report['control']}): fitted to synthetic labels "
                      f"derived from an input channel. Not a model to serve.")
            held = report.get("held")
            if held:
                top1 = held.get("held_top1_hit")
                if top1 is not None:
                    print(f"  last held top-1 {top1}")
            elif report.get("why_no_held"):
                print(f"  no held-out     {report['why_no_held']}")
        else:
            print("  (its card carries no report; written before the card was filled in)")
    else:
        print(f"  (no card beside it: {card.name})")
    return _EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.robot.grasping.deep",
        description="Probe, train and inspect the learned 6-DoF grasp generator.")
    sub = parser.add_subparsers(dest="cmd", required=True)


    train_set = sub.add_parser("train-set", help="fit the SET generator (K slots per seed)")
    train_set.add_argument("--clouds", required=True, help="corpus directory, walked recursively")
    train_set.add_argument("--out", required=True, help="directory for the weights and the report")
    train_set.add_argument("--slots", type=int, default=None, metavar="K",
                           help="grasps predicted per seed. The architecture arm. At K=1 a seed "
                                "binds one grasp and everything else that seed knows is discarded")
    train_set.add_argument("--control", default=None, metavar="LEVEL",
                           help="replace the corpus labels with a target derived from an INPUT "
                                "CHANNEL: `normal` (approach = -normal, fixed axis and width) or "
                                "`local` (width follows the neighbourhood). A stack that cannot fit "
                                "this is broken independently of any data question. ITS NUMBERS ARE "
                                "NOT GRASPING NUMBERS and the report stamps the level")
    train_set.add_argument("--resume", action="store_true",
                           help="continue the run whose checkpoint is in --out. REFUSES on any "
                                "difference in the plan, compared field by field rather than by "
                                "eye: a resume continues one experiment and a changed plan makes it "
                                "a different one wearing the old run's card. Use --init-from "
                                "instead to START a new run from those weights, which is what "
                                "training on your own objects needs. Restores the optimiser "
                                "moments, the schedule position and both seed streams, without "
                                "which a continuation is a warm start rather than a continuation")
    train_set.add_argument("--backbone-width", type=int, default=None, metavar="W",
                           help="encoder channel width. Default 384, which is the 21.5M-parameter "
                                "model. The architecture plan quotes a 6.5M variant at width 256 "
                                "depth 8 and a 38.4M one at width 512 depth 12, and neither could "
                                "be selected from a shell until now, so its own capacity question "
                                "had no second arm. MEASURED on the 5080: 10.0 GB at batch 4 by "
                                "8192 points, and batch 8 does not fit")
    train_set.add_argument("--backbone-depth", type=int, default=None, metavar="D",
                           help="encoder block count. Default 12. Moves with --backbone-width for "
                                "the named variants; the two are independent here so a width sweep "
                                "at fixed depth is possible, which is what separates capacity from "
                                "receptive field")
    train_set.add_argument("--backbone-heads", type=int, default=None, metavar="H",
                           help="attention heads. Default 6, which divides the default width 384. "
                                "The width must be divisible by this, and passing a width that is "
                                "not is refused HERE rather than three frames into the model: the "
                                "plan's own 6.5M variant is width 256, which 6 does not divide, so "
                                "the first person to type the number from the plan would have hit "
                                "it. For width 256 use 8")
    train_set.add_argument("--artifact-gripper", default=None, metavar="NAME",
                           help="which hand the written artifact plans for. Taken from the corpus "
                                "when it carries exactly one, and REFUSED rather than guessed when "
                                "it carries several: a default would be the 2F-85, which is right "
                                "today and silently wrong the first time anybody trains on a varied "
                                "corpus. Without an artifact a trained model cannot be loaded by "
                                "anything, because the fold checkpoint is refused by kind")
    train_set.add_argument("--init-from", default=None, metavar="PATH",
                           help="start from the weights in this checkpoint instead of from random. "
                                "THIS IS FINE-TUNING AND IT IS NOT A RESUME: a new run, with its "
                                "own optimiser, schedule and split, whose weights happen to begin "
                                "somewhere. `train --resume` continues one interrupted experiment "
                                "and refuses any change to the corpus, which is exactly what a "
                                "customer training on their OWN objects needs to change. A "
                                "checkpoint that does not fit the architecture is refused rather "
                                "than loaded partly, and the report records where the weights came "
                                "from")
    train_set.add_argument("--freeze-backbone", action="store_true",
                           help="train only the heads and hold the encoder still. The cheap form of "
                                "fine-tuning and the one to reach for on a SMALL new corpus: 99.0 %% "
                                "of this model's parameters are in the backbone and a few hundred "
                                "scenes cannot re-estimate them. Not the default, because which is "
                                "better has not been measured on our data")
    train_set.add_argument("--axis-mode", default="director", metavar="HOW",
                           help="how the closing axis is written down: `director` (the default, six "
                                "matrix parameters and an eigendecode) or `vector` (three numbers, "
                                "loss 1 - (v.v)^2, decode by normalising). A direct comparison "
                                "puts the director ahead and a matched control agrees, so the "
                                "default stands; the flag exists to keep settling the question. "
                                "Do not settle it through the controls: the approach control "
                                "regresses -normal, a linear function of an input channel, and "
                                "the axis control regresses normalise(cross(normal, z)), which "
                                "is not the same job")
    train_set.add_argument("--target", default="approach", metavar="WHICH",
                           help="which direction the run predicts: `approach`, the default, or "
                                "`axis`, the closing axis. At one seed the labels spread the "
                                "approach over many directions while the closing axis stays single "
                                "valued, so the approach is not a function of the seed and a "
                                "per-seed head regressing it keeps an irreducible floor. `axis` "
                                "also moves the hit criterion, because gating on the approach "
                                "angle while training the axis reports a flat zero for a head "
                                "doing what was asked")
    train_set.add_argument("--slot-mixing", default="affine", metavar="HOW",
                           help="how a slot may differ from its neighbour: `affine` (the default), "
                                "`film` (per-slot scale and shift, starts numerically identical to "
                                "affine) or `mlp` (a nonlinearity between the concatenation and the "
                                "outputs). WHICH IS BETTER IS OPEN. Under affine the slots do "
                                "differ from one another; what can freeze instead is the SEED "
                                "conditioning, which is a different defect with a different "
                                "repair. Do not read this flag as a fix for something")
    train_set.add_argument("--epochs", type=int, default=UNSET)
    train_set.add_argument("--folds", type=int, default=5)
    train_set.add_argument("--run-folds", type=int, default=1,
                           help="how many of --folds to actually run. 1 is a diagnostic, not a "
                                "cross-validation, and the report says which it was")
    train_set.add_argument("--batch", type=int, default=4,
                           help="MEASURED on the 5080: the ceiling is the POINTS per sample, not the "
                                "parameters, and 64 is not reachable")
    train_set.add_argument("--points", type=int, default=8192)
    train_set.add_argument("--train-units", type=int, default=UNSET)
    train_set.add_argument("--eval-units", type=int, default=512)
    train_set.add_argument("--labelled-unit-share", type=float, default=0.5, metavar="F",
                           help="fraction of training units drawn from those that HAVE a label. "
                                "Most units have none, so at 0 most of a batch teaches nothing")
    train_set.add_argument("--generative", action="store_true",
                           help="STAGE 4b: replace the K-slot head with a DENOISING one, trained on "
                                "every label instead of a one-to-one matched subset. A supervised "
                                "point often admits more than one approach further than fifteen "
                                "degrees apart, and K slots bind at most K of them, so the "
                                "baseline's coverage ceiling sits below 1.0. ARM, NOT REPLACEMENT: "
                                "it runs against 4a on identical seeds, crops and folds. WARNING: "
                                "`top1_hit` means something different here, the FIRST SAMPLE DRAWN "
                                "rather than the most confident slot, because a generative head "
                                "emits no confidence. Coverage is comparable; top-1 is comparable "
                                "only as 'what a cell reaches for'")
    train_set.add_argument("--crop-mm", type=float, default=0.0, metavar="MM",
                           dest="crop_mm",
                           help="STAGE 3: give the head a ball of points around its seed, in the "
                                "seed's own frame, scaled by a kappa MEASURED on this corpus. 0 is "
                                "off, which is what every arm so far ran: the head reads ONE "
                                "gathered per-point feature and nothing about the neighbourhood it "
                                "sits in, while the architecture plan describes all three. Valid "
                                "radii are 40 to 120 mm, the band kappa was measured over; 85 is the "
                                "reference jaw's aperture. The crop starts at zero, so an arm with it "
                                "begins numerically identical to one without")
    train_set.add_argument("--crop-neighbours", type=int, default=32, metavar="N",
                           dest="crop_neighbours",
                           help="points kept per crop. Points beyond --crop-mm are MASKED rather "
                                "than dropped, so a sparse neighbourhood reads as sparse instead of "
                                "reaching across a gap for filler")
    train_set.add_argument("--collapse-floor-deg", type=float, default=UNSET, metavar="DEG",
                           dest="collapse_floor_deg",
                           help="stop a fold whose head proposes ONE direction for every seed, "
                                "measured as held-out seed_spread_deg below DEG from epoch three "
                                "on. 0 is off. A collapsed head reaches the physics referee with a "
                                "handful of distinct approach vectors and a within-scene spread of "
                                "zero, and that is only visible three stages downstream")
    train_set.add_argument("--part-roles", action="store_true",
                           help="give the head an AFFORDANCE output: one role per proposed grasp, "
                                "over the fixed vocabulary (empty, body, handle, grip, neck, head). "
                                "This is what 'grasp it by the handle' needs, and the K-slot family "
                                "had no such head at all while the retired binned family trained one "
                                "that nothing read at inference. The supervision is already in the "
                                "corpus: some of its labels carry a non-empty role. Off by default "
                                "and a SEPARATE output layer, so a net without it is unchanged "
                                "tensor for tensor and every artifact ever written still loads. "
                                "UNMEASURED: whether it helps or costs the geometry is the arm "
                                "this flag exists to run")
    train_set.add_argument("--recipe", default=None, metavar="NAME",
                           help="a named, versioned bundle of the settings this project has "
                                "measured or decided, applied before any other flag so an explicit "
                                "flag still wins. Stamped into the artifact, so a model can say "
                                "which recipe produced it. A recipe is FROZEN once it ships: a "
                                "better bundle becomes v2 beside v1, never an edit to v1. It "
                                "deliberately contains no unsettled arm (slot-mixing, axis-mode, "
                                "crop, generative are all being measured)")
    train_set.add_argument("--tier", default=None, metavar="NAME",
                           help="how long to run: `smoke` proves the chain closes on your corpus "
                                "and your box and says NOTHING about grasp quality, `full` is the "
                                "model you deploy. Applied after --recipe and before explicit flags")
    train_set.add_argument("--refit", action="store_true", default=UNSET,
                           help="after the folds, train one more net on EVERY unit and ship THAT "
                                "as the artifact. The fold pass earns the numbers; it has to hold "
                                "assets back to do so, and the shipped model has no such reason. "
                                "With --folds 5 --run-folds 1 the artifact otherwise never sees "
                                "the units the split holds back, which on a customer's corpus is "
                                "part of their own catalogue. Costs a second training run, and "
                                "reports TRAINING metrics only, because a net fitted to every unit "
                                "has no held-out set left. UNMEASURED on this architecture: the "
                                "argument for it is a product argument, not a metric one")
    train_set.add_argument("--seed", type=int, default=0)

    headroom = sub.add_parser("approach-headroom",
                              help="how many degrees of approach signal exist above a constant")
    headroom.add_argument("--clouds", required=True)
    headroom.add_argument("--units", type=int, default=256)
    headroom.add_argument("--points", type=int, default=8192)
    headroom.add_argument("--seed", type=int, default=0)
    headroom.add_argument("--json", default=None, metavar="PATH",
                          help="also write the block as JSON, so a plan can quote it")

    foreign = sub.add_parser("import-foreign",
                             help="fetch a public grasp corpus as scene files this loop reads")
    foreign.add_argument("--out", required=True, help="directory for the imported scene .npz files")
    foreign.add_argument("--limit", type=int, default=200, metavar="N",
                         help="how many scenes to fetch. About 200 KB each by range request, so a "
                              "thousand is 200 MB rather than the source's 229 GB")
    foreign.add_argument("--gripper", default="wide_140", metavar="NAME",
                         help="which jaw profile the imported labels belong to. The source names a "
                              "140 mm hand. STAMPED into every scene, because without it the corpus "
                              "index falls back to `2f85` and grasps for a wider hand would train a "
                              "model that believes an 85 mm hand made them. A derived contact "
                              "separation wider than this profile's aperture is REFUSED: it is a "
                              "statement about the object, not a label our gripper can reproduce")
    foreign.add_argument("--jobs", type=int, default=8, metavar="N",
                         help="scenes fetched in parallel. MEASURED by profiling three real scenes: "
                              "8.30 s of an 8.53 s scene is the fetch, 97.3 percent, and the CPU "
                              "half is 0.23 s. Each scene needs six to ten range requests and every "
                              "one opens a fresh connection, so the time is handshake latency and a "
                              "thread pool is what fixes it. Eight is a polite default against "
                              "somebody else's host")
    foreign.add_argument("--validate", type=int, default=3, metavar="N",
                         help="read this many written scenes back through our own loader and put "
                              "them through the sample contract. 0 skips it. Not free and not "
                              "optional in spirit: a validator nobody invokes catches nothing, and "
                              "the four frame defects this repository has shipped were all of the "
                              "kind it looks for")
    foreign.add_argument("--cache-dir", default=None, metavar="DIR",
                         help="where to keep the archives' central directories, read once and "
                              "reused. Defaults to a folder inside --out, because WITHOUT IT THE "
                              "OBJECT MASKS AND PROMPTS ARRIVE BY "
                              "COINCIDENCE: each archive carries its own directory of a million "
                              "entries or more, and intersecting four-megabyte slices of them "
                              "leaves most scenes with no mask. A scene without a mask collapses "
                              "to ONE training unit however many objects it holds, and nothing "
                              "raises. About 460 MB once, against a 229 GB corpus")
    foreign.add_argument("--centre-offset", type=float, default=None, metavar="M",
                         help="the source stores the gripper MOUNTING FACE, about 0.173 m behind the "
                              "annotated point. Left unset it is re-derived per scene from where the "
                              "cloud actually is, and the fitted value is written to provenance.json. "
                              "Set it only to reproduce an earlier import exactly")

    propose = sub.add_parser("propose",
                             help="what a trained model would grasp, as a list a referee can judge")
    propose.add_argument("--artifact", required=True,
                         help="the weights written by train-set, which the fold checkpoint is NOT: "
                              "a checkpoint is refused by kind, and that refusal is why nothing a "
                              "run produced could be deployed until the artifact writer got a call "
                              "site")
    propose.add_argument("--clouds", required=True, help="scenes to propose on, walked recursively")
    propose.add_argument("--out", required=True, help="where the proposals are written, one per line")
    propose.add_argument("--top", type=int, default=8, metavar="N",
                         help="how many of the model's own ranked grasps to keep per scene. A "
                              "PREFIX of the confidence order, because that prefix is what a cell "
                              "would actually try")
    propose.add_argument("--seeds", type=int, default=64, metavar="N",
                         help="seed points per scene, drawn from the model's OWN graspability field "
                              "rather than from the labels: seeding from the answer key would grade "
                              "the head on poses it was handed")
    propose.add_argument("--scenes", type=int, default=None, metavar="N",
                         help="how many scenes to propose on, drawn EVENLY ACROSS FAMILIES rather "
                              "than as a prefix. A scene id is <family>_<index>, so a sorted prefix "
                              "is one family, and a referee success rate taken from such a prefix "
                              "describes that family rather than the corpus")
    propose.add_argument("--families", default=None, metavar="A,B",
                         help="restrict to these scene families (comma separated). Without it every "
                              "family in the corpus is drawn from")
    propose.add_argument("--spread-mm", type=float, default=0.0, metavar="MM",
                         dest="spread_mm",
                         help="drop a grasp that sits within MM of a higher-ranked one AND "
                              "approaches from within 20 degrees of it, BEFORE --top is taken. Off "
                              "by default. A head can fill every --top slot with grasps whose "
                              "approach vectors agree to six decimal places: a referee handed that "
                              "judges ONE grasp --top times and reports that many trials. The "
                              "distinct share is printed either way")
    propose.add_argument("--seed", type=int, default=0)

    score = sub.add_parser(
        "score-proposals",
        help="stage 3.6: the referee success rate, and a scorer fitted on the model's own grasps")
    score.add_argument("--proposals", required=True,
                       help="the file `deep propose` wrote")
    score.add_argument("--verdicts", required=True,
                       help="the physics file that judged it. The join is by (scene, rank), never "
                            "by pose: a pose join is ambiguous wherever a label and the row "
                            "describing it carry identical poses")
    score.add_argument("--folds", type=int, default=5, metavar="N",
                       help="scene-disjoint folds. Two proposals on one scene share a point cloud, "
                            "so a random split reports memorising a scene rather than judging a "
                            "grasp")
    score.add_argument("--out", default=None, help="write the report as JSON as well")
    score.add_argument("--seed", type=int, default=0)
    score.add_argument("--chart", default=None, metavar="PNG",
                       help="where to write the four-panel evaluation figure. Defaults to the "
                            "proposals file with a .eval.png suffix")
    score.add_argument("--control", default=None, metavar="JSONL",
                       help="the LABEL control's verdicts on the same shard, as written by "
                            "`datagen physics-sample --per-class N`. Without it the hold rate has "
                            "no scale: the referee's own ceiling is nowhere near 100 percent, and a "
                            "customer reading a bare rate out of 100 will conclude the wrong thing")
    score.add_argument("--aperture-mm", type=float, default=85.0, metavar="MM",
                       help="the configured hand's opening, drawn as a line on the width panel. A "
                            "proposal past it is one the cell physically cannot execute")
    score.add_argument("--gripper", default="2f85", metavar="NAME",
                       help="which hand the numbers are about. Labelling only")

    inspect = sub.add_parser("inspect", help="what a weights file says about itself")
    inspect.add_argument("--artifact", required=True)

    asset = sub.add_parser(
        "asset-consistency",
        help="is the approach direction a property of the OBJECT or of the SCENE? Labels only, no "
             "training, minutes")
    asset.add_argument("--clouds", required=True)
    asset.add_argument("--families", default="",
                       help="comma separated scene families to KEEP, e.g. sparse,packed,pile")
    asset.add_argument("--min-scenes", type=int, default=3,
                       help="an asset must appear in at least this many scenes before it can say "
                            "anything about consistency ACROSS scenes")
    asset.add_argument("--out", default=None)

    geo = sub.add_parser(
        "geometric-floor",
        help="what does the approach target look like to a rule that does NO learning? Training-free")
    geo.add_argument("--clouds", required=True)
    geo.add_argument("--families", default="",
                     help="comma separated scene families to KEEP, e.g. sparse,packed,pile")
    geo.add_argument("--train-units", type=int, default=48,
                     help="used ONLY to choose the honest constant bin, exactly as a trained run "
                          "chooses its floor, so this table anchors to theirs")
    geo.add_argument("--test-units", type=int, default=100)
    geo.add_argument("--batch-size", type=int, default=8)
    geo.add_argument("--seeds", default="0,1,2")
    geo.add_argument("--reference", default=None, metavar="JSON",
                     help="a step 9 style approach result to print beside the rules, so the trained "
                          "arms and the training-free ones are read on one page")
    geo.add_argument("--device", default=None)
    geo.add_argument("--out", default=None)

    sanity = sub.add_parser(
        "sanity",
        help="is the ground truth where the network can see it? (dataset and sampling, no training)")
    sanity.add_argument("--clouds", required=True, help="the corpus directory")
    sanity.add_argument("--scenes", type=int, default=40,
                        help="how many scenes to measure. The whole point is a distribution, so one "
                             "scene answers nothing")
    sanity.add_argument("--architecture", default="base",
                        help="whose sampling schedule to measure against")
    sanity.add_argument("--seed", type=int, default=0)
    sanity.add_argument("--visual", default=None, metavar="PNG",
                        help="also draw the FIRST scene to this path: cloud, graspable points, and "
                             "the labelled contacts, in three orthographic views")
    sanity.add_argument("--out", default=None, help="write the report JSON here as well")

    report = sub.add_parser("report",
                            help="what a training run did: the curve, the lift, where it plateaued")
    source = report.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", default=None, metavar="DIR",
                        help="the --out directory of a train-set run, e.g. logs/dl/models/arm_film. "
                             "Reads epochs.json and report.json directly. PREFER THIS: the log "
                             "parser matches the retired single-label trainer's line, so it is blind "
                             "to every set run, and scraping numbers that exist as JSON one "
                             "directory over is how a report comes to disagree with its own run")
    report.add_argument("--name", default=None, help="label for the report (default: the filename)")
    report.add_argument("--curve", default=None, metavar="PNG",
                        help="also write the learning curve here (default: beside the log)")
    report.add_argument("--no-curve", action="store_true", help="text only, skip matplotlib")
    return parser


def _cmd_report(args: argparse.Namespace) -> int:
    """What a training run actually did.

    A final number cannot say whether the run was still learning when it stopped, which is the
    question that decides the next experiment; this report answers it from the run's own epochs
    and curve.
    """
    from pathlib import Path

    from src.robot.grasping.deep.eval.run_report import (
        build_report, format_report, write_curve,
    )

    # A run directory, and nothing else: `epochs.json` and `report.json` are read directly rather
    # than scraped from a log. A log pattern that matches no `SetLoop` line reports an empty run
    # instead of refusing, so the numbers a person reads are absent rather than wrong.
    root = Path(args.run)
    if not root.is_dir():
        print(f"no such run directory: {root}", file=sys.stderr)
        return _EXIT_USAGE
    default_curve = root / "curve.png"
    name = args.name or root.name
    try:
        report = build_report(name, root)
    except ValueError as exc:
        print(f"report: {exc}", file=sys.stderr)
        return _EXIT_USAGE

    print(format_report(report))
    if not args.no_curve:
        target = Path(args.curve) if args.curve else default_curve
        print(f"\n  curve: {write_curve(report, target)}")
    return _EXIT_OK



def _cmd_train_set(args: argparse.Namespace) -> int:
    """Fit the SET generator: K slots per seed, matched one-to-one against the seed's label set.

    A formatter. Validating the enum values, checking the backbone divisibility and the crop
    radius, and merging a recipe all live in `train.plan.build_plan` and
    `train.api.GeneratorTraining`, so the same run is available from code:

        run = GeneratorTraining.from_recipe(corpus="logs/dl/clouds/v5", recipe="v1", tier="full",
                                            out_dir="logs/dl/models/mine", artifact_gripper="2f85")
        report = run.train()

    `--control` is a diagnostic and its numbers are not grasping numbers. It answers exactly one
    question, whether this stack can fit a target it is guaranteed to be able to represent, and both
    the report and the model card stamp the level so no later reader can mistake the answer for a
    result about grasps.
    """
    from src.robot.grasping.deep.train.api import GeneratorTraining  # noqa: PLC0415
    from src.robot.grasping.deep.train.plan import PlanOverrides  # noqa: PLC0415

    # Argparse into data, field by field. Anything the operator did not type is `UNSET` here,
    # which is what lets the recipe merge happen without looking at `sys.argv`.
    overrides = PlanOverrides(
        epochs=args.epochs, folds=args.folds, run_folds=args.run_folds, batch=args.batch,
        seed=args.seed, train_units=args.train_units, eval_units=args.eval_units,
        points=args.points, refit=args.refit,
        labelled_unit_share=args.labelled_unit_share,
        collapse_floor_deg=args.collapse_floor_deg, control=args.control,
        target=args.target, axis_mode=args.axis_mode,
        slots=args.slots if args.slots is not None else UNSET,
        slot_mixing=args.slot_mixing, part_roles=args.part_roles, generative=args.generative,
        crop_mm=args.crop_mm, crop_neighbours=args.crop_neighbours,
        backbone_width=args.backbone_width if args.backbone_width is not None else UNSET,
        backbone_depth=args.backbone_depth if args.backbone_depth is not None else UNSET,
        backbone_heads=args.backbone_heads if args.backbone_heads is not None else UNSET)

    try:
        run = GeneratorTraining.from_recipe(
            corpus=args.clouds, recipe=args.recipe, tier=args.tier, overrides=overrides,
            out_dir=args.out, init_from=args.init_from, freeze_backbone=args.freeze_backbone,
            resume=args.resume, artifact_gripper=args.artifact_gripper)
    except FileNotFoundError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM
    except ValueError as exc:
        # Construction refuses before the corpus walk and the probes, so a misspelled enum costs a
        # second rather than minutes.
        print(f"{exc}", file=sys.stderr)
        return _EXIT_USAGE

    # What actually applies, not what the bundle contains: a run given an explicit `--epochs 1`
    # against a recipe asking for 2 announces 1. This banner is the line a person quotes
    # afterwards, so it may not report a setting the run did not use.
    if args.recipe or args.tier:
        notes = dict(run.recipe_notes)
        overridden = notes.pop("overridden", [])
        label = " ".join(part for part in (f"recipe {args.recipe}" if args.recipe else "",
                                           f"tier {args.tier}" if args.tier else "") if part)
        note = f"{label}: " + (" ".join(f"{k}={v}" for k, v in notes.items()) if notes
                               else "(nothing to apply)")
        if overridden:
            note += "  | your flags win: " + " ".join(overridden)
        print(note, flush=True)

    # What is about to run, before it costs anything, so a mistyped arm is stopped in the first
    # second rather than the sixth hour. `describe()` and `report.render()` do not overlap: the
    # configuration block belongs here and is not printed again after the run.
    print(run.describe(), flush=True)

    try:
        report = run.train()
    except ValueError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM

    if args.init_from:
        print(f"  inherited  {report.raw.get('inherited', {}).get('tensors', '?')} tensor(s) from "
              f"{args.init_from}", flush=True)
    run.write_report(report)
    print(report.render(), flush=True)
    return _EXIT_OK



def _cmd_approach_headroom(args: argparse.Namespace) -> int:
    """How many degrees of approach signal exist above a constant. No network, no GPU."""
    from src.robot.grasping.deep.corpus.sample import SampleSpec  # noqa: PLC0415
    from src.robot.grasping.deep.train.trainer import SetTrainingPlan, corpus_index
    from src.robot.grasping.deep.eval.probes import approach_headroom

    try:
        files = scene_files(args.clouds)
    except FileNotFoundError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM
    index = corpus_index(files)
    plan = SetTrainingPlan(sample=SampleSpec(grasp_set=True, points=args.points))
    result = approach_headroom(index, plan, units=args.units, seed=args.seed)
    for label, block in result.items():
        print(f"  {label}")
        for key in ("straight_down", "global", "per_object", "per_seed"):
            print(f"    {key:14s} {block[key]:6.2f} deg")
        print(f"    over {int(block['labels'])} label(s), {int(block['objects'])} object(s), "
              f"{int(block['seeds'])} seed(s)")
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(f"  wrote {args.json}")
    return _EXIT_OK



def _cmd_import_foreign(args: argparse.Namespace) -> int:
    """Fetch scenes from a public grasp corpus and write them as scene files this loop reads.

    Scene files rather than a second loader: `train-set` then runs on the imported corpus with no
    flag and no change, and so do the folds, the probes and every metric. A user with no simulator
    trains on somebody else's data, a user who wants their own cell generates it, and the consumer
    cannot tell the two producers apart.

    The fetch is by range request. The source is 229 GB and its clouds are a Zip64 split across
    five parts, so a full download is not the price of looking at it: about 200 KB per scene.
    """
    from src.robot.grasping.deep.foreign.grasp_anything import (  # noqa: PLC0415
        DEFAULT_GRIPPER, SOURCE, import_scenes)
    from src.robot.grasping.deep.net.gripper import JAW_GEOMETRY  # noqa: PLC0415

    if args.gripper not in JAW_GEOMETRY:
        print(f"unknown gripper {args.gripper!r}; choose from {', '.join(sorted(JAW_GEOMETRY))}",
              file=sys.stderr)
        return _EXIT_USAGE
    print(f"  source     {SOURCE.title}")
    print(f"  licence    {SOURCE.licence}  ({SOURCE.url})")
    print(f"  gripper    {args.gripper} (default {DEFAULT_GRIPPER})")
    try:
        provenance = import_scenes(args.out, limit=args.limit, gripper=args.gripper,
                                   centre_offset_m=args.centre_offset, cache_dir=args.cache_dir,
                                   validate=args.validate, jobs=args.jobs,
                                   report=lambda line: print(line, flush=True))
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM
    for key in sorted(provenance):
        print(f"  {key:20s} {provenance[key]}")
    if not provenance["written"] and not provenance["skipped_present"]:
        print("nothing was imported", file=sys.stderr)
        return _EXIT_PROBLEM
    print("")
    print(f"  now train on it:  python -m src.robot.grasping.deep train-set "
          f"--clouds {args.out} --out logs/dl/models/foreign")
    return _EXIT_OK



def _cmd_propose(args: argparse.Namespace) -> int:
    """What a trained generator would actually do, written down so a referee can judge it.

    The architecture plan's primary metric is the referee success rate, which needs a trained
    model turned into a list of grasps a physics cell can try. Section 3.6 of that plan asks for a
    scorer trained on the generator's own proposals against a physics target, which needs the same
    list.

        deep propose --artifact W --clouds DIR --out proposals.jsonl
        datagen physics-sample --proposals proposals.jsonl --physics-engine mujoco
    """
    from src.robot.grasping.deep.eval.propose import (  # noqa: PLC0415
        propose_for_scenes, write_proposals)

    try:
        files = scene_files(args.clouds)
    except FileNotFoundError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM
    if args.scenes:
        files = stratified_scenes(files, args.scenes, seed=args.seed, families=args.families)
    elif args.families:
        files = stratified_scenes(files, len(files), seed=args.seed, families=args.families)
    families = sorted({f.stem.rsplit("_", 1)[0] for f in files})
    print(f"  {len(files)} scene(s) over {len(families)} famil(y/ies): {', '.join(families)}")
    proposals = propose_for_scenes(args.artifact, files, top=args.top, seeds=args.seeds,
                                   seed=args.seed, spread_mm=args.spread_mm,
                                   report=lambda line: print(line, flush=True))
    if not proposals:
        print("the model proposed nothing on these scenes", file=sys.stderr)
        return _EXIT_PROBLEM
    written = write_proposals(args.out, proposals)
    print(f"  {written['proposals']} proposal(s) over {written['scenes']} scene(s) "
          f"-> {written['path']}")
    print(f"  now judge them:  python -m datagen physics-sample --proposals {args.out} "
          f"--physics-engine mujoco")
    return _EXIT_OK


def _cmd_score_proposals(args: argparse.Namespace) -> int:
    """Stage 3.6, and the plan's primary metric, out of the two files the pipeline already writes.

    The hold rate printed here is the referee success rate: of the grasps the model chose, how many
    survived a physics cell. A hold rate over grasps drawn out of the labels answers a different
    question, and the two must never be compared across that line.

    The floors print beside the score. The head's own confidence is one of the six features, so a
    scorer that learned nothing beyond "trust the head" would still beat 0.5, and an AUC without
    its floor cannot be read.
    """
    from src.robot.grasping.deep.eval.proposal_scorer_training import (  # noqa: PLC0415
        fit_from_files, summarise, unjudged)

    try:
        report = fit_from_files(args.proposals, args.verdicts, folds=args.folds, seed=args.seed)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return _EXIT_PROBLEM
    for line in summarise(report):
        print(line)
    # Reported, not swallowed. A proposal the referee never judged is dropped from the fit, and a
    # run that quietly halved its own dataset would still print a clean AUC.
    missing = unjudged(args.proposals, args.verdicts)
    if missing:
        print(f"  {missing} proposal(s) had no verdict and were dropped, never scored as failures")
    # The picture, beside the numbers. Five printed lines cannot answer whether a model is any
    # good: a hold rate has no scale without the label control, and three different defects
    # (aimed at nothing, gripped and slipped, refused before physics) arrive as one number.
    #
    # Never fatal. A customer who came for the rate must get the rate even if matplotlib cannot
    # write a file, so a failed chart is a printed line rather than a non-zero exit.
    try:
        from src.robot.grasping.deep.eval.charts import (  # noqa: PLC0415
            ChartInputs, summarise_charts, write_eval_charts)

        chart_path = (Path(args.chart) if args.chart
                      else Path(args.proposals).with_suffix(".eval.png"))
        numbers = write_eval_charts(
            ChartInputs(args.proposals, args.verdicts, control=args.control,
                        aperture_mm=args.aperture_mm, gripper=args.gripper),
            chart_path)
        print("")
        for line in summarise_charts(numbers):
            print(line)
    except Exception as exc:  # noqa: BLE001 (the numbers above are the product; the chart is not)
        print(f"  (no chart: {type(exc).__name__}: {exc})", file=sys.stderr)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"  -> {args.out}")
    return _EXIT_OK


#: Every subcommand, mapped to the function that runs it.
#:
#: A table, not a chain of `if args.cmd == ...`. A branch missing from a chain is invisible: the
#: parser still advertises the subcommand, `--help` still lists it, it still parses, and a valid
#: invocation then falls through to the usage exit and returns 2.
#:
#: The table carries two rules instead: every parser name is a key here, and every key is a
#: parser name. A branch cannot go missing without the count changing.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "train-set": _cmd_train_set,
    "approach-headroom": _cmd_approach_headroom,
    "import-foreign": _cmd_import_foreign,
    "propose": _cmd_propose,
    "score-proposals": _cmd_score_proposals,
    "inspect": _cmd_inspect,
    "asset-consistency": _cmd_asset_consistency,
    "geometric-floor": _cmd_geometric_floor,
    "sanity": _cmd_sanity,
    "report": _cmd_report,
}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = COMMANDS.get(args.cmd)
    if handler is None:
        # argparse already refuses an unknown name, so reaching this means the parser and the table
        # disagree. Say which one, rather than printing a bare usage line.
        print(f"no handler for subcommand {args.cmd!r}; the parser and COMMANDS disagree")
        return _EXIT_USAGE
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
