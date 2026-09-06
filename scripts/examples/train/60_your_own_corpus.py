"""60: the whole chain on your own parts, from a directory of meshes to a report you can read.

    python scripts/examples/train/60_your_own_corpus.py --price-only
    python scripts/examples/train/60_your_own_corpus.py --scenes 24
    python scripts/examples/train/60_your_own_corpus.py --parts custom --my-parts D:/meshes \\
        --licence own --screen 20 --scenes 2000 --engine mujoco --tier full

The decision: train a grasp generator on the objects your cell actually handles. No public dataset
contains them, so nobody else can produce this corpus, and the whole path is here in one file:
price it, get the meshes in, find out which of them a jaw can hold at all, render scenes, label the
grasps, extract the corpus, train, read the report. 61_the_foreign_corpus.py is the other road, for
a reader with no parts of their own yet.

Priced before it starts, because the parts of this that are expensive are expensive by hours. The
defaults are small on purpose and run on any machine: `--engine none` seats objects analytically and
rasterises in numpy, needing no GPU and no simulator. `--price-only` prints the bill and stops.

Three traps this file walks a reader past, each of which costs a whole corpus:

  a request is not a yield.       `--engine none` refuses the pile family by name, because a pile is
                                  the physics. The calculator answers in usable scenes and asks for
                                  the larger number, so a request for 2,000 leaves you fewer.
  naming your meshes is not       `assets.mesh_asset_ids` restricts which meshes are drawn, not
  choosing them.                  which scenes are built. Procedural objects carry their own weight,
                                  so a corpus that names your parts and leaves that weight at its
                                  default comes out part full of objects nobody asked for, and
                                  nothing says so. `--parts custom` zeros it and turns the fallback
                                  into a refusal.
  a licence has no default.       Your parts are yours to declare. The import refuses without an
                                  explicit licence rather than guessing, because the audit reads
                                  that string and a guess would pass an audit nobody performed.

The three data stages are the three verbs of one `datagen.api.DatasetBuild`, and the training stage
is one `deep.train.api.GeneratorTraining`, so every count printed below is the one the builder
computed rather than a directory listing taken afterwards.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the corpus was built, a model was trained and its report was read
  1  a stage ran and produced nothing usable: read the WARN line for which one
  2  nothing to build with: no meshes, no engine, or a request that contradicts itself

the stages, and what each one costs
  assets    minutes to hours, and gigabytes, only when --fetch or --my-parts is passed. The
            collections differ by four orders of magnitude in size, so --fetch takes a count
  screen    minutes, CPU, and it is what tells you which of your parts a jaw can hold
  render    the expensive one; the only stage that wants a GPU, and only on `isaac`
  label     minutes, CPU, closed-form geometry, re-runnable on its own
  clouds    minutes, CPU, writes the .npz the trainer reads
  train     hours on one GPU at --tier full, minutes at --tier smoke
"""

#: Folds to fit. One is a diagnostic rather than a cross-validation; the model card records which.
_RUN_FOLDS: Final[int] = 1

#: Epochs when neither the caller nor a tier names a number. Small on purpose: the first run on a
#: new corpus is there to prove the chain closes.
_SHORT_EPOCHS: Final[int] = 2

#: Which weights each `--parts` answer sets. The mesh answers zero the procedural weight and refuse
#: the fallback, which is the difference between a corpus of your parts and a corpus that is partly
#: your parts. `procedural` is the shipped default restated, so the three answers read side by side.
_PARTS: Final[dict[str, dict[str, Any]]] = {
    "procedural": {"procedural_weight": 1.0},
    "library": {"procedural_weight": 0.0, "gso_weight": 1.0,
                "refuse_procedural_fallback": True},
    "custom": {"procedural_weight": 0.0, "custom_weight": 1.0,
               "refuse_procedural_fallback": True},
}

#: Which collection each answer draws from, or None for the generated objects. Counting the whole
#: library instead would answer the wrong question: the default sweep excludes `custom` on purpose,
#: so a machine with a thousand public meshes and no parts of its own reads as ready for
#: `--parts custom` and then refuses inside the render, several stages later.
_PARTS_SOURCE: Final[dict[str, str | None]] = {
    "procedural": None, "library": "gso", "custom": "custom"}


def _add_your_parts(run: Example, origin: str, *, licence: str, attribution: str) -> int:
    """Copy a directory of meshes into the library as the `custom` source. 0 to carry on.

    The import sits outside the step, and the refusal with it. A `not_ready` returned from inside a
    step leaves that step with no detail, so it is recorded as "did not complete" and the reader
    sees a failed step above a refusal that is not a failure at all.
    """
    from datagen.assets.library import import_from_directory

    try:
        entries = import_from_directory("custom", origin, license=licence,
                                        attribution=attribution)
    except (FileNotFoundError, ValueError) as error:
        # The licence refusal lands here, and it is the point rather than friction: nobody but the
        # owner of a part knows what it is licensed as, and the audit reads that string.
        return not_ready(f"{type(error).__name__}: {error}",
                         "pass --licence own for parts you designed, or the actual identifier for "
                         "anything you did not, and check the directory exists")
    with run.step("copy your parts into the mesh library") as report:
        report(f"{len(entries)} mesh(es) imported as `custom` under {licence!r}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scenes", type=int, default=8, help="usable scenes to end up with")
    parser.add_argument("--engine", default="none", choices=("none", "mujoco", "isaac"),
                        help="none (default) needs nothing beyond this repository")
    parser.add_argument("--parts", default="procedural", choices=tuple(_PARTS),
                        help="which objects go in the scenes")
    parser.add_argument("--my-parts", default=None, metavar="DIR",
                        help="a directory of your own meshes to copy into the library first")
    parser.add_argument("--licence", default=None,
                        help="the licence your parts carry. No default, deliberately")
    parser.add_argument("--attribution", default="",
                        help="who must be named. Obligatory for anything CC-BY")
    parser.add_argument("--fetch", type=int, default=0, metavar="N",
                        help="download N meshes from --fetch-source first. There is no unlimited "
                             "form here, deliberately: one of the collections is terabytes")
    parser.add_argument("--fetch-source", default="gso",
                        help="which collection --fetch draws from. Default gso")
    parser.add_argument("--screen", type=int, default=0, metavar="N",
                        help="screen meshes until N of them earn a jaw label, and write the "
                             "screen. Minutes")
    parser.add_argument("--suction", action="store_true",
                        help="carry suction labels into the corpus as well as jaw labels")
    parser.add_argument("--name", default="my_parts", help="dataset name")
    parser.add_argument("--out", default="logs/examples", help="dataset root")
    parser.add_argument("--model-out", default="logs/examples/my_parts_model",
                        help="where the trained artifact and its report go")
    parser.add_argument("--recipe", default=None, help="see 62_recipe_and_tier.py")
    parser.add_argument("--tier", default=None, help="see 62_recipe_and_tier.py")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"epochs. Default: what the tier asks for, else {_SHORT_EPOCHS}")
    parser.add_argument("--gripper", default=None,
                        help="which hand the artifact plans for. Taken from the corpus when it "
                             "carries exactly one, and refused rather than guessed otherwise")
    parser.add_argument("--price-only", action="store_true", help="print the bill and stop")
    args = parser.parse_args(argv)

    from datagen.api import DatasetBuild
    from datagen.assets.service import MeshPreparation, available_sources
    from datagen.cost import ENGINE_COSTS, estimate, format_estimate
    from src.robot.grasping.deep.train.recipes import settings

    # Resolved once, here, and read by the bill and by the run alike, so the two cannot disagree
    # about what is going to happen.
    try:
        schedule = settings(args.recipe, args.tier)
    except ValueError as error:
        return not_ready(str(error), "62_recipe_and_tier.py lists the names that exist")
    epochs = args.epochs if args.epochs is not None else int(schedule.get("epochs", _SHORT_EPOCHS))
    refit = bool(schedule.get("refit", False))

    library = MeshPreparation.from_sources()
    parts = _PARTS[args.parts]
    with Example("60 your own corpus", f"{args.scenes} scene(s) of {args.parts} objects on "
                                       f"`{args.engine}`, then {epochs} epoch(s)",
                 hardware=False) as run:
        with run.step("price the whole chain, before spending anything") as report:
            # One bill for every stage, not one for the render. On the cheap engines the labelling,
            # the corpus extraction and the training dominate, so a bill covering the render alone
            # is worst on exactly the configuration a customer without a GPU will pick.
            meshes = 0 if args.parts == "procedural" else len(library.entries())
            plan = estimate(args.scenes, engine=args.engine, jobs=1, meshes=meshes,
                            epochs=epochs, folds=_RUN_FOLDS, refit=refit)
            report(f"{plan.hours:.2f} h, {plan.gigabytes:.2f} GB, "
                   f"request {plan.requested_scenes} to get {plan.usable_scenes}")
        run.note("")
        for line in format_estimate(plan).splitlines():
            run.note(line)
        run.note("")
        if plan.requested_scenes != plan.usable_scenes:
            run.note(f"WARN two numbers, not one: `{args.engine}` yields "
                     f"{ENGINE_COSTS[args.engine].yield_fraction:.0%}, so the build below is asked")
            run.note(f"  for {plan.requested_scenes} scenes to leave you {plan.usable_scenes}.")
            run.note("")
        if args.price_only:
            run.note("--price-only: nothing was built and nothing was trained.")
            return run.exit_code

        # ---------------------------------------------------------------- the meshes
        catalogue = available_sources()
        with run.step("what can be downloaded, and on what terms") as report:
            report(f"{len(catalogue)} collection(s), "
                   f"{sum(row['objects'] for row in catalogue):,} object(s) between them")
        run.note("")
        run.note(f"  {'collection':12s} {'objects':>9s} {'GB':>10s}  licence")
        for row in sorted(catalogue, key=lambda item: float(item["approx_gb"])):
            checked = "" if row["licence_verified"] else "  (collection terms, no model checked)"
            run.note(f"  {row['key']:12s} {row['objects']:>9,} {float(row['approx_gb']):>10.1f}  "
                     f"{row['licence']}{checked}")
        run.note("")
        run.note("  read the size column before the licence column. The largest of these is four")
        run.note("  orders of magnitude bigger than the smallest, which is why `--fetch` takes a")
        run.note("  count rather than being a switch: a fetch with no limit would try to pull the")
        run.note("  whole of it. The unverified rows publish terms for the collection and state")
        run.note("  nothing per object, so a dataset built on one inherits a claim rather than a")
        run.note("  check.")
        run.note("")
        if args.fetch:
            with run.step(f"download {args.fetch} mesh(es) from `{args.fetch_source}`") as report:
                # Licence-filtered at the source: a model whose terms this project cannot accept is
                # skipped rather than downloaded and sorted out afterwards. An already-present mesh
                # is skipped too, so an interrupted fetch resumes.
                fetched = MeshPreparation.from_sources([args.fetch_source]).fetch(
                    limit=args.fetch, report=run.note)
                report(f"{fetched.fetched} fetched into {fetched.library}")
            run.note("")
            for line in fetched.render().splitlines():
                run.note(line)
            run.note("")
        if args.my_parts:
            failed = _add_your_parts(run, args.my_parts, licence=args.licence or "",
                                     attribution=args.attribution)
            if failed:
                return failed
            library = MeshPreparation.from_sources(["custom"])

        drawn_from = _PARTS_SOURCE[args.parts]
        with run.step("what the library holds") as report:
            # Counted for the collection this run draws from, not for the library. The default
            # sweep excludes `custom`, so a total across the public collections says nothing about
            # whether a `--parts custom` run has anything to place.
            available = (len(MeshPreparation.from_sources([drawn_from]).entries())
                         if drawn_from else 0)
            report(f"{available} mesh(es) in `{drawn_from}`" if drawn_from
                   else "generated objects; no mesh is read")
        if drawn_from and not available:
            # Before the render, not inside it. The scene layout refuses this too, and refuses it
            # well, but only once a build is under way and once per scene.
            return not_ready(
                f"`--parts {args.parts}` draws from `{drawn_from}` and it holds no mesh",
                "pass --fetch to download the public collections, or --my-parts DIR --licence own "
                "to copy in your own. `--parts procedural` needs neither")
        run.note("")
        run.note(f"  `--parts {args.parts}` sets "
                 + ", ".join(f"{key} {value}" for key, value in sorted(parts.items())))
        if args.parts == "procedural":
            run.note("  which is the shipped default: generated objects, no third-party licence in")
            run.note("  the chain at all, exact geometry and unlimited variation. It is the right")
            run.note("  answer for proving the chain closes and the wrong one for a cell, because")
            run.note("  a cell handles its own parts.")
        else:
            run.note("  the procedural weight is zeroed and the fallback is refused, which is the")
            run.note("  difference between a corpus of your objects and a corpus that is partly")
            run.note("  your objects. With the default False, an empty bank logs one warning for")
            run.note("  the whole render and the draws quietly become procedural.")
        run.note("")

        # ---------------------------------------------------------------- which parts a jaw holds
        if args.screen:
            run.note(f"screening for {args.screen} jaw-graspable mesh(es). Every mesh is labelled")
            run.note("  alone, in every rest pose, which is the part that takes minutes: an object")
            run.note("  that earns nothing standing up is often graspable lying down, and a screen")
            run.note("  that tries one pose grades the pose rather than the part.")
            run.note("")
            with run.step("screen your parts for graspability") as report:
                screen = library.screen(f"{args.out}/{args.name}_screen.json",
                                        want_graspable=args.screen, report=run.note)
                report(f"{screen.graspable} of {screen.rows} mesh(es) earn a jaw label")
            run.note("")
            for line in screen.render().splitlines():
                run.note(line)
            run.note("")
            run.note("  point `assets.jaw_screen_path` and `assets.mesh_asset_ids_path` at those")
            run.note("  two files to build only from the parts that earn a label. For the ones")
            run.note("  that earn none, `MeshPreparation.why_no_jaw(from_screen=...)` answers why,")
            run.note("  and the answer is usually width along the line tried rather than the")
            run.note("  fingers meeting the table underneath.")
            run.note("")

        # ---------------------------------------------------------------- the three data stages
        # One object for all three stages, built from the shipped defaults with this example's
        # settings layered on top. A customer who has described their own cell in a JSON file passes
        # it as the first argument instead of `None`, and nothing below changes.
        build = DatasetBuild.from_file(None, name=args.name, scenes=plan.requested_scenes,
                                       engine=args.engine, out_root=args.out,
                                       overrides={"assets": parts})
        run.note("")
        for line in build.describe().splitlines():
            run.note(line)
        run.note("")

        with run.step("render the scenes") as report:
            try:
                rendered = build.render()
            except ValueError as error:
                # The layout refuses a draw it cannot satisfy rather than filling the scene with
                # something else, and that refusal is a precondition rather than a crash. Without
                # this the reader gets a traceback out of a scene-authoring module.
                run.note(f"    {error}")
                return not_ready("the scene layout could not place the objects asked for",
                                 "read the line above: it names the weight that could not be "
                                 "satisfied and what to do about it")
            report(f"{rendered.summary.get('by_status', {}).get('ok', 0)} scene(s) "
                   f"under {build.root}")
        if not rendered.ok:
            # Exit 2, not 1. Nothing rendered means there was nothing to build with, and the
            # commonest cause is an engine this interpreter cannot reach.
            return not_ready(rendered.reason,
                             "`isaac` must run under Isaac's own python.bat; `none` and `mujoco` "
                             "run under this repository's interpreter")

        with run.step("label the grasps") as report:
            # Closed-form geometry from the settled poses. No GPU and no model: this is the step
            # that turns a pile of renders into supervision.
            labelled = build.label()
            jaw_labels = int(labelled.summary.get("jaw", 0))
            report(f"{jaw_labels} jaw and {labelled.summary.get('suction', 0)} suction label(s)")
        if not labelled.ok:
            # A finding does not move `run.exit_code`, so the status is returned explicitly. An
            # unlabelled dataset extracts into a structurally valid corpus that trains successfully
            # and teaches nothing, so stopping here is the whole point of checking the count.
            run.finding("label the grasps", labelled.reason)
            return EXIT_FAILED

        corpus = f"{args.out}/{args.name}_clouds"
        kinds = ("jaw", "suction") if args.suction else ("jaw",)
        with run.step("extract the corpus the trainer reads") as report:
            extracted = build.clouds(corpus, kinds=kinds)
            report(f"{extracted.summary.get('scenes_written', 0)} scene(s) in {corpus}")
        if not extracted.ok:
            run.finding("extract the corpus the trainer reads", extracted.reason)
            return EXIT_FAILED
        run.note("")
        if args.suction:
            run.note("  suction labels are carried, and they reach exactly one stage: the one that")
            run.note("  asks whether a patch of surface is worth attempting. The jaw half of the")
            run.note("  corpus is unchanged. 64 is the file about that limit.")
        else:
            run.note("  jaw labels only, which is the default. Far more objects earn a suction")
            run.note("  label than a jaw one, so a jaw-only corpus teaches the where stage to call")
            run.note("  an object empty that a cup could lift. `--suction` carries both.")
        run.note("")

        # ---------------------------------------------------------------- the model
        from src.contracts.options import UNSET
        from src.robot.grasping.deep.eval.run_report import build_report, format_report
        from src.robot.grasping.deep.train.api import GeneratorTraining
        from src.robot.grasping.deep.train.plan import PlanOverrides

        # Only what this example decides. Everything else stays UNSET, which is what lets a recipe
        # or a tier fill it in: an explicit value always outranks a bundle, so a setting written
        # here could never be supplied by one.
        overrides = PlanOverrides(epochs=epochs, run_folds=_RUN_FOLDS,
                                  refit=UNSET if (args.recipe or args.tier) else False)
        try:
            training = GeneratorTraining.from_recipe(
                corpus=corpus, recipe=args.recipe, tier=args.tier, overrides=overrides,
                out_dir=args.model_out, artifact_gripper=args.gripper)
        except (FileNotFoundError, ValueError) as error:
            # Construction refuses before the corpus walk and the probes, so a misspelled setting or
            # an ambiguous gripper costs a second rather than the first hour of a run.
            return not_ready(f"{type(error).__name__}: {error}",
                             "correct the setting the message names and run this again")
        run.note("")
        for line in training.describe().splitlines():
            run.note(line)
        run.note("")

        with run.step("train the generator") as report:
            result = training.train()
            report(f"{result.outcome}, {len(result.epochs)} epoch(s)")
        # Written whatever the outcome was: `report.json` carries stamps that exist nowhere else.
        training.write_report(result)
        run.note("")
        for line in result.render().splitlines():
            run.note(line)
        run.note("")
        if not result.succeeded:
            run.finding("train the generator", result.failure_summary())
            return EXIT_FAILED

        with run.step("the lift, which is the number") as report:
            card = build_report(Path(args.model_out).name, args.model_out)
            report(f"test lift {card.final.test_lift:+.4f} over a floor of "
                   f"{card.final.test_floor:.4f}")
        run.note("")
        for line in format_report(card).splitlines():
            run.note(line)
        run.note("")
        if card.collapsed:
            # A finding rather than a failure, and the exit code does not move. The chain did close:
            # every stage ran and an artifact exists. What did not happen is learning, and on a
            # first small run that is the ordinary outcome rather than a broken pipeline. The two
            # causes worth checking are named because they are the two that apply here.
            run.finding("the lift, which is the number",
                        f"the model sits on its own floor. {jaw_labels} jaw label(s) over "
                        f"{plan.usable_scenes} scene(s) at {epochs} epoch(s)")
            run.note("")
            run.note("  a run on its floor learned nothing yet, and nothing about the architecture")
            run.note("  or the corpus follows from it. Two things make it ordinary rather than")
            run.note("  fatal on a first pass: too few epochs, and too few jaw labels. Scale the")
            run.note("  scene count, then `--tier full`, and read the verdict in 63 rather than")
            run.note("  this line. Far more objects earn a suction label than a jaw one, so the")
            run.note("  two counts above are the honest measure of how much supervision a jaw")
            run.note("  head actually got.")
            run.note("")
        run.note("the chain closed. What it does not yet tell you:")
        run.note(f"  63_read_the_report.py --run {args.model_out} reads the curve properly, and")
        run.note("    `--dataset` there lists the assets this model has never seen. A lift")
        run.note("    measured on the objects it trained on is not a held-out number.")
        run.note("  64_what_the_generator_does_not_cover.py is the limit: jaw poses only, no")
        run.note("    calibrated probability, and the analytic stack keeps the reachability")
        run.note("    filter.")
        run.note("  the model card beside the weights records how they came to exist: the recipe")
        run.note("    and tier, whether they are one fold or a refit on everything, the target,")
        run.note("    and the corpus as a unit count, an asset-group count and the grippers. That")
        run.note("    is what lets two of your own models be told apart a year from now.")
        run.note("  serve it by setting grasping.calculator: deep and pointing")
        run.note("    grasping.deep_generator.artifact_path at the artifact under "
                 f"{args.model_out}.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
