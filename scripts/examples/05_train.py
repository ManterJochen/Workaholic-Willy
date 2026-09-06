"""05: train a grasp generator on your corpus, and read its card honestly.

    python scripts/examples/05_train.py --clouds logs/examples/example_corpus_clouds
    python scripts/examples/05_train.py --clouds <dir> --recipe v1 --tier full
    python scripts/examples/05_train.py --clouds <dir> --price-only

The epoch count is the expensive decision, so this example prices the run before it starts it. The
price and the run read the same two settings, epochs and passes, so the two cannot disagree.

One pass or two. `SetTrainingPlan.refit` is off, so a plain run fits one fold and stops there.
`--recipe v1` turns the refit on, which is a second training run over every unit and twice the
hours, and `--tier smoke` turns it off again. `datagen.cost.estimate` defaults the other way, so
this example passes it the value it is about to run with rather than letting the calculator assume.

How to read what comes out, because the two obvious numbers are the wrong ones:

  the loss says almost nothing.   A bigger model drives it down faster; that is what parameters do.
  the raw hit rate says nothing.  Every split has its own floor, the rate a constant predictor
                                  reaches, and two runs with different floors are not comparable.

The number is the lift: the hit rate minus its own floor, on the held-out side. This example prints
it and prints whether the run was still improving when it stopped, which is the only question that
decides what to run next. Both come from `deep.eval.run_report`, computed in this process off the
`epochs.json` the trainer has just written.

The architecture is not a flag here. `--recipe` names a frozen, versioned bundle of settings that is
stamped into the artifact, and `--tier` says how long to run; the network itself is reachable from
code as `PlanOverrides(backbone_width=..., backbone_depth=..., backbone_heads=...)` and `slots`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the model trained and its card was written
  1  training ran and something came out wrong
  2  nothing to train on: no corpus, it carries no grasp labels, or the recipe does not exist
"""

#: Fold passes to run. One is a diagnostic rather than a cross-validation, and both the price below
#: and the model card say which it was.
_RUN_FOLDS: Final[int] = 1

#: Epochs when neither the caller nor a tier names a number. Small on purpose: the first run on a
#: new corpus is there to prove the chain closes, and `--tier full` is the schedule to deploy from.
_SMOKE_EPOCHS: Final[int] = 10


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clouds", required=True, help="the .npz corpus from 04_datagen")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"epochs to run. Default: what --tier asks for, else {_SMOKE_EPOCHS}")
    parser.add_argument("--recipe", default=None,
                        help="a named, versioned settings bundle, stamped into the artifact")
    parser.add_argument("--tier", default=None,
                        help="how long to run: `smoke` proves the chain closes, `full` is the "
                             "model you deploy")
    parser.add_argument("--gripper", default=None,
                        help="which hand the artifact plans for. Taken from the corpus when it "
                             "carries exactly one, and refused rather than guessed otherwise")
    parser.add_argument("--out", default="logs/examples/model", help="where the artifact goes")
    parser.add_argument("--no-refit", action="store_true", help="one pass instead of two")
    parser.add_argument("--price-only", action="store_true", help="print the bill and stop")
    args = parser.parse_args(argv)

    # `scene_files` refuses both of these too, at construction, with the same two messages. They are
    # checked here as well because a missing corpus is a precondition rather than a failure: this
    # returns exit 2 and a line saying which command produces one, where the library raises.
    clouds = Path(args.clouds)
    if not clouds.is_dir():
        return not_ready(f"no corpus at {clouds}",
                         "run 04_datagen.py first; it prints the corpus path at the end")
    scenes = list(clouds.rglob("*.npz"))
    if not scenes:
        return not_ready(f"{clouds} holds no .npz scenes",
                         "extract one with `DatasetBuild.clouds(out_dir)`, which is the third "
                         "step of 04_datagen.py")

    from datagen.cost import estimate
    from src.robot.grasping.deep.train.recipes import settings

    # The recipe and the tier resolved once, here, and read by both the price and the run.
    # `settings` is the same function `build_plan` calls, so the bill cannot describe one schedule
    # while another executes. It refuses an unknown name rather than falling back to the defaults.
    try:
        schedule = settings(args.recipe, args.tier)
    except ValueError as error:
        return not_ready(str(error),
                         "pass one of the names that message lists, or leave --recipe and --tier "
                         "off to run the shipped defaults")
    epochs = args.epochs if args.epochs is not None else int(schedule.get("epochs", _SMOKE_EPOCHS))
    # Absent from the bundle means the trainer's own default, which is a single pass. Two passes
    # come from a recipe that asks for the refit, never from the plain defaults.
    refit = False if args.no_refit else bool(schedule.get("refit", False))

    named = " ".join(part for part in (f"recipe {args.recipe}" if args.recipe else "",
                                       f"tier {args.tier}" if args.tier else "") if part)
    with Example("05 train", f"{len(scenes)} scene(s), {epochs} epoch(s)"
                             + (f", {named}" if named else ""), hardware=False) as run:
        with run.step("check the corpus carries grasp labels") as report:
            # The failure this catches costs a whole run and explains nothing. A corpus extracted
            # without `label-grasps` is structurally valid and completely unsupervised: training
            # succeeds and the card comes back with the grasp metrics simply absent. Nothing in the
            # library enforces this, unlike the two checks above. Reading one scene costs
            # milliseconds and turns a confusing artifact into a refusal with a fix.
            import numpy as np

            with np.load(scenes[0], allow_pickle=False) as handle:
                labelled = (int(np.asarray(handle["grasp_width_mm"]).size)
                            if "grasp_width_mm" in handle.files else 0)
            report(f"{labelled} grasp label(s) in the first scene")
        if not labelled:
            return not_ready(
                f"{scenes[0].name} carries no grasp labels",
                "label the dataset with `DatasetBuild.label()` before extracting the corpus; "
                "training on an unlabelled corpus succeeds and teaches nothing")

        with run.step("price the run") as report:
            passes = _RUN_FOLDS + (1 if refit else 0)
            plan = estimate(len(scenes), engine="none", label=False, corpus=False,
                            epochs=epochs, folds=_RUN_FOLDS, refit=refit)
            train_hours = next(s.hours for s in plan.stages if s.name == "train")
            report(f"{passes} pass(es) x {epochs} epochs is about {train_hours:.2f} h")
        if refit:
            run.note("    two passes: the folds earn the numbers, then a refit trains one more net")
            run.note("      on every unit and that is the one shipped. `--no-refit` halves this")
            run.note("      is the honest comparison between two arms.")

        if args.price_only:
            run.note("--price-only: nothing was trained.")
            return run.exit_code

        from src.contracts.options import UNSET
        from src.robot.grasping.deep.eval.run_report import build_report, format_report
        from src.robot.grasping.deep.train.api import GeneratorTraining
        from src.robot.grasping.deep.train.plan import PlanOverrides

        # Only what this example decides. Everything else stays UNSET, which is what lets the recipe
        # and the tier fill it in: an explicit value always wins over a bundle, so a setting written
        # here can never be supplied by one. `--no-refit` is passed as a value rather than left out,
        # because turning the recipe's refit off is a choice and has to be distinguishable from
        # saying nothing.
        overrides = PlanOverrides(epochs=epochs, run_folds=_RUN_FOLDS,
                                  refit=False if args.no_refit else UNSET)
        try:
            training = GeneratorTraining.from_recipe(
                corpus=str(clouds), recipe=args.recipe, tier=args.tier, overrides=overrides,
                out_dir=args.out, artifact_gripper=args.gripper)
        except (FileNotFoundError, ValueError) as error:
            # Construction refuses before the corpus walk and the probes, so a misspelled setting or
            # a backbone width no head count divides costs a second rather than minutes.
            return not_ready(f"{type(error).__name__}: {error}",
                             "correct the setting the message names and run this again")

        run.note("")
        for line in training.describe().splitlines():
            run.note(line)
        run.note("")

        with run.step("train the generator") as report:
            result = training.train()
            final = result.final
            report(f"{result.outcome}, {len(result.epochs)} epoch(s)"
                   + (f", held top1 {final.held_top1_hit:.4f}" if final is not None else ""))
        # Written whatever the outcome was. `report.json` carries five stamps that exist nowhere
        # else, and the analysis below merges it when `epochs.json` has no floor block.
        training.write_report(result)
        run.note("")
        for line in result.render().splitlines():
            run.note(line)
        run.note("")
        if not result.succeeded:
            # A finding does not move `run.exit_code`, so the status is returned explicitly. A run
            # whose folds completed and whose artifact was refused left the operator no weights,
            # which is a failure however good the numbers above it are.
            run.finding("train the generator", result.failure_summary())
            return EXIT_FAILED

        try:
            card = build_report(Path(args.out).name, args.out)
        except ValueError as error:
            run.finding("read the lift", str(error))
            return EXIT_FAILED

        with run.step("the lift, which is the number") as report:
            report(f"test lift {card.final.test_lift:+.4f} over a floor of "
                   f"{card.final.test_floor:.4f}")
        run.note("")
        for line in format_report(card).splitlines():
            run.note(line)

        run.note("")
        run.note("WARN the card beside the weights records how they came to exist: the recipe and")
        run.note("  tier, whether they are one fold or a refit on everything, the target, whether")
        run.note("  it was a control run, and the corpus as a unit count, an asset-group count and")
        run.note("  the grippers. That is what lets two of your own models be told apart later.")
        run.note("")
        run.note("STOP before trusting any number here: measure on assets the model has not seen.")
        run.note("  `datagen.heldout.held_out_assets(dataset_root)` builds that list and")
        run.note("  `datagen.heldout.format_report` prints it. The argument is the dataset")
        run.note("  directory 04_datagen.py wrote, not the cloud corpus trained on here.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
