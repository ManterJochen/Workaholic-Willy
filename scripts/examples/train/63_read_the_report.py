"""63: read what a training run produced, and read the one number that means anything.

    python scripts/examples/train/63_read_the_report.py --run logs/examples/model
    python scripts/examples/train/63_read_the_report.py --clouds logs/examples/example_corpus_clouds
    python scripts/examples/train/63_read_the_report.py --run <dir> --curve logs/examples/curve.png

The decision: what to run next. A finished run offers four numbers and three of them cannot answer
that, so this example prints all four and says which is which.

  the loss              says almost nothing. A bigger model drives it down faster; that is what
                        parameters do, and it falls in a run that is learning nothing about grasps.
  the raw hit rate      says nothing on its own. Every split has a floor, the rate a constant
                        predictor reaches, and two runs with different floors are not comparable.
  the best epoch        is a maximum over many noisy draws, not a level the model ever held.
  the lift              is the number: the hit rate minus its own measured floor, on the held-out
                        side, read in windows rather than at a point.

And one verdict on top of them, which is what actually decides the next run: was it still improving
when it stopped. A single value is a good result or a wasted day depending entirely on that, and the
report declines to answer it below two full windows rather than printing a placeholder that reads
like a converged run.

Two ways in. `--run DIR` reads a run that already exists. `--clouds DIR` trains a short one first,
which is the honest way to see what a smoke tier produces: a two-epoch run is not comparable, and
the report says so in those words instead of grading it.

Everything printed here is computed in this process from the `epochs.json` the trainer wrote after
every epoch. Nothing is scraped from a log: a report that reads a human-readable line for numbers
that exist as JSON one directory over is how a report comes to disagree with the run it describes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Final

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  the run was read and its verdict printed
  1  the run came out wrong: it sits on its own floor, or training did not finish
  2  nothing to read: no run directory, no corpus to make one from, or no grasp labels
"""

#: Folds to fit when this example trains the run itself. One is a diagnostic rather than a
#: cross-validation, and the model card records which it was.
_RUN_FOLDS: Final[int] = 1

#: Epochs when the caller names none and no tier does either. Small on purpose: a run started here
#: exists to be read, and `--tier full` is the schedule to deploy from.
_SHORT_EPOCHS: Final[int] = 2


def _train_a_run(run: Example, clouds: Path, out: str, *, epochs: int | None,
                 recipe: str | None, tier: str | None, gripper: str | None) -> int:
    """Fit a short run so there is something to read. Returns an exit code, 0 to carry on.

    Separate from the reading below because the two are separate jobs: a customer with a finished
    run never enters this function, and one who has only a corpus needs every check in it.
    """
    scenes = list(clouds.rglob("*.npz"))
    if not scenes:
        return not_ready(f"{clouds} holds no .npz scenes",
                         "extract one with `DatasetBuild.clouds(out_dir)`, which 60 runs as its "
                         "sixth step")

    with run.step("check the corpus carries grasp labels") as report:
        # The failure this catches costs a whole run and explains nothing. A corpus extracted
        # without the label stage is structurally valid and completely unsupervised: training
        # succeeds and the report comes back with the grasp metrics simply absent. Nothing in the
        # library enforces it, and reading one scene costs milliseconds.
        import numpy as np

        with np.load(scenes[0], allow_pickle=False) as handle:
            labelled = (int(np.asarray(handle["grasp_width_mm"]).size)
                        if "grasp_width_mm" in handle.files else 0)
        report(f"{labelled} grasp label(s) in the first scene")
    if not labelled:
        return not_ready(f"{scenes[0].name} carries no grasp labels",
                         "label the dataset before extracting the corpus; training on an "
                         "unlabelled corpus succeeds and teaches nothing")

    from src.contracts.options import UNSET
    from src.robot.grasping.deep.train.api import GeneratorTraining
    from src.robot.grasping.deep.train.plan import PlanOverrides
    from src.robot.grasping.deep.train.recipes import settings

    try:
        schedule = settings(recipe, tier)
    except ValueError as error:
        return not_ready(str(error), "62_recipe_and_tier.py lists the names that exist")
    # An explicit value always outranks a bundle, so a number written here can never be supplied by
    # one. `epochs` is therefore left UNSET unless the caller or a tier named it.
    chosen = epochs if epochs is not None else schedule.get("epochs", _SHORT_EPOCHS)
    overrides = PlanOverrides(epochs=int(chosen), run_folds=_RUN_FOLDS,
                              refit=UNSET if (recipe or tier) else False)
    try:
        training = GeneratorTraining.from_recipe(
            corpus=str(clouds), recipe=recipe, tier=tier, overrides=overrides,
            out_dir=out, artifact_gripper=gripper)
    except (FileNotFoundError, ValueError) as error:
        # Construction refuses before the corpus walk and the probes, so a misspelled setting or a
        # backbone width no head count divides costs a second rather than minutes.
        return not_ready(f"{type(error).__name__}: {error}",
                         "correct the setting the message names and run this again")

    run.note("")
    for line in training.describe().splitlines():
        run.note(line)
    run.note("")

    with run.step("train, so there is a curve to read") as report:
        result = training.train()
        report(f"{result.outcome}, {len(result.epochs)} epoch(s)")
    # Written whatever the outcome was. `report.json` carries stamps that exist nowhere else, and
    # the reader below merges it when `epochs.json` has no floor block.
    training.write_report(result)
    if not result.succeeded:
        # A finding does not move `run.exit_code`, so the status is returned explicitly. A run whose
        # epochs completed and whose artifact was refused left no weights behind, which is a failure
        # however good the numbers above it are.
        run.finding("train, so there is a curve to read", result.failure_summary())
        return EXIT_FAILED
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default=None,
                        help="a finished run directory, the one holding epochs.json")
    parser.add_argument("--clouds", default=None,
                        help="a corpus to train a short run on first, when you have no run yet")
    parser.add_argument("--out", default="logs/examples/model",
                        help="where a run started here writes its weights and its report")
    parser.add_argument("--epochs", type=int, default=None,
                        help=f"epochs for a run started here. Default: the tier's, else "
                             f"{_SHORT_EPOCHS}")
    parser.add_argument("--recipe", default=None, help="see 62_recipe_and_tier.py")
    parser.add_argument("--tier", default=None, help="see 62_recipe_and_tier.py")
    parser.add_argument("--gripper", default=None,
                        help="which hand the artifact plans for. Taken from the corpus when it "
                             "carries exactly one, and refused rather than guessed otherwise")
    parser.add_argument("--curve", default=None,
                        help="write the learning curve to this PNG. Needs matplotlib")
    parser.add_argument("--dataset", default=None,
                        help="the dataset directory the corpus came from, to list the assets this "
                             "model has never seen")
    args = parser.parse_args(argv)

    if not args.run and not args.clouds:
        return not_ready("no run to read and no corpus to make one from",
                         "pass --run <a directory holding epochs.json>, or --clouds <a corpus> to "
                         "train a short one here. 60_your_own_corpus.py produces both")

    directory = Path(args.run) if args.run else Path(args.out)
    with Example("63 read the report", f"the run at {directory}", hardware=False) as run:
        if args.clouds:
            clouds = Path(args.clouds)
            if not clouds.is_dir():
                return not_ready(f"no corpus at {clouds}",
                                 "run 60_your_own_corpus.py first; it prints the corpus path at "
                                 "the end")
            failed = _train_a_run(run, clouds, args.out, epochs=args.epochs, recipe=args.recipe,
                                  tier=args.tier, gripper=args.gripper)
            if failed:
                return failed

        from src.robot.grasping.deep.eval.run_report import build_report, format_report

        with run.step("read the epoch rows the trainer wrote") as report:
            try:
                card = build_report(directory.name, directory)
            except (FileNotFoundError, ValueError) as error:
                return not_ready(f"{type(error).__name__}: {error}",
                                 "point --run at the directory holding epochs.json, which is the "
                                 "--out of the training run")
            report(f"{len(card.epochs)} epoch(s) across {len(card.segments)} sitting(s), "
                   f"{card.compute_hours:.2f} h of compute")
        run.note("")
        run.note("  a resumed run is one experiment split across wall-clock gaps, so the x axis")
        run.note("  is epochs and never time. The sittings are stated rather than drawn as a flat")
        run.note("  spot in a curve, which would mean nothing.")
        run.note("")

        with run.step("the three numbers that are not the answer") as report:
            report(f"loss {card.final.loss:.4f}, raw test hit {card.final.test_hit:.4f}, "
                   f"best epoch {card.best.epoch}")
        run.note("")
        run.note(f"  the loss is {card.final.loss:.4f}. A wider net reaches a lower one on the")
        run.note("  same corpus without proposing a single better grasp, so it ranks")
        run.note("  architectures by capacity rather than by usefulness.")
        run.note(f"  the raw hit rate is {card.final.test_hit:.4f} against a floor of")
        run.note(f"  {card.final.test_floor:.4f}, and the floor is what a constant predictor")
        run.note("  reaches on this split. Quoting the first without the second compares two runs")
        run.note("  that were never measured against the same thing.")
        run.note(f"  the best epoch is {card.best.epoch} at {card.best.test_lift:+.4f}. It is the")
        run.note(f"  maximum over {len(card.epochs)} noisy draws, so it is a level the model")
        run.note("  touched once rather than one it held.")
        run.note("")

        with run.step("the lift, which is the number") as report:
            report(f"test lift {card.final.test_lift:+.4f} over a floor of "
                   f"{card.final.test_floor:.4f}")
        run.note("")
        for line in format_report(card).splitlines():
            run.note(line)
        run.note("")

        with run.step("the verdict that decides the next run") as report:
            if not card.comparable:
                verdict = (f"no verdict: {len(card.epochs)} epoch(s) is under two windows")
            elif card.still_improving:
                verdict = f"still improving over the last {card.window} epoch(s)"
            else:
                verdict = f"flattened, judged over the last {card.window} epoch(s)"
            report(verdict)
        run.note("")
        if not card.comparable:
            run.note("  under two full windows there is nothing to compare, so the report gives no")
            run.note("  verdict at all. That is the honest answer for a smoke run, and it is why a")
            run.note("  smoke tier proves the chain closes and settles nothing else. The")
            run.note("  alternative, a placeholder comparing the mean against itself, prints")
            run.note("  `it had flattened` on a run four epochs into a thirty-six-epoch schedule.")
        elif card.still_improving:
            run.note("  the schedule is the binding constraint, not the architecture. Run more")
            run.note("  epochs before changing anything else, and treat any arm compared against")
            run.note("  this one at fewer epochs as a different experiment rather than a worse")
            run.note("  architecture.")
        else:
            run.note("  more epochs are not the lever. The open questions are the resolution, the")
            run.note("  target and the corpus, in that order, and each of them is a different run")
            run.note("  rather than a longer one.")
        run.note("")
        if not card.resolvable:
            run.note("  and read the [BELOW RESOLUTION] line above before acting on any of that:")
            run.note("  when the whole curve spans less than the band the verdict is judged")
            run.note("  against, every epoch sits inside that band by arithmetic, so no plateau")
            run.note("  claim follows from it.")
            run.note("")

        if args.curve:
            with run.step("draw the curve") as report:
                from src.robot.grasping.deep.eval.run_report import write_curve

                try:
                    written = write_curve(card, args.curve)
                except ImportError as error:
                    run.finding("draw the curve", f"matplotlib is not installed: {error}")
                else:
                    report(f"{written}")
            run.note("")
            run.note("  the floors are drawn as lines rather than described. A reader who sees the")
            run.note("  curve without them reads the test line as rising from zero.")
            run.note("")

        if args.dataset:
            from datagen.heldout import format_report as format_heldout
            from datagen.heldout import held_out_assets

            with run.step("which assets this model has never seen") as report:
                held = held_out_assets(args.dataset)
                report(f"{held.total_unseen} unseen asset(s) over "
                       f"{held.trained_group_count} trained fold group(s)")
            run.note("")
            for line in format_heldout(held).splitlines():
                run.note(line)
            run.note("")

        if card.collapsed:
            # A finding does not move `run.exit_code`, so the status is returned explicitly. A run
            # sitting exactly on its own floor learned nothing, and no conclusion about the
            # architecture, the target or the corpus may be drawn from it.
            run.finding("the lift, which is the number",
                        "the final test metric sits on its own floor")
            return EXIT_FAILED

        run.note("STOP a lift measured on assets the model trained on is not a held-out number,")
        run.note("  whatever it says. `datagen.heldout.held_out_assets(dataset_root)` builds the")
        run.note("  list of assets it has never seen and `--dataset` prints it here. The argument")
        run.note("  is the dataset directory, not the cloud corpus that was trained on. The fold")
        run.note("  key is the asset group rather than the asset id, so the procedural variants of")
        run.note("  one object stay on one side of the split: the honest question is whether it")
        run.note("  can grasp a jug it has never seen, not a seventh jug having seen six.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
