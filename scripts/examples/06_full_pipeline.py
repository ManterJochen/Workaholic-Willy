"""06: every stage of the walkthrough, in the order a real cell needs them.

    python scripts/examples/06_full_pipeline.py                  # rehearse every stage
    python scripts/examples/06_full_pipeline.py --live --rig r1  # calibrate, then pick, for real
    python scripts/examples/06_full_pipeline.py --skip-corpus    # a cell that already has a model

Five of the six sibling examples run here, in dependency order: 01 robot_setup, 02 calibration,
04 datagen, 05 train, 03 pick. Two of the five are conditional, so a plain invocation runs four:
02 calibration runs only with `--rig`, and `--skip-corpus` drops 04 and 05. 07 sim is not in the
chain at all, because Isaac Sim has its own interpreter and cannot be started from this one.

Each stage is the same file you can run alone, called with the arguments you would type. Nothing
is re-implemented here, so a stage that fails in this pipeline fails the same way on its own, and
you debug one thing rather than two.

The order is a dependency chain, and every link is a failure a cell can produce:

    01 robot_setup  a config the safety layer already refuses cannot be diagnosed by connecting
                    to the cell it describes
    02 calibration  without extrinsics the driver refuses every motion as INVALID_TARGET, which
                    reads as a broken cell rather than as an uncalibrated one
    04 datagen      a corpus with no grasp labels trains successfully and teaches nothing
    05 train        weights, which reach a pick only through the cell's own config
    03 pick         last, because it is the only stage here that can move the arm

The step from 05 to 03 is the one this file does not take for you. A cell runs the learned
generator when `robot.grasping.calculator` is `deep` and `grasping.deep_generator.artifact_path`
names weights it can read. This pipeline edits no config, so 03 picks with whatever the cell
already selects, and the shipped selection is `geometric`.

It stops at the first blocking stage, on purpose. Continuing past a failed calibration would
produce a pick attempt whose result means nothing, and a green line under a red one is how a
broken cell gets signed off.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EXIT_FAILED, EXIT_OK, Example, hardware_parser  # noqa: E402

EPILOG = """
exit codes
  0  every stage that ran came out clean
  1  a stage came out wrong, and the run stopped there
  2  a stage could not start (no cell, no corpus, no config)
"""


def _stage(module_name: str, stage_argv: list[str]) -> int:
    """Run one sibling example in this process and return its exit code.

    The exit code is this pipeline's whole vocabulary for what happened, so it is handed straight
    back to the caller: 1 and 2 mean here exactly what they mean in the stage that returned them.

    `SystemExit` is caught because it is not an `Exception` and so passes through `Example.step`
    untouched, killing the pipeline before it can print its own summary. A stage's argument parser
    raises it whenever this file and that stage disagree about a flag, and a disagreement about
    arguments is a stage that could not start rather than a crash.
    """
    module = importlib.import_module(module_name)
    try:
        return int(module.main(stage_argv))
    except SystemExit as stop:
        if stop.code is None:
            return EXIT_OK
        return stop.code if isinstance(stop.code, int) else EXIT_FAILED


def main(argv: list[str] | None = None) -> int:
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument("--rig", default=None,
                        help="rig_id to calibrate. Omitted: the calibration stage is skipped")
    parser.add_argument("--prompt", default="an object", help="what the pick should look for")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="skip datagen + train; the cell already has what it needs")
    args = parser.parse_args(argv)

    shared = ["--profile", args.profile] if args.profile else []
    if args.live:
        shared += ["--live"]

    with Example("06 full_pipeline", "every stage, in dependency order",
                 live=args.live, profile=args.profile) as run:
        with run.step("01 robot_setup") as report:
            code = _stage("01_robot_setup", shared)
            report(f"exit {code}")
        if code != 0:
            run.finding("01 robot_setup", "the cell is not described coherently, so the run stops "
                                          "here: every later stage would be measuring a cell we "
                                          "have already said we cannot drive")
            # The stage's own code, not `run.exit_code`. `Example.finding` records a non-blocking
            # outcome, so this frame's code stays 0 and a failed stage would otherwise exit clean.
            return code

        if args.rig:
            with run.step("02 calibration") as report:
                code = _stage("02_calibration", ["--rig", args.rig, *shared])
                report(f"exit {code}")
            if code != 0:
                run.finding("02 calibration", "stopping: an uncalibrated cell refuses every motion")
                return code
        else:
            run.note("    02 calibration skipped (no --rig). The cell must already carry an")
            run.note("       extrinsics artifact, or the pick below will be refused.")

        if not args.skip_corpus:
            corpus_root, dataset = "logs/examples", "pipeline_corpus"
            with run.step("04 datagen") as report:
                code = _stage("04_datagen", ["--scenes", "6", "--engine", "none",
                                             "--name", dataset, "--out", corpus_root])
                report(f"exit {code}")
            if code != 0:
                run.finding("04 datagen", "stopping: there is nothing to train on")
                return code

            # `04_datagen.py` writes its point clouds to `<out>/<name>_clouds`, and this is the one
            # place the two files have to agree about that name.
            #
            # `--tier smoke` rather than an epoch count: the tier is the shipped name for a run
            # that only has to prove the chain closes, and it narrows the epochs, the units and
            # the refit together. Restating one of those three numbers here would leave the other
            # two saying something this pipeline did not mean.
            with run.step("05 train") as report:
                code = _stage("05_train", ["--clouds", f"{corpus_root}/{dataset}_clouds",
                                           "--tier", "smoke",
                                           "--out", "logs/examples/pipeline_model"])
                report(f"exit {code}")
            if code != 0:
                run.finding("05 train", "stopping: there are no weights to serve")
                return code
            run.note("    WARN the smoke tier is a wiring check, not a model. It proves the chain")
            run.note("       closes on this corpus and this box, and says nothing about grasp")
            run.note("       quality. The schedule to deploy from is `--tier full`, and")
            run.note("       05_train.py prices it before it starts.")

        with run.step("03 pick") as report:
            code = _stage("03_pick", ["--prompt", args.prompt, *shared])
            report(f"exit {code}")
        if code != 0:
            run.finding("03 pick", f"exit {code}")

        run.note("")
        run.note("What a clean run here proves, and what it does not:")
        run.note("  OK   the stages compose, and each one's refusals are reachable")
        if not args.live:
            run.note("  FAIL nothing about your hardware: no arm moved, no camera opened")
        else:
            run.note("  OK   this cell calibrated and executed a grasp")
            run.note("  FAIL not that it will keep doing so: one pick is one sample")
        # The last stage's code, for the same reason the earlier stages return theirs: a finding is
        # not blocking, so `run.exit_code` alone would report a failed pick as a clean pipeline.
        return code if code != 0 else run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
