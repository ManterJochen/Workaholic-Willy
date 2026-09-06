"""80: all of it, in the order a cell needs, stopping at the first blocking stage.

    python scripts/examples/pipeline/80_full_pipeline.py                  # rehearse every stage
    python scripts/examples/pipeline/80_full_pipeline.py --live --rig r1  # calibrate, then pick
    python scripts/examples/pipeline/80_full_pipeline.py --skip-corpus    # a cell with a model

The decision here is whether a stage's result is worth reading at all, and it is answered by
where the run stopped. Each stage is the same file you can run alone, called with the arguments
you would type, so nothing is re-implemented and a stage that fails here fails the same way on
its own. What this file adds is the order and the stop rule.

Four of the examples are stages, meaning each one gates the cell or produces something the next
one consumes. The rest are forks: they put two answers side by side and cost them, and a fork has
no place in a chain because there is nothing to hand on. The order is a dependency chain and
every link is a failure a cell can produce:

    01 cell           a config the safety layer already refuses cannot be diagnosed by
                      connecting to the cell it describes
    10 calibration    without extrinsics the driver refuses every motion as INVALID_TARGET,
                      which reads as a broken cell rather than as an uncalibrated one
    60 corpus         meshes to scenes to labels to point clouds to weights. It is a chain in
                      its own right, and it owns every step of it, so this file runs the one
                      stage rather than restaging its five
    03 pick           last, because it is the only stage here that can move the arm

Notice that this is not numerical order, and that is not an error. The numbers are the reading
order: the pick is 03 because it is the third thing to understand about a cell. Running order is
a different question, and it puts the pick last because it is the only stage that can move an
arm and the only one every other stage exists to serve.

Three of the four run by default. The calibration is opt-in, through `--calibrate` or through
`--rig` naming one, because it is 22 commanded moves and a cell that already carries an
extrinsics artifact does not need them again. Left out, it says what it is assuming, and a cell
that carries no artifact meets that same refusal in the pick instead.

The stop rule is the point. A green line under a red one is how a broken cell gets signed off:
continuing past a failed calibration produces a pick attempt whose result means nothing, and a
reader scanning the bottom of the log sees a pick that ran. So the run stops at the first
blocking stage and the summary names it.

Three exit codes carry that, and they are the stages' own rather than this file's. Zero is clean.
One is a stage that ran and came out wrong. Two is a stage that could not start, and the two are
different answers: "your cell is not connected" sends you to the cell, "the grasp failed" sends
you to the grasp. This file returns the code of the stage it stopped on, unchanged.

How it finds its siblings. Topic folders mean a bare module name no longer resolves, and the two
obvious repairs are both wrong. Hard-coding `cell.01_robot_setup` fixes the folder and the
filename into this file, so any rename breaks a pipeline that has nothing to do with the rename.
And putting `scripts/examples` on `sys.path` to make those dotted names importable stands a
directory called `datagen` next to the repository's own `datagen` package. That name collision is
latent rather than active today: a regular package with an `__init__.py` wins over a namespace
directory found earlier on the path, so the repository still wins it. It becomes real the moment
anybody adds an `__init__.py` to a topic folder, and the symptom is the datagen example's own
`from datagen.api import DatasetBuild` failing with no module named `datagen.api`. An import
mechanism that is one empty file away from changing what its stages import is not one to build a
pipeline on.

So a stage is addressed by its number and loaded from its path. The number is the contract: the
whole directory numbers globally so the reading order is unambiguous, which makes the number the
one identifier a reader, a README and this file already agree on. A file may be renamed or moved
between topic folders and this still resolves it; a number that is missing, or that two files
claim, is refused by name rather than guessed at.

The step from 60 to 03 is the one this file does not take for you. A cell runs the learned
generator when `robot.grasping.calculator` is `deep` and the deep generator's artifact path names
weights it can read. This pipeline edits no config, so 03 picks with whatever the cell already
selects, and the shipped selection is `geometric`. That fork is 41's subject, including what the
cell does when the artifact is missing.

Nothing here names a path. Every stage already has a default for the artifacts it writes and
reads, and 60 owns both ends of the corpus arc, so there is no handoff for this file to state.
Restating a sibling's paths would be a second copy of that sibling's interface, and it would go
stale the first time the sibling moved one. The only argument this file adds to a stage is
`--tier smoke` on 60, because the tier is the difference between a run that proves the chain
closes and one that costs a night.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import (  # noqa: E402
    EXIT_FAILED,
    EXIT_NOT_READY,
    EXIT_OK,
    Example,
    hardware_parser,
    not_ready,
)

EPILOG = """
exit codes
  0  every stage that ran came out clean
  1  a stage came out wrong, and the run stopped there
  2  a stage could not start (no cell, no corpus, no config), or this file could not find it
"""


@dataclass(frozen=True, slots=True)
class Stage:
    """One sibling example, addressed by its number."""

    number: str
    role: str
    #: Arguments beyond the shared ones, exactly as they would be typed.
    argv: tuple[str, ...] = ()
    #: Whether `--live` and `--profile` are passed on. False for the stages that touch no cell.
    shared: bool = True
    #: What is lost by stopping here, printed when this stage is the one that stopped the run.
    stopping_costs: str = ""
    #: What the reader has to have arranged some other way, printed when this stage is skipped.
    skip_costs: str = ""


@dataclass(frozen=True, slots=True)
class StageResult:
    """What one stage did: its own exit code, and whether it returned or raised it."""

    code: int
    #: True when the code arrived as a raised `SystemExit` rather than as a return value. In
    #: practice that means argparse: a stage's own refusals are returned from `main`, and the one
    #: thing that raises out of it is its parser rejecting an argument this file passed.
    raised: bool


def _typed(argv: list[str]) -> str:
    """The stage's arguments as a reader would have to type them.

    Values are quoted where they contain a space, with double quotes because those are the ones
    every shell this repository is run from agrees on. An unquoted `--prompt an object` is a
    different command from the one that ran, and this line's entire purpose is that a reader can
    copy it and get the stage on its own.
    """
    return " ".join(f'"{token}"' if " " in token else token for token in argv)


def _find(number: str, root: Path) -> list[Path]:
    """Every example under a topic folder whose filename starts with `number`.

    Returned as a list rather than resolved to one, so the caller can tell a number nobody claims
    from a number two files claim. Both are refused, and for different reasons: the first stage
    does not exist, the second breaks the reading order the number is here to express, and
    picking one of the two would hide that it happened.
    """
    return sorted(root.glob(f"*/{number}_*.py"))


def _load(path: Path) -> ModuleType:
    """Import one example from its path, under a name that cannot collide with a real package.

    The synthetic module name matters. Loading `datagen/50_build_a_corpus.py` as anything called
    `datagen` would put this file's idea of that name into `sys.modules`, where every later import
    finds it, and the repository has a `datagen` package of its own.
    """
    name = f"_willy_example_{path.stem}"
    cached = sys.modules.get(name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:                              # pragma: no cover
        raise ImportError(f"no import machinery for {path}")
    module = importlib.util.module_from_spec(spec)
    # Registered before execution, so a stage that imports itself sees the partially initialised
    # module rather than executing a second copy. This is what the import system does for a normal
    # import, and skipping it is the classic way one file becomes two objects.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _run(path: Path, argv: list[str]) -> StageResult:
    """Run one sibling example in this process and report its exit code and how it arrived.

    `SystemExit` is caught because it is not an `Exception`: it passes through `Example.step`
    untouched and would kill this file before it printed its own summary. A stage's parser raises
    it whenever this file and that stage disagree about a flag, and a disagreement about arguments
    is a stage that could not start rather than a crash, so it is reported as one.
    """
    module = _load(path)
    try:
        return StageResult(int(module.main(argv)), raised=False)
    except SystemExit as stop:
        if stop.code is None:
            return StageResult(EXIT_OK, raised=True)
        return StageResult(stop.code if isinstance(stop.code, int) else EXIT_FAILED, raised=True)


def main(argv: list[str] | None = None) -> int:
    parser = hardware_parser(__doc__ or "", epilog=EPILOG)
    parser.add_argument("--calibrate", action="store_true",
                        help="include the calibration stage. It is 22 commanded moves under "
                             "--live, so it is opt-in rather than opt-out")
    parser.add_argument("--rig", default=None,
                        help="rig_id to calibrate, which implies --calibrate. Omitted, the "
                             "calibration stage resolves the cell's own primary RGB-D rig")
    parser.add_argument("--prompt", default="an object", help="what the pick should look for")
    parser.add_argument("--skip-corpus", action="store_true",
                        help="skip the corpus stage, which builds and trains; the "
                             "cell already has what it needs")
    args = parser.parse_args(argv)

    shared = ["--profile", args.profile] if args.profile else []
    if args.live:
        shared += ["--live"]

    stages = [
        Stage("01", "cell", shared=True,
              stopping_costs="every later stage would be measuring a cell this run has already "
                             "said it cannot drive"),
        # `--rig` only where the caller named one. The calibration example resolves the primary
        # RGB-D rig from the config itself and refuses by name when there is none, so passing a
        # rig this file invented would replace an answer the config already has.
        Stage("10", "calibration", shared=True,
              argv=(("--rig", args.rig) if args.rig else ()),
              stopping_costs="an uncalibrated cell refuses every motion, so a pick after this "
                             "would be measuring the refusal",
              skip_costs="This run assumes the cell already carries an extrinsics artifact. If "
                         "it does not, the pick below refuses for the same reason, one stage "
                         "later. Pass --calibrate, or --rig to name the camera."),
        # `--tier smoke` rather than an epoch count: the tier is the shipped name for a run that
        # only has to prove the chain closes, and it narrows the epochs, the training units and
        # the refit together. Restating one of those three numbers here would leave the other two
        # saying something this pipeline did not mean.
        Stage("60", "corpus", shared=False, argv=("--tier", "smoke"),
              stopping_costs="there are no weights to serve",
              skip_costs="A cell set to the deep calculator needs weights from somewhere, and "
                         "this run produced none. Nothing else is lost: the pick below runs the "
                         "calculator the config selects, and the shipped selection is geometric."),
        Stage("03", "pick", argv=("--prompt", args.prompt), shared=True,
              stopping_costs="nothing, this is the last stage"),
    ]
    # Calibration is opt-in rather than opt-out, and that asymmetry is deliberate. Under --live
    # it is 22 commanded moves, and re-running it on a cell that is already calibrated is 22
    # motions nobody asked for; that is the same reason --live itself exists. The stage it gates
    # is not skipped quietly, because the assumption it leaves standing is printed.
    skipped = {"60"} if args.skip_corpus else set()
    if not (args.calibrate or args.rig):
        skipped.add("10")

    root = Path(__file__).resolve().parents[1]
    with Example("80 full_pipeline", "every stage, in dependency order",
                 live=args.live, profile=args.profile) as run:
        for stage in stages:
            if stage.number in skipped:
                skip = f"    {stage.number} {stage.role} skipped by request"
                run.note(f"{skip}. {stage.skip_costs}" if stage.skip_costs else skip)
                continue

            found = _find(stage.number, root)
            if len(found) != 1:
                # Refused, not skipped. A stage this file cannot find is a stage that did not run,
                # and continuing would produce a summary in which a missing calibration and a
                # skipped one look the same.
                claimed = ", ".join(f"{p.parent.name}/{p.name}" for p in found) or "nothing"
                return not_ready(
                    f"stage {stage.number} ({stage.role}) is claimed by {len(found)} files under "
                    f"{root}: {claimed}",
                    f"exactly one example must be named {stage.number}_*.py, in exactly one topic "
                    f"folder. The numbering runs globally across the folders for this reason: it "
                    f"is the reading order, so a number is an address")
            path = found[0]

            stage_argv = [*stage.argv, *(shared if stage.shared else [])]
            # Printed before the step rather than inside it, because a step opens a line that the
            # stage's own output lands on. It is also the command a reader can type to get the
            # same stage on its own, which is the claim this whole file rests on.
            run.note(f"    {path.parent.name}/{path.name} {_typed(stage_argv)}")
            with run.step(f"{stage.number} {stage.role}") as report:
                result = _run(path, stage_argv)
                report(f"exit {result.code}")

            if result.code == EXIT_OK:
                continue
            if result.raised:
                # An exit code out of a raised SystemExit is argparse, which means this file
                # passed an argument that stage does not have. That is worth saying plainly: it
                # is a fault in the chain rather than in the cell being tested.
                run.finding(f"{stage.number} {stage.role}",
                            "the stage rejected the arguments printed above, so it never ran. "
                            "Its own error line says which argument")
            else:
                run.finding(f"{stage.number} {stage.role}",
                            ("could not start" if result.code == EXIT_NOT_READY else "came out "
                             "wrong") + f", so the run stops here: {stage.stopping_costs}")
            # The stage's own code, not `run.exit_code`. `Example.finding` records a non-blocking
            # outcome, so this frame's code stays 0 and a failed stage would otherwise exit clean.
            return result.code

        run.note("")
        run.note("What a clean run here proves, and what it does not:")
        run.note("  OK   the stages compose, and each one's refusals are reachable")
        if not args.live:
            run.note("  FAIL nothing about your hardware: no arm moved, no camera opened")
        else:
            run.note("  OK   this cell calibrated and executed a grasp")
            run.note("  FAIL not that it will keep doing so: one pick is one sample")
        if not args.skip_corpus:
            run.note("  FAIL not a model. The smoke tier proves the chain closes on this corpus")
            run.note("       and this box, and says nothing about grasp quality; the schedule to")
            run.note("       deploy from is the full tier, which 60 prices before it starts.")
        run.note("")
        run.note("70_sim_pick.py is deliberately not in this chain. Isaac Sim ships its own")
        run.note("interpreter, so it cannot be started from this one, and the sim answers a")
        run.note("different question than a cell does.")
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
