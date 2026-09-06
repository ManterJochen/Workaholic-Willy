"""41: `robot.grasping.calculator`, and the refusal when the learned generator has no weights.

    python scripts/examples/grasping/41_geometric_or_deep.py
    python scripts/examples/grasping/41_geometric_or_deep.py --bad-artifact
    python scripts/examples/grasping/41_geometric_or_deep.py --profile sim

One key decides which stack proposes grasp candidates. `geometric` is the analytic generator that
ships and runs: a candidate stage (the support footprint or the silhouette), a deterministic
three-axis geometric rank, then the whole rejection tail, the table check, the gripper envelope,
the antipodal test, the corridor filter and the reachability filter. `deep` is a learned 6-DoF
generator that replaces the proposal stage and decodes its own approach and closing axis. It has
no rejection tail of its own; those stages live in the analytic generator, and extracting them into
a stage both generators run is an open build.

No trained weights ship in this repository, deliberately. A customer trains on their own cell's
data, so `deep` here is a fork whose second branch refuses, and the refusal names the command that
would produce the missing file. This example triggers it and prints it.

Selecting `deep` fails closed. It does not fall back to `geometric`, because a cell that asked for
the learned generator and quietly got the analytic one would file every record, every KPI and every
ladder row under the wrong name, and "which one ran" would be unanswerable from outside. The check
is a stamped kind and artifact version, so a fold checkpoint from a run in progress, or weights
from the retired generator, are refused by name rather than loaded. `--bad-artifact` shows that
second refusal; it costs a torch import, which the first one does not.

What `deep` also costs, beyond the weights: the factory carries only the camera matrix, the
candidate cap and the two grip-width bounds across. The analytic construction knobs that shape the
proposal, the oblique-approach sweep and isotropic radial closing among them, have nothing to act
on in a decoder that predicts its own approach, so they are dropped with a log line. Two others are
refused outright when a construction site passes them with an active value, because they change
what a pick does rather than how it looks: the reachability filter, without which nothing discards
an unreachable candidate and `rejected_ik` reads zero, and the per-candidate corridor risk
producer, without which the uncertainty re-rank reorders nothing.

The safety preflight is unaffected by this key. It sits at the arm and gates the joint target
whatever proposed it, so a deep candidate meets exactly the same fail-closed guards.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

EPILOG = """
exit codes
  0  both answers behaved as documented: one built and proposed, the other built or refused
  1  the configured generator built and proposed nothing, or the other one failed untypeably
  2  the config tree has no `robot` block, so there is no generator to select
"""

#: How wide the printed refusal is wrapped. `Example.note` indents by four, so this leaves the
#: whole message inside a 96-column terminal.
_WRAP = 88


def _quote(run: Any, text: str) -> None:
    """Print a library message as its own block, wrapped, without editing a word of it.

    Always called after a step has closed, never inside one. `Example.step` leaves the
    announcement line open on a terminal, so anything printed from inside the body lands on it,
    and off a terminal the block would appear above the verdict it belongs to.
    """
    for line in textwrap.wrap(text, _WRAP) or [""]:
        run.note(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__ or "", epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bad-artifact", action="store_true",
        help="also point the deep branch at a readable torch file that is not a generator "
             "artifact, and print the second refusal. Imports torch, which the run otherwise "
             "does not.")
    parser.add_argument(
        "--profile", default=None,
        help="WILLY_PROFILE chain to load (e.g. `sim`, `ur3e`). Default: the environment's.")
    args = parser.parse_args(argv)

    # After the parse, so `--help` answers without importing the stack.
    from src.config import default_data_dir, load_config
    from src.config.explain import explain_in
    from src.robot.execution.autonomous_grasp.rehearsal import RehearsalPerceptionSource
    from src.robot.grasping.calculator_factory import build_calculator, preflight_calculator

    with Example("41 geometric or deep", "which generator proposes the candidates",
                 hardware=False) as run:
        with run.step("which generator this cell selects") as report:
            tree = load_config(profile=args.profile) if args.profile else load_config()
            robot = tree.robot
            report("no `robot` block in this tree" if robot is None else
                   f"calculator {robot.grasping.calculator}, "
                   f"artifact {robot.grasping.deep_generator.artifact_path or '<none>'}")
        if robot is None:
            return not_ready("this config tree has no `robot` block",
                             "point the loader at a tree that has one; `config/robot/robot.yaml` "
                             "in this repository is one")

        # The schema's own account of the key: its type, its default, and which file set it. A
        # selector that no YAML writes is running on its schema default, and that is worth seeing
        # before anything is built on top of it.
        layers = tuple(part for part in (args.profile or "").split(",") if part)
        for line in explain_in(tree, "robot.grasping.calculator",
                               default_data_dir(), layers).render().splitlines():
            run.note(line)
        run.note("")

        selected = str(robot.grasping.calculator)
        other = "deep" if selected == "geometric" else "geometric"
        flipped = robot.model_copy(update={
            "grasping": robot.grasping.model_copy(update={"calculator": other})})

        said: str = ""
        with run.step("check the selector without building anything") as report:
            # What a sweep calls once, before its first scene. `build_calculator` fails closed, so
            # a loop that constructs one calculator per scene inside `except Exception: continue`
            # turns one configuration error into N warnings and a report full of tidy zeros, which
            # reads as a dead feature rather than as a wrong path.
            try:
                report(f"preflight_calculator says {preflight_calculator(robot)!r}")
            except (FileNotFoundError, ValueError) as refusal:
                report(f"refused before building: {type(refusal).__name__}")
                said = str(refusal)
        if said:
            _quote(run, said)

        # One scene for whichever generator is built, so the two branches are comparable. It is a
        # synthetic box at a known place: what is exercised is the selector and the wiring, never
        # grasp quality, which is measured in simulation and on the bench.
        scene = RehearsalPerceptionSource()
        frame = scene.acquire()
        proposed = 0

        said = ""
        with run.step(f"build `{selected}` from the config, and run it") as report:
            try:
                # Through the factory, never by naming a class. Naming `GraspCalculator` here would
                # let a cell configured for `deep` silently run the analytic stack, which is the
                # one outcome this key must not be able to produce.
                calculator = build_calculator(
                    robot,
                    camera_matrix=frame.intrinsics,
                    min_grip_width_mm=robot.gripper.min_width_mm,
                    max_grip_width_mm=robot.gripper.max_width_mm,
                    max_candidates=8,
                )
                candidates = calculator.compute(
                    frame.segmentations[0], frame.depth_map, unit="mm")
                proposed = len(candidates)
                report(f"{type(calculator).__name__}: {proposed} candidate(s)"
                       + (f", best score {candidates[0].score:.3f}" if candidates else ""))
            except (FileNotFoundError, ValueError) as refusal:
                # The configured generator refusing is a legitimate answer, not a crash: it is what
                # a cell set to `deep` on a machine with no weights meets at boot.
                report(f"refused to build: {type(refusal).__name__}")
                said = str(refusal)
        if said:
            _quote(run, said)
        run.note("")

        said = ""
        with run.step(f"the other answer, `{other}`, on the same cell") as report:
            try:
                alternative = build_calculator(
                    flipped,
                    camera_matrix=frame.intrinsics,
                    min_grip_width_mm=robot.gripper.min_width_mm,
                    max_grip_width_mm=robot.gripper.max_width_mm,
                    max_candidates=8,
                )
                report(f"built {type(alternative).__name__}; this cell has the weights for it")
            except (FileNotFoundError, ValueError) as refusal:
                report(f"{type(refusal).__name__}, which is the documented behaviour")
                said = str(refusal)
        if said:
            _quote(run, said)
        run.note("")
        run.note("A refusal there is the expected answer in this repository, and it is the point:")
        run.note("the key is not inert. A check that enumerates boolean `enabled` fields does not")
        run.note("see a string selector, so a selector is as capable of being dead as a switch is.")
        run.note("")

        if args.bad_artifact:
            said = ""
            with run.step("a readable torch file that is not an artifact") as report:
                import tempfile

                import torch

                with tempfile.TemporaryDirectory() as workspace:
                    # What a training run in progress writes: a fold checkpoint, which carries no
                    # kind stamp at all. It is the wrong file easiest to reach for, and pointing at
                    # it is why the kind is checked at build rather than at first use. Checked
                    # there, a wrong file surfaces as NO_CANDIDATES_GENERATED on every unit, and a
                    # run graded that way reads as a bad generator rather than as a wrong path.
                    checkpoint = Path(workspace) / "fold_checkpoint.pt"
                    torch.save({"epoch": 1, "state_dict": {}}, checkpoint)
                    deep = robot.model_copy(update={"grasping": robot.grasping.model_copy(update={
                        "calculator": "deep",
                        "deep_generator": robot.grasping.deep_generator.model_copy(
                            update={"artifact_path": str(checkpoint)}),
                    })})
                    try:
                        build_calculator(deep, camera_matrix=frame.intrinsics)
                        report("the file was accepted, which it should not have been")
                    except (FileNotFoundError, ValueError) as refusal:
                        report(f"{type(refusal).__name__}, named rather than loaded")
                        said = str(refusal)
            if said:
                _quote(run, said)
            run.note("")
        else:
            run.note("--bad-artifact points the deep branch at a readable torch file that is not a")
            run.note("generator artifact, and prints the second refusal. It imports torch.")
            run.note("")

        run.note("What to pick: `geometric` unless you have trained weights for your own cell and")
        run.note("have measured them against it. The analytic stack is what this repository runs")
        run.note("by default and what every other example here exercises.")
        run.note("")
        run.note("Next: 42_the_thirteen_switches.py, the blocks around the generator.")

        if proposed == 0 and selected == "geometric":
            # The analytic generator proposing nothing on a box it can reach is a real failure of
            # the thing this example is checking, so the status is returned rather than inherited:
            # a finding does not move `Example.exit_code`.
            run.finding("the analytic generator proposed nothing",
                        "on a synthetic box the shipped envelope reaches")
            return EXIT_FAILED
        return run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
