"""70: the same pick where a wrong answer is free. The machine is the requirement, not the cell.

Nothing here reaches a robot, a camera or a gripper. What it needs instead is a workstation:
an Isaac Sim install with its own bundled interpreter, and two motion engines that live outside
this repository and are located through environment variables.

    python scripts/examples/sim/70_sim_pick.py               # check the box, run nothing
    <isaac-sim>\\python.bat scripts/examples/sim/70_sim_pick.py --run m1
    <isaac-sim>\\python.bat scripts/examples/sim/70_sim_pick.py --run m2 --runs 10

The decision is where you spend an experiment. A cell answers about itself and about nothing
else, and every answer it gives costs a fixture reset and a person standing next to it. A sim
scene has ground truth, so an attempt can be scored rather than merely observed, and it can be
repeated until one anecdote becomes a rate. Sim is where a wrong answer is free, so it is where
you send the questions you expect to get wrong: a new sampler, a new gripper, a scene you have
not tried.

What the free answer does not buy, itemised because each item is a way a sim number flatters a
cell. Contact friction is a model rather than a measurement. The depth is a perfect sensor and
yours is not, and two of the three runners below take their depth from the simulator's own
ground truth by default, which is a sensor no supplier sells. A gripper that closes cleanly on a
rendered surface can slip on a real one. And the fail-closed decision gate reads
`arm.is_simulated`, so on this arm it runs permissive: the control flow is exercised, the
hardware refusal is not. A rate from here is evidence about the software and never about a cell.

Two preconditions, in the order this file checks them, which is the order `bootstrap_sim_cell`
checks them in as well:

  1. The engines. The sim cell is configured to route every motion through the cuRobo planner and
     to check it against exact meshes through Coal or fcl. Neither can be a pip dependency, so
     both are located by environment variable: `WILLY_CUROBO_PYTHON` at the cuRobo environment's
     python.exe, `WILLY_COAL_PREFIX` at the Coal environment prefix. Missing either one does not
     make the cell slower, it changes which motions are proposed and which are accepted, so the
     library refuses rather than quietly producing a number that describes a different system.
     This is a filesystem probe costing milliseconds, and its answer is the same in both
     interpreters, so it is asked here before the reader is told to go and change interpreter.
  2. The interpreter. Isaac Sim is a separate multi-gigabyte install with a bundled Python, so
     `import isaacsim` fails in this repository's virtual environment however the config is set.
     That is the single most common way an hour disappears here.

`WILLY_ALLOW_DEGRADED_MOTION` turns the first refusal into a warning. This example does not set
it for you and offers no flag that would: whoever sets it owns the number that comes out, and an
example that quietly disables the check on the reader's behalf is exactly the shape this
directory exists to remove.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # the runner is imported by name, so its return type is not otherwise known
    from src.willy_sim.harness.gate import GateResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402


@dataclass(frozen=True, slots=True)
class Runner:
    """One sim gate this example fronts, and the two kwargs its `run_gate` does not share.

    `prompt` and `rendered_depth` are not styling: a runner that does not take the keyword must
    be told so rather than handed it, because `run_gate(**kwargs)` with an unknown keyword is a
    `TypeError` eight frames deep, and silently dropping the keyword would answer a question the
    caller did not ask.
    """

    module: str
    subject: str
    why: str
    #: `run_gate` takes `prompt`. Only the runner with a text-conditioned front end does.
    prompt: bool
    #: `run_gate` takes `ground_truth_depth`, so the sim's exact depth can be swapped for the
    #: camera's own rendered depth.
    rendered_depth: bool


#: The four runners in `src/willy_sim` that expose `run_gate` are m1, m2, eih and the dense
#: clutter one. Three are fronted here; `src.willy_sim.run_dense_pick` is the fourth and takes
#: some forty further keywords for the bin, the clutter and the recovery arcs, which is a
#: different example than this one.
RUNNERS: dict[str, Runner] = {
    "m1": Runner("src.willy_sim.run_m1_pick", "known-pose pick",
                 "no perception at all, so a failure is motion or geometry",
                 prompt=False, rendered_depth=True),
    "m2": Runner("src.willy_sim.run_m2_pick", "real-vision pick",
                 "detector and segmenter in the loop, the honest end to end",
                 prompt=True, rendered_depth=False),
    "eih": Runner("src.willy_sim.run_eih_pick", "eye-in-hand pick",
                  "the camera rides the wrist and re-perceives from where it moved",
                  prompt=False, rendered_depth=True),
}

EPILOG = """
exit codes
  0  the box is ready, and the requested run came out clean
  1  the run executed and came out wrong: a gate the runner scores came out false, or, on a
     runner that scores no gate, not one pick succeeded
  2  the box is not ready: a motion engine this cell routes through is not installed, Isaac is
     not importable in this interpreter, or the requested combination does not exist

Isaac's python.bat does not pass these through. It reports its own status, and every non-zero
code arrives at the shell as 1, so a script that branches on 2 has to read this file's output
rather than its exit code when it is launched that way.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=sorted(RUNNERS), default=None,
                        help="which sim gate to run. Omitted: check the box and stop")
    parser.add_argument("--runs", type=int, default=1, help="repeat the pick N times")
    parser.add_argument("--prompt", default="a red cube", help="m2 only: what to look for")
    parser.add_argument("--mode", default=None,
                        help="the grasp mode the runner wires. Default: the runner's own, which "
                             "is the deterministic one")
    parser.add_argument("--rendered-depth", action="store_true",
                        help="m1 and eih only: let the camera's rendered depth reach the pick "
                             "instead of the simulator's exact depth")
    args = parser.parse_args(argv)

    # No `hardware=True` banner. `Example.hardware` asks whether this file can touch a robot, and
    # the sim config builds the Isaac driver and only that, so announcing a rehearsal would claim
    # something was held back that never existed. The rule the banner protects is not weakened
    # here: no path in this file reaches a vendor SDK.
    with Example("70 sim_pick", args.run or "readiness check", hardware=False) as run:
        from src.robot.drivers.sim.robot_models import curobo_robot_yml
        from src.robot.safety.planning.environment import probe_planning_environment
        from src.willy_sim.config import load_sim_config, require_robot, sim_driver_config
        from src.willy_sim.harness.bootstrap import (
            DegradedMotionStackError,
            require_motion_stack,
        )
        from src.willy_sim.harness.modes import DEMO_MODES

        # The request is checked before the box is, because a contradictory request is answerable
        # with nothing installed, and reporting "Isaac is missing" to somebody who also asked for
        # a combination that does not exist sends them to fix the wrong thing first.
        if args.mode is not None and args.mode not in DEMO_MODES:
            return not_ready(f"mode {args.mode!r} is not one of {', '.join(DEMO_MODES)}",
                             "drop --mode to take the runner's own default")
        chosen = RUNNERS[args.run] if args.run else None
        if args.rendered_depth and chosen is not None and not chosen.rendered_depth:
            # Refused by name rather than dropped. Passing it anyway is a TypeError inside the
            # runner, and dropping it would report a rendered-depth rate that was measured on the
            # simulator's exact depth, which is the one number this example must never produce.
            return not_ready(
                f"`{args.run}` has no ground_truth_depth switch, so --rendered-depth cannot apply",
                "run it on m1 or eih, whose run_gate takes the keyword; or drop the flag")

        with run.step("what the sim cell routes every motion through") as report:
            # The cell's own config, not this file's opinion of it. `bootstrap_sim_cell` reads the
            # same three values, so a config edit moves the check and the run together.
            config = load_sim_config(None)
            robot = require_robot(config)
            planner = sim_driver_config(config, headless=True).motion_planner
            self_collision = robot.safety.self_collision
            exact_mesh = (self_collision.backend or "").lower() == "fcl"
            model = self_collision.kinematics_model or robot.sim.robot_model
            report(f"robot={robot.sim.robot_model}, motion_planner={planner}, "
                   f"self_collision.backend={self_collision.backend}")

        with run.step("are those engines installed on this box") as report:
            environment = probe_planning_environment(
                robot_config=curobo_robot_yml(robot.sim.robot_model), kinematics_model=model)
            report("fully anchored" if environment.fully_anchored
                   else "partially anchored, so at least one engine is missing")
        run.note("")
        for line in environment.render().splitlines():
            run.note(line)
        run.note("")

        # The precondition itself, run rather than described. This is the function
        # `bootstrap_sim_cell` calls before it boots Isaac, so the block it prints below is the
        # one a reader meets, and not a second copy of it that can drift.
        try:
            require_motion_stack(planner, robot_config=curobo_robot_yml(robot.sim.robot_model),
                                 kinematics_model=model, exact_mesh_collision=exact_mesh)
        except DegradedMotionStackError as refusal:
            return not_ready(
                str(refusal),
                "install both engines into ext_deps/, which is where the code looks by default, "
                "with scripts/ext_deps/install.ps1; or point WILLY_CUROBO_PYTHON and "
                "WILLY_COAL_PREFIX at an environment pair you already built. "
                "`python -m src.robot.safety.planning --check` reads the paths and `--doctor` "
                "loads the engines and names whatever refused.")

        with run.step("is Isaac importable in THIS interpreter") as report:
            # `find_spec`, not a try/except around the import: importing Isaac takes tens of
            # seconds and starts a renderer. The question here is whether it could be imported.
            available = importlib.util.find_spec("isaacsim") is not None
            report("yes" if available else "no, this is the repository venv rather than Isaac's")
        if not available:
            return not_ready(
                "Isaac Sim is not importable in this interpreter",
                r"run this file with Isaac's own Python: "
                r"<isaac-sim>\python.bat scripts/examples/sim/70_sim_pick.py --run m1")

        run.note("")
        for key, runner in sorted(RUNNERS.items()):
            run.note(f"    {key:4} {runner.subject}: {runner.why}")
        run.note("")

        if args.run is None or chosen is None:
            run.note("Pick one with --run. `m1` first: it removes perception from the question,")
            run.note("so if it fails you are looking at motion or geometry, not at a detector.")
            return run.exit_code

        # Built as a dict because the three signatures differ, and every key here was checked
        # against the runner's own `run_gate` rather than assumed from a sibling's.
        kwargs: dict[str, Any] = {"runs": args.runs}
        if chosen.prompt:
            kwargs["prompt"] = args.prompt
        if args.mode is not None:
            kwargs["mode"] = args.mode
        if args.rendered_depth:
            kwargs["ground_truth_depth"] = False

        with run.step(f"run {args.run}: {chosen.subject}") as report:
            module = importlib.import_module(chosen.module)
            # `run_gate` is the runners' library entry point and returns a typed `GateResult`.
            # Their `main` takes no arguments, reads `sys.argv` itself and returns nothing, so it
            # is the wrong door for a caller that already knows what it wants to run.
            result: GateResult = module.run_gate(**kwargs)
            verdict = ("no gate verdict" if result.gate_passed is None
                       else f"gate_passed={result.gate_passed}")
            report(f"{result.passed}/{result.runs} picks passed, {verdict}")

        if result.planner_degraded:
            # Reachable only where the operator set WILLY_ALLOW_DEGRADED_MOTION, since the
            # refusal above is what happens otherwise. The rate is then a rate of a different
            # motion stack, and the number alone cannot say so.
            run.finding(f"{args.run} planner", "the configured motion planner did not start "
                                               f"({result.planner_degraded}), so this rate "
                                               "describes the fallback path, not this cell")
        # Only the runner can say whether a rate is a pass, because only it knows the gate its
        # scene was built around. The vision runner scores no gate, which leaves exactly one
        # unambiguous failure to read off: nothing was picked at all.
        failed = result.gate_passed is False or (result.gate_passed is None and result.passed == 0)
        if failed:
            run.finding(args.run, "the run executed and did not pass; the runner's own summary "
                                  "above says why")

        run.note("")
        run.note("WARN a rate here proves the software, never your cell.")
        if args.runs == 1:
            run.note("  And one run is not a rate. `--runs 10` is the smallest honest sample.")
        if not args.rendered_depth and chosen.rendered_depth:
            run.note("  The depth was the simulator's exact depth. `--rendered-depth` swaps in")
            run.note("  the camera's own, which is the nearer thing to what a cell will see.")
        run.note("  It ran headless: the runners' own `--gui` is what opens a window, and this")
        run.note("  example does not pass it. To watch one instead of scoring it, see")
        run.note("  71_record_a_demo.py, which films a run and scores nothing.")
        return EXIT_FAILED if failed else run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
