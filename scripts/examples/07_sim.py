"""07: the same pick in Isaac Sim, where a wrong answer costs nothing.

    python scripts/examples/07_sim.py                      # check the box, run nothing
    <isaac-sim>\\python.bat scripts/examples/07_sim.py --run m1
    <isaac-sim>\\python.bat scripts/examples/07_sim.py --run m2 --prompt "a red cube" --runs 10

Isaac has its own interpreter, and that is the single most common way an hour disappears here.
Isaac Sim is a separate multi-gigabyte install with a bundled Python, so `import isaacsim` fails
in this repository's `.venv` however the config is set. Run this file with that install's
`python.bat`, which is what `<isaac-sim>` stands for above and in every runner's own docstring.
This example checks that first and refuses with the fix, rather than letting the failure surface
eight imports deep.

Why sim earns a stage of its own: it is the only place a grasp can be wrong for free. The scene
has ground truth, so an attempt can be scored rather than merely observed, and the same pick can
be repeated until one anecdote becomes a rate. On real hardware neither is true.

What sim cannot tell you, itemised because each item is a way a sim result flatters a cell:
contact friction is a model, not a measurement; the depth is a perfect sensor and yours is not; a
gripper that closes cleanly here can slip on a real surface; and a sim camera's near clip can hide
the arm from a mask in ways no real camera does. A sim rate proves the software, never the cell.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the runner is imported by name, so its return type is not otherwise known
    from src.willy_sim.harness.gate import GateResult

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import EXIT_FAILED, Example, not_ready  # noqa: E402

#: The sim runners this example fronts: the module, its subject, and what each one is for.
RUNS: dict[str, tuple[str, str, str]] = {
    "m1": ("src.willy_sim.run_m1_pick", "known-pose pick",
           "no perception at all, so a failure is motion or geometry"),
    "m2": ("src.willy_sim.run_m2_pick", "real-vision pick",
           "detector and segmenter in the loop, the honest end to end"),
    "eih": ("src.willy_sim.run_eih_pick", "eye-in-hand pick",
            "the camera rides the wrist and re-perceives from where it moved"),
}

EPILOG = """
exit codes
  0  the box is ready, and the requested run came out clean
  1  the run executed and came out wrong: a gate the runner scores came out false, or, on a
     runner that scores no gate, not one pick succeeded
  2  Isaac is not importable in this interpreter; run this file with Isaac's own python.bat
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "", epilog=EPILOG,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=sorted(RUNS), default=None,
                        help="which sim pick to run. Omitted: check the box and stop")
    parser.add_argument("--runs", type=int, default=1, help="repeat the pick N times")
    parser.add_argument("--prompt", default="a red cube", help="m2 only: what to look for")
    args = parser.parse_args(argv)

    with Example("07 sim", args.run or "readiness check", hardware=False) as run:
        with run.step("is Isaac importable in THIS interpreter") as report:
            # `find_spec`, not a try/except around the import: importing Isaac takes tens of
            # seconds and starts a renderer. The question here is whether it could be imported.
            available = importlib.util.find_spec("isaacsim") is not None
            report("yes" if available else "no, this is the repository venv rather than Isaac's")
        if not available:
            return not_ready(
                "Isaac Sim is not importable in this interpreter",
                r"run this file with Isaac's own Python: "
                r"<isaac-sim>\python.bat scripts/examples/07_sim.py --run m1")

        run.note("")
        for key, (_module, subject, why) in sorted(RUNS.items()):
            run.note(f"    {key:4} {subject}: {why}")
        run.note("")

        if args.run is None:
            run.note("Pick one with --run. `m1` first: it removes perception from the question,")
            run.note("so if it fails you are looking at motion or geometry, not at a detector.")
            return run.exit_code

        module_name, subject, _why = RUNS[args.run]
        with run.step(f"run {args.run}: {subject}") as report:
            runner = importlib.import_module(module_name)
            # `run_gate` is the runners' library entry point and returns a typed `GateResult`.
            # Their `main` takes no arguments, reads `sys.argv` itself and returns nothing, so it
            # is the wrong door for a caller that already knows what it wants to run.
            result: GateResult = (runner.run_gate(runs=args.runs, prompt=args.prompt)
                                  if args.run == "m2" else runner.run_gate(runs=args.runs))
            verdict = ("no gate verdict" if result.gate_passed is None
                       else f"gate_passed={result.gate_passed}")
            report(f"{result.passed}/{result.runs} picks passed, {verdict}")

        if result.planner_degraded:
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
        run.note("  It ran headless: the runners' own `--gui` is what opens a window, and this")
        run.note("  example does not pass it.")
        return EXIT_FAILED if failed else run.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
