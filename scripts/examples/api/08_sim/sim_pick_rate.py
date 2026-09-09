"""Ten known-pose Isaac picks, scored against the scene's ground truth instead of watched.

A sim scene knows where the object went, so an attempt can be scored rather than merely observed,
and repeated until one anecdote becomes a rate. ``run_gate`` is the runner's library door: its
``main`` reads ``sys.argv`` and returns nothing, while this takes arguments and returns a typed
``GateResult``. A rate from here measures the software and never a cell: the contact friction is a
model, the depth is a perfect sensor, and the decision gate reads ``arm.is_simulated``.
"""

import importlib.util
import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 08_sim, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.willy_sim.run_m1_pick import run_gate  # noqa: E402

# 1. Isaac Sim is a separate multi-gigabyte install with its own bundled Python, so `import
#    isaacsim` fails in this repository's venv however the config is set. Asked with `find_spec`
#    rather than by importing: importing Isaac takes tens of seconds and starts a renderer.
if importlib.util.find_spec("isaacsim") is None:
    print("isaacsim is not importable in this interpreter, so no pick was run.")
    print("Isaac's own bundled Python runs this file unchanged:")
    print(r"    <isaac-sim>\python.bat scripts/examples/api/08_sim/sim_pick_rate.py")
else:
    # 2. Ten known-pose picks on the cell the sim config describes, headless. mode="easy" is the
    #    deterministic path; "auto" swaps in the real DecisionEngine, "closed_loop" refine+verify.
    result = run_gate(runs=10, headless=True, mode="easy")

    # 3. The rate. A run counts as passed only when pick() reported SUCCEEDED and the object rose
    #    past `robot.sim.scene_setup.gate`, which is ground truth the simulator reads off the prim.
    #    `gate_passed` is the runner's own verdict over the sample, and is None where it scores none.
    print(f"{result.passed}/{result.runs} picks passed, gate_passed={result.gate_passed}")

    # 4. The caveat that has to travel with the number. Set only when the cell was configured for a
    #    planner it could not start and ran the fallback instead: a rate of a different motion stack.
    if result.planner_degraded:
        print(f"DEGRADED ({result.planner_degraded}): this does not describe the configured cell")

    # 5. The wire half, for a harness that stores runs instead of reading them. `GateResult` has no
    #    render(); this is the dict the runner's own --result-json persists, under the scene and
    #    mode it stamps on top.
    print(result.to_dict())
