"""Ten known-pose picks in Isaac Sim, each scored by whether the part really rose, and the rate they make.

Run it from the repository root with Isaac Sim's own interpreter and the root on its path, so it finds
willy (in PowerShell, `$env:PYTHONPATH = (Get-Location).Path`; in cmd, `set PYTHONPATH=%CD%`):
    <isaac-sim>/python.bat examples/simulation/03_isaac_pick_rate.py
"""

import importlib.util

from willy import run_gate

if importlib.util.find_spec("isaacsim") is None:
    print("Isaac Sim is not importable here; run this file with Isaac Sim's own python.bat.")
    raise SystemExit

# Headless picks of one part whose pose the scene knows. "easy" is the direct pick path; "auto" puts
# the decision gate before every grasp, and "closed_loop" refines and verifies each one.
result = run_gate(runs=10, headless=True, mode="easy")

# A pick passes only when the service reported success and the simulator saw the part rise past the
# lift the sim profile sets; the gate passes when enough picks did, a fraction that profile sets too.
print(f"{result.passed} of {result.runs} picks passed, gate passed: {result.gate_passed}")

# A cell configured for a planner that could not start ran the fallback: a rate of another cell.
if result.planner_degraded:
    print(f"the configured planner did not start ({result.planner_degraded}); this rate is not the cell's")

# The rate measures the software on a modelled cell: the depth is the simulator's ground truth, the
# contact is a physics model and the part's pose is known. A real cell's rate is measured at the cell.
