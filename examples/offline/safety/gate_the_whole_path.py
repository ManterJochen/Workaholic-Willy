"""A planned path judged whole, not only where it ends, and why the check needs a declared world.

The one-shot guards judge where a move ends. `gate_planned_path` samples every leg so that no step moves any point of
the arm further than the collision margin, and runs the same guards over every sample. With nothing declared around the
arm they judge an empty room and find the path clear.
"""

import numpy as np

from willy import SafetyPreflight, create_arm, load_tree

# The base tree with a hand named, in memory: an exact mesh guard reads the hand, and the base tree names none on
# purpose. Put your own hand's registry name here. A second copy also declares a bench under the arm.
cell = load_tree(None).with_values({"robot.gripper.model": "robotiq_2f85"})
bench = cell.with_values({"robot.safety.self_collision.fixtures": [
    {"name": "bench", "center_mm": [500.0, 0.0, -75.0], "half_extents_mm": [300.0, 300.0, 50.0]}]})

# The arm the guards place their link meshes on and read joint limits from. Nothing connects.
arm = create_arm(cell.robot.vendor, config=cell.robot)

# A path in joint radians from the upright arm down into the bench and back out again. Its last configuration stops
# short of the dip, so both ends are clear and an endpoint check finds nothing.
upright = np.array([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])
dip = np.array([3.14, -0.8, 2.0, -2.15, -1.57, 0.0])
path = [upright + (dip - upright) * t for t in (*np.linspace(0.0, 1.0, 21), 0.9)]

# A gate answers None when every sample passes, else the refused motion result, which names the sample and the guard.
# Ask it with `is None`: a refused result is falsy, as every failed motion is.
for name, declared in (("nothing declared", cell), ("bench declared", bench)):
    gate = SafetyPreflight.from_tree(declared)
    for what, waypoints in (("endpoint alone", [path[-1]]), ("whole path", path)):
        refusal = gate.gate_planned_path(waypoints, arm=arm)
        print(f"{name}, {what}: {'clear' if refusal is None else refusal}")
