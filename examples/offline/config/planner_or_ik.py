"""A planned path around what the cell declares, or the controller's straight line: one key decides.

Nothing here connects, moves or starts the planner. It reads the key and where it was set, asks this
machine which motion engines it holds for the cell's arm, and reads what the desk checklist says
changes when the cell gives the planner up.
"""

from willy import Cell, MotionStack, load_tree

# The base tree, with no profile: the planner key as every profile inherits it.
tree = load_tree(None)
print(tree.explain("robot.ur.motion_planner"))

# Which arm the engines are asked about, which key named it, and whether each engine is present
# here. This reads paths on disk and spawns nothing.
print(MotionStack.from_robot_config(tree.robot).probe())

# The same cell on the straight line, given in memory. The checklist rows that change are what the
# cell gives up: a line judged only where it ends, and no camera world behind any motion.
straight = tree.with_values({"robot.ur.motion_planner": "ik"})
planned = {check.name: check for check in Cell.from_tree(tree).preflight().checks}
for check in Cell.from_tree(straight).preflight().checks:
    if planned.get(check.name) != check:
        print(f"[{check.status.value}] {check.name}: {check.detail}")
