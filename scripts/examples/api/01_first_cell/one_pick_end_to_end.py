"""One grasp, end to end, on a dummy arm and a synthetic scene.

A rehearsal and a live run differ by one factory. ``Cell.rehearsal`` changes only the vendor on the
operator's own config, so the profile chain, the gripper branch and the grasping block are the ones
this cell actually runs. Every advanced ``robot.grasping.*`` block ships disabled, so the default
attempt is open-loop and the report's own layers line says so rather than a comment claiming it.

That one changed field is why this loads the ``console_dummy`` profile. The base tree asks
for ``gripper.vendor: robotiq``, which lives on the UR controller's tool I/O and cannot be built on
the dummy arm a rehearsal swaps in, so the build substitutes a ``NullGripper``. Since 2026-09-09 a
cell whose end-effector could not be built is refused at the connect (``NoRealGripper``) instead of
coming up and reporting a success it did not have -- which is exactly what this file printed before
that date. ``console_dummy`` names a gripper a dummy arm can carry.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 01_first_cell, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import ConfigError, load_robot_config  # noqa: E402
from src.robot.execution.cell import Cell  # noqa: E402
from src.robot.execution.pick_run import PickRun, Recording  # noqa: E402

# 1. The cell this config describes, with a dummy arm and a synthetic scene.
#    Live, on real hardware:  Cell.from_robot_config(robot, prompt="a red cube")
try:
    robot = load_robot_config(profile="console_dummy")
except ConfigError as no_cell:
    # Two refusals arrive as this one class: a tree with no `robot` block at all, and a tree
    # that has one but carries no `console_dummy` overlay for the desk cell asked for here.
    print(f"no console_dummy cell in this config tree ({no_cell})")
    raise SystemExit
cell = Cell.rehearsal(robot)

# 2. Everything decidable at a desk, before anything is constructed. Preflight comes first for
#    that reason: a blocking finding here is one you fix in YAML, not on the bench.
print(cell.preflight().render())

# 3. Build the vendor arm, the gripper branch, perception and the grasp stack.
#    `robot.grasping.calculator` (geometric or deep) selects the calculator; nothing here names one.
cell.build()
print(f"arm {type(cell.arm).__name__}, gripper {type(cell.gripper).__name__}, "
      f"records {cell.record_log_path or '<not logged>'}")

# 4. What this arm will actually refuse, asked of the arm that was built rather than of the config
#    that asked for it. A dummy carries no SafetyPreflight and states that it does not.
print(cell.safety().render())

# 5. One pick, connect to teardown. Connecting is itself motion: activation sweeps a Robotiq's
#    full travel and a vacuum cup asserts its ejector, which is why it sits outside the loop.
run = PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()
print(run.render())

# 6. The service's own account of the attempt. `layers_that_ran()` is read off the report rather
#    than off a config flag, because a block can be enabled in YAML and still produce nothing.
if run.last is not None:
    print(run.last.render())
    print("layers that produced something:", run.last.layers_that_ran() or "(none)")
