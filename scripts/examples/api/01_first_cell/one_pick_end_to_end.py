"""One grasp, end to end, on a dummy arm and a synthetic scene.

A rehearsal and a live run differ by one factory. ``Cell.rehearsal`` changes only the vendor on the
operator's own config, so the profile chain, the gripper branch and the grasping block are the ones
this cell actually runs. Every advanced ``robot.grasping.*`` block ships disabled, so the default
attempt is open-loop and the report's own layers line says so rather than a comment claiming it.
"""

import sys
from pathlib import Path

# The repository is not pip installable, so a file run by path needs the root on `sys.path`.
# Four parents up: 01_first_cell, api, examples, scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from src.config import load_robot_config  # noqa: E402
from src.robot.execution.cell import Cell  # noqa: E402
from src.robot.execution.pick_run import PickRun, Recording  # noqa: E402

# 1. The cell this config describes, with a dummy arm and a synthetic scene.
#    Live, on real hardware:  Cell.from_robot_config(robot, prompt="a red cube")
robot = load_robot_config()
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
