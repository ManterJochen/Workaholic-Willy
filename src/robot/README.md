# The robot: arm, hand, safety and the pick (`src/robot`)

Your code talks to `Robot` and `Cell` and never imports a vendor SDK, so it does not need
to know which arm is attached.

```python
from willy import Pose, Robot, load_tree

robot = Robot.from_tree(load_tree())      # the cell WILLY_PROFILE names; connects nothing
print(robot)                              # arm, hand, lock, safety, planner route, camera world
print(robot.safety())                     # what this arm refuses, asked of the arm that was built
with robot.connected(), robot.without_camera_world("bench run, the table is clear"):
    print(robot.home())
    print(robot.move(Pose.tool_down(450.0, 100.0, 300.0)))
```

Under `WILLY_PROFILE=console_dummy` the same lines run at a desk on a dummy arm
([02_a_robot_at_the_desk.py](../../examples/simulation/02_a_robot_at_the_desk.py)); at a cell,
[03_connect_and_move.py](../../examples/real_robot/03_connect_and_move.py) is the first example that
moves the arm. Poses are millimetres in the robot's base frame, rotations XYZW quaternions, joints
radians, and vendor unit conventions stay inside the drivers.

```bash
python -m src.robot.drivers.doctor --require ur             # which arm and hand drivers this machine can build
python -m src.robot.execution.real_cell --check             # the desk checklist for your cell; touches nothing
python -m src.robot.safety.planning --doctor                # the planner and the exact mesh engine, loaded
python -m src.robot.execution.real_cell --rehearse --runs 3 --profile console_dummy   # the pick path, no hardware
```

Every command and its exit codes: [docs/cli.md](../../docs/cli.md).

## The subpackages

| Subpackage | What it is | You reach it through |
| --- | --- | --- |
| [`execution/`](execution/README.md) | `Robot`, `Cell`, `PickRun`, hand-eye calibration and the real-cell command | `from willy import Robot, Cell, PickRun` |
| [`safety/`](safety/README.md) | `SafetyPreflight`, the ordered fail-closed guards, and the cuRobo planner binding | `from willy import SafetyPreflight` |
| [`grasping/`](grasping/README.md) | generating, scoring and choosing grasps, the motion and its hold check, recovery, and the attempt record | `Scene`, and the pick service |
| [`perception/`](perception/README.md) | the live-camera source and the `Locator` of a real cell | `from willy import Locator` |
| [`drivers/`](drivers/README.md) | the arm registry `create_arm`: `ur`, `kuka`, `sim`, `dummy` | `robot.vendor` in the tree |
| [`grippers/`](grippers/README.md) | the hand registry: `robotiq`, `onrobot`, `vacuum`, `jaw_io`, `dummy`, `none` | `robot.gripper.vendor` in the tree |
| [`core/`](core/README.md) | the `RobotArm` and `Gripper` Protocols, `MotionResult`, the vendor enums, the errors | a driver of your own |

Beside them, `constants.py` holds the log file names and `HOME_JOINTS_DEFAULT`, and `events.py` the
calibration and watchdog events with their listener Protocols. `src.robot` resolves its UR-bound names
lazily, so importing it loads no UR driver.

## The pick, in order

1. Perceive: a camera frame, and the camera's CAMERA to BASE.
2. Generate and score: 6-DoF candidates, ranked.
3. Decide: the AUTO decision gate.
4. Gate: `SafetyPreflight` and IK.
5. Drive: standoff, approach, grasp, close.
6. Verify: the execution policy asks the gripper after its close whether it holds something.
7. Recover: perceive and pick again. A cell built from config never pushes or shakes: the `nudge_target`
   that `dense_clutter` allows is planned without an offset and refused before the arm moves.
8. Log: one `GraspAttemptRecord` per attempt, as JSON lines.

Steps 3 and 7 are opt-in and off by default, so the shipped pick is 1, 2, 4, 5, 6 and 8: open-loop.
Turning a step on is a config change. A switch known to read as on while doing nothing is refused by the
schema (`RobotGraspingConfig.UNWIRED_SWITCHES`, today `occlusion.hard_reject_enabled` alone); every other
flag is held to reaching a runtime carrier, which is not the same as the pick acting on it
([the config reference](../../docs/grasping-config-reference.md), sections 2 and 6.1). The
reinforcement-learning layer is off too: trained offline, shadow-only at run time, and it can never
override a safety rejection. Offline, the logged records feed the KPI
roll-up and the training loop, which are never imported back into the live path.

## Triaging a safety rejection

The guards run in this order before every motion of a real arm, and each refusal reaches
`MotionResult` and `GraspAttemptRecord` as a typed status. If the rejection rate rises in a replay
roll-up, one of them is the cause. Never disable a guard to clear the rate; fix the input that made
it fire.

| Guard | Symptom | What to do |
| --- | --- | --- |
| `workspace` | The target pose is outside the configured workspace box. | Check `robot.workspace_limits` and `robot.safety.limits.workspace_margin_mm`, then the camera calibration. |
| `joint_limit` | The IK solution would drive an axis past its window. | Change the approach angle or prefer another IK seed. Do not widen the limits. |
| `ik_quality` | IK returned a high-residual or near-singular solution. | Rank toward feasible candidates and require a minimum IK quality. |
| `self_collision` | A link, the tool or a fixture comes closer than `min_distance_mm`. | Take another approach through a rescan recovery or a look pose that sees the part from elsewhere, or tune the inflation margin. |
| `payload` | The declared payload is outside the envelope. | Correct the payload estimate. Refuse the pick if the real mass is higher. |
| `motion_continuity` | Successive targets imply a step larger than the cap. | Rank toward joint-continuous candidates and review trajectory blending. |

On call, the procedures are the runbooks under [`docs/runbooks/`](../../docs/runbooks/):
[`real_cell_first_pick.md`](../../docs/runbooks/real_cell_first_pick.md) for a cell taken to its
first pick, and [`cell_bringup.md`](../../docs/runbooks/cell_bringup.md) for a robot the stack has not
run before. Every formula the guards evaluate is in [docs/safety-math.md](../../docs/safety-math.md).

## Status

| Capability | Evidence |
| --- | --- |
| UR-Robot | Measured against UR-Robot |
| Franka | Currently under development |
| KUKA | Currently under development |
| FANUC | Currently under development |
| ABB | Currently under development |
| The Isaac Sim driver with a UR5e and a 2F-85, and the pick path on it | measured in simulation |
| The Robotiq, OnRobot, jaw and suction drivers | Measured against UR-Robot |
| Motion of a physical arm | Measured against UR-Robot |

## Details

- Guide: [robot and safety](../../docs/guide/04-robot-and-safety.md), [the pick loop](../../docs/guide/05-pick-loop.md)
- A hand of your own: [your_own_gripper.md](../../docs/runbooks/your_own_gripper.md)
- Tests: `tests/test_robot.py`, `tests/test_safety_preflight.py`, `tests/test_robot_arm_protocol_conformance.py`, `tests/test_no_web_framework_import.py`
