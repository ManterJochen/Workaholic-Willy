# Isaac Sim arm (`src/robot/drivers/sim`)

The `sim` arm: a UR arm inside NVIDIA Isaac Sim behind the `RobotArm` Protocol, and a mock mode that
answers the same calls with no Isaac installed. It drives Isaac only, never a real robot; the picks
and calibrations that run on it live in [willy_sim](../../../willy_sim/README.md).

```python
from willy import Robot, load_tree

# The sim profile's arm in mock mode: no Isaac, no GPU, identity kinematics.
tree = load_tree("sim").with_values({"robot.sim.mock_mode": True})
robot = Robot.from_tree(tree, gripper=None)   # the Isaac gripper comes from the sim runners
with robot.connected():
    print(robot.arm.get_tcp_pose())
```

With Isaac installed, the picks run under Isaac's own interpreter, as
`<isaac-sim>/python.bat examples/simulation/03_isaac_pick_rate.py` (see
[docs/isaac-ready.md](../../../../docs/isaac-ready.md)). In any other interpreter those examples say so
and exit. This package has no command line of its own.

## The nouns

| Noun | Built by | Verb | Returns |
|---|---|---|---|
| `IsaacRobotArm` | `Robot.from_tree(tree)` on a `sim` cell, or `create_arm("sim", config=SimRobotConfig(...))` | `move(pose)`, `move_joint(joints)`, `fk`, `ik` | `MotionResult`; base-frame poses in millimetres |
| `SimRobotConfig` | `build_sim_driver_config(tree.robot.sim, tree.robot.gripper.tool_frame)` | | frozen settings; importing it loads no Isaac |
| `IsaacSimSession` | the arm, at `connect()` | `start()`, `step()`, `step_n(n)`, `stop()` | the Isaac application and its `World` |

`build_sim_driver_config` is in [execution/robot_parts.py](../../execution/robot_parts.py) and is what
`Robot` calls for a `sim` cell. The gripper is not built here: `IsaacGripper` and `IsaacSuctionGripper`
are in [grippers/sim](../../grippers/README.md), outside the gripper registry so a real cell can never
select one, and the sim runners build them against this arm's `session`.

## What it refuses

| Refusal | When | What to do |
|---|---|---|
| `RobotConnectionError`, host not ready | `Robot.from_tree` on a `sim` cell with mock mode off and no Isaac here | Run under Isaac's `python.bat`, or set `robot.sim.mock_mode: true` |
| `IsaacNotAvailableError` | `connect()` outside mock mode with no Isaac; `fk` or `ik` in mock mode | The same |
| `ValueError` | a `robot_model` outside `ur3`, `ur3e`, `ur5`, `ur5e`, `ur10`, `ur10e` | Use one of the six |
| `CONTROLLER_REJECTED` | the arm plans with `curobo` and the cuRobo environment is missing or will not start | [ext_deps/README.md](../../../../ext_deps/README.md); `python -m src.robot.safety.planning --doctor` |
| `CuroboUnavailableError` at planner start | no hand named in `robot.gripper.model`; a descriptor for another arm; a setup no evidence file measured | [ur_family_bringup.md](../../../../docs/runbooks/ur_family_bringup.md) |
| `CuroboUnavailableError` in a move | planned joint moves are on and no collision-free path exists | Move the goal; nothing interpolates blindly past it |
| `NoRealGripper` at connect | a `sim` cell built through `Robot` with its gripper: the Robotiq needs a UR controller | Build with `gripper=None`, or run the sim runners |

## How it behaves

- **The model comes from config.** `robot_model` (`ur5e` by default) selects the Lula solver, the
  Isaac USD and the cuRobo `willy_{model}.yml` together. A disagreeing `WILLY_CUROBO_ROBOT` environment
  variable is ignored with a warning, because planning one arm against another's geometry shows no
  symptom. The cell's hand is added to the planner as a body when it starts.
- **Baked or mounted hand.** The `ur5e` and `ur10e` assets carry a Robotiq 2F-85 variant, selected
  when the cell's hand is the 2F-85. Every other hand, and every hand on the other four arms, is mounted
  standalone from `willy_sim.grippers`. A hand with a joint profile and no measured mount, such as the
  Hand-E, is refused.
- **One tool transform.** `tool_offset_mm` and `tool_rotation_quat_xyzw` map the flange frame `tool0`
  to the grasp centre, from `robot.gripper.tool_frame`, the same key the real drivers read. `fk`, `ik`
  and `get_tcp_pose` all work in that TCP frame.
- **Joint moves are interpolated.** One large position command stalls short of its target, so
  `move_joint` walks steps of at most 0.03 rad, capped at 250 steps, then settles.
- **The planner does not fall back.** `motion_planner` defaults to `curobo`, a collision-aware plan
  from a separate cuRobo process (cuRobo and Isaac cannot share one). Where that process cannot start,
  every motion is refused as `CONTROLLER_REJECTED`, because a run on blind IK would report the pick rate
  of another motion stack. A runner may choose `ik` or `rmpflow` for a cell; the config has no key for
  it. Every path ends in the same check: within 5 mm and 6 degrees of the target, with up to 3 attempts.
- **Planned joint moves are a runner's choice.** With the private `_plan_joint_moves` flag set and the
  planner on `curobo`, `move_joint` takes its path from cuRobo in joint space, so the goal
  configuration is the one asked for. It is off by default and is not a config key.

## What it does not do

- `is_inside_workspace` accepts every base-frame pose. The workspace box belongs to the
  `WorkspaceGuard` in the injected `SafetyPreflight`, which `move()` runs on every Cartesian command.
- `stop()` holds the current joints and never raises; it does not interrupt a `move()` already walking
  its waypoints.
- `linear=True` is a checked line on `curobo` with a preflight wired, and dropped on `ik` and `rmpflow`;
  `line_motion()` says which before anything moves. `vel` and `acc` are accepted and not applied.
- In mock mode `move`, `move_linear` and `move_to` commit the requested pose, `move_joint` updates the
  joints but not the pose, `fk` and `ik` raise, and the capabilities report no native FK or IK, so no
  guard calls them.
- `IsaacSimSession.stop()` can crash Isaac at a headless shutdown. Keep what you need before you call it.

## Status

| Capability | Evidence |
|---|---|
| Known-pose, real-vision and wrist-camera picks on a UR5e with a 2F-85; hand-eye calibration | measured in simulation: [willy_sim](../../../willy_sim/README.md) |

The other five UR models are selectable; the measured picks are on the UR5e. Mock mode answers the same
calls with no Isaac installed, which is how CI runs the stack above it.

## Files

| File | Holds |
|---|---|
| `arm.py` | `IsaacRobotArm` and `ISAAC_CAPABILITIES`: Lula FK and IK, the TCP transform, the planners, the typed `move()` |
| `session.py` | `IsaacSimSession`: boots the `SimulationApp`, opens the scene, owns the `World` |
| `robot_models.py` | one `URModelSpec` per UR: Lula key, USD path, cuRobo descriptor, baked hand, reach and workspace |
| `config.py` | `SimRobotConfig` and `SimCameraConfig`, with no Isaac import |
| `adapter.py` | millimetres against metres, XYZW against Isaac's WXYZ, pose and joint round trips |
| `_isaac_protocols.py` | typing stubs for the Isaac objects, so type checks need no Isaac |

## Details

- [drivers](../README.md): the registry, the doctor and the other vendors
- [grippers](../../grippers/README.md): `IsaacGripper`, the jaw profiles and the suction cups
- [willy_sim](../../../willy_sim/README.md): the runners, the scenes and the measured pick rates
- [UR driver](../ur/README.md): the real arm these picks stand in for
