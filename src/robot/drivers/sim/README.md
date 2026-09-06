# Isaac Sim arm driver

The `RobotVendor.SIM` arm driver: a UR e-series arm inside NVIDIA Isaac Sim behind the `RobotArm`
Protocol, plus a no-GPU mock so the whole pick stack runs where Isaac is not installed.

It implements the `robot/core` `RobotArm` Protocol on an Isaac Sim UR, owns the Isaac application
lifecycle, and may call `safety` from `move()`.

Every `isaacsim` import is lazy, so importing this package is always safe. `connect()` on a host
without Isaac raises `IsaacNotAvailableError`.

## Contents

| File | Role |
| --- | --- |
| `arm.py` | `IsaacRobotArm` and `ISAAC_CAPABILITIES`: Lula FK and IK, the flange-to-TCP transform, multi-seed IK resolve, interpolated `move_joint`, the cuRobo and RMPflow approach, and the typed `move()`. |
| `session.py` | `IsaacSimSession`: the Isaac app lifecycle (`start`, `stop`, `step`, `step_n`). It lazily boots the `SimulationApp`, opens the scene and owns the `World`. |
| `robot_models.py` | The model registry: a `robot_model` key maps to the Lula config name, the Isaac USD path, the cuRobo `{model}.yml`, the baked gripper variant if any, and reach, payload and workspace limits. |
| `config.py` | `SimRobotConfig` and `SimCameraConfig`, frozen pure-Python config dataclasses with no Isaac import, safe to build, validate and serialise anywhere. |
| `adapter.py` | The unit and convention conversions: millimetres against metres, XYZW against Isaac WXYZ, rotation matrix to quaternion, and `Pose` and `JointPositions` round-trips, all tagged `Frame.BASE`. |
| `_isaac_protocols.py` | Structural `Protocol` stubs for the lazy Isaac runtime surface, so type checking stays clean without importing `isaacsim`. Annotation-only. |
| `__init__.py` | Re-exports the import-safe public surface. |

The gripper lives next door. `IsaacGripper`, `GripperProfile` and `ROBOTIQ_2F85_PROFILE` are in
[`robot/grippers/sim`](../../grippers/README.md), which is sim-only and deliberately unregistered so
it can never be selected for a real robot. The sim runners hand-build it against the arm's shared
`session`.

## Usage

Build the arm through the registry with `create_arm(RobotVendor.SIM, config=SimRobotConfig)`, or
construct the components directly for a full sim cell with a shared session.

```python
# The off-workstation mock: no Isaac needed.
from src.robot.drivers.sim import IsaacRobotArm, SimRobotConfig
from src.robot.grippers.sim import IsaacGripper

cfg = SimRobotConfig(mock_mode=True, home_joint_positions=(0.0,) * 6,
                     gripper_prim_path="/World/Gripper")
arm = IsaacRobotArm(cfg)
arm.connect()                                            # no SDK touched in mock_mode

grip = IsaacGripper(session=arm.session, gripper_prim_path=cfg.gripper_prim_path,
                    mock_mode=True)
grip.connect()
```

`build_sim_driver_config` in `execution/runtime_pick.py` translates the Pydantic `SimConfig` into a
`SimRobotConfig`. The end-to-end picks and calibrations live under
[`willy_sim`](../../../willy_sim/README.md) as `run_m1_pick`, `run_m2_pick`, `run_eih_pick`,
`run_eth_calibrate` and `run_eih_calibrate`, among others. This package has no `python -m` entry.

## Load-bearing details

**Model-selectable, and the config wins.** `SimRobotConfig.robot_model` (`ur5e` by default, or
`ur3e` or `ur10e`) selects the Lula solver and RMPflow config, the Isaac USD and the cuRobo
`{model}.yml` together, through `robot_models.py`. The end-effector frame `tool0` and the six arm
joint names are shared by every UR e-series, so they stay constants. A disagreeing
`WILLY_CUROBO_ROBOT` environment variable is loudly ignored, because planning one cell against
another robot's geometry produces no visible symptom.

**A combined articulation, where the asset bakes a gripper.** The arm addresses its six joints by
name through an `ArticulationSubset`, and the gripper drives only `finger_joint`. The Isaac `ur5e`
and `ur10e` assets bake a Robotiq 2F-85 variant; the `ur3e` asset ships bare, so a UR3e cell mounts
a standalone gripper.

**`move_joint` is interpolated.** A single large point-to-point command stalls the position drive
partway: it reaches equilibrium short of the target and stays there. `move_joint` walks waypoints of
at most `_MAX_JOINT_STEP_RAD` (0.03 rad), capped at `_MAX_INTERP_STEPS` (250), then settles. This is
what makes a known-pose pick work at all.

**The flange-to-TCP transform is one transform in one pair of fields.**
`SimRobotConfig.tool_offset_mm` and `tool_rotation_quat_xyzw` map the Lula end-effector frame
`tool0`, which is the flange, to the gripper's grasp centre. `fk`, `ik` and `get_tcp_pose` all work
in that TCP frame, so motion targets the grasp point rather than the flange, and the identity
targets `tool0` directly. Both halves come from `robot.gripper.tool_frame`, which the real drivers
read too. For the Robotiq 2F-85 on a UR5e flange the sim profile sets a rotation of minus 90 degrees
about the flange X axis, which maps flange +Y onto TCP +Z, the approach direction, with the grasp
centre offset along flange +Y. Two values describing one transform, one of them invisible to config,
is how a cell silently inherits the wrong tool.

**The motion planner.** `SimRobotConfig.motion_planner` defaults to `curobo`, a global
collision-aware trajectory from a process-isolated cuRobo server, and auto-falls-back to the blind
`ik` path where the cuRobo environment is absent, which is what keeps a machine without it green.
`rmpflow` is also available. Every path ends in the same verify: multi-seed IK resolve, interpolated
`move_joint`, then a check against `_MOVE_POS_TOL_MM` (5 mm) and `_MOVE_ORI_TOL_DEG` (6 degrees),
with up to `_MOVE_MAX_ATTEMPTS` (3) retries.

Note the contrast with the real UR driver, whose `curobo` path is fail-closed and never degrades.
Here the fallback is deliberate, so continuous integration and a laptop stay usable.

**Planned joint moves are opt-in, and a runner turns them on.** With the arm's `plan_joint_moves`
flag set and the planner on `curobo`, `move_joint` takes its path from cuRobo in joint space instead
of interpolating a straight line, so the goal configuration is the one that was asked for rather
than any IK branch reaching the same tool pose. It is off by default and is not a config key,
because it changes the trajectory of every `move_joint` call and each runner that adopts it owes its
own measurement. When it is on and no collision-free path exists it raises `CuroboUnavailableError`
rather than interpolating blindly, because interpolating there would hand back exactly the path the
guard rejects.

## Partial and stubbed

- `is_inside_workspace` always returns `True` for a `Frame.BASE` pose, because this driver owns no
  workspace box. The box belongs to the `WorkspaceGuard` inside the injected `SafetyPreflight`,
  which `move()` enforces on every Cartesian command.
- `stop()` is best-effort and never raises: outside mock mode it re-commands the current joint
  positions so the drive holds station instead of tracking a stale target, and drops the preflight
  continuity memo. It does not interrupt a `move()` already iterating its waypoint loop.
- `move()` honours an optional `SafetyPreflight`, but the `linear`, `vel` and `acc` keyword
  arguments are accepted for Protocol parity and are not applied. `move_joint` accepts velocity and
  acceleration for the same reason and does not apply them either, because the drive is
  position-controlled and walks fixed waypoints at a fixed cadence.
- This package owns no camera class. `SimCameraConfig` is consumed by the `willy_sim` scene and
  perception code and by the calibration routine in `execution`, not here.

`mock_mode` gives identity kinematics: `move`, `move_linear` and `move_to` commit the requested TCP;
`move_joint` updates the cached joints but not the TCP, because there is no mock FK; `fk` and `ik`
raise `IsaacNotAvailableError`; and `capabilities` reports no native FK or IK, so the safety guards
never call them.

## Honest caveats

- `AutonomousGraspService.from_robot_config` cannot build a sim gripper, because that gripper is
  unregistered and needs the shared session, so the sim path uses `from_components`. That is by
  design, not a defect.
- `IsaacSimSession.stop()` calls `SimulationApp.close()`, which can segfault on headless shutdown.
  That is a known upstream issue, so capture any result you need before calling `stop()`.
- Sim-only by definition. No validated real hardware, no async motion, no force control. Non-`ur5e`
  models are selectable, but only `ur5e` is validated on the workstation.

## See also

- [drivers](../README.md), the driver layer and the `RobotVendor` registry
- [robot/grippers](../../grippers/README.md), including the sim `IsaacGripper` counterpart
- [robot/core](../../core/README.md), the Protocols and typed motion contracts this implements
- [willy_sim](../../../willy_sim/README.md), the runners that drive this arm end to end
