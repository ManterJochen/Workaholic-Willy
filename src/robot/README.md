# robot

Arm and gripper control, safety gating, and grasp execution: everything that turns a scored 6-DoF
grasp candidate into motion a real cell may perform.

## What this package guarantees

Application code talks to the `RobotArm` and `Gripper` Protocols in [`core/`](core/README.md) and to
the vendor `StrEnum`s. It never imports a vendor SDK, and it never has to know which arm is attached.

This is the lower middle of a strict downward dependency stack:

```
config -> geometry -> perception -> robot -> rl -> replay/KPI
```

Nothing in this package imports a layer above it. Two boundary rules hold across the whole
repository and both are checked mechanically:

1. No module under `src/` imports a web framework (`fastapi`, `uvicorn`, `starlette`).
2. `api/` is the one tree that may, aside from the console's own test modules, which cannot exercise
   an HTTP surface without a client for it. And no module under `src/` imports `api`.

The second rule is what makes the first mean something: without it, a library module could reach a
web framework through the console package and the claim would read as true while being hollow. The
console may depend on the library; the library may never depend on the console. There is no ROS node
in this repository, and no module imports `rclpy`.

## Subsystems

| Subsystem | Role |
| --- | --- |
| [`core/`](core/README.md) | The vendor-neutral contract: `RobotArm` and `Gripper` Protocols, `JointPositions`, `MotionResult` and `MotionStatus`, the vendor enums, the `RobotError` hierarchy. The only part of this package a pipeline imports directly. |
| [`drivers/`](drivers/README.md) | Lazy arm-driver registry (`create_arm`). Registered: `ur`, `kuka`, `sim`, `dummy`. `franka` and `ros2` are named package slots with no driver. |
| [`grippers/`](grippers/README.md) | Lazy gripper-driver registry (`create_gripper`): `robotiq`, `onrobot`, `vacuum`, `jaw_io`, `dummy`, `none`. The Isaac sim jaw and suction drivers sit outside the registry. |
| [`safety/`](safety/README.md) | The fail-closed `SafetyPreflight` guard pipeline in fixed order: workspace, joint limit, IK quality, self-collision, payload, motion continuity. It outranks every learned layer. |
| [`grasping/`](grasping/README.md) | Candidate generation, scoring, the decision gate, refinement, verification, recovery, and the telemetry record. Also the replay and KPI gate and the offline reinforcement-learning seam. |
| [`execution/`](execution/README.md) | The composition root and the operator-facing services: `Cell`, `RuntimePickService`, `AutonomousGraspService`, the `real_cell` entry point, calibration and IK services. |
| [`perception/`](perception/README.md) | The live-camera `PerceptionSource` for real hardware. It lives here because the dependency edge only ever runs from `robot` to `camera`. |

Three loose modules sit beside them. `constants.py` holds the log file names every subsystem writes
to plus `HOME_JOINTS_DEFAULT`, the looking-down home configuration for a 6-axis UR. `events.py`
defines `RobotCalibrationEvent` and `RobotWatchdogEvent` (drift, out-of-distribution, degraded mode,
blocked autonomy, latency budget breach) with their listener Protocols. `__init__.py` re-exports the
vendor-free names eagerly and resolves the UR-bound ones lazily on first attribute access, so
importing this package does not pull in the UR driver chain.

## The pick, in order

1. Perceive. A `PerceptionSource` frame plus a resolver for the camera-to-base transform.
2. Generate and score. `GraspCalculator` produces 6-DoF candidates and ranks them.
3. Decide. The `DecisionEngine` autonomy gate.
4. Refine. A bounded second look at the chosen candidate.
5. Gate. `SafetyPreflight` plus IK.
6. Drive. Standoff, approach, grasp, gripper close.
7. Verify. Width delta, object detection, or vision.
8. Recover. Rescan, next viewpoint, push.
9. Log. One `GraspAttemptRecord` per attempt, as JSON lines.

Steps 3, 4, 7 and 8 are opt-in and off by default. The shipped default pick is 1, 2, 5, 6, 9, and it
is open-loop. Turning a step on is a config change, not a code change, and a switch that would read
as on while doing nothing is refused by the schema rather than accepted quietly.

Offline, the logged records feed the replay and KPI gate and the reinforcement-learning training
loop. Both sit downstream of the telemetry record and are never imported back into the live path.

## Public surface

`from src.robot import ...` gives four tiers.

Eager, vendor-free: `RobotArm`, `Gripper`, `JointPositions`, `RobotCapabilities`; the vendor enums
`RobotVendor` (`ur`, `kuka`, `franka`, `ros2`, `sim`, `dummy`) and `GripperVendor` (`robotiq`,
`franka_hand`, `schunk`, `vacuum`, `jaw_io`, `onrobot`, `dummy`, `none`); the error hierarchy
`RobotError` with `RobotConnectionError`, `RobotKinematicsError`, `RobotMotionRejected`,
`RobotSingularityRisk` and `RobotEmergencyStop`; and the constants above.

Lazy, resolved on first access: the UR facade `Robot` and `URRobotArm`, `URConnection`, `URPose`;
`GripperController`; the safety guards with `SafetyPreflight`, `SafetyContext`, `SafetyDecision` and
`SafetyReason`; `MotionController`, `PoseProvider`, `CalibrationRoutine` and `CalibrationResult`.

Registries: `create_arm(vendor, **kwargs)`, `register_arm_driver` and `available_vendors` from
`drivers/`; `create_gripper(vendor, **kwargs)` and `register_gripper_driver` from `grippers/`.

Facades from `execution/`: `RuntimePickService` returning a `PickSessionReport`, and
`AutonomousGraspService` with a typed `GraspMode` (`easy`, `auto`, `closed_loop`, `dense_clutter`,
`dense_autonomous`), built by `from_robot_config(...)` or `from_components(...)`, whose `pick()`
returns an `AutonomousGraspReport`.

`RobotArm` fixes the vendor contract: `connect`, `disconnect`, `is_connected`, `get_tcp_pose()` and
`get_joint_positions()`, `fk` and `ik`, `move_joint`, `move_linear`, `stop`, and the typed
`move(pose, ...) -> MotionResult`. `Gripper` fixes `activate`, `set_width_mm`, `get_width_mm` and the
width bounds; `ObjectDetectingGripper` adds the opt-in `is_object_detected()` used for post-close
verification.

## Triaging a safety rejection

If the safety rejection rate rises in a replay rollup, one of these guards is the cause. Never
disable a guard to clear the rate; fix the input that made it fire.

| Guard | Symptom | What to do |
| --- | --- | --- |
| `workspace` | The target pose is outside the configured workspace box. | Confirm `robot.workspace_limits` and `robot.safety.limits.workspace_margin_mm`, then verify the camera-to-base calibration. Reject the target. |
| `joint_limit` | The IK solution would drive an axis past its window. | Change the approach angle or prefer another IK seed. Do not widen the limits. |
| `ik_quality` | IK returned a high-residual or near-singular solution. | Re-rank toward feasible candidates and require a minimum IK quality. |
| `self_collision` | The commanded configuration puts a link, the tool or a fixture closer than `min_distance_mm`. | Take another approach through the next-viewpoint recovery, or tune the inflation margin. |
| `payload` | The declared payload is outside the envelope. | Correct the payload estimate. Refuse the pick if the real mass is higher. |
| `motion_continuity` | Successive targets imply a step larger than the cap. | Rank toward joint-continuous candidates and review trajectory blending. |

Every guard reports through `MotionResult` and `GraspAttemptRecord`, so the replay harness rolls the
rejections up without any extra wiring.

## Usage

```python
from src.config.loader import load_config
from src.robot.core import RobotVendor
from src.robot.drivers import create_arm

cfg = load_config()
with create_arm(RobotVendor.from_string(cfg.robot.vendor), config=cfg.robot) as bot:
    pose = bot.get_tcp_pose()        # Pose in Frame.BASE, millimetres and XYZW
    bot.move_linear(target_pose)     # raises on an IK or frame fault
```

This package is a library. The command-line entry points live under `execution/`, `grasping/`,
`safety/planning/`, `drivers/ur/` and `perception/`:

```bash
python -m src.robot.execution.real_cell --rehearse --runs 3   # config, preflight, build, connect, pick
python -m src.robot.safety.planning --check                   # are the external motion engines wired
python -m src.robot.grasping.replay --records run.jsonl       # roll up KPIs from a record log
python -m src.robot.grasping.replay --soak-report             # the soak gate, exit 0 iff it passes
python -m src.robot.grasping.rl build-dataset --dataset-id=v1_bootstrap
```

Units are millimetres, rotations are XYZW quaternions, and every pose and transform carries its
frame. Vendor unit conventions stay inside the driver boundary.

## Traps

The default pick is open-loop. The autonomy gate, closed-loop refinement, verification and recovery,
multi-view fusion, the learned success model and the reinforcement-learning layer are all built and
shipped with their config blocks disabled. Reading the code is not enough to know what a given cell
runs; read its config.

`from_robot_config()` is the real-hardware boot path and `python -m src.robot.execution.real_cell` is
its caller. `--rehearse` drives the whole path on a dummy arm. The path has never been run against a
physical robot controller, and it refuses to build a real cell that has no camera-to-base resolver.

The KUKA driver is registered but has never driven a physical controller. Its payload writes are
config-only, arm-against-arm self-collision is not modelled for it, and its joint limits must be set
explicitly in config or the joint-limit guard reports itself unavailable and fails closed.

`franka/` and `ros2/` are package slots holding a docstring and an empty `__all__`. No driver class
exists and nothing is registered. `GripperVendor.FRANKA_HAND` and `GripperVendor.SCHUNK` are
likewise names with no real-hardware driver behind them.

`MotionResult.from_bool` labels a bare `False` as `CONTROLLER_REJECTED`. That is the most common
cause for a driver that returns a bool with no classification, but a caller that already knows the
real cause must pass `failure_status=` rather than accept the default.

The reinforcement-learning layer is additive and off. It is trained offline, gated by off-policy
evaluation before promotion, and shadow-only at runtime. It can reorder or filter candidates the
safety layer has already cleared and it can never override a rejection. A cell that leaves
`robot.rl.mode` alone imports no reinforcement-learning module at runtime.

## See also

- [`core/`](core/README.md) for the contract every pipeline imports
- [`safety/`](safety/README.md) for the guard pipeline, and `docs/safety-math.md` for its geometry
- [`grasping/`](grasping/README.md) for the generation and telemetry machinery
- [`execution/`](execution/README.md) for the composition root and the pick services
- `docs/guide/04-robot-and-safety.md` for the walkthrough from an empty directory to a gated move
