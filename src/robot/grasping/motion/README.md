# Grasp motion (`src.robot.grasping.motion`)

How the arm and the gripper realise a chosen grasp: the approach, close and retreat choreography,
the transform that places the camera, and the swept-volume check that can refuse the path. Which
grasp to attempt is decided one layer up, in [`loop/`](../loop/README.md).

## What it guarantees

Vendor-neutral. It imports no driver and talks only to the `RobotArm` and `Gripper` Protocols in
[`robot/core`](../../core/README.md). TCP positions are `Pose` in millimetres with XYZW rotations and
a tagged frame; gripper widths and motion offsets are millimetres; forces are newtons.

It fails closed on an unresolved frame. A pose that is valid as a number and wrong as a target
commands no motion.

The package `__init__` re-exports nothing; import from the modules, or from the `src.robot.grasping`
package root, which re-exports the public names.

## The public surface

| Module | Owns |
| --- | --- |
| `execution_policy.py` | `GraspExecutionPolicy`, `PolicyOutcome`, `PolicyReport`: the choreography and the close verification |
| `frame_resolver.py` | The `FrameResolver` Protocol and its three implementations, plus `FrameResolutionFailure` and the typed refusal reasons |
| `trajectory_safety.py` | `ApproachPathPolicy` and the sweep functions: swept-volume validation of the moving gripper along the path |

### The choreography, and its five endings

Pre-grasp standoff along the negated approach, linear descent, `gripper.set_width_mm`, then the close
verification, then the lift.

| `PolicyOutcome` | Means | Moved |
| --- | --- | --- |
| `EXECUTED` | approach, close and retreat completed; the object is held or trusted to be | yes |
| `OBJECT_NOT_DETECTED` | the close succeeded mechanically and the gripper reports an empty jaw | yes |
| `MOTION_FAILED` | `RobotArm.move_to` raised; the exception is carried in the report | partly |
| `CAMERA_FRAME_REJECTED` | a camera-frame grasp with `require_base_frame_grasp` on and no resolver wired | no |
| `APPROACH_PATH_BLOCKED` | every ranked candidate's approach and retreat sweep hit the scene cloud | no |

The close verification is capability-aware. The policy calls
`ObjectDetectingGripper.is_object_detected()` if and only if the configured gripper advertises that
capability; a gripper without it is trusted after the close and the outcome is `EXECUTED`. That is an
honest default rather than a lax one, because inventing a signal a gripper cannot produce would
report a held object on every empty close.

Advertising the capability is not the same as having a sensor, and that is the second half of the
same honesty. Four shipped grippers implement it. The OnRobot driver reads the grip-detected bit out
of the status word, which is a real measurement. The digital-I/O jaw prefers a part-present input,
falls back to inferring from a reed pair, and with neither reports the commanded state. The vacuum
driver reads the vacuum switch when `vacuum_ok_input_pin` names one, and otherwise reports the
commanded state as well. A cell that wires no feedback pin still gets a `True` after every close, so
check the pin before trusting `OBJECT_NOT_DETECTED` to mean anything.

### The frame resolver

A perception frame carries intrinsics but not the camera-to-base transform. Without a resolver a
candidate stays in the camera frame, and a camera-frame pose driven at the arm is a valid number
pointing at the wrong place. The `FrameResolver` Protocol closes that: the orchestrator asks a
resolver for the current CAMERA to BASE transform at the moment a frame is captured, forwards it into
the calculator, and the policy refuses with `CAMERA_FRAME_REJECTED` when no resolver is wired and a
candidate is still camera-frame.

| Implementation | For | Behaviour |
| --- | --- | --- |
| `StaticCameraToBaseResolver` | eye-to-hand | one transform, loaded from the calibration artifact |
| `EyeInHandFrameResolver` | eye-in-hand | recomposed from the live TCP every frame |
| `IdentityFrameResolver` | perception already in BASE | explicit, so no transform needed is a stated choice rather than an omission |

`require_base_frame_grasp` defaults to `False` on the policy. The composition root switches it on
exactly when a resolver is wired, so a cell either transforms its grasps or refuses to move them.

### The swept-path validator

`ApproachPathPolicy` generates a deterministic series of intermediate poses, pre-grasp to grasp and
grasp to retreat, reuses the shared `GripperGeometryStrategy` and `colliding_point_indices` at each
sample so the moving gripper volume is tested against the scene points, and returns a frozen
`ApproachPathReport`.

| `ApproachPathOutcome` | Means |
| --- | --- |
| `CLEAR` | every sampled pose fits |
| `BLOCKED` | at least one does not, and the report names which step |
| `NO_OBSTACLES` | there was no scene cloud to check against |
| `SKIPPED` | the check was not configured |

`NO_OBSTACLES` and `SKIPPED` are separate values on purpose. Nothing was in the way and nobody looked
are different answers, and collapsing them would make an unconfigured check read as a passing one.

This is a validator, not a planner. It never commands motion; it returns a report and the caller
decides whether to refuse the pick or fall back to another candidate.

## Traps

The swept validator is off by default and mode-scoped. `robot.grasping.approach_validation` ships
`enabled: false`, and its `apply_modes` lists only the dense modes, so switching it on in a cell
running another mode leaves it inert. Defaults are `standoff_mm: 80.0`, `retreat_mm: 100.0`,
6 approach samples and 4 retreat samples; zero samples on a leg disables that leg and reports
`SKIPPED`.

Retreat is along the policy's `retreat_direction`, world `+Z` by default. That is a true vertical
lift only for a base-frame grasp.

A single scalar clearance is not this check. `approach_clearance_mm` lives in
[`scoring/occlusion.py`](../scoring/README.md) and inspects one waypoint, ignoring the standoff, the
interpolation and the retreat. It is telemetry, not a gate.

The arm's own path is not checked here. Whole-robot self-collision and world-collision belong to the
planner and to the [safety layer](../../safety/README.md); this package only ever checks the
end-effector.

## See also

- [`../README.md`](../README.md) for the tier this package executes for
- [`../loop/README.md`](../loop/README.md) for the orchestrator that decides which grasp reaches this policy
- [`../planning/README.md`](../planning/README.md) for where the pre-grasp and retreat poses come from
- [`../collision/README.md`](../collision/README.md) for the gripper envelope reused at each sample
- [`../../safety/README.md`](../../safety/README.md) for the guards every commanded motion still passes through
- [`../../core/README.md`](../../core/README.md) for the `RobotArm`, `Gripper` and `ObjectDetectingGripper` Protocols
