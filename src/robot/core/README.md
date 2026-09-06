# robot.core

The vendor-neutral contract layer: the typed surface every arm driver, every gripper driver and
every pipeline above them agrees on.

## What this package guarantees

No vendor SDK is ever imported here, and nothing above `core` is imported here either. It sits just
above `geometry` in the downward stack: it consumes `Pose` and `Frame` from `geometry`, it uses
`numpy`, and it is otherwise a leaf. Drivers, safety and the grasping pipeline depend on these
Protocols; the dependency never runs the other way.

Everything defined here is a Protocol, a frozen value object or a `StrEnum`. There is no behaviour
to configure and no `python -m` entry point.

## Contents

| File | Role |
| --- | --- |
| `robot_arm.py` | The `RobotArm` Protocol, the manipulator surface. |
| `gripper.py` | The `Gripper` Protocol and the opt-in `ObjectDetectingGripper` extension. |
| `arm_capabilities.py` | Opt-in arm capability Protocols `SupportsDigitalIO`, `SupportsForceTorque` and `SupportsRobotStatus`, with the value types `Wrench`, `RobotStatus`, `RobotMode`, `SafetyMode` and `DigitalIOPort`. |
| `motion_result.py` | The typed outcome contract: `MotionStatus`, `MotionCommand`, `MotionResult`, and `NO_PLAN_FAIL_SAFE_MESSAGE`. |
| `joint_positions.py` | `JointPositions`, an immutable validated vector of joint angles in radians. |
| `capabilities.py` | `RobotCapabilities`, the descriptor a driver advertises about itself. |
| `vendor.py`, `gripper_vendor.py` | The `RobotVendor` and `GripperVendor` enums, used as config values and registry keys. |
| `errors.py` | The `RobotError` hierarchy that drivers translate vendor faults into. |

## The contract

`RobotArm` is runtime-checkable, so a candidate driver can be verified with `isinstance`. It carries
`capabilities` and `is_connected`; `connect()`, `disconnect()` and `stop()`; `get_tcp_pose()` and
`get_joint_positions()`; `move_joint` and `move_linear`; `fk` and `ik`; the bool helpers
`is_inside_workspace`, `move_to`, `move_home` and `wait_until_steady`; and the typed
`move(...) -> MotionResult` and `move_to_joints(...) -> MotionResult`.

`Gripper` carries `is_connected`, `min_width_mm`, `max_width_mm`, `connect`, `disconnect`,
`activate`, `set_width_mm` and `get_width_mm`. `ObjectDetectingGripper` adds
`is_object_detected() -> bool` for post-close verification.

The three capability Protocols are opt-in and absent-safe. A driver implements one only if the
hardware offers it, and a caller checks with `isinstance` and falls back when it does not.
`SupportsDigitalIO` reads and writes controller pins, `SupportsForceTorque` returns a `Wrench` in
newtons and newton-metres, and `SupportsRobotStatus` reports a `RobotStatus` through
`get_robot_status()` and clears an active protective stop through `recover_from_protective_stop()`.
A gate written against a capability no attached driver implements
is inert by construction, which is the intended behaviour rather than a silent failure.

`RobotVendor` holds `UR`, `KUKA`, `FRANKA`, `ROS2`, `SIM` and `DUMMY`. `GripperVendor` holds
`ROBOTIQ`, `FRANKA_HAND`, `SCHUNK`, `VACUUM`, `JAW_IO`, `ONROBOT`, `DUMMY` and `NONE`. Both have a
case-insensitive `from_string()` that raises with the full valid set in the message.

`MotionResult` is frozen: `(status, command, target_pose=None, target_joints=None, message="",
exception=None)`, with an `ok` property, truthiness through `__bool__`, and the constructors
`.executed()`, `.failed()` and `.from_bool()`. `JointPositions` validates a finite one-dimensional
radians vector, exposes `.dof`, `.values`, `len`, iteration, indexing, `np.asarray()` support, exact
equality and hashing, `.check_dof()`, `.tolist()` and `.from_list()`. `RobotCapabilities` is a frozen
descriptor `(vendor, model="", dof=6, supports_joint_move=True, supports_linear_move=True,
supports_async_move=False, has_native_fk=False, has_native_ik=False, has_force_control=False,
is_simulated=False)` that validates a lowercase, whitespace-free vendor and a positive DoF.

`RobotError` is the base. `RobotConnectionError`, `RobotKinematicsError`, `RobotMotionRejected` (with
`RobotSingularityRisk` under it) and `RobotEmergencyStop` are its subclasses.
`IsaacNotAvailableError` also subclasses `RuntimeError`, so an existing `except RuntimeError` guard
keeps catching it.

## MotionStatus, and why it is not a string

Fifteen members. Drivers surface them verbatim, so a caller branches on a value instead of scraping
a log line.

| Group | Members |
| --- | --- |
| Success | `EXECUTED`, the only success value |
| Base outcomes | `WORKSPACE_REJECTED`, `IK_FAILED`, `CONTROLLER_REJECTED`, `TIMEOUT`, `CONNECTION_ERROR`, `UNSUPPORTED`, `INVALID_TARGET`, `CANCELLED`, `UNKNOWN` |
| Guard categories, produced by `SafetyPreflight` | `JOINT_LIMIT_REJECTED`, `IK_QUALITY_REJECTED`, `SELF_COLLISION_REJECTED`, `PAYLOAD_REJECTED`, `CONTINUITY_REJECTED` |

`MotionCommand` names what was attempted: `MOVE_TO`, `MOVE_HOME`, `MOVE_JOINTS`, `OTHER`.

`TIMEOUT` covers two events that must not be confused. A planner refusal happens before any command
reaches the controller: nothing moved and the cell is where it was. A genuine execution timeout means
the command was accepted and did not finish in its budget, so the arm may still be moving. The status
alone cannot separate them, so the planner refusal carries `NO_PLAN_FAIL_SAFE_MESSAGE` verbatim, and
both the sim driver and the UR driver emit that exact constant.

## Usage

```python
from src.robot.core import (
    JointPositions,
    MotionCommand,
    MotionResult,
    RobotVendor,
    SupportsForceTorque,
)

q = JointPositions.from_list([0.0, -1.57, 1.57, 0.0, 1.57, 0.0])
assert q.dof == 6

res = MotionResult.executed(MotionCommand.MOVE_TO)
assert res.ok and bool(res) is True

assert RobotVendor.from_string("UR") is RobotVendor.UR

def read_force(arm):
    """Return the TCP wrench in newtons and newton-metres, or None where unsupported."""
    if isinstance(arm, SupportsForceTorque):
        return arm.get_tcp_wrench()
    return None
```

## Traps

`MotionResult.from_bool` assigns `CONTROLLER_REJECTED` when `ok` is `False` and no
`failure_status=` is given. That is the right default for a driver that returns a bare bool, but a
caller that has already proved a workspace or IK rejection must pass the specific status, or the
record will attribute the failure to the controller.

Two motion surfaces coexist on purpose. The bool helpers (`move_to`, `move_home`,
`is_inside_workspace`, `wait_until_steady`) and the typed `move()` both live on `RobotArm`. The typed
surface is what runtime and calibration code should use; the bool helpers stay while callers migrate.
Neither is dead code.

Two capability flags have no method behind them. `has_force_control` is advertised but `RobotArm`
declares no force or admittance call, and `supports_async_move` is advertised while every `move*` is
blocking. There is no asynchronous, streaming or multi-arm contract here.

The frame and error coupling is contract-only. `move_linear`, `ik`, `move_to` and
`is_inside_workspace` are specified to take `Frame.BASE`, and `move_linear` and `ik` to raise
`FrameMismatchError` otherwise, but the enforcement lives in each driver. `core` states the
obligation and checks nothing.

The enums name more than the repository implements. `ur` and `sim` are the exercised backends,
`kuka` has never driven a physical controller, and `franka` and `ros2` are package slots with no
driver. `GripperVendor.FRANKA_HAND` and `GripperVendor.SCHUNK` have no real-hardware driver either.
None of that is decided here; `core` only supplies the names.

## Units and frames

Poses are millimetres with canonical XYZW quaternions and carry their frame. Joints are radians.
Wrenches are newtons and newton-metres. `get_tcp_pose()` and `fk()` return `Frame.BASE`. `Pose` and
`Frame` are defined in `geometry`; `core` only consumes and tags them.

## See also

- [`../safety/`](../safety/README.md) for the pipeline that produces the five guard rejection statuses
- [`../grippers/`](../grippers/README.md) for the `Gripper` and `ObjectDetectingGripper` implementations
- [`../drivers/`](../drivers/README.md) for which driver implements which optional capability
- [`../execution/`](../execution/README.md) for the services that drive `RobotArm` through the typed `move`
