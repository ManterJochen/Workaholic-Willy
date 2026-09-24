# The arm and hand contract (`src/robot/core`)

The typed surface every arm driver, every hand driver and the code above them agree on: the
`RobotArm` and `Gripper` Protocols, the `MotionResult` a motion returns, the vendor enums and the
errors. You reach it through `Robot`, whose reports carry these values; use it directly to write a
driver of your own, or to branch on why a motion was refused.

```python
from src.robot.core import MotionResult, MotionStatus, RobotArm, SupportsForceTorque, Wrench


def why(result: MotionResult) -> str:
    """A caller branches on the status, never on a log line."""
    if result.ok:
        return "executed"
    if result.status is MotionStatus.SELF_COLLISION_REJECTED:
        return "the guard saw a collision: take another approach"
    return f"{result.status.value}: {result.message}"


def tcp_wrench(arm: RobotArm) -> Wrench | None:
    """The TCP wrench in newtons and newton-metres, where the driver offers one."""
    return arm.get_tcp_wrench() if isinstance(arm, SupportsForceTorque) else None
```

`RobotArm` and `Gripper` are runtime-checkable, so `isinstance(my_arm, RobotArm)` checks a driver of
your own. Nothing here imports a vendor SDK or anything above `core`; it reads `Pose` and `Frame`
from `geometry`, and has no config and no command line.

## The nouns

| Noun | What it is |
| --- | --- |
| `RobotArm` | connect, disconnect, stop; `get_tcp_pose()`, `get_joint_positions()`, `fk`, `ik`; `move_joint`, `move_linear` |
| | the typed `move(pose, ...)` and `move_to_joints(...)`, each returning a `MotionResult` |
| | the bool helpers `move_to`, `move_home`, `is_inside_workspace`, `wait_until_steady` |
| `Gripper` | connect, disconnect, `activate`, `set_width_mm`, `get_width_mm`, `min_width_mm`, `max_width_mm` |
| `MotionResult` | frozen: `status`, `command`, target, `message`, `camera_world`; `ok`, and `executed()`, `failed()`, `from_bool()` |
| `MotionStatus`, `MotionCommand` | why a motion ended, and what was attempted (`MOVE_TO`, `MOVE_HOME`, `MOVE_JOINTS`, `OTHER`) |
| `JointPositions` | an immutable, validated vector of joint angles in radians |
| `RobotCapabilities` | what a driver says about itself: vendor, model, DoF, native FK and IK, simulated or not |
| `RobotVendor`, `GripperVendor` | the vendor names used in config and the registries, with `from_string()` |
| `CameraWorldStamp`, `CameraWorldDecline` | whether a world built from a current camera image stood behind a motion |
| `KeepOutBox`, `SegmentationOffer` | a box in BASE that is no obstacle for one motion, and what a perception frame hands the planner |
| `ShutterMotion` | how far the tool moved while a wrist camera's shutter was open |

Opt-in capabilities are Protocols a driver implements only where the hardware offers them, and a
caller checks with `isinstance` and falls back: `SupportsDigitalIO`, `SupportsForceTorque` (a
`Wrench`), `SupportsRobotStatus` (a `RobotStatus`, and recovery from a protective stop), `KeepsLines`
(what `move(pose, linear=True)` keeps of the line: `CHECKED`, `CONTROLLER_LINE`, `TELEPORT` or
`NOT_KEPT`), `CarriesPayload` and `SupportsFreedrive` (a `FreedriveSession` a person moves the arm
in, read as `FreedriveSample`, with every motion verb refused while it is open) on the arm;
`ObjectDetectingGripper`, `StoppableGripper`, `ReportsHoldEvidence` (`HELD`, `EMPTY` or `UNMEASURED`)
and `MeasuresWidth` on the hand. A gate written against a capability no attached driver implements
does nothing, by design.

## What a motion ends as

| Group | `MotionStatus` members |
| --- | --- |
| Success | `EXECUTED`, the only success value |
| Base outcomes | `WORKSPACE_REJECTED`, `IK_FAILED`, `CONTROLLER_REJECTED`, `TIMEOUT`, `CONNECTION_ERROR` |
| | `UNSUPPORTED`, `INVALID_TARGET`, `CANCELLED`, `UNKNOWN` |
| Guard categories, from `SafetyPreflight` | `JOINT_LIMIT_REJECTED`, `IK_QUALITY_REJECTED`, `SELF_COLLISION_REJECTED`, `PAYLOAD_REJECTED`, `CONTINUITY_REJECTED` |

`TIMEOUT` covers two events. A planner refusal happens before any command reaches the controller,
so nothing moved; it carries `NO_PLAN_FAIL_SAFE_MESSAGE` verbatim, and the UR and sim drivers emit
that constant. An execution timeout means the command was accepted and did not finish, so the arm
may still be moving. `MotionResult.from_bool` labels a bare `False` as `CONTROLLER_REJECTED`; a
driver that knows the real cause passes `failure_status=`.

A motion's `camera_world` stamp is one of `UNSTATED` (nothing was said), `PLANNED` (the only one that
vouches), `DECLINED` (a caller's decision), `UNPLANNED` (nothing planned or checked the motion) and
`MISSING` (a planner would need a world and none was declined, so the motion is refused). Every
driver stamps `move` and `move_to_joints`, read off the built arm:

| Arm | Its motions say |
| --- | --- |
| UR on `ik`, KUKA, dummy, Isaac in `mock_mode`, Isaac on `rmpflow` | `UNPLANNED` |
| UR on `curobo`, Isaac on `curobo` | `DECLINED`, else `MISSING` with no live world, else `PLANNED` when its refresh vouched, else `UNSTATED` |

A decline is `camera_world=CameraWorldDecline(reason)` on the verb or a block,
`with arm.without_camera_world(reason):`, and the keyword beats the block. The block is bound to one
arm and does not follow into a thread started inside it.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `UNSUPPORTED`, `NO_CAMERA_WORLD_MESSAGE` | a planned or checked motion with neither a live camera world nor a decline | hand the robot its cameras, or decline with a reason |
| `UNSUPPORTED`, `DECLINE_ON_A_LIVE_WORLD_MESSAGE` | a declined planned motion on an arm whose live world is wired | drop the decline; the world is there |
| `ValueError` from `from_string()` | an unknown vendor name | use a name the message lists as buildable |
| `RobotConnectionError`, `RobotEmergencyStop` | the link or the controller stopped the arm | recover the controller, then connect again |
| `CameraWorldUnavailable`, `PerceptionFrameMoved` | a camera could not vouch for the cell, or the tool moved during a wrist frame | faults of the cell: a campaign stops on them |

Every error here subclasses `RobotError`, and so do `RobotKinematicsError` and `RobotMotionRejected`
(with `RobotSingularityRisk` under it); `IsaacNotAvailableError` also subclasses `RuntimeError`. Asked for a
reserved name (`franka`, `ros2`, `franka_hand`, `schunk`), the refusals list those separately: a
reserved hand comes up as a `NullGripper` that a real cell refuses at the connect, and a reserved arm
fails at `create_arm`.

## Status

This package holds contracts and no behaviour of its own. `ur` and `sim` are the exercised backends;
the drivers' evidence is in [drivers](../drivers/README.md) and [grippers](../grippers/README.md).
`has_force_control` and `supports_async_move` are flags with no method behind them: every `move*`
blocks, and there is no force, streaming or multi-arm contract here. The frame rules (`move_linear`
and `ik` take `Frame.BASE`) are stated here and enforced by each driver.

## Files

| File | Holds |
| --- | --- |
| `robot_arm.py` | `RobotArm` |
| `gripper.py` | `Gripper`, its opt-in extensions, `HoldEvidence`, `hold_evidence_of`, `width_is_measured_of` |
| `arm_capabilities.py` | the arm capability Protocols and their values: `Wrench`, `RobotStatus`, `LineReading`, `PayloadModel` |
| `motion_result.py` | `MotionStatus`, `MotionCommand`, `MotionResult`, `NO_PLAN_FAIL_SAFE_MESSAGE` |
| `camera_world.py` | the camera world stamp and decline, and how a driver declines, stamps and refuses |
| `keep_out.py` | `KeepOutBox`, `SegmentationOffer`, `keeping_out(arm, offer)` |
| `shutter_motion.py` | `ShutterMotion` |
| `joint_positions.py`, `capabilities.py` | `JointPositions`, `RobotCapabilities` |
| `vendor.py`, `gripper_vendor.py` | `RobotVendor`, `GripperVendor` |
| `errors.py` | the `RobotError` hierarchy |

## Details

- Units: poses in millimetres with XYZW quaternions and their frame; joints in radians; wrenches in
  newtons and newton-metres. `get_tcp_pose()` and `fk()` return `Frame.BASE`.
- The guards behind the guard statuses: [safety](../safety/README.md), and their formulas in
  [docs/safety-math.md](../../../docs/safety-math.md)
- Who drives this contract: [execution](../execution/README.md)
- Tests: `tests/test_motion_result.py`, `tests/test_core_value_objects.py`, `tests/test_camera_world_stamp.py`, `tests/test_robot_arm_protocol_conformance.py`
