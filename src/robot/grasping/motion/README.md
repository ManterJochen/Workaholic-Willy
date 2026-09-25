# Grasp motion (`src.robot.grasping.motion`)

How the arm and the hand carry out a chosen grasp: the approach, the close and the lift, the transform
that places the camera, and the swept-volume check that can refuse the path. Which grasp to attempt is
decided one layer up, in [`loop/`](../loop/README.md).

The part you tune is `GraspMotion`, which `willy` exports. The pick service builds the one
`GraspExecutionPolicy` it drives from it, on its own arm and hand, with every guard in place:

```python
from willy import Cell, GraspMotion, load_tree

# Start 60 mm above the grasp, close 5 mm below the measured width, lift 100 mm after the close.
# A field left out keeps the service's own; a negative standoff or a close speed above 1 raises here.
motion = GraspMotion(standoff_mm=60.0, close_squeeze_mm=5.0, retreat_mm=100.0)
cell = Cell.from_tree(load_tree(), motion=motion)      # the cell WILLY_PROFILE names
print(motion.to_dict())
```

[`examples/real_robot/12_pick_with_the_camera.py`](../../../../examples/real_robot/12_pick_with_the_camera.py)
runs a campaign of picks with it. Everything else here is internal: the pick service and the pick loop call it.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspMotion` | `GraspMotion(...)`, every field optional | `to_dict()` | the fields you chose |
| `GraspExecutionPolicy` | `build_execution_policy(motion, arm=..., ...)` in the pick service | `execute(grasp)` | `PolicyReport` |
| `FrameResolver` | one of the three resolvers below | `camera_to_base_for_frame(frame, arm=...)` | a CAMERA to BASE `Transform` |
| `ApproachPathPolicy` | `ApproachPathPolicy(standoff_mm=80.0, ...)` | `validate_approach_and_retreat(grasp, policy=...)` | two `ApproachPathReport`s |

Nothing is re-exported from the package `__init__`; import from the modules, or from `src.robot.grasping`.
Positions are `Pose` in millimetres with XYZW rotations and a tagged frame, widths are millimetres and
forces newtons. The package talks only to the `RobotArm` and `Gripper` Protocols in
[`robot/core`](../../core/README.md) and imports no driver.

## The choreography, and how it ends

The policy first reads what the arm keeps of a straight line (`KeepsLines`). An arm that keeps one gets a
planned move to the standoff, one line down to the grasp, and line lifts. An arm that keeps none is
refused before the jaws open. An arm that does not say drives the interpolated waypoints.
`PolicyReport.line_motion` carries the reading. The jaws open before the approach, close at the grasp,
and the close is verified before the lift. A hand that toggles with no sensor (`core.gripper.TogglesWithoutSensor`)
is never pulsed before the approach: it is asked whether its jaws stand open, and asks a person where it
believes them closed; at the grasp it gets one close, and nothing verifies it.

| `PolicyOutcome` | Means | Moved |
| --- | --- | --- |
| `EXECUTED` | approach, close and lift completed; the part is held or trusted to be | yes |
| `OBJECT_NOT_DETECTED` | the close finished and the hand reports an empty jaw | yes |
| `MOTION_FAILED` | a move raised or was refused; the report carries the status and the message | partly |
| `CAMERA_FRAME_REJECTED` | a camera-frame grasp reached a policy that requires BASE | no |
| `APPROACH_PATH_BLOCKED` | every ranked candidate's approach or lift sweep hit the scene cloud | no |
| `GRIPPER_FAULT` | the gripper raised (the report carries it, never a raise out of `execute`), or a toggle hand would not start | no, or up to the close |

## Did the hand really hold it

The policy asks `is_object_detected()` only of a hand that implements `ObjectDetectingGripper`. A hand
without it is trusted after the close and the outcome is `EXECUTED`: inventing a signal the hand cannot
produce would report a held part on every empty close.

Implementing it is not the same as having a sensor. The Robotiq driver reads the object status (gOBJ)
and the OnRobot driver the grip-detected bit, both measurements. The digital-I/O jaw reads a
part-present input where one is wired, infers from a reed pair otherwise, and with neither reports the
command. The vacuum driver reads the switch named by `vacuum_ok_input_pin`, and without one reports the
command too. The simulated suction cup reports whether Isaac bonded a body. So a cell with no feedback
pin sees a hold after every close. `hold_evidence()` (`ReportsHoldEvidence`) says which answers are
measurements, and the robot's hand verbs read that instead.

## The frame resolver

A perception frame carries the lens but not where the camera is. Without a resolver a candidate stays in
the camera frame, and a camera-frame pose driven at the arm is a valid number pointing at the wrong place.

| Resolver | For | Behaviour |
| --- | --- | --- |
| `StaticCameraToBaseResolver` | a fixed camera | one CAMERA to BASE transform, from the rig's calibration |
| `EyeInHandFrameResolver` | a wrist camera | composed from the tool pose every frame |
| `IdentityFrameResolver` | perception already in BASE | a stated choice rather than an omission |

The pick service sets `require_base_frame_grasp` exactly when a resolver is wired, so a cell either
transforms its grasps or refuses to move them.

## The swept-path validator

`ApproachPathPolicy` samples poses from the standoff to the grasp and from the grasp up the lift, and at
each sample tests the gripper envelope from [`collision/`](../collision/README.md) against the scene
points. It never commands motion; the pick loop takes the first candidate whose sweep is clear.

| `ApproachPathOutcome` | Means |
| --- | --- |
| `CLEAR` | every sampled pose fits |
| `BLOCKED` | at least one does not, and the report names which step |
| `NO_OBSTACLES` | there was no scene cloud to check against |
| `SKIPPED` | the leg was switched off (zero samples) |

`NO_OBSTACLES` and `SKIPPED` are separate on purpose: "nothing was in the way" and "nobody looked" are
different answers.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `TypeError` or `ValueError` from `GraspMotion` | a negative distance, `pre_open_width_mm` of `None` or 0, `close_speed` above 1 | fix the field or leave it unset |
| `ValueError` at the build | a pre-open wider than the hand opens, or any pre-open on a service with no hand | leave `pre_open_width_mm` unset |
| `ValueError` at the build | a `policy=` built around another arm or hand | pass `motion=GraspMotion(...)` instead |
| `TypeError` at the build | both `motion=` and `policy=` | pass one |
| `MOTION_FAILED`, status `unsupported` | the arm keeps no straight line for the final descent | use an arm driver that keeps lines |
| `CAMERA_FRAME_REJECTED` | a grasp still in the camera frame on a cell with a resolver | check the rig's declared calibration |
| `CameraWorldUnavailable`, raised | a camera could not vouch for the cell | the campaign stops; fix the camera first |

## Traps

- The swept validator ships off and mode-scoped: `robot.grasping.approach_validation` has
  `enabled: false` and `apply_modes` of only the dense modes, so switching it on in another mode is
  inert. Its defaults are an 80 mm standoff, a 100 mm lift, 6 approach and 4 lift samples.
- The policy lifts along +Z of the grasp's own frame, and the validator's `retreat_direction` defaults
  to +Z too. Both are a vertical lift only for a BASE grasp.
- `approach_clearance_mm` in [`scoring/`](../scoring/README.md) looks at one waypoint and is telemetry,
  not this check.
- The arm's own path is not checked here. Whole-robot collision belongs to the planner and the
  [safety layer](../../safety/README.md); this package only checks the end-effector.

## Status

| Capability | Evidence |
| --- | --- |
| The choreography, both camera resolvers, the frame guard | measured in simulation: `run_m1_pick`, `run_m2_pick` and `run_eih_pick` drive this policy |
| The swept-path validator | measured in simulation: the dense Isaac scene, `run_dense_pick`, switches it on per flag |
| Close verification on a physical hand | never touched hardware: the drivers read their signals, no physical grasp was judged |

## Files

| File | Holds |
| --- | --- |
| `grasp_motion.py` | `GraspMotion`, `build_execution_policy`, `foreign_policy_refusal` |
| `execution_policy.py` | `GraspExecutionPolicy`, `PolicyOutcome`, `PolicyReport` |
| `frame_resolver.py` | `FrameResolver`, the three resolvers, `FrameResolutionFailure` and `resolve_or_none` |
| `trajectory_safety.py` | `ApproachPathPolicy`, `ApproachPathOutcome`, `ApproachPathReport` and the sweep functions |

## Details

- [`loop/`](../loop/README.md) decides which grasp reaches this policy.
- [`planning/`](../planning/README.md) builds the standoff and lift poses, and
  [`collision/`](../collision/README.md) the envelope each sample reuses.
- [`robot/core`](../../core/README.md) holds `RobotArm`, `Gripper` and `ObjectDetectingGripper`.
- [Guide 05, the pick loop](../../../../docs/guide/05-pick-loop.md), and
  [the grasping config reference](../../../../docs/grasping-config-reference.md) for `approach_validation`.
- Tests: `tests/test_grasp_motion.py`, `tests/test_grasp_execution_policy.py`,
  `tests/test_frame_resolver.py`, `tests/test_grasp_trajectory_safety.py`,
  `tests/test_what_the_hand_verbs_read.py`.
