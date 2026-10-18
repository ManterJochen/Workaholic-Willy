# Grasp motion (`src.robot.grasping.motion`)

How the arm and the hand carry out a chosen grasp: the approach, the close and the lift, and the transform
that places the camera. Which grasp to attempt is decided one layer up, in [`loop/`](../loop/README.md).

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

## Choosing the closing axis

`GraspMotion(closing_axis="-y")` names the axis the jaws close along. Only grasps whose closing axis
already **heads within 30 degrees** of it, either way round, are taken, each turned half a turn about its
approach where that puts it the named way round: the same two contact faces, the jaws swapped. **Choose,
don't twist.** The heading counts, the tilt stays free. The choosing runs in the pick loop before anything
judges or ranks the grasps, so the grasp judged is the grasp gripped and `both_faces` works with it
([loop/](../loop/README.md)).

- **Values.** A name `Pose.tool_down` takes (`x`, `-x`, `y`, `-y`, `radial`, `-radial`, `tangential`,
  `-tangential`, a leading `+` allowed; `radial` and `tangential` read at each grasp's place), or a taught
  orientation, a quaternion `(x, y, z, w)` or a BASE `Pose`, whose tool +X laid onto the base XY plane is
  the heading. `closing_axis="-y"` and `closing_axis=Pose.tool_down(x, y, z, closing_axis="-y")` pick the
  same grasps.
- **None along it.** A pick with no grasp along the axis ends `no_valid_grasp`, its failure line naming
  the axis and the 30 degrees.
- **The cell's own direction** is `robot.motion.natural_closing_axis`, which turns every grasp the nearer way round
  and leaves none out; a named `closing_axis` wins over it.
- **The simulator's twist.** `align_closing_to_base_x` turns a chosen grasp afterwards (the simulator's aid
  for symmetric parts); beside a `closing_axis` it raises `ValueError` before anything moves, whatever true
  value it holds.
- There is no config key: the axis is a program's choice. `build_execution_policy` hands it to
  `GraspExecutionPolicy.closing_axis`, which `execute()` does not act on; the pick loop reads it.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspMotion` | `GraspMotion(...)`, every field optional | `to_dict()` | the fields you chose |
| `GraspExecutionPolicy` | `build_execution_policy(motion, arm=..., ...)` in the pick service | `execute(grasp)` | `PolicyReport` |
| `FrameResolver` | one of the three resolvers below | `camera_to_base_for_frame(frame, arm=...)` | a CAMERA to BASE `Transform` |

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
is never switched before the approach: it is asked whether its jaws stand open, and asks a person where it
believes them closed; at the grasp it gets one close, and nothing verifies it. On an arm that judges a grasp's
lift before its jaws close (`JudgesCarriedLines`, the UR driver), the lift is judged at the part as it will run
once they close, and the jaws close only where it would run, the controller asked last (the owner, 2026-10-01).
No part in the jaws is modelled (every cell since 2026-10-17, the owner's since 2026-10-05): a closed hand is
judged as an empty one, by the exact guard at full stroke. The lift is judged again as it starts, so a lift in
steps, or a new camera frame on an arm that holds no world at the part, can still refuse it after the close; the
arm then stands at the part with the part in its jaws, and the next motion is judged from there as every motion
is (`Robot.release` opens the jaws).

Three of the owner's choices (2026-10-09) move a judgement to where the arm waits anyway and change nothing of what
is judged; all three are fixed since 2026-10-17, as the owner's cell ran them. A cuRobo arm gates its own sends
(`safety.dwell.gate_at: send` until then): it judges first and waits right before it sends; the policy waits for
nothing, the steady gate being the arm's own (P0e E9). Where the arm holds the world the grasp was judged in ahead
(every arm that holds its world: the owner's `planning_world.hold`), the move to the standoff and the line down are
judged in it, with no frame at the standoff (`holding_the_approach`). Where the arm judges the next leg, as every
cuRobo arm does (`robot.motion.judge_next_leg` until 2026-10-17), the close leaves the jaws' stroke to the policy
where the hand can (`wait_settled`): the one change goes out, the joint move a task declared after the pick
(the carry) is judged meanwhile in the world held at the part, and the lift goes out once the stroke is over; a
hand that leaves no stroke has the lift's send judge it while the lift runs, where the arm can.

| `PolicyOutcome` | Means | Moved |
| --- | --- | --- |
| `EXECUTED` | approach, close and lift completed; the part is held or trusted to be | yes |
| `OBJECT_NOT_DETECTED` | the close finished and the hand reports an empty jaw | yes |
| `MOTION_FAILED` | a move raised or was refused; the report carries the status and the message | partly |
| `CAMERA_FRAME_REJECTED` | a camera-frame grasp reached a policy that requires BASE | no |
| `GRIPPER_FAULT` | the gripper raised (the report carries it, never a raise out of `execute`), or a toggle hand would not start | no, or up to the close |
| `CARRIED_RETREAT_REFUSED` | at the part, the lift judged before the jaws close would be refused, or could not be judged: the jaws stayed open and the arm went back up the line to the standoff; the pick loop reads it as a failed execution | down and back up, empty-handed |

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

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `TypeError` or `ValueError` from `GraspMotion` | a negative distance, `pre_open_width_mm` of `None` or 0, `close_speed` above 1 | fix the field or leave it unset |
| `ValueError` at the build | a pre-open wider than the hand opens, or any pre-open on a service with no hand | leave `pre_open_width_mm` unset |
| `ValueError` at the build | a `policy=` built around another arm or hand | pass `motion=GraspMotion(...)` instead |
| `ValueError` before anything moves | `both_faces` asked on a pick whose motion sets `align_closing_to_base_x=True`: that yaw turns a symmetric top-down grasp to close along base X, onto faces nobody judged | ask for one or the other; to close along one axis with `both_faces`, name it with `closing_axis` (`robot.motion.natural_closing_axis` chooses only the way round); `run_dense_pick` sets the yaw whenever its planner is not cuRobo |
| `ValueError` from `GraspMotion`, `GraspExecutionPolicy`, or `run()` before anything moves | a `closing_axis` beside the twist (any true value); a name `Pose.tool_down` does not take; a `Pose` not in BASE; an orientation whose tool +X lies within 10 degrees of the vertical; an axis set on the policy later that names none | name one axis, and no twist |
| `TypeError` from `GraspMotion` | a `closing_axis` that is none of the values above, `None` included | leave it unset instead |
| `TypeError` at the build | both `motion=` and `policy=` | pass one |
| `MOTION_FAILED`, status `unsupported` | the arm keeps no straight line for the final descent | use an arm driver that keeps lines |
| `CAMERA_FRAME_REJECTED` | a grasp still in the camera frame on a cell with a resolver | check the rig's declared calibration |
| `CameraWorldUnavailable`, raised | a camera could not vouch for the cell | the campaign stops; fix the camera first |

## Traps

- The swept validator of the approach and the lift left on 2026-10-16 with
  `robot.grasping.approach_validation`: switched on, it changed no pick of the owner's cell's replay bank,
  and the exact guard judges every sample of every motion anyway.
- The policy lifts along +Z of the grasp's own frame, a vertical lift only for a BASE grasp.
- `approach_clearance_mm` in [`scoring/`](../scoring/README.md) looks at one waypoint and is telemetry.
- The arm's own path is not checked here. Whole-robot collision belongs to the planner and the
  [safety layer](../../safety/README.md); this package only checks the end-effector.

## Status

| Capability | Evidence |
| --- | --- |
| The choreography, both camera resolvers, the frame guard | measured in simulation: `run_m1_pick`, `run_m2_pick` and `run_eih_pick` drive this policy |
| Close verification on a physical hand | never touched hardware: the drivers read their signals, no physical grasp was judged |

## Files

| File | Holds |
| --- | --- |
| `grasp_motion.py` | `GraspMotion`, `build_execution_policy`, `foreign_policy_refusal` |
| `execution_policy.py` | `GraspExecutionPolicy`, `PolicyOutcome`, `PolicyReport`, `closing_axis_twisted` |
| `frame_resolver.py` | `FrameResolver` and the three resolvers |

## Details

- [`loop/`](../loop/README.md) decides which grasp reaches this policy.
- [`planning/`](../planning/README.md) builds the standoff and lift poses.
- [`robot/core`](../../core/README.md) holds `RobotArm`, `Gripper` and `ObjectDetectingGripper`.
- [Guide 05, the pick loop](../../../../docs/guide/05-pick-loop.md).
- Tests: `tests/test_grasp_motion.py`, `tests/test_grasp_execution_policy.py`,
  `tests/test_frame_resolver.py`,
  `tests/test_what_the_hand_verbs_read.py`, `tests/test_a_pick_closes_along_the_axis_its_program_names.py`.
