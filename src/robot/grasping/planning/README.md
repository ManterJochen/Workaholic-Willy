# Grasp planning (`src.robot.grasping.planning`)

Scored contacts become 6-DoF grasp poses: the jaw pose built from two opposing contacts, the standoff
and lift poses around it, the same for a suction cup, and the `IKService` contract a real reachability
solver fulfils. Pure NumPy geometry; it plans poses, not trajectories.

You reach it through the grasp calculator, which builds its contact-based candidates with it. Call it
directly to turn contact pairs of your own into poses:

```python
import numpy as np
from src.robot.grasping.planning import (
    WorkspaceBoxIKService, filter_reachable_poses, generate_grasp_poses, pre_grasp_pose,
)

poses = generate_grasp_poses(pairs)                   # from contacts.find_antipodal_pairs; CAMERA unless frame= says
pre = pre_grasp_pose(poses[0], standoff_mm=80.0)      # 80 mm back along the approach
box = WorkspaceBoxIKService(min_corner_mm=np.array([-500.0, -500.0, 0.0]),
                            max_corner_mm=np.array([500.0, 500.0, 800.0]))
reachable, results = filter_reachable_poses(poses, box)   # order kept; results parallel to poses
print(len(reachable), [r.reason for r in results if not r.reachable])
```

A suction grasp gets its whole recipe under one wrist orientation:

```python
from src.robot.grasping.planning import suction_grasp_poses

recipe = suction_grasp_poses(suction_grasp, standoff_mm=60.0, contact_gap_mm=25.0, lift_mm=100.0)
print(recipe.pre_grasp, recipe.contact, recipe.retreat)   # then the waypoints, the contact, the seal, the lift
```

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `GraspPose` | `grasp_pose_from_contact_pair(pair)`, `generate_grasp_poses(pairs)` | `as_pose()`, `to_dict()` | a `Pose`, or a dict |
| a standoff or a lift | the `GraspPose` you have | `pre_grasp_pose(pose)`, `retreat_pose(pose)`, `approach_waypoints(pose)` | a `GraspPose`, or a list of `Pose` |
| `SuctionApproach` | `suction_grasp_poses(suction_grasp, ...)` | read `pre_grasp`, `waypoints`, `contact`, `retreat` | poses sharing one orientation |
| `IKService` | an adapter in the execution layer, or `WorkspaceBoxIKService` | `query(pose)` | `IKResult` |
| `MultiContactGraspPlanner` | `ParallelJawContactPlanner`, `RadialMultiFingerPlanner` | through `GraspCalculator.plan_multifinger` | `MultiContactGrasp` candidates |

`GraspPose.rotation_matrix` has the columns `[closing, binormal, approach]` in the pose's frame; the
quaternion and the three axes are derived. Construction is strict: finite, orthonormal with determinant
+1, scores in `[0, 1]`, arrays read-only. Every pose is millimetres with an XYZW rotation and a tagged
frame.

## Three grasp value objects, on purpose

Suction and the jaw share the position half of a pick, not the orientation half. A `GraspPose` carries a
full frame built from two contacts. A `SuctionGrasp` carries one contact and a press direction, and the
cup is round, so one wrist orientation serves. A `MultiContactGrasp` carries N contacts and its own
per-finger standoff. That is why there is no single shared pose type.

## What it deliberately leaves out

It imports [`src.geometry`](../../../geometry/README.md) and, from the grasping stack, only
[`contacts/`](../contacts/README.md), the [`geometry/`](../geometry/README.md) validators and the shared
constants: never a driver, a motion planner, a collision backend or a scorer. Obstacle-aware motion
happens below the driver, and self-collision is checked in the safety preflight.

Frame handling stays with the calculator. `transform_grasp_pose(pose, T, target_frame=...)` is the one
rigid transform offered; it re-orthonormalises the composed rotation, so a calibration matrix carrying
the slack `geometry.validate_transform` allows still yields a valid pose, and stamps
`metadata["source_frame"]`.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `GraspPose` | a non-finite position, a rotation that is not orthonormal, a score outside `[0, 1]` | build it from contacts, or fix the rotation |
| `ValueError` from `WorkspaceBoxIKService` | corners not of shape (3,), or min not below max | pass two 3-vectors |
| `IKResult(reachable=False, reason="outside_workspace_box")` | the pose position is outside the box | widen the box, or wire a real `IKService` |
| `ValueError` from `generate_grasp_poses` | `max_poses` below 1 | pass `None` or a positive count |

## Traps

- `retreat_pose` lifts in the grasp's own frame. The default +Z is a vertical lift only for a BASE
  grasp; for a camera-frame grasp +Z is the optical axis, so pass a camera-frame `direction`.
- Nothing on the pick path calls `pre_grasp_pose`, `retreat_pose`, `approach_waypoints` or
  `suction_grasp_poses`: the execution policy in [`motion/`](../motion/README.md) builds its own standoff
  and lift. `approach_waypoints` is a straight interpolation, blind to a bin wall.
- `WorkspaceBoxIKService` is a stub: position in a box, blind to orientation, joint limits and
  singularities. It keeps the calculator useful with no robot attached. Real reachability comes from
  the execution layer's `IKService` adapters, and there is no config key for one: a boot script passes
  it to the calculator ([guide 05](../../../../docs/guide/05-pick-loop.md)).
- `IKQualityMetrics` is not re-exported; import it from `planning.reachability`. The box stub never
  fills it or `IKResult.metadata`.
- Multi-finger planning is heuristic. It runs through `GraspCalculator.plan_multifinger`, apart from
  `compute()`, which stays parallel-jaw. `RadialMultiFingerPlanner` scores radial uniformity and
  coverage, does not prove force closure, and tags its results `confidence_kind="heuristic"`.
- `GripperKind.VACUUM` is telemetry only: a suction grasp is a single-contact `SuctionGrasp`.

## Status

| Capability | Evidence |
| --- | --- |
| Jaw poses from contact pairs, and `transform_grasp_pose` | measured in simulation: the grasp calculator builds its candidates with them in the Isaac picks |
| Standoff, lift, waypoints and the suction recipe | never touched hardware: unit-tested, and no pick path calls them |
| Multi-finger planners | never touched hardware: no multi-finger hand is driven by this code |

## Files

| File | Holds |
| --- | --- |
| `grasp_pose.py` | `GraspPose` |
| `pose_generation.py` | `grasp_pose_from_contact_pair`, `generate_grasp_poses` |
| `approach_planner.py` | `pre_grasp_pose` (80 mm), `retreat_pose` (100 mm, world +Z), `approach_waypoints` |
| `suction_approach.py` | `suction_tool_orientation`, `suction_pose`, `suction_approach_waypoints`, `suction_grasp_poses`, `SuctionApproach` |
| `reachability.py` | `IKService`, `IKResult`, `WorkspaceBoxIKService`, `filter_reachable_poses`, `transform_grasp_pose`, `IKQualityMetrics` |
| `multifinger.py` | `FingerKinematicSpec`, `GripperKind`, `MultiContactGrasp`, `MultiContactPlanRequest`, the planners |

## Details

- [`contacts/`](../contacts/README.md) for the pairs poses are built from, and
  [`collision/`](../collision/README.md) for the filter the poses then pass.
- [`motion/`](../motion/README.md) for the policy that drives a grasp at the arm, and
  [`suction/`](../suction/README.md) for `SuctionGrasp`.
- [`safety/planning/`](../../safety/planning/README.md) for the trajectory planner this package does not
  import.
- [The grasping maths](../../../../docs/grasping-math.md) for the pose construction.
- Tests: `tests/test_ik_service_conformance.py`, `tests/test_execution_ik_service.py`,
  `tests/test_suction_approach.py`, `tests/test_multifinger_planner.py`, `tests/test_grasping_boundaries.py`.
