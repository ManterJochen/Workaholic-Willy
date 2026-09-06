# Grasp planning (`src.robot.grasping.planning`)

Scored contacts become 6-DoF grasp poses, the approach geometry a pick needs, and a vendor-neutral
reachability contract. Parallel-jaw pose construction from antipodal contacts, the small family of
approach poses for both jaw and suction end-effectors, and the `IKService` Protocol the execution
layer fulfils with a real solver.

## What it guarantees

Pure NumPy. Every pose is millimetres with an XYZW rotation and a tagged frame. It imports only
[`src.geometry`](../../../geometry/README.md) and the sibling [`contacts/`](../contacts/README.md)
package, never a driver, a motion planner, a collision backend or a scorer.

It plans geometry, not trajectories. `approach_waypoints` is a straight line in Cartesian space: it
avoids nothing. Obstacle-aware motion planning happens below the driver seam, and self-collision is
checked in the safety preflight. Importing either from here would invert the dependency stack.

Frame handling stays out. Eye-to-hand against eye-in-hand is the calculator's business; this package
neither stores nor caches a camera-to-base transform. `transform_grasp_pose` is the single rigid
transform primitive it offers, and it re-orthonormalises the composed rotation so a calibration
matrix carrying the slack `geometry.validate_transform` allows still yields a valid pose, then
stamps `metadata["source_frame"]`.

## The public surface

| Module | Public surface |
| --- | --- |
| `grasp_pose.py` | `GraspPose`, the immutable 6-DoF jaw value object. `rotation_matrix` columns are `[closing, binormal, approach]` in the pose's frame; `quaternion_xyzw` and the three axis accessors are derived; `as_pose()` and `to_dict()` are the exits. Validation is strict: finite, orthonormal with determinant +1, scores in `[0, 1]`, stored arrays read-only. |
| `pose_generation.py` | `grasp_pose_from_contact_pair` for one antipodal pair, with the approach projected perpendicular to the closing axis, and `generate_grasp_poses` for an order-preserving batch. |
| `approach_planner.py` | `pre_grasp_pose` (standoff along the negated approach, default 80 mm), `retreat_pose` (default 100 mm along `direction`, default world `+Z`), `approach_waypoints`. |
| `suction_approach.py` | `suction_tool_orientation`, `suction_pose`, `suction_approach_waypoints`, `suction_grasp_poses` and the `SuctionApproach` recipe it returns. |
| `reachability.py` | `IKService` (Protocol, `query(pose) -> IKResult`), `IKResult`, `WorkspaceBoxIKService`, `filter_reachable_poses`, `transform_grasp_pose`, and `IKQualityMetrics`. |
| `multifinger.py` | `FingerKinematicSpec`, `GripperKind`, `MultiContactGrasp`, `MultiContactPlanRequest`, the `MultiContactGraspPlanner` Protocol, and the reference planners `ParallelJawContactPlanner` and `RadialMultiFingerPlanner`. |

`IKQualityMetrics` is the one name not re-exported from the package `__init__`; import it from
`planning.reachability`.

### Three grasp value objects, on purpose

Suction and parallel-jaw share the position half of a pick, not the orientation half. A `GraspPose`
carries a full frame built from two contacts. A `SuctionGrasp` carries a contact and a press
direction, and the cup is axisymmetric, so roll about it is free and one fixed wrist orientation
serves. A `MultiContactGrasp` carries N contacts and its own per-finger standoff. That is why
`approach_planner.py` and `suction_approach.py` are separate modules, and why there is deliberately
no single shared pose type.

## Usage

```python
from src.robot.grasping.planning import (
    generate_grasp_poses, pre_grasp_pose, approach_waypoints,
    suction_grasp_poses, WorkspaceBoxIKService, filter_reachable_poses,
)

poses = generate_grasp_poses(contact_pairs)            # GraspPose list, in the contacts' frame
pre = pre_grasp_pose(poses[0], standoff_mm=80.0)       # a GraspPose, standoff along -approach
wps = approach_waypoints(poses[0], num_waypoints=4)    # geometry.Pose list, pre-grasp to grasp
ik = WorkspaceBoxIKService(min_corner_mm=[-500, -500, 0], max_corner_mm=[500, 500, 800])
reachable, diagnostics = filter_reachable_poses(poses, ik)

# Suction: a SuctionGrasp gives the full recipe under one free-roll wrist orientation
recipe = suction_grasp_poses(suction_grasp, standoff_mm=60.0, contact_gap_mm=25.0, lift_mm=100.0)
arm.move(recipe.pre_grasp); arm.move(recipe.contact)   # seal, then arm.move(recipe.retreat)
```

## Traps

`retreat_pose` lifts in the grasp's own frame. The default `+Z` is a true vertical lift only for a
world-aligned base-frame grasp; for a camera-frame grasp `+Z` is the optical axis, so pass a
camera-frame direction.

`approach_waypoints` has no production caller. It is a straight interpolation, correct as geometry
and unaware of a bin wall, and nothing on the pick path calls it today.

`WorkspaceBoxIKService` is a stub. It tests position in a box and ignores orientation, joint limits
and singularities, rejecting with `reason="outside_workspace_box"`. It keeps the calculator useful
with no robot attached; real reachability comes from the execution layer's `IKService` adapter.
`IKResult.metadata` and `IKQualityMetrics` are carriers, not producers, and the box stub never sets
them.

Multi-finger planning is heuristic. It runs through `GraspCalculator.plan_multifinger`, two fingers
to `ParallelJawContactPlanner` and three or more to `RadialMultiFingerPlanner`, independently of
`compute()`, which stays strictly parallel-jaw. `RadialMultiFingerPlanner` produces geometrically
feasible N-contact candidates scored by radial uniformity and coverage, and does not prove dexterous
force closure; its metadata says so, tagged `confidence_kind="heuristic"`.

`GripperKind.VACUUM` is telemetry only. No code path produces it, because a suction grasp is a
single-contact `SuctionGrasp` and not a `MultiContactGrasp`.

## See also

- [`../README.md`](../README.md) for the tier this package belongs to
- [`../contacts/README.md`](../contacts/README.md) for the pairs poses are built from
- [`../collision/README.md`](../collision/README.md) for the filter that runs on the produced poses
- [`../motion/README.md`](../motion/README.md) for the policy that drives these poses at the arm
- [`../suction/README.md`](../suction/README.md) for `SuctionGrasp` synthesis and scoring
- [`../../safety/planning/README.md`](../../safety/planning/README.md) for the trajectory planner this
  package deliberately does not import
