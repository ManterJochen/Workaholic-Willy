# Grasp collision (`src.robot.grasping.collision`)

The geometric rejection stage: does this end-effector fit here, given the scene cloud, the surface
the parts stand on, the container walls and the workspace box. It runs over every candidate at
synthesis time, before any IK or motion planning.

## What it guarantees

Pure NumPy, deterministic, strict input validation. Millimetres throughout, and rotations come from
`GraspPose.rotation_matrix`.

Frame-agnostic, and fail-closed about it. The analytic calculator filters CAMERA-frame poses and the
learned one emits BASE-frame poses, so `filter_candidates` takes the pose, the support plane and the
points in one frame and refuses when the pose and the plane disagree about which. That refusal is
checked first, before any work, because a frame defect here is two quantities in different frames
meeting silently.

The arm's path is a different concern. Whole-robot self-collision and world-collision on the way to
the grasp belong to the motion planner and the [safety layer](../../safety/README.md). This package
only ever asks about the end-effector at a given pose, and the two never overlap.

## The public surface

| File | Role |
| --- | --- |
| `gripper_model.py` | `CollisionBox`, `ParallelJawGripperModel`, `SuctionCupGripperModel`, the `GripperGeometryStrategy` Protocol, `points_to_grasp_frame` |
| `collision_checker.py` | `GraspCollisionResult`, `validate_grasp_collision`, `validate_grasp_collisions`, `colliding_point_indices` |
| `candidate_filter.py` | `filter_candidates`, `FilterOutcome`, `REJECTION_KEYS`, `rejection_reasons`, `grasp_point_to_pose` |
| `table_collision.py` | `SupportPlane`, `gripper_table_clearance_mm` |
| `support_resolver.py` | `resolve_support_plane`, `SupportResolution` |
| `container.py` | `container_wall_points_base_mm` |

`filter_candidates` is the one rejection stage both grasp generators run, so their telemetry
histograms can be read side by side. `REJECTION_KEYS` fixes those names and their order:
`rejected_workspace`, `rejected_table`, `rejected_collision`. It applies them in that order, which
is cheapest first: the workspace box is three comparisons and the cloud test is the expensive one, so
a candidate outside the box never pays for it.

### Two gripper families, one contract

Every check is parametrised by a `GripperGeometryStrategy`, a small contract of `collision_boxes` and
`local_corners_mm`, so the checker treats any end-effector the same way. Both shipped models bound
their parts with axis-aligned boxes in the local grasp frame, where X closes between the contacts, Y
is the binormal and Z is the approach.

| Model | Envelope | Grip width |
| --- | --- | --- |
| `ParallelJawGripperModel` | two fingers, their contact pads, and the palm | sizes the finger gap |
| `SuctionCupGripperModel` | cup, shaft and wrist mount back along the negated approach | ignored, there are no jaws |

Adding a third end-effector means writing a model that satisfies the strategy. Nothing else changes.

## Usage

```python
import numpy as np
from src.robot.grasping.collision import SupportPlane, validate_grasp_collision

# `pose` is a planning.GraspPose; scene_points_mm is (N, 3) in the same frame as the pose.
result = validate_grasp_collision(
    pose,
    scene_points_mm=cloud_mm,
    support_plane=SupportPlane(),
    min_table_clearance_mm=5.0,
)
if not result.valid:
    print(result.reasons)   # "table_clearance", "point_cloud_collision", or both
```

`GraspCollisionResult` is frozen: `valid`, `reasons`, `min_table_clearance_mm`, `collision_count`,
`colliding_indices`, `colliding_boxes`, `metadata`, plus `to_dict()`. Batch with
`validate_grasp_collisions(poses, ...)`, which preserves input order. `colliding_point_indices`
returns the raw hit indices with the box labels, and `gripper_table_clearance_mm` the minimum
envelope clearance.

### Sizing the envelope from config

`robot.grasping.gripper_geometry` selects the envelope, and `build_gripper_geometry` in
`execution/autonomous_grasp/builders.py` turns the block into the matching model:

```yaml
grasping:
  gripper_geometry:
    kind: parallel_jaw          # parallel_jaw or suction
    outer_margin_mm: 0.0        # a safety skin inflating every box on all six sides
    parallel_jaw: { finger_length_mm: 39.98, palm_depth_mm: 35.0, palm_width_mm: 70.0 }
    suction:      { cup_radius_mm: 15.0, shaft_length_mm: 40.0, mount_radius_mm: 30.0 }
```

The shipped defaults are the dimensions of `ParallelJawGripperModel()` built with no arguments, so
leaving the block unset changes nothing. The parallel-jaw numbers are measured off a real
two-finger gripper's collision shapes at the worst case over its aperture range, because every
millimetre understated is finger hidden from the clearance check and pointing at the support surface.

## Traps

The point-cloud test is coarse by design: an axis-aligned box containment test per gripper box, not
a swept volume and not a mesh. A deep overhang or a thin protrusion can be under- or over-counted;
`collision_margin_mm`, or `outer_margin_mm` on the model, compensates.

The support surface is a single half-space. `SupportPlane` defaults to `Frame.CAMERA`, because the
analytic calculator's filter runs on camera-frame poses; `to_camera_frame` converts a BASE plane.
Multi-tiered fixtures and stepped bins are not modelled.

Given no support plane at all, the clearance check is skipped entirely rather than passed, so
`rejected_table` reads zero on a cell that never supplied one. `resolve_support_plane` is what
supplies it: it takes the higher of the declared height and the target's own lowest observed point,
because a declared constant cannot know that a part is standing on another part.

Container walls are declared, never perceived. No detector segments a bin wall as an object, so no
amount of camera fusion reaches one, and nothing else in the grasping stack carries wall geometry.
`container_wall_points_base_mm` turns `robot.grasping.support.container` into points the existing
checks already accept, and it ships off (`wall_collision_enabled: false`). Declaring them does not
raise the pick rate. What it does is stop the ranker offering a grasp that drives a finger into the
side of the bin, which turns a wrong pick into an honest no-pick, so a cell with a bin should switch
it on even though the rate will not thank it.

The suction envelope is analytic. `SuctionCupGripperModel`'s dimensions come from the cup and shaft
geometry rather than a measured fit, and a real seal check stays the seal model's job and real
hardware's.

## See also

- [`../README.md`](../README.md) for the tier this package pre-filters for
- [`../generation/README.md`](../generation/README.md) for `_apply_collision_filters`, the production
  caller
- [`../motion/README.md`](../motion/README.md) for the swept sweep of the moving gripper along the path
- [`../suction/README.md`](../suction/README.md) for the scorer whose candidates the cup envelope is built for
- [`../../safety/planning/README.md`](../../safety/planning/README.md) for the arm-level collision
  engine this package deliberately does not duplicate
