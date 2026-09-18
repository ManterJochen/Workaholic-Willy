# Grasp collision (`src.robot.grasping.collision`)

Does this hand fit here: the check every grasp candidate passes before any IK or motion planning. It
tests the gripper envelope against the scene cloud, the surface the parts stand on, the container walls
and the workspace box.

You reach it through the grasp generators, which both run `filter_candidates`, and through your cell's
config, which sizes the envelope. Call it directly to check one pose against a cloud of your own:

```python
from src.robot.grasping.collision import SupportPlane, validate_grasp_collision

# `pose` is a planning.GraspPose; the points are (N, 3) millimetres in the same frame as the pose.
result = validate_grasp_collision(
    pose,
    scene_points_mm=cloud_mm,
    support_plane=SupportPlane(),         # the table, in the CAMERA frame by default
    min_table_clearance_mm=5.0,
)
if not result.valid:
    print(result.reasons)                 # "table_clearance", "point_cloud_collision", or both
```

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| a `GraspPose` | [`planning/`](../planning/README.md) | `validate_grasp_collision(pose, ...)` | `GraspCollisionResult` |
| a list of `GraspPose` | a grasp generator | `filter_candidates(poses, ...)` | `FilterOutcome`: kept indices and a histogram |
| `SupportPlane` | `SupportPlane(normal=..., offset_mm=..., frame=...)` | `resolve_support_plane(...)` picks one | `SupportResolution` |
| a declared bin | `robot.grasping.support.container` | `container_wall_points_base_mm(...)` | wall points in BASE mm |

`GraspCollisionResult` is frozen: `valid`, `reasons`, `min_table_clearance_mm`, `collision_count`,
`colliding_indices`, `colliding_boxes`, `metadata`, and `to_dict()`. `validate_grasp_collisions` checks
a batch in input order. Everything is NumPy, deterministic and in millimetres.

## The rejection stage

`filter_candidates` is the one rejection stage both generators run, so their histograms read side by
side. `REJECTION_KEYS` fixes the names and the order, cheapest first: `rejected_workspace` (three
comparisons), `rejected_table`, then `rejected_collision` (the cloud test). A candidate outside the box
never pays for the cloud test.

It works in any frame and refuses a mix. The analytic generator filters CAMERA-frame poses and the
learned one BASE-frame poses, so the poses, the support plane and the points must share one frame.
The frame check runs first, before any work.

## Two gripper families, one contract

Every check takes a `GripperGeometryStrategy` (`collision_boxes` and `local_corners_mm`), so any
end-effector is treated the same. Both shipped models bound their parts with boxes in the grasp frame:
X closes between the contacts, Y is the binormal, Z is the approach.

| Model | Envelope | Grip width |
| --- | --- | --- |
| `ParallelJawGripperModel` | two fingers, their contact pads, and the palm | sizes the finger gap |
| `SuctionCupGripperModel` | cup, shaft and wrist mount back along the negated approach | ignored, there are no jaws |

A third end-effector is a model that satisfies the strategy; nothing else changes.

`robot.grasping.gripper_geometry` selects the envelope, and `build_gripper_geometry` in
[`execution/autonomous_grasp/builders.py`](../../execution/autonomous_grasp/builders.py) builds it:

```yaml
grasping:
  gripper_geometry:
    kind: parallel_jaw          # parallel_jaw or suction
    outer_margin_mm: 0.0        # a skin inflating every box on all six sides
    parallel_jaw: { finger_length_mm: 39.98, palm_depth_mm: 35.0, palm_width_mm: 70.0 }
    suction:      { cup_radius_mm: 15.0, shaft_length_mm: 40.0, mount_radius_mm: 30.0 }
```

The defaults are `ParallelJawGripperModel()` with no arguments, measured off a real two-finger gripper's
collision shapes at the worst case over its aperture. A cell that names a hand (`robot.gripper.model`)
does not write this block: the loader fills every unset number and the kind from that hand's registry
file.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ValueError` from `filter_candidates` | the poses are in more than one frame, or the plane and the poses disagree | convert one: `SupportPlane.to_camera_frame(camera_to_base)` for the analytic path |
| `TypeError` | `validate_grasp_collision` given anything but a `GraspPose` | build the pose with `planning/` |
| `ValueError` | a negative or non-finite clearance or margin | pass a finite number of at least 0 |
| refused at load | a stated envelope number that contradicts the named hand's registry file | remove it, or fix the registry |
| refused at load | `wall_collision_enabled` without both interior corners of the container | declare `interior_min_mm` and `interior_max_mm` |

## Traps

- The cloud test is box containment per gripper box, not a swept volume and not a mesh. A deep overhang
  or a thin protrusion can be counted wrong; `collision_margin_mm` or `outer_margin_mm` compensates.
- The support surface is one half-space. Stepped bins and tiered fixtures are not modelled.
- With no support plane the table check is skipped, not passed, so `rejected_table` reads zero.
  `resolve_support_plane` supplies one: the higher of the declared height and the target's lowest
  observed point, because a declared constant cannot know a part stands on another part.
- Container walls are declared, never perceived: no detector segments a bin wall. They ship off
  (`wall_collision_enabled: false`). Declaring them does not raise the pick rate; it stops the ranker
  offering a grasp that drives a finger into the bin wall, which turns a wrong pick into an honest
  no-pick. A cell with a bin should switch them on anyway.
- The suction envelope comes from the cup and shaft dimensions, not a measured fit; a real seal check
  is the seal model's job.
- The arm's path is a different concern: whole-robot collision on the way to the grasp belongs to the
  planner and the [safety layer](../../safety/README.md). This package only asks about the end-effector.

## Status

| Capability | Evidence |
| --- | --- |
| The rejection stage and the jaw envelope | measured in simulation: the Isaac picks run it through the grasp generator |
| Container walls | measured in simulation: top-1 flat on the datagen reference; the bin family's ranking gap falls from 15 objects to 1 |
| The suction envelope | never touched hardware: unit-tested, and no simulated pick filters against it |

## Files

| File | Holds |
| --- | --- |
| `gripper_model.py` | `CollisionBox`, `ParallelJawGripperModel`, `SuctionCupGripperModel`, `GripperGeometryStrategy`, `points_to_grasp_frame` |
| `collision_checker.py` | `GraspCollisionResult`, `validate_grasp_collision`, `validate_grasp_collisions`, `colliding_point_indices` |
| `candidate_filter.py` | `filter_candidates`, `FilterOutcome`, `REJECTION_KEYS`, `rejection_reasons`, `grasp_point_to_pose` |
| `table_collision.py` | `SupportPlane`, `gripper_table_clearance_mm` |
| `support_resolver.py` | `resolve_support_plane`, `SupportResolution` |
| `container.py` | `container_wall_points_base_mm` |

## Details

- [`generation/`](../generation/README.md) for `_apply_collision_filters`, the analytic caller.
- [`motion/`](../motion/README.md) for the swept check of the moving gripper along the path.
- [`suction/`](../suction/README.md) for the candidates the cup envelope is built for.
- [`safety/planning/`](../../safety/planning/README.md) for the arm-level collision engine this package
  does not duplicate.
- [Guide 06, grippers](../../../../docs/guide/06-grippers.md) and
  [your own gripper](../../../../docs/runbooks/your_own_gripper.md) for a hand's registry file.
- Tests: `tests/test_container_walls.py`, `tests/test_support_resolver.py`,
  `tests/test_support_plane_frame.py`, `tests/test_gripper_dimensions_agree.py`,
  `tests/test_deep_rejection_stage.py`.
