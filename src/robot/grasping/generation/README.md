# Where grasp candidates come from (`src/robot/grasping/generation`)

`GraspCalculator`, the analytic grasp generator: a segmentation mask and a depth image go in, ranked
`GraspPoint` candidates come out. Pure numpy and OpenCV, with no robot, no torch and no vendor SDK. A
cell builds it through `build_calculator`, which reads `robot.grasping.calculator`; build it the same
way when you call it directly.

```python
from willy import Frame, build_calculator, load_tree
from src.robot.grasping import SupportPlane

tree = load_tree()                      # the cell WILLY_PROFILE names
calculator = build_calculator(
    tree.robot, data_dir=tree.root, camera_matrix=K,
    min_grip_width_mm=tree.robot.gripper.min_width_mm,
    max_grip_width_mm=tree.robot.gripper.max_width_mm,
)
result = calculator.compute_result(
    segmentation, depth_mm,             # anything with a boolean .mask; depth in millimetres
    camera_to_base=camera_to_base,      # 4x4, millimetres
    support_plane=SupportPlane(offset_mm=0.0, frame=Frame.BASE),   # the table the part stands on
)
print(result.telemetry["geometry_stage"], result.reasons)
best = result.best                      # a GraspPoint in the base frame, or None
```

`best.pose()` and `best.grip_width_mm` are what `Robot.pick` takes. `compute()` returns the ranked list
alone; `compute_result()` wraps it in a `GraspResult` with a typed `GraspFailureReason` and the
telemetry, which is what a caller needs to choose between retry, rescan and abort. With only a
base-frame cloud, [`Scene`](../README.md) is the shorter door.

## What `compute()` does

```
masked depth -> point cloud -> surface normals -> antipodal contact pairs -> 6-DoF poses
             -> geometric, stability and reachability scores -> collision, table, workspace, IK filters
```

Every formula in that chain is in [docs/grasping-math.md](../../../../docs/grasping-math.md).

## The support-footprint stage

The default geometry stage reconstructs the target as a vertical prism, its visible footprint extruded
down to the support plane, and enumerates the grasps that prism admits: the two axes of the footprint's
minimum-area rectangle, plus a radial fan when the footprint is round. For each it solves for the
height at which the fingers still clear the table. A filter can only refuse a grasp that goes into the
table; this stage plans the clearance instead. A cell sets it from `robot.grasping.geometry.stage`
(`support_footprint`, the default); a direct caller passes `support_footprint_geometry=`, default
`True`. The jaw dimensions come from `ParallelJawGripperModel`, the one measurement of the jaw every
consumer reads.

It needs a base-frame support plane and a CAMERA to BASE transform; without either it stands down to
the silhouette stage, which anchors grasps on the visible surface and lets the filters discard what does
not fit. It also gives up when fewer than 20 cloud points survive the support-height cut. The stage that
ran is stamped in `telemetry["geometry_stage"]`: `silhouette`, `support_footprint` or
`silhouette_fallback`. An empty result from the prism is an abstention, and
`support_footprint_fallback` (default `False`) decides whether silhouette candidates survive it. The
stage runs no IK, reachability or safety check: its candidates pass the same filters as any others, and
its own score is kept as `total_score`, because the geometric scorer does not order these candidates
usefully.

When the stage runs, an empty silhouette no longer ends the attempt: the stage replaces the silhouette's
candidates anyway, and on a camera tilted 45 degrees a 40 mm cube's silhouette spans its top and its near
side, 54.5 mm, wider than a 50 mm stroke. Three things keep the prism on the part when the view is
tilted, where the table seen past the part's far edge lies 40 mm behind it rather than under it:

- The mask pixels that measure more than 10 mm behind a pixel of the same mask within 3 px are left out
  of the stage's input (`depth_steps.pixels_behind_depth_steps`); `support_footprint_depth_step_pixels`
  in the telemetry counts them.
- Low fragments that lie apart from the body, and never rise to half its height, are left out of the
  footprint and kept as undilated obstacles, so no finger lands on them.
- `floor_margin_mm` is how far above the support a point still counts as the support: 2.0 by default,
  the number the reference measurements were made with in a simulator whose calibration is exact.
  `Scene.from_robot_config` passes the larger of that and
  `safety.planning_world.perceived.plane_clearance_mm`; a direct caller passes
  `support_footprint_floor_margin_mm=` to the calculator.

## Two closing-axis rules

**The closing axis lies in the support plane.** A jaw closes across an object standing on something,
so `level_closing_to_support_plane` (default `True`) levels the axis into that plane. Zeroing z in the
camera frame instead lands it in the image plane, which is the same only for a camera looking straight
down; on an oblique or wrist camera a box is then gripped on a face and an edge. It needs a CAMERA to
BASE transform, and `False` reproduces the unlevelled axis.

**A symmetric silhouette has no axis.** On a cube from above, a ball or a coin the two extents are equal
within noise, and the principal axis flips between frames of the same scene. The
constructor argument `isotropic_radial_closing` (default `False`) then closes radially from the robot
base, the direction a short arm can reach at any azimuth. It acts only on near-isotropic silhouettes
and near-vertical approaches, and needs a base-frame transform. Left off on an arm that needs it, the
pick rate on an unchanged scene varies from run to run and fails as a motion timeout, which reads as a
bad grasp rather than an unreachable one. The shipped UR3e profile sets
`robot.grasping.isotropic_radial_closing: true`, and the simulation runners read that key; the cell
builders do not pass it, so a config-built cell runs without it. Pass it to `build_calculator`
yourself where you build one.

## What it does not do

The deformable seam is inert. No caller passes `deformable_strategy=`, so the gate is skipped, and
`CablePCAStrategy` always refuses with a telemetry hint. This calculator is built for rigid
parallel-jaw grasps; cables, cloth and bags need another planner, and the seam exists so that the rigid
path never pretends otherwise.

## Status

| Capability | Evidence |
| --- | --- |
| The calculator in the pick service, both geometry stages | measured in simulation |
| The calculator at a physical cell | never touched hardware |

## Files

| File | Holds |
| --- | --- |
| [calculator.py](calculator.py) | `GraspCalculator`, and the `compute()` and `compute_result()` contract every pipeline relies on |
| [_candidate_generator.py](_candidate_generator.py) | the silhouette, geometry-first and dense candidate generators |
| [support_footprint.py](support_footprint.py) | the support-footprint stage; a candidate's `pose()` is the base pose `Robot.pick` takes |
| [_support_footprint_stage.py](_support_footprint_stage.py) | the adapter that brings that stage's base-frame candidates into the camera frame |
| [_camera_geometry.py](_camera_geometry.py) | image to camera-frame lifting, and `level_axis_to_support_plane` |
| [_isotropic_closing.py](_isotropic_closing.py) | the closing direction when the silhouette implies none |
| [mask_analyzer.py](mask_analyzer.py) | centroid, principal and minor axes, extents, oriented and axis-aligned boxes |
| [deformables.py](deformables.py) | the inert deformable seam |
| [_calculator_validation.py](_calculator_validation.py) | input validation |
| [_debug_renderer.py](_debug_renderer.py) | the opt-in overlay, drawn with `draw()` because it makes pixels, not text |

## Details

- [grasping/](../README.md): the stack this is the entry point of, and the generator selector
- [geometry/](../geometry/README.md), [contacts/](../contacts/README.md): the layers it consumes
- [scoring/](../scoring/README.md), [collision/](../collision/README.md), [planning/](../planning/README.md):
  the scores and filters its candidates pass
- [multiview/](../multiview/README.md): `geometry_points_base_mm`, which feeds a fused cloud into this stage
- [deep/](../deep/README.md): the learned generator selected in its place
