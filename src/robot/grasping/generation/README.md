# Generation: where candidates come from

`GraspCalculator` is the analytic generator. A segmentation mask and a depth map go in, ranked
`GraspPoint` candidates come out. Pure numpy and OpenCV: no robot, no torch, no vendor SDK.

| Module | Owns |
|---|---|
| [`calculator.py`](calculator.py) | `GraspCalculator`, and the `compute()` / `compute_result()` contract every pipeline depends on |
| [`_candidate_generator.py`](_candidate_generator.py) | The three candidate generators: silhouette, geometry-first and dense |
| [`support_footprint.py`](support_footprint.py) | The support-footprint geometry stage, which plans the table clearance rather than filtering for it |
| [`_support_footprint_stage.py`](_support_footprint_stage.py) | The adapter that rotates that stage's BASE candidates into CAMERA at the seam |
| [`_camera_geometry.py`](_camera_geometry.py) | Image space to camera frame lifting, and `level_axis_to_support_plane` |
| [`_isotropic_closing.py`](_isotropic_closing.py) | Choosing a closing direction when the silhouette does not imply one |
| [`mask_analyzer.py`](mask_analyzer.py) | Centroid, principal and minor axis, extents, oriented and axis-aligned boxes |
| [`deformables.py`](deformables.py) | The deformable routing seam, which is inert; see below |
| [`_calculator_validation.py`](_calculator_validation.py) | Input validation |
| [`_debug_renderer.py`](_debug_renderer.py) | The opt-in overlay draw. Its verb is `draw()`, not `render()`, because it makes pixels and not text |

## What `compute()` does

```
masked depth -> metric point cloud -> local surface normals -> antipodal contact-pair search
             -> 6-DoF pose generation -> geometric, stability and reachability scoring
             -> collision, table, workspace and IK filters -> ranked GraspPoints
```

`compute()` returns the ranked list. `compute_result()` returns the same candidates wrapped in a
`GraspResult` that carries a typed `GraspFailureReason` and the per-call telemetry, which is what a
caller needs to choose between retry, rescan, escalation and abort. Every formula in the chain above
is written out in [`docs/grasping-math.md`](../../../../docs/grasping-math.md).

## Two geometry stages, and why the default is the second one

The silhouette stage anchors a grasp on the visible surface and lets the collision filter discard
what does not fit. That is the wrong order when the binding constraint is the support surface itself,
because a filter can only refuse a grasp, it cannot relocate one to somewhere legal.

The support-footprint stage reconstructs the target as a convex vertical prism, the visible footprint
extruded down onto the calibrated support plane, and then enumerates the grasps that prism admits:
the antipodal side-face pairs are the minimum-area rectangle's two axes, plus a radial fan when the
footprint is round, and for each axis it solves for the anchor height at which the gripper still
clears the table. Everything it consumes exists in a real cell: a masked depth image, the camera
pose, the calibrated support plane, the gripper dimensions and the rest of the depth image as
obstacles.

It is the default (`support_footprint_geometry=True`, set from `robot.grasping.geometry.stage`), and
it needs a BASE-frame support plane and a CAMERA to BASE transform. Without both it cannot run and
the path stands down to the silhouette stage unchanged. `reconstruct_support_prism` also gives up
when fewer than 20 cloud points survive the support-height cut.

Which stage actually ran is stamped into the `geometry_stage` telemetry key, as `silhouette`,
`support_footprint` or `silhouette_fallback`. The key is written only while the stage is enabled,
because default-off has to mean byte-identical in what gets logged as well. An empty result from the
prism stage is an abstention rather than a veto, and `support_footprint_fallback` (default `False`)
decides whether the silhouette candidates survive it.

Two properties of the stage are worth knowing before you measure it:

- It is a geometry stage, not a pick. It runs no IK, no reachability and no safety preflight. Its
  candidates pass through the same scoring, collision, workspace and IK filters as any others.
- The gripper dimensions come from `ParallelJawGripperModel`, and are not restated here. There is one
  measurement of the jaw and every consumer reads it. A second copy drifts, and a copy that is too
  lenient toward the table is a copy that proposes grasps into the surface.

The stage's own score is used as `total_score` directly rather than being re-ranked by the geometric
scorer, because the geometric score does not order these candidates usefully.

## The closing axis lies in the support plane

A parallel jaw closes across an object standing on something, so its closing axis lies in the support
plane. Building that axis by zeroing z in the CAMERA frame lands it in the image plane instead, which
is the same thing only for a camera looking straight down the support normal. On an oblique or wrist
camera it is not, and a tilted closing axis on a box lands on a face and an edge rather than on two
opposed faces, which reads downstream as a candidate that is not antipodal.

So `level_closing_to_support_plane` defaults to `True` and levels the axis into the plane. It needs a
CAMERA to BASE transform; without one the direction of up is unknown and nothing is levelled. Setting
it `False` reproduces the unlevelled axis exactly.

## The degenerate axis on a symmetric silhouette

A jaw closes across the short dimension, read off the mask's principal axis, and that works while the
object has a long and a short dimension. On a near-isotropic silhouette, a cube from above, a ball, a
coin, the two extents are equal to within segmentation noise and the principal axis is numerically
degenerate: it is decided by which of two nearly equal eigenvalues comes out larger, and it flips
between frames of the same static scene.

Following it is not neutral, because the arm does not treat every yaw equally. A shorter arm can fail
to plan a tangential top-down close at any azimuth while planning the radial one everywhere, and a
longer arm absorbs both. On a symmetric object the yaw is free by definition, so spending it on the
reachable direction costs nothing.

`isotropic_radial_closing` is that choice, and it defaults to `False` because it is a property of the
arm rather than of the scene: only a cell that needs it declares it, which the shipped UR3e profile
does. It needs a BASE-frame transform to mean anything. The scope is deliberately narrow. Only
near-isotropic silhouettes, because an elongated object's axis carries real information and
overriding it would close the jaw across the long dimension. Only near-vertical approaches, because
for a tilted approach the yaw is coupled and not free to spend. The sign is irrelevant, both
directions plan, so the nearer one is chosen.

The symptom when this is left off is a non-deterministic pick rate on a byte-identical scene,
surfacing as a motion timeout, which reads as a bad grasp rather than an unreachable one.

## The deformable seam is inert, and says so

No caller in this library or in the simulation runners passes `deformable_strategy=` or
`deformable_class=` to the calculator, so the gate is skipped by default, and `CablePCAStrategy`
always refuses: it emits a telemetry hint and never yields a deformable grasp.

It is a clean extension point awaiting a real classifier and handler, not an active capability. This
calculator is engineered for rigid parallel-jaw grasps. Cables, cloth and bags need a different
planning regime, and pretending otherwise inside the rigid pipeline is what the seam exists to
prevent.

## See also

- [`../README.md`](../README.md) for the tier this is the entry point of
- [`../../../../docs/grasping-math.md`](../../../../docs/grasping-math.md) for every formula this
  package evaluates
- [`../geometry/README.md`](../geometry/README.md) and [`../contacts/README.md`](../contacts/README.md)
  for the layers it consumes
- [`../scoring/README.md`](../scoring/README.md), [`../collision/README.md`](../collision/README.md)
  and [`../planning/README.md`](../planning/README.md) for the ranks and filters its candidates pass
- [`../multiview/README.md`](../multiview/README.md) for `geometry_points_base_mm`, the seam that
  feeds a fused cloud into this stage
- [`../deep/README.md`](../deep/README.md) for the learned generator this one is selected against
