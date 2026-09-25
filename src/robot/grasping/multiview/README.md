# Multi-camera fusion (`src.robot.grasping.multiview`)

One depth view sees one side of a part, and a parallel-jaw grasp needs two opposite faces. This package
decides which blob in another camera is the same object and fuses the surfaces that agree into one BASE
cloud per object, so the grasp generator sees sides the primary camera cannot.

You reach it through your cell's config and the pick loop; there is nothing to call. A two-camera cell
turns it on under `robot.grasping.fusion`, as the shipped `ur5e,eth2` profile does in
[`config/robot/robot.eth2.yaml`](../../../../config/robot/robot.eth2.yaml):

```yaml
robot:
  grasping:
    fusion:
      enabled: true
      cameras: { cam_left: { enabled: true }, cam_right: { enabled: true } }   # rig ids, primary included
      geometry: { enabled: true, metric: overlap, min_score: 0.30, on_camera_unavailable: degrade }
```

Call it directly only to try a fusion on frames of your own:

```python
from src.robot.grasping.multiview.scene_geometry import ObservedView, fuse_scene_geometry

# Masks, a depth map in mm, a 3x3 lens matrix and a 4x4 CAMERA to BASE matrix per camera.
side = ObservedView(name="side", masks=side_masks, depth_map=side_depth_mm,
                    intrinsics=side_lens, camera_to_base=side_to_base)
fused = fuse_scene_geometry(masks, depth_mm, lens, camera_to_base, [side])
print(fused.views_used, fused.objects_fused)
cloud = fused.cloud_for(0)   # object 0 in BASE mm from every camera that saw it, or None
```

## Two things are called fusion

They share a config prefix and are not the same feature: one changes which grasps exist, the other is
evidence for a gate that ships off.

| | Config key | What it is | At run time |
| --- | --- | --- | --- |
| Geometry fusion | `fusion.geometry.enabled`, and `fusion.enabled` | per-object BASE clouds fused across cameras, fed to the generator | reaches the pick loop when both are on |
| Voxel substrate | `fusion.enabled` | a bounded BASE occupancy grid, accumulated within one pick | read only by the commit gate |

Both ship `false`. The commit gate reads the grid through `SceneFusion.corridor_evidence` only when
`fusion.commit_policy.enabled` is true, which it is not by default; with the gate off the grid is
written and never consulted.

The two switches are not independent. `fusion.enabled` is also what builds the other cameras' CAMERA to
BASE resolvers, so geometry fusion with `fusion.enabled` off drops every second view at the pick with a
warning and plans single-view. Turn both on. `CameraFusionPlan.from_tree(tree)`
(`src/robot/execution/camera_fusion.py`) reads a tree without opening anything and says in one sentence
what a cell is missing to fuse; [`examples/real_robot/17_pick_with_fused_cameras.py`](../../../../examples/real_robot/17_pick_with_fused_cameras.py)
asks it before it builds the cell.

## Which blob is the same object

Three metrics, and which to use is a measurement rather than a preference. `CENTROID` is cheapest and
degrades in a dense pile where neighbouring centres sit closer than the localisation error. `BOX_IOU`
uses extent as well, so a small part against a large one stays separable. `OVERLAP` is the share of the
target's points with a candidate point within `neighbour_mm` (12 mm by default), the only metric that
uses the surfaces themselves.

`OVERLAP` is the default because it abstains. `CENTROID` scores better on the cases that have an answer,
but when the target is not in the other view it takes a neighbour and welds that neighbour's far surface
into the cloud the grasp is planned on, and the generator cannot tell that cloud from a real one. A
camera that cannot see the target should contribute nothing, and below `min_score` (0.30) it does not.
A whole view is assigned globally with `linear_sum_assignment`, so one object in another view is never
claimed by two primary objects.

A view whose transform is unusable, or which segmented nothing, contributes nothing and is absent from
`views_used`, the record of what the fusion was built from.

## Switches that stay off, and why

- `fusion.geometry.neighbour_scene_enabled` also hands the collision filter every other camera's view of
  everything that is not the target. In a bin the finger collisions left after ranking are mostly the
  bin wall, and no detector segments a wall, so fusion cannot supply it. Declare the walls under
  `robot.grasping.support.container` instead ([`collision/`](../collision/README.md)). In a loose pile,
  where the blocker is another part, the switch works as designed.
- `fusion.geometry.promote_unmatched` lets a camera introduce an object the primary did not see, instead
  of only confirming one it did. The cell then holds one calculator per camera. How often a bin holds a
  part only a second camera sees is not measured, so it ships off.

## Configuring a two-camera cell

Read [`robot.eth2.yaml`](../../../../config/robot/robot.eth2.yaml) and
[`cam.eth2.yaml`](../../../../config/camera/cam.eth2.yaml) before wiring one; they carry the bring-up
order. Each camera's calibration is declared on its rig, `camera.cameras.rigs[<id>].extrinsics`, and the
calibration runner prints that block for each camera. `fusion.cameras` names which rigs are fused, keyed
by rig id. List the primary too: the rig builder and the list of cameras a pick waits for both leave it
out by `camera.cameras.primary_rig_id`, and listing it keeps the single-view warning accurate.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `ConfigError` at load | an id in `fusion.cameras` names no rig | use the rig ids from `camera.cameras.rigs` |
| refused at the build | a fused camera's rig declares no calibration, or its artifact does not load | calibrate it and declare the block |
| `RuntimeError` in the pick | a configured camera delivered no frame and `on_camera_unavailable` is `refuse` | fix the camera, or choose `degrade` |
| a WARNING, the view dropped | `degrade` and a missing camera, or a camera with no CAMERA to BASE resolver | read the log line; it names the camera |
| `IngestResult` refused | `SceneFusion.ingest` on a wrong frame, a bad lens, a changed lens, or no valid depth | the reason is on the result |

## Status

| Capability | Evidence |
| --- | --- |
| Geometry fusion across fixed cameras | measured in simulation: `run_multiview_pick`; top-1 43.50 % single-view, 55.93 % fused on the datagen reference |
| Centroid fusion and the fused-cloud grasp | measured in simulation: called by `run_multiview_pick`, not by the pick loop |
| The commit gate on the voxel grid | measured in simulation: `run_commit_gate` shows the evidence grows with distinct views and stays flat on a repeated one |
| `promote_unmatched` | never touched hardware: unit-tested, not measured |
| A physical multi-camera cell | never touched hardware: none has been built with this code |

## Files

| File | Holds |
| --- | --- |
| `association.py` | `AssociationMetric`, `associate_target`, `assign_view`, `cluster_views`, `fuse_target_cloud`, `fuse_scene_clouds` |
| `scene_geometry.py` | `ObservedView`, `fuse_scene_geometry`, `FusedSceneGeometry`, `build_scene_objects`, `SceneObject`, `to_base_mm` |
| `fusion.py` | `SceneFusion` and `FusionConfig`, the voxel substrate |
| `localize.py` | `fuse_view_localizations`, `ViewLocalization`: a visibility-weighted BASE centroid |
| `synthesis.py` | `synthesize_grasp_from_cloud`, `FusedGrasp`: a top-down grasp from the fused footprint, `None` below 8 points |
| `_fusion_geometry.py`, `_fusion_queries.py` | the ingest loop and the read side of the grid |

## Details

- [`loop/`](../loop/README.md) observes the rig and applies the camera policy;
  [`generation/`](../generation/README.md) receives the fused clouds.
- [Calibration](../../../calibration/README.md) for one calibration per camera, declared on its rig.
- [The grasping config reference](../../../../docs/grasping-config-reference.md) for `fusion` and
  `fusion.geometry`, and [guide 01](../../../../docs/guide/01-configuration.md) for the `eth2` profile.
- Tests: `tests/test_multiview_association.py`, `tests/test_pick_loop_fusion_geometry.py`,
  `tests/test_scene_objects.py`, `tests/test_promoted_objects_reach_the_pick.py`,
  `tests/test_u5_multi_view_fusion.py`, `tests/test_fusion_camera_map_means_every_camera.py`.
