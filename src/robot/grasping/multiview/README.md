# Multi-camera fusion (`src.robot.grasping.multiview`)

One depth view sees one side of a part, and a parallel-jaw grasp needs two opposite faces. This package
decides which blob in another view is the same object and fuses the surfaces that agree into one BASE
cloud per object, so the grasp generator sees sides one view cannot.

The other views come from two places. **Fixed cameras** fuse without moving anything: each sees the cell
from where it is mounted. **A wrist camera** gets its other views from the arm: a pick handed looks
visits them and fuses every look with the ones before ([A wrist camera's looks](#a-wrist-cameras-looks)).
Multi-view with arm motion exists only with an eye-in-hand camera.

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

## Two switches, one feature

| Config key | What it does |
| --- | --- |
| `fusion.enabled` | builds the CAMERA to BASE resolver of each camera `fusion.cameras` lists, from the calibration on its rig |
| `fusion.geometry.enabled` | fuses each object's surface across those cameras and feeds the union to the generator |

Both ship `false`, and fusing takes both. With `fusion.enabled` off the resolver map is empty, so every
second view is dropped at the pick with a warning and the cell plans single-view; with
`fusion.geometry.enabled` off nothing asks for the map. Turn both on. `CameraFusionPlan.from_tree(tree)`
(`src/robot/execution/camera_fusion.py`) reads a tree without opening anything and says in one sentence
what a cell is missing to fuse; [`examples/real_robot/17_pick_with_fused_cameras.py`](../../../../examples/real_robot/17_pick_with_fused_cameras.py)
asks it before it builds the cell.

A wrist camera's looks need neither switch: a pick handed looks fuses them whatever the two say. The
switches decide only whether the fixed cameras join them.

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

## A wrist camera's looks

The pick loop fuses each look of a wrist pick with every earlier look, and with the fixed cameras where
the cell fuses them, acquired once per pick ([loop/](../loop/README.md)). `Locator.look_around` fuses its
looks the same way ([perception/](../../perception/README.md)). What this package adds to that:

| Rule | Where | What it does |
| --- | --- | --- |
| **Wrist tolerances** | `association.WRIST_VIEW_*` | two looks are associated by overlap at a score of 0.30, with a 15 mm neighbour radius and 3 mm scoring cells: two views through one hand-eye disagree more the more the wrist turned. A cell that fuses fixed cameras associates with its own `fusion.geometry` metric and `min_score`, and the wider radius and cells of the two |
| **Clean surfaces** | `scene_geometry.grazing_pixels` | before a look is fused, its masks lose the pixels behind a depth step (the Locator's rule) and the ones that see their surface past 75 deg of incidence (`GRAZING_INCIDENCE_DEG`, the normal measured 2 px either way, `GRAZING_BASELINE_PX`): a grazing point says more about the edge it was seen past than about the surface |
| **A footprint less the rim** | `fuse_scene_geometry(primary_footprints=...)` | where the cell cuts a rim off the support-footprint stage's input (`robot.grasping.geometry.footprint_rim_mm`, 2026-10-09), each view carries its masks less their rim beside them (`ObservedView.footprints`), and the fusion builds every fused object's cloud again from them by the association the clouds made (`FusedSceneGeometry.footprint_clouds_base_mm`, `footprint_for`); a view with none gives its masks. The clouds, the associations and the neighbours are the same with footprints as without, and only the stage's footprint reads them |
| **One name per look** | the pick loop | a look is the view `rig@look`, so two looks of one camera are two views; a pick handed no looks keeps the rig id |
| **Label agreement** | `association.label_agreement` | looks that call the associated part by different labels make its grasp uncertain; looks that agree change nothing. One rule for the pick loop and the Locator, and one sentence for both accounts, `label_agreement_said`: `label agreed in N of M looks`, M the looks that saw the part, N those that call it the judged look's label |
| **Jaw contact faces** | `unseen_side.jaw_faces_seen` | a face counts as seen when at least 30 % of the pad's touchable 4 mm cells, and at least 3 cells, hold 3 or more points within 6 mm of the face |
| **The generated view** | `unseen_side.orbit_views` | the judged look turned about the vertical through the part, smallest turn first, up to 180 deg either way (`ORBIT_MAX_DEG`), a face counting as shown within 60 deg of incidence, at most 3 motions (`ORBIT_MAX_MOTIONS`); the safeguards past 120 deg live in [`execution/generated_view.py`](../../execution/generated_view.py) |

**The hand-eye check** compares two looks where they saw the same surface
(`association.nearest_surface_distances_mm`): each point's nearest point of the other look within 15 mm,
only where the two surfaces face the same way (normals within 30 deg, each turned toward its own
camera, `SHARED_SURFACE_*`). The pick measures the median once, on the judgement its looking ends on,
from 20 shared distances or more (`HAND_EYE_MIN_POINTS`). Above 6 mm (`HAND_EYE_DRIFT_WARN_MM`) a WARNING
says the hand-eye calibration may have drifted, a camera loosened on the wrist; nothing else is done. A
healthy cell reads a millimetre or two. Depth noise reads a drift short: in a probe, a 7 mm drift read
about 6 mm at 1 mm of noise and about 5 mm at 2 mm, under the warning. A slide along a surface stays
invisible to it.

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

## Status

| Capability | Evidence |
| --- | --- |
| Geometry fusion across fixed cameras | measured in simulation: `run_multiview_pick`; top-1 43.50 % single-view, 55.93 % fused on the datagen reference |
| Centroid fusion | measured in simulation: called by `run_multiview_pick`, not by the pick loop |
| `promote_unmatched` | never touched hardware: unit-tested, not measured |
| A physical multi-camera cell | never touched hardware: none has been built with this code |
| A wrist camera's looks, the faces check and the hand-eye check | pinned by tests on synthetic clouds and fake arms; the wrist tolerances are argued from the fixed-camera sweep and a hand-eye's error budget, not measured on a real cell; never touched hardware |

## Files

| File | Holds |
| --- | --- |
| `association.py` | `AssociationMetric`, `associate_target`, `assign_view`, `cluster_views`, `fuse_target_cloud`, `fuse_scene_clouds`; `label_agreement`, `label_agreement_said`, `nearest_surface_distances_mm`, `HAND_EYE_DRIFT_WARN_MM`, `HAND_EYE_MIN_POINTS` and the `WRIST_VIEW_*` tolerances |
| `scene_geometry.py` | `ObservedView` (with its `footprints`), `fuse_scene_geometry`, `FusedSceneGeometry` (with its `footprint_clouds_base_mm`), `build_scene_objects`, `SceneObject`, `to_base_mm`, `grazing_pixels` |
| `localize.py` | `fuse_view_localizations`, `ViewLocalization`: a visibility-weighted BASE centroid |
| `unseen_side.py` | `jaw_faces_seen`, `orbit_views`: were both jaw contact faces seen, and the turns of the judged look that would show the one that was not, up to 180 deg |

`synthesis.py` (`synthesize_grasp_from_cloud`, a top-down grasp from the fused footprint) was removed on
2026-09-29 with the two `run_multiview_pick` levers that called it, `--synthesize-3d` and `--closed-loop`:
a fused cell's grasp comes from the generator fed the fused cloud.

## Details

- [`loop/`](../loop/README.md) observes the rig, applies the camera policy and runs a wrist pick's looks;
  [`generation/`](../generation/README.md) receives the fused clouds.
- [Calibration](../../../calibration/README.md) for one calibration per camera, declared on its rig.
- [The grasping config reference](../../../../docs/grasping-config-reference.md) for `fusion` and
  `fusion.geometry`, and [guide 01](../../../../docs/guide/01-configuration.md) for the `eth2` profile.
- Tests: `tests/test_multiview_association.py`, `tests/test_pick_loop_fusion_geometry.py`,
  `tests/test_scene_objects.py`, `tests/test_promoted_objects_reach_the_pick.py`,
  `tests/test_fusion_camera_map_means_every_camera.py`, `tests/test_both_jaw_faces_are_seen.py`,
  `tests/test_the_generated_view_orbits_to_the_unseen_face.py`,
  `tests/test_a_wrist_pick_looks_until_its_grasp_is_safe.py`.
