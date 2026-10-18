# Multi-camera fusion (`src.robot.grasping.multiview`)

One depth view sees one side of a part, and a parallel-jaw grasp needs two opposite faces. This package
decides which blob in another view is the same object and fuses the surfaces that agree into one BASE
cloud per object, so the grasp generator sees sides one view cannot.

The other views come from two places. **Fixed cameras** fuse without moving anything: each sees the cell
from where it is mounted. **A wrist camera** gets its other views from the arm: a pick handed looks
visits them and fuses every look with the ones before ([A wrist camera's looks](#a-wrist-cameras-looks)).
Multi-view with arm motion exists only with an eye-in-hand camera.

You reach it through your cell's camera section and the pick loop; there is nothing to call and no
switch. A cell fuses its fixed cameras wherever `camera.cameras.rigs` declares two or more enabled RGB-D
rigs with their calibration, the primary among them, as the shipped two-camera reference
[`config/camera/cam.eth2.yaml`](../../../../config/camera/cam.eth2.yaml) does (the profile chain
`<your arm>,eth2`):

```yaml
camera:
  cameras:
    primary_rig_id: cam_left          # the grasp is synthesised in its view
    rigs:
      - rig_id: cam_left
        enabled: true
        source: rgbd
        extrinsics: { mounting_mode: eye_to_hand, artifact_path: ../calibration/real/eth_cam_left.json }
      - rig_id: cam_right
        enabled: true
        source: rgbd
        extrinsics: { mounting_mode: eye_to_hand, artifact_path: ../calibration/real/eth_cam_right.json }
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

## Which cameras fuse

Every enabled RGB-D rig that declares its calibration, `camera.cameras.rigs[<id>].extrinsics`, where two
or more do and the primary is one of them (`camera_fusion.fused_cameras`). One camera has nothing to fuse
with. A camera that is not to take part is switched off on its rig. The numbers are the ones every fused
cell ran before the fusion keys left on 2026-10-16: there is nothing to set.

`CameraFusionPlan.from_tree(tree)` (`src/robot/execution/camera_fusion.py`) reads a tree without opening
anything and says which cameras it fuses, or in one sentence what it is missing to fuse any;
[`examples/real_robot/17_pick_with_fused_cameras.py`](../../../../examples/real_robot/17_pick_with_fused_cameras.py)
asks it before it builds the cell. The cell says the same at boot: one INFO line with what it fuses, and
a WARNING where two or more enabled RGB-D rigs are declared and they do not fuse.

A wrist camera's looks fuse whatever the rigs are: a pick handed looks fuses them. The rigs decide only
whether fixed cameras join them.

## Which blob is the same object

The share of the target's points with a point of the candidate within 12 mm (`DEFAULT_NEIGHBOUR_MM`):
the overlap of the two surfaces. Below 0.30 (`DEFAULT_MIN_SCORE`) a camera contributes nothing for that
object, exactly as an occluded camera does.

The overlap is used because it abstains. A centroid distance scores better on the cases that have an
answer, but when the target is not in the other view it takes a neighbour and welds that neighbour's far
surface into the cloud the grasp is planned on, and the generator cannot tell that cloud from a real one.
The centroid and box-IoU metrics left on 2026-10-16: no tree chose one. A whole view is assigned globally
with `linear_sum_assignment`, so one object in another view is never claimed by two primary objects.

A view whose transform is unusable, or which segmented nothing, contributes nothing and is absent from
`views_used`, the record of what the fusion was built from.

The fixed cameras hand the collision filter no neighbour clouds. In a bin the finger collisions left
after ranking are mostly the bin wall, and no detector segments a wall, so fusion cannot supply it:
declare the walls under `robot.grasping.support.container` instead ([`collision/`](../collision/README.md)).
Over 1073 reference objects the fixed cameras' neighbours rescued 5 grasps and destroyed 4; the switch
that handed them over left on 2026-10-16. A wrist pick's earlier looks still hand theirs.

## A wrist camera's looks

The pick loop fuses each look of a wrist pick with every earlier look, and with the fixed cameras where
the cell fuses them, acquired once per pick ([loop/](../loop/README.md)). `Locator.look_around` fuses its
looks the same way ([perception/](../../perception/README.md)). What this package adds to that:

| Rule | Where | What it does |
| --- | --- | --- |
| **Wrist tolerances** | `association.WRIST_VIEW_*` | two looks are associated by overlap at a score of 0.30, with a 15 mm neighbour radius and 3 mm scoring cells: two views through one hand-eye disagree more the more the wrist turned. A first look fused with fixed cameras alone is associated at the fixed cameras' 0.30 and 12 mm |
| **Clean surfaces** | `scene_geometry.grazing_pixels` | before a look is fused, its masks lose the pixels behind a depth step (the Locator's rule) and the ones that see their surface past 75 deg of incidence (`GRAZING_INCIDENCE_DEG`, the normal measured 2 px either way, `GRAZING_BASELINE_PX`): a grazing point says more about the edge it was seen past than about the surface |
| **A footprint less the rim** | `fuse_scene_geometry(primary_footprints=...)` | where the cell cuts a rim off the support-footprint stage's input (`footprint_rim.FOOTPRINT_RIM_MM`, 2026-10-09), each view carries its masks less their rim beside them (`ObservedView.footprints`), and the fusion builds every fused object's cloud again from them by the association the clouds made (`FusedSceneGeometry.footprint_clouds_base_mm`, `footprint_for`); a view with none gives its masks. The clouds, the associations and the neighbours are the same with footprints as without, and only the stage's footprint reads them |
| **One name per look** | the pick loop | a look is the view `rig@look`, so two looks of one camera are two views; a pick handed no looks keeps the rig id |
| **Label agreement** | `association.label_agreement` | looks that call the associated part by different labels make its grasp uncertain; looks that agree change nothing. One rule for the pick loop and the Locator, and one sentence for both accounts, `label_agreement_said`: `label agreed in N of M looks`, M the looks that saw the part, N those that call it the judged look's label |
| **Jaw contact faces** | `unseen_side.jaw_faces_seen` | a face counts as seen when at least 30 % of the pad's touchable 4 mm cells, and at least 3 cells, hold 3 or more points within 6 mm of the face |
| **The generated view** | `unseen_side.orbit_views`, `unseen_side.other_angles` | the judged look turned about the vertical through the part, smallest turn first, up to 180 deg either way (`ORBIT_MAX_DEG`), a face counting as shown within 60 deg of incidence, at most 3 motions (`ORBIT_MAX_MOTIONS`); where no turn can be driven, its other angles (the owner, 2026-10-16): lowered 15 and 30 deg about the part or raised 15, at 0.85 and 1.15 of its distance, each with every turn, nearest the judged look first, between 15 and 80 deg over the table; the safeguards past 120 deg live in [`execution/generated_view.py`](../../execution/generated_view.py) |

**The hand-eye check** compares two looks where they saw the same surface
(`association.nearest_surface_distances_mm`): each point's nearest point of the other look within 15 mm,
only where the two surfaces face the same way (normals within 30 deg, each turned toward its own
camera, `SHARED_SURFACE_*`). The pick measures the median once, on the judgement its looking ends on,
from 20 shared distances or more (`HAND_EYE_MIN_POINTS`). Above 6 mm (`HAND_EYE_DRIFT_WARN_MM`) a WARNING
says the hand-eye calibration may have drifted, a camera loosened on the wrist; nothing else is done. A
healthy cell reads a millimetre or two. Depth noise reads a drift short: in a probe, a 7 mm drift read
about 6 mm at 1 mm of noise and about 5 mm at 2 mm, under the warning. A slide along a surface stays
invisible to it.

## Configuring a two-camera cell

Read [`cam.eth2.yaml`](../../../../config/camera/cam.eth2.yaml) before wiring one; its header carries the
bring-up order. Each camera's calibration is declared on its rig, `camera.cameras.rigs[<id>].extrinsics`,
and the calibration runner prints that block for each camera. Two enabled RGB-D rigs need a serial each
(the schema refuses them without one), or two identical cameras can swap identity after a replug and
mirror every fused position.

## What it refuses and warns of

| What | When | What to do |
| --- | --- | --- |
| refused at the build | a fused camera's artifact does not load | calibrate it and fix the block |
| a WARNING at boot | two or more enabled RGB-D rigs are declared and they do not fuse (one has no calibration, the primary is not among them) | read the line; it names what is missing |
| a WARNING, the pick goes on | a fused camera delivered no frame, with the reason the rig recorded | fix the camera; the others are still fused |
| a WARNING, the view dropped | a camera with no CAMERA to BASE resolver, or one that resolved none this frame | read the log line; it names the camera |

## Status

| Capability | Evidence |
| --- | --- |
| Geometry fusion across fixed cameras | measured in simulation: `run_multiview_pick`; top-1 43.50 % single-view, 55.93 % fused on the datagen reference |
| Centroid fusion | measured in simulation: called by `run_multiview_pick`, not by the pick loop |
| A physical multi-camera cell | never touched hardware: none has been built with this code |
| A wrist camera's looks, the faces check and the hand-eye check | pinned by tests on synthetic clouds and fake arms; the wrist tolerances are argued from the fixed-camera sweep and a hand-eye's error budget, not measured on a real cell; never touched hardware |

## Files

| File | Holds |
| --- | --- |
| `association.py` | `associate_target`, `assign_view`, `fuse_scene_clouds`; `label_agreement`, `label_agreement_said`, `nearest_surface_distances_mm`, `HAND_EYE_DRIFT_WARN_MM`, `HAND_EYE_MIN_POINTS`, the `DEFAULT_*` and the `WRIST_VIEW_*` tolerances |
| `scene_geometry.py` | `ObservedView` (with its `footprints`), `fuse_scene_geometry`, `FusedSceneGeometry` (with its `footprint_clouds_base_mm`), `build_scene_objects`, `SceneObject`, `to_base_mm`, `grazing_pixels` |
| `localize.py` | `fuse_view_localizations`, `ViewLocalization`: a visibility-weighted BASE centroid |
| `unseen_side.py` | `jaw_faces_seen`, `orbit_views`, `other_angles`: were both jaw contact faces seen, the turns of the judged look that would show the one that was not, up to 180 deg, and its other angles where no turn can be driven |

`synthesis.py` (`synthesize_grasp_from_cloud`, a top-down grasp from the fused footprint) was removed on
2026-09-29 with the two `run_multiview_pick` levers that called it, `--synthesize-3d` and `--closed-loop`:
a fused cell's grasp comes from the generator fed the fused cloud. On 2026-10-16 the fusion keys left, and
with them a camera introducing an object the primary did not see (`promote_unmatched`, with
`cluster_views` and a calculator per camera), never switched on in a shipped tree.

## Details

- [`loop/`](../loop/README.md) observes the rig, warns of a missing camera and runs a wrist pick's looks;
  [`generation/`](../generation/README.md) receives the fused clouds.
- [Calibration](../../../calibration/README.md) for one calibration per camera, declared on its rig.
- [Guide 01](../../../../docs/guide/01-configuration.md) for the `eth2` profile.
- Tests: `tests/robot/grasping/multiview/test_multiview_association.py`, `tests/robot/grasping/types/test_pick_loop_fusion_geometry.py`,
  `tests/robot/grasping/multiview/test_scene_objects.py`, `tests/robot/execution/autonomous_grasp/test_a_fused_pick_waits_for_every_camera_but_the_primary.py`,
  `tests/robot/test_a_pick_says_which_cameras_it_fused.py`, `tests/robot/grasping/test_both_jaw_faces_are_seen.py`,
  `tests/robot/grasping/multiview/test_the_generated_view_orbits_to_the_unseen_face.py`,
  `tests/robot/grasping/multiview/test_the_generated_view_tries_other_angles_where_no_turn_can_be_driven.py`,
  `tests/robot/grasping/types/test_a_wrist_pick_looks_until_its_grasp_is_safe.py`.
