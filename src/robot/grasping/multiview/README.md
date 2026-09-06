# Multi-view (`src.robot.grasping.multiview`)

More cameras rather than more time. One depth view observes one side of an object and an antipodal
grasp needs two, so a contact face no view proposed is one no ranker can recover. This package
decides which blob in another camera is the same object, fuses the surfaces that agree, and
accumulates a bounded occupancy grid alongside.

## Two different things are called fusion

They share a config prefix and they are not the same feature. Confusing them is easy and expensive:
one is telemetry, the other changes which grasps exist.

| | Config key | What it is | Runtime status |
| --- | --- | --- | --- |
| Voxel substrate | `grasping.fusion.enabled` | a bounded BASE-frame occupancy grid accumulated across attempts | read by one consumer, the commit gate, which is itself off |
| Geometry fusion | `grasping.fusion.geometry.enabled` | per-object BASE clouds fused across cameras, fed to the generator | reaches the core pick loop when enabled |

Both ship `false`.

## What it guarantees

Pure NumPy, plus the deterministic `linear_sum_assignment` of SciPy for the assignment. Frames are
BASE millimetres throughout. Reductions are bit-stable: the ingest arithmetic is behaviour-locked, so
the same input produces the same counters.

The scope is deliberately narrow. These are pure functions over already-resolved inputs: masks,
depth, intrinsics and one CAMERA to BASE matrix per camera. Resolving those, and deciding what to do
when a camera is missing, is policy, and policy lives with the config in the orchestrator. No Isaac,
no torch, no camera objects, no perception models.

## The public surface

| Module | Owns |
| --- | --- |
| `association.py` | `AssociationMetric`, `ViewCandidates`, `associate_target`, `assign_view`, `fuse_target_cloud`, `fuse_scene_clouds`: is this blob in another camera the same object |
| `scene_geometry.py` | `fuse_scene_geometry`, `ObservedView`, `FusedSceneGeometry`, `to_base_mm`: one BASE cloud per candidate object, carrying the sides one view cannot see |
| `localize.py` | `fuse_view_localizations`, `fuse_scene_points_base`, `ViewLocalization`: visibility-weighted BASE centroid fusion |
| `synthesis.py` | `synthesize_grasp_from_cloud`, `fused_grasp_to_point`, `FusedGrasp`: a grasp derived from the fused cloud rather than from one silhouette |
| `fusion.py` | `SceneFusion` and `FusionConfig`: the bounded voxel-occupancy substrate |
| `_fusion_geometry.py`, `_fusion_queries.py` | the ingest hot loop and the read side |

### Association

Three metrics, and which one to use is a measurement rather than a preference. `CENTROID` is the
cheapest and degrades in a dense pile where neighbouring centres sit closer together than the
localisation error. `BOX_IOU` uses extent as well as position, so a small object nested against a
large one stays separable. `OVERLAP` is the fraction of the target points that have a candidate point
within `neighbour_mm`, the only metric that uses the surfaces themselves.

`OVERLAP` is the default, and the reason is abstention rather than raw accuracy. On the answerable
half of a decision set `CENTROID` scores better, but it never abstains: when the target is simply not
in the other view it takes a neighbour instead, and welds a neighbour far surface into the cloud the
grasp is planned on. The generator cannot tell that cloud from a real one. A camera that cannot see
the target should contribute nothing, and only `OVERLAP` does. `DEFAULT_MIN_SCORE` is 0.30.

Assignment across a whole view is solved globally with `linear_sum_assignment`, not greedily, so one
object in another view is never claimed by two primary objects.

Voxel decimation, applied once per cloud and only for `OVERLAP`, keeps one real observed point per
cell: `key(p) = floor(p / voxel_mm)`, first index of each unique key. Not a cell centroid, because a
centroid is a point that was never observed and `OVERLAP` is a statement about observed points.

### Geometry fusion

`fuse_scene_geometry` takes the primary view as its first four arguments and every other camera as
`other_views`, and returns one fused cloud per primary object in the segmentation order of the
primary. A view whose transform is unusable, or which segmented nothing, contributes nothing and is
absent from `views_used`, the honest record of what the fusion was built from.

`with_neighbours` additionally returns, per object, the view every other camera has of everything
that is not that object, which is the obstacle side of the same observation. It is off by default: it
costs a second pass, and a caller that only feeds the generator has no use for it. The config key is
`fusion.geometry.neighbour_scene_enabled`, and it stays off for a second reason. In a bin, the finger
collisions that survive ranking are dominated by the bin wall rather than by a neighbouring part, and
a wall is not an instance any detector segments, so no amount of cross-camera fusion can supply one.
Wall geometry has to be declared instead, under `robot.grasping.support.container`. Where the
blocker really is another object, as in a loose pile, the switch works as designed.

### The voxel substrate

`SceneFusion` ingests per-attempt depth captures into a bounded BASE-frame occupancy grid, tracking
per-voxel `hits`, meaning a depth sample landed there, and `seen`, meaning how many views touched it,
as saturating `uint16` accumulators. It enforces a strict frame and intrinsics policy so a
misconfigured camera cannot silently poison the grid.

Exactly one thing reads it: the commit gate in the pick loop, through `corridor_evidence`, and only
when `fusion.commit_policy.enabled` is true, which it is not by default. With that gate off the grid
is written and never consulted, so no grasp, ranking, decision or recovery path changes. Telemetry,
replay and debug overlays are welcome readers.

### Localization and synthesis

`fuse_view_localizations` takes the BASE estimate each active camera has of the target centroid, plus
how many target pixels it saw, and fuses them into one BASE point. A camera that did not see the
target reports `centroid_base_mm=None` and `visible_px=0` and contributes nothing, so the fused point
is carried by whichever cameras did. That is the redundancy that lets a fixed oblique resolve a
target the overhead loses, which matters because in an arm-over-bench cell the arm is the dominant
occluder and two opposed views mean it cannot hide the scene from both.

`synthesize_grasp_from_cloud` derives the closing axis and grip width from the true horizontal
footprint over all views, by PCA on the BASE XY projection, closing across the minor extent, rather
than from one foreshortened silhouette. The position is the fused centroid at a grasp depth below the
top surface of the cloud, so symmetric views cancel the surface bias of each single view. It returns
`None` when the cloud has fewer than eight finite points, rather than fitting a footprint anyway.

These two modules have callers in the simulation runners, not in the core pick loop. What the loop
uses is `fuse_scene_geometry`.

## Configuring a two-camera cell

The worked example is `config/robot/robot.eth2.yaml`, loaded as `WILLY_PROFILE=ur5e,eth2`, with its
camera inventory in `config/camera/cam.eth2.yaml`. Read that file before wiring one: it carries the
bring-up order and the two traps below. No physical multi-camera cell has been built by this code.

`fusion.extrinsics_artifact_path` is the calibration of the primary camera, and it is the only key
that satisfies the CAMERA to BASE refusal in `AutonomousGraspService.from_robot_config`. It is easy
to miss, because the calibration runner prints a `fusion.cameras` block to paste after each camera
and that block alone leaves this key null. A cell can be fully calibrated, load both artifacts, and
still be refused at build. Point it at the artifact for the first `source: rgbd` rig in
`camera.cameras.rigs`.

The primary camera does not belong in `fusion.cameras`. That map is every camera except the primary,
one entry per camera with its own artifact. List the primary there and two readers disagree about it:
the rig builder filters the primary out, because it already streams through the main perception
source, while `configured_camera_ids` keeps every enabled entry and the loop then reports each
configured camera that delivered no frame. The primary is then permanently among the missing, which
is a warning on every pick under `on_camera_unavailable: degrade` and a raised error on every pick
under `refuse`. The simulation profile lists all three of its cameras including its primary, and that
is correct there, because the simulation runner builds a rig that includes the primary. Do not copy
the shape of that block. The cost of leaving the primary out is one false line at construction: the
counter that warns about a single-view cell reads the same map, so it reports one calibrated camera
and recommends adding a second while the cell is fusing two.

## See also

- [`../README.md`](../README.md) for the tier this feeds
- [`../generation/README.md`](../generation/README.md) for the seam these clouds arrive through
- [`../loop/README.md`](../loop/README.md) for the orchestrator that observes the rig and applies the policy
- [`../collision/README.md`](../collision/README.md) for the declared container walls that fusion cannot supply
- [`../../../calibration/README.md`](../../../calibration/README.md) for the one calibration per camera
