# Where grasp candidates come from (`src/robot/grasping/generation`)

`GraspCalculator`, the analytic grasp generator: a segmentation mask and a depth image go in, ranked
`GraspPoint` candidates come out. Pure numpy and OpenCV, with no robot, no torch and no vendor SDK. A
cell builds it through `build_calculator`, which reads `robot.grasping.calculator`; build it the same
way when you call it directly.

```python
from willy import Frame, load_tree
from src.robot.grasping import SupportPlane
from src.robot.grasping.calculator_factory import build_calculator

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

The default geometry stage reconstructs the target as a vertical prism, its visible footprint extruded down to the
support plane, and enumerates the grasps that prism admits: the two axes of the footprint's minimum-area rectangle,
plus a radial fan when the footprint is round. A round footprint's fan stands in the base frame, every 30 degrees,
and its closing lines pass through its centroid (`SupportPrism.middle`): the rectangle about a round outline turns
with the noise on it, and on a 40 mm cylinder read in its visual hull it turned 15 degrees and lay 0.64 mm off the
axis, enough to lose the one grasp that fits between two cubes 16 to 18 mm off it (2026-10-09). For each it solves
for the height at which the fingers still clear the table. With side approaches the best grasp straight down always
stays among those kept (`_with_a_vertical`): a tilt the arm cannot take is often refused with every other tilt of its
line, and the vertical one is what the pick loop then tries. A filter can only refuse a grasp that goes into the
table; this stage plans the clearance instead. A cell sets it from `robot.grasping.geometry.stage`
(`support_footprint`, the default); a direct caller passes `support_footprint_geometry=`, default `True`. The jaw
dimensions come from `ParallelJawGripperModel`, the one measurement of the jaw every consumer reads.

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

A fourth is the rim, 2.0 mm by default (`robot.grasping.geometry.footprint_rim_mm`, on since the owner's evening of
2026-10-09; `GraspCalculator(support_footprint_rim_mm=...)`, from `build_calculator`; a direct caller's default is
0.0). The D415 smears a part's far edge into a ramp of depths, the outer ~3 px of the colour mask lie on it, the
depth-step rule keeps it, and the hull follows it: on 23 grey cubes the owner's cell recorded, the centre the closing
line follows lay 2.53 mm off at the median, 2.5 mm of it away from the camera. With the rim, the mask loses a rim of
that many millimetres at the part for the stage's input alone, `ceil(rim * fx / z)` px at its median depth and never
more than 30 % of the mask ([footprint_rim.py](footprint_rim.py)). Through the calculator on the same cubes, the rim
with `inflate_mm` 1.25 left the centre 0.77 mm off at the median and the footprint's sides within 0.8 mm; the visual
hull below gives the faces back instead, and `inflate_mm` stays 0.0. A fused cloud comes with its rim cut where each
look was seen (`compute(footprint_points_base_mm=...)`, the pick loop's looks), and the calculator reads it beside
the fused cloud for the footprint alone. `support_footprint_rim_px` and `support_footprint_rim_points` in the
telemetry, and a `footprint rim:` line in the log every compute, say what the rim took; where it would leave the
stage no point, nothing is cut. Everything else the calculator reads, the neighbours' growth, the scene, the target's
cloud and the support under it, keeps the whole mask.

With the rim comes the part's visual hull ([footprint_hull.py](footprint_hull.py)): every mask pixel the stage does
not read, the rim, the depth steps and the holes the D415 leaves on dark or shiny surfaces, gets a point where its ray
meets the part's top, or just over its support, wherever that point's vertical, from 1 mm over the support to 1 mm
under the top, projects into the mask; a pixel with no such place gives none, so the footprint never grows past what
the mask shows. `inflate_mm` stays 0.0: the hull gives the faces back. On the same 23 cubes the centre lay 0.58 mm off
at the median and 1.79 at the 90th percentile (the rim with `inflate_mm` 1.25: 0.77 and 2.26), and with 35 % of the
parts' depth knocked out in blobs 0.59 and 1.87 (0.87 and 2.28); with 20 % the rim alone kept the better median, 0.77
against 1.02, the hull the better 90th percentile, 1.91 against 2.24. A fused footprint gets this look's points beside
the looks' own. The points reach the stage's input and nothing else, `support_footprint_hull_points` in the telemetry
and a `footprint hull:` line in the log say how many, and where the points read give no prism none is placed.

**Robust grasps first.** Where the search ranks what it found (`rank_found`), the grasps that still hold with the
hand 4 mm off across the table come first, each group in the order it had (`robust_to`, `ROBUST_ERROR_MM`): each open
finger stands 4 mm clear of the face it closes on, and the closing line, moved 4 mm either way across it, still meets
the part within the stroke and the friction cone. The owner, 2026-10-09: 3 to 5 mm is what a real cell gets; the two
looks of the one recorded pick seen from two poses lay 5.4 mm apart at the median. Among the grasps SFE finds valid,
the order by score alone moved the reference by +0.29 points without that error (`_SCORE_WEIGHTS`). A grasp says
whether it is robust in `score_components["support_footprint"]["robust"]`, and `support_footprint_robust` in the
telemetry counts the robust ones kept. The neighbours are not asked again: the guard keeps its distance from them at
the grasp. Measured on the desk (2026-10-09): on the held-out Isaac shard's 273 object views with a grasp ranked
first (a gripper that opens 150 mm) and on the 23 cubes the owner's cell recorded (the Hand-E's 49.99 mm) it changed
the grasp ranked first in none, the best grasps already holding 4 mm off; it reorders the ones after it, which the
pick loop takes where the guard refuses the first. The grasps are asked only until the robust ones fill the list: 1.7
ms a search on a cube.

Where the coarse grid finds fewer than three grasps, the stage searches again on a fine grid: the approach
every 7.5 degrees, each face's closing axis turned and rolled inside the friction cone. That second search
is most of what such a part costs, so the pick loop may hold it back for one ranking
(`GraspCalculator.sfe_fine_pass`, from `robot.grasping.fine_pass_waits`): the result then holds the coarse
grid's grasps alone and says `support_footprint_fine: "deferred"`, and the loop asks again in full before
it takes that part. Two things make a build cheaper without changing a candidate or a count: a closing
line wider than the hand closes on is refused whole, its builds counted under `aperture` as they were
counted one by one, and the open hand is measured only against the boxes of the neighbours it could come
near (`SeenEnvelope.refuses`).

A part's search is a set of independent units, one closing line at one height each, merged in the order the
stage runs them (`plan_support_footprint`, `run_units`, `rank_found`), so any process may run any unit and
the answer is the same, bit for bit, counts included. A cell with `robot.grasping.workers` hands the
calculator a warm pool of worker processes with every call (`compute(sfe_workers=...)`,
[workers.py](../workers.py)): the coarse units first, the fine and rolled ones only where the merged coarse
grasps ask for them. A part the pool cannot answer for is searched in the calculator's own process, from the
start, said in one log line. Three more things cost nothing in the answer: the floor the guard holds reads
its solids once (`HandFloor`), a build's cross products skip numpy's general path (`_cross3`), and where the
stage runs with its fallback off the dense samples and the geometry-first contacts it replaces are not
computed (`telemetry["geometry_skipped"]`).

With `robot.grasping.batched_builds` (off by default) each closing line's builds are made at once, in one
numpy pass per check (`_build_many`, `GraspCalculator.sfe_batched`), in the calculator's process and in every
worker alike: `_build`'s checks in its order over every build still standing, the tilt ladder a rung at a
time so that no build is made the loop would have skipped. Every number a candidate carries is made one
build at a time through a stacked `np.matmul`, the BLAS call one build makes, so the candidates, the counts
and the stages are the same to the bit; a number only a verdict reads, made over many builds' points at once
(the floor, the open fingers in the part, the corridor's pixels and depth, the camera's boxes), is held 1e-6
off its threshold, and a build nearer one is made by `_build` alone. The camera's boxes are measured only
where a face's gap does not already keep them apart and the hand's point nearest a box does not already
stand inside the limit. The recorded boxed-in Zollstock took 0.22 s instead of 2.2 s on the desk, a
boxed-in cube 0.24 s instead of 3.7 s (2026-10-09). `build_calculator` asks this machine's numpy once whether its stacked products answer to the
bit (`batched_builds_hold`) and says the answer in the calculator's log; where they would not, or a pass
cannot be made, the builds are made one at a time and the log says why. `scripts/checks/batched_builds.py`
compares the two on a cell's own PC before the key goes on there.

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

**Which way round is the pick loop's.** The calculator ranks each grasp as generated. A program's
`closing_axis` and the cell's `robot.natural_closing_axis` choose the way round afterwards, in the pick
loop ([loop/](../loop/README.md)); while an axis is named the loop raises `max_candidates` to 36 for that
ranking and gives it back. So an IK service or a feasibility re-rank wired into the calculator judges each
candidate before the loop turns it half a turn; no builder wires one today, and the arm's own guards judge
the grasp executed. Where the loop changed a result it redraws the overlay (`redraw_debug_image`).

## What it sees beside the part

With `grasping.scene_obstacles` on (a cell's default) the calculator holds every object the camera saw beside the
part as an obstacle, named by a prompt or not (`scene_obstacles.py`): the points within 250 mm of the part's mask,
grown about 5 mm and trimmed at depth steps, less what the part stands on (`SupportModel.is_support_point`, else the
bench and its band) and the declared bodies' bands, in 25 mm voxels of at least 12 points. They join the
support-footprint stage's obstacle points, and a post-filter judges the open hand against the support solids at the
guard's distance. The open hand keeps the guard's distance from the boxes the camera world builds of the same frame
(`SceneObstacles.boxes`): of those points, and of every point the world itself keeps of the frame, thinned in its
order (in each 10 mm voxel of the camera's frame the first pixel no support holds, unless it lies in a declared
body's band or the box the world keeps out for the part). Thinned after the 5 mm rim and the depth steps were taken
out alone, the side of a cube 15 mm off a cylinder lost its face's pixels and its box came 2.9 mm short of the
guard's (2026-10-09); now both start at the same millimetre. The failure reasons then follow the stage's refusal
counts by obstacle set: a neighbour the camera saw is `ALL_COLLIDED`, the support or a declared body
`ALL_TABLE_CONFLICT` ("too short for this hand" under about 28 mm on a Hand-E), the part's own fragments
`NO_VALID_GRASP`, with `why_no_grasp`'s sentence in the telemetry (`no_grasp_said`) and the log. With
`side_approaches` on, the stage offers every tilt the hand fits at, scored by the room each keeps
(`SIDE_APPROACH_SCORE_WEIGHTS`), vertical on a tie, and a tilt of 15 deg or more only through space a depth ray saw
(`corridor_seen`). Both off is the calculator of before, byte for byte.

## What it does not do

The deformable seam is inert. No caller passes `deformable_strategy=`, so the gate is skipped, and
`CablePCAStrategy` always refuses with a telemetry hint. This calculator is built for rigid
parallel-jaw grasps; cables, cloth and bags need another planner, and the seam exists so that the rigid
path never pretends otherwise.

## Status

| Capability | Evidence |
| --- | --- |
| The calculator in the pick service, both geometry stages | measured in simulation |
| The calculator at a physical cell | run on a physical cell: the camera picks of a UR10 (CB3) with a wrist D415; no pick rate is kept here |

## Files

| File | Holds |
| --- | --- |
| [calculator.py](calculator.py) | `GraspCalculator`, and the `compute()` and `compute_result()` contract every pipeline relies on; `redraw_debug_image(candidates)`, the last overlay drawn again over the grasps a caller kept, as the pick loop does where a closing axis or the natural orientation changed a result |
| [_candidate_generator.py](_candidate_generator.py) | the silhouette, geometry-first and dense candidate generators |
| [support_footprint.py](support_footprint.py) | the support-footprint stage; a candidate's `pose()` is the base pose `Robot.pick` takes; its refusal counts by cause (`REFUSAL_CAUSES`) and the side approaches; a search's plan, units and ranking (`plan_support_footprint`, `run_units`, `rank_found`), which any process may run |
| [scene_obstacles.py](scene_obstacles.py) | what the camera saw beside the part as obstacles, the support taken out, and `why_no_grasp`; the seen test of a frame (`SeenInTheFrame`), which a worker process can be handed |
| [footprint_rim.py](footprint_rim.py) | `footprint_rim`, `rim_px`, `FootprintRim`, `FOOTPRINT_RIM_MAX_SHARE`: a mask less its rim, for the support-footprint stage's input alone (`robot.grasping.geometry.footprint_rim_mm`) |
| [footprint_hull.py](footprint_hull.py) | `hull_points`, `unread_pixels`, `HULL_LIFT_MM`: the mask pixels the stage does not read, placed in the part's visual hull, for its input alone |
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
- [multiview/](../multiview/README.md): `geometry_points_base_mm`, which feeds a fused cloud into this stage, and
  `footprint_points_base_mm`, the same views less their rim
- [deep/](../deep/README.md): the learned generator selected in its place
