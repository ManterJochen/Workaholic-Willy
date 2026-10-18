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
policy gates -> masked depth -> point cloud -> what the camera saw beside the part
             -> the support-footprint stage: the part's own body over its support, in BASE
             -> collision, table, workspace and IK filters -> base-frame candidates
```

Every formula in that chain is in [docs/grasping-math.md](../../../../docs/grasping-math.md).

## The support-footprint stage

The stage grasps the part's own body as the cameras measured it (below, *The part's own body*), planned in BASE over
the support the part stands on, so the hand clears the support instead of being filtered for it: a filter can only
refuse a grasp that goes into the table. It needs a CAMERA to BASE transform and a BASE support plane; without both a
call offers nothing, `support_footprint_ready` in the telemetry is `False`, `no_grasp_said` says why, and the reason
is `NO_CANDIDATES_GENERATED`. It also gives up when fewer than 20 cloud points survive the support-height cut. An
empty result is the stage's answer, not a failure: it looked and found no legal grasp, or it could not look at all.
The stage runs no IK, reachability or safety check: its candidates pass the same filters as any others, and its own
score is kept as `total_score`, because the geometric scorer does not order these candidates usefully. The jaw
dimensions come from `ParallelJawGripperModel`, the one measurement of the jaw every consumer reads.

`telemetry["geometry_stage"]` says `support_footprint`. The silhouette stage, which anchored grasps on the visible
surface and let the filters discard what did not fit, left on 2026-10-16 with its dense and geometry-first samplers,
its isotropic snap and the depth levers it read (the owner's cleanup): at a cell the support-footprint stage replaced
its candidates on every call. SFE's own search, a convex prism of the footprint and the grasps it admitted, left the
same day; the part's body had been the only grasps the stage offers since 2026-10-14.

The stage reads the part at full resolution, the depth under the mask a cloud of its own: `cloud`, thinned to
`CLOUD_VOXEL_MM` (8 mm), is what the support under the part and its heights in the telemetry are read from. A camera
tilted 45 degrees sees the table past a 40 mm cube's far edge 40 mm behind it rather than under it, and three things
keep the body on the part:

- The mask pixels that measure more than 10 mm behind a pixel of the same mask within 3 px are left out
  of the stage's input (`depth_steps.pixels_behind_depth_steps`); `support_footprint_depth_step_pixels`
  in the telemetry counts them.
- Low fragments that lie apart from the body, and never rise to half its height, are left out of the
  body and kept as undilated obstacles, so no finger lands on them.
- `floor_margin_mm` is how far above the support a point still counts as the support: 2.0 by default,
  the number the reference measurements were made with in a simulator whose calibration is exact.
  `Scene.from_robot_config` passes 5 (`scene.SUPPORT_READ_ERROR_MM`, the figure every cell's camera world
  gives the bench, `perceived.CAMERA_BENCH_BAND_MM`); a direct caller passes
  `support_footprint_floor_margin_mm=` to the calculator.

A fourth is the rim, 2.0 mm on every cell (`footprint_rim.FOOTPRINT_RIM_MM`, on since the owner's evening of
2026-10-09; `build_calculator` hands it every analytic calculator, `GraspCalculator(support_footprint_rim_mm=...)`, a
direct caller's default 0.0). The D415 smears a part's far edge into a ramp of depths, the outer ~3 px of the colour
mask lie on it, the depth-step rule keeps it, and the footprint follows it: on 23 grey cubes the owner's cell recorded,
the centre the closing line follows lay 2.53 mm off at the median, 2.5 mm of it away from the camera. With the rim, the
mask loses a rim of that many millimetres at the part for the stage's input alone, `ceil(rim * fx / z)` px at its
median depth and never more than 30 % of the mask ([footprint_rim.py](footprint_rim.py)). A fused cloud comes with its
rim cut where each look was seen (`compute(footprint_points_base_mm=...)`, the pick loop's looks), and the calculator
reads it beside the fused cloud for the footprint alone. `support_footprint_rim_px` and `support_footprint_rim_points`
in the telemetry, and a `footprint rim:` line in the log every compute, say what the rim took; where it would leave the
stage no point, nothing is cut. Everything else the calculator reads, the neighbours' growth, the scene, the target's
cloud and the support under it, keeps the whole mask.

With the rim comes the part's visual hull ([footprint_hull.py](footprint_hull.py)): every mask pixel the stage does
not read, the rim, the depth steps and the holes the D415 leaves on dark or shiny surfaces, gets a point where its ray
meets the part's top, or just over its support, wherever that point's vertical, from 1 mm over the support to 1 mm
under the top, projects into the mask; a pixel with no such place gives none, so the footprint never grows past what
the mask shows. On the same 23 cubes the centre lay 0.58 mm off at the median and 1.79 at the 90th percentile (the rim
with 1.25 mm pushed back onto every face, the measurement before the hull: 0.77 and 2.26), and with 35 % of the parts'
depth knocked out in blobs 0.59 and 1.87 (0.87 and 2.28); with 20 % the rim alone kept the better median, 0.77
against 1.02, the hull the better 90th percentile, 1.91 against 2.24. A fused footprint gets this look's points beside
the looks' own. The points reach the stage's input and nothing else, `support_footprint_hull_points` in the telemetry
and a `footprint hull:` line in the log say how many, and where the points read give no footprint none is placed.

## The part's own body

The stage searches the part as the cameras measured it ([body_grasps.py](body_grasps.py), the owner's lever A of
2026-10-13, alone since 2026-10-14): its points on a 2 mm grid in BASE x and y, each cell as high as its highest point, the solid
under it carried down to the support. For every closing direction (every 5 degrees), every place across the part and
every height the pads can stand at (every 2 mm), the pads close on one run of the body along the closing axis, or on
two or three runs and what lies between them, so one wall of a ring, one lobe of a star or one leg of an L is a grasp
of its own. Turned to each closing direction the body is binned from its points again, not from the grid's cells, so a
part turned 45 degrees spans what it measures and its straight faces stay straight. A span holds where it fits the
stroke with 2 mm beside each jaw, no part of the body stands over the open fingertips or between them and the span,
and both contacts hold inside the friction cone by the body's own flanks: how steeply it rises at the contact, and how
its outline turns across the pad there, read as the least-squares line over three rows either way and taken on the
gentler side: the face the pad comes to lie on as the squeeze turns the part. A round part's outline crossing a cell's
edge reads as its curvature, a box a few degrees off the closing axis by its face, and a cube across its diagonal, whose
both sides turn steeply, as a corner. The
rest of the scene is asked as the guard holds it: the support's clearance, the solids the guard holds under the
hand (`HandFloor`), the neighbours' points and the guard's boxes of them (`SeenEnvelope`), and an open finger over the
part's outline only through space a ray of the frame reached. Every grasp, kept straight down or refused there for a
neighbour, a wall or a solid under the hand, is asked again leaning on SFE's whole ladder, 15 to 90 degrees either way
about the closing axis (the owner, 2026-10-16: "alles von SFE"), the least tilt first, at most 48 a tilt, one of each
place across the part before a second, its palm kept over the support (SFE's `palm_aware`: past 45 degrees the palm is
the hand's lowest part) and its way in seen from 15 degrees on; a lean wins only with more room, vertical on a tie, and
a vertical grasp stays listed. Where that leaves fewer than three square grasps, SFE's fine pass: the leans every 7.5
degrees between the ladder's, and the part searched again in the frames of the hand rolled 10, 20 and 30 degrees about
its binormal, each rolled grasp leaning on the ladder too. A closing line
within a cell of the part's centre of mass is moved onto it. A part lower than the pads hold it with the fingertips the
clearance over the support counts a `table` refusal, and the result says it is too short for this hand. The grasps are
ranked robust first, by score (with side approaches on, upright weighs nothing and the room a grasp keeps counts more);
`score_components["support_footprint"]["stage"]` says `body`, and `body_grasps`, `body_kept` and `body_refused` in the
telemetry count them.

On a hand that opens as needed (a Robotiq that is told a width) each place is offered narrow, the span and 7 mm beside
each jaw (`NARROW_ROOM_MM`: at 2 mm the owner's Robotiq came down onto the parts, 2026-10-14), and fully open where
the open fingers are clear there too, the fully open one scored down by the stroke it opens past the narrow one, the
search asked on until enough places pass of each opening, `max_candidates` of each; a span the narrow room leaves no
stroke past is offered fully open alone. The grasp carries its opening (`open_width_mm`).

**The hand as the guard holds it.** The calculator holds each finger in three steps along the approach and the housing
in four, measured off the hand's own guard bundle (`gripper_model.hand_steps_of`): each step is the box of its 1 mm
slices of the mesh, so the calculator never holds less of the hand than the guard does. A Hand-E's tip is 6 mm thick
and 21 mm across where its pad sits, its housing 62 mm across the binormal where it meets the fingers; the registry's
one box of each stood 4.8 and 6.5 mm past them, and the calculator refused grasps the guard admitted. A hand without a
bundle keeps the registry's boxes.

## Grasps from inside a bore

On a Robotiq hand the stage also searches the part's bores ([inner_grasps.py](inner_grasps.py); the owner,
2026-10-14: "Nur wo außen nichts geht"), and offers what it finds
after every outer grasp, so a grasp from inside is tried where no outer grasp is admitted. A ring wider than the stroke
cannot be closed across, and a ring or a hub packed among its neighbours leaves no room for a finger beside it.

A bore is a hollow of the body the body encloses in the plane: cells no point stands in, or whose top stands deeper
than the pads need under the part's top, joined to no cell at the grid's edge. Through its centre, every 5 degrees of
closing direction and every height the outer search asks, the closing line meets the wall either side, read off the
body's own points in the finger's band. The fingers come down with 7 mm of room beside each jaw (`NARROW_ROOM_MM`, as
lever G's narrow grasps), closed where the bore leaves less, and open at the close's force until the wall stops them;
both contacts hold inside the friction cone, as the outer search reads them. The fingers' way into the bore and out to
its wall must be space a ray of the frame reached, because the guard does not hold the part it grasps: a bore nobody saw
free offers nothing, nor one with a neighbour's points in it. The guard's boxes of the neighbours are asked with the
fingers at the wall, the widest they stand. A bore reads up to about 8 mm narrow on the synthetic rings (52.5 of 60, 43.6
of 50): the mixed pixels of its rim fall inward, which leaves the fingers more room on the way in, and the open finds
the wall wherever it is.

Each grasp says it grips from inside (`components["support_footprint"]["inner"]`, `stage` `inner`) and carries what
its execution reads: the jaws' width on the way in (`line_in_width_mm`, as `open_width_mm`), where they meet the wall
(`grip_width_mm`), the part's width about the grasp it is carried as (`part_width_mm`), the fingertips' depth and the
bore across. `inner_grasps`, `inner_kept`, `inner_refused` and `inner_bores` in the telemetry say how many it offered,
why it offered no more, and how wide each bore it found read. The execution opens the jaws onto the wall
(`GripsFromInside.grip_from_inside`), and the release closes them where the hand says it holds so
(`execution.handling`).

## Which way round

**Which way round is the pick loop's.** The calculator ranks each grasp as generated. A program's
`closing_axis` and the cell's `robot.motion.natural_closing_axis` choose the way round afterwards, in the pick
loop ([loop/](../loop/README.md)); while an axis is named the loop raises `max_candidates` to 36 for that
ranking and gives it back. So an IK service wired into the calculator judges each candidate before the loop
turns it half a turn (the feasibility re-rank left on 2026-10-16); no builder wires one today, and the arm's own guards judge
the grasp executed. Where the loop changed a result it redraws the overlay (`redraw_debug_image`).

## What it sees beside the part

In every cell the calculator holds every object the camera saw beside the
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
`ALL_TABLE_CONFLICT` ("too short for this hand" under about 8 mm on a Hand-E), the part's own fragments
`NO_VALID_GRASP`, with `why_no_grasp`'s sentence in the telemetry (`no_grasp_said`) and the log. Every cell's
calculator also takes the tilted approaches (`side_approaches=True`; the keys `robot.grasping.scene_obstacles` and
`side_approaches` left on 2026-10-17 at the cell's values): the stage offers every tilt the hand fits at, scored by
the room each keeps (`SIDE_APPROACH_SCORE_WEIGHTS`), vertical on a tie, and a tilt of 15 deg or more only through
space a depth ray saw (`corridor_seen`). A calculator a script builds by hand reads neither, the calculator of
before, byte for byte.

A finger may come to the measured surface of a part a prompt named (the owner, 2026-10-05: "Finger dürfen
streifen"), and every cell names the parts beside it too (the owner, 2026-10-12): the pick reads them off the
depth (`parts_beside.py`): what stands over the supports, outside the named
masks, seen whole, apart from anything larger, within 120 mm of a named part and of a part's size and shape. Their
masks reach the calculator with the named parts' (`PerceptionFrame.parts_beside`), so their points make soft boxes
too, and the camera world names its boxes by the same points. A pixel beside the part is a named part's where it lies
within 5 mm (`fine_world.FINE_LABEL_MM`) of the surface the pick names that part by (`named_surface`), the first part
in order, as the guard names a wrist camera's pixels.

**Every look of the pick.** The guard holds the boxes of every look a pick took, and so does the calculator: the pick
loop hands it each other look as a `SceneView` (its depth, lens and pose), and `scene_obstacle_points` builds one set of
boxes from the union of the looks' points, each other look's view of the target held out by the box the world keeps
out for it (`perceived.target_keep_out_box`). Held out by the target's mask projected into the other look instead, the
faces only that look saw stood as neighbours: the owner's pick of 2026-10-12 at 08:18 lost every grasp (0 the guard
admitted, 22 now). The corridor test asks every look's rays (`SeenInAnyFrame`). The open hand is measured only against
the boxes whose spheres and whose extents along BASE's axes come within the guard's distance of it
(`SeenEnvelope.refuses_many`, `envelope_verdicts`): the same verdicts, measured on the recorded picks.

## What it does not do

The deformable seam is inert. No caller passes `deformable_strategy=`, so the gate is skipped, and
`CablePCAStrategy` always refuses with a telemetry hint. This calculator is built for rigid
parallel-jaw grasps; cables, cloth and bags need another planner, and the seam exists so that the rigid
path never pretends otherwise.

## Status

| Capability | Evidence |
| --- | --- |
| The calculator in the pick service, the support-footprint stage | measured in simulation |
| The calculator at a physical cell | run on a physical cell: the camera picks of a UR10 (CB3) with a wrist D415; no pick rate is kept here |

## Files

| File | Holds |
| --- | --- |
| [calculator.py](calculator.py) | `GraspCalculator`, and the `compute()` and `compute_result()` contract every pipeline relies on; `redraw_debug_image(candidates)`, the last overlay drawn again over the grasps a caller kept, as the pick loop does where a closing axis or the natural orientation changed a result |
| [hand.py](hand.py) | the hand the stage plans with: `SupportFootprintJaw`, the floor the guard holds under it (`HandFloor`), the room its fingers keep, the ladders it leans on and the scores a grasp is ranked by; `SupportFootprintCandidate`, whose `pose()` is the base pose `Robot.pick` takes |
| [support_prism.py](support_prism.py) | `reconstruct_support_prism`: the part's convex footprint extruded to its support, which the visual hull reads the part's top off and the pick loop its short side |
| [body_grasps.py](body_grasps.py) | `body_grasps`, `body_map`: the stage's grasps, on the part's own measured body, leaning where the scene refuses them straight down |
| [inner_grasps.py](inner_grasps.py) | `inner_grasps`, `bores`: the grasps from inside the part's bores, after every outer grasp, on every Robotiq cell (`robot.grasping.inner_grasps` until 2026-10-17) |
| [scene_obstacles.py](scene_obstacles.py) | what the camera saw beside the part as obstacles, the support taken out, and `why_no_grasp`; every look's view of the scene (`SceneView`) and the parts named as the guard names them (`named_surface`); the seen test of a frame (`SeenInTheFrame`) and of every look of a pick (`SeenInAnyFrame`) |
| [parts_beside.py](parts_beside.py) | `parts_beside`, `why_no_part_beside`, `LABEL`: the parts beside the named ones, read off the depth on the supports, on every cell (`robot.grasping.name_neighbours` until 2026-10-17) |
| [footprint_rim.py](footprint_rim.py) | `footprint_rim`, `rim_px`, `FootprintRim`, `FOOTPRINT_RIM_MAX_SHARE`, `FOOTPRINT_RIM_MM`: a mask less its rim, for the support-footprint stage's input alone, and the rim every cell cuts |
| [footprint_hull.py](footprint_hull.py) | `hull_points`, `unread_pixels`, `HULL_LIFT_MM`: the mask pixels the stage does not read, placed in the part's visual hull, for its input alone |
| [_support_footprint_stage.py](_support_footprint_stage.py) | the adapter that brings that stage's base-frame candidates into the camera frame |
| [_camera_geometry.py](_camera_geometry.py) | the lens: the calibrated intrinsics, or ones synthesized from the image size and said once |
| [mask_analyzer.py](mask_analyzer.py) | centroid, principal and minor axes, extents, oriented and axis-aligned boxes; the calculator's first gate, a mask too small to read |
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
