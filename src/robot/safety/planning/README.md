# The motion planner and the collision engines (`src/robot/safety/planning`)

This package holds the two external engines the motion stack builds on, cuRobo for collision-free
trajectories and Coal (or python-fcl) for exact mesh distance calculations.

```python
from willy import MotionStack, load_tree

report = MotionStack.from_robot_config(load_tree().robot).probe()   # the cell WILLY_PROFILE names
print(report)                       # both engines, the arm, and the config key that named it
raise SystemExit(report.exit_code)
```

```bash
python -m src.robot.safety.planning --check                 # reads paths; spawns nothing
python -m src.robot.safety.planning --doctor                # loads every engine; takes seconds
python -m src.robot.safety.planning --check --model ur3e --hand robotiq_hande --json
```

Exit `0` means fully anchored: the planner's interpreter is present, and an exact mesh engine and the
arm's mesh bundle import. Exit `1` means a degraded fallback would run (blind IK instead of the planner,
the capsule proxy instead of meshes), or no hand is named, or the config did not load. Exit `2` comes
only from `--doctor`: an operating-system policy blocked a binary that is present, which needs the
opposite fix to a missing one ([code-integrity.md](../../../../docs/code-integrity.md)). A development
machine without the GPU environment answers `1` by design. The same reading runs in
[planner_or_ik.py](../../../../examples/offline/config/planner_or_ik.py).

## Install

`scripts/ext_deps/install.ps1` installs both engines under `ext_deps/`, where the defaults look, so a
standard machine sets no environment variable ([ext_deps/README.md](../../../../ext_deps/README.md)).
`CuroboPlanClient` spawns it once per session.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `MotionStack` | `from_robot_config(robot)`, `for_this_box()`, `from_model(model=...)` | `probe()` | `MotionStackReport` with `exit_code` |
| `CuroboPlanClient` | the driver, sized by the cell's reservation | `plan(goal)`, `check_joints(configs)` | joint waypoints; a verdict per sample |
| `LivePlannerWorld` | the driver, from the cell's cameras and declared world | `world_for(...)` | a `PlannerWorldSnapshot` with its verdict |
| `PlannerHand` | `planner_hand(robot, data_dir=...)` | read | the hand the planner and the guard model |
| `ExactPairs` | the exact guard, `SelfCollisionGuard.exact_pairs(arm)` | `decides(link_a, link_b)`, `distance_mm(joints, link_a, link_b)` | the guard's pair rule and distances in the planner's link names |
| `PathJudgement` | `CuroboPlanClient.judge_joints(configs, clearance_mm=, name_pairs=)` | read `refused` | one `RefusedSample` per refused configuration: its three terms apart and every pair of links it found |
| `PoseScreen` | `URRobotArm.screen_configuration(joints)` | `line(label)` | `clear`, `band`, `seen_boxes`, `guard_refused`, `planner_refused` or `unscreened`, with a pose nearby both clear where one exists |


## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| exit `1`, no hand | the tree names no `robot.gripper.model` | name the hand, or pass `--hand` |
| a move fails `CONTROLLER_REJECTED` | the planner is unavailable on a UR with `robot.ur.motion_planner: curobo`, the default | run `--doctor`; the arm never moved |
| a move fails `TIMEOUT` | the planner found no collision-free plan | move the goal, or clear the cell |
| `CuroboUnavailableError` from `check_joints` | no reply, a sidecar that exited, or a reply that is not a verdict for every sample | restart the sidecar from this tree |
| `ValueError` from `check_joints` | no configuration, a value that is not finite, or a wrong joint count | send whole configurations |
| `CameraWorldUnavailable` from a verb | a camera stays silent, blind or stale after `perceived.fresh_frame_attempts` more readings; blind is a frame with no depth, a fixed camera's frame less than half of whose pixels hold one, or a wrist camera's asked about no goal | fix the camera, or move it further from what it sees; a pick stops |
| a move fails `CONTROLLER_REJECTED`, world `unseen` | cameras look at the motion's goal and none holds a depth on half the 200 mm about it | look from further away, or use a depth mode with a shorter minimum range |
| a planner that does not start | no committed evidence file for this arm, hand, coupling, placement and margin | measure one, see [`robot/evidence/`](robot/evidence/README.md) |
| a planner that does not start, `..._a16.json` | a declared carried part (`planning_world.payload.length_mm`) reserves 16 attach slots, and nobody measured the combination with them | the refusal's `matrix_gate.py ... --attach 16` command, about a minute on the cell's GPU; the UR10 files are committed |
| a move refused naming `safety.planning_world.perceived.max_boxes` | the camera world still does not fit its slots once every object is one box | give the planner more slots (`max_boxes`, then restart it), or clear the cell |
| `CuroboUnavailableError` from `judge_joints` | a sidecar older than the report, rows out of order, a verdict and a report that disagree, or pair names nobody asked for | restart the sidecar from this tree; the refusal the report was asked about stands |
| `CuroboUnavailableError` from `judge_joints(..., ignore_perceived=)` | a reply that does not say it set aside exactly the camera boxes named (an older sidecar), or a sidecar that refused: a name it does not hold, a camera box not named, a part it may carry | the refusal the second judgement was asked about stands; restart an older sidecar from this tree |


## What the planner is told about the cell

`robot.safety.planning_world` turns on a world rebuilt immediately before every plan, for any motion.
A world nobody can vouch for refuses the motion instead of being planned against. Three kinds of
geometry reach the planner:

| Channel | What it is for | Cost, measured on the workstation |
| --- | --- | --- |
| Boxes | what an operator declares; the only kind a refusal can name. Bounded by the planner's slots | 21.6 ms to register 41 |
| Meshes | a container that keeps its hollow, under `planning_world.meshes`; the sidecar reads the file | 6.4 ms to register one |
| A distance field | the whole scene at the resolution it is cut to, with nothing dropped for want of a slot | 13.7 ms to build 179,560 cells, 1.9 ms to register |

The field is a signed distance in metres, negative inside an obstacle. The guards read boxes only, so a
declared mesh is known to the planner alone. The path guard hears the same refresh and gets the same
perceived boxes, **turned as cuRobo holds them**, and judges them at
`self_collision.perceived_min_distance_mm` (5 mm), each already grown by `perceived.margin_mm`; the arm and
a declared fixture keep `min_distance_mm` (10). The capsule fallback cannot turn a box and judges its
axis-aligned enclosure, the safe side. So the guard and the planner judge the same cell, and a move planned
on a vouched world is stamped `PLANNED` with its cameras and the capture time of its oldest image. A pick's
goal keeps the space between the open jaws out of every view (`goal_keep_out`), so a world that registered
the part still has a plan to it; `WorldRefresh.keep_out` records what was left out, or why nothing was.

### The camera world is a height map

[`height_map.py`](height_map.py) turns each object the cameras saw, each cluster or each piece a keep-out
cut, into a few upright boxes. The object is laid on a grid in its own turn, cells at least
`perceived.cluster_voxel_mm` (25 mm), fitted so its span splits into whole cells; every cell keeps its
highest seen point, and cells whose tops lie within `perceived.voxel_size_mm` (10 mm) merge into rectangles.
Each rectangle is one box turned about base Z, its points' extent plus 15 mm on every side, from its floor
(the bench, or a keep-out's top) up to its highest point plus 15 mm. The turn is the long side of the
smallest-area hull rectangle, snapped square to base X/Y unless the turned one is at least 5 % smaller:

- an **open bin** is its walls with a **free inside**;
- a **low part beside a tall one** keeps its own height;
- a **turned object** is a turned box.

Every thinned point the world keeps lies inside a box, at least the margin from its four sides and its top,
and the same frame always gives the same world. A pixel the thinning dropped can lie nearer a face
([guide 04](../../../../docs/guide/04-robot-and-safety.md), 5.5).

**The budget.** `perceived.max_boxes` defaults to **64**; the reservation follows, 1 + declared + 64 box
slots, so **restart the planner after changing it**. Over budget, `coarsen()` merges two boxes of one object
into the box that holds both, the least added volume first, until the world fits or every object is one
box; the refresh says `N box(es) merged into the boxes holding them to fit the slots`
(`PerceivedWorld.merged_to_fit`, `WorldRefresh`). What still does not fit keeps the nearest boxes, empties
the guard's and refuses the motion naming `safety.planning_world.perceived.max_boxes`.

**The self filter** ([`self_envelope.py`](self_envelope.py), `SelfBody`) takes a point out as the robot only
within `perceived.margin_mm` (15 mm) of a link's own surface, laid over the committed bundle at 4 mm
(188,000 points on a UR10, built once in half a second); the link capsule alone took up to 118 mm around the
UR10's shoulder end. What it took of an object also seen past those 15 mm is put back at the height seen
beside it. The Hand-E is still its padded sphere map, up to 27 mm around the hand, because its fingers move.
About 1 degree of hand-eye or DH error, 15 mm at 0.85 m, makes the arm's own surface an obstacle, and the
box says so (`PerceivedBox.robot_gap_mm`, `note()`: `it may be the robot itself ... check the hand-eye
calibration and the DH table`).

**What the robot hides is not free** (`perceived._RobotShadow`). A base point counts as hidden where some
camera saw the robot more than 5 mm in front of it and no camera saw as far as it. Inside an object's grid a
hidden stretch stands as high as the highest cell seen in it or beside it, never past the one box the object
would have been; a row the self filter took for the hand runs on into hidden cells, at most 150 mm and never
across itself; and `height_map.bridge_columns` fills the robot's shadow between two parts it cut apart, cut
back to the parts' extent. Nothing within a keep-out's enclosure plus the margin is filled, and a cell
another camera saw is not. Boxes and the world count `hidden_cells`, and a refusal says `(N of its cells the
robot's own body hid from the cameras, filled to the height seen beside them)`. What the scene hides from
itself, pixels with no depth and a lone object the robot hides from every camera stay free.

### What the camera world does not see

A pixel with no depth is space the planner receives as free: the camera measured nothing along that
ray, because the surface was closer than its minimum range, could not be read, or lay in its shadowed
band. Intel's datasheet gives a D415 a minimum range of about 450 mm at 1280 x 720 and about 310 mm at
848 x 480 (not measured here), so a camera on the wrist is blind to what is near the hand. The world
counts those pixels (`DropReason.NO_DEPTH`) and reports each camera's share of depth
(`PerceivedWorld.depth_coverage`). A frame with no depth at all is `BLIND`, and so is a fixed camera's
frame less than half of whose pixels hold one: a fixed camera is never carried toward what it sees, so
that is the camera or something in front of it. A camera on the wrist is carried inside its minimum
range by every grasp (the review's ray cast of the owner's pick: 27 % of the frame at the grasp at
848 x 480, 12 % at 1280 x 720, with the hand's line out of view), so its frame is judged where the
motion goes, and by the whole frame only when the motion names no goal. A goal that cameras look at,
none of them holding a depth on half the region about it, is `UNSEEN` and refuses the motion. What
stays below those limits, and a goal no camera looks at, is written on the motion's `PLANNED` stamp as
`unseen`, so the stamp does not vouch for it. A camera on the wrist sees only where it points: a goal
outside its view is planned against what is declared there.

The bench is the support plane plus a band: the configured `perceived.plane_clearance_mm`, plus, for
a camera on the wrist, the tool motion measured across its grab (at most its rig's shutter motion
tolerance, and about nothing on an arm standing still) grown with each point's range, at most 25 mm
more (`PerceivedWorld.bench_band_mm`). Anything lower than the band above the bench is not an obstacle.
The workspace box bounds the TCP only, so the world keeps what the robot's own body can reach: a
sphere about the base that holds every link, the hand, a wrist camera and a held part in every
configuration (`SelfEnvelope.reach`, 1793 mm on a UR10 with a Hand-E).

What the planner already holds is not registered twice. A camera point within
`perceived.DECLARED_SURFACE_MM` (5 mm) of a declared fixture or a declared mesh, plus the view's
measured error as above and never more than the bench band, is that fixture (`DropReason.DECLARED`,
counted per body in `PerceivedWorld.declared_points`), so a declared tote keeps its hollow instead of
coming back as a perceived block over it; what lies in the tote stays an obstacle. The band is not the
bench band itself: a slab sunk below the bench raises `plane_clearance_mm` by the sink, and a part
beside a declared tote is not the tote. A mesh that cannot be read for this is said on the snapshot,
and what the cameras see of it stays an obstacle. A cluster that spans a
keep-out box, the neighbours around a part in a pile or the floor of a tote around it, is cut around
the box before it is fitted, into what lies beyond each of its sides, above it and below it
(`PerceivedWorld.keep_out_cuts`). A cut box reaches into the keep-out by at most its margin, which is
the margin a target's keep-out is grown by, so it never covers the target. Measured 2026-09-23 on a
3 x 3 pile of 40 mm cubes: at a 30 mm gap the Hand-E's pads are free, where one box covered them before.

A camera on the wrist has moved since its last frame. Its depth source reads the tool pose, tells the
handle the camera moved where the handle can drop its temporal filter's history (`camera_moved()`),
throws away five frames (`depth_source.WRIST_WARMUP_GRABS`, as the pick frame does) and keeps the next:
a frame the stream queued during the move, and holes a temporal filter filled with the depth of the
pose before, are among the ones thrown away. A fixed camera grabs once.

### The frames of a wrist pick

A wrist camera sees only where it points, so a pick that looks from several poses keeps **every frame**
it took from its first look. While a pick holds (`LivePlannerWorld.hold_pick_views`, or
`keep_out.holding_views(arm)` as a block), every wrist frame a world was built from is kept, one per pose
the arm stood at: the looks, the generated view, the standoff, the grasp and the retreat. The hold starts
once the arm stands at the first look it reached, never at the pose the pick started from, and a first look
refused before anything was sent starts none. Each goes into every world built until the pick
ends, placed by the tool pose it was stamped with, the robot taken out where it stood then and where it
stands now; what only an older frame saw stays an obstacle. A pick holds 12 frames at most, and a frame of
a new pose past that is not held and is counted on the snapshot (`held_not_kept`). A fixed camera holds
nothing: its newest frame supersedes the one before.

Who starts and ends the hold:

- **The pick loop's looks.** `look_around` starts it for a pick handed looks, and `run()` ends it when
  the pick ends, however it ends; `look_around` lets go itself when it raises.
- **`Locator.look_around`.** It lets go of an earlier look around's hold before it moves
  (`LivePlannerWorld.holds_pick_views`), starts its own at its first look, lets go itself when it raises,
  and `Robot.pick`, the next `look_around` or the disconnect (`Robot.connected()`, `Cell.connected()`) ends
  it. A place does not need the frames.

The one view a wrist pick generates, and its move back, run only against a world that holds them: with
no live world, no wrist camera in it or a camera world that is declined, none is generated
([execution](../../execution/README.md)).

`check_js` judges a whole joint path in one request: up to 1000 configurations, each against the joint
limits, the robot itself and the world the planner holds, with an attached payload counted. A sample
passes when no sphere penetrates; that is not a kept clearance. A request may name one instead
(`clearance_m`, up to 0.1 m): the world is then judged at that clearance, the reply says the clearance it
judged at, and the client refuses a reply that does not. The UR driver asks it of a straight joint line
it would run instead of a plan (`safety.planned_motion.line_clearance_mm`), because nothing shaped that
line to keep clear. A longer path is split across requests and never thinned.

Every plan starts from the same seed, so the same request in the same world gets the same plan, and the
graph planner seeds its roadmaps from the start and the goal alone: cuRobo's shipped config links both
through the retract, which the sidecar switches off (`_curobo_plan_policy.py`). The UR driver asks only
for joint plans (`plan_js`); the pose plan remains for the Isaac driver.

### When the planner's spheres refuse what the meshes keep clear

The owner, 2026-09-30: for the arm's own pairs the exact guard judges, the exact guard decides
([`band.py`](band.py)). cuRobo's sphere cover reaches past the meshes and `planner_margin_mm` pads it
further, so a tilted look can be a pose the padded spheres refuse while the meshes keep 19 mm.

- **The report.** `check_js` takes `report_refused` (and `name_pairs`): the reply then lists every refused
  sample, in order, with `bound_ok`, `self_ok` and `world_ok` and, for a self hit, every overlapping pair
  of links with its padded and unpadded depth, and says `pairs_named`. `_curobo_pairs.refused_rows` builds
  the rows, and `_curobo_pairs.overlapping_pairs` names every pair the kernel counts (padded radius not
  negative, within 0.01 mm of contact). Without the key the request and the reply are byte for byte what
  they were. `CuroboPlanClient.judge_joints(configs, clearance_mm=, name_pairs=)` returns a
  `PathJudgement` of `RefusedSample` rows (`findings()`), batched at 1000 and indexed over the whole path,
  and refuses anything it cannot read whole.
- **The admission.** `band.admission_refusal(sample, exact, margin_mm=)` says why a refusal stands: a
  joint outside the planner's bounds, its world, the carried part, a self hit with no named pair, a pair the
  guard does not judge (the `shoulder_link`, the planner's only model of the robot's base, is never one),
  or padding past `planner_margin_mm`. `None` leaves the sample to the exact guard, which the driver asks
  about that very sample again (`URRobotArm._exact_guard_decides`). `ExactPairs` carries the guard's own
  rule (`MeshSelfCollisionBackend.checks`, the rule `evaluate` skips pairs by) and its exact distances.
- **The legs.** `band.band_neighbours` lays a grid of at most `NEIGHBOUR_BUDGET` (1000) configurations,
  one check request, over the joints between the colliding links, at most `BAND_LEG_MAX_DEG` (**20**) each,
  nearest first by the path gate's measure: 0.5 degrees a step on one joint, 1.33 on two, 5 on three, 10 on
  four, 20 on five or six. A planned move out of a band pose or into one takes a straight leg to the nearest
  pose both clear, judged by both; `PoseScreen` says the verdict of a pose before anything goes there.
- **The camera's boxes** (the owner's Option 1). The same spheres reach 25 to 29 mm past the UR10's shoulder
  housing into the world. `check_js` takes `ignore_perceived`, the names of every box the camera saw that the
  sidecar holds (`PERCEIVED_PREFIX`, `seen_`): it judges with exactly those set aside by cuRobo's own
  `enable_obstacle`, puts every flag back before it replies, reads every flag back, and says so under
  `perceived_ignored` ([`_curobo_perceived.py`](_curobo_perceived.py)). The branch decides nothing of its own:
  `requested_aside` reads the request, `None` for a plain check, and `judged_world` opens the world it judges
  in, the whole world with no storage touched for a plain check, `SetAside` otherwise; the CPU suite runs both.
  It refuses, as a failed call, names that are not every camera box it holds, and anything while it may hold
  a carried part; a flag that does not come back ends the sidecar. Without the key the request and the reply
  are byte for byte what they were. The UR driver asks it where its world refused samples, and admits them
  only where the second judgement clears the world and the bounds and finds the robot itself alike
  (`band.world_admission_refusal`), no part is carried (any attach since the last detach, modelled or not), the
  hand reads empty and open (a toggle's count open, a gripper measured fully open: the owner, 2026-10-01), the
  glue's last confirmed refresh handed the planner exactly the boxes the exact guard holds
  (`CuroboUrPlanner.perceived_in_world`), and the exact guard accepts every refused sample with them. The
  bench, the declared fixtures and meshes and a distance field are never set aside, and cuRobo still plans in
  its whole world: a planned move into or out of such a pose is refused, and so is any move while the hand
  carries a part or cannot say it stands open. A grasp there judges its lift as if the jaws held the part
  before they close, and backs out with them open where that lift would be refused
  (`URRobotArm.carried_line_refusal`). `PoseScreen` asks what a move asks, for a hand known empty and open, and
  says `seen_boxes` for such a pose.

The composed robot, its `composed_sha256` and every evidence file are unchanged.

### The probes on a GPU

```bash
python scripts/curobo/probe_band_admission.py                  # the exact guard decides the arm's own pairs
python scripts/curobo/probe_turned_boxes.py                    # turned boxes, bins beside the base, a plan
python scripts/curobo/probe_turned_boxes.py --cpu-replica ext_deps/curobo/curobo/content   # where the kernels cannot load
```

Both start the planner through the driver's own start, which checks the committed evidence, drive nothing,
and exit 0 when every expectation held. Each starts a sidecar of its own and never looks for one already
running, so stop the console, the API and every other planner first. Both run **a fixed owner-like cell**,
built in code (a UR10, a Hand-E on a 20 mm plate, no wrist camera and so no housing), and read nothing of
the cell's tree. `probe_band_admission.py` holds LOOK[0] and LOOK[1] admitted with their escape legs on that
cell, LOOK[1]'s exactly at the 20-degree cap (wrist_1 +20.0, its nearest clear pose +19.5), and the folded
wrist, a box through the forearm, a joint at 354.6 degrees and the Hand-E at the robot's base refused, plus
the kernel's self term against the pair naming over a 5-degree wrist_1 x wrist_2 torus.
`probe_turned_boxes.py` fills 65 slots from a synthetic camera, compares turned boxes with their
enclosures, holds a bin **30, 40 and 47.7 mm** beside the shoulder housing admitted with the camera's boxes
set aside, at no clearance and at a line's, in the cushion band and with the wrist out of it, where the
planner refuses on its world alone; holds every camera box put back (one configuration inside each, its
verdict and depth the same before and after); runs straight joint lines beside the 30 mm bin, holds the line
out while the hand's count says closed and runs it once open, refuses the grasp's lift judged as if the jaws
held a part while the same lift runs empty-handed; records where the planner clears a declared bin
(about 25 mm at no clearance, 40 at the line clearance); and plans out of the cushion band in that world.
`--cpu-replica` judges with a CPU replica of the sidecar
(cuRobo's spheres by forward kinematics, the kernel's rules, the sidecar's own row builder). Every line it prints
starts with `[replica]` and every log record it makes carries the same tag; its `planner under test` line
says CPU REPLICA, and its JSON names the replica. It plans nothing, so `probe_turned_boxes.py` skips its plan
and says so. It proves the driver's decisions on the real geometry, and only the GPU run holds the kernel.
The cell's own looks are screened by `real_cell --start-planner`, one line per look.

```bash
python scripts/curobo/probe_live_world.py
```

That probe runs the real sidecar with a wall no config file declares: the empty cell plans, the wall
stops the plan, removing it plans again. The middle result alone would prove nothing, because a planner
that refuses everything looks the same.

## Environment variables

| Variable | Default | Read by | Meaning |
| --- | --- | --- | --- |
| `WILLY_CUROBO_PYTHON` | `ext_deps/curobo_env/python.exe` | client | the planner environment's interpreter |
| `WILLY_CUROBO_ROBOT` | `ur5e.yml` | client | a descriptor for a client built without one; a cell names `willy_{arm}.yml` |
| `WILLY_CUROBO_CUBOID_CACHE` | `16` | client | box slots |
| `WILLY_CUROBO_MESH_CACHE` | `0` | client | mesh slots |
| `WILLY_CUROBO_VOXEL_GRID` | unset, no grid | client | the live-scene grid, `x,y,z,voxel` in metres |
| `WILLY_CUROBO_STDERR` | unset, discarded | client | a file for the sidecar's stderr |
| `WILLY_CUROBO_MAX_ATTEMPTS` | `16` | sidecar | plan attempts, each a fresh seed batch |
| `WILLY_CUROBO_GRAPH_FROM_ATTEMPT` | `1` | sidecar | the first graph-seeded attempt |
| `WILLY_CUROBO_ATTACH_SPHERES` | unset, as `0` | sidecar | spheres for a carried payload; the client sets it from the cell's reservation |
| `WILLY_CUROBO_SELF_COLLISION_MARGIN_MM` | unset, as `0` | sidecar | the guard's clearance, raised into the descriptor's link buffers |
| `WILLY_COAL_PREFIX` | `ext_deps/coal_env` | collision engine | the environment that provides Coal |

The names live in `environment.py`, except the attach and margin variables, which `_curobo_attach.py`
and `_curobo_margin.py` name. A cell's reservation (`reservation.py`) wins over the three slot
variables, and the client warns when one disagrees with it.

## Status

| Capability | Evidence |
| --- | --- |
| Planned moves on a UR arm | measured against a UR10 |
| The live world and the batch check | measured against a UR10 |
| Exact mesh collision through Coal or python-fcl | measured against a UR10 |
| A planned move on a physical arm | measured against a UR10 |
| The band admission, the escape legs, 65 slots of turned camera boxes and a plan in them | measured on a GPU with the two probes (2026-09-30, the development box), not yet on the cell PC |
| The camera's boxes set aside and every one put back, a bin 30, 40 and 47.7 mm beside the housing admitted in the band and out of it, the line out held while the hand's count says closed, a grasp's lift judged carrying, with the part in the sidecar on a cell that models it, a declared bin's room | measured on a GPU with `probe_turned_boxes.py` (2026-10-01, the development box), not yet on the cell PC |

`--check` reports that the sidecar's interpreter exists; it does not report that the descriptor beside
it was built, because that lives in an environment this process does not spawn. `--doctor` closes that
gap. Planner collision awareness is not a certified functional-safety stop.

## Files

| File | Holds |
| --- | --- |
| [`environment.py`](environment.py) | every variable name, default path and availability probe; `import_collision_engine` |
| [`stack.py`](stack.py), [`__main__.py`](__main__.py) | `MotionStack` and the `--check` command |
| [`doctor.py`](doctor.py) | `--doctor`: loads every engine and classifies an OS policy block |
| [`curobo_client.py`](curobo_client.py), [`curobo_planner_server.py`](curobo_planner_server.py) | the client, and the sidecar it runs in the planner's interpreter |
| [`world.py`](world.py), [`perceived.py`](perceived.py), [`live_world.py`](live_world.py) | declared geometry, camera geometry, and the world rebuilt before every plan |
| [`height_map.py`](height_map.py) | an object's boxes: its height map in its own turn, the fill of what the robot hid, the merge to fit the slots |
| [`band.py`](band.py) | the planner's cushion band: which refusals the exact guard decides, the escape and approach legs' neighbours, the pose screen |
| [`depth_source.py`](depth_source.py) | a camera rig asked for depth alone |
| [`reservation.py`](reservation.py), [`margin.py`](margin.py) | the slots the sidecar allocates, and the clearance the planner keeps |
| [`hand.py`](hand.py), `_hand_placement.py`, `_hand_bundle.py` | the hand from `robot.gripper.model`, where it sits, and its bundle checked |
| [`body_link.py`](body_link.py) | a wrist camera as one body for the planner, the guard and the self filter |
| [`self_envelope.py`](self_envelope.py) | the robot's own body, filtered out of the camera view |
| [`evidence.py`](evidence.py), [`bundle_index.py`](bundle_index.py) | the measured combination files, and the index of committed bundles |
| `_declared_body.py` | a declared box as bundle arrays, and the proof that a sphere fill covers it |
| `_curobo_*.py` | the sidecar's descriptor, attachment, margin, pair and protocol helpers; `_curobo_pairs.py` names the overlapping pairs and builds the refused rows; `_curobo_perceived.py` sets the camera's boxes aside for one judgement and puts them back |
| [`robot/`](robot/PROVENANCE.md) | sphere maps per arm and hand, the retract table, the hand writers, the evidence |

## Details

- [`../README.md`](../README.md): the guards this package sits under, and what the capsule fallback costs
- [`../../drivers/ur/`](../../drivers/ur/README.md): the UR path that binds the planner fail-closed
- [`../../grasping/collision/`](../../grasping/collision/README.md): another user of the exact mesh engine
- [Robot and safety guide](../../../../docs/guide/04-robot-and-safety.md), section 6; [safety-math.md](../../../../docs/safety-math.md)
- New hand: `scripts/grippers/write_hand_from_mesh.py`, then [your_own_gripper.md](../../../../docs/runbooks/your_own_gripper.md)
- Descriptors: `scripts/curobo/build_ur_config.py` builds `willy_{arm}.yml` inside `ext_deps/`
- Tests: `tests/test_motion_stack.py`, `tests/test_planning_doctor.py`, `tests/test_curobo_batch_check.py`,
  `tests/test_the_exact_guard_decides_the_planners_self_pairs.py`,
  `tests/test_a_pose_only_the_planners_spheres_refuse_is_the_exact_guards.py`,
  `tests/test_the_camera_world_is_a_height_map_of_what_it_saw.py`,
  `tests/test_the_guard_holds_the_boxes_the_planner_holds.py`,
  `tests/test_what_the_robot_hides_from_a_camera_is_not_free.py`,
  `tests/test_the_band_admission_judges_with_the_boxes_the_camera_saw.py`,
  `tests/test_the_planner_judges_again_without_the_boxes_the_camera_saw.py`,
  `tests/test_a_bin_the_camera_saw_beside_the_base_is_the_exact_guards.py`,
  `tests/test_the_cameras_boxes_are_set_aside_only_for_a_hand_known_open.py`,
  `tests/test_a_grasp_judges_its_lift_carrying_before_it_closes.py`
