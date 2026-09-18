# Calibrating a real cell

Connect a camera, teach it where the robot is, and prove the answer before you trust it. Work from
this page at the bench. Derivations, solver internals and the simulator procedure are in
[guide chapter 03](guide/03-calibration.md).

**How far this is proven.** The routine, the AX=XB solve and both mounting modes run in Isaac Sim and
in the offline suite. Nothing on this page has run against a physical controller or a physical depth
camera. Treat your first live sweep as commissioning, not as a validated path.

## Why the cell will not move until you do this

Perception reports grasps in the CAMERA frame. The drivers only accept targets in the BASE frame.
Calibration produces the transform between the two.

`AutonomousGraspService.from_robot_config` refuses to build a cell for a real vendor when no
`CAMERA->BASE` resolver is configured. That refusal is deliberate. Without it the cell builds,
connects, and then rejects every motion with `MotionStatus.INVALID_TARGET`. At the bench that looks
like a broken robot, not a missing file.

No shipped camera section declares a rig's calibration: every rig's `extrinsics` is unset, so the
primary camera has no `CAMERA->BASE`. That is the default state of a fresh cell. Check yours:

```bash
python -m src.robot.execution.real_cell --check
```

Its `camera -> base` row names the primary rig's key, `camera.cameras.rigs[<id>].extrinsics`.

## 1. Pick the mounting mode

| | Eye-to-hand | Eye-in-hand |
|---|---|---|
| Camera | fixed in the cell | on the wrist |
| Marker | rigid on the tool | fixed in the workspace |
| Solves for | `CAMERA -> BASE` | `CAMERA -> TOOL` |
| Written by | `save_extrinsics` | `save_cam_to_tool` |
| At runtime | used as it is | composed with the live TCP on every frame |

A multi-camera cell calibrates each camera on its own and declares each calibration on that camera's
rig. Run this whole procedure once per camera.

## 2. Connect the camera

The live path is an Intel RealSense RGB-D rig. `pip install -r requirements.txt` covers it, and
`pyrealsense2` is imported lazily, so a machine with no device attached is unaffected.

Declare the rig in [`config/camera/cam.yaml`](../config/camera/cam.yaml) under `cameras.rigs`. The
shipped tree already carries one such rig, `realsense_d435`, with `enabled: false`:

```yaml
- rig_id: realsense_d435   # keys the artifact; fusion.cameras names the rig by this id too
  enabled: true
  source: rgbd
  rgbd_backend: realsense  # the schema default is `opencv`, which returns empty depth on a D435
  serial_number: "1234567890"   # required once two RGB-D rigs are enabled
  color_resolution: [1280, 720]
  fps: 30
  align_depth_to_color: true
```

Three things bite here. `serial_number: null` opens the first RealSense the SDK offers, and
`RealSenseRGBDStreamer` reads only the serial, never `device_index`, so a two-camera cell must give
each rig its own serial; the schema refuses two enabled RGB-D rigs where either lacks one, or where
both carry the same one. Two identical cameras that swap identity do not fail, they return a
complete and plausible scene with the views exchanged. And the real pick path opens the rig
`camera.cameras.primary_rig_id` names (`build_real_components` in `cells.py`), whose calibration is
the one a single-view pick uses, so point `primary_rig_id` at the camera the grasp should come from.

Validate the tree, then open the device without moving anything:

```bash
python -m src.config
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --dry-run
```

`--dry-run` runs the arm-vendor readiness gate, builds the arm alone with no gripper, opens exactly
that one rig through its `Camera` owner, and prints the arm, the cell lock the sweep will take,
whether the camera answered with intrinsics, and what the arm's safety pipeline refuses. Then it
stops, before any motion and without taking the lock. On a host without the arm vendor's SDK it
refuses at the build, as a pick run on the same tree does. For the detector and segmenter on
real frames with no robot:

```bash
python -m src.robot.perception --prompt "a red cube" --rig realsense_d435
```

## 3. The board

Print an ArUco marker. The dictionary and the edge length must match the configuration exactly.

- Measure the printed black square with calipers. Not the white border, and not what the PDF was
  called. A board declared 50 mm and printed at 48 mm scales every sample uniformly. The solve
  converges, reports a plausible residual, and is wrong everywhere.
- Mount it flat and rigid. A board that flexes or shifts ruins every sample after it moves.
- Eye-to-hand: fix it to the tool so it cannot move relative to the TCP, and so it faces the camera
  all through the sweep. The poses are tool-down with at least 30 degrees of spread, so a board
  bolted flat to a downward-pointing flange can end up facing away from an overhead camera. Jog the
  extremes first.
- Eye-in-hand: fix the board in the workspace where the wrist camera can see it from many
  viewpoints.
- Diffuse light, no glare, steady exposure, marker in focus and fully in frame.

Rotational diversity is not a nicety. `BaseEyeHandCalibrator._assert_ready` in
`src/calibration/eye_hand/common.py` refuses a sample set whose robot rotations do not span at least
two independent axes.

## 4. Configure

[`config/camera/hand_eye.yaml`](../config/camera/hand_eye.yaml) holds the board spec and the sample
filter:

```yaml
hand_eye:
  eye_to_hand:
    marker_length_mm: 50.0          # what you measured
    aruco_dict_name: "DICT_5X5_100" # simulator runners only, see below
    min_samples: 6                  # hard floor for the solve
    min_distance_mm: 40.0           # a new sample must beat one of these two thresholds
    min_angle: 10.0                 # against every stored sample. Degrees.
```

**The real-cell runner does not read `aruco_dict_name`.** It reads `--dict`, whose default is the
hard-coded `DICT_5X5_100`. That YAML key has exactly two consumers and both are simulator runners.
So if you printed a board from any other dictionary, set it on the command line with
`--dict DICT_4X4_50`. Setting it only in YAML gives you zero detections and a "too few samples"
refusal, while the configuration on screen and the runner's own banner disagree about which
dictionary is in use.

Two further notes on that file. Its `enabled` flags are documentation only, and the file says so:
nothing reads them, they record which mounting the cell intends. And the real-cell runner reads the
`eye_to_hand` block for both modes, so a wrist-camera cell either edits that block or passes
`--marker-length-mm`. Of the five collection keys, only `marker_length_mm` has a command-line
override; `min_samples`, `min_distance_mm` and `min_angle` are read from YAML alone, and
`aruco_dict_name` is not read here at all.

`robot.calibration` in [`config/robot/robot.yaml`](../config/robot/robot.yaml) holds `settle_time_s`
(0.5 s), `orientation_spread_deg` (15.0), `max_attempts_per_pose` (200) and `quality_bands_mm`. The
runner raises the spread to at least 30 degrees on its own, because a planar ArUco viewed
near-frontally has an IPPE flip ambiguity that wrecks the AX=XB rotation.

`robot.workspace_limits` decides where the arm goes. `PoseProvider.generate` samples positions
inside that box and the workspace guard gates every pose. Shrink the box to a region that is safe and
where the marker stays visible before you run a sweep.

## 5. Rehearse, then run the sweep

```bash
python scripts/examples/api/02_calibration/calibrate_fixed_camera.py --rig realsense_d435       # the guided walkthrough
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --check
```

`--check` validates the configuration and the rig and touches no hardware. The example wraps that
same check and then tells you what a live run would do. Neither moves the robot. Both use the rig id
you declared in section 2, so on a tree that has no such rig they refuse by name and list the rigs
that do exist.

The sweep does move. Clear the cell, keep hands out, keep the emergency stop in reach.

```bash
python -m src.robot.execution.real_cell.calibrate \
    --rig realsense_d435 --mode eye_to_hand --poses 22 --marker-length-mm 49.6
```

It takes the cell lock before the arm is commanded. That is the lock a pick run and the operator
console take for the same controller, so while either holds the cell the sweep exits `1` and names
the holder: end the console's session first. Then it connects the arm alone. No gripper is built or
activated, so a Robotiq does not run its activation stroke beside the board. On the way out the arm
comes down and the lock is given back before the camera is, and the teardown is printed.

Per pose: move, settle, read the actual TCP pose with `arm.get_tcp_pose()`, grab a frame, detect the
marker, offer the pair to the calibrator. A sample is kept only if it beats `min_distance_mm` or
`min_angle` against every stored sample, and a refused move or a missing marker drops it too. Every
move declines the camera world, because the sweep is what produces the transform a camera world
needs: on a cuRobo cell each move's result says `DECLINED` with the mounting's reason, where an
undeclined move is refused before planning with `MISSING`.

**On a UR, `get_tcp_pose()` is the controller's reported actual TCP, not a pose this code derives.**
So the TCP offset configured in PolyScope determines every `A_i` the solver sees. Get that offset
wrong on the bench and the calibration converges cleanly onto the wrong answer: the residual looks
healthy because it is internally consistent, and nothing downstream can tell. Verify the offset on
the controller before the first pose, not after a bad result.

Plan well above `min_samples`. The runner writes `eth_<rig_id>.json` (or `eih_<rig_id>.json`) plus
the sample dataset under `calibration/real` unless `--out` says otherwise, and prints the rig block
that declares it in the camera section.

**That block is the `extrinsics` of the rig you declared in section 2.** Copy its `extrinsics:` lines
under that rig's entry rather than pasting a second `- rig_id: realsense_d435`, because a duplicate
rig id is a load error. After an `eye_in_hand` sweep the block ends in two commented lines,
`shutter_motion_tolerance_mm` and `shutter_motion_tolerance_deg`: measure them and write them,
because the loader refuses an `eye_in_hand` block without both and there is no default. The block
alone does not fuse a second camera; section 7 says what does.

Exit codes: `0` done, `1` configuration or build refused, the cell held by another process, or the
connect refused, `2` it ran but wrote no artifact, `3` the sweep raised. Exit `2` is loud on purpose,
because the cell then keeps whatever calibration it had.

## 6. Read the residual honestly

**`rmse_mm` is not millimetres.** It is the mean Frobenius norm `mean ||A_i X - X B_i||_F`, which
mixes the dimensionless rotation block with millimetre translation terms and scales with the solved
translation. It is a self-consistency score, not a distance. `src/calibration/quality.py` says so,
and it is measurable:

```bash
python docs/guide/snippets/rmse_sensitivity.py
```

On an exact, noise-free problem the script perturbs the true answer by 1 mm and by 1 degree in turn
and prints both residuals. The rotation error scores several times worse than the translation error
at a camera standoff of about a metre. Read the number as "the poses agree with each other", nothing
more.

The bands are inclusive upper bounds. The label the runner prints uses your configured
`robot.calibration.quality_bands_mm`; the label stored inside the artifact comes from the defaults in
the calibration package. Both hold the same three numbers as shipped.

| Label | Shipped bound | What to do |
|---|---|---|
| `excellent` | `<= 1.0` | ship it once a physical check agrees |
| `good` | `<= 2.5` | fine for a first bring-up, confirm physically before production |
| `marginal` | `<= 5.0` | investigate before shipping, see section 8 |
| `poor` | anything larger | reject, something is structurally wrong |
| `unknown` | non-finite or negative | treat as a defect, not as a bad calibration |

**Nothing gates on this label.** The solve builds the result regardless, `save_extrinsics` is a call
you make, and neither `load_extrinsics` nor `build_config_frame_resolvers` inspects quality. Gating
is your job. The refusals you will meet first are too few samples, rotations spanning fewer than two
axes, and wrong frames. For the full set:

```bash
grep -rn "raise CalibrationDataError\|raise ExtrinsicsError" src
```

## 7. Wire the artifact into the pick path

Writing the file is half the job. Declare it on the rig, in the camera section:

```yaml
camera:
  cameras:
    primary_rig_id: realsense_d435
    rigs:
      - rig_id: realsense_d435
        # ... the rest of the rig from section 2
        extrinsics:
          mounting_mode: eye_to_hand
          artifact_path: calibration/real/eth_realsense_d435.json
```

That alone gives a one-camera cell its `CAMERA->BASE`. The primary rig's `extrinsics` is read
whether or not `grasping.fusion.enabled` is on, and it is what the real-cell preflight's
`camera -> base` row and `from_robot_config` look for. `build_real_cell` hands the camera section to
`from_robot_config`, and a real cell built without one is refused, naming
`camera.cameras.rigs[<primary rig id>].extrinsics`.

On a cell that plans with cuRobo, the shipped planner, a declared calibration also makes the live
camera world mandatory. Every enabled RGB-D rig that declares `extrinsics` feeds it, so enable
`safety.planning_world` with a measured `support_plane` and `perceived.enabled` in the same step,
with the primary rig among the calibrated ones. Without that world the build is refused, naming the
calibrated rigs, and the `camera world` row of `real_cell --check` blocks.

A wrist camera is declared the same way, with its `eih_<rig_id>.json` and the two tolerances the
calibration command prints as comments:

```yaml
        extrinsics:
          mounting_mode: eye_in_hand
          artifact_path: calibration/real/eih_wrist.json
          # shutter_motion_tolerance_mm: <measure: how far the tool may travel while a frame is taken>
          # shutter_motion_tolerance_deg: <measure: how far the tool may turn while a frame is taken>
```

Uncomment both with the values you measured. Each is required and above 0 on an `eye_in_hand` rig,
with no default, and refused on an `eye_to_hand` one. The resolver built from this block composes the
artifact with the live tool pose on every frame, so a wrist primary needs no `frame_resolver=` in
code.

A second camera takes its own rig with its own `extrinsics`, an entry naming it in `fusion.cameras`,
and two flags. The shipped worked example for a two-camera cell is `config/robot/robot.eth2.yaml`
with `config/camera/cam.eth2.yaml`, loaded as `WILLY_PROFILE=ur5e,eth2`. Its robot half, for a
primary camera `cam_left` and a second camera `cam_right`:

```yaml
robot:
  grasping:
    fusion:
      enabled: true              # the line that makes the per-camera map live
      geometry:
        enabled: true            # what makes a second view reach the grasp
      cameras:
        cam_right:
          enabled: true          # keyed by rig id; the calibration stays on the rig
```

`fusion.cameras` names which rigs are fused and holds nothing else, and an id in it that names no
rig in `camera.cameras.rigs` is refused at load. `fusion.enabled` is the switch the per-camera map
hangs off: with it `false` the map builder returns nothing and your whole `cameras` block is ignored
without a word. The map then reaches the pick loop only when `fusion.geometry.enabled` is true as
well.

**The primary camera may be listed in `fusion.cameras`.** `_build_multi_camera_rig` in
`src/robot/execution/autonomous_grasp/cells.py` leaves it out of the extra-camera rig, because the
primary already streams through the main perception source, and the orchestrator's
`configured_camera_ids`, the cameras a pick waits for, leaves it out as well. Both are decided in code
from `camera.cameras.primary_rig_id`, so listing the primary costs nothing, and a two-camera cell can
name both of its cameras.

Turning `fusion.enabled` on does one thing the name does not advertise. Alongside the resolvers it
constructs the multi-view voxel substrate, which then ingests each view on every pick. Nothing reads
its output in the default path, because the gate that would let fused evidence decide is
`fusion.commit_policy.enabled` and it stays off, so it costs time and buys nothing until that gate
is on.

Three consequences of how the multi-camera path is built. Each camera named in the map must also be
an RGB-D rig in `camera.cameras.rigs` that declares its `extrinsics`: an id that names no rig is
refused at load, and a rig that is not RGB-D or declares no calibration is refused at build time,
rather than warned about on every pick. The two perception models are loaded once and shared, so a four-camera
cell costs four inference passes and one set of weights. And the counter that warns about a
single-view cell reads the same map, so it reports one calibrated camera and recommends adding a
second while the cell is in fact fusing two.

A declared artifact that is missing, stale or invalid raises at construction, naming its rig key,
rather than degrading to one view in silence. That holds for the primary rig and for every enabled
fused camera.

This multi-camera path has never run on hardware. `build_real_components` opens each other enabled
camera in `fusion.cameras` through its `Camera` owner, and the wiring has been proven against fake
cameras and fake models; no physical camera has been opened by it.

## 8. Verify, and what to do when it is bad

A residual proves the poses agree with each other. Only a pick proves the frame is right.

```bash
python -m src.robot.execution.real_cell --check      # the checklist, touches nothing
python -m src.robot.execution.real_cell --dry-run    # build only, no motion
python -m src.robot.execution.real_cell --runs 1 --prompt "a red cube"
```

Aim at a known object and measure where the TCP actually lands.

| Symptom | Likely cause | What to do |
|---|---|---|
| Every pick misses by the same offset in the same direction | the extrinsic itself | recalibrate, re-measuring the board first |
| Picks miss randomly | detector or depth, not calibration | run the perception exerciser on real frames |
| Residual is `marginal` or `poor` | marker size, flexing mount, motion blur, too few tilts, wrong dictionary | fix the physical cause, do not add poses to average it away |
| Far fewer accepted samples than poses | marker not found, or poses too similar | check lighting and framing, widen the pose spread |
| Solve refuses with "two independent axes" | the sweep only rotated about one axis | raise `orientation_spread_deg` above 30 and tilt in more directions. The runner floors it at 30, so a smaller value changes nothing |
| Cell refuses every motion, or still behaves single-view | the primary rig declares no `extrinsics`, or `fusion.enabled` still false | section 7, then `real_cell --check` |
| Cell refuses to build once a rig is calibrated | a cuRobo cell with a calibrated RGB-D rig and `safety.planning_world` off or incomplete | section 7, then `real_cell --check` and its `camera world` row |

## 9. When to recalibrate

| Trigger | Why |
|---|---|
| Any collision involving the camera, its mount, or the tool | assume the geometry moved |
| Tool or gripper change | eye-in-hand above all: `CAMERA -> TOOL` is stale even though the camera did not move |
| Camera moved, re-mounted, or re-focused | true for both modes |
| Robot base moved or re-anchored | eye-to-hand only, and it invalidates everything |
| Picks start missing by a constant offset | the signature of a bad extrinsic |
| A different physical board, or a planned shift change | a new target is itself a calibration change |

Be honest about drift detection. The watchdog carries `hand_eye_residual_trend` thresholds, but only
the synthetic replay datasets fill the field it reads, and it needs `grasping.watchdog.mode` plus
`grasping.decision.enabled` on. The watchdog itself is already on: `mode` defaults to `shadow`,
which evaluates and emits telemetry, and only `disabled` turns it off. `decision.enabled` is the gate
that is false, and the watchdog is built inside the decision path. Your drift signal is picks landing
off target.

Keep the old calibration before you overwrite it. The runner writes atomically, but there is no
archive and no rollback:

```powershell
Copy-Item -Recurse calibration\real "calibration\real.bak.$(Get-Date -Format yyyyMMdd-HHmmss)"
```

If the new calibration does not verify, restore that copy and stop. Do not run a production mode on
a calibration that failed a physical check.

## See also

- [calibration package](../src/calibration/README.md)
- [eye-hand workflows](../src/calibration/eye_hand/README.md)
- [real_cell](../src/robot/execution/real_cell/README.md)
- [camera package](../src/camera/README.md)
- [guide chapter 03](guide/03-calibration.md)
- [first pick](runbooks/real_cell_first_pick.md)
- [UR3e bring-up](runbooks/ur3e_cell_bringup.md)
