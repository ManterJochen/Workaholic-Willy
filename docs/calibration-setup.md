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

The shipped `config/robot/robot.yaml` declares no fusion block, so `grasping.fusion.enabled` is
`false` and no artifact path is set. That is the default state of a fresh cell. Check yours:

```bash
python -m src.config explain robot.grasping.fusion.enabled
```

## 1. Pick the mounting mode

| | Eye-to-hand | Eye-in-hand |
|---|---|---|
| Camera | fixed in the cell | on the wrist |
| Marker | rigid on the tool | fixed in the workspace |
| Solves for | `CAMERA -> BASE` | `CAMERA -> TOOL` |
| Written by | `save_extrinsics` | `save_cam_to_tool` |
| At runtime | used as it is | composed with the live TCP on every frame |

A multi-camera rig calibrates each camera on its own and declares them all in one map. Run this whole
procedure once per camera.

## 2. Connect the camera

The live path is an Intel RealSense RGB-D rig. `pip install -r requirements.txt` covers it, and
`pyrealsense2` is imported lazily, so a machine with no device attached is unaffected.

Declare the rig in [`config/camera/cam.yaml`](../config/camera/cam.yaml) under `cameras.rigs`. The
shipped tree already carries one such rig, `realsense_d435`, with `enabled: false`:

```yaml
- rig_id: realsense_d435   # this id keys the artifact AND the fusion map. Keep them equal.
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
complete and plausible scene with the views exchanged. And the real pick path takes the **first** rig
whose `source` is `rgbd` as the cell's primary camera, in list order and without consulting
`enabled`, so put the camera the single-view path should use first and leave no stale rig above it.

Validate the tree, then open the device without moving anything:

```bash
python -m src.config
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --dry-run
```

`--dry-run` builds the arm, opens exactly that one rig through `FrameProvider.rig`, prints whether
the camera answered with intrinsics, and stops before any motion. For the detector and segmenter on
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

Per pose: move, settle, read the actual TCP pose with `arm.get_tcp_pose()`, grab a frame, detect the
marker, offer the pair to the calibrator. A sample is kept only if it beats `min_distance_mm` or
`min_angle` against every stored sample, and a refused move or a missing marker drops it too.

**On a UR, `get_tcp_pose()` is the controller's reported actual TCP, not a pose this code derives.**
So the TCP offset configured in PolyScope determines every `A_i` the solver sees. Get that offset
wrong on the bench and the calibration converges cleanly onto the wrong answer: the residual looks
healthy because it is internally consistent, and nothing downstream can tell. Verify the offset on
the controller before the first pose, not after a bad result.

Plan well above `min_samples`. The runner writes `eth_<rig_id>.json` (or `eih_<rig_id>.json`) plus
the sample dataset under `calibration/real` unless `--out` says otherwise, and prints a YAML snippet.

**That snippet is the `cameras:` sub-block only.** It carries neither `fusion.enabled` nor
`fusion.geometry.enabled` nor the top-level `extrinsics_artifact_path`, and without them the map is
ignored without a word. Paste it, then add the rest from section 7.

Exit codes: `0` done, `1` configuration refused, `2` it ran but wrote no artifact, `3` unexpected.
Exit `2` is loud on purpose, because the cell then keeps whatever calibration it had.

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

Writing the file is half the job. The shipped worked example for a two-camera cell is
`config/robot/robot.eth2.yaml` with `config/camera/cam.eth2.yaml`, loaded as
`WILLY_PROFILE=ur5e,eth2`. Its shape, for a primary camera `cam_left` and a second camera
`cam_right`:

```yaml
robot:
  grasping:
    fusion:
      enabled: true                                                # the line that makes all of it live
      extrinsics_artifact_path: "calibration/real/eth_cam_left.json"   # the PRIMARY camera
      geometry:
        enabled: true                                              # what makes a second view reach the grasp
      cameras:
        cam_right:                                                 # every camera EXCEPT the primary
          enabled: true
          mounting_mode: eye_to_hand
          extrinsics_artifact_path: "calibration/real/eth_cam_right.json"
```

`fusion.enabled` is the switch everything hangs off. With it `false` the resolver builders return
nothing and your whole `cameras` block is ignored without a word. The map then reaches the pick loop
only when `fusion.geometry.enabled` is true as well. Set the single top-level
`extrinsics_artifact_path` even on a one-camera cell: it is the only key that satisfies the
`CAMERA->BASE` refusal in `from_robot_config`, and the snippet the calibration runner prints leaves
it null.

**The primary camera does not belong in `fusion.cameras`.** `_build_multi_camera_rig` in
`src/robot/execution/autonomous_grasp/cells.py` builds the extra-camera rig from that map with the
primary filtered out, because the primary already streams through the main perception source; but
the orchestrator's `configured_camera_ids` comes from the same map with only disabled entries
dropped, and the pick loop then reports every configured camera that delivered no frame. List the
primary there and it is permanently among the missing: a warning on every pick under
`on_camera_unavailable: degrade`, and a raised error on every pick under `refuse`. The Isaac
simulator tree lists all three of its cameras including its primary, which is correct there because
that runner builds its own rig; do not copy the shape of that block onto a real cell.

Turning `fusion.enabled` on does one thing the name does not advertise. Alongside the resolvers it
constructs the multi-view voxel substrate, which then ingests each view on every pick. Nothing reads
its output in the default path, because the gate that would let fused evidence decide is
`fusion.commit_policy.enabled` and it stays off, so it costs time and buys nothing until that gate
is on.

Three consequences of how the multi-camera path is built. Each camera named in the map must also be
an RGB-D rig in `camera.cameras.rigs`, and one that is not is refused at build time rather than
warned about on every pick. The two perception models are loaded once and shared, so a four-camera
cell costs four inference passes and one set of weights. And the counter that warns about a
single-view cell reads the same map, so it reports one calibrated camera and recommends adding a
second while the cell is in fact fusing two.

An enabled camera whose artifact is missing, stale or invalid raises at construction rather than
degrading to one view in silence. An eye-in-hand cell leaves the path unset and passes
`frame_resolver=` in code, because a live TCP-composed resolver cannot be serialized.

This multi-camera path has never run on hardware. The wiring has been proven against a fake provider
and fake models; no physical camera has been opened by it.

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
| Cell refuses every motion, or still behaves single-view | no resolver, or `fusion.enabled` still false | section 7, then `real_cell --check` |

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
