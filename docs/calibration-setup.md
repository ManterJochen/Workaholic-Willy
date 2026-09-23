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

The target is one printed ArUco marker, or a ChArUco board: a chessboard with an ArUco marker in
every white square. The board is posed from all the corners it shows at once, so it has none of a
single marker's flip ambiguity near a frontal view and keeps its pose when a corner is hidden. The
dictionary, the edges and, for a marker, its id must match the configuration exactly.

- Measure the printed black square with calipers. Not the white border, and not what the PDF was
  called. A board declared 50 mm and printed at 48 mm scales every sample uniformly. The solve
  converges, reports a plausible residual, and is wrong everywhere. On a ChArUco board measure both
  the chessboard square and the marker inside it.
- Do not point a single-marker sweep at a ChArUco board. It poses one of the board's markers at the
  configured edge, and every sample is scaled by square over marker edge. Measured on the stereo
  board `config/camera/cam.yaml` describes (10x7, 26/20 mm, `DICT_5X5_100`, which carries id 0): a
  board 292 mm away was posed at 728 mm with a 50 mm edge and counted. The sweep now says
  `N markers visible: is this a ChArUco board?` on such a pose, and the board belongs in `target`.
- A ChArUco board generated before OpenCV 4.6 (or by a tool that kept that layout) differs for an
  even number of rows: read in the other layout it shows its markers and no corners, and the sweep
  says `legacy_pattern` on every such pose.
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
    aruco_dict_name: "DICT_5X5_100" # the printed board's dictionary; --dict overrides it
    marker_id: 0                    # the marker's id, one the dictionary holds; --marker-id overrides it
    min_samples: 6                  # hard floor for the solve
    min_distance_mm: 40.0           # a new sample must beat one of these two thresholds
    min_angle: 10.0                 # against every stored sample. Degrees.
```

A ChArUco board is named in full, under the same block:

```yaml
    target:
      kind: charuco
      squares_x: 7
      squares_y: 5
      square_length_mm: 30.0        # measured
      marker_length_mm: 22.0        # the marker inside a square, measured
      legacy_pattern: false         # true for a board generated before OpenCV 4.6
```

Beside a board, `marker_length_mm` and `marker_id` are not read, and the board takes the block's
`aruco_dict_name` unless it names its own; a dictionary written in both places must agree. For one
run, `--board charuco:7x5:30:22` (or `aruco:ID:SIZE_MM`, each with an optional `:DICT`, and
`:legacy` for a board) or `--board-file board.yaml` (one such mapping, YAML or JSON) names the
target instead, and neither combines with `--marker-length-mm`, `--marker-id` or `--dict`.
Every key is in [`config/all_keys/camera/hand_eye.yaml`](../config/all_keys/camera/hand_eye.yaml).

The real-cell runner reads the target from the block of the mode it sweeps, and the three
single-marker flags override it for one run. `--check` validates the target it will use against the
config schema, so a dictionary OpenCV does not know, an edge that is not above zero, an id the
dictionary does not hold, or a board whose marker does not fit its square is refused before anything
is built. A dictionary is read the way OpenCV resolves it, case and the `DICT_` prefix optional, so
`--dict 5x5_100` is `DICT_5X5_100`. Its banner prints the target in use, so a board printed from
another dictionary is set once, in YAML, and the configuration and the banner agree.

Two further notes on that file. Its `enabled` flags are documentation only, and the file says so:
nothing reads them, they record which mounting the cell intends. And the real-cell runner reads the
block of the mode it sweeps, `eye_in_hand` for a wrist camera (`HandEyeCalibration`, in
`src/robot/execution/hand_eye.py`). `min_samples`, `min_distance_mm` and `min_angle` are read from
YAML alone. The Isaac runners render one ArUco marker, so they refuse a board `target` in one
sentence.

`robot.calibration` in [`config/robot/robot.yaml`](../config/robot/robot.yaml) holds `settle_time_s`
(0.5 s), `orientation_spread_deg` (15.0), `max_attempts_per_pose` (200) and `quality_bands_mm`. The
runner raises the spread to at least 30 degrees on its own, because a planar ArUco viewed
near-frontally has an IPPE flip ambiguity that wrecks the AX=XB rotation.

`robot.workspace_limits` decides where the arm goes. `PoseProvider.generate` samples positions
inside that box and the workspace guard gates every pose. Shrink the box to a region that is safe and
where the marker stays visible before you run a sweep.

**Until the sweep has run, the rig has no `extrinsics` block at all.** Not a block with an empty
`artifact_path`, which the tree refuses to load, and so every program with it, the sweep included; and
not a block naming the file the sweep will write, which loads but makes `camera.calibration()` and every
camera world refuse until that file exists. Without the block the rig is simply uncalibrated: the camera
opens ([`examples/real_robot/06`](../examples/real_robot/06_open_a_camera.py) says so), the sweep runs,
and the sweep prints the block to paste in (section 5). One exception to "the sweep runs": a wrist
camera on a cell that reads geometry (a cuRobo planner, or the exact-mesh self-collision guard) must
declare its `body` first (the next subsection), and with no calibration yet to place that body from, its first sweep also
takes `--unmodelled-wrist-body "<reason>"`
([`examples/real_robot/09`](../examples/real_robot/09_calibrate_a_wrist_camera.py) does). The sweep's
`--check` names whichever is missing before the arm moves. A wrist block that already holds measured tolerances is
commented out rather than deleted, and after the sweep only its `artifact_path` changes.

### A wrist camera's body, and a bracket beside the gripper

The planner and the guard see a wrist camera only through its rig's `body`: the housing of a camera
model from [`config/cameras/`](../config/cameras/), a bracket, and a margin. A cuRobo cell refuses to
sweep an enabled `eye_in_hand` rig without one, because the camera and its bracket would be parts of the
arm that no collision model sees. `--unmodelled-wrist-body` does not stand in for it: that reason lets a
declared body that nothing can place yet (no calibration) sweep without it, and a rig with no body is
refused whatever it says. Every key is in
[`config/all_keys/camera/cam.yaml`](../config/all_keys/camera/cam.yaml), whose example names a D435 and
a guessed bracket; do not copy those two.

Every box is written in the colour camera's optical frame, the frame the eye-in-hand solve places on
the flange, so measure from the colour lens:

- the origin is the centre of the colour lens, on its front glass;
- `z` points out of the lens, the way the camera looks; anything behind the lens is negative;
- `x` points to the right in the image: along the housing from the colour lens toward the housing's
  middle, which [`realsense_d415.yaml`](../config/cameras/realsense_d415.yaml) puts 35 mm away;
- `y` points down in the image: toward the housing's bottom face, the one with the tripod thread.

The D415 housing is already there: 99.15 x 23.15 x 20.15 mm, its middle at (35, 0, -8.975), so it runs
from x -14.6 to 84.6, y -11.6 to 11.6 and z -19.05 (its back face) to 1.1. `body.bracket` is one more
box, with faces along those same axes. A bracket arm that runs at 45 degrees to them, as one does from
the coupling to a camera tilted 45 degrees, is covered by the box round it: measure its farthest points
along each of the camera's axes, not along the arm. The box may run into the coupling and the hand it
is screwed to, because the camera is never checked against the flange, wrist 3, the hand or the
coupling plates; it is checked against wrist 1, wrist 2, a part in the hand and the cell.

A worked example, for a D415 beside the Hand-E, tilted 45 degrees outward: a 5 mm plate screwed to the
back of the housing, as wide as the camera, reaching 80 mm past the housing's top face (up in the
image), where a tab bends back 40 mm to bolt to the coupling. Along `x` the plate is the housing's width,
-14.6 to 84.6, so 99.2 wide about 35.0. Along `y` it runs from the housing's bottom face, 11.6, to 80 mm
past its top face, -11.6 - 80 = -91.6, so 103.2 about -40.0. Along `z` it starts at the back face, -19.05,
and the tab reaches 40 mm further back, to -59.05, so 40.0 about -39.05:

```yaml
- rig_id: wrist            # letters, digits and underscores: the planner's link is wrist_camera_wrist
  enabled: true
  source: rgbd
  rgbd_backend: realsense
  serial_number: "<the D415's serial>"
  color_resolution: [1280, 720]
  fps: 30
  align_depth_to_color: true
  body:
    model: realsense_d415  # config/cameras/realsense_d415.yaml: the housing from Intel's drawing
    margin_mm: 10.0        # a bracket measured with a ruler, and the USB cable tied along it
    bracket:
      size_mm: [99.2, 103.2, 40.0]
      centre_mm: [35.0, -40.0, -39.05]
```

Every box is grown by `margin_mm` on every side before a collision model reads it, so it covers the
cable, the drawing's tolerance and how well you measured; a larger margin refuses more poses near the
camera. Write `bracket: null` only for a camera held by nothing the planner has to model. Once the
sweep has run, the rig's `extrinsics` block also needs `record_tolerance_mm` and `record_tolerance_deg`
(section 5), because the body is placed from the flange to TCP the calibration recorded.

## 5. Rehearse, then run the sweep

```bash
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --check
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --dry-run
```

`--check` validates the configuration and the rig and touches no hardware. On a cuRobo UR cell that
declares `safety.self_collision.planner_margin_mm` it also looks up the committed planner evidence the
sweep's planner starts on, the lookup `real_cell --check` makes in its `planner margin` row: a
combination nobody measured is refused there, with the `scripts/curobo/matrix_gate.py` command that
measures it, instead of after the arm has connected. A coupling plate of a thickness no file measured
is one such combination, and so is a declared `safety.planning_world.payload.length_mm`, which reserves
16 attach slots and needs a file measured with them (`_a16`), where `payload.enabled: false` plans on the
committed `_a0` file. An undeclared margin is left to `real_cell --check`, which blocks on it with its own
fix, so run that first. `--dry-run` builds the arm and opens that one camera, then stops before any
motion; it starts no planner either. Neither moves the robot. Both use the
rig id you declared in section 2, so on a tree that has no such rig they refuse by name and list the
rigs that do exist. The same check, dry run and sweep from Python are
[`examples/real_robot/07_calibrate_a_fixed_camera.py`](../examples/real_robot/07_calibrate_a_fixed_camera.py),
whose last call is the sweep.

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
target, offer the pair to the calibrator. A sample is kept only if it beats `min_distance_mm` or
`min_angle` against every stored sample, and a refused move or a missing marker drops it too. A
wait for the arm to settle that ends with the arm still moving is waited once more, and after a
second one the pose is skipped as `not_steady` with no frame taken. Every move declines the camera
world, because the sweep is what produces the transform a camera world needs: on a cuRobo cell each
move's result says `DECLINED` with the mounting's reason, where an undeclined move is refused before
planning with `MISSING`.

The generated poses run in bearing order round the base, `atan2(y, x)`, starting at the end nearer
where the tool stands, rather than in the order they were drawn. In the order drawn, the base swings
back and forth across the box: on a 22-pose sweep planned with cuRobo on a UR10 descriptor (the
calibration chain's GPU probe, not a physical arm), the joints travelled 188.8 and 201.2 rad in the
order drawn and 77.1 and 80.0 rad in bearing order. The poses themselves are the same.

A refused move skips its pose and the sweep goes on, when nothing moved: a planner or a gate said
no. The sweep stops at a pose, commands nothing after it, keeps the samples it has in the dataset
and exits `3` naming the pose, when the arm may stand somewhere nobody judged or cannot be commanded
at all: a move that ended `connection_error`, a `controller_rejected` whose message says the command
had been sent (a protective stop in the middle of a move reads like this), or any failed move after
which the controller reports a protective or emergency stop. Clear the cause at the pendant and run
the sweep again.

Each pose prints as it happens, and the result stage lists every pose again:

```text
  pose  3/22 'look_2'  moving, 2 counted so far, need 6
  pose  3/22 'look_2'  target seen: 24 corners, 0.31 px, 531 mm away
  pose  3/22 'look_2'  COUNTED 3 (need 6)
  pose  4/22 'look_3'  REJECTED marker_not_found: marker 0 not in view; DICT_5X5_100 ids in view: [7]
  pose  5/22 'look_4'  REJECTED sample_rejected: pose not diverse: stored sample 2 is 12.4 mm and 3.1 deg away, ...
  pose  6/22 'look_5'  REJECTED move_rejected: workspace_rejected: ...
```

A missing target says what the camera did see: no marker of the dictionary, another id, markers of
another dictionary, or a board with too few corners. From Python the same lines come from
`on_event=print_sweep_progress` (`from willy import print_sweep_progress`), as examples 07 to 10 do.

### The preview window

Beside the printed lines, a window shows what the camera sees. While the arm moves it shows the
live view, labelled `LIVE, not judged`, with whatever of the target is in view drawn on it. When a
pose is judged, the window pins the frame the sweep judged for 1.5 seconds. It draws the markers
with their ids, a board's chessboard corners and the target's axes. Above the frame, a band says
`COUNTED` in green or `REJECTED` in red, with the reason the console prints. A row of boxes along
the bottom shows every pose of the sweep: green, red, or grey for not judged yet.

It opens with the sweep, once the arm is connected, and closes before the camera is given back.
`--check` and `--dry-run` open no window. By default it opens only where three things hold: OpenCV
has a GUI, a display is there, and stdout is a terminal. A piped or logged run therefore prints
exactly what it printed before. `--preview` asks for the window by name, and the build line
`preview  off: <why>` says when none can show. The usual reasons are an SSH session on Windows, no
`DISPLAY` or `WAYLAND_DISPLAY` on Linux, the headless OpenCV wheel, and macOS. `--no-preview`, or
`WILLY_NO_PREVIEW=1` in the environment, keeps it shut. From Python it is
`SweepOptions(preview="auto")`, as examples 07 to 10 do. The library opens no window unless asked.

**The window is not a stop.** Pressing ESC or closing it closes the window, and the sweep goes on:
the pendant stops the robot. On Windows, Ctrl+C typed while the window has the focus reaches the
window, not the console. It does not stop the program either, and the preview says so once. To stop
the program, click into the console first; to stop the robot, use the pendant.

The preview is display only. Its live frames come through the same camera owner as the sweep's own
grabs, at most five a second. It poses them with an estimator of its own, and nothing it sees reaches
the calibrator. The pose that counts is the one the sweep judged. A live grab can delay the judged
grab by at most one frame. If the window fails, the preview switches off with one printed line and
the sweep goes on. The window has been opened, drawn and closed on Windows with a synthetic camera.
It has not run on Linux or beside a physical camera.

**On a UR, `get_tcp_pose()` is the controller's reported actual TCP, not a pose this code derives.**
So the TCP offset configured in PolyScope determines every `A_i` the solver sees. Get that offset
wrong on the bench and the calibration converges cleanly onto the wrong answer: the residual looks
healthy because it is internally consistent, and nothing downstream can tell. Verify the offset on
the controller before the first pose, not after a bad result.

### When the generated sweep sees nothing

The generated poses are tool down, varying yaw and a small orientation spread about it, which is
the right shape for a camera looking down on a board that lies flat. It is the wrong shape twice.
A WRIST camera carried over a board on the table photographs the table beside it from anywhere but
straight overhead, and a board bolted to the FLANGE shows a fixed camera an edge once the arm is
off to one side. Yaw cannot fix either, because yaw about the vertical never tips anything toward
anything. The sweep then collects too few samples and the failure looks like a solver problem.

Aim the poses instead of tilting them. `Pose.aimed_at(x, y, z, target_mm=...)` points the tool's +Z
at a point you name, so a ring of stations around a board all see it, and
`SweepOptions(fixed_poses=[...])` runs exactly those poses in order through the same check, dry run
and sweep. That suits a wrist camera that looks along the tool axis.
[`examples/real_robot/08`](../examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py)
turns the flange board to face a fixed camera, and for that case you have to say roughly where the
camera hangs, which is what the sweep is about to measure: a tape measure is accurate enough,
because the aim only has to bring the board into frame. A wrist camera tilted off the tool axis, or
on a bracket beside the hand, does not look where the tool points: aim the camera itself, as the next
section and [`examples/real_robot/10`](../examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py)
do.

The aim is not a reachability claim. The arm's guards still judge every pose, and the run reports
per pose whether the target was actually decoded, and if not, why; a station that saw nothing is a
station to move. The preview window shows what that station saw.

On a ring of aimed poses, pass `closing_axis="tangential"` (or `"radial"`) to `Pose.aimed_at`, as
example 10's comment says. By default the roll follows the camera's bearing round the board: computed
with `Pose` for a six-station ring round a board at (500, 0, 0) mm (the ring example 10 built before
it aimed the camera), the tool's heading changes by 90 degrees from station to station and covers 270
degrees, and wrist 3 has to follow it. With `"tangential"` the roll follows the base
instead, and the same ring's headings stay within a 37 degree band. Planned with cuRobo on a UR10
descriptor (the calibration chain's GPU probe, not a physical arm), the default ring took 5 of its 6
legs the long way round, 58.4 rad of joint travel, and the tangential ring 11.8 rad.

### A camera tilted beside the hand: aim the camera, not the tool

`SweepOptions(aim=MarkerAim(marker_mm=(x, y, z)))`, or `--aim-at=x,y,z` with `--mode eye_in_hand`,
aims the CAMERA at one marker lying flat, face up, at that point in the base frame (millimetres; a few
centimetres of error is enough to aim, and the solve measures nothing from it). It needs one ArUco
marker as the target, named in full for one run with `--board aruco:ID:SIZE_MM[:DICT]` or
`SweepOptions(target=...)`; a ChArUco board is refused, because its pose sits at its corner and not
where the aim looks. [`examples/real_robot/10`](../examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py)
is this flow with a 150 mm `DICT_4X4_100` marker, id 50, at (-130, -700, 50) mm. What it does, in
order (`src/robot/execution/camera_aim.py`, `CalibrationRoutine.run_aimed`):

1. **One look, nothing moved.** Once the arm is connected, with the preview open, it waits for the
   arm to report steady, reads the TCP and takes one judged frame where the arm stands. So stand the
   arm first where the camera sees the marker, about the aim's distance from it and from one side
   rather than from straight above. A look that sees no marker ends the run with nothing moved, exit
   `3`, and the report says `NotAimed: ... Jog the arm ... then run this again`. The preview opens
   only with the sweep, so to watch the camera while you jog, build the cell in the operator console
   (`python -m api --profile <your cell>`, then Build; a build opens the cameras and moves nothing):
   its camera view, `GET /v1/camera` ([`api/viewfinder.py`](../api/viewfinder.py)), is a frame taken
   now, polled about six times a second. Stop the console before the sweep, which opens the camera
   itself. [`examples/real_robot/06`](../examples/real_robot/06_open_a_camera.py) takes one frame,
   where a still is enough.
2. **Where the camera sits on the tool, from that one view.** The marker's normal is base +Z and the
   line of sight runs to its centre; two such direction pairs fix the camera's rotation in the tool
   frame, taking the camera to stand at the flange at first, and the view then fixes its position. One
   view cannot tell where round the marker's vertical the camera stands, so the estimate takes the
   place nearest the flange: a camera whose bracket runs the way it looks is estimated within 1 to 3
   degrees, and one 80 mm to the side of that line is off by 11 to 13 degrees. What an aimed station
   inherits is much less, because that error is one the look itself cannot see: measured on synthetic
   geometry (2026-09-23), every station of the 17 rolled default views missed the marker by at most 1.1
   degrees for the first camera and 3.7 in the worst case tried (the marker stated 28 mm from where it
   lies), and by at most 1.2 after one more view. The colour image of a D415 is 69 by 42 degrees.
3. **Seventeen views round the marker** (`DEFAULT_VIEWS`): the camera `distance_mm` from it (500 by
   default), at the elevation and azimuth it has with the tool pointing down and at offsets of -12 to
   +30 degrees in elevation and up to 60 in azimuth from there: a ring of seven, five higher, three
   lower and two steep. Each station tilts the tool as little as it can from `Pose.tool_down` with the
   aim's heading (`closing_axis`, `--closing-axis`), and then rolls the camera about its own line of
   sight by 0, +45 or -45 degrees, the three in turn down the list: the camera stands where it stood,
   the marker stays where it was in the image, and the image turns about it. On a camera tilted 45
   degrees no station tilts more than about 41 degrees from its heading, and with the roll the tool
   stands at most 41 degrees from straight down (31 with the heading kept). The roll is what lets
   AX=XB measure the camera's turn about its own axis. With the heading kept, every turn between two
   views is about an axis across the line of sight, and over every pair of the 17 the weakest axis of
   those turns carried 1/70 of the strongest, 6 degrees from the optical axis. Measured on synthetic
   geometry (2026-09-23; the owner's marker, a camera 70 mm out and 45 degrees toward the tool's +X, the
   UR10 box less 20 mm, its reach and a half-turn window about an elbow-up home; each view's marker
   pose disturbed by 0.5 degrees and 1 mm, depth by 0.4 %; 40 seeds), a point 500 mm down the optical
   axis landed 5.0 mm off at the median (7.8 at the 90th percentile) with the heading kept, and 2.0
   (4.7) rolled; at half that noise 2.5 (3.9) against 1.0 (2.4), at twice it 8.9 (12.8) against 4.0
   (9.2). The Hand-E leaves about 5 mm a side on a 40 mm part. A roll the screen refuses gives way to
   the view unrolled, so a roll never costs a view. The first set of eleven (offsets up to 15 and 40
   degrees, no roll) counted fewer samples: with PnP-like noise on the synthetic camera, 8.3 and 9.0
   on average from the two homes below against 11.7 and 15.0 for the seventeen with the heading kept,
   and the solve's rotation error fell from about 1.1 to 0.6-0.8 degrees. Pass `MarkerAim(views=...)`
   for a set of your own: `(elevation, azimuth)` or `(elevation, azimuth, roll)`, in degrees.
4. **A heading that admits the views.** Which way the camera is tilted on the tool decides where it
   has to stand to look at the marker. Under a heading written for a camera tilted toward the tool's
   +X, one tilted toward -X looks away from the marker and has to stand beyond it: for the owner's
   marker, 40 mm inside the box's edge at y -700, all 17 views then fall outside the box and nothing
   moves. So before any station, the sweep counts the views the aim's `closing_axis` admits after the
   screen below. Where that is fewer than the solve's `min_samples`, it counts them for `-y`, `y`, `x`
   and `-x` too and keeps the one that admits the most, the one asked for winning a tie. The log and the
   result stage's `heading` line say which, with every count: `heading 'y', not the '-y' asked: '-y'
   admits 0 of the 17 views, fewer than the 6 the solve needs; 'y' admits the most (-y 0, y 17, x 10,
   -x 11)`. A heading that admits enough is kept, even where another would admit more. Where none admits
   enough, the sweep still moves to what it can reach, and the solve's refusal of too few samples comes
   with every heading's count on that line. The viewing pose of examples 11, 13 and 18 falls back the
   same way where its heading admits no view, and says so; write the heading this sweep kept into
   their `CLOSING_AXIS` and they keep it without a word.
5. **Screened before anything is commanded.** A view whose TCP or flange lies outside
   `workspace_limits` less `safety.limits.workspace_margin_mm` (the margin the arm's own gate keeps),
   whose flange no joint configuration reaches, or whose every configuration lies outside the joint
   window the arm chooses a goal within (half a turn either side of home on a cell that sets
   `safety.joint_limits.within_half_turn_of_home`, so a cable along the arm is never wound further), is
   reported with its reason (`outside_workspace`, `out_of_reach`, `outside_joint_window`,
   `too_tilted`) and never moved to. The rest are ordered by joint travel from where the arm stands,
   each next the one the arm's nearest-goal choice reaches with the smallest turn. That needs the
   arm's kinematic model and tool frame (a UR); without them they are ordered by the tool's travel.
6. **Re-aimed as it sees.** Every view the marker is posed in refines the estimate (a small
   least-squares fit over every view so far, the marker held flat, its measured position a weak prior)
   and re-aims the views still to come. A refinement the views contradict, a marker that moved or does
   not lie flat, is not taken and says so in the log.

Measured on 2026-09-23 with the real UR10 driver and the real cuRobo sidecar on this repository's GPU
PC (a Hand-E on the shipped 20 mm plate, a `willy` tool frame 156.2 mm along the flange, a 4 mm planner
margin, the half-turn window about a home at the first look), a simulated controller that stands
where each `moveJ` puts it, and a synthetic camera 70 mm out and 45 degrees toward the tool's +X
looking at the owner's marker: from an elbow-up home 9 of the 11 views ran, in 20 `moveJ` and 15.5 rad
of joint travel; from an elbow-down home 10 of 11, in 40.8 rad. The views that did not run were
lower-ring views: every configuration near them folds the forearm onto wrist 2 in the planner's model,
and cuRobo's own choice then ended outside the window (refused, `joint_limit_rejected`) or found no
collision-free plan (`timeout`). On the synthetic frames the solve returned the mount exactly. No
physical arm or camera was involved. Expect a view or two of the lower ring to be refused on a cell
whose marker lies near the edge of the arm's reach; the shipped `min_samples` of 6 leaves room for
that, and the refused view is reported with its reason.

Every station is still a pose the arm plans and judges as it moves; nothing here makes a move that
was not judged. The first look is not a sample. The result stage prints what the stations were aimed
from (`aimed from` the first look, `re-aimed from` the last refinement) and the heading they kept
(`heading`), and the pose table lists every view: counted, or why not.

Which way the camera is tilted on the tool is not guessed. To aim from a mount you state when the
first look sees nothing, pass `MarkerAim(..., mount_if_unseen=nominal_camera_in_tool(45.0,
toward="+x", offset_mm=(x, y, z)))`: the optical axis turned 45 degrees from the tool's +Z toward the
tool's +X, the axis the jaws close along, which on a `closing_axis="-y"` pose is base -Y. `"-x"` tilts
it the other way, `"+y"` and `"-y"` toward the tool's other axis. The first station that sees the
marker replaces the stated mount with its own estimate. The heading (item 4) is counted from the stated
mount as it would be from a look.

Plan well above `min_samples`. The runner writes `eth_<rig_id>.json` (or `eih_<rig_id>.json`) plus
the sample dataset under `calibration/real` unless `--out` says otherwise, and prints the rig block
that declares it in the camera section.

**That block is the `extrinsics` of the rig you declared in section 2.** Copy its `extrinsics:` lines
under that rig's entry rather than pasting a second `- rig_id: realsense_d435`, because a duplicate
rig id is a load error. After an `eye_in_hand` sweep the block ends in four commented lines. Two,
`shutter_motion_tolerance_mm` and `shutter_motion_tolerance_deg`, the loader requires on every
`eye_in_hand` block. The other two, `record_tolerance_mm` and `record_tolerance_deg`, it requires on a
rig that declares a `body`, which a wrist camera on a cuRobo cell must: with only the first two filled
in, that tree does not load, and the refusal names the rig's `body`. None of the four has a default.
Section 7 says what each bounds and gives starting values. The block alone does not fuse a second
camera; section 7 says what does.

Exit codes: `0` done, `1` configuration or build refused, the cell held by another process, or the
connect refused, `2` it ran but wrote no artifact, `3` the sweep raised or stopped at a pose. Exit `2`
is loud on purpose, because the cell then keeps whatever calibration it had.

### Stations from a file, and stations as joint angles

`SweepOptions(fixed_poses="stations.json")`, or `--fixed-poses stations.json` on the command line,
runs the stations a JSON file lists, in the order it lists them. The file is a list, and each
record is one of:

```json
[
  {"label": "look_0", "joints_deg": [160.9, -105.2, 130.6, -115.4, -90.0, -19.1]},
  {"label": "look_1", "joints_rad": [2.887, -1.541, 2.042, -1.742, -1.655, -0.241]},
  {"label": "look_2", "x": 500.0, "y": -140.0, "z": 320.0, "rx": -1.6997, "ry": 2.2968, "rz": 0.4804}
]
```

A pose is millimetres and an axis-angle rotation in radians in the robot's base frame, as the pendant
shows a TCP pose, and the arm plans to it as to any pose. A joint station is six joint angles from
the base to the last wrist joint (on a UR the pendant's order: base, shoulder, elbow, wrist 1,
wrist 2, wrist 3), in degrees or in radians, as its key says. Teach a station on the pendant and
copy its joint angles: the arm then goes to exactly that configuration, whichever way a planner
would have solved the pose. [`examples/real_robot/eih_fixed_stations.json`](../examples/real_robot/eih_fixed_stations.json)
mixes both. Its numbers are that six-station ring, solved for a bare flange on one arm model: teach your
own. On a cell that keeps every joint within half a turn of home
(`safety.joint_limits.within_half_turn_of_home`, which the UR10 profile turns on), a taught station must
lie inside that window, less the guard's margin: the file's `look_2`, its base at 181.7 degrees, does for
a home whose base faces the same way (near 180 degrees) and not for the default home (base at 0), where
its move is refused naming the window.

`--check` reads the file and checks every record before anything is built: one unit key, no pose
keys beside it, six finite numbers, and units that can be what the key says. A `joints_rad` value
beyond one full turn (2 pi) is refused as probably degrees, a `joints_deg` value beyond 360 as
beyond any joint, and a `joints_deg` record whose every value lies within 2 pi as probably radians.
Nothing rewrites a value. The config stage then says how many stations the file holds and how many
are joint stations, and names each pair of neighbouring joint stations between which a joint turns
more than half a turn (the same angle a full turn away would be nearer, and the station still runs as
written):

```text
  poses      6 from stations.json, 4 of them joint stations
  !! 'look_1' to 'look_2': wrist 3 turns +270 deg, more than half a turn; it runs as written, the long way round
```

At the cell, each station is screened as the sweep reaches it, before it moves. A joint move does not
pass the workspace box, so for a joint station the sweep reads where its joints put the grasp centre
from the arm's own forward kinematics (on a UR, the controller's, with the tool frame, as the home
gate reads home) and boxes that. A station whose grasp centre lies outside `workspace_limits`, or
within both `min_distance_mm` and `min_angle` of a station kept before it, is reported with its
label and reason (`outside_workspace`, `too_similar`) and never moved to; so is a pose of the file,
and so is a joint station whose grasp centre the arm cannot place (`not_boxed`).
A joint station that passes is sent as one joint move to exactly the joints written. On a cuRobo UR
that move is the straight joint line from where the arm stands, judged by the path guard and by the
planner, and then run as one `moveJ` along that line, so the move that was judged is the move that
runs. It is not routed round an obstacle: a line either of them refuses skips the station. The first
leg starts wherever the arm stands, so start the sweep from home.

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

A wrist camera is declared the same way, with its `eih_<rig_id>.json` and the four tolerances the
calibration command prints as comments:

```yaml
        extrinsics:
          mounting_mode: eye_in_hand
          artifact_path: calibration/real/eih_wrist.json
          # shutter_motion_tolerance_mm: <measure: how far the tool may travel while a frame is taken>
          # shutter_motion_tolerance_deg: <measure: how far the tool may turn while a frame is taken>
          # record_tolerance_mm: <measure, when the rig declares a body: how far the connect derives the tool frame apart>
          # record_tolerance_deg: <measure, when the rig declares a body: the same in degrees>
```

Uncomment them with values for your cell. Each is above 0 and has no default, and all four are refused
on an `eye_to_hand` rig:

| Key | Required on | What it bounds | A starting value on a `willy` tool frame |
|---|---|---|---|
| `shutter_motion_tolerance_mm` | every `eye_in_hand` rig | how far the TCP may move between the reads taken either side of a grab before the frame is refused | 1.0 |
| `shutter_motion_tolerance_deg` | every `eye_in_hand` rig | the same, turned | 0.5 |
| `record_tolerance_mm` | a rig that declares a `body` | how far the flange to TCP the calibration recorded may lie from the one the cell holds now | 1.0 |
| `record_tolerance_deg` | a rig that declares a `body` | the same, turned | 0.5 |

The starting values are the ones the repository's tests use, not measurements. The first two bound a
settled arm across one grab: every refused frame is logged with the travel it measured (`tool moved
across the grab: ... mm, ... deg`), so raise them only as far as a still arm needs. On a `willy` tool
frame the record is written from the declared numbers and compared exactly, so any value above 0 of
the last two behaves the same. On a `polyscope` one the frame is derived from the controller at every
connect: connect a few times and set them just above the spread you see. The resolver built from this block composes the
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
| Far fewer accepted samples than poses | marker not found, or poses too similar | read the per-pose lines: `marker_not_found` says what the camera saw, `sample_rejected` names the stored pose it is too close to. The preview window shows the judged frames. Check lighting and framing, widen the pose spread |
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
- [cell bring-up](runbooks/cell_bringup.md)
