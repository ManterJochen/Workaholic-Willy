# 3. Hand-eye calibration: the derivations

This page is for reasoning about a calibration result, not only producing one: why the problem is
`AX = XB`, what the two mounting cases solve for, what the residual measures (it is not
millimetres), how the CAMERA to BASE transform reaches a grasp, and how each failure shows itself.

A calibration is run by `HandEyeCalibration`, one camera at a time:

```python
from willy import HandEyeCalibration, SweepOptions, load_tree

calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="overhead", mode="eye_to_hand",
                                           options=SweepOptions(marker_length_mm=40.0))
print(calibration.check())  # the config alone: rig, marker, poses, artifact path; opens nothing
```

`calibration.run(dry_run=True)` then builds the arm and opens the camera without moving, and
`calibration.run()` sweeps the arm. The whole flow is
[`examples/real_robot/07_calibrate_a_fixed_camera.py`](../../examples/real_robot/07_calibrate_a_fixed_camera.py)
for a fixed camera and
[`examples/real_robot/08_calibrate_a_wrist_camera.py`](../../examples/real_robot/08_calibrate_a_wrist_camera.py)
for a wrist camera. The command line is `python -m src.robot.execution.real_cell.calibrate`
([docs/cli.md](../cli.md)); in simulation it is `run_eth_calibrate` and `run_eih_calibrate`. The bench
session (print the board, run the sweep, wire the artifact in) is
[docs/calibration-setup.md](../calibration-setup.md). The package references are
[`src/calibration/README.md`](../../src/calibration/README.md) and
[`src/calibration/eye_hand/README.md`](../../src/calibration/eye_hand/README.md).

---

## 1. What calibration is for

The grasp stack computes and commands motion in the robot **BASE** frame. Perception produces depth
and segmentation in the **CAMERA** frame. Exactly one rigid transform sits between them, and
calibration measures it.

Get it wrong and every candidate is confidently and consistently wrong. That is worse than no
candidates, because a wrong transform produces plausible poses the safety stack will accept.

### 1.1 Eye-to-hand and eye-in-hand

| | Eye-to-hand | Eye-in-hand |
|---|---|---|
| Camera is | bolted to the world: post, gantry, ceiling | bolted to the wrist, moves with the arm |
| Marker is | bolted to the tool flange | bolted to the world: table, fixture |
| Solves for | `Transform(CAMERA -> BASE)` | `Transform(CAMERA -> TOOL)` |
| Valid until | the camera or the robot base moves | the mount or the TCP definition changes |
| Runtime consumer | `StaticCameraToBaseResolver`: the same transform every call | `EyeInHandFrameResolver`: reads `arm.get_tcp_pose()` every frame and composes |
| Persisted as | `Extrinsics`, schema `willy.calibration.extrinsics/1` | a frame-tagged `Transform`, schema `willy.calibration.cam_to_tool/1` or `/2` |

Both resolvers live in
[`src/robot/grasping/motion/frame_resolver.py`](../../src/robot/grasping/motion/frame_resolver.py).

**Which one is right.** Eye-to-hand when the scene is fixed and the transform must be valid before the
arm has moved: static overhead bin picking, or any cell where the camera has to see the workspace
while the arm is elsewhere. Eye-in-hand when the view has to come from where the tool is going:
close-range refinement, or deep bins nothing static can see into. Eye-in-hand costs a live TCP read
every frame and a recalibration on every gripper change. Many cells want both.

An eye-in-hand result cannot be saved as `Extrinsics`:
[`src/calibration/eye_hand/types.py`](../../src/calibration/eye_hand/types.py) raises
`"only eye_to_hand results can be saved as Extrinsics"` in `to_extrinsics`. The reason is the
composition:

```
T_cam_to_base(t)  =  T_cam_to_tool  o  T_tool_to_base(t)
                     the calibration   the live TCP read, this instant
```

Half of an eye-in-hand camera-to-base transform does not exist until the arm is somewhere, so there
is nothing to serialize.

**The chain, end to end.** `CalibrationRoutine` collects samples and hands them to
`EyeToHandCalibrator` or `EyeInHandCalibrator`, which build the `A` and `B` pairs and call
`HandEyeAXXB`. The result is saved, the camera's rig in the config points at that file, a resolver
builder loads it into a `FrameResolver`, and the pick loop uses that resolver to carry masked depth
pixels into the BASE frame, where candidates are generated. Section 4 covers the keys and the
builders.

---

## 2. Why it is AX = XB

### 2.1 The eye-to-hand derivation

Fixed camera, marker rigid on the tool. For every pose `i`:

```
cam_T_marker_i  =  cam_T_base . base_T_tool_i . tool_T_marker
```

Take consecutive relative motions. `HandEyeAXXB.relative_motions` in
[`src/calibration/solver/hand_eye_axxb.py`](../../src/calibration/solver/hand_eye_axxb.py) maps
`[T_0, T_1, ...]` to `[T_1 @ inv(T_0), T_2 @ inv(T_1), ...]`, so:

```
A_i = base_T_tool_{i+1} . inv(base_T_tool_i)      relative gripper motion
B_i = cam_T_marker_{i+1} . inv(cam_T_marker_i)    relative marker motion
```

`tool_T_marker` cancels exactly, leaving `B_i = cam_T_base . A_i . base_T_cam`, which is

```
A_i . X  =  X . B_i        with  X = base_T_cam   (the code calls it T_cam_to_base)
```

**The consequence you can rely on:** the marker-on-flange offset `tool_T_marker` is never needed and
never solved for. Where on the flange the board is bolted does not matter. Every public `run_*`
method on `CalibrationRoutine` accepts a `T_tool_to_marker` argument and never uses it: pass `None`.
Both `A` and `B` come from the same helper in
[`src/calibration/eye_hand/eye_to_hand/calibrator.py`](../../src/calibration/eye_hand/eye_to_hand/calibrator.py).

### 2.2 The eye-in-hand derivation, and why the algebra inverts

Camera on the tool, marker fixed in the world. Let `M_i = base_T_tool_i`, `C_i = cam_T_marker_i` and
`X = tool_T_cam`. Constancy of `base_T_marker` gives `M_i . X . C_i = M_{i+1} . X . C_{i+1}`, which
rearranges to `A_i . X = X . B_i` with

```
A_i = inv(M_{i+1}) . M_i      local (left) relative tool motion, inverted order
B_i = C_{i+1} . inv(C_i)      global (right) relative marker motion
```

Only the `B` side comes from `relative_motions`. `A` is built by hand in
[`src/calibration/eye_hand/eye_in_hand/calibrator.py`](../../src/calibration/eye_hand/eye_in_hand/calibrator.py).
On the tool side eye-in-hand takes `inv(next) . prev` where eye-to-hand takes `next . inv(prev)`, and
the difference matters. Swap the two sides and reverse the index order, the natural mistake because
the eye-to-hand recipe puts the gripper term first, and the exact inverse problem is solved instead:
if `A' = inv(B)` and `B' = inv(A)`, then `A' X' = X' B'` is solved by `X' = inv(X)`. What comes back
is `T_tool_to_cam` while every label still says `T_cam_to_tool`, with a plausible residual and no
exception. Use `EyeInHandCalibrator` rather than building the pair yourself.

### 2.3 Which solver, and which one is not used

There is no OpenCV hand-eye call in this repository. Inside the hand-eye path OpenCV does two things
only: ArUco detection, and `cv2.solvePnP` in the marker sources.

The solver is a pure-NumPy Park and Martin closed form in
[`src/calibration/solver/hand_eye_axxb.py`](../../src/calibration/solver/hand_eye_axxb.py): rotation
from the null space of a stacked Kronecker system by SVD, then translation by least squares. Two
properties matter when you read a failure:

- **Input validation is strict**, so a malformed pose raises rather than being quietly projected.
- **It needs at least 3 pairs.** N samples give N-1 pairs, so the floor is 4 samples, and
  `min_samples` refuses anything below 4 in both the settings object and the config schema.

The residual it reports is `mean ||A_i X - X B_i||_F` over all pairs. Section 3 is about what that
number is not.

### 2.4 What the output transform means

`result.transform` is **the pose of the camera in the robot base frame**, and it **maps camera-frame
points into base**: `p_base = X . p_cam`. Its translation column is the camera origin in base, in
millimetres. On an exact, noise-free synthetic problem the solve reaches machine precision, so the
residual sits in the floating-point noise; the exact digit depends on the pose set.

The names are asymmetric. Read this table once rather than trying to fix it:

| Symbol in code | The matrix really is | It maps |
|---|---|---|
| `T_base_to_tool` | `base_T_tool`, the TCP pose in BASE, from `arm.get_tcp_pose()` | tool into base |
| `T_cam_to_marker` | `cam_T_marker`, the marker pose in CAMERA | marker into cam |
| `T_cam_to_base` (eye-to-hand output) | `base_T_cam`, the camera pose in BASE | cam into base |
| `T_cam_to_tool` (eye-in-hand output) | `tool_T_cam`, the camera pose in TOOL | cam into tool |

Inputs read as "pose of Y in X", outputs as "maps X into Y". Both agree with the `Frame` tags, and the
tags are what is enforced: eye-to-hand is checked `CAMERA` to `BASE` and eye-in-hand `CAMERA` to
`TOOL`, at construction. Lengths are millimetres everywhere, solver and artifacts included. Rotations
on the wire are XYZW quaternions, but `PoseProvider`'s `base_orientation` is axis-angle, in radians.

### 2.5 What a pose set must satisfy

`_assert_ready` in [`src/calibration/eye_hand/common.py`](../../src/calibration/eye_hand/common.py)
runs before every solve, in both modes. All three must hold:

1. `len(dataset) >= settings.min_samples`, else `CalibrationDataError: too few samples`;
2. at least **two** relative robot motions that rotate by 1 degree or more, else
   `"sample set needs at least two meaningful robot rotation motions"`;
3. those rotation axes span **two independent directions**, `matrix_rank(axes, tol=0.1) >= 2`, else
   `"sample set needs robot rotations around at least two independent axes"`.

Condition 3 is why a pure-translation sweep can never solve a hand-eye problem, and why an eye-in-hand
viewpoint hemisphere varies azimuth **and** elevation rather than only standoff.

Two filters drop samples before that, on different data:

- `WorkspaceGuard` rejects a **commanded** pose that is close to an accepted one on *both* axes at
  once.
- `add_sample` accepts a **measured** pose only if, against every stored sample,
  `dist > min_distance_mm` **or** `angle > min_angle_deg`.

That "or" is why widely spread translations with near-identical orientations pass `add_sample` and
then fail condition 3. So that error means "tilt more", not "move further". Plan for more poses than
`min_samples`.

`add_sample` returns `False` for a rejected sample and does not raise, so one bad viewpoint cannot
abort a twenty-pose sweep. Check the return value, and read the log line for the reason.

---

## 3. The residual: what `rmse_mm` actually measures

**This is the most misread number in the calibration stack.**

### 3.1 It is not millimetres

`rmse_mm` and `max_error_mm` are the mean and maximum **Frobenius norm** `||A_i X - X B_i||_F`. That
norm mixes the dimensionless rotation block with millimetre translation terms, and it scales with
`|t_X|`. It is a self-consistency score, not a distance. A calibration can sit comfortably inside
`excellent` while carrying a translation error larger than a millimetre.

The module docstring of [`quality.py`](../../src/calibration/quality.py) says the same. The shape of
the sensitivity is what to remember: on an exact, noise-free synthetic problem with a metre-scale
`|t_X|`, perturbing the known `X` by one degree of rotation scores roughly an order of magnitude worse
than perturbing it by one millimetre of translation. The exact multiplier moves with `|t_X|` and with
the pose set. So the band is mostly a **rotation-consistency** gate, and any page that reads a band as
"2.5 mm of error" is wrong.

### 3.2 The bands, and where the YAML sets the label

`classify_rmse` uses inclusive upper bounds: `excellent` at or below 1.0, `good` at or below 2.5,
`marginal` at or below 5.0, `poor` above that, and `unknown` for `None`, non-finite or negative.

On a real cell, `robot.calibration.quality_bands_mm` sets the label on the saved artifact and on the
report: `HandEyeCalibration` builds the solver with those bands. A `CalibrationRoutine` built without
a solver labels with the built-in `DEFAULT_BANDS_MM`, and the simulation runners' artifacts carry that
label. The two simulation calibration runners apply the YAML bands to their own verdict instead, so
in simulation retuning the bands moves the verdict, not the artifact's label. The shipped YAML values
equal the built-in ones, so on a shipped tree the two labels agree.

### 3.3 Nothing auto-applies

There is no quality gate in the library. The code always computes a `quality` label, always builds
the result whatever it says, and never applies anything. Saving is a step the caller takes, and
neither `load_extrinsics` nor either resolver builder reads `quality`. The only hard refusals in the
pipeline are `min_samples`, the two-independent-axes rank check, and the frame tags.

The config offers no `quality_threshold_mm` key, and the schema docstring in
[`calibration_schema.py`](../../src/config/schema/robot/calibration_schema.py) says why: no such
auto-apply gate exists. A YAML that sets it fails to load with an unknown-key error. The quality gates
that do exist protect a pick, not a save: the simulated eye-in-hand pick runner falls back to its
ground-truth oracle when an eye-in-hand artifact is not `good` or `excellent`, and to a live
ground-truth fit when an eye-to-hand one is not. **Gating a save is your job.**

### 3.4 How to judge a result

| Result | What to do |
|---|---|
| `excellent`, a healthy sample count, and the physical check agrees | ship it |
| `good` | acceptable for a first bring-up; check physically again before production |
| `marginal` | investigate; a small marker seen through real perception can land here, but find the cause |
| `poor` | reject: marker size, a flexing mount, motion blur, too few tilts, the wrong dictionary |
| `unknown` | a non-finite or negative residual: a bug, not a bad calibration |

The rule that outranks that table: **the residual is a self-consistency number, so check it against a
physical measurement.** Put a known object at a known position, run a pick that locates it through the
calibrated camera, and measure where the TCP lands. Eye-in-hand gets a second free check: the
translation of `result.transform` is the camera origin in the tool frame, and it should match the
mechanical drawing to a few millimetres.

---

## 4. How the transform reaches a pick

### 4.1 Two artifact types

Both live in [`serialization.py`](../../src/calibration/serialization.py).

- **Eye-to-hand** saves an `Extrinsics` (schema `willy.calibration.extrinsics/1`): the transform plus
  `rmse_mm`, `max_error_mm`, `num_samples`, `captured_at`, `rig_id` and `quality`.
- **Eye-in-hand** saves a frame-tagged `Transform` (schema `willy.calibration.cam_to_tool/1`): the
  transform and `rig_id` only, with no residual and no sample count.
- **The real-cell sweep on a UR arm** writes `willy.calibration.cam_to_tool/2`, which adds
  `flange_to_tcp`: the tool frame the solve was made against. `get_tcp_pose` is the flange times that
  frame, so `CAMERA -> TOOL` is only true while the cell holds it. On a `willy` tool frame the record
  is the declared frame; on a `polyscope` one it is the frame the driver read from the controller at
  connect.

A `/1` eye-in-hand artifact still loads, with no record, but a rig that declares a camera body refuses
it: calibrate that rig again. Once a body is placed from the record, a tool frame that has moved since
refuses the rig: exactly on `willy`, and beyond the rig's `extrinsics.record_tolerance_mm` and
`record_tolerance_deg` on `polyscope`, because the body and the pick frame are both stale.

Both types check their frames on write and on read, and both write a temporary file and then
`Path.replace()` it, so a save is atomic. The schema check on load is strict and raises. An artifact
of unknown vintage should force a new measurement rather than be trusted, so do not edit a schema
string by hand to get past it.

### 4.2 The config keys

A camera's calibration is declared on its rig, in the camera section, and one loader opens every
artifact: `RigCalibration.from_config` in
[`src/calibration/rig_calibration.py`](../../src/calibration/rig_calibration.py). Which cameras are
fused is a second key, in the robot section. Both resolver builders live in
[`src/robot/execution/autonomous_grasp/builders.py`](../../src/robot/execution/autonomous_grasp/builders.py)
and load through that one loader.

| Key | Holds | Read by |
|---|---|---|
| `camera.cameras.rigs[<id>].extrinsics` | the rig's `mounting_mode` and `artifact_path`, and the tolerances below | `build_config_frame_resolver` (the primary rig) and `build_config_frame_resolvers` |
| `robot.grasping.fusion.cameras.<id>` | which rigs are fused, keyed by rig id: `enabled` only | `build_config_frame_resolvers` |

The tolerances on a rig's `extrinsics`:

- `shutter_motion_tolerance_mm` and `shutter_motion_tolerance_deg` are required on an `eye_in_hand`
  rig, with no default, and refused on `eye_to_hand`.
- `record_tolerance_mm` and `record_tolerance_deg` are required on a rig that declares a `body`, and
  refused on `eye_to_hand`.

The mounting picks the resolver: `eye_to_hand` gives a `StaticCameraToBaseResolver` holding the
`CAMERA -> BASE` artifact, and `eye_in_hand` an `EyeInHandFrameResolver` that composes the
`CAMERA -> TOOL` artifact with the tool pose. The robot section carries no calibration:
`robot.grasping.fusion.extrinsics_artifact_path`, and a `fusion.cameras` entry's `mounting_mode` or
`extrinsics_artifact_path`, are refused at load with a sentence naming the rig key. So is a
`fusion.cameras` id that names no rig (`camera_calibration_conflict`).

**The two builders gate differently.** The singular one ignores `robot.grasping.fusion.enabled`: it
returns `None` only when no camera section was handed in or the primary rig declares no `extrinsics`.
The plural one returns `{}` when `robot.grasping.fusion.enabled` is false, before reading anything and
without a warning. Silence in a YAML file means the schema default is in force, so check any key you
rely on under the chain you run:

```bash
python -m src.config explain robot.grasping.fusion.enabled
```

On the shipped base tree it prints `set in (no YAML sets this: the schema default is in force)`.
Other layers write it.

### 4.3 What is wired

`AutonomousGraspService.from_robot_config()` is the config-driven boot path. `build_real_cell` and
`build_rehearsal_cell` call it, and they are what `Cell`, `python -m src.robot.execution.real_cell`,
the operator console and the [examples](../../examples/README.md) reach. The Isaac runner
`run_multiview_pick` takes it by default, through `run_eih_pick.build_service`, and so does
`datagen/rl/occupancy.py`.

On that path, `from_robot_config` builds the singular resolver from the primary rig's `extrinsics`
when the caller passed none, reading the camera section it is handed as `camera=` (`build_real_cell`
passes it). It then fails closed on a real vendor that still has no resolver (section 4.4), and
applies the orchestrator overlays, which reach the plural builder. That one needs two flags:
`fusion.geometry.enabled` for the call to happen, and `fusion.enabled` for it to return anything.

Two things to carry away:

- **The base camera section declares no rig's calibration**, so a freshly configured real cell is
  refused at build until one is declared. The `ur5e,eth2` example declares both of its rigs', at
  artifact paths a sweep writes and the repository does not ship; until both sweeps have run, its
  build is refused naming the file that does not load.
- **The default pick is open-loop** whatever the calibration says. The decision gate, closed-loop
  refinement, verification, recovery, fusion with its commit gate, the learned ranker and the learned
  success model all ship `enabled: false` in [`config/robot/robot.yaml`](../../config/robot/robot.yaml).
  See [05](05-pick-loop.md).

### 4.4 Where it fails closed

- `ValueError: no CAMERA->BASE frame resolver for a '<vendor>' cell`

  `from_robot_config` on a real cell with nothing wired. Declare the primary camera's calibration
  on its rig, `camera.cameras.rigs[<primary rig id>].extrinsics`, and hand the root the camera
  section, or pass `frame_resolver=`.
- `RuntimeError: camera.cameras.rigs['x'].extrinsics names <path> (<mode>), which does not load: ...`

  A declared artifact is missing, stale or invalid, on the primary rig or on an enabled fused
  camera. The cell is refused at construction rather than run with a camera it cannot place.
- `RuntimeError: grasping.fusion.cameras names 'x' and camera.cameras.rigs['x'].extrinsics is not declared ...`

  An enabled fused camera whose rig declares no calibration. An incomplete resolver map must not
  run as a smaller rig than was asked for. `build_real_cell` refuses the same camera earlier, before a
  device is opened.
- `CellBuildRefused: this cell plans with cuRobo and calibrates '<rig>', and no live camera world comes of it: ...`

  A cuRobo cell declares a rig's calibration and gets no live camera world from it. Once a rig is
  calibrated its world is mandatory: enable `safety.planning_world` with a `support_plane` and
  `perceived.enabled`, with the primary rig among the calibrated ones, or remove the rig's
  `extrinsics` while it is not used. `build_real_cell` refuses before a camera opens, and
  `real_cell --check` blocks on its `camera world` row ([04](04-robot-and-safety.md)).
- `PickOutcome.CAMERA_FRAME_REJECTED`, at pick time

  A valid candidate existed, but in the camera frame, while `require_base_frame_grasp` was on. The
  calibration is not reaching the pick path ([05](05-pick-loop.md)).

One case is **not** fail-closed: `fusion.enabled: false` with second cameras named in
`fusion.cameras` and calibrated on their rigs. The primary's resolver loads and the others never do,
so the pick runs single-view. Nothing refuses. With `fusion.geometry.enabled` on, the loop warns on
every pick that each other camera's view was dropped; with it off, nothing asks for the map.

---

## 5. What goes wrong, and how it shows up

| Symptom | What it usually is |
|---|---|
| `CalibrationDataError: too few samples` | too many poses dropped; count `POSE_REJECTED` by reason (below) |
| Either rotation error from section 2.5 | too little rotation, or every tilt about one axis; eye-in-hand varies azimuth and elevation |
| Every pose logs `marker_not_found` | wrong dictionary or marker id, marker out of frame, too few pixels, or a black frame (below) |
| Detection works, the residual is `marginal` or `poor` | a wrong `marker_length_mm` (below), a flexing mount, motion blur, too little tilt, or glare |
| ArUco detects, the rotation is nonsense | planar-marker flip ambiguity at near-frontal views; add oblique views, as both sweeps do |
| Residual `excellent`, picks miss by a **constant** offset | not millimetres (3.1); compare the loaded artifact's translation to the camera's mount |
| A result rotated by a right angle or 180 degrees | check the reference it is compared against first: a wrong reference is harder to find |
| Eye-to-hand expected, a `CAMERA -> TOOL` result arrived | settings with no `mode` make `CalibrationRoutine` default to eye-in-hand; pass `calibration_mode=` |
| Grasps went bad after a **gripper change**, camera untouched | eye-in-hand only: the TCP moved, so `CAMERA` to `TOOL` is stale, and it looks like no camera fault |
| Fused cameras calibrated, the pick stays single-view, no refusal | `fusion.enabled` is false (section 4.2) |

Random misses are not calibration. Three notes on the rows above:

- **`POSE_REJECTED` reasons:** `marker_not_found` is detection, `move_rejected` is reach or safety,
  and `sample_rejected` is the diversity rule of section 2.5.
- **A black frame in simulation** is a camera left at the 1.0 m default near clip: RGB is black while
  depth still works, which looks like a detector fault.
- **Measure the marker.** A declared 50 mm board that is really 48 mm biases every sample by a scale
  factor: the solve converges cleanly and is uniformly wrong. Measure the printed black square's edge,
  not the white border and not what the PDF was called.

**One physical spec is written in more than one place.** `marker_length_mm` and `aruco_dict_name`
exist under `camera.hand_eye.eye_to_hand`, under `camera.hand_eye.eye_in_hand`, and under
`camera.cameras.stereo_calibration`. The third belongs to stereo intrinsics through ChArUco and
describes a different board, but carries the same key names and the same shipped values. If the stereo
marker path is in use, both must describe the board you actually have.

**When to recalibrate:** any collision involving the camera, the mount or the tool; a tool or gripper
change, eye-in-hand especially; the camera moved, refocused or remounted; the robot base moved; a drift
alarm on the residual channels; grasps that start missing by a constant offset.

---

## 6. What has run, and what has not

**Pure logic, pinned by the mock suite with no GPU and no simulator** (the calibration package is
torch-free): the solver, the frame contracts, the artifact round trips and the resolver map.

| Capability | Evidence |
|---|---|
| Eye-to-hand and eye-in-hand through `CalibrationRoutine` | measured in simulation: both run end to end, and the eye-in-hand wrist-camera pick lands on target |
| The real-cell sweep through `HandEyeCalibration` | never touched hardware: the mock suite covers its flow; no physical camera or controller has run it |

**Two kinds of simulated run.** The ground-truth marker source is derived from the same oracle the
result is compared against, so its near-zero residual is **circular**. It validates collection, the
diversity filters, the AX=XB solve, the frame tags and the save-load round trip, and says nothing
about perception accuracy. The ArUco runs are the ones that test perception.

**Calibrating in simulation** is [`run_eth_calibrate.py`](../../src/willy_sim/run_eth_calibrate.py) and
[`run_eih_calibrate.py`](../../src/willy_sim/run_eih_calibrate.py). Both add `--robot-model`, `--profile`
and `--radial-closing` to their own flags, and write artifacts per robot model under
`logs/calibration/`, which is not committed. Their `-h` is the procedure.

**The real-cell sweep.** [`src/robot/execution/real_cell/calibrate.py`](../../src/robot/execution/real_cell/calibrate.py)
is a caller of `HandEyeCalibration` in [`src/robot/execution/hand_eye.py`](../../src/robot/execution/hand_eye.py).
It drives the same `CalibrationRoutine` with a live RGB-D ArUco marker source. `--check` reads the
config only, and `--dry-run` opens the camera and builds the arm without moving. Past `--dry-run` it
is unproven: that marker source has never seen a physical camera, and an aligned stream reports
distortion coefficients near zero, so its residual is unconfirmed on a real bench.

The sweep builds its arm through `Robot.from_config(robot_config, gripper=None)`, so the arm-vendor
readiness gate runs and no gripper is built. It takes the cell lock before the arm connects, connects
no gripper, and exits 1 when another process holds the cell or the connect is refused. Every move the
routine commands declines the camera world for itself, with a reason named for the mounting, because
the sweep is what produces the transform a camera world needs. On a cuRobo cell a move's result says
`DECLINED` with that reason, where an undeclined move would be refused as `MISSING`. The command
prints the robot's camera world line, ending `this sweep declines for itself`.

---

## 7. Where to read next

| For | Go to |
|---|---|
| The typed extrinsics, the stereo runtime, quality and persistence | [`src/calibration/README.md`](../../src/calibration/README.md) |
| The two workflows, the multi-camera map and their traps | [`src/calibration/eye_hand/README.md`](../../src/calibration/eye_hand/README.md) |
| The bench session: print the board, run the sweep, wire the artifact in | [docs/calibration-setup.md](../calibration-setup.md) |
| The calibration commands and their exit codes | [docs/cli.md](../cli.md) |
| One camera against a real robot, and the flags | [`src/robot/execution/real_cell/README.md`](../../src/robot/execution/real_cell/README.md) |
| Both modes driven from Python | [`07_calibrate_a_fixed_camera.py`](../../examples/real_robot/07_calibrate_a_fixed_camera.py), [`08_calibrate_a_wrist_camera.py`](../../examples/real_robot/08_calibrate_a_wrist_camera.py) |
| Config layering, `explain` and `where` | [01](01-configuration.md) |
| Camera intrinsics, and building an arm | [02](02-models.md) . [04](04-robot-and-safety.md) |
| The pick that consumes the transform | [05](05-pick-loop.md) |
