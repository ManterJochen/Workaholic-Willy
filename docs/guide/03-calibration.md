# 3. Hand-eye calibration: the derivations

This page is for anyone who has to reason about a calibration result rather than only produce one:
why the problem is `AX = XB`, what the two mounting cases solve for, what the residual actually
measures (it is not millimetres), how the CAMERA to BASE transform reaches a grasp, and how each
failure announces itself.

It does not run a calibration. The bench session is [docs/calibration-setup.md](../calibration-setup.md);
the commands are `python -m src.robot.execution.real_cell.calibrate` on a real cell and
`run_eth_calibrate` / `run_eih_calibrate` in simulation. The package documentation is
[`src/calibration/README.md`](../../src/calibration/README.md) for the typed extrinsics, the stereo
runtime and persistence, and
[`src/calibration/eye_hand/README.md`](../../src/calibration/eye_hand/README.md) for the two
workflows themselves. [`scripts/examples/api/02_calibration/calibrate_fixed_camera.py`](../../scripts/examples/api/02_calibration/calibrate_fixed_camera.py)
is the runnable walkthrough against the library API.

Sibling guides: [01](01-configuration.md) . [02](02-models.md) . **03** . [04](04-robot-and-safety.md)
. [05](05-pick-loop.md) . [06](06-grippers.md)

---

## 1. What calibration is for

The grasp stack computes and commands motion in the robot **BASE** frame. Perception produces depth
and segmentation in the **CAMERA** frame. Exactly one rigid transform sits between them, and
calibration measures it.

Get it wrong and every candidate is confidently, consistently wrong. That is worse than no
candidates, because a wrong transform produces plausible poses the safety stack will accept.

### 1.1 Eye-to-hand and eye-in-hand

| | Eye-to-hand | Eye-in-hand |
|---|---|---|
| Camera is | bolted to the world: post, gantry, ceiling | bolted to the wrist, moves with the arm |
| Marker is | bolted to the tool flange | bolted to the world: table, fixture |
| Solves for | `Transform(CAMERA -> BASE)` | `Transform(CAMERA -> TOOL)` |
| Valid until | the camera or the robot base moves | the mount or the TCP definition changes |
| Runtime consumer | `StaticCameraToBaseResolver`: the same transform every call, ignoring the arm | `EyeInHandFrameResolver`: reads `arm.get_tcp_pose()` every frame and composes |
| Persisted as | `Extrinsics`, schema `willy.calibration.extrinsics/1` | a frame-tagged `Transform`, schema `willy.calibration.cam_to_tool/1` |

Both resolvers live in
[`src/robot/grasping/motion/frame_resolver.py`](../../src/robot/grasping/motion/frame_resolver.py).

**Which one is right.** Eye-to-hand when the scene is fixed and the transform must be valid before
the arm has moved: static overhead bin picking, or any cell where the camera has to see the
workspace while the arm is elsewhere. Eye-in-hand when the view has to come from where the tool is
going: close-range refinement, or deep bins nothing static has a clear line of sight into.
Eye-in-hand costs a live TCP read every frame and a recalibration on every gripper change. Many
cells want both.

An eye-in-hand result cannot be saved as `Extrinsics`, deliberately:
[`src/calibration/eye_hand/types.py`](../../src/calibration/eye_hand/types.py) raises
`"only eye_to_hand results can be saved as Extrinsics"` in `to_extrinsics`. The reason is the
composition:

```
T_cam_to_base(t)  =  T_cam_to_tool  o  T_tool_to_base(t)
                     the calibration   the live TCP read, this instant
```

Half of an eye-in-hand camera-to-base transform does not exist until the arm is somewhere. There is
nothing to serialize.

**The chain, end to end.** `CalibrationRoutine` collects samples and hands them to
`EyeToHandCalibrator` or `EyeInHandCalibrator`, which build the `A` and `B` pairs and call
`HandEyeAXXB`. You save the result, a config key points at that file, a resolver builder loads it
into a `FrameResolver`, and the pick loop uses that resolver to back-project masked depth pixels
into the BASE frame, where candidates are generated. Section 4 covers the keys and the builders.

---

## 2. Why it is AX = XB

### 2.1 The eye-to-hand derivation

Fixed camera, marker rigid on the tool. For every pose `i`:

```
cam_T_marker_i  =  cam_T_base . base_T_tool_i . tool_T_marker
```

Take consecutive relative motions. `HandEyeAXXB.relative_motions` in
[`src/calibration/solver/hand_eye_axxb.py`](../../src/calibration/solver/hand_eye_axxb.py) maps `[T_0, T_1, ...]` to
`[T_1 @ inv(T_0), T_2 @ inv(T_1), ...]`, so:

```
A_i = base_T_tool_{i+1} . inv(base_T_tool_i)      relative gripper motion
B_i = cam_T_marker_{i+1} . inv(cam_T_marker_i)    relative marker motion
```

`tool_T_marker` cancels exactly, leaving `B_i = cam_T_base . A_i . base_T_cam`, which is

```
A_i . X  =  X . B_i        with  X = base_T_cam   (the code calls it T_cam_to_base)
```

**The consequence you can rely on:** the marker-on-flange offset `tool_T_marker` is never needed and
never solved for. Where on the flange the board was bolted does not matter. Every public `run_*`
method on `CalibrationRoutine` still accepts a `T_tool_to_marker` argument, threads it into
`_execute`, and never references it again. Pass `None`. Both `A` and `B` come from the same helper
in [`src/calibration/eye_hand/eye_to_hand/calibrator.py`](../../src/calibration/eye_hand/eye_to_hand/calibrator.py).

### 2.2 The eye-in-hand derivation, and why the algebra inverts

Camera on the tool, marker fixed in the world. Let `M_i = base_T_tool_i`, `C_i = cam_T_marker_i` and
`X = tool_T_cam`. Constancy of `base_T_marker` gives `M_i . X . C_i = M_{i+1} . X . C_{i+1}`, which
rearranges to `A_i . X = X . B_i` with

```
A_i = inv(M_{i+1}) . M_i      local (left) relative tool motion, inverted order
B_i = C_{i+1} . inv(C_i)      global (right) relative marker motion
```

Only the `B` side comes from `relative_motions`. `A` is built by hand in
[`src/calibration/eye_hand/eye_in_hand/calibrator.py`](../../src/calibration/eye_hand/eye_in_hand/calibrator.py). On the tool
side eye-in-hand takes `inv(next) . prev` where eye-to-hand takes `next . inv(prev)`, and that
difference is not cosmetic. Swap the two sides and reverse the index order, the natural mistake
because the eye-to-hand recipe puts the gripper term first, and the exact inverse problem is solved
instead: if `A' = inv(B)` and `B' = inv(A)`, then `A' X' = X' B'` is solved by `X' = inv(X)`. What
comes back is `T_tool_to_cam` while every label still says `T_cam_to_tool`, with a plausible residual
and no exception. Use `EyeInHandCalibrator` rather than hand-rolling the pair.

### 2.3 Which solver, and which one is not used

There is no OpenCV hand-eye call in this repository. Inside the hand-eye path OpenCV does two things
only: ArUco detection, and `cv2.solvePnP` in the marker sources.

The solver is a hand-written pure-NumPy Park and Martin closed form in
[`src/calibration/solver/hand_eye_axxb.py`](../../src/calibration/solver/hand_eye_axxb.py): rotation from the null
space of a stacked Kronecker system by SVD, then translation by least squares. Two properties matter
when reading a failure. Input validation is strict, so a malformed pose raises rather than being
quietly projected. And `_MIN_PAIRS` is 3, so with N samples giving N-1 pairs the arithmetic floor is
4 samples, which is why `min_samples` refuses anything below 4 in both the settings object and the
config schema. The residual it reports is `mean ||A_i X - X B_i||_F` over all pairs, and section 3 is
about what that number is not.

### 2.4 What the output transform means

`result.transform` is **the pose of the camera in the robot base frame**, and it **maps camera-frame
points into base**: `p_base = X . p_cam`. Its translation column is the camera origin in base, in
millimetres. On an exact, noise-free synthetic problem the solve reaches machine precision, so the
residual sits in the floating-point noise; the exact digit depends on the pose set.

Note the naming asymmetry, worth reading once and not trying to fix:

| Symbol in code | The matrix really is | It maps |
|---|---|---|
| `T_base_to_tool` | `base_T_tool`, the TCP pose in BASE, from `arm.get_tcp_pose()` | tool into base |
| `T_cam_to_marker` | `cam_T_marker`, the marker pose in CAMERA | marker into cam |
| `T_cam_to_base` (eye-to-hand output) | `base_T_cam`, the camera pose in BASE | cam into base |
| `T_cam_to_tool` (eye-in-hand output) | `tool_T_cam`, the camera pose in TOOL | cam into tool |

Inputs read as "pose of Y in X", outputs as "maps X into Y". Both are consistent with the `Frame`
tags, and the tags are what is enforced: eye-to-hand is hard-checked `CAMERA` to `BASE` and
eye-in-hand `CAMERA` to `TOOL`, at construction. Lengths are millimetres everywhere, solver and
artifacts included. Rotations on the wire are XYZW quaternions, but `PoseProvider`'s
`base_orientation` is axis-angle, in radians.

### 2.5 What a pose set must satisfy

`_assert_ready` in [`src/calibration/eye_hand/common.py`](../../src/calibration/eye_hand/common.py) runs before every
solve, for both modes. All three must hold:

1. `len(dataset) >= settings.min_samples`, else `CalibrationDataError: too few samples`;
2. at least **two** relative robot motions with a rotation angle of 1 degree or more, else
   `"sample set needs at least two meaningful robot rotation motions"`;
3. those rotation axes must span **two independent directions**, `matrix_rank(axes, tol=0.1) >= 2`,
   else `"sample set needs robot rotations around at least two independent axes"`.

Condition 3 is why a pure-translation sweep can never solve a hand-eye problem, and why an
eye-in-hand viewpoint hemisphere varies azimuth **and** elevation rather than only standoff.

Two more filters drop samples before that, on different data. `WorkspaceGuard` rejects a
**commanded** pose that is close to an accepted one on *both* axes at once; `add_sample` accepts a
**measured** forward-kinematics pose only if, against every stored sample, `dist > min_distance_mm`
**or** `angle > min_angle_deg`. That disjunction is why widely spread translations with near-identical
orientations sail through `add_sample` and then die at condition 3, which is why that error means
"tilt more", not "move further". Plan for more poses than `min_samples`.

`add_sample` returns `False` for a rejected sample and does not raise, so that one bad viewpoint
cannot abort a twenty-pose sweep. Check the return value, and read the log line for the reason.

---

## 3. The residual: what `rmse_mm` actually measures

**This is the single most misread number in the calibration stack.**

### 3.1 It is not millimetres

`rmse_mm` and `max_error_mm` are the mean and maximum **Frobenius norm** `||A_i X - X B_i||_F`. That
norm mixes the dimensionless rotation block with millimetre translation terms, and it scales with
`|t_X|`. It is a self-consistency score, not a distance. A calibration can sit comfortably inside
`excellent` while carrying a translation error larger than a millimetre.

The module docstring of [`quality.py`](../../src/calibration/quality.py) states the same, and the
shape of the sensitivity is what to carry: on an exact, noise-free synthetic problem with a
metre-scale `|t_X|`, perturbing the known `X` by one degree of rotation scores roughly an order of
magnitude worse than perturbing it by one millimetre of translation. The exact multiplier moves with
`|t_X|` and with the pose set, so no single number describes it. The durable point is that the band
is mostly a **rotation-consistency** gate. Any page that reads a band as "2.5 mm of error" is wrong.

### 3.2 The bands, and why no YAML sets the label

`classify_rmse` uses inclusive upper bounds: `excellent` at or below 1.0, `good` at or below 2.5,
`marginal` at or below 5.0, `poor` above that, and `unknown` for `None`, non-finite or negative.

The `robot.calibration.quality_bands_mm` YAML block does **not** set the label on the result or on
the saved artifact. `BaseEyeHandCalibrator` defaults to the hard-coded `DEFAULT_BANDS_MM` and
`CalibrationRoutine` never passes a `bands=`. What the YAML block does reach is the callers that run
`classify_rmse` themselves: the two simulation calibration runners and the real-cell `calibrate`
command. In `run_eth_calibrate.py` the label feeds `ok_quality`, `ok_quality` feeds the run's `ok`
flag, and that flag decides the gate line, the log level and the value returned to a caller. So
retuning the bands moves a verdict, not a printed line. The shipped YAML values equal the hard-coded
ones, which is why the difference is easy to miss.

### 3.3 Nothing auto-applies

There is no quality gate in the library. The code always computes a `quality` label, always builds
the result regardless of it, and never applies anything anywhere. `save_extrinsics(...)` is a call
you make, and neither `load_extrinsics` nor either resolver builder inspects `quality`. The only hard
refusals in the pipeline are `min_samples`, the two-independent-axes rank check, and the frame tags.

The config deliberately offers no `quality_threshold_mm` key, and the schema docstring in
[`calibration_schema.py`](../../src/config/schema/robot/calibration_schema.py) says why: no such
auto-apply gate exists. A YAML that sets it fails to load with an unknown-key error. The quality
gates that do exist protect a pick, not a save: the simulation eye-in-hand pick runner falls back to
its ground-truth oracle when an eye-in-hand artifact is not `good` or `excellent`, and to a live
ground-truth fit when an eye-to-hand one is not. **Gating a save is your job.**

### 3.4 How to judge a result

| Result | What to do |
|---|---|
| `excellent`, healthy sample count, physical check agrees | ship it |
| `good` | acceptable for a first bring-up. Re-check physically before production |
| `marginal` | investigate. Real fiducial perception at a small marker size can legitimately land here, but on a real cell prefer to find the cause |
| `poor` | reject. Marker size, flexing mount, motion blur, too few tilts, wrong dictionary |
| `unknown` | a non-finite or negative residual. A bug, not a bad calibration |

The rule that outranks that table: **the residual is a self-consistency number, so cross-check it
against a physical measurement.** Put a known object at a known position, run a pick that localizes
it through the calibrated camera, and measure where the TCP lands. Eye-in-hand gets a second free
check, since `result.transform`'s translation is the camera origin in the tool frame: it should match
the mechanical drawing to a few millimetres.

---

## 4. How the transform reaches a pick

### 4.1 Two artifact types

Eye-to-hand persists an `Extrinsics` (schema `willy.calibration.extrinsics/1`) carrying the transform
plus `rmse_mm`, `max_error_mm`, `num_samples`, `captured_at`, `rig_id` and `quality`. Eye-in-hand
persists a bare frame-tagged `Transform` (schema `willy.calibration.cam_to_tool/1`) carrying the
transform and `rig_id` only: no residual, no sample count. Both check frames on write and on read,
and both write to a temporary file and then `Path.replace()`, so a save is atomic. Both live in
[`serialization.py`](../../src/calibration/serialization.py), where the schema check on load is
strict and raises. That is deliberate: an artifact of unknown vintage should force a re-measure
rather than be trusted, so do not hand-edit a stale schema string to get past it.

### 4.2 The config keys

Two keys point at calibration artifacts, and different builders read them. Both builders live in
[`src/robot/execution/autonomous_grasp/builders.py`](../../src/robot/execution/autonomous_grasp/builders.py).

| Key | Shape | Read by |
|---|---|---|
| `robot.grasping.fusion.extrinsics_artifact_path` | one path, eye-to-hand only | `build_config_frame_resolver` (singular) |
| `robot.grasping.fusion.cameras.<id>` | per-camera map: `mounting_mode`, `extrinsics_artifact_path`, `enabled` | `build_config_frame_resolvers` (plural) |

The primary camera's artifact is the singular key and is never an entry in the `cameras` map. It is
also the only one that satisfies the CAMERA to BASE refusal when a real cell is built, so a cell can
load every artifact in the map and still be refused at build because that key is null.

Both builders gate on `robot.grasping.fusion.enabled`. With it false they return `None` and `{}`
before reading anything, and without a warning. Silence in a YAML file means the schema default is in
force, not that the feature is absent, so check any key you rely on:

```powershell
python -m src.config explain robot.grasping.fusion.enabled
```

On the shipped tree that prints `set in (no YAML sets this: the schema default is in force)`. Other
profile layers do write it, so `explain` under the profile chain you actually run is the only honest
answer.

### 4.3 What is wired

`AutonomousGraspService.from_robot_config()` is the config-driven boot path, and it has live callers:
`build_real_cell` and `build_rehearsal_cell` in `src/robot/execution/autonomous_grasp/cells.py`, which is what
`python -m src.robot.execution.real_cell`, the operator console and
[`scripts/examples/api/01_first_cell/one_pick_end_to_end.py`](../../scripts/examples/api/01_first_cell/one_pick_end_to_end.py) all reach through `Cell`; plus
`src/willy_sim/run_multiview_pick.py` and `run_eih_pick.py`, and `datagen/rl/occupancy.py`.

Inside that path, `from_robot_config` builds the singular resolver from
`fusion.extrinsics_artifact_path` when the caller passed none, then fails closed on a real vendor
that still has no resolver (section 4.4). It then applies the orchestrator overlays, which reach the
plural per-camera builder. That one takes two flags, not one: `fusion.geometry.enabled` to reach the
call and `fusion.enabled` for the call to return anything.

Two things to carry away. No shipped YAML declares an `eye_in_hand` camera in `fusion.cameras`, so
that schema branch has no camera exercising it. And the default pick is **open-loop** whatever the
calibration says: the decision gate, closed-loop refinement, verification, recovery, fusion with its
commit gate, the learned ranker and the learned success model all ship `enabled: false` in
[`config/robot/robot.yaml`](../../config/robot/robot.yaml). See [05](05-pick-loop.md).

### 4.4 Where it fails closed

| Refusal | Meaning |
|---|---|
| `ValueError: no CAMERA->BASE frame resolver for a '<vendor>' cell` | `from_robot_config` on a real cell with nothing wired. Set the artifact path, or pass `frame_resolver=`. |
| `RuntimeError: grasping.fusion.cameras['x'] (<mode>) failed to load ...` | an enabled camera's artifact is missing, stale or invalid. An incomplete resolver map must not run silently as a smaller rig than was asked for. |
| `PickOutcome.CAMERA_FRAME_REJECTED` at pick time | a valid candidate existed but was in the camera frame while `require_base_frame_grasp` was on. The calibration is not reaching the pick path. See [05](05-pick-loop.md). |

And the one that is **not** fail-closed: `fusion.enabled: false` with a perfectly good artifact
configured. Nothing loads, nothing warns, and the pick runs single-view.

---

## 5. What goes wrong, and how it shows up

| Symptom | What it usually is |
|---|---|
| `CalibrationDataError: too few samples` | too many poses dropped. Count `POSE_REJECTED` by reason: `marker_not_found` is detection, `move_rejected` is reach or safety, `sample_rejected` is the diversity rule of section 2.5. |
| Either observability error from section 2.5 | too little rotation, or every tilt about one axis. Vary a second axis; for eye-in-hand vary azimuth and elevation, not radius. |
| Every pose logs `marker_not_found` | wrong dictionary, wrong marker id, marker out of frame, too few pixels, or a black frame. In simulation a camera left at the 1.0 m default near clip returns black RGB while depth still works, which is exactly why this misdiagnoses as a detector bug. |
| Detection works, residual is `marginal` or `poor` | a `marker_length_mm` that does not match the printed board, a flexing mount, motion blur, too little tilt diversity, or glare. A declared 50 mm board that is really 48 mm biases every sample by a scale factor: the solve converges cleanly and is uniformly wrong. Measure the printed black square's edge, not the white border and not what the PDF was named. |
| ArUco detects fine, rotation is nonsense | planar-marker pose-estimation flip ambiguity at near-frontal views. Add oblique views. Both the simulation ArUco path and the real-cell command widen the orientation spread for this reason. |
| Residual `excellent` but picks miss by a **constant** offset | the residual is a mixed Frobenius norm, not millimetres (section 3.1), so it can be excellent and wrong. Print which artifact was loaded and compare its translation to the camera's known mounting position. Random misses are not calibration. |
| A result rotated by a right angle or by 180 degrees | check what it is being compared against before touching the solve. A wrong reference is a harder bug than a wrong solver. |
| Eye-to-hand expected, a `CAMERA -> TOOL` result arrived | the settings object has no `mode` attribute, so `CalibrationRoutine` fell back to eye-in-hand. Always pass `calibration_mode=` explicitly. |
| Grasps went bad after a **gripper change**, camera untouched | eye-in-hand only, and it bites hard because nothing about it looks like a camera problem. The TCP definition moved, so `CAMERA` to `TOOL` is stale. |
| Fusion configured, artifact present, nothing happens, no error | `fusion.enabled` is false. Section 4.2. |

**One physical spec is written in more than one place.** `marker_length_mm` and `aruco_dict_name`
exist under `camera.hand_eye.eye_to_hand`, under `camera.hand_eye.eye_in_hand`, and again under
`camera.cameras.stereo_calibration`. The third belongs to a different subsystem, stereo intrinsics
through ChArUco, and describes a different board, but carries the same key names and the same shipped
values. If the stereo marker path is in use, both trees must describe the same object.

**When to recalibrate:** any collision involving the camera, mount or tool; a tool or gripper change,
eye-in-hand especially; the camera moved, refocused or remounted; the robot base moved; a drift alarm
on the residual channels; grasps that start missing by a constant offset.

---

## 6. What has run, and what has not

**In simulation.** Eye-to-hand and eye-in-hand both run end to end through the real
`CalibrationRoutine`, and the eye-in-hand wrist-camera pick lands on target. Two kinds of run need
separating. The ground-truth marker source is derived from the very oracle the result is compared
against, so its near-zero residual is **circular**: it validates collection, the diversity filters,
the AX=XB solve, the frame tags and the save-load round trip, and says nothing about perception
accuracy. The ArUco runs are the perception-honest ones.

Calibrating in simulation is
[`run_eth_calibrate.py`](../../src/willy_sim/run_eth_calibrate.py) and
[`run_eih_calibrate.py`](../../src/willy_sim/run_eih_calibrate.py) and nothing else. Both add
`--robot-model`, `--profile` and `--radial-closing` on top of their own flags, and artifacts are
namespaced per robot model, under `logs/calibration/`, which is not committed. Their `-h` is the
procedure; no page here carries a copy, because the copy is what goes stale.

**Off-box, with no GPU and no simulator.** The solver, the frame contracts, the serialization round
trips and the resolver map are all exercised by the mock suite, because the whole calibration package
is torch-free.

**Not on hardware.** Nothing in the calibration stack has run against a physical camera or
controller. The real-cell entry point,
[`src/robot/execution/real_cell/calibrate.py`](../../src/robot/execution/real_cell/calibrate.py), drives the same
`CalibrationRoutine` with a live RGB-D ArUco marker source, and its `--check` and `--dry-run` stages
touch nothing and open the camera without moving. Below `--dry-run` it is unproven: that marker
source has never seen a physical camera, and an aligned stream reports distortion coefficients near
zero, so its residual is unconfirmed on a real bench.

---

## 7. Where to read next

| For | Go to |
|---|---|
| The typed extrinsics, the stereo runtime, quality and persistence | [`src/calibration/README.md`](../../src/calibration/README.md) |
| The two workflows, the multi-camera map and their traps | [`src/calibration/eye_hand/README.md`](../../src/calibration/eye_hand/README.md) |
| The bench session: print the board, run the sweep, wire the artifact in | [docs/calibration-setup.md](../calibration-setup.md) |
| Running one camera against a real robot, and the flags | [`src/robot/execution/real_cell/README.md`](../../src/robot/execution/real_cell/README.md) |
| Both modes driven from Python | [`scripts/examples/api/02_calibration/calibrate_fixed_camera.py`](../../scripts/examples/api/02_calibration/calibrate_fixed_camera.py) |
| Config layering, `explain` and `where` | [01](01-configuration.md) |
| Camera intrinsics, and building an arm | [02](02-models.md) . [04](04-robot-and-safety.md) |
| The pick that consumes the transform | [05](05-pick-loop.md) |
