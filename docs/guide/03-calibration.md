# 3. Hand-eye calibration: the derivations

This page is for reasoning about a calibration result, not only producing one: why the problem is
`AX = XB`, what the two mounting cases solve for, what the residual measures (it is not
millimetres), how the CAMERA to BASE transform reaches a grasp and a pixel of your own, and how
each failure shows itself.

A calibration is run by `HandEyeCalibration`, one camera at a time:

```python
from willy import HandEyeCalibration, SweepOptions, load_tree

calibration = HandEyeCalibration.from_tree(load_tree(), rig_id="overhead", mode="eye_to_hand",
                                           options=SweepOptions(freedrive=True, marker_length_mm=40.0))
print(calibration.check())  # the config alone: rig, marker, poses, artifact path; opens nothing
```

`calibration.run(dry_run=True)` then builds the arm and opens the camera without moving, and
`calibration.run()` collects the samples. Nothing generates a pose: either a person guides the arm to
each one by hand (`freedrive`, on an arm that offers hand guiding), or the arm drives to stations
somebody wrote down or taught (`fixed_poses`), each optionally fine-tuned by hand (`adjust`). By hand,
[`examples/real_robot/07_calibrate_a_fixed_camera.py`](../../examples/real_robot/07_calibrate_a_fixed_camera.py)
is a fixed camera and
[`examples/real_robot/09_calibrate_a_wrist_camera.py`](../../examples/real_robot/09_calibrate_a_wrist_camera.py)
a wrist camera; from fixed stations,
[`08_calibrate_a_fixed_camera_with_fixed_poses.py`](../../examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py)
and
[`10_calibrate_a_wrist_camera_with_fixed_poses.py`](../../examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py).

Whichever camera a sweep calibrates, the arm it drives carries every wrist camera body the tree
declares (`camera.cameras.rigs[<id>].body`), placed from that camera's calibration as
`Robot.from_tree` places it, so a fixed camera swept over stations the arm drives to itself plans with
the camera on the wrist and its bracket. `check()` and the build print them on a `wrist cameras`
line. A body that cannot be placed yet, because its camera is not calibrated, refuses `check()` unless
`SweepOptions(unmodelled_wrist_body="<reason>")` (`--unmodelled-wrist-body` on the command line) says
why the arm may move without it, and the report then names the camera it moves without. A wrist rig
declared with no `body` at all (its bracket not measured yet), switched on or off, refuses no sweep,
and neither does the camera an eye-in-hand sweep calibrates when it declares none (that one switched
on, as every camera being calibrated must be): nothing is carried for it, and `check()` and the build
name it on one `!!` warning line, which is logged (section 4.4). That holds for calibration only;
`Robot.from_tree` and a pick still refuse an enabled one. A camera an eye-in-hand sweep calibrates
that does declare its body needs that reason on its first sweep, because nothing can place its body
before it. An eye-to-hand sweep on a tree that declares no wrist camera sweeps as it always did.
The command line is `python -m src.robot.execution.real_cell.calibrate`
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
while the arm is elsewhere. Eye-in-hand when the view has to come from where the tool is going: a part
seen from several **looks**, fused until its grasp is safe ([05](05-pick-loop.md)), or deep bins nothing
static can see into. Only an eye-in-hand camera looks around; a fixed camera fuses with the others where it
stands. Eye-in-hand costs a live TCP read every frame and a recalibration on every gripper change. Many
cells want both.

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

**A stereo pair is calibrated twice.** A webcam pair or a side by side stereo device gives depth only
through its own stereo calibration: both lenses, where the right camera stands from the left, and the
rectification that puts a point on one row of both images, solved from pairs of a ChArUco board
(`StereoCalibration`, [`examples/real_robot/19`](../../examples/real_robot/19_calibrate_a_stereo_pair.py)).
That one says nothing of the robot. Then the pair is calibrated against the robot exactly as an RGB-D
camera is, the same command and the same routine, fixed or on the wrist: its owner hands out the left
image rectified and the rectified left lens with no distortion, the board is posed on that image, and
the transform solved is the rectified left camera's, the frame the pair's depth lies in (the owner,
2026-10-17; [`src/camera/README.md`](../../src/camera/README.md#a-stereo-pair-as-the-cells-camera)).
A stereo pair calibrated again after its hand-eye calibration moves its rectified frame, so the
hand-eye calibration is swept again after it.

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
on the wire are XYZW quaternions, but a stations file's `rx`, `ry`, `rz` are axis-angle, in radians, as
the pendant shows them.

### 2.5 What a pose set must satisfy

`_assert_ready` in [`src/calibration/eye_hand/common.py`](../../src/calibration/eye_hand/common.py)
runs before every solve, in both modes. All three must hold:

1. `len(dataset) >= settings.min_samples`, else `CalibrationDataError: too few samples`;
2. at least **two** relative robot motions that rotate by 1 degree or more, else
   `"sample set needs at least two meaningful robot rotation motions"`;
3. those rotation axes span **two independent directions**, `matrix_rank(axes, tol=0.1) >= 2`, else
   `"sample set needs robot rotations around at least two independent axes"`.

Condition 3 is why a pure-translation sweep can never solve a hand-eye problem, and why the poses,
whether guided by hand or written in a file, have to tilt the tool about more than one axis rather
than only move it.

Two filters drop samples before that, on different data:

- The station screen skips a **commanded** station that is close to one kept before it on *both* axes
  at once (a station of a file, and every joint station, on the grasp centre its joints put the tool
  at).
- `add_sample` accepts a **measured** pose only if, against every stored sample,
  `dist > min_distance_mm` **or** `angle > min_angle_deg`. While a person guides the arm, the preview
  says by this same rule whether the pose the arm stands at would count.

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

**And a third, on every wrist pick that fuses two looks: the hand-eye check.** Two looks of one part
through one hand-eye measure the surface they share in the same place, up to depth noise. The pick takes
the median distance between them, over the surface both saw facing the same way, and prints it on the
report's `hand-eye` line. Above **6 mm** a WARNING says the calibration may have drifted (a camera loosened
on the wrist). It needs 20 shared points, speaks once per pick and changes nothing: the pick goes on. It
cannot see a slide along a surface, so a quiet check is no proof. A WARNING is your cue for the physical
check above.

**And a fourth that moves nothing: the marker's corners, clicked in one frame.**
[`examples/real_robot/18_find_clicked_pixels_in_base.py`](../../examples/real_robot/18_find_clicked_pixels_in_base.py)
captures one frame of the wrist camera with the arm standing still, and finds each corner you click in
BASE from the depth about it (section 4.5). The corners' middle should lie a few millimetres from where
the marker lies, about the size of the calibration's own error, and each edge should measure the
printed edge. Measure where the marker lies by other means than this camera, or the check is circular.

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
  is the declared frame; on a `controller` one it is the frame the driver read from the controller at
  connect.

A `/1` eye-in-hand artifact still loads, with no record, but a rig that declares a camera body refuses
it: calibrate that rig again. Once a body is placed from the record, a tool frame that has moved since
refuses the rig: exactly on `willy`, and beyond the rig's `extrinsics.record_tolerance_mm` and
`record_tolerance_deg` on `controller`, because the body and the pick frame are both stale.

Both types check their frames on write and on read, and both write a temporary file and then
`Path.replace()` it, so a save is atomic. The schema check on load is strict and raises. An artifact
of unknown vintage should force a new measurement rather than be trusted, so do not edit a schema
string by hand to get past it.

### 4.2 The config keys

A camera's calibration is declared on its rig, in the camera section, and one loader opens every
artifact: `RigCalibration.from_config` in
[`src/calibration/rig_calibration.py`](../../src/calibration/rig_calibration.py). Which cameras are
fused follows from the rigs: every enabled RGB-D rig that declares its calibration, where two or more
do and the primary is one of them (`camera_fusion.fused_cameras`). Both resolver builders live in
[`src/robot/execution/autonomous_grasp/builders.py`](../../src/robot/execution/autonomous_grasp/builders.py)
and load through that one loader.

| Key | Holds | Read by |
|---|---|---|
| `camera.cameras.rigs[<id>].extrinsics` | the rig's `mounting_mode` and `artifact_path`, and the tolerances below | `build_config_frame_resolver` (the primary rig) and `build_config_frame_resolvers` |

The tolerances on a rig's `extrinsics`:

- `shutter_motion_tolerance_mm` and `shutter_motion_tolerance_deg` are required on an `eye_in_hand`
  rig, with no default, and refused on `eye_to_hand`.
- `record_tolerance_mm` and `record_tolerance_deg` are required on a rig that declares a `body`, and
  refused on `eye_to_hand`.

The mounting picks the resolver: `eye_to_hand` gives a `StaticCameraToBaseResolver` holding the
`CAMERA -> BASE` artifact, and `eye_in_hand` an `EyeInHandFrameResolver` that composes the
`CAMERA -> TOOL` artifact with the tool pose. The robot section carries no calibration and names no
camera: a tree that still writes the fusion block, `robot.grasping.fusion`, is refused at load, and
`python -m src.config explain` answers each of its old calibration keys with the rig key that took its
place.

**The two builders gate differently.** The singular one returns `None` only when no camera section was
handed in or the primary rig declares no `extrinsics`. The plural one returns `{}` where the cell fuses
no camera, before opening anything. Which cameras a tree fuses, a program asks without opening one:

```python
from src.config.tree import load_tree
from src.robot.execution.camera_fusion import CameraFusionPlan

print(CameraFusionPlan.from_tree(load_tree("ur5e,eth2")))   # fusing 'cam_left' (primary) and 'cam_right' ...
```

On the shipped base tree it prints `no camera fusion:` and what the rigs lack.

### 4.3 What is wired

`AutonomousGraspService.from_robot_config()` is the config-driven boot path. `build_real_cell` and
`build_rehearsal_cell` call it, and they are what `Cell`, `python -m src.robot.execution.real_cell`,
the operator console and the [examples](../../examples/README.md) reach. The Isaac runner
`run_multiview_pick` takes it by default, through `run_eih_pick.build_service`.

On that path, `from_robot_config` builds the singular resolver from the primary rig's `extrinsics`
when the caller passed none, reading the camera section it is handed as `camera=` (`build_real_cell`
passes it). It then fails closed on a real vendor that still has no resolver (section 4.4), and
applies the orchestrator overlays, which reach the plural builder. That one builds a resolver for
every camera the cell fuses, and none where it fuses none.

Two things to carry away:

- **The base camera section declares no rig's calibration**, so a freshly configured real cell is
  refused at build until one is declared. The `ur5e,eth2` example declares both of its rigs', at
  artifact paths a sweep writes and the repository does not ship; until both sweeps have run, its
  build is refused naming the file that does not load.
- **The default pick perceives once and grasps** whatever the calibration says. The recovery every
  cell runs rescans and pushes after a miss, a second fixed camera is fused where the camera section
  declares one with its calibration, and the learned ranker ships `enabled: false` in
  [`config/robot/robot.yaml`](../../config/robot/robot.yaml). See [05](05-pick-loop.md).

### 4.4 Where it fails closed

- `ValueError: no CAMERA->BASE frame resolver for a '<vendor>' cell`

  `from_robot_config` on a real cell with nothing wired. Declare the primary camera's calibration
  on its rig, `camera.cameras.rigs[<primary rig id>].extrinsics`, and hand the root the camera
  section, or pass `frame_resolver=`.
- `RuntimeError: camera.cameras.rigs['x'].extrinsics names <path> (<mode>), which does not load: ...`

  A declared artifact is missing, stale or invalid, on the primary rig or on a fused camera. The
  cell is refused at construction rather than run with a camera it cannot place.
- `CellBuildRefused: this cell plans with cuRobo and calibrates '<rig>', and no live camera world comes of it: ...`

  A cuRobo cell declares a rig's calibration and gets no live camera world from it. Once a rig is
  calibrated its world is mandatory: declare `safety.planning_world.support_plane`, which turns the camera world
  on, with the primary rig among the calibrated ones, or remove the rig's
  `extrinsics` while it is not used. `build_real_cell` refuses before a camera opens, and
  `real_cell --check` blocks on its `camera world` row ([04](04-robot-and-safety.md)).
- `[config] REFUSED: this eye_to_hand sweep of '<rig>' moves the arm that carries wrist camera '<wrist rig>', whose declared body cannot be placed yet. For THIS sweep (--rig <rig>), say why the arm may move without that body: --unmodelled-wrist-body "<reason>". Why the body of '<wrist rig>' cannot be placed: ...`

  A sweep on a cell whose arm carries a wrist camera with a declared `body` that nothing can place
  yet. What follows `cannot be placed:` says why: the rig declares no calibration (it then names the
  separate `--mode eye_in_hand` command that calibrates that camera, with a flag of its own), its
  artifact does not load, it records no flange to TCP, or it records one this cell no longer applies
  (`camera.cameras.rigs['<wrist rig>'] was calibrated against a flange to TCP ...`); each of them is
  excused the same way. An `eye_in_hand` sweep is refused the same way for another wrist camera it
  does not calibrate, starting `this eye_in_hand sweep of '<rig>'`. Calibrate that wrist camera
  first, or add the flag the refusal gives first, `--unmodelled-wrist-body "<reason>"`, to this
  command: the sweep then moves without that body, saying so on its `wrist cameras` line
  (`NOT carried`) and on a `!!` line of the build, which is logged.

  A wrist rig declared with no `body` at all (`eye_in_hand` extrinsics, the bracket not measured
  yet), switched on or off, is deliberately not refused by any sweep, eye to hand or eye in hand of
  another camera, over fixed stations or guided by hand. Neither is the camera an `eye_in_hand`
  sweep calibrates when it declares no body. Nothing is carried for such a camera, it needs no
  reason, and the check and the build print one line, which is logged:
  `!! wrist camera '<wrist rig>' is declared on the arm without a body: the planner and the guard do not know it is there during this sweep`.
  That is for calibration only: `Robot.from_tree`, and with it every pick, still refuses an enabled
  one (`WristBodyRequired`).
- `PickOutcome.CAMERA_FRAME_REJECTED`, at pick time

  A valid candidate existed, but in the camera frame, while `require_base_frame_grasp` was on. The
  calibration is not reaching the pick path ([05](05-pick-loop.md)).

One case is said, not refused: two or more enabled rigs, RGB-D or stereo, of which fewer than two
declare their calibration, or the primary not among the ones that do. The cell grasps from one view,
and its build says so in one WARNING naming what is missing (`CameraFusionPlan`).

### 4.5 A pixel of your own, in BASE

The same composition places one pixel for a program of your own, with no detector and no console: a
program that finds its point in the colour image itself (a pen stroke on paper, a mark, a marker's
corner) takes one frame and asks where its pixels lie in the robot's base frame.
[`src/robot/perception/pixel_to_base.py`](../../src/robot/perception/pixel_to_base.py) holds it, and
[`examples/real_robot/18_find_clicked_pixels_in_base.py`](../../examples/real_robot/18_find_clicked_pixels_in_base.py)
runs it at the cell.

```python
from willy import Camera, PixelRefused, Robot, capture, load_tree, pixel_to_base, pixels_to_base, tcp_above

tree = load_tree()
robot = Robot.from_tree(tree, gripper=None)          # the arm alone: no hand is built, nothing is said to it
with Camera.from_tree(tree) as camera, robot.connected():
    frame = capture(camera, robot)                   # one frame, the arm at rest; nothing moves
try:
    point = pixel_to_base(frame, 412.0, 233.5)       # column, row: a pixel's centre at its whole index
except PixelRefused as refused:                      # never a silent 0: refused.reason says why
    if refused.reason != "no_depth":
        raise
    point = pixel_to_base(frame, 412.0, 233.5, plane=50.0)   # no depth there: the ray on the mat, BASE z 50 mm
print(point)                                         # base_mm, tool_mm, normal, source, valid_share, spread_mm
stroke = pixels_to_base(frame, [(400, 230), (401, 231), (402, 233)])   # every pixel, or the first that failed
above = tcp_above(point.base_mm, point.normal, 20.0) # the TCP 20 mm over it, square to its surface: proposed only
```

- **The frame.** `capture` takes the frame a pick takes, through the `Locator`'s own source with no
  detector (`Locator.capture()`): the warm-ups, the median of a burst of grabs as the depth, the rig's
  mixed pixels taken out, and on the wrist the TCP read before and after the burst, held to the rig's
  `shutter_motion_tolerance_*`. Before the burst the arm's steady gate is asked whether it stands still
  (`wait_until_steady` waiting for nothing: on a UR, which reads steady off its joint speeds since 2026-10-17,
  every joint under 0.01 rad/s on three reads in a row), and a moving arm is refused (`LocatorRefused`)
  with nothing grabbed. The frame keeps the colour image (BGR), the depth aligned to it, the lens, the TCP
  the arm read at the shutter (`frame.tcp_pose`), and three transforms, each a 4 x 4 that maps
  a point of the first frame into the second: `camera_to_tool_mm`, the rig's eye-in-hand artifact as
  `camera.calibration()` loads it (section 4.1), `tool_to_base_mm`, the TCP at the shutter, and
  `camera_to_base_mm`, their product. A fixed camera is `capture(camera)`, placed by its own calibration,
  and asks no arm.
- **The pixel.** Never one pixel's depth: the pixel's viewing ray meets a plane fit to the depth in the
  15 x 15 pixel window about it, so a stroke the camera measured nothing on stands on the paper. Where the
  pixel measured a depth of its own, the plane is the surface that depth lies on, which has to hold a
  quarter of the window; otherwise the surface that holds more than half of what the window measured.
  Each is found among planes through three of the window's points, judged by every point's distance
  from the plane along its own ray, as the camera measures, and fit again by least squares to a band that
  narrows to its own points' deviation: a pixel 1 mm from a part's edge still gives the face it shows, not
  a bevel across the edge. `plane=` meets the ray with a plane you name instead: a number is the
  horizontal plane at that BASE z, a pair `(point, normal)` any plane.
- **What it is worth.** On a ray-cast bench the point comes back within 1e-6 mm, and a BASE point goes
  to its pixel and back within 0.1 mm; on depth rounded to whole millimetres, as a RealSense reads it,
  within 0.5 mm, the most where the surface faces the camera square
  ([test](../../tests/calibration/test_a_pixel_of_your_own_lands_in_base.py)). At the cell what is left is the
  calibration's own error, and example 18 measures it on the marker (section 3.4).
- **What it refuses.** `PixelRefused`, its `reason` one of `outside` (the pixel lies off the image),
  `no_depth` (the camera measured less than a quarter of the window: name a plane), `no_surface` (an
  edge, a step, a stray reading), `parallel` (the ray meets the plane at under a degree) and `behind`.
  Never a silent 0. `pixels_to_base` gives every pixel or names the first that failed, by `index`.
- **The pose.** `tcp_above(point, normal, offset_mm)` stands the TCP `offset_mm` above the point along
  its normal, its +Z onto the surface and its +X as near base -Y as a pose square to the surface allows
  (`closing_axis="-y"`, as the owner's poses stand; over a level surface it is
  `Pose.tool_down(..., closing_axis="-y")`). The TCP is the one `robot.move` places and
  `robot.arm.get_tcp_pose` reads, the controller's active TCP on a `controller` tool frame;
  `fingertips_of=tree.robot` stands the hand's fingertips there instead. It only proposes: nothing is sent to the arm, and a program that wants it there drives it
  through `robot.move` and its guards, once a person said so.
- **Without the console, and beside it.** All of it is a plain script (`load_tree`, `Camera`, `Robot`),
  with no API server and no frontend. A capture on the wrist connects the arm as every connect does (on a
  UR its RTDE control link, the payload the tree declares and the tool frame's check), with no hand built
  and no motion, and `robot.connected()` takes the cell's lock first (`CellLock`, the one the operator
  console, `Cell` and the command line runner take). While another process holds the cell, the console
  connected among them, the connect raises `CellBusy` naming that process before the arm is touched, and
  nothing is captured. The console also keeps its camera open from its build until it builds again or
  stops. So a capture runs with the console stopped, and gives the lock and the camera back as its `with`
  block ends.

---

## 5. What goes wrong, and how it shows up

| Symptom | What it usually is |
|---|---|
| `CalibrationDataError: too few samples` | too many poses dropped; count `POSE_REJECTED` by reason (below) |
| Either rotation error from section 2.5 | too little rotation, or every tilt about one axis; eye-in-hand varies azimuth and elevation |
| Every pose logs `marker_not_found` | wrong dictionary or marker id, marker out of frame, too few pixels, or a black frame (below) |
| Detection works, the residual is `marginal` or `poor` | a wrong `marker_length_mm` (below), a flexing mount, motion blur, too little tilt, or glare |
| ArUco detects, the rotation is nonsense | planar-marker flip ambiguity at near-frontal views; add oblique views (the preview shows each view's tilt while you guide the arm) |
| Residual `excellent`, picks miss by a **constant** offset | not millimetres (3.1); compare the loaded artifact's translation to the camera's mount |
| A result rotated by a right angle or 180 degrees | check the reference it is compared against first: a wrong reference is harder to find |
| Eye-to-hand expected, a `CAMERA -> TOOL` result arrived | settings with no `mode` make `CalibrationRoutine` default to eye-in-hand; pass `calibration_mode=` |
| Grasps went bad after a **gripper change**, camera untouched | eye-in-hand only: the TCP moved, so `CAMERA` to `TOOL` is stale, and it looks like no camera fault |
| A wrist pick WARNs that its looks measure the part's shared surface more than 6 mm apart | eye-in-hand only: the hand-eye has drifted, the camera or the tool moved on the flange; run the physical check (3.4) |
| Two cameras calibrated, the pick stays single-view | a rig is switched off, or the primary is not among the calibrated ones: the build's WARNING names it (section 4.4); a stereo pair whose stereo calibration is gone is refused as its camera opens, naming the file |

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
describes a different board, but carries the same key names and the same shipped values. A stereo pair
is solved against that board (`StereoCalibration`) and then swept against the robot with the hand-eye
target, so both must describe the boards you actually have.

**When to recalibrate:** any collision involving the camera, the mount or the tool; a tool or gripper
change, eye-in-hand especially; the camera moved, refocused or remounted; the robot base moved; a drift
alarm on the residual channels; a hand-eye WARNING from the looks of a wrist pick; grasps that start
missing by a constant offset.

---

## 6. What has run, and what has not

**Pure logic, pinned by the mock suite with no GPU and no simulator** (the calibration package is
torch-free): the solver, the frame contracts, the artifact round trips and the resolver map.

| Capability | Evidence |
|---|---|
| Eye-to-hand and eye-in-hand through `CalibrationRoutine` | measured in simulation: both run end to end, and the eye-in-hand wrist-camera pick lands on target |
| The real-cell sweep through `HandEyeCalibration` | run on a physical cell: a wrist D415 on a UR10 (CB3), against one printed ArUco marker; a fixed camera and a ChArUco board have not, and the mock suite covers the flow |
| The same sweep on a stereo pair | never run on a webcam pair or a stereo device; pinned on a rendered pair whose right camera is turned three degrees: the marker is posed on the left image rectified within 6 mm and 4 degrees of where it lies, where the raw lens puts it 48 mm off ([test](../../tests/camera/setup/image_taking/test_a_stereo_rig_is_calibrated_against_the_robot.py)) |

**Two kinds of simulated run.** The ground-truth marker source is derived from the same oracle the
result is compared against, so its near-zero residual is **circular**. It validates collection, the
diversity filters, the AX=XB solve, the frame tags and the save-load round trip, and says nothing
about perception accuracy. The ArUco runs are the ones that test perception.

**Calibrating in simulation** is [`run_eth_calibrate.py`](../../src/willy_sim/run_eth_calibrate.py) and
[`run_eih_calibrate.py`](../../src/willy_sim/run_eih_calibrate.py). Both add `--robot-model` and `--profile`
to their own flags, and write artifacts per robot model under
`logs/calibration/`, which is not committed. Their `-h` is the procedure. They run declared stations,
not generated ones: `src/willy_sim/calibration/stations/eth_<model>.json` and `eih_<model>.json` for
the three models the sim cell drives, or `--stations PATH`. Those files were frozen from the generators
this repository used to carry. On the ur5e they ran in Isaac in aurora on 2026-09-26, the same files:
ETH ground truth 22 of 22, ETH ArUco 10 of 22 at `rmse_mm` 3.44, quality `marginal`, 6.3 mm and 0.49
degrees from the oracle, which the runner's gate accepts for ArUco; EIH ground truth and ArUco 21 of 24
each, the ArUco solve at `rmse_mm` 2.63, quality `marginal`, 4.4 mm and 0.08 degrees from the oracle,
which the EIH runner's gate (good, under 5 mm) calls `CHECK`. The same EIH run on aurora's tree before
these files existed, on the same 24 views, counted 20 of 24 and missed the oracle by 5.8 mm and 0.18
degrees, `CHECK` as well. This repository's runners have not run them in Isaac, and the ur3e and ur10e
files have not run in Isaac at all.

**The real-cell sweep.** [`src/robot/execution/real_cell/calibrate.py`](../../src/robot/execution/real_cell/calibrate.py)
is a caller of `HandEyeCalibration` in [`src/robot/execution/hand_eye.py`](../../src/robot/execution/hand_eye.py).
It drives the same `CalibrationRoutine` with a live RGB-D ArUco marker source. `--check` reads the
config only, and `--dry-run` opens the camera and builds the arm without moving. Past `--dry-run`,
`HandEyeCalibration` and its RGB-D marker source have calibrated one wrist D415 at a physical cell. An
aligned stream reports distortion coefficients near zero, so the source's residual stays unconfirmed: no
bench residual is kept here. The two ways by hand, `--freedrive` and `--adjust`, go through the vendor's
hand guiding (`SupportsFreedrive`, on a UR its teach mode); that D415 was calibrated by hand in
freedrive (examples 09 and 10), and their console, stillness gate, boundaries and stations file are
exercised against a scripted arm in the offline suite.

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
| Both modes driven from Python, by hand and from fixed stations | [`07`](../../examples/real_robot/07_calibrate_a_fixed_camera.py), [`08`](../../examples/real_robot/08_calibrate_a_fixed_camera_with_fixed_poses.py), [`09`](../../examples/real_robot/09_calibrate_a_wrist_camera.py), [`10`](../../examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py) |
| Config layering, `explain` and `where` | [01](01-configuration.md) |
| Camera intrinsics, and building an arm | [02](02-models.md) . [04](04-robot-and-safety.md) |
| The pick that consumes the transform | [05](05-pick-loop.md) |
