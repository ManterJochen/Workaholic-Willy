# Runbook: the first pick on a physical arm

**Scope.** The procedure that starts with an arm sitting powered off on a bench and ends with one
verified, logged pick. It is the hardware counterpart to
[`ur3e_cell_bringup.md`](ur3e_cell_bringup.md), which covers the configuration and geometry side and
verifies itself in simulation. This one takes the same cell to metal.

**Nothing in this repository has ever run against a physical controller.** The rehearsal, the
configuration preflight and the connect path have been driven against a dummy arm and against real
controller software in a simulator container. Everything past that is untested on hardware, and the
ordering below is what makes an untested path survivable: each step de-risks the next, and a step
that fails tells you which thing is wrong instead of only that it does not work.

**The one line version.** Do the desk work until `--check` is clean, prove the camera without the
robot, prove the robot without the camera, calibrate, then join them, and never skip a stage because
the previous one probably passed.

---

## Trigger

Any of:

1. A physical arm is on the bench for the first time.
2. The end effector, the coupling plate or the camera mounting changed. Each of those invalidates a
   measured value the stack depends on.
3. `python -m src.robot.execution.real_cell --check` reports blocking items.
4. The cell connects but rejects every motion, or executes motions that miss by a constant offset.
5. Picks are logged as successes while the gripper is visibly empty.

---

## Diagnose

Work top to bottom. Each stage has a command that answers it and a failure that looks like something
else.

### 1. Is the SDK installed?

```bash
python -m src.robot.drivers.doctor --require ur
```

Exit 0 means the vendor is ready. A miss names the modules that would not import, `rtde_control` and
`rtde_receive` on the UR path, and the fix is `pip install -r requirements.txt`, which pins
`ur_rtde`.

### 2. Is the configuration runnable at all?

```bash
python -m src.robot.execution.real_cell --check
python -m src.robot.execution.real_cell --check --profile ur3e
```

Against the shipped configuration tree as a UR cell this reports three blocking items, and that is
the normal state of a fresh cell rather than a fault.

| Blocking | Why it stops you | What it looks like if you skip it |
|---|---|---|
| `gripper.tool_frame.source` is `undeclared` | nobody has said where the grasp centre sits on the flange | a top-down grasp commanding z = 37 mm drives the flange there and the fingertips through the bench |
| `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses, because pushing `setPayload(0.0)` would overwrite the controller's model of a mounted tool | you drive to the cell and cannot connect |
| no `CAMERA->BASE` resolver | grasps stay in the camera frame | every motion returns `INVALID_TARGET`, which reads as a cell that hangs |

Four more items are reported as warnings and never block: an unset self-collision kinematics model,
no declared fixtures, no declared planning world, and no record log path.

Two rows are reported `[bench]` and never block, because no interface answers them:

* the controller must be powered, with brakes released, in Remote Control, with no pendant program
  owning it, because `ur_rtde` uploads a control script and is refused otherwise;
* the end effector's electrical side. The Robotiq URCap opens port 63352 and is not any pip package;
  a vacuum tool needs its 24 V supply and its solenoid. The schema accepts I/O pins 0 to 7 on any
  bank and does not know how many outputs the bank you named actually has, so a pin the tool block
  does not carry validates cleanly and switches nothing.

A blocking checklist does not necessarily stop you connecting. A cell with blocking items can
connect and then refuse every motion, which is the failure that reads as a broken robot. What
refuses the connect is the driver's own preflight, not this checklist.

`config/robot/robot.ur5e.yaml` is the worked example for a real bench. It leaves exactly those three
items unset on purpose and writes them out as commented blocks saying what to measure, so the
shipped refusals stay armed.

### 3. Does the whole software path work, with no hardware?

```bash
python scripts/examples/cell/01_robot_setup.py            # the config side, commands nothing
python -m src.robot.execution.real_cell --rehearse --runs 3
```

The rehearsal swaps the vendor to `dummy` and a synthetic scene in, keeps your profile chain,
gripper branch and grasping block, and runs the whole path in milliseconds. The runner's verdict
rule is unanimity, so three of three must succeed. This is the last thing that is free: if the
rehearsal fails, the fault is in the software and finding it at a bench costs a day.

A rehearsal continues past a blocking checklist on purpose, which is why `--rehearse --check` and
`--check` can exit differently on the same tree.

The rehearsal is not grasp-quality evidence. The dummy arm carries no safety preflight, and
`Cell.safety()` says so, so it proves the wiring and nothing about what would refuse a bad command.

### 4. Does the arm answer?

```bash
python scripts/examples/cell/01_robot_setup.py --live      # connects, reads back, commands no motion
```

This proves the network path, the SDK, the configuration and the controller state read. `connect()`
on the UR path is where four fail-closed checks run: the payload, the tool frame, the robot model
and the controller's active tool register. The model check asks the controller which model it is,
through the dashboard, and refuses a parsed, positive disagreement with `robot.ur.model`, on the
size class alone, because an e-Series simulator container reports `UR3` for a UR3e. It fails open on
absent evidence, because a false refusal is a cell that will not start, so also confirm `ur.model`
by eye against the label on the arm: the self-collision guard derives its link lengths from that
value and nothing raises if it is wrong.

**The gripper connect does move.** `GripperController.connect()` activates the hand, and Robotiq
activation is a calibration: the fingers travel their full range. That is deliberate, because the
alternative is that it happens lazily on the first commanded width, with the fingers already around
an object. Connecting is motion, which is also why a campaign connects once around N picks rather
than N times. Keep hands clear whenever a runner reaches its connect stage, and have the arm in its
starting pose. The example above touches only the arm and still moves nothing.

### 5. Does the camera answer, with no robot?

```bash
python -m src.robot.perception --prompt "<your object>"
```

One RGB-D frame, detected and segmented, with statistics printed and no robot involved.

Read the depth-hole fraction inside each mask. A D435 returns zeros on dark, shiny and steeply
angled surfaces, and a mask that is mostly holes still yields a confident grasp point computed from
whatever few pixels survived. The line reports the holes against the mask size and the resulting
grasp plane beside it, so you can see what the sensor measured and what the robot would drive to.
That is the honest health signal, and it is also the one thing most likely to differ from
simulation.

The fraction is measured on the streamer's output, so it is post-filter. Turning on
`camera.cameras.rigs.realsense.post_processing.hole_filling` makes a blind camera report close to
zero holes, honestly, because the filter really did fill them. That is why the key ships `false`: a
hole is a surface the sensor could not measure, and filling it invents surface at an invented
distance that a grasp is then planned against.

**Two cameras?** Read their serials and put one on each rig before enabling them. With two identical
D435s the serial is the only stable identity: `RealSenseRGBDStreamer` binds by serial and never
reads `device_index`, which is an OpenCV capture index and survives neither a reboot nor a replug.
Cameras that swap do not fail. They hand back a complete, plausible scene with the views exchanged,
so every fused position is mirrored and the pick goes confidently to the wrong place. The schema
refuses two enabled RGB-D rigs where either has no serial, and refuses a serial shared by both.
That is the guard, not an obstacle.

`config/camera/cam.eth2.yaml` is the worked two-camera layer, and it takes the serials from the
environment:

```bash
rs-enumerate-devices -s
export WILLY_CAM_LEFT_SERIAL=...  WILLY_CAM_RIGHT_SERIAL=...
WILLY_PROFILE=ur5e,eth2 python -m src.config                    # must still validate
WILLY_PROFILE=ur5e,eth2 python -m src.robot.perception --rig cam_left
WILLY_PROFILE=ur5e,eth2 python -m src.robot.perception --rig cam_right
```

Rig order is not cosmetic. The first RGB-D rig is the primary camera: the grasp is synthesised from
its view and every other camera only confirms what it sees.

### 6. Can the planner actually plan for this robot?

Only relevant while `robot.ur.motion_planner` is `curobo`, which is the shipped value.

```bash
python -m src.robot.safety.planning --check --model ur3e    # exit 0 means fully anchored
python -m src.robot.safety.planning --doctor --model ur3e   # and that the engines actually load
```

The difference between the two matters here. `--check` is spawn-free: `cuRobo planner: AVAILABLE`
means the sidecar's Python exists on disk and nothing more, and the banner says so in the same
breath, because the robot descriptor `{model}.yml` lives inside the cuRobo installation in a
separate environment this process deliberately does not spawn. A UR3e configured with no `ur3e.yml`
passes that gate and fails later, inside the sidecar. `--doctor` does spawn it, and reports whether
the descriptor is present.

To list what descriptors exist by hand, in the cuRobo environment:

```bash
"$WILLY_CUROBO_PYTHON" -c "from curobo.content import get_content_root; import os, glob; \
  d = os.path.join(str(get_content_root()), 'configs', 'robot'); \
  print(sorted(os.path.basename(p) for p in glob.glob(d + '/ur*.yml')))"
```

If your model is missing, build it with `scripts/curobo/build_ur_config.py`, on the box and against
the installed cuRobo version, before relying on `motion_planner: "curobo"`. Falling back to `"ik"`
is a valid answer for a first bring-up: it plans nothing, so it cannot plan wrongly, and the cost is
that blind IK proposes self-colliding branches the guard then rejects.

### 7. Is the cell calibrated?

Not with a simulator runner. `run_eth_calibrate` under `src/willy_sim/` calibrates a simulated
camera against a simulated arm, which is useful for proving the routine and useless for this cell.
The real command is one per rig:

```bash
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --check     # touches nothing
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --dry-run   # opens the camera
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --poses 22  # moves the robot
```

It writes `eth_<rig_id>.json` under `calibration/real` and prints the
`robot.grasping.fusion.cameras` entry to paste. Run it once per camera. Until a camera is in that
map it does not reach the pick path at all: geometry fusion stands down to a single view and says so
only in telemetry.

Two traps around it:

* Measure the printed ArUco board before the first sweep. A wrong `--marker-length-mm` scales every
  sample uniformly, so the solve converges and is uniformly wrong, which no residual will tell you.
  It is the edge of the black square in millimetres, not the white border and not what the PDF was
  called.
* The primary camera's artifact also has to be named in
  `robot.grasping.fusion.extrinsics_artifact_path`. The pasted `cameras` block alone leaves that key
  null, and a cell can be fully calibrated, hold both artifacts, and still be refused at build.

The sanity check that beats any residual: command a known TCP pose, detect the marker, and confirm
that the camera-to-base transform predicts the arm's own forward kinematics to within a few
millimetres.

Multi-view is the reason to do this for every camera rather than one. A grasp is synthesised from
the primary camera's cloud, and a surface it never measured is a surface the finger placement is
guessing at.

---

## Mitigate

Fix in this order. Each is a configuration edit plus a bench measurement, so do the measurements in
one session, because three of them share the same setup.

1. **Weigh the whole assembly** and set `safety.payload.mass_kg`. Not the gripper alone, but the
   gripper plus the coupling plate, every cable and hose that rides on the wrist, and any workpiece
   the arm carries. No starting value ships on purpose: a gripper's datasheet mass is not this
   number, and a plausible default is how a cell ends up telling the controller about a tool it is
   not carrying. A genuinely bare flange is `enforce: false`, not a mass of zero. Bring a bench
   scale; it is a procurement item with a lead time.
2. **Measure the centre of gravity** from the flange, in the same session, and set
   `safety.payload.cog_mm`. Leaving it at `[0, 0, 0]` declares a multi-kilogram tool to be a point
   mass at the flange face. For a 3 kg tool whose centre of gravity sits 132 mm out, that is
   3.9 N m of wrist torque the controller does not model, both in the protective-stop calculation
   and in the gravity compensation behind `get_tcp_wrench()`. `connect()` refuses a declared mass
   left at an all-zero centre of gravity for exactly this reason, before it opens a socket.
3. **Measure the flange to grasp-centre transform**, same session, and set `gripper.tool_frame`:
   * `offset_mm`, where the grasp centre sits;
   * `rotation_quat_xyzw`, which flange axis the jaws close along. This half never crashes. Ninety
     degrees out produces run after run of logged successes with the jaws closing across the wrong
     object axis, and hand-eye calibration cannot catch it, because it solves for whatever frame
     `get_tcp_pose()` reports and returns an excellent residual either way;
   * `source`, either `"willy"`, where this driver composes flange to TCP and the controller runs a
     bare flange, or `"polyscope"`, where an operator set the TCP on the pendant and the driver only
     verifies it. Either way `connect()` derives what the controller is actually running and refuses
     a mismatch. The schema also rejects a declared source left at identity, so half a declaration
     does not pass either.
4. **Calibrate eye-to-hand**, point `grasping.fusion.extrinsics_artifact_path` at the primary
   camera's artifact, list every other camera under `grasping.fusion.cameras`, and set
   `grasping.fusion.enabled: true`. That key also arms the shadow voxel substrate, which ingests
   perception frames and emits telemetry. It changes no grasp and no motion, because the gate that
   would let fused evidence decide is `fusion.commit_policy.enabled` and that stays off, but it is
   not free at runtime.
5. **Declare the bench** under `safety.self_collision.fixtures`. With none declared there is no
   surface in the collision world, so nothing refuses a motion that goes through it. Declare
   `safety.planning_world` too: without it cuRobo plans against its own generic table, and the
   one-shot guard cannot see the difference, because that guard judges a destination and not a path.
6. **Set `grasping.record_log_path`.** A bring-up with no telemetry cannot be diagnosed afterwards,
   and it gives the soak, KPI and offline learning tooling nothing to read.
7. **Set `safety.self_collision.kinematics_model`** explicitly to the arm you are holding.

---

## Verify

In order, and do not skip forward on a probably:

```bash
python -m src.robot.execution.real_cell --check                 # 0 blocking
python -m src.robot.execution.real_cell --rehearse --runs 3     # 3/3, no hardware
python -m src.robot.execution.real_cell --dry-run               # builds, connects nothing
python -m src.robot.execution.real_cell --runs 1                # one pick, hand on the stop
python -m src.robot.execution.real_cell --runs 10 --profile ur3e
python -m src.robot.grasping.replay --records logs/<run>.jsonl  # roll up what happened
```

What passing means:

* the executed grasp reports `frame = base` rather than `camera`, which is the single sharpest tell;
* `connect()` logs the tool frame as verified within tolerance;
* the gripper verifier fails an empty close. Test that deliberately: command a pick at an empty
  table and confirm the run is reported as a failure. On a jaw cell the verifier is
  `WidthDeltaGripperVerifier`, the only one available, because the Robotiq driver exposes no
  object-detection capability;
* every failure is classified by the record's typed reason rather than by eyeball.

Know which rule you are being judged by. This runner's default is unanimity: every pick must
succeed, which is stricter than the simulator gate, and it has no independent confirmation of its
own, so a cell that built a `NullGripper` would report success every time. The runner prints the
rule alongside the verdict. From Python the rule is an argument:
`PickRun.from_cell(cell, runs=10, rule=PassRule(fraction=0.8, confirm=...))`.

Exit codes: 0 the campaign passed, or `--check` and `--dry-run` were satisfied; 1 the preflight
blocked or the build or the connect refused; 2 the cell connected and the campaign did not pass its
rule; 3 an exception escaped a pick.

---

## Rollback

* **Stop motion**: the physical emergency stop. Nothing in software outranks it.
* **Back out a configuration change**: every value above is a configuration key, so revert the YAML
  and re-run `--check`. Nothing here writes to the controller except `setPayload`, and that is gated
  on `safety.payload.enforce`.
* **Restore the controller's payload**: if a wrong `mass_kg` was pushed, set the correct value and
  reconnect, or set `enforce: false` and enter the payload on the pendant. The driver rolls the
  connection back on a failed push, so a failed push never leaves a half-set payload behind.
* **The tool frame**: `source: "undeclared"` returns the cell to refusing to connect, which is the
  safe state rather than a broken one.
* **A bad calibration**: point `extrinsics_artifact_path` back at the previous file. The real-cell
  calibration writes under `calibration/real` keyed by `rig_id`, and the simulator runners write
  under `logs/calibration/<robot_model>/`, so the two cannot overwrite each other and one arm's
  simulated run cannot overwrite another's. Copy an artifact aside before moving a camera.
* **Everything else**: `--rehearse` still runs with no hardware attached, so you can always get back
  to a known-good software baseline without the robot.

---

## See also

- [`ur3e_cell_bringup.md`](ur3e_cell_bringup.md), the configuration and geometry side, verified in
  simulation.
- [`real_cell` package](../../src/robot/execution/real_cell/README.md), what the runner does at each
  stage, and the calibration command in full.
- [UR driver](../../src/robot/drivers/ur/README.md), the SDK and the connect-time refusals.
- [`scripts/examples/`](../../scripts/examples/README.md), the same composition driven from Python,
  in dependency order, with nothing moving unless you pass `--live`.
- [`src/calibration/`](../../src/calibration/README.md), the typed extrinsics and what each frame
  means.
