# Runbook: the first pick on a physical arm

**Scope.** The procedure that starts with an arm sitting powered off on a bench and ends with one
verified, logged pick. It is the hardware counterpart to
[`cell_bringup.md`](cell_bringup.md), which takes a cell from its configuration layer through a
simulator to an arm that answers. That one proves the configuration and the geometry, and this one
takes the same cell to metal.

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

Against the shipped configuration tree as a UR cell this reports seven blocking items, and that is
the normal state of a fresh cell rather than a fault.

| Blocking | Why it stops you | What it looks like if you skip it |
|---|---|---|
| `tool frame`: `gripper.tool_frame.source` is `undeclared` | nobody has said where the grasp centre sits on the flange | a top-down grasp commanding z = 37 mm drives the flange there and the fingertips through the bench |
| `payload`: `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses, because pushing `setPayload(0.0)` would overwrite the controller's model of a mounted tool | you drive to the cell and cannot connect |
| `camera -> base`: no CAMERA to BASE resolver | grasps stay in the camera frame | every motion returns `INVALID_TARGET`, which reads as a cell that hangs |
| `camera world`: no live camera world for the planner | the base tree plans with cuRobo, and a cuRobo motion needs a live camera world or a decline; the pick service declines nothing | every pick motion is refused as `UNSUPPORTED` before it moves, naming the missing world |
| `planner margin`: `safety.self_collision.planner_margin_mm` is undeclared | the base tree plans with cuRobo, and a planner never starts without the clearance it keeps; undeclared is not zero | the first planned move is refused as `CONTROLLER_REJECTED`, before the arm moves |
| `carried part`: `safety.planning_world.payload.length_mm` is undeclared | the base tree plans with cuRobo, which models the part a grasp carries, and no length is implied for it | the attach is declined and every lift and transit after a grasp is planned as if the hand were empty; declare the length, or `enabled: false` for a cell that carries nothing |
| `hand`: `gripper.model` is unset | the exact-mesh guard checks the hand `robot.gripper.model` names, and the base tree names none so that no overlay inherits one | the cell refuses to build, naming the key |

A box without the cuRobo environment or without Coal blocks on two more rows, `cuRobo environment`
and `exact mesh engine`: they are facts about the machine the checklist runs on, and they clear
when the `ext_deps` install is on the box.

Four more items are reported as warnings and never block: an unset self-collision kinematics model,
no declared fixtures, no declared planning world, and no record log path.

Two rows are reported `[bench]` and never block, because no interface answers them:

* the controller must be powered, with brakes released, in Remote Control, with no pendant program
  owning it, because `ur_rtde` uploads a control script and is refused otherwise;
* the end effector's electrical side, named for the configured driver, whose row reads the pins and
  the address from the configuration. The Robotiq URCap opens port 63352 and is not any pip package;
  a vacuum tool needs its 24 V supply and its solenoid. The schema accepts I/O pins 0 to 7 on any
  bank and does not know how many outputs the bank you named actually has, so a pin the tool block
  does not carry validates cleanly and switches nothing. A `jaw_io` toggle with no open switch
  (`actuation: single_toggle`) cannot read its jaws, so the driver counts its own pulses and keeps
  that count on disk between programs (`logs/robot/state`, one record per controller, bank and
  pin). A connect starts from the record, and only with no record at all takes the jaws to stand
  open. After the jaws moved any other way (the pendant's I/O tab, a bench `--pulse`, `--set` or
  `--measure` on that pin, a power cut mid stroke), look at them and say where they stand:
  `python -m src.robot.drivers.ur --profile <cell> --jaws-stand open --yes` (or `closed`). A bench
  write on the toggle's pin marks the record before the edge, and the driver refuses to pulse until
  you have. `--jaws open` (or `closed`) moves them through the driver and its count. Never push
  them by hand: a device that keeps its own flip state is not moved by a hand.

A `jaw_io` hand adds a warning, `jaw travel time`, while `close_settle_s` is the schema default
0.3 s and it is the only wait the driver has: with no switch to read, the arm moves on that long
after the edge, on a close and on an open, whether the jaws have arrived or not. A Hand-E on its
I/O coupling can take about 2 s. Time the full stroke both ways by stopwatch or video and set
`close_settle_s` to the longer, with a margin.

A blocking checklist does not necessarily stop you connecting. A cell with blocking items can
connect and then refuse every motion, which is the failure that reads as a broken robot. What
refuses the connect is the driver's own preflight, not this checklist.

`config/robot/robot.ur5e.yaml` is the worked example for a real bench. It leaves the first three of
those items unset on purpose and writes them out as commented blocks saying what to measure, so the
shipped refusals stay armed.

### 3. Does the whole software path work, with no hardware?

```bash
python scripts/checks/cell_bringup.py            # the config side, commands nothing
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
python scripts/checks/cell_bringup.py --live      # connects, reads back, commands no motion
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
python -m src.robot.safety.planning --check --model ur3e --hand robotiq_2f85    # exit 0 means fully anchored
python -m src.robot.safety.planning --doctor --model ur3e --hand robotiq_2f85   # and that the engines actually load
```

The difference between the two matters here. `--check` is spawn-free: `cuRobo planner: AVAILABLE`
means the sidecar's Python exists on disk and nothing more, and the banner says so in the same
breath, because the robot descriptor `willy_{model}.yml`, one per arm with no hand in it, lives
inside the cuRobo installation in a separate environment this process deliberately does not spawn.
A UR3e with no `willy_ur3e.yml` passes that gate and fails later, inside the sidecar. `--doctor`
does spawn it, and reports whether the descriptor is present. Once the sidecar is up it reports the
descriptor's `_provenance` and hashes, and the driver refuses a descriptor built for another arm,
one that carries a hand, or one that records nothing; it then keeps the planner only on a
combination of arm, hand, plate, placement and margin that a committed evidence file measured (the
checklist's `planner margin` row says which file).

To list what descriptors exist by hand, in the cuRobo environment:

```bash
"$WILLY_CUROBO_PYTHON" -c "from curobo.content import get_content_root; import os, glob; \
  d = os.path.join(str(get_content_root()), 'configs', 'robot'); \
  print(sorted(os.path.basename(p) for p in glob.glob(d + '/willy_ur*.yml')))"
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
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --dry-run   # builds the arm, opens the camera
python -m src.robot.execution.real_cell.calibrate --rig realsense_d435 --poses 22  # moves the robot
```

The sweep takes the cell lock and connects the arm alone, with no gripper. While the operator console
holds this cell the sweep exits 1 and names the holder, so end the console's session first. A refused
connect exits 1 as well, and exit 3 is a sweep that raised.

It writes `eth_<rig_id>.json` under `calibration/real` and prints the rig block to paste into the
camera section, `camera.cameras.rigs[<rig_id>].extrinsics`. Run it once per camera. Until the primary
rig declares its block the cell has no camera-to-base transform, and a real cell is refused at build,
naming that key. A second camera also needs an entry naming its rig in
`robot.grasping.fusion.cameras`. Without one it never reaches the pick path: geometry fusion stands
down to a single view, and the build says so in one warning.

Measure the printed ArUco board before the first sweep. A wrong `--marker-length-mm` scales every
sample uniformly, so the solve converges and is uniformly wrong, which no residual will tell you. It
is the edge of the black square in millimetres, not the white border and not what the PDF was called.

The sanity check that beats any residual: command a known TCP pose, detect the marker, and confirm
that the camera-to-base transform predicts the arm's own forward kinematics to within a few
millimetres.

Multi-view is the reason to do this for every camera rather than one. A grasp is synthesised from
the primary camera's cloud, and a surface it never measured is a surface the finger placement is
guessing at.

### 8. Does a wrist camera look at the work before it locates?

Only for a camera the arm carries (`eye_in_hand`). A fixed camera sees the work from where it is
mounted, and nothing here applies to it.

A wrist camera sees what the arm points it at. Asked to locate from wherever the arm stands, it
sees whatever that is: after a connect, where the last program left the arm; in a campaign, the
retreat 100 mm above the last grasp, which for a camera tilted 45 degrees on the wrist is nearer
than a D415 measures and is turned by that grasp's yaw. Neither the locator nor the pick service
moves the arm first, so the examples do (`examples/real_robot/11`, `13` and `18`):

```python
view = viewing_pose(camera, WORK_MM, arm=robot.arm, distance_mm=500.0, closing_axis="-y")
print(view)                        # FOUND, NOT NEEDED (a fixed camera) or REFUSED with its reason
if view.pose is not None:
    print(robot.move(view.pose))   # planned and judged as any move
```

* **It needs the eye-in-hand calibration** (step 7, or examples 09 and 10): the pose is aimed from
  the CAMERA->TOOL the sweep measured, so the camera, not the tool, looks at `WORK_MM`. A camera
  with no calibration is refused, as everywhere.
* **`WORK_MM`** is a point on the table where the parts lie, in base millimetres. The camera
  stands `distance_mm` from it, looking straight at it. A part's top is nearer the camera than the
  table by up to its height.
* **The distance has to be one the depth measures.** Intel gives a D415 a minimum depth of about
  450 mm at the schema's default 1280 x 720 depth mode and about 310 mm at 848 x 480 (datasheet
  figures, not measured here). Where the rig names its camera (`body.model: realsense_d415`), a
  distance nearer than its depth mode's is refused as `too_near`. Stand the camera 500 mm off, or
  stream depth at `depth_resolution: [848, 480]` with colour kept at `[1280, 720]`.
* **`closing_axis`** is the heading the tool keeps; the tool only tilts from `Pose.tool_down` with
  it, as far as aiming needs. Use the one the calibration ran with, so the image is the way up it
  was then.
* **It is screened as a calibration station is**: the workspace box less its margin (TCP and
  flange), the arm's reach, and the joint window half a turn about `robot.home_joint_positions`
  that keeps a cable along the arm from winding. Set the home to one facing the work, because the
  window is centred on it; a home whose wrist 2 stands half a turn from where the arm is leaves
  the arm on the window's seam, and every move is refused until the home is changed. A refusal
  names its reason, and then nothing is located.
* **11, 13 and 18 empty the hand first.** Example 11 and the pick service place nothing, and a
  toggle's record carries jaws left closed into the next program, so the next pick's pre-open
  would drop the part wherever the arm then stands, the viewing pose included. So each viewing
  move is preceded by a release where the last pick left the arm, which lets the part fall back
  where it was picked, as the pre-open always did. A cell that keeps its parts places them between
  picks (`robot.place`).

Measured on 2026-09-23 against the CB3 URSim (PolyScope 3.15.8, UR10 profile, Hand-E driven as
`jaw_io` on tool DO0, 156.2 mm tool frame, planner margin 4 mm, the half-turn window about an
elbow-up home facing the work) with the cuRobo sidecar on the GPU, twice, with the same result.
Example 10 now aims seventeen views. These numbers are from its first set of eleven, and the
seventeen have not been measured this way yet. The eleven views round the marker at (-130, -700,
50) mm, aimed from the stated 45-degree mount (URSim has no camera, so nothing was captured):
8 moved, each as 2 `moveJ` from 21 to 61 planned waypoints, the largest turn of any joint in one
move 113 degrees (wrist 3), no joint sample outside the window. The 3 views 15 degrees above the ring were refused by the arm with
nothing moved: the configuration nearest the arm folds the forearm onto wrist 2 (0.1 to 1.3 mm
inside the planner's margin), and cuRobo's own choice lay outside the window. The viewing pose
then moved as 2 `moveJ` from where the sweep ended (64 degrees at most) and from home (6 degrees).

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
   * `offset_mm`, where the grasp centre sits. `--check` holds it to the hand: along the declared
     approach it should be the registry's `grasp_centre_mm` plus the coupling plates, and the
     `grasp centre` row warns beyond 1 mm. Whichever of the bench and the registry file is wrong is
     the one to correct;
   * `rotation_quat_xyzw`, which flange axis the jaws close along. This half never crashes. Ninety
     degrees out produces run after run of logged successes with the jaws closing across the wrong
     object axis, and hand-eye calibration cannot catch it, because it solves for whatever frame
     `get_tcp_pose()` reports and returns an excellent residual either way;
   * `source`, either `"willy"`, where this driver composes flange to TCP and the controller runs a
     bare flange, or `"polyscope"`, where an operator set the TCP on the pendant and the driver only
     verifies it. Either way `connect()` derives what the controller is actually running and refuses
     a mismatch. The schema also rejects a declared source left at identity, so half a declaration
     does not pass either.
4. **Calibrate eye-to-hand** and declare the primary camera's artifact on its rig,
   `camera.cameras.rigs[<primary rig id>].extrinsics`, by pasting the block the calibration command
   prints; it is read whether or not `grasping.fusion.enabled` is on. To fuse every other camera,
   declare each on its own rig, name it under `grasping.fusion.cameras`, and set
   `grasping.fusion.enabled: true`. That key also arms the shadow voxel substrate, which ingests
   perception frames and emits telemetry. It changes no grasp and no motion, because the gate that
   would let fused evidence decide is `fusion.commit_policy.enabled` and that stays off, but it is
   not free at runtime.
5. **Declare the bench** under `safety.self_collision.fixtures`. With none declared there is no
   surface in the collision world, so nothing refuses a motion that goes through it. Declare
   `safety.planning_world` too, with a measured `support_plane` and `perceived.enabled`: a cuRobo
   cell plans every pick motion against the live world the calibrated cameras build, and the one-shot
   guard cannot stand in for it, because that guard judges a destination and not a path. Without that
   world a cell with a calibrated RGB-D rig refuses to build, and the `camera world` row blocks.
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
rule; 3 a fault of the cell ended a pick, raised or reported.

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
* **A bad calibration**: point the rig's `extrinsics.artifact_path` back at the previous file. The
  real-cell calibration writes under `calibration/real` keyed by `rig_id`, and the simulator runners
  write under `logs/calibration/<robot_model>/`, so the two cannot overwrite each other and one arm's
  simulated run cannot overwrite another's. Copy an artifact aside before moving a camera.
* **Everything else**: `--rehearse` still runs with no hardware attached, so you can always get back
  to a known-good software baseline without the robot.

---

## See also

- [`cell_bringup.md`](cell_bringup.md), the profile, the desk check, the simulator and the connect
  that come before this.
- [`docs/cli.md`](../cli.md), every command line above, by topic.
- [`your_own_gripper.md`](your_own_gripper.md), a gripper this repository never shipped, up to a
  planner that starts.
- [`real_cell` package](../../src/robot/execution/real_cell/README.md), what the runner does at each
  stage, and the calibration command in full.
- [UR driver](../../src/robot/drivers/ur/README.md), the SDK and the connect-time refusals.
- [`examples/`](../../examples/README.md), the same composition driven from Python: `real_robot/`
  in the order a cell comes up, where every file from the third on moves the arm, and `simulation/`
  for the rehearsal.
- [`src/calibration/`](../../src/calibration/README.md), the typed extrinsics and what each frame
  means.
