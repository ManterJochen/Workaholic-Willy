# real_cell

The command-line surface for a config-driven pick on a physical arm, the configuration preflight
that makes a bring-up survivable, and the per-camera hand-eye calibration multi-view needs.

```bash
python -m src.robot.execution.real_cell --check              # the checklist, touches nothing
python -m src.robot.execution.real_cell --rehearse --runs 3 --profile console_dummy  # whole path, no hardware
python -m src.robot.execution.real_cell --dry-run            # real config, build only, no motion
python -m src.robot.execution.real_cell --start-planner --profile ur3e  # does the planner start? no controller
python -m src.robot.execution.real_cell --runs 10 --profile ur3e
```

## What it guarantees

This package does not know how to build a cell. It is a thin shim over
[`Cell`](../cell.py) and [`PickRun`](../pick_run.py): the construction lives in
[`autonomous_grasp/cells.py`](../autonomous_grasp/README.md), so the operator console reaches the
same builders without importing a command-line runner. What lives here is the bench wording, the
banners, and the two things nothing else owns: `run_config_preflight` and `calibrate`.

`scripts/examples/api/01_first_cell/one_pick_end_to_end.py` runs the same composition from Python and does not go through this
runner, which is the check that the shim carries no logic of its own.

## The stages, and what each proves

| Stage | What runs | What it proves |
| --- | --- | --- |
| Config | load and validate with profile layers, then `run_config_preflight` | every stop-the-cell condition decidable at a desk, each with its fix |
| Build | `Cell.build()`, which calls `from_robot_config` | the arm, the gripper branch, safety, the frame resolver and record provenance, refusing rather than guessing |
| Attest | `Cell.safety()` | what this arm will actually refuse, read off the built arm, printed before anything moves and before `--dry-run` exits |
| Connect | `Cell.connected()` | the cross-process lock, then the arm, then the gripper. The tool frame and the payload are verified against the controller and both fail closed |
| Pick | `PickRun` | one typed outcome and reason per attempt, judged by a stated rule |
| Down | the session's teardown | gripper first, then arm, then cameras, reported and never swallowed |

Exit codes: `0` the campaign passed, `--check` and `--dry-run` were satisfied, or `--start-planner`
started; `1` the preflight blocked, the build refused, the connect refused, or the planner start was
refused; `2` the cell connected and the campaign did not pass its rule; `3` an exception escaped a
pick. The codes past the first come from `PickRunReport.exit_code`, so the library and the command
line cannot disagree.

The verdict rule is unanimity: every pick must succeed. That is stricter than the simulator
gate, which configures `pass_fraction: 0.8` and additionally refuses to accept the service's own
`SUCCEEDED` as evidence. This runner has the strictest rule and no independent confirmation, and a
cell that built a `NullGripper` reports `SUCCEEDED` every time, so the rule is printed alongside the
verdict. Since 2026-09-09 a substituted one cannot get that far, because the connect refuses it; a
cell that declares `gripper.vendor: none` still connects, and still reports `SUCCEEDED` on every
run. From Python it is an argument:
`PickRun.from_cell(cell, runs=10, rule=PassRule(fraction=0.8, confirm=...))`.

## Why the preflight exists

Run `--check` against the shipped configuration tree as a UR cell and it reports seven blocking
items. That is the default state of a freshly configured real cell, and every one of them would
otherwise be discovered separately, at the bench, as a different-looking failure.

| Blocking item | What it looks like at the bench if you skip this |
| --- | --- |
| `gripper.tool_frame.source` is `undeclared` | nobody has said where the grasp centre sits on the flange, so a top-down grasp commanding z = 37 mm drives the flange there and the fingertips through the bench |
| `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses this outright, which is better seen here than after driving to the cell. It would otherwise push a zero payload and leave the controller's protective-stop model under-reading a mounted tool |
| no `CAMERA->BASE` resolver | perception reports grasps in the camera frame; without a resolver the driver rejects every motion as `INVALID_TARGET`. Fail-closed and correct, and at the bench it looks exactly like a cell that hangs |
| no live camera world | the base tree plans with cuRobo, where a motion needs a live camera world or a decline, and the pick service declines nothing. Every pick motion is refused as `UNSUPPORTED` before it moves, naming the missing world |
| `safety.self_collision.planner_margin_mm` is undeclared | the base tree plans with cuRobo, and a planner never starts without the clearance it keeps, since undeclared is not zero. The first planned move is refused as `CONTROLLER_REJECTED`, before the arm moves |
| `safety.planning_world.payload.length_mm` is undeclared | the base tree plans with cuRobo, which models the part a grasp carries, and no length is implied for it. The attach is declined and every lift and transit after a grasp is planned as if the hand were empty. Declare how far the longest part hangs past the fingertips, or `enabled: false` for a cell that carries nothing |
| `robot.gripper.model` is unset | the exact-mesh guard checks the hand this key names, and the base tree names none so that no overlay inherits one. The cell refuses to build, naming the key |

A box without the cuRobo environment or without Coal blocks on two more rows, `cuRobo environment`
and `exact mesh engine`: they are facts about the machine the checklist runs on, and they clear
when the `ext_deps` install is on the box.

`config/robot/robot.ur5e.yaml` is the worked example for a real bench, and it leaves the first
three of those unset on purpose. Its own rule decides which keys carry a value: a wrong value that
fails closed ships as an example with its assumption stated, so a workspace box that is too small
refuses a motion visibly and the operator widens it. A wrong value that fails open does not ship at
all, because a plausible tool frame or payload drives the arm into the bench and logs a success. The
three appear in that file as commented blocks saying what to measure.

A blocking item does not necessarily stop you connecting. A cell with a blocking checklist can
connect and then refuse every motion, which is the failure that reads as a broken robot. What
refuses the connect is the driver's own preflight, not this checklist.

Once the tool frame is declared, a `grasp centre` row holds it to the hand: the offset along the
declared approach should be the registry's `grasp_centre_mm` plus the coupling plates, and zero
across it. It warns beyond 1 mm and does not block yet: the 2F-85's registry number, 146.5 mm, is
an estimate that every shipped 2F-85 profile's 132 mm disagrees with, and it is measured before the
row may stop a cell.

A `gripper driver` row states the driver the build constructs for `robot.gripper.vendor` on the UR
arm the config builds, from the same `gripper_driver_verdict` the build reads. It blocks exactly
where the build would put a `NullGripper` on the flange, and warns for `none` and `dummy` on a real
arm.

A `wrist camera body` row runs the resolution the build, `Robot` and the planner start run
(`execution/wrist_bodies.py`). It is OK naming each camera the arm carries, its grown boxes, sphere
count and the calibration that placed it, with its cover proven. On a real cell it blocks for an
enabled eye_in_hand rig without a body, a body nothing places, a calibration without its flange to
TCP record or with a stale one, a camera the repository's registry does not stand for, and a
checklist handed no camera section.

A `camera world` row, on a UR cell, says whether its cameras give the planner a live world, from
the same `CameraWorldPlan` the build reads. It is OK naming the cameras and warns on `ik`, where no
planner reads one. It blocks on `curobo` without one, because every planned motion with neither a
world nor a decline is refused and the pick service declines nothing, and it blocks for a checklist
handed no camera section. A cuRobo cell whose calibrated rig gives no world does not build either
(`CellBuildRefused`, before a camera opens). The row opens no device.

Four further items are reported as warnings and never block: an unset self-collision kinematics
model, no declared fixtures, no declared planning world boxes, and no record log path. Two more are
reported as `[bench]`, because no interface answers them: the controller must be powered with
brakes released, in Remote Control, with no pendant program owning it, since `ur_rtde` uploads a
control script and the controller refuses it otherwise; and the end-effector's electrical side for
the configured driver: the Robotiq URCap that opens port 63352, the vacuum ejector's output pins and
24 V supply, the jaw solenoid's close and open pins, or the OnRobot Compute Box's address.

## The rehearsal

`--rehearse` changes the vendor on the operator's own configuration to `dummy` and swaps in a
synthetic scene: one box on a plane, sized to fit the configured gripper. The profile chain, the
gripper branch and the grasping block stay the ones the operator runs.

It is not a simulator and not grasp-quality evidence. The dummy arm carries no safety preflight and
`Cell.safety()` says so, so a rehearsal proves the wiring and nothing about what would refuse a bad
command. It exists so the same wiring can be driven end to end at a desk, in milliseconds.

A rehearsal continues past a blocking checklist on purpose: a desk is where blocking items are the
expected state. That is a policy the runner applies on top of the preflight's verdict, which is why
`--rehearse --check` and `--check` can exit differently on the same tree.

A rehearsal can manufacture the one cell that is now refused, and on the shipped tree it does. The
swap is `robot.vendor` to `dummy` and nothing else, so `gripper.vendor: robotiq` is still asked for
and can no longer be built, because a Robotiq lives on the UR controller's tool I/O. The build
substitutes a working `NullGripper`, and until 2026-09-09 that cell connected and reported
`3/3 succeeded` while closing on nothing. `connect_cell` refuses it now (`NoRealGripper`, before the
arm is commanded), which is the same answer the operator console has always given
(`403 no_real_gripper`) and now the same sentence. Measured 2026-09-09: 3/3 with
`--profile console_dummy`, whose `gripper.vendor: dummy` a dummy arm can carry, and exit 1 without
it. What that no longer exercises is your own gripper branch, and nothing at a desk can.

## Bring-up order

1. `python -m src.robot.drivers.doctor --require ur`, the SDK is installed.
2. `python -m src.robot.execution.real_cell --check`, fix everything blocking.
   On a `motion_planner: curobo` cell, `--start-planner` then builds the arm alone and starts its
   planner the way the first planned move does, with every refusal that move meets (margin, hand,
   retract row, descriptor, evidence), and stops it, exiting 0 when it started. It opens no camera
   and asks no controller, so it runs at a desk. The library twin is `Cell.start_planner()` and
   `execution.planner_start.PlannerStart`.
3. `python -m src.robot.safety.planning --doctor`, the planner environment, before any motion.
4. `python -m src.robot.perception --prompt "..."`, the camera and the models, no robot.
5. Calibrate each camera, one command per rig, then `--dry-run`, then `--runs 1`, then a campaign.

## calibrate: one camera, against the robot

```bash
python -m src.robot.execution.real_cell.calibrate --rig overhead --check     # touches nothing
python -m src.robot.execution.real_cell.calibrate --rig overhead --dry-run   # opens the camera
python -m src.robot.execution.real_cell.calibrate --rig overhead --poses 22  # this moves the robot
```

This is what unlocks multi-view. `grasping.fusion.geometry` states its own precondition, that
`cameras` must be populated with each camera individually calibrated, and every consuming piece is
finished: the per-camera map, `build_config_frame_resolvers` turning it into resolvers, the pick
loop reading that map, the versioned extrinsics artifact keyed by `rig_id`, and `CalibrationRoutine`
itself. The preflight can only check for an artifact and refuse without one. This command makes one.

Run it once per rig. Each run holds exactly that camera through its `Camera` owner and gives it
back, so it does not fight the console for devices it does not need, and a rig the console holds is
refused naming the holder. `--check` refuses a rig
that is `enabled: false` and names the key: measured 2026-09-10 it called such a rig usable, and the
next command in the sequence moves the arm to 22 poses in front of a camera the cell will not open.
The artifact is keyed by `rig_id`, the id of the rig that declares it
(`camera.cameras.rigs[<rig_id>].extrinsics`) and of its entry in `fusion.cameras` when the camera is
fused, so the file, the rig and the fusion map name one camera.

It builds the arm through `Robot.from_config(robot_config, gripper=None)`. The arm-vendor readiness
gate runs first, and no gripper is built or connected: a Robotiq does not run its activation stroke
beside the board, and a gripper this tree cannot build does not block a calibration. `--dry-run`
prints the arm, the cell lock, the safety attestation and the camera world line, then stops without
taking the lock. The
sweep connects through `Robot.connected()`, which runs the enter and the exit this runner's pick runs
use, so it takes the same cell lock as this runner and the operator console: while either holds the
controller the sweep is refused and names the holder. On the way out the arm comes down and the lock
is given back before the camera is. Every move declines the camera world with a reason naming the
mounting, because the sweep produces the transform a camera world needs.

An eye in hand sweep on a UR arm writes `eih_<rig_id>.json` as `willy.calibration.cam_to_tool/2`,
with the flange to TCP the arm applied while it swept. On a cell that reads geometry the camera's
body is part of the sweep: a rig without a body is refused, a body its previous calibration places
is handed to the arm before it moves, and a body that cannot be placed yet (no calibration, no
record, a stale one) sweeps only with `--unmodelled-wrist-body "<reason>"`, printed and logged,
with no body in the planner and the guard.

Writing the artifact is half the job. Until its rig declares it, in
`camera.cameras.rigs[<rig_id>].extrinsics`, the cell has no CAMERA->BASE for that camera: a real
cell whose primary rig declares none is refused at build, and a second camera also needs its entry
in `grasping.fusion.cameras` before geometry fusion uses it. The runner prints the exact rig block to
paste, with a wrist camera's two shutter motion tolerances as comments to measure and fill in; the
loader refuses a wrist block until they are written.

A cell that enables `safety.planning_world` builds that world from every enabled RGB-D rig that
declares its calibration, the primary first, and opens the ones the pick does not already hold. One
of them that cannot answer stops every planned motion, so calibrating a second camera on such a cell
is also a decision about what stops it. On a cuRobo cell a declared calibration makes that world
mandatory: a cell whose calibrated rigs give no world is refused at build.

| Flag | Why |
| --- | --- |
| `--mode` | `eye_to_hand` for a fixed camera, whose artifact is `CAMERA->BASE` and is what fusion consumes, or `eye_in_hand` for a wrist camera, whose artifact is `CAMERA->TOOL` and is composed with the live TCP each frame |
| `--marker-length-mm` | a wrong value scales every sample uniformly, so the solve converges and is uniformly wrong. Measure the printed board |
| `--poses` | 22 by default. The orientation spread is widened because a planar marker viewed near-frontally has a pose-estimation flip ambiguity that ruins the `AX=XB` rotation |

Exit codes: `0` done; `1` configuration or build refused, the cell held by another process, or the
connect refused; `2` it ran and produced no artifact, which is loud, and the cell keeps its previous
calibration; `3` the sweep raised.

## Status

Everything past the rehearsal has never executed against a physical controller. The rehearsal is
the same wiring with a dummy arm and a synthetic scene. The connect, telemetry and refusal path has
additionally been driven against real controller software in a simulator container through the
operator console, which is still not a physical arm. The calibration routine and its solve are
exercised in simulation only, and the marker source has never seen a physical RGB-D camera.

## See also

- [execution](../README.md), the composition root and the calibration routine
- [autonomous_grasp](../autonomous_grasp/README.md), where `build_real_cell` lives
- [drivers/ur](../../drivers/ur/README.md), the UR driver and its bring-up checklist
- [robot/perception](../../perception/README.md), the live-camera source this runner consumes
- [camera](../../../camera/README.md), `Camera`, the one owner per rig, and the handle both runners take from it
- `scripts/examples/api/01_first_cell/one_pick_end_to_end.py`, the same composition driven from Python
