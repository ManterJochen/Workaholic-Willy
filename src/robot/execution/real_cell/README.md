# real_cell

The command-line surface for a config-driven pick on a physical arm, the configuration preflight
that makes a bring-up survivable, and the per-camera hand-eye calibration multi-view needs.

```bash
python -m src.robot.execution.real_cell --check              # the checklist, touches nothing
python -m src.robot.execution.real_cell --rehearse --runs 3  # the whole path, no camera, no robot
python -m src.robot.execution.real_cell --dry-run            # real config, build only, no motion
python -m src.robot.execution.real_cell --runs 10 --profile ur3e
```

## What it guarantees

This package does not know how to build a cell. It is a thin shim over
[`Cell`](../cell.py) and [`PickRun`](../pick_run.py): the construction lives in
[`autonomous_grasp/cells.py`](../autonomous_grasp/README.md), so the operator console reaches the
same builders without importing a command-line runner. What lives here is the bench wording, the
banners, and the two things nothing else owns: `run_config_preflight` and `calibrate`.

`scripts/examples/03_pick.py` runs the same composition from Python and does not go through this
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

Exit codes: `0` the campaign passed, or `--check` and `--dry-run` were satisfied; `1` the preflight
blocked, the build refused, or the connect refused; `2` the cell connected and the campaign did not
pass its rule; `3` an exception escaped a pick. The codes past the first come from
`PickRunReport.exit_code`, so the library and the command line cannot disagree.

The verdict rule is unanimity: every pick must succeed. That is stricter than the simulator
gate, which configures `pass_fraction: 0.8` and additionally refuses to accept the service's own
`SUCCEEDED` as evidence. This runner has the strictest rule and no independent confirmation, and a
cell that built a `NullGripper` reports `SUCCEEDED` every time, so the rule is printed alongside the
verdict. From Python it is an argument:
`PickRun.from_cell(cell, runs=10, rule=PassRule(fraction=0.8, confirm=...))`.

## Why the preflight exists

Run `--check` against the shipped configuration tree as a UR cell and it reports three blocking
items. That is the default state of a freshly configured real cell, and every one of them would
otherwise be discovered separately, at the bench, as a different-looking failure.

| Blocking item | What it looks like at the bench if you skip this |
| --- | --- |
| `gripper.tool_frame.source` is `undeclared` | nobody has said where the grasp centre sits on the flange, so a top-down grasp commanding z = 37 mm drives the flange there and the fingertips through the bench |
| `safety.payload` has `enforce: true` and `mass_kg: 0.0` | `connect()` refuses this outright, which is better seen here than after driving to the cell. It would otherwise push a zero payload and leave the controller's protective-stop model under-reading a mounted tool |
| no `CAMERA->BASE` resolver | perception reports grasps in the camera frame; without a resolver the driver rejects every motion as `INVALID_TARGET`. Fail-closed and correct, and at the bench it looks exactly like a cell that hangs |

`config/robot/robot.ur5e.yaml` is the worked example for a real bench, and it leaves exactly those
three unset on purpose. Its own rule decides which keys carry a value: a wrong value that fails
closed ships as an example with its assumption stated, so a workspace box that is too small refuses
a motion visibly and the operator widens it. A wrong value that fails open does not ship at all,
because a plausible tool frame or payload drives the arm into the bench and logs a success. The
three appear in that file as commented blocks saying what to measure.

A blocking item does not necessarily stop you connecting. A cell with a blocking checklist can
connect and then refuse every motion, which is the failure that reads as a broken robot. What
refuses the connect is the driver's own preflight, not this checklist.

Four further items are reported as warnings and never block: an unset self-collision kinematics
model, no declared fixtures, no declared planning world, and no record log path. Two more are
reported as `[bench]`, because no interface answers them: the controller must be powered with
brakes released, in Remote Control, with no pendant program owning it, since `ur_rtde` uploads a
control script and the controller refuses it otherwise; and the end-effector's electrical side, the
Robotiq URCap that opens port 63352 or a vacuum solenoid's supply.

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

## Bring-up order

1. `python -m src.robot.drivers.doctor --require ur`, the SDK is installed.
2. `python -m src.robot.execution.real_cell --check`, fix everything blocking.
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

Run it once per rig. Each run holds exactly that camera through `FrameProvider.rig(rig_id)` and
gives it back, so it does not fight the console for devices it does not need. The artifact is keyed
by `rig_id`, which is also its key in `fusion.cameras`, and that alignment is what lets
`build_config_frame_resolvers` find the file.

Writing the artifact is half the job. Until the camera is listed in `grasping.fusion.cameras`,
geometry fusion stands down to a single view and says so only in telemetry. The runner prints the
exact YAML to paste.

| Flag | Why |
| --- | --- |
| `--mode` | `eye_to_hand` for a fixed camera, whose artifact is `CAMERA->BASE` and is what fusion consumes, or `eye_in_hand` for a wrist camera, whose artifact is `CAMERA->TOOL` and is composed with the live TCP each frame |
| `--marker-length-mm` | a wrong value scales every sample uniformly, so the solve converges and is uniformly wrong. Measure the printed board |
| `--poses` | 22 by default. The orientation spread is widened because a planar marker viewed near-frontally has a pose-estimation flip ambiguity that ruins the `AX=XB` rotation |

Exit codes: `0` done; `1` configuration refused; `2` it ran and produced no artifact, which is loud,
and the cell keeps its previous calibration; `3` unexpected.

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
- [camera](../../../camera/README.md), `FrameProvider` and the `rig(rig_id)` handle both runners take
- `scripts/examples/03_pick.py`, the same composition driven from Python
