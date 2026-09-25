# The real cell from a shell (`src/robot/execution/real_cell`)

Two commands for a cell on a bench: `real_cell` takes it from its config to a campaign of picks,
one stage and one banner at a time, and `real_cell.calibrate` calibrates one camera against the arm.
Both are thin callers of the library (`Cell`, `PickRun`, `HandEyeCalibration`), so the reports they
print are the library's and the exit codes cannot disagree with it.

```bash
python -m src.robot.execution.real_cell --check                  # the desk checklist; touches nothing
python -m src.robot.execution.real_cell --start-planner          # does the planner start? no controller
python -m src.robot.execution.real_cell --rehearse --runs 3 --profile console_dummy   # a dummy arm
python -m src.robot.execution.real_cell --dry-run                # build the cell, stop before the connect
python -m src.robot.execution.real_cell --runs 3 --prompt "a red cube"   # this moves the robot
```

Exit codes: `0` the campaign passed, or `--check`, `--dry-run` or `--start-planner` was satisfied;
`1` the checklist blocked, or the build, the connect or the planner start was refused; `2` the cell
connected and the campaign did not pass its rule; `3` a fault of the cell stopped a pick. The flags
`--profile` (else `WILLY_PROFILE`) and `--data-dir` choose the tree.

The same steps from Python are
[02_check_the_cell_at_a_desk.py](../../../../examples/real_robot/02_check_the_cell_at_a_desk.py) and
[12_pick_with_the_camera.py](../../../../examples/real_robot/12_pick_with_the_camera.py):

```python
from willy import Cell, load_tree

cell = Cell.from_tree(load_tree())
checklist = cell.preflight()     # what --check prints
print(checklist)
print(cell.start_planner())      # what --start-planner prints
raise SystemExit(checklist.exit_code)
```

## The stages

| Stage | What runs | What it proves |
| --- | --- | --- |
| Config | the tree, then `Cell.preflight()` | every stop-the-cell condition a desk can decide, each with its fix |
| Build | `Cell.build()` | the arm, the hand, the cameras, the models and the frame resolver, refused rather than guessed |
| Attest | `Cell.safety()` | what the built arm refuses, printed before any motion and before `--dry-run` stops |
| Connect | `Cell.connected()` | the cell lock, the arm (tool frame and payload checked against the controller), then the hand |
| Pick | `PickRun` | one typed outcome per attempt, judged by a stated rule |
| Down | the teardown | the hand, then the arm, then the cameras, each reported |

> [!WARNING]
> Connecting is motion. A Robotiq activation sweeps the full finger travel and a vacuum cup's
> connect drives its output at once, so keep hands clear from the connect on.

The command's rule is unanimity, and it takes the service's word for each success. It prints the
rule beside the verdict, because a cell that declares `gripper.vendor: none` connects and reports
`SUCCEEDED` on every run. A cell whose hand had to be replaced by a `NullGripper` is refused at the
connect (`NoRealGripper`), as the operator console refuses it. From Python the rule is an argument:
`PickRun.from_cell(cell, runs=10, recording=Recording.off(), rule=PassRule(fraction=0.8))`.

## The desk checklist

`--check` marks each row `ok`, `BLOCK`, `warn` or `bench`. A `BLOCK` row stops a real run before the
build; a `warn` row never blocks; a `bench` row is a question no interface answers. On the shipped
base tree, and on the worked `ur5e` profile, these rows block, and each would otherwise show up at the
bench as a different-looking failure:

| Row | What it looks like at the bench if you skip it |
| --- | --- |
| `tool frame`: `gripper.tool_frame.source` is `undeclared` | a top-down grasp drives the flange to the grasp pose and the fingertips into the bench |
| `payload`: `enforce: true` with `mass_kg: 0.0` | `connect()` refuses it; better seen here than at the cell |
| `carried part`: `safety.planning_world.payload.length_mm` undeclared | every lift after a grasp is planned as if the hand were empty |
| `camera -> base`: the primary rig declares no `extrinsics` | the driver refuses every motion as `INVALID_TARGET`, which looks like a hung cell |
| `camera world`: cuRobo with no live world | every pick motion is refused as `UNSUPPORTED` before it moves |
| `planner margin`: `safety.self_collision.planner_margin_mm` undeclared | the first planned move is refused as `CONTROLLER_REJECTED` |
| `hand`: `robot.gripper.model` unset | the cell refuses to build, naming the key |

`config/robot/robot.ur5e.yaml` leaves the tool frame, the payload and the camera calibration unset on
purpose: a plausible wrong value there fails open and drives the arm into the bench, so the file
carries them as commented blocks saying what to measure. A box without the cuRobo environment or
without Coal also blocks on `cuRobo environment` and `exact mesh engine`, facts about the machine that
clear once [ext_deps](../../../../ext_deps/README.md) is installed on it.

The other rows:

- `grasp centre` warns when the declared tool frame puts the grasp centre more than 1 mm from the
  hand's registry `grasp_centre_mm` plus its coupling plates. It does not block: the 2F-85's registry
  number, 146.5 mm, is an estimate that the 132 mm of every shipped 2F-85 profile disagrees with.
- `gripper driver` blocks exactly where the build would put a `NullGripper` on the flange, and warns
  for `none` and `dummy` on a real arm.
- `wrist camera body` blocks on a real cell for an enabled eye in hand rig without a body, a body
  nothing places, a calibration without its flange to TCP record or with a stale one, and a camera
  the registry does not hold.
- `camera world` warns on the `ik` planner, where no planner reads a world.
- `self-collision`, `fixtures`, `planning world` and `record log` warn and never block.
- `controller state` (bench): powered, brakes released, Remote Control, no pendant program. In local
  control the controller refuses the control script; in remote with a pendant program running, the
  upload stops that program and takes the robot.
- `end-effector wiring` (bench): the Robotiq URCap on port 63352, the vacuum ejector's pins and 24 V,
  the jaw solenoid's pins, or the OnRobot Compute Box's address.

The checklist does not refuse a connect; the driver's own checks do. Code that connects past a
blocking checklist can bring up a cell that then refuses every motion, which reads as a broken robot.

## The rehearsal

`--rehearse` sets `robot.vendor` to `dummy` on your own tree and puts a synthetic scene (one box on
a plane, sized to the configured hand) in place of the cameras. It proves the wiring, not a grasp,
and the dummy arm gates nothing and says so. It continues past a blocking checklist on purpose, so
`--rehearse --check` exits 0 on the shipped tree where `--check` exits 1.

The swap keeps your hand, and a hand that lives on the real controller (a Robotiq, a vacuum cup or a
jaw on its digital I/O) cannot be built on a dummy arm: that rehearsal is refused at the connect and
exits 1. With `--profile console_dummy`, whose hand a dummy arm can carry, it measured 3 of 3 and
exits 0.

## calibrate: one camera against the arm

```bash
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive --check    # touches nothing
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive --dry-run  # opens the camera
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --mode eye_to_hand --freedrive            # you move the arm
python -m src.robot.execution.real_cell.calibrate --rig <rig id> --fixed-poses stations.json [--adjust]    # the arm moves
```

Nothing generates a station: a run names its stations (`--fixed-poses`) or is guided by hand
(`--freedrive`), and a run that names neither is refused at `--check`. Exit codes: `0` done; `1` the
config or the build refused, another process holds the cell, or the connect refused; `2` it ran and
wrote no artifact, and the camera keeps its previous calibration; `3` the sweep raised. The Python twins
are [07](../../../../examples/real_robot/07_calibrate_a_fixed_camera.py) to
[10](../../../../examples/real_robot/10_calibrate_a_wrist_camera_with_fixed_poses.py):
`HandEyeCalibration.from_tree(tree, rig_id=, mode=, options=SweepOptions(...))`, then `check()` and
`run(dry_run=)`.

| Flag | What it sets |
| --- | --- |
| `--mode` | `eye_to_hand`, a fixed camera (`eth_<rig>.json`, CAMERA to BASE); `eye_in_hand`, a wrist camera (`eih_<rig>.json`) |
| `--freedrive` | you move the arm by hand to every pose; Enter (console, or Enter or Space in the preview) captures once the arm stands still, `s` skips, `q` finishes; nothing moves by itself |
| `--samples` | with `--freedrive`: how many counted poses end the run (`robot.calibration.freedrive_samples`, 15) |
| `--fixed-poses PATH` | your stations, in the file's order: poses and joint stations (`{"joints_deg": [...]}`); beside `--freedrive` they are targets the preview shows the way to, never moved to |
| `--adjust` | with `--fixed-poses`: frees the arm at each station it reached so you fine-tune it by hand; before the next automatic move, hands off, Enter, and a 3 s countdown |
| `--marker-length-mm` | the printed edge (the mode's `camera.hand_eye` block); a wrong one scales every sample and still converges |
| `--dict`, `--marker-id` | the ArUco dictionary (the mode's `camera.hand_eye` block) and the marker id (0) |
| `--out` | where the artifact, the dataset, the counted frames and a hand-guided run's stations file go (`calibration/real`) |
| `--unmodelled-wrist-body "<reason>"` | moves the arm without a wrist camera's declared body while it cannot be placed yet (not calibrated, its artifact missing, or its flange to TCP record missing or stale): the camera an `eye_in_hand` sweep calibrates, and in either mode any other wrist camera the tree declares on the arm; that camera then has no body in the planner and the guard, and the build says so. A refusal gives this flag first, labelled `For THIS sweep`. Not needed for a wrist rig declared with no `body`, the camera an `eye_in_hand` sweep calibrates included, which refuses no sweep |

`--freedrive` and `--adjust` need an arm that offers hand guiding (on a UR, teach mode); the build
refuses them on any other. Before the arm is first freed, the controller payload is shown and has to
be confirmed. A pose outside the cable window or the workspace box turns the preview red and is not
captured; the arm is never held or stopped for it. Each pose counted by hand is written to
`<out>/<mode>_<rig>_stations.json`, which `--fixed-poses` replays without hands; a replay of that
very file with `--adjust` writes `<mode>_<rig>_stations.adjusted.json` beside it instead of
overwriting the stations you taught.

The sweep builds the arm alone (no hand, so no activation stroke beside the board), takes the same
cell lock as `real_cell` and the console, and declines the camera world on every move, because it
produces the transform that world is built from. Whichever camera it calibrates, the arm carries every
wrist camera body the tree declares, placed as `Robot.from_tree` places it, and `--check` and the build
name them on a `wrist cameras` line; a declared body that cannot be placed yet refuses the sweep at
`--check` unless `--unmodelled-wrist-body` says why, and an `eye_to_hand` sweep on a tree that declares
no wrist camera sweeps as before. A wrist rig declared with no `body` at all (the bracket not
measured yet), switched on or off, refuses no sweep and needs no reason, the `eye_in_hand` sweep of
that camera itself included: nothing is carried for it, and `--check` and the build print `!! wrist camera '<id>' is declared on the arm
without a body: the planner and the guard do not know it is there during this sweep`, which is
logged. That is for calibration only: `real_cell` and `Robot.from_tree` still refuse an enabled one.
The rig being calibrated is another matter when switched off: it is refused at `--check`, naming the
key.

An eye in hand sweep records the flange to TCP the arm applied while it swept, when
`robot.gripper.tool_frame.source` is `willy` or `polyscope`; on an `undeclared` frame the cell and the
`Locator` refuse the artifact for a wrist camera.

Writing the artifact is half the job. The command prints the rig block to paste under
`camera.cameras.rigs[<rig id>].extrinsics`; until it is pasted the cell has no CAMERA to BASE for that
camera, and a second camera also needs its entry in `grasping.fusion.cameras` before fusion uses it.

## Status

| Capability | Evidence |
| --- | --- |
| Connect, telemetry and a refused motion, through the operator console | measured against real controller software |
| The calibration routine and its solve | measured in simulation |
| The ArUco marker source on a physical camera | never touched hardware |
| Everything past the rehearsal on a physical controller | never touched hardware |

## Files

| File | Holds |
| --- | --- |
| `__main__.py` | the pick command: the stage banners over `Cell` and `PickRun` |
| `preflight.py` | `run_config_preflight`, `PreflightReport`, `PreflightCheck`, `CheckStatus` |
| `calibrate.py` | the calibration command over `HandEyeCalibration` |

## Details

- Runbooks: [the first pick on a physical arm](../../../../docs/runbooks/real_cell_first_pick.md), [bringing up a cell](../../../../docs/runbooks/cell_bringup.md)
- Measuring the extrinsics: [calibration-setup.md](../../../../docs/calibration-setup.md); every command: [docs/cli.md](../../../../docs/cli.md)
- The library it calls: [execution](../README.md), [autonomous_grasp](../autonomous_grasp/README.md); the UR driver: [drivers/ur](../../drivers/ur/README.md)
- Tests: `tests/test_real_cell_runner.py`, `tests/test_real_cell_calibrate.py`, `tests/test_calibrate_cli_transcript.py`, `tests/test_a_calibration_can_be_guided_by_hand.py`
