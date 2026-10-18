# Quickstart

From a fresh clone to a pick you can watch, on a desk with no robot and no camera attached, and then
the first steps toward your own cell. You need Python 3.11; nothing else below needs hardware until
the section on your own cell says so.

A camera scene and a text prompt go in; a detector and a segmenter find the object, a geometric
calculator ranks 6-DoF grasps, a fail-closed safety pipeline gates every motion, a driver executes it,
and the attempt is written to a structured log. The default pick is open-loop: perceive, rank, gate,
move, log. Recovery runs on every cell, and fixed-camera fusion wherever two calibrated RGB-D rigs stand,
with no switch; the automatic decision gate, the learned success model and the reinforcement-learning router
(`robot.rl`) left on 2026-10-16 with their code. A **wrist camera** needs no
switch: it looks from its look poses, fuses what it saw and stops as soon as the grasp is safe. No trained
weights for the learned grasp calculator ship either: a cell that wants one trains it on its own data.

## Install

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e . --no-deps
```

`requirements.txt` is the whole dependency set and pins the CUDA 12.8 PyTorch wheels, which install
and import on a machine with no GPU, so it is the file to use even without a card.
`requirements-cpu.txt` installs the same set against CPU torch, for a host that cannot take those
wheels; Isaac Sim and the cuRobo planner need an NVIDIA GPU either way. The second install puts the
repository itself on the environment's path (`src`, `api`, `datagen` and `willy`) and installs no
dependency; the examples import `willy` through it.

Perception model weights are fetched separately, into `assets/models/hf` inside the repository:

```bash
python scripts/model_weights/fetch.py --list       # every key that can be fetched, with its size
python scripts/model_weights/fetch.py vlm-4b sam2   # the VLM and the segmenter a vision pick needs
```

Everything below runs from the repository root.

## Try it without hardware

### Run one pick on a dummy arm

```bash
python examples/simulation/01_rehearse_a_pick.py
```

This builds a cell on a dummy arm against a synthetic scene and runs the real pick path: the config
preflight, the build, the safety attestation, one attempt and the teardown. It prints four reports in
that order: the preflight, what the built arm will refuse, the campaign's verdict, and the attempt's
own account, which ends in its `layers` line.

Read the safety report and the `layers` line before the verdict. The safety report says `UNGATED`,
which is the honest answer: `SafetyPreflight` is built inside the vendor driver, so a dummy arm reaches
no guard at all, and the example asks the built arm what it will refuse rather than asserting that
safety ran. `layers (none)` is read off report fields that stay unset when a layer produced nothing, so
a capability that was not exercised cannot be reported as demonstrated. `SUCCEEDED` here is evidence
about the code path and not about a grasp: the dummy hand commands nothing and holds nothing.

The candidate count and the score on the `candidates` line come from your configuration and the
synthetic scene. They change as soon as you change a grasping key, which is the point of printing
them. The same campaign from a shell, with three picks:

```bash
python -m src.robot.execution.real_cell --rehearse --runs 3 --profile console_dummy
```

`console_dummy` is the desk profile, a dummy arm with a dummy hand. A rehearsal without it keeps the
hand your tree names, and a hand that lives on a real controller cannot be built on a dummy arm, so
that rehearsal is refused at the connect with exit 1.

### Ask the configuration what your cell is

```bash
python -m src.config                               # validate the YAML tree
python -m src.robot.execution.real_cell --check    # then read it back as a cell
```

Against the shipped tree, on a box with the `ext_deps` install, the second command reports a UR cell
and six blocking preflight items. Shown here are the blocking rows and the count; the report also
prints the fix under each row, the rows that pass, the warnings and the bench items:

```
=== 1. CONFIG === vendor=ur profile=<none>
  [BLOCK] tool frame                 gripper.tool_frame.source is 'undeclared'; nobody has said where the grasp centre sits on the flange
  [BLOCK] payload                    safety.payload has mass_kg: 0.0, the shipped 'not weighed yet'
  [BLOCK] camera -> base             camera.cameras.rigs['webcam_main'].extrinsics is not declared, so the primary camera has no CAMERA->BASE transform
  [BLOCK] camera world               this cell plans with cuRobo and its cameras give no live world: safety.planning_world declares no support_plane, so this cell has no live planner world. The pick service declines nothing, so every pick motion would be refused before it moves
  [BLOCK] planner margin             safety.self_collision.planner_margin_mm is undeclared, and a cuRobo cell refuses to start a planner without one: undeclared is not zero
  [BLOCK] hand                       robot.gripper.model is unset, and this cell's self collision guard reads hand geometry: it checks exact meshes (safety.self_collision.backend fcl) on the ur5e arm, and which hand those meshes are is the hand's name.

  6 blocking, 5 warnings, 2 deferred to the bench.
```

That is the correct state of a freshly cloned tree, not a fault. The tool frame, the payload and the camera to base
transform are values only your bench can supply, and each is left unset because a plausible wrong value would fail open:
a guessed tool frame drives the fingertips through the table and logs a success. The camera world follows from the base
tree planning with cuRobo: every cuRobo motion needs a live camera world or a decline, the pick service declines
nothing, and the row clears once the calibrated primary RGB-D camera feeds it over a measured
`safety.planning_world.support_plane`. The last two are declared by a cell for its own arm and hand: a planner starts
only on a combination, margin included, that a committed evidence file measured, and the base tree names no hand so that
no overlay inherits one. A box without the cuRobo environment or without Coal (the exact-mesh collision engine) blocks
on two more rows. The exit code is `1` while anything blocks.

The warnings never block, and the two items deferred to the bench are ones nothing in software can
decide. [`real_cell`](../src/robot/execution/real_cell/README.md) explains what each item looks like
at the cell if you skip it.

## Your own cell

### Choose or write a profile

The YAML tree lives in [`config/`](../config/) at the repository root and is layered by profile;
[`src/config/`](../src/config/README.md) is the loader that validates it. `WILLY_PROFILE` names the
chain, applied left to right, and most commands take `--profile` for a single run:

| Profile | What it describes |
|---|---|
| `ur5e` | a real UR5e bench cell, with fixtures, a planning world and a self-collision model |
| `ur3e` | a real UR3e cell, re-anchored because a UR3e works a 500 mm sphere and a UR5e an 850 mm one |
| `sim` | the Isaac Sim cell: `vendor: sim`, three cameras, the gate thresholds |
| `eth2` | the camera half of a cell with two fixed RGB-D cameras, fused. Use it as `ur5e,eth2` |
| `console_dummy` | the desk: a dummy arm and a dummy hand that command nothing |
| `hande` | the Robotiq Hand-E as a cell's hand, chained onto an arm: `ur5e,hande` or `ur3e,hande` |

```bash
python -m src.robot.execution.real_cell --check --profile ur5e
python -m src.config --profile ur5e,eth2 --print
```

Your own cell is a layer of its own: a `*.<your cell>.yaml` file beside each file it changes. The
profile step of [cell_bringup.md](runbooks/cell_bringup.md) writes one, and
[`examples/real_robot/01_load_your_cell.py`](../examples/real_robot/01_load_your_cell.py) reads it from
Python. Three sub-commands answer the questions a YAML tree usually cannot:

```bash
python -m src.config explain robot.grasping.default_mode     # meaning, default, and who set it
python -m src.config where container                         # find keys by substring, unwritten ones too
python -m src.config decisions                               # only what this cell changes from default
```

### Connect, then move

`python scripts/checks/cell_bringup.py --live` connects the arm alone, reads its pose back and holds it
against the workspace box the guard will use. It is the first command in this document that talks to
hardware, and it moves nothing. From there the files under
[`examples/real_robot/`](../examples/README.md) drive your cell in order, and from the third on they
move the arm. [docs/runbooks/real_cell_first_pick.md](runbooks/real_cell_first_pick.md) is the ordered
bring-up to a first pick on a physical arm.

### Teach the camera where the robot is

A cell with no `CAMERA->BASE` transform builds, connects, and then refuses every motion as
`INVALID_TARGET`, which at the bench looks like a broken robot. Calibration is what clears the
`camera -> base` row, and it is a bench procedure rather than a command you can rehearse away:
[calibration setup](calibration-setup.md) is the page to work from, and
`examples/real_robot/07_calibrate_a_fixed_camera.py` runs the same routine from Python.

## Grasp modes

`robot.grasping.default_mode` sets the recovery a mode allows: `easy` never recovers, `auto` rescans, and
`dense_clutter` rescans and pushes ([guide 05](guide/05-pick-loop.md), section 6). No mode chooses a sampler
since 2026-10-16: the part's own body over its support grasps every part alike. A tree or a call that
still names `closed_loop` is refused with the mode to name instead, `auto`; `dense_autonomous` names
`dense_clutter`. The grasp presets that overlaid a mode with its switches left on 2026-10-16.

The `dense_clutter` mode also allows the **push** (`nudge_target`), which slides a boxed-in part aside on
a wrist camera's pick; a fixed-camera cell never pushes. Every `dense_clutter` cell pushes unless its parts
are critical (`runtime.task.critical_parts`), and the push needs **no** fixture
([guide 05](guide/05-pick-loop.md), section 6.4).

## The gate over a record log

```bash
python -m src.robot.grasping.replay --records run.jsonl       # KPIs from a real record log
python -m src.robot.grasping.replay --records-gate run.jsonl  # the gate over it, and this one can fail
```

`--records` rolls the log up and never fails. `--records-gate` holds it to six keys and exits 1 on a
violation: at least 2000 attempts, no untyped outcome, no pick with more than 8 recovery actions, no record
missing the telemetry its outcome requires, no `extra` field of the wrong type, and a dead-loop rate of at
most 0.005. The numbers are constants in
[`src/robot/grasping/replay/soak.py`](../src/robot/grasping/replay/soak.py). A simulator's log is judged by
`--sim-soak-report`, which reports the pick rate rather than gating it. The synthetic soak that replayed an
authored outcome distribution, and passed by construction, left on 2026-10-16.

Every cell writes one record per attempt, stamped with the robot vendor and model: to
`robot.grasping.record_log_path`, or, where the tree leaves it unset, to `grasp_records.jsonl` in the log
folder (since 2026-10-17). From Python, `Recording.to_file(...)` names a file for one campaign
([`examples/real_robot/12_pick_with_the_camera.py`](../examples/real_robot/12_pick_with_the_camera.py)).

## Isaac Sim

The full perceive, grasp and execute path runs in Isaac Sim against a UR5e with a Robotiq 2F-85, on a
workstation with an NVIDIA GPU. It is a separate multi-gigabyte install with its own interpreter, and
it needs the cuRobo planner and the exact-mesh collision engine from `ext_deps/`, which the sim cell
refuses to boot without. [Make Isaac ready](isaac-ready.md) is the setup, and
[`src/willy_sim/`](../src/willy_sim/README.md) is the runner catalogue.

```bash
<isaac-sim>/python.bat -m src.willy_sim.run_m1_pick --runs 10       # known-pose pick
<isaac-sim>/python.bat examples/simulation/03_isaac_pick_rate.py    # the same gate, from Python
```

## The checks that gate a change

```bash
ruff check src api datagen tests scripts examples willy
mypy src api datagen scripts examples willy
pytest tests --cov=src --cov=api --cov=datagen --cov-fail-under=80
```

`scripts/` and `examples/` are linted and type-checked with the library, so an example or a
workstation tool that stops matching the API fails the same gate the library does. `config/` is not in
those lists: it is the YAML tree, and the Python that reads it lives in `src/config`.

## Where the details live

- [`docs/guide/`](guide/README.md), the sequential walkthrough: configuration, models, calibration,
  robot and safety, the pick loop, grippers. This page gets something running; the guide builds a cell.
- [`examples/`](../examples/README.md), short programs through `from willy import ...`, in three
  folders by what has to be attached, and [`willy/README.md`](../willy/README.md), every name they import.
- [`docs/cli.md`](cli.md), every command line by topic, with the runbook that uses it.
- [`docs/runbooks/cell_bringup.md`](runbooks/cell_bringup.md), which brings a cell up on any robot
  and includes URSim in Docker for a UR, and
  [`docs/runbooks/real_cell_first_pick.md`](runbooks/real_cell_first_pick.md), the ordered bring-up
  from a validated configuration to a commanded motion.
- [`src/robot/grasping/`](../src/robot/grasping/README.md), the grasp stack, its modes and what each
  one requires.
- [`src/robot/README.md`](../src/robot/README.md), the driver contract, and
  [`src/robot/drivers/`](../src/robot/drivers/README.md) for UR, Isaac and the dummy.
- [`src/config/`](../src/config/README.md), the loader and the schemas, and [`config/`](../config/),
  the YAML you edit.
