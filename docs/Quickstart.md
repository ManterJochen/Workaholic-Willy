# Quickstart

From a fresh clone to a pick you can watch, on a desk with no robot and no camera attached.

Two example scripts carry the whole of it. `scripts/examples/01_robot_setup.py` reports what your
configuration claims the cell is and runs the desk-side preflight over it.
`scripts/examples/03_pick.py` builds a cell on a dummy arm and drives one pick through the real
grasp stack. Neither needs hardware, and no example commands a motion unless you pass `--live`.

## What you are installing

A vendor-neutral grasping backend: a camera scene plus a text prompt go in, a detector and a
segmenter find the object, a geometric calculator synthesises and ranks 6-DoF grasps, a fail-closed
safety pipeline gates every motion, a driver executes it, and the attempt is written to a structured
log. It is a library with command-line entry points, plus one optional HTTP console in
[`api/`](../api/README.md).

The default pick is open-loop: perceive, rank, gate, move, log. The layers above that are built and
switched off, including the automatic decision gate, closed-loop refinement and verification,
multi-view fusion, the learned success model and the reinforcement-learning router. Every one of
them is a `robot.grasping.*` block shipping `enabled: false`. Nothing in this document turns one on
by accident, and the fusion section says plainly what turning one on costs.

No trained weights for the learned grasp calculator ship in this repository. A cell that wants one
trains it on its own data.

## 1. Install

Python 3.11.

```bash
python -m venv .venv
.venv\Scripts\activate               # Windows. Elsewhere: source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is the whole dependency set and pins the CUDA 12.8 PyTorch wheels. They install
and import on a machine with no GPU, so this is the file to use even without a card.
`requirements-cpu.txt` is the escape hatch for a host that cannot take those wheels; under it Isaac
Sim, cuRobo, the detectors and the local paraphrase model are all unavailable.

Perception model weights are fetched separately, into the shared Hugging Face cache:

```bash
python scripts/model_weights/fetch.py --list
```

Everything below runs from the repository root.

## 2. Ask the configuration what your cell is

```bash
python -m src.config                       # validate the YAML tree
python scripts/examples/01_robot_setup.py  # then read it back as a cell
```

Against the shipped tree, the second command reports a UR cell and three blocking preflight items:

```
  OK   load the config tree                   vendor ur, profile (default)
  OK   what the config says this cell is      ip 192.168.1.100; tool frame from undeclared; payload 0.0 kg
  OK   safety preflight over the config       10 check(s), 3 blocking
  [BLOCK] tool frame       gripper.tool_frame.source is 'undeclared'
  [BLOCK] payload          safety.payload has enforce: true but mass_kg: 0.0
  [BLOCK] camera -> base   no CAMERA->BASE resolver in config
```

That is the correct state of a freshly cloned tree, not a fault. Each of the three is a value only
your bench can supply, and each is left unset because a plausible wrong value would fail open: a
guessed tool frame drives the fingertips through the table and logs a success. The exit code is `1`
while anything blocks. The same check is `python -m src.robot.execution.real_cell --check`, and the
example calls it rather than reimplementing it.

The run also prints warnings that never block, and two items marked as deferred to the bench,
meaning nothing in software can decide them. [`real_cell`](../src/robot/execution/real_cell/README.md)
explains what each item looks like at the cell if you skip it.

## 3. Run one pick, with no robot

```bash
python scripts/examples/03_pick.py
```

This builds a cell on a dummy arm against a synthetic scene and runs the real pick path: config
preflight, `from_robot_config`, the safety attestation, one attempt, teardown.

```
  OK   load the config tree                   vendor ur, calculator geometric
  OK   config preflight                       0 blocking of 8 check(s)
  OK   build the cell, config-driven          arm DummyRobotArm, gripper NullGripper
  OK   what this arm will refuse              UNGATED, 0 guard(s)
  OK   one pick, connect to teardown          1/1 attempt(s) succeeded

      outcome    SUCCEEDED                    mode=auto
      cell       dummy (simulated), gripper present
      candidates 8, executed #0 at score 0.714
      layers     (none)
```

Read the last three lines before the first. `UNGATED` is the honest answer: `SafetyPreflight` is
constructed inside the vendor driver, so a dummy arm reaches no guard at all, and the example asks
the built arm what it will refuse rather than asserting that safety ran. `layers (none)` is read off
report fields that stay unset when a layer produced nothing, so a capability that was not exercised
cannot be reported as demonstrated. And a `NullGripper` accepts every command and holds nothing, so
`SUCCEEDED` here is evidence about the code path and not about a grasp.

The candidate count and the score come from your configuration and your synthetic scene. They will
differ from the numbers above as soon as you change a grasping key, which is the point of printing
them.

## 4. Point it at your own cell

The YAML tree lives in [`config/`](../config/) at the repository root and is layered by profile;
[`src/config/`](../src/config/README.md) is the loader that validates it. `WILLY_PROFILE` names the
chain, applied left to right:

| Profile | What it describes |
|---|---|
| `ur5e` | a real UR5e bench cell, with fixtures, a planning world and a self-collision model |
| `ur3e` | a real UR3e cell, re-anchored because a UR3e works a 500 mm sphere and a UR5e an 850 mm one |
| `sim` | the Isaac Sim cell: `vendor: sim`, three cameras, the gate thresholds |
| `eth2` | the fusion half of a cell with two fixed RGB-D cameras. Use it as `ur5e,eth2` |

```bash
WILLY_PROFILE=ur5e python scripts/examples/01_robot_setup.py
python -m src.config --profile ur5e,eth2 --print
```

Three sub-commands answer the questions a YAML tree usually cannot:

```bash
python -m src.config explain robot.grasping.fusion.enabled   # meaning, default, and who set it
python -m src.config where fusion                            # find keys by substring, unwritten ones too
python -m src.config decisions                               # only what this cell changes from default
```

`--live` on `01_robot_setup.py` connects to the controller, pushes the payload, verifies the tool
frame against it, and reads the pose back. That is the first command in this document that talks to
hardware.

## 5. Teach the camera where the robot is

A cell with no `CAMERA->BASE` transform builds, connects, and then refuses every motion as
`INVALID_TARGET`, which at the bench looks like a broken robot. Calibration is what removes that
third blocking item, and it is a bench procedure rather than a command you can rehearse away:
[calibration setup](calibration-setup.md) is the page to work from, and
`scripts/examples/02_calibration.py` is the guided walkthrough that runs the same routine.

## 6. Grasp presets

Three preset overlays ship under [`config/grasping_presets/`](../config/grasping_presets/). They are
not part of the validated tree: `apply_preset` in `src/robot/grasping/replay/presets.py` deep-merges
one onto an already loaded `robot:` block, so `python -m src.config` never sees it and a misspelt key
merges in silently. `validate_preset` re-validates the merged result and is what rejects it.

| Preset | Mode | What it arms |
|---|---|---|
| `easy` | `easy` | nothing. Recovery and uncertainty are both switched off |
| `dense_clutter` | `dense_clutter` | recovery bounded to `next_viewpoint`, plus uncertainty fusion |
| `verification_heavy` | `closed_loop` | pre-grasp refinement and post-grasp verification |

Each block below is the shipped overlay. Paste it under your `robot:` block, or call
`apply_preset`.

`easy`, the minimum-risk single-object pick:

```yaml preset=easy
grasping:
  default_mode: easy
  recovery:
    enabled: false
  uncertainty:
    enabled: false
```

`dense_clutter`, bin picking with a bounded second look:

```yaml preset=dense_clutter
grasping:
  default_mode: dense_clutter
  recovery:
    enabled: true
    allowed_actions:
      - next_viewpoint
  uncertainty:
    enabled: true
    fail_closed_threshold: 0.4
```

`verification_heavy`, the closed loop for items where a missed slip is unacceptable:

```yaml preset=verification_heavy
grasping:
  default_mode: closed_loop
  recovery:
    enabled: true
    allowed_actions:
      - next_viewpoint
```

Three things the files say about themselves and the table cannot. `easy` is excluded from
`recovery.apply_modes` anyway, so switching recovery off there restates a guarantee the mode already
carries. In `dense_clutter`, `fail_closed_threshold: 0.4` is the default of
`decision.auto_uncertainty_threshold`, so arming the layer does not by itself move the gate, and
nothing is refused at all unless `decision.enabled` is true. In `verification_heavy`, `closed_loop`
is not in the shipped `recovery.apply_modes` either, so the recovery flag stays inert until the cell
adds the mode to that list; and `closed_loop` needs both a refiner and a verification policy wired
into the service, or the pick is refused with a `MODE_NOT_AVAILABLE` outcome rather than degraded to
an open-loop attempt.

## 7. The soak gate

```bash
python -m src.robot.grasping.replay --soak-report
```

This runs a deterministic synthetic soak of at least 2000 attempts across three scenarios, evaluates
the latency budgets, runs the drift and out-of-distribution watchdogs, compares against the
committed baseline, and writes one artifact to `logs/u12/soak_report.json`. Exit `0` means every
locked key passed; a non-zero exit prints the violating keys.

Read what it proves narrowly. The outcome distribution it replays is authored, so a pass proves that
telemetry, KPIs, the failure taxonomy, the latency packs and the watchdogs are internally
consistent. It is not a measurement of grasp quality, and its own report says so.

The twelve gate keys are `min_attempts_met`, `untyped_outcomes_zero`, `unbounded_retry_loops_zero`,
`telemetry_offenders_zero`, `extra_type_offenders_zero`, `dead_loop_rate_within_gate`,
`pick_success_rate_non_regression`, `slo_packs_pass`, `drift_gate_pass`, `ood_gate_pass`,
`easy_attempt_wall_time_within_budget`, and `passes`, which is the conjunction of the other eleven.
Their thresholds are constants in `src/robot/grasping/replay/soak.py`: 2000 attempts minimum, a
recovery-action count above 8 counts as an unbounded retry loop, the dead-loop rate ceiling is
0.005, and the easy-mode wall-time budget is 1.05 times the baseline.

For a signal about real picks rather than about the contract, roll up a record log:

```bash
python -m src.robot.grasping.replay --records run.jsonl       # KPIs from a real record log
python -m src.robot.grasping.replay --records-gate run.jsonl  # the same thresholds, and this one can fail
python -m src.robot.grasping.replay --baseline-report         # regenerate docs/baselines/u_plus_baseline_v1.json
```

Record logging is opt-in. Set `robot.grasping.record_log_path` and `from_robot_config` wires it,
stamping the robot vendor and model into every record; leave it unset and nothing is written, which
the preflight reports as a warning.

## 8. Isaac Sim

The full perceive, grasp and execute path runs in Isaac Sim against a UR5e with a Robotiq 2F-85, on
a workstation with an NVIDIA GPU. It is a separate multi-gigabyte install with its own interpreter,
and it needs the cuRobo planner and the exact-mesh collision engine from `ext_deps/`, which the sim
cell refuses to boot without. [Make Isaac ready](isaac-ready.md) is the setup, and
[`src/willy_sim/`](../src/willy_sim/README.md) is the runner catalogue.

```bash
python scripts/examples/07_sim.py                                # checks this box, runs nothing
<isaac-sim>\python.bat -m src.willy_sim.run_m1_pick --runs 10    # known-pose pick
```

## 9. The checks that gate a change

```bash
ruff check src config api datagen tests scripts/examples
mypy src api datagen scripts/examples
pytest tests --cov=src --cov-fail-under=80
python -m src.robot.grasping.replay --soak-report
```

`scripts/examples/` is linted and type-checked with the library, so an example that stops matching
the API fails the same gate the library does.

## Where to go next

- [`docs/guide/`](guide/README.md), the sequential walkthrough: configuration, models, calibration,
  robot and safety, the pick loop. This page gets something running; the guide builds a cell.
- [`src/robot/grasping/`](../src/robot/grasping/README.md), the grasp stack, its modes and what each
  one requires.
- [`src/robot/README.md`](../src/robot/README.md), the driver contract, and
  [`src/robot/drivers/`](../src/robot/drivers/README.md) for UR, KUKA, Isaac and the dummy.
- [`src/config/`](../src/config/README.md), the loader and the schemas, and [`config/`](../config/), the YAML you edit.
- [`scripts/examples/`](../scripts/examples/README.md), the seven examples in dependency order.
- [`docs/runbooks/real_cell_first_pick.md`](runbooks/real_cell_first_pick.md), the ordered bring-up
  from a validated configuration to a commanded motion, and
  [`docs/runbooks/ur3e_cell_bringup.md`](runbooks/ur3e_cell_bringup.md), which includes URSim in
  Docker.
