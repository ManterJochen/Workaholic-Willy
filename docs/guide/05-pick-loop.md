# 5. Assembling and running the pick loop

You have a config tree that validates ([01](01-configuration.md)), perception models that load
([02](02-models.md)), a camera-to-base transform ([03](03-calibration.md)) and an arm whose safety pipeline
is anchored ([04](04-robot-and-safety.md)). This guide puts those pieces into one object and makes
something leave the table.

```python
from willy import Cell, PickRun, Recording, load_tree

cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
run = PickRun.from_cell(cell, runs=1, recording=Recording.off()).execute()
print(run)                                                 # what happened, and the rule it passed
```

That runs on a laptop with no GPU, camera or robot, and it is
[`examples/simulation/01_rehearse_a_pick.py`](../../examples/simulation/01_rehearse_a_pick.py). At a real
cell the same campaign starts from `Cell.from_tree(load_tree())`
([`examples/real_robot/10_pick_campaign.py`](../../examples/real_robot/10_pick_campaign.py)).

This page traces the default pick stage by stage, says what each stage may refuse and where the telemetry
record comes from, and lists which advanced layers exist, which are off, and what turns each one on. The
default pick is open-loop. The most expensive mistake here is believing a feature is active because a YAML
key exists for it, so read section 7 before you report a number.

## Prerequisites

| # | Requirement | How to check | Guide |
| --- | --- | --- | --- |
| 1 | The config tree validates | `python -m src.config` (exit 0) | [01](01-configuration.md) |
| 2 | You know which keys exist | `python -m src.config where grasping --limit 500`, or [`config/all_keys/`](../../config/all_keys/) | [01](01-configuration.md) |
| 3 | A `PerceptionSource`: anything with `.acquire() -> PerceptionFrame` | see below | [02](02-models.md) |
| 4 | Camera intrinsics, and a `FrameResolver` for BASE-frame grasps | a calibrated intrinsics file, or `get_intrinsics()` on an RGB-D handle | [03](03-calibration.md) |
| 5 | An arm whose vendor SDK is present | `python -m src.robot.drivers.doctor` | [04](04-robot-and-safety.md) |
| 6 | For the simulator: both motion engines anchored | `python -m src.robot.safety.planning --check` | [04](04-robot-and-safety.md) |

On prerequisite 3: one real-camera adapter ships, `RealSenseVisionPerceptionSource`, which
`build_real_components` in `src/robot/execution/autonomous_grasp/cells.py` constructs for a physical cell.
Everything else that implements `acquire()` is simulated or synthetic. For another camera the adapter is
yours to write, and the contract is small: a depth map in millimetres, a 3x3 intrinsics matrix, and
segmentations that each carry a boolean `.mask`.

Shell blocks are Windows PowerShell from the repository root, with POSIX differences noted where they
matter.

## 1. The two constructors

`AutonomousGraspService` has two, and both are live.

`from_robot_config(robot_cfg, calculator=..., perception=...)` is the config-driven path. It reads
`robot.grasping`, builds the arm and gripper through the vendor registry, resolves the grasp mode and the
attempt budget, builds a sub-policy for each enabled block, computes an `EffectiveGraspingConfig`, applies
the orchestrator overlays, and turns on record logging when the config names a path. It accepts `arm=` and
`gripper=` for what config cannot express: a live device handle, such as a simulator gripper that shares
its arm's session.

`from_components(arm=..., calculator=..., perception=...)` takes objects you already built and reads no
config at all: `effective_config` stays `None` and no overlay runs.

| Caller | Constructor | Why |
| --- | --- | --- |
| `build_real_cell` and `build_rehearsal_cell` in `src/robot/execution/autonomous_grasp/cells.py` | `from_robot_config` | a whole cell in one call: physical, or a desk rehearsal |
| `Cell`, `python -m src.robot.execution.real_cell` and the operator console | `from_robot_config`, through those two builders | the program, the terminal and the browser drive one construction |
| `run_multiview_pick.py` with `--boot config`, its default | `from_robot_config` | the simulator booting the way a real cell boots |
| `datagen/rl/occupancy.py` | `from_robot_config` | features from the real stack, not from a copy of it |
| `run_m1_pick`, `run_m2_pick`, `run_dense_pick`, `run_attribute_pick` and `run_eih_pick` | `from_components` | the simulator gripper needs a session the vendor registry cannot reach |

Check the list against the code: `git grep -n "from_robot_config(" src datagen scripts api`, and the same
for `from_components(`.

### 1.1 Which one to use, and what the other one costs

Use `from_robot_config` for anything meant to be operated: a real cell, the console, a rehearsal, or a
simulation run whose numbers should say something about a real cell. Use `from_components` when you hold
objects config cannot describe and want no config path at all, which in practice means the simulator
runners and one-off probes.

If all you want is a cell, call neither directly. `Cell.from_tree(load_tree(), prompt=...)` gives four
ordered steps, `preflight()`, `build()`, `safety()` and `connected()`, and `build_real_cell` and
`build_rehearsal_cell` underneath build the calculator, the perception adapter and the frame resolver for
you. The contract is in the [execution README](../../src/robot/execution/README.md) and the
[autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md).

With `effective_config` at `None` and no overlays applied, these stay off whatever the YAML says:
uncertainty fusion, the drift and out-of-distribution watchdog, latency budget enforcement, the service
recovery loop, and every orchestrator setting the overlays would have made. That is what "the caller
supplies everything" means, and it means a measurement taken on `from_components` describes the runner's
own wiring, not a configured cell.

## 2. The default open-loop pick, stage by stage

This is the chain when every `grasping.*` block sits at its shipped default. Call trees and signatures are
in the [autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md) and the
[pick loop README](../../src/robot/grasping/loop/README.md); the formulas are in
[grasping-math.md](../grasping-math.md).

| # | Stage | Where it lives | What it can refuse, and why |
| --- | --- | --- | --- |
| 0 | mode check | `service.py`, `_pick_inner` | `MODE_NOT_AVAILABLE`: a per-call `mode=` needs another sampler, or a refiner or verifier that is not wired |
| 1 | perceive | `pick_loop.py`, `_execute_pick` | `NO_PERCEPTION` for a frame with no segmentations. Each retry acquires a fresh frame |
| 2 | resolve the frame | `_best_result_over_segmentations` | one `frame_resolver` call per iteration: all masks in a frame share one TCP pose |
| 3 | generate and rank | `calculator.compute_result` per segmentation | a typed failure reason (below); each mask is computed with the other masks as clutter, and the best score wins |
| 4 | route the failure | `_decide_action` | maps reasons onto `rescan`, `relocate` or `exhausted`; `relocate` without a `viewpoint_planner` becomes `rescan` |
| 5 | execute | `src/robot/grasping/motion/execution_policy.py` | `CAMERA_FRAME_REJECTED` for a grasp not in BASE while `require_base_frame_grasp` is on, before any waypoint |
| 6 | move | the same, `_drive_to` | the arm's route and its `SafetyPreflight` (below); a refusal comes back as `MOTION_FAILED` |
| 7 | close and check | the same | `OBJECT_NOT_DETECTED` when the gripper implements `ObjectDetectingGripper` and reports nothing held |
| 8 | log | `service.py`, `_maybe_log_record` | never refuses: a logging failure cannot break a pick |

The typed failure reasons of stage 3 are `empty_mask`, `no_valid_depth`, `no_candidates_generated`,
`all_collided`, `all_table_conflict`, `all_out_of_workspace`, `ik_failed` and `topology_risk_rejected`.

Stage 6 depends on what the arm says about straight lines (`KeepsLines`). An arm that keeps them (a cuRobo
UR or simulator arm, an ik UR, the dummy, the simulator mock) drives a planned move to the standoff, one line
to the grasp and line lifts. An arm that keeps none, such as a simulator on ik or RMPflow, is refused
before the jaws open as `MOTION_FAILED` with status `unsupported` and the arm's reason. An arm that does not
say keeps the interpolated approach. Inside `move()`, the driver's `SafetyPreflight` refuses with a
`[safety:<guard>/<reason>]` message, and the service maps the policy's `MOTION_FAILED` to
`EXECUTION_FAILED`.

Between stages 3 and 5 sit the optional layers: the shadow success probability, the ranking blend, the
uncertainty rerank, the learned-ranker shadow and the multi-view commit gate. At defaults each is a no-op
that records a skip reason and returns. Section 7 says what turns them on.

Three things about this chain are easy to get wrong.

**Safety is not in the pick loop.** `SafetyPreflight` is built inside each arm driver and called from that
driver's `move()`. The UR, KUKA and simulator drivers build one; the dummy driver carries none and simply
drives. A run on a dummy arm proves the perceive, generate, rank and dispatch chain, and nothing about the
guards. A test double that implements only `move_to` and not `move` bypasses safety entirely, because
`_drive_to` falls back to the bool surface. If you write a fake arm, implement `move()`.

**A fixed close width bites at the grasp point.** `close_width_mm` is a policy value unless
`_resolve_close_width` derives one, and a fixed narrow close will not lift a wide part. The failure looks
like a grasp-quality problem.

**The jaws open before the approach.** The policy `from_robot_config` builds opens the jaws to the hand's
`max_width_mm` before the first motion (`pre_open_width_mm`: 85 mm on the 2F-85, 49.99 on the Hand-E), so
a close onto a wider part never reaches the gripper as an opening. `from_components` does the same when it
builds its own policy, which it does only with a `frame_resolver`. A policy you pass in keeps what you set,
and `None` there leaves the jaws where the last close left them.

## 3. The smallest working pick

No GPU, no camera, no robot: the config-driven path on a dummy arm and dummy hand, with a synthetic one-box
scene. What it exercises is the wiring.

```powershell
python -m src.robot.execution.real_cell --rehearse --profile console_dummy --runs 2
```

Abridged, from a run on the shipped tree:

```
=== 1. CONFIG === vendor=dummy (rehearsal) profile=console_dummy (--profile)
  ... 0 blocking, 6 warnings, 0 deferred to the bench.
=== 2. BUILD === from_robot_config
  arm      DummyRobotArm
  gripper  DummyGripper
  safety     UNGATED   DummyRobotArm
=== 3. CONNECT === arm first, then gripper
=== 4. PICK === 2 run(s)
  outcome    SUCCEEDED                    mode=auto
  camera     0 of 3 motion(s) vouched; camera world  UNPLANNED  DummyRobotArm has no planner
  layers     (none)
=== 5. DOWN === gripper first, then arm, then cameras
RESULT: 2/2 succeeded
  rule: every attempt must succeed  ->  PASS
```

Read four things out of that. The warnings are the default state of a freshly configured cell, each
printed with its fix. `layers (none)` says what section 2 says: at defaults nothing advanced is wired. The
connect order is arm first, because activating a gripper can drive digital I/O. And the safety line says
the dummy arm gates nothing, so this proves wiring, not guarding.

The rehearsal keeps your tree and swaps in a dummy arm and a synthetic scene. Without `--profile console_dummy` the base tree's
Robotiq is asked for on a dummy arm, and the connect is refused with `NoRealGripper` and exit 1: a cell
whose hand cannot be built does not come up.

`--check` runs stage 1 alone and touches nothing; `--dry-run` builds without moving. The exit codes and the
stage table are in the [real_cell README](../../src/robot/execution/real_cell/README.md), and the bring-up
procedure in [real_cell_first_pick.md](../runbooks/real_cell_first_pick.md).

## 4. A worked example in the simulator

The simulator is the only path that has run full motion, measured in simulation. It is not a separate
config tree; it is the `sim` profile of `config/`. Put `--profile` on the subcommand rather than in the
environment, because `$env:WILLY_PROFILE` stays set for the whole shell and every later command then reports
simulation values.

**Gate on the motion-stack probe first.** It runs without the simulator in milliseconds. You want the
planner reported as available and the reading `fully anchored`. Exit 1 means something is missing, and
`require_motion_stack` in the simulator bootstrap refuses a cell configured for an engine it cannot find
([04](04-robot-and-safety.md)).

```powershell
python -m src.robot.safety.planning --check --profile sim
python -m src.config decisions --profile sim
```

**Then launch.** Use one file-redirected `cmd /c`, because PowerShell `>` writes UTF-16 and corrupts the log,
and a compound command hangs the boot. Run one simulator process at a time, so stop the previous one before
relaunching. Create `logs\` first: it is not committed, and on a fresh clone the redirect fails before the
simulator boots, which reads like a launch failure.

```powershell
New-Item -ItemType Directory -Force logs | Out-Null        # POSIX: mkdir -p logs
$ISAAC = "<your isaac-sim install>\python.bat"
cmd /c "$ISAAC -m src.willy_sim.run_m1_pick --runs 10 > logs\m1.log 2>&1"
```

Expect one `RUN n: ...` line per attempt and one gate line with the pass count. Harder scenes follow the
same pattern, in rising difficulty: `run_m2_pick` (real vision, takes `--prompt`), `run_eih_pick`,
`run_dense_pick --vision`, `run_multiview_pick --mode sides` and `run_industrial_bin_pick`. The runner
inventory and the gate rule are in the [willy_sim README](../../src/willy_sim/README.md).

The same known-pose gate from Python, under the simulator's interpreter
([`examples/simulation/03_isaac_pick_rate.py`](../../examples/simulation/03_isaac_pick_rate.py), which
checks the interpreter first):

```python
from willy import run_gate

result = run_gate(runs=10, headless=True, mode="easy")
print(f"{result.passed} of {result.runs} picks passed, gate passed: {result.gate_passed}")
```

Two traps. Several runners declare `def main() -> None` and always exit 0, so read the gate line, not the
exit code. And `run_m2_pick` forces the model library offline, because the simulator's runtime closes the
HTTP client and an online boot crashes, so a machine with no cached detector weights fails rather than
downloading them ([02](02-models.md)).

## 5. The retry loop, and clearing a whole bin

**`BinPickingOrchestrator.run()` executes one grasp and returns one `PickReport`.** Despite the name it is
not a bin-clearing loop. Clearing several objects is your loop over `service.pick()`.

What the orchestrator adds over a bare perceive, compute and execute: a bounded retry loop that
re-perceives each time, routing by failure reason, arbitration between segmentations, one frame-resolver
call per iteration, the commit gate when fusion is wired, per-pick state hygiene, and a typed report with
the whole attempt trail. The reason-to-action table is in the
[pick loop README](../../src/robot/grasping/loop/README.md).

For clearing, `service.set_target_label(label)` sets a hard label target: only that segmentation is
executable, and the rest stay as neighbour clutter so the collision filters still see them. `None` clears
it. Then loop, and stop on a terminal outcome:

```python
from willy import Cell, Pose, load_tree
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome as Outcome

EMPTY = {Outcome.NO_TARGET, Outcome.NO_VALID_GRASP}           # the bin looks empty
STUCK = {Outcome.EXECUTION_FAILED, Outcome.VERIFICATION_FAILED}
tray = Pose.tool_down(300.0, -250.0, 140.0)                    # BASE, inside your workspace

cell = Cell.from_tree(load_tree(), prompt="a red cube")
cell.build()
with cell.connected() as live:
    stuck = 0
    for _ in range(20):
        report = live.service.pick()                  # re-perceives from scratch every time
        print(report)
        if report.outcome in EMPTY or report.outcome is Outcome.CANCELLED:
            break
        stuck = stuck + 1 if report.outcome in STUCK else 0
        if stuck >= 3:
            break                                     # stop and diagnose (section 9)
        if report.outcome is Outcome.SUCCEEDED:
            print(cell.robot.place(tray))             # the service picks; placing is yours
```

The service does not place. `cell.robot` is the same arm and hand as a `Robot`, and `Robot.place` places a
held part at a known pose in BASE, a straight line in and out ([06](06-grippers.md)).

A gripper-level "nothing held" surfaces here as `VERIFICATION_FAILED`; there is no
`AutonomousGraspOutcome.OBJECT_NOT_DETECTED`. `CANCELLED` is not a failure: an operator stopping the run
between attempts is a decision, not a missed grasp, and counting it as `EXECUTION_FAILED` would poison the
pick rate. A stopped or powered-down controller also reads `CANCELLED` at this level; the lower-level
`report.pick_report.outcome` keeps `CONTROLLER_NOT_OPERATIONAL`, which belongs to the `PickOutcome`
enumeration, not to this one.

## 6. Grasp modes and presets

### 6.1 The five modes

Each has a fixed behaviour profile in `src/robot/execution/autonomous_grasp/config.py`.

| Mode | Sampling | Refine | Verify | Recovery actions allowed |
| --- | --- | --- | --- | --- |
| `easy` | single object | no | no | **none**, a guarantee rather than a default |
| `auto` | auto | no | no | `rescan`, `next_viewpoint` |
| `dense_clutter` | dense | no | no | `rescan`, `next_viewpoint` |
| `closed_loop` | auto | yes | yes | `rescan`, `next_viewpoint` |
| `dense_autonomous` | dense | yes | yes | `rescan`, `next_viewpoint`, `nudge_target` |

`robot.yaml` ships `default_mode: "auto"`. `resolve_grasp_mode` accepts aliases (`single`,
`single_object`, `dense`, `closedloop`, `autonomous`, and `None` for `auto`) and raises `ValueError` on
anything else, so a typo never runs the wrong sampler.

### 6.2 The per-call override is restricted on purpose

A per-call `pick(mode=...)` changes the behaviour profile only. It cannot rebuild the runtime, which was
bound to a sampling mode at construction, so a mode with another sampler is refused with a typed
`MODE_NOT_AVAILABLE`. You can move between `auto` and `closed_loop`, and between `dense_clutter` and
`dense_autonomous`. You cannot move into or out of `easy` at call time.

### 6.3 The three presets

Three partial `grasping:` overlays ship in [`config/grasping_presets/`](../../config/grasping_presets/),
each short enough to read in full. `easy.yaml` sets `default_mode: easy` and turns recovery and uncertainty
off, `dense_clutter.yaml` sets the dense mode with both on, and `verification_heavy.yaml` sets
`default_mode: closed_loop` plus a recovery block. **The config loader does not load them.** They are
library objects:

```python
from willy import Cell, load_tree
from src.config.schema.robot import RobotConfig
from src.robot.grasping.replay import apply_preset
from src.robot.grasping.replay.presets import validate_preset

validate_preset("dense_clutter")               # merge, re-validate, raise on a bad key
tree = load_tree()
merged = apply_preset(tree.robot.model_dump(mode="python"), "dense_clutter")
robot = RobotConfig.model_validate(merged)     # always do this: apply_preset does not validate
cell = Cell.from_robot_config(robot, app_config=tree.app_config, data_dir=tree.root)
```

That is what "grasping presets bypass schema validation" means: `apply_preset` merges outside the schema,
so a misspelled key survives, and `validate_preset` is the wrapper that catches it. The mechanism,
including the fact that the preset directory resolves from the repository rather than from your `--data`
tree, belongs to [01](01-configuration.md).

Two traps in the preset table. `verification_heavy.yaml` does not enable verification: it sets the mode and
a recovery block, nothing more. The `closed_loop` profile demands a wired refiner and verifier, so loading
this preset and calling `pick()` gives `MODE_NOT_AVAILABLE` unless those blocks are enabled too. And
`recovery.apply_modes` ships as `auto`, `dense_clutter` and `dense_autonomous`, so the
`recovery.enabled: true` this preset sets stays inert until a cell adds `closed_loop` to that list.

## 7. What is built, what is off, and what turns it on

These blocks under `robot.grasping` carry their own `enabled` switch, and every one ships `false`:
`closed_loop`, `verification`, `dense_recovery`, `decision`, `feasibility`, `ordering`, `recovery`,
`uncertainty`, `success_model`, `performance`, `fusion`, `approach_validation` and `deep_ranker`. Check it
with `python -m src.config where enabled --tier all --limit 500`: every `robot.grasping.*` row reads
`bool, default False`, with one exception. `fusion.cameras.*.enabled` defaults `True`, but that is the
per-entry flag on the multi-camera map, and the map is empty by default.

The shipped [`config/robot/robot.yaml`](../../config/robot/robot.yaml) leaves every one of these switches
at its schema default. Its `grasping:` block sets the always-on tunings plus `default_mode`,
`max_attempts` and `record_log_path`.

So: **the decision gate, closed-loop refinement, post-grasp verification, recovery, multi-view fusion with
its commit gate, the uncertainty rerank, the ranking blend, the learned ranker, the learned success model,
latency budget enforcement and the reinforcement-learning shadow are all built, all exercised, and all
off.** The simulator runners switch several on per command-line flag in runner code rather than in config,
which is why a simulation result can involve a decision engine while `grasping.decision.enabled` is still
false. The same statement from the code's side is in
[`src/robot/grasping/README.md`](../../src/robot/grasping/README.md); if the two ever disagree, that one
wins.

### 7.1 Where to look for a specific block

[grasping-config-reference.md](../grasping-config-reference.md) is the account of the `robot.grasping`
blocks: which mode each fires in, the traps, and what has been measured. Read its mode gate and its traps
before enabling or measuring anything. Two of its facts invalidate a result without a sound. Of the
mode-gated blocks, `easy` reaches only `success_model`, and several blocks collapse every key to its off
value outside their `apply_modes`. And several ship their operative weight at `0.0`, so `enabled: true`
alone computes a signal and multiplies it by nothing. For keys and defaults, use
[`config/all_keys/`](../../config/all_keys/), generated from the schema and valid on its own.

### 7.2 `ordering.enabled` is wired, and still reorders nothing

`apply_orchestrator_overlays` in [`builders.py`](../../src/robot/execution/autonomous_grasp/builders.py)
sets `runtime.orchestrator.target_ordering` from the block, gated on `apply_modes`, and the orchestrator's
selector reads it. But `enabled: true` alone changes no order: `ordering.unlock_weight` ships at `0.0` and
no YAML raises it, so the graph is built and the score it feeds is multiplied by zero. Two things are
missing: a non-zero weight, and a scene where picking order matters enough to measure.

### 7.3 What config cannot reach, and how to check what landed

Some inputs live on the `GraspCalculator`, a caller-supplied argument to both constructors. Config reaches
three of them by carrying them on the orchestrator and passing them per call: `gripper_geometry` becomes
`gripper_model`, `occlusion` becomes `corridor_config`, and `feasibility` becomes `feasibility_config` plus
its weight.

Config cannot supply the **IK service**. No key names one, and without it unreachable candidates are never
dropped and the `rejected_ik` counter stays at zero. The implementations are vendor-neutral and ship in
[`src/robot/execution/ik_service.py`](../../src/robot/execution/ik_service.py): `RobotArmIKService` wraps
any `RobotArm`, `CachedIKService` quantises and caches it, and `URAnalyticIKService` is an offline analytic
solver. Passing one to the calculator is a line in your own boot script.

The schema refuses to load a switch it knows is unwired, and that list lives in the code:

```powershell
python -c "from src.config.schema.robot.grasping_schema import RobotGraspingConfig as G; print(G.UNWIRED_SWITCHES)"
```

Then check what was built rather than trusting the YAML, because several loaders return `None` on purpose.
The service fields worth checking are `effective_config`, `decision_engine`, `refiner`, `verifier` and
`shadow_router`; on `service.runtime.orchestrator` they are `scene_fusion`, `commit_policy`,
`approach_path_policy`, `target_ordering`, `frame_resolver`, `viewpoint_planner`, `gripper_model` and
`corridor_config`. Each is `None` when it did not land.

## 8. Record logging

Off by default, which is why the soak, KPI and reinforcement-learning layers consume simulated and
synthetic data rather than production data. There are four ways to ask for it: `recording=` on
`PickRun.from_cell` (`Recording.to_file(path)`); `robot.grasping.record_log_path` in YAML, which only
`from_robot_config` reads; `record_log_path=` on `from_components`; or `service.enable_record_logging(path)`
at run time. The shipped `robot.yaml` sets it to `null`, and the simulator runners take
`--record-log <path>`.

**What lands.** One `json.dumps(..., sort_keys=True)` line per `pick()`, appended, with the parent
directory created and the whole write wrapped so a logging failure cannot break a pick. The record is a
`GraspAttemptRecord`: four mandatory fields (`timestamp`, `attempt_id`, `mode`, `final_outcome`), a
summary per stage that is `None` when the stage did not run, and a free-form `extra` bag. The frozen
contract is in the [telemetry README](../../src/robot/grasping/telemetry/README.md); changing it means
updating the KPI, taxonomy and reinforcement-learning consumers together.

Three things in `extra` are worth knowing. `safety_rejected` is set from membership in a fixed outcome set,
so a metric can count safety refusals without matching strings. `verification` falls back to a block
labelled as a ground-truth lift when a simulator runner recorded one, so a simulated lift is never taken
for a real verification. And `camera_world` with `camera_world_reason` name the weakest camera world behind
the attempt's typed motions (`planned`, `declined`, `unplanned` or `missing`, with an empty reason on
`planned`); both are absent when no typed motion stated one, and a runner's own `extra` cannot overwrite
them.

**What consumes it.**

```powershell
python -m src.robot.grasping.replay --records logs\attempts.jsonl        # rollup, never fails
python -m src.robot.grasping.replay --records-gate logs\attempts.jsonl   # the real gate, can fail
python -m src.robot.grasping.replay --sim-soak-report logs\attempts.jsonl --sim-min-attempts 50
```

`--records-gate` compares your log against the committed baseline thresholds, which include a minimum
attempt count in the thousands. A bring-up that logged tens of attempts is therefore red on the attempt
floor alone: honest, and useless as a signal. Point `--thresholds` at your own bring-up file, or use
`--sim-min-attempts` with the simulation gate, and say which you did when you quote the result. Do not fix
a red gate by dropping the metric.

`--soak-report` is a **synthetic contract self-check, not a quality measure**, and its own `--help` says
so. The outcome distribution is authored, so a pass proves that the telemetry, KPI, taxonomy, budget and
watchdog pipeline is consistent. It says nothing about whether the robot picks things up, and it stays
green through a broken cell. For a quality signal use `--records-gate`. Details are in the
[replay README](../../src/robot/grasping/replay/README.md).

## 9. Troubleshooting

`pick()` does not raise on an ordinary "no grasp" outcome; it returns a typed report. `print(report)`
shows it. Then read `report.outcome`, `dict(report.telemetry)`, and the attempt trail on
`report.pick_report.attempts`, where each attempt carries an index, an action and typed reasons.

**Nothing moved.**

| Outcome | Cause | Fix |
| --- | --- | --- |
| `mode_not_available`, reason `per_call_mode_requires_different_sampling_mode` | `pick(mode=...)` asked for another sampler | rebuild with that mode, or stay inside the pairs in 6.2 |
| `mode_not_available`, gate on refinement or verification | the mode needs a refiner or verifier object | enable the block *and* check the object landed (7.3) |
| `no_target` | the frame carried no segmentations | perception, not grasping: check the prompt, and that the camera is not returning black ([02](02-models.md)) |
| `missing_camera_frame` | a valid candidate existed in the camera frame, and the policy refused it | declare the primary rig's `camera.cameras.rigs[<id>].extrinsics` ([03](03-calibration.md)) |
| `decision_fail_closed`, `uncertainty_fail_closed` | the decision gate refused | wire a viewpoint planner, or lower the thresholds knowingly |
| `no_commit_insufficient_fusion` | a candidate existed, fused evidence did not, and the budget is spent | check that `scene_fusion` is wired; the commit policy alone is inert |
| `drift_blocked_auto`, `ood_blocked_auto` | the watchdog locked autonomous operation | read the drift and out-of-distribution telemetry; do not bypass it |

**Which camera world stood behind the motions.** Every printed report has a `camera` line, and
`report.pick_report.camera_worlds` holds one stamp per typed motion of the last attempt, read weakest first
through `camera_world`. On a dummy arm, an `ik` cell or the simulator mock every stamp reads `UNPLANNED`
with the driver's reason, as in the rehearsal of section 3. That is the correct answer, not a fault. On a
cuRobo cell:

- No camera world wired and nothing declined: every planned motion is refused before planning. The pick
  ends `execution_failed`, status `unsupported`, stamp `MISSING`, with a message starting
  `Refused before planning: this cell plans with cuRobo, no live camera world is wired`.
- A calibrated rig that gives no world: the cell does not build, and the checklist's `camera world` row
  says so first.
- A declined motion reads `DECLINED`, and where a live world is wired (`safety.planning_world.enabled`) a
  declined planned motion is refused before planning.
- A wired world is built from every enabled RGB-D rig that declares its calibration, the primary first.
  One of them that cannot answer stops every planned motion.

A wired world leaves out the space between the open jaws at each motion's goal: the hand's registry jaw on
the declared TCP, with no padding. A part between the fingers is then no obstacle at the goal, while a post
where a finger closes still is. A part wider than the fingers keeps its box, so the pick also holds its
target out of the world for the whole attempt. A `PLANNED` stamp says what was left out
(`kept out: goal region left out N point(s)`); no recorded run reads `PLANNED`. The grasp record carries
the weakest stamp as `extra.camera_world` and `extra.camera_world_reason`.

**The arm refused.** A safety rejection surfaces as `execution_failed` with a motion message prefixed
`[safety:<guard>/<reason>]`. Guard order and verdicts are in [04](04-robot-and-safety.md). Two things
belong here: `enforce: false` on a guard removes it at construction, so its surface is gone rather than
quiet; and on a cell that runs `ik`, blind IK proposes self-colliding branches that the guard then rejects,
which looks exactly like bad grasping. Run the motion-stack probe before you blame the grasps.

**No candidates.** Read `attempt.reasons` and the calculator's telemetry line, which carries
`candidates_silhouette`, `candidates_geometry`, the `rejected_*` counters, `final` and `best_score`. On a
fresh bring-up check `no_candidates_generated` first: the generator returned nothing at all, which a jaw
too narrow can cause, and so can a mask with no usable geometry, no antipodal pair or no valid depth. If
poses were generated and every one fell outside the jaw range, check `robot.gripper.min_width_mm` and
`max_width_mm` against the part and check the pixel-to-millimetre scale. `ik_failed` is reachable only with
an IK service wired (7.3). Per-stage detail is in the
[generation README](../../src/robot/grasping/generation/README.md).

**It grabbed air.** `object_not_detected` is the honest path and what you want to see. `executed` with the
part still on the table means nothing checked: the gripper does not implement `ObjectDetectingGripper`, or
the cell names no end-effector (`gripper.vendor: none`), which connects ([06](06-grippers.md)). For the debug overlay, `render_debug_images` is `False` by default, so
`service.last_debug_image_png` stays `None` until you call `service.enable_debug_image_rendering(True)`.
The simulator runners do it behind `--debug-frames DIR`, and only the vision cells produce an image.

**A feature was turned on and nothing changed.** In order of likelihood: you are on `from_components` and
`effective_config` is `None`; the mode you ran cannot reach the block; you set `enabled: true` and left the
operative weight at zero; a constructor argument from a runner beat your config block; or you enabled a
consumer whose producer you did not wire. Read section 7, then
[grasping-config-reference.md](../grasping-config-reference.md).

## 10. Next steps for real hardware

Nothing here has touched hardware. `from_robot_config` has live callers and `--rehearse` drives the whole
path on a dummy arm, but the UR driver is measured against real controller software only, the KUKA driver
never touched hardware, and Franka and ROS 2 are empty registry slots that raise on `create_arm`. In
order:

1. **Run the desk checklist and clear the blocking items.** `real_cell --check` reports each with its fix,
   and every one would otherwise surface at the bench as a different-looking failure. Details:
   [04](04-robot-and-safety.md); procedure: [real_cell_first_pick.md](../runbooks/real_cell_first_pick.md).
2. **Get a `PerceptionSource` for your camera.** A RealSense is covered by `build_real_components`.
   Anything else is yours to write, and it has to exist before anything else runs.
3. **Wire an IK service into the calculator.** Config cannot (7.3).
4. **Anchor the motion stack.** Run `python -m src.robot.safety.planning --doctor` and
   `python -m src.robot.execution.real_cell --start-planner`. The checklist blocks on its `cuRobo environment`
   and `exact mesh engine` rows when either engine is missing, and a UR that plans with cuRobo refuses a
   motion rather than falling back to blind IK ([04](04-robot-and-safety.md), section 6).
5. **Turn on record logging from the first motion**, then gate with `--records-gate`, not
   `--soak-report`. Real records are the only thing that turns the soak, KPI and reinforcement-learning
   layers from a self-check into a measurement.
6. **Start in `easy`.** Its recovery allow-list is empty by construction, so it never produces recovery
   motion. `closed_loop` needs both a refiner and a verifier wired and refuses the pick otherwise, so it is
   not where a bring-up starts.
7. **Re-measure after every change, and never carry a planner margin to a robot nobody measured it on.** A
   thinner-linked arm reads as permanently self-colliding at a 10 mm margin and finds no plan at all. The UR
   family was measured at 4 mm, and a cuRobo cell starts its planner only with a committed evidence file
   for its margin ([04](04-robot-and-safety.md)).

**Out of scope, and not reachable by wiring code:** certified functional safety. The planner's collision
awareness and the exact-mesh guard reduce risk in software. A real cell still needs the vendor's
safety-rated stop, hardware emergency-stop circuits, and ISO 10218, ISO/TS 15066 and ISO 13849 compliance.
A green simulation gate does not suggest otherwise.

## Where to look next

| For | Read |
| --- | --- |
| the service, both constructors, the cell builders | [autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md) |
| the cell as one noun, and a campaign of picks | [execution README](../../src/robot/execution/README.md) |
| the same calls from a program | [`examples/`](../../examples/README.md) |
| the config-driven command line, its stages and exit codes | [real_cell README](../../src/robot/execution/real_cell/README.md) |
| the retry loop, reasons to actions, progress events | [pick loop README](../../src/robot/grasping/loop/README.md) |
| candidate generation, the generators, the scoring axes | [generation README](../../src/robot/grasping/generation/README.md) . [scoring README](../../src/robot/grasping/scoring/README.md) |
| motion and gripper choreography | [motion README](../../src/robot/grasping/motion/README.md) |
| the record contract, and the soak and KPI command line | [telemetry README](../../src/robot/grasping/telemetry/README.md) . [replay README](../../src/robot/grasping/replay/README.md) |
| the simulator runners and their gate rule | [willy_sim README](../../src/willy_sim/README.md) |
| every `robot.grasping` block, its mode gate and its traps | [grasping-config-reference.md](../grasping-config-reference.md) |
| every key the schema accepts, with defaults | [`config/all_keys/`](../../config/all_keys/) |
| the formulas: back-projection, normals, antipodal search, scoring | [grasping-math.md](../grasping-math.md) |
| workspace, forward kinematics, Jacobian, capsules, mesh distance | [safety-math.md](../safety-math.md) |
