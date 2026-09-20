# The pick service (`src/robot/execution/autonomous_grasp`)

`AutonomousGraspService` runs one pick attempt per `pick()` and answers with one frozen
`AutonomousGraspReport`: perceive, generate and rank grasps, gate, move and log, plus whichever
opt-in layers the grasp mode and the config switch on. You reach it through `Cell.build()` and
`PickRun`; build it yourself only when you hold a grasp calculator and a perception source of your own.

```python
from willy import Cell, load_tree

cell = Cell.rehearsal(load_tree("console_dummy").robot)   # a dummy arm and a synthetic scene
service = cell.build()                                     # the AutonomousGraspService behind the cell
with cell.connected():
    report = service.pick()                                # one attempt; a refusal is an outcome
print(report)                                              # outcome, candidates, camera world, layers
```

At a real cell `Cell.from_tree(load_tree(), prompt="a red cube")` builds the same service with the
cell's cameras, models and planner, and
[13_pick_campaign.py](../../../../examples/real_robot/13_pick_campaign.py) runs a campaign of them.

## The nouns

| Noun | Built by | Verb | Returns |
| --- | --- | --- | --- |
| `AutonomousGraspService` | `Cell.build()`, `build_real_cell(robot_cfg, prompt=)`, `from_robot_config`, `from_components` | `pick(mode=None)` | `AutonomousGraspReport` |
| `GraspMode` | `resolve_grasp_mode(value)`, or `Cell(..., mode=)` at the build | | `easy`, `auto`, `dense_clutter`, `closed_loop`, `dense_autonomous` |
| `PickPrompt` | `PickPrompt.from_text(text)` | `service.set_prompt(text)` | the prompt it replaced |

`resolve_grasp_mode` also takes the aliases `single`, `single_object`, `dense`, `closedloop` and
`autonomous`, and `None` is `auto`. `mode=` on `pick()` changes the behaviour profile of one attempt,
never the sampler the service was built with -- so the mode a cell RUNS IN is chosen at the build,
`Cell.from_tree(tree, mode="dense_clutter")`, and asking a service built in one sampler for another
comes back `MODE_NOT_AVAILABLE`.
[`simulation/06`](../../../../examples/simulation/06_grasp_modes_and_what_each_needs.py) prints
every mode, what each one locks and which of them this cell can run; `willy` exports the service,
the report, the outcome, `GraspMode` and `PickPrompt`, and
[`real_robot/18`](../../../../examples/real_robot/18_clear_a_bin_with_recovery.py) drives the
service directly to clear a bin.

`set_prompt` changes what the next picks look for (the phrase every camera grounds, the labels the
detector's words map onto, and the label filter) with no camera reopened and no model reloaded:

```python
previous = service.set_prompt("the red cube")
report = service.pick()
service.set_prompt(previous)                       # all three back
```

To build the service from parts of your own, `calculator` and `perception` are required and have no
default, because a forgotten argument must not open a camera. A real vendor also needs the CAMERA to
BASE of its camera, from the camera section or as `frame_resolver=`:

```python
from willy import load_tree
from src.robot.execution.autonomous_grasp import AutonomousGraspService

tree = load_tree()
service = AutonomousGraspService.from_robot_config(
    tree.robot, calculator=my_calculator, perception=my_source, camera=tree.app_config.camera)
```

`build_real_cell(robot_cfg, prompt=...)` supplies all three from the tree, and is what `Cell.build()`
calls. `service.enable_record_logging(path)` appends one `GraspAttemptRecord` per pick.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `MODE_NOT_AVAILABLE` outcome | `closed_loop` without a refiner, `dense_autonomous` without a verifier, a `mode=` of another sampler | enable the block, or build the service in that mode |
| `EXECUTION_FAILED` with `fault` | a `RobotError`, `RuntimeError` or `OSError` during the pick | `PickRun` and the console stop the campaign on it |
| `MISSING_CAMERA_FRAME` outcome | a grasp won and no frame resolver maps it to BASE | declare the camera's calibration on its rig |
| `ValueError` | no `mode` and no `robot.grasping` block declared | declare the block, or pass `mode=` |
| `ValueError` | a real vendor with no CAMERA to BASE, or a physical recovery action with no `recovery_fixture` | as the message says |
| `ValueError`, `TypeError` | a `policy=` on another arm or hand; both `motion=` and `policy=` | pass `motion=GraspMotion(...)` alone |

A programmer's error still raises from `pick()`, and so do `NotImplementedError` and `RecursionError`.

## Status

| Capability | Evidence |
| --- | --- |
| The service on a UR5e with a 2F-85 in Isaac Sim | measured in simulation ([willy_sim](../../../willy_sim/README.md)) |
| The service on a physical arm | never touched hardware |

The default attempt is open-loop. Every advanced `robot.grasping` block ships `enabled: false`
(`decision`, `closed_loop`, `verification`, `uncertainty`, `feasibility`, `ordering`, `recovery`,
`dense_recovery`, `fusion`, `success_model`, `deep_ranker`, `approach_validation`, `performance`),
and `robot.rl.mode` ships `hybrid_ml`, which builds no shadow router. The report's `layers` line
names what actually ran, read off the attempt rather than the config.

Two ways to build the service are not equivalent. `from_robot_config` reads the whole tree and fills
`effective_config`. `from_components`, for a caller holding live handles the config cannot describe,
leaves it `None`, which silences uncertainty fusion, the watchdog, breach events and the recovery
loop until the caller calls `build_effective_config` and `apply_orchestrator_overlays`.
`from_robot_config` takes live `arm` and `gripper` handles for exactly that reason.

Record logging is off unless `grasping.record_log_path` is set (it takes `${WILLY_RECORD_LOG:-}`, and
an empty value is off). Each pick carries a unique `attempt_id`. A cell that logs records while
`robot.rl.mode` is not `rl_shadow` warns once at boot, because a pairwise ranker cannot train on them.
`EffectiveGraspingConfig.to_dict()` is the flat telemetry contract of 82 keys, each the state that
acted on the attempt: a block outside its `apply_modes` reads false even where the YAML says true.

## Files

| File | Holds |
| --- | --- |
| `service.py` | `AutonomousGraspService`: the factories, `pick()`, `set_prompt()`, the attempt, the decision loop, refine and verify |
| `cells.py` | `build_real_cell`, `build_rehearsal_cell` and their component builders, `CellBuildRefused` |
| `config.py` | `GraspMode`, `resolve_grasp_mode`, `GraspBehaviorProfile`, `EffectiveGraspingConfig` and its per-phase parts |
| `report.py` | `AutonomousGraspOutcome`, `AutonomousGraspReport` |
| `prompt.py` | `PickPrompt` |
| `rehearsal.py` | the synthetic one-box scene a rehearsal perceives |
| `builders.py` | the per-phase wiring `from_robot_config` runs, `build_effective_config`, `apply_orchestrator_overlays` |
| `watchdog.py`, `latency.py` | drift and out-of-distribution events; rolling p95 latency, which never fails closed |
| `shadow.py`, `action_mask_eval.py` | the reinforcement-learning shadow router and its per-candidate action mask |
| `record_logging.py` | the report to `GraspAttemptRecord` serializer |

## Details

- Guide: [the pick loop](../../../../docs/guide/05-pick-loop.md); every `robot.grasping` block and the mode gate: [grasping-config-reference.md](../../../../docs/grasping-config-reference.md)
- The parent package: [execution](../README.md); the command line: [real_cell](../real_cell/README.md)
- What it drives: [grasping](../../grasping/README.md), [safety](../../safety/README.md), [grasping/rl](../../grasping/rl/README.md)
- Tests: `tests/test_autonomous_grasp_service.py`, `tests/test_autonomous_grasp_report_contract.py`, `tests/test_cell_builders.py`, `tests/test_pick_reports_a_cell_fault.py`, `tests/test_effective_config_field_split.py`
