# 5. Assembling and running the pick loop

You have a config tree that validates ([01](01-configuration.md)), perception models that load
([02](02-models.md)), a camera-to-base transform ([03](03-calibration.md)) and an arm whose safety
pipeline is anchored ([04](04-robot-and-safety.md)). This is the last mile: putting those pieces into
one object and making something leave the table.

**What you get.** A pick you can run in a minute with no GPU, no camera and no robot. The default
pick traced stage by stage, and what each stage may refuse. Where the telemetry record comes from.
And an account of which advanced layers exist, which are off, and what turns each one on.

**What this page will not tell you.** It will not tell you the advanced stack is running. It is not.
The default pick is open-loop, and the most expensive mistake available here is believing a feature is
active because a YAML key exists for it. Read section 7 before reporting a number.

Sibling guides: [01](01-configuration.md) . [02](02-models.md) . [03](03-calibration.md) .
[04](04-robot-and-safety.md) . **05** . [06](06-grippers.md)

## Prerequisites

| # | Requirement | How to check | Guide |
| --- | --- | --- | --- |
| 1 | The config tree validates | `python -m src.config` (exit 0) | [01](01-configuration.md) |
| 2 | You know which keys exist | `python -m src.config where grasping --limit 500`, or the reference tree [`config/all_keys/`](../../config/all_keys/) | [01](01-configuration.md) |
| 3 | A `PerceptionSource`, meaning anything with `.acquire() -> PerceptionFrame` | see the note below | [02](02-models.md) |
| 4 | Camera intrinsics, and a `FrameResolver` for BASE-frame grasps (on hardware you need one) | a calibrated intrinsics file, `get_intrinsics()` on an RGB-D handle, or a hand-authored 3x3 matrix | [03](03-calibration.md) |
| 5 | An arm whose vendor SDK is present | `python -m src.robot.drivers.doctor` | [04](04-robot-and-safety.md) |
| 6 | For the simulator: the two motion sidecars anchored | `python -m src.robot.safety.planning --check` | [04](04-robot-and-safety.md) |

On prerequisite 3: one real-camera adapter ships, `RealSenseVisionPerceptionSource`, which
`build_real_components` in `src/robot/execution/autonomous_grasp/cells.py` constructs for a physical cell. Everything else
that implements `acquire()` is simulated or synthetic. For any other camera the adapter is yours to
write, and the contract is small: a depth map in millimetres, a 3x3 intrinsics matrix, and
segmentations that each carry a boolean `.mask`.

Shell blocks are Windows PowerShell from the repository root. POSIX differences are noted where they
matter.

## 1. The two constructors

`AutonomousGraspService` has two, and both are live.

`from_robot_config(robot_cfg, calculator=..., perception=...)` is the config-driven path. It reads
`robot.grasping`, builds the arm and gripper through the vendor registry, resolves the grasp mode and
the attempt budget, builds sub-policies for any enabled block, computes an `EffectiveGraspingConfig`,
applies the orchestrator overlays, and switches on record logging if the config names a path. It
accepts `arm=` and `gripper=` for the one thing config cannot express: a live device handle, such as a
simulator gripper that must share its arm's session.

`from_components(arm=..., calculator=..., perception=...)` takes objects you already built. It reads
no config at all: `effective_config` stays `None` and no overlay runs.

| Caller | Constructor | Why that one |
| --- | --- | --- |
| `build_real_cell` and `build_rehearsal_cell` in `src/robot/execution/autonomous_grasp/cells.py` | `from_robot_config` | a whole cell in one call: physical, or a desk rehearsal on a dummy arm |
| `python -m src.robot.execution.real_cell`, the operator console, and [`scripts/examples/cell/03_first_pick.py`](../../scripts/examples/cell/03_first_pick.py) | `from_robot_config`, through `Cell` and those two builders | the terminal, the browser and a Python script drive one construction |
| `run_multiview_pick.py` with `--boot config`, its default | `from_robot_config` | the simulator booting the way a real cell boots |
| `datagen/rl/occupancy.py` | `from_robot_config` | features from the real stack rather than from a copy of it |
| `run_m1_pick`, `run_m2_pick`, `run_dense_pick`, `run_attribute_pick`, and `run_eih_pick` by default | `from_components` | the simulator gripper needs a session the vendor registry cannot reach |

Establish that list yourself rather than trusting any document, including this one:
`git grep -n "from_robot_config(" src datagen scripts api`, and the same for `from_components(`.

### 1.1 Which one to use, and what the other one costs

Use `from_robot_config` for anything meant to be operated: a real cell, the console, a rehearsal, or a
simulation run whose numbers are supposed to say something about a real cell. Use `from_components`
when you hold objects config cannot describe and do not want the config path at all, which in practice
means the simulator runners and one-off probes.

If all you want is a cell, call neither directly. `Cell.from_robot_config(robot_cfg, prompt=...)`
gives four ordered steps, `preflight()`, `build()`, `safety()` and `connected()`, and
`build_real_cell` / `build_rehearsal_cell` underneath build the calculator, the perception adapter and
the frame resolver for you. Contract in the
[execution README](../../src/robot/execution/README.md) and the
[autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md).

With `effective_config` at `None` and no overlays applied, these go quiet whatever the YAML says:
uncertainty fusion, the drift and out-of-distribution watchdog, latency budget enforcement, the
service recovery loop, and every orchestrator carrier the overlays would have set. That is not a
defect, it is what "the caller supplies everything" means. It does mean a measurement taken on
`from_components` describes the runner's hand-wiring, not a configured cell.

## 2. The default open-loop pick, stage by stage

This is the chain when every `grasping.*` block sits at its shipped default. Call trees and signatures
are in the [autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md) and the
[pick loop README](../../src/robot/grasping/loop/README.md); the formulas are in
[grasping-math.md](../grasping-math.md).

| # | Stage | Where it lives | What it can refuse, and why |
| --- | --- | --- | --- |
| 0 | mode check | `service.py`, `_pick_inner` | `MODE_NOT_AVAILABLE` when a per-call `mode=` needs a different sampler than the runtime was built with, or when the mode needs a refiner or verifier that is not wired |
| 1 | perceive | `pick_loop.py`, `_execute_pick` | a frame with no segmentations ends the pick as `NO_PERCEPTION`. Each retry re-acquires a fresh frame, so a retry is a real re-perceive |
| 2 | resolve the frame | `_best_result_over_segmentations` | one `frame_resolver` call per iteration, not per segmentation: all masks in a frame share one TCP pose |
| 3 | generate and rank | `calculator.compute_result` per segmentation | typed failure reasons: `empty_mask`, `no_valid_depth`, `no_candidates_generated`, `all_collided`, `all_table_conflict`, `all_out_of_workspace`, `ik_failed`, `topology_risk_rejected`. Each segmentation is computed with the *other* masks as neighbour clutter, then the best score wins |
| 4 | route the failure | `_decide_action` | maps reasons onto `rescan`, `relocate` or `exhausted`. `relocate` needs a `viewpoint_planner`; without one it degrades to `rescan` |
| 5 | execute | `src/robot/grasping/motion/execution_policy.py` | `CAMERA_FRAME_REJECTED` when the grasp is not in the BASE frame and `require_base_frame_grasp` is on. Checked before any waypoint is built |
| 6 | move | the same, `_drive_to` | the arm driver's `SafetyPreflight` refuses inside `move()`. The refusal comes back as the policy outcome `MOTION_FAILED`, carrying a `[safety:<guard>/<reason>]` message, and the service maps it to `EXECUTION_FAILED` |
| 7 | close and check | the same | `OBJECT_NOT_DETECTED` if the gripper implements `ObjectDetectingGripper` and reports nothing held |
| 8 | log | `service.py`, `_maybe_log_record` | never refuses. Record logging is wrapped so a logging failure cannot break a pick |

Between stages 3 and 5 sit the optional layers: the shadow success probability, the ranking blend, the
uncertainty rerank, the learned-ranker shadow and the multi-view commit gate. At defaults each is a
no-op that stamps a skip reason and returns. Section 7 says what turns them on.

Two things about this chain are easy to get wrong.

**Safety is not in the pick loop.** `SafetyPreflight` is constructed inside each *arm driver* and
called from that driver's `move()`. The UR, KUKA and simulator drivers build one; the dummy driver
says outright that it carries no preflight and simply drives. A run on a dummy arm proves the
perceive, generate, rank and dispatch chain and nothing about the guards. Related: a test double that
implements only `move_to` and not `move` bypasses safety entirely, because `_drive_to` falls back to
the legacy bool surface. If you write a fake arm, implement `move()`.

**A fixed close width bites at the grasp point.** `close_width_mm` is a policy value unless
`_resolve_close_width` derives one, and a fixed narrow close will not lift a wide object. The failure
looks like a grasp-quality problem.

## 3. The smallest working pick

No GPU, no camera, no robot. This is the config-driven path on a dummy arm, with a synthetic one-box
scene, so what it exercises is the wiring.

```powershell
python -m src.robot.execution.real_cell --rehearse --runs 2
```

Abridged, from a run on the shipped tree:

```
=== 1. CONFIG === vendor=dummy (rehearsal) profile=<none>
  ... 0 blocking, 8 warnings, 0 deferred to the bench.
=== 2. BUILD === from_robot_config
  arm      DummyRobotArm
  gripper  NullGripper
  safety     UNGATED   DummyRobotArm
=== 3. CONNECT === arm first, then gripper
=== 4. PICK === 2 run(s)
  outcome    SUCCEEDED                    mode=auto
  layers     (none)
=== 5. DOWN === gripper first, then arm, then cameras
RESULT: 2/2 succeeded
  rule: every attempt must succeed  ->  PASS
```

Four things to read out of that. The warnings are the honest default state of a freshly configured
cell, each printed with its fix. `layers (none)` says in two words what section 2 says at length: at
defaults nothing advanced is wired. The connect order is arm first, because activating a gripper
drives digital I/O. And the safety line says the dummy arm gates nothing, so this proves wiring and
not guarding.

`--check` runs stage 1 alone and touches nothing; `--dry-run` builds without moving. Exit codes and
the stage table are in the [real_cell README](../../src/robot/execution/real_cell/README.md), the
bring-up procedure in [real_cell_first_pick.md](../runbooks/real_cell_first_pick.md), and the same
path with a narrated inventory of which layers ran in
[`scripts/examples/cell/03_first_pick.py`](../../scripts/examples/cell/03_first_pick.py).

## 4. A worked example in the simulator

The simulator is the only path that has run full motion. It is not a separate config tree, it is the
`sim` profile of `config/`. Put `--profile` on the subcommand rather than in the environment, because
`$env:WILLY_PROFILE` is sticky for the whole shell and every later command then silently reports
simulation values.

**Gate on the motion-stack probe first.** It runs off-box in milliseconds. You want the planner
reported as available and the reading `fully anchored`; exit 1 means a degraded fallback is in force,
and `require_motion_stack` in the simulator bootstrap fails closed on a cell configured for an engine
it cannot find ([04](04-robot-and-safety.md)).

```powershell
python -m src.robot.safety.planning --check
python -m src.config decisions --profile sim
```

**Then the launch discipline**, which every guide in this set refers back to. One single
file-redirected `cmd /c`, because PowerShell `>` writes UTF-16 and corrupts the log while a compound
command hangs the boot. One simulator process at a time, so kill the previous one before relaunching.
And create `logs\` first: it is not committed, so on a fresh clone the redirect fails before the
simulator boots, which reads exactly like a launch failure.

```powershell
New-Item -ItemType Directory -Force logs | Out-Null        # POSIX: mkdir -p logs
$ISAAC = "<your isaac-sim install>\python.bat"
cmd /c "$ISAAC -m src.willy_sim.run_m1_pick --runs 10 > logs\m1.log 2>&1"
```

Expect one `RUN n: ...` line per attempt and one gate line naming how many passed. Harder scenes
follow the same pattern, in increasing difficulty: `run_m2_pick` (real vision, takes `--prompt`),
`run_eih_pick`, `run_dense_pick --vision`, `run_multiview_pick --mode sides` and
`run_industrial_bin_pick`. The runner inventory and the gate rule are in the
[willy_sim README](../../src/willy_sim/README.md), which sits next to the code.

Two traps: several runners declare `def main() -> None` and therefore always exit 0, so parse the gate
line rather than the exit code; and `run_m2_pick` deliberately forces the model library offline,
because the simulator's own runtime closes the HTTP client and an online boot crashes, so a machine
with no cached detector weights fails rather than downloading ([02](02-models.md)).
[`scripts/examples/sim/70_sim_pick.py`](../../scripts/examples/sim/70_sim_pick.py) fronts the first runners and checks
the interpreter before anything else, which is the single most common way an hour disappears here.

## 5. The retry loop, and clearing a whole bin

**`BinPickingOrchestrator.run()` executes one grasp and returns one `PickReport`.** Despite the name
it is not a bin-clearing loop: read the return type. Multi-object clearing is the caller's loop over
`service.pick()`, and you write that loop.

What it adds over a bare perceive, compute, execute: a bounded retry loop that re-perceives each time,
failure-reason-driven routing, per-segmentation arbitration, one frame-resolver call per iteration,
the commit gate when fusion is wired, per-pick state hygiene, and a typed report carrying the whole
attempt trail. The reason-to-action table is in the
[pick loop README](../../src/robot/grasping/loop/README.md).

For clearing, `service.set_target_label(label)` sets a hard label target: only that segmentation is
executable, and the rest stay as neighbour clutter so the collision filters still see them. `None`
clears it. Then loop, and stop on a terminal outcome:

```python
from src.robot.execution.autonomous_grasp import AutonomousGraspOutcome as O

TERMINAL = {O.NO_TARGET, O.NO_VALID_GRASP}                  # the bin looks empty
STUCK    = {O.EXECUTION_FAILED, O.VERIFICATION_FAILED}      # the cell is stuck: not the same thing

stuck = 0
for i in range(max_picks):
    report = service.pick()
    if report.outcome in TERMINAL:
        break
    stuck = stuck + 1 if report.outcome in STUCK else 0
    if stuck >= 3:
        break          # stop and diagnose with section 9, do not keep swinging
    # place the object, then loop; the next pick() re-perceives from scratch
```

A gripper-level "nothing held" surfaces here as `VERIFICATION_FAILED`; there is no
`AutonomousGraspOutcome.OBJECT_NOT_DETECTED`. `CANCELLED` is deliberately not a failure: an operator
stopping the run between attempts is a decision, not a missed grasp, and folding it into
`EXECUTION_FAILED` would poison the pick-rate metric. A controller in protective stop surfaces on the
lower-level `PickOutcome` as `CONTROLLER_NOT_OPERATIONAL`, which is a different enumeration:
extending the sets above with it raises `AttributeError`.

## 6. Grasp modes and presets

### 6.1 The five modes

Each has a locked, non-tunable behaviour profile in `src/robot/execution/autonomous_grasp/config.py`.

| Mode | Sampling | Refine | Verify | Recovery actions allowed |
| --- | --- | --- | --- | --- |
| `easy` | single object | no | no | **none**, and that is a guarantee, not a default |
| `auto` | auto | no | no | `rescan`, `next_viewpoint` |
| `dense_clutter` | dense | no | no | `rescan`, `next_viewpoint` |
| `closed_loop` | auto | yes | yes | `rescan`, `next_viewpoint` |
| `dense_autonomous` | dense | yes | yes | `rescan`, `next_viewpoint`, `nudge_target` |

`robot.yaml` ships `default_mode: "auto"`. `resolve_grasp_mode` accepts aliases (`single`,
`single_object`, `dense`, `closedloop`, `autonomous`, and `None` for `auto`) and raises `ValueError`
on anything else, so a typo never silently runs the wrong sampler.

### 6.2 The per-call override is restricted on purpose

A per-call `pick(mode=...)` reinterprets the behaviour profile only. It cannot rebuild the runtime,
which was bound to a sampling mode at construction, so a mode with a different sampler is refused with
a typed `MODE_NOT_AVAILABLE`. You can move between `auto` and `closed_loop`, and between
`dense_clutter` and `dense_autonomous`. You cannot move into or out of `easy` at call time.

### 6.3 The three presets

Three partial `grasping:` overlays ship in
[`config/grasping_presets/`](../../config/grasping_presets/), each short enough to read in full:
`easy.yaml` sets `default_mode: easy` and turns recovery and uncertainty off, `dense_clutter.yaml`
sets the dense mode with both on, and `verification_heavy.yaml` sets `default_mode: closed_loop` plus
a recovery block. **The config loader does not load them.** They are library objects:

```python
from src.config.schema.robot import RobotConfig
from src.robot.grasping.replay import apply_preset
from src.robot.grasping.replay.presets import validate_preset

validate_preset("dense_clutter")                  # deep-merge, re-validate, raise on a bad key
merged = apply_preset(cfg.robot.model_dump(mode="python"), "dense_clutter")
robot  = RobotConfig.model_validate(merged)       # always do this; apply_preset does not
```

That is what "grasping presets bypass schema validation" means: `apply_preset` merges outside the
schema, so a typo'd key survives silently, and `validate_preset` is the wrapper that catches it. The
mechanism, including the trap that the preset directory resolves from the repository rather than from
your `--data` tree, belongs to [01](01-configuration.md).

Two traps in the preset table. `verification_heavy.yaml` does not enable verification: it sets the
mode and a recovery block, nothing more, and the `closed_loop` profile demands a wired refiner and
verifier, so loading this preset and calling `pick()` yields `MODE_NOT_AVAILABLE` unless those blocks
are enabled too. And `recovery.apply_modes` ships as `auto`, `dense_clutter` and `dense_autonomous`,
so the `recovery.enabled: true` that this preset sets stays inert until a cell adds `closed_loop` to
that list.

## 7. What is built, what is off, and what turns it on

Thirteen blocks under `robot.grasping` carry their own `enabled` switch, and every one of them ships
`false`: `closed_loop`, `verification`, `dense_recovery`, `decision`, `feasibility`, `ordering`,
`recovery`, `uncertainty`, `success_model`, `performance`, `fusion`, `approach_validation` and
`deep_ranker`. Verify it with `python -m src.config where enabled --tier all --limit 500`: every
`robot.grasping.*` row reads `bool, default False`, with one exception. `fusion.cameras.*.enabled`
defaults `True`, but that is the per-entry flag on the multi-camera map, and the map is empty by
default.

The shipped [`config/robot/robot.yaml`](../../config/robot/robot.yaml) writes only two of them, both
at their defaults, `deep_ranker.enabled` and `support.container.wall_collision_enabled`, and leaves
the rest to the schema; its `grasping:` block otherwise sets the always-on tunings plus
`default_mode`, `max_attempts` and `record_log_path`.

So the honest summary: **the decision gate, closed-loop refinement, post-grasp verification, recovery,
multi-view fusion with its commit gate, the uncertainty rerank, the ranking blend, the learned ranker,
the learned success model, latency budget enforcement and the reinforcement-learning shadow are all
built, all exercised, and all off.** The simulator runners switch several on per command-line flag in
runner code rather than in config, which is why a simulation result can involve a decision engine
while `grasping.decision.enabled` is still false. The same statement, from the code's side, is in
[`src/robot/grasping/README.md`](../../src/robot/grasping/README.md); if the two ever disagree, that one wins.

### 7.1 Where to look for a specific block

[grasping-config-reference.md](../grasping-config-reference.md) is the account of the
`robot.grasping` blocks: which mode each fires in, the traps, and what has been measured. Read its
mode gate and its traps before enabling or measuring anything. Two of its facts invalidate a result
silently, so carry them. Of the mode-gated blocks, `easy` reaches only `success_model`, and several
blocks collapse every key to its off value outside their `apply_modes`. And several ship their
operative weight at `0.0`, so `enabled: true` alone computes a signal and multiplies it by nothing.
For keys and defaults use the reference tree [`config/all_keys/`](../../config/all_keys/), which is
generated from the schema and validates on its own.

### 7.2 `ordering.enabled` is wired, and still reorders nothing

`apply_orchestrator_overlays` in
[`builders.py`](../../src/robot/execution/autonomous_grasp/builders.py) assigns
`runtime.orchestrator.target_ordering` from the block, gated on `apply_modes`, and the orchestrator's
selector reads it. But `enabled: true` alone changes no order: `ordering.unlock_weight` ships at
`0.0` and no YAML raises it, so the graph is built and the score it feeds is multiplied by zero. Two
things are missing, not one: a non-zero weight, and a scene where picking order matters enough to
measure.

### 7.3 What config cannot reach, and how to check what landed

Some inputs live on the `GraspCalculator`, a caller-supplied argument to both constructors. Config
reaches three of them by carrying them on the orchestrator and passing them per call:
`gripper_geometry` becomes `gripper_model`, `occlusion` becomes `corridor_config`, and `feasibility`
becomes `feasibility_config` plus its weight.

It cannot supply the **IK service**. There is no config key for one, and without it unreachable
candidates are never dropped and the `rejected_ik` counter stays at zero forever. The implementations
are vendor-neutral and ship in [`src/robot/execution/ik_service.py`](../../src/robot/execution/ik_service.py):
`RobotArmIKService` wraps any `RobotArm`, `CachedIKService` quantises and caches it, and
`URAnalyticIKService` is an offline analytic solver. Passing one to the calculator is a line in your
own boot script.

The schema also refuses to load a switch it knows is unwired, and that list belongs in the code rather
than here:

```powershell
python -c "from src.config.schema.robot.grasping_schema import RobotGraspingConfig as G; print(G.UNWIRED_SWITCHES)"
```

Then assert on what was built rather than trusting the YAML, because several loaders degrade to `None`
on purpose. The service fields worth checking are `effective_config`, `decision_engine`, `refiner`,
`verifier` and `shadow_router`; on `service.runtime.orchestrator` they are `scene_fusion`,
`commit_policy`, `approach_path_policy`, `target_ordering`, `frame_resolver`, `viewpoint_planner`,
`gripper_model` and `corridor_config`. Each is `None` when it did not land.

## 8. Record logging

Off by default, which is why the soak, KPI and reinforcement-learning layers consume simulated and
synthetic data rather than production data. Three ways to ask for it: `robot.grasping.record_log_path`
in YAML, which only `from_robot_config` reads because `from_components` reads no config at all;
`record_log_path=` on `from_components`; or `service.enable_record_logging(path)` at runtime. The
shipped `robot.yaml` sets it to `null`, and the simulator runners take `--record-log <path>`.

**What lands.** One `json.dumps(..., sort_keys=True)` line per `pick()`, appended, with the parent
directory created and the whole write wrapped so a logging failure cannot break a pick. The record is
a `GraspAttemptRecord`: four mandatory fields (`timestamp`, `attempt_id`, `mode`, `final_outcome`),
per-stage summaries that are `None` when the stage did not run, and a free-form `extra` bag. The
frozen contract is in the [telemetry README](../../src/robot/grasping/telemetry/README.md); changing
it means updating the KPI, taxonomy and reinforcement-learning consumers together. Two things in
`extra` are worth knowing: `safety_rejected` is set from membership in a fixed outcome set, so a
metric can count safety refusals without string-matching, and `verification` falls back to an
explicitly labelled ground-truth-lift block when a simulator runner stamped a lift, so a simulated
lift can never be mistaken for a real verification.

**What consumes it.**

```powershell
python -m src.robot.grasping.replay --records logs\attempts.jsonl        # rollup, never fails
python -m src.robot.grasping.replay --records-gate logs\attempts.jsonl   # the real gate, can fail
python -m src.robot.grasping.replay --sim-soak-report logs\attempts.jsonl --sim-min-attempts 50
```

`--records-gate` compares your log against the committed baseline thresholds, which include a minimum
attempt count in the thousands. A bring-up logging tens of attempts is therefore permanently red on
the attempt floor alone: honest, and useless as a signal. Point `--thresholds` at your own bring-up
file, or use `--sim-min-attempts` with the simulation gate, and say which you did when you quote the
result. Do not fix a red gate by dropping the metric.

`--soak-report` is a **synthetic contract self-check, not a quality measure**, and its own `--help`
says so: the outcome distribution is authored, so a pass proves the telemetry, KPI, taxonomy, budget
and watchdog pipeline is internally consistent. It says nothing about whether the robot picks things
up, and it stays green through a completely broken cell. For a quality signal use `--records-gate`.
Details in the [replay README](../../src/robot/grasping/replay/README.md).

## 9. Troubleshooting

`pick()` does not raise on an ordinary "no grasp" outcome. It returns a typed report. Read
`report.outcome`, then `dict(report.telemetry)`, then the attempt trail on `report.pick_report`, whose
attempts each carry an index, an action and typed reasons.

**Nothing moved.**

| Outcome | Cause | Fix |
| --- | --- | --- |
| `mode_not_available`, reason `per_call_mode_requires_different_sampling_mode` | `pick(mode=...)` asked for a different sampler | rebuild with that mode, or stay inside the pairs in 6.2 |
| `mode_not_available`, phase gate on refinement or verification | the mode needs a refiner or verifier object | enable the block *and* check the object landed (7.3) |
| `no_target` | the frame carried no segmentations | perception, not grasping. Check the prompt, and that the camera is not returning black ([02](02-models.md)) |
| `missing_camera_frame` | a valid candidate existed, in the camera frame, and the policy refused | wire a `frame_resolver`. This is the fail-closed frame contract working ([03](03-calibration.md)) |
| `decision_fail_closed`, `uncertainty_fail_closed` | the decision gate refused | wire a viewpoint planner, or lower the thresholds knowingly |
| `no_commit_insufficient_fusion` | there was a candidate, not enough fused evidence, and the budget is gone | check `scene_fusion` is wired; the commit policy alone is inert |
| `drift_blocked_auto`, `ood_blocked_auto` | the watchdog locked autonomous operation | inspect the drift and out-of-distribution telemetry. Do not bypass |

**The arm refused.** A safety rejection surfaces as `execution_failed` with a motion message prefixed
`[safety:<guard>/<reason>]`. Guard order and verdicts: [04](04-robot-and-safety.md). Two things belong
here. `enforce: false` on a guard removes it at construction, so its surface is gone rather than
quiet. And blind IK proposes self-colliding branches that the guard then rejects, which reads exactly
like bad grasping: run the motion-stack probe before blaming the grasps.

**No candidates.** Read `attempt.reasons` and the calculator's telemetry line, which carries
`candidates_silhouette`, `candidates_geometry`, the `rejected_*` counters, `final` and `best_score`.
On a fresh bring-up check `no_candidates_generated` first. It means the generator returned nothing at
all, which a jaw too narrow can cause but so can a mask with no usable geometry, no antipodal pair or
no valid depth. If poses were generated and every one fell outside the jaw range, check
`robot.gripper.min_width_mm` and `max_width_mm` against the object and check the pixel-to-millimetre
scale. `ik_failed` is reachable only with an IK service wired (7.3). Per-stage detail: the
[generation README](../../src/robot/grasping/generation/README.md).

**It grabbed air.** `object_not_detected` is the honest path and what you want to see. `executed` with
the object still on the table means nothing checked: either the gripper does not implement
`ObjectDetectingGripper`, or it is a substituted `NullGripper` ([06](06-grippers.md)). To see the
debug overlay, note that `render_debug_images` is `False` by default, so
`service.last_debug_image_png` stays `None` until you call
`service.enable_debug_image_rendering(True)`. The simulator runners do it behind `--debug-frames DIR`,
but only the vision cells produce an image.

**A feature was turned on and nothing changed.** In order of likelihood: you are on `from_components`
and `effective_config` is `None`; the mode you ran cannot reach the block; you set `enabled: true` and
left the operative weight at zero; a runner-supplied constructor argument beat your config block; or
you enabled a consumer whose producer you did not wire. Section 7, then
[grasping-config-reference.md](../grasping-config-reference.md).

## 10. Next steps for real hardware

Nothing here has run against a physical controller. `from_robot_config` has live callers and
`--rehearse` drives the whole path on a dummy arm, but the UR driver has only met controller software,
the KUKA driver has never met a controller at all, and Franka and ROS 2 are empty registry slots that
raise on `create_arm`. In order:

1. **Run the preflight and clear the blocking items.** `real_cell --check` reports each with its fix.
   Against the shipped tree as a UR cell it finds three, and every one would otherwise surface at the
   bench as a different-looking failure. Details: [04](04-robot-and-safety.md), procedure:
   [real_cell_first_pick.md](../runbooks/real_cell_first_pick.md).
2. **Get a `PerceptionSource` for your camera.** A RealSense is already covered by
   `build_real_components`. Anything else is yours to write, and it must exist before anything else.
3. **Wire an IK service into the calculator.** Config cannot (7.3).
4. **Anchor the motion stack, and refuse to run degraded.** The fail-closed gate lives in the
   simulator bootstrap and does not cover a real UR cell, so call `probe_planning_environment`
   yourself and refuse. A real UR cell does not route through the planner until you say so.
5. **Turn on record logging from the first motion**, then gate with `--records-gate`, not
   `--soak-report`. Real records are the only thing that turns the soak, KPI and
   reinforcement-learning layers from a self-check into a measurement.
6. **Start in `easy`.** Its recovery allow-list is locked empty, so it never produces recovery motion.
   `closed_loop` needs both a refiner and a verifier wired and refuses the pick otherwise, so it is
   not where a bring-up starts.
7. **Re-measure after every change, and never transfer a planner margin between robot models.** A
   thinner-linked arm reads as permanently self-colliding at the larger arm's margin and finds no plan
   at all ([04](04-robot-and-safety.md)).

**Not in scope, and not reachable by wiring code:** certified functional safety. The planner's
collision awareness and the exact-mesh guard are simulation-grade risk *reduction*. A real cell still
needs the vendor's safety-rated stop, hardware emergency-stop circuits, and ISO 10218, ISO/TS 15066
and ISO 13849 compliance. A green simulation gate does not suggest otherwise.

## Where to look next

| For | Read |
| --- | --- |
| the service, both constructors, the cell builders | [autonomous_grasp README](../../src/robot/execution/autonomous_grasp/README.md) |
| the cell as one noun, and a campaign of picks | [execution README](../../src/robot/execution/README.md) |
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
