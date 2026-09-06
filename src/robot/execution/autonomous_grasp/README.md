# Autonomous grasp service

The composition root of the stack and the operator-facing grasp facade: pick a `GraspMode`, run
the fail-closed autonomy layers, get back one typed report.

## What it guarantees

`AutonomousGraspService` wraps a single-attempt `RuntimePickService` and adds the operator
concerns: mode selection with a locked per-mode behaviour profile, the fail-closed decision gate,
uncertainty fusion, the drift and out-of-distribution watchdog, latency telemetry, the opt-in
reinforcement-learning shadow router, and the bounded recovery loop. Every `pick()` returns a
frozen `AutonomousGraspReport`.

A mode whose machinery is not wired refuses with `MODE_NOT_AVAILABLE` rather than silently
downgrading to `auto`: `closed_loop` without a refiner, `dense_autonomous` without a verifier.
Safety and the reinforcement-learning action mask outrank any model, by construction.

`__init__.py` and `service.py` re-export the public names, so import paths stay stable wherever a
module moves inside the package.

## What actually runs by default

Much less than this page lists. Every advanced `robot.grasping` block ships `enabled: false`
(`decision`, `closed_loop`, `verification`, `uncertainty`, `feasibility`, `ordering`, `recovery`,
`dense_recovery`, `fusion`, `success_model`, `deep_ranker`, `approach_validation`, `performance`),
and `robot.rl.mode` ships `hybrid_ml`, which builds no shadow router. The default attempt is
open-loop: perceive, generate and score candidates, safety preflight and IK, motion and gripper
close.

## Contents

| File | Role |
| --- | --- |
| `service.py` | `AutonomousGraspService`: `from_components`, `from_robot_config`, `pick()`, and the pick path, split into the open-loop attempt, the decision loop, the refine and verify pipeline, and the watchdog pre-tick. |
| `config.py` | `GraspMode` (`easy`, `auto`, `dense_clutter`, `closed_loop`, `dense_autonomous`), `resolve_grasp_mode`, `GraspBehaviorProfile` and the locked profiles, and `EffectiveGraspingConfig` with its eight nested per-phase sub-configs. |
| `report.py` | `AutonomousGraspOutcome` and `AutonomousGraspReport`, which composes `PickSessionReport` and never widens it. |
| `cells.py` | `build_real_cell` and `build_rehearsal_cell`, a whole cell in one call, plus `build_real_components` and `build_rehearsal_components` for a caller that wants the pieces. |
| `rehearsal.py` | The synthetic one-box scene `build_rehearsal_cell` uses: no model, no camera, no noise, one flat box at a known place, so that what is under test is the wiring. |
| `builders.py` | The per-phase `from_robot_config` wiring helpers, as pure functions or one scoped orchestrator mutation. |
| `watchdog.py` | `WatchdogCoordinator`: drift and out-of-distribution policy build, evaluation, and rising-edge event emission. Stateless; the rolling sample history lives on the service. |
| `latency.py` | `LatencyTelemetryCoordinator`: rolling p95 and breach emission. It never fails closed. |
| `shadow.py` | Shadow-router wiring and post-pick annotation. Fail-safe by contract: any missing field, artifact or load error degrades to no shadow, and routing has no influence on the executed grasp. |
| `record_logging.py` | The opt-in serializer from `AutonomousGraspReport` to `GraspAttemptRecord`, the data source the soak, KPI and offline learning layers consume. |
| `action_mask_eval.py` | Per-candidate action-mask evaluator. It re-runs the real deterministic checks, the safety preflight and IK, per shadow candidate, so the mask reflects only what the stack actually decided. |

## Usage

```python
from src.robot.execution.autonomous_grasp import AutonomousGraspService, GraspMode

# The config-driven path. `calculator` and `perception` are required and have no default:
# config cannot describe a camera adapter that needs a per-run prompt and two models on a GPU.
service = AutonomousGraspService.from_robot_config(robot_cfg, calculator=calc, perception=source)

report = service.pick(mode=GraspMode.AUTO)   # per-call override; default_mode otherwise
print(report.outcome)

service.enable_record_logging("logs/attempts.jsonl")   # opt-in, off by default
```

`build_real_cell(robot_cfg, prompt=...)` supplies both required arguments and hands the result to
the root, so a cell built from configuration takes one call. It is the convenience beside the root,
not a second way to compose: the root keeps its explicit contract, because optional arguments there
would let a forgotten argument open a camera.

## Traps

- **`mode=` reinterprets the behaviour profile only.** It never rebuilds the wrapped
  `RuntimePickService`, so it cannot change which low-level sampling mode ran. A mode whose sampler
  differs from the one that was built is refused with `MODE_NOT_AVAILABLE` rather than lying about
  which sampler executed.
- **`from_components` leaves `effective_config=None`,** which silences uncertainty fusion, the
  watchdog, breach emission and the recovery loop, because those coordinators no-op while it is
  `None`. A caller on that path re-enables them explicitly through `build_effective_config` and
  `apply_orchestrator_overlays`. `from_robot_config` accepts live `arm` and `gripper` handles for
  exactly this reason: a simulator cell that hands over its session-sharing gripper keeps the
  config-driven path and every overlay with it.
- **Record logging is opt-in and off.** `from_robot_config` reads `grasping.record_log_path`, which
  supports `${WILLY_RECORD_LOG:-}` substitution and treats an empty expansion as off;
  `from_components` takes a `record_log_path=` argument. Every `pick()` carries a unique
  `attempt_id`, `pick-<uuid>` on the open-loop path and `auto-<uuid>` through the decision loop, so
  logged records never collide. Unset means no records and a byte-identical run.
- **A cell about to collect records a ranker cannot train on says so at boot.** When
  `record_log_path` is set and `robot.rl.mode` is not `rl_shadow`, the build warns once, because the
  per-candidate feature rows a pairwise ranker needs are written only when the ranking shadow ran.
  It is a warning and never a refusal: those records still feed the KPI roll-up, the failure
  taxonomy and every post-mortem a bring-up depends on.
- **`EffectiveGraspingConfig.to_dict()` is a flat telemetry contract of 82 keys.** The eight nested
  sub-configs are flattened back to per-field names, and each key records the effective state rather
  than the file's: a block outside its `apply_modes` reads false here even where the YAML says true,
  because the question a record answers is whether the block was acting on this attempt.

## See also

- [execution](../README.md), the parent package and its public surface
- [real_cell](../real_cell/README.md), the command-line surface over this service
- [grasping](../../grasping/README.md), the tier this service drives
- [safety](../../safety/README.md), the fail-closed preflight every motion passes through
- [grasping/rl](../../grasping/rl/README.md), the offline layer the shadow router feeds
