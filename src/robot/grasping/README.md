# Grasp pipeline (`src.robot.grasping`)

Everything between a segmentation mask plus a depth map and a logged grasp attempt: candidate
generation, scoring, the decision gate, approach planning, the motion choreography, the closed loop,
recovery, and the telemetry record.

The package depends on `src.geometry` and on the `RobotArm` / `Gripper` Protocols in
[`robot/core`](../core/README.md). It imports no vendor SDK and no web framework. Its consumers are
[`robot/execution`](../execution/README.md) (the composition root and the operator service), the
offline [`replay/`](replay/README.md) and [`rl/`](rl/README.md) tails, and the simulation runners in
`src/willy_sim`.

## What the default pick actually does

With a stock `robot.yaml` the attempt is open loop:

```
perceive -> generate and rank candidates (deterministic geometric score)
         -> safety preflight and IK -> approach, close, retreat -> log
```

That is the whole of it. Thirteen blocks under `robot.grasping` carry their own `enabled` switch and
every one of them ships `false`: `closed_loop`, `verification`, `dense_recovery`, `decision`,
`feasibility`, `ordering`, `recovery`, `uncertainty`, `success_model`, `performance`, `fusion`,
`approach_validation`, `deep_ranker`. So the decision gate, pre-grasp refinement, post-grasp
verification, scene recovery, multi-view fusion and the learned success model are present in this
directory and not on the path a fresh cell takes. The simulation runners switch several of them on
per command-line flag in runner code rather than in config, which is why a sim run and a
config-driven cell can behave differently.

Motion never bypasses the safety layer, and the reason is structural rather than a convention this
package keeps: `SafetyPreflight` is constructed inside the vendor arm driver, so every commanded
move is gated there whatever proposed it. Nothing in this package can relax a guard, and no scorer
here may reject: a scorer that could veto would put a heuristic above the safety layer.

## The tiers

Shared foundations, bottom first. None of them imports a tier above it.

| Tier | Role |
| --- | --- |
| [`types/`](types/README.md) | `GraspPoint`, `GraspResult`, `GraspFailureReason`, sampling modes, the perception Protocols. Imports no other grasping tier. |
| [`geometry/`](geometry/README.md) | Masked point cloud, surface normals, projection, `CameraIntrinsics`. |
| [`contacts/`](contacts/README.md) | Antipodal contact-pair extraction from mask plus depth. |
| [`collision/`](collision/README.md) | Gripper against cloud and table queries, parallel-jaw and suction envelopes. |

The pick path itself.

| Tier | Role |
| --- | --- |
| [`generation/`](generation/README.md) | `GraspCalculator`: mask plus depth to ranked `GraspPoint` candidates. |
| [`scoring/`](scoring/README.md) | The four deterministic axis scorers plus `rank_grasp_poses`, the force-closure certificate, and the learned success predictor. |
| `decision.py` | The `DecisionEngine`, a pure fail-closed gate returning `GRASP_NOW`, `MOVE_CAMERA`, `RECOVER` or `FAIL_CLOSED`. Off by default. |
| [`planning/`](planning/README.md) | 6-DoF approach planner (`GraspPose`), the IK service seam, multi-finger planners. |
| [`motion/`](motion/README.md) | `GraspExecutionPolicy` (approach, close, retreat), swept-path validation, and `frame_resolver`, which supplies the CAMERA to BASE transform and fails closed on an unresolved camera-frame grasp. |
| [`closed_loop/`](closed_loop/README.md) | Two-scan pose refinement, post-grasp verification, next-best-view. Off by default. |
| [`recovery/`](recovery/README.md) | Failure reasons to bounded recovery actions under anti-loop and budget limits. Off by default. |
| [`telemetry/`](telemetry/README.md) | The frozen `GraspAttemptRecord` and the per-stage latency clock. |
| [`loop/`](loop/README.md) | `BinPickingOrchestrator`, which wires the tiers into one attempt and selects the target. |

Second modalities and analysis.

| Tier | Role |
| --- | --- |
| [`multiview/`](multiview/README.md) | Fuse, localize and synthesize grasps across camera views, and the per-object multi-camera geometry fusion. |
| [`suction/`](suction/README.md) | The suction end effector: seal and wrench scoring, approach. |
| [`deep/`](deep/README.md) | The learned 6-DoF generator, selected by `robot.grasping.calculator: deep`. No trained weights ship in this repository. |
| `uncertainty.py` | Seven-channel uncertainty fusion. |
| [`visualization/`](visualization/README.md) | Grasp debug images and scene rendering. |

Offline tail, downward only. [`calibration/`](calibration/README.md), [`replay/`](replay/README.md)
and [`rl/`](rl/README.md) read what the pick path logged and are never imported by it at module top
level.

Two files sit at the package root and matter to a caller. `calculator_factory.build_calculator` is
the only reader of `robot.grasping.calculator`; it returns the analytic or the learned generator and
raises rather than falling back, so a cell that asked for the learned one can never quietly run the
other under its name. `scene.Scene` is the smallest useful entry point: a segmented BASE-frame cloud
and a support height in, ranked candidates out, with no robot and no cell.

## Usage

```python
import numpy as np
from src.robot.grasping import GraspCalculator

calc = GraspCalculator(min_grip_width_mm=5.0, max_grip_width_mm=85.0, camera_matrix=K)
cam_pts = calc.compute(seg, depth, unit="mm")                     # CAMERA frame
base_pts = calc.compute(seg, depth, camera_to_base=T, unit="mm")  # BASE frame
result = calc.compute_result(seg, depth)                          # adds reasons and telemetry
best = base_pts[0]  # sorted by score, descending
```

Ranked grasps without a camera or a calibration, straight from a cloud:

```python
from src.robot.grasping import Scene

scene = Scene.from_cloud(points_base_mm, support_height_mm=0.0)
grasps = scene.grasps()
print(grasps.generator, grasps.render())
```

A full pick is driven through `src.robot.execution.autonomous_grasp.AutonomousGraspService`, which
builds the orchestrator, the execution policy, the frame resolver and the arm together. Worked
examples live in `scripts/examples/`.

This package has no `python -m src.robot.grasping` entry of its own. The runnable command lines
belong to the subpackages:

```bash
python -m src.robot.grasping.replay --records run.jsonl        # roll up KPIs from a record log
python -m src.robot.grasping.replay --soak-report              # the soak gate, exit 0 iff it passes
python -m src.robot.grasping.rl check-dataset --records run.jsonl   # is this log trainable at all
python -m src.robot.grasping.rl build-dataset --dataset-id=<id>
python -m src.robot.grasping.deep --help                       # the learned generator: build, train, evaluate
python -m src.robot.grasping.calibration --replay run.jsonl --out calibration.json
```

### The three shipped presets

The presets are YAML overlays under [`config/grasping_presets/`](../../../config/grasping_presets/).
A preset is not part of the tree the loader validates: `apply_preset` in
[`replay/presets.py`](replay/presets.py) deep-merges one onto an already loaded `robot:` block, so a
misspelt key merges in silently and only `validate_preset` re-checks the merged result against the
schema. `default_mode` is the single field that flips sampling, refinement and verification
together.

| Preset | `default_mode` | What the overlay sets | Use for |
| --- | --- | --- | --- |
| `easy` | `easy` | recover off, uncertainty off | One object on a clean surface. The strictest gate thresholds apply here: `dead_loop_rate` at most 0.0 and `false_positive_grasp_rate` at most 0.005. |
| (base) | `auto` | nothing | Mixed scenes and custom tuning. |
| `dense_clutter` | `dense_clutter` | recover on, restricted to `next_viewpoint`; uncertainty on with a fail-closed threshold of 0.4 | Bins, piles, first-viewpoint occlusion. |
| `verification_heavy` | `closed_loop` | refine and verify demanded by the mode; recover requested | High-value items where a slip is unacceptable, at a higher cycle time. |

Two traps in that table. `closed_loop` demands a wired refiner and a verification policy: a service
that has neither refuses the pick with a `MODE_NOT_AVAILABLE` outcome rather than degrading to an
open-loop attempt. And `recovery.apply_modes` ships as `auto`, `dense_clutter` and
`dense_autonomous`, so the `recovery.enabled: true` that `verification_heavy` sets stays inert until
a cell adds `closed_loop` to that list. The same gate is why `easy` never produces recovery motion,
whatever the overlay says.

### KPI triage

`compute_kpis` in [`replay/kpi.py`](replay/kpi.py) defines every rate from the record log, and the
runbooks under `docs/runbooks/` triage a regression in one. The three an operator meets first:
`false_positive_grasp_rate` (reported successes that later failed verification),
`dead_loop_rate` (attempts that ended in `recovery_exhausted`) and `safety_rejection_rate`
(attempts a guard refused). Never disable a guard to move one of them.

## What has and has not run

- The soak gate is a synthetic self-check. `--soak-report` proves the telemetry, KPI, taxonomy, SLO
  and watchdog pipeline is internally consistent over generated records; the outcome distribution in
  those records is authored, so a pass says nothing about grasp quality. Use `--records-gate` on a
  real log for a signal that can fail. There is no hardware soak in this repository.
- Simulation is the only validated full-motion path. The runners under `src/willy_sim` build the
  service through `from_components` rather than `from_robot_config`, and re-enable the advanced
  blocks in runner code, so they are not evidence about a config-driven cell.
- `python -m src.robot.execution.real_cell` is the config-driven path, and `--rehearse` drives all of
  it on a dummy arm. Nothing below the rehearsal has run against a physical controller.
- The decision gate is hardware gated. `DecisionEngine` emits `FAIL_CLOSED` only when the arm reports
  it is not simulated and `fail_closed_on_real_hardware` is true, which is the default; a simulated
  rig downgrades to `GRASP_NOW` and records the reason it would have refused for. Simulation runs are
  permissive by construction.
- RL is offline only. Policies are trained, evaluated off-policy and promotion-gated offline, and run
  in shadow at runtime. There are no in-process weight updates and no live A/B. A cell that never
  names an RL mode imports no RL module at runtime.
- Multi-view fusion is shadow only unless a frame resolver and a commit policy are configured for the
  running mode. Multi-camera geometry fusion is a separate block and does reach the loop when
  enabled.
- Some switches carry a typed surface and no behaviour. `robot.grasping` refuses at config load any
  switch it knows to be unwired, because a cell that reports a capability it does not have is worse
  than one that will not start.
- Sensor limits are real. Transparent, reflective and specular objects produce depth dropouts and are
  not handled; read `depth_confidence` and rescan or refuse. Deformables have a refuse-safe seam
  only. The force-closure certificate is a Coulomb friction-cone argument under a contact model, not
  a wrench-space proof and no substitute for force feedback.

## See also

- [`robot/`](../README.md) for the parent package: the arm and gripper Protocols, drivers, safety,
  execution
- [`docs/grasping-math.md`](../../../docs/grasping-math.md) for every formula this stack evaluates and
  how each step fails
- [`docs/grasping-config-reference.md`](../../../docs/grasping-config-reference.md) for every
  `robot.grasping` block, which mode it can fire in, and the blocks that ship their operative weight
  at zero
- [`safety/`](../safety/README.md) for the guard pipeline every commanded move passes through
