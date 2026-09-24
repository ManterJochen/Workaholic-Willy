# The grasp stack (`src/robot/grasping`)

Turns an object's points into ranked 6-DoF grasps, and holds the rest of a pick attempt around them:
the decision gate, the approach, a second look, recovery and the record each attempt leaves. It moves
nothing by itself: the UR, KUKA and Isaac arm drivers gate every commanded move through their own
safety preflight, and nothing here can relax a guard.

```python
from willy import Scene, load_tree

tree = load_tree()                      # the cell WILLY_PROFILE names
# cloud_base_mm: the object's points, an (N, 3) array in the robot's base frame, millimetres
grasps = Scene.from_robot_config(tree.robot, cloud_base_mm).grasps()
print(grasps)                           # best first, and which generator made them
best = grasps.best                      # None when no grasp is legal
if best is not None:
    print(robot.pick(best.pose(), best.grip_width_mm))   # robot: a connected Robot
```

`best.pose()` is what `Robot.pick` takes: base frame, +Z the approach, +X the closing axis. The jaw and
the support height come from the tree. `Scene.from_cloud(cloud_base_mm, support_height_mm=0.0)` needs no
tree and plans for the library's default jaw. Run it at a desk with
[grasps_for_a_cloud.py](../../../examples/offline/grasping/grasps_for_a_cloud.py); a camera supplies the
cloud in [13_speak_pick_and_hand_handover.py](../../../examples/real_robot/13_speak_pick_and_hand_handover.py).

## Usage

| You have | Call | Shown in |
| --- | --- | --- |
| an object's cloud in the base frame | `Scene.from_robot_config(tree.robot, cloud).grasps()` | [grasps_for_a_cloud.py](../../../examples/offline/grasping/grasps_for_a_cloud.py) |
| a mask, a depth image, a camera matrix | `build_calculator(tree.robot, data_dir=tree.root, camera_matrix=K)` | [generation/](generation/README.md) |
| a cell and a prompt | `PickRun.from_cell(Cell.from_tree(tree, prompt=...), ...)` | [11_pick_with_the_camera.py](../../../examples/real_robot/11_pick_with_the_camera.py) |

A whole pick, from the camera to the record of the attempt:

```python
from willy import Cell, PickRun, Recording, load_tree

cell = Cell.from_tree(load_tree(), prompt="a red cube")
report = PickRun.from_cell(cell, runs=1, recording=Recording.to_file("picks.jsonl")).execute()
print(report)
```

> [!WARNING]
> `execute()` connects the cell and moves the arm. Rehearse it on a dummy arm first
> ([01_rehearse_a_pick.py](../../../examples/simulation/01_rehearse_a_pick.py)), and at a real cell
> follow [docs/runbooks/real_cell_first_pick.md](../../../docs/runbooks/real_cell_first_pick.md).

`robot.grasping.calculator` chooses the generator: `geometric`, the default, or `deep`, the learned one
in [deep/](deep/README.md). Every cell builds through `build_calculator`, the only reader of that key,
and `preflight_calculator` checks the choice without building anything
([select_grasp_generator.py](../../../examples/offline/grasping/select_grasp_generator.py)). `Scene`
always runs the analytic generator and names it in `SceneGrasps.generator`.

The command lines belong to the subpackages; `src.robot.grasping` itself has no `python -m` entry.

```bash
python -m src.robot.grasping.replay --records picks.jsonl             # the KPIs of a record log
python -m src.robot.grasping.replay --records-gate picks.jsonl        # a gate that can fail
python -m src.robot.grasping.rl check-dataset --records picks.jsonl   # is this log trainable
python -m src.robot.grasping.deep --help                              # the learned generator
python -m src.robot.grasping.calibration --replay picks.jsonl --out calibration.json   # uncertainty calibration
```

### What the default pick does

A stock tree runs an open-loop attempt: perceive, generate and rank candidates, safety preflight and
IK, approach, close and retreat, log. Every block under `robot.grasping` with its own `enabled` switch
ships `false`: `closed_loop`, `verification`, `dense_recovery`, `decision`, `feasibility`, `ordering`,
`recovery`, `uncertainty`, `success_model`, `performance`, `fusion`, `approach_validation` and
`deep_ranker`. So the decision gate, refinement, verification, recovery, multi-view fusion and the
learned success model are here, and off the path a fresh cell takes. `python scripts/checks/grasping_switches.py` prints which block is
reachable in which grasp mode, and [the config reference](../../../docs/grasping-config-reference.md)
explains each block.

### The shipped presets

A preset is a YAML overlay in [`config/grasping_presets/`](../../../config/grasping_presets/).
`apply_preset` in [replay/presets.py](replay/presets.py) merges one onto a loaded `robot:` block. The
loader does not validate presets, so a misspelt key merges in silently; `validate_preset` re-checks the
merged result against the schema. `default_mode` is the one field that switches sampling, refinement
and verification together.

| Preset | `default_mode` | What the overlay sets | Use for |
| --- | --- | --- | --- |
| `easy` | `easy` | recover off, uncertainty off | one object on a clean surface, under the strictest gate |
| (base) | `auto` | nothing | mixed scenes and your own tuning |
| `dense_clutter` | `dense_clutter` | recover on, `next_viewpoint` only; uncertainty on, fail-closed at 0.4 | bins, piles, occlusion |
| `verification_heavy` | `closed_loop` | refine and verify, which the mode demands; recover on, `next_viewpoint` only | parts where a slip costs most |

The gate for `easy` is the strictest: `dead_loop_rate` at 0.0 and `false_positive_grasp_rate` at most
0.005. `verification_heavy` costs cycle time. Two traps. `closed_loop` demands a refiner and a
verification policy: a cell without them refuses the pick with a `MODE_NOT_AVAILABLE` outcome rather
than running open loop. And `recovery.apply_modes` ships as `auto`, `dense_clutter` and
`dense_autonomous`, so the recovery `verification_heavy` switches on stays inert until the cell adds
`closed_loop` to that list. The same gate keeps `easy` free of recovery motion whatever an overlay
says.

### KPI triage

`compute_kpis` in [replay/kpi.py](replay/kpi.py) defines every rate from the record log. The three an
operator meets first: `safety_rejection_rate` (attempts a guard refused), `dead_loop_rate` (attempts
that ended in `recovery_exhausted`) and `false_positive_grasp_rate` (reported successes that later
failed a re-check). Nothing on this stack writes the field the last one counts, so `--records` withholds
it rather than printing 0.0. The runbooks in `docs/runbooks/` bring a cell up and take it to its first
pick; none of them triages a KPI, so start from the rate's definition in `kpi.py`. Never disable a guard
to move a rate.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `FileNotFoundError` from `build_calculator` | `calculator: deep` and no file at `deep_generator.artifact_path` | train one ([deep/](deep/README.md)) or set `geometric` |
| `ValueError` from `build_calculator` | not a generator artifact of this version, or the cell's hand is unset or not trained on | name a finished run's artifact and the cell's hand |
| `ConfigError` at load | a switch that reaches nothing is on: `occlusion.hard_reject_enabled` | set it back to `false` |
| outcome `MODE_NOT_AVAILABLE` | the mode demands refinement or verification the cell did not build | enable `closed_loop` and `verification`, or change mode |
| `FAIL_CLOSED` from the decision gate | the gate is on, the arm is real and the evidence falls short | read the reason on the record; a simulated arm logs it and goes on |
| the execution policy fails closed | a camera-frame grasp and no CAMERA to BASE transform | calibrate the camera ([guide 03](../../../docs/guide/03-calibration.md)) |

## Status

| Capability | Evidence |
| --- | --- |
| Analytic generation, scoring and the pick service on a UR5e with a 2F-85 | measured in simulation ([willy_sim](../../willy_sim/README.md)) |
| Refinement, verification, recovery and multi-view fusion | measured in simulation, switched on per flag by the runners |
| A grasp from this package on a physical arm | never touched hardware |

The simulation runners build the pick service through `from_components` and switch the advanced blocks
on in runner code, so a simulation result says nothing about a config-built cell with those blocks
off. A desk rehearsal on a dummy arm proves the config-built wiring, not a grasp. The soak gate is a
synthetic self-check of the telemetry and KPI pipeline, not of grasp quality
([replay/](replay/README.md)). RL runs offline and in shadow only ([rl/](rl/README.md)). Transparent,
reflective and specular objects drop out of depth and are not handled; read `depth_confidence` and
rescan or refuse. Deformables have a refuse-safe seam only. The force-closure certificate is a
friction-cone argument under a contact model, not a substitute for force feedback.

## Files

| Folder or file | Holds |
| --- | --- |
| [types/](types/README.md) | `GraspPoint`, `GraspResult`, `GraspFailureReason`, the sampling modes, the perception value objects |
| [geometry/](geometry/README.md), [contacts/](contacts/README.md) | masked point cloud, normals, projection; antipodal contact pairs |
| [collision/](collision/README.md) | the gripper against the cloud and the table; jaw and suction-cup envelopes |
| [generation/](generation/README.md) | `GraspCalculator`: a mask and a depth image to ranked `GraspPoint`s |
| [scoring/](scoring/README.md) | the deterministic scorers, `rank_grasp_poses`, force closure, the learned success predictor |
| `decision.py` | `DecisionEngine`: `GRASP_NOW`, `MOVE_CAMERA`, `RECOVER` or `FAIL_CLOSED`; off by default |
| [planning/](planning/README.md), [motion/](motion/README.md) | the approach pose, the IK seam; approach, close and retreat, and the CAMERA to BASE resolver |
| [closed_loop/](closed_loop/README.md), [recovery/](recovery/README.md) | the second look, verification, next-best view; bounded recovery. Off by default |
| [loop/](loop/README.md), [telemetry/](telemetry/README.md) | `BinPickingOrchestrator`, one attempt across the tiers; the frozen `GraspAttemptRecord` |
| [multiview/](multiview/README.md), [suction/](suction/README.md) | fusion across camera views; suction candidates |
| [deep/](deep/README.md) | the learned 6-DoF generator; no trained weights ship |
| `uncertainty.py`, [visualization/](visualization/README.md) | fusion of the uncertainty channels; grasp debug images |
| [calibration/](calibration/README.md), [replay/](replay/README.md), [rl/](rl/README.md) | the offline tail: reads what the pick path logged, never imported by it at module top level |
| `scene.py`, `calculator_factory.py` | `Scene`; `build_calculator` and `preflight_calculator` |

## Details

- [Guide 05](../../../docs/guide/05-pick-loop.md): the pick loop, from the config tree to an attempt
- [The grasping maths](../../../docs/grasping-math.md): every formula this stack evaluates, and how it fails
- [The config reference](../../../docs/grasping-config-reference.md): every block, and the modes it fires in
- [safety/](../safety/README.md) guards every commanded move; [robot/core](../core/README.md) holds the
  `RobotArm` and `Gripper` Protocols this package depends on. It imports no vendor SDK and no web
  framework.
- [tests/test_t8_docs.py](../../../tests/test_t8_docs.py) pins the presets and the KPI names on this page
