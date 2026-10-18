# The grasp stack (`src/robot/grasping`)

Turns an object's points into ranked 6-DoF grasps, and holds the rest of a pick attempt around them:
the approach, recovery and the record each attempt leaves. It moves
nothing by itself: the UR and Isaac arm drivers gate every commanded move through their own
safety preflight, and nothing here can relax a guard.

```python
from willy import load_tree
from src.robot.grasping.scene import Scene

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
tree and plans for the library's default jaw. A camera supplies the cloud in
[15_speak_pick_and_hand_handover.py](../../../examples/real_robot/15_speak_pick_and_hand_handover.py).

**Which way the jaws close.** `Scene.from_robot_config` reads `robot.motion.natural_closing_axis`: `grasps()`
asked for no axis turns every grasp, of its two wrist turns half a turn apart, to the one whose tool +X
lies nearer that direction, and drops none. `grasps(closing_axis="-y")` takes only the grasps whose
closing axis heads within 30 degrees of the axis, each turned the named way round, chosen before
`max_candidates` (the generator is asked for 36, or `max_candidates` where that is more);
`SceneGrasps.withheld` says what the axis left out, and a result it left empty says so instead of "too
sparse". The scene of a looked-around part (`located.scene(0, ...)`) takes the axis its looks judged and
refuses any other ([perception/](../perception/README.md)).

**Why there is none, and the next one.** An empty `SceneGrasps` says why in `said` where the scene can tell: a
neighbour the camera saw, the support, a declared body, a part too short for the hand; `refused_by_obstacles` and
`refused_by_support` count what was refused for it. `grasps().other_than(refused)` leaves out every grasp within
10 mm and 15 degrees of one already refused, so a program that looks again after a refused pick tries another
grasp, not the same one seen again (example 13).

## Usage

| You have | Call | Shown in |
| --- | --- | --- |
| an object's cloud in the base frame | `Scene.from_robot_config(tree.robot, cloud).grasps()` | the block above |
| a mask, a depth image, a camera matrix | `build_calculator(tree.robot, data_dir=tree.root, camera_matrix=K)` | [generation/](generation/README.md) |
| a cell and a prompt | `PickRun.from_cell(Cell.from_tree(tree, prompt=...), ...)` | [12_pick_with_the_camera.py](../../../examples/real_robot/12_pick_with_the_camera.py) |

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
and `preflight_calculator` checks the choice without building anything. `Scene` always runs the
analytic generator and names it in `SceneGrasps.generator`.

`deep` builds for a cell only from weights whose proof has passed: the promotion record beside them
says so at phase `active` ([deep/promotion.py](deep/promotion.py), the owner's decision of 2026-10-09).
No finished models ship and every customer trains their own, the pipeline is not yet proven to
generalise, and nothing writes a promotion yet (`deep judge`, coming), so today every trained artifact
refuses a cell. `build_calculator(..., purpose="evaluate")` is how the ladder and the simulation
runners build one anyway, to measure it; it skips that check and no other.

The command lines belong to the subpackages; `src.robot.grasping` itself has no `python -m` entry.

```bash
python -m src.robot.grasping.replay --records picks.jsonl             # the KPIs of a record log
python -m src.robot.grasping.replay --records-gate picks.jsonl        # a gate that can fail
python -m src.robot.grasping.deep --help                              # the learned generator
```

### What the default pick does

A stock tree runs an open-loop attempt: perceive, generate and rank candidates, safety preflight and
IK, approach, close and retreat, log. The one block under `robot.grasping` with its own `enabled` switch,
`deep_ranker`, ships `false`; following a task's parts has no switch since 2026-10-17 (every task on a cell
built from a tree follows them), nor have the grasping switches that left with it, each fixed at the owner's
cell's value. The recovery every cell runs has no switch (fixed since 2026-10-16), and neither has the fusion of
fixed cameras: a cell fuses its calibrated RGB-D rigs where it
declares two or more. A wrist camera's looks need no switch either: a pick
handed several looks fuses them and stops at the first safe grasp ([loop/](loop/README.md)). The hold
is checked after every close by the execution policy, where the gripper can report one
([motion/](motion/README.md)): it reads the gripper's `is_object_detected` and `hold_evidence` (a
Robotiq's gOBJ). The separate verification stage with its `verification` block, and the
`dense_recovery` block, were removed on 2026-09-29: no pick consulted them. The decision gate and the
`feasibility`, `occlusion`, `ordering`, `uncertainty`, `success_model`, `watchdog`, `performance` and
`approach_validation` blocks were removed on 2026-10-16 with their code: switched on one at a time,
none changed a pick of the owner's cell's replay bank (479 picks). A tree that still writes one of
them is refused at load, and [the config reference](../../../docs/grasping-config-reference.md)
explains each block that stays.

### The three grasp modes

`default_mode` is the one field that switches the sampling and the recovery a mode allows together.

| `default_mode` | Samples | Recovery the mode allows | Use for |
| --- | --- | --- | --- |
| `easy` | the single-object sampler | none, whatever the tree says | one object on a clean surface |
| `auto` | the dense sampler where other parts stand beside the part or its mask is far from convex, else the single-object one | `rescan` | mixed scenes and your own tuning |
| `dense_clutter` | the dense point-cloud sampler | `rescan`, `nudge_target` | bins, piles, occlusion |

The grasp presets that overlaid a mode with its switches (`config/grasping_presets/`) left on 2026-10-16.
`verification_heavy`, the preset that set `closed_loop` to refine and verify, was deleted on 2026-09-29
with that mode and with `dense_autonomous`: a tree or a call that still names either is refused with
the mode to name instead (`auto` for `closed_loop`, `dense_clutter` for `dense_autonomous`).
The built-in profiles set what recovery may ever do, and every cell recovers as the owner's cell did
(fixed since 2026-10-16, no block to write). `nudge_target`, the one push a built-in profile allows, runs
on every `dense_clutter` cell whose parts are not critical and needs **no fixture**. It runs inside a wrist
camera's pick attempt, planned from what the looks saw, with nobody asked ([recovery/](recovery/README.md)).
A fixed-camera cell never pushes. `next_viewpoint`, which `dense_clutter` allowed until then, was merged
into `rescan` on 2026-09-29, and `next_target` left on 2026-10-16; a run that still names either is refused
with `removed on purpose: use rescan`.

The records gate holds `dead_loop_rate` at most 0.005 over at least 2,000 attempts of a real log. `easy`
stays free of recovery motion twice over: its profile allows none, and the cell's recovery applies in
`auto` and `dense_clutter` alone.

### KPI triage

`compute_kpis` in [replay/kpi.py](replay/kpi.py) defines every rate from the record log. The three an
operator meets first: `safety_rejection_rate` (attempts a guard refused, a push that stopped among
them), `dead_loop_rate` (attempts that ended in `recovery_exhausted`: a recovery loop ran an action and
had none left) and `false_positive_grasp_rate` (reported successes that later
failed a re-check). Nothing on this stack writes the field the last one counts, so `--records` withholds
it rather than printing 0.0. The runbooks in `docs/runbooks/` bring a cell up and take it to its first
pick; none of them triages a KPI, so start from the rate's definition in `kpi.py`. Never disable a guard
to move a rate.

## What it refuses

| Refusal | When | What to do |
| --- | --- | --- |
| `FileNotFoundError` from `build_calculator` | `calculator: deep` and no file at `deep_generator.artifact_path` | train one ([deep/](deep/README.md)) or set `geometric` |
| `ValueError` from `build_calculator` | not a generator artifact of this version, or the cell's hand is unset or not trained on | name a finished run's artifact and the cell's hand |
| `ValueError` from `build_calculator` for a cell | `calculator: deep` and an artifact whose proof has not passed, which today is every one | set `geometric`, or evaluate it ([deep/](deep/README.md#what-it-refuses)) |
| `ConfigError` at load, `removed on purpose` | a removed mode (`closed_loop`, `dense_autonomous`), recovery action (`next_viewpoint`) or block (`closed_loop`, `verification`, `dense_recovery`, and since 2026-10-16 `decision`, `feasibility`, `occlusion`, `ordering`, `uncertainty`, `success_model`, `watchdog`, `performance`, `approach_validation` and `robot.rl`, and since 2026-10-17 the twelve grasping switches fixed at the owner's cell's values and `gripper_geometry`) | write what the sentence names; `python -m src.config migrate PATH --write` rewrites an old tree |
| outcome `MODE_NOT_AVAILABLE` | a `mode=` per pick whose sampler the service was not built with | build the service in that mode |
| the execution policy fails closed | a camera-frame grasp and no CAMERA to BASE transform | calibrate the camera ([guide 03](../../../docs/guide/03-calibration.md)) |

## Status

| Capability | Evidence |
| --- | --- |
| Analytic generation, scoring and the pick service on a UR5e with a 2F-85 | measured in simulation ([willy_sim](../../willy_sim/README.md)) |
| The recovery loop and fixed-camera fusion | measured in simulation |
| A wrist camera's looks and the push | pinned by tests on fake arms and cameras, and the Isaac arm generates no view; the push ran on URSim CB3 with a recorded look as the camera (2026-10-02); never touched hardware |
| The hold check after the close | pinned by tests on the Robotiq driver's gOBJ seam; never touched hardware |
| A grasp from this package on a physical arm | run on a physical cell: camera picks on a UR10 (CB3) with a wrist D415 and a `jaw_io` Hand-E; no pick rate is kept here |

The simulation runners build the pick service through `from_components` and switch the advanced blocks
on in runner code, so a simulation result says nothing about a config-built cell with those blocks
off. A desk rehearsal on a dummy arm proves the config-built wiring, not a grasp. The records gate
judges a real log of picks and can fail ([replay/](replay/README.md)). Transparent,
reflective and specular objects drop out of depth and are not handled; read `depth_confidence` and
rescan or refuse. Deformables have an inert seam only ([generation/](generation/README.md)). The friction
cone the generator holds each contact to is an argument under a contact model, not a substitute for force
feedback.

## Files

| Folder or file | Holds |
| --- | --- |
| [types/](types/README.md) | `GraspPoint`, `GraspResult`, `GraspFailureReason`, the sampling modes, the perception value objects |
| [geometry/](geometry/README.md), [contacts/](contacts/README.md) | masked point cloud, normals, projection; antipodal contact pairs |
| [collision/](collision/README.md) | the gripper against the cloud and the table; jaw and suction-cup envelopes |
| [generation/](generation/README.md) | `GraspCalculator`: a mask and a depth image to ranked `GraspPoint`s |
| [scoring/](scoring/README.md) | the score record a candidate carries, the workspace box, the occlusion and outline signals |
| [planning/](planning/README.md), [motion/](motion/README.md) | the grasp pose, the IK seam; approach, close and retreat, and the CAMERA to BASE resolver |
| [recovery/](recovery/README.md) | bounded recovery: rescan, skip a failed part, the push, on every cell. `closed_loop/`, which held the post-grasp verifiers no pick path ran, left on 2026-09-29 |
| [loop/](loop/README.md), [telemetry/](telemetry/README.md) | `BinPickingOrchestrator`, one attempt across the tiers and a wrist camera's looks; the frozen `GraspAttemptRecord` |
| [multiview/](multiview/README.md), [suction/](suction/README.md) | fusion across fixed cameras and a wrist camera's looks, and which jaw contact faces were seen; suction candidates |
| [deep/](deep/README.md) | the learned 6-DoF generator; no trained weights ship, and a cell grasps only with weights whose proof passed |
| [visualization/](visualization/README.md) | grasp debug images |
| [replay/](replay/README.md) | the offline tail: reads what the pick path logged, never imported by it at module top level |
| `scene.py`, `calculator_factory.py` | `Scene`; `build_calculator` and `preflight_calculator`; `build_calculator` hands every analytic calculator the cell's rim (`footprint_rim.FOOTPRINT_RIM_MM`, as `support_footprint_rim_mm`), which `Scene` does not read |

## Details

- [Guide 05](../../../docs/guide/05-pick-loop.md): the pick loop, from the config tree to an attempt
- [The grasping maths](../../../docs/grasping-math.md): every formula this stack evaluates, and how it fails
- [The config reference](../../../docs/grasping-config-reference.md): every block, and the modes it fires in
- [safety/](../safety/README.md) guards every commanded move; [robot/core](../core/README.md) holds the
  `RobotArm` and `Gripper` Protocols this package depends on. It imports no vendor SDK and no web
  framework.
- [tests/repo/test_t8_docs.py](../../../tests/repo/test_t8_docs.py) pins the mode table and the KPI names on this page
